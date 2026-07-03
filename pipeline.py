"""
长文本理解管线 — 滑动窗口 + 并行分段总结 + 汇总回答 + 三元组抽取

流程:
  1. 滑动窗口切分文本 (sentence-aware)
  2. 并行总结所有窗口 (ThreadPoolExecutor, 按 index 排序拼合)
  3. 后处理三元组: 实体对齐(LLM) → 质量过滤 → 去重(合并evidence)
  4. 汇总所有总结, 基于总结回答问题 / 自动生成 QA 对

外部依赖文件（同学主要修改这些）:
  config.py  — API/模型/窗口/并发参数
  prompts.py — 所有提示词 & 功能开关（含 ENABLE_QUESTION_INPUT / ENABLE_TRIPLE_INPUT）
"""

import os, re, time, json, threading, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from config import (
    API_BASE, API_KEY, MODEL, DATA_DIR,
    WINDOW_SIZE, OVERLAP, STRIDE,
    WORKERS, MAX_TOKENS_SUMMARIZE, MAX_TOKENS_ANSWER, MAX_RETRIES,
)
from prompts import (
    ENABLE_QUESTION_INPUT,
    ENABLE_TRIPLE_INPUT,
    TRIPLE_INSTRUCTION,
    SUMMARIZE_SYSTEM_PROMPT,
    ENTITY_CANON_SYSTEM_PROMPT,
    ENTITY_CANON_USER_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    QA_GENERATION_PROMPT,
)

client = OpenAI(base_url=API_BASE, api_key=API_KEY, timeout=120.0)


# ============ 滑动窗口 ============

def sliding_window_chunks(text: str, window_size: int = WINDOW_SIZE,
                          stride: int = STRIDE) -> list[dict]:
    if not text:
        return []
    chunks = []
    start = 0
    total = len(text)
    while start < total:
        end = min(start + window_size, total)
        if end < total:
            search_from = max(start, end - 200)
            last_break = -1
            for m in re.finditer(r'[.!?]', text[search_from:end]):
                last_break = search_from + m.end()
            if last_break > start:
                end = last_break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"start": start, "end": end, "text": chunk})
        start += stride
    return chunks


# ============ 模型调用 ============

def _call_model(messages: list, max_tokens: int) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(f"API call failed: {e}")


def _parse_response(text: str) -> tuple[str, list[tuple[str, str, str]]]:
    """从模型回复中解析 [SUMMARY]...[/SUMMARY] 和 [TRIPLES]...[/TRIPLES] 块
    返回 (summary, [(subject, predicate, object), ...])"""
    summary = ""
    triples: list[tuple[str, str, str]] = []

    sm = re.search(r'\[SUMMARY\]\s*(.*?)\s*\[/SUMMARY\]', text, re.DOTALL | re.IGNORECASE)
    if sm:
        summary = sm.group(1).strip()

    tm = re.search(r'\[TRIPLES\]\s*(.*?)\s*\[/TRIPLES\]', text, re.DOTALL | re.IGNORECASE)
    if tm:
        for line in tm.group(1).strip().split('\n'):
            line = line.strip().lstrip('- ').strip()
            if not line:
                continue
            parts = re.split(r'\|\|', line)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))

    if not summary:
        summary = text.strip()

    return summary, triples


# ============ 窗口总结 ============

def summarize_one(window_text: str, book_name: str,
                  idx: int, total: int) -> tuple[str, list[tuple[str, str, str]]]:
    """总结单个窗口并同步提取三元组（线程安全，可被并行调用）
    返回 (summary, [(subject, predicate, object), ...])"""
    system_prompt = SUMMARIZE_SYSTEM_PROMPT.format(
        triple_instruction=TRIPLE_INSTRUCTION,
    )
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"[{book_name} | {idx}/{total}]\n{window_text}"},
    ]
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = _call_model(msgs, MAX_TOKENS_SUMMARIZE)
            summary, triples = _parse_response(r)
            if summary:
                return summary, triples
            if attempt < MAX_RETRIES:
                r = _call_model(msgs, MAX_TOKENS_SUMMARIZE * 2)
                summary, triples = _parse_response(r)
                if summary:
                    return summary, triples
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"  [Warn] summarize_one failed after {MAX_RETRIES + 1} attempts: {e}")
    return "[summary unavailable]", []


def summarize_all_parallel(tasks: list[dict], workers: int = WORKERS,
                           resume_path: str | None = None) -> tuple[list[tuple[int, str]], list[dict]]:
    """
    tasks: [{global_idx, book_name, text, section_idx, section_total}]
    返回 ([(global_idx, summary)], [triple_dicts])，summary 已按 global_idx 排序。
    triple_dict = {subject, predicate, object, book, window}
    """
    total = len(tasks)
    done: set[int] = set()
    results: dict[int, str] = {}
    all_triples: list[dict] = []

    if resume_path and os.path.exists(resume_path):
        saved = json.load(open(resume_path, "r", encoding="utf-8"))
        done = set(saved.get("done", []))
        results = {int(k): v for k, v in saved.get("results", {}).items()}
        print(f"  Resumed: {len(done)}/{total} windows already done.")

    pending = [t for t in tasks if t["global_idx"] not in done]
    lock = threading.Lock()

    def process(t: dict) -> tuple[int, str, list[tuple[str, str, str]]]:
        idx = t["global_idx"]
        summary, triples = summarize_one(t["text"], t["book_name"],
                                         t["section_idx"], t["section_total"])
        with lock:
            done.add(idx)
            results[idx] = summary
            for subj, rel, obj in triples:
                all_triples.append({
                    "subject": subj, "predicate": rel, "object": obj,
                    "book": t["book_name"], "window": t["section_idx"],
                })
            if resume_path and len(done) % 10 == 0:
                _save_resume(resume_path, done, results, all_triples)
        return idx, summary, triples

    if pending:
        print(f"  Processing {len(pending)} windows ({workers} workers)...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process, t): t for t in pending}
            for f in as_completed(futures):
                idx, _, _ = f.result()
                print(f"  [{idx}/{total}] done", flush=True)

    if resume_path:
        _save_resume(resume_path, done, results, all_triples)

    return sorted(results.items()), all_triples


def _save_resume(path: str, done: set, results: dict, triples: list[dict] | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {str(k): v for k, v in results.items()}
    data = {"done": sorted(done), "results": serializable}
    if triples is not None:
        data["triples_count"] = len(triples)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


# ============ 书籍加载 ============

def load_books(data_dir: str = DATA_DIR) -> list[dict]:
    books = []
    files = sorted(os.listdir(data_dir))
    for fname in files:
        if fname.endswith(".cleaned.txt"):
            path = os.path.join(data_dir, fname)
            size_kb = os.path.getsize(path) // 1024
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                windows = sliding_window_chunks(content)
                books.append({
                    "name": fname, "size_kb": size_kb,
                    "chars": len(content),
                    "est_tokens": len(content) // 4,
                    "windows": len(windows),
                })
    return books


def get_book_content(name: str) -> str:
    path = os.path.join(DATA_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ============ 三元组后处理 ============

def _canonicalize_entities(triples: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """使用 LLM 识别实体别名，将所有变体映射到规范实体名。
    返回 (更新后的triples, {variant: canonical} 映射表)"""
    if not triples:
        return triples, {}

    entities: set[str] = set()
    for t in triples:
        entities.add(t["subject"])
        entities.add(t["object"])

    candidates = sorted(e for e in entities if len(e) >= 2)

    if len(candidates) <= 1:
        return triples, {}

    entities_text = "\n".join(f"- {e}" for e in candidates)
    user_prompt = ENTITY_CANON_USER_PROMPT.format(entities_text=entities_text)

    msgs = [
        {"role": "system", "content": ENTITY_CANON_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    mapping: dict[str, str] = {}
    try:
        response = _call_model(msgs, 4096)
        groups = re.findall(r'\[GROUP\]\s*(.*?)\s*\[/GROUP\]', response, re.DOTALL | re.IGNORECASE)
        for group in groups:
            lines = [l.strip() for l in group.strip().split('\n') if l.strip()]
            if not lines:
                continue
            canonical = lines[0]
            mapping[canonical] = canonical
            for alias in lines[1:]:
                if alias and alias != canonical:
                    mapping[alias] = canonical
        if mapping:
            print(f"  [Entity Canon] {len(mapping)} entity variants mapped to canonical forms.")
    except Exception as e:
        print(f"  [Entity Canon] LLM call failed: {e}, skipping canonicalization.")
        return triples, {}

    if not mapping:
        return triples, {}

    alias_triples: list[dict] = []
    for t in triples:
        old_subj = t["subject"]
        old_obj = t["object"]
        new_subj = mapping.get(old_subj, old_subj)
        new_obj = mapping.get(old_obj, old_obj)
        t["subject"] = new_subj
        t["object"] = new_obj
        if new_subj != old_subj:
            alias_triples.append({
                "subject": new_subj, "predicate": "ALIAS", "object": old_subj,
                "book": t["book"], "window": t["window"],
            })
        if new_obj != old_obj and new_obj != new_subj:
            alias_triples.append({
                "subject": new_obj, "predicate": "ALIAS", "object": old_obj,
                "book": t["book"], "window": t["window"],
            })

    seen_alias: set[tuple[str, str, str]] = set()
    for t in alias_triples:
        key = (t["subject"], t["predicate"], t["object"])
        if key not in seen_alias:
            seen_alias.add(key)
            triples.append(t)

    return triples, mapping


def _deduplicate_triples(triples: list[dict]) -> list[dict]:
    """对 (subject, predicate, object) 去重，合并 evidence 记录多出处"""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for t in triples:
        key = (t["subject"], t["predicate"], t["object"])
        if key not in groups:
            groups[key] = []
        groups[key].append(t)

    unique: list[dict] = []
    for key, items in groups.items():
        evidence = [f"{it['book']}:{it['window']}" for it in items]
        evidence = sorted(set(evidence), key=lambda x: (x.split(':')[0], int(x.split(':')[1])))
        first = items[0]
        unique.append({
            "subject": key[0],
            "predicate": key[1],
            "object": key[2],
            "evidence": "; ".join(evidence),
            "evidence_count": len(evidence),
        })

    print(f"  [Dedup] {len(triples)} raw → {len(unique)} unique triples.")
    return unique


def _quality_filter(triples: list[dict]) -> list[dict]:
    """过滤低质量三元组：自引用、ALIAS事件描述、实体误用为predicate object等"""
    filtered = []
    stats: dict[str, int] = {}

    def _inc(key: str):
        stats[key] = stats.get(key, 0) + 1

    attr_objects: set[str] = set()
    role_objects: set[str] = set()
    entity_subjects: set[str] = set()

    for t in triples:
        entity_subjects.add(t["subject"])
        if t["predicate"] == "HAS_ATTRIBUTE":
            attr_objects.add(t["object"])
        if t["predicate"] == "HAS_ROLE":
            role_objects.add(t["object"])

    def _is_bad_alias(obj: str, subj: str) -> bool:
        if "'" in obj:
            return True
        if len(obj.split()) > 5:
            return True
        if f"of {subj}" in obj.lower():
            return True
        obj_lower = obj.lower()
        # preposition phrases → descriptive event/location, not a name
        if re.search(r'\b(at|on|in|near|by|from|for|with|of|to|over|under)\b', obj_lower):
            return True
        first_word = obj.split()[0].lower() if obj.split() else ""
        if first_word.endswith("ing") and first_word not in ("nothing", "something", "everything"):
            return True
        return False

    for t in triples:
        subj, pred, obj = t["subject"], t["predicate"], t["object"]

        # 1. 自引用
        if subj == obj:
            _inc("self_ref")
            continue

        # 2. ALIAS 的 subject 是属性值或角色值
        if pred == "ALIAS":
            if subj in attr_objects:
                _inc("alias_attr_subj")
                continue
            if subj in role_objects:
                _inc("alias_role_subj")
                continue
            if _is_bad_alias(obj, subj):
                _inc("alias_event_desc")
                continue

        # 3. PARTICIPATES_IN 的 object 是已知实体
        if pred == "PARTICIPATES_IN":
            if obj in entity_subjects:
                _inc("participates_entity")
                continue
            if "'s" in obj:
                _inc("participates_possessive")
                continue

        # 4. HAS_ATTRIBUTE 的 object 过长或包含所有格
        if pred == "HAS_ATTRIBUTE":
            if len(obj.split()) > 5:
                _inc("attr_too_long")
                continue
            if obj in entity_subjects:
                _inc("attr_is_entity")
                continue
            if "'s" in obj:
                _inc("attr_possessive")
                continue

        # 5. HAS_ROLE 的 object 是已知实体
        if pred == "HAS_ROLE":
            if obj in entity_subjects:
                _inc("role_is_entity")
                continue

        # 6. OWNS 的 object 是已知实体
        if pred == "OWNS":
            if obj in entity_subjects:
                _inc("owns_person")
                continue

        # 7. LOVES/MARRIES/REJECTS/PROPOSES_TO — object 不应是模糊实体
        if pred in ("LOVES", "MARRIES", "REJECTS", "PROPOSES_TO", "RIVALS"):
            if obj.lower() in ("nobody", "someone", "soldier", "anyone"):
                _inc("relation_vague_obj")
                continue

        filtered.append(t)

    label_map = {
        "self_ref": "self-referencing",
        "alias_attr_subj": "ALIAS with attribute-as-subject",
        "alias_role_subj": "ALIAS with role-as-subject",
        "alias_event_desc": "ALIAS with event-description object",
        "participates_entity": "PARTICIPATES_IN with entity-as-event",
        "participates_possessive": "PARTICIPATES_IN with possessive object",
        "attr_too_long": "HAS_ATTRIBUTE with >5 word object",
        "attr_is_entity": "HAS_ATTRIBUTE with entity-as-attribute",
        "attr_possessive": "HAS_ATTRIBUTE with possessive object",
        "role_is_entity": "HAS_ROLE with entity-as-role",
        "owns_person": "OWNS with person-as-possession",
        "relation_vague_obj": "relation with vague/non-entity object",
    }
    for key, label in label_map.items():
        if key in stats:
            print(f"  [Quality] Removed {stats[key]} triples: {label}.")

    return filtered


def _post_process_triples(triples: list[dict]) -> list[dict]:
    """三元组后处理管线：实体对齐 → 质量过滤 → 去重"""
    if not triples:
        return triples

    print(f"\n  Post-processing {len(triples)} raw triples...")

    triples, _ = _canonicalize_entities(triples)
    triples = _quality_filter(triples)
    triples = _deduplicate_triples(triples)

    return triples


def _save_triples_csv(triples: list[dict]) -> str:
    """保存三元组到 CSV 文件"""
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"triples_{ts}.csv")

    fieldnames = ["subject", "predicate", "object", "evidence", "evidence_count"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(triples)

    print(f"  Triples saved: {csv_path} ({len(triples)} rows)")
    return csv_path


# ============ 主流程 ============

def process_books(selected_names: list[str],
                  question: str | None = None,
                  triples_guide: str | None = None,
                  resume_path: str | None = None) -> dict:
    t_start = time.time()
    all_tasks = []
    total_windows = 0

    for name in selected_names:
        content = get_book_content(name)
        windows = sliding_window_chunks(content)
        section_total = len(windows)
        for i, w in enumerate(windows, 1):
            all_tasks.append({
                "global_idx": total_windows + i,
                "book_name": name,
                "section_idx": i,
                "section_total": section_total,
                "text": w["text"],
                "start": w["start"],
                "end": w["end"],
            })
        total_windows += section_total
        print(f"  [{name}] {section_total} windows")

    # 并行总结
    sorted_results, all_triples = summarize_all_parallel(all_tasks, WORKERS, resume_path)

    summaries = [s for _, s in sorted_results]

    # 三元组后处理
    if all_triples:
        all_triples = _post_process_triples(all_triples)

    # 最终回答 / QA 生成
    print(f"\n  All {len(summaries)} windows summarized ({len(all_triples)} triples after post-processing).")

    summaries_text = "\n\n".join(
        f"[Section {i+1}]\n{s}" for i, s in enumerate(summaries)
    )

    if question:
        user_content = f"Summaries:\n\n{summaries_text}\n\nQuestion: {question}"
    else:
        user_content = f"Summaries:\n\n{summaries_text}"

    if triples_guide:
        user_content += f"\n\nReference Knowledge Graph triples (use these to guide your output):\n{triples_guide}"

    if question:
        print(f"  Generating final answer...\n")
        answer_messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    else:
        print(f"  Generating QA pairs from summaries...\n")
        answer_messages = [
            {"role": "system", "content": QA_GENERATION_PROMPT},
            {"role": "user", "content": user_content},
        ]

    t0 = time.time()
    answer = _call_model(answer_messages, MAX_TOKENS_ANSWER)
    answer_time = time.time() - t0

    # 保存三元组 CSV
    csv_path = None
    if all_triples:
        csv_path = _save_triples_csv(all_triples)

    # 清理进度文件
    if resume_path and os.path.exists(resume_path):
        os.remove(resume_path)

    return {
        "answer": answer,
        "total_windows": total_windows,
        "total_time": time.time() - t_start,
        "answer_time": answer_time,
        "triples_count": len(all_triples),
        "triples_csv": csv_path,
    }


# ============ 交互界面 ============

def print_books_table():
    books = load_books()
    print(f"\n  {'Idx':<5} {'Filename':<25} {'Size':<8} {'Chars':<8} {'Windows':<8}")
    print("  " + "-" * 58)
    for i, b in enumerate(books):
        print(f"  {i:<5} {b['name']:<25} {b['size_kb']:<8} KB {b['chars']:<8,} {b['windows']:<8}")


def interactive():
    print("=" * 60)
    print("Long Text Comprehension Pipeline (Parallel)")
    print(f"Model: {MODEL} | Workers: {WORKERS}")
    print(f"Window: {WINDOW_SIZE}ch | Overlap: {OVERLAP}ch")
    print("=" * 60)

    books = load_books()
    print(f"\n{len(books)} books available.")
    print_books_table()

    # ── 选择书籍（保留原有逻辑） ──
    print("\nSelect books (indices like 0,1,2 or range 0-5 or 'all'):")
    while True:
        try:
            choice = input(">>> ").strip()
        except EOFError:
            choice = "all"
        if not choice:
            continue
        if choice.lower() == "all":
            selected = [b["name"] for b in books]
            break
        indices = []
        for token in re.split(r"[,，]", choice):
            token = token.strip()
            m = re.match(r"(\d+)\s*-\s*(\d+)", token)
            if m:
                indices.extend(range(int(m.group(1)), int(m.group(2)) + 1))
            elif token.isdigit():
                indices.append(int(token))
        indices = sorted(set(i for i in indices if 0 <= i < len(books)))
        if indices:
            selected = [books[i]["name"] for i in indices]
            break
        print(f"Invalid. Enter 0-{len(books)-1}.")

    tw = sum(b["windows"] for b in books if b["name"] in selected)
    est_serial = tw * 35
    est_parallel = est_serial // WORKERS
    print(f"\nSelected {len(selected)} book(s), ~{tw} windows.")
    print(f"Workers: {WORKERS}, estimated: ~{est_parallel // 60} min")

    # ── 问题输入（可通过 prompts.py 中 ENABLE_QUESTION_INPUT 开关控制） ──
    question: str | None = None
    if ENABLE_QUESTION_INPUT:
        question = input("\nQuestion (press Enter to skip → auto-generate QA pairs): ").strip()
        if not question:
            question = None
            print("  (No question entered, will auto-generate QA pairs from summaries.)")
    else:
        print("\n  [Question input disabled — will auto-generate QA pairs from summaries.]")

    # ── 三元组输入（可通过 prompts.py 中 ENABLE_TRIPLE_INPUT 开关控制） ──
    triples_guide: str | None = None
    if ENABLE_TRIPLE_INPUT:
        print("\nEnter reference triples (one per line: subject||predicate||object, empty line to finish):")
        lines = []
        while True:
            line = input().strip()
            if not line:
                break
            lines.append(line)
        if lines:
            triples_guide = "\n".join(lines)
            print(f"  ({len(lines)} reference triples recorded.)")
        else:
            print("  (No triples entered.)")

    # ── 确认执行 ──
    if question:
        proceed = input(f"\nProceed? [y/N]: ").strip().lower()
    else:
        proceed = input(f"\nProceed with auto QA generation? [y/N]: ").strip().lower()
    if proceed != "y":
        print("Cancelled.")
        return

    resume_path = os.path.join(os.path.dirname(__file__), "output", ".progress.json")
    result = process_books(selected, question=question,
                           triples_guide=triples_guide,
                           resume_path=resume_path)

    # ── 输出结果 ──
    print(f"\n{'='*60}")
    if question:
        print(f"Q: {question}")
    else:
        print("Auto-generated QA Pairs:")
    print(f"{'='*60}\n")
    print(result["answer"])
    print(f"\n{'='*60}")
    print(f"Time: {result['total_time']:.0f}s | Windows: {result['total_windows']} | Triples: {result.get('triples_count', 0)}")

    # ── 保存结果 ──
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"result_{ts}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        if question:
            f.write(f"Q: {question}\n\nA:\n{result['answer']}\n\n")
        else:
            f.write(f"Auto-generated QA Pairs:\n\n{result['answer']}\n\n")
        f.write(f"--- Stats ---\n")
        f.write(f"Workers: {WORKERS}\n")
        f.write(f"Windows: {result['total_windows']}\n")
        f.write(f"Total time: {result['total_time']:.0f}s\n")
        f.write(f"Triples: {result.get('triples_count', 0)}\n")
        if result.get('triples_csv'):
            f.write(f"Triples CSV: {result['triples_csv']}\n")
    print(f"\nSaved: {out_path}")
    if result.get('triples_csv'):
        print(f"Triples CSV: {result['triples_csv']}")


if __name__ == "__main__":
    interactive()
