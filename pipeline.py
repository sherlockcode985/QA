"""
长文本理解管线 — 对抗式三元组抽取 + 滑动窗口 + 汇总回答

流程:
  1. 滑动窗口切分文本 (sentence-aware)
  2. 对抗式窗口处理 (每个窗口 4 步):
     Step 1: 总结者 —— 概括章节内容
     Step 2: 抽取者 —— 读总结+原文，提取 ALIAS 三元组
     Step 3: 校验者 —— 逐条检查三元组合理性
     Step 4: 修正者 —— 根据反馈重新抽取（最多 MAX_ITERATIONS 轮）
  3. 并行处理所有窗口 (ThreadPoolExecutor, 按 index 排序拼合)
  4. 后处理三元组: 实体对齐(LLM) → 质量过滤 → 去重(合并evidence)
  5. 汇总所有总结, 基于总结回答问题 / 自动生成 QA 对

外部依赖文件（同学主要修改这些）:
  config.py  — API/模型/窗口/并发参数 & MAX_ITERATIONS
  prompts.py — 4 个角色提示词 & 功能开关
"""

import os, re, time, json, threading, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from config import (
    API_BASE, API_KEY, MODEL, DATA_DIR,
    WINDOW_SIZE, OVERLAP, STRIDE,
    WORKERS, MAX_TOKENS_SUMMARIZE, MAX_TOKENS_ANSWER, MAX_RETRIES, MAX_ITERATIONS,
    API_MAX_RETRIES, API_RETRY_BASE_DELAY,
)
from prompts import (
    ENABLE_QUESTION_INPUT,
    ENABLE_TRIPLE_INPUT,
    ENABLE_TRIPLE_EXTRACTION,
    ENABLE_EVIDENCE_VERIFICATION,
    TRIPLE_INSTRUCTION,
    SUMMARIZE_CHUNK_PROMPT,
    EXTRACT_TRIPLES_PROMPT,
    VALIDATE_TRIPLES_PROMPT,
    REVISE_TRIPLES_PROMPT,
    ENTITY_CANON_SYSTEM_PROMPT,
    ENTITY_CANON_USER_PROMPT,
    ANSWER_SYSTEM_PROMPT,
    QA_GENERATION_PROMPT,
    EVIDENCE_VERIFICATION_PROMPT,
)

client = OpenAI(base_url=API_BASE, api_key=API_KEY, timeout=600.0)


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
    """调用 LLM，带指数退避重试。
    重试次数由 config.py 中的 API_MAX_RETRIES 控制。
    """
    for attempt in range(API_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < API_MAX_RETRIES:
                delay = API_RETRY_BASE_DELAY * (attempt + 1)
                print(f"  [Retry {attempt + 1}/{API_MAX_RETRIES}] API call failed, waiting {delay}s: {e}")
                time.sleep(delay)
                continue
            raise RuntimeError(f"API call failed after {API_MAX_RETRIES + 1} attempts: {e}")


def _parse_triples_block(text: str) -> list[tuple[str, str, str]]:
    """从模型回复中解析 [TRIPLES]...[/TRIPLES] 块"""
    triples: list[tuple[str, str, str]] = []
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
    return triples


def _format_triples_text(triples: list[tuple[str, str, str]]) -> str:
    """将三元组列表格式化为 subject||predicate||object 文本"""
    return "\n".join(f"{s}||{p}||{o}" for s, p, o in triples)


def _parse_verdict(text: str) -> tuple[bool, str]:
    """解析 [VERDICT] 和 [FEEDBACK] 块，返回 (passed, feedback)"""
    passed = False
    feedback = ""

    vm = re.search(r'\[VERDICT\]\s*(.*?)\s*\[/VERDICT\]', text, re.DOTALL | re.IGNORECASE)
    if vm:
        verdict_text = vm.group(1).strip().upper()
        passed = "PASS" in verdict_text and "FAIL" not in verdict_text

    fm = re.search(r'\[FEEDBACK\]\s*(.*?)\s*\[/FEEDBACK\]', text, re.DOTALL | re.IGNORECASE)
    if fm:
        feedback = fm.group(1).strip()

    return passed, feedback


# ============ 对抗式窗口总结 ============

def _summarize_chunk(window_text: str, book_name: str,
                     idx: int, total: int) -> str:
    """Step 1: 总结者 —— 只做章节概括，不抽取三元组"""
    msgs = [
        {"role": "system", "content": SUMMARIZE_CHUNK_PROMPT},
        {"role": "user", "content": f"[{book_name} | {idx}/{total}]\n{window_text}"},
    ]
    r = _call_model(msgs, MAX_TOKENS_SUMMARIZE)
    return r.strip() if r else "[summary unavailable]"


def _extract_triples(summary: str, window_text: str) -> list[tuple[str, str, str]]:
    """Step 2: 抽取者 —— 读 summary + 原文，从中提取 ALIAS 三元组"""
    system_prompt = EXTRACT_TRIPLES_PROMPT.format(triple_instruction=TRIPLE_INSTRUCTION)
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Summary:\n{summary}\n\nOriginal Text:\n{window_text}"},
    ]
    r = _call_model(msgs, MAX_TOKENS_SUMMARIZE)
    return _parse_triples_block(r)


def _validate_triples(triples: list[tuple[str, str, str]],
                      window_text: str) -> tuple[bool, str]:
    """Step 3: 校验者 —— 逐条检查三元组是否合理，返回 (passed, feedback)"""
    triples_text = _format_triples_text(triples)
    msgs = [
        {"role": "system", "content": VALIDATE_TRIPLES_PROMPT},
        {"role": "user", "content": f"Original Text:\n{window_text}\n\nTriples to validate:\n{triples_text}"},
    ]
    r = _call_model(msgs, max(512, MAX_TOKENS_SUMMARIZE // 2))
    return _parse_verdict(r)


def _revise_triples(summary: str, window_text: str,
                    triples: list[tuple[str, str, str]],
                    feedback: str) -> list[tuple[str, str, str]]:
    """Step 4: 修正者 —— 根据校验反馈重新抽取三元组"""
    triples_text = _format_triples_text(triples)
    system_prompt = REVISE_TRIPLES_PROMPT.format(triple_instruction=TRIPLE_INSTRUCTION)
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Summary:\n{summary}\n\nOriginal Text:\n{window_text}\n\nYour previous triples:\n{triples_text}\n\nReviewer feedback:\n{feedback}"},
    ]
    r = _call_model(msgs, MAX_TOKENS_SUMMARIZE)
    revised = _parse_triples_block(r)
    return revised if revised else triples


def summarize_one_adversarial(window_text: str, book_name: str,
                              idx: int, total: int) -> tuple[str, list[tuple[str, str, str]]]:
    """对抗式多步总结：总结 → 抽取 → 校验 → 修正（最多 MAX_ITERATIONS 轮）
    线程安全，返回 (summary, [(subject, predicate, object), ...])
    当 ENABLE_TRIPLE_EXTRACTION=False 时跳过三元组相关步骤。"""
    for attempt in range(MAX_RETRIES + 1):
        try:
            # Step 1: 总结
            summary = _summarize_chunk(window_text, book_name, idx, total)

            # 三元组抽取开关
            triples: list[tuple[str, str, str]] = []
            if ENABLE_TRIPLE_EXTRACTION:
                # Step 2: 抽取
                triples = _extract_triples(summary, window_text)

                # Step 3-4: 校验→修正 循环
                if triples:
                    for iteration in range(MAX_ITERATIONS):
                        passed, feedback = _validate_triples(triples, window_text)
                        if passed:
                            break
                        if iteration < MAX_ITERATIONS - 1:
                            triples = _revise_triples(summary, window_text, triples, feedback)

            if summary:
                return summary, triples
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"  [Warn] summarize_one_adversarial failed after {MAX_RETRIES + 1} attempts: {e}")

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
        summary, triples = summarize_one_adversarial(t["text"], t["book_name"],
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


def _parse_qa_pairs(answer: str) -> list[tuple[int, str, str]]:
    """从自动生成的 QA 文本中解析出单个 Q/A 对。
    返回 [(q_number, question_text, answer_text), ...]"""
    m = re.search(r'\[QA_PAIRS\]\s*(.*?)\s*\[/QA_PAIRS\]', answer, re.DOTALL | re.IGNORECASE)
    if not m:
        return []
    content = m.group(1)
    pairs = []
    for qm in re.finditer(r'Q(\d+):\s*(.*?)\s*A\1:\s*(.*?)(?=Q\d+:|$)', content, re.DOTALL):
        pairs.append((int(qm.group(1)), qm.group(2).strip(), qm.group(3).strip()))
    return pairs


def _group_evidence_by_q(evidence_text: str) -> dict[int, list[str]]:
    """将证据文本按 Q 编号分组。
    返回 {q_number: [verbatim_quote, ...]}
    如果证据块没有 Q 编号，按出现顺序分配。"""
    by_q: dict[int, list[str]] = {}
    # 按出现顺序收集所有证据块
    unassigned: list[str] = []

    for em in re.finditer(r'\[EVIDENCE\]\s*(.*?)\s*\[/EVIDENCE\]', evidence_text, re.DOTALL | re.IGNORECASE):
        block = em.group(1)

        # 提取 Q 编号
        qm = re.search(r'Q:\s*(\d+)', block)
        qnum = int(qm.group(1)) if qm else None

        # 提取 Verbatim Evidence：从 "Verbatim Evidence:" 后的第一个 " 到最后一个 "
        quote = None
        vi = block.find('Verbatim Evidence:')
        if vi >= 0:
            qs = block.find('"', vi)
            if qs >= 0:
                qe = block.rfind('"', qs + 1)
                if qe > qs:
                    quote = block[qs + 1:qe].strip()

        if quote:
            if qnum is not None:
                by_q.setdefault(qnum, []).append(quote)
            else:
                unassigned.append(quote)

    # 如果没有 Q 编号的证据，按出现顺序分配给 1, 2, 3...
    if unassigned:
        max_q = max(by_q.keys()) if by_q else 0
        for i, q in enumerate(unassigned, 1):
            by_q.setdefault(max_q + i, []).append(q)

    return by_q


def _is_auto_qa_answer(answer: str) -> bool:
    """判断回答是否为自动生成的 QA pairs"""
    return bool(re.search(r'\[QA_PAIRS\]', answer, re.IGNORECASE))


def _verify_evidence(answer: str, chunk_registry: dict,
                      question: str | None = None,
                      max_sections: int = 30) -> tuple[str, dict[int, list[str]]]:
    """从原文中为答案中的每个事实性陈述提取逐字证据。

    answer 中应包含 [Section N] 引用，函数会：
    1. 解析出所有被引用的 section
    2. 从 chunk_registry 中取出对应的原文
    3. 调用 LLM 提取逐字证据

    返回 (evidence_full_text, per_q_evidence)
    per_q_evidence: {q_number: [verbatim_quote, ...]}，非 QA 模式时为 {}
    """
    is_qa = _is_auto_qa_answer(answer)

    # 1. 解析答案中引用的所有 section 编号
    cited: set[int] = set()
    # Pattern 1: [Section N] or [Sections N-M] (range via hyphen/dash)
    for m in re.finditer(r'\[Section(?:s)?\s*(\d+)(?:\s*[-–—]\s*(\d+))?\]', answer, re.IGNORECASE):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        for s in range(start, end + 1):
            cited.add(s)
    # Pattern 2: [Sections N, M, ...] (comma-separated list)
    for m in re.finditer(r'\[Section(?:s)?\s+(\d+(?:\s*,\s*\d+)+)\]', answer, re.IGNORECASE):
        for token in re.split(r'\s*,\s*', m.group(1)):
            s = int(token.strip())
            if 1 <= s <= len(chunk_registry) * 2:
                cited.add(s)

    if not cited:
        msg = "\n\n[EVIDENCE]\n(答案中未发现 [Section N] 引用，无法验证原文证据。)\n[/EVIDENCE]"
        return msg, {}

    # 2. 取出被引用 section 的原文
    cited_texts = []
    for sec in sorted(cited):
        if sec in chunk_registry:
            entry = chunk_registry[sec]
            text_preview = entry["text"]
            if len(text_preview) > 3000:
                text_preview = text_preview[:1500] + "\n...[...]...\n" + text_preview[-1500:]
            cited_texts.append(
                f"[Section {sec}] ({entry['book']}, chars {entry['start']}-{entry['end']})\n{text_preview}"
            )
    # If too many sections cited, sample the most important ones
    if len(cited_texts) > max_sections:
        cited_texts = cited_texts[:max_sections]
        cited_texts.append("... (余下 cited sections 省略)")

    if not cited_texts:
        msg = "\n\n[EVIDENCE]\n(引用的 sections 在 registry 中未找到对应原文。)\n[/EVIDENCE]"
        return msg, {}

    cited_sections_text = "\n\n---\n\n".join(cited_texts)

    # 3. 调用 LLM 提取证据
    qa_hint = (
        "\n\nThe answer contains multiple QA pairs (Q1, Q2, ...). "
        "For each evidence block, identify which Q number the claim belongs to "
        "and include it in the output as 'Q: <number>'."
        if is_qa else ""
    )
    msgs = [
        {"role": "system", "content": EVIDENCE_VERIFICATION_PROMPT},
        {"role": "user", "content": (
            f"CITED SECTIONS (original text):\n{cited_sections_text}\n\n"
            f"QUESTION: {question or 'N/A'}{qa_hint}\n\n"
            f"ANSWER:\n{answer}"
        )},
    ]

    try:
        evidence = _call_model(msgs, 4096)
    except Exception as e:
        evidence = f"\n\n[EVIDENCE]\n(证据验证调用失败: {e})\n[/EVIDENCE]"
        return "\n\n" + evidence.strip(), {}

    evidence_full = "\n\n" + evidence.strip()
    per_q = _group_evidence_by_q(evidence) if is_qa else {}
    return evidence_full, per_q


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
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

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

    # 构建 Chunk Registry：section_idx → {book, start, end, text}
    chunk_registry: dict[int, dict] = {}
    for t in all_tasks:
        chunk_registry[t["global_idx"]] = {
            "book": t["book_name"],
            "section_within_book": t["section_idx"],
            "start": t.get("start", 0),
            "end": t.get("end", 0),
            "text": t["text"],
        }

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

    # 保存 Chunk Registry
    ts = time.strftime("%Y%m%d_%H%M%S")
    registry_path = os.path.join(out_dir, f"chunk_registry_{ts}.json")
    # 保存时去掉原文 text 以减少体积（仅保留 start/end/book 用于追溯）
    registry_slim = {}
    for idx, entry in chunk_registry.items():
        registry_slim[str(idx)] = {
            "book": entry["book"],
            "start": entry["start"],
            "end": entry["end"],
            "section_within_book": entry["section_within_book"],
        }
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry_slim, f, ensure_ascii=False, indent=2)
    print(f"  Chunk registry saved: {registry_path}")

    # 原文证据验证
    evidence_text = ""
    per_q_evidence: dict[int, list[str]] = {}
    if ENABLE_EVIDENCE_VERIFICATION and answer.strip():
        print(f"  Verifying evidence from original text...")
        evidence_text, per_q_evidence = _verify_evidence(answer, chunk_registry, question)

    # 格式化最终输出：QA 模式 → Q/A/E 交错；单问题模式 → 附带证据块
    is_qa = _is_auto_qa_answer(answer)
    if is_qa and per_q_evidence:
        qa_pairs = _parse_qa_pairs(answer)
        formatted_parts: list[str] = []
        for qnum, qtext, atext in qa_pairs:
            formatted_parts.append(f"Q{qnum}: {qtext}\nA{qnum}: {atext}")
            quotes = per_q_evidence.get(qnum, [])
            if quotes:
                joined = "; ".join(f'"{q}"' for q in quotes)
                formatted_parts.append(f"E{qnum}: {joined}")
        answer_with_evidence = "\n\n".join(formatted_parts)
    else:
        answer_with_evidence = answer + evidence_text

    # 清理进度文件
    if resume_path and os.path.exists(resume_path):
        os.remove(resume_path)

    return {
        "answer": answer,
        "answer_with_evidence": answer_with_evidence,
        "evidence_text": evidence_text,
        "per_q_evidence": per_q_evidence,
        "total_windows": total_windows,
        "total_time": time.time() - t_start,
        "answer_time": answer_time,
        "triples_count": len(all_triples),
        "triples_csv": csv_path,
        "registry_path": registry_path,
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
    print(f"Triple extraction: {'ON' if ENABLE_TRIPLE_EXTRACTION else 'OFF'}")
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
    print(result["answer_with_evidence"])
    print(f"\n{'='*60}")
    print(f"Time: {result['total_time']:.0f}s | Windows: {result['total_windows']} | Triples: {result.get('triples_count', 0)}")

    # ── 保存结果 ──
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"result_{ts}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{result['answer_with_evidence']}\n\n")
        f.write(f"--- Stats ---\n")
        f.write(f"Workers: {WORKERS}\n")
        f.write(f"Windows: {result['total_windows']}\n")
        f.write(f"Total time: {result['total_time']:.0f}s\n")
        f.write(f"Triples: {result.get('triples_count', 0)}\n")
        if result.get('triples_csv'):
            f.write(f"Triples CSV: {result['triples_csv']}\n")
        if result.get('registry_path'):
            f.write(f"Chunk Registry: {result['registry_path']}\n")
    print(f"\nSaved: {out_path}")
    if result.get('triples_csv'):
        print(f"Triples CSV: {result['triples_csv']}")


if __name__ == "__main__":
    interactive()
