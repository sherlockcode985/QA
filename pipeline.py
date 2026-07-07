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


# 文件名全是数字 ID，书名在文本第一行
# 少数文件第一行不是书名，需要用数字 ID 手动映射
BOOK_TITLE_MAP = {
    "75170": "The Sound and the Fury",
    "45839": "Dracula",
    "9603": "The Dream of the Red Chamber",
    "62897": "The Arabian Nights",
    "19860": "The Arabian Nights Entertainments",
}


def get_book_title(name: str) -> str:
    book_id = name.replace(".cleaned.txt", "")
    if book_id in BOOK_TITLE_MAP:
        return BOOK_TITLE_MAP[book_id]
    path = os.path.join(DATA_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return f.readline().strip()


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


def _normalize_text(text: str) -> str:
    """统一所有可能造成匹配失败的文本差异，用于 evidence 软匹配"""
    import unicodedata
    t = text.replace('\\"', '"').replace("\\'", "'")
    t = re.sub(r'\\n', '\n', t)
    # Unicode 正规化：弯引号→直引号，非断空格→空格，全角→半角等
    t = unicodedata.normalize('NFKC', t)
    # 常见标点变体统一
    t = t.replace('—', '--').replace('–', '-')  # 长/短破折号
    t = t.replace('…', '...')  # 省略号
    t = re.sub(r'\s+', ' ', t).strip().lower()
    return t


def _evidence_confidence(evidence: str, source_text: str) -> float:
    """计算 evidence 在原文中的匹配置信度 0~1，不做硬丢弃"""
    ev = _normalize_text(evidence)
    src = _normalize_text(source_text)
    if ev in src:
        return 1.0
    # 滑动窗口找最长连续匹配比例
    ev_len = len(ev)
    best = 0
    # 只看合理长度的窗口（至少 20 字符，不超过原文最长匹配）
    for start in range(ev_len):
        for end in range(start + 20, min(start + ev_len, ev_len) + 1):
            chunk = ev[start:end]
            if chunk in src:
                best = max(best, (end - start) / ev_len)
            else:
                break
    return best


def _strip_code_fence(text: str) -> str:
    """去掉 LLM 在 JSON 外包的 ```json 代码块标记"""
    s = re.sub(r'^```(?:json)?\s*\n?', '', text.strip(), flags=re.IGNORECASE)
    s = re.sub(r'\n?```\s*$', '', s, flags=re.IGNORECASE)
    return s.strip()


def _parse_qa_pairs(answer: str) -> list[tuple[int, str, str]]:
    """从 JSON 格式的 QA 数组中解析单个 Q/A 对。
    返回 [(index, question_text, answer_text), ...]"""
    try:
        data = json.loads(answer)
        if not isinstance(data, list):
            data = [data]
        return [(i + 1, item.get("question", ""), item.get("answer", "")) for i, item in enumerate(data)]
    except (json.JSONDecodeError, TypeError):
        return []


def _strip_section_refs(text: str) -> str:
    """去掉答案中的 [N] 引用标记，仅用于显示。"""
    return re.sub(r'\s*\[(\d+)\]', '', text).strip()


def _group_evidence_by_q(qa_array_str: str) -> dict[int, list[str]]:
    """从 JSON 格式的 QA 数组中提取 evidence 字段，按顺序分组。
    返回 {index: [verbatim_quote, ...]}"""
    by_q: dict[int, list[str]] = {}
    try:
        data = json.loads(qa_array_str)
        if not isinstance(data, list):
            data = [data]
        for i, item in enumerate(data):
            ev = item.get("evidence", [])
            if isinstance(ev, str):
                ev = [ev]
            if ev:
                by_q[i + 1] = ev
    except (json.JSONDecodeError, TypeError):
        pass
    return by_q


def _is_auto_qa_answer(answer: str) -> bool:
    """判断回答是否为 JSON 数组格式的 QA pairs"""
    try:
        data = json.loads(answer)
        return isinstance(data, list)
    except (json.JSONDecodeError, TypeError):
        return False


def _verify_evidence(answer: str, chunk_registry: dict,
                      question: str | None = None,
                      max_sections: int = 30,
                      out_dir: str | None = None) -> tuple[str, dict[int, list[str]]]:
    """从原文中为 JSON 格式的 QA 数组提取逐字证据。

    answer 应为 JSON 数组/对象，answer/response 字段含 [Section N] 引用。
    返回 (json_string_with_evidence, per_q_evidence)
    """
    # 1. 解析 JSON，收集所有 [Section N] 引用
    try:
        qa_list = json.loads(answer)
        if not isinstance(qa_list, list):
            qa_list = [qa_list]
    except json.JSONDecodeError:
        return answer, {}

    # 从 answer 字段或 response 字段中提取 [N] 引用
    def _get_ans(item: dict) -> str:
        return item.get("answer") or item.get("response") or ""

    cited: set[int] = set()
    for qa in qa_list:
        a_text = _get_ans(qa)
        for m in re.finditer(r'\[(\d+)\]', a_text):
            s = int(m.group(1))
            if 1 <= s <= len(chunk_registry) * 2:
                cited.add(s)

    if not cited:
        return answer, {}

    # 2. 取出被引用 section 的原文
    cited_texts = []
    for sec in sorted(cited):
        if sec in chunk_registry:
            entry = chunk_registry[sec]
            cited_texts.append(
                f"[{sec}] ({entry['book']}, chars {entry['start']}-{entry['end']})\n{entry['text']}"
            )
    if len(cited_texts) > max_sections:
        cited_texts = cited_texts[:max_sections]
        cited_texts.append("... (余下 cited sections 省略)")

    if not cited_texts:
        return answer, {}

    cited_sections_text = "\n\n---\n\n".join(cited_texts)

    # 3. 调用 LLM 在原 JSON 上补充 evidence 字段
    msgs = [
        {"role": "system", "content": EVIDENCE_VERIFICATION_PROMPT},
        {"role": "user", "content": (
            f"CITED SECTIONS (original text):\n{cited_sections_text}\n\n"
            f"QA ARRAY:\n{json.dumps(qa_list, ensure_ascii=False, indent=2)}"
        )},
    ]

    try:
        result_text = _call_model(msgs, MAX_TOKENS_ANSWER)
        cleaned = _strip_code_fence(result_text)
        enriched = json.loads(cleaned)
    except Exception as e:
        print(f"  [Evidence] 验证失败: {e}")
        return answer, {}

    per_q = _group_evidence_by_q(cleaned)
    total_ev = sum(len(v) for v in per_q.values())
    print(f"  [Evidence] 验证完成, {total_ev} 条证据")

    # ── 软校验：检查每条 evidence 在原文中是否有匹配 ──
    low_conf: list[tuple[int, int, str, float]] = []  # (qa_idx, ev_idx, snippet, score)
    combined_src = "\n".join(
        re.sub(r'\[(\d+)\] \(.*?\)\n', '', s)
        for s in cited_texts
    )
    for q_idx, ev_list in per_q.items():
        for ev_i, ev_text in enumerate(ev_list):
            conf = _evidence_confidence(ev_text, combined_src)
            if conf < 0.5:
                snippet = ev_text[:80].replace('\n', ' ')
                low_conf.append((q_idx, ev_i, snippet, conf))

    if low_conf:
        print(f"  [Evidence] 软校验: {len(low_conf)}/{total_ev} 条置信度偏低 (<0.5):")
        for q_idx, ev_i, snippet, conf in low_conf:
            print(f"    Q{q_idx}[{ev_i}] conf={conf:.2f} | {snippet}...")

    # ── 保存审计文件（不污染输出 JSON） ──
    if out_dir and low_conf:
        ts = time.strftime("%Y%m%d_%H%M%S")
        audit_path = os.path.join(out_dir, f"evidence_audit_{ts}.txt")
        with open(audit_path, "w", encoding="utf-8") as f:
            f.write("Evidence 软校验审计报告\n")
            f.write(f"共 {len(low_conf)}/{total_ev} 条置信度 < 0.5，建议人工核查\n")
            f.write("=" * 50 + "\n\n")
            for q_idx, ev_i, snippet, conf in low_conf:
                f.write(f"Q{q_idx}[{ev_i}]  conf={conf:.2f}\n")
                f.write(f"  Evidence: {snippet}\n\n")
        print(f"  [Evidence] 审计报告: {audit_path}")

    return cleaned, per_q


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

    # 将真实书名（从文件第一行提取）显式传给 LLM
    book_titles = {name: get_book_title(name) for name in selected_names}
    book_context = f"Books: {', '.join(book_titles.values())}\n\n"

    if question:
        user_content = book_context + f"Summaries:\n\n{summaries_text}\n\nQuestion: {question}"
    else:
        user_content = book_context + f"Summaries:\n\n{summaries_text}"

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
    answer = _strip_code_fence(_call_model(answer_messages, MAX_TOKENS_ANSWER))
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
        evidence_text, per_q_evidence = _verify_evidence(answer, chunk_registry, question, out_dir=out_dir)

    # 格式化最终输出：evidence 验证成功则用补充后的 JSON
    final_json = evidence_text if evidence_text and per_q_evidence else answer
    # 去掉 answer 中的 [N] 引用，并确保每个 QA 项都有 evidence 字段
    try:
        parsed = json.loads(final_json)
        if isinstance(parsed, list):
            for item in parsed:
                if "answer" in item:
                    item["answer"] = _strip_section_refs(item["answer"])
                if "answer" in item and "evidence" not in item:
                    item["evidence"] = []
                if "book" in item:
                    # LLM 可能输出文件名或真实书名，统一修正为 `book_titles` 中的值
                    raw = item["book"]
                    for fname, title in book_titles.items():
                        if raw == title or raw in fname or fname.startswith(raw):
                            item["book"] = title
                            break
                    else:
                        item["book"] = raw
        elif isinstance(parsed, dict) and "answer" in parsed:
            parsed["answer"] = _strip_section_refs(parsed["answer"])
            if "evidence" not in parsed:
                parsed["evidence"] = []
            if "book" in parsed:
                raw = parsed["book"]
                for fname, title in book_titles.items():
                    if raw == title or raw in fname or fname.startswith(raw):
                        parsed["book"] = title
                        break
                else:
                    parsed["book"] = raw
        answer_with_evidence = json.dumps(parsed, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        answer_with_evidence = final_json

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

    result = process_books(selected, question=question,
                           triples_guide=triples_guide,
                           resume_path=None)

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
