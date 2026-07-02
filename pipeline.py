"""
长文本理解管线 — 滑动窗口 + 并行分段总结 + 汇总回答 + 三元组抽取

流程:
  1. 滑动窗口切分文本 (sentence-aware)
  2. 并行总结所有窗口 (ThreadPoolExecutor, 按 index 排序拼合)
  3. 后处理三元组: 关系统一 → 实体对齐(LLM) → 去重(合并evidence)
  4. 汇总所有总结, 基于总结回答问题
"""

import os, re, time, json, threading, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ============ 配置 ============
API_BASE = "https://api.v3.cm/v1"
API_KEY = os.environ.get("API_KEY", "sk-7UYrjDTvNGkCiSof5bAb604870C1401b88Ac44FfF4C569Cc")
MODEL = "claude-sonnet-5"
DATA_DIR = os.path.join(os.path.dirname(__file__), "books", "train")

# --- 规范谓词集合（模型只能使用这些 predicate） ---
CANONICAL_PREDICATES = {
    "ALIAS", "HAS_ROLE", "HAS_ATTRIBUTE",
    "LOVES", "MARRIES", "PROPOSES_TO", "REJECTS",
    "KNOWS", "FRIEND_OF", "RIVALS",
    "WORKS_FOR", "EMPLOYS",
    "LIVES_IN", "OWNS",
    "HAS_FAMILY_RELATION", "PARENT_OF", "CHILD_OF", "SIBLING_OF",
    "AUNT_OF", "UNCLE_OF",
    "LOCATED_IN", "VISITS", "PARTICIPATES_IN",
}

# 自由文本谓词 → 规范谓词映射（兜底归一化）
PREDICATE_NORMALIZATION = {
    # 别名
    "外号": "ALIAS", "别名": "ALIAS", "绰号": "ALIAS", "人称": "ALIAS", "又称": "ALIAS",
    # 身份 / 属性
    "身份": "HAS_ROLE", "职位": "HAS_ROLE", "职业": "HAS_ROLE", "是": "HAS_ROLE",
    "担任": "HAS_ROLE", "成为": "HAS_ROLE", "角色": "HAS_ROLE",
    "特点": "HAS_ATTRIBUTE", "性格": "HAS_ATTRIBUTE",
    # 爱慕
    "爱慕": "LOVES", "喜欢": "LOVES", "爱": "LOVES", "暗恋": "LOVES",
    "爱慕对象": "LOVES", "被追求": "LOVES", "喜欢的人": "LOVES", "心上人": "LOVES",
    # 求婚
    "追求": "PROPOSES_TO", "求婚": "PROPOSES_TO", "求爱": "PROPOSES_TO",
    # 拒绝
    "拒绝": "REJECTS", "拒绝求婚": "REJECTS", "拒绝求爱": "REJECTS",
    # 婚姻
    "夫妻": "MARRIES", "结婚": "MARRIES", "嫁给": "MARRIES",
    "娶": "MARRIES", "丈夫": "MARRIES", "妻子": "MARRIES",
    "未婚妻": "PROPOSES_TO", "未婚夫": "PROPOSES_TO",
    # 亲属
    "师徒": "HAS_FAMILY_RELATION", "父亲": "PARENT_OF", "母亲": "PARENT_OF",
    "儿子": "CHILD_OF", "女儿": "CHILD_OF",
    "兄弟": "SIBLING_OF", "姐妹": "SIBLING_OF",
    "叔": "UNCLE_OF", "伯": "UNCLE_OF", "舅": "UNCLE_OF",
    "姨": "AUNT_OF", "姑": "AUNT_OF",
    "侄": "HAS_FAMILY_RELATION", "亲属": "HAS_FAMILY_RELATION",
    "亲戚": "HAS_FAMILY_RELATION", "家人": "HAS_FAMILY_RELATION",
    # 持有
    "持有": "OWNS", "拥有": "OWNS", "所属": "OWNS", "拥有者": "OWNS",
    "主人": "OWNS", "归属": "OWNS",
    # 雇佣
    "雇主": "WORKS_FOR", "仆人": "WORKS_FOR", "雇佣": "WORKS_FOR",
    "员工": "WORKS_FOR", "打工": "WORKS_FOR", "服务": "WORKS_FOR",
    "雇佣者": "EMPLOYS",
    # 位置
    "位于": "LOCATED_IN", "住在": "LIVES_IN", "来自": "LOCATED_IN",
    "出生地": "LOCATED_IN", "在": "LOCATED_IN",
    # 社交
    "认识": "KNOWS", "相识": "KNOWS", "知道": "KNOWS",
    "朋友": "FRIEND_OF", "好友": "FRIEND_OF",
    "敌人": "RIVALS", "仇人": "RIVALS", "对手": "RIVALS",
    # 杀人
    "杀死": "KILLS", "杀害": "KILLS", "谋杀": "KILLS",
    # 组织
    "成员": "MEMBER_OF", "加入": "MEMBER_OF",
    # 教导
    "教导": "MENTOR_OF", "老师": "MENTOR_OF", "指导": "MENTOR_OF",
    "师傅": "MENTOR_OF",
    # 参与
    "参加": "PARTICIPATES_IN", "参与": "PARTICIPATES_IN",
    # 拜访
    "拜访": "VISITS", "访问": "VISITS",
}

# 三元组提取指令（Knowledge Graph Triple: subject || predicate || object）
TRIPLE_INSTRUCTION = f"""Extract Knowledge Graph triples representing CHARACTER-LEVEL long-term facts.
Be SELECTIVE — 15-25 high-quality triples per section is better than 50 noisy ones.

FORMAT: subject||predicate||object

═══ VALID SUBJECT ═══
  ONLY proper named entities: a specific person (Gabriel Oak, William Boldwood, not "Boldwood"),
  named place (Weatherbury, Casterbridge), named organization, or named significant object.
  NEVER use: adjectives, abstract concepts, events, roles, or descriptions as subjects.
  WRONG: the maltster || HAS_ATTRIBUTE || very old
  WRONG: Farm Worker || ALIAS || Farmer
  RIGHT: maltster's actual name || HAS_ROLE || Maltster

═══ VALID OBJECT by predicate ═══
  HAS_ROLE:      GENERIC role/title ONLY: Farmer, Shepherd, Servant, Soldier, Bailiff, Maid, Clerk.
                 NEVER a person's name. WRONG: X||HAS_ROLE||Baily Pennyways; X||HAS_ROLE||Gabriel Oak
  HAS_ATTRIBUTE: a TRAIT (1-4 words): brave, wealthy, tall, "28 years old", headstrong, handsome.
                 NEVER a place, organization, role, or person name.
                 WRONG: X||HAS_ATTRIBUTE||Church of England; X||HAS_ATTRIBUTE||Eleventh Dragoon-Guards Soldier
  ALIAS:         an alternate name/nickname for the subject. Subject MUST be a person or named place.
                 WRONG: Farm Worker||ALIAS||Farmer (roles are not entities)
  OWNS:          significant named possessions only (farm, house, horse, dog). NOT trivial items.
  KNOWS:         established acquaintances only. If A KNOWS B, do NOT also create B KNOWS A.
  LIVES_IN:      primary residence only (Gabriel Oak||LIVES_IN||Weatherbury), not temporary stays.
  PARTICIPATES_IN: NAMED events ONLY: "The Storm", "sheep-shearing", "Greenhill Fair".
                 NEVER a person. WRONG: Gabriel Oak||PARTICIPATES_IN||Bathsheba Everdene
                 NEVER a place. WRONG: Gabriel Oak||PARTICIPATES_IN||Casterbridge

═══ FORBIDDEN ═══
  X||HAS_ROLE||<person name>    ← role objects must be generic roles, never persons
  X||HAS_ATTRIBUTE||<place/org>  ← attributes are traits, not locations or organizations
  X||HAS_ATTRIBUTE||<person name> ← attributes are traits, not people
  <role>||ALIAS||<role variant>  ← roles are not entities, don't create ALIAS for them
  <description>||ANY||ANY        ← "young girl", "the maltster", "a soldier" are not entities
  X||KNOWS||Y and Y||KNOWS||X   ← KNOWS is symmetric, only output one direction
  X||PARTICIPATES_IN||<person>   ← events are not people
  X||PARTICIPATES_IN||<place>    ← events are not locations

Canonical predicates: {', '.join(sorted(CANONICAL_PREDICATES))}
Extract only facts EXPLICITLY stated in the text."""

# --- 滑动窗口 ---
WINDOW_SIZE = 4000
OVERLAP = 1000
STRIDE = WINDOW_SIZE - OVERLAP

# --- 并行 ---
WORKERS = 6
MAX_TOKENS_SUMMARIZE = 2048
MAX_TOKENS_ANSWER = 8192
MAX_RETRIES = 2

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
                # subject || predicate || object
                triples.append((parts[0], parts[1], parts[2]))

    if not summary:
        summary = text.strip()

    return summary, triples


def summarize_one(window_text: str, book_name: str,
                  idx: int, total: int) -> tuple[str, list[tuple[str, str, str]]]:
    """总结单个窗口并同步提取三元组（线程安全，可被并行调用）
    返回 (summary, [(subject, predicate, object), ...])"""
    rel_list = ', '.join(sorted(CANONICAL_PREDICATES))
    system_prompt = (
        "You are a literary analyst. For the given book section:\n"
        "1. Write a 2-3 sentence summary including key events, characters, and details.\n"
        "2. Extract 15-25 high-quality Knowledge Graph triples (subject||predicate||object).\n\n"
        "=== TRIPLE EXTRACTION RULES (follow strictly) ===\n\n"
        f"PREDICATES (use ONLY these): {rel_list}\n\n"
        "═══ SUBJECT — only proper named entities ═══\n"
        "  RIGHT: Gabriel Oak, William Boldwood, Bathsheba Everdene, Weatherbury\n"
        "  WRONG: Boldwood (use full name), the maltster (find the name), a soldier (find the name)\n"
        "  WRONG: 'young girl', 'Liddy's sister', 'the stranger' — descriptions, NOT entities\n"
        "  Use the MOST COMPLETE name known: William Boldwood, not Boldwood.\n\n"
        "═══ OBJECT by predicate — what is valid ═══\n"
        "  HAS_ROLE: GENERIC role ONLY — Farmer, Shepherd, Servant, Soldier, Bailiff, Maid, Clerk.\n"
        "    WRONG: Gabriel Oak||HAS_ROLE||Baily Pennyways (person, not role)\n"
        "    WRONG: Cain Ball||HAS_ROLE||Gabriel Oak (person, not role)\n"
        "  HAS_ATTRIBUTE: a TRAIT (1-4 words) — brave, wealthy, tall, '28 years old', headstrong.\n"
        "    WRONG: X||HAS_ATTRIBUTE||Church of England (that's a religion/organization)\n"
        "    WRONG: X||HAS_ATTRIBUTE||Eleventh Dragoon-Guards Soldier (that's a role)\n"
        "    WRONG: 'clever man in talents', 'observant of stars' (verbose descriptions)\n"
        "  OWNS: significant named possessions only (farm, house, horse, dog). Skip trivial items.\n"
        "    NEVER a person: Gabriel Oak||OWNS||Fanny Robin is WRONG.\n"
        "  KNOWS: established acquaintances. Do NOT create both A||KNOWS||B and B||KNOWS||A.\n"
        "  ALIAS: alternate name for a person/place. NOT for roles (Farm Worker is a role, not entity).\n"
        "  PARTICIPATES_IN: NAMED events only. NEVER a person or place name.\n"
        "    WRONG: Gabriel Oak||PARTICIPATES_IN||Bathsheba Everdene (person, not event)\n"
        "    RIGHT: Gabriel Oak||PARTICIPATES_IN||The Storm\n"
        "  HAS_ATTRIBUTE: a TRAIT in 1-4 words. WRONG: 'imposing height and breadth', 'clever man in talents'.\n\n"
        "Output format:\n"
        "[SUMMARY]\n<your 2-3 sentence summary>\n[/SUMMARY]\n"
        "[TRIPLES]\nsubject||predicate||object\n[/TRIPLES]\n\n"
        f"Triple extraction guide:\n{TRIPLE_INSTRUCTION}"
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


# ============ 书籍加载 ============

def load_books(data_dir: str) -> list[dict]:
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


# ============ 并行总结 ============

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


# ============ 三元组后处理 ============

def _normalize_predicates(triples: list[dict]) -> list[dict]:
    """将自由文本谓词映射到规范谓词集合，过滤无法映射的"""
    normalized = []
    dropped = 0
    for t in triples:
        pred = t["predicate"].strip()
        # 如果已是规范谓词，直接保留
        if pred in CANONICAL_PREDICATES:
            t["predicate"] = pred
            normalized.append(t)
            continue
        # 尝试通过映射表归一化
        mapped = PREDICATE_NORMALIZATION.get(pred)
        if not mapped:
            upper = pred.upper()
            if upper in CANONICAL_PREDICATES:
                mapped = upper
        if mapped:
            t["predicate"] = mapped
            normalized.append(t)
        else:
            dropped += 1
    if dropped:
        print(f"  [Predicate Norm] Dropped {dropped} triples with unrecognized predicates.")
    return normalized


def _canonicalize_entities(triples: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """使用 LLM 识别实体别名，将所有变体映射到规范实体名。
    返回 (更新后的triples, {variant: canonical} 映射表)"""
    if not triples:
        return triples, {}

    # 收集所有唯一的实体名（主语和宾语）
    entities: set[str] = set()
    for t in triples:
        entities.add(t["subject"])
        entities.add(t["object"])

    # 过滤掉明显不是人物的实体（纯数字、过短等）
    candidates = sorted(e for e in entities if len(e) >= 2)

    if len(candidates) <= 1:
        return triples, {}

    entities_text = "\n".join(f"- {e}" for e in candidates)
    prompt = f"""Here are entity names extracted from a book. Group names that refer to the SAME entity.
For each group, pick ONE canonical name (the most complete/formal version).

CRITICAL RULES:
- Surname-only references MUST be merged with the full name if unambiguous:
  "Boldwood" + "William Boldwood" → canonical: "William Boldwood"
  "Troy" + "Francis Troy" → canonical: "Francis Troy"
  ONLY if there are no other characters sharing the surname.
- Title+variant: "Mr. Boldwood" + "William Boldwood" → canonical: "William Boldwood"
- First-name-only: "Gabriel" + "Gabriel Oak" → canonical: "Gabriel Oak"
- Nickname to full: "Cainy Ball" + "Cain Ball" → pick the most consistent form
- Place aliases: "Weatherbury" + "Little Weatherbury" → canonical: "Weatherbury"
- DIFFERENT people MUST stay separate. If unsure, keep them separate.

DO NOT GROUP these as entity names — they are EVENTS or DESCRIPTIONS, not entity variants:
  "Fanny Robin's funeral arrangements" — event, NOT an alias for Fanny Robin
  "shooting of Francis Troy" — event, NOT an alias for Francis Troy
  "Valentine Letter Prank" — event, NOT an alias for any character
  "The Storm" — event, NOT an alias
  Anything with "'s" (possessive) — NOT a name
  Anything with "of <name>" — descriptive phrase, NOT a name
  Sentence-length phrases starting with verbs — NOT names

Entity names:
{entities_text}

Output format — one group per block:
[GROUP]
canonical_name
alias1
alias2
[/GROUP]

For entities with no aliases, list them alone:
[GROUP]
canonical_name
[/GROUP]"""

    msgs = [
        {"role": "system", "content": "You are an expert at entity resolution for literary texts. "
         "Group names referring to the same person/place/organization. "
         "Surname-only references (e.g., 'Boldwood') map to the full name (e.g., 'William Boldwood') "
         "UNLESS multiple characters share that surname. "
         "DO NOT group event descriptions, action phrases, or possessive forms as entity names. "
         "Only real names, nicknames, title variants, and place-name variants should be grouped. "
         "Output ONLY the [GROUP] blocks, no other text."},
        {"role": "user", "content": prompt},
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

    # 应用映射
    alias_triples: list[dict] = []
    for t in triples:
        old_subj = t["subject"]
        old_obj = t["object"]
        new_subj = mapping.get(old_subj, old_subj)
        new_obj = mapping.get(old_obj, old_obj)
        t["subject"] = new_subj
        t["object"] = new_obj
        # 如果 subject 被映射了，生成 ALIAS 三元组
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

    # 添加 ALIAS 三元组并去重
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
        # 去重 evidence 并排序
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


# ALIAS 对象中不应出现的事件/描述关键词
_EVENT_DESC_WORDS = {
    "funeral", "shooting", "burial", "wedding", "marriage", "elopement",
    "arrangements", "transport", "procession", "search", "pursuit",
    "conversation", "discussion", "gossip", "dispute", "meeting",
    "performance", "celebration", "gathering", "supper", "dance", "feast",
    "journey", "removal", "farewell", "departure", "arrival",
    "identification", "investigation", "trial", "inquest", "sentencing",
    "death", "fate", "aftermath", "grave", "coffin", "laying", "laying out",
    "watch", "clothes", "marker", "gift", "letter", "note",
    "fire", "storm", "rescue", "fight", "swimming", "drowning",
    "service", "prayer", "church", "christmas",
    "race", "fair", "market",
    "preparations", "planting", "covering", "thatching", "protecting",
    "theft", "robbery", "disappearance", "escape", "release",
    "courtship", "courting", "proposal", "engagement",
    "coronation", "inauguration", "election",
    "rebellion", "war", "battle", "siege", "invasion",
}


def _quality_filter(triples: list[dict]) -> list[dict]:
    """过滤低质量三元组：自引用、ALIAS事件描述、实体误用为predicate object等"""
    filtered = []
    stats: dict[str, int] = {}

    def _inc(key: str):
        stats[key] = stats.get(key, 0) + 1

    # 收集各类信息
    attr_objects: set[str] = set()
    role_objects: set[str] = set()
    entity_subjects: set[str] = set()  # 所有作为 subject 出现过的实体名（用于判断 object 是否是人物）

    for t in triples:
        entity_subjects.add(t["subject"])
        if t["predicate"] == "HAS_ATTRIBUTE":
            attr_objects.add(t["object"])
        if t["predicate"] == "HAS_ROLE":
            role_objects.add(t["object"])

    def _is_bad_alias(obj: str, subj: str) -> bool:
        """检查ALIAS的object是否像事件描述而非真正的名字"""
        # 包含所有格 's — 如 "Fanny Robin's funeral arrangements"
        if "'s" in obj or "'s" in obj:
            return True
        # 超过5个单词 — 不太可能是名字
        if len(obj.split()) > 5:
            return True
        # 包含 "of <subject>" 模式 — 如 "shooting of Francis Troy"
        if f"of {subj}" in obj.lower():
            return True
        # 包含事件/描述关键词
        obj_lower = obj.lower()
        for w in _EVENT_DESC_WORDS:
            if w in obj_lower:
                return True
        # 以动词-ing开头 — 如 "Watching Eleventh...", "March to..."
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

        # 2. ALIAS 的 subject 是属性值或角色值（不是真正实体）
        if pred == "ALIAS":
            if subj in attr_objects:
                _inc("alias_attr_subj")
                continue
            if subj in role_objects:
                _inc("alias_role_subj")
                continue
            # ALIAS object 是事件描述而非名字
            if _is_bad_alias(obj, subj):
                _inc("alias_event_desc")
                continue

        # 3. PARTICIPATES_IN 的 object 是已知实体（人物/地点），不是事件
        if pred == "PARTICIPATES_IN":
            if obj in entity_subjects:
                _inc("participates_entity")
                continue
            # object 包含所有格 's — 如 "Fanny Robin's funeral"
            if "'s" in obj:
                _inc("participates_possessive")
                continue

        # 4. HAS_ATTRIBUTE 的 object 过长（超过5个单词）或包含所有格
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

        # 5. HAS_ROLE 的 object 是已知实体（人物名），不是角色
        if pred == "HAS_ROLE":
            if obj in entity_subjects:
                _inc("role_is_entity")
                continue

        # 6. OWNS 的 object 是已知实体（人物名），不是财产
        if pred == "OWNS":
            if obj in entity_subjects:
                _inc("owns_person")
                continue

        # 7. LOVES/MARRIES/REJECTS/PROPOSES_TO — object 不应是 "nobody", "soldier" 等非实体
        if pred in ("LOVES", "MARRIES", "REJECTS", "PROPOSES_TO", "RIVALS"):
            if obj.lower() in ("nobody", "someone", "soldier", "anyone"):
                _inc("relation_vague_obj")
                continue

        filtered.append(t)

    # 汇总日志
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
    """三元组后处理管线：谓词统一 → 实体对齐 → 质量过滤 → 去重"""
    if not triples:
        return triples

    print(f"\n  Post-processing {len(triples)} raw triples...")

    # Step 1: 谓词统一
    triples = _normalize_predicates(triples)

    # Step 2: 实体对齐 (LLM)
    triples, _ = _canonicalize_entities(triples)

    # Step 3: 质量过滤（自引用、属性实体等）
    triples = _quality_filter(triples)

    # Step 4: 去重 + 合并 evidence
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

def process_books(selected_names: list[str], question: str,
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

    # 最终回答
    print(f"\n  All {len(summaries)} windows summarized ({len(all_triples)} triples after post-processing).")
    print(f"  Generating final answer...\n")

    summaries_text = "\n\n".join(
        f"[Section {i+1}]\n{s}" for i, s in enumerate(summaries)
    )
    answer_messages = [
        {"role": "system", "content": (
            "You are given section summaries of one or more books. "
            "Answer based on ALL summaries. Cite sections as evidence.")},
        {"role": "user", "content": f"Summaries:\n\n{summaries_text}\n\nQuestion: {question}"},
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
    books = load_books(DATA_DIR)
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

    books = load_books(DATA_DIR)
    print(f"\n{len(books)} books available.")
    print_books_table()

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

    question = input("\nQuestion: ").strip()
    if not question:
        print("Question required.")
        return

    proceed = input(f"Proceed? [y/N]: ").strip().lower()
    if proceed != "y":
        print("Cancelled.")
        return

    resume_path = os.path.join(os.path.dirname(__file__), "output", ".progress.json")
    result = process_books(selected, question, resume_path=resume_path)

    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}\n")
    print(result["answer"])
    print(f"\n{'='*60}")
    print(f"Time: {result['total_time']:.0f}s | Windows: {result['total_windows']} | Triples: {result.get('triples_count', 0)}")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"result_{ts}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Q: {question}\n\nA:\n{result['answer']}\n\n")
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
