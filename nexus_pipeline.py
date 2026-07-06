"""
NexusSum-inspired Hierarchical Summarization Pipeline
Based on ACL 2025 paper: "NexusSum: Hierarchical LLM Agents for
Long-Form Narrative Summarization"

4-stage pipeline:
  1. Preprocessor  — dialogue-to-description + pronoun resolution (parallel)
  2. Summarizer    — variable-length chunk summaries (parallel)
  3. Compressor    — iterative hierarchical compression + triple extraction
  4. Polisher      — semantic coherence polish + QA
"""

import os, re, time, json, threading, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ============ 配置 ============
API_BASE = "http://162.105.19.243:11451/v1"
API_KEY = os.environ.get("API_KEY", "sulab")
MODEL = "Qwen3.6-27B"
DATA_DIR = os.path.join(os.path.dirname(__file__), "books", "train")

# --- 滑动窗口 ---
WINDOW_SIZE = 4000
OVERLAP = 1000
STRIDE = WINDOW_SIZE - OVERLAP

# --- 并行 ---
WORKERS = 8
MAX_RETRIES = 2

# --- Token 限制 ---
MAX_TOKENS_PREPROCESS = 4096
MAX_TOKENS_SUMMARIZE = 2048
MAX_TOKENS_COMPRESS = 2048
MAX_TOKENS_TRIPLES = 4096
MAX_TOKENS_POLISH = 4096
MAX_TOKENS_ANSWER = 8192

# --- 压缩参数 (Stage 3) ---
TARGET_CHARS = 4000       # θ: target character count
COMPRESS_CHUNK = 800      # δ: sentence-group chunk size
MAX_COMPRESS_ITER = 10     # max compression iterations

# --- 规范谓词集合 ---
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

PREDICATE_NORMALIZATION = {
    "外号": "ALIAS", "别名": "ALIAS", "绰号": "ALIAS", "人称": "ALIAS", "又称": "ALIAS",
    "身份": "HAS_ROLE", "职位": "HAS_ROLE", "职业": "HAS_ROLE", "是": "HAS_ROLE",
    "担任": "HAS_ROLE", "成为": "HAS_ROLE", "角色": "HAS_ROLE",
    "特点": "HAS_ATTRIBUTE", "性格": "HAS_ATTRIBUTE",
    "爱慕": "LOVES", "喜欢": "LOVES", "爱": "LOVES", "暗恋": "LOVES",
    "爱慕对象": "LOVES", "被追求": "LOVES", "喜欢的人": "LOVES", "心上人": "LOVES",
    "追求": "PROPOSES_TO", "求婚": "PROPOSES_TO", "求爱": "PROPOSES_TO",
    "拒绝": "REJECTS", "拒绝求婚": "REJECTS", "拒绝求爱": "REJECTS",
    "夫妻": "MARRIES", "结婚": "MARRIES", "嫁给": "MARRIES",
    "娶": "MARRIES", "丈夫": "MARRIES", "妻子": "MARRIES",
    "未婚妻": "PROPOSES_TO", "未婚夫": "PROPOSES_TO",
    "师徒": "HAS_FAMILY_RELATION", "父亲": "PARENT_OF", "母亲": "PARENT_OF",
    "儿子": "CHILD_OF", "女儿": "CHILD_OF",
    "兄弟": "SIBLING_OF", "姐妹": "SIBLING_OF",
    "叔": "UNCLE_OF", "伯": "UNCLE_OF", "舅": "UNCLE_OF",
    "姨": "AUNT_OF", "姑": "AUNT_OF",
    "侄": "HAS_FAMILY_RELATION", "亲属": "HAS_FAMILY_RELATION",
    "亲戚": "HAS_FAMILY_RELATION", "家人": "HAS_FAMILY_RELATION",
    "持有": "OWNS", "拥有": "OWNS", "所属": "OWNS", "拥有者": "OWNS",
    "主人": "OWNS", "归属": "OWNS",
    "雇主": "WORKS_FOR", "仆人": "WORKS_FOR", "雇佣": "WORKS_FOR",
    "员工": "WORKS_FOR", "打工": "WORKS_FOR", "服务": "WORKS_FOR",
    "雇佣者": "EMPLOYS",
    "位于": "LOCATED_IN", "住在": "LIVES_IN", "来自": "LOCATED_IN",
    "出生地": "LOCATED_IN", "在": "LOCATED_IN",
    "认识": "KNOWS", "相识": "KNOWS", "知道": "KNOWS",
    "朋友": "FRIEND_OF", "好友": "FRIEND_OF",
    "敌人": "RIVALS", "仇人": "RIVALS", "对手": "RIVALS",
    "杀死": "KILLS", "杀害": "KILLS", "谋杀": "KILLS",
    "成员": "MEMBER_OF", "加入": "MEMBER_OF",
    "教导": "MENTOR_OF", "老师": "MENTOR_OF", "指导": "MENTOR_OF",
    "师傅": "MENTOR_OF",
    "参加": "PARTICIPATES_IN", "参与": "PARTICIPATES_IN",
    "拜访": "VISITS", "访问": "VISITS",
}

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


def _call_with_retry(msgs: list, max_tokens: int,
                     validate=None) -> str:
    """带重试和可选验证的模型调用"""
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = _call_model(msgs, max_tokens)
            if r.strip():
                if validate is None or validate(r):
                    return r.strip()
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(2 * (attempt + 1))  # backoff: 2s, 4s
                continue
    if last_err:
        print(f"  [Warn] API call failed after {MAX_RETRIES + 1} attempts: {last_err}")
    return ""


# ============ Stage 1: Preprocessor ============

PREPROCESS_SYSTEM = """You are a narrative preprocessing agent. Transform the given text into unified third-person narrative prose.

RULES:
1. Replace pronouns (he, she, they, him, her, them, his, hers, their) with the actual character names when the referent is clear from context. Keep the pronoun if the referent is ambiguous.
2. Transform dialogue into descriptive third-person narration. Instead of:
     "I cannot go," she said.
   Write:
     Elizabeth firmly declared she could not go.
   Add emotional/tonal context to dialogue attributions (angrily, quietly, with hesitation, etc.) when evident from context.
3. Preserve ALL factual information, events, settings, and details exactly.
4. Do NOT add new information, interpretation, or commentary.
5. Maintain the original language. Do NOT translate.

Output ONLY the transformed text, no explanations."""


def preprocess_chunk(window_text: str, book_name: str,
                     idx: int, total: int) -> str:
    msgs = [
        {"role": "system", "content": PREPROCESS_SYSTEM},
        {"role": "user", "content": f"[{book_name} | {idx}/{total}]\n{window_text}"},
    ]
    result = _call_with_retry(msgs, MAX_TOKENS_PREPROCESS)
    return result if result else window_text


# ============ Stage 2: Summarizer ============

SUMMARIZE_SYSTEM = """You are a narrative summarization agent. Summarize the book section below.

IMPORTANT — adjust your summary length based on information density:
- Sections with many key events, character developments, or plot twists: write a detailed summary.
- Sections with mainly descriptive passages, transitions, or minor details: write a brief summary.
- A summary can be anywhere from 2 sentences to a full paragraph depending on content richness.

Include: key events, character actions and motivations, important dialogue content, and plot-relevant details.
Output ONLY the summary, no explanations or headers."""


def summarize_chunk(window_text: str, book_name: str,
                    idx: int, total: int) -> str:
    msgs = [
        {"role": "system", "content": SUMMARIZE_SYSTEM},
        {"role": "user", "content": f"[{book_name} | section {idx}/{total}]\n{window_text}"},
    ]
    result = _call_with_retry(msgs, MAX_TOKENS_SUMMARIZE)
    return result if result else f"[Section {idx} summary unavailable]"


# ============ Stage 3: Compressor ============

COMPRESS_SYSTEM = """You are a text compression agent. Compress the following text by:
1. Removing redundant or repetitive sentences.
2. Merging similar information into concise statements.
3. Removing filler and low-value descriptive passages.
4. Preserving ALL key facts, events, character names, relationships, and plot points.

Output ONLY the compressed text, no explanations or headers."""


# Abbreviation patterns to avoid splitting on (Mr. Dr. etc.)
_ABBREV_PAT = re.compile(
    r'\b(?:Mr|Mrs|Ms|Dr|Prof|Rev|Hon|St|Ave|Esq|Jr|Sr|Capt|Col|Gen|Lt|Maj|Sgt|Dept|Govt)\.$'
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting abbreviations."""
    sentences = []
    current = ""
    i = 0
    while i < len(text):
        ch = text[i]
        current += ch
        if ch in '.!?':
            # Check if this period is part of an abbreviation
            before = current.rstrip()[:-1]
            if ch == '.' and _ABBREV_PAT.search(before):
                i += 1
                continue
            # Look ahead: sentence end if followed by space+capital or end
            rest = text[i+1:]
            if not rest.strip() or (
                rest.lstrip() and rest.lstrip()[0].isupper() and ' ' in rest[:2]
            ):
                sentences.append(current.strip())
                current = ""
        i += 1
    if current.strip():
        sentences.append(current.strip())
    # Split long sentences on double newlines
    result = []
    for s in sentences:
        if len(s) > COMPRESS_CHUNK * 2 and '\n\n' in s:
            result.extend(p.strip() for p in s.split('\n\n') if p.strip())
        else:
            result.append(s)
    return [s for s in result if s]


def _group_sentences(sentences: list[str], chunk_size: int) -> list[str]:
    """Group sentences into chunks of approximately chunk_size characters."""
    chunks = []
    current = ""
    for s in sentences:
        if current and len(current) + len(s) > chunk_size:
            chunks.append(current.strip())
            current = s
        else:
            current += " " + s if current else s
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _compress_single(chunk_text: str, idx: int, total: int) -> tuple[int, str]:
    """Compress a single sentence-group chunk (thread-safe)."""
    msgs = [
        {"role": "system", "content": COMPRESS_SYSTEM},
        {"role": "user", "content": f"[Chunk {idx}/{total}]\n{chunk_text}"},
    ]
    result = _call_with_retry(msgs, MAX_TOKENS_COMPRESS)
    return idx, result if result else chunk_text


def compress_iteratively(text: str, target_chars: int = TARGET_CHARS,
                         chunk_size: int = COMPRESS_CHUNK,
                         max_iter: int = MAX_COMPRESS_ITER,
                         workers: int = WORKERS) -> str:
    """
    Iterative hierarchical compression (NexusSum algorithm):
    1. Split text into sentences
    2. Group into chunks of ~chunk_size chars
    3. Compress each chunk in parallel
    4. Merge and repeat if len > target_chars and iter < max_iter
    """
    current = text
    for iteration in range(1, max_iter + 1):
        prev_len = len(current)
        if prev_len <= target_chars:
            print(f"  Compression: already at target ({prev_len} ≤ {target_chars} chars).")
            break

        sentences = _split_sentences(current)
        chunks = _group_sentences(sentences, chunk_size)

        if len(chunks) <= 1 and prev_len <= target_chars * 1.5:
            break

        print(f"  Compression iter {iteration}: {prev_len} chars → "
              f"{len(chunks)} sentence-chunks, target={target_chars}")

        results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_compress_single, c, i, len(chunks)): i
                       for i, c in enumerate(chunks, 1)}
            for f in as_completed(futures):
                idx, compressed = f.result()
                results[idx] = compressed

        current = " ".join(results[i] for i in sorted(results))
        new_len = len(current)
        ratio = (1 - new_len / prev_len) * 100 if prev_len else 0
        print(f"    → {new_len} chars ({ratio:.0f}% reduction)")

        if new_len <= target_chars:
            break

    return current


# ============ Stage 3b: Triple Extraction ============

TRIPLE_SYSTEM = """You are a literary knowledge graph analyst. Extract character-level knowledge graph triples from the book summary below.

FORMAT: subject||predicate||object

Use ONLY these canonical predicates:
ALIAS, HAS_ROLE, HAS_ATTRIBUTE, LOVES, MARRIES, PROPOSES_TO, REJECTS,
KNOWS, FRIEND_OF, RIVALS, WORKS_FOR, EMPLOYS, LIVES_IN, OWNS,
HAS_FAMILY_RELATION, PARENT_OF, CHILD_OF, SIBLING_OF, AUNT_OF, UNCLE_OF,
LOCATED_IN, VISITS, PARTICIPATES_IN

=== RULES ===
SUBJECT: Only proper named entities (specific people, named places, named organizations).
  Use the MOST COMPLETE name: "William Boldwood" not "Boldwood".
  NEVER use descriptions ("the young girl", "a soldier") as subjects.

OBJECT by predicate:
  HAS_ROLE: GENERIC role only (Farmer, Shepherd, Servant, Soldier, Maid). NEVER a person name.
  HAS_ATTRIBUTE: a trait in 1-4 words (brave, wealthy, tall, "28 years old"). NEVER a person/place.
  ALIAS: alternate name/nickname for the subject.
  OWNS: significant named possessions only (farm, house, horse). NEVER a person.
  KNOWS: established acquaintances. Do NOT create symmetric pairs (A KNOWS B + B KNOWS A).
  PARTICIPATES_IN: NAMED events only. NEVER a person or place as event.
  LIVES_IN: primary residence only, not temporary stays.

FORBIDDEN:
  X||HAS_ROLE||<person name>
  X||HAS_ATTRIBUTE||<place/organization/person>
  <description>||ANY||ANY
  X||KNOWS||Y AND Y||KNOWS||X (symmetric, only one direction)
  X||PARTICIPATES_IN||<person or place>
  X||OWNS||<person>

Extract 20-40 high-quality triples. Output ONLY triples, one per line:
subject||predicate||object"""


def extract_triples(compressed_summary: str) -> list[dict]:
    """Extract KG triples from the compressed summary (has global context)."""
    msgs = [
        {"role": "system", "content": TRIPLE_SYSTEM},
        {"role": "user", "content": compressed_summary},
    ]
    response = _call_with_retry(msgs, MAX_TOKENS_TRIPLES)
    if not response:
        print("  [Triples] Extraction returned empty response.")
        return []

    triples: list[dict] = []
    for line in response.strip().split('\n'):
        line = line.strip().lstrip('- ').strip()
        if not line or '||' not in line:
            continue
        parts = [p.strip() for p in line.split('||')]
        if len(parts) >= 3:
            triples.append({
                "subject": parts[0], "predicate": parts[1], "object": parts[2],
            })

    print(f"  [Triples] Extracted {len(triples)} raw triples.")
    return triples


# ============ Triple Post-Processing ============

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


def _normalize_predicates(triples: list[dict]) -> list[dict]:
    normalized = []
    dropped = 0
    for t in triples:
        pred = t["predicate"].strip()
        if pred in CANONICAL_PREDICATES:
            normalized.append(t)
            continue
        mapped = PREDICATE_NORMALIZATION.get(pred)
        if not mapped:
            upper = pred.upper()
            if upper in CANONICAL_PREDICATES:
                mapped = upper
        if mapped and mapped in CANONICAL_PREDICATES:
            t["predicate"] = mapped
            normalized.append(t)
        else:
            dropped += 1
    if dropped:
        print(f"  [Predicate Norm] Dropped {dropped} triples with unrecognized predicates.")
    return normalized


def _quality_filter(triples: list[dict]) -> list[dict]:
    filtered = []
    stats: dict[str, int] = {}

    def _inc(key: str):
        stats[key] = stats.get(key, 0) + 1

    entity_subjects: set[str] = {t["subject"] for t in triples}
    attr_objects: set[str] = {t["object"] for t in triples if t["predicate"] == "HAS_ATTRIBUTE"}
    role_objects: set[str] = {t["object"] for t in triples if t["predicate"] == "HAS_ROLE"}

    def _is_bad_alias(obj: str, subj: str) -> bool:
        if "'s" in obj:
            return True
        if len(obj.split()) > 5:
            return True
        if f"of {subj}" in obj.lower():
            return True
        obj_lower = obj.lower()
        if any(w in obj_lower for w in _EVENT_DESC_WORDS):
            return True
        first_word = obj.split()[0].lower() if obj.split() else ""
        if first_word.endswith("ing") and first_word not in ("nothing", "something", "everything"):
            return True
        return False

    for t in triples:
        subj, pred, obj = t["subject"], t["predicate"], t["object"]

        if subj == obj:
            _inc("self_ref")
            continue

        if pred == "ALIAS":
            if subj in attr_objects:
                _inc("alias_attr_subj"); continue
            if subj in role_objects:
                _inc("alias_role_subj"); continue
            if _is_bad_alias(obj, subj):
                _inc("alias_event_desc"); continue

        if pred == "PARTICIPATES_IN":
            if obj in entity_subjects:
                _inc("participates_entity"); continue
            if "'s" in obj:
                _inc("participates_possessive"); continue

        if pred == "HAS_ATTRIBUTE":
            if len(obj.split()) > 5:
                _inc("attr_too_long"); continue
            if obj in entity_subjects:
                _inc("attr_is_entity"); continue
            if "'s" in obj:
                _inc("attr_possessive"); continue

        if pred == "HAS_ROLE":
            if obj in entity_subjects:
                _inc("role_is_entity"); continue

        if pred == "OWNS":
            if obj in entity_subjects:
                _inc("owns_person"); continue

        if pred in ("LOVES", "MARRIES", "REJECTS", "PROPOSES_TO", "RIVALS"):
            if obj.lower() in ("nobody", "someone", "soldier", "anyone"):
                _inc("relation_vague_obj"); continue

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


def _deduplicate_triples(triples: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for t in triples:
        key = (t["subject"], t["predicate"], t["object"])
        if key not in groups:
            groups[key] = []
        groups[key].append(t)

    unique: list[dict] = []
    for key, items in groups.items():
        first = items[0]
        unique.append({
            "subject": key[0],
            "predicate": key[1],
            "object": key[2],
        })

    print(f"  [Dedup] {len(triples)} raw → {len(unique)} unique triples.")
    return unique


def post_process_triples(triples: list[dict]) -> list[dict]:
    """谓词归一化 → 质量过滤 → 去重"""
    if not triples:
        return triples
    print(f"\n  Post-processing {len(triples)} raw triples...")
    triples = _normalize_predicates(triples)
    triples = _quality_filter(triples)
    triples = _deduplicate_triples(triples)
    return triples


def _save_triples_csv(triples: list[dict]) -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"triples_{ts}.csv")
    fieldnames = ["subject", "predicate", "object"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(triples)
    print(f"  Triples saved: {csv_path} ({len(triples)} rows)")
    return csv_path


# ============ Stage 4: Polisher ============

POLISH_SYSTEM = """You are a text polishing agent. Improve the semantic coherence and readability of the following summary.

CRITICAL RULES:
1. Do NOT add any new information, facts, or details.
2. Do NOT remove any facts, events, character names, or plot points.
3. Do NOT change the meaning of any sentence.
4. Do NOT significantly shorten the text — keep approximately the same length.
5. ONLY improve: transitions between sentences, logical flow, paragraph structure, grammar, and readability.

Output ONLY the polished text, no explanations or headers."""


def polish_summary(compressed_text: str) -> str:
    msgs = [
        {"role": "system", "content": POLISH_SYSTEM},
        {"role": "user", "content": compressed_text},
    ]
    result = _call_with_retry(msgs, MAX_TOKENS_POLISH)
    return result if result else compressed_text


# ============ QA ============

QA_SYSTEM = """You are given a polished book summary and knowledge graph triples.
Answer the user's question based on ALL provided information.
Cite specific sections or triples as evidence where relevant.
Be thorough and comprehensive."""


def answer_question(polished_summary: str, triples: list[dict],
                    question: str) -> str:
    triples_text = "\n".join(
        f"- {t['subject']} || {t['predicate']} || {t['object']}"
        for t in triples
    ) if triples else "(no triples extracted)"

    msgs = [
        {"role": "system", "content": QA_SYSTEM},
        {"role": "user", "content": (
            f"BOOK SUMMARY:\n\n{polished_summary}\n\n"
            f"KNOWLEDGE GRAPH TRIPLES:\n{triples_text}\n\n"
            f"QUESTION: {question}"
        )},
    ]
    result = _call_with_retry(msgs, MAX_TOKENS_ANSWER)
    return result if result else "Unable to generate answer."


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
                    "windows": len(windows),
                })
    return books


def get_book_content(name: str) -> str:
    path = os.path.join(DATA_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ============ Resume ============

def _save_resume(path: str, done: set, results: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {str(k): v for k, v in results.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"done": sorted(done), "results": serializable},
                  f, ensure_ascii=False)


# ============ 并行执行辅助 ============

def _run_parallel_stage(tasks: list[dict], stage_fn, stage_name: str,
                        workers: int = WORKERS,
                        resume_path: str | None = None) -> list[tuple[int, str]]:
    """
    通用并行执行框架。
    tasks: [{global_idx, book_name, text, section_idx, section_total}]
    stage_fn(text, book_name, idx, total) -> str
    返回 [(global_idx, result)]，按 global_idx 排序。
    """
    total = len(tasks)
    done: set[int] = set()
    results: dict[int, str] = {}

    if resume_path and os.path.exists(resume_path):
        saved = json.load(open(resume_path, "r", encoding="utf-8"))
        done = set(saved.get("done", []))
        results = {int(k): v for k, v in saved.get("results", {}).items()}
        print(f"  [{stage_name}] Resumed: {len(done)}/{total} already done.")

    pending = [t for t in tasks if t["global_idx"] not in done]
    lock = threading.Lock()

    def process(t: dict) -> tuple[int, str]:
        idx = t["global_idx"]
        result = stage_fn(t["text"], t["book_name"],
                         t["section_idx"], t["section_total"])
        with lock:
            done.add(idx)
            results[idx] = result
            if resume_path and len(done) % 10 == 0:
                _save_resume(resume_path, done, results)
        return idx, result

    if pending:
        print(f"  [{stage_name}] Processing {len(pending)} windows "
              f"({workers} workers)...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process, t): t for t in pending}
            for f in as_completed(futures):
                idx, _ = f.result()
                print(f"  [{stage_name}] [{idx}/{total}] done", flush=True)

    if resume_path:
        _save_resume(resume_path, done, results)

    return sorted(results.items())


# ============ 主流程 ============

def process_books(selected_names: list[str], question: str,
                  resume_dir: str | None = None) -> dict:
    t_start = time.time()
    all_tasks = []
    total_windows = 0

    # 构建所有窗口任务
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

    if resume_dir is None:
        resume_dir = os.path.join(os.path.dirname(__file__), "output")

    # ── Stage 1: Preprocess ──
    print(f"\n{'='*60}")
    print("STAGE 1: Preprocessing (dialogue → narrative)")
    print(f"{'='*60}")
    pp_resume = os.path.join(resume_dir, ".progress_stage1.json")
    pp_results = _run_parallel_stage(all_tasks, preprocess_chunk,
                                     "Preprocessor", WORKERS, pp_resume)

    # 更新 tasks 的 text 为预处理后的文本
    pp_map = dict(pp_results)
    for t in all_tasks:
        if t["global_idx"] in pp_map:
            t["text"] = pp_map[t["global_idx"]]

    if os.path.exists(pp_resume):
        os.remove(pp_resume)

    # ── Stage 2: Summarize ──
    print(f"\n{'='*60}")
    print("STAGE 2: Summarizing (variable-length, info-density aware)")
    print(f"{'='*60}")
    s2_resume = os.path.join(resume_dir, ".progress_stage2.json")
    s2_results = _run_parallel_stage(all_tasks, summarize_chunk,
                                     "Summarizer", WORKERS, s2_resume)

    s2_map = dict(s2_results)
    summaries = [s2_map[i] for i in sorted(s2_map)]

    if os.path.exists(s2_resume):
        os.remove(s2_resume)

    # ── Stage 3: Compress + Triples ──
    print(f"\n{'='*60}")
    print("STAGE 3: Compression + Triple Extraction")
    print(f"{'='*60}")

    summaries_text = "\n\n".join(
        f"[Section {i+1}] {s}" for i, s in enumerate(summaries)
    )
    print(f"  Combined summaries: {len(summaries_text)} chars from "
          f"{len(summaries)} sections")

    compressed = compress_iteratively(summaries_text)

    # Extract triples from compressed summary
    raw_triples = extract_triples(compressed)
    triples = post_process_triples(raw_triples)

    # ── Stage 4: Polish ──
    print(f"\n{'='*60}")
    print("STAGE 4: Polishing (semantic coherence)")
    print(f"{'='*60}")

    polished = polish_summary(compressed)

    # ── QA ──
    print(f"\n{'='*60}")
    print("QA: Answering question")
    print(f"{'='*60}")

    t0 = time.time()
    answer = answer_question(polished, triples, question)
    answer_time = time.time() - t0

    # ── Save outputs ──
    csv_path = _save_triples_csv(triples) if triples else None

    total_time = time.time() - t_start

    return {
        "answer": answer,
        "compressed_summary": compressed,
        "polished_summary": polished,
        "total_windows": total_windows,
        "total_time": total_time,
        "answer_time": answer_time,
        "triples_count": len(triples),
        "triples_csv": csv_path,
        "compressed_chars": len(compressed),
        "polished_chars": len(polished),
    }


# ============ 交互界面 ============

def print_books_table():
    books = load_books(DATA_DIR)
    print(f"\n  {'Idx':<5} {'Filename':<25} {'Size':<8} {'Chars':<8} {'Windows':<8}")
    print("  " + "-" * 58)
    for i, b in enumerate(books):
        print(f"  {i:<5} {b['name']:<25} {b['size_kb']:<8} KB "
              f"{b['chars']:<8,} {b['windows']:<8}")


def interactive():
    print("=" * 60)
    print("NexusSum-Inspired Hierarchical Summarization Pipeline")
    print(f"Model: {MODEL} | Workers: {WORKERS}")
    print(f"Window: {WINDOW_SIZE}ch | Compression target: {TARGET_CHARS}ch")
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
    # Stage 1 + Stage 2 = 2 parallel rounds per chunk
    est_calls = tw * 2 + MAX_COMPRESS_ITER * 3 + 3  # rough estimate
    est_parallel = est_calls // WORKERS
    print(f"\nSelected {len(selected)} book(s), ~{tw} windows.")
    print(f"Estimated LLM calls: ~{est_calls}, time: ~{est_parallel * 15 // 60} min")

    question = input("\nQuestion: ").strip()
    if not question:
        print("Question required.")
        return

    proceed = input(f"Proceed? [y/N]: ").strip().lower()
    if proceed != "y":
        print("Cancelled.")
        return

    result = process_books(selected, question)

    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}\n")
    print(result["answer"])
    print(f"\n{'='*60}")
    print(f"Total time: {result['total_time']:.0f}s | "
          f"Windows: {result['total_windows']} | "
          f"Triples: {result['triples_count']}")
    print(f"Compressed: {result['compressed_chars']} chars | "
          f"Polished: {result['polished_chars']} chars")

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"nexus_result_{ts}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Q: {question}\n\n")
        f.write(f"=== POLISHED SUMMARY ===\n{result['polished_summary']}\n\n")
        f.write(f"=== ANSWER ===\n{result['answer']}\n\n")
        f.write(f"--- Stats ---\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Workers: {WORKERS}\n")
        f.write(f"Windows: {result['total_windows']}\n")
        f.write(f"Triples: {result['triples_count']}\n")
        f.write(f"Compressed chars: {result['compressed_chars']}\n")
        f.write(f"Polished chars: {result['polished_chars']}\n")
        f.write(f"Total time: {result['total_time']:.0f}s\n")
        f.write(f"Answer time: {result['answer_time']:.0f}s\n")
        if result.get('triples_csv'):
            f.write(f"Triples CSV: {result['triples_csv']}\n")
    print(f"\nSaved: {out_path}")
    if result.get('triples_csv'):
        print(f"Triples CSV: {result['triples_csv']}")


if __name__ == "__main__":
    interactive()
