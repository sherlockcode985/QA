"""
长文本理解管线 — 滑动窗口 + 并行分段总结 + 汇总回答

流程:
  1. 滑动窗口切分文本 (sentence-aware)
  2. 并行总结所有窗口 (ThreadPoolExecutor, 按 index 排序拼合)
  3. 汇总所有总结, 基于总结回答问题
"""

import os, re, time, json, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ============ 配置 ============
# 在服务器上 export API_KEY=your_key_here 或直接修改下方
API_BASE = "https://api.v3.cm/v1"
API_KEY = os.environ.get("API_KEY", "sk-7UYrjDTvNGkCiSof5bAb604870C1401b88Ac44FfF4C569Cc")
MODEL = "claude-sonnet-5"
DATA_DIR = os.path.join(os.path.dirname(__file__), "books", "train")

# --- 三元组提取配置 ---
# 在此定义需要提取的三元组类型，模型会根据此提示词在总结时同步提取
TRIPLE_INSTRUCTION = """从文本中提取以下三元组（格式：主体||属性值||关系类型，每行一个）：
- 人物外号/别名: 如 张三||狗蛋||外号
- 人物身份/职位: 如 李四||掌门||身份
- 人物关系: 如 张三||李四||师徒
- 重要物品归属: 如 屠龙刀||张三||持有"""

# --- 滑动窗口 ---
WINDOW_SIZE = 4000
OVERLAP = 1000
STRIDE = WINDOW_SIZE - OVERLAP

# --- 并行 ---
WORKERS = 6            # 并发请求数
MAX_TOKENS_SUMMARIZE = 2048
MAX_TOKENS_ANSWER = 8192
MAX_RETRIES = 2

client = OpenAI(base_url=API_BASE, api_key=API_KEY)


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
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def _parse_response(text: str) -> tuple[str, list[tuple[str, str, str]]]:
    """从模型回复中解析 [SUMMARY]...[/SUMMARY] 和 [TRIPLES]...[/TRIPLES] 块"""
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


def summarize_one(window_text: str, book_name: str,
                  idx: int, total: int) -> tuple[str, list[tuple[str, str, str]]]:
    """总结单个窗口并同步提取三元组（线程安全，可被并行调用）"""
    system_prompt = (
        "You are a literary analyst. For the given book section:\n"
        "1. Write a 2-3 sentence summary including key events, characters, and details.\n"
        "2. Extract structured triples (entity||attribute||relation_type) as instructed below.\n"
        "IMPORTANT: Only include triples that are explicitly supported by the text.\n\n"
        "Output format:\n"
        "[SUMMARY]\n<your summary here>\n[/SUMMARY]\n"
        "[TRIPLES]\nentity||attribute||relation\nentity||attribute||relation\n[/TRIPLES]\n\n"
        f"Triple extraction instructions:\n{TRIPLE_INSTRUCTION}"
    )
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"[{book_name} | {idx}/{total}]\n{window_text}"},
    ]
    for attempt in range(MAX_RETRIES + 1):
        r = _call_model(msgs, MAX_TOKENS_SUMMARIZE)
        summary, triples = _parse_response(r)
        if summary:
            return summary, triples
        if attempt < MAX_RETRIES:
            r = _call_model(msgs, MAX_TOKENS_SUMMARIZE * 2)
            summary, triples = _parse_response(r)
            if summary:
                return summary, triples
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

    并行执行，支持进度保存/恢复。
    """
    total = len(tasks)
    done: set[int] = set()
    results: dict[int, str] = {}
    all_triples: list[dict] = []

    if resume_path and os.path.exists(resume_path):
        saved = json.load(open(resume_path, "r"))
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
            for entity, attr, rel in triples:
                all_triples.append({
                    "entity": entity, "attribute": attr, "relation": rel,
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


def _save_triples_csv(triples: list[dict]) -> str:
    """去重并保存三元组到 CSV 文件"""
    import csv

    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"triples_{ts}.csv")

    # 去重（entity + attribute + relation 相同视为重复）
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict] = []
    for t in triples:
        key = (t["entity"], t["attribute"], t["relation"])
        if key not in seen:
            seen.add(key)
            unique.append(t)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["entity", "attribute", "relation", "book", "window"])
        writer.writeheader()
        writer.writerows(unique)

    print(f"  Triples: {len(unique)} unique (from {len(triples)} raw)")
    return csv_path


# ============ 主流程 ============

def process_books(selected_names: list[str], question: str,
                  resume_path: str | None = None) -> dict:
    t_start = time.time()
    all_tasks = []
    total_windows = 0

    # 构建所有窗口任务（分配全局编号）
    for name in selected_names:
        content = get_book_content(name)
        windows = sliding_window_chunks(content)
        section_total = len(windows)
        for i, w in enumerate(windows, 1):
            all_tasks.append({
                "global_idx": total_windows + i,  # 1-based
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

    # 最终回答
    print(f"\n  All {len(summaries)} windows summarized ({len(all_triples)} triples extracted).")
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

    # 去重三元组并输出 CSV
    csv_path = None
    if all_triples:
        csv_path = _save_triples_csv(all_triples)
        print(f"  Triples saved: {csv_path}")

    # 清理进度
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
