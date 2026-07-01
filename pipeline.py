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
API_BASE = "http://162.105.19.243:11451/v1"
API_KEY = "sulab"
MODEL = "Qwen3.6-27B"
DATA_DIR = os.path.join(os.path.dirname(__file__), "books", "train")

# --- 滑动窗口 ---
WINDOW_SIZE = 4000
OVERLAP = 1000
STRIDE = WINDOW_SIZE - OVERLAP

# --- 并行 ---
WORKERS = 3            # 并发请求数 (27B 模型建议 2-3)
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
        extra_body={"enable_thinking": False},
    )
    return resp.choices[0].message.content or ""


def summarize_one(window_text: str, book_name: str,
                  idx: int, total: int) -> str:
    """总结单个窗口（线程安全，可被并行调用）"""
    msgs = [
        {"role": "system", "content": (
            "Summarize the book section below in 2-3 sentences. "
            "Include key events, characters, and details.")},
        {"role": "user", "content": f"[{book_name} | {idx}/{total}]\n{window_text}"},
    ]
    for attempt in range(MAX_RETRIES + 1):
        r = _call_model(msgs, MAX_TOKENS_SUMMARIZE)
        if r.strip():
            return r.strip()
        if attempt < MAX_RETRIES:
            msgs[0]["content"] += " Be thorough."
            r = _call_model(msgs, MAX_TOKENS_SUMMARIZE * 2)
            if r.strip():
                return r.strip()
    return "[summary unavailable]"


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
                           resume_path: str | None = None) -> list[tuple[int, str]]:
    """
    tasks: [{global_idx, book_name, text, section_idx, section_total}]
    返回 [(global_idx, summary)]，已按 global_idx 排序。

    并行执行，支持进度保存/恢复。
    """
    total = len(tasks)
    # 恢复已完成的
    done = set()        # global_idx
    results: dict[int, str] = {}
    if resume_path and os.path.exists(resume_path):
        saved = json.load(open(resume_path, "r"))
        done = set(saved.get("done", []))
        results = {int(k): v for k, v in saved.get("results", {}).items()}
        print(f"  Resumed: {len(done)}/{total} windows already done.")

    # 筛选未完成的
    pending = [t for t in tasks if t["global_idx"] not in done]
    lock = threading.Lock()

    def process(t: dict) -> tuple[int, str]:
        idx = t["global_idx"]
        summary = summarize_one(t["text"], t["book_name"],
                                t["section_idx"], t["section_total"])
        with lock:
            done.add(idx)
            results[idx] = summary
            # 每完成 10 个保存一次
            if resume_path and len(done) % 10 == 0:
                _save_resume(resume_path, done, results)
        return idx, summary

    if pending:
        print(f"  Processing {len(pending)} windows ({workers} workers)...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process, t): t for t in pending}
            for f in as_completed(futures):
                idx, _ = f.result()
                print(f"  [{idx}/{total}] done", flush=True)

    # 最终保存
    if resume_path:
        _save_resume(resume_path, done, results)

    # 按 global_idx 排序返回
    return sorted(results.items())


def _save_resume(path: str, done: set, results: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 只序列化 int key
    serializable = {str(k): v for k, v in results.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"done": sorted(done), "results": serializable},
                  f, ensure_ascii=False)


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
    sorted_results = summarize_all_parallel(all_tasks, WORKERS, resume_path)

    summaries = [s for _, s in sorted_results]

    # 最终回答
    print(f"\n  All {len(summaries)} windows summarized.")
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

    # 清理进度
    if resume_path and os.path.exists(resume_path):
        os.remove(resume_path)

    return {
        "answer": answer,
        "total_windows": total_windows,
        "total_time": time.time() - t_start,
        "answer_time": answer_time,
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
    print(f"Time: {result['total_time']:.0f}s | Windows: {result['total_windows']}")

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
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    interactive()
