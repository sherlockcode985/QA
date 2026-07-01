"""
长文本理解管线 — 滑动窗口 + 分段总结 + 汇总回答

流程:
  1. 滑动窗口切分文本 (sentence-aware, 可配置 window/overlap)
  2. 对每个窗口调用模型总结 (含 retry)
  3. 汇总所有总结, 基于总结回答问题
"""

import os, re, time, json
from openai import OpenAI

# ============ 配置 ============
API_BASE = "http://162.105.19.243:11451/v1"
API_KEY = "sulab"
MODEL = "Qwen3.6-27B"
DATA_DIR = os.path.join(os.path.dirname(__file__), "books", "train")

# --- 滑动窗口参数 (基于论文推荐 + 文本分析) ---
# 推荐: window=4000 (~1000 tokens), overlap=1000 (25%)
# 依据:
#   - IEEE 2025: Sliding Window 在摘要任务上 ROUGE-1 提升 22.7%
#   - UiO 2025: 叙事文本推荐 25-33% overlap
#   - LangChain/Chrome AI: ~3000 chars = ~750 tokens
#   - Skkuhg 2025: stride = 512 tokens
WINDOW_SIZE = 4000    # 每窗口字符数
OVERLAP = 1000        # 重叠字符数 (~25%)
STRIDE = WINDOW_SIZE - OVERLAP  # 滑动步长

MAX_TOKENS_SUMMARIZE = 2048
MAX_TOKENS_ANSWER = 8192
MAX_RETRIES = 2

client = OpenAI(base_url=API_BASE, api_key=API_KEY)


# ============ 滑动窗口 ============

def sliding_window_chunks(text: str, window_size: int = WINDOW_SIZE,
                          stride: int = STRIDE) -> list[dict]:
    """
    滑动窗口切分文本, 返回 [{start, end, text}]。
    窗口结束边界对齐到最近的句子结束符 (.!?)，避免截断语义。
    """
    if not text:
        return []

    chunks = []
    start = 0
    total = len(text)

    while start < total:
        end = min(start + window_size, total)

        # 对齐到句子结束符 (保留后 200 字符内搜索)
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

def call_model(messages: list, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=max_tokens,
        extra_body={"enable_thinking": False},
    )
    return resp.choices[0].message.content or ""


def summarize_chunk(chunk_text: str, book_name: str,
                    chunk_idx: int, total: int) -> str:
    """总结一个窗口, 失败时自动重试"""
    messages = [
        {
            "role": "system",
            "content": (
                "You are reading a book section by section. "
                "For each section, provide a concise summary (2-3 sentences) "
                "of what happens, key characters, and important details. "
                "Focus on factual information. Be specific."
            ),
        },
        {
            "role": "user",
            "content": f"[{book_name} | Section {chunk_idx}/{total}]\n\n{chunk_text}",
        },
    ]

    for attempt in range(MAX_RETRIES + 1):
        result = call_model(messages, max_tokens=MAX_TOKENS_SUMMARIZE)
        if result.strip():
            return result.strip()
        if attempt < MAX_RETRIES:
            # 重试: 稍微加大 max_tokens
            messages[0]["content"] += " Provide a thorough summary."
            result = call_model(messages, max_tokens=MAX_TOKENS_SUMMARIZE * 2)
            if result.strip():
                return result.strip()
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
                    "name": fname,
                    "size_kb": size_kb,
                    "chars": len(content),
                    "est_tokens": len(content) // 4,
                    "windows": len(windows),
                })
    return books


def get_book_content(name: str) -> str:
    path = os.path.join(DATA_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ============ 持久化: 保存/恢复进度 ============

def save_progress(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_progress(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ============ 数据处理 ============

def process_books(selected_names: list[str], question: str,
                  resume_path: str | None = None) -> dict:
    """
    完整管线:
      1. 加载选中书籍
      2. 滑动窗口切分
      3. 逐窗口总结 (带进度保存)
      4. 汇总总结 + 回答问题
    """
    result = {
        "books": [],
        "answer": "",
        "total_windows": 0,
        "total_time": 0,
    }
    t_start = time.time()

    # 尝试恢复进度
    completed_windows = set()
    all_summaries = []
    if resume_path:
        saved = load_progress(resume_path)
        if saved and saved.get("selected_names") == selected_names:
            all_summaries = saved.get("summaries", [])
            for idx in saved.get("completed", []):
                completed_windows.add(idx)
            print(f"Resumed: {len(completed_windows)} windows already done.")

    for name in selected_names:
        content = get_book_content(name)
        windows = sliding_window_chunks(content)
        print(f"\n  [{name}] {len(windows)} windows")

        book_summaries = []
        for i, w in enumerate(windows, 1):
            global_idx = len(all_summaries) + 1

            if global_idx in completed_windows:
                idx_in_saved = list(completed_windows).index(global_idx)
                summary = all_summaries[idx_in_saved] if idx_in_saved < len(all_summaries) else ""
                book_summaries.append(summary)
                print(f"  [{name}] Window {i}/{len(windows)} (cached)", flush=True)
                continue

            print(f"  [{name}] Window {i}/{len(windows)}...", end=" ", flush=True)
            t0 = time.time()
            summary = summarize_chunk(w["text"], name, i, len(windows))
            elapsed = time.time() - t0
            print(f"({elapsed:.1f}s)")
            book_summaries.append(summary)
            all_summaries.append(summary)
            completed_windows.add(global_idx)

            # 每 5 个窗口保存一次进度
            if resume_path and i % 5 == 0:
                save_progress({
                    "selected_names": selected_names,
                    "summaries": all_summaries,
                    "completed": sorted(completed_windows),
                }, resume_path)

        result["books"].append({
            "name": name,
            "windows": len(windows),
            "summaries": book_summaries,
        })

    # 最终回答
    result["total_windows"] = len(all_summaries)
    print(f"\n  All {len(all_summaries)} windows summarized.")
    print(f"  Generating final answer...\n")

    summaries_text = "\n\n".join(
        f"[Section {i+1}]\n{s}" for i, s in enumerate(all_summaries)
    )

    answer_messages = [
        {
            "role": "system",
            "content": (
                "You are given section summaries of one or more books. "
                "Answer the user's question based on ALL summaries. "
                "Cite relevant sections as evidence. Be thorough."
            ),
        },
        {
            "role": "user",
            "content": f"Book Summaries:\n\n{summaries_text}\n\nQuestion: {question}",
        },
    ]

    t0 = time.time()
    answer = call_model(answer_messages, max_tokens=MAX_TOKENS_ANSWER)
    result["answer"] = answer
    result["answer_time"] = time.time() - t0
    result["total_time"] = time.time() - t_start

    # 清理进度文件
    if resume_path and os.path.exists(resume_path):
        os.remove(resume_path)

    return result


# ============ 交互界面 ============

def print_books_table():
    books = load_books(DATA_DIR)
    print(f"\n  {'Idx':<5} {'Filename':<25} {'Size':<8} {'Chars':<8} {'Windows':<8}")
    print("  " + "-" * 58)
    for i, b in enumerate(books):
        print(f"  {i:<5} {b['name']:<25} {b['size_kb']:<8} KB {b['chars']:<8,} {b['windows']:<8}")


def interactive():
    print("=" * 60)
    print("Long Text Comprehension Pipeline")
    print(f"Model: {MODEL}")
    print(f"Window: {WINDOW_SIZE}ch | Overlap: {OVERLAP}ch ({OVERLAP*100//WINDOW_SIZE}%)")
    print("=" * 60)

    books = load_books(DATA_DIR)
    print(f"\n{len(books)} books available.")
    print_books_table()

    # 选书
    print("\nSelect books (comma separated indices, range like 0-3, or 'all'):")
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
        # support range syntax: 0-3
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
    print(f"\nSelected {len(selected)} book(s), ~{tw} windows to process.")
    print(f"Estimated time: ~{tw * 30 // 60}-{tw * 40 // 60} min")

    # 问题
    question = input("\nQuestion: ").strip()
    if not question:
        print("Question required.")
        return

    proceed = input(f"\nProceed? (~{tw * 35 // 60} min) [y/N]: ").strip().lower()
    if proceed != "y":
        print("Cancelled.")
        return

    # 开始处理
    resume_path = os.path.join(os.path.dirname(__file__), "output", ".progress.json")
    result = process_books(selected, question, resume_path=resume_path)

    # 输出结果
    print(f"\n{'='*60}")
    print(f"Q: {question}")
    print(f"{'='*60}\n")
    print(result["answer"])
    print(f"\n{'='*60}")
    print(f"Time: {result['total_time']:.0f}s | Windows: {result['total_windows']}")

    # 保存
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"result_{ts}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Q: {question}\n\nA:\n{result['answer']}\n\n")
        f.write(f"--- Stats ---\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Window: {WINDOW_SIZE}ch, Overlap: {OVERLAP}ch\n")
        f.write(f"Books: {len(result['books'])}, Windows: {result['total_windows']}\n")
        f.write(f"Total time: {result['total_time']:.0f}s\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    interactive()
