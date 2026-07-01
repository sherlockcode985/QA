"""
长文本理解测试 Demo (Qwen3.6-27B)
1. 选择要阅读的书籍
2. 输入问题
3. 将书籍内容分块构造为对话历史，最后一次性让模型回答
"""

import os
import sys
from openai import OpenAI

# ============ 配置 ============
API_BASE = "http://162.105.19.243:11451/v1"
API_KEY = "sulab"
MODEL = "Qwen3.6-27B"
DATA_DIR = os.path.join(os.path.dirname(__file__), "books", "train")

CHUNK_SIZE = 3000          # 每段字符数
MAX_TOKENS_ANSWER = 8192   # 回答阶段的 max_tokens

client = OpenAI(base_url=API_BASE, api_key=API_KEY)


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
                books.append({"name": fname, "content": content, "size_kb": size_kb})
    return books


def chunk_text(content: str, size: int = CHUNK_SIZE) -> list[str]:
    lines = content.split("\n")
    chunks = []
    current = ""
    for line in lines:
        if len(current) + len(line) + 1 > size and current:
            chunks.append(current.strip())
            current = line + "\n"
        else:
            current += line + "\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def call_model(messages: list, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def print_books_table(books: list[dict]):
    print(f"\n{'Index':<6} {'Filename':<25} {'Size':<8} {'Est. Sections':<14}")
    print("-" * 55)
    for i, b in enumerate(books):
        c = chunk_text(b["content"])
        print(f"{i:<6} {b['name']:<25} {b['size_kb']:<8} KB ~{len(c)}")


def select_books_interactive(books: list[dict]) -> list[dict]:
    print_books_table(books)
    print("\nEnter book indices (comma separated, e.g. 0,1,2) or 'all' for all books.")
    while True:
        try:
            choice = input(">>> ").strip()
        except EOFError:
            print("\nNo input, using first book.")
            return [books[0]]
        if not choice:
            continue
        if choice.lower() == "all":
            return books
        indices = []
        parts = choice.replace("，", ",").split(",")
        for part in parts:
            part = part.strip()
            if part.isdigit():
                idx = int(part)
                if 0 <= idx < len(books):
                    indices.append(idx)
        if not indices:
            print(f"Invalid. Enter indices 0-{len(books)-1}.")
            continue
        return [books[i] for i in sorted(set(indices))]


def build_reading_messages(selected_books: list[dict]) -> list:
    """
    将选中的书籍内容分块构造为对话历史。
    每段作为一条 user message，assistant 用简短 "OK." 回复。
    这样模型在最后回答时能看到完整对话上下文。
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a long-text comprehension assistant. "
                "Below is book content sent in sections. Read each section carefully "
                "and remember the key information. "
                "At the end you will answer questions based on everything you read."
            ),
        },
    ]

    for book in selected_books:
        chunks = chunk_text(book["content"])
        for i, chunk in enumerate(chunks, 1):
            messages.append({
                "role": "user",
                "content": f"[{book['name']} | Section {i}/{len(chunks)}]\n{chunk}",
            })
            messages.append({
                "role": "assistant",
                "content": "OK.",
            })

    return messages


def main():
    print("=" * 60)
    print("Long Text Comprehension Demo (Qwen3.6-27B)")
    print("=" * 60)

    # 1. Load
    books = load_books(DATA_DIR)
    print(f"\nLoaded {len(books)} cleaned book files.")

    # 2. Select books
    selected = select_books_interactive(books)
    print(f"\nSelected {len(selected)} book(s):")
    for b in selected:
        c = chunk_text(b["content"])
        print(f"  [{b['name']}] {b['size_kb']} KB, ~{len(c)} sections")

    # 3. Question
    print()
    while True:
        try:
            question = input("Enter your question: ").strip()
        except EOFError:
            question = ""
        if question:
            break
        print("Question cannot be empty.")

    # 4. Build reading context (no API calls)
    print(f"\n{'=' * 60}")
    print("Building reading context...\n")
    messages = build_reading_messages(selected)

    total_sections = (len(messages) - 1) // 2  # subtract system message, divide by 2
    print(f"Context built: {len(selected)} book(s), {total_sections} sections.")
    print(f"Total message turns: {len(messages) - 1}")

    # 5. Answer (single API call)
    print(f"\n{'=' * 60}")
    print("Sending to model for answer...\n")

    messages.append({
        "role": "user",
        "content": (
            f"Based on ALL the book content above, answer:\n\n{question}\n\n"
            f"Provide a detailed answer and cite relevant text as evidence. "
            f"If the books don't contain relevant info, say so honestly."
        ),
    })

    answer = call_model(messages, max_tokens=MAX_TOKENS_ANSWER)

    print(f"{'=' * 60}")
    print(f"Q: {question}")
    print(f"{'=' * 60}\n")
    print(answer)
    print(f"\n{'=' * 60}")

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "result.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Q: {question}\n\nA:\n{answer}\n")
    print(f"\nResult saved to {out_path}")


if __name__ == "__main__":
    main()
