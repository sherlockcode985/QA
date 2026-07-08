"""
批量生成 QA — 每个题型 × 每本目标书，用该题型定稿后的 prompt 生成 QA。

用法:
  python batch_generate.py                           # 跑所有题型的所有书
  python batch_generate.py --types relationship timeline  # 只跑指定题型
  python batch_generate.py --types default               # 只跑默认

输出:
  output/{question_type}/{book_name}_{timestamp}.json
"""

import os, sys, time, json, argparse
from pipeline import process_books, get_book_content, load_books
from question_types import get_type_config, list_types, list_books_for_type


def run_all(selected_types: list[str] | None = None):
    """批量运行指定的题型。selected_types=None 表示跑所有题型。"""
    type_keys = selected_types if selected_types else [k for k, _ in list_types()]

    print("=" * 60)
    print("Batch QA Generation")
    print("=" * 60)

    for type_name in type_keys:
        cfg = get_type_config(type_name)
        books = list_books_for_type(type_name)

        if books is None:
            print(f"\n  [{cfg['label']}] books=None (不限), 跳过批量模式。")
            print(f"    请在命令行中用 --books 指定书籍，或用交互模式。")
            continue

        print(f"\n{'='*60}")
        print(f"  [{cfg['label']}] {len(books)} 本书")
        print(f"{'='*60}")

        for book in books:
            # 检查书是否存在
            data_dir = os.path.join(os.path.dirname(__file__), "books", "train")
            book_path = os.path.join(data_dir, book)
            if not os.path.exists(book_path):
                print(f"    [Skip] {book} not found.")
                continue

            print(f"\n  --- {book} ---")
            t0 = time.time()

            try:
                result = process_books(
                    selected_names=[book],
                    question=None,          # 自动生成 QA
                    triples_guide=None,
                    resume_path=None,
                    question_type=type_name,
                )

                elapsed = time.time() - t0
                print(f"  Done in {elapsed:.0f}s")

                # 保存
                out_dir = os.path.join(
                    os.path.dirname(__file__), "output", type_name
                )
                os.makedirs(out_dir, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(out_dir, f"{book}_{ts}.json")

                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(result["answer_with_evidence"].strip() + "\n")
                print(f"  Saved: {out_path}")

            except Exception as e:
                print(f"  [Error] {book}: {e}")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量生成 QA")
    parser.add_argument(
        "--types", nargs="+",
        help="要跑的题型，如 relationship timeline (默认: 跑所有)",
    )
    args = parser.parse_args()

    run_all(selected_types=args.types)
