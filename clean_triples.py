"""
清洗脚本：两阶段过滤三元组中的低质量数据。

  Phase 1 — 代词/冠词过滤：移除 subject 或 object 中包含人称代词、物主代词、
            反身代词、冠词的行（说明 object 是描述性短语而非真正的名字）。
  Phase 2 — 低价值别名过滤：移除 object 只是 subject 机械去掉名字、或仅在姓氏
            前加 Mr./Miss/Mrs. 等标题的低价值别名。

用法：
    python clean_triples.py <输入CSV路径> [输出CSV路径]

不指定输出路径时，输出到 <输入文件名>_cleaned.csv
"""

import csv
import re
import sys
import os

# =============================================================================
# Phase 1: 代词/冠词过滤
# =============================================================================

FILTER_WORDS = [
    # 人称代词 (主格/宾格)
    "i", "me", "you", "he", "him", "she", "her", "it", "we", "us", "they", "them",
    # 物主代词/形容词
    "my", "mine", "your", "yours", "his", "hers", "our", "ours", "their", "theirs", "its",
    # 反身代词
    "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "yourselves", "themselves",
    # 冠词
    "the", "a", "an",
]

FILTER_PATTERNS = [re.compile(rf"\b{w}\b", re.IGNORECASE) for w in FILTER_WORDS]

# =============================================================================
# Phase 2: 低价值别名过滤
# =============================================================================

# 英文常见标题/尊称前缀（按单词边界 + 可选的句点 + 空格）
_TITLES = (
    r"Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.?|Sir|Lord|Lady|"
    r"Colonel|Inspector|Captain|Professor|Reverend|Bishop|"
    r"General|Admiral|Major|Sergeant|old|young|Old|Young"
)
TITLE_STRIP_RE = re.compile(rf"^(?:{_TITLES})(?:\s+|$)", re.IGNORECASE)

# 后缀（senior / junior / the elder / the younger）
_SUFFIXES = r"(?:\s*,\s*)?\s+(?:senior|junior|Jr\.?|Sr\.?|the\s+elder|the\s+younger)"
SUFFIX_STRIP_RE = re.compile(_SUFFIXES + r"$", re.IGNORECASE)


def _strip_titles_and_suffixes(name: str) -> str:
    """去掉名字前后的标题和尊称后缀，返回纯名字部分。循环剥离以处理堆叠标题。"""
    s = name
    while True:
        s2 = TITLE_STRIP_RE.sub("", s)
        s2 = SUFFIX_STRIP_RE.sub("", s2)
        if s2.strip() == s.strip():
            break
        s = s2.strip()
    return s.strip()


def _is_low_value_alias(subject: str, object_: str) -> bool:
    """
    检测 object 是否只是 subject 的机械简化形式，没有信息增量。
    例如：
      - 去掉了名字:       Jabez Wilson → Wilson
      - 加了个标题:       Jabez Wilson → Mr. Wilson
      - 标题+全名:        Jabez Wilson → Mr. Jabez Wilson
      - 去掉标题:         Mr. Jabez Wilson → Jabez Wilson
      - 去掉名+加标题:    Mary Holder → Miss Holder
    """
    subj_core = _strip_titles_and_suffixes(subject)
    obj_core = _strip_titles_and_suffixes(object_)

    subj_tokens = subj_core.split()
    obj_tokens = obj_core.split()

    # 去掉标题后 object 为空（纯标题，如 "Colonel"）→ 垃圾数据
    if not obj_tokens:
        return True

    if not subj_tokens:
        return False

    # 去掉标题后完全一致 → 标题差异，无信息增量
    # 如: "Jabez Wilson" → "Mr. Jabez Wilson" / "Mr. Jabez Wilson" → "Jabez Wilson"
    if subj_core.lower() == obj_core.lower():
        return True

    # object 只是 subject 的后半截（姓氏部分）
    # 如: "Jabez Wilson" → "Wilson", "Neville St. Clair" → "St. Clair"
    for n in range(1, len(subj_tokens)):
        candidate = " ".join(subj_tokens[-n:])
        if candidate.lower() == obj_core.lower():
            return True

    # object 只是 subject 的前半截（名字部分）
    # 如: "Irene Adler" → "Irene", "James McCarthy" → "James"
    for n in range(1, len(subj_tokens)):
        candidate = " ".join(subj_tokens[:n])
        if candidate.lower() == obj_core.lower():
            return True

    return False


# =============================================================================
# 综合过滤 & CSV 处理
# =============================================================================

def should_filter(subject: str, object_: str) -> tuple[bool, str]:
    """两阶段过滤。返回 (是否过滤, 原因)。"""
    # Phase 1: 代词/冠词
    for word, pattern in zip(FILTER_WORDS, FILTER_PATTERNS):
        if pattern.search(subject) or pattern.search(object_):
            return True, f"代词/冠词: {word}"

    # Phase 2: 低价值别名
    if _is_low_value_alias(subject, object_):
        return True, "低价值别名"

    return False, ""


def clean_csv(input_path: str, output_path: str):
    phase1_removed = 0
    phase2_removed = 0
    kept_count = 0
    total = 0

    with open(input_path, "r", encoding="utf-8-sig") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or ["subject", "predicate", "object", "evidence", "evidence_count"]
        rows = []
        for row in reader:
            total += 1
            subject = row.get("subject", "")
            object_ = row.get("object", "")
            filtered, reason = should_filter(subject, object_)
            if filtered:
                if "代词/冠词" in reason:
                    phase1_removed += 1
                else:
                    phase2_removed += 1
                print(f"  [移除] {subject} || {row.get('predicate', '')} || {object_}  ({reason})")
            else:
                kept_count += 1
                rows.append(row)

    with open(output_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n总行数: {total}")
    print(f"Phase 1 移除 (代词/冠词): {phase1_removed}")
    print(f"Phase 2 移除 (低价值别名): {phase2_removed}")
    print(f"保留: {kept_count}")
    print(f"输出文件: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("用法: python clean_triples.py <输入CSV> [输出CSV]")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"文件不存在: {input_path}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_cleaned{ext}"

    print(f"输入: {input_path}")
    print()
    clean_csv(input_path, output_path)


if __name__ == "__main__":
    main()
