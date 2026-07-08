"""
题型注册表 — 每类题 = (目标书籍, summary_prompt, qa_prompt)

用法:
  from question_types import get_type_config, list_types
  cfg = get_type_config("relationship")
  cfg["summary_prompt"]   # 该题型的总结提示词
  cfg["qa_prompt"]        # 该题型的 QA 生成提示词
  cfg["books"]            # 适合出这类题的书目列表
"""

from prompts import SUMMARIZE_CHUNK_PROMPT, QA_GENERATION_PROMPT

# ============================================================
#  题型注册表
#  每增加一类题，在这里加一个条目即可，无需改动 pipeline
# ============================================================

TYPE_REGISTRY = {

    # ── 默认（通用）────────────────────────────────────────────
    "default": {
        "label": "通用题型",
        "description": "原有的通用出题方式，不限定题型",
        "books": None,  # None = 不限制书目，由用户在交互界面选择
        "summary_prompt": SUMMARIZE_CHUNK_PROMPT,
        "qa_prompt": QA_GENERATION_PROMPT,
        "temperature": 0.3,
        "num_questions": "3-5",
    },

    # ── 人物关系 ────────────────────────────────────────────────
    "relationship": {
        "label": "人物关系",
        "description": "角色之间的社会关系、情感关系、家庭关系",
        "books": [
            "Alice's Adventures in Wonderland.cleaned.txt",
            "Dracula.cleaned.txt",
        ],
        "summary_prompt": """You are a literary analyst specializing in character relationships.
For the given book section, write a summary focusing on:

- Character introductions and interactions between characters
- Relationships — romantic, familial, social, professional
- Emotional dynamics: who loves, hates, respects, fears, or rivals whom
- Changes in relationships during this section
- How characters refer to each other (titles, nicknames, etc.)

Keep other plot details brief — only include them if they affect character relationships.
Output ONLY the summary text.""",
        "qa_prompt": """You are a QA generation expert specializing in character relationship questions.

Based on the section summaries below, generate 3-5 high-quality question-answer pairs
about character RELATIONSHIPS — romantic, familial, social, or professional.

Requirements:
- Each question must involve at least TWO named characters.
- Focus on: who loves whom, who is married to whom, who works for whom,
  who is related to whom, who is rivals with whom.
- Questions should be open-ended (not yes/no).
- Answers should be accurate and cite [N] section numbers for each factual claim.
- Questions must NOT contain section numbers.

Output format — a JSON array:
[
    {
        "book": "<book title>",
        "question": "<question>",
        "answer": "<answer with [N] citations>"
    },
    ...
]

Return ONLY the JSON array, no other text.""",
        "temperature": 0.3,
        "num_questions": "3-5",
    },

    # ── 剧情时序 ────────────────────────────────────────────────
    "timeline": {
        "label": "剧情时序",
        "description": "事件发生的先后顺序、时间线索",
        "books": [
            "1342.cleaned.txt",
            "768.cleaned.txt",
        ],
        "summary_prompt": """You are a literary analyst specializing in plot chronology.
For the given book section, write a summary focusing on:

- The sequence of events in this section
- Time markers: specific times, dates, seasons, or temporal transitions
- Cause-and-effect chains between events
- Character actions in chronological order
- Any flashbacks, flash-forwards, or non-linear narrative devices

Keep character descriptions brief — focus on WHAT happens WHEN.
Output ONLY the summary text.""",
        "qa_prompt": """You are a QA generation expert specializing in plot chronology questions.

Based on the section summaries below, generate 3-5 high-quality question-answer pairs
about the TIMELINE and SEQUENCE of events.

Requirements:
- Focus on: what happened before/after what, when events occurred,
  how much time passed between events, the order of character actions.
- Include questions about cause-and-effect: what event triggered another.
- Questions should be open-ended (not yes/no).
- Answers should be accurate and cite [N] section numbers for each factual claim.
- Questions must NOT contain section numbers.

Output format — a JSON array:
[
    {
        "book": "<book title>",
        "question": "<question>",
        "answer": "<answer with [N] citations>"
    },
    ...
]

Return ONLY the JSON array, no other text.""",
        "temperature": 0.2,
        "num_questions": "3-5",
    },

    # ── 细节定位 ────────────────────────────────────────────────
    "detail": {
        "label": "细节定位",
        "description": "特定事实的精确查找（年龄、地点、身份、物品等）",
        "books": [
            "43.cleaned.txt",
            "84.cleaned.txt",
        ],
        "summary_prompt": """You are a literary analyst specializing in factual details.
For the given book section, write a summary focusing on:

- Specific factual information: character ages, occupations, physical descriptions
- Locations and settings mentioned in this section
- Objects, possessions, and their significance
- Specific numbers, dates, measurements, or quantities
- Character titles, ranks, and formal roles

Write concisely but preserve ALL explicit factual details.
Output ONLY the summary text.""",
        "qa_prompt": """You are a QA generation expert specializing in detail-oriented questions.

Based on the section summaries below, generate 3-5 high-quality question-answer pairs
about SPECIFIC FACTUAL DETAILS in the text.

Requirements:
- Focus on: character ages, occupations, physical traits, locations,
  specific objects, numbers, dates, titles, and other verifiable facts.
- Questions should require precise answers (e.g., "What was X's occupation?")
- Answers should cite [N] section numbers for each factual claim.
- Questions must NOT contain section numbers.

Output format — a JSON array:
[
    {
        "book": "<book title>",
        "question": "<question>",
        "answer": "<answer with [N] citations>"
    },
    ...
]

Return ONLY the JSON array, no other text.""",
        "temperature": 0.2,
        "num_questions": "3-5",
    },

    # ── 全局综合 ────────────────────────────────────────────────
    "global": {
        "label": "全局综合",
        "description": "需要跨段落整合信息的综合性问题",
        "books": [
            "43.cleaned.txt",
            "84.cleaned.txt",
        ],
        "summary_prompt": SUMMARIZE_CHUNK_PROMPT,  # 全局题复用通用总结
        "qa_prompt": """You are a QA generation expert specializing in comprehensive, cross-section questions.

Based on ALL the section summaries below, generate 3-5 high-quality question-answer pairs
that require SYNTHESIZING information from MULTIPLE sections.

Requirements:
- Each question must require information from at least 2 different sections to answer.
- Focus on: character development across the book, long-term plot arcs,
  recurring themes, changes in relationships over time.
- Questions should be open-ended and require reasoning, not just lookup.
- Answers should cite [N] section numbers for each factual claim.
- Questions must NOT contain section numbers.

Output format — a JSON array:
[
    {
        "book": "<book title>",
        "question": "<question>",
        "answer": "<answer with [N] citations>"
    },
    ...
]

Return ONLY the JSON array, no other text.""",
        "temperature": 0.3,
        "num_questions": "3-5",
    },
}


# ============================================================
#  工具函数
# ============================================================

def get_type_config(type_name: str) -> dict:
    """获取指定题型的配置，不存在则返回 default"""
    if type_name in TYPE_REGISTRY:
        return TYPE_REGISTRY[type_name]
    print(f"  [Warning] Unknown question type '{type_name}', falling back to 'default'.")
    return TYPE_REGISTRY["default"]


def list_types() -> list[tuple[str, str]]:
    """返回所有题型列表：[(key, label), ...]"""
    return [(k, v["label"]) for k, v in TYPE_REGISTRY.items()]


def list_books_for_type(type_name: str) -> list[str] | None:
    """返回该题型推荐的书目列表，None 表示不限制"""
    cfg = get_type_config(type_name)
    return cfg.get("books")
