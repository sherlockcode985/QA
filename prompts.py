"""
模型提示词常量 & 功能开关。

============================================================================
同学看这里 —— 如何生成高质量 QA 对和三元组：
  你只需要修改本文件中的提示词（TRIPLE_INSTRUCTION / SUMMARIZE_SYSTEM_PROMPT /
  QA_GENERATION_PROMPT 等），不需要在交互界面输入任何问题示例或三元组示例，
  管线就能自动输出 QA 对和三元组。

  使用方式：
    1. 将 ENABLE_QUESTION_INPUT 设为 False → 不手输问题，靠 QA_GENERATION_PROMPT 自动生成
    2. 将 ENABLE_TRIPLE_INPUT 设为 False → 不手输三元组，靠提示词引导模型抽取
    3. 修改下方各 PROMPT 来优化生成质量

  如果你需要借助三元组来引导问题生成，将 ENABLE_TRIPLE_INPUT 设为 True，
  并在交互界面输入三元组作为引导即可。
============================================================================
"""

# ============ 功能开关 ============

# 是否需要手动输入问题。True=需要输入问题来让模型回答；False=不输入问题，靠 QA_GENERATION_PROMPT 自动生成 QA 对。
ENABLE_QUESTION_INPUT: bool = True

# 是否需要输入三元组来引导生成。True=在交互界面输入三元组引导；False=不输入三元组，完全靠提示词。
ENABLE_TRIPLE_INPUT: bool = False

# 默认三元组示例（当 ENABLE_TRIPLE_INPUT=True 但同学想直接用常量里的示例时修改此处）
# 格式：subject||predicate||object，一行一个
DEFAULT_TRIPLE_EXAMPLE: str = """Gabriel Oak||HAS_ROLE||Shepherd
Bathsheba Everdene||LOVES||Gabriel Oak"""

# ============ 三元组提取指令 ============

# 窗口级三元组提取详细规范，嵌入到 SUMMARIZE_SYSTEM_PROMPT 末尾使用。
TRIPLE_INSTRUCTION: str = """THIS PHASE: extract ONLY ALIAS triples. Focus exclusively on entity name variations.

FORMAT: subject||ALIAS||object
  subject = the entity's FULL/CANONICAL name (longest, most complete form)
  object  = an alternate name, nickname, shortened form, or title+surname variant

═══ DIRECTION RULE ═══
  ALWAYS: FULL_NAME||ALIAS||shorter_variant
  Compare both sides — whichever is the LONGER, more complete form MUST be the subject.
  CORRECT: John Watson||ALIAS||Dr. Watson
  WRONG:   Dr. Watson||ALIAS||John Watson   (reversed — John Watson is the fuller name)

═══ VALID ALIAS PATTERNS ═══

Create ALIAS only when BOTH subject and object are genuine NAME FORMS of the same entity:

  • Formal-name||ALIAS||nickname            (Robert||ALIAS||Bob, Elizabeth||ALIAS||Lizzy)
  • Full-name||ALIAS||surname-only          (William Boldwood||ALIAS||Boldwood)
  • Full-name||ALIAS||first-name-only       (Gabriel Oak||ALIAS||Gabriel)
  • Full-name||ALIAS||title+surname         (William Boldwood||ALIAS||Mr. Boldwood)
  • Full-title+name||ALIAS||surname-only    (Inspector Bradstreet||ALIAS||Bradstreet)
  • Full-title+name||ALIAS||bare-name       (Miss Hunter||ALIAS||Violet Hunter)
  • Real-name||ALIAS||pseudonym             (real identity ← fake name used in text)
  • Full-name||ALIAS||abbreviation          (Ku Klux Klan||ALIAS||K. K. K.)
  • Place-full-name||ALIAS||place-variant   (Weatherbury||ALIAS||Little Weatherbury)

═══ STRICTLY FORBIDDEN — these are NOT aliases ═══

1. BARE TITLES / HONORIFICS as object — these are roles, not names:
   X||ALIAS||Doctor / Mademoiselle / Sir / Madam / your Highness / Your Majesty / my lord / Herr / Monsieur

2. OCCUPATIONAL DESCRIPTIONS as object — job descriptions, not names:
   X||ALIAS||the butler / the maid / the coachman / the constable / the commissionaire / the shepherd / the baker / the carpenter / the nurse / the lawyer / the priest

3. Roles as subject — roles/jobs are not entities:
   Farm Worker||ALIAS||Farmer  /  Shepherd||ALIAS||PersonName  /  the servant||ALIAS||X

4. Descriptive phrases as subject — descriptions are not names:
   the stranger||ALIAS||X  /  young girl||ALIAS||X  /  a tall man||ALIAS||X  /  the old woman||ALIAS||X

5. Events as subject or object — events are not entities:
   The Accident||ALIAS||X  /  X||ALIAS||The Storm  /  anything containing " at " " on " " of the " " in the "

6. Possessive / pronoun / self-loop:
   X||ALIAS||X  (same name)  /  she||ALIAS||X  /  anything with 's  /  his X||ALIAS||Y

═══ DECISION FLOW ═══
For each candidate pair, ask:
  (a) Are BOTH sides genuine NAME FORMS of a person or place?
  (b) Is the LONGER name on the LEFT (subject)?
  (c) Is the object NOT a bare title, job description, event, or pronoun?
  If any answer is NO → skip it.

Extract ONLY ALIAS triples in this phase. Output 0 triples if no aliases are found — do not invent.
Prefer UNDER-extraction over OVER-extraction. If unsure, skip it."""


# ============ 窗口总结系统提示词 ============

# 同学可通过修改此提示词来调整窗口总结和三元组抽取的行为。
# {triple_instruction} 会被替换为上述 TRIPLE_INSTRUCTION。
SUMMARIZE_SYSTEM_PROMPT: str = """You are a literary analyst. For the given book section:
1. Write a concise summary covering key events, character introductions, and name variations used for each character. Pay special attention to how characters are referred to — full names, titles, nicknames, surname-only references — as this will help with alias extraction.
2. Extract ALIAS triples ONLY, following the rules below exactly.

Output format:
[SUMMARY]
<your summary>
[/SUMMARY]
[TRIPLES]
subject||predicate||object
[/TRIPLES]

Triple extraction guide:
{triple_instruction}"""


# ============ 实体对齐提示词 ============

# 实体对齐 —— 系统提示词
ENTITY_CANON_SYSTEM_PROMPT: str = (
    "You are an expert at entity resolution for literary texts. "
    "Group names referring to the same person/place/organization. "
    "Surname-only references (e.g., 'Boldwood') map to the full name (e.g., 'William Boldwood') "
    "UNLESS multiple characters share that surname. "
    "DO NOT group event descriptions, action phrases, or possessive forms as entity names. "
    "Only real names, nicknames, title variants, and place-name variants should be grouped. "
    "Output ONLY the [GROUP] blocks, no other text."
)

# 实体对齐 —— 用户提示词模板，{entities_text} 会被替换为候选实体列表。
ENTITY_CANON_USER_PROMPT: str = """Here are entity names extracted from a book. Group names that refer to the SAME entity.
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


# ============ 最终回答 / QA 生成提示词 ============

# 同学可直接修改此提示词来优化最终回答的质量。
ANSWER_SYSTEM_PROMPT: str = (
    "You are given section summaries of one or more books. "
    "Answer based on ALL summaries. Cite sections as evidence."
)

# 当 ENABLE_QUESTION_INPUT=False 时，使用此提示词从总结中自动生成 QA 对。
# 同学可修改此提示词来控制自动生成的问题风格、数量和难度。
QA_GENERATION_PROMPT: str = """You are given section summaries of one or more books.
Based on the summaries, generate 3-5 high-quality question-answer pairs about the book content.
Each QA pair should test deep understanding of characters, relationships, events, and plot.

Requirements:
- Questions should be diverse: some about characters, some about events, some about relationships.
- Answers should be accurate and cite specific details from the summaries.
- Avoid yes/no questions; prefer open-ended questions requiring reasoning.

Output format:
[QA_PAIRS]
Q1: <question>
A1: <answer>

Q2: <question>
A2: <answer>
[/QA_PAIRS]"""
