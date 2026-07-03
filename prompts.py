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
TRIPLE_INSTRUCTION: str = """THIS PHASE: extract ONLY ALIAS triples — genuine NAME variations of the SAME entity.

FORMAT: subject||ALIAS||object
  subject = the most complete/canonical name form
  object  = a genuine alternate name, nickname, or pseudonym

═══ THE ONE TEST THAT MATTERS ═══

Before outputting ANY triple, apply the CHARACTER ROSTER TEST:

  Imagine you are writing a character list:
    Character: Alexander Holder
    Also Known As: ???

  The object MUST be something that belongs in "Also Known As".
  Ask yourself: "If someone read 'Who is <object>?', could I answer with just the character's name?"

    "Who is Jem?" → "James Ryder"                         ✓ PASS → ALIAS
    "Who is Vincent Spaulding?" → "John Clay"              ✓ PASS → ALIAS (pseudonym)
    "Who is Mr. Boldwood?" → "William Boldwood"            ✓ PASS → ALIAS (title+surname)
    "Who is the banker?" → ??? "Alexander Holder, but..."  ✗ FAIL — it's a job, not a name
    "Who is her stepfather?" → ???                         ✗ FAIL — it's a relation, not a name
    "Who is your Highness?" → ???                          ✗ FAIL — it's an honorific, not a name
    "Who is the old man?" → ???                            ✗ FAIL — it's a description, not a name
    "Who is a lawyer?" → ???                               ✗ FAIL — indefinite, not a specific name

  This test alone eliminates most noise. Use it on EVERY candidate.

═══ VALID ALIAS PATTERNS ═══

  nickname:          James Ryder||ALIAS||Jem
  surname-only:      William Boldwood||ALIAS||Boldwood
  first-name-only:   Gabriel Oak||ALIAS||Gabriel
  title+surname:     William Boldwood||ALIAS||Mr. Boldwood
  pseudonym:         John Clay||ALIAS||Vincent Spaulding
  initial-based:     Henry Baker||ALIAS||H. B.
  name-shortening:   Francis Hay Moulton||ALIAS||Francis H. Moulton
  place-variant:     Weatherbury||ALIAS||Little Weatherbury

═══ CATEGORIES TO IGNORE ENTIRELY ═══

These are NOT aliases. Do not extract them under any predicate:

  INDEFINITE DESCRIPTIONS (use "a/an"):
    a lawyer, a soldier, a tall man, a young girl, a stranger
    → No specific referent. Discard.

  FAMILIAL REFERENCES (possessive):
    her stepfather, his wife, my uncle, X's daughter
    → Relations, not names. Discard.

  HONORIFICS / FORMAL ADDRESS:
    your Highness, Your Majesty, Sir, Madame, my lord
    → Forms of address, not names. Discard.

  OCCUPATIONAL LABELS (bare or with "the"):
    the banker, the colonel, the doctor, the commissionaire, the butler
    → Job titles used as shorthand, not names. Discard.

  NARRATIVE CIRCUMLOCUTIONS:
    the old man, the unfortunate bridegroom, my dear girl, the man himself
    → Temporary narrative references. Discard.

  PRONOUNS: she, he, they, I, you → Discard.
  SELF-LOOPS: X||ALIAS||X → Discard.
  POSSESSIVE FORMS: anything with 's → Discard.
  OBJECT-AS-SUBJECT: her stepfather||ALIAS||my stepfather → Discard.

═══ DIRECTION ═══
  Longer/more-complete name on the LEFT (subject).
  John Watson||ALIAS||Dr. Watson ✓
  Dr. Watson||ALIAS||John Watson ✗

Extract ONLY ALIAS triples. If nothing passes the roster test, output 0 triples.
When in doubt, DISCARD."""


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
