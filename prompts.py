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
  subject = the entity's FULL/CANONICAL name
  object  = an alternate name, nickname, shortened form, or title-based reference

═══ WHAT MAKES A QUALITY ALIAS ═══

A valid ALIAS triple captures that the same person/place is referred to by different names in the text. The subject MUST be the most complete/formal name you can find; the object is the variant.

GOOD examples:
  William Boldwood||ALIAS||Mr. Boldwood       (title + surname → full name)
  William Boldwood||ALIAS||Boldwood             (surname-only reference)
  Gabriel Oak||ALIAS||Gabriel                    (first-name-only reference)
  Gabriel Oak||ALIAS||Farmer Oak               (role + surname variant)
  Bathsheba Everdene||ALIAS||Bathsheba          (first-name-only reference)
  Bathsheba Everdene||ALIAS||Miss Everdene      (title + surname)
  Cain Ball||ALIAS||Cainy Ball                  (nickname)
  Weatherbury||ALIAS||Little Weatherbury        (place name variant)

═══ WHAT IS NOT AN ALIAS ═══

DO NOT create ALIAS for:
  • Role-to-role mappings:  Farm Worker||ALIAS||Farmer  ← roles are not entities
  • Role-to-person:         Shepherd||ALIAS||Gabriel Oak ← role is not an entity
  • Description-to-person:  the maltster||ALIAS||Warren  ← description, not entity
  • Description-to-any:     young girl||ALIAS||Fanny Robin ← "young girl" is not a name
  • Event-to-anything:      The Accident||ALIAS||The Fall on the Cobb ← events are not entities
  • Possessive phrases:     anything with 's — Batheba's aunt||ALIAS||Mrs. Hurst
  • Prepositional phrases:  names containing "at", "on", "of", "in", "by" etc. are usually events/descriptions
  • Generic references:     the stranger||ALIAS||Francis Troy ← "the stranger" is not a name
  • Pronoun:                she||ALIAS||Bathsheba Everdene ← pronouns are not names
  • Same name twice:        Gabriel Oak||ALIAS||Gabriel Oak ← meaningless

═══ ALIAS DIRECTION RULE ═══
  ALWAYS: FULL_NAME||ALIAS||variant  (canonical → variant)
  The subject holds the most complete name. If you only see "Boldwood" and "Mr. Boldwood" in the text,
  decide on the canonical form and make it the subject:
    William Boldwood||ALIAS||Boldwood
    William Boldwood||ALIAS||Mr. Boldwood

═══ HOW TO FIND ALIASES IN TEXT ═══

Look for these patterns:
  • "X, also called Y" / "X, otherwise known as Y"
  • "X, or Y" where Y is clearly a name variant
  • Surname used alone when full name was given earlier
  • First name used alone when full name was given earlier
  • Title + surname where full name is known
  • Diminutive/nickname forms: Johnny for John, Lizzy for Elizabeth
  • Character introduced with a descriptive phrase that resolves to their name:
    "the maltster, Warren Malten" → Warren Malten||ALIAS||the maltster

Extract ONLY ALIAS triples in this phase. Output 0 triples if no aliases are found — do not invent.
Extract only facts EXPLICITLY stated or CLEARLY implied by name variation in the text."""


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
