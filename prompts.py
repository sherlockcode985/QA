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
# {canonical_predicates} 会被替换为 CANONICAL_PREDICATES 的排序列表。
TRIPLE_INSTRUCTION: str = """Extract Knowledge Graph triples representing CHARACTER-LEVEL long-term facts.
Be SELECTIVE — 15-25 high-quality triples per section is better than 50 noisy ones.

FORMAT: subject||predicate||object

═══ VALID SUBJECT ═══
  ONLY proper named entities: a specific person (Gabriel Oak, William Boldwood, not "Boldwood"),
  named place (Weatherbury, Casterbridge), named organization, or named significant object.
  NEVER use: adjectives, abstract concepts, events, roles, or descriptions as subjects.
  WRONG: the maltster || HAS_ATTRIBUTE || very old
  WRONG: Farm Worker || ALIAS || Farmer
  RIGHT: maltster's actual name || HAS_ROLE || Maltster

═══ VALID OBJECT by predicate ═══
  HAS_ROLE:      GENERIC role/title ONLY: Farmer, Shepherd, Servant, Soldier, Bailiff, Maid, Clerk.
                 NEVER a person's name. WRONG: X||HAS_ROLE||Baily Pennyways; X||HAS_ROLE||Gabriel Oak
  HAS_ATTRIBUTE: a TRAIT (1-4 words): brave, wealthy, tall, "28 years old", headstrong, handsome.
                 NEVER a place, organization, role, or person name.
                 WRONG: X||HAS_ATTRIBUTE||Church of England; X||HAS_ATTRIBUTE||Eleventh Dragoon-Guards Soldier
  ALIAS:         an alternate name/nickname for the subject. Subject MUST be a person or named place.
                 WRONG: Farm Worker||ALIAS||Farmer (roles are not entities)
  OWNS:          significant named possessions only (farm, house, horse, dog). NOT trivial items.
  KNOWS:         established acquaintances only. If A KNOWS B, do NOT also create B KNOWS A.
  LIVES_IN:      primary residence only (Gabriel Oak||LIVES_IN||Weatherbury), not temporary stays.
  PARTICIPATES_IN: NAMED events ONLY: "The Storm", "sheep-shearing", "Greenhill Fair".
                 NEVER a person. WRONG: Gabriel Oak||PARTICIPATES_IN||Bathsheba Everdene
                 NEVER a place. WRONG: Gabriel Oak||PARTICIPATES_IN||Casterbridge

═══ FORBIDDEN ═══
  X||HAS_ROLE||<person name>    ← role objects must be generic roles, never persons
  X||HAS_ATTRIBUTE||<place/org>  ← attributes are traits, not locations or organizations
  X||HAS_ATTRIBUTE||<person name> ← attributes are traits, not people
  <role>||ALIAS||<role variant>  ← roles are not entities, don't create ALIAS for them
  <description>||ANY||ANY        ← "young girl", "the maltster", "a soldier" are not entities
  X||KNOWS||Y and Y||KNOWS||X   ← KNOWS is symmetric, only output one direction
  X||PARTICIPATES_IN||<person>   ← events are not people
  X||PARTICIPATES_IN||<place>    ← events are not locations

Canonical predicates: {canonical_predicates}
Extract only facts EXPLICITLY stated in the text."""


# ============ 窗口总结系统提示词 ============

# 同学可通过修改此提示词来调整窗口总结和三元组抽取的行为。
# {rel_list} 会被替换为规范谓词列表，{triple_instruction} 会被替换为上述 TRIPLE_INSTRUCTION。
SUMMARIZE_SYSTEM_PROMPT: str = """You are a literary analyst. For the given book section:
1. Write a 2-3 sentence summary including key events, characters, and details.
2. Extract 15-25 high-quality Knowledge Graph triples (subject||predicate||object).

=== TRIPLE EXTRACTION RULES (follow strictly) ===

PREDICATES (use ONLY these): {rel_list}

═══ SUBJECT — only proper named entities ═══
  RIGHT: Gabriel Oak, William Boldwood, Bathsheba Everdene, Weatherbury
  WRONG: Boldwood (use full name), the maltster (find the name), a soldier (find the name)
  WRONG: 'young girl', 'Liddy's sister', 'the stranger' — descriptions, NOT entities
  Use the MOST COMPLETE name known: William Boldwood, not Boldwood.

═══ OBJECT by predicate — what is valid ═══
  HAS_ROLE: GENERIC role ONLY — Farmer, Shepherd, Servant, Soldier, Bailiff, Maid, Clerk.
    WRONG: Gabriel Oak||HAS_ROLE||Baily Pennyways (person, not role)
    WRONG: Cain Ball||HAS_ROLE||Gabriel Oak (person, not role)
  HAS_ATTRIBUTE: a TRAIT (1-4 words) — brave, wealthy, tall, '28 years old', headstrong.
    WRONG: X||HAS_ATTRIBUTE||Church of England (that's a religion/organization)
    WRONG: X||HAS_ATTRIBUTE||Eleventh Dragoon-Guards Soldier (that's a role)
    WRONG: 'clever man in talents', 'observant of stars' (verbose descriptions)
  OWNS: significant named possessions only (farm, house, horse, dog). Skip trivial items.
    NEVER a person: Gabriel Oak||OWNS||Fanny Robin is WRONG.
  KNOWS: established acquaintances. Do NOT create both A||KNOWS||B and B||KNOWS||A.
  ALIAS: alternate name for a person/place. NOT for roles (Farm Worker is a role, not entity).
  PARTICIPATES_IN: NAMED events only. NEVER a person or place name.
    WRONG: Gabriel Oak||PARTICIPATES_IN||Bathsheba Everdene (person, not event)
    RIGHT: Gabriel Oak||PARTICIPATES_IN||The Storm
  HAS_ATTRIBUTE: a TRAIT in 1-4 words. WRONG: 'imposing height and breadth', 'clever man in talents'.

Output format:
[SUMMARY]
<your 2-3 sentence summary>
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
