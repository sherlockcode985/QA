"""
模型提示词常量 & 功能开关。

============================================================================
同学看这里 —— 对抗式三元组抽取系统：
  本文件包含 4 个角色提示词，组成「总结→抽取→校验→修正」对抗管线：
    Step 1: SUMMARIZE_CHUNK_PROMPT — 总结者，只做章节概括
    Step 2: EXTRACT_TRIPLES_PROMPT  — 抽取者，从原文中找 ALIAS 三元组
    Step 3: VALIDATE_TRIPLES_PROMPT — 校验者，检查三元组是否合理
    Step 4: REVISE_TRIPLES_PROMPT  — 修正者，根据反馈重新抽取

  使用方式：
    1. 修改下方各 PROMPT 来优化各角色的表现
    2. TRIPLE_INSTRUCTION 会被注入到 Step 2 和 Step 4 的提示词中
    3. ENABLE_QUESTION_INPUT / ENABLE_TRIPLE_INPUT 控制交互界面行为
============================================================================
"""

# ============ 功能开关 ============

# 是否需要手动输入问题。True=需要输入问题来让模型回答；False=不输入问题，靠 QA_GENERATION_PROMPT 自动生成 QA 对。
ENABLE_QUESTION_INPUT: bool = True

# 是否需要输入三元组来引导生成。True=在交互界面输入三元组引导；False=不输入三元组，完全靠提示词。
ENABLE_TRIPLE_INPUT: bool = False

# 是否启用原文证据验证。True=生成答案后，从原文中提取逐字证据来支撑答案中的每个事实性陈述。
ENABLE_EVIDENCE_VERIFICATION: bool = True

# 是否开启对抗式三元组抽取。True=正常抽取三元组；False=仅做文本总结与QA生成，跳过所有三元组相关步骤。
ENABLE_TRIPLE_EXTRACTION: bool = False

# 默认三元组示例（当 ENABLE_TRIPLE_INPUT=True 但同学想直接用常量里的示例时修改此处）
# 格式：subject||predicate||object，一行一个
DEFAULT_TRIPLE_EXAMPLE: str = """Gabriel Oak||HAS_ROLE||Shepherd
Bathsheba Everdene||LOVES||Gabriel Oak"""

# ============ 三元组提取指令（被 Step 2 & Step 4 共用）============

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


# ============ Step 1: 总结者提示词 ============

SUMMARIZE_CHUNK_PROMPT: str = """You are a literary analyst. For the given book section, write a concise summary covering:
- Key events and plot developments
- Character introductions, interactions, and relationships
- How each character is referred to — full names, titles, nicknames, surname-only references, pseudonyms

Pay special attention to name variations: if a character is called by different names in this section (e.g., "Mr. Boldwood" and "William Boldwood"), note this explicitly — it will be crucial for alias extraction later.

Output ONLY the summary text. Do NOT extract triples or add any other sections."""


# ============ Step 2: 抽取者提示词 ============

# {triple_instruction} 会被替换为上方 TRIPLE_INSTRUCTION
EXTRACT_TRIPLES_PROMPT: str = """You are a knowledge graph expert specializing in entity alias extraction from literary texts.

You will receive:
1. A summary of a book section (for context)
2. The original text of that section (your extraction source)

Your task: extract ALIAS triples from the ORIGINAL TEXT. Use the summary to understand who's who, but ONLY extract triples where BOTH the subject and object names ACTUALLY APPEAR in the original text.

CRITICAL: You must find the name variations IN THE TEXT. Do not invent aliases. If the text calls a character "Mr. Boldwood" in one sentence and "William Boldwood" in another, and you can confirm they refer to the same person, then output:
William Boldwood||ALIAS||Mr. Boldwood

If you cannot confirm two names refer to the same person from the text, do NOT output a triple for them.

{triple_instruction}

Output format:
[TRIPLES]
subject||ALIAS||object
[/TRIPLES]

If no valid ALIAS triples exist in this section, output an empty [TRIPLES] block."""


# ============ Step 3: 校验者提示词 ============

VALIDATE_TRIPLES_PROMPT: str = """You are a strict quality auditor for knowledge graph triples extracted from literary texts.

You will receive:
1. A set of ALIAS triples to validate (format: subject||ALIAS||object)
2. The original book text they were extracted from

For EVERY triple, apply these checks:

1. CHARACTER ROSTER TEST (the most important check):
   Ask: "If someone reads 'Who is <object>?', can I answer with '<subject>'?"
   - "Who is Jem?" → "James Ryder" ✓ → this is an ALIAS
   - "Who is the banker?" → "Alexander Holder, but that's his job, not his name" ✗ → NOT an alias
   - "Who is a lawyer?" → "I don't know which specific person" ✗ → indefinite, NOT an alias
   - "Who is her stepfather?" → "He is related to her, his name is..." ✗ → relation, NOT an alias

2. SPECIFICITY CHECK — the object must be a DEFINITE reference to a specific person:
   INVALID (indefinite): "a lawyer", "a soldier", "a stranger", "a young girl"
   INVALID (relation): "her stepfather", "his wife", "my uncle", "X's daughter"
   INVALID (job title): "the banker", "the colonel", "the doctor", "the butler"
   INVALID (honorific): "your Highness", "Sir", "Madame", "my lord"
   INVALID (description): "the old man", "the tall woman", "the unfortunate bridegroom"

3. DIRECTION CHECK — the more complete/canonical name must be the subject (left side):
   "John Watson||ALIAS||Dr. Watson" ✓
   "Dr. Watson||ALIAS||John Watson" ✗
   "William Boldwood||ALIAS||Mr. Boldwood" ✓

4. TEXT SUPPORT CHECK — does the original text actually show or imply these two names refer to the same person?
   If the text never connects the two names, flag it as unsupported.

Output format:

If ALL triples pass ALL checks:
[VERDICT]
PASS
[/VERDICT]

If ANY triple fails ANY check:
[VERDICT]
FAIL
[/VERDICT]
[FEEDBACK]
- "X||ALIAS||Y": FAILS <check name>. <brief explanation of why it fails and how to fix>
- "A||ALIAS||B": FAILS <check name>. <brief explanation>
[/FEEDBACK]

Be strict. When in doubt, FAIL the triple and explain why."""


# ============ Step 4: 修正者提示词 ============

# {triple_instruction} 会被替换为上方 TRIPLE_INSTRUCTION
REVISE_TRIPLES_PROMPT: str = """You are a knowledge graph expert. Your previous ALIAS triple extraction was reviewed and issues were found.

You will receive:
1. The book section summary
2. The original text
3. Your previous triples
4. Reviewer feedback (specific triples that failed and why)

Your task: re-extract ALIAS triples from the original text, addressing ALL issues raised in the feedback:
- REMOVE any triple the reviewer flagged as invalid
- FIX direction errors (more complete name on the LEFT as subject)
- Only output triples where both names appear in the text and clearly refer to the same person
- Apply the Character Roster Test to every triple before outputting

{triple_instruction}

Output format:
[TRIPLES]
subject||ALIAS||object
[/TRIPLES]

If after fixing all issues there are no valid triples, output an empty [TRIPLES] block."""


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

ANSWER_SYSTEM_PROMPT: str = (
    "You are given section summaries of one or more books. "
    "Answer based on ALL summaries.\n\n"
    "CRITICAL — For EACH factual claim you make, cite its source section(s) in brackets, "
    "like [Section 3] or [Sections 3-5]. "
    "Every distinct fact (character introduction, event, relationship, plot point) must be "
    "attributed to a specific section.\n\n"
    "If a claim spans multiple sections, list all of them: [Sections 3, 7, 12].\n"
    "If the question has multiple sub-parts, organize your answer with clear "
    "paragraphs or bullet points, each with its own citations.\n\n"
    "Output format — a JSON object:\n"
    '{\n'
    '    "book": "<book filename>",\n'
    '    "question": "<the question>",\n'
    '    "answer": "<your answer with [Section N] citations>"\n'
    "}\n\n"
    "Return ONLY the JSON object, no other text."
)

QA_GENERATION_PROMPT: str = """You are given section summaries of one or more books.
Based on the summaries, generate 3-5 high-quality question-answer pairs about the book content.
Each QA pair should test deep understanding of characters, relationships, events, and plot.

Requirements:
- Questions should be diverse: some about characters, some about events, some about relationships.
- Answers should be accurate and cite specific details from the summaries.
- CRITICAL: For each factual claim in each answer, cite the source section(s) in brackets:
  [Section 3] or [Sections 5-7]. Every distinct fact must have a citation.
- Avoid yes/no questions; prefer open-ended questions requiring reasoning.

Output format — a JSON array of objects:
[
    {
        "book": "<book filename>",
        "question": "<question>",
        "answer": "<answer with [Section N] citations>"
    },
    ...
]

Return ONLY the JSON array, no other text."""


# ============ 原文证据验证提示词 ============

EVIDENCE_VERIFICATION_PROMPT: str = """You are a literary evidence verification expert. Your task is to find VERBATIM supporting evidence from the original text for each QA pair.

You will receive:
1. A CITED SECTIONS block containing the original text of all cited sections
2. A QA ARRAY — JSON array of {book, question, answer} objects

For EACH QA pair's answer:
1. Read the factual claims and their [Section N] citations
2. Find VERBATIM (exact, word-for-word) quotes from the corresponding original text sections that support each claim
3. Add the evidence to that QA object

RULES:
- Evidence MUST be exact quotes from the original text — no paraphrasing, no rewording
- If a claim has support but no single sentence captures it, use the most relevant 1-3 consecutive sentences
- If a claim CANNOT be supported, do NOT include evidence for it
- Be honest — do not fabricate or stretch evidence.

Output format — a JSON array of objects, same as input but with an "evidence" field added:
[
    {
        "book": "...",
        "question": "...",
        "answer": "...",
        "evidence": ["verbatim quote 1", "verbatim quote 2"]
    },
    ...
]

IMPORTANT: Escape any double quotes (") inside verbatim evidence with backslashes (\").
If no evidence can be found for any QA pair, return the input array unchanged (without evidence field).
Return ONLY the JSON array, no other text."""
