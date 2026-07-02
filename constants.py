"""
数据常量 — 规范谓词集合、谓词归一化映射、事件描述关键词等。

同学如需扩展谓词集合或归一化规则，直接修改此文件即可。
"""

# ============ 规范谓词集合（模型只能使用这些 predicate） ============
CANONICAL_PREDICATES = {
    "ALIAS", "HAS_ROLE", "HAS_ATTRIBUTE",
    "LOVES", "MARRIES", "PROPOSES_TO", "REJECTS",
    "KNOWS", "FRIEND_OF", "RIVALS",
    "WORKS_FOR", "EMPLOYS",
    "LIVES_IN", "OWNS",
    "HAS_FAMILY_RELATION", "PARENT_OF", "CHILD_OF", "SIBLING_OF",
    "AUNT_OF", "UNCLE_OF",
    "LOCATED_IN", "VISITS", "PARTICIPATES_IN",
}

# ============ 自由文本谓词 → 规范谓词映射（兜底归一化） ============
PREDICATE_NORMALIZATION = {
    # 别名
    "外号": "ALIAS", "别名": "ALIAS", "绰号": "ALIAS", "人称": "ALIAS", "又称": "ALIAS",
    # 身份 / 属性
    "身份": "HAS_ROLE", "职位": "HAS_ROLE", "职业": "HAS_ROLE", "是": "HAS_ROLE",
    "担任": "HAS_ROLE", "成为": "HAS_ROLE", "角色": "HAS_ROLE",
    "特点": "HAS_ATTRIBUTE", "性格": "HAS_ATTRIBUTE",
    # 爱慕
    "爱慕": "LOVES", "喜欢": "LOVES", "爱": "LOVES", "暗恋": "LOVES",
    "爱慕对象": "LOVES", "被追求": "LOVES", "喜欢的人": "LOVES", "心上人": "LOVES",
    # 求婚
    "追求": "PROPOSES_TO", "求婚": "PROPOSES_TO", "求爱": "PROPOSES_TO",
    # 拒绝
    "拒绝": "REJECTS", "拒绝求婚": "REJECTS", "拒绝求爱": "REJECTS",
    # 婚姻
    "夫妻": "MARRIES", "结婚": "MARRIES", "嫁给": "MARRIES",
    "娶": "MARRIES", "丈夫": "MARRIES", "妻子": "MARRIES",
    "未婚妻": "PROPOSES_TO", "未婚夫": "PROPOSES_TO",
    # 亲属
    "师徒": "HAS_FAMILY_RELATION", "父亲": "PARENT_OF", "母亲": "PARENT_OF",
    "儿子": "CHILD_OF", "女儿": "CHILD_OF",
    "兄弟": "SIBLING_OF", "姐妹": "SIBLING_OF",
    "叔": "UNCLE_OF", "伯": "UNCLE_OF", "舅": "UNCLE_OF",
    "姨": "AUNT_OF", "姑": "AUNT_OF",
    "侄": "HAS_FAMILY_RELATION", "亲属": "HAS_FAMILY_RELATION",
    "亲戚": "HAS_FAMILY_RELATION", "家人": "HAS_FAMILY_RELATION",
    # 持有
    "持有": "OWNS", "拥有": "OWNS", "所属": "OWNS", "拥有者": "OWNS",
    "主人": "OWNS", "归属": "OWNS",
    # 雇佣
    "雇主": "WORKS_FOR", "仆人": "WORKS_FOR", "雇佣": "WORKS_FOR",
    "员工": "WORKS_FOR", "打工": "WORKS_FOR", "服务": "WORKS_FOR",
    "雇佣者": "EMPLOYS",
    # 位置
    "位于": "LOCATED_IN", "住在": "LIVES_IN", "来自": "LOCATED_IN",
    "出生地": "LOCATED_IN", "在": "LOCATED_IN",
    # 社交
    "认识": "KNOWS", "相识": "KNOWS", "知道": "KNOWS",
    "朋友": "FRIEND_OF", "好友": "FRIEND_OF",
    "敌人": "RIVALS", "仇人": "RIVALS", "对手": "RIVALS",
    # 杀人
    "杀死": "KILLS", "杀害": "KILLS", "谋杀": "KILLS",
    # 组织
    "成员": "MEMBER_OF", "加入": "MEMBER_OF",
    # 教导
    "教导": "MENTOR_OF", "老师": "MENTOR_OF", "指导": "MENTOR_OF",
    "师傅": "MENTOR_OF",
    # 参与
    "参加": "PARTICIPATES_IN", "参与": "PARTICIPATES_IN",
    # 拜访
    "拜访": "VISITS", "访问": "VISITS",
}

# ============ ALIAS 对象中不应出现的事件/描述关键词 ============
EVENT_DESC_WORDS = {
    "funeral", "shooting", "burial", "wedding", "marriage", "elopement",
    "arrangements", "transport", "procession", "search", "pursuit",
    "conversation", "discussion", "gossip", "dispute", "meeting",
    "performance", "celebration", "gathering", "supper", "dance", "feast",
    "journey", "removal", "farewell", "departure", "arrival",
    "identification", "investigation", "trial", "inquest", "sentencing",
    "death", "fate", "aftermath", "grave", "coffin", "laying", "laying out",
    "watch", "clothes", "marker", "gift", "letter", "note",
    "fire", "storm", "rescue", "fight", "swimming", "drowning",
    "service", "prayer", "church", "christmas",
    "race", "fair", "market",
    "preparations", "planting", "covering", "thatching", "protecting",
    "theft", "robbery", "disappearance", "escape", "release",
    "courtship", "courting", "proposal", "engagement",
    "coronation", "inauguration", "election",
    "rebellion", "war", "battle", "siege", "invasion",
}
