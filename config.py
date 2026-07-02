"""
项目全局配置 — API、模型、窗口参数、并发等。

同学可按需修改此文件中的参数来适配自己的运行环境。
"""

import os

# ============ API 配置 ============
API_BASE = "https://api.v3.cm/v1"
API_KEY = os.environ.get("API_KEY", "sk-7UYrjDTvNGkCiSof5bAb604870C1401b88Ac44FfF4C569Cc")
MODEL = "claude-sonnet-5"
DATA_DIR = os.path.join(os.path.dirname(__file__), "books", "train")

# ============ 滑动窗口 ============
WINDOW_SIZE = 4000
OVERLAP = 1000
STRIDE = WINDOW_SIZE - OVERLAP

# ============ 并发 & 令牌 ============
WORKERS = 6
MAX_TOKENS_SUMMARIZE = 2048
MAX_TOKENS_ANSWER = 8192
MAX_RETRIES = 2
