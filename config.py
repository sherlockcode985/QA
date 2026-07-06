"""
项目全局配置 — API、模型、窗口参数、并发等。

同学可按需修改此文件中的参数来适配自己的运行环境。
"""

import os

# ============ API 配置 ============
API_BASE = "http://162.105.19.152:11451/v1"
API_KEY = os.environ.get("API_KEY", "sulab")
MODEL = "Qwen3.6-27B"
DATA_DIR = os.path.join(os.path.dirname(__file__), "books", "train")

# ============ 滑动窗口 ============
WINDOW_SIZE = 4000
OVERLAP = 1000
STRIDE = WINDOW_SIZE - OVERLAP

# ============ 并发 & 令牌 ============
WORKERS = 2
MAX_TOKENS_SUMMARIZE = 2048
MAX_TOKENS_ANSWER = 8192
MAX_RETRIES = 2
MAX_ITERATIONS = 3  # 对抗系统：抽取→校验 最大迭代次数
