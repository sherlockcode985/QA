"""
项目全局配置 — API、模型、窗口参数、并发等。

同学可按需修改此文件中的参数来适配自己的运行环境。
"""

import os

# ============ API 配置 ============
API_BASE = "https://api.v3.cm/v1"
API_KEY = os.environ.get("API_KEY", "#")
MODEL = "qwen3.5-flash"
DATA_DIR = os.path.join(os.path.dirname(__file__), "books", "train")

# ============ 滑动窗口 ============
WINDOW_SIZE = 4000
OVERLAP = 1000
STRIDE = WINDOW_SIZE - OVERLAP

# ============ 并发 & 令牌 ============
WORKERS = 8
MAX_TOKENS_SUMMARIZE = 2048
MAX_TOKENS_ANSWER = 8192
MAX_RETRIES = 2
MAX_ITERATIONS = 3  # 对抗系统：抽取→校验 最大迭代次数

# ============ API 调用重试（应对不稳定 API） ============
API_MAX_RETRIES = 5      # 每次 API 调用的最大重试次数（指数退避）
API_RETRY_BASE_DELAY = 2  # 基础等待秒数（第 N 次重试等待 = base * N）
