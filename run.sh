#!/bin/bash
# 一键启动：自动建 venv、装依赖、跑管线
cd "$(dirname "$0")"
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
python3 -m pip install -q openai 2>/dev/null
# 请先设置 API_KEY: export API_KEY=your_key
python3 pipeline.py
