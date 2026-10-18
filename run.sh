#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install -q transformers accelerate sentencepiece
python3 02_kv_cache/measure_kv_cache.py
