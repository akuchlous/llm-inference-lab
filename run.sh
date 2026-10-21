#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo "Dependencies installed from requirements.txt"
echo "Run a lab explicitly, for example:"
echo "  python 06_quantization/bf16_baseline.py"
echo "  python 06_quantization/int8_baseline.py"
