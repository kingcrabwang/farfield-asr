#!/usr/bin/env bash
# 先 conda activate farfield-asr。下载只读公开数据，不上传音频或账户信息。
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
python -c 'import sys; from pathlib import Path; assert Path(sys.prefix).name == "farfield-asr", "请先激活 farfield-asr"'
command -v curl >/dev/null
# 固定线程数量，减少多进程之间争抢 BLAS 线程，便于复现。
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python scripts/prepare_aishell_subset.py
python scripts/prepare_noise_subset.py
python scripts/generate_testset.py
python scripts/verify_testset.py --reproduce-one
python scripts/evaluate_frontends.py
python scripts/plot_phase3.py
python scripts/summarize_phase3.py
printf '%s\n' '第三阶段完成：results/generated/phase3/results.md'
