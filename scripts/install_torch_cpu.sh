#!/usr/bin/env bash
# 两套环境都需要 PyTorch：训练使用主环境；FunASR ONNX 的前处理也调用 torch。
set -euo pipefail
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
for environment in farfield-asr farfield-asr-decode; do
  if "$CONDA_ROOT/envs/$environment/bin/python" -c \
    'import torch, torchaudio; assert torch.__version__ == torchaudio.__version__ == "2.6.0+cpu"' \
    >/dev/null 2>&1; then
    printf '%s\n' "$environment: 已核验 PyTorch CPU 版本，跳过下载。"
    continue
  fi
  "$CONDA_ROOT/envs/$environment/bin/python" -m pip install \
    torch==2.6.0+cpu torchaudio==2.6.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
done
