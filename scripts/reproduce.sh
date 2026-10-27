#!/usr/bin/env bash
# 完整入口。已完成的训练与增强/ASR 缓存会校验后复用；不覆盖不同协议结果。
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
main_python="$CONDA_ROOT/envs/farfield-asr/bin/python"
asr_python="$CONDA_ROOT/envs/farfield-asr-decode/bin/python"
if [[ "${1:-}" == "--verify-only" ]]; then
  "$main_python" -m pytest -q
  "$CONDA_ROOT/envs/farfield-asr/bin/ruff" check src tests scripts
  "$main_python" -m pip check
  "$asr_python" -m pip check
  "$main_python" scripts/verify_testset.py
  "$main_python" scripts/summarize_asr.py
  "$main_python" scripts/summarize_ablations.py
  printf '%s\n' '已有数据、ASR 和消融验收通过。'
  exit 0
fi
if [[ -n "${1:-}" ]]; then
  printf '%s\n' '用法: bash scripts/reproduce.sh [--verify-only]' >&2
  exit 2
fi
bash scripts/bootstrap.sh
if [[ ! -x "$asr_python" ]]; then
  conda env create -f environment-asr.yml
fi
"$asr_python" -m pip install --no-deps --no-build-isolation -e .
bash scripts/install_torch_cpu.sh
conda run --no-capture-output -n farfield-asr bash scripts/run_phase3.sh
"$main_python" scripts/run_doa.py
"$main_python" scripts/prepare_training_speech.py
"$main_python" scripts/prepare_mask_training.py
"$main_python" scripts/train_mask.py
"$main_python" scripts/export_project1_denoiser.py
"$main_python" scripts/evaluate_mask_frontends.py
"$main_python" scripts/plot_phase5.py
"$asr_python" scripts/prepare_asr_model.py
"$asr_python" scripts/evaluate_asr.py
"$main_python" scripts/summarize_asr.py
"$main_python" scripts/run_ablations.py
"$main_python" scripts/summarize_ablations.py
"$main_python" scripts/create_public_demo.py
"$main_python" scripts/package_results.py
printf '%s\n' '完整实验完成。结果: results/asr.md / results/ablations.md'
