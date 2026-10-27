#!/usr/bin/env bash
# 在 Ubuntu 执行。仅创建专用环境；不修改现有学习环境。
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if ! command -v conda >/dev/null 2>&1; then
    for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3"; do
        if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
            source "$candidate/etc/profile.d/conda.sh"
            break
        fi
    done
fi
if ! command -v conda >/dev/null 2>&1; then
    printf '%s\n' '未找到 Conda。请先提供已有 Conda 路径。' >&2
    exit 1
fi
mkdir -p "$PROJECT_ROOT/results/generated"
conda env list --json > "$PROJECT_ROOT/results/generated/conda-envs.json"
if conda run -n farfield-asr python -c 'import sys; print(sys.executable)' >/dev/null 2>&1; then
    printf '%s\n' 'farfield-asr 已存在；为避免覆盖，跳过创建并检查依赖。'
else
    conda env create -f "$PROJECT_ROOT/environment.yml"
fi
conda run -n farfield-asr python -m pip install --no-deps --no-build-isolation -e "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
conda run -n farfield-asr python -c 'import pyroomacoustics; print("pyroomacoustics:", pyroomacoustics.__version__)'
conda run -n farfield-asr python scripts/check_environment.py
conda run -n farfield-asr python -m pytest -q
conda run -n farfield-asr ruff check src tests scripts
conda run -n farfield-asr python -m pip check
conda run -n farfield-asr python scripts/plot_beampattern.py
conda list -n farfield-asr --explicit > results/generated/conda-linux-64.explicit.txt
conda run -n farfield-asr python -m pip freeze > results/generated/pip-freeze.txt
printf '%s\n' '第一阶段检查通过。激活命令：conda activate farfield-asr'

