#!/usr/bin/env bash
# 从 VMware 共享目录复制一次到 Ubuntu 本地磁盘，再创建环境。
set -euo pipefail
SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_ROOT="$HOME/farfield-asr"
if [[ -e "$TARGET_ROOT" ]]; then
    printf '目标已存在，为避免覆盖，请先检查：%s\n' "$TARGET_ROOT" >&2
    exit 1
fi
cp -R -- "$SOURCE_ROOT" "$TARGET_ROOT"
# VMware 共享目录可能把普通源码标成 777；复制后恢复本地常规权限。
find "$TARGET_ROOT" -type d -exec chmod 755 {} +
find "$TARGET_ROOT" -type f -exec chmod 644 {} +
chmod 755 "$TARGET_ROOT"/scripts/*.sh
mkdir -p "$TARGET_ROOT/results/generated"
# pipefail 保留安装失败状态；日志同时写到本地项目和共享目录，便于排查。
bash "$TARGET_ROOT/scripts/bootstrap.sh" 2>&1 | tee \
    "$TARGET_ROOT/results/generated/bootstrap.log" "$SOURCE_ROOT/bootstrap-ubuntu.log"

