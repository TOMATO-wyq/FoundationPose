#!/usr/bin/env bash
set -euo pipefail

if ! command -v ZED_Explorer >/dev/null 2>&1; then
    echo "找不到 ZED_Explorer，请先安装 Stereolabs ZED SDK。" >&2
    exit 1
fi

exec ZED_Explorer "$@"
