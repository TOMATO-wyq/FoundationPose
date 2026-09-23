#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$script_dir/build"
program="$build_dir/zed_photo_capture"

if [[ ! -x "$program" ]]; then
    echo "首次运行，正在编译 ZED 拍照工具..."
    cmake -S "$script_dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$build_dir" --parallel
fi

exec "$program" --output "$script_dir/captures" "$@"
