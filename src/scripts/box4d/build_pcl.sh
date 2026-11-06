#!/usr/bin/env bash
set -euo pipefail
BOX4D_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cmake -S "${BOX4D_ROOT}/src/pcl" -B "${BOX4D_ROOT}/build/box4d_pcl" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(command -v "${BOX4D_PYTHON:-python3}")"
cmake --build "${BOX4D_ROOT}/build/box4d_pcl" --parallel "${BOX4D_BUILD_JOBS:-2}"
