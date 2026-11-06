#!/usr/bin/env bash
set -euo pipefail
kfs_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  exec python3 "$kfs_root/src_fp/benchmark_kfs_resolutions.py" --help
fi
set +u
source /opt/ros/humble/setup.bash
source "$HOME/realsense_ws/install/setup.bash"
set -u
if ! docker info >/dev/null 2>&1; then sudo -v; fi
exec python3 "$kfs_root/src_fp/benchmark_kfs_resolutions.py" --static-scene "$@"
