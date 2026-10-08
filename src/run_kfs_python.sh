#!/usr/bin/env bash
# Run with dependencies isolated in this project, using a chosen read-only Python base.
set -euo pipefail
kfs_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
kfs_python=${KFS_PYTHON:-python3}
export PYTHONPATH="$kfs_root/.kfs-deps:$kfs_root/third_party/sam2:$kfs_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$kfs_python" "$@"
