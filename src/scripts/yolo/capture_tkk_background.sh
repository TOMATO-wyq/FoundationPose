#!/usr/bin/env bash
# Reuse the RGB camera lifecycle used by the live comparison.
set -eo pipefail
background_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
exec bash "$background_root/src/scripts/yolo/run_tkk_seg_compare.sh" --capture-background "$@"
