#!/usr/bin/env bash
# Compatibility entry; implementation lives under scripts/yolo.
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/scripts/yolo/start_tkk_capture.sh" "$@"
