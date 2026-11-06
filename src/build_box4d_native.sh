#!/usr/bin/env bash
# Compatibility launcher; implementation is grouped under scripts/box4d.
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/scripts/box4d/build_native.sh" "$@"
