#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$repo_root/logs"
{
  bash "$repo_root/docker/cuda130.sh" cleanup-cu128
  bash "$repo_root/docker/cuda130.sh" build
  bash "$repo_root/docker/cuda130.sh" check
  echo 'CUDA 13.0 migration and GPU smoke checks completed.'
} 2>&1 | tee "$repo_root/logs/migrate-cu130-$(date +%Y%m%d-%H%M%S).log"
