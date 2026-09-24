#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SVO_FILE="${1:-${ROOT_DIR}/zed_photo_capture/svo/recording.svo2}"
SEQUENCE_DIR="${2:-${ROOT_DIR}/outputs/zed_sequence}"
RESULT_DIR="${3:-${ROOT_DIR}/outputs/zed_foundationpose}"
CONTAINER_NAME="${FOUNDATIONPOSE_CONTAINER:-foundationpose}"

if [[ ! -x "${ROOT_DIR}/src/build/zed_svo_export" ]]; then
  cmake -S "${ROOT_DIR}/src" -B "${ROOT_DIR}/src/build"
  cmake --build "${ROOT_DIR}/src/build" -j"$(nproc)"
fi

"${ROOT_DIR}/src/build/zed_svo_export" \
  --svo "${SVO_FILE}" \
  --output "${SEQUENCE_DIR}"

if ! docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "找不到 Docker 容器 ${CONTAINER_NAME}。请先按 FoundationPose/docker/run_container.sh 启动容器。" >&2
  exit 2
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
  docker start "${CONTAINER_NAME}" >/dev/null
fi

docker exec -it \
  -e DISPLAY="${DISPLAY:-}" \
  "${CONTAINER_NAME}" \
  bash -lc "cd '${ROOT_DIR}' && python src/run_foundationpose_zed.py --sequence '${SEQUENCE_DIR}' --output '${RESULT_DIR}'"
