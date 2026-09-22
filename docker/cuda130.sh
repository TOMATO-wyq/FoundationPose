#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=foundationpose:rtx50-cu130
docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then
  docker_cmd=(sudo docker)
fi
case "${1:-shell}" in
  cleanup-cu128)
    # Remove only explicitly named images from this task. Never force removal
    # of containers, prune all images, or delete project files/volumes.
    for old_image in foundationpose:zed-ros-humble-cu128 foundationpose:zed-test-cu128 \
      foundationpose:rtx50-cu128 stereolabs/zed:5.4-gl-devel-cuda12.8-ubuntu22.04 \
      nvidia/cuda:12.8.1-devel-ubuntu22.04 nvidia/cuda:12.8.1-base-ubuntu22.04; do
      if "${docker_cmd[@]}" image inspect "$old_image" >/dev/null 2>&1; then
        echo "Removing old image: $old_image"
        "${docker_cmd[@]}" image rm "$old_image"
      fi
    done
    ;;
  build)
    build_args=(--progress=plain)
    if [[ -n "${PIP_INDEX_URL:-}" ]]; then
      build_args+=(--build-arg "PIP_INDEX_URL=$PIP_INDEX_URL")
    fi
    "${docker_cmd[@]}" build "${build_args[@]}" -t "$image" -f "$repo_root/docker/Dockerfile.cu130" "$repo_root"
    ;;
  shell|check)
    "${docker_cmd[@]}" image inspect "$image" >/dev/null 2>&1 || {
      echo 'CUDA 13.0 image is not built. Run: bash docker/cuda130.sh build' >&2
      exit 1
    }
    args=(--rm --gpus all --shm-size=4g --hostname foundationpose
      --user "$(id -u):$(id -g)"
      --mount type=bind,src=/etc/passwd,dst=/etc/passwd,readonly
      --mount type=bind,src=/etc/group,dst=/etc/group,readonly
      -e HOME=/tmp -e MPLCONFIGDIR=/tmp/matplotlib
      -v "$repo_root:/workspace/FoundationPose"
      -w /workspace/FoundationPose)
    if [[ "${1:-shell}" == shell ]]; then
      # Pass the host X11 display and its authentication cookie to OpenCV.
      if [[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]]; then
        args+=(-e "DISPLAY=$DISPLAY" -e QT_X11_NO_MITSHM=1
          --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix)
        xauthority_file=${XAUTHORITY:-$HOME/.Xauthority}
        if [[ -f "$xauthority_file" ]]; then
          args+=(-e XAUTHORITY=/tmp/.foundationpose.Xauthority
            --mount "type=bind,src=$xauthority_file,dst=/tmp/.foundationpose.Xauthority,readonly")
        fi
      fi
      exec "${docker_cmd[@]}" run -it "${args[@]}" "$image" bash
    fi
    "${docker_cmd[@]}" run "${args[@]}" "$image" bash -euc '
      nvcc --version
      python -m pip check
      cmake -S mycpp -B mycpp/build -DPYTHON_EXECUTABLE=/usr/bin/python3 -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
      cmake --build mycpp/build --clean-first -j4
      python -X faulthandler -u tools/check_cuda130.py
      python check_env.py
      mkdir -p logs
      python -m pip freeze > logs/packages-cu130.txt
    '
    ;;
  *) echo "Usage: bash $0 {cleanup-cu128|build|check|shell}" >&2; exit 2 ;;
esac
