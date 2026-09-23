#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=foundationpose:rtx50-cu130
zed_sdk_installer=${ZED_SDK_INSTALLER:-/home/tomato/下载/ZED_SDK_Ubuntu22_cuda13.0_tensorrt11.0_v5.5.0.zstd.run}
zed_ros2_source=${ZED_ROS2_SOURCE:-/home/tomato/zed_ws/src/zed-ros2-wrapper}
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
    [[ -f "$zed_sdk_installer" ]] || {
      echo "ZED SDK installer not found: $zed_sdk_installer" >&2
      echo 'Set ZED_SDK_INSTALLER to the host ZED SDK 5.5.0 .run file.' >&2
      exit 1
    }
    [[ -d "$zed_ros2_source/zed_components" && -d "$zed_ros2_source/zed_wrapper" ]] || {
      echo "ZED ROS 2 wrapper checkout not found: $zed_ros2_source" >&2
      echo 'Set ZED_ROS2_SOURCE to the host zed-ros2-wrapper v5.5.0 checkout.' >&2
      exit 1
    }
    expected_zed_ros2_ref=431dcf4b1ea39caf894ca6ea294437437dbdb034
    actual_zed_ros2_ref=$(git -C "$zed_ros2_source" rev-parse HEAD 2>/dev/null || true)
    [[ "$actual_zed_ros2_ref" == "$expected_zed_ros2_ref" ]] || {
      echo "ZED ROS 2 wrapper commit mismatch: $actual_zed_ros2_ref" >&2
      echo "Expected host v5.5.0 commit: $expected_zed_ros2_ref" >&2
      exit 1
    }
    zed_group_id=$(getent group zed | cut -d: -f3)
    zed_group_id=${zed_group_id:-1001}
    # The host uses a TUN/virtual adapter whose DNS returns Fake-IP addresses
    # (for example 28.0.0.0/8).  BuildKit's default bridge cannot route those
    # addresses, while host networking follows the host TUN routing table.
    build_args=(--progress=plain --network="${DOCKER_BUILD_NETWORK:-host}")
    if [[ -n "${PIP_INDEX_URL:-}" ]]; then
      build_args+=(--build-arg "PIP_INDEX_URL=$PIP_INDEX_URL")
    fi
    "${docker_cmd[@]}" build "${build_args[@]}" \
      --build-context "zed_sdk=$(dirname -- "$zed_sdk_installer")" \
      --build-context "zed_ros2=$zed_ros2_source" \
      --build-arg "ZED_SDK_RUN_NAME=$(basename -- "$zed_sdk_installer")" \
      --build-arg "ZED_GROUP_ID=$zed_group_id" \
      -t "$image" -f "$repo_root/docker/Dockerfile.cu130" "$repo_root"
    ;;
  shell|check|zed-check)
    "${docker_cmd[@]}" image inspect "$image" >/dev/null 2>&1 || {
      echo 'CUDA 13.0 image is not built. Run: bash docker/cuda130.sh build' >&2
      exit 1
    }
    args=(--rm --gpus all --shm-size=4g --hostname foundationpose
      --privileged
      --mount type=bind,src=/dev,dst=/dev
      --network=host
      --user "$(id -u):$(id -g)"
      --mount type=bind,src=/etc/passwd,dst=/etc/passwd,readonly
      --mount type=bind,src=/etc/group,dst=/etc/group,readonly
      -e HOME=/tmp -e MPLCONFIGDIR=/tmp/matplotlib
      -v "$repo_root:/workspace/FoundationPose"
      -w /workspace/FoundationPose)
    for group_name in video zed; do
      group_id=$(getent group "$group_name" | cut -d: -f3)
      [[ -n "$group_id" ]] && args+=(--group-add "$group_id")
    done
    [[ -d /usr/local/zed/settings ]] && args+=(
      --mount type=bind,src=/usr/local/zed/settings,dst=/usr/local/zed/settings)
    [[ -d /usr/local/zed/resources ]] && args+=(
      --mount type=bind,src=/usr/local/zed/resources,dst=/usr/local/zed/resources)
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
    if [[ "${1:-shell}" == zed-check ]]; then
      "${docker_cmd[@]}" run "${args[@]}" "$image" python tools/check_zed_env.py
      exit
    fi
    "${docker_cmd[@]}" run "${args[@]}" "$image" bash -euc '
      nvcc --version
      python -m pip check
      python -c "import pyzed.sl as sl; version=sl.Camera().get_sdk_version(); assert version.startswith(\"5.5\"), version; print(\"ZED SDK\", version)"
      ros2 pkg prefix zed_wrapper
      cmake -S mycpp -B mycpp/build -DPYTHON_EXECUTABLE=/usr/bin/python3 -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
      cmake --build mycpp/build --clean-first -j4
      python -X faulthandler -u tools/check_cuda130.py
      python check_env.py
      mkdir -p logs
      python -m pip freeze > logs/packages-cu130.txt
    '
    ;;
  *) echo "Usage: bash $0 {cleanup-cu128|build|check|zed-check|shell}" >&2; exit 2 ;;
esac
