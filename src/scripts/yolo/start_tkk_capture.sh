#!/usr/bin/env bash
# Run in a desktop terminal: own camera/source/container and save samples at 1 Hz.
set -eo pipefail
capture_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: bash src/scripts/yolo/start_tkk_capture.sh [outputs/tkk_capture_NAME]"
  echo "Start camera, RGB-D source and FoundationPose capture in a desktop terminal."
  exit 0
fi
capture_run=${1:-"$capture_root/outputs/tkk_capture_$(date +%Y%m%d_%H%M%S)"}
case "$capture_run" in
  "$capture_root"/outputs/*) ;;
  *) echo 'Capture output must be under the project outputs directory.' >&2; exit 2 ;;
esac
mkdir -p "$capture_run"
exec 8>"$capture_root/outputs/tkk_capture.lock"
flock -n 8 || { echo 'A TKK capture session is already running.' >&2; exit 2; }
capture_camera_pid=
capture_source_pid=
capture_container="tkk-fp-capture-$(date +%s)-$$"
capture_fp_started=0
capture_cleanup() {
  trap - EXIT INT TERM HUP
  # A closed terminal must not leave an attached Docker client/container behind.
  # The container is unique to this launcher, so no other session is affected.
  if [[ "$capture_fp_started" == 1 ]]; then
    sudo -n timeout -k 1s 3s docker kill --signal=INT "$capture_container" >/dev/null 2>&1 || true
    sudo -n timeout -k 1s 4s docker stop --time 3 "$capture_container" >/dev/null 2>&1 || true
  fi
  for capture_pid in "$capture_source_pid" "$capture_camera_pid"; do
    if [[ -n "$capture_pid" ]] && kill -0 "$capture_pid" 2>/dev/null; then
      kill -INT "$capture_pid" 2>/dev/null || true
    fi
  done
}
trap capture_cleanup EXIT
trap 'exit 130' INT TERM
trap 'exit 129' HUP
echo 'TKK FP capture: red/blue 20 cm model; whole object class tkk.'
echo "Output: $capture_run"
echo 'Docker requires sudo authentication in this terminal.'
sudo -v
if ! sudo -n timeout -k 1s 8s docker info >/dev/null 2>&1; then
  echo 'Docker did not respond within 8 seconds; camera was not started.' >&2
  exit 1
fi
capture_image=
for capture_candidate in foundationpose:rtx50-cu130 foundationpose:kfs-yolo foundationpose:kfs-sam2; do
  if sudo -n timeout -k 1s 5s docker image inspect "$capture_candidate" >/dev/null 2>&1; then
    capture_image=$capture_candidate
    break
  fi
done
[[ -n "$capture_image" ]] || { echo 'No prepared FoundationPose image found.' >&2; exit 1; }
echo "Using $capture_image"
export ROS_LOG_DIR="$capture_run/ros_logs"
mkdir -p "$ROS_LOG_DIR"
bash "$capture_root/src/scripts/box4d/camera.sh" usb2 8>&- >"$capture_run/camera.log" 2>&1 &
capture_camera_pid=$!
(
  source /opt/ros/humble/setup.bash
  source "$HOME/realsense_ws/install/setup.bash"
  exec /usr/bin/python3 -u "$capture_root/src_fp/camera_ros_source.py" \
    --camera d435i --scale 1 --socket "$capture_run/rgbd.sock"
) 8>&- >"$capture_run/source.log" 2>&1 &
capture_source_pid=$!
capture_ready=0
for ((capture_wait=0; capture_wait<45; capture_wait++)); do
  if ! kill -0 "$capture_camera_pid" 2>/dev/null || ! kill -0 "$capture_source_pid" 2>/dev/null; then
    tail -n 25 "$capture_run/camera.log" "$capture_run/source.log"
    exit 1
  fi
  if rg -q 'RGB-D ready' "$capture_run/source.log"; then capture_ready=1; break; fi
  sleep 1
done
[[ "$capture_ready" == 1 ]] || {
  echo 'No aligned RGB-D received within 45 seconds.' >&2
  tail -n 25 "$capture_run/camera.log" "$capture_run/source.log"
  exit 1
}
capture_relative=${capture_run#"$capture_root/"}
capture_container_run="/workspace/6Dpose/$capture_relative"
# GUI keyboard/mouse use X11; an interactive Docker stdin/PTY is unnecessary.
capture_args=(--rm --init --name "$capture_container" --gpus all --shm-size=4g --network=host
  --user "$(id -u):$(id -g)"
  --mount type=bind,src=/etc/passwd,dst=/etc/passwd,readonly
  --mount type=bind,src=/etc/group,dst=/etc/group,readonly
  -e HOME=/tmp -e MPLCONFIGDIR=/tmp/matplotlib
  -e PYTHONPATH=/workspace/6Dpose/src
  -v "$capture_root:/workspace/6Dpose" -w /workspace/6Dpose/FoundationPose)
[[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]] || {
  echo 'A desktop display is required for first-frame mask and capture controls.' >&2; exit 1;
}
capture_args+=(-e "DISPLAY=$DISPLAY" -e QT_X11_NO_MITSHM=1
  --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix)
capture_authority=${XAUTHORITY:-$HOME/.Xauthority}
if [[ -f "$capture_authority" ]]; then
  capture_args+=(-e XAUTHORITY=/tmp/.tkk.Xauthority
    --mount "type=bind,src=$capture_authority,dst=/tmp/.tkk.Xauthority,readonly")
fi
echo '圈住完整红蓝方块的可见外轮廓，按 Enter 初始化；请暂时保持相机和方块静止。'
echo '首帧定位成功后每秒自动保存一组。C 暂停/继续采样，S 额外保存，R 重定位，Q 退出。'
echo '标签为模型投影，后续需人工复核；原图不含画框。'
capture_fp_started=1
sudo -n docker run "${capture_args[@]}" "$capture_image" \
  python -u ../src_fp/run_foundationpose_live.py --camera d435i \
  --socket "$capture_container_run/rgbd.sock" \
  --mesh /workspace/6Dpose/kfs_model/tkk/tkk.obj --mask-source manual \
  --output "$capture_container_run/fp" --save-poses --dynamic-roi 0 \
  --capture-dir "$capture_container_run/dataset" --capture-class tkk --capture-interval 1 --capture-auto \
  8>&- 2>&1 | tee "$capture_run/fp.log" 8>&- &
capture_pipeline_pid=$!
# A trapped signal interrupts this wait immediately, so cleanup also runs when
# the desktop terminal closes while Docker is still attached.
wait "$capture_pipeline_pid"
