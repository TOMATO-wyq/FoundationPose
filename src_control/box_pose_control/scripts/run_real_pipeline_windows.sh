#!/usr/bin/env bash
# 一个命令打开四个 GNOME 终端窗口：
#   1 相机   2 识别   3 控制   4 海龟只读显示
#
# 用法：
#   bash src_control/box_pose_control/scripts/run_real_pipeline_windows.sh
#   bash .../run_real_pipeline_windows.sh --print     # 只打印命令，不开窗口
#
# 可覆盖的环境变量：
#   ROS_DOMAIN_ID（默认 86）、DRY_RUN（默认 true）、SERIAL_PORT、CONFIG、
#   DURATION、OUTPUT、VIEW_SCALE（默认 0.2）、BOX_SIZE（big/small）、
#   TARGET_DISTANCE_M（默认 0.30）、TARGET_LATERAL_M（默认 0.0）
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
DOMAIN="${ROS_DOMAIN_ID:-86}"
DRY_RUN="${DRY_RUN:-true}"
SERIAL_PORT="${SERIAL_PORT:-/dev/ttyUSB0}"
BOX_SIZE="${BOX_SIZE:-big}"
case "$BOX_SIZE" in
  big) DEFAULT_CONFIG="src/config/box4d_d435i_live.json" ;;
  small) DEFAULT_CONFIG="src/config/box4d_d435i_yolo_live.json" ;;
  *) echo "BOX_SIZE must be big or small" >&2; exit 2 ;;
esac
CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"
TARGET_DISTANCE_M="${TARGET_DISTANCE_M:-0.30}"
TARGET_LATERAL_M="${TARGET_LATERAL_M:-0.0}"
DURATION="${DURATION:-3600}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT="${OUTPUT:-${REPO_ROOT}/outputs/box4d_control_${TIMESTAMP}}"
VIEW_SCALE="${VIEW_SCALE:-0.2}"

PRINT_ONLY=0
if [ "${1:-}" = "--print" ]; then
  PRINT_ONLY=1
fi

if [ "$PRINT_ONLY" -eq 0 ] && ! command -v gnome-terminal >/dev/null 2>&1; then
  echo "未找到 gnome-terminal；请用一键 launch：" >&2
  echo "  ros2 launch box_pose_control real_pipeline.launch.py" >&2
  exit 1
fi

launch_window() {
  local title="$1"
  local command="$2"
  if [ "$PRINT_ONLY" -eq 1 ]; then
    printf '[%s]\n%s\n\n' "$title" "$command"
    return
  fi
  gnome-terminal --window --title="$title" -- bash -c "$command; exec bash" &
}

launch_window "1 camera" \
  "cd '${REPO_ROOT}'; bash src/start_box4d_camera.sh --domain ${DOMAIN} speed"

launch_window "2 recognition" \
  "cd '${REPO_ROOT}'; bash src/box4d_ros.sh --domain ${DOMAIN} live --size ${BOX_SIZE} --config '${CONFIG}' --auto-up --publish-poses --output '${OUTPUT}' --duration ${DURATION} --display --display-smoothed --display-hz 30"

launch_window "3 control" \
  "cd '${REPO_ROOT}'; source /opt/ros/humble/setup.bash; source install/setup.bash; export ROS_DOMAIN_ID=${DOMAIN}; ros2 launch box_pose_control control.launch.xml size:=${BOX_SIZE} target_distance_m:=${TARGET_DISTANCE_M} target_lateral_m:=${TARGET_LATERAL_M} dry_run:=${DRY_RUN} serial_port:=${SERIAL_PORT}"

launch_window "4 turtlesim view" \
  "cd '${REPO_ROOT}'; source /opt/ros/humble/setup.bash; source install/setup.bash; export ROS_DOMAIN_ID=${DOMAIN}; ros2 launch box_pose_control turtlesim_real_view.launch.xml size:=${BOX_SIZE} meters_per_unit:=${VIEW_SCALE}"

if [ "$PRINT_ONLY" -eq 1 ]; then
  exit 0
fi

echo "四个终端已启动：camera / recognition / control / turtlesim view"
echo "domain=${DOMAIN} dry_run=${DRY_RUN} serial_port=${SERIAL_PORT}"
echo "recognition output: ${OUTPUT}"
wait
