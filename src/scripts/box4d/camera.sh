#!/usr/bin/env bash
# Start only when no other process owns D435i. This script never kills a driver.
set -eo pipefail
BOX4D_DOMAIN="${ROS_DOMAIN_ID:-0}"
if [ "${1:-}" = "--domain" ]; then
  BOX4D_DOMAIN="${2:-}"
  if [ "$#" -lt 2 ]; then
    echo "--domain requires an integer in [0,232]" >&2
    exit 2
  fi
  shift 2
fi
case "${1:-speed}" in
  speed) BOX4D_PROFILE="640,480,60" ;;
  precision) BOX4D_PROFILE="1280,720,30" ;;
  usb2) BOX4D_PROFILE="640,480,30" ;;
  rgb) BOX4D_PROFILE="640,480,30" ;;
  -h|--help) echo "Usage: bash src/start_box4d_camera.sh [--domain ID] [speed|precision|usb2|rgb] (rgb: color only; default domain: ROS_DOMAIN_ID or 0)"; exit 0 ;;
  *) echo "Expected speed (640x480@60), precision (1280x720@30), usb2 (640x480@30) or rgb (color only)" >&2; exit 2 ;;
esac
BOX4D_COLOR_ONLY=0
[[ "${1:-speed}" == rgb ]] && BOX4D_COLOR_ONLY=1
if ! [[ "${BOX4D_DOMAIN}" =~ ^[0-9]+$ ]] || [ "${BOX4D_DOMAIN}" -gt 232 ]; then
  echo "ROS domain must be an integer in [0,232]" >&2
  exit 2
fi
export ROS_DOMAIN_ID="${BOX4D_DOMAIN}"
BOX4D_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
# One physical camera, one driver. Hold the lock for the launch lifetime and
# also detect drivers started before this guard or outside this script.
mkdir -p "${BOX4D_ROOT}/outputs"
exec 9>"${BOX4D_ROOT}/outputs/box4d_camera.lock"
if ! flock -n 9; then
  echo "Box4D camera launch is already running. Reuse it; start recognition in another terminal." >&2
  exit 2
fi
python3 - <<'PY'
from pathlib import Path
import sys

owners=[]
for process in Path('/proc').iterdir():
    if not process.name.isdigit():
        continue
    try:
        args=[part.decode(errors='replace') for part in (process/'cmdline').read_bytes().split(b'\0') if part]
    except OSError:
        continue
    if args and Path(args[0]).name in ('realsense2_camera_node','realsense-viewer','rs-capture'):
        try:
            env=dict(part.split(b'=',1) for part in (process/'environ').read_bytes().split(b'\0') if b'=' in part)
            domain=env.get(b'ROS_DOMAIN_ID',b'0').decode(errors='replace')
        except OSError:
            domain='unknown'
        owners.append((process.name,Path(args[0]).name,domain))
if owners:
    print('RealSense camera is already in use: '+', '.join(f'{name} PID {pid} (ROS_DOMAIN_ID={domain})' for pid,name,domain in owners),file=sys.stderr)
    print('Do not start another driver. Reuse the existing driver and launch only Box4D recognition.',file=sys.stderr)
    sys.exit(2)
PY
source /opt/ros/humble/setup.bash
if [ -f "${HOME}/realsense_ws/install/setup.bash" ]; then
  source "${HOME}/realsense_ws/install/setup.bash"
fi
source "${BOX4D_ROOT}/src/scripts/box4d/dds.sh"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${BOX4D_ROOT}/outputs/box4d_ros_logs}"
mkdir -p "${ROS_LOG_DIR}"
echo "Starting Box4D camera: ROS_DOMAIN_ID=${ROS_DOMAIN_ID}, profile=${BOX4D_PROFILE}. Use --domain ${ROS_DOMAIN_ID} for recognition."
if [[ "$BOX4D_COLOR_ONLY" == 1 ]]; then
  exec ros2 launch realsense2_camera rs_launch.py \
    enable_color:=true enable_depth:=false enable_accel:=false enable_gyro:=false \
    enable_rgbd:=false enable_sync:=false align_depth.enable:=false \
    config_file:="${BOX4D_ROOT}/src/config/box4d_camera.yaml" \
    rgb_camera.color_profile:="${BOX4D_PROFILE}"
fi
exec ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true \
  enable_accel:=true enable_gyro:=true \
  enable_rgbd:=true enable_sync:=true align_depth.enable:=true \
  config_file:="${BOX4D_ROOT}/src/config/box4d_camera.yaml" \
  rgb_camera.color_profile:="${BOX4D_PROFILE}" \
  depth_module.depth_profile:="${BOX4D_PROFILE}"
