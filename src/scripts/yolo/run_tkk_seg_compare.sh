#!/usr/bin/env bash
# One camera source, two models, same decoded frames. Reuse an existing driver.
set -eo pipefail
compare_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
export PYTHONPATH="$compare_root/src${PYTHONPATH:+:$PYTHONPATH}"
compare_python=/home/tomato/code/box_seg_annotation/env/bin/python
compare_entry=tools.yolo.compare_tkk_seg
compare_prefix=tkk_seg_compare
if [[ "${1:-}" == --capture-background ]]; then
  compare_entry=tools.yolo.capture_tkk_background
  compare_prefix=tkk_background
  shift
fi
compare_domain=${ROS_DOMAIN_ID:-}
if [[ "${1:-}" == --domain ]]; then
  compare_domain=${2:-}
  [[ "$#" -ge 2 ]] || { echo '--domain requires a number' >&2; exit 2; }
  shift 2
fi
if [[ -z "$compare_domain" ]]; then
  compare_domain=$(python3 - <<'PY'
from pathlib import Path
domains=set()
for process in Path('/proc').iterdir():
    if not process.name.isdigit():continue
    try:
        command=(process/'cmdline').read_bytes().split(b'\0')
        if command and Path(command[0].decode()).name=='realsense2_camera_node':
            env=dict(part.split(b'=',1) for part in (process/'environ').read_bytes().split(b'\0') if b'=' in part)
            domains.add(env.get(b'ROS_DOMAIN_ID',b'0').decode())
    except (OSError,UnicodeError):pass
if len(domains)>1:raise SystemExit('Multiple camera domains found; specify --domain ID')
print(next(iter(domains),'0'))
PY
  )
fi
if ! [[ "$compare_domain" =~ ^[0-9]+$ ]] || [[ "$compare_domain" -gt 232 ]]; then
  echo 'ROS domain must be in [0,232]' >&2; exit 2
fi
compare_offline=0
compare_output=
compare_args=("$@")
for ((i=0; i<${#compare_args[@]}; i++)); do
  case "${compare_args[i]}" in
    -h|--help) exec "$compare_python" -m "$compare_entry" --help ;;
    --image|--image=*) compare_offline=1 ;;
    --output) compare_output=${compare_args[i+1]:-} ;;
    --output=*) compare_output=${compare_args[i]#--output=} ;;
  esac
done
compare_output=${compare_output:-"$compare_root/outputs/${compare_prefix}_$(date +%Y%m%d_%H%M%S)_$$"}
mkdir -p "$compare_output"
compare_output=$(realpath -- "$compare_output")
export YOLO_CONFIG_DIR=/home/tomato/code/box_seg_annotation/config
export MPLCONFIGDIR=/home/tomato/code/box_seg_annotation/config/matplotlib
export ROS_DOMAIN_ID="$compare_domain"
if [[ "$compare_offline" == 1 ]]; then
  exec "$compare_python" -m "$compare_entry" --output "$compare_output" "$@"
fi
source /opt/ros/humble/setup.bash
source /home/tomato/realsense_ws/install/setup.bash
export ROS_LOG_DIR="$compare_output/ros_logs"
mkdir -p "$ROS_LOG_DIR"
compare_camera_pid=
compare_process_pid=
compare_cleanup() {
  trap - EXIT INT TERM HUP
  if [[ -n "$compare_process_pid" ]] && kill -0 "$compare_process_pid" 2>/dev/null; then
    kill -INT "$compare_process_pid" 2>/dev/null || true
  fi
  if [[ -n "$compare_camera_pid" ]]; then
    # This process group belongs only to the driver started by this launcher.
    kill -TERM -- "-$compare_camera_pid" 2>/dev/null || true
    wait "$compare_camera_pid" 2>/dev/null || true
  fi
}
trap compare_cleanup EXIT
trap 'exit 130' INT TERM
trap 'exit 129' HUP
if timeout 5 ros2 topic info /camera/camera/color/image_raw --no-daemon 2>/dev/null | rg -q 'Publisher count: [1-9]'; then
  echo "Reusing camera RGB in ROS domain $compare_domain"
else
  echo "Starting D435i RGB-only 640x480@30 in ROS domain $compare_domain"
  setsid bash "$compare_root/src/scripts/box4d/camera.sh" --domain "$compare_domain" rgb \
    >"$compare_output/camera.log" 2>&1 &
  compare_camera_pid=$!
  sleep 1
  if ! kill -0 "$compare_camera_pid" 2>/dev/null; then
    cat "$compare_output/camera.log" >&2
    exit 1
  fi
fi
echo "Session output: $compare_output"
"$compare_python" -u -m "$compare_entry" --output "$compare_output" "$@" &
compare_process_pid=$!
wait "$compare_process_pid"
