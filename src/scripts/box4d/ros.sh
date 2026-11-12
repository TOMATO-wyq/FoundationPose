#!/usr/bin/env bash
# Source the matching ROS environment before one of the CPU box4d entry points.
set -eo pipefail
BOX4D_DOMAIN="${ROS_DOMAIN_ID:-0}"
if [ "${1:-}" = "--domain" ]; then
  BOX4D_DOMAIN="${2:-}"
  shift 2
fi
if ! [[ "${BOX4D_DOMAIN}" =~ ^[0-9]+$ ]] || [ "${BOX4D_DOMAIN}" -gt 232 ]; then
  echo "ROS domain must be an integer in [0,232]" >&2
  exit 2
fi
case "${1:-}" in
  live) BOX4D_ENTRY="box4d.cli.live" ;;
  capture) BOX4D_ENTRY="box4d.cli.capture" ;;
  estimate-up) BOX4D_ENTRY="box4d.cli.estimate_up" ;;
  -h|--help|"") echo "Usage: bash src/box4d_ros.sh [--domain ID] {live|capture|estimate-up} [arguments...]"; exit 0 ;;
  *) echo "Expected live, capture or estimate-up" >&2; exit 2 ;;
esac
shift
BOX4D_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
source /opt/ros/humble/setup.bash
if [ -f "${HOME}/realsense_ws/install/setup.bash" ]; then
  source "${HOME}/realsense_ws/install/setup.bash"
fi
source "${BOX4D_ROOT}/src/scripts/box4d/dds.sh"
export ROS_DOMAIN_ID="${BOX4D_DOMAIN}"
# Reserve CPU capacity for RGB-D tracking while the detector runs. Explicit
# user environment settings still take precedence over these launch defaults.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export ROS_LOG_DIR="${ROS_LOG_DIR:-${BOX4D_ROOT}/outputs/box4d_ros_logs}"
mkdir -p "${ROS_LOG_DIR}"
export PYTHONPATH="${BOX4D_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${BOX4D_PYTHON:-python3}" -m "${BOX4D_ENTRY}" "$@"
