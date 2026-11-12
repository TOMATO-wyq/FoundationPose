#!/usr/bin/env bash
# Always select this bundle's controller, even when the parent workspace is sourced.
set -eo pipefail

BOX4D_CONTROL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BOX4D_CONTROL_PREFIX="${BOX4D_CONTROL_ROOT}/install/box_pose_control"
if [[ ! -f "${BOX4D_CONTROL_ROOT}/install/setup.bash" ||
      ! -x "${BOX4D_CONTROL_PREFIX}/lib/box_pose_control/box_pose_control_node" ]]; then
  echo "Build the controller first from ${BOX4D_CONTROL_ROOT}:" >&2
  echo "source /opt/ros/humble/setup.bash" >&2
  echo "colcon build --base-paths src_control --packages-select box_pose_control" >&2
  exit 2
fi

source /opt/ros/humble/setup.bash
source "${BOX4D_CONTROL_ROOT}/install/setup.bash"
# Put the local package first even if another workspace contains the same package.
export AMENT_PREFIX_PATH="${BOX4D_CONTROL_PREFIX}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"
BOX4D_SELECTED_PREFIX="$(ros2 pkg prefix box_pose_control)"
if [[ "${BOX4D_SELECTED_PREFIX}" != "${BOX4D_CONTROL_PREFIX}" ]]; then
  echo "Wrong controller package selected: ${BOX4D_SELECTED_PREFIX}" >&2
  exit 2
fi

echo "Box4D controller package: ${BOX4D_SELECTED_PREFIX}"
exec ros2 launch box_pose_control real_pipeline.launch.py \
  "repo_root:=${BOX4D_CONTROL_ROOT}" "$@"
