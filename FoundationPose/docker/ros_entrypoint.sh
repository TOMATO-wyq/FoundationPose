#!/usr/bin/env bash
set -e

if [[ -f /opt/ros/humble/setup.bash ]]; then
  # ROS setup scripts may reference variables that are not defined yet.
  set +u
  source /opt/ros/humble/setup.bash
  [[ -f /opt/zed_ws/install/setup.bash ]] && source /opt/zed_ws/install/setup.bash
fi

exec "$@"
