#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
source /home/tomato/realsense_ws/install/setup.bash
camera_pid=''
cleanup(){
 if [ -n "$camera_pid" ]; then
  # 后台 ros2 launch 可能继承忽略 SIGINT 的设置；SIGTERM 由 launch 转发给相机子进程。
  kill -TERM "$camera_pid" 2>/dev/null || true
  wait "$camera_pid" 2>/dev/null || true
 fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
if ! timeout 4 ros2 topic info /camera/camera/color/image_raw --no-daemon 2>/dev/null | rg -q 'Publisher count: [1-9]'; then
 if ros2 node list --no-daemon 2>/dev/null | rg -q '^/camera/camera$'; then
  echo '相机节点存在但无 RGB 发布者，请先检查驱动，避免重复启动。' >&2; exit 1
 fi
 echo '启动 D435i 彩色流（退出测试时只关闭本脚本启动的驱动）'
 mkdir -p /home/tomato/code/box_seg_annotation/live_tests
 ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=false rgb_camera.color_profile:=1280,720,30 > /home/tomato/code/box_seg_annotation/live_tests/camera_latest.log 2>&1 &
 camera_pid=$!
fi
export YOLO_CONFIG_DIR=/home/tomato/code/box_seg_annotation/config
export MPLCONFIGDIR=/home/tomato/code/box_seg_annotation/config/matplotlib
/home/tomato/code/box_seg_annotation/env/bin/python /home/tomato/6Dpose/src/box_test_seg_d435i.py "$@"
