#!/usr/bin/env bash
# Run inside the IsaacFP container. No camera topics are required for this startup check.
set -o pipefail

timeout 60s ros2 launch isaac_ros_foundationpose isaac_ros_foundationpose.launch.py \
  mesh_file_path:=/workspace/6Dpose/kfs_model/BlueTrueKFS13/BlueTrueKFS13.obj \
  texture_path:=/workspace/6Dpose/kfs_model/BlueTrueKFS13/BlueTrueKFS13.png \
  refine_model_file_path:=/work/models/refine_model.onnx \
  score_model_file_path:=/work/models/score_model.onnx \
  refine_engine_file_path:=/work/engines/refine_trt_engine.plan \
  score_engine_file_path:=/work/engines/score_trt_engine.plan \
  2>&1 | tee /work/ros-start.log

status=${PIPESTATUS[0]}
if grep -Eq 'Failed to load library|Failed to load node|cannot open shared object file|\[FATAL\]' \
    /work/ros-start.log; then
  printf '\nROS 组件加载失败，详细错误见 /work/ros-start.log。\n' >&2
  exit 1
fi
if [[ $status -eq 124 ]]; then
  printf '\n60 秒内节点未崩溃；这是启动检查的预期超时，不代表已经完成相机接入。\n'
  exit 0
fi
exit "$status"
