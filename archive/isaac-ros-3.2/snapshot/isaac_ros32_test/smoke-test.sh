#!/usr/bin/env bash
set -euo pipefail
cd /work
mkdir -p models engines
nvidia-smi
dpkg-query -W libnvinfer10 libnvinfer-plugin10 libnvonnxparsers10
for model in refine score; do
  if [[ ! -s "models/${model}_model.onnx" ]]; then
    curl -fL --retry 3 \
      "https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.0_onnx/files/${model}_model.onnx" \
      -o "models/${model}_model.onnx.download"
    mv "models/${model}_model.onnx.download" "models/${model}_model.onnx"
  fi
done

# The 3.2 quickstart uses FP32 and different maximum batch sizes for
# refinement (42) and scoring (252). Keep those profiles for compatibility.
for model in refine score; do
  max_batch=42
  [[ "$model" == score ]] && max_batch=252
  engine="engines/${model}_trt_engine.plan"
  if [[ ! -s "$engine" ]]; then
    trtexec --onnx="models/${model}_model.onnx" \
      --saveEngine="${engine}.building" \
      --minShapes=input1:1x160x160x6,input2:1x160x160x6 \
      --optShapes=input1:1x160x160x6,input2:1x160x160x6 \
      --maxShapes="input1:${max_batch}x160x160x6,input2:${max_batch}x160x160x6" \
      --skipInference \
      2>&1 | tee "${model}-build.log"
    mv "${engine}.building" "$engine"
  fi
  trtexec --loadEngine="$engine" \
    --shapes=input1:1x160x160x6,input2:1x160x160x6 \
    --warmUp=500 --duration=3 2>&1 | tee "${model}-inference.log"
done
printf '\nFoundationPose TensorRT model smoke test passed.\n'
printf 'This verifies model inference only, not Isaac ROS GXF or the ZED pipeline.\n'
