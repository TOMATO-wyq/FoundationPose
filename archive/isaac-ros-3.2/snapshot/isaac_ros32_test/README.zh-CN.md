# Isaac ROS 3.2 / RTX 5060 兼容性验证

这是一个独立的模型推理验证环境，并非完整的 Isaac ROS 安装。它不会修改现有的 `foundationpose:rtx50-cu130` 镜像，也不会修改主机上的 CUDA 和 ZED SDK 安装。

通过检查 NVIDIA 的 release-3 软件包索引和 FoundationPose `.deb` 包，确认了以下几点：

- `ros-humble-gxf-isaac-tensor-rt` 依赖 TensorRT 10 主版本。
- `libgxf_isaac_foundationpose.so` 依赖 `libcudart.so.12` 和 CV-CUDA。
- FoundationPose 库包含 PTX，但仅凭这一点无法证明所有 GXF 依赖或内核都能在 RTX 5060 上运行。
- 现有的原版 FoundationPose 容器包含 TensorRT 11 的 C++ 库，不能用它们替代所需的 TensorRT 10 库。

此验证环境使用 Ubuntu 22.04、CUDA 12.8.1 和 TensorRT 10.9.0.34。NVIDIA 的 Ubuntu 22.04 软件包索引中提供了该 TensorRT 版本。模型配置与 Isaac ROS 3.2 快速入门一致：refine 模型的最大批量大小为 42，score 模型为 252。根据官方快速入门，模型转换可能至少需要 7.5 GB 空闲显存。这里没有人为设置 TensorRT 工作空间上限。

在主机上构建：

```bash
cd ~/code/isaac_ros32_test
sudo docker build --progress=plain -t foundationpose:isaac32-trt109-probe .
```

运行前请停止 ZED 节点和其他 GPU 负载，为引擎转换留出显存。结果会以调用用户的 UID 写入此目录。

```bash
sudo docker run --rm --gpus all --network host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" foundationpose:isaac32-trt109-probe
```

如果此验证成功，下一步是验证完整的 Isaac ROS 3.2 GXF 计算图。此验证不对端到端跟踪进行性能测试：`trtexec` 使用合成输入计时，不包括渲染、预处理、ROS 传输和 ZED。

## Isaac ROS 软件包镜像

使用 TensorRT 10.9 的模型冒烟测试已在 RTX 5060 上通过。下一层独立验证环境安装 ROS 2 Humble 和官方 Isaac ROS 3.2 FoundationPose Debian 软件包，不会修改原有的 CUDA 13 镜像和主机环境：

```bash
sudo docker build --progress=plain -f Dockerfile.isaac-ros \
  -t foundationpose:isaac32-ros .
```

镜像构建完成后，运行 `source ~/code/isaac_ros32_test/activate` 进入环境。该脚本共享 `/work` 目录以存放引擎和日志，并以只读方式挂载 `/workspace/6Dpose` 中现有的物体模型。ZED ROS 节点仍运行在主机上。

要将此镜像视为可用的 ZED 流水线，还需要完成 GXF 加载测试以及相机和掩码的重映射。

镜像包含 NVIDIA 软件包提供的 CV-CUDA 0.5，与 release-3.2 的 `Dockerfile.base` 所用版本一致。GXF FoundationPose 扩展运行时需要 `libcvcuda.so.0` 和 `libnvcv_types.so.0`。

## 紫色箱体录像的离线失跟恢复验证

先启动 tracker-only launch 和 `seed_tracking_bridge.py`，再在宿主机运行：

```bash
python3 ~/code/isaac_ros32_test/play_recording.py \
  --tracker-test --tracker-full \
  --frame-stride 2 --max-frames 100 --result-timeout 0
```

播放器用紫色区域与预测 3D 框的覆盖率判断明显漂移；连续 3 帧低于 0.65
时，桥接端读取下一帧的原版 FP 位姿并重新播种 Isaac tracker。
日志会标出 `DRIFT_TRIGGER` 和 `ORIGINAL_POSE_RESEED`。这只是用**预先保存的
原版位姿**验证恢复逻辑，不是 Isaac pose 节点实时重新定位，也不适用于任意颜色的物体。
完整说明见 [README.md](README.md)。

资料来源：

- https://nvidia-isaac-ros.github.io/v/release-3.2/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/index.html
- https://isaac.download.nvidia.com/isaac-ros/release-3/dists/jammy/release-3.0/binary-amd64/Packages
- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/Packages.gz
- https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.8.0.html

## 相机桥接脚本区分

- `zed_bridge.py`：ZED 专用相机桥，输入 BGRA8 / 32FC1 米。
- `d435_bridge.py`：RealSense D435/D435i 相机桥，支持 RGB/BGR/RGBA/BGRA；16UC1 默认毫米转米，32FC1 按米处理。复用 ZED 脚本中的同步、内参缩放和掩码选择实现。
- `seed_tracking_bridge.py`：离线种子及追踪反馈桥，与上述相机格式适配桥不同；实时相机不能直接沿用录像首帧位姿。

D435i 驱动启动：

```bash
source /opt/ros/humble/setup.bash
source ~/realsense_ws/install/setup.bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

另一个宿主机终端启动格式适配：

```bash
source /opt/ros/humble/setup.bash
python3 ~/code/isaac_ros32_test/d435_bridge.py
```

ZED 对应命令为 `python3 ~/code/isaac_ros32_test/zed_bridge.py`。两者都发布 `/isaacfp/...`，一次只运行一个相机桥，不能同时混用。D435 默认相机 namespace/name 都是 camera；更改驱动名称后要通过三个 topic 参数指定对应话题。必须使用对齐到彩色的深度，初始掩码仅用于首次定位，不能当成逐帧分割。桥接不是完整实时定位/追踪/画框程序；新增 D435 支持目前仅做语法检查，尚未运行相机验证。
