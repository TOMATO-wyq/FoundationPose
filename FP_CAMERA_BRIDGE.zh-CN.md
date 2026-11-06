# 可选：相机桥接到原版 FP

**只想跑原版 demo 不需要本篇。** 本篇在[原版 FP 已跑通](FP_INSTALL_FROM_ZERO.zh-CN.md)的基础上，增加 D435i 实时 RGB-D 输入，先用手画 Mask 定位，不安装 YOLO / SAM。桥接脚本是本项目扩展，FP 仍使用你自行下载的 NVIDIA 原版代码。

## 1. 只获取桥接所需文件

向维护者获取 `fp_camera_bridge.tar.gz`，放在 `~/Downloads`，执行：

```bash
tar -xzf ~/Downloads/fp_camera_bridge.tar.gz -C ~/fp_workspace
```

目录如下，原版 `FoundationPose/` 不被覆盖：

```text
~/fp_workspace/
├── FoundationPose/          # 原版 FP + 官方权重，已完成 demo
├── src/                    # 桥接、手画 Mask 入口及共用辅助文件
├── models/                 # 自己实物的米制 OBJ / PLY 和纹理
└── outputs/                # 相机 socket 与输出
```

包中相机数据部分为 `camera_ros_source.py`、`live_rgbd_transport.py`、`kfs_roi.py`；手画入口为 `run_foundationpose_live.py`，并带有其导入的辅助文件。当前入口尚未彻底拆成独立库，因此包内有共用的分割参数定义和显示 / 计时代码，**手画路线不会加载 YOLO / SAM 权重，也不启用自动失跟恢复或动态 ROI**。不需要完整 KFS 模型、训练数据或实验资产包。

## 2. 安装 ROS2 Humble 和宿主机 Python 依赖

这一步让宿主机能启动相机并读取 ROS 图像。参考 [ROS2 Humble 官方安装说明](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)，以下只选择 Humble，不混装其他 ROS 发行版。

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y locales software-properties-common curl python3
sudo locale-gen en_US.UTF-8
export LANG=en_US.UTF-8
sudo add-apt-repository -y universe

fp_ros_apt_version=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | python3 -c 'import json,sys; print(json.load(sys.stdin)["tag_name"])')
curl -fL -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${fp_ros_apt_version}/ros2-apt-source_${fp_ros_apt_version}.jammy_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
sudo apt update
sudo apt install -y ros-humble-ros-base ros-dev-tools \
  ros-humble-cv-bridge ros-humble-message-filters python3-opencv python3-numpy
source /opt/ros/humble/setup.bash
ros2 pkg prefix rclpy
python3 -c 'import rclpy, cv_bridge, cv2, numpy; print("ROS Python OK")'
```

**成功标志：能找到 `rclpy`，并输出 `ROS Python OK`。** `source` 只对当前终端有效，后续新开终端要重新执行。此阶段使用系统 `python3`，不要激活 Conda 或另行用 pip 升级系统 NumPy。若系统更新要求重启，先重启，再重新 source ROS 环境。

## 3. 安装 RealSense SDK 与 ROS 驱动

本文为新机器选定一条路线：**源码安装 SDK 2.58.4，再编译固定版本 realsense-ros 4.58.4**。已有其他版本 SDK / apt 相机驱动时先确认安装来源，不把多套 SDK 混在一起。[RealSense 官方安装流程](https://github.com/realsenseai/realsense-ros/tree/9a11121700cb4780e273e34141f6402fe184321d#installation-on-ubuntu)

### 3.1 安装 SDK

先拔下相机。下面选择 SDK 的 RSUSB 后端，避免新手先处理内核补丁；这是本文选用的安装配置，不代表与历史实验的 SDK 后端完全一致。

```bash
sudo apt install -y build-essential cmake pkg-config libusb-1.0-0-dev \
  libudev-dev libssl-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev
mkdir -p ~/fp_deps
git clone --branch v2.58.4 --depth 1 https://github.com/realsenseai/librealsense.git ~/fp_deps/librealsense
cd ~/fp_deps/librealsense
bash scripts/setup_udev_rules.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DFORCE_RSUSB_BACKEND=ON -DBUILD_WITH_DDS=OFF \
  -DCHECK_FOR_UPDATES=OFF -DBUILD_EXAMPLES=ON \
  -DBUILD_GRAPHICAL_EXAMPLES=ON -DBUILD_PYTHON_BINDINGS=OFF
cmake --build build -j4
sudo cmake --install build
sudo ldconfig
```

SDK 编译选项见[官方定义](https://github.com/realsenseai/librealsense/blob/v2.58.4/CMake/lrs_options.cmake)，源码安装与 udev 设置见[官方指南](https://github.com/realsenseai/librealsense/blob/v2.58.4/doc/installation.md)。通过 USB 3 接回相机，执行 `realsense-viewer`，开启 Color 和 Depth，确认都有画面，**然后关闭 Viewer**，避免与 ROS 争用相机。编译内存不足时将 `-j4` 改为 `-j2`。

### 3.2 编译 ROS 相机驱动

```bash
source /opt/ros/humble/setup.bash
mkdir -p ~/realsense_ws/src
git clone --branch 4.58.4 --depth 1 https://github.com/realsenseai/realsense-ros.git ~/realsense_ws/src/realsense-ros
cd ~/realsense_ws
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi
rosdep update --rosdistro humble
rosdep install --from-paths src --ignore-src --rosdistro humble \
  --skip-keys librealsense2 -y
colcon build --symlink-install --packages-up-to realsense2_camera \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
source ~/realsense_ws/install/setup.bash
ros2 pkg prefix realsense2_camera
python3 -c 'from realsense2_camera_msgs.msg import RGBD; print("RGBD message OK")'
```

**成功标志：驱动路径位于 `~/realsense_ws/install`，并输出 `RGBD message OK`。** 后者是本项目单话题 RGB-D 输入所需的消息类型；只装好 SDK 不够。

## 4. 启动相机和桥接

三个终端中前两个在宿主机，第三个进入 FP 容器。先关闭 Viewer，接好 USB 3 相机。将真实目标对应的模型放在 `~/fp_workspace/models/object.obj`，带齐其引用纹理，确认坐标单位为米；模型必须与当前实物一致。

**终端 1：相机驱动。**

```bash
source /opt/ros/humble/setup.bash
source ~/realsense_ws/install/setup.bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true \
  enable_rgbd:=true enable_sync:=true align_depth.enable:=true \
  rgb_camera.color_profile:=640,480,30 \
  depth_module.depth_profile:=640,480,30
```

**终端 2：RGB-D 桥接。**

```bash
source /opt/ros/humble/setup.bash
source ~/realsense_ws/install/setup.bash
python3 ~/fp_workspace/src_fp/camera_ros_source.py --camera d435i \
  --scale 1 --socket "$HOME/fp_workspace/outputs/live_rgbd.sock"
```

看到 `RGB-D ready` 再启动消费端。桥接把对齐的彩色、米制深度和内参传给 FP；这里只取最新帧，不积压旧画面。

**终端 3：原版 FP 环境中的手画入口。** 若原版容器已退出，先在宿主机执行 `sudo docker start fp-upstream-demo`，再进入：

```bash
sudo docker exec -it fp-upstream-demo bash
```

在容器中执行：

```bash
cd /workspace/FoundationPose
python ../src_fp/run_foundationpose_live.py --camera d435i \
  --socket /workspace/outputs/live_rgbd.sock \
  --mesh /workspace/models/object.obj \
  --output /workspace/outputs/manual_live_01 \
  --mask-source manual --dynamic-roi 0 --save-poses
```

使用新的输出目录。鼠标左键沿目标可见轮廓点选、Enter 确认，首次定位时相机和目标保持静止；R 对新画面重新画 Mask，Q / Esc 退出。**成功标志：当前实物出现跟随的三维框，输出目录生成位姿 TXT。** 手画路线失跟后需要人工按 R 恢复。

可调：桥接 `--scale` 默认 0.5，本文设 1；评分显存不足时 FP 参数 `--score-batch-size` 从默认 32 降至 16。图像缩放必须同步作用于 RGB、深度和内参，不能单独缩放一项。

## 5. 只需要相机数据，不用本项目 FP 入口

保留前两个终端，在一个加载 ROS 环境的宿主机新终端执行下面的检查。它不运行 FP、不需要分割，也不操作相机硬件，只读取桥接提供的一帧：

```bash
PYTHONPATH="$HOME/fp_workspace/src" python3 - <<'PY'
import os, socket
from live_rgbd_transport import receive_frame
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
    connection.settimeout(10)
    connection.connect(os.path.expanduser('~/fp_workspace/outputs/live_rgbd.sock'))
    rgb, depth, K, metadata = receive_frame(connection)
    print('RGB:', rgb.shape, rgb.dtype)
    print('Depth:', depth.shape, depth.dtype, 'metres')
    print('K:', K)
PY
```

该桥接一次只服务一个客户端；FP 和上述检查不能同时连接。自写程序可使用同一接口：`rgb` 为 H×W×3 的 RGB uint8，`depth` 为同尺寸 float32 米制深度，`K` 为对应 3×3 内参；全图布尔 Mask 可送入 FP 的 `register`。ROS 通信与 socket 搬运属于本篇，目标分割见[独立分割篇](FP_SEGMENTATION.zh-CN.md)。

本教程复用了已有桥接实现；“独立下载原版 + 此文件子集”的组合尚未在新机器完成相机 / GPU 验证。不同版本不混用 socket 协议文件。ZED 相机需另外安装其 SDK 和驱动，不属于本文 D435i 安装必需项。
