# 完整 KFS 实验环境安装（可选）

本文适用于安装本项目 D435i + YOLO / SAM + FP 环境。只用原版 FP 请看[原版安装](FP_INSTALL_FROM_ZERO.zh-CN.md)，只接相机请看[相机桥接](FP_CAMERA_BRIDGE.zh-CN.md)。按本文安装后，使用[当前运行指南](src/README.md)中的相机、采集与推理入口。安装步骤更新日期：2026-10-17。

**适用环境：原生 Ubuntu 22.04 桌面版、x86_64、NVIDIA 显卡、D435i 和 USB 3 数据线。** 本项目实测为 RTX 5060 Laptop 8GB；当前 Dockerfile 的 CUDA 扩展编译目标固定为 `12.0`，不是适用于所有显卡的通用镜像。其他架构需修改 `FoundationPose/docker/Dockerfile.cu130` 的 `TORCH_CUDA_ARCH_LIST` 后重建并验证；Windows、WSL、macOS、Jetson 不在本文安装路线内。

不要求提前安装 ROS、Docker、CUDA Toolkit 或 Python 虚拟环境。**FP 的 CUDA / Torch 在容器内安装，相机与 ROS 在宿主机安装。** 以下未特别注明的命令都在 Ubuntu 自带终端、普通用户下执行；`sudo` 要求输入该用户密码。命令按顺序执行，报错后先处理，不要继续跳步。

安装顺序：**显卡驱动 → 项目代码 → Docker GPU 支持 → ROS → 相机驱动 → 权重和模型 → FP 镜像与检查 → 实时运行**。需要联网下载较大的镜像、源码与权重，磁盘应预留数十 GB；这是容量建议，不是测得的最低配置。

## 1. 检查系统，安装显卡驱动

```bash
cat /etc/os-release
uname -m
nvidia-smi
```

前两项应显示 Ubuntu 22.04 和 `x86_64`。第三项应列出 NVIDIA 显卡及驱动版本；本 CUDA 13 路线需要 R580 或更新驱动作为兼容性起点，最终以第 7 步实际 GPU 检查为准。[NVIDIA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

如果 `nvidia-smi` 不存在或失败，先安装驱动。新机器可执行：

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices
sudo ubuntu-drivers install
sudo reboot
```

重启后重新运行 `nvidia-smi`。若推荐驱动低于 R580，或启动时要求 Secure Boot / MOK 注册，先按系统提示及 [Ubuntu 驱动说明](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/)完成处理；显卡检查通过后再继续。已有可用且满足版本要求的驱动无需重装。

## 2. 获取本项目代码

```bash
sudo apt update
sudo apt install -y git curl ca-certificates python3
git clone https://github.com/TOMATO-wyq/FoundationPose.git ~/6Dpose
cd ~/6Dpose
ls src_fp/scripts/start_kfs_fp.sh FoundationPose/docker/install_host.sh
```

应能列出这两个文件。这里需要的是**本实验项目**，只下载 NVIDIA 原版 FP 仓库不会包含自动分割与相机桥接入口。如果仓库需要权限，向维护者获取访问权限或完整代码包，解压后的结构也应满足上述检查。已有 `~/6Dpose` 时不要重复克隆或覆盖，直接检查文件。

请使用维护者交付的最新代码（含本文及 `cuda130.sh` 的用户组修复）；仅拿旧版本文档不能补齐旧脚本。代码中无需把 `/home/tomato` 替换成自己的用户名，后面会显式设置必要的外部路径。

## 3. 安装 Docker 和 GPU 容器支持

新机器执行：

```bash
cd ~/6Dpose
bash FoundationPose/docker/install_host.sh
```

该脚本安装 Docker Engine、Buildx 和 NVIDIA Container Toolkit，并运行测试；**它不安装 ROS、相机驱动、FP 权重或 FP 镜像**。它会配置 apt 软件源和重启 Docker，已有服务的机器需先安排停机。若提示软件包冲突，保留现有服务，按 [Docker 安装说明](https://docs.docker.com/engine/install/ubuntu/)与 [NVIDIA Container Toolkit 安装说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)处理，不能直接删除不认识的容器环境。

已有 Docker 可先跳过安装，用下面命令检查：

```bash
sudo docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu22.04 nvidia-smi
```

**成功标志：容器也能列出同一块显卡。** 若出现 GPU device driver / runtime 错误，先完成 Toolkit 配置；没有通过本项时不用尝试构建 FP。

## 4. 安装 ROS2 Humble 和宿主机 Python 依赖

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
  ros-humble-cv-bridge python3-opencv python3-numpy
source /opt/ros/humble/setup.bash
ros2 pkg prefix rclpy
python3 -c 'import rclpy, cv_bridge, cv2, numpy; print("ROS Python OK")'
```

**成功标志：能找到 `rclpy`，并输出 `ROS Python OK`。** `source` 只对当前终端有效，后续新开终端要重新执行。此阶段使用系统 `python3`，不要激活 Conda 或另行用 pip 升级系统 NumPy。若系统更新要求重启，先重启，再重新 source ROS 环境。

## 5. 安装 RealSense SDK 与 ROS 驱动

本文为新机器选定一条路线：**源码安装 SDK 2.58.4，再编译固定版本 realsense-ros 4.58.4**。已有其他版本 SDK / apt 相机驱动时先确认安装来源，不把多套 SDK 混在一起。[RealSense 官方安装流程](https://github.com/realsenseai/realsense-ros/tree/9a11121700cb4780e273e34141f6402fe184321d#installation-on-ubuntu)

### 5.1 安装 SDK

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

### 5.2 编译 ROS 相机驱动

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

## 6. 放置权重和三维模型

本文配套的实验资产包名为 **`fp_kfs_assets.tar.gz`**，校验文件为 `fp_kfs_assets.tar.gz.sha256`。从维护者获取并放入 `~/Downloads`；本次已在维护者工作区生成，**尚无公共下载地址，需要随教程提供**。

```bash
cd ~/Downloads
sha256sum -c fp_kfs_assets.tar.gz.sha256
tar -xzf fp_kfs_assets.tar.gz -C ~/6Dpose
cd ~/6Dpose
python3 - <<'PY'
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('asset_manifest.json').read_text())
for item in manifest['files']:
    path = Path(item['path'])
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == item['sha256'], str(path)
print('FP / YOLO / Mesh assets OK')
PY
```

**成功标志：压缩包校验为 `OK`，逐文件检查输出 `FP / YOLO / Mesh assets OK`。** 资产包中包括：

```text
6Dpose/
├── FoundationPose/weights/
│   ├── 2023-10-28-18-33-37/{config.yml, model_best.pth}
│   └── 2024-01-11-20-02-45/{config.yml, model_best.pth}
├── weights/kfs_yolo_seg.pt
├── kfs_model/BlueTrueKFS13/{BlueTrueKFS13.obj, BlueTrueKFS13.mtl, BlueTrueKFS13.png}
└── asset_manifest.json
```

FP 权重也可从[官方数据准备入口](FoundationPose/README.zh-CN.md#数据准备)下载；YOLO 权重和当前箱体 Mesh 属于本实验材料，不能拿其他模型随意替代。当前模型为米制、边长 0.35m，并非任意大小的蓝色箱子都适用。

## 7. 构建 FP 环境并检查

### 7.1 准备构建输入

当前仓库的基础镜像包含 ZED 支持，**即使本实验只接 D435i，源码构建仍需要以下 ZED 文件**。这是现有构建的依赖，不需要额外购买或连接 ZED 相机。

从 [ZED SDK 下载页](https://www.stereolabs.com/developers/release)选择 **5.5.0 / Ubuntu 22 / CUDA 13 / TensorRT 11**，保存到 `~/Downloads`，文件名应为：

```text
ZED_SDK_Ubuntu22_cuda13.0_tensorrt11.0_v5.5.0.zstd.run
```

只下载，**不在宿主机运行安装器**。如果官网后续默认显示新版，选择旧版本或从维护者获取同版本文件，不要用别的版本改名冒充。

```bash
git clone --branch v5.5.0 https://github.com/stereolabs/zed-ros2-wrapper.git ~/fp_deps/zed-ros2-wrapper
git -C ~/fp_deps/zed-ros2-wrapper checkout 431dcf4b1ea39caf894ca6ea294437437dbdb034
export ZED_SDK_INSTALLER="$HOME/Downloads/ZED_SDK_Ubuntu22_cuda13.0_tensorrt11.0_v5.5.0.zstd.run"
export ZED_ROS2_SOURCE="$HOME/fp_deps/zed-ros2-wrapper"
test -s "$ZED_SDK_INSTALLER" && echo "ZED installer found"
```

### 7.2 构建基础镜像，编译本机扩展

保持上面的终端继续执行：

```bash
cd ~/6Dpose
bash FoundationPose/docker/cuda130.sh build
bash FoundationPose/docker/cuda130.sh check
```

**成功标志：生成 `foundationpose:rtx50-cu130` 镜像，检查中 GPU kernel、`mycpp` 与 `estimater` 导入全部通过。** `check` 同时会在项目目录编译 C++ 扩展，因此即使导入维护者提供的基础镜像，也必须执行一次 `check`；不要复制别人机器上已有的 `mycpp/build`。

`check_env.py` 遇到缺失权重会给 WARNING，退出码仍可能为 0；所以第 6 步的资产检查不能省略。

### 7.3 构建 YOLO 镜像，检查模型加载

```bash
cd ~/6Dpose
sudo docker build --network=host -t foundationpose:kfs-yolo \
  -f src_fp/Dockerfile.kfs-yolo src_fp
sudo docker run --rm --gpus all --shm-size=4g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp/matplotlib \
  -v "$HOME/6Dpose:/workspace/6Dpose" \
  -w /workspace/6Dpose/FoundationPose foundationpose:kfs-yolo \
  python -c 'from learning.training.predict_pose_refine import PoseRefinePredictor; from learning.training.predict_score import ScorePredictor; from ultralytics import YOLO; PoseRefinePredictor(); ScorePredictor(); m=YOLO("../weights/kfs_yolo_seg.pt"); assert m.task=="segment"; assert "blue_kfs" in m.names.values(); print("FP and YOLO models OK")'
```

**成功标志：输出 `FP and YOLO models OK`。** 这验证模型加载，不代表已经完成相机位姿推理；下一步才检验整条链路。这里显式构建 YOLO 镜像是为了把安装问题提前暴露，平时启动器也会在镜像不存在时自动构建。

## 8. 启动实时实验

回到[当前 D435i 启动指南](src/README.md#d435i-全流程启动当前版本)，按三个终端顺序启动：

1. 相机驱动：开启彩色、深度、同步与深度对齐。
2. RGB-D 数据源：看到 `RGB-D ready`。
3. YOLO + FP：完成首次定位，显示跟随当前箱体的三维框，并生成运行目录。

第一次定位时固定相机和目标；按 R 可重新定位，Q / Esc 退出。若本地桌面没有可用 DISPLAY / X11，使用 `--debug 0 --no-display`，通过位姿和日志检查，不要同时传 `--debug 1`。

## 安装遇到问题，按失败层级排查

| 首个失败位置 | 先处理什么 |
|---|---|
| `nvidia-smi` | 宿主机显卡驱动、重启、Secure Boot；先不处理 FP |
| Docker GPU 测试 | Docker 服务、Container Toolkit、驱动兼容性 |
| apt / Git / pip 下载 | 网络、代理、仓库访问权限；下载失败不代表代码编译失败 |
| Viewer 无相机 / 无深度 | USB 3 线缆与端口、udev 权限、相机占用、SDK 安装 |
| `RGBD` 导入失败 | source ROS 与相机工作空间；检查驱动版本和编译结果 |
| `No kernel image` / CUDA 架构错误 | 当前 GPU 是否匹配 Dockerfile 的 `12.0` 编译目标；重新选择架构构建 |
| `ZED SDK installer not found` | SDK 文件是否完整、两个 `export` 是否在当前终端执行 |
| `mycpp` 导入 / 编译失败 | 执行 `cuda130.sh check`，不要复用旧机器的编译目录 |
| 权重 / Mesh 缺失 | 第 6 步解压路径和校验，模型配置与 checkpoint 均需完整 |

本文依据项目源码与官方安装说明补齐，**尚未在另一台干净机器上完成全流程复装验证**；当前实测结果见主指南。传播前维护者需交付包含本文修改的代码及配套资产包；公开分发时保留各组件许可并确认材料的分发权限。
