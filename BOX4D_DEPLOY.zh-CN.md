# 大小方块识别、水平面跟随与新电脑部署

本说明部署的是 **D435i → Box4D 识别/跟踪 → ROS 2 C++ 底盘控制**。目标环境为 Ubuntu 22.04 x86_64、ROS 2 Humble、系统 Python 3.10。大小块均可在 CPU 上运行；无需安装 FoundationPose、SAM 或 CUDA。其他系统/ARM 电脑需要匹配其 ROS 和 PyTorch 环境，下面的命令没有在这些平台验收。

## 1. 行为与坐标

| 参数 | 行为 |
|---|---|
| `size:=big` | 只识别、跟踪和控制 35 cm 大块；使用原几何识别 |
| `size:=small` | 只识别、跟踪和控制 20 cm 小块；默认使用随仓库提供的 YOLO 权重＋RGB-D 位姿 |
| `target_distance_m:=0.30` | 目标中心位于相机水平前方 0.30 m |
| `target_lateral_m:=0.0` | 目标中心的水平左侧偏移为 0 m；正值左侧，负值右侧 |

两种尺寸使用同一个控制器：先左右对齐，再调整前后距离，到位停车，偏离后恢复跟随。仅控制底盘水平面 XY，不控制高度，也不旋转底盘。识别结果仍保留完整三维位置、旋转四元数、尺寸、颜色/朝向分析和跟踪 ID；“忽略高度”只作用于控制误差。

相机光学坐标是 X 向右、Y 向下、Z 向前。程序用本次静止 IMU 标定的 UP 构造水平前方 `f` 和水平左侧 `l`，对完整位置 `p` 计算 `(p·f, p·l)`。沿 UP 的高度变化不会影响控制。目标距离指**中心的水平距离**，不是光学 Z，也不是前表面距离。相机应固定在底盘上，光轴的水平投影与底盘前进方向一致；UP 无法测量安装偏航。

`size` 是启动参数，切换时结束上一轮再重新启动。控制端还会检查 `bbox.size` 三轴是否在所选边长 ±1 cm 内；锁定后不自动换 ID，目标丢失或观测超时停车。启动参数默认为 `big`、`dry_run=true`。

## 2. 安装系统依赖

先按 [ROS 2 Humble 官方 Ubuntu 安装说明](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html) 完成 ROS 软件源配置并安装 Humble。以下命令在软件源已配置后执行：

```bash
sudo apt update
sudo apt install -y git build-essential cmake libeigen3-dev \
  python3-venv python3-pip python3-colcon-common-extensions python3-pytest \
  python3-numpy python3-scipy python3-opencv \
  ros-humble-ros-base ros-humble-rclcpp ros-humble-rclpy \
  ros-humble-rcl-interfaces ros-humble-geometry-msgs ros-humble-vision-msgs \
  ros-humble-std-srvs ros-humble-rosgraph-msgs ros-humble-cv-bridge \
  ros-humble-tf2-ros ros-humble-launch-ros ros-humble-launch-xml \
  ros-humble-turtlesim ros-humble-ament-cmake-gtest ros-humble-ament-cmake-pytest \
  ros-humble-realsense2-camera ros-humble-realsense2-camera-msgs \
  ros-humble-realsense2-description
source /opt/ros/humble/setup.bash
ros2 pkg prefix realsense2_camera
ros2 interface show realsense2_camera_msgs/msg/RGBD
```

RealSense 包的安装方式参见 [官方 ROS Wrapper 安装说明](https://github.com/realsenseai/realsense-ros#installation)。如果驱动启动时报 SDK/内核或 USB 错误，按该文档为目标电脑安装匹配的 SDK/驱动。D435i 的 RGB-D、IMU 与 TF 都需要正常工作。

## 3. 拉取代码和权重

```bash
git clone https://github.com/TOMATO-wyq/FoundationPose.git "$HOME/6Dpose"
cd "$HOME/6Dpose"
sha256sum src/models/tkk_small/best.pt
```

权重已直接纳入 Git，约 5.8 MB，无需 Git LFS。校验值应为：

```text
6e3b4283239fb9299ea956db2a2f70d27cff7d4be8f1a109569766baeca57aed
```

已有 clone 时先 `git pull --ff-only`，再重新编译。不要复制旧电脑的 `build/`、`install/`、虚拟环境或 `.so` 文件；这些目录不在 Git 中，应在目标电脑生成。

## 4. Python 识别环境

仅运行大块几何识别可使用前面安装的系统 Python 依赖。要使用默认小块 YOLO 流程，建立能访问 ROS 系统包的虚拟环境：

```bash
cd "$HOME/6Dpose"
/usr/bin/python3 -m venv --system-site-packages .venv-box4d
.venv-box4d/bin/python -m pip install --upgrade pip
.venv-box4d/bin/python -m pip install \
  torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.venv-box4d/bin/python -m pip install -r src/requirements/box4d-yolo.txt
export BOX4D_PYTHON="$PWD/.venv-box4d/bin/python"
source /opt/ros/humble/setup.bash
"$BOX4D_PYTHON" -c 'import rclpy, cv_bridge, realsense2_camera_msgs, vision_msgs, cv2, scipy; print("ROS and vision imports OK")'
"$BOX4D_PYTHON" -c 'from ultralytics import YOLO; m=YOLO("src/models/tkk_small/best.pt", task="segment"); print(m.task, m.names)'
```

最后应看到 `segment {0: 'tkk'}`。依赖版本来自本项目已有运行环境，CPU wheel 的安装方式见 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v271)。这里固定 NumPy 1.x，避免与 Humble 的二进制图像桥接不兼容。无需激活 venv；控制节点及 colcon 继续使用系统 ROS，只有识别脚本使用 `BOX4D_PYTHON`。

原始纯几何配置仍可用于小块：显式传 `config:=src/config/box4d_d435i_live.json`。它保留原来的颜色/朝向分析，但不使用神经网络分割。

## 5. 编译与无硬件测试

```bash
cd "$HOME/6Dpose"
source /opt/ros/humble/setup.bash
colcon build --base-paths src_control --packages-select box_pose_control \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DBUILD_TESTING=ON
source install/setup.bash
colcon test --base-paths src_control --packages-select box_pose_control
colcon test-result --test-result-base build/box_pose_control/test_results --verbose
PYTHONPATH="$PWD/src:$PYTHONPATH" /usr/bin/python3 -m pytest -q \
  src/tests/test_box4d_size_selection.py src/tests/test_box4d_ros_output.py
```

串口测试使用伪终端，不连接底盘。可选 C++ 识别内核可用 `bash src/build_box4d_native.sh` 编译；不编译时 `geometry_backend=auto` 使用 NumPy。默认流程不需要 PCL。

## 6. 一条命令启动

每次新开终端：

```bash
cd "$HOME/6Dpose"
source /opt/ros/humble/setup.bash
source install/setup.bash
export BOX4D_PYTHON="$PWD/.venv-box4d/bin/python"
```

只有大块且未建立 venv 时，改为 `export BOX4D_PYTHON=/usr/bin/python3`。

大块预览（相机＋识别＋控制，默认不打开底盘串口）：

```bash
ros2 launch box_pose_control real_pipeline.launch.py \
  repo_root:="$PWD" size:=big dry_run:=true with_view:=false
```

小块预览（自动加载仓库权重，保留原朝向计算）：

```bash
ros2 launch box_pose_control real_pipeline.launch.py \
  repo_root:="$PWD" size:=small dry_run:=true with_view:=false \
  target_distance_m:=0.50 target_lateral_m:=0.05
```

上例目标为相机水平前方 50 cm、左侧 5 cm。首次约 2 秒保持相机和底盘静止，等待 UP 标定。相机安装角度改变后重新启动标定。`display:=false with_view:=false` 可禁用所有显示窗口；要看海龟平面视图用 `with_view:=true`。USB 2 连接可加 `camera_profile:=usb2`。默认运行时长为 3600 秒，可传 `duration:=...`。

已有同 domain 的相机驱动时加 `start_camera:=false`，不要重复启动。默认 ROS domain 为 86；复用其他 domain 的驱动时传 `domain:=对应编号`。外部独立识别应使用同样的 `--size big/small`、`--auto-up --publish-poses`，控制 launch 加 `start_recognition:=false`。

## 7. 连接底盘

串口访问需当前用户属于 `dialout` 组：

```bash
sudo usermod -aG dialout "$USER"
```

注销并重新登录后，查看 `ls -l /dev/serial/by-id/`，优先使用稳定的设备路径。启动实车：

```bash
ros2 launch box_pose_control real_pipeline.launch.py \
  repo_root:="$PWD" size:=small dry_run:=false with_view:=false \
  serial_port:=/dev/ttyUSB0 target_distance_m:=0.50 target_lateral_m:=0.0
```

串口为 8N1，默认 `961200` 波特率，采用新版 **10 字节**协议：`A5 + int16_le(vx*1000) + int16_le(vy*1000) + int16_le(0) + mode(0) + CRC8 + FF`。CRC8 多项式 `0x07`、初值 0，只覆盖速度和 Mode 的 7 字节。波特率未从参考代码中获得新值；若电控另有配置，修改 `src_control/box_pose_control/config/control.yaml` 后重新 build。

调试终端需 `source` 同样的环境并 `export ROS_DOMAIN_ID=86`：

```bash
ros2 topic echo /box_pose_control/cmd_vel
ros2 param set /box_pose_control target_distance_m 0.50
ros2 param set /box_pose_control target_lateral_m -0.05
ros2 service call /box_pose_control/set_enabled std_srvs/srv/SetBool '{data: false}'
ros2 service call /box_pose_control/set_enabled std_srvs/srv/SetBool '{data: true}'
```

INFO 的 `size`、`id` 指明当前规格/目标；`plane_x/plane_y` 是水平前方/左侧距离，`ex/ey` 是相对目标点的误差。`serial=sent` 只表示本次写入成功。关闭、退出、目标丢失或超时会发送零速；识别进程重启导致 ID 改变时，关闭再开启控制以重新选目标。

## 8. 常见部署问题

- `No module named rclpy` / `cv_bridge`：使用系统 Python 3.10 创建 `--system-site-packages` venv，并先 source Humble；不要用不同 Python 版本的 Conda 环境。
- 找不到权重：检查 Git 文件和上述 SHA-256；默认配置里的路径相对 JSON 文件目录解析，CLI `--yolo-weights` 则相对当前工作目录。
- `waiting_pose`：确认识别用了 `--publish-poses --auto-up`，两端 domain 一致，UP 标定成功，时间戳/光学 frame 一致。
- `no_target`：确认 `size` 与真实边长相符；默认只接受 35 cm 或 20 cm 规格。
- `invalid_geometry`：UP 无效，或光轴几乎竖直，无法稳定定义水平前方。
- 串口错误：确认设备路径、用户组、波特率及设备未被其他节点独占。

本机自动测试覆盖尺寸筛选、原位姿发布、水平投影、倾斜/不同高度、目标丢失停车和串口帧。新电脑的相机吞吐、现场朝向精度、底盘速度符号和到位误差仍需实物验收。
