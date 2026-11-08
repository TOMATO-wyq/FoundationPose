# 方块位姿识别、TKK 训练与 FoundationPose

> 大小方块分开识别、底盘水平面 XY 跟随、新版串口协议和跨电脑部署：见 [Box4D 部署说明](BOX4D_DEPLOY.zh-CN.md)。小块 YOLO 权重已随仓库提供，启动使用 `size:=big` 或 `size:=small`。

代码按工程分在两个目录：[src/](src/README.md) 放 Box4D 与 YOLO 工具，[src_fp/](src_fp/README.md) 放 FoundationPose/KFS 和相机桥接。各目录的 README 都有中文导航。

独立的 [ROS 2 C++ 底盘控制模块](src_control/box_pose_control/README.md) 位于 `src_control/`：使用 Box4D 相机位姿保持方块中心在相机前约 30 cm，默认不开串口，以 10 Hz 输出速度预览和 INFO。识别命令增加 `--auto-up --publish-poses` 后即可连接。

当前入口集中在 BOX4D 实时识别、TKK 数据采集与训练。最新的小块 YOLO 前端见 [YOLO-seg 使用说明](src/docs/box4d/BOX4D_YOLO_SEG.zh-CN.md)，CPU C++ 内核见 [代码结构与编译](src/docs/box4d/BOX4D_代码结构与CPP.zh-CN.md)。下面的两步命令使用不带 YOLO 的几何配置；使用新训练的 TKK 权重时，按 YOLO-seg 说明选择配置和 Python 环境。

## CPU 方块识别：D435i 两步启动

当前红蓝方块 CPU 程序使用两个终端：**相机驱动 → 识别程序**。接好相机后，用下面的脚本启动 ROS 相机驱动；单独打开 RealSense Viewer 只能看相机画面，不能给这个识别程序提供所需的 ROS 数据。启动前关闭占用相机的 Viewer 或其他相机驱动。

下面使用 `speed` 请求彩色和深度均为 640×480@60，需要 USB 3 连接；稳定采集和识别 60 FPS 尚需实测。

**终端 1：启动相机驱动，保持该终端运行。**

```bash
cd /home/tomato/6Dpose
bash src/start_box4d_camera.sh --domain 86 speed
```

该脚本配置同步 RGB-D、深度对齐到彩色图像、内参和加速度流，并防止重复启动相机驱动。启动相机本身不会打开识别窗口。三个采集档位为：`speed` = 640×480@60，`precision` = 1280×720@30，`usb2` = 640×480@30；档位名称不会改变硬件 USB 协商速度。

**终端 2：启动识别与预览。**

```bash
cd /home/tomato/6Dpose
bash src/box4d_ros.sh --domain 86 live \
  --config src/config/box4d_d435i_live.json \
  --auto-up \
  --output "outputs/box4d_live_$(date +%Y%m%d_%H%M%S)" \
  --duration 600 --display --display-smoothed --display-hz 30
```

- 已有上述脚本启动的相机驱动时，只执行终端 2；重启识别程序可以继续复用相机驱动。切换采集档位时，先在终端 1 按 `Ctrl+C`，再按新档位启动。
- `--auto-up` 在启动时用约 2 秒测量当前竖直方向，这段时间相机保持静止；运行中调整相机后，重新启动识别。
- 识别窗口按 `Q` / `Esc` 退出，也可在终端 2 按 `Ctrl+C`。`--duration 600` 表示最多运行 10 分钟；每次结果保存在带时间戳的新目录。
- `--display-hz 30` 只限制预览提交频率，不把相机或识别处理限制为 30 FPS。需要观察更高刷新率可改为 60，实际速率以运行统计为准。

相机和识别必须使用相同的 ROS domain。上面两条命令都指定 86；已有驱动在默认 domain 0 时，直接把识别命令改为 `--domain 0`，无需重启驱动或另开第二个驱动。相机脚本未传 `--domain` 时沿用环境变量 `ROS_DOMAIN_ID`，未设置则为 0；脚本会打印实际 domain。

已有其他 ROS 相机驱动时，需要在对应 domain 中正确提供 `/camera/camera/rgbd`（同步、对齐深度、彩色内参）、`/camera/camera/accel/sample` 和相应 TF 才能复用。本机直接使用上述相机脚本即可。

详细参数和数据格式见 [CPU 使用说明](src/docs/box4d/BOX4D_CPU_使用说明.zh-CN.md)。本流程直接订阅 ROS 相机消息；下文 FoundationPose 的安装与三终端启动属于另一套流程。

漏检、双面视角或识别框闪断时，在识别命令后增加 `--debug`，可查看八个识别阶段和一个实时参数滑块窗口。滑块可调深度、Canny、深度边缘、PCL 平面、面校验和跟踪门限；按 S 保存当前配置到启动配置文件及本次输出目录，并保存调试快照。详见 [逐阶段调试说明](src/docs/box4d/BOX4D_DEBUG.zh-CN.md)。

平面分割也可试用 PCL：先运行 `bash src/build_box4d_pcl.sh`，再把实时配置换成 `src/config/box4d_pcl_live.json`。默认平面流程为 NumPy，数值内核可自动使用已编译的 C++ 扩展。后端对比工具为 `src/box4d/cli/compare_backends.py`，运行参数见其 `--help`；调试命令见 [逐阶段调试说明](src/docs/box4d/BOX4D_DEBUG.zh-CN.md)。

## 常用功能与文档

| 功能 | 入口与说明 |
|---|---|
| TKK 小块 YOLO-seg + RGB-D 位姿 | [运行与参数](src/docs/box4d/BOX4D_YOLO_SEG.zh-CN.md) |
| C++ 内核编译、代码结构与基准测试 | [代码结构与 C++](src/docs/box4d/BOX4D_代码结构与CPP.zh-CN.md) |
| 几何识别、录制与离线回放 | [CPU 使用说明](src/docs/box4d/BOX4D_CPU_使用说明.zh-CN.md) |
| 调试窗口与滑块 | [逐阶段调试](src/docs/box4d/BOX4D_DEBUG.zh-CN.md) · [滑块中文对照表](src/docs/box4d/BOX4D_滑块中文对照表.zh-CN.md) |
| TKK 正样本采集、标注转换与训练 | [采集与训练](src_fp/README.md#新红蓝模型的-fp-训练数据采集) |
| TKK 负样本补拍与双模型对比 | [对比说明](src/docs/yolo/TKK_SEG_COMPARE_中文.md) |

相机安装方向改变后，用 `--auto-up` 重新测量竖直方向。尺寸和位姿绝对精度仍需实物验收。多层堆叠与红蓝双色塔顶的样例说明见 [样例分析与验证](方块堆叠情况/识别方案与验证.zh-CN.md)。

## FoundationPose 安装与运行

先安装 NVIDIA 原版 FP，再按需加入相机与分割：

| 需求 | 文档 |
|---|---|
| 安装原版、跑官方 demo | [原版 FP 从零安装](FP_INSTALL_FROM_ZERO.zh-CN.md) |
| 给原版接 D435i，手画 Mask | [可选相机桥接](FP_CAMERA_BRIDGE.zh-CN.md) |
| 用 YOLO / SAM 生成目标 Mask | [可选自动分割](FP_SEGMENTATION.zh-CN.md) |
| 安装 FP + YOLO / SAM 环境 | [完整安装](FP_KFS_FULL_INSTALL.zh-CN.md) · [当前运行指南](src_fp/README.md) |
| 迁移到 Jetson Xavier NX | [板端 Codex 部署交接](JETSON_XAVIER_NX_HANDOFF.zh-CN.md) |

已有完整环境的启动命令见 [启动指南](src_fp/README.md)，可选自动分割的接口与验证说明见 [KFS 流程说明](src_fp/docs/KFS_PIPELINE.md)。
