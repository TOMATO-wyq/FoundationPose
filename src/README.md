# Box4D 与 YOLO 工具目录

当前方块识别和训练代码按下面的目录分类。先看这里选入口，再进入对应目录；FoundationPose/KFS 在独立的 [src_fp](../src_fp/README.md) 目录。

```text
src/
├── box4d/                识别核心：分割、找面、位姿、跟踪、debug
│   └── cli/             实时识别、录制、回放、UP 标定、性能测试
├── common/              与 FP 共用的显示和静止 IMU 数学
├── native/              CPU C++ 图像、平面、边线与优化计算
├── pcl/                 可选 PCL 平面分割
├── tools/yolo/          数据准备、训练、负样本采集、模型对比
├── scripts/box4d/       ROS、相机、C++/PCL 编译脚本
├── scripts/yolo/        训练数据采集和分割测试启动脚本
├── config/              相机与识别参数
├── requirements/        当前识别程序的 Python 依赖
├── tests/               回归测试
└── docs/
    ├── box4d/           运行说明、debug 中文对照、代码结构
    └── yolo/            分割训练、测试、模型对比说明
```

顶层保留常用 `.sh` 命令作为兼容入口，实际实现都在 `scripts/`。例如原来的 `bash src/box4d_ros.sh ...`、`bash src/build_box4d_native.sh` 仍可直接使用。

| 常用任务 | 脚本 / 文档 |
|---|---|
| 相机启动 | `scripts/box4d/camera.sh` |
| 当前小块 YOLO + 几何识别 | [YOLO-seg 说明](docs/box4d/BOX4D_YOLO_SEG.zh-CN.md) |
| C++ 内核编译 | [代码结构与 C++](docs/box4d/BOX4D_代码结构与CPP.zh-CN.md) |
| 录制、回放和离线基准 | [CPU 使用说明](docs/box4d/BOX4D_CPU_使用说明.zh-CN.md) |
| TKK 正样本采集 | `scripts/yolo/start_tkk_capture.sh`，见 [FP 采集章节](../src_fp/README.md#新红蓝模型的-fp-训练数据采集) |
| TKK 标注转换与训练 | `tools/yolo/prepare_tkk_yolo_seg.py` → `tools/yolo/train_tkk_seg.py` |
| TKK 负样本补拍 | `scripts/yolo/capture_tkk_background.sh` |
| 双模型实时对比 | `scripts/yolo/run_tkk_seg_compare.sh` · [对比说明](docs/yolo/TKK_SEG_COMPARE_中文.md) |

`tools/yolo/box_test_seg_d435i.py` 中的图像解码仍被 TKK 对比程序复用；`run_foundationpose_zed.py` 中的 FP 加载与绘图也被实时采集入口复用。

## 当前 CPU 方块识别入口

CPU 红蓝方块程序的两终端完整命令已放在 [仓库 README 开头](../README.md)：先运行 `bash src/start_box4d_camera.sh --domain 86 speed` 启动相机驱动，再运行 `bash src/box4d_ros.sh --domain 86 live ... --auto-up` 启动识别。已有相同配置的相机驱动时只启动识别。相机脚本会配好同步 RGB-D、对齐深度和加速度流；单独打开 RealSense Viewer 不会提供这套程序所需的 ROS 消息。

新训练的 `tkk` YOLO-seg 模型可以接入 20 cm 小块的实例 ROI 前端，35 cm 大块保留原有几何判断。默认全部使用 CPU、后台检测只保留最新待处理帧。完整命令、分割 debug 和参数说明见 [YOLO-seg 小块接入说明](docs/box4d/BOX4D_YOLO_SEG.zh-CN.md)。

详细参数见 [CPU 使用说明](docs/box4d/BOX4D_CPU_使用说明.zh-CN.md)。FoundationPose 三终端启动与投影采集见 [FP 运行指南](../src_fp/README.md)。
