# 给 Xavier NX 端 Codex 的 FP 部署交接

日期：2026-10-20。用户已确认目标为 **Jetson Xavier NX，不是 Orin NX**。目标是在该设备上部署 FoundationPose、D435i，并复现当前蓝色 KFS 的分割、深度修正、跟踪和恢复逻辑。本文是迁移交接，不是已在 NX 验证的安装脚本。

## 1. 先核对板端环境，按 Xavier 选依赖

请先读取并记录以下信息，再决定下载哪些依赖，不沿用桌面端版本号：

```bash
tr -d '\0' < /proc/device-tree/model
uname -m
cat /etc/os-release
cat /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core nvidia-jetpack
python3 --version
nvcc --version
free -h
df -h /
```

某项命令或元包不存在时记录实际情况，不用因此推断设备没有 JetPack。补查内存容量、可用磁盘、容器运行时、已装 Torch / torchvision、实际 CUDA、ROS 版本及相机 USB 连接情况。

- Xavier NX 为 aarch64，GPU 计算能力 **7.2**。CUDA 扩展应按实际设备编译，不能沿用桌面端 `TORCH_CUDA_ARCH_LIST=12.0`。[NVIDIA 设备表](https://developer.nvidia.com/cuda/gpus/legacy)
- **不能使用这里的 CUDA 13 环境。** CUDA 13 已移除 Volta 的离线编译和相关库支持，修改架构参数并不能解决这个问题。[CUDA 13 发布说明](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#deprecated-architectures)
- 先围绕现有 JetPack / L4T 选择受支持的 GPU PyTorch、torchvision 和容器。NVIDIA 要求 Jetson PyTorch 与指定 JetPack 匹配；不要先用普通 pip 包覆盖板端 GPU Torch。[Jetson PyTorch 安装说明](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
- JetPack 5.1.5 是支持 Xavier NX 的一个官方例子，带 Ubuntu 20.04、L4T 35.6.1、CUDA 11.4.19；这不代表当前设备就是该版本，也不表示需要刷成它。[官方版本说明](https://developer.nvidia.com/embedded/jetpack-sdk-515)
- 不执行本仓库桌面端 `install_host.sh`、`migrate_cuda130.sh`，不直接复用 `Dockerfile.cu130` 或 `setup/Dockerfile.fp-cu130`。不要按桌面 Ubuntu 的方法安装独立 NVIDIA 显卡驱动覆盖 Jetson BSP，也不要在未确认方案时刷机或升级系统。

**当前阶段优先完成 FP + YOLO + D435i。** SAM2 为第二条分割路线，待基础链路跑通后再验证板端 Torch / Python 是否满足其要求；无法兼容时明确记录缺口，不能把 YOLO 单路跑通称为双路全部复现。

## 2. 应复现的现有逻辑

```text
D435i 对齐 RGB-D + 彩色内参
         ↓
宿主机 camera_ros_source.py → Unix socket → FP 进程 / 容器
         ↓
SEARCH → YOLO（或 HSV+SAM2）→ Mask 深度修正 → REGISTER
                                                   ↓
                       全图重新分割 ← LOST ← TRACK + 几何检查
                                                   ↓
                              可选动态 ROI / 位姿与耗时输出
```

要求保留：

1. 分割只用于初始化与恢复，正常帧调用 FP `track_one`。不要改成每帧 YOLO + 每帧重新注册。
2. 使用 NVIDIA 原版 FP 算法接口及现有外围逻辑；先不要替换成 Isaac ROS / TensorRT 的另一套位姿实现。
3. 状态为 SEARCH / REGISTER / TRACK / LOST。无效位姿立即拒绝；几何检查默认间隔 2s，连续 3 次实际检查失败触发恢复，持续失败通常约 4～6s；搜索重试间隔默认 0.5s。
4. 深度支持率阈值 0.3、深度残差阈值 0.03m、有效支持像素至少 64，详见 `src_fp/config/kfs_blue.json`。这不是 FP 置信概率，也不能可靠识别跟错相似物体。
5. ROI 来自上一帧有效位姿投影的 Mesh 包围框，默认每边外扩 0.3；源端裁剪 RGB / Depth 并修正 K 主点，位姿仍在原相机坐标系。初始化、失跟或按 R 回全图。先关闭 ROI 跑基线，再开启核对。
6. Mask 深度修正与深度空间滤波是两项功能。`--depth-refinement` 用深度修正 Mask；双边空间滤波 `--depth-spatial-filter` 目前仅在离线入口实现、默认关闭，不补洞也不跨帧。需要实时滤波时另行移植并验证，不能宣称已默认接入。

## 3. 代码入口与安装分工

| 部分 | 对应文件 / 注意事项 |
|---|---|
| FP 核心 | `FoundationPose/`：scorer、refiner、渲染和跟踪 |
| 相机桥接 | `src_fp/camera_ros_source.py`、`live_rgbd_transport.py`、`kfs_roi.py` |
| 实时入口 | `src_fp/run_foundationpose_live.py`，共用 `run_foundationpose_zed.py` 的加载、手画和分批评分辅助 |
| 自动分割 | `src_fp/kfs_segmentation.py`、`kfs_yolo_segmentation.py`；SAM 锁定记录在 `src_fp/sam2-lock.json` |
| 健康检查 / 恢复 | `src_fp/kfs_pose_controller.py`、`src_fp/config/kfs_blue.json` |
| 显示与日志 | `src/common/display.py`、`src_fp/kfs_debug.py`、`src_fp/kfs_runtime_metrics.py` |
| 深度空间滤波 | `src_fp/depth_spatial_filter.py`，当前仅由离线入口调用 |
| 桌面启动器 | `src_fp/scripts/start_kfs_fp.sh`，保留其运行参数语义，但镜像名 / 构建方案必须为 NX 重新适配 |

建议宿主机负责 D435i / ROS，FP 使用隔离环境或匹配 JetPack 的容器，避免相机侧 Python 与 GPU Torch 互相污染。所有扩展在 NX 环境重编译；桌面 `.so`、`mycpp/build`、镜像不能搬过去直接用。

本仓库 `requirements.txt`、cu130 constraints、SAM extra requirements 等记录的是桌面环境，不能整份盲装；尤其检查 NumPy / Python / Torch / pybind11 ABI。迁移时可以调整平台依赖和构建适配，先保留算法参数与数据语义。

FP 依赖需验证实际运算：Torch CUDA tensor、torchvision CUDA NMS（YOLO）、PyTorch3D KNN、NVDiffRast CUDA rasterization、Warp GPU kernel、`mycpp.cluster_poses`，最后加载 scorer / refiner 并做注册与跟踪。不能只看 import 成功。

`FoundationPose/tools/check_cuda130.py` 包含这些 GPU 测试，但写死 Torch 2.9.1 / CUDA 13 断言；可参考并改成匹配板端版本的检查，不能直接把原脚本当作 NX 验收标准。PyTorch3D / NVDiffRast / Warp 如需换兼容版本，应记录理由并做相同运算检查。

## 4. D435i 输入必须核对

- 用 USB 3 连接，先单独验证 SDK 的 Color / Depth；Viewer 与 ROS 驱动不能同时抢相机。
- 桌面端默认单条 `/camera/camera/rgbd`，启用 `enable_color`、`enable_depth`、`enable_rgbd`、`enable_sync`、`align_depth.enable`。桥接使用 Reliable、Keep Last 1；本机曾在 Best Effort 下严重丢帧，但板端实际帧率仍需实测。
- 输入 RGB 为 uint8 RGB，Depth 为与 RGB 对齐的 float32 米制深度，K 属于同一幅彩色图。D435i 原始整数深度按正确比例转米；缩放 RGB、Depth、K 必须同步，不能重复转换单位。
- 桌面记录为 realsense-ros 4.58.4 / SDK 教程选用 2.58.4；板端需按 L4T、SDK 与 ROS 支持情况选兼容组合，不强行锁成桌面版本。优先保留单条 RGBD 及时间戳语义。
- 本仓库已有 `--input-mode separate` 支持独立图像话题近似同步；如板端无法使用单条 RGBD，可作为明确记录的适配路线，但要验证时间差和对齐，不把它当作完全相同的采集实现。
- 先看实际 Ubuntu 再选 ROS；如果是 Ubuntu 20.04，不要照桌面文档添加 jammy 源安装 Humble。先验证 SDK，评估匹配系统的 ROS 或隔离环境；如需新增直接 SDK 的桥接路线，要记录它与原 ROS 路线的区别。
- Jetson 使用定制内核。RSUSB 可作为原型验证方案，不能照搬桌面 DKMS / 内核补丁。[RealSense Jetson 官方说明](https://github.com/realsenseai/librealsense/blob/master/doc/installation_jetson.md)

首次验证可用 RGB / Depth 640×480@30、桥接 `--scale 1`；这是功能基线，不是 NX 上实时帧率承诺。

## 5. 模型与数据交付

完整源码包：`fp_xavier_nx_source.tar.gz`；配套权重 / Mesh 包：`fp_kfs_assets.tar.gz`。两者需从维护者获得并各自解压到同一个**新建项目目录**，另有同名 `.sha256` 校验文件。源码包含当前未提交的文档与脚本改动，不能只凭远端分支已存在就认定包含了这些内容。

所需材料：

- FP refiner：`FoundationPose/weights/2023-10-28-18-33-37/{config.yml,model_best.pth}`。
- FP scorer：`FoundationPose/weights/2024-01-11-20-02-45/{config.yml,model_best.pth}`。
- YOLO 实例分割：`weights/kfs_yolo_seg.pt`，类别 `blue_kfs`；桌面 Ultralytics 版本为 8.3.161，板端安装时约束现有 Torch / torchvision，防止 pip 自动替换。
- Mesh：`kfs_model/BlueTrueKFS13/BlueTrueKFS13.obj` 及 MTL / PNG；单位米，边长 0.35m。实物与模型必须对应；不要自动换成其他尺寸。
- SAM（后续可选）：官方 SAM2.1 Small，锁定提交 / checkpoint SHA256 在 `src_fp/sam2-lock.json`。不要求一开始装好它，也不要求下载 ZED SDK。

这些材料不包含相机实物、官方 demo 全套数据或历史录像；需要离线验证时另外下载官方 demo 或请用户提供同一段 RGB-D。YOLO `.pt` / FP checkpoint 需与板端可用 Torch 实际做加载检查，不能仅凭文件完整推断兼容。

## 6. 分阶段验收与回报

按以下顺序推进，某项有阻塞时继续可独立完成的工作，并报告具体失败依赖和日志：

1. **环境盘点**：给出 Xavier NX 内存、JetPack / L4T、Ubuntu、Python、CUDA、Torch / torchvision 及选择该组合的依据。
2. **GPU 与扩展**：上述实际 GPU 运算通过；本机编译 `mycpp`，记录包版本 / 源码提交与安装日志。
3. **FP 离线**：官方 demo 或同一段已知 RGB-D 完成首次注册和多帧 tracking，保存位姿与叠图。
4. **相机输入**：验证对齐、单位、K、时间戳和实际输入帧率；只保留最新帧，不积压。
5. **实时手画基线**：关闭自动分割 / ROI，确认桥接与 FP 跟踪可用。
6. **YOLO 全链路**：分割 → Mask 深度修正 → FP → 几何检查 / 恢复；先关闭窗口与 ROI 测基线，再分别开启显示与 ROI。核对移出 / 重入、遮挡及 R 重定位。
7. **第二条路线与优化**：再判断 SAM2 是否能兼容、是否需性能优化。首轮尽量沿用注册 5 次 / tracking 2 次 / score batch 32；内存不足先降评分 batch 至 16。其他参数修改需记录并比较效果。

必须保存 `runtime.jsonl`、`events.jsonl`、`loop_timing.csv` 和正常退出汇总；位姿为物体到相机的 4×4 变换，平移单位米，失败帧不编造位姿。区分初次注册耗时、稳态整循环 FPS、输入等待与显示开销。Jetson 内存为共享资源，另用 `tegrastats` 等记录整机内存 / 热状态；不把桌面 GPU 的指标直接当作 NX 的性能预期。

最终请交付：可重复的 NX 安装 / 构建记录、可启动入口、环境锁定文件、实际验证结果、仍未完成的功能。**当前没有 Xavier NX 实测结果，不要把桌面运行记录称为板端验证。**
