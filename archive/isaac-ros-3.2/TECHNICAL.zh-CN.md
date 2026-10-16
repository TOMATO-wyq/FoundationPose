# Isaac ROS 3.2 / FoundationPose 技术归档

日期：2026-10-16。本文依据本机源码、Dockerfile、launch、日志摘要与 `NEXT_STEPS.zh-CN.md` 编写。历史运行结果属于已有记录，本次只做归档及静态检查，没有新增 GPU/相机验证。

## 1. 目标、范围及现状

实验目标是在 RTX 5060 / 8 GB 上评估 Isaac ROS 3.2 FoundationPose 与原版 FoundationPose 的兼容性、跟踪效果和性能。结论是：模型推理与部分 tracker 流程有历史通过记录，但不能证明完整定位→跟踪可稳定运行，也没有证明 Isaac 整体速度优于原版。

本次保存三个实际目录，而非只保存可追踪 Git 文件：

| 原目录 | 角色 | 保存方式 |
| --- | --- | --- |
| `/home/tomato/code/isaac_ros32_test` | 自定义集成、模型、引擎、诊断日志 | 完整工作文件快照 |
| `/home/tomato/code/isaac_ros32_common` | 上游 common 源码研究副本 | 工作文件 + 原始 `.git` 压缩包 |
| `/home/tomato/code/isaac_ros32_pose_estimation` | 上游节点/GXF/launch 研究副本 | 工作文件 + 原始 `.git` 压缩包 |

自定义目录没有独立 Git 仓库；其父目录会落到空的 `/home/tomato/.git`。因此不在 home 仓库执行批量 add。归档目标由实际个人项目 remote 确定为 `TOMATO-wyq/FoundationPose`，从主分支 `78df58aa1973c7a5c6859c10db0591d25535add0` 建专用分支。

## 2. 环境隔离和镜像

探针镜像由 `nvidia/cuda:12.8.1-devel-ubuntu22.04` 构建，安装 TensorRT 10.9 系列库与 `trtexec`。ROS 镜像继承该探针，再安装 Humble、Isaac FoundationPose、message compositor、CV-CUDA 与 VPI。

| 项目 | 本地配置 |
| --- | --- |
| TensorRT | `10.9.0.34-1+cuda12.8` |
| CV-CUDA | `nvcv-lib-0.5.0_beta-cuda12-x86_64-linux.deb` |
| VPI | NVIDIA r36.4 jammy 源的 `libnvvpi3` |
| 通信 | `rmw_fastrtps_cpp`，host 网络，`ROS_DOMAIN_ID` 默认 0 |
| 数据挂载 | 实验目录→`/work`；`6Dpose`→`/workspace/6Dpose`，后者只读 |
| 库路径 | compositor 的实际目录注入容器 `LD_LIBRARY_PATH`；CV-CUDA/VPI 通过容器 ldconfig 配置 |

归档时 Docker 清单显示两个实验镜像：

- `foundationpose:isaac32-trt109-probe`：`sha256:5f1465eb9b8c8bec3fba13aab6459d7987a637bec082968fa50057f353b3779b`，逻辑大小约 22.49 GB。
- `foundationpose:isaac32-ros`：`sha256:66a0d1c10f3054b503e040e084f5261368b520a4bd426974325a076ae324747f`，逻辑大小约 25.31 GB。

两者共享 15 层，不能把两者大小相加作为可释放空间。全机还存在约 82.4 GB 构建缓存，涉及不同项目，不能整体归为 Isaac。镜像删除后的物理释放量必须实际测量，缓存可能继续持有图层。

`foundationpose:rtx50-cu130`、`foundationpose:kfs-sam2`、`foundationpose:kfs-yolo` 均保留。归档不含 Docker 镜像二进制；Dockerfile 中的在线系统包并非全部锁定版本，未来重建结果可能不同。

## 3. 数据流

完整路径：相机桥/录像播放器 → `/isaacfp/*` → Selector → 首帧 pose → tracker → Detection3D / 位姿矩阵 → 显示。

隔离跟踪路径：录像播放器 → 种子/反馈桥 → tracker → 桥接确认当前位姿并转发检测 → 播放器推进下一帧。首帧种子来自原版 FoundationPose 保存的历史 4×4 位姿，不运行 Isaac 首帧定位或 score 引擎。

| 接口 | 消息与约束 |
| --- | --- |
| `/isaacfp/rgb` | `sensor_msgs/Image`，桥接输出 RGB |
| `/isaacfp/depth` | 深度以米表示，与彩色对齐 |
| `/isaacfp/camera_info` | 与缩放后图像一致的内参 |
| `/isaacfp/mask` | 初始目标区域；tracker 本身不使用 mask |
| `/tracking/image`、`depth_image`、`camera_info`、`pose_input` | 同一处理帧时间戳；默认输入 QoS 深度 10 |
| `/tracking/pose_matrix_output` | `TensorList` 位姿反馈 |
| `/tracking/output` | `Detection3DArray`，官方输出按 sensor-data QoS 订阅 |
| `/isaacfp/tracking_debug_output` | 桥接确认当前帧后可靠发布的检测回执 |
| `/isaacfp/tracking_pose_ready` | 保留给旧探针的 Header ACK |

ZED 输入 BGRA8/32FC1，D435 输入支持不同彩色编码和 16UC1 毫米深度。D435 必须使用对齐到彩色的深度；默认三个 topic 位于 `/camera/camera/`。

## 4. 位姿、模型与同步契约

原版首帧路径为 `/workspace/6Dpose/outputs/zed_foundationpose/ob_in_cam/000000.txt`。加载器要求有限值 4×4、齐次底行为 `[0,0,0,1]`，并检查旋转矩阵正交性。变换方向是物体坐标系→相机光学坐标系，平移单位米。

位姿张量名称 `poses`，rank=3，dims=`[1,4,4]`，data_type=9，strides=`[64,16,4]`，小端 float32，列优先存储，以适配 GXF/Eigen 内存顺序。

当前对象是 `BlueTrueKFS13` 箱体，网格原点位于物体中心。因此可在某些情况下用检测框中心恢复姿态。更换网格后必须处理包围盒中心到模型原点的偏移；不能普遍假设二者相同。

反馈桥一次只允许一个不同录像帧在途；上帧姿态值配合当前帧时间戳输入 tracker。新版本在 RGB、depth、camera_info 同 stamp 到齐后立即转发，定时器只保留启动发现/等待告警职责。完成当前位姿更新后再发匹配检测，避免播放器分别等待两路反馈产生额外停顿。

Selector 与 seed bridge 不能同时驱动 `pose_input`。播放器保留混跑检查、同帧匹配、严格等待以及流式吞吐观察模式，流式模式会丢弃积压帧，不宜用于逐帧准确性结论。

## 5. 引擎与节点配置

ONNX 来源标识为 NVIDIA NGC FoundationPose `1.0.0_onnx`。refine 和 score 的输入均为两个 `Nx160x160x6` 张量，min/opt batch=1，max batch 分别为 42 与 252。保存了实际 ONNX 和 TensorRT plan，但 plan 不保证跨硬件/版本可移植。

完整 launch 常驻 Selector、pose、tracker，pose `refine_iterations=5`。tracker-only 只加载 refine，不加载 score。历史记录注明官方 tracker 图为一次 refine，与原版默认两次不同，性能/精度比较必须控制此变量。

tracker-only 当前默认 `component_container` 单线程 executor、四路 QoS depth=10、诊断关闭、ROS INFO、GXF 日志级别 2。可切回多线程容器作对照。

`ISAACFP_TRACKING_GRAPH_OVERLAY=1` 才会挂载容量 4 的实验 YAML。NITROS 启动时会按 ROS QoS 参数重设入口容量，因此只改 YAML 不能证明有效扩大实际输入容量。

## 6. 历史证据与未解决问题

| 证据 | 可以支持的结论 | 不能支持的结论 |
| --- | --- | --- |
| 模型冒烟测试通过记录 | TRT 10.9 下模型转换/合成输入推理曾成功 | 相机端到端可用或更快 |
| `tracking_depth10_no_kicks_full_summary.json` | 502 帧匹配完成、0 离线重播种、0 等待警告；延迟中位数 60.88 ms、P95 1.876 s、最大 3.014 s | 当前最终代码已回归通过；精度有真值保证 |
| `tracking_no_preview_full_summary.json` 与 NEXT_STEPS | 另一次 303/502 后停住，第 608 帧缺检测；去掉 GUI 后仍会卡 | GUI 是唯一原因；关闭窗口导致退化的因果结论 |
| 慢帧时间线 | 有样本主要时间花在四路输入到齐前，RGB 到齐后约 12 ms 出结果 | 已证明具体 DDS/GXF 根因 |
| 约 2.7 GB TensorRT 上下文分配失败记录 | 完整图存在显存压力 | 所有超时或 exit -11 都属于 OOM |

最新 2026-10-02 修改（到齐立即转发、可靠调试输出、统一检测回执）只做过语法检查。历史完整通过记录早于这些改动，不能混为一谈。

补送副本默认 0，离线恢复默认关闭。`--auto-recover-original` 只用预先计算的原版逐帧位姿给 Isaac 重播种，颜色覆盖率检测仅适用于该紫色箱体，不是 Isaac 自主重新定位。

## 7. 恢复后验证计划

1. 校验 MANIFEST；恢复代码，核对新录像、模型和首帧姿态属于同一场景。
2. 在隔离容器中检查依赖/库加载，确认 `pose_input` 只有一个发布者。
3. 先单帧探针，再 10/25 帧，再完整序列；记录匹配数、重播种数、等待、峰值显存和端到端耗时。
4. 按同一 stamp 记录发送、桥接接收/转发、四路 ingress、矩阵/检测输出与播放器确认，定位长尾等待。
5. 每次只改一个配置，再比较 executor、诊断、同容器通信、队列深度。
6. 稳定后才测试完整 pose→tracking，最后与原版在相同帧选择/refine 次数下比较。

## 8. 备份与清理验收

源码、模型、引擎和日志均纳入 manifest；上游 `.git` 另压缩保存，包含工作副本异常状态的复原依据。推送后必须从远端独立获取归档提交，并从 Git blob 解码或 checkout 后逐项校验 SHA-256、文件清单及符号链接。清理前再次比较原目录，任何新增或变更都中止清理。

原版 `6Dpose`、共享 ROS2/ZED/RealSense/CUDA 环境、模型/权重、KFS 镜像保留。旧原始录像及导出 RGB-D 帧不在此归档；只有用户明确同意后才永久删除。删除旧数据后，历史性能记录和报告仍可阅读，但不能直接重跑旧录像实验。

Docker 镜像删除须检查精确镜像 ID、标签、无引用容器，并确认保留镜像还存在。不使用全局 `system prune` 或宽泛缓存清理。实际清理记录和释放空间以本机交付的清理报告为准。
