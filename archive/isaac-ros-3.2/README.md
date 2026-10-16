# Isaac ROS 3.2 实验归档

归档日期：2026-10-16。目的：保存 Isaac ROS 版本的实际源码和实验材料，释放本机磁盘；原版 FoundationPose 继续保留。

这是冻结的实验快照。历史上 tracker-only 曾完成隔帧录像，但随后仍复现数秒停顿，完整 pose + tracking 的显存和稳定性问题没有解决。归档时不重新启动 GPU 或相机测试。

## 内容与入口

| 路径 | 内容 |
| --- | --- |
| `snapshot/isaac_ros32_test/` | 自定义 Dockerfile、相机桥、录像播放器、tracking 反馈桥、launch、实验图、原始说明、模型、引擎和日志 |
| `snapshot/isaac_ros32_common/` | 本机 Isaac ROS common 工作目录快照 |
| `snapshot/isaac_ros32_pose_estimation/` | 本机 Isaac ROS pose estimation 工作目录快照，包括 GXF/C++ 与 FoundationPose launch |
| `provenance/` | 两个上游副本的 Git 元数据压缩包、原始 refs/HEAD/status/remote，以及从 HEAD 取出的许可证与说明 |
| `MANIFEST.json` | 原目录每个文件的 SHA-256、字节数、权限，符号链接和目录清单；包含原 `.git` 内容 |
| `TECHNICAL.zh-CN.md` | 架构、数据契约、已知问题、验证边界、清理范围 |
| `restore.py` | 校验快照并恢复原始三个目录，不覆盖已有目录 |

上游副本的 `.git/index` 原本就不存在；不能根据 `git status` 的批量删除提示丢弃本地文件。本归档保存磁盘实际内容，另存原始 Git 元数据，不把副本当作干净的上游 checkout。

许可证保留原文件；缺失或特殊的许可证另存于 `provenance/`。第三方材料沿用原有许可，本目录不改变其许可条件。

## 获取与完整性验证

本归档位于仓库 `TOMATO-wyq/FoundationPose` 的专用分支 `codex/archive-isaac-ros-3.2-20261016`，路径 `archive/isaac-ros-3.2`。主分支和现有工作目录不受影响。

```bash
git clone --single-branch --depth 1 \
  --branch codex/archive-isaac-ros-3.2-20261016 \
  https://github.com/TOMATO-wyq/FoundationPose.git isaac-archive
cd isaac-archive/archive/isaac-ros-3.2
python3 restore.py --verify-only
```

ONNX、TensorRT plan、CV-CUDA 安装包和日志均作为实际 Git 文件归档，不依赖 Git LFS 或本地软链接。单文件均小于 100 MiB。下载本分支会额外占用约 350 MB 工作区，另需 Git 对象空间。

原上游 `.gitattributes` 以 `.gitattributes.snapshot` 原样保存，避免其 LFS 规则影响此归档；MANIFEST 记录对应关系，恢复时还原原文件名。

## 恢复目录

默认只验证，不写入原工作目录。恢复须显式指定目标父目录：

```bash
python3 restore.py --destination /home/tomato/code
```

目标 `isaac_ros32_test`、`isaac_ros32_common`、`isaac_ros32_pose_estimation` 必须全部不存在。脚本校验所有快照文件和 Git 元数据，再恢复原路径与权限。若只想查看源码，无需恢复。

归档的两个上游 Git HEAD：

- common：`fcf4d9e17f8f0a7f47f1d22d6a18421ce3768c01`
- pose estimation：`9caca619bcc9d637b3107e17c1a77132c9d7863b`

恢复原 `.git` 后仍保留原先缺失索引的状态。若以后要正常开发，请先复制恢复目录，再决定如何重建索引；不要把批量删除状态直接提交到上游。

## 历史运行环境

以下版本来自本地 Dockerfile 和实验记录，不是对当前在线软件源的可用性保证：Ubuntu 22.04、ROS 2 Humble、Isaac ROS 3.2、CUDA 12.8.1、TensorRT `10.9.0.34-1+cuda12.8`、CV-CUDA 0.5 beta、VPI 3，RTX 5060 / 8 GB。

保留的原版环境是 `foundationpose:rtx50-cu130`，其 CUDA/TensorRT 与本实验隔离。不要把 Isaac 的库路径写到主机全局环境中。

## 重建实验镜像

先恢复三个目录，检查宿主机已有 NVIDIA 驱动、Docker GPU runtime、ROS2 及相机驱动。恢复环境不会安装或卸载这些共享依赖。

```bash
cd /home/tomato/code/isaac_ros32_test
sudo docker build --progress=plain \
  -t foundationpose:isaac32-trt109-probe .
sudo docker run --rm --gpus all --network host \
  --user "$(id -u):$(id -g)" -v "$PWD:/work" \
  foundationpose:isaac32-trt109-probe
sudo docker build --progress=plain -f Dockerfile.isaac-ros \
  -t foundationpose:isaac32-ros .
```

第一层运行 `smoke-test.sh`。现有模型/引擎非空时会复用；换 GPU、TensorRT 或 CUDA 后应移走旧 plan，再生成新引擎。脚本的模型 URL 是 NVIDIA NGC FoundationPose `1.0.0_onnx`；保存的 ONNX 可以避免重新下载。

第二层使用 `vendor/nvcv-lib-0.5.0_beta-cuda12-x86_64-linux.deb`，并通过在线软件源安装 Isaac ROS 和 VPI。部分系统包未固定完整版本；即使源码和模型逐字节恢复，也不能保证未来重建出与原镜像完全相同的环境。此次归档没有保存 Docker 镜像二进制。

## Tracker-only 历史流程

该流程依赖保留在 `6Dpose` 中的物体模型，以及历史录像导出帧和原版首帧位姿。若这些旧数据已按用户要求删除，需要重新录制、导出并生成匹配的首帧位姿，修改脚本路径后再运行；不能直接拿旧位姿用于新场景。

容器终端一：

```bash
source /home/tomato/code/isaac_ros32_test/activate
ros2 launch /work/zed_foundationpose_tracker_only.launch.py
```

容器终端二：

```bash
source /home/tomato/code/isaac_ros32_test/activate
python3 /work/seed_tracking_bridge.py --kick-copies 0
```

宿主机终端三（仅历史数据存在且匹配时）：

```bash
source /opt/ros/humble/setup.bash
python3 /home/tomato/code/isaac_ros32_test/play_recording.py \
  --tracker-test --tracker-full --frame-stride 2 \
  --max-frames 10 --result-timeout 120
```

先检查短段；确认同帧匹配后再把 `--max-frames` 改为 `0`。默认关闭诊断、补送副本和离线恢复。`--result-timeout 0` 会无限等待，只适合主动观察。

不要同时运行完整 Selector 和种子桥：它们都会向 `/tracking/pose_input` 发布。不要在 tracker 工作时热卸载 pose 组件来规避显存不足。

## 相机和完整流水线

- `zed_bridge.py`：ZED 的 BGRA8 彩色、32FC1 米制深度；发布 `/isaacfp/{rgb,depth,camera_info,mask}`。
- `d435_bridge.py`：适配 RGB/BGR/RGBA/BGRA，16UC1 默认毫米转米，32FC1 按米。必须使用对齐到彩色的深度。
- 两个桥一次运行一个。手绘掩码只提供初始目标区域，不是逐帧分割。
- `zed_foundationpose.launch.py`：Selector + pose + tracking；pose refine 5 次；8 GB GPU 上曾 OOM，不能当作已验证可用的实时链路。
- `seed_tracking_bridge.py`：固定历史录像首帧种子与后续反馈；不是通用实时相机初始化程序。

`activate` 将实验目录挂到 `/work`，把 `/home/tomato/6Dpose` 只读挂到 `/workspace/6Dpose`，使用 host 网络和 IPC。当前 launch 的模型路径为 `kfs_model/BlueTrueKFS13`，必须匹配新录制的对象。

## 排查顺序与说明优先级

优先阅读本 README、技术文档和快照 `NEXT_STEPS.zh-CN.md` 顶部。快照旧 README 包含历史命令和已过时的“默认恢复/补送”说明；以代码实际默认值为准。

历史 tracker 曾完成 502/502 隔帧输入，但关闭预览的另一次运行在 303/502 后停止。2026-10-02 最后修改了回调转发和统一完成回执，未复测。不能把前次通过记录归为当前源码已通过。

重新实验时依次检查：发布者是否混跑、数据时间戳/单位/对齐、四路输入到齐时间、矩阵与检测反馈、GPU 内存。先保存日志，再修改单一因素作对照。具体历史证据见技术文档。

## 磁盘清理边界

原 Isaac 专用目录只有在远端独立下载校验全部通过后才能清理。旧相机原始录像和 RGB-D 导出数据属于另一项经用户明确确认的清理，不作为本源码备份的一部分。

ROS2、ZED、RealSense、CUDA、原版 FoundationPose、KFS 镜像、模型、权重和评估报告保留。Docker 仅考虑已核对 ID 的两个 `isaac32` 镜像，构建缓存含共享内容，未识别来源的缓存不清理。
