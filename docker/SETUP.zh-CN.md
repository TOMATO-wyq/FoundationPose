# FoundationPose CUDA 13.0 环境

当前机器：Ubuntu 22.04、RTX 5060 Laptop GPU、NVIDIA 驱动 595.91.07。主机 ZED SDK 5.5 使用 CUDA 13。`nvidia-smi` 显示的 CUDA 13.2 是驱动支持上限，容器使用自己的 CUDA Toolkit。

用户选择统一到 CUDA 13.0。本镜像配置为 NVIDIA CUDA 13.0.2、PyTorch 2.9.1+cu130、torchvision 0.24.1+cu130。PyTorch 官方提供这组安装包；PyTorch3D 与 NVDiffRast 固定到记录的 Git 提交，在镜像内用 CUDA 13.0 编译。FoundationPose 没有要求必须使用 CUDA 12.8。

**状态：CUDA 13.0 环境及官方芥末瓶 demo 推理已验证。** 2026-09-22 的检查确认 PyTorch 2.9.1+cu130、PyTorch3D KNN、NVDiffRast 光栅化、Warp GPU kernel、mycpp 位姿聚类和 estimater 导入均正常。pybind11 为 2.13.6。官方预训练权重和 `demo_data/mustard0` 已放入项目目录；用户运行 demo 后生成了 737 帧位姿文件。当前 ROS 2 与 ZED 的容器整合暂停。

## 迁移并验证

此步骤已由用户在宿主机完成；以下命令仅作为重新配置时的记录，不需要再次运行：

```bash
bash ~/6Dpose/FoundationPose/docker/migrate_cuda130.sh
```

脚本按顺序移除这项工作创建的指定 CUDA 12.8 镜像标签、构建新的 `foundationpose:rtx50-cu130`、运行环境检查。它不会运行全局 `docker system prune`，不会删除主机代码、数据、权重或 Docker 命名卷。旧镜像的构建缓存和匿名层可能仍占磁盘，待新环境验证后才能有选择地清理。若旧镜像正被容器使用，删除可能失败；请先退出旧容器再运行。

镜像构建时会下载 CUDA、PyTorch 和扩展源码，可能耗时较长。命令输出保存到 `~/6Dpose/FoundationPose/logs/migrate-cu130-日期时间.log`。如官方 PyPI 连续超时，可在确认镜像源可信后设置 `PIP_INDEX_URL` 再重新运行构建。

检查会确认 CUDA Toolkit 与 PyTorch 均为 13.0，在 RTX 5060 上实际执行 PyTorch 矩阵运算、PyTorch3D KNN、NVDiffRast 光栅化和 Warp kernel，并重新编译 `mycpp`，最后执行项目自带的 `check_env.py`。模型权重未下载会出现警告；完整位姿推理还需权重和示例数据。

## 2026-09-22 mycpp 修复结果

最初的 CUDA 13.0 检查在调用 `mycpp.cluster_poses` 时出现段错误。镜像更新至 pybind11 2.13.6，并将测试输入改为与实际调用一致的 NumPy 矩阵数组后，重新编译和运行通过。C++ 输出格式警告也已修复。检查输出和最终 Python 包版本分别保存在用户终端记录及 `logs/packages-cu130.txt`。目前仅验证无权重的基础计算链路。Warp 1.17.0 自身报告“CUDA Toolkit 12.9”属于其编译工具链信息；本容器 `nvcc` 和 PyTorch CUDA 均为 13.0，Warp 的 GPU kernel 已在本机运行成功。

## 日常使用

迁移通过后：

```bash
source ~/6Dpose/activate
```

输入 `exit` 离开容器。启动脚本将项目目录映射到容器 `/workspace/FoundationPose`；项目内的修改会保存在主机。镜像构建配置是 [Dockerfile.cu130](Dockerfile.cu130)，运行配置是 [cuda130.sh](cuda130.sh)。

在有 X11 桌面的主机终端中重新进入容器，启动脚本会将桌面显示和认证文件映射给容器，无需重新构建镜像。执行 `python run_demo.py --debug 1` 可显示实时画面；`--debug 2` 还会将每帧画面保存到 `debug/track_vis/`。`--debug 0` 不显示窗口，仅写入 `debug/ob_in_cam/` 的位姿结果。官方窗口在运行结束后会自动关闭。

## 参考

- [PyTorch 官方版本安装命令](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA CUDA 驱动兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/why-cuda-compatibility.html)
- [FoundationPose 原文安装说明](../readme.md)
