# 从零安装原版 FoundationPose（FP）

**先下载 NVIDIA 原版并跑通官方 demo，相机与分割按需添加。** 本文采用 Ubuntu 22.04 / x86_64 / NVIDIA GPU / Docker 路线，更新日期：2026-10-17。

| 你想做什么 | 阅读范围 |
|---|---|
| 只体验原版 FP | 本文；无需相机、ROS、YOLO、SAM 或蓝色箱体材料 |
| 接相机，手动画目标 | 本文 + [相机桥接](FP_CAMERA_BRIDGE.zh-CN.md) |
| 自动得到目标 Mask | 本文 + [可选分割](FP_SEGMENTATION.zh-CN.md)，可以先用图片测试 |
| 安装并运行 FP + YOLO / SAM | [完整安装](FP_KFS_FULL_INSTALL.zh-CN.md) + [当前运行指南](src/README.md) |

以下命令按顺序执行；未特别标注时在宿主机 Ubuntu 终端、普通用户下操作。Windows、WSL、macOS 和 Jetson 不在本文路线范围内。

##

按需继续[相机桥接](FP_CAMERA_BRIDGE.zh-CN.md)、[自动分割](FP_SEGMENTATION.zh-CN.md)或[完整 KFS 实验](FP_KFS_FULL_INSTALL.zh-CN.md)。本文已作静态核对，尚未在另一台干净机器上完整复装。
