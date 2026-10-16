# Isaac ROS Common

用于 Isaac ROS 软件套件开发的 Dockerfile 和脚本。

## 概述

Isaac ROS Common 仓库提供多种脚本和 Dockerfile，帮助简化 Isaac ROS 软件套件的开发与测试。

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-3.2/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_common/isaac_ros_common_tools.png/"><img alt="Isaac ROS 开发运维工具" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-3.2/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_common/isaac_ros_common_tools.png/" width="auto"/></a></div>

Docker 容器可帮助你快速配置一组对版本和兼容性要求较高的框架及依赖，使 Isaac ROS 软件包更容易正常运行。x86_64 平台的 Dockerfile 基于 [Deep Learning Frameworks Containers](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) 的 23.10 版镜像。在 Jetson 平台上，这些依赖由 JetPack 管理。

使用 Docker 镜像还可以让 CI/CD 系统在 Kubernetes 等云原生平台上扩展开发运维工作并运行自动化测试。

已知问题的解决方法参见[故障排查](https://nvidia-isaac-ros.github.io/troubleshooting/index.html)。

---

## 文档

有关此仓库的使用方法，请访问 [Isaac ROS 文档](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_common/index.html)。

---

## 最新更新

2024-12-10：重构了 Dockerfile。
