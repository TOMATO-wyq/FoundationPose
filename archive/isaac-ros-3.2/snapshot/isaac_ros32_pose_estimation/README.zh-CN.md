# Isaac ROS Pose Estimation（姿态估计）

基于深度学习、由 NVIDIA 加速的三维物体姿态估计。

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-3.2/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_pose_estimation/dope_objects.png/"><img alt="示意图" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-3.2/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_pose_estimation/dope_objects.png/" width="400px"/></a></div>

## 概述

Isaac ROS Pose Estimation 包含三个用于预测物体姿态的 ROS 2 软件包。它们的区别如下：

| 节点 | 无需重新训练即可识别新物体 | 支持 TAO | 速度 | 质量 | 成熟度 |
|---|---|---|---|---|---|
| `isaac_ros_foundationpose` | ✓ | 不适用 | 快 | 最佳 | 较新 |
| `isaac_ros_dope` | × | × | 最快 | 良好 | 久经验证 |
| `isaac_ros_centerpose` | × | ✓ | 较快 | 较好 | 成熟 |

这些软件包利用 GPU 加速深度神经网络（DNN）推理来估计物体姿态。感知功能可以将预测结果与对应的深度信息融合，得到物体的三维姿态和距离，供导航或操作任务使用。

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-3.2/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_pose_estimation_nodegraph.png/"><img alt="姿态估计节点图" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/release-3.2/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_pose_estimation_nodegraph.png/" width="500px"/></a></div>

`isaac_ros_foundationpose` 在节点图中利用三维边界立方体的尺寸估计新物体的姿态。它基于 NVLabs 开发的预训练深度学习模型 [FoundationPose](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/foundationpose)。FoundationPose 无需微调即可对未见过的物体进行姿态估计和跟踪，其准确率优于文档所比较的已有方法。

FoundationPose 包含两个不同的模型：refine 模型和 score 模型。refine 模型接收初始姿态假设并反复优化，再将优化后的结果交给 score 模型，由后者选择并确定最终姿态。refine 模型也可用于跟踪：它根据新输入的图像和上一帧的姿态估计更新当前姿态。跟踪比重新进行姿态估计更高效，在 Jetson Orin 平台上的速度超过 120 FPS。

`isaac_ros_dope` 在节点图中根据三维边界立方体尺寸估计已知物体的姿态。它需要一个预训练的 [DOPE](https://github.com/NVlabs/Deep_Object_Pose)（Deep Object Pose Estimation，深度物体姿态估计）模型。输入图像可能需要裁剪和缩放，以保持长宽比并匹配 DOPE 的输入分辨率。DNN 推理得到估计结果后，DNN 解码器会结合指定的物体类型和模型推理生成的置信图，输出物体姿态。

NVLabs 提供了一个使用 [HOPE](https://github.com/swtyree/hope-dataset) 数据集预训练的 DOPE 模型。HOPE 是 `Household Objects for Pose Estimation`（用于姿态估计的家用物体）的缩写。它是一个面向研究的数据集，使用玩具杂货物品及其带纹理的三维网格，通过合成数据进行训练。如果要用 DOPE 识别应用中的其他物体，需要针对这些物体使用另一数据集训练模型。例如，DOPE 曾被训练用于检测一种手推车，以供移动机器人从车下驶入、抬起并搬运它。训练自己的 DOPE 模型的方法参见[自定义 DOPE 模型训练教程](https://nvidia-isaac-ros.github.io/concepts/pose_estimation/dope/tutorial_custom_model.html)。

`isaac_ros_centerpose` 与 `isaac_ros_dope` 一样可以估计物体姿态，但还提供额外功能。[CenterPose](https://github.com/NVlabs/CenterPose) DNN 可在图像中检测物体、生成其二维关键点、估计具有尺度不确定性的六自由度（6-DoF）姿态，并回归相对的三维边界立方体尺寸。它针对已知物体类别工作，不要求模型见过该类别中的具体实例。例如，CenterPose 模型可以检测一把椅子，即使训练图像中从未出现过这把椅子。

姿态估计需要大量计算，因此通常不会以输入相机的帧率逐帧运行。为了有效利用资源，系统可以先对单帧估计物体姿态并将其用于导航，再以低于普通相机输入帧率的频率计算新的姿态，以便在导航过程中继续修正结果。

此仓库中的软件包通过 [Isaac ROS DNN Inference](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference) 提供的 [Triton](https://github.com/triton-inference-server/server) 或 [TensorRT](https://developer.nvidia.com/tensorrt) 加速 DNN 模型推理。预处理依赖 `Isaac ROS DNN Image Encoder`，它也包含在 [Isaac ROS DNN Inference](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/blob/release-3.2/isaac_ros_dnn_image_encoder) 中。

## 性能

下表保留原 README 中的基准测试数值。`ms @ 30Hz` 为原文给出的延迟指标。

| 示例计算图 | 输入尺寸 | AGX Orin | Orin NX | Orin Nano Super 8GB | 搭载 RTX 4090 的 x86_64 |
|---|---|---|---|---|---|
| [FoundationPose 姿态估计节点](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-3.2/benchmarks/isaac_ros_foundationpose_benchmark/scripts/isaac_ros_foundationpose_node.py) | 720p | [1.54 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-3.2/results/isaac_ros_foundationpose_node-agx_orin.json)；780 ms @ 30Hz | — | — | [9.56 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-3.2/results/isaac_ros_foundationpose_node-x86-4090.json)；110 ms @ 30Hz |
| [DOPE 姿态估计计算图](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-3.2/benchmarks/isaac_ros_dope_benchmark/scripts/isaac_ros_dope_graph.py) | VGA | [27.3 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-3.2/results/isaac_ros_dope_graph-agx_orin.json)；54 ms @ 30Hz | [15.2 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-3.2/results/isaac_ros_dope_graph-orin_nx.json)；73 ms @ 30Hz | — | [186 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-3.2/results/isaac_ros_dope_graph-x86-4090.json)；12 ms @ 30Hz |
| [CenterPose 姿态估计计算图](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-3.2/benchmarks/isaac_ros_centerpose_benchmark/scripts/isaac_ros_centerpose_graph.py) | VGA | [44.8 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-3.2/results/isaac_ros_centerpose_graph-agx_orin.json)；43 ms @ 30Hz | [29.0 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-3.2/results/isaac_ros_centerpose_graph-orin_nx.json)；51 ms @ 30Hz | [29.8 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-3.2/results/isaac_ros_centerpose_graph-orin_nano.json)；50 ms @ 30Hz | [50.2 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/release-3.2/results/isaac_ros_centerpose_graph-x86-4090.json)；14 ms @ 30Hz |

---

## 文档

有关此仓库的使用方法，请访问 [Isaac ROS 文档](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/index.html)。

---

## 软件包

- [`isaac_ros_centerpose`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_centerpose/index.html)
  - [快速入门](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_centerpose/index.html#quickstart)
  - [故障排查](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_centerpose/index.html#troubleshooting)
  - [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_centerpose/index.html#api)
- [`isaac_ros_dope`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_dope/index.html)
  - [快速入门](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_dope/index.html#quickstart)
  - [更多示例](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_dope/index.html#try-more-examples)
  - [使用其他模型](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_dope/index.html#use-different-models)
  - [故障排查](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_dope/index.html#troubleshooting)
  - [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_dope/index.html#api)
- [`isaac_ros_foundationpose`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/index.html)
  - [快速入门](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/index.html#quickstart)
  - [更多示例](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/index.html#try-more-examples)
  - [故障排查](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/index.html#troubleshooting)
  - [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/index.html#api)

## 最新更新

2024-12-10：添加了姿态估计结果的后处理工具。
