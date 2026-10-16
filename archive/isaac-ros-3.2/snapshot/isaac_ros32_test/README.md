# Isaac ROS 3.2 / RTX 5060 compatibility probe

This is an isolated model-inference probe, not a complete Isaac ROS installation.
It does not alter the existing `foundationpose:rtx50-cu130` image or the host CUDA
and ZED SDK installations.

Confirmed by inspecting NVIDIA's release-3 package index and FoundationPose .deb:

- `ros-humble-gxf-isaac-tensor-rt` depends on TensorRT major 10.
- `libgxf_isaac_foundationpose.so` depends on `libcudart.so.12` and CV-CUDA.
- The FoundationPose library contains PTX, but this alone does not prove that
  every GXF dependency or kernel works on RTX 5060.
- The existing original FoundationPose container contains TensorRT 11 C++
  libraries. These cannot substitute for the required major-10 libraries.

The probe uses Ubuntu 22.04, CUDA 12.8.1, and TensorRT 10.9.0.34. This TensorRT
version is present in NVIDIA's Ubuntu 22.04 package index. The model profiles
match the Isaac ROS 3.2 quickstart (refine max batch 42, score max batch 252).
Model conversion can require at least 7.5 GB of free GPU memory per the
official quickstart. No artificial TensorRT workspace cap is applied.

Build from the host:

```bash
cd ~/code/isaac_ros32_test
sudo docker build --progress=plain -t foundationpose:isaac32-trt109-probe .
```

Run with the ZED node and other GPU workloads stopped to leave memory for engine
conversion. Results are written under this directory as the calling user's UID.

```bash
sudo docker run --rm --gpus all --network host \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/work" foundationpose:isaac32-trt109-probe
```

After this succeeds, the next validation is the complete Isaac ROS 3.2 GXF
graph. This probe does not benchmark end-to-end tracking: `trtexec` timings use
synthetic inputs and exclude rendering, preprocessing, ROS transport and ZED.

## Isaac ROS package image

The model smoke test has passed on RTX 5060 with TensorRT 10.9. The next
isolated layer installs ROS 2 Humble and the official Isaac ROS 3.2 FoundationPose
Debian package. It leaves the CUDA 13 original image and host unchanged:

```bash
sudo docker build --progress=plain -f Dockerfile.isaac-ros \
  -t foundationpose:isaac32-ros .
```

After the image is built, enter it with `source ~/code/isaac_ros32_test/activate`.
The script shares `/work` for engines and logs and mounts `/workspace/6Dpose`
read-only for the existing object model. The ZED ROS node stays on the host.

The ROS image still needs a GXF load test and camera/mask remapping before it
can be called a working ZED pipeline.

The image includes CV-CUDA 0.5 from the NVIDIA package used in the release-3.2
`Dockerfile.base`; the GXF FoundationPose extension needs `libcvcuda.so.0`
and `libnvcv_types.so.0` at runtime.

## 完整 Isaac pose + tracking 试运行

完整 launch 在同一组件容器中同时运行 Selector、Isaac FoundationPose 和
Isaac FoundationPoseTrackingNode。pose 的 `refine_iterations` 设为 5；
这表示一次首帧估计内部迭代 5 轮。运行期间不卸载 pose 组件。

先结束旧的 Isaac launch、单独 tracker 桥接和录像回放。容器终端运行：

```bash
source ~/code/isaac_ros32_test/activate
ros2 launch /work/zed_foundationpose.launch.py
```

宿主机另一个终端运行录像（先用少量帧检查）：

```bash
source /opt/ros/humble/setup.bash
python3 ~/code/isaac_ros32_test/play_recording.py --max-frames 30 --fps 2
```

官方 Selector 在首帧 pose 位姿矩阵返回后才把后续帧交给 tracking；
它不会根据跟踪结果缺失自动重新 pose，只按 `reset_period` 定时重置。
播放器日志中的 `preview_frame_cache=miss` 表示没有缓存对应的预览图像，
不代表跟踪失败。之前该完整图在 8 GB 显卡上出现过 TensorRT OOM；
若再次出现，应先看容器日志里的显存分配错误，不能靠提前结束 pose
来称作“完整同时运行”。当前执行环境无 Docker 权限，以上完整链路尚未实测。

## 用原版首帧位姿测试 Isaac tracker

这是隔离测试：只启动 Isaac ROS tracker，固定读取原版保存在
`/workspace/6Dpose/outputs/zed_foundationpose/ob_in_cam/000000.txt` 的
首帧位姿，不启动 Isaac 的首帧定位、Selector 或 score 引擎，也不热卸载组件。
使用同一份 `zed_sequence` 录像，并从第二帧开始。先停止旧的 Isaac launch
和旧的录像回放进程。

容器终端 1：

```bash
source ~/code/isaac_ros32_test/activate
ros2 launch /work/zed_foundationpose_tracker_only.launch.py
```

先确认终端 1 出现 `foundationpose_tracker ... Node was started`，再启动后两步。
tracker 的 launch 会自动把 GXF message compositor 所在目录加入动态库路径；
如果提示该库在 `/opt/ros/humble` 中不存在，才需要按更新后的
`Dockerfile.isaac-ros` 重建镜像。

容器终端 2：

```bash
source ~/code/isaac_ros32_test/activate
python3 /work/seed_tracking_bridge.py
```

宿主机终端 3，先只验证录像第二帧能否产生一次 tracking 位姿：

```bash
source /opt/ros/humble/setup.bash
python3 ~/code/isaac_ros32_test/play_recording.py \
  --tracker-test --tracker-probe --max-frames 1 \
  --result-timeout 30 --no-preview
```

应先看到桥打印 `Loaded original first-frame pose`、四路 tracker 订阅数，
然后看到 `TRACKING PASS`。探针每 2 秒重发同一录像帧，避免启动时单次消息丢失。
通过后再运行 `--tracker-test --max-frames 10` 检查连续帧和绿色 3D 框。
桥会把上一帧的 tracker 位姿传给下一帧。此流程只做离线录像验证；
框的位置和稳定性须实际观察，当前环境没有 Docker 权限，尚未做 GPU 实测。

完整录像有 1005 帧。重新回放前先重启桥接脚本，让它重新读取原版首帧位姿；
tracker 容器可保持运行。然后在宿主机运行：

```bash
python3 ~/code/isaac_ros32_test/play_recording.py \
  --tracker-test --tracker-full --max-frames 0 \
  --result-timeout 0
```

此模式处理录像第 2 到第 1005 帧；每张录像帧只发送一次，必须等到该帧的
位姿矩阵反馈和非空检测都返回才显示下一帧。`--result-timeout 0` 表示一直等；
若需限定等待时间，可改成秒数，届时停止并报告缺少哪一种结果，不跳帧。
完成时打印已完成的录像帧数和匹配检测数。
调试窗口会在同一张录像图像上画绿色 3D 框和物体坐标轴（X 红、Y 绿、Z 蓝）。
桥接程序在每一帧都会把上帧位姿数值配上当前 RGB-D 时间戳；
Isaac tracking 输出沿用这个位姿输入的时间戳，否则结果无法匹配录像帧和画框。
桥接端一次只处理一张不同的录像帧；等待结果时最多补送两次同帧副本，
推动 GXF 刷新输出。桥会将补送副本的检测时间戳映射回原录像帧，画框与图像保持对应。
如果同帧非空 `Detection3D` 已到但 `pose_matrix_output` 没到，桥会用该检测框中心
还原这个中心化箱体的 4×4 位姿，作为下一帧的反馈，并标记 `detection_fallback`。
超过 10 秒无位姿反馈时只打印警告，不再放开下一张录像帧。
当前官方 tracking GXF 图使用默认的 1 次 refine；原版 run_demo.py 默认是 2 次。
官方 TrackingNode 未暴露 refine 次数的 ROS 参数，现阶段的精度比较须考虑此差异。

若希望录像按 20 FPS 连续播放并在 tracker 忙时直接跳过积压帧，重启桥接脚本后运行：

```bash
python3 ~/code/isaac_ros32_test/play_recording.py \
  --tracker-test --tracker-stream --max-frames 0 --fps 20
```

流式模式只供吞吐量观察；桥接端仍等前一帧位姿，长时间无反馈时可能停在该帧。
实际位姿帧率可能低于 20 FPS，桥接程序会统计丢弃的积压帧及反馈延迟。

临时观察 tracking 效果时，可改用逐帧等待并隔帧处理，约 502 帧：

```bash
python3 ~/code/isaac_ros32_test/play_recording.py \
  --tracker-test --tracker-full --frame-stride 2 --max-frames 0 \
  --frame-retries 0 --result-timeout 0
```

此模式用原版第 1 帧位姿作种子，只送录像第 2、4、6… 帧；
等待桥接端确认收到每帧的位姿矩阵和检测并画框后再继续。
桥接端会补送最多两次同一图像，以推动 GXF 流水线刷新，不按固定 FPS 强制推进。
不会因时间超过 10 秒跳过录像帧；如果某帧始终没有输出，按 Ctrl+C 停止并查看桥接与 tracker 日志。
隔帧选项仅供本次观察，
之后恢复逐帧处理时删掉 `--frame-stride 2` 即可。

### 离线失跟恢复验证（使用原版逐帧位姿）

重新启动桥接脚本，确认它加载最新代码；tracker-only 容器可以保持运行。
宿主机运行：

```bash
python3 ~/code/isaac_ros32_test/play_recording.py \
  --tracker-test --tracker-full \
  --frame-stride 2 --max-frames 0 --result-timeout 0
```

此模式专用于这段**紫色箱体**录像。每帧独立提取紫色区域，检查它有多少落在
tracker 预测的 3D 框投影内。默认覆盖率低于 0.65 连续 3 个处理帧时，
向桥接端请求下一录像帧对应的原版 `ob_in_cam/*.txt` 位姿；桥接端完成播种后
才让回放继续。触发和确认分别打印 `DRIFT_TRIGGER`、`ORIGINAL_POSE_RESEED`。
恢复后等待 10 个处理帧才允许再次触发，以免短时遮挡造成反复重置。
这些文件是原版 FP 的预先算好的位姿，**并不是 Isaac 重新运行 pose**；
本实验只验证漂移检测和 tracker 重新播种。若紫色区域提取失败，
该帧不作漂移决定；若位姿反馈或检测始终不返回，严格回放会停在当前帧。


Sources:

- https://nvidia-isaac-ros.github.io/v/release-3.2/repositories_and_packages/isaac_ros_pose_estimation/isaac_ros_foundationpose/index.html
- https://isaac.download.nvidia.com/isaac-ros/release-3/dists/jammy/release-3.0/binary-amd64/Packages
- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/Packages.gz
- https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.8.0.html
