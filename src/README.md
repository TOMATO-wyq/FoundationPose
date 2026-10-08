# ZED SVO2 + FoundationPose


## D435i 全流程启动（当前版本）

按顺序使用三个终端。启动前停止旧 RealSense 驱动和 RealSense Viewer，确保相机只有一个驱动在运行。

### 1. 相机驱动：开启单条 RGBD 和对齐深度

```bash
source /opt/ros/humble/setup.bash
source ~/realsense_ws/install/setup.bash
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true enable_depth:=true \
  enable_rgbd:=true enable_sync:=true align_depth.enable:=true \
  rgb_camera.color_profile:=640,480,30 \
  depth_module.depth_profile:=640,480,30
```

### 2. 宿主机 RGBD 数据源

```bash
source /opt/ros/humble/setup.bash
source ~/realsense_ws/install/setup.bash
python3 /home/tomato/6Dpose/src/camera_ros_source.py --camera d435i
```

默认订阅 `/camera/camera/rgbd`，无需 Python 双话题配对。默认 `--scale 0.5` 将 640×480 缩到 320×240；需要原分辨率时增加 `--scale 1`。等出现 `RGB-D ready` 后启动 FP。

### 可选：实时 AprilTag 对照

在数据源命令后加参数，不需要改变 FP 启动命令：

```bash
python3 ~/6Dpose/src/camera_ros_source.py --camera d435i --scale 1 \
  --tag-overlay --tag-size 0.075 --tag-id 0 --tag-hz 5
```

当前使用宿主机已有的 `~/apriltag_ws/install/apriltag/lib/` 和其中 Python 3.10 的官方 AprilTag 扩展。此选项默认关闭；启用前需完成该本地 AprilTag 安装。只检测 `tag36h11`，尺寸为黑色正方形外边长，单位米。

Tag 使用独立 CPU 进程，单线程检测，默认最多提交5Hz；输入队列最多一个待处理帧，满时丢帧，RGB-D 不等待检测。绿色为 FP 箱体，青色为 Tag 换算箱体。当前换算假设35cm正方体、标签位于面的中心，沿标签负Z移动175mm；不是通用物体外参。Tag与OBJ具体轴向未独立标定。

叠加采用两结果源时间戳差不超过300ms的参考，屏幕显示时间差；只有源帧号完全相同时才计算中心差值。不同帧的青色框仅供观察，运动时不能当作同帧精度对照，缺失或过期Tag不沿用旧框。

### 实时窗口和性能计时

主位姿窗口默认宽1280像素，保持比例并可拖动缩放。左上 `Image` 是桥接缩放后的完整画面尺寸，`FP input` 是实际推理尺寸（动态ROI时更小）；放大窗口不会增加推理分辨率。

窗口运行在独立进程中，单帧共享缓冲只保存最新画面，FP以非阻塞方式提交，显示慢时允许跳过画面。窗口使用Q/Esc退出、R重新定位。显示进程异常时主流程继续运行并停止GUI提交；退出时清理显示进程。

`Inference` 是当帧推理耗时的倒数，`FP loop`/右上FPS是主流程完成帧率，`Display FPS` 是窗口提交显示的帧率（不等于显示器实际刷新率）。计时保存到运行目录 `loop_timing.csv`：

- `fp_draw_s`、`tag_draw_s`、`text_s`：画框、Tag叠加、文字和颜色转换。
- 新版 `imshow_s`：提交共享缓冲耗时；新版 `waitkey_s`：读取控制命令耗时。它们不再代表主进程调用imshow/waitKey。
- `display_s`：主流程绘图及提交总开销，`inference_s`：推理耗时，`rgbd_wait_s`：获取下一份RGB-D耗时。

修改代码后重启FP程序；启用Tag还需重启宿主机桥接。相机驱动可以继续运行。

### 切换三维模型

保持蓝色目标分割权重，仅切换FP网格示例：

```bash
bash src/start_kfs_fp.sh d435i --mask_mode yolo \
  --mesh /workspace/6Dpose/kfs_model/BlueR1KFS/BlueR1KFS.obj
```

模型、纹理、训练权重属于本地数据，不包含在Git仓库中，需要在对应路径准备。

### 数据源启动报 `Address already in use`

当前版本会自动清理未被占用的残留 `outputs/live_rgbd.sock`。如果提示 `RGB-D source is already running`，先在旧数据源终端按 Ctrl+C，然后重新启动；不要删除正在运行的数据源的 socket。

旧版本或需要手动排查时，先检查进程和 socket 占用：

```bash
pgrep -af '[c]amera_ros_source.py'
ss -xapn | rg 'live_rgbd.sock'
```

若发现旧数据源，先用 Ctrl+C 停止（找不到原终端时，可对确认属于旧数据源的 PID 执行 `kill -INT PID`）。确认两项检查都没有该数据源或 socket 占用后，删除残留文件，再启动：

```bash
rm -f /home/tomato/6Dpose/outputs/live_rgbd.sock
source /opt/ros/humble/setup.bash
source ~/realsense_ws/install/setup.bash
python3 /home/tomato/6Dpose/src/camera_ros_source.py --camera d435i
```

若检查命令报权限错误，不能据此认定没有占用，应先解决检查权限。

### 3. FoundationPose：选择分割方式

```bash
cd /home/tomato/6Dpose
# YOLO：默认最新训练权重 weights/kfs_yolo_seg.pt，类别 blue_kfs
bash src/start_kfs_fp.sh d435i --mask_mode yolo --debug 1
# 或 OpenCV + SAM2
bash src/start_kfs_fp.sh d435i --mask_mode sam --debug 1
```

首次使用 YOLO 自动构建独立镜像，可能需要输入 sudo 密码。项目内 YOLO 权重复制自 `/home/tomato/code/box_seg_annotation/runs/blue_kfs_20261005_150ep_p50/weights/best.pt`。覆盖模型时使用 `--yolo-weights /workspace/6Dpose/weights/other.pt --yolo-class blue_kfs`（容器路径）。

首次初始化保持相机和方块静止。R 重新初始化，Q 或 Esc 退出，Ctrl+C 停止。每次运行创建新的 `outputs/kfs_d435i_*` 目录，实时入口不保存视频。

`--debug 0`（默认）关闭分割窗口；`--debug 1` 开启当前路线的 OpenCV/SAM 或 YOLO 以及 Depth 结果窗口。绿色是 Mask，Depth 窗口蓝色为新增、红色为删除。窗口仅在初始化/恢复时更新，tracking 时保留最后分割帧，不代表当前帧分割。不能与 `--no-display` 同用。

首次/恢复：分割 → Depth 修正 → FP Register。正常帧只运行 FP Tracking；几何检查默认每 2 秒一次，连续 3 次检查失败才恢复。间隔在 `src/config/kfs_blue.json` 的 `health_interval_s` 修改。

性能排查：终端 `Loop:` 分开显示推理、保存、显示、RGBD 等待耗时；输出目录包含 `loop_timing.csv`、`runtime.jsonl` 和退出时生成的 `runtime_summary.json`。数据源每 30 帧打印转换、缩放和输入间隔。不要只用推理时间推算整条流程 FPS。

## KFS 自动分割入口

离线与实时入口已支持可选 `--mask-source opencv-sam2`、安全 Prompt、
`--depth-refinement` 和基于几何检查的自动失跟重定位；默认仍使用手画 Mask。
模型隔离安装、A/B 命令、40 帧人工标注和统计说明见 [KFS_PIPELINE.md](KFS_PIPELINE.md)。

## 原版 FP 实时相机：ZED2i / D435i

新增共用实时入口 `run_foundationpose_live.py --camera zed2i|d435i`，不改变录像程序。
宿主机 `camera_ros_source.py` 订阅相机 ROS2，转换 RGB/米制深度、缩放图像和内参，
通过项目目录里的本地 Unix socket 供原版容器拉取。只缓存最新完整帧，不积压旧画面。
默认话题与当前 ZED / RealSense 默认配置匹配，可用 `--rgb-topic`、`--depth-topic`、
`--info-topic` 覆盖。一次只运行一个相机源，D435i 必须开启深度对齐。

### D435i

宿主机终端 1：

```bash
source /opt/ros/humble/setup.bash
source ~/realsense_ws/install/setup.bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```

宿主机终端 2：

```bash
source /opt/ros/humble/setup.bash
python3 ~/6Dpose/src/camera_ros_source.py --camera d435i
```

终端 3 进入原版容器并运行：

```bash
source ~/6Dpose/activate
python -u /workspace/6Dpose/src/run_foundationpose_live.py --camera d435i
```

### ZED2i

先按已有命令启动宿主机 ZED ROS2 驱动（若已经在运行，不要重复启动）。
宿主机终端 2：

```bash
source /opt/ros/humble/setup.bash
python3 ~/6Dpose/src/camera_ros_source.py --camera zed2i
```

终端 3：

```bash
source ~/6Dpose/activate
python -u /workspace/6Dpose/src/run_foundationpose_live.py --camera zed2i
```

启动前停止 Isaac 推理，避免争用 8GB 显存。实时入口默认加载同一个箱体网格，
可用 `--mesh` 切换；首次必须对当前画面重新画掩码，不加载录像位姿。
鼠标左键画目标轮廓、Enter 确认；首帧定位期间保持相机和物体静止。
随后显示 3D 框、坐标轴、右上角完成帧率；`Q`/Esc 退出、`R` 对当前新画面重新画掩码并定位。
没有自动失跟重定位。FPS 包含取帧和处理间隔，首帧注册不计入稳态统计。

默认源端 `--scale 0.5`；D435i 整数深度默认毫米转米，ZED 浮点深度按米处理。
推理端默认首帧 refine 5 次、tracking refine 2 次、评分 batch 32。
可加 `--max-frames 10` 做短段，或 `--score-batch-size 16` 降低评分显存峰值。
默认不保存视频或每帧 PNG，只保存新时间戳目录中的 `timing.csv` 和手画掩码；
需要位姿文件加 `--save-poses`。`timing.csv` 中耗时是 Python 推理调用墙钟时间，不是独立 CUDA profiler 数据。

程序目前仅完成语法检查和入口参数检查，尚未运行相机/GPU 验证。
相机 ROS 驱动负责彩色标定与深度对齐；不要用原始未对齐深度。
两个 ROS 图像时间戳通过近似同步配对（默认容差 0.05 秒），内参来自彩色话题。
切换相机需停止旧相机源和 FP 入口，再重新启动并定位。正常退出源会删除 socket；
若源被强制杀死，确认源已停止后手动删除 `~/6Dpose/outputs/live_rgbd.sock` 再启动。

这套程序不修改 FoundationPose 原版核心代码：宿主机 ZED SDK 负责解码 SVO2，
FoundationPose Docker 容器负责 6D 位姿估计与后续跟踪。

> 录像完成时必须正常停止录制并关闭 ZED 相机。运行前用 `ls -lh 文件.svo2`
> 确认文件不是 0 字节；0 字节文件不包含任何可恢复的视频帧。

## 在当前 CUDA 13 Docker 环境中运行

先在宿主机进入容器：

```bash
cd /home/tomato/6Dpose
source ./activate
```

容器默认位于 `/workspace/6Dpose/FoundationPose`，运行：

```bash
python -u ../src/run_foundationpose_zed.py
```

所有当前路径和参数都已写为程序默认值。程序会自动复用输出目录中的 mask 和首帧位姿，
并处理完整录像。默认值集中在 `src/run_foundationpose_zed.py` 顶部的“默认配置”区域。
推理默认使用 `--input-scale 0.5`，把 1280x720 的 RGB、深度、mask 和内参同步缩放
到 640x360；使用 `--input-scale 1` 可恢复原分辨率。
默认保存每帧位姿和结果 MP4，但不再额外压缩一千张逐帧 PNG；需要 PNG 时添加
`--save-frame-images`。中断后可用 `--start-frame N` 续跑，程序会自动加载第 N-1 帧姿态。

建议首次在命令末尾加 `--max-frames 10` 验证前 10 帧；确认正常后直接运行上述短命令。
程序默认用 `--score-batch-size 32` 分批评分全部 252 个首帧姿态候选，以适配 8GB 显卡；
如果仍然显存不足可改为 `--score-batch-size 16`。
如果已有成功保存的首帧姿态，可用 `--initial-pose ob_in_cam/000000.txt` 跳过耗时的
首帧注册并直接恢复跟踪。
想忽略已有姿态并重新执行首帧注册时运行
`python -u ../src/run_foundationpose_zed.py --fresh-register`。

默认输入和模型分别是：

- `/home/tomato/6Dpose/zed_photo_capture/svo/recording.svo2`
- `/home/tomato/6Dpose/kfs_model/BlueTrueKFS13/BlueTrueKFS13.obj`

程序第一次运行会显示录像首帧。用鼠标左键沿目标轮廓依次点选，按回车完成；
右键或 `U` 撤销上一点，`R` 清空，`Q`/Esc 退出。生成的 mask 会保存到
`outputs/zed_foundationpose/initial_mask.png`。

结果位于 `outputs/zed_foundationpose/`：

- `ob_in_cam/*.txt`：每帧 4x4 的 object-to-camera 位姿，平移单位为米；
- `track_vis/*.png`：带 3D 包围盒和坐标轴的逐帧结果；
- `result.mp4`：结果视频；
- `initial_mask.png`：首帧目标 mask。
- `run.log`：完整运行日志；如果推理提前退出，错误会保留在这里。

首帧 `register` 比后续跟踪慢很多，第一次运行还可能编译 CUDA kernel。标完 mask 后，
窗口会显示 `Loading FoundationPose models` 和 `Registering first frame`；首帧完成后才开始
逐帧显示 3D 包围盒。再次运行可用 `--mask` 复用已保存的 mask。

## 分步运行和复用 mask

```bash
cmake -S src -B src/build
cmake --build src/build -j"$(nproc)"

src/build/zed_svo_export \
  --svo /home/tomato/6Dpose/zed_photo_capture/svo/recording.svo2 \
  --output /home/tomato/6Dpose/outputs/zed_sequence \
  --stride 1

python ../src/run_foundationpose_zed.py \
  --sequence /workspace/6Dpose/outputs/zed_sequence \
  --mask /workspace/6Dpose/outputs/zed_foundationpose/initial_mask.png \
  --output /workspace/6Dpose/outputs/zed_foundationpose
```

录像很长时可在导出命令中添加 `--start-frame`、`--end-frame` 和 `--stride`。

## D435i bag 离线导出与评估数据

`export_d435_bag.py` 在宿主机 ROS2 环境运行，将当前 sqlite bag 导出为原版 FP 可读取的 PNG 序列。
精确匹配 RGB、对齐深度、彩色内参的源 header 时间戳；不做近邻重配、不编造缺失帧。
`rgb/`、`depth/` 含完整匹配帧，命名使用原始 RGB 帧序号；`frames.csv` 记录匹配序号、
原始帧号和纳秒时间戳。`all_frames.csv` 含全部 RGB 的匹配/缺失状态，未匹配 RGB 保存在
`rgb_unmatched/`，可供 YOLO 标注或独立 tag 检测。深度 PNG 为 16UC1 毫米，原版 FP 读入后转米。
`camera_calibration.json` 保存畸变系数及完整内参；`cam_K.txt` 供现有 FP 入口使用。
导出不会覆盖已有输出目录；失败目录是部分结果，不应视为导出完成，以 export_summary.json 为完成标记。

```bash
source /opt/ros/humble/setup.bash
python3 ~/6Dpose/src/export_d435_bag.py   ~/6Dpose/recordings/d435i_static_20261002_225751   ~/6Dpose/outputs/d435i_static_sequence
```

动态序列对应 `d435i_dynamic_20261002_225839` 和 `outputs/d435i_dynamic_sequence`。
这两组序列目前用于评估准备；FP/tag 位姿尚未运行。标签与模型固定 TF 还需确认，
对称正方体的旋转误差需考虑实际几何/纹理对称性。贴标签数据不能代替无标签 YOLO 训练场景。

### RGBD 接收 QoS 修复

D435i 的单条 RGBD 使用 Reliable、Keep Last depth=1。实测同话题 Best Effort 约 2.79 Hz、Reliable 约 30 Hz；较大的 RGBD 消息在本机 Best Effort 接收出现严重丢帧。空间对齐和同步仍在 RealSense 驱动完成，Python 不再次配对。修改后重启数据源即可，无需重建 FP 镜像。

## 动态 ROI 跟踪（实时自动分割模式）

SAM 和 YOLO 实时路线默认开启 `--dynamic-roi 1`。首次全图分割/register，随后用上一帧有效 FP 位姿投影 Mesh bbox，左右各外扩 bbox 宽度的 30%、上下各外扩高度的 30%（`--roi-margin 0.3`），请求下一帧 ROI。宿主机裁剪 RGB/Depth，不额外缩放，并更新内参主点；FP 位姿始终保持原相机坐标系。完整 RGB 与 ROI 来自同一帧，预览显示全画面、位姿框和黄色 ROI 边界。只传 ROI 深度；全图 RGB 仍为预览传输，所以没有完全消除全图传输/显示开销。

更新后必须同时重启数据源和 FP（socket 新增 ROI 请求协议），无需重建镜像。为保留高分辨率像素，数据源显式设置 `--scale 1`：

```bash
source /opt/ros/humble/setup.bash
source ~/realsense_ws/install/setup.bash
python3 /home/tomato/6Dpose/src/camera_ros_source.py --camera d435i --scale 1
# 另一个终端
cd /home/tomato/6Dpose
bash src/start_kfs_fp.sh d435i --mask_mode yolo --dynamic-roi 1 --roi-margin 0.3
```

无有效 pose、ROI 超出视野而不可用、失跟或按 R 时请求全图；失跟后重新全图分割。正常 tracking 不逐帧运行检测器。若 FP 错误位姿仍通过健康检查，ROI 也可能跟随漂移；当前尚未添加独立 RGB 位置验证。快速运动可能跑出上一帧预测 ROI，可增大 margin。关闭使用 `--dynamic-roi 0`。`loop_timing.csv` 追加当前 ROI 边界，`runtime.jsonl` 包含宿主机 ROI 裁剪耗时 `source_roi_crop_s`。离线流程未开启动态 ROI；手动 Mask 路线继续全图。

## 分辨率 / 帧率静态对比测试

固定相机和方块，停止旧 FP、数据源和相机驱动后执行：

```bash
cd /home/tomato/6Dpose
bash src/run_kfs_benchmark.sh
```

脚本要求终端输入 sudo 密码（如 Docker 需要），自动启动/停止每组相机和数据源，YOLO + Depth + FP，原分辨率 `scale=1`，不加 ROI，不录视频。默认每组 360 帧、关闭窗口；如要把主预览开销计入测试，四组统一使用 `--display`。每次 Register 后前 30 个 tracking 帧不计入指标。输出 `outputs/kfs_benchmark_*/summary.csv` 和 `summary.json`，包括稳态帧率、逐帧位移抖动 p50/p95（mm）和旋转抖动 p50/p95（deg）、各环节耗时及显存峰值。抖动不是绝对位姿误差；不同处理频率会影响逐帧变化量。

实机支持四组：RGB 1920×1080@30 / Depth 1280×720@30；RGB/Depth 1280×720@30；RGB/Depth 640×480@30、@60。1080p60 和 720p60 彩色不支持，记录为 unsupported。1080p 的对齐深度虽映射到 1080p 像素网格，其原生深度分辨率仍为 720p；不能解释为 1080p 原生测距精度。脚本核对输出尺寸和驱动回报参数，回退配置记为失败而不伪装成所请求配置。

测试后相机自动停止；按正常启动步骤恢复。镜像构建和真实四组测试需在本机实际执行；CPU 单元测试不能证明实测性能。

分辨率 benchmark 默认 `--dynamic-roi 0`，保持原全图实验可比；加入 `--dynamic-roi 1` 可运行动态 ROI 对照实验。
