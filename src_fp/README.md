# FoundationPose / KFS 运行指南

FP、KFS 分割和 RGB-D 桥接集中在本目录。CPU Box4D 与 YOLO 工具见 [src 目录导航](../src/README.md)。

- Python：本目录中的实时/离线入口和 FP 辅助模块。
- `scripts/`：容器与环境启动脚本。
- `config/`：FP 分割参数。
- `tests/`：FP、桥接和 ROI 回归测试。
- `docs/`：KFS 完整流程。
- `Dockerfile.kfs*`、SAM 锁记录和依赖：FP 扩展镜像构建。
- `CMakeLists.txt` / `zed_svo_export.cpp`：ZED SVO 导出程序。

共用的显示与静止 IMU 数学位于 `src/common/`；FP 可执行入口已配置好这个公共目录。

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
python3 /home/tomato/6Dpose/src_fp/camera_ros_source.py --camera d435i
```

默认订阅 `/camera/camera/rgbd`，无需 Python 双话题配对。默认 `--scale 0.5` 将 640×480 缩到 320×240；需要原分辨率时增加 `--scale 1`。等出现 `RGB-D ready` 后启动 FP。

### 可选：实时 AprilTag 对照

在数据源命令后加参数，不需要改变 FP 启动命令：

```bash
python3 ~/6Dpose/src_fp/camera_ros_source.py --camera d435i --scale 1 \
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

### 新红蓝模型的 FP 训练数据采集

在桌面终端运行 `bash src/start_tkk_capture.sh`。启动器使用已安装的 FP 镜像，
需要在终端完成 sudo 认证；自动启动 D435i 640×480@30、全分辨率 RGB-D 数据源，
以及 `kfs_model/tkk/tkk.obj` 的手工首帧注册和跟踪。已有相机驱动时应先停止，
启动器通过相机锁拒绝重复占用。退出 FP 后会向本次启动的相机和源发送 Ctrl+C。

首帧用左键圈住整个红蓝方块的可见轮廓，Enter 确认；首次注册时保持相机和方块静止。
定位成功后自动每秒保存一组。主窗口 `S` 额外保存一组、`C` 暂停/继续每秒采样、
`R` 暂停采样并重新圈选定位（定位后可按 C 继续）、`Q` 退出。
整块方块是一个类别 `tkk`（ID 0），采集期间关闭动态 ROI。

每次创建新目录 `outputs/tkk_capture_时间戳/`，其中 `dataset/` 包含：

- `images/`：无文字、框或坐标轴的原始彩色 PNG；`depth/`：同帧浮点米制 NPY。
- `ob_in_cam/`、`intrinsics/`：同帧物体位姿和相机内参。
- `masks/`：FP 模型的完整投影；`labelme/`：待复核多边形 JSON；`labels/`：YOLO 分割多边形。
- `preview/`：投影叠加图；`manifest.jsonl`：对应源帧号、时间戳及复核状态。

这些是待复核的投影伪标签，不剔除手或其他物体的遮挡。修改 Labelme JSON 后需要
重新生成 YOLO 标签；此流程不会自动导入现有训练集。相机、桥接和 FP 日志保存在
本次运行目录中。实时入口也可单独用 `--capture-dir 新目录 --capture-class tkk
--capture-interval 1 --capture-auto` 启用自动采样。

整理全部新红蓝模型照片为 YOLO 实例分割快照：

```bash
cd /home/tomato/6Dpose
/home/tomato/code/box_seg_annotation/env/bin/python src/tools/yolo/prepare_tkk_yolo_seg.py
```

转换器扫描 `outputs/*/dataset/capture_config.json` 中类别为 `tkk` 的批次，
根据当前 Labelme JSON 重新生成类别 0 的分割多边形。每次创建新的
`outputs/tkk_yolo_seg_时间戳/`，包含 `data.yaml`、训练/验证图片和标签、
二值 mask、全部叠加预览及来源清单；按完整拍摄日期留出验证集，原采集数据不改动。
补拍完成后再次运行即可将新照片一起转换。转换期间应先暂停采样。

FP 跟踪漂移会产生错位标签，格式检查通过不代表轮廓正确。转换器将颜色覆盖率过低的
疑似漂移/遮挡帧列在 `review_flags.csv`，但全部照片仍保留在 `data.yaml` 中；
其他照片也应复核。请先在源 `dataset/labelme/` 中修正，再重新转换并训练。
训练命令保存在新快照的 `README_中文.md`，转换不会自动启动训练。

人工复核后训练使用 `train_tkk_seg.py`，其 `--snapshot` 指向在 Labelme 中修改的
`tkk_yolo_seg_*` 快照。程序直接读取该快照的最新 `labelme/`，备份 JSON 后重新生成
训练标签，保留拍摄日期划分，排除空标注图片，避免把删掉轮廓的目标图片当负样本。
用户已明确确认复核完成时，保存的标注可用于训练，不要求额外勾选 `reviewed` 标志。
默认 YOLO11n-seg、640 输入、batch 8、AdamW、AMP，完整训练 400 轮并关闭早停。
`--job` 指定新的数据和日志目录，`--name` 指定 `runs/` 下新的训练名称；支持
`--prepare-only` 只导出，以及 `--resume /绝对路径/last.pt` 从同一 job 的检查点恢复。
训练结束保存最佳/最后权重、汇总和验证集预测叠加图；每轮进度在 job 的 `status.json`。

训练后对比两个 TKK 分割模型：

```bash
bash /home/tomato/6Dpose/src/run_tkk_seg_compare.sh
```

同一个窗口左侧显示 `best.pt`，右侧显示第 276 轮的 `epoch275.pt`，共享同一帧相机原图和
推理阈值。启动器复用已有相机；需要新启动时只开启 640×480@30 彩色流。
空格保存两侧结果，`+` / `-` 同时调阈值，`Q` 退出；详见 [双模型对比说明](../src/docs/yolo/TKK_SEG_COMPARE_中文.md)。

补拍容易误检的纯背景负样本：

```bash
bash /home/tomato/6Dpose/src/capture_tkk_background.sh
```

先把真正的 `tkk` 移出画面，保留需要抑制误检的背景物体，每次确认无目标后按空格拍一张，
`Q` 结束。原始 PNG、空 YOLO 标签、`shapes=[]` 的 Labelme JSON 和逐张确认清单保存在
`outputs/tkk_background_时间戳_PID/`；不会生成 FP 投影。拍摄器复用已有相机驱动，
没有驱动时自动开启彩色流。若打算关闭已有的对比窗口，请先关闭再启动拍摄，让拍摄器拥有相机。
另一次独立验证批次可加 `--split val`；`--check` 只检查收图，不保存或确认负样本。
`manifest.jsonl` 中 `confirmed_negative=true` 和 `confirmation=manual_SPACE_no_TKK` 是
人工确认依据；拍完后应按该清单另行纳入新的训练快照，不能用“所有空标注都当负样本”的方式导入。

`train_tkk_seg.py --background-source /背景采集目录` 可将上述确认负样本加入新训练，
参数可重复。多批次优先整批留出验证；仅有一个批次时，中间 20% 连续图片用于验证，
两侧各隔离 10 张，报告会明确标为同次拍摄的连续留出。正样本保留原划分。
`--model /某轮权重.pt --lr0 0.0003` 在新数据上微调指定权重；`--epochs` 表示本轮新增的
训练轮数。增加数据时使用新 job 和 run，`--resume` 仅恢复本轮已有的冻结数据快照。
加入背景后会在训练前和训练结束分别以置信度 0.5 统计背景验证图片的误检情况。
后台训练清单如有 `comparison_reference_weights`，实时对比右侧使用该原始权重，
左侧使用最新训练的 `best.pt`，便于比较补拍前后的效果。

### 切换三维模型

2026-10-21：实时与离线 FP 入口的默认模型为 `kfs_model/tkk/tkk.obj`，
现已从桌面新的 SolidWorks `tkk.glb` 解压转换，保留上半红色、下半蓝紫色的两种材质。
外形约为 0.20 × 0.20 × 0.20 m，原点位于底面中心；保留源模型的米制坐标，
没有另行缩放或旋转。源 GLB 使用 Draco 压缩，解压后的高度约 0.2000061 m。
FP 运行版将两种材质颜色写入对应顶点，以使用当前 FP 的无纹理渲染分支。
标准材质版在同目录 `tkk.materials.obj` 和 `tkk.mtl`，也已打包为
`tkk_obj_with_materials.zip`（包内文件名为 `tkk.obj`、`tkk.mtl`）。
原始 GLB 另存为 `tkk.source.glb`；导入及验证信息在 `import.json`，
之前导入的灰色 OBJ / MTL 和运行模型保留在同目录 `backups/` 中。
旧的 `BlueTrueKFS13` 模型完整保留。分割权重仍为 `blue_kfs`；换模型后须重新定位，
离线运行使用新的 `--output` 目录及 `--fresh-register`，避免复用旧模型的位姿。
下文及历史实验文档中的 35 cm 箱体 / AprilTag 外参仍对应旧模型。

实时窗口默认用 OpenCV 识别目标范围内的实际红蓝颜色，并结合对齐深度显示方向文字：

- `redup` / `blueup`：实际朝上的颜色，以重力为参照。
- `redfront` / `bluefront`：平放时更靠近相机的一侧的颜色。
- `redleft` / `blueleft`：平放时左侧的颜色。

FP 只提供目标的几何范围和三个无符号轴；每帧用当前 RGB-D 重新判断六个方向的颜色，
不把模型的 +Z / −Z 当作颜色方向。立方体对称导致 FP 轴置换或反向时，标签仍由颜色决定。
仅 `redup` / `blueup` 额外显示对应颜色的大箭头，从目标中心指向朝上的颜色侧；
前后、左右和 `unknown` 状态只显示文字。箭头长度约为画面短边的 1/4，和当前帧一起绘制。
俯视时如果实际向上方向投影成一点，则使用屏幕向上的箭头作为朝上标记。
颜色和深度不足、方向接近分界、没有重力参照时显示 `unknown`，
不沿用上一帧标签；FP 整体定位偏离目标时需按 R 重定位。

D435i 图像桥接默认订阅 `/camera/camera/accel/sample`，使用 ROS TF 转到彩色光学坐标系。
相机驱动需要 `enable_accel:=true`；`start_box4d_camera.sh` 和 `start_tkk_capture.sh` 已开启。
启动后保持相机静止约 1 秒；移动相机时暂不报告实际上下，静止后自动恢复。
升级后须重启图像桥接和 FP；旧桥接没有重力信息。其他相机可配置桥接 `--imu-topic`，
或给 FP 指定已标定的固定 `--up-camera X Y Z`（相机姿态改变后须重新标定）。

颜色判断只采样目标局部最多约 6000 个像素，与当前帧一起提交到现有单帧非阻塞显示；
不累计图像，也不做标签时间平滑。逐帧结果保存在输出目录 `color_orientation.jsonl`，
耗时记入 `runtime.jsonl` 的 `color_orientation_s`。关闭功能使用 `--color-orientation off`；
旧的 `--blue-arrow off` 参数仍兼容。

保持蓝色目标分割权重，仅切换FP网格示例：

```bash
bash src_fp/scripts/start_kfs_fp.sh d435i --mask_mode yolo \
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
python3 /home/tomato/6Dpose/src_fp/camera_ros_source.py --camera d435i
```

若检查命令报权限错误，不能据此认定没有占用，应先解决检查权限。

### 3. FoundationPose：选择分割方式

```bash
cd /home/tomato/6Dpose
# YOLO：默认最新训练权重 weights/kfs_yolo_seg.pt，类别 blue_kfs
bash src_fp/scripts/start_kfs_fp.sh d435i --mask_mode yolo --debug 1
# 或 OpenCV + SAM2
bash src_fp/scripts/start_kfs_fp.sh d435i --mask_mode sam --debug 1
```

首次使用 YOLO 自动构建独立镜像，可能需要输入 sudo 密码。项目内 YOLO 权重复制自 `/home/tomato/code/box_seg_annotation/runs/blue_kfs_20261005_150ep_p50/weights/best.pt`。覆盖模型时使用 `--yolo-weights /workspace/6Dpose/weights/other.pt --yolo-class blue_kfs`（容器路径）。

首次初始化保持相机和方块静止。R 重新初始化，Q 或 Esc 退出，Ctrl+C 停止。每次运行创建新的 `outputs/kfs_d435i_*` 目录，实时入口不保存视频。

`--debug 0`（默认）关闭分割窗口；`--debug 1` 开启当前路线的 OpenCV/SAM 或 YOLO 以及 Depth 结果窗口。绿色是 Mask，Depth 窗口蓝色为新增、红色为删除。窗口仅在初始化/恢复时更新，tracking 时保留最后分割帧，不代表当前帧分割。不能与 `--no-display` 同用。

首次/恢复：分割 → Depth 修正 → FP Register。正常帧只运行 FP Tracking；几何检查默认每 2 秒一次，连续 3 次检查失败才恢复。间隔在 `src_fp/config/kfs_blue.json` 的 `health_interval_s` 修改。

性能排查：终端 `Loop:` 分开显示推理、保存、显示、RGBD 等待耗时；输出目录包含 `loop_timing.csv`、`runtime.jsonl` 和退出时生成的 `runtime_summary.json`。数据源每 30 帧打印转换、缩放和输入间隔。不要只用推理时间推算整条流程 FPS。

## KFS 自动分割入口

离线与实时入口已支持可选 `--mask-source opencv-sam2`、安全 Prompt、
`--depth-refinement` 和基于几何检查的自动失跟重定位；默认仍使用手画 Mask。
模型隔离安装、A/B 命令、40 帧人工标注和统计说明见 [KFS_PIPELINE.md](docs/KFS_PIPELINE.md)。

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
python3 ~/6Dpose/src_fp/camera_ros_source.py --camera d435i
```

终端 3 进入原版容器并运行：

```bash
source ~/6Dpose/activate
python -u /workspace/6Dpose/src_fp/run_foundationpose_live.py --camera d435i
```

### ZED2i

先按已有命令启动宿主机 ZED ROS2 驱动（若已经在运行，不要重复启动）。
宿主机终端 2：

```bash
source /opt/ros/humble/setup.bash
python3 ~/6Dpose/src_fp/camera_ros_source.py --camera zed2i
```

终端 3：

```bash
source ~/6Dpose/activate
python -u /workspace/6Dpose/src_fp/run_foundationpose_live.py --camera zed2i
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
python -u ../src_fp/run_foundationpose_zed.py
```

所有当前路径和参数都已写为程序默认值。程序会自动复用输出目录中的 mask 和首帧位姿，
并处理完整录像。默认值集中在 `src_fp/run_foundationpose_zed.py` 顶部的“默认配置”区域。
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
`python -u ../src_fp/run_foundationpose_zed.py --fresh-register`。

默认输入和模型分别是：

- `/home/tomato/6Dpose/zed_photo_capture/svo/recording.svo2`
- `/home/tomato/6Dpose/kfs_model/tkk/tkk.obj`

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
cmake -S src_fp -B build/zed_svo_export
cmake --build build/zed_svo_export -j"$(nproc)"

build/zed_svo_export/zed_svo_export \
  --svo /home/tomato/6Dpose/zed_photo_capture/svo/recording.svo2 \
  --output /home/tomato/6Dpose/outputs/zed_sequence \
  --stride 1

python ../src_fp/run_foundationpose_zed.py \
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
python3 ~/6Dpose/src_fp/export_d435_bag.py   ~/6Dpose/recordings/d435i_static_20261002_225751   ~/6Dpose/outputs/d435i_static_sequence
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
python3 /home/tomato/6Dpose/src_fp/camera_ros_source.py --camera d435i --scale 1
# 另一个终端
cd /home/tomato/6Dpose
bash src_fp/scripts/start_kfs_fp.sh d435i --mask_mode yolo --dynamic-roi 1 --roi-margin 0.3
```

无有效 pose、ROI 超出视野而不可用、失跟或按 R 时请求全图；失跟后重新全图分割。正常 tracking 不逐帧运行检测器。若 FP 错误位姿仍通过健康检查，ROI 也可能跟随漂移；当前尚未添加独立 RGB 位置验证。快速运动可能跑出上一帧预测 ROI，可增大 margin。关闭使用 `--dynamic-roi 0`。`loop_timing.csv` 追加当前 ROI 边界，`runtime.jsonl` 包含宿主机 ROI 裁剪耗时 `source_roi_crop_s`。离线流程未开启动态 ROI；手动 Mask 路线继续全图。

## 分辨率 / 帧率静态对比测试

固定相机和方块，停止旧 FP、数据源和相机驱动后执行：

```bash
cd /home/tomato/6Dpose
bash src_fp/scripts/run_kfs_benchmark.sh
```

脚本要求终端输入 sudo 密码（如 Docker 需要），自动启动/停止每组相机和数据源，YOLO + Depth + FP，原分辨率 `scale=1`，不加 ROI，不录视频。默认每组 360 帧、关闭窗口；如要把主预览开销计入测试，四组统一使用 `--display`。每次 Register 后前 30 个 tracking 帧不计入指标。输出 `outputs/kfs_benchmark_*/summary.csv` 和 `summary.json`，包括稳态帧率、逐帧位移抖动 p50/p95（mm）和旋转抖动 p50/p95（deg）、各环节耗时及显存峰值。抖动不是绝对位姿误差；不同处理频率会影响逐帧变化量。

实机支持四组：RGB 1920×1080@30 / Depth 1280×720@30；RGB/Depth 1280×720@30；RGB/Depth 640×480@30、@60。1080p60 和 720p60 彩色不支持，记录为 unsupported。1080p 的对齐深度虽映射到 1080p 像素网格，其原生深度分辨率仍为 720p；不能解释为 1080p 原生测距精度。脚本核对输出尺寸和驱动回报参数，回退配置记为失败而不伪装成所请求配置。

测试后相机自动停止；按正常启动步骤恢复。镜像构建和真实四组测试需在本机实际执行；CPU 单元测试不能证明实测性能。

分辨率 benchmark 默认 `--dynamic-roi 0`，保持原全图实验可比；加入 `--dynamic-roi 1` 可运行动态 ROI 对照实验。

## CPU 回归测试

从仓库根目录运行：

```bash
CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src_fp:src \
  /home/tomato/code/box_seg_annotation/env/bin/python -m unittest discover -s src_fp/tests
```

这些测试使用模拟 FP/分割后端和录制/合成数据，不会打开相机或启动 GPU 推断。
