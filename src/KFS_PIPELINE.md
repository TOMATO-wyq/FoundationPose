# OpenCV + SAM2 + Depth + FoundationPose

第一版只估计一个蓝色 KFS，使用现有 BlueTrueKFS13 的 0.35m 网格。
自动分割是显式选项，现有手画 Mask 流程仍为默认。

## 快速测试入口

从宿主机项目目录运行，首次自动构建独立镜像，sudo 密码由用户在终端输入：

```bash
cd /home/tomato/6Dpose
bash src/start_kfs_fp.sh offline --max-frames 10
```

实时模式先保持相机 ROS 驱动运行，在另一终端启动相应 RGB-D 源：

```bash
source /opt/ros/humble/setup.bash
python3 /home/tomato/6Dpose/src/camera_ros_source.py --camera zed2i
```

然后运行 `bash src/start_kfs_fp.sh zed2i`；使用 D435i 时两条命令都换成 d435i，
并确保相机驱动开启对齐深度。启动器默认开启 Depth Refinement、保存有效位姿，
每次生成新的输出目录；有 DISPLAY 时传入 X11 设置，没有则自动使用无窗口模式。
首次注册时保持目标与相机静止，成功后只跟踪，`R` 重定位、`Q`/Esc 退出。
宿主机已有 socket 不等于相机源仍运行；若源提示 Address already in use，请先确认旧源已停止，
再按 README 的残留 socket 处理步骤操作。

## 已提供的接口

`run_foundationpose_zed.py` 和 `run_foundationpose_live.py` 均支持：

```text
--mask-source manual|opencv-sam2
--segmentation-config src/config/kfs_blue.json
--sam-checkpoint PATH
--sam-checkpoint-sha256 SHA256
--sam-model-config configs/sam2.1/sam2.1_hiera_s.yaml
--sam-device cuda|cpu
--prompt-mode box|box1|box3|box3neg
--depth-refinement
```

默认 `box1`。A 不加 `--depth-refinement`，B 加该参数。
两组使用相同 RGB、深度和 FP 参数；Depth Refinement 的中值滤波仅用于 Mask 判断。
不要同时更改 `--depth-spatial-filter` 来比较 A/B。

RGB 为 RGB uint8，对齐深度为 float32 米，内参属于同一彩色图像，Mask 为 bool。
离线序列的 uint16 PNG 深度在入口毫米转米。像素阈值属于缩放后的推理分辨率。
自动模式不加载已有 Mask 或初始位姿；必须用新输出目录，不覆盖旧位姿文件。
自动模式的 `--start-frame` 从指定帧重新检测注册，不恢复之前的跟踪。

## 模型与隔离环境

代码、权重 URL 和 SHA256 在 `sam2-lock.json` 中固定：

```bash
cd /home/tomato/6Dpose
python3 src/setup_kfs_sam2.py
python3 src/setup_kfs_sam2.py --verify-only
```

SAM2 使用官方 SAM2.1 Small，不启用自动填洞、sprinkle removal 或 Mask decoder 的自动回退。
同帧 Prompt 比较共用 embedding；正常 FP 跟踪不运行 SAM。
实时/FP 入口的每次分割结束后，清除 embedding、SAM 模型移回 CPU、释放空闲 CUDA 缓存。
同时清除未注册为 buffer 的位置编码缓存与 RoPE 表；SAM 调用期间临时使用 CPU tensor factory 默认值，
结束后恢复 FP 的全局设置，隔离图像预处理的 CPU 默认值。
底层数学库预热后可保留少量 allocator 工作区；offload 检查要求 SAM 参数、buffer 和缓存回到 CPU，
并检查重复循环相对预热基线没有持续增长。

本机分割测试复用已有 Python 的 Torch 库，只把额外依赖安装到本项目 `.kfs-deps`，
没有修改 `box_seg_annotation/env`：

```bash
/home/tomato/code/box_seg_annotation/env/bin/python -m pip install \
  --target .kfs-deps -r src/requirements-sam2-extra.txt
KFS_PYTHON=/home/tomato/code/box_seg_annotation/env/bin/python \
  bash src/run_kfs_python.sh src/evaluate_kfs_masks.py --help
```

其他机器可将 `KFS_PYTHON` 指向独立安装、满足 SAM2 要求的 Python 环境。
FP + SAM2 使用新派生镜像，不升级已有 `foundationpose:rtx50-cu130`：

```bash
python3 src/setup_kfs_sam2.py --verify-only
sudo docker build --network=host \
  --build-context sam2=./third_party/sam2 \
  -t foundationpose:kfs-sam2 -f src/Dockerfile.kfs src
```

该镜像沿用已有 FP 的 Torch/CUDA，只安装额外 Python 包；不编译 SAM 的可选填洞 CUDA 扩展。
下面的离线验证无需桌面或相机：

```bash
sudo docker run --rm --gpus all --shm-size=4g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp/matplotlib \
  -v /home/tomato/6Dpose:/workspace/6Dpose \
  -w /workspace/6Dpose/FoundationPose foundationpose:kfs-sam2 \
  python ../src/run_foundationpose_zed.py \
    --mask-source opencv-sam2 \
    --sam-checkpoint ../third_party/sam2/checkpoints/sam2.1_hiera_small.pt \
    --segmentation-config ../src/config/kfs_blue.json \
    --output ../outputs/kfs_fp_A --max-frames 10 --no-display --no-video
```

运行 B 时换成新目录 `../outputs/kfs_fp_B`，并加 `--depth-refinement`。
脚本从原序列取同一组帧、每次独立注册，不复用旧位姿。

## 分割验证与人工真值

已准备 `outputs/kfs_validation_v1/manifest.json`：ZED 14 帧、D435i 静态命名序列 13 帧、
D435i 动态序列 13 帧，20 tune / 20 eval。均匀采样仅是候选集，须人工确认场景覆盖。
新建其他验证集：

```bash
python3 src/prepare_kfs_validation.py --output outputs/kfs_validation_v2
python3 src/annotate_kfs_validation.py outputs/kfs_validation_v1/manifest.json
```

标注窗口：左键画多边形，`A` 加入可见区域，`X` 减去手/遮挡，右键撤销点，
`U` 撤销，`C` 清空，没有未完成多边形时 Enter 保存，`Q` 停止。
可叠加多个可见组件，不强行补全立方体；无目标帧可保存空 Mask。
只有手工确认保存后，manifest 的 `reviewed` 才为 true。
用 `--scenario` 指定标注场景；混合场景需人工校正 manifest 的 scenario。

```bash
KFS_PYTHON=/home/tomato/code/box_seg_annotation/env/bin/python \
  bash src/run_kfs_python.sh src/evaluate_kfs_masks.py \
    --manifest outputs/kfs_validation_v1/manifest.json --split eval \
    --sam-checkpoint third_party/sam2/checkpoints/sam2.1_hiera_small.pt \
    --segmentation-config src/config/kfs_blue.json \
    --prompt-modes box box1 box3 box3neg \
    --output outputs/kfs_masks_eval
```

先仅用 `--split tune` 调 HSV/深度阈值，冻结配置，再使用 eval。
没有人工真值时 IoU/Boundary F1 为 null；分割失败按空预测计分，不能从均值里丢掉失败帧。
Boundary F1 容差为推理分辨率的 2 px。

`mask_summary.json` 包含 Prompt/A/B 成功生成 Mask 的数量、质量指标、各阶段 p50/p95。
“Mask 生成成功”不是 IoU 合格、对象身份确认或位姿成功。
目录中保存 coarse/safe/raw/refined、Candidate Band、深度边缘、added/removed、Prompt 和叠图。

## 实时接入

复用 README 中的 ZED ROS2 驱动和 `camera_ros_source.py --camera zed2i`。
启动宿主机源后，推理容器通过挂载目录中的 Unix socket 取最新完整帧。
将上述容器命令的 Python 部分替换为：

```bash
python ../src/run_foundationpose_live.py --camera zed2i \
  --mask-source opencv-sam2 \
  --sam-checkpoint ../third_party/sam2/checkpoints/sam2.1_hiera_small.pt \
  --segmentation-config ../src/config/kfs_blue.json \
  --depth-refinement --save-poses --no-display
```

容器需要同样的项目 bind mount；Unix socket 不需要 GPU 相机 SDK。
带窗口运行时采用原 `cuda130.sh` 中的 DISPLAY、X11 socket、XAUTHORITY 挂载方式，
改用派生镜像。`R` 对最新帧重新检测注册，`Q`/Esc 退出。

状态：SEARCH → REGISTER → TRACK → LOST。
几何健康检查使用当前 Mesh 渲染深度与测量深度，前方遮挡不计为目标支持。
默认支持率 ≥30%、残差中位数 ≤30mm，最少 64 个有效支持像素；
连续 3 帧失败转 LOST，非有限/无效姿态立即 LOST。每个检查失败帧均不输出位姿。
LOST 清空 FP 内部跟踪姿态，默认每 0.5 秒重试；离线模式逐帧重试。
这不是 FP confidence，CUDA/OOM/模型环境错误会直接终止，而非无限重试。

## 安全边界与统计

- Mask 外无效深度不加入，Mask 内无效深度保留 SAM 的视觉判断但不作为生长桥梁。
- 生长半径相对 Raw Mask 固定，不随区域扩大累积；原始及中值深度均须局部连续。
- Closing 默认关闭；开启后的新增像素再次检查深度、边缘与距离。
- 默认 Negative 深度间隔 0.65m，超过当前 0.35m 方块的空间对角线；换 Mesh 时需重新设置。
- 接触背景/手与物体深度接近时，深度不能保证分离；严重遮挡、无目标、同色干扰、重入应补采验证。
- **录像名 static 不保证整段静止**：现有 static 序列包含搬动物体的片段，不能整段当静态抖动真值。

FP 输出 `events.jsonl`、`runtime.jsonl`、`runtime_summary.json`；失败帧没有 pose TXT。
日志包含注册成功数/尝试数、几何健康状态、各阶段耗时、稳态 FPS、CPU 核使用率和 RSS。
GPU 数字为 PyTorch allocator 的 allocated/reserved 峰值，不包含驱动和其他进程。
模型加载在计时前完成；正常入口的分割总时间包含编码、解码、Mask 判断和模型 offload。
Prompt 比较工具的单帧时间包含所有实验，不是部署 FPS。

```bash
python3 src/compare_kfs_pose_runs.py --a outputs/kfs_fp_A --b outputs/kfs_fp_B \
  --output outputs/kfs_fp_comparison.json
```

仅在相机和目标都确实静止的同一片段运行时添加 `--static-scene`。
抖动是相邻有效帧的平移/旋转变化代理，不跨越缺失帧，不处理立方体旋转对称性。
动态准确率可复用现有 `evaluate_fp_apriltag.py`，前提是标签变换与参考位姿已经确认。

CPU 回归测试（不需要 Torch/SAM/GPU）：

```bash
PYTHONPATH=src python3 -m unittest discover -s src/tests -v
```

### D435i 单条 RGBD 与性能诊断

启动 RealSense 时开启 `enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true`。数据源默认直接订阅 `/camera/camera/rgbd`（包含 RGB、Depth 和内参），不再进行 Python 双图像配对。数据源终端须 source `/opt/ros/humble/setup.bash` 和 `~/realsense_ws/install/setup.bash`，以加载自定义 RGBD 消息。ZED 保持原双话题同步；`--input-mode separate` 可显式使用旧方式。默认 `--scale 0.5`；`--scale 1` 保持原分辨率。

`runtime.jsonl` 包含输入转换、缩放清理、输入帧间隔和 RGB/Depth 时间戳差；`loop_timing.csv` 包含 FP、位姿保存、显示和 RGB-D 请求接收时间。数据源每 30 帧输出输入计时。未额外录制视频。

`health_interval_s` 默认 2 秒，0 表示逐帧检查。初始化仍立即检查；非法位姿每帧拒绝。`failure_frames=3` 现在表示连续三次实际几何检查失败，跳过检查不清空计数。因此持续失败通常需约 4–6 秒才触发恢复，期间跳过几何检查的位姿不代表经过当前帧深度验证。仍是深度代理判断，尚未加入 RGB 身份验证或遮挡专用状态。

### SAM / YOLO 切换

`bash src/start_kfs_fp.sh d435i --mask_mode sam` 使用 OpenCV HSV/轮廓产生提示，再调用 SAM2。`bash src/start_kfs_fp.sh d435i --mask_mode yolo` 直接使用 YOLO 实例分割，不运行 HSV/SAM。两者共用 Depth refinement、FP Register/Tracking 和恢复状态机。离线入口也支持相同参数。旧 `--mask-source opencv-sam2` 仍兼容。

YOLO 启动脚本默认权重 `weights/kfs_yolo_seg.pt`，来自 `blue_kfs_20261005_150ep_p50/weights/best.pt`，目标类别 `blue_kfs`。可覆盖 `--yolo-weights /workspace/6Dpose/weights/other.pt --yolo-class blue_kfs --yolo-conf 0.5 --yolo-imgsz 640 --yolo-device 0`。直接 Python 入口必须显式提供权重。权重须为实例分割模型；retina mask 返回当前输入像素坐标，按类别过滤、置信度排序，并通过面积/有效深度过滤选择一个目标。YOLO 权重不能代替正确尺寸/身份的 FP Mesh。

首次 YOLO 模式自动构建独立 `foundationpose:kfs-yolo` 镜像，不修改原 SAM 镜像；不要求 SAM 源码/权重。`yolo_s` 单独记录耗时。Mask 模式只影响初始化和恢复，正常跟踪仍调用 FP。

### 分割调试窗口

启动参数 `--debug 0/1`（默认 0）控制全部分割窗口。SAM 路线显示 OpenCV 粗 Mask/bbox/提示点、SAM 原始 Mask 和 Depth 结果；YOLO 路线显示 YOLO 原始 Mask 和 Depth 结果。绿色覆盖目标、黄色标示轮廓；Depth 窗口蓝色标记新增像素、红色标记删除像素。窗口标题/画面标明最后一次分割的帧号；仅初始化和恢复时更新，正常 tracking 保留该图，不逐帧重新分割。不开启额外视频保存。`debug_display_s` 记录绘制耗时。`--debug 1` 与 `--no-display` 不兼容。离线原 FP 内部诊断级别改用 `--fp-debug`，与分割窗口开关分开。

示例：`bash src/start_kfs_fp.sh d435i --mask_mode sam --debug 1`，或 `bash src/start_kfs_fp.sh d435i --mask_mode yolo --debug 1`。
