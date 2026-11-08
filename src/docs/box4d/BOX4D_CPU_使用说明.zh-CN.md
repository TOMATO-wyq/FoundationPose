# 纯色方块 CPU 位姿识别

本文说明不带 YOLO 的几何识别、D435i 同步输入、录制与离线回放。当前两步启动命令在 [仓库 README 开头](../../../README.md) 和本文第 5 节。最新 TKK 小块模型接入见 [YOLO-seg 说明](BOX4D_YOLO_SEG.zh-CN.md)，C++ 数值内核见 [代码结构与 C++](BOX4D_代码结构与CPP.zh-CN.md)。绝对精度和稳定 60 FPS 尚未验收。

使用 `--auto-up` 在启动时测量当前竖直方向；相机角度调整后重启识别并重新测量。

[D435i 全图彩色目标配置](../../config/box4d_d435i_live.json) 启用当前 RGB-D 特征跟踪与后台重新检测，支持底部缺边、半面配对和共面底座处理。逐阶段诊断见 [调试说明](BOX4D_DEBUG.zh-CN.md)。

更新：已结合用户提供的两张堆叠示意图，新增三层与红蓝双色塔顶入口。见 [堆叠样例分析、颜色语义与验证](../../../方块堆叠情况/识别方案与验证.zh-CN.md) 和 [塔顶配置](../../config/box4d_tower.json)。该配置允许 3 个目标，20 cm 件按已确认的红蓝等半结构识别；默认 `box4d_cpu.json` 仍用于原纯色双块流程。

目标是约 20 cm、35 cm 的正方体，通常两个、同色、相邻有间隙或上下堆叠。相机可以低于顶面，算法使用可见侧面。只输出视觉位姿，不控制机器人。

## 1. 当前电脑直接看效果

在仓库根目录执行：

```bash
cd /home/tomato/6Dpose
python3 src/box4d/cli/demo.py outputs/box4d_demo_new --frames-per-scene 2
python3 src/box4d/cli/offline.py outputs/box4d_demo_new --output outputs/box4d_replay_new --save-debug --video --video-fps 2
```

所有输出目录要求是新目录，重复执行请更换名字。入口不会覆盖已有数据。

生成的合成演示覆盖两种尺寸的多个角度、两块同色、混合尺寸、无接缝整齐堆叠、旋转堆叠、空场景及错误尺寸。绿色表示可见面支持，橙色表示使用已知尺寸与堆叠模型推断。画面标注 SYNTHETIC，播放速度由 `--video-fps` 指定。

## 2. 环境与依赖

基础几何模式已在本机系统 Python 3.10.12、NumPy 1.21.5、OpenCV 4.5.4、SciPy 1.8.0 上运行，无须先安装新环境；不需要 Torch、FoundationPose、YOLO、SAM、Open3D 或 CUDA。启用 YOLO 配置时，还需要 Torch、Ultralytics 和相应权重，使用 [YOLO-seg 说明](BOX4D_YOLO_SEG.zh-CN.md) 中的环境。OpenCL 显式关闭。

新电脑可单独建立 CPU 环境，不要安装整份 FoundationPose requirements：

```bash
python3 -m venv --system-site-packages .venv-box4d
source .venv-box4d/bin/activate
python -m pip install -r src/requirements/box4d.txt
```

仅在新环境缺少依赖时安装；本机已有依赖可直接用 `python3`。实时显示需要带 GUI 的 OpenCV 和桌面会话；无显示环境不要加 `--display`。ROS 实时入口还需要与 Python ABI 匹配的 `rclpy`、`cv_bridge`、`realsense2_camera_msgs` 及相机驱动；优先复用 ROS 系统包。依赖约束将 NumPy 限于 2 以下以兼容当前 Humble 图像桥接。

验证离线环境：

```bash
PYTHONPATH=src python3 -m unittest discover -s src/tests -p 'test_box4d_*.py' -v
```

测试覆盖投影畸变、内参变换、稳健平面、PnP、两种尺寸、侧视、堆叠、颜色轴、静止加速度参考、ROI 截断、重叠假设、跨帧关联、底部遮挡、共面底座、错误尺寸、目标颜色门控、显示节奏、当前 RGB-D 特征跟踪、目标移走/缺深度时拒绝旧位姿、后台检测结果复核及相机俯仰变化后的重力参考；不替代实物精度验收。

## 3. 算法与配置

默认配置是 [config/box4d_cpu.json](../../config/box4d_cpu.json)。可复制一份再通过 `--config` 传入，未知字段会报错。

| 参数 | 含义 |
|---|---|
| `sizes_m` | 正方体实测边长，当前 `[0.20, 0.35]`，单位米 |
| `dimensions_nominal` | 当前为 `true`，表示尺寸尚未实测；实测并更新尺寸后改为 `false` |
| `up_camera` | RGB 光学坐标中的物理向上单位向量；未标定时为 `null` |
| `roi_xyxy` | 检测区域 `[x0,y0,x1,y1]`，右/下界不包含；默认全图 |
| `max_objects` | 最多输出两个有证据支持的目标，并不强制输出两个 |
| `method` | `depth`、`pnp` 或 `hybrid` |
| `plane_proposals` | 使用稀疏深度平面补充图像边缘候选，默认开启 |
| `allow_stack_inference` | 允许用完整堆叠轮廓、竖直参考和唯一尺寸解释推断两块，默认开启 |
| `refine_4d` | 有竖直参考时联合细化中心 XYZ 与 yaw |
| `opencv_threads` | OpenCV 线程数，当前为 1，方便复现耗时 |
| `analyze_colors` / `two_color_sizes_m` | 开启可见面颜色分析，以及已知等分红蓝结构的规格；默认关闭/空，tower 配置开启并设为 `[0.20]` |
| `red_hue_ranges` / `blue_hue_range` | OpenCV HSV 色相区间 0–179；红色默认 `[0,12]`、`[168,179]`，蓝色 `[95,140]` |
| `color_min_saturation` / `color_min_value` | 低饱和或过暗像素按未知处理，默认 70 / 40 |
| `color_line_fallback` | 常规检测没有有效位姿时尝试 RGB 边线候选，默认关闭；更多候选会增加耗时 |
| `color_half_proposals` | 开启颜色分析及双色规格时，尝试配对两个已测得的异色半面，默认开启 |
| `color_plane_priority` | 颜色分析开启时优先搜索红蓝区域深度面，默认开启；保留通用图像候选 |
| `upright_outline_proposals` / `allow_bottom_occlusion` | 有 UP 时利用可见顶边/侧边与当前深度证据恢复底部缺边，结果明确标为模型推断；不补发旧 valid 位姿 |
| `require_target_colors` | 默认关闭；全图彩色配置开启，颜色证据不足的同尺寸几何候选记录到 `appearance_rejections`，不强制判断有歧义的颜色轴 |
| `temporal_recovery` | 默认关闭，全图彩色配置开启；用当前图像特征与当前深度延续几何位姿，失败时不补发旧有效位姿 |
| `async_detection` | 实时入口启用连续跟踪时，将重新检测放到后台；离线入口仍同步处理，保证回放可重复 |
| `redetect_interval_s` | 重新搜索的周期，默认 0.15 秒；失去跟踪时提前搜索，后台只留最新待处理帧 |
| `motion_prediction` | 默认 `true`；用位置/速度匀速 KF 预测短时漏检位置，与当前帧观测分开输出 |
| `prediction_max_gap_s` | 从最后一次有效观测起最多预测的时间，默认 0.3 秒 |
| `prediction_max_std_m` | 预测位置不确定度上限（标准差），默认 0.1 米；超过后停止预测输出 |
| `prediction_accel_std_m_s2` | 匀速模型中未建模加速度的标准差，默认 1.0 m/s²；越大，不确定度增长越快 |
| `prediction_measurement_std_m` | 视觉位置测量噪声的标准差，默认 0.01 米 |
| `prediction_initial_velocity_std_m_s` | 新轨迹初始速度不确定度的标准差，默认 0.5 m/s |

目前只实现正方体模型。若三边明显不同，不能只把某一条边填入 `sizes_m`；需要扩展长方体模型及其 180° 对称性。12% 尺寸门限用于候选校验，不代表允许 12% 的定位误差。

前端结合图像边缘、深度断层和稀疏平面找可见四边形。最小外接矩形只用于平面坐标，避免把透视图像中的侧面强行当成矩形。尺寸和深度共同检查候选，两个可见侧面合并为同一物体。

- `depth`：局部深度平面与角点射线求交，再从面中心换算到物体中心。
- `pnp`：单面 IPPE_SQUARE 枚举平面解，再用当前深度、法向和竖直约束校验；歧义时拒绝。
- `hybrid`：汇总深度/PnP 候选，有条件时加入多面 PnP，并进行 4 参数细化。合成数据中更稳妥的当前入口，最终主路径仍需真实数据对比。

`pnp` 路径也需要深度，因为实例检测、尺度检验和平面双解校验都使用 RGB-D。它不是纯 RGB 单目方案。

### 短时漏检预测

KF 的状态为相机光学坐标系中的 `[X,Y,Z,Vx,Vy,Vz]`，速度表示目标相对相机的运动。它按**源图像时间戳**计算时间间隔，不按显示刷新次数预测；至少取得两次有效观测后，才在漏检时输出预测位置。重新得到观测时，用当前视觉位置校正状态，原始观测字段保持不变。

预测位置用于目标关联、光流初值、迭代 PnP／联合求解初值，以及位移残差门控；当前 RGB-D 的深度、重投影和边界检查仍需通过。漏检预测单独输出到 `predicted_objects`。显示中以**紫色虚线三维框、`PRED` 和距最后观测的秒数**标识；普通显示和 `--display-smoothed` 都会显示。框的平移来自 KF，旋转保持最后一次实测值。它不预测转向，也没有融合 IMU 或里程计，因此突然加速、急转弯或长时间遮挡仍可能跟丢。超过预测时间或不确定度上限后，预测框消失。已有启动命令无需增加参数，重启识别进程即可使用默认配置；输出预测项需要开启 `tracking`，光流／PnP 预测初值还需要开启 `temporal_recovery`。

`objects` 仍只包含当前帧通过校验的真实 RGB-D 观测，ROS `/box4d/detections` 也仍只发布这些有效观测。预测项带 `source="kf_prediction"`、`status="predicted"`、`valid=false`，不会冒充有效检测；当前控制器在没有有效观测时仍执行漏检停车。显示中的 `observed=0 predicted=1` 表示只有预测框，并未重新检测到目标。

进行 A/B 对照时，复制同一份配置，把其中 `motion_prediction` 改为 `false`，用 `--config` 指向该副本，并重放同一段录制；其余检测参数保持一致。原有 `center_camera_smoothed_m` 是显示平滑结果，和这里用于补短时漏检的预测分开。

## 4. 坐标、yaw 与竖直参考

`center_camera_m` 是方块几何中心，采用 RGB CameraInfo 的光学坐标：通常 X 向右、Y 向下、Z 向前。`T_camera_from_box` 将以方块中心为原点的模型点变换到相机坐标。立方体面中心与物体中心相差半个边长，程序已做转换。

物理 yaw 定义在竖直对齐参考系 U：原点仍在相机光心，Z 为物理向上，X 为相机前向在水平面的投影，Y 按右手系构建。输出 `U_from_camera`、`position_u_m` 和 `yaw_u_rad`。

没有竖直参考时仍可估计相机坐标下的几何位姿，但 `yaw_u_rad`、`position_u_m` 是 `null`，状态为 `pose_unreferenced`。纯色正方体的朝向按 90° 等价，无法识别其绝对 0–360° 面编号。

竖直参考有三种已实现的输入方式：

1. 明确标定后用 `--up-camera UX UY UZ`，或写入配置。只有光学 Y 轴确实竖直向下时，才可使用 `0 -1 0`。合成数据已知此方向，并保存在数据 manifest；真实数据不会自动沿用它。
2. 实时入口 `--up-plane-roi X0 Y0 X1 Y1`：人工选择**相机以下、已知水平的地板或台面**，程序从第一帧拟合其朝向相机的法向。区域需完整处于该平面上，不能包含箱子边界或把侧面选成水平面。只验证平面性，程序无法自动证明它物理水平。
3. D435i 实时入口 `--auto-up`：启动时采约 2 秒静止加速度，按实际 ROS TF 转到 RGB 坐标，并保存本次 `up_reference.json`；需要相机驱动开启 `enable_accel:=true`，当前启动脚本已开启。相机安装改变后使用新测量。只在启动时测量，运行中调整相机后重启识别。

第二种方式把得到的向量写入 `run_config.json`，之后同一固定安装可复用。相机移动后要重做。另提供 `estimate_box4d_up.py`，按实际 TF 把静止加速度转到 RGB 坐标，结果带有初步估计的质量标记；当前设备提示 IMU 标定缺失，不能作为独立角度基准。相机到机械臂基座的外参未标定，相机坐标数值不能直接用作基座坐标。

## 5. 接 D435i 后运行

先用 ROS 相机驱动提供数据，再启动识别程序。单独在 RealSense Viewer 中打开相机不够；本机用相机脚本配置同步 RGB-D、对齐深度、内参、加速度和陀螺仪流。启动脚本前关闭占用相机的 Viewer 或其他驱动。

2026-10-22 已确认当前连接为 USB 3 / 5000M，下面使用 `speed` 请求 640×480@60。终端 A 使用 ROS 域 86 启动驱动并保持运行；已有相同配置的驱动则直接复用：

```bash
cd /home/tomato/6Dpose
bash src/start_box4d_camera.sh --domain 86 speed
```

speed 请求颜色、深度均为 640×480@60，开启同步、深度对齐和 RGBD 单条消息，同时开启 `enable_accel:=true`、`enable_gyro:=true`。`precision` 请求 1280×720@30，`usb2` 请求已实测的 640×480@30。`rgb` 是仅彩色模式，深度和两种 IMU 流均关闭。具体能否启用取决于设备、USB 和驱动版本；已有正确 RGBD 驱动时直接复用，不再启动第二个。

终端 B，启动当前彩色方块配置，并重新测量相机方向：

```bash
cd /home/tomato/6Dpose
bash src/box4d_ros.sh --domain 86 live \
  --config src/config/box4d_d435i_live.json --auto-up \
  --output "outputs/box4d_live_$(date +%Y%m%d_%H%M%S)" \
  --duration 600 --display --display-smoothed --display-hz 30
```

封装脚本加载 ROS 与本机 overlay。默认 topic 为 `/camera/camera/rgbd`，可用 `--topic` 更改。启动测方向时相机保持静止。直接使用 Python 入口时需先自行加载 ROS 环境；未传入竖直参考时 yaw 为 null。

相机和识别的 ROS domain 必须一致。相机脚本支持 `--domain ID`，不传时沿用环境变量 `ROS_DOMAIN_ID`，
未设置则使用 0。已有相机驱动在 domain 0 时，终端 B 改为 `--domain 0` 即可复用。
RGB-D 超时信息会打印话题、domain、发布者数量、有效／拒绝／重复帧数；`Publishers=0` 优先检查 domain、
`--topic` 和 `enable_rgbd`，有发布者但收不到数据再检查相机流与 QoS。延长超时不能解决 domain 不匹配。

如果 `--auto-up` 超时，程序会报告 IMU 话题、ROS 域、发布者数量、收到的消息数、重复/倒退消息数及有效样本时间跨度。`enable_accel=true` 且话题存在，也可能没有实际数据；不能仅凭参数值判断 IMU 正常。没有发布者时检查域、`--imu-topic` 和 `enable_accel`。有发布者但收到 0 条消息时检查驱动日志；本机曾出现 `Motion Module failure`，关闭再开启 `enable_accel` 也未恢复，调用设备复位后恢复约 100 Hz 加速度数据。

仅在上述 Motion Module 故障时，复位现有驱动中的相机，再重跑识别（复位会短暂停止图像；不要再启动第二个驱动）：

```bash
source /opt/ros/humble/setup.bash
source "$HOME/realsense_ws/install/setup.bash"
ROS_DOMAIN_ID=86 ros2 service call /camera/camera/hw_reset std_srvs/srv/Empty '{}'
```

相机节点名或域不同则相应修改服务路径或域。若复位仍不恢复，重新插拔 USB 后检查驱动日志和加速度话题。校准仍要求至少 50 个独立新样本、足够时间跨度和静止质量检查，不会沿用旧 UP 或跳过校验。

已有相机脚本运行时，只启动终端 B。切换相机档位需先在终端 A 按 `Ctrl+C` 再启动；识别退出后相机驱动可以继续运行。

实时配置的 `max_reprojection_px=3` 以 `reprojection_reference_width_px=640` 为参考：1280 宽输入的实际门限为 6 像素，640 宽为 3 像素。检测、4D 优化和当前帧跟踪使用相同门限，实际值写入目标的 `metrics.reprojection_limit_px`。自定义配置不填写参考宽度时，仍按输入原始像素解释该门限。尺寸、深度残差和竖直约束保持原来的物理单位。

`detector_max_age_s=0.75` 单独限制后台检测结果的年龄；`track_max_gap_s=0.3` 限制普通轨迹中断。可信 KF 预测可将轨迹恢复窗口延长至 `prediction_max_gap_s`（默认也是 0.3 秒），但不能绕过不确定度上限。1280×720 下检测耗时可能超过 300 毫秒，不能用轨迹间隔直接丢弃所有检测结果。延迟检测仍需通过当前图像的双向特征跟踪、当前深度和边界校验后才输出；目标移走或当前深度缺失时不会补发旧的有效观测。启用 KF 时，短时间内可能仍显示明确标记的紫色虚线预测框。结果记录 `expired_detections` 和 `detector_seed_failures`，便于区分过期与当前帧校验失败。调整相机后应重启识别重新执行 `--auto-up`。

显示窗口按 Q/Esc 结束，R 重置跨帧关联。默认显示最多 30 Hz，识别和显示分别运行，显示刷新数不计作新位姿观测；该显示上限不限制相机采集与识别处理频率，可将 `--display-hz` 改为 60 观察更高刷新率。相机档位为请求值，稳定采集及处理 60 FPS 尚需实测。无界面验收去掉 `--display`。

实时输入检查：

- RGB 和对齐深度必须同一 optical frame、分辨率一致；读取完整 K/D，而非仅焦距。
- 默认 RGB/深度时间戳要求相同；`--max-sync-ms` 可显式放宽，但应记录放宽原因。
- `16UC1` 默认每单位 0.001 m。若设备 depth units 改过，实时和录制均传 `--depth-scale`；`32FC1` 已按米读取。
- 默认按 CameraInfo 的原始畸变模型处理。**确认图像已经校正**时才用 `--rectified-input`，此时使用 P、零剩余畸变。
- 不支持的畸变模型、标定中途变化、重复/倒退时间戳会拒绝；不能靠忽略错误继续产生有效位姿。

## 6. 录制与离线对照

驱动运行、ROS 环境已加载后，先录制约 5 秒：

```bash
python3 src/box4d/cli/capture.py outputs/box4d_capture_new --duration 5 --max-frames 300 --display
python3 src/box4d/cli/offline.py outputs/box4d_capture_new --output outputs/box4d_capture_replay_new --save-debug --video
```

`--display` 显示可缩放的录制预览：左侧为 RGB，右侧为同一帧对齐深度，顶部显示时间、已保存/已提交帧数和写盘丢帧。深度预览使用固定 0～3 m 色标，黑色表示缺失或无效深度；色标只影响显示，原始深度不裁剪、不填洞。按 Q、Esc 或关闭预览窗口会正常结束并排空写盘队列。默认 `--display-hz 15` 只限制预览刷新，不设定录制帧率；实时吞吐仍受相机、显示和磁盘影响。断流期间窗口退出可能等待输入超时。

### 录制 60 秒 RGB-D + IMU，并显示预览

需要检查相机转动时的识别，使用 `--record-imu` 同时保存加速度和陀螺仪。旧相机驱动若只有加速度，修改脚本不会自动改变正在运行的进程：先回到旧驱动所在终端按 **Ctrl+C**，再用当前脚本重启。脚本不会自动杀死旧驱动，也不要同时启动两个相机进程。

终端 A，开启带加速度与陀螺仪的 RGB-D 驱动：

```bash
cd /home/tomato/6Dpose
bash src/start_box4d_camera.sh --domain 0 speed
```

终端 B，录制 60 秒，将 RGB-D 最大帧数设为 6000，避免默认 300 帧提前结束：

```bash
cd /home/tomato/6Dpose
bash src/box4d_ros.sh --domain 0 capture \
  "outputs/box4d_capture_imu_$(date +%Y%m%d_%H%M%S)" \
  --duration 60 --max-frames 6000 --record-imu \
  --display --display-hz 15
```

录制开始的前 **3 秒保持相机静止**，给离线重力和陀螺仪初始状态估计提供数据，再做需要测试的相机俯仰或方块运动。尽量让方块完整留在画面内，避免把物体出画与姿态求解问题混在一起。IMU 话题可通过 `--accel-topic`、`--gyro-topic` 指定；相机、录制端的 ROS domain 必须一致。

RGB-D 仍按原格式保存，额外增加：

| 输出 | 内容 |
|---|---|
| `imu_bag/` | 加速度和陀螺仪的 ROS bag 数据，保留 IMU 自身的时间戳 |
| `imu_reference.json` | IMU 录制参考信息，供核对话题、坐标关系和离线读取 |
| `imu_record.log` | IMU 录制进程日志，检查启动、异常及结束情况 |
| `imu_summary.json` | 原生消息条数、源时间戳范围、数据质量计数和正常收尾状态 |
| `imu_qos.yaml` | 本次 IMU 与静态 TF 订阅使用的 QoS 配置 |

启动会检查加速度和陀螺仪确实收到新数据，以及它们到 RGB 光学坐标系的外参；录制期间任一路 IMU 超过 1 秒没有有效新数据会报错。按 Q、Esc、关闭预览或达到时长时，会先正常结束 IMU bag、检查实际记录内容，再完成 RGB-D 摘要。查看完整性时同时检查 `capture_summary.json` 与 `imu_summary.json`，不能只凭 RGB-D 帧数判断 IMU 录制成功。

RGB-D 与 IMU 的采样频率不同，离线需按源时间戳关联，不按行号或帧号强行一一配对。60 秒、约 60 FPS 的未压缩 RGB-D 通常需要约 7.7 GB 空间，实际帧数以录制结果为准；`6000` 只是上限，不会强制相机按 100 FPS 采集。

**保存了 IMU 数据不等于实时识别已经融合动态 IMU。** 当前 `--auto-up` 仍只在识别启动时做静止标定；这里新增的是可用于后续离线检查和实现动态参考的数据。相机转动后不能直接沿用之前固定安装的 UP，也不要把另一时刻的标定向量写成这段运动录制全程有效的 UP。

仅在整段录制的相机安装方向保持固定、且已标定向上向量对本段有效时，录制时加 `--up-camera UX UY UZ` 记入数据，或离线运行时补传。运动录制或不能确定时留空。真实数据没有自动真值，需要独立位置/角度参考才能报告绝对误差。

数据包含 `manifest.json`（标定、单位、坐标、up）、`frames.jsonl`（整数纳秒时间戳索引）、`frames/*.npz`（原始 RGB 与浮点米制深度）、`capture_summary.json`。每帧无损、未压缩，640×480 的 RGB+float32 深度约 2.15 MB，300 帧约 645 MB，不适合无上限录制。写盘队列最多 4 帧，慢盘会计数丢帧；`complete`、`dropped_writer_frames` 及收到的源帧数应一起检查。

异常中断或写盘失败的数据默认拒绝重放。诊断时可显式 `--allow-partial`，并保留不完整说明。相机实时入口采用最新帧，不积压历史图像；位姿 JSON 写入为同步轻量写入，慢盘仍会影响真实端到端吞吐，必须包含在现场验收中。

比较相同输入与前端上的三条路径：

```bash
python3 src/box4d/cli/benchmark.py outputs/box4d_capture_new --output outputs/box4d_benchmark_new --repeat 10
```

该基准预加载帧、预热后重复计算。`depth`/`pnp` 不做联合细化，`hybrid` 默认细化；可加 `--no-refine` 对照。不含采集、图像桥接、关联、文件写入和显示，因此只报告离线计算速度。反复处理同一段数据不增加独立精度样本数。全场景均值可能受空场景影响，同时查看 `by_scene` 的双目标和堆叠耗时。

## 7. 结果怎么读

每行 `poses.jsonl` 对应一帧，不要求每帧恰好有两个目标。

| 字段 | 解释 |
|---|---|
| `stamp_ns` | 源图像时间戳，整数纳秒 |
| `received_monotonic_ns` | ROS 回调收到帧时的主机单调时钟；离线没有该时间时为 null |
| `frame_id` / `calibration_id` | 相机光学坐标与标定编号 |
| `objects` | 当前帧通过校验的目标，缺失时是空数组 |
| `predicted_objects` | 当前帧未观测到、仍在时间及不确定度范围内的 KF 预测项；与 `objects` 分开，不作为有效检测发布 |
| `track_id` | 位置、规格与周期角度关联得到的短时 ID，同色完全遮挡后不保证身份 |
| `center_camera_m` / `T_camera_from_box` | 当前帧原始中心与模型变换，未做时间平滑 |
| `yaw_u_rad` / `yaw_period_rad` | 原始 yaw 与 π/2 周期；无竖直参考时 yaw 为 null |
| `center_camera_smoothed_m` / `yaw_u_smoothed_rad` | 另行保存的滤波显示结果，有滞后，不能用其代替原始重复性 |
| `evidence` | `observed_faces`、`model_inferred` 或 `tracked_current_rgbd`；跟踪结果使用当前特征及当前深度，投影四角不是独立检测角点 |
| `dimensions_nominal` | 尺寸是否仍为名义值 |
| `faces` / `alternatives` / `metrics` | 四角、来源、候选解、重投影/深度残差与细化情况 |
| `quality_score` | 残差构成的启发式诊断分数，不是识别正确概率 |
| `rejected_faces` | 被拒绝候选的原因，如缺深度、尺寸不符、边界不足或姿态不一致 |
| `lost_track_ids` | 此帧没有观测到的既有 ID；不附带旧坐标冒充新测量 |
| `timing` | 检测、求解、细化与总核心计算耗时，单位 ms |
| `temporal` | 当前特征跟踪数、失败原因、后台检测帧龄和耗时等；异步模式前台 `processing_ms` 不包含后台检测，不能用其倒数表示整体 FPS |
| `appearance` | 开启颜色配置时的可见颜色、蓝→红轴、模型分界面与朝上/侧放状态；含歧义及推断依据 |

`predicted_objects` 中的 `last_observed_stamp_ns` 为末次有效观测时间，`prediction_age_s` 为截至当前源帧的漏检时长，`position_covariance_m2` 为 3×3 位置协方差（单位 m²）。预测项没有复用旧 `faces` 或旧测量残差来表示当前证据。

帧状态为 `observed`、`partial_or_rejected` 或 `no_target`。部分可见但无法可靠求完整中心的目标记录在候选拒绝信息中，当前没有独立输出“只有方向”的 partial 目标接口。

实时 `summary.json` 分别记录源时间戳估计的接收帧率、实际处理不同帧的速率、收到但没处理的帧数、计算耗时与接收到结果的延迟，以及进程 CPU 和峰值 RSS。没有核对相机/主机时钟对应，不能把主机接收到结果的延迟称为曝光到输出的延迟。进程 CPU 百分比按单核 100% 计算，可能超过 100%。

## 8. 已知限制与下一步验收

- 严重遮挡、面边界不可见、物体超出画面、深度大面积缺失时，优先拒绝完整位姿。当前旋转堆叠下层是已知漏检场景。
- 无接缝堆叠需要完整外轮廓、已知规格、竖直参考和唯一拆分解释；橙色推断不能视为看到了真实接缝。不希望使用先验时把 `allow_stack_inference` 设为 `false`。
- 仅支持近似直立正方体。弯曲边缘、软包装、反光/透明表面、与背景共面且无边界等情况尚未实测。两个同色目标贴紧且边界消失时不保证分离。
- PnP 对角点误差和平面双解敏感，组合方案也不能恢复被完全遮挡的信息。当前合成结果不能证明 PnP 在真实 D435i 上更优或更差。
- 实测尺寸误差、相机内参/深度尺度误差与竖直方向偏差都会传到中心和 yaw。先测量这些量，再收紧残差门限。
- 初版 basic 合成场景平均离线速度超过 60 FPS，但组合路径 P95 仍超过 16.7 ms。实拍堆叠的耗时和缺测见独立实拍记录；不得将合成速度当作真实稳定 60 FPS。

接相机后的顺序：实测三边与距离 → 检查同步、深度对齐和档位 → 标定 up → 保存单块、同色双块、堆叠的原始片段 → 同输入比较 A/B → 用独立测量参考评估位置/yaw/覆盖率 → 约两个目标连续运行至少 60 秒测源帧率、有效观测与延迟。建议目标仍为位置 P95 ≤15 mm、yaw P95 ≤2°、核心计算 P95 ≤16.7 ms；这些是待验证目标。

部署其他电脑时先复制本实现、配置及同一段离线数据，运行测试和基准，再接相机；不必复制 FoundationPose 的 CUDA 环境。不同 ROS/CPU 平台需重新测性能。机器人基座外参和抓取规划在本轮范围之外。
