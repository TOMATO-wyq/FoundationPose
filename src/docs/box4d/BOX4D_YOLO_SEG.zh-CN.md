# YOLO-seg 小块前端与原有方块位姿链路

新配置：`src/config/box4d_d435i_yolo_live.json`。
模型使用这次 400 轮训练的 `runs/tkk_seg_bg_from276_20261029_125307_400e/weights/best.pt`，任务为实例分割，类别为 `tkk`。

CPU C++ 计算阶段和代码结构见 [代码结构与 C++](BOX4D_代码结构与CPP.zh-CN.md)。编译后原命令默认自动使用 C++ 颜色/深度处理、平面与边线拟合、面内采样和优化计算；加 `--geometry-backend cpp` 可以强制确认启用。YOLO 继续在 CPU 上推断。

## 识别流程

1. YOLO 在检测帧上输出多个小块实例，每个实例保留独立 mask 和外扩 ROI。
2. 20 cm 小块：先使用 02 窗口同一套 HSV 红蓝掩膜测量颜色占比。默认要求实例内红色与蓝色的总覆盖率至少 60%，不要求各占一半，也不强制两种颜色同时可见。
3. 颜色通过后，在各实例对应的 ROI 内分别拟合实测平面，并用实际 RGB 边缘上的像素段做 Huber 直线拟合、求交点。完整面进入原来的 PnP/深度几何解算与一致性检查。
4. 如果底部被遮挡，要求可见顶边与两条侧边、已知宽度、正确 UP，以及原始深度中真实的前景遮挡证据，再恢复隐藏底角。这种恢复使用深度几何，隐藏角点不充当 PnP 的实测输入。
5. 35 cm 大块：继续原来的全画面几何检测与验收，不要求 YOLO 找到它。
6. 合并结果并检查空间冲突，再交给原来的当前帧 RGB-D 跟踪。

所有几何计算保持原图尺寸、像素坐标和相机内参。分割轮廓不是 PnP 的四个角点；一个 mask 可以包含两个可见面，由原有逻辑分别找面。没有深度的地方不会由模型补出深度。

小块首次获取需要有分割实例；YOLO 暂时漏分时，已有轨迹可以继续使用当前图像和深度通过验收。当前帧验收失败便不输出旧位姿。

小方块转动时，原来的追踪面逐渐变窄，而另一个面在当前帧已经通过验收且面积明显更大，可受控切换追踪面。默认只对 20 cm 启用；同一个面不因重复检测而反复重置，35 cm 继续原行为。参数与本次日志证据见 [窄侧面检查与修正](../../../outputs/box4d_small_face_audit_20261030/窄侧面检查与修正.zh-CN.md)。

连续跟踪每帧使用双向 LK 特征点和单应性筛点，再运行 IPPE/ITERATIVE PnP；传播的四边形来自特征运动，不是每帧重新拟合四条边线。当前深度参与平移修正、候选评分和验收。普通候选全部失败时仍执行图像与深度联合优化；默认还在近正对（`view_cos > 0.95`）、且观测代价最低的普通候选方向步长超过 5° 时加入联合候选，随后一起验收和比较。

新增短时方向窗口保存每条轨迹的角速度趋势。图像和深度证据接近时，优先选择符合趋势的姿态；真实观测优势明显、只有一个可行候选或所有候选都偏离预测时，允许立即转向。它参与候选评分，不对发布角度做时间低通，也不锁定 pitch/roll。后台检测匹配上已有轨迹时，通常仍只确认身份，受控换面才接管新检测几何。`refine_4d` 是检测优化选项，不能理解为后续跟踪锁定了 pitch/roll。`--display-smoothed` 当前平滑位置和 yaw，不平滑 pitch/roll。

特征补点也不再无条件用旧姿态自证，或无条件让实测法线接管。程序先在同一批当前图像、深度观测上比较“当前姿态”和“实测平面纠偏姿态”，加上方向软代价和纠偏所需的证据优势；有明确优势才纠偏。接受后同时更新发布位姿、对象和特征模型。可疑补点被拒绝时，保留本帧已经通过验收的跟踪结果，后续再尝试补点，不发布更早帧的旧位姿。

新日志无需打开 debug 窗口。`detector_diagnostics.objects` 和 `detection_checks` 保留源检测帧的实际位姿、面候选、求解与失败记录；同步检测同样保存源快照。当前对象 `metrics.measured_plane_normal_camera`、`measured_plane_normal_step_deg` 与 `pose_normal_step_deg` 分开记录测量面和输出姿态的变化。`temporal.tracking_checks[*].pose_trials` 保存主跟踪每路 PnP 的方向、残差、代价分项、拒绝原因和 `selected` 标记；`detector_seed_checks` 保留异步检测从源帧推进到当前帧的证据，`reseed_checks` 保留补点比较。对象的 `metrics.tracking_check_group` / `tracking_check_index` 指向对应记录。角度步长与该状态的上次测量比较，须结合时间差看待，不能一律当作相邻输入变化。

`detector_confirmed` 只表示当帧消费的新检测结果完成确认，不能解读为检测与当前 RGB-D 同一时间；源帧时刻见 `detector_diagnostics.source_frame`。`detector_match_action` 区分 `confirm`（确认身份）和 `face_switch`（接管新面）。`detector_seed_stamp_ns` 是当帧接收新种子的事件，下一帧清除；`origin_detection_stamp_ns` 是轨迹持续保留的检测来源。跨帧引用 trial 要同时携带运行目录、当前时间戳、检查组/序号和 trial 序号。

早期正对抖动证据和未采用的两种替代求解实验见 [排查结论](../../../outputs/box4d_drift_audit_20261030/排查结论.zh-CN.md)。当前版本在这些实验后新增了方向窗口、按需联合候选和受控补点纠偏，仍不增加 pitch 时间滤波。参数和诊断作用域见 [跟踪方向与诊断](BOX4D_跟踪方向与诊断.zh-CN.md)。23:10:04 那次运行没有原始 RGB-D，无法直接重放其静止翻转，不能据其他片段就承诺实际抖动全部消失。

## 运行

相机已经启动时直接运行第二段。相机和识别程序须使用相同的 ROS domain，下面统一用 0。启动时保持相机静止，以便 `--auto-up` 重新测量重力方向。

横放、竖放和倒置会按当前 UP 自动为 YOLO 选择 0/90/180/270° 输入补偿，mask 随后映回原始图像；不改变 RGB-D 几何坐标或内参。先把相机摆到测试方向，再启动识别。`--auto-up` 当前只在启动时标定，运行中转动相机后需要停稳并重启识别重新标定。

```bash
cd /home/tomato/6Dpose
bash src/start_box4d_camera.sh --domain 0
```

另一个终端：

```bash
cd /home/tomato/6Dpose
CUDA_VISIBLE_DEVICES=-1 \
BOX4D_PYTHON=/home/tomato/code/box_seg_annotation/env/bin/python \
bash src/box4d_ros.sh --domain 0 live \
  --config src/config/box4d_d435i_yolo_live.json --auto-up \
  --output "outputs/box4d_yolo_$(date +%Y%m%d_%H%M%S)" \
  --duration 600 --display --display-smoothed \
  --debug --debug-hz 4 --debug-layout windows
```

该 Python 环境已有 Torch 和 Ultralytics，并已验证可以加载 ROS 与 RealSense 消息。新配置强制模型使用 CPU，运行命令也隐藏 GPU；无需 CUDA 推理。环境中 Torch 安装包本身支持 CUDA，不影响本次全部计算在 CPU 上执行。大块及小块后续几何仍按配置的 `plane_backend` 运行，新配置沿用当前调试配置的 `numpy` 后端。要使用 PCL，按原方法构建后加 `--plane-backend pcl`。

## 调整与 debug

02 窗口增加绿色分割轮廓、黄色搜索 ROI、实例置信度，以及红色/蓝色/总覆盖率、颜色 PASS/REJECT 和位姿数量。比例在实例 mask 内计算，防止外扩 ROI 中的大块蓝色混入身份判断；日志另外保留整个搜索 ROI 的颜色占比。

按 **L** 在大块全图检测和各小块 ROI 的实际几何结果之间切换。03 窗口的青色线段为真实 RGB 边线上的稳健拟合结果；05 看分开的实测平面；06 的 **O** 是观测边线交点，**I** 是推算的隐藏底角；07 看几何验收。08 显示当前帧跟踪，不把旧分割图画成当前图。

01～07 显示后台检测自己的源图，标题注明帧号和年龄。新小块链路保留原始深度用于验收：mask 只限定平面拟合的种子区域，不再把遮挡物的深度清零。否则会删掉“大块挡住小块底边”的关键证据。保存的 `current_rgbd.npz` 与检测源图的深度都保留原始输入。

09 调参窗口用 **[ / ]** 翻页，第 9～11 页是以下小块参数，第 12 页是“方向连续性与补点纠偏”；**S** 保存所有页面参数、新配置和调试快照。第 12 页的默认值与调整方向见 [参数说明](BOX4D_跟踪方向与诊断.zh-CN.md#参数与第-12-页滑杆)。

| 中文滑杆 | 配置字段 | 默认值 | 调整含义 |
|---|---|---:|---|
| YOLO 置信度 % | `yolo_conf` | 0.40 | 提高会减少低分误检，也可能增加漏分；这里只控制实例是否进入几何检测 |
| YOLO 实例数量上限 | `yolo_max_instances` | 6 | 每次最多处理几个实例，调大增加后台计算量 |
| ROI 外扩像素 | `yolo_roi_padding_px` | 20 | 搜索外扩像素下限，与相对外扩取较大值，给边线和隐藏底部留上下文 |
| 面落在分割内占比 % | `yolo_min_face_mask_fraction` | 0.65 | 原有几何产生的可见面，至少多少比例落在实例 mask 内；提高更严格 |
| ROI 红蓝总占比 % | `yolo_color_min_coverage_fraction` | 0.60 | 实例内红色与蓝色之和的最低比例，不依赖深度或已求得的位姿 |
| ROI 单色最低占比 % | `yolo_color_min_each_fraction` | 0.0 | 两种颜色各自最低比例；默认 0，允许只看见蓝面。提高会拒绝另一颜色被遮住或位于背面的角度 |
| ROI 相对外扩 % | `yolo_roi_padding_fraction` | 0.35 | 按实例较长边的 35% 外扩；避免固定 20 px 截掉推算出的底部与遮挡证据 |
| 平面取点外扩 px | `yolo_geometry_margin_px` | 4 | 给 mask 少量边缘误差留余量，仅影响拟合取点，不扩大整幅搜索 |
| 边线搜索半径 px | `yolo_line_band_px` | 4 | 在平面轮廓附近多宽的带内收集真实 RGB 边缘点 |
| 每条边最少像素点 | `yolo_line_min_points` | 8 | 一条直线至少要有多少实际边缘点，缺边不会直接补成四个观测角 |
| 边线可见跨度 % | `yolo_line_min_span_fraction` | 0.30 | 支持点覆盖预期边长的最低比例，防止靠一个短噪声段外推整条边 |
| 遮挡面可见高度上限 % | `yolo_partial_max_visible_fraction` | 0.98 | 允许浅遮挡进入专门的恢复逻辑；仍须同时有顶边、侧边和前景证据 |
| 遮挡前景最低占比 % | `yolo_partial_min_foreground_fraction` | 0.04 | 推算完整面内至少多少网格点被更近的物体遮住。深度空洞和远处背景不算前景 |

其他字段在配置文件中修改，重启生效：

| 配置字段 | 默认值 | 含义 |
|---|---|---|
| `yolo_weights` | 新模型 `best.pt` 的绝对路径 | 可填权重目录，自动取 `best.pt`；设为 `null` 恢复全部原有几何链路 |
| `yolo_class` | `tkk` | 分割类别名，也可以填类别编号字符串 |
| `yolo_device` | `"cpu"` | 固定 CPU 推理 |
| `yolo_cpu_threads` | 2 | YOLO 的 Torch CPU 运算线程数，避免推理占满 CPU 影响当前帧跟踪 |
| `yolo_upright_inference` | `true` | 根据当前标定 UP 为网络输入补偿整 90° 旋转，mask 映回原始像素；无 UP 或近垂直俯视时不猜测旋转方向 |
| `yolo_imgsz` | 640 | 网络推理输入尺寸，mask 会还原为原图尺寸，几何坐标不缩放 |
| `yolo_sizes_m` | `[0.2]` | 使用分割分支的实体尺寸；其他尺寸仍走原有全图几何 |
| `yolo_roi_color_geometry` | `true` | 启用本轮颜色前置、ROI 平面与边线链路；设为 `false` 可对照上一版硬 mask 小块链路 |

## 延迟与积压

实时配置启用 `temporal_recovery=true`、`async_detection=true`。YOLO 和各 ROI 几何在同一个后台检测线程执行，YOLO 默认最多使用 2 个 CPU 运算线程；主线程继续处理当前帧光流和深度。后台最多保留 **1 个待处理帧**，忙时覆盖为最新输入，不按相机帧率积累任务。

启动脚本还默认限制 OpenBLAS 为 1 线程、OMP/MKL 为 2 线程，避免数学库各自占满 CPU；显式设置同名环境变量时会保留用户的值。

调参只短暂锁住配置复制，不等待推理。旧检测结果返回后还要在当前图像和深度上重新验证；超过 `detector_max_age_s` 的结果丢弃。首次模型初始化和预热较慢，此时主线程仍继续处理，初始化结果可能因过期而被丢弃。

日志 `poses.jsonl` 提供：

| 字段 | 检查用途 |
|---|---|
| `timing.processing_ms` | 当前帧处理耗时 |
| `temporal.background_detection_ms` | 后台检测完整耗时，包括 YOLO 和各 ROI |
| `temporal.detector_frame_age_ms` | 后台结果对应的源图距当前帧多久 |
| `temporal.detector_queue_depth` | 待处理帧数，最大为 1 |
| `temporal.detector_pending_replaced` | 忙时被更新图像替换的待处理任务累计数，不是队列长度 |
| `temporal.expired_detections` | 过期而丢弃的候选数量 |
| `yolo.inference_ms` | 模型推理、mask 转换与 ROI 提取耗时 |
| `yolo.input_rotation_ccw_deg` | 本次网络输入逆时针补偿角度；02 窗口会显示非零补偿，画面与 ROI 仍使用原始坐标 |
| `yolo.instances` | 每个实例的置信度、ROI、`color_gate` 各颜色比例与通过情况、找到的物体数、几何拒绝原因计数 |

离线回放默认同步检测，便于逐帧复现；离线逐帧耗时不能直接当作实时主线程耗时。
