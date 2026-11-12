# Box4D 相机位姿底盘控制（ROS 2 / C++）

控制方块**几何中心**在以相机为原点的底盘水平面满足 `X≈target_distance_m、Y≈target_lateral_m`，默认前方 0.30 m、左侧 0 m。使用完整光学位姿和 UP 投影到水平面，忽略沿 UP 的高度。XY 与方块 yaw 同时闭环调整；按当前角差小角度转正，角差进入 ±5° 后停转，超过 ±8° 才恢复。各轴到位后独立停车并保持监测，偏离恢复门限再调整。默认横向停车容差为 ±2 cm、前后为 ±3 cm；到位后分别超过 3 cm、4.5 cm 才恢复运动，减少观测抖动引起的启停。

本机相机位于车体正中间，水平朝前，光轴与车头方向一致。实车 YAML、所有 launch 和控制库默认 `camera_yaw_deg=0.0`，无需安装偏航补偿。保持水平面 XY 跟随并加入 yaw 对正，目标位置以相机为原点，不加入安装平移。以后若改变朝向，可显式设置 `camera_yaw_deg`（左偏为正，右偏为负）。

通过 `size:=big`（35 cm，默认）或 `size:=small`（20 cm）选择识别和控制规格。小块保留完整位姿、朝向和跟踪；一键 launch 默认选择随仓库权重的 YOLO 配置。识别端只处理所选尺寸，控制端检查 bbox 三轴尺寸（±1 cm）后才锁目标。`size` 在启动时设置，切换需重启。完整安装、权重和跨电脑部署见 [部署说明](../../BOX4D_DEPLOY.zh-CN.md)。

默认 `dry_run=true`，启动即计算速度，每 100 ms 发布一次 Twist 并打印一条 INFO，完全不打开串口。可以先不开底盘观察效果。

## 构建

使用系统 ROS 2 Humble、C++17、Eigen。新控制包与现有识别代码分开，通过 ROS 话题连接。

```bash
cd /home/tomato/6Dpose
source /opt/ros/humble/setup.bash
colcon build --base-paths src_control --packages-select box_pose_control \
  --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

运行依赖：`rclcpp`、`geometry_msgs`、`vision_msgs`、`std_srvs`、Eigen；串口直接使用 Linux `termios2`，默认波特率为 `115200`，与已验证可用的 `run_keyboard_test/launch/teleop_run.launch.py` 所调用的 `get_and_pub.py` 一致。默认采用标准 `B115200` 编码，并显式置位 DTR/RTS、清空串口输入，沿用键盘端 pySerial 的初始化方式；自定义波特率保留 `BOTHER` 支持。测试另使用 GTest、pytest、rclpy。已有识别端的 Python 环境也需要能导入 `vision_msgs.msg`。

## 三个终端启动

所有终端使用同一个 `ROS_DOMAIN_ID`，以下以 **86** 为例。

**终端 1：相机。** 已有同 domain 的相机驱动时复用即可。

```bash
cd /home/tomato/6Dpose
bash src/start_box4d_camera.sh --domain 86 speed
```

**终端 2：识别并发布位姿。** 启动时相机和底盘保持静止约 2 秒，测量 UP。

```bash
cd /home/tomato/6Dpose
bash src/box4d_ros.sh --domain 86 live \
  --config src/config/box4d_d435i_live.json --size big \
  --auto-up --publish-poses \
  --output "outputs/box4d_control_$(date +%Y%m%d_%H%M%S)" \
  --duration 3600 --display --display-smoothed --display-hz 30
```

使用 YOLO 配置时替换 `--config` 和对应 Python 环境即可，发布接口相同。`--publish-poses` 要求配置开启 tracking。只有本次 `--auto-up` 实测成功才发布 UP；仅加载旧 UP 配置时控制端会等待观测配对并保持零速。

**终端 3：只观察控制输出。**

```bash
cd /home/tomato/6Dpose
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=86
ros2 launch box_pose_control control.launch.xml
```

日志示例：

```text
mode=dry_run size=big state=adjusting_distance id=... plane_x=0.600 plane_y=0.000 ex=0.300 ey=0.000 cmd_vx=+0.5000 cmd_vy=+0.0000 vw=0.000 wire_vx=+0.500 wire_vy=+0.000 grip_mode=0 serial=not_opened
```

`cmd_vx/cmd_vy` 是计算指令（m/s），`wire_vx/wire_vy` 是按协议截断到 mm/s 后的指令；不是实测车速。`plane_x/plane_y` 是水平前方/左侧位置，`ex/ey` 是相对目标点的误差。等待或丢失时位置显示 `nan`，速度为零。`grip_mode` 是本周期的夹取指令，dry_run 下仅模拟。实车模式 `serial=sent` 表示写入成功，`failed` 表示发送失败。

常见状态：`waiting_pose` 等待配对数据，`adjusting_position` 同时调整 XY，`aligning` 仅横移对中，`adjusting_distance` 仅调前后距离，`in_position` 到位，`target_lost` 锁定目标丢失，`pose_timeout` 超时，`invalid_geometry` 安装方向无法控制，`serial_fault` 串口故障，`disabled` 已关闭。

## turtlesim 二维演示（不需要相机和底盘）

用 turtlesim 验证同一份 C++ 控制器的速度输出：原有 `turtle1` 当作方块，新海龟 `camera` 当作相机。桥接节点根据两只海龟的真实位姿发布 `/box4d/detections` 与 `/box4d/up_camera`，再订阅 `/box_pose_control/cmd_vel` 驱动 camera。turtlesim 本身是差速模型、不能直接接收 `linear.y`，所以桥接节点以 60 Hz `teleport_absolute` 模拟全向底盘，camera 同时积分 `angular.z` 更新朝向，检测包含方块相对相机的旋转姿态。

```bash
cd /home/tomato/6Dpose
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch box_pose_control turtlesim_demo.launch.xml
```

启动后 `turtle1` 先按“首次知道的相对位置”瞬移到 camera 前方，随后 camera 同时调整横向和前后位置到目标点；控制节点仍以 `dry_run=true` 运行，只发布 Twist。默认 1 个 turtlesim 世界单位 = 0.1 m，因此 `target_distance_m=0.30` 对应 3 个单位。

这个 launch 一个终端就启动了 turtlesim、桥接节点和控制节点，不需要相机/识别端。要让方块持续运动、camera 实时跟踪，加 `box_motion:=sweep`：

```bash
ros2 launch box_pose_control turtlesim_demo.launch.xml box_motion:=sweep
```

也可以只启动 launch，再开第二个终端手动遥控方块（默认控制 `turtle1`）：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=<与 launch 相同>
ros2 run turtlesim turtle_teleop_key
```

`turtle_teleop_key` 的默认速度比 camera 上限快，点按方向键即可；方块跑到 camera 后方（光学 Z≤0）时控制器会暂时停车，把它移回前方即可恢复。

| launch 参数 | 默认 | 含义 |
|---|---:|---|
| `target_distance_m` | 0.30 | 传给控制器的目标距离，m |
| `max_forward_speed` / `max_lateral_speed` | 0.50 / 0.50 | 控制器最大速度，m/s；调小可放慢演示 |
| `meters_per_unit` | 0.1 | 1 个世界单位对应多少米 |
| `initial_forward_units` | 3.5 | 首次检测到的方块前方距离，世界单位 |
| `initial_left_units` | 1.5 | 首次检测到的方块左侧距离，世界单位 |
| `spawn_x` / `spawn_y` / `spawn_theta` | 2.0 / 2.0 / 0.0 | camera 初始位姿 |
| `box_motion` | `static` | `static` 方块静止；`sweep` 方块持续扫动 |
| `box_sweep_lateral_units` / `box_sweep_forward_units` | 1.2 / 0.5 | 扫动幅度，世界单位 |
| `box_sweep_period_s` | 10.0 | 扫动周期，s |

无显示器时可加 `QT_QPA_PLATFORM=offscreen`。两轴均有误差时日志显示 `state=adjusting_position`，某一轴到位后只调整另一轴，最终显示 `state=in_position`；最后 `X≈0、Z≈0.30`（默认横向 ±0.02 m、前后 0.30±0.03 m）。桥接实现见 `scripts/turtlesim_pose_sim.py`。

## 真实数据 + 海龟只读可视化

真实链路（相机、识别、控制）照常运行，再加一个**只读**显示桥接。它只订阅 `/box4d/detections`、`/box4d/up_camera` 和 `/box_pose_control/cmd_vel`，不发布任何 `/box4d/*`，所以不会和控制节点抢目标。

终端 4，必须和真实链路使用同一个 `ROS_DOMAIN_ID`（例如 86）：

```bash
cd /home/tomato/6Dpose
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=86
ros2 launch box_pose_control turtlesim_real_view.launch.xml
```

- `turtle1` 显示真实方块相对相机的位置，`camera` 海龟按真实 `cmd_vel` 平移和旋转。
- 默认比例 `meters_per_unit=0.2`（1 个世界单位 = 0.2 m）；方块超出 turtlesim 世界时调大该值。
- 默认自动锁定最接近光轴的目标；多目标时用 `target_id:=<控制节点 INFO 里的 id>` 指定。
- 左右方向由真实 `/box4d/up_camera` 决定，倒装相机不需要改代码。
- 与控制器一致，先用 UP 投影到水平面，再补偿 `camera_yaw_deg`，二维显示使用底盘前后/左右轴向。

## 一键启动真实链路 + 海龟显示

不想手动开四个终端时，用一个 launch 启动相机驱动、识别、控制、turtlesim 和只读显示：

```bash
cd /home/tomato/6Dpose
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch box_pose_control real_pipeline.launch.py
```

默认 `dry_run=true`；实车运行：

```bash
ros2 launch box_pose_control real_pipeline.launch.py \
  dry_run:=false serial_port:=/dev/ttyUSB0 target_distance_m:=0.30
```

常用参数：

| 参数 | 默认 | 含义 |
|---|---|---|
| `domain` | 86 | 所有子进程使用的 `ROS_DOMAIN_ID` |
| `start_camera` / `start_recognition` | true / true | 是否启动相机驱动／识别；已在别处运行时设为 false |
| `size` | `big` | big=35 cm，small=20 cm，同时作用于识别、控制和视图 |
| `config` | 空，按 size 选择 | big 使用几何配置，small 使用仓库 YOLO 配置；可显式覆盖 |
| `duration` | 3600 | 识别运行时长，s |
| `display` | true | 是否打开识别预览窗口 |
| `with_view` | true | 是否启动 turtlesim + 只读显示 |
| `dry_run` / `serial_port` | true / `/dev/ttyUSB0` | 控制节点输出模式 |
| `target_distance_m` / `target_lateral_m` | 0.30 / 0.0 | 水平前方/左侧目标位置，m |
| `camera_yaw_deg` | 0.0 | 相机居中、水平朝前，无安装偏航；控制与视图共用 |
| `max_forward_speed` / `max_lateral_speed` | 0.50 / 0.50 | 控制器速度上限，m/s |
| `view_meters_per_unit` | 0.2 | 海龟显示比例，m/单位 |
| `view_target_id` | 空 | 多目标时指定显示 ID |
| `view_spawn_x` / `view_spawn_y` / `view_spawn_theta` | 1.0 / 5.5 / 0.0 | camera 海龟初始位姿 |

这个 launch 的所有子进程日志都带前缀输出在**同一个终端**，turtlesim 会另开一个窗口；不是四个独立终端窗口。相机和识别脚本会自行 source ROS 环境；使用 YOLO 或虚拟环境时按原 README 配置好 `BOX4D_PYTHON` 或先激活环境。`Ctrl+C` 会一起停止；实车模式下控制节点退出前会尝试发送停车帧。

如果确实想要四个独立的 GNOME 终端窗口，可以用仓库里的包装脚本（需要 `gnome-terminal` 和图形会话）：

```bash
bash src_control/box_pose_control/scripts/run_real_pipeline_windows.sh
```

```bash
# 只打印四条命令，不打开窗口
bash src_control/box_pose_control/scripts/run_real_pipeline_windows.sh --print

# 覆盖常用配置
ROS_DOMAIN_ID=86 DRY_RUN=false SERIAL_PORT=/dev/ttyUSB0 \
  bash src_control/box_pose_control/scripts/run_real_pipeline_windows.sh
```

四个窗口分别是：`1 camera`、`2 recognition`、`3 control`、`4 turtlesim view`。脚本里的识别输出目录默认带时间戳，可用 `OUTPUT=...`、`CONFIG=...`、`DURATION=...`、`VIEW_SCALE=...` 覆盖。

## 实车、启停和调参

实车运行时由该节点独占串口，关闭使用同一串口的旧 `yolo_control`、`get_and_pub` 等发送节点。保持相机启动标定时静止，首次观测有效后才会输出运动指令。

```bash
ros2 launch box_pose_control control.launch.xml \
  dry_run:=false serial_port:=/dev/ttyUSB0 target_distance_m:=0.30
```

停止、重新选目标和修改距离：

```bash
ros2 service call /box_pose_control/set_enabled std_srvs/srv/SetBool '{data: false}'
ros2 service call /box_pose_control/set_enabled std_srvs/srv/SetBool '{data: true}'
ros2 param set /box_pose_control target_distance_m 0.40
ros2 topic echo /box_pose_control/cmd_vel
ros2 topic hz /box_pose_control/cmd_vel
```

关闭控制会发零速度、释放串口并清除目标；重新开启等待新帧。重复开启不会切换已锁定目标。`Ctrl+C` 退出也尝试发送停车帧并关闭串口；串口故障后保持停止，修复后通过服务重新开启。

默认参数在 `config/control.yaml`：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `control_rate` | 10.0 | 速度与 INFO 输出频率，Hz |
| `dry_run` / `start_enabled` | true / true | 预览模式／启动后启用计算 |
| `size` | big | 35 cm 大块；small 为 20 cm 小块 |
| `target_distance_m` / `target_lateral_m` | 0.30 / 0.0 | 中心水平前方/左侧目标位置，m |
| `camera_yaw_deg` | 0.0 | 本机 YAML 的安装偏航角，度；相机居中、水平朝前 |
| `lateral_gain` / `forward_gain` | 2.0 / 4.0 | 比例增益，s⁻¹；横向调整较缓、前后调整较快 |
| `lateral_tolerance_m` / `lateral_resume_m` | 0.02 / 0.03 | 横向停车／恢复门限，m |
| `distance_tolerance_m` / `distance_resume_m` | 0.03 / 0.045 | 距离停车／恢复门限，m |
| `max_forward_speed` / `max_lateral_speed` | 0.50 / 0.50 | 两轴速度上限，m/s，各不超过 0.50 |
| `observation_timeout_s` | 0.5 | 源消息和接收超时，s |
| `target_reacquire_timeout_s` | 1.0 | 距上次有效目标观测超过此时间且目标已丢失后，允许重新选取同规格 ID，s |
| `serial_port` / `baudrate` | /dev/ttyUSB0 / 115200 | 串口配置 |

`target_distance_m` 和 `target_lateral_m` 支持运行时修改，其余为启动参数。使用自定义 YAML 时可直接启动节点，保持文件中的全部配置：

```bash
ros2 run box_pose_control box_pose_control_node --ros-args \
  --params-file /home/tomato/6Dpose/src_control/box_pose_control/config/control.yaml
```

一键实车与仿真 launch 的 `max_forward_speed`、`max_lateral_speed` 默认均为 `0.50 m/s`；一键实车 launch 会覆盖 YAML 同名值。可在启动命令中传 `max_forward_speed:=0.20 max_lateral_speed:=0.20` 降低速度上限。

`control.launch.xml` 也接受 `params_file:=...`，但 launch 的 `dry_run`、`size`、`target_distance_m`、`target_lateral_m`、`camera_yaw_deg`、`serial_port`、`yaw_enabled`、`serial_yaw_sign`、`serial_yaw_scale` 参数会覆盖 YAML 同名值。

## 坐标、选目标与代码入口

- 输入 `/box4d/detections` 为 `vision_msgs/Detection3DArray`：`bbox.center` 是当前帧原始完整位姿，`bbox.size` 是边长，`id` 为进程标识加跟踪 ID。无目标发布空数组。
- 输入 `/box4d/up_camera` 为 `geometry_msgs/Vector3Stamped`，两路使用同一图像时间戳和光学 frame；QoS 为 reliable、volatile、depth 10。控制端缓存最近 10 个时间戳配对，不拿旧姿态填补丢失帧。
- 输出 `/box_pose_control/cmd_vel` 是计算结果，供调试观察；节点不经过旧 `/cmd_vel` 转发器，实车模式直接编码串口帧。
- 先按所选尺寸过滤，再按 `(X/Z)²+(Y/Z)²` 选择最接近光轴的有效目标，之后锁 ID。目标丢失时停车；短暂遮挡期间只等待原 ID。距上次有效观测超过 `target_reacquire_timeout_s`（默认 1 秒）且目标仍丢失时，允许重新选择同规格目标，原 ID 在候选中时仍优先。新 ID 必须通过新鲜度、尺寸、坐标系和同帧 UP 配对检查才会恢复控制，并打印 `目标重新锁定: old_id=... new_id=...`。识别进程重启导致 ID 改变也可以自动恢复；多个同规格方块在场时，重新选择可能对应另一个方块。
- 相机光学坐标 X 向右、Y 向下、Z 向前；底盘 vx 向前、vy 向左。控制同时使用水平前方/左侧误差生成 XY 速度，各轴独立判断到位、独立限幅到各自速度上限。前后误差较大时不会连带压低横向指令，例如 `ex=0.836、ey=0.045` 时默认输出约 `vx=0.50、vy=0.09 m/s`。启动日志打印实际生效的两轴增益和速度上限。
- 水平投影：`u=normalize(UP)`，`f=normalize(ez−u·u.z)`，`l=u×f`；先计算相机水平位置 `(a,b)=(p·f,p·l)`，再按安装角 `θ=camera_yaw_deg` 换算底盘轴向：`X=cosθ·a−sinθ·b`，`Y=sinθ·a+cosθ·b`。目标为 `(target_distance_m,target_lateral_m)`，INFO 的 `plane_x/plane_y` 和误差也使用同一换算结果。原点仍是相机。沿 UP 的高度分量自动消除；光轴几乎竖直（水平投影长度 <0.25）时停车。
- 假定相机固定，底盘在水平地面运动。UP 补偿俯仰/横滚，不能测量安装偏航，偏航由 `camera_yaw_deg` 显式给出。安装改变后重新设置偏航角并重启识别、测量 UP；高度不参与闭环；方块完整旋转姿态投影到该水平面后参与 yaw 闭环。

`src/controller.cpp` 负责纯控制计算，`src/control_node.cpp` 负责配对、锁目标、参数、10 Hz 调度和 INFO，`src/serial_port.cpp` 负责底盘协议和串口 I/O。

## 方块 yaw 自动对正

目标是根据方块当前的朝向误差小角度转正，**没有固定转 90° 的动作**。
例如偏左 12° 就向左调整，进入 ±5° 即停转；±5°～±8° 是停转后的保持区，超过 ±8° 才恢复调整。
无标记方块的四个侧面等价，角差按 90° 周期折算到 [-45°,45°)，选择最近的一面对正。
使用完整姿态四元数与 UP，并应用 `camera_yaw_deg` 安装补偿，不使用图像边缘斜率代替水平 yaw。

默认三帧不同源时间的连续观测通过变化门限后才允许转向。同一帧重复计算不增加确认次数；
相邻帧角差变化超过 `yaw_jump_deg + max_yaw_speed × dt × 180/π`（度）、无效姿态或断帧超过 0.5 秒后重新确认。
这只是抑制明显跳变，不代表视觉真实角度误差已有 ±5° 精度保证。

- 水平中心距离大于 `yaw_fine_distance_m=0.7` m 时，转速上限为 `far_max_yaw_speed=0.2` rad/s。
- 近处转速上限为 `max_yaw_speed=0.4` rad/s；角差超过 15° 时，前后指令降为原值的 25%。
- yaw 未确认时不转向：远处 XY 分别限速到 0.15 m/s，近处暂停所有运动并禁止夹取。
- XY 与 yaw 同时反馈；转动后按新观测继续调整 XY。丢失、超时、关闭及退出均发送三轴零速。
- 默认夹取要求 XY 和 yaw 的当前观测都进入停车容差并连续稳定 0.4 秒；保持区不能触发夹取。
- `yaw_enabled=false` 恢复旧 XY 控制与原夹取时序，可用于独立回归比较。

一键启动和 `control.launch.xml` 可传 `yaw_enabled:=false` 恢复 XY 模式，也可传 `serial_yaw_sign`、`serial_yaw_scale`。
配置在 `config/control.yaml`，上述值均为启动参数；另有 `yaw_gain=1.2`、`yaw_jump_deg=8.0`、
`yaw_confirm_frames=3`、`yaw_unreliable_approach_speed=0.15`。
日志新增 `yaw_error_deg`、`yaw_ready`、`vw`（ROS rad/s）和 `wire_wz`（串口截断后的接口值）。
`aligning_yaw` 表示 XY 已到位但仍需转正，`waiting_yaw` 表示姿态等待确认。

串口单位和方向沿用父目录旧控制链的接口推断：
`run_keyboard_test/launch/teleop_run.launch.py` 将键盘 `turn` 设为 3.14；
实际安装的键盘程序 `j/l` 持续输出正负角速度，没有固定角度动作；
`get_and_pub.py` 将 `angular.z` 取反后乘 10 编码。
旁边 `nav_to_pose.py::adj_angle` 用弧度角差除以转向速度估算持续时间（还带经验修正）。
内部仍采用 rad/s，默认 `serial_yaw_sign=1.0`、`serial_yaw_scale=1.0`。
早期按旧键盘程序推断取反，但实车反馈为“越转越歪，最后对到相邻面”；
这与反向反馈一致，所以当前默认直接发送正向角速度。旧取反约定仅保留为可配置选项，不能照搬到这台车。
角差本身始终在 ±45° 内；方向错误仍会使实际车头绕到相邻面，限角差不能修复转向符号。
重启后观察 `yaw_error_deg` 的绝对值是否随转动减小，以确认实车收敛方向。
这些源码支持接口约定推断，不能证明下位机实际速度或方向已实测；二者均留作可调启动参数。
串口角速度分辨率为接口值 0.1，容差外的默认最小指令为 0.1 rad/s，避免截断为零。

## 底盘串口协议

按电控提供的 Python `struct.pack('<B h h h B', ...)` 编码（Mode 传入 `mode & 0xFF`），每帧共 **10 字节**，下标从 0 开始：

| 下标 | 字段 | 编码 |
|---|---|---|
| 0 | SOF | `0xA5` |
| 1～2 | Vx | `int(vx * 1000)`，有符号 int16，小端序 |
| 3～4 | Vy | `int(vy * 1000)`，有符号 int16，小端序 |
| 5～6 | Vw | `int(wire_wz * 10)`，有符号 int16，小端序；`wire_wz=serial_yaw_sign × serial_yaw_scale × wz` |
| 7 | Mode | 有符号单字节：平时为 0，到位夹取为 1，1 秒后发 -1（`0xFF`） |
| 8 | CRC8 | 仅覆盖下标 1～7，初值 `0x00`，多项式 `0x07`，高位优先，无最终异或 |
| 9 | EOF | `0xFF` |

线速度单位为 m/s，编码前仍限幅到 ±0.50 m/s，乘以 1000 后向零截断。控制器同时输出平移和转向，ROS 的 `angular.z` 为左转正值、单位 rad/s。新协议删除旧 12 字节帧中多余的两个标志字节，并相应调整 CRC 范围和位置。

例如 `vx=0.1239, vy=-0.0459` 时发送 `A5 7B 00 D3 FF 00 00 00 A1 FF`；停车帧为 `A5 00 00 00 00 00 00 00 00 FF`。

默认 yaw 模式下，锁定目标前后、左右和角度均进入停车容差，且源时间连续稳定至少 `grip_stable_s=0.4` 秒后，仅一帧发送 `mode=1`；从成功写入开始用单调时钟计时，满 1 秒后的首个控制周期仅一帧发送 `mode=-1`（默认 10 Hz，间隔约 1.0～1.1 秒）。其他所有帧为 `mode=0`，控制循环不会等待或阻塞。零速夹取帧为 `A5 00 00 00 00 00 00 01 07 FF`，零速 `-1` 帧为 `A5 00 00 00 00 00 00 FF F3 FF`。

同一目标只触发一次，保持到位、位置抖动或同 ID 丢失重现都不会重复夹取。修改目标坐标、重新锁定不同 ID、关闭再开启控制后允许再次触发；重复开启服务不会重置动作。夹取开始后，即使目标遮挡或观测超时，仍按时发送 `-1`；关闭控制、退出或串口故障会取消未完成动作，并按原流程尝试发送 `mode=0` 的零速帧。dry_run 仅计算并记录 `grip_mode`，不打开串口。

## 测试

```bash
cd /home/tomato/6Dpose
source /opt/ros/humble/setup.bash
source install/setup.bash
colcon test --base-paths src_control --packages-select box_pose_control --event-handlers console_direct+
colcon test-result --verbose
PYTHONPATH="$PWD/src:$PYTHONPATH" /usr/bin/python3 -m pytest -q src/tests/test_box4d_ros_output.py
```

GTest 测控制公式和真实伪终端协议；ROS 测试启动 C++ 节点，以独立 domain、测试专用话题和伪终端验证 10 Hz、INFO、目标锁定、丢失停车、俯仰、启停、改距离和退出停车。测试不连接真实底盘；实车距离误差与动态响应需要现场验证。
