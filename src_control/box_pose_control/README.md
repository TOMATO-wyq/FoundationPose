# Box4D 相机位姿底盘控制（ROS 2 / C++）

控制方块**几何中心**在以相机为原点的底盘水平面满足 `X≈target_distance_m、Y≈target_lateral_m`，默认前方 0.30 m、左侧 0 m。使用完整光学位姿和 UP 投影到水平面，忽略沿 UP 的高度。先横移对齐，再前后调整；角速度始终为零，到位后保持监测，偏离再调整。

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

运行依赖：`rclcpp`、`geometry_msgs`、`vision_msgs`、`std_srvs`、Eigen；串口直接使用 Linux `termios2`，默认波特率沿用 `961200`（本次电控协议示例未提供新的波特率）。测试另使用 GTest、pytest、rclpy。已有识别端的 Python 环境也需要能导入 `vision_msgs.msg`。

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
mode=dry_run size=big state=adjusting_distance id=... plane_x=0.600 plane_y=0.000 ex=0.300 ey=0.000 cmd_vx=+0.1000 cmd_vy=+0.0000 vw=0.000 wire_vx=+0.100 wire_vy=+0.000 serial=not_opened
```

`cmd_vx/cmd_vy` 是计算指令（m/s），`wire_vx/wire_vy` 是按协议截断到 mm/s 后的指令；不是实测车速。`plane_x/plane_y` 是水平前方/左侧位置，`ex/ey` 是相对目标点的误差。等待或丢失时位置显示 `nan`，速度为零。实车模式 `serial=sent` 表示写入成功，`failed` 表示发送失败。

常见状态：`waiting_pose` 等待配对数据，`aligning` 横移对中，`adjusting_distance` 调距离，`in_position` 到位，`target_lost` 锁定目标丢失，`pose_timeout` 超时，`invalid_geometry` 安装方向无法控制，`serial_fault` 串口故障，`disabled` 已关闭。

## turtlesim 二维演示（不需要相机和底盘）

用 turtlesim 验证同一份 C++ 控制器的速度输出：原有 `turtle1` 当作方块，新海龟 `camera` 当作相机。桥接节点根据两只海龟的真实位姿发布 `/box4d/detections` 与 `/box4d/up_camera`，再订阅 `/box_pose_control/cmd_vel` 驱动 camera。turtlesim 本身是差速模型、不能直接接收 `linear.y`，所以桥接节点以 60 Hz `teleport_absolute` 模拟全向底盘，camera 朝向保持不变。

```bash
cd /home/tomato/6Dpose
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch box_pose_control turtlesim_demo.launch.xml
```

启动后 `turtle1` 先按“首次知道的相对位置”瞬移到 camera 前方，随后 camera 先横移对中，再前进到目标距离；控制节点仍以 `dry_run=true` 运行，只发布 Twist。默认 1 个 turtlesim 世界单位 = 0.1 m，因此 `target_distance_m=0.30` 对应 3 个单位。

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
| `max_forward_speed` / `max_lateral_speed` | 0.10 / 0.10 | 控制器最大速度，m/s；调小可放慢演示 |
| `meters_per_unit` | 0.1 | 1 个世界单位对应多少米 |
| `initial_forward_units` | 3.5 | 首次检测到的方块前方距离，世界单位 |
| `initial_left_units` | 1.5 | 首次检测到的方块左侧距离，世界单位 |
| `spawn_x` / `spawn_y` / `spawn_theta` | 2.0 / 2.0 / 0.0 | camera 初始位姿 |
| `box_motion` | `static` | `static` 方块静止；`sweep` 方块持续扫动 |
| `box_sweep_lateral_units` / `box_sweep_forward_units` | 1.2 / 0.5 | 扫动幅度，世界单位 |
| `box_sweep_period_s` | 10.0 | 扫动周期，s |

无显示器时可加 `QT_QPA_PLATFORM=offscreen`。正常情况下日志依次出现 `state=aligning`、`state=adjusting_distance`、`state=in_position`，最后 `X≈0、Z≈0.30`（默认容差内为 0.30±0.02 m）。桥接实现见 `scripts/turtlesim_pose_sim.py`。

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

- `turtle1` 显示真实方块相对相机的位置，`camera` 海龟按真实 `cmd_vel` 平移。
- 默认比例 `meters_per_unit=0.2`（1 个世界单位 = 0.2 m）；方块超出 turtlesim 世界时调大该值。
- 默认自动锁定最接近光轴的目标；多目标时用 `target_id:=<控制节点 INFO 里的 id>` 指定。
- 左右方向由真实 `/box4d/up_camera` 决定，倒装相机不需要改代码。
- 当前角度量按控制器的假设使用 UP 的水平投影；二维显示忽略高度 Y。

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
| `max_forward_speed` / `max_lateral_speed` | 0.10 / 0.10 | 控制器速度上限，m/s |
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
| `lateral_gain` / `forward_gain` | 1.0 / 1.0 | 比例增益，s⁻¹ |
| `lateral_tolerance_m` / `lateral_resume_m` | 0.01 / 0.015 | 横向停车／恢复门限，m |
| `distance_tolerance_m` / `distance_resume_m` | 0.02 / 0.03 | 距离停车／恢复门限，m |
| `max_forward_speed` / `max_lateral_speed` | 0.10 / 0.10 | 两轴速度上限，m/s，各不超过 0.30 |
| `observation_timeout_s` | 0.5 | 源消息和接收超时，s |
| `serial_port` / `baudrate` | /dev/ttyUSB0 / 961200 | 串口配置 |

`target_distance_m` 和 `target_lateral_m` 支持运行时修改，其余为启动参数。使用自定义 YAML 时可直接启动节点，保持文件中的全部配置：

```bash
ros2 run box_pose_control box_pose_control_node --ros-args \
  --params-file /home/tomato/6Dpose/src_control/box_pose_control/config/control.yaml
```

launch 也接受 `params_file:=...`，但 launch 的 `dry_run`、`size`、`target_distance_m`、`target_lateral_m`、`serial_port` 参数会覆盖 YAML 同名值。

## 坐标、选目标与代码入口

- 输入 `/box4d/detections` 为 `vision_msgs/Detection3DArray`：`bbox.center` 是当前帧原始完整位姿，`bbox.size` 是边长，`id` 为进程标识加跟踪 ID。无目标发布空数组。
- 输入 `/box4d/up_camera` 为 `geometry_msgs/Vector3Stamped`，两路使用同一图像时间戳和光学 frame；QoS 为 reliable、volatile、depth 10。控制端缓存最近 10 个时间戳配对，不拿旧姿态填补丢失帧。
- 输出 `/box_pose_control/cmd_vel` 是计算结果，供调试观察；节点不经过旧 `/cmd_vel` 转发器，实车模式直接编码串口帧。
- 先按所选尺寸过滤，再按 `(X/Z)²+(Y/Z)²` 选择最接近光轴的有效目标，之后锁 ID。遮挡后 ID 改变或识别进程重启时保持停车，需要关闭再开启以重新选择。
- 相机光学坐标 X 向右、Y 向下、Z 向前；底盘 vx 向前、vy 向左。控制仅使用水平前方/左侧误差，按当前阶段只调整其中一个方向。
- 水平投影：`u=normalize(UP)`，`f=normalize(ez−u·u.z)`，`l=u×f`；位置为 `(p·f,p·l)`，目标为 `(target_distance_m,target_lateral_m)`。沿 UP 的高度分量自动消除；光轴几乎竖直（水平投影长度 <0.25）时停车。
- 假定相机固定，光轴的水平投影与底盘前进方向一致，底盘在水平地面运动。UP 不能测量相机相对底盘的安装偏航。安装改变后重启识别、重新测量 UP；高度与方块朝向不参与闭环，但完整位姿仍保留在识别输出中。

`src/controller.cpp` 负责纯控制计算，`src/control_node.cpp` 负责配对、锁目标、参数、10 Hz 调度和 INFO，`src/serial_port.cpp` 负责底盘协议和串口 I/O。

## 底盘串口协议

按电控提供的 Python `struct.pack('<B h h h B', ...)` 编码，每帧共 **10 字节**，下标从 0 开始：

| 下标 | 字段 | 编码 |
|---|---|---|
| 0 | SOF | `0xA5` |
| 1～2 | Vx | `int(vx * 1000)`，有符号 int16，小端序 |
| 3～4 | Vy | `int(vy * 1000)`，有符号 int16，小端序 |
| 5～6 | Vw | `int(vw * 10)`，有符号 int16，小端序；当前固定为 0 |
| 7 | Mode | 单字节，当前固定为 0 |
| 8 | CRC8 | 仅覆盖下标 1～7，初值 `0x00`，多项式 `0x07`，高位优先，无最终异或 |
| 9 | EOF | `0xFF` |

线速度单位为 m/s，编码前仍限幅到 ±0.30 m/s，乘以 1000 后向零截断。当前控制器仅做平移。新协议删除旧 12 字节帧中多余的两个标志字节，并相应调整 CRC 范围和位置。

例如 `vx=0.1239, vy=-0.0459` 时发送 `A5 7B 00 D3 FF 00 00 00 A1 FF`；停车帧为 `A5 00 00 00 00 00 00 00 00 FF`。

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
