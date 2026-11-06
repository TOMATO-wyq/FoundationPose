# Box4D 相机位姿底盘控制（ROS 2 / C++）

控制方块**几何中心**在相机光学坐标中满足 `X≈0、Z≈0.30 m`。30 cm 指光学 Z 值，不是方块表面距离。先横移对中，再前后调整；角速度始终为零，到位后保持监测，偏离再调整。

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

运行依赖：`rclcpp`、`geometry_msgs`、`vision_msgs`、`std_srvs`、Eigen；串口直接使用 Linux `termios2`，支持旧底盘的 `961200` 波特率。测试另使用 GTest、pytest、rclpy。已有识别端的 Python 环境也需要能导入 `vision_msgs.msg`。

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
  --config src/config/box4d_d435i_live.json \
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
mode=dry_run state=adjusting_distance id=... X=0.000 Z=0.600 ex=0.000 ez=0.300 cmd_vx=+0.1000 cmd_vy=+0.0000 vw=0.000 wire_vx=+0.100 wire_vy=+0.000 serial=not_opened
```

`cmd_vx/cmd_vy` 是计算指令（m/s），`wire_vx/wire_vy` 是按旧协议截断到 mm/s 后的指令；不是实测车速。等待或丢失时 X/Z 显示 `nan`，速度为零。实车模式 `serial=sent` 表示写入成功，`failed` 表示发送失败。

常见状态：`waiting_pose` 等待配对数据，`aligning` 横移对中，`adjusting_distance` 调距离，`in_position` 到位，`target_lost` 锁定目标丢失，`pose_timeout` 超时，`invalid_geometry` 安装方向无法控制，`serial_fault` 串口故障，`disabled` 已关闭。

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
| `target_distance_m` | 0.30 | 方块中心目标 Z，m |
| `lateral_gain` / `forward_gain` | 1.0 / 1.0 | 比例增益，s⁻¹ |
| `lateral_tolerance_m` / `lateral_resume_m` | 0.01 / 0.015 | 横向停车／恢复门限，m |
| `distance_tolerance_m` / `distance_resume_m` | 0.02 / 0.03 | 距离停车／恢复门限，m |
| `max_forward_speed` / `max_lateral_speed` | 0.10 / 0.10 | 两轴速度上限，m/s，各不超过 0.30 |
| `observation_timeout_s` | 0.5 | 源消息和接收超时，s |
| `serial_port` / `baudrate` | /dev/ttyUSB0 / 961200 | 串口配置 |

只有目标距离支持运行时修改，其余为启动参数。使用自定义 YAML 时可直接启动节点，保持文件中的全部配置：

```bash
ros2 run box_pose_control box_pose_control_node --ros-args \
  --params-file /home/tomato/6Dpose/src_control/box_pose_control/config/control.yaml
```

launch 也接受 `params_file:=...`，但 launch 的 `dry_run`、`target_distance_m`、`serial_port` 三个参数会覆盖 YAML 同名值。

## 坐标、选目标与代码入口

- 输入 `/box4d/detections` 为 `vision_msgs/Detection3DArray`：`bbox.center` 是当前帧原始完整位姿，`bbox.size` 是边长，`id` 为进程标识加跟踪 ID。无目标发布空数组。
- 输入 `/box4d/up_camera` 为 `geometry_msgs/Vector3Stamped`，两路使用同一图像时间戳和光学 frame；QoS 为 reliable、volatile、depth 10。控制端缓存最近 10 个时间戳配对，不拿旧姿态填补丢失帧。
- 输出 `/box_pose_control/cmd_vel` 是计算结果，供调试观察；节点不经过旧 `/cmd_vel` 转发器，实车模式直接编码串口帧。
- 首次按 `(X/Z)²+(Y/Z)²` 选择最接近光轴的有效目标，之后锁 ID。遮挡后 ID 改变或识别进程重启时保持停车，需要关闭再开启以重新选择。
- 相机光学坐标 X 向右、Y 向下、Z 向前；底盘 vx 向前、vy 向左。水平安装时 `vx=Kz·(Z−目标距离)`、`vy=−Kx·X`，具体按当前控制阶段只调整一个光学方向。
- 俯仰补偿：`u=normalize(UP)`，`f=normalize(ez−u·u.z)`，`l=u×f`；取 f/l 的 X/Z 分量组成 J，求解 `J·v=期望相机平移`。静止目标相对相机的运动为 `−J·v`，因此误差减小。两轴共同缩放限速，保留该映射关系。
- 假定相机固定，光轴的水平投影与底盘前进方向一致，底盘在水平地面运动。UP 不能测量相机相对底盘的安装偏航。安装改变后重启识别、重新测量 UP；Y 与方块朝向不参与闭环。

`src/controller.cpp` 负责纯控制计算，`src/control_node.cpp` 负责配对、锁目标、参数、10 Hz 调度和 INFO，`src/serial_port.cpp` 负责旧底盘协议和串口 I/O。

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
