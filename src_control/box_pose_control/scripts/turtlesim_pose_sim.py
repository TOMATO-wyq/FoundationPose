#!/usr/bin/env python3
"""用 turtlesim 演示 box_pose_control 的视觉伺服。

角色分配：
  * 原有 ``turtle1``：被识别的方块，启动时瞬移到相机前方的初始相对位置，之后静止；
  * 新建 ``camera``：相机/底盘，由 ``/box_pose_control/cmd_vel`` 驱动。

数据流与真实系统一致：
  camera/turtle1 真实位姿 -> 本节点的假检测 -> /box4d/detections + /box4d/up_camera
  -> box_pose_control_node（dry_run）-> /box_pose_control/cmd_vel
  -> 本节点把 vx/vy/wz 积分成 camera 的世界位姿。

注意：turtlesim 是差速模型，``/camera/cmd_vel`` 只接受 linear.x，直接发布
linear.y 会被忽略。这里在 60 Hz 定时器里用 teleport_absolute 同时模拟
全向平移和旋转，并发布方块相对相机的真实旋转姿态。
"""

import math
import time

import rclpy
from geometry_msgs.msg import Twist, Vector3Stamped
from rclpy.node import Node
from turtlesim.msg import Pose
from turtlesim.srv import Spawn, TeleportAbsolute
from vision_msgs.msg import Detection3D, Detection3DArray

# turtlesim 世界约为 11.088 x 11.088，留出 turtle 半径，避免贴墙。
WORLD_MIN = 0.6
WORLD_MAX = 10.4


class TurtlesimPoseSim(Node):
    def __init__(self):
        super().__init__("turtlesim_pose_sim")

        # ---- 仿真布局参数（除 scale 外，长度单位均为 turtlesim 世界单位）----
        self.camera_name = self.declare_parameter("camera_name", "camera").value
        self.box_name = self.declare_parameter("box_name", "turtle1").value
        self.spawn_x = float(self.declare_parameter("spawn_x", 2.0).value)
        self.spawn_y = float(self.declare_parameter("spawn_y", 2.0).value)
        self.spawn_theta = float(self.declare_parameter("spawn_theta", 0.0).value)
        # “第一次知道的方块相对相机位置”：前方 initial_forward、左侧 initial_left。
        self.initial_forward = float(
            self.declare_parameter("initial_forward_units", 3.5).value
        )
        self.initial_left = float(
            self.declare_parameter("initial_left_units", 1.5).value
        )

        # 控制器按米计算，这里声明 1 个 turtlesim 单位等于多少米。
        # 0.1 m/unit 时默认 target_distance_m=0.30 正好对应 3.0 个世界单位。
        self.scale = float(self.declare_parameter("meters_per_unit", 0.1).value)
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("meters_per_unit must be a positive finite number")

        # 30 Hz 发布检测（控制器 10 Hz 计算），60 Hz 执行运动。
        self.publish_rate = float(self.declare_parameter("publish_rate", 30.0).value)
        self.motion_rate = float(self.declare_parameter("motion_rate", 60.0).value)
        self.cmd_timeout = float(self.declare_parameter("cmd_timeout_s", 0.3).value)
        self.frame_id = str(
            self.declare_parameter("frame_id", "turtlesim_camera_optical").value
        )
        size = self.declare_parameter("size", "big").value
        if size not in ("big", "small"):
            raise ValueError("size must be big or small")
        self.size_m = 0.35 if size == "big" else 0.20
        self.target_id = str(
            self.declare_parameter("target_id", "turtlesim:box").value
        )
        for name, value in (
            ("publish_rate", self.publish_rate),
            ("motion_rate", self.motion_rate),
            ("cmd_timeout_s", self.cmd_timeout),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
        if not self.frame_id or not self.target_id:
            raise ValueError("frame_id and target_id must be non-empty")

        # static：方块瞬移一次后静止；sweep：方块持续横向/前后扫动，camera 实时跟踪。
        self.box_motion = str(self.declare_parameter("box_motion", "static").value)
        self.box_sweep_lateral = float(
            self.declare_parameter("box_sweep_lateral_units", 1.2).value
        )
        self.box_sweep_forward = float(
            self.declare_parameter("box_sweep_forward_units", 0.5).value
        )
        self.box_sweep_period = float(
            self.declare_parameter("box_sweep_period_s", 10.0).value
        )
        if self.box_motion not in ("static", "sweep"):
            raise ValueError("box_motion must be 'static' or 'sweep'")
        for name, value in (
            ("box_sweep_lateral_units", self.box_sweep_lateral),
            ("box_sweep_forward_units", self.box_sweep_forward),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a nonnegative finite number")
        if not math.isfinite(self.box_sweep_period) or self.box_sweep_period <= 0.0:
            raise ValueError("box_sweep_period_s must be a positive finite number")

        # ---- 状态 ----
        self.camera_pose = None   # /camera/pose 的最新真实位姿
        self.box_pose = None      # /turtle1/pose 的最新真实位姿
        self.cam_x = None         # 用于发 teleport 的本地积分位姿
        self.cam_y = None
        self.cam_theta = None
        self.ready = False
        self.state = "wait_spawn"
        self.spawn_future = None
        self.box_teleport_future = None
        self.camera_teleport_future = None
        self.box_motion_future = None
        self.teleport_target = None
        self.box_origin = None
        self.ready_at = None
        self.retry_at = 0.0
        self.last_step = time.monotonic()

        self.cmd = Twist()
        self.cmd_at = None  # monotonic 时间；超时后按零速处理

        # ---- ROS 接口 ----
        self.detection_pub = self.create_publisher(
            Detection3DArray, "/box4d/detections", 10
        )
        self.up_pub = self.create_publisher(
            Vector3Stamped, "/box4d/up_camera", 10
        )
        self.create_subscription(
            Twist, "/box_pose_control/cmd_vel", self.on_cmd, 10
        )
        self.create_subscription(
            Pose, f"/{self.box_name}/pose", self.on_box_pose, 10
        )
        self.create_subscription(
            Pose, f"/{self.camera_name}/pose", self.on_camera_pose, 10
        )

        self.spawn_client = self.create_client(Spawn, "/spawn")
        self.box_teleport_client = self.create_client(
            TeleportAbsolute, f"/{self.box_name}/teleport_absolute"
        )
        self.camera_teleport_client = self.create_client(
            TeleportAbsolute, f"/{self.camera_name}/teleport_absolute"
        )

        self.create_timer(0.05, self.startup_tick)
        self.create_timer(1.0 / self.publish_rate, self.publish_tick)
        self.create_timer(1.0 / self.motion_rate, self.motion_tick)
        self.create_timer(1.0 / self.publish_rate, self.box_motion_tick)

        self.get_logger().info(
            f"等待 turtlesim：camera={self.camera_name}, box={self.box_name}, "
            f"首次相对位置=前方 {self.initial_forward:.2f} / 左侧 {self.initial_left:.2f} 单位"
        )

    # ------------------------------------------------------------------ 回调
    def on_cmd(self, message: Twist):
        self.cmd = message
        self.cmd_at = time.monotonic()

    def on_box_pose(self, message: Pose):
        self.box_pose = message

    def on_camera_pose(self, message: Pose):
        self.camera_pose = message
        if self.cam_x is None:
            self.cam_x = message.x
            self.cam_y = message.y
            self.cam_theta = message.theta
            return
        # teleport 服务异常或有人手动拖拽海龟时，用真实位姿纠正本地积分。
        pending = (
            self.camera_teleport_future is not None
            and not self.camera_teleport_future.done()
        )
        angle_error = math.atan2(
            math.sin(message.theta - self.cam_theta),
            math.cos(message.theta - self.cam_theta),
        )
        if not pending and (
            math.hypot(message.x - self.cam_x, message.y - self.cam_y) > 0.5
            or abs(angle_error) > 0.1
        ):
            self.cam_x = message.x
            self.cam_y = message.y
            self.cam_theta = message.theta

    # -------------------------------------------------------------- 启动流程
    def startup_tick(self):
        now = time.monotonic()
        if self.state == "wait_spawn":
            if now < self.retry_at:
                return
            if not self.spawn_client.service_is_ready():
                return
            request = Spawn.Request()
            request.x = self.spawn_x
            request.y = self.spawn_y
            request.theta = self.spawn_theta
            request.name = self.camera_name
            self.spawn_future = self.spawn_client.call_async(request)
            self.state = "spawning"

        elif self.state == "spawning":
            if not self.spawn_future.done():
                return
            error = self.spawn_future.exception()
            if error is not None:
                # 名字已存在时 turtlesim 可能仍返回成功；这里只记录并继续等位姿。
                self.get_logger().warn(f"/spawn 调用失败，假定 {self.camera_name} 已存在: {error}")
            self.state = "wait_poses"

        elif self.state == "wait_poses":
            if self.camera_pose is None or self.box_pose is None:
                return
            if not self.box_teleport_client.service_is_ready():
                return
            # 第一次知道方块相对相机的位姿：把它换算成世界坐标并瞬移 turtle1。
            target = self.world_from_relative(
                self.camera_pose, self.initial_forward, self.initial_left
            )
            request = TeleportAbsolute.Request()
            request.x, request.y = target
            request.theta = 0.0
            self.teleport_target = target
            self.box_teleport_future = self.box_teleport_client.call_async(request)
            self.state = "teleporting"

        elif self.state == "teleporting":
            if not self.box_teleport_future.done():
                return
            error = self.box_teleport_future.exception()
            if error is not None:
                self.get_logger().error(f"方块瞬移失败，1 秒后重试: {error}")
                self.state = "wait_poses"
                self.retry_at = now + 1.0
                return
            self.state = "wait_teleport_pose"

        elif self.state == "wait_teleport_pose":
            # 等 /turtle1/pose 真正反映瞬移结果，避免第一帧检测用旧位置。
            if self.box_pose is None:
                return
            if math.hypot(
                self.box_pose.x - self.teleport_target[0],
                self.box_pose.y - self.teleport_target[1],
            ) > 0.1:
                return
            self.box_origin = (
                self.box_pose.x,
                self.box_pose.y,
                self.box_pose.theta,
            )
            self.ready_at = time.monotonic()
            self.ready = True
            self.state = "running"
            self.get_logger().info(
                "方块已瞬移到位，开始发布检测并由 cmd_vel 驱动 camera"
            )
            if self.box_motion == "sweep":
                self.get_logger().info(
                    "box_motion=sweep：turtle1 开始实时扫动，camera 持续跟踪"
                )

    def world_from_relative(self, pose: Pose, forward: float, left: float):
        cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
        return (
            pose.x + forward * cos_t - left * sin_t,
            pose.y + forward * sin_t + left * cos_t,
        )

    # -------------------------------------------------------------- 假检测发布
    def publish_tick(self):
        if not self.ready or self.camera_pose is None or self.box_pose is None:
            return

        stamp = self.get_clock().now().to_msg()
        relative = self.relative_pose(self.camera_pose, self.box_pose)
        x_m = relative[0] * self.scale
        z_m = relative[1] * self.scale

        detection = Detection3D()
        detection.header.stamp = stamp
        detection.header.frame_id = self.frame_id
        detection.id = self.target_id
        detection.bbox.center.position.x = x_m
        detection.bbox.center.position.y = 0.0
        detection.bbox.center.position.z = z_m
        half = (self.box_pose.theta - self.camera_pose.theta) / 2
        c, sn = math.cos(half), math.sin(half)
        q = detection.bbox.center.orientation
        q.x = q.w = 0.5 * (c - sn)
        q.y = -0.5 * (c + sn)
        q.z = 0.5 * (c + sn)
        detection.bbox.size.x = detection.bbox.size.y = detection.bbox.size.z = self.size_m

        detections = Detection3DArray()
        detections.header.stamp = stamp
        detections.header.frame_id = self.frame_id
        detections.detections.append(detection)
        self.detection_pub.publish(detections)

        # 二维仿真里相机水平安装：光学坐标 y 向下，UP=(0,-1,0)。
        up = Vector3Stamped()
        up.header.stamp = stamp
        up.header.frame_id = self.frame_id
        up.vector.x = 0.0
        up.vector.y = -1.0
        up.vector.z = 0.0
        self.up_pub.publish(up)

    @staticmethod
    def relative_pose(camera: Pose, box: Pose):
        """返回 (右向 X, 前向 Z)，单位为 turtlesim 世界单位。

        camera 朝向 theta 时，前向单位向量为 (cos, sin)，右向为 (sin, -cos)。
        控制器使用光学坐标：x 向右、z 向前，正好对应这里的两项。
        """
        dx = box.x - camera.x
        dy = box.y - camera.y
        cos_t, sin_t = math.cos(camera.theta), math.sin(camera.theta)
        right = dx * sin_t - dy * cos_t
        forward = dx * cos_t + dy * sin_t
        return right, forward

    # -------------------------------------------------------------- 运动执行
    def motion_tick(self):
        now = time.monotonic()
        dt = min(now - self.last_step, 0.05)
        self.last_step = now
        if not self.ready or self.cam_x is None:
            return

        # 控制器停止发布时，最多保持 cmd_timeout，防止最后一条速度一直生效。
        if self.cmd_at is None or now - self.cmd_at > self.cmd_timeout:
            vx, vy, wz = 0.0, 0.0, 0.0
        else:
            # m/s -> turtlesim 单位/s
            vx = self.cmd.linear.x / self.scale
            vy = self.cmd.linear.y / self.scale
            wz = self.cmd.angular.z

        cos_t, sin_t = math.cos(self.cam_theta), math.sin(self.cam_theta)
        dx = (vx * cos_t - vy * sin_t) * dt
        dy = (vx * sin_t + vy * cos_t) * dt
        self.cam_x = min(max(self.cam_x + dx, WORLD_MIN), WORLD_MAX)
        self.cam_y = min(max(self.cam_y + dy, WORLD_MIN), WORLD_MAX)

        self.cam_theta = math.atan2(math.sin(self.cam_theta + wz * dt),
                                    math.cos(self.cam_theta + wz * dt))

        # 上一次 teleport 还没完成时不重复排队；本地积分继续，服务空闲后会追到最新位姿。
        if (
            self.camera_teleport_future is not None
            and not self.camera_teleport_future.done()
        ):
            return
        if not self.camera_teleport_client.service_is_ready():
            return
        if self.camera_teleport_future is not None:
            error = self.camera_teleport_future.exception()
            self.camera_teleport_future = None
            if error is not None:
                self.get_logger().error(
                    f"camera teleport 失败，按真实位姿重新同步: {error}",
                    throttle_duration_sec=1.0,
                )
                if self.camera_pose is not None:
                    self.cam_x = self.camera_pose.x
                    self.cam_y = self.camera_pose.y
                    self.cam_theta = self.camera_pose.theta
                return

        # 速度为 0 时不重复发 teleport；at rest 的位姿已经由上一帧写入。
        if vx == 0.0 and vy == 0.0 and wz == 0.0:
            return

        request = TeleportAbsolute.Request()
        request.x = self.cam_x
        request.y = self.cam_y
        request.theta = self.cam_theta
        self.camera_teleport_future = self.camera_teleport_client.call_async(request)

    def box_motion_tick(self):
        """box_motion=sweep 时，让 turtle1 在相机前方做实时横向/前后扫动。

        方块在“相机初始朝向”的坐标系里运动，camera 同时响应平移和转向指令。
        运动幅度保持在相机前方，避免方块跑到 Z<=0 被控制器判为丢失。
        """
        if self.box_motion != "sweep" or not self.ready or self.box_origin is None:
            return

        future = self.box_motion_future
        if future is not None and not future.done():
            return
        if future is not None:
            error = future.exception()
            self.box_motion_future = None
            if error is not None:
                self.get_logger().error(
                    f"方块扫动 teleport 失败: {error}",
                    throttle_duration_sec=1.0,
                )
                return
        if not self.box_teleport_client.service_is_ready():
            return

        elapsed = time.monotonic() - self.ready_at
        phase = 2.0 * math.pi * elapsed / self.box_sweep_period
        # 相机初始朝向定义前方 f 和左侧 l；方块围绕瞬移后的位置做椭圆运动。
        # 横向用 sin、前后用 cos，使横向对中和距离调整交替出现。
        cos_t, sin_t = math.cos(self.cam_theta), math.sin(self.cam_theta)
        forward = self.box_sweep_forward * math.cos(phase)
        left = self.box_sweep_lateral * math.sin(phase)
        x = self.box_origin[0] + forward * cos_t - left * sin_t
        y = self.box_origin[1] + forward * sin_t + left * cos_t

        request = TeleportAbsolute.Request()
        request.x = min(max(x, WORLD_MIN), WORLD_MAX)
        request.y = min(max(y, WORLD_MIN), WORLD_MAX)
        request.theta = self.box_origin[2]
        self.box_motion_future = self.box_teleport_client.call_async(request)


def main(args=None):
    rclpy.init(args=args)
    node = TurtlesimPoseSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # SIGINT 到来时 rclpy 可能已经关闭 context，此时再 shutdown 会抛 RCLError。
        # 清理阶段的失败不应让 launch 把正常演示退出报告成错误。
        try:
            node.destroy_node()
        except (KeyboardInterrupt, Exception):
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except (KeyboardInterrupt, Exception):
                pass


if __name__ == "__main__":
    main()
