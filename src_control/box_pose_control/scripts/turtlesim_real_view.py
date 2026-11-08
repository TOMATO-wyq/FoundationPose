#!/usr/bin/env python3
"""真实 Box4D 位姿的 turtlesim 只读可视化。

这里的终端分工是：
  1. 相机驱动
  2. Box4D 识别（--publish-poses）
  3. box_pose_control 控制节点
  4. 本节点 + turtlesim（只读显示）

本节点只订阅：
  /box4d/detections、/box4d/up_camera、/box_pose_control/cmd_vel
不发布任何 /box4d/* 话题，因此不会改变真实控制链路，也不会和控制节点抢目标。

显示方式：
  * turtle1：真实方块，按检测到的“方块相对相机”位置放到 camera 海龟附近；
  * camera：相机/底盘，按真实 cmd_vel 做 2D 平移，朝向固定。
turtlesim 是差速模型，不能直接接收 linear.y，所以 camera 用 teleport_absolute
以 60 Hz 更新位姿；控制器仍然只做全向平移、不转向，和真实假设一致。
"""

import math
import time

import rclpy
from geometry_msgs.msg import Twist, Vector3Stamped
from rclpy.node import Node
from turtlesim.msg import Pose
from turtlesim.srv import Spawn, TeleportAbsolute
from vision_msgs.msg import Detection3DArray

WORLD_MIN = 0.6
WORLD_MAX = 10.4


class TurtlesimRealView(Node):
    def __init__(self):
        super().__init__("turtlesim_real_view")

        self.camera_name = self.declare_parameter("camera_name", "camera").value
        self.box_name = self.declare_parameter("box_name", "turtle1").value
        self.spawn_x = float(self.declare_parameter("spawn_x", 1.0).value)
        self.spawn_y = float(self.declare_parameter("spawn_y", 5.5).value)
        self.spawn_theta = float(self.declare_parameter("spawn_theta", 0.0).value)
        # 1 个 turtlesim 世界单位对应多少米。真实方块常在 0.5~1.5 m，
        # 默认 0.2 m/unit 可以让 1.3 m 在 11x11 的世界里显示得下。
        self.scale = float(self.declare_parameter("meters_per_unit", 0.2).value)
        self.cmd_timeout = float(self.declare_parameter("cmd_timeout_s", 0.3).value)
        self.motion_rate = float(self.declare_parameter("motion_rate", 60.0).value)
        # 留空时自动选最接近光轴的目标并锁定；多目标时填控制节点 INFO 里的 id。
        self.target_id = str(self.declare_parameter("target_id", "").value)
        size = self.declare_parameter("size", "big").value
        if size not in ("big", "small"):
            raise ValueError("size must be big or small")
        self.target_size_m = 0.35 if size == "big" else 0.20

        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("meters_per_unit must be a positive finite number")
        if not math.isfinite(self.cmd_timeout) or self.cmd_timeout <= 0.0:
            raise ValueError("cmd_timeout_s must be a positive finite number")
        if not math.isfinite(self.motion_rate) or self.motion_rate <= 0.0:
            raise ValueError("motion_rate must be a positive finite number")

        self.camera_pose = None
        self.box_pose = None
        self.cam_x = None
        self.cam_y = None
        self.cam_theta = None
        self.ready = False
        self.state = "wait_spawn"
        self.spawn_future = None
        self.camera_teleport_future = None
        self.box_teleport_future = None
        self.retry_at = 0.0
        self.last_step = time.monotonic()

        self.up = (0.0, -1.0, 0.0)  # 正常安装的默认 UP；收到真实 UP 后覆盖
        self.up_frame = ""
        self.up_received = False
        self.locked_id = self.target_id
        self.last_target = None  # (X, Y, Z) in camera optical frame, m
        self.last_detection_at = None

        self.cmd = Twist()
        self.cmd_at = None

        self.create_subscription(
            Detection3DArray, "/box4d/detections", self.on_detections, 10
        )
        self.create_subscription(
            Vector3Stamped, "/box4d/up_camera", self.on_up, 10
        )
        self.create_subscription(
            Twist, "/box_pose_control/cmd_vel", self.on_cmd, 10
        )
        self.create_subscription(
            Pose, f"/{self.camera_name}/pose", self.on_camera_pose, 10
        )
        self.create_subscription(
            Pose, f"/{self.box_name}/pose", self.on_box_pose, 10
        )

        self.spawn_client = self.create_client(Spawn, "/spawn")
        self.camera_teleport_client = self.create_client(
            TeleportAbsolute, f"/{self.camera_name}/teleport_absolute"
        )
        self.box_teleport_client = self.create_client(
            TeleportAbsolute, f"/{self.box_name}/teleport_absolute"
        )

        self.create_timer(0.05, self.startup_tick)
        self.create_timer(1.0 / self.motion_rate, self.motion_tick)

        self.get_logger().info(
            f"只读可视化启动：等待真实 /box4d/detections、/box4d/up_camera 和 "
            f"/box_pose_control/cmd_vel（比例 {self.scale} m/unit）"
        )

    # ------------------------------------------------------------------ 真实输入
    def on_up(self, message: Vector3Stamped):
        v = message.vector
        norm = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        if (
            math.isfinite(norm)
            and norm > 1e-9
            and math.isfinite(v.x)
            and math.isfinite(v.y)
            and math.isfinite(v.z)
        ):
            self.up = (v.x / norm, v.y / norm, v.z / norm)
            self.up_frame = message.header.frame_id
            self.up_received = True
        else:
            self.get_logger().warn(
                "忽略无效 UP，继续使用上一帧", throttle_duration_sec=2.0
            )

    def on_detections(self, message: Detection3DArray):
        target = self.select_target(message)
        if target is None:
            self.get_logger().warn(
                f"没有找到目标 id={self.locked_id or '<auto>'}",
                throttle_duration_sec=2.0,
            )
            return

        self.locked_id = target.id
        p = target.bbox.center.position
        self.last_target = (p.x, p.y, p.z)
        self.last_detection_at = time.monotonic()
        if self.ready:
            self.place_box()

    def select_target(self, message: Detection3DArray):
        if not message.detections:
            return None
        best = None
        best_score = float("inf")
        for detection in message.detections:
            size = detection.bbox.size
            if any(not math.isfinite(s) or abs(s - self.target_size_m) > 0.01
                   for s in (size.x, size.y, size.z)):
                continue
            p = detection.bbox.center.position
            q = detection.bbox.center.orientation
            qnorm = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
            if (
                not detection.id
                or not math.isfinite(p.x)
                or not math.isfinite(p.y)
                or not math.isfinite(p.z)
                or p.z <= 0.0
                or not math.isfinite(qnorm)
                or qnorm < 1e-12
            ):
                continue
            if self.locked_id:
                if detection.id == self.locked_id:
                    return detection
                continue
            score = (p.x / p.z) ** 2 + (p.y / p.z) ** 2
            if score < best_score:
                best_score = score
                best = detection
        return best

    def on_cmd(self, message: Twist):
        self.cmd = message
        self.cmd_at = time.monotonic()

    def on_camera_pose(self, message: Pose):
        self.camera_pose = message
        if self.cam_x is None:
            self.cam_x = message.x
            self.cam_y = message.y
            self.cam_theta = message.theta
            return
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

    def on_box_pose(self, message: Pose):
        self.box_pose = message

    # -------------------------------------------------------------- 启动/放置方块
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
                self.get_logger().warn(
                    f"/spawn 调用失败，假定 {self.camera_name} 已存在: {error}"
                )
            self.state = "wait_poses"

        elif self.state == "wait_poses":
            if self.camera_pose is None or self.box_pose is None:
                return
            self.ready = True
            self.state = "running"
            self.get_logger().info(
                "camera 海龟就绪；收到真实检测后开始放置 turtle1"
            )

    def place_box(self):
        if self.camera_pose is None or self.last_target is None:
            return
        if not self.up_received:
            # 倒装/正装的左右映射依赖真实 UP；先等一帧 UP，避免方块先跳到错误一侧。
            self.get_logger().warn(
                "等待 /box4d/up_camera，暂不放置 turtle1",
                throttle_duration_sec=2.0,
            )
            return
        if (
            self.box_teleport_future is not None
            and not self.box_teleport_future.done()
        ):
            return
        if not self.box_teleport_client.service_is_ready():
            return

        forward, left = self.relative_in_robot_frame(*self.last_target)
        local_x = forward / self.scale
        local_y = left / self.scale
        cos_t = math.cos(self.camera_pose.theta)
        sin_t = math.sin(self.camera_pose.theta)
        x = self.camera_pose.x + local_x * cos_t - local_y * sin_t
        y = self.camera_pose.y + local_x * sin_t + local_y * cos_t

        clamped = x < WORLD_MIN or x > WORLD_MAX or y < WORLD_MIN or y > WORLD_MAX
        x = min(max(x, WORLD_MIN), WORLD_MAX)
        y = min(max(y, WORLD_MIN), WORLD_MAX)
        if clamped:
            self.get_logger().warn(
                "方块超出 turtlesim 世界，已裁剪；可调大 meters_per_unit 或 spawn 位置",
                throttle_duration_sec=5.0,
            )

        request = TeleportAbsolute.Request()
        request.x = x
        request.y = y
        request.theta = 0.0
        self.box_teleport_future = self.box_teleport_client.call_async(request)

    def relative_in_robot_frame(self, x, y, z):
        """把光学系的 (X,Y,Z) 投到相机水平面，返回 (forward, left)。

        与控制器一致：u=UP，f 为光轴水平投影，l=u×f。
        正常安装 u≈(0,-1,0) -> l≈-x_optical；
        倒装 u≈(0,+1,0) -> l≈+x_optical，左右会自动正确。
        完整光学位置投影到水平面，消除沿 UP 的高度。
        """
        ux, uy, uz = self.up
        fx, fy, fz = -ux * uz, -uy * uz, 1.0 - uz * uz
        norm = math.sqrt(fx * fx + fy * fy + fz * fz)
        if norm < 1e-9:
            fx, fy, fz = 0.0, 0.0, 1.0
        else:
            fx, fy, fz = fx / norm, fy / norm, fz / norm
        # l = u × f
        lx = uy * fz - uz * fy
        ly = uz * fx - ux * fz
        lz = ux * fy - uy * fx
        forward = fx * x + fy * y + fz * z
        left = lx * x + ly * y + lz * z
        return forward, left

    # -------------------------------------------------------------- camera 运动
    def motion_tick(self):
        now = time.monotonic()
        dt = min(now - self.last_step, 0.05)
        self.last_step = now
        if not self.ready or self.cam_x is None:
            return

        if self.cmd_at is None or now - self.cmd_at > self.cmd_timeout:
            vx, vy = 0.0, 0.0
        else:
            vx = self.cmd.linear.x / self.scale
            vy = self.cmd.linear.y / self.scale

        cos_t, sin_t = math.cos(self.cam_theta), math.sin(self.cam_theta)
        self.cam_x = min(
            max(self.cam_x + (vx * cos_t - vy * sin_t) * dt, WORLD_MIN),
            WORLD_MAX,
        )
        self.cam_y = min(
            max(self.cam_y + (vx * sin_t + vy * cos_t) * dt, WORLD_MIN),
            WORLD_MAX,
        )

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

        if vx == 0.0 and vy == 0.0:
            return

        request = TeleportAbsolute.Request()
        request.x = self.cam_x
        request.y = self.cam_y
        request.theta = self.cam_theta
        self.camera_teleport_future = self.camera_teleport_client.call_async(request)


def main(args=None):
    rclpy.init(args=args)
    node = TurtlesimRealView()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
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
