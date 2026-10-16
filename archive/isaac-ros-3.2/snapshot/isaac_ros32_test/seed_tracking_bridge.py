#!/usr/bin/env python3
"""Feed Isaac ROS tracking from recorded RGB-D, seeded by the original FP pose."""

from collections import OrderedDict
import argparse
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import rclpy
from isaac_ros_tensor_list_interfaces.msg import Tensor, TensorList, TensorShape
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, UInt32
from vision_msgs.msg import Detection3DArray
from rclpy.qos import qos_profile_sensor_data


POSE_FILE = Path(
    "/workspace/6Dpose/outputs/zed_foundationpose/ob_in_cam/000000.txt"
)
POSE_DIR = POSE_FILE.parent
FEEDBACK_TIMEOUT_SECONDS = 10.0
# 正常帧约 0.3 秒返回；只在明显停顿时补送，避免旧副本挤占 GXF 的单帧缓冲。
KICK_INTERVAL_SECONDS = 2.0
# 默认只送一次：补送副本会滞留在流水线，干扰下一帧与位姿的对应关系。
MAX_KICK_COPIES = 0


def load_seed_pose(pose_file=POSE_FILE):
    # 原版保存的是物体坐标系到相机光学坐标系的 4×4 变换，单位为米。
    matrix = np.loadtxt(pose_file, dtype=np.float32)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"Invalid 4x4 pose in {pose_file}")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-4):
        raise ValueError("Pose homogeneous bottom row is invalid")
    if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-2):
        raise ValueError("Pose rotation is not orthogonal")
    return matrix


def make_pose_message(matrix):
    pose = TensorList()
    tensor = Tensor()
    tensor.name = "poses"
    tensor.shape = TensorShape(rank=3, dims=[1, 4, 4])
    tensor.data_type = 9  # Isaac ROS Tensor.msg: 9 = float32
    tensor.strides = [64, 16, 4]
    # GXF FoundationPoseDecoder 将 Eigen::Matrix4f.data() 原样写入张量：列优先。
    tensor.data = list(np.asarray(matrix, dtype="<f4").tobytes(order="F"))
    pose.tensors = [tensor]
    return pose


def pose_from_detection(detection):
    """Recover the object's camera-frame transform from a Detection3D result."""
    # 这个箱体网格以物体中心为原点，因此检测框中心就是 tracker 输出的物体位姿。
    center = detection.bbox.center
    q = center.orientation
    quaternion = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    translation = np.array([center.position.x, center.position.y,
                            center.position.z], dtype=np.float64)
    if norm < 1e-8 or not np.all(np.isfinite(quaternion)) or not np.all(np.isfinite(translation)):
        raise ValueError("Detection3D has an invalid center pose")
    x, y, z, w = quaternion / norm
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = [
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ]
    matrix[:3, 3] = translation
    return make_pose_message(matrix)


class SeedTrackingBridge(Node):
    def __init__(self, max_kick_copies=MAX_KICK_COPIES):
        super().__init__("isaacfp_seed_tracking_bridge")
        self.max_kick_copies = max_kick_copies
        self.seed = make_pose_message(load_seed_pose())
        self.current_pose = self.seed
        self.pending = OrderedDict()
        self.frame_count = 0
        self.dropped_input_frames = 0
        self.pose_update_count = 0
        self.latest_pose_origin = None
        self.sent_stamps = OrderedDict()
        self.frame_headers = OrderedDict()
        self.forwarded_detection_origins = set()
        self.inflight_stamp = None
        self.inflight_since = None
        self.inflight_frame = None
        self.inflight_pose = None
        self.last_kick_at = None
        self.kick_count = 0
        self.ready_logged = False
        self.next_wait_log = 0.0
        self.next_feedback_warning_at = 0.0
        self.rgb_pub = self.create_publisher(Image, "/tracking/image", 10)
        self.depth_pub = self.create_publisher(Image, "/tracking/depth_image", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/tracking/camera_info", 10)
        self.pose_pub = self.create_publisher(TensorList, "/tracking/pose_input", 10)
        # 轻量反馈：播放器等位姿矩阵返回，再发送下一张录像帧。
        self.pose_ready_pub = self.create_publisher(Header, "/isaacfp/tracking_pose_ready", 10)
        self.recovery_ready_pub = self.create_publisher(UInt32, "/isaacfp/recovery_ready", 10)
        self.debug_output_pub = self.create_publisher(
            Detection3DArray, "/isaacfp/tracking_debug_output", 10)
        self.create_subscription(Image, "/isaacfp/rgb", self.on_rgb, 10)
        self.create_subscription(Image, "/isaacfp/depth", self.on_depth, 10)
        self.create_subscription(CameraInfo, "/isaacfp/camera_info", self.on_info, 10)
        # 兼容现有回放脚本的四路订阅检查；tracker 本身不需要 mask。
        self.create_subscription(Image, "/isaacfp/mask", lambda _: None, 10)
        self.create_subscription(TensorList, "/tracking/pose_matrix_output",
                                 self.on_pose_update, 10)
        self.create_subscription(Detection3DArray, "/tracking/output",
                                 self.on_tracking_output, qos_profile_sensor_data)
        self.create_subscription(UInt32, "/isaacfp/recovery_request",
                                 self.on_recovery_request, 10)
        # 输入回调负责立即转发；定时器只作启动发现和等待告警兜底。
        self.create_timer(0.05, self.flush_pending)
        self.get_logger().info(f"Loaded original first-frame pose from {POSE_FILE}")
        self.get_logger().info(
            f"Same-frame extra copies={self.max_kick_copies}; 0 means each frame is sent once")

    @staticmethod
    def key(msg):
        return (msg.header.stamp.sec, msg.header.stamp.nanosec)

    def remember(self, msg, kind):
        key = self.key(msg)
        self.pending.setdefault(key, {})[kind] = msg
        while len(self.pending) > 3:
            self.pending.popitem(last=False)
            self.dropped_input_frames += 1
        # 三路到齐即转发，去掉每帧最多 50 ms 的人为定时等待。
        if {"rgb", "depth", "info"}.issubset(self.pending.get(key, {})):
            self.flush_pending()

    def on_rgb(self, msg):
        self.remember(msg, "rgb")

    def on_depth(self, msg):
        self.remember(msg, "depth")

    def on_info(self, msg):
        self.remember(msg, "info")

    def on_pose_update(self, msg):
        if not msg.tensors or msg.tensors[0].name != "poses":
            self.get_logger().error("Tracker returned an unexpected pose tensor")
            return
        stamp = self.key(msg)
        if stamp not in self.sent_stamps:
            self.get_logger().warning(
                f"Ignoring pose update not sent by this bridge stamp={stamp}")
            return
        sent_at, origin = self.sent_stamps[stamp]
        self.accept_pose_update(msg, origin, stamp, sent_at, "pose_matrix")

    def accept_pose_update(self, msg, origin, stamp, sent_at, source):
        if self.latest_pose_origin is not None and origin <= self.latest_pose_origin:
            self.get_logger().warning(
                f"Ignoring duplicate or stale tracker pose update stamp={stamp}, origin={origin}, "
                f"source={source}")
            return
        self.current_pose = msg
        self.latest_pose_origin = origin
        latency_ms = (time.monotonic() - sent_at) * 1000
        if origin == self.inflight_stamp:
            # ACK 使用原始录像帧时间戳；补送帧只负责推动 GXF 流水线。
            self.pose_ready_pub.publish(self.inflight_frame["rgb"].header)
            self.inflight_stamp = None
            self.inflight_since = None
            self.inflight_frame = None
            self.inflight_pose = None
            self.last_kick_at = None
            self.kick_count = 0
        else:
            # 旧帧反馈只能更新位姿；不能确认当前正在处理的录像帧。
            self.get_logger().warning(
                f"Accepted late tracker pose stamp={stamp}, origin={origin}, "
                f"active={self.inflight_stamp}, latency={latency_ms:.0f} ms")
        self.pose_update_count += 1
        if self.pose_update_count == 1 or self.pose_update_count % 5 == 0:
            self.get_logger().info(
                f"Received {self.pose_update_count} tracker pose updates; "
                f"feedback latency={latency_ms:.0f} ms; source={source}")

    def on_tracking_output(self, msg):
        stamp = self.key(msg)
        sent = self.sent_stamps.get(stamp)
        if sent is None or not msg.detections:
            return
        sent_at, origin = sent
        if origin in self.forwarded_detection_origins:
            return
        original_header = self.frame_headers.get(origin)
        if original_header is None:
            return
        # 完成回执必须同时代表：位姿已更新、当前帧非空检测已取得。
        # 先完成位姿更新，再发布检测；播放器无需等两条独立反馈互相到齐。
        if origin == self.inflight_stamp:
            try:
                pose = pose_from_detection(msg.detections[0])
            except ValueError as error:
                self.get_logger().error(f"Cannot use detection as tracker pose feedback: {error}")
                return
            pose.header = deepcopy(original_header)
            self.accept_pose_update(pose, origin, stamp, sent_at, "detection_fallback")
        elif self.latest_pose_origin != origin:
            # 旧帧或尚未完成位姿更新的结果不能确认当前录像帧完成。
            return
        result = deepcopy(msg)
        result.header = deepcopy(original_header)
        self.debug_output_pub.publish(result)
        self.forwarded_detection_origins.add(origin)
        self.get_logger().debug(
            f"Completed tracker frame source={stamp} original={origin}; "
            f"latency={(time.monotonic() - sent_at) * 1000:.0f} ms")

    def on_recovery_request(self, msg):
        if self.inflight_stamp is not None:
            self.get_logger().error("Recovery requested while a tracker frame is still in flight")
            return
        frame_index = int(msg.data)
        pose_file = POSE_DIR / f"{frame_index:06d}.txt"
        try:
            pose = make_pose_message(load_seed_pose(pose_file))
        except (OSError, ValueError) as error:
            self.get_logger().error(f"Cannot load offline recovery pose: {error}")
            return
        # 仅供离线验证：下一帧使用原版保存的真实位姿作种子，不是 Isaac 重新定位。
        self.current_pose = pose
        self.recovery_ready_pub.publish(UInt32(data=frame_index))
        self.get_logger().warning(
            f"ORIGINAL_POSE_RESEED next_recorded_frame={frame_index + 1} file={pose_file}")

    def tracker_is_ready(self):
        counts = [pub.get_subscription_count() for pub in
                  (self.rgb_pub, self.depth_pub, self.info_pub, self.pose_pub)]
        if min(counts) > 0:
            if not self.ready_logged:
                self.get_logger().info(
                    f"Tracker subscribers image/depth/info/pose={tuple(counts)}")
                self.ready_logged = True
            return True
        if self.pending and self.get_clock().now().nanoseconds >= self.next_wait_log:
            self.get_logger().warning(
                f"Waiting for tracker subscribers image/depth/info/pose={tuple(counts)}")
            self.next_wait_log = self.get_clock().now().nanoseconds + 5_000_000_000
        return False

    def flush_pending(self):
        if self.inflight_stamp is not None:
            # 不在上一帧反馈到来前发送下一帧，否则会复用旧姿态。
            if (self.kick_count < self.max_kick_copies and
                    time.monotonic() - self.last_kick_at >= KICK_INTERVAL_SECONDS):
                self.send_kick_copy()
                return
            if time.monotonic() >= self.next_feedback_warning_at:
                self.get_logger().warning(
                    f"Still waiting for tracker pose stamp={self.inflight_stamp}; "
                    f"elapsed={time.monotonic() - self.inflight_since:.0f}s; "
                    "not forwarding another recorded frame")
                self.next_feedback_warning_at = time.monotonic() + FEEDBACK_TIMEOUT_SECONDS
            return
        if not self.tracker_is_ready():
            return
        # 实时流可能在等待反馈时积压；只取最新的完整 RGB-D 帧。
        complete = [(key, frame) for key, frame in self.pending.items()
                    if {"rgb", "depth", "info"}.issubset(frame)]
        if not complete:
            return
        key, frame = max(complete, key=lambda item: item[0])
        self.dropped_input_frames += sum(pending_key < key for pending_key in self.pending)
        self.pending = OrderedDict(
            (pending_key, pending_frame) for pending_key, pending_frame in self.pending.items()
            if pending_key > key)
        # 数值使用上一帧位姿；时间戳必须改成当前 RGB-D 帧的时间戳。
        # GXF 的 tracking 输出沿用 pose_input 的时间戳，不改会让画框匹配失败。
        pose_for_frame = deepcopy(self.current_pose)
        pose_for_frame.header = frame["rgb"].header
        self.inflight_stamp = key
        self.inflight_since = time.monotonic()
        self.next_feedback_warning_at = self.inflight_since + FEEDBACK_TIMEOUT_SECONDS
        self.inflight_frame = frame
        self.inflight_pose = pose_for_frame
        self.last_kick_at = self.inflight_since
        self.kick_count = 0
        self.sent_stamps[key] = (self.inflight_since, key)
        self.frame_headers[key] = deepcopy(frame["rgb"].header)
        if len(self.sent_stamps) > 1200:
            self.sent_stamps.popitem(last=False)
        if len(self.frame_headers) > 1200:
            old_origin, _ = self.frame_headers.popitem(last=False)
            self.forwarded_detection_origins.discard(old_origin)
        self.rgb_pub.publish(frame["rgb"])
        self.info_pub.publish(frame["info"])
        self.depth_pub.publish(frame["depth"])
        self.pose_pub.publish(pose_for_frame)
        self.frame_count += 1
        if self.frame_count == 1 or self.frame_count % 5 == 0:
            self.get_logger().info(
                f"Forwarded {self.frame_count} RGB-D frames to tracker; "
                f"pose updates={self.pose_update_count}; "
                f"dropped input frames={self.dropped_input_frames}; "
                f"input stamp={pose_for_frame.header.stamp.sec}."
                f"{pose_for_frame.header.stamp.nanosec:09d}; "
                f"pose tensor bytes={len(pose_for_frame.tensors[0].data)}")

    def send_kick_copy(self):
        # Isaac GXF 输出在下一批输入到来时才可能刷新；同帧副本不推进录像内容。
        stamp = self.get_clock().now().to_msg()
        copy_key = (stamp.sec, stamp.nanosec)
        rgb = deepcopy(self.inflight_frame["rgb"])
        depth = deepcopy(self.inflight_frame["depth"])
        info = deepcopy(self.inflight_frame["info"])
        pose = deepcopy(self.inflight_pose)
        for message in (rgb, depth, info, pose):
            message.header.stamp = stamp
        self.sent_stamps[copy_key] = (time.monotonic(), self.inflight_stamp)
        self.rgb_pub.publish(rgb)
        self.info_pub.publish(info)
        self.depth_pub.publish(depth)
        self.pose_pub.publish(pose)
        self.kick_count += 1
        self.last_kick_at = time.monotonic()
        self.get_logger().info(
            f"Sent same-frame GXF kick copy {self.kick_count}/{self.max_kick_copies}; "
            f"origin={self.inflight_stamp}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kick-copies", type=int, default=MAX_KICK_COPIES,
                        help="diagnostic only: extra same-frame copies; default 0")
    args, ros_args = parser.parse_known_args()
    if args.kick_copies < 0:
        parser.error("--kick-copies must be nonnegative")
    rclpy.init(args=ros_args)
    bridge = SeedTrackingBridge(args.kick_copies)
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
