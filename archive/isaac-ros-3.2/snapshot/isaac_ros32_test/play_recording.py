#!/usr/bin/env python3
"""Replay RGB-D frames exported from recording.svo2 into IsaacFP ROS topics."""

import argparse
import csv
import time
from collections import OrderedDict, deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, UInt32

try:
    from vision_msgs.msg import Detection3DArray
except ImportError:
    Detection3DArray = None


DEFAULT_SEQUENCE = Path("/home/tomato/6Dpose/outputs/zed_sequence")
DEFAULT_MASK = Path("/home/tomato/6Dpose/outputs/zed_foundationpose/initial_mask.png")
PREVIEW_WINDOW = "IsaacFP recording debug (Q to stop)"
POSE_WINDOW = "IsaacFP estimated 3D pose"


class RollingFPS:
    """最近两秒的有效结果帧率；等待 ROS/推理结果时会自动下降。"""

    def __init__(self):
        self.started = time.monotonic()
        self.completed = deque()

    def mark(self):
        self.completed.append(time.monotonic())

    def value(self):
        now = time.monotonic()
        while self.completed and self.completed[0] <= now - 2.0:
            self.completed.popleft()
        return len(self.completed) / max(min(now - self.started, 2.0), 0.1)


def draw_fps(image, fps):
    label = f"FPS: {fps:.1f}"
    size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = max(4, image.shape[1] - size[0] - 12)
    cv2.rectangle(image, (x - 4, 4), (image.shape[1] - 4, 14 + size[1] + baseline),
                  (0, 0, 0), -1)
    cv2.putText(image, label, (x, 10 + size[1]), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return image


def box_pose(bbox):
    """Return a box center and rotation in the camera optical frame."""
    center = bbox.center.position
    q = bbox.center.orientation
    quaternion = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Invalid pose quaternion")
    x, y, z, w = quaternion / norm
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    origin = np.array([center.x, center.y, center.z], dtype=np.float64)
    return origin, rotation


def project_box_corners(bbox, k):
    """Project a vision_msgs BoundingBox3D in the camera optical frame."""
    # bbox 给出相机坐标系中的中心、姿态和长宽高；先恢复 8 个三维角点。
    origin, rotation = box_pose(bbox)
    size = np.array([bbox.size.x, bbox.size.y, bbox.size.z], dtype=np.float64)
    if not np.all(np.isfinite(size)) or np.any(size <= 0):
        raise ValueError(f"Invalid 3D box size: {size}")
    signs = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=np.float64)
    corners = (signs * (size / 2)) @ rotation.T
    corners += origin
    if np.any(corners[:, 2] <= 1e-3):
        raise ValueError("3D box crosses or lies behind the camera")
    # 用相机内参 K 投影到 RGB 图像；深度小于零的角点不能投影。
    projected = (k @ corners.T).T
    uv = projected[:, :2] / projected[:, 2:3]
    if not np.all(np.isfinite(uv)):
        raise ValueError("Projected box contains non-finite coordinates")
    return np.rint(np.clip(uv, -10000, 10000)).astype(np.int32)


def draw_pose_box(image, bbox, k):
    uv = project_box_corners(bbox, k)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)):
        cv2.line(image, tuple(uv[a]), tuple(uv[b]), (0, 255, 0), 2, cv2.LINE_AA)
    origin, rotation = box_pose(bbox)
    size = np.array([bbox.size.x, bbox.size.y, bbox.size.z], dtype=np.float64)
    # 物体坐标轴从位姿中心出发；OpenCV 使用 BGR，所以 X 红、Y 绿、Z 蓝。
    axis_length = 0.4 * np.max(size)
    axes = np.vstack([origin, origin + rotation.T * axis_length])
    if np.any(axes[:, 2] <= 1e-3):
        return image
    projected = (k @ axes.T).T
    points = np.rint(projected[:, :2] / projected[:, 2:3]).astype(np.int32)
    for tip, color, label in ((1, (0, 0, 255), "X"),
                              (2, (0, 255, 0), "Y"),
                              (3, (255, 0, 0), "Z")):
        cv2.arrowedLine(image, tuple(points[0]), tuple(points[tip]), color, 2,
                        cv2.LINE_AA, tipLength=0.15)
        cv2.putText(image, label, tuple(points[tip]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    return image


def purple_box_mask(rgb):
    """Segment this recording's purple box independently of the tracked pose."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    candidate = cv2.inRange(hsv, (115, 80, 25), (145, 255, 255))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    if count <= 1:
        return None
    target = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[target, cv2.CC_STAT_AREA] < 0.005 * rgb.shape[0] * rgb.shape[1]:
        return None
    return np.asarray(labels == target, dtype=np.uint8)


def color_coverage(bbox, k, object_mask):
    """Fraction of independently segmented object pixels inside the tracked 3D box."""
    try:
        corners = project_box_corners(bbox, k)
    except ValueError:
        return 0.0
    projected = np.zeros(object_mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(projected, cv2.convexHull(corners), 1)
    return float(np.count_nonzero(projected & object_mask) /
                 max(1, np.count_nonzero(object_mask)))


def read_assets(sequence, mask_path, scale):
    # recording.svo2 已提前导出为逐帧 PNG 和相机内参，这里不需要启动 ZED SDK。
    with (sequence / "frames.csv").open(newline="") as stream:
        frames = list(csv.DictReader(stream))
    if not frames:
        raise ValueError("frames.csv has no frames")
    k = np.loadtxt(sequence / "cam_K.txt", dtype=np.float64).reshape(3, 3)
    rgb0 = cv2.imread(str(sequence / "rgb" / frames[0]["rgb_file"]), cv2.IMREAD_UNCHANGED)
    depth0 = cv2.imread(str(sequence / "depth" / frames[0]["depth_file"]), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if rgb0 is None or depth0 is None or mask is None:
        raise FileNotFoundError("First RGB, depth or mask file is missing")
    if rgb0.shape[:2] != depth0.shape[:2] or rgb0.shape[:2] != mask.shape[:2]:
        raise ValueError("RGB, depth and mask dimensions differ")
    if depth0.dtype != np.uint16:
        raise ValueError(f"Expected uint16 millimeter depth, got {depth0.dtype}")
    width = round(rgb0.shape[1] * scale)
    height = round(rgb0.shape[0] * scale)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(mask):
        raise ValueError("Initial mask is empty")
    k[0, :] *= width / rgb0.shape[1]
    k[1, :] *= height / rgb0.shape[0]
    # 图像缩小后，焦距和主点也必须按同样比例缩放。
    return frames, k, mask, width, height


class RecordingPlayer(Node):
    def __init__(self, k, mask, width, height, direct_pose=False, preview_enabled=True,
                 tracker_test=False):
        super().__init__("isaacfp_recording_player")
        self.bridge = CvBridge()
        self.direct_pose = direct_pose
        self.preview_enabled = preview_enabled
        self.tracker_test = tracker_test
        self.k = k
        self.mask = mask
        self.width = width
        self.height = height
        self.first_pose_count = 0
        self.tracking_count = 0
        self.pose_output_count = 0
        self.pose_result_stamps = OrderedDict()
        self.tracking_output_count = 0
        self.tracking_result_stamps = OrderedDict()
        self.tracking_pose_ready_stamps = OrderedDict()
        self.recovery_ready_indices = set()
        self.object_masks_by_stamp = OrderedDict()
        self.tracking_coverage_by_stamp = OrderedDict()
        self.frame_index_by_stamp = OrderedDict()
        self.matched_tracking_frames = set()
        self.frame_cache = OrderedDict()
        self.fps_meter = RollingFPS()
        self.fps_seen_results = set()
        self.last_pose_view = None
        self.pose_view_dirty = False
        self.last_pose_fps_label = None
        self.next_preview_draw = 0.0
        self.next_preview_subscriber_check = 0.0
        self.preview_subscriber_counts = None
        topics = (
            ("/pose_estimation/image", "/pose_estimation/depth_image",
             "/pose_estimation/camera_info", "/pose_estimation/segmentation")
            if direct_pose else
            ("/isaacfp/rgb", "/isaacfp/depth", "/isaacfp/camera_info", "/isaacfp/mask")
        )
        self.rgb_pub = self.create_publisher(Image, topics[0], 10)
        self.depth_pub = self.create_publisher(Image, topics[1], 10)
        self.info_pub = self.create_publisher(CameraInfo, topics[2], 10)
        self.mask_pub = self.create_publisher(Image, topics[3], 10)
        if not direct_pose:
            self.pose_sub = self.create_subscription(
                CameraInfo, "/pose_estimation/camera_info", self.on_first_pose,
                qos_profile_sensor_data)
            self.tracking_sub = self.create_subscription(
                CameraInfo, "/tracking/camera_info", self.on_tracking,
                qos_profile_sensor_data)
            self.tracking_pose_ready_sub = self.create_subscription(
                Header, "/isaacfp/tracking_pose_ready", self.on_tracking_pose_ready, 10)
            self.recovery_pub = self.create_publisher(UInt32, "/isaacfp/recovery_request", 10)
            self.recovery_ready_sub = self.create_subscription(
                UInt32, "/isaacfp/recovery_ready", self.on_recovery_ready, 10)
        if Detection3DArray is not None:
            self.pose_output_sub = self.create_subscription(
                Detection3DArray, "/pose_estimation/output",
                lambda msg: self.on_pose_output(msg, "pose"), qos_profile_sensor_data)
            if not direct_pose:
                self.tracking_output_sub = self.create_subscription(
                    Detection3DArray,
                    "/isaacfp/tracking_debug_output" if tracker_test else "/tracking/output",
                    lambda msg: self.on_pose_output(msg, "tracking"),
                    # 自写桥接采用可靠发布，逐帧完成回执也要可靠接收。
                    # 官方直接输出保留原 QoS，避免与其发布端不兼容。
                    10 if tracker_test else qos_profile_sensor_data)
        else:
            self.get_logger().warning(
                "vision_msgs is not installed on the host; 3D pose overlay is unavailable. "
                "Install ros-humble-vision-msgs to enable it.")

    def on_pose_output(self, msg, kind):
        if kind == "pose":
            self.pose_output_count += 1
            count = self.pose_output_count
        else:
            self.tracking_output_count += 1
            count = self.tracking_output_count
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        recorded_index = self.frame_index_by_stamp.get(key)
        # 只数本次录像的非空结果；旧消息、未知帧及重复回执不抬高 FPS。
        result_key = (kind, key)
        if msg.detections and recorded_index is not None and result_key not in self.fps_seen_results:
            self.fps_meter.mark()
            self.fps_seen_results.add(result_key)
        if kind == "pose" and msg.detections:
            self.pose_result_stamps[key] = True
            if len(self.pose_result_stamps) > 600:
                self.pose_result_stamps.popitem(last=False)
        if kind == "tracking" and msg.detections:
            # 种子桥只在更新当前位姿后才发检测，这是同一帧的统一完成回执。
            # 仅 tracker-test 使用此约定；完整 Isaac selector 链路保持原行为。
            if self.tracker_test and recorded_index is not None:
                self.tracking_pose_ready_stamps[key] = True
                if len(self.tracking_pose_ready_stamps) > 1200:
                    self.tracking_pose_ready_stamps.popitem(last=False)
            self.tracking_result_stamps[key] = True
            if recorded_index is not None:
                self.matched_tracking_frames.add(recorded_index)
            if len(self.tracking_result_stamps) > 600:
                self.tracking_result_stamps.popitem(last=False)
            object_mask = self.object_masks_by_stamp.get(key)
            if object_mask is not None:
                coverage = max(color_coverage(item.bbox, self.k, object_mask)
                               for item in msg.detections)
                self.tracking_coverage_by_stamp[key] = coverage
                if len(self.tracking_coverage_by_stamp) > 1200:
                    self.tracking_coverage_by_stamp.popitem(last=False)
        # 推理可能比回放慢：按时间戳取回当时的录像帧，避免把旧位姿画到新帧上。
        cached = self.frame_cache.get(key)
        # --no-preview 不缓存图像；此处的缓存命中仅表示能否画框，不用于判断位姿结果是否匹配。
        cache_status = ("disabled" if not self.preview_enabled else
                        "hit" if cached is not None else "miss")
        self.get_logger().info(
            f"{kind} result {count}: detections={len(msg.detections)}, "
            f"stamp={key[0]}.{key[1]:09d}, "
            f"recorded_frame={recorded_index + 1 if recorded_index is not None else 'unknown'}, "
            f"preview_frame_cache={cache_status}"
            + (f", purple_coverage={self.tracking_coverage_by_stamp[key]:.3f}"
               if kind == "tracking" and key in self.tracking_coverage_by_stamp else ""))
        if not self.preview_enabled or cached is None or not msg.detections:
            return
        index, jpeg = cached
        view = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if view is None:
            self.get_logger().error("Could not decode cached RGB frame")
            return
        for detection in msg.detections:
            try:
                draw_pose_box(view, detection.bbox, self.k)
            except ValueError as error:
                self.get_logger().warning(f"Could not draw 3D box: {error}")
        cv2.putText(view, f"{kind} result: recorded frame {index + 1}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        self.last_pose_view = view
        # 回调只准备图像；窗口统一在主循环刷新，避免同一结果重复 imshow/waitKey。
        self.pose_view_dirty = True

    def on_first_pose(self, _msg):
        self.first_pose_count += 1
        self.get_logger().info("Selector sent first frame to FoundationPose")

    def on_tracking(self, _msg):
        self.tracking_count += 1
        if self.tracking_count == 1 or self.tracking_count % 25 == 0:
            self.get_logger().info(
                f"Tracking camera_info observed: {self.tracking_count} frames")

    def on_tracking_pose_ready(self, msg):
        key = (msg.stamp.sec, msg.stamp.nanosec)
        self.tracking_pose_ready_stamps[key] = True
        if len(self.tracking_pose_ready_stamps) > 1200:
            self.tracking_pose_ready_stamps.popitem(last=False)

    def on_recovery_ready(self, msg):
        self.recovery_ready_indices.add(int(msg.data))

    def subscriber_counts(self):
        return tuple(pub.get_subscription_count() for pub in (
            self.rgb_pub, self.depth_pub, self.info_pub, self.mask_pub))

    def publish_frame(self, rgb_path, depth_path, index):
        raw_rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw_rgb is None or depth_mm is None:
            raise FileNotFoundError(f"Could not read {rgb_path} or {depth_path}")
        if raw_rgb.shape[2] == 4:
            rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
        depth_mm = cv2.resize(depth_mm, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        # SVO 导出的深度 PNG 是毫米；Isaac ROS FoundationPose 输入要求米制 32FC1。
        depth = depth_mm.astype(np.float32) / 1000.0

        rgb_msg = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        mask_msg = self.bridge.cv2_to_imgmsg(self.mask, encoding="mono8")
        info_msg = CameraInfo()
        info_msg.width = self.width
        info_msg.height = self.height
        info_msg.distortion_model = "plumb_bob"
        info_msg.d = [0.0] * 5
        info_msg.k = self.k.flatten().tolist()
        info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info_msg.p = [self.k[0, 0], self.k[0, 1], self.k[0, 2], 0.0,
                      self.k[1, 0], self.k[1, 1], self.k[1, 2], 0.0,
                      0.0, 0.0, 1.0, 0.0]
        stamp = self.get_clock().now().to_msg()
        self.last_published_stamp = (stamp.sec, stamp.nanosec)
        self.frame_index_by_stamp[self.last_published_stamp] = index
        if len(self.frame_index_by_stamp) > 1200:
            self.frame_index_by_stamp.popitem(last=False)
        object_mask = purple_box_mask(rgb)
        if object_mask is not None:
            self.object_masks_by_stamp[self.last_published_stamp] = object_mask
            if len(self.object_masks_by_stamp) > 8:
                self.object_masks_by_stamp.popitem(last=False)
        if self.preview_enabled:
            success, jpeg = cv2.imencode(
                ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                key = (stamp.sec, stamp.nanosec)
                self.frame_cache[key] = (index, jpeg.tobytes())
                if len(self.frame_cache) > 600:
                    self.frame_cache.popitem(last=False)
        for msg in (rgb_msg, depth_msg, mask_msg, info_msg):
            # 四路输入必须属于同一帧，使用完全相同的时间戳供同步器匹配。
            msg.header.stamp = stamp
            msg.header.frame_id = "zed_left_camera_optical_frame"
        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.mask_pub.publish(mask_msg)
        self.info_pub.publish(info_msg)
        return rgb


def show_preview(rgb, mask, frame_index, total, player, status="Playing"):
    now = time.monotonic()
    # 等待时最多 10 Hz 重画；新位姿仍立即显示，FPS 的采样统计不变。
    if now < player.next_preview_draw and not player.pose_view_dirty:
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            raise KeyboardInterrupt
        return
    player.next_preview_draw = now + 0.1
    view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if frame_index == 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, contours, -1, (0, 255, 255), 2)
    # 订阅数量属于状态信息，每秒查一次即可，无需每次画窗口都查询 ROS 图。
    if player.preview_subscriber_counts is None or now >= player.next_preview_subscriber_check:
        player.preview_subscriber_counts = player.subscriber_counts()
        player.next_preview_subscriber_check = now + 1.0
    counts = player.preview_subscriber_counts
    lines = [
        f"{status}  frame {frame_index + 1}/{total}  (Q: stop)",
        ("Direct FoundationPose input (selector bypassed)" if player.direct_pose else
         "Tracker-only input from bridge" if player.tracker_test else
         f"selector first_pose={player.first_pose_count}  tracking={player.tracking_count}"),
        f"Isaac subscribers RGB/depth/info/mask={counts}",
        f"pose/tracking outputs={player.pose_output_count}/{player.tracking_output_count}",
    ]
    cv2.rectangle(view, (0, 0), (view.shape[1], 86), (0, 0, 0), -1)
    for row, line in enumerate(lines):
        cv2.putText(view, line, (8, 18 + 20 * row), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
    fps = player.fps_meter.value()
    draw_fps(view, fps)
    cv2.imshow(PREVIEW_WINDOW, view)
    fps_label = f"{fps:.1f}"
    if player.last_pose_view is not None and (player.pose_view_dirty or
                                             fps_label != player.last_pose_fps_label):
        # 没有新结果时保留上一帧框，但刷新 FPS，避免窗口一直显示旧速度。
        cv2.imshow(POSE_WINDOW, draw_fps(player.last_pose_view.copy(), fps))
        player.pose_view_dirty = False
        player.last_pose_fps_label = fps_label
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--mask", type=Path, default=DEFAULT_MASK)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--result-timeout", type=float, default=60.0,
                        help="seconds to wait; 0 waits indefinitely in --tracker-full mode")
    parser.add_argument("--first-result-timeout", type=float, default=120.0,
                        help="tracker-full: allow extra startup warmup time on the first frame")
    parser.add_argument("--max-frames", type=int, default=300, help="0: play all frames")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="temporary tracking review: 2 processes every other recorded frame")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-preview", action="store_true", help="do not open an OpenCV window")
    parser.add_argument("--direct-pose", action="store_true",
                        help="publish directly to FoundationPose inputs, bypassing selector")
    parser.add_argument("--tracker-test", action="store_true",
                        help="skip recorded frame 0 and wait for each tracker result")
    parser.add_argument("--require-pose-result", action="store_true",
                        help="require a matching nonempty pose result for each direct-pose frame")
    parser.add_argument("--pose-probe", action="store_true",
                        help="repeat one recorded frame until direct pose returns a detection")
    parser.add_argument("--tracker-probe", action="store_true",
                        help="repeat recorded frame 2 until tracking returns a detection")
    parser.add_argument("--tracker-full", action="store_true",
                        help="strictly wait for each frame's pose and detection before advancing")
    parser.add_argument("--tracker-stream", action="store_true",
                        help="publish at --fps without waiting; bridge drops backlogged frames")
    parser.add_argument("--frame-retries", type=int, default=0,
                        help="deprecated in --tracker-full; must be 0 for strict playback")
    parser.add_argument("--auto-recover-original", action="store_true",
                        help="offline only: reseed tracker from saved original FP poses after drift")
    parser.add_argument("--drift-coverage", type=float, default=0.65,
                        help="minimum purple-object coverage inside tracked 3D box")
    parser.add_argument("--drift-consecutive", type=int, default=3,
                        help="consecutive low-coverage frames before offline reseed")
    parser.add_argument("--recovery-cooldown", type=int, default=10,
                        help="processed frames to wait after an offline reseed")
    args = parser.parse_args()
    if args.direct_pose and args.tracker_test:
        parser.error("--direct-pose and --tracker-test cannot be combined")
    if args.require_pose_result and not args.direct_pose:
        parser.error("--require-pose-result requires --direct-pose")
    if args.pose_probe and (not args.direct_pose or args.tracker_test or args.max_frames != 1):
        parser.error("--pose-probe requires --direct-pose --max-frames 1")
    if args.tracker_probe and (not args.tracker_test or args.direct_pose or args.max_frames != 1):
        parser.error("--tracker-probe requires --tracker-test --max-frames 1")
    if args.tracker_probe and args.frame_stride != 1:
        parser.error("--tracker-probe requires --frame-stride 1")
    if args.tracker_full and (not args.tracker_test or args.tracker_probe):
        parser.error("--tracker-full requires --tracker-test without --tracker-probe")
    if args.tracker_stream and (not args.tracker_test or args.tracker_probe or args.tracker_full):
        parser.error("--tracker-stream requires --tracker-test without another tracker mode")
    if args.frame_retries < 0:
        parser.error("--frame-retries must be nonnegative")
    if args.tracker_full and args.frame_retries:
        parser.error("--tracker-full sends each recorded frame once; use --frame-retries 0")
    if args.auto_recover_original and not args.tracker_full:
        parser.error("--auto-recover-original requires --tracker-test --tracker-full")
    if not 0 < args.drift_coverage < 1 or args.drift_consecutive < 1 or args.recovery_cooldown < 0:
        parser.error("invalid drift coverage, consecutive count or recovery cooldown")
    if args.frame_stride < 1 or (args.frame_stride != 1 and not args.tracker_test):
        parser.error("--frame-stride must be positive and only applies to --tracker-test")
    if (not 0 < args.scale <= 1 or args.fps <= 0 or args.max_frames < 0 or
            args.result_timeout < 0 or args.first_result_timeout < 0 or
            (args.result_timeout == 0 and not args.tracker_full)):
        parser.error("scale must be in (0,1], fps positive; timeout 0 needs --tracker-full")
    frames, k, mask, width, height = read_assets(args.sequence, args.mask, args.scale)
    total_recorded_frames = len(frames)
    first_frame_index = 1 if args.tracker_test else 0
    # 临时隔帧检查：保留原版首帧种子，从录像第 2 帧起每隔一帧送一次 tracker。
    frames = frames[first_frame_index::args.frame_stride]
    frames = frames[:args.max_frames] if args.max_frames else frames
    if not frames:
        parser.error("no frames remain after selecting the playback range")
    print(f"Ready: {len(frames)} frames, {width}x{height}, "
          f"mask pixels={np.count_nonzero(mask)}, depth is converted mm -> meters", flush=True)
    if args.check_only:
        return

    rclpy.init()
    player = RecordingPlayer(k, mask, width, height, direct_pose=args.direct_pose,
                             preview_enabled=not args.no_preview,
                             tracker_test=args.tracker_test)
    # 临时纯 tracking 对照：默认关闭离线恢复；仅显式传参数时才使用原版逐帧位姿。
    # 首帧种子仍保留，之后正常用 Isaac 的上一帧输出，不重复运行 pose。
    if args.tracker_full:
        player.get_logger().info(
            f"Tracking offline recovery={'ON' if args.auto_recover_original else 'OFF'}; "
            "original FP pose is used for the initial seed")
    low_coverage_streak = 0
    recovery_cooldown = 0
    recovery_count = 0
    try:
        first_image = cv2.imread(
            str(args.sequence / "rgb" / frames[0]["rgb_file"]), cv2.IMREAD_COLOR)
        if first_image is None:
            raise FileNotFoundError("First RGB file is missing")
        first_rgb = cv2.cvtColor(
            cv2.resize(first_image, (width, height), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB)
        if not args.no_preview:
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
        # 等四路订阅都建立后才发首帧，避免唯一的首次定位帧被 ROS 发现过程丢掉。
        deadline = time.monotonic() + 60
        next_report = time.monotonic()
        while time.monotonic() < deadline and min(player.subscriber_counts()) == 0:
            rclpy.spin_once(player, timeout_sec=0.2)
            if time.monotonic() >= next_report:
                player.get_logger().info(
                    f"Waiting for Isaac subscribers RGB/depth/info/mask={player.subscriber_counts()}")
                next_report = time.monotonic() + 5
            if not args.no_preview:
                show_preview(first_rgb, mask, first_frame_index,
                             total_recorded_frames, player, "Waiting for Isaac")
        if min(player.subscriber_counts()) == 0:
            raise RuntimeError(f"Isaac subscribers missing: {player.subscriber_counts()}")
        player.get_logger().info(f"Isaac subscribers rgb/depth/info/mask={player.subscriber_counts()}")
        if not args.direct_pose and not args.tracker_test:
            rgb_subscribers = player.get_subscriptions_info_by_topic("/isaacfp/rgb")
            if any(endpoint.node_name == "isaacfp_seed_tracking_bridge"
                   for endpoint in rgb_subscribers):
                raise RuntimeError(
                    "seed_tracking_bridge.py is running during full Selector playback. "
                    "Stop the seed bridge before testing zed_foundationpose.launch.py.")
        if args.tracker_test:
            # tracker-only 回放不能和完整 Selector 链路共用这些话题；二者会同时向
            # tracking/pose_input 发位姿，导致一帧收到两套不同的种子与时间戳。
            rgb_subscribers = player.get_subscriptions_info_by_topic("/isaacfp/rgb")
            pose_publishers = player.get_publishers_info_by_topic("/tracking/pose_input")
            if (any(endpoint.node_name == "selector" for endpoint in rgb_subscribers) or
                    any(endpoint.node_name == "selector" for endpoint in pose_publishers)):
                raise RuntimeError(
                    "Full Isaac /selector is running alongside seed_tracking_bridge.py. "
                    "Stop zed_foundationpose.launch.py; for --tracker-test run only "
                    "zed_foundationpose_tracker_only.launch.py and the seed bridge.")
            # 四路订阅属于桥接节点；另查 tracker 自己的位姿订阅，避免空等 60 秒。
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                endpoints = player.get_subscriptions_info_by_topic("/tracking/pose_input")
                if any(endpoint.node_name == "foundationpose_tracker" for endpoint in endpoints):
                    break
                rclpy.spin_once(player, timeout_sec=0.2)
            else:
                raise RuntimeError(
                    "Bridge is connected, but /foundationpose_tracker did not subscribe "
                    "to /tracking/pose_input. Check the tracker launch error first.")
            if args.tracker_full:
                deadline = time.monotonic() + 10
                while (player.count_publishers("/isaacfp/tracking_pose_ready") == 0
                       or player.count_publishers("/isaacfp/tracking_debug_output") == 0) and time.monotonic() < deadline:
                    rclpy.spin_once(player, timeout_sec=0.1)
                if (player.count_publishers("/isaacfp/tracking_pose_ready") == 0 or
                        player.count_publishers("/isaacfp/tracking_debug_output") == 0):
                    raise RuntimeError(
                        "Bridge is missing strict pose/detection feedback publishers; "
                        "restart seed_tracking_bridge.py to load the latest code")
        if args.direct_pose:
            player.get_logger().info(
                "Direct pose mode: input reaches FoundationPose without the selector; "
                "watch /pose_estimation/output in another container terminal")
        playback_started = time.monotonic()
        for index, frame in enumerate(frames):
            started = time.monotonic()
            recorded_index = first_frame_index + index * args.frame_stride
            rgb = player.publish_frame(args.sequence / "rgb" / frame["rgb_file"],
                                       args.sequence / "depth" / frame["depth_file"],
                                       recorded_index)
            if not args.no_preview:
                show_preview(rgb, mask, recorded_index, total_recorded_frames,
                             player)
            if index == 0 or (index + 1) % 25 == 0:
                player.get_logger().info(
                    f"Published {index + 1}/{len(frames)} frames; "
                    f"stamp={player.last_published_stamp[0]}."
                    f"{player.last_published_stamp[1]:09d}; "
                    + ("direct pose mode" if args.direct_pose else
                       f"tracker input camera_info={player.tracking_count}" if args.tracker_test else
                       f"selector first_pose/tracking="
                       f"{player.first_pose_count}/{player.tracking_count}"))
            if not args.tracker_full:
                while time.monotonic() - started < 1 / args.fps:
                    rclpy.spin_once(player, timeout_sec=0.01)
            if args.pose_probe:
                # 单次发布可能落在节点刚完成发现、但内部图尚未就绪的窗口。
                # 反复发送同一录像帧，每次用新时间戳；只接受对应时间戳的非空结果。
                stamps = {player.last_published_stamp}
                attempts = 1
                deadline = time.monotonic() + args.result_timeout
                next_publish = time.monotonic() + 2.0
                while not stamps.intersection(player.pose_result_stamps):
                    now = time.monotonic()
                    if now >= deadline:
                        raise RuntimeError(
                            f"No matching nonempty pose result after {attempts} publications "
                            f"of recorded frame 1 in {args.result_timeout:g} seconds; "
                            "inspect FoundationPose container logs")
                    if now >= next_publish:
                        player.publish_frame(args.sequence / "rgb" / frame["rgb_file"],
                                             args.sequence / "depth" / frame["depth_file"],
                                             recorded_index)
                        stamps.add(player.last_published_stamp)
                        attempts += 1
                        player.get_logger().info(
                            f"Pose probe sent recorded frame 1 again; attempt={attempts}, "
                            f"pose messages={player.pose_output_count}")
                        next_publish = now + 2.0
                    rclpy.spin_once(player, timeout_sec=0.1)
                    if not args.no_preview:
                        show_preview(rgb, mask, recorded_index, 1, player, "Probing pose")
                player.get_logger().info(
                    f"POSE-ONLY PASS: recorded frame 1 returned a matching nonempty detection "
                    f"after {attempts} publication(s)")
            elif args.require_pose_result:
                # 单帧基础测试必须有同时间戳的非空位姿；节点启动日志不算通过。
                deadline = time.monotonic() + args.result_timeout
                while (player.last_published_stamp not in player.pose_result_stamps
                       and time.monotonic() < deadline):
                    rclpy.spin_once(player, timeout_sec=0.1)
                    if not args.no_preview:
                        show_preview(rgb, mask, recorded_index,
                                     len(frames) + first_frame_index, player, "Estimating pose")
                if player.last_published_stamp not in player.pose_result_stamps:
                    raise RuntimeError(
                        f"No matching nonempty pose result for recorded frame "
                        f"{recorded_index + 1} in {args.result_timeout:g} seconds; "
                        "inspect FoundationPose logs")
                player.get_logger().info(
                    f"POSE-ONLY PASS: recorded frame {recorded_index + 1} produced a detection")
            if args.tracker_stream:
                # 按录像帧率连续发布；桥接端等待上一帧位姿并丢弃积压的旧帧。
                pass
            elif args.tracker_full:
                # 一帧同时拿到位姿矩阵与非空检测，才显示下一张录像帧。
                stamp = player.last_published_stamp
                # 实测首帧 render 曾耗时约 70 秒；预热预算仅影响首帧，不跳过结果。
                wait_timeout = (max(args.result_timeout, args.first_result_timeout)
                                if index == 0 else args.result_timeout)
                deadline = (time.monotonic() + wait_timeout
                            if args.result_timeout else None)
                next_wait_log = time.monotonic() + 10
                while (stamp not in player.tracking_pose_ready_stamps or
                       stamp not in player.tracking_result_stamps):
                    if deadline is not None and time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"Recorded frame {recorded_index + 1} did not complete: "
                            f"pose_ready={stamp in player.tracking_pose_ready_stamps}, "
                            f"detection={stamp in player.tracking_result_stamps}, "
                            f"stamp={stamp}; playback stopped without skipping")
                    rclpy.spin_once(player, timeout_sec=0.1)
                    if time.monotonic() >= next_wait_log:
                        player.get_logger().warning(
                            f"Still waiting for recorded frame {recorded_index + 1}: "
                            f"pose_ready={stamp in player.tracking_pose_ready_stamps}, "
                            f"detection={stamp in player.tracking_result_stamps}")
                        next_wait_log = time.monotonic() + 10
                    if not args.no_preview:
                        show_preview(rgb, mask, recorded_index,
                                     total_recorded_frames, player, "Tracking")
                if args.auto_recover_original:
                    # 紫色区域独立于 tracker 提取；连续漏出投影框才判为明显漂移。
                    coverage = player.tracking_coverage_by_stamp.get(stamp)
                    if recovery_cooldown:
                        recovery_cooldown -= 1
                        low_coverage_streak = 0
                    elif coverage is None:
                        low_coverage_streak = 0
                        player.get_logger().warning(
                            f"No reliable purple mask for recorded frame {recorded_index + 1}; "
                            "drift decision skipped")
                    elif coverage < args.drift_coverage:
                        low_coverage_streak += 1
                    else:
                        low_coverage_streak = 0
                    if low_coverage_streak >= args.drift_consecutive and index + 1 < len(frames):
                        next_recorded_index = first_frame_index + (index + 1) * args.frame_stride
                        player.get_logger().warning(
                            f"DRIFT_TRIGGER frame={recorded_index + 1} "
                            f"purple_coverage={coverage:.3f} "
                            f"consecutive={low_coverage_streak}; "
                            f"requesting original pose for frame {next_recorded_index + 1}")
                        player.recovery_pub.publish(UInt32(data=next_recorded_index))
                        recovery_deadline = time.monotonic() + 10
                        while (next_recorded_index not in player.recovery_ready_indices and
                               time.monotonic() < recovery_deadline):
                            rclpy.spin_once(player, timeout_sec=0.1)
                        if next_recorded_index not in player.recovery_ready_indices:
                            raise RuntimeError(
                                f"Offline reseed for recorded frame {next_recorded_index + 1} "
                                "was not acknowledged by seed_tracking_bridge.py")
                        recovery_count += 1
                        recovery_cooldown = args.recovery_cooldown
                        low_coverage_streak = 0
                        player.get_logger().warning(
                            f"ORIGINAL_POSE_RESEED confirmed for recorded frame "
                            f"{next_recorded_index + 1}; total={recovery_count}")
            elif args.tracker_probe:
                # 与 pose 探针一致：重发录像第二帧，排除首次发帧落在 GXF 启动边界。
                stamps = {player.last_published_stamp}
                attempts = 1
                deadline = time.monotonic() + args.result_timeout
                next_publish = time.monotonic() + 2.0
                while not stamps.intersection(player.tracking_result_stamps):
                    now = time.monotonic()
                    if now >= deadline:
                        raise RuntimeError(
                            f"No matching nonempty tracking result after {attempts} "
                            f"publications of recorded frame 2 in {args.result_timeout:g} "
                            "seconds; inspect bridge and tracker logs")
                    if now >= next_publish:
                        player.publish_frame(args.sequence / "rgb" / frame["rgb_file"],
                                             args.sequence / "depth" / frame["depth_file"],
                                             recorded_index)
                        stamps.add(player.last_published_stamp)
                        attempts += 1
                        player.get_logger().info(
                            f"Tracker probe sent recorded frame 2 again; attempt={attempts}, "
                            f"tracking messages={player.tracking_output_count}")
                        next_publish = now + 2.0
                    rclpy.spin_once(player, timeout_sec=0.1)
                    if not args.no_preview:
                        show_preview(rgb, mask, recorded_index, 2, player, "Probing tracking")
                player.get_logger().info(
                    f"TRACKING PASS: recorded frame 2 returned a matching nonempty detection "
                    f"after {attempts} publication(s)")
            elif args.tracker_test:
                # 离线验证逐帧等待结果，避免 5 FPS 回放把慢于此速度的推理队列淹没。
                deadline = time.monotonic() + args.result_timeout
                while (player.last_published_stamp not in player.tracking_result_stamps
                       and time.monotonic() < deadline):
                    rclpy.spin_once(player, timeout_sec=0.1)
                    if not args.no_preview:
                        show_preview(rgb, mask, recorded_index,
                                     total_recorded_frames, player, "Tracking")
                if player.last_published_stamp not in player.tracking_result_stamps:
                    raise RuntimeError(
                        f"No matching tracking detection for recorded frame "
                        f"{recorded_index + 1} in {args.result_timeout:g} seconds; "
                        "inspect tracker logs")
        player.get_logger().info(
            "Playback complete; " + ("direct pose mode" if args.direct_pose else
                                    f"tracker input camera_info={player.tracking_count}" if args.tracker_test else
                                    f"selector first_pose/tracking="
                                    f"{player.first_pose_count}/{player.tracking_count}")
            + f"; pose/tracking outputs={player.pose_output_count}/{player.tracking_output_count}")
        playback_seconds = time.monotonic() - playback_started
        # Keep callbacks alive briefly for a final pose result after the last frame.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            rclpy.spin_once(player, timeout_sec=0.1)
            if not args.no_preview:
                show_preview(rgb, mask, recorded_index,
                             total_recorded_frames, player, "Finished")
        if args.tracker_full:
            player.get_logger().info(
                f"TRACKING STRICT SUMMARY: completed_frames={len(frames)}, "
                f"matched_detections={len(player.matched_tracking_frames)}, "
                f"original_pose_reseeds={recovery_count}")
        if args.tracker_stream:
            player.get_logger().info(
                f"TRACKING STREAM SUMMARY: input_frames={len(frames)}, "
                f"input_rate={len(frames) / playback_seconds:.1f} fps, "
                f"bridge_forwarded={player.tracking_count}, "
                f"matched_tracking_frames={len(player.matched_tracking_frames)}, "
                f"tracking_messages={player.tracking_output_count}")
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_preview:
            cv2.destroyAllWindows()
        player.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
