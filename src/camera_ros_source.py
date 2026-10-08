#!/usr/bin/env python3
"""Host ROS2 RGB-D input for original FoundationPose live tracking."""
import argparse
import os
import socket
import stat
import threading
import time
import json
import struct
from pathlib import Path

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from live_rgbd_transport import send_frame, recv_exact

TOPICS = {
    "zed2i": ("/zed/zed_node/rgb/color/rect/image",
              "/zed/zed_node/depth/depth_registered",
              "/zed/zed_node/rgb/color/rect/camera_info"),
    "d435i": ("/camera/camera/color/image_raw",
              "/camera/camera/aligned_depth_to_color/image_raw",
              "/camera/camera/color/camera_info"),
}


def clear_stale_socket(path):
    """Remove only an unregistered Unix socket left by a previous process."""
    try:
        previous = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(previous.st_mode):
        raise RuntimeError(f'Refusing to replace non-socket file: {path}')
    # Do not connect to the source: that would occupy its single-client loop.
    entries = Path('/proc/net/unix').read_text().splitlines()
    for line in entries:
        fields = line.split(maxsplit=7)
        if len(fields) == 8 and Path(fields[7]).resolve() == path.resolve():
            raise RuntimeError(f'RGB-D source is already running at {path}; stop the old source with Ctrl+C')
    current = path.lstat()
    if (current.st_dev, current.st_ino) != (previous.st_dev, previous.st_ino):
        raise RuntimeError(f'Socket changed during startup: {path}; retry')
    path.unlink()
    print(f'Removed stale RGB-D socket: {path}', flush=True)


class CameraSource(Node):
    def __init__(self, args):
        super().__init__(f"original_fp_{args.camera}_source")
        self.args = args
        self.bridge = CvBridge()
        self.condition = threading.Condition()
        self.latest = None
        self.sequence = 0
        self.info = None
        self.last_arrival = None
        self.source_timings = {}
        if args.camera == 'd435i' and args.input_mode != 'separate':
            from realsense2_camera_msgs.msg import RGBD
            self.rgbd_sub = self.create_subscription(
                RGBD, args.rgbd_topic, self.on_rgbd,
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
            self.get_logger().info(f'Single RGBD input: {args.rgbd_topic}; enable_rgbd/enable_sync/align_depth must be true')
            return
        topics = TOPICS[args.camera]
        self.info_sub = self.create_subscription(
            CameraInfo, args.info_topic or topics[2], self.on_info, qos_profile_sensor_data)
        self.subscribers = [message_filters.Subscriber(
            self, Image, topic, qos_profile=qos_profile_sensor_data)
            for topic in (args.rgb_topic or topics[0], args.depth_topic or topics[1])]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            self.subscribers, queue_size=5, slop=args.slop)
        self.sync.registerCallback(self.on_frame)
        self.get_logger().info(f"Camera={args.camera}; waiting for aligned RGB-D and color intrinsics")

    def on_info(self, message):
        self.info = message

    def on_rgbd(self, message):
        self.info = message.rgb_camera_info
        self.on_frame(message.rgb, message.depth)

    def on_frame(self, color, depth_message):
        if self.info is None:
            return
        try:
            started = time.perf_counter()
            interval = None if self.last_arrival is None else started-self.last_arrival
            self.last_arrival = started
            rgb = self.bridge.imgmsg_to_cv2(color, "rgb8")
            depth = self.bridge.imgmsg_to_cv2(depth_message, "passthrough")
            if depth_message.encoding.upper() == "16UC1":
                # RealSense ROS 整数深度为毫米；原版 FP 输入要求浮点米。
                depth = depth.astype(np.float32) * self.args.depth_scale
            elif depth_message.encoding.upper() == "32FC1":
                depth = depth.astype(np.float32)
            else:
                raise ValueError(f"Unsupported depth encoding: {depth_message.encoding}")
            converted = time.perf_counter()
            if depth.shape != rgb.shape[:2]:
                raise ValueError("Depth must be aligned to color; image sizes differ")
            if (self.info.width, self.info.height) != (color.width, color.height):
                raise ValueError("Color CameraInfo dimensions do not match RGB")
            k = np.asarray(self.info.k, dtype=np.float64).reshape(3, 3).copy()
            if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
                raise ValueError("Invalid color camera intrinsics")
            height, width = depth.shape
            size = (max(1, round(width * self.args.scale)),
                    max(1, round(height * self.args.scale)))
            if size != (width, height):
                rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
                k[0] *= size[0] / width
                k[1] *= size[1] / height
            depth[~np.isfinite(depth) | (depth < 0.001)] = 0
            finished = time.perf_counter()
            self.source_timings = dict(source_convert_s=converted-started,
                source_resize_cleanup_s=finished-converted, source_interval_s=interval,
                source_rgb_depth_stamp_delta_s=abs(
                    color.header.stamp.sec+color.header.stamp.nanosec/1e9
                    -depth_message.header.stamp.sec-depth_message.header.stamp.nanosec/1e9))
            stamp = color.header.stamp.sec + color.header.stamp.nanosec / 1e9
            # 只保留最新完整帧；首帧定位或 tracking 慢时，不累计相机旧帧。
            with self.condition:
                self.sequence += 1
                self.latest = (np.ascontiguousarray(rgb), depth, k, stamp,
                               color.header.frame_id, self.sequence, dict(self.source_timings))
                self.condition.notify_all()
            if self.sequence == 1:
                self.get_logger().info(f"RGB-D ready: {size}; {color.encoding}/{depth_message.encoding}")
            if self.sequence % 30 == 0:
                self.get_logger().info(f'RGB-D timings: {self.source_timings}')
        except Exception as error:
            self.get_logger().error(f"RGB-D conversion failed: {error}", throttle_duration_sec=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True, choices=TOPICS)
    parser.add_argument("--socket", default=str(Path(__file__).resolve().parents[1] / "outputs/live_rgbd.sock"))
    parser.add_argument('--input-mode', choices=('auto', 'separate'), default='auto',
                        help='auto: D435i single RGBD, ZED separate synchronized topics')
    parser.add_argument('--rgbd-topic', default='/camera/camera/rgbd')
    for name in ("rgb", "depth", "info"):
        parser.add_argument(f"--{name}-topic")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--slop", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 < args.scale <= 1 or not np.isfinite(args.depth_scale) or args.depth_scale <= 0:
        parser.error("invalid scale/depth-scale")
    if not np.isfinite(args.slop) or args.slop < 0:
        parser.error("invalid slop")
    path = Path(args.socket)
    path.parent.mkdir(parents=True, exist_ok=True)
    clear_stale_socket(path)
    rclpy.init()
    source = CameraSource(args)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    bound = False
    thread = None
    try:
        server.bind(str(path))
        bound = True
        os.chmod(path, 0o600)
        server.listen(1)
        server.settimeout(1)
        thread = threading.Thread(target=rclpy.spin, args=(source,), daemon=True)
        thread.start()
        print(f"Local RGB-D socket: {path}", flush=True)
        while rclpy.ok():
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            with connection:
                connection.settimeout(None)  # 首次 register 可能数分钟，不因客户端计算而断开。
                last_sequence = -1
                try:
                    while rclpy.ok():
                        command = connection.recv(1)
                        if not command:
                            break
                        roi = None
                        if command == b'R':
                            length = struct.unpack('!I', recv_exact(connection, 4))[0]
                            if length > 128:
                                raise ValueError('Invalid ROI request size')
                            roi = json.loads(recv_exact(connection, length))
                        elif command != b'N':
                            raise ValueError('Unknown frame request')
                        with source.condition:
                            ready = source.condition.wait_for(
                                lambda: source.latest is not None and source.sequence > last_sequence,
                                timeout=30)
                            if not ready:
                                raise TimeoutError("No new RGB-D frame for 30 seconds")
                            frame = source.latest
                        send_frame(connection, frame, roi=roi)
                        last_sequence = frame[5]
                except (OSError, TimeoutError, ValueError, EOFError) as error:
                    print(f"Client disconnected: {error}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        if bound:
            path.unlink(missing_ok=True)
        if rclpy.ok():
            rclpy.shutdown()
        if thread:
            thread.join(timeout=2)
        source.destroy_node()


if __name__ == "__main__":
    main()
