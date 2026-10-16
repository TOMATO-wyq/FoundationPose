#!/usr/bin/env python3
"""Publish synchronized, scaled ZED RGB-D and a hand-drawn mask for Isaac ROS."""

import argparse
import copy
import time
from pathlib import Path

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def choose_polygon(rgb):
    points = []
    window = "IsaacFP: draw object mask"
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def redraw():
        canvas = bgr.copy()
        if points:
            vertices = np.asarray(points, dtype=np.int32)
            cv2.polylines(canvas, [vertices], len(points) >= 3, (0, 255, 0), 2)
            for point in points:
                cv2.circle(canvas, point, 4, (0, 255, 255), -1)
        cv2.putText(canvas, "LMB: add  RMB/U: undo  R: clear  Enter: start  Q: quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow(window, canvas)

    def mouse(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13) and len(points) >= 3:
            break
        if key in (ord("u"), 8, 127) and points:
            points.pop()
            redraw()
        elif key == ord("r"):
            points.clear()
            redraw()
        elif key in (ord("q"), 27):
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("mask selection cancelled")
    cv2.destroyWindow(window)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
    return mask


class ZedBridge(Node):
    camera_label = "ZED"
    node_name = "isaacfp_zed_bridge"

    def __init__(self, args):
        super().__init__(self.node_name)
        self.args = args
        self.bridge = CvBridge()
        self.mask = None
        self.frames = 0
        self.last_publish_time = 0.0
        self.received = {"rgb": 0, "depth": 0, "camera_info": 0, "rgb_depth_pairs": 0,
                         "selector_first_pose": 0, "selector_tracking": 0}
        self.info_msg = None
        self.rgb_pub = self.create_publisher(Image, "/isaacfp/rgb", 10)
        self.depth_pub = self.create_publisher(Image, "/isaacfp/depth", 10)
        self.info_pub = self.create_publisher(CameraInfo, "/isaacfp/camera_info", 10)
        self.mask_pub = self.create_publisher(Image, "/isaacfp/mask", 10)

        rgb_sub = message_filters.Subscriber(
            self, Image, args.rgb_topic, qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(
            self, Image, args.depth_topic, qos_profile=qos_profile_sensor_data)
        self.subscribers = [rgb_sub, depth_sub]
        rgb_sub.registerCallback(lambda _msg: self.count("rgb"))
        depth_sub.registerCallback(lambda _msg: self.count("depth"))
        self.info_sub = self.create_subscription(
            CameraInfo, args.info_topic, self.on_info, qos_profile_sensor_data)
        self.pose_info_sub = self.create_subscription(
            CameraInfo, "/pose_estimation/camera_info",
            lambda _msg: self.count("selector_first_pose"), qos_profile_sensor_data)
        self.tracking_info_sub = self.create_subscription(
            CameraInfo, "/tracking/camera_info",
            lambda _msg: self.count("selector_tracking"), qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            self.subscribers, queue_size=10, slop=args.slop)
        self.sync.registerCallback(self.on_frame)
        self.create_timer(5.0, self.report_waiting)
        self.get_logger().info(f"Waiting for {self.camera_label} RGB, depth and camera_info")

    def count(self, stream):
        self.received[stream] += 1

    def on_info(self, msg):
        self.info_msg = msg
        self.count("camera_info")

    def report_waiting(self):
        if self.frames == 0:
            self.get_logger().warning(
                "Waiting for first published frame; received "
                + ", ".join(f"{key}={value}" for key, value in self.received.items()))

    def convert_images(self, rgb_msg, depth_msg):
        # ZED 专用输入：彩色 BGRA，深度浮点米；D435 在独立脚本中覆盖此转换。
        if rgb_msg.encoding.lower() != "bgra8" or depth_msg.encoding != "32FC1":
            raise ValueError(
                f"Expected bgra8 / 32FC1, got {rgb_msg.encoding} / {depth_msg.encoding}")
        rgb = cv2.cvtColor(self.bridge.imgmsg_to_cv2(rgb_msg, "bgra8"), cv2.COLOR_BGRA2RGB)
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        return rgb, depth

    def on_frame(self, rgb_msg, depth_msg):
        self.count("rgb_depth_pairs")
        info_msg = self.info_msg
        if info_msg is None:
            return
        if self.mask is not None and time.monotonic() - self.last_publish_time < 1 / self.args.max_fps:
            return
        try:
            rgb, depth = self.convert_images(rgb_msg, depth_msg)
        except ValueError as error:
            self.get_logger().error(str(error))
            return
        if rgb.shape[:2] != depth.shape[:2]:
            self.get_logger().error(f"RGB/depth sizes differ: {rgb.shape[:2]} / {depth.shape[:2]}")
            return
        if (info_msg.width, info_msg.height) != (rgb_msg.width, rgb_msg.height):
            self.get_logger().error(
                f"CameraInfo size {info_msg.width}x{info_msg.height} does not match "
                f"RGB {rgb_msg.width}x{rgb_msg.height}")
            return

        width = round(rgb.shape[1] * self.args.scale)
        height = round(rgb.shape[0] * self.args.scale)
        if (width, height) != (rgb.shape[1], rgb.shape[0]):
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)

        if self.mask is None:
            if self.args.mask_file:
                loaded = cv2.imread(self.args.mask_file, cv2.IMREAD_GRAYSCALE)
                if loaded is None:
                    raise FileNotFoundError(self.args.mask_file)
                self.mask = cv2.resize(loaded, (width, height), interpolation=cv2.INTER_NEAREST)
                self.mask = np.where(self.mask > 0, 255, 0).astype(np.uint8)
            else:
                self.mask = choose_polygon(rgb)
            if not np.any(self.mask):
                raise ValueError("The selected mask is empty")
            self.get_logger().info(
                f"Mask ready: {width}x{height}, {np.count_nonzero(self.mask)} foreground pixels")

        rgb_out = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        depth_out = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        mask_out = self.bridge.cv2_to_imgmsg(self.mask, encoding="mono8")
        info_out = copy.deepcopy(info_msg)
        # The source messages can have nearby, but non-identical timestamps.
        # Isaac ROS requires all four inputs for one frame to match exactly.
        stamp = rgb_msg.header.stamp
        for msg in (rgb_out, depth_out, mask_out, info_out):
            msg.header.stamp = stamp
            msg.header.frame_id = rgb_msg.header.frame_id
        sx = width / rgb_msg.width
        sy = height / rgb_msg.height
        info_out.width = width
        info_out.height = height
        for index in range(3):
            info_out.k[index] *= sx
        for index in range(3, 6):
            info_out.k[index] *= sy
        for index in range(4):
            info_out.p[index] *= sx
        for index in range(4, 8):
            info_out.p[index] *= sy
        info_out.roi.x_offset = round(info_out.roi.x_offset * sx)
        info_out.roi.y_offset = round(info_out.roi.y_offset * sy)
        info_out.roi.width = round(info_out.roi.width * sx)
        info_out.roi.height = round(info_out.roi.height * sy)
        self.rgb_pub.publish(rgb_out)
        self.depth_pub.publish(depth_out)
        self.info_pub.publish(info_out)
        self.mask_pub.publish(mask_out)
        self.last_publish_time = time.monotonic()
        self.frames += 1
        if self.frames == 1 or self.frames % 30 == 0:
            subscriber_counts = (
                self.rgb_pub.get_subscription_count(),
                self.depth_pub.get_subscription_count(),
                self.info_pub.get_subscription_count(),
                self.mask_pub.get_subscription_count(),
            )
            self.get_logger().info(
                f"Published {self.frames} synchronized RGB-D/mask frames; "
                f"Isaac subscribers rgb/depth/info/mask={subscriber_counts}; "
                f"selector first_pose/tracking="
                f"{self.received['selector_first_pose']}/{self.received['selector_tracking']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-topic", default="/zed/zed_node/rgb/color/rect/image")
    parser.add_argument("--depth-topic", default="/zed/zed_node/depth/depth_registered")
    parser.add_argument("--info-topic", default="/zed/zed_node/rgb/color/rect/camera_info")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--max-fps", type=float, default=10.0)
    parser.add_argument("--slop", type=float, default=0.05)
    parser.add_argument("--mask-file", type=str, help="existing mask PNG; default: draw a polygon")
    args = parser.parse_args()
    if not 0 < args.scale <= 1:
        parser.error("--scale must be in (0, 1]")
    if args.max_fps <= 0:
        parser.error("--max-fps must be positive")
    if args.mask_file and not Path(args.mask_file).is_file():
        parser.error(f"mask file does not exist: {args.mask_file}")
    rclpy.init()
    node = ZedBridge(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
