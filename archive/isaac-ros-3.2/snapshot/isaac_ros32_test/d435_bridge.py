#!/usr/bin/env python3
"""Adapt aligned RealSense D435/D435i ROS2 RGB-D for Isaac FoundationPose."""

import argparse
from pathlib import Path

import numpy as np
import rclpy

from zed_bridge import ZedBridge


class D435Bridge(ZedBridge):
    # 复用同步、缩放内参和手画掩码；相机话题及编码转换独立，避免混用 ZED 数据。
    camera_label = "D435/D435i"
    node_name = "isaacfp_d435_bridge"

    def convert_images(self, rgb_msg, depth_msg):
        if rgb_msg.encoding.lower() not in ("rgb8", "bgr8", "rgba8", "bgra8"):
            raise ValueError(f"Unsupported D435 color encoding: {rgb_msg.encoding}")
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        encoding = depth_msg.encoding.upper()
        if encoding == "16UC1":
            # 本机 RealSense ROS 驱动已将 Z16 深度规范化为毫米；转换成 FP 所需的米。
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
            depth = depth.astype(np.float32) * self.args.depth_scale
        elif encoding == "32FC1":
            # 浮点深度约定为米，不再乘毫米系数。
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough").astype(np.float32)
        else:
            raise ValueError(f"Unsupported D435 depth encoding: {depth_msg.encoding}")
        depth[~np.isfinite(depth) | (depth <= 0)] = 0.0
        return rgb, depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # 必须开启 align_depth.enable，不能将原始深度直接配给彩色内参。
    parser.add_argument("--rgb-topic", default="/camera/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--info-topic", default="/camera/camera/color/camera_info")
    parser.add_argument("--depth-scale", type=float, default=0.001,
                        help="meters per 16UC1 unit; RealSense ROS default: 0.001")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--max-fps", type=float, default=10.0)
    parser.add_argument("--slop", type=float, default=0.05)
    parser.add_argument("--mask-file", help="existing mask PNG; otherwise draw a polygon")
    args = parser.parse_args()
    if not 0 < args.scale <= 1:
        parser.error("--scale must be in (0, 1]")
    if not np.isfinite(args.max_fps) or args.max_fps <= 0:
        parser.error("--max-fps must be finite and positive")
    if not np.isfinite(args.depth_scale) or args.depth_scale <= 0:
        parser.error("--depth-scale must be finite and positive")
    if not np.isfinite(args.slop) or args.slop < 0:
        parser.error("--slop must be finite and nonnegative")
    if args.mask_file and not Path(args.mask_file).is_file():
        parser.error(f"mask file does not exist: {args.mask_file}")
    rclpy.init()
    node = D435Bridge(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
