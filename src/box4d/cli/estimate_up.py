#!/usr/bin/env python3
"""Estimate physical UP from stationary D435i acceleration and the actual ROS TF."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

# Keep direct invocation of this organized entry point equivalent to -m.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from box4d.config import Config
from box4d.io import dump_json
from box4d.reference import stationary_up


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "config/box4d_tower.json")
    p.add_argument("--topic", default="/camera/camera/accel/sample")
    p.add_argument("--target-frame", default="camera_color_optical_frame")
    p.add_argument("--duration", type=float, default=2.)
    p.add_argument("--timeout", type=float, default=10.)
    p.add_argument("--note", default="Factory IMU calibration status must be checked in the camera driver log.")
    args = p.parse_args()
    if args.duration < 1 or args.timeout <= args.duration:
        p.error("duration must be >=1 second and timeout longer than duration")
    if args.output.exists():
        p.error("output directory already exists")
    cfg = Config.load(args.config)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = Node("box4d_stationary_up")
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    samples, stamps = [], []
    source_frames = set()
    first_received = None

    def receive(msg):
        nonlocal first_received
        stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        if stamps and stamp <= stamps[-1]:
            return
        source_frames.add(msg.header.frame_id)
        a = msg.linear_acceleration
        samples.append([a.x, a.y, a.z])
        stamps.append(stamp)
        if first_received is None:
            first_received = time.monotonic()

    subscription = node.create_subscription(Imu, args.topic, receive, qos_profile_sensor_data)
    begin = time.monotonic()
    try:
        while time.monotonic() - begin < args.timeout:
            rclpy.spin_once(node, timeout_sec=.05)
            if first_received is not None and time.monotonic() - first_received >= args.duration:
                break
        if len(source_frames) != 1 or not next(iter(source_frames), ""):
            raise ValueError("missing or changing acceleration optical frame")
        if len(stamps) < 50 or (stamps[-1] - stamps[0]) / 1e9 < args.duration * .8:
            raise ValueError("insufficient independent acceleration samples/time span")
        source_frame = next(iter(source_frames))
        tf = buffer.lookup_transform(args.target_frame, source_frame, rclpy.time.Time()).transform
        q = tf.rotation
        quaternion = [q.x, q.y, q.z, q.w]
        rotation = Rotation.from_quat(quaternion).as_matrix()
        up, quality = stationary_up(samples, rotation)
        report = dict(status="stationary_estimate", source="accelerometer_with_ros_tf", camera_frame_id=args.target_frame,
                      accel_frame_id=source_frame, up_camera=up, sample_span_s=(stamps[-1] - stamps[0]) / 1e9,
                      camera_from_accel_quaternion_xyzw=quaternion, camera_from_accel_rotation=rotation.tolist(),
                      first_stamp_ns=stamps[0], last_stamp_ns=stamps[-1], quality=quality,
                      driver_note=args.note, valid_only_while_camera_mount_is_unchanged=True)
        cfg.up_camera = up
        cfg.validate()
        args.output.mkdir(parents=True)
        dump_json(args.output / "up_reference.json", report)
        dump_json(args.output / "config.json", cfg.to_dict())
        np.savez(args.output / "accel_samples.npz", accel_m_s2=np.asarray(samples), stamp_ns=np.asarray(stamps, np.int64))
        print(f"UP in {args.target_frame}: {up}; RMS acceleration scatter: {quality['rms_scatter_m_s2']:.4f}m/s^2")
        print(f"Provisional reference saved: {args.output}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
