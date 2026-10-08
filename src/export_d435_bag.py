#!/usr/bin/env python3
"""Export exact-stamp D435 ROS2 sqlite bags to original FP RGB-D sequences."""
import argparse
import csv
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CameraInfo


def stamp(message):
    return message.header.stamp.sec * 10**9 + message.header.stamp.nanosec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    databases = list(args.bag.glob("*.db3"))
    if len(databases) != 1:
        parser.error("This exporter expects one sqlite db3 file per recording")
    if args.output.exists():
        parser.error("Output already exists; choose a new directory to avoid overwriting")
    connection = sqlite3.connect(f"file:{databases[0]}?mode=ro", uri=True)
    topics = {name: index for index, name in connection.execute("SELECT id,name FROM topics")}
    color_id = topics["/camera/camera/color/image_raw"]
    depth_id = topics["/camera/camera/aligned_depth_to_color/image_raw"]
    info_id = topics["/camera/camera/color/camera_info"]
    # 只按源消息时间戳精确配对，不用录包接收时间或成功结果序号配对。
    depths = {}
    duplicate_depth = 0
    for row_id, data in connection.execute(
            "SELECT id,data FROM messages WHERE topic_id=? ORDER BY timestamp,id", (depth_id,)):
        message = deserialize_message(data, Image)
        timestamp = stamp(message)
        if timestamp in depths:
            duplicate_depth += 1
        else:
            depths[timestamp] = row_id
    infos = {}
    first_info = None
    for data, in connection.execute("SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp,id", (info_id,)):
        info = deserialize_message(data, CameraInfo)
        signature = (info.width, info.height, tuple(info.k), tuple(info.d), tuple(info.p), info.distortion_model)
        if first_info is None:
            first_info, calibration = info, signature
        elif signature != calibration:
            raise ValueError("Camera calibration changed within recording")
        infos[stamp(info)] = info
    if first_info is None:
        raise ValueError("No color calibration recorded")
    args.output.mkdir(parents=True)
    for name in ("rgb", "depth", "rgb_unmatched"):
        (args.output / name).mkdir()
    np.savetxt(args.output / "cam_K.txt", np.asarray(first_info.k).reshape(3, 3), fmt="%.12g")
    (args.output / "camera_calibration.json").write_text(json.dumps(dict(
        width=first_info.width, height=first_info.height, k=list(first_info.k),
        d=list(first_info.d), r=list(first_info.r), p=list(first_info.p),
        distortion_model=first_info.distortion_model, frame_id=first_info.header.frame_id), indent=2))
    bridge = CvBridge()
    matched, unmatched = 0, 0
    used_depths = set()
    with (args.output / "frames.csv").open("w", newline="") as paired_file, \
            (args.output / "all_frames.csv").open("w", newline="") as all_file:
        paired = csv.writer(paired_file)
        paired.writerow(["output_index", "source_frame", "timestamp_ns", "rgb_file", "depth_file"])
        all_frames = csv.writer(all_file)
        all_frames.writerow(["source_frame", "timestamp_ns", "bag_timestamp_ns", "status", "rgb_path"])
        for index, (bag_time, data) in enumerate(connection.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp,id", (color_id,))):
            rgb_message = deserialize_message(data, Image)
            timestamp = stamp(rgb_message)
            filename = f"{index:06d}.png"
            info = infos.get(timestamp)
            has_pair = timestamp in depths and info is not None and timestamp not in used_depths
            folder = "rgb" if has_pair else "rgb_unmatched"
            rgb = bridge.imgmsg_to_cv2(rgb_message, "bgr8")
            if not cv2.imwrite(str(args.output / folder / filename), rgb, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OSError("Failed writing RGB")
            if not has_pair:
                unmatched += 1
                reason = ("missing_depth" if timestamp not in depths else
                          "missing_camera_info" if info is None else "duplicate_rgb_stamp")
                all_frames.writerow([index, timestamp, bag_time, reason, f"{folder}/{filename}"])
                continue
            depth_data = connection.execute("SELECT data FROM messages WHERE id=?", (depths[timestamp],)).fetchone()[0]
            depth_message = deserialize_message(depth_data, Image)
            if depth_message.encoding != "16UC1":
                raise ValueError(f"Expected RealSense millimeter 16UC1 depth, got {depth_message.encoding}")
            depth = bridge.imgmsg_to_cv2(depth_message, "passthrough")
            if depth.shape != rgb.shape[:2] or (info.width, info.height) != (rgb.shape[1], rgb.shape[0]):
                raise ValueError("RGB/depth/intrinsics dimensions differ")
            if depth_message.header.frame_id != rgb_message.header.frame_id:
                raise ValueError("Aligned depth and RGB optical frames differ")
            if not cv2.imwrite(str(args.output / "depth" / filename), depth, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise OSError("Failed writing depth")
            paired.writerow([matched, index, timestamp, filename, filename])
            all_frames.writerow([index, timestamp, bag_time, "matched", f"rgb/{filename}"])
            used_depths.add(timestamp)
            matched += 1
            if matched % 100 == 0:
                print(f"{args.bag.name}: exported {matched} exact RGB-D pairs", flush=True)
    summary = dict(bag=str(args.bag), matched_rgb_depth_info=matched,
                   unmatched_rgb=unmatched, unused_depth=len(depths)-len(used_depths),
                   duplicate_depth_stamps=duplicate_depth, depth_unit="millimeters",
                   matching="exact source header timestamp", resolution=[first_info.width, first_info.height])
    (args.output / "export_summary.json").write_text(json.dumps(summary, indent=2))
    connection.close()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
