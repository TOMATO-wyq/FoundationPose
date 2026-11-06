#!/usr/bin/env python3
"""Record synchronized D435i RGB-D for offline development; bounded writer queue."""
import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

# Keep direct invocation of this organized entry point equivalent to -m.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from box4d.ros_input import AsyncRecorder, RosSource
from box4d.geometry import unit
from box4d.imu_capture import ImuBagRecorder


class CapturePreview:
    """Rate-limited view of the submitted RGB-D; never change recorded arrays."""

    name = "Box4D RGB-D recording - Q / Esc to finish"

    def __init__(self, hz=15.0):
        self.interval = 1.0 / hz
        self.next_show = 0.0
        self.opened = False

    def show(self, frame, submitted, saved, dropped, elapsed, duration):
        now = time.monotonic()
        if now < self.next_show:
            return False
        self.next_show = now + self.interval

        rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        valid = np.isfinite(frame.depth) & (frame.depth > 0)
        scaled = np.zeros(frame.depth.shape, dtype=np.uint8)
        scaled[valid] = (np.minimum(frame.depth[valid], 3.0) * (255.0 / 3.0)).astype(np.uint8)
        depth = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        depth[~valid] = 0
        panels = np.hstack((rgb, depth))
        banner = np.full((56, panels.shape[1], 3), 24, dtype=np.uint8)
        cv2.putText(banner, f"REC {elapsed:.1f}/{duration:g}s  Saved {saved} / submitted {submitted}"
                    f"  Writer drops {dropped}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (60, 240, 80) if not dropped else (40, 180, 255), 1, cv2.LINE_AA)
        cv2.putText(banner, "Left: RGB   Right: depth 0..3m (black: no depth)   Q / Esc: finish and save",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        image = np.vstack((banner, panels))
        if not self.opened:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            self.opened = True
            width = min(960, image.shape[1])
            cv2.resizeWindow(self.name, width, round(image.shape[0] * width / image.shape[1]))
        cv2.imshow(self.name, image)
        if (cv2.waitKey(1) & 255) in (ord("q"), ord("Q"), 27):
            return True
        try:
            return cv2.getWindowProperty(self.name, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            # Some HighGUI backends remove the window before visibility can be queried.
            return True

    def close(self):
        if self.opened:
            try:
                cv2.destroyWindow(self.name)
            except cv2.error:
                pass  # The user may already have closed this window.
            self.opened = False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--topic", default="/camera/camera/rgbd")
    p.add_argument("--duration", type=float, default=5.)
    p.add_argument("--timeout", type=float, default=10.)
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--up-camera", type=float, nargs=3)
    p.add_argument("--max-sync-ms", type=float, default=0.)
    p.add_argument("--depth-scale", type=float, default=.001, help="meters per 16UC1 unit; 32FC1 is already meters")
    p.add_argument("--rectified-input", action="store_true")
    p.add_argument("--display", action="store_true", help="preview synchronized RGB and depth while recording")
    p.add_argument("--display-hz", type=float, default=15., help="preview refresh rate; recording is not throttled")
    p.add_argument("--record-imu", action="store_true", help="also record timestamped accel, gyro and ROS TF in imu_bag")
    p.add_argument("--accel-topic", "--imu-topic", default="/camera/camera/accel/sample")
    p.add_argument("--gyro-topic", default="/camera/camera/gyro/sample")
    args = p.parse_args()
    if args.duration <= 0 or args.timeout <= 0 or args.max_frames < 1 or args.max_sync_ms < 0:
        p.error("invalid recording limits")
    if not np.isfinite(args.display_hz) or args.display_hz <= 0:
        p.error("--display-hz must be positive and finite")
    up_camera = None if args.up_camera is None else unit(args.up_camera).tolist()
    source = RosSource(args.topic, args.depth_scale, args.max_sync_ms, args.rectified_input)
    recorder = None
    imu_recorder = None
    imu_summary = None
    preview = CapturePreview(args.display_hz) if args.display else None
    count, complete = 0, False
    submitted_stamps = [None, None]
    try:
        frame = source.get(timeout=args.timeout)
        recorder = AsyncRecorder(args.output, frame.camera, up_camera,
                                  dict(source="ros2", topic=args.topic, max_sync_ms=args.max_sync_ms,
                                       depth_scale=args.depth_scale, rectified_input=args.rectified_input,
                                       imu_recording=args.record_imu,
                                       imu_bag="imu_bag" if args.record_imu else None))
        if args.record_imu:
            imu_recorder = ImuBagRecorder(source, args.output, accel_topic=args.accel_topic,
                                          gyro_topic=args.gyro_topic)
            imu_recorder.start(timeout=args.timeout)
            # Motion recording is ready before the first submitted RGB-D frame.
            frame = source.get(frame.sequence, args.timeout)
            print("RGB-D + accel/gyro recording ready. Keep the camera still for the first 3 seconds,"
                  " then move it. IMU data is saved separately in imu_bag.", flush=True)
        begin, sequence = time.monotonic(), -1
        while time.monotonic() - begin < args.duration and count < args.max_frames:
            if count:
                frame = source.get(sequence, args.timeout)
            if imu_recorder:
                imu_recorder.check()
            recorder.submit(frame)
            if count == 0:
                submitted_stamps[0] = frame.stamp_ns
            submitted_stamps[1] = frame.stamp_ns
            sequence = frame.sequence
            count += 1
            if preview and preview.show(frame, count, recorder.writer.count, recorder.dropped,
                                        time.monotonic() - begin, args.duration):
                break
        complete = True
    except KeyboardInterrupt:
        complete = count > 0
    finally:
        try:
            try:
                if imu_recorder:
                    imu_summary = imu_recorder.close()
                    complete = complete and imu_summary["complete"]
            except BaseException:
                complete = False
                raise
            finally:
                if recorder:
                    metadata = dict(camera=source.statistics(),
                                    submitted_rgbd_stamp_range_ns=submitted_stamps)
                    if args.record_imu:
                        metadata["imu"] = imu_summary or dict(complete=False)
                    recorder.close(complete, **metadata)
                    print(f"Saved {recorder.writer.count} frames; bounded-writer drops={recorder.dropped}")
                    if imu_summary and imu_summary["complete"]:
                        print(f"IMU bag and reference saved: {args.output / 'imu_bag'}")
        finally:
            try:
                source.close()
            finally:
                if preview:
                    preview.close()


if __name__ == "__main__":
    main()
