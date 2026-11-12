"""Source-timed gyro rotation for a rigid camera observing a static scene.

R_reference_from_camera stabilizes axes only: its origin still follows the
camera. Translation is learned from vision, never by integrating acceleration.
"""

from bisect import bisect_left
from collections import deque
import json
from pathlib import Path

import cv2
import numpy as np


def rotation_matrix(value):
    rotation = np.asarray(value, dtype=np.float64)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)):
        raise ValueError("camera motion rotation must be a finite proper rotation")
    return rotation


def motion_context(frame, cfg):
    return frame.camera_motion if cfg.camera_rotation_compensation else None


def context_rotation(context):
    return np.eye(3) if context is None else np.asarray(context["rotation_reference_from_camera"])


def context_id(context):
    return None if context is None else context["reference_id"]


def relative_camera_rotation(previous, current):
    if previous is None or current is None or context_id(previous) != context_id(current):
        return None
    return context_rotation(current).T @ context_rotation(previous)


class GyroRotations:
    """Integrate body angular rates; interpolate only within complete segments.

    Callers own synchronization. A gap starts a new reference, so stale motion
    estimates cannot be mixed with the new integrated axes. No extrapolation.
    """

    def __init__(self, camera_from_gyro, *, max_gap_s=.03, bias=(0., 0., 0.),
                 time_offset_s=0., capacity=4096):
        self.extrinsic = rotation_matrix(camera_from_gyro)
        self.bias = np.asarray(bias, dtype=np.float64)
        if self.bias.shape != (3,) or not np.isfinite(self.bias).all():
            raise ValueError("gyro bias must be three finite rad/s values")
        if not np.isfinite(max_gap_s) or max_gap_s <= 0 or not np.isfinite(time_offset_s):
            raise ValueError("invalid gyro timing settings")
        self.max_gap_ns = int(max_gap_s * 1e9)
        self.offset_ns = int(time_offset_s * 1e9)
        self.rows = deque(maxlen=capacity)
        self.rejected = 0
        self.gaps = 0

    def add(self, stamp_ns, angular_velocity):
        if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, (int, np.integer)):
            raise ValueError("gyro source stamp must be integer nanoseconds")
        stamp = int(stamp_ns) + self.offset_ns
        raw = np.asarray(angular_velocity, dtype=np.float64)
        if stamp < 0 or raw.shape != (3,) or not np.isfinite(raw).all():
            raise ValueError("gyro source stamp and rad/s vector must be finite and valid")
        rate = self.extrinsic @ (raw - self.bias)
        if self.rows and stamp <= self.rows[-1][0]:
            self.rejected += 1
            return False
        if not self.rows or stamp - self.rows[-1][0] > self.max_gap_ns:
            self.gaps += bool(self.rows)
            rotation = np.eye(3)
            reference = f"gyro:{stamp}"
        else:
            previous, previous_rate, previous_rotation, reference = self.rows[-1]
            step = .5 * (previous_rate + rate) * ((stamp - previous) / 1e9)
            rotation = previous_rotation @ cv2.Rodrigues(step)[0]
        self.rows.append((stamp, rate, rotation, reference))
        return True

    def at(self, stamp_ns):
        rows = list(self.rows)
        index = bisect_left([row[0] for row in rows], stamp_ns)
        if index < len(rows) and rows[index][0] == stamp_ns:
            _, _, rotation, reference = rows[index]
        elif index == 0 or index == len(rows):
            return None
        else:
            first, second = rows[index - 1], rows[index]
            if first[3] != second[3]:
                return None
            fraction = (stamp_ns - first[0]) / (second[0] - first[0])
            step = cv2.Rodrigues(first[2].T @ second[2])[0] * fraction
            rotation, reference = first[2] @ cv2.Rodrigues(step)[0], first[3]
        return dict(reference_id=reference, rotation_reference_from_camera=rotation.tolist(),
                    source="gyro_integration", stamp_ns=int(stamp_ns))


def recorded_rotations(path, camera_frame, cfg):
    """Load existing native IMU sidecars; ROS imports stay optional and local."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu

    path = Path(path)
    reference = json.loads((path / "imu_reference.json").read_text())
    if reference["camera_frame_id"] != camera_frame:
        raise ValueError("IMU sidecar does not describe this RGB optical frame")
    topic = cfg.camera_gyro_topic
    extrinsic = reference["extrinsics"][topic]
    if extrinsic["target_frame_id"] != camera_frame:
        raise ValueError("gyro extrinsic target does not match RGB optical frame")
    gyro = GyroRotations(extrinsic["rotation"], max_gap_s=cfg.camera_gyro_max_gap_s,
                         bias=cfg.camera_gyro_bias_rad_s, time_offset_s=cfg.camera_gyro_time_offset_s,
                         capacity=None)
    reader = rosbag2_py.SequentialReader()
    bag = (path / reference["bag"]).resolve()
    if not bag.is_relative_to(path.resolve()):
        raise ValueError("IMU sidecar path escapes dataset")
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=reference["storage_id"]),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    while reader.has_next():
        name, payload, _ = reader.read_next()
        if name == topic:
            message = deserialize_message(payload, Imu)
            if message.header.frame_id != extrinsic["source_frame_id"]:
                raise ValueError("gyro source frame changed in recording")
            if message.angular_velocity_covariance[0] < 0:
                continue
            stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            w = message.angular_velocity
            gyro.add(stamp, [w.x, w.y, w.z])
    return gyro
