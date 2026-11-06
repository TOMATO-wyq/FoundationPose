"""Nonblocking, fresh physical-UP reference for the RGB-D bridge."""
from collections import deque

import numpy as np

from common.imu import stationary_up


def rotation_from_quaternion(q):
    values = np.array([q.x, q.y, q.z, q.w], float)
    if not np.isfinite(values).all() or np.linalg.norm(values) < 1e-8:
        raise ValueError('invalid IMU-to-color quaternion')
    x, y, z, w = values / np.linalg.norm(values)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class LiveUpReference:
    def __init__(self, node, topic):
        from sensor_msgs.msg import Imu
        from rclpy.qos import qos_profile_sensor_data
        from tf2_ros import Buffer, TransformListener
        self.samples = deque(maxlen=1000)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node)
        self.subscription = node.create_subscription(Imu, topic, self.receive, qos_profile_sensor_data)

    def receive(self, message):
        stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
        if self.samples and stamp <= self.samples[-1][0]:
            return
        a = message.linear_acceleration
        self.samples.append((stamp, message.header.frame_id, [a.x, a.y, a.z]))

    def for_frame(self, target_frame, stamp):
        from rclpy.time import Time
        rows = [row for row in self.samples if stamp-1.2 <= row[0] <= stamp+.03]
        result = dict(status='waiting_for_imu', up_camera=None, camera_frame_id=target_frame)
        if not rows or stamp - rows[-1][0] > .25:
            return result
        if len(rows) < 50 or rows[-1][0] - rows[0][0] < .6:
            result['status'] = 'collecting_stationary_imu'
            return result
        frames = {row[1] for row in rows}
        if len(frames) != 1 or not next(iter(frames)):
            result['status'] = 'imu_frame_changed'
            return result
        source_frame = rows[-1][1]
        # The default zero timeout must stay zero: image delivery never waits for TF.
        try:
            transform = self.buffer.lookup_transform(target_frame, source_frame, Time()).transform
        except Exception:
            result['status'] = 'imu_transform_unavailable'
            return result
        try:
            rotation = rotation_from_quaternion(transform.rotation)
            up, quality = stationary_up([row[2] for row in rows], rotation)
        except ValueError:
            result['status'] = 'camera_moving_or_bad_imu'
            return result
        result.update(status='ok', up_camera=up, accel_frame_id=source_frame,
                      last_imu_stamp=rows[-1][0], rms_scatter_m_s2=quality['rms_scatter_m_s2'])
        return result
