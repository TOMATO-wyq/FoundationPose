"""Gravity metadata remains fresh and TF lookup never waits for the camera."""
from collections import deque
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from live_up_reference import LiveUpReference


class LiveUpReferenceTests(unittest.TestCase):
    def reference(self, acceleration=(0, -9.8, 0)):
        reference = LiveUpReference.__new__(LiveUpReference)
        reference.samples = deque((99.+i*.01, 'accel_optical', list(acceleration)) for i in range(101))
        reference.buffer = Mock()
        reference.buffer.lookup_transform.return_value = NS(transform=NS(rotation=NS(x=0., y=0., z=0., w=1.)))
        return reference

    def current(self, reference, stamp=100.):
        with patch.dict('sys.modules', {'rclpy.time': NS(Time=lambda: 'latest')}):
            return reference.for_frame('color_optical', stamp)

    def test_fresh_up_is_transformed_to_rgb_frame_without_wait(self):
        reference = self.reference()
        angle = np.sqrt(.5)
        reference.buffer.lookup_transform.return_value.transform.rotation = NS(x=0, y=0, z=angle, w=angle)
        report = self.current(reference)
        self.assertEqual(report['status'], 'ok')
        np.testing.assert_allclose(report['up_camera'], [1, 0, 0], atol=1e-8)
        self.assertEqual(reference.buffer.lookup_transform.call_args.args,
                         ('color_optical', 'accel_optical', 'latest'))
        self.assertEqual(reference.buffer.lookup_transform.call_args.kwargs, {})

    def test_old_up_is_not_reused_after_camera_or_imu_stops(self):
        reference = self.reference()
        self.assertEqual(self.current(reference)['status'], 'ok')
        self.assertIsNone(self.current(reference, 100.5)['up_camera'])
        reference.samples.clear()
        self.assertIsNone(self.current(reference)['up_camera'])

    def test_motion_tf_failure_and_changed_frames_are_rejected(self):
        reference = self.reference()
        reference.samples = deque((t, frame, [i % 2, -9.8, 0])
                                  for i, (t, frame, _) in enumerate(reference.samples))
        self.assertEqual(self.current(reference)['status'], 'camera_moving_or_bad_imu')
        reference = self.reference()
        reference.buffer.lookup_transform.side_effect = RuntimeError('no transform')
        self.assertIsNone(self.current(reference)['up_camera'])
        reference = self.reference()
        reference.samples.append((100.01, 'other_imu', [0, -9.8, 0]))
        self.assertEqual(self.current(reference)['status'], 'imu_frame_changed')

    def test_duplicate_stamp_cannot_create_enough_independent_samples(self):
        reference = self.reference()
        reference.samples.clear()
        message = NS(header=NS(stamp=NS(sec=100, nanosec=0), frame_id='accel'),
                     linear_acceleration=NS(x=0, y=-9.8, z=0))
        for _ in range(100):
            reference.receive(message)
        self.assertEqual(len(reference.samples), 1)
        self.assertIsNone(self.current(reference)['up_camera'])


if __name__ == '__main__':
    unittest.main()
