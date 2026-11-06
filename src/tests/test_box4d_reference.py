"""Sign/frame conventions and rejection conditions for stationary acceleration."""
import unittest
import numpy as np
from box4d.reference import stationary_up


class StationaryReferenceTests(unittest.TestCase):
    def test_camera_up_uses_actual_imu_rotation_without_sign_flip(self):
        # IMU +Z is physical up; RGB optical -Y is physical up.
        rotation = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
        samples = np.tile([0., 0., 9.80665], (100, 1))
        up, quality = stationary_up(samples, rotation)
        np.testing.assert_allclose(up, [0, -1, 0], atol=1e-10)
        self.assertFalse(quality["absolute_accuracy_validated"])

    def test_motion_bad_units_and_too_few_samples_rejected(self):
        moving = np.tile([0., -9.8, 0], (100, 1))
        moving[::2, 0] = 1.
        for values in (moving, np.tile([0., -1., 0], (100, 1)), moving[:10]):
            with self.assertRaises(ValueError):
                stationary_up(values, np.eye(3))

    def test_reflection_not_accepted_as_sensor_rotation(self):
        with self.assertRaises(ValueError):
            stationary_up(np.tile([0., -9.8, 0], (100, 1)), np.diag([1., -1, 1.]))
