"""Motion priors learn from real measurements and expire without fabricating them."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from box4d.config import Config
from box4d.motion import PositionKalman


class PositionKalmanTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config().validate()

    def learned(self, cfg=None):
        cfg = self.cfg if cfg is None else cfg
        motion = PositionKalman([0, 0, 1], 0, cfg)
        for index in range(1, 21):
            motion = motion.updated([index * 0.01, 0, 1], index * 50_000_000)
        return motion

    def test_constant_velocity_prediction_across_missing_frames_uses_real_time(self):
        cfg = Config(prediction_accel_std_m_s2=0.05, prediction_measurement_std_m=0.002).validate()
        origin = np.array([0.2, -0.1, 1.3])
        velocity = np.array([0.25, -0.1, 0.06])
        motion = PositionKalman(origin, 4_000_000_000, cfg)
        elapsed_ns = 0
        for index in range(30):
            elapsed_ns += (20_000_000, 47_000_000, 63_000_000)[index % 3]
            motion = motion.updated(origin + velocity * (elapsed_ns / 1e9), 4_000_000_000 + elapsed_ns)
        stamp = motion.stamp_ns + 240_000_000
        expected = origin + velocity * ((elapsed_ns + 240_000_000) / 1e9)
        predicted, covariance = motion.predict(stamp)
        np.testing.assert_allclose(predicted[:3], expected, atol=0.002)
        np.testing.assert_allclose(predicted[3:], velocity, atol=0.003)
        self.assertEqual(covariance.shape, (6, 6))
        self.assertTrue(motion.can_predict(stamp))
        # Last-position hold would lag this movement by more than six cm.
        self.assertGreater(np.linalg.norm(predicted[:3] - motion.predict(motion.stamp_ns)[0][:3]), 0.06)

    def test_predictions_do_not_bootstrap_motion_or_refresh_measurement_age(self):
        motion = PositionKalman([0, 0, 1], 100, self.cfg)
        for offset in (0, 50_000_000, 100_000_000):
            motion.predict(100 + offset)
            self.assertFalse(motion.can_predict(100 + offset))
        self.assertEqual(motion.stamp_ns, 100)
        self.assertEqual(motion.observations, 1)
        learned = motion.updated([0.01, 0, 1], 50_000_100)
        self.assertEqual(learned.observations, 2)
        self.assertTrue(learned.can_predict(learned.stamp_ns))
        self.assertEqual(motion.observations, 1)

    def test_covariance_grows_with_missing_observations_and_limits_prediction(self):
        cfg = Config(prediction_max_std_m=0.015, prediction_accel_std_m_s2=4).validate()
        motion = self.learned(cfg)
        deviations = []
        for offset in (0, 50_000_000, 100_000_000, 200_000_000):
            _, covariance = motion.predict(motion.stamp_ns + offset)
            deviations.append(np.sqrt(np.linalg.eigvalsh(covariance[:3, :3])[-1]))
        self.assertTrue(np.all(np.diff(deviations) > 0))
        self.assertTrue(motion.can_predict(motion.stamp_ns))
        self.assertFalse(motion.can_predict(motion.stamp_ns + 200_000_000))

    def test_prediction_age_limit_uses_last_measurement_not_previous_prediction(self):
        motion = self.learned(Config(prediction_max_std_m=1).validate())
        self.assertTrue(motion.can_predict(motion.stamp_ns + 300_000_000))
        motion.predict(motion.stamp_ns + 290_000_000)
        self.assertFalse(motion.can_predict(motion.stamp_ns + 300_000_001))

    def test_behind_camera_prediction_is_unusable(self):
        motion = PositionKalman([0, 0, 0.03], 0, self.cfg)
        motion = motion.updated([0, 0, 0.01], 100_000_000)
        self.assertTrue(motion.can_predict(motion.stamp_ns))
        self.assertLess(motion.predict(200_000_000)[0][2], 0)
        self.assertFalse(motion.can_predict(200_000_000))

    def test_update_and_returned_arrays_cannot_contaminate_previous_state(self):
        position = np.array([0.0, 0.0, 1.0])
        motion = PositionKalman(position, 0, self.cfg)
        original = motion.predict(0)
        position[:] = 99
        updated = motion.updated([0.02, 0, 1], 100_000_000)
        updated.predict(updated.stamp_ns)[0][:] = -100
        mean, covariance = motion.predict(0)
        mean[:] = -100
        covariance[:] = -100
        for actual, expected in zip(motion.predict(0), original):
            np.testing.assert_array_equal(actual, expected)
        self.assertEqual(motion.stamp_ns, 0)
        self.assertEqual(motion.observations, 1)
        self.assertEqual(updated.stamp_ns, 100_000_000)
        self.assertEqual(updated.observations, 2)
        self.assertGreater(updated.predict(updated.stamp_ns)[0][0], 0.01)

    def test_joseph_update_remains_finite_symmetric_and_psd_with_unequal_scales(self):
        cfg = Config(prediction_measurement_std_m=1e-7,
                     prediction_initial_velocity_std_m_s=1000,
                     prediction_accel_std_m_s2=0.2).validate()
        origin = np.array([10000.0, -10000.0, 10000.0])
        velocity = np.array([0.1, -0.2, 0.3])
        motion = PositionKalman(origin, 0, cfg)
        random = np.random.default_rng(4)
        stamp = 0
        for index in range(600):
            stamp += (1, 100_000_000, 1_000_000)[index % 3]
            motion = motion.updated(origin + velocity * (stamp / 1e9) + random.normal(0, 1e-7, 3), stamp)
            mean, covariance = motion.predict(stamp)
            self.assertTrue(np.isfinite(mean).all())
            self.assertTrue(np.isfinite(covariance).all())
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-14, rtol=0)
            self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-12)
        np.testing.assert_allclose(mean[:3], origin + velocity * (stamp / 1e9), atol=1e-6)

    def test_repeated_and_reversed_observations_are_rejected(self):
        motion = PositionKalman([0, 0, 1], 100, self.cfg)
        np.testing.assert_array_equal(motion.predict(100)[0], [0, 0, 1, 0, 0, 0])
        for stamp in (99, 100):
            with self.subTest(stamp=stamp):
                with self.assertRaises(ValueError):
                    motion.updated([0, 0, 1], stamp)
        with self.assertRaises(ValueError):
            motion.predict(99)
        self.assertFalse(motion.can_predict(99))

    def test_invalid_timestamps_are_rejected_and_integer_nanoseconds_preserved(self):
        motion = PositionKalman([0, 0, 1], np.int64(100), self.cfg)
        for stamp in (-1, 100.0, np.nan, np.inf, True, np.bool_(False), None, "100"):
            with self.subTest(stamp=stamp):
                with self.assertRaises(ValueError):
                    PositionKalman([0, 0, 1], stamp, self.cfg)
                with self.assertRaises(ValueError):
                    motion.predict(stamp)
                with self.assertRaises(ValueError):
                    motion.updated([0, 0, 1], stamp)
                self.assertFalse(motion.can_predict(stamp))
        motion = PositionKalman([0, 0, 1], 1_800_000_000_000_000_000, self.cfg)
        motion = motion.updated([0, 0, 1], motion.stamp_ns + 1)
        self.assertEqual(motion.stamp_ns, 1_800_000_000_000_000_001)

    def test_invalid_measurements_do_not_change_estimate(self):
        motion = self.learned()
        original = motion.predict(motion.stamp_ns)
        for position in ([0, np.nan, 1], [0, 0, np.inf], [0, 1], [[0, 0, 1]], None):
            with self.subTest(position=position):
                with self.assertRaises(ValueError):
                    motion.updated(position, motion.stamp_ns + 1)
        for actual, expected in zip(motion.predict(motion.stamp_ns), original):
            np.testing.assert_array_equal(actual, expected)


class MotionConfigTests(unittest.TestCase):
    def test_prediction_defaults_are_inherited_by_existing_configs_and_saved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text("{}")
            cfg = Config.load(path)
            self.assertTrue(cfg.motion_prediction)
            self.assertEqual(cfg.prediction_max_gap_s, 0.3)
            self.assertEqual(cfg.prediction_max_std_m, 0.1)
            path.write_text(json.dumps(cfg.to_dict()))
            self.assertEqual(json.loads(json.dumps(Config.load(path).to_dict())),
                             json.loads(json.dumps(cfg.to_dict())))

    def test_motion_switch_requires_boolean(self):
        self.assertFalse(Config(motion_prediction=False).validate().motion_prediction)
        for value in (0, 1, "false", None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "motion_prediction"):
                    Config(motion_prediction=value).validate()

    def test_motion_noise_and_limits_require_finite_positive_nonboolean_numbers(self):
        fields = ("prediction_max_gap_s", "prediction_max_std_m", "prediction_accel_std_m_s2",
                  "prediction_measurement_std_m", "prediction_initial_velocity_std_m_s")
        for field in fields:
            for value in (0, -1, np.nan, np.inf, True, False, None, "0.1"):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, field):
                        Config(**{field: value}).validate()


if __name__ == "__main__":
    unittest.main()
