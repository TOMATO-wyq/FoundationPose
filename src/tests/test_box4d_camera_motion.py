"""Rigid-camera geometry, gyro timing, and acceleration prediction regressions."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from collections import deque
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from box4d.camera_motion import GyroRotations
from box4d.config import Config
from box4d.io import Dataset, DatasetWriter
from box4d.motion import PositionKalman
from box4d.results import empty_result
from box4d.ros_input import RosSource
from box4d.synthetic import Renderer, default_camera, world_box
from box4d.temporal import TemporalPipeline
from box4d.tracking import Tracker


def context(stamp, angle=0., reference="gyro:test"):
    return dict(stamp_ns=stamp, reference_id=reference,
                rotation_reference_from_camera=cv2.Rodrigues(np.array([0., angle, 0.]))[0].tolist())


class GyroGeometryTests(unittest.TestCase):
    def test_camera_yaw_sign_and_nontrivial_imu_extrinsic(self):
        extrinsic = cv2.Rodrigues(np.array([0., 0., np.pi / 2]))[0]
        gyro = GyroRotations(extrinsic)
        for stamp in range(0, 1_000_000_001, 10_000_000):
            gyro.add(stamp, [1., 0., 0.])  # gyro x becomes camera y
        rotation = np.asarray(gyro.at(1_000_000_000)["rotation_reference_from_camera"])
        np.testing.assert_allclose(rotation, cv2.Rodrigues(np.array([0., 1., 0.]))[0], atol=1e-12)
        # A stationary point goes LEFT when the camera turns RIGHT.
        point = rotation.T @ [0., 0., 2.]
        self.assertLess(point[0], 0)
        middle = np.asarray(gyro.at(505_000_000)["rotation_reference_from_camera"])
        np.testing.assert_allclose(middle, cv2.Rodrigues(np.array([0., .505, 0.]))[0], atol=1e-12)

    def test_gap_duplicate_time_offset_bias_and_no_extrapolation(self):
        gyro = GyroRotations(np.eye(3), bias=[0., .1, 0.], time_offset_s=.005)
        gyro.add(10_000_000, [0., .1, 0.])
        gyro.add(20_000_000, [0., .1, 0.])
        self.assertFalse(gyro.add(20_000_000, [0., 100., 0.]))
        old = gyro.at(20_000_000)
        np.testing.assert_array_equal(old["rotation_reference_from_camera"], np.eye(3))
        self.assertIsNone(gyro.at(14_999_999))
        self.assertIsNone(gyro.at(25_000_001))
        gyro.add(100_000_000, [0., .1, 0.])
        self.assertIsNone(gyro.at(50_000_000))
        self.assertNotEqual(old["reference_id"], gyro.at(105_000_000)["reference_id"])
        self.assertEqual(gyro.gaps, 1)
        self.assertEqual(gyro.rejected, 1)

    def test_invalid_rotations_and_nonfinite_vectors_fail_explicitly(self):
        with self.assertRaises(ValueError):
            GyroRotations(-np.eye(3))
        gyro = GyroRotations(np.eye(3))
        for stamp, vector in ((0, [np.nan, 0, 0]), (-1, [0, 0, 0]), (1., [0, 0, 0])):
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                gyro.add(stamp, vector)


class StabilizedMotionTests(unittest.TestCase):
    def test_static_target_does_not_acquire_camera_rotation_as_translation_velocity(self):
        position = np.array([.2, .1, 2.])
        for model in ("cv", "ca"):
            with self.subTest(model=model):
                cfg = Config(prediction_model=model).validate()
                motion = PositionKalman(position, 0, cfg, context(0))
                for i in range(1, 31):
                    stamp = i * 20_000_000
                    ctx = context(stamp, .5 * (i / 30.)**2)
                    rotation = np.asarray(ctx["rotation_reference_from_camera"])
                    motion = motion.updated(rotation.T @ position, stamp, ctx)
                ctx = context(stamp + 200_000_000, .8)
                prediction, covariance = motion.predict(ctx["stamp_ns"], ctx)
                expected = np.asarray(ctx["rotation_reference_from_camera"]).T @ position
                np.testing.assert_allclose(prediction[:3], expected, atol=1e-10)
                np.testing.assert_allclose(prediction[3:6], 0., atol=1e-10)
                self.assertTrue(motion.can_predict(ctx["stamp_ns"], ctx))
                np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
                self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-12)
                self.assertFalse(motion.can_predict(ctx["stamp_ns"], None))
                self.assertFalse(motion.can_predict(ctx["stamp_ns"], context(ctx["stamp_ns"], reference="new")))

    def test_ca_follows_accelerating_relative_translation_better_than_cv(self):
        acceleration = np.array([.8, 0., -.2])
        origin = np.array([0., 0., 2.])
        errors = {}
        for model in ("cv", "ca"):
            cfg = Config(prediction_model=model, prediction_accel_std_m_s2=1.,
                         prediction_measurement_std_m=.002).validate()
            motion = PositionKalman(origin, 0, cfg)
            for i in range(1, 61):
                t = i * .02
                motion = motion.updated(origin + .5 * acceleration * t*t, i * 20_000_000)
            t = 1.4
            mean, _ = motion.predict(1_400_000_000)
            errors[model] = np.linalg.norm(mean[:3] - (origin + .5 * acceleration * t*t))
            if model == "ca":
                np.testing.assert_allclose(mean[6:9], acceleration, atol=.01)
        self.assertLess(errors["ca"], .003)
        self.assertLess(errors["ca"], errors["cv"] / 5.)

    def test_rotation_covariance_is_transformed_into_current_camera_axes(self):
        cfg = Config().validate()
        motion = PositionKalman([0, 0, 2], 0, cfg, context(0))
        motion._covariance[:3, :3] = np.diag([.01, .02, .03])
        ctx = context(0, np.pi / 2)
        _, covariance = motion.predict(0, ctx)
        np.testing.assert_allclose(covariance[:3, :3], np.diag([.03, .02, .01]), atol=1e-12)


class RotatingCameraPipelineTests(unittest.TestCase):
    def test_delayed_detector_seed_uses_gyro_with_detector_age_budget(self):
        cfg = Config(camera_rotation_compensation=True, track_max_gap_s=.3, detector_max_age_s=.75)
        pipe = TemporalPipeline(cfg)
        first, _, _ = Renderer(default_camera(64, 48)).render([])
        last = replace(first, stamp_ns=first.stamp_ns + 500_000_000,
                       camera_motion=context(first.stamp_ns + 500_000_000, .2))
        pose = np.eye(4)
        pose[:3, 3] = [0, 0, 2]
        state = dict(pose=pose, stamp=first.stamp_ns, motion=None, camera_motion=context(first.stamp_ns))
        self.assertIsNone(pipe._predicted_pose(last, state))
        predicted = pipe._predicted_pose(last, state, max_gap_s=cfg.detector_max_age_s)
        rotation = np.asarray(last.camera_motion["rotation_reference_from_camera"])
        np.testing.assert_allclose(predicted[:3, :], rotation.T @ pose[:3, :])
        self.assertIsNone(pipe._predicted_pose(last, state, max_gap_s=.49))

    def test_static_big_cube_tracks_rotation_then_recovers_after_depth_loss(self):
        cfg = Config(up_camera=[0, -1, 0], temporal_recovery=True,
                     camera_rotation_compensation=True, track_max_distance_m=.02).validate()
        pipe, tracker = TemporalPipeline(cfg), Tracker(cfg)
        renderer = Renderer(default_camera())
        original = world_box((0, 0, .175), .35, 15.)
        observations = []
        for index in range(14):
            stamp = 1_000_000_000 + index * 16_666_667
            ctx = context(stamp, .004 * index * index / 2.)
            rotation = np.asarray(ctx["rotation_reference_from_camera"])
            box = deepcopy(original)
            box["pose"][:3, :] = rotation.T @ original["pose"][:3, :]
            frame, truth, _ = renderer.render([box], index, depth_noise_m=.001)
            frame = replace(frame, camera_motion=ctx)
            if index in (9, 10):
                frame.depth[:] = 0
            if index == 0:
                result, _ = pipe.process(frame)
            else:
                # Test actual LK/PnP/depth tracking, with no detector rescuing it.
                with patch.object(pipe, "_full_detect", return_value=(empty_result(frame, cfg), None, [], (None, ()))):
                    result, _ = pipe.process(frame)
            tracker.update(result)
            observations.append(result)
            if index in (9, 10):
                self.assertEqual(result["objects"], [])
                self.assertEqual(len(result["predicted_objects"]), 1)
                self.assertFalse(result["predicted_objects"][0]["valid"])
                continue
            self.assertEqual(len(result["objects"]), 1, result["temporal"])
            np.testing.assert_allclose(result["objects"][0]["center_camera_m"], truth[0]["center_camera_m"], atol=.012)
            if index:
                self.assertTrue(result["temporal"]["tracking_checks"][0]["camera_rotation_used"])
        identity = observations[0]["objects"][0]["track_id"]
        self.assertTrue(all(not r["objects"] or r["objects"][0]["track_id"] == identity for r in observations))
        # A missing gyro context resets the learned coordinate reference safely.
        old = pipe.states[0]
        raw_frame = replace(frame, camera_motion=None, stamp_ns=frame.stamp_ns + 1)
        motion = pipe._motion_observation(raw_frame, old["pose"], old)
        self.assertEqual(motion.observations, 1)
        self.assertIsNone(motion.camera_motion)

    def test_saved_rotations_round_trip_without_ros_and_check_source_stamp(self):
        frame, _, _ = Renderer(default_camera(64, 48)).render([])
        frame = replace(frame, camera_motion=context(frame.stamp_ns, .1))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data"
            writer = DatasetWriter(path, frame.camera)
            writer.write(frame)
            writer.close()
            data = Dataset(path)
            self.assertEqual(data.frame(0).camera_motion, frame.camera_motion)
            report = data.enable_camera_motion(Config(camera_rotation_compensation=True))
            self.assertEqual(report["source"], "saved_frame_rotations")
        with self.assertRaises(ValueError):
            replace(frame, camera_motion=context(frame.stamp_ns + 1))


class LiveGyroTests(unittest.TestCase):
    def source(self):
        source = RosSource.__new__(RosSource)
        source.motion_config = Config(camera_rotation_compensation=True).validate()
        source.gyro_condition = threading.Condition()
        source.gyro_rows = deque()
        source.gyro_rotation = None
        source.gyro_origin = source.gyro_target = None
        source.rotation_frames = source.rotation_missing = 0
        source.node = Mock()
        source.rclpy = NS(time=NS(Time=Mock()))
        source.imu_buffer = Mock()
        source.imu_buffer.lookup_transform.return_value = NS(transform=NS(rotation=NS(x=0., y=0., z=0., w=1.)))
        return source

    def test_live_attach_uses_tf_source_timing_and_falls_back_on_gyro_stall(self):
        source = self.source()
        frame, _, _ = Renderer(default_camera(64, 48)).render([])
        frame = replace(frame, stamp_ns=1_005_000_000)
        source.gyro_rows.extend([(1_000_000_000, "gyro", [0., 1., 0.]),
                                (1_010_000_000, "gyro", [0., 1., 0.])])
        with patch.dict(sys.modules, {'tf2_ros': NS(TransformException=RuntimeError)}):
            compensated = source._attach_camera_motion(frame)
            self.assertEqual(compensated.stamp_ns, frame.stamp_ns)
            self.assertEqual(compensated.camera_motion["stamp_ns"], frame.stamp_ns)
            rotation = np.asarray(compensated.camera_motion["rotation_reference_from_camera"])
            np.testing.assert_allclose(rotation, cv2.Rodrigues(np.array([0., .005, 0.]))[0], atol=1e-12)
            source.imu_buffer.lookup_transform.assert_called_once()
            self.assertIs(source._attach_camera_motion(replace(frame, stamp_ns=2_000_000_000)).camera_motion, None)
            self.assertEqual(source.rotation_missing, 1)
            self.assertEqual(source.rotation_frames, 1)

    def test_missing_extrinsic_does_not_fabricate_identity_camera_rotation(self):
        source = self.source()
        frame, _, _ = Renderer(default_camera(64, 48)).render([])
        source.gyro_rows.append((frame.stamp_ns, "gyro", [0., 1., 0.]))
        source.imu_buffer.lookup_transform.side_effect = RuntimeError("no fixed transform")
        with patch.dict(sys.modules, {'tf2_ros': NS(TransformException=RuntimeError)}):
            self.assertIsNone(source._attach_camera_motion(frame).camera_motion)
        self.assertIn("no fixed transform", source.gyro_error)


class NewMotionConfigTests(unittest.TestCase):
    def test_model_and_gyro_configuration_validation(self):
        Config(prediction_model="ca", camera_rotation_compensation=True).validate()
        bad = [dict(prediction_model="ekf"), dict(camera_rotation_compensation=1),
               dict(camera_gyro_bias_rad_s=[0, np.nan, 0]), dict(camera_gyro_topic=""),
               dict(camera_gyro_time_offset_s=True), dict(camera_gyro_time_offset_s=np.inf)]
        for name in ("prediction_jerk_std_m_s3", "prediction_initial_accel_std_m_s2", "camera_gyro_max_gap_s"):
            bad.extend({name: value} for value in (0, -1, True, None, np.inf))
        for options in bad:
            with self.subTest(options=options), self.assertRaises(ValueError):
                Config(**options).validate()


if __name__ == "__main__":
    unittest.main()
