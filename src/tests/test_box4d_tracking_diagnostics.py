"""Tracking diagnostics describe measured data without changing pose selection."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from box4d.config import Config
from box4d.edges import rgb_depth_edges
from box4d.geometry import Plane
from box4d.results import empty_result
from box4d.synthetic import Renderer, default_camera, world_box
from box4d.temporal import TemporalPipeline


def vector_angle_degrees(a, b):
    """Independent atan2 form also behaves well for nearly identical vectors."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


class TrackingDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = Renderer(default_camera())
        cls.cfg = Config(
            up_camera=[0, -1, 0],
            analyze_colors=True,
            two_color_sizes_m=(0.2,),
            require_target_colors=True,
            temporal_recovery=True,
            plane_grid_step=8,
        )
        pipeline = TemporalPipeline(deepcopy(cls.cfg))
        cls.source, _, _ = cls.frame(0)
        result, _ = pipeline.process(cls.source)
        if len(result["objects"]) != 1 or not pipeline.states:
            raise AssertionError("real rendered cube must produce a usable measured track")
        cls.detected = deepcopy(result)
        cls.seed = deepcopy(pipeline.states[0])

    @classmethod
    def frame(cls, sequence):
        box = world_box((sequence * 0.002, sequence * 0.0005, 0.1), 0.2, 15.0 + sequence * 0.15)
        box["red_axis_local"] = [0, 0, 1]
        return cls.renderer.render([box], sequence, depth_noise_m=0.001)

    def inputs(self):
        pipeline = TemporalPipeline(deepcopy(self.cfg))
        frame, truth, _ = self.frame(1)
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        edges = rgb_depth_edges(frame, pipeline.cfg, gray=gray)[2]
        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        return pipeline, frame, gray, distance

    def assert_trial_output_agrees(self, obj, details):
        trials = details["pose_trials"]
        self.assertEqual([trial["trial_index"] for trial in trials], list(range(len(trials))))
        for trial in trials:
            self.assertIsInstance(trial["accepted"], bool)
            self.assertIsInstance(trial["selected"], bool)
        selected = [trial for trial in trials if trial["selected"]]
        self.assertEqual(len(selected), 1)
        chosen = selected[0]
        self.assertTrue(chosen["accepted"])
        self.assertFalse(chosen["failed_checks"])
        self.assertEqual(details["chosen_pose_trial"], chosen["trial_index"])
        self.assertEqual(details["chosen_pose_solver"], chosen["solver"])
        self.assertEqual(obj["metrics"]["pose_solver"], chosen["solver"])
        self.assertEqual(obj["metrics"]["pose_trial_index"], chosen["trial_index"])
        accepted = [trial for trial in trials if trial["accepted"]]
        self.assertEqual(chosen["selection_cost"], min(trial["selection_cost"] for trial in accepted))
        np.testing.assert_allclose(obj["center_camera_m"], chosen["center_camera_m"], atol=1e-12)
        np.testing.assert_allclose(obj["faces"][0]["normal_camera"], chosen["normal_camera"], atol=1e-12)
        return chosen

    def test_successful_candidates_have_unique_selection_and_match_output(self):
        pipeline, frame, gray, distance = self.inputs()
        details = {}
        obj, state = pipeline._advance(frame, deepcopy(self.seed), gray, distance, diagnostics=details)
        self.assert_trial_output_agrees(obj, details)
        self.assertEqual(state["stamp"], frame.stamp_ns)
        # Requesting annotations must have no influence on the selected pose.
        other, _ = pipeline._advance(frame, deepcopy(self.seed), gray, distance)
        np.testing.assert_allclose(obj["T_camera_from_box"], other["T_camera_from_box"], atol=1e-12)

    def test_rejected_trial_is_explicit_and_cannot_be_selected(self):
        pipeline, frame, gray, distance = self.inputs()
        details = {}
        # Corrupt only ITERATIVE. Real IPPE still measures the current target.
        rv = cv2.Rodrigues(self.seed["pose"][:3, :3])[0]
        tv = self.seed["pose"][:3, 3].reshape(3, 1) + np.array([[0.2], [0.0], [0.0]])
        with patch("box4d.temporal.cv2.solvePnP", return_value=(True, rv, tv)):
            obj, _ = pipeline._advance(frame, deepcopy(self.seed), gray, distance, diagnostics=details)
        self.assert_trial_output_agrees(obj, details)
        first = details["pose_trials"][0]
        self.assertEqual(first["solver"], "iterative")
        self.assertFalse(first["accepted"])
        self.assertFalse(first["selected"])
        self.assertIn("feature_reprojection", first["failed_checks"])
        json.dumps(details, allow_nan=False)

    def test_candidate_order_does_not_override_a_better_measurement(self):
        pipeline, frame, gray, distance = self.inputs()
        baseline, _ = pipeline._advance(frame, deepcopy(self.seed), gray, distance)
        pose = np.asarray(baseline["T_camera_from_box"])
        good_rv = cv2.Rodrigues(pose[:3, :3])[0]
        tilt = cv2.Rodrigues(np.array([np.deg2rad(0.8), 0.0, 0.0]))[0]
        worse_rv = cv2.Rodrigues(tilt @ pose[:3, :3])[0]
        tv = pose[:3, 3].reshape(3, 1)
        outputs = []
        for iterative_rv, ippe_rv in ((worse_rv, good_rv), (good_rv, worse_rv)):
            details = {}
            with patch("box4d.temporal.cv2.solvePnP", return_value=(True, iterative_rv, tv)), patch(
                "box4d.temporal.cv2.solvePnPGeneric", return_value=(1, (ippe_rv,), (tv,), None)
            ):
                obj, _ = pipeline._advance(frame, deepcopy(self.seed), gray, distance, diagnostics=details)
            self.assert_trial_output_agrees(obj, details)
            self.assertEqual(sum(trial["accepted"] for trial in details["pose_trials"]), 2)
            outputs.append(np.asarray(obj["T_camera_from_box"]))
        np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-12)
        np.testing.assert_allclose(outputs[0][:3, :3], pose[:3, :3], atol=1e-12)

    def test_measured_plane_normal_and_step_are_independent_of_pose_normal(self):
        pipeline, frame, gray, distance = self.inputs()
        original_sample = pipeline._sample_tracking_plane
        measured = []

        def tilted_measurement(*args, **kwargs):
            samples, fraction, plane = original_sample(*args, **kwargs)
            rotation = cv2.Rodrigues(np.array([np.deg2rad(0.8), 0.0, 0.0]))[0]
            normal = rotation @ plane.normal
            raw = Plane(normal, -float(normal @ samples.mean(axis=0)), plane.rmse_m, plane.support)
            measured.append(raw)
            return samples, fraction, raw

        details = {}
        seed = deepcopy(self.seed)
        with patch.object(pipeline, "_sample_tracking_plane", side_effect=tilted_measurement):
            obj, state = pipeline._advance(frame, seed, gray, distance, diagnostics=details)
        raw = measured[0]
        expected_step = vector_angle_degrees(raw.normal, seed["measured_plane_normal"])
        np.testing.assert_allclose(details["measured_plane_normal_camera"], raw.normal, atol=1e-12)
        np.testing.assert_allclose(obj["metrics"]["measured_plane_normal_camera"], raw.normal, atol=1e-12)
        np.testing.assert_allclose(state["measured_plane_normal"], raw.normal, atol=1e-12)
        self.assertAlmostEqual(details["measured_plane_normal_step_deg"], expected_step, places=8)
        self.assertAlmostEqual(obj["metrics"]["measured_plane_normal_step_deg"], expected_step, places=8)
        self.assertAlmostEqual(details["measured_plane_d_m"], raw.d)
        self.assertGreater(vector_angle_degrees(raw.normal, obj["faces"][0]["normal_camera"]), 0.2)
        # Mutable fitting arrays must not alter the next frame's reference.
        before = state["measured_plane_normal"].copy()
        raw.normal[:] = 0
        np.testing.assert_array_equal(state["measured_plane_normal"], before)

    def test_reseed_compares_measurement_without_forcing_small_normal_noise_into_pose(self):
        pipeline, frame, gray, distance = self.inputs()
        obj, _ = pipeline._advance(frame, deepcopy(self.seed), gray, distance)
        original_sample = pipeline._sample_tracking_plane
        observed = []

        def tilted_measurement(*args, **kwargs):
            samples, fraction, plane = original_sample(*args, **kwargs)
            rotation = cv2.Rodrigues(np.array([np.deg2rad(0.8), 0.0, 0.0]))[0]
            normal = rotation @ plane.normal
            observed.append(normal.copy())
            return (
                samples,
                fraction,
                Plane(normal, -float(normal @ samples.mean(axis=0)), plane.rmse_m, plane.support),
            )

        with patch.object(pipeline, "_sample_tracking_plane", side_effect=tilted_measurement):
            state = pipeline._seed(frame, obj, gray)
        self.assertIsNotNone(state)
        np.testing.assert_allclose(state["measured_plane_normal"], observed[-1], atol=1e-12)
        model_normal = state["pose"][:3, :3] @ state["normal_model"]
        self.assertEqual(state["obj"]["metrics"]["reseed_geometry_source"], "current_pose")
        np.testing.assert_allclose(state["pose"], obj["T_camera_from_box"], atol=1e-12)
        self.assertGreater(vector_angle_degrees(state["measured_plane_normal"], model_normal), 0.2)

    def test_confirmation_fields_are_events_and_do_not_survive_another_advance(self):
        pipeline, frame, gray, distance = self.inputs()
        seed = deepcopy(self.seed)
        fields = ("detector_confirmed", "detector_center_difference_m", "detector_confirmation_stamp_ns")
        seed["obj"].update(
            detector_confirmed=True,
            detector_center_difference_m=0.023,
            detector_confirmation_stamp_ns=self.source.stamp_ns,
        )
        obj, state = pipeline._advance(frame, seed, gray, distance)
        for key in fields:
            self.assertNotIn(key, obj)
            self.assertNotIn(key, state["obj"])

    def test_process_logs_successful_trials_without_debug_and_strict_json(self):
        pipeline, frame, gray, distance = self.inputs()
        pipeline.states = [deepcopy(self.seed)]
        pipeline.reference = (frame.camera.calibration_id, tuple(pipeline.cfg.up_camera))
        pipeline.last_stamp = self.source.stamp_ns
        pipeline.last_detection = self.source.stamp_ns
        with patch.object(
            pipeline,
            "_full_detect",
            return_value=(empty_result(frame, pipeline.cfg), None, [], (None, ())),
        ):
            result, _ = pipeline.process(frame)
        self.assertFalse(pipeline.debug_enabled)
        self.assertEqual(len(result["objects"]), 1)
        checks = result["temporal"]["tracking_checks"]
        self.assertEqual(len(checks), 1)
        self.assertTrue(checks[0]["accepted"])
        self.assertEqual(checks[0]["previous_stamp_ns"], self.source.stamp_ns)
        obj = result["objects"][0]
        self.assertEqual(obj["metrics"]["tracking_check_group"], "tracking_checks")
        self.assertEqual(obj["metrics"]["tracking_check_index"], 0)
        linked = result["temporal"][obj["metrics"]["tracking_check_group"]][
            obj["metrics"]["tracking_check_index"]
        ]
        self.assertIs(linked, checks[0])
        self.assert_trial_output_agrees(obj, linked)
        # This is the same strict serialization needed by poses.jsonl.
        decoded = json.loads(json.dumps(result, allow_nan=False))
        self.assertTrue(decoded["temporal"]["tracking_checks"][0]["pose_trials"])

    def test_async_detector_seed_links_to_its_current_validation_trials(self):
        pipeline, current, gray, distance = self.inputs()
        source_gray = cv2.cvtColor(self.source.rgb, cv2.COLOR_RGB2GRAY)
        expected_seed = pipeline._seed(self.source, deepcopy(self.detected["objects"][0]), source_gray)
        self.assertIsNotNone(expected_seed)
        # Exercise the real completed-result handoff without creating a thread.
        pipeline.asynchronous = True
        pipeline.worker_completed = (self.source, deepcopy(self.detected), [], 20.0, (None, ()))
        result, _ = pipeline.process(current)
        self.assertEqual(len(result["objects"]), 1)
        self.assertEqual(result["temporal"]["tracking_checks"], [])
        seed_checks = result["temporal"]["detector_seed_checks"]
        self.assertEqual(len(seed_checks), 1)
        obj = result["objects"][0]
        metrics = obj["metrics"]
        self.assertEqual(metrics["tracking_check_group"], "detector_seed_checks")
        self.assertEqual(metrics["tracking_check_index"], 0)
        linked = result["temporal"][metrics["tracking_check_group"]][metrics["tracking_check_index"]]
        self.assertIs(linked, seed_checks[0])
        self.assertTrue(linked["accepted"])
        self.assertEqual(linked["check_index"], 0)
        self.assertEqual(linked["source_stamp_ns"], self.source.stamp_ns)
        self.assertEqual(linked["current_stamp_ns"], current.stamp_ns)
        self.assertEqual(obj["detector_seed_stamp_ns"], self.source.stamp_ns)
        self.assertEqual(result["stamp_ns"], current.stamp_ns)
        np.testing.assert_allclose(linked["seed_camera_from_box"], expected_seed["pose"], atol=1e-12)
        np.testing.assert_allclose(
            linked["seed_measured_plane_normal_camera"], expected_seed["measured_plane_normal"], atol=1e-12
        )
        self.assert_trial_output_agrees(obj, linked)
        decoded = json.loads(json.dumps(result, allow_nan=False))
        persisted = decoded["temporal"][metrics["tracking_check_group"]][metrics["tracking_check_index"]]
        self.assertEqual(persisted["current_stamp_ns"], current.stamp_ns)
        self.assertTrue(persisted["pose_trials"][persisted["chosen_pose_trial"]]["selected"])


if __name__ == "__main__":
    unittest.main()
