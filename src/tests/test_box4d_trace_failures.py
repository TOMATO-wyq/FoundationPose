"""Failure traces identify each returned candidate without inventing solutions."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from box4d.config import Config
from box4d.edges import rgb_depth_edges
from box4d.results import empty_result
from box4d.synthetic import Renderer, default_camera, world_box
from box4d.temporal import TemporalPipeline


class TraceFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = Renderer(default_camera())
        cls.cfg = Config(
            up_camera=[0, -1, 0], analyze_colors=True,
            two_color_sizes_m=(.2,), require_target_colors=True,
            temporal_recovery=True, plane_grid_step=8,
        ).validate()
        pipeline = TemporalPipeline(deepcopy(cls.cfg))
        cls.source = cls.render(0)
        cls.detected, _ = pipeline.process(cls.source)
        if len(cls.detected["objects"]) != 1 or not pipeline.states:
            raise AssertionError("real rendered cube must produce a usable measured seed")
        cls.seed = deepcopy(pipeline.states[0])

    @classmethod
    def render(cls, sequence):
        box = world_box((sequence * .002, sequence * .0005, .1), .2, 15. + sequence * .15)
        box["red_axis_local"] = [0, 0, 1]
        return cls.renderer.render([box], sequence, depth_noise_m=.001)[0]

    def inputs(self):
        pipeline = TemporalPipeline(deepcopy(self.cfg))
        current = self.render(1)
        gray = cv2.cvtColor(current.rgb, cv2.COLOR_RGB2GRAY)
        edges = rgb_depth_edges(current, pipeline.cfg, gray=gray)[2]
        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        return pipeline, current, gray, distance

    def assert_failed_trial(self, details, solver, stage, check):
        failed = [trial for trial in details["pose_trials"] if trial["solver"] == solver and not trial["accepted"]]
        self.assertEqual(len(failed), 1)
        trial = failed[0]
        self.assertFalse(trial["selected"])
        self.assertEqual(trial["stage"], stage)
        self.assertIn(check, trial["failed_checks"])
        self.assertTrue(trial["reason"])
        self.assertEqual(
            [item["trial_index"] for item in details["pose_trials"]],
            list(range(len(details["pose_trials"]))),
        )
        self.assertNotIn("NaN", json.dumps(details, allow_nan=False))
        attempt = details["solver_attempts"][trial["solver_attempt_index"]]
        self.assertEqual(attempt["solver"], solver)
        self.assertIn(trial["trial_index"], attempt["candidate_trial_indices"])
        return trial

    def invalid_pose_cases(self):
        pose = self.seed["pose"]
        rv = cv2.Rodrigues(pose[:3, :3])[0]
        tv = pose[:3, 3].reshape(3, 1)
        normal = pose[:3, :3] @ self.seed["normal_model"]
        axis = np.cross(normal, np.array([1., 0., 0.]))
        if np.linalg.norm(axis) < .01:
            axis = np.cross(normal, np.array([0., 1., 0.]))
        opposite = cv2.Rodrigues(axis / np.linalg.norm(axis) * np.pi)[0] @ pose[:3, :3]
        ray = tv.ravel() / np.linalg.norm(tv)
        tangent = np.cross(ray, np.array([0., 1., 0.]))
        tangent /= np.linalg.norm(tangent)
        grazing_normal = -.15 * ray + np.sqrt(1 - .15**2) * tangent
        axis = np.cross(normal, grazing_normal)
        angle = np.arctan2(np.linalg.norm(axis), normal @ grazing_normal)
        grazing = cv2.Rodrigues(axis / np.linalg.norm(axis) * angle)[0] @ pose[:3, :3]
        return {
            "nonfinite_rotation": (np.full((3, 1), np.nan), tv, "parameters", "nonfinite_pose_parameters"),
            "nonfinite_translation": (rv, np.full((3, 1), np.inf), "parameters", "nonfinite_pose_parameters"),
            "malformed_rotation": (np.zeros(2), tv, "parameters", "invalid_pose_solution"),
            "behind_camera": (rv, np.array([[0.], [0.], [-1.]]), "pose_geometry", "behind_camera"),
            "back_facing": (cv2.Rodrigues(opposite)[0], tv, "pose_geometry", "back_facing"),
            "grazing_geometry": (cv2.Rodrigues(grazing)[0], tv, "pose_geometry", "grazing_geometry"),
        }

    def test_parameter_and_early_geometry_rejections_have_complete_candidate_records(self):
        for name, (rv, tv, stage, check) in self.invalid_pose_cases().items():
            with self.subTest(case=name):
                pipeline, current, gray, distance = self.inputs()
                details = {}
                with patch("box4d.temporal.cv2.solvePnPGeneric", return_value=(0, (), (), None)), patch(
                    "box4d.temporal.cv2.solvePnP", return_value=(True, rv, tv)
                ):
                    obj, _ = pipeline._advance(current, deepcopy(self.seed), gray, distance, diagnostics=details)
                self.assert_failed_trial(details, "iterative", stage, check)
                self.assertEqual(obj["metrics"]["pose_solver"], "joint")
                self.assertEqual(len(details["pose_trials"]), 2)

    def test_each_ippe_returned_branch_has_its_own_trial(self):
        pipeline, current, gray, distance = self.inputs()
        rv = cv2.Rodrigues(self.seed["pose"][:3, :3])[0]
        tv = self.seed["pose"][:3, 3].reshape(3, 1)
        details = {}
        with patch("box4d.temporal.cv2.solvePnPGeneric", return_value=(
            2, (np.full((3, 1), np.nan), rv), (tv, np.array([[0.], [0.], [-1.]])), None
        )):
            pipeline._advance(current, deepcopy(self.seed), gray, distance, diagnostics=details)
        ippe_trials = [trial for trial in details["pose_trials"] if trial["solver"] == "ippe"]
        self.assertEqual([trial["candidate_index"] for trial in ippe_trials], [0, 1])
        self.assertEqual([trial["failed_checks"] for trial in ippe_trials], [["nonfinite_pose_parameters"], ["behind_camera"]])
        self.assertEqual(len({trial["trial_index"] for trial in details["pose_trials"]}), len(details["pose_trials"]))
        attempt = next(item for item in details["solver_attempts"] if item["solver"] == "ippe")
        self.assertEqual(attempt["candidate_trial_indices"], [trial["trial_index"] for trial in ippe_trials])
        json.dumps(details, allow_nan=False)

    def test_nonfinite_image_residual_and_boundary_projection_are_traced_without_nan(self):
        for corrupt_call, stage, check in (
            (2, "current_pose_residuals", "nonfinite_pose_residuals"),
            (3, "boundary_validation", "nonfinite_projected_face"),
        ):
            with self.subTest(stage=stage):
                pipeline, current, gray, distance = self.inputs()
                rv = cv2.Rodrigues(self.seed["pose"][:3, :3])[0]
                tv = self.seed["pose"][:3, 3].reshape(3, 1)
                project = current.camera.project
                calls = 0

                def corrupted_projection(points):
                    nonlocal calls
                    calls += 1
                    value = project(points)
                    return np.full_like(value, np.nan) if calls == corrupt_call else value

                details = {}
                with patch.object(current.camera, "project", side_effect=corrupted_projection), patch(
                    "box4d.temporal.cv2.solvePnPGeneric", return_value=(0, (), (), None)
                ), patch("box4d.temporal.cv2.solvePnP", return_value=(True, rv, tv)):
                    pipeline._advance(current, deepcopy(self.seed), gray, distance, diagnostics=details)
                self.assert_failed_trial(details, "iterative", stage, check)

    def test_solver_no_solution_creates_an_attempt_and_no_fake_candidate(self):
        pipeline, current, gray, distance = self.inputs()
        details = {}
        with patch("box4d.temporal.cv2.solvePnPGeneric", return_value=(0, (), (), None)), patch(
            "box4d.temporal.cv2.solvePnP", return_value=(False, None, None)
        ):
            pipeline._advance(current, deepcopy(self.seed), gray, distance, diagnostics=details)
        self.assertEqual([trial["solver"] for trial in details["pose_trials"]], ["joint"])
        for attempt in details["solver_attempts"][:2]:
            self.assertEqual(attempt["status"], "no_solution")
            self.assertEqual(attempt["candidate_trial_indices"], [])
        json.dumps(details, allow_nan=False)

    def test_solver_exceptions_and_joint_failure_do_not_invent_candidate_trials(self):
        pipeline, current, gray, distance = self.inputs()
        details = {}
        with patch("box4d.temporal.cv2.solvePnPGeneric", side_effect=cv2.error("IPPE failed")), patch(
            "box4d.temporal.cv2.solvePnP", side_effect=cv2.error("iterative failed")
        ), patch.object(pipeline, "_joint_pose", side_effect=ValueError("joint failed")):
            with self.assertRaises(ValueError):
                pipeline._advance(current, deepcopy(self.seed), gray, distance, diagnostics=details)
        self.assertEqual(details.get("pose_trials", []), [])
        self.assertEqual([item["solver"] for item in details["solver_attempts"]], ["ippe", "iterative", "joint"])
        for attempt in details["solver_attempts"]:
            self.assertEqual(attempt["status"], "solver_error")
            self.assertTrue(attempt["reason"])
            self.assertEqual(attempt["candidate_trial_indices"], [])
        self.assertEqual(len(details["pose_solver_errors"]), 3)
        json.dumps(details, allow_nan=False)

    def test_seed_event_is_cleared_after_advance_but_origin_is_preserved(self):
        pipeline, current, gray, distance = self.inputs()
        state = deepcopy(self.seed)
        state["obj"].update(
            detector_seed_stamp_ns=self.source.stamp_ns,
            detector_confirmed=True,
            detector_match_action="face_switch",
            origin_detection_stamp_ns=self.source.stamp_ns - 1,
        )
        obj, advanced = pipeline._advance(current, state, gray, distance)
        for value in (obj, advanced["obj"]):
            self.assertNotIn("detector_seed_stamp_ns", value)
            self.assertNotIn("detector_confirmed", value)
            self.assertNotIn("detector_match_action", value)
            self.assertEqual(value["origin_detection_stamp_ns"], self.source.stamp_ns - 1)
        self.assertEqual(advanced["stamp"], current.stamp_ns)
        json.dumps(obj, allow_nan=False)

    def test_async_acquisition_separates_source_seed_event_and_current_observation(self):
        pipeline, current, _, _ = self.inputs()
        pipeline.asynchronous = True
        pipeline.worker_completed = (self.source, deepcopy(self.detected), [], 20., (None, ()))
        result, _ = pipeline.process(current)
        self.assertEqual(len(result["objects"]), 1)
        obj = result["objects"][0]
        self.assertEqual(obj["detector_seed_stamp_ns"], self.source.stamp_ns)
        self.assertEqual(obj["origin_detection_stamp_ns"], self.source.stamp_ns)
        self.assertEqual(result["stamp_ns"], current.stamp_ns)
        check = result["temporal"]["detector_seed_checks"][0]
        self.assertEqual(check["source_stamp_ns"], self.source.stamp_ns)
        self.assertEqual(check["current_stamp_ns"], current.stamp_ns)
        self.assertTrue(check["accepted"])
        next_result, _ = pipeline.process(self.render(2))
        self.assertEqual(len(next_result["objects"]), 1)
        self.assertNotIn("detector_seed_stamp_ns", next_result["objects"][0])
        self.assertEqual(next_result["objects"][0]["origin_detection_stamp_ns"], self.source.stamp_ns)
        json.dumps(next_result, allow_nan=False)

    def test_trial_identifiers_are_local_and_complete_scope_keys_are_unique(self):
        pipeline, current, _, _ = self.inputs()
        pipeline.states = [deepcopy(self.seed), deepcopy(self.seed)]
        pipeline.last_detection = self.source.stamp_ns
        pipeline.last_stamp = self.source.stamp_ns
        pipeline.reference = (current.camera.calibration_id, tuple(pipeline.cfg.up_camera))
        scoped = []
        unscoped = []
        for frame in (current, self.render(2)):
            with patch.object(pipeline, "_full_detect", return_value=(empty_result(frame, pipeline.cfg), None, [], (None, ()))):
                result, _ = pipeline.process(frame)
            checks = result["temporal"]["tracking_checks"]
            for index, check in enumerate(checks):
                self.assertEqual(check["check_index"], index)
                for trial in check["pose_trials"]:
                    scoped.append((result["stamp_ns"], "tracking_checks", index, trial["trial_index"]))
                    unscoped.append(trial["trial_index"])
            for obj in result["objects"]:
                metrics = obj["metrics"]
                group, index = metrics["tracking_check_group"], metrics["tracking_check_index"]
                trial = result["temporal"][group][index]["pose_trials"][metrics["pose_trial_index"]]
                self.assertTrue(trial["accepted"] and trial["selected"])
            json.dumps(result, allow_nan=False)
        self.assertEqual(len(scoped), len(set(scoped)))
        self.assertLess(len(set(unscoped)), len(unscoped))


if __name__ == "__main__":
    unittest.main()
