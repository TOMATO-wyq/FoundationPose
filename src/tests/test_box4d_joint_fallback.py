"""Joint tracking recovery still passes real current RGB-D validation."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from box4d.config import Config
from box4d.edges import rgb_depth_edges
from box4d.results import empty_result
from box4d.synthetic import Renderer, default_camera, world_box
from box4d.temporal import TemporalPipeline


class JointFallbackTests(unittest.TestCase):
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
        box = world_box((0, 0, 0.1), 0.2, 15.0)
        box["red_axis_local"] = [0, 0, 1]
        cls.source = cls.renderer.render([box], 0, depth_noise_m=0.001)[0]
        result, _ = pipeline.process(cls.source)
        if len(result["objects"]) != 1 or not pipeline.states:
            raise AssertionError("rendered target must seed a measured track")
        cls.seed = deepcopy(pipeline.states[0])

    def inputs(self):
        pipeline = TemporalPipeline(deepcopy(self.cfg))
        box = world_box((0.002, 0.0005, 0.1), 0.2, 15.15)
        box["red_axis_local"] = [0, 0, 1]
        current, truth, _ = self.renderer.render([box], 1, depth_noise_m=0.001)
        gray = cv2.cvtColor(current.rgb, cv2.COLOR_RGB2GRAY)
        edges = rgb_depth_edges(current, pipeline.cfg, gray=gray)[2]
        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        return pipeline, current, truth[0], gray, distance

    def assert_current_measurement(self, obj, state, current, truth):
        self.assertEqual(state["stamp"], current.stamp_ns)
        self.assertEqual(obj["source"], "current_rgbd_flow")
        self.assertTrue(np.isfinite(obj["T_camera_from_box"]).all())
        np.testing.assert_allclose(obj["center_camera_m"], truth["center_camera_m"], atol=0.012)
        metrics = obj["metrics"]
        self.assertLessEqual(metrics["reprojection_rms_px"], metrics["reprojection_limit_px"])
        self.assertLessEqual(metrics["depth_median_residual_m"], self.cfg.max_depth_residual_m)
        self.assertLessEqual(metrics["normal_error_deg"], self.cfg.max_normal_error_deg)
        self.assertGreaterEqual(metrics["edge_support"], self.cfg.min_edge_support)

    def test_regular_success_does_not_attempt_joint(self):
        pipeline, current, truth, gray, distance = self.inputs()
        details = {}
        with patch.object(pipeline, "_joint_pose", wraps=pipeline._joint_pose) as joint:
            obj, state = pipeline._advance(current, deepcopy(self.seed), gray, distance, diagnostics=details)
        joint.assert_not_called()
        self.assertFalse(details["joint_pose_attempted"])
        self.assert_current_measurement(obj, state, current, truth)

    def test_zero_ippe_solutions_still_accepts_final_iterative_candidate_without_joint(self):
        pipeline, current, truth, gray, distance = self.inputs()
        details = {}
        with patch("box4d.temporal.cv2.solvePnPGeneric", return_value=(0, (), (), None)), patch.object(
            pipeline, "_joint_pose", wraps=pipeline._joint_pose
        ) as joint:
            obj, state = pipeline._advance(current, deepcopy(self.seed), gray, distance, diagnostics=details)
        joint.assert_not_called()
        self.assertFalse(details["joint_pose_attempted"])
        self.assertEqual([trial["solver"] for trial in details["pose_trials"]], ["iterative"])
        self.assert_current_measurement(obj, state, current, truth)

    def test_both_pnp_solvers_empty_runs_real_joint_recovery_once(self):
        pipeline, current, truth, gray, distance = self.inputs()
        details = {}
        with patch("box4d.temporal.cv2.solvePnPGeneric", return_value=(0, (), (), None)), patch(
            "box4d.temporal.cv2.solvePnP", return_value=(False, None, None)
        ), patch.object(pipeline, "_joint_pose", wraps=pipeline._joint_pose) as joint:
            obj, state = pipeline._advance(current, deepcopy(self.seed), gray, distance, diagnostics=details)
        joint.assert_called_once()
        self.assertTrue(details["joint_pose_attempted"])
        self.assertEqual([trial["solver"] for trial in details["pose_trials"]], ["joint"])
        self.assert_current_measurement(obj, state, current, truth)

    def test_both_pnp_solvers_throw_still_runs_real_joint_recovery_once(self):
        pipeline, current, truth, gray, distance = self.inputs()
        details = {}
        with patch("box4d.temporal.cv2.solvePnPGeneric", side_effect=cv2.error("forced IPPE failure")), patch(
            "box4d.temporal.cv2.solvePnP", side_effect=cv2.error("forced iterative failure")
        ), patch.object(pipeline, "_joint_pose", wraps=pipeline._joint_pose) as joint:
            obj, state = pipeline._advance(current, deepcopy(self.seed), gray, distance, diagnostics=details)
        joint.assert_called_once()
        self.assertEqual([item["solver"] for item in details["pose_solver_errors"]], ["ippe", "iterative"])
        self.assert_current_measurement(obj, state, current, truth)

    def test_joint_failure_rejects_instead_of_issuing_old_pose(self):
        pipeline, current, _, gray, distance = self.inputs()
        details = {}
        with patch("box4d.temporal.cv2.solvePnPGeneric", return_value=(0, (), (), None)), patch(
            "box4d.temporal.cv2.solvePnP", return_value=(False, None, None)
        ), patch.object(pipeline, "_joint_pose", side_effect=ValueError("joint fit failed")) as joint:
            with self.assertRaises(ValueError):
                pipeline._advance(current, deepcopy(self.seed), gray, distance, diagnostics=details)
        joint.assert_called_once()
        self.assertTrue(details["joint_pose_attempted"])
        self.assertEqual(details["pose_solver_errors"], [dict(solver="joint", reason="joint fit failed")])
        self.assertEqual(self.seed["stamp"], self.source.stamp_ns)

    def test_all_regular_candidates_are_checked_before_joint(self):
        pipeline, current, truth, gray, distance = self.inputs()
        # Reject ITERATIVE first; the real last IPPE candidate must get its full
        # validation before the expensive recovery decision.
        with patch(
            "box4d.temporal.cv2.solvePnP", return_value=(True, np.zeros((3, 1)), np.zeros((3, 1)))
        ), patch.object(pipeline, "_joint_pose", wraps=pipeline._joint_pose) as joint:
            obj, state = pipeline._advance(current, deepcopy(self.seed), gray, distance)
        joint.assert_not_called()
        self.assert_current_measurement(obj, state, current, truth)

    def test_rejected_regular_candidates_then_run_real_joint_once(self):
        pipeline, current, truth, gray, distance = self.inputs()
        rv = cv2.Rodrigues(self.seed["pose"][:3, :3])[0]
        tv = self.seed["pose"][:3, 3].reshape(3, 1) + np.array([[0.5], [0.0], [0.0]])
        details = {}
        with patch("box4d.temporal.cv2.solvePnPGeneric", return_value=(1, (rv,), (tv,), None)), patch(
            "box4d.temporal.cv2.solvePnP", return_value=(False, None, None)
        ), patch.object(pipeline, "_joint_pose", wraps=pipeline._joint_pose) as joint:
            obj, state = pipeline._advance(current, deepcopy(self.seed), gray, distance, diagnostics=details)
        joint.assert_called_once()
        self.assertEqual([trial["solver"] for trial in details["pose_trials"]], ["ippe", "joint"])
        self.assertIn("feature_reprojection", details["pose_trials"][0]["failed_checks"])
        self.assertEqual(details["pose_trials"][1]["failed_checks"], [])
        self.assert_current_measurement(obj, state, current, truth)

    def test_joint_bad_data_or_geometry_never_publishes_a_pose(self):
        good_rv = cv2.Rodrigues(self.seed["pose"][:3, :3])[0]
        good_tv = self.seed["pose"][:3, 3].reshape(3, 1)
        cases = {
            "nonfinite_rotation": (np.full((3, 1), np.nan), good_tv),
            "nonfinite_translation": (good_rv, np.full((3, 1), np.inf)),
            "malformed_rotation": (np.zeros(2), good_tv),
            "malformed_translation": (good_rv, None),
            "behind_camera": (good_rv, np.array([[0.0], [0.0], [-1.0]])),
            "wrong_image_pose": (good_rv, good_tv + np.array([[0.5], [0.0], [0.0]])),
        }
        for kind, candidate in cases.items():
            with self.subTest(kind=kind):
                pipeline, current, _, _, _ = self.inputs()
                pipeline.states = [deepcopy(self.seed)]
                pipeline.last_stamp = self.source.stamp_ns
                pipeline.reference = (current.camera.calibration_id, tuple(pipeline.cfg.up_camera))
                pipeline.last_detection = self.source.stamp_ns
                with patch("box4d.temporal.cv2.solvePnPGeneric", return_value=(0, (), (), None)), patch(
                    "box4d.temporal.cv2.solvePnP", return_value=(False, None, None)
                ), patch.object(pipeline, "_joint_pose", return_value=candidate) as joint, patch.object(
                    pipeline,
                    "_full_detect",
                    return_value=(empty_result(current, pipeline.cfg), None, [], (None, ())),
                ):
                    result, _ = pipeline.process(current)
                joint.assert_called_once()
                self.assertEqual(result["objects"], [])
                self.assertEqual(result["frame_status"], "partial_or_rejected")
                self.assertTrue(result["temporal"]["tracking_failures"])
                self.assertEqual(result["temporal"]["stale_poses_published"], 0)

    def test_real_joint_pose_still_fails_without_current_boundary_support(self):
        pipeline, current, _, gray, distance = self.inputs()
        distance[:] = 100.0
        details = {}
        with patch("box4d.temporal.cv2.solvePnPGeneric", return_value=(0, (), (), None)), patch(
            "box4d.temporal.cv2.solvePnP", return_value=(False, None, None)
        ), patch.object(pipeline, "_joint_pose", wraps=pipeline._joint_pose) as joint:
            with self.assertRaisesRegex(ValueError, "current_boundary_support"):
                pipeline._advance(current, deepcopy(self.seed), gray, distance, diagnostics=details)
        joint.assert_called_once()
        self.assertEqual(details["pose_trials"][0]["failed_checks"], ["current_boundary_support"])

    def test_real_joint_recovery_keeps_every_current_pose_gate(self):
        gates = (
            ("max_reprojection_px", "feature_reprojection"),
            ("max_depth_residual_m", "current_depth_residual"),
            ("max_normal_error_deg", "current_plane_normal"),
            ("max_upright_tilt_deg", "upright_tilt"),
            ("track_quad_max_drift_px", "quad_drift"),
            ("track_max_distance_m", "position_jump"),
        )
        for setting, failed_check in gates:
            with self.subTest(gate=failed_check):
                pipeline, current, _, gray, distance = self.inputs()
                # Tighten one existing gate while retaining the real LK,
                # current depth sampling, optimizer and pose validation.
                setattr(pipeline.cfg, setting, 1e-8)
                details = {}
                with patch("box4d.temporal.cv2.solvePnPGeneric", return_value=(0, (), (), None)), patch(
                    "box4d.temporal.cv2.solvePnP", return_value=(False, None, None)
                ), patch.object(pipeline, "_joint_pose", wraps=pipeline._joint_pose) as joint:
                    with self.assertRaisesRegex(ValueError, failed_check):
                        pipeline._advance(current, deepcopy(self.seed), gray, distance, diagnostics=details)
                joint.assert_called_once()
                self.assertIn(failed_check, details["pose_trials"][0]["failed_checks"])


if __name__ == "__main__":
    unittest.main()
