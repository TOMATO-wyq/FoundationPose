"""Detection attempts remain attributable and JSON-safe without debug windows."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from box4d.config import Config
from box4d.detector import Face
from box4d.diagnostics import detector_snapshot
from box4d.errors import FaceSizeMismatch
from box4d.geometry import Camera, square_points, transform, transform_points
from box4d.io import Frame
from box4d.pose import Pipeline, _make_estimate, estimate_face, estimate_partial_face, prepare_face
from box4d.results import append_detection_checks, empty_result, refresh_detection_selection
from box4d.synthetic import Renderer, default_camera, world_box
from box4d.yolo import YoloGuidedPipeline


class DetectionDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(320, 240, [[300, 0, 160], [0, 300, 120], [0, 0, 1]], [0] * 5)
        rotation = cv2.Rodrigues(np.array([0.02, -0.08, 0.01]))[0] @ np.diag([1., -1., -1.])
        self.pose = transform(rotation, [0.01, 0.02, 1.0])
        yy, xx = np.indices((240, 320))
        rays = self.camera.rays(np.column_stack((xx.ravel(), yy.ravel())))
        normal = self.pose[:3, 2]
        depth = ((normal @ self.pose[:3, 3]) / (rays @ normal)).reshape(240, 320)
        self.frame = Frame(np.zeros((240, 320, 3), np.uint8), depth.astype(np.float32), self.camera, 100, 7)
        self.face = Face(self.camera.project(transform_points(self.pose, square_points(0.2))), "test", 1.)
        self.cfg = Config(
            sizes_m=(0.2,), geometry_backend="numpy", up_camera=None, refine_4d=False,
            allow_bottom_occlusion=False, color_line_fallback=False,
        ).validate()
        self.prepared = prepare_face(self.frame, self.face, self.cfg)

    def run_pipeline(self, frame=None, *, debug=False, cfg=None):
        frame = self.frame if frame is None else frame
        pipeline = Pipeline(self.cfg if cfg is None else cfg, debug=debug)
        with patch.object(pipeline.detector, "propose", return_value=(
            [self.face], dict(detector_ms=0., proposal_count=1), np.zeros(frame.depth.shape, np.uint8)
        )), patch.object(pipeline.detector, "join_color_halves", return_value=[]):
            result, _ = pipeline.process(frame)
        return pipeline, result

    def assert_strict_json(self, value):
        return json.loads(json.dumps(value, allow_nan=False))

    def test_early_invalid_pose_attempts_are_complete_and_sequential(self):
        samples, fraction, plane, _, _ = self.prepared
        nonfinite, behind, back = self.pose.copy(), self.pose.copy(), self.pose.copy()
        nonfinite[0, 3], behind[2, 3] = np.nan, -1.
        back[:3, :3] = np.eye(3)
        diagnostics = {"face_index": 4, "trials": []}
        for bad, check in ((nonfinite, "invalid_parameters"), (behind, "behind_camera"), (back, "back_facing")):
            with self.subTest(check=check), self.assertRaises(ValueError):
                _make_estimate(self.frame, self.face, plane, samples, fraction, .2, bad, "test", self.cfg, diagnostics)
            trial = diagnostics["trials"][-1]
            self.assertEqual(trial["failed_checks"], [check])
            self.assertFalse(trial["accepted"])
            self.assertFalse(trial["selected"])
            self.assertEqual(trial["stage"], "pose_validation")
        self.assertEqual([t["trial_index"] for t in diagnostics["trials"]], [0, 1, 2])
        self.assert_strict_json(diagnostics)

    def test_nonfinite_projection_is_a_failed_trial(self):
        samples, fraction, plane, _, _ = self.prepared
        diagnostics = {"trials": []}
        with patch.object(self.camera, "project", return_value=np.full((4, 2), np.nan)):
            with self.assertRaisesRegex(ValueError, "nonfinite projected"):
                _make_estimate(self.frame, self.face, plane, samples, fraction, .2, self.pose, "test", self.cfg, diagnostics)
        self.assertEqual(diagnostics["trials"][0]["failed_checks"], ["invalid_projection"])
        self.assert_strict_json(diagnostics)

    def test_size_rejection_and_solver_exception_are_recorded(self):
        diagnostics = {}
        cfg = Config(sizes_m=(.35,), geometry_backend="numpy").validate()
        with self.assertRaises(FaceSizeMismatch):
            estimate_face(self.frame, self.face, cfg, prepared=self.prepared, diagnostics=diagnostics)
        self.assertEqual(diagnostics["trials"][0]["failed_checks"], ["size_mismatch"])
        self.assertFalse(diagnostics["trials"][0]["accepted"])
        cfg = Config(sizes_m=(.2,), method="pnp", geometry_backend="numpy", up_camera=None).validate()
        diagnostics = {}
        with patch("box4d.pose.cv2.solvePnPGeneric", side_effect=cv2.error("injected solver failure")):
            with self.assertRaisesRegex(ValueError, "injected solver failure"):
                estimate_face(self.frame, self.face, cfg, prepared=self.prepared, diagnostics=diagnostics)
        self.assertEqual(diagnostics["trials"][0]["failed_checks"], ["solver_error"])
        self.assertEqual(diagnostics["trials"][0]["stage"], "solver")
        self.assert_strict_json(diagnostics)

    def test_empty_solver_and_partial_mode_rejections_are_not_silent(self):
        cfg = Config(sizes_m=(.2,), method="pnp", geometry_backend="numpy", up_camera=None).validate()
        diagnostics = {}
        with patch("box4d.pose.cv2.solvePnPGeneric", return_value=(0, (), (), None)):
            with self.assertRaises(ValueError):
                estimate_face(self.frame, self.face, cfg, prepared=self.prepared, diagnostics=diagnostics)
        self.assertEqual(diagnostics["trials"][0]["failed_checks"], ["no_solution"])
        partial = deepcopy(self.face)
        partial.evidence = {"size_m": .2}
        diagnostics = {}
        with self.assertRaisesRegex(ValueError, "PnP-only"):
            estimate_partial_face(self.frame, partial, cfg, diagnostics)
        self.assertFalse(diagnostics["trials"][0]["accepted"])
        self.assertTrue(diagnostics["trials"][0]["failed_checks"])
        self.assert_strict_json(diagnostics)

    def test_no_debug_success_has_selected_evidence_and_only_small_arrays(self):
        pipeline, result = self.run_pipeline()
        self.assertIsNone(pipeline.last_debug)
        self.assertEqual(len(result["objects"]), 1)
        check = result["detection_checks"][0]
        self.assertEqual((check["sequence"], check["stamp_ns"]), (7, 100))
        self.assertEqual(check["status"], "geometry_pass")
        self.assertNotIn("sample_uv", check)
        self.assertNotIn("sample_residual_m", check)
        self.assertTrue(check["trials"])
        obj = result["objects"][0]
        selected = check["trials"][obj["detection_trial_index"]]
        self.assertTrue(selected["accepted"])
        self.assertTrue(selected["selected"])
        self.assertEqual(selected["selected_object_indices"], [0])
        self.assertEqual(obj["method"], selected["method"])
        self.assertEqual(sum(t["geometry_selected"] for t in check["trials"]), 1)
        self.assert_strict_json(result)

    def test_no_depth_leaves_prepare_failure_without_debug(self):
        frame = Frame(self.frame.rgb, np.zeros_like(self.frame.depth), self.camera, 101, 8)
        _, result = self.run_pipeline(frame)
        self.assertEqual(result["objects"], [])
        check = result["detection_checks"][0]
        self.assertEqual(check["stage"], "depth_sampling")
        self.assertEqual(check["trials"][0]["failed_checks"], ["depth_sampling"])
        self.assertIn("insufficient interior depth coverage", check["trials"][0]["reason"])
        self.assert_strict_json(result)

    def test_debug_collection_does_not_change_numeric_solution_or_duplicate_trials(self):
        _, plain = self.run_pipeline()
        pipeline, debug = self.run_pipeline(debug=True)
        np.testing.assert_array_equal(plain["objects"][0]["T_camera_from_box"], debug["objects"][0]["T_camera_from_box"])
        self.assertEqual(plain["detection_checks"], debug["detection_checks"])
        self.assertEqual(len(pipeline.last_debug.faces[0]["trials"]), len(plain["detection_checks"][0]["trials"]))
        self.assertIn("sample_uv", pipeline.last_debug.faces[0])

    def test_snapshot_checks_keep_source_identity_and_are_deep_copied(self):
        _, result = self.run_pipeline()
        current = Frame(self.frame.rgb, self.frame.depth, self.camera, 200, 9)
        snapshot = detector_snapshot(self.frame, current, result, [], 10., .75)
        result["detection_checks"][0]["trials"][0]["selected"] = "mutated"
        result["detection_checks"][0]["quad"][0][0] = 999
        self.assertNotEqual(snapshot["detection_checks"][0]["trials"][0]["selected"], "mutated")
        self.assertNotEqual(snapshot["detection_checks"][0]["quad"][0][0], 999)
        self.assertEqual(snapshot["detection_checks"][0]["stamp_ns"], self.frame.stamp_ns)
        self.assertEqual(snapshot["current_frame"]["stamp_ns"], current.stamp_ns)
        self.assert_strict_json(snapshot)

    def test_append_remaps_child_indices_without_mutating_child_or_pixels(self):
        _, child = self.run_pipeline()
        original = deepcopy(child)
        destination = empty_result(self.frame, self.cfg)
        destination["detection_checks"] = [deepcopy(child["detection_checks"][0])]
        destination["objects"] = [deepcopy(child["objects"][0])]
        remapped = append_detection_checks(
            destination, child, source=dict(branch="local_search", roi=[20, 30, 200, 210])
        )
        destination["objects"].extend(remapped)
        refresh_detection_selection(destination)
        self.assertEqual(child, original)
        self.assertEqual(remapped[0]["detection_check_index"], 1)
        self.assertEqual(remapped[0]["faces"][0]["detection_check_index"], 1)
        copied = destination["detection_checks"][1]
        self.assertEqual((copied["face_index"], copied["source_face_index"]), (1, 0))
        self.assertEqual(copied["quad"], child["detection_checks"][0]["quad"])
        self.assertEqual(copied["stamp_ns"], child["stamp_ns"])
        self.assertTrue(all(trial["face_index"] == 1 for trial in copied["trials"]))
        selected = copied["trials"][remapped[0]["detection_trial_index"]]
        self.assertEqual(selected["selected_object_indices"], [1])
        self.assert_strict_json(destination)

    def yolo_scene(self, *, duplicate=False):
        renderer = Renderer(default_camera())
        boxes = [world_box((x, 0, .1), .2, angle) for x, angle in ((-.22, 15), (.22, 35))]
        frame, _, ids = renderer.render(boxes, sequence=11, depth_noise_m=.001)

        def instance(mask):
            yy, xx = np.nonzero(mask)
            return dict(
                mask=mask, roi=[max(0, int(xx.min()) - 20), max(0, int(yy.min()) - 20),
                                min(frame.camera.width, int(xx.max()) + 21), min(frame.camera.height, int(yy.max()) + 21)],
                score=.95, mask_pixels=int(mask.sum()), polygons=[],
            )

        instances = [instance(ids == index) for index in ((0, 0) if duplicate else (0, 1))]

        class Frontend:
            metadata = dict(class_name="test")

            def predict(self, frame):
                return instances

        cfg = Config(
            sizes_m=(.2,), yolo_weights="test-seg.pt", yolo_roi_color_geometry=False,
            geometry_backend="numpy", up_camera=[0, -1, 0], plane_grid_step=8,
            max_objects=3, allow_bottom_occlusion=False,
        ).validate()
        return frame, YoloGuidedPipeline(cfg, frontend=Frontend())

    def test_yolo_multiple_same_size_rois_have_global_links_and_source_snapshot(self):
        frame, pipeline = self.yolo_scene()
        result, _ = pipeline.process(frame)
        self.assertEqual(len(result["objects"]), 2)
        checks = result["detection_checks"]
        self.assertEqual([check["face_index"] for check in checks], list(range(len(checks))))
        for object_index, obj in enumerate(result["objects"]):
            check = checks[obj["detection_check_index"]]
            self.assertEqual(check["yolo_instance"], obj["segmentation"]["instance"])
            self.assertEqual((check["stamp_ns"], check["sequence"]), (frame.stamp_ns, frame.sequence))
            for source in obj["faces"]:
                measured = checks[source["detection_check_index"]]
                self.assertEqual(measured["quad"], source["quad"])
                self.assertEqual(measured["yolo_instance"], obj["segmentation"]["instance"])
                selected = measured["trials"][source["detection_trial_index"]]
                self.assertTrue(selected["selected"])
                self.assertEqual(selected["selected_object_indices"], [object_index])
        current = Frame(frame.rgb, frame.depth, frame.camera, frame.stamp_ns + 1_000_000, 12)
        snapshot = detector_snapshot(frame, current, result, [], 1., .75)
        self.assertEqual(snapshot["detection_checks"], checks)
        result["detection_checks"][0]["trials"][0]["selected"] = "mutated"
        self.assertNotEqual(snapshot["detection_checks"][0]["trials"][0]["selected"], "mutated")
        self.assert_strict_json(snapshot)

    def test_yolo_mask_filter_clears_selected_without_erasing_accepted_geometry(self):
        frame, pipeline = self.yolo_scene()
        with patch("box4d.yolo._face_mask_fraction", return_value=0.):
            result, _ = pipeline.process(frame)
        self.assertEqual(result["objects"], [])
        trials = [trial for check in result["detection_checks"] for trial in check["trials"]]
        self.assertTrue(any(trial["accepted"] for trial in trials))
        self.assertTrue(all(not trial["selected"] for trial in trials))
        self.assertTrue(all("selected_object_indices" not in trial for trial in trials))
        self.assertTrue(any(face["source"] == "yolo_instance_gate" for face in result["rejected_faces"]))
        self.assert_strict_json(result)

    def test_yolo_scene_suppression_marks_only_surviving_roi_trials(self):
        frame, pipeline = self.yolo_scene(duplicate=True)
        result, _ = pipeline.process(frame)
        self.assertEqual(len(result["objects"]), 1)
        self.assertTrue(result["suppressed_hypotheses"])
        selected_instances = {
            check["yolo_instance"] for check in result["detection_checks"]
            if any(trial["selected"] for trial in check["trials"])
        }
        self.assertEqual(selected_instances, {result["objects"][0]["segmentation"]["instance"]})
        self.assert_strict_json(result)


if __name__ == "__main__":
    unittest.main()
