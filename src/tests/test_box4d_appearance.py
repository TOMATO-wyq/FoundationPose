"""Color-axis semantics, visibility failures and reference stacking regressions."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from box4d.appearance import analyze_colors, classify_hsv
from box4d.config import Config
from box4d.io import Frame, json_value
from box4d.metrics import evaluate
from box4d.pose import Pipeline
from box4d.synthetic import Renderer, default_camera, reference_stacking_scenes, world_box


class AppearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = Renderer(default_camera())

    def config(self, **kwargs):
        return Config(analyze_colors=True, two_color_sizes_m=(.20,), up_camera=[0, -1, 0],
                      max_objects=3, plane_grid_step=8, **kwargs).validate()

    def render(self, axis=(0, 0, 1), angle=15):
        box = world_box((0, 0, .1), .20, angle)
        box["red_axis_local"] = axis
        return self.renderer.render([box], 7, .001, .005)

    def test_hue_wrap_low_saturation_and_dark_pixels(self):
        hsv = np.array([[0, 200, 200], [179, 200, 200], [120, 200, 200],
                        [120, 0, 200], [0, 200, 10], [50, 200, 200]], np.uint8)
        np.testing.assert_array_equal(classify_hsv(hsv, self.config()), [1, 1, -1, 0, 0, 0])

    def test_six_color_directions_and_two_yaws(self):
        pipeline = Pipeline(self.config())
        for angle in (15, 55):
            for axis in ((0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)):
                with self.subTest(angle=angle, axis=axis):
                    frame, truth, _ = self.render(axis, angle)
                    result, _ = pipeline.process(frame)
                    self.assertEqual(len(result["objects"]), 1)
                    appearance = result["objects"][0]["appearance"]
                    self.assertEqual(appearance["orientation_status"], truth[0]["color_orientation"])
                    error = np.rad2deg(np.arccos(np.clip(np.dot(appearance["red_axis_camera"], truth[0]["red_axis_camera"]), -1, 1)))
                    self.assertLess(error, 2.)
                    plane = np.asarray(appearance["partition_plane_camera"])
                    center = np.asarray(truth[0]["center_camera_m"])
                    direction = np.asarray(truth[0]["red_axis_camera"])
                    self.assertGreater(plane[:3] @ (center + direction * .05) + plane[3], .02)
                    self.assertLess(plane[:3] @ (center - direction * .05) + plane[3], -.02)
                    if axis[2] != 0:
                        self.assertIsNone(appearance["red_direction_yaw_u_rad"])
                    else:
                        self.assertIsNotNone(appearance["red_direction_yaw_u_rad"])
                    self.assertAlmostEqual(result["objects"][0]["yaw_period_rad"], np.pi / 2)

    def test_opposite_color_directions_are_not_mod90_yaw(self):
        pipeline = Pipeline(self.config())
        directions = []
        for sign in (-1, 1):
            frame, _, _ = self.render((sign, 0, 0))
            result, _ = pipeline.process(frame)
            directions.append(result["objects"][0]["appearance"]["red_axis_camera"])
        self.assertLess(np.dot(*directions), -.999)

    def test_missing_or_incompatible_color_cannot_produce_orientation(self):
        frame, _, _ = self.render()
        cfg = self.config()
        result, _ = Pipeline(cfg).process(frame)
        for kind in ("gray", "checkerboard"):
            rgb = np.full_like(frame.rgb, 150)
            if kind == "checkerboard":
                yy, xx = np.indices(frame.depth.shape)
                rgb[:] = np.where(((xx // 3 + yy // 3) % 2)[..., None], (235, 35, 35), (40, 45, 225))
            bad = Frame(rgb.astype(np.uint8), frame.depth, frame.camera, frame.stamp_ns)
            objects = deepcopy(result["objects"])
            analyze_colors(bad, objects, cfg)
            appearance = objects[0]["appearance"]
            self.assertIsNone(appearance["red_axis_camera"])
            self.assertIn(appearance["orientation_status"], ("insufficient_color", "color_model_mismatch", "ambiguous"))

    def test_no_up_reference_never_reports_red_up(self):
        frame, _, _ = self.render()
        cfg = self.config()
        cfg.up_camera = None
        result, _ = Pipeline(cfg).process(frame)
        self.assertTrue(result["objects"])
        appearance = result["objects"][0]["appearance"]
        self.assertEqual(appearance["orientation_status"], "axis_unreferenced")
        self.assertIsNone(appearance["red_axis_u"])
        self.assertIsNone(appearance["top_color"])

    def test_uniform_face_inference_is_explicit(self):
        frame, _, _ = self.render((0, 1, 0))
        result, _ = Pipeline(self.config()).process(frame)
        appearance = result["objects"][0]["appearance"]
        self.assertEqual(appearance["evidence"], "model_inferred_from_uniform_face")
        self.assertIn("assumption", appearance)

    def test_face_with_no_valid_color_depth_does_not_crash(self):
        frame, _, _ = self.render()
        cfg = self.config()
        result, _ = Pipeline(cfg).process(frame)
        empty = Frame(frame.rgb, np.zeros_like(frame.depth), frame.camera, frame.stamp_ns)
        objects = deepcopy(result["objects"])
        analyze_colors(empty, objects, cfg)
        self.assertEqual(objects[0]["appearance"]["orientation_status"], "insufficient_color")
        self.assertEqual(frame.camera.rays([]).shape, (0, 3))
        self.assertEqual(frame.camera.project([]).shape, (0, 2))

    def test_rgb_edge_fallback_recovers_pose_when_primary_has_no_face(self):
        # All four outer face edges are visible in this front view. An oblique
        # silhouette alone must not be mistaken for a single square face.
        box = world_box((0, 0, .1), .20, 0, camera_world=(0, -1.2, .1))
        box["red_axis_local"] = (0, 0, 1)
        frame, truth, _ = self.renderer.render([box], 7, .001, .005, floor_height_camera_m=.1)
        cfg = self.config(color_line_fallback=True)
        pipeline = Pipeline(cfg)
        propose = pipeline.detector.propose

        def no_primary_face(f):
            _, timing, edges = propose(f)
            return [], timing, edges

        with patch.object(pipeline.detector, "propose", side_effect=no_primary_face):
            result, _ = pipeline.process(frame)
        self.assertTrue(result["timing"]["color_line_fallback_used"])
        self.assertEqual(len(result["objects"]), 1)
        obj = result["objects"][0]
        self.assertLess(np.linalg.norm(np.asarray(obj["center_camera_m"]) - truth[0]["center_camera_m"]), .015)
        self.assertEqual(obj["appearance"]["orientation_status"], "red_up")

    def test_line_fallback_does_not_treat_oblique_silhouette_as_one_face(self):
        frame, _, _ = self.render(angle=15)
        pipeline = Pipeline(self.config(color_line_fallback=True))
        propose = pipeline.detector.propose

        def no_primary_face(f):
            _, timing, edges = propose(f)
            return [], timing, edges

        with patch.object(pipeline.detector, "propose", side_effect=no_primary_face):
            result, _ = pipeline.process(frame)
        self.assertEqual(result["objects"], [])

    def test_colored_wrong_size_rejected_with_line_fallback(self):
        box = world_box((0, 0, .135), .27, 15)
        box["red_axis_local"] = (0, 0, 1)
        frame, _, _ = self.renderer.render([box], 9, .001, .005)
        result, _ = Pipeline(self.config(color_line_fallback=True)).process(frame)
        self.assertEqual(result["objects"], [])

    def test_three_levels_and_color_evaluation_count_missing_pose(self):
        cfg = self.config()
        scene, boxes = reference_stacking_scenes()[8]
        frame, truth, _ = self.renderer.render(boxes, 8, .001, .005)
        result, _ = Pipeline(cfg).process(frame)
        self.assertEqual(len(result["objects"]), 3)
        row = dict(stamp_ns=frame.stamp_ns, scene=scene, objects=truth)
        evaluation = evaluate([result], [row], cfg)
        self.assertEqual(evaluation["matched"], 3)
        self.assertEqual(evaluation["color_orientation"]["state_correct"], 1)
        missing = deepcopy(result)
        missing["objects"] = [o for o in missing["objects"] if o["size_m"] != .20]
        evaluation = evaluate([missing], [row], cfg)
        self.assertEqual(evaluation["color_orientation"]["target_observations"], 1)
        self.assertEqual(evaluation["color_orientation"]["unresolved_or_missing_axis"], 1)
        json_value(result)  # all output fields can pass strict JSON serialization

    def test_color_config_rejects_incompatible_sizes_and_bad_thresholds(self):
        for kwargs in (dict(two_color_sizes_m=(.10,)), dict(red_hue_ranges=((170, 190),)),
                       dict(blue_hue_range=(120, 95)), dict(color_min_fit=float("nan"))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs).validate()


if __name__ == "__main__":
    unittest.main()
