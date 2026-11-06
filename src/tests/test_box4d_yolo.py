"""Segmentation selects instances; calibrated geometry still estimates poses."""

from copy import deepcopy
from contextlib import nullcontext
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from box4d.config import Config
from box4d.debug import DebugSession, render_stages, tuning_positions
from box4d.io import json_value
from box4d.pose import Pipeline
from box4d.synthetic import Renderer, default_camera, world_box
from box4d.temporal import TemporalPipeline
from box4d.yolo import YoloGuidedPipeline, YoloRegions, instance_colors, upright_quarter_turns
from box4d.roi_faces import fit_boundary_lines


class Frontend:
    metadata = dict(class_name="tkk")

    def __init__(self, instances=()):
        self.instances = list(instances)

    def predict(self, frame):
        return self.instances


def instance(mask, pad=20):
    yy, xx = np.nonzero(mask)
    h, w = mask.shape
    contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return dict(
        mask=mask,
        roi=[
            max(0, int(xx.min()) - pad),
            max(0, int(yy.min()) - pad),
            min(w, int(xx.max()) + 1 + pad),
            min(h, int(yy.max()) + 1 + pad),
        ],
        score=0.95,
        mask_pixels=int(mask.sum()),
        polygons=[c.reshape(-1, 2).tolist() for c in contours],
    )


class YoloTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = Renderer(default_camera())

    def cfg(self, **kwargs):
        return Config(
            up_camera=[0, -1, 0],
            analyze_colors=True,
            require_target_colors=True,
            plane_grid_step=8,
            max_objects=3,
            yolo_weights="test-seg.pt",
            **kwargs
        ).validate()

    def small(self, x=0.0, yaw=30.0, height=0.1):
        box = world_box((x, 0, height), 0.2, yaw)
        box["red_axis_local"] = [0, 0, 1]
        return box

    def test_config_snapshot_reuses_equal_configs_and_detects_in_place_tuning(self):
        cfg = self.cfg()
        pipeline = YoloGuidedPipeline(cfg, frontend=Frontend())
        original = pipeline.full
        pipeline.cfg = deepcopy(cfg)
        pipeline._sync_config()
        self.assertIs(pipeline.full, original)
        pipeline.cfg.red_hue_ranges = [[0, 10], [170, 179]]
        pipeline._sync_config()
        updated = pipeline.full
        self.assertIsNot(updated, original)
        pipeline.cfg.red_hue_ranges[0][1] = 11
        pipeline._sync_config()
        self.assertIsNot(pipeline.full, updated)
        self.assertEqual(pipeline.full.cfg.red_hue_ranges[0][1], 11)

    def test_two_instances_keep_two_independent_poses_and_original_pixels(self):
        frame, truth, ids = self.renderer.render(
            [self.small(-0.22, 15), self.small(0.22, 35)], depth_noise_m=0.001
        )
        depth = frame.depth.copy()
        pipeline = YoloGuidedPipeline(
            self.cfg(), debug=True, frontend=Frontend([instance(ids == i) for i in (0, 1)])
        )
        result, _ = pipeline.process(frame)
        self.assertEqual(len(result["objects"]), 2)
        for obj in result["objects"]:
            reference = min(
                truth, key=lambda t: np.linalg.norm(np.asarray(t["center_camera_m"]) - obj["center_camera_m"])
            )
            np.testing.assert_allclose(obj["center_camera_m"], reference["center_camera_m"], atol=0.006)
            self.assertEqual(obj["segmentation"]["stamp_ns"], frame.stamp_ns)
            self.assertGreater(obj["segmentation"]["face_mask_fraction"], 0.95)
        np.testing.assert_array_equal(frame.depth, depth)
        for child in pipeline.region_pipelines.values():
            self.assertIs(child.last_debug.frame.camera, frame.camera)
            self.assertIs(child.last_debug.frame.rgb, frame.rgb)
            self.assertEqual(child.last_debug.frame.depth.shape, frame.depth.shape)
        self.assertEqual(len(pipeline.local_debug), 2)
        panels = render_stages(frame, result, pipeline.last_debug, pipeline.local_debug)
        self.assertEqual(panels.shape[0], 8)

    def test_camera_gravity_selects_network_quarter_turn_without_top_down_guess(self):
        for up, expected in (
            ([0, -1, 0], 0),
            ([1, 0, 0], 1),
            ([0, 1, 0], 2),
            ([-1, 0, 0], 3),
            ([0, 0, 1], 0),
            (None, 0),
        ):
            self.assertEqual(upright_quarter_turns(up), expected)

    def test_network_rotation_maps_masks_back_to_original_calibrated_pixels(self):
        class Tensor:
            def __init__(self, value):
                self.value = value

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        frame, _, _ = self.renderer.render([self.small()])
        mask = np.zeros(frame.depth.shape, bool)
        mask[80:110, 170:210] = True
        for turns, up in enumerate(([0, -1, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0])):
            with self.subTest(turns=turns):
                cfg = self.cfg()
                cfg.up_camera = up
                front = YoloRegions.__new__(YoloRegions)
                front.cfg, front.class_id, front.metadata = cfg, 0, {}
                front.torch = SimpleNamespace(get_num_threads=lambda: 2)

                def predict(bgr, **kwargs):
                    np.testing.assert_array_equal(
                        bgr, cv2.cvtColor(np.ascontiguousarray(np.rot90(frame.rgb, turns)), cv2.COLOR_RGB2BGR)
                    )
                    rotated = np.ascontiguousarray(np.rot90(mask, turns))[None].astype(np.float32)
                    return [
                        SimpleNamespace(
                            masks=SimpleNamespace(data=Tensor(rotated)),
                            boxes=SimpleNamespace(conf=Tensor(np.array([0.95]))),
                        )
                    ]

                front.model = SimpleNamespace(predict=predict, predictor=SimpleNamespace(device="cpu"))
                result = front.predict(frame)
                self.assertEqual(len(result), 1)
                np.testing.assert_array_equal(result[0]["mask"], mask)
                self.assertEqual(result[0]["roi"], [150, 60, 230, 130])
                self.assertEqual(front.metadata["input_rotation_ccw_deg"], turns * 90)

    def test_roi_pose_solver_handles_camera_roll_in_all_four_orientations(self):
        for turns in range(4):
            with self.subTest(turns=turns):
                angle = turns * np.pi / 2
                rotation = np.array(
                    [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.0]]
                )
                box = self.small()
                original = box["pose"].copy()
                box["pose"][:3] = rotation @ box["pose"][:3]
                frame, _, ids = self.renderer.render([box], depth_noise_m=0.001, floor_height_camera_m=0.0)
                cfg = self.cfg()
                cfg.up_camera = (rotation @ [0.0, -1.0, 0.0]).tolist()
                result, _ = YoloGuidedPipeline(cfg, frontend=Frontend([instance(ids == 0)])).process(frame)
                self.assertEqual(len(result["objects"]), 1)
                np.testing.assert_allclose(
                    rotation.T @ result["objects"][0]["center_camera_m"], original[:3, 3], atol=0.006
                )

    def test_big_cube_uses_identical_original_geometry_when_yolo_misses(self):
        frame, _, _ = self.renderer.render([world_box((0, 0, 0.175), 0.35, 20)], depth_noise_m=0.001)
        cfg = self.cfg()
        old_cfg = deepcopy(cfg)
        old_cfg.yolo_weights = None
        old_cfg.sizes_m = (0.35,)
        old_cfg.two_color_sizes_m = ()
        expected, _ = Pipeline(old_cfg).process(frame)
        actual, _ = YoloGuidedPipeline(cfg, frontend=Frontend()).process(frame)
        self.assertEqual(len(actual["objects"]), 1)
        self.assertEqual(json_value(actual["objects"]), json_value(expected["objects"]))

    def test_touching_stacked_big_and_small_are_kept_separate(self):
        frame, truth, ids = self.renderer.render(
            [world_box((0, 0, 0.175), 0.35, 15), self.small(height=0.45)], depth_noise_m=0.001
        )
        result, _ = YoloGuidedPipeline(self.cfg(), frontend=Frontend([instance(ids == 1)])).process(frame)
        self.assertEqual(sorted(o["size_m"] for o in result["objects"]), [0.2, 0.35])
        for obj in result["objects"]:
            reference = next(t for t in truth if t["size_m"] == obj["size_m"])
            np.testing.assert_allclose(obj["center_camera_m"], reference["center_camera_m"], atol=0.006)

    def test_small_cube_needs_segmentation_for_acquisition(self):
        frame, _, _ = self.renderer.render([self.small()], depth_noise_m=0.001)
        cfg = self.cfg()
        with patch(
            "box4d.temporal.make_detection_pipeline",
            return_value=YoloGuidedPipeline(cfg, frontend=Frontend()),
        ):
            pipeline = TemporalPipeline(cfg)
        with patch.object(pipeline, "_search_uncovered_colors", side_effect=AssertionError("bypassed YOLO")):
            result, _ = pipeline.process(frame)
        self.assertEqual(result["objects"], [])

    def test_duplicate_masks_do_not_duplicate_the_cube(self):
        frame, _, ids = self.renderer.render([self.small()], depth_noise_m=0.001)
        item = instance(ids == 0)
        result, _ = YoloGuidedPipeline(self.cfg(), frontend=Frontend([item, item])).process(frame)
        self.assertEqual(len(result["yolo"]["instances"]), 2)
        self.assertEqual(len(result["objects"]), 1)

    def test_confident_segmentation_cannot_replace_missing_depth(self):
        frame, _, ids = self.renderer.render([self.small()], depth_noise_m=0.001)
        frame.depth[:] = 0
        result, _ = YoloGuidedPipeline(self.cfg(), frontend=Frontend([instance(ids == 0)])).process(frame)
        self.assertEqual(len(result["yolo"]["instances"]), 1)
        self.assertEqual(result["objects"], [])

    def test_red_blue_gate_precedes_geometry_and_does_not_count_padding_colors(self):
        frame, _, ids = self.renderer.render([self.small()], depth_noise_m=0.001)
        mask = ids == 0
        frame.rgb[mask] = (230, 20, 20)
        frame.rgb[~mask] = (20, 20, 230)
        pipeline = YoloGuidedPipeline(
            self.cfg(yolo_color_min_each_fraction=0.04), debug=True, frontend=Frontend([instance(mask)])
        )
        result, _ = pipeline.process(frame)
        self.assertEqual(result["objects"], [])
        report = result["yolo"]["instances"][0]
        self.assertFalse(report["candidate_confirmed"])
        self.assertEqual(report["color_gate"]["blue_fraction"], 0)
        self.assertGreater(report["color_gate"]["roi_blue_fraction"], 0)
        self.assertEqual(pipeline.region_pipelines, {})
        self.assertEqual(render_stages(frame, result, pipeline.last_debug, pipeline.local_debug).shape[0], 8)

    def test_red_blue_gate_does_not_require_equal_color_areas(self):
        labels = np.ones((100, 100), np.int8)
        labels[:20] = -1
        gate = instance_colors(labels, instance(np.ones(labels.shape, bool)), self.cfg())
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["red_fraction"], 0.8)
        self.assertEqual(gate["blue_fraction"], 0.2)
        labels[:] = -1
        gate = instance_colors(labels, instance(np.ones(labels.shape, bool)), self.cfg())
        self.assertTrue(gate["passed"], "a visible blue-only face must remain a small-cube candidate")

    def test_observed_partial_segments_produce_intersections_and_seam_is_ignored(self):
        quad = np.array([[30.0, 20.0], [130.0, 25.0], [125.0, 125.0], [20.0, 115.0]])
        edges = np.zeros((160, 160), np.uint8)
        for a, b in zip(quad, np.roll(quad, -1, axis=0)):
            cv2.line(
                edges,
                tuple(np.rint(a + 0.2 * (b - a)).astype(int)),
                tuple(np.rint(a + 0.8 * (b - a)).astype(int)),
                255,
            )
        cv2.line(edges, (24, 70), (127, 75), 255)  # Internal paint seam.
        perturbed = quad + np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [-1.0, 1.0]])
        fit = fit_boundary_lines(perturbed, edges, self.cfg())
        self.assertIsNotNone(fit)
        corners, lines = fit
        self.assertLess(np.max(np.linalg.norm(corners - quad, axis=1)), 1.5)
        self.assertEqual(len(lines), 4)
        self.assertTrue(all(line["span_fraction"] > 0.4 for line in lines))
        edges[100:] = 0  # No observed bottom line: no invented four-corner PnP input.
        self.assertIsNone(fit_boundary_lines(perturbed, edges, self.cfg()))

    def test_roi_plane_line_corners_are_actually_evaluated_by_pnp(self):
        frame, _, ids = self.renderer.render([self.small(yaw=30)], depth_noise_m=0.001)
        pipeline = YoloGuidedPipeline(self.cfg(), debug=True, frontend=Frontend([instance(ids == 0)]))
        result, _ = pipeline.process(frame)
        self.assertEqual(len(result["objects"]), 1)
        faces = [
            f for trace in pipeline.local_debug for f in trace.faces if f["source"] == "roi_plane_rgb_lines"
        ]
        self.assertTrue(faces)
        self.assertTrue(
            any(t["method"].startswith("pnp_") and t["accepted"] for f in faces for t in f["trials"])
        )
        self.assertTrue(any(f["status"] == "geometry_pass" for f in faces))

    def test_segmentation_crop_inside_big_face_cannot_invent_small_cube_boundaries(self):
        frame, _, ids = self.renderer.render(
            [world_box((0, 0, 0.175), 0.35, 0, color=(40, 45, 225))], depth_noise_m=0.001
        )
        center = np.rint(frame.camera.project([[0, 0.175, 1.025]])[0]).astype(int)
        mask = np.zeros(frame.depth.shape, bool)
        x, y = center
        mask[y - 44 : y + 45, x - 44 : x + 45] = True
        self.assertTrue(np.all(ids[mask] == 0))
        result, _ = YoloGuidedPipeline(self.cfg(), frontend=Frontend([instance(mask)])).process(frame)
        self.assertEqual([o["size_m"] for o in result["objects"]], [0.35])
        self.assertTrue(result["yolo"]["instances"][0]["candidate_confirmed"])

    def test_shallow_bottom_occlusion_uses_raw_foreground_not_masked_out_depth(self):
        box = world_box((0, 0, 0.1), 0.2, 0, (0, -1.2, 0.1))
        box["red_axis_local"] = [0, 0, 1]
        occluder = dict(
            quad_camera=[[-0.3, 0.063, 0.85], [0.3, 0.063, 0.85], [0.3, 0.4, 0.85], [-0.3, 0.4, 0.85]]
        )
        frame, truth, ids = self.renderer.render(
            [box], depth_noise_m=0.001, floor_height_camera_m=0.1, occluders=[occluder]
        )
        mask = ids == 0
        old, _ = YoloGuidedPipeline(
            self.cfg(yolo_roi_color_geometry=False), frontend=Frontend([instance(mask)])
        ).process(frame)
        self.assertEqual(old["objects"], [])
        pipeline = YoloGuidedPipeline(self.cfg(), frontend=Frontend([instance(mask)]))
        result, _ = pipeline.process(frame)
        self.assertEqual(len(result["objects"]), 1)
        obj = result["objects"][0]
        self.assertEqual(obj["method"], "depth_partial")
        self.assertIsNone(obj["metrics"]["reprojection_rms_px"])
        self.assertGreater(obj["metrics"]["foreground_depth_fraction"], 0.04)
        np.testing.assert_allclose(obj["center_camera_m"], truth[0]["center_camera_m"], atol=0.005)
        for background in (0.0, 2.5):
            negative = deepcopy(frame)
            negative.depth[ids == -2] = background
            actual, _ = pipeline.process(negative)
            self.assertFalse(
                any(f["source"] == "bottom_occlusion_model" for o in actual["objects"] for f in o["faces"])
            )

    def test_live_config_updates_reach_both_geometry_branches(self):
        frame, _, ids = self.renderer.render([self.small()], depth_noise_m=0.001)
        pipeline = YoloGuidedPipeline(self.cfg(), frontend=Frontend([instance(ids == 0)]))
        pipeline.process(frame)
        pipeline.cfg.plane_threshold_m = 0.009
        pipeline.process(frame)
        self.assertEqual(pipeline.full.cfg.plane_threshold_m, 0.009)
        self.assertTrue(pipeline.region_pipelines)
        self.assertTrue(all(p.cfg.plane_threshold_m == 0.009 for p in pipeline.region_pipelines.values()))

    def test_slow_yolo_keeps_tracking_thread_running_and_only_latest_pending_frame(self):
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()
        frames = [self.renderer.render([self.small()], i)[0] for i in range(12)]
        front = Frontend()

        def slow_predict(frame):
            entered.set()
            release.wait(timeout=4.0)
            return []

        front.predict = slow_predict
        cfg = self.cfg(async_detection=True)
        detector = YoloGuidedPipeline(cfg, frontend=front)
        with patch("box4d.temporal.make_detection_pipeline", return_value=detector):
            pipeline = TemporalPipeline(cfg, live=True)
        errors = []

        def current_frames():
            try:
                for frame in frames[1:]:
                    result, _ = pipeline.process(frame)
                    self.assertEqual(result["temporal"]["detector_queue_depth"], 1)
            except Exception as error:
                errors.append(error)
            finally:
                returned.set()

        consumer = threading.Thread(target=current_frames)
        try:
            pipeline.process(frames[0])
            self.assertTrue(entered.wait(timeout=1.0))
            acquired = pipeline.config_lock.acquire(blocking=False)
            self.assertTrue(acquired, "GPU work held the live tuning lock")
            if acquired:
                pipeline.config_lock.release()
            consumer.start()
            self.assertTrue(returned.wait(timeout=1.0), "YOLO blocked current RGB-D processing")
            self.assertEqual(errors, [])
            with pipeline.worker_condition:
                self.assertEqual(pipeline.worker_pending[0].sequence, frames[-1].sequence)
                self.assertGreaterEqual(pipeline.worker_pending_replaced, 9)
        finally:
            release.set()
            consumer.join(timeout=1.0)
            pipeline.close()

    def test_delayed_segmentation_still_requires_current_depth(self):
        for missing in (False, True):
            with self.subTest(missing_depth=missing):
                first, _, ids = self.renderer.render([self.small()], 0, depth_noise_m=0.001)
                current, truth, _ = self.renderer.render([self.small(0.002)], 1, depth_noise_m=0.001)
                if missing:
                    current.depth[:] = 0
                cfg = self.cfg(async_detection=True)
                detector = YoloGuidedPipeline(cfg, frontend=Frontend([instance(ids == 0)]))
                with patch("box4d.temporal.make_detection_pipeline", return_value=detector):
                    pipeline = TemporalPipeline(cfg, live=True)
                try:
                    pipeline.process(first)
                    with pipeline.worker_condition:
                        self.assertTrue(
                            pipeline.worker_condition.wait_for(
                                lambda: pipeline.worker_completed is not None
                                or pipeline.worker_error is not None,
                                timeout=2.0,
                            )
                        )
                    result, _ = pipeline.process(current)
                    self.assertEqual(result["stamp_ns"], current.stamp_ns)
                    if missing:
                        self.assertEqual(result["objects"], [])
                        self.assertGreater(len(result["temporal"]["detector_seed_failures"]), 0)
                    else:
                        self.assertEqual(len(result["objects"]), 1, result["temporal"])
                        obj = result["objects"][0]
                        self.assertEqual(obj["source"], "current_rgbd_flow")
                        self.assertEqual(obj["segmentation"]["stamp_ns"], first.stamp_ns)
                        np.testing.assert_allclose(
                            obj["center_camera_m"], truth[0]["center_camera_m"], atol=0.012
                        )
                finally:
                    pipeline.close()

    def test_slider_and_save_do_not_wait_for_busy_detector(self):
        class SharedPositions(list):
            def get_lock(self):
                return nullcontext()

        cfg = self.cfg()
        display = DebugSession.__new__(DebugSession)
        display.tuning_shared = SharedPositions(tuning_positions(cfg))
        display.applied_positions = None
        display.config = cfg
        display.save_pending = False
        display.last = None
        display._save_config = Mock()
        lock = threading.Lock()
        pipeline = SimpleNamespace(config_lock=lock, cfg=cfg, region_pipelines={})
        lock.acquire()
        try:
            self.assertFalse(display._apply_tuning(pipeline))
            display.save(pipeline)
            self.assertTrue(display.save_pending)
            display._save_config.assert_not_called()
        finally:
            lock.release()
        display.save(pipeline)
        self.assertFalse(display.save_pending)
        display._save_config.assert_called_once()


if __name__ == "__main__":
    unittest.main()
