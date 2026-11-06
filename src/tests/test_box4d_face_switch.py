"""Switch measured face anchors without weakening current RGB-D validation."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from box4d.config import Config
from box4d.geometry import transform_points
from box4d.synthetic import Renderer, default_camera, world_box
from box4d.temporal import TemporalPipeline


class FaceSwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = Renderer(default_camera())

    def cfg(self, **kwargs):
        return Config(
            up_camera=[0, -1, 0],
            analyze_colors=True,
            two_color_sizes_m=(0.2,),
            require_target_colors=True,
            temporal_recovery=True,
            plane_grid_step=8,
            **kwargs,
        )

    def frame(self, i, size=0.2, yaw=30.0, x=None):
        box = world_box((i * 0.002 if x is None else x, 0, size / 2), size, yaw)
        box["red_axis_local"] = [0, 0, 1]
        return self.renderer.render([box], i, depth_noise_m=0.001)

    def seeded(self, size=0.2, yaw=30.0, narrow=True, **kwargs):
        pipeline = TemporalPipeline(self.cfg(**kwargs))
        first, _, _ = self.frame(0, size, yaw)
        result, _ = pipeline.process(first)
        self.assertEqual(len(result["objects"]), 1)
        obj = deepcopy(result["objects"][0])
        sides = [
            f
            for f in obj["faces"]
            if abs(np.asarray(f["normal_camera"]) @ np.asarray(pipeline.cfg.up_camera)) < 0.3
        ]
        self.assertGreaterEqual(len(sides), 2)
        sides.sort(key=lambda f: cv2.contourArea(np.asarray(f["quad"], np.float32)))
        obj["faces"] = [sides[0 if narrow else -1]]
        obj["face_count"] = 1
        seed = pipeline._seed(first, obj, cv2.cvtColor(first.rgb, cv2.COLOR_RGB2GRAY))
        self.assertIsNotNone(seed)
        pipeline.states = [seed]
        pipeline.last_detection = None
        return pipeline, first, result

    def empty_detection(self, frame):
        return dict(
            objects=[],
            rejected_faces=[],
            suppressed_hypotheses=[],
            appearance_rejections=[],
            timing=dict(
                detector_ms=0.0, solve_ms=0.0, refinement_ms=0.0, appearance_ms=0.0, proposal_count=0
            ),
        ), np.zeros(frame.depth.shape, np.uint8)

    @staticmethod
    def normal(state):
        return state["pose"][:3, :3] @ state["normal_model"]

    def assert_anchor_matches_state(self, frame, event, state):
        np.testing.assert_allclose(event["new_normal_camera"], self.normal(state))
        np.testing.assert_allclose(
            event["new_quad"],
            frame.camera.project(transform_points(state["pose"], state["quad_model"])),
        )

    def test_sync_switch_to_large_face_survives_next_detector_miss(self):
        pipeline, first, _ = self.seeded()
        old_normal = self.normal(pipeline.states[0])
        current, truth, _ = self.frame(1)
        result, _ = pipeline.process(current)
        events = result["temporal"]["face_switches"]
        self.assertEqual(len(events), 1, result["temporal"])
        event = events[0]
        self.assertEqual(event["size_m"], 0.2)
        self.assertLess(event["old_view_cos"], 0.6)
        self.assertGreaterEqual(event["new_area_px"], event["old_area_px"] * 1.5)
        self.assertGreater(event["new_view_cos"], event["old_view_cos"])
        self.assertEqual(event["source_stamp_ns"], current.stamp_ns)
        self.assertEqual(event["current_stamp_ns"], current.stamp_ns)
        new_normal = self.normal(pipeline.states[0])
        self.assertLess(abs(old_normal @ new_normal), 0.5)
        obj = result["objects"][0]
        self.assert_anchor_matches_state(current, event, pipeline.states[0])
        self.assertEqual(obj["face_count"], len(obj["faces"]))
        np.testing.assert_allclose(obj["center_camera_m"], truth[0]["center_camera_m"], atol=0.012)
        next_frame, next_truth, _ = self.frame(2)
        pipeline.last_detection = None
        with patch.object(pipeline.detector, "process", side_effect=self.empty_detection), patch.object(
            pipeline, "_search_uncovered_colors", return_value=([], [])
        ):
            next_result, _ = pipeline.process(next_frame)
        self.assertEqual(len(next_result["objects"]), 1, next_result["temporal"])
        self.assertEqual(next_result["objects"][0]["source"], "current_rgbd_flow")
        self.assertEqual(next_result["temporal"]["face_switches"], [])
        self.assertGreater(abs(new_normal @ self.normal(pipeline.states[0])), 0.98)
        np.testing.assert_allclose(
            next_result["objects"][0]["center_camera_m"], next_truth[0]["center_camera_m"], atol=0.012
        )

    def test_same_face_balanced_faces_and_other_sizes_keep_established_features(self):
        for case, options in (
            ("same_face", dict(narrow=False)),
            ("balanced", dict(yaw=45.0)),
            ("not_narrow", dict(yaw=40.0)),
            ("large_box", dict(size=0.35)),
            ("disabled", dict(track_face_switch_sizes_m=())),
        ):
            with self.subTest(case=case):
                pipeline, first, _ = self.seeded(**options)
                old_normal = self.normal(pipeline.states[0])
                current, _, _ = self.frame(1, options.get("size", 0.2), options.get("yaw", 30.0))
                with patch.object(pipeline, "_seed", wraps=pipeline._seed) as seed:
                    result, _ = pipeline.process(current)
                self.assertEqual(len(result["objects"]), 1, result["temporal"])
                self.assertEqual(result["temporal"]["face_switches"], [])
                self.assertEqual(seed.call_count, 0)
                self.assertEqual(pipeline.states[0]["last_reseed"], first.stamp_ns)
                self.assertGreater(abs(old_normal @ self.normal(pipeline.states[0])), 0.98)

    def test_async_switch_uses_source_detection_revalidated_now(self):
        pipeline, first, detected = self.seeded()
        pipeline.asynchronous = True
        pipeline.worker_completed = (first, deepcopy(detected), [], 0.0, (None, ()))
        current, _, _ = self.frame(1)
        result, _ = pipeline.process(current)
        self.assertEqual(len(result["temporal"]["face_switches"]), 1, result["temporal"])
        event = result["temporal"]["face_switches"][0]
        self.assertEqual(event["source_stamp_ns"], first.stamp_ns)
        self.assertEqual(event["current_stamp_ns"], current.stamp_ns)
        obj = result["objects"][0]
        self.assertEqual(obj["source"], "current_rgbd_flow")
        self.assertEqual(obj["detector_seed_stamp_ns"], first.stamp_ns)
        self.assert_anchor_matches_state(current, event, pipeline.states[0])
        np.testing.assert_allclose(obj["T_camera_from_box"], pipeline.states[0]["pose"])
        next_frame, _, _ = self.frame(2)
        next_result, _ = pipeline.process(next_frame)
        self.assertEqual(len(next_result["objects"]), 1)
        self.assertEqual(next_result["temporal"]["face_switches"], [])

    def test_no_current_candidate_or_expired_detection_does_not_switch(self):
        for kind in ("missing", "expired", "stale_state"):
            with self.subTest(kind=kind):
                pipeline, first, detected = self.seeded()
                old_normal = self.normal(pipeline.states[0])
                pipeline.asynchronous = True
                current, _, _ = self.frame(1)
                if kind == "expired":
                    current.stamp_ns += 800_000_000
                    # Preserve a current old track independently of the old detection.
                    pipeline.states[0]["stamp"] = current.stamp_ns - 16_666_667
                    pipeline.worker_completed = (first, deepcopy(detected), [], 0.0, (None, ()))
                if kind == "stale_state":
                    obj = detected["objects"][0]
                    stale = pipeline._seed(first, obj, cv2.cvtColor(first.rgb, cv2.COLOR_RGB2GRAY))
                    candidate, event = pipeline._face_switch_candidate(
                        current,
                        dict(pipeline.states[0], stamp=current.stamp_ns),
                        obj,
                        stale,
                        cv2.cvtColor(current.rgb, cv2.COLOR_RGB2GRAY),
                    )
                    self.assertIsNone(candidate)
                    self.assertIsNone(event)
                result, _ = pipeline.process(current)
                self.assertEqual(len(result["objects"]), 1, result["temporal"])
                self.assertEqual(result["temporal"]["face_switches"], [])
                self.assertGreater(abs(old_normal @ self.normal(pipeline.states[0])), 0.98)

    def test_invalid_current_new_face_cannot_replace_valid_narrow_face(self):
        pipeline, first, detected = self.seeded()
        old_normal = self.normal(pipeline.states[0])
        pipeline.asynchronous = True
        pipeline.worker_completed = (first, deepcopy(detected), [], 0.0, (None, ()))
        current, _, _ = self.frame(1)
        largest = max(
            detected["objects"][0]["faces"],
            key=lambda f: cv2.contourArea(np.asarray(f["quad"], np.float32)),
        )
        quad = np.asarray(largest["quad"])
        interior = quad.mean(axis=0) + (quad - quad.mean(axis=0)) * 0.8
        mask = np.zeros(current.depth.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.rint(interior).astype(np.int32), 1)
        current.depth[mask > 0] = 0
        result, _ = pipeline.process(current)
        self.assertEqual(len(result["objects"]), 1, result["temporal"])
        self.assertTrue(result["temporal"]["detector_seed_failures"])
        self.assertEqual(result["temporal"]["face_switches"], [])
        self.assertGreater(abs(old_normal @ self.normal(pipeline.states[0])), 0.98)

    def test_missing_depth_after_switch_never_publishes_previous_pose(self):
        pipeline, _, _ = self.seeded()
        current, _, _ = self.frame(1)
        switched, _ = pipeline.process(current)
        self.assertEqual(len(switched["temporal"]["face_switches"]), 1)
        missing, _, _ = self.frame(2)
        missing.depth[:] = 0
        with patch.object(pipeline.detector, "process", side_effect=self.empty_detection), patch.object(
            pipeline, "_search_uncovered_colors", return_value=([], [])
        ):
            result, _ = pipeline.process(missing)
        self.assertEqual(result["objects"], [])
        self.assertEqual(result["temporal"]["face_switches"], [])
        self.assertEqual(result["temporal"]["stale_poses_published"], 0)


class FaceSwitchConfigTests(unittest.TestCase):
    def test_size_list_normalizes_and_supports_disabled_or_derived_configs(self):
        cfg = Config(track_face_switch_sizes_m=[0.35, 0.2, 0.2]).validate()
        self.assertEqual(cfg.track_face_switch_sizes_m, (0.2, 0.35))
        self.assertEqual(Config(track_face_switch_sizes_m=[]).validate().track_face_switch_sizes_m, ())
        cfg = Config(sizes_m=(0.35,)).validate()
        self.assertEqual(cfg.track_face_switch_sizes_m, (0.2,))

    def test_invalid_face_switch_settings_are_rejected(self):
        for value in ([-0.2], [0], [float("nan")], [float("inf")], [[0.2]], 0.2):
            with self.subTest(sizes=value), self.assertRaises(ValueError):
                Config(track_face_switch_sizes_m=value).validate()
        for value in (0, -1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(view_cos=value), self.assertRaises(ValueError):
                Config(track_face_switch_max_view_cos=value).validate()
        for value in (0, 1, float("nan"), float("inf"), True):
            with self.subTest(area_ratio=value), self.assertRaises(ValueError):
                Config(track_face_switch_min_area_ratio=value).validate()


if __name__ == "__main__":
    unittest.main()
