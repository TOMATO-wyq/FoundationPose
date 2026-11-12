"""KF-assisted recovery is tested against real synthetic RGB-D geometry."""

from contextlib import ExitStack
from copy import deepcopy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from box4d.config import Config
from box4d.results import empty_result
from box4d.synthetic import Renderer, default_camera, world_box
from box4d.temporal import TemporalPipeline
from box4d.tracking import Tracker


class TemporalPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = Renderer(default_camera())
        cls.frames = []
        cls.truth = []
        for index in range(17):
            box = world_box((index * 0.005, 0, 0.1), 0.2, 15.0)
            box["red_axis_local"] = [0, 0, 1]
            frame, truth, _ = cls.renderer.render([box], index, depth_noise_m=0.001)
            cls.frames.append(frame)
            cls.truth.append(truth[0])

    def pipeline(self, **options):
        cfg = Config(
            up_camera=[0, -1, 0], analyze_colors=True, two_color_sizes_m=(0.2,),
            require_target_colors=True, temporal_recovery=True, plane_grid_step=8, **options
        )
        return TemporalPipeline(cfg), Tracker(cfg)

    def no_detector(self, pipeline):
        stack = ExitStack()
        stack.enter_context(patch.object(
            pipeline.detector, "process",
            side_effect=lambda frame: (empty_result(frame, pipeline.cfg), np.zeros(frame.depth.shape, np.uint8)),
        ))
        stack.enter_context(patch.object(pipeline, "_search_uncovered_colors", return_value=([], [])))
        return stack

    def warmup(self, pipeline, tracker):
        first, _ = pipeline.process(deepcopy(self.frames[0]))
        tracker.update(first)
        self.assertEqual(len(first["objects"]), 1)
        with self.no_detector(pipeline):
            for frame in self.frames[1:6]:
                result, _ = pipeline.process(deepcopy(frame))
                tracker.update(result)
                self.assertEqual(len(result["objects"]), 1, result["temporal"])
        return result

    def test_missing_depth_predicts_motion_then_recovers_without_new_detection(self):
        outcomes = {}
        for enabled in (False, True):
            # A tight ID association radius isolates prediction's contribution
            # to identity continuity; it must not discard current RGB-D poses.
            pipeline, tracker = self.pipeline(motion_prediction=enabled, track_max_distance_m=0.015)
            last = self.warmup(pipeline, tracker)
            identity = last["objects"][0]["track_id"]
            last_stamp = self.frames[5].stamp_ns
            old_motion = pipeline.states[0]["motion"]
            with self.no_detector(pipeline):
                for index in range(6, 11):
                    frame = deepcopy(self.frames[index])
                    frame.depth[:] = 0
                    result, _ = pipeline.process(frame)
                    tracker.update(result)
                    self.assertEqual(result["objects"], [])
                    self.assertEqual(result["valid_object_count"], 0)
                    self.assertEqual(len(result["predicted_objects"]), int(enabled))
                    if enabled:
                        prediction = result["predicted_objects"][0]
                        self.assertFalse(prediction["valid"])
                        self.assertEqual(prediction["track_id"], identity)
                        self.assertEqual(prediction["last_observed_stamp_ns"], last_stamp)
                        self.assertGreater(prediction["center_camera_m"][0], last["objects"][0]["center_camera_m"][0])
                        np.testing.assert_allclose(
                            prediction["center_camera_m"], self.truth[index]["center_camera_m"], atol=0.012
                        )
                        self.assertIs(pipeline.states[0]["motion"], old_motion)
                recovered, _ = pipeline.process(deepcopy(self.frames[11]))
                tracker.update(recovered)
            outcomes[enabled] = recovered
            if enabled:
                self.assertEqual(recovered["objects"][0]["track_id"], identity)
                self.assertEqual(recovered["objects"][0]["source"], "current_rgbd_flow")
                self.assertEqual(recovered["predicted_objects"], [])
                self.assertTrue(recovered["temporal"]["tracking_checks"][0]["flow_prediction_used"])
                self.assertEqual(pipeline.states[0]["motion"].observations, old_motion.observations + 1)
                np.testing.assert_allclose(
                    recovered["objects"][0]["center_camera_m"], self.truth[11]["center_camera_m"], atol=0.012
                )
                details = {}
                reseed = pipeline._seed(
                    self.frames[11], recovered["objects"][0],
                    cv2.cvtColor(self.frames[11].rgb, cv2.COLOR_RGB2GRAY),
                    previous_state=pipeline.states[0], diagnostics=details,
                )
                self.assertIsNotNone(reseed, details)
                self.assertEqual(reseed["motion"].observations, old_motion.observations + 1)
                self.assertTrue(any(t.get("position_reference") == "kf_prediction"
                                    and t.get("accepted") for t in details["pose_trials"]))
        self.assertEqual(len(outcomes[True]["objects"]), 1)
        self.assertEqual(len(outcomes[False]["objects"]), 1)
        self.assertEqual(outcomes[False]["objects"][0]["source"], "current_rgbd_flow")
        np.testing.assert_allclose(
            outcomes[False]["objects"][0]["center_camera_m"], self.truth[11]["center_camera_m"], atol=.012
        )
        self.assertNotEqual(outcomes[False]["objects"][0]["track_id"], identity)

    def test_predictions_expire_from_last_observation_not_last_prediction(self):
        pipeline, tracker = self.pipeline()
        self.warmup(pipeline, tracker)
        with self.no_detector(pipeline):
            for index in range(6, 17):
                frame = deepcopy(self.frames[index])
                if index == 16:
                    frame.stamp_ns = self.frames[5].stamp_ns + 310_000_000
                frame.depth[:] = 0
                result, _ = pipeline.process(frame)
                tracker.update(result)
                self.assertEqual(result["objects"], [])
        self.assertEqual(result["predicted_objects"], [])
        self.assertEqual(pipeline.states, [])

    def test_fresh_redetection_preserves_velocity_without_duplicate_tracks(self):
        pipeline, tracker = self.pipeline()
        last = self.warmup(pipeline, tracker)
        previous_motion = pipeline.states[0]["motion"]
        # Force a feature failure, but allow the actual current-image detector.
        with patch.object(pipeline, "_advance", side_effect=ValueError("feature outage")):
            current, _ = pipeline.process(deepcopy(self.frames[7]))
        tracker.update(current)
        self.assertEqual(len(current["objects"]), 1)
        self.assertEqual(current["objects"][0]["track_id"], last["objects"][0]["track_id"])
        self.assertEqual(len(pipeline.states), 1)
        motion = pipeline.states[0]["motion"]
        self.assertEqual(motion.observations, previous_motion.observations + 1)
        self.assertGreater(motion.predict(self.frames[8].stamp_ns)[0][3], 0.05)

    def test_coordinate_frame_change_clears_motion_and_predictions(self):
        pipeline, tracker = self.pipeline()
        self.warmup(pipeline, tracker)
        frame = deepcopy(self.frames[6])
        frame.camera.frame_id = "different_camera_optical_frame"
        with self.no_detector(pipeline):
            current, _ = pipeline.process(frame)
        tracker.update(current)
        self.assertEqual(current["objects"], [])
        self.assertEqual(current["predicted_objects"], [])
        self.assertEqual(pipeline.states, [])


if __name__ == "__main__":
    unittest.main()
