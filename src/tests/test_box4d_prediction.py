"""Missing detections use bounded motion predictions without inventing evidence."""

from copy import deepcopy
import json
import unittest

import numpy as np

from box4d.config import Config
from box4d.tracking import Tracker


def observation(x, *, size=0.2, z=1.2):
    pose = np.eye(4)
    angle = 0.3
    pose[:3, :3] = [
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ]
    pose[:3, 3] = [x, 0, z]
    return dict(
        center_camera_m=pose[:3, 3].tolist(), T_camera_from_box=pose.tolist(),
        size_m=size, yaw_u_rad=angle, yaw_reference_ready=True, valid=True,
        faces=[dict(quad=[[1, 2], [3, 4], [5, 6], [7, 8]])],
        metrics=dict(depth_median_residual_m=0.001), quality_score=0.9,
        evidence="measured_current_rgbd",
    )


def frame(time_s, objects=(), **changes):
    result = dict(
        stamp_ns=round(time_s * 1e9), frame_id="camera_optical",
        calibration_id="camera-test", up_camera=[0, -1, 0],
        objects=list(objects), rejected_faces=[],
    )
    result.update(changes)
    return result


class TrackerPredictionTests(unittest.TestCase):
    def moving(self, **settings):
        tracker = Tracker(Config(**settings).validate())
        first = tracker.update(frame(1.0, [observation(0.0)]))
        second = tracker.update(frame(1.1, [observation(0.1)]))
        identity = first["objects"][0]["track_id"]
        self.assertEqual(second["objects"][0]["track_id"], identity)
        return tracker, identity, second

    def test_missing_frame_predicts_motion_separately_and_keeps_raw_measurements(self):
        tracker, identity, measured = self.moving()
        raw = deepcopy(measured["objects"][0])
        missing = tracker.update(frame(1.2))
        self.assertEqual(missing["objects"], [])
        self.assertEqual(missing["frame_status"], "no_target")
        self.assertEqual(missing["lost_track_ids"], [identity])
        self.assertEqual(len(missing["predicted_objects"]), 1)
        prediction = missing["predicted_objects"][0]
        self.assertEqual(prediction["track_id"], identity)
        self.assertFalse(prediction["valid"])
        self.assertEqual(prediction["status"], "predicted")
        self.assertEqual(prediction["source"], "kf_prediction")
        self.assertGreater(prediction["center_camera_m"][0], 0.15)
        self.assertLess(prediction["center_camera_m"][0], 0.23)
        self.assertEqual(prediction["last_observed_stamp_ns"], 1_100_000_000)
        self.assertAlmostEqual(prediction["prediction_age_s"], 0.1)
        pose = np.asarray(prediction["T_camera_from_box"])
        np.testing.assert_array_equal(pose[:3, :3], np.asarray(raw["T_camera_from_box"])[:3, :3])
        np.testing.assert_array_equal(pose[:3, 3], prediction["center_camera_m"])
        self.assertEqual(raw, measured["objects"][0])
        self.assertEqual(raw["center_camera_m"], [0.1, 0.0, 1.2])
        for key in ("faces", "metrics", "quality_score", "evidence", "yaw_u_rad"):
            self.assertNotIn(key, prediction)
        covariance = np.asarray(prediction["position_covariance_m2"])
        self.assertEqual(covariance.shape, (3, 3))
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) >= 0))
        json.dumps(missing, allow_nan=False)

    def test_reacquisition_uses_predicted_position_outside_old_position_gate(self):
        tracker, identity, _ = self.moving()
        tracker.update(frame(1.2))
        # 20 cm from the last observation is outside the default 15 cm gate.
        recovered = tracker.update(frame(1.3, [observation(0.3)]))
        self.assertEqual(recovered["objects"][0]["track_id"], identity)
        self.assertEqual(recovered["objects"][0]["center_camera_m"], [0.3, 0.0, 1.2])
        self.assertEqual(recovered["predicted_objects"], [])
        self.assertEqual(recovered["lost_track_ids"], [])

    def test_repeated_predictions_do_not_refresh_last_observation_or_reduce_uncertainty(self):
        tracker, identity, _ = self.moving()
        state = tracker.tracks[identity]["motion"]
        first = tracker.update(frame(1.2))["predicted_objects"][0]
        second = tracker.update(frame(1.3))["predicted_objects"][0]
        self.assertIs(tracker.tracks[identity]["motion"], state)
        self.assertEqual(state.stamp_ns, 1_100_000_000)
        self.assertEqual(state.observations, 2)
        self.assertAlmostEqual(second["prediction_age_s"], 0.2)
        self.assertGreater(second["center_camera_m"][0], first["center_camera_m"][0])
        self.assertGreater(
            np.trace(second["position_covariance_m2"]),
            np.trace(first["position_covariance_m2"]),
        )
        expired = tracker.update(frame(1.400000001))
        self.assertEqual(expired["predicted_objects"], [])
        self.assertEqual(expired["objects"], [])

    def test_single_observation_does_not_publish_unlearned_velocity_prediction(self):
        tracker = Tracker(Config().validate())
        tracker.update(frame(1.0, [observation(0.0)]))
        missing = tracker.update(frame(1.1))
        self.assertEqual(missing["predicted_objects"], [])

    def test_uncertain_predictions_are_not_published(self):
        tracker, _, _ = self.moving(prediction_max_std_m=0.001)
        self.assertEqual(tracker.update(frame(1.2))["predicted_objects"], [])

    def test_prediction_disabled_preserves_last_observation_association(self):
        tracker, identity, _ = self.moving(motion_prediction=False)
        missing = tracker.update(frame(1.2))
        self.assertEqual(missing["objects"], [])
        self.assertEqual(missing["predicted_objects"], [])
        recovered = tracker.update(frame(1.3, [observation(0.3)]))
        self.assertNotEqual(recovered["objects"][0]["track_id"], identity)

    def test_same_size_crossing_tracks_use_motion_and_global_assignment(self):
        tracker = Tracker(Config().validate())
        first = tracker.update(frame(1.0, [observation(-0.14), observation(0.14)]))
        left_id, right_id = [obj["track_id"] for obj in first["objects"]]
        tracker.update(frame(1.1, [observation(-0.06), observation(0.06)]))
        missing = tracker.update(frame(1.2))
        self.assertEqual(len(missing["predicted_objects"]), 2)
        # Detector order is reversed and each object has passed the old center of the other.
        recovered = tracker.update(frame(1.3, [observation(-0.10), observation(0.10)]))
        self.assertEqual([obj["track_id"] for obj in recovered["objects"]], [right_id, left_id])
        self.assertEqual(recovered["predicted_objects"], [])

    def test_only_missing_track_is_predicted_and_size_gate_still_applies(self):
        tracker, identity, _ = self.moving()
        changed_size = tracker.update(frame(1.2, [observation(0.2, size=0.35)]))
        self.assertNotEqual(changed_size["objects"][0]["track_id"], identity)
        self.assertEqual([obj["track_id"] for obj in changed_size["predicted_objects"]], [identity])

    def test_coordinate_reference_changes_clear_motion_history(self):
        for change in (
            dict(calibration_id="camera-new"),
            dict(frame_id="other_camera_optical"),
            dict(up_camera=[0, -0.8, -0.6]),
        ):
            with self.subTest(change=change):
                tracker, identity, _ = self.moving()
                reset = tracker.update(frame(1.2, **change))
                self.assertEqual(reset["predicted_objects"], [])
                self.assertEqual(tracker.tracks, {})
                detected = tracker.update(frame(1.3, [observation(0.3)], **change))
                self.assertNotEqual(detected["objects"][0]["track_id"], identity)
                self.assertEqual(tracker.update(frame(1.4, **change))["predicted_objects"], [])

    def test_repeated_and_out_of_order_frames_do_not_mutate_motion_state(self):
        tracker, identity, _ = self.moving()
        state = tracker.tracks[identity]["motion"]
        tracker.update(frame(1.2))
        for time in (1.2, 1.05):
            with self.subTest(time=time), self.assertRaises(ValueError):
                tracker.update(frame(time, [observation(9.0)]))
        self.assertIs(tracker.tracks[identity]["motion"], state)
        self.assertEqual(tracker.last_stamp, 1_200_000_000)

    def test_expired_filter_restarts_even_with_longer_association_memory(self):
        tracker, identity, _ = self.moving(track_max_gap_s=1.0)
        recovered = tracker.update(frame(1.5, [observation(0.1)]))
        self.assertEqual(recovered["objects"][0]["track_id"], identity)
        self.assertEqual(tracker.tracks[identity]["motion"].observations, 1)
        self.assertEqual(tracker.update(frame(1.6))["predicted_objects"], [])

    def test_rejected_current_frame_retains_diagnostics_with_separate_prediction(self):
        tracker, _, _ = self.moving()
        rejected = tracker.update(frame(1.2, frame_status="partial_or_rejected"))
        self.assertEqual(rejected["frame_status"], "partial_or_rejected")
        self.assertEqual(rejected["objects"], [])
        self.assertEqual(len(rejected["predicted_objects"]), 1)


if __name__ == "__main__":
    unittest.main()
