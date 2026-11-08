"""Prediction overlays remain visibly separate from current measurements."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from box4d.geometry import Camera
from box4d.io import Frame
from box4d import visualization


class PredictionDisplayTests(unittest.TestCase):
    def setUp(self):
        camera = Camera(640, 480, [[500, 0, 320], [0, 500, 240], [0, 0, 1]])
        self.frame = Frame(
            np.zeros((480, 640, 3), np.uint8),
            np.zeros((480, 640), np.float32),
            camera,
            1_000_000_000,
        )
        pose = np.eye(4)
        pose[:3, 3] = [0.2, 0.0, 1.0]
        self.prediction = dict(
            track_id=2,
            size_m=0.2,
            center_camera_m=pose[:3, 3].tolist(),
            T_camera_from_box=pose.tolist(),
            source="kf_prediction",
            status="predicted",
            valid=False,
            last_observed_stamp_ns=900_000_000,
            prediction_age_s=0.1,
            position_covariance_m2=(np.eye(3) * 0.001).tolist(),
        )
        self.result = dict(
            objects=[],
            predicted_objects=[self.prediction],
            timing=dict(processing_ms=1.0),
            frame_status="partial_or_rejected",
        )

    def test_predictions_draw_without_faces_in_both_display_modes(self):
        original = deepcopy(self.result)
        images = []
        for smoothed in (False, True):
            with patch.object(visualization, "_dashed", wraps=visualization._dashed) as dashed:
                with patch.object(cv2, "putText", wraps=cv2.putText) as put_text:
                    images.append(visualization.overlay(self.frame, self.result, smoothed=smoothed))
            self.assertEqual(dashed.call_count, 12)
            texts = [call.args[1] for call in put_text.call_args_list]
            self.assertTrue(any("PRED ID2 20cm age=0.10s" in text for text in texts))
            self.assertTrue(any("observed=0 predicted=1" in text for text in texts))
            self.assertIn("KF PREDICTION - no current target observation", texts)
            self.assertTrue(np.any(np.all(images[-1][50:] == (220, 80, 255), axis=2)))
        np.testing.assert_array_equal(images[0][43:], images[1][43:])
        self.assertEqual(self.result, original)

    def test_measurement_and_prediction_keep_distinct_labels_and_colors(self):
        pose = np.eye(4)
        pose[:3, 3] = [-0.2, 0.0, 1.0]
        self.result["objects"] = [dict(
            track_id=1,
            size_m=0.2,
            evidence="observed_faces",
            faces=[],
            T_camera_from_box=pose.tolist(),
            yaw_u_rad=None,
            dimensions_nominal=False,
        )]
        with patch.object(cv2, "putText", wraps=cv2.putText) as put_text:
            image = visualization.overlay(self.frame, self.result)
        texts = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("ID1 20cm yaw: UNREFERENCED", texts)
        self.assertIn("PRED ID2 20cm age=0.10s", texts)
        self.assertTrue(any("observed=1 predicted=1" in text for text in texts))
        self.assertTrue(np.any(np.all(image[50:, :320] == (60, 245, 70), axis=2)))
        self.assertTrue(np.any(np.all(image[50:, 320:] == (220, 80, 255), axis=2)))

    def test_invalid_or_offscreen_predictions_are_not_drawn(self):
        baseline = visualization.overlay(self.frame, dict(self.result, predicted_objects=[]))
        for center in ([0, 0, -1], [0, 0, 0.05], [np.nan, 0, 1], [np.inf, 0, 1], [1e12, 0, 1], [3, 0, 1]):
            with self.subTest(center=center):
                prediction = deepcopy(self.prediction)
                pose = np.eye(4)
                pose[:3, 3] = center
                prediction["T_camera_from_box"] = pose.tolist()
                image = visualization.overlay(self.frame, dict(self.result, predicted_objects=[prediction]))
                np.testing.assert_array_equal(image, baseline)
        for bad_projection in (np.full((9, 2), np.nan), np.full((9, 2), 1e20)):
            with patch.object(self.frame.camera, "project", return_value=bad_projection):
                np.testing.assert_array_equal(visualization.overlay(self.frame, self.result), baseline)

    def test_legacy_result_without_prediction_field_is_unchanged(self):
        legacy = deepcopy(self.result)
        legacy.pop("predicted_objects")
        with patch.object(cv2, "putText", wraps=cv2.putText) as put_text:
            image = visualization.overlay(self.frame, legacy)
        texts = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("POSE REJECTED - see 08 Current tracking", texts)
        self.assertFalse(any("PRED" in text for text in texts))
        np.testing.assert_array_equal(image, visualization.overlay(self.frame, dict(legacy, predicted_objects=[])))


if __name__ == "__main__":
    unittest.main()
