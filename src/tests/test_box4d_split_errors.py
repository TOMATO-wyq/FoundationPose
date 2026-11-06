"""Only an explicit measured-size rejection enables stack splitting."""

import unittest
from unittest.mock import patch

import numpy as np

from box4d.config import Config
from box4d.detector import Face
from box4d.errors import FaceSizeMismatch
from box4d.pose import Pipeline, estimate_face
from box4d.synthetic import Renderer, default_camera, world_box


class SplitErrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = Renderer(default_camera())

    def scene(self, **kwargs):
        cfg = Config(
            up_camera=[0, -1, 0], geometry_backend="numpy", allow_bottom_occlusion=False, **kwargs
        ).validate()
        frame, _, _ = self.renderer.render(
            [world_box((0, 0, 0.1), 0.2, 0, (0, -1.2, 0.1))], floor_height_camera_m=0.1
        )
        points = np.array([[-0.1, -0.1, 1.1], [0.1, -0.1, 1.1], [0.1, 0.1, 1.1], [-0.1, 0.1, 1.1]])
        face = Face(frame.camera.project(points), "observed_test", 1.0)
        return frame, face, cfg

    def run_rejection(self, error, *, inferred=False):
        frame, face, cfg = self.scene()
        face.inferred = inferred
        pipeline = Pipeline(cfg)
        with patch.object(
            pipeline.detector,
            "propose",
            return_value=(
                [face],
                dict(detector_ms=0.0, proposal_count=1),
                np.zeros(frame.depth.shape, np.uint8),
            ),
        ), patch.object(pipeline.detector, "join_color_halves", return_value=[]), patch(
            "box4d.pose.estimate_face", side_effect=error
        ), patch.object(
            pipeline.detector, "split_stack", return_value=[]
        ) as split:
            result, _ = pipeline.process(frame)
        self.assertEqual(result["objects"], [])
        self.assertEqual(result["rejected_faces"][0]["reason"], str(error))
        return split

    def test_actual_size_mismatch_has_specific_type_and_existing_log_message(self):
        frame, face, cfg = self.scene(sizes_m=(0.35,))
        with self.assertRaisesRegex(
            FaceSizeMismatch, "observed face dimensions do not match configured square sizes"
        ):
            estimate_face(frame, face, cfg)
        self.assertTrue(issubclass(FaceSizeMismatch, ValueError))

    def test_size_error_splits_even_with_completely_changed_message(self):
        split = self.run_rejection(FaceSizeMismatch("measured rectangle needs child-size validation"))
        split.assert_called_once()

    def test_plain_value_error_with_same_message_never_splits(self):
        split = self.run_rejection(
            ValueError("observed face dimensions do not match configured square sizes")
        )
        split.assert_not_called()

    def test_inferred_face_cannot_request_another_split(self):
        self.run_rejection(FaceSizeMismatch("size mismatch"), inferred=True).assert_not_called()

    def test_other_pose_and_depth_rejections_do_not_split(self):
        for message in (
            "reprojection error exceeds limit",
            "ambiguous planar PnP: depth does not distinguish candidates",
            "PnP/geometry inconsistent with measured depth plane",
            "nonplanar or mixed-depth face",
        ):
            with self.subTest(reason=message):
                self.run_rejection(ValueError(message)).assert_not_called()

    def test_real_pose_failure_keeps_plain_value_error(self):
        frame, face, cfg = self.scene(method="depth")
        with patch("box4d.pose._make_estimate", side_effect=ValueError("reprojection error exceeds limit")):
            with self.assertRaises(ValueError) as caught:
                estimate_face(frame, face, cfg)
        self.assertNotIsInstance(caught.exception, FaceSizeMismatch)

    def test_insufficient_current_depth_never_reaches_splitting(self):
        frame, face, cfg = self.scene()
        frame.depth[:] = 0
        pipeline = Pipeline(cfg)
        with patch.object(
            pipeline.detector,
            "propose",
            return_value=(
                [face],
                dict(detector_ms=0.0, proposal_count=1),
                np.zeros(frame.depth.shape, np.uint8),
            ),
        ), patch.object(pipeline.detector, "split_stack", return_value=[]) as split:
            result, _ = pipeline.process(frame)
        split.assert_not_called()
        self.assertEqual(result["objects"], [])
        self.assertIn("insufficient interior depth coverage", result["rejected_faces"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
