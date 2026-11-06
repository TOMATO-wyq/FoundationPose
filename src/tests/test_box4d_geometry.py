"""Geometric correctness tests with independent analytic projections."""
import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from box4d.config import Config
from box4d.geometry import (Camera, angle_difference, box_from_face, fit_plane, order_quad,
                            ray_plane, rigid_fit, rotation_z, square_points, transform,
                            transform_points, upright_from_camera, yaw_from_pose)
from box4d.io import Frame, Dataset, DatasetWriter
from box4d.synthetic import C_FROM_WORLD, Renderer, default_camera, world_box
from box4d.ros_input import up_from_plane_roi


class GeometryTests(unittest.TestCase):
    def test_distorted_projection_backprojection(self):
        camera = Camera(640, 480, [[460, 0, 319], [0, 455, 241], [0, 0, 1]], [.08, -.03, .001, -.002, .01])
        points = np.array([[-.3, -.2, 1.2], [.2, .15, .8], [.03, -.01, 2.]])
        reconstructed = camera.rays(camera.project(points)) * points[:, 2, None]
        np.testing.assert_allclose(reconstructed, points, atol=1e-6)

    def test_crop_and_resize_preserves_camera_rays(self):
        camera = default_camera()
        crop = camera.crop_resize((100, 50, 500, 350), 200, 150)
        pixels = np.array([[180., 130.], [400., 280.]])
        np.testing.assert_allclose(camera.rays(pixels), crop.rays((pixels - (100, 50)) * .5), atol=1e-12)

    def test_unknown_distortion_and_invalid_intrinsics_rejected(self):
        with self.assertRaises(ValueError):
            Camera(640, 480, np.eye(3), (), "unknown")
        with self.assertRaises(ValueError):
            Camera(640, 480, [[0, 0, 0], [0, 1, 0], [0, 0, 1]])
        with self.assertRaises(ValueError):
            Camera(640, 480, np.eye(3), [.1, 0, 0, 0], rectified=True)

    def test_robust_plane_with_outliers(self):
        rng = np.random.default_rng(51)
        xy = rng.uniform(-.2, .2, (200, 2))
        points = np.column_stack((xy, 1.2 + .1 * xy[:, 0] - .2 * xy[:, 1]))
        points += rng.normal(0, .0005, points.shape)
        points[:35, 2] += .12
        plane = fit_plane(points)
        self.assertGreater(plane.support, .8)
        self.assertLess(plane.rmse_m, .001)
        expected = np.array([.1, -.2, -1.])
        expected /= np.linalg.norm(expected)
        self.assertGreater(plane.normal @ expected, .9999)

    def test_collinear_plane_and_parallel_intersections_rejected(self):
        with self.assertRaises(ValueError):
            fit_plane(np.column_stack((np.arange(30) * .01, np.zeros(30), np.ones(30))))
        from box4d.geometry import Plane
        with self.assertRaises(ValueError):
            ray_plane(default_camera(), [[320, 240]], Plane(np.array([1., 0, 0]), -1, 0, 1))

    def test_square_face_pose_and_center_offset(self):
        face_pose = transform(np.diag([1., -1., -1.]), [.03, .05, 1.])
        measured = transform_points(face_pose, square_points(.35))
        recovered = rigid_fit(square_points(.35), measured)
        np.testing.assert_allclose(recovered, face_pose, atol=1e-12)
        box, kind, tilt = box_from_face(recovered, .35, [0, -1, 0])
        np.testing.assert_allclose(box[:3, 3], [.03, .05, 1.175], atol=1e-12)
        self.assertEqual(kind, "side")
        # acos(1-eps) amplifies SVD rounding near zero. NumPy/BLAS builds
        # differ by about 1e-6 degrees even when the matrix matches to 1e-12.
        self.assertLess(tilt, 1e-5)

    def test_upright_reference_and_periodic_yaw(self):
        u = upright_from_camera([0, -1, 0])
        np.testing.assert_allclose(u @ u.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(u), 1.)
        for angle in [0, 15, 89, 90, 135, -1]:
            pose = transform(C_FROM_WORLD @ rotation_z(np.deg2rad(angle)), [0, 0, 1])
            yaw = yaw_from_pose(pose, u)
            self.assertLess(abs(angle_difference(yaw, np.deg2rad(angle))), 1e-12)
        self.assertIsNone(yaw_from_pose(np.eye(4), None))
        self.assertAlmostEqual(np.rad2deg(angle_difference(np.deg2rad(1), np.deg2rad(89))), 2.)

    def test_non_upright_face_detects_in_plane_roll(self):
        face = transform(np.diag([1., -1., -1.]) @ rotation_z(np.pi / 4), [0, 0, 1])
        _, kind, tilt = box_from_face(face, .35, [0, -1, 0])
        self.assertEqual(kind, "side")
        self.assertGreater(tilt, 40)

    def test_quad_sorting_does_not_make_a_rectangle(self):
        p = np.array([[100., 100.], [250., 110.], [230., 230.], [120., 240.]])
        q = order_quad(p[[2, 0, 3, 1]])
        np.testing.assert_allclose(q, p)
        self.assertGreater(abs((q[1] - q[0]) @ (q[2] - q[1])), 1)

    def test_known_horizontal_reference_roi(self):
        frame, _, _ = Renderer(default_camera()).render([])
        up, report = up_from_plane_roi(frame, [100, 360, 540, 470])
        np.testing.assert_allclose(up, [0, -1, 0], atol=1e-6)
        self.assertGreater(report["plane_support"], .99)


class DatasetTests(unittest.TestCase):
    def test_roundtrip_and_duplicate_stamp_rejection(self):
        frame, _, _ = Renderer(default_camera(160, 120)).render([])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sequence"
            writer = DatasetWriter(path, frame.camera, [0, -1, 0], synthetic=True)
            writer.write(frame)
            with self.assertRaises(ValueError):
                writer.write(frame)
            writer.close()
            data = Dataset(path)
            restored = data.frame(0)
            np.testing.assert_array_equal(restored.rgb, frame.rgb)
            np.testing.assert_array_equal(restored.depth, frame.depth)
            self.assertEqual(restored.camera.calibration_id, frame.camera.calibration_id)
            self.assertEqual(restored.stamp_ns, frame.stamp_ns)

    def test_partial_capture_requires_explicit_opt_in(self):
        frame, _, _ = Renderer(default_camera(160, 120)).render([])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sequence"
            writer = DatasetWriter(path, frame.camera)
            writer.write(frame)
            writer.close(complete=False)
            with self.assertRaises(ValueError):
                Dataset(path)
            self.assertEqual(len(Dataset(path, allow_partial=True)), 1)

    def test_uint16_depth_not_silently_interpreted_as_meters(self):
        camera = default_camera(160, 120)
        with self.assertRaises(ValueError):
            Frame(np.zeros((120, 160, 3), np.uint8), np.ones((120, 160), np.uint16), camera, 1)

    def test_config_rejects_unknown_fields_and_invalid_reference(self):
        with self.assertRaises(ValueError):
            Config(up_camera=[0, 0, 0]).validate()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text('{"size_mm": 350}')
            with self.assertRaises(ValueError):
                Config.load(path)


if __name__ == "__main__":
    unittest.main()
