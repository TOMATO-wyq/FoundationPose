"""Parity at the Python/C++ boundary, including outliers and clipped edges."""

import unittest
from unittest.mock import patch
from itertools import permutations
from types import SimpleNamespace

import numpy as np

from box4d.config import Config
from box4d.detector import _edge_support
from box4d.geometry import fit_plane, order_quad, rigid_fit, square_points
from box4d.native import geometry_kernel


class BackendTests(unittest.TestCase):
    def test_explicit_cpp_does_not_silently_run_numpy(self):
        with patch("box4d.native._load", return_value=None):
            self.assertIsNone(geometry_kernel("auto"))
            with self.assertRaisesRegex(RuntimeError, "build_box4d_native.sh"):
                geometry_kernel("cpp")
        with self.assertRaisesRegex(ValueError, "geometry_backend"):
            Config(geometry_backend="cuda").validate()

    def test_stale_cpp_build_requires_rebuild_instead_of_mixed_versions(self):
        with patch("box4d.native._load", return_value=SimpleNamespace()):
            for backend in ("auto", "cpp"):
                with self.assertRaisesRegex(RuntimeError, "build_box4d_native.sh"):
                    geometry_kernel(backend)
            self.assertIsNone(geometry_kernel("numpy"))


@unittest.skipUnless(geometry_kernel() is not None, "build CPU kernels to test native parity")
class NativeParityTests(unittest.TestCase):
    def test_cpp_opencv_obeys_thread_budget_without_resetting_it_for_child_rois(self):
        import cv2
        from box4d.pose import Pipeline

        native = geometry_kernel("cpp")
        previous_cpp, previous_python = native.opencv_num_threads(), cv2.getNumThreads()
        try:
            Pipeline(Config(geometry_backend="cpp", opencv_threads=2))
            self.assertEqual(native.opencv_num_threads(), 2)
            self.assertEqual(cv2.getNumThreads(), 2)
            Pipeline(Config(geometry_backend="cpp", opencv_threads=1), configure_opencv=False)
            self.assertEqual(native.opencv_num_threads(), 2)
        finally:
            native.configure_opencv(previous_cpp)
            cv2.setNumThreads(previous_python)

    def test_quad_order_all_permutations_and_degenerate_inputs(self):
        rng = np.random.default_rng(27)
        for _ in range(20):
            quad = np.array([[10.0, 13.0], [160.0, 24.0], [150.0, 150.0], [3.0, 138.0]])
            quad += rng.normal(0, 2, (4, 2))
            for order in permutations(range(4)):
                points = quad[list(order)]
                np.testing.assert_array_equal(
                    order_quad(points, backend="cpp"), order_quad(points, backend="numpy")
                )
        cases = [
            np.zeros((4, 2)),
            np.array([[0, 0], [1, 0], [2, 0], [3, 0]]),
            np.array([[0, 0], [5, 0], [1, 1], [0, 5]]),
            np.full((4, 2), np.nan),
            np.array([[0, 0], [0.2, 0], [0.2, 0.2], [0, 0.2]]),
        ]
        for points in cases:
            for backend in ("cpp", "numpy"):
                with self.assertRaises(ValueError):
                    order_quad(points, backend=backend)

    def test_rigid_fit_planar_spatial_noise_and_reflection_match_reference(self):
        rng = np.random.default_rng(36)
        for count in (4, 40):
            for noise in (0.0, 0.002):
                source = square_points(0.2) if count == 4 else rng.normal(0, 0.2, (count, 3))
                for reflect in (False, True):
                    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                    if reflect:
                        rotation[:, -1] *= -1
                    target = source @ rotation.T + [0.1, -0.2, 1.3] + rng.normal(0, noise, source.shape)
                    cpp = rigid_fit(source, target, backend="cpp")
                    reference = rigid_fit(source, target, backend="numpy")
                    # Planar data still determine one proper rotation; reflection correction is retained.
                    np.testing.assert_allclose(cpp, reference, atol=2e-12)
                    self.assertAlmostEqual(np.linalg.det(cpp[:3, :3]), 1.0)

    def test_planes_with_noise_outliers_and_invalid_depth_match_reference(self):
        rng = np.random.default_rng(381)
        for count in (32, 384, 2400):
            for slope in ((0.0, 0.0), (0.17, -0.34), (1.8, 0.9)):
                with self.subTest(count=count, slope=slope):
                    xy = rng.uniform(-0.2, 0.2, (count, 2))
                    z = 1.4 + xy @ slope + rng.normal(0, 0.0015, count)
                    z[: count // 5] += rng.uniform(0.04, 0.25, count // 5)
                    points = np.vstack((np.column_stack((xy, z)), [[np.nan, 0, 1], [0, 0, 0], [1, 1, -1]]))
                    reference = fit_plane(points, backend="numpy")
                    native = fit_plane(points, backend="cpp")
                    np.testing.assert_allclose(native.normal, reference.normal, atol=2e-12)
                    np.testing.assert_allclose(
                        [native.d, native.rmse_m, native.support],
                        [reference.d, reference.rmse_m, reference.support],
                        atol=2e-12,
                    )

    def test_degenerate_and_insufficient_planes_reject_in_both_backends(self):
        cases = [
            np.zeros((50, 3)),
            np.tile([0.0, 0.0, 1.0], (50, 1)),
            np.column_stack((np.arange(50) * 0.01, np.zeros(50), np.ones(50))),
        ]
        for backend in ("numpy", "cpp"):
            for points in cases:
                with self.subTest(backend=backend):
                    with self.assertRaises(ValueError):
                        fit_plane(points, backend=backend)

    def test_parallel_surfaces_remain_separate_ransac_layers(self):
        rng = np.random.default_rng(52)
        xy = rng.uniform(-0.2, 0.2, (500, 2))
        z = np.r_[np.full(300, 1.1), np.full(200, 1.3)]
        points = np.column_stack((xy, z))
        for backend in ("numpy", "cpp"):
            with self.subTest(backend=backend):
                first = fit_plane(points, backend=backend)
                self.assertAlmostEqual(first.support, 0.6)
                self.assertAlmostEqual(first.d, 1.1)
                remaining = points[np.abs(points @ first.normal + first.d) >= 0.008]
                second = fit_plane(remaining, backend=backend)
                self.assertAlmostEqual(second.support, 1.0)
                self.assertAlmostEqual(second.d, 1.3)

    def test_fractional_and_clipped_edge_sampling_matches_reference(self):
        rng = np.random.default_rng(43)
        distance = rng.uniform(0, 8, (70, 90)).astype(np.float32)
        distance[10:15] = 3.5  # The shared boundary includes distance == 3.5.
        quads = [
            np.array([[-20.0, 10.0], [110.0, 10.0], [110.0, 80.0], [-20.0, 80.0]]),
            np.array([[10.5, 10.5], [25.5, 10.5], [25.5, 25.5], [10.5, 25.5]]),
        ]
        quads.extend(rng.uniform(-20, 100, (120, 4, 2)))
        for quad in quads:
            a, sides_a = _edge_support(quad, distance, backend="numpy")
            b, sides_b = _edge_support(quad, distance, backend="cpp")
            self.assertEqual(a, b)
            self.assertEqual(sides_a, sides_b)

    def test_native_boundary_rejects_wrong_shapes_and_sample_indices(self):
        native = geometry_kernel("cpp")
        with self.assertRaises(ValueError):
            native.edge_support(np.ones((5, 2)), np.zeros((10, 10), np.float32))
        with self.assertRaises(ValueError):
            native.fit_plane(np.ones((30, 3)), np.array([[0, 1, 30]]), 0.008, 24)


if __name__ == "__main__":
    unittest.main()
