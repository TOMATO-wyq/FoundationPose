"""Exact CPU image-kernel parity, including invalid depth and threshold endpoints."""

from dataclasses import replace
import unittest

import numpy as np

from box4d.appearance import classify_hsv
from box4d.config import Config
from box4d.detector import Detector
from box4d.edges import rgb_depth_edges
from box4d.geometry import Camera
from box4d.io import Frame
from box4d.native import geometry_kernel
from box4d.synthetic import Renderer, default_camera, demo_scenes


@unittest.skipUnless(geometry_kernel() is not None, "build CPU kernels to test native image parity")
class NativeImageParityTests(unittest.TestCase):
    def color_parity(self, hsv, **kwargs):
        cfg = Config(geometry_backend="numpy", **kwargs).validate()
        expected = classify_hsv(hsv, cfg)
        actual = classify_hsv(hsv, replace(cfg, geometry_backend="cpp"))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, hsv.shape[:-1])
        self.assertEqual(actual.dtype, np.int8)
        return actual

    def depth_parity(self, depth, minimum=0.1, maximum=4.0, threshold=0.02):
        valid = np.isfinite(depth) & (depth >= minimum) & (depth <= maximum)
        expected = np.zeros(depth.shape, np.uint8)
        with np.errstate(invalid="ignore"):
            x = valid[:, 1:] & valid[:, :-1] & (np.abs(depth[:, 1:] - depth[:, :-1]) > threshold)
            y = valid[1:] & valid[:-1] & (np.abs(depth[1:] - depth[:-1]) > threshold)
        expected[:, 1:][x] = 255
        expected[1:][y] = 255
        actual = geometry_kernel("cpp").depth_edges(depth, minimum, maximum, threshold)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.uint8)
        return actual

    def dense_plane_parity(self, depth, rays, normal, offset, minimum=0.1, maximum=4.0, threshold=0.008):
        flat = depth.ravel()
        valid = np.isfinite(flat) & (flat >= minimum) & (flat <= maximum)
        residual = np.full(flat.size, np.inf)
        with np.errstate(invalid="ignore"):
            points = rays.reshape(-1, 3)[valid] * flat[valid, None]
            residual[valid] = np.abs(np.einsum("ni,i->n", points, normal) + offset)
        expected = (residual.reshape(depth.shape) < threshold).astype(np.uint8) * 255
        actual = geometry_kernel("cpp").dense_plane_mask(
            depth, rays, normal, offset, minimum, maximum, threshold
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.uint8)
        return actual

    def test_all_hues_and_saturation_value_boundaries(self):
        hue, saturation, value = np.meshgrid(
            np.arange(256, dtype=np.uint8), [0, 69, 70, 71, 255], [0, 49, 50, 51, 255], indexing="ij"
        )
        hsv = np.stack((hue, saturation, value), axis=-1).astype(np.uint8)
        self.color_parity(hsv, color_min_saturation=70, color_min_value=50)

    def test_red_blue_overlap_stays_unknown(self):
        hsv = np.column_stack((np.arange(180), np.full((180, 2), 255))).astype(np.uint8)
        labels = self.color_parity(hsv, red_hue_ranges=((0, 25), (110, 179)), blue_hue_range=(20, 120))
        self.assertTrue((labels[20:26] == 0).all())
        self.assertTrue((labels[110:121] == 0).all())
        self.assertEqual(labels[19], 1)
        self.assertEqual(labels[26], -1)

    def test_arbitrary_rank_empty_and_strided_hsv_arrays(self):
        rng = np.random.default_rng(463)
        for shape in ((3,), (0, 3), (2, 0, 3), (13, 3), (4, 7, 3), (2, 4, 7, 3)):
            with self.subTest(shape=shape):
                self.color_parity(rng.integers(0, 256, shape, dtype=np.uint8))
        pixels = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
        self.color_parity(pixels[::-2, ::3])
        self.color_parity(pixels.transpose(1, 0, 2))

    def test_color_thresholds_include_endpoints(self):
        hsv = np.array([[0, 0, 0], [12, 255, 255], [95, 255, 255], [140, 255, 255]], np.uint8)
        np.testing.assert_array_equal(
            self.color_parity(hsv, color_min_saturation=0, color_min_value=0), [1, 1, -1, -1]
        )
        np.testing.assert_array_equal(
            self.color_parity(hsv, color_min_saturation=255, color_min_value=255), [0, 1, -1, -1]
        )

    def test_non_uint8_color_input_keeps_reference_semantics(self):
        hsv = np.array([[12.5, 200, 200], [95.25, 200, 200]], np.float64)
        np.testing.assert_array_equal(self.color_parity(hsv), [0, -1])

    def test_depth_invalid_samples_do_not_generate_edges(self):
        depth = np.array([[1.0, 1.3, np.nan, 1.0, np.inf], [1.1, 0.0, -1.0, -np.inf, 0.099]], np.float32)
        expected = self.depth_parity(depth)
        self.assertEqual(expected[0, 1], 255)
        self.assertEqual(expected[1, 0], 255)
        self.assertEqual(np.count_nonzero(expected), 2)

    def test_depth_range_and_jump_threshold_boundaries(self):
        minimum, maximum, threshold = 0.125, 4.0, 0.03125
        below_min = np.nextafter(np.float32(minimum), np.float32(0))
        above_max = np.nextafter(np.float32(maximum), np.float32(np.inf))
        above_jump = np.nextafter(np.float32(minimum + threshold), np.float32(np.inf))
        depth = np.array(
            [[below_min, minimum, minimum + threshold, above_jump, maximum, above_max]], np.float32
        )
        actual = self.depth_parity(depth, minimum, maximum, threshold)
        self.assertEqual(actual[0, 1], 0)
        self.assertEqual(actual[0, 2], 0)
        self.assertEqual(actual[0, 4], 255)
        self.assertEqual(actual[0, 5], 0)
        self.assertEqual(self.depth_parity(depth[:, [1, 3]], minimum, maximum, threshold)[0, 1], 255)

    def test_dense_random_strided_and_degenerate_depth(self):
        rng = np.random.default_rng(467)
        depth = rng.uniform(0.05, 4.1, (127, 193)).astype(np.float32)
        depth[::11, ::7] = np.nan
        depth[::9, ::5] = 0
        for image in (depth, depth[::-2, ::3], depth[:1], depth[:, :1], depth[:0], depth[:, :0]):
            with self.subTest(shape=image.shape, strides=image.strides):
                self.depth_parity(image)

    def test_depth_edges_assign_right_and_lower_pixels(self):
        depth = np.array([[1, 1, 2], [1, 2, 2]], np.float32)
        np.testing.assert_array_equal(self.depth_parity(depth), [[0, 0, 255], [0, 255, 0]])

    def test_combined_edges_match_with_and_without_roi(self):
        rng = np.random.default_rng(469)
        camera = Camera(64, 48, [[55, 0, 32], [0, 55, 24], [0, 0, 1]])
        rgb = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        depth = rng.uniform(0.3, 1.4, (48, 64)).astype(np.float32)
        depth[::5, ::3] = np.nan
        frame = Frame(rgb, depth, camera, 1)
        for roi in (None, (7, 8, 51, 33)):
            with self.subTest(roi=roi):
                cfg = Config(geometry_backend="numpy", roi_xyxy=roi).validate()
                expected = rgb_depth_edges(frame, cfg)
                actual = rgb_depth_edges(frame, replace(cfg, geometry_backend="cpp"))
                for observed, reference in zip(actual, expected):
                    np.testing.assert_array_equal(observed, reference)
                if roi:
                    self.assertFalse(actual[1][:8].any())
                    self.assertFalse(actual[1][:, 51:].any())

    def test_both_edge_backends_reject_roi_outside_camera(self):
        camera = Camera(32, 24, [[25, 0, 16], [0, 25, 12], [0, 0, 1]])
        frame = Frame(np.zeros((24, 32, 3), np.uint8), np.ones((24, 32), np.float32), camera, 1)
        for backend in ("numpy", "cpp"):
            cfg = Config(geometry_backend=backend, roi_xyxy=(0, 0, 33, 24)).validate()
            with self.assertRaisesRegex(ValueError, "ROI exceeds"):
                rgb_depth_edges(frame, cfg)

    def test_native_image_entrypoints_reject_wrong_shapes(self):
        kernel = geometry_kernel("cpp")
        for hsv in (np.ones((2, 4), np.uint8), np.array(0, np.uint8)):
            with self.assertRaises(ValueError):
                kernel.classify_hsv(hsv, 70, 50, ((0, 12),), (95, 140))
        with self.assertRaises(ValueError):
            kernel.depth_edges(np.ones(3, np.float32), 0.1, 4.0, 0.02)

    def test_dense_plane_random_rays_invalid_depth_and_invalid_rays(self):
        rng = np.random.default_rng(473)
        rays = rng.uniform(-1.5, 1.5, (97, 131, 3)).astype(np.float32)
        rays[:, :, 2] = 1
        depth = rng.uniform(0.05, 4.1, (97, 131)).astype(np.float32)
        for index, value in enumerate((np.nan, np.inf, -np.inf, 0, -1)):
            depth[index::17, index::13] = value
        rays[7, 8, 0] = np.nan
        rays[9, 8, 1] = np.inf
        for normal in (np.array([0.0, 0.0, -1.0]), np.array([0.2, -0.78, 0.59])):
            with self.subTest(normal=normal):
                self.dense_plane_parity(depth, rays, normal, -1.0, threshold=0.5)

    def test_dense_plane_strided_empty_single_row_and_column(self):
        rng = np.random.default_rng(479)
        rays = rng.uniform(-1, 1, (37, 53, 3)).astype(np.float32)
        depth = rng.uniform(0.2, 3, (37, 53)).astype(np.float32)
        normal = np.array([0.37, -0.52, 0.77])
        for ys, xs in (
            (slice(None), slice(None)),
            (slice(None, None, -2), slice(None, None, 3)),
            (slice(0, 1), slice(None)),
            (slice(None), slice(0, 1)),
            (slice(0, 0), slice(None)),
            (slice(None), slice(0, 0)),
        ):
            with self.subTest(ys=ys, xs=xs):
                self.dense_plane_parity(depth[ys, xs], rays[ys, xs], normal, -0.75, threshold=0.2)

    def test_dense_plane_uses_float32_xyz_before_double_normal(self):
        rng = np.random.default_rng(487)
        rays = rng.uniform(-1, 1, (11, 19, 3)).astype(np.float32)
        depth = rng.uniform(0.2, 3, (11, 19)).astype(np.float32)
        normal = np.array([0.2, -0.78, 0.59])
        offset = -0.73
        points = rays.reshape(-1, 3) * depth.ravel()[:, None]
        residual = np.abs(np.einsum("ni,i->n", points, normal) + offset)
        # Exact thresholds catch changes to float32 multiplication or einsum's sum order.
        for index in (0, 1, 7, 13, 29, 57, 99, 208):
            threshold = residual[index]
            for boundary in (np.nextafter(threshold, 0), threshold, np.nextafter(threshold, np.inf)):
                with self.subTest(index=index, threshold=boundary):
                    self.dense_plane_parity(depth, rays, normal, offset, threshold=boundary)

    def test_dense_plane_strict_threshold_and_float32_depth_range(self):
        below_min = np.nextafter(np.float32(0.125), np.float32(0))
        above_max = np.nextafter(np.float32(4.0), np.float32(np.inf))
        depth = np.array([[below_min, 0.125, 1, 1.03125, 4, above_max]], np.float32)
        rays = np.zeros((*depth.shape, 3), np.float32)
        rays[:, :, 2] = 1
        normal = np.array([0.0, 0.0, 1.0])
        actual = self.dense_plane_parity(depth, rays, normal, -1, 0.125, 4, 0.03125)
        np.testing.assert_array_equal(actual, [[0, 0, 255, 0, 0, 0]])
        actual = self.dense_plane_parity(depth, rays, normal, 0, 0.125, 4, 5)
        np.testing.assert_array_equal(actual, [[0, 255, 255, 255, 255, 0]])

    def test_dense_plane_with_distorted_camera_and_real_metric_plane(self):
        camera = Camera(96, 72, [[83, 0, 48], [0, 84, 36], [0, 0, 1]], [0.08, -0.02, 0.001, 0, 0])
        yy, xx = np.indices((72, 96))
        rays = camera.rays(np.column_stack((xx.ravel(), yy.ravel()))).reshape(72, 96, 3)
        normal = np.array([0.23, -0.41, -0.88])
        normal /= np.linalg.norm(normal)
        offset = 1.1
        depth = (-offset / (rays @ normal)).astype(np.float32)
        depth[17:31, 22:44] += 0.03
        depth[::7, ::9] = np.nan
        observed = self.dense_plane_parity(depth, rays.astype(np.float32), normal, offset)
        self.assertEqual(np.count_nonzero(observed[17:31, 22:44]), 0)
        self.assertGreater(np.count_nonzero(observed), 6000)

    def test_detector_dense_masks_match_for_recorded_style_rendered_faces(self):
        renderer = Renderer(default_camera())
        for index in (0, 2, 6, 8):
            with self.subTest(scene=index):
                frame, _, _ = renderer.render(demo_scenes()[index][1], index, 0.001, 0.005)
                cfg = Config(up_camera=[0, -1, 0], geometry_backend="numpy").validate()
                expected = Detector(cfg)._depth_masks(frame)
                actual = Detector(replace(cfg, geometry_backend="cpp"))._depth_masks(frame)
                self.assertEqual(len(actual), len(expected))
                for (mask, plane), (reference, reference_plane) in zip(actual, expected):
                    np.testing.assert_array_equal(mask, reference)
                    np.testing.assert_allclose(plane.normal, reference_plane.normal, atol=2e-12)

    def test_native_dense_plane_rejects_inconsistent_shapes(self):
        kernel = geometry_kernel("cpp")
        for depth, rays, normal in (
            (np.ones(3, np.float32), np.ones((1, 3, 3), np.float32), np.ones(3)),
            (np.ones((2, 3), np.float32), np.ones((2, 4, 3), np.float32), np.ones(3)),
            (np.ones((2, 3), np.float32), np.ones((2, 3, 2), np.float32), np.ones(3)),
            (np.ones((2, 3), np.float32), np.ones((2, 3, 3), np.float32), np.ones((1, 3))),
        ):
            with self.subTest(depth=depth.shape, rays=rays.shape, normal=normal.shape):
                with self.assertRaises(ValueError):
                    kernel.dense_plane_mask(depth, rays, normal, 0.0, 0.1, 4.0, 0.008)


if __name__ == "__main__":
    unittest.main()
