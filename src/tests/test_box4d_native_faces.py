"""Face stages retain measured pixels, robust lines and original rejection gates."""

import unittest

import cv2
import numpy as np

from box4d.config import Config
from box4d.detector import _refine_quad, sample_face
from box4d.geometry import Camera, order_quad
from box4d.io import Frame
from box4d.native import geometry_kernel
from box4d.roi_faces import fit_boundary_lines
from box4d.synthetic import Renderer, default_camera, world_box


@unittest.skipUnless(geometry_kernel() is not None, "build CPU kernels to test native face stages")
class NativeFaceTests(unittest.TestCase):
    def config(self, backend="numpy", **values):
        return Config(geometry_backend=backend, **values).validate()

    def frame(self, depth):
        height, width = depth.shape
        camera = Camera(
            width,
            height,
            [[240, 0, width / 2], [0, 240, height / 2], [0, 0, 1]],
            [0.03, -0.01, 0.001, 0, 0],
        )
        return Frame(np.zeros((height, width, 3), np.uint8), depth, camera, 1)

    def assert_sample_parity(self, frame, quad, **config):
        observations = []
        for backend in ("numpy", "cpp"):
            diagnostics = {}
            try:
                xyz, fraction = sample_face(frame, quad, self.config(backend, **config), diagnostics)
                outcome = (xyz, fraction)
            except ValueError as error:
                outcome = str(error)
            observations.append((outcome, diagnostics))
        (reference, ref_info), (actual, info) = observations
        if isinstance(reference, str):
            self.assertEqual(actual, reference)
        else:
            self.assertEqual(reference[1], actual[1])
            np.testing.assert_array_equal(actual[0], reference[0])
        self.assertEqual(set(ref_info), set(info))
        for key, value in ref_info.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(info[key], value)
            else:
                self.assertEqual(info[key], value)
        return observations

    def test_polygon_rounding_missing_depth_and_distorted_rays_match(self):
        rng = np.random.default_rng(671)
        depth = rng.uniform(0.15, 4.0, (120, 180)).astype(np.float32)
        depth[::7, ::3] = np.nan
        depth[::11, ::5] = np.inf
        depth[::3, ::7] = 0
        frame = self.frame(depth)
        for offset in (0.0, 0.25, 0.5, 0.75):
            quad = np.array([[20.5, 20.5], [150.5, 30.5], [130.5, 100.5], [30.5, 90.5]]) + offset
            self.assert_sample_parity(frame, quad, max_depth_samples=61)

    def test_exact_range_boundary_and_uniform_sampling_endpoints_match(self):
        depth = np.ones((80, 120), np.float32)
        depth[:, :40] = 0.15
        depth[:, 80:] = 2.1
        frame = self.frame(depth)
        quad = np.array([[10.0, 10.0], [110.0, 10.0], [110.0, 70.0], [10.0, 70.0]])
        observations = self.assert_sample_parity(
            frame, quad, min_depth_m=0.15, max_depth_m=2.1, max_depth_samples=24
        )
        info = observations[1][1]
        self.assertEqual(info["depth_fraction"], 1.0)
        self.assertGreater(info["valid_depth_pixels"], len(info["sample_uv"]))

    def test_insufficient_depth_reports_absolute_interior_denominator(self):
        depth = np.zeros((80, 120), np.float32)
        depth[25:35, 45:55] = 1.2
        frame = self.frame(depth)
        quad = np.array([[10.0, 10.0], [110.0, 10.0], [110.0, 70.0], [10.0, 70.0]])
        observations = self.assert_sample_parity(frame, quad)
        self.assertEqual(observations[1][0], "insufficient interior depth coverage")
        self.assertEqual(observations[1][1]["valid_depth_pixels"], 100)
        self.assertGreater(observations[1][1]["interior_pixels"], 1000)

    def test_image_and_roi_clipping_gates_are_preserved(self):
        frame = self.frame(np.ones((80, 120), np.float32))
        for quad, extra in (
            (np.array([[0.0, 10.0], [80.0, 10.0], [80.0, 60.0], [0.0, 60.0]]), {}),
            (
                np.array([[20.0, 20.0], [80.0, 20.0], [80.0, 60.0], [20.0, 60.0]]),
                {"roi_xyxy": [20, 10, 100, 70]},
            ),
            (np.array([[20.0, 20.0], [21.0, 20.0], [21.0, 21.0], [20.0, 21.0]]), {}),
        ):
            self.assert_sample_parity(frame, quad, **extra)

    def test_refined_observed_edges_with_clutter_match(self):
        rng = np.random.default_rng(902)
        reference = np.array([[25.0, 25.0], [145.0, 36.0], [131.0, 115.0], [34.0, 104.0]])
        edges = np.zeros((140, 180), np.uint8)
        cv2.polylines(edges, [reference.astype(np.int32)], True, 255, 1)
        yy, xx = rng.integers([0, 0], [140, 180], (300, 2)).T
        edges[yy, xx] = 255
        for _ in range(25):
            initial = reference + rng.uniform(-1.7, 1.7, (4, 2))
            expected = _refine_quad(initial, edges, backend="numpy")
            actual = _refine_quad(initial, edges, backend="cpp")
            np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=0)

    def test_refinement_without_enough_evidence_keeps_original(self):
        quad = np.array([[20.0, 20.0], [100.0, 20.0], [100.0, 70.0], [20.0, 70.0]])
        for pixels in (0, 10):
            edges = np.zeros((90, 120), np.uint8)
            edges[20, 20 : 20 + pixels] = 255
            np.testing.assert_array_equal(_refine_quad(quad, edges, backend="cpp"), quad)
            np.testing.assert_array_equal(_refine_quad(quad, edges, backend="numpy"), quad)

    def assert_boundary_parity(self, quad, edges, **config):
        expected = fit_boundary_lines(quad, edges, self.config("numpy", **config))
        actual = fit_boundary_lines(quad, edges, self.config("cpp", **config))
        self.assertEqual(expected is None, actual is None)
        if expected is None:
            return
        np.testing.assert_allclose(actual[0], expected[0], atol=2e-10, rtol=0)
        for actual_line, expected_line in zip(actual[1], expected[1]):
            self.assertEqual(actual_line["support_pixels"], expected_line["support_pixels"])
            np.testing.assert_allclose(actual_line["segment"], expected_line["segment"], atol=2e-10, rtol=0)
            self.assertAlmostEqual(actual_line["span_fraction"], expected_line["span_fraction"], places=12)
            self.assertAlmostEqual(actual_line["rms_px"], expected_line["rms_px"], places=12)

    def test_robust_boundary_lines_and_reports_match_with_clutter(self):
        rng = np.random.default_rng(48)
        quad = np.array([[25.0, 25.0], [145.0, 36.0], [131.0, 115.0], [34.0, 104.0]])
        edges = np.zeros((140, 180), np.uint8)
        cv2.polylines(edges, [quad.astype(np.int32)], True, 255, 1)
        yy, xx = rng.integers([0, 0], [140, 180], (450, 2)).T
        edges[yy, xx] = 255
        for _ in range(25):
            self.assert_boundary_parity(quad + rng.uniform(-1.2, 1.2, (4, 2)), edges)

    def test_boundary_requires_observed_lines_and_original_span(self):
        quad = np.array([[20.0, 20.0], [100.0, 20.0], [100.0, 90.0], [20.0, 90.0]])
        for last_side in (False, True):
            edges = np.zeros((120, 140), np.uint8)
            cv2.polylines(edges, [quad.astype(np.int32)], last_side, 255, 1)
            self.assert_boundary_parity(quad, edges)
            self.assert_boundary_parity(quad, edges, yolo_line_min_span_fraction=0.99)
        self.assertIsNone(fit_boundary_lines(quad, np.zeros_like(edges), self.config("cpp")))

    def test_rendered_faces_retain_depth_and_rgb_line_evidence(self):
        renderer = Renderer(default_camera())
        for yaw in (0.0, 25.0, 65.0):
            frame, _, ids = renderer.render([world_box((0, 0, 0.1), 0.2, yaw)], depth_noise_m=0.001)
            mask = (ids == 0).astype(np.uint8)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            rectangle = cv2.boxPoints(cv2.minAreaRect(max(contours, key=cv2.contourArea)))
            quad = order_quad(rectangle, backend="numpy")
            self.assert_sample_parity(frame, quad, min_depth_fraction=0.1)
            edges = cv2.Canny(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY), 50, 100)
            self.assert_boundary_parity(quad, edges)

    def test_native_face_array_contracts_reject_wrong_shapes(self):
        native = geometry_kernel("cpp")
        with self.assertRaises(ValueError):
            native.refine_quad(np.zeros((3, 2)), np.zeros((20, 20), np.uint8))
        with self.assertRaises(ValueError):
            native.sample_face_pixels(np.zeros(20, np.float32), np.zeros((4, 2)), 0.15, 4.0, 50)
        with self.assertRaises(ValueError):
            native.fit_boundary_lines(np.zeros((4, 2)), np.zeros(20, np.uint8), 4.0, 8.0, 8, 0.3)


if __name__ == "__main__":
    unittest.main()
