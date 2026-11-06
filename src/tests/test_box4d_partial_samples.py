"""Partial outlines remain usable with sparse holes, without inflating support."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from box4d.config import Config
from box4d.edges import rgb_depth_edges
from box4d.geometry import Plane, fit_plane
from box4d.native import geometry_kernel
from box4d.partial import upright_patch_faces, visibility
from box4d.pose import estimate_partial_face
from box4d.synthetic import Renderer, default_camera, world_box


class PartialSampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = Renderer(default_camera())

    def scene(self, backend="numpy"):
        cfg = Config(up_camera=[0, -1, 0], geometry_backend=backend).validate()
        box = world_box((0, 0, 0.1), 0.2, 0, (0, -1.2, 0.1))
        occluder = dict(
            quad_camera=[[-0.3, 0.04, 0.85], [0.3, 0.04, 0.85], [0.3, 0.4, 0.85], [-0.3, 0.4, 0.85]]
        )
        frame, truth, instance = self.renderer.render([box], floor_height_camera_m=0.1, occluders=[occluder])
        mask = (instance == 0).astype(np.uint8) * 255
        _, _, edges = rgb_depth_edges(frame, cfg)
        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        hint = Plane(np.array([0.0, 0.0, -1.0]), 1.1, 0.0, 1.0)
        # Reproduce the proposal's bounded, eroded sampling locations, so the
        # invalid pixel is guaranteed to reach the actual consumer.
        bx, by, bw, bh = cv2.boundingRect(mask)
        yy, xx = np.nonzero(cv2.erode(mask[by : by + bh, bx : bx + bw], np.ones((5, 5), np.uint8)))
        ids = np.linspace(0, len(xx) - 1, min(len(xx), cfg.max_depth_samples)).astype(int)
        uv = np.column_stack((xx[ids] + bx, yy[ids] + by))
        return frame, truth, mask, hint, cfg, distance, uv

    def backends(self):
        return ("numpy", "cpp") if geometry_kernel() is not None else ("numpy",)

    def test_single_nan_zero_and_infinity_preserve_real_partial_pose(self):
        for backend in self.backends():
            for invalid in (np.nan, 0.0, np.inf):
                with self.subTest(backend=backend, invalid=invalid):
                    frame, truth, mask, hint, cfg, distance, uv = self.scene(backend)
                    baseline = upright_patch_faces(frame, mask, hint, cfg, distance)
                    self.assertEqual(len(baseline), 1)
                    x, y = uv[0]
                    frame.depth[y, x] = invalid
                    with patch("box4d.partial.fit_plane", wraps=fit_plane) as fitted:
                        faces = upright_patch_faces(frame, mask, hint, cfg, distance)
                    self.assertEqual(len(faces), 1)
                    samples = fitted.call_args.args[0]
                    self.assertTrue(np.isfinite(samples).all())
                    self.assertTrue(np.all(samples[:, 2] > 0))
                    self.assertEqual(len(samples), len(uv) - 1)
                    plane = faces[0].plane_hint
                    self.assertTrue(np.isfinite(np.r_[plane.normal, plane.d, plane.rmse_m]).all())
                    self.assertAlmostEqual(plane.support, (len(uv) - 1) / len(uv))
                    np.testing.assert_allclose(faces[0].quad, baseline[0].quad, atol=1e-9)
                    estimates, corners, measured = estimate_partial_face(frame, faces[0], cfg)
                    self.assertTrue(np.isfinite(corners).all())
                    self.assertTrue(np.isfinite(np.r_[measured.normal, measured.d]).all())
                    np.testing.assert_allclose(
                        estimates[0].box_pose[:3, 3], truth[0]["center_camera_m"], atol=0.005
                    )

    def test_invalid_samples_keep_original_support_denominator(self):
        for backend in self.backends():
            for invalid in (np.nan, 0.0, np.inf):
                with self.subTest(backend=backend, invalid=invalid):
                    frame, _, mask, hint, cfg, distance, uv = self.scene(backend)
                    invalid_count = int(len(uv) * (1 - cfg.partial_min_plane_support_fraction)) + 1
                    # Spread holes over the patch; leave real top/side edge and
                    # foreground evidence intact on the rendered scene.
                    ids = np.random.default_rng(21).permutation(len(uv))[:invalid_count]
                    frame.depth[uv[ids, 1], uv[ids, 0]] = invalid
                    self.assertEqual(upright_patch_faces(frame, mask, hint, cfg, distance), [])
                    # The otherwise identical observation passes a lower support
                    # gate, proving rejection comes from the original denominator.
                    relaxed = replace(cfg, partial_min_plane_support_fraction=0.70)
                    faces = upright_patch_faces(frame, mask, hint, relaxed, distance)
                    self.assertEqual(len(faces), 1)
                    self.assertAlmostEqual(faces[0].plane_hint.support, 1 - invalid_count / len(uv))
                    self.assertLess(faces[0].plane_hint.support, cfg.partial_min_plane_support_fraction)

    def test_fewer_than_minimum_valid_points_reject_before_fitting(self):
        frame, _, mask, hint, cfg, distance, uv = self.scene()
        frame.depth[uv[cfg.min_depth_samples - 1 :, 1], uv[cfg.min_depth_samples - 1 :, 0]] = np.nan
        with patch("box4d.partial.fit_plane", wraps=fit_plane) as fitted:
            self.assertEqual(upright_patch_faces(frame, mask, hint, cfg, distance), [])
        fitted.assert_not_called()

    def test_nonfinite_xyz_is_filtered_before_the_constrained_fit(self):
        frame, _, mask, hint, cfg, distance, uv = self.scene()
        original_rays = frame.camera.rays

        def rays(points):
            result = original_rays(points)
            if np.array_equal(points, uv):
                result[0, 0] = np.nan
            return result

        with patch.object(frame.camera, "rays", side_effect=rays), patch(
            "box4d.partial.fit_plane", wraps=fit_plane
        ) as fitted:
            faces = upright_patch_faces(frame, mask, hint, cfg, distance)
        self.assertEqual(len(faces), 1)
        self.assertTrue(np.isfinite(fitted.call_args.args[0]).all())
        self.assertAlmostEqual(faces[0].plane_hint.support, (len(uv) - 1) / len(uv))

    def test_visibility_counts_holes_against_entire_grid(self):
        frame, _, mask, hint, cfg, distance, _ = self.scene()
        face = upright_patch_faces(frame, mask, hint, cfg, distance)[0]
        baseline = visibility(frame, face.quad, face.plane_hint, cfg)
        self.assertGreater(baseline["explained_depth_fraction"], 0.85)
        # Keep only one in ten pixels. The full grid remains the denominator.
        yy, xx = np.indices(frame.depth.shape)
        frame.depth[(xx + yy) % 10 != 0] = np.nan
        evidence = visibility(frame, face.quad, face.plane_hint, cfg)
        self.assertLess(evidence["valid_depth_fraction"], 0.20)
        self.assertLess(evidence["explained_depth_fraction"], 0.20)
        self.assertLessEqual(evidence["explained_depth_fraction"], evidence["valid_depth_fraction"])


if __name__ == "__main__":
    unittest.main()
