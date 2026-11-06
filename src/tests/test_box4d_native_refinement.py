"""One native optimizer evaluation must retain measured image/depth derivatives."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from box4d.config import Config
from box4d.detector import Face
from box4d.geometry import cube_vertices, order_quad, transform_points, upright_from_camera
from box4d.native import geometry_kernel
from box4d.pose import estimate_face, refine_pose
from box4d.synthetic import Renderer, default_camera, world_box


@unittest.skipUnless(geometry_kernel() is not None, "build CPU kernels to test native refinement")
class NativeRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config(up_camera=[0, -1, 0], geometry_backend="numpy").validate()
        cls.renderer = Renderer(default_camera())
        cls.frame, _, _ = cls.renderer.render(
            [world_box((0.03, 0.02, 0.1), 0.2, 33)], sequence=991, depth_noise_m=0.0005
        )
        box = world_box((0.03, 0.02, 0.1), 0.2, 33)
        vertices = cube_vertices(0.2)
        points = transform_points(box["pose"], vertices)
        cls.observations = []
        for axis in range(3):
            for sign in (-1, 1):
                ids = np.flatnonzero(np.isclose(vertices[:, axis], sign * 0.1))
                center = points[ids].mean(axis=0)
                normal = box["pose"][:3, axis] * sign
                if normal @ center >= -1e-8:
                    continue
                quad = order_quad(cls.frame.camera.project(points[ids]), backend="numpy")
                estimates, _, _ = estimate_face(cls.frame, Face(quad, "analytic", 1.0), cls.cfg)
                cls.observations.append(min(estimates, key=lambda e: e.cost))
        if len(cls.observations) < 2:
            raise AssertionError("rendered regression requires multiple measured visible faces")
        cls.anchor = min(cls.observations, key=lambda e: e.cost)

    def problem(self, backend, frame=None, observations=None):
        captured = {}

        def capture(fun, initial, **kwargs):
            captured.update(fun=fun, jac=kwargs["jac"], initial=initial.copy())
            return SimpleNamespace(x=initial, success=True, nfev=0)

        with patch("box4d.pose.least_squares", side_effect=capture):
            refine_pose(
                self.frame if frame is None else frame,
                self.anchor,
                self.observations if observations is None else observations,
                replace(self.cfg, geometry_backend=backend),
            )
        return captured

    def test_multiple_measured_faces_residual_and_jacobian_match_python(self):
        reference, native = self.problem("numpy"), self.problem("cpp")
        np.testing.assert_allclose(reference["initial"], native["initial"], atol=1e-14)
        rng = np.random.default_rng(382)
        for offset in np.r_[np.zeros((1, 4)), rng.normal(size=(12, 4)) * [0.01, 0.01, 0.01, 0.02]]:
            parameters = reference["initial"] + offset
            np.testing.assert_allclose(native["fun"](parameters), reference["fun"](parameters), atol=2e-10)
            np.testing.assert_allclose(native["jac"](parameters), reference["jac"](parameters), atol=2e-9)

    def test_distorted_camera_retains_projection_derivatives(self):
        coefficients = np.array(
            [
                0.08,
                -0.04,
                0.001,
                -0.002,
                0.006,
                0.002,
                -0.001,
                0.0005,
                0.0002,
                -0.0001,
                0.00015,
                -0.00008,
                0.002,
                -0.003,
            ]
        )
        for count in (4, 5, 8, 12, 14):
            with self.subTest(coefficient_count=count):
                camera = replace(self.frame.camera, d=coefficients[:count])
                frame = replace(self.frame, camera=camera)
                reference, native = self.problem("numpy", frame), self.problem("cpp", frame)
                x = reference["initial"] + [0.01, -0.005, 0.004, 0.035]
                np.testing.assert_allclose(native["fun"](x), reference["fun"](x), atol=2e-10)
                np.testing.assert_allclose(native["jac"](x), reference["jac"](x), atol=2e-9)

    def test_native_analytic_jacobian_matches_finite_differences(self):
        native = self.problem("cpp")
        x = native["initial"] + [0.005, -0.006, 0.01, 0.03]
        numeric = np.column_stack(
            [
                (native["fun"](x + np.eye(4)[i] * 1e-6) - native["fun"](x - np.eye(4)[i] * 1e-6)) / 2e-6
                for i in range(4)
            ]
        )
        np.testing.assert_allclose(native["jac"](x), numeric, atol=2e-5, rtol=2e-6)

    def test_behind_camera_returns_original_large_residual_and_zero_jacobian(self):
        reference, native = self.problem("numpy"), self.problem("cpp")
        x = reference["initial"].copy()
        x[:3] = upright_from_camera(self.cfg.up_camera) @ [0.0, 0.0, -1.0]
        for key in ("fun", "jac"):
            np.testing.assert_array_equal(native[key](x), reference[key](x))
        self.assertTrue(np.all(native["fun"](x) == 1e6))
        self.assertTrue(np.all(native["jac"](x) == 0))

    def test_single_and_multi_face_optimized_pose_and_acceptance_match(self):
        for observations in ([self.anchor], self.observations):
            with self.subTest(face_count=len(observations)):
                reference_pose, reference_metrics = refine_pose(
                    self.frame, self.anchor, observations, self.cfg
                )
                native_pose, native_metrics = refine_pose(
                    self.frame, self.anchor, observations, replace(self.cfg, geometry_backend="cpp")
                )
                np.testing.assert_allclose(native_pose, reference_pose, atol=2e-10)
                self.assertEqual(native_metrics.keys(), reference_metrics.keys())
                self.assertEqual(native_metrics["refinement"], "upright_4d")
                for key in ("nfev", "converged", "multi_face_pnp"):
                    self.assertEqual(native_metrics[key], reference_metrics[key])
                for key in ("refined_reprojection_rms_px", "refined_depth_median_residual_m"):
                    self.assertAlmostEqual(native_metrics[key], reference_metrics[key], places=10)

    def test_disabled_and_inferred_only_paths_do_not_need_native_evaluator(self):
        with patch("box4d.native.geometry_kernel", side_effect=AssertionError("kernel should not run")):
            _, disabled = refine_pose(
                self.frame, self.anchor, self.observations, replace(self.cfg, refine_4d=False)
            )
            inferred = replace(self.anchor, face=replace(self.anchor.face, source="bottom_occlusion_model"))
            _, no_evidence = refine_pose(self.frame, inferred, [inferred], self.cfg)
        self.assertEqual(disabled["refinement"], "disabled_or_no_vertical_reference")
        self.assertEqual(no_evidence["refinement"], "no_consistent_correspondences")


if __name__ == "__main__":
    unittest.main()
