"""Joint fallback weights and robust thresholds must not depend on sample count."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from box4d.config import Config
from box4d.geometry import Camera, Plane, square_points
from box4d.joint_tracking import _JointObjective, solve_joint
from box4d.synthetic import default_camera


def make_case(noise=False):
    camera = default_camera()
    initial = np.r_[.01, -.02, .005, .005, -.01, 1.1]
    truth = np.r_[.08, -.11, .04, .025, -.02, 1.14]
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(initial[:3])[0]
    pose[:3, 3] = initial[3:]
    quad = square_points(.2) + [0, 0, -.1]
    state = dict(pose=pose, quad_model=quad, normal_model=np.array([0., 0., -1.]))
    xx, yy = np.meshgrid(np.linspace(-.09, .09, 4), np.linspace(-.09, .09, 3))
    model = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, -.1)))
    xx, yy = np.meshgrid(np.linspace(-.09, .09, 15), np.linspace(-.09, .09, 11))
    depth_model = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, -.1)))
    rotation = cv2.Rodrigues(truth[:3])[0]
    normal = rotation @ state["normal_model"]
    depth = depth_model @ rotation.T + truth[3:]
    uv = camera.project(model @ rotation.T + truth[3:])
    if noise:
        rng = np.random.default_rng(725)
        uv += rng.normal(0, .15, uv.shape)
        depth += rng.normal(0, .0007, (len(depth), 1)) * normal
    center = rotation @ quad.mean(axis=0) + truth[3:]
    plane = Plane(normal, -float(normal @ center), .0007 if noise else 0., 1.)
    return SimpleNamespace(camera=camera), state, model, uv, depth, plane, initial, truth


class JointObjectiveTests(unittest.TestCase):
    def test_analytic_jacobian_matches_finite_difference(self):
        for distortion in ((), (.09, -.03, .001, -.002, .004)):
            with self.subTest(distortion=distortion):
                frame, state, model, uv, depth, plane, initial, _ = make_case(noise=True)
                camera = Camera(640, 480, frame.camera.k, distortion)
                objective = _JointObjective(camera, state, model, uv, depth, plane)
                epsilon = 1e-7
                numeric = np.column_stack([
                    (objective.residual(initial + np.eye(6)[axis] * epsilon)
                     - objective.residual(initial - np.eye(6)[axis] * epsilon)) / (2 * epsilon)
                    for axis in range(6)
                ])
                np.testing.assert_allclose(objective.jacobian(initial), numeric, atol=2e-6, rtol=2e-6)

    def test_repeating_either_observation_group_preserves_cost_and_gradient(self):
        frame, state, model, uv, depth, plane, initial, _ = make_case(noise=True)
        reference = _JointObjective(frame.camera, state, model, uv, depth, plane)

        def cost_gradient(objective):
            residual = objective.residual(initial)
            weighted_derivative = objective.loss(residual**2)[1]
            return np.asarray(objective.costs(initial)), objective.jacobian(initial).T @ (weighted_derivative * residual)

        costs, gradient = cost_gradient(reference)
        for image_repeats, depth_repeats in ((3, 1), (1, 5), (3, 5)):
            with self.subTest(image=image_repeats, depth=depth_repeats):
                repeated = _JointObjective(
                    frame.camera, state, np.tile(model, (image_repeats, 1)),
                    np.tile(uv, (image_repeats, 1)), np.tile(depth, (depth_repeats, 1)), plane,
                )
                repeated_costs, repeated_gradient = cost_gradient(repeated)
                np.testing.assert_allclose(repeated_costs, costs, rtol=1e-13, atol=1e-12)
                np.testing.assert_allclose(repeated_gradient, gradient, rtol=1e-12, atol=1e-10)

    def test_custom_loss_derivatives_are_weighted_consistently(self):
        frame, state, model, uv, depth, plane, _, _ = make_case()
        objective = _JointObjective(frame.camera, state, model, uv, depth, plane)
        z = np.linspace(.2, 20., len(objective.weights))
        epsilon = 1e-5
        result = objective.loss(z)
        numeric_first = (objective.loss(z + epsilon)[0] - objective.loss(z - epsilon)[0]) / (2 * epsilon)
        numeric_second = (objective.loss(z + epsilon)[1] - objective.loss(z - epsilon)[1]) / (2 * epsilon)
        np.testing.assert_allclose(result[1], numeric_first, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(result[2], numeric_second, rtol=1e-8, atol=1e-10)

    def test_group_normalization_does_not_move_physical_robust_threshold(self):
        frame, state, model, uv, depth, plane, _, _ = make_case()
        reference = _JointObjective(frame.camera, state, model, uv, depth, plane)
        repeated = _JointObjective(frame.camera, state, np.tile(model, (5, 1)), np.tile(uv, (5, 1)), np.tile(depth, (3, 1)), plane)
        # rho' / group weight is the same robust influence for a 1 sigma or
        # 5 sigma observation, whether there are 12 or 60 image points.
        for standardized_error in (1., 5.):
            influences = []
            for objective in (reference, repeated):
                derivative = objective.loss(np.full(len(objective.weights), standardized_error**2))[1]
                influences.append(derivative / objective.weights)
            self.assertTrue(all(np.allclose(value, 1 / np.sqrt(1 + standardized_error**2)) for value in influences))
        self.assertEqual(reference.depth_sigma_m, .002)

    def test_repeating_groups_preserves_the_optimized_pose(self):
        frame, state, model, uv, depth, plane, _, _ = make_case(noise=True)
        rv, tv, report = solve_joint(frame, state, model, uv, depth, plane, Config())
        for image_repeats, depth_repeats in ((3, 1), (1, 5), (3, 5)):
            with self.subTest(image=image_repeats, depth=depth_repeats):
                repeated_rv, repeated_tv, repeated_report = solve_joint(
                    frame, state, np.tile(model, (image_repeats, 1)),
                    np.tile(uv, (image_repeats, 1)), np.tile(depth, (depth_repeats, 1)), plane, Config(),
                )
                np.testing.assert_allclose(repeated_rv, rv, atol=2e-10, rtol=1e-8)
                np.testing.assert_allclose(repeated_tv, tv, atol=2e-10, rtol=1e-8)
                self.assertAlmostEqual(repeated_report["final_cost"], report["final_cost"], places=11)

    def test_current_translation_and_full_rotation_are_solved_without_temporal_lowpass(self):
        frame, state, model, uv, depth, plane, _, truth = make_case()
        original_state = deepcopy(state)
        rv, tv, report = solve_joint(frame, state, model, uv, depth, plane, Config())
        # The existing eight-evaluation budget can stop before machine precision
        # on this 4 cm / 7 degree step. It must still follow the current frame,
        # rather than apply a temporal interpolation toward the previous pose.
        rotation_error = cv2.Rodrigues(cv2.Rodrigues(rv)[0].T @ cv2.Rodrigues(truth[:3])[0])[0]
        self.assertLess(np.rad2deg(np.linalg.norm(rotation_error)), .5)
        self.assertLess(np.linalg.norm(tv.ravel() - truth[3:]), .002)
        np.testing.assert_array_equal(state["pose"], original_state["pose"])
        self.assertEqual(report["depth_points"], len(depth))
        self.assertGreater(report["depth_points"], 64)
        self.assertEqual(report["image_points"], len(model))
        self.assertLessEqual(report["nfev"], 8)
        self.assertLess(report["final_cost"], report["initial_cost"] * .01)

    def test_small_fraction_of_depth_outliers_does_not_pull_pose_to_outliers(self):
        frame, state, model, uv, depth, plane, _, truth = make_case(noise=True)
        contaminated = depth.copy()
        contaminated[::12] += plane.normal * .007
        rv, tv, _ = solve_joint(frame, state, model, uv, contaminated, plane, Config())
        self.assertLess(np.linalg.norm(tv.ravel() - truth[3:]), .0015)
        angle_error = cv2.Rodrigues(cv2.Rodrigues(rv)[0].T @ cv2.Rodrigues(truth[:3])[0])[0]
        self.assertLess(np.rad2deg(np.linalg.norm(angle_error)), .6)

    def test_empty_depth_and_mismatched_correspondences_are_explicit_errors(self):
        frame, state, model, uv, depth, plane, _, _ = make_case()
        with self.assertRaisesRegex(ValueError, "current plane inliers"):
            solve_joint(frame, state, model, uv, np.empty((0, 3)), plane, Config())
        with self.assertRaisesRegex(ValueError, "correspondence counts"):
            solve_joint(frame, state, model, uv[:-1], depth, plane, Config())

    def test_larger_plane_noise_changes_sigma_not_its_point_count(self):
        frame, state, model, uv, depth, plane, _, _ = make_case()
        plane.rmse_m = .004
        objective = _JointObjective(frame.camera, state, model, uv, depth, plane)
        repeated = _JointObjective(frame.camera, state, model, uv, np.tile(depth, (3, 1)), plane)
        self.assertEqual(objective.depth_sigma_m, .004)
        self.assertEqual(repeated.depth_sigma_m, .004)
        self.assertAlmostEqual(objective.depth_count * objective.depth_weight, 9.)
        self.assertAlmostEqual(repeated.depth_count * repeated.depth_weight, 9.)


if __name__ == "__main__":
    unittest.main()
