"""Current-frame RGB-D objective for the planar tracking fallback.

Robust thresholds refer to physical observation noise: one image pixel and
at least two millimetres of depth noise. Group weights are applied *after*
the robust loss, so collecting more points cannot move those thresholds.
This solver does not select candidates or replace temporal geometry checks.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares


class _JointObjective:
    """Balanced robust group means, with an analytic six-parameter Jacobian."""

    def __init__(self, camera, state, model, uv, samples, plane):
        self.camera = camera
        self.model = np.asarray(model, dtype=np.float64)
        self.uv = np.asarray(uv, dtype=np.float64)
        self.depth = np.asarray(samples, dtype=np.float64)
        if self.model.ndim != 2 or self.model.shape[1] != 3 or not len(self.model):
            raise ValueError("joint image model must contain Nx3 points")
        if self.uv.shape != (len(self.model), 2):
            raise ValueError("joint image/model correspondence counts must match")
        if self.depth.ndim != 2 or self.depth.shape[1] != 3 or not len(self.depth):
            raise ValueError("joint depth must contain current plane inliers")
        if not all(np.isfinite(value).all() for value in (self.model, self.uv, self.depth)):
            raise ValueError("joint image and depth observations must be finite")
        self.normal = np.asarray(state["normal_model"], dtype=np.float64)
        quad = np.asarray(state["quad_model"], dtype=np.float64)
        if self.normal.shape != (3,) or quad.shape != (4, 3):
            raise ValueError("joint tracked face must have a model normal and four model corners")
        if not np.isfinite(self.normal).all() or not np.isfinite(quad).all():
            raise ValueError("joint tracked face geometry must be finite")
        self.face = quad.mean(axis=0)
        if not np.isfinite(plane.rmse_m) or plane.rmse_m < 0:
            raise ValueError("joint plane RMS must be finite and nonnegative")
        self.image_sigma_px = 1.0
        self.depth_sigma_m = max(0.002, float(plane.rmse_m))
        self.image_count = len(self.model)
        self.depth_count = len(self.depth)
        # Four image correspondences reproduce the original four-point image
        # scale. More correspondences improve sampling, not total group weight.
        self.image_weight = 4.0 / self.image_count
        self.depth_weight = 9.0 / self.depth_count
        self.weights = np.r_[
            np.full(2 * self.image_count, self.image_weight),
            np.full(self.depth_count, self.depth_weight),
        ]
        self._last_x = None

    def _evaluate(self, x):
        if self._last_x is not None and np.array_equal(x, self._last_x):
            return
        rotation, derivative = cv2.Rodrigues(x[:3])
        projected, image_jacobian = cv2.projectPoints(
            self.model, x[:3], x[3:], self.camera.k, self.camera.d
        )
        normal = rotation @ self.normal
        center = rotation @ self.face + x[3:]
        offsets = self.depth - center
        image = (projected.reshape(-1, 2) - self.uv).ravel() / self.image_sigma_px
        depth = (offsets @ normal) / self.depth_sigma_m
        depth_jacobian = np.empty((self.depth_count, 6), dtype=np.float64)
        # OpenCV returns one flattened dR/dr_i per row for vector->matrix.
        for axis in range(3):
            d_rotation = derivative[axis].reshape(3, 3)
            d_normal = d_rotation @ self.normal
            d_center = d_rotation @ self.face
            depth_jacobian[:, axis] = (offsets @ d_normal - normal @ d_center) / self.depth_sigma_m
        depth_jacobian[:, 3:] = -normal / self.depth_sigma_m
        self._last_residual = np.r_[image, depth]
        self._last_jacobian = np.vstack((image_jacobian[:, :6] / self.image_sigma_px, depth_jacobian))
        self._last_x = np.array(x, copy=True)

    def residual(self, x):
        self._evaluate(x)
        return self._last_residual.copy()

    def jacobian(self, x):
        self._evaluate(x)
        return self._last_jacobian.copy()

    def loss(self, squared_residual):
        """SciPy loss values and both derivatives; normalize after soft_l1."""
        root = np.sqrt(1.0 + squared_residual)
        return self.weights[None, :] * np.vstack((
            2.0 * squared_residual / (root + 1.0),
            1.0 / root,
            -0.5 / root**3,
        ))

    def costs(self, x):
        contributions = 0.5 * self.loss(self.residual(x)**2)[0]
        split = 2 * self.image_count
        return float(contributions[:split].sum()), float(contributions[split:].sum())


def solve_joint(frame, state, model, uv, samples, plane, cfg):
    """Return a fallback proposal and diagnostics; caller validates its geometry.

    The previous pose supplies only the initial value and existing search
    bounds. No history residual or display smoothing is applied here. Every
    supplied current-plane inlier participates in the objective.
    """
    objective = _JointObjective(frame.camera, state, model, uv, samples, plane)
    pose = np.asarray(state["pose"], dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("joint initial pose must be a finite 4x4 transform")
    start = np.r_[cv2.Rodrigues(pose[:3, :3])[0].ravel(), pose[:3, 3]]
    bounds = np.r_[np.full(3, 0.35), np.full(3, cfg.track_max_distance_m)]
    initial_image_cost, initial_depth_cost = objective.costs(start)
    fit = least_squares(
        objective.residual,
        start,
        jac=objective.jacobian,
        bounds=(start - bounds, start + bounds),
        max_nfev=8,
        loss=objective.loss,
        ftol=1e-4,
        xtol=1e-4,
        gtol=1e-4,
    )
    image_cost, depth_cost = objective.costs(fit.x)
    report = dict(
        objective="group_mean_standardized_soft_l1",
        image_points=objective.image_count,
        depth_points=objective.depth_count,
        image_sigma_px=objective.image_sigma_px,
        depth_sigma_m=objective.depth_sigma_m,
        image_loss_weight=objective.image_weight,
        depth_loss_weight=objective.depth_weight,
        image_reference_points=4,
        depth_group_weight=9.0,
        initial_cost=initial_image_cost + initial_depth_cost,
        final_cost=image_cost + depth_cost,
        image_cost=image_cost,
        depth_cost=depth_cost,
        converged=bool(fit.success),
        convergence_status=int(fit.status),
        convergence_message=str(fit.message),
        nfev=int(fit.nfev),
        njev=int(fit.njev),
        optimality=float(fit.optimality),
    )
    return fit.x[:3].reshape(3, 1), fit.x[3:].reshape(3, 1), report
