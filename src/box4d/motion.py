"""Bounded CV/CA position estimates for tracking search priors.

CV estimates position/velocity; CA also estimates acceleration. With gyro
context the internal axes are rotation-stabilized, with a camera-following origin.
Predictions never count as observations or change the last measurement time.
This filter does not establish that a target is visible in the current frame.
"""

from copy import copy, deepcopy

import numpy as np

from .camera_motion import context_id, context_rotation, rotation_matrix


def _timestamp(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("stamp_ns must be a nonnegative integer timestamp")
    return int(value)


def _position(value):
    try:
        position = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("position must contain three finite coordinates") from error
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("position must contain three finite coordinates")
    return position.copy()


class PositionKalman:
    """Immutable-on-update Kalman state, seeded by one measured 3D position.

    Acceleration noise is constant within each prediction interval and
    independent between intervals: Q = G sigma_a**2 G.T, with
    G = [0.5 dt**2 I; dt I]. The configured acceleration standard deviation
    therefore has units m/s². CA uses independent interval jerk noise with
    G = [dt**3/6 I; dt**2/2 I; dt I], sigma_j in m/s³.
    All propagation uses source timestamps.
    """

    def __init__(self, position, stamp_ns, cfg, camera_motion=None):
        self.stamp_ns = _timestamp(stamp_ns)
        self.observations = 1
        self.model = cfg.prediction_model
        if self.model not in ("cv", "ca"):
            raise ValueError("prediction_model must be cv or ca")
        self.camera_motion = deepcopy(camera_motion)
        # Snapshot the filter settings so later UI edits cannot mutate an
        # existing estimate while a predicted/corrected copy is being used.
        for name in (
            "prediction_max_gap_s",
            "prediction_max_std_m",
            "prediction_accel_std_m_s2",
            "prediction_measurement_std_m",
            "prediction_initial_velocity_std_m_s",
            "prediction_jerk_std_m_s3",
            "prediction_initial_accel_std_m_s2",
        ):
            value = getattr(cfg, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")
            setattr(self, "_" + name, float(value))
        self._mean = np.zeros(9 if self.model == "ca" else 6, dtype=np.float64)
        rotation = rotation_matrix(context_rotation(camera_motion))
        self._mean[:3] = rotation @ _position(position)
        self._covariance = np.diag(
            [self._prediction_measurement_std_m ** 2] * 3
            + [self._prediction_initial_velocity_std_m_s ** 2] * 3
            + ([self._prediction_initial_accel_std_m_s2 ** 2] * 3 if self.model == "ca" else [])
        )

    def compatible(self, camera_motion):
        return context_id(self.camera_motion) == context_id(camera_motion)

    def _predict_reference(self, stamp_ns):
        stamp = _timestamp(stamp_ns)
        if stamp < self.stamp_ns:
            raise ValueError("prediction timestamp precedes the last observation")
        try:
            dt = np.float64((stamp - self.stamp_ns) / 1e9)
            with np.errstate(over="raise", invalid="raise"):
                transition = np.eye(len(self._mean))
                transition[:3, 3:6] = np.eye(3) * dt
                if self.model == "ca":
                    transition[:3, 3:6] = np.eye(3) * dt
                    transition[:3, 6:] = np.eye(3) * (0.5 * dt * dt)
                    transition[3:6, 6:] = np.eye(3) * dt
                    derivative = np.vstack((np.eye(3) * (dt**3 / 6.),
                                            np.eye(3) * (0.5 * dt * dt), np.eye(3) * dt))
                    noise = derivative * self._prediction_jerk_std_m_s3
                else:
                    derivative = np.vstack((np.eye(3) * (0.5 * dt * dt), np.eye(3) * dt))
                    noise = derivative * self._prediction_accel_std_m_s2
                mean = transition @ self._mean
                covariance = transition @ self._covariance @ transition.T + noise @ noise.T
                covariance = (covariance + covariance.T) * 0.5
        except (OverflowError, FloatingPointError) as error:
            raise ValueError("prediction interval exceeds finite numeric range") from error
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError("prediction is not finite")
        return mean, covariance

    def predict(self, stamp_ns, camera_motion=None):
        """Predict in the requested current camera axes, including covariance."""
        if not self.compatible(camera_motion):
            raise ValueError("camera rotation reference changed or is unavailable")
        mean, covariance = self._predict_reference(stamp_ns)
        rotation = context_rotation(camera_motion).T
        axes = np.kron(np.eye(len(mean) // 3), rotation)
        return axes @ mean, axes @ covariance @ axes.T

    def updated(self, position, stamp_ns, camera_motion=None):
        """Return a new state corrected by a strictly newer real observation."""
        stamp = _timestamp(stamp_ns)
        if stamp <= self.stamp_ns:
            raise ValueError("observation timestamp must follow the last observation")
        if not self.compatible(camera_motion):
            raise ValueError("camera rotation reference changed or is unavailable")
        measured = context_rotation(camera_motion) @ _position(position)
        mean, covariance = self._predict_reference(stamp)
        measurement_noise = np.eye(3) * self._prediction_measurement_std_m ** 2
        innovation_covariance = covariance[:3, :3] + measurement_noise
        gain = np.linalg.solve(innovation_covariance, covariance[:, :3].T).T
        mean += gain @ (measured - mean[:3])
        residual = np.eye(len(mean))
        residual[:, :3] -= gain
        # Joseph form preserves symmetry/positive semidefiniteness despite
        # roundoff when measurements are much more certain than the prediction.
        covariance = residual @ covariance @ residual.T + gain @ measurement_noise @ gain.T
        covariance = (covariance + covariance.T) * 0.5
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError("updated estimate is not finite")
        result = copy(self)
        result.stamp_ns = stamp
        result.camera_motion = deepcopy(camera_motion)
        result.observations = self.observations + 1
        result._mean = mean
        result._covariance = covariance
        return result

    def can_predict(self, stamp_ns, camera_motion=None):
        """Whether a short, learned-motion search prior remains usable."""
        try:
            stamp = _timestamp(stamp_ns)
            age_ns = stamp - self.stamp_ns
            if (not self.compatible(camera_motion)
                    or self.observations < (3 if self.model == "ca" else 2) or age_ns < 0
                    or age_ns / 1e9 > self._prediction_max_gap_s):
                return False
            mean, covariance = self.predict(stamp, camera_motion)
            largest_variance = float(np.linalg.eigvalsh(covariance[:3, :3])[-1])
            return bool(mean[2] > 0 and largest_variance >= 0
                        and np.sqrt(largest_variance) <= self._prediction_max_std_m)
        except (ValueError, OverflowError, np.linalg.LinAlgError):
            return False
