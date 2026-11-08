"""Bounded constant-velocity position estimates for tracking search priors.

The six-dimensional state is [x, y, z, vx, vy, vz] in camera coordinates.
Predictions never count as observations or change the last measurement time.
This filter does not establish that a target is visible in the current frame.
"""

from copy import copy

import numpy as np


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
    therefore has units m/s². All propagation uses source timestamps.
    """

    def __init__(self, position, stamp_ns, cfg):
        self.stamp_ns = _timestamp(stamp_ns)
        self.observations = 1
        # Snapshot the filter settings so later UI edits cannot mutate an
        # existing estimate while a predicted/corrected copy is being used.
        for name in (
            "prediction_max_gap_s",
            "prediction_max_std_m",
            "prediction_accel_std_m_s2",
            "prediction_measurement_std_m",
            "prediction_initial_velocity_std_m_s",
        ):
            value = getattr(cfg, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive and finite")
            setattr(self, "_" + name, float(value))
        self._mean = np.zeros(6, dtype=np.float64)
        self._mean[:3] = _position(position)
        self._covariance = np.diag(
            [self._prediction_measurement_std_m ** 2] * 3
            + [self._prediction_initial_velocity_std_m_s ** 2] * 3
        )

    def predict(self, stamp_ns):
        """Return independent (mean, covariance) arrays at this or a later time."""
        stamp = _timestamp(stamp_ns)
        if stamp < self.stamp_ns:
            raise ValueError("prediction timestamp precedes the last observation")
        try:
            dt = np.float64((stamp - self.stamp_ns) / 1e9)
            with np.errstate(over="raise", invalid="raise"):
                transition = np.eye(6)
                transition[:3, 3:] = np.eye(3) * dt
                acceleration = np.vstack((np.eye(3) * (0.5 * dt * dt), np.eye(3) * dt))
                noise = (acceleration * self._prediction_accel_std_m_s2)
                mean = transition @ self._mean
                covariance = transition @ self._covariance @ transition.T + noise @ noise.T
                covariance = (covariance + covariance.T) * 0.5
        except (OverflowError, FloatingPointError) as error:
            raise ValueError("prediction interval exceeds finite numeric range") from error
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError("prediction is not finite")
        return mean, covariance

    def updated(self, position, stamp_ns):
        """Return a new state corrected by a strictly newer real observation."""
        stamp = _timestamp(stamp_ns)
        if stamp <= self.stamp_ns:
            raise ValueError("observation timestamp must follow the last observation")
        measured = _position(position)
        mean, covariance = self.predict(stamp)
        measurement_noise = np.eye(3) * self._prediction_measurement_std_m ** 2
        innovation_covariance = covariance[:3, :3] + measurement_noise
        gain = np.linalg.solve(innovation_covariance, covariance[:, :3].T).T
        mean += gain @ (measured - mean[:3])
        residual = np.eye(6)
        residual[:, :3] -= gain
        # Joseph form preserves symmetry/positive semidefiniteness despite
        # roundoff when measurements are much more certain than the prediction.
        covariance = residual @ covariance @ residual.T + gain @ measurement_noise @ gain.T
        covariance = (covariance + covariance.T) * 0.5
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError("updated estimate is not finite")
        result = copy(self)
        result.stamp_ns = stamp
        result.observations = self.observations + 1
        result._mean = mean
        result._covariance = covariance
        return result

    def can_predict(self, stamp_ns):
        """Whether a short, learned-motion search prior remains usable."""
        try:
            stamp = _timestamp(stamp_ns)
            age_ns = stamp - self.stamp_ns
            if (self.observations < 2 or age_ns < 0
                    or age_ns / 1e9 > self._prediction_max_gap_s):
                return False
            mean, covariance = self.predict(stamp)
            largest_variance = float(np.linalg.eigvalsh(covariance[:3, :3])[-1])
            return bool(mean[2] > 0 and largest_variance >= 0
                        and np.sqrt(largest_variance) <= self._prediction_max_std_m)
        except (ValueError, OverflowError, np.linalg.LinAlgError):
            return False
