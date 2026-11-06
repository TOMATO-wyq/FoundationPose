"""Stationary accelerometer UP, explicitly separate from validated metrology."""

import numpy as np


def stationary_up(accel, camera_from_accel):
    values = np.asarray(accel, np.float64)
    rotation = np.asarray(camera_from_accel, np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 50 or not np.isfinite(values).all():
        raise ValueError("need at least 50 finite three-axis acceleration samples")
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or np.linalg.det(rotation) < 0.999
    ):
        raise ValueError("IMU-to-camera transform must be a proper rotation")
    mean = values.mean(axis=0)
    magnitude = float(np.linalg.norm(mean))
    scatter = float(np.sqrt(np.mean(np.sum((values - mean) ** 2, axis=1))))
    if not 8.5 <= magnitude <= 11.2:
        raise ValueError("acceleration magnitude inconsistent with stationary gravity in m/s^2")
    if scatter > 0.20:
        raise ValueError("camera is moving/vibrating; stationary UP estimate rejected")
    # RealSense publishes proper acceleration: a stationary accelerometer points UP.
    up = rotation @ mean / magnitude
    return up.tolist(), dict(
        samples=len(values),
        mean_accel_m_s2=mean.tolist(),
        mean_magnitude_m_s2=magnitude,
        rms_scatter_m_s2=scatter,
        absolute_accuracy_validated=False,
        note="Stationary estimate; accelerometer bias and factory extrinsics are not independently calibrated.",
    )
