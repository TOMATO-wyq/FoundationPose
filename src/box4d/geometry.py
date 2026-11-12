"""Projection, robust planes, cube geometry and explicitly referenced yaw."""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json

import cv2
import numpy as np

from .native import geometry_kernel


def unit(value):
    a = np.asarray(value, dtype=np.float64)
    length = np.linalg.norm(a)
    if not np.isfinite(a).all() or length < 1e-10:
        raise ValueError("invalid direction vector")
    return a / length


@dataclass
class Camera:
    width: int
    height: int
    k: object
    d: object = ()
    distortion_model: str = "plumb_bob"
    frame_id: str = "camera_color_optical_frame"
    rectified: bool = False

    def __post_init__(self):
        self.width, self.height = int(self.width), int(self.height)
        self.k = np.asarray(self.k, np.float64).reshape(3, 3).copy()
        self.d = np.asarray(self.d, np.float64).reshape(-1).copy()
        if (
            self.width < 8
            or self.height < 8
            or not np.isfinite(self.k).all()
            or not np.isfinite(self.d).all()
        ):
            raise ValueError("invalid camera dimensions or calibration")
        if self.k[0, 0] <= 0 or self.k[1, 1] <= 0 or not np.allclose(self.k[2], (0, 0, 1)):
            raise ValueError("invalid intrinsic matrix")
        if self.distortion_model not in ("plumb_bob", "rational_polynomial", "brown_conrady", "none"):
            raise ValueError(
                f"unsupported distortion model {self.distortion_model!r}; rectify explicitly first"
            )
        if self.d.size not in (0, 4, 5, 8, 12, 14):
            raise ValueError("unsupported OpenCV distortion coefficient count")
        if self.rectified and np.any(np.abs(self.d) > 1e-12):
            raise ValueError("rectified images must use zero residual distortion")
        self.k.flags.writeable = False
        self.d.flags.writeable = False
        self._calibration_id = hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:16]

    @property
    def calibration_id(self):
        return self._calibration_id

    def to_dict(self):
        return dict(
            width=self.width,
            height=self.height,
            k=self.k.tolist(),
            d=self.d.tolist(),
            distortion_model=self.distortion_model,
            frame_id=self.frame_id,
            rectified=self.rectified,
        )

    def rays(self, uv):
        uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        if not len(uv):
            return np.empty((0, 3), np.float64)
        xy = cv2.undistortPoints(uv, self.k, self.d).reshape(-1, 2)
        return np.column_stack((xy, np.ones(len(xy))))

    def project(self, xyz):
        xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
        if not len(xyz):
            return np.empty((0, 2), np.float64)
        return cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), self.k, self.d)[0].reshape(-1, 2)

    def crop_resize(self, xyxy, width, height):
        x0, y0, x1, y1 = map(int, xyxy)
        if not (0 <= x0 < x1 <= self.width and 0 <= y0 < y1 <= self.height):
            raise ValueError("crop lies outside calibration image")
        k = self.k.copy()
        k[0, 2] -= x0
        k[1, 2] -= y0
        k[0] *= width / (x1 - x0)
        k[1] *= height / (y1 - y0)
        return Camera(width, height, k, self.d, self.distortion_model, self.frame_id, self.rectified)


@dataclass
class Plane:
    normal: np.ndarray
    d: float
    rmse_m: float
    support: float


@lru_cache(maxsize=256)
def _plane_samples(count, trials):
    # 固定种子原本每次都重建 RNG；同样的点数可以复用同样的假设。
    ids = np.random.default_rng(19).integers(0, count, size=(trials, 3))
    ids.flags.writeable = False
    return ids


def fit_plane(points, threshold=0.008, min_points=24, trials=48, *, backend="auto"):
    """Same deterministic RANSAC in NumPy/C++; normal faces the camera."""
    p = np.asarray(points, np.float64).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1) & (p[:, 2] > 0)]
    if len(p) < min_points:
        raise ValueError("insufficient valid depth samples")
    ids = _plane_samples(len(p), trials)
    native = geometry_kernel(backend)
    if native is not None:
        return Plane(*native.fit_plane(p, ids, threshold, min_points))
    return _fit_plane_numpy(p, ids, threshold, min_points)


def _fit_plane_numpy(p, ids, threshold, min_points):
    """Reference implementation retained for numerical comparisons."""
    normals = np.cross(p[ids[:, 1]] - p[ids[:, 0]], p[ids[:, 2]] - p[ids[:, 0]])
    lengths = np.linalg.norm(normals, axis=1)
    ok = lengths > 1e-9
    if not ok.any():
        raise ValueError("degenerate plane samples")
    normals = normals[ok] / lengths[ok, None]
    offsets = -np.sum(normals * p[ids[ok, 0]], axis=1)
    # einsum avoids launching an oversized BLAS thread pool for tiny 3D products.
    residuals = np.abs(np.einsum("ni,ki->nk", p, normals) + offsets)
    counts = np.sum(residuals < threshold, axis=0)
    mask = residuals[:, int(np.argmax(counts))] < threshold
    for _ in range(3):
        if int(mask.sum()) < min_points:
            raise ValueError("plane has insufficient inliers")
        selected = p[mask]
        center = selected.mean(axis=0)
        q = selected - center
        values, vectors = np.linalg.eigh(q.T @ q / len(q))
        if values[1] < 1e-9:
            raise ValueError("collinear plane samples")
        normal = vectors[:, 0]
        d = -float(normal @ center)
        mask = np.abs(p @ normal + d) < threshold
    if normal @ center > 0:
        normal, d = -normal, -d
    rms = float(np.sqrt(np.mean((p[mask] @ normal + d) ** 2)))
    return Plane(normal, d, rms, float(mask.mean()))


def ray_plane(camera, uv, plane):
    rays = camera.rays(uv)
    denominator = rays @ plane.normal
    if np.any(np.abs(denominator) < 1e-5):
        raise ValueError("ray nearly parallel to face plane")
    z = -plane.d / denominator
    if np.any(z <= 0) or not np.isfinite(z).all():
        raise ValueError("face intersection behind camera")
    return rays * z[:, None]


def plane_limits(points, cfg):
    """Bounded stereo depth uncertainty, referenced to one metre.

    Keep the near-field floor; distance must never relax the metric size,
    pose reprojection or median depth residual checks.
    """
    threshold, rmse = cfg.plane_threshold_m, cfg.plane_max_rmse_m
    if cfg.plane_distance_noise and len(points):
        scale = max(1.0, float(np.median(np.asarray(points)[:, 2])) ** 2)
        threshold = max(threshold, min(cfg.plane_noise_cap_m, threshold * scale))
        rmse = max(rmse, min(cfg.plane_noise_cap_m * 2 / 3, rmse * scale))
    return threshold, rmse


def fit_plane_configured(points, cfg):
    if cfg.plane_backend == "pcl":
        from .pcl_backend import fit_plane as pcl_fit_plane

        from dataclasses import replace

        return pcl_fit_plane(points, replace(cfg, plane_threshold_m=plane_limits(points, cfg)[0]))
    return fit_plane(points, plane_limits(points, cfg)[0], cfg.min_depth_samples, backend=cfg.geometry_backend)


def plane_spatial_support(points, plane, threshold):
    """Fraction of populated image cells supported by the measured plane.

    A good global inlier ratio alone can hide a different depth on one side.
    Empty/undersampled cells contribute no fabricated evidence.
    """
    uv = np.asarray(points)[:, :2] / np.asarray(points)[:, 2, None]
    span = np.maximum(np.ptp(uv, axis=0), 1e-9)
    cells = np.minimum(3, ((uv - uv.min(axis=0)) / span * 4).astype(int))
    ids = cells[:, 0] + 4 * cells[:, 1]
    counts = np.bincount(ids, minlength=16)
    inliers = np.abs(points @ plane.normal + plane.d) < threshold
    supported = np.bincount(ids, weights=inliers, minlength=16)
    occupied = counts >= 3
    return float(np.mean(supported[occupied] >= counts[occupied] * 0.5)) if occupied.any() else 0.0


def order_quad(points, *, backend="auto"):
    """Clockwise image order, starting near image top-left; never invent corners."""
    p = np.asarray(points, np.float64).reshape(4, 2)
    native = geometry_kernel(backend)
    if native is not None:
        return native.order_quad(p)
    if not np.isfinite(p).all() or len(np.unique(p, axis=0)) != 4:
        raise ValueError("invalid quadrilateral")
    p = p[np.argsort(np.arctan2(p[:, 1] - p[:, 1].mean(), p[:, 0] - p[:, 0].mean()))]
    p = np.roll(p, -int(np.argmin(p[:, 0] + p[:, 1])), axis=0)
    if not cv2.isContourConvex(p.astype(np.float32)) or cv2.contourArea(p.astype(np.float32)) < 1:
        raise ValueError("nonconvex or degenerate quadrilateral")
    return p


def square_points(size):
    h = float(size) / 2
    # OpenCV IPPE_SQUARE mandated order.
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def cube_vertices(size):
    h = float(size) / 2
    return np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)])


def cube_edges():
    return [(i, j) for i in range(8) for j in range(i + 1, 8) if (i ^ j) in (1, 2, 4)]


def transform_points(matrix, points):
    t = np.asarray(matrix, np.float64).reshape(4, 4)
    return np.asarray(points) @ t[:3, :3].T + t[:3, 3]


def transform(rotation, translation):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, translation
    return result


def rigid_fit(source, target, *, backend="auto"):
    a, b = np.asarray(source, np.float64), np.asarray(target, np.float64)
    native = geometry_kernel(backend)
    if native is not None:
        return native.rigid_fit(a, b)
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - ac).T @ (b - bc))
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    return transform(r, bc - r @ ac)


def upright_from_camera(up_camera):
    if up_camera is None:
        return None
    up = unit(up_camera)
    forward = np.array([0.0, 0.0, 1.0])
    x = forward - up * (up @ forward)
    if np.linalg.norm(x) < 0.1:
        forward = np.array([1.0, 0.0, 0.0])
        x = forward - up * (up @ forward)
    x = unit(x)
    return np.stack((x, np.cross(up, x), up))


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def angle_difference(a, b, period=np.pi / 2):
    return (a - b + period / 2) % period - period / 2


def yaw_from_pose(pose, u_from_c):
    if u_from_c is None:
        return None
    direction = u_from_c @ pose[:3, 0]
    return float(np.arctan2(direction[1], direction[0]) % (np.pi / 2))


def box_from_face(face_pose, size, up_camera=None):
    n = unit(face_pose[:3, 2])
    center = face_pose[:3, 3] - n * size / 2
    if up_camera is None:
        return transform(face_pose[:3, :3], center), "unreferenced", 0.0
    up = unit(up_camera)
    alignment = float(n @ up)
    if abs(alignment) > np.sqrt(0.5):
        x = unit(face_pose[:3, 0] - up * (face_pose[:3, 0] @ up))
        kind = "top" if alignment > 0 else "bottom"
        tilt = np.rad2deg(np.arccos(np.clip(abs(alignment), -1, 1)))
    else:
        x = unit(n - up * alignment)
        kind = "side"
        normal_tilt = np.arcsin(np.clip(abs(alignment), 0, 1))
        vertical_edge_alignment = max(abs(face_pose[:3, 0] @ up), abs(face_pose[:3, 1] @ up))
        edge_tilt = np.arccos(np.clip(vertical_edge_alignment, 0, 1))
        tilt = np.rad2deg(max(normal_tilt, edge_tilt))
    return transform(np.column_stack((x, np.cross(up, x), up)), center), kind, float(tilt)
