"""Depth geometry and planar PnP, with current-frame depth and upright checks."""

from dataclasses import dataclass
import time

import cv2
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

from .detector import sample_face
from .errors import FaceSizeMismatch
from .results import frame_metadata
from .geometry import (
    Plane,
    box_from_face,
    cube_vertices,
    fit_plane_configured,
    ray_plane,
    rigid_fit,
    rotation_z,
    square_points,
    transform,
    transform_points,
    upright_from_camera,
    yaw_from_pose,
    angle_difference,
)


@dataclass
class Estimate:
    size: float
    face: object
    plane: object
    samples: np.ndarray
    face_pose: np.ndarray
    box_pose: np.ndarray
    method: str
    metrics: dict
    cost: float
    diagnostic_trial: object = None


def _new_trial(diagnostics, method, size, stage):
    """One local attempt, allocated before a solver or validation can fail."""
    if diagnostics is None:
        return None
    trials = diagnostics.setdefault("trials", [])
    trial = dict(
        trial_index=len(trials), method=method, size_m=size, stage=stage,
        face_index=diagnostics.get("face_index"),
        accepted=False, selected=False, geometry_selected=False, failed_checks=[],
        reprojection_rms_px=None, reprojection_limit_px=None,
        depth_median_residual_m=None, normal_error_deg=None, upright_tilt_deg=None,
    )
    trials.append(trial)
    return trial


def _failed_trial(trial, stage, check, error):
    if trial is not None:
        trial.update(stage=stage, accepted=False, reason=str(error))
        if check not in trial["failed_checks"]:
            trial["failed_checks"].append(check)


def _face_failure(diagnostics, error):
    """Preparation failures have no pose yet, but must still leave an attempt."""
    if diagnostics is None:
        return
    diagnostics.update(status="rejected", reason=str(error))
    check = "size_mismatch" if isinstance(error, FaceSizeMismatch) else diagnostics.get("stage", "geometry")
    diagnostics.setdefault("failed_checks", [])
    if check not in diagnostics["failed_checks"]:
        diagnostics["failed_checks"].append(check)
    if not diagnostics.setdefault("trials", []):
        trial = _new_trial(diagnostics, "prepare", None, diagnostics.get("stage", "prepare"))
        _failed_trial(trial, trial["stage"], check, error)


def _diagnostic_value(value):
    """Keep attempted invalid values JSON-safe without hiding their rejection."""
    if isinstance(value, np.ndarray):
        return _diagnostic_value(value.tolist())
    if isinstance(value, np.generic):
        return _diagnostic_value(value.item())
    if isinstance(value, dict):
        return {key: _diagnostic_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _face_summary(diagnostics):
    # These are scalar summaries or fixed-size geometry. Per-pixel samples and
    # residual arrays remain confined to opt-in debug snapshots.
    fields = (
        "face_index", "sequence", "stamp_ns", "source", "quad", "inferred",
        "edge_support", "status", "stage", "reason", "failed_checks", "trials",
        "depth_fraction", "valid_depth_pixels", "interior_pixels", "plane_rmse_m",
        "plane_support", "plane_support_min", "plane_normal", "plane_d", "edge_lengths_m",
    )
    return _diagnostic_value({key: diagnostics[key] for key in fields if key in diagnostics})


def _make_estimate(frame, face, plane, samples, fraction, size, face_pose, method, cfg, diagnostics=None):
    trial = _new_trial(diagnostics, method, size, "pose_validation")
    if not np.isfinite(face_pose).all() or face_pose[2, 3] <= 0:
        error = ValueError("nonfinite or behind-camera pose")
        check = "invalid_parameters" if not np.isfinite(face_pose).all() else "behind_camera"
        _failed_trial(trial, "pose_validation", check, error)
        raise error
    if face_pose[:3, 2] @ face_pose[:3, 3] >= 0:
        error = ValueError("back-facing planar solution")
        _failed_trial(trial, "pose_validation", "back_facing", error)
        raise error
    if trial is not None:
        trial.update(
            face_normal_camera=face_pose[:3, 2].tolist(),
            face_center_camera_m=face_pose[:3, 3].tolist(),
        )
    projected = frame.camera.project(transform_points(face_pose, square_points(size)))
    if not np.isfinite(projected).all():
        error = ValueError("nonfinite projected face corners")
        _failed_trial(trial, "projection", "invalid_projection", error)
        raise error
    reprojection = float(np.sqrt(np.mean(np.sum((projected - face.quad) ** 2, axis=1))))
    depth_error = float(np.median(np.abs((samples - face_pose[:3, 3]) @ face_pose[:3, 2])))
    normal_error = float(np.rad2deg(np.arccos(np.clip(plane.normal @ face_pose[:3, 2], -1, 1))))
    pose, kind, tilt = box_from_face(face_pose, size, cfg.up_camera)
    if not np.isfinite([reprojection, depth_error, normal_error, tilt]).all():
        error = ValueError("nonfinite geometry residual")
        _failed_trial(trial, "residual_validation", "invalid_residual", error)
        raise error
    reprojection_limit = cfg.reprojection_limit(frame.camera)
    if trial is not None:
        failures = []
        if reprojection > reprojection_limit:
            failures.append("reprojection")
        if depth_error > cfg.max_depth_residual_m:
            failures.append("depth_residual")
        if normal_error > cfg.max_normal_error_deg:
            failures.append("normal_error")
        if cfg.up_camera is not None and tilt > cfg.max_upright_tilt_deg:
            failures.append("upright_tilt")
        trial.update(
                stage="residual_validation",
                accepted=not failures,
                failed_checks=failures,
                reprojection_rms_px=reprojection,
                reprojection_limit_px=reprojection_limit,
                depth_median_residual_m=depth_error,
                normal_error_deg=normal_error,
                upright_tilt_deg=tilt,
        )
    if reprojection > reprojection_limit:
        raise ValueError("reprojection error exceeds limit")
    if depth_error > cfg.max_depth_residual_m or normal_error > cfg.max_normal_error_deg:
        raise ValueError("PnP/geometry inconsistent with measured depth plane")
    if cfg.up_camera is not None and tilt > cfg.max_upright_tilt_deg:
        raise ValueError("observed face inconsistent with upright-box model")
    cost = (
        reprojection / reprojection_limit
        + depth_error / cfg.max_depth_residual_m
        + normal_error / cfg.max_normal_error_deg
    )
    metrics = dict(
        reprojection_rms_px=reprojection,
        reprojection_limit_px=reprojection_limit,
        depth_median_residual_m=depth_error,
        depth_fraction=fraction,
        plane_rmse_m=plane.rmse_m,
        plane_support=plane.support,
        normal_error_deg=normal_error,
        upright_tilt_deg=tilt,
        face_kind=kind,
        edge_support=face.edge_support,
    )
    if trial is not None:
        trial.update(stage="accepted", selection_cost=float(cost), box_center_camera_m=pose[:3, 3].tolist())
    return Estimate(size, face, plane, samples, face_pose, pose, method, metrics, float(cost), trial)


def prepare_face(frame, face, cfg, diagnostics=None):
    if diagnostics is not None:
        diagnostics["stage"] = "depth_sampling"
    samples, fraction = sample_face(frame, face.quad, cfg, diagnostics)
    if diagnostics is not None:
        diagnostics["stage"] = "plane_fit"
    if face.plane_hint is None or face.inferred:
        # A stack outline is only a proposal. Each child needs its own measured
        # plane; copying one parent normal couples the yaw of distinct cubes.
        plane = fit_plane_configured(samples, cfg)
    else:
        parent = face.plane_hint
        residual = np.abs(samples @ parent.normal + parent.d)
        selected = residual < cfg.plane_threshold_m
        if selected.sum() < cfg.min_depth_samples:
            raise ValueError("inferred face lacks current depth support")
        plane = Plane(
            parent.normal, parent.d, float(np.sqrt(np.mean(residual[selected] ** 2))), float(selected.mean())
        )
    if diagnostics is not None:
        diagnostics.update(
            plane_rmse_m=plane.rmse_m,
            plane_support=plane.support,
            plane_support_min=cfg.plane_min_support_fraction,
            plane_normal=plane.normal,
            plane_d=plane.d,
        )
        if not diagnostics.get("_lightweight"):
            diagnostics["sample_residual_m"] = np.abs(samples @ plane.normal + plane.d)
        diagnostics["stage"] = "plane_validation"
    if plane.rmse_m > cfg.plane_max_rmse_m or plane.support < cfg.plane_min_support_fraction:
        raise ValueError("nonplanar or mixed-depth face")
    samples = samples[np.abs(samples @ plane.normal + plane.d) < cfg.plane_threshold_m]
    if diagnostics is not None:
        diagnostics["stage"] = "ray_intersection"
    corners = ray_plane(frame.camera, face.quad, plane)
    lengths = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    if diagnostics is not None:
        diagnostics["edge_lengths_m"] = lengths
    return samples, fraction, plane, corners, lengths


def estimate_face(frame, face, cfg, prepared=None, diagnostics=None):
    if face.source == "bottom_occlusion_model":
        return estimate_partial_face(frame, face, cfg, diagnostics)
    try:
        samples, fraction, plane, corners, lengths = (
            prepare_face(frame, face, cfg, diagnostics) if prepared is None else prepared
        )
    except ValueError as error:
        _face_failure(diagnostics, error)
        raise
    if diagnostics is not None:
        diagnostics["stage"] = "pose_solving"
    estimates, rejected = [], []
    for size in cfg.sizes_m:
        size_error = float(np.max(np.abs(lengths / size - 1)))
        if size_error > cfg.size_tolerance_fraction:
            trial = _new_trial(diagnostics, "size_check", size, "size_validation")
            if trial is not None:
                trial["size_max_relative_error"] = size_error
            _failed_trial(trial, "size_validation", "size_mismatch", "observed face dimensions do not match size")
            continue
        methods = []
        if cfg.method in ("depth", "hybrid"):
            try:
                methods.append(("depth", rigid_fit(square_points(size), corners, backend=cfg.geometry_backend)))
            except ValueError as error:
                trial = _new_trial(diagnostics, "depth", size, "solver")
                _failed_trial(trial, "solver", "solver_error", error)
                _face_failure(diagnostics, error)
                raise
        if cfg.method in ("pnp", "hybrid"):
            try:
                result = cv2.solvePnPGeneric(
                    square_points(size),
                    np.ascontiguousarray(face.quad),
                    frame.camera.k,
                    frame.camera.d,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                for index, (rvec, tvec) in enumerate(zip(result[1], result[2])):
                    methods.append((f"pnp_{index}", transform(cv2.Rodrigues(rvec)[0], tvec.ravel())))
                if len(result[1]) == 0:
                    trial = _new_trial(diagnostics, "pnp", size, "solver")
                    _failed_trial(trial, "solver", "no_solution", "IPPE returned no pose")
            except cv2.error as error:
                trial = _new_trial(diagnostics, "pnp", size, "solver")
                _failed_trial(trial, "solver", "solver_error", error)
                rejected.append(f"IPPE: {error}")
        for method, pose in methods:
            try:
                estimate = _make_estimate(
                    frame, face, plane, samples, fraction, size, pose, method, cfg, diagnostics
                )
                estimate.metrics["size_max_relative_error"] = size_error
                estimates.append(estimate)
            except ValueError as error:
                if diagnostics is not None:
                    trial = diagnostics["trials"][-1]
                    if not trial["failed_checks"]:
                        _failed_trial(trial, "pose_validation", "geometry_error", error)
                    else:
                        trial["reason"] = str(error)
                rejected.append(f"{method}: {error}")
    if not estimates:
        if diagnostics is not None:
            diagnostics["stage"] = "candidate_validation"
        if not any(np.max(np.abs(lengths / s - 1)) <= cfg.size_tolerance_fraction for s in cfg.sizes_m):
            error = FaceSizeMismatch("observed face dimensions do not match configured square sizes")
        else:
            error = ValueError("; ".join(rejected) or "no valid pose")
        _face_failure(diagnostics, error)
        raise error
    estimates.sort(key=lambda e: e.cost)
    # In PnP-only mode don't hide an unresolved geometrically different second solution.
    if cfg.method == "pnp" and len(estimates) > 1:
        first, second = estimates[:2]
        normal_difference = np.rad2deg(
            np.arccos(np.clip(first.face_pose[:3, 2] @ second.face_pose[:3, 2], -1, 1))
        )
        if second.cost - first.cost < 0.08 and normal_difference > 5:
            error = ValueError("ambiguous planar PnP: depth does not distinguish candidates")
            if diagnostics is not None:
                diagnostics["stage"] = "ambiguity_validation"
            _face_failure(diagnostics, error)
            raise error
    if diagnostics is not None:
        diagnostics.update(status="geometry_pass", stage="geometry_pass")
        estimates[0].diagnostic_trial["geometry_selected"] = True
    return estimates, corners, plane


def estimate_partial_face(frame, face, cfg, diagnostics=None):
    """Give the partial-geometry path the same complete attempt accounting."""
    trial = _new_trial(diagnostics, "depth_partial", face.evidence.get("size_m"), "partial_geometry")
    try:
        estimates, corners, plane = _estimate_partial_face(frame, face, cfg, diagnostics)
    except ValueError as error:
        stage = "partial_geometry" if diagnostics is None else diagnostics.get("stage", "partial_geometry")
        _failed_trial(trial, stage, stage, error)
        _face_failure(diagnostics, error)
        raise
    if trial is not None:
        estimate = estimates[0]
        trial.update(
            stage="accepted", accepted=True, geometry_selected=True,
            selection_cost=estimate.cost,
            **{key: estimate.metrics.get(key) for key in (
                "reprojection_rms_px", "depth_median_residual_m", "normal_error_deg", "upright_tilt_deg"
            )},
        )
        estimate.diagnostic_trial = trial
    return estimates, corners, plane


def _estimate_partial_face(frame, face, cfg, diagnostics=None):
    """Depth/UP + an observed top/width anchor; no PnP on invented corners."""
    from .partial import partial_samples

    if cfg.method == "pnp":
        raise ValueError("bottom occlusion requires measured depth geometry, unavailable in PnP-only mode")
    if diagnostics is not None:
        diagnostics["stage"] = "depth_sampling"
    samples = partial_samples(frame, face, cfg)
    if diagnostics is not None:
        diagnostics["stage"] = "plane_fit"
    plane = fit_plane_configured(samples, cfg)
    if diagnostics is not None:
        diagnostics.update(
            plane_rmse_m=plane.rmse_m,
            plane_support=plane.support,
            plane_support_min=cfg.partial_min_plane_support_fraction,
            plane_normal=plane.normal,
            plane_d=plane.d,
        )
        if not diagnostics.get("_lightweight"):
            diagnostics.update(
                sample_uv=frame.camera.project(samples),
                sample_residual_m=np.abs(samples @ plane.normal + plane.d),
            )
        diagnostics["stage"] = "plane_validation"
    up = np.asarray(cfg.up_camera)
    tilt = float(np.rad2deg(np.arcsin(np.clip(abs(plane.normal @ up), 0, 1))))
    if (
        plane.rmse_m > cfg.plane_max_rmse_m
        or plane.support < cfg.partial_min_plane_support_fraction
        or tilt > cfg.max_upright_tilt_deg
    ):
        raise ValueError("occluded face inconsistent with measured upright plane")
    normal = plane.normal - up * (plane.normal @ up)
    normal /= np.linalg.norm(normal)
    offset = -float(np.median(samples @ normal))
    plane = Plane(normal, offset, plane.rmse_m, plane.support)
    if diagnostics is not None:
        diagnostics["stage"] = "partial_width_validation"
    top = ray_plane(frame.camera, face.quad[:2], plane)
    size = face.evidence["size_m"]
    width_error = float(abs(np.linalg.norm(top[1] - top[0]) / size - 1))
    if width_error > min(0.06, cfg.size_tolerance_fraction):
        raise ValueError("partial observed width does not match known size")
    if diagnostics is not None:
        diagnostics["stage"] = "partial_visibility_validation"
    if -normal @ top.mean(axis=0) / np.linalg.norm(top.mean(axis=0)) < 0.40:
        raise ValueError("partial face viewed at a grazing angle; width unobservable")
    horizontal = top[1] - top[0]
    horizontal -= up * (horizontal @ up)
    horizontal /= np.linalg.norm(horizontal)
    # Ensure a proper outward-facing face frame independent of image ordering.
    if np.cross(horizontal, up) @ normal < 0:
        horizontal = -horizontal
    face_center = top.mean(axis=0) - up * size / 2
    face_pose = transform(np.column_stack((horizontal, up, normal)), face_center)
    pose, kind, _ = box_from_face(face_pose, size, cfg.up_camera)
    depth_error = float(np.median(np.abs((samples - face_center) @ normal)))
    if diagnostics is not None:
        diagnostics["stage"] = "partial_depth_validation"
    if depth_error > cfg.max_depth_residual_m:
        raise ValueError("partial geometry inconsistent with current depth")
    metrics = dict(
        reprojection_rms_px=None,
        reprojection_definition="not evaluated on model-inferred bottom corners",
        depth_median_residual_m=depth_error,
        depth_fraction=face.evidence["visible_depth_fraction"],
        plane_rmse_m=plane.rmse_m,
        plane_support=plane.support,
        normal_error_deg=tilt,
        upright_tilt_deg=tilt,
        face_kind=kind,
        edge_support=face.edge_support,
        size_max_relative_error=width_error,
        observed_anchor="top edge and both side boundaries",
        **{k: v for k, v in face.evidence.items() if k.endswith("depth_fraction")},
    )
    cost = 0.5 + width_error / cfg.size_tolerance_fraction + depth_error / cfg.max_depth_residual_m
    estimate = Estimate(size, face, plane, samples, face_pose, pose, "depth_partial", metrics, float(cost))
    if diagnostics is not None:
        diagnostics.update(status="geometry_pass", stage="geometry_pass", partial_metrics=metrics)
    return [estimate], ray_plane(frame.camera, face.quad, plane), plane


def _group(estimates, cfg):
    groups = []
    u_from_c = upright_from_camera(cfg.up_camera)
    for estimate in sorted(estimates, key=lambda e: e.cost):
        matched = False
        for group in groups:
            anchor = group[0]
            if estimate.size != anchor.size:
                continue
            if np.linalg.norm(estimate.box_pose[:3, 3] - anchor.box_pose[:3, 3]) > max(
                0.012, anchor.size * 0.13
            ):
                continue
            if u_from_c is not None:
                a = yaw_from_pose(estimate.box_pose, u_from_c)
                b = yaw_from_pose(anchor.box_pose, u_from_c)
                if abs(angle_difference(a, b)) > np.deg2rad(10):
                    continue
            group.append(estimate)
            matched = True
            break
        if not matched:
            groups.append([estimate])
    # One observation per physical face; don't count depth and PnP as separate evidence.
    result = []
    for group in groups:
        unique = []
        for e in group:
            if not any(
                abs(e.face_pose[:3, 2] @ other.face_pose[:3, 2]) > 0.985
                or np.linalg.norm(e.face.quad - other.face.quad, axis=1).mean() < 5
                for other in unique
            ):
                unique.append(e)
        result.append((group, unique))
    return sorted(result, key=lambda pair: pair[0][0].cost - 0.25 * (len(pair[1]) - 1))


def _correspondences(anchor, observations):
    vertices = cube_vertices(anchor.size)
    camera_vertices = transform_points(anchor.box_pose, vertices)
    data = []
    for estimate in observations:
        if estimate.face.source == "bottom_occlusion_model":
            continue  # Inferred bottom corners are not independent image evidence.
        observed = transform_points(estimate.face_pose, square_points(estimate.size))
        rows, columns = linear_sum_assignment(
            np.linalg.norm(observed[:, None] - camera_vertices[None], axis=2)
        )
        if np.max(np.linalg.norm(observed[rows] - camera_vertices[columns], axis=1)) > anchor.size * 0.25:
            continue
        model = vertices[columns]
        data.append(
            (
                model,
                estimate.face.quad[rows],
                estimate.samples[:: max(1, len(estimate.samples) // 64)],
                estimate,
            )
        )
    return data


def refine_pose(frame, anchor, observations, cfg):
    from .native import geometry_kernel

    u_from_c = upright_from_camera(cfg.up_camera)
    if u_from_c is None or not cfg.refine_4d:
        return anchor.box_pose.copy(), dict(refinement="disabled_or_no_vertical_reference")
    data = _correspondences(anchor, observations)
    if not data:
        return anchor.box_pose.copy(), dict(refinement="no_consistent_correspondences")
    origin = u_from_c @ anchor.box_pose[:3, 3]
    axis = u_from_c @ anchor.box_pose[:3, 0]
    initial = np.r_[origin, np.arctan2(axis[1], axis[0])]
    multi_face_initialization = False
    if len(data) > 1:
        obj = np.concatenate([d[0] for d in data])
        img = np.concatenate([d[1] for d in data])
        if len(np.unique(obj, axis=0)) >= 6 and np.linalg.matrix_rank(obj - obj.mean(axis=0)) == 3:
            rvec = cv2.Rodrigues(anchor.box_pose[:3, :3])[0]
            try:
                ok, rv, tv = cv2.solvePnP(
                    obj,
                    img,
                    frame.camera.k,
                    frame.camera.d,
                    rvec,
                    anchor.box_pose[:3, 3].copy(),
                    True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                rotation = cv2.Rodrigues(rv)[0]
                if (
                    ok
                    and np.isfinite(tv).all()
                    and np.linalg.norm(tv.ravel() - anchor.box_pose[:3, 3]) < anchor.size * 0.15
                    and rotation[:, 2] @ np.asarray(cfg.up_camera) > 0.97
                ):
                    direction = u_from_c @ rotation[:, 0]
                    initial = np.r_[u_from_c @ tv.ravel(), np.arctan2(direction[1], direction[0])]
                    multi_face_initialization = True
            except cv2.error:
                pass

    def decode(parameters):
        return transform(u_from_c.T @ rotation_z(parameters[3]), u_from_c.T @ parameters[:3])

    prepared_data = []
    for obj, img, samples, observation in data:
        normal_local = np.cross(obj[1] - obj[0], obj[2] - obj[1])
        normal_local /= np.linalg.norm(normal_local)
        factor = 3 / (max(0.002, observation.plane.rmse_m) * np.sqrt(len(samples)))
        prepared_data.append((obj, img, samples, normal_local, obj.mean(axis=0), factor))
    kernel = geometry_kernel(cfg.geometry_backend)
    # Copy immutable observations once; each optimizer iteration crosses Python/C++ only once.
    native_evaluator = (
        None
        if kernel is None
        else kernel.RefinementEvaluator(prepared_data, u_from_c, frame.camera.k, frame.camera.d)
    )
    cache = {}

    def residual_and_jacobian(parameters):
        if "x" in cache and np.array_equal(parameters, cache["x"]):
            return cache["r"], cache["j"]
        if native_evaluator is not None:
            residual, jacobian = native_evaluator.evaluate(parameters)
            cache.update(x=parameters.copy(), r=residual, j=jacobian)
            return residual, jacobian
        pose = decode(parameters)
        angle = parameters[3]
        c, s = np.cos(angle), np.sin(angle)
        derivative_rotation = u_from_c.T @ np.array([[-s, -c, 0], [c, -s, 0], [0.0, 0, 0]])
        result, jacobians = [], []
        for obj, img, samples, normal_local, face_center_local, factor in prepared_data:
            points = transform_points(pose, obj)
            if np.any(points[:, 2] <= 0):
                length = sum(8 + len(d[2]) for d in data)
                return np.full(length, 1e6), np.zeros((length, 4))
            projected, jacobian = cv2.projectPoints(
                points, np.zeros(3), np.zeros(3), frame.camera.k, frame.camera.d
            )
            result.extend((projected.reshape(-1, 2) - img).ravel())
            point_jacobian = np.empty((len(obj), 3, 4))
            point_jacobian[:, :, :3] = u_from_c.T
            point_jacobian[:, :, 3] = obj @ derivative_rotation.T
            projection_jacobian = np.einsum(
                "nki,nij->nkj", jacobian[:, 3:6].reshape(-1, 2, 3), point_jacobian
            )
            jacobians.append(projection_jacobian.reshape(-1, 4))
            normal = pose[:3, :3] @ normal_local
            center = pose[:3, :3] @ face_center_local + pose[:3, 3]
            differences = samples - center
            result.extend((differences @ normal) * factor)
            depth_jacobian = np.empty((len(samples), 4))
            depth_jacobian[:, :3] = -(normal @ u_from_c.T) * factor
            depth_jacobian[:, 3] = (
                differences @ (derivative_rotation @ normal_local)
                - normal @ (derivative_rotation @ face_center_local)
            ) * factor
            jacobians.append(depth_jacobian)
        residual, jacobian = np.asarray(result), np.concatenate(jacobians)
        cache.update(x=parameters.copy(), r=residual, j=jacobian)
        return residual, jacobian

    delta = np.array([anchor.size * 0.2] * 3 + [np.deg2rad(15)])
    fit = least_squares(
        lambda x: residual_and_jacobian(x)[0],
        initial,
        jac=lambda x: residual_and_jacobian(x)[1],
        bounds=(initial - delta, initial + delta),
        loss="soft_l1",
        max_nfev=cfg.refine_max_nfev,
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
    )
    pose = decode(fit.x)
    if not np.isfinite(pose).all() or pose[2, 3] <= 0:
        return anchor.box_pose.copy(), dict(refinement="rejected_nonfinite")
    reprojection, depth_errors = [], []
    for obj, img, samples, _ in data:
        points = transform_points(pose, obj)
        reprojection.extend(np.sum((frame.camera.project(points) - img) ** 2, axis=1))
        normal = np.cross(points[1] - points[0], points[2] - points[1])
        normal /= np.linalg.norm(normal)
        depth_errors.extend(np.abs((samples - points.mean(axis=0)) @ normal))
    rms = float(np.sqrt(np.mean(reprojection)))
    depth_median = float(np.median(depth_errors))
    if rms > cfg.reprojection_limit(frame.camera) or depth_median > cfg.max_depth_residual_m:
        return anchor.box_pose.copy(), dict(refinement="rejected_residual", trial_reprojection_px=rms)
    return pose, dict(
        refinement="upright_4d",
        converged=bool(fit.success),
        nfev=int(fit.nfev),
        multi_face_pnp=multi_face_initialization,
        refined_reprojection_rms_px=rms,
        refined_depth_median_residual_m=depth_median,
    )


class Pipeline:
    """面候选 → 当前深度验收 → PnP/几何 → 4D 优化 → 颜色与场景验收。"""

    def __init__(self, cfg, configure_opencv=True, debug=False):
        from .detector import Detector

        self.cfg = cfg.validate()
        if configure_opencv:
            cv2.setNumThreads(cfg.opencv_threads)
            cv2.ocl.setUseOpenCL(False)
        self.detector = Detector(cfg)
        self.u_from_c = upright_from_camera(cfg.up_camera)
        self.debug_enabled = debug
        self.last_debug = None
        from .native import geometry_kernel

        kernel = geometry_kernel(cfg.geometry_backend)
        if configure_opencv and kernel is not None:
            kernel.configure_opencv(cfg.opencv_threads)
        if cfg.plane_backend == "pcl":
            from .pcl_backend import native

            native()

    def process(self, frame):
        start = time.perf_counter()
        debug = None
        if self.debug_enabled:
            from .debug import DetectionDebug

            debug = DetectionDebug(frame, self.cfg)
        faces, timing, edges = (
            self.detector.propose(frame) if debug is None else self.detector.propose(frame, debug)
        )

        face_checks, face_lookup, face_refs = [], {}, []

        def diagnostics_for(face):
            key = id(face)
            if key not in face_lookup:
                entry = dict(_lightweight=True, trials=[]) if debug is None else debug.face(face)
                entry.update(
                    face_index=len(face_checks), sequence=frame.sequence, stamp_ns=frame.stamp_ns,
                    source=face.source, quad=face.quad, inferred=face.inferred,
                    edge_support=face.edge_support, status="proposed", stage="prepare", failed_checks=[],
                )
                face_lookup[key] = entry
                face_refs.append(face)  # Temporary candidates must not reuse a Python id.
                face_checks.append(entry)
            return face_lookup[key]

        def measure(face, prepared=None):
            return estimate_face(
                frame, face, self.cfg, prepared, diagnostics=diagnostics_for(face)
            )

        estimates, rejected = [], list(self.detector.proposal_rejections)
        split_faces, prepared_halves = [], []
        for face in faces:
            prepared = None
            try:
                prepared = (
                    None
                    if face.source == "bottom_occlusion_model"
                    else prepare_face(frame, face, self.cfg, diagnostics_for(face))
                )
                if prepared is not None:
                    prepared_halves.append((face, prepared))
                found, _, _ = measure(face, prepared)
                estimates.extend(found)
            except ValueError as error:
                _face_failure(diagnostics_for(face), error)
                rejected.append(dict(quad=face.quad.tolist(), source=face.source, reason=str(error)))
                if isinstance(error, FaceSizeMismatch) and not face.inferred and prepared is not None:
                    try:
                        _, _, plane, points, _ = prepared
                        split_faces.extend(self.detector.split_stack(frame, face, points, plane))
                    except ValueError:
                        pass
        for face in self.detector.join_color_halves(frame, prepared_halves):
            try:
                found, _, _ = measure(face)
                estimates.extend(found)
            except ValueError as error:
                _face_failure(diagnostics_for(face), error)
                rejected.append(dict(quad=face.quad.tolist(), source=face.source, reason=str(error)))
        if self.cfg.allow_bottom_occlusion and self.cfg.up_camera is not None:
            partial_quads = []
            for original, prepared in prepared_halves:
                face = self.detector.anchored_partial_face(frame, original, prepared)
                if face is None or any(
                    np.linalg.norm(face.quad - q, axis=1).mean() < 5 for q in partial_quads
                ):
                    continue
                partial_quads.append(face.quad)
                try:
                    found, _, _ = measure(face)
                    estimates.extend(found)
                except ValueError as error:
                    _face_failure(diagnostics_for(face), error)
                    rejected.append(dict(quad=face.quad.tolist(), source=face.source, reason=str(error)))
        unique_splits = []
        for face in split_faces:
            if any(
                np.linalg.norm(face.quad - previous.quad, axis=1).mean() < 5 for previous in unique_splits
            ):
                continue
            unique_splits.append(face)
            try:
                found, _, _ = measure(face)
                estimates.extend(found)
            except ValueError as error:
                _face_failure(diagnostics_for(face), error)
                rejected.append(dict(quad=face.quad.tolist(), source=face.source, reason=str(error)))
        timing["color_line_fallback_used"] = False
        # Success on one configured size must not disable recovery of another.
        # Keep the extra search bounded and do not replace existing estimates.
        missing_sizes = set(self.cfg.sizes_m) - {e.size for e in estimates}
        if missing_sizes and self.cfg.color_line_fallback:
            fallback_start = time.perf_counter()
            fallback_faces = self.detector.color_line_fallback(frame, edges)
            timing["detector_ms"] += (time.perf_counter() - fallback_start) * 1000
            timing["proposal_count"] += len(fallback_faces)
            timing["color_line_fallback_used"] = True
            for face in fallback_faces:
                try:
                    found, _, _ = measure(face)
                    estimates.extend(e for e in found if e.size in missing_sizes)
                except ValueError as error:
                    _face_failure(diagnostics_for(face), error)
                    rejected.append(dict(quad=face.quad.tolist(), source=face.source, reason=str(error)))
        solve_end = time.perf_counter()
        groups = _group(estimates, self.cfg)
        objects = []
        candidate_limit = 1 if self.cfg.max_objects == 1 else min(12, self.cfg.max_objects * 4)
        for all_estimates, observations in groups[:candidate_limit]:
            anchor = observations[0]
            pose, refinement = refine_pose(frame, anchor, observations, self.cfg)
            yaw = yaw_from_pose(pose, self.u_from_c)
            sources = [
                dict(
                    quad=e.face.quad.tolist(),
                    source=e.face.source,
                    inferred=e.face.inferred,
                    evidence=e.face.evidence,
                    normal_camera=e.face_pose[:3, 2].tolist(),
                    metrics=e.metrics,
                    detection_check_index=diagnostics_for(e.face)["face_index"],
                    detection_trial_index=(None if e.diagnostic_trial is None else e.diagnostic_trial["trial_index"]),
                )
                for e in observations
            ]
            objects.append(
                dict(
                    status="detected" if yaw is not None else "pose_unreferenced",
                    valid=True,
                    yaw_reference_ready=yaw is not None,
                    size_m=anchor.size,
                    dimensions_nominal=self.cfg.dimensions_nominal,
                    center_camera_m=pose[:3, 3].tolist(),
                    T_camera_from_box=pose.tolist(),
                    position_u_m=None if self.u_from_c is None else (self.u_from_c @ pose[:3, 3]).tolist(),
                    yaw_u_rad=yaw,
                    yaw_period_rad=float(np.pi / 2),
                    method=anchor.method,
                    detection_check_index=diagnostics_for(anchor.face)["face_index"],
                    detection_trial_index=(None if anchor.diagnostic_trial is None else anchor.diagnostic_trial["trial_index"]),
                    evidence=(
                        "model_inferred" if any(e.face.inferred for e in observations) else "observed_faces"
                    ),
                    face_count=len(observations),
                    faces=sources,
                    quality_score=float(1 / (1 + anchor.cost)),
                    quality_score_definition="heuristic residual score, not a probability",
                    metrics={**anchor.metrics, **refinement},
                    alternatives=[
                        dict(
                            method=e.method,
                            cost=e.cost,
                            center_camera_m=e.box_pose[:3, 3].tolist(),
                            yaw_u_rad=yaw_from_pose(e.box_pose, self.u_from_c),
                            metrics=e.metrics,
                        )
                        for e in all_estimates
                    ],
                )
            )
        refinement_end = time.perf_counter()
        from .scene import select_scene

        if not self.cfg.require_target_colors:
            objects, suppressed = select_scene(objects, self.cfg.max_objects, self.cfg.up_camera)
        selection_first_end = time.perf_counter()
        if self.cfg.analyze_colors:
            from .appearance import analyze_colors

            analyze_colors(frame, objects, self.cfg)
        appearance_rejections = []
        if self.cfg.require_target_colors:
            kept = []
            for obj in objects:
                color = obj["appearance"]
                if (
                    color["classified_samples"] < self.cfg.color_min_samples
                    or color["classified_samples"] / max(1, color["surface_samples"])
                    < self.cfg.color_min_coverage_fraction
                ):
                    appearance_rejections.append(
                        dict(
                            reason="insufficient_configured_target_colors",
                            size_m=obj["size_m"],
                            center_camera_m=obj["center_camera_m"],
                            surface_samples=color["surface_samples"],
                            classified_samples=color["classified_samples"],
                        )
                    )
                else:
                    kept.append(obj)
            objects = kept
        appearance_end = time.perf_counter()
        if self.cfg.require_target_colors:
            objects, suppressed = select_scene(objects, self.cfg.max_objects, self.cfg.up_camera)
        # selected means this measured pose supplied evidence for a published
        # object. A subsequent 4D refinement is separately reported in metrics;
        # it is not claimed to equal the unrefined trial pose.
        for object_index, obj in enumerate(objects):
            for source in obj["faces"]:
                trial_index = source["detection_trial_index"]
                if trial_index is None:
                    continue
                trial = face_checks[source["detection_check_index"]]["trials"][trial_index]
                trial["selected"] = True
                trial.setdefault("selected_object_indices", []).append(object_index)
        finished = time.perf_counter()
        timing.update(
            solve_ms=(solve_end - start) * 1000 - timing["detector_ms"],
            refinement_ms=((selection_first_end - solve_end) + (finished - appearance_end)) * 1000,
            appearance_ms=(appearance_end - selection_first_end) * 1000,
            processing_ms=(finished - start) * 1000,
        )
        result = dict(
            frame_metadata(frame, self.cfg, self.u_from_c),
            valid_object_count=len(objects),
            objects=objects,
            rejected_faces=rejected,
            suppressed_hypotheses=suppressed,
            appearance_rejections=appearance_rejections,
            detection_checks=[_face_summary(check) for check in face_checks],
            timing=timing,
            frame_status=(
                "observed"
                if objects
                else "partial_or_rejected" if rejected or appearance_rejections else "no_target"
            ),
        )
        if debug is not None:
            debug.finish(result)
            self.last_debug = debug
        return result, edges
