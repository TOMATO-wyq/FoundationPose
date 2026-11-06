"""Observed surface colors and an explicitly configured equal red/blue half-cube.

The directed color axis points from the blue half to the red half. It does not
remove the cube's residual rotational symmetry about that axis.
"""

import cv2
import numpy as np

from .geometry import upright_from_camera
from .native import geometry_kernel


def classify_hsv(hsv, cfg):
    hsv = np.asarray(hsv)
    kernel = geometry_kernel(cfg.geometry_backend)
    if kernel is not None and hsv.dtype == np.uint8:
        return kernel.classify_hsv(
            hsv, cfg.color_min_saturation, cfg.color_min_value, cfg.red_hue_ranges, cfg.blue_hue_range
        )
    colored = (hsv[..., 1] >= cfg.color_min_saturation) & (hsv[..., 2] >= cfg.color_min_value)
    red = np.zeros(hsv.shape[:-1], bool)
    for low, high in cfg.red_hue_ranges:
        red |= (hsv[..., 0] >= low) & (hsv[..., 0] <= high)
    blue = (hsv[..., 0] >= cfg.blue_hue_range[0]) & (hsv[..., 0] <= cfg.blue_hue_range[1])
    # Overlapping user thresholds are treated as unknown rather than double-counted.
    labels = np.zeros(hsv.shape[:-1], np.int8)
    labels[colored & red & ~blue] = 1
    labels[colored & blue & ~red] = -1
    return labels


def _samples(frame, obj, hsv, cfg):
    h, w = frame.depth.shape
    pose = np.asarray(obj["T_camera_from_box"])
    size = obj["size_m"]
    points, labels, reports = [], [], []
    used = set()
    for face in obj["faces"]:
        quad = np.asarray(face["quad"])
        low = np.maximum(np.floor(quad.min(axis=0)).astype(int), 0)
        high = np.minimum(np.ceil(quad.max(axis=0)).astype(int) + 1, (w, h))
        if np.any(high <= low):
            continue
        mask = np.zeros((high[1] - low[1], high[0] - low[0]), np.uint8)
        cv2.fillConvexPoly(mask, np.rint(quad - low).astype(np.int32), 255)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        yy, xx = np.nonzero(mask)
        if len(xx) > cfg.max_depth_samples:
            ids = np.linspace(0, len(xx) - 1, cfg.max_depth_samples).astype(int)
            xx, yy = xx[ids], yy[ids]
        xx, yy = xx + low[0], yy + low[1]
        unique = np.asarray([int(y) * w + int(x) not in used for x, y in zip(xx, yy)], bool)
        xx, yy = xx[unique], yy[unique]
        used.update(int(y) * w + int(x) for x, y in zip(xx, yy))
        z = frame.depth[yy, xx]
        good = np.isfinite(z) & (z >= cfg.min_depth_m) & (z <= cfg.max_depth_m)
        xx, yy, z = xx[good], yy[good], z[good]
        xyz = frame.camera.rays(np.column_stack((xx, yy))) * z[:, None]
        local = (xyz - pose[:3, 3]) @ pose[:3, :3]
        n = np.asarray(face["normal_camera"])
        plane_center = pose[:3, 3] + n * size / 2
        on_face = np.abs((xyz - plane_center) @ n) <= cfg.max_depth_residual_m
        on_face &= np.all(np.abs(local) <= size / 2 + cfg.max_depth_residual_m, axis=1)
        xyz, xx, yy = xyz[on_face], xx[on_face], yy[on_face]
        color = classify_hsv(hsv[yy, xx], cfg)
        count = len(color)
        reports.append(
            dict(
                source=face["source"],
                samples=count,
                red_fraction=float(np.mean(color == 1)) if count else 0.0,
                blue_fraction=float(np.mean(color == -1)) if count else 0.0,
                unknown_fraction=float(np.mean(color == 0)) if count else 0.0,
            )
        )
        points.extend(xyz)
        labels.extend(color)
    return np.asarray(points).reshape(-1, 3), np.asarray(labels), reports


def analyze_object(frame, obj, hsv, cfg):
    xyz, labels, reports = _samples(frame, obj, hsv, cfg)
    known = labels != 0
    result = dict(
        surface_samples=len(labels),
        classified_samples=int(known.sum()),
        faces=reports,
        red_fraction=float(np.mean(labels == 1)) if len(labels) else 0.0,
        blue_fraction=float(np.mean(labels == -1)) if len(labels) else 0.0,
        orientation_status="model_not_configured",
        red_axis_camera=None,
        red_axis_u=None,
        top_color=None,
        red_direction_yaw_u_rad=None,
        partition_plane_camera=None,
        red_half_center_camera_m=None,
        blue_half_center_camera_m=None,
        model="equal_red_blue_halves" if obj["size_m"] in cfg.two_color_sizes_m else None,
    )
    red, blue = result["red_fraction"], result["blue_fraction"]
    result["surface_label"] = (
        "red_blue"
        if red > 0.10 and blue > 0.10
        else "red" if red > 0.85 else "blue" if blue > 0.85 else "unknown"
    )
    if result["model"] is None:
        return result
    result["assumption"] = (
        "Known equal red/blue halves through cube center; colors alone do not identify all rotations."
    )
    if known.sum() < cfg.color_min_samples or known.mean() < cfg.color_min_coverage_fraction:
        result["orientation_status"] = "insufficient_color"
        return result
    pose = np.asarray(obj["T_camera_from_box"])
    delta = xyz[known] - pose[:3, 3]
    observed = labels[known]
    candidates = []
    for axis in range(3):
        for sign in (-1, 1):
            direction = pose[:3, axis] * sign
            distance = delta @ direction
            # Exclude only a narrow physical seam band, not an entire mismatching half.
            away = np.abs(distance) >= max(0.002, obj["size_m"] * 0.025)
            support = float(away.mean())
            accuracy = (
                float(np.mean(np.where(distance[away] >= 0, 1, -1) == observed[away])) if away.any() else 0.0
            )
            score = accuracy * support
            candidates.append(
                dict(
                    red_axis_camera=direction.tolist(),
                    fit_accuracy=accuracy,
                    usable_fraction=support,
                    score=score,
                )
            )
    candidates.sort(key=lambda c: c["score"], reverse=True)
    first, second = candidates[:2]
    result["candidates"] = candidates
    if first["fit_accuracy"] < cfg.color_min_fit or first["usable_fraction"] < 0.65:
        result["orientation_status"] = "color_model_mismatch"
        return result
    if first["score"] - second["score"] < cfg.color_ambiguity_margin:
        result["orientation_status"] = "ambiguous"
        return result
    direction = np.asarray(first["red_axis_camera"])
    result.update(
        red_axis_camera=first["red_axis_camera"],
        fit_accuracy=first["fit_accuracy"],
        evidence=(
            "observed_color_split"
            if np.any(observed == 1) and np.any(observed == -1)
            else "model_inferred_from_uniform_face"
        ),
        orientation_status="axis_unreferenced",
    )
    center = pose[:3, 3]
    result.update(
        partition_plane_camera=[*direction.tolist(), -float(direction @ center)],
        red_half_center_camera_m=(center + direction * obj["size_m"] / 4).tolist(),
        blue_half_center_camera_m=(center - direction * obj["size_m"] / 4).tolist(),
        remaining_symmetry_about_color_axis_rad=float(np.pi / 2),
        partition_definition="n dot p + d = 0; red positive, blue negative; model-derived partition",
    )
    u_from_c = upright_from_camera(cfg.up_camera)
    if u_from_c is None:
        return result
    d = u_from_c @ direction
    result["red_axis_u"] = d.tolist()
    if d[2] > np.cos(np.deg2rad(15)):
        result.update(orientation_status="red_up", top_color="red")
    elif d[2] < -np.cos(np.deg2rad(15)):
        result.update(orientation_status="blue_up", top_color="blue")
    elif abs(d[2]) < np.sin(np.deg2rad(15)):
        result.update(
            orientation_status="sideways",
            top_color="red_blue_split",
            red_direction_yaw_u_rad=float(np.arctan2(d[1], d[0]) % (2 * np.pi)),
        )
    else:
        result["orientation_status"] = "tilted_or_reference_error"
    return result


def analyze_colors(frame, objects, cfg):
    hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
    for obj in objects:
        obj["appearance"] = analyze_object(frame, obj, hsv, cfg)
