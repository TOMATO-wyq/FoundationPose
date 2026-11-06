"""Upright side outlines and explicitly evidenced bottom occlusion.

The visible top and BOTH side edges anchor translation. A plane alone does
not. Missing bottom corners never become PnP/image observations. Foreground
depth must explain the missing region, or a measured T outline must establish
a wider coplanar support. Holes, background and ROI cuts are not occluders.
"""

import cv2
import numpy as np

from .detector import Face, _edge_support
from .geometry import Plane, fit_plane, ray_plane, unit


def _inside(quad, frame, cfg):
    w, h = frame.camera.width, frame.camera.height
    a, b, c, d = (0, 0, w, h) if cfg.roi_xyxy is None else cfg.roi_xyxy
    return (
        np.all(quad[:, 0] > a + 2)
        and np.all(quad[:, 0] < c - 3)
        and np.all(quad[:, 1] > b + 2)
        and np.all(quad[:, 1] < d - 3)
    )


def visibility(frame, quad, plane, cfg):
    """Current raw-depth evidence on an interior grid, including the hidden area."""
    t = np.linspace(0.06, 0.94, 17)
    s, v = np.meshgrid(t, t)
    uv = (1 - v.ravel()[:, None]) * (
        (1 - s.ravel()[:, None]) * quad[0] + s.ravel()[:, None] * quad[1]
    ) + v.ravel()[:, None] * ((1 - s.ravel()[:, None]) * quad[3] + s.ravel()[:, None] * quad[2])
    ij = np.rint(uv).astype(int)
    z = frame.depth[ij[:, 1], ij[:, 0]]
    predicted = ray_plane(frame.camera, uv, plane)[:, 2]
    good = np.isfinite(z) & (z > 0)
    delta = z - predicted
    on = good & (np.abs(delta) < cfg.plane_threshold_m * 1.5)
    front = good & (delta < -max(0.025, 3 * cfg.plane_threshold_m))
    return dict(
        visible_depth_fraction=float(on.mean()),
        foreground_depth_fraction=float(front.mean()),
        explained_depth_fraction=float((on | front).mean()),
        valid_depth_fraction=float(good.mean()),
    )


def _top_supported(frame, top_points, up, plane, cfg):
    t = np.linspace(0.15, 0.85, 15)[:, None]
    probes = (1 - t) * top_points[0] + t * top_points[1] + up * 0.025
    uv = np.rint(frame.camera.project(probes)).astype(int)
    if np.any(uv < 0) or np.any(uv[:, 0] >= frame.camera.width) or np.any(uv[:, 1] >= frame.camera.height):
        return False
    z = frame.depth[uv[:, 1], uv[:, 0]]
    predicted = ray_plane(frame.camera, uv, plane)[:, 2]
    valid = np.isfinite(z) & (z > 0)
    if np.mean(valid & (np.abs(z - predicted) < cfg.plane_threshold_m * 2)) > 0.4:
        return False
    if np.mean(valid & (z > predicted + 0.02)) >= 0.6:
        return True
    inside = np.rint(frame.camera.project(probes - up * 0.05)).astype(int)
    inside[:, 0] = np.clip(inside[:, 0], 0, frame.camera.width - 1)
    inside[:, 1] = np.clip(inside[:, 1], 0, frame.camera.height - 1)
    contrast = np.linalg.norm(
        frame.rgb[uv[:, 1], uv[:, 0]].astype(float) - frame.rgb[inside[:, 1], inside[:, 0]], axis=1
    )
    return np.mean(~valid & (contrast > 40)) >= 0.6


def _edges_on_plane(frame, quad, plane, bottom_ids, cfg):
    center = quad.mean(axis=0)
    for index in range(4):
        nxt = (index + 1) % 4
        if {index, nxt} == set(map(int, bottom_ids)):
            continue
        a, b = quad[index], quad[nxt]
        inward = center - (a + b) / 2
        inward = inward / max(1.0, np.linalg.norm(inward)) * 4.0
        uv = a + np.linspace(0.12, 0.88, 17)[:, None] * (b - a) + inward
        ij = np.rint(uv).astype(int)
        if (
            np.any(ij < 0)
            or np.any(ij[:, 0] >= frame.camera.width)
            or np.any(ij[:, 1] >= frame.camera.height)
        ):
            return False
        z = frame.depth[ij[:, 1], ij[:, 0]]
        valid = np.isfinite(z) & (z > 0)
        xyz = frame.camera.rays(uv[valid]) * z[valid, None]
        finite = np.isfinite(xyz).all(axis=1)
        on = np.zeros(len(z), dtype=bool)
        on[np.flatnonzero(valid)[finite]] = (
            np.abs(xyz[finite] @ plane.normal + plane.d) < cfg.plane_threshold_m * 1.5
        )
        if on.mean() < cfg.partial_min_plane_support_fraction:
            return False
    return True


def _wider_support_contact(frame, top, up, plane, size, cfg):
    """A coplanar wider base can hide the seam without foreground depth.

    Require two real side boundaries above the nominal bottom, and measured
    plane continuation OUTSIDE both sides just below it (a visible T outline).
    Merely extending the same plane downwards is not sufficient evidence.
    """
    horizontal = unit(top[1] - top[0])
    xyz = []
    for i, sign in ((0, -1), (1, 1)):
        for fraction in (0.30, 0.55, 0.80, 1.05, 1.10, 1.15):
            xyz.append(top[i] - up * size * fraction + horizontal * sign * 0.025)
    uv = np.rint(frame.camera.project(xyz)).astype(int)
    if np.any(uv < 0) or np.any(uv[:, 0] >= frame.camera.width) or np.any(uv[:, 1] >= frame.camera.height):
        return False
    z = frame.depth[uv[:, 1], uv[:, 0]]
    predicted = ray_plane(frame.camera, uv, plane)[:, 2]
    on = (np.isfinite(z) & (z > 0) & (np.abs(z - predicted) < cfg.plane_threshold_m * 1.5)).reshape(2, 6)
    return bool(np.all(on[:, :3].mean(axis=1) <= 0.34) and np.all(on[:, 3:].mean(axis=1) >= 0.66))


def anchored_partial_face(
    frame, face, prepared, cfg, *, max_visible_fraction=0.90, min_foreground_fraction=0.08
):
    """A measured partial rectangle can anchor recovery despite patch holes."""
    if not cfg.allow_bottom_occlusion or cfg.up_camera is None or face.inferred:
        return None
    samples, _, measured, points, lengths = prepared
    up = np.asarray(cfg.up_camera)
    ids = np.argsort(points @ up)[-2:]
    # The two highest vertices must be one observed edge, not a diagonal.
    if (int(ids[0]) - int(ids[1])) % 4 == 2:
        return None
    top = points[ids]
    width = np.linalg.norm(top[1] - top[0])
    height = np.ptp(points @ up)
    # Fewer observed boundaries justify a tighter width gate than complete faces.
    sizes = [
        s
        for s in cfg.sizes_m
        if abs(width / s - 1) <= min(0.06, cfg.size_tolerance_fraction)
        and 0.30 * s < height < max_visible_fraction * s
    ]
    if len(sizes) != 1 or abs((top[1] - top[0]) @ up) / width > 0.15:
        return None
    if abs(measured.normal @ up) > np.sin(np.deg2rad(cfg.max_upright_tilt_deg)):
        return None
    size = sizes[0]
    if not _edges_on_plane(frame, face.quad, measured, np.argsort(points @ up)[:2], cfg):
        return None
    normal = unit(measured.normal - up * (measured.normal @ up))
    if -normal @ points.mean(axis=0) / np.linalg.norm(points.mean(axis=0)) < 0.40:
        return None  # A grazing sliver cannot reliably anchor metric width.
    plane = Plane(normal, -float(np.median(samples @ normal)), measured.rmse_m, measured.support)
    top_uv = face.quad[ids]
    top_uv = top_uv[np.argsort(top_uv[:, 0])]
    top = ray_plane(frame.camera, top_uv, plane)
    q = frame.camera.project(np.array([top[0], top[1], top[1] - up * size, top[0] - up * size]))
    if not _inside(q, frame, cfg):
        return None
    report = visibility(frame, q, plane, cfg)
    contact = report["visible_depth_fraction"] >= 0.85 and _wider_support_contact(
        frame, top, up, plane, size, cfg
    )
    occluded = (
        report["foreground_depth_fraction"] >= min_foreground_fraction
        and report["explained_depth_fraction"] >= 0.85
    )
    if report["visible_depth_fraction"] < 0.30 or not (contact or occluded):
        return None
    if not _top_supported(frame, top, up, plane, cfg):
        return None
    # Both measured vertical sides must align with the upright model.
    directions = np.roll(points, -1, axis=0) - points
    aligned = np.abs(directions @ up) / lengths
    if np.count_nonzero(aligned > 0.96) != 2 or face.edge_support < cfg.partial_min_edge_support_fraction:
        return None
    report.update(
        inferred_from=(
            "visible T outline on a wider coplanar support and known size"
            if contact
            else "observed top and side segments, known size, measured foreground occluder"
        ),
        support_contact=contact,
        size_m=size,
        missing_edge="support_contact" if contact else "bottom",
        visible_outline_quad=face.quad.tolist(),
        metric_width_m=float(width),
        visible_height_m=float(height),
    )
    return Face(q, "bottom_occlusion_model", face.edge_support, True, report, plane)


def upright_patch_faces(frame, mask, hint, cfg, distance, color_labels=None):
    if cfg.up_camera is None or not cfg.upright_outline_proposals:
        return []
    up = np.asarray(cfg.up_camera)
    if abs(hint.normal @ up) > np.sin(np.deg2rad(cfg.max_upright_tilt_deg)):
        return []
    # The existing detector already handles complete outlines. Avoid a second
    # fit on the common complete frontal case; partial recovery has a budget.
    bx, by, bw, bh = cv2.boundingRect(mask)
    ratio = bh / max(1, bw)
    if any(abs(ratio / level - 1) < 0.07 for level in range(1, cfg.max_objects + 1)):
        return []
    mask = mask[by : by + bh, bx : bx + bw].copy()
    if cfg.analyze_colors:
        from .appearance import classify_hsv

        if color_labels is None:
            color_labels = classify_hsv(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV), cfg)
        colored = color_labels[by : by + bh, bx : bx + bw] != 0
        if np.count_nonzero((mask > 0) & colored) > np.count_nonzero(mask) * 0.60:
            mask = np.where(colored, mask, 0).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    yy, xx = np.nonzero(cv2.erode(mask, np.ones((5, 5), np.uint8)))
    if len(xx) < cfg.min_depth_samples:
        return []
    ids = np.linspace(0, len(xx) - 1, min(len(xx), cfg.max_depth_samples)).astype(int)
    uv = np.column_stack((xx[ids] + bx, yy[ids] + by))
    z = frame.depth[uv[:, 1], uv[:, 0]]
    # fit_plane filters its own input, but the constrained fit below also uses
    # these samples. Keep invalid depth out of every subsequent calculation.
    # Retain the original sample count so holes still reduce plane support.
    sample_count = len(z)
    valid = np.isfinite(z) & (z >= cfg.min_depth_m) & (z <= cfg.max_depth_m)
    samples = frame.camera.rays(uv[valid]) * z[valid, None]
    samples = samples[np.isfinite(samples).all(axis=1) & (samples[:, 2] > 0)]
    if len(samples) < cfg.min_depth_samples:
        return []
    try:
        measured = fit_plane(
            samples, cfg.plane_threshold_m, cfg.min_depth_samples, backend=cfg.geometry_backend
        )
        # Constrain the normal to be horizontal using the explicit UP reference.
        n = unit(measured.normal - up * (measured.normal @ up))
        d = -float(np.median(samples @ n))
        residual = samples @ n + d
        good = np.abs(residual) < cfg.plane_threshold_m
        support_fraction = float(good.sum() / sample_count)
        if (
            good.sum() < cfg.min_depth_samples
            or support_fraction < cfg.partial_min_plane_support_fraction
            or np.sqrt(np.mean(residual[good] ** 2)) > cfg.plane_max_rmse_m
        ):
            return []
        plane = Plane(n, d, float(np.sqrt(np.mean(residual[good] ** 2))), support_fraction)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
        contour = max(contours, key=cv2.contourArea).reshape(-1, 2) + (bx, by)
        points = ray_plane(frame.camera, contour, plane)
    except ValueError:
        return []
    horizontal = unit(np.cross(up, n))
    if horizontal[0] < 0:
        horizontal = -horizontal
    x, y = points @ horizontal, points @ up
    left, right = np.quantile(x, [0.01, 0.99])
    bottom, top = np.quantile(y, [0.01, 0.99])
    width, height = right - left, top - bottom
    matches = [s for s in cfg.sizes_m if abs(width / s - 1) <= cfg.size_tolerance_fraction]
    if len(matches) != 1:
        return []
    size = matches[0]
    if height < 0.30 * size or height > max(1, cfg.max_objects) * size * 1.15:
        return []

    def corners(z_top, z_bottom):
        xyz = np.array(
            [
                horizontal * left + up * z_top - n * d,
                horizontal * right + up * z_top - n * d,
                horizontal * right + up * z_bottom - n * d,
                horizontal * left + up * z_bottom - n * d,
            ]
        )
        return frame.camera.project(xyz)

    whole = corners(top, bottom)
    _, support = _edge_support(whole, distance, backend=cfg.geometry_backend)
    # Top and both visible side lines must be real current-image boundaries.
    if min(support[0], support[1], support[3]) < 0.60:
        return []
    if not _edges_on_plane(frame, whole, plane, (2, 3), cfg):
        return []  # A T-shaped patch cannot anchor a full-width larger top.
    faces = []
    levels = int(np.round(height / size))
    full = levels >= 1 and abs(height / (levels * size) - 1) <= cfg.size_tolerance_fraction
    if full and _inside(whole, frame, cfg) and support[2] >= 0.60:
        return []  # Complete faces belong to the ordinary observation path.
    if not cfg.allow_bottom_occlusion or not cfg.allow_stack_inference and height > size:
        return []
    # A coplanar paint seam or a hole cannot anchor the top of a hidden cube.
    if not _top_supported(frame, ray_plane(frame.camera, whole[:2], plane), up, plane, cfg):
        return []
    # Full upper cubes may anchor a partly occluded bottom cube. Explicitly
    # bound stack count; do not tile a plane indefinitely into imaginary boxes.
    count = int(np.ceil(height / size))
    if count > cfg.max_objects or height / size - (count - 1) < 0.30:
        return []
    missing = count * size - height
    if not 0.10 * size < missing < 0.70 * size:
        return []
    for index in range(count):
        q = corners(top - index * size, top - (index + 1) * size)
        if not _inside(q, frame, cfg):
            continue
        evidence = visibility(frame, q, plane, cfg)
        if index == count - 1:
            if (
                evidence["visible_depth_fraction"] < 0.30
                or evidence["foreground_depth_fraction"] < 0.08
                or evidence["explained_depth_fraction"] < 0.85
            ):
                continue
        elif evidence["visible_depth_fraction"] < 0.80:
            continue
        evidence.update(
            inferred_from="visible top and two sides, known size, measured foreground occluder",
            size_m=size,
            stack_index_from_top=index,
            stack_levels=count,
            missing_edge="bottom" if index == count - 1 else None,
            visible_outline_quad=corners(top - index * size, max(bottom, top - (index + 1) * size)).tolist(),
            metric_width_m=float(width),
            visible_height_m=float(min(size, height - index * size)),
        )
        # Keep top-first order: it indexes the two genuinely observed top corners.
        faces.append(
            Face(
                q,
                "bottom_occlusion_model",
                float(np.mean([support[0], support[1], support[3]])),
                True,
                evidence,
                plane,
            )
        )
    return faces


def partial_samples(frame, face, cfg):
    """Reject foreground before fitting; never sample hidden corners as depth."""
    q = face.quad
    # Enforce the same image/ROI boundary policy as complete faces.
    if not _inside(q, frame, cfg):
        raise ValueError("occluded face clipped by image or search ROI")
    lo = np.floor(q.min(axis=0)).astype(int)
    hi = np.ceil(q.max(axis=0)).astype(int) + 1
    mask = np.zeros((hi[1] - lo[1], hi[0] - lo[0]), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(q - lo).astype(np.int32), 255)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    yy, xx = np.nonzero(mask)
    xx, yy = xx + lo[0], yy + lo[1]
    z = frame.depth[yy, xx]
    good = np.isfinite(z) & (z >= cfg.min_depth_m) & (z <= cfg.max_depth_m)
    uv = np.column_stack((xx[good], yy[good]))
    xyz = frame.camera.rays(uv) * z[good, None]
    good_plane = np.abs(xyz @ face.plane_hint.normal + face.plane_hint.d) < cfg.plane_threshold_m
    xyz = xyz[good_plane]
    if len(xyz) < cfg.min_depth_samples or len(xyz) / max(1, len(xx)) < 0.30:
        raise ValueError("occluded face lacks current measured plane support")
    if len(xyz) > cfg.max_depth_samples:
        xyz = xyz[np.linspace(0, len(xyz) - 1, cfg.max_depth_samples).astype(int)]
    return xyz
