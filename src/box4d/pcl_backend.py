"""PCL adapters. Missing native dependencies are errors, never silent fallbacks."""

import importlib
import time

import cv2
import numpy as np

from .geometry import Plane


def native():
    try:
        return importlib.import_module("box4d._pcl_backend")
    except ImportError as error:
        raise RuntimeError(
            "PCL backend is unavailable. Build for this Python with: bash src/build_box4d_pcl.sh"
        ) from error


def backend_info():
    return dict(
        name="pcl",
        version=native().pcl_version,
        segmentation="OrganizedMultiPlaneSegmentation.segmentAndRefine",
        normals="IntegralImageNormalEstimation",
        local_fit="SACSegmentation/SAC_RANSAC",
    )


def fit_plane(points, cfg):
    result = native().fit_plane(
        np.ascontiguousarray(points, np.float32),
        cfg.plane_threshold_m,
        cfg.min_depth_samples,
        cfg.pcl_ransac_iterations,
    )
    return Plane(
        np.asarray(result["normal"], np.float64),
        float(result["d"]),
        float(result["rmse_m"]),
        float(result["support"]),
    )


def segment_faces(detector, frame, hsv, debug=None):
    """Keep organized adjacency; refine each returned region against raw depth.

    Nearest-neighbor label upsampling does not interpolate depth or bridge holes.
    PCL region labels restrict each patch, so a fit cannot grow across a crease
    solely because points there happen to be near the fitted plane.
    """
    cfg = detector.cfg
    start = time.perf_counter()
    if detector._ray_cache is None or detector._ray_cache[0] != frame.camera.calibration_id:
        yy, xx = np.indices(frame.depth.shape)
        rays = frame.camera.rays(np.column_stack((xx.ravel(), yy.ravel()))).reshape(*frame.depth.shape, 3)
        detector._ray_cache = frame.camera.calibration_id, rays.astype(np.float32)
    rays = detector._ray_cache[1]
    valid = np.isfinite(frame.depth) & (frame.depth >= cfg.min_depth_m) & (frame.depth <= cfg.max_depth_m)
    if cfg.roi_xyxy is not None:
        a, b, c, d = cfg.roi_xyxy
        roi = np.zeros(valid.shape, bool)
        roi[b:d, a:c] = True
        valid &= roi
    stride = cfg.pcl_stride
    depth = np.where(valid, frame.depth, np.nan)
    xyz = np.ascontiguousarray(rays[::stride, ::stride] * depth[::stride, ::stride, None], np.float32)
    if min(xyz.shape[:2]) < 8:
        return []
    result = native().segment(
        xyz,
        max(cfg.min_depth_samples, int(np.ceil(cfg.min_face_area_px / stride**2))),
        cfg.plane_threshold_m,
        cfg.pcl_angular_threshold_deg,
        max(3.0, cfg.pcl_normal_smoothing_px / stride),
        cfg.pcl_max_curvature,
        cfg.pcl_depth_change_factor,
        cfg.pcl_normal_method == "covariance",
        debug is not None,
    )
    coarse_labels = result["labels"]
    if debug is not None:
        normals = result.get("normals")
        normal_image = np.zeros((*normals.shape[:2], 3), np.uint8)
        good = np.isfinite(normals).all(axis=2)
        normal_image[good] = np.clip((normals[good, ::-1] + 1) * 127.5, 0, 255).astype(np.uint8)
        debug.images["normals"] = normal_image
        debug.images["plane_labels"] = np.zeros(frame.depth.shape, np.int32)
        debug.backend = dict(
            **backend_info(),
            normal_method=cfg.pcl_normal_method,
            normal_ms=result["normal_ms"],
            segmentation_ms=result["segmentation_ms"],
            stride=stride,
            valid_points=result["valid_points"]
        )
    if hsv is None:
        hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
    from .appearance import classify_hsv

    target = classify_hsv(hsv, cfg) != 0
    coarse_target = target[::stride, ::stride]
    color_counts = np.bincount(
        coarse_labels[coarse_target & (coarse_labels >= 0)], minlength=len(result["planes"])
    )
    planes = sorted(
        result["planes"],
        key=lambda p: (
            -int(color_counts[p["id"]]) if cfg.analyze_colors and cfg.color_plane_priority else 0,
            -p["inlier_count"],
        ),
    )
    masks = []
    kept_planes = 0
    for info in planes:
        normal = np.asarray(info["normal"], np.float64)
        report = dict(
            id=int(info["id"]),
            normal=normal.tolist(),
            d=float(info["d"]),
            rmse_m=float(info["rmse_m"]),
            support=1.0,
            support_definition="fraction within already segmented PCL region",
            samples=int(info["inlier_count"]),
            prioritized=False,
            patches=0,
            status="no_valid_patch",
            backend="pcl_organized",
        )
        if debug is not None:
            debug.planes.append(report)
        # Do not upsample/background-test hundreds of non-target PCL regions.
        # PCL has already fitted each plane on the organized coarse cloud.
        if (
            cfg.require_target_colors
            and cfg.color_plane_priority
            and color_counts[info["id"]] < max(6, cfg.color_min_samples // stride**2)
        ):
            report["status"] = "no_target_color_support"
            continue
        if kept_planes >= cfg.max_scene_planes:
            report["status"] = "plane_budget"
            continue
        coarse_y, coarse_x = np.nonzero(coarse_labels == info["id"])
        if not len(coarse_x):
            report["status"] = "empty_pcl_region"
            continue
        x0, y0 = int(coarse_x.min() * stride), int(coarse_y.min() * stride)
        x1 = min(frame.camera.width, int((coarse_x.max() + 1) * stride))
        y1 = min(frame.camera.height, int((coarse_y.max() + 1) * stride))
        if x1 - x0 < 2 or y1 - y0 < 2:
            report["status"] = "region_too_small"
            continue
        local_labels = cv2.resize(
            coarse_labels[coarse_y.min() : coarse_y.max() + 1, coarse_x.min() : coarse_x.max() + 1],
            (x1 - x0, y1 - y0),
            interpolation=cv2.INTER_NEAREST,
        )
        local_depth = frame.depth[y0:y1, x0:x1]
        local_valid = valid[y0:y1, x0:x1]
        local_rays = rays[y0:y1, x0:x1]
        # Check residual on measured full-resolution depth; never fill holes.
        local_points = local_rays * np.where(local_valid, local_depth, 0)[..., None]
        residual = np.abs(np.einsum("ijk,k->ij", local_points, normal) + info["d"])
        selected = (local_labels == info["id"]) & local_valid & (residual < cfg.plane_threshold_m)
        raw_count = int(np.count_nonzero((local_labels == info["id"]) & local_valid))
        report["support"] = float(selected.sum() / max(1, raw_count))
        count, components, stats, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8))
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < cfg.min_face_area_px:
                continue
            yy, xx = np.nonzero(components == index)
            pick = np.linspace(0, len(xx) - 1, min(len(xx), cfg.max_depth_samples)).astype(int)
            points = local_rays[yy[pick], xx[pick]] * local_depth[yy[pick], xx[pick], None]
            if np.linalg.norm(np.ptp(points, axis=0)) > max(cfg.sizes_m) * 3.2:
                report["status"] = "extent_too_large"
                continue
            patch = np.zeros(frame.depth.shape, np.uint8)
            patch[y0:y1, x0:x1] = (components == index).astype(np.uint8) * 255
            rms = float(np.sqrt(np.mean(residual[yy, xx] ** 2)))
            masks.append((patch, Plane(normal, float(info["d"]), rms, 1.0)))
            if not report["patches"]:
                kept_planes += 1
            report["patches"] += 1
            report["status"] = "kept"
            if debug is not None:
                debug.images["plane_labels"][y0:y1, x0:x1][components == index] = info["id"] + 1
    if debug is not None:
        debug.backend["adapter_total_ms"] = (time.perf_counter() - start) * 1000
    return masks
