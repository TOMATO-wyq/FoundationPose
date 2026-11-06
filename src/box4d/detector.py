"""RGB edges and organized-depth planes propose *observed* face quadrilaterals."""

from dataclasses import dataclass, field
import time

import cv2
import numpy as np

from .geometry import fit_plane, order_quad, ray_plane
from .edges import rgb_depth_edges
from .native import geometry_kernel


@dataclass
class Face:
    quad: np.ndarray
    source: str
    edge_support: float = 0.0
    inferred: bool = False
    evidence: dict = field(default_factory=dict)
    plane_hint: object = None


def sample_face(frame, quad, cfg, diagnostics=None):
    """Sample interior depth, never directly reading noisy silhouette corners."""
    h, w = frame.depth.shape
    if np.any(quad < 1) or np.any(quad[:, 0] > w - 2) or np.any(quad[:, 1] > h - 2):
        raise ValueError("face boundary clipped by image; complete center pose unobservable")
    if cfg.roi_xyxy is not None:
        x0, y0, x1, y1 = cfg.roi_xyxy
        if (
            np.any(quad[:, 0] <= x0 + 1)
            or np.any(quad[:, 0] >= x1 - 2)
            or np.any(quad[:, 1] <= y0 + 1)
            or np.any(quad[:, 1] >= y1 - 2)
        ):
            raise ValueError("face boundary clipped by search ROI; expand ROI to recover the whole face")
    native = geometry_kernel(cfg.geometry_backend)
    if native is not None:
        uv, z, fraction, valid_count, interior_count = native.sample_face_pixels(
            frame.depth, quad, cfg.min_depth_m, cfg.max_depth_m, cfg.max_depth_samples
        )
        if diagnostics is not None:
            diagnostics.update(
                depth_fraction=fraction, valid_depth_pixels=valid_count, interior_pixels=interior_count
            )
        if valid_count < cfg.min_depth_samples or fraction < cfg.min_depth_fraction:
            raise ValueError("insufficient interior depth coverage")
        if diagnostics is not None:
            diagnostics["sample_uv"] = uv
        return frame.camera.rays(uv) * z[:, None], fraction
    lo = np.maximum(np.floor(quad.min(axis=0)).astype(int), 0)
    hi = np.minimum(np.ceil(quad.max(axis=0)).astype(int) + 1, (w, h))
    if np.any(hi - lo < 4):
        raise ValueError("face outside image or too small")
    mask = np.zeros((hi[1] - lo[1], hi[0] - lo[0]), np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad - lo).astype(np.int32), 255)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    local = frame.depth[lo[1] : hi[1], lo[0] : hi[0]]
    valid = (mask > 0) & np.isfinite(local) & (local >= cfg.min_depth_m) & (local <= cfg.max_depth_m)
    fraction = float(valid.sum() / max(1, np.count_nonzero(mask)))
    yy, xx = np.nonzero(valid)
    if diagnostics is not None:
        diagnostics.update(
            depth_fraction=fraction, valid_depth_pixels=len(xx), interior_pixels=int(np.count_nonzero(mask))
        )
    if len(xx) < cfg.min_depth_samples or fraction < cfg.min_depth_fraction:
        raise ValueError("insufficient interior depth coverage")
    if len(xx) > cfg.max_depth_samples:
        selected = np.linspace(0, len(xx) - 1, cfg.max_depth_samples).astype(int)
        xx, yy = xx[selected], yy[selected]
    uv = np.column_stack((xx + lo[0], yy + lo[1]))
    xyz = frame.camera.rays(uv) * local[yy, xx, None]
    if diagnostics is not None:
        diagnostics["sample_uv"] = uv
    return xyz, fraction


def _edge_support(quad, distance, *, backend="auto"):
    native = geometry_kernel(backend)
    if native is not None:
        return native.edge_support(quad, distance)
    h, w = distance.shape
    scores = []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        t = np.linspace(0.08, 0.92, max(12, min(60, int(np.linalg.norm(a - b) / 3))))
        uv = np.rint(a + t[:, None] * (b - a)).astype(int)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        scores.append(float(np.mean(distance[uv[inside, 1], uv[inside, 0]] <= 3.5)) if inside.any() else 0.0)
    return float(np.mean(scores)), scores


def _refine_quad(quad, edge_image, *, backend="auto"):
    """Fit each observed line locally; do not force perspective edges perpendicular."""
    native = geometry_kernel(backend)
    if native is not None:
        return native.refine_quad(quad, edge_image)
    x0, y0 = np.maximum(np.floor(quad.min(axis=0) - 5).astype(int), 0)
    x1, y1 = np.minimum(np.ceil(quad.max(axis=0) + 6).astype(int), edge_image.shape[::-1])
    yy, xx = np.nonzero(edge_image[y0:y1, x0:x1])
    points = np.column_stack((xx + x0, yy + y0)).astype(np.float64)
    if len(points) < 16:
        return quad
    lines = []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        edge = b - a
        length = np.linalg.norm(edge)
        tangent = edge / length
        normal = np.array([-tangent[1], tangent[0]])
        along = (points - a) @ tangent
        subset = points[
            (np.abs((points - a) @ normal) < 2.5) & (along > length * 0.12) & (along < length * 0.88)
        ]
        if len(subset) < 8:
            lines.append((normal, -float(normal @ a)))
        else:
            line = cv2.fitLine(subset.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            n = np.array([-line[1], line[0]], np.float64)
            lines.append((n, -float(n @ line[2:])))
    corners = []
    for i in range(4):
        n0, d0 = lines[i - 1]
        n1, d1 = lines[i]
        matrix = np.stack((n0, n1))
        if abs(np.linalg.det(matrix)) < 0.05:
            return quad
        corners.append(np.linalg.solve(matrix, -np.array([d0, d1])))
    corners = np.asarray(corners)
    if np.max(np.linalg.norm(corners - quad, axis=1)) > 5:
        return quad
    try:
        return order_quad(corners, backend=backend)
    except ValueError:
        return quad


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        self._grid_cache = None
        self._ray_cache = None
        self.proposal_rejections = []

    def _grid(self, camera):
        key = (camera.calibration_id, self.cfg.plane_grid_step, tuple(self.cfg.roi_xyxy or ()))
        if self._grid_cache is None or self._grid_cache[0] != key:
            step = self.cfg.plane_grid_step
            xx, yy = np.meshgrid(
                np.arange(step // 2, camera.width, step), np.arange(step // 2, camera.height, step)
            )
            uv = np.column_stack((xx.ravel(), yy.ravel()))
            if self.cfg.roi_xyxy is not None:
                a, b, c, d = self.cfg.roi_xyxy
                uv = uv[(uv[:, 0] >= a) & (uv[:, 0] < c) & (uv[:, 1] >= b) & (uv[:, 1] < d)]
            self._grid_cache = key, uv, camera.rays(uv)
        return self._grid_cache[1:]

    def _depth_masks(self, frame, hsv=None, debug=None):
        """稀疏点 RANSAC 找模型，再用稠密点到平面的距离提取连通面片。

        平行面仍按各自的 d 区分；同一平面上不相连的区域也保留独立面片。
        深度空洞不作为拟合点，也不会由形态学补出深度。
        """
        cfg = self.cfg
        if cfg.plane_backend == "pcl":
            from .pcl_backend import segment_faces

            return segment_faces(self, frame, hsv, debug)
        uv, rays = self._grid(frame.camera)
        z = frame.depth[uv[:, 1], uv[:, 0]]
        good = np.isfinite(z) & (z >= cfg.min_depth_m) & (z <= cfg.max_depth_m)
        uv, xyz = uv[good], rays[good] * z[good, None]
        priority = np.zeros(len(xyz), bool)
        if cfg.analyze_colors and cfg.color_plane_priority:
            from .appearance import classify_hsv

            if hsv is None:
                hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
            priority = classify_hsv(hsv[uv[:, 1], uv[:, 0]], cfg) != 0
        remaining = np.ones(len(xyz), bool)
        masks = []
        for attempt in range(cfg.max_scene_planes):
            preferred = remaining & priority
            prioritize = preferred.sum() >= cfg.min_depth_samples
            p = xyz[preferred if prioritize else remaining]
            if len(p) < cfg.min_depth_samples:
                break
            report = dict(
                id=attempt, samples=len(p), prioritized=bool(prioritize), status="fitting", patches=0
            )
            if debug is not None:
                debug.planes.append(report)
            try:
                plane = fit_plane(
                    p, cfg.plane_threshold_m, cfg.min_depth_samples, trials=24, backend=cfg.geometry_backend
                )
            except ValueError as error:
                report.update(status="fit_failed", reason=str(error))
                break
            report.update(normal=plane.normal.tolist(), d=plane.d, rmse_m=plane.rmse_m, support=plane.support)
            selected = remaining & (np.abs(xyz @ plane.normal + plane.d) < cfg.plane_threshold_m)
            remaining[selected] = False
            if selected.sum() < cfg.min_depth_samples:
                report["status"] = "too_few_inliers"
                continue
            bounded = selected & priority if prioritize else selected
            if bounded.sum() < cfg.min_depth_samples:
                report["status"] = "too_few_colored_inliers"
                continue
            points = xyz[bounded]
            # Large wall/floor planes cannot be one or two boxes of these sizes.
            if np.linalg.norm(np.ptp(points, axis=0)) > max(cfg.sizes_m) * 3.2:
                report["status"] = "extent_too_large"
                continue
            # A target plane intersects the floor/wall in a narrow strip. Do
            # not let unrelated coplanar background pixels expand its bounds.
            hull = cv2.convexHull(uv[bounded].astype(np.int32))
            x, y, width, height = cv2.boundingRect(hull)
            pad = cfg.plane_grid_step * 2
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(frame.camera.width, x + width + pad), min(frame.camera.height, y + height + pad)
            if cfg.roi_xyxy is not None:
                rx0, ry0, rx1, ry1 = cfg.roi_xyxy
                x0, y0, x1, y1 = max(x0, rx0), max(y0, ry0), min(x1, rx1), min(y1, ry1)
            if self._ray_cache is None or self._ray_cache[0] != frame.camera.calibration_id:
                fy, fx = np.indices(frame.depth.shape)
                full_rays = frame.camera.rays(np.column_stack((fx.ravel(), fy.ravel()))).reshape(
                    *frame.depth.shape, 3
                )
                self._ray_cache = frame.camera.calibration_id, full_rays.astype(np.float32)
            local_depth = frame.depth[y0:y1, x0:x1]
            local_rays = self._ray_cache[1][y0:y1, x0:x1]
            kernel = geometry_kernel(cfg.geometry_backend)
            if kernel is not None:
                mask = kernel.dense_plane_mask(
                    local_depth,
                    local_rays,
                    plane.normal,
                    plane.d,
                    cfg.min_depth_m,
                    cfg.max_depth_m,
                    cfg.plane_threshold_m,
                )
            else:
                depth = local_depth.ravel()
                valid = np.isfinite(depth) & (depth >= cfg.min_depth_m) & (depth <= cfg.max_depth_m)
                residual = np.full(len(depth), np.inf)
                local_points = local_rays.reshape(-1, 3)[valid] * depth[valid, None]
                residual[valid] = np.abs(np.einsum("ni,i->n", local_points, plane.normal) + plane.d)
                mask = (residual.reshape(local_depth.shape) < cfg.plane_threshold_m).astype(np.uint8) * 255
            report["dense_inlier_pixels"] = int(np.count_nonzero(mask))
            chroma = cv2.cvtColor(frame.rgb[y0:y1, x0:x1], cv2.COLOR_RGB2HSV)[:, :, 1] > 55
            if np.count_nonzero((mask > 0) & chroma) > np.count_nonzero(mask) * 0.6:
                mask[~chroma] = 0
            # Remove thin wall/floor intersection bands before fitting a face outline.
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            # Keep separate connected patches instead of taking one convex hull across gaps.
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            for label in range(1, count):
                if stats[label, cv2.CC_STAT_AREA] < cfg.min_face_area_px:
                    continue
                component = np.zeros(frame.depth.shape, np.uint8)
                component[y0:y1, x0:x1] = (labels == label).astype(np.uint8) * 255
                masks.append((component, plane))
                report["patches"] += 1
                if debug is not None:
                    debug.images["plane_labels"][component > 0] = attempt + 1
            report["status"] = "kept" if report["patches"] else "no_patch_after_morphology"
        return masks

    def _plane_quad(self, frame, mask, plane):
        contour = max(
            cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea
        )
        points = ray_plane(frame.camera, contour.reshape(-1, 2), plane)
        origin = points.mean(axis=0)
        axis = np.array([0.0, -1.0, 0.0]) if self.cfg.up_camera is None else np.asarray(self.cfg.up_camera)
        axis = axis - plane.normal * (axis @ plane.normal)
        if np.linalg.norm(axis) < 0.1:
            axis = np.array([1.0, 0.0, 0.0]) - plane.normal * plane.normal[0]
        axis /= np.linalg.norm(axis)
        other = np.cross(plane.normal, axis)
        basis = np.column_stack((axis, other))
        coordinates = (points - origin) @ basis
        rectangle = cv2.minAreaRect(coordinates.astype(np.float32))
        corners = cv2.boxPoints(rectangle).astype(np.float64) @ basis.T + origin
        return order_quad(frame.camera.project(corners), backend=self.cfg.geometry_backend)

    def _color_line_quads(self, frame, hsv, rgb_edges, distance):
        """Observed line intersections survive a red/blue paint seam splitting contours.

        This is only a proposal: all original metric-size, depth, PnP and upright
        checks still run. No depth holes or occluded image corners are filled.
        """
        if not self.cfg.analyze_colors or self.cfg.up_camera is None:
            return []
        from itertools import combinations
        from .appearance import classify_hsv

        colored = (classify_hsv(hsv, self.cfg) != 0).astype(np.uint8)
        if self.cfg.roi_xyxy is not None:
            x0, y0, x1, y1 = self.cfg.roi_xyxy
            area = np.zeros_like(colored)
            area[y0:y1, x0:x1] = 1
            colored &= area
        colored = cv2.morphologyEx(colored, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(colored)
        regions = sorted(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
        proposals = []
        for index in regions[: self.cfg.max_objects * 3]:
            x, y, width, height, size = stats[index]
            if size < self.cfg.min_face_area_px or min(width, height) < self.cfg.min_edge_px:
                continue
            x0, y0 = max(0, x - 5), max(0, y - 5)
            x1, y1 = min(frame.camera.width, x + width + 5), min(frame.camera.height, y + height + 5)
            center_uv = np.array([x + width / 2, y + height / 2])
            center = frame.camera.rays([center_uv])[0]
            vertical = frame.camera.project([center + np.asarray(self.cfg.up_camera) * 0.1])[0] - center_uv
            if np.linalg.norm(vertical) < 1:
                continue
            vertical /= np.linalg.norm(vertical)
            minimum = max(12, min(30, int(min(width, height) * 0.28)))
            lines = cv2.HoughLinesP(
                rgb_edges[y0:y1, x0:x1],
                1,
                np.pi / 180,
                threshold=max(12, minimum // 2),
                minLineLength=minimum,
                maxLineGap=6,
            )
            if lines is None:
                continue
            families = [[], []]
            for line in sorted(lines[:, 0], key=lambda l: -np.linalg.norm(l[2:] - l[:2])):
                a, b = line[:2].astype(float) + (x0, y0), line[2:].astype(float) + (x0, y0)
                length = float(np.linalg.norm(b - a))
                tangent = (b - a) / length
                alignment = abs(tangent @ vertical)
                if alignment > 0.94:
                    family = families[0]
                elif alignment < 0.65:
                    family = families[1]
                else:
                    continue
                normal = np.array([-tangent[1], tangent[0]])
                if normal[int(np.argmax(np.abs(normal)))] < 0:
                    normal = -normal
                offset = -float(normal @ a)
                if len(family) >= 6 or any(
                    abs(normal @ n) > 0.996 and abs((normal - n) @ center_uv + offset - d) < 3
                    for n, d, _ in family
                ):
                    continue
                family.append((normal, offset, length))
            # Cache each line intersection once instead of solving it repeatedly
            # for every quadrilateral combination in a multi-level stack.
            intersections = {}
            for i, (n, d, _) in enumerate(families[0]):
                for j, (m, e, _) in enumerate(families[1]):
                    determinant = n[0] * m[1] - n[1] * m[0]
                    if abs(determinant) >= 0.3:
                        intersections[i, j] = np.array(
                            [(n[1] * e - d * m[1]) / determinant, (d * m[0] - n[0] * e) / determinant]
                        )
            for a, b in combinations(range(len(families[0])), 2):
                for c, d in combinations(range(len(families[1])), 2):
                    keys = ((a, c), (a, d), (b, c), (b, d))
                    if any(key not in intersections for key in keys):
                        continue
                    corners = np.array([intersections[key] for key in keys])
                    if np.any(corners < (x0, y0)) or np.any(corners > (x1 - 1, y1 - 1)):
                        continue
                    try:
                        quad = order_quad(corners, backend=self.cfg.geometry_backend)
                    except ValueError:
                        continue
                    lengths = np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)
                    area = cv2.contourArea(quad.astype(np.float32))
                    if (
                        area < self.cfg.min_face_area_px
                        or lengths.min() < self.cfg.min_edge_px
                        or lengths.max() / lengths.min() > 3
                    ):
                        continue
                    support, per_edge = _edge_support(quad, distance, backend=self.cfg.geometry_backend)
                    if support < max(self.cfg.stack_min_edge_support_fraction, self.cfg.min_edge_support):
                        continue
                    if proposals and np.any(
                        np.linalg.norm(quad - np.array([f.quad for f in proposals]), axis=2).mean(axis=1) < 4
                    ):
                        continue
                    proposals.append(
                        Face(
                            quad,
                            "rgb_line_intersections",
                            support,
                            evidence=dict(
                                edge_support_per_side=per_edge, corners="intersections of observed edge lines"
                            ),
                        )
                    )
                    if len(proposals) > self.cfg.max_faces * 2:
                        proposals.sort(
                            key=lambda f: (f.edge_support, cv2.contourArea(f.quad.astype(np.float32))),
                            reverse=True,
                        )
                        proposals = proposals[: self.cfg.max_faces]
        proposals.sort(
            key=lambda f: (f.edge_support, cv2.contourArea(f.quad.astype(np.float32))), reverse=True
        )
        return proposals[: self.cfg.max_faces]

    def color_line_fallback(self, frame, edges):
        hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
        rgb_edges, _, _ = rgb_depth_edges(frame, self.cfg, hsv=hsv)
        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        return self._color_line_quads(frame, hsv, rgb_edges, distance)

    def propose(self, frame, debug=None):
        """Build shared image evidence once, then extend proposals per detector."""
        started = time.perf_counter()
        self.proposal_rejections = []
        cfg = self.cfg
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV)
        rgb_edges, depth_edges, edges = rgb_depth_edges(frame, cfg, gray, hsv)
        color_labels = None
        if debug is not None:
            from .appearance import classify_hsv

            color_labels = classify_hsv(hsv, cfg)
            debug.images.update(
                rgb_edges=rgb_edges.copy(),
                depth_edges=depth_edges,
                color_labels=color_labels.copy(),
                plane_labels=np.zeros_like(edges),
            )
        if cfg.roi_xyxy is not None:
            x0, y0, x1, y1 = cfg.roi_xyxy
            # rgb_depth_edges 已在唯一的图像入口检查 ROI 越界并屏蔽边缘。
            roi = np.zeros_like(edges)
            roi[y0:y1, x0:x1] = 255
            if debug is not None:
                debug.images["color_labels"][roi == 0] = 0
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        masks = [(closed, "rgb_depth_edges")]
        # Saturation is an additional proposal source, not a required target color.
        saturation = (
            (hsv[:, :, 1] > cfg.color_proposal_min_saturation) & (hsv[:, :, 2] > cfg.color_proposal_min_value)
        ).astype(np.uint8) * 255
        if cfg.roi_xyxy is not None:
            saturation &= roi
        masks.append((saturation, "color_region"))
        if debug is not None:
            debug.images["saturation"] = saturation.copy()
        plane_quads = []
        plane_patches = []
        if cfg.plane_proposals:
            for mask, plane in self._depth_masks(frame, hsv, debug):
                plane_patches.append((mask, plane))
                masks.append((mask, "depth_plane"))
                try:
                    plane_quads.append(self._plane_quad(frame, mask, plane))
                except ValueError:
                    pass
        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        faces = []
        for quad in plane_quads:
            quad = _refine_quad(quad, edges, backend=cfg.geometry_backend)
            support, per_edge = _edge_support(quad, distance, backend=cfg.geometry_backend)
            faces.append(
                Face(
                    quad,
                    "rectified_depth_plane",
                    support,
                    evidence=dict(edge_support_per_side=per_edge, rectangle_fit_in="observed face plane"),
                )
            )
        for mask, source in masks:
            contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
            for contour in contours:
                if cv2.contourArea(contour) < cfg.min_face_area_px:
                    continue
                perimeter = cv2.arcLength(contour, True)
                for fraction in (0.012, 0.025, 0.04):
                    poly = cv2.approxPolyDP(contour, perimeter * fraction, True)
                    if len(poly) != 4 or not cv2.isContourConvex(poly):
                        continue
                    try:
                        q = order_quad(poly.reshape(4, 2), backend=cfg.geometry_backend)
                    except ValueError:
                        continue
                    lengths = np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)
                    if lengths.min() < cfg.min_edge_px or lengths.max() / lengths.min() > 8:
                        break
                    if any(np.linalg.norm(q - f.quad, axis=1).mean() < 5 for f in faces):
                        break
                    q = _refine_quad(q, edges, backend=cfg.geometry_backend)
                    support, per_edge = _edge_support(q, distance, backend=cfg.geometry_backend)
                    if support < cfg.min_edge_support and source != "depth_plane":
                        break
                    if any(np.linalg.norm(q - f.quad, axis=1).mean() < 5 for f in faces):
                        break
                    faces.append(Face(q, source, support, evidence=dict(edge_support_per_side=per_edge)))
                    break
        # Strong observed boundaries first; preserve a bounded number of hypotheses.
        faces.sort(key=lambda f: (f.edge_support, cv2.contourArea(f.quad.astype(np.float32))), reverse=True)
        faces = faces[: cfg.max_faces]
        # Reuse the measured patches; do not run a second scene-plane RANSAC.
        if cfg.upright_outline_proposals and cfg.up_camera is not None:
            from .partial import upright_patch_faces

            if cfg.analyze_colors and color_labels is None:
                from .appearance import classify_hsv

                color_labels = classify_hsv(hsv, cfg)
            for mask, plane in sorted(
                plane_patches, key=lambda item: cv2.countNonZero(item[0]), reverse=True
            )[: cfg.max_scene_planes]:
                for face in upright_patch_faces(frame, mask, plane, cfg, distance, color_labels):
                    if any(
                        f.source == face.source and np.linalg.norm(f.quad - face.quad, axis=1).mean() < 5
                        for f in faces
                    ):
                        continue
                    faces.append(face)
        faces = self._additional_faces(frame, faces, plane_patches, rgb_edges, distance, debug)
        return (
            faces,
            dict(proposal_count=len(faces), detector_ms=(time.perf_counter() - started) * 1000),
            edges,
        )

    def _additional_faces(self, frame, faces, plane_patches, rgb_edges, distance, debug):
        """ROI 子类复用本轮边缘、距离图和面片，避免再次处理整张图。"""
        return faces

    def split_stack(self, frame, face, points, plane):
        """Only infer two upright cubes when the whole 1:2 outer face supports it."""
        cfg = self.cfg
        if not cfg.allow_stack_inference or cfg.up_camera is None:
            return []
        lengths = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
        up = np.asarray(cfg.up_camera)
        vertical = np.abs((np.roll(points, -1, axis=0) - points) @ up) / lengths
        if not (vertical.max() > 0.96 and vertical.min() < 0.15):
            return []
        vertical_length = float(np.median(lengths[vertical > 0.96]))
        horizontal_length = float(np.median(lengths[vertical < 0.15]))
        matches = [
            s
            for s in cfg.sizes_m
            if abs(horizontal_length / s - 1) < cfg.size_tolerance_fraction
            and abs(vertical_length / (2 * s) - 1) < cfg.size_tolerance_fraction
        ]
        if len(matches) != 1 or face.edge_support < cfg.stack_min_edge_support_fraction:
            return []
        i = int(np.argmax(vertical))
        p = np.roll(points, -i, axis=0)
        middle0, middle1 = (p[0] + p[1]) / 2, (p[2] + p[3]) / 2
        halves = ([p[0], middle0, middle1, p[3]], [middle0, p[1], p[2], middle1])
        return [
            Face(
                order_quad(frame.camera.project(h), backend=cfg.geometry_backend),
                "stack_model",
                face.edge_support,
                True,
                dict(
                    size_m=matches[0],
                    inferred_from="observed full 1:2 outer face",
                    parent_quad=face.quad.tolist(),
                ),
                plane,
            )
            for h in halves
        ]

    def anchored_partial_face(self, frame, face, prepared):
        from .partial import anchored_partial_face

        return anchored_partial_face(frame, face, prepared, self.cfg)

    def join_color_halves(self, frame, prepared_faces):
        """Join measured red/blue half rectangles before attempting square PnP.

        One half is not a cube detection. Opposite colors, measured half-size,
        adjacency and a shared depth plane are all required. Scene selection
        still rejects overlapping stack-phase alternatives.
        """
        cfg = self.cfg
        if (
            not cfg.color_half_proposals
            or not cfg.analyze_colors
            or not cfg.two_color_sizes_m
            or cfg.up_camera is None
        ):
            return []
        from .appearance import classify_hsv

        labels = classify_hsv(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV), cfg)
        halves = []
        for face, prepared in prepared_faces:
            samples, fraction, plane, points, lengths = prepared
            short, long = float(np.min(lengths)), float(np.max(lengths))
            sizes = [
                s
                for s in cfg.two_color_sizes_m
                if abs(long / s - 1) <= cfg.size_tolerance_fraction
                and abs(short / (s / 2) - 1) <= cfg.size_tolerance_fraction
            ]
            if len(sizes) != 1 or plane.support < cfg.partial_min_plane_support_fraction:
                continue
            size = sizes[0]
            if any(np.linalg.norm(points.mean(axis=0) - p[3].mean(axis=0)) < size * 0.10 for p in halves):
                continue
            lo = np.maximum(np.floor(face.quad.min(axis=0)).astype(int), 0)
            hi = np.minimum(np.ceil(face.quad.max(axis=0)).astype(int) + 1, labels.shape[::-1])
            mask = np.zeros((hi[1] - lo[1], hi[0] - lo[0]), np.uint8)
            cv2.fillConvexPoly(mask, np.rint(face.quad - lo).astype(np.int32), 1)
            mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
            color = labels[lo[1] : hi[1], lo[0] : hi[0]][mask > 0]
            red = float(np.mean(color == 1)) if len(color) else 0.0
            blue = float(np.mean(color == -1)) if len(color) else 0.0
            if max(red, blue) < 0.85:
                continue
            halves.append((size, 1 if red > blue else -1, face, points, plane))
        joined = []
        for i, (size, color, face, points, plane) in enumerate(halves):
            for other_size, other_color, other, other_points, other_plane in halves[i + 1 :]:
                if size != other_size or color == other_color or plane.normal @ other_plane.normal < 0.985:
                    continue
                if np.median(np.abs(other_points @ plane.normal + plane.d)) > cfg.max_depth_residual_m:
                    continue
                delta = other_points.mean(axis=0) - points.mean(axis=0)
                if abs(np.linalg.norm(delta) / (size / 2) - 1) > cfg.size_tolerance_fraction * 2:
                    continue
                # The join crosses the short dimension. Pick the two farthest
                # corners along that axis from each actual observed rectangle.
                direction = delta / np.linalg.norm(delta)
                if abs(direction @ plane.normal) > 0.15:
                    continue
                a = np.argsort(points @ direction)[:2]
                b = np.argsort(other_points @ direction)[-2:]
                q = order_quad(np.concatenate((face.quad[a], other.quad[b])), backend=cfg.geometry_backend)
                if any(np.linalg.norm(q - f.quad, axis=1).mean() < 5 for f in joined):
                    continue
                joined.append(
                    Face(
                        q,
                        "paired_color_halves",
                        min(face.edge_support, other.edge_support),
                        False,
                        dict(
                            inferred_from="two observed opposite-color half-size rectangles",
                            half_quads=[face.quad.tolist(), other.quad.tolist()],
                            size_m=size,
                        ),
                    )
                )
        return joined
