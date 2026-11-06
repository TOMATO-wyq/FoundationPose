"""Measured planes and robust boundary lines inside a colored YOLO instance.

Segmentation selects plane seeds only. Validation always sees original depth,
including foreground that can explain a hidden bottom edge.
"""

import cv2
import numpy as np

from .detector import Detector, Face, _edge_support
from .geometry import order_quad
from .io import Frame
from .native import geometry_kernel


def fit_boundary_lines(quad, rgb_edges, cfg):
    """Intersect four robust observed edge lines; never use an ROI rectangle."""
    native = geometry_kernel(cfg.geometry_backend)
    if native is not None:
        return native.fit_boundary_lines(
            quad,
            rgb_edges,
            cfg.yolo_line_band_px,
            cfg.min_edge_px,
            cfg.yolo_line_min_points,
            cfg.yolo_line_min_span_fraction,
        )
    band = cfg.yolo_line_band_px
    lo = np.maximum(np.floor(quad.min(axis=0) - band - 2).astype(int), 0)
    hi = np.minimum(np.ceil(quad.max(axis=0) + band + 3).astype(int), rgb_edges.shape[::-1])
    yy, xx = np.nonzero(rgb_edges[lo[1] : hi[1], lo[0] : hi[0]])
    points = np.column_stack((xx + lo[0], yy + lo[1])).astype(float)
    lines, reports = [], []
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        tangent = b - a
        length = np.linalg.norm(tangent)
        if length < cfg.min_edge_px:
            return None
        tangent /= length
        normal = np.array([-tangent[1], tangent[0]])
        along = (points - a) @ tangent
        selected = points[
            (np.abs((points - a) @ normal) <= band) & (along > 0.08 * length) & (along < 0.92 * length)
        ]
        if len(selected) < cfg.yolo_line_min_points:
            return None
        fitted = cv2.fitLine(selected.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel().astype(float)
        direction, origin = fitted[:2], fitted[2:]
        if abs(direction @ tangent) < np.cos(np.deg2rad(12)):
            return None
        n = np.array([-direction[1], direction[0]])
        errors = np.abs((selected - origin) @ n)
        selected = selected[errors <= min(band, 2.0)]
        if len(selected) < cfg.yolo_line_min_points:
            return None
        fitted = cv2.fitLine(selected.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel().astype(float)
        direction, origin = fitted[:2], fitted[2:]
        n = np.array([-direction[1], direction[0]])
        positions = (selected - origin) @ direction
        span = float(np.ptp(positions) / length)
        if span < cfg.yolo_line_min_span_fraction:
            return None
        lines.append((n, -float(n @ origin)))
        reports.append(
            dict(
                segment=np.array(
                    [origin + direction * positions.min(), origin + direction * positions.max()]
                ).tolist(),
                support_pixels=len(selected),
                span_fraction=span,
                rms_px=float(np.sqrt(np.mean(((selected - origin) @ n) ** 2))),
            )
        )
    corners = []
    for i in range(4):
        n0, d0 = lines[i - 1]
        n1, d1 = lines[i]
        matrix = np.stack((n0, n1))
        if abs(np.linalg.det(matrix)) < 0.1:
            return None
        corners.append(np.linalg.solve(matrix, -np.array([d0, d1])))
    corners = np.asarray(corners)
    if np.max(np.linalg.norm(corners - quad, axis=1)) > 2 * band:
        return None
    try:
        return order_quad(corners, backend=cfg.geometry_backend), reports
    except ValueError:
        return None


class RoiFaceDetector(Detector):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.instance_mask = None

    def _depth_masks(self, frame, hsv=None, debug=None):
        mask = self.instance_mask.astype(np.uint8)
        margin = self.cfg.yolo_geometry_margin_px
        if margin:
            mask = cv2.dilate(mask, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
        depth = frame.depth.copy()
        depth[mask == 0] = 0
        selected = Frame(
            frame.rgb, depth, frame.camera, frame.stamp_ns, frame.sequence, frame.received_monotonic_ns
        )
        return super()._depth_masks(selected, hsv, debug)

    def _additional_faces(self, frame, faces, plane_patches, rgb_edges, distance, debug):
        fitted = []
        for mask, plane in plane_patches[: self.cfg.max_scene_planes]:
            try:
                outline = self._plane_quad(frame, mask, plane)
            except ValueError:
                continue
            fit = fit_boundary_lines(outline, rgb_edges, self.cfg)
            if fit is None:
                continue
            corners, lines = fit
            support, per_edge = _edge_support(corners, distance, backend=self.cfg.geometry_backend)
            if support < self.cfg.min_edge_support:
                continue
            fitted.append(
                Face(
                    corners,
                    "roi_plane_rgb_lines",
                    support,
                    False,
                    dict(
                        observed_lines=lines,
                        edge_support_per_side=per_edge,
                        corners="intersections of robust measured RGB boundary lines",
                        observed_corner_indices=[0, 1, 2, 3],
                    ),
                    plane,
                )
            )
        anchored = []
        for face in fitted + faces:
            if face.edge_support < self.cfg.min_edge_support:
                self.proposal_rejections.append(
                    dict(
                        quad=face.quad.tolist(),
                        source=face.source,
                        reason="ROI plane outline lacks actual image boundary support",
                    )
                )
                if debug is not None:
                    debug.face(face)
            else:
                anchored.append(face)
        return anchored[: self.cfg.max_faces]

    def anchored_partial_face(self, frame, face, prepared):
        from .partial import anchored_partial_face

        return anchored_partial_face(
            frame,
            face,
            prepared,
            self.cfg,
            max_visible_fraction=self.cfg.yolo_partial_max_visible_fraction,
            min_foreground_fraction=self.cfg.yolo_partial_min_foreground_fraction,
        )
