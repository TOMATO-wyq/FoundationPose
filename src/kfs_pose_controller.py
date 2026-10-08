"""Geometry health checks and fail-closed SEARCH/REGISTER/TRACK/LOST control."""
from dataclasses import asdict
from pathlib import Path
import json
import time

import cv2
import numpy as np

from kfs_segmentation import NoTarget, mask_acceptable, save_segmentation, valid_depth


def render_mesh_depth(vertices, faces, pose, k, shape):
    """CPU z-buffer for health checking, in the original mesh object coordinates.

    KFS has 12 triangles; plane/ray intersections preserve perspective depth.
    This renderer is diagnostic only and does not replace FP's CUDA renderer.
    """
    h, w = shape
    xyz = np.asarray(vertices) @ pose[:3, :3].T + pose[:3, 3]
    depth = np.full((h, w), np.inf, np.float32)
    inverse_k = np.linalg.inv(k)
    for face in faces:
        tri = xyz[face]
        if np.any(tri[:, 2] <= .001):
            continue
        homogeneous = tri @ k.T
        uv = homogeneous[:, :2] / homogeneous[:, 2:]
        left, top = np.maximum(np.floor(uv.min(axis=0)), (0, 0)).astype(int)
        right, bottom = np.minimum(np.ceil(uv.max(axis=0)), (w-1, h-1)).astype(int)
        if right < left or bottom < top:
            continue
        polygon = np.zeros((bottom-top+1, right-left+1), np.uint8)
        cv2.fillConvexPoly(polygon, np.round(uv - (left, top)).astype(np.int32), 1)
        y, x = np.nonzero(polygon)
        rays = np.column_stack((x+left, y+top, np.ones(len(x)))) @ inverse_k.T
        normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
        numerator = np.dot(normal, tri[0])
        denominator = rays @ normal
        distance = np.full(len(x), np.inf)
        np.divide(numerator, denominator, out=distance, where=np.abs(denominator) > 1e-10)
        z = distance * rays[:, 2]
        z[z <= .001] = np.inf
        depth[y+top, x+left] = np.minimum(depth[y+top, x+left], z)
    depth[~np.isfinite(depth)] = 0
    return depth


def pose_health(pose, depth, k, vertices, faces, cfg):
    if (np.shape(pose) != (4, 4) or not np.isfinite(pose).all()
            or pose[2, 3] <= 0 or not np.allclose(pose[3], (0, 0, 0, 1), atol=1e-4)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-2)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-2)):
        return False, {"reason": "invalid_pose", "fatal_pose": True}
    rendered = render_mesh_depth(vertices, faces, pose, k, depth.shape)
    silhouette = cv2.erode((rendered > 0).astype(np.uint8), np.ones((3, 3), np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    total = int(silhouette.sum())
    valid = valid_depth(depth, cfg)
    occluded = silhouette & valid & (depth < rendered - cfg.occlusion_delta_m)
    eligible = silhouette & ~occluded
    measured = eligible & valid
    residual = np.abs(depth[measured]-rendered[measured])
    supported = measured & (np.abs(depth-rendered) <= cfg.health_residual_m)
    # The denominator includes occlusion: a tiny visible sliver cannot pass.
    support_fraction = int(supported.sum()) / max(total, 1)
    median = float(np.median(residual)) if len(residual) else None
    healthy = (int(supported.sum()) >= cfg.min_valid_depth_pixels
               and support_fraction >= cfg.health_support_fraction
               and median is not None and median <= cfg.health_residual_m)
    return healthy, {"reason": "ok" if healthy else "depth_support_failed",
                     "fatal_pose": False, "projected_pixels": total,
                     "supported_pixels": int(supported.sum()),
                     "support_fraction": support_fraction,
                     "occluded_fraction": int(occluded.sum()) / max(total, 1),
                     "residual_median_m": median}


class PoseController:
    def __init__(self, estimator, segmenter, mesh, output, register_iterations=5,
                 track_iterations=2, search_interval=None, debug=False):
        self.debug = debug
        self.estimator, self.segmenter, self.cfg = estimator, segmenter, segmenter.cfg
        self.vertices = np.asarray(mesh.vertices)
        self.faces = np.asarray(mesh.faces)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.register_iterations, self.track_iterations = register_iterations, track_iterations
        self.search_interval = self.cfg.search_interval_s if search_interval is None else search_interval
        self.state, self.failures, self.attempts, self.next_search = "SEARCH", 0, 0, 0.
        self.last_health = {}
        self.next_health = 0.
        (self.output / "segmentation_config.json").write_text(json.dumps({
            "config": asdict(self.cfg), "sam": getattr(segmenter.sam, "metadata", {}),
            "prompt_mode": segmenter.prompt_mode, "depth_refinement": segmenter.refinement,
            "health_type": "geometry_proxy_not_FP_confidence"}, indent=2))

    def reset(self):
        self.next_health = 0.
        self.state, self.failures, self.next_search = "SEARCH", 0, 0.
        self.estimator.pose_last = None

    def _lost(self, now):
        self.state, self.failures = "LOST", 0
        self.next_search = now + self.search_interval
        self.estimator.pose_last = None

    def process(self, rgb, depth, k, frame_key, now=None):
        now = time.monotonic() if now is None else now
        timings, health, pose = {}, {}, None
        mode = "search"
        reason = "waiting_for_target"
        try:
            if self.state in ("SEARCH", "LOST"):
                if now < self.next_search:
                    reason = "search_cooldown"
                else:
                    self.attempts += 1
                    self.next_search = now + self.search_interval
                    start = time.perf_counter()
                    result = self.segmenter.segment(rgb, depth)
                    timings.update(result.timings)
                    timings["segmentation_total_s"] = time.perf_counter()-start
                    save_segmentation(result, rgb, self.output / "segmentation" / f"{self.attempts:06d}_{frame_key}")
                    if self.debug:
                        from kfs_debug import show_segmentation
                        start = time.perf_counter()
                        show_segmentation(result, rgb, frame_key, self.segmenter.refinement)
                        timings['debug_display_s'] = time.perf_counter()-start
                    if not mask_acceptable(result.mask, depth, self.cfg):
                        raise NoTarget("Insufficient mask / valid depth")
                    self.state = "REGISTER"
                    mode = "register"
                    start = time.perf_counter()
                    candidate_pose = self.estimator.register(K=k, rgb=rgb, depth=depth,
                        ob_mask=result.mask, iteration=self.register_iterations)
                    timings["fp_register_s"] = time.perf_counter()-start
                    start = time.perf_counter()
                    healthy, health = pose_health(candidate_pose, depth, k, self.vertices, self.faces, self.cfg)
                    timings["health_s"] = time.perf_counter()-start
                    if healthy and self.estimator.pose_last is not None:
                        self.next_health = now + self.cfg.health_interval_s
                        pose, self.state, self.failures = candidate_pose, "TRACK", 0
                        reason = "registered"
                    else:
                        reason = "register_geometry_failed"
                        self._lost(now)
            else:
                mode = "tracking"
                start = time.perf_counter()
                candidate_pose = self.estimator.track_one(rgb=rgb, depth=depth, K=k,
                                                          iteration=self.track_iterations)
                timings["fp_track_s"] = time.perf_counter()-start
                start = time.perf_counter()
                # Always reject invalid transforms; expensive geometry is periodic.
                invalid = (np.shape(candidate_pose) != (4, 4) or not np.isfinite(candidate_pose).all()
                           or candidate_pose[2, 3] <= 0
                           or not np.allclose(candidate_pose[3], (0, 0, 0, 1), atol=1e-4)
                           or not np.allclose(candidate_pose[:3, :3].T @ candidate_pose[:3, :3], np.eye(3), atol=1e-2)
                           or not np.isclose(np.linalg.det(candidate_pose[:3, :3]), 1, atol=1e-2))
                if invalid or now >= self.next_health:
                    healthy, health = pose_health(candidate_pose, depth, k, self.vertices, self.faces, self.cfg)
                    self.next_health = now + self.cfg.health_interval_s
                else:
                    healthy, health = True, {'reason': 'geometry_check_skipped', 'checked': False}
                timings["health_s"] = time.perf_counter()-start
                if healthy:
                    pose, reason = candidate_pose, "tracked"
                    if health.get('reason') != 'geometry_check_skipped':
                        self.failures = 0
                else:
                    self.failures += 1
                    reason = "tracking_geometry_failed"
                    if health.get("fatal_pose") or self.failures >= self.cfg.failure_frames:
                        self._lost(now)
        except NoTarget as exc:
            reason = str(exc)
        except RuntimeError as exc:
            # Resource/environment failures are fatal, not silently retried forever.
            if any(word in str(exc).lower() for word in ("cuda", "out of memory", "install", "checkpoint")):
                raise
            reason = f"inference_error: {exc}"
            self._lost(now)
        self.last_health = health
        record = {"frame": str(frame_key), "mode": mode, "state": self.state,
                  "valid_pose": pose is not None, "reason": reason, "health": health,
                  "timings": timings}
        with (self.output / "events.jsonl").open("a") as file:
            file.write(json.dumps(record, allow_nan=False) + "\n")
        return pose, record
