"""CPU feature tracking with current RGB-D validation and periodic redetection.

Every recovered pose is measured from forward/backward LK correspondences,
planar PnP and this frame's raw depth. No missing pose is held or extrapolated.
"""

from copy import deepcopy
import threading
import time

import cv2
import numpy as np

from .appearance import analyze_colors, classify_hsv
from .detector import _edge_support, sample_face
from .direction import copy_direction_window, select_direction_candidate
from .diagnostics import detector_snapshot, has_current_rejections
from .edges import rgb_depth_edges
from .errors import TrackingPoseRejected
from .joint_tracking import solve_joint
from .geometry import (
    Plane,
    angle_difference,
    cube_vertices,
    fit_plane_configured,
    order_quad,
    ray_plane,
    rigid_fit,
    square_points,
    transform,
    transform_points,
    upright_from_camera,
    yaw_from_pose,
)
from .pose import Pipeline
from .scene import select_scene
from .results import append_detection_checks, empty_result, refresh_detection_selection
from .yolo import make_detection_pipeline


class TemporalPipeline:
    """主线程只推进当前帧轨迹，后台线程执行完整检测。

    worker_condition 保护单槽任务/结果；config_lock 只保护 UI 配置快照。
    后台返回的是源帧观测，必须经过当前帧 RGB-D 验收后才能发布。
    """

    def __init__(self, cfg, live=False, debug=False):
        self.cfg = cfg.validate()
        self.debug_enabled = debug
        self.last_debug = None
        self.local_debug = ()
        self.debug_tracks = ()
        self.detector = make_detection_pipeline(cfg, debug=debug)
        self.up = upright_from_camera(cfg.up_camera)
        self.states = []
        self.last_detection = None
        self.last_stamp = None
        self.reference = None
        self.region_pipelines = {}
        self.asynchronous = bool(live and cfg.async_detection)
        self.worker_condition = threading.Condition()
        self.worker_pending = None
        self.worker_completed = None
        self.worker_error = None
        self.worker_stopping = False
        self.worker_busy = False
        self.worker_pending_replaced = 0
        self.worker_results_replaced = 0
        self.worker = None
        self.config_lock = threading.RLock()
        if self.asynchronous:
            self.worker = threading.Thread(target=self._detector_loop, daemon=True)
            self.worker.start()

    def _full_detect(self, frame, covered):
        if self.cfg.yolo_weights:
            # Copy tuning atomically, then release the UI/configuration lock
            # before inference. A slider edit never waits for a whole ROI pass.
            with self.config_lock:
                cfg = deepcopy(self.cfg)
            result, edges = self.detector.process(frame, config=cfg)
            return (
                result,
                edges,
                self.detector.region_reports,
                (self.detector.last_debug, self.detector.local_debug),
            )
        with self.config_lock:
            result, edges = self.detector.process(frame)
            self._local_debug_pending = []
            local, reports = self._search_uncovered_colors(frame, result["objects"] + covered)
            for index, report in enumerate(reports):
                accepted = [obj for obj in local if obj.get("_local_trace_index") == index]
                for obj in accepted:
                    obj.pop("_local_trace_index")
                result["objects"].extend(append_detection_checks(
                    result, report, source=dict(branch="local_search", roi=report["roi"]), objects=accepted
                ))
                for name in ("detector_ms", "solve_ms", "refinement_ms", "appearance_ms"):
                    result["timing"][name] += report["timing"][name]
                for name in ("rejected_faces", "appearance_rejections", "suppressed_hypotheses"):
                    result[name].extend(
                        dict(item, local_search_roi=report["roi"]) for item in report.get(name, [])
                    )
            result["valid_object_count"] = len(result["objects"])
            refresh_detection_selection(result)
            result["frame_status"] = (
                "observed"
                if result["objects"]
                else (
                    "partial_or_rejected"
                    if any(
                        result[name]
                        for name in ("rejected_faces", "appearance_rejections", "suppressed_hypotheses")
                    )
                    else "no_target"
                )
            )
            return result, edges, reports, (self.detector.last_debug, tuple(self._local_debug_pending))

    def _detector_loop(self):
        while True:
            with self.worker_condition:
                self.worker_condition.wait_for(
                    lambda: self.worker_pending is not None or self.worker_stopping
                )
                if self.worker_stopping:
                    return
                frame, covered = self.worker_pending
                self.worker_pending = None
                self.worker_busy = True
            try:
                begin = time.perf_counter()
                result, _, reports, debug = self._full_detect(frame, covered)
                with self.worker_condition:
                    if self.worker_completed is not None:
                        self.worker_results_replaced += 1
                    self.worker_completed = (
                        frame,
                        result,
                        reports,
                        (time.perf_counter() - begin) * 1000,
                        debug,
                    )
                    self.worker_busy = False
                    self.worker_condition.notify_all()
            except Exception as error:
                with self.worker_condition:
                    self.worker_busy = False
                    self.worker_error = error
                    self.worker_condition.notify_all()
                return

    def close(self):
        if self.worker is not None:
            with self.worker_condition:
                self.worker_stopping = True
                self.worker_pending = None
                self.worker_condition.notify_all()
            self.worker.join(timeout=3.0)

    def _empty_result(self, frame):
        return empty_result(frame, self.cfg, self.up)

    def _joint_pose(self, frame, state, model, uv, samples, plane, diagnostics=None):
        rv, tv, report = solve_joint(frame, state, model, uv, samples, plane, self.cfg)
        if diagnostics is not None:
            diagnostics.update(report)
        return rv, tv

    def _search_uncovered_colors(self, frame, covered):
        """Local acquisition gives small colored targets their own plane budget."""
        if not self.cfg.analyze_colors:
            return [], []
        labels = classify_hsv(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV), self.cfg)
        mask = cv2.morphologyEx((labels != 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        # Touching targets and a same-color support can form one component.
        # Remove only measured target footprints; skipping the entire connected
        # component would prevent acquisition of its still-untracked neighbor.
        for obj in covered:
            xyz = transform_points(np.asarray(obj["T_camera_from_box"]), cube_vertices(obj["size_m"]))
            if np.any(xyz[:, 2] <= 0):
                continue
            uv = frame.camera.project(xyz)
            hull = cv2.convexHull(np.rint(uv).astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 0)
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask)
        found = []
        reports = []
        regions = sorted(range(1, count), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
        for i in regions:
            if len(reports) >= 2:
                break
            x, y, w, h, area = map(int, stats[i])
            if area < self.cfg.min_face_area_px or min(w, h) < self.cfg.min_edge_px:
                continue
            depths = frame.depth[y : y + h, x : x + w][components[y : y + h, x : x + w] == i]
            depths = depths[
                np.isfinite(depths) & (depths >= self.cfg.min_depth_m) & (depths <= self.cfg.max_depth_m)
            ]
            if len(depths) < self.cfg.min_depth_samples:
                continue
            z = float(np.median(depths))
            extent = np.array([w / frame.camera.k[0, 0], h / frame.camera.k[1, 1]]) * z
            if (
                max(extent) < min(self.cfg.sizes_m) * 0.70
                or min(extent) < min(self.cfg.sizes_m) * 0.30
                or max(extent) > max(self.cfg.sizes_m) * 2.5
            ):
                continue
            q = (
                max(0, (x - 20) // 16 * 16),
                max(0, (y - 20) // 16 * 16),
                min(frame.camera.width, ((x + w + 35) // 16) * 16),
                min(frame.camera.height, ((y + h + 35) // 16) * 16),
            )
            if self.cfg.roi_xyxy is not None:
                a, b, c, d = self.cfg.roi_xyxy
                q = (max(q[0], a), max(q[1], b), min(q[2], c), min(q[3], d))
            if q[2] - q[0] < 8 or q[3] - q[1] < 8:
                continue
            if q not in self.region_pipelines:
                cfg = deepcopy(self.cfg)
                cfg.roi_xyxy = list(q)
                cfg.temporal_recovery = False
                if len(self.region_pipelines) >= 6:
                    self.region_pipelines.pop(next(iter(self.region_pipelines)))
                self.region_pipelines[q] = Pipeline(cfg, configure_opencv=False, debug=self.debug_enabled)
            # Calibration rays are independent of ROI. Reuse the full detector's
            # array instead of undistorting 307200 pixels for every new window.
            self.region_pipelines[q].detector._ray_cache = self.detector.detector._ray_cache
            result, _ = self.region_pipelines[q].process(frame)
            if self.debug_enabled:
                self._local_debug_pending.append(self.region_pipelines[q].last_debug)
            reports.append(
                dict(
                    roi=list(q),
                    objects=len(result["objects"]),
                    timing=result["timing"],
                    frame_status=result["frame_status"],
                    **{
                        name: result[name]
                        for name in ("rejected_faces", "appearance_rejections", "suppressed_hypotheses")
                    },
                    detection_checks=result.get("detection_checks", []),
                )
            )
            for obj in result["objects"]:
                if not any(
                    obj["size_m"] == o["size_m"]
                    and np.linalg.norm(np.asarray(obj["center_camera_m"]) - o["center_camera_m"]) < 0.04
                    for o in covered + found
                ):
                    obj["_local_trace_index"] = len(reports) - 1
                    found.append(obj)
        return found, reports

    def _tracking_depth_quad(self, quad):
        # A model projection can extend a few pixels beyond the observed cloth
        # boundary. Sample its interior, retaining full-quad RGB/edge checks.
        center = quad.mean(axis=0)
        return center + (quad - center) * (1 - 2 * self.cfg.track_depth_inset_fraction)

    def _sample_tracking_plane(self, frame, quad, diagnostics=None):
        cfg = self.cfg
        details = {}
        try:
            samples, fraction = sample_face(frame, quad, cfg, details)
            plane = fit_plane_configured(samples, cfg)
            details.update(full_face_plane_support=plane.support, depth_sampling_inset_fraction=0.0)
            if (
                plane.support < cfg.plane_min_support_fraction or plane.rmse_m > cfg.plane_max_rmse_m
            ) and cfg.track_depth_inset_fraction > 0:
                interior = {}
                try:
                    inner, inner_fraction = sample_face(frame, self._tracking_depth_quad(quad), cfg, interior)
                    inner_plane = fit_plane_configured(inner, cfg)
                    if (
                        inner_plane.support >= cfg.plane_min_support_fraction
                        and inner_plane.rmse_m <= cfg.plane_max_rmse_m
                    ):
                        samples, fraction, plane = inner, inner_fraction, inner_plane
                        details.update(interior, depth_sampling_inset_fraction=cfg.track_depth_inset_fraction)
                except (ValueError, cv2.error):
                    pass
            return samples, fraction, plane
        finally:
            if diagnostics is not None:
                diagnostics.update({k: v for k, v in details.items() if k != "sample_uv"})

    @staticmethod
    def _new_pose_trial(diagnostics, solver, candidate_index=None):
        trials = diagnostics.setdefault("pose_trials", [])
        trial = dict(
            solver=solver, candidate_index=candidate_index, trial_index=len(trials),
            stage="parameters", accepted=False, selected=False, failed_checks=[],
        )
        trials.append(trial)
        return trial

    @staticmethod
    def _fail_pose_trial(trial, error):
        trial.update(accepted=False, selected=False, reason=str(error))
        if isinstance(error, TrackingPoseRejected):
            trial.update(stage=error.stage, failed_checks=error.checks)
        else:
            trial["failed_checks"] = ["invalid_pose_solution"]

    def _evaluate_tracking_pose(self, frame, state, pose, model, uv, samples, plane, observed, distance, trial):
        """跟踪、联合解和补点纠偏共用验收，拒绝原因与 trial 一一对应。"""
        cfg = self.cfg
        trial["stage"] = "pose_geometry"
        if not np.isfinite(pose).all():
            raise TrackingPoseRejected("pose_geometry", "nonfinite_pose")
        trial.update(
            rotation_vector_camera=cv2.Rodrigues(pose[:3, :3])[0].ravel().tolist(),
            center_camera_m=pose[:3, 3].tolist(),
        )
        if pose[2, 3] <= 0:
            raise TrackingPoseRejected("pose_geometry", "behind_camera")
        normal = pose[:3, :3] @ state["normal_model"]
        face_center = transform_points(pose, state["quad_model"]).mean(axis=0)
        trial["normal_camera"] = normal.tolist()
        if normal @ face_center >= 0:
            raise TrackingPoseRejected("pose_geometry", "back_facing")
        ray = pose[:3, 3] / np.linalg.norm(pose[:3, 3])
        denominator = float(normal @ ray)
        if abs(denominator) < 0.2:
            raise TrackingPoseRejected("pose_geometry", "grazing_geometry")
        # 普通图像 PnP 仍以当前深度校正平移。联合解已经同时优化位置和
        # 朝向，不再叠加另一个中位数目标把优化结果移走。
        if trial["solver"] not in ("joint", "measured_plane_reseed", "current_pose_reseed"):
            pose[:3, 3] += ray * (float(np.median((samples - face_center) @ normal)) / denominator)
        face_center = transform_points(pose, state["quad_model"]).mean(axis=0)
        trial["stage"] = "current_pose_residuals"
        reprojection = float(np.sqrt(np.mean(np.sum(
            (frame.camera.project(transform_points(pose, model)) - uv) ** 2, axis=1
        ))))
        depth = float(np.median(np.abs((samples - face_center) @ normal)))
        angle = float(np.rad2deg(np.arccos(np.clip(normal @ plane.normal, -1, 1))))
        if not np.isfinite([reprojection, depth, angle]).all():
            raise TrackingPoseRejected("current_pose_residuals", "nonfinite_pose_residuals")
        rotation_step_cos = (np.trace(pose[:3, :3] @ state["pose"][:3, :3].T) - 1) / 2
        normal_step_cos = normal @ (state["pose"][:3, :3] @ state["normal_model"])
        limit = cfg.reprojection_limit(frame.camera)
        trial.update(
            center_camera_m=pose[:3, 3].tolist(),
            rotation_step_deg=float(np.rad2deg(np.arccos(np.clip(rotation_step_cos, -1, 1)))),
            normal_step_deg=float(np.rad2deg(np.arccos(np.clip(normal_step_cos, -1, 1)))),
            reprojection_rms_px=reprojection, reprojection_limit_px=limit,
            depth_median_residual_m=depth, depth_limit_m=cfg.max_depth_residual_m,
            normal_error_deg=angle, normal_limit_deg=cfg.max_normal_error_deg,
        )
        failed = [name for name, bad in (
            ("feature_reprojection", reprojection > limit),
            ("current_depth_residual", depth > cfg.max_depth_residual_m),
            ("current_plane_normal", angle > cfg.max_normal_error_deg),
        ) if bad]
        if failed:
            raise TrackingPoseRejected("current_pose_residuals", *failed)
        if np.linalg.norm(pose[:3, 3] - state["pose"][:3, 3]) > cfg.track_max_distance_m:
            raise TrackingPoseRejected("motion_validation", "position_jump")
        if cfg.up_camera is not None and pose[:3, 2] @ cfg.up_camera < np.cos(np.deg2rad(cfg.max_upright_tilt_deg)):
            raise TrackingPoseRejected("motion_validation", "upright_tilt")
        trial["stage"] = "boundary_validation"
        quad = frame.camera.project(transform_points(pose, state["quad_model"]))
        drift = float(np.linalg.norm(quad - observed, axis=1).mean())
        if not np.isfinite(quad).all() or not np.isfinite(drift):
            raise TrackingPoseRejected("boundary_validation", "nonfinite_projected_face")
        trial.update(quad_drift_px=drift, quad_drift_limit_px=cfg.track_quad_max_drift_px)
        if drift > cfg.track_quad_max_drift_px:
            raise TrackingPoseRejected("boundary_validation", "quad_drift")
        support, per_edge = _edge_support(quad, distance, backend=cfg.geometry_backend)
        trial.update(edge_support=support, edge_support_min=cfg.min_edge_support,
                     edge_support_per_side=per_edge, min_supported_edges=2, per_edge_min=0.4)
        if support < cfg.min_edge_support or sum(v > 0.4 for v in per_edge) < 2:
            raise TrackingPoseRejected("boundary_validation", "current_boundary_support")
        yaw = yaw_from_pose(pose, self.up)
        previous = yaw_from_pose(state["pose"], self.up)
        prior = 0.0 if yaw is None else abs(angle_difference(yaw, previous))
        terms = dict(image=reprojection / limit, depth=depth / cfg.max_depth_residual_m,
                     normal=angle / cfg.max_normal_error_deg, yaw=prior * 0.2)
        cost = sum(terms.values())
        trial.update(stage="accepted", accepted=True, base_cost=cost, selection_cost=cost,
                     cost_components=terms, yaw_prior_rad=prior)
        return dict(base_cost=cost, pose=pose, quad=quad, normal=normal, yaw=yaw,
                    reprojection=reprojection, depth=depth, angle=angle, support=support, trial=trial)

    def _seed(self, frame, obj, gray, *, previous_state=None, distance=None, diagnostics=None):
        size = obj["size_m"]
        old_pose = np.asarray(obj["T_camera_from_box"])
        diagnostics = {} if diagnostics is None else diagnostics
        vertices = cube_vertices(size)
        choices = sorted(
            obj["faces"], key=lambda f: cv2.contourArea(np.asarray(f["quad"], np.float32)), reverse=True
        )
        for face in choices:
            trial = None
            seed_window = copy_direction_window(
                None if previous_state is None else previous_state.get("direction_window")
            )
            try:
                observed = np.asarray(face["quad"], float)
                sample_details = {}
                samples, depth_fraction, plane = self._sample_tracking_plane(frame, observed, sample_details)
                model_plane = plane
                measured_plane_normal = plane.normal.copy()
                diagnostics["measured_plane_normal_camera"] = measured_plane_normal.tolist()
                physical_face = rigid_fit(
                    square_points(size),
                    ray_plane(frame.camera, observed, plane),
                    backend=self.cfg.geometry_backend,
                )
                direction = old_pose[:3, :3].T @ np.asarray(face["normal_camera"])
                axis = int(np.argmax(np.abs(direction)))
                sign = 1 if direction[axis] > 0 else -1
                local = vertices[np.isclose(vertices[:, axis], sign * size / 2)]
                projected = frame.camera.project(transform_points(old_pose, local))
                q = order_quad(projected, backend=self.cfg.geometry_backend)
                ids = [int(np.argmin(np.linalg.norm(projected - point, axis=1))) for point in q]
                local = local[ids]
                local_face = rigid_fit(square_points(size), local, backend=self.cfg.geometry_backend)
                expected = old_pose[:3, 3] + np.asarray(face["normal_camera"]) * size / 2
                if (
                    plane.support < self.cfg.plane_min_support_fraction
                    or plane.rmse_m > self.cfg.plane_max_rmse_m
                    or abs(plane.normal @ expected + plane.d) > self.cfg.max_depth_residual_m
                ):
                    raise TrackingPoseRejected("seed_plane", "seed_plane_inconsistent")
                normal = np.zeros(3)
                normal[axis] = sign
                seed_obj = deepcopy(obj)
                if obj.get("source") == "current_rgbd_flow":
                    rotation = physical_face[:3, :3] @ local_face[:3, :3].T
                    center = physical_face[:3, 3] - rotation @ local.mean(axis=0)
                    pose = transform(rotation, center)
                    trial = self._new_pose_trial(diagnostics, "measured_plane_reseed")
                    rotation_deg = float(np.rad2deg(np.arccos(np.clip(
                        (np.trace(rotation @ old_pose[:3, :3].T) - 1) / 2, -1, 1
                    ))))
                    trial["reseed_rotation_deg"] = rotation_deg
                    if rotation_deg > self.cfg.track_reseed_max_rotation_deg:
                        error = TrackingPoseRejected("seed_rotation", "reseed_rotation_jump")
                        self._fail_pose_trial(trial, error)
                        raise error
                    # 先用已有的对应点验收纠偏，不能先按新姿态造出一套
                    # 自洽模型，再用它自己证明自己正确。
                    check_model = local if previous_state is None else previous_state["model"]
                    check_uv = observed if previous_state is None else previous_state["points"].reshape(-1, 2)
                    prior_pose = None if previous_state is None else previous_state.get("previous_pose")
                    if prior_pose is None:
                        prior_pose = old_pose
                    check_state = dict(pose=prior_pose, normal_model=normal, quad_model=local)
                    if distance is None:
                        edges = rgb_depth_edges(frame, self.cfg, gray=gray)[2]
                        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
                    inliers = samples[np.abs(samples @ plane.normal + plane.d) < self.cfg.plane_threshold_m]
                    options = []
                    try:
                        measured = self._evaluate_tracking_pose(
                            frame, check_state, pose, check_model, check_uv, inliers, plane,
                            observed, distance, trial,
                        )
                    except (ValueError, cv2.error) as error:
                        self._fail_pose_trial(trial, error)
                    else:
                        measured["trial"]["reseed_observation_cost"] = measured["base_cost"]
                        measured["base_cost"] += self.cfg.track_direction_escape_margin
                        measured["trial"]["base_cost"] = measured["base_cost"]
                        measured["trial"]["cost_components"]["reseed_evidence_margin"] = self.cfg.track_direction_escape_margin
                        options.append(measured)
                    # 实测平面提供纠偏候选，而不是每次补点都强制接管。
                    # 对同一批当前观测比较原姿态；微小改善不足以反复改朝向。
                    current_trial = self._new_pose_trial(diagnostics, "current_pose_reseed")
                    try:
                        current = self._evaluate_tracking_pose(
                            frame, check_state, old_pose.copy(), check_model, check_uv,
                            inliers, plane, observed, distance, current_trial,
                        )
                        options.insert(0, current)
                    except (ValueError, cv2.error) as error:
                        self._fail_pose_trial(current_trial, error)
                    if not options:
                        raise TrackingPoseRejected("seed_candidates", "no_valid_reseed_candidate")
                    dt = (frame.stamp_ns - previous_state["previous_stamp_ns"]) / 1e9 if (
                        previous_state is not None and "previous_stamp_ns" in previous_state
                    ) else 1.0 / 60.0
                    index, selected_window, direction = select_direction_candidate(
                        options, prior_pose[:3, :3],
                        None if previous_state is None else previous_state.get("direction_window_before"), dt,
                        prior_per_deg=self.cfg.track_direction_prior_per_deg,
                        tau_s=self.cfg.track_direction_window_tau_s,
                        escape_margin=self.cfg.track_direction_escape_margin,
                        escape_frames=self.cfg.track_direction_escape_frames,
                    )
                    for option, terms in zip(options, direction.pop("candidates")):
                        option["trial"].update(terms)
                        option["trial"]["cost_components"]["direction"] = terms["applied_direction_cost"]
                    diagnostics["direction"] = direction
                    measured = options[index]
                    trial = measured["trial"]
                    source = "measured_plane" if trial["solver"] == "measured_plane_reseed" else "current_pose"
                    if (len(options) > 1 or source == "measured_plane") and (
                        previous_state is not None and "direction_window_before" in previous_state
                    ):
                        # 从本帧之前的窗口重算并替换，而不是把同一帧更新两遍。
                        seed_window = selected_window
                    trial["reseed_rotation_deg"] = rotation_deg if source == "measured_plane" else 0.0
                    pose = measured["pose"]
                    rotation, center = pose[:3, :3], pose[:3, 3]
                    if source == "current_pose":
                        n = rotation @ normal
                        model_plane = Plane(n, -float(n @ transform_points(pose, local).mean(axis=0)),
                                            plane.rmse_m, plane.support)
                    seed_metrics = dict(
                        obj["metrics"], pose_solver=trial["solver"], reseed_geometry_source=source,
                        pose_trial_index=trial["trial_index"],
                        reprojection_rms_px=measured["reprojection"],
                        depth_median_residual_m=measured["depth"], normal_error_deg=measured["angle"],
                        measured_plane_normal_camera=plane.normal.tolist(), measured_plane_d_m=plane.d,
                        pose_rotation_step_deg=trial["rotation_step_deg"],
                        pose_normal_step_deg=trial["normal_step_deg"],
                        reseed_rotation_deg=trial["reseed_rotation_deg"], reseed_proposal_rotation_deg=rotation_deg,
                        plane_rmse_m=plane.rmse_m, plane_support=plane.support,
                        depth_fraction=depth_fraction,
                        depth_sampling_inset_fraction=sample_details.get("depth_sampling_inset_fraction", 0.0),
                        direction_prior_deg=trial["direction_prior_deg"],
                        direction_window_norm_deg=float(np.rad2deg(np.linalg.norm(
                            seed_window["angular_velocity"] * dt
                        ))),
                    )
                    previous_normal = (None if previous_state is None
                                       else previous_state.get("previous_measured_plane_normal"))
                    seed_metrics["measured_plane_normal_step_deg"] = (
                        None if previous_normal is None else float(np.rad2deg(np.arccos(np.clip(
                            measured_plane_normal @ previous_normal, -1, 1
                        ))))
                    )
                    seed_metrics["direction_decision_reason"] = "reseed_" + source
                    for key in ("tracking_check_group", "tracking_check_index"):
                        seed_metrics.pop(key, None)
                    seed_obj.update(
                        T_camera_from_box=pose.tolist(), center_camera_m=center.tolist(),
                        yaw_u_rad=measured["yaw"],
                        position_u_m=None if self.up is None else (self.up @ center).tolist(),
                        metrics=seed_metrics, quality_score=float(1 / (1 + trial["selection_cost"])),
                        tracking_reseed_stamp_ns=frame.stamp_ns,
                        faces=[dict(face, quad=measured["quad"].tolist(),
                                    normal_camera=measured["normal"].tolist(), metrics=seed_metrics)],
                        face_count=1,
                    )
                else:
                    rotation = physical_face[:3, :3] @ local_face[:3, :3].T
                    center = physical_face[:3, 3] - rotation @ local.mean(axis=0)
                    pose = transform(rotation, center)
                mask = np.zeros(gray.shape, np.uint8)
                cv2.fillConvexPoly(mask, np.rint(observed).astype(np.int32), 255)
                mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
                segmentation = obj.get("segmentation")
                if segmentation is not None and segmentation.get("stamp_ns") == frame.stamp_ns:
                    instance = np.zeros(gray.shape, np.uint8)
                    polygons = [
                        np.asarray(p, np.int32) for p in segmentation.get("polygons", []) if len(p) >= 3
                    ]
                    if polygons:
                        cv2.fillPoly(instance, polygons, 255)
                        mask &= instance
                # Harris remains the first choice on textured faces. On plain
                # fabric it can return only 2--4 points, making a stationary
                # target impossible to reseed. Shi--Tomasi supplies a second
                # pool; the same current flow/depth/boundary checks still apply.
                methods = (
                    ("harris", "shi_tomasi")
                    if self.cfg.track_feature_method == "adaptive"
                    else (self.cfg.track_feature_method,)
                )
                chosen = None
                for method in methods:
                    features = cv2.goodFeaturesToTrack(
                        gray,
                        self.cfg.track_seed_max_features,
                        0.005,
                        5,
                        mask=mask,
                        blockSize=5,
                        useHarrisDetector=method == "harris",
                        k=0.06,
                    )
                    if features is None or len(features) < self.cfg.track_min_features:
                        continue
                    uv = features.reshape(-1, 2)
                    model = (ray_plane(frame.camera, uv, model_plane) - center) @ rotation
                    inside = np.all(np.abs(model) <= size / 2 + 0.008, axis=1)
                    uv, model = uv[inside], model[inside]
                    if (
                        len(uv) < self.cfg.track_min_features
                        or cv2.contourArea(cv2.convexHull(uv.astype(np.float32)))
                        < cv2.contourArea(observed.astype(np.float32))
                        * self.cfg.track_min_feature_coverage_fraction
                    ):
                        continue
                    chosen = (uv, model, method)
                    # Preserve even a small usable Harris set: replacing real
                    # corners with many edge points can degrade motion accuracy.
                    break
                if chosen is None:
                    diagnostics.setdefault("face_rejections", []).append("seed_has_too_few_spread_features")
                    continue
                uv, model, method = chosen
                if trial is not None:
                    trial["selected"] = True
                    seed_obj["metrics"]["tracked_features"] = len(uv)
                    seed_obj["metrics"]["feature_seed_method"] = method
                    seed_obj["metrics"]["feature_coverage"] = float(
                        cv2.contourArea(cv2.convexHull(uv.astype(np.float32)))
                        / max(cv2.contourArea(np.asarray(seed_obj["faces"][0]["quad"], np.float32)), 1.0)
                    )
                seed_obj.setdefault("origin_detection_stamp_ns", frame.stamp_ns)
                diagnostics.update(accepted=True, chosen_face_axis=axis, chosen_face_sign=sign)
                return dict(
                    obj=seed_obj,
                    pose=pose,
                    quad_model=local,
                    normal_model=normal,
                    points=uv.astype(np.float32).reshape(-1, 1, 2),
                    model=model,
                    gray=gray,
                    stamp=frame.stamp_ns,
                    last_reseed=frame.stamp_ns,
                    feature_method=method,
                    measured_plane_normal=measured_plane_normal,
                    previous_pose=None if previous_state is None else previous_state.get("previous_pose"),
                    previous_measured_plane_normal=(None if previous_state is None
                                                    else previous_state.get("previous_measured_plane_normal")),
                    direction_window=seed_window,
                )
            except (ValueError, cv2.error) as error:
                diagnostics.setdefault("face_rejections", []).append(str(error))
                continue
        diagnostics["accepted"] = False
        return None

    @staticmethod
    def _tracking_face_view(frame, state):
        """用当前位姿计算面面积和视角，不使用后台源帧的旧投影。"""
        points = transform_points(state["pose"], state["quad_model"])
        normal = state["pose"][:3, :3] @ state["normal_model"]
        normal = normal / np.linalg.norm(normal)
        # Use the same box-center viewing ray as the tracking depth correction.
        center = state["pose"][:3, 3]
        quad = frame.camera.project(points).astype(np.float32)
        area = float(cv2.contourArea(cv2.convexHull(quad)))
        view_cos = abs(float(normal @ center)) / np.linalg.norm(center)
        return normal, view_cos, area

    def _prefer_tracking_face(self, old_view, new_view):
        # 方向不同且面积显著增加才交接，避免同面重置或两面之间反复跳动。
        old_normal, old_cos, old_area = old_view
        new_normal, _, new_area = new_view
        return (
            old_area > 0
            and old_cos < self.cfg.track_face_switch_max_view_cos
            and abs(float(old_normal @ new_normal)) < 0.5
            and new_area >= old_area * self.cfg.track_face_switch_min_area_ratio
        )

    def _face_switch_candidate(self, frame, state, obj, candidate, gray):
        """只将窄面交接给当前帧已有实测支持的另一个面。"""
        size = state["obj"]["size_m"]
        if (
            not any(abs(size - s) < 1e-6 for s in self.cfg.track_face_switch_sizes_m)
            or state["stamp"] != frame.stamp_ns
            or obj["size_m"] != size
        ):
            return None, None
        old_view = self._tracking_face_view(frame, state)
        if old_view[1] >= self.cfg.track_face_switch_max_view_cos:
            return None, None
        if candidate is None:
            if self.asynchronous:
                return None, None
            # Synchronous detections already passed current depth/pose checks.
            # Only seed after cheap projected-face checks promise a useful swap.
            pose = np.asarray(obj["T_camera_from_box"])
            vertices = cube_vertices(size)
            eligible = []
            for face in obj["faces"]:
                direction = pose[:3, :3].T @ np.asarray(face["normal_camera"])
                axis = int(np.argmax(np.abs(direction)))
                sign = 1 if direction[axis] > 0 else -1
                local = vertices[np.isclose(vertices[:, axis], sign * size / 2)]
                normal = np.zeros(3)
                normal[axis] = sign
                view = self._tracking_face_view(frame, dict(pose=pose, quad_model=local, normal_model=normal))
                if self._prefer_tracking_face(old_view, view):
                    eligible.append(face)
            if not eligible:
                return None, None
            candidate = self._seed(frame, dict(obj, faces=eligible), gray)
        if candidate is None or candidate["stamp"] != frame.stamp_ns:
            return None, None
        new_view = self._tracking_face_view(frame, candidate)
        if not self._prefer_tracking_face(old_view, new_view):
            return None, None
        return candidate, dict(
            size_m=size,
            old_view_cos=float(old_view[1]),
            new_view_cos=float(new_view[1]),
            old_area_px=float(old_view[2]),
            new_area_px=float(new_view[2]),
            new_quad=frame.camera.project(
                transform_points(candidate["pose"], candidate["quad_model"])
            ).tolist(),
            new_normal_camera=new_view[0].tolist(),
            source_stamp_ns=int(obj.get("detector_seed_stamp_ns", candidate["last_reseed"])),
            current_stamp_ns=int(frame.stamp_ns),
        )

    def _advance(self, frame, state, gray, distance, max_gap_s=None, diagnostics=None):
        cfg = self.cfg
        diagnostics = {} if diagnostics is None else diagnostics
        reprojection_limit = cfg.reprojection_limit(frame.camera)
        gap = (frame.stamp_ns - state["stamp"]) / 1e9
        diagnostics.update(gap_s=gap, features_before=len(state["points"]))
        if gap <= 0 or gap > (cfg.track_max_gap_s if max_gap_s is None else max_gap_s):
            raise ValueError("image tracking gap exceeds limit")
        p0 = state["points"]
        lk = dict(
            winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01)
        )
        p1, s1, error = cv2.calcOpticalFlowPyrLK(state["gray"], gray, p0, None, **lk)
        if p1 is None:
            raise ValueError("optical flow failed")
        back, s0, _ = cv2.calcOpticalFlowPyrLK(gray, state["gray"], p1, None, **lk)
        if back is None:
            raise ValueError("backward optical flow failed")
        fb_error_px = np.linalg.norm(back - p0, axis=2).ravel()
        valid = (
            (s1.ravel() > 0)
            & (s0.ravel() > 0)
            & (fb_error_px < cfg.track_fb_max_error_px)
            & (error.ravel() < cfg.track_lk_max_error)
        )
        valid &= np.all(np.isfinite(p1.reshape(-1, 2)), axis=1)
        a, b = p0[valid].reshape(-1, 2), p1[valid].reshape(-1, 2)
        model = state["model"][valid]
        diagnostics["bidirectional_features"] = len(a)
        if len(a) < cfg.track_min_features:
            raise ValueError("too few bidirectionally tracked features")
        homography, inliers = cv2.findHomography(a, b, cv2.RANSAC, cfg.track_homography_ransac_px)
        if homography is None or inliers is None:
            raise ValueError("feature geometry is degenerate")
        inliers = inliers.ravel() > 0
        diagnostics.update(
            homography_inliers=int(inliers.sum()), homography_inlier_fraction=float(inliers.mean())
        )
        if inliers.sum() < cfg.track_min_features or inliers.mean() < cfg.track_min_inlier_fraction:
            raise ValueError("inconsistent feature motion")
        a, b, model = a[inliers], b[inliers], model[inliers]
        diagnostics.update(
            feature_fb_median_px=float(np.median(fb_error_px[valid][inliers])),
            feature_lk_median_error=float(np.median(error.ravel()[valid][inliers])),
            feature_motion_median_px=float(np.median(np.linalg.norm(b - a, axis=1))),
        )
        old_quad = frame.camera.project(transform_points(state["pose"], state["quad_model"]))
        q = cv2.perspectiveTransform(old_quad.astype(np.float32).reshape(1, 4, 2), homography).reshape(4, 2)
        area = cv2.contourArea(q)
        coverage = cv2.contourArea(cv2.convexHull(b)) / max(area, 1.0)
        diagnostics.update(
            feature_coverage=coverage, feature_coverage_min=cfg.track_min_feature_coverage_fraction
        )
        # Keep the homography's image observation separate from the quadrilateral
        # projected by the chosen PnP pose. They are not four newly fitted lines.
        diagnostics["observed_quad_from_flow"] = q.tolist()
        if area < cfg.min_face_area_px or coverage < cfg.track_min_feature_coverage_fraction:
            raise ValueError("tracked feature coverage too narrow")
        samples, fraction, plane = self._sample_tracking_plane(frame, q, diagnostics)
        diagnostics.update(
            depth_fraction=fraction,
            plane_support=plane.support,
            plane_support_min=cfg.plane_min_support_fraction,
            plane_rmse_m=plane.rmse_m,
            plane_rmse_max_m=cfg.plane_max_rmse_m,
            measured_plane_normal_camera=plane.normal.tolist(),
            measured_plane_d_m=plane.d,
        )
        previous_plane_normal = state.get("measured_plane_normal")
        plane_normal_step = (
            None
            if previous_plane_normal is None
            else float(np.rad2deg(np.arccos(np.clip(plane.normal @ previous_plane_normal, -1, 1))))
        )
        diagnostics["measured_plane_normal_step_deg"] = plane_normal_step
        if plane.support < cfg.plane_min_support_fraction or plane.rmse_m > cfg.plane_max_rmse_m:
            raise ValueError("current tracked region has mixed or nonplanar depth")
        samples = samples[np.abs(samples @ plane.normal + plane.d) < cfg.plane_threshold_m]
        solutions, candidates, rejected, solver_errors = [], [], [], []
        attempts = diagnostics.setdefault("solver_attempts", [])

        def solve_regular(solver):
            attempt = dict(attempt_index=len(attempts), solver=solver, stage="solve", status="pending",
                           candidate_trial_indices=[])
            attempts.append(attempt)
            try:
                if solver == "ippe":
                    result = cv2.solvePnPGeneric(
                        np.ascontiguousarray(model), np.ascontiguousarray(b, dtype=np.float64),
                        frame.camera.k, frame.camera.d, flags=cv2.SOLVEPNP_IPPE,
                    )
                    found = list(zip(result[1], result[2]))
                else:
                    ok, rv, tv = cv2.solvePnP(
                        np.ascontiguousarray(model), np.ascontiguousarray(b, dtype=np.float64),
                        frame.camera.k, frame.camera.d,
                        cv2.Rodrigues(state["pose"][:3, :3])[0], state["pose"][:3, 3].copy(),
                        True, flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                    found = [(rv, tv)] if ok else []
                attempt["status"] = "returned_candidates" if found else "no_solution"
                return [(rv, tv, solver, i, attempt) for i, (rv, tv) in enumerate(found)]
            except (TypeError, ValueError, cv2.error) as error:
                attempt.update(status="solver_error", reason=str(error))
                solver_errors.append(dict(solver=solver, reason=str(error)))
                return []

        # 保留原有平手顺序：ITERATIVE 在前，两路 IPPE 随后。求解调用顺序也不变。
        ippe = solve_regular("ippe")
        solutions = solve_regular("iterative") + ippe

        def check_solution(rv, tv, solver, index=None, attempt=None):
            trial = self._new_pose_trial(diagnostics, solver, index)
            if attempt is not None:
                trial["solver_attempt_index"] = attempt["attempt_index"]
                attempt["candidate_trial_indices"].append(trial["trial_index"])
            try:
                rv = np.asarray(rv, np.float64).reshape(3, 1)
                tv = np.asarray(tv, np.float64).reshape(3, 1)
                if not np.isfinite(rv).all() or not np.isfinite(tv).all():
                    raise TrackingPoseRejected("parameters", "nonfinite_pose_parameters")
                pose = transform(cv2.Rodrigues(rv)[0], tv.ravel())
                candidate = self._evaluate_tracking_pose(
                    frame, state, pose, model, b, samples, plane, q, distance, trial
                )
                candidates.append(candidate)
            except (TypeError, ValueError, cv2.error) as error:
                self._fail_pose_trial(trial, error)
                rejected.extend(trial["failed_checks"])
                if not isinstance(error, TrackingPoseRejected):
                    solver_errors.append(dict(solver=solver, reason=str(error)))

        for rv, tv, solver, index, attempt in solutions:
            check_solution(rv, tv, solver, index, attempt)

        # 仅在普通解全灭，或近正对的观测最优解发生大方向跳变时加入联合解。
        # 这不是每帧以深度面替代图像解，也不放宽任何原有几何门槛。
        joint_reason = "all_regular_rejected" if not candidates else None
        if candidates and cfg.track_joint_on_ambiguity:
            best = min(candidates, key=lambda c: c["base_cost"])
            center_ray = best["pose"][:3, 3] / np.linalg.norm(best["pose"][:3, 3])
            view_cos = abs(float(best["normal"] @ center_ray))
            if view_cos > 0.95 and best["trial"]["rotation_step_deg"] > 5.0:
                joint_reason = "front_facing_direction_conflict"
        diagnostics.update(joint_pose_attempted=joint_reason is not None, joint_pose_reason=joint_reason)
        if joint_reason is not None:
            attempt = dict(attempt_index=len(attempts), solver="joint", stage="solve", status="pending",
                           candidate_trial_indices=[])
            attempts.append(attempt)
            try:
                rv, tv = self._joint_pose(frame, state, model, b, samples, plane,
                                          diagnostics=diagnostics.setdefault("joint_solver", {}))
                attempt["status"] = "returned_candidates"
                check_solution(rv, tv, "joint", 0, attempt)
            except (TypeError, ValueError, cv2.error) as error:
                attempt.update(status="solver_error", reason=str(error))
                solver_errors.append(dict(solver="joint", reason=str(error)))
        if solver_errors:
            diagnostics["pose_solver_errors"] = solver_errors
        if not candidates:
            raise ValueError("tracked pose rejected: " + ",".join(sorted(set(rejected))) if rejected
                             else "tracked pose inconsistent with current image/depth")

        window = state.get("direction_window")
        if max_gap_s is not None and gap > cfg.track_max_gap_s:
            window = None
        chosen_index, next_window, direction = select_direction_candidate(
            candidates, state["pose"][:3, :3], window, gap,
            prior_per_deg=cfg.track_direction_prior_per_deg,
            tau_s=cfg.track_direction_window_tau_s,
            escape_margin=cfg.track_direction_escape_margin,
            escape_frames=cfg.track_direction_escape_frames,
        )
        for candidate, terms in zip(candidates, direction.pop("candidates")):
            trial = candidate["trial"]
            trial.update(terms)
            trial["cost_components"]["direction"] = terms["applied_direction_cost"]
        chosen = candidates[chosen_index]
        pose, quad, normal, yaw = (chosen[key] for key in ("pose", "quad", "normal", "yaw"))
        reprojection, depth, angle, support = (chosen[key] for key in ("reprojection", "depth", "angle", "support"))
        chosen_trial = chosen["trial"]
        cost = chosen_trial["selection_cost"]
        chosen_trial["selected"] = True
        diagnostics.update(chosen_pose_solver=chosen_trial["solver"],
                           chosen_pose_trial=chosen_trial["trial_index"], direction=direction)
        obj = deepcopy(state["obj"])
        for key in (
            "track_id",
            "yaw_u_smoothed_rad",
            "yaw_u_continuous_rad",
            "center_camera_smoothed_m",
            "detector_confirmed",
            "detector_center_difference_m",
            "detector_confirmation_stamp_ns",
            "detector_seed_stamp_ns",
            "detector_match_action",
            "detector_center_difference_definition",
            "tracking_reseed_stamp_ns",
            "detection_check_index",
            "detection_trial_index",
        ):
            obj.pop(key, None)
        metrics = dict(
            reprojection_rms_px=reprojection,
            reprojection_limit_px=reprojection_limit,
            reprojection_definition="actual current LK feature correspondences",
            depth_median_residual_m=depth,
            depth_fraction=fraction,
            plane_rmse_m=plane.rmse_m,
            depth_sampling_inset_fraction=diagnostics["depth_sampling_inset_fraction"],
            plane_support=plane.support,
            normal_error_deg=angle,
            measured_plane_normal_camera=plane.normal.tolist(),
            measured_plane_d_m=plane.d,
            measured_plane_normal_step_deg=plane_normal_step,
            pose_solver=chosen_trial["solver"],
            pose_trial_index=chosen_trial["trial_index"],
            pose_rotation_step_deg=chosen_trial["rotation_step_deg"],
            pose_normal_step_deg=chosen_trial["normal_step_deg"],
            direction_prior_deg=chosen_trial["direction_prior_deg"],
            direction_window_norm_deg=direction["direction_window_norm_deg"],
            direction_decision_reason=direction["decision_reason"],
            tracked_features=len(b),
            feature_coverage=coverage,
            feature_seed_method=state.get("feature_method", "harris"),
            forward_backward_limit_px=cfg.track_fb_max_error_px,
            edge_support=support,
        )
        obj.update(
            status="tracked_rgbd",
            valid=True,
            source="current_rgbd_flow",
            method="flow_pnp",
            center_camera_m=pose[:3, 3].tolist(),
            T_camera_from_box=pose.tolist(),
            yaw_u_rad=yaw,
            position_u_m=None if self.up is None else (self.up @ pose[:3, 3]).tolist(),
            evidence="tracked_current_rgbd",
            face_count=1,
            quality_score=float(1 / (1 + cost)),
            metrics=metrics,
            faces=[
                dict(
                    quad=quad.tolist(),
                    normal_camera=normal.tolist(),
                    source="rgbd_feature_tracking",
                    inferred=True,
                    evidence=dict(tracked_features=len(b), depth_checked_this_frame=True),
                    metrics=metrics,
                )
            ],
            alternatives=[],
        )
        obj.pop("appearance", None)
        if cfg.analyze_colors:
            analyze_colors(frame, [obj], cfg)
        if cfg.require_target_colors:
            c = obj["appearance"]
            if (
                c["classified_samples"] < cfg.color_min_samples
                or c["classified_samples"] / max(1, c["surface_samples"]) < cfg.color_min_coverage_fraction
            ):
                raise ValueError("tracked region no longer has target colors")
        new = dict(
            state,
            obj=deepcopy(obj),
            pose=pose,
            points=b.reshape(-1, 1, 2),
            model=model,
            gray=gray,
            stamp=frame.stamp_ns,
            measured_plane_normal=plane.normal.copy(),
            previous_pose=state["pose"].copy(),
            previous_stamp_ns=state["stamp"],
            direction_window_before=copy_direction_window(window),
            previous_measured_plane_normal=(None if previous_plane_normal is None else previous_plane_normal.copy()),
            direction_window=next_window,
        )
        return obj, new

    def process(self, frame):
        start = time.perf_counter()
        cfg = self.cfg
        if self.last_stamp is not None and frame.stamp_ns <= self.last_stamp:
            raise ValueError("temporal pipeline requires increasing frame timestamps")
        reference = (frame.camera.calibration_id, tuple(cfg.up_camera or ()))
        if reference != self.reference:
            self.states = []
            self.last_detection = None
            self.reference = reference
            self.region_pipelines = {}
            self.up = upright_from_camera(cfg.up_camera)
            self.last_debug = None
            self.local_debug = ()
        self.last_stamp = frame.stamp_ns
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        _, _, edges = rgb_depth_edges(frame, cfg, gray=gray)
        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        advanced = []
        failures = []
        pending = []
        failure_details = []
        tracking_checks = []
        for state in self.states:
            details = dict(
                size_m=state["obj"]["size_m"],
                check_index=len(tracking_checks),
                previous_stamp_ns=state["stamp"],
                accepted=False,
            )
            try:
                obj, current_state = self._advance(frame, state, gray, distance, diagnostics=details)
                obj["metrics"].update(
                    tracking_check_group="tracking_checks", tracking_check_index=details["check_index"]
                )
                advanced.append((obj, current_state))
                details["accepted"] = True
            except (ValueError, cv2.error) as error:
                failures.append(str(error))
                failure_details.append(dict(details, reason=str(error)))
                if (frame.stamp_ns - state["stamp"]) / 1e9 <= cfg.track_max_gap_s:
                    pending.append(state)
            tracking_checks.append(details)
        tracking_end = time.perf_counter()
        detect = (
            not self.states
            or len(advanced) < len(self.states)
            or self.last_detection is None
            or (frame.stamp_ns - self.last_detection) / 1e9 >= cfg.redetect_interval_s
        )
        completed = False
        detector_age = None
        stale = 0
        background_ms = None
        fresh_states = {}
        seed_failures = []
        seed_failure_details = []
        seed_checks = []
        expired = 0
        raw_detector_objects = 0
        if self.asynchronous:
            result = self._empty_result(frame)
            reports = []
            with self.worker_condition:
                if self.worker_error is not None:
                    raise RuntimeError("background detector failed") from self.worker_error
                ready = self.worker_completed
                self.worker_completed = None
                if detect or self.worker_pending is not None:
                    # 单槽覆盖为最新帧：处理速度低于相机帧率时也不积压任务。
                    if self.worker_pending is not None:
                        self.worker_pending_replaced += 1
                    self.worker_pending = (frame, deepcopy([o for o, s in advanced]))
                    if detect:
                        self.last_detection = frame.stamp_ns
                    self.worker_condition.notify_all()
                worker_status = dict(
                    detector_busy=self.worker_busy,
                    detector_queue_depth=int(self.worker_pending is not None),
                    detector_pending_replaced=self.worker_pending_replaced,
                    detector_results_replaced=self.worker_results_replaced,
                )
            if ready is not None:
                source, old_result, reports, background_ms, debug = ready
                completed = True
                raw_detector_objects = len(old_result["objects"])
                if "yolo" in old_result:
                    # These polygons belong to the detector's source image,
                    # never claim they are a fresh mask on the current image.
                    result["yolo"] = old_result["yolo"]
                    # Retain the legacy inference-time summary; foreground
                    # processing_ms below is measured independently on this call.
                    result["timing"]["yolo_ms"] = old_result["timing"]["yolo_ms"]
                if self.debug_enabled:
                    self.last_debug, self.local_debug = debug
                snapshot = detector_snapshot(
                    source, frame, old_result, reports, background_ms, cfg.detector_max_age_s
                )
                result["detector_diagnostics"] = snapshot
                detector_age = snapshot["age_ms"]
                if snapshot["compatible_with_current_frame"]:
                    source_gray = cv2.cvtColor(source.rgb, cv2.COLOR_RGB2GRAY)
                    for obj in old_result["objects"]:
                        details = dict(
                            size_m=obj["size_m"],
                            check_index=len(seed_checks),
                            source_stamp_ns=source.stamp_ns,
                            current_stamp_ns=frame.stamp_ns,
                            accepted=False,
                        )
                        seed_checks.append(details)
                        seed = self._seed(source, obj, source_gray, diagnostics=details.setdefault("seed", {}))
                        if seed is None:
                            seed_failures.append("detected face lacks usable tracking features")
                            details["reason"] = seed_failures[-1]
                            seed_failure_details.append(dict(details, reason=seed_failures[-1]))
                            continue
                        details["seed_camera_from_box"] = seed["pose"].tolist()
                        details["seed_measured_plane_normal_camera"] = seed["measured_plane_normal"].tolist()
                        try:
                            # Detection latency and a gap in an established track
                            # have separate budgets. Every delayed acquisition is
                            # still remeasured in the current RGB-D before output.
                            current, current_state = self._advance(
                                frame,
                                seed,
                                gray,
                                distance,
                                max_gap_s=cfg.detector_max_age_s,
                                diagnostics=details,
                            )
                            details["accepted"] = True
                            current["metrics"].update(
                                tracking_check_group="detector_seed_checks",
                                tracking_check_index=details["check_index"],
                            )
                            current["detector_seed_stamp_ns"] = source.stamp_ns
                            current_state["direction_window"] = copy_direction_window()
                            current["metrics"].update(direction_window_norm_deg=0.0,
                                                       direction_window_reset_reason="fresh_detection")
                            result["objects"].append(current)
                            fresh_states[id(current)] = current_state
                        except (ValueError, cv2.error) as error:
                            seed_failures.append(str(error))
                            details["reason"] = str(error)
                            seed_failure_details.append(dict(details, reason=str(error)))
                            stale += 1
                else:
                    stale = len(old_result["objects"])
                    expired = stale
        elif detect:
            result, edges, reports, debug = self._full_detect(frame, [o for o, s in advanced])
            if self.debug_enabled:
                self.last_debug, self.local_debug = debug
            self.last_detection = frame.stamp_ns
            completed = True
            raw_detector_objects = len(result["objects"])
            result["detector_diagnostics"] = detector_snapshot(
                frame, frame, result, reports, result["timing"].get("processing_ms", 0.0), cfg.detector_max_age_s
            )
            result["detector_diagnostics"]["execution_mode"] = "synchronous"
            result.pop("detection_checks", None)
        else:
            reports = []
            result = self._empty_result(frame)
        fresh = result["objects"]
        merged = []
        states = []
        face_switches = []
        used = set()
        for obj, state in advanced:
            matches = [
                i
                for i, f in enumerate(fresh)
                if i not in used
                and f["size_m"] == obj["size_m"]
                and np.linalg.norm(np.asarray(f["center_camera_m"]) - obj["center_camera_m"])
                < max(0.04, obj["size_m"] * 0.2)
            ]
            if matches:
                i = min(
                    matches,
                    key=lambda i: np.linalg.norm(
                        np.asarray(fresh[i]["center_camera_m"]) - obj["center_camera_m"]
                    ),
                )
                used.add(i)
                difference = float(
                    np.linalg.norm(np.asarray(fresh[i]["center_camera_m"]) - obj["center_camera_m"])
                )
                candidate, switch = self._face_switch_candidate(
                    frame, state, fresh[i], fresh_states.get(id(fresh[i])), gray
                )
                if switch is not None:
                    # Feature model coordinates belong to the new pose: adopt
                    # the complete measured object/state together.
                    obj, state = fresh[i], candidate
                    state["direction_window"] = copy_direction_window()
                    face_switches.append((obj, switch))
                # 两种匹配都确认身份；只有换面才接管 fresh 的整套几何。
                obj["detector_confirmed"] = True
                obj["detector_match_action"] = "face_switch" if switch is not None else "confirm"
                obj["detector_confirmation_stamp_ns"] = frame.stamp_ns
                obj["detector_center_difference_m"] = difference
                obj["detector_center_difference_definition"] = "previous_track_vs_current_detector_candidate"
                state["obj"] = deepcopy(obj)
            merged.append(obj)
            states.append(state)
        for i, obj in enumerate(fresh):
            if i in used:
                continue
            merged.append(obj)
            states.append(fresh_states.get(id(obj)) or self._seed(frame, obj, gray))
        selected, suppressed = select_scene(merged, cfg.max_objects, cfg.up_camera)
        active_pairs = [(o, s) for o, s in zip(merged, states)
                        if s is not None and any(o is chosen for chosen in selected)]
        self.states = [s for o, s in active_pairs]
        reseed_checks = []
        # Replenish features from a currently validated pose, never from a miss.
        for i, (published, state) in enumerate(active_pairs):
            if (
                len(state["points"]) < cfg.track_min_reseed_features
                and (frame.stamp_ns - state["last_reseed"]) / 1e9 > cfg.track_reseed_interval_s
            ):
                details = dict(check_index=len(reseed_checks), size_m=state["obj"]["size_m"],
                               previous_stamp_ns=state["stamp"], current_stamp_ns=frame.stamp_ns,
                               accepted=False)
                reseed_checks.append(details)
                seed = self._seed(frame, published, gray, previous_state=state,
                                  distance=distance, diagnostics=details)
                if seed is not None:
                    seed["obj"]["metrics"].update(
                        tracking_check_group="reseed_checks", tracking_check_index=details["check_index"]
                    )
                    self.states[i] = seed
                    # 发布与内部特征模型使用同一个已验收姿态，避免本帧画旧框、
                    # 下一帧突然继承另一个内部姿态。
                    published.clear()
                    published.update(deepcopy(seed["obj"]))
        active = len(self.states)
        # Keep association/image memory briefly, without publishing any pose on
        # a failed frame. The next frame can retry against the last valid image.
        for state in pending:
            if not any(
                o["size_m"] == state["obj"]["size_m"]
                and np.linalg.norm(np.asarray(o["center_camera_m"]) - state["pose"][:3, 3]) < 0.06
                for o in selected
            ):
                self.states.append(state)
        result["objects"] = selected
        result["valid_object_count"] = len(selected)
        result["suppressed_hypotheses"].extend(suppressed)
        result["frame_status"] = (
            "observed"
            if selected
            else (
                "partial_or_rejected"
                if has_current_rejections(result) or failures or seed_failures
                else "no_target"
            )
        )
        result["temporal"] = dict(
            detector_ran=completed,
            detector_requested=detect,
            async_detection=self.asynchronous,
            detector_frame_age_ms=detector_age,
            discarded_old_detections=stale,
            expired_detections=expired,
            detector_seed_failures=seed_failures,
            background_detection_ms=background_ms,
            current_rgbd_tracks=len(advanced),
            tracking_failures=failures,
            tracking_failure_details=failure_details,
            tracking_checks=tracking_checks,
            detector_seed_failure_details=seed_failure_details,
            detector_seed_checks=seed_checks,
            reseed_checks=reseed_checks,
            face_switches=[event for obj, event in face_switches if any(obj is o for o in selected)],
            active_feature_tracks=active,
            pending_feature_tracks=len(self.states) - active,
            detector_raw_objects=raw_detector_objects,
            detector_objects=len(fresh),
            output_flow_objects=sum(o.get("source") == "current_rgbd_flow" for o in selected),
            local_searches=reports,
            stale_poses_published=0,
        )
        if self.asynchronous:
            result["temporal"].update(worker_status)
        result["timing"].update(
            feature_tracking_ms=(tracking_end - start) * 1000,
            processing_ms=(time.perf_counter() - start) * 1000,
        )
        if self.debug_enabled:
            # Only points remeasured in this frame; pending old tracks are omitted.
            self.debug_tracks = tuple(
                s["points"].reshape(-1, 2).copy() for s in self.states if s["stamp"] == frame.stamp_ns
            )
        return result, edges


def make_pipeline(cfg, live=False, debug=False):
    return (
        TemporalPipeline(cfg, live=live, debug=debug)
        if cfg.temporal_recovery
        else make_detection_pipeline(cfg, debug=debug)
    )
