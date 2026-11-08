"""Validated, portable configuration for the box pose pipeline."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import numpy as np


@dataclass
class Config:
    sizes_m: tuple = (0.20, 0.35)
    dimensions_nominal: bool = True
    up_camera: object = None
    roi_xyxy: object = None
    min_depth_m: float = 0.15
    max_depth_m: float = 4.0
    min_face_area_px: float = 180.0
    min_edge_px: float = 12.0
    max_faces: int = 24
    max_objects: int = 2
    canny_low: int = 35
    canny_high: int = 100
    depth_edge_threshold_m: float = 0.02
    plane_threshold_m: float = 0.008
    plane_max_rmse_m: float = 0.006
    plane_min_support_fraction: float = 0.65
    partial_min_plane_support_fraction: float = 0.75
    stack_min_edge_support_fraction: float = 0.65
    partial_min_edge_support_fraction: float = 0.60
    min_depth_samples: int = 24
    max_depth_samples: int = 384
    min_depth_fraction: float = 0.45
    size_tolerance_fraction: float = 0.12
    max_reprojection_px: float = 3.0
    reprojection_reference_width_px: object = None
    max_depth_residual_m: float = 0.012
    max_normal_error_deg: float = 15.0
    max_upright_tilt_deg: float = 12.0
    min_edge_support: float = 0.35
    plane_proposals: bool = True
    plane_backend: str = "numpy"
    geometry_backend: str = "auto"  # auto: 编译后用 C++，未编译用 NumPy；cpp: 强制要求已编译。
    pcl_stride: int = 2
    pcl_normal_smoothing_px: float = 12.0
    pcl_normal_method: str = "covariance"
    pcl_angular_threshold_deg: float = 15.0
    pcl_max_curvature: float = 0.03
    pcl_depth_change_factor: float = 0.02
    pcl_ransac_iterations: int = 128
    plane_grid_step: int = 12
    max_scene_planes: int = 6
    allow_stack_inference: bool = True
    upright_outline_proposals: bool = True
    allow_bottom_occlusion: bool = True
    method: str = "hybrid"
    refine_4d: bool = True
    refine_max_nfev: int = 12
    tracking: bool = True
    temporal_recovery: bool = False
    async_detection: bool = True
    redetect_interval_s: float = 0.15
    track_max_gap_s: float = 0.30
    detector_max_age_s: float = 0.75
    track_max_distance_m: float = 0.15
    smoothing_alpha: float = 0.5
    motion_prediction: bool = True
    prediction_max_gap_s: float = 0.30
    prediction_max_std_m: float = 0.10
    prediction_accel_std_m_s2: float = 1.0
    prediction_measurement_std_m: float = 0.01
    prediction_initial_velocity_std_m_s: float = 0.5
    track_min_features: int = 4
    track_seed_max_features: int = 40
    track_feature_method: str = "adaptive"
    track_depth_inset_fraction: float = 0.10
    track_min_reseed_features: int = 8
    track_min_feature_coverage_fraction: float = 0.12
    track_fb_max_error_px: float = 1.0
    track_lk_max_error: float = 25.0
    track_homography_ransac_px: float = 2.0
    track_min_inlier_fraction: float = 0.65
    track_quad_max_drift_px: float = 5.0
    track_reseed_interval_s: float = 0.30
    track_joint_on_ambiguity: bool = True
    # Candidate selection uses recent rotation trends; zero weight enables an A/B baseline.
    track_direction_window_tau_s: float = 0.20
    track_direction_prior_per_deg: float = 0.02
    track_direction_escape_margin: float = 0.15
    track_direction_escape_frames: int = 3
    track_reseed_max_rotation_deg: float = 15.0
    track_face_switch_sizes_m: tuple = (0.20,)
    track_face_switch_max_view_cos: float = 0.60
    track_face_switch_min_area_ratio: float = 1.50
    opencv_threads: int = 1
    analyze_colors: bool = False
    color_line_fallback: bool = False
    color_half_proposals: bool = True
    color_plane_priority: bool = True
    require_target_colors: bool = False
    two_color_sizes_m: tuple = ()
    color_min_samples: int = 32
    color_min_saturation: int = 70
    color_min_value: int = 40
    color_proposal_min_saturation: int = 55
    color_proposal_min_value: int = 30
    red_hue_ranges: tuple = ((0, 12), (168, 179))
    blue_hue_range: tuple = (95, 140)
    color_min_fit: float = 0.90
    color_ambiguity_margin: float = 0.08
    color_min_coverage_fraction: float = 0.60
    yolo_weights: object = None
    yolo_class: str = "tkk"
    yolo_device: str = "cpu"
    yolo_cpu_threads: int = 2
    yolo_upright_inference: bool = True
    yolo_conf: float = 0.40
    yolo_imgsz: int = 640
    yolo_max_instances: int = 6
    yolo_sizes_m: tuple = (0.20,)
    yolo_roi_padding_px: int = 20
    yolo_min_face_mask_fraction: float = 0.65
    yolo_roi_color_geometry: bool = True
    yolo_color_min_coverage_fraction: float = 0.60
    yolo_color_min_each_fraction: float = 0.0
    yolo_geometry_margin_px: int = 4
    yolo_roi_padding_fraction: float = 0.35
    yolo_line_band_px: float = 4.0
    yolo_line_min_points: int = 8
    yolo_line_min_span_fraction: float = 0.30
    yolo_partial_max_visible_fraction: float = 0.98
    yolo_partial_min_foreground_fraction: float = 0.04

    def validate(self):
        for key in (
            "dimensions_nominal",
            "plane_proposals",
            "allow_stack_inference",
            "upright_outline_proposals",
            "allow_bottom_occlusion",
            "refine_4d",
            "tracking",
            "motion_prediction",
            "temporal_recovery",
            "async_detection",
            "track_joint_on_ambiguity",
            "analyze_colors",
            "color_line_fallback",
            "color_half_proposals",
            "color_plane_priority",
            "require_target_colors",
            "yolo_roi_color_geometry",
            "yolo_upright_inference",
        ):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be a JSON boolean")
        if self.require_target_colors and not self.analyze_colors:
            raise ValueError("require_target_colors requires analyze_colors")
        sizes = np.asarray(self.sizes_m, dtype=float)
        if sizes.ndim != 1 or not len(sizes) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError("sizes_m must contain positive finite measured edge lengths")
        self.sizes_m = tuple(sorted(set(map(float, sizes))))
        switch_sizes = np.asarray(self.track_face_switch_sizes_m, dtype=float)
        if switch_sizes.ndim != 1 or not np.isfinite(switch_sizes).all() or np.any(switch_sizes <= 0):
            raise ValueError(
                "track_face_switch_sizes_m must contain positive finite edge lengths or be empty"
            )
        # Derived detection branches may handle only the other configured sizes.
        self.track_face_switch_sizes_m = tuple(sorted(set(map(float, switch_sizes))))
        if (
            isinstance(self.track_face_switch_max_view_cos, bool)
            or not np.isfinite(self.track_face_switch_max_view_cos)
            or not 0 < self.track_face_switch_max_view_cos <= 1
        ):
            raise ValueError("track_face_switch_max_view_cos must be in (0,1]")
        if (
            isinstance(self.track_face_switch_min_area_ratio, bool)
            or not np.isfinite(self.track_face_switch_min_area_ratio)
            or self.track_face_switch_min_area_ratio <= 1
        ):
            raise ValueError("track_face_switch_min_area_ratio must be finite and greater than 1")
        yolo_sizes = np.asarray(self.yolo_sizes_m, dtype=float)
        if (
            yolo_sizes.ndim != 1
            or not len(yolo_sizes)
            or not np.isfinite(yolo_sizes).all()
            or np.any(yolo_sizes <= 0)
        ):
            raise ValueError("yolo_sizes_m must contain positive finite measured edge lengths")
        self.yolo_sizes_m = tuple(sorted(set(map(float, yolo_sizes))))
        if self.yolo_weights is not None:
            if not isinstance(self.yolo_weights, (str, Path)) or not str(self.yolo_weights).strip():
                raise ValueError("yolo_weights must be a checkpoint path/directory or null")
            self.yolo_weights = str(self.yolo_weights)
            if not set(self.yolo_sizes_m).issubset(self.sizes_m):
                raise ValueError("enabled yolo_sizes_m must be a subset of sizes_m")
        if not isinstance(self.yolo_class, str) or not self.yolo_class.strip():
            raise ValueError("yolo_class must be a class name or numeric class ID string")
        if not isinstance(self.yolo_device, str) or not self.yolo_device.strip():
            raise ValueError("yolo_device must be a device string, e.g. '0' or 'cpu'")
        for key in ("yolo_conf", "yolo_min_face_mask_fraction"):
            if not np.isfinite(getattr(self, key)) or not 0 < getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in (0,1]")
        for key in ("yolo_imgsz", "yolo_max_instances", "yolo_cpu_threads"):
            if (
                isinstance(getattr(self, key), bool)
                or not isinstance(getattr(self, key), int)
                or getattr(self, key) < 1
            ):
                raise ValueError(f"{key} must be a positive integer")
        if (
            isinstance(self.yolo_roi_padding_px, bool)
            or not isinstance(self.yolo_roi_padding_px, int)
            or self.yolo_roi_padding_px < 0
        ):
            raise ValueError("yolo_roi_padding_px must be a nonnegative integer")
        for key in (
            "yolo_color_min_coverage_fraction",
            "yolo_line_min_span_fraction",
            "yolo_partial_min_foreground_fraction",
        ):
            if not np.isfinite(getattr(self, key)) or not 0 < getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in (0,1]")
        if (
            not np.isfinite(self.yolo_color_min_each_fraction)
            or not 0 <= self.yolo_color_min_each_fraction <= 0.5
        ):
            raise ValueError("yolo_color_min_each_fraction must be in [0,.5]")
        if not np.isfinite(self.yolo_roi_padding_fraction) or not 0 <= self.yolo_roi_padding_fraction <= 1:
            raise ValueError("yolo_roi_padding_fraction must be in [0,1]")
        if (
            not np.isfinite(self.yolo_partial_max_visible_fraction)
            or not 0.90 <= self.yolo_partial_max_visible_fraction < 1
        ):
            raise ValueError("yolo_partial_max_visible_fraction must be in [.90,1)")
        if (
            isinstance(self.yolo_geometry_margin_px, bool)
            or not isinstance(self.yolo_geometry_margin_px, int)
            or not 0 <= self.yolo_geometry_margin_px <= 50
        ):
            raise ValueError("yolo_geometry_margin_px must be an integer in [0,50]")
        if not np.isfinite(self.yolo_line_band_px) or not 0 < self.yolo_line_band_px <= 20:
            raise ValueError("yolo_line_band_px must be in (0,20]")
        if (
            isinstance(self.yolo_line_min_points, bool)
            or not isinstance(self.yolo_line_min_points, int)
            or self.yolo_line_min_points < 4
        ):
            raise ValueError("yolo_line_min_points must be an integer >=4")
        color_sizes = np.asarray(self.two_color_sizes_m, dtype=float)
        if (
            color_sizes.ndim != 1
            or not np.isfinite(color_sizes).all()
            or any(not any(abs(s - t) < 1e-6 for t in self.sizes_m) for s in color_sizes)
        ):
            raise ValueError("two_color_sizes_m must be a subset of sizes_m")
        self.two_color_sizes_m = tuple(map(float, color_sizes))
        for key in (
            "color_min_saturation",
            "color_min_value",
            "color_proposal_min_saturation",
            "color_proposal_min_value",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
                raise ValueError(f"{key} must be an integer in [0,255]")
        for name, ranges in (
            ("red_hue_ranges", self.red_hue_ranges),
            ("blue_hue_range", [self.blue_hue_range]),
        ):
            if not len(ranges):
                raise ValueError(f"{name} must not be empty")
            for interval in ranges:
                if (
                    len(interval) != 2
                    or any(isinstance(v, bool) or not isinstance(v, int) for v in interval)
                    or not 0 <= interval[0] <= interval[1] <= 179
                ):
                    raise ValueError(f"{name} must contain integer OpenCV hue ranges in [0,179]")
        if not 0 < self.color_min_fit <= 1 or not 0 < self.color_ambiguity_margin < 1:
            raise ValueError("color fit/margin must be finite fractions in (0,1]")
        if self.up_camera is not None:
            up = np.asarray(self.up_camera, dtype=float)
            if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
                raise ValueError("up_camera must be a nonzero finite vector in RGB optical coordinates")
            self.up_camera = (up / np.linalg.norm(up)).tolist()
        if self.roi_xyxy is not None:
            r = np.asarray(self.roi_xyxy)
            if (
                r.shape != (4,)
                or not np.isfinite(r).all()
                or np.any(r < 0)
                or not (r[2] > r[0] and r[3] > r[1])
            ):
                raise ValueError("roi_xyxy must be x0,y0,x1,y1 with exclusive upper bounds")
            self.roi_xyxy = list(map(int, r))
        positive = (
            "min_depth_m",
            "max_depth_m",
            "min_face_area_px",
            "min_edge_px",
            "plane_threshold_m",
            "plane_max_rmse_m",
            "depth_edge_threshold_m",
            "max_reprojection_px",
            "max_depth_residual_m",
            "max_normal_error_deg",
            "max_upright_tilt_deg",
            "track_max_gap_s",
            "detector_max_age_s",
            "track_max_distance_m",
            "redetect_interval_s",
            "track_fb_max_error_px",
            "track_lk_max_error",
            "track_homography_ransac_px",
            "track_quad_max_drift_px",
            "track_reseed_interval_s",
            "pcl_normal_smoothing_px",
            "pcl_angular_threshold_deg",
            "pcl_max_curvature",
            "pcl_depth_change_factor",
        )
        for key in positive:
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in (
            "track_direction_window_tau_s",
            "track_reseed_max_rotation_deg",
            "prediction_max_gap_s",
            "prediction_max_std_m",
            "prediction_accel_std_m_s2",
            "prediction_measurement_std_m",
            "prediction_initial_velocity_std_m_s",
        ):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{key} must be positive and finite")
        if self.track_reseed_max_rotation_deg > 180:
            raise ValueError("track_reseed_max_rotation_deg must not exceed 180 degrees")
        for key in ("track_direction_prior_per_deg", "track_direction_escape_margin"):
            value = getattr(self, key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{key} must be nonnegative and finite")
        reference = self.reprojection_reference_width_px
        if reference is not None and (
            isinstance(reference, bool) or not isinstance(reference, int) or reference < 8
        ):
            raise ValueError("reprojection_reference_width_px must be an integer >=8 or null")
        if self.min_depth_m >= self.max_depth_m:
            raise ValueError("min_depth_m must be below max_depth_m")
        for key in (
            "max_faces",
            "max_objects",
            "min_depth_samples",
            "max_depth_samples",
            "plane_grid_step",
            "max_scene_planes",
            "refine_max_nfev",
            "opencv_threads",
            "color_min_samples",
            "pcl_stride",
            "pcl_ransac_iterations",
            "track_min_features",
            "track_seed_max_features",
            "track_min_reseed_features",
            "track_direction_escape_frames",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.min_depth_samples < 6 or self.max_depth_samples < self.min_depth_samples:
            raise ValueError("depth sample limits must satisfy 6 <= min <= max")
        if not 0 < self.size_tolerance_fraction < 0.4:
            raise ValueError("size_tolerance_fraction must be in (0, 0.4)")
        if (
            self.track_seed_max_features < self.track_min_features
            or self.track_min_reseed_features < self.track_min_features
        ):
            raise ValueError("tracking feature counts must be >= track_min_features")
        if (
            not 0 < self.track_min_inlier_fraction <= 1
            or not 0 < self.track_min_feature_coverage_fraction <= 1
        ):
            raise ValueError("tracking fractions must be in (0, 1]")
        if (
            not np.isfinite(self.track_depth_inset_fraction)
            or not 0 <= self.track_depth_inset_fraction <= 0.25
        ):
            raise ValueError("track_depth_inset_fraction must be in [0, 0.25]")
        for key in (
            "min_depth_fraction",
            "min_edge_support",
            "smoothing_alpha",
            "plane_min_support_fraction",
            "partial_min_plane_support_fraction",
            "stack_min_edge_support_fraction",
            "partial_min_edge_support_fraction",
            "track_min_feature_coverage_fraction",
            "track_min_inlier_fraction",
            "color_min_coverage_fraction",
        ):
            if not np.isfinite(getattr(self, key)) or not 0 < getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in (0, 1]")
        if not 0 <= self.canny_low < self.canny_high <= 255:
            raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255")
        if self.method not in ("depth", "pnp", "hybrid"):
            raise ValueError("method must be depth, pnp or hybrid")
        if self.track_feature_method not in ("adaptive", "harris", "shi_tomasi"):
            raise ValueError("track_feature_method must be adaptive, harris or shi_tomasi")
        if self.plane_backend not in ("numpy", "pcl"):
            raise ValueError("plane_backend must be numpy or pcl")
        if self.geometry_backend not in ("auto", "numpy", "cpp"):
            raise ValueError("geometry_backend must be auto, numpy or cpp")
        if self.pcl_normal_method not in ("covariance", "average_gradient"):
            raise ValueError("pcl_normal_method must be covariance or average_gradient")
        if self.pcl_stride > 8 or self.pcl_normal_smoothing_px < 3 or self.pcl_angular_threshold_deg >= 90:
            raise ValueError(
                "PCL requires stride <=8, smoothing >=3 pixels and angular threshold <90 degrees"
            )
        return self

    def to_dict(self):
        return asdict(self)

    def reprojection_limit(self, camera):
        """Keep the configured image-error budget consistent across resolutions."""
        reference = self.reprojection_reference_width_px
        return (
            self.max_reprojection_px
            if reference is None
            else self.max_reprojection_px * camera.width / reference
        )

    @classmethod
    def load(cls, path=None, *, size=None, **overrides):
        values = {} if path is None else json.loads(Path(path).read_text())
        if not isinstance(values, dict):
            raise ValueError("configuration must be a JSON object")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration options: {sorted(unknown)}")
        values.update({k: v for k, v in overrides.items() if v is not None})
        # JSON 内的相对权重路径以配置文件目录为基准；CLI 覆盖仍相对当前目录。
        if path is not None and values.get("yolo_weights") and overrides.get("yolo_weights") is None:
            weights = Path(values["yolo_weights"]).expanduser()
            if not weights.is_absolute():
                values["yolo_weights"] = str((Path(path).resolve().parent / weights).resolve())
        cfg = cls(**values).validate()
        if size is not None:
            if size not in ("big", "small"):
                raise ValueError("size must be big or small")
            selected = 0.35 if size == "big" else 0.20
            if not any(abs(s - selected) < 1e-6 for s in cfg.sizes_m):
                raise ValueError(f"size={size} requires {selected} m in sizes_m")
            cfg.sizes_m = (selected,)
            cfg.two_color_sizes_m = tuple(s for s in cfg.two_color_sizes_m if abs(s - selected) < 1e-6)
            cfg.track_face_switch_sizes_m = tuple(
                s for s in cfg.track_face_switch_sizes_m if abs(s - selected) < 1e-6
            )
            yolo_sizes = tuple(s for s in cfg.yolo_sizes_m if abs(s - selected) < 1e-6)
            if yolo_sizes:
                cfg.yolo_sizes_m = yolo_sizes
            else:
                # 大块走原几何分支，不加载只为小块训练的分割模型。
                cfg.yolo_weights = None
            cfg.validate()
        return cfg
