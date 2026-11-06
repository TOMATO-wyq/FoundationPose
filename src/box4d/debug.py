"""Opt-in measurements from the actual pipeline, plus independent debug windows.

Detection images always use their own source frame. Background detections are
never drawn over a newer RGB image. Diagnostic normals do not affect inference.
"""

from collections import deque
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import textwrap
import time

import cv2
import numpy as np

from .io import dump_json


STAGES = (
    "01 Input + aligned depth",
    "02 Color proposals",
    "03 RGB + depth edges",
    "04 Local normals",
    "05 Plane regions",
    "06 Face candidates",
    "07 Face validation",
    "08 Current tracking",
)
PALETTE = (
    (80, 220, 80),
    (240, 180, 40),
    (180, 80, 240),
    (40, 190, 255),
    (220, 220, 60),
    (120, 140, 255),
    (255, 140, 140),
    (200, 255, 160),
)
TUNING_WINDOW = "09 Tuning - live parameters"
# label, Config field, display multiplier, displayed minimum, step, max position
# The file config remains the source of all fields; sliders expose the most
# useful thresholds for diagnosing the eight stages.
TUNING_SPECS = (
    ("Z min mm", "min_depth_m", 1000, 100, 10, 290),
    ("Z max mm", "max_depth_m", 1000, 300, 10, 570),
    ("Depth coverage %", "min_depth_fraction", 100, 10, 1, 90),
    ("Min depth samples", "min_depth_samples", 1, 6, 1, 378),
    ("Max depth samples", "max_depth_samples", 1, 24, 1, 1000),
    ("Canny low", "canny_low", 1, 0, 1, 254),
    ("Canny high", "canny_high", 1, 1, 1, 254),
    ("Depth edge mm", "depth_edge_threshold_m", 1000, 2, 1, 98),
    ("Color min S", "color_min_saturation", 1, 0, 1, 255),
    ("Color min V", "color_min_value", 1, 0, 1, 255),
    ("Proposal min S", "color_proposal_min_saturation", 1, 0, 1, 255),
    ("Proposal min V", "color_proposal_min_value", 1, 0, 1, 255),
    ("Color min samples", "color_min_samples", 1, 4, 1, 124),
    ("Color coverage %", "color_min_coverage_fraction", 100, 20, 1, 80),
    ("Color fit %", "color_min_fit", 100, 50, 1, 50),
    ("Color ambiguity %", "color_ambiguity_margin", 100, 1, 1, 29),
    ("Plane dist mm", "plane_threshold_m", 1000, 1, 1, 29),
    ("Plane RMS mm", "plane_max_rmse_m", 1000, 1, 1, 19),
    ("Plane support %", "plane_min_support_fraction", 100, 30, 1, 70),
    ("Partial plane %", "partial_min_plane_support_fraction", 100, 30, 1, 70),
    ("Partial edge %", "partial_min_edge_support_fraction", 100, 20, 1, 80),
    ("PCL stride", "pcl_stride", 1, 1, 1, 3),
    ("PCL smooth px", "pcl_normal_smoothing_px", 1, 3, 1, 37),
    ("PCL angle deg", "pcl_angular_threshold_deg", 1, 1, 1, 59),
    ("PCL curvature x.01", "pcl_max_curvature", 100, 1, 1, 19),
    ("PCL depth change x1000", "pcl_depth_change_factor", 1000, 1, 1, 199),
    ("PCL local RANSAC iters", "pcl_ransac_iterations", 1, 1, 1, 511),
    ("Max scene planes", "max_scene_planes", 1, 1, 1, 31),
    ("Min face area px", "min_face_area_px", 1, 20, 4, 495),
    ("Max face candidates", "max_faces", 1, 1, 1, 63),
    ("Max objects", "max_objects", 1, 1, 1, 11),
    ("Min edge px", "min_edge_px", 1, 4, 1, 46),
    ("Edge support %", "min_edge_support", 100, 5, 1, 90),
    ("Stack edge %", "stack_min_edge_support_fraction", 100, 20, 1, 80),
    ("Size tolerance %", "size_tolerance_fraction", 100, 2, 1, 28),
    ("Reprojection x10 px", "max_reprojection_px", 10, 10, 1, 90),
    ("Depth residual mm", "max_depth_residual_m", 1000, 1, 1, 49),
    ("Normal error deg", "max_normal_error_deg", 1, 1, 1, 44),
    ("Upright tilt deg", "max_upright_tilt_deg", 1, 1, 1, 44),
    ("Redetect ms", "redetect_interval_s", 1000, 50, 10, 95),
    ("Detect age ms", "detector_max_age_s", 1000, 50, 10, 195),
    ("Track gap ms", "track_max_gap_s", 1000, 50, 10, 95),
    ("Track travel cm", "track_max_distance_m", 100, 2, 1, 48),
    ("Track min features", "track_min_features", 1, 4, 1, 16),
    ("Seed max features", "track_seed_max_features", 1, 4, 4, 49),
    ("Reseed min points", "track_min_reseed_features", 1, 4, 1, 36),
    ("Feature coverage %", "track_min_feature_coverage_fraction", 100, 2, 1, 78),
    ("Flow back x10 px", "track_fb_max_error_px", 10, 1, 1, 49),
    ("LK error", "track_lk_max_error", 1, 1, 1, 99),
    ("Track RANSAC x10 px", "track_homography_ransac_px", 10, 5, 1, 95),
    ("Track inliers %", "track_min_inlier_fraction", 100, 40, 1, 60),
    ("Track quad drift px", "track_quad_max_drift_px", 1, 1, 1, 29),
    ("Reseed interval ms", "track_reseed_interval_s", 1000, 50, 10, 95),
    ("Track depth inset %", "track_depth_inset_fraction", 100, 0, 1, 25),
    ("YOLO confidence %", "yolo_conf", 100, 1, 1, 99),
    ("YOLO max instances", "yolo_max_instances", 1, 1, 1, 11),
    ("YOLO ROI padding px", "yolo_roi_padding_px", 1, 0, 1, 100),
    ("Face inside mask %", "yolo_min_face_mask_fraction", 100, 1, 1, 99),
    ("ROI red blue coverage %", "yolo_color_min_coverage_fraction", 100, 1, 1, 99),
    ("ROI each color min %", "yolo_color_min_each_fraction", 100, 0, 1, 50),
    ("ROI padding fraction %", "yolo_roi_padding_fraction", 100, 0, 1, 100),
    ("Plane seed margin px", "yolo_geometry_margin_px", 1, 0, 1, 50),
    ("Line search band px", "yolo_line_band_px", 1, 1, 1, 19),
    ("Line min points", "yolo_line_min_points", 1, 4, 1, 60),
    ("Line visible span %", "yolo_line_min_span_fraction", 100, 1, 1, 99),
    ("Partial max visible %", "yolo_partial_max_visible_fraction", 100, 90, 1, 9),
    ("Partial foreground min %", "yolo_partial_min_foreground_fraction", 100, 1, 1, 49),
    ("Direction window ms", "track_direction_window_tau_s", 1000, 50, 10, 95),
    ("Direction prior x.001/deg", "track_direction_prior_per_deg", 1000, 0, 1, 100),
    ("Direction escape margin x.01", "track_direction_escape_margin", 100, 0, 1, 50),
    ("Direction escape frames", "track_direction_escape_frames", 1, 1, 1, 7),
    ("Reseed max rotation deg", "track_reseed_max_rotation_deg", 1, 1, 1, 44),
)

# Keep every page short enough for a laptop screen. The native HighGUI
# trackbars cannot scroll, so paging must recreate only the visible controls.
TUNING_PAGES = (
    ("输入深度与 Canny", "Depth and Canny", 0, 7),
    ("深度边缘与颜色", "Depth edges and color", 7, 14),
    ("颜色模型与平面验收", "Color and plane checks", 14, 21),
    ("PCL 法线与平面", "PCL normals and planes", 21, 28),
    ("候选与尺寸验收", "Candidates and size", 28, 35),
    ("位姿检查与重检测", "Pose and redetection", 35, 41),
    ("跟踪与特征点", "Tracking and features", 41, 47),
    ("光流与补点", "Optical flow and reseeding", 47, 54),
    ("YOLO 小块分割", "YOLO small cubes", 54, 58),
    ("小块颜色与线段范围", "Small cube color and line region", 58, 63),
    ("小块线段与底部遮挡", "Small cube lines and occlusion", 63, 67),
    ("方向连续性与补点纠偏", "Rotation continuity and reseeding", 67, 72),
)
TUNING_ZH = (
    "最近距离 mm",
    "最远距离 mm",
    "有效深度占比 %",
    "最少深度点数",
    "最多深度点数",
    "Canny 低阈值",
    "Canny 高阈值",
    "深度边缘门限 mm",
    "分类最低饱和度",
    "分类最低亮度",
    "候选最低饱和度",
    "候选最低亮度",
    "最少颜色样本",
    "颜色覆盖率 %",
    "颜色模型拟合率 %",
    "颜色模型分差 %",
    "平面内点距离 mm",
    "平面 RMS 上限 mm",
    "平面内点占比 %",
    "半面内点占比 %",
    "半面边缘支持率 %",
    "PCL 采样步长",
    "法线平滑范围 px",
    "法线容许夹角 deg",
    "曲率上限 ×0.01",
    "深度变化系数 ×0.001",
    "局部 RANSAC 次数",
    "场景平面预算",
    "最小面面积 px²",
    "初始面候选上限",
    "输出方块数量上限",
    "最短边长 px",
    "边缘支持率 %",
    "堆叠边缘支持率 %",
    "尺寸容差 %",
    "重投影上限 ×0.1 px",
    "位姿深度残差 mm",
    "法线误差上限 deg",
    "竖直倾角上限 deg",
    "重检测间隔 ms",
    "检测结果年龄 ms",
    "跟踪中断上限 ms",
    "更新平移上限 cm",
    "最少跟踪点数",
    "初始特征点上限",
    "低于此点数补点",
    "特征点覆盖率 %",
    "光流往返误差 ×0.1 px",
    "LK 图像块误差",
    "单应内点门限 ×0.1 px",
    "运动内点占比 %",
    "四边形偏差 px",
    "补点最短间隔 ms",
    "跟踪深度内缩 %",
    "YOLO 置信度 %",
    "YOLO 实例数量上限",
    "ROI 外扩像素",
    "面落在分割内占比 %",
    "ROI 红蓝总占比 %",
    "ROI 单色最低占比 %",
    "ROI 相对外扩 %",
    "平面取点外扩 px",
    "边线搜索半径 px",
    "每条边最少像素点",
    "边线可见跨度 %",
    "遮挡面可见高度上限 %",
    "遮挡前景最低占比 %",
    "方向窗口时间 ms",
    "方向先验权重 ×0.001/deg",
    "反向逃生代价优势 ×0.01",
    "反向逃生连续帧数",
    "补点最大旋转 deg",
)


def _tuning_caption(page):
    title, english, start, end = TUNING_PAGES[page]
    caption = np.full((56, 700, 3), 28, np.uint8)
    # OpenCV's built-in font cannot draw Chinese. Pillow is optional: the
    # controls and an English caption remain usable without it or a CJK font.
    font_paths = (
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
    )
    try:
        from PIL import Image, ImageDraw, ImageFont

        font_path = next(path for path in font_paths if path.is_file())
        font = ImageFont.truetype(str(font_path), 14)
        canvas = Image.fromarray(caption)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((8, 3, 138, 28), outline=(150, 150, 150))
        draw.rectangle((562, 3, 692, 28), outline=(150, 150, 150))
        draw.text((28, 5), "上一页 [", font=font, fill=(245, 220, 110))
        draw.text((584, 5), "下一页 ]", font=font, fill=(245, 220, 110))
        draw.text((168, 5), f"第 {page + 1}/{len(TUNING_PAGES)} 页：{title}", font=font, fill=(245, 220, 110))
        draw.text(
            (10, 31),
            "点击两侧按钮或按 [ / ] 翻页；S 保存全部页参数。切页保留修改。",
            font=font,
            fill=(230, 230, 230),
        )
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    except (ImportError, OSError, StopIteration):
        _text(caption, "< Prev [", (10, 24), scale=0.48)
        _text(caption, "Next ] >", (590, 24), scale=0.48)
        _text(caption, f"{page + 1}/{len(TUNING_PAGES)}: {english}", (150, 24), scale=0.48)
        _text(caption, "Click Prev/Next or [ / ]: switch | S: save ALL pages", (10, 50), scale=0.48)
        return caption


def _tuning_label(index, start):
    # HighGUI's native GTK/Qt labels support Unicode; only cv2.putText needs
    # the separate font renderer used in the small navigation image.
    return f"{index - start + 1}. {TUNING_ZH[index]} | {TUNING_SPECS[index][0]}"


class _TuningPanel:
    def __init__(self, shared):
        self.shared = shared
        self.page = None
        self.requested_page = None
        self.show_page(0)

    def show_page(self, page):
        if self.page is not None:
            cv2.destroyWindow(TUNING_WINDOW)
        self.page = page % len(TUNING_PAGES)
        cv2.namedWindow(TUNING_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.moveWindow(TUNING_WINDOW, 20, 20)
        _, _, start, end = TUNING_PAGES[self.page]
        with self.shared.get_lock():
            positions = tuple(self.shared[:])
        for index in range(start, end):
            label, _, _, minimum, step, max_position = TUNING_SPECS[index]
            initial = minimum + positions[index] * step
            cv2.createTrackbar(
                _tuning_label(index, start),
                TUNING_WINDOW,
                int(initial),
                int(minimum + max_position * step),
                lambda _value: None,
            )
            self._set_minimum(_tuning_label(index, start), minimum)
        cv2.imshow(TUNING_WINDOW, _tuning_caption(self.page))
        cv2.setMouseCallback(TUNING_WINDOW, self._on_mouse)
        cv2.resizeWindow(TUNING_WINDOW, 700, 56)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP and 3 <= y <= 28:
            if 8 <= x <= 138:
                self.requested_page = (self.page - 1) % len(TUNING_PAGES)
            elif 562 <= x <= 692:
                self.requested_page = (self.page + 1) % len(TUNING_PAGES)

    @staticmethod
    def _set_minimum(label, minimum):
        if minimum and hasattr(cv2, "setTrackbarMin"):
            try:
                cv2.setTrackbarMin(label, TUNING_WINDOW, int(minimum))
            except cv2.error:
                pass  # Readback also clamps with older GUI backends.

    def poll(self):
        _, _, start, end = TUNING_PAGES[self.page]
        with self.shared.get_lock():
            for index in range(start, end):
                label, _, _, minimum, step, max_position = TUNING_SPECS[index]
                name = _tuning_label(index, start)
                displayed = cv2.getTrackbarPos(name, TUNING_WINDOW)
                position = int(np.clip(round((displayed - minimum) / step), 0, max_position))
                actual = int(minimum + position * step)
                if displayed != actual:
                    cv2.setTrackbarPos(name, TUNING_WINDOW, actual)
                self.shared[index] = position
        requested, self.requested_page = self.requested_page, None
        if requested is not None and requested != self.page:
            self.show_page(requested)


def tuning_positions(cfg):
    return [
        int(np.clip(round((getattr(cfg, key) * multiplier - minimum) / step), 0, maximum))
        for _, key, multiplier, minimum, step, maximum in TUNING_SPECS
    ]


def tuning_values(positions):
    values = {
        key: (minimum + int(position) * step) / multiplier
        for position, (_, key, multiplier, minimum, step, _) in zip(positions, TUNING_SPECS)
    }
    integer_fields = {
        "min_depth_samples",
        "max_depth_samples",
        "canny_low",
        "canny_high",
        "color_min_saturation",
        "color_min_value",
        "color_proposal_min_saturation",
        "color_proposal_min_value",
        "color_min_samples",
        "pcl_stride",
        "pcl_ransac_iterations",
        "max_scene_planes",
        "min_face_area_px",
        "max_faces",
        "max_objects",
        "min_edge_px",
        "track_min_features",
        "track_seed_max_features",
        "track_min_reseed_features",
        "track_direction_escape_frames",
        "track_lk_max_error",
        "yolo_max_instances",
        "yolo_roi_padding_px",
        "yolo_geometry_margin_px",
        "yolo_line_min_points",
    }
    return {key: int(value) if key in integer_fields else value for key, value in values.items()}


class DetectionDebug:
    def __init__(self, frame, cfg):
        self.frame = frame
        self.config = cfg.to_dict()
        self.images = {}
        self.planes = []
        self.faces = []
        self._face_ids = {}
        self._face_refs = []  # Prevent Python id reuse for temporary candidates.
        self.result = None
        self.backend = dict(name="numpy", segmentation="sequential RANSAC")

    def face(self, face):
        key = id(face)
        if key not in self._face_ids:
            self._face_ids[key] = len(self.faces)
            self._face_refs.append(face)
            self.faces.append(
                dict(
                    id=len(self.faces),
                    source=face.source,
                    quad=face.quad.copy(),
                    inferred=face.inferred,
                    evidence=deepcopy(face.evidence),
                    edge_support=face.edge_support,
                    status="proposed",
                    trials=[],
                )
            )
        return self.faces[self._face_ids[key]]

    def finish(self, result):
        for rejected in result["rejected_faces"]:
            for face in self.faces:
                if face["source"] == rejected["source"] and np.allclose(face["quad"], rejected["quad"]):
                    face.update(status="rejected", reason=rejected["reason"])
        self.result = deepcopy(result)

    def report(self):
        return dict(
            sequence=self.frame.sequence,
            stamp_ns=self.frame.stamp_ns,
            camera=self.frame.camera.to_dict(),
            config=self.config,
            planes=self.planes,
            faces=self.faces,
            result=self.result,
            backend=self.backend,
            normals_note=(
                "actual PCL input normals used in segmentation"
                if self.backend["name"] == "pcl"
                else "visual diagnostic only; not used by NumPy plane segmentation"
            ),
        )


def _text(image, value, xy, color=(235, 235, 235), scale=0.43):
    cv2.putText(image, str(value), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), 3, cv2.LINE_AA)
    cv2.putText(image, str(value), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _panel(image, title, source, lines=()):
    """Fixed dimensions make independent-window IPC bounded and predictable."""
    canvas = np.full((600, 720, 3), 22, np.uint8)
    _text(canvas, title, (10, 21), scale=0.54)
    _text(canvas, source, (10, 43), color=(100, 215, 255))
    if image is not None:
        h, w = image.shape[:2]
        factor = min(720 / w, 350 / h)
        resized = cv2.resize(image, (max(1, round(w * factor)), max(1, round(h * factor))))
        y, x = 52 + (350 - resized.shape[0]) // 2, (720 - resized.shape[1]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    row = 420
    for line in lines:
        for part in textwrap.wrap(str(line), width=98) or [""]:
            if row > 575:
                break
            _text(canvas, part, (10, row))
            row += 18
    _text(
        canvas,
        "09 Tuning sliders live | S save config + snapshot | SPACE freeze | N/P face | L ROI | Q quit",
        (10, 593),
        scale=0.36,
    )
    return canvas


def _normals(frame):
    """Central-difference normals on a subsampled organized cloud; no Z bins."""
    yy, xx = np.mgrid[0 : frame.camera.height : 2, 0 : frame.camera.width : 2]
    z = frame.depth[::2, ::2]
    safe_z = np.where(np.isfinite(z) & (z > 0), z, 0.0)
    xyz = (
        frame.camera.rays(np.column_stack((xx.ravel(), yy.ravel()))).reshape(*z.shape, 3) * safe_z[..., None]
    )
    # A 16-source-pixel baseline reduces the extreme sensitivity of one-pixel
    # depth derivatives to quantization/noise. No depth holes are filled.
    radius = min(4, max(1, (min(z.shape) - 1) // 2))
    center = safe_z[radius:-radius, radius:-radius]
    dx = xyz[radius:-radius, 2 * radius :] - xyz[radius:-radius, : -2 * radius]
    dy = xyz[2 * radius :, radius:-radius] - xyz[: -2 * radius, radius:-radius]
    normal = np.cross(dx, dy)
    length = np.linalg.norm(normal, axis=2)
    good = np.isfinite(normal).all(axis=2) & (length > 1e-10)
    for local in (
        center,
        safe_z[radius:-radius, 2 * radius :],
        safe_z[radius:-radius, : -2 * radius],
        safe_z[2 * radius :, radius:-radius],
        safe_z[: -2 * radius, radius:-radius],
    ):
        good &= (local > 0) & (np.abs(local - center) < 0.05)
    normal[good] /= length[good, None]
    flip = np.einsum("ijk,ijk->ij", normal, xyz[radius:-radius, radius:-radius]) > 0
    normal[flip] *= -1
    image = np.zeros((*z.shape, 3), np.uint8)
    encoded = np.zeros_like(normal, dtype=np.uint8)
    encoded[good] = np.clip((normal[good, ::-1] + 1) * 127.5, 0, 255).astype(np.uint8)
    image[radius:-radius, radius:-radius] = encoded
    return image


def render_stages(frame, result, snapshot, local_snapshots=(), tracks=(), face_index=0, detector_index=0):
    snapshots = [s for s in (snapshot, *local_snapshots) if s is not None]
    trace = snapshots[detector_index % len(snapshots)] if snapshots else None
    current = f"CURRENT frame={frame.sequence} stamp={frame.stamp_ns}"
    rgb = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    if trace is None:
        panels = [
            _panel(None, name, current, ["Waiting for the first completed detection."])
            for name in STAGES[:-1]
        ]
    else:
        detection = trace.frame
        cfg = trace.config
        base = cv2.cvtColor(detection.rgb, cv2.COLOR_RGB2BGR)
        age = (frame.stamp_ns - detection.stamp_ns) / 1e6
        source = f"DETECT frame={detection.sequence} age={age:.0f}ms ROI={cfg['roi_xyxy']}"
        valid = (
            np.isfinite(detection.depth)
            & (detection.depth >= cfg["min_depth_m"])
            & (detection.depth <= cfg["max_depth_m"])
        )
        scaled = np.zeros(detection.depth.shape, np.uint8)
        scaled[valid] = np.clip(
            (detection.depth[valid] - cfg["min_depth_m"]) / (cfg["max_depth_m"] - cfg["min_depth_m"]) * 254
            + 1,
            1,
            255,
        ).astype(np.uint8)
        depth = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        depth[~valid] = 0
        image = np.concatenate((base, depth), axis=1)
        panels = [
            _panel(
                image,
                STAGES[0],
                source,
                [
                    f"RGB left; aligned raw depth right. Black = invalid/out of range. Valid {valid.mean():.1%}",
                    f"Depth range {cfg['min_depth_m']:.2f}..{cfg['max_depth_m']:.2f}m; no hole filling.",
                ],
            )
        ]
        color = base // 4
        color[trace.images.get("saturation", np.zeros(valid.shape, np.uint8)) > 0] = (30, 180, 180)
        labels = trace.images.get("color_labels", np.zeros(valid.shape, np.uint8))
        color[labels == 1] = (50, 50, 255)
        color[labels == -1] = (255, 130, 50)
        yolo = (trace.result or {}).get("yolo")
        if yolo:
            for instance in yolo["instances"]:
                polygons = [np.asarray(p, np.int32) for p in instance["polygons"] if len(p) >= 3]
                if polygons:
                    cv2.polylines(color, polygons, True, (70, 255, 70), 2, cv2.LINE_AA)
                a, b, c, d = instance["roi"]
                cv2.rectangle(color, (a, b), (c - 1, d - 1), (0, 220, 255), 1)
                _text(
                    color,
                    f"YOLO {instance['index']} {instance['confidence']:.2f}",
                    (a, max(15, b - 4)),
                    (70, 255, 70),
                )
        color_lines = [
            f"Red/blue = configured HSV labels (S>={cfg['color_min_saturation']} V>={cfg['color_min_value']}).",
            f"Yellow = broad color proposals (S>{cfg['color_proposal_min_saturation']} V>{cfg['color_proposal_min_value']}); dim = neither.",
            "Color confirms a target candidate; geometric pose can still be rejected.",
        ]
        if yolo:
            color_lines.append(
                f"YOLO: {len(yolo['instances'])} ROIs; green=mask, yellow=search ROI; L selects ROI geometry."
            )
            if yolo.get("input_rotation_ccw_deg", 0):
                color_lines.append(
                    f"Network input rotated {yolo['input_rotation_ccw_deg']}deg CCW; masks mapped back to RAW pixels."
                )
            for item in yolo["instances"][:3]:
                gate = item.get("color_gate")
                if gate:
                    color_lines.append(
                        f"ROI {item['index']}: red={gate['red_fraction']:.0%} blue={gate['blue_fraction']:.0%} "
                        f"total={gate['coverage_fraction']:.0%}; color {'PASS' if gate['passed'] else 'REJECT'}; poses={item['objects']}"
                    )
        else:
            color_lines.append(
                f"Final color gate: {cfg['require_target_colors']}; >={cfg['color_min_coverage_fraction']:.0%} classified."
            )
        panels.append(_panel(color, STAGES[1], source, color_lines))
        edge = np.zeros_like(base)
        edge[trace.images.get("rgb_edges", np.zeros(valid.shape, np.uint8)) > 0] = (230, 230, 230)
        edge[trace.images.get("depth_edges", np.zeros(valid.shape, np.uint8)) > 0] = (40, 80, 255)
        for face in trace.faces:
            for line in face.get("evidence", {}).get("observed_lines", ()):
                segment = np.rint(line["segment"]).astype(int)
                cv2.line(edge, tuple(segment[0]), tuple(segment[1]), (255, 230, 30), 2, cv2.LINE_AA)
        panels.append(
            _panel(
                edge,
                STAGES[2],
                source,
                [
                    f"White = RGB edges (Canny {cfg['canny_low']}/{cfg['canny_high']}); red = depth jump >{cfg['depth_edge_threshold_m']*1000:.0f}mm.",
                    "Cyan segments = robust fitted CURRENT RGB boundary lines inside the selected face plane.",
                    "A shared cube edge can have continuous depth: no red line does not mean one plane.",
                ],
            )
        )
        if "normals" not in trace.images:
            trace.images["normals"] = _normals(detection)
        pcl = trace.backend["name"] == "pcl"
        normal_lines = [
            "Color encodes local 3D normal direction; black = invalid/unstable neighborhood.",
            "Different face directions should have different colors even at similar Z.",
        ]
        if pcl:
            normal_lines += [
                f"ACTUAL PCL segmentation normals: {cfg['pcl_normal_method']}; stride={cfg['pcl_stride']}.",
                f"Smoothing={cfg['pcl_normal_smoothing_px']} source pixels; angle gate={cfg['pcl_angular_threshold_deg']}deg.",
            ]
        else:
            normal_lines += [
                "Finite differences span 16 source pixels; boundary neighborhoods may be mixed.",
                "DIAGNOSTIC ONLY: NumPy RANSAC does not use these local normals.",
            ]
        panels.append(_panel(trace.images["normals"], STAGES[3], source, normal_lines))
        planes = base // 3
        plane_labels = trace.images.get("plane_labels", np.zeros(valid.shape, np.uint8))
        for i in np.unique(plane_labels):
            if i <= 0:
                continue
            planes[plane_labels == i] = PALETTE[(i - 1) % len(PALETTE)]
        lines = [
            f"{'PCL organized multi-plane' if pcl else 'NumPy sequential RANSAC'}: {len(trace.planes)} regions/attempts; distance <{cfg['plane_threshold_m'] * 1000:.1f}mm."
        ]
        for p in trace.planes[:4]:
            n = p.get("normal")
            lines.append(
                f"P{p['id']} {p.get('status')}"
                + (
                    f" n=({n[0]:.2f},{n[1]:.2f},{n[2]:.2f}) RMS={p['rmse_m'] * 1000:.1f}mm support={p['support']:.0%} patches={p.get('patches', 0)}"
                    if n is not None
                    else f" {p.get('reason', '')}"
                )
            )
        normals = [p["normal"] for p in trace.planes if p.get("normal") is not None]
        if len(normals) >= 2:
            angle = np.rad2deg(np.arccos(np.clip(abs(np.dot(normals[0], normals[1])), 0, 1)))
            lines.append(f"P0/P1 normal angle (unsigned) = {angle:.1f}deg; cube neighbors ideally 90deg.")
        panels.append(_panel(planes, STAGES[4], source, lines))
        candidates = base.copy()
        for f in trace.faces:
            q = np.rint(f["quad"]).astype(np.int32)
            color = (
                (80, 230, 80)
                if f["status"] == "geometry_pass"
                else (60, 80, 255) if f["status"] == "rejected" else (30, 210, 255)
            )
            cv2.polylines(candidates, [q], True, color, 1, cv2.LINE_AA)
            if f["source"] in ("roi_plane_rgb_lines", "bottom_occlusion_model"):
                for i, point in enumerate(q):
                    inferred = f["source"] == "bottom_occlusion_model" and i >= 2
                    marker = (0, 160, 255) if inferred else (255, 230, 30)
                    cv2.circle(candidates, tuple(point), 4, marker, -1)
                    _text(
                        candidates,
                        ("I" if inferred else "O") + str(i),
                        tuple(point + (3, -4)),
                        marker,
                        scale=0.35,
                    )
            _text(candidates, f"F{f['id']}", tuple(q[0]), color)
        passed = sum(f["status"] == "geometry_pass" for f in trace.faces)
        detection_result = trace.result or {}
        color_rejections = detection_result.get("appearance_rejections", [])
        suppressed = detection_result.get("suppressed_hypotheses", [])
        panels.append(
            _panel(
                candidates,
                STAGES[5],
                source,
                [
                    f"{len(trace.faces)} tested faces; geometry pass={passed}. Green=geometry pass; red=rejected.",
                    "Geometry pass is before color/scene selection; it is not a final valid object.",
                    f"THIS detector: final objects={len(detection_result.get('objects', []))}; color rejects={len(color_rejections)}; scene suppressions={len(suppressed)}",
                    f"Selection rejection: {(color_rejections + suppressed)[0].get('reason', 'scene conflict') if color_rejections or suppressed else 'none'}",
                    "O=observed line intersection; I=inferred hidden corner (not a PnP observation). N/P face; L ROI.",
                ],
            )
        )
        detail = base.copy()
        lines = ["No face candidate reached validation."]
        if trace.faces:
            f = trace.faces[face_index % len(trace.faces)]
            cv2.polylines(detail, [np.rint(f["quad"]).astype(np.int32)], True, (20, 220, 255), 2)
            uv, residual = f.get("sample_uv", []), f.get("sample_residual_m", [])
            for pixel, error in zip(uv, residual):
                cv2.circle(
                    detail,
                    tuple(np.rint(pixel).astype(int)),
                    2,
                    (60, 230, 60) if error < cfg["plane_threshold_m"] else (50, 50, 255),
                    -1,
                )
            reason = f.get("reason", "geometry accepted; see final selection in saved JSON")
            lines = [
                f"F{f['id']} {f['source']} inferred={f['inferred']} status={f['status']}",
                f"Reason: {textwrap.shorten(reason, width=180, placeholder='... (full reason in JSON)')}",
            ]
            if "depth_fraction" in f:
                lines.append(
                    f"Depth coverage {f['depth_fraction']:.1%} >= {cfg['min_depth_fraction']:.1%}; valid pixels {f['valid_depth_pixels']} >= {cfg['min_depth_samples']}"
                )
            if "plane_rmse_m" in f:
                lines.append(
                    f"Plane RMS {f['plane_rmse_m']*1000:.2f} <= {cfg['plane_max_rmse_m']*1000:.2f}mm; support {f['plane_support']:.1%} >= {f.get('plane_support_min', cfg['plane_min_support_fraction']):.0%}"
                )
            if "edge_lengths_m" in f:
                lines.append(
                    "3D edges mm: "
                    + ", ".join(f"{v*1000:.1f}" for v in f["edge_lengths_m"])
                    + f"; size tolerance {cfg['size_tolerance_fraction']:.0%}"
                )
            if f.get("partial_metrics"):
                partial = f["partial_metrics"]
                lines.append(
                    f"Partial recovery: visible={partial['visible_depth_fraction']:.0%}; foreground={partial['foreground_depth_fraction']:.0%}; inferred bottom is NOT PnP input."
                )
            if f["trials"]:
                metric_keys = (
                    "reprojection_rms_px", "reprojection_limit_px", "depth_median_residual_m",
                    "normal_error_deg", "upright_tilt_deg",
                )
                complete = [
                    trial for trial in f["trials"]
                    if all(trial.get(key) is not None and np.isfinite(trial[key]) for key in metric_keys)
                ]
                t = next(
                    (trial for trial in f["trials"] if trial.get("accepted")),
                    complete[0] if complete else f["trials"][0],
                )
                if all(t.get(key) is not None and np.isfinite(t[key]) for key in metric_keys):
                    lines.append(
                        f"{t['method']}: reproj {t['reprojection_rms_px']:.2f}/{t['reprojection_limit_px']:.2f}px depth {t['depth_median_residual_m']*1000:.1f}/{cfg['max_depth_residual_m']*1000:.1f}mm"
                    )
                    lines.append(
                        f"Normal error {t['normal_error_deg']:.1f}/{cfg['max_normal_error_deg']:.1f}deg tilt {t['upright_tilt_deg']:.1f}/{cfg['max_upright_tilt_deg']:.1f}deg; all trials saved in JSON"
                    )
                else:
                    # Early rejection has no measured residuals; do not display invented zeros.
                    stage = t.get("stage", "early rejection")
                    reason = t.get("reason") or ", ".join(map(str, t.get("failed_checks", ())))
                    lines.append(f"{t['method']}: {stage}; residuals not evaluated")
                    lines.append(f"Trial reason: {reason or 'see trial diagnostics in JSON'}")
        panels.append(_panel(detail, STAGES[6], source, lines))
    for points in tracks:
        for uv in np.asarray(points).reshape(-1, 2):
            cv2.circle(rgb, tuple(np.rint(uv).astype(int)), 3, (70, 255, 70), -1)
    temporal = result.get("temporal", {})
    failures = temporal.get("tracking_failures", []) + temporal.get("detector_seed_failures", [])
    lines = [
        f"Current output objects={len(result['objects'])}; tracked={temporal.get('output_flow_objects', 0)}; features={sum(len(p) for p in tracks)}",
        f"New detector result={temporal.get('detector_raw_objects', 0) if temporal.get('detector_ran') else '--'}; current RGB-D passed={temporal.get('detector_objects', 0)}; status={result.get('frame_status', 'unknown')}",
        f"Current failures: {'; '.join(failures[:3]) or 'none reported'}",
        f"Color rejections={len(result.get('appearance_rejections', []))}; scene suppressions={len(result.get('suppressed_hypotheses', []))}",
        "Green dots = successfully measured CURRENT image features. Detection panels have a separate frame/age.",
    ]
    if temporal.get("async_detection"):
        age = temporal.get("detector_frame_age_ms")
        lines.insert(
            2,
            f"Background busy={temporal.get('detector_busy', False)}; queued={temporal.get('detector_queue_depth', 0)}/1; "
            f"completion age={round(age) if age is not None else '--'}ms; replaced pending={temporal.get('detector_pending_replaced', 0)}",
        )
    for f in temporal.get("tracking_failure_details", [])[:2]:
        detail = f"{f['size_m']*100:.0f}cm: features {f.get('bidirectional_features', f.get('features_before', 0))}"
        if "feature_coverage" in f:
            detail += f"; spread {f['feature_coverage']:.0%}/{f['feature_coverage_min']:.0%}"
        if "plane_support" in f:
            detail += f"; plane {f['plane_support']:.0%}/{f['plane_support_min']:.0%}"
        elif "depth_fraction" in f:
            detail += f"; depth {f['depth_fraction']:.0%}"
        lines.append(detail)
    panels.append(_panel(rgb, STAGES[7], current, lines))
    return np.stack(panels)


def _display_stages(buffer, lock, version, shown, command, stopping, shape, layout, tuning_shared):
    cv2.setNumThreads(1)
    shared = np.frombuffer(buffer, np.uint8).reshape(shape)
    names = STAGES if layout == "windows" else ("Box4D stages",)
    for i, name in enumerate(names):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(name, 640 if layout == "windows" else 1440, 535 if layout == "windows" else 1200)
        if layout == "windows":
            cv2.moveWindow(name, (i % 3) * 420, (i // 3) * 300)
    tuning = _TuningPanel(tuning_shared)
    seen = 0
    try:
        while not stopping.value:
            panels = None
            if lock.acquire(False):
                try:
                    if version.value != seen:
                        panels = shared.copy()
                        seen = version.value
                finally:
                    lock.release()
            if panels is not None:
                if layout == "windows":
                    for name, panel in zip(names, panels):
                        cv2.imshow(name, panel)
                else:
                    blank = np.zeros_like(panels[0])
                    cells = list(panels) + [blank]
                    grid = np.concatenate(
                        [np.concatenate(cells[i : i + 3], axis=1) for i in (0, 3, 6)], axis=0
                    )
                    cv2.imshow(names[0], grid)
                shown.value = seen
            tuning.poll()
            key = cv2.waitKey(1) & 255
            if key in (ord("["), ord("]")):
                tuning.show_page(tuning.page + (1 if key == ord("]") else -1))
            if key in (ord("q"), 27, ord("s"), ord("S"), ord("n"), ord("p"), ord("l"), ord(" ")):
                command.value = ord("q") if key == 27 else ord("s") if key == ord("S") else key
                if command.value == ord("q"):
                    break
            # GTK builds may return -1 for an unsupported visibility property.
            if seen and (
                any(cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) == 0 for name in names)
                or cv2.getWindowProperty(TUNING_WINDOW, cv2.WND_PROP_VISIBLE) == 0
            ):
                command.value = ord("q")
                break
            time.sleep(0.005)
    finally:
        cv2.destroyAllWindows()


class _SnapshotSaver:
    """Bound memory and keep PNG/NPZ compression off the tracking thread."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="box4d-snapshot")
        self.pending = {}
        # Keep only small descriptions, never exceptions retaining RGB-D packets
        # through their tracebacks. Failures remain inspectable after a retry.
        self.errors = deque(maxlen=8)
        self.failure_count = 0
        self.closed = False
        self.last_submit_error = None

    def _failed(self, path, error):
        self.failure_count += 1
        record = _save_failure("Debug snapshot", path, error)
        self.errors.append(record)
        return record

    def check(self):
        for future, path in tuple(self.pending.items()):
            if future.done():
                del self.pending[future]
                try:
                    future.result()
                except Exception as error:
                    self._failed(path, error)
                else:
                    print(f"Debug snapshot saved: {path}", flush=True)

    def submit(self, path, packet, panels):
        self.last_submit_error = None
        self.check()
        if self.closed:
            self.last_submit_error = self._failed(path, RuntimeError("snapshot saver is closed"))
            return False
        if len(self.pending) >= 2:
            return False
        try:
            future = self.executor.submit(self._write, path, packet, panels)
        except Exception as error:
            self.last_submit_error = self._failed(path, error)
            return False
        self.pending[future] = Path(path)
        return True

    @staticmethod
    def _write(path, packet, panels):
        save_snapshot(path, *packet, panels=panels)

    def close(self):
        if not self.closed:
            self.closed = True
            self.executor.shutdown(wait=True)
        self.check()


def _save_failure(operation, path, error):
    # A bounded string also avoids retaining an unusually large error payload.
    record = dict(path=str(path), error_type=type(error).__name__, message=str(error)[:2000])
    print(
        f"{operation} save failed: {path}: {record['error_type']}: {record['message']}",
        file=sys.stderr,
        flush=True,
    )
    return record


def _atomic_dump_config(path, value):
    """Replace a config only after its complete JSON has been written."""
    path = Path(path)
    temporary = None
    try:
        # Close the temporary handle before reopening/replacing it on Windows.
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        dump_json(temporary, value)
        temporary.replace(path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                _save_failure("Temporary config cleanup", temporary, error)


class DebugSession:
    """Latest-only GUI in a separate process; save exactly the submitted frames."""

    def __init__(self, output, config_path, cfg, layout="windows"):
        self.output = Path(output)
        self.config_path = Path(config_path)
        self.config = cfg
        self.applied_positions = None
        self.face_index = self.detector_index = self.saved = 0
        self.frozen = False
        self.last = None
        self.save_pending = False
        self.save_errors = deque(maxlen=8)
        self.shape = (len(STAGES), 600, 720, 3)
        ctx = mp.get_context("spawn")
        self.buffer = ctx.RawArray("B", int(np.prod(self.shape)))
        self.lock = ctx.Lock()
        self.version = ctx.RawValue("L", 0)
        self.shown = ctx.RawValue("L", 0)
        self.command = ctx.RawValue("i", -1)
        self.stopping = ctx.RawValue("b", False)
        self.tuning_shared = ctx.Array("i", tuning_positions(cfg), lock=True)
        self.process = ctx.Process(
            target=_display_stages,
            args=(
                self.buffer,
                self.lock,
                self.version,
                self.shown,
                self.command,
                self.stopping,
                self.shape,
                layout,
                self.tuning_shared,
            ),
            daemon=True,
        )
        self.process.start()
        self.snapshot_saver = _SnapshotSaver()

    def update(self, frame, result, pipeline):
        if not self.frozen:
            packet = (
                frame,
                deepcopy(result),
                getattr(pipeline, "last_debug", None),
                tuple(getattr(pipeline, "local_debug", ())),
                tuple(p.copy() for p in getattr(pipeline, "debug_tracks", ())),
            )
            self._submit(packet)

    def _submit(self, packet):
        panels = render_stages(*packet, face_index=self.face_index, detector_index=self.detector_index)
        if self.frozen:
            for panel in panels:
                _text(panel, "FROZEN", (615, 21), color=(40, 180, 255), scale=0.52)
        if not self.process.is_alive() or not self.lock.acquire(False):
            return
        try:
            np.copyto(np.frombuffer(self.buffer, np.uint8).reshape(self.shape), panels)
            self.version.value += 1
            self.last = (packet, panels)
        finally:
            self.lock.release()

    def _apply_tuning(self, pipeline):
        with self.tuning_shared.get_lock():
            positions = tuple(int(value) for value in self.tuning_shared[:])
        if positions == self.applied_positions:
            return True
        values = tuning_values(positions)
        # Keep paired controls valid while the user is moving either slider.
        values["canny_low"] = min(values["canny_low"], values["canny_high"] - 1)
        values["canny_high"] = max(values["canny_high"], values["canny_low"] + 1)
        values["min_depth_m"] = min(values["min_depth_m"], values["max_depth_m"] - 0.01)
        values["max_depth_m"] = max(values["max_depth_m"], values["min_depth_m"] + 0.01)
        values["max_depth_samples"] = max(values["max_depth_samples"], values["min_depth_samples"])
        values["track_seed_max_features"] = max(
            values["track_seed_max_features"], values["track_min_features"]
        )
        values["track_min_reseed_features"] = max(
            values["track_min_reseed_features"], values["track_min_features"]
        )
        lock = getattr(pipeline, "config_lock", None)
        if lock is None:
            for key, value in values.items():
                setattr(pipeline.cfg, key, value)
            pipeline.cfg.validate()
        else:
            # Detection can include model warmup and several ROIs. Never make
            # the current-frame tracking loop wait for that background work.
            if not lock.acquire(blocking=False):
                return False
            try:
                for key, value in values.items():
                    setattr(pipeline.cfg, key, value)
                pipeline.cfg.validate()
                pipeline.region_pipelines.clear()
            finally:
                lock.release()
        self.config = pipeline.cfg
        self.applied_positions = positions
        return True

    def _save_config(self):
        value = self.config.to_dict()
        complete = True
        for operation, path in (
            ("Tuned config output copy", self.output / "tuned_config.json"),
            ("Tuned source config", self.config_path),
        ):
            try:
                _atomic_dump_config(path, value)
            except OSError as error:
                self.save_errors.append(_save_failure(operation, path, error))
                complete = False
            else:
                print(f"{operation} saved: {path}", flush=True)
        return complete

    def save(self, pipeline):
        if not self._apply_tuning(pipeline):
            if not self.save_pending:
                print(
                    "Configuration save pending: applying sliders after background detection finishes.",
                    flush=True,
                )
            self.save_pending = True
            return
        self.save_pending = False
        if not self._save_config():
            print(
                "Debug snapshot not queued: configuration was not fully saved. Press S to retry.",
                flush=True,
            )
            return
        if self.last:
            packet, panels = self.last
            path = self.output / "debug" / f"snapshot_{self.saved:04d}_frame_{packet[0].sequence}"
            if self.snapshot_saver.submit(path, packet, panels):
                self.saved += 1
                print(f"Debug snapshot queued: {path}", flush=True)
            elif self.snapshot_saver.last_submit_error is None:
                print(
                    "Debug snapshot busy (2 pending); configuration saved. Press S again after saving finishes.",
                    flush=True,
                )

    def poll(self, pipeline):
        self.snapshot_saver.check()
        applied = self._apply_tuning(pipeline)
        if applied and self.save_pending:
            self.save(pipeline)
        key, self.command.value = self.command.value, -1
        if not self.process.is_alive() and key != ord("q"):
            return ord("q")
        if key == ord(" "):
            self.frozen = not self.frozen
            if self.last:
                self._submit(self.last[0])
        elif key in (ord("n"), ord("p"), ord("l")):
            if key == ord("l"):
                self.detector_index += 1
                self.face_index = 0
            else:
                self.face_index += 1 if key == ord("n") else -1
            if self.last:
                self._submit(self.last[0])
        elif key == ord("s"):
            self.save(pipeline)
        return key

    def close(self):
        self.stopping.value = True
        self.process.join(timeout=1)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        self.snapshot_saver.close()


def save_snapshot(path, frame, result, snapshot, local_snapshots=(), tracks=(), panels=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    if panels is None:
        panels = render_stages(frame, result, snapshot, local_snapshots, tracks)
    for name, panel in zip(STAGES, panels):
        if not cv2.imwrite(str(path / (name[:2] + ".png")), panel):
            raise OSError(f"could not save debug panel {name}")
    np.savez_compressed(path / "current_rgbd.npz", rgb=frame.rgb, depth=frame.depth)
    if snapshot is not None:
        np.savez_compressed(path / "detection_rgbd.npz", rgb=snapshot.frame.rgb, depth=snapshot.frame.depth)
    dump_json(
        path / "measurements.json",
        dict(
            current=dict(
                sequence=frame.sequence,
                stamp_ns=frame.stamp_ns,
                camera=frame.camera.to_dict(),
                result=result,
                tracked_points=tracks,
            ),
            detection=None if snapshot is None else snapshot.report(),
            local_detections=[s.report() for s in local_snapshots],
        ),
    )
