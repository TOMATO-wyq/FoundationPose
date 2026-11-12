"""YOLO instance masks guide the existing face/depth/PnP pipeline.

Each instance gets its own ROI and original calibrated image coordinates.
The segmentation boundary is never passed to PnP as a set of cube corners.
"""

from collections import Counter
from copy import deepcopy
import threading
import time

import cv2
import numpy as np

from .appearance import classify_hsv
from .segmentation import YoloRegions, instance_colors, upright_quarter_turns
from .edges import rgb_depth_edges
from .io import Frame
from .pose import Pipeline
from .results import empty_result, merge_region, refresh_detection_selection
from .scene import select_scene


def _face_mask_fraction(obj, mask):
    fractions = []
    for face in obj["faces"]:
        quad = face.get("evidence", {}).get("visible_outline_quad", face["quad"])
        sample = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(sample, np.rint(quad).astype(np.int32), 1)
        sample = cv2.erode(sample, np.ones((3, 3), np.uint8)).astype(bool)
        fractions.append(float(mask[sample].mean()) if sample.any() else 0.0)
    return min(fractions, default=0.0)


class YoloGuidedPipeline:
    """One classical branch for other sizes, one existing Pipeline per mask."""

    def __init__(self, cfg, debug=False, frontend=None):
        self.cfg = cfg.validate()
        self.debug_enabled = debug
        self.frontend = frontend if frontend is not None else YoloRegions(cfg)
        self.config_lock = threading.RLock()
        self.region_pipelines = {}
        self.last_debug = None
        self.local_debug = ()
        self.region_reports = []
        self._config_snapshot = None
        self.full = None
        self._ray_cache = None
        self._sync_config()

    def _geometry_config(self, sizes, roi=None):
        cfg = deepcopy(self.cfg)
        cfg.sizes_m = tuple(sizes)
        cfg.two_color_sizes_m = tuple(s for s in cfg.two_color_sizes_m if s in sizes)
        cfg.yolo_weights = None
        cfg.temporal_recovery = False
        cfg.roi_xyxy = self.cfg.roi_xyxy if roi is None else list(roi)
        # 父配置已在入口校验；Pipeline 创建时再校验派生配置，无需重复校验两次。
        return cfg

    def _sync_config(self):
        # Config 只有标量和小列表，dataclass 按值比较即可。快照独立于滑杆原地修改。
        if self.cfg != self._config_snapshot:
            sizes = [s for s in self.cfg.sizes_m if s not in self.cfg.yolo_sizes_m]
            self.full = Pipeline(self._geometry_config(sizes), debug=self.debug_enabled) if sizes else None
            self.region_pipelines.clear()
            self._ray_cache = None
            self._config_snapshot = deepcopy(self.cfg)

    def _expanded_instance(self, frame, instance):
        """给线拟合保留边界余量；颜色占比仍然只在实例掩膜内统计。"""
        if not self.cfg.yolo_roi_color_geometry:
            return instance
        yy, xx = np.nonzero(instance["mask"])
        pad = max(
            self.cfg.yolo_roi_padding_px,
            int(np.ceil(max(np.ptp(xx) + 1, np.ptp(yy) + 1) * self.cfg.yolo_roi_padding_fraction)),
        )
        left, top, right, bottom = instance["roi"]
        roi = [
            max(0, min(left, int(xx.min()) - pad)),
            max(0, min(top, int(yy.min()) - pad)),
            min(frame.camera.width, max(right, int(xx.max()) + 1 + pad)),
            min(frame.camera.height, max(bottom, int(yy.max()) + 1 + pad)),
        ]
        if self.cfg.roi_xyxy is not None:
            a, b, c, d = self.cfg.roi_xyxy
            roi = [max(a, roi[0]), max(b, roi[1]), min(c, roi[2]), min(d, roi[3])]
        return dict(instance, roi=roi)

    def _region_pipeline(self, roi):
        """缓存只保存少量几何分支，不能随移动的 ROI 无限增长。"""
        key = tuple(roi)
        if key not in self.region_pipelines:
            if len(self.region_pipelines) >= self.cfg.yolo_max_instances * 4:
                self.region_pipelines.pop(next(iter(self.region_pipelines)))
            cfg = self._geometry_config(self.cfg.yolo_sizes_m, roi)
            cfg.max_objects = 1
            pipeline = Pipeline(cfg, configure_opencv=False, debug=self.debug_enabled)
            if self.cfg.yolo_roi_color_geometry:
                from .roi_faces import RoiFaceDetector

                pipeline.detector = RoiFaceDetector(cfg)
            self.region_pipelines[key] = pipeline
        return self.region_pipelines[key]

    def _run_region(self, frame, instance):
        pipeline = self._region_pipeline(instance["roi"])
        pipeline.detector._ray_cache = self._ray_cache
        if self.cfg.yolo_roi_color_geometry:
            # 掩膜只限制平面种子。验收保留完整原始深度，才能判断下边缘是否被遮挡。
            pipeline.detector.instance_mask = instance["mask"]
            selected = frame
        else:
            # 兼容旧的“掩膜外深度清零”模式，由配置显式选择。
            depth = frame.depth.copy()
            depth[~instance["mask"]] = 0
            selected = Frame(
                frame.rgb, depth, frame.camera, frame.stamp_ns, frame.sequence, frame.received_monotonic_ns,
                camera_motion=frame.camera_motion,
            )
        child, _ = pipeline.process(selected)
        self._ray_cache = pipeline.detector._ray_cache
        return pipeline, child

    def _filter_instance_objects(self, frame, instance, info, child, class_name):
        """几何通过后再检查可见面是否落在实例上，防止邻近物体借用此 ROI。"""
        accepted = []
        for obj in child["objects"]:
            fraction = _face_mask_fraction(obj, instance["mask"])
            if fraction < self.cfg.yolo_min_face_mask_fraction:
                info.setdefault("mask_rejections", []).append(dict(face_mask_fraction=fraction))
                child["rejected_faces"].append(
                    dict(
                        quad=obj["faces"][0]["quad"],
                        source="yolo_instance_gate",
                        reason="visible face outside YOLO instance mask",
                        face_mask_fraction=fraction,
                        required_fraction=self.cfg.yolo_min_face_mask_fraction,
                    )
                )
                continue
            obj["segmentation"] = dict(
                instance=info["index"],
                confidence=instance["score"],
                roi=list(instance["roi"]),
                polygons=instance["polygons"],
                stamp_ns=frame.stamp_ns,
                face_mask_fraction=fraction,
                class_name=class_name,
            )
            accepted.append(obj)
        info["objects"] = len(accepted)
        info["rejected_reason_counts"] = dict(Counter(f["reason"] for f in child["rejected_faces"]))
        return accepted

    def process(self, frame, *, config=None):
        with self.config_lock:
            if config is not None:
                self.cfg = config
                if isinstance(self.frontend, YoloRegions):
                    self.frontend.cfg = config
            return self._process(frame)

    def _process(self, frame):
        begin = time.perf_counter()
        self._sync_config()
        instances = self.frontend.predict(frame)  # First stage, before geometric search.
        yolo_ms = (time.perf_counter() - begin) * 1000
        traces = []
        if self.full is not None:
            result, edges = self.full.process(frame)
            self._ray_cache = self.full.detector._ray_cache
            if self.full.last_debug is not None:
                traces.append(self.full.last_debug)
        else:
            result = empty_result(frame, self.cfg)
            edges = rgb_depth_edges(frame, self.cfg)[2]
        metadata = dict(
            getattr(self.frontend, "metadata", {}),
            sequence=frame.sequence,
            stamp_ns=frame.stamp_ns,
            inference_ms=yolo_ms,
            sizes_m=list(self.cfg.yolo_sizes_m),
            instances=[],
        )
        self.region_reports = []
        labels = (
            classify_hsv(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV), self.cfg)
            if self.cfg.yolo_roi_color_geometry
            else None
        )
        for index, instance in enumerate(instances):
            instance = self._expanded_instance(frame, instance)
            roi = instance["roi"]
            info = dict(
                index=index,
                roi=list(roi),
                confidence=instance["score"],
                mask_pixels=instance["mask_pixels"],
                polygons=instance["polygons"],
                objects=0,
                rejected_reason_counts={},
            )
            if labels is not None:
                info["color_gate"] = instance_colors(labels, instance, self.cfg)
                info["candidate_confirmed"] = info["color_gate"]["passed"]
                if not info["candidate_confirmed"]:
                    info["skipped"] = "instance_red_blue_color_gate"
                    metadata["instances"].append(info)
                    self.region_reports.append(
                        dict(
                            roi=list(roi),
                            objects=0,
                            yolo_instance=index,
                            color_gate=info["color_gate"],
                            skipped=info["skipped"],
                        )
                    )
                    if self.debug_enabled:
                        from .debug import DetectionDebug

                        trace = DetectionDebug(frame, self._geometry_config(self.cfg.yolo_sizes_m, roi))
                        trace.images["color_labels"] = labels
                        trace.finish(empty_result(frame, self.cfg))
                        trace.result["yolo"] = dict(metadata, instances=[info])
                        traces.append(trace)
                    continue
            pipeline, child = self._run_region(frame, instance)
            accepted = self._filter_instance_objects(
                frame, instance, info, child, metadata.get("class_name", self.cfg.yolo_class)
            )
            result["objects"].extend(merge_region(result, child, index, objects=accepted))
            metadata["instances"].append(info)
            self.region_reports.append(
                dict(
                    roi=list(roi),
                    objects=info["objects"],
                    yolo_instance=index,
                    confidence=instance["score"],
                    timing=child["timing"],
                )
            )
            if pipeline.last_debug is not None:
                pipeline.last_debug.result["objects"] = deepcopy(accepted)
                pipeline.last_debug.result["yolo"] = dict(metadata, instances=[info])
                traces.append(pipeline.last_debug)
        # ROI coordinates move each detection. Retain historical cache entries
        # for their configuration/rays, without retaining old full RGB-D debug
        # images for every past ROI. Active source traces remain independently owned.
        active_rois = {tuple(item["roi"]) for item in metadata["instances"]}
        for roi, pipeline in self.region_pipelines.items():
            if roi not in active_rois:
                pipeline.last_debug = None
        result["objects"], suppressed = select_scene(
            result["objects"], self.cfg.max_objects, self.cfg.up_camera
        )
        refresh_detection_selection(result)
        result["suppressed_hypotheses"].extend(suppressed)
        result["valid_object_count"] = len(result["objects"])
        result["frame_status"] = (
            "observed"
            if result["objects"]
            else "partial_or_rejected" if result["rejected_faces"] or instances else "no_target"
        )
        result["yolo"] = metadata
        result["timing"].update(yolo_ms=yolo_ms, processing_ms=(time.perf_counter() - begin) * 1000)
        self.last_debug = traces[0] if traces else None
        self.local_debug = tuple(traces[1:])
        if self.last_debug is not None:
            self.last_debug.result["yolo"] = deepcopy(metadata)
        return result, edges


def make_detection_pipeline(cfg, debug=False):
    return YoloGuidedPipeline(cfg, debug=debug) if cfg.yolo_weights else Pipeline(cfg, debug=debug)
