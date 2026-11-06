"""YOLO-seg 前端和颜色身份判定；输出始终回到原始 RGB-D 像素坐标。

网络可以转正输入图，几何分支的图像、深度和内参始终保持原样。
Torch/Ultralytics 仅在创建真实前端时加载，普通大方块程序无需安装它们。
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np


def upright_quarter_turns(up_camera):
    """Normalize camera roll for the network; geometry stays in the raw frame.

    An optical +X UP needs one counterclockwise turn to point towards image -Y.
    Near a top-down view gravity has no reliable image-plane direction.
    """
    if up_camera is None:
        return 0
    x, y, z = np.asarray(up_camera, float)
    if np.hypot(x, y) < 0.25:
        return 0
    return int(np.argmax([-y, x, y, -x]))


def instance_colors(labels, instance, cfg):
    """Measure identity on the instance; padding does not dilute its colors."""
    selected = labels[instance["mask"]]
    count = len(selected)
    red = int(np.count_nonzero(selected == 1))
    blue = int(np.count_nonzero(selected == -1))
    # 前端只输出面积达标的非空实例；这里不再用 max(1, count) 掩盖契约错误。
    coverage = (red + blue) / count
    red_fraction, blue_fraction = red / count, blue / count
    a, b, c, d = instance["roi"]
    roi = labels[b:d, a:c]
    failures = []
    if red + blue < cfg.color_min_samples:
        failures.append("too_few_colored_pixels")
    if coverage < cfg.yolo_color_min_coverage_fraction:
        failures.append("red_blue_coverage")
    if min(red_fraction, blue_fraction) < cfg.yolo_color_min_each_fraction:
        failures.append("missing_red_or_blue")
    return dict(
        passed=not failures,
        failures=failures,
        mask_pixels=count,
        red_pixels=red,
        blue_pixels=blue,
        red_fraction=red_fraction,
        blue_fraction=blue_fraction,
        coverage_fraction=coverage,
        min_coverage_fraction=cfg.yolo_color_min_coverage_fraction,
        min_each_fraction=cfg.yolo_color_min_each_fraction,
        roi_red_fraction=float(np.mean(roi == 1)),
        roi_blue_fraction=float(np.mean(roi == -1)),
    )


class YoloRegions:
    def __init__(self, cfg):
        # Keep torch optional for the original CPU-only configuration.
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "YOLO-seg needs ultralytics and torch. Run box4d_ros.sh with "
                "BOX4D_PYTHON=/home/tomato/code/box_seg_annotation/env/bin/python"
            ) from error
        path = Path(cfg.yolo_weights).expanduser()
        if path.is_dir():
            path = path / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"YOLO segmentation weights not found: {path}")
        self.cfg = cfg
        self.torch = torch
        self.model = YOLO(str(path), task="segment")
        # Ultralytics selects its own CPU thread count while setting up the
        # predictor. Apply our bound after setup, before actual inference.
        self.model.add_callback("on_predict_start", self._limit_cpu_threads)
        if self.model.task != "segment":
            raise ValueError("yolo_weights must contain an instance segmentation model")
        names = self.model.names
        target = cfg.yolo_class
        matches = [int(i) for i, name in names.items() if name == target or str(i) == target]
        if len(matches) != 1:
            raise ValueError(f"Unknown/ambiguous yolo_class={target}; names={names}")
        self.class_id = matches[0]
        self.metadata = dict(
            weights=str(path.resolve()),
            weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            class_id=self.class_id,
            class_name=names[self.class_id],
        )

    def _limit_cpu_threads(self, predictor):
        if predictor.device.type == "cpu" and self.torch.get_num_threads() != self.cfg.yolo_cpu_threads:
            self.torch.set_num_threads(self.cfg.yolo_cpu_threads)

    def predict(self, frame):
        cfg = self.cfg
        turns = upright_quarter_turns(cfg.up_camera) if cfg.yolo_upright_inference else 0
        rgb = np.ascontiguousarray(np.rot90(frame.rgb, turns)) if turns else frame.rgb
        result = self.model.predict(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            classes=[self.class_id],
            conf=cfg.yolo_conf,
            imgsz=cfg.yolo_imgsz,
            device=cfg.yolo_device,
            max_det=cfg.yolo_max_instances,
            retina_masks=True,
            verbose=False,
        )[0]
        self.metadata.update(
            inference_device=str(self.model.predictor.device),
            cpu_threads=self.torch.get_num_threads(),
            input_rotation_ccw_deg=turns * 90,
            network_input_shape=list(rgb.shape[:2]),
        )
        if result.masks is None or result.boxes is None:
            return []
        masks = result.masks.data.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        if masks.shape[1:] != rgb.shape[:2]:
            raise RuntimeError("YOLO masks must match the network source image dimensions")
        if turns:
            masks = np.rot90(masks, -turns, axes=(1, 2))
        if masks.shape[1:] != frame.depth.shape or len(masks) != len(scores):
            raise RuntimeError("YOLO masks must match the original calibrated RGB-D image dimensions")
        instances = []
        for index in np.argsort(scores)[::-1]:
            mask = masks[index] > 0.5
            if cfg.roi_xyxy is not None:
                a, b, c, d = cfg.roi_xyxy
                allowed = np.zeros(mask.shape, bool)
                allowed[b:d, a:c] = True
                mask &= allowed
            yy, xx = np.nonzero(mask)
            if len(xx) < cfg.min_face_area_px:
                continue
            pad = cfg.yolo_roi_padding_px
            roi = [
                max(0, int(xx.min()) - pad),
                max(0, int(yy.min()) - pad),
                min(frame.camera.width, int(xx.max()) + 1 + pad),
                min(frame.camera.height, int(yy.max()) + 1 + pad),
            ]
            if cfg.roi_xyxy is not None:
                a, b, c, d = cfg.roi_xyxy
                roi = [max(a, roi[0]), max(b, roi[1]), min(c, roi[2]), min(d, roi[3])]
            contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            polygons = [
                cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).tolist() for c in contours if len(c) >= 3
            ]
            instances.append(
                dict(mask=mask, roi=roi, score=float(scores[index]), mask_pixels=len(xx), polygons=polygons)
            )
        return instances
