"""YOLO instance segmentation backend; shares visible-depth refinement with SAM."""
from pathlib import Path
import hashlib
import time
from types import SimpleNamespace

import cv2
import numpy as np

from kfs_segmentation import (Candidate, NoTarget, SegmentationResult,
                              cpu_factory_defaults, mask_acceptable, refine_mask)


class YOLOSegmenter:
    def __init__(self, cfg, args):
        import torch
        from ultralytics import YOLO
        path = Path(args.yolo_weights)
        if not path.is_file():
            raise FileNotFoundError(f'YOLO weights not found: {path}')
        self.cfg, self.torch = cfg, torch
        self.refinement, self.prompt_mode = args.depth_refinement, 'yolo'
        with cpu_factory_defaults(torch):
            self.model = YOLO(str(path), task='segment')
        if self.model.task != 'segment':
            raise ValueError('YOLO weights must be an instance segmentation model')
        names = self.model.names
        target = args.yolo_class
        if target.isdecimal():
            self.class_id = int(target)
            if self.class_id not in names:
                raise ValueError(f'Unknown YOLO class ID: {target}; names={names}')
        else:
            matches = [i for i, name in names.items() if name == target]
            if not matches:
                raise ValueError(f'Unknown YOLO class: {target}; names={names}')
            self.class_id = matches[0]
        self.conf, self.imgsz, self.device = args.yolo_conf, args.yolo_imgsz, args.yolo_device
        self.sam = SimpleNamespace(metadata={'mask_mode': 'yolo', 'weights': str(path.resolve()),
            'weights_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'class_id': self.class_id, 'class_name': names[self.class_id]})

    def segment(self, rgb, depth, reuse_embedding=False):
        if rgb.dtype != np.uint8 or rgb.shape[:2] != depth.shape:
            raise ValueError('Expected uint8 RGB and aligned depth')
        started = time.perf_counter()
        with cpu_factory_defaults(self.torch), self.torch.inference_mode():
            result = self.model.predict(source=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                classes=[self.class_id], conf=self.conf, imgsz=self.imgsz,
                device=self.device, retina_masks=True, verbose=False)[0]
            if self.torch.cuda.is_available() and str(self.device) != 'cpu':
                self.torch.cuda.synchronize()
        timings = {'yolo_s': time.perf_counter()-started}
        if result.masks is None or result.boxes is None:
            raise NoTarget('YOLO found no target instance')
        masks = result.masks.data.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        for index in np.argsort(scores)[::-1]:
            raw = masks[index] > .5
            if raw.shape != depth.shape:
                raise RuntimeError('YOLO retina mask is not in input image coordinates')
            if not mask_acceptable(raw, depth, self.cfg):
                continue
            safe = cv2.erode(raw.astype(np.uint8),
                np.ones((self.cfg.erosion_size, self.cfg.erosion_size), np.uint8),
                borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
            if not safe.any():
                continue
            y, x = np.nonzero(raw)
            candidate = Candidate(np.array([x.min(), y.min(), x.max(), y.max()], np.float32),
                raw.copy(), safe, np.empty((0, 2), np.float32), np.empty(0, np.int32))
            started = time.perf_counter()
            try:
                mask, diagnostics = refine_mask(raw, depth, safe, self.cfg) if self.refinement else (raw.copy(), {})
            except NoTarget:
                continue
            timings['depth_refine_s'] = time.perf_counter()-started
            if mask_acceptable(mask, depth, self.cfg):
                diagnostics.update(mask_mode='yolo', yolo_score=float(scores[index]))
                return SegmentationResult(mask, raw, candidate, diagnostics, timings)
        raise NoTarget('YOLO masks failed depth / area checks')
