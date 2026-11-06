"""Identical current-image boundary evidence for detection and tracking."""

import cv2
import numpy as np

from .native import geometry_kernel


def rgb_depth_edges(frame, cfg, gray=None, hsv=None):
    gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY) if gray is None else gray
    hsv = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2HSV) if hsv is None else hsv
    rgb = cv2.Canny(gray, cfg.canny_low, cfg.canny_high)
    rgb |= cv2.Canny(hsv[:, :, 2], cfg.canny_low, cfg.canny_high)
    z = frame.depth
    kernel = geometry_kernel(cfg.geometry_backend)
    if kernel is not None:
        depth = kernel.depth_edges(z, cfg.min_depth_m, cfg.max_depth_m, cfg.depth_edge_threshold_m)
    else:
        valid = np.isfinite(z) & (z >= cfg.min_depth_m) & (z <= cfg.max_depth_m)
        depth = np.zeros(z.shape, np.uint8)
        jump_x = valid[:, 1:] & valid[:, :-1] & (np.abs(z[:, 1:] - z[:, :-1]) > cfg.depth_edge_threshold_m)
        jump_y = valid[1:] & valid[:-1] & (np.abs(z[1:] - z[:-1]) > cfg.depth_edge_threshold_m)
        depth[:, 1:][jump_x] = 255
        depth[1:][jump_y] = 255
    if cfg.roi_xyxy is not None:
        x0, y0, x1, y1 = cfg.roi_xyxy
        if x1 > frame.camera.width or y1 > frame.camera.height:
            raise ValueError("configured ROI exceeds actual image dimensions")
        roi = np.zeros_like(rgb)
        roi[y0:y1, x0:x1] = 255
        rgb &= roi
        depth &= roi
    return rgb, depth, rgb | depth
