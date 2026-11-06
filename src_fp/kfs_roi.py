"""ROI bounds use exclusive right/bottom edges; poses stay in camera coordinates."""
import numpy as np


def project_roi(vertices, pose, k, shape, margin=.3):
    xyz = np.asarray(vertices) @ pose[:3, :3].T + pose[:3, 3]
    if not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= .001):
        return None
    uvw = xyz @ k.T
    uv = uvw[:, :2]/uvw[:, 2:]
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    size = np.maximum(hi-lo+1, 1)
    lo = np.floor(lo-margin*size).astype(int)
    hi = np.ceil(hi+margin*size+1).astype(int)
    h, w = shape[:2]
    lo = np.maximum(lo, [0, 0])
    hi = np.minimum(hi, [w, h])
    if np.any(hi-lo < 16):
        return None
    return [int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])]


def crop_frame(rgb, depth, k, roi):
    h, w = depth.shape
    if len(roi) != 4 or any(not isinstance(x, int) for x in roi):
        raise ValueError('ROI must contain four integer bounds')
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('ROI outside frame')
    cropped_k = k.copy()
    cropped_k[0, 2] -= x0
    cropped_k[1, 2] -= y0
    return (np.ascontiguousarray(rgb[y0:y1, x0:x1]),
            np.ascontiguousarray(depth[y0:y1, x0:x1]), cropped_k)
