"""有效深度加权的双边空间滤波；输入/输出均为米，不补洞、不跨帧。"""
import numpy as np


def bilateral_depth(depth, radius=2, spatial_sigma=2.0, range_sigma_m=0.015):
    if radius < 1 or radius > 8 or spatial_sigma <= 0 or range_sigma_m <= 0:
        raise ValueError('radius必须1–8，两个sigma必须大于0')
    src = np.asarray(depth, dtype=np.float32)
    if src.ndim != 2:
        raise ValueError('depth必须为二维深度图')
    valid = np.isfinite(src) & (src >= .001)
    clean = np.where(valid, src, 0)
    h, w = src.shape
    padded = np.pad(clean, radius, mode='constant')
    mask = np.pad(valid, radius, mode='constant')
    total = np.zeros_like(clean)
    weights = np.zeros_like(clean)
    # 仅平均有效邻居；原本没有深度的像素继续为0，防止背景填入目标轮廓。
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = padded[radius+dy:radius+dy+h, radius+dx:radius+dx+w]
            neighbor_valid = mask[radius+dy:radius+dy+h, radius+dx:radius+dx+w]
            delta = neighbor - clean
            weight = np.exp(-.5 * ((dx*dx+dy*dy)/(spatial_sigma**2) + (delta/range_sigma_m)**2))
            weight *= valid & neighbor_valid
            total += weight * neighbor
            weights += weight
    output = np.zeros_like(clean)
    np.divide(total, weights, out=output, where=valid & (weights > 0))
    return np.ascontiguousarray(output)
