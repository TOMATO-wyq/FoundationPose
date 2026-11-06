"""Segmentation diagnostics; windows retain the last initialization frame."""
import cv2
import numpy as np


def mask_overlay(rgb, mask, label):
    vis = rgb.copy()
    vis[mask] = (.55*vis[mask] + .45*np.array([0, 255, 0])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 0), 1)
    cv2.putText(vis, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def show_segmentation(result, rgb, frame_key, refinement):
    mode = result.diagnostics.get('mask_mode', 'sam')
    tag = f'frame {frame_key} (last segmentation)'
    views = {}
    if mode == 'sam':
        coarse = mask_overlay(rgb, result.candidate.coarse, f'OpenCV / {tag}')
        box = result.candidate.box.astype(int)
        cv2.rectangle(coarse, tuple(box[:2]), tuple(box[2:]), (255, 255, 0), 1)
        for point, label in zip(result.candidate.points, result.candidate.labels):
            cv2.circle(coarse, tuple(point.astype(int)), 3,
                       (0, 255, 0) if label else (255, 0, 0), -1)
        views['KFS debug - OpenCV prompts'] = coarse
    views[f'KFS debug - {mode.upper()} raw mask'] = mask_overlay(
        rgb, result.raw, f'{mode.upper()} raw / {tag}')
    refined = mask_overlay(rgb, result.mask,
                           f'Depth {"refined" if refinement else "OFF"} / {tag}')
    # Blue = added visible pixels, red = removed pixels (RGB coordinates).
    refined[result.mask & ~result.raw] = (0, 100, 255)
    refined[result.raw & ~result.mask] = (255, 60, 60)
    cv2.putText(refined, 'blue: added / red: removed', (8, rgb.shape[0]-8),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    views['KFS debug - Depth result'] = refined
    for name, view in views.items():
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
