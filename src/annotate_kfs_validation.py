#!/usr/bin/env python3
"""Human-visible mask annotation; supports multiple components and occlusion cuts."""
import argparse
from pathlib import Path
import json

import cv2
import numpy as np


def edit_mask(rgb, initial=None):
    mask = np.zeros(rgb.shape[:2], np.uint8) if initial is None else initial.copy()
    points, history = [], []
    window = "Visible KFS: A:add X:subtract U:undo C:clear Enter:save Q:stop"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    def click(event, x, y, flags, context):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
    cv2.setMouseCallback(window, click)
    try:
        while True:
            overlay = rgb.copy()
            overlay[mask > 0] = (overlay[mask > 0]*.5 + (0, 100, 0)).astype(np.uint8)
            if len(points) > 1:
                cv2.polylines(overlay, [np.array(points, np.int32)], False, (0, 255, 255), 2)
            for point in points:
                cv2.circle(overlay, point, 3, (0, 255, 255), -1)
            cv2.imshow(window, overlay)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("a"), ord("x")) and len(points) >= 3:
                history.append(mask.copy())
                cv2.fillPoly(mask, [np.array(points, np.int32)], 255 if key == ord("a") else 0)
                points.clear()
            elif key == ord("u"):
                if points: points.pop()
                elif history: mask = history.pop()
            elif key == ord("c"):
                history.append(mask.copy())
                mask[:] = 0
                points.clear()
            elif key in (10, 13) and not points:
                return mask
            elif key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return None
    finally:
        cv2.destroyWindow(window)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--split", choices=("tune", "eval", "all"), default="all")
    parser.add_argument("--scenario", choices=("normal", "tilted", "occluded", "depth_holes", "edge", "distractor", "absent"), default="normal")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for frame in manifest["frames"]:
        if frame["reviewed"] or (args.split != "all" and frame["split"] != args.split):
            continue
        rgb = cv2.imread(str(Path(frame["sequence"]) / "rgb" / frame["rgb_file"]))
        print(f"Annotating {frame['key']}; draw visible components with A, remove occluders with X", flush=True)
        mask = edit_mask(rgb)
        if mask is None:
            break
        if not cv2.imwrite(frame["mask_path"], mask):
            raise IOError(f"Cannot save {frame['mask_path']}")
        frame["reviewed"], frame["scenario"] = True, args.scenario
        args.manifest.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
