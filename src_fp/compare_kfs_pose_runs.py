#!/usr/bin/env python3
"""Compare runtime and optional static-scene pose jitter (not pose accuracy)."""
import argparse
from pathlib import Path
import json

import numpy as np


def static_jitter(pose_dir):
    translation, rotation = [], []
    files = sorted(pose_dir.glob("*.txt"))
    previous = None
    for file in files:
        pose = np.loadtxt(file).reshape(4, 4)
        if not np.isfinite(pose).all():
            previous = None
            continue
        try:
            index = int(file.stem)
        except ValueError:
            raise ValueError("Jitter requires numeric per-frame pose filenames")
        if previous is not None and index == previous[0]+1:
            p = previous[1]
            translation.append(float(np.linalg.norm(pose[:3, 3]-p[:3, 3])*1000))
            relative = p[:3, :3].T @ pose[:3, :3]
            rotation.append(float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1)))))
        previous = index, pose
    return {"adjacent_pairs": len(translation),
            "translation_step_p50_mm": float(np.median(translation)) if translation else None,
            "translation_step_p95_mm": float(np.percentile(translation, 95)) if translation else None,
            "rotation_step_p50_deg": float(np.median(rotation)) if rotation else None,
            "rotation_step_p95_deg": float(np.percentile(rotation, 95)) if rotation else None,
            "note": "static scene proxy; rotations not symmetry-adjusted; missing frames never bridged"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, type=Path)
    parser.add_argument("--b", required=True, type=Path)
    parser.add_argument("--static-scene", action="store_true", help="explicitly certify camera AND object remained stationary")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file")
    result = {}
    for label, directory in (("A", args.a), ("B", args.b)):
        runtime = json.loads((directory / "runtime_summary.json").read_text())
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        result[label] = {"runtime": runtime, "processed_frames": len(events),
            "geometry_valid_frames": sum(e["valid_pose"] for e in events),
            "warning": "geometry validity is not ground-truth pose success"}
        if args.static_scene:
            result[label]["static_jitter"] = static_jitter(directory / "ob_in_cam")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(args.output)


if __name__ == "__main__":
    main()
