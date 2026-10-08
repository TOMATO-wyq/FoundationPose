#!/usr/bin/env python3
"""Prepare 40 native-resolution frames for human visible-mask annotation."""
import argparse
from pathlib import Path
import json

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, nargs="+", default=[
        Path(__file__).resolve().parents[1] / "outputs" / name
        for name in ("zed_sequence", "d435i_static_sequence", "d435i_dynamic_sequence")])
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()
    if args.count < 2 or args.output.exists():
        parser.error("count>=2 and a new output directory required")
    selected = []
    for i, sequence in enumerate(args.sequences):
        files = sorted((sequence / "rgb").glob("*.png"))
        count = args.count//len(args.sequences) + int(i < args.count % len(args.sequences))
        if len(files) < count:
            raise ValueError(f"Not enough frames in {sequence}")
        for j, index in enumerate(np.linspace(0, len(files)-1, count).round().astype(int)):
            file = files[index]
            if not (sequence / "depth" / file.name).is_file():
                raise FileNotFoundError(f"Missing matched depth for {file}")
            selected.append((sequence.resolve(), file, j))
    args.output.mkdir(parents=True)
    (args.output / "rgb").mkdir()
    (args.output / "masks").mkdir()
    frames = []
    for index, (sequence, file, _) in enumerate(selected):
        key = f"{index:03d}_{sequence.name}_{file.stem}"
        image = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable image: {file}")
        cv2.imwrite(str(args.output / "rgb" / f"{key}.png"), image)
        frames.append({"key": key, "sequence": str(sequence), "rgb_file": file.name,
            "split": "tune" if index % 2 == 0 else "eval",
            "mask_path": str((args.output / "masks" / f"{key}.png").resolve()),
            "reviewed": False, "scenario": "unreviewed"})
    (args.output / "manifest.json").write_text(json.dumps({"frames": frames,
        "note": "Evenly sampled suggestions. Human review must confirm visibility, scenario coverage and labels; SAM output is not truth."}, indent=2))
    print(args.output / "manifest.json")


if __name__ == "__main__":
    main()
