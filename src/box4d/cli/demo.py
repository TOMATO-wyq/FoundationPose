#!/usr/bin/env python3
"""Generate a reproducible low-camera synthetic RGB-D test dataset on CPU."""
import argparse
from pathlib import Path
import sys

# Keep direct invocation of this organized entry point equivalent to -m.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from box4d.io import DatasetWriter, dump_json
from box4d.synthetic import Renderer, default_camera, demo_scenes, reference_stacking_scenes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--frames-per-scene", type=int, default=3)
    p.add_argument("--depth-noise-mm", type=float, default=1.)
    p.add_argument("--holes", type=float, default=.005)
    p.add_argument("--suite", choices=("basic", "stacking"), default="basic")
    args = p.parse_args()
    if args.frames_per_scene < 1 or args.depth_noise_mm < 0 or not 0 <= args.holes < .5:
        p.error("invalid scene count or depth noise options")
    camera = default_camera()
    renderer = Renderer(camera)
    writer = DatasetWriter(args.output, camera, [0., -1., 0.], synthetic=True,
                           metadata=dict(camera_height_m=.35, nominal_distance_m=1.8 if args.suite == "stacking" else 1.2,
                                         suite=args.suite,
                                         description="CPU pinhole renderer; low camera, visible sides; not sensor measurements"))
    truth, complete = [], False
    try:
        sequence = 0
        scenes = reference_stacking_scenes() if args.suite == "stacking" else demo_scenes()
        for scene, boxes in scenes:
            for _ in range(args.frames_per_scene):
                frame, objects, _ = renderer.render(boxes, sequence, args.depth_noise_mm / 1000, args.holes)
                writer.write(frame)
                truth.append(dict(stamp_ns=frame.stamp_ns, scene=scene, objects=objects))
                sequence += 1
        dump_json(args.output / "truth.json", truth)
        complete = True
    finally:
        writer.close(complete=complete, synthetic=True)
    print(f"Generated {sequence} synthetic frames in {args.output}; not real-camera validation")


if __name__ == "__main__":
    main()
