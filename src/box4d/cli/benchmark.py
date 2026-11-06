#!/usr/bin/env python3
"""Benchmark compute on preloaded frames; explicitly NOT live sensor FPS."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

# Keep direct invocation of this organized entry point equivalent to -m.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from box4d.config import Config
from box4d.io import Dataset, dump_json
from box4d.metrics import environment, evaluate, summarize
from box4d.yolo import make_detection_pipeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "config/box4d_cpu.json")
    p.add_argument("--method", choices=("depth", "pnp", "hybrid", "all"), default="all")
    p.add_argument("--up-camera", type=float, nargs=3)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument(
        "--sample-uniform",
        action="store_true",
        help="Sample across the whole clip, rather than its first frames",
    )
    p.add_argument("--geometry-backend", choices=("auto", "numpy", "cpp"), default="auto")
    p.add_argument("--no-refine", action="store_true")
    args = p.parse_args()
    if args.repeat < 1 or args.max_frames < 1:
        p.error("repeat and max-frames must be positive")
    dataset = Dataset(args.dataset)
    count = min(len(dataset), args.max_frames)
    indices = np.linspace(0, len(dataset) - 1, count).astype(int) if args.sample_uniform else range(count)
    frames = [dataset.frame(i) for i in indices]
    truth = (
        json.loads((args.dataset / "truth.json").read_text())
        if (args.dataset / "truth.json").exists()
        else None
    )
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    methods = ("depth", "pnp", "hybrid") if args.method == "all" else (args.method,)
    for method in methods:
        cfg = Config.load(args.config, method=method)
        cfg.geometry_backend = args.geometry_backend
        if args.up_camera is not None:
            cfg.up_camera = args.up_camera
        elif cfg.up_camera is None:
            cfg.up_camera = dataset.manifest.get("up_camera")
        cfg.refine_4d = method == "hybrid" and not args.no_refine
        cfg.validate()
        pipeline = make_detection_pipeline(cfg)
        for frame in frames[: min(4, len(frames))]:
            pipeline.process(frame)
        results = []
        first_pass = []
        begin = time.perf_counter()
        for iteration in range(args.repeat):
            for frame in frames:
                result, _ = pipeline.process(frame)
                results.append(result)
                if iteration == 0:
                    first_pass.append(result)
        elapsed = time.perf_counter() - begin
        report = dict(
            method=method,
            config=cfg.to_dict(),
            environment=environment(cfg),
            synthetic=dataset.manifest.get("synthetic", False),
            unique_recorded_frames=len(frames),
            repeat=args.repeat,
            compute_iterations=len(results),
            wall_compute_fps=len(results) / elapsed,
            **summarize(results)
        )
        if truth is not None:
            evaluation = evaluate(first_pass, truth, cfg)
            report["evaluation"] = {k: v for k, v in evaluation.items() if k != "frames"}
            scene_by_stamp = {r["stamp_ns"]: r["scene"] for r in truth}
            report["by_scene"] = {
                scene: summarize([r for r in results if scene_by_stamp.get(r["stamp_ns"]) == scene])
                for scene in dict.fromkeys(scene_by_stamp.values())
            }
        summaries.append(report)
        print(
            json.dumps(
                dict(
                    method=method,
                    wall_compute_fps=report["wall_compute_fps"],
                    processing_ms=report["processing_ms"],
                    evaluation=report.get("evaluation"),
                ),
                indent=2,
            ),
            flush=True,
        )
    dump_json(
        args.output / "summary.json",
        dict(
            kind="offline_compute_only",
            summaries=summaries,
            disclaimer="Repeated recorded/synthetic frames measure computation, not independent live camera FPS or real scene accuracy.",
        ),
    )


if __name__ == "__main__":
    main()
