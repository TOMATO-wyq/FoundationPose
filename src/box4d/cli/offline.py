#!/usr/bin/env python3
"""Replay a recorded RGB-D dataset with the CPU box pose pipeline."""
import argparse
import json
from pathlib import Path
import sys

import cv2

# Keep direct invocation of this organized entry point equivalent to -m.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from box4d.config import Config
from box4d.io import Dataset, RunWriter
from box4d.metrics import environment, evaluate, summarize
from box4d.temporal import make_pipeline
from box4d.tracking import Tracker
from box4d.visualization import overlay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", choices=("big", "small"), help="recognize and track only big (0.35 m) or small (0.20 m) cubes")
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "config/box4d_cpu.json")
    p.add_argument("--method", choices=("depth", "pnp", "hybrid"))
    p.add_argument("--plane-backend", choices=("numpy", "pcl"))
    p.add_argument("--geometry-backend", choices=("auto", "numpy", "cpp"))
    p.add_argument(
        "--up-camera",
        type=float,
        nargs=3,
        help="physical UP in RGB optical coordinates; explicit calibration",
    )
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--yolo-weights", type=Path)
    p.add_argument("--yolo-device")
    p.add_argument("--yolo-conf", type=float)
    p.add_argument("--no-tracking", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--save-debug", action="store_true")
    p.add_argument(
        "--save-stages",
        action="store_true",
        help="save eight stage panels, source RGB-D and measured checks for every replay frame",
    )
    p.add_argument("--video", action="store_true")
    p.add_argument(
        "--display-smoothed",
        action="store_true",
        help="filtered debug/video projection, with raw poses still logged",
    )
    p.add_argument(
        "--video-fps", type=float, default=10.0, help="offline playback rate, not processing/camera FPS"
    )
    args = p.parse_args()
    if args.video_fps <= 0 or (args.max_frames is not None and args.max_frames < 1):
        p.error("invalid frame limit/playback rate")
    dataset = Dataset(args.dataset, args.allow_partial)
    truth_path = args.dataset / "truth.json"
    truth_rows = json.loads(truth_path.read_text()) if truth_path.is_file() else None
    scenes = {int(row["stamp_ns"]): row.get("scene", "") for row in truth_rows or []}
    cfg = Config.load(
        args.config,
        size=args.size,
        method=args.method,
        plane_backend=args.plane_backend,
        geometry_backend=args.geometry_backend,
        yolo_weights=None if args.yolo_weights is None else str(args.yolo_weights),
        yolo_device=args.yolo_device,
        yolo_conf=args.yolo_conf,
        refine_4d=False if args.no_refine else None,
        tracking=False if args.no_tracking else None,
    )
    if args.no_tracking:
        cfg.temporal_recovery = False
    if args.display_smoothed and not cfg.tracking:
        p.error("--display-smoothed requires tracking")
    if args.up_camera is not None:
        cfg.up_camera = args.up_camera
    elif cfg.up_camera is None:
        cfg.up_camera = dataset.manifest.get("up_camera")
    cfg.validate()
    pipeline, tracker = make_pipeline(cfg, debug=args.save_stages), Tracker(cfg)
    writer = RunWriter(
        args.output,
        cfg,
        dict(
            dataset=str(args.dataset),
            synthetic=dataset.manifest.get("synthetic", False),
            environment=environment(cfg),
            video_playback_fps=args.video_fps if args.video else None,
            display_smoothed=args.display_smoothed,
            smoothing_alpha=cfg.smoothing_alpha,
        ),
    )
    results, video, complete = [], None, False
    try:
        if args.save_debug:
            (args.output / "debug").mkdir()
        if args.video:
            video = cv2.VideoWriter(
                str(args.output / "preview.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.video_fps,
                (dataset.camera.width, dataset.camera.height),
            )
            if not video.isOpened():
                raise RuntimeError("could not create offline preview video")
        count = len(dataset) if args.max_frames is None else min(len(dataset), args.max_frames)
        for index in range(count):
            frame = dataset.frame(index)
            result, edges = pipeline.process(frame)
            if cfg.tracking:
                tracker.update(result)
            results.append(result)
            writer.write(result)
            if args.save_stages:
                from box4d.debug import save_snapshot

                save_snapshot(
                    args.output / "stages" / f"{index:06d}",
                    frame,
                    result,
                    pipeline.last_debug,
                    getattr(pipeline, "local_debug", ()),
                    getattr(pipeline, "debug_tracks", ()),
                )
            if video is not None or args.save_debug:
                image = overlay(frame, result, smoothed=args.display_smoothed)
                label = (
                    "SYNTHETIC OFFLINE REPLAY"
                    if dataset.manifest.get("synthetic", False)
                    else "OFFLINE REPLAY"
                )
                cv2.putText(
                    image,
                    label,
                    (8, frame.camera.height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                if scenes.get(frame.stamp_ns):
                    cv2.putText(
                        image,
                        scenes[frame.stamp_ns][:80],
                        (8, frame.camera.height - 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                if video is not None:
                    video.write(image)
                if args.save_debug:
                    cv2.imwrite(str(args.output / "debug" / f"{index:06d}.png"), image)
                    cv2.imwrite(str(args.output / "debug" / f"{index:06d}_edges.png"), edges)
        complete = True
    finally:
        if hasattr(pipeline, "close"):
            pipeline.close()
        if video is not None:
            video.release()
        summary = dict(
            complete=complete, synthetic=dataset.manifest.get("synthetic", False), **summarize(results)
        )
        if truth_rows is not None:
            summary["evaluation"] = evaluate(results, truth_rows, cfg)
        writer.close(summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "evaluation"}, indent=2))
    if "evaluation" in summary:
        print(json.dumps({k: v for k, v in summary["evaluation"].items() if k != "frames"}, indent=2))
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
