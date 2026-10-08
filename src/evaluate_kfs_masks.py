#!/usr/bin/env python3
"""Compare SAM prompts and A/B masks on an exported aligned RGB-D sequence.

Human labels are optional. Missing labels never produce invented IoU values.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import json
import time

import cv2
import numpy as np

from kfs_segmentation import (KFSSegmenter, NoTarget, SegmentationResult,
                              add_segmentation_args, refine_mask, save_segmentation)


def mask_metrics(prediction, truth, tolerance=2):
    if prediction.shape != truth.shape:
        raise ValueError("Prediction and truth dimensions differ")
    union = (prediction | truth).sum()
    iou = float((prediction & truth).sum()/union) if union else 1.
    kernel = np.ones((3, 3), np.uint8)
    def boundary(mask):
        eroded = cv2.erode(mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        return mask & ~eroded.astype(bool)
    pb, tb = boundary(prediction), boundary(truth)
    if not pb.any() or not tb.any():
        f1 = 1. if not pb.any() and not tb.any() else 0.
    else:
        dt_truth = cv2.distanceTransform((~tb).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dt_pred = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        precision = float((dt_truth[pb] <= tolerance).mean())
        recall = float((dt_pred[tb] <= tolerance).mean())
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
    return {"iou": iou, "boundary_f1_2px": float(f1)}


def read_frame(sequence, name, scale):
    rgb = cv2.imread(str(sequence / "rgb" / name), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(sequence / "depth" / name), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise FileNotFoundError(f"Missing RGB/depth: {sequence}/{name}")
    if depth.ndim != 2 or depth.dtype != np.uint16 or depth.shape != rgb.shape[:2]:
        raise ValueError("Expected aligned uint16 millimetre depth PNG")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if scale != 1:
        size = (max(1, round(rgb.shape[1]*scale)), max(1, round(rgb.shape[0]*scale)))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
    return rgb, depth.astype(np.float32)/1000.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, help="rgb/depth sequence; omit when using --manifest")
    parser.add_argument("--manifest", type=Path, help="prepared validation manifest")
    parser.add_argument("--split", choices=("tune", "eval", "all"), default="eval")
    parser.add_argument("--labels", type=Path, help="human mask directory (same PNG names as sequence)")
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--input-scale", type=float, default=.5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--prompt-modes", nargs="+", choices=("box", "box1", "box3", "box3neg"), default=("box", "box1", "box3"))
    add_segmentation_args(parser)
    args = parser.parse_args()
    if not args.sam_checkpoint or not 0 < args.input_scale <= 1 or args.stride < 1 or args.max_frames < 0:
        parser.error("checkpoint required; scale in (0,1], stride>=1, max-frames>=0")
    if bool(args.sequence) == bool(args.manifest):
        parser.error("Specify exactly one of --sequence or --manifest")
    if args.output.exists():
        parser.error("Use a new --output directory")
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        frames = [f for f in manifest["frames"] if args.split == "all" or f["split"] == args.split]
        entries = [(Path(f["sequence"]), f["rgb_file"], f["key"],
                    Path(f["mask_path"]) if f.get("reviewed", False) else None) for f in frames]
    else:
        entries = [(args.sequence, p.name, p.stem, args.labels / p.name if args.labels else None)
                   for p in sorted((args.sequence / "rgb").glob("*.png"))]
    entries = entries[::args.stride]
    if args.max_frames:
        entries = entries[:args.max_frames]
    if not entries:
        parser.error("No matching frames")
    args.output.mkdir(parents=True)
    segmenter = KFSSegmenter.from_args(args)
    segmenter.refinement = False
    (args.output / "config.json").write_text(json.dumps({"segmentation": asdict(segmenter.cfg),
        "sam": segmenter.sam.metadata, "input_scale": args.input_scale, "split": args.split,
        "prompt_modes": args.prompt_modes, "depth_units": "metres", "boundary_tolerance": "2 inference pixels"}, indent=2))
    rows = []
    from kfs_runtime_metrics import RuntimeMetrics
    runtime = RuntimeMetrics(args.output)
    try:
        with (args.output / "mask_metrics.jsonl").open("w", buffering=1) as file:
            for sequence, name, key, label_path in entries:
                runtime.begin()
                rgb, depth = read_frame(sequence, name, args.input_scale)
                truth = None
                if label_path and label_path.is_file():
                    truth_image = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
                    if truth_image is None:
                        raise ValueError(f"Cannot read truth mask {label_path}")
                    original_rgb = cv2.imread(str(sequence / "rgb" / name), cv2.IMREAD_COLOR)
                    if truth_image.shape != original_rgb.shape[:2]:
                        raise ValueError(f"Human mask must match native RGB dimensions: {label_path}")
                    truth = cv2.resize(truth_image, rgb.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
                start = time.perf_counter()
                try:
                    segmenter.sam.begin(rgb)
                    encode_s = time.perf_counter()-start
                    for mode in args.prompt_modes:
                        segmenter.prompt_mode = mode
                        error = None
                        try:
                            a = segmenter.segment(rgb, depth, reuse_embedding=True)
                        except NoTarget as exc:
                            a, error = None, str(exc)
                        b, refine_error = None, None
                        if a:
                            save_segmentation(a, rgb, args.output / mode / "A" / key)
                            start = time.perf_counter()
                            try:
                                mask, diagnostics = refine_mask(a.raw, depth, a.candidate.safe, segmenter.cfg)
                                from kfs_segmentation import mask_acceptable
                                if not mask_acceptable(mask, depth, segmenter.cfg):
                                    raise NoTarget("Refined mask lacks sufficient measured depth")
                                diagnostics["sam_score"] = a.diagnostics["sam_score"]
                                b = SegmentationResult(mask, a.raw, a.candidate, diagnostics,
                                    dict(a.timings, depth_refine_s=time.perf_counter()-start))
                                save_segmentation(b, rgb, args.output / mode / "B" / key)
                            except NoTarget as exc:
                                refine_error = str(exc)
                        for branch, result in (("A", a), ("B", b)):
                            row = {"frame": key, "sequence": str(sequence), "prompt": mode, "branch": branch,
                                   "success": result is not None, "human_truth": truth is not None,
                                   "error": error or (refine_error if branch == "B" else None),
                                   "shared_encode_s": encode_s,
                                   "timings": result.timings if result else {}}
                            if truth is not None:
                                # Failed predictions are empty, not dropped from quality aggregates.
                                row.update(mask_metrics(result.mask if result else np.zeros_like(truth), truth))
                            rows.append(row)
                            file.write(json.dumps(row, allow_nan=False) + "\n")
                finally:
                    segmenter.sam.end()
                runtime.end(key, {"mode": "segmentation", "state": "EVALUATE", "valid_pose": False, "timings": {}})
                print(f"{key}: labelled={truth is not None}", flush=True)
    finally:
        runtime.close()
    summary = []
    for mode in args.prompt_modes:
        for branch in ("A", "B"):
            values = [r for r in rows if r["prompt"] == mode and r["branch"] == branch]
            labelled = [r for r in values if r["human_truth"]]
            stages = {}
            for stage in ("opencv_s", "sam_decode_s", "depth_refine_s"):
                timings = [r["timings"][stage]*1000 for r in values if stage in r["timings"]]
                stages[stage.replace("_s", "_ms")] = {
                    "p50": float(np.percentile(timings, 50)) if timings else None,
                    "p95": float(np.percentile(timings, 95)) if timings else None}
            encode = [r["shared_encode_s"]*1000 for r in values]
            stages["shared_encode_ms"] = {"p50": float(np.percentile(encode, 50)),
                                           "p95": float(np.percentile(encode, 95))}
            summary.append({"prompt": mode, "branch": branch, "frames": len(values),
                "successes": sum(r["success"] for r in values), "human_labelled_frames": len(labelled),
                "mean_iou": float(np.mean([r["iou"] for r in labelled])) if labelled else None,
                "mean_boundary_f1_2px": float(np.mean([r["boundary_f1_2px"] for r in labelled])) if labelled else None,
                "timings": stages, "timing_note": "encoder shared per image; comparison frame time contains ALL prompt/branch experiments"})
    (args.output / "mask_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
