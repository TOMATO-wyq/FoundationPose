#!/usr/bin/env python3
"""Compare NumPy and real PCL on identical preloaded RGB-D frames, without GUI."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import numpy as np

# Keep direct invocation of this organized entry point equivalent to -m.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from box4d.config import Config
from box4d.geometry import angle_difference
from box4d.io import Dataset, dump_json, json_value
from box4d.metrics import environment, evaluate, stats, summarize
from box4d.pcl_backend import backend_info
from box4d.temporal import make_pipeline


def differences(first, second):
    centers, yaws = [], []
    for a, b in zip(first, second):
        used = set()
        for obj in a['objects']:
            candidates = [(np.linalg.norm(np.asarray(obj['center_camera_m']) - other['center_camera_m']), j, other)
                          for j, other in enumerate(b['objects'])
                          if j not in used and abs(obj['size_m'] - other['size_m']) < 1e-6]
            if not candidates:
                continue
            distance, j, other = min(candidates, key=lambda value: value[0])
            if distance > obj['size_m'] * .4:
                continue
            used.add(j)
            centers.append(distance * 1000)
            if obj['yaw_u_rad'] is not None and other['yaw_u_rad'] is not None:
                yaws.append(np.rad2deg(abs(angle_difference(obj['yaw_u_rad'], other['yaw_u_rad']))))
    return dict(center_difference_mm=stats(centers), yaw_difference_deg_mod90=stats(yaws),
                note='Differences between two estimators are not absolute errors.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2] / 'config/box4d_d435i_live.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--up-camera', type=float, nargs=3)
    parser.add_argument('--max-frames', type=int, default=60)
    parser.add_argument('--modes', nargs='+', choices=('detection', 'temporal'), default=['detection', 'temporal'])
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error('--max-frames must be positive')
    data = Dataset(args.dataset)
    frames = [data.frame(i) for i in range(min(len(data), args.max_frames))]
    cfg = Config.load(args.config)
    cfg.up_camera = args.up_camera if args.up_camera is not None else (cfg.up_camera or data.manifest.get('up_camera'))
    cfg.validate()
    # Load/validate the native library before measuring either estimator.
    pcl = backend_info()
    truth_path = args.dataset / 'truth.json'
    truth = json.loads(truth_path.read_text()) if truth_path.is_file() else None
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(dataset=str(args.dataset), frames=len(frames), synthetic=data.manifest.get('synthetic', False),
                  environment=environment(), pcl=pcl, config=cfg.to_dict(), modes={},
                  note='Identical preloaded frames; synchronous offline processing, not live camera FPS. '
                       'Real-data output counts measure coverage, not accuracy without labeled truth.')
    for mode in dict.fromkeys(args.modes):
        runs, raw = {}, {}
        for backend in ('numpy', 'pcl'):
            current = deepcopy(cfg)
            current.plane_backend = backend
            current.temporal_recovery = mode == 'temporal'
            pipeline = make_pipeline(current)
            rows = []
            begin = time.perf_counter()
            try:
                for frame in frames:
                    result, _ = pipeline.process(frame)
                    rows.append(result)
            finally:
                if hasattr(pipeline, 'close'):
                    pipeline.close()
            elapsed = time.perf_counter() - begin
            run = dict(config=current.to_dict(), elapsed_processing_s=elapsed, **summarize(rows))
            run['tracking_failure_counts'] = dict(Counter(reason for row in rows
                        for reason in row.get('temporal', {}).get('tracking_failures', [])))
            if truth is not None:
                run['evaluation'] = {k: v for k, v in evaluate(rows, truth, current).items() if k != 'frames'}
            with (args.output / f'{mode}_{backend}_poses.jsonl').open('x') as stream:
                for row in rows:
                    stream.write(json.dumps(json_value(row), allow_nan=False) + '\n')
            runs[backend], raw[backend] = run, rows
            print(json.dumps(dict(mode=mode, backend=backend, counts=run['observation_count_histogram'],
                                  mean_ms=run['processing_ms']['mean'], p95_ms=run['processing_ms']['p95'],
                                  evaluation=run.get('evaluation')), ensure_ascii=False), flush=True)
        report['modes'][mode] = dict(runs=runs, comparison=differences(raw['numpy'], raw['pcl']))
    dump_json(args.output / 'comparison.json', report)
    print(f"Comparison: {args.output / 'comparison.json'}", flush=True)


if __name__ == '__main__':
    main()
