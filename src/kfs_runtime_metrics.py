"""Per-frame wall/CPU/allocator metrics; registration and tracking kept separate."""
from collections import defaultdict
from pathlib import Path
import json
import resource
import sys
import time

import numpy as np


class RuntimeMetrics:
    def __init__(self, output):
        self.output = Path(output)
        self.file = (self.output / "runtime.jsonl").open("w", buffering=1)
        self.samples = defaultdict(list)
        self.previous_end = None
        self.tracking_intervals = []
        self.register_attempts = self.register_successes = 0
        self.peak_reserved = self.peak_allocated = self.peak_rss = 0

    def begin(self):
        self.started, self.cpu_started = time.perf_counter(), time.process_time()
        torch = sys.modules.get("torch")
        if torch and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def end(self, frame, event):
        end = time.perf_counter()
        seconds = end-self.started
        cpu = time.process_time()-self.cpu_started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux bytes
        allocated = reserved = None
        torch = sys.modules.get("torch")
        if torch and torch.cuda.is_available():
            allocated = torch.cuda.max_memory_allocated()
            reserved = torch.cuda.max_memory_reserved()
            self.peak_allocated = max(self.peak_allocated, allocated)
            self.peak_reserved = max(self.peak_reserved, reserved)
        self.peak_rss = max(self.peak_rss, rss)
        mode = event["mode"]
        self.samples[f"{mode}_frame_s"].append(seconds)
        for name, value in event.get("timings", {}).items():
            self.samples[name].append(value)
        if mode == "register":
            self.register_attempts += 1
            self.register_successes += int(event["valid_pose"])
        if self.previous_end is not None and mode == "tracking" and self.previous_mode == "tracking":
            self.tracking_intervals.append(end-self.previous_end)
        self.previous_end, self.previous_mode = end, mode
        self.file.write(json.dumps({"frame": str(frame), "mode": mode, "state": event["state"],
            "valid_pose": event["valid_pose"], "frame_to_pose_s": seconds,
            "cpu_core_percent": 100*cpu/max(seconds, 1e-9), "peak_process_rss_bytes": rss,
            "torch_peak_allocated_bytes": allocated, "torch_peak_reserved_bytes": reserved,
            "timings": event.get("timings", {})}, allow_nan=False) + "\n")

    def close(self):
        self.file.close()
        summary = {name: {"count": len(values), "p50": float(np.percentile(values, 50)),
                         "p95": float(np.percentile(values, 95))}
                   for name, values in self.samples.items() if values}
        summary.update(register_attempts=self.register_attempts, register_successes=self.register_successes,
                       steady_tracking_fps=(len(self.tracking_intervals)/sum(self.tracking_intervals)
                                            if self.tracking_intervals else None),
                       peak_process_rss_bytes=self.peak_rss,
                       torch_peak_allocated_bytes=self.peak_allocated,
                       torch_peak_reserved_bytes=self.peak_reserved,
                       gpu_memory_note="PyTorch allocator only; excludes driver/other processes",
                       timing_note="frame_to_pose excludes drawing; steady FPS includes intervals between frames")
        (self.output / "runtime_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
