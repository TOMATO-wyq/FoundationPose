"""Lossless RGB-D datasets, calibration and strict JSON results."""

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .geometry import Camera


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    camera: Camera
    stamp_ns: int
    sequence: int = 0
    received_monotonic_ns: int = 0

    def __post_init__(self):
        if self.rgb.dtype != np.uint8 or self.rgb.shape != (self.camera.height, self.camera.width, 3):
            raise ValueError("RGB must be uint8 RGB in the calibration image dimensions")
        if self.depth.dtype.kind != "f" or self.depth.shape != self.rgb.shape[:2]:
            raise ValueError("depth must be aligned floating-point meters, not raw uint16")
        if int(self.stamp_ns) != self.stamp_ns or self.stamp_ns < 0:
            raise ValueError("source stamp must be nonnegative integer nanoseconds")
        self.stamp_ns, self.sequence = int(self.stamp_ns), int(self.sequence)
        self.rgb = np.ascontiguousarray(self.rgb)
        self.depth = np.ascontiguousarray(self.depth, dtype=np.float32)


def json_value(value):
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("nonfinite result cannot be serialized as a valid observation")
    return value


def dump_json(path, value):
    Path(path).write_text(json.dumps(json_value(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n")


class DatasetWriter:
    def __init__(self, path, camera, up_camera=None, synthetic=False, metadata=None):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "frames").mkdir()
        self.camera = camera
        self.count = 0
        self.stamps = set()
        self.last_stamp = -1
        self.manifest = dict(
            format="box4d-rgbd",
            version=1,
            camera=camera.to_dict(),
            depth_unit="meters",
            rgb_order="RGB",
            up_camera=up_camera,
            synthetic=bool(synthetic),
            metadata=metadata or {},
        )
        dump_json(self.path / "manifest.json", self.manifest)
        self.index = (self.path / "frames.jsonl").open("x")

    def write(self, frame):
        if frame.camera.calibration_id != self.camera.calibration_id:
            raise ValueError("calibration changed: start a new dataset")
        if frame.stamp_ns <= self.last_stamp:
            raise ValueError("refusing duplicate or out-of-order source timestamp")
        name = f"frames/{self.count:06d}.npz"
        np.savez(self.path / name, rgb=frame.rgb, depth=frame.depth)
        row = dict(file=name, sequence=frame.sequence, stamp_ns=frame.stamp_ns)
        self.index.write(json.dumps(row) + "\n")
        self.index.flush()
        self.count += 1
        self.stamps.add(frame.stamp_ns)
        self.last_stamp = frame.stamp_ns

    def close(self, complete=True, **metadata):
        if not self.index.closed:
            self.index.close()
        dump_json(
            self.path / "capture_summary.json", dict(complete=bool(complete), frames=self.count, **metadata)
        )


class Dataset:
    def __init__(self, path, allow_partial=False):
        self.path = Path(path)
        if (self.path / "manifest.json").is_file():
            self.manifest = json.loads((self.path / "manifest.json").read_text())
            if self.manifest.get("format") != "box4d-rgbd" or self.manifest.get("version") != 1:
                raise ValueError("unsupported dataset format or version")
            if self.manifest.get("depth_unit") != "meters" or self.manifest.get("rgb_order") != "RGB":
                raise ValueError("dataset units or color order unsupported")
            self.camera = Camera(**self.manifest["camera"])
            self.rows = [
                json.loads(line)
                for line in (self.path / "frames.jsonl").read_text().splitlines()
                if line.strip()
            ]
            summary_path = self.path / "capture_summary.json"
            if not allow_partial and (
                not summary_path.exists() or not json.loads(summary_path.read_text()).get("complete")
            ):
                raise ValueError("incomplete capture; use --allow-partial explicitly to inspect")
            if summary_path.exists() and json.loads(summary_path.read_text())["frames"] != len(self.rows):
                raise ValueError("capture frame count disagrees with index")
            self.legacy = False
        else:
            calibration = json.loads((self.path / "camera_calibration.json").read_text())
            if "d" not in calibration or "distortion_model" not in calibration:
                raise ValueError("legacy data is missing distortion calibration")
            self.camera = Camera(
                calibration["width"],
                calibration["height"],
                calibration["k"],
                calibration["d"],
                calibration["distortion_model"],
                calibration.get("frame_id", "camera_color_optical_frame"),
            )
            summary = json.loads((self.path / "export_summary.json").read_text())
            if summary.get("depth_unit") != "millimeters":
                raise ValueError("legacy depth unit must be explicitly millimeters")
            with (self.path / "frames.csv").open() as stream:
                self.rows = list(csv.DictReader(stream))
            self.manifest = dict(synthetic=False, up_camera=None, format="legacy-fp-export")
            self.legacy = True
        if not self.rows:
            raise ValueError("dataset contains no RGB-D frames")
        stamps = [int(r["timestamp_ns"] if self.legacy else r["stamp_ns"]) for r in self.rows]
        if len(stamps) != len(set(stamps)) or any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError("source timestamps must be strictly increasing and unique")

    def __len__(self):
        return len(self.rows)

    def frame(self, index):
        row = self.rows[index]
        if self.legacy:
            bgr = cv2.imread(str(self.path / "rgb" / row["rgb_file"]), cv2.IMREAD_COLOR)
            raw = cv2.imread(str(self.path / "depth" / row["depth_file"]), cv2.IMREAD_UNCHANGED)
            if bgr is None or raw is None or raw.dtype != np.uint16:
                raise ValueError("missing or invalid legacy RGB/depth image")
            rgb, depth = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), raw.astype(np.float32) * 0.001
            stamp, sequence = int(row["timestamp_ns"]), int(row["source_frame"])
        else:
            path = (self.path / row["file"]).resolve()
            if not path.is_relative_to(self.path.resolve()):
                raise ValueError("frame path escapes dataset")
            with np.load(path, allow_pickle=False) as archive:
                rgb, depth = archive["rgb"].copy(), archive["depth"].copy()
            stamp, sequence = int(row["stamp_ns"]), int(row["sequence"])
        return Frame(rgb, depth, self.camera, stamp, sequence)

    def __iter__(self):
        return (self.frame(i) for i in range(len(self)))


class RunWriter:
    def __init__(self, output, config, metadata=None):
        self.path = Path(output)
        self.path.mkdir(parents=True, exist_ok=False)
        dump_json(self.path / "run_config.json", dict(config=config.to_dict(), metadata=metadata or {}))
        self.stream = (self.path / "poses.jsonl").open("x")
        self.rows = 0
        self.started = time.monotonic()

    def write(self, result):
        self.stream.write(json.dumps(json_value(result), ensure_ascii=False, allow_nan=False) + "\n")
        self.rows += 1

    def close(self, summary):
        self.stream.close()
        dump_json(
            self.path / "summary.json",
            dict(frames=self.rows, elapsed_s=time.monotonic() - self.started, **summary),
        )
