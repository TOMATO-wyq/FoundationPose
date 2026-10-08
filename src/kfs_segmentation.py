"""Single blue KFS detection, safe SAM prompts and bounded depth refinement.

RGB is uint8 RGB, depth is aligned float32 metres, masks are HxW bool.
No CUDA/SAM imports occur until the official predictor is requested.
"""
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import subprocess
import time

import cv2
import numpy as np


class NoTarget(RuntimeError):
    """Expected detection/segmentation failure; callers may retry another frame."""


@dataclass
class SegmentationConfig:
    hsv_lower: tuple = (90, 70, 35)
    hsv_upper: tuple = (135, 255, 255)
    min_area_fraction: float = 0.001
    max_area_fraction: float = 0.65
    min_aspect: float = 0.25
    max_aspect: float = 4.0
    morphology_size: int = 3
    erosion_size: int = 3
    bbox_margin: float = 0.1
    point_spacing: float = 0.2
    negative_depth_gap_m: float = 0.65  # exceeds the 0.35m cube's 3D diameter
    depth_min_m: float = 0.1
    depth_max_m: float = 5.0
    min_seed_coverage: float = 0.6
    min_mask_pixels: int = 64
    min_valid_depth_pixels: int = 64
    median_size: int = 3
    growth_px: int = 5
    local_delta_m: float = 0.02
    edge_delta_m: float = 0.04
    small_component_pixels: int = 16
    closing_size: int = 0
    health_support_fraction: float = 0.3
    health_residual_m: float = 0.03
    occlusion_delta_m: float = 0.04
    failure_frames: int = 3
    health_interval_s: float = 2.0
    search_interval_s: float = 0.5

    def validate(self):
        lo, hi = np.asarray(self.hsv_lower), np.asarray(self.hsv_upper)
        if (lo.shape != (3,) or hi.shape != (3,) or np.any(lo < 0)
                or np.any(hi > (179, 255, 255)) or np.any(lo > hi)):
            raise ValueError("Invalid OpenCV HSV range")
        if not 0 < self.min_area_fraction < self.max_area_fraction <= 1:
            raise ValueError("Invalid area fractions")
        if not 0 < self.min_aspect <= self.max_aspect:
            raise ValueError("Invalid aspect limits")
        for name in ("morphology_size", "erosion_size", "median_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1 or value % 2 != 1:
                raise ValueError(f"{name} must be a positive odd integer")
        if self.median_size not in (3, 5) or self.closing_size not in (0, 3):
            raise ValueError("median_size must be 3/5; closing_size must be 0/3")
        if not isinstance(self.growth_px, int) or not 0 <= self.growth_px <= 32:
            raise ValueError("growth_px must be an integer in [0, 32]")
        for name in ("min_mask_pixels", "min_valid_depth_pixels", "small_component_pixels", "failure_frames"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("min_seed_coverage", "health_support_fraction"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 < self.depth_min_m < self.depth_max_m:
            raise ValueError("Invalid depth range")
        if self.bbox_margin < 0 or not 0 < self.point_spacing <= 1:
            raise ValueError("Invalid prompt geometry")
        for name in ("local_delta_m", "edge_delta_m", "health_residual_m", "occlusion_delta_m", "search_interval_s", "negative_depth_gap_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if any(not np.isfinite(v).all() for v in asdict(self).values()):
            raise ValueError("Config must contain finite numbers")
        if self.health_interval_s < 0:
            raise ValueError('health_interval_s must be nonnegative (0 checks every frame)')
        return self

    @classmethod
    def load(cls, path=None):
        return cls(**(json.loads(Path(path).read_text()) if path else {})).validate()


def add_segmentation_args(parser):
    parser.add_argument('--debug', type=int, choices=(0, 1), default=0,
                        help='show segmentation stage windows (0 off, 1 on)')
    parser.add_argument("--mask-source", choices=("manual", "opencv-sam2", "yolo"), default="manual")
    parser.add_argument("--mask_mode", "--mask-mode", choices=("sam", "yolo"))
    parser.add_argument("--yolo-weights", help="YOLO segmentation .pt weights")
    parser.add_argument("--yolo-class", default="blue_kfs", help="target class name or numeric ID")
    parser.add_argument("--yolo-conf", type=float, default=0.5)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--segmentation-config", help="KFS JSON configuration")
    parser.add_argument("--sam-checkpoint", help="official SAM2.1 checkpoint")
    parser.add_argument("--sam-checkpoint-sha256", help="expected checkpoint SHA256")
    parser.add_argument("--sam-model-config", default="configs/sam2.1/sam2.1_hiera_s.yaml")
    parser.add_argument("--sam-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--prompt-mode", choices=("box", "box1", "box3", "box3neg"), default="box1")
    parser.add_argument("--depth-refinement", action="store_true")


def validate_segmentation_args(args, parser):
    if args.debug and args.no_display:
        parser.error('--debug 1 requires display; remove --no-display')
    if args.mask_mode:
        args.mask_source = "opencv-sam2" if args.mask_mode == "sam" else "yolo"
    if not 0 < args.yolo_conf <= 1 or args.yolo_imgsz <= 0:
        parser.error("invalid YOLO confidence / image size")
    if args.mask_source == "yolo" and not args.yolo_weights:
        parser.error("--mask_mode yolo requires --yolo-weights")
    if args.mask_source in ("opencv-sam2", "yolo"):
        if args.mask_source == "opencv-sam2" and not args.sam_checkpoint:
            parser.error("--mask-source opencv-sam2 requires --sam-checkpoint")
        if args.mask:
            parser.error("automatic segmentation cannot reuse --mask")
        if getattr(args, "initial_pose", None):
            parser.error("automatic segmentation cannot reuse --initial-pose")
    elif args.depth_refinement:
        parser.error("--depth-refinement requires --mask-source opencv-sam2")


def valid_depth(depth, cfg):
    return np.isfinite(depth) & (depth >= cfg.depth_min_m) & (depth <= cfg.depth_max_m)


@dataclass
class Candidate:
    box: np.ndarray  # XYXY, inclusive pixel coordinates
    coarse: np.ndarray
    safe: np.ndarray
    points: np.ndarray
    labels: np.ndarray


def safe_points(safe, box, count, spacing):
    # Pad so distanceTransform also sees a boundary for masks touching the image edge.
    distance = cv2.distanceTransform(np.pad(safe.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    yy, xx = np.indices(safe.shape)
    result = []
    min_distance = max(1., spacing * min(box[2] - box[0] + 1, box[3] - box[1] + 1))
    for _ in range(count):
        y, x = np.unravel_index(np.argmax(distance), distance.shape)
        if distance[y, x] <= 0:
            break
        result.append((x, y))
        distance[(xx - x)**2 + (yy - y)**2 < min_distance**2] = 0
    return np.asarray(result, dtype=np.float32).reshape(-1, 2)


def detect_candidates(rgb, depth, cfg, prompt_mode="box1"):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]:
        raise ValueError("Expected uint8 RGB and aligned HxW depth")
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    colour = cv2.inRange(hsv, np.asarray(cfg.hsv_lower, np.uint8), np.asarray(cfg.hsv_upper, np.uint8))
    # Missing depth does not erase RGB evidence. Valid depths outside the work range do.
    measured = np.isfinite(depth) & (depth > 0)
    colour[measured & ~valid_depth(depth, cfg)] = 0
    kernel = np.ones((cfg.morphology_size, cfg.morphology_size), np.uint8)
    coarse = cv2.morphologyEx(colour, cv2.MORPH_OPEN, kernel)
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, kernel)
    n, components, stats, _ = cv2.connectedComponentsWithStats(coarse, connectivity=8)
    h, w = depth.shape
    result = []
    for component in sorted(range(1, n), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True):
        x, y, bw, bh, area = stats[component]
        if not (cfg.min_area_fraction <= area / (h*w) <= cfg.max_area_fraction
                and cfg.min_aspect <= bw / bh <= cfg.max_aspect):
            continue
        mask = components == component
        # Padding prevents OpenCV's default erosion border from creating unsafe edge seeds.
        safe = cv2.erode(mask.astype(np.uint8), np.ones((cfg.erosion_size, cfg.erosion_size), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        if not safe.any():
            continue
        mx, my = int(np.ceil(bw*cfg.bbox_margin)), int(np.ceil(bh*cfg.bbox_margin))
        box = np.array([max(0, x-mx), max(0, y-my), min(w-1, x+bw-1+mx), min(h-1, y+bh-1+my)], np.float32)
        count = {"box": 0, "box1": 1, "box3": 3, "box3neg": 3}[prompt_mode]
        points = safe_points(safe, box, count, cfg.point_spacing)
        labels = np.ones(len(points), np.int32)
        if prompt_mode == "box3neg":
            # Only deep, valid, non-blue pixels outside the padded bbox are negatives.
            # Colour alone cannot label dark target faces as background.
            z = depth[safe & valid_depth(depth, cfg)]
            if len(z):
                reference = float(np.max(z))
                options = [(int(box[0])-2, int((box[1]+box[3])/2)),
                           (int(box[2])+2, int((box[1]+box[3])/2)),
                           (int((box[0]+box[2])/2), int(box[1])-2),
                           (int((box[0]+box[2])/2), int(box[3])+2)]
                for px, py in options:
                    if (0 <= px < w and 0 <= py < h and colour[py, px] == 0
                            and valid_depth(depth[py:py+1, px:px+1], cfg)[0, 0]
                            and depth[py, px] > reference + cfg.negative_depth_gap_m):
                        points = np.vstack((points, (px, py))).astype(np.float32)
                        labels = np.append(labels, 0).astype(np.int32)
        result.append(Candidate(box, mask, safe, points, labels))
    return result


def masked_median(depth, cfg):
    valid = valid_depth(depth, cfg)
    r = cfg.median_size // 2
    source = np.pad(np.where(valid, depth, np.nan), r, constant_values=np.nan)
    # Process rows in chunks: a full 5x5 sliding window stack is large at native resolution.
    out = np.zeros(depth.shape, np.float32)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for start in range(0, depth.shape[0], 128):
            end = min(depth.shape[0], start + 128)
            windows = [source[start+dy:end+dy, dx:dx+depth.shape[1]]
                       for dy in range(cfg.median_size) for dx in range(cfg.median_size)]
            out[start:end] = np.nanmedian(np.stack(windows), axis=0)
    out[~valid] = 0
    return out, valid


def depth_edges(depth, valid, threshold):
    edge = np.zeros(valid.shape, bool)
    # Mark both endpoints of measured jumps. Invalid neighbours are not numeric zero.
    for dy, dx in ((0, 1), (1, 0)):
        a = (slice(0, depth.shape[0]-dy), slice(0, depth.shape[1]-dx))
        b = (slice(dy, depth.shape[0]), slice(dx, depth.shape[1]))
        jump = valid[a] & valid[b] & (np.abs(depth[a]-depth[b]) > threshold)
        edge[a] |= jump
        edge[b] |= jump
    return edge


def depth_flood(seeds, allowed, depth, original, valid, edge, cfg):
    """Four-connected graph traversal; tests both original and denoised depth.

    Original-depth checks prevent median filtering from bridging flying pixels.
    Edges are conservative barriers: SAM evidence at edges survives separately.
    """
    reached = seeds & allowed & valid
    queue = deque(map(tuple, np.argwhere(reached)))
    h, w = depth.shape
    while queue:
        y, x = queue.popleft()
        if edge[y, x]:
            continue
        for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if not (0 <= ny < h and 0 <= nx < w) or reached[ny, nx] or not allowed[ny, nx] or not valid[ny, nx] or edge[ny, nx]:
                continue
            if (abs(float(depth[ny, nx])-float(depth[y, x])) <= cfg.local_delta_m
                    and abs(float(original[ny, nx])-float(original[y, x])) <= cfg.local_delta_m):
                reached[ny, nx] = True
                queue.append((ny, nx))
    return reached


def remove_small(mask, cfg):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg.small_component_pixels
    return keep[labels]


def refine_mask(raw, depth, safe, cfg):
    raw = np.asarray(raw, bool)
    if raw.shape != depth.shape or safe.shape != depth.shape:
        raise ValueError("Mask / depth shapes differ")
    filtered, valid = masked_median(depth, cfg)
    # Raw jumps remain barriers even when a median shifts an edge by a pixel.
    edge = depth_edges(depth, valid, cfg.edge_delta_m) | depth_edges(filtered, valid, cfg.edge_delta_m)
    seeds = raw & safe & valid
    if not seeds.any():
        raise NoTarget("No reliable depth seeds")
    inner = depth_flood(seeds, raw, filtered, depth, valid, edge, cfg)
    # Retain original edge evidence only when adjacent to a reached target pixel
    # and locally depth-compatible. Do not keep isolated SAM-labelled foreground.
    adjacent = cv2.dilate(inner.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    border = raw & edge & valid & adjacent
    edge_keep = np.zeros_like(raw)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        moved = np.roll(inner, (dy, dx), (0, 1))
        z = np.roll(depth, (dy, dx), (0, 1))
        if dy < 0: moved[dy:] = False
        if dy > 0: moved[:dy] = False
        if dx < 0: moved[:, dx:] = False
        if dx > 0: moved[:, :dx] = False
        edge_keep |= border & moved & (np.abs(depth-z) <= cfg.local_delta_m)
    kept = inner | edge_keep | (raw & ~valid)
    # Euclidean distance is fixed against RAW, never against the growing frontier.
    if raw.any():
        distance = cv2.distanceTransform((~raw).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        band = (~raw) & (distance <= cfg.growth_px) & valid & ~edge
    else:
        band = np.zeros_like(raw)
    grown = depth_flood(inner, inner | band, filtered, depth, valid, edge, cfg)
    refined = remove_small(kept | (grown & band), cfg)
    if cfg.closing_size:
        proposed = cv2.morphologyEx(refined.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
        allowed = proposed & valid & ~edge & ((raw) | band)
        closed = depth_flood(refined & valid & ~edge, allowed | refined, filtered, depth, valid, edge, cfg)
        refined = remove_small(refined | (closed & allowed), cfg)
    return refined, {"candidate_band": band, "depth_edges": edge,
                     "added": refined & ~raw, "removed": raw & ~refined}


@contextmanager
def cpu_factory_defaults(torch):
    """Isolate CPU transforms from FP's legacy global CUDA tensor type.

    SAM's image transforms use CPU tensors. This pipeline is single-threaded;
    always restore FP's setting after those tensor factories run.
    """
    previous = torch.empty(0).type()
    try:
        torch.set_default_tensor_type(torch.FloatTensor)
        yield
    finally:
        torch.set_default_tensor_type(previous)


class OfficialSAM:
    def __init__(self, checkpoint, model_config, device="cuda", sha256=None):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"SAM checkpoint not found: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lock = json.loads((Path(__file__).parent / "sam2-lock.json").read_text())
        if not sha256 and path.name == lock["checkpoint_file"]:
            sha256 = lock["checkpoint_sha256"]
        if sha256 and digest.lower() != sha256.lower():
            raise ValueError("SAM checkpoint SHA256 mismatch")
        self.metadata = {"checkpoint": str(path.resolve()), "sha256": digest,
                         "model_config": model_config, "device": device}
        try:
            import torch
            import sam2
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError("Install the isolated SAM2 environment described in src/KFS_PIPELINE.md") from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --sam-device cpu for segmentation only")
        self.torch = torch
        self.metadata.update(torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                             sam_source=str(Path(sam2.__file__).resolve()))
        source = Path(sam2.__file__).resolve().parents[1]
        revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                  capture_output=True, text=True)
        self.metadata["sam_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
        self.device = device
        # SAM's optional hole filling is disabled: visible masks must preserve occlusion.
        with cpu_factory_defaults(torch):
            model = build_sam2(model_config, str(path), device="cpu", apply_postprocessing=False)
            self.predictor = SAM2ImagePredictor(model, max_hole_area=0, max_sprinkle_area=0)
        self.frame_active = False
        self._clear_device_caches()

    def _clear_device_caches(self):
        # Upstream positional caches and RoPE tables are plain attributes:
        # Module.to('cpu') alone does NOT offload them.
        for module in self.predictor.model.modules():
            cache = getattr(module, "cache", None)
            if isinstance(cache, dict) and all(self.torch.is_tensor(v) for v in cache.values()):
                cache.clear()
            frequencies = getattr(module, "freqs_cis", None)
            if self.torch.is_tensor(frequencies):
                module.freqs_cis = frequencies.cpu()

    def begin(self, rgb):
        self.predictor.model.to(self.device)
        self.frame_active = True
        with cpu_factory_defaults(self.torch), self.torch.inference_mode():
            self.predictor.set_image(rgb)
        if self.device == "cuda":
            # Otherwise encode time leaks into the first decode's numpy transfer.
            self.torch.cuda.synchronize()

    def predict(self, candidate):
        with cpu_factory_defaults(self.torch), self.torch.inference_mode():
            return self.predictor.predict(
                point_coords=candidate.points if len(candidate.points) else None,
                point_labels=candidate.labels if len(candidate.labels) else None,
                box=candidate.box, multimask_output=True)[:2]

    def end(self):
        self.predictor.reset_predictor()
        self.predictor.model.to("cpu")
        self._clear_device_caches()
        self.frame_active = False
        if self.device == "cuda":
            self.torch.cuda.synchronize()
            self.torch.cuda.empty_cache()


@dataclass
class SegmentationResult:
    mask: np.ndarray
    raw: np.ndarray
    candidate: Candidate
    diagnostics: dict
    timings: dict


def mask_acceptable(mask, depth, cfg):
    return (cfg.min_mask_pixels <= int(mask.sum()) <= cfg.max_area_fraction * mask.size
            and int((mask & valid_depth(depth, cfg)).sum()) >= cfg.min_valid_depth_pixels)


class KFSSegmenter:
    def __init__(self, cfg, sam, prompt_mode="box1", refinement=False):
        self.cfg, self.sam = cfg, sam
        self.prompt_mode, self.refinement = prompt_mode, refinement

    @classmethod
    def from_args(cls, args):
        cfg = SegmentationConfig.load(args.segmentation_config)
        if args.mask_source == "yolo":
            from kfs_yolo_segmentation import YOLOSegmenter
            return YOLOSegmenter(cfg, args)
        sam = OfficialSAM(args.sam_checkpoint, args.sam_model_config, args.sam_device, args.sam_checkpoint_sha256)
        return cls(cfg, sam, args.prompt_mode, args.depth_refinement)

    def segment(self, rgb, depth, reuse_embedding=False):
        start = time.perf_counter()
        candidates = detect_candidates(rgb, depth, self.cfg, self.prompt_mode)
        timings = {"opencv_s": time.perf_counter()-start}
        if not candidates:
            raise NoTarget("No blue KFS candidate")
        try:
            start = time.perf_counter()
            if not reuse_embedding:
                self.sam.begin(rgb)
            timings["sam_encode_s"] = time.perf_counter()-start
            decode_s = refine_s = 0.
            for candidate in candidates:
                start = time.perf_counter()
                masks, scores = self.sam.predict(candidate)
                decode_s += time.perf_counter()-start
                for i in np.argsort(scores)[::-1]:
                    raw = np.asarray(masks[i], bool)
                    if raw.shape != depth.shape or not np.isfinite(scores[i]):
                        continue
                    positives = candidate.points[candidate.labels == 1].astype(int)
                    negatives = candidate.points[candidate.labels == 0].astype(int)
                    if (len(positives) and not raw[positives[:, 1], positives[:, 0]].all()) or (len(negatives) and raw[negatives[:, 1], negatives[:, 0]].any()):
                        continue
                    if (raw & candidate.safe).sum() / candidate.safe.sum() < self.cfg.min_seed_coverage:
                        continue
                    if not mask_acceptable(raw, depth, self.cfg):
                        continue
                    start = time.perf_counter()
                    try:
                        mask, diagnostics = refine_mask(raw, depth, candidate.safe, self.cfg) if self.refinement else (raw.copy(), {})
                    except NoTarget:
                        continue
                    refine_s += time.perf_counter()-start
                    if not mask_acceptable(mask, depth, self.cfg):
                        continue
                    timings.update(sam_decode_s=decode_s, depth_refine_s=refine_s)
                    result = SegmentationResult(mask, raw, candidate, diagnostics, timings)
                    result.diagnostics["sam_score"] = float(scores[i])
                    return result
            raise NoTarget("SAM masks failed prompt / geometry checks")
        finally:
            if not reuse_embedding:
                start = time.perf_counter()
                self.sam.end()
                timings["sam_offload_s"] = time.perf_counter()-start


def save_segmentation(result, rgb, output):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    masks = {"coarse": result.candidate.coarse, "safe": result.candidate.safe,
             "raw": result.raw, "refined": result.mask}
    masks.update({k: v for k, v in result.diagnostics.items() if isinstance(v, np.ndarray)})
    for name, mask in masks.items():
        cv2.imwrite(str(path / f"{name}.png"), mask.astype(np.uint8)*255)
    overlay = rgb.copy()
    overlay[result.mask] = (0.5*overlay[result.mask] + np.array([0, 127, 0])).astype(np.uint8)
    x0, y0, x1, y1 = result.candidate.box.astype(int)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 0), 1)
    for point, label in zip(result.candidate.points, result.candidate.labels):
        cv2.circle(overlay, tuple(point.astype(int)), 3, (0, 255, 0) if label else (255, 0, 0), -1)
    cv2.imwrite(str(path / "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    (path / "prompt.json").write_text(json.dumps({"bbox_xyxy": result.candidate.box.tolist(),
        "points_xy": result.candidate.points.tolist(), "labels": result.candidate.labels.tolist(),
        "timings": result.timings, "mask_score": result.diagnostics.get("sam_score", result.diagnostics.get("yolo_score")),
        "mask_mode": result.diagnostics.get("mask_mode", "sam")}, indent=2))
