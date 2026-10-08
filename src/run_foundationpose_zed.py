#!/usr/bin/env python3
"""Run the original FoundationPose model-based pipeline on an exported ZED RGB-D sequence."""

import argparse
import json
import faulthandler
import glob
import os
import sys
import traceback
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from depth_spatial_filter import bilateral_depth
from kfs_segmentation import add_segmentation_args, validate_segmentation_args


ROOT = Path(__file__).resolve().parents[1]
FOUNDATIONPOSE_DIR = ROOT / "FoundationPose"
sys.path.insert(0, str(FOUNDATIONPOSE_DIR))

def load_foundationpose():
    """Keep --help and argument validation independent of GPU dependencies."""
    global FoundationPose, PoseRefinePredictor, ScorePredictor, dr, trimesh, torch
    global draw_posed_3d_box, draw_xyz_axis, set_logging_format, set_seed
    import torch
    import trimesh
    import nvdiffrast.torch as dr
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor
    from Utils import draw_posed_3d_box, draw_xyz_axis, set_logging_format, set_seed


# ---------------------------------------------------------------------------
# 默认配置：平时直接运行本程序即可；需要时也能用同名命令行参数覆盖。
# ---------------------------------------------------------------------------
DEFAULT_SEQUENCE = ROOT / "outputs" / "zed_sequence"
DEFAULT_MESH = ROOT / "kfs_model" / "BlueTrueKFS13" / "BlueTrueKFS13.obj"
DEFAULT_OUTPUT = ROOT / "outputs" / "zed_foundationpose"
DEFAULT_EST_REFINE_ITER = 5
DEFAULT_TRACK_REFINE_ITER = 2
DEFAULT_SCORE_BATCH_SIZE = 32
DEFAULT_AXIS_SCALE = 0.1
DEFAULT_SAVE_FRAME_IMAGES = False
DEFAULT_INPUT_SCALE = 0.5  # 1280x720 -> 640x360


class RollingFPS:
    """最近两秒内完成的帧数；包含读图、等待、推理及显示的整体速度。"""

    def __init__(self):
        self.started = time.monotonic()
        self.completed = deque()

    def mark(self):
        self.completed.append(time.monotonic())

    def value(self):
        now = time.monotonic()
        while self.completed and self.completed[0] <= now - 2.0:
            self.completed.popleft()
        return len(self.completed) / max(min(now - self.started, 2.0), 0.1)


def draw_fps(image, fps):
    # 按文字宽度右对齐，黑底保证浅色物体背景上也能看清。
    label = f"FPS: {fps:.1f}"
    size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = max(4, image.shape[1] - size[0] - 12)
    cv2.rectangle(image, (x - 4, 4), (image.shape[1] - 4, 14 + size[1] + baseline),
                  (0, 0, 0), -1)
    cv2.putText(image, label, (x, 10 + size[1]), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return image


class Tee:
    """Write console output to both the terminal and a persistent log file."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, data):
        self.terminal.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用原版 FoundationPose 处理从 ZED SVO2 导出的 RGB-D 序列"
    )
    parser.add_argument("--sequence", default=str(DEFAULT_SEQUENCE),
                        help="含 rgb/、depth/、cam_K.txt 的目录")
    parser.add_argument("--mesh", default=str(DEFAULT_MESH), help="OBJ/PLY 网格路径")
    parser.add_argument("--mask",
                        help="首帧二值 mask；默认复用输出目录的 initial_mask.png")
    parser.add_argument("--initial-pose",
                        help="已有首帧 4x4 位姿；默认自动复用输出目录的 ob_in_cam/000000.txt")
    parser.add_argument("--fresh-register", action="store_true",
                        help="忽略已有首帧位姿，强制重新执行 register")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--est-refine-iter", type=int, default=DEFAULT_EST_REFINE_ITER)
    parser.add_argument("--track-refine-iter", type=int, default=DEFAULT_TRACK_REFINE_ITER)
    parser.add_argument("--score-batch-size", type=int, default=DEFAULT_SCORE_BATCH_SIZE,
                        help="首帧候选评分批大小；8GB 显卡推荐 32，显存不足可改为 16")
    parser.add_argument("--axis-scale", type=float, default=DEFAULT_AXIS_SCALE,
                        help="坐标轴长度，单位与模型一致")
    parser.add_argument("--input-scale", type=float, default=DEFAULT_INPUT_SCALE,
                        help="RGB-D 推理缩放比例；默认 0.5，即 640x360")
    parser.add_argument("--fp-debug", type=int, default=1,
                        help="FoundationPose internal diagnostic level")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最多处理多少帧；0 表示处理全部，适合先用 10 做测试")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="从第几帧继续处理；会自动加载前一帧已保存的姿态")
    parser.add_argument("--no-display", action="store_true", help="不显示逐帧结果窗口")
    parser.add_argument("--no-video", action="store_true", help="不写 result.mp4")
    parser.add_argument("--save-frame-images", action="store_true",
                        default=DEFAULT_SAVE_FRAME_IMAGES,
                        help="额外保存 track_vis/*.png；默认关闭以提高速度")
    # 临时A/B实验：默认关闭，单位统一用米，保持旧运行行为。
    parser.add_argument("--depth-spatial-filter", action="store_true", help="推理前启用保边深度滤波；不补洞、不做时间滤波")
    parser.add_argument("--depth-filter-radius", type=int, default=2, help="滤波半径像素，默认2即5x5邻域（推理分辨率）")
    parser.add_argument("--depth-filter-spatial-sigma", type=float, default=2.0)
    parser.add_argument("--depth-filter-range-mm", type=float, default=15.0, help="深度相似性sigma，默认15mm")
    add_segmentation_args(parser)
    args = parser.parse_args()
    validate_segmentation_args(args, parser)
    if args.depth_spatial_filter and not (1 <= args.depth_filter_radius <= 8 and args.depth_filter_spatial_sigma > 0 and args.depth_filter_range_mm > 0):
        parser.error("滤波半径须1–8，sigma须大于0")
    return args


def draw_polygon_mask(rgb, save_path):
    """Collect a polygon around the object on the first RGB frame."""
    points = []
    window = "First-frame mask"
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def redraw():
        canvas = bgr.copy()
        if points:
            pts = np.asarray(points, dtype=np.int32)
            cv2.polylines(canvas, [pts], len(points) >= 3, (0, 255, 0), 2, cv2.LINE_AA)
            for point in points:
                cv2.circle(canvas, point, 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(canvas, "LMB:add  RMB/U:undo  R:reset  ENTER:confirm  Q:quit",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(window, canvas)

    def mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(points) >= 3:
            break
        if key in (ord("u"), 8, 127) and points:
            points.pop()
            redraw()
        elif key == ord("r"):
            points.clear()
            redraw()
        elif key in (ord("q"), 27):
            cv2.destroyWindow(window)
            raise RuntimeError("用户取消了首帧 mask 标注")
    cv2.destroyWindow(window)

    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), mask)
    print(f"首帧 mask 已保存: {save_path}")
    return mask > 0


def load_mask(path, expected_shape):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取 mask: {path}")
    if mask.shape != expected_shape:
        raise ValueError(f"mask 尺寸 {mask.shape[::-1]} 与图像尺寸 {expected_shape[::-1]} 不一致")
    return mask > 0


def show_status(rgb, message, enabled=True):
    if not enabled:
        return
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 75), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
    cv2.putText(canvas, message, (20, 47), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("FoundationPose ZED", canvas)
    cv2.waitKey(50)


def enable_batched_scoring(estimator, batch_size):
    """Limit scorer peak VRAM while preserving all registration hypotheses."""
    if batch_size <= 0:
        return
    original_predict = estimator.scorer.predict

    def predict_in_batches(*, ob_in_cams, get_vis=False, **kwargs):
        count = len(ob_in_cams)
        if count <= batch_size:
            return original_predict(
                ob_in_cams=ob_in_cams, get_vis=get_vis, **kwargs
            )
        if get_vis:
            print("分批评分模式下跳过 scorer 调试拼图，以降低显存占用。", flush=True)
        score_parts = []
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            torch.cuda.empty_cache()
            print(f"首帧候选评分: {start + 1}-{end}/{count}", flush=True)
            scores, _ = original_predict(
                ob_in_cams=ob_in_cams[start:end], get_vis=False, **kwargs
            )
            score_parts.append(scores)
        return torch.cat(score_parts, dim=0), None

    estimator.scorer.predict = predict_in_batches


def run_pipeline(args):
    import imageio.v2 as imageio
    load_foundationpose()
    set_logging_format()
    set_seed(0)

    sequence = Path(args.sequence).resolve()
    output = Path(args.output).resolve()
    pose_dir = output / "ob_in_cam"
    vis_dir = output / "track_vis"
    automatic = args.mask_source in ("opencv-sam2", "yolo")
    if automatic and pose_dir.exists() and any(pose_dir.iterdir()):
        raise ValueError("自动分割必须使用新的 --output，不复用或覆盖旧位姿")
    pose_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    (output / "depth_filter_config.json").write_text(json.dumps({
        "enabled": args.depth_spatial_filter, "radius": args.depth_filter_radius,
        "spatial_sigma_px": args.depth_filter_spatial_sigma,
        "range_sigma_mm": args.depth_filter_range_mm,
        "stage": "after resize, before FP inference", "fills_holes": False,
        "temporal_filter": False
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    all_rgb_files = sorted(Path(p) for p in glob.glob(str(sequence / "rgb" / "*.png")))
    if not all_rgb_files:
        raise FileNotFoundError(f"{sequence / 'rgb'} 中没有 PNG 图像")
    if args.start_frame < 0 or args.start_frame >= len(all_rgb_files):
        raise ValueError(
            f"--start-frame 必须在 0 到 {len(all_rgb_files) - 1} 之间"
        )
    if args.max_frames < 0:
        raise ValueError("--max-frames 不能小于 0")
    if not 0 < args.input_scale <= 1:
        raise ValueError("--input-scale 必须大于 0 且不超过 1")
    rgb_files = all_rgb_files[args.start_frame:]
    if args.max_frames:
        rgb_files = rgb_files[:args.max_frames]
    depth_files = [sequence / "depth" / p.name for p in rgb_files]
    missing_depth = [p for p in depth_files if not p.is_file()]
    if missing_depth:
        raise FileNotFoundError(f"缺少与 RGB 对应的深度图: {missing_depth[0]}")
    camera_path = sequence / "cam_K.txt"
    if not camera_path.is_file():
        raise FileNotFoundError(f"缺少相机内参: {camera_path}")
    K = np.loadtxt(camera_path).reshape(3, 3)

    # ZED PNGs are RGBA. Slicing to RGB can leave a non-contiguous view, while
    # OpenCV's drawing functions require a writable C-contiguous array.
    first_rgb = np.ascontiguousarray(imageio.imread(rgb_files[0])[..., :3])
    original_shape = first_rgb.shape[:2]
    default_mask_path = output / "initial_mask.png"
    mask_path = Path(args.mask).resolve() if args.mask else default_mask_path
    if automatic:
        initial_mask = np.zeros(first_rgb.shape[:2], bool)
    elif mask_path.is_file():
        initial_mask = load_mask(mask_path, first_rgb.shape[:2])
        print(f"自动加载首帧 mask: {mask_path}", flush=True)
    else:
        if args.no_display or not os.environ.get("DISPLAY"):
            raise RuntimeError("无图形界面时必须用 --mask 指定首帧二值 mask")
        initial_mask = draw_polygon_mask(first_rgb, default_mask_path)

    input_size = (original_shape[1], original_shape[0])
    if args.input_scale != 1.0:
        new_width = max(1, int(round(original_shape[1] * args.input_scale)))
        new_height = max(1, int(round(original_shape[0] * args.input_scale)))
        input_size = (new_width, new_height)
        scale_x = new_width / original_shape[1]
        scale_y = new_height / original_shape[0]
        K[0, :] *= scale_x
        K[1, :] *= scale_y
        first_rgb = cv2.resize(first_rgb, input_size, interpolation=cv2.INTER_AREA)
        initial_mask = cv2.resize(
            initial_mask.astype(np.uint8), input_size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        print(
            f"推理分辨率: {original_shape[1]}x{original_shape[0]} -> "
            f"{new_width}x{new_height} (scale={args.input_scale:g})",
            flush=True,
        )

    show_status(first_rgb, "Loading FoundationPose models...", not args.no_display)
    print("正在加载 FoundationPose scorer/refiner 模型...", flush=True)
    mesh = trimesh.load(args.mesh, force="mesh")
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        debug_dir=str(output / "debug"),
        debug=args.fp_debug,
        glctx=dr.RasterizeCudaContext(),
    )
    enable_batched_scoring(estimator, args.score_batch_size)
    print(
        f"FoundationPose 初始化完成；本次处理 {len(rgb_files)} 帧，"
        f"从原序列第 {args.start_frame} 帧开始",
        flush=True,
    )

    initial_pose = None
    pose_source_index = max(0, args.start_frame - 1)
    default_pose_path = pose_dir / all_rgb_files[pose_source_index].with_suffix(".txt").name
    initial_pose_path = Path(args.initial_pose).resolve() if args.initial_pose else default_pose_path
    if not automatic and not args.fresh_register and initial_pose_path.is_file():
        initial_pose = np.loadtxt(initial_pose_path).reshape(4, 4)
        # register() stores the centered-mesh pose internally, while its return
        # value and our TXT file use the original model origin.
        tf_to_center = estimator.get_tf_to_centered_mesh()
        estimator.pose_last = (
            torch.as_tensor(initial_pose, device="cuda", dtype=torch.float32)
            @ torch.linalg.inv(tf_to_center)
        )
        print(f"自动加载首帧姿态并恢复跟踪状态: {initial_pose_path}", flush=True)
    elif not automatic and args.start_frame > 0:
        raise FileNotFoundError(
            f"续跑需要前一帧姿态，但未找到: {initial_pose_path}"
        )

    controller = None
    if automatic:
        from kfs_segmentation import KFSSegmenter
        from kfs_pose_controller import PoseController
        controller = PoseController(estimator, KFSSegmenter.from_args(args), mesh, output,
                                    args.est_refine_iter, args.track_refine_iter, search_interval=0,
                                    debug=bool(args.debug))
    from kfs_runtime_metrics import RuntimeMetrics
    metrics = RuntimeMetrics(output)
    writer = None
    fps_meter = RollingFPS()
    try:
        for index, (rgb_path, depth_path) in enumerate(zip(rgb_files, depth_files)):
            metrics.begin()
            rgb = np.ascontiguousarray(imageio.imread(rgb_path)[..., :3])
            depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_mm is None:
                raise RuntimeError(f"无法读取深度图: {depth_path}")
            if rgb.shape[1::-1] != input_size:
                rgb = cv2.resize(rgb, input_size, interpolation=cv2.INTER_AREA)
                depth_mm = cv2.resize(
                    depth_mm, input_size, interpolation=cv2.INTER_NEAREST
                )
            rgb = np.ascontiguousarray(rgb)
            depth = depth_mm.astype(np.float32) / 1000.0
            depth[~np.isfinite(depth) | (depth < 0.001)] = 0
            if args.depth_spatial_filter:
                # 缩放后、register/track_one之前处理，确保FP实际使用滤波深度。
                depth = bilateral_depth(depth, args.depth_filter_radius,
                                        args.depth_filter_spatial_sigma,
                                        args.depth_filter_range_mm / 1000.0)
                if index == 0:
                    print(f"Depth spatial filter ON: radius={args.depth_filter_radius}, range_sigma={args.depth_filter_range_mm}mm; holes preserved", flush=True)

            if automatic:
                pose, event = controller.process(rgb, depth, K, rgb_path.stem)
            elif index == 0 and args.start_frame == 0:
                if initial_pose is not None:
                    pose = initial_pose
                else:
                    show_status(
                        rgb,
                        "Registering first frame - this can take several minutes...",
                        not args.no_display,
                    )
                    print("正在进行首帧 register；第一次运行可能需要数分钟编译 CUDA kernel...", flush=True)
                    pose = estimator.register(
                        K=K, rgb=rgb, depth=depth, ob_mask=initial_mask,
                        iteration=args.est_refine_iter,
                    )
            else:
                pose = estimator.track_one(
                    rgb=rgb, depth=depth, K=K, iteration=args.track_refine_iter,
                )

            if not automatic:
                event = {"mode": "register" if index == 0 and initial_pose is None else "tracking",
                         "valid_pose": True, "state": "TRACK", "timings": {}}
            metrics.end(rgb_path.stem, event)
            vis = rgb.copy()
            if pose is not None:
                np.savetxt(pose_dir / f"{rgb_path.stem}.txt", pose.reshape(4, 4), fmt="%.9g")
                center_pose = pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(
                    vis, ob_in_cam=center_pose, scale=args.axis_scale, K=K,
                    thickness=3, transparency=0, is_input_rgb=True,
                )
            else:
                cv2.putText(vis, f"{event['state']}: pose unavailable", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 80, 80), 2)
            if args.save_frame_images:
                cv2.imwrite(
                    str(vis_dir / f"{rgb_path.stem}.png"),
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 1],
                )

            bgr_vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            if writer is None and not args.no_video:
                height, width = bgr_vis.shape[:2]
                video_name = (
                    "result.mp4" if args.start_frame == 0
                    else f"result_from_{args.start_frame:06d}.mp4"
                )
                writer = cv2.VideoWriter(
                    str(output / video_name), cv2.VideoWriter_fourcc(*"mp4v"),
                    30.0, (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("无法创建结果视频 result.mp4")
            if writer is not None:
                writer.write(bgr_vis)
            if not args.no_display:
                fps_meter.mark()
                # 只叠加到窗口，保存的视频和逐帧图像仍可用于原始结果比较。
                cv2.imshow("FoundationPose ZED", draw_fps(bgr_vis.copy(), fps_meter.value()))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("r") and controller:
                    controller.reset()
                if key in (ord("q"), 27):
                    print("用户终止处理")
                    break
            absolute_index = args.start_frame + index
            print(
                f"[{absolute_index + 1}/{len(all_rgb_files)}] {rgb_path.name}",
                flush=True,
            )
    finally:
        metrics.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"完成。姿态、逐帧可视化和视频位于: {output}")


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with log_path.open("w", buffering=1) as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)
        faulthandler.enable(file=log_file, all_threads=True)
        try:
            print(f"运行日志: {log_path}", flush=True)
            return run_pipeline(args)
        except BaseException:
            traceback.print_exc()
            print(f"运行失败；完整错误已保存到: {log_path}", flush=True)
            raise
        finally:
            faulthandler.disable()
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == "__main__":
    main()
