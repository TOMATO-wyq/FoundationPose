#!/usr/bin/env python3
"""Run the original FoundationPose model-based pipeline on an exported ZED RGB-D sequence."""

import argparse
import faulthandler
import glob
import os
import sys
import traceback
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
FOUNDATIONPOSE_DIR = ROOT / "FoundationPose"
sys.path.insert(0, str(FOUNDATIONPOSE_DIR))

from estimater import FoundationPose  # noqa: E402
from learning.training.predict_pose_refine import PoseRefinePredictor  # noqa: E402
from learning.training.predict_score import ScorePredictor  # noqa: E402
from Utils import (  # noqa: E402
    draw_posed_3d_box,
    draw_xyz_axis,
    set_logging_format,
    set_seed,
)
import nvdiffrast.torch as dr  # noqa: E402
import trimesh  # noqa: E402


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
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最多处理多少帧；0 表示处理全部，适合先用 10 做测试")
    parser.add_argument("--no-display", action="store_true", help="不显示逐帧结果窗口")
    parser.add_argument("--no-video", action="store_true", help="不写 result.mp4")
    return parser.parse_args()


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
    set_logging_format()
    set_seed(0)

    sequence = Path(args.sequence).resolve()
    output = Path(args.output).resolve()
    pose_dir = output / "ob_in_cam"
    vis_dir = output / "track_vis"
    pose_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(Path(p) for p in glob.glob(str(sequence / "rgb" / "*.png")))
    if not rgb_files:
        raise FileNotFoundError(f"{sequence / 'rgb'} 中没有 PNG 图像")
    if args.max_frames < 0:
        raise ValueError("--max-frames 不能小于 0")
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
    default_mask_path = output / "initial_mask.png"
    mask_path = Path(args.mask).resolve() if args.mask else default_mask_path
    if mask_path.is_file():
        initial_mask = load_mask(mask_path, first_rgb.shape[:2])
        print(f"自动加载首帧 mask: {mask_path}", flush=True)
    else:
        if args.no_display or not os.environ.get("DISPLAY"):
            raise RuntimeError("无图形界面时必须用 --mask 指定首帧二值 mask")
        initial_mask = draw_polygon_mask(first_rgb, default_mask_path)

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
        debug=args.debug,
        glctx=dr.RasterizeCudaContext(),
    )
    enable_batched_scoring(estimator, args.score_batch_size)
    print(f"FoundationPose 初始化完成；共 {len(rgb_files)} 帧")

    initial_pose = None
    default_pose_path = pose_dir / rgb_files[0].with_suffix(".txt").name
    initial_pose_path = Path(args.initial_pose).resolve() if args.initial_pose else default_pose_path
    if not args.fresh_register and initial_pose_path.is_file():
        initial_pose = np.loadtxt(initial_pose_path).reshape(4, 4)
        # register() stores the centered-mesh pose internally, while its return
        # value and our TXT file use the original model origin.
        tf_to_center = estimator.get_tf_to_centered_mesh()
        estimator.pose_last = (
            torch.as_tensor(initial_pose, device="cuda", dtype=torch.float32)
            @ torch.linalg.inv(tf_to_center)
        )
        print(f"自动加载首帧姿态并恢复跟踪状态: {initial_pose_path}", flush=True)

    writer = None
    try:
        for index, (rgb_path, depth_path) in enumerate(zip(rgb_files, depth_files)):
            rgb = np.ascontiguousarray(imageio.imread(rgb_path)[..., :3])
            depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_mm is None:
                raise RuntimeError(f"无法读取深度图: {depth_path}")
            depth = depth_mm.astype(np.float32) / 1000.0
            depth[~np.isfinite(depth) | (depth < 0.001)] = 0

            if index == 0:
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

            np.savetxt(pose_dir / f"{rgb_path.stem}.txt", pose.reshape(4, 4), fmt="%.9g")
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(
                vis, ob_in_cam=center_pose, scale=args.axis_scale, K=K,
                thickness=3, transparency=0, is_input_rgb=True,
            )
            imageio.imwrite(vis_dir / f"{rgb_path.stem}.png", vis)

            bgr_vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            if writer is None and not args.no_video:
                height, width = bgr_vis.shape[:2]
                writer = cv2.VideoWriter(
                    str(output / "result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                    30.0, (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("无法创建结果视频 result.mp4")
            if writer is not None:
                writer.write(bgr_vis)
            if not args.no_display:
                cv2.imshow("FoundationPose ZED", bgr_vis)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("用户终止处理")
                    break
            print(f"[{index + 1}/{len(rgb_files)}] {rgb_path.name}", flush=True)
    finally:
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
