#!/usr/bin/env python3
"""Original FoundationPose live viewer: --camera zed2i or d435i."""
import argparse
from datetime import datetime
from pathlib import Path
import socket
import time

from live_rgbd_transport import receive_frame
from kfs_segmentation import add_segmentation_args, validate_segmentation_args


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True, choices=("zed2i", "d435i"))
    parser.add_argument("--socket", default="/workspace/6Dpose/outputs/live_rgbd.sock")
    parser.add_argument("--mesh", default="/workspace/6Dpose/kfs_model/BlueTrueKFS13/BlueTrueKFS13.obj")
    parser.add_argument("--output", help="default: a new timestamped live output directory")
    parser.add_argument("--mask", help="mask for CURRENT first image, not the old recording")
    parser.add_argument("--est-refine-iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument("--axis-scale", type=float, default=0.1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument('--dynamic-roi', type=int, choices=(0, 1), default=1,
                        help='follow previous valid FP pose with ROI (automatic mask modes)')
    parser.add_argument('--roi-margin', type=float, default=.3,
                        help='expand each bbox side by this fraction')
    parser.add_argument("--save-poses", action="store_true")
    add_segmentation_args(parser)
    args = parser.parse_args()
    validate_segmentation_args(args, parser)
    import math
    if not math.isfinite(args.roi_margin) or args.roi_margin < 0:
        parser.error('ROI margin must be finite and nonnegative')
    if args.max_frames < 0 or min(args.est_refine_iter, args.track_refine_iter,
                                 args.score_batch_size) <= 0 or args.axis_scale <= 0:
        parser.error("invalid frame count / refinement / batch size / axis scale")
    if args.no_display and not args.mask and args.mask_source == "manual":
        parser.error("--no-display requires a mask for the current scene")
    return args


def main():
    args = parse_args()
    # 在解析参数后加载 GPU 模块，使 --help 不依赖 CUDA/模型环境。
    import cv2
    import numpy as np
    from run_foundationpose_zed import load_foundationpose
    from kfs_roi import project_roi
    load_foundationpose()
    from run_foundationpose_zed import (
        FoundationPose, ScorePredictor, PoseRefinePredictor, dr, trimesh,
        draw_polygon_mask, load_mask, enable_batched_scoring,
        draw_posed_3d_box, draw_xyz_axis, RollingFPS, draw_fps,
        set_logging_format, set_seed,
    )
    set_logging_format()
    set_seed(0)
    output = Path(args.output or (
        Path(__file__).resolve().parents[1] / "outputs" /
        f"live_{args.camera}_{datetime.now():%Y%m%d_%H%M%S}"))
    automatic = args.mask_source in ("opencv-sam2", "yolo")
    if automatic and output.exists() and any(output.iterdir()):
        raise ValueError("自动分割必须使用新的 --output，不复用或覆盖旧结果")
    output.mkdir(parents=True, exist_ok=True)
    if args.save_poses:
        (output / "ob_in_cam").mkdir(exist_ok=True)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(35)
    window = f"Original FP live {args.camera} (Q:quit R:register)"
    count = 0
    metrics = None
    try:
        connection.connect(args.socket)
        rgb, depth, k, metadata = receive_frame(connection)
        print(f"Live input {rgb.shape[1]}x{rgb.shape[0]}; frame={metadata['frame_id']}", flush=True)
        print("首帧定位期间请保持物体和相机静止；不会加载旧录像位姿。", flush=True)
        mask = None if automatic else (load_mask(Path(args.mask), rgb.shape[:2]) if args.mask else
                                      draw_polygon_mask(rgb, output / "initial_mask.png"))
        mesh = trimesh.load(args.mesh, force="mesh")
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0)
        estimator = FoundationPose(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
            scorer=ScorePredictor(), refiner=PoseRefinePredictor(), debug=0,
            debug_dir=str(output / "debug"), glctx=dr.RasterizeCudaContext())
        enable_batched_scoring(estimator, args.score_batch_size)
        controller = None
        if automatic:
            from kfs_segmentation import KFSSegmenter
            from kfs_pose_controller import PoseController
            controller = PoseController(estimator, KFSSegmenter.from_args(args), mesh, output,
                                        args.est_refine_iter, args.track_refine_iter, debug=bool(args.debug))
        from kfs_runtime_metrics import RuntimeMetrics
        metrics = RuntimeMetrics(output)
        meter = RollingFPS()
        expected_shape = rgb.shape
        expected_k = k.copy()
        expected_frame = metadata["frame_id"]
        register = True
        with (output / "timing.csv").open("w", buffering=1) as timing, \
                (output / "loop_timing.csv").open("w", buffering=1) as loop_timing:
            timing.write("processed_index,source_sequence,stamp,mode,inference_seconds\n")
            loop_timing.write("processed_index,source_sequence,mode,inference_s,pose_save_s,display_s,rgbd_wait_s,loop_s,roi_x0,roi_y0,roi_x1,roi_y1\n")
            while True:
                loop_started = time.monotonic()
                metrics.begin()
                full_k = np.asarray(metadata.get('full_k', k), dtype=np.float64)
                full_shape = tuple(metadata.get('full_shape', rgb.shape[:2]))
                if full_shape != expected_shape[:2] or metadata["frame_id"] != expected_frame or not np.allclose(full_k, expected_k):
                    raise RuntimeError("相机分辨率/内参/坐标系发生变化，请重新初始化")
                roi_current = metadata.get('roi_xyxy')
                if roi_current and metadata.get('preview_rgb') is None:
                    raise RuntimeError('ROI source must provide same-frame full preview')
                started = time.monotonic()
                if automatic:
                    pose, event = controller.process(rgb, depth, k, str(metadata["sequence"]))
                    mode = event["mode"]
                    if mode == "register" and event["valid_pose"]:
                        meter = RollingFPS()
                elif register:
                    # 当前帧 register 一次；后续每帧只调用 track_one，不重复 pose。
                    print("Registering current scene...", flush=True)
                    pose = estimator.register(K=k, rgb=rgb, depth=depth,
                                              ob_mask=mask, iteration=args.est_refine_iter)
                    mode = "register"
                    register = False
                    meter = RollingFPS()  # 初次定位耗时不计入稳态 tracking FPS。
                else:
                    pose = estimator.track_one(rgb=rgb, depth=depth, K=k,
                                               iteration=args.track_refine_iter)
                    mode = "tracking"
                seconds = time.monotonic() - started
                if pose is not None and not np.isfinite(pose).all():
                    raise RuntimeError("FoundationPose returned an invalid pose")
                count += 1
                timing.write(f"{count},{metadata['sequence']},{metadata['stamp']:.9f},{mode},{seconds:.6f}\n")
                if not automatic:
                    event = {"mode": mode, "valid_pose": True, "state": "TRACK", "timings": {}}
                event.setdefault('timings', {}).update({
                    name: value for name, value in metadata.get('source_timings', {}).items()
                    if value is not None})
                metrics.end(metadata["sequence"], event)
                save_started = time.monotonic()
                if args.save_poses and pose is not None:
                    np.savetxt(output / "ob_in_cam" / f"{count - 1:06d}.txt", pose, fmt="%.9g")
                save_seconds = time.monotonic() - save_started
                meter.mark()
                display_started = time.monotonic()
                key = -1
                if not args.no_display:
                    vis = metadata.get('preview_rgb', rgb).copy()
                    if pose is not None:
                        center_pose = pose @ np.linalg.inv(to_origin)
                        vis = draw_posed_3d_box(full_k, img=vis, ob_in_cam=center_pose, bbox=bbox)
                        vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=args.axis_scale,
                                            K=full_k, thickness=3, transparency=0, is_input_rgb=True)
                    if roi_current:
                        x0, y0, x1, y1 = roi_current
                        cv2.rectangle(vis, (x0, y0), (x1-1, y1-1), (255, 200, 0), 2)
                    else:
                        cv2.putText(vis, f"{event['state']}: pose unavailable", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 80, 80), 2)
                    cv2.imshow(window, draw_fps(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), meter.value()))
                    key = cv2.waitKey(1) & 0xFF
                display_seconds = time.monotonic() - display_started
                if count == 1 or count % 30 == 0:
                    print(f"Processed {count}; source={metadata['sequence']}; {mode}={seconds * 1000:.1f}ms; FPS={meter.value():.1f}", flush=True)
                if key in (ord("q"), 27) or (args.max_frames and count >= args.max_frames):
                    break
                # 拉取最新帧，不把模型加载、标掩码或推理期间的旧帧排队。
                source_sequence = metadata['sequence']
                next_roi = None
                if automatic and args.dynamic_roi and pose is not None and controller.state == 'TRACK':
                    next_roi = project_roi(mesh.vertices, pose, full_k, full_shape, args.roi_margin)
                if key == ord('r'):
                    next_roi = None
                    if controller:
                        controller.reset()
                wait_started = time.monotonic()
                if next_roi is None:
                    rgb, depth, k, metadata = receive_frame(connection)
                else:
                    rgb, depth, k, metadata = receive_frame(connection, roi=next_roi)
                wait_seconds = time.monotonic() - wait_started
                loop_seconds = time.monotonic() - loop_started
                bounds = roi_current or [0, 0, full_shape[1], full_shape[0]]
                loop_timing.write(f"{count},{source_sequence},{mode},{seconds:.6f},{save_seconds:.6f},{display_seconds:.6f},{wait_seconds:.6f},{loop_seconds:.6f},"+','.join(str(x) for x in bounds)+'\n')
                if count == 1 or count % 30 == 0:
                    print(f"Loop: inference={seconds*1000:.1f}ms; save={save_seconds*1000:.1f}ms; display={display_seconds*1000:.1f}ms; RGB-D wait={wait_seconds*1000:.1f}ms", flush=True)
                if key == ord("r"):
                    if not controller:
                        mask = draw_polygon_mask(rgb, output / "initial_mask.png")
                        register = True
    except KeyboardInterrupt:
        pass
    finally:
        if metrics:
            metrics.close()
        connection.close()
        cv2.destroyAllWindows()
        print(f"Stopped; processed={count}; logs={output}", flush=True)


if __name__ == "__main__":
    main()
