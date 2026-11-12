#!/usr/bin/env python3
"""D435i CPU box poses from ROS2 synchronized RGBD; camera driver runs separately."""
import argparse
import json
from collections import deque
from pathlib import Path
import sys
import time
import numpy as np

# Keep direct invocation of this organized entry point equivalent to -m.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from box4d.config import Config
from box4d.io import RunWriter, dump_json
from box4d.metrics import environment, stats, process_usage
from box4d.temporal import make_pipeline
from box4d.pacing import DisplayPacer
from box4d.ros_input import RosSource, up_from_plane_roi
from box4d.tracking import Tracker
from box4d.visualization import overlay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "config/box4d_cpu.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--size", choices=("big", "small"), help="recognize and track only big (0.35 m) or small (0.20 m) cubes")
    p.add_argument("--topic", default="/camera/camera/rgbd")
    p.add_argument("--up-camera", type=float, nargs=3)
    p.add_argument("--up-calibration", type=Path, help="stationary IMU UP report, with reference provenance")
    p.add_argument(
        "--auto-up",
        action="store_true",
        help="measure current stationary IMU UP at startup; an optional old report is compared, not reused",
    )
    p.add_argument("--imu-topic", default="/camera/camera/accel/sample")
    p.add_argument("--prediction-model", choices=("cv", "ca"), help="override position/velocity or position/velocity/acceleration model")
    p.add_argument("--no-camera-rotation", action="store_true", help="disable gyro rotation compensation for comparison")
    p.add_argument(
        "--up-plane-roi",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="known horizontal floor/table below camera",
    )
    p.add_argument("--method", choices=("depth", "pnp", "hybrid"))
    p.add_argument(
        "--plane-backend",
        choices=("numpy", "pcl"),
        help="override scene segmentation and local plane fitting backend",
    )
    p.add_argument(
        "--geometry-backend",
        choices=("auto", "numpy", "cpp"),
        help="CPU numerical kernels; cpp requires a native build",
    )
    p.add_argument("--no-refine", action="store_true")
    p.add_argument(
        "--yolo-weights", type=Path, help="optional YOLO-seg checkpoint or directory containing best.pt"
    )
    p.add_argument("--yolo-device", help="YOLO device override (default: cpu)")
    p.add_argument("--yolo-conf", type=float, help="YOLO instance confidence threshold")
    p.add_argument("--max-sync-ms", type=float, default=0.0)
    p.add_argument("--depth-scale", type=float, default=0.001)
    p.add_argument("--rectified-input", action="store_true")
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--display", action="store_true")
    p.add_argument(
        "--display-smoothed",
        action="store_true",
        help="show existing tracked filtered fields; raw poses remain logged",
    )
    p.add_argument("--display-hz", type=float, default=30.0)
    p.add_argument(
        "--publish-poses",
        action="store_true",
        help="publish current tracked poses to ROS; --auto-up also publishes the measured camera UP",
    )
    p.add_argument(
        "--debug", action="store_true", help="show eight pipeline stages and live threshold sliders"
    )
    p.add_argument(
        "--debug-hz",
        type=float,
        default=4.0,
        help="debug drawing rate; additional diagnostic overhead is not inference timing",
    )
    p.add_argument("--debug-layout", choices=("windows", "grid"), default="windows")
    args = p.parse_args()
    if (
        args.duration <= 0
        or args.timeout <= 0
        or args.display_hz <= 0
        or args.debug_hz <= 0
        or args.max_sync_ms < 0
    ):
        p.error("duration, timeout and display rate must be positive; sync tolerance nonnegative")
    if sum(v is not None for v in (args.up_camera, args.up_plane_roi, args.up_calibration)) > 1:
        p.error("choose one vertical reference method")
    if args.auto_up and (args.up_camera is not None or args.up_plane_roi is not None):
        p.error("--auto-up cannot be combined with manual UP vector/plane")
    cfg = Config.load(
        args.config,
        size=args.size,
        method=args.method,
        plane_backend=args.plane_backend,
        geometry_backend=args.geometry_backend,
        up_camera=args.up_camera,
        prediction_model=args.prediction_model,
        camera_rotation_compensation=False if args.no_camera_rotation else None,
        yolo_weights=None if args.yolo_weights is None else str(args.yolo_weights),
        yolo_device=args.yolo_device,
        yolo_conf=args.yolo_conf,
        refine_4d=False if args.no_refine else None,
    )
    if args.display_smoothed and not cfg.tracking:
        p.error("--display-smoothed requires tracking")
    if args.publish_poses and not cfg.tracking:
        p.error("--publish-poses requires tracking")
    up_report = None
    current_up = None
    if args.up_calibration is not None:
        up_report = json.loads(args.up_calibration.read_text())
        if up_report.get("status") != "stationary_estimate" or not up_report.get("camera_frame_id"):
            p.error("UP report must be a successful stationary estimate with camera_frame_id")
        cfg.up_camera = up_report["up_camera"]
        cfg.validate()
    source = RosSource(
        args.topic,
        args.depth_scale,
        args.max_sync_ms,
        args.rectified_input,
        imu_topic=args.imu_topic if args.auto_up else None,
        motion_config=cfg,
    )
    writer = display = pipeline = debug_display = None
    count = 0
    begin = time.monotonic()
    processed_stamps, durations, latencies, counts = deque(maxlen=7200), [], [], []
    complete = False
    display_submissions = []
    flow_counts = []
    full_detection_frames = 0
    initial_usage = process_usage()
    try:
        frame = source.get(timeout=args.timeout)
        if args.auto_up:
            print("Measuring current IMU UP for 2 seconds; keep the camera stationary.", flush=True)
            previous_up = cfg.up_camera
            previous_report = args.up_calibration
            try:
                up_report = source.estimate_up(frame.camera.frame_id, timeout=max(3.0, args.timeout))
            except ValueError as error:
                p.exit(2, f"IMU UP calibration failed: {error}\n")
            if previous_up is not None:
                change = float(
                    np.rad2deg(np.arccos(np.clip(np.asarray(previous_up) @ up_report["up_camera"], -1, 1)))
                )
                up_report.update(
                    previous_up_camera=previous_up,
                    change_from_previous_deg=change,
                    previous_report_path=None if previous_report is None else str(previous_report),
                )
                print(
                    f"UP differs from previous reference by {change:.2f} degrees; using current measurement.",
                    flush=True,
                )
            cfg.up_camera = up_report["up_camera"]
            cfg.validate()
            current_up = tuple(cfg.up_camera)
            # Startup calibration takes seconds; process the newest RGB-D frame.
            frame = source.get(frame.sequence, timeout=args.timeout)
            print(f"Current provisional UP: {cfg.up_camera}", flush=True)
        reference = dict(source="configured_up_vector" if cfg.up_camera is not None else "not_calibrated")
        if up_report is not None:
            if frame.camera.frame_id != up_report["camera_frame_id"]:
                raise ValueError("UP calibration optical frame does not match live camera")
            reference = dict(
                report_path="up_reference.json" if args.auto_up else str(args.up_calibration), **up_report
            )
        if args.up_plane_roi:
            cfg.up_camera, reference = up_from_plane_roi(frame, args.up_plane_roi, cfg.plane_threshold_m)
            cfg.validate()
        pose_publisher = None
        if args.publish_poses:
            from box4d.ros_output import RosPosePublisher

            pose_publisher = RosPosePublisher(source.node, current_up)
        pipeline, tracker = make_pipeline(cfg, live=True, debug=args.debug), Tracker(cfg)
        if cfg.yolo_weights:
            print(
                f"YOLO-seg {cfg.yolo_device} ({cfg.yolo_cpu_threads} CPU threads), ROIs for {cfg.yolo_sizes_m} m: "
                f"{cfg.yolo_weights}; other sizes use original geometry.",
                flush=True,
            )
        writer = RunWriter(
            args.output,
            cfg,
            dict(
                source="ros2",
                topic=args.topic,
                camera=frame.camera.to_dict(),
                vertical_reference=reference,
                environment=environment(cfg),
                real_camera_validation=True,
                display_smoothed=args.display_smoothed,
                smoothing_alpha=cfg.smoothing_alpha,
                debug=args.debug,
                debug_hz=args.debug_hz,
                debug_layout=args.debug_layout,
            ),
        )
        if args.auto_up:
            dump_json(args.output / "up_reference.json", up_report)
        if cfg.up_camera is None:
            print(
                "No vertical reference: reporting camera-frame geometry, yaw_u_rad remains null.", flush=True
            )
        if args.display:
            from common.display import AsyncPoseDisplay

            display = AsyncPoseDisplay(frame.rgb.shape, "Box4D CPU")
        if args.debug:
            from box4d.debug import DebugSession

            debug_display = DebugSession(args.output, args.config, cfg, args.debug_layout)
            print(
                "Debug: adjust 09 Tuning sliders; changes apply live. S saves config and RGB-D snapshot; SPACE freezes; N/P selects a face; L selects full/local detector; Q quits.",
                flush=True,
            )
        begin = time.monotonic()
        last_print = begin
        display_pacer = DisplayPacer(args.display_hz, begin)
        debug_pacer = DisplayPacer(args.debug_hz, begin)
        previous_sequence = -1
        calibration_id = frame.camera.calibration_id
        while time.monotonic() - begin < args.duration:
            if count:
                frame = source.get(previous_sequence, args.timeout)
            if frame.camera.calibration_id != calibration_id:
                raise ValueError("camera calibration changed during run; start a new run")
            result, _ = pipeline.process(frame)
            if cfg.tracking:
                tracker.update(result)
            result["received_to_result_ms"] = (time.monotonic_ns() - frame.received_monotonic_ns) / 1e6
            result["calibration_dimensions_nominal"] = cfg.dimensions_nominal
            result["vertical_reference_status"] = (
                "provisional_imu"
                if up_report is not None
                else "configured_unverified" if cfg.up_camera is not None else "not_calibrated"
            )
            if pose_publisher is not None:
                pose_publisher.publish(result)
            writer.write(result)
            previous_sequence = frame.sequence
            processed_stamps.append(frame.stamp_ns)
            durations.append(result["timing"]["processing_ms"])
            latencies.append(result["received_to_result_ms"])
            counts.append(len(result["objects"]))
            flow_counts.append(sum(o.get("source") == "current_rgbd_flow" for o in result["objects"]))
            full_detection_frames += int(result.get("temporal", {}).get("detector_ran", False))
            count += 1
            now = time.monotonic()
            if display and display_pacer.ready(now):
                image = overlay(
                    frame, result, dict(processed_fps=count / (now - begin)), smoothed=args.display_smoothed
                )
                if display.submit(image):
                    display_submissions.append(time.monotonic())
                key = display.poll_key()
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    tracker.reset()
                if key in (ord("s"), ord("S")) and debug_display:
                    debug_display.save(pipeline)
            if debug_display:
                if debug_pacer.ready(now):
                    debug_display.update(frame, result, pipeline)
                if debug_display.poll(pipeline) == ord("q"):
                    break
            if now - last_print >= 2:
                print(
                    f"CPU observations: {count / (now-begin):.1f} frames/s; objects={counts[-1]}; last={durations[-1]:.1f}ms",
                    flush=True,
                )
                last_print = now
        complete = True
    except KeyboardInterrupt:
        complete = True
    finally:
        elapsed = time.monotonic() - begin
        display_fps = display.fps.value if display else None
        display_drops = display.dropped if display else 0
        if display:
            display.close()
        if debug_display:
            debug_display.close()
        if pipeline is not None and hasattr(pipeline, "close"):
            pipeline.close()
        usage = process_usage()
        if usage["cpu_seconds"] is not None:
            usage["self_cpu_percent"] = (
                100 * (usage["cpu_seconds"] - initial_usage["cpu_seconds"]) / max(elapsed, 1e-9)
            )
        try:
            if writer:
                writer.close(
                    dict(
                        complete=complete,
                        kind="live_camera",
                        camera=source.statistics(),
                        distinct_processed_frames=count,
                        processed_wall_fps=count / elapsed if elapsed else 0.0,
                        received_but_unprocessed_frames=max(0, source.received_count - count),
                        observed_result_counts=stats(counts),
                        processing_ms=stats(durations),
                        observation_count_histogram={str(n): counts.count(n) for n in sorted(set(counts))},
                        observation_count_transitions=sum(a != b for a, b in zip(counts, counts[1:])),
                        temporal_detector_frames=full_detection_frames,
                        temporal_flow_object_observations=sum(flow_counts),
                        temporal_frames_with_flow=sum(n > 0 for n in flow_counts),
                        receive_to_result_ms=stats(latencies),
                        process_usage=usage,
                        display_submitted_frames=len(display_submissions),
                        display_last_fps=display_fps,
                        display_busy_drops=display_drops,
                        display_submission_interval_ms=stats(
                            [(b - a) * 1000 for a, b in zip(display_submissions, display_submissions[1:])]
                        ),
                        note="No robot commands. Capture-to-output latency not claimed without clock calibration.",
                    )
                )
        finally:
            source.close()


if __name__ == "__main__":
    main()
