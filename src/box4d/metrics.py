"""Explicit compute metrics and optional known-pose evaluation."""

import platform
import sys

import cv2
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment

from .geometry import angle_difference, upright_from_camera, yaw_from_pose


def stats(values):
    a = np.asarray(values, np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(n=0, mean=None, p50=None, p95=None, max=None)
    return dict(
        n=len(a),
        mean=float(a.mean()),
        p50=float(np.percentile(a, 50)),
        p95=float(np.percentile(a, 95)),
        max=float(a.max()),
    )


def environment(cfg=None):
    result = dict(
        python=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        numpy=np.__version__,
        opencv=cv2.__version__,
        scipy=scipy.__version__,
        opencv_threads=cv2.getNumThreads(),
        opencl_enabled=cv2.ocl.useOpenCL(),
        dependencies="NumPy/OpenCV/SciPy; no Torch/CUDA inference",
    )
    if cfg is not None:
        from .native import geometry_backend, geometry_kernel

        result["geometry_backend"] = geometry_backend(cfg.geometry_backend)
        result["plane_backend"] = cfg.plane_backend
        kernel = geometry_kernel(cfg.geometry_backend)
        if kernel is not None:
            result.update(
                native_api_version=kernel.api_version,
                native_opencv=kernel.opencv_version,
                native_opencv_threads=kernel.opencv_num_threads(),
            )
        if cfg.yolo_weights:
            result["dependencies"] = f"NumPy/OpenCV/SciPy; Torch/Ultralytics on {cfg.yolo_device}"
    return result


def process_usage():
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF)
        return dict(
            cpu_seconds=value.ru_utime + value.ru_stime,
            peak_rss_mib=value.ru_maxrss / (1024 if sys.platform != "darwin" else 1024 * 1024),
        )
    except ImportError:
        return dict(cpu_seconds=None, peak_rss_mib=None)


def summarize(results):
    times = [r["timing"]["processing_ms"] for r in results]
    total = sum(times) / 1000
    counts = [len(r["objects"]) for r in results]
    asynchronous = any(r.get("temporal", {}).get("async_detection", False) for r in results)
    background = [
        r["temporal"]["background_detection_ms"]
        for r in results
        if r.get("temporal", {}).get("background_detection_ms") is not None
    ]
    return dict(
        processing_ms=stats(times),
        detector_ms=stats([r["timing"]["detector_ms"] for r in results]),
        solve_ms=stats([r["timing"]["solve_ms"] for r in results]),
        refinement_ms=stats([r["timing"]["refinement_ms"] for r in results]),
        appearance_ms=stats([r["timing"].get("appearance_ms", 0.0) for r in results]),
        # Foreground time excludes a concurrent detector. Its inverse
        # cannot represent the throughput of the entire pipeline.
        compute_fps_excluding_io_and_display=len(times) / total if total and not asynchronous else None,
        async_detection=asynchronous,
        background_detection_ms=stats(background),
        frames_with_observation=sum(bool(r["objects"]) for r in results),
        object_observations=sum(len(r["objects"]) for r in results),
        observation_count_histogram={str(n): counts.count(n) for n in sorted(set(counts))},
        observation_count_transitions=sum(a != b for a, b in zip(counts, counts[1:])),
        temporal_detector_frames=sum(bool(r.get("temporal", {}).get("detector_ran")) for r in results),
        temporal_flow_object_observations=sum(
            o.get("source") == "current_rgbd_flow" for r in results for o in r["objects"]
        ),
        temporal_frames_with_flow=sum(
            any(o.get("source") == "current_rgbd_flow" for o in r["objects"]) for r in results
        ),
        note=(
            "Async processing_ms covers foreground work only; background detection is reported separately. "
            "No whole-pipeline compute FPS is inferred."
            if asynchronous
            else "Compute throughput is not measured live camera FPS; detection coverage reported separately."
        ),
    )


def evaluate(results, truth_rows, cfg):
    truth = {int(r["stamp_ns"]): r for r in truth_rows}
    errors, yaws, misses, false_positives, total, hits, within_target = [], [], 0, 0, 0, 0, 0
    frame_details = []
    color_total = color_matched = axis_resolved = state_resolved = state_correct = 0
    axis_errors = []
    u_from_c = upright_from_camera(cfg.up_camera)
    for result in results:
        reference = truth.get(result["stamp_ns"])
        if reference is None:
            continue
        targets = [o for o in reference["objects"] if any(abs(o["size_m"] - s) < 1e-6 for s in cfg.sizes_m)]
        observations = result["objects"]
        total += len(targets)
        color_total += sum("red_axis_camera" in t for t in targets)
        cost = np.full((len(targets), len(observations)), 1e6)
        for i, target in enumerate(targets):
            for j, observation in enumerate(observations):
                if abs(target["size_m"] - observation["size_m"]) < 1e-6:
                    cost[i, j] = np.linalg.norm(
                        np.asarray(target["center_camera_m"]) - observation["center_camera_m"]
                    )
        matches = []
        if cost.size:
            rows, columns = linear_sum_assignment(cost)
            for i, j in zip(rows, columns):
                if cost[i, j] > targets[i]["size_m"] * 0.4:
                    continue
                error = float(cost[i, j] * 1000)
                errors.append(error)
                target_yaw = yaw_from_pose(np.asarray(targets[i]["T_camera_from_box"]), u_from_c)
                yaw_error = None
                if target_yaw is not None and observations[j]["yaw_u_rad"] is not None:
                    yaw_error = float(
                        np.rad2deg(abs(angle_difference(observations[j]["yaw_u_rad"], target_yaw)))
                    )
                    yaws.append(yaw_error)
                matches.append(
                    dict(
                        reference_id=targets[i]["id"],
                        observation=j,
                        position_error_mm=error,
                        yaw_error_deg=yaw_error,
                    )
                )
                if "red_axis_camera" in targets[i]:
                    color_matched += 1
                    appearance = observations[j].get("appearance", {})
                    direction = appearance.get("red_axis_camera")
                    if direction is not None:
                        axis_resolved += 1
                        reference_axis = np.asarray(targets[i]["red_axis_camera"])
                        axis_errors.append(
                            float(np.rad2deg(np.arccos(np.clip(np.dot(direction, reference_axis), -1, 1))))
                        )
                    state = appearance.get("orientation_status")
                    if state in ("red_up", "blue_up", "sideways"):
                        state_resolved += 1
                        state_correct += int(state == targets[i].get("color_orientation"))
                within_target += int(error <= 15 and yaw_error is not None and yaw_error <= 2)
        hits += len(matches)
        misses += len(targets) - len(matches)
        false_positives += len(observations) - len(matches)
        frame_details.append(
            dict(
                stamp_ns=result["stamp_ns"],
                scene=reference.get("scene"),
                expected=len(targets),
                observed=len(observations),
                matches=matches,
            )
        )
    return dict(
        target_observations=total,
        matched=hits,
        missed=misses,
        false_positive_observations=false_positives,
        matched_fraction=None if not total else hits / total,
        within_15mm_2deg=within_target,
        within_target_fraction=None if not total else within_target / total,
        position_error_mm=stats(errors),
        yaw_error_mod90_deg=stats(yaws),
        frames=frame_details,
        color_orientation=dict(
            target_observations=color_total,
            pose_matched=color_matched,
            red_axis_resolved=axis_resolved,
            unresolved_or_missing_axis=color_total - axis_resolved,
            state_resolved=state_resolved,
            state_correct=state_correct,
            state_wrong=state_resolved - state_correct,
            red_axis_error_deg=stats(axis_errors),
        ),
        note="Errors describe matched observations; missed/false outputs are counted separately. Synthetic truth is not real-camera accuracy.",
    )
