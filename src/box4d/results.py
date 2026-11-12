"""Shared observation schema for classical, YOLO and temporal branches."""

from copy import deepcopy

from .geometry import upright_from_camera
from .native import geometry_backend
from .camera_motion import motion_context


TIMING_COMPONENTS = ("detector_ms", "solve_ms", "refinement_ms", "appearance_ms", "proposal_count")


def frame_metadata(frame, cfg, up=None):
    """时间戳与标定属于这一次观测，不能从上一帧结果拷贝。"""
    return dict(
        sequence=frame.sequence,
        stamp_ns=frame.stamp_ns,
        frame_id=frame.camera.frame_id,
        calibration_id=frame.camera.calibration_id,
        received_monotonic_ns=frame.received_monotonic_ns or None,
        plane_backend=cfg.plane_backend,
        geometry_backend=geometry_backend(cfg.geometry_backend),
        up_camera=cfg.up_camera,
        U_from_camera=None if up is None else up.tolist(),
        camera_motion=motion_context(frame, cfg),
    )


def empty_result(frame, cfg, up=None):
    if up is None:
        up = upright_from_camera(cfg.up_camera)
    timing = dict.fromkeys(TIMING_COMPONENTS, 0.0)
    timing["proposal_count"] = 0
    return dict(
        frame_metadata(frame, cfg, up),
        objects=[],
        rejected_faces=[],
        suppressed_hypotheses=[],
        appearance_rejections=[],
        detection_checks=[],
        timing=timing,
    )


def append_detection_checks(result, child, *, source=None, objects=None):
    """Append source-frame traces and return objects with destination indices.

    Geometry stays in original calibrated pixel coordinates. Only diagnostic
    indices change; the child result/debug snapshot remains independently
    addressable. Call refresh_detection_selection after all final filtering.
    """
    destination = result.setdefault("detection_checks", [])
    indices = {}
    for check in child.get("detection_checks", []):
        copied = deepcopy(check)
        original = check["face_index"]
        indices[original] = len(destination)
        copied.setdefault("source_face_index", original)
        copied["face_index"] = len(destination)
        if source is not None:
            copied.setdefault("source_context", []).append(deepcopy(source))
            if "yolo_instance" in source:
                copied["yolo_instance"] = source["yolo_instance"]
        for trial in copied["trials"]:
            trial["face_index"] = copied["face_index"]
            trial["selected"] = False
            trial.pop("selected_object_indices", None)
        destination.append(copied)

    def remap(source_object):
        copied = dict(source_object)
        if "detection_check_index" in copied:
            copied["detection_check_index"] = indices[copied["detection_check_index"]]
        copied["faces"] = [dict(face) for face in source_object.get("faces", [])]
        for face in copied["faces"]:
            if "detection_check_index" in face:
                face["detection_check_index"] = indices[face["detection_check_index"]]
        return copied

    return [remap(obj) for obj in (child["objects"] if objects is None else objects)]


def refresh_detection_selection(result):
    """selected denotes evidence used by the final objects of this result."""
    checks = {check["face_index"]: check for check in result.get("detection_checks", [])}
    for check in checks.values():
        for trial in check["trials"]:
            trial["selected"] = False
            trial.pop("selected_object_indices", None)
    for object_index, obj in enumerate(result["objects"]):
        for evidence in (obj, *obj.get("faces", [])):
            check_index, trial_index = evidence.get("detection_check_index"), evidence.get("detection_trial_index")
            if check_index is None or trial_index is None:
                continue  # Older detector integrations may not supply traces.
            trial = checks[check_index]["trials"][trial_index]
            trial["selected"] = True
            selected = trial.setdefault("selected_object_indices", [])
            if object_index not in selected:
                selected.append(object_index)


def merge_region(result, child, instance_index, *, objects=None):
    """汇总独立 ROI 的诊断；场景去重在所有实例处理完以后统一执行。"""
    for key in TIMING_COMPONENTS:
        result["timing"][key] += child["timing"][key]
    result["timing"]["color_line_fallback_used"] = bool(
        result["timing"].get("color_line_fallback_used") or child["timing"].get("color_line_fallback_used")
    )
    result["rejected_faces"].extend(
        dict(face, yolo_instance=instance_index) for face in child["rejected_faces"]
    )
    for key in ("appearance_rejections", "suppressed_hypotheses"):
        result[key].extend(child[key])
    return append_detection_checks(
        result, child, source=dict(branch="yolo_roi", yolo_instance=instance_index), objects=objects
    )
