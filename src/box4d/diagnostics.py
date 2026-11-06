"""Source-qualified diagnostics for delayed detector observations."""

from collections import Counter
from copy import deepcopy


def detector_snapshot(source, current, result, reports, elapsed_ms, max_age_s):
    """Keep detector pixels/timing with their source instead of the live frame.

    Detector results use the shared observation schema. Optional summary keys
    also permit the small detector doubles used by downstream integrations.
    """
    age_ms = (current.stamp_ns - source.stamp_ns) / 1e6
    unusable = []
    if source.camera.calibration_id != current.camera.calibration_id:
        unusable.append("calibration_mismatch")
    if age_ms < 0:
        unusable.append("source_in_future")
    elif age_ms > max_age_s * 1000:
        unusable.append("result_too_old")
    rejected = deepcopy(result.get("rejected_faces", []))
    appearance = deepcopy(result.get("appearance_rejections", []))
    suppressed = deepcopy(result.get("suppressed_hypotheses", []))
    status = result.get("frame_status") or (
        "observed" if result["objects"] else "partial_or_rejected" if rejected or appearance else "no_target"
    )
    return dict(
        # Preserve the original summary keys for JSON/report consumers.
        stamp_ns=source.stamp_ns,
        sequence=source.sequence,
        object_count=len(result["objects"]),
        objects=deepcopy(result["objects"]),
        detection_checks=deepcopy(result.get("detection_checks", [])),
        rejected_reason_counts=dict(Counter(face["reason"] for face in rejected)),
        source_frame=dict(
            sequence=source.sequence,
            stamp_ns=source.stamp_ns,
            calibration_id=source.camera.calibration_id,
            frame_id=source.camera.frame_id,
            camera=source.camera.to_dict(),
            received_monotonic_ns=source.received_monotonic_ns or None,
        ),
        current_frame=dict(
            sequence=current.sequence,
            stamp_ns=current.stamp_ns,
            calibration_id=current.camera.calibration_id,
        ),
        age_ms=age_ms,
        max_age_ms=max_age_s * 1000,
        compatible_with_current_frame=not unusable,
        used_for_current_frame_status=not unusable,
        unusable_reasons=unusable,
        frame_status=status,
        has_rejections=bool(rejected or appearance or suppressed or status == "partial_or_rejected"),
        rejected_faces=rejected,
        appearance_rejections=appearance,
        suppressed_hypotheses=suppressed,
        timing=deepcopy(result["timing"]),
        background_detection_ms=elapsed_ms,
        local_searches=deepcopy(reports),
        yolo=deepcopy(result.get("yolo")),
    )


def has_current_rejections(result):
    """Only compatible, consumed detector evidence can describe this frame."""
    detector = result.get("detector_diagnostics")
    return bool(
        result.get("rejected_faces")
        or result.get("appearance_rejections")
        or (detector and detector["used_for_current_frame_status"] and detector["has_rejections"])
    )
