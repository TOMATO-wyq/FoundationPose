"""CPU debug overlays, with separate labels for observation and model inference."""

import cv2
import numpy as np

from .geometry import angle_difference, cube_vertices, cube_edges, transform_points, rotation_z


def _dashed(image, a, b, color):
    for t in np.arange(0.0, 1.0, 0.16):
        start = np.rint(a + t * (b - a)).astype(int)
        end = np.rint(a + min(1.0, t + 0.08) * (b - a)).astype(int)
        cv2.line(image, tuple(start), tuple(end), color, 1, cv2.LINE_AA)


def overlay(frame, result, rates=None, smoothed=False):
    image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
    for obj in result["objects"]:
        color = (0, 190, 255) if obj["evidence"] == "model_inferred" else (60, 245, 70)
        if obj.get("source") == "current_rgbd_flow":
            color = (255, 210, 40)
        for face in obj["faces"]:
            quad = np.rint(face["quad"]).astype(np.int32)
            if face["source"] == "rgbd_feature_tracking":
                cv2.polylines(image, [quad], True, color, 1, cv2.LINE_AA)
                continue  # Projected model corners are not newly detected corners.
            if face["source"] == "bottom_occlusion_model":
                visible = np.rint(face["evidence"]["visible_outline_quad"]).astype(np.int32)
                cv2.polylines(image, [visible], True, (60, 245, 70), 1, cv2.LINE_AA)
                # The hidden bottom is model geometry, never an observed corner.
                for a, b in ((quad[2], quad[3]), (visible[2], quad[2]), (visible[3], quad[3])):
                    _dashed(image, a, b, color)
                continue
            cv2.polylines(image, [quad], True, color, 1, cv2.LINE_AA)
            for index, point in enumerate(quad):
                cv2.circle(image, tuple(point), 3, color, -1)
                cv2.putText(
                    image, str(index), tuple(point + (3, -3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1
                )
        raw_pose = np.asarray(obj["T_camera_from_box"])
        pose = raw_pose.copy()
        yaw = obj["yaw_u_rad"]
        if smoothed and "center_camera_smoothed_m" in obj:
            pose[:3, 3] = obj["center_camera_smoothed_m"]
            filtered_yaw = obj.get("yaw_u_smoothed_rad")
            if filtered_yaw is not None and result.get("U_from_camera") is not None:
                yaw = filtered_yaw
                up = np.asarray(result["U_from_camera"])
                delta = angle_difference(yaw, obj["yaw_u_rad"])
                pose[:3, :3] = up.T @ rotation_z(delta) @ up @ raw_pose[:3, :3]
        vertices = transform_points(pose, cube_vertices(obj["size_m"]))
        if np.all(vertices[:, 2] > 0.01):
            uv = np.rint(frame.camera.project(vertices)).astype(int)
            for a, b in cube_edges():
                if any(f["source"] == "bottom_occlusion_model" for f in obj["faces"]):
                    _dashed(image, uv[a], uv[b], color)
                else:
                    cv2.line(image, tuple(uv[a]), tuple(uv[b]), color, 1, cv2.LINE_AA)
        center = np.rint(frame.camera.project(pose[:3, 3:4].T)[0]).astype(int)
        cv2.drawMarker(image, tuple(center), color, cv2.MARKER_CROSS, 14, 2)
        angle = "yaw: UNREFERENCED" if yaw is None else f"yaw={np.rad2deg(yaw):.1f}deg mod90"
        prefix = f"ID{obj.get('track_id','?')} {obj['size_m']*100:.0f}cm"
        text = f"{prefix} {angle}"
        location = (
            max(4, min(frame.camera.width - 220, center[0] - 70)),
            max(55, min(frame.camera.height - 35, center[1] - 8)),
        )
        cv2.putText(image, text, location, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, location, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        xyz = pose[:3, 3]
        subtitle = f"C=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})m {obj['evidence']}"
        cv2.putText(
            image,
            subtitle,
            (location[0], location[1] + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
        appearance = obj.get("appearance")
        if appearance:
            status = (
                appearance["orientation_status"]
                if appearance["model"]
                else f"surface={appearance['surface_label']}"
            )
            cv2.putText(
                image,
                f"COLOR: {status}",
                (location[0], location[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                color,
                1,
                cv2.LINE_AA,
            )
            axis = appearance["red_axis_camera"]
            if axis is not None:
                axis = pose[:3, :3] @ raw_pose[:3, :3].T @ np.asarray(axis)
                endpoints = np.array([pose[:3, 3], pose[:3, 3] + np.asarray(axis) * obj["size_m"] * 0.7])
                if np.all(endpoints[:, 2] > 0.01):
                    a, b = np.rint(frame.camera.project(endpoints)).astype(int)
                    cv2.arrowedLine(image, tuple(a), tuple(b), (40, 40, 255), 2, cv2.LINE_AA, tipLength=0.2)
    processing = result["timing"]["processing_ms"]
    text = f"CPU {processing:.1f}ms  frame={frame.sequence}  observed={len(result['objects'])}"
    if "temporal" in result:
        text += f" tracked={result['temporal']['output_flow_objects']}"
    cv2.rectangle(image, (0, 0), (frame.camera.width, 43), (20, 20, 20), -1)
    cv2.putText(image, text, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    note = (
        "Nominal dimensions; camera optical coordinates"
        if any(o["dimensions_nominal"] for o in result["objects"])
        else "Camera optical coordinates; current-frame observations"
    )
    if rates:
        note = "  ".join(f"{k}={v:.1f}" for k, v in rates.items())
    if result.get("vertical_reference_status") == "provisional_imu":
        note += "  UP: provisional IMU"
    if smoothed:
        note += "  DISPLAY: filtered; raw logged"
    cv2.putText(image, note, (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (210, 210, 210), 1, cv2.LINE_AA)
    if not result["objects"]:
        message = (
            "POSE REJECTED - see 08 Current tracking"
            if result.get("frame_status") == "partial_or_rejected"
            else "SEARCHING - no target observation"
        )
        cv2.putText(
            image,
            message,
            (8, frame.camera.height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 190, 255),
            1,
            cv2.LINE_AA,
        )
    return image
