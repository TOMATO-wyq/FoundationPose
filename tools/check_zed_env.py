#!/usr/bin/env python3
"""Validate the ZED SDK and report cameras visible inside the container."""

import sys

import pyzed.sl as sl


def main() -> int:
    sdk_version = sl.Camera().get_sdk_version()
    print(f"ZED SDK: {sdk_version}")
    if not sdk_version.startswith("5.5"):
        print("ERROR: expected ZED SDK 5.5.x", file=sys.stderr)
        return 2

    devices = sl.Camera.get_device_list()
    if not devices:
        print("No ZED camera detected. Check USB connection and container --privileged settings.")
        return 3

    for device in devices:
        print(
            "Camera: "
            f"model={device.camera_model} serial={device.serial_number} "
            f"state={device.camera_state} id={device.id}"
        )

    camera = sl.Camera()
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR: camera open failed: {status}", file=sys.stderr)
        return 4

    info = camera.get_camera_information()
    calibration = info.camera_configuration.calibration_parameters.left_cam
    resolution = info.camera_configuration.resolution
    print(
        f"Opened: {info.camera_model} SN{info.serial_number}, "
        f"{resolution.width}x{resolution.height}, "
        f"fx={calibration.fx:.3f}, fy={calibration.fy:.3f}, "
        f"cx={calibration.cx:.3f}, cy={calibration.cy:.3f}"
    )
    camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
