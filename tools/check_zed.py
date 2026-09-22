"""Capture a rectified left image, aligned metric depth and intrinsics, without GUI."""
import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl


def main():
    camera = sl.Camera()
    params = sl.InitParameters()
    params.camera_resolution = sl.RESOLUTION.HD720
    params.camera_fps = 30
    params.depth_mode = sl.DEPTH_MODE.NEURAL
    params.coordinate_units = sl.UNIT.METER
    params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    result = camera.open(params)
    if result != sl.ERROR_CODE.SUCCESS:
        camera.close()
        raise RuntimeError(f'ZED open failed: {result}')
    try:
        info = camera.get_camera_information()
        print(f'SDK: {camera.get_sdk_version()}; camera: {info.camera_model}; serial: {info.serial_number}', flush=True)
        image, depth = sl.Mat(), sl.Mat()
        runtime = sl.RuntimeParameters()
        deadline = time.monotonic() + 90
        frames = 0
        while time.monotonic() < deadline:
            if camera.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                time.sleep(0.02)
                continue
            frames += 1
            if frames < 30:
                continue
            for result in (camera.retrieve_image(image, sl.VIEW.LEFT),
                           camera.retrieve_measure(depth, sl.MEASURE.DEPTH)):
                if result != sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError(f'ZED retrieve failed: {result}')
            meters = depth.get_data().copy()
            valid = np.isfinite(meters) & (meters > 0)
            if not valid.any():
                continue
            folder = Path('data/zed') / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            folder.mkdir(parents=True, exist_ok=False)
            # ZED's left image is BGRA; OpenCV saves BGR with correct PNG colors.
            if not cv2.imwrite(str(folder / 'left.png'), image.get_data()[:, :, :3]):
                raise RuntimeError('Failed to save left.png')
            np.save(folder / 'depth_m.npy', meters)
            cam = info.camera_configuration.calibration_parameters.left_cam
            intrinsic = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]])
            np.savetxt(folder / 'cam_K.txt', intrinsic)
            summary = dict(camera=str(info.camera_model), serial=int(info.serial_number),
                           depth_units='meters', aligned_to='rectified left image',
                           shape=list(meters.shape), valid_fraction=float(valid.mean()),
                           median_depth_m=float(np.median(meters[valid])))
            (folder / 'capture.json').write_text(json.dumps(summary, indent=2))
            if os.geteuid() == 0 and 'OUTPUT_UID' in os.environ:
                for path in [folder, *folder.iterdir()]:
                    os.chown(path, int(os.environ['OUTPUT_UID']), int(os.environ['OUTPUT_GID']))
            print(json.dumps(summary, indent=2))
            print(f'Capture saved: {folder.resolve()}')
            return
        raise RuntimeError('No valid depth frame within 90 seconds; check lighting, scene texture and camera distance.')
    finally:
        camera.close()


if __name__ == '__main__':
    main()
