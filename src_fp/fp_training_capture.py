"""Save same-frame RGB-D, FP pose and reviewable projection labels."""
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from kfs_pose_controller import render_mesh_depth


class TrainingCapture:
    def __init__(self, output, mesh, mesh_path, class_name='tkk', interval=1.0, automatic=False):
        self.output = Path(output)
        if self.output.exists() and any(self.output.iterdir()):
            raise ValueError('Capture requires a new, empty directory')
        self.vertices = np.asarray(mesh.vertices).copy()
        self.faces = np.asarray(mesh.faces).copy()
        self.class_name = class_name
        self.interval = interval
        self.automatic = automatic
        self.next_capture = 0.
        self.count = 0
        for name in ('images', 'depth', 'ob_in_cam', 'intrinsics', 'masks',
                     'labels', 'labelme', 'preview'):
            (self.output / name).mkdir(parents=True, exist_ok=True)
        mesh_path = Path(mesh_path)
        (self.output / 'capture_config.json').write_text(json.dumps({
            'mesh': str(mesh_path),
            'mesh_sha256': hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
            'class_name': class_name, 'class_id': 0, 'interval_s': interval,
            'automatic_at_start': automatic,
            'annotation_origin': 'FP model projection, requires human review',
            'depth_occlusion_filter': False, 'depth_units': 'metres',
        }, indent=2, ensure_ascii=False) + '\n')
        (self.output / 'classes.txt').write_text(class_name + '\n')
        (self.output / 'README_中文.md').write_text(
            '# FP 训练图像采集\n\n'
            '主窗口按 S 保存一组，C 开关定时采样，R 暂停采样并重新定位，Q 退出。\n\n'
            'images 是无叠加的原始图像；depth 是浮点米制 NPY；ob_in_cam 和 intrinsics '
            '保存同帧位姿及内参。masks 是模型投影，labels 是类别 0 的 YOLO 分割多边形；'
            'labelme 可供人工修改，preview 用于检查对齐。\n\n'
            '投影标签不剔除手或其他物体的遮挡，所有标签均待人工复核。'
            '修改 Labelme JSON 后需重新生成 YOLO labels；不会自动加入已有训练集。\n'
        )

    def handle(self, key, rgb, depth, k, pose, metadata, processed_index):
        now = time.monotonic()
        if key == ord('r'):
            self.automatic = False
        if key == ord('c'):
            self.automatic = not self.automatic
            self.next_capture = now
            print(f"Capture auto: {'ON' if self.automatic else 'OFF'}", flush=True)
        if key not in (ord('q'), 27, ord('r')) and (
                key == ord('s') or self.automatic and now >= self.next_capture):
            self.next_capture = now + self.interval
            if pose is None or np.shape(pose) != (4, 4) or not np.isfinite(pose).all():
                print('Capture skipped: pose unavailable', flush=True)
                return
            self.save(rgb, depth, k, pose, metadata, processed_index)

    def save(self, rgb, depth, k, pose, metadata, processed_index):
        if metadata.get('roi_xyxy') or rgb.shape[:2] != depth.shape:
            raise ValueError('Capture requires full-frame aligned RGB-D; disable dynamic ROI')
        rendered = render_mesh_depth(self.vertices, self.faces, pose, k, depth.shape)
        mask = (rendered > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = [cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
                    for c in contours if cv2.contourArea(c) >= 25]
        polygons = [p for p in polygons if len(p) >= 3]
        if len(polygons) != 1:
            print('Capture skipped: model projection is empty or disconnected', flush=True)
            return
        polygon = polygons[0]
        h, w = depth.shape
        name = f'{self.count:06d}'
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        annotation = {
            'version': '6.3.1', 'flags': {'reviewed': False, 'fp_projection': True},
            'shapes': [dict(label=self.class_name, points=polygon.tolist(), group_id=1,
                            shape_type='polygon', flags={})],
            'imagePath': f'../images/{name}.png', 'imageData': None,
            'imageHeight': h, 'imageWidth': w,
        }
        preview = image.copy()
        preview[mask > 0] = (preview[mask > 0] * .65 + [0, 89.25, 0]).astype(np.uint8)
        cv2.polylines(preview, [polygon], True, (0, 255, 0), 2)
        cv2.putText(preview, f'FP projection - REVIEW {name}', (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        for directory, pixels, extension in [('images', image, '.png'),
                                              ('masks', mask, '.png'),
                                              ('preview', preview, '.jpg')]:
            if not cv2.imwrite(str(self.output / directory / (name + extension)), pixels):
                raise OSError(f'Failed to save {directory}/{name}{extension}')
        np.save(self.output / 'depth' / f'{name}.npy', np.asarray(depth, dtype=np.float32))
        np.savetxt(self.output / 'ob_in_cam' / f'{name}.txt', pose, fmt='%.12g')
        np.savetxt(self.output / 'intrinsics' / f'{name}.txt', k, fmt='%.12g')
        if self.count == 0:
            np.savetxt(self.output / 'cam_K.txt', k, fmt='%.12g')
        (self.output / 'labelme' / f'{name}.json').write_text(
            json.dumps(annotation, indent=2, ensure_ascii=False) + '\n')
        normalized = polygon.astype(float) / [w, h]
        (self.output / 'labels' / f'{name}.txt').write_text(
            '0 ' + ' '.join(f'{v:.8f}' for v in normalized.ravel()) + '\n')
        with (self.output / 'manifest.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(
                sample=name, source_sequence=metadata['sequence'], stamp=metadata['stamp'],
                frame_id=metadata['frame_id'], processed_index=processed_index,
                image=f'images/{name}.png', reviewed=False,
                annotation_origin='FP projection without depth occlusion filtering')) + '\n')
        self.count += 1
        print(f'CAPTURE SAVED {name}: {self.output} (total {self.count})', flush=True)
