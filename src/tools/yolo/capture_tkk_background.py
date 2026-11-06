#!/usr/bin/env python3
"""Capture user-confirmed TKK-free backgrounds as empty-label segmentation samples."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from datetime import datetime
import hashlib
import json
import os
import shutil
import time

import cv2
import numpy as np

from tools.yolo.compare_tkk_seg import RosFrames, write_json


PROJECT = Path(__file__).resolve().parents[3]
TITLE = 'TKK BACKGROUND | SPACE: confirm NO TKK and save | Q: finish'


def save_background(output, image, stamp, sequence, number, split):
    if shutil.disk_usage(output).free < 1024 ** 3:
        raise RuntimeError('Less than 1 GiB free disk space; capture stopped')
    key = f'{number:06d}'
    for directory in ('images', 'labels', 'labelme'):
        (output / directory).mkdir(exist_ok=True)
    image_path = output / 'images' / (key + '.png')
    annotation_path = output / 'labelme' / (key + '.json')
    label_path = output / 'labels' / (key + '.txt')
    if any(path.exists() for path in (image_path, annotation_path, label_path)):
        raise FileExistsError(f'Capture would overwrite {key}')
    height, width = image.shape[:2]
    if not cv2.imwrite(str(image_path), image):
        raise OSError(f'Failed to save {image_path}')
    label_path.write_text('', encoding='utf-8')
    annotation = dict(version='6.3.1', flags=dict(reviewed=True, confirmed_negative=True),
        shapes=[], imagePath=f'../images/{key}.png', imageData=None,
        imageHeight=height, imageWidth=width,
        annotation_origin='User confirmed no TKK in the displayed RGB frame by pressing SPACE')
    write_json(annotation_path, annotation)
    row = dict(id=output.name + '_' + key, sample=key, group=output.name, split=split,
        image=f'images/{key}.png', label=f'labels/{key}.txt', annotation=f'labelme/{key}.json',
        stamp=stamp, source_sequence=sequence, width=width, height=height,
        confirmed_negative=True, reviewed=True, confirmation='manual_SPACE_no_TKK',
        image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        saved_at=datetime.now().astimezone().isoformat())
    with (output / 'manifest.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', default='/camera/camera/color/image_raw')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--split', choices=('train', 'val'), default='train')
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--check', action='store_true', help='Read one camera frame without saving or confirming a negative')
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout must be positive')
    if not args.check and not os.environ.get('DISPLAY'):
        parser.error('Run in a desktop terminal, or use --check for a read-only camera test')
    output = (args.output or PROJECT / 'outputs' /
              datetime.now().strftime('tkk_background_%Y%m%d_%H%M%S_%f')).resolve()
    if (output / 'background_capture.json').exists():
        parser.error('This output already contains a background capture; choose a new directory')
    output.mkdir(parents=True, exist_ok=True)
    if not args.check:
        write_json(output / 'background_capture.json', dict(
            target_class='tkk', class_id=0, split=args.split, topic=args.topic,
            camera='D435i', mode='manual_confirmed_negative',
            ros_domain_id=os.environ.get('ROS_DOMAIN_ID', '0'),
            confirmation='Each SPACE press confirms the displayed frame contains no TKK',
            automatic_capture=False, training_started=False))
        (output / 'classes.txt').write_text('tkk\n')
        (output / 'README_中文.md').write_text(
            '# TKK 纯背景负样本\n\n'
            '每张由拍摄者确认画面中完全没有 tkk 后按空格保存。images 是无叠加原始 PNG，'
            'labels 是空 YOLO 标签，labelme 是 shapes=[] 的已确认负样本 JSON，'
            'manifest.jsonl 记录时间戳、来源、集合、SHA256 和确认方式。\n\n'
            '如果误拍了带目标的图片，请重新标出所有目标，或明确排除该图，不能继续作为负样本。'
            '同一批次默认只进入一个集合；--split val 用于另外拍摄的验证批次。'
            '本程序不启动 FP、不生成目标投影，也不自动训练或修改旧数据。\n', encoding='utf-8')
    source = None
    count = 0
    last_saved_stamp = None
    saved_notice_until = 0.
    started = last_received = time.monotonic()
    last_sequence = -1
    error = None
    print(f'BACKGROUND output={output} split={args.split}', flush=True)
    print('Remove every TKK from the image. SPACE confirms NO TKK and saves one background; Q exits.', flush=True)
    try:
        source = RosFrames(args.topic, node_name='tkk_background_photo_capture')
        if not args.check:
            cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(TITLE, 960, 800)
        while True:
            current = source.latest()
            now = time.monotonic()
            if current is not None and current[3] != last_sequence:
                last_sequence = current[3]
                last_received = current[2]
            if now - last_received > args.timeout:
                raise RuntimeError('No fresh RGB frames; check camera and ROS domain')
            fresh = current is not None and now - current[2] < 1.
            if args.check:
                if fresh:
                    print('CAMERA_CHECK ' + json.dumps(dict(shape=list(current[0].shape),
                          stamp=current[1], source_sequence=current[3], saved=0,
                          negative_confirmation=False)), flush=True)
                    return
                time.sleep(.01)
                continue
            image = current[0] if current is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            header = np.zeros((82, image.shape[1], 3), dtype=np.uint8)
            cv2.putText(header, f'BACKGROUND ONLY | {args.split} | saved: {count}',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .63, (0, 240, 255), 2)
            cv2.putText(header, 'SPACE = I confirm NO TKK in this image + SAVE',
                        (10, 51), cv2.FONT_HERSHEY_SIMPLEX, .53, (240, 240, 240), 1)
            status = 'WAITING: no fresh frame' if not fresh else (
                     'SAVED. Change angle / distance / background before next photo.'
                     if now < saved_notice_until else 'Q: finish | Original RGB saved without this header')
            cv2.putText(header, status, (10, 74), cv2.FONT_HERSHEY_SIMPLEX, .43,
                        (0, 200, 0) if fresh else (0, 0, 255), 1)
            cv2.imshow(TITLE, np.vstack((header, image)))
            key = cv2.waitKey(15) & 255
            if key in (ord('q'), 27) or cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 32:
                if not fresh or time.monotonic() - current[2] >= 1.:
                    print('Not saved: no fresh RGB frame', flush=True)
                    continue
                stamp_key = (current[1]['sec'], current[1]['nanosec'])
                if stamp_key == last_saved_stamp:
                    continue
                row = save_background(output, image, current[1], current[3], count, args.split)
                count += 1
                last_saved_stamp = stamp_key
                saved_notice_until = time.monotonic() + 1.5
                print(f'SAVED background {count}: {output / row["image"]}', flush=True)
    except KeyboardInterrupt:
        pass
    except Exception as caught:
        error = str(caught)
        raise
    finally:
        if source:
            source.close()
        if not args.check:
            cv2.destroyAllWindows()
        write_json(output / 'capture_summary.json', dict(
            state='failed' if error else 'complete', error=error, saved_images=count,
            split=args.split, check_only=args.check,
            elapsed_seconds=round(time.monotonic() - started, 2),
            finished_at=datetime.now().astimezone().isoformat()))
        print(f'FINISHED backgrounds={count} output={output}', flush=True)


if __name__ == '__main__':
    main()
