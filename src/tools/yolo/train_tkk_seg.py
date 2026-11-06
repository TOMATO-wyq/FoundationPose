#!/usr/bin/env python3
"""Train TKK segmentation from saved, manually reviewed Labelme snapshot annotations."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
import shutil
import time
import traceback


PROJECT = Path(__file__).resolve().parents[3]
PRETRAINED = Path('/home/tomato/code/box_seg_annotation/models/yolo11n-seg.pt')


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(snapshot, output, background_sources=(), background_val_fraction=.2, background_gap=10):
    import cv2
    import numpy as np
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset, verify_image_label

    rows = json.loads((snapshot / 'manifest.json').read_text())
    expected = {row['id'] for row in rows}
    if len(expected) != len(rows) or expected != {p.stem for p in (snapshot / 'labelme').glob('*.json')}:
        raise ValueError('复核目录的 manifest 与 Labelme JSON 不一一对应')
    output.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'val'):
        for kind in ('images', 'labels'):
            (output / kind / split).mkdir(parents=True)
    (output / 'source_annotations').mkdir()
    accepted, excluded, seen, changed, boundary = [], [], {}, [], []
    for row in rows:
        key, split = row['id'], row['split']
        if Path(key).name != key or split not in ('train', 'val'):
            raise ValueError(f'无效样本 ID 或集合：{key}')
        annotation_path = snapshot / 'labelme' / (key + '.json')
        annotation_hash = digest(annotation_path)
        annotation = json.loads(annotation_path.read_text())
        shutil.copy2(annotation_path, output / 'source_annotations' / annotation_path.name)
        original = json.loads((snapshot / 'source_annotations' / annotation_path.name).read_text())
        if annotation['shapes'] != original['shapes']:
            changed.append(key)
        if not annotation['shapes']:
            excluded.append(dict(id=key, split=split, reason='empty_annotation_after_review',
                                 source_annotation=str(annotation_path), sha256=annotation_hash))
            continue
        source_image = snapshot / row['image']
        pixels = cv2.imread(str(source_image))
        if pixels is None:
            raise ValueError(f'无法读取图片：{source_image}')
        height, width = pixels.shape[:2]
        if (annotation.get('imageWidth'), annotation.get('imageHeight')) != (width, height):
            raise ValueError(f'标注尺寸与图片不符：{key}')
        lines, polygons, groups = [], [], set()
        for index, shape in enumerate(annotation['shapes']):
            if shape['label'] != 'tkk' or shape.get('shape_type') != 'polygon':
                raise ValueError(f'只接受 tkk 多边形：{key}')
            group = shape.get('group_id')
            if group is not None and group in groups:
                raise ValueError(f'同一实例包含多个轮廓，需要合并后再训练：{key}')
            groups.add(group if group is not None else ('shape', index))
            points = np.asarray(shape['points'], dtype=np.float64)
            if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                    or not np.isfinite(points).all() or len(np.unique(points, axis=0)) < 3
                    or (points < -1e-6).any() or (points > [width + 1e-6, height + 1e-6]).any()
                    or abs(cv2.contourArea(points.astype(np.float32))) < 1):
                raise ValueError(f'无效多边形：{key}')
            # Labelme permits continuous image boundaries x=width, y=height;
            # YOLO represents those boundaries by normalized coordinate 1.
            points = np.clip(points, [0, 0], [width, height])
            if (points > [width - 1, height - 1]).any():
                boundary.append(key)
            polygons.append(points)
            lines.append('0 ' + ' '.join(f'{value:.8f}' for value in (points / [width, height]).ravel()))
        image_hash = digest(source_image)
        if image_hash != row['source_image_sha256']:
            raise ValueError(f'图片内容已改变：{source_image}')
        if image_hash in seen and seen[image_hash] != split:
            raise ValueError(f'完全相同图片跨训练/验证集合：{key}')
        seen[image_hash] = split
        target = output / 'images' / split / (key + '.png')
        # Preserve the reviewed snapshot and avoid copying another gigabyte of RGB.
        target.symlink_to(source_image.resolve())
        label = output / 'labels' / split / (key + '.txt')
        label.write_text('\n'.join(lines) + '\n')
        checked = verify_image_label((str(target), str(label), '', False, 1, 0, 0, False))
        if checked[0] is None or len(checked[3]) != len(polygons) or checked[-1]:
            raise ValueError(f'官方分割标签检查失败：{key} {checked[-1]}')
        if digest(annotation_path) != annotation_hash:
            raise ValueError(f'导出时标注仍在修改：{key}')
        accepted.append(dict(row, image=str(target.relative_to(output)),
            label=str(label.relative_to(output)), instances=len(polygons),
            reviewed_annotation=str(annotation_path), reviewed_annotation_sha256=annotation_hash,
            source_reviewed_flag=annotation.get('flags', {}).get('reviewed') is True,
            training_approved=True, approval='User confirmed annotation complete; use saved Labelme polygons'))
    from tools.yolo.tkk_background_training import append_backgrounds
    background_report = append_backgrounds(output, accepted, excluded, seen, background_sources,
                                            digest, background_val_fraction, background_gap)
    counts = dict(Counter(row['split'] for row in accepted))
    if not all(counts.get(split, 0) for split in ('train', 'val')):
        raise ValueError('训练集或验证集为空')
    (output / 'data.yaml').write_text(
        f'path: {json.dumps(str(output))}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: tkk\n')
    data = check_det_dataset(str(output / 'data.yaml'), autodownload=False)
    loader = {}
    for split in ('train', 'val'):
        dataset = YOLODataset(img_path=data[split], data=data, task='segment', imgsz=640,
                              augment=False, cache=False, batch_size=8)
        if len(dataset) != counts[split] or any(len(r['segments']) != len(r['cls']) for r in dataset.labels):
            raise ValueError(f'{split} 官方分割加载数量不符')
        positive_index = next((i for i, row in enumerate(dataset.labels) if len(row['cls'])), None)
        if positive_index is None or not dataset[positive_index]['masks'].any():
            raise ValueError(f'{split} 未生成分割 mask')
        negative_index = next((i for i, row in enumerate(dataset.labels) if not len(row['cls'])), None)
        if negative_index is not None and dataset[negative_index]['masks'].any():
            raise ValueError(f'{split} 背景图片不应产生目标 mask')
        loader[split] = dict(images=len(dataset), instances=sum(len(r['cls']) for r in dataset.labels),
                            background_images=sum(not len(r['cls']) for r in dataset.labels))
    write_json(output / 'manifest.json', accepted)
    write_json(output / 'excluded_samples.json', excluded)
    report = dict(source_snapshot=str(snapshot),
                  source_images=len(rows) + sum(batch['images'] for batch in background_report['batches']),
                  source_positive_snapshot_images=len(rows), counts=counts,
                  included=len(accepted),
                  excluded_empty=sum(row['reason'] == 'empty_annotation_after_review' for row in excluded),
                  background=background_report,
                  changed_annotations=len(changed), changed_annotation_ids=changed,
                  image_boundary_annotations=sorted(set(boundary)), official_loader=loader,
                  approval='User explicitly confirmed saved annotations and requested training',
                  original_review_flags_preserved=True,
                  split='Existing capture-date split retained; no resampling of adjacent frames',
                  source_annotation_backup=str(output / 'source_annotations'))
    write_json(output / 'export_report.json', report)
    print('PREPARED ' + json.dumps({k: v for k, v in report.items() if not k.endswith('_ids')},
                                    ensure_ascii=False), flush=True)
    return report


def train(args, job, data):
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError('未检测到 CUDA')
    started = time.monotonic()
    run_dir = PROJECT / 'runs' / args.name
    from tools.yolo.tkk_background_training import evaluate_backgrounds
    initialization = dict(weights=str(args.resume or args.model),
                          weights_sha256=digest(args.resume or args.model))
    if not args.resume:
        baseline = YOLO(str(args.model))
        epoch = (baseline.ckpt or {}).get('epoch', -1)
        initialization['completed_epochs_in_checkpoint'] = epoch + 1 if epoch >= 0 else None
        baseline_report = evaluate_backgrounds(baseline, data, args, job / 'background_baseline.json')
        if baseline_report:
            print('BACKGROUND_BASELINE ' + json.dumps(baseline_report), flush=True)
        del baseline
        torch.cuda.empty_cache()
    # Load a fresh model for training: prediction can fuse layers in place.
    model = YOLO(str(args.resume or args.model))

    def progress(trainer):
        completed = trainer.epoch + 1
        elapsed = time.monotonic() - started
        status = dict(state='training', completed_epochs=completed, target_epochs=trainer.epochs,
                      updated_at=datetime.now().astimezone().isoformat(),
                      elapsed_seconds=round(elapsed, 1),
                      estimated_remaining_seconds=round(elapsed / completed * (trainer.epochs - completed)),
                      run_dir=str(trainer.save_dir),
                      best_weights=str(trainer.best), last_weights=str(trainer.last),
                      initialization=initialization,
                      metrics={key: float(value) for key, value in trainer.metrics.items()})
        write_json(job / 'status.json', status)
        print('EPOCH_STATUS ' + json.dumps(status, ensure_ascii=False), flush=True)

    model.add_callback('on_fit_epoch_end', progress)
    if args.resume:
        options = dict(resume=True, device=args.device, batch=args.batch)
    else:
        if run_dir.exists():
            raise FileExistsError(f'训练目录已存在：{run_dir}')
        options = dict(data=str(data), epochs=args.epochs, patience=0, imgsz=args.imgsz,
            batch=args.batch, device=args.device, workers=args.workers, project=str(PROJECT / 'runs'),
            name=args.name, exist_ok=False, seed=42, deterministic=True, val=True, plots=True,
            cache=False, amp=True, save=True, save_period=25, optimizer='AdamW', lr0=args.lr0,
            lrf=.01, cos_lr=True, warmup_epochs=3, warmup_bias_lr=0., weight_decay=.0005, close_mosaic=10,
            hsv_h=.005, hsv_s=.35, hsv_v=.35, degrees=15, translate=.1, scale=.4,
            fliplr=.5, flipud=0., mosaic=1.)
    write_json(job / 'training_config.json', dict(model=str(args.model), options=options,
        initialization=initialization,
        gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        ultralytics_version=ultralytics.__version__, started_at=datetime.now().astimezone().isoformat()))
    model.train(**options)
    trainer = model.trainer
    run_dir = Path(trainer.save_dir)
    preview_dir = run_dir / 'predictions_val'
    preview_dir.mkdir(exist_ok=True)
    best = run_dir / 'weights' / 'best.pt'
    predictor = YOLO(str(best))
    background_report = evaluate_backgrounds(predictor, data, args, run_dir / 'background_validation.json')
    images = sorted((data.parent / 'images' / 'val').glob('*.png'))
    predictions = []
    for index in np.linspace(0, len(images) - 1, min(12, len(images)), dtype=int):
        source = images[index]
        result = predictor.predict(str(source), imgsz=args.imgsz, device=args.device,
                                   conf=.25, retina_masks=True, verbose=False)[0]
        cv2.imwrite(str(preview_dir / (source.stem + '.jpg')), result.plot())
        mask = np.zeros(result.orig_shape, dtype=np.uint8)
        if result.masks is not None:
            mask = (result.masks.data.any(0).cpu().numpy().astype(np.uint8) * 255)
        cv2.imwrite(str(preview_dir / (source.stem + '_mask.png')), mask)
        predictions.append(dict(image=source.name, confidences=result.boxes.conf.tolist()))
    write_json(preview_dir / 'prediction_report.json', predictions)
    status = json.loads((job / 'status.json').read_text())
    completed = trainer.epoch + 1
    status.update(state='complete' if completed == trainer.epochs else 'stopped_before_target',
                  completed_epochs=completed, target_epochs=trainer.epochs,
                  finished_at=datetime.now().astimezone().isoformat(),
                  elapsed_seconds=round(time.monotonic() - started, 1),
                  estimated_remaining_seconds=0, prediction_dir=str(preview_dir))
    if background_report:
        status['background_validation'] = background_report
        baseline_path = job / 'background_baseline.json'
        if baseline_path.exists():
            baseline_report = json.loads(baseline_path.read_text())
            status['background_baseline'] = {key: value for key, value in baseline_report.items()
                                             if key != 'records'}
    write_json(job / 'status.json', status)
    write_json(run_dir / 'training_summary.json', status)
    print('TRAINING_FINISHED ' + json.dumps(status, ensure_ascii=False), flush=True)
    if completed != trainer.epochs:
        raise RuntimeError(f'训练只完成 {completed}/{trainer.epochs} 轮')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True, help='已在 Labelme 中人工复核的快照')
    parser.add_argument('--job', type=Path, required=True, help='日志及本次训练数据快照目录')
    parser.add_argument('--name', required=True, help='runs 下新的训练名称')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--model', type=Path, default=PRETRAINED)
    parser.add_argument('--lr0', type=float, default=.001)
    parser.add_argument('--background-source', action='append', type=Path, default=[],
                        help='人工确认的背景采集目录，可重复；按完整批次或带隔离的连续片段划分')
    parser.add_argument('--background-val-fraction', type=float, default=.2)
    parser.add_argument('--background-gap', type=int, default=10)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='0')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if (min(args.epochs, args.imgsz, args.batch) < 1 or args.workers < 0
            or Path(args.name).name != args.name or args.lr0 <= 0 or args.background_gap < 0
            or not 0 < args.background_val_fraction < .5):
        parser.error('训练参数或运行名称无效')
    job = args.job.resolve()
    job.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YOLO_CONFIG_DIR', str(job / 'ultralytics_config'))
    os.environ.setdefault('MPLCONFIGDIR', str(job / 'matplotlib'))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    try:
        dataset = job / 'dataset'
        if not args.resume:
            prepare(args.snapshot.resolve(), dataset, args.background_source,
                    args.background_val_fraction, args.background_gap)
        if args.prepare_only:
            return
        write_json(job / 'status.json', dict(state='starting', target_epochs=args.epochs,
                                            completed_epochs=0, dataset=str(dataset)))
        train(args, job, dataset / 'data.yaml')
    except BaseException as error:
        status_file = job / 'status.json'
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        status.update(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                      error=str(error), traceback=traceback.format_exc(),
                      updated_at=datetime.now().astimezone().isoformat())
        write_json(status_file, status)
        raise


if __name__ == '__main__':
    main()
