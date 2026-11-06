#!/usr/bin/env python3
"""Collect TKK captures into a new YOLO segmentation snapshot without changing sources."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import shlex
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw


PROJECT = Path(__file__).resolve().parents[3]
ANNOTATION_ENV = Path('/home/tomato/code/box_seg_annotation/env/bin')
PRETRAINED = Path('/home/tomato/code/box_seg_annotation/models/yolo11n-seg.pt')


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read_sources(paths):
    sources = []
    for path in paths:
        path = path.resolve()
        config_file = path / 'capture_config.json'
        config = json.loads(config_file.read_text(encoding='utf-8'))
        if config.get('class_name') != 'tkk' or config.get('class_id') != 0:
            raise ValueError(f'不是类别 0 tkk 的采集目录：{path}')
        images = sorted((path / 'images').glob('*.png'))
        if not images:
            raise ValueError(f'没有采集图片：{path}')
        image_ids = {image.stem for image in images}
        annotation_ids = {ann.stem for ann in (path / 'labelme').glob('*.json')}
        if image_ids != annotation_ids:
            raise ValueError(f'图片与 Labelme JSON 不一一对应，可能仍在采集中：{path}')
        manifest = [json.loads(line) for line in (path / 'manifest.jsonl').read_text().splitlines()
                    if line.strip()]
        metadata = {row['sample']: row for row in manifest}
        if len(metadata) != len(manifest) or set(metadata) != image_ids:
            raise ValueError(f'manifest 与图片不一一对应：{path}')
        name = path.parent.name
        match = re.search(r'20\d{6}', name)
        sources.append(dict(path=path, name=name, date=match.group() if match else name,
                            config=config, config_sha256=sha256(config_file),
                            images=images, metadata=metadata))
    if not sources:
        raise ValueError('未找到 TKK 采集目录')
    if len({s['name'] for s in sources}) != len(sources):
        raise ValueError('采集批次名称重复，无法安全生成唯一文件名')
    if len({s['config']['mesh_sha256'] for s in sources}) != 1:
        raise ValueError('采集使用了不同的三维模型，不能自动合并为同一个 tkk 数据集')
    return sources


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validation_sources(sources, requested, fraction):
    names = {s['name'] for s in sources}
    if requested:
        chosen = set(requested)
        if chosen - names or chosen == names:
            raise ValueError('--val-source 必须指定已发现的部分批次，且保留非空训练集')
        return chosen, 'explicit_capture_sessions'
    groups = defaultdict(list)
    for source in sources:
        groups[source['date']].append(source)
    method = 'whole_capture_dates'
    if len(groups) < 2:
        groups = {s['name']: [s] for s in sources}
        method = 'whole_capture_sessions_same_date'
    if len(groups) < 2:
        raise ValueError('至少需要两个采集批次才能整批留出验证集；请再拍一批')
    # Keep whole dates together, including adjacent short initialization captures.
    keys = sorted(groups)
    if len(keys) > 16:
        raise ValueError('采集日期超过 16 组，请用 --val-source 明确指定验证批次')
    total = sum(len(s['images']) for s in sources)
    sizes = {key: sum(len(s['images']) for s in groups[key]) for key in keys}
    candidates = (subset for n in range(1, len(keys)) for subset in combinations(keys, n))
    best = min(candidates, key=lambda subset: (
        abs(sum(sizes[key] for key in subset) / total - fraction), len(subset), subset))
    return {s['name'] for key in best for s in groups[key]}, method


def annotation_polygons(annotation, width, height, source):
    if (annotation.get('imageWidth'), annotation.get('imageHeight')) != (width, height):
        raise ValueError(f'Labelme 图片尺寸不符：{source}')
    polygons = []
    # These captures contain one polygon per physical object. Do not split a
    # disconnected group_id into extra instances during conversion.
    instances = set()
    for i, shape in enumerate(annotation['shapes']):
        if shape.get('label') != 'tkk' or shape.get('shape_type') != 'polygon':
            raise ValueError(f'出现其他类别或非多边形标注：{source}')
        instance = shape.get('group_id')
        if instance is not None and instance in instances:
            raise ValueError(f'同一实例含多个轮廓，需要人工处理：{source}')
        instances.add(instance if instance is not None else ('shape', i))
        points = np.asarray(shape['points'], dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all() or len(np.unique(points, axis=0)) < 3
                or (points < 0).any() or (points > [width - 1, height - 1]).any()
                or abs(cv2.contourArea(points.astype(np.float32))) < 1):
            raise ValueError(f'无效或越界多边形：{source}')
        polygons.append(points)
    if not polygons:
        raise ValueError(f'空标注不能自动当作负样本：{source}')
    return polygons


def contact_sheets(output, records):
    directory = output / 'contact_sheets'
    directory.mkdir()
    for page, start in enumerate(range(0, len(records), 24), 1):
        sheet = Image.new('RGB', (1280, 6 * 264), '#20252b')
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(records[start:start + 24]):
            with Image.open(output / row['overlay']) as original:
                thumbnail = original.convert('RGB')
            thumbnail.thumbnail((320, 240))
            x, y = index % 4 * 320, index // 4 * 264
            sheet.paste(thumbnail, (x, y))
            draw.text((x + 3, y + 241), row['id'], fill='white')
        sheet.save(directory / f'{page:03d}.jpg', quality=90)
    selected = records if len(records) <= 12 else [records[i] for i in
                np.linspace(0, len(records) - 1, 12, dtype=int)]
    sheet = Image.new('RGB', (1280, 3 * 264), '#20252b')
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(selected):
        with Image.open(output / row['overlay']) as original:
            thumbnail = original.convert('RGB')
        thumbnail.thumbnail((320, 240))
        x, y = index % 4 * 320, index // 4 * 264
        sheet.paste(thumbnail, (x, y))
        draw.text((x + 3, y + 241), row['id'], fill='white')
    sheet.save(output / 'contact_sheet.jpg', quality=90)


def export(sources, val_names, method, output, threshold):
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset, verify_image_label
    import ultralytics

    output.mkdir(parents=True, exist_ok=False)
    for split in ('train', 'val'):
        for kind in ('images', 'labels', 'masks', 'overlays'):
            (output / kind / split).mkdir(parents=True)
    (output / 'source_annotations').mkdir()
    (output / 'labelme').mkdir()
    records, hashes = [], {}
    for source in sources:
        split = 'val' if source['name'] in val_names else 'train'
        for image in source['images']:
            key = source['name'] + '_' + image.stem
            annotation_file = source['path'] / 'labelme' / (image.stem + '.json')
            annotation_hash = sha256(annotation_file)
            annotation = json.loads(annotation_file.read_text(encoding='utf-8'))
            pixels = cv2.imread(str(image))
            if pixels is None:
                raise ValueError(f'图片无法解码：{image}')
            height, width = pixels.shape[:2]
            polygons = annotation_polygons(annotation, width, height, annotation_file)
            image_hash = sha256(image)
            if image_hash in hashes and hashes[image_hash] != split:
                raise ValueError(f'训练集与验证集存在完全相同的图片：{image}')
            hashes[image_hash] = split
            target = output / 'images' / split / (key + '.png')
            label = output / 'labels' / split / (key + '.txt')
            shutil.copy2(image, target)
            lines = ['0 ' + ' '.join(f'{value:.8f}' for value in
                     (points / [width, height]).ravel()) for points in polygons]
            label.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            check = verify_image_label((str(target), str(label), '', False, 1, 0, 0, False))
            if check[0] is None or len(check[3]) != len(polygons) or check[-1]:
                raise ValueError(f'官方 loader 检查失败：{key} {check[-1]}')
            if sha256(target) != image_hash or sha256(annotation_file) != annotation_hash:
                raise ValueError(f'转换过程中源数据发生变化：{key}')
            shutil.copy2(annotation_file, output / 'source_annotations' / (key + '.json'))
            annotation['imagePath'] = f'../images/{split}/{key}.png'
            annotation['imageData'] = None
            annotation['conversion_source'] = str(annotation_file)
            write_json(output / 'labelme' / (key + '.json'), annotation)
            mask = np.zeros((height, width), dtype=np.uint8)
            contours = [np.rint(points).astype(np.int32) for points in polygons]
            cv2.fillPoly(mask, contours, 255)
            hue, saturation, value = cv2.split(cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV))
            red_blue = (((hue <= 15) | (hue >= 165) | ((hue >= 90) & (hue <= 150)))
                        & (saturation >= 70) & (value >= 30))
            coverage = float(red_blue[mask > 0].mean())
            reasons = ['low_red_blue_coverage_possible_drift_or_occlusion'] if coverage < threshold else []
            overlay = pixels.copy()
            overlay[mask > 0] = (overlay[mask > 0] * .75 + np.array([0, 64, 0])).astype(np.uint8)
            color = (0, 0, 255) if reasons else (0, 255, 0)
            cv2.polylines(overlay, contours, True, color, 2)
            cv2.putText(overlay, f'{split} {image.stem} FP - REVIEW color={coverage:.2f}',
                        (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
            for path, data in [(output / 'masks' / split / (key + '.png'), mask),
                               (output / 'overlays' / split / (key + '.jpg'), overlay)]:
                if not cv2.imwrite(str(path), data):
                    raise OSError(f'无法写入：{path}')
            records.append(dict(
                id=key, split=split, source_group=source['name'], source_image=str(image),
                source_annotation=str(annotation_file), source_image_sha256=image_hash,
                source_annotation_sha256=annotation_hash,
                image=str(target.relative_to(output)), label=str(label.relative_to(output)),
                overlay=f'overlays/{split}/{key}.jpg', width=width, height=height,
                instances=len(polygons), reviewed=annotation.get('flags', {}).get('reviewed') is True,
                annotation_origin=source['config']['annotation_origin'],
                red_blue_coverage=coverage, review_reasons=reasons,
                source_metadata=source['metadata'][image.stem]))
    counts = dict(Counter(row['split'] for row in records))
    yaml = (f'path: {json.dumps(str(output), ensure_ascii=False)}\n'
            'train: images/train\nval: images/val\nnc: 1\nnames:\n  0: tkk\n')
    (output / 'data.yaml').write_text(yaml, encoding='utf-8')
    (output / 'classes.txt').write_text('tkk\n', encoding='utf-8')
    data = check_det_dataset(str(output / 'data.yaml'), autodownload=False)
    loader_counts = {}
    for split in ('train', 'val'):
        dataset = YOLODataset(img_path=data[split], data=data, task='segment',
                              imgsz=640, augment=False, cache=False, batch_size=2)
        if len(dataset) != counts[split] or any(len(row['segments']) != len(row['cls'])
                                              for row in dataset.labels):
            raise ValueError(f'{split} 官方分割数据集扫描结果与导出数量不一致')
        item = dataset[0]
        if item['masks'].numel() == 0 or not item['masks'].any():
            raise ValueError(f'{split} 官方分割变换未生成 mask')
        loader_counts[split] = dict(images=len(dataset),
                                    instances=sum(len(row['segments']) for row in dataset.labels),
                                    sample_mask_shape=list(item['masks'].shape))
    write_json(output / 'manifest.json', records)
    write_json(output / 'sources.json', [dict(
        dataset=str(s['path']), group=s['name'], capture_date=s['date'],
        images=len(s['images']), split='val' if s['name'] in val_names else 'train',
        capture_config=s['config'], config_sha256=s['config_sha256']) for s in sources])
    with (output / 'review_flags.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'split', 'red_blue_coverage', 'reason', 'overlay'])
        writer.writeheader()
        for row in records:
            if row['review_reasons']:
                writer.writerow(dict(id=row['id'], split=row['split'],
                    red_blue_coverage=f"{row['red_blue_coverage']:.6f}",
                    reason=';'.join(row['review_reasons']), overlay=row['overlay']))
    report = dict(total_images=len(records), instances=sum(r['instances'] for r in records),
                  counts=counts, classes={'0': 'tkk'}, split_method=method,
                  val_sources=sorted(val_names), skipped_images=0,
                  reviewed=sum(r['reviewed'] for r in records),
                  flagged_for_review=sum(bool(r['review_reasons']) for r in records),
                  color_coverage_threshold=threshold, ultralytics_version=ultralytics.__version__,
                  official_loader=loader_counts, training_started=False,
                  warnings=['FP projection pseudo labels require review; some tracking has visibly drifted.',
                            'Color coverage only flags suspicious labels and cannot certify alignment.',
                            'All samples, including suspicious labels, are retained in data.yaml.',
                            'Different capture dates still share the same room and background.'])
    write_json(output / 'export_report.json', report)
    contact_sheets(output, records)
    train_command = shlex.join([str(ANNOTATION_ENV / 'yolo'), 'segment', 'train',
        f'model={PRETRAINED}', f'data={output / "data.yaml"}', 'epochs=150', 'patience=0',
        'imgsz=640', 'batch=2', 'device=0', f'project={PROJECT / "runs"}', f'name={output.name}'])
    (output / 'train_after_review.sh').write_text(
        '#!/usr/bin/env bash\n# Run after correcting the Labelme labels and exporting a new snapshot.\n'
        'set -euo pipefail\n' + train_command + '\n', encoding='utf-8')
    (output / 'README_中文.md').write_text(
        '# TKK YOLO 实例分割数据快照\n\n'
        f'共 {len(records)} 张，训练集 {counts["train"]} 张，验证集 {counts["val"]} 张。'
        '整个红蓝方块为一个实例，类别 ID 0，名称 tkk。原图、源 JSON、源 manifest 不修改。\n\n'
        '根据当前 Labelme JSON 重新生成归一化多边形，不使用可能过期的源 labels。'
        '按完整拍摄日期（或显式批次）留出验证集，完全相同图片不得跨集合。'
        '仍在同一房间拍摄，验证指标不代表新场景表现。\n\n'
        '## 训练前复核\n\n'
        '这些标签是 FP 完整模型投影，不会扣除手和其他遮挡。已观察到部分跟踪漂移，'
        '全部样本均保留，data.yaml 也包含疑似错标，不能将格式检查通过视为标注正确。'
        f'颜色覆盖率低于 {threshold:.0%} 的 {report["flagged_for_review"]} 张列在 review_flags.csv；'
        '该检查可能误报或漏报，其余图片也应检查。\n\n'
        'contact_sheet.jpg 为抽样预览，contact_sheets/ 为全部图片，overlays/ 为逐张叠加，'
        'masks/ 是从本次多边形重建的二值 mask。manifest.json 保存来源及 SHA256。'
        'source_annotations/ 是原始 JSON 备份；labelme/ 是修正了 imagePath 的可编辑副本。\n\n'
        '建议在原采集 dataset/labelme/ 中修正标注后重新运行转换，以便后续补拍一起导出。'
        '若修改本快照 labelme/，需将对应 points 同步回 manifest.json 所指的源 JSON 再重新导出；'
        '直接修改 JSON 不会自动更新本快照 labels/。\n\n'
        '## 补拍与重新转换\n\n```bash\n'
        f'cd {PROJECT}\nbash src/scripts/yolo/start_tkk_capture.sh\n'
        f'PYTHONPATH=src {ANNOTATION_ENV / "python"} -m tools.yolo.prepare_tkk_yolo_seg\n```\n\n'
        '每次转换创建新快照，并扫描所有 TKK 采集批次。拍摄过程中建议暂停采样后再转换，'
        '避免读取尚未写完整的样本。\n\n'
        '## 训练\n\n'
        '完成标注修正并重新转换后，用新快照的 data.yaml 训练；没有自动启动训练。\n\n'
        f'```bash\n{train_command}\n```\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', help='采集的 dataset 目录，可重复')
    parser.add_argument('--val-source', action='append', help='验证批次名称，可重复；默认按完整日期划分')
    parser.add_argument('--val-fraction', type=float, default=.2)
    parser.add_argument('--review-color-threshold', type=float, default=.3)
    parser.add_argument('--output', type=Path, help='必须为尚不存在的新目录')
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1 or not 0 <= args.review_color_threshold <= 1:
        parser.error('val-fraction 应在 0 和 1 之间，review-color-threshold 应在 0 到 1 之间')
    paths = args.source or [p.parent for p in sorted(
        (PROJECT / 'outputs').glob('*/dataset/capture_config.json'))
        if json.loads(p.read_text()).get('class_name') == 'tkk']
    sources = read_sources(paths)
    val_names, method = validation_sources(sources, args.val_source, args.val_fraction)
    output = (args.output or PROJECT / 'outputs' /
              ('tkk_yolo_seg_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    # Never write exports into any source, or choose an ancestor of a source.
    if any(output == s['path'] or output in s['path'].parents or s['path'] in output.parents
           for s in sources):
        parser.error('输出目录不能与源采集目录重叠')
    report = export(sources, val_names, method, output, args.review_color_threshold)
    print(json.dumps(dict(output=str(output), **report), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
