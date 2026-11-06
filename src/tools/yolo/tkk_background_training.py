"""Validate and split explicitly confirmed TKK background capture batches."""
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
import shutil


def read_background_batches(paths):
    batches = []
    for path in paths:
        path = Path(path).resolve()
        config = json.loads((path / 'background_capture.json').read_text())
        if (config.get('target_class') != 'tkk' or config.get('class_id') != 0
                or config.get('mode') != 'manual_confirmed_negative'):
            raise ValueError(f'不是人工确认的 TKK 背景采集：{path}')
        rows = [json.loads(line) for line in (path / 'manifest.jsonl').read_text().splitlines()
                if line.strip()]
        samples = {row['sample'] for row in rows}
        if not rows or len(samples) != len(rows):
            raise ValueError(f'背景清单为空或样本重复：{path}')
        for directory, suffix in [('images', '*.png'), ('labels', '*.txt'), ('labelme', '*.json')]:
            if samples != {p.stem for p in (path / directory).glob(suffix)}:
                raise ValueError(f'背景 {directory} 与 manifest 不一一对应：{path}')
        for row in rows:
            sample = row['sample']
            if (Path(sample).name != sample or row['id'] != path.name + '_' + sample
                    or row.get('group') != path.name or row.get('split') not in ('train', 'val')
                    or row.get('confirmed_negative') is not True or row.get('reviewed') is not True
                    or row.get('confirmation') != 'manual_SPACE_no_TKK'):
                raise ValueError(f'未明确确认的背景样本：{row.get("id")}')
            for key, expected in [('image', f'images/{sample}.png'),
                                  ('label', f'labels/{sample}.txt'),
                                  ('annotation', f'labelme/{sample}.json')]:
                if row.get(key) != expected:
                    raise ValueError(f'背景来源路径不符合采集格式：{row["id"]}')
        if len({row['split'] for row in rows}) != 1:
            raise ValueError(f'同一个原始背景批次混有 train 和 val：{path}')
        rows.sort(key=lambda row: (row['stamp']['sec'], row['stamp']['nanosec'], row['sample']))
        batches.append(dict(path=path, name=path.name, config=config, rows=rows))
    if len({batch['name'] for batch in batches}) != len(batches):
        raise ValueError('背景批次名称重复')
    return batches


def split_background_batches(batches, val_fraction=.2, gap=10):
    """Whole sessions when available; one session uses a named temporal holdout."""
    if not batches:
        return {}, dict(method='no_backgrounds', batches=[], excluded_gap_ids=[])
    if not 0 < val_fraction < .5 or gap < 0:
        raise ValueError('背景验证比例或隔离数量无效')
    assignments, excluded = {}, []
    explicit_val = {batch['name'] for batch in batches if batch['rows'][0]['split'] == 'val'}
    if explicit_val:
        if len(explicit_val) == len(batches):
            raise ValueError('背景批次全部指定为 val，没有用于抑制误检的训练背景')
        val_names = explicit_val
        method = 'explicit_whole_background_capture_sessions'
    elif len(batches) > 1:
        names = sorted(batch['name'] for batch in batches)
        if len(names) > 16:
            raise ValueError('背景批次超过 16 个，请明确指定独立的 val 采集批次')
        sizes = {batch['name']: len(batch['rows']) for batch in batches}
        total = sum(sizes.values())
        candidates = (subset for count in range(1, len(names)) for subset in combinations(names, count))
        chosen = min(candidates, key=lambda subset: (
            abs(sum(sizes[name] for name in subset) / total - val_fraction), len(subset), subset))
        val_names = set(chosen)
        method = 'whole_background_capture_sessions'
    else:
        batch = batches[0]
        count = len(batch['rows'])
        nval = max(1, math.ceil(count * val_fraction))
        lo = (count - nval) // 2
        hi = lo + nval
        if lo <= gap or count - hi <= gap:
            raise ValueError('背景照片不足以保留中间验证片段及两侧隔离带')
        for index, row in enumerate(batch['rows']):
            split = 'val' if lo <= index < hi else 'train'
            if lo - gap <= index < lo or hi <= index < hi + gap:
                split = 'gap'
                excluded.append(row['id'])
            assignments[row['id']] = split
        val_names = None
        method = 'same_capture_middle_contiguous_validation_with_two_sided_gap'
    if val_names is not None:
        for batch in batches:
            for row in batch['rows']:
                assignments[row['id']] = 'val' if batch['name'] in val_names else 'train'
    report = dict(method=method, val_fraction=val_fraction, gap=gap,
                  independent_background_capture_validation=val_names is not None,
                  excluded_gap_ids=excluded, counts=dict(Counter(assignments.values())),
                  batches=[dict(source=str(batch['path']), group=batch['name'],
                                images=len(batch['rows']), original_split=batch['rows'][0]['split'])
                           for batch in batches])
    return assignments, report


def append_backgrounds(output, accepted, excluded, seen, paths, digest, val_fraction=.2, gap=10):
    import cv2
    from ultralytics.data.utils import verify_image_label

    batches = read_background_batches(paths)
    assignments, report = split_background_batches(batches, val_fraction, gap)
    known_ids = {row['id'] for row in accepted} | {row['id'] for row in excluded}
    for batch in batches:
        path = batch['path']
        for source in batch['rows']:
            key = source['id']
            if key in known_ids:
                raise ValueError(f'背景样本 ID 冲突：{key}')
            known_ids.add(key)
            image = path / source['image']
            annotation_file = path / source['annotation']
            annotation_hash = digest(annotation_file)
            annotation = json.loads(annotation_file.read_text())
            if (annotation.get('shapes') != [] or annotation.get('flags', {}).get('confirmed_negative') is not True
                    or annotation.get('flags', {}).get('reviewed') is not True
                    or (path / source['label']).read_text().strip()):
                raise ValueError(f'背景标签与人工确认状态矛盾：{key}')
            image_hash = digest(image)
            if image_hash != source['image_sha256']:
                raise ValueError(f'背景图像自采集后发生变化：{key}')
            pixels = cv2.imread(str(image))
            if pixels is None:
                raise ValueError(f'背景无法解码：{key}')
            height, width = pixels.shape[:2]
            if (annotation.get('imageWidth'), annotation.get('imageHeight')) != (width, height):
                raise ValueError(f'背景标注尺寸不符：{key}')
            shutil.copy2(annotation_file, output / 'source_annotations' / (key + '.json'))
            split = assignments[key]
            if split == 'gap':
                excluded.append(dict(id=key, reason='background_temporal_gap',
                                     source_image=str(image), source_annotation=str(annotation_file),
                                     sha256=annotation_hash, source_group=batch['name']))
                continue
            if image_hash in seen and seen[image_hash] != split:
                raise ValueError(f'背景与其他图片完全相同且跨集合：{key}')
            seen[image_hash] = split
            target = output / 'images' / split / (key + '.png')
            label = output / 'labels' / split / (key + '.txt')
            target.symlink_to(image)
            label.write_text('')
            checked = verify_image_label((str(target), str(label), '', False, 1, 0, 0, False))
            if checked[0] is None or len(checked[3]) or checked[-1]:
                raise ValueError(f'官方背景标签检查失败：{key} {checked[-1]}')
            if digest(annotation_file) != annotation_hash:
                raise ValueError(f'背景标注仍在修改：{key}')
            accepted.append(dict(source, split=split, capture_split=source['split'],
                source_group=batch['name'], source_image=str(image), source_image_sha256=image_hash,
                image=str(target.relative_to(output)), label=str(label.relative_to(output)), instances=0,
                reviewed_annotation=str(annotation_file), reviewed_annotation_sha256=annotation_hash,
                source_reviewed_flag=True, training_approved=True,
                approval='User confirmed target-free backgrounds at capture and requested training inclusion',
                background_split_method=report['method']))
    report['included_counts'] = dict(Counter(row['split'] for row in accepted if row.get('confirmed_negative')))
    return report


def evaluate_backgrounds(model, data, args, report_file):
    """Measure false-positive frames at the same confidence used in live comparison."""
    rows = [row for row in json.loads((data.parent / 'manifest.json').read_text())
            if row['split'] == 'val' and row.get('confirmed_negative') is True]
    if not rows:
        return None
    records = []
    for start in range(0, len(rows), args.batch):
        batch = rows[start:start + args.batch]
        results = model.predict([str(data.parent / row['image']) for row in batch],
            imgsz=args.imgsz, device=args.device, conf=.5, iou=.7, classes=[0], verbose=False)
        for row, result in zip(batch, results):
            confidences = result.boxes.conf.cpu().tolist()
            records.append(dict(id=row['id'], detections=len(confidences), confidences=confidences))
    false_positives = sum(record['detections'] > 0 for record in records)
    split_report = json.loads((data.parent / 'export_report.json').read_text()).get('background', {})
    independent = split_report.get('independent_background_capture_validation', False)
    report = dict(confidence_threshold=.5, nms_iou=.7, images=len(records),
                  false_positive_images=false_positives,
                  false_positive_image_rate=false_positives / len(records),
                  false_positive_instances=sum(record['detections'] for record in records), records=records,
                  independent_background_capture_validation=independent,
                  note=('Whole background capture sessions held out; capture conditions can still be similar'
                        if independent else
                        'Same-capture held-out backgrounds are correlated; this is not an independent scene test'))
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return {key: value for key, value in report.items() if key != 'records'}
