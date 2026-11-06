"""Backgrounds require explicit confirmation and must not leak between splits."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tools.yolo.tkk_background_training import append_backgrounds, read_background_batches, split_background_batches


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BackgroundTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def capture(self, name, split='train', value=40):
        path = self.root / name
        for directory in ('images', 'labels', 'labelme'):
            (path / directory).mkdir(parents=True)
        (path / 'background_capture.json').write_text(json.dumps(dict(
            target_class='tkk', class_id=0, mode='manual_confirmed_negative')))
        image = path / 'images/000000.png'
        cv2.imwrite(str(image), np.full((20, 30, 3), value, dtype=np.uint8))
        (path / 'labels/000000.txt').write_text('')
        (path / 'labelme/000000.json').write_text(json.dumps(dict(
            shapes=[], flags=dict(reviewed=True, confirmed_negative=True), imageWidth=30, imageHeight=20)))
        row = dict(id=name + '_000000', sample='000000', group=name, split=split,
                   confirmed_negative=True, reviewed=True, confirmation='manual_SPACE_no_TKK',
                   image='images/000000.png', label='labels/000000.txt',
                   annotation='labelme/000000.json', image_sha256=digest(image),
                   stamp=dict(sec=1, nanosec=0))
        (path / 'manifest.jsonl').write_text(json.dumps(row) + '\n')
        return path

    def output(self):
        path = self.root / 'output'
        for split in ('train', 'val'):
            for directory in ('images', 'labels'):
                (path / directory / split).mkdir(parents=True)
        (path / 'source_annotations').mkdir()
        return path

    def test_single_capture_temporal_holdout_has_two_sided_gap(self):
        rows = [dict(id=f'frame_{i:06d}', split='train') for i in range(338)]
        batch = dict(name='capture', path=Path('/capture'), rows=rows)
        assigned, report = split_background_batches([batch])
        self.assertEqual(Counter(assigned.values()), {'train': 250, 'val': 68, 'gap': 20})
        train = [i for i, row in enumerate(rows) if assigned[row['id']] == 'train']
        val = [i for i, row in enumerate(rows) if assigned[row['id']] == 'val']
        self.assertEqual(min(abs(a - b) for a in train for b in val), 11)
        self.assertFalse(report['independent_background_capture_validation'])
        self.assertEqual(len(report['excluded_gap_ids']), 20)

    def test_explicit_validation_batch_stays_whole(self):
        paths = [self.capture('train_capture'), self.capture('val_capture', 'val', 90)]
        batches = read_background_batches(paths)
        assigned, report = split_background_batches(batches)
        self.assertEqual(assigned, {'train_capture_000000': 'train', 'val_capture_000000': 'val'})
        self.assertTrue(report['independent_background_capture_validation'])

    def test_missing_confirmation_never_becomes_a_negative(self):
        path = self.capture('unconfirmed')
        row = json.loads((path / 'manifest.jsonl').read_text())
        row['confirmed_negative'] = False
        (path / 'manifest.jsonl').write_text(json.dumps(row) + '\n')
        with self.assertRaisesRegex(ValueError, '未明确确认'):
            read_background_batches([path])

    def test_saved_empty_negatives_pass_loader_and_preserve_source_annotations(self):
        paths = [self.capture('train_capture'), self.capture('val_capture', 'val', 90)]
        original = {path: digest(path / 'labelme/000000.json') for path in paths}
        accepted, excluded = [], []
        output = self.output()
        report = append_backgrounds(output, accepted, excluded, {}, paths, digest)
        self.assertEqual(report['included_counts'], {'train': 1, 'val': 1})
        self.assertEqual(excluded, [])
        self.assertEqual(len(accepted), 2)
        for row in accepted:
            self.assertTrue(row['confirmed_negative'])
            self.assertTrue((output / row['image']).is_file())
            self.assertEqual((output / row['label']).read_bytes(), b'')
        self.assertTrue(all(digest(path / 'labelme/000000.json') == old for path, old in original.items()))

    def test_cross_split_duplicate_images_are_rejected(self):
        paths = [self.capture('train_capture'), self.capture('val_capture', 'val')]
        with self.assertRaisesRegex(ValueError, '跨集合'):
            append_backgrounds(self.output(), [], [], {}, paths, digest)

    def test_polygon_inside_claimed_background_is_rejected(self):
        paths = [self.capture('train_capture'), self.capture('val_capture', 'val', 90)]
        path = paths[0] / 'labelme/000000.json'
        data = json.loads(path.read_text())
        data['shapes'] = [dict(label='tkk', shape_type='polygon', points=[[1, 1], [10, 1], [10, 10]])]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, '矛盾'):
            append_backgrounds(self.output(), [], [], {}, paths, digest)


if __name__ == '__main__':
    unittest.main()
