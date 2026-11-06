import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import cv2
import numpy as np

from fp_training_capture import TrainingCapture


class TrainingCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mesh_path = self.root / 'mesh.obj'
        self.mesh_path.write_text('# synthetic mesh\n')
        self.mesh = SimpleNamespace(
            vertices=np.array([[-.1, -.1, 0], [.1, -.1, 0], [.1, .1, 0], [-.1, .1, 0]]),
            faces=np.array([[0, 1, 2], [0, 2, 3]]))
        self.rgb = np.zeros((60, 80, 3), np.uint8)
        self.rgb[..., 0] = np.arange(80)
        self.rgb[..., 1] = 123
        self.rgb[..., 2] = 200
        self.depth = np.full((60, 80), .8, np.float32)
        self.k = np.array([[120., 0, 40], [0, 120., 30], [0, 0, 1]])
        self.pose = np.eye(4)
        self.pose[2, 3] = .8
        self.metadata = dict(sequence=42, stamp=123.5, frame_id='color_optical')
        self.capture = TrainingCapture(self.root / 'dataset', self.mesh, self.mesh_path)

    def save(self, key=ord('s'), metadata=None):
        self.capture.handle(key, self.rgb, self.depth, self.k, self.pose,
                            metadata or self.metadata, 7)

    def test_original_pixels_pose_intrinsics_and_projection_share_one_frame(self):
        self.save()
        folder = self.capture.output
        image = cv2.cvtColor(cv2.imread(str(folder / 'images/000000.png')), cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(image, self.rgb)
        np.testing.assert_array_equal(np.load(folder / 'depth/000000.npy'), self.depth)
        np.testing.assert_allclose(np.loadtxt(folder / 'ob_in_cam/000000.txt'), self.pose)
        np.testing.assert_allclose(np.loadtxt(folder / 'intrinsics/000000.txt'), self.k)
        mask = cv2.imread(str(folder / 'masks/000000.png'), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(mask[30, 40], 255)
        self.assertEqual(mask[0, 0], 0)
        annotation = json.loads((folder / 'labelme/000000.json').read_text())
        self.assertFalse(annotation['flags']['reviewed'])
        self.assertEqual(annotation['shapes'][0]['label'], 'tkk')
        values = (folder / 'labels/000000.txt').read_text().split()
        self.assertEqual(values[0], '0')
        xy = np.asarray(values[1:], dtype=float).reshape(-1, 2) * [80, 60]
        np.testing.assert_allclose(xy, annotation['shapes'][0]['points'], atol=1e-5)
        record = json.loads((folder / 'manifest.jsonl').read_text())
        self.assertEqual((record['source_sequence'], record['processed_index']), (42, 7))

    def test_capture_is_opt_in_and_r_pauses_auto_sampling(self):
        self.save(-1)
        self.assertEqual(self.capture.count, 0)
        self.save(ord('c'))
        self.assertTrue(self.capture.automatic)
        self.assertEqual(self.capture.count, 1)
        self.save(ord('r'))
        self.assertFalse(self.capture.automatic)
        self.assertEqual(self.capture.count, 1)

    def test_reject_cropped_data_and_existing_capture_directory(self):
        with self.assertRaisesRegex(ValueError, 'full-frame'):
            self.save(metadata=dict(self.metadata, roi_xyxy=[10, 10, 40, 40]))
        with self.assertRaisesRegex(ValueError, 'empty'):
            TrainingCapture(self.capture.output, self.mesh, self.mesh_path)

    def test_explicit_auto_mode_saves_without_a_keypress(self):
        self.capture = TrainingCapture(self.root / 'auto_dataset', self.mesh,
                                       self.mesh_path, automatic=True)
        self.save(-1)
        self.assertEqual(self.capture.count, 1)
        self.save(-1)
        self.assertEqual(self.capture.count, 1)  # The interval prevents duplicate bursts.


if __name__ == '__main__':
    unittest.main()
