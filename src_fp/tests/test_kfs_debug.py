import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from kfs_debug import show_segmentation


class DebugTests(unittest.TestCase):
    def test_route_windows_and_depth_changes(self):
        rgb = np.zeros((80, 100, 3), np.uint8)
        raw = np.zeros((80, 100), bool)
        raw[30:50, 30:50] = True
        mask = raw.copy()
        mask[35, 29] = True
        mask[35, 30] = False
        candidate = SimpleNamespace(coarse=raw, box=np.array([30, 30, 49, 49]),
            points=np.array([[40., 40.]]), labels=np.array([1]))
        for mode, count in [('sam', 3), ('yolo', 2)]:
            result = SimpleNamespace(raw=raw, mask=mask, candidate=candidate,
                                     diagnostics={'mask_mode': mode})
            with patch('kfs_debug.cv2.namedWindow'), patch('kfs_debug.cv2.waitKey'), \
                    patch('kfs_debug.cv2.imshow') as show:
                show_segmentation(result, rgb, '123', True)
                self.assertEqual(show.call_count, count)
                names = [call.args[0] for call in show.call_args_list]
                self.assertEqual(any('OpenCV' in name for name in names), mode == 'sam')
                depth = show.call_args_list[-1].args[1]
                np.testing.assert_array_equal(depth[35, 29], [255, 100, 0])
                np.testing.assert_array_equal(depth[35, 30], [60, 60, 255])
        self.assertFalse(rgb.any())
