import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmark_kfs_resolutions import summarize


class BenchmarkTests(unittest.TestCase):
    def test_warmup_and_failed_pose_do_not_bridge(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory/'ob_in_cam').mkdir()
            with (directory/'loop_timing.csv').open('w') as handle:
                keys = ['processed_index', 'mode', 'loop_s', 'inference_s',
                        'display_s', 'rgbd_wait_s', 'pose_save_s']
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                for index in range(1, 36):
                    writer.writerow(dict(processed_index=index,
                        mode='register' if index == 1 else 'tracking', loop_s=.05,
                        inference_s=.02, display_s=0, rgbd_wait_s=.03, pose_save_s=0))
                    if index != 34:
                        pose = np.eye(4)
                        pose[2, 3] = 1
                        np.savetxt(directory/'ob_in_cam'/f'{index-1:06d}.txt', pose)
            (directory/'runtime_summary.json').write_text(json.dumps({
                'register_attempts': 1, 'torch_peak_reserved_bytes': 100}))
            result = summarize(directory)
            self.assertEqual(result['measured_tracking_frames'], 4)
            self.assertAlmostEqual(result['steady_loop_fps'], 20)
            self.assertEqual(result['static_jitter']['adjacent_pairs'], 1)
