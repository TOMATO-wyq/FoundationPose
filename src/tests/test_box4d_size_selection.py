"""Size selection restricts detection/tracking without stripping orientation."""

from pathlib import Path
import unittest

import numpy as np

from box4d.config import Config
from box4d.pose import Pipeline
from box4d.synthetic import Renderer, default_camera, world_box
from box4d.tracking import Tracker


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


class SizeSelectionTests(unittest.TestCase):
    def test_small_keeps_orientation_and_yolo_configuration(self):
        cfg = Config.load(CONFIG_DIR / "box4d_d435i_yolo_live.json", size="small")
        self.assertEqual(cfg.sizes_m, (0.20,))
        self.assertEqual(cfg.two_color_sizes_m, (0.20,))
        self.assertEqual(cfg.track_face_switch_sizes_m, (0.20,))
        self.assertTrue(cfg.tracking and cfg.analyze_colors and cfg.temporal_recovery)
        self.assertTrue(Path(cfg.yolo_weights).is_file())
        self.assertEqual(cfg.yolo_sizes_m, (0.20,))

    def test_big_does_not_load_small_model(self):
        cfg = Config.load(CONFIG_DIR / "box4d_d435i_yolo_live.json", size="big")
        self.assertEqual(cfg.sizes_m, (0.35,))
        self.assertIsNone(cfg.yolo_weights)
        self.assertEqual(cfg.two_color_sizes_m, ())
        self.assertEqual(cfg.track_face_switch_sizes_m, ())
        self.assertTrue(cfg.tracking and cfg.analyze_colors)

    def test_invalid_or_missing_size_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "size must"):
            Config.load(size="medium")
        with self.assertRaisesRegex(ValueError, "requires"):
            Config.load(size="big", sizes_m=(0.20,))
        self.assertEqual(Config.load().sizes_m, (0.20, 0.35))

    def test_mixed_scene_only_detects_and_tracks_selected_size_with_full_pose(self):
        renderer = Renderer(default_camera())
        boxes = [world_box((-0.30, 0, 0.10), 0.20, 15),
                 world_box((0.30, 0, 0.175), 0.35, -20)]
        for size, edge in (("small", 0.20), ("big", 0.35)):
            cfg = Config.load(size=size, up_camera=(0, -1, 0), plane_grid_step=8)
            pipeline, tracker = Pipeline(cfg), Tracker(cfg)
            ids = []
            for sequence in range(2):
                frame, _, _ = renderer.render(boxes, sequence, depth_noise_m=0.001)
                result, _ = pipeline.process(frame)
                tracker.update(result)
                self.assertTrue(result["objects"], size)
                self.assertEqual({obj["size_m"] for obj in result["objects"]}, {edge})
                obj = result["objects"][0]
                pose = np.asarray(obj["T_camera_from_box"])
                self.assertEqual(pose.shape, (4, 4))
                np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
                ids.append(obj["track_id"])
            self.assertEqual(ids[0], ids[1])
