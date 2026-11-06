"""Exercise offline/live entrypoint wiring with a fake FP and CPU SAM backend."""
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import run_foundationpose_zed as offline
import run_foundationpose_live as live
from kfs_pose_controller import render_mesh_depth
from kfs_segmentation import KFSSegmenter, SegmentationConfig


class EntryPointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        vertices = np.array([[-.2, -.2, 0], [.2, -.2, 0], [.2, .2, 0], [-.2, .2, 0]])
        self.mesh = SimpleNamespace(vertices=vertices, faces=np.array([[0, 1, 2], [0, 2, 3]]),
                                    vertex_normals=np.tile((0, 0, 1), (4, 1)))
        self.pose = np.eye(4)
        self.pose[2, 3] = 1
        self.k = np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1]])
        self.depth = render_mesh_depth(self.mesh.vertices, self.mesh.faces, self.pose, self.k, (100, 100))
        self.rgb = np.zeros((100, 100, 3), np.uint8)
        self.rgb[self.depth > 0] = (0, 0, 200)
        pose = self.pose
        class FakeEstimator:
            last = None
            def __init__(s, **kwargs):
                s.pose_last = None
                s.registers = s.tracks = 0
                FakeEstimator.last = s
            def register(s, **kwargs):
                s.registers += 1
                s.pose_last = pose.copy()
                return pose.copy()
            def track_one(s, **kwargs):
                s.tracks += 1
                return pose.copy()
        class FakeSAM:
            metadata = {"test_backend": True}
            def begin(s, rgb): pass
            def predict(s, candidate): return np.array([candidate.coarse]), np.array([.99])
            def end(s): pass
        self.estimator_class = FakeEstimator
        self.segmenter = KFSSegmenter(SegmentationConfig().validate(), FakeSAM())
        patches = {
            "load_foundationpose": lambda: None,
            "FoundationPose": FakeEstimator,
            "ScorePredictor": lambda: None, "PoseRefinePredictor": lambda: None,
            "dr": SimpleNamespace(RasterizeCudaContext=lambda: None),
            "trimesh": SimpleNamespace(load=lambda *a, **kw: self.mesh,
                bounds=SimpleNamespace(oriented_bounds=lambda mesh: (np.eye(4), np.ones(3)))),
            "draw_posed_3d_box": lambda k, img, **kw: img,
            "draw_xyz_axis": lambda img, **kw: img,
            "set_logging_format": lambda: None, "set_seed": lambda seed: None,
            "enable_batched_scoring": lambda *a: None,
        }
        self.context = patch.multiple(offline, create=True, **patches)
        self.context.start()
        self.addCleanup(self.context.stop)
        backend_patch = patch.object(KFSSegmenter, "from_args", return_value=self.segmenter)
        backend_patch.start()
        self.addCleanup(backend_patch.stop)
        imageio = ModuleType("imageio")
        v2 = ModuleType("imageio.v2")
        v2.imread = lambda file: cv2.cvtColor(cv2.imread(str(file)), cv2.COLOR_BGR2RGB)
        imageio.v2 = v2
        modules_patch = patch.dict("sys.modules", {"imageio": imageio, "imageio.v2": v2})
        modules_patch.start()
        self.addCleanup(modules_patch.stop)

    def sequence(self):
        path = self.root / "sequence"
        (path / "rgb").mkdir(parents=True)
        (path / "depth").mkdir()
        np.savetxt(path / "cam_K.txt", self.k)
        for index in range(2):
            cv2.imwrite(str(path / "rgb" / f"{index:06d}.png"), cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(path / "depth" / f"{index:06d}.png"), (self.depth*1000).astype(np.uint16))
        return path

    def offline_args(self, output):
        argv = ["offline", "--sequence", str(self.sequence()), "--output", str(output),
                "--mask-source", "opencv-sam2", "--sam-checkpoint", "fake.pt",
                "--no-display", "--no-video", "--input-scale", "1"]
        with patch("sys.argv", argv):
            return offline.parse_args()

    def test_offline_auto_ignores_old_mask_registers_once(self):
        output = self.root / "result"
        output.mkdir()
        cv2.imwrite(str(output / "initial_mask.png"), np.zeros((10, 10), np.uint8))
        args = self.offline_args(output)
        offline.run_pipeline(args)
        estimator = self.estimator_class.last
        self.assertEqual((estimator.registers, estimator.tracks), (1, 1))
        self.assertEqual(len(list((output / "ob_in_cam").glob("*.txt"))), 2)
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        self.assertEqual([e["mode"] for e in events], ["register", "tracking"])

    def test_offline_auto_rejects_old_pose_directory(self):
        output = self.root / "old"
        (output / "ob_in_cam").mkdir(parents=True)
        np.savetxt(output / "ob_in_cam" / "000000.txt", self.pose)
        with self.assertRaisesRegex(ValueError, "新的"):
            offline.run_pipeline(self.offline_args(output))

    def test_live_auto_no_display_registers_once(self):
        class Connection:
            def settimeout(s, timeout): pass
            def connect(s, path): pass
            def close(s): s.closed = True
        connection = Connection()
        seq = iter(range(3))
        def receive(connection, roi=None):
            index = next(seq)
            if roi is not None:
                from kfs_roi import crop_frame
                rgb, depth, k = crop_frame(self.rgb, self.depth, self.k, roi)
                return rgb, depth, k, {"sequence": index, "stamp": float(index),
                    "frame_id": "optical", "full_shape": list(self.depth.shape),
                    "full_k": self.k.tolist(), "roi_xyxy": roi, "preview_rgb": self.rgb}
            return self.rgb, self.depth, self.k, {"sequence": index, "stamp": float(index), "frame_id": "optical"}
        output = self.root / "live_result"
        argv = ["live", "--camera", "zed2i", "--mask-source", "opencv-sam2",
                "--sam-checkpoint", "fake.pt", "--output", str(output),
                "--no-display", "--max-frames", "2", "--save-poses"]
        with patch("sys.argv", argv), patch.object(live.socket, "socket", return_value=connection), patch.object(live, "receive_frame", side_effect=receive):
            live.main()
        self.assertTrue(connection.closed)
        estimator = self.estimator_class.last
        self.assertEqual((estimator.registers, estimator.tracks), (1, 1))
        self.assertEqual(len(list((output / "ob_in_cam").glob("*.txt"))), 2)

    def test_manual_headless_requires_mask(self):
        with patch("sys.argv", ["live", "--camera", "zed2i", "--no-display"]):
            with self.assertRaises(SystemExit):
                live.parse_args()

    def test_live_color_labels_survive_flip_and_roi_then_clear_for_stale_gravity(self):
        from itertools import product
        from box4d.synthetic import Renderer, default_camera, world_box
        box = world_box((0, 0, .1), .2, 15)
        box['red_axis_local'] = (0, 0, 1)
        frame, _, _ = Renderer(default_camera(320, 240)).render([box])
        vertices = np.array(list(product((-.1, .1), (-.1, .1), (0., .2))))
        colors = np.array([(235, 35, 35, 255) if z > .1 else (40, 45, 225, 255)
                           for x, y, z in vertices], np.uint8)
        self.mesh = SimpleNamespace(vertices=vertices, vertex_normals=np.zeros_like(vertices),
                                    visual=SimpleNamespace(kind='vertex', vertex_colors=colors))
        sequence = iter(range(3))
        def receive(connection, roi=None):
            index = next(sequence)
            self.pose[:] = box['pose']
            if index:
                self.pose[:3, :3] = self.pose[:3, :3] @ np.diag([1, -1, -1])
            self.pose[:3, 3] -= self.pose[:3, :3] @ [0, 0, .1]
            metadata = dict(sequence=index, stamp=100.+index, frame_id='optical',
                            up_reference=dict(status='ok', up_camera=[0, -1, 0],
                                camera_frame_id='optical', last_imu_stamp=100.+index if index < 2 else 100.))
            rgb, depth, k = frame.rgb, frame.depth, frame.camera.k
            if index == 1:
                from kfs_roi import crop_frame
                roi = [80, 70, 240, 235]
                rgb, depth, k = crop_frame(rgb, depth, k, roi)
                metadata.update(roi_xyxy=roi, preview_rgb=frame.rgb,
                                full_shape=list(frame.depth.shape), full_k=frame.camera.k.tolist())
            return rgb, depth, k, metadata

        class Connection:
            def settimeout(self, value): pass
            def connect(self, path): pass
            def close(self): pass
        submitted = []
        class Display:
            def __init__(self, shape, title): self.process = SimpleNamespace(is_alive=lambda: True)
            def submit(self, image): submitted.append(image.copy())
            def poll_key(self): return -1
            def close(self): pass
        mask = self.root / 'color_mask.png'
        cv2.imwrite(str(mask), np.ones(frame.depth.shape, np.uint8) * 255)
        output = self.root / 'color_live_result'
        argv = ['live', '--camera', 'd435i', '--mask', str(mask),
                '--output', str(output), '--max-frames', '3']
        with patch('sys.argv', argv), patch.object(live.socket, 'socket', return_value=Connection()), \
             patch.object(live, 'receive_frame', side_effect=receive), \
             patch('common.display.AsyncPoseDisplay', Display), \
             patch.object(offline, 'draw_xyz_axis', side_effect=lambda image, **kw: np.full_like(image, 128)):
            live.main()
        records = [json.loads(line) for line in (output / 'color_orientation.jsonl').read_text().splitlines()]
        self.assertEqual([record['label'] for record in records], ['redup', 'redup', 'unknown'])
        self.assertEqual(records[-1]['status'], 'gravity_unavailable')
        self.assertEqual(len(submitted), 3)
        self.assertTrue(all(image.shape == (240, 320, 3) for image in submitted))
        # Overlay changes cannot enter the color classifier; labels reach the submitted image.
        self.assertTrue(all(np.any(image[-40:] != 128) for image in submitted))


if __name__ == "__main__":
    unittest.main()
