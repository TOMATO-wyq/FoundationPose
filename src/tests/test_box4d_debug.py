"""Debug evidence must explain the actual checks without changing inference."""
from copy import deepcopy
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from box4d.config import Config
from box4d.debug import (STAGES, TUNING_PAGES, TUNING_SPECS, TUNING_ZH,
                        _SnapshotSaver, _display_stages, render_stages, save_snapshot, tuning_positions, tuning_values)
from box4d.detector import Face
from box4d.io import Frame, json_value
from box4d.pose import Pipeline
from box4d.synthetic import Renderer, default_camera, world_box
from box4d.temporal import TemporalPipeline


class DebugTests(unittest.TestCase):
    def setUp(self):
        self.renderer = Renderer(default_camera())

    def test_slow_snapshot_writes_do_not_block_tracking_and_queue_is_bounded(self):
        entered,release,returned=threading.Event(),threading.Event(),threading.Event()
        saver=_SnapshotSaver()
        def slow_save(*args,**kwargs):
            entered.set()
            release.wait(timeout=3.)
        def submit():
            saver.submit(Path('first'),(),None)
            returned.set()
        with patch('box4d.debug.save_snapshot',side_effect=slow_save) as write:
            submitter=threading.Thread(target=submit)
            try:
                submitter.start()
                self.assertTrue(entered.wait(timeout=1.))
                self.assertTrue(returned.wait(timeout=1.),'disk compression blocked the caller')
                self.assertTrue(saver.submit(Path('second'),(),None))
                self.assertFalse(saver.submit(Path('third'),(),None))
            finally:
                release.set()
                submitter.join(timeout=1.)
                saver.close()
            self.assertEqual(write.call_count,2)

    def test_snapshot_write_failure_is_reported_when_closing(self):
        saver=_SnapshotSaver()
        output, errors = io.StringIO(), io.StringIO()
        with patch('box4d.debug.save_snapshot',side_effect=OSError('disk full')), \
                redirect_stdout(output), redirect_stderr(errors):
            saver.submit(Path('failed'),(),None)
            saver.close()
            saver.close()
        self.assertIn('failed: failed: OSError: disk full', errors.getvalue())
        self.assertNotIn('snapshot saved:', output.getvalue())
        self.assertEqual(saver.failure_count, 1)
        self.assertEqual(list(saver.errors), [dict(path='failed', error_type='OSError', message='disk full')])
        self.assertFalse(saver.pending)

    def test_depth_edge_slider_number_is_the_actual_millimeter_threshold(self):
        class SharedPositions(list):
            def get_lock(self):
                return nullcontext()

        shared = SharedPositions(tuning_positions(Config().validate()))
        created, displayed, pages = {}, {}, []

        def open_window(name, flags):
            if name.startswith('09'):
                pages.append([])

        def create(label, window, initial, maximum, callback):
            english = label.split(' | ')[1]
            created[english] = (initial, maximum)
            pages[-1].append(label)
            # Simulate changing the displayed depth threshold to 40 mm.
            displayed[label] = 40 if english == 'Depth edge mm' else initial

        shape = (len(STAGES), 600, 720, 3)
        with patch('box4d.debug.cv2.namedWindow', side_effect=open_window), patch('box4d.debug.cv2.resizeWindow'), \
                patch('box4d.debug.cv2.moveWindow'), patch('box4d.debug.cv2.createTrackbar', side_effect=create), \
                patch('box4d.debug.cv2.imshow'), patch('box4d.debug.cv2.destroyWindow'), \
                patch('box4d.debug.cv2.setMouseCallback'), \
                patch('box4d.debug.cv2.setTrackbarMin', create=True), \
                patch('box4d.debug.cv2.getTrackbarPos', side_effect=lambda label, window: displayed[label]), \
                patch('box4d.debug.cv2.setTrackbarPos', side_effect=lambda label, window, value: displayed.update({label: value})), \
                patch('box4d.debug.cv2.waitKey', side_effect=[ord(']')] * (len(TUNING_PAGES) + 1) + [ord('q')]), \
                patch('box4d.debug.cv2.destroyAllWindows'):
            _display_stages(np.zeros(shape, np.uint8), threading.Lock(), SimpleNamespace(value=0),
                            SimpleNamespace(value=0), SimpleNamespace(value=-1), SimpleNamespace(value=False),
                            shape, 'grid', shared)
        self.assertEqual(created['Depth edge mm'], (40, 100))  # Still 40 when revisiting its page.
        self.assertEqual(created['PCL smooth px'][0], 12)
        self.assertEqual(created['Canny high'][0], 100)
        self.assertAlmostEqual(tuning_values(shared)['depth_edge_threshold_m'], .04)
        self.assertEqual(len(pages), len(TUNING_PAGES) + 2)
        self.assertTrue(all(len(labels) <= 7 for labels in pages))
        covered = [i for _, _, start, end in TUNING_PAGES for i in range(start, end)]
        self.assertEqual(covered, list(range(len(TUNING_SPECS))))
        self.assertEqual(len(TUNING_ZH), len(TUNING_SPECS))

    def test_front_and_two_face_debug_does_not_change_geometry(self):
        for angle in (0, 30, 45):
            with self.subTest(angle=angle):
                frame, _, _ = self.renderer.render([world_box((0, 0, .175), .35, angle)], depth_noise_m=.001)
                cfg = Config(up_camera=[0, -1, 0], plane_grid_step=8).validate()
                plain, _ = Pipeline(deepcopy(cfg)).process(frame)
                diagnostic = Pipeline(deepcopy(cfg), debug=True)
                result, _ = diagnostic.process(frame)
                self.assertEqual(json_value(plain['objects']), json_value(result['objects']))
                self.assertEqual(plain['rejected_faces'], result['rejected_faces'])
                self.assertEqual(len(result['objects']), 1)
                trace = diagnostic.last_debug
                self.assertIs(trace.frame, frame)
                self.assertTrue(all(f['status'] in ('geometry_pass', 'rejected') for f in trace.faces))
                if angle == 45:
                    normals = [p['normal'] for p in trace.planes if p['status'] == 'kept']
                    self.assertTrue(any(abs(np.dot(a, b)) < .15 for a in normals for b in normals))
                panels = render_stages(frame, result, trace)
                self.assertEqual(panels.shape, (len(STAGES), 600, 720, 3))
                self.assertEqual(panels.dtype, np.uint8)

    def _forced_face(self, depth):
        camera = default_camera()
        frame = Frame(np.zeros((480, 640, 3), np.uint8), depth, camera, 1)
        face = Face(np.array([[150., 100.], [490., 100.], [490., 380.], [150., 380.]]), 'test', 1.)
        pipeline = Pipeline(Config(allow_stack_inference=False, allow_bottom_occlusion=False).validate(), debug=True)
        with patch.object(pipeline.detector, 'propose', return_value=([face], dict(detector_ms=0., proposal_count=1), np.zeros((480, 640), np.uint8))):
            result, _ = pipeline.process(frame)
        return frame, result, pipeline.last_debug

    def test_mixed_face_records_rejected_plane_support_before_returning(self):
        depth = np.ones((480, 640), np.float32)
        depth[:, 320:] = 1.15
        _, result, trace = self._forced_face(depth)
        self.assertEqual(result['objects'], [])
        face = trace.faces[0]
        self.assertEqual(face['reason'], 'nonplanar or mixed-depth face')
        self.assertLess(face['plane_support'], .65)
        self.assertLess(face['plane_rmse_m'], .006)
        self.assertEqual(len(face['sample_uv']), len(face['sample_residual_m']))

    def test_missing_depth_reports_coverage_without_inventing_a_plane(self):
        frame, result, trace = self._forced_face(np.zeros((480, 640), np.float32))
        face = trace.faces[0]
        self.assertEqual(face['depth_fraction'], 0.)
        self.assertEqual(face['valid_depth_pixels'], 0)
        self.assertNotIn('plane_normal', face)
        self.assertIn('insufficient interior depth', face['reason'])
        self.assertEqual(render_stages(frame, result, trace).shape[0], 8)

    def test_async_snapshot_keeps_detection_frame_separate_from_current_frame(self):
        cfg = Config(up_camera=[0, -1, 0], temporal_recovery=True, plane_grid_step=8).validate()
        pipeline = TemporalPipeline(cfg, live=True, debug=True)
        try:
            first, _, _ = self.renderer.render([world_box((0, 0, .175), .35, 30)], sequence=0)
            pipeline.process(first)
            with pipeline.worker_condition:
                self.assertTrue(pipeline.worker_condition.wait_for(
                    lambda: pipeline.worker_completed is not None or pipeline.worker_error is not None, timeout=3.))
            current, _, _ = self.renderer.render([], sequence=1)
            result, _ = pipeline.process(current)
            self.assertIs(pipeline.last_debug.frame, first)
            self.assertEqual(result['stamp_ns'], current.stamp_ns)
            self.assertEqual(result['objects'], [])
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'snapshot'
                save_snapshot(path, current, result, pipeline.last_debug, pipeline.local_debug, pipeline.debug_tracks)
                report = json.loads((path / 'measurements.json').read_text())
                self.assertEqual(report['current']['stamp_ns'], current.stamp_ns)
                self.assertEqual(report['detection']['stamp_ns'], first.stamp_ns)
                self.assertEqual(len(list(path.glob('*.png'))), 8)
                with np.load(path / 'current_rgbd.npz') as saved:
                    np.testing.assert_equal(saved['depth'], current.depth)
                with np.load(path / 'detection_rgbd.npz') as saved:
                    np.testing.assert_equal(saved['depth'], first.depth)
        finally:
            pipeline.close()


if __name__ == '__main__':
    unittest.main()
