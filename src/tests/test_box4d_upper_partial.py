"""Upper-only RGB-D recovery and progressive history validation."""

from copy import deepcopy
import unittest

import cv2
import numpy as np

from box4d.config import Config
from box4d.io import Frame
from box4d.partial_history import UpperPartialHistory
from box4d.pose import Pipeline
from box4d.synthetic import Renderer, default_camera, world_box
from box4d.temporal import TemporalPipeline


class UpperPartialTests(unittest.TestCase):
    def cfg(self, **overrides):
        options = dict(sizes_m=(.35,), up_camera=[0, -1, 0], analyze_colors=True,
                       require_target_colors=True, temporal_recovery=True,
                       async_detection=False, max_depth_samples=512, plane_grid_step=8)
        options.update(overrides)
        return Config(**options).validate()

    def scene(self, sequence=0, center_y=.30, center_x=0., occluded=False):
        z = .9 if occluded else .65
        camera = default_camera()
        box = world_box((center_x, 0, .175), .35, 0,
                        (0, -z-.175, .175+center_y), color=(225, 35, 35))
        occluders = ([dict(quad_camera=[[-1, 0, .6], [1, 0, .6], [1, 1, .6], [-1, 1, .6]])]
                     if occluded else [])
        frame, truth, _ = Renderer(camera).render([box], sequence, depth_noise_m=.001,
                                                  floor_height_camera_m=1., occluders=occluders)
        for x in np.linspace(-.12, .12, 5):
            for y in np.linspace(-.12, .12, 5):
                point = [center_x+x, center_y+y, z]
                if occluded and point[1] >= 0:
                    continue
                p = camera.project([point])[0]
                cv2.rectangle(frame.rgb, tuple(np.rint(p-3).astype(int)),
                              tuple(np.rint(p+3).astype(int)), (230, 230, 230), -1)
        return frame, truth[0]['center_camera_m']

    def test_cold_start_upper_half_at_image_edge_has_current_depth_pose(self):
        frame, truth = self.scene()
        result, _ = Pipeline(self.cfg()).process(frame)
        self.assertEqual(len(result['objects']), 1)
        obj = result['objects'][0]
        self.assertTrue(obj['metrics']['upper_partial'])
        self.assertEqual(obj['method'], 'depth_partial')
        self.assertGreater(min(obj['metrics']['anchor_edge_support']), .6)
        np.testing.assert_allclose(obj['center_camera_m'], truth, atol=.008)

    def test_only_lower_half_or_one_side_cannot_initiate_pose(self):
        for offsets in [dict(center_y=-.30), dict(center_x=.45)]:
            frame, _ = self.scene(**offsets)
            result, _ = Pipeline(self.cfg()).process(frame)
            self.assertEqual(result['objects'], [], offsets)

    def test_current_upper_depth_holes_do_not_inherit_a_previous_pose(self):
        frame, _ = self.scene()
        first, _ = Pipeline(self.cfg()).process(frame)
        self.assertTrue(first['objects'])
        frame.depth[::2] = 0
        failed, _ = Pipeline(self.cfg()).process(frame)
        self.assertEqual(failed['objects'], [])

    def test_async_partial_seed_and_advance_use_only_visible_measurements(self):
        source, truth = self.scene(center_y=0., occluded=True)
        cfg = self.cfg()
        detection, _ = Pipeline(cfg).process(source)
        self.assertEqual(len(detection['objects']), 1)
        self.assertTrue(detection['objects'][0]['metrics']['upper_partial'])
        pipeline = TemporalPipeline(cfg)
        # A completed background result, with real measured source geometry.
        # No worker thread or wall-clock scheduling is needed for this check.
        pipeline.asynchronous = True
        pipeline.worker_completed = (source, detection, [], 10., (None, ()))
        frame, _ = self.scene(1, center_y=0., occluded=True)
        result, _ = pipeline.process(frame)
        self.assertEqual(len(result['objects']), 1, result['temporal']['detector_seed_failures'])
        obj = result['objects'][0]
        np.testing.assert_allclose(obj['center_camera_m'], truth, atol=.01)
        self.assertTrue(obj['metrics']['upper_partial'])
        self.assertEqual(obj['metrics']['partial_history']['partial_streak'], 1)
        next_frame, _ = self.scene(2, center_y=0., occluded=True)
        result, _ = pipeline.process(next_frame)
        self.assertEqual(len(result['objects']), 1, result['temporal'])
        self.assertTrue(result['objects'][0]['metrics']['partial_history']['position_gate_enabled'])
        gone, _ = self.scene(3, center_y=0., occluded=True)
        gone.depth[:] = 0
        result, _ = pipeline.process(gone)
        self.assertEqual(result['objects'], [])


class PartialHistoryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(sizes_m=(.35,), up_camera=[0, -1, 0])
        self.history = UpperPartialHistory(self.cfg)

    def frame(self, t, rotation=None):
        camera = default_camera()
        frame = Frame(np.zeros((480, 640, 3), np.uint8), np.ones((480, 640), np.float32),
                      camera, int(t*1e9))
        if rotation is not None:
            frame.camera_motion = dict(reference_id='same', stamp_ns=frame.stamp_ns,
                                       rotation_reference_from_camera=rotation.tolist())
        return frame

    def obj(self, x=0., partial=True, center=None, support=1., edges=1.):
        return dict(size_m=.35, center_camera_m=[x, 0., 1.] if center is None else center,
                    quality_score=1., metrics=dict(upper_partial=partial,
                    visible_plane_support_fraction=support, anchor_edge_support=[edges]*3))

    def observe(self, t, objects):
        kept, rejected = self.history.evaluate(self.frame(t), objects)
        self.history.commit(kept)
        return kept, rejected

    def test_first_partial_uses_soft_history_even_after_large_translation(self):
        self.observe(1., [self.obj(partial=False)])
        kept, rejected = self.observe(1.05, [self.obj(x=.4)])
        self.assertFalse(rejected)
        self.assertFalse(kept[0]['metrics']['partial_history']['position_gate_enabled'])
        self.assertLess(kept[0]['quality_score'], 1.)

    def test_second_partial_rejects_jump_and_cannot_reset_by_a_missing_frame(self):
        self.observe(1., [self.obj()])
        self.observe(1.02, [])
        kept, rejected = self.observe(1.04, [self.obj(x=.4)])
        self.assertEqual(kept, [])
        self.assertIn('partial_history_position', rejected[0]['failed_checks'])
        kept, _ = self.observe(1.06, [self.obj(x=.01)])
        self.assertEqual(kept[0]['metrics']['partial_history']['partial_streak'], 2)

    def test_second_partial_demands_stronger_depth_and_each_anchor_edge(self):
        for fields, reason in [(dict(support=.8), 'partial_history_depth'),
                               (dict(edges=.65), 'partial_history_boundaries')]:
            self.history.clear()
            self.observe(1., [self.obj()])
            kept, rejected = self.observe(1.04, [self.obj(**fields)])
            self.assertEqual(kept, [])
            self.assertIn(reason, rejected[0]['failed_checks'])

    def test_position_allowance_grows_with_elapsed_time(self):
        self.observe(1., [self.obj()])
        fast, _ = self.history.evaluate(self.frame(1.02), [self.obj(x=.15)])
        slow, _ = self.history.evaluate(self.frame(1.15), [self.obj(x=.15)])
        self.assertFalse(fast)
        self.assertTrue(slow)

    def test_complete_observation_resets_partial_streak(self):
        self.observe(1., [self.obj()])
        self.observe(1.04, [self.obj(partial=False)])
        kept, rejected = self.observe(1.08, [self.obj(x=.4)])
        self.assertFalse(rejected)
        self.assertEqual(kept[0]['metrics']['partial_history']['partial_streak'], 1)

    def test_current_camera_rotation_transforms_the_historical_reference(self):
        self.cfg.camera_rotation_compensation = True
        old = self.obj()
        kept, _ = self.history.evaluate(self.frame(1., np.eye(3)), [old])
        self.history.commit(kept)
        rotation = cv2.Rodrigues(np.array([0., .3, 0.]))[0]
        center = rotation.T @ np.asarray(old['center_camera_m'])
        kept, rejected = self.history.evaluate(self.frame(1.02, rotation), [self.obj(center=center.tolist())])
        self.assertFalse(rejected)
        self.assertLess(kept[0]['metrics']['partial_history']['position_residual_m'], 1e-9)

    def test_duplicate_fresh_detection_cannot_bypass_the_gate(self):
        self.observe(1., [self.obj()])
        kept, rejected = self.observe(1.02, [self.obj(x=.4), self.obj(x=.41)])
        self.assertEqual(kept, [])
        self.assertEqual(len(rejected), 2)

    def test_two_targets_keep_independent_histories(self):
        self.observe(1., [self.obj(x=-.4), self.obj(x=.4, partial=False)])
        kept, rejected = self.observe(1.02, [self.obj(x=-.39), self.obj(x=.6)])
        self.assertFalse(rejected)
        self.assertEqual([o['metrics']['partial_history']['partial_streak'] for o in kept], [2, 1])


if __name__ == '__main__':
    unittest.main()
