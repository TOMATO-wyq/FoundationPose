"""CPU regression tests for visible-mask safety and recovery (no SAM/GPU needed)."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import cv2
import numpy as np

from kfs_segmentation import (KFSSegmenter, NoTarget, SegmentationConfig,
                              detect_candidates, masked_median, refine_mask)
from kfs_pose_controller import PoseController, pose_health, render_mesh_depth
from evaluate_kfs_masks import mask_metrics


class DepthRefinementTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SegmentationConfig(min_mask_pixels=4, min_valid_depth_pixels=4,
                                      small_component_pixels=1).validate()
        self.raw = np.zeros((40, 60), bool)
        self.raw[12:28, 20:36] = True
        self.safe = np.zeros_like(self.raw)
        self.safe[17:23, 25:31] = True

    def test_growth_stops_at_fixed_distance(self):
        depth = np.ones(self.raw.shape, np.float32)
        mask, diagnostics = refine_mask(self.raw, depth, self.safe, self.cfg)
        distance = cv2.distanceTransform((~self.raw).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        self.assertTrue(mask[self.raw].all())
        self.assertTrue(np.all(distance[mask & ~self.raw] <= self.cfg.growth_px))
        self.assertFalse(mask[12, 14])
        self.assertTrue(diagnostics["added"].any())

    def test_depth_jump_blocks_background(self):
        depth = np.full(self.raw.shape, 2., np.float32)
        depth[10:30, 18:38] = 1.
        mask, _ = refine_mask(self.raw, depth, self.safe, self.cfg)
        self.assertFalse(np.any(mask & (depth > 1.5)))

    def test_tilted_faces_not_globally_clipped(self):
        y, x = np.indices(self.raw.shape)
        depth = (1.0 + x*.008).astype(np.float32)
        mask, _ = refine_mask(self.raw, depth, self.safe, self.cfg)
        self.assertTrue(mask[self.raw].all())
        self.assertGreater(float(depth[self.raw].max()-depth[self.raw].min()), .1)

    def test_occluder_deleted_not_closed(self):
        depth = np.ones(self.raw.shape, np.float32)
        depth[12:28, 20:24] = .6
        self.cfg.closing_size = 3
        mask, diagnostics = refine_mask(self.raw, depth, self.safe, self.cfg)
        self.assertFalse(mask[16, 21])
        self.assertTrue(diagnostics["removed"][16, 21])

    def test_invalid_depth_retained_inside_never_added_outside(self):
        depth = np.ones(self.raw.shape, np.float32)
        depth[18, 27] = 0
        depth[12:28, 36:40] = np.nan
        mask, _ = refine_mask(self.raw, depth, self.safe, self.cfg)
        self.assertTrue(mask[18, 27])
        self.assertFalse(mask[12:28, 36:40].any())
        filtered, valid = masked_median(depth, self.cfg)
        self.assertTrue((filtered[~valid] == 0).all())

    def test_flying_pixel_cannot_become_growth_bridge(self):
        depth = np.ones(self.raw.shape, np.float32)
        depth[20, 36] = 1.4
        mask, _ = refine_mask(self.raw, depth, self.safe, self.cfg)
        self.assertFalse(mask[20, 36])

    def test_no_seed_rejected(self):
        with self.assertRaises(NoTarget):
            refine_mask(self.raw, np.zeros(self.raw.shape, np.float32), self.safe, self.cfg)


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((100, 120, 3), np.uint8)
        self.rgb[20:80, 30:90] = (0, 0, 200)
        self.depth = np.full((100, 120), 2., np.float32)
        self.depth[20:80, 30:90] = 1.
        self.cfg = SegmentationConfig().validate()

    def test_points_safe_and_separated(self):
        candidate = detect_candidates(self.rgb, self.depth, self.cfg, "box3")[0]
        self.assertEqual(len(candidate.points), 3)
        p = candidate.points.astype(int)
        self.assertTrue(candidate.safe[p[:, 1], p[:, 0]].all())
        for i in range(3):
            for j in range(i):
                self.assertGreaterEqual(np.linalg.norm(p[i]-p[j]), .2*min(candidate.box[2]-candidate.box[0]+1, candidate.box[3]-candidate.box[1]+1))

    def test_negative_only_in_measured_background(self):
        candidate = detect_candidates(self.rgb, self.depth, self.cfg, "box3neg")[0]
        negatives = candidate.points[candidate.labels == 0].astype(int)
        self.assertGreater(len(negatives), 0)
        self.assertTrue((self.depth[negatives[:, 1], negatives[:, 0]] == 2).all())
        self.depth[:] = 1
        candidate = detect_candidates(self.rgb, self.depth, self.cfg, "box3neg")[0]
        self.assertFalse((candidate.labels == 0).any())

    def test_frame_edge_points_eroded(self):
        self.rgb[:] = 0
        self.rgb[:60, :60] = (0, 0, 200)
        candidate = detect_candidates(self.rgb, self.depth, self.cfg)[0]
        self.assertFalse(candidate.safe[0].any())
        self.assertFalse(candidate.safe[:, 0].any())

    def test_failed_sam_masks_rejected_and_offloaded(self):
        class FakeSAM:
            def begin(s, rgb): pass
            def predict(s, candidate): return np.zeros((3, 100, 120), bool), np.array([.9, .8, .7])
            def end(s): s.ended = True
        sam = FakeSAM()
        with self.assertRaises(NoTarget):
            KFSSegmenter(self.cfg, sam).segment(self.rgb, self.depth)
        self.assertTrue(sam.ended)

    def test_validation_rejects_nonfinite_configuration(self):
        with self.assertRaises(ValueError):
            SegmentationConfig(local_delta_m=float("nan")).validate()


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = SegmentationConfig(min_mask_pixels=4, min_valid_depth_pixels=4,
                                      health_interval_s=0).validate()
        self.vertices = np.array([[-.2, -.2, 0], [.2, -.2, 0], [.2, .2, 0], [-.2, .2, 0]])
        self.faces = np.array([[0, 1, 2], [0, 2, 3]])
        self.mesh = SimpleNamespace(vertices=self.vertices, faces=self.faces)
        self.pose = np.eye(4)
        self.pose[2, 3] = 1
        self.k = np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1]])
        self.depth = render_mesh_depth(self.vertices, self.faces, self.pose, self.k, (100, 100))
        self.rgb = np.zeros((100, 100, 3), np.uint8)
        self.rgb[self.depth > 0] = (0, 0, 200)
        class FakeSAM:
            metadata = {"fake": True}
            def begin(s, rgb): pass
            def predict(s, candidate): return np.array([candidate.coarse]), np.array([.99])
            def end(s): pass
        self.segmenter = KFSSegmenter(self.cfg, FakeSAM())
        pose = self.pose
        class FakeEstimator:
            pose_last = None
            registers = tracks = 0
            fail = False
            def register(s, **kwargs):
                s.registers += 1
                s.pose_last = pose.copy()
                return pose.copy()
            def track_one(s, **kwargs):
                s.tracks += 1
                return np.full((4, 4), np.nan) if s.fail else pose.copy()
        self.estimator = FakeEstimator()
        self.controller = PoseController(self.estimator, self.segmenter, self.mesh, self.temp.name)

    def test_register_then_track_no_segmentation(self):
        pose, event = self.controller.process(self.rgb, self.depth, self.k, "0", now=0)
        self.assertIsNotNone(pose)
        self.assertEqual(event["state"], "TRACK")
        for frame in range(1, 5):
            pose, event = self.controller.process(self.rgb, self.depth, self.k, str(frame), now=frame)
            self.assertIsNotNone(pose)
        self.assertEqual(self.estimator.registers, 1)
        self.assertEqual(self.estimator.tracks, 4)
        self.assertEqual(self.controller.attempts, 1)

    def test_three_bad_frames_suppress_pose_and_recover(self):
        self.controller.process(self.rgb, self.depth, self.k, "0", now=0)
        bad = np.zeros_like(self.depth)
        for frame in range(1, 4):
            pose, event = self.controller.process(self.rgb, bad, self.k, str(frame), now=frame)
            self.assertIsNone(pose)
        self.assertEqual(self.controller.state, "LOST")
        self.assertIsNone(self.estimator.pose_last)
        pose, event = self.controller.process(self.rgb, self.depth, self.k, "4", now=3.1)
        self.assertIsNone(pose)
        self.assertEqual(event["reason"], "search_cooldown")
        pose, event = self.controller.process(self.rgb, self.depth, self.k, "5", now=4)
        self.assertIsNotNone(pose)
        self.assertEqual(self.estimator.registers, 2)

    def test_invalid_pose_immediately_lost(self):
        self.controller.process(self.rgb, self.depth, self.k, "0", now=0)
        self.estimator.fail = True
        pose, event = self.controller.process(self.rgb, self.depth, self.k, "1", now=1)
        self.assertIsNone(pose)
        self.assertEqual(event["state"], "LOST")

    def test_no_target_skips_registration(self):
        pose, event = self.controller.process(np.zeros_like(self.rgb), self.depth, self.k, "0", now=0)
        self.assertIsNone(pose)
        self.assertEqual(self.estimator.registers, 0)

    def test_manual_reset_reinitializes_current_scene(self):
        self.controller.process(self.rgb, self.depth, self.k, "0", now=0)
        self.controller.reset()
        self.controller.process(self.rgb, self.depth, self.k, "1", now=.1)
        self.assertEqual(self.estimator.registers, 2)

    def test_periodic_health_counts_checks_not_skipped_frames(self):
        self.cfg.health_interval_s = 2
        self.controller.process(self.rgb, self.depth, self.k, '0', now=0)
        bad = np.full_like(self.depth, 2.)
        _, event = self.controller.process(self.rgb, bad, self.k, '1', now=1)
        self.assertEqual(event['health']['reason'], 'geometry_check_skipped')
        self.assertEqual(self.controller.failures, 0)
        self.controller.process(self.rgb, bad, self.k, '2', now=2)
        self.assertEqual(self.controller.failures, 1)
        self.controller.process(self.rgb, bad, self.k, '3', now=3)
        self.assertEqual(self.controller.failures, 1)
        self.controller.process(self.rgb, bad, self.k, '4', now=4)
        _, event = self.controller.process(self.rgb, bad, self.k, '5', now=6)
        self.assertEqual(event['state'], 'LOST')

    def test_foreground_occlusion_not_counted_as_support(self):
        covered = self.depth.copy()
        covered[covered > 0] = .5
        healthy, metrics = pose_health(self.pose, covered, self.k, self.vertices, self.faces, self.cfg)
        self.assertFalse(healthy)
        self.assertEqual(metrics["support_fraction"], 0)
        self.assertEqual(metrics["occluded_fraction"], 1)

    def test_perspective_renderer_depth(self):
        depth = self.depth[self.depth > 0]
        np.testing.assert_allclose(depth, 1)


class QualityMetricsTests(unittest.TestCase):
    def test_failed_predictions_count_as_zero_for_nonempty_truth(self):
        truth = np.zeros((20, 20), bool)
        truth[5:15, 5:15] = True
        result = mask_metrics(np.zeros_like(truth), truth)
        self.assertEqual(result["iou"], 0)
        self.assertEqual(result["boundary_f1_2px"], 0)
        self.assertEqual(mask_metrics(truth, truth)["boundary_f1_2px"], 1)


if __name__ == "__main__":
    unittest.main()
