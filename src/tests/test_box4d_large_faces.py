"""Large observed faces, bounded depth noise and close-range tracking."""

import unittest

import cv2
import numpy as np

from box4d.config import Config
from box4d.detector import Detector, Face, sample_face
from box4d.geometry import plane_limits
from box4d.io import Frame
from box4d.pose import Pipeline, prepare_face
from box4d.synthetic import Renderer, default_camera, world_box
from box4d.temporal import TemporalPipeline


class LargeFaceTests(unittest.TestCase):
    def cfg(self, **overrides):
        options = dict(sizes_m=(.35,), up_camera=[0, -1, 0], analyze_colors=True,
                       require_target_colors=True, temporal_recovery=True, plane_distance_noise=True)
        options.update(overrides)
        return Config(**options)

    def close_frame(self, z, sequence=0, size=.35):
        camera = default_camera()
        box = world_box((0, 0, size / 2), size, 0, (0, -z-size/2, size/2), color=(40, 45, 225))
        frame, truth, _ = Renderer(camera).render([box], sequence, depth_noise_m=.001, floor_height_camera_m=1.)
        # Physical, spatially distributed white markings on the front plane.
        for x in np.linspace(-.12, .12, 5):
            for y in np.linspace(-.12, .12, 5):
                p = camera.project([[x, y, z]])[0]
                cv2.rectangle(frame.rgb, tuple(np.rint(p-3).astype(int)), tuple(np.rint(p+3).astype(int)), (230, 230, 230), -1)
        return frame, truth

    def test_complete_face_filling_93_percent_of_image_height(self):
        frame, truth = self.close_frame(.36)
        for backend in ("numpy", "auto"):
            result, _ = Pipeline(self.cfg(geometry_backend=backend)).process(frame)
            self.assertEqual(len(result["objects"]), 1)
            np.testing.assert_allclose(result["objects"][0]["center_camera_m"], truth[0]["center_camera_m"], atol=.005)

    def test_existing_face_stops_when_physical_top_is_clipped(self):
        pipeline = TemporalPipeline(self.cfg())
        for i, z in enumerate((.40, .38, .36, .34, .32, .30)):
            frame, truth = self.close_frame(z, i)
            result, _ = pipeline.process(frame)
            if z <= .32:
                self.assertEqual(result['objects'], [], z)
                continue
            self.assertEqual(len(result["objects"]), 1, z)
            obj = result["objects"][0]
            np.testing.assert_allclose(obj["center_camera_m"], truth[0]["center_camera_m"], atol=.005)
        self.assertEqual(obj["source"], "current_rgbd_flow")
        self.assertGreaterEqual(obj["metrics"]["tracked_features"], 8)
        # No stale pose when the current target/depth disappears.
        frame, _ = self.close_frame(.30, 6)
        frame.rgb[:] = 48
        frame.depth[:] = 0
        result, _ = pipeline.process(frame)
        self.assertEqual(result["objects"], [])

    def test_clipped_cold_start_does_not_invent_complete_box_pose(self):
        frame, _ = self.close_frame(.30)
        result, _ = Pipeline(self.cfg()).process(frame)
        self.assertEqual(result["objects"], [])
        quad = np.array([[50, -30], [590, -30], [590, 510], [50, 510]], float)
        with self.assertRaisesRegex(ValueError, "clipped by image"):
            sample_face(frame, quad, self.cfg())
        with self.assertRaisesRegex(ValueError, "clipped by search ROI"):
            sample_face(frame, quad, self.cfg(roi_xyxy=[0, 0, 640, 480]), allow_clipped=True)

    def noisy_face(self):
        c = default_camera()
        q = c.project([[-.175, -.175, 1.4], [.175, -.175, 1.4], [.175, .175, 1.4], [-.175, .175, 1.4]])
        rgb = np.full((480, 640, 3), 48, np.uint8)
        depth = np.zeros((480, 640), np.float32)
        mask = np.zeros(depth.shape, np.uint8)
        cv2.fillConvexPoly(mask, np.rint(q).astype(np.int32), 1)
        rgb[mask > 0] = (40, 45, 225)
        depth[mask > 0] = 1.4 + np.random.default_rng(12).normal(0, .009, np.count_nonzero(mask))
        return Frame(rgb, depth, c, 1), Face(q, "measured_test", 1.)

    def test_distance_noise_passes_without_accepting_a_depth_step(self):
        frame, face = self.noisy_face()
        cfg = self.cfg(max_depth_samples=1024, min_depth_samples=109)
        with self.assertRaises(ValueError):
            prepare_face(frame, face, self.cfg(plane_distance_noise=False, max_depth_samples=1024, min_depth_samples=109))
        details = {}
        prepare_face(frame, face, cfg, details)
        self.assertEqual(details["plane_threshold_m"], .012)
        self.assertLessEqual(details["plane_rmse_m"], .008)
        # A flat majority must not conceal another depth on one side.
        frame.depth[frame.depth > 0] = 1.4
        frame.depth[:, :295] += (frame.depth[:, :295] > 0) * .08
        for limit in (3., 5.):
            cfg.max_reprojection_px = limit
            with self.assertRaisesRegex(ValueError, "mixed-depth"):
                prepare_face(frame, face, cfg)

    def test_noise_limits_keep_near_floor_and_far_cap(self):
        cfg = self.cfg()
        self.assertEqual(plane_limits(np.array([[0., 0., .3]]), cfg), (.008, .006))
        np.testing.assert_allclose(plane_limits(np.array([[0., 0., 3.]]), cfg), (.012, .008))

    def test_logo_hole_is_proposal_only_and_coplanar_targets_remain_separate(self):
        camera = default_camera()
        rgb = np.full((480, 640, 3), 48, np.uint8)
        depth = np.full((480, 640), 1.4, np.float32)
        # These disconnected patches have a large combined metric extent.
        rgb[150:270, 10:130] = (225, 35, 35)
        rgb[150:270, 510:630] = (40, 45, 225)
        rgb[180:240, 40:100] = 240
        frame = Frame(rgb, depth, camera, 1)
        before = frame.depth.copy()
        detector = Detector(self.cfg(min_depth_samples=24, plane_grid_step=8))
        regions = detector._color_regions(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
        self.assertEqual(regions[0][200, 60], 255)
        patches = detector._depth_masks(frame)
        self.assertTrue(any(mask[200, 60] for mask, _ in patches))
        self.assertTrue(any(mask[200, 550] for mask, _ in patches))
        np.testing.assert_array_equal(frame.depth, before)

    def test_wrong_size_large_face_is_rejected(self):
        frame, _ = self.close_frame(.36, size=.27)
        for limit in (3., 5.):
            result, _ = Pipeline(self.cfg(max_reprojection_px=limit)).process(frame)
            self.assertEqual(result["objects"], [])

    def test_nominal_big_cube_with_small_surface_size_change(self):
        # A 34 cm measured surface is still a nominal 35 cm old cube. At
        # close range its small metric mismatch exceeds a 3 px image budget.
        size, z = .34, .70
        box = world_box((0, 0, size/2), size, 0, (0, -z-size/2, size/2), color=(40, 45, 225))
        frame, _, _ = Renderer(default_camera()).render(
            [box], depth_noise_m=.001, floor_height_camera_m=1.
        )
        strict, _ = Pipeline(self.cfg(max_reprojection_px=3.)).process(frame)
        self.assertEqual(strict["objects"], [])
        relaxed, _ = Pipeline(self.cfg(max_reprojection_px=5.)).process(frame)
        self.assertEqual(len(relaxed["objects"]), 1)
        metrics = relaxed["objects"][0]["metrics"]
        self.assertGreater(metrics["reprojection_rms_px"], 3.)
        self.assertLessEqual(metrics["reprojection_rms_px"], 5.)
        self.assertLess(metrics["depth_median_residual_m"], .012)


if __name__ == "__main__":
    unittest.main()
