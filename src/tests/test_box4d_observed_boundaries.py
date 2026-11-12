"""Nominal cube sizes, inset observed faces and painted floor negatives."""

import unittest

import cv2
import numpy as np

from box4d.config import Config
from box4d.detector import Face, check_horizontal_boundary
from box4d.geometry import Plane, order_quad, square_points, transform, transform_points
from box4d.io import Frame
from box4d.pose import Pipeline, estimate_face
from box4d.synthetic import Renderer, default_camera, world_box
from box4d.temporal import TemporalPipeline


class ObservedBoundaryTests(unittest.TestCase):
    def cfg(self, **overrides):
        options = dict(sizes_m=(.35,), up_camera=[0, -1, 0],
                       fit_observed_face_size=True, max_reprojection_px=5.,
                       temporal_recovery=True, analyze_colors=True,
                       require_target_colors=True, async_detection=False)
        options.update(overrides)
        return Config(**options).validate()

    def inset_face(self, z, sequence=0):
        size = .33
        box = world_box((0, 0, size / 2), size, 0,
                        (0, -z-size/2, size/2), color=(40, 45, 225))
        frame, _, _ = Renderer(default_camera()).render(
            [box], sequence, depth_noise_m=.001, floor_height_camera_m=1.)
        for x in np.linspace(-.12, .12, 5):
            for y in np.linspace(-.12, .12, 5):
                p = frame.camera.project([[x, y, z]])[0]
                cv2.rectangle(frame.rgb, tuple(np.rint(p-3).astype(int)),
                              tuple(np.rint(p+3).astype(int)), (230, 230, 230), -1)
        return frame

    def test_inset_face_keeps_depth_and_nominal_size_at_close_range(self):
        frame = self.inset_face(.7)
        old, _ = Pipeline(self.cfg(fit_observed_face_size=False)).process(frame)
        self.assertEqual(old['objects'], [])
        result, _ = Pipeline(self.cfg()).process(frame)
        self.assertEqual(len(result['objects']), 1)
        obj = result['objects'][0]
        self.assertEqual(obj['size_m'], .35)
        self.assertAlmostEqual(obj['metrics']['observed_face_size_m'], .33, delta=.004)
        self.assertLess(obj['metrics']['reprojection_rms_px'], 5.)
        np.testing.assert_allclose(obj['center_camera_m'], [0, 0, .875], atol=.006)

    def test_measured_extent_survives_tracking_and_reseeding(self):
        pipe = TemporalPipeline(self.cfg(redetect_interval_s=10., track_reseed_interval_s=.04))
        for i, z in enumerate((.7, .69, .68, .67, .66, .65)):
            result, _ = pipe.process(self.inset_face(z, i))
            self.assertEqual(len(result['objects']), 1, i)
            obj = result['objects'][0]
            self.assertEqual(obj['size_m'], .35)
            np.testing.assert_allclose(obj['center_camera_m'], [0, 0, z+.175], atol=.008)
            self.assertAlmostEqual(obj['metrics']['observed_face_size_m'], .33, delta=.004)
        self.assertEqual(obj['source'], 'current_rgbd_flow')

    def floor_patch(self, isolated=False, inverted=False):
        camera = default_camera()
        yy, xx = np.indices((camera.height, camera.width))
        rays = camera.rays(np.column_stack((xx.ravel(), yy.ravel())))
        normal = np.array([0., -1., 0.])
        pose = transform(np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]), [0., .6, 1.7])
        quad = order_quad(camera.project(transform_points(pose, square_points(.35))))
        depth = np.zeros(len(rays), np.float32)
        valid = rays[:, 1] > .01
        depth[valid] = .6 / rays[valid, 1]
        depth = depth.reshape(camera.height, camera.width)
        mask = np.zeros_like(depth, np.uint8)
        cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 1)
        rgb = np.full((*depth.shape, 3), 80, np.uint8)
        rgb[mask > 0] = (225, 35, 35)
        if isolated:
            depth[mask == 0] = 0
        if inverted:
            rgb = np.ascontiguousarray(rgb[::-1, ::-1])
            depth = np.ascontiguousarray(depth[::-1, ::-1])
            quad = order_quad(np.array([camera.width-1, camera.height-1]) - quad)
            normal = -normal
        return Frame(rgb, depth, camera, 1), Face(quad, 'observed_red_patch', 1.), normal

    def test_red_floor_square_is_rejected_in_both_camera_orientations(self):
        for inverted in (False, True):
            frame, face, normal = self.floor_patch(inverted=inverted)
            cfg = self.cfg(up_camera=normal.tolist())
            with self.assertRaisesRegex(ValueError, 'surrounding depth plane'):
                estimate_face(frame, face, cfg)

    def test_bounded_horizontal_top_still_passes(self):
        frame, face, normal = self.floor_patch(isolated=True)
        estimates, _, _ = estimate_face(frame, face, self.cfg())
        self.assertTrue(estimates)
        plane = Plane(normal, .6, 0., 1.)
        self.assertEqual(check_horizontal_boundary(frame, face.quad, plane, self.cfg())['continuing_edges'], 0)

    def test_extent_fit_cannot_accept_wrong_size_or_non_square_geometry(self):
        frame = self.inset_face(.7)
        wrong, _ = Pipeline(self.cfg(sizes_m=(.2,))).process(frame)
        self.assertEqual(wrong['objects'], [])
        # All edge lengths individually fit the nominal tolerance, but this
        # sheared quadrilateral cannot fit one rigid square within 5 pixels.
        quad = frame.camera.project([[-.165, -.165, .7], [.165, -.165, .7],
                                     [.195, .165, .7], [-.135, .165, .7]])
        frame.depth[:] = .7
        with self.assertRaisesRegex(ValueError, 'reprojection'):
            estimate_face(frame, Face(quad, 'sheared_patch', 1.), self.cfg(method='depth'))


if __name__ == '__main__':
    unittest.main()
