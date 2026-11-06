"""Pose, detector and association tests with analytic and rendered RGB-D."""
from copy import deepcopy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from box4d.config import Config
from box4d.detector import Face
from box4d.geometry import (Camera, order_quad, square_points, transform, transform_points, angle_difference,
                            upright_from_camera, yaw_from_pose)
from box4d.io import Frame
from box4d.pose import Pipeline, estimate_face, refine_pose
from box4d.ros_input import frame_from_message
from box4d.synthetic import Renderer, default_camera, demo_scenes, world_box
from box4d.tracking import Tracker


class AnalyticPoseTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(640, 480, [[460, 0, 320], [0, 460, 240], [0, 0, 1]], [.03, -.01, .001, 0, 0])
        rotation = cv2.Rodrigues(np.array([.02, -.18, .03]))[0] @ np.diag([1., -1., -1.])
        self.face_pose = transform(rotation, [.035, .06, 1.1])
        yy, xx = np.indices((480, 640))
        rays = self.camera.rays(np.column_stack((xx.ravel(), yy.ravel())))
        n = self.face_pose[:3, 2]
        depth = ((n @ self.face_pose[:3, 3]) / (rays @ n)).reshape(480, 640)
        self.frame = Frame(np.zeros((480, 640, 3), np.uint8), depth.astype(np.float32), self.camera, 1)
        self.face = Face(self.camera.project(transform_points(self.face_pose, square_points(.35))), "analytic", 1.)

    def test_both_solvers_recover_distorted_face(self):
        cfg = Config(up_camera=None, refine_4d=False).validate()
        estimates, _, _ = estimate_face(self.frame, self.face, cfg)
        self.assertTrue(any(e.method == "depth" for e in estimates))
        self.assertTrue(any(e.method.startswith("pnp") for e in estimates))
        center = self.face_pose[:3, 3] - self.face_pose[:3, 2] * .175
        for method in ("depth", "pnp"):
            best = min([e for e in estimates if e.method.startswith(method)], key=lambda e: e.cost)
            np.testing.assert_allclose(best.box_pose[:3, 3], center, atol=2e-5)

    def test_bad_correspondence_and_missing_depth_rejected(self):
        bad = deepcopy(self.face)
        bad.quad = bad.quad[[0, 2, 1, 3]]
        with self.assertRaises(ValueError):
            estimate_face(self.frame, bad, Config().validate())
        bad_depth = Frame(self.frame.rgb, np.zeros_like(self.frame.depth), self.camera, 2)
        with self.assertRaises(ValueError):
            estimate_face(bad_depth, self.face, Config().validate())

    def test_wrong_size_is_rejected_using_depth(self):
        with self.assertRaises(ValueError):
            estimate_face(self.frame, self.face, Config(sizes_m=(.20,)).validate())

    def test_resolution_scaling_keeps_same_corner_error_budget(self):
        for displacement in (6., 8.):
            for scale in (1, 2):
                with self.subTest(displacement=displacement, scale=scale):
                    face = deepcopy(self.face)
                    face.quad[2, 0] += displacement
                    face.quad *= scale
                    camera = self.camera.crop_resize((0, 0, 640, 480), 640*scale, 480*scale)
                    depth = cv2.resize(self.frame.depth, (camera.width, camera.height),
                                       interpolation=cv2.INTER_NEAREST)
                    frame = Frame(np.zeros((camera.height, camera.width, 3), np.uint8), depth, camera, 1)
                    cfg = Config(method='depth', refine_4d=False, reprojection_reference_width_px=640).validate()
                    if displacement == 6.:
                        found, _, _ = estimate_face(frame, face, cfg)
                        self.assertEqual(found[0].metrics['reprojection_limit_px'], 3.*scale)
                        if scale == 2:
                            fixed_pixels = Config(method='depth', refine_4d=False).validate()
                            with self.assertRaisesRegex(ValueError, 'reprojection error exceeds limit'):
                                estimate_face(frame, face, fixed_pixels)
                    else:
                        # The larger, genuinely inconsistent corner error must
                        # still be rejected at both image resolutions.
                        with self.assertRaisesRegex(ValueError, 'reprojection error exceeds limit'):
                            estimate_face(frame, face, cfg)

    def test_analytic_4d_jacobian_matches_finite_differences(self):
        from scipy.optimize import least_squares as original_solver
        cfg = Config(up_camera=[0, -1, 0]).validate()
        estimates, _, _ = estimate_face(self.frame, self.face, cfg)
        anchor = min(estimates, key=lambda e: e.cost)
        checked = []

        def verify(fun, x, **kwargs):
            analytic = kwargs["jac"](x)
            numeric = np.column_stack([(fun(x + np.eye(4)[i] * 1e-6) - fun(x - np.eye(4)[i] * 1e-6)) / 2e-6 for i in range(4)])
            np.testing.assert_allclose(analytic, numeric, atol=2e-4, rtol=2e-5)
            checked.append(True)
            return original_solver(fun, x, **kwargs)

        with patch("box4d.pose.least_squares", side_effect=verify):
            refine_pose(self.frame, anchor, [anchor], cfg)
        self.assertTrue(checked)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = Renderer(default_camera())

    def test_single_low_camera_both_sizes_and_angles(self):
        pipeline = Pipeline(Config(up_camera=[0, -1, 0]))
        for index in (0, 2, 4, 5, 6, 8, 9):
            name, boxes = demo_scenes()[index]
            frame, truth, _ = self.renderer.render(boxes, index, .001, .005)
            result, _ = pipeline.process(frame)
            self.assertEqual(len(result["objects"]), 1, name)
            error = np.linalg.norm(np.asarray(result["objects"][0]["center_camera_m"]) - truth[0]["center_camera_m"])
            self.assertLess(error, .01, name)

    def test_two_visible_faces_are_one_object(self):
        frame, _, _ = self.renderer.render(demo_scenes()[8][1], 8, .001)
        result, _ = Pipeline(Config(up_camera=[0, -1, 0])).process(frame)
        self.assertEqual(len(result["objects"]), 1)
        self.assertGreaterEqual(result["objects"][0]["face_count"], 1)

    def test_success_on_large_size_does_not_disable_small_size_fallback(self):
        cfg=Config(up_camera=[0,-1,0],color_line_fallback=True).validate()
        frame,_,_=self.renderer.render(demo_scenes()[11][1],11,.001)
        initial,edges=Pipeline(cfg).process(frame)
        self.assertEqual({o['size_m'] for o in initial['objects']},{.2,.35})
        faces={o['size_m']:Face(np.asarray(o['faces'][0]['quad']),o['faces'][0]['source'],1.)
               for o in initial['objects']}
        for sizes in ((.35,),(.35,.2)):
            with self.subTest(initial_sizes=sizes):
                pipeline=Pipeline(cfg)
                with patch.object(pipeline.detector,'propose',return_value=(
                        [faces[s] for s in sizes],dict(detector_ms=0.,proposal_count=len(sizes)),edges)), \
                        patch.object(pipeline.detector,'color_line_fallback',return_value=[faces[.2]]) as fallback:
                    result,_=pipeline.process(frame)
                self.assertEqual({o['size_m'] for o in result['objects']},{.2,.35})
                self.assertEqual(fallback.call_count,1 if len(sizes)==1 else 0)

    def test_two_same_color_and_mixed_sizes(self):
        pipeline = Pipeline(Config(up_camera=[0, -1, 0]))
        for index in (10, 11):
            frame, truth, _ = self.renderer.render(demo_scenes()[index][1], index, .001, .005)
            result, _ = pipeline.process(frame)
            self.assertEqual(len(result["objects"]), 2)
            centers = np.asarray([o["center_camera_m"] for o in result["objects"]])
            for target in truth:
                self.assertLess(np.min(np.linalg.norm(centers - target["center_camera_m"], axis=1)), .01)

    def test_unmarked_stack_inference_is_labeled(self):
        frame, truth, _ = self.renderer.render(demo_scenes()[12][1], 12, .001)
        result, _ = Pipeline(Config(up_camera=[0, -1, 0])).process(frame)
        self.assertEqual(len(result["objects"]), 2)
        self.assertTrue(all(o["evidence"] == "model_inferred" for o in result["objects"]))
        no_inference, _ = Pipeline(Config(up_camera=[0, -1, 0], allow_stack_inference=False)).process(frame)
        self.assertEqual(no_inference["objects"], [])

    def test_empty_wrong_size_and_absent_depth_no_valid_pose(self):
        pipeline = Pipeline(Config(up_camera=[0, -1, 0]))
        for index in (14, 15):
            frame, _, _ = self.renderer.render(demo_scenes()[index][1], index)
            result, _ = pipeline.process(frame)
            self.assertEqual(result["objects"], [])
        frame, _, _ = self.renderer.render(demo_scenes()[6][1], 16)
        frame.depth[:] = 0
        result, _ = pipeline.process(frame)
        self.assertEqual(result["objects"], [])

    def test_no_vertical_reference_does_not_fabricate_yaw(self):
        frame, _, _ = self.renderer.render(demo_scenes()[6][1], 17)
        result, _ = Pipeline(Config(up_camera=None)).process(frame)
        self.assertTrue(result["objects"])
        for observation in result["objects"]:
            self.assertIsNone(observation["yaw_u_rad"])
            self.assertFalse(observation["yaw_reference_ready"])

    def test_camera_tilt_requires_current_gravity_reference(self):
        from box4d.reference import stationary_up
        rotation = cv2.Rodrigues(np.array([.28, 0., 0.]))[0]
        box = world_box((0, 0, .1), .2, 15, camera_world=(0, -1.3, .65))
        box['pose'][:3, :] = rotation @ box['pose'][:3, :]
        box['red_axis_local'] = [0, 0, 1]
        frame, truth, _ = self.renderer.render([box], depth_noise_m=.001)
        kwargs = dict(analyze_colors=True, require_target_colors=True, two_color_sizes_m=(.2,), plane_grid_step=8)
        stale, _ = Pipeline(Config(up_camera=[0, -1, 0], **kwargs)).process(frame)
        self.assertEqual(stale['objects'], [])
        # A new stationary IMU measurement restores the reference without
        # loosening the upright, depth or reprojection acceptance thresholds.
        accel = np.tile(rotation @ np.array([0., -9.80665, 0.]), (100, 1))
        current_up, _ = stationary_up(accel, np.eye(3))
        result, _ = Pipeline(Config(up_camera=current_up, **kwargs)).process(frame)
        self.assertEqual(len(result['objects']), 1)
        obj = result['objects'][0]
        np.testing.assert_allclose(obj['center_camera_m'], truth[0]['center_camera_m'], atol=.01)
        expected = yaw_from_pose(np.asarray(truth[0]['T_camera_from_box']), upright_from_camera(current_up))
        self.assertLess(abs(angle_difference(obj['yaw_u_rad'], expected)), np.deg2rad(2))

    def test_white_box_does_not_require_saturation_segmentation(self):
        boxes = deepcopy(demo_scenes()[6][1])
        boxes[0]["color"] = (230, 230, 230)
        frame, truth, _ = self.renderer.render(boxes, 18, .001)
        result, _ = Pipeline(Config(up_camera=[0, -1, 0])).process(frame)
        self.assertEqual(len(result["objects"]), 1)
        self.assertLess(np.linalg.norm(np.asarray(result["objects"][0]["center_camera_m"]) - truth[0]["center_camera_m"]), .015)

    def test_roi_excludes_objects_outside_search_area(self):
        frame, _, _ = self.renderer.render(demo_scenes()[6][1], 19)
        result, _ = Pipeline(Config(up_camera=[0, -1, 0], roi_xyxy=[0, 0, 100, 100])).process(frame)
        self.assertEqual(result["objects"], [])


class AssociationTests(unittest.TestCase):
    def record(self, stamp, yaw, objects=True):
        observation = dict(center_camera_m=[0., 0., 1.], yaw_u_rad=np.deg2rad(yaw), size_m=.35,
                           yaw_reference_ready=True, valid=True)
        return dict(stamp_ns=stamp, calibration_id="test", up_camera=[0, -1, 0],
                    objects=[observation] if objects else [], rejected_faces=[])

    def test_yaw_wrap_and_missing_frame_never_republish_pose(self):
        tracker = Tracker(Config().validate())
        a = tracker.update(self.record(1_000_000_000, 89))
        b = tracker.update(self.record(1_016_666_667, 1))
        self.assertEqual(a["objects"][0]["track_id"], b["objects"][0]["track_id"])
        smooth = b["objects"][0]["yaw_u_smoothed_rad"]
        self.assertLess(abs(angle_difference(smooth, 0)), np.deg2rad(1))
        missing = tracker.update(self.record(1_033_333_334, 1, objects=False))
        self.assertEqual(missing["objects"], [])
        self.assertTrue(missing["lost_track_ids"])
        with self.assertRaises(ValueError):
            tracker.update(self.record(1_033_333_334, 1))


class RosInputContractTests(unittest.TestCase):
    def message(self):
        header = NS(stamp=NS(sec=3, nanosec=100), frame_id="color_optical")
        color = NS(header=header, width=16, height=12, array=np.zeros((12, 16, 3), np.uint8))
        depth = NS(header=deepcopy(header), encoding="16UC1", array=np.full((12, 16), 1250, np.uint16))
        info = NS(header=header, width=16, height=12, k=[12, 0, 8, 0, 12, 6, 0, 0, 1],
                  d=[.01, 0, 0, 0, 0], distortion_model="plumb_bob")
        return NS(rgb=color, depth=depth, rgb_camera_info=info)

    def test_millimeter_conversion_and_distortion_retained(self):
        bridge = NS(imgmsg_to_cv2=lambda message, _: message.array)
        frame = frame_from_message(self.message(), bridge, 1)
        np.testing.assert_allclose(frame.depth, 1.25)
        self.assertAlmostEqual(frame.camera.d[0], .01)
        self.assertEqual(frame.stamp_ns, 3_000_000_100)

    def test_mismatched_stamp_and_frame_rejected(self):
        bridge = NS(imgmsg_to_cv2=lambda message, _: message.array)
        message = self.message()
        message.depth.header.stamp.nanosec += 1
        with self.assertRaises(ValueError):
            frame_from_message(message, bridge, 1)
        message = self.message()
        message.depth.header.frame_id = "depth_optical"
        with self.assertRaises(ValueError):
            frame_from_message(message, bridge, 1)


if __name__ == "__main__":
    unittest.main()
