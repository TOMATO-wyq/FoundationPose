"""Physical color placement survives the cube symmetries that confuse FP."""
from itertools import permutations, product
import unittest

import numpy as np

from box4d.synthetic import Renderer, default_camera, world_box
from fp_color_orientation import ColorOrientation, placement
from kfs_roi import crop_frame


def symmetries():
    for permutation in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            rotation = np.eye(3)[:, permutation] * signs
            if np.linalg.det(rotation) > .9:
                yield rotation


class ColorOrientationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = Renderer(default_camera(320, 240))
        cls.analyzer = ColorOrientation(np.array([[-.1, -.1, 0], [.1, .1, .2]]))

    def render(self, axis=(0, 0, 1), yaw=15):
        box = world_box((0, 0, .1), .2, yaw)
        box['red_axis_local'] = axis
        frame, truth, _ = self.renderer.render([box], holes=.005, depth_noise_m=.001)
        return frame, np.array(truth[0]['T_camera_from_box'])

    def raw_pose(self, centered_pose, symmetry=None):
        pose = centered_pose.copy()
        if symmetry is not None:
            pose[:3, :3] = pose[:3, :3] @ symmetry
        pose[:3, 3] -= pose[:3, :3] @ self.analyzer.bounds.mean(0)
        return pose

    def analyze(self, frame, centered_pose, symmetry=None, up=(0, -1, 0)):
        return self.analyzer.analyze(frame.rgb, frame.depth, frame.camera.k,
                                     self.raw_pose(centered_pose, symmetry), up)

    def test_six_directions_survive_all_24_fp_cube_symmetries(self):
        cases = [((0, 0, 1), 'redup'), ((0, 0, -1), 'blueup'),
                 ((0, -1, 0), 'redfront'), ((0, 1, 0), 'bluefront'),
                 ((-1, 0, 0), 'redleft'), ((1, 0, 0), 'blueleft')]
        rotations = list(symmetries())
        self.assertEqual(len(rotations), 24)
        for axis, expected in cases:
            frame, pose = self.render(axis)
            for rotation in rotations:
                with self.subTest(expected=expected, rotation=rotation.tolist()):
                    result = self.analyze(frame, pose, rotation)
                    self.assertEqual(result['label'], expected, result)

    def test_camera_roll_uses_real_up_instead_of_screen_up(self):
        box = world_box((0, 0, .1), .2, 15)
        roll = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        box['pose'][:3, :3] = roll @ box['pose'][:3, :3]
        box['pose'][:3, 3] = [0, 0, 1.]
        box['red_axis_local'] = (0, 0, 1)
        frame, _, _ = self.renderer.render([box])
        result = self.analyze(frame, box['pose'], up=roll @ np.array([0., -1, 0]))
        self.assertEqual(result['label'], 'redup', result)

    def test_pitched_camera_preserves_up_and_flat_front_back_labels(self):
        for degrees in (-35, 35):
            angle = np.deg2rad(degrees)
            pitch = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)],
                              [0, np.sin(angle), np.cos(angle)]])
            for axis, expected in (((0, 0, 1), 'redup'), ((0, 0, -1), 'blueup'),
                                   ((0, -1, 0), 'redfront'), ((0, 1, 0), 'bluefront')):
                box = world_box((0, 0, .1), .2, 0)
                box['pose'][:3, :3] = pitch @ box['pose'][:3, :3]
                box['pose'][:3, 3] = [0, 0, 1.]
                box['red_axis_local'] = axis
                frame, _, _ = self.renderer.render([box])
                result = self.analyze(frame, box['pose'], up=pitch @ np.array([0., -1, 0]))
                self.assertEqual(result['label'], expected, (degrees, expected, result))

    def test_single_visible_flat_face_is_front_or_up_depending_on_gravity(self):
        box = dict(size_m=.2, pose=np.eye(4), red_axis_local=(0, 0, -1))
        box['pose'][2, 3] = 1.
        frame, _, _ = self.renderer.render([box])
        front = self.analyze(frame, box['pose'])
        above = self.analyze(frame, box['pose'], up=(0, 0, -1))
        self.assertEqual(front['label'], 'redfront', front)
        self.assertEqual(above['label'], 'redup', above)
        self.assertEqual(front['evidence'], 'single_visible_color')

    def test_rgbd_roi_uses_its_own_intrinsics(self):
        frame, pose = self.render()
        full = self.analyze(frame, pose)
        rgb, depth, k = crop_frame(frame.rgb, frame.depth, frame.camera.k, [80, 70, 240, 235])
        cropped = self.analyzer.analyze(rgb, depth, k, self.raw_pose(pose), [0, -1, 0])
        self.assertEqual(cropped['label'], full['label'])
        np.testing.assert_allclose(cropped['red_axis_camera'], full['red_axis_camera'])

    def test_color_reversal_is_immediate_with_unchanged_fp_pose(self):
        frame, pose = self.render()
        first = self.analyze(frame, pose)
        reverse = self.analyzer.analyze(frame.rgb[..., ::-1].copy(), frame.depth, frame.camera.k,
                                       self.raw_pose(pose), [0, -1, 0])
        self.assertEqual((first['label'], reverse['label']), ('redup', 'blueup'))

    def test_missing_depth_color_gravity_or_pose_never_reuses_a_label(self):
        frame, pose = self.render()
        raw_pose = self.raw_pose(pose)
        variants = [(frame.rgb, np.zeros_like(frame.depth), raw_pose, [0, -1, 0]),
                    (np.full_like(frame.rgb, 128), frame.depth, raw_pose, [0, -1, 0]),
                    (frame.rgb, frame.depth, raw_pose, None),
                    (frame.rgb, frame.depth, None, [0, -1, 0])]
        self.assertEqual(self.analyze(frame, pose)['label'], 'redup')
        for rgb, depth, raw_pose, up in variants:
            self.assertEqual(self.analyzer.analyze(rgb, depth, frame.camera.k, raw_pose, up)['label'], 'unknown')

    def test_diagonal_placement_is_unknown_and_front_is_not_optical_z_sign(self):
        direction = np.array([0., -1., -1.]) / np.sqrt(2)
        self.assertEqual(placement(direction, np.array([0., 0., 1.]), [0, -1, 0])[0], 'unknown')
        # A downward-looking camera can see the real UP direction along optical -Z.
        self.assertEqual(placement([0, 0, -1], np.array([0., 0., 1.]), [0, 0, -1])[0], 'redup')


if __name__ == '__main__':
    unittest.main()
