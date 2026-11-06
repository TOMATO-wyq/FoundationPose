"""Physical duplicate/ghost suppression and artificial ROI-boundary rejection."""
import unittest
import numpy as np

from box4d.config import Config
from box4d.detector import sample_face
from box4d.geometry import Camera, transform, rotation_z
from box4d.io import Frame
from box4d.scene import select_scene


class SceneTests(unittest.TestCase):
    def obj(self, center, quality=1., angle=0):
        pose = transform(rotation_z(angle), center)
        return dict(size_m=.2, center_camera_m=list(center), T_camera_from_box=pose.tolist(),
                    quality_score=quality, faces=[dict(quad=[[0, 0], [10, 0], [10, 10], [0, 10]])])

    def test_two_real_cubes_preferred_to_middle_ghost(self):
        lower = self.obj([0, 0, 1])
        upper = self.obj([0, 0, 1.2])
        ghost = self.obj([0, 0, 1.1])
        selected, rejected = select_scene([ghost, lower, upper], 3, [0, 0, 1])
        self.assertEqual(selected, [lower, upper])
        self.assertEqual(rejected[0]["reason"], "overlapping_volume_hypothesis")

    def test_refined_duplicate_does_not_add_a_third_object(self):
        lower = self.obj([0, 0, 1])
        upper = self.obj([0, 0, 1.2])
        duplicate = self.obj([.001, 0, 1.2], .8, np.deg2rad(12))
        selected, _ = select_scene([lower, upper, duplicate], 3, [0, 0, 1])
        self.assertEqual(selected, [lower, upper])

    def test_contact_and_small_pose_noise_do_not_remove_valid_stack(self):
        lower, upper = self.obj([0, 0, 1]), self.obj([0, 0, 1.195])
        selected, _ = select_scene([lower, upper], 3, [0, 0, 1])
        self.assertEqual(len(selected), 2)

    def test_unknown_up_still_suppresses_near_identical_centers(self):
        objects = [self.obj([0, 0, 1]), self.obj([.001, 0, 1], .8), self.obj([.3, 0, 1])]
        selected, _ = select_scene(objects, 3, None)
        self.assertEqual(len(selected), 2)

    def test_roi_cut_is_not_a_measured_face_edge(self):
        camera = Camera(64, 64, [[50, 0, 32], [0, 50, 32], [0, 0, 1]], [])
        frame = Frame(np.zeros((64, 64, 3), np.uint8), np.ones((64, 64), np.float32), camera, 1)
        quad = np.array([[10, 10], [40, 10], [40, 40], [10, 40]], float)
        with self.assertRaisesRegex(ValueError, "search ROI"):
            sample_face(frame, quad, Config(roi_xyxy=[10, 5, 50, 50]).validate())
