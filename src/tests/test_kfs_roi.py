import socket
import threading
import unittest

import numpy as np

from kfs_roi import crop_frame, project_roi
from live_rgbd_transport import receive_frame, send_frame, recv_exact


class ROITests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((80, 100, 3), np.uint8)
        self.rgb[20:60, 30:70] = [10, 50, 100]
        self.depth = np.ones((80, 100), np.float32)
        self.k = np.array([[100., 0, 50], [0, 100., 40], [0, 0, 1]])

    def test_projection_coordinates_shift_only(self):
        roi = [20, 10, 80, 70]
        rgb, depth, k = crop_frame(self.rgb, self.depth, self.k, roi)
        xyz = np.array([.1, .2, 1])
        uv = self.k @ xyz
        cropped_uv = k @ xyz
        np.testing.assert_allclose(cropped_uv[:2], uv[:2]-[20, 10])
        np.testing.assert_array_equal(k[:2, :2], self.k[:2, :2])
        self.assertEqual(depth.shape, (60, 60))
        self.assertTrue(rgb.flags.c_contiguous)

    def test_roi_follows_pose_with_margin_and_clips(self):
        vertices = np.array([[-.1, -.1, 0], [.1, .1, 0]])
        pose = np.eye(4)
        pose[2, 3] = 1
        first = project_roi(vertices, pose, self.k, self.depth.shape)
        pose[0, 3] = .1
        second = project_roi(vertices, pose, self.k, self.depth.shape)
        self.assertEqual(second[0]-first[0], 10)
        pose[0, 3] = 5
        self.assertIsNone(project_roi(vertices, pose, self.k, self.depth.shape))

    def test_transport_preserves_same_frame_preview(self):
        a, b = socket.socketpair()
        a.settimeout(2)
        b.settimeout(2)
        roi = [20, 10, 80, 70]
        frame = (self.rgb, self.depth, self.k, 123., 'camera', 5, {})
        def server():
            import json, struct
            self.assertEqual(recv_exact(b, 1), b'R')
            size = struct.unpack('!I', recv_exact(b, 4))[0]
            request = json.loads(recv_exact(b, size))
            send_frame(b, frame, roi=request)
        thread = threading.Thread(target=server)
        thread.start()
        try:
            rgb, depth, k, meta = receive_frame(a, roi)
            self.assertEqual(meta['sequence'], 5)
            self.assertEqual(meta['roi_xyxy'], roi)
            np.testing.assert_array_equal(meta['preview_rgb'], self.rgb)
            np.testing.assert_array_equal(depth, self.depth[10:70, 20:80])
            np.testing.assert_allclose(k[0, 2], 30)
        finally:
            a.close()
            thread.join(timeout=2)
            b.close()
