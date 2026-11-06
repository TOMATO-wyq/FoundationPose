"""短窗口、当前 RGB-D 验收和补点纠偏必须组成自洽的跟踪链路。"""

from copy import deepcopy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from box4d.config import Config
from box4d.direction import copy_direction_window
from box4d.edges import rgb_depth_edges
from box4d.geometry import Plane, transform_points
from box4d.synthetic import Renderer, default_camera, world_box
from box4d.temporal import TemporalPipeline, TrackingPoseRejected


def camera_rotation(degrees, axis=(1.0, 0.0, 0.0)):
    axis = np.asarray(axis, dtype=np.float64)
    return cv2.Rodrigues(np.deg2rad(degrees) * axis / np.linalg.norm(axis))[0]


class DirectionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = Renderer(default_camera())
        cls.cfg = Config(
            up_camera=[0, -1, 0], analyze_colors=True, two_color_sizes_m=(0.2,),
            require_target_colors=True, temporal_recovery=True, plane_grid_step=8,
            track_joint_on_ambiguity=False,
        )
        cls.source, _, _ = cls.frame(0)
        pipeline = TemporalPipeline(deepcopy(cls.cfg))
        result, _ = pipeline.process(cls.source)
        if len(result["objects"]) != 1 or not pipeline.states:
            raise AssertionError("真实合成 RGB-D 必须能够初始化方块轨迹")
        cls.seed = deepcopy(pipeline.states[0])

    @classmethod
    def frame(cls, sequence, yaw=15.0, x=0.0):
        box = world_box((x, 0.0, 0.1), 0.2, yaw)
        box["red_axis_local"] = [0, 0, 1]
        return cls.renderer.render([box], sequence, depth_noise_m=0.001)

    def pipeline(self, **changes):
        cfg = deepcopy(self.cfg)
        for name, value in changes.items():
            setattr(cfg, name, value)
        return TemporalPipeline(cfg)

    @staticmethod
    def inputs(pipeline, frame):
        gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
        edges = rgb_depth_edges(frame, pipeline.cfg, gray=gray)[2]
        distance = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
        return gray, distance

    @staticmethod
    def with_pose(state, pose):
        """只构造已存在的跟踪偏差；模型/像点仍来自真实源帧。"""
        state = deepcopy(state)
        state["pose"] = pose.copy()
        obj = state["obj"]
        obj.update(source="current_rgbd_flow", T_camera_from_box=pose.tolist(),
                   center_camera_m=pose[:3, 3].tolist())
        obj["faces"] = [dict(obj["faces"][0], normal_camera=(pose[:3, :3] @ state["normal_model"]).tolist())]
        obj["face_count"] = 1
        return state

    @staticmethod
    def tilted_pose(state, degrees):
        pose = state["pose"].copy()
        face_center = transform_points(pose, state["quad_model"]).mean(axis=0)
        pose[:3, :3] = camera_rotation(degrees) @ pose[:3, :3]
        pose[:3, 3] = face_center - pose[:3, :3] @ state["quad_model"].mean(axis=0)
        return pose

    def assert_state_object_model_agree(self, frame, state):
        obj = state["obj"]
        np.testing.assert_allclose(obj["T_camera_from_box"], state["pose"], atol=1e-12)
        np.testing.assert_allclose(obj["center_camera_m"], state["pose"][:3, 3], atol=1e-12)
        np.testing.assert_allclose(
            obj["faces"][0]["normal_camera"], state["pose"][:3, :3] @ state["normal_model"], atol=1e-12
        )
        np.testing.assert_allclose(
            obj["faces"][0]["quad"], frame.camera.project(transform_points(state["pose"], state["quad_model"])),
            atol=1e-8,
        )
        np.testing.assert_allclose(
            frame.camera.project(transform_points(state["pose"], state["model"])),
            state["points"].reshape(-1, 2), atol=1e-4,
        )

    def test_real_motion_updates_window_without_aliasing_previous_frames(self):
        pipeline = self.pipeline()
        state = deepcopy(self.seed)
        before = deepcopy(state["direction_window"])
        snapshots = []
        for sequence in range(1, 7):
            frame, truth, _ = self.frame(sequence, yaw=15.0 + sequence * 0.5, x=sequence * 0.001)
            gray, distance = self.inputs(pipeline, frame)
            previous = state
            obj, state = pipeline._advance(frame, previous, gray, distance)
            self.assertEqual(state["stamp"], frame.stamp_ns)
            self.assertFalse(np.shares_memory(
                state["direction_window"]["angular_velocity"], previous["direction_window"]["angular_velocity"]
            ))
            np.testing.assert_allclose(obj["center_camera_m"], truth[0]["center_camera_m"], atol=0.012)
            snapshots.append(deepcopy(state["direction_window"]["angular_velocity"]))
        self.assertGreater(max(np.linalg.norm(value) for value in snapshots), np.deg2rad(2.0))
        np.testing.assert_array_equal(self.seed["direction_window"]["angular_velocity"], before["angular_velocity"])

    def test_alternating_observation_cost_keeps_stable_valid_pnp_branch(self):
        pipeline = self.pipeline()
        positive = self.tilted_pose(self.seed, 4.0)
        negative = self.tilted_pose(self.seed, -4.0)
        state = self.with_pose(self.seed, positive)
        real_evaluate = pipeline._evaluate_tracking_pose
        for sequence in range(1, 7):
            frame, _, _ = self.frame(sequence)
            gray, distance = self.inputs(pipeline, frame)
            details = {}

            def controlled_cost(*args, **kwargs):
                result = real_evaluate(*args, **kwargs)
                # 两个解仍要通过全部真实图像/深度/边界门槛。这里只独立控制
                # 已验收解的小幅观测代价差，模拟像素噪声交替占优。
                positive_branch = result["trial"]["solver"] == "iterative"
                cost = 0.01 if positive_branch == bool(sequence % 2) else 0.0
                result["base_cost"] = cost
                result["trial"].update(base_cost=cost, selection_cost=cost,
                                       cost_components=dict(image=cost, depth=0.0, normal=0.0, yaw=0.0))
                return result

            with patch("box4d.temporal.cv2.solvePnPGeneric", return_value=(
                1, (cv2.Rodrigues(negative[:3, :3])[0],), (negative[:3, 3].reshape(3, 1),), None
            )), patch("box4d.temporal.cv2.solvePnP", return_value=(
                True, cv2.Rodrigues(positive[:3, :3])[0], positive[:3, 3].reshape(3, 1)
            )), patch.object(pipeline, "_evaluate_tracking_pose", side_effect=controlled_cost):
                obj, state = pipeline._advance(frame, state, gray, distance, diagnostics=details)
            accepted = [trial for trial in details["pose_trials"] if trial["accepted"]]
            self.assertEqual(len(accepted), 2, details)
            self.assertEqual(details["chosen_pose_solver"], "iterative")
            np.testing.assert_allclose(np.asarray(obj["T_camera_from_box"])[:3, :3], positive[:3, :3], atol=1e-12)
            self.assertEqual(sum(trial["selected"] for trial in accepted), 1)

    def test_only_valid_reverse_pnp_candidate_is_published_immediately(self):
        pipeline = self.pipeline()
        positive = self.tilted_pose(self.seed, 4.0)
        negative = self.tilted_pose(self.seed, -4.0)
        state = self.with_pose(self.seed, positive)
        state["direction_window"]["angular_velocity"][:] = np.deg2rad([60.0, 0.0, 0.0])
        frame, _, _ = self.frame(1)
        gray, distance = self.inputs(pipeline, frame)
        details = {}
        with patch("box4d.temporal.cv2.solvePnPGeneric", return_value=(0, (), (), None)), patch(
            "box4d.temporal.cv2.solvePnP", return_value=(
                True, cv2.Rodrigues(negative[:3, :3])[0], negative[:3, 3].reshape(3, 1)
            )
        ):
            obj, advanced = pipeline._advance(frame, state, gray, distance, diagnostics=details)
        self.assertEqual(details["direction"]["decision_reason"], "only_candidate")
        np.testing.assert_allclose(np.asarray(obj["T_camera_from_box"])[:3, :3], negative[:3, :3], atol=1e-12)
        np.testing.assert_allclose(advanced["direction_window"]["angular_velocity"], 0.0)

    def test_real_motion_can_reverse_without_waiting_for_escape_votes(self):
        pipeline = self.pipeline()
        state = deepcopy(self.seed)
        observed_yaws = []
        for sequence, yaw in enumerate((18.0, 21.0, 24.0, 21.0, 18.0, 15.0), start=1):
            frame, truth, _ = self.frame(sequence, yaw=yaw)
            gray, distance = self.inputs(pipeline, frame)
            obj, state = pipeline._advance(frame, state, gray, distance)
            observed_yaws.append(obj["yaw_u_rad"])
            np.testing.assert_allclose(obj["center_camera_m"], truth[0]["center_camera_m"], atol=0.012)
            self.assertNotEqual(obj["metrics"]["direction_decision_reason"], "escape_pending")
        self.assertGreater(observed_yaws[2], observed_yaws[0])
        # 真正反向的第一帧就能反向，不先保持三帧旧姿态。
        self.assertLess(observed_yaws[3], observed_yaws[2])
        self.assertLess(observed_yaws[-1], observed_yaws[3])

    def test_reseed_large_measured_rotation_is_rejected_without_mutating_state(self):
        pipeline = self.pipeline()
        state = self.with_pose(self.seed, self.tilted_pose(self.seed, 20.0))
        before = deepcopy(state)
        gray, distance = self.inputs(pipeline, self.source)
        details = {}
        seeded = pipeline._seed(self.source, state["obj"], gray, previous_state=state,
                                distance=distance, diagnostics=details)
        self.assertIsNone(seeded)
        self.assertFalse(details["accepted"])
        self.assertTrue(any("reseed_rotation_jump" in trial["failed_checks"] for trial in details["pose_trials"]))
        np.testing.assert_array_equal(state["pose"], before["pose"])
        np.testing.assert_array_equal(state["model"], before["model"])
        np.testing.assert_array_equal(state["points"], before["points"])
        self.assertEqual(state["obj"], before["obj"])

    def test_reseed_uses_measured_plane_and_updates_pose_object_and_feature_model_together(self):
        pipeline = self.pipeline()
        state = self.with_pose(self.seed, self.tilted_pose(self.seed, 4.0))
        state["direction_window"]["angular_velocity"][:] = [0.2, 0.1, 0.0]
        gray, distance = self.inputs(pipeline, self.source)
        details = {}
        seeded = pipeline._seed(self.source, state["obj"], gray, previous_state=state,
                                distance=distance, diagnostics=details)
        self.assertIsNotNone(seeded, details)
        self.assertTrue(details["accepted"])
        self.assertEqual(seeded["obj"]["metrics"]["pose_solver"], "measured_plane_reseed")
        self.assertGreater(details["pose_trials"][0]["reseed_rotation_deg"], 2.0)
        self.assert_state_object_model_agree(self.source, seeded)
        np.testing.assert_allclose(seeded["direction_window"]["angular_velocity"], [0.2, 0.1, 0.0])
        self.assertFalse(np.shares_memory(
            seeded["direction_window"]["angular_velocity"], state["direction_window"]["angular_velocity"]
        ))
        # 这些新特征确实来自当前实测面，不是按旧的错误法线造出的平面。
        _, _, measured = pipeline._sample_tracking_plane(self.source, np.asarray(state["obj"]["faces"][0]["quad"]))
        samples = transform_points(seeded["pose"], seeded["model"])
        np.testing.assert_allclose(samples @ measured.normal + measured.d, 0.0, atol=1e-8)

    def test_invalid_raw_reseed_candidate_does_not_block_current_candidate_or_new_quality(self):
        pipeline = self.pipeline()
        state = self.with_pose(self.seed, self.seed["pose"])
        state["direction_window"]["angular_velocity"][:] = [0.2, 0.1, 0.0]
        state["obj"]["metrics"].update(plane_rmse_m=0.09, plane_support=0.1, depth_fraction=0.2)
        gray, distance = self.inputs(pipeline, self.source)
        real_evaluate = pipeline._evaluate_tracking_pose
        real_sample = pipeline._sample_tracking_plane

        def measured_quality(frame, quad, diagnostics=None):
            samples, _, plane = real_sample(frame, quad, diagnostics)
            if diagnostics is not None:
                diagnostics["depth_sampling_inset_fraction"] = 0.07
            return samples, 0.83, Plane(plane.normal, plane.d, 0.0022, 0.91)

        def invalid_raw(*args, **kwargs):
            if args[-1]["solver"] == "measured_plane_reseed":
                # 隔离非旋转 guard 的失败；current 仍经过完整真实 RGB-D 验收。
                raise TrackingPoseRejected("current_pose_residuals", "feature_reprojection")
            return real_evaluate(*args, **kwargs)

        details = {}
        with patch.object(pipeline, "_sample_tracking_plane", side_effect=measured_quality), patch.object(
            pipeline, "_evaluate_tracking_pose", side_effect=invalid_raw
        ):
            seeded = pipeline._seed(self.source, state["obj"], gray, previous_state=state,
                                    distance=distance, diagnostics=details)
        self.assertIsNotNone(seeded, details)
        self.assertTrue(details["accepted"])
        raw, current = details["pose_trials"]
        self.assertFalse(raw["accepted"])
        self.assertIn("feature_reprojection", raw["failed_checks"])
        self.assertTrue(current["accepted"])
        self.assertTrue(current["selected"])
        metrics = seeded["obj"]["metrics"]
        self.assertEqual(metrics["reseed_geometry_source"], "current_pose")
        self.assertEqual(metrics["plane_rmse_m"], 0.0022)
        self.assertEqual(metrics["plane_support"], 0.91)
        self.assertEqual(metrics["depth_fraction"], 0.83)
        self.assertEqual(metrics["depth_sampling_inset_fraction"], 0.07)
        self.assertEqual(metrics["tracked_features"], len(seeded["points"]))
        self.assert_state_object_model_agree(self.source, seeded)
        np.testing.assert_allclose(seeded["direction_window"]["angular_velocity"], [0.2, 0.1, 0.0])

    def test_raw_reselection_replaces_advance_window_with_one_final_pose_update(self):
        pipeline = self.pipeline()
        initial = deepcopy(self.seed)
        initial["direction_window"]["angular_velocity"][:] = [0.1, 0.05, 0.0]
        frame, _, _ = self.frame(1, yaw=15.5)
        gray, distance = self.inputs(pipeline, frame)
        _, advanced = pipeline._advance(frame, initial, gray, distance)
        state = self.with_pose(advanced, self.tilted_pose(advanced, 4.0))
        # 若错误沿用 advance 的窗口或再更新一次，这个明确不同的值会暴露问题。
        state["direction_window"]["angular_velocity"][:] = [9.0, 8.0, 7.0]
        details = {}
        seeded = pipeline._seed(frame, state["obj"], gray, previous_state=state,
                                distance=distance, diagnostics=details)
        self.assertIsNotNone(seeded, details)
        self.assertEqual(seeded["obj"]["metrics"]["reseed_geometry_source"], "measured_plane")
        dt = (frame.stamp_ns - state["previous_stamp_ns"]) / 1e9
        before = state["direction_window_before"]["angular_velocity"]
        delta = cv2.Rodrigues(seeded["pose"][:3, :3] @ state["previous_pose"][:3, :3].T)[0].ravel()
        alpha = 1.0 - np.exp(-dt / pipeline.cfg.track_direction_window_tau_s)
        deadzone = min(1.0, 5.0 * dt)
        conflict = np.rad2deg(np.linalg.norm(delta - before * dt)) > 5.0
        expected = before * (1.0 - alpha) if (
            np.rad2deg(np.linalg.norm(delta)) <= deadzone or conflict
        ) else before + alpha * (delta / dt - before)
        np.testing.assert_allclose(seeded["direction_window"]["angular_velocity"], expected, atol=1e-10)
        self.assert_state_object_model_agree(frame, seeded)

    def test_reseed_pending_escape_vote_is_carried_to_the_next_frame(self):
        pipeline = self.pipeline()
        state = self.with_pose(self.seed, self.tilted_pose(self.seed, -9.0))
        state["previous_pose"] = state["pose"].copy()
        state["previous_stamp_ns"] = self.source.stamp_ns - 16_666_667
        state["direction_window_before"] = copy_direction_window()
        state["direction_window_before"]["escape_count"] = 1
        state["direction_window_before"]["escape_axis"][:] = [1.0, 0.0, 0.0]
        gray, distance = self.inputs(pipeline, self.source)
        real_evaluate = pipeline._evaluate_tracking_pose

        def controlled_observation(*args, **kwargs):
            result = real_evaluate(*args, **kwargs)
            cost = 0.0 if result["trial"]["solver"] == "measured_plane_reseed" else 0.31
            result["base_cost"] = cost
            result["trial"].update(base_cost=cost, selection_cost=cost,
                                   cost_components=dict(image=cost, depth=0.0, normal=0.0, yaw=0.0))
            return result

        details = {}
        with patch.object(pipeline, "_evaluate_tracking_pose", side_effect=controlled_observation):
            seeded = pipeline._seed(self.source, state["obj"], gray, previous_state=state,
                                    distance=distance, diagnostics=details)
        self.assertIsNotNone(seeded, details)
        self.assertEqual(details["direction"]["decision_reason"], "escape_pending")
        self.assertEqual(seeded["obj"]["metrics"]["reseed_geometry_source"], "current_pose")
        self.assertEqual(seeded["direction_window"]["escape_count"], 2)
        self.assertEqual(state["direction_window"]["escape_count"], 0)

    def test_real_pitch_motion_is_measured_and_can_reverse_within_upright_limits(self):
        pipeline = self.pipeline(track_joint_on_ambiguity=True)
        state = deepcopy(self.seed)
        estimates = []
        target_pitch = (3.0, 6.0, 3.0, 0.0)
        for sequence, pitch in enumerate(target_pitch, start=1):
            box = world_box((0, 0, 0.1), 0.2, 15.0)
            box["pose"][:3, :3] = camera_rotation(pitch) @ box["pose"][:3, :3]
            box["red_axis_local"] = [0, 0, 1]
            frame, truth, _ = self.renderer.render([box], sequence, depth_noise_m=0.001)
            gray, distance = self.inputs(pipeline, frame)
            obj, state = pipeline._advance(frame, state, gray, distance)
            up = state["pose"][:3, 2]
            estimates.append(float(np.rad2deg(np.arctan2(-up[2], -up[1]))))
            np.testing.assert_allclose(obj["center_camera_m"], truth[0]["center_camera_m"], atol=0.015)
            self.assertLess(abs(estimates[-1] - pitch), 3.5)
        self.assertGreater(estimates[1], 2.0)
        self.assertGreater(estimates[1], estimates[0] + 0.5)
        self.assertLess(estimates[2], estimates[1])
        self.assertLess(estimates[-1], estimates[1] - 1.0)

    def test_process_reseed_preserves_advanced_window_and_publishes_same_pose(self):
        pipeline = self.pipeline(track_min_reseed_features=999, track_reseed_interval_s=0.001)
        pipeline.process(self.source)
        pipeline.states = [deepcopy(self.seed)]
        pipeline.states[0]["direction_window"]["angular_velocity"][:] = [0.2, 0.1, 0.0]
        advanced = []
        real_advance = pipeline._advance

        def record_advance(*args, **kwargs):
            result = real_advance(*args, **kwargs)
            advanced.append(deepcopy(result[1]))
            return result

        frame, _, _ = self.frame(1, yaw=15.2, x=0.001)
        with patch.object(pipeline, "_advance", side_effect=record_advance), patch.object(
            pipeline, "_seed", wraps=pipeline._seed
        ) as seed:
            result, _ = pipeline.process(frame)
        seed.assert_called_once()
        self.assertEqual(len(result["objects"]), 1, result["temporal"])
        self.assertTrue(result["temporal"]["reseed_checks"][0]["accepted"])
        state = pipeline.states[0]
        np.testing.assert_allclose(
            state["direction_window"]["angular_velocity"], advanced[0]["direction_window"]["angular_velocity"]
        )
        self.assertFalse(np.shares_memory(
            state["direction_window"]["angular_velocity"], advanced[0]["direction_window"]["angular_velocity"]
        ))
        np.testing.assert_allclose(result["objects"][0]["T_camera_from_box"], state["pose"], atol=1e-12)
        self.assert_state_object_model_agree(frame, state)
        self.assertEqual(result["objects"][0]["metrics"]["tracking_check_group"], "reseed_checks")

    def test_failed_process_reseed_keeps_current_valid_advance_instead_of_old_frame(self):
        pipeline = self.pipeline(track_min_reseed_features=999, track_reseed_interval_s=0.001)
        pipeline.process(self.source)
        pipeline.states = [deepcopy(self.seed)]
        advanced = []
        real_advance = pipeline._advance

        def record_advance(*args, **kwargs):
            result = real_advance(*args, **kwargs)
            advanced.append(deepcopy(result[1]))
            return result

        def reject_reseed(*args, **kwargs):
            # 大朝向变化的真实拒绝在独立测试覆盖；此处隔离 process 的失败分支。
            kwargs["diagnostics"].update(accepted=False, face_rejections=["reseed_rotation_jump"])
            return None

        frame, _, _ = self.frame(1, yaw=15.2, x=0.001)
        with patch.object(pipeline, "_advance", side_effect=record_advance), patch.object(
            pipeline, "_seed", side_effect=reject_reseed
        ):
            result, _ = pipeline.process(frame)
        self.assertEqual(len(result["objects"]), 1, result["temporal"])
        self.assertFalse(result["temporal"]["reseed_checks"][0]["accepted"])
        current = pipeline.states[0]
        self.assertEqual(current["stamp"], frame.stamp_ns)
        self.assertEqual(current["last_reseed"], self.source.stamp_ns)
        np.testing.assert_array_equal(current["pose"], advanced[0]["pose"])
        np.testing.assert_array_equal(current["model"], advanced[0]["model"])
        np.testing.assert_array_equal(current["direction_window"]["angular_velocity"],
                                      advanced[0]["direction_window"]["angular_velocity"])
        np.testing.assert_array_equal(result["objects"][0]["T_camera_from_box"], advanced[0]["pose"])

    def test_new_seed_and_real_face_switch_start_with_zero_direction_history(self):
        pipeline = self.pipeline()
        gray, _ = self.inputs(pipeline, self.source)
        fresh = pipeline._seed(self.source, deepcopy(self.seed["obj"]), gray)
        self.assertIsNotNone(fresh)
        np.testing.assert_allclose(fresh["direction_window"]["angular_velocity"], 0.0)
        first, _, _ = self.frame(0, yaw=30.0)
        result, _ = pipeline.process(first)
        obj = deepcopy(result["objects"][0])
        sides = [face for face in obj["faces"] if abs(np.asarray(face["normal_camera"]) @ pipeline.cfg.up_camera) < 0.3]
        self.assertGreaterEqual(len(sides), 2)
        obj["faces"] = [min(sides, key=lambda face: cv2.contourArea(np.asarray(face["quad"], np.float32)))]
        obj["face_count"] = 1
        gray, _ = self.inputs(pipeline, first)
        narrow = pipeline._seed(first, obj, gray)
        self.assertIsNotNone(narrow)
        narrow["direction_window"] = copy_direction_window()
        narrow["direction_window"]["angular_velocity"][:] = [0.2, 0.1, 0.0]
        pipeline.states = [narrow]
        pipeline.last_detection = None
        current, _, _ = self.frame(1, yaw=30.0, x=0.002)
        switched, _ = pipeline.process(current)
        self.assertEqual(len(switched["temporal"]["face_switches"]), 1, switched["temporal"])
        np.testing.assert_allclose(pipeline.states[0]["direction_window"]["angular_velocity"], 0.0)
        self.assertEqual(pipeline.states[0]["direction_window"]["escape_count"], 0)


if __name__ == "__main__":
    unittest.main()
