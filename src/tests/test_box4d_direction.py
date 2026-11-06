"""方向趋势只辅助分支选择，不能平滑真运动或拒绝原有效观测。"""

from copy import deepcopy
import json
import unittest

import cv2
import numpy as np

from box4d.direction import copy_direction_window, select_direction_candidate


def rotation(degrees, axis=(1.0, 0.0, 0.0)):
    axis = np.asarray(axis, dtype=np.float64)
    return cv2.Rodrigues(np.deg2rad(degrees) * axis / np.linalg.norm(axis))[0]


def candidate(rot, cost):
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = [0.1, 0.2, 1.0]
    return {"pose": pose, "base_cost": cost}


class DirectionWindowTests(unittest.TestCase):
    def test_stationary_mirror_solutions_do_not_alternate_with_pixel_noise(self):
        previous = rotation(10)
        window = None
        for index in range(60):
            # 镜像分支的细小重投影优势交替，当前可信分支保持不变。
            costs = (0.02, 0.0) if index % 2 else (0.0, 0.02)
            candidates = [candidate(rotation(10), costs[0]), candidate(rotation(-10), costs[1])]
            selected, window, diagnostic = select_direction_candidate(candidates, previous, window, 0.02)
            self.assertEqual(selected, 0)
            np.testing.assert_allclose(candidates[selected]["pose"][:3, :3], previous)
            np.testing.assert_allclose(window["angular_velocity"], 0.0)
            self.assertEqual(diagnostic["escape_count"], 0)

    def test_static_deadzone_decays_velocity_instead_of_preserving_momentum(self):
        window = copy_direction_window()
        window["angular_velocity"][:] = np.deg2rad([0.0, 30.0, 0.0])
        previous = np.eye(3)
        for _ in range(20):
            # 0.02°/帧低于静止死区，不能累积成运动。
            observed = rotation(0.02) @ previous
            _, window, _ = select_direction_candidate(
                [candidate(observed, 0.0), candidate(observed, 0.1)], previous, window, 0.02
            )
            previous = observed
        np.testing.assert_allclose(
            window["angular_velocity"], np.deg2rad([0.0, 30.0, 0.0]) * np.exp(-0.4 / 0.2), atol=1e-12
        )

    def test_variable_frame_times_follow_real_motion_without_filtering_output(self):
        previous = np.eye(3)
        window = None
        duration = 0.0
        speed_deg_s = 35.0
        for dt in [0.01, 0.04, 0.02, 0.03] * 10:
            observed = rotation(speed_deg_s * dt, axis=(0, 1, 0)) @ previous
            candidates = [candidate(observed, 0.0), candidate(previous, 0.3)]
            selected, window, _ = select_direction_candidate(candidates, previous, window, dt)
            self.assertEqual(selected, 0)
            # 发布候选原值，而非历史与当前的插值。
            np.testing.assert_array_equal(candidates[selected]["pose"][:3, :3], observed)
            previous = observed
            duration += dt
        expected = np.deg2rad(speed_deg_s) * (1.0 - np.exp(-duration / 0.2))
        np.testing.assert_allclose(window["angular_velocity"], [0, expected, 0], atol=1e-10)

    def test_strong_current_observation_allows_immediate_true_reversal(self):
        window = copy_direction_window()
        window["angular_velocity"][:] = np.deg2rad([60.0, 0.0, 0.0])
        candidates = [candidate(rotation(-10), 0.0), candidate(rotation(1.2), 1.0)]
        selected, updated, diagnostic = select_direction_candidate(candidates, np.eye(3), window, 0.02)
        self.assertEqual(selected, 0)
        self.assertEqual(diagnostic["decision_reason"], "observation_wins")
        self.assertTrue(diagnostic["candidates"][0]["is_reverse"])
        np.testing.assert_allclose(updated["angular_velocity"], 0.0)

    def test_sustained_observation_advantage_escapes_within_three_frames(self):
        window = None
        previous = np.eye(3)
        candidates = [candidate(rotation(-10), 0.0), candidate(previous, 0.16)]
        choices = []
        for frame in range(3):
            selected, window, diagnostic = select_direction_candidate(candidates, previous, window, 0.02)
            choices.append(selected)
            if frame < 2:
                self.assertEqual(diagnostic["decision_reason"], "escape_pending")
                self.assertEqual(diagnostic["escape_count"], frame + 1)
        self.assertEqual(choices, [1, 1, 0])
        self.assertEqual(diagnostic["decision_reason"], "escape_evidence")
        # 原惩罚仍可审查；实际选择成本明确反映本帧逃生规则。
        self.assertGreater(diagnostic["candidates"][0]["direction_prior_cost"], 0.16)
        self.assertEqual(diagnostic["candidates"][0]["applied_direction_cost"], 0.0)
        self.assertEqual(diagnostic["candidates"][0]["selection_cost"], 0.0)
        self.assertEqual(window["escape_count"], 0)

    def test_alternating_noise_cannot_accumulate_escape_votes(self):
        window = None
        for index in range(12):
            candidates = [candidate(rotation(10 if index % 2 else -10), 0.0), candidate(np.eye(3), 0.16)]
            selected, window, diagnostic = select_direction_candidate(candidates, np.eye(3), window, 0.02)
            self.assertEqual(selected, 1)
            self.assertEqual(diagnostic["escape_count"], 1)
            np.testing.assert_allclose(window["angular_velocity"], 0.0)

    def test_escape_requires_consecutive_observation_advantage_not_equal_cost(self):
        window = None
        for stable_cost in (0.16, 0.16, 0.14, 0.16):
            selected, window, diagnostic = select_direction_candidate(
                [candidate(rotation(-10), 0.0), candidate(np.eye(3), stable_cost)],
                np.eye(3), window, 0.02
            )
            self.assertEqual(selected, 1)
        self.assertEqual(diagnostic["escape_count"], 1)
        for _ in range(5):
            selected, window, diagnostic = select_direction_candidate(
                [candidate(rotation(-10), 0.0), candidate(np.eye(3), 0.0)],
                np.eye(3), window, 0.02, escape_margin=0.0
            )
            self.assertEqual(selected, 1)
            self.assertEqual(diagnostic["escape_count"], 0)

    def test_only_valid_reverse_candidate_is_never_withheld(self):
        window = copy_direction_window()
        window["angular_velocity"][:] = np.deg2rad([100.0, 0.0, 0.0])
        candidates = [candidate(rotation(-14), 0.0)]
        selected, window, diagnostic = select_direction_candidate(candidates, np.eye(3), window, 0.02)
        self.assertEqual(selected, 0)
        self.assertEqual(diagnostic["decision_reason"], "only_candidate")
        np.testing.assert_allclose(window["angular_velocity"], 0.0)
        self.assertEqual(diagnostic["candidates"][0]["applied_direction_cost"], 0.0)

    def test_all_valid_candidates_disagree_with_prediction_use_best_measurement(self):
        candidates = [candidate(rotation(-12), 0.0), candidate(rotation(10), 0.1)]
        selected, window, diagnostic = select_direction_candidate(candidates, np.eye(3), None, 0.02)
        self.assertEqual(selected, 0)
        self.assertEqual(diagnostic["decision_reason"], "all_candidates_conflict")
        np.testing.assert_allclose(window["angular_velocity"], 0.0)

    def test_zero_weight_has_exact_old_min_cost_and_tie_order(self):
        rng = np.random.default_rng(41)
        window = copy_direction_window()
        window["angular_velocity"][:] = [1.0, -2.0, 3.0]
        for _ in range(50):
            costs = rng.integers(0, 4, size=6).astype(float)
            candidates = [candidate(rotation(float(rng.uniform(-45, 45))), cost) for cost in costs]
            selected, updated, diagnostic = select_direction_candidate(
                candidates, rotation(25), window, 0.02, prior_per_deg=0.0
            )
            self.assertEqual(selected, int(np.argmin(costs)))
            self.assertEqual([trial["selection_cost"] for trial in diagnostic["candidates"]], costs.tolist())
            self.assertFalse(diagnostic["enabled"])
            np.testing.assert_allclose(updated["angular_velocity"], 0.0)

    def test_chooser_and_copy_do_not_mutate_previous_state_or_candidates(self):
        original = copy_direction_window()
        original["angular_velocity"][:] = [0.1, 0.2, 0.3]
        original["escape_axis"][:] = [1.0, 0.0, 0.0]
        before = deepcopy(original)
        candidates = [candidate(rotation(1), 0.0), candidate(rotation(-10), 0.1)]
        old_candidates = deepcopy(candidates)
        _, updated, diagnostic = select_direction_candidate(candidates, np.eye(3), original, 0.02)
        updated["angular_velocity"][:] = 123.0
        updated["escape_axis"][:] = 456.0
        for name in ("angular_velocity", "escape_axis"):
            np.testing.assert_array_equal(original[name], before[name])
        for old, current in zip(old_candidates, candidates):
            np.testing.assert_array_equal(old["pose"], current["pose"])
            self.assertEqual(old["base_cost"], current["base_cost"])
        copied = copy_direction_window(original)
        copied["angular_velocity"][:] = 789.0
        np.testing.assert_array_equal(original["angular_velocity"], before["angular_velocity"])
        json.dumps(diagnostic, allow_nan=False)

    def test_rotated_object_still_measures_increment_in_fixed_camera_axes(self):
        previous = rotation(90, (0, 0, 1))
        observed = rotation(2, (1, 0, 0)) @ previous
        _, window, _ = select_direction_candidate(
            [candidate(observed, 0.0), candidate(previous, 1.0)], previous, None, 0.02
        )
        # Rprev.T @ Rcand 会把这个相机 x 轴转动错记成物体 y 轴。
        self.assertGreater(window["angular_velocity"][0], 0.1)
        np.testing.assert_allclose(window["angular_velocity"][1:], 0.0, atol=1e-12)

    def test_camera_frame_change_transports_window_and_keeps_selection(self):
        previous = rotation(31, (0.2, 0.7, 0.4))
        window = copy_direction_window()
        window["angular_velocity"][:] = np.deg2rad([20.0, 10.0, -15.0])
        candidates = [
            candidate(rotation(2, (0.3, -0.2, 0.6)) @ previous, 0.1),
            candidate(rotation(-8, (0.3, -0.2, 0.6)) @ previous, 0.0),
        ]
        selected, next_window, diagnostic = select_direction_candidate(candidates, previous, window, 0.03)
        frame_change = rotation(67, (-0.2, 0.5, 0.4))
        changed = [candidate(frame_change @ item["pose"][:3, :3], item["base_cost"]) for item in candidates]
        changed_window = copy_direction_window(window)
        changed_window["angular_velocity"] = frame_change @ window["angular_velocity"]
        selected_changed, next_changed, changed_diagnostic = select_direction_candidate(
            changed, frame_change @ previous, changed_window, 0.03
        )
        self.assertEqual(selected_changed, selected)
        np.testing.assert_allclose(
            next_changed["angular_velocity"], frame_change @ next_window["angular_velocity"], atol=1e-10
        )
        np.testing.assert_allclose(
            [item["direction_prior_deg"] for item in diagnostic["candidates"]],
            [item["direction_prior_deg"] for item in changed_diagnostic["candidates"]], atol=1e-9
        )

    def test_long_gap_forgets_prior_and_uses_current_best_measurement(self):
        window = copy_direction_window()
        window["angular_velocity"][:] = np.deg2rad([100.0, 0.0, 0.0])
        candidates = [candidate(rotation(-10), 0.0), candidate(np.eye(3), 0.1)]
        selected, updated, diagnostic = select_direction_candidate(candidates, np.eye(3), window, 1.0)
        self.assertEqual(selected, 0)
        self.assertEqual(diagnostic["decision_reason"], "gap_reset")
        self.assertTrue(diagnostic["gap_reset"])
        np.testing.assert_allclose(updated["angular_velocity"], 0.0)

    def test_nonpositive_dt_cannot_generate_infinite_velocity_or_reject_output(self):
        for dt in (0.0, -0.02):
            selected, window, diagnostic = select_direction_candidate(
                [candidate(rotation(-10), 0.0), candidate(np.eye(3), 0.1)], np.eye(3), None, dt
            )
            self.assertEqual(selected, 0)
            np.testing.assert_allclose(window["angular_velocity"], 0.0)
            json.dumps(diagnostic, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
