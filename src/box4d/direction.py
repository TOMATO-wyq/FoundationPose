"""用短时运动趋势辅助选择当前帧姿态，不平滑输出，也不拒绝有效候选。"""

import cv2
import numpy as np


# 窗口存角速度，因此不同帧率下预测的角增量仍对应同一真实运动。
# 低于 5°/s 的变化视作静止噪声，单帧死区最多 1°；窗口仍会衰减。
_DEADZONE_SPEED_DEG_S = 5.0
_MAX_DEADZONE_DEG = 1.0
_CONFLICT_DEG = 5.0


def copy_direction_window(window=None):
    """每条轨迹独占窗口数组；浅拷贝 state 时也不会改到上一帧。"""
    if window is None:
        return {
            "angular_velocity": np.zeros(3, dtype=np.float64),
            "escape_count": 0,
            "escape_axis": np.zeros(3, dtype=np.float64),
        }
    return {
        "angular_velocity": np.asarray(window["angular_velocity"], dtype=np.float64).copy(),
        "escape_count": int(window["escape_count"]),
        "escape_axis": np.asarray(window["escape_axis"], dtype=np.float64).copy(),
    }


def select_direction_candidate(
    candidates,
    previous_rotation,
    window,
    dt_s,
    *,
    prior_per_deg=0.02,
    tau_s=0.2,
    escape_margin=0.15,
    escape_frames=3,
):
    """返回 ``(候选下标, 新窗口, 诊断)``，输入候选及窗口均不修改。

    candidates 是已通过原验收的非空列表，每项只需 ``pose`` 和
    ``base_cost``。base_cost 包含原有观测代价及 yaw 项，不含本模块先验。
    旋转增量 Rcand @ Rprev.T 和窗口角速度都在固定相机坐标系中。

    证据接近时靠趋势选分支；真实观测足够强时立即接管，较小但持续的
    优势累计 escape_frames 帧后接管。逃生比较原始 base_cost，避免先验
    把自己认定的方向永久固化。所有候选都不符合预测时也立即交还观测。
    """
    base = np.asarray([candidate["base_cost"] for candidate in candidates], dtype=np.float64)
    base_index = int(np.argmin(base))
    current = copy_direction_window(window)
    dt = float(dt_s)
    gap_reset = dt <= 0.0 or dt > max(0.5, 3.0 * tau_s)
    if gap_reset or prior_per_deg == 0.0:
        current = copy_direction_window()

    enabled = prior_per_deg > 0.0 and not gap_reset
    # 禁用时不让历史窗口改变排序，连同相等代价的原始先后顺序也保持。
    if not enabled:
        trials = [
            {
                "direction_prior_deg": 0.0,
                "direction_prior_cost": 0.0,
                "applied_direction_cost": 0.0,
                "selection_cost": float(cost),
                "motion_conflict": False,
                "is_reverse": False,
            }
            for cost in base
        ]
        return base_index, current, {
            "enabled": False,
            "decision_reason": "disabled" if prior_per_deg == 0.0 else "gap_reset",
            "candidates": trials,
            "direction_window_norm_deg": 0.0,
            "window_angular_velocity_deg_s": [0.0, 0.0, 0.0],
            "escape_count": 0,
            "gap_reset": gap_reset,
        }

    previous = np.asarray(previous_rotation, dtype=np.float64)
    increments = np.asarray([
        cv2.Rodrigues(np.asarray(candidate["pose"], dtype=np.float64)[:3, :3] @ previous.T)[0].ravel()
        for candidate in candidates
    ])
    predicted = current["angular_velocity"] * dt
    differences = increments - predicted
    prior_deg = np.rad2deg(np.linalg.norm(differences, axis=1))
    raw_prior = prior_per_deg * prior_deg
    applied_prior = raw_prior.copy()
    conflict = prior_deg > _CONFLICT_DEG
    deadzone_deg = min(_MAX_DEADZONE_DEG, _DEADZONE_SPEED_DEG_S * dt)
    predicted_moving = np.rad2deg(np.linalg.norm(predicted)) > deadzone_deg
    reverse = (increments @ predicted < 0.0) & predicted_moving
    selected = int(np.argmin(base + raw_prior))
    reason = "continuity_cost"
    reset_motion = False

    if len(candidates) == 1 or np.all(conflict):
        # 趋势只是辅助证据，不能阻止唯一可行运动或制造额外零输出。
        selected = base_index
        applied_prior.fill(0.0)
        reason = "only_candidate" if len(candidates) == 1 else "all_candidates_conflict"
        reset_motion = True
    elif conflict[base_index]:
        if selected == base_index:
            # 观测优势已胜过趋势惩罚，本帧立即响应真实转动。
            reason = "observation_wins"
            reset_motion = True
        else:
            compatible = np.flatnonzero(~conflict)
            compatible_base = float(np.min(base[compatible]))
            advantage = compatible_base - base[base_index]
            if advantage > 0.0 and advantage >= escape_margin:
                axis = differences[base_index] / np.linalg.norm(differences[base_index])
                # 连续支持同一偏移方向才累计，左右交替的噪声不合并计票。
                same_direction = float(np.dot(axis, current["escape_axis"])) > 0.5
                current["escape_count"] = current["escape_count"] + 1 if same_direction else 1
                current["escape_axis"] = axis.copy()
                if current["escape_count"] >= escape_frames:
                    selected = base_index
                    applied_prior.fill(0.0)
                    reason = "escape_evidence"
                    reset_motion = True
                else:
                    reason = "escape_pending"
            else:
                current["escape_count"] = 0
                current["escape_axis"].fill(0.0)
    else:
        current["escape_count"] = 0
        current["escape_axis"].fill(0.0)

    if reset_motion:
        current = copy_direction_window()
    else:
        alpha = -np.expm1(-dt / tau_s)
        step_deg = float(np.rad2deg(np.linalg.norm(increments[selected])))
        if step_deg <= deadzone_deg or conflict[selected]:
            # 死区不是“保持旧趋势”；静止时必须向零衰减。
            # 未经观测确认的孤立跳变也不能写成下一帧的运动趋势。
            current["angular_velocity"] *= 1.0 - alpha
        else:
            current["angular_velocity"] += alpha * (
                increments[selected] / dt - current["angular_velocity"]
            )

    trials = [
        {
            "direction_prior_deg": float(prior_deg[index]),
            "direction_prior_cost": float(raw_prior[index]),
            "applied_direction_cost": float(applied_prior[index]),
            "selection_cost": float(base[index] + applied_prior[index]),
            "motion_conflict": bool(conflict[index]),
            "is_reverse": bool(reverse[index]),
        }
        for index in range(len(candidates))
    ]
    return selected, current, {
        "enabled": True,
        "decision_reason": reason,
        "candidates": trials,
        "direction_window_norm_deg": float(np.rad2deg(np.linalg.norm(current["angular_velocity"] * dt))),
        "window_angular_velocity_deg_s": np.rad2deg(current["angular_velocity"]).tolist(),
        "escape_count": current["escape_count"],
        "gap_reset": gap_reset,
    }
