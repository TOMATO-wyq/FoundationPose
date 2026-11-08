"""Instance association with separate, bounded predictions for missed observations."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import angle_difference
from .diagnostics import has_current_rejections
from .motion import PositionKalman


class Tracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = {}
        self.next_id = 1
        self.last_stamp = None
        self.calibration_id = None
        self.reference = None
        self.frame_id = None

    def reset(self):
        self.tracks.clear()
        self.last_stamp = None

    def _prediction(self, track, stamp):
        motion = track.get("motion")
        if self.cfg.motion_prediction and motion is not None and motion.can_predict(stamp):
            return motion.predict(stamp)
        return None

    @staticmethod
    def _predicted_object(identity, track, prediction, stamp):
        # The last rotation is only an assumption; no old image/depth evidence is copied.
        if track["pose"] is None:
            return None
        mean, covariance = prediction
        pose = track["pose"].copy()
        pose[:3, 3] = mean[:3]
        return dict(
            track_id=identity,
            size_m=track["size"],
            center_camera_m=mean[:3].tolist(),
            T_camera_from_box=pose.tolist(),
            source="kf_prediction",
            status="predicted",
            valid=False,
            last_observed_stamp_ns=track["stamp_ns"],
            prediction_age_s=(stamp - track["stamp_ns"]) / 1e9,
            position_covariance_m2=covariance[:3, :3].tolist(),
        )

    def update(self, result):
        stamp = result["stamp_ns"]
        if self.last_stamp is not None and stamp <= self.last_stamp:
            raise ValueError("tracker rejects repeated or out-of-order source frames")
        reference = tuple(result.get("up_camera") or ())
        frame_id = result.get("frame_id")
        if (
            result["calibration_id"] != self.calibration_id
            or reference != self.reference
            or frame_id != self.frame_id
        ):
            self.reset()
            self.calibration_id = result["calibration_id"]
            self.reference = reference
            self.frame_id = frame_id
        self.last_stamp = stamp
        predictions = {
            identity: prediction
            for identity, track in self.tracks.items()
            if (prediction := self._prediction(track, stamp)) is not None
        }
        old = {
            i: t for i, t in self.tracks.items()
            if (stamp - t["stamp_ns"]) / 1e9 <= self.cfg.track_max_gap_s or i in predictions
        }
        objects = result["objects"]
        ids = list(old)
        costs = np.full((len(ids), len(objects)), 1e6)
        for row, identity in enumerate(ids):
            track = old[identity]
            center = predictions[identity][0][:3] if identity in predictions else track["center"]
            for column, observation in enumerate(objects):
                distance = np.linalg.norm(np.asarray(observation["center_camera_m"]) - center)
                if observation["size_m"] != track["size"] or distance > self.cfg.track_max_distance_m:
                    continue
                yaw = observation["yaw_u_rad"]
                angle = 0 if yaw is None or track["yaw"] is None else abs(angle_difference(yaw, track["yaw"]))
                costs[row, column] = distance / self.cfg.track_max_distance_m + angle
        assigned = {}
        if costs.size:
            rows, columns = linear_sum_assignment(costs)
            assigned = {int(c): ids[int(r)] for r, c in zip(rows, columns) if costs[r, c] < 1e5}
        tracks = {}
        for column, observation in enumerate(objects):
            identity = assigned.get(column)
            previous = old.get(identity)
            if identity is None:
                identity = self.next_id
                self.next_id += 1
            yaw = observation["yaw_u_rad"]
            smooth = yaw
            if yaw is not None and previous is not None and previous["smooth_yaw"] is not None:
                smooth = previous["smooth_yaw"] + self.cfg.smoothing_alpha * angle_difference(
                    yaw, previous["smooth_yaw"]
                )
            center = np.asarray(observation["center_camera_m"], dtype=float)
            smooth_center = (
                center
                if previous is None
                else previous["smooth_center"]
                + self.cfg.smoothing_alpha * (center - previous["smooth_center"])
            )
            observation.setdefault("source", "current_rgbd")
            observation.update(
                track_id=identity,
                yaw_u_smoothed_rad=None if smooth is None else float(smooth % (np.pi / 2)),
                yaw_u_continuous_rad=smooth,
                center_camera_smoothed_m=smooth_center.tolist(),
            )
            if observation["yaw_reference_ready"]:
                observation["status"] = (
                    "tracked_rgbd"
                    if observation["source"] == "current_rgbd_flow"
                    else "tracking" if previous is not None else "detected"
                )
            motion = None
            if self.cfg.motion_prediction:
                previous_motion = None if previous is None else previous.get("motion")
                if (
                    previous_motion is None
                    or (stamp - previous_motion.stamp_ns) / 1e9 > self.cfg.prediction_max_gap_s
                ):
                    motion = PositionKalman(center, stamp, self.cfg)
                else:
                    motion = previous_motion.updated(center, stamp)
            pose = observation.get("T_camera_from_box")
            tracks[identity] = dict(
                stamp_ns=stamp,
                center=center,
                smooth_center=smooth_center,
                size=observation["size_m"],
                yaw=yaw,
                smooth_yaw=smooth,
                motion=motion,
                pose=None if pose is None else np.array(pose, dtype=float, copy=True),
            )
        result["lost_track_ids"] = [identity for identity in self.tracks if identity not in tracks]
        result["predicted_objects"] = [
            predicted for identity, prediction in predictions.items()
            if identity not in tracks
            and (predicted := self._predicted_object(identity, old[identity], prediction, stamp)) is not None
        ]
        # Missing frames never update a KF, extend its lifetime, or become valid observations.
        self.tracks = {**old, **tracks}
        temporal = result.get("temporal", {})
        rejected = (
            result.get("frame_status") == "partial_or_rejected"
            or has_current_rejections(result)
            or temporal.get("tracking_failures")
            or temporal.get("detector_seed_failures")
        )
        result["frame_status"] = "observed" if objects else "partial_or_rejected" if rejected else "no_target"
        return result
