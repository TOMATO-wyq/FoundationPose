"""History validates repeated partial observations; it never creates a pose."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .camera_motion import motion_context, relative_camera_rotation


class UpperPartialHistory:
    def __init__(self, cfg):
        self.cfg = cfg
        self.entries = []
        self.proposals = []

    def clear(self):
        self.entries.clear()
        self.proposals.clear()

    def _reference(self, frame, entry):
        dt = (frame.stamp_ns - entry['stamp']) / 1e9
        rotation = relative_camera_rotation(entry['context'], motion_context(frame, self.cfg))
        rotation = np.eye(3) if rotation is None else rotation
        velocity = rotation @ entry['velocity']
        return rotation @ entry['center'] + velocity * dt, rotation, dt

    def evaluate(self, frame, objects):
        cfg = self.cfg
        horizon = max(cfg.track_max_gap_s, cfg.detector_max_age_s)
        self.entries = [e for e in self.entries if 0 < (frame.stamp_ns-e['stamp'])/1e9 <= horizon]
        self.proposals = []
        references = [self._reference(frame, e) for e in self.entries]
        matches = {}
        if objects and self.entries:
            costs = np.full((len(objects), len(self.entries)), 1e6)
            for i, obj in enumerate(objects):
                for j, entry in enumerate(self.entries):
                    if obj['size_m'] == entry['size']:
                        costs[i, j] = np.linalg.norm(np.asarray(obj['center_camera_m']) - references[j][0])
            rows, columns = linear_sum_assignment(costs)
            matches = {int(i): int(j) for i, j in zip(rows, columns) if costs[i, j] < 1e5}
            # Overlapping detector/flow duplicates share a reference. An
            # unassigned duplicate must not bypass a rejected repeated partial.
            for i, obj in enumerate(objects):
                if i in matches or not obj['metrics'].get('upper_partial'):
                    continue
                duplicate = next((k for k in matches if objects[k]['size_m'] == obj['size_m']
                                  and np.linalg.norm(np.asarray(objects[k]['center_camera_m'])
                                                     - obj['center_camera_m']) < .5*obj['size_m']), None)
                if duplicate is not None:
                    matches[i] = matches[duplicate]
        kept, rejected = [], []
        for i, obj in enumerate(objects):
            metrics = obj['metrics']
            partial = bool(metrics.get('upper_partial', False))
            previous = self.entries[matches[i]] if i in matches else None
            center = np.asarray(obj['center_camera_m'], float)
            velocity = np.zeros(3)
            residual, limit, dt = None, None, None
            streak = 1 if partial else 0
            failures = []
            if previous is not None:
                reference, rotation, dt = references[matches[i]]
                residual = float(np.linalg.norm(center - reference))
                # A full measurement can always establish a new object instead
                # of borrowing a distant object's history.
                if not partial and residual > cfg.track_max_distance_m:
                    previous = None
                else:
                    streak = previous['streak'] + 1 if partial else 0
                    velocity = (center - rotation @ previous['center']) / dt
                    velocity *= min(1., 1.5 / max(np.linalg.norm(velocity), 1e-9))
                    # The first partial has a soft reference only. From the
                    # second on, allow metric noise plus displacement growing
                    # with elapsed time and motion uncertainty, not a fixed
                    # position sphere around an old camera-frame coordinate.
                    limit = (max(4*cfg.max_depth_residual_m, .10*obj['size_m'])
                             + max(np.linalg.norm(previous['velocity']),
                                   3*cfg.prediction_initial_velocity_std_m_s)*dt
                             + .5*cfg.prediction_accel_std_m_s2*dt*dt)
                    if partial and previous['streak'] > 0:
                        if residual > limit:
                            failures.append('partial_history_position')
                        support = metrics.get('visible_plane_support_fraction', metrics.get('plane_support', 0.))
                        if support < max(.85, cfg.partial_min_plane_support_fraction):
                            failures.append('partial_history_depth')
                        edges = metrics.get('anchor_edge_support', ())
                        if len(edges) != 3 or min(edges) < max(.70, cfg.partial_min_edge_support_fraction):
                            failures.append('partial_history_boundaries')
            report = dict(partial_streak=streak, reference_age_s=dt,
                          position_residual_m=residual, position_limit_m=limit,
                          position_gate_enabled=bool(partial and previous is not None and previous['streak'] > 0),
                          reference_cost=0. if residual is None else residual / max(limit, 1e-9),
                          failed_checks=failures)
            metrics['partial_history'] = report
            if failures:
                rejected.append(dict(source='upper_partial_history', reason=','.join(failures),
                                     center_camera_m=obj['center_camera_m'], size_m=obj['size_m'], **report))
                continue
            # Use first-partial history as a soft preference between competing
            # partial hypotheses. Current complete observations keep their score.
            if partial and previous is not None:
                obj['quality_score'] /= 1 + .10 * min(report['reference_cost'], 3.)
            entry = dict(center=center.copy(), size=obj['size_m'], stamp=frame.stamp_ns,
                         context=motion_context(frame, cfg), velocity=velocity, streak=streak)
            self.proposals.append((obj, entry, previous))
            kept.append(obj)
        return kept, rejected

    def commit(self, selected):
        # Rejected/unselected hypotheses never move the reference or reset its
        # partial streak. A brief missing frame therefore cannot bypass a gate.
        for obj, entry, previous in self.proposals:
            if not any(obj is accepted for accepted in selected):
                continue
            self.entries = [e for e in self.entries if e is not previous]
            self.entries.append(entry)
