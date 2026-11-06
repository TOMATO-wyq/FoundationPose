"""Select a consistent set of cube hypotheses instead of overlapping duplicates."""

from itertools import combinations

import cv2
import numpy as np

from .geometry import upright_from_camera


def _footprint(obj, upright):
    pose = np.asarray(obj["T_camera_from_box"])
    half = obj["size_m"] / 2
    corners = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]])
    points = (corners @ pose[:3, :3].T + pose[:3, 3]) @ upright.T
    center = upright @ pose[:3, 3]
    return cv2.convexHull(points[:, :2].astype(np.float32)), float(center[2] - half), float(center[2] + half)


def select_scene(objects, max_objects, up_camera, max_overlap_fraction=0.08):
    """Maximum explained-face score under nonpenetration and count constraints.

    Small overlap is tolerated for measured pose noise. The score ranks hypotheses;
    it is not a probability. Missing targets are not synthesized to fill capacity.
    """
    if not objects or max_objects == 1:
        return objects[:max_objects], []
    upright = upright_from_camera(up_camera)
    footprints = [_footprint(o, upright) for o in objects] if upright is not None else None
    conflicts = set()
    for i, j in combinations(range(len(objects)), 2):
        a, b = objects[i], objects[j]
        if footprints is None:
            distance = np.linalg.norm(np.asarray(a["center_camera_m"]) - b["center_camera_m"])
            overlap = distance < 0.6 * min(a["size_m"], b["size_m"])
        else:
            pa, lowa, higha = footprints[i]
            pb, lowb, highb = footprints[j]
            height = max(0.0, min(higha, highb) - max(lowa, lowb))
            area = cv2.intersectConvexConvex(pa, pb)[0] if height else 0.0
            overlap = area * height / min(a["size_m"] ** 3, b["size_m"] ** 3) > max_overlap_fraction
        if overlap:
            conflicts.add((i, j))
    scores = [
        sum(cv2.contourArea(np.asarray(f["quad"], np.float32)) for f in o["faces"]) * o["quality_score"]
        for o in objects
    ]
    best, best_score = (), -1.0
    for count in range(1, min(max_objects, len(objects)) + 1):
        for indices in combinations(range(len(objects)), count):
            if any(pair in conflicts for pair in combinations(indices, 2)):
                continue
            score = sum(scores[i] for i in indices)
            if score > best_score:
                best, best_score = indices, score
    rejected = []
    for index, obj in enumerate(objects):
        if index in best:
            continue
        collides = any(tuple(sorted((index, j))) in conflicts for j in best)
        rejected.append(
            dict(
                center_camera_m=obj["center_camera_m"],
                size_m=obj["size_m"],
                reason="overlapping_volume_hypothesis" if collides else "lower_scene_evidence_or_capacity",
                explained_face_score=float(scores[index]),
            )
        )
    return [objects[i] for i in best], rejected
