"""Observe red/blue placement in RGB-D; FP supplies only the unsigned box geometry."""
from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np


def color_labels(rgb):
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    colored = (saturation >= 70) & (value >= 40)
    labels = np.zeros(hue.shape, np.int8)
    labels[colored & ((hue <= 12) | (hue >= 168))] = 1
    labels[colored & (hue >= 95) & (hue <= 140)] = -1
    return labels


def placement(red_direction, center_camera, up_camera):
    """Name the color at physical UP, camera-facing FRONT, or camera LEFT."""
    if up_camera is None:
        return 'unknown', 'gravity_unavailable'
    up = np.asarray(up_camera, dtype=float)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-6:
        return 'unknown', 'gravity_unavailable'
    up = up / np.linalg.norm(up)
    direction = np.asarray(red_direction)
    direction = direction / np.linalg.norm(direction)
    vertical = float(direction @ up)
    # Ground-plane bearing toward the camera: height is kept separate from depth.
    front = -np.asarray(center_camera, dtype=float)
    front -= up * (front @ up)
    if np.linalg.norm(front) < .15 * np.linalg.norm(center_camera):
        # Overhead view has no well-defined horizontal near/far bearing.
        left = np.array([-1., 0., 0.])
        left -= up * (left @ up)
        if np.linalg.norm(left) < 1e-6:
            return 'unknown', 'reference_degenerate'
        left /= np.linalg.norm(left)
        values = np.array([vertical, float(direction @ left)])
        names = ('up', 'left')
    else:
        front /= np.linalg.norm(front)
        left = np.cross(front, up)
        values = np.array([vertical, float(direction @ front), float(direction @ left)])
        names = ('up', 'front', 'left')
    order = np.argsort(np.abs(values))[::-1]
    best = int(order[0])
    if abs(values[best]) < .75 or abs(values[best]) - abs(values[order[1]]) < .12:
        return 'unknown', 'between_directions'
    return ('red' if values[best] > 0 else 'blue') + names[best], 'ok'


@dataclass
class ColorOrientation:
    bounds: np.ndarray

    @classmethod
    def from_mesh(cls, mesh):
        """Enable automatically only for the project's red/blue runtime models."""
        visual = getattr(mesh, 'visual', None)
        if getattr(visual, 'kind', None) != 'vertex':
            return None
        colors = np.asarray(visual.vertex_colors)[:, :3].astype(np.uint8)
        labels = color_labels(colors.reshape(-1, 1, 3))
        if not np.any(labels == 1) or not np.any(labels == -1):
            return None
        vertices = np.asarray(mesh.vertices, dtype=float)
        return cls(np.stack((vertices.min(0), vertices.max(0))))

    def analyze(self, rgb, depth, k, pose, up_camera=None):
        result = dict(label='unknown', status='pose_unavailable', red_axis_camera=None,
                      samples=0, red_samples=0, blue_samples=0)
        if pose is None or np.shape(pose) != (4, 4) or not np.isfinite(pose).all():
            return result
        pose, k = np.asarray(pose), np.asarray(k)
        if depth.shape != rgb.shape[:2] or k.shape != (3, 3):
            raise ValueError('Color orientation needs aligned RGB-D and matching intrinsics')
        center = self.bounds.mean(0)
        half = (self.bounds[1] - self.bounds[0]) / 2
        corners = np.array(list(product(*zip(self.bounds[0], self.bounds[1]))))
        camera = corners @ pose[:3, :3].T + pose[:3, 3]
        if not np.isfinite(camera).all() or np.any(camera[:, 2] <= .01):
            return result
        uvw = camera @ k.T
        uv = uvw[:, :2] / uvw[:, 2:]
        h, w = depth.shape
        low = np.maximum(np.floor(uv.min(0)).astype(int), 0)
        high = np.minimum(np.ceil(uv.max(0)).astype(int) + 1, (w, h))
        if np.any(high - low < 4):
            result['status'] = 'target_outside_view'
            return result
        x0, y0 = low
        x1, y1 = high
        # At most ~6000 current-frame pixels; no full-image HSV or temporal filter.
        step = max(1, int(np.ceil(np.sqrt((x1-x0) * (y1-y0) / 6000))))
        labels = color_labels(rgb[y0:y1:step, x0:x1:step])
        yy, xx = np.mgrid[y0:y1:step, x0:x1:step]
        z = depth[yy, xx]
        good = (labels != 0) & np.isfinite(z) & (z > .1) & (z < 5.)
        pixels = np.column_stack((xx[good], yy[good], np.ones(int(good.sum()))))
        xyz = (pixels @ np.linalg.inv(k).T) * z[good, None]
        local = (xyz - pose[:3, 3]) @ pose[:3, :3] - center
        inside = np.all(np.abs(local) <= half + .015, axis=1)
        surface = np.min(np.abs(np.abs(local) - half), axis=1) <= .02
        local, observed = local[inside & surface], labels[good][inside & surface]
        result.update(samples=len(observed), red_samples=int(np.sum(observed == 1)),
                      blue_samples=int(np.sum(observed == -1)))
        if len(observed) < 40:
            result['status'] = 'insufficient_color_depth'
            return result
        both = min(result['red_samples'], result['blue_samples']) >= max(12, len(observed) * .03)
        candidates = []
        # All six directions are re-evaluated from observed color. Permuting or
        # reversing the FP axes of a symmetric cube preserves this physical result.
        for axis in range(3):
            away = np.abs(local[:, axis]) >= max(.003, half[axis] * .05)
            support = float(away.mean())
            for sign in (-1, 1):
                correct = np.where(local[:, axis] * sign >= 0, 1, -1) == observed
                groups = (observed == 1, observed == -1) if both else (np.ones(len(observed), bool),)
                fractions = [float(correct[away & group].mean()) if np.any(away & group) else 0.
                             for group in groups]
                accuracy = float(np.mean(fractions))
                candidates.append((accuracy * support, accuracy, support, axis, sign))
        candidates.sort(reverse=True)
        score, accuracy, support, axis, sign = candidates[0]
        margin = score - candidates[1][0]
        result.update(fit=accuracy, margin=margin,
                      evidence='red_blue_split' if both else 'single_visible_color')
        if accuracy < .88 or support < .65 or margin < .08:
            result['status'] = 'ambiguous_color'
            return result
        direction = pose[:3, axis] * sign
        center_camera = center @ pose[:3, :3].T + pose[:3, 3]
        result['red_axis_camera'] = direction.tolist()
        result['center_camera_m'] = center_camera.tolist()
        result['label'], result['status'] = placement(direction, center_camera, up_camera)
        return result

    @staticmethod
    def draw(rgb, result, k=None):
        label = result['label']
        if label in ('redup', 'blueup') and result['status'] == 'ok' and k is not None:
            center = np.asarray(result.get('center_camera_m'), dtype=float)
            direction = np.asarray(result.get('red_axis_camera'), dtype=float)
            k = np.asarray(k)
            if (center.shape == (3,) and direction.shape == (3,) and k.shape == (3, 3)
                    and np.isfinite(center).all() and np.isfinite(direction).all()
                    and np.isfinite(k).all()):
                if label == 'blueup':
                    direction = -direction
                camera = np.stack((center, center + .1 * direction))
                if np.all(camera[:, 2] > .001):
                    projected = camera @ k.T
                    pixels = projected[:, :2] / projected[:, 2:]
                    h, w = rgb.shape[:2]
                    start = pixels[0]
                    if 0 <= start[0] < w and 0 <= start[1] < h:
                        delta = pixels[1] - start
                        length = np.linalg.norm(delta)
                        # Overhead views collapse physical UP along the viewing ray;
                        # use a screen-UP status marker in that degenerate case.
                        delta = delta / length if length >= 1 else np.array([0., -1.])
                        end = start + delta * max(40, round(min(h, w) * .25))
                        arrow_color = (255, 40, 40) if label == 'redup' else (40, 110, 255)
                        cv2.arrowedLine(rgb, tuple(np.rint(start).astype(int)),
                                        tuple(np.rint(end).astype(int)), arrow_color,
                                        max(4, round(min(h, w) * .0125)), cv2.LINE_AA, tipLength=.25)
        reasons = dict(gravity_unavailable='no gravity', insufficient_color_depth='color/depth',
                       ambiguous_color='color unclear', between_directions='diagonal',
                       pose_unavailable='no target', target_outside_view='outside view')
        text = label if label != 'unknown' else 'unknown: ' + reasons.get(result['status'], 'no reference')
        scale = max(.65, min(rgb.shape[:2]) / 480)
        (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        if width > rgb.shape[1] - 24:
            scale *= (rgb.shape[1] - 24) / width
            (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        x, y = 12, rgb.shape[0] - 18
        color = (255, 100, 100) if label.startswith('red') else (50, 190, 255)
        if label == 'unknown':
            color = (255, 220, 80)
        cv2.rectangle(rgb, (x-6, y-height-7), (x+width+6, y+baseline+5), (0, 0, 0), -1)
        cv2.putText(rgb, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        return rgb
