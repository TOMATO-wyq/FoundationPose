"""Small CPU z-buffer renderer for repeatable tests; not a real-camera benchmark."""

import numpy as np
import cv2

from .geometry import Camera, cube_vertices, rotation_z, transform, transform_points, unit
from .io import Frame


def default_camera(width=640, height=480):
    focal = width * 0.72
    return Camera(width, height, [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], ())


C_FROM_WORLD = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


def world_box(center, size, yaw_deg, camera_world=(0, -1.2, 0.35), color=(35, 120, 220)):
    rotation = C_FROM_WORLD @ rotation_z(np.deg2rad(yaw_deg))
    position = C_FROM_WORLD @ (np.asarray(center) - camera_world)
    return dict(size_m=float(size), pose=transform(rotation, position), color=color)


class Renderer:
    def __init__(self, camera):
        if np.any(camera.d):
            raise ValueError("image renderer supports pinhole images; test distorted PnP analytically")
        self.camera = camera
        yy, xx = np.indices((camera.height, camera.width))
        self.rays = camera.rays(np.column_stack((xx.ravel(), yy.ravel()))).reshape(
            camera.height, camera.width, 3
        )

    def render(
        self, boxes, sequence=0, depth_noise_m=0.0, holes=0.0, floor_height_camera_m=0.35, occluders=()
    ):
        c = self.camera
        rgb = np.full((c.height, c.width, 3), 48, np.uint8)
        depth = np.full((c.height, c.width), np.inf, np.float64)
        instance = np.full((c.height, c.width), -1, np.int16)
        # Floor normal in RGB optical coordinates; camera is above the floor.
        denom = self.rays[:, :, 1]
        z = np.divide(floor_height_camera_m, denom, out=np.full_like(denom, np.inf), where=denom > 0)
        visible = (z > 0) & (z < 4.0)
        depth[visible] = z[visible]
        rgb[visible] = (155, 158, 162)
        truth = []
        light = unit(C_FROM_WORLD @ np.array([-1.0, -1.5, 2.0]))
        for identity, box in enumerate(boxes):
            size, pose = box["size_m"], box["pose"]
            vertices = cube_vertices(size)
            xyz = transform_points(pose, vertices)
            uv = c.project(xyz)
            for axis in range(3):
                for sign in (-1, 1):
                    indices = np.flatnonzero(np.isclose(vertices[:, axis], sign * size / 2))
                    normal = pose[:3, axis] * sign
                    center = xyz[indices].mean(axis=0)
                    if normal @ center >= -1e-8 or np.any(xyz[indices, 2] <= 0):
                        continue
                    hull = cv2.convexHull(uv[indices].astype(np.float32)).reshape(-1, 2)
                    mask = np.zeros(depth.shape, np.uint8)
                    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1)
                    yy, xx = np.nonzero(mask)
                    distances = self.rays[yy, xx] @ normal
                    face_z = (normal @ center) / distances
                    update = (face_z > 0) & (face_z < depth[yy, xx])
                    yy, xx, face_z = yy[update], xx[update], face_z[update]
                    depth[yy, xx] = face_z
                    intensity = 0.65 + 0.35 * max(0.0, float(normal @ light))
                    if "red_axis_local" in box:
                        local = (self.rays[yy, xx] * face_z[:, None] - pose[:3, 3]) @ pose[:3, :3]
                        red = local @ np.asarray(box["red_axis_local"]) >= 0
                        paint = np.where(red[:, None], (235, 35, 35), (40, 45, 225))
                    else:
                        paint = np.asarray(box["color"])
                    rgb[yy, xx] = np.clip(paint * intensity, 0, 255).astype(np.uint8)
                    instance[yy, xx] = identity
            reference = dict(
                id=identity,
                size_m=size,
                center_camera_m=pose[:3, 3].tolist(),
                T_camera_from_box=pose.tolist(),
            )
            if "red_axis_local" in box:
                direction = pose[:3, :3] @ np.asarray(box["red_axis_local"])
                vertical = direction @ np.array([0.0, -1.0, 0.0])
                reference.update(
                    red_axis_camera=direction.tolist(),
                    color_orientation=(
                        "red_up" if vertical > 0.9 else "blue_up" if vertical < -0.9 else "sideways"
                    ),
                )
            truth.append(reference)
        # Physical foreground planes with their own z-buffer depth. They are
        # distractors, not target cubes and not holes painted over an image.
        for occluder in occluders:
            xyz = np.asarray(occluder["quad_camera"], float)
            normal = unit(np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0]))
            mask = np.zeros(depth.shape, np.uint8)
            cv2.fillConvexPoly(mask, np.rint(c.project(xyz)).astype(np.int32), 1)
            yy, xx = np.nonzero(mask)
            z = (normal @ xyz[0]) / (self.rays[yy, xx] @ normal)
            update = (z > 0) & (z < depth[yy, xx])
            yy, xx, z = yy[update], xx[update], z[update]
            depth[yy, xx] = z
            rgb[yy, xx] = occluder.get("color", (170, 165, 150))
            instance[yy, xx] = -2
        rng = np.random.default_rng(1700 + sequence)
        good = np.isfinite(depth)
        if depth_noise_m:
            depth[good] += rng.normal(0, depth_noise_m, int(good.sum()))
        if holes:
            good &= rng.random(depth.shape) >= holes
        depth[~good] = 0
        return (
            Frame(rgb, depth.astype(np.float32), c, 1_000_000_000 + sequence * 16_666_667, sequence),
            truth,
            instance,
        )


def demo_scenes():
    """Low camera, side-view boxes, two objects, aligned stack and a size negative."""
    scenes = []
    for size in (0.20, 0.35):
        for angle in (0, 15, 35, 60, 85):
            scenes.append((f"single_{size:.2f}_{angle}", [world_box((0, 0, size / 2), size, angle)]))
    scenes.append(
        (
            "two_same_color",
            [world_box((-0.27, 0.15, 0.175), 0.35, 20), world_box((0.27, 0.15, 0.175), 0.35, 50)],
        )
    )
    scenes.append(
        ("mixed_sizes", [world_box((-0.23, 0, 0.10), 0.20, 15), world_box((0.24, 0.10, 0.175), 0.35, 30)])
    )
    scenes.append(
        ("stack_aligned_no_seam", [world_box((0, 0, 0.175), 0.35, 20), world_box((0, 0, 0.525), 0.35, 20)])
    )
    scenes.append(
        ("stack_rotated", [world_box((0, 0, 0.175), 0.35, 15), world_box((0.025, 0, 0.525), 0.35, 40)])
    )
    scenes.append(("empty", []))
    scenes.append(("wrong_size_27cm", [world_box((0, 0, 0.135), 0.27, 20)]))
    return scenes


def reference_stacking_scenes():
    """Topology/paint inspired by the user's 2 x 6 diagrams, not measured replicas.

    Assumptions: 35 cm base/middle, 20 cm equal red/blue top, camera 35 cm
    high and 1.8 m away. The green scoring region/rules are deliberately absent.
    """
    camera = (0, -1.8, 0.35)
    red, blue = (230, 35, 35), (40, 45, 225)

    def big(height=0.175, yaw=15, xy=(0, 0), color=red):
        return world_box((*xy, height), 0.35, yaw, camera, color)

    def top(bottom, axis=(0, 0, 1), yaw=15, xy=(0, 0)):
        box = world_box((*xy, bottom + 0.1), 0.20, yaw, camera)
        box["red_axis_local"] = list(axis)
        return box

    return [
        ("ref1_01_single_red_base", [big()]),
        ("ref1_02_single_blue_base", [big(color=blue)]),
        ("ref1_03_top_sideways_on_floor", [top(0, (1, 0, 0))]),
        ("ref1_04_top_red_up_on_floor", [top(0)]),
        ("ref1_05_two_large_red_offset", [big(), big(0.525, 20, (0.035, 0))]),
        ("ref1_06_two_large_red_blue_offset", [big(), big(0.525, 35, (0.07, 0.02), blue)]),
        ("ref2_01_two_large_offset_rotated", [big(), big(0.525, 35, (-0.065, 0.02))]),
        ("ref2_02_base_and_top_red_up", [big(), top(0.35)]),
        ("ref2_03_three_levels_red_blue_red_up", [big(), big(0.525, color=blue), top(0.70)]),
        ("ref2_04_base_and_top_blue_up", [big(), top(0.35, (0, 0, -1), 25)]),
        ("ref2_05_three_levels_blue_up", [big(), big(0.525), top(0.70, (0, 0, -1), 30)]),
        (
            "ref2_06_three_levels_sideways",
            [big(), big(0.525, 25, (0.035, 0.02)), top(0.70, (1, 0, 0), 25, (0.035, 0.02))],
        ),
    ]
