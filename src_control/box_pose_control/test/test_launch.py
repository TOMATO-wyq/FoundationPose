"""Verify one size choice reaches recognition, control and view without hardware."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions


PACKAGE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("real_pipeline", PACKAGE / "launch/real_pipeline.launch.py")
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


def actions(**overrides):
    with patch.object(PIPELINE, "get_package_share_directory", return_value=str(PACKAGE)):
        context = LaunchContext()
        for action in PIPELINE.generate_launch_description().entities:
            if isinstance(action, DeclareLaunchArgument):
                context.launch_configurations[action.name] = perform_substitutions(context, action.default_value)
        context.launch_configurations.update(repo_root=str(PACKAGE.parents[1]), **overrides)
        with patch.object(PIPELINE, "Node", side_effect=lambda **kw: ("node", kw)), \
             patch.object(PIPELINE, "ExecuteProcess", side_effect=lambda **kw: ("process", kw)):
            return PIPELINE._launch_setup(context)


@pytest.mark.parametrize("size,config", [("big", "box4d_d435i_live.json"),
                                       ("small", "box4d_d435i_yolo_live.json")])
def test_size_selects_matching_pipeline_and_preserves_target_coordinates(size, config):
    launched = actions(size=size, target_distance_m="0.5", target_lateral_m="-0.08")
    recognition = next(kw for kind, kw in launched if kw["name"] == "box4d_recognition")
    cmd = recognition["cmd"]
    assert cmd[cmd.index("--size") + 1] == size
    assert Path(cmd[cmd.index("--config") + 1]).name == config
    assert "--auto-up" in cmd and "--publish-poses" in cmd
    control = next(kw for kind, kw in launched if kw["name"] == "box_pose_control")
    assert control["parameters"][1]["size"] == size
    assert control["parameters"][1]["target_distance_m"] == 0.5
    assert control["parameters"][1]["target_lateral_m"] == -0.08
    assert control["parameters"][1]["dry_run"] is True
    assert control["parameters"][1]["max_forward_speed"] == 0.5
    assert control["parameters"][1]["max_lateral_speed"] == 0.5
    assert control["parameters"][1]["camera_yaw_deg"] == pytest.approx(0.0)
    assert control["parameters"][1]["serial_yaw_sign"] == 1.0
    view = next(kw for kind, kw in launched if kw["name"] == "turtlesim_real_view")
    assert view["parameters"][0]["size"] == size
    assert view["parameters"][0]["camera_yaw_deg"] == pytest.approx(0.0)


@pytest.mark.parametrize("yaw", ["0.0", "-36.86989765"])
def test_mount_yaw_override_reaches_controller_and_view(yaw):
    launched = actions(camera_yaw_deg=yaw)
    control = next(kw for _, kw in launched if kw["name"] == "box_pose_control")
    view = next(kw for _, kw in launched if kw["name"] == "turtlesim_real_view")
    assert control["parameters"][1]["camera_yaw_deg"] == float(yaw)
    assert view["parameters"][0]["camera_yaw_deg"] == float(yaw)


@pytest.mark.parametrize("yaw,expected_left", [(36.86989765, 0.3), (-36.86989765, -0.3)])
def test_readonly_view_projects_with_same_mount_yaw(yaw, expected_left):
    spec = importlib.util.spec_from_file_location("real_view", PACKAGE / "scripts/turtlesim_real_view.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    view = SimpleNamespace(up=(0.0, -1.0, 0.0), camera_yaw_deg=yaw)
    projected = module.TurtlesimRealView.relative_in_robot_frame(view, 0.0, 0.0, 0.5)
    assert projected == pytest.approx((0.4, expected_left))


def test_external_sources_and_headless_mode_launch_only_controller():
    launched = actions(size="small", start_camera="false", start_recognition="false", with_view="false")
    assert len(launched) == 1
    assert launched[0][1]["name"] == "box_pose_control"


def test_invalid_size_fails_before_starting_processes():
    with pytest.raises(ValueError, match="size"):
        actions(size="medium")


def test_yaw_startup_arguments_reach_controller():
    launched = actions(yaw_enabled="false", serial_yaw_sign="-1.0", serial_yaw_scale="0.75")
    control = next(kw for _, kw in launched if kw["name"] == "box_pose_control")
    params = control["parameters"][1]
    assert params["yaw_enabled"] is False
    assert params["serial_yaw_sign"] == -1.0
    assert params["serial_yaw_scale"] == 0.75


def load_bridge(filename):
    spec = importlib.util.spec_from_file_location(filename, PACKAGE / "scripts" / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("filename,classname", [("turtlesim_pose_sim", "TurtlesimPoseSim"),
                                                 ("turtlesim_real_view", "TurtlesimRealView")])
def test_bridge_rotates_in_place_and_stops_rotation_on_timeout(filename, classname):
    from geometry_msgs.msg import Twist
    module = load_bridge(filename)
    requests = []
    bridge = SimpleNamespace(
        last_step=0.9, ready=True, cam_x=5.0, cam_y=5.0, cam_theta=0.0,
        cmd_at=0.9, cmd_timeout=0.3, cmd=Twist(), scale=0.1,
        camera_teleport_future=None,
        camera_teleport_client=SimpleNamespace(service_is_ready=lambda: True,
                                               call_async=lambda request: requests.append(request)),
    )
    bridge.cmd.angular.z = 0.2
    tick = getattr(module, classname).motion_tick
    with patch.object(module.time, "monotonic", return_value=1.0):
        tick(bridge)
    assert len(requests) == 1
    assert requests[0].theta == pytest.approx(0.01)
    assert requests[0].x == requests[0].y == 5.0
    with patch.object(module.time, "monotonic", return_value=2.0):
        tick(bridge)
    assert len(requests) == 1
    assert bridge.cam_theta == pytest.approx(0.01)


def test_simulated_detection_contains_relative_box_yaw():
    import math
    from builtin_interfaces.msg import Time
    module = load_bridge("turtlesim_pose_sim")
    messages = []
    bridge = SimpleNamespace(
        ready=True, camera_pose=SimpleNamespace(x=0.0, y=0.0, theta=0.3),
        box_pose=SimpleNamespace(x=1.0, y=0.0, theta=0.5),
        scale=0.1, frame_id="camera", target_id="cube", size_m=0.2,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=1))),
        relative_pose=module.TurtlesimPoseSim.relative_pose,
        detection_pub=SimpleNamespace(publish=messages.append),
        up_pub=SimpleNamespace(publish=lambda message: None),
    )
    module.TurtlesimPoseSim.publish_tick(bridge)
    q = messages[0].detections[0].bbox.center.orientation
    # 方块 X 轴在光学系的方向；前方为 +Z，左侧为 -X。
    axis_x = 1 - 2*(q.y*q.y + q.z*q.z)
    axis_z = 2*(q.x*q.z - q.w*q.y)
    assert math.atan2(-axis_x, axis_z) == pytest.approx(0.2)
