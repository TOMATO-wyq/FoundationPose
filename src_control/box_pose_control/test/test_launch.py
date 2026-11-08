"""Verify one size choice reaches recognition, control and view without hardware."""

import importlib.util
from pathlib import Path
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
    view = next(kw for kind, kw in launched if kw["name"] == "turtlesim_real_view")
    assert view["parameters"][0]["size"] == size


def test_external_sources_and_headless_mode_launch_only_controller():
    launched = actions(size="small", start_camera="false", start_recognition="false", with_view="false")
    assert len(launched) == 1
    assert launched[0][1]["name"] == "box_pose_control"


def test_invalid_size_fails_before_starting_processes():
    with pytest.raises(ValueError, match="size"):
        actions(size="medium")
