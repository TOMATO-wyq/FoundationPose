#!/usr/bin/env python3
"""一键启动真实链路 + turtlesim 只读显示。

启动内容（全部使用同一个 ROS_DOMAIN_ID）：
  1. RealSense 相机驱动（src/start_box4d_camera.sh）
  2. Box4D 识别并发布位姿（src/box4d_ros.sh live --publish-poses）
  3. box_pose_control C++ 控制节点
  4. turtlesim + turtlesim_real_view 只读可视化

一个命令即可，所有子进程日志都带前缀输出在同一个终端；turtlesim 会另开窗口。
这些子进程默认 dry_run；要让真实底盘动，传 dry_run:=false serial_port:=/dev/ttyUSB0。
"""

import os
from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _as_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _launch_setup(context, *args, **kwargs):
    def value(name):
        return context.perform_substitution(LaunchConfiguration(name))

    repo = Path(value("repo_root")).expanduser().resolve()
    size = value("size")
    if size not in ("big", "small"):
        raise ValueError("size must be big or small")
    domain = value("domain").strip()
    try:
        domain_id = int(domain)
    except ValueError as error:
        raise ValueError(f"domain must be an integer: {domain}") from error
    if not 0 <= domain_id <= 232:
        raise ValueError("domain must be in [0, 232]")

    share = Path(get_package_share_directory("box_pose_control"))
    default_config = "box4d_d435i_yolo_live.json" if size == "small" else "box4d_d435i_live.json"
    config = Path(value("config") or f"src/config/{default_config}").expanduser()
    if not config.is_absolute():
        config = repo / config
    output = Path(value("output")).expanduser()
    if not output.is_absolute():
        output = repo / output

    display = _as_bool(value("display"))
    dry_run = _as_bool(value("dry_run"))
    start_camera = _as_bool(value("start_camera"))
    start_recognition = _as_bool(value("start_recognition"))
    with_view = _as_bool(value("with_view"))

    env = {"ROS_DOMAIN_ID": domain}
    actions = []

    if start_camera:
        actions.append(
            ExecuteProcess(
                cmd=[
                    "bash",
                    str(repo / "src/start_box4d_camera.sh"),
                    "--domain",
                    domain,
                    value("camera_profile"),
                ],
                additional_env=env,
                cwd=str(repo),
                output="screen",
                name="box4d_camera",
            )
        )

    if start_recognition:
        recognition_cmd = [
            "bash",
            str(repo / "src/box4d_ros.sh"),
            "--domain",
            domain,
            "live",
            "--size",
            size,
            "--config",
            str(config),
            "--auto-up",
            "--publish-poses",
            "--output",
            str(output),
            "--duration",
            value("duration"),
        ]
        if display:
            recognition_cmd += ["--display", "--display-smoothed", "--display-hz", "30"]
        actions.append(
            ExecuteProcess(
                cmd=recognition_cmd,
                additional_env=env,
                cwd=str(repo),
                output="screen",
                name="box4d_recognition",
            )
        )

    actions.append(
        Node(
            package="box_pose_control",
            executable="box_pose_control_node",
            name="box_pose_control",
            parameters=[
                str(share / "config" / "control.yaml"),
                {
                    "dry_run": dry_run,
                    "size": size,
                    "start_enabled": True,
                    "target_distance_m": float(value("target_distance_m")),
                    "target_lateral_m": float(value("target_lateral_m")),
                    "serial_port": value("serial_port"),
                    "max_forward_speed": float(value("max_forward_speed")),
                    "max_lateral_speed": float(value("max_lateral_speed")),
                },
            ],
            additional_env=env,
            output="screen",
        )
    )

    if with_view:
        actions.append(
            Node(
                package="turtlesim",
                executable="turtlesim_node",
                name="turtlesim",
                additional_env=env,
                output="screen",
            )
        )
        actions.append(
            Node(
                package="box_pose_control",
                executable="turtlesim_real_view",
                name="turtlesim_real_view",
                parameters=[
                    {
                        "meters_per_unit": float(value("view_meters_per_unit")),
                        "target_id": value("view_target_id"),
                        "size": size,
                        "spawn_x": float(value("view_spawn_x")),
                        "spawn_y": float(value("view_spawn_y")),
                        "spawn_theta": float(value("view_spawn_theta")),
                    }
                ],
                additional_env=env,
                output="screen",
            )
        )

    return actions


def generate_launch_description():
    share = Path(get_package_share_directory("box_pose_control"))
    default_repo = str(share.parents[3])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return LaunchDescription([
        DeclareLaunchArgument("repo_root", default_value=default_repo),
        DeclareLaunchArgument("domain", default_value="86"),
        DeclareLaunchArgument("camera_profile", default_value="speed"),
        DeclareLaunchArgument("start_camera", default_value="true"),
        DeclareLaunchArgument("start_recognition", default_value="true"),
        DeclareLaunchArgument(
            "config", default_value="",
            description="Empty selects geometry for big, bundled YOLO + geometry for small"
        ),
        DeclareLaunchArgument("duration", default_value="3600"),
        DeclareLaunchArgument("display", default_value="true"),
        DeclareLaunchArgument(
            "output", default_value=f"outputs/box4d_control_{timestamp}"
        ),
        DeclareLaunchArgument("with_view", default_value="true"),
        DeclareLaunchArgument("dry_run", default_value="true"),
        DeclareLaunchArgument("size", default_value="big", choices=["big", "small"]),
        DeclareLaunchArgument("serial_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("target_distance_m", default_value="0.30"),
        DeclareLaunchArgument("target_lateral_m", default_value="0.0"),
        DeclareLaunchArgument("max_forward_speed", default_value="0.10"),
        DeclareLaunchArgument("max_lateral_speed", default_value="0.10"),
        DeclareLaunchArgument("view_meters_per_unit", default_value="0.2"),
        DeclareLaunchArgument("view_target_id", default_value=""),
        DeclareLaunchArgument("view_spawn_x", default_value="1.0"),
        DeclareLaunchArgument("view_spawn_y", default_value="5.5"),
        DeclareLaunchArgument("view_spawn_theta", default_value="0.0"),
        OpaqueFunction(function=_launch_setup),
    ])
