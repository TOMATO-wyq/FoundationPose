"""Run only the Isaac ROS FoundationPose tracking component on an 8 GB GPU."""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # tracker 的 .so 直接依赖此 GXF 库；先定位文件，再把目录传给容器进程。
    libraries = list(Path("/opt/ros/humble").rglob("libgxf_isaac_message_compositor.so"))
    if not libraries:
        raise RuntimeError(
            "libgxf_isaac_message_compositor.so is absent from /opt/ros/humble; "
            "check whether ros-humble-gxf-isaac-message-compositor is installed")
    library_dir = str(libraries[0].parent)
    old_path = os.environ.get("LD_LIBRARY_PATH", "")
    loader_path = library_dir + (":" + old_path if old_path else "")

    # 首帧位姿由 seed_tracking_bridge.py 提供；这里不加载占显存的 score 引擎。
    tracker = ComposableNode(
        name="foundationpose_tracker",
        package="isaac_ros_foundationpose",
        plugin="nvidia::isaac_ros::foundationpose::FoundationPoseTrackingNode",
        parameters=[{
            # 临时诊断开关：在节点创建时生效，用于区分输入、推理和输出通知停顿。
            "enable_all_diagnostics": ParameterValue(
                LaunchConfiguration("diagnostics"), value_type=bool),
            "extension_log_level": ParameterValue(
                LaunchConfiguration("extension_log_level"), value_type=int),
            # NITROS 启动时会按这些 QoS 参数重设 GXF 接收容量；只改 YAML 不够。
            "tracking/image_qos_depth": ParameterValue(
                LaunchConfiguration("input_qos_depth"), value_type=int),
            "tracking/depth_image_qos_depth": ParameterValue(
                LaunchConfiguration("input_qos_depth"), value_type=int),
            "tracking/camera_info_qos_depth": ParameterValue(
                LaunchConfiguration("input_qos_depth"), value_type=int),
            "tracking/pose_input_qos_depth": ParameterValue(
                LaunchConfiguration("input_qos_depth"), value_type=int),
            "mesh_file_path": "/workspace/6Dpose/kfs_model/BlueTrueKFS13/BlueTrueKFS13.obj",
            "refine_model_file_path": "/work/models/refine_model.onnx",
            "refine_engine_file_path": "/work/engines/refine_trt_engine.plan",
            "refine_input_tensor_names": ["input_tensor1", "input_tensor2"],
            "refine_input_binding_names": ["input1", "input2"],
            "refine_output_tensor_names": ["output_tensor1", "output_tensor2"],
            "refine_output_binding_names": ["output1", "output2"],
        }],
    )
    return LaunchDescription([
                              DeclareLaunchArgument("diagnostics", default_value="false"),
                              DeclareLaunchArgument("extension_log_level", default_value="2"),
                              DeclareLaunchArgument("ros_log_level", default_value="info"),
                              DeclareLaunchArgument("input_qos_depth", default_value="10"),
                              # 单线程 executor 对照仅改变 ROS 回调调度；GXF/GPU 仍按原图运行。
                              DeclareLaunchArgument("container_executable",
                                  # 采用已跑完隔帧录像的调度配置；仍可显式切回 mt 做对照。
                                  default_value="component_container",
                                  choices=["component_container_mt", "component_container"]),
                              SetEnvironmentVariable(name="LD_LIBRARY_PATH", value=loader_path),
                              ComposableNodeContainer(
        name="isaacfp_tracker_container",
        namespace="",
        package="rclcpp_components",
        executable=LaunchConfiguration("container_executable"),
        arguments=["--ros-args", "--log-level", LaunchConfiguration("ros_log_level")],
        composable_node_descriptions=[tracker],
        output="screen",
    )])
