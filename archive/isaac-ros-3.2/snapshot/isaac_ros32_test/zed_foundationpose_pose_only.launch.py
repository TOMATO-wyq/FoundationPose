"""Load only Isaac ROS FoundationPose for an 8 GB GPU memory probe."""

from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    pose = ComposableNode(
        name="foundationpose",
        package="isaac_ros_foundationpose",
        plugin="nvidia::isaac_ros::foundationpose::FoundationPoseNode",
        parameters=[{
            "mesh_file_path": "/workspace/6Dpose/kfs_model/BlueTrueKFS13/BlueTrueKFS13.obj",
            "texture_path": "/workspace/6Dpose/kfs_model/BlueTrueKFS13/BlueTrueKFS13.png",
            # 首帧定位连续 refine 5 次；不是失败后重发 5 帧。
            "refine_iterations": 5,
            "refine_model_file_path": "/work/models/refine_model.onnx",
            "refine_engine_file_path": "/work/engines/refine_trt_engine.plan",
            "refine_input_tensor_names": ["input_tensor1", "input_tensor2"],
            "refine_input_binding_names": ["input1", "input2"],
            "refine_output_tensor_names": ["output_tensor1", "output_tensor2"],
            "refine_output_binding_names": ["output1", "output2"],
            "score_model_file_path": "/work/models/score_model.onnx",
            "score_engine_file_path": "/work/engines/score_trt_engine.plan",
            "score_input_tensor_names": ["input_tensor1", "input_tensor2"],
            "score_input_binding_names": ["input1", "input2"],
            "score_output_tensor_names": ["output_tensor"],
            "score_output_binding_names": ["output1"],
        }],
    )
    return LaunchDescription([ComposableNodeContainer(
        name="isaacfp_pose_only_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[pose],
        output="screen",
    )])
