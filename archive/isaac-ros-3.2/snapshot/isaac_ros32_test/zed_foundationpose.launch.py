"""Isaac ROS 3.2 FoundationPose estimation and tracking with the ZED bridge."""

from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


MESH = "/workspace/6Dpose/kfs_model/BlueTrueKFS13/BlueTrueKFS13.obj"
TEXTURE = "/workspace/6Dpose/kfs_model/BlueTrueKFS13/BlueTrueKFS13.png"
REFINE_MODEL = "/work/models/refine_model.onnx"
REFINE_ENGINE = "/work/engines/refine_trt_engine.plan"
SCORE_MODEL = "/work/models/score_model.onnx"
SCORE_ENGINE = "/work/engines/score_trt_engine.plan"


def generate_launch_description():
    selector = ComposableNode(
        name="selector",
        package="isaac_ros_foundationpose",
        plugin="nvidia::isaac_ros::foundationpose::Selector",
        parameters=[{"reset_period": 3600000}],
        remappings=[
            ("image", "/isaacfp/rgb"),
            ("depth_image", "/isaacfp/depth"),
            ("camera_info", "/isaacfp/camera_info"),
            ("segmentation", "/isaacfp/mask"),
        ],
    )

    refine = {
        "mesh_file_path": MESH,
        "texture_path": TEXTURE,
        "refine_model_file_path": REFINE_MODEL,
        "refine_engine_file_path": REFINE_ENGINE,
        "refine_input_tensor_names": ["input_tensor1", "input_tensor2"],
        "refine_input_binding_names": ["input1", "input2"],
        "refine_output_tensor_names": ["output_tensor1", "output_tensor2"],
        "refine_output_binding_names": ["output1", "output2"],
    }
    pose = ComposableNode(
        name="foundationpose",
        package="isaac_ros_foundationpose",
        plugin="nvidia::isaac_ros::foundationpose::FoundationPoseNode",
        parameters=[{
            **refine,
            # 首帧定位连续 refine 5 次；跟踪节点保留独立默认设置。
            "refine_iterations": 5,
            "score_model_file_path": SCORE_MODEL,
            "score_engine_file_path": SCORE_ENGINE,
            "score_input_tensor_names": ["input_tensor1", "input_tensor2"],
            "score_input_binding_names": ["input1", "input2"],
            "score_output_tensor_names": ["output_tensor"],
            "score_output_binding_names": ["output1"],
        }],
    )
    tracker = ComposableNode(
        name="foundationpose_tracker",
        package="isaac_ros_foundationpose",
        plugin="nvidia::isaac_ros::foundationpose::FoundationPoseTrackingNode",
        parameters=[refine],
    )
    container = ComposableNodeContainer(
        name="isaacfp_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[selector, pose, tracker],
        output="screen",
    )
    return LaunchDescription([container])
