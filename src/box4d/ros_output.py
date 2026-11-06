"""Optional ROS 2 output for the current, unsmoothed Box4D observations."""

from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation


class RosPosePublisher:
    def __init__(self, node, up_camera=None):
        # 只在启用发布时加载 ROS，离线识别仍然不依赖 ROS 环境。
        from geometry_msgs.msg import Vector3Stamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from vision_msgs.msg import Detection3D, Detection3DArray

        self.Detection3D = Detection3D
        self.Detection3DArray = Detection3DArray
        self.Vector3Stamped = Vector3Stamped
        self.process_id = uuid4().hex
        # 固定本次现场标定，不跟随调试界面对 cfg.up_camera 的后续修改。
        self.up_camera = None if up_camera is None else tuple(float(v) for v in up_camera)
        qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE
        )
        self.detections = node.create_publisher(Detection3DArray, "/box4d/detections", qos)
        self.up = node.create_publisher(Vector3Stamped, "/box4d/up_camera", qos)

    def publish(self, result):
        message = self.Detection3DArray()
        message.header.frame_id = result["frame_id"]
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(int(result["stamp_ns"]), 1_000_000_000)
        for observation in result["objects"]:
            if not observation["valid"]:
                continue
            detection = self.Detection3D()
            detection.header = message.header
            detection.id = f"{self.process_id}:{observation['track_id']}"
            # T 的平移就是方块几何中心；不使用仅供显示的平滑位置或 yaw。
            pose = np.asarray(observation["T_camera_from_box"], dtype=float)
            center = detection.bbox.center
            center.position.x, center.position.y, center.position.z = map(float, pose[:3, 3])
            quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()
            center.orientation.x, center.orientation.y, center.orientation.z, center.orientation.w = map(
                float, quaternion
            )
            detection.bbox.size.x = detection.bbox.size.y = detection.bbox.size.z = float(observation["size_m"])
            message.detections.append(detection)

        # 无目标也发布空数组，让控制端立即停车，不必等待超时。
        self.detections.publish(message)
        if self.up_camera is not None:
            up_message = self.Vector3Stamped()
            up_message.header = message.header
            up_message.vector.x, up_message.vector.y, up_message.vector.z = self.up_camera
            self.up.publish(up_message)
