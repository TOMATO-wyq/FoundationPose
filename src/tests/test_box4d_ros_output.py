"""ROS messages preserve current raw geometry and the live IMU reference."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from box4d.cli import live
from box4d.config import Config
from box4d.ros_output import RosPosePublisher

try:
    from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
    from vision_msgs.msg import Detection3DArray

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


def observation(track_id=7):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.4, 1.1]).as_matrix()
    pose[:3, 3] = [0.04, -0.02, 0.6]
    return dict(
        valid=True, track_id=track_id, size_m=0.2, T_camera_from_box=pose.tolist(),
        center_camera_m=pose[:3, 3].tolist(), center_camera_smoothed_m=[8.0, 9.0, 10.0],
        yaw_u_rad=None, yaw_reference_ready=False,
    )


def result(objects=None, stamp_ns=12_345_678_901):
    return dict(
        stamp_ns=stamp_ns, frame_id="camera_color_optical_frame", calibration_id="camera-test",
        objects=[observation()] if objects is None else objects, up_camera=[0.0, -1.0, 0.0],
        timing=dict(processing_ms=1.0),
    )


class FakeNode:
    def __init__(self):
        self.publishers = {}

    def create_publisher(self, message_type, topic, qos):
        publisher = SimpleNamespace(message_type=message_type, qos=qos, messages=[])
        publisher.publish = publisher.messages.append
        self.publishers[topic] = publisher
        return publisher


@unittest.skipUnless(ROS_AVAILABLE, "source /opt/ros/humble/setup.bash to test real ROS messages")
class RosPosePublisherTests(unittest.TestCase):
    def test_raw_pose_complete_rotation_size_and_original_header(self):
        node = FakeNode()
        publisher = RosPosePublisher(node, [0.0, -1.0, 0.0])
        current = result()
        publisher.publish(current)
        message = publisher.detections.messages[0]
        self.assertIsInstance(message, Detection3DArray)
        self.assertEqual(message.header.frame_id, current["frame_id"])
        self.assertEqual(message.header.stamp.sec, 12)
        self.assertEqual(message.header.stamp.nanosec, 345_678_901)
        detection = message.detections[0]
        self.assertEqual(detection.header, message.header)
        self.assertEqual(detection.id, f"{publisher.process_id}:7")
        position = detection.bbox.center.position
        np.testing.assert_allclose([position.x, position.y, position.z], [0.04, -0.02, 0.6])
        q = detection.bbox.center.orientation
        np.testing.assert_allclose(
            Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
            np.asarray(current["objects"][0]["T_camera_from_box"])[:3, :3],
        )
        self.assertAlmostEqual(np.linalg.norm([q.x, q.y, q.z, q.w]), 1.0)
        self.assertEqual([detection.bbox.size.x, detection.bbox.size.y, detection.bbox.size.z], [0.2] * 3)
        self.assertEqual(publisher.up.messages[0].header, message.header)
        for output in node.publishers.values():
            self.assertEqual(output.qos.depth, 10)
            self.assertEqual(output.qos.reliability, ReliabilityPolicy.RELIABLE)
            self.assertEqual(output.qos.durability, DurabilityPolicy.VOLATILE)

    def test_empty_or_invalid_observations_publish_empty_array_with_current_stamp(self):
        publisher = RosPosePublisher(FakeNode(), [0.0, -1.0, 0.0])
        publisher.publish(result())
        publisher.publish(result([], 13_000_000_001))
        invalid = observation()
        invalid["valid"] = False
        publisher.publish(result([invalid], 14_000_000_002))
        self.assertEqual([len(m.detections) for m in publisher.detections.messages], [1, 0, 0])
        self.assertEqual(publisher.detections.messages[-1].header.stamp.sec, 14)
        self.assertEqual(publisher.up.messages[-1].header, publisher.detections.messages[-1].header)

    def test_ids_stay_stable_within_run_and_are_distinct_after_restart(self):
        first, second = RosPosePublisher(FakeNode()), RosPosePublisher(FakeNode())
        first.publish(result())
        first.publish(result(stamp_ns=13_000_000_000))
        second.publish(result())
        first_id = first.detections.messages[0].detections[0].id
        self.assertEqual(first_id, first.detections.messages[1].detections[0].id)
        self.assertNotEqual(first_id, second.detections.messages[0].detections[0].id)

    def test_prediction_only_frame_does_not_publish_a_visual_measurement(self):
        publisher = RosPosePublisher(FakeNode(), [0.0, -1.0, 0.0])
        current = result([])
        prediction = observation()
        prediction.update(valid=False, status="predicted", source="kf_prediction")
        current["predicted_objects"] = [prediction]
        publisher.publish(current)
        self.assertEqual(publisher.detections.messages[0].detections, [])
        self.assertEqual(publisher.up.messages[0].header, publisher.detections.messages[0].header)

    def test_up_is_a_snapshot_and_result_up_does_not_override_it(self):
        up = [0.0, -1.0, 0.0]
        publisher = RosPosePublisher(FakeNode(), up)
        up[:] = [1.0, 0.0, 0.0]
        current = result()
        current["up_camera"] = up
        publisher.publish(current)
        vector = publisher.up.messages[0].vector
        self.assertEqual([vector.x, vector.y, vector.z], [0.0, -1.0, 0.0])
        no_calibration = RosPosePublisher(FakeNode())
        no_calibration.publish(current)
        self.assertEqual(no_calibration.up.messages, [])


class LivePublishingTests(unittest.TestCase):
    def run_live(self, *, auto_up=False, publish=True):
        cfg = Config(up_camera=[0.0, -1.0, 0.0])
        current = result()
        source, pipeline, publisher = Mock(), Mock(), Mock()
        camera = SimpleNamespace(
            frame_id=current["frame_id"], calibration_id=current["calibration_id"], to_dict=lambda: {}
        )
        frame = SimpleNamespace(camera=camera, sequence=1, stamp_ns=current["stamp_ns"], received_monotonic_ns=0)
        source.get.side_effect = [frame] * (2 if auto_up else 1) + [KeyboardInterrupt()]
        source.received_count = 1
        source.statistics.return_value = {}
        source.estimate_up.return_value = dict(
            status="stationary_estimate", camera_frame_id=current["frame_id"], up_camera=[0.0, -0.8, -0.6]
        )
        pipeline.process.return_value = (current, None)
        args = ["live", "--output", "unused-output"]
        if auto_up:
            args.append("--auto-up")
        if publish:
            args.append("--publish-poses")
        with ExitStack() as stack:
            stack.enter_context(patch("sys.argv", args))
            stack.enter_context(patch.object(live.Config, "load", return_value=cfg))
            stack.enter_context(patch.object(live, "RosSource", return_value=source))
            stack.enter_context(patch.object(live, "make_pipeline", return_value=pipeline))
            stack.enter_context(patch.object(live, "RunWriter"))
            stack.enter_context(patch.object(live, "dump_json"))
            stack.enter_context(patch.object(live, "environment", return_value={}))
            publisher_type = stack.enter_context(patch("box4d.ros_output.RosPosePublisher", return_value=publisher))
            stack.enter_context(redirect_stdout(io.StringIO()))
            live.main()
        source.close.assert_called_once()
        return publisher_type, publisher, source, current

    def test_live_auto_up_is_passed_and_publishing_occurs_after_tracking(self):
        publisher_type, publisher, source, current = self.run_live(auto_up=True)
        publisher_type.assert_called_once_with(source.node, (0.0, -0.8, -0.6))
        publisher.publish.assert_called_once_with(current)
        self.assertEqual(current["objects"][0]["track_id"], 1)

    def test_configured_up_is_not_claimed_as_live_calibration(self):
        publisher_type, _, source, _ = self.run_live()
        publisher_type.assert_called_once_with(source.node, None)

    def test_default_does_not_construct_publisher(self):
        publisher_type, _, _, _ = self.run_live(publish=False)
        publisher_type.assert_not_called()

    def test_publishing_requires_tracking_before_camera_is_opened(self):
        with patch("sys.argv", ["live", "--output", "unused-output", "--publish-poses"]), patch.object(
            live.Config, "load", return_value=Config(tracking=False)
        ), patch.object(live, "RosSource") as source, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                live.main()
        self.assertEqual(error.exception.code, 2)
        source.assert_not_called()


if __name__ == "__main__":
    unittest.main()
