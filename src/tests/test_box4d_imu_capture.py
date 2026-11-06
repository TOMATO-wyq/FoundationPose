"""IMU sidecar lifecycle and evidence checks without ROS nodes or recording."""

from contextlib import ExitStack
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from box4d.imu_capture import ImuBagRecorder, _read_bag_summary


def message(stamp, frame="imu_optical", vector=(0., 0., 9.8)):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp // 1_000_000_000, nanosec=stamp % 1_000_000_000), frame_id=frame),
        linear_acceleration=SimpleNamespace(x=vector[0], y=vector[1], z=vector[2]),
        angular_velocity=SimpleNamespace(x=vector[0], y=vector[1], z=vector[2]),
    )


class Process:
    pid = 98765

    def __init__(self):
        self.returncode = None
        self.wait_calls = []

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.wait_calls.append(timeout)
        self.returncode = 0
        return 0


class ImuRecorderTests(unittest.TestCase):
    def setup_recorder(self, output, ready=True):
        source = Mock()
        source.latest.camera.frame_id = "camera_color_optical_frame"
        recorder = ImuBagRecorder(source, output)
        source.node.get_subscriptions_info_by_topic.return_value = [SimpleNamespace(node_name=recorder.node_name)]

        def probes():
            recorder.subscriptions = ["accel_probe", "gyro_probe"]
            recorder.listener = Mock()
            if ready:
                for topic in recorder.topics:
                    recorder._receive(topic, message(1_000_000_000), "linear_acceleration")
                    recorder._receive(topic, message(1_010_000_000), "linear_acceleration")

        recorder._setup_probes = Mock(side_effect=probes)
        recorder._transforms = Mock(return_value={topic: dict(source_frame_id="imu_optical") for topic in recorder.topics})
        return recorder, source

    def run_started(self, stack, output, ready=True):
        recorder, source = self.setup_recorder(output, ready=ready)
        process = Process()
        spawn = stack.enter_context(patch("box4d.imu_capture.subprocess.Popen", return_value=process))
        kill = stack.enter_context(patch("box4d.imu_capture.os.killpg"))
        summary = dict(counts={recorder.topics[0]: 30, recorder.topics[1]: 60, "/tf_static": 1},
                       source_stamp_ranges_ns={topic: [100, 200] for topic in recorder.topics}, tf_static_connects_to_rgb=True)
        read = stack.enter_context(patch("box4d.imu_capture._read_bag_summary", return_value=summary))
        return recorder, source, process, spawn, kill, read

    def test_start_checks_real_messages_and_graph_before_success_and_retains_raw_bag_options(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            recorder, source, process, spawn, kill, read = self.run_started(stack, Path(temporary))
            reference = recorder.start()
            args, kwargs = spawn.call_args
            command = args[0]
            self.assertEqual(command[:3], ["ros2", "bag", "record"])
            self.assertIn("sqlite3", command)
            self.assertEqual(command[-3:], [*recorder.topics, "/tf_static"])
            self.assertNotIn("/camera/camera/rgbd", command)
            self.assertTrue(kwargs["start_new_session"])
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            qos = json.loads((Path(temporary) / "imu_qos.yaml").read_text())
            self.assertEqual(qos[recorder.topics[0]]["reliability"], "best_effort")
            self.assertEqual(qos["/tf_static"]["durability"], "transient_local")
            self.assertEqual(reference["camera_frame_id"], "camera_color_optical_frame")
            self.assertTrue(reference["raw_messages_preserved"])
            self.assertFalse(reference["live_up_modified"])
            self.assertTrue((Path(temporary) / "imu_reference.json").is_file())
            recorder.close()

    def test_close_sends_sigint_waits_verifies_bag_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            recorder, source, process, _, kill, read = self.run_started(stack, Path(temporary))
            recorder.start()
            listener = recorder.listener
            summary = recorder.close()
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["counts"][recorder.topics[0]], 30)
            kill.assert_called_once_with(process.pid, signal.SIGINT)
            self.assertEqual(process.wait_calls, [10])
            self.assertEqual(source.node.destroy_subscription.call_count, 2)
            listener.unregister.assert_called_once()
            self.assertIsNone(recorder.log)
            summary["counts"].clear()
            self.assertTrue(recorder.close()["counts"])
            read.assert_called_once()

    def test_fresh_probe_requires_two_unique_finite_messages_and_stable_frame(self):
        recorder = ImuBagRecorder(Mock(), Path("unused"))
        topic = recorder.topics[0]
        for value in (message(100), message(100), message(101, vector=(0, float("nan"), 0)),
                      message(102, frame="different"), message(103)):
            recorder._receive(topic, value, "linear_acceleration")
        self.assertEqual(recorder.probes[topic]["valid_messages"], 2)
        self.assertEqual(recorder.probes[topic]["rejected"], 3)
        self.assertEqual(recorder.probes[topic]["first_stamp_ns"], 100)
        self.assertEqual(recorder.probes[topic]["last_stamp_ns"], 103)

    def test_timeout_is_actionable_and_cleans_up_process_and_probes(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            recorder, source, process, _, kill, read = self.run_started(stack, Path(temporary), ready=False)
            stack.enter_context(patch("box4d.imu_capture.time.monotonic", side_effect=[0., 11.]))
            with self.assertRaisesRegex(RuntimeError, "enable_accel and enable_gyro"):
                recorder.start(timeout=10)
            self.assertTrue(recorder.closed)
            self.assertEqual(source.node.destroy_subscription.call_count, 2)
            kill.assert_called_once_with(process.pid, signal.SIGINT)
            self.assertFalse(json.loads((Path(temporary) / "imu_summary.json").read_text())["complete"])
            read.assert_not_called()
            with self.assertRaises(RuntimeError):
                recorder.close()

    def test_check_detects_subprocess_exit_and_close_cannot_report_success(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            recorder, source, process, _, _, read = self.run_started(stack, Path(temporary))
            recorder.start()
            process.returncode = 7
            with self.assertRaisesRegex(RuntimeError, "exited unexpectedly"):
                recorder.check()
            with self.assertRaisesRegex(RuntimeError, "code 7"):
                recorder.close()
            self.assertFalse(json.loads((Path(temporary) / "imu_summary.json").read_text())["complete"])
            read.assert_not_called()

    def test_missing_recorded_stream_or_extrinsics_is_a_failed_capture(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            recorder, source, _, _, _, read = self.run_started(stack, Path(temporary))
            recorder.start()
            read.side_effect = RuntimeError("missing gyro messages")
            with self.assertRaisesRegex(RuntimeError, "missing gyro"):
                recorder.close()
            self.assertEqual(source.node.destroy_subscription.call_count, 2)
            self.assertFalse(recorder.summary["complete"])

    def test_live_rosbag_process_cannot_hide_a_stalled_gyro_stream(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            recorder, source, process, _, kill, read = self.run_started(stack, Path(temporary))
            recorder.start()
            recorder.probes[recorder.topics[0]]["last_received_monotonic"] = 20.
            recorder.probes[recorder.topics[1]]["last_received_monotonic"] = 18.
            stack.enter_context(patch("box4d.imu_capture.time.monotonic", return_value=20.))
            with self.assertRaisesRegex(RuntimeError, "stream stalled.*gyro"):
                recorder.check()
            self.assertIsNone(process.poll())
            with self.assertRaisesRegex(RuntimeError, "stream stalled"):
                recorder.close()
            self.assertFalse(recorder.summary["complete"])
            self.assertEqual(recorder.summary["runtime_probe"][recorder.topics[1]]["valid_messages"], 2)
            self.assertEqual(recorder.summary["runtime_probe"][recorder.topics[1]]["rejected"], 0)
            self.assertEqual(source.node.destroy_subscription.call_count, 2)
            kill.assert_called_once_with(process.pid, signal.SIGINT)
            read.assert_not_called()

    def test_shutdown_timeout_escalates_but_never_claims_complete(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            recorder, _, process, _, kill, read = self.run_started(stack, Path(temporary))
            recorder.start()
            process.wait = Mock(side_effect=[subprocess.TimeoutExpired("ros2", 10), 0])
            with self.assertRaisesRegex(RuntimeError, "did not finalize"):
                recorder.close()
            self.assertEqual([call.args[1] for call in kill.call_args_list], [signal.SIGINT, signal.SIGTERM])
            read.assert_not_called()
            self.assertFalse(recorder.summary["complete"])


class BagEvidenceTests(unittest.TestCase):
    def fake_ros(self, records, messages):
        pending = list(records)
        reader = Mock()
        reader.has_next.side_effect = lambda: bool(pending)
        reader.read_next.side_effect = lambda: pending.pop(0)
        modules = {
            "rosbag2_py": SimpleNamespace(SequentialReader=lambda: reader, StorageOptions=lambda **kw: kw, ConverterOptions=lambda *a: a),
            "rclpy": SimpleNamespace(),
            "rclpy.serialization": SimpleNamespace(deserialize_message=lambda data, typ: messages[data]),
            "sensor_msgs": SimpleNamespace(), "sensor_msgs.msg": SimpleNamespace(Imu=object),
            "tf2_msgs": SimpleNamespace(), "tf2_msgs.msg": SimpleNamespace(TFMessage=object),
        }
        return patch.dict(sys.modules, modules)

    def test_bag_counts_and_ranges_use_original_message_stamps_and_require_tf_chain(self):
        topics = ("/accel", "/gyro")
        transforms = [SimpleNamespace(header=SimpleNamespace(frame_id="camera_link"), child_frame_id=child)
                      for child in ("camera_color_optical_frame", "imu_optical")]
        messages = {"a": message(100), "b": message(200), "tf": SimpleNamespace(transforms=transforms)}
        records = [("/accel", "a", 99999), ("/gyro", "b", 999999), ("/tf_static", "tf", 9999999)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "metadata.yaml").write_text("finalized")
            with self.fake_ros(records, messages):
                result = _read_bag_summary(path, topics, "camera_color_optical_frame")
            self.assertEqual(result["source_stamp_ranges_ns"], {"/accel": [100, 100], "/gyro": [200, 200]})
            self.assertEqual(result["units"], {"/accel": "m/s^2", "/gyro": "rad/s"})
            self.assertTrue(result["tf_static_connects_to_rgb"])
            with self.fake_ros(records[:-1], messages):
                with self.assertRaisesRegex(RuntimeError, "does not connect"):
                    _read_bag_summary(path, topics, "camera_color_optical_frame")
            with self.fake_ros([records[0], records[-1]], messages):
                with self.assertRaisesRegex(RuntimeError, "missing acceleration or gyro"):
                    _read_bag_summary(path, topics, "camera_color_optical_frame")

    def test_bad_source_observations_are_counted_without_deleting_raw_messages(self):
        topics = ("/accel", "/gyro")
        transforms = [SimpleNamespace(header=SimpleNamespace(frame_id="camera_color_optical_frame"), child_frame_id="imu_optical")]
        messages = {
            "a": message(100), "duplicate": message(100), "old": message(90),
            "bad": message(110, vector=(float("nan"), 0., 0.)),
            "g": message(200), "tf": SimpleNamespace(transforms=transforms),
        }
        records = [("/accel", value, index) for index, value in enumerate(("a", "duplicate", "old", "bad"))]
        records += [("/gyro", "g", 5), ("/tf_static", "tf", 6)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "metadata.yaml").write_text("finalized")
            with self.fake_ros(records, messages):
                result = _read_bag_summary(path, topics, "camera_color_optical_frame")
        self.assertEqual(result["counts"]["/accel"], 4)
        row = result["source_observations"]["/accel"]
        self.assertEqual(row["first_stamp_ns"], 100)
        self.assertEqual(row["last_stamp_ns"], 110)
        self.assertEqual(row["duplicate_or_old_source_stamps"], 2)
        self.assertEqual(row["nonfinite_vectors"], 1)
        self.assertEqual(result["source_stamp_ranges_ns"]["/accel"], [90, 110])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
