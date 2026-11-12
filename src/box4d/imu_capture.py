"""Native ROS IMU sidecar, recorded independently of the RGB-D writer."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from .io import dump_json


class _TransformPending(RuntimeError):
    """ROS graph discovery has not supplied a required fixed transform yet."""


def _read_bag_summary(path, topics, target_frame):
    """Read small IMU messages after clean shutdown, preserving source clocks."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu
    from tf2_msgs.msg import TFMessage

    if not (path / "metadata.yaml").is_file():
        raise RuntimeError("IMU bag has no finalized metadata.yaml")
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    counts, stamps, frame_ids, edges = {}, {}, {}, set()
    observations = {topic: dict(first_stamp_ns=None, last_stamp_ns=None,
                               duplicate_or_old_source_stamps=0, nonpositive_source_stamps=0,
                               nonfinite_vectors=0) for topic in topics}
    while reader.has_next():
        topic, payload, _ = reader.read_next()
        counts[topic] = counts.get(topic, 0) + 1
        if topic in topics:
            message = deserialize_message(payload, Imu)
            stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            row = observations[topic]
            if row["last_stamp_ns"] is not None and stamp <= row["last_stamp_ns"]:
                row["duplicate_or_old_source_stamps"] += 1
            if stamp <= 0:
                row["nonpositive_source_stamps"] += 1
            row["first_stamp_ns"] = stamp if row["first_stamp_ns"] is None else row["first_stamp_ns"]
            row["last_stamp_ns"] = stamp
            vector = message.linear_acceleration if topic == topics[0] else message.angular_velocity
            if not all(math.isfinite(value) for value in (vector.x, vector.y, vector.z)):
                row["nonfinite_vectors"] += 1
            bounds = stamps.setdefault(topic, [stamp, stamp])
            bounds[0], bounds[1] = min(bounds[0], stamp), max(bounds[1], stamp)
            frame_ids.setdefault(topic, set()).add(message.header.frame_id)
        elif topic == "/tf_static":
            message = deserialize_message(payload, TFMessage)
            edges.update((item.header.frame_id, item.child_frame_id) for item in message.transforms)
    del reader  # Humble SequentialReader has no explicit close().
    if any(not counts.get(topic) for topic in topics):
        raise RuntimeError("IMU bag is missing acceleration or gyro messages; check enabled camera streams")
    graph = {}
    for a, b in edges:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    connected, pending = set(), [target_frame]
    while pending:
        current = pending.pop()
        if current not in connected:
            connected.add(current)
            pending.extend(graph.get(current, ()))
    if any(not frames or not frames.issubset(connected) for frames in frame_ids.values()):
        raise RuntimeError("IMU bag /tf_static does not connect IMU optical frames to RGB optical frame")
    return dict(counts=counts, source_stamp_ranges_ns=stamps,
                frame_ids={topic: sorted(frames) for topic, frames in frame_ids.items()},
                source_observations=observations,
                units={topics[0]: "m/s^2", topics[1]: "rad/s"},
                tf_static_connects_to_rgb=True)


class ImuBagRecorder:
    """Capture acceleration, angular velocity and their fixed camera extrinsics.

    The source already owns ROS spinning. Probe callbacks store only counters;
    rosbag2 performs serialization and disk writes in a separate process.
    """

    stream_stall_timeout_s = 1.0

    def __init__(self, source, output, accel_topic="/camera/camera/accel/sample",
                 gyro_topic="/camera/camera/gyro/sample"):
        self.source, self.output = source, Path(output)
        self.topics = (accel_topic, gyro_topic)
        self.bag = self.output / "imu_bag"
        # Humble's ros2 bag record has no --node-name option.
        self.node_name = "rosbag2_recorder"
        self._existing_subscription_gids = {}
        self.process = self.log = self.listener = self.buffer = None
        self.subscriptions = []
        self.condition = threading.Condition()
        self.probes = {topic: dict(valid_messages=0, rejected=0, frame_id=None,
                                  first_stamp_ns=None, last_stamp_ns=None,
                                  last_received_monotonic=None) for topic in self.topics}
        self.reference = self.summary = self.failure = None
        self.closed = False

    def _receive(self, topic, message, component):
        stamp = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
        vector = getattr(message, component)
        with self.condition:
            row = self.probes[topic]
            if (stamp <= (row["last_stamp_ns"] or 0) or not message.header.frame_id
                or (row["frame_id"] is not None and row["frame_id"] != message.header.frame_id)
                or not all(math.isfinite(v) for v in (vector.x, vector.y, vector.z))):
                row["rejected"] += 1
                return
            row["valid_messages"] += 1
            row["frame_id"] = message.header.frame_id
            row["first_stamp_ns"] = row["first_stamp_ns"] or stamp
            row["last_stamp_ns"] = stamp
            row["last_received_monotonic"] = time.monotonic()
            self.condition.notify_all()

    def _setup_probes(self):
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Imu
        from tf2_ros import Buffer, TransformListener

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self.source.node, spin_thread=False)
        for topic, component in zip(self.topics, ("linear_acceleration", "angular_velocity")):
            def receive(message, topic=topic, component=component):
                self._receive(topic, message, component)
            self.subscriptions.append(self.source.node.create_subscription(Imu, topic, receive, qos_profile_sensor_data))

    def _ready(self):
        with self.condition:
            have_messages = all(row["valid_messages"] >= 2 for row in self.probes.values())
        if not have_messages:
            return False
        # A live probe alone cannot prove rosbag2 finished discovering its topics.
        return all(any(info.node_name == self.node_name
                       and tuple(info.endpoint_gid) not in self._existing_subscription_gids.get(topic, set())
                       for info in self.source.node.get_subscriptions_info_by_topic(topic))
                   for topic in (*self.topics, "/tf_static"))

    def _transforms(self, target):
        from rclpy.time import Time
        from scipy.spatial.transform import Rotation
        from tf2_ros import TransformException

        transforms = {}
        for topic in self.topics:
            origin = self.probes[topic]["frame_id"]
            try:
                tf = self.buffer.lookup_transform(target, origin, Time()).transform
            except TransformException as error:
                raise _TransformPending(str(error)) from error
            q, t = tf.rotation, tf.translation
            quaternion = [q.x, q.y, q.z, q.w]
            transforms[topic] = dict(
                source_frame_id=origin, target_frame_id=target,
                translation_m=[t.x, t.y, t.z], quaternion_xyzw=quaternion,
                rotation=Rotation.from_quat(quaternion).as_matrix().tolist(),
            )
        return transforms

    def start(self, timeout=10):
        if self.process is not None or self.closed:
            raise RuntimeError("IMU recorder can only be started once")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("IMU startup timeout must be positive and finite")
        if self.bag.exists():
            raise FileExistsError(f"IMU bag already exists: {self.bag}")
        target = self.source.latest.camera.frame_id
        qos = {topic: dict(reliability="best_effort", durability="volatile", history="keep_last", depth=1000)
               for topic in self.topics}
        qos["/tf_static"] = dict(reliability="reliable", durability="transient_local", history="keep_last", depth=100)
        qos_path = self.output / "imu_qos.yaml"
        # JSON is valid YAML and avoids adding a second configuration serializer.
        qos_path.write_text(json.dumps(qos, indent=2) + "\n")
        self.log = (self.output / "imu_record.log").open("w")
        try:
            # The CLI uses a fixed node name. An already-running recorder must
            # not satisfy readiness for this process before it subscribes.
            self._existing_subscription_gids = {
                topic: {tuple(info.endpoint_gid)
                        for info in self.source.node.get_subscriptions_info_by_topic(topic)}
                for topic in (*self.topics, "/tf_static")
            }
            self.process = subprocess.Popen(
                ["ros2", "bag", "record", "-s", "sqlite3", "-o", str(self.bag),
                 "--max-cache-size", "1048576",
                 "--qos-profile-overrides-path", str(qos_path), *self.topics, "/tf_static"],
                stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self._setup_probes()
            deadline = time.monotonic() + timeout
            tf_error = None
            while True:
                self.check()
                if self._ready():
                    try:
                        transforms = self._transforms(target)
                        break
                    except _TransformPending as error:
                        # TF discovery can finish after the IMU subscriptions.
                        tf_error = str(error)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    with self.condition:
                        counts = {topic: row["valid_messages"] for topic, row in self.probes.items()}
                    raise RuntimeError(
                        f"IMU startup timed out: fresh messages={counts}; TF={tf_error or 'not ready'}. "
                        "Enable both enable_accel and enable_gyro in the existing camera driver, "
                        "use its ROS_DOMAIN_ID, and check /tf_static. "
                        f"Recorder log: {self.output / 'imu_record.log'}"
                    )
                with self.condition:
                    self.condition.wait(timeout=min(.05, remaining))
            with self.condition:
                probes = deepcopy(self.probes)
            self.reference = dict(
                format="box4d-imu-sidecar", version=1, storage_id="sqlite3",
                bag="imu_bag", topics=list(self.topics), tf_topic="/tf_static",
                camera_frame_id=target, extrinsics=transforms, startup_probe=probes,
                units={self.topics[0]: "m/s^2", self.topics[1]: "rad/s"},
                stream_stall_timeout_s=self.stream_stall_timeout_s,
                raw_messages_preserved=True, live_up_modified=False,
                note="Native source stamps, frame IDs and covariances; no IMU fusion or UP update.",
            )
            dump_json(self.output / "imu_reference.json", self.reference)
            return deepcopy(self.reference)
        except Exception as error:
            self.failure = str(error)
            try:
                self.close()
            except RuntimeError:
                pass
            raise

    def check(self):
        if self.process is not None and self.process.poll() is not None:
            self.failure = f"IMU recorder exited unexpectedly ({self.process.returncode}); see {self.output / 'imu_record.log'}"
        if self.reference is not None and self.failure is None:
            now = time.monotonic()
            with self.condition:
                stalled = [topic for topic, row in self.probes.items()
                           if row["last_received_monotonic"] is None
                           or now - row["last_received_monotonic"] > self.stream_stall_timeout_s]
            if stalled:
                self.failure = f"IMU stream stalled for over {self.stream_stall_timeout_s:g}s: {stalled}; check camera/USB and enabled accel/gyro streams"
        if self.failure is not None:
            raise RuntimeError(self.failure)

    def _stop(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGINT)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=2)
                raise RuntimeError("IMU recorder did not finalize after SIGINT; bag may be incomplete")
        if self.process.returncode != 0:
            raise RuntimeError(f"IMU recorder exited with code {self.process.returncode}; see imu_record.log")

    def _release(self):
        for subscription in self.subscriptions:
            self.source.node.destroy_subscription(subscription)
        self.subscriptions.clear()
        if self.listener is not None:
            self.listener.unregister()
            self.listener = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def close(self):
        if self.closed:
            if self.failure is not None:
                raise RuntimeError(self.failure)
            return deepcopy(self.summary)
        summary = dict(complete=False, counts={})
        try:
            # Also cover a stream dying just before the user finishes capture.
            try:
                self.check()
            except RuntimeError as error:
                self.failure = str(error)
            self._stop()
            if self.failure is not None:
                raise RuntimeError(self.failure)
            if self.reference is None:
                raise RuntimeError("IMU recorder never completed startup")
            summary.update(_read_bag_summary(self.bag, self.topics, self.reference["camera_frame_id"]))
            summary["complete"] = True
        except Exception as error:
            self.failure = str(error)
            summary["error"] = self.failure
        finally:
            with self.condition:
                summary["runtime_probe"] = deepcopy(self.probes)
            self._release()
            self.closed = True
            self.summary = summary
            dump_json(self.output / "imu_summary.json", summary)
        if self.failure is not None:
            raise RuntimeError(self.failure)
        return deepcopy(summary)
