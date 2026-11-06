"""Optional ROS2 input. Offline users do not need ROS or a camera installed."""

import os
import queue
from collections import deque
import threading
import time

import numpy as np

from .geometry import Camera, fit_plane
from .io import Frame, DatasetWriter


def stamp_ns(message):
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def frame_from_message(message, bridge, sequence, depth_scale=0.001, max_sync_ns=0, rectified=False):
    received = time.monotonic_ns()
    color, depth_message, info = message.rgb, message.depth, message.rgb_camera_info
    delta = abs(stamp_ns(color) - stamp_ns(depth_message))
    if delta > max_sync_ns:
        raise ValueError(f"RGB/depth source stamp mismatch: {delta}ns > {max_sync_ns}ns")
    if not color.header.frame_id or color.header.frame_id != depth_message.header.frame_id:
        raise ValueError("RGB/depth optical frame_id mismatch; enable align_depth")
    if info.header.frame_id and info.header.frame_id != color.header.frame_id:
        raise ValueError("CameraInfo does not describe RGB optical frame")
    if (info.width, info.height) != (color.width, color.height):
        raise ValueError("RGB CameraInfo dimensions mismatch")
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth scale must be positive finite meters per raw unit")
    k, d, model = np.asarray(info.k).reshape(3, 3), list(info.d), info.distortion_model
    if rectified:
        k = np.asarray(info.p).reshape(3, 4)[:, :3]
        d, model = [], "none"
    camera = Camera(info.width, info.height, k, d, model, color.header.frame_id, rectified)
    rgb = bridge.imgmsg_to_cv2(color, "rgb8")
    raw = bridge.imgmsg_to_cv2(depth_message, "passthrough")
    encoding = depth_message.encoding.upper()
    if encoding == "16UC1":
        depth = raw.astype(np.float32) * depth_scale
    elif encoding == "32FC1":
        depth = raw.astype(np.float32)
    else:
        raise ValueError(f"unsupported depth encoding: {encoding}")
    return Frame(rgb, depth, camera, stamp_ns(color), sequence, received)


class RosSource:
    def __init__(
        self, topic="/camera/camera/rgbd", depth_scale=0.001, max_sync_ms=0.0, rectified=False, imu_topic=None
    ):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from cv_bridge import CvBridge
        from realsense2_camera_msgs.msg import RGBD

        self.rclpy = rclpy
        self.topic = topic
        self.condition = threading.Condition()
        self.latest = None
        self.received_count = self.rejected_count = self.duplicate_count = 0
        self.source_stamps = []
        self._last_stamp = -1
        self.error = None
        self.owns_context = not rclpy.ok()
        if self.owns_context:
            rclpy.init()
        self.node = Node("box4d_cpu_rgbd")
        bridge = CvBridge()

        def callback(message):
            try:
                frame = frame_from_message(
                    message, bridge, self.received_count + 1, depth_scale, int(max_sync_ms * 1e6), rectified
                )
                if frame.stamp_ns <= self._last_stamp:
                    self.duplicate_count += 1
                    return
                self._last_stamp = frame.stamp_ns
                with self.condition:
                    self.received_count += 1
                    self.source_stamps.append(frame.stamp_ns)
                    if len(self.source_stamps) > 7200:
                        self.source_stamps = self.source_stamps[-7200:]
                    self.latest = frame
                    self.condition.notify_all()
            except (ValueError, RuntimeError) as error:
                self.rejected_count += 1
                self.error = str(error)
                self.node.get_logger().error(str(error), throttle_duration_sec=2.0)

        self.subscription = self.node.create_subscription(
            RGBD, topic, callback, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        )
        self.imu_condition = threading.Condition()
        self.imu_topic = imu_topic
        self.imu_samples = deque(maxlen=2048)
        self.imu_received_count = self.imu_duplicate_count = 0
        self.imu_buffer = self.imu_listener = self.imu_subscription = None
        if imu_topic is not None:
            from sensor_msgs.msg import Imu
            from rclpy.qos import qos_profile_sensor_data
            from tf2_ros import Buffer, TransformListener

            self.imu_buffer = Buffer()
            self.imu_listener = TransformListener(self.imu_buffer, self.node)

            def receive_imu(message):
                a = message.linear_acceleration
                row = (time.monotonic(), stamp_ns(message), message.header.frame_id, [a.x, a.y, a.z])
                with self.imu_condition:
                    self.imu_received_count += 1
                    if self.imu_samples and row[1] <= self.imu_samples[-1][1]:
                        self.imu_duplicate_count += 1
                        return
                    self.imu_samples.append(row)
                    self.imu_condition.notify_all()

            self.imu_subscription = self.node.create_subscription(
                Imu, imu_topic, receive_imu, qos_profile_sensor_data
            )
        self.thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.thread.start()

    def get(self, previous_sequence=-1, timeout=5.0):
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.latest is not None and self.latest.sequence > previous_sequence, timeout
            )
            if not ready:
                publishers = self.node.count_publishers(self.topic)
                detail = (
                    f"no new synchronized RGB-D on {self.topic} "
                    f"(ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')}) in {timeout:g}s. "
                    f"Publishers={publishers}, received_unique={self.received_count}, "
                    f"rejected={self.rejected_count}, duplicate_or_old={self.duplicate_count}; "
                    f"last rejection: {self.error}. "
                )
                if not publishers:
                    detail += (
                        "No RGB-D publisher found. Use the same ROS_DOMAIN_ID for the camera "
                        "and recognition, check --topic, and enable_rgbd:=true in the camera driver."
                    )
                elif self.error is not None:
                    detail += (
                        "RGB-D data was rejected; check the rejection above, synchronized "
                        "RGB/depth timestamps, and align_depth.enable:=true."
                    )
                elif self.duplicate_count:
                    detail += "Check that the camera publishes increasing source timestamps."
                else:
                    detail += (
                        "A publisher exists but no fresh usable RGB-D arrived. "
                        "Check the camera stream/USB logs and publisher/subscriber QoS."
                    )
                raise TimeoutError(detail)
            return self.latest

    def statistics(self):
        stamps = self.source_stamps
        fps = (len(stamps) - 1) * 1e9 / (stamps[-1] - stamps[0]) if len(stamps) > 1 else None
        return dict(
            received_unique_rgbd=self.received_count,
            rejected=self.rejected_count,
            duplicate_or_old=self.duplicate_count,
            observed_source_fps=fps,
            last_error=self.error,
        )

    def estimate_up(self, target_frame, duration=2.0, timeout=10.0):
        """Measure fresh stationary acceleration; never silently reuse an old UP."""
        from scipy.spatial.transform import Rotation
        from rclpy.duration import Duration
        from .reference import stationary_up

        if self.imu_buffer is None or duration < 1 or timeout <= duration:
            raise ValueError("live UP requires an IMU subscription and timeout > duration >= 1 second")
        begin = time.monotonic()
        with self.imu_condition:
            initial_received = self.imu_received_count
            initial_duplicates = self.imu_duplicate_count
            while True:
                rows = [r for r in self.imu_samples if r[0] >= begin]
                span = (rows[-1][1] - rows[0][1]) / 1e9 if len(rows) > 1 else 0.0
                if time.monotonic() - begin >= duration and len(rows) >= 50 and span >= duration * 0.8:
                    break
                remaining = timeout - (time.monotonic() - begin)
                if remaining <= 0:
                    publishers = self.node.count_publishers(self.imu_topic)
                    received = self.imu_received_count - initial_received
                    duplicates = self.imu_duplicate_count - initial_duplicates
                    detail = (
                        f"insufficient fresh acceleration on {self.imu_topic} "
                        f"(ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')}): "
                        f"{len(rows)} unique samples, {span:.3f}s source span in {timeout:g}s; "
                        f"need >=50 samples spanning >={duration*.8:.3f}s. "
                        f"Publishers={publishers}, received={received}, duplicate_or_old={duplicates}. "
                    )
                    if not publishers:
                        detail += (
                            "No IMU publisher found; enable_accel:=true is required. "
                            "Check the camera's ROS domain and --imu-topic."
                        )
                    elif not received:
                        detail += (
                            "An IMU publisher exists but no messages arrived; enable_accel=true "
                            "does not guarantee a working stream. Check camera/USB/HID logs. "
                            "If RealSense reports 'Motion Module failure', use its hw_reset "
                            "service or reconnect USB, then retry --auto-up."
                        )
                    else:
                        detail += (
                            "Check the IMU sample rate and increasing source timestamps; "
                            "increase --timeout if startup is slow."
                        )
                    raise ValueError(detail)
                self.imu_condition.wait(timeout=min(0.1, remaining))
        frames = {r[2] for r in rows}
        if len(frames) != 1 or not next(iter(frames)):
            raise ValueError("missing or changing acceleration optical frame")
        source_frame = next(iter(frames))
        tf = self.imu_buffer.lookup_transform(
            target_frame, source_frame, self.rclpy.time.Time(), timeout=Duration(seconds=2.0)
        ).transform
        q = tf.rotation
        quaternion = [q.x, q.y, q.z, q.w]
        rotation = Rotation.from_quat(quaternion).as_matrix()
        up, quality = stationary_up([r[3] for r in rows], rotation)
        return dict(
            status="stationary_estimate",
            source="live_accelerometer_with_ros_tf",
            imu_topic=self.imu_topic,
            camera_frame_id=target_frame,
            accel_frame_id=source_frame,
            up_camera=up,
            sample_span_s=span,
            first_stamp_ns=rows[0][1],
            last_stamp_ns=rows[-1][1],
            camera_from_accel_quaternion_xyzw=quaternion,
            camera_from_accel_rotation=rotation.tolist(),
            quality=quality,
            valid_only_while_camera_mount_is_unchanged=True,
            driver_note="Factory IMU bias/extrinsics are not independently calibrated; provisional reference.",
        )

    def close(self):
        self.node.destroy_node()
        if self.owns_context and self.rclpy.ok():
            self.rclpy.shutdown()
        self.thread.join(timeout=2.0)


def up_from_plane_roi(frame, roi, threshold=0.008):
    x0, y0, x1, y1 = map(int, roi)
    if not (0 <= x0 < x1 <= frame.camera.width and 0 <= y0 < y1 <= frame.camera.height):
        raise ValueError("horizontal plane ROI lies outside image")
    yy, xx = np.mgrid[y0:y1:3, x0:x1:3]
    depth = frame.depth[yy, xx]
    valid = np.isfinite(depth) & (depth > 0.1)
    uv = np.column_stack((xx[valid], yy[valid]))
    points = frame.camera.rays(uv) * depth[valid, None]
    if len(points) > 600:
        points = points[np.linspace(0, len(points) - 1, 600).astype(int)]
    plane = fit_plane(points, threshold)
    if plane.support < 0.85 or plane.rmse_m > threshold / 2:
        raise ValueError("selected horizontal reference region is noisy or not planar")
    # User must select a known horizontal floor/table BELOW the camera, not a side face.
    return plane.normal.tolist(), dict(
        roi_xyxy=list(roi),
        plane_rmse_m=plane.rmse_m,
        plane_support=plane.support,
        source="user_selected_horizontal_plane",
    )


class AsyncRecorder:
    def __init__(self, output, camera, up_camera, metadata=None):
        self.writer = DatasetWriter(output, camera, up_camera, metadata=metadata)
        self.queue = queue.Queue(maxsize=4)
        self.dropped = 0
        self.error = None
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stopping.is_set() or not self.queue.empty():
                try:
                    frame = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                self.writer.write(frame)
        except Exception as error:
            self.error = error
            self.stopping.set()

    def submit(self, frame):
        if self.error:
            raise RuntimeError(f"recording failed: {self.error}") from self.error
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            self.dropped += 1

    def close(self, complete=True, **metadata):
        self.stopping.set()
        self.thread.join(timeout=10.0)
        if self.thread.is_alive():
            # Do not close a file that is still being written from another thread.
            raise RuntimeError("recording writer did not drain; dataset remains incomplete")
        self.writer.close(
            complete=complete and self.error is None, dropped_writer_frames=self.dropped, **metadata
        )
        if self.error:
            raise RuntimeError(f"recording failed: {self.error}") from self.error
