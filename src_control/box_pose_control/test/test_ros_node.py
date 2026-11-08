"""真实 DDS + C++ 进程测试；底盘输出仅使用 dry_run 或伪终端。"""

import math
import os
from pathlib import Path
import pty
import signal
import subprocess
import tempfile
import time

import pytest
import rclpy
from geometry_msgs.msg import Twist, Vector3Stamped
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from rosgraph_msgs.msg import Clock
from std_srvs.srv import SetBool
from vision_msgs.msg import Detection3D, Detection3DArray


class Harness:
    def __init__(self, extra=()):
        # 独立 domain 和测试话题，避免测试消息进入已有识别/底盘链路。
        os.environ["ROS_DOMAIN_ID"] = "217"
        os.environ["ROS_LOCALHOST_ONLY"] = "1"
        rclpy.init()
        self.node = rclpy.create_node("box_control_test")
        self.directory = tempfile.TemporaryDirectory()
        self.log_path = Path(self.directory.name) / "node.log"
        self.log_file = self.log_path.open("w")
        executable = os.environ["NODE_EXECUTABLE"]
        self.process = subprocess.Popen([
            executable, "--ros-args",
            "-p", "detections_topic:=/box_control_test/detections",
            "-p", "up_topic:=/box_control_test/up",
            "-r", "/box_pose_control/cmd_vel:=/box_control_test/velocity",
            "-p", "serial_port:=/does-not-exist-dry-run-must-not-open",
            "-p", "size:=small",
            *extra,
        ], stdout=self.log_file, stderr=subprocess.STDOUT)
        self.commands = []
        self.node.create_subscription(Twist, "/box_control_test/velocity",
            lambda message: self.commands.append((time.monotonic(), message)), 50)
        self.detections = self.node.create_publisher(Detection3DArray, "/box_control_test/detections", 10)
        self.up = self.node.create_publisher(Vector3Stamped, "/box_control_test/up", 10)
        self.enabled = self.node.create_client(SetBool, "/box_pose_control/set_enabled")
        self.parameters = self.node.create_client(SetParameters, "/box_pose_control/set_parameters")
        assert self.enabled.wait_for_service(timeout_sec=5.0), self.logs()
        self.spin(0.3)
        assert self.process.poll() is None, self.logs()

    def logs(self):
        return self.log_path.read_text()

    def spin(self, duration):
        end = time.monotonic() + duration
        while time.monotonic() < end:
            rclpy.spin_once(self.node, timeout_sec=min(0.02, max(0.0, end - time.monotonic())))

    def publish(self, boxes=(("run:a", 0.0, 0.0, 0.6),), *, up=(0.0, -1.0, 0.0),
                frame="camera_color_optical_frame", stamp=None, up_frame=None, up_first=False, sizes=None):
        message = Detection3DArray()
        message.header.frame_id = frame
        message.header.stamp = self.node.get_clock().now().to_msg() if stamp is None else stamp
        for identity, x, y, z in boxes:
            detection = Detection3D()
            detection.header = message.header
            detection.id = identity
            detection.bbox.center.position.x = float(x)
            detection.bbox.center.position.y = float(y)
            detection.bbox.center.position.z = float(z)
            detection.bbox.center.orientation.w = 1.0
            edge = (sizes or {}).get(identity, 0.2)
            detection.bbox.size.x = detection.bbox.size.y = detection.bbox.size.z = edge
            message.detections.append(detection)
        vector = Vector3Stamped()
        vector.header.stamp = message.header.stamp
        vector.header.frame_id = frame if up_frame is None else up_frame
        if up is not None:
            vector.vector.x, vector.vector.y, vector.vector.z = up
        if up_first and up is not None:
            self.up.publish(vector)
        self.detections.publish(message)
        if not up_first and up is not None:
            self.up.publish(vector)
        return message.header.stamp

    def stream(self, duration=0.4, **kwargs):
        end = time.monotonic() + duration
        start = len(self.commands)
        while time.monotonic() < end:
            self.publish(**kwargs)
            self.spin(0.04)
        self.spin(0.12)
        return self.commands[start:]

    def call(self, client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=3.0)
        assert future.done(), self.logs()
        return future.result()

    def enable(self, enabled):
        response = self.call(self.enabled, SetBool.Request(data=enabled))
        assert response.success, response.message

    def distance(self, value, name="target_distance_m"):
        request = SetParameters.Request(parameters=[Parameter(name=name,
            value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=value))])
        return self.call(self.parameters, request).results[0]

    def stop(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=5.0)
        self.log_file.close()

    def close(self):
        self.stop()
        self.node.destroy_node()
        rclpy.shutdown()
        self.directory.cleanup()


@pytest.fixture
def harness():
    instance = Harness()
    try:
        yield instance
    finally:
        instance.close()


def last(harness):
    assert harness.commands
    return harness.commands[-1][1]


def stopped(command):
    return command.linear.x == 0.0 and command.linear.y == 0.0 and command.angular.z == 0.0


def test_dry_run_control_frequency_info_and_target_parameters(harness):
    h = harness
    assert stopped(last(h))
    samples = h.stream(duration=1.2)
    assert last(h).linear.x == pytest.approx(0.1)
    assert last(h).linear.y == 0.0
    periods = [b[0] - a[0] for a, b in zip(samples, samples[1:])]
    assert 0.085 < sum(periods) / len(periods) < 0.115
    logs = h.logs()
    rows = [line for line in logs.splitlines() if "mode=dry_run" in line]
    assert len(rows) >= len(samples) - 1
    assert "serial=not_opened" in logs
    assert "cmd_vx=" in logs and "cmd_vy=" in logs and "vw=0.000" in logs
    assert "启用串口失败" not in logs
    h.stream(boxes=(("run:a", 0.04, 0.0, 0.6),))
    assert last(h).linear.x == pytest.approx(0.0)
    assert last(h).linear.y < 0.0
    h.stream(boxes=(("run:a", 0.0, 0.0, 0.30),))
    assert stopped(last(h))
    assert h.distance(0.50).successful
    h.stream(boxes=(("run:a", 0.0, 0.0, 0.30),))
    assert last(h).linear.x < 0.0
    assert not h.distance(-0.1).successful
    h.enable(False)
    h.stream()
    assert stopped(last(h))
    h.enable(True)
    h.spin(0.15)
    assert stopped(last(h))


def test_lock_loss_recovery_and_freshness(harness):
    h = harness
    # B 虽然更近，但 A 更靠近光轴，首次锁 A。
    h.stream(boxes=(("run:b", 0.10, 0.0, 0.2), ("run:a", 0.0, 0.0, 0.6)))
    assert last(h).linear.x > 0.0
    assert "id=run:a" in h.logs()
    h.stream(boxes=(("run:b", 0.0, 0.0, 0.6),), up=None, duration=0.15)
    assert stopped(last(h)), "目标消失必须直接停车，不能等缺失的 UP"
    h.stream()
    assert last(h).linear.x > 0.0
    h.spin(0.65)
    assert stopped(last(h))
    stamp = h.publish()
    h.spin(0.15)
    # 重复时间戳不能不断刷新接收看门狗。
    h.stream(duration=0.65, stamp=stamp)
    assert stopped(last(h))
    h.stream(boxes=(), up=None, duration=0.15)
    assert stopped(last(h))
    h.enable(False)
    h.enable(True)
    h.stream(boxes=(("newrun:b", 0.0, 0.0, 0.2),))
    assert last(h).linear.x < 0.0
    assert "id=newrun:b" in h.logs()


def test_pairing_tilt_and_invalid_inputs(harness):
    h = harness
    h.stream(up=None)
    assert stopped(last(h))
    # 向下俯仰 60°：构造水平前方 0.38m、高度 0.1m 的方块，期望 vx=0.08。
    tilted_up = (0.0, -0.5, -math.sqrt(3.0) / 2.0)
    h.stream(boxes=(("run:a", 0.0, -0.38 * math.sqrt(3.0) / 2.0 - 0.05,
                     0.19 - 0.1 * math.sqrt(3.0) / 2.0),), up=tilted_up, up_first=True)
    assert last(h).linear.x == pytest.approx(0.08, abs=1e-6)
    h.stream(up=(0.0, 0.0, 1.0))
    assert stopped(last(h))
    h.stream(up=(float("nan"), -1.0, 0.0))
    assert stopped(last(h))
    h.stream(up_frame="wrong_frame")
    assert stopped(last(h))
    h.stream()
    assert last(h).linear.x > 0.0
    h.stream(boxes=(("run:a", float("nan"), 0.0, 0.6),))
    assert stopped(last(h))
    future = h.node.get_clock().now().to_msg()
    future.sec += 10
    h.stream(stamp=future)
    assert stopped(last(h))


def test_wall_timer_stops_when_simulation_clock_pauses():
    h = Harness(["-p", "use_sim_time:=true"])
    try:
        clock = h.node.create_publisher(Clock, "/clock", 10)
        clock_message = Clock()
        clock_message.clock.sec = 100
        for _ in range(10):
            clock.publish(clock_message)
            h.spin(0.04)
        # 控制器使用固定仿真时刻，测试端显式使用同一时间戳。
        stamp = clock_message.clock
        h.publish(stamp=stamp)
        h.spin(0.2)
        assert last(h).linear.x > 0.0
        before = len(h.commands)
        h.spin(0.7)
        assert len(h.commands) - before >= 5
        assert stopped(last(h))
        assert "pose_timeout" in h.logs()
    finally:
        h.close()


def test_serial_mode_sends_twist_protocol_and_stop_on_exit():
    master, slave = pty.openpty()
    path = os.ttyname(slave)
    os.set_blocking(master, False)
    h = Harness(["-p", "dry_run:=false", "-p", f"serial_port:={path}"])
    try:
        h.stream(duration=0.4)
        assert last(h).linear.x > 0.0
        h.stop()
        data = bytearray()
        while True:
            try:
                data.extend(os.read(master, 4096))
            except BlockingIOError:
                break
        assert data and len(data) % 10 == 0
        frames = [data[i:i + 10] for i in range(0, len(data), 10)]
        # 电控 Python 参考协议的完整帧，验证非零速度 CRC 和退出停车帧。
        moving_frame = bytes.fromhex("a5 64 00 00 00 00 00 00 91 ff")
        stop_frame = bytes.fromhex("a5 00 00 00 00 00 00 00 00 ff")
        assert moving_frame in frames
        assert all(frame in (moving_frame, stop_frame) for frame in frames)
        assert frames[-1] == stop_frame
        assert "serial=sent" in h.logs()
        assert h.process.returncode == 0
    finally:
        h.close()
        os.close(slave)
        os.close(master)


@pytest.mark.parametrize("size,edge,other", [("big", 0.35, 0.2), ("small", 0.2, 0.35)])
def test_selected_size_filters_before_lock_and_after_lock(size, edge, other):
    h = Harness(["-p", f"size:={size}"])
    try:
        boxes = (("wrong", 0.0, 0.0, 0.6), ("chosen", 0.04, 0.0, 0.6))
        h.stream(boxes=boxes, sizes={"wrong": other, "chosen": edge})
        assert last(h).linear.y < 0
        assert "id=chosen" in h.logs()
        assert f"size={size}" in h.logs()
        h.stream(boxes=(("chosen", 0.0, 0.0, 0.6),), sizes={"chosen": other}, up=None)
        assert stopped(last(h)), "同 ID 改成另一种尺寸也必须停车"
        h.stream(boxes=(("wrong", 0.0, 0.0, 0.6),), sizes={"wrong": edge})
        assert stopped(last(h)), "不能自动换成其他同尺寸目标"
        h.stream(boxes=(("chosen", 0.0, 0.0, 0.6),), sizes={"chosen": edge})
        assert last(h).linear.x > 0
        assert h.distance(0.08, "target_lateral_m").successful
        h.stream(boxes=(("chosen", -0.08, 0.4, 0.30),), sizes={"chosen": edge})
        assert stopped(last(h))
        assert not h.distance(float("nan"), "target_lateral_m").successful
    finally:
        h.close()
