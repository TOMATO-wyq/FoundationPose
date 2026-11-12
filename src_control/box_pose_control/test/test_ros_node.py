"""真实 DDS + C++ 进程测试；底盘输出仅使用 dry_run 或伪终端。"""

from array import array
import fcntl
import math
import os
from pathlib import Path
import pty
import signal
import subprocess
import tempfile
import termios
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
            "-p", "yaw_enabled:=false",
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
                frame="camera_color_optical_frame", stamp=None, up_frame=None, up_first=False, sizes=None, yaw_deg=None):
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
            if yaw_deg is not None:
                # R_camera_from_box: forward, left, UP；绕光学 -Y 转动 yaw。
                half = math.radians(yaw_deg) / 2
                c, sn = math.cos(half), math.sin(half)
                orientation = detection.bbox.center.orientation
                orientation.x = orientation.w = 0.5 * (c - sn)
                orientation.y = -0.5 * (c + sn)
                orientation.z = 0.5 * (c + sn)
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
    assert last(h).linear.x == pytest.approx(0.5)
    assert last(h).linear.y == 0.0
    periods = [b[0] - a[0] for a, b in zip(samples, samples[1:])]
    assert 0.085 < sum(periods) / len(periods) < 0.115
    logs = h.logs()
    rows = [line for line in logs.splitlines() if "mode=dry_run" in line]
    assert len(rows) >= len(samples) - 1
    assert "serial=not_opened" in logs
    assert "cmd_vx=" in logs and "cmd_vy=" in logs and "vw=0.000" in logs
    assert "forward_gain=4.000 lateral_gain=2.000" in logs
    assert "启用串口失败" not in logs
    h.stream(boxes=(("run:a", 0.04, 0.0, 0.6),))
    assert last(h).linear.x > 0.0
    assert last(h).linear.y == pytest.approx(-0.08)
    assert "state=adjusting_position" in h.logs()
    h.stream(boxes=(("run:a", 0.04, 0.0, 0.38),))
    assert last(h).linear.x == pytest.approx(0.32)
    assert last(h).linear.y == pytest.approx(-0.08)
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


@pytest.mark.parametrize("loss", ["empty", "timeout", "other_visible"])
def test_reacquires_new_id_after_loss_only_with_fresh_paired_observations(harness, loss):
    h = harness
    h.stream(duration=0.25)
    assert last(h).linear.x > 0.0
    replacement = (("newrun:b", 0.0, 0.0, 0.2),)
    if loss == "timeout":
        h.spin(0.65)
    else:
        h.stream(boxes=() if loss == "empty" else replacement, up=None, duration=0.15)
    assert stopped(last(h))
    # 即使等待时间已满足，新 ID 缺少同帧 UP 时仍必须停车。
    h.stream(boxes=replacement, up=None, duration=1.05)
    assert stopped(last(h))
    h.stream(boxes=replacement, up_first=True)
    assert last(h).linear.x < 0.0, h.logs()
    assert "目标重新锁定: old_id=run:a new_id=newrun:b" in h.logs()


def test_original_id_has_priority_when_it_returns_with_another_candidate(harness):
    h = harness
    original = ("run:a", 0.04, 0.0, 0.6)
    h.stream(boxes=(original,))
    assert last(h).linear.y < 0.0
    h.stream(boxes=(), duration=1.05)
    assert stopped(last(h))
    # 新 ID 更靠近光轴且排列在前，但旧 ID 在场时应优先恢复它。
    h.stream(boxes=(("run:b", 0.0, 0.0, 0.6), original))
    assert last(h).linear.y < 0.0
    assert "目标重新锁定" not in h.logs()


def test_pairing_tilt_and_invalid_inputs(harness):
    h = harness
    h.stream(up=None)
    assert stopped(last(h))
    # 向下俯仰 60°：构造水平前方 0.38m、高度 0.1m 的方块，期望 vx=0.32。
    tilted_up = (0.0, -0.5, -math.sqrt(3.0) / 2.0)
    h.stream(boxes=(("run:a", 0.0, -0.38 * math.sqrt(3.0) / 2.0 - 0.05,
                     0.19 - 0.1 * math.sqrt(3.0) / 2.0),), up=tilted_up, up_first=True)
    assert last(h).linear.x == pytest.approx(0.32, abs=1e-6)
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


@pytest.mark.parametrize("serial_args,baudrate", [
    pytest.param([], 115200, id="node-default"),
    pytest.param(["--params-file", str(Path(__file__).resolve().parents[1] / "config/control.yaml")],
                 115200, id="yaml-default"),
    pytest.param(["-p", "baudrate:=961200"], 961200, id="custom-baudrate"),
])
def test_serial_mode_sends_twist_protocol_and_stop_on_exit(serial_args, baudrate):
    master, slave = pty.openpty()
    path = os.ttyname(slave)
    os.set_blocking(master, False)
    # 使用节点名限定覆盖项，使其与 YAML 的 box_pose_control 作用域一致。
    h = Harness([*serial_args, "-p", "box_pose_control:size:=small",
                 "-p", "box_pose_control:camera_yaw_deg:=0.0",
                 "-p", "box_pose_control:dry_run:=false",
                 "-p", f"box_pose_control:serial_port:={path}",
                 "-p", "box_pose_control:detections_topic:=/box_control_test/detections",
                 "-p", "box_pose_control:up_topic:=/box_control_test/up"])
    try:
        # Linux TCGETS2：termios2 共 44 字节，最后两个 uint32 是实际输入/输出波特率。
        # 从 PTY 读回内核设置，避免只检查参数或日志而漏掉串口实际配置错误。
        settings = array("I", [0] * 11)
        fcntl.ioctl(slave, 0x802C542A, settings)
        assert list(settings[-2:]) == [baudrate, baudrate], h.logs()
        # 标准波特率沿用键盘端 pySerial 的 B115200 编码，自定义速率才用 BOTHER。
        assert settings[2] & termios.CBAUD == (
            termios.B115200 if baudrate == 115200 else 0o010000
        ), h.logs()
        h.stream(duration=0.4)
        assert last(h).linear.x > 0.0
        h.stream(boxes=(("run:a", 0.2, 0.0, 0.6),))
        assert last(h).linear.x == pytest.approx(0.5)
        assert last(h).linear.y == pytest.approx(-0.4)
        h.stream(boxes=(), duration=1.05)
        assert stopped(last(h))
        h.stream(boxes=(("newrun:b", 0.0, 0.0, 0.1),))
        assert last(h).linear.x < 0.0, h.logs()
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
        moving_frame = bytes.fromhex("a5 f4 01 00 00 00 00 00 e1 ff")
        diagonal_frame = bytes.fromhex("a5 f4 01 70 fe 00 00 00 b8 ff")
        reverse_frame = bytes.fromhex("a5 0c fe 00 00 00 00 00 19 ff")
        stop_frame = bytes.fromhex("a5 00 00 00 00 00 00 00 00 ff")
        assert moving_frame in frames
        assert diagonal_frame in frames
        assert reverse_frame in frames
        assert stop_frame in frames[frames.index(moving_frame) + 1:frames.index(reverse_frame)]
        assert all(frame in (moving_frame, diagonal_frame, reverse_frame, stop_frame) for frame in frames)
        assert frames[-1] == stop_frame
        assert "serial=sent" in h.logs()
        assert f"port={path} baudrate={baudrate} 8N1 frame_bytes=10" in h.logs()
        assert h.process.returncode == 0
    finally:
        h.close()
        os.close(slave)
        os.close(master)


class SerialCapture:
    """持续读回真实串口帧并记录单调时钟时间，用于验证动作间隔。"""

    def __init__(self, harness, master):
        self.harness = harness
        self.master = master
        self.buffer = bytearray()
        self.frames = []
        self.read()

    def read(self):
        while True:
            try:
                self.buffer.extend(os.read(self.master, 4096))
            except BlockingIOError:
                break
        while len(self.buffer) >= 10:
            frame = bytes(self.buffer[:10])
            del self.buffer[:10]
            assert frame[0] == 0xA5 and frame[9] == 0xFF
            crc = 0
            for value in frame[1:8]:
                crc ^= value
                for _ in range(8):
                    crc = ((crc << 1) ^ (0x07 if crc & 0x80 else 0)) & 0xFF
            assert frame[8] == crc
            self.frames.append((time.monotonic(), frame))

    def stream(self, duration, boxes=(("run:a", 0.0, 0.0, 0.30),)):
        end = time.monotonic() + duration
        while time.monotonic() < end:
            if boxes is not None:
                self.harness.publish(boxes=boxes)
            self.harness.spin(0.01)
            self.read()

    @property
    def events(self):
        return [(stamp, frame) for stamp, frame in self.frames if frame[7] != 0]


@pytest.fixture
def serial_capture():
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    h = None
    try:
        h = Harness(["-p", "dry_run:=false", "-p", f"serial_port:={os.ttyname(slave)}"])
        yield h, SerialCapture(h, master)
    finally:
        if h is not None:
            h.close()
        os.close(slave)
        os.close(master)


@pytest.mark.parametrize("after_grip", ["in_position", "target_lost", "pose_timeout"])
def test_grip_is_one_shot_and_resets_after_one_second(serial_capture, after_grip):
    h, capture = serial_capture
    # 仅距离到位、仅横向到位都不能触发夹取。
    capture.stream(0.25, boxes=(("run:a", 0.08, 0.0, 0.30),))
    capture.stream(0.25, boxes=(("run:a", 0.0, 0.0, 0.60),))
    assert not capture.events
    assert any(frame[1:5] != bytes(4) for _, frame in capture.frames)
    capture.stream(0.25)
    assert [frame[7] for _, frame in capture.events] == [1], h.logs()
    boxes = {"in_position": (("run:a", 0.0, 0.0, 0.30),),
             "target_lost": (), "pose_timeout": None}[after_grip]
    capture.stream(1.4, boxes=boxes)
    events = capture.events
    assert [frame[7] for _, frame in events] == [1, 0xFF], h.logs()
    assert 0.98 <= events[1][0] - events[0][0] <= 1.30
    assert events[0][1] == bytes.fromhex("a5 00 00 00 00 00 00 01 07 ff")
    assert events[1][1] == bytes.fromhex("a5 00 00 00 00 00 00 ff f3 ff")
    between = [frame for stamp, frame in capture.frames if events[0][0] < stamp < events[1][0]]
    assert len(between) >= 8 and all(frame[7] == 0 for frame in between)
    # 重复开启、同一目标丢失重现、离开后重新到位，都不能重复夹取。
    h.enable(True)
    capture.stream(0.25, boxes=(("run:a", 0.0, 0.0, 0.60),))
    capture.stream(0.35)
    assert len(capture.events) == 2
    assert capture.frames[-1][1][7] == 0
    assert "grip_mode=1 serial=sent" in h.logs()
    assert "grip_mode=-1 serial=sent" in h.logs()


@pytest.mark.parametrize("rearm", ["enable", "target", "new_id"])
def test_new_control_task_allows_another_grip(serial_capture, rearm):
    h, capture = serial_capture
    capture.stream(1.4)
    assert [frame[7] for _, frame in capture.events] == [1, 0xFF]
    boxes = (("run:a", 0.0, 0.0, 0.30),)
    if rearm == "enable":
        h.enable(False)
        h.enable(True)
        capture.stream(0.15, boxes=None)
        assert len(capture.events) == 2
    elif rearm == "target":
        assert h.distance(0.50).successful
        capture.stream(0.25)
        assert len(capture.events) == 2
        boxes = (("run:a", 0.0, 0.0, 0.50),)
    else:
        capture.stream(1.1, boxes=())
        boxes = (("run:b", 0.0, 0.0, 0.30),)
    capture.stream(1.4, boxes=boxes)
    assert [frame[7] for _, frame in capture.events] == [1, 0xFF, 1, 0xFF], h.logs()
    assert capture.frames[-1][1][7] == 0


def test_disabling_cancels_pending_grip_reset(serial_capture):
    h, capture = serial_capture
    capture.stream(0.25)
    assert [frame[7] for _, frame in capture.events] == [1]
    h.enable(False)
    capture.read()
    assert capture.frames[-1][1] == bytes.fromhex("a5 00 00 00 00 00 00 00 00 ff")
    capture.stream(1.2, boxes=None)
    assert [frame[7] for _, frame in capture.events] == [1]
    h.enable(True)
    capture.stream(1.4)
    assert [frame[7] for _, frame in capture.events] == [1, 1, 0xFF]


def test_mount_yaw_controls_chassis_axes_and_reports_transformed_position():
    h = Harness(["-p", "camera_yaw_deg:=36.86989765", "-p", "target_distance_m:=0.5"])
    try:
        # 相机光轴上 0.5 m 的点，在底盘轴向为前方 0.4 m、左方 0.3 m。
        h.stream(boxes=(("run:a", 0.0, 0.0, 0.5),))
        assert last(h).linear.x == pytest.approx(-0.4)
        assert last(h).linear.y == pytest.approx(0.5)
        assert "plane_x=0.400 plane_y=0.300" in h.logs()
        assert "camera_yaw_deg=36.86990" in h.logs()
        # 对准相机原点沿车头方向的 0.5 m 目标点，不叠加未知的安装位置偏移。
        h.stream(boxes=(("run:a", 0.3, 0.0, 0.4),))
        assert stopped(last(h))
        # 继续沿车头方向远离 0.15 m，只应产生底盘前进速度。
        h.stream(boxes=(("run:a", 0.39, 0.0, 0.52),))
        assert last(h).linear.x == pytest.approx(0.5)
        assert last(h).linear.y == pytest.approx(0.0)
    finally:
        h.close()


@pytest.mark.parametrize("size,edge,other", [("big", 0.35, 0.2), ("small", 0.2, 0.35)])
def test_selected_size_filters_before_lock_and_after_lock(size, edge, other):
    h = Harness(["-p", f"size:={size}"])
    try:
        boxes = (("wrong", 0.0, 0.0, 0.6), ("chosen", 0.04, 0.0, 0.6))
        h.stream(boxes=boxes, sizes={"wrong": other, "chosen": edge})
        assert last(h).linear.y < 0
        assert "id=chosen" in h.logs()
        assert f"size={size}" in h.logs()
        h.stream(boxes=(("chosen", 0.0, 0.0, 0.6),), sizes={"chosen": other}, up=None, duration=0.15)
        assert stopped(last(h)), "同 ID 改成另一种尺寸也必须停车"
        h.stream(boxes=(("wrong", 0.0, 0.0, 0.6),), sizes={"wrong": edge}, duration=0.15)
        assert stopped(last(h)), "等待期内不能切换到其他同尺寸目标"
        h.stream(boxes=(("chosen", 0.0, 0.0, 0.6),), sizes={"chosen": edge})
        assert last(h).linear.x > 0
        assert h.distance(0.08, "target_lateral_m").successful
        h.stream(boxes=(("chosen", -0.08, 0.4, 0.30),), sizes={"chosen": edge})
        assert stopped(last(h))
        assert not h.distance(float("nan"), "target_lateral_m").successful
        # 超过等待期也不能接管错误尺寸；随后正确尺寸的新 ID 可以自动恢复。
        h.stream(boxes=(("new", 0.0, 0.0, 0.6),), sizes={"new": other}, duration=1.05)
        assert stopped(last(h))
        h.stream(boxes=(("new", -0.08, 0.0, 0.6),), sizes={"new": edge})
        assert last(h).linear.x > 0.0
    finally:
        h.close()


def test_yaw_closed_loop_output_deadband_jumps_and_timeout():
    h = Harness(["-p", "yaw_enabled:=true"])
    try:
        centered = (("run:a", 0.0, 0.0, 0.30),)
        h.stream(duration=0.6, boxes=centered, yaw_deg=20)
        assert last(h).angular.z > 0
        assert last(h).linear.x == last(h).linear.y == 0
        assert "grip_mode=1" not in h.logs()
        h.publish(boxes=centered, yaw_deg=-20)
        h.spin(0.12)
        # 突变要重新确认，不能立即反向转动。
        assert last(h).angular.z == 0
        h.stream(duration=0.5, boxes=centered, yaw_deg=-20)
        assert last(h).angular.z < 0
        h.stream(duration=0.4, boxes=centered, yaw_deg=-4)
        assert stopped(last(h))
        assert "grip_mode=1" not in h.logs(), "稳定停留时间尚未满足时不得夹取"
        h.stream(duration=0.6, boxes=centered, yaw_deg=-4)
        assert "grip_mode=1" in h.logs()
        h.stream(duration=0.4, boxes=centered, yaw_deg=-7)
        assert last(h).angular.z == 0
        h.stream(duration=0.4, boxes=centered, yaw_deg=-9)
        assert last(h).angular.z < 0
        h.spin(0.65)
        assert stopped(last(h))
    finally:
        h.close()


@pytest.mark.parametrize("serial_sign,serial_scale", [(None, 1.0), (-1.0, 1.0), (1.0, 1.5)])
def test_yaw_serial_direction_scale_crc_and_stable_grip(serial_sign, serial_scale):
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    h = None
    try:
        sign_args = [] if serial_sign is None else ["-p", f"serial_yaw_sign:={serial_sign}"]
        h = Harness(["-p", "yaw_enabled:=true", "-p", "dry_run:=false",
                     "-p", f"serial_port:={os.ttyname(slave)}",
                     *sign_args, "-p", f"serial_yaw_scale:={serial_scale}"])
        if serial_sign is None:
            serial_sign = 1.0  # 不传参数时，也必须向角差减小的方向转。
        capture = SerialCapture(h, master)
        centered = (("run:a", 0.0, 0.0, 0.30),)
        for angle, expected_sign in ((20, serial_sign), (-20, -serial_sign)):
            end = time.monotonic() + 0.8
            while time.monotonic() < end:
                h.publish(boxes=centered, yaw_deg=angle)
                h.spin(0.03)
                capture.read()
            wire = int.from_bytes(capture.frames[-1][1][5:7], "little", signed=True)
            assert wire * expected_sign > 0
            assert abs(wire) == int(4 * serial_scale)
            assert not capture.events
        h.stream(duration=0.8, boxes=centered, yaw_deg=0)
        capture.read()
        # 在姿态已确认后仍需连续到位 0.4s。
        h.stream(duration=0.6, boxes=centered, yaw_deg=0)
        capture.read()
        modes = [frame[7] for _, frame in capture.events]
        assert modes in ([1], [1, 0xFF]), h.logs()
        assert capture.events[0][1][1:7] == bytes(6)
        h.enable(False)
        capture.read()
        assert capture.frames[-1][1] == bytes.fromhex("a5 00 00 00 00 00 00 00 00 ff")
    finally:
        if h is not None:
            h.close()
        os.close(slave)
        os.close(master)
