"""Recording preview must keep original RGB-D intact and close cleanly."""

from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from box4d.cli import capture
from box4d.io import Frame
from box4d.synthetic import default_camera


def frame(sequence=0):
    camera = default_camera(32, 24)
    rgb = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    depth = np.linspace(.2, 2., 24 * 32, dtype=np.float32).reshape(24, 32)
    depth[0, :3] = [0., np.nan, np.inf]
    return Frame(rgb, depth, camera, 1_000_000_000 + sequence * 16_666_667, sequence)


class CaptureMainTests(unittest.TestCase):
    def recorder(self):
        value = Mock()
        value.writer = SimpleNamespace(count=0)
        value.dropped = 0

        def submit(_frame):
            value.writer.count += 1

        value.submit.side_effect = submit
        return value

    def assert_complete_close(self, recorder, statistics):
        recorder.close.assert_called_once()
        args, kwargs = recorder.close.call_args
        self.assertTrue(args[0] if args else kwargs["complete"])
        self.assertEqual(kwargs["camera"], statistics)

    def test_preview_stop_finishes_recording_and_closes_all_resources(self):
        source, recorder, preview = Mock(), self.recorder(), Mock()
        current = frame()
        source.get.return_value = current
        statistics = dict(received_unique_rgbd=1, rejected=0, duplicate_or_old=0)
        source.statistics.return_value = statistics
        preview.show.return_value = True
        with patch.object(capture, "RosSource", return_value=source), patch.object(
            capture, "AsyncRecorder", return_value=recorder
        ), patch.object(capture, "CapturePreview", return_value=preview) as preview_type, patch(
            "sys.argv", ["capture", "unused-output", "--duration", "60", "--max-frames", "1000", "--display"]
        ), redirect_stdout(io.StringIO()):
            capture.main()
        preview_type.assert_called_once_with(15.)
        recorder.submit.assert_called_once_with(current)
        preview.show.assert_called_once()
        self.assert_complete_close(recorder, statistics)
        source.close.assert_called_once()
        preview.close.assert_called_once()

    def test_without_display_records_normally_without_constructing_preview(self):
        source, recorder = Mock(), self.recorder()
        frames = [frame(0), frame(1)]
        source.get.side_effect = frames
        statistics = dict(received_unique_rgbd=2, rejected=0, duplicate_or_old=0)
        source.statistics.return_value = statistics
        with patch.object(capture, "RosSource", return_value=source), patch.object(
            capture, "AsyncRecorder", return_value=recorder
        ), patch.object(capture, "CapturePreview") as preview_type, patch.object(
            capture, "ImuBagRecorder", create=True
        ) as imu_type, patch(
            "sys.argv", ["capture", "unused-output", "--duration", "60", "--max-frames", "2"]
        ), redirect_stdout(io.StringIO()):
            capture.main()
        preview_type.assert_not_called()
        imu_type.assert_not_called()
        self.assertEqual(recorder.submit.call_count, 2)
        self.assertEqual([call.args[0] for call in recorder.submit.call_args_list], frames)
        self.assert_complete_close(recorder, statistics)
        source.close.assert_called_once()


class CaptureImuMainTests(unittest.TestCase):
    def resources(self):
        source = Mock()
        frames = [frame(0), frame(10), frame(11)]
        source.get.side_effect = frames
        source.statistics.return_value = dict(received_unique_rgbd=3, rejected=0, duplicate_or_old=0)
        recorder = CaptureMainTests.recorder(self)
        imu = Mock()
        imu.start.return_value = dict(source="rosbag2", accel_topic="/imu/accel_test", gyro_topic="/imu/gyro_test")
        imu.close.return_value = dict(complete=True, accel_messages=12, gyro_messages=30)
        preview = Mock()
        preview.show.return_value = False
        return source, frames, recorder, imu, preview

    def enter(self, stack, source, recorder, imu, preview, *, max_frames=2):
        source_type = stack.enter_context(patch.object(capture, "RosSource", return_value=source))
        recorder_type = stack.enter_context(patch.object(capture, "AsyncRecorder", return_value=recorder))
        imu_type = stack.enter_context(patch.object(capture, "ImuBagRecorder", return_value=imu, create=True))
        stack.enter_context(patch.object(capture, "CapturePreview", return_value=preview))
        stack.enter_context(patch("sys.argv", [
            "capture", "unused-output", "--duration", "60", "--timeout", "3", "--max-frames", str(max_frames),
            "--display", "--record-imu", "--accel-topic", "/imu/accel_test", "--gyro-topic", "/imu/gyro_test",
        ]))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return source_type, recorder_type, imu_type

    def assert_rgbd_close(self, recorder, complete, source, imu_summary):
        recorder.close.assert_called_once()
        positional, keywords = recorder.close.call_args
        self.assertEqual(positional[0] if positional else keywords["complete"], complete)
        self.assertEqual(keywords["camera"], source.statistics.return_value)
        self.assertEqual(keywords["imu"], imu_summary)

    def test_flag_starts_imu_and_refreshes_rgbd_after_startup(self):
        source, frames, recorder, imu, preview = self.resources()
        with ExitStack() as stack:
            _, _, imu_type = self.enter(stack, source, recorder, imu, preview)
            capture.main()
        imu_type.assert_called_once_with(source, Path("unused-output"),
                                         accel_topic="/imu/accel_test", gyro_topic="/imu/gyro_test")
        imu.start.assert_called_once()
        start_args, start_keywords = imu.start.call_args
        self.assertEqual(start_args[0] if start_args else start_keywords["timeout"], 3.0)
        # IMU 发现话题/启动可能耗时，不能把启动前的旧 RGB-D 当首帧录进去。
        self.assertEqual([item.args[0].sequence for item in recorder.submit.call_args_list], [10, 11])
        self.assertGreaterEqual(imu.check.call_count, 2)
        imu.close.assert_called_once()
        self.assert_rgbd_close(recorder, True, source, imu.close.return_value)
        source.close.assert_called_once()
        preview.close.assert_called_once()

    def test_preview_q_with_imu_finishes_both_streams_and_closes_all_resources(self):
        source, frames, recorder, imu, preview = self.resources()
        preview.show.return_value = True
        with ExitStack() as stack:
            self.enter(stack, source, recorder, imu, preview, max_frames=1000)
            capture.main()
        recorder.submit.assert_called_once_with(frames[1])
        preview.show.assert_called_once()
        imu.close.assert_called_once()
        self.assert_rgbd_close(recorder, True, source, imu.close.return_value)
        source.close.assert_called_once()
        preview.close.assert_called_once()

    def test_incomplete_imu_summary_prevents_complete_rgbd_claim(self):
        source, _, recorder, imu, preview = self.resources()
        imu.close.return_value = dict(complete=False, accel_messages=12, gyro_messages=0)
        with ExitStack() as stack:
            self.enter(stack, source, recorder, imu, preview, max_frames=1)
            capture.main()
        self.assert_rgbd_close(recorder, False, source, imu.close.return_value)
        source.close.assert_called_once()
        preview.close.assert_called_once()

    def test_imu_start_check_or_close_error_never_marks_rgbd_complete_and_releases_resources(self):
        for phase in ("start", "check", "close"):
            with self.subTest(phase=phase):
                source, _, recorder, imu, preview = self.resources()
                getattr(imu, phase).side_effect = RuntimeError(f"imu {phase} failed")
                with ExitStack() as stack:
                    _, recorder_type, _ = self.enter(stack, source, recorder, imu, preview, max_frames=1)
                    with self.assertRaisesRegex(RuntimeError, f"imu {phase} failed"):
                        capture.main()
                # 启动失败时 RGB-D writer 可能尚未建立；一旦建立就必须显式不完整。
                if recorder_type.called:
                    recorder.close.assert_called_once()
                    positional, keywords = recorder.close.call_args
                    self.assertFalse(positional[0] if positional else keywords["complete"])
                imu.close.assert_called_once()
                source.close.assert_called_once()
                preview.close.assert_called_once()


class CapturePreviewTests(unittest.TestCase):
    def gui(self, stack, key=-1, visible=1.):
        for name in ("namedWindow", "resizeWindow", "imshow", "destroyWindow"):
            stack.enter_context(patch.object(capture.cv2, name))
        stack.enter_context(patch.object(capture.cv2, "waitKey", return_value=key))
        stack.enter_context(patch.object(capture.cv2, "getWindowProperty", return_value=visible))

    def test_preview_does_not_modify_recorded_rgb_or_depth(self):
        current = frame()
        original_rgb = current.rgb.copy()
        original_depth = current.depth.copy()
        with ExitStack() as stack:
            self.gui(stack)
            preview = capture.CapturePreview(hz=15.)
            try:
                self.assertFalse(preview.show(current, 1, 1, 0, 1., 30.))
            finally:
                preview.close()
        np.testing.assert_array_equal(current.rgb, original_rgb)
        np.testing.assert_array_equal(current.depth, original_depth)

    def test_q_escape_and_closed_window_request_normal_stop(self):
        for key, visible in ((ord("q"), 1.), (27, 1.), (-1, 0.)):
            with self.subTest(key=key, visible=visible), ExitStack() as stack:
                self.gui(stack, key=key, visible=visible)
                preview = capture.CapturePreview(hz=15.)
                try:
                    self.assertTrue(preview.show(frame(), 1, 1, 0, 1., 30.))
                finally:
                    preview.close()

    def test_preview_refresh_limit_skips_rendering_without_sleeping(self):
        with ExitStack() as stack:
            self.gui(stack)
            stack.enter_context(patch.object(capture.time, "monotonic", side_effect=[1., 1.01, 1.1]))
            sleep = stack.enter_context(patch.object(capture.time, "sleep"))
            preview = capture.CapturePreview(hz=15.)
            try:
                for submitted in range(1, 4):
                    self.assertFalse(preview.show(frame(submitted), submitted, submitted, 0, 1., 30.))
                self.assertEqual(capture.cv2.imshow.call_count, 2)
                sleep.assert_not_called()
            finally:
                preview.close()


if __name__ == "__main__":
    unittest.main()
