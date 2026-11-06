"""Saving optional debug evidence must not stop the recognition loop."""

from collections import deque
from concurrent.futures import wait
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from box4d.config import Config
from box4d.debug import DebugSession, _SnapshotSaver, _atomic_dump_config
from box4d.io import json_value


class SnapshotErrorTests(unittest.TestCase):
    def finish_pending(self, saver):
        _, unfinished = wait(tuple(saver.pending), timeout=2.0)
        self.assertFalse(unfinished, "snapshot writer did not finish")

    def session(self, output, config_path, saver=None):
        # Exercise the main-process control flow without spawning HighGUI.
        session = DebugSession.__new__(DebugSession)
        session.output = Path(output)
        session.config_path = Path(config_path)
        session.config = Config().validate()
        session.save_errors = deque(maxlen=8)
        session.save_pending = False
        session.saved = 0
        session.last = ((SimpleNamespace(sequence=7),), None)
        session.snapshot_saver = saver if saver is not None else Mock()
        session._apply_tuning = Mock(return_value=True)
        session.process = Mock()
        session.process.is_alive.return_value = True
        session.command = SimpleNamespace(value=-1)
        return session

    def test_failed_write_releases_slot_and_next_snapshot_succeeds(self):
        saver = _SnapshotSaver()
        output, errors = io.StringIO(), io.StringIO()
        with patch(
            "box4d.debug.save_snapshot", side_effect=[OSError("disk full"), None]
        ) as write, redirect_stdout(output), redirect_stderr(errors):
            try:
                self.assertTrue(saver.submit(Path("failed"), (), None))
                self.finish_pending(saver)
                saver.check()
                self.assertFalse(saver.pending)
                self.assertTrue(saver.submit(Path("retry"), (), None))
            finally:
                saver.close()
        self.assertEqual(write.call_count, 2)
        self.assertEqual(saver.failure_count, 1)
        self.assertEqual(saver.errors[-1]["path"], "failed")
        self.assertIn("failed: failed: OSError: disk full", errors.getvalue())
        self.assertIn("Debug snapshot saved: retry", output.getvalue())
        self.assertNotIn("Debug snapshot saved: failed", output.getvalue())

    def test_repeated_failures_keep_only_bounded_string_records(self):
        saver = _SnapshotSaver()
        with patch("box4d.debug.save_snapshot", side_effect=OSError("disk full")), redirect_stderr(
            io.StringIO()
        ):
            try:
                for index in range(12):
                    self.assertTrue(saver.submit(Path(str(index)), (), None))
                    self.finish_pending(saver)
                    saver.check()
            finally:
                saver.close()
        self.assertEqual(saver.failure_count, 12)
        self.assertEqual(len(saver.errors), 8)
        self.assertEqual(saver.errors[0]["path"], "4")
        self.assertTrue(all(isinstance(value, str) for record in saver.errors for value in record.values()))

    def test_submit_failure_and_submit_after_close_are_reported(self):
        saver = _SnapshotSaver()
        errors = io.StringIO()
        with redirect_stderr(errors):
            with patch.object(saver.executor, "submit", side_effect=RuntimeError("cannot start thread")):
                self.assertFalse(saver.submit(Path("not-started"), (), None))
            saver.close()
            self.assertFalse(saver.submit(Path("closed"), (), None))
            saver.close()
        self.assertFalse(saver.pending)
        self.assertEqual(saver.failure_count, 2)
        self.assertIn("not-started: RuntimeError: cannot start thread", errors.getvalue())
        self.assertIn("closed: RuntimeError: snapshot saver is closed", errors.getvalue())

    def test_poll_reports_worker_failure_and_continues(self):
        saver = _SnapshotSaver()
        session = self.session("output", "config.json", saver)
        errors = io.StringIO()
        with patch("box4d.debug.save_snapshot", side_effect=OSError("permission denied")), redirect_stderr(
            errors
        ):
            try:
                saver.submit(Path("failed"), (), None)
                self.finish_pending(saver)
                self.assertEqual(session.poll(None), -1)
                self.assertEqual(session.poll(None), -1)
            finally:
                session.stopping = SimpleNamespace(value=False)
                session.process.is_alive.return_value = False
                session.close()
        self.assertIn("failed: OSError: permission denied", errors.getvalue())
        self.assertEqual(saver.failure_count, 1)
        self.assertTrue(session.stopping.value)

    def test_config_failure_skips_snapshot_and_reports_each_saved_step(self):
        for failed_step in ("output", "source"):
            with self.subTest(failed_step=failed_step), tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / "output"
                output.mkdir()
                source = Path(temp) / "source.json"
                tuned = output / "tuned_config.json"
                source.write_text("original source\n")
                tuned.write_text("original tuned\n")
                session = self.session(output, source)
                failed_path = tuned if failed_step == "output" else source
                stdout, stderr = io.StringIO(), io.StringIO()

                def write(path, value):
                    if path == failed_path:
                        raise OSError("read-only filesystem")
                    _atomic_dump_config(path, value)

                with patch("box4d.debug._atomic_dump_config", side_effect=write), redirect_stdout(
                    stdout
                ), redirect_stderr(stderr):
                    # Test the S key path as well as the configuration writer.
                    session.command.value = ord("s")
                    self.assertEqual(session.poll(None), ord("s"))
                    self.assertEqual(session.poll(None), -1)
                session.snapshot_saver.submit.assert_not_called()
                self.assertEqual(session.saved, 0)
                self.assertFalse(session.save_pending)
                self.assertEqual(session.save_errors[-1]["path"], str(failed_path))
                self.assertIn(str(failed_path), stderr.getvalue())
                self.assertIn("OSError: read-only filesystem", stderr.getvalue())
                self.assertIn("Debug snapshot not queued", stdout.getvalue())
                self.assertEqual(
                    failed_path.read_text(), f'original {"tuned" if failed_step == "output" else "source"}\n'
                )
                saved_path = source if failed_step == "output" else tuned
                self.assertEqual(json.loads(saved_path.read_text()), json_value(session.config.to_dict()))
                self.assertIn(f"saved: {saved_path}", stdout.getvalue())

    def test_partial_config_write_preserves_original_and_can_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text("original config\n")

            def partial_write(temporary, value):
                Path(temporary).write_text('{"partial":')
                raise OSError("disk full")

            with patch("box4d.debug.dump_json", side_effect=partial_write):
                with self.assertRaisesRegex(OSError, "disk full"):
                    _atomic_dump_config(path, dict(saved=True))
            self.assertEqual(path.read_text(), "original config\n")
            self.assertEqual(list(Path(temp).iterdir()), [path])
            _atomic_dump_config(path, dict(saved=True))
            self.assertEqual(json.loads(path.read_text()), dict(saved=True))
            self.assertEqual(list(Path(temp).iterdir()), [path])

    def test_successful_config_save_updates_both_files_before_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output"
            output.mkdir()
            source = Path(temp) / "source.json"
            session = self.session(output, source)

            def submit(path, packet, panels):
                self.assertEqual(json.loads(source.read_text()), json_value(session.config.to_dict()))
                self.assertEqual(
                    json.loads((output / "tuned_config.json").read_text()),
                    json_value(session.config.to_dict()),
                )
                return True

            session.snapshot_saver.submit.side_effect = submit
            with redirect_stdout(io.StringIO()):
                session.save(None)
            self.assertEqual(session.saved, 1)
            session.snapshot_saver.submit.assert_called_once()
            self.assertFalse(session.save_errors)


if __name__ == "__main__":
    unittest.main()
