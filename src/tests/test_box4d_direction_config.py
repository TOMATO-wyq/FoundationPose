"""Direction controls must survive live tuning and saved JSON configuration."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from box4d.config import Config
from box4d.debug import (
    TUNING_PAGES,
    TUNING_SPECS,
    TUNING_ZH,
    STAGES,
    DetectionDebug,
    _panel,
    _atomic_dump_config,
    render_stages,
    tuning_positions,
    tuning_values,
)
from box4d.detector import Face
from box4d.io import Frame
from box4d.synthetic import default_camera


DIRECTION_DEFAULTS = dict(
    track_direction_window_tau_s=0.20,
    track_direction_prior_per_deg=0.02,
    track_direction_escape_margin=0.15,
    track_direction_escape_frames=3,
    track_reseed_max_rotation_deg=15.0,
)


class DirectionConfigTests(unittest.TestCase):
    def test_existing_config_inherits_direction_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "existing.json"
            path.write_text(json.dumps(dict(geometry_backend="cpp", yolo_device="cpu")))
            cfg = Config.load(path)
        for key, value in DIRECTION_DEFAULTS.items():
            self.assertEqual(getattr(cfg, key), value)
        self.assertEqual(cfg.geometry_backend, "cpp")
        self.assertEqual(cfg.yolo_device, "cpu")

    def test_direction_scalar_validation_reports_the_field(self):
        invalid_values = {
            "track_direction_window_tau_s": (0, -0.1, float("nan"), float("inf"), True, None, "0.2"),
            "track_direction_prior_per_deg": (-0.01, float("nan"), float("inf"), False, None, "0.02"),
            "track_direction_escape_margin": (-0.01, float("nan"), float("inf"), False, None, "0.15"),
            "track_reseed_max_rotation_deg": (0, -1, 180.1, float("nan"), float("inf"), True, None, "15"),
        }
        for key, values in invalid_values.items():
            for value in values:
                with self.subTest(field=key, value=value):
                    with self.assertRaisesRegex(ValueError, key):
                        Config(**{key: value}).validate()

    def test_escape_frames_is_a_positive_integer(self):
        for value in (0, -1, 1.5, 3.0, True, False, None, "3"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "track_direction_escape_frames"):
                    Config(track_direction_escape_frames=value).validate()
        self.assertEqual(Config(track_direction_escape_frames=1).validate().track_direction_escape_frames, 1)

    def test_zero_weight_and_margin_are_valid_for_ab_comparison(self):
        cfg = Config(
            track_direction_prior_per_deg=0,
            track_direction_escape_margin=0,
            track_reseed_max_rotation_deg=180,
        ).validate()
        self.assertEqual(cfg.track_direction_prior_per_deg, 0)
        self.assertEqual(cfg.track_direction_escape_margin, 0)
        self.assertEqual(cfg.track_reseed_max_rotation_deg, 180)

    def test_direction_slider_defaults_round_trip_without_quantization(self):
        values = tuning_values(tuning_positions(Config().validate()))
        for key, value in DIRECTION_DEFAULTS.items():
            self.assertAlmostEqual(values[key], value)
        self.assertIsInstance(values["track_direction_escape_frames"], int)
        self.assertEqual(len(TUNING_ZH), len(TUNING_SPECS))
        direction_indices = {index for index, (_, key, *_) in enumerate(TUNING_SPECS) if key in DIRECTION_DEFAULTS}
        self.assertEqual(len(direction_indices), len(DIRECTION_DEFAULTS))
        self.assertTrue(any(
            direction_indices == set(range(start, end)) and end - start <= 7
            for _, _, start, end in TUNING_PAGES
        ))

    def test_direction_slider_endpoints_validate(self):
        expected_ranges = {
            "track_direction_window_tau_s": (0.05, 1.0),
            "track_direction_prior_per_deg": (0, 0.1),
            "track_direction_escape_margin": (0, 0.5),
            "track_direction_escape_frames": (1, 8),
            "track_reseed_max_rotation_deg": (1, 45),
        }
        for endpoint in (0, 1):
            positions = tuning_positions(Config().validate())
            for index, (_, key, _, _, _, maximum) in enumerate(TUNING_SPECS):
                if key in expected_ranges:
                    positions[index] = maximum if endpoint else 0
            values = tuning_values(positions)
            cfg = Config(**values).validate()
            for key, bounds in expected_ranges.items():
                self.assertAlmostEqual(getattr(cfg, key), bounds[endpoint])

    def test_tuned_values_survive_atomic_save_and_load(self):
        values = dict(
            track_direction_window_tau_s=0.35,
            track_direction_prior_per_deg=0.0,
            track_direction_escape_margin=0.27,
            track_direction_escape_frames=7,
            track_reseed_max_rotation_deg=23,
        )
        cfg = Config(**values, geometry_backend="cpp", yolo_device="cpu").validate()
        slider_values = tuning_values(tuning_positions(cfg))
        for key in values:
            setattr(cfg, key, slider_values[key])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "saved.json"
            _atomic_dump_config(path, cfg.validate().to_dict())
            restored = Config.load(path)
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])
        for key, value in values.items():
            self.assertAlmostEqual(getattr(restored, key), value)
        self.assertIsInstance(restored.track_direction_escape_frames, int)
        self.assertEqual(restored.geometry_backend, "cpp")
        self.assertEqual(restored.yolo_device, "cpu")

    def test_live_yolo_config_explicitly_uses_direction_defaults(self):
        path = Path(__file__).resolve().parents[1] / "config" / "box4d_d435i_yolo_live.json"
        cfg = Config.load(path)
        for key, value in DIRECTION_DEFAULTS.items():
            self.assertEqual(getattr(cfg, key), value)


class TrialDisplayTests(unittest.TestCase):
    def render_trial_lines(self, trials):
        frame = Frame(
            np.zeros((480, 640, 3), np.uint8),
            np.ones((480, 640), np.float32),
            default_camera(),
            1,
        )
        trace = DetectionDebug(frame, Config().validate())
        face = Face(np.array([[150., 100.], [490., 100.], [490., 380.], [150., 380.]]), "test", 1.)
        trace.face(face).update(status="rejected", reason="pose failed", trials=trials)
        trace.images["normals"] = np.zeros_like(frame.rgb)
        validation_lines = []

        def panel(image, title, source, lines=()):
            if title == STAGES[6]:
                validation_lines.extend(lines)
            return _panel(image, title, source, lines)

        with patch("box4d.debug._panel", side_effect=panel):
            images = render_stages(frame, dict(objects=[]), trace)
        self.assertEqual(images.shape, (8, 600, 720, 3))
        return "\n".join(validation_lines)

    @staticmethod
    def early_trial():
        return dict(
            method="iterative",
            accepted=False,
            stage="front_facing",
            reason="candidate is behind camera",
            reprojection_rms_px=None,
            reprojection_limit_px=3.,
            depth_median_residual_m=None,
            normal_error_deg=None,
            upright_tilt_deg=None,
        )

    def test_all_early_trials_show_failure_without_invented_metrics(self):
        lines = self.render_trial_lines([self.early_trial()])
        self.assertIn("front_facing", lines)
        self.assertIn("candidate is behind camera", lines)
        self.assertIn("residuals not evaluated", lines)
        self.assertNotIn("reproj 0.00", lines)

    def test_complete_trial_is_shown_after_an_early_rejected_trial(self):
        complete = dict(
            method="joint",
            accepted=False,
            reprojection_rms_px=4.,
            reprojection_limit_px=3.,
            depth_median_residual_m=.002,
            normal_error_deg=2.,
            upright_tilt_deg=3.,
        )
        lines = self.render_trial_lines([self.early_trial(), complete])
        self.assertIn("joint: reproj 4.00/3.00px depth 2.0/12.0mm", lines)
        self.assertNotIn("residuals not evaluated", lines)

    def test_accepted_trial_is_preferred_over_complete_rejected_trial(self):
        rejected = dict(
            method="iterative", accepted=False, reprojection_rms_px=4.,
            reprojection_limit_px=3., depth_median_residual_m=.002,
            normal_error_deg=2., upright_tilt_deg=3.,
        )
        accepted = dict(rejected, method="joint", accepted=True, reprojection_rms_px=1.)
        lines = self.render_trial_lines([self.early_trial(), rejected, accepted])
        self.assertIn("joint: reproj 1.00/3.00px", lines)


if __name__ == "__main__":
    unittest.main()
