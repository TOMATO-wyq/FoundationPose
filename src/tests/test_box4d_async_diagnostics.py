"""Delayed detector evidence retains its source frame and timing domain."""

from copy import deepcopy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from box4d.config import Config
from box4d.geometry import Camera
from box4d.results import empty_result
from box4d.synthetic import Renderer, default_camera, world_box
from box4d.temporal import TemporalPipeline
from box4d.tracking import Tracker
from box4d.diagnostics import detector_snapshot


class AsyncDiagnosticsTests(unittest.TestCase):
    def test_source_accepted_pose_is_saved_independently_of_current_tracking(self):
        source, current = self.frame(1), self.frame(2)
        raw = self.source_result(source)
        raw["objects"] = [
            dict(
                size_m=0.2,
                center_camera_m=[0.1, 0.2, 1.0],
                faces=[dict(quad=[[10, 12], [30, 12], [30, 32], [10, 32]])],
            )
        ]
        diagnostic = detector_snapshot(source, current, raw, [], 10.0, 0.75)
        raw["objects"][0]["center_camera_m"][0] = 999
        raw["objects"][0]["faces"][0]["quad"][0][0] = 999
        self.assertEqual(diagnostic["objects"][0]["center_camera_m"][0], 0.1)
        self.assertEqual(diagnostic["objects"][0]["faces"][0]["quad"][0][0], 10)
        self.assertEqual(diagnostic["source_frame"]["stamp_ns"], source.stamp_ns)

    @classmethod
    def setUpClass(cls):
        cls.renderer = Renderer(default_camera())

    def cfg(self):
        return Config(up_camera=[0, -1, 0], temporal_recovery=True, plane_grid_step=8)

    def frame(self, sequence):
        return self.renderer.render([], sequence)[0]

    def consume(self, source, current, old_result, reports=()):
        # Feed the actual single-slot handoff without starting a worker or camera.
        pipeline = TemporalPipeline(self.cfg())
        pipeline.asynchronous = True
        pipeline.worker_completed = (source, old_result, list(reports), 25000.0, (None, ()))
        result, _ = pipeline.process(current)
        return pipeline, result

    def source_result(self, source):
        result = empty_result(source, self.cfg())
        result.update(
            rejected_faces=[
                dict(quad=[[10, 12], [30, 12], [30, 32], [10, 32]], reason="depth_support", source="plane")
            ],
            appearance_rejections=[
                dict(reason="insufficient_configured_target_colors", size_m=0.2, classified_samples=2)
            ],
            suppressed_hypotheses=[dict(reason="overlapping_volume_hypothesis", size_m=0.2)],
            frame_status="partial_or_rejected",
            yolo=dict(
                sequence=source.sequence,
                stamp_ns=source.stamp_ns,
                instances=[dict(polygons=[[[10, 12], [30, 12], [30, 32]]])],
            ),
        )
        result["timing"].update(detector_ms=10000.0, solve_ms=2000.0, processing_ms=20000.0, yolo_ms=9000.0)
        return result

    def test_compatible_rejections_have_source_pixels_and_keep_current_frame_status(self):
        source, current = self.frame(1), self.frame(2)
        old_result = self.source_result(source)
        reports = [dict(roi=[1, 2, 40, 50], timing=dict(detector_ms=12.0))]
        pipeline, result = self.consume(source, current, old_result, reports)
        diagnostic = result["detector_diagnostics"]
        self.assertEqual(result["stamp_ns"], current.stamp_ns)
        self.assertEqual(result["rejected_faces"], [])
        self.assertEqual(result["appearance_rejections"], [])
        self.assertEqual(result["suppressed_hypotheses"], [])
        self.assertEqual(
            diagnostic["source_frame"],
            dict(
                sequence=source.sequence,
                stamp_ns=source.stamp_ns,
                calibration_id=source.camera.calibration_id,
                frame_id=source.camera.frame_id,
                camera=source.camera.to_dict(),
                received_monotonic_ns=None,
            ),
        )
        self.assertEqual(diagnostic["current_frame"]["stamp_ns"], current.stamp_ns)
        self.assertEqual(diagnostic["stamp_ns"], source.stamp_ns)
        self.assertEqual(diagnostic["sequence"], source.sequence)
        self.assertAlmostEqual(diagnostic["age_ms"], 16.666667)
        self.assertTrue(diagnostic["compatible_with_current_frame"])
        self.assertTrue(diagnostic["used_for_current_frame_status"])
        self.assertEqual(diagnostic["unusable_reasons"], [])
        self.assertEqual(diagnostic["object_count"], 0)
        self.assertEqual(diagnostic["rejected_reason_counts"], {"depth_support": 1})
        for key in ("rejected_faces", "appearance_rejections", "suppressed_hypotheses", "timing", "yolo"):
            self.assertEqual(diagnostic[key], old_result[key])
        self.assertEqual(diagnostic["local_searches"], reports)
        self.assertEqual(diagnostic["background_detection_ms"], 25000.0)
        self.assertEqual(result["timing"]["detector_ms"], 0.0)
        self.assertEqual(result["timing"]["solve_ms"], 0.0)
        self.assertEqual(result["timing"]["yolo_ms"], 9000.0)
        self.assertLess(result["timing"]["processing_ms"], 25000.0)
        self.assertEqual(result["frame_status"], "partial_or_rejected")
        Tracker(pipeline.cfg).update(result)
        self.assertEqual(result["frame_status"], "partial_or_rejected")
        old_result["rejected_faces"][0]["quad"][0][0] = 999
        reports[0]["timing"]["detector_ms"] = 999
        self.assertEqual(diagnostic["rejected_faces"][0]["quad"][0][0], 10)
        self.assertEqual(diagnostic["local_searches"][0]["timing"]["detector_ms"], 12.0)
        self.assertEqual(result["temporal"]["detector_queue_depth"], 1)

    def test_incompatible_rejections_do_not_describe_the_current_frame(self):
        for kind, reason in (
            ("old", "result_too_old"),
            ("calibration", "calibration_mismatch"),
            ("future", "source_in_future"),
        ):
            with self.subTest(kind=kind):
                source, current = self.frame(1), self.frame(2)
                if kind == "old":
                    current.stamp_ns += 2_000_000_000
                elif kind == "future":
                    source.stamp_ns = current.stamp_ns + 1
                else:
                    k = current.camera.k.copy()
                    k[0, 0] += 1
                    current.camera = Camera(current.camera.width, current.camera.height, k)
                pipeline, result = self.consume(source, current, self.source_result(source))
                diagnostic = result["detector_diagnostics"]
                self.assertFalse(diagnostic["compatible_with_current_frame"])
                self.assertFalse(diagnostic["used_for_current_frame_status"])
                self.assertIn(reason, diagnostic["unusable_reasons"])
                self.assertTrue(diagnostic["rejected_faces"])
                self.assertEqual(result["temporal"]["detector_seed_failures"], [])
                self.assertEqual(result["frame_status"], "no_target")
                Tracker(pipeline.cfg).update(result)
                self.assertEqual(result["frame_status"], "no_target")

    def test_appearance_only_and_suppression_only_are_reported(self):
        for key in ("appearance_rejections", "suppressed_hypotheses"):
            with self.subTest(key=key):
                source, current = self.frame(1), self.frame(2)
                old_result = empty_result(source, self.cfg())
                old_result[key] = [dict(reason="source_rejection")]
                pipeline, result = self.consume(source, current, old_result)
                self.assertEqual(result["frame_status"], "partial_or_rejected")
                Tracker(pipeline.cfg).update(result)
                self.assertEqual(result["frame_status"], "partial_or_rejected")

    def test_minimal_empty_detector_double_still_works(self):
        source, current = self.frame(1), self.frame(2)
        old_result = dict(
            objects=[],
            rejected_faces=[],
            timing=dict(
                detector_ms=0.0, solve_ms=0.0, refinement_ms=0.0, appearance_ms=0.0, proposal_count=0
            ),
        )
        _, result = self.consume(source, current, old_result)
        self.assertEqual(result["frame_status"], "no_target")
        diagnostic = result["detector_diagnostics"]
        self.assertEqual(diagnostic["appearance_rejections"], [])
        self.assertEqual(diagnostic["suppressed_hypotheses"], [])
        self.assertFalse(diagnostic["has_rejections"])

    def test_diagnostic_consumption_keeps_the_latest_frame_in_one_pending_slot(self):
        source, current = self.frame(1), self.frame(2)
        pipeline, first = self.consume(source, current, self.source_result(source))
        self.assertEqual(first["temporal"]["detector_queue_depth"], 1)
        latest = self.frame(3)
        second, _ = pipeline.process(latest)
        self.assertEqual(second["temporal"]["detector_queue_depth"], 1)
        self.assertEqual(second["temporal"]["detector_pending_replaced"], 1)
        self.assertIs(pipeline.worker_pending[0], latest)
        self.assertEqual(second["frame_status"], "no_target")

    def test_expired_objects_are_separate_from_current_seed_failures(self):
        source, current = self.frame(1), self.frame(2)
        old_result = empty_result(source, self.cfg())
        old_result["objects"] = [dict(size_m=0.2)]
        current.stamp_ns += 2_000_000_000
        _, expired = self.consume(source, current, old_result)
        self.assertEqual(expired["temporal"]["expired_detections"], 1)
        self.assertEqual(expired["temporal"]["detector_seed_failures"], [])
        self.assertEqual(expired["temporal"]["detector_seed_failure_details"], [])
        self.assertEqual(expired["frame_status"], "no_target")

        box = world_box((0, 0, 0.1), 0.2, 15)
        box["red_axis_local"] = [0, 0, 1]
        source = self.renderer.render([box], 1, depth_noise_m=0.001)[0]
        pipeline = TemporalPipeline(self.cfg())
        old_result, _ = pipeline.detector.process(source)
        self.assertTrue(old_result["objects"])
        current = self.frame(2)
        _, invalid = self.consume(source, current, deepcopy(old_result))
        self.assertEqual(invalid["objects"], [])
        self.assertEqual(invalid["temporal"]["expired_detections"], 0)
        self.assertTrue(invalid["temporal"]["detector_seed_failures"])
        self.assertTrue(invalid["temporal"]["detector_seed_failure_details"])

    def test_classical_local_search_preserves_rejections_through_async_handoff(self):
        box = world_box((0, 0, 0.1), 0.2, 15)
        box["red_axis_local"] = [0, 0, 1]
        source = self.renderer.render([box], 1, depth_noise_m=0.001)[0]
        pipeline = TemporalPipeline(self.cfg())
        pipeline.cfg.analyze_colors = True
        child = self.source_result(source)
        local = SimpleNamespace(
            detector=SimpleNamespace(_ray_cache=None), process=Mock(return_value=(child, None))
        )
        with patch("box4d.temporal.Pipeline", return_value=local), patch.object(
            pipeline.detector, "process", return_value=(empty_result(source, pipeline.cfg), None)
        ):
            result, _, reports, _ = pipeline._full_detect(source, [])
        self.assertTrue(reports)
        self.assertEqual(result["valid_object_count"], 0)
        self.assertEqual(result["frame_status"], "partial_or_rejected")
        for name in ("rejected_faces", "appearance_rejections", "suppressed_hypotheses"):
            self.assertEqual(reports[0][name], child[name])
            self.assertEqual(result[name][0]["local_search_roi"], reports[0]["roi"])
        _, current = self.consume(source, self.frame(2), result, reports)
        self.assertEqual(current["rejected_faces"], [])
        self.assertTrue(current["detector_diagnostics"]["rejected_faces"])
        self.assertEqual(current["frame_status"], "partial_or_rejected")


if __name__ == "__main__":
    unittest.main()
