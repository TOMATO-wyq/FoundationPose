"""Live UP freshness and timeout diagnosis without requiring a ROS installation."""
from collections import deque
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

from box4d.ros_input import RosSource


class RgbdTimeoutTests(unittest.TestCase):
    def source(self, publishers=0, rejected=0, duplicates=0, error=None):
        source = RosSource.__new__(RosSource)
        source.topic = '/camera/camera/rgbd'
        source.condition = threading.Condition()
        source.latest = None
        source.received_count = 0
        source.rejected_count = rejected
        source.duplicate_count = duplicates
        source.error = error
        source.node = Mock()
        source.node.count_publishers.return_value = publishers
        return source

    def test_absent_rgbd_publisher_explains_domain_mismatch(self):
        source = self.source()
        with patch.dict('os.environ', {'ROS_DOMAIN_ID': '86'}):
            with self.assertRaisesRegex(TimeoutError, 'No RGB-D publisher found') as error:
                source.get(timeout=0)
        self.assertIn(source.topic, str(error.exception))
        self.assertIn('ROS_DOMAIN_ID=86', str(error.exception))
        self.assertIn('Publishers=0, received_unique=0', str(error.exception))
        self.assertIn('same ROS_DOMAIN_ID', str(error.exception))

    def test_present_rgbd_publisher_without_frames_reports_stream_or_qos(self):
        source = self.source(publishers=1)
        with self.assertRaisesRegex(TimeoutError, 'A publisher exists') as error:
            source.get(timeout=0)
        self.assertIn('QoS', str(error.exception))
        self.assertNotIn('No RGB-D publisher found', str(error.exception))

    def test_rejected_rgbd_preserves_specific_input_error(self):
        source = self.source(publishers=1, rejected=3, error='RGB/depth source stamp mismatch')
        with self.assertRaisesRegex(TimeoutError, 'RGB-D data was rejected') as error:
            source.get(timeout=0)
        self.assertIn('rejected=3', str(error.exception))
        self.assertIn(source.error, str(error.exception))

    def test_duplicate_rgbd_is_diagnosed_as_a_timestamp_problem(self):
        source = self.source(publishers=1, duplicates=4)
        with self.assertRaisesRegex(TimeoutError, 'increasing source timestamps') as error:
            source.get(timeout=0)
        self.assertIn('duplicate_or_old=4', str(error.exception))

    def test_available_rgbd_does_not_need_a_graph_query(self):
        source = self.source()
        source.latest = NS(sequence=2)
        self.assertIs(source.get(previous_sequence=1, timeout=0), source.latest)
        source.node.count_publishers.assert_not_called()


class LiveImuTests(unittest.TestCase):
    def source(self, publishers=1):
        source = RosSource.__new__(RosSource)
        source.imu_topic = '/camera/camera/accel/sample'
        source.imu_condition = threading.Condition()
        source.imu_samples = deque(maxlen=2048)
        source.imu_received_count = source.imu_duplicate_count = 0
        source.node = Mock()
        source.node.count_publishers.return_value = publishers
        source.rclpy = NS(time=NS(Time=Mock()))
        source.imu_buffer = Mock()
        source.imu_buffer.lookup_transform.return_value = NS(
            transform=NS(rotation=NS(x=0., y=0., z=0., w=1.)))
        return source

    def measure(self, source, samples=0, stamp_scale=1., duplicates=False):
        now = [100.]
        pending = deque((100.+i*.01, int(i*.01*stamp_scale*1e9),
                         'accel_optical', [0., -9.8, 0.]) for i in range(1, samples+1))

        def advance(timeout):
            now[0] = round(now[0]+timeout, 9)
            while pending and pending[0][0] <= now[0]:
                row = pending.popleft()
                source.imu_received_count += 1
                if duplicates:
                    source.imu_duplicate_count += 1
                else:
                    source.imu_samples.append(row)

        # estimate_up imports only Duration from ROS; TF is a controlled stand-in.
        with patch.dict('sys.modules', {'rclpy.duration': NS(Duration=NS)}), \
             patch('box4d.ros_input.time.monotonic', side_effect=lambda: now[0]), \
             patch.object(source.imu_condition, 'wait', side_effect=advance):
            return source.estimate_up('color_optical', duration=2., timeout=3.)

    def test_publisher_without_data_reports_stalled_stream(self):
        source = self.source()
        with self.assertRaisesRegex(ValueError, 'publisher exists but no messages arrived') as error:
            self.measure(source)
        self.assertIn('Publishers=1, received=0', str(error.exception))
        self.assertIn('hw_reset', str(error.exception))
        self.assertNotIn('enable_accel:=true is required', str(error.exception))
        source.imu_buffer.lookup_transform.assert_not_called()

    def test_absent_publisher_reports_topic_and_domain(self):
        source = self.source(publishers=0)
        with patch.dict('os.environ', {'ROS_DOMAIN_ID': '86'}):
            with self.assertRaisesRegex(ValueError, 'No IMU publisher found') as error:
                self.measure(source)
        self.assertIn(source.imu_topic, str(error.exception))
        self.assertIn('ROS_DOMAIN_ID=86', str(error.exception))
        self.assertIn('--imu-topic', str(error.exception))

    def test_buffered_old_measurement_is_not_reused(self):
        source = self.source()
        source.imu_samples.extend((90.+i*.01, i*10_000_000, 'accel_optical', [0., -9.8, 0.])
                                  for i in range(201))
        with self.assertRaisesRegex(ValueError, '0 unique samples'):
            self.measure(source)

    def test_fresh_samples_still_require_sufficient_source_time_span(self):
        source = self.source()
        with self.assertRaisesRegex(ValueError, '200 unique samples, 0.199s source span'):
            self.measure(source, samples=200, stamp_scale=.1)
        source.imu_buffer.lookup_transform.assert_not_called()

    def test_repeated_messages_are_diagnosed_as_timestamp_problem(self):
        source = self.source()
        with self.assertRaisesRegex(ValueError, 'increasing source timestamps') as error:
            self.measure(source, samples=200, duplicates=True)
        self.assertIn('received=200, duplicate_or_old=200', str(error.exception))

    def test_fresh_stationary_measurement_preserves_tf_and_provenance(self):
        source = self.source()
        report = self.measure(source, samples=200)
        np.testing.assert_allclose(report['up_camera'], [0., -1., 0.], atol=1e-10)
        self.assertEqual(report['status'], 'stationary_estimate')
        self.assertEqual(report['imu_topic'], source.imu_topic)
        self.assertEqual(report['quality']['samples'], 200)
        self.assertFalse(report['quality']['absolute_accuracy_validated'])
        self.assertGreaterEqual(report['sample_span_s'], 1.6)
        self.assertEqual(source.imu_buffer.lookup_transform.call_args.args[:2],
                         ('color_optical', 'accel_optical'))


if __name__ == '__main__':
    unittest.main()
