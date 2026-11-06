#!/usr/bin/env python3
"""Compare two TKK segmenters on the same RGB frame from ROS or a saved image."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
import threading
import time

import cv2
import numpy as np

from tools.yolo.box_test_seg_d435i import decode_image


PROJECT = Path(__file__).resolve().parents[3]
TITLE = 'TKK YOLO comparison | LEFT: A / RIGHT: B | SPACE save | +/- confidence | Q quit'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def default_weights():
    pointer = PROJECT / 'outputs' / 'tkk_latest_training.json'
    if pointer.is_file():
        info = json.loads(pointer.read_text())
        run = Path(info['run_dir'])
        reference = Path(info.get('comparison_reference_weights', run / 'weights' / 'epoch275.pt'))
    else:
        run = PROJECT / 'runs' / 'tkk_seg_20261029_014048_400e'
        reference = run / 'weights' / 'epoch275.pt'
    return run / 'weights' / 'best.pt', reference


def checkpoint_epoch(model, path):
    checkpoint = model.ckpt or {}
    epoch = checkpoint.get('epoch', -1)
    if epoch >= 0:
        return epoch + 1
    history = checkpoint.get('train_results', {})
    epochs = history.get('epoch', [])
    if path.name == 'best.pt' and epochs:
        # Ultralytics replaces the stripped best checkpoint's history with the
        # full run history. Its final row is not the epoch of the best model.
        keys = [f'metrics/mAP{suffix}({task})' for task in ('B', 'M')
                for suffix in ('50', '50-95')]
        if all(key in history for key in keys):
            scores = [.1 * history[keys[0]][i] + .9 * history[keys[1]][i]
                      + .1 * history[keys[2]][i] + .9 * history[keys[3]][i]
                      for i in range(len(epochs))]
            return int(epochs[int(np.argmax(scores))])
    return int(epochs[-1]) if epochs else None


class RosFrames:
    def __init__(self, topic, node_name='tkk_yolo_same_frame_comparison'):
        import rclpy
        from rclpy.executors import ExternalShutdownException
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        self.rclpy = rclpy
        self.lock = threading.Lock()
        self.frame = None
        self.sequence = 0
        rclpy.init()
        self.node = Node(node_name)

        def on_image(message):
            try:
                image = decode_image(message)
                stamp = dict(sec=message.header.stamp.sec, nanosec=message.header.stamp.nanosec,
                             frame_id=message.header.frame_id)
                with self.lock:
                    self.sequence += 1
                    self.frame = image, stamp, time.monotonic(), self.sequence
            except Exception as error:
                self.node.get_logger().error(str(error), throttle_duration_sec=5)

        self.subscription = self.node.create_subscription(
            Image, topic, on_image, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))

        def spin():
            try:
                rclpy.spin(self.node)
            except (ExternalShutdownException, KeyboardInterrupt):
                pass

        self.thread = threading.Thread(target=spin, daemon=True)
        self.thread.start()

    def latest(self):
        with self.lock:
            return self.frame

    def close(self):
        if self.rclpy.ok():
            self.rclpy.shutdown()
        self.thread.join(timeout=3)
        self.node.destroy_node()


def predict(model, frame, args, threshold, torch):
    synchronize = args.device != 'cpu'
    if synchronize:
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = model.predict(frame.copy(), imgsz=args.imgsz, device=args.device, conf=threshold,
                           iou=args.iou, classes=[0], retina_masks=True, verbose=False)[0]
    confidences = result.boxes.conf.cpu().tolist()
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        if masks.shape[1:] != frame.shape[:2]:
            raise RuntimeError('retina mask dimensions do not match the original image')
        mask[np.any(masks > .5, axis=0)] = 255
    if synchronize:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000
    return dict(overlay=result.plot(), mask=mask, confidences=confidences,
                detections=len(confidences), max_confidence=max(confidences, default=0.),
                call_ms=round(elapsed, 3), mask_pixels=int(np.count_nonzero(mask)))


def mask_agreement(a, b):
    foreground_a, foreground_b = a > 0, b > 0
    union = int(np.logical_or(foreground_a, foreground_b).sum())
    intersection = int(np.logical_and(foreground_a, foreground_b).sum())
    return intersection / union if union else None


def comparison_view(results, labels, report):
    panels = []
    for key, label in zip(('a', 'b'), labels):
        result = results[key]
        image = result['overlay']
        header = np.zeros((64, image.shape[1], 3), dtype=np.uint8)
        cv2.putText(header, f'{key.upper()}: {label}', (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 240, 255), 2)
        text = (f"targets={result['detections']} conf={result['max_confidence']:.3f} "
                f"call={result['call_ms']:.1f}ms")
        cv2.putText(header, text, (10, 51), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1)
        panels.append(np.vstack((header, image)))
    body = np.hstack(panels)
    footer = np.zeros((54, body.shape[1], 3), dtype=np.uint8)
    agreement = 'N/A (both empty)' if report['mask_agreement_iou'] is None else f"{report['mask_agreement_iou']:.3f}"
    text = (f"SAME FRAME {report['source_sequence']} | threshold={report['confidence_threshold']:.2f} "
            f"| mask agreement IoU={agreement}")
    cv2.putText(footer, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (220, 220, 220), 1)
    cv2.putText(footer, 'SPACE: save | +/-: confidence | Q: quit | agreement is NOT accuracy',
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, .5, (180, 180, 180), 1)
    return np.vstack((body, footer))


def save_comparison(output, frame, results, view, report, config):
    folder = output / 'snapshots'
    folder.mkdir(exist_ok=True)
    key = datetime.now().strftime('%H%M%S_%f') + f"_f{report['source_sequence']:06d}"
    files = {'rgb': frame, 'a_overlay': results['a']['overlay'], 'a_mask': results['a']['mask'],
             'b_overlay': results['b']['overlay'], 'b_mask': results['b']['mask'], 'comparison': view}
    for suffix, pixels in files.items():
        if not cv2.imwrite(str(folder / f'{key}_{suffix}.png'), pixels):
            raise OSError(f'Failed to write {suffix}')
    metadata = dict(report, models=config['models'],
                    raw_bgr_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                    image_width=frame.shape[1], image_height=frame.shape[0])
    write_json(folder / (key + '.json'), metadata)
    print(f'SAVED {folder / key}', flush=True)


def main():
    default_a, default_b = default_weights()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-a', type=Path, default=default_a)
    parser.add_argument('--weights-b', type=Path, default=default_b)
    parser.add_argument('--topic', default='/camera/camera/color/image_raw')
    parser.add_argument('--image', type=Path, help='Compare one saved image instead of opening the camera')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--conf', type=float, default=.5)
    parser.add_argument('--iou', type=float, default=.7, help='NMS IoU threshold, identical for both models')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', choices=('0', 'cpu'), default='0')
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--duration', type=float, default=0)
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--save-first', action='store_true')
    args = parser.parse_args()
    if not (0 < args.conf < 1 and 0 < args.iou < 1 and args.imgsz > 0
            and args.timeout > 0 and args.max_frames >= 0 and args.duration >= 0):
        parser.error('Invalid confidence, image size, timeout or frame limit')
    if not all(p.is_file() for p in (args.weights_a, args.weights_b)):
        parser.error('Both weight files must exist')
    if args.weights_a.resolve() == args.weights_b.resolve():
        parser.error('Choose two different weight files')
    if args.image and not args.image.is_file():
        parser.error('Image file does not exist')
    if not args.no_display and not os.environ.get('DISPLAY'):
        parser.error('A desktop DISPLAY is required; use --no-display for a saved test')
    output = (args.output or PROJECT / 'outputs' /
              datetime.now().strftime('tkk_seg_compare_%Y%m%d_%H%M%S_%f')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'comparison_config.json').exists():
        parser.error('Output directory already contains a comparison; choose a new directory')
    import torch
    from ultralytics import YOLO

    if args.device != 'cpu' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu')
    models = {'a': YOLO(str(args.weights_a)), 'b': YOLO(str(args.weights_b))}
    descriptors = {}
    for key, path in [('a', args.weights_a), ('b', args.weights_b)]:
        model = models[key]
        if model.task != 'segment' or model.names != {0: 'tkk'}:
            raise ValueError(f'{path}: expected a single-class TKK segmentation model')
        epoch = checkpoint_epoch(model, path)
        descriptors[key] = dict(weights=str(path.resolve()),
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest(), epoch=epoch)
        descriptors[key]['label'] = f'{path.name} / epoch {epoch}' if epoch else path.name
        model.predict(np.zeros((480, 640, 3), dtype=np.uint8), imgsz=args.imgsz,
                      device=args.device, conf=args.conf, verbose=False)
        print(f"{key.upper()}: {descriptors[key]['label']} {path}", flush=True)
    config = dict(models=descriptors, topic=None if args.image else args.topic,
                  image=str(args.image.resolve()) if args.image else None,
                  ros_domain_id=os.environ.get('ROS_DOMAIN_ID', '0'),
                  imgsz=args.imgsz, confidence=args.conf, nms_iou=args.iou, device=args.device,
                  comparison='Same decoded BGR frame, identical settings; sequential GPU inference',
                  note='Inter-model mask IoU measures agreement, not ground-truth accuracy')
    write_json(output / 'comparison_config.json', config)
    source = None
    processed, last_sequence = 0, -1
    threshold = args.conf
    sums = Counter()
    iou_count = 0
    started = last_frame = last_report = time.monotonic()
    error = None
    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            raise ValueError(f'Cannot decode {args.image}')
        static_frame = image, dict(image=str(args.image)), time.monotonic(), 1
    try:
        if not args.no_display:
            cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(TITLE, 1280, 620)
        if not args.image:
            source = RosFrames(args.topic)
            print(f'Waiting for RGB on {args.topic}; ROS_DOMAIN_ID={config["ros_domain_id"]}', flush=True)
        with (output / 'frames.jsonl').open('w', buffering=1) as records:
            while True:
                now = time.monotonic()
                if args.duration and now - started >= args.duration:
                    break
                current = static_frame if args.image else source.latest()
                if current is None or current[3] == last_sequence or (not args.image and now - current[2] > 2):
                    if now - last_frame > args.timeout:
                        raise RuntimeError('No fresh RGB frames: check the camera and ROS domain')
                    if args.no_display:
                        time.sleep(.005)
                    else:
                        if cv2.waitKey(10) & 255 in (ord('q'), 27):
                            break
                        if cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    continue
                frame, stamp, received, sequence = current
                last_sequence, last_frame = sequence, now
                pair_started = time.perf_counter()
                results = {}
                # Alternate which model runs first to reduce fixed-order timing bias.
                for key in (('a', 'b') if processed % 2 == 0 else ('b', 'a')):
                    results[key] = predict(models[key], frame, args, threshold, torch)
                processed += 1
                agreement = mask_agreement(results['a']['mask'], results['b']['mask'])
                report = dict(processed_frame=processed, source_sequence=sequence, stamp=stamp,
                    confidence_threshold=threshold, frame_age_before_inference_ms=round((now - received) * 1000, 2),
                    pair_call_ms=round((time.perf_counter() - pair_started) * 1000, 3),
                    mask_agreement_iou=agreement,
                    a={key: value for key, value in results['a'].items() if key not in ('mask', 'overlay')},
                    b={key: value for key, value in results['b'].items() if key not in ('mask', 'overlay')})
                records.write(json.dumps(report) + '\n')
                for key in ('a', 'b'):
                    sums[key + '_detected_frames'] += int(results[key]['detections'] > 0)
                    sums[key + '_call_ms'] += results[key]['call_ms']
                if agreement is not None:
                    sums['agreement_iou'] += agreement
                    iou_count += 1
                view = comparison_view(results, [descriptors[key]['label'] for key in ('a', 'b')], report)
                if args.save_first and processed == 1:
                    save_comparison(output, frame, results, view, report, config)
                if now - last_report >= 2:
                    print(f"frame={processed} source={sequence} A={results['a']['max_confidence']:.3f} "
                          f"B={results['b']['max_confidence']:.3f} mask_agreement={agreement} "
                          f"pair={report['pair_call_ms']:.1f}ms", flush=True)
                    last_report = now
                if args.no_display:
                    if args.image:
                        break
                else:
                    cv2.imshow(TITLE, view)
                    key = cv2.waitKey(0 if args.image else 1) & 255
                    if key == 32:
                        save_comparison(output, frame, results, view, report, config)
                    elif key in (ord('+'), ord('=')):
                        threshold = min(.95, round(threshold + .05, 2))
                    elif key in (ord('-'), ord('_')):
                        threshold = max(.05, round(threshold - .05, 2))
                    elif key in (ord('q'), 27):
                        break
                    if cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if args.image:
                        last_sequence = -1
                if args.max_frames and processed >= args.max_frames:
                    break
    except KeyboardInterrupt:
        pass
    except Exception as caught:
        error = str(caught)
        raise
    finally:
        if source:
            source.close()
        if not args.no_display:
            cv2.destroyAllWindows()
        summary = dict(state='failed' if error else 'complete', error=error, processed_frames=processed,
                       elapsed_seconds=round(time.monotonic() - started, 2),
                       a_detected_frames=sums['a_detected_frames'], b_detected_frames=sums['b_detected_frames'],
                       a_mean_call_ms=sums['a_call_ms'] / processed if processed else None,
                       b_mean_call_ms=sums['b_call_ms'] / processed if processed else None,
                       mean_mask_agreement_iou=sums['agreement_iou'] / iou_count if iou_count else None,
                       agreement_frames=iou_count, note=config['note'])
        write_json(output / 'comparison_summary.json', summary)
        print('COMPARISON_REPORT ' + json.dumps(dict(output=str(output), **summary)), flush=True)


if __name__ == '__main__':
    main()
