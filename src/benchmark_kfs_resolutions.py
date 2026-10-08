#!/usr/bin/env python3
"""Four supported D435i profiles; static jitter, not absolute pose accuracy."""
import argparse
import csv
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import numpy as np

from compare_kfs_pose_runs import static_jitter
from live_rgbd_transport import receive_frame

ROOT = Path(__file__).resolve().parents[1]
PROFILES = [(1920, 1080, 30, '1280,720,30'),
            (1280, 720, 30, '1280,720,30'),
            (640, 480, 30, '640,480,30'),
            (640, 480, 60, '640,480,60')]


def stop(process):
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=8)


def summarize(directory):
    with (directory/'loop_timing.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    # Exclude registration and the first 30 tracking frames after every register.
    selected, warm = [], 30
    for row in rows:
        if row['mode'] != 'tracking':
            warm = 30
        elif warm:
            warm -= 1
        else:
            selected.append(row)
    summary = {'measured_tracking_frames': len(selected)}
    if selected:
        total = sum(float(r['loop_s']) for r in selected)
        summary['steady_loop_fps'] = len(selected)/total
        for name in ('inference_s', 'display_s', 'rgbd_wait_s', 'pose_save_s'):
            values = [float(r[name])*1000 for r in selected]
            summary[name.replace('_s', '_ms')] = {
                'mean': float(np.mean(values)), 'p95': float(np.percentile(values, 95))}
    # Only adjacent, measured, valid poses; do not bridge warmup, failures, or recovery.
    pose_dir = directory/'measured_poses'
    pose_dir.mkdir()
    for row in selected:
        name = f"{int(row['processed_index'])-1:06d}.txt"
        source = directory/'ob_in_cam'/name
        if source.exists():
            (pose_dir/name).write_bytes(source.read_bytes())
    summary['static_jitter'] = static_jitter(pose_dir)
    runtime = json.loads((directory/'runtime_summary.json').read_text())
    summary['register_attempts'] = runtime['register_attempts']
    summary['torch_peak_reserved_bytes'] = runtime['torch_peak_reserved_bytes']
    summary['note'] = 'static frame-to-frame jitter, not absolute error; no symmetry adjustment'
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=360)
    parser.add_argument('--static-scene', action='store_true', required=True)
    parser.add_argument('--display', action='store_true', help='also measure preview overhead')
    parser.add_argument('--dynamic-roi', type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.frames < 90:
        parser.error('at least 90 frames required')
    docker = ['docker']
    if subprocess.run(docker+['info'], stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode:
        docker = ['sudo', '-n', 'docker']
    if subprocess.run(docker+['info'], stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode:
        raise SystemExit('Run sudo -v in this terminal first.')
    # Fail rather than reset another active camera or overwrite its source socket.
    nodes = subprocess.check_output(['ros2', 'node', 'list', '--no-daemon'], text=True)
    if '/camera/camera' in nodes:
        raise SystemExit('Stop existing RealSense launch with Ctrl+C before benchmarking.')
    image = 'foundationpose:kfs-yolo'
    if subprocess.run(docker+['image', 'inspect', image], stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode:
        subprocess.run(docker+['build', '--network=host', '-t', image, '-f',
            str(ROOT/'src/Dockerfile.kfs-yolo'), str(ROOT/'src')], check=True)
    output = ROOT/'outputs'/time.strftime('kfs_benchmark_%Y%m%d_%H%M%S')
    output.mkdir(exist_ok=False)
    results = [dict(rgb_profile='1920,1080,60', status='unsupported'),
               dict(rgb_profile='1280,720,60', status='unsupported')]
    print(f'Output: {output}\nKeep camera and object stationary throughout all tests.', flush=True)
    for width, height, fps, depth_profile in PROFILES:
        label = f'{width}x{height}_{fps}'
        directory = output/label
        directory.mkdir()
        sock = output/f'{label}.sock'
        driver = source = None
        handles = []
        record = dict(rgb_profile=f'{width},{height},{fps}', depth_profile=depth_profile,
                      scale=1, mask_mode='yolo', display=args.display, dynamic_roi=args.dynamic_roi)
        try:
            def spawn(command, log):
                handle = (directory/log).open('w')
                handles.append(handle)
                return subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                        start_new_session=True)
            driver = spawn(['ros2', 'launch', 'realsense2_camera', 'rs_launch.py',
                'enable_color:=true', 'enable_depth:=true', 'enable_rgbd:=true',
                'enable_sync:=true', 'align_depth.enable:=true',
                f'rgb_camera.color_profile:={width},{height},{fps}',
                f'depth_module.depth_profile:={depth_profile}'], 'camera.log')
            source = spawn(['python3', str(ROOT/'src/camera_ros_source.py'), '--camera',
                'd435i', '--scale', '1', '--socket', str(sock)], 'source.log')
            deadline = time.monotonic()+40
            while not sock.exists():
                if time.monotonic() > deadline or source.poll() is not None:
                    raise RuntimeError('Source startup failed; see source.log')
                time.sleep(.1)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(32)
                connection.connect(str(sock))
                rgb, depth, k, metadata = receive_frame(connection)
            if rgb.shape[:2] != (height, width):
                raise RuntimeError(f'Profile fallback: actual {rgb.shape}, expected {height}x{width}')
            actual = {}
            for name in ('rgb_camera.color_profile', 'depth_module.depth_profile'):
                actual[name] = subprocess.check_output(['ros2', 'param', 'get',
                    '/camera/camera', name], text=True).strip()
            record['reported_profiles'] = actual
            for name, expected in [('rgb_camera.color_profile', f'{width},{height},{fps}'),
                                   ('depth_module.depth_profile', depth_profile)]:
                reported = actual[name].split(':', 1)[-1].strip().replace('x', ',')
                if reported != expected:
                    raise RuntimeError(f'Profile fallback: {name}={reported}, expected {expected}')
            # Allow auto-exposure and depth to settle before initializing FP.
            time.sleep(3)
            container_output = '/workspace/6Dpose/'+str(directory.relative_to(ROOT))+'/fp'
            cmd = docker+['run', '--rm', '--gpus', 'all', '--shm-size=4g', '--network=host',
                '--user', f'{os.getuid()}:{os.getgid()}', '-e', 'HOME=/tmp',
                '-e', 'MPLCONFIGDIR=/tmp/matplotlib', '-e', 'YOLO_CONFIG_DIR=/tmp/yolo',
                '-v', f'{ROOT}:/workspace/6Dpose', '-w', '/workspace/6Dpose/FoundationPose']
            if args.display:
                if not os.environ.get('DISPLAY'):
                    raise RuntimeError('--display requires DISPLAY')
                cmd += ['-e', 'DISPLAY='+os.environ['DISPLAY'], '-v', '/tmp/.X11-unix:/tmp/.X11-unix']
                auth = Path(os.environ.get('XAUTHORITY', str(Path.home()/'.Xauthority')))
                if auth.exists():
                    cmd += ['-v', f'{auth}:/tmp/kfs.Xauthority:ro', '-e', 'XAUTHORITY=/tmp/kfs.Xauthority']
            cmd += [image, 'python', '-u', '../src/run_foundationpose_live.py', '--camera',
                'd435i', '--socket', '/workspace/6Dpose/'+str(sock.relative_to(ROOT)),
                '--mask_mode', 'yolo', '--yolo-weights', '../weights/kfs_yolo_seg.pt',
                '--segmentation-config', '../src/config/kfs_blue.json', '--depth-refinement',
                '--debug', '0', '--save-poses', '--max-frames', str(args.frames),
                '--dynamic-roi', str(args.dynamic_roi),
                '--output', container_output]
            if not args.display:
                cmd += ['--no-display']
            with (directory/'fp.log').open('w') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            record.update(status='complete', **summarize(directory/'fp'))
            print(label, json.dumps(record, ensure_ascii=False), flush=True)
        except (RuntimeError, subprocess.CalledProcessError, OSError) as error:
            record.update(status='failed', error=str(error))
            print(label, 'FAILED', error, flush=True)
        finally:
            stop(source)
            stop(driver)
            for handle in handles:
                handle.close()
        results.append(record)
        (output/'summary.json').write_text(json.dumps(results, indent=2))
    columns = ['rgb_profile', 'depth_profile', 'status', 'steady_loop_fps',
               'translation_step_p50_mm', 'translation_step_p95_mm',
               'rotation_step_p50_deg', 'rotation_step_p95_deg']
    with (output/'summary.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in results:
            merged = {**record, **record.get('static_jitter', {})}
            writer.writerow({name: merged.get(name) for name in columns})
    print(f'Results: {output}/summary.csv', flush=True)


if __name__ == '__main__':
    main()
