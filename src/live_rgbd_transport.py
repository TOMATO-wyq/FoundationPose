"""Local RGB-D transport shared by the host ROS source and FP container."""
import json
import struct
import time

import numpy as np


def recv_exact(connection, size):
    result = bytearray(size)
    view = memoryview(result)
    while view:
        count = connection.recv_into(view)
        if not count:
            raise EOFError("RGB-D source disconnected")
        view = view[count:]
    return result


def send_frame(connection, frame, roi=None):
    rgb, depth, k, stamp, frame_id, sequence = frame[:6]
    full_rgb, full_k, full_shape = rgb, k, depth.shape
    crop_started = time.perf_counter()
    if roi is not None:
        from kfs_roi import crop_frame
        rgb, depth, k = crop_frame(rgb, depth, k, roi)
    source_timings = dict(frame[6]) if len(frame) > 6 else {}
    source_timings['source_roi_crop_s'] = time.perf_counter()-crop_started
    header = json.dumps(dict(shape=list(depth.shape), k=k.tolist(), stamp=stamp,
                             frame_id=frame_id, sequence=sequence,
                             roi_xyxy=roi, full_shape=list(full_shape), full_k=full_k.tolist(),
                             has_preview=roi is not None,
                             tag_result=frame[7] if len(frame) > 7 else None,
                             source_timings=source_timings)).encode()
    connection.sendall(struct.pack("!I", len(header)))
    connection.sendall(header)
    connection.sendall(memoryview(np.ascontiguousarray(rgb)).cast("B"))
    connection.sendall(memoryview(np.asarray(depth, dtype="<f4", order="C")).cast("B"))
    if roi is not None:
        connection.sendall(memoryview(np.ascontiguousarray(full_rgb)).cast('B'))


def receive_frame(connection, roi=None):
    if roi is None:
        connection.sendall(b"N")
    else:
        request = json.dumps(roi).encode()
        connection.sendall(b'R'+struct.pack('!I', len(request))+request)
    size = struct.unpack("!I", recv_exact(connection, 4))[0]
    if size > 65536:
        raise ValueError("Invalid frame header size")
    metadata = json.loads(recv_exact(connection, size))
    height, width = metadata["shape"]
    if not (0 < height <= 4096 and 0 < width <= 4096):
        raise ValueError("Invalid RGB-D dimensions")
    rgb = np.frombuffer(recv_exact(connection, height * width * 3), np.uint8)
    depth = np.frombuffer(recv_exact(connection, height * width * 4), "<f4")
    if metadata.get('has_preview'):
        h, w = metadata['full_shape']
        if not (0 < h <= 4096 and 0 < w <= 4096):
            raise ValueError('Invalid full frame dimensions')
        metadata['preview_rgb'] = np.frombuffer(recv_exact(connection, h*w*3), np.uint8).reshape(h, w, 3).copy()
    return (rgb.reshape(height, width, 3).copy(), depth.reshape(height, width).copy(),
            np.asarray(metadata["k"], dtype=np.float64), metadata)
