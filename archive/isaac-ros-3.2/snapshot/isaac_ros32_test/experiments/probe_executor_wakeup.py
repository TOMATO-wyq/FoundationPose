#!/usr/bin/env python3
"""只读对照实验：参数服务请求是否会唤醒延迟的 tracker 输出。"""
import time

import rclpy
from rcl_interfaces.srv import GetParameters


def main():
    rclpy.init()
    node = rclpy.create_node('isaacfp_executor_wakeup_probe')
    client = node.create_client(GetParameters, '/foundationpose_tracker/get_parameters')
    if not client.wait_for_service(timeout_sec=3.0):
        raise RuntimeError('Tracker parameter service is unavailable')
    request = GetParameters.Request(names=['extension_log_level'])
    print(f'PROBE_START wall={time.time():.6f}', flush=True)
    deadline = time.monotonic() + 20.0
    count = 0
    # 不发布图像或位姿，不修改参数；每次等上一个请求结束，避免请求堆积。
    while rclpy.ok() and time.monotonic() < deadline:
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=1.0)
        if not future.done():
            print('Parameter request timed out; stopping probe', flush=True)
            break
        future.result()
        count += 1
        time.sleep(0.1)
    print(f'PROBE_END wall={time.time():.6f} replies={count}', flush=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
