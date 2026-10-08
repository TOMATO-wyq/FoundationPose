# D435i 实时测试 YOLO 箱体分割

## 启动

```bash
/home/tomato/6Dpose/src/run_box_seg_d435i.sh
```

启动器优先订阅已有 `/camera/camera/color/image_raw`；没有发布者且没有相机节点时，启动已有 RealSense ROS 驱动（1280×720、30 FPS，仅彩色流）。关闭时只停止本脚本启动的驱动，已有相机节点不动。

推理使用 `/home/tomato/code/box_seg_annotation/env` 的独立 Conda 环境和 CUDA，默认权重为 `runs/box_seg_trial_20261004/weights/best.pt`。没有修改 FP、Isaac ROS 或 Docker。相机输入只保留最新帧，避免推理画面积压。

## 操作

- 左侧：模型预测多边形/框及置信度。右侧：目标二值 mask。
- **空格**：保存原 RGB、叠加图、二值 mask、相机时间戳和置信度。
- **Q / Esc**：退出。关闭窗口也会退出。
- 目标不存在时 mask 全黑，不能把空 mask 用于 FP 初始化。多目标时右侧和保存 mask 选择 purple_box 中置信度最高的一个。

请把箱体放进画面，尝试不同距离、姿态、背景和遮挡；再把箱体移出画面检查是否误检。不要只测试训练时同一场景。

保存目录：`/home/tomato/code/box_seg_annotation/live_tests/d435i_时间/`。PNG mask 和原 RGB 尺寸相同，只有 0（背景）/255（箱体）。保存的 RGB 和 overlay 不含顶部预览状态文字。

## 参数

```bash
# 提高阈值，观察误检是否减少
/home/tomato/6Dpose/src/run_box_seg_d435i.sh --conf 0.7
# 更换权重
/home/tomato/6Dpose/src/run_box_seg_d435i.sh --weights /绝对路径/best.pt
# 无界面短测试，保存首帧
/home/tomato/6Dpose/src/run_box_seg_d435i.sh --no-display --max-frames 20 --save-first
```

图像等待默认超时 15 秒；若相机启动较慢可加 `--timeout 30`。相机节点存在但未发布 RGB 时，启动器会提示检查，避免重复打开 USB 设备。驱动日志：`live_tests/camera_latest.log`。

界面里的 inference 毫秒统计包含 Python 侧预测调用，不等于严格相机到显示的总延迟，也不代表实时 FP 的性能。本程序仅测试 RGB 分割，没有集成深度和 FoundationPose。

## 本机验证记录

2026-10-04 实际读取 D435i 的 20 帧 1280×720 RGB，GPU 推理平均调用 7.37ms。当前镜头只包含椅子/背景，没有 purple_box，20 帧均无检测，输出 mask 正常全黑。首帧 RGB/overlay/mask 和 test_report.json 保存在 `live_tests/d435i_20261004_224314_260243/`。此结果验证了实时采集、模型加载、推理及保存通路；有箱体时的实时识别效果仍需把目标放入画面检查。
