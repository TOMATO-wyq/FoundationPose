# TKK 两个 YOLO 分割模型实时对比

在桌面终端运行：

```bash
bash /home/tomato/6Dpose/src/run_tkk_seg_compare.sh
```

默认读取最新一次 TKK 训练的位置，左侧为 `best.pt`（本次第 28 轮），右侧为
`epoch275.pt`（本次第 276 轮，文件名按零开始计数）。两个模型驻留在同一个进程和
GPU 中，对同一张原始彩色图依次推理，然后并排显示。两侧的输入分辨率、置信度阈值、
NMS 阈值相同；逐帧交替推理顺序，减少固定先后顺序对耗时的影响。

若最新训练是补拍后微调，后台训练清单会记录 `comparison_reference_weights`；此时右侧
改用补拍前的原始第 276 轮权重，左侧为补拍后训练的 `best.pt`，用于比较改进效果。

启动器复用已有相机驱动，自动识别它的 ROS domain；没有驱动时以 640×480@30
仅启动彩色流。退出时只停止本次启动的驱动，不关闭已有驱动。
也可以用 `--domain 86` 明确指定 ROS domain，此参数放在其他参数之前。

## 窗口操作

- `Q` / `Esc` 或关闭窗口：退出。
- 空格：保存同一帧的原图、两侧预测叠加图、两侧原始尺寸二值 mask、并排图和元数据。
- `+` / `-`：同时提高或降低两侧置信度阈值，每次 0.05，默认 0.50。

窗口显示目标数量、最高置信度、每次预测调用耗时，以及两侧 mask 的重合度。
mask 重合度只表示两模型是否一致，不是标注准确率；置信度高也不保证轮廓正确。
有多个 `tkk` 时显示全部实例，重合度和保存的二值 mask 使用全部实例的并集。
耗时包含推理及结果传回 CPU，不代表严格的端到端相机延迟，也不是独立模型吞吐量测试。

输出在 `/home/tomato/6Dpose/outputs/tkk_seg_compare_时间戳_PID/`。
`comparison_config.json` 保存权重 SHA256、实际训练轮数和设置，`frames.jsonl` 保存每帧
时间戳及两侧结果，`comparison_summary.json` 在退出时生成，`snapshots/` 为按空格保存的图片。

## 更换右侧模型

例如与第 400 轮的最后权重比较：

```bash
bash /home/tomato/6Dpose/src/run_tkk_seg_compare.sh \
  --weights-b /home/tomato/6Dpose/runs/tkk_seg_20261029_014048_400e/weights/last.pt
```

两侧均可用 `--weights-a` / `--weights-b` 指定其他单类别 `tkk` 分割权重。

## 无窗口检查与离线图片

```bash
# 实际相机 30 帧，保存首帧对比结果，完成后退出。
bash /home/tomato/6Dpose/src/run_tkk_seg_compare.sh \
  --no-display --max-frames 30 --save-first

# 对比一张原始图片，不启动相机；窗口中按空格保存、Q 退出。
bash /home/tomato/6Dpose/src/run_tkk_seg_compare.sh --image /绝对路径/图片.png
```

彩色图只保留最新帧，两侧共享该帧，不各自读取不同时间的相机画面。相机超过默认
30 秒没有新帧时退出并记录错误；可以用 `--timeout` 调整。
