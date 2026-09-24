# ZED SVO2 + FoundationPose

这套程序不修改 FoundationPose 原版核心代码：宿主机 ZED SDK 负责解码 SVO2，
FoundationPose Docker 容器负责 6D 位姿估计与后续跟踪。

> 录像完成时必须正常停止录制并关闭 ZED 相机。运行前用 `ls -lh 文件.svo2`
> 确认文件不是 0 字节；0 字节文件不包含任何可恢复的视频帧。

## 在当前 CUDA 13 Docker 环境中运行

先在宿主机进入容器：

```bash
cd /home/tomato/6Dpose
source ./activate
```

容器默认位于 `/workspace/6Dpose/FoundationPose`，运行：

```bash
python -u ../src/run_foundationpose_zed.py
```

所有当前路径和参数都已写为程序默认值。程序会自动复用输出目录中的 mask 和首帧位姿，
并处理完整录像。默认值集中在 `src/run_foundationpose_zed.py` 顶部的“默认配置”区域。

建议首次在命令末尾加 `--max-frames 10` 验证前 10 帧；确认正常后直接运行上述短命令。
程序默认用 `--score-batch-size 32` 分批评分全部 252 个首帧姿态候选，以适配 8GB 显卡；
如果仍然显存不足可改为 `--score-batch-size 16`。
如果已有成功保存的首帧姿态，可用 `--initial-pose ob_in_cam/000000.txt` 跳过耗时的
首帧注册并直接恢复跟踪。
想忽略已有姿态并重新执行首帧注册时运行
`python -u ../src/run_foundationpose_zed.py --fresh-register`。

默认输入和模型分别是：

- `/home/tomato/6Dpose/zed_photo_capture/svo/recording.svo2`
- `/home/tomato/6Dpose/kfs_model/BlueTrueKFS13/BlueTrueKFS13.obj`

程序第一次运行会显示录像首帧。用鼠标左键沿目标轮廓依次点选，按回车完成；
右键或 `U` 撤销上一点，`R` 清空，`Q`/Esc 退出。生成的 mask 会保存到
`outputs/zed_foundationpose/initial_mask.png`。

结果位于 `outputs/zed_foundationpose/`：

- `ob_in_cam/*.txt`：每帧 4x4 的 object-to-camera 位姿，平移单位为米；
- `track_vis/*.png`：带 3D 包围盒和坐标轴的逐帧结果；
- `result.mp4`：结果视频；
- `initial_mask.png`：首帧目标 mask。
- `run.log`：完整运行日志；如果推理提前退出，错误会保留在这里。

首帧 `register` 比后续跟踪慢很多，第一次运行还可能编译 CUDA kernel。标完 mask 后，
窗口会显示 `Loading FoundationPose models` 和 `Registering first frame`；首帧完成后才开始
逐帧显示 3D 包围盒。再次运行可用 `--mask` 复用已保存的 mask。

## 分步运行和复用 mask

```bash
cmake -S src -B src/build
cmake --build src/build -j"$(nproc)"

src/build/zed_svo_export \
  --svo /home/tomato/6Dpose/zed_photo_capture/svo/recording.svo2 \
  --output /home/tomato/6Dpose/outputs/zed_sequence \
  --stride 1

python ../src/run_foundationpose_zed.py \
  --sequence /workspace/6Dpose/outputs/zed_sequence \
  --mask /workspace/6Dpose/outputs/zed_foundationpose/initial_mask.png \
  --output /workspace/6Dpose/outputs/zed_foundationpose
```

录像很长时可在导出命令中添加 `--start-frame`、`--end-frame` 和 `--stride`。
