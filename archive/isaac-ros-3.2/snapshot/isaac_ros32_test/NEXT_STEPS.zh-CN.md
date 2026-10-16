# 下次继续：Isaac ROS FoundationPose + ZED 录像

原版实时入口已新增（尚未相机/GPU验证）：`~/6Dpose/src/camera_ros_source.py --camera zed2i|d435i` 在宿主机订阅 ROS2；原版容器运行 `/workspace/6Dpose/src/run_foundationpose_live.py --camera zed2i|d435i`。共用 Unix socket 拉取最新完整 RGB-D，首帧当前场景手画掩码并 register，随后 track_one；框/坐标轴/FPS，R 手动重新定位。默认新输出目录，录像入口未改变。详见 `~/6Dpose/src/README.md`；先停 Isaac 释放显存。

相机命名已区分：`zed_bridge.py`（原文件已经包含 zed，保留名称）、新增 `d435_bridge.py`（RealSense D435i）。新增脚本复用同步与缩放逻辑，独立适配 RGB/整数毫米深度；仅语法检查，未启动相机测试。后续需验证三个实际话题、编码、对齐及输出深度单位。详情见 README.zh-CN.md。

## 2026-10-02 已改代码，尚未运行验证

用户要求先修代码、稍后测试。本次未启动容器或回放，仅完成 Python 语法检查。

- 种子桥在 RGB/depth/camera_info 同时间戳到齐时立即调用转发，不再等 50 ms 定时器；定时器保留作为启动发现和告警兜底。
- `/isaacfp/tracking_debug_output` 的播放器订阅改为可靠接收，匹配桥接的可靠发布；官方直连输出继续沿用 sensor-data QoS，避免不兼容。
- 桥接先确保当前位姿更新，再转发对应非空检测；播放器将此检测作为统一完成回执。保留 Header ACK 兼容旧探针，但 tracker-test 不再依赖两条反馈分别抵达。
- 检测位姿恢复沿用现有箱体中心等于网格原点的假设；若未来更换非中心原点网格，需先处理网格中心偏移，不能直接套用。
- 每帧完成映射日志降为 DEBUG；重复补送仍默认 0、离线恢复仍默认关闭。tracker launch 原有默认已是 diagnostics=false、ROS INFO，无需修改。

待验证：重启种子桥并重新运行播放器（两者要同时使用新代码）；保持 tracker 不卸载。随后完整隔帧对照，记录成功帧数/端到端吞吐/停顿。此次改动仅减少已知等待和反馈复杂度，未证明修复 DDS/GXF 数秒停顿，也未做精度验证。更换容器布局和推理优化尚未实施。

## 2026-10-02 当前结论与优先待办（下次从这里继续）

用户观察：原版 tracking 每帧处理稳定，当前 Isaac 接入会交替流畅和停顿；当前没有证明 Isaac 比原版更快或更稳定。保留原版作为可靠方案和性能基准，不因 TensorRT 名称直接推断整体更快。

关闭 debug 窗口的最新对照：计划处理 502 个隔帧输入，完成 303 帧约 160.08 秒，卡住前吞吐约 1.89 FPS；输入到检测返回耗时中位数 74 ms、P95 2.96 秒、最大 6.08 秒。录像第 608 帧 pose_ready=True、detection=False，等待 120 秒后停止，未跑完整段。日志与摘要：`logs/tracking_no_preview_full_playback.log`、`logs/tracking_no_preview_full_summary.json`。此前开启预览曾完成 502/502，约 127 秒（约 3.95 FPS）；运行间存在波动，不能将两次差异直接视为窗口开关的因果效果。关闭窗口仍卡，说明 GUI 不是全部原因。

当前纯 tracking 配置：重复补送默认 0，推荐回放命令不带 `--auto-recover-original`，离线恢复默认关闭；启动日志明确显示这些状态，代码保留中文注释和可恢复开关。首帧仍取原版保存位姿，之后使用 Isaac tracker 输出。之前完整通过测试的离线恢复次数也是 0，不能认定恢复逻辑是三秒停顿根因。

显存结论：此前 TensorRT 执行上下文申请约 2.7 GB 失败；当前 tracker-only 不驻留 Isaac pose 节点，缓冲复用，并不按视频帧数累计显存。tracker-only 整段通过不能证明完整 pose+tracking 的 OOM 已解决。历史反馈超时 RuntimeError 和 exit code -11 也不能一律归为 OOM。

按以下顺序继续，先简化并诊断，不再不断叠加补送与超时兜底：

- [ ] 关闭 ROS DEBUG / GXF 详细诊断，其他配置保持一致，做完整录像对照；记录完成帧数、总时间、延迟分位数与卡住时间戳，验证日志开销。
- [ ] 定位同一时间戳的播放器发送、桥接接收/转发、tracker 四路接收、矩阵与检测输出、桥接转发、播放器接收；重点排查四路到达延迟，以及 pose_ready=True 但 detection=False 的情况。具体 DDS/GXF 原因尚未证实。
- [ ] 先试录像播放器与桥接/tracker 在同一容器内运行，减少跨容器传输，其他设置保持一致，验证停顿是否减少。
- [ ] 评估简化反馈：由桥接统一确认同一帧的有效姿态和检测结果，避免两路消息不同步造成等待；不得直接忽略缺失结果、误用旧姿态或提前推进下一帧来掩盖问题。
- [ ] 清理不再需要的临时补送与恢复机制，先保留可回退配置；每次只改一个因素，用户要求中文注释。
- [ ] 消除链路停顿后，再分析 TensorRT、渲染与真正的 tracking 计算耗时，评估精度/批次配置优化。
- [ ] 最后用相同录像、相同帧选择与追踪设置比较原版和 Isaac 的速度及效果。原版历史文件时间估计约 13 FPS（端到端粗估，非纯 GPU 推理基准）；Isaac 只有实测更快、更稳定才考虑替换。

## 2026-10-02 早先诊断与待验证修复

原版 `6Dpose/src/run_foundationpose_zed.py` 与 Isaac 回放 `play_recording.py` 已加入窗口右上角 FPS：统计最近两秒内完成的结果帧，包含整体等待时间，非纯 GPU 推理 FPS。Isaac 仅统计本次录像的匹配、非空、未重复结果；等待时两个窗口都会刷新数值。原版叠加只用于显示窗口，不写入结果视频。无需重建镜像，重新运行对应脚本即可。

FPS 窗口后续优化：Isaac 结果回调只准备图像，主循环统一处理 GUI；等待时最多 10 Hz 重画，结果窗口仅在新结果或显示 FPS 改变时重画，订阅数查询缓存 1 秒。样例慢帧发布 1790930519.506，到 0522.541 才收到位姿，0525.551/0525.562 才收到内参/RGB，0525.574 已发布输出；约 6 秒主要在数据进入 tracker 之前。需要继续对照关闭详细日志及 ROS/DDS 传输设置，不能认定 FPS 绘制导致数秒延迟。

**早先一次实测通过（后续仍复现停顿，见顶部）：** 单线程 ROS executor、四路输入深度 10、`--kick-copies 0`、官方图，在开启详细诊断的情况下，25/25 短段通过，随后 502/502 完整隔帧录像通过；所有检测匹配录像帧，0 次离线恢复，0 次超时等待警告。原版 FP 只提供初始种子；后续由 Isaac tracker 更新。保存结果：`logs/tracking_depth10_no_kicks_full_playback.log`、`logs/tracking_depth10_no_kicks_full_summary.json`。tracker-only launch 默认 executor 已采用本次单线程配置。仍有少量数秒延迟；尚未验证关闭诊断后的整段、逐帧精度、完整 Isaac pose→tracking 或实时相机。

慢帧例证：录像第 136 帧发布 stamp=1790929159.481003640，tracker 在 9159.517/9159.528 收到位姿/深度，9162.329 才收到 camera_info，9162.440 才收到 RGB，9162.452 就发布结果。约 3 秒停顿主要在四路输入到齐之前；RGB 到齐后约 12 毫秒完成输出。本例不支持“GPU 推理用了 3 秒”，具体 ROS/DDS 传输或回调延迟原因仍待定位。整段输入发布到检测返回的中位数约 61 毫秒，P95 约 1.88 秒，最大约 3.01 秒。

日常启动（诊断默认关闭；首帧有额外 120 秒预热预算）：

```bash
# tracker 容器
ros2 launch /work/zed_foundationpose_tracker_only.launch.py
# 另一容器
python3 /work/seed_tracking_bridge.py
# 宿主机回放
source /opt/ros/humble/setup.bash
python3 ~/code/isaac_ros32_test/play_recording.py \
  --tracker-test --tracker-full \
  --frame-stride 2 --max-frames 0 --result-timeout 120
```

以下诊断记录保留用于解释改动，已通过的结果以上述最新实测为准。

- 单线程 ROS executor + 官方图：冷启动首帧 render 约 70 秒后才进入 TensorRT；30 秒测试提前超时。预热后的 25 帧测试 25/25 匹配，没有离线恢复；完整隔帧测试完成 66 帧，在录像第 134 帧停止。尚未跑完整段。
- 详细日志 `logs/tracker_single_executor_debug.log` 显示推理和 decoder 在正常帧只占很短时间，后续出现两秒补送副本和队列积压。不能仅凭低 GPU 利用率判断推理算力不足，也不能认定单线程已解决问题。
- NITROS release-3.2 在启动时按 `<topic>_qos_depth` 重设 GXF ingress 容量；此前仅改 YAML 的 capacity4 实验未验证实际接收容量，不能作为扩大队列的有效对照。
- 当前改动：tracker launch 四路 `tracking/image`、`depth_image`、`camera_info`、`pose_input` 的 QoS 深度均可用 `input_qos_depth` 设置，默认 10；种子桥默认 `--kick-copies 0`，每帧只发一次。播放器严格模式首帧额外等待预算 `--first-result-timeout 120`，后续仍使用 `--result-timeout`。改动有中文注释，语法检查通过，**等待重启后的实测**。

当前验证命令（取消旧 capacity4 图覆盖，使用官方图；现有官方图容器不用退出）：

```bash
# tracker 容器：停止旧 launch 后执行
ros2 launch /work/zed_foundationpose_tracker_only.launch.py \
  container_executable:=component_container input_qos_depth:=10 \
  diagnostics:=true extension_log_level:=4 ros_log_level:=debug \
  2>&1 | tee /work/logs/tracker_depth10_no_kicks.log
# 桥接容器：停止旧桥接后执行
python3 /work/seed_tracking_bridge.py --kick-copies 0
```

先跑 25 帧并保留框和坐标轴窗口，再跑全段。以下为历史记录，旧的扩容及补送试验命令不作为当前方案。

## 当前进度

- 环境：Ubuntu 22.04、ROS 2 Humble、Isaac ROS 3.2；Isaac 容器使用 CUDA 12.8 / TensorRT 10.9，宿主机的 CUDA 13 与之隔离。
- `recording.svo2` 已导出到 `/home/tomato/6Dpose/outputs/zed_sequence`，回放脚本为 `play_recording.py`。
- Isaac pose-only 曾对录像首帧返回非空检测；Isaac tracker-only 用原版 FP 保存的首帧位姿，也曾返回非空检测。**这尚不能证明整段跟踪稳定或完整 Isaac pose→tracking 链路跑通。**
- 严格回放会等同一录像帧的位姿反馈和检测都到达，再显示下一帧。`--frame-stride 2` 是临时隔帧选项，依次处理录像第 2、4、6… 帧；`--result-timeout 0` 可以无限等待，所以无输出时不会自动进入下一帧。
- 最近一次回放报错，是因为 `/selector` 与 `/isaacfp_seed_tracking_bridge` **同时向 `/tracking/pose_input` 发布**。脚本现已加入混跑检查；先清除这个冲突，再分析慢帧或漂移。
- 2026-09-30 实测：清除混跑后，单帧通过；10 帧测试处理完录像第 2–12 帧，第 14 帧两路结果均未返回。旧桥接在正常帧约 0.3 秒返回之前的 0.25 秒就补送一帧，出现旧副本回执。已把补送延迟改为 2 秒；**待重启桥接脚本后复测**，尚未证明这就是根因。
- 补送延迟 2 秒后，相同 10 帧测试 10/10 通过；完整隔帧回放到录像第 38 帧成功，第 40 帧两路输出再次停住。第 20–38 帧大多等到 2 秒补送后才返回。因此又试过最多 10 次同帧补送，检查是否只是偶发同步丢失。
- 最多补送 10 次的诊断版在独立的 25 帧测试中 25/25 通过（越过录像第 40 帧），但紧接着的带窗口测试又在录像第 8 帧停住。用户确认前几帧绿色 3D 框和坐标轴基本贴合紫色箱体。故**补送 10 次并非可靠修复**，已回退到最多 2 次，仍保留 2 秒延迟和逐次日志。GPU 进程在停顿时只占约 1.6 GB 显存，当前无 OOM 证据。
- 下一项 A/B 实验已准备：`experiments/nitros_foundationpose_tracking_node_capacity4.yaml` 只把 tracker 的 RGB、相机内参、深度和位姿输入接收缓冲从 1 扩到 4。普通 `source activate` 仍使用官方图；仅设置 `ISAACFP_TRACKING_GRAPH_OVERLAY=1` 进入的**tracker 容器**会只读挂载实验图。尚未实机验证。若启动失败，停掉实验容器并用普通 `source activate` 恢复即可；不用重建镜像。

## 下次先做：隔离验证 tracker-only

2026-10-02 继续诊断：容量 4 实验首帧约 73 秒后才返回；后续 25 帧回放完成 17 帧，在录像第 36 帧停止。20 秒、198 次只读参数查询没有解除停顿，因此未证实 ROS 服务请求能唤醒输出。容量与补帧都不能视作已修复。当前没有运行中的 tracker/桥接进程。

下一次先取消图覆盖，在官方图上用单线程 ROS executor 做对照，并采集详细日志（GXF/GPU 图不变）：

```bash
# 宿主机进入 tracker 容器
unset ISAACFP_TRACKING_GRAPH_OVERLAY
source ~/code/isaac_ros32_test/activate
# 以下在容器内执行；日志保存到宿主机项目 logs 目录
mkdir -p /work/logs
ros2 launch /work/zed_foundationpose_tracker_only.launch.py \
  container_executable:=component_container diagnostics:=true \
  extension_log_level:=4 ros_log_level:=debug \
  2>&1 | tee /work/logs/tracker_single_executor_debug.log
```

另一容器按下面原命令启动种子桥；随后先跑 25 个处理帧并保留预览。单线程是待验证的对照方案，尚未实机测试。日志较多，诊断结束后使用原 launch 命令即可关闭这些开关。

- [ ] 在原有终端按 `Ctrl+C` 停止录像回放、完整的 `zed_foundationpose.launch.py` 和旧的 `seed_tracking_bridge.py`。不要在 tracker 工作时卸载组件。
- [ ] 容器终端 1：

  ```bash
  source ~/code/isaac_ros32_test/activate
  ros2 launch /work/zed_foundationpose_tracker_only.launch.py
  ```

- [ ] 看到 `foundationpose_tracker ... Node was started` 后，在容器终端 2：

  ```bash
  source ~/code/isaac_ros32_test/activate
  python3 /work/seed_tracking_bridge.py
  ```

- [ ] 宿主机终端检查 ROS 图；`/selector` 应不存在，`/tracking/pose_input` 应只有桥接节点发布、tracker 订阅：

  ```bash
  source /opt/ros/humble/setup.bash
  ros2 node list --no-daemon
  ros2 topic info /tracking/pose_input -v --no-daemon
  ```

- [ ] 先跑 10 个处理帧观察调试窗口、绿色 3D 框和坐标轴（X 红、Y 绿、Z 蓝）：

  ```bash
  python3 ~/code/isaac_ros32_test/play_recording.py \
    --tracker-test --tracker-full \
    --frame-stride 2 --max-frames 10 --result-timeout 0
  ```

- [ ] 10 帧正常后，将 `--max-frames 10` 改成 `--max-frames 0` 跑完隔帧录像。记录完成帧数、`DRIFT_TRIGGER` / `ORIGINAL_POSE_RESEED`、框是否偏离，以及卡住时**同一时间戳**的播放器、桥接和 tracker 日志。
- [ ] 若某帧一直显示 `pose_ready=False`，检查桥接日志是否收到该帧位姿、是否用了 `detection_fallback`，以及 tracker 日志中的 WARN/ERROR；不要仅凭低 GPU 使用率判断没有推理。若只出现 `preview_frame_cache=miss`，那是预览缓存未命中，不等于推理失败。

## 可选：输入缓冲容量实验

当前默认图在不同回放中分别停在录像第 8、14、40 帧；这项实验用相同回放命令做对照。先停止旧 tracker launch 和旧桥接进程，再启动：

```bash
# 宿主机终端 1：此容器挂载容量为 4 的实验图
export ISAACFP_TRACKING_GRAPH_OVERLAY=1
source ~/code/isaac_ros32_test/activate
ros2 launch /work/zed_foundationpose_tracker_only.launch.py
```

```bash
# 宿主机终端 2：桥接容器无需图覆盖
source ~/code/isaac_ros32_test/activate
python3 /work/seed_tracking_bridge.py
```

确认 `/foundationpose_tracker` 启动后，再用上面的 `--max-frames 25` 命令和完整隔帧命令分别验证。实验图只读挂载，结束后取消 `ISAACFP_TRACKING_GRAPH_OVERLAY` 即可回到官方图。**容量调整会改变 GXF 排队方式，必须用同一录像的匹配检测数和画框效果判断，不能只看是否不崩。**

## 后续待办

- [ ] 评估隔帧跟踪的精度与耗时，再删除临时 `--frame-stride 2`，改用逐帧测试。
- [ ] 单独测试完整 Isaac `zed_foundationpose.launch.py` 的 pose→tracking 链路：先停掉 tracker-only launch 和种子桥，再运行完整 launch；宿主机回放时**不要**加 `--tracker-test`。完整 launch 的首帧 `refine_iterations=5`；8 GB 显卡曾出现 TensorRT OOM，需要记录实际显存和容器报错。
- [ ] 当前 `--auto-recover-original` 只是使用**原版 FP 预先保存的逐帧位姿**验证失跟后重新播种，并非 Isaac 实时重新 pose。待离线流程稳定后，再设计真正的 Isaac 重新定位与实时 ZED 接入。

详细参数和背景见 [README.md](README.md)。

## 2026-10-02 D435i 录制数据已导出

静态 bag `d435i_static_20261002_225751` 导出到 `~/6Dpose/outputs/d435i_static_sequence`：1106 组精确匹配 RGB-D/内参，24 张未匹配 RGB。
动态 bag `d435i_dynamic_20261002_225839` 导出到 `~/6Dpose/outputs/d435i_dynamic_sequence`：1921 组匹配，23 张未匹配 RGB。
原始 bag 未修改。完整 1280x720 RGB PNG + uint16 毫米深度，原始 RGB 帧号命名，frames.csv / all_frames.csv 保留源帧号和时间戳；未匹配 RGB 单独保存。
export_d435_bag.py 可复用，完成摘要 export_summary.json。已检查清单计数、唯一时间戳以及首中末帧图像/深度尺寸和 dtype。尚未运行 FP/tag 离线推理或误差统计。
下一步：确认模型坐标系与标签固定 TF；静态和动态分别运行原版 FP 与 tag；严格按同源帧号匹配，考虑正方体旋转对称性。当前标签实测边长用户填写 10cm，需要测量黑色方形而非纸张白边。


## 2026-10-03 FP / AprilTag 对照结果

已完成几何转换、逐帧配对、统计和图表，报告：`/home/tomato/6Dpose/outputs/fp_tag_evaluation_075_final/REPORT.zh-CN.md`。35cm 箱体、7.5cm 标签，面中心到箱体中心175mm，未拟合 FP 数据。用户确认 static 为主动翻面测试，不作为静止抖动实验。

待完成：
- 固定相机和箱体，补录20–30秒真正静止数据，沿用7.5cm标签尺寸。
- 回看动态录像原始帧1829（约61.08秒），核查222mm中心差异尖峰，区分FP与Tag参考异常。
- 若需严格模型轴向旋转误差，独立标定标签到具体OBJ面的旋转；当前报告使用正方体几何对称角。
