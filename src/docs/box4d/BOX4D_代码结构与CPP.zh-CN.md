# Box4D 代码结构与 CPU C++ 计算阶段

本轮整理保留现有识别流程和验收阈值。35 cm 大块继续全图几何检测；20 cm 小块继续 YOLO-seg → 红蓝颜色门控 → ROI 平面/实测边线 → PnP 或遮挡几何 → 当前帧跟踪。不是通过降低验收标准提高输出数量。

## 代码入口

| 文件 | 职责和主要数据 |
|---|---|
| `box4d/cli/live.py` | ROS 输入、启动 UP 标定、显示和日志 |
| `box4d/cli/` 其它入口 | `capture.py` 录制，`offline.py` 回放，`benchmark.py` 计算测速，`estimate_up.py` 标定 |
| `box4d/segmentation.py` | YOLO 前端、输入转正/掩膜映回、实例内颜色占比；不算位姿 |
| `box4d/yolo.py` | 大块分支与各小块 ROI 的调度、有限缓存、实例归属检查 |
| `box4d/detector.py` | 一轮计算边缘/距离图，提出平面和面轮廓候选 |
| `box4d/roi_faces.py` | 复用同一轮图像证据，在小块 ROI 内拟合真实边线 |
| `box4d/geometry.py` | 相机投影、确定性 RANSAC、几何运算；保留 NumPy 参考实现 |
| `native/geometry.cpp` | C++/Eigen 平面 RANSAC、三轮协方差拟合、四边支持度采样 |
| `native/core_ops.cpp` | 四角排序与凸性检查、SVD 刚体配准 |
| `native/image_ops.cpp` | HSV 红蓝分类、深度跳变边缘、稠密点到平面筛选 |
| `native/face_ops.cpp` | 实测边线精修、ROI 四边 Huber 拟合、面内深度像素筛选与采样 |
| `native/refinement_ops.cpp` | 位姿联合优化每轮残差和解析雅可比，一次准备各面数据 |
| `box4d/native.py` | 加载 CPU 扩展、选择执行实现 |
| `box4d/pose.py` | 深度、尺寸、PnP、UP 验收和位姿优化 |
| `box4d/partial.py` | 有观测锚点和真实前景遮挡证据时恢复部分面 |
| `box4d/temporal.py` | 后台完整检测；主线程推进当前 RGB-D 轨迹；单槽覆盖最新任务 |
| `box4d/results.py` | 统一帧元数据、空观测和 ROI 诊断汇总 |
| `box4d/debug.py` | 原有 8 个步骤窗口、分页调参、S 保存 |

原来从 `box4d.yolo` 导入的 `YoloRegions`、`instance_colors`、`upright_quarter_turns` 仍可使用。

## 清理了什么

- 拆开分割前端与 ROI 调度，给外扩 ROI、几何计算、实例验收各自独立函数。
- 移除每次检测时对整个配置做 JSON 序列化的比较，改为配置值和独立快照比较。滑杆原地修改嵌套列表也能触发更新。
- 派生 ROI 配置交给 `Pipeline` 统一校验，不连续校验两次。
- ROI 子类复用主检测步骤已有的 RGB 边缘、距离图和实测面片，删除重复计算与暂存面片成员。
- 调试颜色图和部分面候选复用一次 HSV 分类结果；显示的 ROI 掩膜独立复制，不改变几何取色数据。
- ROI 范围在图像边缘入口检查一次；检测器统一提供候选拒绝列表，不再用 `getattr` 猜测是否有这个字段。
- 前端已保证实例非空，颜色比例直接使用真实像素数，删除用 `max(1, count)` 掩盖异常输入的处理。
- 统一 Python 格式，展开压在同一行的条件和赋值；补充关键坐标、深度来源、模型推断和线程所有权注释。

配置、标定、深度有效性、退化平面、源帧年龄以及当前帧证据的检查仍然必要。它们区分“测到了”和“复用了过期结果”。

## 这次实际迁移了什么

原先只迁移了平面拟合和边缘支持度两个内核。现在进一步迁移图像、面片和优化计算，均在原有链路中调用：

| 阶段 | C++ 处理的数据 | 保留的条件 |
|---|---|---|
| 颜色分类 | 整幅 HSV 或采样像素 → 红、蓝、未知标签 | 原 HSV 阈值，红蓝重叠按未知处理 |
| 深度边缘 | 深度图 → 相邻有效深度的跳变图 | 不跨缺失深度造边；仍标右侧/下侧像素 |
| 稠密面筛选 | 深度、标定射线、平面 → 内点掩膜 | 原点到面距离阈值，先按 float32 构造点 |
| 面内采样 | 四边形、深度 → 像素、深度和计数 | 原侵蚀、覆盖分母、均匀抽样位置 |
| 边线与角点 | 实际 RGB 边缘点 → Huber 直线和交点 | 原支持点数、跨度、角点位移门槛 |
| 角点排序 | 四个观测角点 → 固定绕序 | 原唯一性、凸性和面积检查 |
| 刚体配准 | 模型/实测 3D 点 → 旋转和平移 | 原最小二乘 SVD 和反射修正 |
| 联合优化计算 | 当前参数和各面 RGB-D → 残差、雅可比 | 原目标函数、权重和解析导数 |

Python 仍负责 ROS 输入、配置、实例/候选调度、跟踪状态、SciPy 优化器调用和调试界面。相机畸变射线仍通过现有 OpenCV 接口计算。这里还不是一个独立的全 C++ 可执行程序。

平面算法沿用原来的固定种子 RANSAC 假设、内点阈值、三轮协方差重估和朝相机的法线方向。采样索引由 Python 缓存后传入 C++，避免修改执行语言时同时修改随机假设。四边支持度沿用原来的采样位置和 3.5 px 距离门槛。内核执行时释放 GIL，不创建额外 CPU 线程池，也不使用 CUDA。

OpenCV、PCL 和 Torch 本身已经有原生实现。PnP、光流与 YOLO 推断的主要计算原本就不由 Python 解释执行；重写它们的外层调用不等于重新加速底层算法。现在边线筛选和拟合也已一起进入 C++，YOLO 仍使用 CPU。

这些阶段执行时释放 GIL，不使用 CUDA。Python 和 C++ 各自链接的 OpenCV 都受 `opencv_threads` 约束，默认 1；新扩展导入时也限制为 1，避免先启动另一份默认线程池。`run_config.json` 的环境记录包含两边的 OpenCV 版本和线程数。`auto` 只在扩展完全未编译时回退；如果加载到旧版或损坏的扩展，会明确要求重编，避免混用不同接口。

两个配置不要混淆：

| 参数 | 含义 |
|---|---|
| `plane_backend: "numpy"` | 保留兼容字段名，选原来的 RANSAC/面片流程；数值内核可在 NumPy 和 C++ 之间选择 |
| `plane_backend: "pcl"` | 选已有 PCL 面分割与局部平面拟合，需要原来的 PCL 扩展 |
| `geometry_backend: "auto"` | 默认值；CPU 扩展存在时用 C++，未编译时用 NumPy |
| `geometry_backend: "cpp"` | 强制 C++，未编译会明确报错，不会悄悄改用 NumPy |
| `geometry_backend: "numpy"` | 强制参考实现，便于数值和性能对照 |

`poses.jsonl` 的顶层 `geometry_backend` 和 `run_config.json` 中 `metadata.environment.geometry_backend` 记录实际选择。即使 `plane_backend=pcl`，四边支持度仍由 `geometry_backend` 选择。实时/离线命令均可用 `--geometry-backend` 覆盖配置。

## 编译与完整运行命令

本机已具备编译依赖，并已完成 Release 编译。新机器需要 CMake、C++17 编译器、对应 Python 开发头文件、pybind11、Eigen3 和 OpenCV 的 core/imgproc/calib3d 开发库；这个扩展不依赖 PCL。Linux 常用开发包名为 `python3-dev pybind11-dev libeigen3-dev libopencv-dev`。切换 Python 主/次版本或机器架构后，需要重新编译。

先编译，使用运行 YOLO 的同一个 Python：

```bash
cd /home/tomato/6Dpose
BOX4D_PYTHON=/home/tomato/code/box_seg_annotation/env/bin/python \
  bash src/build_box4d_native.sh
```

相机未启动时，在一个终端执行：

```bash
cd /home/tomato/6Dpose
bash src/start_box4d_camera.sh --domain 0
```

另一个终端运行，保持相机静止完成 `--auto-up`：

```bash
cd /home/tomato/6Dpose
CUDA_VISIBLE_DEVICES=-1 \
BOX4D_PYTHON=/home/tomato/code/box_seg_annotation/env/bin/python \
bash src/box4d_ros.sh --domain 0 live \
  --config src/config/box4d_d435i_yolo_live.json \
  --geometry-backend cpp --auto-up \
  --output "outputs/box4d_cpp_$(date +%Y%m%d_%H%M%S)" \
  --duration 600 --display --display-smoothed \
  --debug --debug-hz 4 --debug-layout windows
```

相机与识别程序保持同一 domain。原来的滑杆、分页和 S 保存照常使用。运行中改变相机安装方向仍需要重新启动并标定 UP；本轮没有增加动态 IMU 跟随。

## 自己复测

```bash
cd /home/tomato/6Dpose
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python3 -m unittest discover -s src/tests -p 'test_box4d_*.py'
```

`benchmark_box4d.py` 现在会按配置选择实际 YOLO/普通检测链路，不再在填了 YOLO 权重时只测普通 `Pipeline`。以下使用均匀抽样、预热、预加载图像，测量完整检测调用；将 `cpp` 换成 `numpy` 可作内核对照，输出目录必须不同。

```bash
cd /home/tomato/6Dpose
CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/tomato/code/box_seg_annotation/env/bin/python src/box4d/cli/benchmark.py \
  outputs/box4d_capture_20261028_223710 \
  --config src/config/box4d_d435i_yolo_live.json \
  --up-camera -0.0565213053 -0.9983147689 0.01315158565 \
  --method hybrid --geometry-backend cpp \
  --sample-uniform --max-frames 32 --repeat 3 \
  --output "outputs/box4d_cpp_bench_$(date +%Y%m%d_%H%M%S)"
```

这只测离线计算，不包含实时 ROS 输入、GUI、写盘，也不能换算成实际相机帧率。带跟踪的连续回放和带 debug 的异步回放应分别验证。

## 本轮五处逻辑修复

- 部分面恢复先过滤无效深度与非有限 3D 点，再拟合和计算中位数；缺失点仍计入原支持率分母，不把缺测伪装成更高覆盖率。
- 异步拒绝诊断完整保留源帧标定、时间、角点、外观拒绝、局部搜索和耗时，并注明年龄和可用性。后台耗时与旧像素不混入当前帧计算时延/拒绝列表。普通几何的局部颜色搜索也汇总完整诊断。
- 快照写入失败记录路径与错误，释放队列位置，识别继续运行。S 保存配置采用同目录临时文件替换；原配置与输出副本逐项报告成功/失败，失败后不误报快照成功。
- IPPE 与 ITERATIVE 候选全部验收后，若没有有效结果，再执行一次联合优化兜底；空解和求解异常也可进入兜底。兜底仍经过相同深度、重投影、法线、UP、漂移、边界和跳变门槛。
- 堆叠拆分只捕获 `FaceSizeMismatch` 类型，不再依赖错误文案。其它 `ValueError` 不会误触发拆分。
