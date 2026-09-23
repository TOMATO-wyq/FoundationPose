# ZED 相机拍照工具

这个目录包含两种使用方式：

1. `start_capture.sh`：本项目定制的双目拍照工具，照片始终保存到本目录的 `captures/`。
2. `start_zed_explorer.sh`：启动 Stereolabs 官方 **ZED Explorer** 图形工具。

当前电脑已安装 ZED SDK 5.5.0，检测到的相机为 `ZED`（S/N `24807`，`/dev/video4`）。

## 最简单的用法

连接相机后，在终端运行：

```bash
cd /home/tomato/6Dpose/zed_photo_capture
./start_capture.sh
```

窗口内：

- 按 `空格`、`s` 或 `回车`：拍一组照片；
- 按 `q` 或 `Esc`：退出。

首次运行会自动编译，以后会直接启动。

## 照片保存在哪里

每次启动会建立一个按时间命名的会话目录：

```text
/home/tomato/6Dpose/zed_photo_capture/captures/YYYYMMDD_HHMMSS/
├── rgb/                # 与深度对齐的左目校正彩色图，PNG
├── right/              # 右目校正彩色图，PNG
├── stereo/             # 左右目并排图，PNG
├── depth/              # 16 位深度 PNG，单位毫米；0 表示无效
├── depth_npy/          # float32 原始深度，单位毫米；保留 NaN/Inf
├── depth_visual/       # 伪彩色深度预览，不用于计算
├── point_cloud/        # XYZRGBA 彩色点云，PLY，坐标单位毫米
├── cam_K.txt           # 3×3 相机内参矩阵，FoundationPose 可直接读取
├── camera_info.json    # 内参、畸变、单位及对齐方式说明
└── capture_info.csv    # 拍摄时间、相机时间戳、序列号、分辨率和文件名
```

做 6D Pose 时使用同一序号的 `rgb/` 与 `depth/`，并读取根目录中的 `cam_K.txt`。
ZED 输出的深度已经对齐到校正后的 `rgb/` 图像。`depth/*.png` 可由 OpenCV 以
`cv2.imread(path, -1)` 读取，再乘 `0.001` 转换为米。
`rgb/` 与 `depth/` 中对应帧的文件名完全相同，可直接适配本项目
`FoundationPose/datareader.py` 的目录读取规则。

`depth_npy/*.npy` 可用 `numpy.load()` 读取，值也是毫米；它保留 ZED 对过近、过远及无效像素的
NaN/Inf 表达。`depth_visual/` 仅用于肉眼检查，不能作为算法输入。

## 无窗口自动拍照

拍 10 组，每 2 秒一组：

```bash
./start_capture.sh --no-preview --count 10 --interval 2
```

只拍 1 组：

```bash
./start_capture.sh --no-preview --count 1
```

选择分辨率、帧率或相机编号：

```bash
./start_capture.sh --resolution HD720 --fps 60 --camera-id 0
```

查看全部参数：

```bash
./start_capture.sh --help
```

## 官方工具：ZED Explorer

Stereolabs 官方提供的工具就是 **ZED Explorer**。本机可直接运行：

```bash
ZED_Explorer
```

或者在本目录运行：

```bash
./start_zed_explorer.sh
```

用法：

1. 右上角选择相机；
2. 左下角的相机图标用于截图，红色录制按钮用于录制 SVO/SVO2；
3. 点击序列号旁的齿轮，进入 **Application Settings**；
4. 在 **SVO and Screenshot Folder** 中设置截图和 SVO 的保存目录；
5. **Image Save Format** 可选 PNG/JPEG/BMP，建议 PNG。

列出相机：

```bash
ZED_Explorer --all
```

无界面录制一段 SVO2（按 `Ctrl+C` 停止）：

```bash
mkdir -p /home/tomato/6Dpose/zed_photo_capture/svo
ZED_Explorer -o /home/tomato/6Dpose/zed_photo_capture/svo/recording.svo2
```

SVO2 是 ZED 的原生录像格式，包含原始相机流、时间戳和可用的传感器数据；深度图等派生数据在回放时重新计算，并不是直接存进文件。

## 排错

如果打不开相机：

```bash
ZED_Explorer --all
/usr/local/zed/tools/ZED_Diagnostic
```

确认状态为 `AVAILABLE`，并退出其他正在占用相机的程序（同一台相机不能同时被本工具和 ZED Explorer 打开）。
