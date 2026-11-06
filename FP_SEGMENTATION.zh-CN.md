# 可选：给原版 FP 提供目标分割 Mask

**FP 的 `register` 需要当前图像中目标的二值 Mask，来源可以是人工标注、YOLO、SAM 或你自己的算法。** 正常跟踪使用 `track_one`，不要求每帧重新分割。官方 demo 已有 Mask，不需要本篇。

本篇把分割作为单独工具：先对一张图片生成 Mask，再把它交给 FP；不要求下载完整 KFS 版本，也不要求接相机。安装原版见[基础篇](FP_INSTALL_FROM_ZERO.zh-CN.md)，实时图像输入见[相机桥接篇](FP_CAMERA_BRIDGE.zh-CN.md)。

## 1. 单独建立分割环境

在 Ubuntu 宿主机执行，**不进入 FP 容器，也不修改 FP 已安装的 Torch**。这里选择 CPU 验证输出，尚不是实时性能配置。

```bash
sudo apt update
sudo apt install -y python3-venv git curl
python3 -m venv ~/fp-seg-env
source ~/fp-seg-env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install ultralytics==8.3.161
mkdir -p ~/fp_workspace/sample
```

将待处理彩色图片保存为 `~/fp_workspace/sample/rgb.png`，使用目标的原始图像，不要给它另做裁剪缩放。以后打开新终端时先 `source ~/fp-seg-env/bin/activate`。

## 2. YOLO：按类别自动选择目标

以下使用与固定 Ultralytics 版本兼容的 `yolo11n-seg.pt` 作通用示例，首次运行需联网下载权重。`bottle` 是示例类别，需要按你的模型与目标修改；通用模型不认识所有自定义零件，蓝色 KFS 需要专门的分割权重。参考 [YOLO 实例分割接口](https://docs.ultralytics.com/tasks/segment/)。

```bash
cd ~/fp_workspace/sample
python - <<'PY'
import cv2
import numpy as np
from ultralytics import YOLO

image = cv2.imread('rgb.png')
assert image is not None, '请先放入 rgb.png'
model = YOLO('yolo11n-seg.pt')  # 自定义目标替换成自己的 best.pt
assert model.task == 'segment', '需要实例分割权重，不是仅检测框的权重'
target_name = 'bottle'         # 修改为权重实际支持的目标类别
class_ids = [i for i, name in model.names.items() if name == target_name]
assert class_ids, f'模型不含目标类别，可用类别：{model.names}'
result = model.predict(image, classes=class_ids, conf=0.5, imgsz=640,
                       retina_masks=True, device='cpu', verbose=False)[0]
assert result.masks is not None and len(result.boxes), '没有检测到目标，不能继续注册'
index = int(result.boxes.conf.argmax().item())
mask = result.masks.data[index].cpu().numpy() > 0.5
assert mask.shape == image.shape[:2] and mask.any()
assert cv2.imwrite('mask.png', mask.astype(np.uint8) * 255)
print('Mask saved:', mask.shape)
PY
```

**成功标志：生成与 `rgb.png` 同尺寸的 `mask.png`，前景 255、背景 0，并且肉眼核对覆盖的是正确目标。** 置信度高不代表 Mask 一定正确；多实例时此示例仅取目标类别中置信度最高的一个，若要跟踪指定实例，需要另行选择。

常调参数：`conf` 降低可能少漏检但多误检；`imgsz` 增大可能保留更多细节并增加开销；`retina_masks=True` 让返回 Mask 对应原图像素。

<details>
<summary>可选 SAM2：用框或点提示生成 Mask</summary>

SAM2 根据提示分割，不会仅凭名称自动识别“我要的蓝色箱体”。本实验中的 HSV 粗定位是额外模块，不是 SAM2 自带功能。仍在同一独立分割环境操作，CPU 首次验证较慢。

```bash
mkdir -p ~/fp_deps
git clone https://github.com/facebookresearch/sam2.git ~/fp_deps/sam2
git -C ~/fp_deps/sam2 checkout 2b90b9f5ceec907a1c18123530e92e794ad901a4
SAM2_BUILD_CUDA=0 python -m pip install -e ~/fp_deps/sam2
mkdir -p ~/fp_deps/sam2/checkpoints
curl -fL -o ~/fp_deps/sam2/checkpoints/sam2.1_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

参考 [SAM2 官方接口](https://github.com/facebookresearch/sam2)；不编译可选 CUDA 扩展。下面示例的框必须按图片中的目标位置修改：

```bash
cd ~/fp_workspace/sample
python - <<'PY'
import os
import cv2
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

image = np.asarray(Image.open('rgb.png').convert('RGB'))
checkpoint = os.path.expanduser('~/fp_deps/sam2/checkpoints/sam2.1_hiera_small.pt')
model = build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml', checkpoint,
                   device='cpu', apply_postprocessing=False)
predictor = SAM2ImagePredictor(model)
predictor.set_image(image)
box = np.array([50, 50, 250, 200], dtype=np.float32)  # 示例框：改成目标的 x0,y0,x1,y1
masks, scores, _ = predictor.predict(box=box, multimask_output=False)
mask = masks[0] > 0
assert mask.shape == image.shape[:2] and mask.any()
assert cv2.imwrite('mask_sam.png', mask.astype(np.uint8) * 255)
print('SAM mask saved')
PY
```

生成文件后仍需查看轮廓；不合适时先改提示框 / 提示点。单张图的分割成功不等于位姿估计成功。

</details>

## 3. Mask 怎么交给 FP

**RGB、深度、内参和 Mask 必须对应同一帧、同一像素坐标。** 独立环境可以只保存 PNG 交给 FP，无需把 YOLO / SAM 安装进 FP。FP 读入后用 `mask > 0` 转为布尔数组；深度按米，Mesh 尺寸与实物一致。

- **离线序列：**原版 `run_demo.py` 的 reader 从 `masks/` 读取与首帧 `rgb/` 同名的 Mask。将生成结果放在该位置，保留同帧深度及 `cam_K.txt` 等输入，然后用 `--test_scene_dir` 与 `--mesh_file` 指向自己的数据与模型。不要覆盖官方演示数据作混用。
- **现有自写程序：**在首次初始化前调用分割器，将结果作为 `est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)` 的 `ob_mask`；后续调用 `est.track_one`。这是 API 接法，`est` 需先按原版示例创建。
- **实时相机：**若在不同进程中分割，需保留该帧的 RGB-D / 内参与帧号一起注册，不能把几秒前保存的 Mask 套到新画面。相机篇的手画入口不会自动读取本篇分割输出；要用自动分割入口需接入上述接口或选择下一项。
- **直接复现本项目：**使用[完整 KFS 方案](FP_KFS_FULL_INSTALL.zh-CN.md)，其中 YOLO / SAM、Mask 深度修正、检查和恢复已接好。它是额外选择，不是原版 FP 的安装前提。

本篇的图片输出与接口示例已静态核对，独立环境尚未重新运行验证；不要将其耗时与本项目 GPU 实时实验表直接比较。
