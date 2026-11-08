# TKK 小方块分割权重

`best.pt` 是本项目 20 cm 红蓝小方块的 YOLO 实例分割权重，类别为 `{0: tkk}`，模型任务为 `segment`。识别端仍通过 RGB-D 几何估计完整位姿和朝向；网络仅提供实例区域。

- 来源：`runs/tkk_seg_bg_from276_20261029_125307_400e/weights/best.pt`。
- SHA-256：`6e3b4283239fb9299ea956db2a2f70d27cff7d4be8f1a109569766baeca57aed`。
- 已验证加载环境：Python 3.10、Ultralytics 8.3.161、Torch 2.7.1、Torchvision 0.22.1。
- 文件直接纳入 Git，普通 clone 即可取得，无需 Git LFS 或另行下载。
- 配置 `src/config/box4d_d435i_yolo_live.json` 通过相对配置文件目录的路径引用该权重，不依赖用户名或 clone 目录。

部署步骤见仓库根目录的 [部署说明](../../../BOX4D_DEPLOY.zh-CN.md)。大块 35 cm 分支使用几何识别，不加载本权重。
