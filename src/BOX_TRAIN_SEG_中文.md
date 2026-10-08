# purple_box 实例分割训练

训练脚本：`box_train_seg.py`。使用独立标注环境，不使用 FP/Isaac ROS 环境。

```bash
/home/tomato/code/box_seg_annotation/env/bin/python /home/tomato/6Dpose/src/box_train_seg.py \
  --include-saved-group d435_20261003_193107_325620 \
  --epochs 150 --patience 0 --batch 2
```

默认从 Labelme JSON 导出 purple_box 多边形，并按拍摄顺序留出末尾 20% 验证，隔离中间 10 张；不修改原 JSON 和 manifest。上一次 138 train、37 val，仅属同场景粗略试训；上限 150 轮、patience=20，在第 48 轮早停，best.pt 来自第 28 轮。下次新训练默认 150 轮、patience=0，关闭早停；仍按验证 fitness 保存 best.pt，并保留 last.pt。运行输出保存在 `/home/tomato/code/box_seg_annotation/runs/`，结束后生成预测叠加图和二值 mask。

未使用的图片不自动等于负样本。确认图片中没有 purple_box 后，可将对应 manifest ID 每行一个写入文本文件，并在新训练命令增加 `--confirmed-negative-ids /path/to/confirmed_ids.txt`。仅清单内没有 JSON 的图片导出空标签，源图、源 JSON 和原 manifest 保持不变；负样本参与同一拍摄顺序划分及隔离带，并记录在本次 split/export report。已有 JSON 的无目标图片应保存空 shapes 并通过 reviewed 流程使用。被隔离的已标注图片、未复核草稿不能当成无目标图片。

用户已确认 `d435_20261003_193107_325620` 中未标注的 68 张图片都没有 purple_box。
确认清单保存在 `config/box_confirmed_negative_ids.txt`，下次新训练默认加入，不需要额外参数。
需要正样本基线时加 `--no-confirmed-negatives`；新确认清单可通过 `--confirmed-negative-ids` 替换。
原始 68 张会参与连续划分，隔离带内图片仍排除，所以实际导出数以 split report 为准。
加入这批负样本后，temporal 模式选取满足两类验证配额的最短连续验证片段，两侧各留 10 张隔离带，防止尾段全负导致无法验证分割。
当前预检为 182 train（130 正样本、52 负样本）和 51 val（37 正样本、14 负样本）；
共 20 张隔离，其中 2 张属于确认负样本。仍是同一场景的初步验证。

已运行 prepare-only 并通过官方 loader 检查，新快照为
`/home/tomato/code/box_seg_annotation/training_datasets/box_seg_20261004_223745_213216/`。
已核验 182 train、51 val，66 个空标签负样本，原图内容与原始标注均未改动；没有启动新训练。

新增负样本请开启新训练；`--resume` 使用旧 checkpoint 的参数和数据快照，不会应用新的 epochs/patience 默认值，也禁止追加负样本。

完整采集、标注、导出、训练和恢复步骤请打开 `/home/tomato/code/box_seg_annotation/README_中文.md`。
