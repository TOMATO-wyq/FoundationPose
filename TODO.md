# 待续：上半部分方块定位与历史约束

记录日期：2026-11-11。用户因电量不足准备关机，要求停止实现、把未完成工作写为代办。

**状态：本轮改动已写入文件，但尚未完成验收。不要把上一轮的“369 passed”当作当前工作区已通过测试；当前代码须先完成下面的检查，实机复查先使用 `dry_run:=true`。没有自动启动相机或实车。**

## 用户确认的行为

- 只恢复“上边完整、两侧边可见”的上半部分目标，必须有当前深度校验。
- 上一次位置作为参考，第一次不完整观测不能仅因位置跳变被固定门限一票否决。
- 第二次不完整观测可以更严格，并允许历史位置成为验收门限。
- 当前实现按“同一目标连续两次不完整观测”理解；完整观测后重置计数。短暂漏检不得通过换 ID 或重新检测绕过连续检查。
- 历史约束考虑源时间间隔、运动和相机旋转；历史只能验证当前观测，不能用旧位姿代替当前深度。
- 其余实现细节用户授权自行判断。下次直接继续，不必重复询问已确认的规则。

## 已写入的在途实现（仍需检查）

- `src/box4d/partial.py`
  - 新增 `upper_model_allowed`、`upper_boundary_evidence`。
  - 支持仅推算出的下角离开物理图像边界；有意设置的搜索 ROI 裁切仍拒绝。
  - `visibility` 对画外采样安全取值并单独记录 `outside_image_fraction`。
  - `partial_samples` 改采样真实 `visible_outline_quad`，记录可见区域的原始平面支持率；缺失深度不能抬高支持率。
- `src/box4d/pose.py`
  - 部分面求解重新检查当前上边、两侧边与深度，输出 `upper_partial`、`anchor_edge_support`、`visible_plane_support_fraction` 等。
- `src/box4d/partial_history.py`（新文件）
  - `UpperPartialHistory.evaluate/commit`：首次不完整观测使用软参考；连续第二次开始检查动态位置范围、至少 85% 可见平面支持、三条锚定边分别至少 70% 支持。
  - 动态位置范围随时间和速度不确定性增长；用 gyro 变换历史相机坐标。
  - 只有最终选中且通过的当前观测才能更新历史；短暂漏检和重叠检测候选不能清空历史绕过检查。
- `src/box4d/temporal.py`
  - 在最终场景选择前执行部分观测历史验收，最终输出后提交历史；校准/坐标参考变化时清空历史。
  - 异步部分目标种子仅从可见区域取深度和特征，不把推算的下角当成 PnP 观测。
  - 保存 `visible_quad_model` 并在当前图像重新采样、验证。
  - 暂停对不完整目标的普通整面补点；重新得到完整检测后可以接管完整几何并重置部分状态。
  - 图像裁切跟踪现在要求上边和两侧可见；顶部也出画时应停止输出。
- `src/tests/test_box4d_upper_partial.py`（新文件）：12 项测试，包含几何、确定性的异步种子/推进、历史软/硬约束、相机旋转、多目标、重复候选等。
- `src/tests/test_box4d_large_faces.py`：旧的“上下均出画仍继续跟踪”用例已按新的上半部分规则改为“物理上边出画后停止”。

## 待办（按顺序）

- [ ] 修正新测试的场景构造，再跑目标测试。
  - 最近一轮新测试：`10 passed, 2 failed`。
  - 两个失败：`test_cold_start_upper_half_at_image_edge_has_current_depth_pose`、`test_only_lower_half_or_one_side_cannot_initiate_pose`。
  - 已定位一个测试场景问题：渲染完整立方体时，即使侧面出画，仍可能看到完整的顶面/底面。检测器此时通过完整水平面定位并不等于错误地恢复残缺侧面，因此不能直接断言“没有任何输出”或一定有 `upper_partial` 字段。
  - 建议在这两个测试中隔离前侧面：非遮挡场景只保留接近前侧面 z 的 RGB-D，屏蔽其它完整面；或者只断言残缺侧面路径不能输出，保留完整面独立定位能力。遮挡场景必须保留真正的前景遮挡物深度，不可一并屏蔽。
- [ ] 重新跑已有跟踪/诊断回归。
  - 曾试图按单条弱边就认定“其它方向残缺”，引发 13 项既有跟踪测试失败：真实完整面有时也只有三条明显 RGB 边。
  - **这段过宽的 `unsupported_partial_outline` 检查已在暂停前删除，但尚未重新跑测试。旧日志中的这 13 个失败不代表删掉检查后的结果。**
  - 不要仅因完整面一条 Canny 边弱就拒绝；严格上边/两侧检查应针对实际的部分几何路径和图像裁切路径。
- [ ] 核查历史约束集成，尤其完整重新获取后是否正确重置、重复候选是否能绕过、未选中候选是否污染历史、同规格多个目标是否各自关联。
- [ ] 核查异步部分目标到当前帧的校验，不得直接发布旧检测源帧的位姿；当前深度/特征失效必须无输出。
- [ ] 补充上半部分底部出画、上下同时出画、相机倒装、ROI 裁切、前景遮挡、完整恢复等覆盖；保留水平完整面合法定位能力。
- [ ] 跑全量 Python 测试；有失败就修复，不要沿用上一轮已通过的测试数字。
- [ ] 用已有真实录制做新旧对照，检查支持上半部分后是否增加地面/墙面/图案误检；同步回放与异步节奏分别记录，不把输出覆盖率叫准确率。
- [ ] 更新中文说明和在途诊断记录，写明首次软参考、连续第二次加严、完整观测重置、上下文失效与坐标旋转策略。
- [ ] 最后告知用户实际通过的测试、回放结果和剩余限制；不自动恢复实车运动。

## 继续执行的命令

从本仓库根目录执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python3 -m pytest -q \
  src/tests/test_box4d_upper_partial.py \
  src/tests/test_box4d_occlusion.py src/tests/test_box4d_partial_samples.py \
  src/tests/test_box4d_large_faces.py src/tests/test_box4d_tracking_diagnostics.py \
  src/tests/test_box4d_temporal.py src/tests/test_box4d_yolo.py --tb=short

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python3 -m pytest -q src/tests --tb=short
```

## 现有证据和上一轮已完成工作

- 原始录制：`outputs/box4d_car_20261111_214634`，901 帧、约 60 秒，RGB-D 和 IMU 完整。
- 原始线上结果：`outputs/box4d_control_20261111_214559/poses.jsonl` 和同目录 `run_config.json`。
- 上一轮诊断、脚本、对照图：`outputs/car_diagnosis_20261111_214634/诊断与修正.md`、`comparison.jpg`、`replay_comparison.py`、`paced_async_check.py`。
- 上一轮已增加“实测面轮廓尺度拟合”与“水平颜色块周围平面延续拒绝”，不要回滚这些已完成改动。
- 上一轮 901 帧同步回放中有目标输出帧数 588 → 762；这不是准确率，不能直接代表当前在途代码。
- 地面误检发生在录制结束约 7.072 秒后，只有日志，没有当时原始 RGB-D；必须如实标注这部分仍需现场验证。
- 本轮运行日志备份位置：`outputs/upper_partial_wip_20261111/`。
- 工作区有大量本轮开始前已有的未提交改动；禁止 `git reset`、整文件回滚或覆盖其它工作。部分文件本来就是未跟踪状态，不代表可以删除。
