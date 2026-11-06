# 方块识别滑块中文对照表

英文名称与 **09 Tuning - live parameters** 窗口逐项对应，共 53 项。可以先按所在阶段找到要调的项目，
每次只改一项，再观察对应窗口。下面“调大后的作用”描述参数的方向，不表示调大一定更准确。

参数窗口分为 **8 页，每页最多 7 项**。点击底部 **上一页／下一页**，或在调试窗口按 **[ / ]** 翻页，
无需滚动；每个滑块直接显示中文与英文名称。切页保留修改，按 **S** 保存全部页。
8 页依次是：输入深度与 Canny、深度边缘与颜色、颜色模型与平面验收、PCL 法线与平面、候选与尺寸验收、位姿检查与重检测、跟踪与特征点、光流与补点。
已经打开的旧窗口需先按 S 保存，退出并用原命令重启后才会变为分页窗口。

## 先看截图中的红线

截图标题是 **03 RGB + depth edges**，底部显示 `Canny 76/255`、`depth jump >20mm`。

- **白线**：彩色图的 Canny 边缘，包含方块外轮廓、表面印刷图案和背景纹理。
- **红线**：水平或垂直相邻的两个有效深度像素，深度差超过 20 mm。
- 红线来自原始深度差，不是 PCL 法线异常标记。方块轮廓、遮挡边界和杂物之间本来就可能出现红线。
- 03 窗口的黑底表示“没有检测到边缘”；01 右侧深度图的黑色才表示无效或超出范围的深度。

这张图的大方块正面内部没有密集红色碎线；红色主要沿方块部分轮廓、接触处和左右下方杂物分布。
仅凭这张边缘图不能确认传感器深度有问题。如果一块实际平整的面内部持续出现、且随时间乱跳的红色碎点，
再结合 01 深度图和保存的 RGB-D 检查是否有深度噪声。

**针对红线，先调 `Depth edge mm`：按实际门限从 20 → 30 → 40 mm 逐档比较。**
调大后，小于新门限的深度差不再画成红线，也不再作为这一路的边缘证据；同时可能丢掉一些真实边缘。
这是改变边缘判定条件，原始深度和点云质量不会因此改善。

`Canny low/high` 控制白线；`PCL smooth px` 控制法线估计，它们都不直接减少 03 的红线。
截图里 Canny 高阈值已到当前配置允许的 255，表面图案仍有白线，是因为图案本身对比强，不能指望只提高门限就留下外轮廓。

## 数字、单位和保存

新启动的版本会让滑块旁的数字表示名称中的单位：`Depth edge mm = 40` 就是 40 mm，
保存到 JSON 的 `depth_edge_threshold_m` 则是 `0.04`。JSON 中距离用米、时间用秒、百分比用 0～1 的小数。

以下名称包含缩放：

| 英文名称 | 滑块数字的读法 |
|---|---|
| Reprojection x10 px | 30 表示 3.0 像素；实际校验门限还可能随相机图像宽度缩放，以 07 为准 |
| Flow back x10 px | 10 表示 1.0 像素 |
| Track RANSAC x10 px | 20 表示 2.0 像素 |
| PCL curvature x.01 | 3 表示 0.03 |
| PCL depth change x1000 | 20 表示 0.020，这是 PCL 法线邻域的深度变化系数 |
| Min face area px | 面积单位是像素平方，不是边长像素 |

旧版运行进程的滑块数字是带偏移的刻度编号。例如 `Depth edge mm` 的旧数字 18 实际是 20 mm，
`PCL smooth px` 的旧数字 9 实际是 12 像素。**这次先按 S 保存现有参数，再重启即可使用修正后的数字显示。**
当前运行进程里可先看 03、04、07 底部打印的实际门限。

按 **S** 会把完整配置写回启动时的 `--config` 文件，同时在输出目录另存 `tuned_config.json`，并保存最近显示的调试画面、
RGB-D 与测量值。冻结的画面仍属于原来的检测帧；保存的当前配置和画面中检测时的配置可能不同，`measurements.json` 保留后者。

## 01：距离范围与有效深度

| 英文滑块 | 中文含义／单位 | 调大后的作用 | config 字段 |
|---|---|---|---|
| Z min mm | 最近有效距离，毫米 | 排除更多近处深度；真实目标太近也会被排除 | `min_depth_m` |
| Z max mm | 最远有效距离，毫米 | 纳入更远的深度和背景 | `max_depth_m` |
| Depth coverage % | 候选面内部最低有效深度比例 | 要求更完整的深度，缺洞多的面更容易被拒绝 | `min_depth_fraction` |
| Min depth samples | 最少有效深度点数 | 小面、窄面或缺洞多的面更容易被拒绝 | `min_depth_samples` |
| Max depth samples | 单个候选面最多采样点数 | 拟合使用更多点，通常增加计算量；不会填补深度孔洞 | `max_depth_samples` |

`Z min` 必须小于 `Z max`，`Max depth samples` 必须不小于 `Min depth samples`。

## 03：白色图像边缘与红色深度边缘

| 英文滑块 | 中文含义／单位 | 调大后的作用 | config 字段 |
|---|---|---|---|
| Canny low | Canny 低阈值 | 减少连接到强边缘的弱边缘和纹理；也可能断开方块边界 | `canny_low` |
| Canny high | Canny 高阈值 | 对强边缘种子要求更高，通常白线变少 | `canny_high` |
| Depth edge mm | 相邻有效深度的跳变门限，毫米 | 红线变少，较小的真实深度边界也可能消失 | `depth_edge_threshold_m` |

`Canny high` 要大于 `Canny low`。法线方向不同的两面可以深度连续，不能要求公共棱一定有红线。

## 02：颜色候选与颜色验收

| 英文滑块 | 中文含义／单位 | 调大后的作用 | config 字段 |
|---|---|---|---|
| Color min S | 红蓝分类的最低饱和度，0～255 | 排除灰白、低饱和像素；褪色或反光区域也可能漏掉 | `color_min_saturation` |
| Color min V | 红蓝分类的最低亮度，0～255 | 排除暗部；阴影中的方块像素更容易漏掉 | `color_min_value` |
| Proposal min S | 黄色宽松候选区域的最低饱和度，0～255 | 减少宽松颜色候选区域 | `color_proposal_min_saturation` |
| Proposal min V | 黄色宽松候选区域的最低亮度，0～255 | 减少暗部宽松候选区域 | `color_proposal_min_value` |
| Color min samples | 最少红蓝分类样本数 | 对目标颜色证据要求更高 | `color_min_samples` |
| Color coverage % | 表面采样点中红蓝分类的最低占比 | 对目标颜色覆盖要求更高 | `color_min_coverage_fraction` |
| Color fit % | 红蓝半块模型的最低拟合正确率 | 颜色朝向判断更严格，更多情况显示朝向未确定 | `color_min_fit` |
| Color ambiguity % | 最佳与次佳颜色朝向模型的最低分差 | 更谨慎地确认颜色朝向，歧义结果更多 | `color_ambiguity_margin` |

红蓝色相范围在 `red_hue_ranges`、`blue_hue_range` 中编辑。颜色已经稳定时，可以先不动这一组。

## 04、05：法线与平面区域

| 英文滑块 | 中文含义／单位 | 调大后的作用 | config 字段 |
|---|---|---|---|
| Plane dist mm | 点到平面的内点距离门限，毫米 | 容忍更多深度误差，区域可能更完整；也更可能混入其他面 | `plane_threshold_m` |
| Plane RMS mm | 面内平面拟合 RMS 误差上限，毫米 | 放宽平整度验收；不会主动把平面变平滑 | `plane_max_rmse_m` |
| Plane support % | 普通候选面最低平面内点比例 | 更严格地拒绝混合深度或跨面的候选 | `plane_min_support_fraction` |
| Partial plane % | 遮挡／半面相关路径的最低平面支持率 | 对不完整面的平面证据要求更高 | `partial_min_plane_support_fraction` |
| Partial edge % | 部分遮挡候选的最低边缘支持率 | 可见边界要求更严格 | `partial_min_edge_support_fraction` |
| PCL stride | 点云采样步长；1 为不降采样 | 点数更少、通常更快，但窄面和小面更容易漏掉 | `pcl_stride` |
| PCL smooth px | 法线估计的邻域平滑范围，原图像素 | 大面法线可能更稳定；窄侧面和转折处可能被混合 | `pcl_normal_smoothing_px` |
| PCL angle deg | 平面区域允许的法线方向差，度 | 区域更容易连接；不同朝向的面也更可能被合并 | `pcl_angular_threshold_deg` |
| PCL curvature x.01 | PCL 允许的最大局部曲率 | 对局部不平整的容忍更高 | `pcl_max_curvature` |
| PCL depth change x1000 | 法线邻域对深度突变的容忍系数 | 深度变化处可能保留更大的法线邻域，也更容易跨过边界；不改变 03 的红线门限 | `pcl_depth_change_factor` |
| PCL local RANSAC iters | 局部平面拟合的 RANSAC 迭代上限 | 尝试更多随机模型，耗时可能增加；不会保证找出更多面 | `pcl_ransac_iterations` |
| Max scene planes | 供后续候选生成使用的平面预算 | 给更多区域生成候选的机会，也增加背景候选和工作量 | `max_scene_planes` |

PCL 参数仅在 `plane_backend: "pcl"` 时生效。`Plane dist` 同时影响场景平面区域和局部面拟合，
`Plane RMS/support` 主要是验收条件。不要把降低验收标准当成输入深度已经变好。

## 06：面候选和已知尺寸

| 英文滑块 | 中文含义／单位 | 调大后的作用 | config 字段 |
|---|---|---|---|
| Min face area px | 最小候选面面积，像素平方 | 丢掉更多小碎片；真实小面、窄侧面也可能被丢掉 | `min_face_area_px` |
| Max face candidates | 初始四边形候选预算 | 检查更多候选，耗时和杂乱框可能增加；派生候选还可能另加 | `max_faces` |
| Max objects | 最终场景最多输出的方块数量 | 允许更多目标输出，不保证一定找到这些目标 | `max_objects` |
| Min edge px | 最短候选边长，像素 | 更严格地排除小面和窄面 | `min_edge_px` |
| Edge support % | 候选边界与实测边缘的最低吻合比例 | 候选和跟踪边界验收更严格 | `min_edge_support` |
| Stack edge % | 堆叠和部分直线候选的最低边缘支持率 | 对堆叠拆分／直线候选的证据要求更高 | `stack_min_edge_support_fraction` |
| Size tolerance % | 面边长相对已知边长的允许误差 | 更多尺寸偏差能通过，也可能接受错误轮廓 | `size_tolerance_fraction` |

实际边长列表在 `sizes_m` 中，单位为米。应先保证实际尺寸正确，再调尺寸容差。
把半个面当作完整正方形的情况，通常需要半面组合或遮挡模型，不能靠无限放宽尺寸误差解决。

## 07：位姿结果的几何一致性

| 英文滑块 | 中文含义／单位 | 调大后的作用 | config 字段 |
|---|---|---|---|
| Reprojection x10 px | 模型角点到观测角点的重投影误差上限 | 允许更大的图像对齐误差；30 表示参考宽度下 3.0 像素 | `max_reprojection_px` |
| Depth residual mm | 位姿所代表的面与当前深度的残差上限，毫米 | 允许位姿与深度更不一致 | `max_depth_residual_m` |
| Normal error deg | 位姿面法线与深度平面法线的夹角上限，度 | 允许更大的方向不一致 | `max_normal_error_deg` |
| Upright tilt deg | 相对竖直方块模型的最大倾斜角，度 | 放宽直立约束；需要配置或测得 `up_camera` 才生效 | `max_upright_tilt_deg` |

## 08：跟踪、漂移检查与重检测

| 英文滑块 | 中文含义／单位 | 调大后的作用 | config 字段 |
|---|---|---|---|
| Redetect ms | 正常跟踪期间的全图重检测间隔，毫秒 | 正常时更少全图搜索；目标丢失仍可立即触发检测 | `redetect_interval_s` |
| Detect age ms | 可接受的后台检测结果最大年龄，毫秒 | 允许使用更早的检测来重新获取，但仍要通过当前图像／深度检查 | `detector_max_age_s` |
| Track gap ms | 距上次成功跟踪允许的时间间隔，毫秒 | 容忍更久的短暂中断；不会直接发布旧帧位姿 | `track_max_gap_s` |
| Track travel cm | 单次位姿更新允许的平移量，厘米 | 容忍更快移动，也更容易接受大跳变 | `track_max_distance_m` |
| Track min features | 最少可用跟踪点数 | 跟踪证据要求更高，少点时更容易判丢失 | `track_min_features` |
| Seed max features | 初始化时最多选取的特征点数 | 有机会覆盖更多纹理，计算量增加；目前仍主要从一个可用面选点 | `track_seed_max_features` |
| Reseed min points | 点数低于此值时尝试补点 | 更早触发补点，仍受补点间隔限制 | `track_min_reseed_features` |
| Feature coverage % | 跟踪点凸包相对该面面积的最低占比 | 更严格要求点分散；不会主动生成侧面特征点 | `track_min_feature_coverage_fraction` |
| Flow back x10 px | 光流前后向一致性误差上限 | 接受更不一致的点；10 表示 1.0 像素 | `track_fb_max_error_px` |
| LK error | LK 光流图像块误差上限，非像素位移 | 接受外观变化更大的点，也可能保留错误匹配 | `track_lk_max_error` |
| Track RANSAC x10 px | 跟踪单应矩阵的 RANSAC 重投影门限 | 更多运动偏差点能成为内点；20 表示 2.0 像素 | `track_homography_ransac_px` |
| Track inliers % | 同一运动模型内点的最低比例 | 对跟踪运动一致性要求更高 | `track_min_inlier_fraction` |
| Track quad drift px | 位姿投影四边形与光流预测四边形的平均偏差上限 | 放宽两种几何结果之间的核对；不是累计漂移距离 | `track_quad_max_drift_px` |
| Reseed interval ms | 两次补点之间的最短时间，毫秒 | 补点不那么频繁；调小会更频繁尝试补点 | `track_reseed_interval_s` |
| Track depth inset % | 跟踪平面失败后备用采样区域每边内缩的比例；第 8 页 | 更少混入面边缘的背景，但能使用的深度区域更小；10 表示每边内缩 10%，0 关闭 | `track_depth_inset_fraction` |

JSON 另有 `track_feature_method`：`adaptive` 为默认值，Harris 角点不可用时尝试
Shi–Tomasi；`harris` 为原来的选点方式；`shi_tomasi` 为直接使用 Shi–Tomasi。
这是算法选择，在配置文件中修改，没有对应的数字滑块。补选更多点后，后续几何和深度检查仍然有效。

## 建议的排查顺序

1. 只看 03 的红线：先动 `Depth edge mm`，看平整面内部的碎线与真实轮廓是否一起消失。
2. 04、05 目标面太碎：再分别试 `PCL smooth px`、`PCL angle deg`、`Plane dist mm`；每次动一项。
3. 06 的面框正确但 07 拒绝：按 N/P 找到对应 F 编号，针对它实际失败的尺寸、深度覆盖、RMS 或支持率调节。
4. 首次检测能过、随后框消失：看 08 的失败原因，再调整相应跟踪门限。
5. 每轮按 S 保存。比较 `measurements.json` 中的实际测量值，避免同时放宽所有校验后看不出是哪一步产生影响。

完整窗口说明见 [逐阶段调试说明](BOX4D_DEBUG.zh-CN.md)。
