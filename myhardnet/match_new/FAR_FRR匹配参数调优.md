# FAR / FRR 匹配参数调优指南

本文针对 `match_new/config_match_new.yaml`，整理会影响最终 FAR（错误接受率）和 FRR（错误拒绝率）的参数，并按调参优先级从高到低排序。

## 1. 先明确两个结论

1. `identification.match_score_threshold` 对最终数值影响最直接，但它只能在 FAR 和 FRR 之间取舍，不能提高模型本身的区分能力。
2. 想同时降低 FAR 和 FRR，需要优先改善 genuine 与 impostor 分数的分离度，重点调整纹理融合、描述子候选、RANSAC 和注册模板融合参数。

参数影响方向均指“其他参数和分数阈值固定”时的常见趋势。除最终分数阈值外，大部分中间参数会同时改变 genuine 和 impostor 分数，实际结果可能非单调，必须以全量实验为准。

## 2. 当前已有实验基线

仓库内目前可用的一组完整 identity 级实验为：

- 模型：`hardnet_strongv2_256_fixed`
- 数据：`normal_pic`
- identity 数：31
- 每个 identity 注册模板数：40
- genuine 尝试：1865
- impostor 尝试：55950
- 候选策略：`ratio_only`
- 距离：L2
- 纹理融合：开启，几何/纹理权重为 `0.70/0.30`

关键结果：

| 操作点 | FAR | FRR | 说明 |
| --- | ---: | ---: | --- |
| 阈值 `0.5326` | `1.787e-5` | `4.29%` | FAR 达到 `1/50000` 目标附近，FRR 未达到 2% |
| 阈值约 `0.40` | `0.1001%` | `2.198%` | FRR 接近目标，但 FAR 明显超标 |
| 阈值 `0.54` | `0` | `4.665%` | 当前样本中无错误接受，但 FRR 更高 |

因此，现有证据不支持“只微调最终阈值即可同时达到 FAR < 1/50000、FRR < 2%”。另外，当前配置已切换到 `m2_256_512_v2`，但尚未找到该模型对应的完整匹配实验，不能直接复用上述阈值。

当前 impostor 总数为 55950，出现 1 次错误接受就对应 FAR `1/55950 = 1.787e-5`。该样本量下，目标 FAR 对单个错误样本非常敏感，最终结论应在更大数据集上复核。

## 3. 核心参数优先级

### 第 1 级：直接决定工作点

| 排名 | 参数 | 当前值 | 增大或放宽后的常见影响 | 合理调整区间 | 首轮建议尝试 |
| ---: | --- | ---: | --- | --- | --- |
| 1 | `identification.match_score_threshold` | `0.65` | 阈值增大：FAR 降、FRR 升；阈值减小则相反 | 代码合法范围 `[0,1]`；当前分数体系重点观察 `0.45~0.60` | 不建议手工猜单点。先使用完整阈值曲线，再重点检查 `0.50~0.56`；旧模型参考点为 `0.5326`，新模型必须重标定 |

注意：`evaluation.auto_thresholds` 和 `score_threshold_step` 主要控制曲线扫描与输出精度，不改变匹配分数。当前 `true/0.01` 可保持。

### 第 2 级：identity 融合与注册模板覆盖

| 排名 | 参数 | 当前值 | 增大或改变后的常见影响 | 合理调整区间 | 首轮建议尝试 |
| ---: | --- | ---: | --- | --- | --- |
| 3 | `identification.fusion_method` | `max` | `max` 命中任一模板即可，通常 FRR 较低；`top3_mean` 更保守，通常 FAR 降、FRR 升 | `max`、`top3_mean`；`mean` 仅建议作为对照 | 对比 `max` 与 `top3_mean`，每种方式必须分别重标定阈值 |
| 4 | `enrollment.enrollment_images_per_identity` | `40` | 在 `max` 融合下，模板增多通常使 FRR 降低，但也增加 impostor 碰到高分模板的机会 | 受每指图片数限制，必须至少保留 1 张 query；建议 `10~40` | `20、30、40`。保持相同数据、随机种子并分别重标定阈值 |

`enrollment.random_seed` 不改变算法，但会改变注册/query 划分。首轮固定为 42；最终至少补测 3 个随机种子，检查结果是否稳定。

### 第 3 级：几何与纹理最终分数融合

| 排名 | 参数 | 当前值 | 增大或放宽后的常见影响 | 合理调整区间 | 首轮建议尝试 |
| ---: | --- | ---: | --- | --- | --- |
| 5 | `texture_verification.enabled` | `true` | 开启后可压低纹理不一致的 impostor，也可能拒绝低质量 genuine | `false/true` | 必做完整对照：纯几何 `false` 与当前融合 `true` |
| 6 | `texture_weight` / `geometry_weight` | `0.30/0.70` | 提高纹理占比通常使 FAR 降，但弱纹理 genuine 的 FRR 可能升高 | 权重非负且总和大于 0；归一化后的纹理占比建议 `0~0.45` | 纹理占比 `0、0.20、0.30、0.40`，几何占比取 `1-纹理占比` |
| 7 | `geometry_saturation_inliers` | `12` | 增大后，相同内点数得到的几何分数降低，通常 FAR 降、FRR 升 | 代码要求 `>0`；保守建议 `8~20` | `10、12、14、16` |
| 8 | `low_unique_inliers` | `3` | 增大后硬门槛更严，通常 FAR 降、FRR 升 | 代码要求 `>=2`；建议 `2~6` | 当前目标主要受 FRR 限制，先试 `2、3、4`，不建议直接升到 5 以上 |
| 9 | `min_overlap_fraction` | `0.20` | 增大后纹理有效区域门槛更严，通常 FAR 降、FRR 升 | `[0,1]`；保守建议 `0.10~0.35` | `0.15、0.20、0.25` |
| 10 | `min_valid_blocks` | `4` | 增大后更容易因有效纹理块不足而直接得 0 分 | `>=1`；建议 `2~8` | `3、4、6` |
| 11 | `min_block_std` | `5.0` | 增大后过滤更多低对比块，通常更严格 | `>=0`；建议 `3~8` | `4、5、6` |
| 12 | `min_block_valid_fraction` | `0.60` | 增大后要求块内有效重叠更多，通常 FAR 降、FRR 升 | `[0,1]`；建议 `0.40~0.80` | `0.50、0.60、0.70` |
| 13 | `block_size` | `16` | 小块更关注局部细节但波动较大；大块更稳定但可能掩盖局部错位 | 代码最小为 4；建议 `8~24` | `8、16、24` |
| 14 | `texture_verification.blur_sigma` | `0.8` | 增大可抑制噪声，也会损失细脊线细节，方向不一定单调 | `>=0`；建议 `0.4~1.2` | `0.6、0.8、1.0` |

现有实验中，纹理融合相对记录到的纯几何分数改变了 23 次 genuine 和 10 次 impostor 的阈值判定，但这不是一次独立的“关闭纹理”全量消融。因此 `enabled=false/true` 对照仍然需要实际运行。

### 第 4 级：描述子候选生成

L2 和 Hamming 的距离范围不同，阈值不能混用。使用二值 checkpoint 时，`matching.hamming.*` 会覆盖同名通用项。

| 排名 | 参数 | 当前值 | 增大或放宽后的常见影响 | 合理调整区间 | 首轮建议尝试 |
| ---: | --- | ---: | --- | --- | --- |
| 15 | `matching.ratio_threshold` | L2 `0.85`；Hamming `0.90` | 增大表示放宽，候选增多，通常 FAR 升、FRR 降 | L2 建议 `0.80~0.95`；Hamming 建议 `0.85~0.95` | L2：`0.82、0.85、0.88、0.90`；Hamming：`0.88、0.90、0.92`。`0.95/0.98` 仅作宽松压力测试 |
| 16 | `bidirectional_ratio_test` | `true` | `true` 更严格，通常 FAR 降、FRR 升 | `false/true` | 在 ratio 阈值固定后对比两种取值 |
| 17 | `candidate_policy` | `ratio_only` | `topk_or_ratio` 在 ratio 失败时补候选，通常 FRR 降，但可能抬高 FAR | `ratio_only`、`topk_or_ratio`、`topk_only` | 先比较 `ratio_only` 与 `topk_or_ratio`；`topk_only` 仅作压力测试 |
| 18 | `top_k` | `1` | 在 top-k 策略下增大会放宽候选；当前 `ratio_only` 下大于 1 基本无效 | 建议 `1~5` | 使用 `topk_or_ratio` 时试 `1、2、3、5` |
| 19 | `abs_distance_threshold` | L2 `1.5`；Hamming `0.40` | 增大表示放宽，通常 FAR 升、FRR 降 | 归一化 L2 合法物理范围约 `[0,2]`，建议 `1.0~1.6`；Hamming 为 `[0,1]`，建议 `0.30~0.50` | L2：`1.2、1.4、1.5`；Hamming：`0.35、0.40、0.45` |
| 20 | `distance_margin` | L2 `0.15`；Hamming `0.05` | 增大后 top-k 自适应距离上限更松 | L2 建议 `0.05~0.25`；Hamming 建议 `0.02~0.10` | L2：`0.10、0.15、0.20`；Hamming：`0.03、0.05、0.08` |
| 21 | `allow_many_to_one_before_ransac` | `false` | 开启可保留重复纹理中的真候选，也可能增加噪声 | `false/true` | 仅在关闭双向 ratio 或使用 top-k 分支时比较；当前双向 ratio 下作用很小 |

关键耦合关系：

- 当前 `candidate_policy=ratio_only` 时，`top_k>1` 和 `distance_margin` 基本不参与最终候选选择。
- `ratio_threshold`、`abs_distance_threshold` 和 `bidirectional_ratio_test` 共同限制候选，只放宽其中一项可能没有明显变化。
- 候选放宽后 genuine 和 impostor 分数都可能上升，不能只看 FRR，必须同时检查目标 FAR 下的 FRR。

### 第 5 级：RANSAC 与几何约束

| 排名 | 参数 | 当前值 | 增大或放宽后的常见影响 | 合理调整区间 | 首轮建议尝试 |
| ---: | --- | ---: | --- | --- | --- |
| 22 | `ransac_reproj_threshold` | `0.5` 像素 | 增大后更多点成为内点，通常 FAR 升、FRR 降 | 与训练数据构建的 `0.5` 对齐；建议主范围 `0.3~1.0` | `0.4、0.5、0.75、1.0`；`1.5` 仅作宽松压力测试 |
| 23 | `min_scale` / `max_scale` | `0.80/1.20` | 区间变宽会减少尺度拒绝，通常 FAR 升、FRR 降 | 建议整体不超过 `0.75~1.25` | 对比 `[0.85,1.15]`、`[0.80,1.20]`、`[0.75,1.25]` |
| 24 | `reproj_error_weight` | `0.05` | 增大后 one-to-one 去重更偏向低重投影误差候选，影响可能非单调 | 建议 `0~0.15` | `0、0.05、0.10` |
| 25 | `max_candidates_for_ransac` | `200` | 增大可保留更多候选，同时增加噪声与耗时；超限时仅按描述子距离保留最近候选 | 建议 `150~500`；`<=0` 表示不截断，不建议正式使用 | `200、300、400` |
| 26 | `ransac_max_iters` / `ransac_confidence` | `3000/0.995` | 主要影响找到稳定模型的概率和耗时，达到饱和后对指标影响较小 | 迭代 `1000~5000`；置信度 `0.99~0.999` | 首轮保持当前值，只在 RANSAC 失败率异常时调整 |

最终内点仍会输出方向一致性诊断，但方向不参与候选筛选或最终匹配分数。

### 第 6 级：关键点与 patch 质量

这一组参数需要重新构建图像模板，且与 HardNet 训练时的数据处理强耦合，不应和后端匹配参数同时大范围改动。

| 排名 | 参数 | 当前值 | 增大或放宽后的常见影响 | 合理调整区间 | 首轮建议尝试 |
| ---: | --- | ---: | --- | --- | --- |
| 28 | `sift.contrastThreshold` | `0.03` | 减小会保留更多弱纹理点，通常 FRR 降，但噪声可能使 FAR 升 | 配置已有建议 `0.02~0.04` | `0.025、0.030、0.035` |
| 29 | `sift.nfeatures` / `keypoint_filter.max_keypoints` | `500/500` | 增大会增加匹配机会和耗时，也可能增加噪声 | 当前固定推理批量为 512，建议暂时保持 `300~500` | 同步尝试 `300、400、500`；若其中一个仍为 500，另一个较大值会被较小上限限制 |
| 30 | `sift.edgeThreshold` | `17.5` | 增大可保留更多沿脊线边缘点，可能同时增加真候选和重复纹理噪声 | 仓库注释存在“当前 17.5”和“建议 10~14”的不一致；保守范围 `10~20` | `12、14、17.5` |
| 31 | `sift.nOctaveLayers` | `3` | 增大提高尺度采样密度，也会增加点数和耗时 | 建议 `2~4` | `2、3、4` |
| 32 | `sift.sigma` | `1.70` | 改变关键点检测尺度，影响方向不一定单调 | 配置已有建议 `1.5~1.7` | `1.5、1.6、1.7` |
| 33 | `patch.min_overlap_ratio` | `0.75` | 增大后丢弃更多边界关键点，通常 FAR 降、FRR 升 | `[0,1]`；建议 `0.65~0.85` | `0.70、0.75、0.80` |
| 34 | `sift.enable_clahe` / `enable_blur` | `false/false` | 可能改善低质量图，也可能使输入分布偏离训练数据 | 两个布尔开关 | 只做单独对照：全关、仅 CLAHE、仅轻微模糊；不要首轮同时开启 |

以下参数应视为模型契约，不是常规调参项：

- `patch.crop_size=32`
- `patch.out_size=32`
- `patch.normalize=true`
- `model.descriptor_kind/architecture/descriptor_dim/binary_storage/binary_bitorder`
- `matching.distance`

它们必须与 checkpoint 和训练流程一致。修改后可能显著改变 FAR/FRR，但结果通常代表输入或模型契约被破坏，而不是有效调优。

`model.checkpoint` 对指标的实际影响可能大于任何单个匹配参数，但换 checkpoint 等于换模型，必须使用新模型重新构建图像模板，并重新标定全部距离阈值和最终分数阈值。

## 4. 不影响离线 FAR / FRR，或只影响统计可信度的参数

| 参数 | 结论 |
| --- | --- |
| `runtime.max_impostor_identities_per_query` | 不改变单次匹配算法，但非 0 时只抽样 impostor，会降低 FAR 估计可信度；正式实验必须为 0 |
| `runtime.limit_identities` / `limit_images_per_identity` | 仅用于调试裁剪；正式实验必须为 0 |
| `identification.early_stop_*` | 离线与在线共用；关闭后完整遍历模板，`offline_full_template_scoring=true`；开启后主要影响延迟，阈值曲线上分数可能低于真实 identity 最大值 |
| `template_management.*` | 当前离线标定明确不执行动态模板学习，因此不影响本次离线 FAR/FRR |
| `model.batch_size`、设备、内存和推理优化项 | 主要影响速度；半精度可能造成极小数值扰动，不应作为精度调参手段 |
| `matching.hamming.backend` | CPU/CUDA 后端主要影响速度，理论匹配语义相同 |
| `evaluation.failure_export.*` | 只影响失败样本导出 |
| `output.*`、`online_unlock.*` | 不改变离线匹配分数 |

`data.image_root`、`identity_depth`、注册/query 划分会显著改变最终数值，但属于评估协议而非匹配算法。对比实验必须保持一致。

## 5. 推荐实验顺序

为避免参数耦合导致无法判断原因，建议一次只调整一个参数组，并在每组实验后重新标定最终阈值。

1. **建立 m2 基线**
   - 使用当前 `m2_256_512_v2` checkpoint 跑一次全量实验。
   - 保持 31 指、40 模板、seed 42、全量 impostor。
   - 记录目标 FAR 下 FRR、EER、最大 impostor 分数和失败阶段分布。

2. **先做纹理融合消融**
   - `enabled=false/true`
   - 纹理占比：`0.20、0.30、0.40`
   - `geometry_saturation_inliers`：`10、12、14、16`
   - `low_unique_inliers`：`2、3、4`

3. **再做候选生成消融**
   - L2 ratio：`0.82、0.85、0.88、0.90`
   - 双向 ratio：`true/false`
   - 候选策略：`ratio_only/topk_or_ratio`
   - top-k：`1、2、3`

4. **再调几何容差**
   - RANSAC 阈值：`0.4、0.5、0.75、1.0`
   - 尺度范围：`[0.85,1.15]`、`[0.80,1.20]`、`[0.75,1.25]`

5. **最后调整模板和关键点**
   - 模板数：`20、30、40`
   - 关键点上限：`300、400、500`
   - SIFT 对比度阈值：`0.025、0.030、0.035`

每个实验至少比较：

- `FRR @ FAR=2e-5`
- `FRR @ FAR=1e-4`
- EER 和 AUC
- 错误接受绝对数量
- 错误拒绝绝对数量
- genuine / impostor 的最大分数和分位数
- `candidate_ready`、RANSAC 成功率、`unique_inlier_ready`、纹理可用率
- 单次匹配耗时

优先选择“在 FAR 不恶化的情况下使 FRR 下降”的组合，而不是只追求某一个固定阈值下的 FRR。

## 6. 证据与限制

本文的当前值和实现约束来自：

- `match_new/config_match_new.yaml`
- `match_new/hardnet_matcher.py`
- `match_new/identity_matcher.py`
- `match_new/evaluation.py`
- `match_new/README.md`
- `pair_build/config.yaml`
- 现有 strongv2 完整匹配实验的 `metrics.json` 和阈值曲线

需要特别注意：

- 当前没有找到 `m2_256_512_v2` 的 identity 级 FAR/FRR 结果。
- README 提到的 `ablation.py` 当前仓库中不存在，因此 ratio、top-k、候选策略等范围大多是待验证建议，不是已有消融结论。
- 训练阶段 patch 对的 EER/AUC 与最终 identity 级 FAR/FRR 统计口径不同，不能直接互相替代。
- L2 与 Hamming 必须分别标定，不得复用距离阈值。
- 修改 SIFT、patch、模型或预处理参数后必须重新构建模板；只修改匹配、RANSAC、纹理融合和最终阈值时，可在模板格式兼容的前提下复用模板。
