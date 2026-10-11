# 前端掩码与查询质量自适应匹配

本次将前端 `data/output_v8` 的输出接入 `myhardnet/match_new`。前端质量分数是
FFQ-Lite 的 **0–100 图像质量分数**，后端最终 `score` 是 **0–1 身份匹配分数**，
两者用途不同。匹配器原有的 `quality_score` / `best_quality_score` 是内点等信息
组成的匹配诊断分数，也不是前端质量分数。

当前策略是经验初值，尚未完成全量 FAR/FRR 标定。FAR 表示非本人被误接受的
比例，FRR 表示本人被误拒绝的比例；放宽候选条件仍可能提高 FAR，需要用本人
和非本人数据同时验证。

## 1. 数据如何关联

配置位于 `config_match_tuning.yaml`，由 `config_match_new.yaml` 自动继承。
所有 YAML 相对路径以 `myhardnet/match_new` 为基准，与运行命令的工作目录无关。

```yaml
data:
  image_root: ../../data/output_v8/04_denoise

frontend:
  mask:
    enabled: true
    root: ../../data/output_v8/05_mask
    binarization_threshold: 128
    missing_policy: error
  quality:
    enabled: true
    csv_path: ../../data/output_v8/06_quality/fingerprint_quality_results.csv
    missing_policy: error
```

当前前端运行包含 V3 UNet 去噪，因此后端读取 `04_denoise` 的最终图像，与
生成 `05_mask` 和 `06_quality` 时的图像一致。例如：

- 图像：`04_denoise/crp_L0/pair_1.bmp`。
- 掩码：`05_mask/crp_L0/pair_1.bmp`。
- 质量 CSV：`relative_path=crp_L0\pair_1.bmp` 对应的 `quality_score`。

关联键是相对图像根目录的完整路径 `手指目录/文件名`，同时接受 `/` 和 `\`。
不同手指可以有同名的 `pair_1.bmp`，不会混用质量记录。CSV 一次加载并建立索引，
不会在每次 query/gallery 匹配时重新读取文件。重复的相对路径直接报错。

若以后前端使用 `--skip-denoise`，将 `data.image_root` 改为同一次前端运行中的
`03_diffraction_v8`，并使用该次生成的掩码和质量记录。更换数据集时要一起修改
图像根目录、掩码根目录和质量 CSV；同名文件不代表来自同一次采集或处理。

## 2. 掩码在特征提取中的作用

注册图像与查询图像使用同一规则：

1. 读取与图像同相对路径的灰度掩码。
2. 像素值 `>= binarization_threshold` 的区域为指纹前景，其余为背景。
3. 将二值掩码直接传给 `SIFT.detect`，限制关键点中心的位置。
4. 对检测返回的关键点再次检查中心最近像素，仅保留前景内、坐标有效的点。
5. 在前景过滤之后，执行响应强度排序、`max_keypoints` 截断及 patch 边界检查。
6. 只对保留点裁剪局部块、计算 HardNet 描述子，后续匹配不再使用背景点。

图像和掩码尺寸必须一致。尺寸不符时报错，避免自动缩放造成坐标错配。空掩码
会产生零关键点、零描述子，匹配分数为 0；不会自动切回全图检测。

这里过滤的是 **关键点中心**，没有要求整个 32×32 局部块都在前景中，也没有
将背景涂黑后再检测，以免掩码边缘制造人为关键点。原有 patch 的图像边界重叠
检查仍保留。当前前端掩码暂不参与纹理相关分数或模板学习的公共区域计算。

`missing_policy: error` 是默认值：缺少掩码或有效质量记录时明确报错。
如果明确接受部分缺失，可设为 `warn`：缺少掩码的该图警告后用全图检测；缺少、
失败、NaN、无穷或超出 0–100 的质量分数警告后用基础匹配参数，记为 `unknown`，
不会误当作质量 0 分。整个掩码目录或质量文件不存在仍然报错。

## 3. 当前质量自适应规则

自适应只读取 **查询图像** 的前端质量，与被比较的注册模板质量、身份和是否
本人无关。同一个 query 与所有本人、非本人模板使用相同参数，不能依据匹配
结果或真实身份再选择宽松参数。

```yaml
quality_adaptation:
  enabled: true
  normal_quality_min: 60.0
  low_quality_below: 40.0
  quality_floor: 20.0
  ratio_threshold_increment: 0.04
  ratio_threshold_cap: 0.97
  ransac_reproj_multiplier_max: 2.0
  ransac_max_iters_increment: 2000
```

质量分档用于结果日志：

- `normal`：`Q >= 60`，使用基础参数。
- `medium`：`40 <= Q < 60`，适度放宽。
- `low`：`0 <= Q < 40`，继续放宽，但 `Q <= 20` 不再增加。
- `unknown`：无有效分数，使用基础参数。

真正的参数变化是连续线性插值，不会在 40 分处突然跳变：

```text
s = clip((normal_quality_min - Q) / (normal_quality_min - quality_floor), 0, 1)

ratio(Q) = max(base_ratio,
               min(ratio_threshold_cap,
                   base_ratio + ratio_threshold_increment * s))

ransac_reproj_threshold(Q)
    = base_reproj_threshold * (1 + (ransac_reproj_multiplier_max - 1) * s)

ransac_max_iters(Q)
    = base_max_iters + round(ransac_max_iters_increment * s)
```

关闭自适应或无有效分数时，`s=0`。`low_quality_below` 只改变日志分档，不改变
上述插值公式。程序要求 `0 <= quality_floor < low_quality_below < normal_quality_min <= 100`。
基础 ratio 如果已经高于 cap，则保持原值，不因为“放宽”反而收紧。

按当前基础参数，典型工作点为：

- `Q >= 60`：L2 ratio `0.85`，Hamming ratio `0.90`；RANSAC 重投影阈值
  `0.50` 像素，最大迭代 `3000`。
- `Q = 50`：L2 ratio `0.86`，Hamming ratio `0.91`；重投影阈值 `0.625` 像素，
  最大迭代 `3500`。
- `Q = 40`：L2 ratio `0.87`，Hamming ratio `0.92`；重投影阈值 `0.75` 像素，
  最大迭代 `4000`。
- `Q = 30`：L2 ratio `0.88`，Hamming ratio `0.93`；重投影阈值 `0.875` 像素，
  最大迭代 `4500`。
- `Q <= 20`：L2 ratio `0.89`，Hamming ratio `0.94`；重投影阈值 `1.00` 像素，
  最大迭代 `5000`。

Lowe ratio 是最近邻距离与次近邻距离的比例测试，阈值略大可以保留更多较弱的
真实对应点。RANSAC 是从候选点中估计几何变换并排除不一致匹配的算法；适当
提高重投影容差，可容纳低质量图像的关键点定位误差，提高迭代次数则增加寻找
正确变换的机会，但不保证一定运行到最大次数。

当前选择在 100×110 原图上最多放宽到 1 像素，而不是直接使用 3–5 像素的大
容差。更换图像分辨率后，需要重新确认这个以原图像素为单位的范围。

### 保持不变的参数

- 最终身份接受阈值 `identification.match_score_threshold=0.80`。
- L2 绝对距离上限 `1.50`、归一化 Hamming 距离上限 `0.50`。
- 候选策略 `ratio_only`、每点近邻数及 RANSAC 候选数量上限。
- 尺度范围 `0.80–1.20` 与 RANSAC 后一对一内点去重。
- 最少唯一内点 `3`、几何分数饱和内点数 `12`。
- 几何/纹理权重 `0.70/0.30`、纹理区域和有效块门槛。
- SIFT 检测参数、注册图像选择策略以及模板学习准入阈值。

初期只放宽“能否形成候选和通过几何验证”，避免同时降低最终接受标准或使
少量内点自动得到很高分。保持这些门槛不变仍不保证 FAR 不变，需要重评估。

## 4. 怎么运行和检查结果

在已安装模型依赖的环境中，从项目根目录执行：

```powershell
conda activate hardnet
python myhardnet\match_new\run_hardnet_matching.py
```

当前默认模板数 40，输出到新的 `myhardnet/outputs/frontend_v8/` 下，避免混用旧
数据集模板。第一次接入掩码必须重建模板，保持 `runtime.skip_template_build: false`。

先做两名手指的小规模流程验证：

```powershell
python myhardnet\match_new\run_hardnet_matching.py `
  --limit_identities 2 `
  --limit_images_per_identity 42 `
  --output-dir myhardnet\outputs\frontend_v8\smoke `
  --no-failure-export
```

命令行可覆盖前端路径或开关，例如：

```powershell
python myhardnet\match_new\run_hardnet_matching.py `
  --image-root data\output_v8\04_denoise `
  --mask-root data\output_v8\05_mask `
  --quality-csv data\output_v8\06_quality\fingerprint_quality_results.csv
```

做“只用掩码、不做质量自适应”的对照：

```powershell
python myhardnet\match_new\run_hardnet_matching.py `
  --no-quality-adaptation `
  --output-dir myhardnet\outputs\frontend_v8\mask_only
```

使用没有前端配套输出的旧数据集时，显式关闭两项读取：

```powershell
python myhardnet\match_new\run_hardnet_matching.py `
  --image-root myhardnet\datasets\normal_pic `
  --no-frontend-mask `
  --no-frontend-quality `
  --no-quality-adaptation `
  --output-dir myhardnet\outputs\normal_without_frontend
```

检查以下输出：

- `metadata_all.csv`：`frontend_relative_path`、`frontend_mask_path`、
  `frontend_quality_score`、`frontend_quality_csv`。
- 图像模板 `.npz`：上述来源信息、`frontend_mask_applied`、
  `frontend_mask_signature`。签名保存实际二值掩码内容，用于模板复用检查。
- `template_build_timings.csv`：新增 `avg_frontend_mask_load_ms`，记录模板构建
  中的掩码读取、二值化和校验时间；目录扫描/质量索引建立不计入逐图推理时间。
- `eval_hardnet_<metric>/verification_scores.csv`：每次本人/非本人尝试的
  `query_frontend_quality_score`、`query_quality_band`、`quality_adaptation_applied`、
  `quality_relaxation_strength`、`effective_ratio_threshold`、
  `effective_abs_distance_threshold`、`effective_ransac_reproj_threshold`、
  `effective_ransac_max_iters` 和 `query_frontend_mask_applied`。
- `effective_config.yaml` 和 `metrics.json` 的 `effective_config`：保存基础配置、
  前端绝对路径及自适应规则。这里保留基础参数，每个 query 的实际参数以
  `verification_scores.csv` 为准。

### 模板复用

只修改质量分档阈值、自适应增量、ratio、RANSAC 或质量 CSV 的分数，且原图、
掩码、模型、patch 和 SIFT 均不变时，可以使用 `--skip-template-build`。
评估会用本次 CSV 的分数覆盖模板保存的历史质量，不必重新运行 HardNet。

开启/关闭掩码过滤、更换输入图像目录、掩码内容变化或二值化结果变化后，需要
重建模板。主入口会对全部复用模板检查掩码状态与内容签名，避免旧的背景描述子
继续参与匹配。旧版未保存前景过滤状态的模板不能用于开启掩码后的运行。

模板来源目录检查不能识别“原图在相同路径被覆盖但掩码保持不变”的情况；同路径
更新原图时仍应主动重建模板。图像模板磁盘中的质量分数只记录构建时状态，
本次评估实际使用的分数见结果 CSV。

## 5. 后续怎么调整

优先只修改 `config_match_tuning.yaml` 的 `quality_adaptation`，不用改 Python。
所有参数相对当前基础匹配参数计算，每次 query 生成独立副本，不会将前一张
低质量图像的宽松参数留给后一张图像。

建议初期搜索范围：

- `normal_quality_min`：55–65；改变从多少分开始放宽。
- `quality_floor`：15–30；改变从多少分开始停止继续放宽。
- `low_quality_below`：35–45；只改变 medium/low 统计分档。
- `ratio_threshold_increment`：0.02–0.06；上限保持 0.95–0.97 左右。
- `ransac_reproj_multiplier_max`：1.5–2.5；当前基础 0.5 像素对应最终
  0.75–1.25 像素，先避免大幅增加。
- `ransac_max_iters_increment`：1000–3000；按精度和耗时一起选择。

调参顺序：

1. 固定相同图像、注册划分、模型和掩码，先跑“仅掩码”和“掩码+当前自适应”
   两份结果，分别统计全体和各质量段的 FAR/FRR。
2. 在固定质量阈值下先调 ratio 增量，再调几何容差，最后调迭代上限；每次只
   改一项，检查低质量本人是否得到更多正确唯一内点。
3. 如果低质量 FRR 有下降但 FAR 上升，先收紧 ratio 增量或 RANSAC 容差；
   保持身份接受阈值和最少内点门槛，不同时放宽全部条件。
4. 如果主要损失发生在 RANSAC 前，检查 `num_ratio_pass`、`num_candidates`；
   如果主要损失发生在几何/纹理验证，检查 `unique_inliers`、`texture_decision`、
   `texture_valid_blocks`，避免盲目提高迭代次数。
5. 定型时使用独立验证数据或留出采集批次，再验证未参与调参的数据。
   全量运行时恢复 `limit_identities=0`、`limit_images_per_identity=0`、
   `max_impostor_identities_per_query=-1`，不能把小规模流程测试当作精度结论。

极低质量、几乎没有前景或持续无法形成可靠几何证据的查询，更适合重新采集，
而不是无限提高 ratio 或降低接受阈值。当前代码对质量 Q<=20 仅采用有界放宽，
尚未加入硬性质量拒绝或自动重采集。

### 测试结果

本次集成测试已在完成验证后删除；这里保留功能说明和验证结论。仓库中原有
`test_path_names.py`、`test_run_hardnet_matching.py` 等测试文件未删除。

2026-10-10 在本机 `hardnet` 环境完成以下验证：

- 当前 36 个手指、3600 张图像均关联到同相对路径的掩码和有效质量分数。
- 两个手指、84 张图像用实际 checkpoint 和 ONNX Runtime 完成建模与匹配，
  共 4 个查询、8 次本人/非本人身份尝试；重用模板的流程也通过校验。
  结果位于 `myhardnet/outputs/frontend_v8/frontend_integration_smoke_20261010/`。
- 实际低质量查询 `dhy_L3/pair_95.bmp` 的 Q=20.9627，应用 Hamming ratio
  0.9390373、重投影阈值 0.98796625 像素、最大迭代 4952，符合连续插值规则。
- 本次前端接入、身份匹配和推理回归的 53 项相关测试通过；测试脚本已按约定在验证后删除。

以上属于关联和功能验证，尚未完成全量精度对照，不能据此推断 FAR/FRR 有所改善。
RANSAC 上限的全量统计与敏感性实验见 [RANSAC 迭代次数分析](RANSAC_ITERATION_ANALYSIS.md)。
仓库原有 `test_path_names.py` 仍因 `utils.py` 缺少 `compact_path_token` 等函数
发生测试收集错误；该历史问题没有在本次修改中处理，53 项通过数不包含它。

验证命令记录如下（对应的临时测试脚本已删除）：

覆盖 Windows/POSIX 相对路径、多身份同名文件、失败/缺失质量、掩码尺寸和
空掩码、过滤在关键点限额之前执行、模板持久化及复用校验、连续阈值边界、
L2/Hamming 两种实际匹配参数、查询间配置隔离，以及本人/非本人结果 CSV。
