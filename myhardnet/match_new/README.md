# match_new 命令说明

在项目根目录执行，例如：

```powershell
cd D:\新建文件夹\research\指纹识别\code\myhardnet
conda activate hardnet-cuda
```

默认配置文件：`match_new/config_match_new.yaml`

常调实验参数集中在 `match_new/config_match_tuning.yaml`；主配置通过
`extends` 自动继承，运行命令和 `--config` 路径无需修改。

## 前端输出接入

当前常调配置已经连接 `data/output_v8/04_denoise` 的最终图像、`05_mask` 的
指纹前景掩码和 `06_quality/fingerprint_quality_results.csv` 的质量分数。
注册/查询模板只提取前景关键点；低质量查询按 0–100 分连续调整 ratio 与
RANSAC 容差/迭代次数，最终身份接受阈值保持不变。首次接入必须重建模板。
经验初值、后续搜索范围、关闭对照及复用条件详见
[前后端联动与质量自适应](FRONTEND_INTEGRATION.md)。

---

## 1. `run_hardnet_matching.py`

**作用**：HardNet 全量离线阈值标定实验。只检测 SIFT 关键点位置/方向，计算 HardNet 描述子并匹配；支持 float32/L2 与 packed binary/Hamming 两种 checkpoint 契约，后续候选生成、RANSAC、纹理融合和 identity 级评估共用。所有未注册 query 都会遍历本人和全部非本人 identity 的全部注册模板，不再拆分 validation/test；阈值曲线和最终 FAR/FRR 均基于同一份全量匹配结果。

**运行参数**默认写在 `match_new/config_match_new.yaml`：

| 配置项 | 作用 |
|--------|------|
| `output.output_dir` | 实验结果输出目录 |
| `runtime.skip_template_build` | 是否复用已有模板 |
| `runtime.max_impostor_identities_per_query` | 每个 query 最多测几个非本人身份：`-1`=全量，`0`=不比对，正整数=抽样上限 |
| `runtime.limit_identities` / `limit_images_per_identity` | 调试裁剪，正式实验保持 `0` |
| `data.image_root` | 原始指纹根目录；固定为 `image_root/<手指目录>/图像文件` |
| `model.checkpoint` | 浮点或二值 HardNet 权重 |
| `model.descriptor_kind` / `model.binary_storage` / `model.binary_bitorder` | 描述子类型与二值模板存储契约，默认 `auto`/`packed_uint8`/`auto` |
| `model.hadamard_binarization.*` | 启用时把浮点 checkpoint 的描述子按 `Hf -> sign -> packed_uint8` 二值化；不使用 P、D 或额外 state 文件 |
| `matching.distance` | `auto` 跟随 checkpoint，也可显式写 `l2` 或 `hamming` |
| `matching.hamming.backend` | packed-Hamming 候选后端；默认 `cpu` 使用 OpenCV SIMD/POPCNT，`cuda` 仅用于显式实验 |
| `matching.hamming.*` | 二值 Hamming 的 ratio、绝对距离和自适应 margin，需按验证集重新标定 |
| `enrollment.random_seed` | 注册策略为 `random` 时控制抽样；其它用途仍可复用 |
| `enrollment.selection_strategy` | `first`（默认，取前 N 张）、`stride`（等步长取样）或 `random`（随机抽样） |
| `texture_verification.*` | 局部脊线纹理二次筛选及灰区提升参数 |
| `template_management.*` | 离线评估中的模板学习、LRU排序和固定容量替换 |

优先级：`命令行覆盖 > 配置文件 > 程序默认值`。

**推荐命令**（全部走配置）：

```powershell
python match_new\run_hardnet_matching.py
```

```powershell
python match_new\run_hardnet_matching.py `
  --image-root datasets\new_data_V3 `
  --output-dir outputs\newdataV3-8-float256+模板数30 `
```

**输出模板目录**：`<output_dir>/image_templates/`，每个 `.npz` 包含关键点字段、`hardnet_descriptors` 和 `overlap_image`。浮点模板使用 `float32` 列；二值模板使用 `packed_uint8` 列，`hardnet_descriptor_dim` 表示 bit 数，`hardnet_descriptor_bitorder` 保存位序。Hadamard 后处理还会保存 `hardnet_descriptor_transform_name` 和 `hardnet_descriptor_transform_id`，防止旧的 P/D 模板与当前直接 `Hf` 模板混用。灰度图保持原始尺寸和关键点坐标系，由 `np.savez_compressed` 无损压缩，`np.load` 时自动解压。

主要输出结构：

```text
<output_dir>/
  metadata_all.csv
  image_templates/
  identity_templates_<N>.json
  template_build_timings.csv
  eval_hardnet_<l2|hamming>/
    verification_scores.csv
    metrics.json
    effective_config.yaml
    match_score_threshold_curve.csv
    far_frr_thresholds_<min>_<max>.csv
    per_finger_far_frr_at_global_zero_far.csv
    unlock_timing.json
    plots/
      score_distribution.png
      far_frr_vs_threshold.png
    failure_cases/
```

`template_build_timings.csv` 只有一行，统计全部成功模板在读图、SIFT 检测、
关键点过滤、patch 裁切旋转、HardNet 推理、模板组装和保存等阶段的平均耗时，
不再输出逐图耗时；使用 `--skip-template-build` 时本次没有模板构建计时，因此
不生成该文件。`metrics.json` 保存全量匹配生成的 FAR/FRR、EER 和 AUC 等评估汇总。
`far_frr_vs_threshold.png` 标出 EER 及配置阈值对应的 FAR/FRR。

离线耗时保持现有输出格式：`verification_scores.csv` 的 `attempt_match_ms` 记录
每次本人/非本人匹配耗时，`metrics.json` 的 `matching_stage_summary` 汇总全部
尝试的各阶段平均耗时；`unlock_timing.json` 只汇总在配置阈值下被接受的本人记录。
这里沿用 `unlock_*` 字段名称，它们属于离线统计，不是独立解锁入口。
端到端耗时由模板构建时间加匹配时间得到，不包含模型初始化、传感器采集和模板
预加载时间，也不代表实时请求延迟。复用模板时没有本次构建计时，端到端统计
可能缺失或引用已有 metadata 中的历史构建耗时。

### CPU 推理融合与细分计时

当前主配置指定 `model.device: cpu`、`inference_precision: fp32`，并开启
`model.fuse_conv_bn: true`。权重先按原模型结构加载，再在推理实例的 `eval()`
模式下融合 Sequential 内直接相邻的 Conv2d 和 BatchNorm2d；训练模型、原始
checkpoint、GroupNorm/LayerNorm 不会被修改。`fuse_conv_bn: false` 可用于对照。
融合数学等价，但浮点计算顺序可能产生微小误差，Hadamard 系数接近零时可能
改变 bit，正式采用前仍应使用固定数据验证 FAR/FRR。

`hardnet_inference_ms` 仍表示 `describe` 完整调用时间，细分为互不重叠的：

- `hardnet_input_prepare_ms`：连续 FP32 输入数组及 CPU Tensor 准备。
- `hardnet_batch_prepare_ms`：分批、padding、锁页内存和设备传输；不含单独计时的布局转换。
- `hardnet_layout_conversion_ms`：CPU PyTorch 的 NCHW→channels_last 转换；外部后端需要的 NumPy/NCHW 连续化也计入此项。
- `hardnet_cnn_ms`：模型/backbone 前向（含输出归一化），不含 hash head 或二值化。
- `hardnet_hash_head_ms`：二值 checkpoint 的 hash head；浮点/Hadamard 路径为 0。
- `hardnet_output_transfer_ms`：输出拼接、回传 CPU、类型转换及 NumPy 转换。
- `hardnet_binarization_ms`：Hadamard + sign + packbits，或二值网络 sign + bit packing；纯浮点路径为 0。
- `hardnet_inference_overhead_ms`：结果校验、计时和包装等剩余开销；回退时含失败尝试耗时。

以上子项之和等于总时间，不要把子项与 `hardnet_inference_ms` 重复相加。
计时多 batch 时累加每个 batch；CNN 时间包含配置启用的 padding 计算。
空 patch 输入 CNN/hash/二值化耗时均为 0。

### CPU 内存布局、oneDNN 与推理后端对比

主配置 `config_match_new.yaml` 默认使用本机基准中速度最快的 **ONNX Runtime CPU**：`model.inference_backend: onnxruntime`、FP32、`CPUExecutionProvider`、图优化 `all`，主干推理 28 线程；`cpu_num_threads: 1` 用于限制 PyTorch 后处理的线程数。ONNX Runtime/OpenVINO 要求 `device: cpu`，外部后端固定使用 NCHW 输入，`channels_last` 和 `mkldnn_enabled` 不参与其主干推理。

切换回 PyTorch CPU 时，`model.channels_last` 会转换输入和模型权重，`model.mkldnn_enabled: true` 允许 PyTorch 使用 oneDNN/MKLDNN CPU 运算；布局转换耗时单独记录在 `hardnet_layout_conversion_ms`。`model.cpu_num_threads` 是 PyTorch 线程数。ONNX Runtime 的 `backend_graph_optimization` 可设为 `disabled`、`basic`、`extended` 或 `all`；`backend_num_threads` 是 ONNX Runtime 主干推理线程数，也是 OpenVINO 的推理线程数。

本机基准是在 Intel Core i7-14700、PyTorch 2.13.0+cpu 上，用真实图像提取的 394 个 32×32 patch 测得（2 次预热、8 次计时）：ONNX Runtime 28 线程中位数 48.77 ms、P95 50.21 ms；PyTorch oneDNN + `channels_last`、28 线程为 71.15 ms、P95 80.26 ms；OpenVINO 8 线程为 64.13 ms、P95 65.45 ms。初始化/编译约 150–155 ms，不计入单次 `describe()` 耗时。以上只说明该机器和样本上的测速结果；部署机器、patch 数和 batch 设置不同，最快选项可能改变。尚未完成全数据集 FAR/FRR 回归，正式使用前应在独立输出目录重建模板并重新评估。

**默认后端的配置**位于 `config_match_new.yaml` 的 `model:` 节。修改 `inference_backend` 即可切换运行时，同时按对应后端调整线程项：

```yaml
model:
  device: cpu
  inference_precision: fp32
  inference_backend: onnxruntime
  cpu_num_threads: 1
  backend_num_threads: 28
  backend_inter_op_num_threads: 1
  backend_graph_optimization: all
  channels_last: false
```

切换为 PyTorch CPU + oneDNN 时，设 `inference_backend: pytorch`、`cpu_num_threads: 28`、`mkldnn_enabled: true` 和 `channels_last: true`；`backend_num_threads` 对 PyTorch 不生效。切换为 OpenVINO 时，设 `inference_backend: openvino`、`cpu_num_threads: 1`、`backend_num_threads: 8`；`backend_num_threads` 即 OpenVINO 推理线程数。可在部署机器上用下方基准命令重新扫描线程数和 PyTorch 内存布局。

ONNX Runtime 默认依赖 `onnx` 和 `onnxruntime`，在项目使用的 Python 环境中安装：

```powershell
python -m pip install onnx onnxruntime
```

依赖缺失时程序会报安装提示，不会自动切回 PyTorch；请在实际运行脚本的 Python 环境中执行安装。直接修改主配置中已有的 `model:` 项即可，不要再添加第二个同名配置节。`config_match_tuning.yaml` 用于常调实验参数，但以上推理项已在主配置中明确指定，应直接修改 `config_match_new.yaml`。

若要选择 OpenVINO，再安装：

```powershell
python -m pip install openvino
```

后端根据当前 checkpoint 和 Conv–BN 融合状态导出/缓存 ONNX 图。浮点模型输出
保持 L2 归一化；Hadamard、`zero_is_one`、sign 和 bit packing 继续走相同的工程
后处理代码，不由后端重写。原生二值 checkpoint 由图输出 backbone 的归一化前特征，
随后仍使用原 Torch hash head 和原位序打包。ONNX 图默认缓存到系统临时目录；
也可通过 `model.onnx_path` 指定位置。外部后端要求 CPU，且始终使用 NCHW 输入；
此时 `channels_last` 与 `mkldnn_enabled` 不生效。

CPU 推理后端的线程参数仍保留在 `config_match_new.yaml`；本次用于比较 CPU 后端的临时基准脚本已在完成分析后删除。正式运行不依赖该脚本。

离线 `template_build_timings.csv` 和 `eval_hardnet_<metric>/unlock_timing.json` 分别记录模板构建阶段、本人匹配阶段的耗时。

CPU 计时使用 `perf_counter`，模型加载/融合/预热不计入单次推理；CUDA
细分计时在阶段边界同步设备，避免把异步 CNN 时间误算到回传阶段，但会增加
同步开销，不能与未同步的 GPU 吞吐测试直接比较。普通 `describe()` 不添加
这些阶段同步。实际设备/精度写入结果，CUDA 失败回退时记录最终 CPU 设备。

**常用命令行覆盖**（均可不传，改用配置）：

| 参数 | 作用 |
|------|------|
| `--config` | 配置文件路径，默认 `match_new/config_match_new.yaml` |
| `--image-root` | 覆盖 `data.image_root` |
| `--image-root` | 覆盖 `data.image_root` |
| `--model-path` / `--model_path` | 覆盖 `model.checkpoint` |
| `--output-dir` / `--output_dir` | 覆盖 `output.output_dir` |
| `--hadamard-binarization` / `--no-hadamard-binarization` | 开启/关闭 `model.hadamard_binarization.enabled` |
| `--ratio-threshold` | 覆盖浮点/L2 的 `matching.ratio_threshold` |
| `--hamming-ratio-threshold` | 覆盖二值/Hamming 的 `matching.hamming.ratio_threshold` |
| `--skip-template-build` / `--no-skip-template-build` | 覆盖是否跳过模板构建 |
| `--max_impostor_identities_per_query` | 覆盖 impostor 上限 |
| `--random_seed` | 覆盖注册随机种子 |
| `--failure-export` / `--no-failure-export` | 覆盖是否导出失败样本 |
| `--max_failure_cases_per_type` | 覆盖每类失败样本导出上限 |
| `--limit_identities` | 调试：限制 identity 数量 |
| `--limit_images_per_identity` | 调试：限制每个 identity 的图像数量 |

浮点/L2 实验示例：

```powershell
python match_new\run_hardnet_matching.py `
  --image-root datasets\wet_select `
  --model-path outputs\models\3\m0\m0_256_v2\best.pt `
  --output-dir outputs\wet_select_float `
  --no-hadamard-binarization `
  --ratio-threshold 0.85
```

Hadamard 二值/Hamming 实验示例：

```powershell
python match_new\run_hardnet_matching.py `
  --image-root datasets\wet_select `
  --model-path outputs\models\3\m0\m0_256_v2\best.pt `
  --output-dir outputs\wet_select_hadamard `
  --hadamard-binarization `
  --hamming-ratio-threshold 0.90
```

`matching.hamming.*` 会覆盖同名顶层参数，所以二值/Hamming 实验应使用
`--hamming-ratio-threshold`。切换 checkpoint 或 Hadamard 开关会改变描述子契约，
必须重新构建模板，不能同时使用 `--skip-template-build`；只修改 ratio 阈值时可以复用模板。

**离线全量匹配与阈值评估**：

每个 query 都会对本人 identity 和配置范围内的非本人 identity 遍历全部注册模板，使用完整的 identity 分数计算 FAR/FRR、阈值曲线、EER 和 AUC。正式 FAR 评估应将 `runtime.max_impostor_identities_per_query` 设为 `-1`。`identification.match_score_threshold` 只用于选择并报告工作点，不会截断模板遍历。

`verification_scores.csv` 中，`num_templates` 表示该 identity 注册模板总数；`metrics.json` 中 `scoring_mode` 为 `full_query_pool_evaluation`，表示使用全量 query 分数评估。

## 2. 离线评估参数参考

`identification` 配置用于融合每个 identity 的模板分数，并选择 FAR/FRR 工作点：

| 配置项 | 作用 |
|--------|------|
| `fusion_method` | `max`、`mean`、`top3_mean` 或 `max_quality_tiebreak` |
| `match_score_threshold` | `[0,1]` 连续分数阈值，用于报告选定工作点的 FAR/FRR |


### 模型与特征参数（`model.*` / `sift.*` / `patch.*`）

离线模板构建和评估使用相同的 HardNet + SIFT 特征提取流水线。

| 配置段 | 主要项 | 程序缺省值 | 说明 |
|--------|--------|------------|------|
| `model.checkpoint` | — | 无 | HardNet 权重路径；必须与注册模板描述子类型一致 |
| `model.descriptor_kind` | `auto` | 跟随 checkpoint：`float` 或 `binary` |
| `model.binary_storage` / `binary_bitorder` | `auto` | 二值 checkpoint 的 packed 存储契约 |
| `model.device` | `auto` | 优先 CUDA，失败可回退 CPU（`fallback_to_cpu: true`） |
| `model.batch_size` | `512` | HardNet patch 批量推理大小 |
| `model.fixed_inference_batch_size` | `0` | `0` | 固定推理 batch 大小；设为正数时补齐最后一个 batch |
| `model.inference_precision` | `fp32` | `fp32` | CNN 前向精度；描述子仍返回 FP32 / packed uint8 |
| `sift.nfeatures` | OpenCV 默认 | 当前配置常用 `500` |
| `sift.nOctaveLayers` | `3` | 尺度层数 |
| `sift.contrastThreshold` | `0.04`（OpenCV） | 当前配置常用 `0.03` |
| `sift.edgeThreshold` | `10`（OpenCV） | 当前配置常用 `17.5` |
| `sift.sigma` | `1.6`（OpenCV） | 当前配置常用 `1.70` |
| `sift.enable_clahe` / `enable_blur` | `false` | 提特征前预处理 |
| `keypoint_filter.max_keypoints` | 不截断 | 当前配置常用 `500` |
| `patch.crop_size` / `out_size` | — | 必须与训练一致，当前为 `32` |
| `patch.normalize` | — | 单 patch 减均值除标准差，当前为 `true` |
| `patch.min_overlap_ratio` | — | 边界关键点丢弃阈值，当前常用 `0.75` |

### 匹配参数（`matching.*`）

| 配置项 | 程序缺省值 | `config_match_new.yaml` 常见值 | 作用 |
|--------|------------|-------------------------------|------|
| `distance` | `auto` | `auto` | `auto` 跟随 checkpoint：`l2` 或 `hamming` |
| `candidate_policy` | `topk_or_ratio` | `ratio_only` | 候选生成策略 |
| `top_k` | `5` | `1` | 每个 query 点保留的近邻数（`ratio_only` 下多为 1） |
| `ratio_threshold` | `0.95` | L2 `0.85`；Hamming 见 `hamming.ratio_threshold` | Lowe ratio 阈值 |
| `bidirectional_ratio_test` | `false` | 按实验配置 | 双向最近邻 + 双向 ratio |
| `abs_distance_threshold` | `1.10` | L2 `1.5`；Hamming 见 `hamming.abs_distance_threshold` | 绝对距离上限 |
| `distance_margin` | `0.12` | L2 `0.15` | top-k 自适应距离上限（ratio 策略下影响小） |
| `allow_many_to_one_before_ransac` | `true` | 按实验配置 | RANSAC 前是否保留一对多 |
| `max_candidates_for_ransac` | `250` | `200` | 进入 RANSAC 前的候选上限；超限时仅按描述子距离保留最近候选 |
| `ransac_reproj_threshold` | `5.0` | `0.5` | RANSAC 重投影阈值（像素） |
| `ransac_max_iters` | `3000` | `3000` | RANSAC 最大迭代 |
| `ransac_confidence` | `0.995` | `0.995` | RANSAC 置信度 |
| `min_scale` / `max_scale` | `0.0` / `∞` | `0.80` / `1.20` | partial affine 尺度范围 |
| `reproj_error_weight` | `0.05` | `0.05` | one-to-one 去重时的重投影误差权重 |
| `hamming.backend` | `cpu` | `cpu` | packed-Hamming 后端：`cpu` / `cuda` |
| `hamming.ratio_threshold` | 继承顶层 | `0.85` | 二值 ratio 阈值 |
| `hamming.abs_distance_threshold` | 继承顶层 | `0.50` | 二值绝对距离上限 |
| `hamming.distance_margin` | 继承顶层 | `0.05` | 二值自适应 margin |

Hamming 路径下，`matching.hamming.*` 会覆盖同名顶层 L2 参数。

### 纹理融合参数（`texture_verification.*`）

| 配置项 | 程序缺省值 | 常见值 | 作用 |
|--------|------------|--------|------|
| `enabled` | `false` | `true` | 是否启用几何 + 脊线纹理融合 |
| `low_unique_inliers` | `4` | `3` | 低于该 unique 内点数直接 score=0 |
| `geometry_saturation_inliers` | `12.0` | `12.0` | 几何分数饱和内点数 |
| `geometry_weight` / `texture_weight` | `0.70` / `0.30` | 同上 | 融合权重（自动归一化） |
| `block_size` | `16` | `16` | ZNCC 分块大小 |
| `blur_sigma` | `0.8` | `0.8` | 纹理计算前高斯模糊 |
| `min_block_std` | `5.0` | `5.0` | 有效块最低对比度 |
| `min_block_valid_fraction` | `0.60` | `0.60` | 块内有效重叠比例 |
| `min_valid_blocks` | `4` | `4` | 至少多少有效块才出纹理分 |
| `min_overlap_fraction` | `0.20` | `0.20` | warp 后有效重叠面积比例 |

启用纹理融合时，注册模板必须含 `overlap_image` 字段。

离线评估输出的 `template_build_timings.csv` 汇总全量模板构建阶段耗时；各次 query 的匹配耗时、本人/非本人分数和 FAR/FRR 指标分别写入评估结果目录。

**换 checkpoint 示例**：

```powershell
python match_new\run_hardnet_matching.py `
  --model-path outputs\hardnet_train_xxx\best.pt `
  --output-dir match_new\outputs_xxx
```

**复用已有模板**（只改了匹配参数、未改模型/patch/SIFT 检测参数和纹理模板格式时）：

```powershell
python match_new\run_hardnet_matching.py `
  --output-dir match_new\outputs `
  --skip-template-build
```

注意：旧版双描述子模板（同时含 HardNet 和 RootSIFT）**不再支持复用**，需要重新生成 HardNet 单描述子模板。

启用 `texture_verification.enabled: true` 后，模板还必须包含 `overlap_image`。旧 HardNet 模板没有该字段，不能使用 `--skip-template-build`，需要重新生成一次模板。

### 几何与脊线纹理融合（C 方案）

纹理分数依赖 RANSAC 得到的 query→template 仿射矩阵，因此仍保留一个仅用于保证仿射可靠性的低内点门槛。达到门槛后，不再用纹理做硬通过/拒绝，而是计算连续融合分数：

```math
geometry\_similarity = clip(unique\_inliers / 12, 0, 1)
```

```math
score = 0.70 \times geometry\_similarity + 0.30 \times texture\_similarity
```

```text
unique_inliers < low_unique_inliers
    -> score = 0，不计算纹理

unique_inliers >= low_unique_inliers
    -> 计算 geometry_similarity 和 texture_similarity
    -> 加权得到 [0,1] match score
```

纹理分数使用分块 ZNCC：先将 query 灰度图仿射变换到模板坐标系，只在有效重叠区域计算；低对比块被排除，最终取有效块正相关系数的中位数。相关配置位于：

```yaml
texture_verification:
  enabled: true
  low_unique_inliers: 3
  geometry_saturation_inliers: 12.0
  geometry_weight: 0.70
  texture_weight: 0.30

identification:
  match_score_threshold: 0.55
```

`unique_inliers` 继续作为诊断字段输出，但不再直接充当 `score` 或匹配阈值。离线评估会直接使用全部 query 的 identity 级分数扫描 FAR/FRR，并在 `identification.match_score_threshold` 下报告最终结果。`verification_scores.csv` 会额外输出 `geometry_similarity`、`texture_similarity`、有效重叠比例、有效块数和实际融合权重。

FAR/FRR 阈值曲线写入 `match_score_threshold_curve.csv`。默认按 `0.01` 在 `[0,1]` 范围扫描，不再生成整数内点阈值曲线。

固定阈值区间的 FAR/FRR 对照表由 `evaluation.far_frr_table` 控制，默认
`0.50--0.85`（步长 `0.01`），写入 `far_frr_thresholds_0.50_0.85.csv`。此外，
`per_finger_far_frr_at_global_zero_far.csv` 使用所有手指共同的
“全局 FAR=0 且 FRR 最小”阈值：第一行是全体统计，后续各行按
`owner_identity` 汇总单个注册手指的 FAR/FRR 及接受、拒绝计数。
只有一个 identity、没有 impostor 尝试时，表中会将选择状态标记为
`unavailable_no_impostor_attempts`，此时 FAR 不可估计。

`metrics.json` 的 `texture_verification_config` 会记录实际融合参数，`score_component_summary` 会分别汇总 genuine/impostor 中纹理参与次数，以及相对纯几何分数改变了多少次阈值判定。`matching_backend` 名称带有 `texture_fusion` 时，表示本次结果确实启用了 C 方案。

### 离线动态模板学习、替换与 LRU 排序

`run_hardnet_matching.py` 可以在离线评估过程中启用模板学习。每次启动都会先重新构建 `identity_templates_40.json`，因此默认从本次注册模板重新开始，不继承上一次运行的学习结果。

动态模板学习复用以下模块：

| 模块 | 职责 |
|------|------|
| `template_learning.py` | 计算仿射对齐后的公共纹理区域率，并判断学习准入 |
| `template_ranking.py` | 成功命中的模板移到首位，新模板插入首位 |
| `template_replacement.py` | 模板库满载时从末尾选择非保护模板 |
| `template_library.py` | 更新活动模板列表并原子写回 `identity_templates_40.json` |

默认配置：

```yaml
template_management:
  enabled: true

  # 活动模板库最大数量；初始受保护模板数量由 enrollment.enrollment_images_per_identity 决定。
  max_active_templates: 40

  # 本人匹配分数必须达到该值。
  learn_score_threshold: 0.85
  # 仿射对齐后的公共纹理区域率必须不高于该值，可按实验调节。
  max_common_area_ratio: 0.50

  coverage_window_size: 9
  coverage_min_std: 5.0
  coverage_morph_kernel: 3
```

每个 query 的处理顺序固定为：

```text
1. 使用当前活动模板库完成本人匹配
2. 本人分数达到 match_score_threshold 后，检查模板学习条件
3. 满足 learn_score_threshold 且公共纹理区域率 <= max_common_area_ratio 时加入模板库
4. 学习结果写回 identity_templates_40.json，仅影响后续 query
5. 使用更新后的模板库执行非本人匹配
```

学习不要求多个模板确认，也不要求初始模板确认；仍会拒绝完全重复的模板内容。初始注册模板数量由 `enrollment_images_per_identity` 决定，并全部受保护。`max_active_templates` 是活动模板库上限；当前两者都为 40 时，模板库已经满载，不能再加入学习模板。如需在保留 40 张初始模板的同时学习新模板，应将 `max_active_templates` 设置为大于 40 的值，例如 60。模板库满载后，才会从未受保护的学习模板中按 LRU 规则替换。写回 JSON 的重试次数和等待时间固定在代码内部，分别为 2 次和 1 毫秒，不放在 YAML 配置中。

主要输出仍保留原有评估文件，模板库只有一个动态索引：

| 输出 | 含义 |
|------|------|
| `identity_templates_40.json` | 当前活动模板路径、模板来源、保护状态、LRU顺序和更新计数 |
| `eval_hardnet_<metric>/metrics.json` | FAR、FRR及模板学习统计 |
| `eval_hardnet_<metric>/verification_scores.csv` | 每次本人/非本人验证分数 |

`metrics.json` 中的 `template_management` 会记录处理 query 数、通过本人匹配数、学习模板数和替换模板数。模板学习更新发生在当前 query 的本人结果记录之后，因此新模板不会改变当前 query 已经写入的本人分数；它只参与后续 query。

---

## 3. `ablation.py`

**作用**：在已有模板基础上，扫描不同匹配参数组合（`top_k`、`ratio_threshold`、`candidate_policy`），快速比较调参效果。不重新提特征。

**前置条件**：先运行 `run_hardnet_matching.py` 生成 `image_templates/`、`metadata_all.csv`、`identity_templates_20.json`；query 划分由注册模板索引在内存中推导。

**命令**：

```powershell
python match_new\ablation.py `
  --base_output_dir match_new\outputs `
  --output_dir match_new\outputs_ablation `
  --topk_values 1 2 5 10 `
  --ratio_thresholds 0.85 0.90 0.95 0.98
```

**常用参数**：

| 参数 | 作用 |
|------|------|
| `--base_output_dir` | 已有主实验输出目录 |
| `--output_dir` | 消融结果输出目录 |
| `--topk_values` | 要扫描的 top-k 列表 |
| `--ratio_thresholds` | 要扫描的 ratio 阈值列表 |
| `--candidate_policies` | 候选策略：`ratio_only` / `topk_only` / `topk_or_ratio` |
| `--max_impostor_identities_per_query` | 每个 query 最多测试几个非本人 identity |

---

## 推荐执行顺序

**HardNet 验证**：

```text
run_hardnet_matching.py
```

**只调匹配参数**：

```text
run_hardnet_matching.py          # 首次构建模板
  -> ablation.py                 # 复用模板扫参数
```

---

## 4. `visualize_hardnet_inliers.py`

**作用**：单独查看 HardNet 的内点匹配可视化。脚本从一个原始图像目录扫描指纹图像，构建 HardNet 模板，然后按 `run_hardnet_matching.py` / FAR-FRR 评估中相同的 genuine 匹配逻辑，为每个 query 匹配本人注册模板库中的最佳模板，最后导出：

- `unique_inliers` 最高的 top N，默认 10
- `unique_inliers` 最低且不低于阈值的 bottom N，默认 10，最低阈值默认 3

匹配参数与 `config_match_new.yaml` 共用，脚本内部复用 `match_templates_descriptor_l2(..., descriptor_source="hardnet", include_debug=True)`，因此 Lowe ratio、top-k、RANSAC、unique inlier 去重等逻辑和正式 HardNet 评估一致。

双向 Lowe 验证由以下配置控制：

```yaml
matching:
  ratio_threshold: 0.85
  bidirectional_ratio_test: true
```

启用后，候选必须在 `query -> gallery` 和 `gallery -> query` 两个方向都通过同一个 Lowe 比值阈值，并且双方互为最近邻。关闭后恢复原来的单向比值验证。

**推荐命令**：

```powershell
python match_new\visualize_hardnet_inliers.py `
  --image_root pair_build\select_top500 `
  --output_dir match_new\hardnet_inlier_visuals
```

**复用已构建模板**：

```powershell
python match_new\visualize_hardnet_inliers.py `
  --image_root pair_build\select_top500 `
  --output_dir match_new\hardnet_inlier_visuals `
  --skip_template_build
```

**如果每个 identity 注册 20 张导致 query 太少，可以调小注册数**：

```powershell
python match_new\visualize_hardnet_inliers.py `
  --image_root pair_build\select_top500 `
  --output_dir match_new\hardnet_inlier_visuals_enroll10 `
  --enrollment_count 10
```

**常用参数**：

| 参数 | 作用 |
|------|------|
| `--config` | 配置文件路径，默认 `match_new/config_match_new.yaml` |
| `--image_root` | 输入原始图像目录，默认读取配置中的 `data.image_root` |
| `--output_dir` | 可视化输出目录，默认 `match_new/hardnet_inlier_visuals` |
| `--model_path` | 覆盖配置中的 HardNet checkpoint |
| `--skip_template_build` | 按当前原图 metadata 复用 `<output_dir>/image_templates` 中已存在的模板 |
| `--enrollment_count` | 覆盖每个 identity 的注册模板数量，默认读取 config |
| `--random_seed` | 覆盖注册模板随机种子，默认读取 config |
| `--top_k` | 导出内点数最高的 case 数量，默认 10 |
| `--bottom_k` | 导出内点数最低的 case 数量，默认 10 |
| `--bottom_min_unique_inliers` | bottom case 的最低 unique 内点数阈值，默认 3 |
| `--max_lines` | 每张连线图最多画多少条线，默认 200 |
| `--include_impostor` | 额外计算 query 对非本人 identity 的匹配，默认关闭 |

**主要输出**：

```text
match_new/hardnet_inlier_visuals/
  image_templates/
  metadata_all.csv
  identity_templates_20.json
  top_unique_inliers/
  bottom_unique_inliers_nonzero/
  selected_cases.csv
  summary.json
```

每个 case 目录包含：

| 文件 | 含义 |
|------|------|
| `match_lines_unique.png` | HardNet unique inliers 左右连线图 |
| `match_lines_raw.png` | RANSAC raw inliers 左右连线图 |
| `query_warped_to_gallery.png` | 按 affine matrix 把 query warp 到 gallery 坐标系后的图 |
| `overlap_color.png` | warp 后 query 和 gallery 的彩色重叠图 |
| `overlap_and.png` | 两张图二值化后重叠区域的 AND |
| `overlap_xor.png` | 两张图二值化后差异区域的 XOR |
| `match_result.json` | 匹配指标、affine matrix、输出路径和 debug 计数 |
