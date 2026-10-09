# 超声指纹前端处理工具包

本目录可独立复制给他人使用，覆盖原始 CSV 的配对/差分、三种衍射校正、
V3 UNet 去噪、指纹前景分割、质量打分，以及一个可选的研究型纹线生长模块。
不包含 HardNet、特征提取、模板生成或匹配代码。

## 目录

```text
fingerprint_frontend_toolkit/
├── 00_preprocess/       # CSV 配对编号、wo-wi 差分、CSV 转 BMP
├── 01_diffraction/      # V7/V8/V9 三种衍射校正
├── 02_segmentation/     # MobileSeg 推理代码和已训练权重
├── 03_quality/          # FFQ-Lite V1.5.1 质量评分
├── 04_denoising/        # 固定的 V3 UNet 推理代码和 model_Unet_V3_best.pth
├── 05_optional_growth/  # 实验性结构生长（不进入默认流水线）
└── run_frontend_pipeline.py
```

## 环境

推荐 Python 3.10 与已有 `pyfing` 环境。安装依赖：

```bash
pip install -r requirements.txt
```

GPU 仅被 PyTorch 的去噪/分割/生长模型使用；缺少 CUDA 时会回退 CPU（较慢）。

## 输入格式

所有原始数据放在一个根目录，每个手指一个子目录。每个采集 pair 必须有五个
Rgd：`1421, 1423, 1425, 1427, 1429`，并同时存在 `wi` 和 `wo`。

```text
input_root/
└── finger_A/
    ├── wi-pair_1-...-Rgd=1421-....csv
    ├── wi-pair_1-...-Rgd=1423-....csv
    ├── ...
    ├── wo-pair_1-...-Rgd=1421-....csv
    └── ...
```

CSV 是二维数值矩阵。`wi`/`wo`、pair 编号与 Rgd 必须从文件名可解析。若原始
文件尚无 `pair_N`，先预览并确认配对，再执行重命名：

```bash
python 00_preprocess/rename_pairs.py --input-root /data/raw
python 00_preprocess/rename_pairs.py --input-root /data/raw --apply
```

该命令会同时处理 wi 和 wo；默认不写入任何文件。

## 一键流程（推荐）

从工具包根目录运行。输出根目录必须是新的空目录。

```bash
# 默认 V7：差分 -> 五相位 BMP -> V7 -> V3 UNet 去噪
python run_frontend_pipeline.py \
  --input-root /data/normal \
  --output-root /data/output_v7

# V8：独立 wo/wi 初相；附带 mask 与质量评分
python run_frontend_pipeline.py \
  --input-root /data/normal \
  --output-root /data/output_v8 \
  --diffraction v8 --run-segmentation --run-quality --device cuda

# V9：共同初相后进行 wo/wi 复数全局相位对齐
python run_frontend_pipeline.py \
  --input-root /data/normal \
  --output-root /data/output_v9 \
  --diffraction v9 --run-segmentation --run-quality
```

V7 的输出为：

```text
output_v7/
├── 01_diff/<finger>/diff-pair_N-Rgd=*.csv
├── 02_pic/<finger>/diff-pair_N-Rgd=*.bmp
├── 03_diffraction_v7/<finger>/pair_N.bmp
├── 04_denoise/<finger>/pair_N.bmp       # 默认最终图像
├── 05_mask/<finger>/pair_N.bmp          # 仅 --run-segmentation
└── 06_quality/quality_results.csv       # 仅 --run-quality
```

V8/V9 直接读取原始 CSV，因此没有 `01_diff`、`02_pic`；校正结果在
`03_diffraction_v8` 或 `03_diffraction_v9`。`--skip-denoise` 可让后续分割/
评分直接使用衍射校正结果。初次验证可加 `--folder finger_A --limit-pairs 3`；
其中 `--limit-pairs` 当前适用于 V8/V9，V7 可用 `--folder` 限制范围。

## 衍射校正接口与参数

| 接口 | 输入 | 核心差异 | 输出 |
| --- | --- | --- | --- |
| `diffaractionv7.py` | 五张 `wo-wi` 差分 BMP | 五相位解调、反卷积 | `pair_N.bmp` |
| `diffaractionv8.py` | 原始 wi/wo CSV | wi 与 wo 各自估计初始相位后相减 | `pair_N.bmp` |
| `diffaractionv9.py` | 原始 wi/wo CSV | 共同初相，wi 复数场全局对齐至 wo 后相减 | `pair_N.bmp` |

三种版本都会依次执行：五相位复数解调、整行/整列均值去除、非重叠 `10×10`
局部复数均值去除、衍射反卷积、显示相位优化。边缘小块只使用图像内像素。

可在 `01_diffraction/diffaractionv7.py` 顶部修改物理参数（频率、采样间隔、
层厚、反卷积正则项）和局部滤波参数：

```python
DO_LOCAL_BLOCK_CORRECTION = True
LOCAL_BLOCK_ROW = 10
LOCAL_BLOCK_COL = 10
```

V8/V9 复用 V7 的这些公共物理与滤波参数。单独运行时使用 `--help` 查看全部
参数；V8/V9 可加 `--folder finger_A --limit-pairs 3 --flat-output` 做小样本验证。

## 去噪、分割与质量评分

`04_denoising/models/model_Unet_V3_best.pth` 是固定使用的 V3 UNet 权重。可单独运行：

```bash
python 04_denoising/infer_unet_v3.py \
  --input-root /data/corrected --output-root /data/denoised
```

分割模型为 MobileSeg，输入任意灰度图；内部缩放到训练尺寸 `100×110` 推理，
输出时最近邻还原为原图尺寸。输出 mask 保持与输入完全相同的相对路径，能够
直接作为质量评分的 `--mask_dir`：

```bash
python 02_segmentation/infer_mask.py \
  --input-root /data/denoised --output-root /data/mask --threshold 0.5 --device cuda

python 03_quality/quality_score.py \
  --input_dir /data/denoised --mask_dir /data/mask \
  --output_dir /data/quality --recursive --no_score_bins
```

质量评分输出 `fingerprint_quality_results.csv`、`ffq_summary.json`；`--save_debug` 会额外输出
方向场和中间可视化，便于排查，但会显著增大文件量。主要可改参数：
`--max_side`（默认 320）、`--block_size`（默认 16）、`--sobel_ksize`（3/5/7）。

## 可选纹线生长

`05_optional_growth` 放入了已训练的 Structural Growth V3 及其训练/导出依赖，
只用于研究“严重缺失区域的结构补全”，不属于默认稳定流程。它需要预先构造
严重区域 mask 与候选目录，详细参数见：

```bash
python 05_optional_growth/export_structural_growth_v3.py --help
```

请先在少量样本上人工检查，避免在低置信区域生成不可信纹线。

## 注意

- 所有推理脚本拒绝覆盖非空输出目录；每次运行请使用新输出目录。
- 本工具包不携带原始采集数据、训练数据、历史实验输出或 HardNet/匹配代码。
- V7/V8/V9 是三种可比校正接口，不应在同一个输出目录内混用结果。
