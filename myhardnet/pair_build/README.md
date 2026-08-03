# pair_build

`pair_build` 是 HardNet 指纹训练数据构建目录，负责把原始小指纹图像转换成
HardNet 可训练的正样本 patch 对。

## 主要文件

- `config.yaml`：数据构建配置，包括原始数据路径、输出路径、SIFT 参数、匹配过滤参数、patch 裁剪参数。
- `main.py`：数据构建入口，支持 `index`、`split`、`build_positive`、`extract_patches`、`qa`、`all` 阶段。
- `index_builder.py`：扫描原始图像，建立样本索引。
- `split_builder.py`：按 `finger_id` 划分 train/val，避免同一手指泄漏到不同 split。
- `positive_builder.py`：使用 SIFT、RANSAC 和几何扩展构建正样本对应点；两类来源统一经过局部块 ZNCC 硬过滤，并按 16px 方形坐标间隔去除以后可能形成近邻伪负样本的 correspondence。
- `texture_quality.py`：提供训练与 ZNCC 共用的方向对齐 patch 裁剪，以及带平移搜索和 180° 二义性处理的块级 ZNCC。
- `patch_extractor.py`：根据正样本对应点裁剪并保存 32x32 patch，同时把 `image_a_id/image_b_id` 与 `x_a/y_a/x_b/y_b` 原样写入各 split CSV。
- `qa_tools.py`：生成 QA 报告和可视化预览图。
- `dataset_hardnet.py`：早期/辅助的 HardNet 正样本 DataLoader。
- `utils.py`：路径、CSV、JSON、图像读写等通用工具。
- `scheme.md`：数据构建思路记录。

## 常用命令

完整构建：

```powershell
python pair_build/main.py --config pair_build/config.yaml --stage all
```

只生成 QA：

```powershell
python pair_build/main.py --config pair_build/config.yaml --stage qa
```

## 输出关系

默认输出到：

```text
outputs/hardnet_dataset/
```

`hardnet_train` 会读取其中的：

- `train_pairs.csv`
- `val_pairs.csv`
- `patches/`

每个 pairs CSV 都必须保留以下空间字段：

```text
image_a_id, image_b_id, x_a, y_a, x_b, y_b
```

构建阶段的 `matching.negative_min_coordinate_separation_px: 16` 使用
`max(|dx|, |dy|) >= 16` 规则，并要求一组 correspondence 在 A、B 两侧都满足。
它不会生成负样本，而是提前去掉以后在同一个 image pair 内可能互为近邻伪负样本的点。

## 正样本 ZNCC 硬过滤

`matching.positive_zncc` 同时约束 `sift_inlier` 和 `geometry_expanded`，两类来源使用同一个
`min_similarity` 门槛。处理顺序为：

1. 按各自 SIFT 主方向裁剪 patch，裁剪尺寸直接复用 `patch.patch_crop_size/patch_out_size`。
2. 对 B 侧 patch 同时检查原方向和 180° 旋转，并在 `max_shift_px` 范围搜索最佳平移。
3. 逐块计算 ZNCC，使用有效纹理块相关系数的中位数作为最终相似度。
4. 有效块少于 `min_valid_blocks`，或相似度低于 `min_similarity`，直接拒绝候选，不进入 `stability_score`、去重和最终选择。

通过过滤的样本会在 pairs CSV 中记录：

```text
zncc_available, zncc_similarity, zncc_valid_blocks,
zncc_best_shift_x, zncc_best_shift_y, zncc_used_180_rotation
```

`match_diagnostics.csv` 分别记录 `sift_inlier_zncc_rejected` 和
`geometry_expanded_zncc_rejected`；`positive_build_summary.json` 在
`zncc_rejection_counts` 中汇总两类来源的拒绝数量。
