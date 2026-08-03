# hardnet_train

`hardnet_train` 是指纹 patch 版 HardNet 训练包。它不负责生成 patch，而是读取
`pair_build` 已经生成好的 `train_pairs.csv` / `val_pairs.csv` 和 patch PNG 文件。

## 文件说明

- `model.py`：保留原 HardNet、MobileHardNet 与 HardNet Strong V2，统一接收 `[B, 1, 32, 32]` patch。
- `loss.py`：top-k hardest-in-batch triplet margin loss。
- `data.py`：CSV 数据集、坐标契约、PIL patch 读取、union-find 物理点分组和训练 batch sampler。
- `negative_sampling.py`：同指负样本的 16px 方形邻域规则，供训练和验证共同使用。
- `validation.py`：固定正样本、空间过滤后的同指负样本和跨指纹负样本协议。
- `metrics.py`：运行均值、ROC/ranking、距离分位数和 active-triplet 指标。
- `optim.py`：SGD/AdamW 工厂与 decay/no-decay 参数分组。
- `train.py`：训练主入口，负责配置解析、训练、固定验证、checkpoint 和 metrics 输出。
- `config.yaml`：唯一的正式训练配置，默认训练 Strong V2。
- `smoke_config.yaml`：快速自检配置，只跑极少 step。

## 输出文件

训练结果默认写入 `../outputs/models/hardnet_train_strong_v2_256/`：

- `best.pt`：固定协议 `val_fpr_at_tpr95` 最低的 checkpoint。
- `last.pt`：最后一个 epoch 的 checkpoint。
- `metrics.csv`：每个 epoch 只记录与最终匹配最相关的精简指标，浮点值最多保留 4 位有效数字。
- `resolved_config.json`：包含命令行覆盖后的实际配置快照。

### 精简训练指标

新的 `metrics.csv` 每个 epoch 只写入以下字段：

- `epoch`
- `train_loss`
- `val_loss`
- `val_same_finger_loss`
- `val_pos_p95`
- `val_same_finger_neg_p01`
- `val_same_finger_tail_gap`：`val_same_finger_neg_p01 - val_pos_p95`
- `val_fpr_at_tpr95`
- `val_same_finger_fpr_at_tpr95`
- `val_same_finger_tpr_at_fpr_1e_4`
- `val_same_finger_recall_at_1`
- `lr`
- `early_stop_best_fpr_at_tpr95`
- `no_improve_epochs`

浮点指标以 4 位有效数字写入 CSV；epoch 和早停计数保持整数。删除的 ROC AUC、EER、跨指分项和均值仍会在内存中用于 checkpoint 验证结果，但不再扩散到逐 epoch CSV。

训练曲线同步聚焦于同指误接受、严格低 FPR 下的 TPR、Recall@1，以及正负困难尾部。旧 schema 的 `metrics.csv` 不能与新 schema 混写，继续训练必须使用匹配新协议的新输出目录。

## 网络架构切换

`config.yaml` 使用 `descriptor_dim: auto` 和 `output_dir: auto`，会按架构选择兼容维度及独立输出目录：

| `model.architecture` | 自动维度 | 自动输出目录 | 网络状态 |
| --- | ---: | --- | --- |
| `hardnet_strong_v2` | 256 | `outputs/models/hardnet_train_strong_v2_256` | 默认高精度主干 |
| `mobile_hardnet` | 128 | `outputs/models/hardnet_train_mobile_128` | 保留的 MobileHardNet 轻量主干 |
| `hardnet` | 128 | `outputs/models/hardnet_train_hardnet_128` | 论文版 HardNet 主干 |

因此，只把下面一行改为 `mobile_hardnet`，即可使用原先的 MobileHardNet 网络结构训练：

```yaml
model:
  architecture: mobile_hardnet
  descriptor_dim: auto
```

该切换会自动改变网络结构、描述子维度和输出目录。当前 `dropout: 0.12`、`warmup_epochs: 3`、`lr: 0.08` 是 Strong V2 默认训练参数；若要严格复现旧 Mobile 配置，还应改回 `dropout: 0.1`、`warmup_epochs: 2`、`lr: 0.1`。

## 早停规则

训练监控固定协议上的 `val_fpr_at_tpr95`：阈值达到 95% 正样本 TPR 时，负样本被误接受的比例。该指标越低越好。

当前正式配置将 `early_stop_patience` 设为 `0`，即关闭早停。启用后，规则为：

```text
连续 patience 个 epoch 没有让 val_fpr_at_tpr95 相对历史 best 达到配置的下降比例，则停止训练。
```

例如历史 best 为 `0.9868`，那么下一次必须低于：

```text
0.9868 * (1 - 0.01) = 0.976932
```

才算一次显著改善。否则累计 `no_improve_epochs`。

可在配置中调整：

```yaml
training:
  early_stop_patience: 3
  early_stop_min_relative_improvement: 0.01
```

也可以用命令行覆盖：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/config.yaml `
  --early-stop-patience 3 `
  --early-stop-min-delta 0.01
```

## Margin

HardNet 论文原始 triplet margin 为 `1.0`。

命令行覆盖示例：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/config.yaml `
  --margin 0.5
```

## 采样策略

HardNet 的 top-k hardest-in-batch loss 会从 batch 内选择最相似的若干非正样本。
指纹数据里同一物理点可能跨多张图重复出现，因此普通随机 batch 容易产生伪负样本。

负样本候选策略由 `training.hard_negative_strategy` 控制：

- `same_finger_allowed`：允许同一 `finger_id` 内的样本成为负样本，但会先屏蔽同一 `point_group`。如果 CSV 中存在 A-B、B-C 两条正样本关系，即使没有显式 A-C，union-find 也会把 A、B、C 合并为同一物理点组，loss 不会把 A-C 当负样本。对仍然合法的同指候选，仅在同一原图坐标系内比较坐标，并要求 `max(|dx|, |dy|) >= same_finger_min_coordinate_separation_px`；默认阈值为 16px，等于 16 时允许。
- `different_finger`：最难负样本只能来自不同 `finger_id`。使用这个策略时，每个 batch 至少需要 2 根手指，因此 `fingers_per_batch` 也必须大于等于 2；同指空间规则不会影响跨指候选。

难负样本数量由 `training.hard_negative_top_k` 控制，默认是 `3`。对每条正样本，loss 会合并
anchor→positive 和 positive→anchor 两个方向的合法负样本候选，选距离最小的 k 个，分别计算
triplet margin loss 后求均值。合法候选不足 k 个时只使用现有候选；设为 `1` 可复现原来的
单一 hardest-negative 行为。日志中的 `neg_dist` 是实际选中 top-k 负样本的平均距离。

本包还采用两层采样保护：

1. 每个 batch 选多个 `finger_id`。
2. 每个 `finger_id` 在当前 batch 中只使用一个 `image_pair_id`，也就是只来自两张图。
3. 通过 union-find 把正样本关系连起来的关键点合并为 `point_group`。
4. loss 计算时屏蔽同一 `point_group` 的候选负样本。
5. 对同指候选，anchor→positive 方向使用 A 图坐标，positive→anchor 方向使用 B 图坐标；只有同一坐标参考图内、位于锚点中心 32x32 方形邻域之外的点才合法。

固定验证使用 `fixed_pairs_v2_spatial`。每个同指负样本必须来自 anchor 所在的原图、属于不同 `point_group`，并满足相同的 16px 方形间隔；跨指负样本不比较坐标。该协议改变了验证配对集合，旧 `fixed_pairs_v1` 指标不能与新指标直接混写，旧 checkpoint 也不能在原输出目录中续训。

命令行可临时覆盖：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/config.yaml `
  --hard-negative-strategy different_finger `
  --hard-negative-top-k 3 `
  --fingers-per-batch 8
```

## 常用命令

快速自检：

```powershell
python -m hardnet_train.train --config hardnet_train/smoke_config.yaml
```

正式训练：

```powershell
python -m hardnet_train.train --config hardnet_train/config.yaml
```

CUDA 大 batch 训练：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/config.yaml `
  --device cuda `
  --batch-size 512 `
  --fingers-per-batch 64 `
  --output-dir ../outputs/hardnet_train_cuda_b512
```

## 续训

续训必须继续使用原训练的有效配置和输出目录。命令行覆盖后的完整配置可在输出目录的 `resolved_config.json` 中确认。

当前 `hardnet_train_cuda_b512` 模型从 `last.pt` 续训：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/config.yaml `
  --device cuda `
  --batch-size 512 `
  --fingers-per-batch 64 `
  --output-dir ../outputs/hardnet_train_cuda_b512 `
  --resume auto
```

也可以显式指定同一个 checkpoint：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/config.yaml `
  --device cuda `
  --batch-size 512 `
  --fingers-per-batch 64 `
  --output-dir ../outputs/hardnet_train_cuda_b512 `
  --resume ../outputs/hardnet_train_cuda_b512/last.pt
```

`--resume auto` 表示必须找到当前输出目录的 `last.pt`，否则直接报错。`--resume-auto` 适合脚本化启动：checkpoint 存在时恢复，不存在时仅在空输出目录中从头训练。

续训会恢复：

- 模型权重；
- optimizer 状态；
- epoch；
- global step；
- 既有 `metrics.csv` 中的 best `val_fpr_at_tpr95` 和早停计数。

只有模型架构、优化器名称和参数分组兼容时才允许恢复训练。旧模型 checkpoint 仍可用于推理；旧指标 CSV 不兼容当前 schema，不能继续写入。

## 学习率调度

当前默认策略是：

```text
warmup + cosine decay
```

配置项：

```yaml
training:
  scheduler: warmup_cosine
  warmup_epochs: 2
  eta_min: 0.0001
```

含义：

1. 前 `warmup_epochs` 个 epoch 从 0 线性升到初始学习率；
2. 后续用余弦退火缓慢下降到 `eta_min`；
3. 不会像原来的线性衰减一样最后直接变成 0。

如果要复现论文原始线性衰减，可改为：

```yaml
training:
  scheduler: linear
```

## 训练曲线

训练结束后会自动生成：

```text
training_curves.png
```

图中包含：

- train/val loss；
- train/val 正负样本距离；
- `val_fpr_at_tpr95`；
- 学习率曲线。

如不需要绘图，可加：

```powershell
--no-plot
```
