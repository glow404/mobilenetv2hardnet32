# hardnet_train

`hardnet_train` 是指纹 patch 版 HardNet 训练包。它不负责生成 patch，而是读取
`pair_build` 已经生成好的 `train_pairs.csv` / `val_pairs.csv` 和 patch PNG 文件。

## 文件说明

- `model.py`：保留原 HardNet、MobileHardNet 与 HardNet Strong V2，统一接收 `[B, 1, 32, 32]` patch。
- `loss.py`：top-k hardest-in-batch triplet margin loss。
- `data.py`：CSV 数据集、坐标契约、PIL patch 读取、union-find 物理点分组和训练 batch sampler。
- `negative_sampling.py`：同指负样本的 16px 方形邻域规则，供训练和验证共同使用。
- `validation.py`：固定验证 batch 计划；batch 构造、候选掩码和 top-k 规则与训练共用。
- `metrics.py`：运行均值、ROC/ranking、距离分位数和 active-triplet 指标。
- `optim.py`：SGD/AdamW 工厂与 decay/no-decay 参数分组。
- `train.py`：训练主入口，负责配置解析、训练、固定验证、checkpoint 和 metrics 输出。
- `config.yaml`：唯一的正式训练配置，默认训练 Strong V2。
- `smoke_config.yaml`：快速自检配置，只跑极少 step。

## 输出文件

训练结果默认写入 `../outputs/models/hardnet_train_strong_v2_256_fixed_in_batch_v1/`：

- `best.pt`：固定 in-batch 计划上 `val_fpr_at_tpr95` 最低的 checkpoint。
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
- `val_valid_anchor_count`：当前固定计划中至少存在一个合法负样本的 anchor 数量。
- `val_skipped_anchor_count`：当前 batch 内完全没有合法负样本、因此未参与 loss/ROC 的 anchor 数量。
- `lr`
- `early_stop_best_fpr_at_tpr95`
- `no_improve_epochs`

浮点指标以 4 位有效数字写入 CSV；epoch 和早停计数保持整数。删除的 ROC AUC、EER、跨指分项和均值仍会在内存中用于 checkpoint 验证结果，但不再扩散到逐 epoch CSV。

训练曲线同步聚焦于同指误接受、严格低 FPR 下的 TPR、Recall@1，以及正负困难尾部。旧 schema 的 `metrics.csv` 不能与新 schema 混写，继续训练必须使用匹配新协议的新输出目录。

## 网络架构切换

`config.yaml` 使用 `descriptor_dim: auto` 和 `output_dir: auto`，会按架构选择兼容维度及独立输出目录：

| `model.architecture` | 自动维度 | 自动输出目录 | 网络状态 |
| --- | ---: | --- | --- |
| `hardnet_strong_v2` | 256 | `outputs/models/hardnet_train_strong_v2_256_fixed_in_batch_v1` | 默认高精度主干 |
| `mobile_hardnet` | 128 | `outputs/models/hardnet_train_mobile_128_fixed_in_batch_v1` | 保留的 MobileHardNet 轻量主干 |
| `hardnet` | 128 | `outputs/models/hardnet_train_hardnet_128_fixed_in_batch_v1` | 论文版 HardNet 主干 |

因此，只把下面一行改为 `mobile_hardnet`，即可使用原先的 MobileHardNet 网络结构训练：

```yaml
model:
  architecture: mobile_hardnet
  descriptor_dim: auto
```

该切换会自动改变网络结构、描述子维度和输出目录。当前 `dropout: 0.12`、`warmup_epochs: 3`、`lr: 0.08` 是 Strong V2 默认训练参数；若要严格复现旧 Mobile 配置，还应改回 `dropout: 0.1`、`warmup_epochs: 2`、`lr: 0.1`。

## 早停规则

训练监控固定验证 batch 计划上的 `val_fpr_at_tpr95`：阈值达到 95% 正样本 TPR 时，实际被 top-k 选中的合法负样本被误接受的比例。该指标越低越好。

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

固定验证使用 `fixed_in_batch_v1`，不再为每个 anchor 预先凑固定数量的同指/跨指负样本。训练启动时按固定 `seed`：

1. `validation.finger_count: auto` 时使用 `min(验证集可用手指数, validation.batch_size)`，并把实际值写入 `resolved_config.json` 和 checkpoint；
2. 构造 `validation.batch_count` 个固定验证 batch；
3. 每根手指的 `image_pair_id` 固定洗牌后依次使用，全部用完时重新洗牌循环；
4. 同一 batch 中每根手指仍只使用一个 `image_pair_id`；
5. 每轮验证复用完全相同的 batch 顺序；
6. 直接复用训练的 `point_group`、同指 16px 空间过滤、双向候选和 top-k 规则。

例如：

```yaml
validation:
  protocol: fixed_in_batch_v1
  seed: 10042
  finger_count: auto
  batch_count: 256
  batch_size: 24
```

当前验证 CSV 有 3 根手指，因此 `auto` 会解析为 3，生成 `256` 个固定 batch，每个 batch 共 `24` 对正样本，平均每根手指提供 `8` 对。若验证集手指数超过 `batch_size`，`auto` 最多使用 `batch_size` 根手指，以保证每根手指至少贡献一对样本。

`batch_count` 与训练的 `steps_per_epoch` 类似，负责控制每轮执行多少个 batch，但验证不反向传播。某根手指拥有的 image pair 少于 `batch_count` 时会循环使用，不会因为 image pair 数量不足而报错。

合法候选少于 `hard_negative_top_k` 时只使用现有候选；某个 anchor 完全没有合法候选时只跳过该 anchor 并计入 `val_skipped_anchor_count`。只有整个固定计划都没有有效 anchor 时才报错。

新协议改变了验证样本与指标语义，旧验证计划 checkpoint 不能恢复优化器状态，也不能与新 `metrics.csv` 混写。

命令行可临时覆盖：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/config.yaml `
  --hard-negative-strategy different_finger `
  --hard-negative-top-k 3 `
  --fingers-per-batch 8 `
  --val-finger-count auto `
  --val-batch-count 256 `
  --val-batch-size 24
```

## 训练与验证风险检查

程序会在启动阶段提前拒绝以下确定性错误：

- `training.epochs <= 0`、`steps_per_epoch <= 0` 或 `batch_size < 2`；
- `training.fingers_per_batch < 1`，或实际每 batch 手指数超过 `batch_size`；
- `validation.batch_count < 1`、`batch_size < 2`，或显式手指数超过验证集/`batch_size`；
- `different_finger` 策略实际少于 2 根训练或验证手指；
- 验证计划超过一千万个正样本槽位；
- CSV 缺失、为空、坐标字段不完整，或配置要求的 CUDA/BF16 不可用；
- checkpoint、模型结构、优化器或训练/验证协议与当前配置不兼容；
- 非续训模式下输出目录已有 `metrics.csv`，避免覆盖既有实验。

以下风险依赖实际运行环境，无法仅靠 YAML 完全排除：

- **训练显存不足**：HardNet 距离矩阵按 `training.batch_size²` 增长；出现 CUDA OOM 时优先减小训练 `batch_size`。
- **patch 文件不存在或损坏**：CSV 可以正常读取，但 DataLoader 真正打开图片时仍会失败；错误路径中会包含具体 patch 文件。
- **没有合法负样本**：空间过滤、`point_group` 屏蔽和手指数量可能让部分 anchor 被跳过；若整个验证计划都没有有效 anchor，验证会明确报错。
- **验证预算过大**：image pair 不足只会循环使用，不会报错；每批指标计算完会立即转存 CPU，因此 `batch_count` 不直接提高单步 GPU 显存峰值，但固定计划、验证耗时和 CPU 指标缓存仍会随 `batch_count × batch_size` 近似线性增长。超过十万个正样本槽位会发出警告，超过一千万会提前拒绝。
- **自动手指数过多**：若 `auto` 最终解析为 `finger_count == batch_size`，每根手指每 batch 只有一对样本，同指负样本指标不可用，程序会发出警告。

启动日志会显示实际验证手指数、固定 batch 数、每根手指使用的唯一 image pair 范围，以及发生 image pair 循环的手指数。

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

只有模型架构、优化器名称、参数分组以及训练/验证候选协议完全兼容时才允许恢复训练。旧模型 checkpoint 仍可用于推理或作为预训练权重；旧 fixed-pair checkpoint 和旧指标 CSV 不能在当前输出目录续写。

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
