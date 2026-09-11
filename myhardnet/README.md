# myhardnet

本项目用于构建指纹局部块数据集、训练 HardNet 描述子网络，并完成指纹匹配识别。
完整流程分为三个阶段：

1. `pair_build/`：从原始指纹图像生成正样本局部块、训练/验证 CSV 和质量检查结果。
2. `hardnet_train/`：读取构建好的数据，训练浮点描述子或二值描述子网络。
3. `match_new/`：使用训练完成的模型构建注册模板，执行离线评估和在线匹配识别。

## 目录说明

- `hardnet.pdf`：HardNet 论文原文。
- `pair_build/`：正样本数据集构建代码与配置。
- `hardnet_train/`：网络、损失函数、采样器和训练入口。
- `match_new/`：注册模板构建、匹配评估和在线识别入口。
- `outputs/`：数据构建与训练输出目录，包含 patch、CSV、checkpoint 和日志。
- `finger/`：已有的指纹匹配/C++ 相关代码。

以下命令均在 `myhardnet` 目录执行：

```powershell
cd C:\Users\ZYH\Desktop\harnet32\myhardnet
```

## 一、构建训练数据集

先在 `pair_build/config.yaml` 中确认原始数据目录和输出目录，然后完整执行：

```powershell
python pair_build/main.py --config pair_build/config.yaml --stage all
```

需要分阶段排查时，可以依次执行：

```powershell
python pair_build/main.py --config pair_build/config.yaml --stage index
python pair_build/main.py --config pair_build/config.yaml --stage split
python pair_build/main.py --config pair_build/config.yaml --stage build_positive
python pair_build/main.py --config pair_build/config.yaml --stage extract_patches
python pair_build/main.py --config pair_build/config.yaml --stage qa
```

默认训练配置读取数据构建阶段生成的：

```text
outputs/butieping/train_pairs.csv
outputs/butieping/val_pairs.csv
outputs/butieping/patches/
```

## 二、训练网络

浮点描述子正式训练：

```powershell
python -m hardnet_train.train --config hardnet_train/config.yaml
```

CUDA 训练示例：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/config.yaml `
  --device cuda `
  --batch-size 512 `
  --fingers-per-batch 64 `
  --output-dir ../outputs/hardnet_train_cuda_b512
```

长训练续训时，应继续使用原训练的有效参数和输出目录。当前 `hardnet_train_cuda_b512` 模型的命令为：

```powershell
python -m hardnet_train.train `
  --config hardnet_train/config.yaml `
  --device cuda `
  --batch-size 512 `
  --fingers-per-batch 64 `
  --output-dir ../outputs/hardnet_train_cuda_b512 `
  --resume auto
```

训练完成后会在输出目录生成 `training_curves.png`。

当前训练 best checkpoint 监控固定协议的四项 `val_checkpoint_selection_score`；`val_fpr_at_tpr95` 是其中一项。正式配置当前关闭 early stopping；启用时按综合分相对改善比例累计耐心轮数。

浮点网络训练完成后，如果需要二值描述子，先确认
`hardnet_train/config_binary_256.yaml` 中的 `backbone.checkpoint`
指向浮点训练生成的 `best.pt`，然后执行：

```powershell
python -m hardnet_train.train_binary `
  --config hardnet_train/config_binary_256.yaml `
  --device cuda
```

二值网络续训：

```powershell
python -m hardnet_train.train_binary `
  --config hardnet_train/config_binary_256.yaml `
  --resume auto
```

## 三、匹配识别

先在 `match_new/config_match_tuning.yaml` 中设置：

- 待评估指纹数据集 `data.image_root`；
- 浮点或二值模型 `model.checkpoint`；
- 输出目录、注册模板数量和匹配阈值。

完整执行模板构建、注册划分、匹配及 FAR/FRR 评估：

```powershell
python match_new\run_hardnet_matching.py `
  --config match_new\config_match_new.yaml
```

如果只修改了匹配、RANSAC、纹理融合或最终判定阈值，可以复用已有模板：

```powershell
python match_new\run_hardnet_matching.py `
  --config match_new\config_match_new.yaml `
  --skip-template-build
```

如果修改了模型、SIFT、局部块提取方式或关键点参数，不能复用旧模板。

对一张指纹执行在线身份验证：

```powershell
python match_new\run_online_unlock.py `
  --config match_new\config_match_new.yaml `
  --image "datasets\normal_pic\dy_L0\pair_41.bmp" `
  --identity "dy_L0"
```

在线批量延迟测试：

```powershell
python match_new\run_online_unlock.py `
  --config match_new\config_match_new.yaml `
  --benchmark `
  --limit 100
```

## 关键设计

- 同时保留原 HardNet、MobileHardNet 和 Strong V2；输入为 `1x32x32`，描述子维度由训练配置决定。
- loss 使用 HardNet 论文的 hardest-in-batch triplet margin。
- batch 采样结合指纹数据特点：每个 batch 包含多个手指，每个手指只使用一个两图组合。
- 使用 union-find 合并跨图正样本连通的关键点，避免同一物理点被误当负样本。
- 验证使用由 seed 确定的固定正样本和合法负样本候选池；完整候选池计算 ROC/距离分布，动态 top-k 只计算 loss，不按同指纹/跨指纹拆分。
- 默认优化器为 SGD + Nesterov，并对 BatchNorm 参数和 bias 禁用 weight decay。
