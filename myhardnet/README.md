# myhardnet

本项目用于训练面向小指纹图像的 HardNet 局部 patch 描述子。当前流程分为两层：

1. `pair_build/`：已有的数据构建管线，负责从指纹图像中生成正样本 patch 对、训练/验证/测试 CSV、QA 图像等。
2. `hardnet_train/`：HardNet 训练管线，读取 `pair_build` 的 CSV 和 patch，训练 128 维 L2 归一化描述子。

## 目录说明

- `hardnet.pdf`：HardNet 论文原文。
- `pair_build/`：正样本数据集构建代码与配置。
- `hardnet_train/`：本次新增的网络、loss、采样器和训练入口。
- `outputs/`：数据构建与训练输出目录，包含 patch、CSV、checkpoint 和日志。
- `finger/`：已有的指纹匹配/C++ 相关代码。

## 训练入口

正式训练：

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

快速自检：

```powershell
python -m hardnet_train.train --config hardnet_train/smoke_config.yaml
```

## 关键设计

- 同时保留原 HardNet 和轻量 MobileHardNet；输入 `1x32x32`，输出 128 维单位描述子。
- loss 使用 HardNet 论文的 hardest-in-batch triplet margin。
- batch 采样结合指纹数据特点：每个 batch 包含多个手指，每个手指只使用一个两图组合。
- 使用 union-find 合并跨图正样本连通的关键点，避免同一物理点被误当负样本。
- 验证使用由 seed 确定的固定正样本和合法负样本候选池；完整候选池计算 ROC/距离分布，动态 top-k 只计算 loss，不按同指纹/跨指纹拆分。
- 默认优化器为 SGD + Nesterov，并对 BatchNorm 参数和 bias 禁用 weight decay。
