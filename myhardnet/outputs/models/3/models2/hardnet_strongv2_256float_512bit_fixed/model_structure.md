# BinaryDescriptorModel 模型结构报告

## 汇总

| 项目 | 数值 |
| --- | ---: |
| 架构标识 | `residual_binary_hash_v1` |
| 模型类 | `BinaryDescriptorModel` |
| 描述子维度 | 512 |
| 输入形状 | `[1,1,32,32]` |
| 输出形状 | `[1,512]` |
| 总参数量 | 2,609,154 |
| 可训练参数量 | 1,447,424 |
| Conv/Linear MACs / patch | 129,548,288 |
| 约算 FLOPs / patch | 259,096,576 |

> 统计输入为单个 32×32 灰度 patch。MACs 只统计 Conv/Linear 的乘加；FLOPs 按 1 MAC≈2 FLOPs 约算。归一化、激活、池化、插值、拼接、注意力逐元素运算和残差相加未计入 MACs，因此该数值用于统一比较主干计算量，不等同于设备实测延迟。复合层的参数只统计其直接持有参数，子层参数与 MACs在各自明细行统计，避免重复。

## PyTorch 模块树

```text
BinaryDescriptorModel(
  (backbone): HardNetStrongV2(
    (stem): ContrastAwareStem(
      (local_branch): Sequential(
        (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): GroupNorm(8, 16, eps=1e-05, affine=True)
        (2): SiLU(inplace=True)
      )
      (contrast_branch): Sequential(
        (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): GroupNorm(8, 16, eps=1e-05, affine=True)
        (2): SiLU(inplace=True)
      )
      (fuse): ConvNormAct(
        (0): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): GroupNorm(8, 32, eps=1e-05, affine=True)
        (2): SiLU(inplace=True)
      )
    )
    (stage1): Sequential(
      (0): StableResBlock(
        (downsample): Identity()
        (shortcut): Identity()
        (norm1): GroupNorm(8, 32, eps=1e-05, affine=True)
        (conv1): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (norm2): GroupNorm(8, 32, eps=1e-05, affine=True)
        (conv2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (drop_path): DropPath()
      )
    )
    (down1): StableResBlock(
      (downsample): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (shortcut): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (norm1): GroupNorm(8, 32, eps=1e-05, affine=True)
      (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (norm2): GroupNorm(8, 64, eps=1e-05, affine=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (drop_path): DropPath()
    )
    (stage2): Sequential(
      (0): StableResBlock(
        (downsample): Identity()
        (shortcut): Identity()
        (norm1): GroupNorm(8, 64, eps=1e-05, affine=True)
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (norm2): GroupNorm(8, 64, eps=1e-05, affine=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (drop_path): DropPath()
      )
      (1): StableResBlock(
        (downsample): Identity()
        (shortcut): Identity()
        (norm1): GroupNorm(8, 64, eps=1e-05, affine=True)
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (norm2): GroupNorm(8, 64, eps=1e-05, affine=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (drop_path): DropPath()
      )
    )
    (down2): StableResBlock(
      (downsample): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (shortcut): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (norm1): GroupNorm(8, 64, eps=1e-05, affine=True)
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (norm2): GroupNorm(8, 128, eps=1e-05, affine=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (drop_path): DropPath()
    )
    (stage3): Sequential(
      (0): StableResBlock(
        (downsample): Identity()
        (shortcut): Identity()
        (norm1): GroupNorm(8, 128, eps=1e-05, affine=True)
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (norm2): GroupNorm(8, 128, eps=1e-05, affine=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (drop_path): DropPath()
      )
      (1): StableResBlock(
        (downsample): Identity()
        (shortcut): Identity()
        (norm1): GroupNorm(8, 128, eps=1e-05, affine=True)
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (norm2): GroupNorm(8, 128, eps=1e-05, affine=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (drop_path): DropPath()
      )
    )
    (fine_projection): ConvNormAct(
      (0): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): GroupNorm(8, 32, eps=1e-05, affine=True)
      (2): SiLU(inplace=True)
    )
    (context): ContextMixer(
      (input_projection): ConvNormAct(
        (0): Conv2d(160, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): GroupNorm(8, 128, eps=1e-05, affine=True)
        (2): SiLU(inplace=True)
      )
      (depthwise): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), groups=128, bias=False)
      (norm): GroupNorm(8, 128, eps=1e-05, affine=True)
      (pointwise): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (drop_path): DropPath()
    )
    (head_norm): Sequential(
      (0): GroupNorm(8, 128, eps=1e-05, affine=True)
      (1): SiLU(inplace=True)
    )
    (spatial_projection): Sequential(
      (0): Conv2d(128, 128, kernel_size=(8, 8), stride=(1, 1), groups=128, bias=False)
      (1): GroupNorm(8, 128, eps=1e-05, affine=True)
      (2): SiLU(inplace=True)
      (3): Dropout2d(p=0.06, inplace=False)
      (4): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    )
    (global_pool): GeM()
    (global_projection): Linear(in_features=128, out_features=256, bias=False)
    (fusion): ScalarWeightedFusion(
      (dropout): Dropout(p=0.12, inplace=False)
    )
    (output_norm): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
  )
  (hash_head): ResidualHashHead(
    (input_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    (input_projection): Linear(in_features=256, out_features=512, bias=False)
    (residual_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    (expand): Linear(in_features=512, out_features=1024, bias=True)
    (activation): GELU(approximate='none')
    (dropout): Dropout(p=0.1, inplace=False)
    (contract): Linear(in_features=1024, out_features=512, bias=True)
    (output_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    (output_projection): Linear(in_features=512, out_features=512, bias=True)
  )
)
```

## 逐层明细（按实际前向调用顺序）

| # | 层路径 | 类型 | 输入 | 输出 | 作用 | 参数量 | 可训练参数 | MACs/patch | FLOPs/patch≈ |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `backbone.stem` | ContrastAwareStem | `[1,1,32,32]` | `[1,32,32,32]` | 融合 3×3 局部纹理和局部对比度两条输入分支；计算量由子层统计 | 0 | 0 | — | — |
| 2 | `backbone.stem.local_branch.0` | Conv2d | `[1,1,32,32]` | `[1,16,32,32]` | 局部空间特征提取；Conv 1→16，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 144 | 0 | 147,456 | 294,912 |
| 3 | `backbone.stem.local_branch.1` | GroupNorm | `[1,16,32,32]` | `[1,16,32,32]` | 组归一化；groups=8，channels=16 | 32 | 0 | — | — |
| 4 | `backbone.stem.local_branch.2` | SiLU | `[1,16,32,32]` | `[1,16,32,32]` | SiLU 平滑非线性激活 | 0 | 0 | — | — |
| 5 | `backbone.stem.contrast_branch.0` | Conv2d | `[1,1,32,32]` | `[1,16,32,32]` | 局部空间特征提取；Conv 1→16，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 144 | 0 | 147,456 | 294,912 |
| 6 | `backbone.stem.contrast_branch.1` | GroupNorm | `[1,16,32,32]` | `[1,16,32,32]` | 组归一化；groups=8，channels=16 | 32 | 0 | — | — |
| 7 | `backbone.stem.contrast_branch.2` | SiLU | `[1,16,32,32]` | `[1,16,32,32]` | SiLU 平滑非线性激活 | 0 | 0 | — | — |
| 8 | `backbone.stem.fuse.0` | Conv2d | `[1,32,32,32]` | `[1,32,32,32]` | 通道投影与融合；Conv 32→32，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 1,024 | 0 | 1,048,576 | 2,097,152 |
| 9 | `backbone.stem.fuse.1` | GroupNorm | `[1,32,32,32]` | `[1,32,32,32]` | 组归一化；groups=8，channels=32 | 64 | 0 | — | — |
| 10 | `backbone.stem.fuse.2` | SiLU | `[1,32,32,32]` | `[1,32,32,32]` | SiLU 平滑非线性激活 | 0 | 0 | — | — |
| 11 | `backbone.stage1.0` | StableResBlock | `[1,32,32,32]` | `[1,32,32,32]` | 预激活残差块，含抗混叠下采样、LayerScale 与 DropPath；计算量由子层统计 | 32 | 0 | — | — |
| 12 | `backbone.stage1.0.downsample` | Identity | `[1,32,32,32]` | `[1,32,32,32]` | 恒等映射，保持残差捷径不变 | 0 | 0 | — | — |
| 13 | `backbone.stage1.0.shortcut` | Identity | `[1,32,32,32]` | `[1,32,32,32]` | 恒等映射，保持残差捷径不变 | 0 | 0 | — | — |
| 14 | `backbone.stage1.0.norm1` | GroupNorm | `[1,32,32,32]` | `[1,32,32,32]` | 组归一化；groups=8，channels=32 | 64 | 0 | — | — |
| 15 | `backbone.stage1.0.conv1` | Conv2d | `[1,32,32,32]` | `[1,32,32,32]` | 局部空间特征提取；Conv 32→32，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 9,216 | 0 | 9,437,184 | 18,874,368 |
| 16 | `backbone.stage1.0.norm2` | GroupNorm | `[1,32,32,32]` | `[1,32,32,32]` | 组归一化；groups=8，channels=32 | 64 | 0 | — | — |
| 17 | `backbone.stage1.0.conv2` | Conv2d | `[1,32,32,32]` | `[1,32,32,32]` | 局部空间特征提取；Conv 32→32，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 9,216 | 0 | 9,437,184 | 18,874,368 |
| 18 | `backbone.stage1.0.drop_path` | DropPath | `[1,32,32,32]` | `[1,32,32,32]` | 训练时按样本随机丢弃残差路径，p=0.0 | 0 | 0 | — | — |
| 19 | `backbone.down1` | StableResBlock | `[1,32,32,32]` | `[1,64,16,16]` | 预激活残差块，含抗混叠下采样、LayerScale 与 DropPath；计算量由子层统计 | 64 | 0 | — | — |
| 20 | `backbone.down1.downsample` | AvgPool2d | `[1,32,32,32]` | `[1,32,16,16]` | 平均池化抗混叠下采样；kernel=2，stride=2 | 0 | 0 | — | — |
| 21 | `backbone.down1.shortcut` | Conv2d | `[1,32,16,16]` | `[1,64,16,16]` | 通道投影与融合；Conv 32→64，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 2,048 | 0 | 524,288 | 1,048,576 |
| 22 | `backbone.down1.norm1` | GroupNorm | `[1,32,16,16]` | `[1,32,16,16]` | 组归一化；groups=8，channels=32 | 64 | 0 | — | — |
| 23 | `backbone.down1.conv1` | Conv2d | `[1,32,16,16]` | `[1,64,16,16]` | 局部空间特征提取；Conv 32→64，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 18,432 | 0 | 4,718,592 | 9,437,184 |
| 24 | `backbone.down1.norm2` | GroupNorm | `[1,64,16,16]` | `[1,64,16,16]` | 组归一化；groups=8，channels=64 | 128 | 0 | — | — |
| 25 | `backbone.down1.conv2` | Conv2d | `[1,64,16,16]` | `[1,64,16,16]` | 局部空间特征提取；Conv 64→64，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 36,864 | 0 | 9,437,184 | 18,874,368 |
| 26 | `backbone.down1.drop_path` | DropPath | `[1,64,16,16]` | `[1,64,16,16]` | 训练时按样本随机丢弃残差路径，p=0.013333333333333334 | 0 | 0 | — | — |
| 27 | `backbone.stage2.0` | StableResBlock | `[1,64,16,16]` | `[1,64,16,16]` | 预激活残差块，含抗混叠下采样、LayerScale 与 DropPath；计算量由子层统计 | 64 | 0 | — | — |
| 28 | `backbone.stage2.0.downsample` | Identity | `[1,64,16,16]` | `[1,64,16,16]` | 恒等映射，保持残差捷径不变 | 0 | 0 | — | — |
| 29 | `backbone.stage2.0.shortcut` | Identity | `[1,64,16,16]` | `[1,64,16,16]` | 恒等映射，保持残差捷径不变 | 0 | 0 | — | — |
| 30 | `backbone.stage2.0.norm1` | GroupNorm | `[1,64,16,16]` | `[1,64,16,16]` | 组归一化；groups=8，channels=64 | 128 | 0 | — | — |
| 31 | `backbone.stage2.0.conv1` | Conv2d | `[1,64,16,16]` | `[1,64,16,16]` | 局部空间特征提取；Conv 64→64，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 36,864 | 0 | 9,437,184 | 18,874,368 |
| 32 | `backbone.stage2.0.norm2` | GroupNorm | `[1,64,16,16]` | `[1,64,16,16]` | 组归一化；groups=8，channels=64 | 128 | 0 | — | — |
| 33 | `backbone.stage2.0.conv2` | Conv2d | `[1,64,16,16]` | `[1,64,16,16]` | 局部空间特征提取；Conv 64→64，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 36,864 | 0 | 9,437,184 | 18,874,368 |
| 34 | `backbone.stage2.0.drop_path` | DropPath | `[1,64,16,16]` | `[1,64,16,16]` | 训练时按样本随机丢弃残差路径，p=0.02666666666666667 | 0 | 0 | — | — |
| 35 | `backbone.stage2.1` | StableResBlock | `[1,64,16,16]` | `[1,64,16,16]` | 预激活残差块，含抗混叠下采样、LayerScale 与 DropPath；计算量由子层统计 | 64 | 0 | — | — |
| 36 | `backbone.stage2.1.downsample` | Identity | `[1,64,16,16]` | `[1,64,16,16]` | 恒等映射，保持残差捷径不变 | 0 | 0 | — | — |
| 37 | `backbone.stage2.1.shortcut` | Identity | `[1,64,16,16]` | `[1,64,16,16]` | 恒等映射，保持残差捷径不变 | 0 | 0 | — | — |
| 38 | `backbone.stage2.1.norm1` | GroupNorm | `[1,64,16,16]` | `[1,64,16,16]` | 组归一化；groups=8，channels=64 | 128 | 0 | — | — |
| 39 | `backbone.stage2.1.conv1` | Conv2d | `[1,64,16,16]` | `[1,64,16,16]` | 局部空间特征提取；Conv 64→64，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 36,864 | 0 | 9,437,184 | 18,874,368 |
| 40 | `backbone.stage2.1.norm2` | GroupNorm | `[1,64,16,16]` | `[1,64,16,16]` | 组归一化；groups=8，channels=64 | 128 | 0 | — | — |
| 41 | `backbone.stage2.1.conv2` | Conv2d | `[1,64,16,16]` | `[1,64,16,16]` | 局部空间特征提取；Conv 64→64，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 36,864 | 0 | 9,437,184 | 18,874,368 |
| 42 | `backbone.stage2.1.drop_path` | DropPath | `[1,64,16,16]` | `[1,64,16,16]` | 训练时按样本随机丢弃残差路径，p=0.04 | 0 | 0 | — | — |
| 43 | `backbone.down2` | StableResBlock | `[1,64,16,16]` | `[1,128,8,8]` | 预激活残差块，含抗混叠下采样、LayerScale 与 DropPath；计算量由子层统计 | 128 | 0 | — | — |
| 44 | `backbone.down2.downsample` | AvgPool2d | `[1,64,16,16]` | `[1,64,8,8]` | 平均池化抗混叠下采样；kernel=2，stride=2 | 0 | 0 | — | — |
| 45 | `backbone.down2.shortcut` | Conv2d | `[1,64,8,8]` | `[1,128,8,8]` | 通道投影与融合；Conv 64→128，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 8,192 | 0 | 524,288 | 1,048,576 |
| 46 | `backbone.down2.norm1` | GroupNorm | `[1,64,8,8]` | `[1,64,8,8]` | 组归一化；groups=8，channels=64 | 128 | 0 | — | — |
| 47 | `backbone.down2.conv1` | Conv2d | `[1,64,8,8]` | `[1,128,8,8]` | 局部空间特征提取；Conv 64→128，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 73,728 | 0 | 4,718,592 | 9,437,184 |
| 48 | `backbone.down2.norm2` | GroupNorm | `[1,128,8,8]` | `[1,128,8,8]` | 组归一化；groups=8，channels=128 | 256 | 0 | — | — |
| 49 | `backbone.down2.conv2` | Conv2d | `[1,128,8,8]` | `[1,128,8,8]` | 局部空间特征提取；Conv 128→128，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 147,456 | 0 | 9,437,184 | 18,874,368 |
| 50 | `backbone.down2.drop_path` | DropPath | `[1,128,8,8]` | `[1,128,8,8]` | 训练时按样本随机丢弃残差路径，p=0.05333333333333334 | 0 | 0 | — | — |
| 51 | `backbone.stage3.0` | StableResBlock | `[1,128,8,8]` | `[1,128,8,8]` | 预激活残差块，含抗混叠下采样、LayerScale 与 DropPath；计算量由子层统计 | 128 | 0 | — | — |
| 52 | `backbone.stage3.0.downsample` | Identity | `[1,128,8,8]` | `[1,128,8,8]` | 恒等映射，保持残差捷径不变 | 0 | 0 | — | — |
| 53 | `backbone.stage3.0.shortcut` | Identity | `[1,128,8,8]` | `[1,128,8,8]` | 恒等映射，保持残差捷径不变 | 0 | 0 | — | — |
| 54 | `backbone.stage3.0.norm1` | GroupNorm | `[1,128,8,8]` | `[1,128,8,8]` | 组归一化；groups=8，channels=128 | 256 | 0 | — | — |
| 55 | `backbone.stage3.0.conv1` | Conv2d | `[1,128,8,8]` | `[1,128,8,8]` | 局部空间特征提取；Conv 128→128，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 147,456 | 0 | 9,437,184 | 18,874,368 |
| 56 | `backbone.stage3.0.norm2` | GroupNorm | `[1,128,8,8]` | `[1,128,8,8]` | 组归一化；groups=8，channels=128 | 256 | 0 | — | — |
| 57 | `backbone.stage3.0.conv2` | Conv2d | `[1,128,8,8]` | `[1,128,8,8]` | 局部空间特征提取；Conv 128→128，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 147,456 | 0 | 9,437,184 | 18,874,368 |
| 58 | `backbone.stage3.0.drop_path` | DropPath | `[1,128,8,8]` | `[1,128,8,8]` | 训练时按样本随机丢弃残差路径，p=0.06666666666666667 | 0 | 0 | — | — |
| 59 | `backbone.stage3.1` | StableResBlock | `[1,128,8,8]` | `[1,128,8,8]` | 预激活残差块，含抗混叠下采样、LayerScale 与 DropPath；计算量由子层统计 | 128 | 0 | — | — |
| 60 | `backbone.stage3.1.downsample` | Identity | `[1,128,8,8]` | `[1,128,8,8]` | 恒等映射，保持残差捷径不变 | 0 | 0 | — | — |
| 61 | `backbone.stage3.1.shortcut` | Identity | `[1,128,8,8]` | `[1,128,8,8]` | 恒等映射，保持残差捷径不变 | 0 | 0 | — | — |
| 62 | `backbone.stage3.1.norm1` | GroupNorm | `[1,128,8,8]` | `[1,128,8,8]` | 组归一化；groups=8，channels=128 | 256 | 0 | — | — |
| 63 | `backbone.stage3.1.conv1` | Conv2d | `[1,128,8,8]` | `[1,128,8,8]` | 局部空间特征提取；Conv 128→128，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 147,456 | 0 | 9,437,184 | 18,874,368 |
| 64 | `backbone.stage3.1.norm2` | GroupNorm | `[1,128,8,8]` | `[1,128,8,8]` | 组归一化；groups=8，channels=128 | 256 | 0 | — | — |
| 65 | `backbone.stage3.1.conv2` | Conv2d | `[1,128,8,8]` | `[1,128,8,8]` | 局部空间特征提取；Conv 128→128，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 147,456 | 0 | 9,437,184 | 18,874,368 |
| 66 | `backbone.stage3.1.drop_path` | DropPath | `[1,128,8,8]` | `[1,128,8,8]` | 训练时按样本随机丢弃残差路径，p=0.08 | 0 | 0 | — | — |
| 67 | `backbone.fine_projection.0` | Conv2d | `[1,64,16,16]` | `[1,32,16,16]` | 通道投影与融合；Conv 64→32，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 2,048 | 0 | 524,288 | 1,048,576 |
| 68 | `backbone.fine_projection.1` | GroupNorm | `[1,32,16,16]` | `[1,32,16,16]` | 组归一化；groups=8，channels=32 | 64 | 0 | — | — |
| 69 | `backbone.fine_projection.2` | SiLU | `[1,32,16,16]` | `[1,32,16,16]` | SiLU 平滑非线性激活 | 0 | 0 | — | — |
| 70 | `backbone.context` | ContextMixer | `[1,160,8,8]` | `[1,128,8,8]` | 以扩张 depthwise 3×3 和 pointwise 1×1 单路径混合上下文；计算量由子层统计 | 128 | 0 | — | — |
| 71 | `backbone.context.input_projection.0` | Conv2d | `[1,160,8,8]` | `[1,128,8,8]` | 通道投影与融合；Conv 160→128，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 20,480 | 0 | 1,310,720 | 2,621,440 |
| 72 | `backbone.context.input_projection.1` | GroupNorm | `[1,128,8,8]` | `[1,128,8,8]` | 组归一化；groups=8，channels=128 | 256 | 0 | — | — |
| 73 | `backbone.context.input_projection.2` | SiLU | `[1,128,8,8]` | `[1,128,8,8]` | SiLU 平滑非线性激活 | 0 | 0 | — | — |
| 74 | `backbone.context.depthwise` | Conv2d | `[1,128,8,8]` | `[1,128,8,8]` | 逐通道空间特征提取；Conv 128→128，kernel=3×3，stride=1×1，dilation=2×2，groups=128 | 1,152 | 0 | 73,728 | 147,456 |
| 75 | `backbone.context.norm` | GroupNorm | `[1,128,8,8]` | `[1,128,8,8]` | 组归一化；groups=8，channels=128 | 256 | 0 | — | — |
| 76 | `backbone.context.pointwise` | Conv2d | `[1,128,8,8]` | `[1,128,8,8]` | 通道投影与融合；Conv 128→128，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 16,384 | 0 | 1,048,576 | 2,097,152 |
| 77 | `backbone.context.drop_path` | DropPath | `[1,128,8,8]` | `[1,128,8,8]` | 训练时按样本随机丢弃残差路径，p=0.08 | 0 | 0 | — | — |
| 78 | `backbone.head_norm.0` | GroupNorm | `[1,128,8,8]` | `[1,128,8,8]` | 组归一化；groups=8，channels=128 | 256 | 0 | — | — |
| 79 | `backbone.head_norm.1` | SiLU | `[1,128,8,8]` | `[1,128,8,8]` | SiLU 平滑非线性激活 | 0 | 0 | — | — |
| 80 | `backbone.spatial_projection.0` | Conv2d | `[1,128,8,8]` | `[1,128,1,1]` | 逐通道空间特征提取；Conv 128→128，kernel=8×8，stride=1×1，dilation=1×1，groups=128 | 8,192 | 0 | 8,192 | 16,384 |
| 81 | `backbone.spatial_projection.1` | GroupNorm | `[1,128,1,1]` | `[1,128,1,1]` | 组归一化；groups=8，channels=128 | 256 | 0 | — | — |
| 82 | `backbone.spatial_projection.2` | SiLU | `[1,128,1,1]` | `[1,128,1,1]` | SiLU 平滑非线性激活 | 0 | 0 | — | — |
| 83 | `backbone.spatial_projection.3` | Dropout2d | `[1,128,1,1]` | `[1,128,1,1]` | 训练时按通道随机失活，p=0.06 | 0 | 0 | — | — |
| 84 | `backbone.spatial_projection.4` | Conv2d | `[1,128,1,1]` | `[1,256,1,1]` | 通道投影与融合；Conv 128→256，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 32,768 | 0 | 32,768 | 65,536 |
| 85 | `backbone.global_pool` | GeM | `[1,128,8,8]` | `[1,128,1,1]` | 可学习广义均值全局池化，兼顾峰值响应和整体稳定性 | 1 | 0 | — | — |
| 86 | `backbone.global_projection` | Linear | `[1,128]` | `[1,256]` | 全连接特征投影 128→256 | 32,768 | 0 | 32,768 | 65,536 |
| 87 | `backbone.fusion` | ScalarWeightedFusion | `([1,256], [1,256])` | `[1,256]` | 单位化空间/全局描述子，并以一个可学习标量加权融合；计算量由子层统计 | 1 | 0 | — | — |
| 88 | `backbone.fusion.dropout` | Dropout | `[1,256]` | `[1,256]` | 训练时随机失活，p=0.12 | 0 | 0 | — | — |
| 89 | `backbone.output_norm` | BatchNorm1d | `[1,256]` | `[1,256]` | 批归一化，稳定特征分布；affine=False | 0 | 0 | — | — |
| 90 | `hash_head` | ResidualHashHead | `[1,256]` | `[1,512]` | 浮点描述子投影及残差哈希变换；计算量由子层统计 | 1,024 | 1,024 | — | — |
| 91 | `hash_head.input_norm` | LayerNorm | `[1,256]` | `[1,256]` | 对末端维度 (256,) 做层归一化 | 512 | 512 | — | — |
| 92 | `hash_head.input_projection` | Linear | `[1,256]` | `[1,512]` | 全连接特征投影 256→512 | 131,072 | 131,072 | 131,072 | 262,144 |
| 93 | `hash_head.residual_norm` | LayerNorm | `[1,512]` | `[1,512]` | 对末端维度 (512,) 做层归一化 | 1,024 | 1,024 | — | — |
| 94 | `hash_head.expand` | Linear | `[1,512]` | `[1,1024]` | 全连接特征投影 512→1024 | 525,312 | 525,312 | 524,288 | 1,048,576 |
| 95 | `hash_head.activation` | GELU | `[1,1024]` | `[1,1024]` | GELU 平滑非线性激活 | 0 | 0 | — | — |
| 96 | `hash_head.dropout` | Dropout | `[1,1024]` | `[1,1024]` | 训练时随机失活，p=0.1 | 0 | 0 | — | — |
| 97 | `hash_head.contract` | Linear | `[1,1024]` | `[1,512]` | 全连接特征投影 1024→512 | 524,800 | 524,800 | 524,288 | 1,048,576 |
| 98 | `hash_head.output_norm` | LayerNorm | `[1,512]` | `[1,512]` | 对末端维度 (512,) 做层归一化 | 1,024 | 1,024 | — | — |
| 99 | `hash_head.output_projection` | Linear | `[1,512]` | `[1,512]` | 全连接特征投影 512→512 | 262,656 | 262,656 | 262,144 | 524,288 |
