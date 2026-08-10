# BinaryDescriptorModel 模型结构报告

## 汇总

| 项目 | 数值 |
| --- | ---: |
| 架构标识 | `residual_binary_hash_v1` |
| 模型类 | `BinaryDescriptorModel` |
| 描述子维度 | 512 |
| 输入形状 | `[1,1,32,32]` |
| 输出形状 | `[1,512]` |
| 总参数量 | 1,560,176 |
| 可训练参数量 | 1,447,424 |
| Conv/Linear MACs / patch | 6,446,080 |
| 约算 FLOPs / patch | 12,892,160 |

> 统计输入为单个 32×32 灰度 patch。MACs 只统计 Conv/Linear 的乘加；FLOPs 按 1 MAC≈2 FLOPs 约算。归一化、激活、池化、插值、拼接、注意力逐元素运算和残差相加未计入 MACs，因此该数值用于统一比较主干计算量，不等同于设备实测延迟。复合层的参数只统计其直接持有参数，子层参数与 MACs在各自明细行统计，避免重复。

## PyTorch 模块树

```text
BinaryDescriptorModel(
  (backbone): MobileHardNet(
    (stem): Sequential(
      (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (features): Sequential(
      (0): InvertedResidual(
        (block): Sequential(
          (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
          (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
          (3): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): InvertedResidual(
        (block): Sequential(
          (0): Conv2d(16, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
          (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=32, bias=False)
          (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6(inplace=True)
          (6): Conv2d(32, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (7): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (2): InvertedResidual(
        (block): Sequential(
          (0): Conv2d(24, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
          (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
          (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6(inplace=True)
          (6): Conv2d(48, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (7): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (3): InvertedResidual(
        (block): Sequential(
          (0): Conv2d(24, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
          (3): Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=48, bias=False)
          (4): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6(inplace=True)
          (6): Conv2d(48, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (7): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (4): InvertedResidual(
        (block): Sequential(
          (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
          (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6(inplace=True)
          (6): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (7): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (5): InvertedResidual(
        (block): Sequential(
          (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
          (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
          (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6(inplace=True)
          (6): Conv2d(64, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (7): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (6): InvertedResidual(
        (block): Sequential(
          (0): Conv2d(48, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
          (4): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6(inplace=True)
          (6): Conv2d(96, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (7): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (7): InvertedResidual(
        (block): Sequential(
          (0): Conv2d(48, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
          (3): Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
          (4): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU6(inplace=True)
          (6): Conv2d(96, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
    (adapter): Sequential(
      (0): Conv2d(64, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (coordinate_attention): CoordinateAttention(
      (shared_projection): Sequential(
        (0): Conv2d(96, 8, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): Hardswish()
      )
      (height_projection): Conv2d(8, 96, kernel_size=(1, 1), stride=(1, 1))
      (width_projection): Conv2d(8, 96, kernel_size=(1, 1), stride=(1, 1))
    )
    (descriptor_head): Sequential(
      (0): Dropout(p=0.12, inplace=False)
      (1): Conv2d(96, 192, kernel_size=(8, 8), stride=(1, 1), groups=96, bias=False)
      (2): Conv2d(192, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    )
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
| 1 | `backbone.stem.0` | Conv2d | `[1,1,32,32]` | `[1,16,32,32]` | 局部空间特征提取；Conv 1→16，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 144 | 0 | 147,456 | 294,912 |
| 2 | `backbone.stem.1` | BatchNorm2d | `[1,16,32,32]` | `[1,16,32,32]` | 批归一化，稳定特征分布；affine=True | 32 | 0 | — | — |
| 3 | `backbone.stem.2` | ReLU6 | `[1,16,32,32]` | `[1,16,32,32]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 4 | `backbone.features.0` | InvertedResidual | `[1,16,32,32]` | `[1,16,32,32]` | MobileNetV2 倒残差块：扩展、深度卷积、线性投影；带残差相加；计算量由子层统计 | 0 | 0 | — | — |
| 5 | `backbone.features.0.block.0` | Conv2d | `[1,16,32,32]` | `[1,16,32,32]` | 逐通道空间特征提取；Conv 16→16，kernel=3×3，stride=1×1，dilation=1×1，groups=16 | 144 | 0 | 147,456 | 294,912 |
| 6 | `backbone.features.0.block.1` | BatchNorm2d | `[1,16,32,32]` | `[1,16,32,32]` | 批归一化，稳定特征分布；affine=True | 32 | 0 | — | — |
| 7 | `backbone.features.0.block.2` | ReLU6 | `[1,16,32,32]` | `[1,16,32,32]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 8 | `backbone.features.0.block.3` | Conv2d | `[1,16,32,32]` | `[1,16,32,32]` | 通道投影与融合；Conv 16→16，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 256 | 0 | 262,144 | 524,288 |
| 9 | `backbone.features.0.block.4` | BatchNorm2d | `[1,16,32,32]` | `[1,16,32,32]` | 批归一化，稳定特征分布；affine=True | 32 | 0 | — | — |
| 10 | `backbone.features.1` | InvertedResidual | `[1,16,32,32]` | `[1,24,16,16]` | MobileNetV2 倒残差块：扩展、深度卷积、线性投影；不使用残差；计算量由子层统计 | 0 | 0 | — | — |
| 11 | `backbone.features.1.block.0` | Conv2d | `[1,16,32,32]` | `[1,32,32,32]` | 通道投影与融合；Conv 16→32，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 512 | 0 | 524,288 | 1,048,576 |
| 12 | `backbone.features.1.block.1` | BatchNorm2d | `[1,32,32,32]` | `[1,32,32,32]` | 批归一化，稳定特征分布；affine=True | 64 | 0 | — | — |
| 13 | `backbone.features.1.block.2` | ReLU6 | `[1,32,32,32]` | `[1,32,32,32]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 14 | `backbone.features.1.block.3` | Conv2d | `[1,32,32,32]` | `[1,32,16,16]` | 逐通道空间特征提取；Conv 32→32，kernel=3×3，stride=2×2，dilation=1×1，groups=32 | 288 | 0 | 73,728 | 147,456 |
| 15 | `backbone.features.1.block.4` | BatchNorm2d | `[1,32,16,16]` | `[1,32,16,16]` | 批归一化，稳定特征分布；affine=True | 64 | 0 | — | — |
| 16 | `backbone.features.1.block.5` | ReLU6 | `[1,32,16,16]` | `[1,32,16,16]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 17 | `backbone.features.1.block.6` | Conv2d | `[1,32,16,16]` | `[1,24,16,16]` | 通道投影与融合；Conv 32→24，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 768 | 0 | 196,608 | 393,216 |
| 18 | `backbone.features.1.block.7` | BatchNorm2d | `[1,24,16,16]` | `[1,24,16,16]` | 批归一化，稳定特征分布；affine=True | 48 | 0 | — | — |
| 19 | `backbone.features.2` | InvertedResidual | `[1,24,16,16]` | `[1,24,16,16]` | MobileNetV2 倒残差块：扩展、深度卷积、线性投影；带残差相加；计算量由子层统计 | 0 | 0 | — | — |
| 20 | `backbone.features.2.block.0` | Conv2d | `[1,24,16,16]` | `[1,48,16,16]` | 通道投影与融合；Conv 24→48，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 1,152 | 0 | 294,912 | 589,824 |
| 21 | `backbone.features.2.block.1` | BatchNorm2d | `[1,48,16,16]` | `[1,48,16,16]` | 批归一化，稳定特征分布；affine=True | 96 | 0 | — | — |
| 22 | `backbone.features.2.block.2` | ReLU6 | `[1,48,16,16]` | `[1,48,16,16]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 23 | `backbone.features.2.block.3` | Conv2d | `[1,48,16,16]` | `[1,48,16,16]` | 逐通道空间特征提取；Conv 48→48，kernel=3×3，stride=1×1，dilation=1×1，groups=48 | 432 | 0 | 110,592 | 221,184 |
| 24 | `backbone.features.2.block.4` | BatchNorm2d | `[1,48,16,16]` | `[1,48,16,16]` | 批归一化，稳定特征分布；affine=True | 96 | 0 | — | — |
| 25 | `backbone.features.2.block.5` | ReLU6 | `[1,48,16,16]` | `[1,48,16,16]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 26 | `backbone.features.2.block.6` | Conv2d | `[1,48,16,16]` | `[1,24,16,16]` | 通道投影与融合；Conv 48→24，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 1,152 | 0 | 294,912 | 589,824 |
| 27 | `backbone.features.2.block.7` | BatchNorm2d | `[1,24,16,16]` | `[1,24,16,16]` | 批归一化，稳定特征分布；affine=True | 48 | 0 | — | — |
| 28 | `backbone.features.3` | InvertedResidual | `[1,24,16,16]` | `[1,32,8,8]` | MobileNetV2 倒残差块：扩展、深度卷积、线性投影；不使用残差；计算量由子层统计 | 0 | 0 | — | — |
| 29 | `backbone.features.3.block.0` | Conv2d | `[1,24,16,16]` | `[1,48,16,16]` | 通道投影与融合；Conv 24→48，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 1,152 | 0 | 294,912 | 589,824 |
| 30 | `backbone.features.3.block.1` | BatchNorm2d | `[1,48,16,16]` | `[1,48,16,16]` | 批归一化，稳定特征分布；affine=True | 96 | 0 | — | — |
| 31 | `backbone.features.3.block.2` | ReLU6 | `[1,48,16,16]` | `[1,48,16,16]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 32 | `backbone.features.3.block.3` | Conv2d | `[1,48,16,16]` | `[1,48,8,8]` | 逐通道空间特征提取；Conv 48→48，kernel=3×3，stride=2×2，dilation=1×1，groups=48 | 432 | 0 | 27,648 | 55,296 |
| 33 | `backbone.features.3.block.4` | BatchNorm2d | `[1,48,8,8]` | `[1,48,8,8]` | 批归一化，稳定特征分布；affine=True | 96 | 0 | — | — |
| 34 | `backbone.features.3.block.5` | ReLU6 | `[1,48,8,8]` | `[1,48,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 35 | `backbone.features.3.block.6` | Conv2d | `[1,48,8,8]` | `[1,32,8,8]` | 通道投影与融合；Conv 48→32，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 1,536 | 0 | 98,304 | 196,608 |
| 36 | `backbone.features.3.block.7` | BatchNorm2d | `[1,32,8,8]` | `[1,32,8,8]` | 批归一化，稳定特征分布；affine=True | 64 | 0 | — | — |
| 37 | `backbone.features.4` | InvertedResidual | `[1,32,8,8]` | `[1,32,8,8]` | MobileNetV2 倒残差块：扩展、深度卷积、线性投影；带残差相加；计算量由子层统计 | 0 | 0 | — | — |
| 38 | `backbone.features.4.block.0` | Conv2d | `[1,32,8,8]` | `[1,64,8,8]` | 通道投影与融合；Conv 32→64，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 2,048 | 0 | 131,072 | 262,144 |
| 39 | `backbone.features.4.block.1` | BatchNorm2d | `[1,64,8,8]` | `[1,64,8,8]` | 批归一化，稳定特征分布；affine=True | 128 | 0 | — | — |
| 40 | `backbone.features.4.block.2` | ReLU6 | `[1,64,8,8]` | `[1,64,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 41 | `backbone.features.4.block.3` | Conv2d | `[1,64,8,8]` | `[1,64,8,8]` | 逐通道空间特征提取；Conv 64→64，kernel=3×3，stride=1×1，dilation=1×1，groups=64 | 576 | 0 | 36,864 | 73,728 |
| 42 | `backbone.features.4.block.4` | BatchNorm2d | `[1,64,8,8]` | `[1,64,8,8]` | 批归一化，稳定特征分布；affine=True | 128 | 0 | — | — |
| 43 | `backbone.features.4.block.5` | ReLU6 | `[1,64,8,8]` | `[1,64,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 44 | `backbone.features.4.block.6` | Conv2d | `[1,64,8,8]` | `[1,32,8,8]` | 通道投影与融合；Conv 64→32，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 2,048 | 0 | 131,072 | 262,144 |
| 45 | `backbone.features.4.block.7` | BatchNorm2d | `[1,32,8,8]` | `[1,32,8,8]` | 批归一化，稳定特征分布；affine=True | 64 | 0 | — | — |
| 46 | `backbone.features.5` | InvertedResidual | `[1,32,8,8]` | `[1,48,8,8]` | MobileNetV2 倒残差块：扩展、深度卷积、线性投影；不使用残差；计算量由子层统计 | 0 | 0 | — | — |
| 47 | `backbone.features.5.block.0` | Conv2d | `[1,32,8,8]` | `[1,64,8,8]` | 通道投影与融合；Conv 32→64，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 2,048 | 0 | 131,072 | 262,144 |
| 48 | `backbone.features.5.block.1` | BatchNorm2d | `[1,64,8,8]` | `[1,64,8,8]` | 批归一化，稳定特征分布；affine=True | 128 | 0 | — | — |
| 49 | `backbone.features.5.block.2` | ReLU6 | `[1,64,8,8]` | `[1,64,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 50 | `backbone.features.5.block.3` | Conv2d | `[1,64,8,8]` | `[1,64,8,8]` | 逐通道空间特征提取；Conv 64→64，kernel=3×3，stride=1×1，dilation=1×1，groups=64 | 576 | 0 | 36,864 | 73,728 |
| 51 | `backbone.features.5.block.4` | BatchNorm2d | `[1,64,8,8]` | `[1,64,8,8]` | 批归一化，稳定特征分布；affine=True | 128 | 0 | — | — |
| 52 | `backbone.features.5.block.5` | ReLU6 | `[1,64,8,8]` | `[1,64,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 53 | `backbone.features.5.block.6` | Conv2d | `[1,64,8,8]` | `[1,48,8,8]` | 通道投影与融合；Conv 64→48，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 3,072 | 0 | 196,608 | 393,216 |
| 54 | `backbone.features.5.block.7` | BatchNorm2d | `[1,48,8,8]` | `[1,48,8,8]` | 批归一化，稳定特征分布；affine=True | 96 | 0 | — | — |
| 55 | `backbone.features.6` | InvertedResidual | `[1,48,8,8]` | `[1,48,8,8]` | MobileNetV2 倒残差块：扩展、深度卷积、线性投影；带残差相加；计算量由子层统计 | 0 | 0 | — | — |
| 56 | `backbone.features.6.block.0` | Conv2d | `[1,48,8,8]` | `[1,96,8,8]` | 通道投影与融合；Conv 48→96，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 4,608 | 0 | 294,912 | 589,824 |
| 57 | `backbone.features.6.block.1` | BatchNorm2d | `[1,96,8,8]` | `[1,96,8,8]` | 批归一化，稳定特征分布；affine=True | 192 | 0 | — | — |
| 58 | `backbone.features.6.block.2` | ReLU6 | `[1,96,8,8]` | `[1,96,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 59 | `backbone.features.6.block.3` | Conv2d | `[1,96,8,8]` | `[1,96,8,8]` | 逐通道空间特征提取；Conv 96→96，kernel=3×3，stride=1×1，dilation=1×1，groups=96 | 864 | 0 | 55,296 | 110,592 |
| 60 | `backbone.features.6.block.4` | BatchNorm2d | `[1,96,8,8]` | `[1,96,8,8]` | 批归一化，稳定特征分布；affine=True | 192 | 0 | — | — |
| 61 | `backbone.features.6.block.5` | ReLU6 | `[1,96,8,8]` | `[1,96,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 62 | `backbone.features.6.block.6` | Conv2d | `[1,96,8,8]` | `[1,48,8,8]` | 通道投影与融合；Conv 96→48，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 4,608 | 0 | 294,912 | 589,824 |
| 63 | `backbone.features.6.block.7` | BatchNorm2d | `[1,48,8,8]` | `[1,48,8,8]` | 批归一化，稳定特征分布；affine=True | 96 | 0 | — | — |
| 64 | `backbone.features.7` | InvertedResidual | `[1,48,8,8]` | `[1,64,8,8]` | MobileNetV2 倒残差块：扩展、深度卷积、线性投影；不使用残差；计算量由子层统计 | 0 | 0 | — | — |
| 65 | `backbone.features.7.block.0` | Conv2d | `[1,48,8,8]` | `[1,96,8,8]` | 通道投影与融合；Conv 48→96，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 4,608 | 0 | 294,912 | 589,824 |
| 66 | `backbone.features.7.block.1` | BatchNorm2d | `[1,96,8,8]` | `[1,96,8,8]` | 批归一化，稳定特征分布；affine=True | 192 | 0 | — | — |
| 67 | `backbone.features.7.block.2` | ReLU6 | `[1,96,8,8]` | `[1,96,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 68 | `backbone.features.7.block.3` | Conv2d | `[1,96,8,8]` | `[1,96,8,8]` | 逐通道空间特征提取；Conv 96→96，kernel=3×3，stride=1×1，dilation=1×1，groups=96 | 864 | 0 | 55,296 | 110,592 |
| 69 | `backbone.features.7.block.4` | BatchNorm2d | `[1,96,8,8]` | `[1,96,8,8]` | 批归一化，稳定特征分布；affine=True | 192 | 0 | — | — |
| 70 | `backbone.features.7.block.5` | ReLU6 | `[1,96,8,8]` | `[1,96,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 71 | `backbone.features.7.block.6` | Conv2d | `[1,96,8,8]` | `[1,64,8,8]` | 通道投影与融合；Conv 96→64，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 6,144 | 0 | 393,216 | 786,432 |
| 72 | `backbone.features.7.block.7` | BatchNorm2d | `[1,64,8,8]` | `[1,64,8,8]` | 批归一化，稳定特征分布；affine=True | 128 | 0 | — | — |
| 73 | `backbone.adapter.0` | Conv2d | `[1,64,8,8]` | `[1,96,8,8]` | 通道投影与融合；Conv 64→96，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 6,144 | 0 | 393,216 | 786,432 |
| 74 | `backbone.adapter.1` | BatchNorm2d | `[1,96,8,8]` | `[1,96,8,8]` | 批归一化，稳定特征分布；affine=True | 192 | 0 | — | — |
| 75 | `backbone.adapter.2` | ReLU6 | `[1,96,8,8]` | `[1,96,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 76 | `backbone.coordinate_attention` | CoordinateAttention | `[1,96,8,8]` | `[1,96,8,8]` | 分别编码高度与宽度上下文并重标定特征；计算量由子层统计 | 0 | 0 | — | — |
| 77 | `backbone.coordinate_attention.shared_projection.0` | Conv2d | `[1,96,16,1]` | `[1,8,16,1]` | 通道投影与融合；Conv 96→8，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 768 | 0 | 12,288 | 24,576 |
| 78 | `backbone.coordinate_attention.shared_projection.1` | BatchNorm2d | `[1,8,16,1]` | `[1,8,16,1]` | 批归一化，稳定特征分布；affine=True | 16 | 0 | — | — |
| 79 | `backbone.coordinate_attention.shared_projection.2` | Hardswish | `[1,8,16,1]` | `[1,8,16,1]` | Hardswish 轻量平滑激活 | 0 | 0 | — | — |
| 80 | `backbone.coordinate_attention.height_projection` | Conv2d | `[1,8,8,1]` | `[1,96,8,1]` | 通道投影与融合；Conv 8→96，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 864 | 0 | 6,144 | 12,288 |
| 81 | `backbone.coordinate_attention.width_projection` | Conv2d | `[1,8,1,8]` | `[1,96,1,8]` | 通道投影与融合；Conv 8→96，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 864 | 0 | 6,144 | 12,288 |
| 82 | `backbone.descriptor_head.0` | Dropout | `[1,96,8,8]` | `[1,96,8,8]` | 训练时随机失活，p=0.12 | 0 | 0 | — | — |
| 83 | `backbone.descriptor_head.1` | Conv2d | `[1,96,8,8]` | `[1,192,1,1]` | 逐通道空间特征提取；Conv 96→192，kernel=8×8，stride=1×1，dilation=1×1，groups=96 | 12,288 | 0 | 12,288 | 24,576 |
| 84 | `backbone.descriptor_head.2` | Conv2d | `[1,192,1,1]` | `[1,256,1,1]` | 通道投影与融合；Conv 192→256，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 49,152 | 0 | 49,152 | 98,304 |
| 85 | `backbone.descriptor_head.3` | BatchNorm2d | `[1,256,1,1]` | `[1,256,1,1]` | 批归一化，稳定特征分布；affine=False | 0 | 0 | — | — |
| 86 | `hash_head` | ResidualHashHead | `[1,256]` | `[1,512]` | 浮点描述子投影及残差哈希变换；计算量由子层统计 | 1,024 | 1,024 | — | — |
| 87 | `hash_head.input_norm` | LayerNorm | `[1,256]` | `[1,256]` | 对末端维度 (256,) 做层归一化 | 512 | 512 | — | — |
| 88 | `hash_head.input_projection` | Linear | `[1,256]` | `[1,512]` | 全连接特征投影 256→512 | 131,072 | 131,072 | 131,072 | 262,144 |
| 89 | `hash_head.residual_norm` | LayerNorm | `[1,512]` | `[1,512]` | 对末端维度 (512,) 做层归一化 | 1,024 | 1,024 | — | — |
| 90 | `hash_head.expand` | Linear | `[1,512]` | `[1,1024]` | 全连接特征投影 512→1024 | 525,312 | 525,312 | 524,288 | 1,048,576 |
| 91 | `hash_head.activation` | GELU | `[1,1024]` | `[1,1024]` | GELU 平滑非线性激活 | 0 | 0 | — | — |
| 92 | `hash_head.dropout` | Dropout | `[1,1024]` | `[1,1024]` | 训练时随机失活，p=0.1 | 0 | 0 | — | — |
| 93 | `hash_head.contract` | Linear | `[1,1024]` | `[1,512]` | 全连接特征投影 1024→512 | 524,800 | 524,800 | 524,288 | 1,048,576 |
| 94 | `hash_head.output_norm` | LayerNorm | `[1,512]` | `[1,512]` | 对末端维度 (512,) 做层归一化 | 1,024 | 1,024 | — | — |
| 95 | `hash_head.output_projection` | Linear | `[1,512]` | `[1,512]` | 全连接特征投影 512→512 | 262,656 | 262,656 | 262,144 | 524,288 |
