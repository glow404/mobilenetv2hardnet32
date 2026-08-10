# MobileHardNet 模型结构报告

## 汇总

| 项目 | 数值 |
| --- | ---: |
| 架构标识 | `mobile_hardnet` |
| 模型类 | `MobileHardNet` |
| 描述子维度 | 256 |
| 输入形状 | `[1,1,32,32]` |
| 输出形状 | `[1,256]` |
| 总参数量 | 112,752 |
| 可训练参数量 | 112,752 |
| Conv/Linear MACs / patch | 5,004,288 |
| 约算 FLOPs / patch | 10,008,576 |

> 统计输入为单个 32×32 灰度 patch。MACs 只统计 Conv/Linear 的乘加；FLOPs 按 1 MAC≈2 FLOPs 约算。归一化、激活、池化、插值、拼接、注意力逐元素运算和残差相加未计入 MACs，因此该数值用于统一比较主干计算量，不等同于设备实测延迟。复合层的参数只统计其直接持有参数，子层参数与 MACs在各自明细行统计，避免重复。

## PyTorch 模块树

```text
MobileHardNet(
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
```

## 逐层明细（按实际前向调用顺序）

| # | 层路径 | 类型 | 输入 | 输出 | 作用 | 参数量 | 可训练参数 | MACs/patch | FLOPs/patch≈ |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `stem.0` | Conv2d | `[1,1,32,32]` | `[1,16,32,32]` | 局部空间特征提取；Conv 1→16，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 144 | 144 | 147,456 | 294,912 |
| 2 | `stem.1` | BatchNorm2d | `[1,16,32,32]` | `[1,16,32,32]` | 批归一化，稳定特征分布；affine=True | 32 | 32 | — | — |
| 3 | `stem.2` | ReLU6 | `[1,16,32,32]` | `[1,16,32,32]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 4 | `features.0` | InvertedResidual | `[1,16,32,32]` | `[1,16,32,32]` | MobileNetV2 倒残差块：扩展、深度卷积、线性投影；带残差相加；计算量由子层统计 | 0 | 0 | — | — |
| 5 | `features.0.block.0` | Conv2d | `[1,16,32,32]` | `[1,16,32,32]` | 逐通道空间特征提取；Conv 16→16，kernel=3×3，stride=1×1，dilation=1×1，groups=16 | 144 | 144 | 147,456 | 294,912 |
| 6 | `features.0.block.1` | BatchNorm2d | `[1,16,32,32]` | `[1,16,32,32]` | 批归一化，稳定特征分布；affine=True | 32 | 32 | — | — |
| 7 | `features.0.block.2` | ReLU6 | `[1,16,32,32]` | `[1,16,32,32]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 8 | `features.0.block.3` | Conv2d | `[1,16,32,32]` | `[1,16,32,32]` | 通道投影与融合；Conv 16→16，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 256 | 256 | 262,144 | 524,288 |
| 9 | `features.0.block.4` | BatchNorm2d | `[1,16,32,32]` | `[1,16,32,32]` | 批归一化，稳定特征分布；affine=True | 32 | 32 | — | — |
| 10 | `features.1` | InvertedResidual | `[1,16,32,32]` | `[1,24,16,16]` | MobileNetV2 倒残差块：扩展、深度卷积、线性投影；不使用残差；计算量由子层统计 | 0 | 0 | — | — |
| 11 | `features.1.block.0` | Conv2d | `[1,16,32,32]` | `[1,32,32,32]` | 通道投影与融合；Conv 16→32，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 512 | 512 | 524,288 | 1,048,576 |
| 12 | `features.1.block.1` | BatchNorm2d | `[1,32,32,32]` | `[1,32,32,32]` | 批归一化，稳定特征分布；affine=True | 64 | 64 | — | — |
| 13 | `features.1.block.2` | ReLU6 | `[1,32,32,32]` | `[1,32,32,32]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 14 | `features.1.block.3` | Conv2d | `[1,32,32,32]` | `[1,32,16,16]` | 逐通道空间特征提取；Conv 32→32，kernel=3×3，stride=2×2，dilation=1×1，groups=32 | 288 | 288 | 73,728 | 147,456 |
| 15 | `features.1.block.4` | BatchNorm2d | `[1,32,16,16]` | `[1,32,16,16]` | 批归一化，稳定特征分布；affine=True | 64 | 64 | — | — |
| 16 | `features.1.block.5` | ReLU6 | `[1,32,16,16]` | `[1,32,16,16]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 17 | `features.1.block.6` | Conv2d | `[1,32,16,16]` | `[1,24,16,16]` | 通道投影与融合；Conv 32→24，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 768 | 768 | 196,608 | 393,216 |
| 18 | `features.1.block.7` | BatchNorm2d | `[1,24,16,16]` | `[1,24,16,16]` | 批归一化，稳定特征分布；affine=True | 48 | 48 | — | — |
| 19 | `features.2` | InvertedResidual | `[1,24,16,16]` | `[1,24,16,16]` | MobileNetV2 倒残差块：扩展、深度卷积、线性投影；带残差相加；计算量由子层统计 | 0 | 0 | — | — |
| 20 | `features.2.block.0` | Conv2d | `[1,24,16,16]` | `[1,48,16,16]` | 通道投影与融合；Conv 24→48，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 1,152 | 1,152 | 294,912 | 589,824 |
| 21 | `features.2.block.1` | BatchNorm2d | `[1,48,16,16]` | `[1,48,16,16]` | 批归一化，稳定特征分布；affine=True | 96 | 96 | — | — |
| 22 | `features.2.block.2` | ReLU6 | `[1,48,16,16]` | `[1,48,16,16]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 23 | `features.2.block.3` | Conv2d | `[1,48,16,16]` | `[1,48,16,16]` | 逐通道空间特征提取；Conv 48→48，kernel=3×3，stride=1×1，dilation=1×1，groups=48 | 432 | 432 | 110,592 | 221,184 |
| 24 | `features.2.block.4` | BatchNorm2d | `[1,48,16,16]` | `[1,48,16,16]` | 批归一化，稳定特征分布；affine=True | 96 | 96 | — | — |
| 25 | `features.2.block.5` | ReLU6 | `[1,48,16,16]` | `[1,48,16,16]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 26 | `features.2.block.6` | Conv2d | `[1,48,16,16]` | `[1,24,16,16]` | 通道投影与融合；Conv 48→24，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 1,152 | 1,152 | 294,912 | 589,824 |
| 27 | `features.2.block.7` | BatchNorm2d | `[1,24,16,16]` | `[1,24,16,16]` | 批归一化，稳定特征分布；affine=True | 48 | 48 | — | — |
| 28 | `features.3` | InvertedResidual | `[1,24,16,16]` | `[1,32,8,8]` | MobileNetV2 倒残差块：扩展、深度卷积、线性投影；不使用残差；计算量由子层统计 | 0 | 0 | — | — |
| 29 | `features.3.block.0` | Conv2d | `[1,24,16,16]` | `[1,48,16,16]` | 通道投影与融合；Conv 24→48，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 1,152 | 1,152 | 294,912 | 589,824 |
| 30 | `features.3.block.1` | BatchNorm2d | `[1,48,16,16]` | `[1,48,16,16]` | 批归一化，稳定特征分布；affine=True | 96 | 96 | — | — |
| 31 | `features.3.block.2` | ReLU6 | `[1,48,16,16]` | `[1,48,16,16]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 32 | `features.3.block.3` | Conv2d | `[1,48,16,16]` | `[1,48,8,8]` | 逐通道空间特征提取；Conv 48→48，kernel=3×3，stride=2×2，dilation=1×1，groups=48 | 432 | 432 | 27,648 | 55,296 |
| 33 | `features.3.block.4` | BatchNorm2d | `[1,48,8,8]` | `[1,48,8,8]` | 批归一化，稳定特征分布；affine=True | 96 | 96 | — | — |
| 34 | `features.3.block.5` | ReLU6 | `[1,48,8,8]` | `[1,48,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 35 | `features.3.block.6` | Conv2d | `[1,48,8,8]` | `[1,32,8,8]` | 通道投影与融合；Conv 48→32，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 1,536 | 1,536 | 98,304 | 196,608 |
| 36 | `features.3.block.7` | BatchNorm2d | `[1,32,8,8]` | `[1,32,8,8]` | 批归一化，稳定特征分布；affine=True | 64 | 64 | — | — |
| 37 | `features.4` | InvertedResidual | `[1,32,8,8]` | `[1,32,8,8]` | MobileNetV2 倒残差块：扩展、深度卷积、线性投影；带残差相加；计算量由子层统计 | 0 | 0 | — | — |
| 38 | `features.4.block.0` | Conv2d | `[1,32,8,8]` | `[1,64,8,8]` | 通道投影与融合；Conv 32→64，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 2,048 | 2,048 | 131,072 | 262,144 |
| 39 | `features.4.block.1` | BatchNorm2d | `[1,64,8,8]` | `[1,64,8,8]` | 批归一化，稳定特征分布；affine=True | 128 | 128 | — | — |
| 40 | `features.4.block.2` | ReLU6 | `[1,64,8,8]` | `[1,64,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 41 | `features.4.block.3` | Conv2d | `[1,64,8,8]` | `[1,64,8,8]` | 逐通道空间特征提取；Conv 64→64，kernel=3×3，stride=1×1，dilation=1×1，groups=64 | 576 | 576 | 36,864 | 73,728 |
| 42 | `features.4.block.4` | BatchNorm2d | `[1,64,8,8]` | `[1,64,8,8]` | 批归一化，稳定特征分布；affine=True | 128 | 128 | — | — |
| 43 | `features.4.block.5` | ReLU6 | `[1,64,8,8]` | `[1,64,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 44 | `features.4.block.6` | Conv2d | `[1,64,8,8]` | `[1,32,8,8]` | 通道投影与融合；Conv 64→32，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 2,048 | 2,048 | 131,072 | 262,144 |
| 45 | `features.4.block.7` | BatchNorm2d | `[1,32,8,8]` | `[1,32,8,8]` | 批归一化，稳定特征分布；affine=True | 64 | 64 | — | — |
| 46 | `features.5` | InvertedResidual | `[1,32,8,8]` | `[1,48,8,8]` | MobileNetV2 倒残差块：扩展、深度卷积、线性投影；不使用残差；计算量由子层统计 | 0 | 0 | — | — |
| 47 | `features.5.block.0` | Conv2d | `[1,32,8,8]` | `[1,64,8,8]` | 通道投影与融合；Conv 32→64，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 2,048 | 2,048 | 131,072 | 262,144 |
| 48 | `features.5.block.1` | BatchNorm2d | `[1,64,8,8]` | `[1,64,8,8]` | 批归一化，稳定特征分布；affine=True | 128 | 128 | — | — |
| 49 | `features.5.block.2` | ReLU6 | `[1,64,8,8]` | `[1,64,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 50 | `features.5.block.3` | Conv2d | `[1,64,8,8]` | `[1,64,8,8]` | 逐通道空间特征提取；Conv 64→64，kernel=3×3，stride=1×1，dilation=1×1，groups=64 | 576 | 576 | 36,864 | 73,728 |
| 51 | `features.5.block.4` | BatchNorm2d | `[1,64,8,8]` | `[1,64,8,8]` | 批归一化，稳定特征分布；affine=True | 128 | 128 | — | — |
| 52 | `features.5.block.5` | ReLU6 | `[1,64,8,8]` | `[1,64,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 53 | `features.5.block.6` | Conv2d | `[1,64,8,8]` | `[1,48,8,8]` | 通道投影与融合；Conv 64→48，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 3,072 | 3,072 | 196,608 | 393,216 |
| 54 | `features.5.block.7` | BatchNorm2d | `[1,48,8,8]` | `[1,48,8,8]` | 批归一化，稳定特征分布；affine=True | 96 | 96 | — | — |
| 55 | `features.6` | InvertedResidual | `[1,48,8,8]` | `[1,48,8,8]` | MobileNetV2 倒残差块：扩展、深度卷积、线性投影；带残差相加；计算量由子层统计 | 0 | 0 | — | — |
| 56 | `features.6.block.0` | Conv2d | `[1,48,8,8]` | `[1,96,8,8]` | 通道投影与融合；Conv 48→96，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 4,608 | 4,608 | 294,912 | 589,824 |
| 57 | `features.6.block.1` | BatchNorm2d | `[1,96,8,8]` | `[1,96,8,8]` | 批归一化，稳定特征分布；affine=True | 192 | 192 | — | — |
| 58 | `features.6.block.2` | ReLU6 | `[1,96,8,8]` | `[1,96,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 59 | `features.6.block.3` | Conv2d | `[1,96,8,8]` | `[1,96,8,8]` | 逐通道空间特征提取；Conv 96→96，kernel=3×3，stride=1×1，dilation=1×1，groups=96 | 864 | 864 | 55,296 | 110,592 |
| 60 | `features.6.block.4` | BatchNorm2d | `[1,96,8,8]` | `[1,96,8,8]` | 批归一化，稳定特征分布；affine=True | 192 | 192 | — | — |
| 61 | `features.6.block.5` | ReLU6 | `[1,96,8,8]` | `[1,96,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 62 | `features.6.block.6` | Conv2d | `[1,96,8,8]` | `[1,48,8,8]` | 通道投影与融合；Conv 96→48，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 4,608 | 4,608 | 294,912 | 589,824 |
| 63 | `features.6.block.7` | BatchNorm2d | `[1,48,8,8]` | `[1,48,8,8]` | 批归一化，稳定特征分布；affine=True | 96 | 96 | — | — |
| 64 | `features.7` | InvertedResidual | `[1,48,8,8]` | `[1,64,8,8]` | MobileNetV2 倒残差块：扩展、深度卷积、线性投影；不使用残差；计算量由子层统计 | 0 | 0 | — | — |
| 65 | `features.7.block.0` | Conv2d | `[1,48,8,8]` | `[1,96,8,8]` | 通道投影与融合；Conv 48→96，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 4,608 | 4,608 | 294,912 | 589,824 |
| 66 | `features.7.block.1` | BatchNorm2d | `[1,96,8,8]` | `[1,96,8,8]` | 批归一化，稳定特征分布；affine=True | 192 | 192 | — | — |
| 67 | `features.7.block.2` | ReLU6 | `[1,96,8,8]` | `[1,96,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 68 | `features.7.block.3` | Conv2d | `[1,96,8,8]` | `[1,96,8,8]` | 逐通道空间特征提取；Conv 96→96，kernel=3×3，stride=1×1，dilation=1×1，groups=96 | 864 | 864 | 55,296 | 110,592 |
| 69 | `features.7.block.4` | BatchNorm2d | `[1,96,8,8]` | `[1,96,8,8]` | 批归一化，稳定特征分布；affine=True | 192 | 192 | — | — |
| 70 | `features.7.block.5` | ReLU6 | `[1,96,8,8]` | `[1,96,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 71 | `features.7.block.6` | Conv2d | `[1,96,8,8]` | `[1,64,8,8]` | 通道投影与融合；Conv 96→64，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 6,144 | 6,144 | 393,216 | 786,432 |
| 72 | `features.7.block.7` | BatchNorm2d | `[1,64,8,8]` | `[1,64,8,8]` | 批归一化，稳定特征分布；affine=True | 128 | 128 | — | — |
| 73 | `adapter.0` | Conv2d | `[1,64,8,8]` | `[1,96,8,8]` | 通道投影与融合；Conv 64→96，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 6,144 | 6,144 | 393,216 | 786,432 |
| 74 | `adapter.1` | BatchNorm2d | `[1,96,8,8]` | `[1,96,8,8]` | 批归一化，稳定特征分布；affine=True | 192 | 192 | — | — |
| 75 | `adapter.2` | ReLU6 | `[1,96,8,8]` | `[1,96,8,8]` | ReLU6 非线性激活，限制移动网络激活范围 | 0 | 0 | — | — |
| 76 | `coordinate_attention` | CoordinateAttention | `[1,96,8,8]` | `[1,96,8,8]` | 分别编码高度与宽度上下文并重标定特征；计算量由子层统计 | 0 | 0 | — | — |
| 77 | `coordinate_attention.shared_projection.0` | Conv2d | `[1,96,16,1]` | `[1,8,16,1]` | 通道投影与融合；Conv 96→8，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 768 | 768 | 12,288 | 24,576 |
| 78 | `coordinate_attention.shared_projection.1` | BatchNorm2d | `[1,8,16,1]` | `[1,8,16,1]` | 批归一化，稳定特征分布；affine=True | 16 | 16 | — | — |
| 79 | `coordinate_attention.shared_projection.2` | Hardswish | `[1,8,16,1]` | `[1,8,16,1]` | Hardswish 轻量平滑激活 | 0 | 0 | — | — |
| 80 | `coordinate_attention.height_projection` | Conv2d | `[1,8,8,1]` | `[1,96,8,1]` | 通道投影与融合；Conv 8→96，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 864 | 864 | 6,144 | 12,288 |
| 81 | `coordinate_attention.width_projection` | Conv2d | `[1,8,1,8]` | `[1,96,1,8]` | 通道投影与融合；Conv 8→96，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 864 | 864 | 6,144 | 12,288 |
| 82 | `descriptor_head.0` | Dropout | `[1,96,8,8]` | `[1,96,8,8]` | 训练时随机失活，p=0.12 | 0 | 0 | — | — |
| 83 | `descriptor_head.1` | Conv2d | `[1,96,8,8]` | `[1,192,1,1]` | 逐通道空间特征提取；Conv 96→192，kernel=8×8，stride=1×1，dilation=1×1，groups=96 | 12,288 | 12,288 | 12,288 | 24,576 |
| 84 | `descriptor_head.2` | Conv2d | `[1,192,1,1]` | `[1,256,1,1]` | 通道投影与融合；Conv 192→256，kernel=1×1，stride=1×1，dilation=1×1，groups=1 | 49,152 | 49,152 | 49,152 | 98,304 |
| 85 | `descriptor_head.3` | BatchNorm2d | `[1,256,1,1]` | `[1,256,1,1]` | 批归一化，稳定特征分布；affine=False | 0 | 0 | — | — |
