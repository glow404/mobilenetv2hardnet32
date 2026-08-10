# HardNet 模型结构报告

## 汇总

| 项目 | 数值 |
| --- | ---: |
| 架构标识 | `hardnet` |
| 模型类 | `HardNet` |
| 描述子维度 | 256 |
| 输入形状 | `[1,1,32,32]` |
| 输出形状 | `[1,256]` |
| 总参数量 | 2,384,032 |
| 可训练参数量 | 2,384,032 |
| Conv/Linear MACs / patch | 40,140,800 |
| 约算 FLOPs / patch | 80,281,600 |

> 统计输入为单个 32×32 灰度 patch。MACs 只统计 Conv/Linear 的乘加；FLOPs 按 1 MAC≈2 FLOPs 约算。归一化、激活、池化、插值、拼接、注意力逐元素运算和残差相加未计入 MACs，因此该数值用于统一比较主干计算量，不等同于设备实测延迟。复合层的参数只统计其直接持有参数，子层参数与 MACs在各自明细行统计，避免重复。

## PyTorch 模块树

```text
HardNet(
  (features): Sequential(
    (0): Sequential(
      (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (1): Sequential(
      (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (2): Sequential(
      (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (3): Sequential(
      (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (4): Sequential(
      (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (5): Sequential(
      (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (6): Dropout(p=0.12, inplace=False)
    (7): Conv2d(128, 256, kernel_size=(8, 8), stride=(1, 1), bias=False)
    (8): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
  )
)
```

## 逐层明细（按实际前向调用顺序）

| # | 层路径 | 类型 | 输入 | 输出 | 作用 | 参数量 | 可训练参数 | MACs/patch | FLOPs/patch≈ |
| ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `features.0.0` | Conv2d | `[1,1,32,32]` | `[1,32,32,32]` | 局部空间特征提取；Conv 1→32，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 288 | 288 | 294,912 | 589,824 |
| 2 | `features.0.1` | BatchNorm2d | `[1,32,32,32]` | `[1,32,32,32]` | 批归一化，稳定特征分布；affine=True | 64 | 64 | — | — |
| 3 | `features.0.2` | ReLU | `[1,32,32,32]` | `[1,32,32,32]` | ReLU 非线性激活 | 0 | 0 | — | — |
| 4 | `features.1.0` | Conv2d | `[1,32,32,32]` | `[1,32,32,32]` | 局部空间特征提取；Conv 32→32，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 9,216 | 9,216 | 9,437,184 | 18,874,368 |
| 5 | `features.1.1` | BatchNorm2d | `[1,32,32,32]` | `[1,32,32,32]` | 批归一化，稳定特征分布；affine=True | 64 | 64 | — | — |
| 6 | `features.1.2` | ReLU | `[1,32,32,32]` | `[1,32,32,32]` | ReLU 非线性激活 | 0 | 0 | — | — |
| 7 | `features.2.0` | Conv2d | `[1,32,32,32]` | `[1,64,16,16]` | 局部空间特征提取；Conv 32→64，kernel=3×3，stride=2×2，dilation=1×1，groups=1 | 18,432 | 18,432 | 4,718,592 | 9,437,184 |
| 8 | `features.2.1` | BatchNorm2d | `[1,64,16,16]` | `[1,64,16,16]` | 批归一化，稳定特征分布；affine=True | 128 | 128 | — | — |
| 9 | `features.2.2` | ReLU | `[1,64,16,16]` | `[1,64,16,16]` | ReLU 非线性激活 | 0 | 0 | — | — |
| 10 | `features.3.0` | Conv2d | `[1,64,16,16]` | `[1,64,16,16]` | 局部空间特征提取；Conv 64→64，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 36,864 | 36,864 | 9,437,184 | 18,874,368 |
| 11 | `features.3.1` | BatchNorm2d | `[1,64,16,16]` | `[1,64,16,16]` | 批归一化，稳定特征分布；affine=True | 128 | 128 | — | — |
| 12 | `features.3.2` | ReLU | `[1,64,16,16]` | `[1,64,16,16]` | ReLU 非线性激活 | 0 | 0 | — | — |
| 13 | `features.4.0` | Conv2d | `[1,64,16,16]` | `[1,128,8,8]` | 局部空间特征提取；Conv 64→128，kernel=3×3，stride=2×2，dilation=1×1，groups=1 | 73,728 | 73,728 | 4,718,592 | 9,437,184 |
| 14 | `features.4.1` | BatchNorm2d | `[1,128,8,8]` | `[1,128,8,8]` | 批归一化，稳定特征分布；affine=True | 256 | 256 | — | — |
| 15 | `features.4.2` | ReLU | `[1,128,8,8]` | `[1,128,8,8]` | ReLU 非线性激活 | 0 | 0 | — | — |
| 16 | `features.5.0` | Conv2d | `[1,128,8,8]` | `[1,128,8,8]` | 局部空间特征提取；Conv 128→128，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 147,456 | 147,456 | 9,437,184 | 18,874,368 |
| 17 | `features.5.1` | BatchNorm2d | `[1,128,8,8]` | `[1,128,8,8]` | 批归一化，稳定特征分布；affine=True | 256 | 256 | — | — |
| 18 | `features.5.2` | ReLU | `[1,128,8,8]` | `[1,128,8,8]` | ReLU 非线性激活 | 0 | 0 | — | — |
| 19 | `features.6` | Dropout | `[1,128,8,8]` | `[1,128,8,8]` | 训练时随机失活，p=0.12 | 0 | 0 | — | — |
| 20 | `features.7` | Conv2d | `[1,128,8,8]` | `[1,256,1,1]` | 可学习的全局空间投影；Conv 128→256，kernel=8×8，stride=1×1，dilation=1×1，groups=1 | 2,097,152 | 2,097,152 | 2,097,152 | 4,194,304 |
| 21 | `features.8` | BatchNorm2d | `[1,256,1,1]` | `[1,256,1,1]` | 批归一化，稳定特征分布；affine=False | 0 | 0 | — | — |
