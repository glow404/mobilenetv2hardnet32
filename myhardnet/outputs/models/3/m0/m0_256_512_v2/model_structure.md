# BinaryDescriptorModel 模型结构报告

## 汇总

| 项目 | 数值 |
| --- | ---: |
| 架构标识 | `residual_binary_hash_v1` |
| 模型类 | `BinaryDescriptorModel` |
| 描述子维度 | 512 |
| 输入形状 | `[1,1,32,32]` |
| 输出形状 | `[1,512]` |
| 总参数量 | 3,831,456 |
| 可训练参数量 | 1,447,424 |
| Conv/Linear MACs / patch | 41,582,592 |
| 约算 FLOPs / patch | 83,165,184 |

> 统计输入为单个 32×32 灰度 patch。MACs 只统计 Conv/Linear 的乘加；FLOPs 按 1 MAC≈2 FLOPs 约算。归一化、激活、池化、插值、拼接、注意力逐元素运算和残差相加未计入 MACs，因此该数值用于统一比较主干计算量，不等同于设备实测延迟。复合层的参数只统计其直接持有参数，子层参数与 MACs在各自明细行统计，避免重复。

## PyTorch 模块树

```text
BinaryDescriptorModel(
  (backbone): HardNet(
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
| 1 | `backbone.features.0.0` | Conv2d | `[1,1,32,32]` | `[1,32,32,32]` | 局部空间特征提取；Conv 1→32，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 288 | 0 | 294,912 | 589,824 |
| 2 | `backbone.features.0.1` | BatchNorm2d | `[1,32,32,32]` | `[1,32,32,32]` | 批归一化，稳定特征分布；affine=True | 64 | 0 | — | — |
| 3 | `backbone.features.0.2` | ReLU | `[1,32,32,32]` | `[1,32,32,32]` | ReLU 非线性激活 | 0 | 0 | — | — |
| 4 | `backbone.features.1.0` | Conv2d | `[1,32,32,32]` | `[1,32,32,32]` | 局部空间特征提取；Conv 32→32，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 9,216 | 0 | 9,437,184 | 18,874,368 |
| 5 | `backbone.features.1.1` | BatchNorm2d | `[1,32,32,32]` | `[1,32,32,32]` | 批归一化，稳定特征分布；affine=True | 64 | 0 | — | — |
| 6 | `backbone.features.1.2` | ReLU | `[1,32,32,32]` | `[1,32,32,32]` | ReLU 非线性激活 | 0 | 0 | — | — |
| 7 | `backbone.features.2.0` | Conv2d | `[1,32,32,32]` | `[1,64,16,16]` | 局部空间特征提取；Conv 32→64，kernel=3×3，stride=2×2，dilation=1×1，groups=1 | 18,432 | 0 | 4,718,592 | 9,437,184 |
| 8 | `backbone.features.2.1` | BatchNorm2d | `[1,64,16,16]` | `[1,64,16,16]` | 批归一化，稳定特征分布；affine=True | 128 | 0 | — | — |
| 9 | `backbone.features.2.2` | ReLU | `[1,64,16,16]` | `[1,64,16,16]` | ReLU 非线性激活 | 0 | 0 | — | — |
| 10 | `backbone.features.3.0` | Conv2d | `[1,64,16,16]` | `[1,64,16,16]` | 局部空间特征提取；Conv 64→64，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 36,864 | 0 | 9,437,184 | 18,874,368 |
| 11 | `backbone.features.3.1` | BatchNorm2d | `[1,64,16,16]` | `[1,64,16,16]` | 批归一化，稳定特征分布；affine=True | 128 | 0 | — | — |
| 12 | `backbone.features.3.2` | ReLU | `[1,64,16,16]` | `[1,64,16,16]` | ReLU 非线性激活 | 0 | 0 | — | — |
| 13 | `backbone.features.4.0` | Conv2d | `[1,64,16,16]` | `[1,128,8,8]` | 局部空间特征提取；Conv 64→128，kernel=3×3，stride=2×2，dilation=1×1，groups=1 | 73,728 | 0 | 4,718,592 | 9,437,184 |
| 14 | `backbone.features.4.1` | BatchNorm2d | `[1,128,8,8]` | `[1,128,8,8]` | 批归一化，稳定特征分布；affine=True | 256 | 0 | — | — |
| 15 | `backbone.features.4.2` | ReLU | `[1,128,8,8]` | `[1,128,8,8]` | ReLU 非线性激活 | 0 | 0 | — | — |
| 16 | `backbone.features.5.0` | Conv2d | `[1,128,8,8]` | `[1,128,8,8]` | 局部空间特征提取；Conv 128→128，kernel=3×3，stride=1×1，dilation=1×1，groups=1 | 147,456 | 0 | 9,437,184 | 18,874,368 |
| 17 | `backbone.features.5.1` | BatchNorm2d | `[1,128,8,8]` | `[1,128,8,8]` | 批归一化，稳定特征分布；affine=True | 256 | 0 | — | — |
| 18 | `backbone.features.5.2` | ReLU | `[1,128,8,8]` | `[1,128,8,8]` | ReLU 非线性激活 | 0 | 0 | — | — |
| 19 | `backbone.features.6` | Dropout | `[1,128,8,8]` | `[1,128,8,8]` | 训练时随机失活，p=0.12 | 0 | 0 | — | — |
| 20 | `backbone.features.7` | Conv2d | `[1,128,8,8]` | `[1,256,1,1]` | 可学习的全局空间投影；Conv 128→256，kernel=8×8，stride=1×1，dilation=1×1，groups=1 | 2,097,152 | 0 | 2,097,152 | 4,194,304 |
| 21 | `backbone.features.8` | BatchNorm2d | `[1,256,1,1]` | `[1,256,1,1]` | 批归一化，稳定特征分布；affine=False | 0 | 0 | — | — |
| 22 | `hash_head` | ResidualHashHead | `[1,256]` | `[1,512]` | 浮点描述子投影及残差哈希变换；计算量由子层统计 | 1,024 | 1,024 | — | — |
| 23 | `hash_head.input_norm` | LayerNorm | `[1,256]` | `[1,256]` | 对末端维度 (256,) 做层归一化 | 512 | 512 | — | — |
| 24 | `hash_head.input_projection` | Linear | `[1,256]` | `[1,512]` | 全连接特征投影 256→512 | 131,072 | 131,072 | 131,072 | 262,144 |
| 25 | `hash_head.residual_norm` | LayerNorm | `[1,512]` | `[1,512]` | 对末端维度 (512,) 做层归一化 | 1,024 | 1,024 | — | — |
| 26 | `hash_head.expand` | Linear | `[1,512]` | `[1,1024]` | 全连接特征投影 512→1024 | 525,312 | 525,312 | 524,288 | 1,048,576 |
| 27 | `hash_head.activation` | GELU | `[1,1024]` | `[1,1024]` | GELU 平滑非线性激活 | 0 | 0 | — | — |
| 28 | `hash_head.dropout` | Dropout | `[1,1024]` | `[1,1024]` | 训练时随机失活，p=0.1 | 0 | 0 | — | — |
| 29 | `hash_head.contract` | Linear | `[1,1024]` | `[1,512]` | 全连接特征投影 1024→512 | 524,800 | 524,800 | 524,288 | 1,048,576 |
| 30 | `hash_head.output_norm` | LayerNorm | `[1,512]` | `[1,512]` | 对末端维度 (512,) 做层归一化 | 1,024 | 1,024 | — | — |
| 31 | `hash_head.output_projection` | Linear | `[1,512]` | `[1,512]` | 全连接特征投影 512→512 | 262,656 | 262,656 | 262,144 | 524,288 |
