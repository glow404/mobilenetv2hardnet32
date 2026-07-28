当前 HardNet/L2Net 大约有：
133.5 万个可训练参数
3900 万 MACs/patch
最后的 128→128, kernel=8 卷积约有 104.9 万参数，占总参数量约 79%

MobileHardNet 为 85,664 参数。
CPU下，batch 64 下从约 1.263 ms/patch 降到 0.783 ms/patch，约快 1.61×。
单 patch 下 MobileHardNet 在当前桌面 PyTorch CPU 反而略慢（2.84 ms 对 2.57 ms），原因是倒残差网络算子数量更多、单样本时调度开销占主导。实际会批量处理关键点。

1×32×32
→ 3×3 Conv，16通道
→ InvertedResidual，16→24
→ InvertedResidual，24→32，stride=2
→ InvertedResidual，32→48
→ InvertedResidual，48→64，stride=2
→ InvertedResidual，64→96
→ 8×8 Depthwise Conv
→ 1×1 Conv，96→128
→ Flatten
→ L2 Normalize

IR倒残差块：
输入
  │
  ├─ 1×1 Conv：通道扩展 t 倍
  ├─ BN + ReLU6
  ├─ 3×3 Depthwise Conv
  ├─ BN + ReLU6
  ├─ 1×1 Conv：线性压缩到输出通道
  ├─ BN，不加激活
  │
  └─ 当 stride=1 且输入输出通道相同时做残差相加



Input                     [B,   1, 32, 32]
Stem                      [B,  16, 32, 32]
IR(t=1, c=16)             [B,  16, 32, 32]
IR(t=2, c=24, s=2) ×2     [B,  24, 16, 16]
IR(t=2, c=32, s=2) ×2     [B,  32,  8,  8]
IR(t=2, c=48, s=1) ×2     [B,  48,  8,  8]
IR(t=2, c=64, s=1) ×1     [B,  64,  8,  8]
Adapter 1×1               [B,  96,  8,  8]
Spatial descriptor head   [B, 192,  1,  1]
Linear projection         [B, 128,  1,  1]
Flatten + normalize       [B, 128]


原 HardNet 的参数主要集中在最后一层：
128→128, kernel=8×8
1,048,576
约占原模型参数的 79%。

这层同时学习：
每个通道的空间模式；
不同通道之间的组合关系。

MobileHardNet 把它因式分解成：
8×8 Depthwise：分别学习每个通道的空间模式
1×1 Conv：学习通道之间的组合关系
本质上是加入了一个假设：
空间关系和通道关系可以先分开学习，再组合。


重点：
不使用 Global Average Pooling
全局平均池化会抹掉脊线结构在 patch 中的位置。当前使用 8×8 Depthwise Conv，能学习不同空间位置的权重。

可能存在问题的点：
85K 参数可能容量不足，网络太浅
后续可用一下结构
输入 [B, 1, 32, 32]

Stem
  3×3 Conv: 1 → 24

低层特征
  IR: 24 → 24, expansion=1, stride=1

中层特征
  IR: 24 → 32, expansion=4, stride=2
  IR: 32 → 32, expansion=4, stride=1

  IR: 32 → 48, expansion=4, stride=2
  IR: 48 → 48, expansion=4, stride=1

高层特征
  IR: 48 → 64, expansion=4, stride=1
  IR: 64 → 64, expansion=4, stride=1

  IR: 64 → 96, expansion=4, stride=1
  IR: 96 → 96, expansion=4, stride=1

描述子准备
  1×1 Conv: 96 → 128
  3×3 Full Conv: 128 → 128
  BN + ReLU6

空间描述子头
  8×8 Depthwise Conv: 128 → 1024
  depth multiplier=8
  1×1 Linear Conv: 1024 → 128
  Final BN
  Flatten
  L2 Normalize

参数量约为原 HardNet 的 44%

后续可以考虑以下技巧：
SE；
self-attention；
SiLU/Hard-Swish；
多分支特征融合；
蒸馏；
新的损失函数。
将原hardnet描述子作为教师，进行蒸馏？
