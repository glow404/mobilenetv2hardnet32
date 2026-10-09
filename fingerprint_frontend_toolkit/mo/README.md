# 膜结构三维特征识别

## 1. 项目简介

本项目使用三维编码网络从连续多帧膜结构数据中提取128维特征，并完成膜类别识别。

训练阶段使用三维编码器和角度分类头联合训练。应用阶段不使用训练分类头，而是：

1. 使用训练好的编码器提取每个类别的原型特征；
2. 使用同一个编码器提取待识别样本特征；
3. 计算待识别特征与全部类别原型的余弦相似度；
4. 将相似度最高的类别作为预测结果。

## 2. 主要文件

- `dataset.py`：扫描类别和帧文件，构造16帧三维输入。
- `model.py`：定义 `Encoder3D` 三维编码网络。
- `arcface.py`：定义训练阶段使用的角度分类头。
- `train.py`：训练入口，生成 `best_model.pth`。
- `build_prototype.py`：从各类别模板数据生成 `prototype_db/*.npy`。
- `test.py`：加载模型和 `feature_db.pt`，执行相似度匹配。
- `pre/rename_1.py`、`pre/rename_2.py`：数据文件重命名辅助脚本。
- `pre/is_blank.py`：检查空文件或无法读取的 CSV。

## 3. 环境与运行目录

推荐使用 Python 3.9 或更高版本，主要依赖：

- PyTorch
- NumPy
- pandas
- Matplotlib

所有命令都建议在 `mo` 目录执行，确保模型和特征库的相对路径正确：

```powershell
cd C:\Users\ZYH\Desktop\harnet32\mo
```

检查运行环境：

```powershell
python -c "import torch, numpy, pandas, matplotlib; print('CUDA可用:', torch.cuda.is_available())"
```

缺少普通依赖时可以执行：

```powershell
pip install numpy pandas matplotlib
```

PyTorch 应根据当前处理器或显卡环境安装对应版本。

## 4. 数据目录和文件名要求

训练、模板构建和测试都使用相同的目录结构。`ROOT` 必须指向包含全部类别文件夹的
父目录，不能直接指向某一个类别文件夹：

```text
数据根目录/
├── GHM_ID1/
│   ├── GHM_ID1-pair_1-Rgd=1360.csv
│   ├── GHM_ID1-pair_1-Rgd=1361.csv
│   └── ...
├── GHM_ID2/
│   └── ...
└── GHM_ID11/
    └── ...
```

文件名必须包含：

```text
pair_数字 ... Rgd=数字
```

例如：

```text
GHM_ID11-pair_1-Rgd=1360.csv
```

当前数据加载规则：

- 每个类别文件夹代表一个膜类别；
- 同一个 `pair_数字` 组成一组样本；
- 每组必须恰好包含61个 CSV，否则会被跳过；
- 网络实际取按 `Rgd` 排序后的前16帧；
- 每个 CSV 使用 `0::10, 0::10` 进行行列降采样；
- 16帧整体执行减均值、除标准差。

如果原始文件名不符合要求，需要先修改 `pre/rename_1.py` 和
`pre/rename_2.py` 顶部的 `folder` 路径，再运行对应脚本。重命名不可撤销，
正式运行前应备份原始数据。

检查空文件时，先修改 `pre/is_blank.py` 中的 `root`：

```powershell
python pre\is_blank.py
```

## 5. 训练部分

### 5.1 修改训练参数

打开 `train.py`，至少确认以下参数：

```python
ROOT = r"D:\你的数据根目录"
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4
VAL_RATIO = 0.2
SAVE_AFTER_EPOCH = 10
```

`ROOT` 的直接子目录必须是类别文件夹。建议每个类别至少提供5组有效 pair，
否则按20%划分后可能没有验证样本。

### 5.2 启动训练

```powershell
python train.py
```

程序会自动优先使用 CUDA，没有可用显卡时使用处理器。

训练过程中会输出每轮训练损失、验证损失和验证准确率。满足以下条件时，
模型保存到当前目录的 `best_model.pth`：

- 已达到第10轮，具体由 `SAVE_AFTER_EPOCH` 控制；
- 验证准确率大于0.9；
- 当前准确率高于此前最佳值。

如果训练结束后没有生成 `best_model.pth`，应检查有效样本数量以及验证准确率，
而不是直接进入应用阶段。

## 6. 应用部分

应用流程分为“构建类别原型库”和“执行测试识别”两步。

### 6.1 构建类别原型

打开 `build_prototype.py`，修改：

```python
ROOT = r"D:\你的模板数据根目录"
MODEL_PATH = "best_model.pth"
SAVE_DIR = "prototype_db"
NUM_FRAMES = 16
```

然后执行：

```powershell
python build_prototype.py
```

脚本会在每个类别中选择第一组包含61帧的完整 pair，取前16帧生成原型，并输出：

```text
prototype_db/
├── 0.npy
├── 1.npy
└── ...
```

数字类别编号由 `ROOT` 下类别文件夹名称排序后生成。

### 6.2 生成测试程序需要的特征库

当前 `build_prototype.py` 输出多个 `.npy`，而 `test.py` 读取
`feature_db.pt`，因此重新构建原型后还需要执行一次格式转换：

```powershell
@'
from pathlib import Path
import numpy as np
import torch

feature_db = {}
for path in Path("prototype_db").glob("*.npy"):
    feature = torch.from_numpy(np.load(path)).float().reshape(-1)
    feature = feature / (feature.norm() + 1e-12)
    feature_db[int(path.stem)] = feature

if not feature_db:
    raise RuntimeError("prototype_db 中没有找到 .npy 原型")

torch.save(feature_db, "feature_db.pt")
print("已生成 feature_db.pt，类别数:", len(feature_db))
'@ | python -
```

如果直接使用项目中已有的 `feature_db.pt`，可以跳过原型构建和格式转换。

### 6.3 执行测试识别

打开 `test.py`，将 `ROOT` 修改为测试数据的父目录：

```python
ROOT = r"D:\你的测试数据根目录"
```

同时确认当前目录存在：

```text
best_model.pth
feature_db.pt
```

启动测试：

```powershell
python test.py
```

测试程序会：

- 输出每个样本与全部类别原型的相似度；
- 输出预测类别和最大相似度；
- 统计最高相似度超过0.97的高置信度结果；
- 输出总体最高相似度的均值、标准差和分位数；
- 显示相似度直方图。

高置信度阈值目前直接写在 `test.py` 中：

```python
if best_score > 0.97:
```

需要调整时直接修改该数值。测试集类别文件夹的名称和排序应与构建原型库时一致，
否则数字类别编号和准确率统计可能不对应。

## 7. 完整运行顺序

首次训练并应用：

```powershell
cd C:\Users\ZYH\Desktop\harnet32\mo
python train.py
python build_prototype.py
# 按第6.2节生成 feature_db.pt
python test.py
```

已有模型和特征库时，直接应用：

```powershell
cd C:\Users\ZYH\Desktop\harnet32\mo
python test.py
```
