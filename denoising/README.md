# 湿指纹图像去噪

## 1. 项目简介

本项目使用轻量 U-Net 对湿指纹或含噪指纹图像进行恢复。训练时从干净指纹图像
动态合成噪声和模糊，以原图作为监督目标。

除了像素和梯度损失，训练还使用冻结的 HardNet 比较去噪结果与原图局部描述子，
使网络尽量保留后续指纹匹配需要的纹理结构。

## 2. 方法流程

```text
干净指纹图像
      ↓
恢复后的指纹图像
      ↓
像素损失 + 梯度损失 + HardNet描述子损失
```

训练损失：

```text
总损失 = L1损失 + 0.2 × 梯度损失 + 0.15 × HardNet描述子损失
```

当前推荐主流程：

```text
train_1.py
    ↓
model_Unet_V6_best.pth
    ↓
test.py 或 test_snr.py
```

## 3. 主要文件

- `train_1.py`：当前 V6 训练入口，使用 PNG 原图，并缓存原图的 SIFT 关键点。
- `test.py`：对真实输入目录递归执行去噪并保存结果。
- `test_snr.py`：给干净图像添加指定强度的高斯噪声，比较去噪前后的信噪比。
- `train.py`：较早的 BMP 训练版本，输出 V3 最佳模型。
- `train_stable_patch.py`：使用固定网格局部块计算 HardNet 损失的实验版本。
- `runtime.py`：SIFT 关键点检测和 HardNet 局部块提取工具。
- `HardNet.py`：HardNet 网络定义。
- `hardnet.pt`：训练去噪网络时使用的冻结 HardNet 权重。
- `config_match_new.yaml`：`train_1.py` 提取 SIFT 关键点和局部块时使用的配置。
- `laplas.py`：可选的拉普拉斯锐化脚本，不属于 U-Net 主流程。

## 4. 环境与运行目录

推荐使用 Python 3.9 或更高版本，依赖 PyTorch、OpenCV、NumPy、Pillow 和 PyYAML。

安装普通依赖：

```powershell
python -m pip install numpy opencv-python pillow pyyaml
```

PyTorch 应根据当前处理器或显卡环境安装对应版本。安装完成后检查：

```powershell
python -c "import torch, cv2, numpy, PIL, yaml; print('CUDA可用:', torch.cuda.is_available())"
```

训练脚本同时使用 `denoising.HardNet` 和当前目录下的 `runtime.py`，因此建议从
`denoising` 目录运行，并把项目根目录加入模块搜索路径：

```powershell
cd C:\Users\ZYH\Desktop\harnet32\denoising
$env:PYTHONPATH="..;$env:PYTHONPATH"
```

同一个终端后续训练和测试不需要重复设置。

## 5. 训练部分

### 5.1 准备训练数据

当前推荐的 `train_1.py` 只读取训练目录第一层中的 `.png` 灰度图：

```text
你的训练目录/
├── image_0001.png
├── image_0002.png
└── ...
```

图像推荐保持为100×110。网络会先反射填充到128×128，处理后再裁回原尺寸；
输入高度或宽度不能超过128。

训练过程中会动态执行：

- 高斯噪声：标准差随机取 `0.01～0.15`；
- 高斯模糊：70%概率启用，半径随机取 `0.2～1.5`；
- 下采样退化和 JPEG 退化当前默认关闭；
- 图像灰度归一化到 `[0,1]`。

### 5.2 修改训练路径

打开 `train_1.py`，将 `train_target` 改为实际 PNG 训练目录：

```python
train_target = r"D:\你的干净指纹PNG目录"
```

同时确认 `denoising` 目录中存在：

```text
hardnet.pt
config_match_new.yaml
```

`config_match_new.yaml` 当前主要提供 `sift`、`patch` 和 `keypoint_filter` 参数；
冻结的 HardNet 权重实际由 `train_1.py` 中的 `hardnet_checkpoint="hardnet.pt"` 指定。

### 5.3 当前训练参数

`train_1.py` 中的主要参数为：

- 训练轮数：200；
- 批量大小：16；
- 学习率：0.001；
- 训练集比例：80%；
- 验证集比例：20%；
- 随机种子：42；
- 第42轮开始根据验证损失保存最佳模型。

训练图像太少时验证集可能为空并导致除零错误，建议至少准备5张以上图像，
正式训练应使用更多数据。

### 5.4 启动训练

```powershell
python train_1.py
```

程序自动优先使用 CUDA，没有可用显卡时使用处理器。启动时会先为所有原图提取
并缓存 SIFT 关键点，然后进入正式训练。

训练完成后生成：

```text
model_Unet_V6_best.pth
model_Unet_V6_final.pth
```

- `model_Unet_V6_best.pth`：第42轮以后验证损失最低的模型；
- `model_Unet_V6_final.pth`：最后一轮模型。

如果没有生成最佳模型，通常说明训练未运行到第42轮或中途退出。

## 6. 应用部分

### 6.1 修改输入、输出路径

打开 `test.py`，修改文件末尾：

```python
input_dir = r"D:\待去噪指纹目录"
output_dir = r"D:\去噪结果目录"
```

默认加载：

```text
model_Unet_V6_best.pth
```

如需使用其他模型，将调用改为：

```python
test_folder(
    input_dir,
    output_dir,
    model_path="你的模型.pth",
)
```

### 6.2 启动批量去噪

确认当前目录为 `denoising` 后执行：

```powershell
python test.py
```

程序会递归扫描输入目录，支持：

- `.bmp`
- `.png`
- `.jpg`
- `.jpeg`

去噪结果保存到输出目录，并保持原有子目录结构。已存在的同名文件会被覆盖。

### 6.3 信噪比测试

`test_snr.py` 用于定量检查模型对合成高斯噪声的恢复效果。打开文件末尾并修改：

```python
input_dir = r"D:\干净指纹测试目录"
output_dir = r"D:\信噪比测试结果"

test_folder(
    input_dir,
    output_dir,
    model_path="model_Unet_V6_best.pth",
    sigma=0.15,
)
```

然后执行：

```powershell
python test_snr.py
```

输出目录包含：

```text
信噪比测试结果/
├── GT/
├── Noisy/
└── Denoised/
```

终端会输出每张图的加噪信噪比、去噪信噪比和提升倍数。当前代码使用线性信噪比，
不是以分贝表示的 `10×log10` 信噪比。

## 7. 可选辅助处理

如果需要在去噪后进行拉普拉斯锐化，先修改 `laplas.py` 中的：

```python
input_root = r"D:\去噪结果目录"
output_root = r"D:\锐化结果目录"
```

然后执行：

```powershell
python laplas.py
```

当前锐化强度由下列代码控制：

```python
img_sharp = laplacian_sharpen(img, alpha=0.2)
```

## 8. 完整运行命令

首次训练并应用：

```powershell
cd C:\Users\ZYH\Desktop\harnet32\denoising
$env:PYTHONPATH="..;$env:PYTHONPATH"
python train_1.py
python test.py
```

已有 V6 模型时直接应用：

```powershell
cd C:\Users\ZYH\Desktop\harnet32\denoising
python test.py
```

只运行信噪比测试：

```powershell
cd C:\Users\ZYH\Desktop\harnet32\denoising
python test_snr.py
```

## 9. 常见问题

### 找不到 `denoising` 或 `runtime` 模块

确认运行目录和模块搜索路径：

```powershell
cd C:\Users\ZYH\Desktop\harnet32\denoising
$env:PYTHONPATH="..;$env:PYTHONPATH"
python train_1.py
```

### 找不到训练图像

`train_1.py` 只读取训练目录第一层中的 `.png` 文件，不会递归读取子目录，也不会
读取 BMP。检查 `train_target`、文件扩展名和目录层级。

### 模型参数无法加载

`test.py` 的网络结构与 V6 模型一致。不要把其他结构的 checkpoint 直接改名为
`model_Unet_V6_best.pth`；应通过 `model_path` 明确选择与当前网络结构兼容的权重。

### 验证损失计算时报除零错误

通常是训练数据太少，使20%验证集取整后为0。增加训练图像数量后重新运行。
