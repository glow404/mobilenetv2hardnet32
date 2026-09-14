# 超声指纹解锁项目

本仓库实现一套从超声采集到指纹解锁的完整链路，并附带钢化膜识别子项目。

主流程：

```text
多帧原始采集图
      ↓
ysjz/          衍射校准复原 → 单张指纹图
      ↓
denoising/     去噪等预处理 → 预处理指纹图
      ↓
myhardnet/     HardNet 描述子匹配 → 注册 / 识别 / 解锁
```

旁路子项目：

```text
mo/            钢化膜（膜结构）类别识别，不参与指纹解锁主链路
```

---

## 1. 目录一览

| 目录 | 作用 |
|------|------|
| `ysjz/` | 多帧差分 BMP 做衍射复原，输出单张指纹图 |
| `denoising/` | 湿指纹 / 含噪指纹去噪，得到预处理图像 |
| `myhardnet/` | 描述子训练、模板注册、离线评估、在线解锁 |
| `mo/` | 钢化膜三维特征提取与类别识别 |

各子目录另有更细的说明文档，本文件汇总整仓启动方式与数据衔接。

---

## 2. 运行环境

### 2.1 通用要求

- 操作系统：Windows（当前路径与脚本按 Windows 编写）
- Python：建议 3.9 及以上；`myhardnet` 依赖声明为 3.10+
- 显卡：训练与大批量匹配建议使用 CUDA；无显卡时可走 CPU（更慢）

### 2.2 依赖安装

**整仓常用依赖（一次装齐）：**

```powershell
cd C:\Users\ZYH\Desktop\harnet32
python -m pip install numpy opencv-python scipy pillow pyyaml pandas matplotlib scikit-learn tqdm
```

**PyTorch** 请按本机 CPU / CUDA 版本单独安装，例如访问 [pytorch.org](https://pytorch.org) 选择对应命令。

**指纹匹配子项目额外依赖**（也可直接装该目录清单）：

```powershell
cd C:\Users\ZYH\Desktop\harnet32\myhardnet
python -m pip install -r requirements.txt
```

**分模块快速自检：**

```powershell
# 衍射复原
python -c "import cv2, numpy, scipy; print('ysjz 依赖正常')"

# 去噪 / HardNet / 膜识别（需已安装 torch）
python -c "import torch, cv2, numpy, PIL, yaml; print('CUDA可用:', torch.cuda.is_available())"
python -c "import torch, numpy, pandas, matplotlib; print('mo 依赖正常')"
```

---

## 3. 端到端主流程（解锁）

推荐按下面顺序跑；前一步输出作为后一步输入。路径请改成你本机目录。

### 阶段 A：衍射校准复原（`ysjz`）

**作用**：同一接触条件下的 5 帧差分图 → 一张复原指纹 BMP。

**修改** `ysjz/ysjz_V4.py` 顶部：

```python
INPUT_DIR = r"D:\原始多帧BMP父目录"
OUTPUT_DIR = r"D:\复原结果目录"
```

**输入**：每个子文件夹内，同一 `pair` 必须凑齐：

```text
Rgd = 1421, 1423, 1425, 1427, 1429
```

文件名示例：`diff-pair_1-Rgd=1421.bmp`

**运行：**

```powershell
cd C:\Users\ZYH\Desktop\harnet32\ysjz
python ysjz_V4.py
```

**输出：**

```text
OUTPUT_DIR/
└── 同名子文件夹/
    ├── pair_1.bmp
    └── pair_2.bmp
```

缺任一 `Rgd` 的 pair 会被跳过。详细说明见 `ysjz/README.md`。

---

### 阶段 B：预处理去噪（`denoising`）

**作用**：对复原后的指纹做 U-Net 去噪（可选再锐化），得到更适合匹配的图像。

**首次使用 / 重新训练**（需要干净 PNG 训练集；图像建议约 100×110）：

```powershell
cd C:\Users\ZYH\Desktop\harnet32\denoising
$env:PYTHONPATH="..;$env:PYTHONPATH"
# 先改 train_1.py 中的 train_target
python train_1.py
```

训练产物：`model_Unet_V6_best.pth`、`model_Unet_V6_final.pth`。

**日常批量去噪**（已有最佳权重时）：打开 `test.py`，设置：

```python
input_dir = r"D:\复原结果目录"      # 通常接 ysjz 的 OUTPUT_DIR
output_dir = r"D:\预处理指纹目录"
```

然后：

```powershell
cd C:\Users\ZYH\Desktop\harnet32\denoising
python test.py
```

**输出**：递归保持子目录结构的去噪图（支持 bmp/png/jpg）。已有同名文件会被覆盖。

可选锐化：改 `laplas.py` 的输入输出后执行 `python laplas.py`。

详细说明见 `denoising/README.md`。

---

### 阶段 C：HardNet 匹配识别（`myhardnet`）

所有命令默认在 `myhardnet` 目录执行：

```powershell
cd C:\Users\ZYH\Desktop\harnet32\myhardnet
```

#### C1. （可选）训练描述子

若已有可用 `best.pt`，可跳过本小节，直接用现成权重做匹配。

1. **构建局部块数据集** — 先改 `pair_build/config.yaml` 中的原始图路径与输出路径：

```powershell
python pair_build/main.py --config pair_build/config.yaml --stage all
```

2. **浮点描述子训练**：

```powershell
python -m hardnet_train.train --config hardnet_train/config.yaml --device cuda
```

3. **（可选）二值描述子**：确认 `hardnet_train/config_binary_256.yaml` 中浮点权重路径后：

```powershell
python -m hardnet_train.train_binary --config hardnet_train/config_binary_256.yaml --device cuda
```

#### C2. 离线注册 + 评估

常调参数集中在 `match_new/config_match_tuning.yaml`，例如：

- `data.image_root`：预处理后的指纹根目录（接阶段 B 输出）
- `model.checkpoint`：HardNet 权重
- `output.output_dir`：实验输出目录
- `enrollment.enrollment_images_per_identity`：每人注册图数量
- 匹配阈值、纹理融合权重等

完整跑模板构建与 FAR/FRR 评估：

```powershell
python match_new\run_hardnet_matching.py --config match_new\config_match_new.yaml
```

仅改匹配阈值、未改模型/关键点时，可复用模板：

```powershell
python match_new\run_hardnet_matching.py `
  --config match_new\config_match_new.yaml `
  --skip-template-build
```

**主要输出（在 `output_dir` 下）：**

| 产物 | 含义 |
|------|------|
| `image_templates/` | 每张图的关键点 + HardNet 描述子模板（`.npz`） |
| `identity_templates_*.json` | 每人注册了哪些模板 |
| `metadata_all.csv` | 原图索引 |
| 匹配结果 CSV / `metrics.json` 等 | 离线分数、FAR/FRR、是否早停等 |

身份判定默认看配置里的 `identification.match_score_threshold`（分数越高越像本人）。

#### C3. 在线解锁

需先完成 C2，保证输出目录里有模板与索引。

**单次验证**（指定一张图、一个声称身份）：

```powershell
python match_new\run_online_unlock.py `
  --config match_new\config_match_new.yaml `
  --image "你的预处理图\某人\pair_41.bmp" `
  --identity "某人身份名"
```

终端会打印是否解锁、匹配分数等 JSON 结果。

**批量延迟基准：**

```powershell
python match_new\run_online_unlock.py `
  --config match_new\config_match_new.yaml `
  --benchmark `
  --limit 100
```

更细的字段说明见 `myhardnet/README.md` 与 `myhardnet/match_new/README.md`。

---

## 4. 主流程命令速查

已有去噪模型与 HardNet 权重、只做「采集 → 解锁」时：

```powershell
# 1) 衍射复原
cd C:\Users\ZYH\Desktop\harnet32\ysjz
# 改 ysjz_V4.py 的 INPUT_DIR / OUTPUT_DIR
python ysjz_V4.py

# 2) 去噪预处理
cd C:\Users\ZYH\Desktop\harnet32\denoising
# 改 test.py 的 input_dir / output_dir（input 接上一步输出）
python test.py

# 3) 离线建库 / 评估（image_root 接去噪输出）
cd C:\Users\ZYH\Desktop\harnet32\myhardnet
# 改 match_new/config_match_tuning.yaml
python match_new\run_hardnet_matching.py --config match_new\config_match_new.yaml

# 4) 在线解锁
python match_new\run_online_unlock.py `
  --config match_new\config_match_new.yaml `
  --image "预处理图路径\xxx.bmp" `
  --identity "注册身份名"
```

数据衔接关系：

```text
ysjz 输出 pair_*.bmp
    → denoising 输入
    → 预处理后的指纹目录
    → myhardnet 的 data.image_root
```

---

## 5. 钢化膜子项目（`mo`）

与指纹解锁并行：用三维编码网络识别膜类别（训练 → 原型库 → 相似度测试）。

```powershell
cd C:\Users\ZYH\Desktop\harnet32\mo
```

数据要求简述：

- 根目录下每个子文件夹 = 一类膜；
- 同一 `pair` 需 61 个 CSV（文件名含 `pair_数字` 与 `Rgd=数字`）；
- 网络实际用按 `Rgd` 排序后的前 16 帧。

**完整顺序：**

```powershell
# 改 train.py 的 ROOT 等参数
python train.py

# 改 build_prototype.py 的 ROOT
python build_prototype.py

# 将 prototype_db/*.npy 转为 test.py 需要的 feature_db.pt（见 mo/README.md 6.2）
# 然后改 test.py 的 ROOT
python test.py
```

已有 `best_model.pth` 与 `feature_db.pt` 时，直接 `python test.py`。

测试会打印各类相似度、预测类别，并统计高置信度（默认相似度 > 0.97）结果。细节见 `mo/README.md`。

---

## 6. 输入 / 输出对照总表

| 阶段 | 入口 | 输入 | 输出 |
|------|------|------|------|
| 衍射复原 | `ysjz/ysjz_V4.py` | 多帧 `diff-pair_*-Rgd=*.bmp`（5 个指定 Rgd） | `pair_*.bmp` |
| 去噪 | `denoising/test.py` | 复原后的指纹图目录 | 同结构去噪图 |
| 描述子训练 | `myhardnet` 的 pair_build + hardnet_train | 指纹图 → 局部块 CSV/patch | `best.pt` 等权重 |
| 离线匹配 | `match_new/run_hardnet_matching.py` | 预处理指纹根目录 + 权重 | 模板库、FAR/FRR 等 |
| 在线解锁 | `match_new/run_online_unlock.py` | 单张图 + 声称身份 + 已有模板 | 接受/拒绝与分数 |
| 膜识别 | `mo/train.py` / `test.py` | 多帧膜 CSV | 类别预测与相似度 |

---

## 7. 输出结果怎么读（主流程）

1. **`ysjz`**：能看到清晰脊线即可；糊、反相、对比度过弱时先核对频率、声速、厚度、DPI 等物理参数。
2. **`denoising`**：噪声应减弱且脊线不过度抹平；可用 `test_snr.py` 做加噪前后对比（当前为线性信噪比，不是分贝）。
3. **`myhardnet` 离线**：关注 FAR（认错他人）与 FRR（拒真本人）；阈值在配置的 `match_score_threshold` 一带调节。
4. **`myhardnet` 在线**：分数 ≥ 阈值则解锁；开启早停时，命中阈值后不再扫完剩余注册模板，本人匹配更快。

---

## 8. 常见注意点

1. 各脚本大量路径写在文件顶部或 yaml 里，**换机必须先改路径再跑**。
2. `denoising` 训练需设置 `$env:PYTHONPATH="..;$env:PYTHONPATH"`，且从 `denoising` 目录启动。
3. HardNet 换了模型或关键点提取方式后，**不能复用旧模板**，需重新跑离线建库。
4. `mo` 的数字类别编号依赖文件夹名字排序，训练、建原型、测试的类别目录命名应保持一致。
5. 子项目细节以各自 README 为准：`ysjz/`、`denoising/`、`myhardnet/`、`mo/`。

---

## 9. 相关文档

- `ysjz/README.md` — 衍射复原参数与文件命名
- `denoising/README.md` — 去噪训练 / 测试 / 锐化
- `myhardnet/README.md` — 数据构建、训练、匹配总览
- `myhardnet/match_new/README.md` — 离线评估与在线解锁参数详解
- `mo/README.md` — 钢化膜训练与识别
