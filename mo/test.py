import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from dataset import MembraneDataset
from model import Encoder3D


# ===================== 1. 路径 =====================
ROOT = r"C:\Users\qwe\Desktop\GHM_ID11"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ===================== 2. 加载模型 =====================
encoder = Encoder3D().to(device)

ckpt = torch.load(
    "best_model.pth",
    map_location=device
)

encoder.load_state_dict(
    ckpt["encoder"]
)
encoder.eval()

# ===================== 3. 加载特征库 =====================
feature_db = torch.load(
    "feature_db.pt",
    map_location="cpu"
)

print("特征库类别数:", len(feature_db))

# ===================== 4. 数据集 =====================
dataset = MembraneDataset(ROOT)

print("测试样本数:", len(dataset))

# ===================== 5. 测试 =====================
correct = 0
total = 0

all_scores = []   # ⭐ 收集所有 best_score

with torch.no_grad():

    for i in range(len(dataset)):

        x, true_label = dataset[i]

        x = x.unsqueeze(0).to(device)

        feat = encoder(x)

        feat = F.normalize(
            feat,
            p=2,
            dim=1
        )

        feat = feat.squeeze(0).cpu()

        best_label = None
        best_score = -999

        print("\n==============================")
        print(f"样本 {i} | GT = {true_label}")

        # ---- 和所有类别比 ----
        for label, prototype in feature_db.items():

            score = torch.dot(
                feat,
                prototype
            ).item()

            print(
                f"Class={label} Score={score:.4f}"
            )

            if score > best_score:
                best_score = score
                best_label = label

        # ---- 输出结果 ----
        print("------------------------------")
        print("预测类别:", best_label)
        print("最大相似度:", best_score)

        # ⭐ 收集score
        all_scores.append(best_score)

        # ⭐ 你原来的阈值判断
        if best_score > 0.97:
            print(">>> 高置信度预测")

        # ---- 统计准确率 ----
        if best_label == true_label:
            correct += 1

        total += 1


# ===================== 6. 准确率 =====================
acc = correct / total

print("\n==============================")
print("测试完成")
print("Top1 Accuracy:", acc)
print("==============================")


# ===================== 7. score统计 =====================
scores = np.array(all_scores)

print("\n==============================")
print("Score统计信息")
print("==============================")

print("样本数:", len(scores))
print("均值:", scores.mean())
print("标准差:", scores.std())
print("最小值:", scores.min())
print("最大值:", scores.max())
print("50%:", np.percentile(scores, 50))
print("95%:", np.percentile(scores, 95))
print("97%:", np.percentile(scores, 97))


# ===================== 8. 直方图 =====================
plt.figure(figsize=(8, 5))

plt.hist(scores, bins=30)

plt.title("Best Score Distribution")
plt.xlabel("Cosine Similarity (best_score)")
plt.ylabel("Count")

plt.grid(True, alpha=0.3)

plt.show()