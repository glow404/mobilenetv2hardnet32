import random
from collections import defaultdict
from multiprocessing import freeze_support

import torch
from torch.utils.data import DataLoader, Subset

from dataset import MembraneDataset
from model import Encoder3D
from arcface import ArcFace


ROOT = r"C:\Users\qwe\Desktop\finger\mo"

BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4

VAL_RATIO = 0.2
SAVE_AFTER_EPOCH = 10
SEED = 42


def split_by_class(dataset, val_ratio=0.2):
    """
    每个膜内部按比例划分
    例如:
        300 pair
        -> 240 train
        -> 60 val
    """

    class_indices = defaultdict(list)

    for idx, (_, label) in enumerate(dataset.samples):
        class_indices[label].append(idx)

    train_indices = []
    val_indices = []

    random.seed(SEED)

    for label, indices in class_indices.items():

        random.shuffle(indices)

        n_val = int(len(indices) * val_ratio)

        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    return train_indices, val_indices


def evaluate(
        encoder,
        arcface,
        loader,
        criterion,
        device):

    encoder.eval()
    arcface.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            feat = encoder(x)

            logits = arcface(
                feat,
                y
            )

            loss = criterion(
                logits,
                y
            )

            total_loss += loss.item()

            pred = logits.argmax(dim=1)

            correct += (
                pred == y
            ).sum().item()

            total += y.size(0)

    avg_loss = total_loss / len(loader)
    acc = correct / total

    return avg_loss, acc


def main():

    dataset = MembraneDataset(ROOT)

    print("=" * 60)
    print("类别数:", len(dataset.class_to_idx))
    print("总样本数:", len(dataset))
    print("=" * 60)

    train_idx, val_idx = split_by_class(
        dataset,
        VAL_RATIO
    )

    train_dataset = Subset(
        dataset,
        train_idx
    )

    val_dataset = Subset(
        dataset,
        val_idx
    )

    print("训练样本:", len(train_dataset))
    print("验证样本:", len(val_dataset))
    print("=" * 60)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    num_classes = len(
        dataset.class_to_idx
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    encoder = Encoder3D().to(device)

    arcface = ArcFace(
        embedding_size=128,
        num_classes=num_classes
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(arcface.parameters()),
        lr=LR
    )

    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0

    for epoch in range(EPOCHS):

        encoder.train()
        arcface.train()

        total_loss = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            feat = encoder(x)

            logits = arcface(
                feat,
                y
            )

            loss = criterion(
                logits,
                y
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        train_loss = (
            total_loss /
            len(train_loader)
        )

        val_loss, val_acc = evaluate(
            encoder,
            arcface,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch+1:03d}/{EPOCHS} | "
            f"Train Loss={train_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val Acc={val_acc:.4f}"
        )

        # 第10轮后开始保存最佳模型
        if epoch + 1 >= SAVE_AFTER_EPOCH :

            if val_acc>0.9 and val_acc > best_acc:

                best_acc = val_acc

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "best_acc": best_acc,
                        "encoder": encoder.state_dict(),
                        "arcface": arcface.state_dict()
                    },
                    "best_model.pth"
                )

                print(
                    f"*** 保存最佳模型 "
                    f"(Epoch={epoch+1}, "
                    f"Acc={best_acc:.4f}) ***"
                )

    print("=" * 60)
    print(
        f"训练结束，最佳验证准确率: "
        f"{best_acc:.4f}"
    )
    print("=" * 60)


if __name__ == "__main__":

    freeze_support()

    main()