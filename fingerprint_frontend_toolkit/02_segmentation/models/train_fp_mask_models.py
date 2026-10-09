import os
import csv
import json
import time
import math
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =========================
# Utils
# =========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_pairs_csv(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "input_csv" not in row or "mask_csv" not in row:
                raise ValueError("train_pairs.csv must contain columns: input_csv, mask_csv")
            rows.append(row)
    if len(rows) == 0:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def load_float_csv(path: str, expected_h=None, expected_w=None):
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"CSV is not 2D: {path}, shape={arr.shape}")

    if expected_h is not None and expected_w is not None:
        if arr.shape == (expected_h, expected_w):
            return arr
        if arr.shape == (expected_w, expected_h):
            # Be tolerant to accidentally transposed csv files.
            return arr.T
        raise ValueError(
            f"Unexpected CSV shape at {path}: {arr.shape}, "
            f"expected {(expected_h, expected_w)} or transposed {(expected_w, expected_h)}"
        )
    return arr


def normalize_like_segmentation(x: np.ndarray) -> np.ndarray:
    """
    Match the earlier supervision-generation behavior:
    min-max normalize each sample to [0,1].
    """
    x = x.astype(np.float32)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - mn) / (mx - mn)
    return x.astype(np.float32)


def split_rows(rows, val_ratio=0.1, seed=42):
    idx = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    val_n = int(round(len(rows) * val_ratio))
    if len(rows) > 1:
        val_n = min(max(val_n, 1), len(rows) - 1)
    else:
        val_n = 0

    val_idx = set(idx[:val_n])
    train_rows = [rows[i] for i in range(len(rows)) if i not in val_idx]
    val_rows = [rows[i] for i in range(len(rows)) if i in val_idx]
    return train_rows, val_rows


def write_rows_csv(rows, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =========================
# Dataset
# =========================

class FingerprintMaskCsvDataset(Dataset):
    def __init__(self, rows, expected_h=110, expected_w=100, augment=False):
        self.rows = rows
        self.expected_h = expected_h
        self.expected_w = expected_w
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def _augment(self, x, y):
        if np.random.rand() < 0.5:
            x = np.fliplr(x).copy()
            y = np.fliplr(y).copy()
        if np.random.rand() < 0.5:
            x = np.flipud(x).copy()
            y = np.flipud(y).copy()
        # For non-square inputs like 110x100, 90/270-degree rotation changes shape
        # and breaks DataLoader stacking. Keep only 180-degree rotation.
        if np.random.rand() < 0.25:
            x = np.rot90(x, 2).copy()
            y = np.rot90(y, 2).copy()
        return x, y

    def __getitem__(self, idx):
        row = self.rows[idx]
        x = load_float_csv(row["input_csv"], self.expected_h, self.expected_w)
        y = load_float_csv(row["mask_csv"], self.expected_h, self.expected_w)

        x = normalize_like_segmentation(x)
        y = (y > 0.5).astype(np.float32)

        if self.augment:
            x, y = self._augment(x, y)

        return {
            "image": torch.from_numpy(x[None, ...]).float(),
            "mask": torch.from_numpy(y[None, ...]).float(),
            "sample_id": row.get("sample_id", f"idx_{idx:06d}"),
        }


# =========================
# Models
# =========================

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, groups=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        layers = [
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = ConvBNAct(in_ch, in_ch, k=3, s=stride, groups=in_ch)
        self.pw = ConvBNAct(in_ch, out_ch, k=1, s=1, p=0)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x


class TinyUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=12):
        super().__init__()
        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 4
        c4 = base_ch * 6

        self.enc1 = nn.Sequential(
            ConvBNAct(in_ch, c1, 3),
            ConvBNAct(c1, c1, 3),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            ConvBNAct(c1, c2, 3),
            ConvBNAct(c2, c2, 3),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            ConvBNAct(c2, c3, 3),
            ConvBNAct(c3, c3, 3),
        )
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ConvBNAct(c3, c4, 3),
            ConvBNAct(c4, c4, 3),
        )

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBNAct(c3 + c3, c3, 3),
            ConvBNAct(c3, c3, 3),
        )

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBNAct(c2 + c2, c2, 3),
            ConvBNAct(c2, c2, 3),
        )

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBNAct(c1 + c1, c1, 3),
            ConvBNAct(c1, c1, 3),
        )

        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        b = self.bottleneck(self.pool3(s3))

        x = self.up3(b)
        if x.shape[-2:] != s3.shape[-2:]:
            x = F.interpolate(x, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, s3], dim=1))

        x = self.up2(x)
        if x.shape[-2:] != s2.shape[-2:]:
            x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, s2], dim=1))

        x = self.up1(x)
        if x.shape[-2:] != s1.shape[-2:]:
            x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, s1], dim=1))

        return self.head(x)


class TinyMobileSeg(nn.Module):
    """
    Very small encoder-decoder with depthwise separable convs.
    """
    def __init__(self, in_ch=1, base_ch=16):
        super().__init__()
        c1 = base_ch
        c2 = base_ch * 2
        c3 = base_ch * 3
        c4 = base_ch * 4

        self.stem = ConvBNAct(in_ch, c1, 3, s=2)            # 110x100 -> 55x50
        self.enc1 = DSConv(c1, c2, stride=2)                # -> 28x25
        self.enc2 = DSConv(c2, c3, stride=2)                # -> 14x13
        self.enc3 = DSConv(c3, c4, stride=2)                # -> 7x7 approx

        self.mid = nn.Sequential(
            DSConv(c4, c4, stride=1),
            DSConv(c4, c4, stride=1),
        )

        self.dec3 = DSConv(c4 + c3, c3, stride=1)
        self.dec2 = DSConv(c3 + c2, c2, stride=1)
        self.dec1 = DSConv(c2 + c1, c1, stride=1)
        self.dec0 = ConvBNAct(c1, c1, 3, s=1)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    def _up_to(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.enc1(s1)
        s3 = self.enc2(s2)
        x = self.enc3(s3)
        x = self.mid(x)

        x = self._up_to(x, s3)
        x = self.dec3(torch.cat([x, s3], dim=1))

        x = self._up_to(x, s2)
        x = self.dec2(torch.cat([x, s2], dim=1))

        x = self._up_to(x, s1)
        x = self.dec1(torch.cat([x, s1], dim=1))

        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.dec0(x)
        x = F.interpolate(x, size=(110, 100), mode="bilinear", align_corners=False)
        x = self.head(x)
        return x


def build_model(model_name: str, base_ch: int):
    model_name = model_name.lower()
    if model_name == "tiny_unet":
        return TinyUNet(in_ch=1, base_ch=base_ch)
    if model_name == "mobile_seg":
        return TinyMobileSeg(in_ch=1, base_ch=base_ch)
    raise ValueError(f"Unknown model: {model_name}")


# =========================
# Loss / Metrics
# =========================

def dice_loss_from_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def compute_batch_metrics_from_logits(logits, target, threshold=0.5, eps=1e-6):
    prob = torch.sigmoid(logits)
    pred = (prob >= threshold).float()

    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))
    tn = ((1 - pred) * (1 - target)).sum(dim=(1, 2, 3))

    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {
        "iou": iou.mean().item(),
        "dice": dice.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "acc": acc.mean().item(),
    }


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =========================
# Visualization
# =========================

def save_visual(batch, logits, save_path, max_items=4):
    x = batch["image"].detach().cpu().numpy()
    y = batch["mask"].detach().cpu().numpy()
    p = torch.sigmoid(logits).detach().cpu().numpy()

    n = min(x.shape[0], max_items)
    rows = []
    for i in range(n):
        img = (x[i, 0] * 255.0).clip(0, 255).astype(np.uint8)
        gt = (y[i, 0] * 255.0).clip(0, 255).astype(np.uint8)
        pr = (p[i, 0] * 255.0).clip(0, 255).astype(np.uint8)
        pb = (pr >= 128).astype(np.uint8) * 255
        err = np.abs(pb.astype(np.int16) - gt.astype(np.int16)).astype(np.uint8)

        def to_bgr(z):
            return np.stack([z, z, z], axis=-1)

        panels = [to_bgr(img), to_bgr(gt), to_bgr(pr), to_bgr(pb), to_bgr(err)]
        titles = ["input", "gt_mask", "prob", "pred_bin", "abs_err"]
        titled = []
        for panel, title in zip(panels, titles):
            canvas = panel.copy()
            import cv2
            cv2.putText(canvas, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            canvas = cv2.copyMakeBorder(canvas, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            titled.append(canvas)
        rows.append(np.concatenate(titled, axis=1))

    if rows:
        import cv2
        board = np.concatenate(rows, axis=0)
        cv2.imwrite(str(save_path), board)


# =========================
# Train / Eval
# =========================

def run_epoch(model, loader, optimizer, device, lambda_dice=1.0, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_acc = 0.0
    total_count = 0

    vis_batch = None
    vis_logits = None

    for batch in loader:
        img = batch["image"].to(device)
        mask = batch["mask"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(img)
            bce = F.binary_cross_entropy_with_logits(logits, mask)
            dloss = dice_loss_from_logits(logits, mask)
            loss = bce + lambda_dice * dloss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = img.size(0)
        metric = compute_batch_metrics_from_logits(logits.detach(), mask.detach())

        total_loss += loss.item() * bs
        total_bce += bce.item() * bs
        total_dice_loss += dloss.item() * bs
        total_iou += metric["iou"] * bs
        total_dice += metric["dice"] * bs
        total_precision += metric["precision"] * bs
        total_recall += metric["recall"] * bs
        total_acc += metric["acc"] * bs
        total_count += bs

        if vis_batch is None:
            vis_batch = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in batch.items()}
            vis_logits = logits.detach().cpu()

    return {
        "loss": total_loss / total_count,
        "bce": total_bce / total_count,
        "dice_loss": total_dice_loss / total_count,
        "iou": total_iou / total_count,
        "dice": total_dice / total_count,
        "precision": total_precision / total_count,
        "recall": total_recall / total_count,
        "acc": total_acc / total_count,
    }, vis_batch, vis_logits


def benchmark_inference_ms(model, device, h=110, w=100, warmup=20, repeat=100):
    x = torch.rand(1, 1, h, w, device=device)
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times = []
    with torch.no_grad():
        for _ in range(repeat):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "warmup": warmup,
        "repeat": repeat,
    }


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_csv", type=str, required=True, help="train_pairs.csv path")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="tiny_unet", choices=["tiny_unet", "mobile_seg"])
    parser.add_argument("--expected_h", type=int, default=110)
    parser.add_argument("--expected_w", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--base_ch", type=int, default=12)
    parser.add_argument("--lambda_dice", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    vis_dir = save_dir / "vis"
    ensure_dir(save_dir)
    ensure_dir(vis_dir)

    set_seed(args.seed)

    rows = read_pairs_csv(args.pairs_csv)
    train_rows, val_rows = split_rows(rows, val_ratio=args.val_ratio, seed=args.seed)
    write_rows_csv(train_rows, save_dir / "train_split.csv")
    write_rows_csv(val_rows, save_dir / "val_split.csv")

    train_set = FingerprintMaskCsvDataset(
        train_rows,
        expected_h=args.expected_h,
        expected_w=args.expected_w,
        augment=True,
    )
    val_set = FingerprintMaskCsvDataset(
        val_rows,
        expected_h=args.expected_h,
        expected_w=args.expected_w,
        augment=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples:   {len(val_set)}")

    model = build_model(args.model, args.base_ch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_params, trainable_params = count_parameters(model)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    best_key = -1.0
    best_epoch = -1
    log_path = save_dir / "log.csv"

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_bce", "train_dice_loss", "train_iou", "train_dice", "train_precision", "train_recall", "train_acc",
            "val_loss", "val_bce", "val_dice_loss", "val_iou", "val_dice", "val_precision", "val_recall", "val_acc",
        ])

    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(
            model, train_loader, optimizer, device,
            lambda_dice=args.lambda_dice, train=True
        )
        val_metrics, vis_batch, vis_logits = run_epoch(
            model, val_loader, optimizer, device,
            lambda_dice=args.lambda_dice, train=False
        )

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_iou={train_metrics['iou']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} | "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_iou={val_metrics['iou']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["bce"], train_metrics["dice_loss"], train_metrics["iou"], train_metrics["dice"], train_metrics["precision"], train_metrics["recall"], train_metrics["acc"],
                val_metrics["loss"], val_metrics["bce"], val_metrics["dice_loss"], val_metrics["iou"], val_metrics["dice"], val_metrics["precision"], val_metrics["recall"], val_metrics["acc"],
            ])

        if vis_batch is not None and (epoch == 1 or epoch % 5 == 0):
            save_visual(vis_batch, vis_logits, vis_dir / f"epoch_{epoch:03d}.png")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(ckpt, save_dir / "last.pt")

        # prioritize val dice, tie-break by iou
        key = val_metrics["dice"] + 0.1 * val_metrics["iou"]
        if key > best_key:
            best_key = key
            best_epoch = epoch
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  -> saved best checkpoint at epoch {epoch}")

    # benchmark best model
    best_ckpt = torch.load(save_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    model.eval()
    bench = benchmark_inference_ms(model, device, h=args.expected_h, w=args.expected_w)

    summary = {
        "model": args.model,
        "base_ch": args.base_ch,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "pairs_csv": args.pairs_csv,
        "train_count": len(train_set),
        "val_count": len(val_set),
        "best_epoch": best_epoch,
        "best_val_metrics": best_ckpt["val_metrics"],
        "inference_benchmark_ms": bench,
    }
    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTraining finished.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()
