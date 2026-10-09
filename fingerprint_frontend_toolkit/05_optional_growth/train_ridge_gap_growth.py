#!/usr/bin/env python3
"""Train a flow-conditioned fingerprint ridge-gap growth/completion model.

Training pairs are generated online from clean, identity-disjoint fingerprints:
  clean target -> tangent-aligned broken-ridge masks -> masked image + mask +
  ridge-flow fields estimated exclusively from the unmasked boundary.

Known pixels are hard-copied at train/eval time. The network therefore learns
only the missing area, while orientation and confidence channels tell it how
the visible ridge field enters the gap. Outputs/checkpoints are new and never
overwrite the older growth or restoration experiments.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Block(nn.Module):
    def __init__(self, ci: int, co: int):
        super().__init__()
        groups = min(8, co)
        self.body = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.GroupNorm(groups, co), nn.SiLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.GroupNorm(groups, co), nn.SiLU(inplace=True),
        )
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x):
        return self.body(x) + self.skip(x)


class FlowGrowthUNet(nn.Module):
    """5 inputs: corrupted gray, mask, sin(2*tangent), cos(2*tangent), flow confidence."""
    def __init__(self, base: int = 24):
        super().__init__()
        self.e1 = Block(5, base)
        self.e2 = Block(base, 2 * base)
        self.e3 = Block(2 * base, 4 * base)
        self.e4 = Block(4 * base, 6 * base)
        self.mid = Block(6 * base, 8 * base)
        self.d4 = Block(14 * base, 6 * base)
        self.d3 = Block(10 * base, 4 * base)
        self.d2 = Block(6 * base, 2 * base)
        self.d1 = Block(3 * base, base)
        self.out = nn.Conv2d(base, 1, 1)

    @staticmethod
    def up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.max_pool2d(a, 2))
        c = self.e3(F.max_pool2d(b, 2))
        d = self.e4(F.max_pool2d(c, 2))
        z = self.mid(F.max_pool2d(d, 2))
        z = self.d4(torch.cat([self.up(z, d), d], 1))
        z = self.d3(torch.cat([self.up(z, c), c], 1))
        z = self.d2(torch.cat([self.up(z, b), b], 1))
        z = self.d1(torch.cat([self.up(z, a), a], 1))
        return torch.sigmoid(self.out(z))


def tensor_fields(gray: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate tangent/coherence from visible pixels only; suppress mask-edge leakage."""
    valid = (mask < 0.5).astype(np.uint8)
    valid = cv2.erode(valid, np.ones((7, 7), np.uint8), iterations=1).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    weight = cv2.GaussianBlur(valid, (0, 0), 4.0) + 1e-5
    xx = cv2.GaussianBlur(gx * gx * valid, (0, 0), 4.0) / weight
    yy = cv2.GaussianBlur(gy * gy * valid, (0, 0), 4.0) / weight
    xy = cv2.GaussianBlur(gx * gy * valid, (0, 0), 4.0) / weight
    tr = xx + yy + 1e-7
    coherence = np.sqrt((xx - yy) ** 2 + 4 * xy * xy) / tr
    tangent = 0.5 * np.arctan2(2 * xy, xx - yy) + np.pi / 2
    # Flow confidence decays where the tensor is weak and where no visible
    # boundary evidence reaches the hole.
    confidence = np.clip(coherence, 0, 1) * np.clip(weight / (weight.max() * 0.18 + 1e-7), 0, 1)
    return tangent.astype(np.float32), coherence.astype(np.float32), confidence.astype(np.float32)


def make_growth_mask(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = gray.shape
    # Use the clean sample only to place realistic synthetic failures; these
    # fields are recomputed from the masked input before entering the model.
    tangent, coherence, _ = tensor_fields(gray, np.zeros_like(gray, np.float32))
    for _attempt in range(12):
        mask = np.zeros((h, w), np.uint8)
        for _ in range(int(rng.integers(1, 3))):
            y = int(rng.integers(18, h - 18)); x = int(rng.integers(18, w - 18))
            theta = float(tangent[y, x])
            if coherence[y, x] < 0.18:
                theta = float(rng.uniform(-np.pi, np.pi))
            # Longer along-ridge breaks force actual continuation, not just
            # interpolation of isolated bad pixels.
            axes = (int(rng.integers(11, 23)), int(rng.integers(3, 6)))
            cv2.ellipse(mask, (x, y), axes, np.degrees(theta), 0, 360, 1, -1)
        # Add an irregular clustered dropout, as found in low-quality captures.
        if rng.random() < 0.65:
            cy = int(rng.integers(16, h - 16)); cx = int(rng.integers(16, w - 16))
            for _ in range(int(rng.integers(7, 18))):
                px = int(np.clip(cx + rng.normal(0, 5.5), 0, w - 1))
                py = int(np.clip(cy + rng.normal(0, 5.5), 0, h - 1))
                cv2.circle(mask, (px, py), int(rng.integers(1, 4)), 1, -1)
        if 0.008 < mask.mean() < 0.28:
            return mask.astype(np.float32)
    return mask.astype(np.float32)


class GrowthDataset(Dataset):
    def __init__(self, files: list[Path], patch: int, train: bool, seed: int):
        self.files, self.patch, self.train, self.seed = files, patch, train, seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = cv2.imread(str(self.files[index]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Cannot read {self.files[index]}")
        h, w = image.shape
        if h < self.patch or w < self.patch:
            raise ValueError(f"Image smaller than patch: {self.files[index]} {image.shape}")
        rng = np.random.default_rng(self.seed + index + (random.randrange(1 << 20) if self.train else 0))
        y = int(rng.integers(0, h - self.patch + 1)) if self.train else (h - self.patch) // 2
        x = int(rng.integers(0, w - self.patch + 1)) if self.train else (w - self.patch) // 2
        target = image[y:y + self.patch, x:x + self.patch].astype(np.float32) / 255.0
        if self.train and rng.random() < 0.5: target = np.fliplr(target).copy()
        if self.train and rng.random() < 0.5: target = np.flipud(target).copy()
        mask = make_growth_mask(target, rng)
        mu, sigma = float(target.mean()), max(float(target.std()), 0.04)
        corrupt_noise = np.clip(rng.normal(mu, sigma, target.shape), 0, 1).astype(np.float32)
        impulse = rng.random(target.shape) < 0.06
        if impulse.any(): corrupt_noise[impulse] = rng.choice([0.0, 1.0], size=int(impulse.sum()))
        corrupted = target * (1 - mask) + corrupt_noise * mask
        tangent, _, confidence = tensor_fields(corrupted, mask)
        angle2 = 2 * tangent
        model_input = np.stack([corrupted, mask, np.sin(angle2), np.cos(angle2), confidence]).astype(np.float32)
        return torch.from_numpy(model_input), torch.from_numpy(target[None].copy()), torch.from_numpy(mask[None])


def gabor_filters(device, dtype=torch.float32):
    key = (device.type, device.index, dtype)
    cache = getattr(gabor_filters, "_cache", {})
    if key not in cache:
        size, sigma, wavelength, gamma = 15, 2.4, 6.0, 0.65
        c = np.arange(size, dtype=np.float32) - (size - 1) / 2
        yy, xx = np.meshgrid(c, c, indexing="ij")
        kernels = []
        for theta in np.linspace(0, np.pi, 8, endpoint=False):
            xr = xx * np.cos(theta) + yy * np.sin(theta)
            yr = -xx * np.sin(theta) + yy * np.cos(theta)
            env = np.exp(-(xr * xr + gamma * gamma * yr * yr) / (2 * sigma * sigma))
            for phase in (0.0, np.pi / 2):
                k = env * np.cos(2 * np.pi * xr / wavelength + phase)
                k -= k.mean(); k /= np.abs(k).sum() + 1e-8
                kernels.append(k)
        cache[key] = torch.from_numpy(np.stack(kernels)[:, None]).to(device=device, dtype=dtype)
        gabor_filters._cache = cache
    return cache[key]


def grad_xy(x):
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None] / 8
    ky = kx.transpose(-1, -2)
    return F.conv2d(x, kx, padding=1), F.conv2d(x, ky, padding=1)


def dilate(mask, radius=3):
    return F.max_pool2d(mask, 2 * radius + 1, stride=1, padding=radius)


def losses(pred, target, mask, lambda_gabor: float):
    support = dilate(mask)
    l1 = ((pred - target).abs() * mask).sum() / (mask.sum() + 1e-6)
    px, py = grad_xy(pred); tx, ty = grad_xy(target)
    grad = (((px - tx).abs() + (py - ty).abs()) * support).sum() / (2 * support.sum() + 1e-6)
    pn = torch.sqrt(px.square() + py.square() + 1e-6)
    tn = torch.sqrt(tx.square() + ty.square() + 1e-6)
    ori = (((1 - ((px * tx + py * ty) / (pn * tn)).clamp(-1, 1).abs()) * support).sum()
           / (support.sum() + 1e-6))
    bank = gabor_filters(pred.device, pred.dtype)
    pg = F.conv2d(pred, bank, padding=bank.shape[-1] // 2)
    tg = F.conv2d(target, bank, padding=bank.shape[-1] // 2)
    gabor = ((pg - tg).abs() * support).sum() / (support.sum() * bank.shape[0] + 1e-6)
    total = l1 + 0.25 * grad + 0.20 * ori + lambda_gabor * gabor
    return total, l1, grad, ori, gabor


@torch.no_grad()
def evaluate(model, loader, device, lambda_gabor):
    model.eval()
    sums = np.zeros(7, dtype=np.float64)
    preview_rows = []
    for inp, target, mask in loader:
        inp, target, mask = inp.to(device), target.to(device), mask.to(device)
        proposal = model(inp)
        result = proposal * mask + inp[:, :1] * (1 - mask)
        parts = losses(result.float(), target.float(), mask.float(), lambda_gabor)
        mse = (((result - target).square()) * mask).sum() / (mask.sum() + 1e-6)
        lap = target.new_tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])[None, None]
        lap_r = F.conv2d(result, lap, padding=1); lap_t = F.conv2d(target, lap, padding=1)
        lap_mean_r = (lap_r * mask).sum() / (mask.sum() + 1e-6)
        lap_mean_t = (lap_t * mask).sum() / (mask.sum() + 1e-6)
        lv_r = (((lap_r - lap_mean_r).square() * mask).sum() / (mask.sum() + 1e-6))
        lv_t = (((lap_t - lap_mean_t).square() * mask).sum() / (mask.sum() + 1e-6))
        sums += np.array([parts[0].item(), mse.item(), parts[1].item(), parts[2].item(), parts[3].item(),
                          lv_r.item(), lv_t.item()]) * target.shape[0]
        if len(preview_rows) < 8:
            for i in range(min(target.shape[0], 8 - len(preview_rows))):
                preview_rows.append([target[i, 0].cpu().numpy(), inp[i, 0].cpu().numpy(),
                                     result[i, 0].cpu().numpy(), mask[i, 0].cpu().numpy()])
    n = max(len(loader.dataset), 1)
    avg = sums / n
    return {"val_score": float(avg[0]), "masked_psnr_db": float(-10 * np.log10(max(avg[1], 1e-12))),
            "masked_l1": float(avg[2]), "masked_gradient_l1": float(avg[3]),
            "masked_orientation_loss": float(avg[4]), "completion_laplacian_variance": float(avg[5]),
            "target_laplacian_variance": float(avg[6])}, preview_rows


def save_preview(rows, path: Path):
    panels = []
    for row in rows:
        cells = []
        for im, title in zip(row, ("clean", "masked", "growth", "mask")):
            u8 = (np.clip(im, 0, 1) * 255).astype(np.uint8)
            tile = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
            cv2.putText(tile, title, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 0), 1, cv2.LINE_AA)
            cells.append(tile)
        panels.append(np.hstack(cells))
    if panels: cv2.imwrite(str(path), np.vstack(panels))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean-root", type=Path, default=Path("指纹识别数据集/不贴屏不贴膜/butieping"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--max-train", type=int, default=0, help="0 uses all training images")
    p.add_argument("--max-val", type=int, default=0, help="0 uses all held-out identities")
    p.add_argument("--lambda-gabor", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20260929)
    a = p.parse_args()

    root = Path.cwd()
    if not a.clean_root.is_absolute(): a.clean_root = root / a.clean_root
    if not a.clean_root.exists(): raise FileNotFoundError(a.clean_root)
    if a.output is None:
        a.output = root / "fingergrowthV5" / "experiments" / f"ridge_gap_growth_{time.strftime('%Y%m%d_%H%M%S')}"
    elif not a.output.is_absolute(): a.output = root / a.output
    if a.output.exists(): raise FileExistsError(f"Refusing to overwrite: {a.output}")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    files = sorted(a.clean_root.rglob("*.bmp"))
    by_identity: dict[str, list[Path]] = {}
    for f in files: by_identity.setdefault(f.parents[1].name, []).append(f)
    identities = sorted(by_identity)
    if len(identities) < 5: raise RuntimeError(f"Expected identity folders, found {len(identities)}")
    n_val = max(2, round(0.2 * len(identities)))
    val_ids = set(identities[-n_val:])
    train_files = [f for name in identities if name not in val_ids for f in by_identity[name]]
    val_files = [f for name in identities if name in val_ids for f in by_identity[name]]
    random.Random(a.seed).shuffle(train_files); random.Random(a.seed + 1).shuffle(val_files)
    if a.max_train: train_files = train_files[:a.max_train]
    if a.max_val: val_files = val_files[:a.max_val]
    if not train_files or not val_files: raise RuntimeError("Empty train or validation split")
    device = torch.device(a.device if a.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    a.output.mkdir(parents=True)
    train_ds = GrowthDataset(train_files, a.patch, True, a.seed)
    val_ds = GrowthDataset(val_files, a.patch, False, a.seed + 100_000)
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                              pin_memory=device.type == "cuda", persistent_workers=a.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers,
                            pin_memory=device.type == "cuda", persistent_workers=a.workers > 0)
    model = FlowGrowthUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    state = {"train_identities": sorted(set(f.parents[1].name for f in train_files)),
             "validation_identities": sorted(val_ids), "train_count": len(train_files),
             "validation_count": len(val_files), "config": vars(a) | {"output": str(a.output), "clean_root": str(a.clean_root)}}
    (a.output / "split.json").write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    best = float("inf")
    fields = ["epoch", "train_loss", "val_score", "masked_psnr_db", "masked_l1",
              "masked_gradient_l1", "masked_orientation_loss", "completion_laplacian_variance",
              "target_laplacian_variance", "learning_rate"]
    with (a.output / "train.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields); writer.writeheader()
        for epoch in range(1, a.epochs + 1):
            model.train(); batch_losses = []
            progress = tqdm(train_loader, desc=f"epoch {epoch:03d}/{a.epochs}", dynamic_ncols=True)
            for inp, target, mask in progress:
                inp = inp.to(device, non_blocking=True); target = target.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    proposal = model(inp)
                # Keep derivative/filter losses in FP32 (RR-09 AMP overflow fix).
                with torch.autocast(device_type=device.type, enabled=False):
                    result = proposal.float() * mask.float() + inp[:, :1].float() * (1 - mask.float())
                    loss = losses(result, target.float(), mask.float(), a.lambda_gabor)[0]
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                batch_losses.append(float(loss.detach().item()))
                progress.set_postfix(loss=f"{batch_losses[-1]:.4f}")
            scheduler.step()
            metrics, preview = evaluate(model, val_loader, device, a.lambda_gabor)
            row = {"epoch": epoch, "train_loss": float(np.mean(batch_losses)), **metrics,
                   "learning_rate": optimizer.param_groups[0]["lr"]}
            writer.writerow(row); fp.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if metrics["val_score"] < best:
                best = metrics["val_score"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics,
                            "model_name": "FlowGrowthUNet", "config": state["config"]}, a.output / "best.pt")
                (a.output / "best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                save_preview(preview, a.output / "comparison.png")
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics,
                        "model_name": "FlowGrowthUNet", "config": state["config"]}, a.output / "last.pt")
    print(f"训练完成；最佳模型与对比图：{a.output}", flush=True)


if __name__ == "__main__":
    main()
