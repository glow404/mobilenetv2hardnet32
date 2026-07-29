"""HardNet 指纹 patch 描述子训练入口。

作用：
    1. 读取 YAML 配置与命令行覆盖参数。
    2. 构造训练随机 sampler 与确定性固定验证协议。
    3. 构造描述子网络、top-k hardest-in-batch triplet loss 和可配置优化器。
    4. 执行带预热的余弦学习率调度训练。
    5. 每个 epoch 输出固定验证指标、checkpoint 和 metrics.csv。

典型用法：
    python -m hardnet_train.train --config hardnet_train/config.yaml --device cuda

输出：
    outputs/hardnet_train/
        best.pt              固定协议 `val_fpr_at_tpr95` 最好的模型
        last.pt              最后一个 epoch 的模型
        metrics.csv          每个 epoch 的训练/固定验证指标
        resolved_config.json 实际使用的配置快照
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

# 当前 Windows 环境中，PyTorch / OpenCV / numpy 可能同时带 Intel OpenMP runtime。
# 训练入口不导入 cv2，但为了避免某些依赖间接触发冲突，这里提前设置兜底环境变量。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from hardnet_train.data import FingerImagePairBatchSampler, FingerprintPairDataset
from hardnet_train.loss import HardNetLoss, normalize_hard_negative_strategy
from hardnet_train.metrics import RunningMean
from hardnet_train.model import (
    build_descriptor_model,
    checkpoint_model_architecture,
    count_parameters,
    model_architecture,
    normalize_model_architecture,
)
from hardnet_train.optim import build_optimizer, normalize_optimizer_name, optimizer_name
from hardnet_train.validation import (
    build_fixed_validation_protocol,
    evaluate_fixed_protocol,
    make_validation_patch_loader,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置，并记录配置文件绝对路径，方便解析相对路径。"""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def resolve_path(config: dict[str, Any], raw_path: str | Path) -> Path:
    """把配置中的路径解析成绝对路径。

    相对路径一律相对配置文件所在目录，而不是相对当前 shell 所在目录。
    """
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (Path(config["_config_path"]).parent / path).resolve()


def set_seed(seed: int) -> None:
    """固定 Python / numpy / PyTorch 随机种子，便于复现实验。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(config: dict[str, Any], split: str, batch_size: int, steps: int, seed: int) -> DataLoader:
    """构造使用随机 hardest-in-batch sampler 的训练 DataLoader。

    固定验证协议不调用此函数，避免验证距离对随 epoch 改变。
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    csv_path = resolve_path(config, data_cfg[f"{split}_csv"])
    max_rows = data_cfg.get(f"max_{split}_rows")
    dataset = FingerprintPairDataset(
        csv_path=csv_path,
        max_rows=max_rows,
        max_rows_per_finger=data_cfg.get(f"max_{split}_rows_per_finger"),
        normalize=bool(data_cfg.get("normalize", True)),
    )
    available_fingers = len({record.finger_id for record in dataset.records})
    # 验证集可能只有 3 根手指；如果配置的 fingers_per_batch 更大，自动降到可用手指数。
    fingers_per_batch = min(int(train_cfg.get("fingers_per_batch", 8)), available_fingers)
    hard_negative_strategy = normalize_hard_negative_strategy(train_cfg.get("hard_negative_strategy", "same_finger_allowed"))
    if hard_negative_strategy == "different_finger" and fingers_per_batch < 2:
        raise ValueError(
            "hard_negative_strategy='different_finger' requires at least 2 fingers per batch "
            f"for split={split}. Got fingers_per_batch={fingers_per_batch}, available_fingers={available_fingers}."
        )
    sampler = FingerImagePairBatchSampler(
        records=dataset.records,
        batch_size=batch_size,
        fingers_per_batch=fingers_per_batch,
        batches_per_epoch=steps,
        seed=seed,
        drop_incomplete=True,
    )
    num_workers = max(0, int(data_cfg.get("num_workers", 0)))
    loader_options: dict[str, Any] = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", False)) and torch.cuda.is_available(),
    }
    # 只有多进程 DataLoader 支持持续 worker 和预取参数。持续 worker 可以避免
    # Windows 在每个 epoch 都重新创建进程，预取则让 CPU 读图与 GPU 计算重叠。
    if num_workers > 0:
        loader_options["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        loader_options["prefetch_factor"] = max(1, int(data_cfg.get("prefetch_factor", 2)))
    return DataLoader(
        **loader_options,
    )


def resolve_device(raw_device: str) -> torch.device:
    """解析训练设备，并在明确要求 CUDA 但环境不可用时给出可操作错误。"""

    requested = str(raw_device).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "训练配置要求 CUDA，但当前 PyTorch 不支持 CUDA。请安装 CUDA 版 PyTorch，"
            "并确认 torch.cuda.is_available() 返回 True；如需主动使用 CPU，请设置 "
            "training.device=cpu。"
        )
    return torch.device(requested)


def resolve_amp(
    device: torch.device,
    raw_precision: str,
) -> tuple[bool, torch.dtype | None]:
    """解析混合精度配置，返回是否启用 autocast 及其数据类型。"""

    precision = str(raw_precision).strip().lower()
    if precision in {"", "off", "none", "fp32", "float32"}:
        return False, None
    if device.type != "cuda":
        return False, None
    if precision in {"fp16", "float16"}:
        return True, torch.float16
    if precision in {"bf16", "bfloat16"}:
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 GPU/PyTorch 不支持 BF16，请改用 training.mixed_precision=fp16。")
        return True, torch.bfloat16
    raise ValueError(f"不支持的 training.mixed_precision: {raw_precision!r}")


def configure_cuda(device: torch.device, train_cfg: dict[str, Any]) -> None:
    """应用只影响 CUDA 执行性能的后端选项。"""

    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", True))
    allow_tf32 = bool(train_cfg.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def linear_lr(base_lr: float, step: int, total_steps: int) -> float:
    """论文设置：学习率在总训练步数内线性衰减到 0。"""
    if total_steps <= 1:
        return base_lr
    return base_lr * max(0.0, 1.0 - step / float(total_steps - 1))


def scheduled_lr(
    base_lr: float,
    step: int,
    total_steps: int,
    scheduler: str,
    warmup_steps: int = 0,
    eta_min: float = 1e-4,
) -> float:
    """计算当前 step 的学习率。

    支持：
        linear:
            论文原始策略，从 base_lr 线性衰减到 0。
        warmup_cosine:
            先从 0 线性 warmup 到 base_lr，再余弦退火到 eta_min。
            这是当前指纹实验的默认策略，避免训练后期直接 lr=0。
    """
    scheduler = scheduler.lower()
    if scheduler == "linear":
        return linear_lr(base_lr, step, total_steps)
    if scheduler != "warmup_cosine":
        raise ValueError(f"Unsupported lr scheduler: {scheduler}")

    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    cosine_total = max(1, total_steps - warmup_steps)
    cosine_step = min(max(step - warmup_steps, 0), cosine_total)
    progress = cosine_step / float(cosine_total)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(eta_min) + (float(base_lr) - float(eta_min)) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """更新优化器中所有参数组的学习率。"""
    for group in optimizer.param_groups:
        group["lr"] = lr


def train_one_epoch(
    model: torch.nn.Module,
    criterion: HardNetLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_steps: int,
    global_step: int,
    base_lr: float,
    scheduler: str,
    warmup_steps: int,
    eta_min: float,
    log_interval: int,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    channels_last: bool,
) -> tuple[dict[str, float], int]:
    """训练一个 epoch。

    返回：
        metrics:
            当前 epoch 的 loss、正样本距离、负样本距离均值。
        global_step:
            全局 step 计数，用于跨 epoch 线性衰减学习率。
    """
    model.train()
    loss_meter = RunningMean()
    pos_meter = RunningMean()
    neg_meter = RunningMean()
    start = time.time()
    lr = base_lr

    for step, batch in enumerate(loader, start=1):
        # 当前默认使用 warmup + cosine decay；这里每个 step 更新一次学习率。
        lr = scheduled_lr(base_lr, global_step, total_steps, scheduler, warmup_steps, eta_min)
        set_optimizer_lr(optimizer, lr)

        # DataLoader 返回 CPU tensor；non_blocking=True 在 pin_memory + CUDA 时可加速拷贝。
        anchor = batch["anchor"].to(device, non_blocking=True)
        positive = batch["positive"].to(device, non_blocking=True)
        point_group = batch["point_group"].to(device, non_blocking=True)
        finger_group = batch["finger_group"].to(device, non_blocking=True)
        if channels_last:
            anchor = anchor.contiguous(memory_format=torch.channels_last)
            positive = positive.contiguous(memory_format=torch.channels_last)

        # 两个分支共享同一个 HardNet 权重，等价于 siamese/two-stream CNN。
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            anchor_desc = model(anchor)
            positive_desc = model(positive)
        # point_group 会屏蔽同一物理点，避免伪负样本参与 hardest negative。
        # 描述子网络使用 AMP，距离矩阵和 hardest-negative loss 保持 FP32，
        # 避免极接近样本在半精度下出现排序抖动。
        loss, stats = criterion(
            anchor_desc.float(),
            positive_desc.float(),
            point_group=point_group,
            finger_group=finger_group,
        )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 日志中的 pos/neg 是 batch 内平均距离，用于观察正负分布是否逐渐拉开。
        batch_size = anchor.size(0)
        valid = stats["valid_triplets"]
        loss_meter.update(float(loss.item()), batch_size)
        pos_meter.update(float(stats["pos_dist"].mean().item()), batch_size)
        if torch.any(valid):
            neg_meter.update(float(stats["neg_dist"][valid].mean().item()), int(valid.sum().item()))

        if log_interval > 0 and step % log_interval == 0:
            elapsed = max(time.time() - start, 1e-6)
            print(
                f"epoch={epoch} step={step}/{len(loader)} lr={lr:.6g} "
                f"loss={loss_meter.value:.4f} pos={pos_meter.value:.4f} "
                f"neg={neg_meter.value:.4f} samples/s={loss_meter.count / elapsed:.1f}",
                flush=True,
            )
        global_step += 1

    return {"loss": loss_meter.value, "pos_dist": pos_meter.value, "neg_dist": neg_meter.value, "lr": lr}, global_step


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """保存模型、优化器、配置和当前验证指标。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_architecture": model_architecture(model),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_name": optimizer_name(optimizer),
        "metrics": metrics,
        "config": {key: value for key, value in config.items() if key != "_config_path"},
    }
    if scaler is not None and scaler.is_enabled():
        payload["amp_scaler"] = scaler.state_dict()
    torch.save(payload, path)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """把 optimizer state 中的 tensor 移动到当前训练设备。

    从 CPU checkpoint 恢复到 CUDA，或从 CUDA 恢复到 CPU 时都需要这一步。
    """
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps_per_epoch: int,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, int]:
    """加载 checkpoint，返回下一轮 epoch 和 global_step。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_architecture = checkpoint_model_architecture(checkpoint)
    current_architecture = model_architecture(model)
    if saved_architecture != current_architecture:
        raise ValueError(
            "Checkpoint architecture mismatch: "
            f"checkpoint={saved_architecture}, model={current_architecture}."
        )
    model.load_state_dict(checkpoint["model"])
    saved_optimizer_name = checkpoint.get("optimizer_name")
    if saved_optimizer_name is not None:
        normalized_saved_optimizer = normalize_optimizer_name(str(saved_optimizer_name))
        if normalized_saved_optimizer != optimizer_name(optimizer):
            raise ValueError(
                "Checkpoint optimizer mismatch: "
                f"checkpoint={normalized_saved_optimizer}, configured={optimizer_name(optimizer)}. "
                "Changing optimizer requires a new output directory and training from scratch."
            )
    if "optimizer" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError as exc:
            raise ValueError(
                "Checkpoint optimizer state is incompatible with the configured decay/no-decay groups. "
                "Start a new experiment instead of restoring the old optimizer state."
            ) from exc
        move_optimizer_state_to_device(optimizer, device)
    if scaler is not None and scaler.is_enabled() and "amp_scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["amp_scaler"])
    finished_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(checkpoint.get("global_step", finished_epoch * steps_per_epoch))
    return finished_epoch + 1, global_step


def best_fpr_at_tpr95_from_metrics(metrics_path: Path) -> tuple[float, float, int]:
    """从当前指标 CSV 恢复主选模指标和早停状态。"""
    if not metrics_path.exists():
        return float("inf"), float("inf"), 0
    rows: list[dict[str, str]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader]
    if not rows:
        return float("inf"), float("inf"), 0

    fprs = [
        float(row["val_fpr_at_tpr95"])
        for row in rows
        if row.get("val_fpr_at_tpr95")
    ]
    best_fpr = min(fprs) if fprs else float("inf")
    last = rows[-1]
    early_best = float(last.get("early_stop_best_fpr_at_tpr95") or best_fpr)
    no_improve = int(float(last.get("no_improve_epochs") or 0))
    return best_fpr, early_best, no_improve


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    """把一个 epoch 的指标追加写入 CSV，并拒绝混写不同验证协议的 schema。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fieldnames = list(row.keys())
    if not write_header:
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing_fields = csv.DictReader(handle).fieldnames
        if existing_fields != fieldnames:
            raise ValueError(
                "metrics.csv schema does not match the fixed validation protocol. "
                "Use a new output directory instead of appending to an old experiment."
            )
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def plot_training_curves(metrics_path: Path, output_path: Path) -> None:
    """根据 metrics.csv 输出训练曲线图。

    图中包含 loss、正负样本距离、`val_fpr_at_tpr95` 和学习率曲线。
    如果 matplotlib 不可用，则跳过绘图，不影响训练结果。
    """
    if not metrics_path.exists():
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"skip plotting: matplotlib unavailable ({exc})", flush=True)
        return

    rows: list[dict[str, str]] = []
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)]
    if not rows:
        return

    def values(name: str, *fallback_names: str) -> list[float]:
        names = (name, *fallback_names)
        result: list[float] = []
        for row in rows:
            value = next((row.get(candidate) for candidate in names if row.get(candidate) not in (None, "")), None)
            if value is not None:
                result.append(float(value))
        return result

    epochs = values("epoch")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)

    axes[0, 0].plot(epochs, values("train_loss"), label="train_loss")
    axes[0, 0].plot(epochs, values("val_loss"), label="val_loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, values("train_pos_dist"), label="train_pos")
    axes[0, 1].plot(epochs, values("train_neg_dist"), label="train_neg")
    axes[0, 1].plot(epochs, values("val_pos_mean", "val_pos_dist"), label="val_pos")
    axes[0, 1].plot(epochs, values("val_neg_mean", "val_neg_dist"), label="val_neg")
    axes[0, 1].set_title("Descriptor Distances")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(
        epochs,
        values("val_fpr_at_tpr95"),
        label="val_fpr_at_tpr95",
        color="tab:red",
    )
    axes[1, 0].set_title("FPR@TPR=95%")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    if rows and "lr" in rows[0]:
        axes[1, 1].plot(epochs, values("lr"), label="lr", color="tab:green")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    YAML 提供默认值；命令行参数用于临时覆盖常用训练超参。
    """
    parser = argparse.ArgumentParser(description="Train HardNet fingerprint patch descriptor.")
    parser.add_argument("--config", default="hardnet_train/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument(
        "--val-steps",
        type=int,
        default=None,
        help="Deprecated: fixed validation size is configured under validation.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--fingers-per-batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--hard-negative-strategy", choices=["same_finger_allowed", "different_finger"], default=None)
    parser.add_argument(
        "--hard-negative-top-k",
        type=int,
        default=None,
        help="Number of hardest legal negatives used per positive pair (default from YAML, normally 3).",
    )
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--scheduler", choices=["linear", "warmup_cosine"], default=None)
    parser.add_argument("--warmup-epochs", type=float, default=None)
    parser.add_argument("--eta-min", type=float, default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from, or 'auto' for output_dir/last.pt.")
    parser.add_argument("--resume-auto", action="store_true", help="Resume from output_dir/last.pt if it exists.")
    parser.add_argument("--no-plot", action="store_true", help="Do not generate training_curves.png after training.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    """训练主流程。"""
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config.setdefault("training", {})
    model_cfg = config.setdefault("model", {})
    optim_cfg = config.setdefault("optimizer", {})
    validation_cfg = config.setdefault("validation", {})
    seed = int(config.get("seed", 42))
    set_seed(seed)

    device_name = args.device or train_cfg.get("device", "auto")
    device = resolve_device(str(device_name))
    configure_cuda(device, train_cfg)
    amp_enabled, amp_dtype = resolve_amp(
        device,
        str(train_cfg.get("mixed_precision", "fp16")),
    )
    channels_last = device.type == "cuda" and bool(train_cfg.get("channels_last", True))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )

    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    if args.steps_per_epoch is not None:
        train_cfg["steps_per_epoch"] = int(args.steps_per_epoch)
    if args.val_steps is not None:
        raise ValueError("--val-steps is no longer used; fixed validation size is configured under validation.")
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.device is not None:
        train_cfg["device"] = str(args.device)
    if args.fingers_per_batch is not None:
        train_cfg["fingers_per_batch"] = int(args.fingers_per_batch)
    if args.lr is not None:
        optim_cfg["lr"] = float(args.lr)
    if args.optimizer is not None:
        optim_cfg["name"] = args.optimizer
    if args.dropout is not None:
        model_cfg["dropout"] = float(args.dropout)
    if args.margin is not None:
        train_cfg["margin"] = float(args.margin)
    if args.hard_negative_strategy is not None:
        train_cfg["hard_negative_strategy"] = args.hard_negative_strategy
    if args.hard_negative_top_k is not None:
        train_cfg["hard_negative_top_k"] = int(args.hard_negative_top_k)
    if args.early_stop_patience is not None:
        train_cfg["early_stop_patience"] = int(args.early_stop_patience)
    if args.early_stop_min_delta is not None:
        train_cfg["early_stop_min_relative_improvement"] = float(args.early_stop_min_delta)
    if args.scheduler is not None:
        train_cfg["scheduler"] = args.scheduler
    if args.warmup_epochs is not None:
        train_cfg["warmup_epochs"] = float(args.warmup_epochs)
    if args.eta_min is not None:
        train_cfg["eta_min"] = float(args.eta_min)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    epochs = int(train_cfg.get("epochs", 10))
    steps_per_epoch = int(train_cfg.get("steps_per_epoch", 1000))
    batch_size = int(train_cfg.get("batch_size", 128))
    train_cfg["epochs"] = epochs
    train_cfg["steps_per_epoch"] = steps_per_epoch
    train_cfg["batch_size"] = batch_size
    optim_cfg["name"] = normalize_optimizer_name(optim_cfg.get("name"))
    optim_cfg["lr"] = float(optim_cfg.get("lr", 0.1))
    optim_cfg["weight_decay"] = float(optim_cfg.get("weight_decay", 1e-4))
    validation_protocol_name = str(validation_cfg.get("protocol", "fixed_pairs_v1")).strip().lower()
    if validation_protocol_name != "fixed_pairs_v1":
        raise ValueError(
            f"Unsupported validation.protocol: {validation_protocol_name!r}. Expected 'fixed_pairs_v1'."
        )
    validation_cfg["protocol"] = validation_protocol_name
    validation_cfg["seed"] = int(validation_cfg.get("seed", seed + 10_000))
    validation_cfg["positive_count"] = int(validation_cfg.get("positive_count", 16_384))
    validation_cfg["same_finger_negatives_per_anchor"] = int(
        validation_cfg.get("same_finger_negatives_per_anchor", 32)
    )
    validation_cfg["cross_finger_negatives_per_anchor"] = int(
        validation_cfg.get("cross_finger_negatives_per_anchor", 32)
    )
    validation_cfg["batch_size"] = int(validation_cfg.get("batch_size", batch_size))

    model_cfg["architecture"] = normalize_model_architecture(
        model_cfg.get("architecture")
    )
    model_cfg["descriptor_dim"] = int(model_cfg.get("descriptor_dim", 128))
    train_cfg["hard_negative_strategy"] = normalize_hard_negative_strategy(train_cfg.get("hard_negative_strategy", "same_finger_allowed"))
    train_cfg["hard_negative_top_k"] = int(train_cfg.get("hard_negative_top_k", 3))
    if train_cfg["hard_negative_top_k"] < 1:
        raise ValueError(
            f"training.hard_negative_top_k must be >= 1, got {train_cfg['hard_negative_top_k']}."
        )

    output_dir = resolve_path(config, config.get("output_dir", "../outputs/hardnet_train"))
    resume_arg = "auto" if args.resume_auto else args.resume
    metrics_path = output_dir / "metrics.csv"
    if metrics_path.exists() and not resume_arg:
        raise FileExistsError(
            f"Output directory already contains metrics: {metrics_path}. "
            "Use a new output directory for a fresh fixed-protocol experiment, or explicitly resume it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # 训练 sampler 保持随机；验证数据流独立扫描 val CSV 并固化索引。
    train_loader = make_loader(config, "train", batch_size=batch_size, steps=steps_per_epoch, seed=seed)
    data_cfg = config["data"]
    validation_dataset = FingerprintPairDataset(
        csv_path=resolve_path(config, data_cfg["val_csv"]),
        max_rows=data_cfg.get("max_val_rows"),
        max_rows_per_finger=data_cfg.get("max_val_rows_per_finger"),
        normalize=bool(data_cfg.get("normalize", True)),
    )
    validation_protocol = build_fixed_validation_protocol(
        records=validation_dataset.records,
        positive_count=int(validation_cfg["positive_count"]),
        same_finger_negatives_per_anchor=int(validation_cfg["same_finger_negatives_per_anchor"]),
        cross_finger_negatives_per_anchor=int(validation_cfg["cross_finger_negatives_per_anchor"]),
        seed=int(validation_cfg["seed"]),
    )
    validation_loader, validation_patch_refs = make_validation_patch_loader(
        dataset=validation_dataset,
        protocol=validation_protocol,
        batch_size=int(validation_cfg["batch_size"]),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
    )
    model = build_descriptor_model(model_cfg).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    criterion = HardNetLoss(
        margin=float(train_cfg.get("margin", 1.0)),
        hard_negative_strategy=str(train_cfg.get("hard_negative_strategy", "same_finger_allowed")),
        hard_negative_top_k=int(train_cfg.get("hard_negative_top_k", 3)),
    )
    optimizer = build_optimizer(model, optim_cfg)

    scheduler = str(train_cfg.get("scheduler", "warmup_cosine"))
    warmup_epochs = float(train_cfg.get("warmup_epochs", 2.0))
    warmup_steps = int(round(warmup_epochs * steps_per_epoch))
    eta_min = float(train_cfg.get("eta_min", 1e-4))
    total_steps = max(1, epochs * steps_per_epoch)
    early_stop_patience = int(train_cfg.get("early_stop_patience", 3))
    early_stop_min_relative = float(train_cfg.get("early_stop_min_relative_improvement", 0.01))
    optimizer_detail = (
        f"momentum={optim_cfg.get('momentum', 0.9)} nesterov={optim_cfg.get('nesterov', True)}"
        if optimizer_name(optimizer) == "sgd"
        else f"betas={optim_cfg.get('betas', [0.9, 0.999])} eps={optim_cfg.get('eps', 1e-8)}"
    )
    print(
        f"device={device} architecture={model_architecture(model)} "
        f"params={count_parameters(model)} batch={batch_size} "
        f"fingers_per_batch={train_cfg.get('fingers_per_batch', 8)} "
        f"dropout={model_cfg.get('dropout', 0.1)} optimizer={optimizer_name(optimizer)} "
        f"lr={optim_cfg.get('lr', 0.1)} {optimizer_detail} "
        f"margin={train_cfg.get('margin', 1.0)} steps_per_epoch={steps_per_epoch} epochs={epochs} "
        f"hard_negative_strategy={train_cfg.get('hard_negative_strategy', 'same_finger_allowed')} "
        f"hard_negative_top_k={train_cfg.get('hard_negative_top_k', 3)} "
        f"scheduler={scheduler} warmup_epochs={warmup_epochs} eta_min={eta_min} "
        f"early_stop_patience={early_stop_patience} early_stop_min_delta={early_stop_min_relative} "
        f"mixed_precision={train_cfg.get('mixed_precision', 'fp16') if amp_enabled else 'fp32'} "
        f"channels_last={channels_last} "
        f"validation_positives={validation_protocol.positive_count} "
        f"validation_unique_patches={validation_patch_refs.size}",
        flush=True,
    )

    best_fpr_at_tpr95, early_stop_best_fpr_at_tpr95, no_improve_epochs = (
        best_fpr_at_tpr95_from_metrics(metrics_path) if resume_arg else (float("inf"), float("inf"), 0)
    )
    start_epoch = 1
    global_step = 0
    if resume_arg:
        resume_path = output_dir / "last.pt" if resume_arg == "auto" else Path(resume_arg).expanduser()
        if not resume_path.is_absolute():
            resume_path = (Path.cwd() / resume_path).resolve()
        if resume_path.exists():
            start_epoch, global_step = load_checkpoint(
                resume_path,
                model,
                optimizer,
                device,
                steps_per_epoch,
                scaler=scaler,
            )
            print(f"resumed from {resume_path} | start_epoch={start_epoch} global_step={global_step}", flush=True)
        elif args.resume_auto or resume_arg == "auto":
            print(f"resume auto skipped: checkpoint not found at {resume_path}", flush=True)
        else:
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")

    # 只有初始化/恢复检查全部通过后才写实际配置，避免失败命令污染既有目录。
    (output_dir / "resolved_config.json").write_text(
        json.dumps({key: value for key, value in config.items() if key != "_config_path"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if start_epoch > epochs:
        print(f"nothing to train: start_epoch={start_epoch} > epochs={epochs}", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        train_metrics, global_step = train_one_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_steps=total_steps,
            global_step=global_step,
            base_lr=float(optim_cfg.get("lr", 0.1)),
            scheduler=scheduler,
            warmup_steps=warmup_steps,
            eta_min=eta_min,
            log_interval=int(train_cfg.get("log_interval", 50)),
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
        )
        val_metrics = evaluate_fixed_protocol(
            model=model,
            loader=validation_loader,
            unique_refs=validation_patch_refs,
            protocol=validation_protocol,
            device=device,
            margin=float(train_cfg.get("margin", 1.0)),
            hard_negative_top_k=int(train_cfg.get("hard_negative_top_k", 3)),
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
        )
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            config,
            epoch,
            global_step,
            val_metrics,
            scaler=scaler,
        )
        # 固定协议的 FPR@TPR=95% 越低越好，以它选择 best checkpoint。
        current_fpr_at_tpr95 = val_metrics["fpr_at_tpr95"]
        if current_fpr_at_tpr95 <= best_fpr_at_tpr95:
            best_fpr_at_tpr95 = current_fpr_at_tpr95
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                config,
                epoch,
                global_step,
                val_metrics,
                scaler=scaler,
            )

        # 只有相对下降达到 min_delta，才重置早停计数。
        if early_stop_best_fpr_at_tpr95 == float("inf"):
            early_stop_best_fpr_at_tpr95 = current_fpr_at_tpr95
            no_improve_epochs = 0
            early_stop_message = "early_stop=init"
        else:
            improvement_threshold = early_stop_best_fpr_at_tpr95 * (1.0 - early_stop_min_relative)
            if current_fpr_at_tpr95 < improvement_threshold:
                early_stop_best_fpr_at_tpr95 = current_fpr_at_tpr95
                no_improve_epochs = 0
                early_stop_message = "early_stop=improved"
            else:
                no_improve_epochs += 1
                early_stop_message = f"early_stop=no_improve({no_improve_epochs}/{early_stop_patience})"

        metric_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_pos_dist": train_metrics["pos_dist"],
            "train_neg_dist": train_metrics["neg_dist"],
            **{f"val_{name}": value for name, value in val_metrics.items()},
            "lr": train_metrics["lr"],
            "early_stop_best_fpr_at_tpr95": early_stop_best_fpr_at_tpr95,
            "no_improve_epochs": no_improve_epochs,
        }
        append_metrics(output_dir / "metrics.csv", metric_row)

        print(
            f"epoch={epoch} done train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_fpr_at_tpr95={current_fpr_at_tpr95:.6f} "
            f"val_tpr_at_fpr_1e_4={val_metrics['tpr_at_fpr_1e_4']:.6f} "
            f"val_recall_at_1={val_metrics['recall_at_1']:.4f} "
            f"best_for_stop={early_stop_best_fpr_at_tpr95:.6f} {early_stop_message}",
            flush=True,
        )
        if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
            print(
                f"early stopping: val_fpr_at_tpr95 has not improved by "
                f"{early_stop_min_relative * 100:.2f}% for {early_stop_patience} epochs.",
                flush=True,
            )
            break

    if not args.no_plot:
        plot_training_curves(output_dir / "metrics.csv", output_dir / "training_curves.png")
        print(f"saved training curves to {output_dir / 'training_curves.png'}", flush=True)


if __name__ == "__main__":
    main()
