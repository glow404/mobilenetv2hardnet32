"""独立 residual binary descriptor 训练入口。"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

from hardnet_train.binary_loss import BinaryDescriptorLoss
from hardnet_train.binary_model import (
    BinaryDescriptorModel,
    binary_model_metadata,
    build_binary_descriptor_model,
)
from hardnet_train.binary_validation import evaluate_binary_fixed_protocol
from hardnet_train.data import FingerprintPairDataset
from hardnet_train.loss import normalize_hard_negative_strategy
from hardnet_train.metrics import RunningMean
from hardnet_train.model import (
    build_descriptor_model,
    checkpoint_descriptor_dim,
    checkpoint_model_architecture,
    checkpoint_model_config,
    count_parameters,
    normalize_model_architecture,
)
from hardnet_train.optim import (
    build_optimizer,
    normalize_optimizer_name,
    optimizer_name,
)
from hardnet_train.train import (
    append_metrics,
    best_fpr_at_tpr95_from_metrics,
    configure_cuda,
    load_config,
    make_loader,
    move_optimizer_state_to_device,
    resolve_amp,
    resolve_device,
    resolve_path,
    resolve_resume_checkpoint,
    scheduled_lr,
    set_optimizer_lr,
    set_seed,
)
from hardnet_train.validation import (
    build_fixed_validation_protocol,
    make_validation_patch_loader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train residual binary fingerprint descriptors."
    )
    parser.add_argument("--config", default="hardnet_train/config_binary_256.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hash-bits", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--backbone-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--resume-auto", action="store_true")
    return parser.parse_args()


def _configured_optional_int(value: Any) -> int | None:
    text = str(value if value is not None else "").strip().lower()
    return None if text in {"", "auto", "none"} else int(text)


def load_float_backbone(
    config: dict[str, Any],
) -> tuple[torch.nn.Module, Path, Mapping[str, Any]]:
    """按 checkpoint 事实元数据重建浮点 teacher，并拒绝二值 checkpoint。"""

    backbone_cfg = config.setdefault("backbone", {})
    checkpoint_value = backbone_cfg.get("checkpoint")
    if not checkpoint_value:
        raise ValueError("backbone.checkpoint is required for binary descriptor training.")
    checkpoint_path = resolve_path(config, checkpoint_value)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Unsupported backbone checkpoint format: {checkpoint_path}")
    kind = str(checkpoint.get("descriptor_kind", "float")).strip().lower()
    metric = str(checkpoint.get("descriptor_metric", "l2")).strip().lower()
    if kind != "float" or metric != "l2":
        raise ValueError(
            "Binary training requires a continuous float/L2 teacher checkpoint: "
            f"kind={kind}, metric={metric}, path={checkpoint_path}."
        )

    saved_architecture = checkpoint_model_architecture(checkpoint)
    requested_architecture = str(
        backbone_cfg.get("architecture", "auto")
    ).strip().lower()
    if requested_architecture not in {"", "auto"}:
        requested_architecture = normalize_model_architecture(requested_architecture)
        if requested_architecture != saved_architecture:
            raise ValueError(
                "Configured backbone architecture does not match checkpoint: "
                f"config={requested_architecture}, checkpoint={saved_architecture}."
            )

    saved_dimension = checkpoint_descriptor_dim(checkpoint)
    requested_dimension = _configured_optional_int(
        backbone_cfg.get("descriptor_dim", "auto")
    )
    if requested_dimension is not None and requested_dimension != saved_dimension:
        raise ValueError(
            "Configured float descriptor dimension does not match checkpoint: "
            f"config={requested_dimension}, checkpoint={saved_dimension}."
        )

    model_config = checkpoint_model_config(checkpoint)
    model_config["architecture"] = saved_architecture
    model_config["descriptor_dim"] = saved_dimension
    backbone = build_descriptor_model(model_config)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    backbone.load_state_dict(state)

    backbone_cfg["checkpoint"] = str(checkpoint_path)
    backbone_cfg["architecture"] = saved_architecture
    backbone_cfg["descriptor_dim"] = saved_dimension
    return backbone, checkpoint_path, checkpoint


def quantization_temperature(
    start: float,
    end: float,
    step: int,
    total_steps: int,
) -> float:
    """在训练过程中按指数曲线从软 tanh 平滑退火到接近 sign。"""

    if start <= 0.0 or end <= 0.0:
        raise ValueError("Quantization temperatures must be positive.")
    if total_steps <= 1:
        return float(end)
    progress = min(max(float(step) / float(total_steps - 1), 0.0), 1.0)
    return float(start) * math.pow(float(end) / float(start), progress)


def train_binary_one_epoch(
    model: BinaryDescriptorModel,
    criterion: BinaryDescriptorLoss,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_steps: int,
    global_step: int,
    base_lr: float,
    scheduler: str,
    warmup_steps: int,
    eta_min: float,
    temperature_start: float,
    temperature_end: float,
    log_interval: int,
    grad_clip_norm: float,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    channels_last: bool,
) -> tuple[dict[str, float], int]:
    """训练一个二值 epoch；网络前向可混合精度，组合损失保持 FP32。"""

    model.train()
    meter_names = (
        "loss",
        "loss_metric",
        "loss_teacher_similarity",
        "loss_positive_consistency",
        "loss_quantization",
        "loss_balance",
        "loss_decorrelation",
        "positive_hamming",
        "mean_abs_bit_balance",
        "mean_abs_continuous",
        "pos_dist",
        "neg_dist",
    )
    meters = {name: RunningMean() for name in meter_names}
    started = time.time()
    lr = base_lr
    temperature = temperature_start

    for step, batch in enumerate(loader, start=1):
        lr = scheduled_lr(
            base_lr,
            global_step,
            total_steps,
            scheduler,
            warmup_steps,
            eta_min,
        )
        set_optimizer_lr(optimizer, lr)
        temperature = quantization_temperature(
            temperature_start,
            temperature_end,
            global_step,
            total_steps,
        )
        model.set_quantization_temperature(temperature)

        anchor = batch["anchor"].to(device, non_blocking=True)
        positive = batch["positive"].to(device, non_blocking=True)
        point_group = batch["point_group"].to(device, non_blocking=True)
        finger_group = batch["finger_group"].to(device, non_blocking=True)
        if channels_last:
            anchor = anchor.contiguous(memory_format=torch.channels_last)
            positive = positive.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            anchor_features = model.forward_features(anchor)
            positive_features = model.forward_features(positive)
        loss, stats = criterion(
            anchor_features,
            positive_features,
            point_group=point_group,
            finger_group=finger_group,
        )
        scaler.scale(loss).backward()
        if grad_clip_norm > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=grad_clip_norm,
            )
        scaler.step(optimizer)
        scaler.update()

        batch_size = int(anchor.shape[0])
        for name in meter_names:
            if name == "neg_dist":
                valid = stats["valid_triplets"]
                if torch.any(valid):
                    meters[name].update(
                        float(stats[name][valid].mean().item()),
                        int(valid.sum().item()),
                    )
                continue
            value = stats[name]
            scalar = value.mean() if value.ndim > 0 else value
            meters[name].update(float(scalar.item()), batch_size)

        if log_interval > 0 and step % log_interval == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"epoch={epoch} step={step}/{len(loader)} lr={lr:.6g} "
                f"temperature={temperature:.4f} loss={meters['loss'].value:.4f} "
                f"metric={meters['loss_metric'].value:.4f} "
                f"quant={meters['loss_quantization'].value:.4f} "
                f"pos_hamming={meters['positive_hamming'].value:.4f} "
                f"samples/s={meters['loss'].count / elapsed:.1f}",
                flush=True,
            )
        global_step += 1

    return {
        **{name: meter.value for name, meter in meters.items()},
        "lr": float(lr),
        "temperature": float(temperature),
    }, global_step


def save_binary_checkpoint(
    path: Path,
    model: BinaryDescriptorModel,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    backbone_checkpoint_path: Path,
    backbone_checkpoint: Mapping[str, Any],
    epoch: int,
    global_step: int,
    metrics: dict[str, float],
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """保存自包含模型，并持久化二值解释所需的完整元数据。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        key: value for key, value in config.items() if key != "_config_path"
    }
    metadata = binary_model_metadata(model)
    payload: dict[str, Any] = {
        "checkpoint_format_version": 1,
        "training_task": "binary_descriptor",
        "epoch": int(epoch),
        "global_step": int(global_step),
        **metadata,
        "quantization_method": "tanh_ste_sign",
        "quantization_temperature": model.quantization_temperature,
        "backbone_checkpoint": str(backbone_checkpoint_path),
        "backbone_checkpoint_epoch": backbone_checkpoint.get("epoch"),
        "model_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameter_count": count_parameters(model),
        "hash_head_parameter_count": sum(
            parameter.numel() for parameter in model.hash_head.parameters()
        ),
        "model": model.state_dict(),
        "hash_head": model.hash_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_name": optimizer_name(optimizer),
        "metrics": metrics,
        "config": resolved_config,
        "resolved_config": resolved_config,
    }
    if scaler is not None and scaler.is_enabled():
        payload["amp_scaler"] = scaler.state_dict()
    torch.save(payload, path)


def load_binary_checkpoint(
    checkpoint_path: Path,
    model: BinaryDescriptorModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    steps_per_epoch: int,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, int]:
    """恢复二值训练，并严格校验 backbone/hash 契约。"""

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Unsupported binary checkpoint format: {checkpoint_path}")
    if checkpoint.get("training_task") != "binary_descriptor":
        raise ValueError(
            f"Checkpoint is not a binary descriptor training checkpoint: {checkpoint_path}"
        )
    expected = binary_model_metadata(model)
    for field in (
        "model_architecture",
        "descriptor_kind",
        "descriptor_metric",
        "hash_bits",
        "hash_head_architecture",
        "hash_head_input_dim",
        "hash_head_hidden_dim",
        "hash_head_dropout",
        "binary_encoding",
        "binary_storage",
        "binary_bitorder",
        "backbone_architecture",
        "float_descriptor_dim",
        "backbone_trainable",
    ):
        if checkpoint.get(field) != expected[field]:
            raise ValueError(
                f"Binary checkpoint metadata mismatch for {field}: "
                f"checkpoint={checkpoint.get(field)!r}, model={expected[field]!r}."
            )
    model.load_state_dict(checkpoint["model"])

    saved_optimizer_name = checkpoint.get("optimizer_name")
    if saved_optimizer_name is not None:
        saved_name = normalize_optimizer_name(str(saved_optimizer_name))
        if saved_name != optimizer_name(optimizer):
            raise ValueError(
                "Binary checkpoint optimizer mismatch: "
                f"checkpoint={saved_name}, configured={optimizer_name(optimizer)}."
            )
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
    if scaler is not None and scaler.is_enabled() and "amp_scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["amp_scaler"])

    finished_epoch = int(checkpoint.get("epoch", 0))
    global_step = int(
        checkpoint.get("global_step", finished_epoch * steps_per_epoch)
    )
    return finished_epoch + 1, global_step


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    train_cfg = config.setdefault("training", {})
    binary_cfg = config.setdefault("binary_model", {})
    loss_cfg = config.setdefault("binary_loss", {})
    optim_cfg = config.setdefault("optimizer", {})
    validation_cfg = config.setdefault("validation", {})
    backbone_cfg = config.setdefault("backbone", {})

    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    if args.steps_per_epoch is not None:
        train_cfg["steps_per_epoch"] = int(args.steps_per_epoch)
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.hash_bits is not None:
        binary_cfg["hash_bits"] = int(args.hash_bits)
    if args.lr is not None:
        optim_cfg["lr"] = float(args.lr)
    if args.device is not None:
        train_cfg["device"] = str(args.device)
    if args.backbone_checkpoint is not None:
        backbone_cfg["checkpoint"] = str(args.backbone_checkpoint)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)

    epochs = int(train_cfg.get("epochs", 100))
    steps_per_epoch = int(train_cfg.get("steps_per_epoch", 1000))
    batch_size = int(train_cfg.get("batch_size", 256))
    if epochs <= 0 or steps_per_epoch <= 0 or batch_size <= 0:
        raise ValueError("epochs, steps_per_epoch and batch_size must be positive.")
    train_cfg["epochs"] = epochs
    train_cfg["steps_per_epoch"] = steps_per_epoch
    train_cfg["batch_size"] = batch_size
    train_cfg["hard_negative_strategy"] = normalize_hard_negative_strategy(
        train_cfg.get("hard_negative_strategy", "same_finger_allowed")
    )
    train_cfg["hard_negative_top_k"] = int(
        train_cfg.get("hard_negative_top_k", 3)
    )
    if train_cfg["hard_negative_top_k"] < 1:
        raise ValueError("training.hard_negative_top_k must be >= 1.")
    validation_protocol_name = str(
        validation_cfg.get("protocol", "fixed_pairs_v1")
    ).strip().lower()
    if validation_protocol_name != "fixed_pairs_v1":
        raise ValueError(
            f"Unsupported validation.protocol: {validation_protocol_name!r}."
        )
    validation_cfg["protocol"] = validation_protocol_name
    binary_cfg["hash_bits"] = int(binary_cfg.get("hash_bits", 256))
    binary_cfg["backbone_trainable"] = bool(
        binary_cfg.get("backbone_trainable", False)
    )
    optim_cfg["name"] = normalize_optimizer_name(optim_cfg.get("name", "adamw"))
    optim_cfg["lr"] = float(optim_cfg.get("lr", 1e-3))
    optim_cfg["weight_decay"] = float(optim_cfg.get("weight_decay", 1e-4))

    device = resolve_device(str(train_cfg.get("device", "auto")))
    configure_cuda(device, train_cfg)
    amp_enabled, amp_dtype = resolve_amp(
        device,
        str(train_cfg.get("mixed_precision", "fp16")),
    )
    channels_last = device.type == "cuda" and bool(
        train_cfg.get("channels_last", True)
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )

    output_dir = resolve_path(
        config,
        config.get("output_dir", "../outputs/models/hardnet_binary_256_v1"),
    )
    resume_arg = "auto" if args.resume_auto else args.resume
    resume_path = resolve_resume_checkpoint(
        output_dir,
        resume_arg,
        allow_missing=args.resume_auto,
    )
    metrics_path = output_dir / "metrics.csv"
    if metrics_path.exists() and resume_path is None:
        raise FileExistsError(
            f"Output directory already contains metrics: {metrics_path}. "
            "Use a new output directory or explicitly resume it."
        )

    backbone, backbone_checkpoint_path, backbone_checkpoint = load_float_backbone(
        config
    )
    model = build_binary_descriptor_model(backbone, binary_cfg).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    criterion = BinaryDescriptorLoss.from_config(train_cfg, loss_cfg)
    optimizer = build_optimizer(model, optim_cfg)

    train_loader = make_loader(
        config,
        "train",
        batch_size=batch_size,
        steps=steps_per_epoch,
        seed=seed,
    )
    data_cfg = config["data"]
    validation_dataset = FingerprintPairDataset(
        csv_path=resolve_path(config, data_cfg["val_csv"]),
        max_rows=data_cfg.get("max_val_rows"),
        max_rows_per_finger=data_cfg.get("max_val_rows_per_finger"),
        normalize=bool(data_cfg.get("normalize", True)),
    )
    validation_cfg["seed"] = int(validation_cfg.get("seed", seed + 10_000))
    validation_cfg["positive_count"] = int(
        validation_cfg.get("positive_count", 16_384)
    )
    validation_cfg["same_finger_negatives_per_anchor"] = int(
        validation_cfg.get("same_finger_negatives_per_anchor", 32)
    )
    validation_cfg["cross_finger_negatives_per_anchor"] = int(
        validation_cfg.get("cross_finger_negatives_per_anchor", 32)
    )
    validation_cfg["batch_size"] = int(
        validation_cfg.get("batch_size", batch_size)
    )
    validation_protocol = build_fixed_validation_protocol(
        records=validation_dataset.records,
        positive_count=validation_cfg["positive_count"],
        same_finger_negatives_per_anchor=validation_cfg[
            "same_finger_negatives_per_anchor"
        ],
        cross_finger_negatives_per_anchor=validation_cfg[
            "cross_finger_negatives_per_anchor"
        ],
        seed=validation_cfg["seed"],
    )
    validation_loader, validation_patch_refs = make_validation_patch_loader(
        dataset=validation_dataset,
        protocol=validation_protocol,
        batch_size=validation_cfg["batch_size"],
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
    )

    scheduler = str(train_cfg.get("scheduler", "warmup_cosine"))
    warmup_steps = int(
        round(float(train_cfg.get("warmup_epochs", 2.0)) * steps_per_epoch)
    )
    eta_min = float(train_cfg.get("eta_min", 1e-5))
    total_steps = max(1, epochs * steps_per_epoch)
    temperature_start = float(binary_cfg.get("temperature_start", 1.0))
    temperature_end = float(binary_cfg.get("temperature_end", 0.1))
    if temperature_start <= 0.0 or temperature_end <= 0.0:
        raise ValueError("binary_model temperatures must be positive.")
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 5.0))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 0))
    early_stop_min_relative = float(
        train_cfg.get("early_stop_min_relative_improvement", 0.002)
    )

    best_fpr, early_best_fpr, no_improve_epochs = (
        best_fpr_at_tpr95_from_metrics(metrics_path)
        if resume_path is not None
        else (float("inf"), float("inf"), 0)
    )
    start_epoch = 1
    global_step = 0
    if resume_path is not None:
        start_epoch, global_step = load_binary_checkpoint(
            resume_path,
            model,
            optimizer,
            device,
            steps_per_epoch,
            scaler,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(
            {key: value for key, value in config.items() if key != "_config_path"},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = binary_model_metadata(model)
    print(
        f"device={device} backbone={metadata['backbone_architecture']} "
        f"float_dim={metadata['float_descriptor_dim']} hash_bits={metadata['hash_bits']} "
        f"backbone_trainable={metadata['backbone_trainable']} "
        f"trainable_params={count_parameters(model)} optimizer={optimizer_name(optimizer)} "
        f"lr={optim_cfg['lr']} batch={batch_size} epochs={epochs} "
        f"steps_per_epoch={steps_per_epoch} temperature={temperature_start}->{temperature_end}",
        flush=True,
    )

    for epoch in range(start_epoch, epochs + 1):
        train_metrics, global_step = train_binary_one_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_steps=total_steps,
            global_step=global_step,
            base_lr=optim_cfg["lr"],
            scheduler=scheduler,
            warmup_steps=warmup_steps,
            eta_min=eta_min,
            temperature_start=temperature_start,
            temperature_end=temperature_end,
            log_interval=int(train_cfg.get("log_interval", 50)),
            grad_clip_norm=grad_clip_norm,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
        )
        val_metrics = evaluate_binary_fixed_protocol(
            model=model,
            loader=validation_loader,
            unique_refs=validation_patch_refs,
            protocol=validation_protocol,
            device=device,
            margin=float(validation_cfg.get("margin", 0.2)),
            hard_negative_top_k=train_cfg["hard_negative_top_k"],
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
        )
        save_binary_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            config,
            backbone_checkpoint_path,
            backbone_checkpoint,
            epoch,
            global_step,
            val_metrics,
            scaler,
        )
        current_fpr = float(val_metrics["fpr_at_tpr95"])
        if current_fpr <= best_fpr:
            best_fpr = current_fpr
            save_binary_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                config,
                backbone_checkpoint_path,
                backbone_checkpoint,
                epoch,
                global_step,
                val_metrics,
                scaler,
            )

        if early_best_fpr == float("inf"):
            early_best_fpr = current_fpr
            no_improve_epochs = 0
        elif current_fpr < early_best_fpr * (1.0 - early_stop_min_relative):
            early_best_fpr = current_fpr
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        metric_row = {
            "epoch": epoch,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"val_{name}": value for name, value in val_metrics.items()},
            "early_stop_best_fpr_at_tpr95": early_best_fpr,
            "no_improve_epochs": no_improve_epochs,
        }
        append_metrics(metrics_path, metric_row)
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_fpr_at_tpr95={current_fpr:.6f} "
            f"val_eer={val_metrics['eer']:.6f} "
            f"val_bit_balance_error={val_metrics['bit_balance_error']:.6f}",
            flush=True,
        )
        if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
            print(
                f"early stop at epoch={epoch}: no improvement for "
                f"{no_improve_epochs} epochs",
                flush=True,
            )
            break


if __name__ == "__main__":
    main()