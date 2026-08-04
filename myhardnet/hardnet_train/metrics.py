"""训练运行均值与固定描述子验证指标。"""

from __future__ import annotations

import math

import torch


class RunningMean:
    """按样本数加权的运行均值。"""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        """加入一个均值观测值及其对应样本数。"""
        self.total += float(value) * int(count)
        self.count += int(count)

    @property
    def value(self) -> float:
        """返回当前累计均值；没有观测值时返回 0。"""
        if self.count == 0:
            return 0.0
        return self.total / self.count


def pair_l2_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """计算末维单位描述子的一一对应 L2 距离，支持前置维度广播。"""
    if left.shape[-1] != right.shape[-1]:
        raise ValueError(f"Descriptor dimension mismatch: {left.shape} vs {right.shape}")
    similarity = (left.float() * right.float()).sum(dim=-1)
    return torch.sqrt(torch.clamp(2.0 - 2.0 * similarity, min=0.0))


def _finite_flatten(values: torch.Tensor, name: str) -> torch.Tensor:
    flattened = values.detach().float().cpu().reshape(-1)
    if flattened.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if not torch.all(torch.isfinite(flattened)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return flattened


def _binary_roc(positive_dist: torch.Tensor, negative_dist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """返回按距离阈值递增的 `(FPR, TPR)`，相同距离按同一阈值处理。"""
    positives = _finite_flatten(positive_dist, "positive_dist")
    negatives = _finite_flatten(negative_dist, "negative_dist")
    distances = torch.cat([positives, negatives])
    labels = torch.cat(
        [
            torch.ones(positives.numel(), dtype=torch.int64),
            torch.zeros(negatives.numel(), dtype=torch.int64),
        ]
    )
    order = torch.argsort(distances, stable=True)
    sorted_distances = distances[order]
    sorted_labels = labels[order]

    # 每组相同距离只在组尾生成 ROC 点，避免人为规定并列样本先后顺序。
    group_end = torch.ones(sorted_distances.numel(), dtype=torch.bool)
    group_end[:-1] = sorted_distances[:-1] != sorted_distances[1:]
    true_positives = sorted_labels.cumsum(dim=0)[group_end].double()
    false_positives = (1 - sorted_labels).cumsum(dim=0)[group_end].double()
    tpr = true_positives / float(positives.numel())
    fpr = false_positives / float(negatives.numel())
    return torch.cat([torch.zeros(1, dtype=torch.double), fpr]), torch.cat(
        [torch.zeros(1, dtype=torch.double), tpr]
    )


def _eer(fpr: torch.Tensor, tpr: torch.Tensor) -> float:
    fnr = 1.0 - tpr
    difference = fpr - fnr
    crossing = torch.nonzero(difference >= 0.0, as_tuple=False)
    if crossing.numel() == 0:
        return float((fpr[-1] + fnr[-1]).item() / 2.0)
    right = int(crossing[0].item())
    if right == 0:
        return float((fpr[0] + fnr[0]).item() / 2.0)
    left = right - 1
    left_difference = float(difference[left].item())
    right_difference = float(difference[right].item())
    denominator = right_difference - left_difference
    weight = 0.0 if denominator == 0.0 else -left_difference / denominator
    interpolated_fpr = float(fpr[left].item()) + weight * float((fpr[right] - fpr[left]).item())
    interpolated_fnr = float(fnr[left].item()) + weight * float((fnr[right] - fnr[left]).item())
    return (interpolated_fpr + interpolated_fnr) / 2.0


def _distance_metrics(positive_dist: torch.Tensor, negative_dist: torch.Tensor) -> dict[str, float]:
    positives = _finite_flatten(positive_dist, "positive_dist")
    negatives = _finite_flatten(negative_dist, "negative_dist")
    fpr, tpr = _binary_roc(positives, negatives)

    at_tpr95 = torch.nonzero(tpr >= 0.95, as_tuple=False)
    fpr_at_tpr95 = float(fpr[int(at_tpr95[0].item())].item()) if at_tpr95.numel() else 1.0
    allowed_fpr = fpr <= 1e-4
    tpr_at_fpr_1e_4 = float(tpr[allowed_fpr].max().item()) if torch.any(allowed_fpr) else 0.0
    auc = float(torch.trapz(tpr, fpr).item())

    return {
        "fpr_at_tpr95": fpr_at_tpr95,
        "tpr_at_fpr_1e_4": tpr_at_fpr_1e_4,
        "roc_auc": auc,
        "eer": _eer(fpr, tpr),
        "neg_mean": float(negatives.mean().item()),
        "neg_p01": float(torch.quantile(negatives, 0.01).item()),
        "neg_p05": float(torch.quantile(negatives, 0.05).item()),
    }


def _ranking_metrics(
    positive_dist: torch.Tensor,
    negative_dist: torch.Tensor,
    margin: float,
    hard_negative_top_k: int,
) -> dict[str, float]:
    if negative_dist.ndim != 2 or positive_dist.ndim != 1:
        raise ValueError("Ranking metrics expect positive_dist=[N] and negative_dist=[N, K].")
    if negative_dist.shape[0] != positive_dist.shape[0]:
        raise ValueError("Positive and negative distance rows must have the same anchor count.")
    if hard_negative_top_k < 1:
        raise ValueError("hard_negative_top_k must be >= 1.")

    selected_k = min(int(hard_negative_top_k), int(negative_dist.shape[1]))
    hardest = torch.topk(negative_dist, k=selected_k, dim=1, largest=False).values
    violations = torch.clamp(float(margin) + positive_dist[:, None] - hardest, min=0.0)
    return {
        "recall_at_1": float((positive_dist < negative_dist.min(dim=1).values).float().mean().item()),
        "active_triplet_ratio": float((violations > 0.0).any(dim=1).float().mean().item()),
        "loss": float(violations.mean().item()),
    }


def _selected_ranking_metrics(
    positive_dist: torch.Tensor,
    negative_dist: torch.Tensor,
    negative_anchor_index: torch.Tensor,
    margin: float,
) -> dict[str, float]:
    """按每个 anchor 实际选中的可变数量 top-k 候选计算 ranking 指标。"""

    positives = _finite_flatten(positive_dist, "positive_dist")
    negatives = _finite_flatten(negative_dist, "negative_dist")
    anchor_index = negative_anchor_index.detach().long().cpu().reshape(-1)
    if anchor_index.numel() != negatives.numel():
        raise ValueError("negative_anchor_index must match negative_dist length.")
    if torch.any(anchor_index < 0) or torch.any(anchor_index >= positives.numel()):
        raise ValueError("negative_anchor_index contains an out-of-range anchor.")

    count = torch.zeros(positives.numel(), dtype=torch.long)
    count.index_add_(0, anchor_index, torch.ones_like(anchor_index))
    included = count > 0
    if not torch.any(included):
        return {
            "recall_at_1": math.nan,
            "active_triplet_ratio": math.nan,
            "loss": math.nan,
        }

    positive_for_negative = positives[anchor_index]
    violations = torch.clamp(
        float(margin) + positive_for_negative - negatives,
        min=0.0,
    )
    violation_sum = torch.zeros(positives.numel(), dtype=torch.float32)
    violation_sum.index_add_(0, anchor_index, violations)
    per_anchor_loss = violation_sum / count.clamp_min(1)

    active_count = torch.zeros(positives.numel(), dtype=torch.long)
    active_count.index_add_(0, anchor_index, violations.gt(0.0).long())

    min_negative = torch.full(
        (positives.numel(),),
        float("inf"),
        dtype=torch.float32,
    )
    min_negative.scatter_reduce_(
        0,
        anchor_index,
        negatives,
        reduce="amin",
        include_self=True,
    )
    return {
        "recall_at_1": float(
            (positives[included] < min_negative[included]).float().mean().item()
        ),
        "active_triplet_ratio": float(
            active_count[included].gt(0).float().mean().item()
        ),
        "loss": float(per_anchor_loss[included].mean().item()),
    }


def in_batch_descriptor_validation_metrics(
    positive_dist: torch.Tensor,
    negative_dist: torch.Tensor,
    negative_anchor_index: torch.Tensor,
    negative_is_same_finger: torch.Tensor,
    margin: float,
    *,
    valid_anchor_count: int,
    skipped_anchor_count: int,
) -> dict[str, float]:
    """汇总固定 batch 计划中实际被训练规则选中的难负样本。"""

    positives = _finite_flatten(positive_dist, "positive_dist")
    negatives = _finite_flatten(negative_dist, "negative_dist")
    anchor_index = negative_anchor_index.detach().long().cpu().reshape(-1)
    same_mask = negative_is_same_finger.detach().bool().cpu().reshape(-1)
    if positives.numel() != int(valid_anchor_count):
        raise ValueError(
            "positive_dist length must equal valid_anchor_count: "
            f"{positives.numel()} vs {valid_anchor_count}."
        )
    if anchor_index.numel() != negatives.numel():
        raise ValueError("negative_anchor_index must match negative_dist length.")
    if same_mask.numel() != negatives.numel():
        raise ValueError("negative_is_same_finger must match negative_dist length.")

    result = {
        "pos_mean": float(positives.mean().item()),
        "pos_p95": float(torch.quantile(positives, 0.95).item()),
        "pos_p99": float(torch.quantile(positives, 0.99).item()),
        "valid_anchor_count": float(valid_anchor_count),
        "skipped_anchor_count": float(skipped_anchor_count),
        "valid_anchor_ratio": float(valid_anchor_count)
        / float(valid_anchor_count + skipped_anchor_count),
        "selected_negative_count": float(negatives.numel()),
        "same_finger_selected_negative_count": float(same_mask.sum().item()),
        "cross_finger_selected_negative_count": float((~same_mask).sum().item()),
    }
    result.update(_distance_metrics(positives, negatives))
    result.update(
        _selected_ranking_metrics(
            positives,
            negatives,
            anchor_index,
            margin,
        )
    )

    for prefix, mask in (
        ("same_finger", same_mask),
        ("cross_finger", ~same_mask),
    ):
        if not torch.any(mask):
            group_metrics = {
                "fpr_at_tpr95": math.nan,
                "tpr_at_fpr_1e_4": math.nan,
                "roc_auc": math.nan,
                "eer": math.nan,
                "neg_mean": math.nan,
                "neg_p01": math.nan,
                "neg_p05": math.nan,
                "recall_at_1": math.nan,
                "active_triplet_ratio": math.nan,
                "loss": math.nan,
            }
        else:
            group_negatives = negatives[mask]
            group_metrics = _distance_metrics(positives, group_negatives)
            group_metrics.update(
                _selected_ranking_metrics(
                    positives,
                    group_negatives,
                    anchor_index[mask],
                    margin,
                )
            )
        result.update(
            {f"{prefix}_{name}": value for name, value in group_metrics.items()}
        )
    return result


def descriptor_validation_metrics(
    positive_dist: torch.Tensor,
    same_finger_negative_dist: torch.Tensor,
    cross_finger_negative_dist: torch.Tensor,
    margin: float,
    hard_negative_top_k: int = 3,
) -> dict[str, float]:
    """计算固定协议总指标，以及 same-finger / cross-finger 分项指标。"""
    positives = _finite_flatten(positive_dist, "positive_dist")
    if positive_dist.ndim != 1:
        raise ValueError("positive_dist must be one-dimensional.")
    if same_finger_negative_dist.ndim != 2 or cross_finger_negative_dist.ndim != 2:
        raise ValueError("Negative distances must be shaped [anchor_count, negatives_per_anchor].")
    if same_finger_negative_dist.shape[0] != positives.numel():
        raise ValueError("same-finger negative rows must match positive count.")
    if cross_finger_negative_dist.shape[0] != positives.numel():
        raise ValueError("cross-finger negative rows must match positive count.")
    if not torch.all(torch.isfinite(same_finger_negative_dist)):
        raise ValueError("same_finger_negative_dist contains NaN or infinite values.")
    if not torch.all(torch.isfinite(cross_finger_negative_dist)):
        raise ValueError("cross_finger_negative_dist contains NaN or infinite values.")

    combined = torch.cat([same_finger_negative_dist, cross_finger_negative_dist], dim=1).float().cpu()
    same = same_finger_negative_dist.float().cpu()
    cross = cross_finger_negative_dist.float().cpu()

    result = {
        "pos_mean": float(positives.mean().item()),
        "pos_p95": float(torch.quantile(positives, 0.95).item()),
        "pos_p99": float(torch.quantile(positives, 0.99).item()),
    }
    result.update(_distance_metrics(positives, combined.reshape(-1)))
    result.update(_ranking_metrics(positives, combined, margin, hard_negative_top_k))

    for prefix, negatives in (("same_finger", same), ("cross_finger", cross)):
        group_metrics = _distance_metrics(positives, negatives.reshape(-1))
        group_metrics.update(_ranking_metrics(positives, negatives, margin, hard_negative_top_k))
        result.update({f"{prefix}_{name}": value for name, value in group_metrics.items()})
    return result


def fpr_at_recall(positive_dist: torch.Tensor, negative_dist: torch.Tensor, recall: float = 0.95) -> float:
    """兼容旧调用的 FPR@TPR；新日志字段使用明确的 `fpr_at_tpr95`。"""
    if not 0.0 <= float(recall) <= 1.0:
        raise ValueError(f"recall must be in [0, 1], got {recall}.")
    try:
        fpr, tpr = _binary_roc(positive_dist, negative_dist)
    except ValueError:
        return math.nan
    indices = torch.nonzero(tpr >= float(recall), as_tuple=False)
    return float(fpr[int(indices[0].item())].item()) if indices.numel() else 1.0