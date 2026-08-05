"""面向匹配部署的多指标 checkpoint 选择。"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEFAULT_CHECKPOINT_SELECTION = {
    "strategy": "matching_composite_v1",
    "weights": {
        "false_acceptance": 0.45,
        "low_fpr_recall": 0.20,
        "hard_negative_recall": 0.15,
        "mean_gap": 0.10,
        "tail_gap": 0.10,
    },
}


def normalize_checkpoint_selection_config(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """规范化并校验匹配导向的 checkpoint 选择配置。"""

    raw = value if isinstance(value, Mapping) else {}
    strategy = str(raw.get("strategy", DEFAULT_CHECKPOINT_SELECTION["strategy"])).strip().lower()
    if strategy != "matching_composite_v1":
        raise ValueError(
            "Unsupported checkpoint_selection.strategy: "
            f"{strategy!r}. Expected 'matching_composite_v1'."
        )
    raw_weights = raw.get("weights", {})
    if not isinstance(raw_weights, Mapping):
        raise ValueError("checkpoint_selection.weights must be a mapping.")
    weights = {
        name: float(raw_weights.get(name, default))
        for name, default in DEFAULT_CHECKPOINT_SELECTION["weights"].items()
    }
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights.values()):
        raise ValueError("checkpoint_selection weights must be finite and non-negative.")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("checkpoint_selection weights must have a positive sum.")
    weights = {name: weight / total for name, weight in weights.items()}
    return {"strategy": strategy, "weights": weights}


def _finite_metric(metrics: Mapping[str, Any], name: str) -> float | None:
    try:
        value = float(metrics[name])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _worst_case_metric(
    metrics: Mapping[str, Any],
    name: str,
    *,
    lower_is_better: bool,
) -> float | None:
    """优先采用同指/跨指两组的最差值；没有分组时回退到总体值。"""

    candidates = [
        _finite_metric(metrics, f"same_finger_{name}"),
        _finite_metric(metrics, f"cross_finger_{name}"),
    ]
    candidates = [value for value in candidates if value is not None]
    if not candidates:
        return _finite_metric(metrics, name)
    return min(candidates) if lower_is_better else max(candidates)


def _unit_interval(value: float | None, *, lower_is_better: bool = False) -> float:
    if value is None:
        return 0.0
    bounded = min(max(float(value), 0.0), 1.0)
    return 1.0 - bounded if lower_is_better else bounded


def checkpoint_selection_metrics(
    validation_metrics: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """从一次验证结果生成可解释、越高越好的综合选择分数。

    选择目标对应部署中的三层行为：误接受安全性、低误报区间的召回，
    以及正负描述子距离的可分性。same/cross-finger 使用最差分项，避免
    某一类负样本变差时被另一类样本的平均值掩盖。
    """

    selection_config = normalize_checkpoint_selection_config(
        config.get("checkpoint_selection") if isinstance(config, Mapping) else None
    )
    weights = selection_config["weights"]
    descriptor_metric = str(
        config.get("descriptor_metric", "l2") if isinstance(config, Mapping) else "l2"
    ).strip().lower()
    distance_cap = 1.0 if descriptor_metric in {"hamming", "binary_hamming"} else 2.0

    fpr = _worst_case_metric(
        validation_metrics,
        "fpr_at_tpr95",
        lower_is_better=True,
    )
    low_fpr_recall = _worst_case_metric(
        validation_metrics,
        "tpr_at_fpr_1e_4",
        lower_is_better=False,
    )
    hard_negative_recall = _worst_case_metric(
        validation_metrics,
        "recall_at_1",
        lower_is_better=False,
    )
    positive_mean = _finite_metric(validation_metrics, "pos_mean")
    negative_mean = _worst_case_metric(
        validation_metrics,
        "neg_mean",
        lower_is_better=False,
    )
    mean_gap = (
        None
        if positive_mean is None or negative_mean is None
        else (negative_mean - positive_mean) / distance_cap
    )
    positive_p95 = _finite_metric(validation_metrics, "pos_p95")
    negative_p01 = _worst_case_metric(
        validation_metrics,
        "neg_p01",
        lower_is_better=False,
    )
    tail_gap = (
        None
        if positive_p95 is None or negative_p01 is None
        else (negative_p01 - positive_p95) / distance_cap
    )

    components = {
        "false_acceptance": _unit_interval(fpr, lower_is_better=True),
        "low_fpr_recall": _unit_interval(low_fpr_recall),
        "hard_negative_recall": _unit_interval(hard_negative_recall),
        "mean_gap": _unit_interval(mean_gap),
        "tail_gap": _unit_interval(tail_gap),
    }
    score = sum(weights[name] * components[name] for name in weights)
    result = {
        "checkpoint_selection_score": float(score),
        **{
            f"checkpoint_selection_{name}": float(value)
            for name, value in components.items()
        },
    }
    return result


def checkpoint_selection_state_from_metrics(
    metrics_path: Path,
) -> tuple[float, float, int]:
    """从 metrics.csv 恢复综合分数、早停分数和未改善计数。"""

    if not metrics_path.exists():
        return float("-inf"), float("-inf"), 0
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return float("-inf"), float("-inf"), 0
    scores = [
        float(row["val_checkpoint_selection_score"])
        for row in rows
        if row.get("val_checkpoint_selection_score") not in (None, "")
        and math.isfinite(float(row["val_checkpoint_selection_score"]))
    ]
    if not scores:
        raise ValueError(
            "Existing metrics.csv has no val_checkpoint_selection_score. "
            "Start a new output directory; old single-metric runs cannot be resumed "
            "under matching_composite_v1."
        )
    last = rows[-1]
    early_best = float(
        last.get("early_stop_best_selection_score")
        or max(scores)
    )
    no_improve = int(float(last.get("no_improve_epochs") or 0))
    return max(scores), early_best, no_improve