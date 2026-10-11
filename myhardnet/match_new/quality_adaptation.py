"""只根据查询的前端图像质量连续调整候选 ratio 和 RANSAC 容差。"""

from __future__ import annotations

import math
from typing import Any, Mapping

from match_new.frontend_data import quality_score_or_none


QUALITY_RESULT_FIELDS = (
    "query_frontend_quality_score",
    "query_quality_band",
    "quality_adaptation_applied",
    "quality_relaxation_strength",
    "effective_ratio_threshold",
    "effective_abs_distance_threshold",
    "effective_ransac_reproj_threshold",
    "effective_ransac_max_iters",
)


def adapt_query_matching_config(
    config: Mapping[str, Any], quality_score: Any, metric: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """生成查询专属配置副本，不改写共享配置、不累积放宽、不改变接受阈值。

    strength = clip((normal_quality_min - Q) / (normal_quality_min - quality_floor), 0, 1).
    默认 Q>=60 不变，Q<=20 达到放宽上限；40 分只用于诊断分档。
    Hamming 的 ratio 写入 matching.hamming，L2 写入 matching 顶层。
    """

    cfg = dict(config.get("quality_adaptation", {}))
    normal = float(cfg.get("normal_quality_min", 60.0))
    low = float(cfg.get("low_quality_below", 40.0))
    floor = float(cfg.get("quality_floor", 20.0))
    ratio_increment = float(cfg.get("ratio_threshold_increment", 0.04))
    ratio_cap = float(cfg.get("ratio_threshold_cap", 0.97))
    reproj_max = float(cfg.get("ransac_reproj_multiplier_max", 2.0))
    iters_increment = int(cfg.get("ransac_max_iters_increment", 2000))
    if not all(math.isfinite(x) for x in (normal, low, floor, ratio_increment, ratio_cap, reproj_max)):
        raise ValueError("quality_adaptation 参数必须为有限数值")
    if not 0.0 <= floor < low < normal <= 100.0:
        raise ValueError("quality_adaptation 要求 0 <= quality_floor < low_quality_below < normal_quality_min <= 100")
    if ratio_increment < 0.0 or not 0.0 < ratio_cap < 1.0 or reproj_max < 1.0 or iters_increment < 0:
        raise ValueError("quality_adaptation 的增量必须非负、ratio cap 在 (0,1)、重投影倍率至少为 1")

    score = quality_score_or_none(quality_score)
    band = "unknown" if score is None else "normal" if score >= normal else "medium" if score >= low else "low"
    strength = (
        min(1.0, max(0.0, (normal - score) / (normal - floor)))
        if bool(cfg.get("enabled", False)) and score is not None
        else 0.0
    )
    matching = dict(config.get("matching", {}))
    hamming = dict(matching.get("hamming", {}))
    metric_cfg = {**matching, **hamming} if metric == "hamming" else matching
    ratio = float(metric_cfg.get("ratio_threshold", 0.95))
    absolute = float(metric_cfg.get("abs_distance_threshold", 1.10))
    reproj = float(metric_cfg.get("ransac_reproj_threshold", 5.0))
    iterations = int(metric_cfg.get("ransac_max_iters", 3000))
    if not 0.0 < ratio < 1.0 or not math.isfinite(reproj) or reproj <= 0.0 or iterations <= 0:
        raise ValueError("匹配 ratio 必须在 (0,1)，RANSAC 容差与迭代次数必须为正")
    if strength > 0.0:
        # 基础 ratio 若已大于 cap，不反向收紧它；cap 只限制新增放宽。
        ratio = max(ratio, min(ratio_cap, ratio + ratio_increment * strength))
        reproj *= 1.0 + (reproj_max - 1.0) * strength
        iterations += int(round(iters_increment * strength))
        if metric == "hamming":
            hamming.update(
                ratio_threshold=ratio,
                ransac_reproj_threshold=reproj,
                ransac_max_iters=iterations,
            )
            matching["hamming"] = hamming
        else:
            matching.update(
                ratio_threshold=ratio,
                ransac_reproj_threshold=reproj,
                ransac_max_iters=iterations,
            )
    effective = {**config, "matching": matching}
    diagnostics = {
        "query_frontend_quality_score": score if score is not None else "",
        "query_quality_band": band,
        "quality_adaptation_applied": int(strength > 0.0),
        "quality_relaxation_strength": strength,
        "effective_ratio_threshold": ratio,
        "effective_abs_distance_threshold": absolute,
        "effective_ransac_reproj_threshold": reproj,
        "effective_ransac_max_iters": iterations,
    }
    return effective, diagnostics
