"""根据查询图像质量自适应描述子候选和身份接受参数。"""

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
    "effective_identity_acceptance_threshold",
    # 保留 RANSAC 实际参数诊断字段；当前版本始终记录基础配置值。
    "effective_ransac_reproj_threshold",
    "effective_ransac_max_iters",
)


def adapt_query_matching_config(
    config: Mapping[str, Any], quality_score: Any, metric: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """生成查询专属配置副本，不改写共享配置、不累积放宽。

    ``strength = clip((normal_quality_min - Q) /
    (normal_quality_min - quality_floor), 0, 1)``。
    默认 Q>=60 不变，Q<=20 达到放宽上限；40 分只用于诊断分档。

    当前根据质量调整：
        1. 描述子绝对距离阈值；
        2. 描述子 ratio 阈值；
        3. 最终 identity 接受阈值。

    RANSAC 参数始终沿用基础配置。Hamming 的描述子参数写入
    ``matching.hamming``，L2 写入顶层 ``matching``。
    """

    cfg = dict(config.get("quality_adaptation", {}))
    normal = float(cfg.get("normal_quality_min", 60.0))
    low = float(cfg.get("low_quality_below", 40.0))
    floor = float(cfg.get("quality_floor", 20.0))
    ratio_increment = float(cfg.get("ratio_threshold_increment", 0.04))
    ratio_cap = float(cfg.get("ratio_threshold_cap", 0.97))
    abs_distance_multiplier_max = float(
        cfg.get("abs_distance_threshold_multiplier_max", 1.0)
    )
    identity_threshold_decrement = float(
        cfg.get("identity_threshold_decrement", 0.0)
    )
    identity_threshold_floor = float(
        cfg.get("identity_threshold_floor", 0.0)
    )
    if not all(
        math.isfinite(x)
        for x in (
            normal,
            low,
            floor,
            ratio_increment,
            ratio_cap,
            abs_distance_multiplier_max,
            identity_threshold_decrement,
            identity_threshold_floor,
        )
    ):
        raise ValueError("quality_adaptation 参数必须为有限数值")
    if not 0.0 <= floor < low < normal <= 100.0:
        raise ValueError(
            "quality_adaptation 要求 0 <= quality_floor < low_quality_below "
            "< normal_quality_min <= 100"
        )
    if (
        ratio_increment < 0.0
        or not 0.0 < ratio_cap < 1.0
        or abs_distance_multiplier_max < 1.0
        or identity_threshold_decrement < 0.0
        or not 0.0 <= identity_threshold_floor <= 1.0
    ):
        raise ValueError(
            "quality_adaptation 的 ratio 增量必须非负、ratio cap 在 (0,1)、"
            "绝对距离放大倍率必须至少为 1、身份阈值下降量必须非负且阈值下限在 [0,1]"
        )

    identification = dict(config.get("identification", {}))
    configured_identity_threshold = float(
        identification.get("match_score_threshold", 0.55)
    )
    if not 0.0 <= configured_identity_threshold <= 1.0:
        raise ValueError(
            "identification.match_score_threshold must be in [0,1], "
            f"got {configured_identity_threshold}"
        )
    if identity_threshold_floor > configured_identity_threshold:
        raise ValueError(
            "quality_adaptation.identity_threshold_floor cannot exceed "
            "identification.match_score_threshold"
        )

    score = quality_score_or_none(quality_score)
    band = (
        "unknown"
        if score is None
        else "normal"
        if score >= normal
        else "medium"
        if score >= low
        else "low"
    )
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
    if (
        not 0.0 < ratio < 1.0
        or not math.isfinite(absolute)
        or absolute <= 0.0
        or not math.isfinite(reproj)
        or reproj <= 0.0
        or iterations <= 0
    ):
        raise ValueError(
            "匹配 ratio、绝对距离阈值、RANSAC 容差与最大迭代次数必须有效"
        )

    identity_threshold = configured_identity_threshold
    if strength > 0.0:
        # 基础 ratio 若已大于 cap，不反向收紧它；cap 只限制新增放宽。
        ratio = max(ratio, min(ratio_cap, ratio + ratio_increment * strength))
        absolute *= 1.0 + (abs_distance_multiplier_max - 1.0) * strength
        identity_threshold = max(
            identity_threshold_floor,
            configured_identity_threshold
            - identity_threshold_decrement * strength,
        )
        if metric == "hamming":
            hamming.update(
                ratio_threshold=ratio,
                abs_distance_threshold=absolute,
            )
            matching["hamming"] = hamming
        else:
            matching.update(
                ratio_threshold=ratio,
                abs_distance_threshold=absolute,
            )

    identification["match_score_threshold"] = identity_threshold
    effective = {
        **config,
        "matching": matching,
        "identification": identification,
    }
    diagnostics = {
        "query_frontend_quality_score": score if score is not None else "",
        "query_quality_band": band,
        "quality_adaptation_applied": int(strength > 0.0),
        "quality_relaxation_strength": strength,
        "effective_ratio_threshold": ratio,
        "effective_abs_distance_threshold": absolute,
        "effective_identity_acceptance_threshold": identity_threshold,
        # RANSAC 暂时不做质量自适应，仅记录基础配置，保持输出字段兼容。
        "effective_ransac_reproj_threshold": reproj,
        "effective_ransac_max_iters": iterations,
    }
    return effective, diagnostics
