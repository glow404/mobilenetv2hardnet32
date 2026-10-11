"""离线全量阈值评估使用的 identity 级匹配实现。

一个已注册手指由多张图像模板组成。本模块逐一计算查询模板与该 identity 全部
注册模板的图像级分数，再按配置融合为 identity 级分数。所有模板均参与评分，
以便阈值曲线和 FAR/FRR 使用完整分数。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from match_new.hardnet_matcher import match_templates_descriptor
from match_new.quality_adaptation import QUALITY_RESULT_FIELDS
from match_new.template_builder import load_image_template


def load_template_cached(
    path: str | Path,
    cache: dict[str, dict[str, Any]],
    *,
    require: str | None = None,
) -> dict[str, Any]:
    """加载一份图像模板并按路径和描述子类型约束缓存。"""

    key = str(Path(path).expanduser())
    cache_key = f"{key}::{require or 'any'}"
    if cache_key not in cache:
        cache[cache_key] = load_image_template(key, require=require)
    return cache[cache_key]


def fuse_image_results(results: list[dict[str, Any]], method: str) -> float:
    """把同一手指全部模板的图像级分数融合为 identity 级分数。

    ``max`` 和 ``max_quality_tiebreak`` 取最高分；``mean`` 和 ``top3_mean``
    分别计算全体分数均值和最高三项均值。
    """

    if not results:
        return 0.0
    scores = np.asarray([float(item.get("score", 0.0)) for item in results], dtype=np.float32)
    method = str(method).lower()
    if method == "max":
        return float(np.max(scores))
    if method == "mean":
        return float(np.mean(scores))
    if method == "top3_mean":
        top = np.sort(scores)[-min(3, scores.size) :]
        return float(np.mean(top)) if top.size else 0.0
    if method == "max_quality_tiebreak":
        best = max(
            results,
            key=lambda item: (float(item.get("score", 0.0)), float(item.get("quality_score", 0.0))),
        )
        return float(best.get("score", 0.0))
    raise ValueError(f"unsupported fusion method: {method}")


END_TO_END_EXCLUDED_MATCH_OVERHEAD_MS_FIELDS: tuple[str, ...] = (
    "descriptor_prepare_ms",
    "postprocess_ms",
    "identity_match_overhead_ms",
    "matching_wrapper_overhead_ms",
)


def sum_end_to_end_excluded_match_overhead_ms(row: dict[str, Any]) -> float:
    """端到端耗时统计中不计入的匹配阶段杂项耗时之和。"""

    return sum(
        float(row.get(field, 0.0))
        for field in END_TO_END_EXCLUDED_MATCH_OVERHEAD_MS_FIELDS
    )


def compute_end_to_end_core_ms(end_to_end_ms: float, row: dict[str, Any]) -> float:
    """从墙钟端到端耗时中扣除不计入统计的匹配杂项。"""

    return max(0.0, float(end_to_end_ms) - sum_end_to_end_excluded_match_overhead_ms(row))


def score_query_against_identity(
    query_template: dict[str, Any],
    identity: dict[str, Any],
    config: dict[str, Any],
    template_cache: dict[str, dict[str, Any]],
    descriptor_source: str = "hardnet",
) -> dict[str, Any]:
    """完整遍历一份查询模板与一个已注册手指的全部模板并融合分数。"""

    identity_started = time.perf_counter()
    paths = [str(path) for path in identity.get("template_paths", [])]
    method = str(dict(config.get("identification", {})).get("fusion_method", "max")).lower()
    require = "sift" if str(descriptor_source).lower() in {"sift", "rootsift"} else "hardnet"
    low_unique_inliers = int(
        dict(config.get("texture_verification", {})).get(
            "low_unique_inliers",
            3,
        )
    )

    results: list[dict[str, Any]] = []
    best_index = -1
    template_load_ms = 0.0
    timing_fields = (
        "descriptor_prepare_ms",
        "candidate_generation_ms",
        "candidate_filter_ms",
        "ransac_ms",
        "inlier_refinement_ms",
        "unique_inlier_dedup_ms",
        "texture_similarity_ms",
        "score_fusion_ms",
        "postprocess_ms",
        "image_match_total_ms",
    )
    timing_totals = {field: 0.0 for field in timing_fields}
    for index, path in enumerate(paths):
        load_started = time.perf_counter()
        gallery_template = load_template_cached(
            path,
            template_cache,
            require=require,
        )
        template_load_ms += (time.perf_counter() - load_started) * 1000.0
        result = match_templates_descriptor(
            query_template,
            gallery_template,
            config,
            descriptor_source=descriptor_source,
        )
        results.append(result)
        for field in timing_fields:
            timing_totals[field] += float(result.get(field, 0.0))
        if best_index < 0:
            best_index = index
        else:
            best = results[best_index]
            best_key = (float(best.get("score", 0.0)), float(best.get("quality_score", 0.0)))
            result_key = (float(result.get("score", 0.0)), float(result.get("quality_score", 0.0)))
            if result_key > best_key:
                best_index = index
    if results:
        best_index = max(
            range(len(results)),
            key=lambda idx: (
                float(results[idx].get("score", 0.0)),
                float(results[idx].get("quality_score", 0.0)),
            ),
        )
    best = results[best_index] if best_index >= 0 else {}
    best_template = (
        load_template_cached(paths[best_index], template_cache, require=require)
        if best_index >= 0
        else {}
    )
    fusion_started = time.perf_counter()
    identity_score = float(fuse_image_results(results, method))
    identity_fusion_ms = (time.perf_counter() - fusion_started) * 1000.0
    identity_match_total_ms = (
        time.perf_counter() - identity_started
    ) * 1000.0
    identity_match_overhead_ms = max(
        0.0,
        identity_match_total_ms
        - template_load_ms
        - timing_totals["image_match_total_ms"]
        - identity_fusion_ms,
    )
    return {
        "score": identity_score,
        "descriptor_source": str(best.get("descriptor_source", descriptor_source)),
        "descriptor_kind": str(best.get("descriptor_kind", "float")),
        "descriptor_metric": str(best.get("descriptor_metric", "l2")),
        "descriptor_dim": int(best.get("descriptor_dim", 0)),
        "best_template_path": paths[best_index] if best_index >= 0 else "",
        "best_template_image_id": str(best_template.get("image_id", "")),
        "best_template_image_path": str(best_template.get("image_path", "")),
        "best_affine_matrix": best.get("affine_matrix"),
        "num_templates": len(paths),
        "best_image_score": float(best.get("score", 0.0)),
        "best_quality_score": float(best.get("quality_score", 0.0)),
        **{field: best.get(field, "") for field in QUALITY_RESULT_FIELDS},
        "query_frontend_mask_applied": int(bool(query_template.get("frontend_mask_applied", False))),
        "num_raw_matches": int(best.get("num_raw_matches", 0)),
        "num_abs_distance_pass": int(best.get("num_abs_distance_pass", 0)),
        "num_one_way_ratio_pass": int(best.get("num_one_way_ratio_pass", 0)),
        "num_ratio_pass": int(best.get("num_ratio_pass", 0)),
        "num_queries_with_candidates": int(best.get("num_queries_with_candidates", 0)),
        "candidate_query_coverage": float(best.get("candidate_query_coverage", 0.0)),
        "templates_with_candidates": sum(
            int(item.get("num_candidates", 0) > 0) for item in results
        ),
        "templates_with_abs_distance_pass": sum(
            int(item.get("num_abs_distance_pass", 0) > 0) for item in results
        ),
        "templates_with_ratio_pass": sum(
            int(item.get("num_ratio_pass", 0) > 0) for item in results
        ),
        "templates_with_ransac_success": sum(
            int(bool(item.get("ransac_success", False))) for item in results
        ),
        "templates_with_unique_inlier_ready": sum(
            int(int(item.get("unique_inliers", 0)) >= low_unique_inliers)
            for item in results
        ),
        "templates_with_texture_evaluated": sum(
            int(bool(item.get("texture_evaluated", False))) for item in results
        ),
        "templates_with_texture_available": sum(
            int(bool(item.get("texture_available", False))) for item in results
        ),
        "templates_with_fused_score": sum(
            int(str(item.get("texture_decision", "")) == "fused_score")
            for item in results
        ),
        "max_candidate_query_coverage": max(
            (float(item.get("candidate_query_coverage", 0.0)) for item in results),
            default=0.0,
        ),
        "max_num_candidates": max(
            (int(item.get("num_candidates", 0)) for item in results),
            default=0,
        ),
        "max_raw_inliers": max(
            (int(item.get("raw_inliers", 0)) for item in results),
            default=0,
        ),
        "max_unique_inliers": max(
            (int(item.get("unique_inliers", 0)) for item in results),
            default=0,
        ),
        "max_geometry_similarity": max(
            (float(item.get("geometry_similarity", 0.0)) for item in results),
            default=0.0,
        ),
        "max_texture_similarity": max(
            (float(item.get("texture_similarity", 0.0)) for item in results),
            default=0.0,
        ),
        "ransac_success": int(bool(best.get("ransac_success", False))),
        "scale_rejected": int(bool(best.get("scale_rejected", False))),
        "num_candidates": int(best.get("num_candidates", 0)),
        "num_inliers": int(best.get("num_inliers", 0)),
        "raw_inliers": int(best.get("raw_inliers", 0)),
        "unique_inliers": int(best.get("unique_inliers", 0)),
        "unique_query_inliers": int(best.get("unique_query_inliers", 0)),
        "unique_gallery_inliers": int(best.get("unique_gallery_inliers", 0)),
        "inlier_ratio": float(best.get("inlier_ratio", 0.0)),
        "mean_descriptor_distance": float(
            best.get("mean_descriptor_distance", 0.0)
        ),
        "mean_l2_distance": float(best.get("mean_l2_distance", 0.0)),
        "mean_hamming_distance": float(
            best.get("mean_hamming_distance", 0.0)
        ),
        "mean_reproj_error": float(best.get("mean_reproj_error", 0.0)),
        "orientation_consistency": float(best.get("orientation_consistency", 0.0)),
        "dominant_angle_delta": float(best.get("dominant_angle_delta", 0.0)),
        "mean_similarity": float(best.get("mean_similarity", 0.0)),
        "texture_enabled": int(bool(best.get("texture_enabled", False))),
        "texture_evaluated": int(bool(best.get("texture_evaluated", False))),
        "texture_available": int(bool(best.get("texture_available", False))),
        "texture_similarity": float(best.get("texture_similarity", 0.0)),
        "texture_overlap_fraction": float(best.get("texture_overlap_fraction", 0.0)),
        "texture_valid_blocks": int(best.get("texture_valid_blocks", 0)),
        "geometry_similarity": float(best.get("geometry_similarity", 0.0)),
        "geometry_weight": float(best.get("geometry_weight", 1.0)),
        "texture_weight": float(best.get("texture_weight", 0.0)),
        "texture_decision": str(best.get("texture_decision", "")),
        "registered_template_load_ms": template_load_ms,
        **timing_totals,
        "identity_fusion_ms": identity_fusion_ms,
        "identity_match_overhead_ms": identity_match_overhead_ms,
        "identity_match_total_ms": identity_match_total_ms,
    }
