"""正样本构建使用的方向对齐 patch 与局部 ZNCC 质量计算。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class LocalZNCCResult:
    """局部块 ZNCC 搜索结果。"""

    available: bool
    similarity: float
    valid_blocks: int
    best_shift_x: int
    best_shift_y: int
    used_180_rotation: bool


def extract_aligned_patch(
    image: np.ndarray,
    x: float,
    y: float,
    angle: float,
    crop_size: int,
    out_size: int,
) -> np.ndarray:
    """按关键点方向旋转原图并提取固定大小灰度 patch。"""

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((float(x), float(y)), float(angle), 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        dsize=(width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    patch = cv2.getRectSubPix(
        rotated,
        patchSize=(int(crop_size), int(crop_size)),
        center=(float(x), float(y)),
    )
    return cv2.resize(
        patch,
        (int(out_size), int(out_size)),
        interpolation=cv2.INTER_AREA,
    )


def _shifted_common_views(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
    shift_x: int,
    shift_y: int,
) -> tuple[np.ndarray, np.ndarray]:
    """返回平移后的公共区域，不用零填充污染相关系数。"""

    height, width = patch_a.shape
    if shift_x >= 0:
        a_left, a_right = shift_x, width
        b_left, b_right = 0, width - shift_x
    else:
        a_left, a_right = 0, width + shift_x
        b_left, b_right = -shift_x, width
    if shift_y >= 0:
        a_top, a_bottom = shift_y, height
        b_top, b_bottom = 0, height - shift_y
    else:
        a_top, a_bottom = 0, height + shift_y
        b_top, b_bottom = -shift_y, height
    return (
        patch_a[a_top:a_bottom, a_left:a_right],
        patch_b[b_top:b_bottom, b_left:b_right],
    )


def _block_zncc(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
    block_size: int,
    min_block_std: float,
) -> tuple[float, int]:
    """计算有效纹理块 ZNCC 的中位数。"""

    if patch_a.shape != patch_b.shape or patch_a.ndim != 2 or patch_a.size == 0:
        return 0.0, 0
    height, width = patch_a.shape
    scores: list[float] = []
    for top in range(0, height, block_size):
        bottom = min(top + block_size, height)
        for left in range(0, width, block_size):
            right = min(left + block_size, width)
            values_a = patch_a[top:bottom, left:right].reshape(-1).astype(np.float64)
            values_b = patch_b[top:bottom, left:right].reshape(-1).astype(np.float64)
            if values_a.size < 16:
                continue
            centered_a = values_a - float(values_a.mean())
            centered_b = values_b - float(values_b.mean())
            energy_a = float(np.dot(centered_a, centered_a))
            energy_b = float(np.dot(centered_b, centered_b))
            if math.sqrt(energy_a / values_a.size) < min_block_std:
                continue
            if math.sqrt(energy_b / values_b.size) < min_block_std:
                continue
            denominator = math.sqrt(energy_a * energy_b)
            if denominator <= 1e-12:
                continue
            correlation = float(np.dot(centered_a, centered_b) / denominator)
            scores.append(float(np.clip(correlation, 0.0, 1.0)))
    if not scores:
        return 0.0, 0
    return float(np.median(np.asarray(scores, dtype=np.float32))), len(scores)


def local_patch_zncc(
    image_a: np.ndarray,
    point_a: Mapping[str, Any],
    image_b: np.ndarray,
    point_b: Mapping[str, Any],
    config: Mapping[str, Any],
) -> LocalZNCCResult:
    """方向对齐后，在小范围平移和 180° 二义性中寻找最可信的块级 ZNCC。"""

    crop_size = max(4, int(config.get("crop_size", 32)))
    out_size = max(4, int(config.get("out_size", 32)))
    block_size = max(4, int(config.get("block_size", 8)))
    max_shift_px = max(0, int(config.get("max_shift_px", 2)))
    min_block_std = max(0.0, float(config.get("min_block_std", 5.0)))
    min_valid_blocks = max(1, int(config.get("min_valid_blocks", 4)))
    blur_sigma = max(0.0, float(config.get("blur_sigma", 0.8)))

    patch_a = extract_aligned_patch(
        image_a,
        float(point_a["x"]),
        float(point_a["y"]),
        float(point_a["angle"]),
        crop_size,
        out_size,
    ).astype(np.float32)
    patch_b = extract_aligned_patch(
        image_b,
        float(point_b["x"]),
        float(point_b["y"]),
        float(point_b["angle"]),
        crop_size,
        out_size,
    ).astype(np.float32)
    if blur_sigma > 0.0:
        patch_a = cv2.GaussianBlur(patch_a, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        patch_b = cv2.GaussianBlur(patch_b, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    orientations = [(patch_b, False)]
    if bool(config.get("allow_180_rotation", True)):
        orientations.append((np.rot90(patch_b, 2), True))

    best = LocalZNCCResult(False, 0.0, 0, 0, 0, False)
    best_rank = (-1.0, -1, -10**9, -1)
    for candidate_b, used_180 in orientations:
        for shift_y in range(-max_shift_px, max_shift_px + 1):
            for shift_x in range(-max_shift_px, max_shift_px + 1):
                common_a, common_b = _shifted_common_views(
                    patch_a,
                    candidate_b,
                    shift_x,
                    shift_y,
                )
                similarity, valid_blocks = _block_zncc(
                    common_a,
                    common_b,
                    block_size,
                    min_block_std,
                )
                available = valid_blocks >= min_valid_blocks
                rank = (
                    similarity if available else -1.0,
                    valid_blocks,
                    -(abs(shift_x) + abs(shift_y)),
                    -int(used_180),
                )
                if rank > best_rank:
                    best_rank = rank
                    best = LocalZNCCResult(
                        available=available,
                        similarity=similarity if available else 0.0,
                        valid_blocks=valid_blocks,
                        best_shift_x=shift_x,
                        best_shift_y=shift_y,
                        used_180_rotation=used_180,
                    )
    return best


def minimum_zncc_similarity(config: Mapping[str, Any]) -> float:
    """读取两类正样本共用的 ZNCC 硬阈值。"""

    value = config.get("min_similarity", 0.60)
    if isinstance(value, Mapping):
        raise ValueError(
            "matching.positive_zncc.min_similarity must be one shared numeric threshold."
        )
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "matching.positive_zncc.min_similarity must be within [0, 1], "
            f"got {value!r}."
        )
    return threshold