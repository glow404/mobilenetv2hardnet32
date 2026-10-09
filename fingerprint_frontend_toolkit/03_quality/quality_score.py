#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FFQ-Lite V1.5 Traditional-Enhanced
==================================

质量优先版（无学习模型）：

相对 V1.4.2 的主要变化
---------------------
1. 保留外部 Mask 入口；
2. 方向场从简单 16x16 block 结构张量，升级为 GBFOE 风格：
   - 百分位对比度拉伸
   - Gaussian + Median 预滤波
   - Sobel 梯度
   - Gaussian 加权结构张量
   - 自适应方向估计尺度
   - 输出像素级 orientation + strength(coherence)
3. Coherence 直接基于 GBFOE strength 统计；
4. Orientation Flow 使用 GBFOE orientation，在有效块上做双角度连续性分析；
5. 新增 X-Signature 风格 Ridge Frequency Consistency：
   - 沿局部纹线法向提取灰度投影
   - 检测 ridge/valley 周期
   - 统计有效周期率、周期 CV、邻域周期连续性
6. Contrast、Sharpness 保留原图统计，避免“增强后再评价”掩盖原始质量；
7. Noise 保留高频残差，同时引入方向/频率特征共同抑制“高梯度噪声”误判；
8. 最终由 7 项子分数融合：
   Area 6%
   Coherence 30%
   Flow 18%
   Frequency 20%
   Contrast 13%
   Sharpness 5%
   Noise 8%

说明
----
这是“质量优先”的传统图像处理版本，不追求 1 ms 级速度。
推荐输入为约 500 dpi 指纹。

运行：
python ffq_lite_v1_5_traditional_enhanced.py \
    --input_dir /path/to/images \
    --output_dir ./ffq_v15_results

带外部 Mask：
python ffq_lite_v1_5_traditional_enhanced.py \
    --input_dir /path/to/images \
    --output_dir ./ffq_v15_results \
    --mask_dir /path/to/masks
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
EPS = 1e-8


@dataclass
class FFQConfig:
    # ===== General =====
    max_side: int = 320
    block_size: int = 16
    sobel_ksize: int = 3

    # ===== Auto foreground (only when external mask absent) =====
    min_block_std: float = 0.035
    min_block_gradient: float = 0.020
    min_block_mean: float = 0.03
    max_block_mean: float = 0.97
    keep_largest_component: bool = True

    # external mask block acceptance
    # V1.5 不再要求 50%，只要一个块有 >=10% 有效 mask 即保留
    external_mask_block_ratio: float = 0.10

    # ===== Coherence / flow =====
    low_coherence_threshold: float = 0.35
    high_coherence_threshold: float = 0.65
    orientation_jump_threshold: float = 0.30

    # ===== GBFOE-style parameters =====
    gbfoe_percentile: float = 19.0
    gbfoe_sigma_smooth: float = 1.25
    gbfoe_median_size: int = 5
    # 原实现默认 27/43 针对较大 500dpi 图。这里按 max_side=320 缩小一些，
    # 同时保持“先基础尺度，再自适应尺度”的机制。
    gbfoe_sigma_base: float = 9.0
    gbfoe_sigma_multiplier: float = 15.0
    gbfoe_sigma_min: float = 3.0
    gbfoe_sigma_max: float = 18.0

    # ===== X-signature frequency =====
    freq_window_width: int = 23
    freq_window_height: int = 43
    freq_period_min: float = 5.0
    freq_period_max: float = 20.0
    freq_min_valid_distances: int = 4
    freq_min_mask_ratio: float = 0.55
    freq_peak_tolerance: float = 2.0

    # ===== Final weights =====
    # V1.5.1 rebalanced weights:
    # - Area / Sharpness 在当前按压数据上接近饱和，降低权重；
    # - Coherence 是最有区分度的主指标，提高权重；
    # - Frequency 保留较高权重，用于约束纹线周期稳定性；
    # - Flow / Contrast 维持中等权重；
    # - Noise 保留为辅助惩罚项。
    weight_area: float = 0.06
    weight_coherence: float = 0.30
    weight_flow: float = 0.18
    weight_frequency: float = 0.20
    weight_contrast: float = 0.13
    weight_sharpness: float = 0.05
    weight_noise: float = 0.08


@dataclass
class ImageQualityResult:
    filename: str
    relative_path: str
    width: int
    height: int
    resized_width: int
    resized_height: int

    quality_score: float
    quality_level: str

    area_score: float
    orientation_coherence_score: float
    orientation_flow_score: float
    frequency_score: float
    contrast_score: float
    sharpness_score: float
    noise_score: float

    foreground_ratio: float
    largest_component_ratio: float
    center_coverage_ratio: float
    border_touch_ratio: float
    border_touch_sides: int

    coherence_mean: float
    coherence_std: float
    coherence_p10: float
    low_coherence_ratio: float
    high_coherence_ratio: float

    orientation_difference_mean: float
    orientation_difference_p90: float
    orientation_jump_ratio: float
    orientation_pair_count: int

    frequency_valid_ratio: float
    ridge_period_mean: float
    ridge_period_std: float
    ridge_period_cv: float
    ridge_period_p10: float
    ridge_period_p90: float
    frequency_neighbor_diff_mean: float
    frequency_neighbor_diff_p90: float
    frequency_jump_ratio: float
    frequency_pair_count: int

    local_std_mean: float
    local_std_p10: float
    low_contrast_ratio: float
    over_dark_ratio: float
    over_bright_ratio: float

    gradient_energy_mean: float
    gradient_energy_p10: float
    weak_gradient_ratio: float

    noise_residual_mean: float
    noise_residual_p90: float
    strong_noise_ratio: float
    noise_block_std: float

    read_time_ms: float
    preprocess_time_ms: float
    foreground_time_ms: float
    orientation_time_ms: float
    frequency_time_ms: float
    feature_time_ms: float
    scoring_time_ms: float
    processing_time_ms: float
    total_time_ms: float

    error: str = ""


# =============================================================================
# Utilities
# =============================================================================

def clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def linear_score(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("high 必须大于 low")
    return clamp01((value - low) / (high - low))


def inverse_linear_score(value: float, good: float, bad: float) -> float:
    if bad <= good:
        raise ValueError("bad 必须大于 good")
    return clamp01((bad - value) / (bad - good))


def quality_level(score: float) -> str:
    if score < 40:
        return "较差"
    if score < 60:
        return "一般"
    if score < 80:
        return "良好"
    return "优秀"


def safe_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def discover_images(input_dir: Path, recursive: bool) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是目录：{input_dir}")

    iterator: Iterable[Path] = input_dir.rglob("*") if recursive else input_dir.glob("*")
    images = [
        p for p in iterator
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(images, key=lambda p: str(p).lower())


def read_grayscale_image(path: Path) -> np.ndarray:
    try:
        file_bytes = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    except Exception as exc:
        raise RuntimeError(f"读取图像失败：{exc}") from exc

    if image is None:
        raise RuntimeError("OpenCV 无法解码该图像")
    if image.ndim != 2:
        raise RuntimeError(f"期望灰度图，实际维度为 {image.shape}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"图像编码失败：{path}")
    encoded.tofile(str(path))


def resize_keep_aspect(image: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return image
    h, w = image.shape
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def normalize_grayscale(image: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float32) / 255.0
    sample = image_f[::4, ::4]
    low = float(np.percentile(sample, 1.0))
    high = float(np.percentile(sample, 99.0))
    if high - low > 0.05:
        image_f = (image_f - low) / (high - low)
        image_f = np.clip(image_f, 0.0, 1.0)
    return image_f.astype(np.float32, copy=False)


def crop_to_block_grid(image: np.ndarray, block_size: int) -> Tuple[np.ndarray, int, int]:
    h, w = image.shape
    grid_h = h // block_size
    grid_w = w // block_size
    if grid_h < 2 or grid_w < 2:
        raise ValueError(
            f"图像太小，无法按 block_size={block_size} 分块，当前尺寸={w}x{h}"
        )
    target_h = grid_h * block_size
    target_w = grid_w * block_size
    y0 = (h - target_h) // 2
    x0 = (w - target_w) // 2
    return image[y0:y0 + target_h, x0:x0 + target_w], grid_h, grid_w


def block_mean(array: np.ndarray, block_size: int) -> np.ndarray:
    h, w = array.shape
    return array.reshape(
        h // block_size, block_size,
        w // block_size, block_size
    ).mean(axis=(1, 3))


def largest_component(mask: np.ndarray) -> Tuple[np.ndarray, float]:
    mask_u8 = mask.astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    foreground_count = int(mask.sum())
    if count <= 1 or foreground_count == 0:
        return np.zeros_like(mask, dtype=bool), 0.0

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_index = 1 + int(np.argmax(component_areas))
    largest_mask = labels == largest_index
    ratio = float(largest_mask.sum() / max(foreground_count, 1))
    return largest_mask, ratio


# =============================================================================
# Mask
# =============================================================================

def find_external_mask(image_path: Path, input_dir: Path, mask_dir: Optional[Path]):
    if mask_dir is None:
        return None

    rel = image_path.relative_to(input_dir)

    candidates = [
        mask_dir / rel,
        mask_dir / rel.parent / (image_path.stem + "_mask" + image_path.suffix),
        mask_dir / rel.parent / (image_path.stem + "_mask.bmp"),
        mask_dir / rel.parent / (image_path.stem + "_mask.png"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def read_binary_mask(path: Path) -> np.ndarray:
    img = read_grayscale_image(path)
    return img >= 128


def mask_area_limit(ratio: float) -> float:
    if ratio < 0.10:
        return 15.0
    if ratio < 0.20:
        return 25.0
    if ratio < 0.30:
        return 40.0
    return 100.0


def external_mask_to_block_mask(mask: np.ndarray, block_size: int, min_ratio: float) -> np.ndarray:
    h, w = mask.shape
    gh = h // block_size
    gw = w // block_size
    mask = mask[:gh * block_size, :gw * block_size]
    ratio = block_mean(mask.astype(np.float32), block_size)
    return (ratio >= min_ratio).astype(bool)


# =============================================================================
# Basic gradients / block statistics
# =============================================================================

def compute_gradients(
    image: np.ndarray,
    sobel_ksize: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx = cv2.Sobel(
        image, cv2.CV_32F, 1, 0,
        ksize=sobel_ksize,
        borderType=cv2.BORDER_REPLICATE
    )
    gy = cv2.Sobel(
        image, cv2.CV_32F, 0, 1,
        ksize=sobel_ksize,
        borderType=cv2.BORDER_REPLICATE
    )

    if sobel_ksize == 1:
        scale = 2.0
    elif sobel_ksize == 3:
        scale = 4.0
    elif sobel_ksize == 5:
        scale = 48.0
    elif sobel_ksize == 7:
        scale = 640.0
    else:
        raise ValueError("sobel_ksize 只允许 1、3、5、7")

    gx = gx / scale
    gy = gy / scale
    magnitude = cv2.magnitude(gx, gy)
    return gx, gy, magnitude


def compute_block_statistics(
    image: np.ndarray,
    magnitude: np.ndarray,
    block_size: int
) -> Dict[str, np.ndarray]:
    mean = block_mean(image, block_size)
    mean_sq = block_mean(image * image, block_size)
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(variance)

    grad_mean = block_mean(magnitude, block_size)
    grad_energy = block_mean(magnitude * magnitude, block_size)

    return {
        "mean": mean,
        "std": std,
        "grad_mean": grad_mean,
        "grad_energy": grad_energy,
    }


def build_foreground_mask(
    stats: Dict[str, np.ndarray],
    config: FFQConfig
) -> Tuple[np.ndarray, float]:
    block_mean_gray = stats["mean"]
    block_std = stats["std"]
    block_grad = stats["grad_mean"]

    mask = (
        (block_std >= config.min_block_std)
        & (block_grad >= config.min_block_gradient)
        & (block_mean_gray >= config.min_block_mean)
        & (block_mean_gray <= config.max_block_mean)
    )

    mask_u8 = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)

    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = mask_u8 > 0

    if config.keep_largest_component:
        mask, largest_ratio = largest_component(mask)
    else:
        _, largest_ratio = largest_component(mask)

    return mask, largest_ratio


# =============================================================================
# GBFOE-style orientation estimation
# =============================================================================

def _odd_gaussian_kernel_size(sigma: float) -> int:
    sigma = max(float(sigma), 0.1)
    return max(3, int(math.ceil(3.0 * sigma)) * 2 + 1)


def _gbfoe_strength_components(
    gx2: np.ndarray,
    gy2: np.ndarray,
    g2xy: np.ndarray,
    sigma: float,
    mask_pixel: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = _odd_gaussian_kernel_size(sigma)

    sum_gx2 = cv2.GaussianBlur(gx2, (k, k), sigmaX=sigma, sigmaY=sigma)
    sum_gy2 = cv2.GaussianBlur(gy2, (k, k), sigmaX=sigma, sigmaY=sigma)
    n = sum_gx2 - sum_gy2
    d = cv2.GaussianBlur(g2xy, (k, k), sigmaX=sigma, sigmaY=sigma)

    denom = sum_gx2 + sum_gy2
    strength = np.divide(
        np.sqrt(n * n + d * d),
        denom,
        out=np.zeros_like(denom, dtype=np.float32),
        where=denom > EPS,
    )

    strength = np.clip(strength, 0.0, 1.0).astype(np.float32)
    if mask_pixel is not None:
        strength[~mask_pixel] = 0.0

    return strength, n.astype(np.float32), d.astype(np.float32)


def compute_orientation_field_gbfoe(
    image: np.ndarray,
    mask_pixel: Optional[np.ndarray],
    config: FFQConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    GBFOE 风格的传统方向场估计。

    image: float32 [0,1]
    mask_pixel: bool pixel mask
    return:
      orientation_pixel  [-pi/2, pi/2)
      strength_pixel     [0,1]
      adaptive_sigma
    """
    img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    # 1) percentile contrast stretching (仅用于方向估计，不影响 Contrast Score)
    p = float(config.gbfoe_percentile)
    if p > 0:
        valid = img_u8[mask_pixel] if mask_pixel is not None and np.any(mask_pixel) else img_u8.ravel()
        if valid.size:
            v1 = float(np.percentile(valid, p))
            v2 = float(np.percentile(valid, 100.0 - p))
            if v2 > v1 + 1.0:
                img_u8 = np.clip(
                    (img_u8.astype(np.float32) - v1) * 255.0 / (v2 - v1),
                    0, 255
                ).astype(np.uint8)

    # 2) Gaussian smooth
    if config.gbfoe_sigma_smooth > 0:
        k = _odd_gaussian_kernel_size(config.gbfoe_sigma_smooth)
        img_u8 = cv2.GaussianBlur(
            img_u8, (k, k),
            sigmaX=config.gbfoe_sigma_smooth,
            sigmaY=config.gbfoe_sigma_smooth,
            borderType=cv2.BORDER_REPLICATE
        )

    # 3) Median
    if config.gbfoe_median_size > 1:
        m = int(config.gbfoe_median_size)
        if m % 2 == 0:
            m += 1
        img_u8 = cv2.medianBlur(img_u8, m)

    # 4) Sobel
    gx = cv2.Sobel(
        img_u8, cv2.CV_32F, 1, 0,
        ksize=3, borderType=cv2.BORDER_REPLICATE
    )
    gy = cv2.Sobel(
        img_u8, cv2.CV_32F, 0, 1,
        ksize=3, borderType=cv2.BORDER_REPLICATE
    )

    if mask_pixel is not None:
        gx[~mask_pixel] = 0.0
        gy[~mask_pixel] = 0.0

    gx2 = gx * gx
    gy2 = gy * gy
    # 与上传 GBFOE 一致使用 -2*gxy
    g2xy = -2.0 * gx * gy

    # 5) 基础尺度 strength
    base_strength, _, _ = _gbfoe_strength_components(
        gx2, gy2, g2xy,
        config.gbfoe_sigma_base,
        mask_pixel,
    )

    if mask_pixel is not None and np.any(mask_pixel):
        avg_strength = float(base_strength[mask_pixel].mean())
    else:
        avg_strength = float(base_strength.mean())

    # 6) 自适应 sigma
    adaptive_sigma = config.gbfoe_sigma_multiplier * (1.0 - avg_strength)
    adaptive_sigma = float(np.clip(
        adaptive_sigma,
        config.gbfoe_sigma_min,
        config.gbfoe_sigma_max
    ))

    strength, n, d = _gbfoe_strength_components(
        gx2, gy2, g2xy,
        adaptive_sigma,
        mask_pixel,
    )

    # 原 GBFOE：((phase(n,d)+pi)/2)%pi
    # OpenCV phase 对 (x=n,y=d)
    phase = cv2.phase(n, d, angleInDegrees=False)
    orientation = ((phase + np.pi) / 2.0) % np.pi

    # 统一映射到 [-pi/2, pi/2)
    orientation = ((orientation + np.pi / 2.0) % np.pi) - np.pi / 2.0

    if mask_pixel is not None:
        orientation = orientation.astype(np.float32)
        orientation[~mask_pixel] = 0.0

    return orientation.astype(np.float32), strength.astype(np.float32), adaptive_sigma


def pixel_field_to_block_field(
    orientation_pixel: np.ndarray,
    strength_pixel: np.ndarray,
    foreground_mask: np.ndarray,
    block_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将像素级方向用 double-angle vector 做 block 聚合；
    coherence/strength 取块平均。
    """
    cos2 = np.cos(2.0 * orientation_pixel) * strength_pixel
    sin2 = np.sin(2.0 * orientation_pixel) * strength_pixel

    bx = block_mean(cos2.astype(np.float32), block_size)
    by = block_mean(sin2.astype(np.float32), block_size)
    bs = block_mean(strength_pixel.astype(np.float32), block_size)

    orientation_block = 0.5 * np.arctan2(by, bx)
    orientation_block = orientation_block.astype(np.float32)
    strength_block = np.clip(bs, 0.0, 1.0).astype(np.float32)

    orientation_block[~foreground_mask] = 0.0
    strength_block[~foreground_mask] = 0.0

    return orientation_block, strength_block


# =============================================================================
# Orientation flow
# =============================================================================

def orientation_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (1.0 - np.cos(2.0 * (a - b))) * 0.5


def compute_orientation_flow_features(
    orientation: np.ndarray,
    coherence: np.ndarray,
    foreground_mask: np.ndarray,
    config: FFQConfig
) -> Tuple[float, float, float, int]:

    foreground_mask = np.asarray(foreground_mask, dtype=np.bool_)
    differences: List[np.ndarray] = []
    weights: List[np.ndarray] = []

    valid_h = foreground_mask[:, :-1] & foreground_mask[:, 1:]
    reliable_h = (coherence[:, :-1] > 0.20) & (coherence[:, 1:] > 0.20)
    valid_h &= reliable_h

    if np.any(valid_h):
        diff_h = orientation_difference(
            orientation[:, :-1], orientation[:, 1:]
        )[valid_h]
        weight_h = np.minimum(
            coherence[:, :-1], coherence[:, 1:]
        )[valid_h]
        differences.append(diff_h)
        weights.append(weight_h)

    valid_v = foreground_mask[:-1, :] & foreground_mask[1:, :]
    reliable_v = (coherence[:-1, :] > 0.20) & (coherence[1:, :] > 0.20)
    valid_v &= reliable_v

    if np.any(valid_v):
        diff_v = orientation_difference(
            orientation[:-1, :], orientation[1:, :]
        )[valid_v]
        weight_v = np.minimum(
            coherence[:-1, :], coherence[1:, :]
        )[valid_v]
        differences.append(diff_v)
        weights.append(weight_v)

    if not differences:
        return 1.0, 1.0, 1.0, 0

    diff = np.concatenate(differences).astype(np.float32)
    weight = np.concatenate(weights).astype(np.float32)

    weighted_mean = float(np.sum(diff * weight) / (np.sum(weight) + EPS))
    p90 = safe_percentile(diff, 90)
    jump_ratio = float(np.mean(diff >= config.orientation_jump_threshold))

    return weighted_mean, p90, jump_ratio, int(diff.size)


# =============================================================================
# X-signature ridge period consistency
# =============================================================================

def _estimate_local_ridge_period(
    image_u8: np.ndarray,
    mask_pixel: np.ndarray,
    x: int,
    y: int,
    theta: float,
    config: FFQConfig,
) -> float:
    """
    在 block 中心附近取一个旋转窗口，使纹线近似水平，
    对行方向做投影（X-signature），根据相邻峰/谷距离估计 ridge period。
    """
    ww = int(config.freq_window_width)
    wh = int(config.freq_window_height)
    if ww % 2 == 0:
        ww += 1
    if wh % 2 == 0:
        wh += 1

    # 对上传 XSFFE 的逻辑做轻量复现：
    # 旋转局部窗口，使 ridge 方向对齐。
    angle_deg = 180.0 - math.degrees(theta)
    M = cv2.getRotationMatrix2D((float(x), float(y)), angle_deg, 1.0)
    M[0, 2] += ww / 2.0 - x
    M[1, 2] += wh / 2.0 - y

    region = cv2.warpAffine(
        image_u8, M, (ww, wh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    region_mask = cv2.warpAffine(
        (mask_pixel.astype(np.uint8) * 255),
        M, (ww, wh),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    if float(np.mean(region_mask > 0)) < config.freq_min_mask_ratio:
        return 0.0

    # 先做小幅平滑，降低噪声峰
    region = cv2.medianBlur(region, 5)
    region = cv2.GaussianBlur(region, (3, 3), 0)

    xs = np.sum(region.astype(np.float32), axis=1)
    if xs.size < 5:
        return 0.0

    xs1 = xs[1:-1]
    local_maxima = 1 + np.nonzero(
        (xs1 > xs[:-2]) & (xs1 >= xs[2:])
    )[0]
    local_minima = 1 + np.nonzero(
        (xs1 < xs[:-2]) & (xs1 <= xs[2:])
    )[0]

    distances = []
    if local_maxima.size >= 2:
        distances.append(local_maxima[1:] - local_maxima[:-1])
    if local_minima.size >= 2:
        distances.append(local_minima[1:] - local_minima[:-1])

    if not distances:
        return 0.0

    valid = np.concatenate(distances).astype(np.float32)
    valid = valid[
        (valid >= config.freq_period_min)
        & (valid <= config.freq_period_max)
    ]

    if valid.size < config.freq_min_valid_distances:
        return 0.0

    med = float(np.median(valid))
    close = valid[np.abs(valid - med) <= config.freq_peak_tolerance]
    if close.size == 0:
        return 0.0

    return float(close.mean())


def compute_frequency_map_and_features(
    image: np.ndarray,
    mask_pixel: np.ndarray,
    orientation_block: np.ndarray,
    coherence_block: np.ndarray,
    foreground_mask: np.ndarray,
    config: FFQConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    每个有效 block 中心做一次 X-signature ridge period 估计。
    """
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    gh, gw = foreground_mask.shape
    bs = config.block_size

    periods = np.zeros((gh, gw), dtype=np.float32)

    for r in range(gh):
        for c in range(gw):
            if not foreground_mask[r, c]:
                continue
            # 方向本身太不可靠时，不做频率估计
            if coherence_block[r, c] < 0.20:
                continue

            x = int((c + 0.5) * bs)
            y = int((r + 0.5) * bs)

            x = min(max(x, 0), image.shape[1] - 1)
            y = min(max(y, 0), image.shape[0] - 1)

            periods[r, c] = _estimate_local_ridge_period(
                image_u8,
                mask_pixel,
                x, y,
                float(orientation_block[r, c]),
                config
            )

    fg_count = int(np.count_nonzero(foreground_mask))
    valid_mask = foreground_mask & (periods > 0)
    valid = periods[valid_mask]

    if fg_count == 0:
        valid_ratio = 0.0
    else:
        valid_ratio = float(valid.size / fg_count)

    if valid.size == 0:
        period_mean = 0.0
        period_std = 0.0
        period_cv = 1.0
        period_p10 = 0.0
        period_p90 = 0.0
    else:
        period_mean = float(valid.mean())
        period_std = float(valid.std())
        period_cv = float(period_std / max(period_mean, EPS))
        period_p10 = safe_percentile(valid, 10)
        period_p90 = safe_percentile(valid, 90)

    diffs: List[np.ndarray] = []

    # horizontal
    vh = valid_mask[:, :-1] & valid_mask[:, 1:]
    if np.any(vh):
        diffs.append(np.abs(periods[:, :-1] - periods[:, 1:])[vh])

    # vertical
    vv = valid_mask[:-1, :] & valid_mask[1:, :]
    if np.any(vv):
        diffs.append(np.abs(periods[:-1, :] - periods[1:, :])[vv])

    if diffs:
        d = np.concatenate(diffs).astype(np.float32)
        neighbor_mean = float(d.mean())
        neighbor_p90 = safe_percentile(d, 90)
        jump_ratio = float(np.mean(d >= 3.0))
        pair_count = int(d.size)
    else:
        neighbor_mean = 10.0
        neighbor_p90 = 10.0
        jump_ratio = 1.0
        pair_count = 0

    return periods, {
        "frequency_valid_ratio": valid_ratio,
        "ridge_period_mean": period_mean,
        "ridge_period_std": period_std,
        "ridge_period_cv": period_cv,
        "ridge_period_p10": period_p10,
        "ridge_period_p90": period_p90,
        "frequency_neighbor_diff_mean": neighbor_mean,
        "frequency_neighbor_diff_p90": neighbor_p90,
        "frequency_jump_ratio": jump_ratio,
        "frequency_pair_count": pair_count,
    }


# =============================================================================
# Other features
# =============================================================================

def compute_border_touch_features(mask: np.ndarray) -> Tuple[float, int]:
    if mask.size == 0:
        return 0.0, 0

    top = mask[0, :]
    bottom = mask[-1, :]
    left = mask[:, 0]
    right = mask[:, -1]

    border_values = np.concatenate([
        top.astype(np.float32),
        bottom.astype(np.float32),
        left.astype(np.float32),
        right.astype(np.float32),
    ])
    touch_ratio = float(border_values.mean())

    side_ratios = [
        float(top.mean()),
        float(bottom.mean()),
        float(left.mean()),
        float(right.mean()),
    ]
    touched_sides = int(sum(ratio >= 0.20 for ratio in side_ratios))

    return touch_ratio, touched_sides


def compute_center_coverage(mask: np.ndarray) -> float:
    h, w = mask.shape
    y1 = int(round(h * 0.25))
    y2 = int(round(h * 0.75))
    x1 = int(round(w * 0.25))
    x2 = int(round(w * 0.75))
    center = mask[y1:y2, x1:x2]
    if center.size == 0:
        return 0.0
    return float(center.mean())


def compute_noise_features(
    image: np.ndarray,
    foreground_mask: np.ndarray,
    block_size: int,
) -> Tuple[float, float, float, float]:
    blurred = cv2.GaussianBlur(
        image,
        (3, 3),
        sigmaX=0.8,
        sigmaY=0.8,
        borderType=cv2.BORDER_REPLICATE,
    )
    residual = np.abs(image - blurred)

    pixel_mask = cv2.resize(
        foreground_mask.astype(np.uint8),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    valid = residual[pixel_mask]
    if valid.size == 0:
        return 1.0, 1.0, 1.0, 1.0

    residual_mean = float(valid.mean())
    residual_p90 = float(np.percentile(valid, 90))
    strong_noise_ratio = float(np.mean(valid >= 0.08))

    block_residual = block_mean(residual, block_size)
    valid_blocks = block_residual[foreground_mask]
    noise_block_std = float(valid_blocks.std()) if valid_blocks.size > 0 else 1.0

    return residual_mean, residual_p90, strong_noise_ratio, noise_block_std


# =============================================================================
# Feature extraction
# =============================================================================

def extract_features(
    image: np.ndarray,
    config: FFQConfig,
    external_mask: Optional[np.ndarray] = None
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, float]]:

    stage_times: Dict[str, float] = {}

    t0 = time.perf_counter()

    resized = resize_keep_aspect(image, config.max_side)
    normalized = normalize_grayscale(resized)
    cropped, _, _ = crop_to_block_grid(normalized, config.block_size)

    external_mask_resized = None
    if external_mask is not None:
        external_mask_resized = cv2.resize(
            external_mask.astype(np.uint8),
            (resized.shape[1], resized.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ) > 0

        # 使用与 image 相同的中心裁剪逻辑
        h, w = resized.shape
        ch, cw = cropped.shape
        y0 = (h - ch) // 2
        x0 = (w - cw) // 2
        external_mask_resized = external_mask_resized[
            y0:y0 + ch, x0:x0 + cw
        ]

    stage_times["preprocess_time_ms"] = (time.perf_counter() - t0) * 1000.0

    # Basic gradients for contrast/sharpness/auto foreground
    t1 = time.perf_counter()
    gx_basic, gy_basic, magnitude = compute_gradients(
        cropped, config.sobel_ksize
    )
    stats = compute_block_statistics(
        cropped, magnitude, config.block_size
    )

    if external_mask_resized is None:
        foreground_mask, largest_component_ratio = build_foreground_mask(
            stats, config
        )
    else:
        foreground_mask = external_mask_to_block_mask(
            external_mask_resized,
            config.block_size,
            config.external_mask_block_ratio
        )
        largest_component_ratio = 1.0

    foreground_mask = np.asarray(foreground_mask, dtype=np.bool_)

    # pixel-level mask used by GBFOE/frequency
    if external_mask_resized is not None:
        pixel_mask = external_mask_resized.astype(bool)
    else:
        pixel_mask = cv2.resize(
            foreground_mask.astype(np.uint8),
            (cropped.shape[1], cropped.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    stage_times["foreground_time_ms"] = (time.perf_counter() - t1) * 1000.0

    # GBFOE
    t2 = time.perf_counter()
    orientation_pixel, strength_pixel, adaptive_sigma = compute_orientation_field_gbfoe(
        cropped,
        pixel_mask,
        config,
    )
    orientation, coherence = pixel_field_to_block_field(
        orientation_pixel,
        strength_pixel,
        foreground_mask,
        config.block_size,
    )
    stage_times["orientation_time_ms"] = (time.perf_counter() - t2) * 1000.0

    # Frequency
    t_freq = time.perf_counter()
    period_map, frequency_features = compute_frequency_map_and_features(
        cropped,
        pixel_mask,
        orientation,
        coherence,
        foreground_mask,
        config,
    )
    stage_times["frequency_time_ms"] = (time.perf_counter() - t_freq) * 1000.0

    # Other scalar features
    t3 = time.perf_counter()

    foreground_ratio = float(foreground_mask.mean())
    center_coverage_ratio = compute_center_coverage(foreground_mask)
    border_touch_ratio, border_touch_sides = compute_border_touch_features(foreground_mask)

    (
        noise_residual_mean,
        noise_residual_p90,
        strong_noise_ratio,
        noise_block_std,
    ) = compute_noise_features(
        cropped,
        foreground_mask,
        config.block_size,
    )

    valid_coherence = coherence[foreground_mask]
    valid_std = stats["std"][foreground_mask]
    valid_mean = stats["mean"][foreground_mask]
    valid_grad_energy = stats["grad_energy"][foreground_mask]

    if valid_coherence.size == 0:
        coherence_mean = 0.0
        coherence_std = 0.0
        coherence_p10 = 0.0
        low_coherence_ratio = 1.0
        high_coherence_ratio = 0.0
    else:
        coherence_mean = float(valid_coherence.mean())
        coherence_std = float(valid_coherence.std())
        coherence_p10 = safe_percentile(valid_coherence, 10)
        low_coherence_ratio = float(
            np.mean(valid_coherence < config.low_coherence_threshold)
        )
        high_coherence_ratio = float(
            np.mean(valid_coherence >= config.high_coherence_threshold)
        )

    (
        orientation_difference_mean,
        orientation_difference_p90,
        orientation_jump_ratio,
        orientation_pair_count,
    ) = compute_orientation_flow_features(
        orientation,
        coherence,
        foreground_mask,
        config,
    )

    if valid_std.size == 0:
        local_std_mean = 0.0
        local_std_p10 = 0.0
        low_contrast_ratio = 1.0
        over_dark_ratio = 1.0
        over_bright_ratio = 1.0
    else:
        local_std_mean = float(valid_std.mean())
        local_std_p10 = safe_percentile(valid_std, 10)
        low_contrast_ratio = float(np.mean(valid_std < 0.05))
        over_dark_ratio = float(np.mean(valid_mean < 0.08))
        over_bright_ratio = float(np.mean(valid_mean > 0.92))

    if valid_grad_energy.size == 0:
        gradient_energy_mean = 0.0
        gradient_energy_p10 = 0.0
        weak_gradient_ratio = 1.0
    else:
        gradient_energy_mean = float(valid_grad_energy.mean())
        gradient_energy_p10 = safe_percentile(valid_grad_energy, 10)
        weak_gradient_ratio = float(np.mean(valid_grad_energy < 0.0025))

    stage_times["feature_time_ms"] = (time.perf_counter() - t3) * 1000.0

    scalar_features = {
        "resized_width": int(resized.shape[1]),
        "resized_height": int(resized.shape[0]),

        "foreground_ratio": foreground_ratio,
        "largest_component_ratio": largest_component_ratio,
        "center_coverage_ratio": center_coverage_ratio,
        "border_touch_ratio": border_touch_ratio,
        "border_touch_sides": border_touch_sides,

        "coherence_mean": coherence_mean,
        "coherence_std": coherence_std,
        "coherence_p10": coherence_p10,
        "low_coherence_ratio": low_coherence_ratio,
        "high_coherence_ratio": high_coherence_ratio,

        "orientation_difference_mean": orientation_difference_mean,
        "orientation_difference_p90": orientation_difference_p90,
        "orientation_jump_ratio": orientation_jump_ratio,
        "orientation_pair_count": orientation_pair_count,

        **frequency_features,

        "local_std_mean": local_std_mean,
        "local_std_p10": local_std_p10,
        "low_contrast_ratio": low_contrast_ratio,
        "over_dark_ratio": over_dark_ratio,
        "over_bright_ratio": over_bright_ratio,

        "gradient_energy_mean": gradient_energy_mean,
        "gradient_energy_p10": gradient_energy_p10,
        "weak_gradient_ratio": weak_gradient_ratio,

        "noise_residual_mean": noise_residual_mean,
        "noise_residual_p90": noise_residual_p90,
        "strong_noise_ratio": strong_noise_ratio,
        "noise_block_std": noise_block_std,

        "gbfoe_adaptive_sigma": adaptive_sigma,
    }

    debug_arrays = {
        "normalized": cropped,
        "foreground_mask": foreground_mask,
        "pixel_mask": pixel_mask,
        "orientation": orientation,
        "coherence": coherence,
        "orientation_pixel": orientation_pixel,
        "strength_pixel": strength_pixel,
        "period_map": period_map,
    }

    return scalar_features, debug_arrays, stage_times


# =============================================================================
# Scoring
# =============================================================================

def compute_subscores(features: Dict[str, float]) -> Dict[str, float]:

    # 1. Area
    area_ratio_score = linear_score(features["foreground_ratio"], 0.22, 0.78)
    largest_score = linear_score(features["largest_component_ratio"], 0.70, 0.995)
    center_score = linear_score(features["center_coverage_ratio"], 0.40, 0.95)
    border_score = inverse_linear_score(features["border_touch_ratio"], 0.08, 0.55)

    area_score_01 = (
        0.42 * area_ratio_score
        + 0.18 * largest_score
        + 0.22 * center_score
        + 0.18 * border_score
    )

    # 2. GBFOE coherence
    coherence_mean_score = linear_score(features["coherence_mean"], 0.42, 0.88)
    coherence_p10_score = linear_score(features["coherence_p10"], 0.20, 0.72)
    low_coherence_score = inverse_linear_score(features["low_coherence_ratio"], 0.05, 0.50)
    high_coherence_score = linear_score(features["high_coherence_ratio"], 0.25, 0.85)

    coherence_score_01 = (
        0.35 * coherence_mean_score
        + 0.30 * coherence_p10_score
        + 0.20 * low_coherence_score
        + 0.15 * high_coherence_score
    )

    # 3. Orientation flow
    flow_mean_score = inverse_linear_score(
        features["orientation_difference_mean"], 0.015, 0.22
    )
    flow_p90_score = inverse_linear_score(
        features["orientation_difference_p90"], 0.06, 0.50
    )
    jump_score = inverse_linear_score(
        features["orientation_jump_ratio"], 0.02, 0.32
    )

    flow_score_01 = (
        0.45 * flow_mean_score
        + 0.25 * flow_p90_score
        + 0.30 * jump_score
    )

    if features["orientation_pair_count"] < 4:
        flow_score_01 *= 0.20

    # 4. Ridge frequency consistency
    valid_freq_score = linear_score(
        features["frequency_valid_ratio"], 0.20, 0.80
    )
    cv_score = inverse_linear_score(
        features["ridge_period_cv"], 0.06, 0.32
    )
    freq_neighbor_mean_score = inverse_linear_score(
        features["frequency_neighbor_diff_mean"], 0.35, 3.0
    )
    freq_neighbor_p90_score = inverse_linear_score(
        features["frequency_neighbor_diff_p90"], 1.0, 5.0
    )
    freq_jump_score = inverse_linear_score(
        features["frequency_jump_ratio"], 0.05, 0.45
    )

    frequency_score_01 = (
        0.35 * valid_freq_score
        + 0.20 * cv_score
        + 0.20 * freq_neighbor_mean_score
        + 0.10 * freq_neighbor_p90_score
        + 0.15 * freq_jump_score
    )

    if features["frequency_pair_count"] < 3:
        frequency_score_01 *= 0.55

    # 5. Contrast
    std_mean_score = linear_score(features["local_std_mean"], 0.045, 0.19)
    std_p10_score = linear_score(features["local_std_p10"], 0.020, 0.12)
    low_contrast_score = inverse_linear_score(features["low_contrast_ratio"], 0.05, 0.60)
    exposure_bad_ratio = features["over_dark_ratio"] + features["over_bright_ratio"]
    exposure_score = inverse_linear_score(exposure_bad_ratio, 0.01, 0.35)

    contrast_score_01 = (
        0.35 * std_mean_score
        + 0.30 * std_p10_score
        + 0.25 * low_contrast_score
        + 0.10 * exposure_score
    )

    # 6. Sharpness
    grad_mean_score = linear_score(features["gradient_energy_mean"], 0.0025, 0.040)
    grad_p10_score = linear_score(features["gradient_energy_p10"], 0.0006, 0.016)
    weak_grad_score = inverse_linear_score(features["weak_gradient_ratio"], 0.05, 0.65)

    sharpness_score_01 = (
        0.45 * grad_mean_score
        + 0.25 * grad_p10_score
        + 0.30 * weak_grad_score
    )

    # 7. Noise
    residual_mean_score = inverse_linear_score(
        features["noise_residual_mean"], 0.010, 0.060
    )
    residual_p90_score = inverse_linear_score(
        features["noise_residual_p90"], 0.025, 0.120
    )
    strong_noise_score = inverse_linear_score(
        features["strong_noise_ratio"], 0.02, 0.35
    )
    block_uniformity_score = inverse_linear_score(
        features["noise_block_std"], 0.004, 0.030
    )

    noise_score_01 = (
        0.35 * residual_mean_score
        + 0.30 * residual_p90_score
        + 0.25 * strong_noise_score
        + 0.10 * block_uniformity_score
    )

    return {
        "area_score": 100.0 * clamp01(area_score_01),
        "orientation_coherence_score": 100.0 * clamp01(coherence_score_01),
        "orientation_flow_score": 100.0 * clamp01(flow_score_01),
        "frequency_score": 100.0 * clamp01(frequency_score_01),
        "contrast_score": 100.0 * clamp01(contrast_score_01),
        "sharpness_score": 100.0 * clamp01(sharpness_score_01),
        "noise_score": 100.0 * clamp01(noise_score_01),
    }


def combine_scores(
    subscores: Dict[str, float],
    features: Dict[str, float],
    config: FFQConfig
) -> float:

    weighted_score = (
        config.weight_area * subscores["area_score"]
        + config.weight_coherence * subscores["orientation_coherence_score"]
        + config.weight_flow * subscores["orientation_flow_score"]
        + config.weight_frequency * subscores["frequency_score"]
        + config.weight_contrast * subscores["contrast_score"]
        + config.weight_sharpness * subscores["sharpness_score"]
        + config.weight_noise * subscores["noise_score"]
    )

    # 核心纹理短板惩罚
    weakest_core = min(
        subscores["orientation_coherence_score"],
        subscores["orientation_flow_score"],
        subscores["frequency_score"],
        subscores["contrast_score"],
        subscores["sharpness_score"],
        subscores["noise_score"],
    )

    robust_score = 0.82 * weighted_score + 0.18 * weakest_core

    # V1.5 稍弱于 V1.4.2 的 1.70 压缩，
    # 因为 frequency 已经提供额外区分能力
    calibrated_score = 100.0 * (
        clamp01(robust_score / 100.0) ** 1.55
    )

    # ===== Hard caps =====

    fg_ratio = features["foreground_ratio"]
    if fg_ratio < 0.05:
        calibrated_score = min(calibrated_score, 5.0)
    elif fg_ratio < 0.10:
        calibrated_score = min(calibrated_score, 15.0)
    elif fg_ratio < 0.18:
        calibrated_score = min(calibrated_score, 32.0)
    elif fg_ratio < 0.25:
        calibrated_score = min(calibrated_score, 50.0)

    coherence_mean = features["coherence_mean"]
    if coherence_mean < 0.15:
        calibrated_score = min(calibrated_score, 15.0)
    elif coherence_mean < 0.25:
        calibrated_score = min(calibrated_score, 35.0)
    elif coherence_mean < 0.35:
        calibrated_score = min(calibrated_score, 55.0)

    if features["low_coherence_ratio"] > 0.65:
        calibrated_score = min(calibrated_score, 40.0)
    elif features["low_coherence_ratio"] > 0.45:
        calibrated_score = min(calibrated_score, 60.0)

    if features["strong_noise_ratio"] > 0.40:
        calibrated_score = min(calibrated_score, 45.0)
    elif features["strong_noise_ratio"] > 0.25:
        calibrated_score = min(calibrated_score, 65.0)

    # 边界触碰惩罚：
    # 当前按压式/局部指纹数据中，自动前景可能覆盖整个裁剪区域，
    # 此时 border_touch_sides=4 并不一定意味着质量差。
    # 只有当前景没有几乎填满画面时，才启用原来的边界 hard cap，
    # 避免大量正常样本被机械地卡在 70 分。
    if features["foreground_ratio"] < 0.90:
        if features["border_touch_sides"] >= 4:
            calibrated_score = min(calibrated_score, 70.0)
        elif features["border_touch_sides"] >= 3:
            calibrated_score = min(calibrated_score, 82.0)

    # 新增：几乎估计不出稳定 ridge period，则不允许高分
    if features["frequency_valid_ratio"] < 0.08:
        calibrated_score = min(calibrated_score, 35.0)
    elif features["frequency_valid_ratio"] < 0.18:
        calibrated_score = min(calibrated_score, 55.0)

    return float(np.clip(calibrated_score, 0.0, 100.0))


# =============================================================================
# Single image
# =============================================================================

def score_one_image(
    image_path: Path,
    input_dir: Path,
    config: FFQConfig,
    save_debug: bool,
    debug_dir: Optional[Path],
    mask_dir: Optional[Path] = None,
) -> ImageQualityResult:

    total_start = time.perf_counter()

    read_start = time.perf_counter()
    image = read_grayscale_image(image_path)
    read_time_ms = (time.perf_counter() - read_start) * 1000.0

    height, width = image.shape

    processing_start = time.perf_counter()

    mask_path = find_external_mask(image_path, input_dir, mask_dir)
    external_mask = read_binary_mask(mask_path) if mask_path else None

    features, debug_arrays, stage_times = extract_features(
        image, config, external_mask
    )

    scoring_start = time.perf_counter()
    subscores = compute_subscores(features)
    final_score = combine_scores(subscores, features, config)

    if mask_path:
        final_score = min(
            final_score,
            mask_area_limit(float(external_mask.mean()))
        )

    scoring_time_ms = (time.perf_counter() - scoring_start) * 1000.0
    processing_time_ms = (time.perf_counter() - processing_start) * 1000.0

    if save_debug and debug_dir is not None:
        save_debug_visualization(
            image_path,
            input_dir,
            debug_dir,
            debug_arrays,
            final_score,
        )

    total_time_ms = (time.perf_counter() - total_start) * 1000.0

    try:
        relative_path = str(image_path.relative_to(input_dir))
    except ValueError:
        relative_path = image_path.name

    return ImageQualityResult(
        filename=image_path.name,
        relative_path=relative_path,
        width=width,
        height=height,
        resized_width=int(features["resized_width"]),
        resized_height=int(features["resized_height"]),

        quality_score=round(final_score, 4),
        quality_level=quality_level(final_score),

        area_score=round(subscores["area_score"], 4),
        orientation_coherence_score=round(subscores["orientation_coherence_score"], 4),
        orientation_flow_score=round(subscores["orientation_flow_score"], 4),
        frequency_score=round(subscores["frequency_score"], 4),
        contrast_score=round(subscores["contrast_score"], 4),
        sharpness_score=round(subscores["sharpness_score"], 4),
        noise_score=round(subscores["noise_score"], 4),

        foreground_ratio=round(features["foreground_ratio"], 6),
        largest_component_ratio=round(features["largest_component_ratio"], 6),
        center_coverage_ratio=round(features["center_coverage_ratio"], 6),
        border_touch_ratio=round(features["border_touch_ratio"], 6),
        border_touch_sides=int(features["border_touch_sides"]),

        coherence_mean=round(features["coherence_mean"], 6),
        coherence_std=round(features["coherence_std"], 6),
        coherence_p10=round(features["coherence_p10"], 6),
        low_coherence_ratio=round(features["low_coherence_ratio"], 6),
        high_coherence_ratio=round(features["high_coherence_ratio"], 6),

        orientation_difference_mean=round(features["orientation_difference_mean"], 6),
        orientation_difference_p90=round(features["orientation_difference_p90"], 6),
        orientation_jump_ratio=round(features["orientation_jump_ratio"], 6),
        orientation_pair_count=int(features["orientation_pair_count"]),

        frequency_valid_ratio=round(features["frequency_valid_ratio"], 6),
        ridge_period_mean=round(features["ridge_period_mean"], 6),
        ridge_period_std=round(features["ridge_period_std"], 6),
        ridge_period_cv=round(features["ridge_period_cv"], 6),
        ridge_period_p10=round(features["ridge_period_p10"], 6),
        ridge_period_p90=round(features["ridge_period_p90"], 6),
        frequency_neighbor_diff_mean=round(features["frequency_neighbor_diff_mean"], 6),
        frequency_neighbor_diff_p90=round(features["frequency_neighbor_diff_p90"], 6),
        frequency_jump_ratio=round(features["frequency_jump_ratio"], 6),
        frequency_pair_count=int(features["frequency_pair_count"]),

        local_std_mean=round(features["local_std_mean"], 6),
        local_std_p10=round(features["local_std_p10"], 6),
        low_contrast_ratio=round(features["low_contrast_ratio"], 6),
        over_dark_ratio=round(features["over_dark_ratio"], 6),
        over_bright_ratio=round(features["over_bright_ratio"], 6),

        gradient_energy_mean=round(features["gradient_energy_mean"], 8),
        gradient_energy_p10=round(features["gradient_energy_p10"], 8),
        weak_gradient_ratio=round(features["weak_gradient_ratio"], 6),

        noise_residual_mean=round(features["noise_residual_mean"], 8),
        noise_residual_p90=round(features["noise_residual_p90"], 8),
        strong_noise_ratio=round(features["strong_noise_ratio"], 6),
        noise_block_std=round(features["noise_block_std"], 8),

        read_time_ms=round(read_time_ms, 4),
        preprocess_time_ms=round(stage_times["preprocess_time_ms"], 4),
        foreground_time_ms=round(stage_times["foreground_time_ms"], 4),
        orientation_time_ms=round(stage_times["orientation_time_ms"], 4),
        frequency_time_ms=round(stage_times["frequency_time_ms"], 4),
        feature_time_ms=round(stage_times["feature_time_ms"], 4),
        scoring_time_ms=round(scoring_time_ms, 4),
        processing_time_ms=round(processing_time_ms, 4),
        total_time_ms=round(total_time_ms, 4),
    )


# =============================================================================
# Debug visualization
# =============================================================================

def save_debug_visualization(
    image_path: Path,
    input_dir: Path,
    debug_dir: Path,
    debug_arrays: Dict[str, np.ndarray],
    final_score: float,
) -> None:
    try:
        relative = image_path.relative_to(input_dir)
    except ValueError:
        relative = Path(image_path.name)

    stem_dir = debug_dir / relative.parent
    stem_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    normalized = debug_arrays["normalized"]
    foreground_mask = debug_arrays["foreground_mask"]
    orientation = debug_arrays["orientation"]
    coherence = debug_arrays["coherence"]
    period_map = debug_arrays["period_map"]

    image_u8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    base_bgr = cv2.cvtColor(image_u8, cv2.COLOR_GRAY2BGR)

    gh, gw = foreground_mask.shape
    bh = normalized.shape[0] // gh
    bw = normalized.shape[1] // gw

    mask_pixel = cv2.resize(
        foreground_mask.astype(np.uint8) * 255,
        (normalized.shape[1], normalized.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    write_image(
        stem_dir / f"{stem}_foreground.png",
        cv2.cvtColor(mask_pixel, cv2.COLOR_GRAY2BGR)
    )

    coherence_u8 = np.clip(coherence * 255.0, 0, 255).astype(np.uint8)
    coherence_big = cv2.resize(
        coherence_u8,
        (normalized.shape[1], normalized.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    coherence_heatmap = cv2.applyColorMap(
        coherence_big, cv2.COLORMAP_JET
    )
    write_image(stem_dir / f"{stem}_coherence_gbfoe.png", coherence_heatmap)

    # Orientation overlay
    overlay = base_bgr.copy()
    line_length = int(0.36 * min(bh, bw))
    for r in range(gh):
        for c in range(gw):
            if not foreground_mask[r, c]:
                continue

            theta = float(orientation[r, c])
            q = float(coherence[r, c])
            cx = int((c + 0.5) * bw)
            cy = int((r + 0.5) * bh)
            dx = int(round(line_length * math.cos(theta)))
            dy = int(round(line_length * math.sin(theta)))

            intensity = int(np.clip(80 + 175 * q, 0, 255))
            cv2.line(
                overlay,
                (cx - dx, cy - dy),
                (cx + dx, cy + dy),
                (0, intensity, 255 - intensity // 3),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        overlay,
        f"FFQ V1.5: {final_score:.2f}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    write_image(stem_dir / f"{stem}_orientation_gbfoe.png", overlay)

    # Frequency map
    freq_vis = np.zeros_like(period_map, dtype=np.uint8)
    valid = period_map > 0
    if np.any(valid):
        pmin = float(period_map[valid].min())
        pmax = float(period_map[valid].max())
        if pmax > pmin:
            freq_vis[valid] = np.clip(
                (period_map[valid] - pmin) * 255.0 / (pmax - pmin),
                0, 255
            ).astype(np.uint8)
        else:
            freq_vis[valid] = 127

    freq_big = cv2.resize(
        freq_vis,
        (normalized.shape[1], normalized.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    freq_heat = cv2.applyColorMap(freq_big, cv2.COLORMAP_TURBO)
    freq_heat[cv2.resize(
        (~valid).astype(np.uint8),
        (normalized.shape[1], normalized.shape[0]),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)] = 0
    write_image(stem_dir / f"{stem}_ridge_period.png", freq_heat)


# =============================================================================
# Output helpers
# =============================================================================

def result_to_error_row(
    image_path: Path,
    input_dir: Path,
    error: str,
    total_time_ms: float,
) -> ImageQualityResult:
    try:
        relative_path = str(image_path.relative_to(input_dir))
    except ValueError:
        relative_path = image_path.name

    zeros = dict(
        filename=image_path.name,
        relative_path=relative_path,
        width=0, height=0, resized_width=0, resized_height=0,
        quality_score=0.0, quality_level="处理失败",
        area_score=0.0, orientation_coherence_score=0.0,
        orientation_flow_score=0.0, frequency_score=0.0,
        contrast_score=0.0, sharpness_score=0.0, noise_score=0.0,
        foreground_ratio=0.0, largest_component_ratio=0.0,
        center_coverage_ratio=0.0, border_touch_ratio=0.0, border_touch_sides=0,
        coherence_mean=0.0, coherence_std=0.0, coherence_p10=0.0,
        low_coherence_ratio=0.0, high_coherence_ratio=0.0,
        orientation_difference_mean=0.0, orientation_difference_p90=0.0,
        orientation_jump_ratio=0.0, orientation_pair_count=0,
        frequency_valid_ratio=0.0, ridge_period_mean=0.0, ridge_period_std=0.0,
        ridge_period_cv=0.0, ridge_period_p10=0.0, ridge_period_p90=0.0,
        frequency_neighbor_diff_mean=0.0, frequency_neighbor_diff_p90=0.0,
        frequency_jump_ratio=0.0, frequency_pair_count=0,
        local_std_mean=0.0, local_std_p10=0.0, low_contrast_ratio=0.0,
        over_dark_ratio=0.0, over_bright_ratio=0.0,
        gradient_energy_mean=0.0, gradient_energy_p10=0.0, weak_gradient_ratio=0.0,
        noise_residual_mean=0.0, noise_residual_p90=0.0,
        strong_noise_ratio=0.0, noise_block_std=0.0,
        read_time_ms=0.0, preprocess_time_ms=0.0, foreground_time_ms=0.0,
        orientation_time_ms=0.0, frequency_time_ms=0.0, feature_time_ms=0.0,
        scoring_time_ms=0.0, processing_time_ms=0.0,
        total_time_ms=round(total_time_ms, 4),
        error=error,
    )
    return ImageQualityResult(**zeros)


def write_results_csv(output_path: Path, results: Sequence[ImageQualityResult]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys()) if results else []
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def build_score_bin_counts(
    results: Sequence[ImageQualityResult]
) -> List[Dict[str, object]]:
    valid_scores = np.asarray(
        [r.quality_score for r in results if not r.error],
        dtype=np.float64,
    )
    rows = []
    for low in range(0, 100, 10):
        high = low + 10
        if high < 100:
            count = int(np.sum((valid_scores >= low) & (valid_scores < high)))
            label = f"{low}-{high - 1}"
        else:
            count = int(np.sum((valid_scores >= low) & (valid_scores <= high)))
            label = "90-100"

        ratio = count / valid_scores.size if valid_scores.size > 0 else 0.0
        rows.append({
            "score_range": label,
            "count": count,
            "ratio": round(float(ratio), 6),
        })
    return rows


def write_score_bin_csv(output_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["score_range", "count", "ratio"],
        )
        writer.writeheader()
        writer.writerows(rows)


def score_to_bin_name(score: float) -> str:
    """将 0-100 分映射到固定的 10 分段文件夹名。"""
    score = float(np.clip(score, 0.0, 100.0))
    if score >= 90.0:
        return "90-100"
    low = int(score // 10) * 10
    high = low + 9
    return f"{low:02d}-{high:02d}"


def save_images_by_score_bins(
    input_dir: Path,
    output_dir: Path,
    results: Sequence[ImageQualityResult],
) -> Dict[str, int]:
    """
    将成功评分的原始图片按最终 quality_score 分段复制。

    输出结构：
      output_dir/score_bins/00-09/...
      output_dir/score_bins/10-19/...
      ...
      output_dir/score_bins/90-100/...

    如果输入使用 --recursive，保留原始相对目录结构，避免同名文件冲突。
    """
    bins_root = output_dir / "score_bins"
    bins_root.mkdir(parents=True, exist_ok=True)

    # 预先创建所有分段，便于即使某段为 0 张也能直接查看
    all_bins = [
        "00-09", "10-19", "20-29", "30-39", "40-49",
        "50-59", "60-69", "70-79", "80-89", "90-100",
    ]
    counts = {name: 0 for name in all_bins}
    for name in all_bins:
        (bins_root / name).mkdir(parents=True, exist_ok=True)

    for result in results:
        if result.error:
            continue

        src = input_dir / Path(result.relative_path)
        if not src.exists():
            print(f"警告：分段复制时找不到原图：{src}")
            continue

        bin_name = score_to_bin_name(result.quality_score)
        dst = bins_root / bin_name / Path(result.relative_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        counts[bin_name] += 1

    # 同时输出一个分段复制统计，便于核对
    with (bins_root / "score_bin_image_counts.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=["score_range", "image_count"])
        writer.writeheader()
        for name in all_bins:
            writer.writerow({
                "score_range": name,
                "image_count": counts[name],
            })

    return counts


def summarize_results(
    results: Sequence[ImageQualityResult]
) -> Dict[str, object]:
    valid = [r for r in results if not r.error]
    failed = [r for r in results if r.error]

    summary = {
        "total_images": len(results),
        "successful_images": len(valid),
        "failed_images": len(failed),
    }

    level_counts = {"较差": 0, "一般": 0, "良好": 0, "优秀": 0}
    for r in valid:
        level_counts[r.quality_level] = level_counts.get(r.quality_level, 0) + 1

    summary["quality_level_counts"] = level_counts
    summary["score_bin_counts"] = build_score_bin_counts(results)

    if valid:
        scores = [r.quality_score for r in valid]
        processing_times = [r.processing_time_ms for r in valid]
        total_times = [r.total_time_ms for r in valid]

        summary.update({
            "score_mean": float(np.mean(scores)),
            "score_median": float(np.median(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "score_std": float(np.std(scores)),

            "processing_time_mean_ms": float(np.mean(processing_times)),
            "processing_time_median_ms": float(np.median(processing_times)),
            "processing_time_p95_ms": percentile(processing_times, 95),
            "processing_time_p99_ms": percentile(processing_times, 99),
            "processing_time_min_ms": float(np.min(processing_times)),
            "processing_time_max_ms": float(np.max(processing_times)),
            "processing_time_std_ms": float(np.std(processing_times)),

            "total_time_mean_ms": float(np.mean(total_times)),
            "total_time_p95_ms": percentile(total_times, 95),

            "subscore_mean": {
                "area": float(np.mean([r.area_score for r in valid])),
                "coherence": float(np.mean([r.orientation_coherence_score for r in valid])),
                "flow": float(np.mean([r.orientation_flow_score for r in valid])),
                "frequency": float(np.mean([r.frequency_score for r in valid])),
                "contrast": float(np.mean([r.contrast_score for r in valid])),
                "sharpness": float(np.mean([r.sharpness_score for r in valid])),
                "noise": float(np.mean([r.noise_score for r in valid])),
            },
        })

    return summary


def save_summary(
    output_dir: Path,
    summary: Dict[str, object],
    config: FFQConfig
) -> None:
    data = {
        "algorithm": "FFQ-Lite V1.5 Traditional-Enhanced",
        "config": asdict(config),
        "summary": summary,
    }

    with (output_dir / "ffq_summary.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    lines = [
        "=" * 84,
        "FFQ-Lite V1.5 Traditional-Enhanced 指纹质量评分汇总",
        "=" * 84,
        f"图像总数：{summary.get('total_images', 0)}",
        f"成功数量：{summary.get('successful_images', 0)}",
        f"失败数量：{summary.get('failed_images', 0)}",
        "",
    ]

    if summary.get("successful_images", 0):
        counts = summary["quality_level_counts"]
        lines += [
            "质量等级数量：",
            f"  较差（0-39）   ：{counts.get('较差', 0)}",
            f"  一般（40-59）  ：{counts.get('一般', 0)}",
            f"  良好（60-79）  ：{counts.get('良好', 0)}",
            f"  优秀（80-100） ：{counts.get('优秀', 0)}",
            "",
            "分数统计：",
            f"  平均分：{summary['score_mean']:.4f}",
            f"  中位数：{summary['score_median']:.4f}",
            f"  最低分：{summary['score_min']:.4f}",
            f"  最高分：{summary['score_max']:.4f}",
            f"  标准差：{summary['score_std']:.4f}",
            "",
            "七项子分数平均值：",
        ]
        for k, v in summary["subscore_mean"].items():
            lines.append(f"  {k:<10s}: {v:.4f}")

        lines += [
            "",
            "算法处理耗时（质量优先版，不以速度为目标）：",
            f"  平均耗时：{summary['processing_time_mean_ms']:.4f} ms",
            f"  中位数：{summary['processing_time_median_ms']:.4f} ms",
            f"  P95：{summary['processing_time_p95_ms']:.4f} ms",
            f"  P99：{summary['processing_time_p99_ms']:.4f} ms",
            f"  最大耗时：{summary['processing_time_max_ms']:.4f} ms",
        ]

    (output_dir / "ffq_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


def save_distribution_plots(
    output_dir: Path,
    results: Sequence[ImageQualityResult]
) -> None:
    valid = [r for r in results if not r.error]
    if not valid:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("提示：未安装 matplotlib，跳过分布图。")
        return

    scores = np.asarray([r.quality_score for r in valid], dtype=np.float64)

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=np.arange(0, 105, 5), edgecolor="black")
    plt.axvline(np.mean(scores), linestyle="--", label=f"Mean={np.mean(scores):.2f}")
    plt.axvline(np.median(scores), linestyle=":", label=f"Median={np.median(scores):.2f}")
    plt.xlim(0, 100)
    plt.xlabel("FFQ quality score")
    plt.ylabel("Image count")
    plt.title("FFQ-Lite V1.5 Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=180)
    plt.close()

    subscore_data = [
        [r.area_score for r in valid],
        [r.orientation_coherence_score for r in valid],
        [r.orientation_flow_score for r in valid],
        [r.frequency_score for r in valid],
        [r.contrast_score for r in valid],
        [r.sharpness_score for r in valid],
        [r.noise_score for r in valid],
    ]
    labels = [
        "Area", "Coherence", "Flow", "Frequency",
        "Contrast", "Sharpness", "Noise"
    ]

    plt.figure(figsize=(12, 6))
    plt.boxplot(subscore_data, tick_labels=labels, showmeans=True)
    plt.ylim(0, 100)
    plt.ylabel("Subscore")
    plt.title("FFQ-Lite V1.5 Subscore Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "subscore_distribution.png", dpi=180)
    plt.close()


def print_progress(index: int, total: int, result: ImageQualityResult) -> None:
    if result.error:
        print(f"[{index:>5}/{total}] 失败 {result.relative_path} | {result.error}")
    else:
        print(
            f"[{index:>5}/{total}] "
            f"分数={result.quality_score:6.2f} "
            f"等级={result.quality_level:<2} "
            f"耗时={result.processing_time_ms:8.2f} ms "
            f"| C={result.orientation_coherence_score:5.1f} "
            f"F={result.orientation_flow_score:5.1f} "
            f"RF={result.frequency_score:5.1f} "
            f"| {result.relative_path}"
        )


def validate_config(config: FFQConfig) -> None:
    if config.block_size < 4:
        raise ValueError("block_size 不应小于 4")
    if config.max_side != 0 and config.max_side < 64:
        raise ValueError("max_side 建议不小于 64，或设置为 0")
    if config.sobel_ksize not in {1, 3, 5, 7}:
        raise ValueError("sobel_ksize 只允许 1、3、5、7")

    weights = [
        config.weight_area,
        config.weight_coherence,
        config.weight_flow,
        config.weight_frequency,
        config.weight_contrast,
        config.weight_sharpness,
        config.weight_noise,
    ]
    if any(w < 0 for w in weights):
        raise ValueError("权重不能为负数")
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"七项权重之和必须为 1，当前为 {sum(weights):.6f}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FFQ-Lite V1.5 Traditional-Enhanced"
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("./ffq_v15_results"))
    parser.add_argument("--mask_dir", type=Path, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max_side", type=int, default=320)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--sobel_ksize", type=int, default=3, choices=[1, 3, 5, 7])
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument(
        "--no_score_bins",
        action="store_true",
        help="不按最终分数把原图复制到 score_bins 分段文件夹",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = args.mask_dir.expanduser().resolve() if args.mask_dir else None

    config = FFQConfig(
        max_side=args.max_side,
        block_size=args.block_size,
        sobel_ksize=args.sobel_ksize,
    )
    validate_config(config)

    images = discover_images(input_dir, args.recursive)
    if not images:
        print("没有找到支持的图像。", file=sys.stderr)
        return 1

    print("=" * 84)
    print("FFQ-Lite V1.5 Traditional-Enhanced")
    print("=" * 84)
    print(f"输入目录：{input_dir}")
    print(f"输出目录：{output_dir}")
    print(f"外部Mask：{mask_dir if mask_dir else 'None (自动前景)'}")
    print(f"图像数量：{len(images)}")
    print(f"最长边：{config.max_side}")
    print(f"Block：{config.block_size}")
    print("方向方法：GBFOE-style traditional orientation field")
    print("频率方法：X-Signature ridge period consistency")
    print("模式：质量优先（不优化速度）")
    print()

    debug_dir = output_dir / "debug" if args.save_debug else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    results: List[ImageQualityResult] = []

    batch_start = time.perf_counter()

    for i, image_path in enumerate(images, start=1):
        item_start = time.perf_counter()
        try:
            result = score_one_image(
                image_path=image_path,
                input_dir=input_dir,
                config=config,
                save_debug=args.save_debug,
                debug_dir=debug_dir,
                mask_dir=mask_dir,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - item_start) * 1000.0
            result = result_to_error_row(
                image_path,
                input_dir,
                f"{type(exc).__name__}: {exc}",
                elapsed_ms,
            )

        results.append(result)
        print_progress(i, len(images), result)

    batch_time_s = time.perf_counter() - batch_start

    csv_path = output_dir / "fingerprint_quality_results.csv"
    write_results_csv(csv_path, results)

    rows = build_score_bin_counts(results)
    write_score_bin_csv(output_dir / "score_distribution_counts.csv", rows)

    summary = summarize_results(results)
    summary["batch_wall_time_s"] = batch_time_s
    summary["batch_wall_fps"] = len(images) / batch_time_s if batch_time_s > 0 else 0.0

    save_summary(output_dir, summary, config)
    save_distribution_plots(output_dir, results)

    score_bin_image_counts = None
    if not args.no_score_bins:
        score_bin_image_counts = save_images_by_score_bins(
            input_dir=input_dir,
            output_dir=output_dir,
            results=results,
        )

    print()
    print("=" * 84)
    print("处理完成")
    print("=" * 84)
    print(f"结果 CSV：{csv_path}")
    print(f"汇总 TXT：{output_dir / 'ffq_summary.txt'}")
    print(f"汇总 JSON：{output_dir / 'ffq_summary.json'}")
    print(f"分数统计：{output_dir / 'score_distribution_counts.csv'}")
    print(f"分数图：{output_dir / 'score_distribution.png'}")
    print(f"子分数图：{output_dir / 'subscore_distribution.png'}")
    if score_bin_image_counts is not None:
        print(f"分段原图：{output_dir / 'score_bins'}")
        print("分段复制数量：")
        for bin_name, count in score_bin_image_counts.items():
            print(f"  {bin_name}: {count}")
    if args.save_debug:
        print(f"调试图：{debug_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
