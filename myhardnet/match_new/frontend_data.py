"""按图像相对路径关联前端分割掩码和 FFQ-Lite 质量记录。"""

from __future__ import annotations

import hashlib
import math
import warnings
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import cv2
import numpy as np

from match_new.utils import read_csv_rows, resolve_path


FRONTEND_ROW_FIELDS = (
    "frontend_relative_path",
    "frontend_mask_path",
    "frontend_quality_score",
    "frontend_quality_csv",
)


def frontend_section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    return dict(dict(config.get("frontend", {})).get(name, {}))


def quality_score_or_none(value: Any) -> float | None:
    """未知、失败、非有限或超出 0–100 的质量不能作为低分使用。"""

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) and 0.0 <= score <= 100.0 else None


def _relative_key(value: str) -> str:
    """同时接受 Windows/POSIX 相对路径，避免不同手指的 pair_N 串配。"""

    text = str(value).strip().replace("\\", "/")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or PureWindowsPath(text).drive
        or ".." in path.parts
        or len(path.parts) < 2
    ):
        raise ValueError(f"前端 relative_path 必须为 <手指目录>/<图像文件>：{value!r}")
    return path.as_posix().casefold()


def _missing(message: str, section: Mapping[str, Any]) -> None:
    policy = str(section.get("missing_policy", "error")).lower()
    if policy == "error":
        raise ValueError(message)
    if policy != "warn":
        raise ValueError(f"frontend missing_policy 仅支持 error 或 warn，收到 {policy!r}")
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def attach_frontend_metadata(
    rows: list[dict[str, str]], config: dict[str, Any]
) -> list[dict[str, str]]:
    """一次读取质量 CSV，给每张图补充掩码路径和质量；关闭时保持旧行为。"""

    mask_cfg = frontend_section(config, "mask")
    quality_cfg = frontend_section(config, "quality")
    mask_enabled = bool(mask_cfg.get("enabled", False))
    quality_enabled = bool(quality_cfg.get("enabled", False))
    if not mask_enabled and not quality_enabled:
        return rows
    for section in (mask_cfg, quality_cfg):
        if str(section.get("missing_policy", "error")).lower() not in {"error", "warn"}:
            raise ValueError("frontend missing_policy 仅支持 error 或 warn")
    image_root = resolve_path(config, dict(config.get("data", {}))["image_root"]).resolve()
    mask_root: Path | None = None
    if mask_enabled:
        if not mask_cfg.get("root"):
            raise ValueError("frontend.mask.root is required when mask filtering is enabled.")
        mask_root = resolve_path(config, mask_cfg["root"]).resolve()
        if not mask_root.is_dir():
            raise FileNotFoundError(f"前端掩码目录不存在：{mask_root}")

    quality_path: Path | None = None
    quality_index: dict[str, dict[str, str]] = {}
    if quality_enabled:
        if not quality_cfg.get("csv_path"):
            raise ValueError("frontend.quality.csv_path is required when frontend quality is enabled.")
        quality_path = resolve_path(config, quality_cfg["csv_path"]).resolve()
        if not quality_path.is_file():
            raise FileNotFoundError(f"前端质量 CSV 不存在：{quality_path}")
        records = read_csv_rows(quality_path)
        if not records or not {"relative_path", "quality_score"}.issubset(records[0]):
            raise ValueError(f"质量 CSV 必须含 relative_path、quality_score 列和图像记录：{quality_path}")
        for record in records:
            key = _relative_key(record["relative_path"])
            if key in quality_index:
                raise ValueError(f"质量 CSV 中 relative_path 重复：{record['relative_path']}")
            quality_index[key] = record

    enriched: list[dict[str, str]] = []
    for source_row in rows:
        row = dict(source_row)
        path = Path(row["image_path"]).expanduser().resolve()
        try:
            relative = path.relative_to(image_root)
        except ValueError as exc:
            raise ValueError(f"图像不在 data.image_root 下：{path}") from exc
        key = _relative_key(relative.as_posix())
        row.update(
            frontend_relative_path=relative.as_posix(),
            frontend_mask_path="",
            frontend_quality_score="",
            frontend_quality_csv=str(quality_path) if quality_path else "",
        )
        if mask_root is not None:
            mask_path = mask_root / relative
            if not mask_path.is_file():
                _missing(f"图像 {relative} 缺少同路径前端掩码：{mask_path}", mask_cfg)
            else:
                row["frontend_mask_path"] = str(mask_path)
        if quality_path is not None:
            record = quality_index.get(key)
            score = quality_score_or_none(record.get("quality_score")) if record else None
            error = str(record.get("error", "")).strip() if record else "缺少 relative_path 记录"
            if score is None or error:
                _missing(
                    f"图像 {relative} 没有有效前端质量分数：{error or 'quality_score 非有限或不在 [0,100]'}；"
                    "missing_policy=warn 时使用基础匹配参数，不按质量 0 分处理。",
                    quality_cfg,
                )
            else:
                row["frontend_quality_score"] = str(score)
        enriched.append(row)
    return enriched


def load_frontend_mask(
    row: Mapping[str, Any], image_shape: tuple[int, int], config: Mapping[str, Any]
) -> tuple[np.ndarray | None, str]:
    """读取 0/255 掩码并验证坐标系；尺寸不符直接报错，不静默缩放。"""

    cfg = frontend_section(config, "mask")
    if not bool(cfg.get("enabled", False)):
        return None, ""
    path = str(row.get("frontend_mask_path", ""))
    if not path:
        _missing(f"图像 {row.get('image_path', '')} 缺少前端掩码", cfg)
        return None, ""
    raw = np.fromfile(path, dtype=np.uint8)
    gray = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE) if raw.size else None
    if gray is None:
        raise ValueError(f"无法读取前端掩码：{path}")
    if gray.shape != tuple(image_shape):
        raise ValueError(
            f"前端掩码尺寸 {gray.shape} 与图像尺寸 {image_shape} 不一致：{path}。"
            "请使用同一前端运行的最终图像和掩码，或重新生成掩码。"
        )
    threshold = int(cfg.get("binarization_threshold", 128))
    if not 1 <= threshold <= 255:
        raise ValueError("frontend.mask.binarization_threshold 必须在 [1,255]")
    mask = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    digest = hashlib.sha256()
    digest.update(np.asarray(mask.shape, dtype=np.int32).tobytes())
    digest.update(mask.tobytes())
    return mask, digest.hexdigest()


def refresh_frontend_quality(
    template: dict[str, Any], row: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """匹配使用本次 CSV 的分数；重调质量分数时无需重新推理描述子。"""

    updated = dict(template)
    enabled = bool(frontend_section(config, "quality").get("enabled", False))
    updated["frontend_quality_score"] = (
        quality_score_or_none(row.get("frontend_quality_score")) if enabled else None
    )
    updated["frontend_quality_csv"] = str(row.get("frontend_quality_csv", "")) if enabled else ""
    return updated


def validate_frontend_template(
    template: Mapping[str, Any], row: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    """复用模板时校验掩码处理状态及内容，拒绝旧的未过滤/不同掩码模板。"""

    mask_enabled = bool(frontend_section(config, "mask").get("enabled", False))
    expected_applied = mask_enabled and bool(row.get("frontend_mask_path"))
    saved_applied = bool(template.get("frontend_mask_applied", False))
    message = "请关闭 --skip-template-build，重新构建模板。"
    if saved_applied != expected_applied:
        raise ValueError(f"模板掩码过滤状态与本次配置不同：{template.get('template_path', '')}。{message}")
    if mask_enabled or saved_applied or bool(frontend_section(config, "quality").get("enabled", False)):
        saved_path = Path(str(template.get("image_path", ""))).expanduser().resolve()
        current_path = Path(str(row["image_path"])).expanduser().resolve()
        if saved_path != current_path:
            raise ValueError(f"模板来自不同输入图像目录：{saved_path} != {current_path}。{message}")
    if expected_applied:
        image = np.asarray(template.get("overlap_image", np.empty((0, 0))))
        if image.ndim != 2 or not image.size:
            raise ValueError(f"模板缺少掩码坐标系对应的 overlap_image。{message}")
        _mask, signature = load_frontend_mask(row, image.shape, config)
        if signature != str(template.get("frontend_mask_signature", "")):
            raise ValueError(f"前端掩码内容或二值化设置已经改变：{row.get('frontend_mask_path', '')}。{message}")
