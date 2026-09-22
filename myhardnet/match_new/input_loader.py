"""Raw fingerprint image discovery and input validation."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .utils import resolve_path


DEFAULT_IMAGE_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def _normalize_extensions(values: Iterable[str] | None) -> set[str]:
    extensions: set[str] = set()
    for value in values or DEFAULT_IMAGE_EXTENSIONS:
        extension = str(value).strip().lower()
        if not extension:
            continue
        extensions.add(extension if extension.startswith(".") else f".{extension}")
    if not extensions:
        raise ValueError("data.image_extensions must contain at least one extension.")
    return extensions


def _is_readable_image(path: Path) -> bool:
    raw = np.fromfile(str(path), dtype=np.uint8)
    return bool(raw.size and cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE) is not None)


def _duplicate_safe_image_ids(
    items: list[tuple[Path, Path, str]],
) -> dict[Path, str]:
    """Keep simple stems where possible and hash only duplicate stems."""

    counts = Counter((identity_id, path.stem) for path, _relative, identity_id in items)
    image_ids: dict[Path, str] = {}
    for path, relative, identity_id in items:
        image_id = path.stem
        if counts[(identity_id, image_id)] > 1:
            digest = hashlib.sha1(relative.as_posix().encode("utf-8")).hexdigest()[:10]
            image_id = f"{image_id}__{digest}"
        image_ids[path] = image_id
    return image_ids


def scan_image_metadata(
    image_root: str | Path,
    identity_depth: int,
    image_extensions: Iterable[str] | None = None,
    validate_readable: bool = True,
) -> list[dict[str, str]]:
    """Scan raw images and derive identity IDs from leading directories."""

    root = Path(image_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Raw image directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Raw image path is not a directory: {root}")
    if int(identity_depth) <= 0:
        raise ValueError(f"data.identity_depth must be greater than 0, got {identity_depth}.")

    allowed = _normalize_extensions(image_extensions)
    items: list[tuple[Path, Path, str]] = []
    shallow_paths: list[Path] = []
    bad_paths: list[Path] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed:
            continue
        relative = path.relative_to(root)
        directory_parts = relative.parts[:-1]
        if len(directory_parts) < int(identity_depth):
            shallow_paths.append(relative)
            continue
        if validate_readable and not _is_readable_image(path):
            bad_paths.append(relative)
            continue
        identity_id = "/".join(directory_parts[: int(identity_depth)])
        items.append((path.resolve(), relative, identity_id))

    if shallow_paths:
        examples = ", ".join(path.as_posix() for path in shallow_paths[:5])
        raise ValueError(
            f"{len(shallow_paths)} image(s) do not have {identity_depth} identity directory level(s) "
            f"under {root}. Examples: {examples}"
        )
    if bad_paths:
        examples = ", ".join(path.as_posix() for path in bad_paths[:5])
        raise ValueError(f"{len(bad_paths)} unreadable image(s) found under {root}. Examples: {examples}")
    if not items:
        suffixes = ", ".join(sorted(allowed))
        raise RuntimeError(f"No raw images with extensions [{suffixes}] found under {root}.")

    image_ids = _duplicate_safe_image_ids(items)
    rows = [
        {
            "identity_id": identity_id,
            "image_id": image_ids[path],
            "image_path": str(path),
            "split": "",
        }
        for path, _relative, identity_id in items
    ]
    keys = [(row["identity_id"], row["image_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Raw image indexing produced duplicate (identity_id, image_id) keys.")
    return rows


def load_raw_image_metadata(config: dict[str, Any]) -> list[dict[str, str]]:
    """Load matching input exclusively from data.image_root."""

    data_cfg = dict(config.get("data", {}))
    image_root = data_cfg.get("image_root")
    if not image_root:
        raise ValueError("data.image_root is required.")
    return scan_image_metadata(
        resolve_path(config, image_root),
        identity_depth=int(data_cfg.get("identity_depth", 1)),
        image_extensions=data_cfg.get("image_extensions", DEFAULT_IMAGE_EXTENSIONS),
        validate_readable=bool(data_cfg.get("validate_readable", True)),
    )


def validate_identity_image_counts(
    rows: list[dict[str, str]],
    minimum: int,
    *,
    context: str,
) -> list[dict[str, str]]:
    """保留图像数足够的身份；不足的跳过并打印提示，而不是直接终止。

    每个身份至少需要 ``minimum`` 张图（通常为注册数 + 1 张查询）。
    若全部身份都被跳过，则仍报错。
    """

    groups: dict[str, int] = defaultdict(int)
    for row in rows:
        groups[str(row["identity_id"])] += 1
    min_count = int(minimum)
    insufficient = {
        identity_id
        for identity_id, count in groups.items()
        if count < min_count
    }
    for identity_id in sorted(insufficient):
        count = groups[identity_id]
        print(
            f"[跳过] 身份目录 `{identity_id}` 仅有 {count} 张图"
            f"（{context}），少于所需的 {min_count} 张"
            f"（注册 {min_count - 1} + 查询 1），已跳过该目录。"
        )
    kept = [row for row in rows if str(row["identity_id"]) not in insufficient]
    if not kept:
        raise ValueError(
            f"全部 {len(insufficient)} 个身份在 {context} 中图像数都少于 "
            f"{min_count}，无法继续。每个身份至少需要 "
            f"{min_count - 1} 张注册图和 1 张查询图。"
        )
    if insufficient:
        print(
            f"[提示] 已跳过 {len(insufficient)} 个图像数不足的身份目录，"
            f"剩余 {len(groups) - len(insufficient)} 个身份继续处理。"
        )
    return kept
