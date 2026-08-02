"""match_new 实验通用工具函数。

创建目录、读取 YAML、解析配置相对路径、读写 CSV/JSON、生成安全模板名。
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置；mapping 深合并，其他值由子配置完整覆盖。"""

    merged = dict(base)
    for key, value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            merged[key] = _merge_config(base_value, value)
        else:
            merged[key] = value
    return merged


def _load_config_mapping(
    config_path: Path,
    stack: tuple[Path, ...] = (),
) -> tuple[dict[str, Any], list[str]]:
    """读取单个 YAML，并解析同目录下可选的 ``extends`` 父配置。"""

    if config_path in stack:
        chain = " -> ".join(str(path) for path in (*stack, config_path))
        raise ValueError(f"Circular config inheritance: {chain}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    parent_value = raw.pop("extends", None)
    if parent_value is None or parent_value == "":
        return raw, [str(config_path)]
    if not isinstance(parent_value, (str, Path)):
        raise ValueError(
            f"Config extends must be a path string, got {type(parent_value).__name__}: {config_path}"
        )
    parent_path = Path(parent_value).expanduser()
    if not parent_path.is_absolute():
        parent_path = (config_path.parent / parent_path).resolve()
    if parent_path.parent != config_path.parent:
        raise ValueError(
            "Inherited match configs must be in the same directory so relative data/model "
            f"paths keep one origin: child={config_path}, parent={parent_path}"
        )
    parent, sources = _load_config_mapping(parent_path, (*stack, config_path))
    return _merge_config(parent, raw), [*sources, str(config_path)]


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，并返回 Path 对象。"""

    target = Path(path).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML，解析可选继承，并记录路径供相对路径与运行清单使用。"""

    config_path = Path(path).expanduser().resolve()
    config, sources = _load_config_mapping(config_path)
    config["_config_path"] = str(config_path)
    config["_config_sources"] = sources
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    """把配置中的路径解析为绝对路径。"""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(config["_config_path"]).parent / path).resolve()


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """读取 CSV 文件，返回字典行列表。"""

    with Path(path).expanduser().open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    """写 CSV；未传 fieldnames 时按 rows 中字段首次出现顺序收集列名。"""

    target = Path(path).expanduser()
    ensure_dir(target.parent)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target


def write_json(path: str | Path, payload: Any) -> Path:
    """写 UTF-8 JSON，保留中文。"""

    target = Path(path).expanduser()
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def write_yaml(path: str | Path, payload: Any) -> Path:
    """写 UTF-8 YAML，保留中文。"""

    target = Path(path).expanduser()
    ensure_dir(target.parent)
    target.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return target


def safe_id(value: str) -> str:
    """把 identity_id/image_id 转为适合文件名使用的安全字符串。"""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def template_filename(identity_id: str, image_id: str) -> str:
    """生成单张图像模板 `.npz` 文件名。"""

    return f"{safe_id(identity_id)}__{safe_id(image_id)}.npz"
