"""匹配模板共享的描述子类型、维度和距离度量契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


FLOAT_DESCRIPTOR_KIND = "float"
L2_DISTANCE_METRIC = "l2"
FLOAT32_STORAGE = "float32"
DESCRIPTOR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DescriptorContract:
    """一组描述子在模板中的不可变解释方式。"""

    source: str
    kind: str
    metric: str
    dimension: int
    storage: str


def _template_label(template: Mapping[str, Any]) -> str:
    return str(
        template.get(
            "template_path",
            template.get("image_id", "<in-memory-template>"),
        )
    )


def _normalize_float_kind(value: Any) -> str:
    key = str(value or FLOAT_DESCRIPTOR_KIND).strip().lower()
    aliases = {
        "float": FLOAT_DESCRIPTOR_KIND,
        "float32": FLOAT_DESCRIPTOR_KIND,
        "continuous": FLOAT_DESCRIPTOR_KIND,
    }
    return aliases.get(key, key)


def _normalize_metric(value: Any) -> str:
    key = str(value or L2_DISTANCE_METRIC).strip().lower()
    aliases = {
        "euclidean": L2_DISTANCE_METRIC,
        "l2": L2_DISTANCE_METRIC,
    }
    return aliases.get(key, key)


def _positive_dimension(value: Any, *, field: str, label: str) -> int:
    dimension = int(value)
    if dimension <= 0:
        raise ValueError(f"{field} must be positive for {label}, got {dimension}.")
    return dimension


def resolve_descriptor_contract(
    template: Mapping[str, Any],
    descriptor_source: str,
) -> DescriptorContract:
    """从模板元数据和数组形状解析描述子契约。

    HardNet 旧模板没有类型字段时按历史浮点 L2 模板解释，但维度必须能从
    `hardnet_descriptor_dim` 或实际 `[N,D]` 数组得到，空数组也不得猜测成 128。
    """

    source = str(descriptor_source).strip().lower()
    label = _template_label(template)
    if source == "hardnet":
        raw = template.get("hardnet_descriptors")
        descriptors = np.asarray(raw if raw is not None else [])
        metadata_dimension = template.get("hardnet_descriptor_dim")
        if metadata_dimension is None:
            metadata_dimension = template.get("descriptor_dim")
        if metadata_dimension is not None:
            dimension = _positive_dimension(
                metadata_dimension,
                field="hardnet_descriptor_dim",
                label=label,
            )
        elif descriptors.ndim == 2 and descriptors.shape[1] > 0:
            dimension = int(descriptors.shape[1])
        else:
            raise ValueError(
                "Cannot determine HardNet descriptor dimension from template metadata or "
                f"array shape: {label}, descriptors={descriptors.shape}."
            )
        kind = _normalize_float_kind(
            template.get("hardnet_descriptor_kind", template.get("descriptor_kind"))
        )
        metric = _normalize_metric(
            template.get("hardnet_descriptor_metric", template.get("descriptor_metric"))
        )
        storage = str(
            template.get(
                "hardnet_descriptor_storage",
                template.get("descriptor_storage", FLOAT32_STORAGE),
            )
        ).strip().lower()
    elif source in {"sift", "rootsift"}:
        source = "sift"
        dimension = 128
        kind = FLOAT_DESCRIPTOR_KIND
        metric = L2_DISTANCE_METRIC
        storage = FLOAT32_STORAGE
    else:
        raise ValueError(f"unsupported descriptor_source: {descriptor_source}")

    return DescriptorContract(
        source=source,
        kind=kind,
        metric=metric,
        dimension=dimension,
        storage=storage,
    )


def require_l2_float_contract(contract: DescriptorContract, *, label: str) -> None:
    """拒绝把二值或非 L2 描述子交给连续浮点 matcher。"""

    if contract.kind != FLOAT_DESCRIPTOR_KIND or contract.metric != L2_DISTANCE_METRIC:
        raise ValueError(
            "L2 matcher only accepts continuous float descriptors: "
            f"{label}: kind={contract.kind}, metric={contract.metric}, "
            f"dimension={contract.dimension}."
        )


def require_compatible_contracts(
    query: DescriptorContract,
    gallery: DescriptorContract,
) -> None:
    """确保 query/gallery 使用完全一致的描述子解释，不做隐式维度转换。"""

    left = (query.source, query.kind, query.metric, query.dimension)
    right = (gallery.source, gallery.kind, gallery.metric, gallery.dimension)
    if left != right:
        raise ValueError(
            "Query/gallery descriptor contract mismatch: "
            f"query={left}, gallery={right}. Rebuild templates with the same checkpoint; "
            "descriptor truncation, padding and implicit conversion are not supported."
        )