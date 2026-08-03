"""数据构建流水线支持的唯一数据划分契约。"""

from __future__ import annotations


DATASET_SPLITS: tuple[str, ...] = ("train", "val")


def require_supported_split(value: str) -> str:
    """校验并返回标准 split 名称。"""

    split = str(value).strip()
    if split not in DATASET_SPLITS:
        supported = ", ".join(DATASET_SPLITS)
        raise ValueError(
            f"不支持的数据 split: {value!r}；当前只允许: {supported}。请重新执行 split 阶段。"
        )
    return split