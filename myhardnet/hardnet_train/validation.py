"""确定性的 patch 描述子验证协议。

验证协议与训练 batch sampler 解耦：固定抽取正样本行，并为每个 anchor 固定选择
跨指负样本；同指负样本必须来自 anchor 原图中的其他物理点组，且位于以 anchor
为中心的配置方形邻域之外。协议只存在于当前训练进程中。
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hardnet_train.data import FingerprintPairDataset, PairRecord
from hardnet_train.metrics import descriptor_validation_metrics, pair_l2_distance
from hardnet_train.negative_sampling import (
    DEFAULT_SAME_FINGER_MIN_COORDINATE_SEPARATION_PX,
    is_outside_square_neighborhood,
    normalize_min_coordinate_separation,
)


ANCHOR_SIDE = 0
POSITIVE_SIDE = 1


def _encoded_reference(record_index: int, side: int) -> int:
    return int(record_index) * 2 + int(side)


def _reference_xy(records: list[PairRecord], encoded_reference: int) -> tuple[float, float]:
    record_index, side = divmod(int(encoded_reference), 2)
    record = records[record_index]
    if side == ANCHOR_SIDE:
        return record.anchor_x, record.anchor_y
    return record.positive_x, record.positive_y


def _stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class FixedValidationProtocol:
    """固定验证索引；负样本引用编码为 `record_index * 2 + side`。"""

    positive_indices: np.ndarray
    same_finger_negative_refs: np.ndarray
    cross_finger_negative_refs: np.ndarray
    same_finger_min_coordinate_separation_px: float

    @property
    def positive_count(self) -> int:
        return int(self.positive_indices.shape[0])

    @property
    def same_finger_negatives_per_anchor(self) -> int:
        return int(self.same_finger_negative_refs.shape[1])

    @property
    def cross_finger_negatives_per_anchor(self) -> int:
        return int(self.cross_finger_negative_refs.shape[1])

    def positive_anchor_refs(self) -> np.ndarray:
        return self.positive_indices.astype(np.int64, copy=False) * 2 + ANCHOR_SIDE

    def positive_match_refs(self) -> np.ndarray:
        return self.positive_indices.astype(np.int64, copy=False) * 2 + POSITIVE_SIDE

    def unique_patch_refs(self) -> np.ndarray:
        return np.unique(
            np.concatenate(
                [
                    self.positive_anchor_refs(),
                    self.positive_match_refs(),
                    self.same_finger_negative_refs.reshape(-1),
                    self.cross_finger_negative_refs.reshape(-1),
                ]
            )
        )

def build_fixed_validation_protocol(
    records: list[PairRecord],
    positive_count: int,
    same_finger_negatives_per_anchor: int,
    cross_finger_negatives_per_anchor: int,
    seed: int,
    same_finger_min_coordinate_separation_px: float = (
        DEFAULT_SAME_FINGER_MIN_COORDINATE_SEPARATION_PX
    ),
) -> FixedValidationProtocol:
    """构建确定性空间过滤协议，不使用重复采样补齐。"""
    if not records:
        raise ValueError("Fixed validation protocol requires at least one record.")
    if not 1 <= positive_count <= len(records):
        raise ValueError(
            f"validation positive_count must be in [1, {len(records)}], got {positive_count}."
        )
    if same_finger_negatives_per_anchor < 1 or cross_finger_negatives_per_anchor < 1:
        raise ValueError("Both validation negative counts must be >= 1.")
    min_coordinate_separation_px = normalize_min_coordinate_separation(
        same_finger_min_coordinate_separation_px
    )

    refs_by_point: dict[tuple[str, int], list[int]] = defaultdict(list)
    refs_by_frame: dict[
        tuple[str, str], dict[tuple[str, int], list[int]]
    ] = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        key = (record.finger_id, record.point_group)
        anchor_ref = _encoded_reference(index, ANCHOR_SIDE)
        positive_ref = _encoded_reference(index, POSITIVE_SIDE)
        refs_by_point[key].append(anchor_ref)
        refs_by_point[key].append(positive_ref)
        refs_by_frame[(record.finger_id, record.image_a_id)][key].append(
            anchor_ref
        )
        refs_by_frame[(record.finger_id, record.image_b_id)][key].append(
            positive_ref
        )

    finger_ids = tuple(sorted({record.finger_id for record in records}))
    if len(finger_ids) < 2:
        raise ValueError("Cross-finger validation requires at least two distinct finger_id values.")

    cross_point_keys: dict[str, list[tuple[str, int]]] = {
        finger_id: sorted(key for key in refs_by_point if key[0] != finger_id)
        for finger_id in finger_ids
    }
    representative_ref = {
        key: refs[_stable_seed(seed, "representative", key[0], key[1]) % len(refs)]
        for key, refs in refs_by_point.items()
    }

    unique_positive_indices: list[int] = []
    seen_positive_pairs: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        # L2 距离对 A/P 方向对称；路径排序后也能排除方向相反的重复 pair。
        positive_key = tuple(sorted((record.patch_a_path, record.patch_p_path)))
        if positive_key in seen_positive_pairs:
            continue
        seen_positive_pairs.add(positive_key)
        unique_positive_indices.append(index)
    if positive_count > len(unique_positive_indices):
        raise ValueError(
            f"validation positive_count={positive_count} exceeds {len(unique_positive_indices)} unique pairs."
        )
    selected_positive_indices = np.asarray(
        random.Random(_stable_seed(seed, "positives")).sample(unique_positive_indices, positive_count),
        dtype=np.int64,
    )
    same_refs = np.empty((positive_count, same_finger_negatives_per_anchor), dtype=np.int64)
    cross_refs = np.empty((positive_count, cross_finger_negatives_per_anchor), dtype=np.int64)

    for row, record_index in enumerate(selected_positive_indices):
        record = records[int(record_index)]
        anchor_key = (record.finger_id, record.point_group)
        anchor_xy = (record.anchor_x, record.anchor_y)
        frame_points = refs_by_frame.get(
            (record.finger_id, record.image_a_id), {}
        )
        same_candidates: list[tuple[tuple[str, int], int]] = []
        for key, refs in sorted(frame_points.items()):
            if key == anchor_key:
                continue
            representative_index = _stable_seed(
                seed,
                "same-frame-representative",
                record.finger_id,
                record.image_a_id,
                key[1],
            ) % len(refs)
            candidate_ref = refs[representative_index]
            if is_outside_square_neighborhood(
                anchor_xy=anchor_xy,
                candidate_xy=_reference_xy(records, candidate_ref),
                min_coordinate_separation_px=min_coordinate_separation_px,
            ):
                same_candidates.append((key, candidate_ref))
        cross_candidates = cross_point_keys[record.finger_id]
        if len(same_candidates) < same_finger_negatives_per_anchor:
            raise ValueError(
                f"finger_id={record.finger_id!r}, image_id={record.image_a_id!r}, "
                f"anchor=({record.anchor_x:.3f}, {record.anchor_y:.3f}) has only "
                f"{len(same_candidates)} same-finger point groups outside the "
                f"{min_coordinate_separation_px:g}px square neighborhood; need "
                f"{same_finger_negatives_per_anchor}."
            )
        if len(cross_candidates) < cross_finger_negatives_per_anchor:
            raise ValueError(
                f"finger_id={record.finger_id!r} has only {len(cross_candidates)} cross-finger "
                f"point groups; need {cross_finger_negatives_per_anchor}."
            )

        rng = random.Random(_stable_seed(seed, "anchor", record.pair_id, int(record_index)))
        selected_same = rng.sample(
            same_candidates, same_finger_negatives_per_anchor
        )
        selected_cross = rng.sample(cross_candidates, cross_finger_negatives_per_anchor)
        same_refs[row] = [candidate_ref for _, candidate_ref in selected_same]
        cross_refs[row] = [representative_ref[key] for key in selected_cross]

    return FixedValidationProtocol(
        positive_indices=selected_positive_indices,
        same_finger_negative_refs=same_refs,
        cross_finger_negative_refs=cross_refs,
        same_finger_min_coordinate_separation_px=min_coordinate_separation_px,
    )


class _ValidationPatchDataset(Dataset):
    def __init__(self, dataset: FingerprintPairDataset, encoded_refs: np.ndarray) -> None:
        self.dataset = dataset
        self.encoded_refs = encoded_refs

    def __len__(self) -> int:
        return int(self.encoded_refs.size)

    def __getitem__(self, index: int) -> torch.Tensor:
        encoded = int(self.encoded_refs[index])
        record_index, side = divmod(encoded, 2)
        side_name = "anchor" if side == ANCHOR_SIDE else "positive"
        return self.dataset.load_patch(record_index, side_name)


def make_validation_patch_loader(
    dataset: FingerprintPairDataset,
    protocol: FixedValidationProtocol,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
) -> tuple[DataLoader, np.ndarray]:
    """按排序后的唯一引用构造顺序固定的 patch 编码 DataLoader。"""
    unique_refs = protocol.unique_patch_refs()
    options: dict[str, Any] = {
        "dataset": _ValidationPatchDataset(dataset, unique_refs),
        "batch_size": int(batch_size),
        "shuffle": False,
        "drop_last": False,
        "num_workers": max(0, int(num_workers)),
        "pin_memory": bool(pin_memory) and torch.cuda.is_available(),
    }
    if options["num_workers"] > 0:
        options["persistent_workers"] = bool(persistent_workers)
        options["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(**options), unique_refs


def _descriptor_rows(unique_refs: np.ndarray, refs: np.ndarray) -> torch.Tensor:
    rows = np.searchsorted(unique_refs, refs)
    if np.any(rows >= unique_refs.size) or not np.array_equal(unique_refs[rows], refs):
        raise RuntimeError("Validation protocol contains an unresolved patch reference.")
    return torch.from_numpy(rows.astype(np.int64, copy=False))


@torch.inference_mode()
def evaluate_fixed_protocol(
    model: torch.nn.Module,
    loader: DataLoader,
    unique_refs: np.ndarray,
    protocol: FixedValidationProtocol,
    device: torch.device,
    margin: float,
    hard_negative_top_k: int,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    channels_last: bool,
) -> dict[str, float]:
    """只编码唯一 patch，再按固定索引计算正样本及两类负样本距离。"""
    model.eval()
    descriptor_batches: list[torch.Tensor] = []
    for patches in loader:
        patches = patches.to(device, non_blocking=True)
        if channels_last:
            patches = patches.contiguous(memory_format=torch.channels_last)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            descriptors = model(patches)
        descriptor_batches.append(descriptors.float().cpu())
    if not descriptor_batches:
        raise RuntimeError("Fixed validation patch loader produced no descriptors.")
    descriptors = torch.cat(descriptor_batches, dim=0)

    anchor_rows = _descriptor_rows(unique_refs, protocol.positive_anchor_refs())
    positive_rows = _descriptor_rows(unique_refs, protocol.positive_match_refs())
    same_rows = _descriptor_rows(unique_refs, protocol.same_finger_negative_refs)
    cross_rows = _descriptor_rows(unique_refs, protocol.cross_finger_negative_refs)

    anchor_desc = descriptors[anchor_rows]
    positive_desc = descriptors[positive_rows]
    positive_dist = pair_l2_distance(anchor_desc, positive_desc)
    same_dist = pair_l2_distance(anchor_desc[:, None, :], descriptors[same_rows])
    cross_dist = pair_l2_distance(anchor_desc[:, None, :], descriptors[cross_rows])

    metrics = descriptor_validation_metrics(
        positive_dist=positive_dist,
        same_finger_negative_dist=same_dist,
        cross_finger_negative_dist=cross_dist,
        margin=float(margin),
        hard_negative_top_k=int(hard_negative_top_k),
    )
    return metrics