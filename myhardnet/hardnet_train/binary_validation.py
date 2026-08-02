"""固定协议下的二值描述子 Hamming 验证。"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from hardnet_train.binary_model import BinaryDescriptorModel
from hardnet_train.metrics import descriptor_validation_metrics
from hardnet_train.validation import FixedValidationProtocol


def pair_normalized_hamming(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """计算末维 0/1 码的一一对应归一化 Hamming 距离。"""

    if left.shape[-1] != right.shape[-1]:
        raise ValueError(f"Binary descriptor dimension mismatch: {left.shape} vs {right.shape}")
    return left.ne(right).float().mean(dim=-1)


def _binary_rows(unique_refs: np.ndarray, refs: np.ndarray) -> torch.Tensor:
    rows = np.searchsorted(unique_refs, refs)
    if np.any(rows >= unique_refs.size) or not np.array_equal(unique_refs[rows], refs):
        raise RuntimeError("Binary validation protocol contains an unresolved patch reference.")
    return torch.from_numpy(rows.astype(np.int64, copy=False))


@torch.inference_mode()
def evaluate_binary_fixed_protocol(
    model: BinaryDescriptorModel,
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
    """编码唯一 patch，并以真实 0/1 code 计算固定 Hamming 验证指标。"""

    model.eval()
    code_batches: list[torch.Tensor] = []
    for patches in loader:
        patches = patches.to(device, non_blocking=True)
        if channels_last:
            patches = patches.contiguous(memory_format=torch.channels_last)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            codes = model.encode_binary(patches, packed=False)
        code_batches.append(codes.bool().cpu())
    if not code_batches:
        raise RuntimeError("Fixed validation patch loader produced no binary descriptors.")
    codes = torch.cat(code_batches, dim=0)

    anchor_rows = _binary_rows(unique_refs, protocol.positive_anchor_refs())
    positive_rows = _binary_rows(unique_refs, protocol.positive_match_refs())
    same_rows = _binary_rows(unique_refs, protocol.same_finger_negative_refs)
    cross_rows = _binary_rows(unique_refs, protocol.cross_finger_negative_refs)

    anchor_codes = codes[anchor_rows]
    positive_codes = codes[positive_rows]
    positive_dist = pair_normalized_hamming(anchor_codes, positive_codes)
    same_dist = pair_normalized_hamming(
        anchor_codes[:, None, :],
        codes[same_rows],
    )
    cross_dist = pair_normalized_hamming(
        anchor_codes[:, None, :],
        codes[cross_rows],
    )
    metrics = descriptor_validation_metrics(
        positive_dist=positive_dist,
        same_finger_negative_dist=same_dist,
        cross_finger_negative_dist=cross_dist,
        margin=float(margin),
        hard_negative_top_k=int(hard_negative_top_k),
    )
    bit_one_fraction = codes.float().mean(dim=0)
    metrics.update(
        {
            "bit_one_fraction": float(bit_one_fraction.mean().item()),
            "bit_balance_error": float(
                (bit_one_fraction - 0.5).abs().mean().item()
            ),
            "constant_bit_ratio": float(
                ((bit_one_fraction == 0.0) | (bit_one_fraction == 1.0))
                .float()
                .mean()
                .item()
            ),
        }
    )
    return metrics