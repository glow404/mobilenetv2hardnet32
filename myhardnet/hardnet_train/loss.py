"""HardNet 的 top-k hardest-in-batch triplet loss。

作用：
    给定一个 batch 的正样本 patch 对 `(A_i, P_i)`，网络分别输出 anchor
    描述子 `a_i` 和 positive 描述子 `p_i`。本文件计算 batch 内所有
    `a_i` 与 `p_j` 的距离矩阵，并为每个正样本寻找 top-k 难负样本。

为什么需要 point_group：
    指纹数据中同一个真实物理点可能出现在多张图、多条 CSV 正样本记录里。
    如果不屏蔽同一物理点，hardest negative 很容易选到“其实也是正样本”的
    伪负样本，导致网络一边被要求拉近、一边又被要求推远。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hardnet_train.negative_sampling import (
    DEFAULT_SAME_FINGER_MIN_COORDINATE_SEPARATION_PX,
    normalize_min_coordinate_separation,
    same_finger_nearby_mask,
)


HARD_NEGATIVE_STRATEGY_ALIASES = {
    "same_finger_allowed": "same_finger_allowed",
    "point_group": "same_finger_allowed",
    "connected_point": "same_finger_allowed",
    "different_finger": "different_finger",
    "cross_finger_only": "different_finger",
}


def normalize_hard_negative_strategy(strategy: str | None) -> str:
    """规范化 hardest negative 候选策略名称。"""

    if strategy is None:
        return "same_finger_allowed"
    key = str(strategy).strip().lower()
    if not key:
        return "same_finger_allowed"
    if key not in HARD_NEGATIVE_STRATEGY_ALIASES:
        supported = ", ".join(sorted(HARD_NEGATIVE_STRATEGY_ALIASES))
        raise ValueError(f"Unsupported hard negative strategy: {strategy!r}. Supported: {supported}")
    return HARD_NEGATIVE_STRATEGY_ALIASES[key]


def pairwise_l2_for_unit_vectors(anchor: torch.Tensor, positive: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """计算两组单位向量描述子之间的两两 L2 距离。

    对单位向量 x、y，有：
        ||x - y||_2 = sqrt(2 - 2 * dot(x, y))

    这样比显式广播相减更省显存，尤其 batch_size 较大时很重要。
    """
    similarity = anchor @ positive.t()
    distance_sq = torch.clamp(2.0 - 2.0 * similarity, min=eps)
    return torch.sqrt(distance_sq)


@dataclass(frozen=True)
class HardNegativeSelection:
    """一次 batch 内 top-k 负样本选择的完整结果。"""

    positive_dist: torch.Tensor
    topk_negative_dist: torch.Tensor
    valid_topk: torch.Tensor
    negative_count: torch.Tensor
    valid_anchor: torch.Tensor
    mean_negative_dist: torch.Tensor
    topk_same_finger: torch.Tensor
    topk_candidate_index: torch.Tensor


def select_hard_negatives(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    *,
    hard_negative_strategy: str,
    hard_negative_top_k: int,
    same_finger_min_coordinate_separation_px: float,
    point_group: torch.Tensor | None = None,
    finger_group: torch.Tensor | None = None,
    anchor_xy: torch.Tensor | None = None,
    positive_xy: torch.Tensor | None = None,
    anchor_coordinate_frame_group: torch.Tensor | None = None,
    positive_coordinate_frame_group: torch.Tensor | None = None,
) -> HardNegativeSelection:
    """按训练契约构建合法候选掩码并选择双向 top-k 难负样本。"""

    if anchor.shape != positive.shape:
        raise ValueError(
            f"anchor/positive shape mismatch: {anchor.shape} vs {positive.shape}"
        )
    if anchor.ndim != 2:
        raise ValueError(
            "Hard negative selection expects [batch, descriptor_dim] tensors."
        )
    strategy = normalize_hard_negative_strategy(hard_negative_strategy)
    top_k = int(hard_negative_top_k)
    if top_k < 1:
        raise ValueError(
            f"hard_negative_top_k must be >= 1, got {hard_negative_top_k!r}."
        )
    min_separation = normalize_min_coordinate_separation(
        same_finger_min_coordinate_separation_px
    )

    batch_size = int(anchor.size(0))
    distances = pairwise_l2_for_unit_vectors(anchor, positive)
    positive_dist = distances.diag()
    invalid = torch.eye(
        batch_size,
        dtype=torch.bool,
        device=distances.device,
    )

    finger: torch.Tensor | None = None
    same_finger = torch.zeros_like(invalid)
    if finger_group is not None:
        finger = finger_group.to(device=distances.device).view(-1)
        if finger.numel() != batch_size:
            raise ValueError("finger_group length must match batch size.")
        same_finger = finger[:, None].eq(finger[None, :])
    if strategy == "different_finger":
        if finger is None:
            raise ValueError(
                "finger_group is required when "
                "hard_negative_strategy='different_finger'."
            )
        invalid = invalid | same_finger

    if point_group is not None:
        point = point_group.to(device=distances.device).view(-1)
        if point.numel() != batch_size:
            raise ValueError("point_group length must match batch size.")
        invalid = invalid | point[:, None].eq(point[None, :])

    anchor_invalid = invalid
    positive_invalid = invalid.t()
    if strategy == "same_finger_allowed" and min_separation > 0.0:
        spatial_inputs = {
            "finger_group": finger_group,
            "anchor_xy": anchor_xy,
            "positive_xy": positive_xy,
            "anchor_coordinate_frame_group": anchor_coordinate_frame_group,
            "positive_coordinate_frame_group": positive_coordinate_frame_group,
        }
        missing = [name for name, value in spatial_inputs.items() if value is None]
        if missing:
            raise ValueError(
                "Same-finger coordinate filtering requires batch metadata: "
                f"{missing}."
            )
        assert finger_group is not None
        assert anchor_xy is not None
        assert positive_xy is not None
        assert anchor_coordinate_frame_group is not None
        assert positive_coordinate_frame_group is not None
        anchor_invalid = anchor_invalid | same_finger_nearby_mask(
            point_xy=anchor_xy.to(device=distances.device),
            finger_group=finger_group,
            coordinate_frame_group=anchor_coordinate_frame_group,
            min_coordinate_separation_px=min_separation,
        )
        positive_invalid = positive_invalid | same_finger_nearby_mask(
            point_xy=positive_xy.to(device=distances.device),
            finger_group=finger_group,
            coordinate_frame_group=positive_coordinate_frame_group,
            min_coordinate_separation_px=min_separation,
        )

    large_value = torch.finfo(distances.dtype).max / 16.0
    negative_candidates = torch.cat(
        [
            distances.masked_fill(anchor_invalid, large_value),
            distances.t().masked_fill(positive_invalid, large_value),
        ],
        dim=1,
    )
    candidate_same_finger = torch.cat(
        [same_finger, same_finger.t()],
        dim=1,
    )
    selected_k = min(top_k, int(negative_candidates.size(1)))
    topk_negative_dist, topk_indices = torch.topk(
        negative_candidates,
        k=selected_k,
        dim=1,
        largest=False,
        sorted=True,
    )
    topk_same_finger = torch.gather(candidate_same_finger, 1, topk_indices)

    valid_topk = topk_negative_dist < large_value / 2.0
    negative_count = valid_topk.sum(dim=1)
    valid_anchor = negative_count > 0
    negative_sum = torch.where(
        valid_topk,
        topk_negative_dist,
        torch.zeros_like(topk_negative_dist),
    ).sum(dim=1)
    mean_negative_dist = negative_sum / negative_count.clamp_min(1)
    mean_negative_dist = torch.where(
        valid_anchor,
        mean_negative_dist,
        torch.full_like(mean_negative_dist, large_value),
    )
    return HardNegativeSelection(
        positive_dist=positive_dist,
        topk_negative_dist=topk_negative_dist,
        valid_topk=valid_topk,
        negative_count=negative_count,
        valid_anchor=valid_anchor,
        mean_negative_dist=mean_negative_dist,
        topk_same_finger=topk_same_finger,
        topk_candidate_index=topk_indices,
    )


class HardNetLoss(nn.Module):
    """HardNet 的 top-k hardest-in-batch triplet margin loss。"""

    def __init__(
        self,
        margin: float = 1.0,
        hard_negative_strategy: str = "same_finger_allowed",
        hard_negative_top_k: int = 3,
        same_finger_min_coordinate_separation_px: float = (
            DEFAULT_SAME_FINGER_MIN_COORDINATE_SEPARATION_PX
        ),
    ) -> None:
        super().__init__()
        self.margin = float(margin)
        self.hard_negative_strategy = normalize_hard_negative_strategy(hard_negative_strategy)
        self.hard_negative_top_k = int(hard_negative_top_k)
        self.same_finger_min_coordinate_separation_px = (
            normalize_min_coordinate_separation(
                same_finger_min_coordinate_separation_px
            )
        )
        if self.hard_negative_top_k < 1:
            raise ValueError(f"hard_negative_top_k must be >= 1, got {hard_negative_top_k!r}.")

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        point_group: torch.Tensor | None = None,
        finger_group: torch.Tensor | None = None,
        anchor_xy: torch.Tensor | None = None,
        positive_xy: torch.Tensor | None = None,
        anchor_coordinate_frame_group: torch.Tensor | None = None,
        positive_coordinate_frame_group: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """计算一个 batch 的 HardNet loss。

        参数：
            anchor:
                anchor 分支输出的描述子，形状 `[B, D]`。
            positive:
                positive 分支输出的描述子，形状 `[B, D]`，且 D 必须与 anchor 一致。
            point_group:
                每条正样本对应的物理点组编号。同组样本不能互相当负样本。
            finger_group:
                每条正样本对应的手指编号。`different_finger` 策略下，同一手指
                的所有候选都会被屏蔽。
            anchor_xy / positive_xy:
                两个分支关键点在各自原图坐标系中的 `[B, 2]` 坐标。
            anchor_coordinate_frame_group / positive_coordinate_frame_group:
                两个分支各自的原图编号。仅同指且同一原图坐标系的候选执行
                16px 方形邻域过滤，跨图不直接比较原始坐标。

        返回：
            loss:
                标量损失。
            stats:
                训练日志使用的正样本距离、top-k 负样本平均距离、有效样本 mask
                以及每条样本实际选中的负样本数量。
        """
        selection = select_hard_negatives(
            anchor,
            positive,
            hard_negative_strategy=self.hard_negative_strategy,
            hard_negative_top_k=self.hard_negative_top_k,
            same_finger_min_coordinate_separation_px=(
                self.same_finger_min_coordinate_separation_px
            ),
            point_group=point_group,
            finger_group=finger_group,
            anchor_xy=anchor_xy,
            positive_xy=positive_xy,
            anchor_coordinate_frame_group=anchor_coordinate_frame_group,
            positive_coordinate_frame_group=positive_coordinate_frame_group,
        )

        if not torch.any(selection.valid_anchor):
            zero = selection.positive_dist.sum() * 0.0
            stats = {
                "pos_dist": selection.positive_dist.detach(),
                "neg_dist": selection.mean_negative_dist.detach(),
                "valid_triplets": selection.valid_anchor.detach(),
                "hard_negative_count": selection.negative_count.detach(),
                "selected_negative_dist": selection.topk_negative_dist.detach(),
                "selected_negative_mask": selection.valid_topk.detach(),
                "selected_negative_is_same_finger": (
                    selection.topk_same_finger.detach()
                ),
                "selected_negative_candidate_index": (
                    selection.topk_candidate_index.detach()
                ),
            }
            return zero, stats

        # 每个 top-k 负样本各自计算 triplet margin loss；先在单条正样本内部
        # 对有效负样本求均值，再在有效正样本间求均值，避免候选较少的样本权重变低。
        per_negative = torch.clamp(
            self.margin
            + selection.positive_dist[:, None]
            - selection.topk_negative_dist,
            min=0.0,
        )
        per_sample = torch.where(
            selection.valid_topk,
            per_negative,
            torch.zeros_like(per_negative),
        ).sum(dim=1)
        per_sample = per_sample / selection.negative_count.clamp_min(1)
        loss = per_sample[selection.valid_anchor].mean()
        stats = {
            "pos_dist": selection.positive_dist.detach(),
            "neg_dist": selection.mean_negative_dist.detach(),
            "valid_triplets": selection.valid_anchor.detach(),
            "hard_negative_count": selection.negative_count.detach(),
            "selected_negative_dist": selection.topk_negative_dist.detach(),
            "selected_negative_mask": selection.valid_topk.detach(),
            "selected_negative_is_same_finger": (
                selection.topk_same_finger.detach()
            ),
            "selected_negative_candidate_index": (
                selection.topk_candidate_index.detach()
            ),
        }
        return loss, stats
