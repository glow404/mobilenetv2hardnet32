"""从 m0 或 P7-L 浮点权重初始化的直接 256-bit 描述子。

将最后的卷积、固定 BN 统计和 Sylvester Hadamard 变换融合为一个卷积头。
本模块只保存 student；固定教师由训练器管理，部署只需输入 patch 并输出 logits。
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from hardnet_train.binary_model import (
    BINARY_DESCRIPTOR_KIND,
    BINARY_ENCODING,
    BINARY_STORAGE,
    HAMMING_DISTANCE_METRIC,
    BinaryDescriptorFeatures,
    pack_binary_codes,
)
from hardnet_train.model import HardNet, PlainSpatialL, normalize_model_architecture


DIRECT_BINARY_ARCHITECTURE = "hardnet_direct_binary_v1"
DIRECT_BINARY_MODEL_ARCHITECTURE = DIRECT_BINARY_ARCHITECTURE
DIRECT_HEAD_ARCHITECTURE = "direct_conv8x8_v1"
_DIRECT_BACKBONES = {"hardnet": HardNet, "plain_spatial_l": PlainSpatialL}


def _direct_channels(architecture: str) -> tuple[int, ...]:
    architecture = normalize_model_architecture(architecture)
    if architecture not in _DIRECT_BACKBONES:
        raise ValueError("Direct binary heads support only hardnet (m0) or plain_spatial_l (P7-L).")
    return _DIRECT_BACKBONES[architecture].channels


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value}.")
    return value


def _normalized_hadamard_256(reference: torch.Tensor) -> torch.Tensor:
    """构造与现有 FWHT 顺序相同的 Sylvester H256 / 16，仅用于初始化。"""

    matrix = reference.new_ones((1, 1), dtype=torch.float64)
    while matrix.shape[0] < 256:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)),
            dim=0,
        )
    return matrix / 16.0


class DirectBinaryDescriptorModel(nn.Module):
    """普通CNN主干与直接卷积头；二值训练和部署使用相同 logits。"""

    architecture = DIRECT_BINARY_ARCHITECTURE
    backbone_architecture = "hardnet"
    float_descriptor_dim = 256
    hash_bits = 256
    descriptor_dim = 256

    def __init__(
        self,
        *,
        backbone_architecture: str = "hardnet",
        dropout: float = 0.1,
        temperature: float = 1.0,
        backbone_trainable: bool = True,
        freeze_backbone_bn: bool = True,
        quantization_fp32: bool = False,
        bitorder: str = "little",
    ) -> None:
        super().__init__()
        dropout = float(dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        bitorder = str(bitorder).strip().lower()
        if bitorder not in {"little", "big"}:
            raise ValueError(f"bitorder must be 'little' or 'big', got {bitorder!r}.")
        self.bitorder = bitorder
        self.freeze_backbone_bn = bool(freeze_backbone_bn)
        self.quantization_fp32 = bool(quantization_fp32)
        self.backbone_architecture = normalize_model_architecture(backbone_architecture)
        channels = _direct_channels(self.backbone_architecture)
        widths = (1, *channels)
        self.backbone = nn.Sequential(
            *(HardNet._conv_block(widths[i], widths[i + 1], stride=stride)
              for i, stride in enumerate((1, 1, 2, 1, 2, 1))),
            nn.Dropout(p=dropout),
        )
        # bias 保存被折叠的最终 BN 平移量；头后不再添加 BN 或 L2。
        self.head = nn.Conv2d(channels[-1], 256, kernel_size=8, bias=True)
        self.register_buffer(
            "_quantization_temperature",
            torch.tensor(_positive_finite(temperature, "temperature"), dtype=torch.float32),
        )
        self.backbone_trainable = bool(backbone_trainable)
        self.set_backbone_trainable(backbone_trainable)

    @classmethod
    def from_float_backbone(
        cls,
        backbone: nn.Module,
        config: Mapping[str, Any] | None = None,
    ) -> DirectBinaryDescriptorModel:
        """复制同架构浮点主干，并用其最终 BN running statistics 融合头。"""

        options = config or {}
        if not isinstance(backbone, HardNet) or backbone.descriptor_dim != 256:
            raise ValueError("Direct binary training requires a HardNet (m0) or P7-L float backbone with descriptor_dim=256.")
        channels = _direct_channels(backbone.architecture)
        if int(options.get("hash_bits", 256)) != 256:
            raise ValueError("Direct binary training supports exactly 256 bits; binary_model.hash_bits must be 256.")
        if len(backbone.features) != 9:
            raise ValueError("Direct binary training requires the six-block plain CNN feature layout.")
        output_conv, output_bn = backbone.features[-2:]
        if (
            not isinstance(output_conv, nn.Conv2d)
            or output_conv.weight.shape != (256, channels[-1], 8, 8)
            or output_conv.groups != 1
            or not isinstance(output_bn, nn.BatchNorm2d)
            or output_bn.num_features != 256
            or output_bn.running_mean is None
            or output_bn.running_var is None
        ):
            raise ValueError("Direct binary training requires a matching Conv8×8 and final BN with running statistics.")
        source_dropout = backbone.features[6]
        if not isinstance(source_dropout, nn.Dropout):
            raise ValueError("Direct binary training requires dropout immediately before the final convolution.")
        model = cls(
            backbone_architecture=backbone.architecture,
            dropout=float(source_dropout.p),
            temperature=float(options.get("temperature_start", 1.0)),
            backbone_trainable=bool(options.get("backbone_trainable", True)),
            freeze_backbone_bn=bool(options.get("freeze_backbone_bn", True)),
            quantization_fp32=bool(options.get("quantization_fp32", False)),
            bitorder=str(options.get("bitorder", "little")),
        ).to(device=output_conv.weight.device, dtype=output_conv.weight.dtype)
        # deepcopy 避免 student 更新污染初始化来源，也允许训练器保留独立固定教师。
        model.backbone = copy.deepcopy(backbone.features[:7])
        with torch.no_grad():
            weight = output_conv.weight.detach().double().flatten(start_dim=1)
            mean = output_bn.running_mean.detach().double()
            variance = output_bn.running_var.detach().double()
            gamma = output_bn.weight.detach().double() if output_bn.affine else torch.ones_like(mean)
            beta = output_bn.bias.detach().double() if output_bn.affine else torch.zeros_like(mean)
            bias = output_conv.bias.detach().double() if output_conv.bias is not None else torch.zeros_like(mean)
            scale = gamma / torch.sqrt(variance + output_bn.eps)
            if not torch.isfinite(scale).all():
                raise ValueError("The source final BN has invalid running variance or affine weights.")
            transform = _normalized_hadamard_256(weight)
            # q=A·W·u+c；sign(H·normalize(q))=sign(H·q)，因此可以去掉最终 L2。
            fused_weight = transform @ (scale[:, None] * weight)
            fused_bias = transform @ (beta + scale * (bias - mean))
            if not torch.isfinite(fused_weight).all() or not torch.isfinite(fused_bias).all():
                raise ValueError("The source float checkpoint produces non-finite fused head weights.")
            model.head.weight.copy_(fused_weight.reshape_as(model.head.weight))
            model.head.bias.copy_(fused_bias)
        model.set_backbone_trainable(model.backbone_trainable)
        return model

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, Any]) -> DirectBinaryDescriptorModel:
        """从自包含checkpoint加载student，按元数据恢复宽度，不读取浮点源文件。"""

        architecture = str(checkpoint.get("backbone_architecture", ""))
        channels = _direct_channels(architecture)
        expected = {
            "model_architecture": DIRECT_BINARY_ARCHITECTURE,
            "descriptor_kind": BINARY_DESCRIPTOR_KIND,
            "descriptor_metric": HAMMING_DISTANCE_METRIC,
            "descriptor_dim": 256,
            "hash_bits": 256,
            "hash_packed_bytes": 32,
            "binary_encoding": BINARY_ENCODING,
            "binary_storage": BINARY_STORAGE,
            "backbone_architecture": normalize_model_architecture(architecture),
            "float_descriptor_dim": 256,
            "hash_head_architecture": DIRECT_HEAD_ARCHITECTURE,
            "hash_head_input_dim": channels[-1] * 8 * 8,
            "hash_head_hidden_dim": 0,
            "hash_head_dropout": 0.0,
        }
        for name, value in expected.items():
            if checkpoint.get(name) != value:
                raise ValueError(
                    f"Direct checkpoint metadata mismatch for {name}: "
                    f"checkpoint={checkpoint.get(name)!r}, expected={value!r}."
                )
        model = cls(
            backbone_architecture=architecture,
            dropout=float(checkpoint.get("backbone_dropout", 0.1)),
            temperature=float(checkpoint.get("quantization_temperature", 1.0)),
            backbone_trainable=bool(checkpoint.get("backbone_trainable", True)),
            freeze_backbone_bn=bool(checkpoint.get("freeze_backbone_bn", True)),
            quantization_fp32=bool(checkpoint.get("quantization_fp32", False)),
            bitorder=str(checkpoint.get("binary_bitorder", "little")),
        )
        state = checkpoint.get("model")
        if not isinstance(state, Mapping):
            raise ValueError("Direct checkpoint must contain a self-contained model state dictionary.")
        model.load_state_dict(state, strict=True)
        _positive_finite(model.quantization_temperature, "checkpoint temperature")
        return model

    @property
    def hash_head(self) -> nn.Conv2d:
        """兼容训练器的头参数统计；property 避免同一模块注册两遍。"""

        return self.head

    @property
    def quantization_temperature(self) -> float:
        return float(self._quantization_temperature.item())

    def set_quantization_temperature(self, value: float) -> None:
        self._quantization_temperature.fill_(_positive_finite(value, "temperature"))

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.backbone_trainable = bool(trainable)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(self.backbone_trainable)
        self.train(self.training)

    def train(self, mode: bool = True) -> DirectBinaryDescriptorModel:
        super().train(mode)
        if not self.backbone_trainable:
            self.backbone.eval()
        elif self.freeze_backbone_bn:
            # 冻结的是 running statistics；联合阶段 BN affine 参数仍可学习。
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward_logits(self, patches: torch.Tensor) -> torch.Tensor:
        if self.backbone_trainable:
            features = self.backbone(patches)
        else:
            with torch.no_grad():
                features = self.backbone(patches)
        return self.head(features).flatten(start_dim=1)

    def forward_features(self, patches: torch.Tensor) -> BinaryDescriptorFeatures:
        logits = self.forward_logits(patches)
        # FP16 tanh可能把接近1的数舍入成1，使导数提前归零。P7-L用FP32计算
        # 量化代理；这只改善数值梯度，不改变部署sign，也不能消除真实的tanh饱和。
        values = logits.float() if self.quantization_fp32 else logits
        temperature = self._quantization_temperature.to(values).clamp_min(1e-6)
        continuous = torch.tanh(values / temperature)
        hard = torch.where(values >= 0, torch.ones_like(values), -torch.ones_like(values))
        # 前向采用硬码，反向借用 tanh 梯度；固定教师由训练器显式替换占位字段。
        quantized = continuous + (hard - continuous).detach()
        detached = logits.detach()
        return BinaryDescriptorFeatures(
            teacher_q=detached,
            teacher_f=F.normalize(detached, p=2, dim=1),
            logits=logits,
            continuous=continuous,
            quantized=quantized,
            codes=logits >= 0,
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.forward_features(patches).quantized, p=2, dim=1)

    @torch.inference_mode()
    def encode_binary(self, patches: torch.Tensor, *, packed: bool = False) -> torch.Tensor:
        # 推理不计算教师、tanh 或 STE，符号判定和打包复用统一二值契约。
        codes = (self.forward_logits(patches) >= 0).to(torch.uint8)
        return pack_binary_codes(codes, self.bitorder) if packed else codes

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "model_architecture": self.architecture,
            "descriptor_kind": BINARY_DESCRIPTOR_KIND,
            "descriptor_metric": HAMMING_DISTANCE_METRIC,
            "descriptor_dim": self.hash_bits,
            "hash_bits": self.hash_bits,
            "hash_packed_bytes": 32,
            "hash_head_architecture": DIRECT_HEAD_ARCHITECTURE,
            "hash_head_input_dim": self.head.in_channels * 8 * 8,
            "hash_head_hidden_dim": 0,
            "hash_head_dropout": 0.0,
            "binary_encoding": BINARY_ENCODING,
            "binary_storage": BINARY_STORAGE,
            "binary_bitorder": self.bitorder,
            "backbone_architecture": self.backbone_architecture,
            "float_descriptor_dim": self.float_descriptor_dim,
            "backbone_trainable": self.backbone_trainable,
            "backbone_dropout": float(self.backbone[6].p),
            "freeze_backbone_bn": self.freeze_backbone_bn,
            "quantization_fp32": self.quantization_fp32,
            "binary_initialization": "fold_final_bn_hadamard256_v1",
        }


def build_direct_binary_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> DirectBinaryDescriptorModel:
    """提供给独立推理入口的自包含 checkpoint 加载接口。"""

    return DirectBinaryDescriptorModel.from_checkpoint(checkpoint)
