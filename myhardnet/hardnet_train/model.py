"""HardNet 系列 32×32 灰度 patch 描述子网络。

本模块保留旧 HardNet 的稳定 checkpoint 协议，以带 Coordinate Attention 的
MobileHardNet 提供轻量主干。两个模型通过 `forward_features()` 暴露归一化前
特征 `q` 和 L2 单位描述子 `f`，
供浮点匹配及独立 hash head 共用。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


class DescriptorFeatures(NamedTuple):
    """描述子网络的连续特征接口。

    ``q`` 保留归一化前的信息，供后续独立 hash head 使用；``f`` 是现有训练和
    L2 matcher 使用的单位向量。NamedTuple 保持返回值只读且便于 TorchScript。
    """

    q: torch.Tensor
    f: torch.Tensor


class HardNet(nn.Module):
    """HardNet/L2Net 描述子网络。

    参数：
        dropout:
            论文原始设置为 0.1；
            小指纹数据量和噪声情况不确定时，建议先用 0.1 做基线。
        descriptor_dim:
            输出描述子维度。默认 128 保持 HardNet/SIFT 对齐以及旧 checkpoint
            兼容；使用其他维度时只改变最后一个 8×8 空间投影的输出通道数。
        final_bn_affine:
            最后一层 BN 默认不学习 affine 参数，更接近常见 HardNet 实现。
    """

    architecture = "hardnet"

    def __init__(self, dropout: float = 0.1, descriptor_dim: int = 128, final_bn_affine: bool = False) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        if self.descriptor_dim <= 0:
            raise ValueError(f"descriptor_dim must be positive, got {descriptor_dim}.")

        # 尺寸变化：
        #   32x32 -> 32x32 -> 32x32 -> 16x16 -> 16x16 -> 8x8 -> 8x8 -> 1x1
        # 通道变化：
        #   1 -> 32 -> 32 -> 64 -> 64 -> 128 -> 128 -> 128
        self.features = nn.Sequential(
            self._conv_block(1, 32, stride=1),
            self._conv_block(32, 32, stride=1),
            self._conv_block(32, 64, stride=2),
            self._conv_block(64, 64, stride=1),
            self._conv_block(64, 128, stride=2),
            self._conv_block(128, 128, stride=1),
            nn.Dropout(p=float(dropout)),
            nn.Conv2d(
                128,
                self.descriptor_dim,
                kernel_size=8,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.descriptor_dim, affine=final_bn_affine),
        )
        self.reset_parameters()

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        """构造论文中重复出现的 `3x3 Conv + BN + ReLU` 块。"""
        return nn.Sequential(
            # bias=False 是因为后面接 BatchNorm，卷积 bias 会被 BN 平移项吸收。
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def reset_parameters(self) -> None:
        """按论文描述初始化参数。

        论文写明卷积权重使用 orthogonal 初始化，gain=0.6，bias=0.01。
        当前卷积层 bias=False，因此 bias 分支主要是给未来改结构时兜底。
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=0.6)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)

    def forward_features(self, patches: torch.Tensor) -> DescriptorFeatures:
        """返回归一化前特征 ``q`` 与 L2 单位描述子 ``f``。"""
        q = self.features(patches).flatten(start_dim=1)
        return DescriptorFeatures(q=q, f=F.normalize(q, p=2, dim=1))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """前向计算配置维度的单位长度描述子。"""
        return self.forward_features(patches).f


class InvertedResidual(nn.Module):
    """MobileNetV2 倒残差块，使用线性瓶颈保护低维描述特征。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion: int,
    ) -> None:
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError(f"stride must be 1 or 2, got {stride}.")
        if expansion < 1:
            raise ValueError(f"expansion must be >= 1, got {expansion}.")

        hidden_channels = in_channels * expansion
        layers: list[nn.Module] = []
        if expansion != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU6(inplace=True),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.block(inputs)
        return inputs + outputs if self.use_residual else outputs


class CoordinateAttention(nn.Module):
    """保留横纵坐标信息的轻量通道注意力。

    分别沿宽、高方向聚合特征，将两个方向的上下文联合编码后再生成独立的
    高度和宽度注意力图。模块不改变输入形状，适合放在全图空间投影之前。
    """

    def __init__(self, channels: int, reduction: int = 32) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}.")

        bottleneck_channels = max(8, int(channels) // int(reduction))
        self.channels = int(channels)
        self.bottleneck_channels = bottleneck_channels
        self.shared_projection = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.bottleneck_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.Hardswish(inplace=True),
        )
        self.height_projection = nn.Conv2d(
            self.bottleneck_channels,
            self.channels,
            kernel_size=1,
        )
        self.width_projection = nn.Conv2d(
            self.bottleneck_channels,
            self.channels,
            kernel_size=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.channels:
            raise ValueError(
                f"CoordinateAttention expects [N,{self.channels},H,W], "
                f"got {tuple(inputs.shape)}."
            )

        height, width = int(inputs.shape[2]), int(inputs.shape[3])
        height_context = inputs.mean(dim=3, keepdim=True)
        width_context = inputs.mean(dim=2, keepdim=True).transpose(2, 3)
        encoded = self.shared_projection(
            torch.cat([height_context, width_context], dim=2)
        )
        height_encoded, width_encoded = torch.split(
            encoded,
            [height, width],
            dim=2,
        )
        width_encoded = width_encoded.transpose(2, 3)
        height_attention = torch.sigmoid(
            self.height_projection(height_encoded)
        )
        width_attention = torch.sigmoid(
            self.width_projection(width_encoded)
        )
        return inputs * height_attention * width_attention


class MobileHardNet(nn.Module):
    """面向 32x32 灰度 patch 的 MobileNetV2 风格轻量描述子网络。

    网络只做两次空间下采样，并用全图 depthwise 卷积学习每个通道的空间布局，
    adapter 后使用 Coordinate Attention 联合建模横纵方向的位置与通道信息，
    再生成全图空间描述子。输出协议与 HardNet 一致；其他维度只改变末端
    1×1 线性投影。
    """

    architecture = "mobile_hardnet"

    def __init__(
        self,
        dropout: float = 0.1,
        descriptor_dim: int = 128,
        final_bn_affine: bool = False,
    ) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        if self.descriptor_dim <= 0:
            raise ValueError(f"descriptor_dim must be positive, got {descriptor_dim}.")

        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )
        self.features = nn.Sequential(
            InvertedResidual(16, 16, stride=1, expansion=1),
            InvertedResidual(16, 24, stride=2, expansion=2),
            InvertedResidual(24, 24, stride=1, expansion=2),
            InvertedResidual(24, 32, stride=2, expansion=2),
            InvertedResidual(32, 32, stride=1, expansion=2),
            InvertedResidual(32, 48, stride=1, expansion=2),
            InvertedResidual(48, 48, stride=1, expansion=2),
            InvertedResidual(48, 64, stride=1, expansion=2),
        )
        self.adapter = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU6(inplace=True),
        )
        self.coordinate_attention = CoordinateAttention(96, reduction=32)
        self.descriptor_head = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            # depth multiplier=2：每个通道保留两组完整 patch 空间响应。
            nn.Conv2d(96, 192, kernel_size=8, groups=96, bias=False),
            # 两层之间保持线性，使其近似原 HardNet 的全通道空间投影。
            nn.Conv2d(192, self.descriptor_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.descriptor_dim, affine=final_bn_affine),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """使用适合 ReLU6 和残差网络的 Kaiming 初始化。"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, patches: torch.Tensor) -> DescriptorFeatures:
        q = self.stem(patches)
        q = self.features(q)
        q = self.adapter(q)
        q = self.coordinate_attention(q)
        q = self.descriptor_head(q).flatten(start_dim=1)
        return DescriptorFeatures(q=q, f=F.normalize(q, p=2, dim=1))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.forward_features(patches).f


_MODEL_ARCHITECTURE_ALIASES = {
    "hardnet": "hardnet",
    "l2net": "hardnet",
    "mobile_hardnet": "mobile_hardnet",
    "mobilehardnet": "mobile_hardnet",
    "mobilenet_v2": "mobile_hardnet",
    "mobilev2": "mobile_hardnet",
}


def normalize_model_architecture(value: str | None) -> str:
    """把配置中的模型名称规整为 checkpoint 使用的稳定标识。"""
    key = str(value or "hardnet").strip().lower()
    if key not in _MODEL_ARCHITECTURE_ALIASES:
        supported = ", ".join(sorted(_MODEL_ARCHITECTURE_ALIASES))
        raise ValueError(f"Unsupported model architecture: {value!r}. Supported: {supported}")
    return _MODEL_ARCHITECTURE_ALIASES[key]


def default_descriptor_dim(architecture: str | None) -> int:
    """返回架构默认维度；旧 checkpoint 的回退规则集中维护在此处。"""

    normalize_model_architecture(architecture)
    return 128


def resolve_descriptor_dim(architecture: str | None, value: Any = None) -> int:
    """解析显式维度或按架构处理 ``auto`` 配置。"""

    normalized = normalize_model_architecture(architecture)
    if value is None or str(value).strip().lower() in {"", "auto"}:
        return default_descriptor_dim(normalized)
    descriptor_dim = int(value)
    if descriptor_dim <= 0:
        raise ValueError(f"descriptor_dim must be positive, got {descriptor_dim}.")
    return descriptor_dim


def build_descriptor_model(model_config: Mapping[str, Any] | None = None) -> nn.Module:
    """从统一配置构建训练和推理共用的 patch 描述子网络。"""
    config = model_config or {}
    architecture = normalize_model_architecture(config.get("architecture"))
    options = {
        "dropout": float(config.get("dropout", 0.1)),
        "descriptor_dim": resolve_descriptor_dim(
            architecture,
            config.get("descriptor_dim"),
        ),
        "final_bn_affine": bool(config.get("final_bn_affine", False)),
    }
    if architecture == "hardnet":
        return HardNet(**options)
    return MobileHardNet(**options)


def checkpoint_model_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """读取 checkpoint 中保存的模型配置，并兼容新旧配置字段。"""
    for key in ("resolved_config", "config"):
        saved_config = checkpoint.get(key)
        if not isinstance(saved_config, Mapping):
            continue
        saved_model_config = saved_config.get("model")
        if isinstance(saved_model_config, Mapping):
            return dict(saved_model_config)
    return {}


def checkpoint_model_architecture(checkpoint: Mapping[str, Any]) -> str:
    """读取 checkpoint 架构；旧格式缺少元数据时按原 HardNet 解释。"""
    architecture = checkpoint.get("model_architecture")
    if architecture is None:
        architecture = checkpoint_model_config(checkpoint).get("architecture")
    return normalize_model_architecture(architecture)


def checkpoint_descriptor_dim(checkpoint: Mapping[str, Any]) -> int:
    """读取描述子维度；旧 HardNet checkpoint 缺少元数据时返回 128。"""
    value = checkpoint.get("descriptor_dim")
    if value is None:
        value = checkpoint_model_config(checkpoint).get("descriptor_dim")
    if value is None:
        value = default_descriptor_dim(checkpoint_model_architecture(checkpoint))
    descriptor_dim = int(value)
    if descriptor_dim <= 0:
        raise ValueError(f"Checkpoint descriptor_dim must be positive, got {descriptor_dim}.")
    return descriptor_dim


def model_architecture(model: nn.Module) -> str:
    """返回 checkpoint 中持久化的模型架构标识。"""
    architecture = getattr(model, "architecture", None)
    return normalize_model_architecture(architecture)


def model_descriptor_dim(model: nn.Module) -> int:
    """返回训练、checkpoint 和模板共享的描述子维度。"""
    value = getattr(model, "descriptor_dim", None)
    if value is None:
        raise ValueError(f"Model {type(model).__name__} does not expose descriptor_dim.")
    descriptor_dim = int(value)
    if descriptor_dim <= 0:
        raise ValueError(f"Model descriptor_dim must be positive, got {descriptor_dim}.")
    return descriptor_dim


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量，训练启动时用于打印模型规模。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def estimate_macs_per_patch(
    model: nn.Module,
    input_size: tuple[int, int] = (32, 32),
) -> int:
    """用一次无梯度前向统计 Conv2d/Linear 的乘加次数，不依赖额外分析库。"""

    total_macs = 0
    handles: list[Any] = []

    def conv_hook(module: nn.Conv2d, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        nonlocal total_macs
        output_per_patch = output.numel() // max(int(output.shape[0]), 1)
        kernel_macs = (
            module.kernel_size[0]
            * module.kernel_size[1]
            * module.in_channels
            // module.groups
        )
        total_macs += int(output_per_patch * kernel_macs)

    def linear_hook(module: nn.Linear, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        nonlocal total_macs
        output_per_patch = output.numel() // max(int(output.shape[0]), 1)
        total_macs += int(output_per_patch * module.in_features)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    parameter = next(model.parameters())
    sample = torch.zeros(
        (1, 1, int(input_size[0]), int(input_size[1])),
        device=parameter.device,
        dtype=parameter.dtype,
    )
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            model(sample)
    finally:
        model.train(was_training)
        for handle in handles:
            handle.remove()
    return total_macs
