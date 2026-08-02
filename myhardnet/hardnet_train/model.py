"""HardNet 系列 32×32 灰度 patch 描述子网络。

本模块保留旧 HardNet、MobileHardNet 的稳定 checkpoint 协议，并以唯一的
Strong V2 架构提供高精度主干。所有模型通过 `forward_features()` 暴露
归一化前特征 `q` 和 L2 单位描述子 `f`，供浮点匹配及独立 hash head 共用。
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
            HardNet/SIFT 对齐为 128 维。这里保留参数但限制为 128，
            避免不小心改坏论文结构。
        final_bn_affine:
            最后一层 BN 默认不学习 affine 参数，更接近常见 HardNet 实现。
    """

    architecture = "hardnet"

    def __init__(self, dropout: float = 0.1, descriptor_dim: int = 128, final_bn_affine: bool = False) -> None:
        super().__init__()
        if descriptor_dim != 128:
            raise ValueError("The paper architecture expects descriptor_dim=128.")
        self.descriptor_dim = int(descriptor_dim)

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
            nn.Conv2d(128, descriptor_dim, kernel_size=8, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(descriptor_dim, affine=final_bn_affine),
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
        """前向计算 128 维单位长度描述子。"""
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


class MobileHardNet(nn.Module):
    """面向 32x32 灰度 patch 的 MobileNetV2 风格轻量描述子网络。

    网络只做两次空间下采样，并用全图 depthwise 卷积学习每个通道的空间布局，
    而不是用全局平均池化抹去指纹局部结构。输出协议与 HardNet 一致。
    """

    architecture = "mobile_hardnet"

    def __init__(
        self,
        dropout: float = 0.1,
        descriptor_dim: int = 128,
        final_bn_affine: bool = False,
    ) -> None:
        super().__init__()
        if descriptor_dim != 128:
            raise ValueError("MobileHardNet currently expects descriptor_dim=128.")
        self.descriptor_dim = int(descriptor_dim)

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
        self.descriptor_head = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            # depth multiplier=2：每个通道保留两组完整 patch 空间响应。
            nn.Conv2d(96, 192, kernel_size=8, groups=96, bias=False),
            # 两层之间保持线性，使其近似原 HardNet 的全通道空间投影。
            nn.Conv2d(192, descriptor_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(descriptor_dim, affine=final_bn_affine),
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
        q = self.descriptor_head(q).flatten(start_dim=1)
        return DescriptorFeatures(q=q, f=F.normalize(q, p=2, dim=1))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.forward_features(patches).f


class ContrastAwareStem(nn.Module):
    """同时编码原始纹理、多尺度上下文和局部对比度。"""

    def __init__(self, out_channels: int = 48) -> None:
        super().__init__()
        if out_channels % 3 != 0:
            raise ValueError("ContrastAwareStem out_channels must be divisible by 3.")
        branch_channels = out_channels // 3
        self.local_branch = nn.Sequential(
            nn.Conv2d(1, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(branch_channels), branch_channels),
            nn.SiLU(inplace=True),
        )
        self.context_branch = nn.Sequential(
            nn.Conv2d(1, branch_channels, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(_group_count(branch_channels), branch_channels),
            nn.SiLU(inplace=True),
        )
        self.contrast_branch = nn.Sequential(
            nn.Conv2d(1, branch_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(branch_channels), branch_channels),
            nn.SiLU(inplace=True),
        )
        self.fuse = ConvNormAct(out_channels, out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        local_contrast = inputs - F.avg_pool2d(
            inputs,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )
        features = torch.cat(
            [
                self.local_branch(inputs),
                self.context_branch(inputs),
                self.contrast_branch(local_contrast),
            ],
            dim=1,
        )
        return self.fuse(features)


def _group_count(channels: int, max_groups: int = 8) -> int:
    """返回可整除通道数的最大 GroupNorm 分组数。"""

    for groups in range(min(int(max_groups), int(channels)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Sequential):
    """Strong V2 共用的 Conv + GroupNorm + SiLU。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        if padding is None:
            padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class SqueezeExcitation(nn.Module):
    """以轻量通道注意力增强稳定脊线响应，抑制背景噪声通道。"""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden_channels = max(8, channels // max(int(reduction), 1))
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.projection(inputs)


class StableResBlock(nn.Module):
    """预激活残差块；抗混叠下采样与 LayerScale 提升深层训练稳定性。"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        if stride not in {1, 2}:
            raise ValueError(f"stride must be 1 or 2, got {stride}.")
        self.downsample = (
            nn.AvgPool2d(kernel_size=2, stride=2)
            if stride == 2
            else nn.Identity()
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.attention = SqueezeExcitation(out_channels)
        self.layer_scale = nn.Parameter(torch.full((out_channels,), 0.1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        source = self.downsample(inputs)
        shortcut = self.shortcut(source)
        outputs = self.conv1(F.silu(self.norm1(source), inplace=True))
        outputs = self.conv2(F.silu(self.norm2(outputs), inplace=True))
        outputs = self.attention(outputs)
        scale = self.layer_scale.view(1, -1, 1, 1)
        return shortcut + outputs * scale


class MultiScaleContext(nn.Module):
    """融合点响应、邻域纹理、扩张上下文和整块统计信息。"""

    def __init__(self, in_channels: int, out_channels: int = 256) -> None:
        super().__init__()
        branch_channels = out_channels // 4
        if branch_channels * 4 != out_channels:
            raise ValueError("MultiScaleContext out_channels must be divisible by 4.")
        self.point_branch = ConvNormAct(
            in_channels,
            branch_channels,
            kernel_size=1,
            padding=0,
        )
        self.local_branch = nn.Sequential(
            ConvNormAct(
                in_channels,
                in_channels,
                kernel_size=3,
                groups=in_channels,
            ),
            ConvNormAct(in_channels, branch_channels, kernel_size=1, padding=0),
        )
        self.dilated_branch = nn.Sequential(
            ConvNormAct(
                in_channels,
                in_channels,
                kernel_size=3,
                dilation=2,
                groups=in_channels,
            ),
            ConvNormAct(in_channels, branch_channels, kernel_size=1, padding=0),
        )
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvNormAct(in_channels, branch_channels, kernel_size=1, padding=0),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.layer_scale = nn.Parameter(torch.full((out_channels,), 0.1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        global_context = F.interpolate(
            self.global_branch(inputs),
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        context = torch.cat(
            [
                self.point_branch(inputs),
                self.local_branch(inputs),
                self.dilated_branch(inputs),
                global_context,
            ],
            dim=1,
        )
        scale = self.layer_scale.view(1, -1, 1, 1)
        return self.shortcut(inputs) + self.fuse(context) * scale


class GeM(nn.Module):
    """可学习 generalized-mean pooling，兼顾峰值纹理与全局稳定性。"""

    def __init__(self, initial_p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(initial_p)))
        self.eps = float(eps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        power = self.p.clamp(min=1.0, max=8.0)
        pooled = F.adaptive_avg_pool2d(inputs.clamp_min(self.eps).pow(power), 1)
        return pooled.pow(power.reciprocal())


class HardNetStrongV2(nn.Module):
    """版本化的高精度 32×32 浮点描述子主干。

    V2 保留局部空间投影，同时引入局部对比度、多尺度上下文、抗混叠残差和
    GeM 全局分支。输出维度不再绑定到 256，可独立训练 128/256/384/512 维模型。
    """

    architecture = "hardnet_strong_v2"

    def __init__(
        self,
        dropout: float = 0.1,
        descriptor_dim: int = 256,
        final_bn_affine: bool = False,
    ) -> None:
        super().__init__()
        if descriptor_dim <= 0:
            raise ValueError(f"descriptor_dim must be positive, got {descriptor_dim}.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        self.descriptor_dim = int(descriptor_dim)

        self.stem = ContrastAwareStem(48)
        self.stage1 = nn.Sequential(
            StableResBlock(48, 48),
            StableResBlock(48, 48),
        )
        self.down1 = StableResBlock(48, 96, stride=2)
        self.stage2 = nn.Sequential(
            StableResBlock(96, 96),
            StableResBlock(96, 96),
            StableResBlock(96, 96),
        )
        self.down2 = StableResBlock(96, 192, stride=2)
        self.stage3 = nn.Sequential(
            StableResBlock(192, 192),
            StableResBlock(192, 192),
            StableResBlock(192, 192),
            StableResBlock(192, 192),
        )
        self.fine_projection = ConvNormAct(96, 64, kernel_size=1, padding=0)
        self.context = MultiScaleContext(192 + 64, out_channels=256)
        self.head_norm = nn.Sequential(
            nn.GroupNorm(_group_count(256), 256),
            nn.SiLU(inplace=True),
        )
        self.spatial_projection = nn.Sequential(
            ConvNormAct(256, 256, kernel_size=3, stride=2),
            ConvNormAct(256, 256, kernel_size=3, stride=2),
            nn.Dropout2d(p=float(dropout) * 0.5),
            nn.Conv2d(256, self.descriptor_dim, kernel_size=2, bias=False),
        )
        self.global_pool = GeM()
        self.global_projection = nn.Linear(256, self.descriptor_dim, bias=False)
        self.fusion_norm = nn.LayerNorm(self.descriptor_dim * 2)
        self.fusion_dropout = nn.Dropout(p=float(dropout))
        self.fusion = nn.Linear(
            self.descriptor_dim * 2,
            self.descriptor_dim,
            bias=False,
        )
        self.output_norm = nn.BatchNorm1d(
            self.descriptor_dim,
            affine=bool(final_bn_affine),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化卷积/投影，并保留残差块的小尺度起点。"""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, patches: torch.Tensor) -> DescriptorFeatures:
        if patches.ndim != 4 or patches.shape[1:] != (1, 32, 32):
            raise ValueError(
                "HardNetStrongV2 expects input patches shaped [B,1,32,32], "
                f"got {tuple(patches.shape)}."
            )
        features = self.stem(patches)
        features = self.stage1(features)
        features = self.down1(features)
        fine_features = self.stage2(features)
        coarse_features = self.down2(fine_features)
        coarse_features = self.stage3(coarse_features)
        fine_context = F.avg_pool2d(
            self.fine_projection(fine_features),
            kernel_size=2,
            stride=2,
        )
        context = self.context(torch.cat([coarse_features, fine_context], dim=1))
        context = self.head_norm(context)

        spatial = self.spatial_projection(context).flatten(start_dim=1)
        global_context = self.global_pool(context).flatten(start_dim=1)
        global_context = self.global_projection(global_context)
        if spatial.shape[1] != self.descriptor_dim:
            raise ValueError(
                "HardNetStrongV2 spatial projection did not produce one descriptor per patch: "
                f"got {tuple(spatial.shape)}."
            )
        fused = torch.cat([spatial, global_context], dim=1)
        q = self.fusion(self.fusion_dropout(self.fusion_norm(fused)))
        q = self.output_norm(q)
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
    "hardnet_strong_v2": "hardnet_strong_v2",
    "hardnetstrongv2": "hardnet_strong_v2",
    "hardnet_strong_v2_256": "hardnet_strong_v2",
    "strong_v2": "hardnet_strong_v2",
    "strongv2": "hardnet_strong_v2",
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

    normalized = normalize_model_architecture(architecture)
    return 256 if normalized == "hardnet_strong_v2" else 128


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
    if architecture == "mobile_hardnet":
        return MobileHardNet(**options)
    return HardNetStrongV2(**options)


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
