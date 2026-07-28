"""HardNet 网络结构定义。

作用：
    1. 按论文中的 L2Net/HardNet 结构构建 32x32 灰度 patch 描述子网络。
    2. 输入形状为 `[B, 1, 32, 32]`，输出形状为 `[B, 128]`。
    3. 输出描述子会做 L2 归一化，因此可以直接用欧氏距离比较 patch 相似度。

设计依据：
    HardNet 论文采用无池化卷积网络，通过 stride=2 卷积降低空间尺寸，
    最后用 8x8 卷积把 8x8 特征图变成 128 维描述子。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class HardNet(nn.Module):
    """HardNet/L2Net 描述子网络。

    参数：
        dropout:
            论文原始设置为 0.1；post-NIPS 版本曾使用 0.3。
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

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """前向计算 128 维单位长度描述子。"""
        descriptors = self.features(patches)
        # 最后一层卷积输出 `[B, 128, 1, 1]`，压平成 `[B, 128]`。
        descriptors = descriptors.view(descriptors.size(0), -1)
        # HardNet 距离公式假设描述子是单位向量；训练和评估都依赖这一点。
        return F.normalize(descriptors, p=2, dim=1)


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

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        descriptors = self.stem(patches)
        descriptors = self.features(descriptors)
        descriptors = self.adapter(descriptors)
        descriptors = self.descriptor_head(descriptors)
        descriptors = descriptors.flatten(start_dim=1)
        return F.normalize(descriptors, p=2, dim=1)


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


def build_descriptor_model(model_config: Mapping[str, Any] | None = None) -> nn.Module:
    """从统一配置构建训练和推理共用的 patch 描述子网络。"""
    config = model_config or {}
    architecture = normalize_model_architecture(config.get("architecture"))
    options = {
        "dropout": float(config.get("dropout", 0.1)),
        "descriptor_dim": int(config.get("descriptor_dim", 128)),
        "final_bn_affine": bool(config.get("final_bn_affine", False)),
    }
    if architecture == "hardnet":
        return HardNet(**options)
    return MobileHardNet(**options)


def checkpoint_model_architecture(checkpoint: Mapping[str, Any]) -> str:
    """读取 checkpoint 架构；旧格式缺少元数据时按原 HardNet 解释。"""
    architecture = checkpoint.get("model_architecture")
    saved_config = checkpoint.get("config")
    if architecture is None and isinstance(saved_config, Mapping):
        saved_model_config = saved_config.get("model")
        if isinstance(saved_model_config, Mapping):
            architecture = saved_model_config.get("architecture")
    return normalize_model_architecture(architecture)


def model_architecture(model: nn.Module) -> str:
    """返回 checkpoint 中持久化的模型架构标识。"""
    architecture = getattr(model, "architecture", None)
    return normalize_model_architecture(architecture)


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量，训练启动时用于打印模型规模。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
