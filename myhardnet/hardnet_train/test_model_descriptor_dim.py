"""Descriptor-dimension contracts for all float HardNet architectures."""

from __future__ import annotations

import pytest
import torch

from hardnet_train.model import (
    CoordinateAttention,
    build_descriptor_model,
    checkpoint_descriptor_dim,
    checkpoint_model_architecture,
)
from hardnet_train.train import default_output_dir


# 显式保留已移除名称，避免以后误把旧 m2 配置静默映射到 m0/m1。
REMOVED_ARCHITECTURES = (
    "hardnet_strong_v2",
    "hardnetstrongv2",
    "hardnet_strong_v2_256",
    "strong_v2",
    "strongv2",
    "m2",
)


@pytest.mark.parametrize("architecture", ["hardnet", "mobile_hardnet"])
@pytest.mark.parametrize("descriptor_dim", [64, 128, 256])
def test_float_architectures_support_configurable_descriptor_dim(
    architecture: str,
    descriptor_dim: int,
) -> None:
    model = build_descriptor_model(
        {
            "architecture": architecture,
            "descriptor_dim": descriptor_dim,
            "dropout": 0.1,
            "final_bn_affine": False,
        }
    ).eval()

    with torch.inference_mode():
        features = model.forward_features(torch.randn(2, 1, 32, 32))

    assert model.descriptor_dim == descriptor_dim
    assert features.q.shape == (2, descriptor_dim)
    assert features.f.shape == (2, descriptor_dim)
    torch.testing.assert_close(
        torch.linalg.vector_norm(features.f, dim=1),
        torch.ones(2),
    )


@pytest.mark.parametrize("architecture", ["hardnet", "mobile_hardnet"])
@pytest.mark.parametrize("descriptor_dim", [64, 256])
def test_custom_descriptor_dim_supports_model_backward(
    architecture: str,
    descriptor_dim: int,
) -> None:
    model = build_descriptor_model(
        {"architecture": architecture, "descriptor_dim": descriptor_dim}
    )
    anchor = model(torch.randn(4, 1, 32, 32))
    positive = model(torch.randn(4, 1, 32, 32))
    loss = (1.0 - (anchor * positive).sum(dim=1)).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("architecture", ["hardnet", "mobile_hardnet"])
def test_float_architectures_keep_128_as_default(architecture: str) -> None:
    automatic = build_descriptor_model(
        {"architecture": architecture, "descriptor_dim": "auto"}
    )
    explicit = build_descriptor_model(
        {"architecture": architecture, "descriptor_dim": 128}
    )

    assert automatic.descriptor_dim == explicit.descriptor_dim == 128
    assert automatic.state_dict().keys() == explicit.state_dict().keys()
    for name, tensor in automatic.state_dict().items():
        assert tensor.shape == explicit.state_dict()[name].shape

    # Changing descriptor dimension must only resize the configured output head.
    expected_output_shape = {
        "hardnet": (128, 128, 8, 8),
        "mobile_hardnet": (128, 192, 1, 1),
    }[architecture]
    output_weight_key = {
        "hardnet": "features.7.weight",
        "mobile_hardnet": "descriptor_head.2.weight",
    }[architecture]
    assert automatic.state_dict()[output_weight_key].shape == expected_output_shape


def test_coordinate_attention_preserves_shape_and_supports_backward() -> None:
    attention = CoordinateAttention(96, reduction=32)
    inputs = torch.randn(2, 96, 8, 8, requires_grad=True)

    outputs = attention(inputs)
    outputs.square().mean().backward()

    assert attention.bottleneck_channels == 8
    assert outputs.shape == inputs.shape
    assert torch.all(torch.isfinite(outputs))
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in attention.parameters())


def test_mobile_hardnet_contains_coordinate_attention() -> None:
    model = build_descriptor_model({"architecture": "mobile_hardnet"})

    assert model.architecture == "mobile_hardnet"
    assert isinstance(model.coordinate_attention, CoordinateAttention)


@pytest.mark.parametrize("architecture", REMOVED_ARCHITECTURES)
def test_removed_architectures_are_rejected(architecture: str) -> None:
    with pytest.raises(ValueError, match="Unsupported model architecture"):
        build_descriptor_model({"architecture": architecture})
    with pytest.raises(ValueError, match="Unsupported model architecture"):
        default_output_dir(architecture, 256)


@pytest.mark.parametrize("architecture", REMOVED_ARCHITECTURES)
@pytest.mark.parametrize("metadata_field", ["model_architecture", "config", "resolved_config"])
def test_removed_checkpoint_architectures_are_rejected(
    architecture: str,
    metadata_field: str,
) -> None:
    checkpoint = (
        {metadata_field: architecture}
        if metadata_field == "model_architecture"
        else {metadata_field: {"model": {"architecture": architecture}}}
    )

    with pytest.raises(ValueError, match="Unsupported model architecture"):
        checkpoint_model_architecture(checkpoint)


@pytest.mark.parametrize(
    ("architecture", "directory_label"),
    [("hardnet", "hardnet"), ("mobile_hardnet", "mobile")],
)
@pytest.mark.parametrize(("descriptor_dim", "directory_dim"), [(None, 128), (64, 64), (256, 256)])
def test_default_output_directories_remain_compatible(
    architecture: str,
    directory_label: str,
    descriptor_dim: int | None,
    directory_dim: int,
) -> None:
    assert default_output_dir(architecture, descriptor_dim) == (
        f"../outputs/models/hardnet_train_{directory_label}_{directory_dim}_"
        "fixed_candidate_pool_v3"
    )


def test_legacy_checkpoint_without_metadata_keeps_hardnet_128_fallback() -> None:
    assert checkpoint_model_architecture({}) == "hardnet"
    assert checkpoint_descriptor_dim({}) == 128


@pytest.mark.parametrize("architecture", ["hardnet", "mobile_hardnet"])
@pytest.mark.parametrize("metadata_field", ["model_architecture", "config", "resolved_config"])
def test_supported_checkpoint_metadata_keeps_default_descriptor_dim(
    architecture: str,
    metadata_field: str,
) -> None:
    checkpoint = (
        {metadata_field: architecture}
        if metadata_field == "model_architecture"
        else {metadata_field: {"model": {"architecture": architecture}}}
    )

    assert checkpoint_model_architecture(checkpoint) == architecture
    assert checkpoint_descriptor_dim(checkpoint) == 128


@pytest.mark.parametrize("architecture", ["hardnet", "mobile_hardnet"])
@pytest.mark.parametrize("descriptor_dim", [128, 256])
def test_state_dict_roundtrip_preserves_float_descriptor_outputs(
    architecture: str,
    descriptor_dim: int,
) -> None:
    options = {"architecture": architecture, "descriptor_dim": descriptor_dim}
    original = build_descriptor_model(options).eval()
    restored = build_descriptor_model(options).eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    patches = torch.randn(2, 1, 32, 32)

    with torch.inference_mode():
        torch.testing.assert_close(restored(patches), original(patches))


@pytest.mark.parametrize("architecture", ["hardnet", "mobile_hardnet"])
@pytest.mark.parametrize("descriptor_dim", [0, -1])
def test_float_architectures_reject_non_positive_descriptor_dim(
    architecture: str,
    descriptor_dim: int,
) -> None:
    with pytest.raises(ValueError, match="descriptor_dim must be positive"):
        build_descriptor_model(
            {"architecture": architecture, "descriptor_dim": descriptor_dim}
        )
