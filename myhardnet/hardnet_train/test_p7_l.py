"""P7-L浮点训练、直接头融合和自包含部署的回归测试。

使用小批patch验证结构、梯度和Torch/ORT协议，不用于认证精度或识别延时。
"""

from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from hardnet_train.binary_loss import BinaryDescriptorLoss
from hardnet_train.binary_model import binary_model_metadata, build_binary_descriptor_model
from hardnet_train.direct_binary_model import DirectBinaryDescriptorModel
from hardnet_train.model import (
    PlainSpatialL, build_descriptor_model, checkpoint_model_architecture,
    default_descriptor_dim,
)
from hardnet_train.model_report import profile_model
from hardnet_train.train import default_output_dir, load_config
from hardnet_train.train_binary import (
    build_binary_optimizer, direct_training_contract, forward_binary_training_features,
)
from match_new.hadamard_binarization import fwht_torch
from match_new.runtime import HardNetDescriptor


class P7LTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        cls.source = PlainSpatialL(final_bn_affine=True).eval()
        with torch.no_grad():
            for layer in cls.source.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.running_mean.normal_(0, 0.1)
                    layer.running_var.uniform_(0.5, 1.5)
            cls.source.features[-1].weight.uniform_(0.8, 1.2)
            cls.source.features[-1].bias.normal_(0, 0.1)
        cls.patches = torch.randn(4, 1, 32, 32)

    def make_direct(self):
        return build_binary_descriptor_model(self.source, {
            "architecture": "hardnet_direct_binary_v1", "hash_bits": 256,
            "backbone_trainable": True, "quantization_fp32": True,
        })

    def test_architecture_shapes_and_budget(self):
        model = build_descriptor_model({"architecture": "p7-l"}).eval()
        self.assertIsInstance(model, PlainSpatialL)
        self.assertEqual(default_descriptor_dim("p7_l"), 256)
        self.assertEqual(checkpoint_model_architecture({"model_architecture": "plain_spatial_l"}), "plain_spatial_l")
        self.assertIn("p7_l_256", default_output_dir("plain_spatial_l"))
        self.assertEqual([block[0].out_channels for block in model.features[:6]], [24, 24, 48, 48, 96, 96])
        output = model(self.patches)
        self.assertEqual(output.shape, (4, 256))
        torch.testing.assert_close(output.norm(dim=1), torch.ones(4))
        float_profile = profile_model(model)
        direct_profile = profile_model(self.make_direct())
        self.assertEqual(float_profile.parameters, 1734456)
        self.assertEqual(direct_profile.parameters, 1734712)
        self.assertEqual(direct_profile.macs_per_patch, 23027712)
        self.assertEqual(float_profile.macs_per_patch, direct_profile.macs_per_patch)

    def test_hadamard_fusion_and_checkpoint_round_trip(self):
        model = self.make_direct().eval()
        before = {key: value.clone() for key, value in self.source.state_dict().items()}
        with torch.no_grad():
            expected = fwht_torch(self.source.forward_features(self.patches).q, normalize=True)
            actual = model.forward_logits(self.patches)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        self.assertTrue(torch.equal(actual >= 0, expected >= 0))
        payload = {**binary_model_metadata(model), "model": model.state_dict()}
        restored = DirectBinaryDescriptorModel.from_checkpoint(payload).eval()
        self.assertEqual(payload["hash_head_input_dim"], 6144)
        self.assertTrue(restored.quantization_fp32)
        torch.testing.assert_close(restored.encode_binary(self.patches, packed=True), model.encode_binary(self.patches, packed=True))
        for key, value in self.source.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        invalid = {**payload, "backbone_architecture": "hardnet"}
        with self.assertRaisesRegex(ValueError, "hash_head_input_dim"):
            DirectBinaryDescriptorModel.from_checkpoint(invalid)

    def test_joint_step_has_backbone_gradient_and_fixed_teacher(self):
        model = self.make_direct().train()
        teacher = copy.deepcopy(self.source).eval().requires_grad_(False)
        before = teacher.features[0][0].weight.detach().clone()
        optimizer = build_binary_optimizer(model, {"name": "adamw", "lr": 1e-4})
        features = forward_binary_training_features(model, self.patches, teacher)
        positive = forward_binary_training_features(model, self.patches + 0.05 * torch.randn_like(self.patches), teacher)
        loss, _ = BinaryDescriptorLoss(margin=0.5)(
            features, positive, point_group=torch.arange(4), finger_group=torch.arange(4),
            anchor_xy=torch.zeros(4, 2), positive_xy=torch.zeros(4, 2),
            anchor_coordinate_frame_group=torch.arange(4),
            positive_coordinate_frame_group=torch.arange(4) + 4,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.backbone[0][0].weight.grad.abs().sum()), 0)
        optimizer.step()
        torch.testing.assert_close(teacher.features[0][0].weight, before, atol=0, rtol=0)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_fp32_quantization_preserves_small_gradients(self):
        model = self.make_direct()
        logits = torch.full((2, 256), 5.0, dtype=torch.float16, requires_grad=True)
        with patch.object(model, "forward_logits", return_value=logits):
            features = model.forward_features(torch.empty(0))
            self.assertEqual(features.continuous.dtype, torch.float32)
            self.assertTrue(torch.equal(features.quantized, torch.ones_like(features.quantized)))
            features.quantized.sum().backward()
        self.assertTrue(torch.all(logits.grad > 0))
        model.quantization_fp32 = False
        logits.grad = None
        with patch.object(model, "forward_logits", return_value=logits):
            model.forward_features(torch.empty(0)).quantized.sum().backward()
        self.assertTrue(torch.all(logits.grad == 0))

    def test_configs_and_quantization_resume_contract(self):
        base = Path(__file__).parent
        floating = load_config(base / "config_p7_l_256.yaml")
        binary = load_config(base / "config_binary_p7_l_direct_256.yaml")
        self.assertEqual(floating["data"], binary["data"])
        self.assertEqual(floating["model"]["architecture"], binary["backbone"]["architecture"])
        self.assertTrue(binary["binary_model"]["quantization_fp32"])
        binary["backbone"]["checkpoint_sha256"] = "test"
        fp32_contract = direct_training_contract(binary)
        binary["binary_model"]["quantization_fp32"] = False
        self.assertNotEqual(fp32_contract, direct_training_contract(binary))

    def _runtime_round_trip(self, backend):
        patches = self.patches.numpy()
        with tempfile.TemporaryDirectory() as directory:
            for kind in ("float", "binary"):
                with self.subTest(kind=kind, backend=backend):
                    model = self.source if kind == "float" else self.make_direct().eval()
                    metadata = (binary_model_metadata(model) if kind == "binary" else {
                        "model_architecture": "plain_spatial_l", "descriptor_dim": 256,
                        "config": {"model": {"final_bn_affine": True}},
                    })
                    path = Path(directory) / f"{kind}.pt"
                    torch.save({**metadata, "model": model.state_dict(), "backbone_checkpoint": "missing.pt"}, path)
                    runtime = HardNetDescriptor({"model": {
                        "checkpoint": str(path), "device": "cpu", "architecture": "auto",
                        "descriptor_dim": "auto", "batch_size": 2, "cpu_num_threads": 1,
                        "channels_last": False, "fuse_conv_bn": True, "inference_precision": "fp32",
                        "inference_backend": backend, "onnx_path": str(Path(directory) / f"{kind}.onnx"),
                        "backend_intra_op_num_threads": 1, "backend_inter_op_num_threads": 1,
                    }})
                    with torch.inference_mode():
                        expected = model.encode_binary(self.patches, packed=True) if kind == "binary" else model(self.patches)
                    actual = runtime.describe(patches)
                    if kind == "binary":
                        np.testing.assert_array_equal(actual, expected.numpy())
                        self.assertEqual(runtime.describe(patches[:0]).shape, (0, 32))
                    else:
                        np.testing.assert_allclose(actual, expected.numpy(), atol=2e-5, rtol=2e-4)

    def test_torch_runtime_float_and_binary(self):
        self._runtime_round_trip("pytorch")

    def test_ort_runtime_float_and_binary(self):
        if not all(importlib.util.find_spec(name) for name in ("onnx", "onnxruntime")):
            self.skipTest("ORT packages absent in this Python environment")
        self._runtime_round_trip("onnxruntime")


if __name__ == "__main__":
    unittest.main()
