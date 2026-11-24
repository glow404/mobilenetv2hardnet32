"""P9融合、冻结/联合梯度、阶段调度、续训及部署契约回归测试。

使用小批随机patch验证机制，不评估训练精度或完整识别性能。
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from hardnet_train.binary_model import binary_model_metadata, build_binary_descriptor_model
from hardnet_train.direct_binary_model import DirectBinaryDescriptorModel
from hardnet_train.data import FingerImagePairBatchSampler
from hardnet_train.model import HardNet
from hardnet_train.model_report import profile_model
from hardnet_train.train import load_config
from hardnet_train.train_binary import (
    binary_step_schedule, build_binary_optimizer, forward_binary_training_features,
    load_binary_checkpoint, save_binary_checkpoint,
)
from match_new.hadamard_binarization import fwht_torch


class DirectBinaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cls.source = HardNet(descriptor_dim=256, final_bn_affine=True).eval()
        with torch.no_grad():
            for layer in cls.source.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.running_mean.normal_(0, 0.1)
                    layer.running_var.uniform_(0.5, 1.5)
            cls.source.features[-1].weight.uniform_(0.8, 1.2)
            cls.source.features[-1].bias.normal_(0, 0.1)
        cls.patches = torch.randn(4, 1, 32, 32)

    def make_model(self):
        return build_binary_descriptor_model(self.source, {
            "architecture": "hardnet_direct_binary_v1", "hash_bits": 256,
            "backbone_trainable": True, "freeze_backbone_bn": True,
        })

    def test_fusion_matches_existing_fwht_and_does_not_mutate_source(self):
        before = self.source.features[-2].weight.detach().clone()
        model = self.make_model().eval()
        with torch.no_grad():
            reference = fwht_torch(self.source.forward_features(self.patches).q, normalize=True)
            actual = model.forward_logits(self.patches)
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-4)
        self.assertTrue(torch.equal(actual >= 0, reference >= 0))
        self.assertEqual(model.encode_binary(self.patches, packed=True).shape, (4, 32))
        torch.testing.assert_close(before, self.source.features[-2].weight)
        self.assertNotEqual(model.backbone[0][0].weight.data_ptr(), self.source.features[0][0].weight.data_ptr())
        profile = profile_model(model)
        self.assertEqual((profile.parameters, profile.macs_per_patch), (2384288, 40140800))

    def test_stage_gradients_bn_and_independent_teacher(self):
        model = self.make_model()
        teacher = copy.deepcopy(self.source).eval().requires_grad_(False)
        teacher_before = {k: v.clone() for k, v in teacher.state_dict().items()}
        model.set_backbone_trainable(False)
        model.train()
        features = forward_binary_training_features(model, self.patches, teacher)
        features.continuous.square().mean().backward()
        self.assertIsNotNone(model.head.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        model.zero_grad(set_to_none=True)
        model.set_backbone_trainable(True)
        model.train()
        means = [m.running_mean.clone() for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d)]
        features = forward_binary_training_features(model, self.patches, teacher)
        features.continuous.square().mean().backward()
        self.assertGreater(float(model.backbone[0][0].weight.grad.abs().sum()), 0)
        for before, layer in zip(means, [m for m in model.backbone.modules() if isinstance(m, nn.BatchNorm2d)]):
            torch.testing.assert_close(before, layer.running_mean, rtol=0, atol=0)
        for key, value in teacher.state_dict().items():
            torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertFalse(features.teacher_f.requires_grad)

    def test_schedule_resets_at_joint_boundary_and_lr_groups_survive(self):
        kwargs = dict(total_steps=8, head_warmup_steps=2, base_lr=1e-4,
                      scheduler="warmup_cosine", warmup_steps=0, eta_min=1e-6,
                      temperature_start=1.0, temperature_end=0.1)
        self.assertEqual(binary_step_schedule(global_step=1, **kwargs), (1e-4, 1.0, False))
        self.assertEqual(binary_step_schedule(global_step=2, **kwargs), (1e-4, 1.0, True))
        self.assertAlmostEqual(binary_step_schedule(global_step=7, **kwargs)[1], 0.1)
        model = self.make_model()
        optimizer = build_binary_optimizer(model, {"name": "adamw", "lr": 1e-4, "backbone_lr_multiplier": 0.1})
        self.assertEqual(len(optimizer.param_groups), 4)
        self.assertEqual({g["lr_scale"] for g in optimizer.param_groups}, {0.1, 1.0})
        model.set_backbone_trainable(False)
        before = model.backbone[0][0].weight.detach().clone()
        model.forward_features(self.patches).continuous.square().mean().backward()
        optimizer.step()
        torch.testing.assert_close(model.backbone[0][0].weight, before, atol=0, rtol=0)

    def test_checkpoint_self_contained_and_resume_rejects_changed_schedule(self):
        config = load_config(Path(__file__).with_name("config_binary_m0_direct_256.yaml"))
        config["backbone"]["checkpoint_sha256"] = "test-source-hash"
        model = self.make_model()
        optimizer = build_binary_optimizer(model, config["optimizer"])
        model.set_backbone_trainable(False)
        model.forward_features(self.patches).continuous.square().mean().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"last.pt"
            save_binary_checkpoint(path, model, optimizer, config, Path("missing_source.pt"), {}, 1, 1000, {})
            payload = torch.load(path, weights_only=True)
            restored = DirectBinaryDescriptorModel.from_checkpoint(payload).eval()
            torch.testing.assert_close(restored.forward_logits(self.patches), model.eval().forward_logits(self.patches))
            self.assertFalse(any(k.startswith("teacher") for k in payload["model"]))
            fresh = self.make_model()
            opt = build_binary_optimizer(fresh, config["optimizer"])
            self.assertEqual(load_binary_checkpoint(path, fresh, opt, torch.device("cpu"), 1000, config), (2, 1000))
            self.assertEqual({g["lr_scale"] for g in opt.param_groups}, {0.1, 1.0})
            config["training"]["head_warmup_epochs"] += 1
            with self.assertRaisesRegex(ValueError, "contract changed"):
                load_binary_checkpoint(path, fresh, opt, torch.device("cpu"), 1000, config)

    def test_invalid_source_and_legacy_factory(self):
        with self.assertRaisesRegex(ValueError, "exactly 256"):
            build_binary_descriptor_model(self.source, {"architecture": "hardnet_direct_binary_v1", "hash_bits": 512})
        legacy = build_binary_descriptor_model(copy.deepcopy(self.source), {"hash_bits": 256})
        self.assertEqual(binary_model_metadata(legacy)["model_architecture"], "residual_binary_hash_v1")

    def test_absolute_epoch_restores_batch_sampling_sequence(self):
        records = [SimpleNamespace(finger_id="a", image_pair_id="a-b", point_group=i) for i in range(12)]
        original = FingerImagePairBatchSampler(records, 4, 1, 2, seed=42)
        list(original)
        expected = list(original)
        resumed = FingerImagePairBatchSampler(records, 4, 1, 2, seed=42)
        resumed.set_epoch(1)
        self.assertEqual(list(resumed), expected)


if __name__ == "__main__":
    unittest.main()
