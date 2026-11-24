"""P9自包含checkpoint的Torch/ORT部署回归，保留旧浮点和residual导出语义。"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hardnet_train.binary_model import binary_model_metadata, build_binary_descriptor_model
from hardnet_train.direct_binary_model import DirectBinaryDescriptorModel
from hardnet_train.model import HardNet
from match_new.cpu_inference_backends import _InferenceExportGraph
from match_new.runtime import HardNetDescriptor


class DirectRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_torch_and_ort_load_without_source_checkpoint(self):
        model = DirectBinaryDescriptorModel().eval()
        patches = np.random.default_rng(17).normal(size=(5, 1, 32, 32)).astype(np.float32)
        # 推理接口始终以最终256个logits作为符号判定输入。
        expected = model.encode_binary(torch.from_numpy(patches), packed=True).numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"student.pt"
            torch.save({**binary_model_metadata(model), "model": model.state_dict(),
                        "backbone_checkpoint": "does_not_exist.pt"}, path)
            config = {"model": {"checkpoint": str(path), "device": "cpu", "architecture": "auto",
                                 "batch_size": 2, "cpu_num_threads": 1, "channels_last": False,
                                 "fuse_conv_bn": True, "inference_precision": "fp32"}}
            runtime = HardNetDescriptor(config)
            np.testing.assert_array_equal(runtime.describe(patches), expected)
            codes, timing = runtime.describe_with_timing(patches)
            np.testing.assert_array_equal(codes, expected)
            self.assertEqual(timing["hardnet_hash_head_ms"], 0.0)
            self.assertEqual(runtime.describe(patches[:0]).shape, (0, 32))
            if not all(importlib.util.find_spec(name) for name in ("onnx", "onnxruntime")):
                self.skipTest("ORT packages absent; Torch path above passed")
            config["model"].update(inference_backend="onnxruntime", onnx_path=str(Path(directory)/"student.onnx"),
                                   backend_intra_op_num_threads=1, backend_inter_op_num_threads=1)
            ort_runtime = HardNetDescriptor(config)
            np.testing.assert_array_equal(ort_runtime.describe(patches), expected)
            # ORT直接输出logits，无额外PyTorch hash_head调用。
            graph = _InferenceExportGraph(ort_runtime.model, native_binary=True)
            torch.testing.assert_close(graph(torch.from_numpy(patches)), model.forward_logits(torch.from_numpy(patches)))

    def test_legacy_export_semantics_and_unknown_architecture_rejected(self):
        backbone = HardNet(descriptor_dim=256).eval()
        patches = torch.randn(2, 1, 32, 32)
        torch.testing.assert_close(_InferenceExportGraph(backbone, native_binary=False)(patches), backbone(patches))
        residual = build_binary_descriptor_model(backbone, {"hash_bits": 256}).eval()
        torch.testing.assert_close(_InferenceExportGraph(residual, native_binary=True)(patches), backbone.forward_features(patches).q)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"invalid.pt"
            payload = {**binary_model_metadata(residual), "model": residual.state_dict()}
            payload["model_architecture"] = "unknown_binary"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "Unsupported binary checkpoint architecture"):
                HardNetDescriptor({"model": {"checkpoint": str(path), "device": "cpu"}})


if __name__ == "__main__":
    unittest.main()
