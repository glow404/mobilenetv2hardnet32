"""Optional ONNX Runtime/OpenVINO FP32 CPU inference adapters.

The exported graph only produces the continuous HardNet output (or the raw
pre-hash backbone feature for a native binary checkpoint). Existing NumPy/Torch
postprocessing remains in ``runtime.py`` so normalization, Hadamard/sign rules,
zero handling, and packed bit order stay centralized.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class _InferenceExportGraph(nn.Module):
    """Expose the same continuous tensor consumed by runtime postprocessing."""

    def __init__(self, model: nn.Module, *, native_binary: bool) -> None:
        super().__init__()
        self.model = model
        self.native_binary = bool(native_binary)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if self.native_binary:
            return self.model.backbone.forward_features(patches).q
        return self.model(patches)


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_key(metadata: dict[str, Any]) -> str:
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp_path, path)


def ensure_onnx_model(
    model: nn.Module,
    *,
    checkpoint_path: str | Path,
    native_binary: bool,
    patch_size: int,
    fused_conv_bn_pairs: int,
    configured_path: str | Path | None = None,
    opset_version: int = 17,
) -> Path:
    """Export/cache the inference graph, invalidating it when model inputs change."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    metadata = {
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "native_binary": bool(native_binary),
        "patch_size": int(patch_size),
        "fused_conv_bn_pairs": int(fused_conv_bn_pairs),
        "opset_version": int(opset_version),
        "torch_version": str(torch.__version__),
        "graph_kind": type(model).__name__,
    }
    if configured_path:
        onnx_path = Path(configured_path).expanduser().resolve()
    else:
        cache_root = Path(tempfile.gettempdir()) / "hardnet32_onnx_cache"
        onnx_path = cache_root / f"hardnet_{_cache_key(metadata)}.onnx"
    manifest_path = onnx_path.with_suffix(onnx_path.suffix + ".json")
    try:
        cached_metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        cached_metadata = None
    if onnx_path.is_file() and cached_metadata == metadata:
        return onnx_path

    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "ONNX export requires the optional 'onnx' package. Install it in the "
            "active environment with: python -m pip install onnx"
        ) from exc

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = onnx_path.with_name(
        f"{onnx_path.stem}.{os.getpid()}.tmp.onnx"
    )
    sample = torch.zeros((1, 1, patch_size, patch_size), dtype=torch.float32)
    export_graph = _InferenceExportGraph(model.eval(), native_binary=native_binary).eval()
    try:
        with torch.inference_mode():
            torch.onnx.export(
                export_graph,
                sample,
                str(temp_path),
                input_names=["patches"],
                output_names=["features"],
                dynamic_axes={"patches": {0: "batch"}, "features": {0: "batch"}},
                opset_version=int(opset_version),
                do_constant_folding=True,
                export_params=True,
                dynamo=False,
            )
        graph = onnx.load(str(temp_path))
        onnx.checker.check_model(graph)
        os.replace(temp_path, onnx_path)
        _write_json_atomic(manifest_path, metadata)
    finally:
        temp_path.unlink(missing_ok=True)
    return onnx_path


class OnnxRuntimeCPUBackend:
    """Run an exported FP32 graph through ONNX Runtime's CPU provider."""

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        intra_op_num_threads: int = 0,
        inter_op_num_threads: int = 1,
        graph_optimization_level: str = "all",
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "ONNX Runtime backend requires 'onnxruntime'. Install it with: "
                "python -m pip install onnxruntime"
            ) from exc
        if "CPUExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable.")
        options = ort.SessionOptions()
        if int(intra_op_num_threads) > 0:
            options.intra_op_num_threads = int(intra_op_num_threads)
        if int(inter_op_num_threads) > 0:
            options.inter_op_num_threads = int(inter_op_num_threads)
        levels = {
            "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        level_name = str(graph_optimization_level).strip().lower()
        if level_name not in levels:
            raise ValueError(
                "model.backend_graph_optimization must be one of "
                "disabled/basic/extended/all."
            )
        options.graph_optimization_level = levels[level_name]
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @staticmethod
    def prepare_input(batch: torch.Tensor) -> np.ndarray:
        return np.ascontiguousarray(batch.detach().numpy(), dtype=np.float32)

    def __call__(self, batch: np.ndarray) -> torch.Tensor:
        output = self.session.run([self.output_name], {self.input_name: batch})[0]
        return torch.from_numpy(np.asarray(output, dtype=np.float32))


class OpenVINOCPUBackend:
    """Compile and execute the same ONNX FP32 graph on OpenVINO CPU."""

    def __init__(
        self,
        onnx_path: str | Path,
        *,
        num_threads: int = 0,
    ) -> None:
        try:
            import openvino as ov
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "OpenVINO backend requires 'openvino'. Install it with: "
                "python -m pip install openvino"
            ) from exc
        core = ov.Core()
        network = core.read_model(str(onnx_path))
        compile_config: dict[Any, Any] = {}
        if int(num_threads) > 0:
            try:
                compile_config[ov.properties.inference_num_threads()] = int(num_threads)
            except AttributeError:  # pragma: no cover - older OpenVINO API
                compile_config["INFERENCE_NUM_THREADS"] = int(num_threads)
        try:
            compile_config[ov.properties.hint.inference_precision()] = ov.Type.f32
        except AttributeError:  # pragma: no cover - older OpenVINO API
            compile_config["INFERENCE_PRECISION_HINT"] = "f32"
        self.compiled_model = core.compile_model(network, "CPU", compile_config)
        self.request = self.compiled_model.create_infer_request()
        self.input_port = self.compiled_model.input(0)
        self.output_port = self.compiled_model.output(0)

    @staticmethod
    def prepare_input(batch: torch.Tensor) -> np.ndarray:
        return np.ascontiguousarray(batch.detach().numpy(), dtype=np.float32)

    def __call__(self, batch: np.ndarray) -> torch.Tensor:
        result = self.request.infer({self.input_port: batch})
        output = result[self.output_port]
        return torch.from_numpy(np.asarray(output, dtype=np.float32).copy())


def create_cpu_backend(
    backend_name: str,
    onnx_path: str | Path,
    config: dict[str, Any],
) -> OnnxRuntimeCPUBackend | OpenVINOCPUBackend:
    """Create a configured optional backend without importing it for PyTorch runs."""

    model_config = config.get("model", {})
    model_config = model_config if isinstance(model_config, dict) else {}
    if backend_name == "onnxruntime":
        return OnnxRuntimeCPUBackend(
            onnx_path,
            intra_op_num_threads=int(model_config.get("backend_num_threads", 0)),
            inter_op_num_threads=int(model_config.get("backend_inter_op_num_threads", 1)),
            graph_optimization_level=str(
                model_config.get("backend_graph_optimization", "all")
            ),
        )
    if backend_name == "openvino":
        return OpenVINOCPUBackend(
            onnx_path,
            num_threads=int(model_config.get("backend_num_threads", 0)),
        )
    raise ValueError(f"Unsupported CPU inference backend: {backend_name!r}")
