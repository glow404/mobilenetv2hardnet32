"""HardNet 工程匹配模块使用的生产运行时。

本模块只提供离线模板构建和匹配评估共用的底层能力：
    1. 读取原始灰度指纹图像；
    2. 创建 SIFT 检测器并提取关键点；
    3. 按训练阶段相同的方向裁剪并归一化 patch；
    4. 加载 HardNet checkpoint，并在 CPU 或 CUDA 上批量生成描述子。

这里不包含阈值扫描、ROC/FAR/FRR 计算或算法对照实验。工程匹配代码因此不再
依赖 ``val_new`` 验证模块，两个模块可以独立修改、测试和部署。
"""

from __future__ import annotations

import math
import time
import warnings
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch

from hardnet_train.binary_model import (
    BINARY_DESCRIPTOR_KIND,
    BINARY_STORAGE,
    HAMMING_DISTANCE_METRIC,
    build_binary_descriptor_model,
    pack_binary_codes,
)
from hardnet_train.model import (
    build_descriptor_model,
    checkpoint_descriptor_dim,
    checkpoint_model_architecture,
    checkpoint_model_config,
    normalize_model_architecture,
)
from match_new.cpu_inference_backends import (
    create_cpu_backend,
    ensure_onnx_model,
)
from match_new.descriptor_contract import (
    DEFAULT_BINARY_BITORDER,
    FLOAT32_STORAGE,
    FLOAT_DESCRIPTOR_KIND,
    L2_DISTANCE_METRIC,
)
from match_new.hadamard_binarization import (
    HadamardBinarizer,
    hadamard_binarization_enabled,
)
from match_new.utils import resolve_path


HARDNET_INFERENCE_STAGE_FIELDS = (
    "hardnet_input_prepare_ms",
    "hardnet_batch_prepare_ms",
    "hardnet_layout_conversion_ms",
    "hardnet_cnn_ms",
    "hardnet_hash_head_ms",
    "hardnet_output_transfer_ms",
    "hardnet_binarization_ms",
    "hardnet_inference_overhead_ms",
)


def fuse_inference_conv_bn(model: torch.nn.Module) -> int:
    """原地融合 eval 模型中 Sequential 内直接相邻的 Conv2d–BatchNorm2d。

    仅在推理实例加载原 checkpoint 后使用。不会跨残差分支、激活或其他模块
    融合，也不折叠 GroupNorm/LayerNorm。返回融合对数；重复调用无副作用。
    """

    if any(module.training for module in model.modules()):
        raise ValueError("Conv-BN inference fusion requires all modules in eval mode.")
    count = 0
    for child in model.children():
        count += fuse_inference_conv_bn(child)
    if isinstance(model, torch.nn.Sequential):
        names = list(model._modules)
        for left_name, right_name in zip(names, names[1:]):
            conv = model._modules[left_name]
            bn = model._modules[right_name]
            if (
                isinstance(conv, torch.nn.Conv2d)
                and isinstance(bn, torch.nn.BatchNorm2d)
                and bn.running_mean is not None
                and bn.running_var is not None
            ):
                model._modules[left_name] = torch.nn.utils.fusion.fuse_conv_bn_eval(
                    conv, bn
                )
                model._modules[right_name] = torch.nn.Identity().eval()
                count += 1
    return count


def get_nested(
    mapping: Mapping[str, Any],
    *keys: str,
    default: Any = None,
) -> Any:
    """安全读取多层配置；任意一级不存在时返回默认值。"""

    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def choose_device(
    raw_device: str,
    *,
    fallback_to_cpu: bool = True,
) -> torch.device:
    """解析匹配设备，并在 CUDA 不可用时按配置回退到 CPU。

    ``auto`` 始终优先选择 CUDA，没有可用 CUDA 时选择 CPU。即使显式配置
    ``cuda``，默认也会回退 CPU 并发出警告；只有关闭 ``fallback_to_cpu`` 后
    才会将 CUDA 不可用视为错误。
    """

    requested = str(raw_device).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if fallback_to_cpu:
            warnings.warn(
                "CUDA 不可用，HardNet 匹配将自动回退到 CPU。",
                RuntimeWarning,
                stacklevel=2,
            )
            return torch.device("cpu")
        raise RuntimeError(
            "工程匹配配置要求 CUDA，但当前 PyTorch 是 CPU 版本或 CUDA 初始化失败。"
            "请安装 CUDA 版 PyTorch，并确认 torch.cuda.is_available() 返回 True；"
            "如需显式使用 CPU，请设置 model.device=cpu。"
        )
    return torch.device(requested)


def imread_grayscale(path: str | Path) -> np.ndarray | None:
    """读取灰度图像，并兼容 Windows 中文路径。"""

    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def build_sift(config: Mapping[str, Any]) -> cv2.SIFT:
    """根据工程匹配配置创建 OpenCV SIFT 关键点检测器。"""

    return cv2.SIFT_create(
        nfeatures=int(get_nested(config, "sift", "nfeatures", default=300)),
        nOctaveLayers=int(
            get_nested(config, "sift", "nOctaveLayers", default=4)
        ),
        contrastThreshold=float(
            get_nested(config, "sift", "contrastThreshold", default=0.03)
        ),
        edgeThreshold=float(
            get_nested(config, "sift", "edgeThreshold", default=17.5)
        ),
        sigma=float(get_nested(config, "sift", "sigma", default=1.70)),
    )


def preprocess_for_sift(
    image: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    """按照工程配置执行 SIFT 检测前的可选增强和降噪。"""

    output = image
    if bool(get_nested(config, "sift", "enable_clahe", default=True)):
        tile = get_nested(config, "sift", "clahe_tile", default=[8, 8])
        clahe = cv2.createCLAHE(
            clipLimit=float(
                get_nested(config, "sift", "clahe_clip", default=4.0)
            ),
            tileGridSize=(int(tile[0]), int(tile[1])),
        )
        output = clahe.apply(output)
    if bool(get_nested(config, "sift", "enable_blur", default=True)):
        output = cv2.GaussianBlur(
            output,
            (0, 0),
            sigmaX=float(
                get_nested(config, "sift", "blur_sigma", default=0.8)
            ),
        )
    return output


def filter_keypoints_by_mask(
    keypoints: list[cv2.KeyPoint], mask: np.ndarray
) -> list[cv2.KeyPoint]:
    """以关键点中心的最近像素检查前景，拒绝越界或非指纹区域的点。"""

    height, width = mask.shape
    kept: list[cv2.KeyPoint] = []
    for keypoint in keypoints:
        x, y = keypoint.pt
        if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < width and 0 <= y < height):
            continue
        ix, iy = int(math.floor(x + 0.5)), int(math.floor(y + 0.5))
        if ix < width and iy < height and mask[iy, ix] > 0:
            kept.append(keypoint)
    return kept


def detect_sift_keypoints(
    image: np.ndarray,
    sift: cv2.SIFT,
    config: Mapping[str, Any],
    mask: np.ndarray | None = None,
) -> list[cv2.KeyPoint]:
    """只检测 SIFT 关键点；可用同尺寸前端 mask 限制检测中心区域。"""

    if mask is not None and (mask.shape != image.shape or mask.dtype != np.uint8):
        raise ValueError("SIFT mask 必须为与原图同尺寸的 uint8 二维掩码")
    processed = preprocess_for_sift(image, config)
    keypoints = list(sift.detect(processed, mask) or [])
    # 明确复核亚像素中心；也保证其它检测器实现不会把背景点送入 HardNet。
    return filter_keypoints_by_mask(keypoints, mask) if mask is not None else keypoints


def overlap_ratio(
    image_shape: tuple[int, int],
    x: float,
    y: float,
    crop_size: int,
) -> float:
    """计算未旋转裁剪窗口落在原图内部的面积比例。"""

    height, width = image_shape
    half = crop_size / 2.0
    x0, y0, x1, y1 = x - half, y - half, x + half, y + half
    ix0, iy0 = max(0.0, x0), max(0.0, y0)
    ix1, iy1 = min(float(width), x1), min(float(height), y1)
    intersection = (
        max(0.0, ix1 - ix0)
        * max(0.0, iy1 - iy0)
    )
    return intersection / max(float(crop_size * crop_size), 1.0)


def extract_aligned_patch(
    image: np.ndarray,
    keypoint: cv2.KeyPoint,
    crop_size: int,
    out_size: int,
) -> np.ndarray:
    """按关键点位置和方向直接从原图采样局部 patch。

    只为输出 patch 构造反向坐标映射，不再为每个关键点旋转整张图。非整数
    坐标使用双线性插值，越界位置补零。
    """

    x, y = keypoint.pt
    angle = float(keypoint.angle if keypoint.angle >= 0 else 0.0)
    offset_x, offset_y = _centered_patch_grid(int(crop_size))
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    map_x = (
        float(x) + cos_a * offset_x - sin_a * offset_y
    ).astype(np.float32)
    map_y = (
        float(y) + sin_a * offset_x + cos_a * offset_y
    ).astype(np.float32)
    patch = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return cv2.resize(
        patch,
        (int(out_size), int(out_size)),
        interpolation=cv2.INTER_AREA,
    )


@lru_cache(maxsize=8)
def _centered_patch_grid(crop_size: int) -> tuple[np.ndarray, np.ndarray]:
    """缓存局部 patch 相对中心的坐标网格。"""

    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    offsets = (
        np.arange(crop_size, dtype=np.float32)
        - (float(crop_size) - 1.0) / 2.0
    )
    return np.meshgrid(offsets, offsets)


def normalize_patch(
    patch: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """把灰度 patch 转为 HardNet 输入，并执行单 patch 标准化。"""

    values = patch.astype(np.float32)
    if not normalize:
        return values / 255.0
    std = max(float(values.std()), 1e-6)
    return (values - float(values.mean())) / std


def patchable_keypoints(
    image: np.ndarray,
    keypoints: list[cv2.KeyPoint],
    config: Mapping[str, Any],
) -> tuple[list[cv2.KeyPoint], np.ndarray, list[int]]:
    """过滤严重越界的关键点，并用局部反向坐标采样构建对齐的 patch 数组。

    标准化在全部 patch 上批量完成。
    """

    crop_size = int(
        get_nested(config, "patch", "crop_size", default=32)
    )
    out_size = int(
        get_nested(config, "patch", "out_size", default=32)
    )
    min_overlap = float(
        get_nested(config, "patch", "min_overlap_ratio", default=0.55)
    )
    normalize_flag = bool(
        get_nested(config, "patch", "normalize", default=True)
    )

    # ── 过滤越界关键点 ──
    selected_keypoints: list[cv2.KeyPoint] = []
    selected_indices: list[int] = []
    for index, keypoint in enumerate(keypoints):
        x, y = keypoint.pt
        if (
            overlap_ratio(
                image.shape[:2],
                float(x),
                float(y),
                crop_size,
            )
            < min_overlap
        ):
            continue
        selected_keypoints.append(keypoint)
        selected_indices.append(index)

    if not selected_keypoints:
        return (
            [],
            np.zeros((0, out_size, out_size), dtype=np.float32),
            [],
        )

    patches_raw = np.stack(
        [
            extract_aligned_patch(
                image, kp, crop_size=crop_size, out_size=out_size,
            ).astype(np.float32)
            for kp in selected_keypoints
        ]
    )

    # ── 批量标准化 ──
    if normalize_flag:
        mean = patches_raw.mean(axis=(1, 2), keepdims=True)
        std = np.maximum(patches_raw.std(axis=(1, 2), keepdims=True), 1e-6)
        patches = (patches_raw - mean) / std
    else:
        patches = patches_raw / 255.0

    return (
        selected_keypoints,
        patches.astype(np.float32),
        selected_indices,
    )


class HardNetDescriptor:
    """HardNet 批量推理器，供离线模板构建和评估流程共用。

    CUDA 模式支持 FP16/BF16 autocast、channels-last、锁页内存异步传输和
    cuDNN benchmark；PyTorch CPU 模式支持 FP32、oneDNN 和 channels-last。
    CPU 也可选择 ONNX Runtime/OpenVINO，统一共享后处理和输出契约。
    所有 batch 在设备侧拼接后只回传一次，减少同步次数。
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.fallback_to_cpu = bool(
            get_nested(
                config,
                "model",
                "fallback_to_cpu",
                default=True,
            )
        )
        self.inference_backend = str(
            get_nested(config, "model", "inference_backend", default="pytorch")
        ).strip().lower()
        if self.inference_backend not in {"pytorch", "onnxruntime", "openvino"}:
            raise ValueError(
                "model.inference_backend must be pytorch, onnxruntime, or openvino; "
                f"got {self.inference_backend!r}."
            )
        requested_device = str(
            get_nested(
                config,
                "model",
                "device",
                default=("cpu" if self.inference_backend != "pytorch" else "auto"),
            )
        )
        self.device = choose_device(
            requested_device,
            fallback_to_cpu=self.fallback_to_cpu,
        )
        if self.inference_backend != "pytorch" and self.device.type != "cpu":
            raise ValueError(
                f"model.inference_backend={self.inference_backend} requires model.device=cpu."
            )
        self.cpu_num_threads = int(
            get_nested(config, "model", "cpu_num_threads", default=0) or 0
        )
        if self.cpu_num_threads < 0:
            raise ValueError("model.cpu_num_threads cannot be negative.")
        if self.device.type == "cpu" and self.cpu_num_threads > 0:
            torch.set_num_threads(self.cpu_num_threads)
        self.mkldnn_requested = bool(
            get_nested(config, "model", "mkldnn_enabled", default=True)
        )
        self.mkldnn_enabled = bool(
            self.device.type == "cpu"
            and self.inference_backend == "pytorch"
            and self.mkldnn_requested
            and torch.backends.mkldnn.is_available()
        )
        self.cpu_backend = None
        self.batch_size = int(
            get_nested(config, "model", "batch_size", default=512)
        )
        if self.batch_size <= 0:
            raise ValueError("model.batch_size 必须大于 0")
        self.fixed_inference_batch_size = int(
            get_nested(
                config,
                "model",
                "fixed_inference_batch_size",
                default=0,
            )
        )
        if self.fixed_inference_batch_size < 0:
            raise ValueError(
                "model.fixed_inference_batch_size 不能小于 0"
            )
        self.patch_size = int(
            get_nested(config, "patch", "out_size", default=32)
        )
        if self.patch_size <= 0:
            raise ValueError("patch.out_size 必须大于 0")
        self.pin_memory = (
            self.device.type == "cuda"
            and bool(
                get_nested(config, "model", "pin_memory", default=True)
            )
        )
        self.channels_last_requested = bool(
            get_nested(config, "model", "channels_last", default=True)
        )
        self.channels_last = bool(
            self.channels_last_requested
            and (
                self.device.type == "cuda"
                or (
                    self.device.type == "cpu"
                    and self.inference_backend == "pytorch"
                )
            )
        )
        self._configure_precision(config)
        self._configure_cuda_backend(config)

        checkpoint_path = resolve_path(
            config,
            get_nested(config, "model", "checkpoint"),
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
        descriptor_kind = str(
            checkpoint.get("descriptor_kind", FLOAT_DESCRIPTOR_KIND)
        ).strip().lower()
        descriptor_metric = str(
            checkpoint.get("descriptor_metric", L2_DISTANCE_METRIC)
        ).strip().lower()
        supported = {
            (FLOAT_DESCRIPTOR_KIND, L2_DISTANCE_METRIC),
            (BINARY_DESCRIPTOR_KIND, HAMMING_DISTANCE_METRIC),
        }
        if (descriptor_kind, descriptor_metric) not in supported:
            raise ValueError(
                "Unsupported descriptor checkpoint contract: "
                f"kind={descriptor_kind}, metric={descriptor_metric}, path={checkpoint_path}."
            )
        requested_kind = str(
            get_nested(config, "model", "descriptor_kind", default="auto")
        ).strip().lower()
        output_descriptor_kind = (
            BINARY_DESCRIPTOR_KIND
            if hadamard_binarization_enabled(config)
            else descriptor_kind
        )
        if requested_kind not in {"", "auto", output_descriptor_kind}:
            raise ValueError(
                "Configured descriptor kind does not match runtime output: "
                f"config={requested_kind}, output={output_descriptor_kind}."
            )

        saved_descriptor_dim = checkpoint_descriptor_dim(checkpoint)
        requested_descriptor_dim = get_nested(
            config,
            "model",
            "descriptor_dim",
            default="auto",
        )
        requested_descriptor_dim_text = str(
            requested_descriptor_dim if requested_descriptor_dim is not None else ""
        ).strip().lower()
        if requested_descriptor_dim_text not in {"", "auto"}:
            configured_descriptor_dim = int(requested_descriptor_dim_text)
            if configured_descriptor_dim != saved_descriptor_dim:
                raise ValueError(
                    "Configured descriptor dimension does not match checkpoint: "
                    f"config={configured_descriptor_dim}, checkpoint={saved_descriptor_dim}."
                )

        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        requested_architecture = str(
            get_nested(config, "model", "architecture", default="auto")
        ).strip().lower()
        if descriptor_kind == FLOAT_DESCRIPTOR_KIND:
            saved_architecture = checkpoint_model_architecture(checkpoint)
            if requested_architecture in {"", "auto"}:
                architecture = saved_architecture
            else:
                architecture = normalize_model_architecture(requested_architecture)
                if architecture != saved_architecture:
                    raise ValueError(
                        "Configured model architecture does not match checkpoint: "
                        f"config={architecture}, checkpoint={saved_architecture}."
                    )

            saved_model_config = checkpoint_model_config(checkpoint)
            runtime_model_config = config.get("model")
            if isinstance(runtime_model_config, Mapping):
                for key in ("dropout", "final_bn_affine"):
                    if key in runtime_model_config:
                        saved_model_config[key] = runtime_model_config[key]
            saved_model_config["architecture"] = architecture
            saved_model_config["descriptor_dim"] = saved_descriptor_dim
            self.model = build_descriptor_model(saved_model_config)
            self.model.load_state_dict(state)
            self.descriptor_storage = FLOAT32_STORAGE
            self.descriptor_bitorder = ""
        else:
            binary_architecture = str(
                checkpoint.get("model_architecture", "residual_binary_hash_v1")
            ).strip().lower()
            if requested_architecture not in {"", "auto", binary_architecture}:
                raise ValueError(
                    "Configured model architecture does not match binary checkpoint: "
                    f"config={requested_architecture}, checkpoint={binary_architecture}."
                )
            saved_config = checkpoint.get("resolved_config", checkpoint.get("config", {}))
            saved_config = saved_config if isinstance(saved_config, Mapping) else {}
            saved_backbone = saved_config.get("backbone", {})
            saved_backbone = saved_backbone if isinstance(saved_backbone, Mapping) else {}
            backbone_architecture = str(
                checkpoint.get(
                    "backbone_architecture",
                    saved_backbone.get("architecture", "hardnet"),
                )
            )
            float_descriptor_dim = int(
                checkpoint.get(
                    "float_descriptor_dim",
                    saved_backbone.get("descriptor_dim", 128),
                )
            )
            hidden_dim = int(
                checkpoint.get("hash_head_hidden_dim", saved_descriptor_dim * 2)
            )
            binary_config = {
                "hash_bits": saved_descriptor_dim,
                "hidden_multiplier": hidden_dim / saved_descriptor_dim,
                "dropout": float(checkpoint.get("hash_head_dropout", 0.1)),
                "temperature_start": float(
                    checkpoint.get("quantization_temperature", 1.0)
                ),
                "backbone_trainable": bool(
                    checkpoint.get("backbone_trainable", False)
                ),
                "bitorder": str(
                    checkpoint.get("binary_bitorder", DEFAULT_BINARY_BITORDER)
                ),
            }
            backbone_config = {
                "architecture": backbone_architecture,
                "descriptor_dim": float_descriptor_dim,
                "dropout": float(saved_backbone.get("dropout", 0.1)),
            }
            load_error: RuntimeError | None = None
            self.model = None
            for final_bn_affine in (False, True):
                backbone_config["final_bn_affine"] = final_bn_affine
                candidate = build_binary_descriptor_model(
                    build_descriptor_model(backbone_config),
                    binary_config,
                )
                try:
                    candidate.load_state_dict(state)
                except RuntimeError as exc:
                    load_error = exc
                    continue
                self.model = candidate
                break
            if self.model is None:
                raise ValueError(
                    f"Binary checkpoint model structure is incompatible: {checkpoint_path}"
                ) from load_error
            self.descriptor_storage = str(
                checkpoint.get("binary_storage", BINARY_STORAGE)
            ).strip().lower()
            self.descriptor_bitorder = str(
                checkpoint.get("binary_bitorder", DEFAULT_BINARY_BITORDER)
            ).strip().lower()
            requested_storage = str(
                get_nested(
                    config,
                    "model",
                    "binary_storage",
                    default=self.descriptor_storage,
                )
            ).strip().lower()
            requested_bitorder = str(
                get_nested(
                    config,
                    "model",
                    "binary_bitorder",
                    default="auto",
                )
            ).strip().lower()
            if requested_storage not in {"", "auto", self.descriptor_storage}:
                raise ValueError(
                    "Configured binary storage does not match checkpoint: "
                    f"config={requested_storage}, checkpoint={self.descriptor_storage}."
                )
            if requested_bitorder not in {"", "auto", self.descriptor_bitorder}:
                raise ValueError(
                    "Configured binary bitorder does not match checkpoint: "
                    f"config={requested_bitorder}, checkpoint={self.descriptor_bitorder}."
                )

        self.model_descriptor_kind = descriptor_kind
        self.hadamard_binarizer: HadamardBinarizer | None = None
        self.descriptor_transform_metadata: dict[str, Any] = {}
        if hadamard_binarization_enabled(config):
            if descriptor_kind != FLOAT_DESCRIPTOR_KIND:
                raise ValueError(
                    "Hadamard post-binarization requires a float/L2 checkpoint; "
                    f"got kind={descriptor_kind}, metric={descriptor_metric}."
                )
            self.hadamard_binarizer = HadamardBinarizer.from_config(config)
            if saved_descriptor_dim != self.hadamard_binarizer.order:
                raise ValueError(
                    "Hadamard order must equal the float checkpoint descriptor dimension: "
                    f"order={self.hadamard_binarizer.order}, "
                    f"descriptor_dim={saved_descriptor_dim}."
                )
            descriptor_kind = BINARY_DESCRIPTOR_KIND
            descriptor_metric = HAMMING_DISTANCE_METRIC
            saved_descriptor_dim = self.hadamard_binarizer.order
            self.descriptor_storage = BINARY_STORAGE
            self.descriptor_bitorder = self.hadamard_binarizer.bitorder
            requested_storage = str(
                get_nested(config, "model", "binary_storage", default="auto")
            ).strip().lower()
            requested_bitorder = str(
                get_nested(config, "model", "binary_bitorder", default="auto")
            ).strip().lower()
            if requested_storage not in {"", "auto", self.descriptor_storage}:
                raise ValueError(
                    "Configured binary storage does not match Hadamard output: "
                    f"config={requested_storage}, output={self.descriptor_storage}."
                )
            if requested_bitorder not in {"", "auto", self.descriptor_bitorder}:
                raise ValueError(
                    "Configured binary bitorder does not match Hadamard output: "
                    f"config={requested_bitorder}, output={self.descriptor_bitorder}."
                )
            self.descriptor_transform_metadata = self.hadamard_binarizer.metadata()

        self.descriptor_kind = descriptor_kind
        self.descriptor_metric = descriptor_metric
        self.descriptor_dim = saved_descriptor_dim
        self.descriptor_transform_name = str(
            self.descriptor_transform_metadata.get("descriptor_transform_name", "")
        )
        self.descriptor_transform_id = str(
            self.descriptor_transform_metadata.get("descriptor_transform_id", "")
        )
        self.descriptor_columns = (
            (self.descriptor_dim + 7) // 8
            if self.descriptor_kind == BINARY_DESCRIPTOR_KIND
            else self.descriptor_dim
        )
        # 在 CPU 上加载原始权重后再融合；不修改训练模型或 checkpoint。
        self.model.eval()
        self.fuse_conv_bn = bool(
            get_nested(config, "model", "fuse_conv_bn", default=True)
        )
        self.fused_conv_bn_pairs = (
            fuse_inference_conv_bn(self.model) if self.fuse_conv_bn else 0
        )
        try:
            self.model.to(self.device)
            if self.channels_last:
                self.model.to(memory_format=torch.channels_last)
        except RuntimeError as exc:
            if self.device.type != "cuda" or not self.fallback_to_cpu:
                raise
            warnings.warn(
                f"HardNet 模型初始化 CUDA 失败，将回退到 CPU：{exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._activate_cpu_fallback()
        self.model.eval()
        if self.inference_backend != "pytorch":
            configured_onnx_path = get_nested(
                config, "model", "onnx_path", default=None
            )
            if configured_onnx_path:
                configured_onnx_path = resolve_path(config, configured_onnx_path)
            onnx_path = ensure_onnx_model(
                self.model,
                checkpoint_path=checkpoint_path,
                native_binary=self.model_descriptor_kind == BINARY_DESCRIPTOR_KIND,
                patch_size=self.patch_size,
                fused_conv_bn_pairs=self.fused_conv_bn_pairs,
                configured_path=configured_onnx_path,
                opset_version=int(
                    get_nested(config, "model", "onnx_opset_version", default=17)
                ),
            )
            self.backend_model_path = str(onnx_path)
            self.cpu_backend = create_cpu_backend(
                self.inference_backend,
                onnx_path,
                dict(config),
            )
        else:
            self.backend_model_path = ""

    def _activate_cpu_fallback(self) -> None:
        """把模型和推理选项切换为 CPU FP32。"""

        self.device = torch.device("cpu")
        self.pin_memory = False
        self.channels_last = bool(
            self.channels_last_requested
            and self.inference_backend == "pytorch"
        )
        self.mkldnn_enabled = bool(
            self.inference_backend == "pytorch"
            and self.mkldnn_requested
            and torch.backends.mkldnn.is_available()
        )
        if self.cpu_num_threads > 0:
            torch.set_num_threads(self.cpu_num_threads)
        self.amp_enabled = False
        self.amp_dtype = None
        self.inference_precision = "fp32"
        self.model.to(
            device=self.device,
            memory_format=(
                torch.channels_last
                if self.channels_last
                else torch.contiguous_format
            ),
        )

    def _configure_precision(self, config: Mapping[str, Any]) -> None:
        """解析工程推理精度，并配置 autocast 数据类型。"""

        precision = str(
            get_nested(
                config,
                "model",
                "inference_precision",
                default="fp32",
            )
        ).strip().lower()
        if precision in {"", "off", "none", "fp32", "float32"}:
            self.amp_enabled = False
            self.amp_dtype: torch.dtype | None = None
            self.inference_precision = "fp32"
            return
        if precision in {"fp16", "float16"}:
            self.amp_enabled = self.device.type == "cuda"
            self.amp_dtype = (
                torch.float16 if self.amp_enabled else None
            )
            self.inference_precision = (
                "fp16" if self.amp_enabled else "fp32"
            )
            return
        if precision in {"bf16", "bfloat16"}:
            if (
                self.device.type == "cuda"
                and not torch.cuda.is_bf16_supported()
            ):
                raise RuntimeError(
                    "当前 GPU/PyTorch 不支持 BF16 推理，请设置 "
                    "model.inference_precision=fp16。"
                )
            self.amp_enabled = self.device.type == "cuda"
            self.amp_dtype = (
                torch.bfloat16 if self.amp_enabled else None
            )
            self.inference_precision = (
                "bf16" if self.amp_enabled else "fp32"
            )
            return
        raise ValueError(
            f"不支持的 model.inference_precision: {precision!r}"
        )

    def _configure_cuda_backend(
        self,
        config: Mapping[str, Any],
    ) -> None:
        """配置只在 CUDA 模式生效的 cuDNN 和 TF32 选项。"""

        if self.device.type != "cuda":
            return
        torch.backends.cudnn.benchmark = bool(
            get_nested(
                config,
                "model",
                "cudnn_benchmark",
                default=True,
            )
        )
        allow_tf32 = bool(
            get_nested(
                config,
                "model",
                "allow_tf32",
                default=True,
            )
        )
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.set_float32_matmul_precision(
            "high" if allow_tf32 else "highest"
        )

    @torch.inference_mode()
    def describe(self, patches: np.ndarray) -> np.ndarray:
        """批量生成描述子；CUDA 运行失败时自动切换 CPU 并重试一次。"""

        try:
            return self._describe_impl(patches)
        except RuntimeError as exc:
            if self.device.type != "cuda" or not self.fallback_to_cpu:
                raise
            warnings.warn(
                f"HardNet CUDA 推理失败，将回退到 CPU 并重试：{exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._activate_cpu_fallback()
            return self._describe_impl(patches)

    @torch.inference_mode()
    def describe_with_timing(
        self, patches: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """描述子生成的细分墙钟计时；CPU 无设备同步，CUDA 同步后计时。

        CNN 阶段含 backbone 输出及其归一化，不含二值网络 hash head、
        Hadamard/sign/packbits 或输出转换。普通 describe() 保持不插入同步。
        CUDA 失败回退时，失败尝试的时间计入 inference_overhead，设备字段
        记录最后成功的实际设备。空输入所有计算阶段为 0。
        """

        total_started = time.perf_counter()
        timings = {field: 0.0 for field in HARDNET_INFERENCE_STAGE_FIELDS}
        try:
            descriptors = self._describe_impl(patches, timings=timings)
        except RuntimeError as exc:
            if self.device.type != "cuda" or not self.fallback_to_cpu:
                raise
            warnings.warn(
                f"HardNet CUDA 推理失败，将回退到 CPU 并重试：{exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            self._activate_cpu_fallback()
            timings = {field: 0.0 for field in HARDNET_INFERENCE_STAGE_FIELDS}
            descriptors = self._describe_impl(patches, timings=timings)
        total_ms = (time.perf_counter() - total_started) * 1000.0
        timings["hardnet_inference_overhead_ms"] = max(
            0.0, total_ms - sum(timings.values())
        )
        return descriptors, {
            **timings,
            "hardnet_inference_ms": total_ms,
            "hardnet_inference_device": str(self.device),
            "hardnet_inference_precision": self.inference_precision,
            "hardnet_inference_backend": self.inference_backend,
            "hardnet_inference_mkldnn": bool(self.mkldnn_enabled),
            "hardnet_inference_channels_last": bool(self.channels_last),
            "hardnet_inference_cpu_threads": (
                int(torch.get_num_threads()) if self.device.type == "cpu" else ""
            ),
            "hardnet_fused_conv_bn_pairs": self.fused_conv_bn_pairs,
        }

    def _describe_impl(
        self,
        patches: np.ndarray,
        *,
        timings: dict[str, float] | None = None,
    ) -> np.ndarray:
        """执行描述子推理；可选分段计时不改变普通 describe 的返回协议。"""

        def start_stage() -> float:
            if timings is None:
                return 0.0
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            return time.perf_counter()

        def end_stage(field: str, started: float) -> None:
            if timings is not None:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                timings[field] += (time.perf_counter() - started) * 1000.0

        if len(patches) == 0:
            dtype = (
                np.uint8
                if self.descriptor_kind == BINARY_DESCRIPTOR_KIND
                else np.float32
            )
            return np.zeros((0, self.descriptor_columns), dtype=dtype)
        stage_started = start_stage()
        patch_array = np.ascontiguousarray(
            patches,
            dtype=np.float32,
        )
        if patch_array.ndim == 3:
            patch_array = patch_array[:, None, :, :]
        host_tensor = torch.from_numpy(patch_array)
        end_stage("hardnet_input_prepare_ms", stage_started)

        inference_batch_size = (
            self.fixed_inference_batch_size or self.batch_size
        )
        outputs: list[torch.Tensor] = []
        for start in range(0, len(patch_array), inference_batch_size):
            stage_started = start_stage()
            source_batch = host_tensor[
                start : start + inference_batch_size
            ]
            valid_count = len(source_batch)
            if (
                self.fixed_inference_batch_size
                and valid_count < inference_batch_size
            ):
                padded_batch = torch.zeros(
                    (inference_batch_size, *source_batch.shape[1:]),
                    dtype=source_batch.dtype,
                )
                padded_batch[:valid_count].copy_(source_batch)
                source_batch = padded_batch
            if self.pin_memory:
                source_batch = source_batch.pin_memory()

            batch = source_batch.to(
                self.device,
                non_blocking=self.pin_memory,
            )
            end_stage("hardnet_batch_prepare_ms", stage_started)
            stage_started = start_stage()
            if self.channels_last:
                batch = batch.contiguous(
                    memory_format=torch.channels_last
                )
            backend_batch = (
                self.cpu_backend.prepare_input(batch)
                if self.cpu_backend is not None
                else None
            )
            end_stage("hardnet_layout_conversion_ms", stage_started)
            stage_started = start_stage()
            if self.cpu_backend is not None:
                batch_output = self.cpu_backend(backend_batch)
                end_stage("hardnet_cnn_ms", stage_started)
                if self.model_descriptor_kind == BINARY_DESCRIPTOR_KIND:
                    stage_started = start_stage()
                    logits = self.model.hash_head(batch_output)
                    end_stage("hardnet_hash_head_ms", stage_started)
                    stage_started = start_stage()
                    batch_output = pack_binary_codes(
                        (logits >= 0).to(torch.uint8), self.model.bitorder
                    )
                    end_stage("hardnet_binarization_ms", stage_started)
            else:
                mkldnn_context = (
                    torch.backends.mkldnn.flags(enabled=self.mkldnn_enabled)
                    if self.device.type == "cpu"
                    else nullcontext()
                )
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ), mkldnn_context:
                    if self.model_descriptor_kind == BINARY_DESCRIPTOR_KIND:
                        if timings is None:
                            batch_output = self.model.encode_binary(batch, packed=True)
                        else:
                            features = self.model.backbone.forward_features(batch)
                            end_stage("hardnet_cnn_ms", stage_started)
                            stage_started = start_stage()
                            logits = self.model.hash_head(features.q)
                            end_stage("hardnet_hash_head_ms", stage_started)
                            stage_started = start_stage()
                            batch_output = pack_binary_codes(
                                (logits >= 0).to(torch.uint8), self.model.bitorder
                            )
                            end_stage("hardnet_binarization_ms", stage_started)
                    else:
                        batch_output = self.model(batch)
                        end_stage("hardnet_cnn_ms", stage_started)
            outputs.append(batch_output[:valid_count])
        if not outputs:
            dtype = (
                np.uint8
                if self.descriptor_kind == BINARY_DESCRIPTOR_KIND
                else np.float32
            )
            return np.zeros((0, self.descriptor_columns), dtype=dtype)
        stage_started = start_stage()
        output = torch.cat(outputs, dim=0).cpu()
        if self.hadamard_binarizer is not None:
            float_descriptors = output.float().numpy()
            end_stage("hardnet_output_transfer_ms", stage_started)
            stage_started = start_stage()
            descriptors = self.hadamard_binarizer.binarize(
                float_descriptors,
                packed=True,
            )
            end_stage("hardnet_binarization_ms", stage_started)
        else:
            descriptors = (
                output.to(dtype=torch.uint8).numpy()
                if self.descriptor_kind == BINARY_DESCRIPTOR_KIND
                else output.float().numpy()
            )
            end_stage("hardnet_output_transfer_ms", stage_started)
        if descriptors.ndim != 2 or descriptors.shape[1] != self.descriptor_columns:
            raise ValueError(
                "Descriptor model output shape mismatch: "
                f"expected [N,{self.descriptor_columns}], got {descriptors.shape}."
            )
        return descriptors
