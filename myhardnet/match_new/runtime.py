"""HardNet 工程匹配模块使用的生产运行时。

本模块只提供注册和在线解锁都需要的底层能力：
    1. 读取原始灰度指纹图像；
    2. 创建 SIFT 检测器并提取关键点；
    3. 按训练阶段相同的方向裁剪并归一化 patch；
    4. 加载 HardNet checkpoint，并在 CPU 或 CUDA 上批量生成描述子。

这里不包含阈值扫描、ROC/FAR/FRR 计算或算法对照实验。工程匹配代码因此不再
依赖 ``val_new`` 验证模块，两个模块可以独立修改、测试和部署。
"""

from __future__ import annotations

import warnings
import math
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
)
from hardnet_train.model import (
    build_descriptor_model,
    checkpoint_descriptor_dim,
    checkpoint_model_architecture,
    checkpoint_model_config,
    normalize_model_architecture,
)
from match_new.descriptor_contract import (
    DEFAULT_BINARY_BITORDER,
    FLOAT32_STORAGE,
    FLOAT_DESCRIPTOR_KIND,
    L2_DISTANCE_METRIC,
)
from match_new.utils import resolve_path


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


def detect_sift_keypoints(
    image: np.ndarray,
    sift: cv2.SIFT,
    config: Mapping[str, Any],
) -> list[cv2.KeyPoint]:
    """只检测 SIFT 关键点，不计算工程匹配不需要的 SIFT 描述子。"""

    processed = preprocess_for_sift(image, config)
    return list(sift.detect(processed, None) or [])


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
    """按关键点方向旋转原图，再裁剪与训练阶段一致的局部 patch。

    当前保留已有模型使用的旧裁剪语义。后续若切换成直接局部仿射采样，必须让
    训练数据生成和本函数同时切换，并重新训练模型、标定阈值。
    """

    x, y = keypoint.pt
    angle = float(keypoint.angle if keypoint.angle >= 0 else 0.0)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        (float(x), float(y)),
        angle,
        1.0,
    )
    rotated = cv2.warpAffine(
        image,
        matrix,
        dsize=(width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    patch = cv2.getRectSubPix(
        rotated,
        patchSize=(int(crop_size), int(crop_size)),
        center=(float(x), float(y)),
    )
    return cv2.resize(
        patch,
        (int(out_size), int(out_size)),
        interpolation=cv2.INTER_AREA,
    )


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


def extract_aligned_patches_batch(
    image: np.ndarray,
    keypoints: list[cv2.KeyPoint],
    crop_size: int,
    out_size: int,
    device: torch.device | None = None,
) -> np.ndarray:
    """GPU 批量旋转裁切：一次 ``grid_sample`` 完成全部 patch 的方向对齐。

    策略：
        1. 从原图按关键点坐标直接裁取 padded patch（无旋转，比最终 patch 大一圈）；
        2. 为每个关键点构造 2×3 仿射矩阵（旋转 + 缩放）；
        3. 一次 ``affine_grid`` + ``grid_sample`` 批量完成旋转；
        4. 输出形状 ``[N, out_size, out_size]`` 的 float32 数组，像素值域 0–255。

    GPU 不可用或 ``grid_sample`` 失败时，自动回退到逐点
    ``extract_aligned_patch``（保留旧语义）。

    ``crop_size`` 仅用于确定 padded patch 尺寸以保证旋转后不越界，
    实际输出尺寸由 ``out_size`` 决定。
    """

    if not keypoints:
        return np.zeros((0, out_size, out_size), dtype=np.float32)

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # padded patch 需要容纳旋转后的 out_size×out_size 区域。
    # 极端情况下旋转 45° 后的轴对齐边界框宽度为 out_size·√2，
    # 因此 pad ≥ out_size·√2 ≈ out_size×1.414；ceil(×1.5) 留有安全余量。
    pad_size = int(math.ceil(out_size * 1.5))

    # ── 阶段 1：从原图裁取 N 个 padded patch（无旋转，纯像素索引）──
    # 用 numpy 零填充处理越界区域，与 OpenCV warpAffine 的
    # BORDER_CONSTANT + borderValue=0 语义一致。
    half_pad = pad_size / 2.0
    height, width = image.shape[:2]
    patches_host: list[np.ndarray] = []
    for kp in keypoints:
        x, y = kp.pt
        x0 = int(math.floor(x - half_pad))
        y0 = int(math.floor(y - half_pad))
        x1 = x0 + pad_size
        y1 = y0 + pad_size
        patch = np.zeros((pad_size, pad_size), dtype=np.float32)
        # 原图与 padded patch 的交集区域
        ix0 = max(0, x0)
        iy0 = max(0, y0)
        ix1 = min(width, x1)
        iy1 = min(height, y1)
        if ix0 < ix1 and iy0 < iy1:
            px0 = ix0 - x0
            py0 = iy0 - y0
            patch[py0 : py0 + (iy1 - iy0), px0 : px0 + (ix1 - ix0)] = (
                image[iy0:iy1, ix0:ix1].astype(np.float32)
            )
        patches_host.append(patch)

    # ── 阶段 2：构造仿射矩阵 batch ──
    # 每个关键点的 2×3 矩阵：把输出像素的相对位移旋转 -θ 后定位到 padded
    # patch 中的对应采样坐标。数学上与 OpenCV 整图 warpAffine +
    # getRectSubPix + resize 等价（双线性插值，align_corners=False）。
    #
    # warpAffine 默认无 WARP_INVERSE_MAP，使用正向矩阵（src→dest），
    # 内部求逆后等效 dest→src 映射为 R(θ)·dst_displacement（CCW 旋转）。
    # 代入 grid_sample 归一化公式得到：
    #     A = [[s·cosθ,  -s·sinθ,  1/pad_size],
    #          [s·sinθ,   s·cosθ,  1/pad_size]]
    # 其中 s = crop_size / pad_size。剩余微小差异（< ~0.02
    # 在 0-255 尺度下 < 5 像素值）源于 grid_sample bilinear 与
    # OpenCV bilinear + INTER_AREA resize 的插值细节不同。
    scale = crop_size / pad_size
    translation = 1.0 / pad_size
    thetas: list[torch.Tensor] = []
    for kp in keypoints:
        angle_rad = math.radians(
            float(kp.angle) if kp.angle >= 0 else 0.0
        )
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        thetas.append(
            torch.tensor(
                [
                    [scale * cos_a, -scale * sin_a, translation],
                    [scale * sin_a,  scale * cos_a, translation],
                ],
                dtype=torch.float32,
                device=device,
            )
        )
    theta_batch = torch.stack(thetas)  # [N, 2, 3]

    # ── 阶段 3：GPU 批量旋转 ──
    try:
        patches_t = (
            torch.from_numpy(np.stack(patches_host))
            .unsqueeze(1)
            .to(device, non_blocking=False)
        )
        grid = torch.nn.functional.affine_grid(
            theta_batch,
            torch.Size((len(keypoints), 1, out_size, out_size)),
            align_corners=False,
        )
        rotated = torch.nn.functional.grid_sample(
            patches_t,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return rotated.squeeze(1).cpu().numpy().astype(np.float32)
    except RuntimeError:
        pass

    # ── 阶段 4：回退到逐点 OpenCV 路径 ──
    result = np.empty(
        (len(keypoints), out_size, out_size), dtype=np.float32
    )
    for idx, kp in enumerate(keypoints):
        result[idx] = extract_aligned_patch(
            image, kp, crop_size=crop_size, out_size=out_size,
        ).astype(np.float32)
    return result


def patchable_keypoints(
    image: np.ndarray,
    keypoints: list[cv2.KeyPoint],
    config: Mapping[str, Any],
) -> tuple[list[cv2.KeyPoint], np.ndarray, list[int]]:
    """过滤严重越界的关键点，并构建与关键点行号对齐的 patch 数组。

    当 ``patch.batch_rotate=true``（默认）时，使用 GPU 批量旋转；
    否则使用逐点 OpenCV 整图旋转。标准化始终批量完成。
    """

    crop_size = int(
        get_nested(config, "patch", "crop_size", default=64)
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
    batch_rotate = bool(
        get_nested(config, "patch", "batch_rotate", default=True)
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

    # ── 旋转裁切：批量 GPU 或逐点 OpenCV ──
    if batch_rotate:
        try:
            patches_raw = extract_aligned_patches_batch(
                image,
                selected_keypoints,
                crop_size=crop_size,
                out_size=out_size,
            )
        except RuntimeError:
            batch_rotate = False  # 回退到逐点路径

    if not batch_rotate:
        patches_list = [
            extract_aligned_patch(
                image, kp, crop_size=crop_size, out_size=out_size,
            ).astype(np.float32)
            for kp in selected_keypoints
        ]
        patches_raw = np.stack(patches_list)

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
    """工程注册和在线解锁共用的 HardNet 批量推理器。

    CUDA 模式支持 FP16/BF16 autocast、channels-last、锁页内存异步传输和
    cuDNN benchmark。所有 batch 在 GPU 上拼接后只回传一次，减少同步次数。
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
        self.device = choose_device(
            str(get_nested(config, "model", "device", default="auto")),
            fallback_to_cpu=self.fallback_to_cpu,
        )
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
        self.channels_last = (
            self.device.type == "cuda"
            and bool(
                get_nested(config, "model", "channels_last", default=True)
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
        if requested_kind not in {"", "auto", descriptor_kind}:
            raise ValueError(
                "Configured descriptor kind does not match checkpoint: "
                f"config={requested_kind}, checkpoint={descriptor_kind}."
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

        self.descriptor_kind = descriptor_kind
        self.descriptor_metric = descriptor_metric
        self.descriptor_dim = saved_descriptor_dim
        self.descriptor_columns = (
            (self.descriptor_dim + 7) // 8
            if self.descriptor_kind == BINARY_DESCRIPTOR_KIND
            else self.descriptor_dim
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

    def _activate_cpu_fallback(self) -> None:
        """把模型和推理选项切换为 CPU FP32。"""

        self.device = torch.device("cpu")
        self.pin_memory = False
        self.channels_last = False
        self.amp_enabled = False
        self.amp_dtype = None
        self.inference_precision = "fp32"
        self.model.to(
            device=self.device,
            memory_format=torch.contiguous_format,
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

    def warmup(self, runs: int) -> None:
        """按真实推理 batch 形状预热模型，耗时由调用方单独统计。"""

        warmup_runs = max(0, int(runs))
        if warmup_runs == 0:
            return
        warmup_batch_size = self.fixed_inference_batch_size or 1
        patches = np.zeros(
            (warmup_batch_size, self.patch_size, self.patch_size),
            dtype=np.float32,
        )
        for _ in range(warmup_runs):
            self.describe(patches)

    def _describe_impl(self, patches: np.ndarray) -> np.ndarray:
        """在当前设备上执行描述子推理，并隐藏固定 batch 的 padding。"""

        if len(patches) == 0:
            dtype = (
                np.uint8
                if self.descriptor_kind == BINARY_DESCRIPTOR_KIND
                else np.float32
            )
            return np.zeros((0, self.descriptor_columns), dtype=dtype)
        patch_array = np.ascontiguousarray(
            patches,
            dtype=np.float32,
        )
        if patch_array.ndim == 3:
            patch_array = patch_array[:, None, :, :]
        host_tensor = torch.from_numpy(patch_array)

        inference_batch_size = (
            self.fixed_inference_batch_size or self.batch_size
        )
        outputs: list[torch.Tensor] = []
        for start in range(0, len(patch_array), inference_batch_size):
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
            if self.channels_last:
                batch = batch.contiguous(
                    memory_format=torch.channels_last
                )
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_enabled,
            ):
                if self.descriptor_kind == BINARY_DESCRIPTOR_KIND:
                    batch_output = self.model.encode_binary(batch, packed=True)
                else:
                    batch_output = self.model(batch)
            outputs.append(batch_output[:valid_count])
        if not outputs:
            dtype = (
                np.uint8
                if self.descriptor_kind == BINARY_DESCRIPTOR_KIND
                else np.float32
            )
            return np.zeros((0, self.descriptor_columns), dtype=dtype)
        output = torch.cat(outputs, dim=0).cpu()
        descriptors = (
            output.to(dtype=torch.uint8).numpy()
            if self.descriptor_kind == BINARY_DESCRIPTOR_KIND
            else output.float().numpy()
        )
        if descriptors.ndim != 2 or descriptors.shape[1] != self.descriptor_columns:
            raise ValueError(
                "Descriptor model output shape mismatch: "
                f"expected [N,{self.descriptor_columns}], got {descriptors.shape}."
            )
        return descriptors
