"""固定 PD + Sylvester Hadamard 变换的浮点描述子二值化。

该模块实现一个不含可训练参数的、可复现的二值化后处理：

``f' = P D f``
    ``P`` 是固定维度置换，``D`` 是固定的 ``+1/-1`` 对角翻转；

``z = H_n f'``
    ``H_n`` 是 Sylvester 型 Hadamard 矩阵，使用 FWHT 原地分块计算，
    不显式分配 ``n x n`` 矩阵；

``bit = sign(z)``
    零值按配置映射为 0 或 1，然后按固定 bit order 打包为 ``uint8``。

P 和 D 首次使用时由固定 seed 生成，并保存到 JSON state 文件。后续进程
优先读取 state 文件中的显式 permutation/sign 数组；如果 state 已存在但其
seed、维度或位序与当前配置不一致，会直接报错，避免训练、验证、测试、模板
库和查询阶段悄悄使用不同的变换。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


HADAMARD_ORDER = 256
HADAMARD_STATE_FORMAT_VERSION = 1
HADAMARD_TRANSFORM_NAME = "hadamard_pd_fwht_v1"


def _mapping_section(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """读取 Hadamard 配置，并兼容几个早期/简写段名。"""

    if config is None or not isinstance(config, Mapping):
        return {}

    # ``HadamardBinarizer.from_config`` 也允许直接传入配置段，便于单元测试
    # 和训练/验证代码复用；完整工程配置优先使用 canonical 段名。
    canonical_names = (
        "hadamard_binarization",
        "hadamard_binary",
        "hadamard",
    )
    for name in canonical_names:
        value = config.get(name)
        if value is not None:
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping.")
            return dict(value)

    for parent_name in ("model", "matching"):
        parent = config.get(parent_name)
        if not isinstance(parent, Mapping):
            continue
        for name in canonical_names:
            value = parent.get(name)
            if value is not None:
                if not isinstance(value, Mapping):
                    raise ValueError(f"{parent_name}.{name} must be a mapping.")
                return dict(value)

    # 直接传入 {enabled: ..., order: ...} 的配置段。
    if any(
        key in config
        for key in (
            "enabled",
            "order",
            "hadamard_order",
            "state_path",
            "seed",
        )
    ):
        return dict(config)
    return {}


def hadamard_binarization_config(
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """返回规范化前的 Hadamard 配置段。"""

    return _mapping_section(config)


def hadamard_binarization_enabled(config: Mapping[str, Any] | None) -> bool:
    """判断是否启用浮点 HardNet 的 Hadamard 二值化后处理。"""

    return bool(_mapping_section(config).get("enabled", False))


def _config_path(config: Mapping[str, Any] | None) -> Path | None:
    if not isinstance(config, Mapping):
        return None
    raw = config.get("_config_path")
    if raw in {None, ""}:
        return None
    return Path(str(raw)).expanduser().resolve()


def _resolve_relative_path(
    config: Mapping[str, Any] | None,
    raw_path: str | Path,
) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    config_path = _config_path(config)
    if config_path is not None:
        return (config_path.parent / path).resolve()
    return (Path.cwd() / path).resolve()


def _configured_value(section: Mapping[str, Any], *names: str, default: Any) -> Any:
    for name in names:
        if name not in section:
            continue
        value = section[name]
        # Lists/arrays (for explicit P/D) are valid values and cannot be
        # compared with a set containing ``None``/``""``.
        if value is None or (isinstance(value, str) and value == ""):
            continue
        return value
    return default


def resolve_hadamard_order(config: Mapping[str, Any] | None) -> int:
    """解析 Hadamard 阶数；工程默认并要求使用 256 阶。"""

    section = _mapping_section(config)
    order = int(
        _configured_value(
            section,
            "order",
            "hadamard_order",
            "dimension",
            "transform_dim",
            default=HADAMARD_ORDER,
        )
    )
    if order < 2 or order & (order - 1):
        raise ValueError(
            f"Hadamard order must be a power of two >= 2, got {order}."
        )
    return order


def resolve_hadamard_state_path(
    config: Mapping[str, Any] | None,
    *,
    order: int | None = None,
) -> Path | None:
    """解析 P/D state 文件路径。

    未显式配置时，state 放在配置文件同目录，保证同一份配置在不同 output
    目录或不同当前工作目录下仍然引用同一个 P/D。``None`` 配置只表示调用
    者希望使用内存态，工程配置默认会得到一个稳定路径。
    """

    section = _mapping_section(config)
    configured = _configured_value(
        section,
        "state_path",
        "pd_state_path",
        "transform_state_path",
        "transform_path",
        default=None,
    )
    if configured not in {None, ""}:
        return _resolve_relative_path(config, configured)
    if config is None:
        return None
    resolved_order = int(order if order is not None else resolve_hadamard_order(config))
    config_path = _config_path(config)
    if config_path is None:
        return (Path.cwd() / f"hadamard_pd_{resolved_order}.json").resolve()
    return (config_path.parent / f"hadamard_pd_{resolved_order}.json").resolve()


def _normalize_bitorder(value: Any) -> str:
    bitorder = str(value if value is not None else "little").strip().lower()
    if bitorder not in {"little", "big"}:
        raise ValueError(
            f"Hadamard binary bitorder must be 'little' or 'big', got {value!r}."
        )
    return bitorder


def _normalize_zero_is_one(section: Mapping[str, Any]) -> bool:
    value = _configured_value(
        section,
        "zero_is_one",
        "sign_zero_is_one",
        "zero_bit_is_one",
        default=True,
    )
    return bool(value)


def fwht(
    values: np.ndarray,
    *,
    axis: int = -1,
    normalize: bool = False,
) -> np.ndarray:
    """沿指定轴执行 Sylvester FWHT。

    变换采用未归一化定义：``H_2=[[1,1],[1,-1]]``，因此输出等价于
    显式 Sylvester Hadamard 矩阵左乘输入。``normalize=True`` 时额外除以
    ``sqrt(n)``；这不会改变后续 sign bit，但可用于数值分析。
    """

    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("fwht expects an array with at least one dimension.")
    axis = int(axis) % array.ndim
    n = int(array.shape[axis])
    if n < 2 or n & (n - 1):
        raise ValueError(
            f"FWHT length must be a power of two >= 2, got {n}."
        )

    # FWHT 的蝶形加减需要浮点精度；保留 float64 输入，否则统一使用 FP32。
    dtype = np.float64 if np.issubdtype(array.dtype, np.float64) else np.float32
    work = np.moveaxis(np.asarray(array, dtype=dtype), axis, -1).copy()
    width = 1
    while width < n:
        grouped = work.reshape(*work.shape[:-1], n // (2 * width), 2, width)
        left = grouped[..., 0, :].copy()
        right = grouped[..., 1, :].copy()
        grouped[..., 0, :] = left + right
        grouped[..., 1, :] = left - right
        width *= 2
    if normalize:
        work /= math.sqrt(float(n))
    return np.moveaxis(work, -1, axis)


def fwht_torch(
    values: Any,
    *,
    dim: int = -1,
    normalize: bool = False,
) -> Any:
    """Torch 版本 FWHT，保留 autograd 图，可供训练/验证辅助指标使用。"""

    # 延迟导入 torch，避免只使用 NumPy 模块时不必要地初始化 PyTorch。
    import torch

    if not torch.is_tensor(values):
        raise TypeError("fwht_torch expects a torch.Tensor.")
    if values.ndim == 0:
        raise ValueError("fwht_torch expects a tensor with at least one dimension.")
    dim = int(dim) % values.ndim
    n = int(values.shape[dim])
    if n < 2 or n & (n - 1):
        raise ValueError(
            f"FWHT length must be a power of two >= 2, got {n}."
        )
    work = values.movedim(dim, -1).clone()
    width = 1
    while width < n:
        grouped = work.reshape(*work.shape[:-1], n // (2 * width), 2, width)
        left = grouped[..., 0, :]
        right = grouped[..., 1, :]
        work = torch.stack(
            (left + right, left - right),
            dim=-2,
        ).reshape(*work.shape[:-1], n)
        width *= 2
    if normalize:
        work = work / math.sqrt(float(n))
    return work.movedim(-1, dim)


def generate_pd(
    order: int = HADAMARD_ORDER,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """用固定 seed 生成 permutation P 和 diagonal signs D。"""

    order = int(order)
    if order < 2 or order & (order - 1):
        raise ValueError(
            f"P/D order must be a power of two >= 2, got {order}."
        )
    # RandomState/MT19937 的序列在 NumPy 版本间更稳定；state 文件仍保存
    # 生成结果，因此后续运行不会依赖随机数实现细节。
    rng = np.random.RandomState(int(seed))
    permutation = np.asarray(rng.permutation(order), dtype=np.int64)
    signs = np.where(
        rng.randint(0, 2, size=order) == 0,
        -1,
        1,
    ).astype(np.int8)
    return permutation, signs


def _validate_pd(
    order: int,
    permutation: Any,
    diagonal_signs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    permutation_array = np.asarray(permutation, dtype=np.int64).reshape(-1)
    signs_array = np.asarray(diagonal_signs, dtype=np.int8).reshape(-1)
    expected = np.arange(int(order), dtype=np.int64)
    if permutation_array.shape != expected.shape or not np.array_equal(
        np.sort(permutation_array), expected
    ):
        raise ValueError(
            "Hadamard permutation must contain every index exactly once: "
            f"expected length {order}, got {permutation_array.shape}."
        )
    if signs_array.shape != expected.shape or not np.all(
        np.isin(signs_array, np.asarray([-1, 1], dtype=np.int8))
    ):
        raise ValueError(
            "Hadamard diagonal signs must have one +/-1 value per dimension: "
            f"expected length {order}, got {signs_array.shape}."
        )
    return np.ascontiguousarray(permutation_array), np.ascontiguousarray(signs_array)


def _transform_id(
    order: int,
    permutation: np.ndarray,
    diagonal_signs: np.ndarray,
    bitorder: str,
    zero_is_one: bool,
) -> str:
    digest = hashlib.sha256()
    digest.update(HADAMARD_TRANSFORM_NAME.encode("ascii"))
    digest.update(np.asarray([int(order)], dtype="<i8").tobytes())
    digest.update(np.asarray(permutation, dtype="<i8").tobytes())
    digest.update(np.asarray(diagonal_signs, dtype="i1").tobytes())
    digest.update(str(bitorder).encode("ascii"))
    digest.update(b"1" if zero_is_one else b"0")
    return f"{HADAMARD_TRANSFORM_NAME}-{digest.hexdigest()[:24]}"


def _state_payload(
    order: int,
    seed: int,
    permutation: np.ndarray,
    diagonal_signs: np.ndarray,
    bitorder: str,
    zero_is_one: bool,
    transform_id: str,
) -> dict[str, Any]:
    return {
        "format_version": HADAMARD_STATE_FORMAT_VERSION,
        "transform_name": HADAMARD_TRANSFORM_NAME,
        "order": int(order),
        "seed": int(seed),
        "bitorder": str(bitorder),
        "zero_is_one": bool(zero_is_one),
        "permutation": [int(value) for value in permutation.tolist()],
        "diagonal_signs": [int(value) for value in diagonal_signs.tolist()],
        "transform_id": str(transform_id),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_state(
    path: Path,
    *,
    order: int,
    seed: int,
    bitorder: str,
    zero_is_one: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Cannot read Hadamard P/D state file: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Hadamard P/D state must be a JSON object: {path}")
    if int(payload.get("format_version", -1)) != HADAMARD_STATE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported Hadamard P/D state format in {path}: "
            f"{payload.get('format_version')!r}."
        )
    saved_name = str(payload.get("transform_name", "")).strip()
    if saved_name != HADAMARD_TRANSFORM_NAME:
        raise ValueError(
            f"Hadamard P/D state transform mismatch: {path} has {saved_name!r}."
        )
    saved_order = int(payload.get("order", -1))
    if saved_order != int(order):
        raise ValueError(
            "Hadamard P/D state order mismatch: "
            f"state={saved_order}, configured={order}, path={path}."
        )
    saved_seed = int(payload.get("seed", seed))
    if saved_seed != int(seed):
        raise ValueError(
            "Hadamard P/D state seed mismatch; refusing to regenerate a different "
            f"transform: state={saved_seed}, configured={seed}, path={path}."
        )
    saved_bitorder = _normalize_bitorder(payload.get("bitorder", bitorder))
    if saved_bitorder != bitorder:
        raise ValueError(
            "Hadamard P/D state bitorder mismatch: "
            f"state={saved_bitorder}, configured={bitorder}, path={path}."
        )
    saved_zero_is_one = bool(payload.get("zero_is_one", zero_is_one))
    if saved_zero_is_one != bool(zero_is_one):
        raise ValueError(
            "Hadamard P/D state zero-sign convention mismatch: "
            f"state={saved_zero_is_one}, configured={zero_is_one}, path={path}."
        )
    permutation, signs = _validate_pd(
        order,
        payload.get("permutation", []),
        payload.get("diagonal_signs", []),
    )
    transform_id = _transform_id(
        order,
        permutation,
        signs,
        bitorder,
        zero_is_one,
    )
    saved_id = str(payload.get("transform_id", "")).strip()
    if saved_id and saved_id != transform_id:
        raise ValueError(
            f"Hadamard P/D state checksum mismatch: {path}; "
            f"saved={saved_id}, calculated={transform_id}."
        )
    return permutation, signs, transform_id


@dataclass(frozen=True)
class HadamardTransformState:
    """已解析的固定 P/D 状态及其稳定身份。"""

    order: int
    seed: int
    permutation: np.ndarray
    diagonal_signs: np.ndarray
    bitorder: str
    zero_is_one: bool
    transform_id: str
    state_path: str = ""

    @property
    def pd_id(self) -> str:
        """兼容 ``transform_id`` 的简短属性名。"""

        return self.transform_id


class HadamardBinarizer:
    """对浮点描述子执行固定 PD、FWHT、sign 和 bit packing。"""

    def __init__(
        self,
        *,
        order: int = HADAMARD_ORDER,
        seed: int = 42,
        state_path: str | Path | None = None,
        bitorder: str = "little",
        zero_is_one: bool = True,
        permutation: Sequence[int] | np.ndarray | None = None,
        diagonal_signs: Sequence[int] | np.ndarray | None = None,
        persist: bool = True,
    ) -> None:
        order = int(order)
        if order < 2 or order & (order - 1):
            raise ValueError(
                f"Hadamard order must be a power of two >= 2, got {order}."
            )
        bitorder = _normalize_bitorder(bitorder)
        seed = int(seed)
        zero_is_one = bool(zero_is_one)
        path = Path(state_path).expanduser().resolve() if state_path is not None else None

        if path is not None and path.exists():
            loaded_permutation, loaded_signs, transform_id = _load_state(
                path,
                order=order,
                seed=seed,
                bitorder=bitorder,
                zero_is_one=zero_is_one,
            )
            if permutation is not None or diagonal_signs is not None:
                if permutation is None or diagonal_signs is None:
                    raise ValueError(
                        "Hadamard permutation and diagonal_signs must be supplied together."
                    )
                configured_permutation, configured_signs = _validate_pd(
                    order,
                    permutation,
                    diagonal_signs,
                )
                if not np.array_equal(configured_permutation, loaded_permutation):
                    raise ValueError(
                        "Hadamard state permutation does not match the explicitly "
                        f"configured transform: path={path}."
                    )
                if not np.array_equal(configured_signs, loaded_signs):
                    raise ValueError(
                        "Hadamard state diagonal signs do not match the explicitly "
                        f"configured transform: path={path}."
                    )
            permutation_array = loaded_permutation
            signs_array = loaded_signs
        else:
            if permutation is None and diagonal_signs is None:
                permutation_array, signs_array = generate_pd(order, seed)
            elif permutation is not None and diagonal_signs is not None:
                permutation_array, signs_array = _validate_pd(
                    order,
                    permutation,
                    diagonal_signs,
                )
            else:
                raise ValueError(
                    "Hadamard permutation and diagonal_signs must be supplied together."
                )
            transform_id = _transform_id(
                order,
                permutation_array,
                signs_array,
                bitorder,
                zero_is_one,
            )
            if path is not None and bool(persist):
                payload = _state_payload(
                    order,
                    seed,
                    permutation_array,
                    signs_array,
                    bitorder,
                    zero_is_one,
                    transform_id,
                )
                _atomic_write_json(path, payload)

        self.state = HadamardTransformState(
            order=order,
            seed=seed,
            permutation=np.asarray(permutation_array, dtype=np.int64),
            diagonal_signs=np.asarray(signs_array, dtype=np.int8),
            bitorder=bitorder,
            zero_is_one=zero_is_one,
            transform_id=transform_id,
            state_path=str(path) if path is not None else "",
        )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None,
        *,
        state_path: str | Path | None = None,
    ) -> "HadamardBinarizer":
        """从完整工程配置或直接配置段构造二值化器。"""

        section = _mapping_section(config)
        order = resolve_hadamard_order(config)
        seed = int(
            _configured_value(
                section,
                "seed",
                "random_seed",
                "pd_seed",
                default=42,
            )
        )
        bitorder = _normalize_bitorder(
            _configured_value(section, "bitorder", "binary_bitorder", default="little")
        )
        zero_is_one = _normalize_zero_is_one(section)
        configured_state_path = (
            state_path
            if state_path is not None
            else resolve_hadamard_state_path(config, order=order)
        )
        permutation = _configured_value(
            section,
            "permutation",
            "p",
            default=None,
        )
        diagonal_signs = _configured_value(
            section,
            "diagonal_signs",
            "signs",
            "d",
            default=None,
        )
        persist = bool(_configured_value(section, "persist", default=True))
        return cls(
            order=order,
            seed=seed,
            state_path=configured_state_path,
            bitorder=bitorder,
            zero_is_one=zero_is_one,
            permutation=permutation,
            diagonal_signs=diagonal_signs,
            persist=persist,
        )

    @property
    def order(self) -> int:
        return self.state.order

    @property
    def seed(self) -> int:
        return self.state.seed

    @property
    def permutation(self) -> np.ndarray:
        return self.state.permutation

    @property
    def diagonal_signs(self) -> np.ndarray:
        return self.state.diagonal_signs

    @property
    def bitorder(self) -> str:
        return self.state.bitorder

    @property
    def zero_is_one(self) -> bool:
        return self.state.zero_is_one

    @property
    def transform_id(self) -> str:
        return self.state.transform_id

    @property
    def state_path(self) -> str:
        return self.state.state_path

    def transform(self, values: np.ndarray, *, normalize: bool = False) -> np.ndarray:
        """计算 ``H(PDf)``，返回未二值化的 Hadamard 系数。"""

        array = np.asarray(values)
        if array.ndim == 0 or int(array.shape[-1]) != self.order:
            raise ValueError(
                "Hadamard binarizer expects descriptors with last dimension "
                f"{self.order}, got {array.shape}."
            )
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"Hadamard descriptors must be numeric, got {array.dtype}.")
        if not np.all(np.isfinite(array)):
            raise ValueError("Hadamard descriptors contain NaN or infinite values.")
        permuted = np.take(array, self.permutation, axis=-1)
        permuted = permuted * self.diagonal_signs
        return fwht(permuted, axis=-1, normalize=normalize)

    def bits(self, values: np.ndarray) -> np.ndarray:
        """计算未打包的 ``uint8`` 0/1 bits。"""

        transformed = self.transform(values)
        bits = transformed >= 0.0 if self.zero_is_one else transformed > 0.0
        return np.asarray(bits, dtype=np.uint8)

    def binarize(
        self,
        values: np.ndarray,
        *,
        packed: bool = True,
    ) -> np.ndarray:
        """将浮点描述子转成未打包 bit 或 packed ``uint8`` 描述子。"""

        bits = self.bits(values)
        if not packed:
            return bits
        return np.packbits(bits, axis=-1, bitorder=self.bitorder).astype(np.uint8)

    def __call__(self, values: np.ndarray, *, packed: bool = True) -> np.ndarray:
        return self.binarize(values, packed=packed)

    def transform_torch(self, values: Any, *, normalize: bool = False) -> Any:
        """Torch 版 ``H(PDf)``，用于需要固定 P/D 的训练/验证辅助计算。"""

        import torch

        if not torch.is_tensor(values):
            raise TypeError("Hadamard transform expects a torch.Tensor.")
        if values.ndim == 0 or int(values.shape[-1]) != self.order:
            raise ValueError(
                "Hadamard torch transform expects last dimension "
                f"{self.order}, got {tuple(values.shape)}."
            )
        if not torch.all(torch.isfinite(values)):
            raise ValueError("Hadamard torch descriptors contain non-finite values.")
        permutation = torch.as_tensor(
            self.permutation,
            dtype=torch.long,
            device=values.device,
        )
        signs = torch.as_tensor(
            self.diagonal_signs,
            dtype=values.dtype,
            device=values.device,
        )
        permuted = values.index_select(-1, permutation) * signs
        return fwht_torch(permuted, dim=-1, normalize=normalize)

    def metadata(self) -> dict[str, Any]:
        """返回可写入 checkpoint、模板和评估报告的稳定元数据。"""

        return {
            "hadamard_binarization_enabled": True,
            "hadamard_transform_name": HADAMARD_TRANSFORM_NAME,
            "hadamard_order": int(self.order),
            "hadamard_seed": int(self.seed),
            "hadamard_bitorder": self.bitorder,
            "hadamard_zero_is_one": bool(self.zero_is_one),
            "hadamard_transform_id": self.transform_id,
            "descriptor_transform_id": self.transform_id,
            "hadamard_state_path": self.state_path,
            "hadamard_state_format_version": HADAMARD_STATE_FORMAT_VERSION,
        }


def binarize_float_descriptors(
    descriptors: np.ndarray,
    *,
    order: int = HADAMARD_ORDER,
    seed: int = 42,
    state_path: str | Path | None = None,
    bitorder: str = "little",
    zero_is_one: bool = True,
    packed: bool = True,
) -> np.ndarray:
    """便捷函数：用固定 seed/state 对一批浮点描述子执行二值化。"""

    binarizer = HadamardBinarizer(
        order=order,
        seed=seed,
        state_path=state_path,
        bitorder=bitorder,
        zero_is_one=zero_is_one,
    )
    return binarizer.binarize(descriptors, packed=packed)


# 常见简写别名，保持调用方可以用 ``HadamardTransform`` 表达同一组件。
HadamardTransform = HadamardBinarizer
