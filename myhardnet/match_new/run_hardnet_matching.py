"""HardNet 浮点/二值手机解锁式指纹验证主入口。

作用：
    1. 扫描原始指纹图像目录，构建每张图像的 `.npz` 模板；
    2. 每个 identity 随机选择注册模板；
    3. 用其余图像作为 query，分别对本人和非本人 identity 模板库打分；
    4. 输出 FAR/FRR/EER/AUC、满足目标 FAR/FRR 的推荐阈值；
    5. 在目标阈值下导出 false reject / false accept 的原图和拼接预览。

运行参数默认写在 `config_match_new.yaml` 的 output/runtime/data/model/evaluation 等段；
命令行仅作覆盖，优先级：命令行 > 配置文件 > 程序默认值。

典型命令：
    python match_new\\run_hardnet_matching.py

调试：在配置里设置 runtime.limit_identities / limit_images_per_identity，
或临时用命令行覆盖。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardnet_train.model import (
    checkpoint_descriptor_dim,
    checkpoint_model_architecture,
)
from match_new.descriptor_contract import (
    BINARY_DESCRIPTOR_KIND,
    FLOAT32_STORAGE,
    FLOAT_DESCRIPTOR_KIND,
    HAMMING_DISTANCE_METRIC,
    L2_DISTANCE_METRIC,
    PACKED_UINT8_STORAGE,
    descriptor_columns,
    require_supported_contract,
    resolve_descriptor_contract,
)
from match_new.evaluation import run_hardnet_evaluation, summary_row
from match_new.input_loader import load_raw_image_metadata, validate_identity_image_counts
from match_new.template_builder import build_hardnet_templates, build_identity_templates, load_image_template
from match_new.utils import ensure_dir, load_config, resolve_path, template_filename, write_csv_rows, write_json


MATCH_NEW_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = MATCH_NEW_DIR / "config_match_new.yaml"
DEFAULT_OUTPUT_DIR = "outputs"

# 单张注册模板总时长的完整组成。这里不包含 template_total_ms、sift_ms 等
# 聚合/兼容字段，避免在按手指求和时重复计算同一阶段。
ENROLLMENT_STAGE_FIELDS = (
    "image_read_ms",
    "sift_keypoint_detection_ms",
    "keypoint_filter_ms",
    "patch_crop_rotate_ms",
    "hardnet_inference_ms",
    "template_assembly_ms",
    "template_pipeline_overhead_ms",
    "template_persist_ms",
    "template_registration_overhead_ms",
)


def numeric(value: Any) -> float | None:
    """尽可能把 CSV/JSON 中的值转换为浮点数；空值或非法值返回 ``None``。"""

    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], q: float) -> float | None:
    """用最近秩方式计算小规模耗时样本的百分位数。"""

    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * float(q)))
    index = max(0, min(len(ordered) - 1, index))
    return float(ordered[index])


def summarize_values(values: list[float]) -> dict[str, Any]:
    """汇总一组毫秒耗时，输出总计、均值、范围及 P50/P95。"""

    if not values:
        return {"count": 0}
    total = float(sum(values))
    return {
        "count": len(values),
        "total_ms": total,
        "avg_ms": total / len(values),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
    }


def build_enrollment_timing_report(success_rows: list[dict[str, str]], identity_payload: dict[str, Any]) -> dict[str, Any]:
    """按手指汇总被选中注册图像的模板构建时间。

    单个手指的注册总时长等于其全部注册图像模板构建耗时之和；缺失的耗时会单独
    计数，避免把不完整数据误当作真实注册时长。除总时长外，还会累计图像读取、
    SIFT 检测、关键点筛选、局部块裁剪旋转、HardNet 推理和模板持久化等阶段。
    """

    row_by_key = {(row["identity_id"], row["image_id"]): row for row in success_rows}
    per_identity: list[dict[str, Any]] = []
    for identity in identity_payload.get("identities", []):
        identity_id = str(identity.get("identity_id", ""))
        image_ids = [str(image_id) for image_id in identity.get("template_image_ids", [])]
        values: list[float] = []
        missing = 0
        stage_values = {
            field: []
            for field in ENROLLMENT_STAGE_FIELDS
        }
        missing_stage_timing_values = 0
        for image_id in image_ids:
            row = row_by_key.get((identity_id, image_id))
            elapsed = numeric(row.get("template_build_ms") if row else None)
            if elapsed is None:
                missing += 1
            else:
                values.append(elapsed)
            for field in ENROLLMENT_STAGE_FIELDS:
                stage_elapsed = numeric(row.get(field) if row else None)
                if stage_elapsed is None:
                    missing_stage_timing_values += 1
                else:
                    stage_values[field].append(stage_elapsed)
        summary = summarize_values(values)
        stage_complete = bool(image_ids) and all(
            len(stage_values[field]) == len(image_ids)
            for field in ENROLLMENT_STAGE_FIELDS
        )
        stage_totals = {
            field: (
                float(sum(stage_values[field]))
                if len(stage_values[field]) == len(image_ids) and image_ids
                else ""
            )
            for field in ENROLLMENT_STAGE_FIELDS
        }
        accounted_stage_ms = (
            float(sum(float(value) for value in stage_totals.values()))
            if stage_complete
            else ""
        )
        registration_total_ms = summary.get("total_ms", "")
        per_identity.append(
            {
                "identity_id": identity_id,
                "num_enrollment_templates": len(image_ids),
                "num_timed_templates": len(values),
                "missing_timing_count": missing,
                "stage_timing_complete": int(stage_complete),
                "missing_stage_timing_values": missing_stage_timing_values,
                "registration_total_ms": registration_total_ms,
                "registration_avg_template_ms": summary.get("avg_ms", ""),
                "registration_min_template_ms": summary.get("min_ms", ""),
                "registration_max_template_ms": summary.get("max_ms", ""),
                "registration_p50_template_ms": summary.get("p50_ms", ""),
                "registration_p95_template_ms": summary.get("p95_ms", ""),
                **{
                    f"registration_{field}": value
                    for field, value in stage_totals.items()
                },
                "registration_accounted_stage_ms": accounted_stage_ms,
                "registration_accounting_error_ms": (
                    float(registration_total_ms) - float(accounted_stage_ms)
                    if registration_total_ms not in {"", None}
                    and accounted_stage_ms not in {"", None}
                    else ""
                ),
            }
        )
    totals = [float(row["registration_total_ms"]) for row in per_identity if row.get("registration_total_ms") not in {"", None}]
    return {"per_identity": per_identity, "summary": summarize_values(totals)}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。默认值都在配置文件中；这里只提供覆盖项。"""

    parser = argparse.ArgumentParser(
        description="运行 HardNet 浮点/L2 或二值/Hamming 手机指纹解锁离线评估；默认参数来自 --config 指定的 YAML。",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置文件路径。")
    parser.add_argument("--image-root", "--image_root", dest="image_root", default=None, help="覆盖 data.image_root。")
    parser.add_argument(
        "--identity-depth",
        "--identity_depth",
        dest="identity_depth",
        type=int,
        default=None,
        help="覆盖 data.identity_depth。",
    )
    parser.add_argument("--model_path", default=None, help="覆盖 model.checkpoint。")
    parser.add_argument("--output_dir", default=None, help="覆盖 output.output_dir。")
    parser.add_argument(
        "--skip-template-build",
        "--skip_template_build",
        dest="skip_template_build",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 runtime.skip_template_build。可用 --skip-template-build / --no-skip-template-build。",
    )
    parser.add_argument("--max_impostor_identities_per_query", type=int, default=None, help="覆盖 runtime.max_impostor_identities_per_query。")
    parser.add_argument("--random_seed", type=int, default=None, help="覆盖 enrollment.random_seed。")
    parser.add_argument("--target_far", type=float, default=None, help="覆盖 evaluation.target_far。")
    parser.add_argument("--target_frr", type=float, default=None, help="覆盖 evaluation.target_frr。")
    parser.add_argument("--max_failure_cases_per_type", type=int, default=None, help="覆盖 evaluation.failure_export.max_cases_per_type。")
    parser.add_argument(
        "--failure-export",
        "--failure_export",
        dest="failure_export",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="覆盖 evaluation.failure_export.enabled。可用 --failure-export / --no-failure-export。",
    )
    parser.add_argument("--limit_identities", type=int, default=None, help="覆盖 runtime.limit_identities。")
    parser.add_argument("--limit_images_per_identity", type=int, default=None, help="覆盖 runtime.limit_images_per_identity。")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """把命令行覆盖项写回 config。仅当命令行显式传入时覆盖。"""

    if args.image_root is not None:
        config.setdefault("data", {})["image_root"] = str(Path(args.image_root).expanduser().resolve())
    if args.identity_depth is not None:
        config.setdefault("data", {})["identity_depth"] = int(args.identity_depth)
    if args.model_path:
        config.setdefault("model", {})["checkpoint"] = str(Path(args.model_path).expanduser().resolve())
    if args.output_dir is not None:
        # 命令行路径按当前工作目录理解，写成绝对路径避免歧义。
        config.setdefault("output", {})["output_dir"] = str(Path(args.output_dir).expanduser().resolve())
        config.setdefault("output", {})["_output_dir_from_cli"] = True
    if args.skip_template_build is not None:
        config.setdefault("runtime", {})["skip_template_build"] = bool(args.skip_template_build)
    if args.max_impostor_identities_per_query is not None:
        config.setdefault("runtime", {})["max_impostor_identities_per_query"] = int(args.max_impostor_identities_per_query)
    if args.random_seed is not None:
        config.setdefault("enrollment", {})["random_seed"] = int(args.random_seed)
    if args.target_far is not None:
        config.setdefault("evaluation", {})["target_far"] = float(args.target_far)
    if args.target_frr is not None:
        config.setdefault("evaluation", {})["target_frr"] = float(args.target_frr)
    if args.max_failure_cases_per_type is not None:
        config.setdefault("evaluation", {}).setdefault("failure_export", {})["max_cases_per_type"] = int(args.max_failure_cases_per_type)
    if args.failure_export is not None:
        config.setdefault("evaluation", {}).setdefault("failure_export", {})["enabled"] = bool(args.failure_export)
    if args.limit_identities is not None:
        config.setdefault("runtime", {})["limit_identities"] = int(args.limit_identities)
    if args.limit_images_per_identity is not None:
        config.setdefault("runtime", {})["limit_images_per_identity"] = int(args.limit_images_per_identity)


def resolve_run_settings(config: dict[str, Any]) -> dict[str, Any]:
    """从配置解析本次运行的路径与开关。"""

    output_cfg = dict(config.get("output", {}))
    runtime_cfg = dict(config.get("runtime", {}))
    failure_cfg = dict(dict(config.get("evaluation", {})).get("failure_export", {}))
    raw_output = output_cfg.get("output_dir", DEFAULT_OUTPUT_DIR)
    if bool(output_cfg.get("_output_dir_from_cli")):
        output_dir = Path(str(raw_output)).expanduser()
    else:
        output_dir = resolve_path(config, raw_output)
    return {
        "output_dir": output_dir,
        "skip_template_build": bool(runtime_cfg.get("skip_template_build", False)),
        "max_impostor_identities_per_query": int(runtime_cfg.get("max_impostor_identities_per_query", 0)),
        "limit_identities": int(runtime_cfg.get("limit_identities", 0)),
        "limit_images_per_identity": int(runtime_cfg.get("limit_images_per_identity", 0)),
        "export_failures": bool(failure_cfg.get("enabled", True)),
    }


def limit_rows_for_debug(rows: list[dict[str, str]], limit_identities: int, limit_images_per_identity: int) -> list[dict[str, str]]:
    """调试用数据裁剪。

    正式实验不要设置这两个参数；它们只用于快速 smoke test，
    例如 2 个 identity、每个 identity 22 张图，刚好能形成 20 张注册图和 2 张 query。
    """

    if limit_identities <= 0 and limit_images_per_identity <= 0:
        return rows
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["identity_id"]].append(row)
    selected: list[dict[str, str]] = []
    for identity_id in sorted(groups)[: limit_identities or None]:
        identity_rows = sorted(groups[identity_id], key=lambda item: item["image_id"])
        if limit_images_per_identity > 0:
            identity_rows = identity_rows[:limit_images_per_identity]
        selected.extend(identity_rows)
    return selected


def checkpoint_descriptor_metadata(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """从浮点或二值 checkpoint 解析统一描述子元数据。"""

    kind = str(checkpoint.get("descriptor_kind", FLOAT_DESCRIPTOR_KIND)).strip().lower()
    metric = str(checkpoint.get("descriptor_metric", L2_DISTANCE_METRIC)).strip().lower()
    if (kind, metric) not in {
        (FLOAT_DESCRIPTOR_KIND, L2_DISTANCE_METRIC),
        (BINARY_DESCRIPTOR_KIND, HAMMING_DISTANCE_METRIC),
    }:
        raise ValueError(
            f"Unsupported checkpoint descriptor contract: kind={kind}, metric={metric}."
        )
    dimension = checkpoint_descriptor_dim(checkpoint)
    if kind == BINARY_DESCRIPTOR_KIND:
        architecture = str(
            checkpoint.get("model_architecture", "residual_binary_hash_v1")
        )
        storage = str(
            checkpoint.get("binary_storage", PACKED_UINT8_STORAGE)
        ).strip().lower()
        bitorder = str(checkpoint.get("binary_bitorder", "little")).strip().lower()
    else:
        architecture = checkpoint_model_architecture(checkpoint)
        storage = FLOAT32_STORAGE
        bitorder = ""
    return {
        "model_architecture": architecture,
        "descriptor_kind": kind,
        "descriptor_metric": metric,
        "descriptor_dim": dimension,
        "descriptor_storage": storage,
        "descriptor_bitorder": bitorder,
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "model_parameter_count": checkpoint.get("model_parameter_count"),
        "model_macs_per_patch": checkpoint.get("model_macs_per_patch"),
    }


def validate_configured_descriptor(
    config: dict[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    """将配置中的显式描述子选项作为 checkpoint 契约断言。"""

    model_cfg = dict(config.get("model", {}))
    matching_cfg = dict(config.get("matching", {}))
    checks = {
        "model.descriptor_kind": (
            model_cfg.get("descriptor_kind", "auto"),
            metadata["descriptor_kind"],
        ),
        "matching.distance": (
            matching_cfg.get("distance", "auto"),
            metadata["descriptor_metric"],
        ),
    }
    for field, (configured, actual) in checks.items():
        value = str(configured if configured is not None else "").strip().lower()
        if value == "euclidean":
            value = L2_DISTANCE_METRIC
        if value not in {"", "auto", str(actual).lower()}:
            raise ValueError(
                f"{field} does not match checkpoint: config={value}, checkpoint={actual}."
            )
    configured_dim = str(
        model_cfg.get("descriptor_dim", "auto")
    ).strip().lower()
    if configured_dim not in {"", "auto"} and int(configured_dim) != int(
        metadata["descriptor_dim"]
    ):
        raise ValueError(
            "model.descriptor_dim does not match checkpoint: "
            f"config={configured_dim}, checkpoint={metadata['descriptor_dim']}."
        )
    if metadata["descriptor_kind"] == BINARY_DESCRIPTOR_KIND:
        configured_storage = str(
            model_cfg.get("binary_storage", "auto")
        ).strip().lower()
        configured_bitorder = str(
            model_cfg.get("binary_bitorder", "auto")
        ).strip().lower()
        if configured_storage not in {
            "",
            "auto",
            str(metadata["descriptor_storage"]).lower(),
        }:
            raise ValueError(
                "model.binary_storage does not match checkpoint: "
                f"config={configured_storage}, checkpoint={metadata['descriptor_storage']}."
            )
        if configured_bitorder not in {
            "",
            "auto",
            str(metadata["descriptor_bitorder"]).lower(),
        }:
            raise ValueError(
                "model.binary_bitorder does not match checkpoint: "
                f"config={configured_bitorder}, checkpoint={metadata['descriptor_bitorder']}."
            )


def resolve_expected_descriptor_dim(
    config: dict[str, Any],
    checkpoint_path: str | Path,
) -> int:
    """从 checkpoint 解析描述子维度并校验显式配置。"""

    target = Path(checkpoint_path).expanduser()
    if not target.exists():
        raise FileNotFoundError(f"Descriptor checkpoint does not exist: {target}")
    checkpoint = torch.load(target, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Unsupported checkpoint format: {target}")
    metadata = checkpoint_descriptor_metadata(checkpoint)
    validate_configured_descriptor(config, metadata)
    return int(metadata["descriptor_dim"])


def resolve_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    """读取浮点或二值 checkpoint 的统一元数据。"""

    target = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(target, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Unsupported checkpoint format: {target}")
    return checkpoint_descriptor_metadata(checkpoint)


def validate_templates(
    template_dir: str | Path,
    rows: list[dict[str, str]],
    config: dict[str, Any],
    descriptor_metadata: Mapping[str, Any],
    max_checks: int = 10,
) -> dict[str, Any]:
    """抽样检查模板字段是否与 checkpoint 描述子契约一致。"""

    require_overlap_image = bool(dict(config.get("texture_verification", {})).get("enabled", False))
    checked = 0
    for row in rows[:max_checks]:
        template = load_image_template(Path(template_dir) / template_filename(row["identity_id"], row["image_id"]), require="hardnet")
        n = int(template["keypoints_xy"].shape[0])
        hardnet = template["hardnet_descriptors"]
        contract = resolve_descriptor_contract(template, "hardnet")
        require_supported_contract(
            contract,
            label=str(template.get("template_path", row["image_id"])),
        )
        actual = {
            "descriptor_kind": contract.kind,
            "descriptor_metric": contract.metric,
            "descriptor_dim": contract.dimension,
            "descriptor_storage": contract.storage,
            "descriptor_bitorder": contract.bitorder,
        }
        expected = {
            field: descriptor_metadata[field]
            for field in actual
        }
        if actual != expected:
            raise ValueError(
                f"Template/checkpoint descriptor mismatch for {row['image_id']}: "
                f"template={actual}, checkpoint={expected}. Rebuild templates."
            )
        expected_shape = (n, descriptor_columns(contract))
        if hardnet.shape != expected_shape:
            raise ValueError(
                f"HardNet shape mismatch for {row['image_id']}: "
                f"{hardnet.shape} vs {expected_shape}"
            )
        if require_overlap_image and not bool(template.get("has_overlap_image", False)):
            raise ValueError(
                f"Template {row['image_id']} has no overlap_image required by texture verification. "
                "Rebuild templates without --skip-template-build."
            )
        checked += 1
    patch_cfg = dict(config.get("patch", {}))
    data_cfg = dict(config.get("data", {}))
    image_root = resolve_path(config, data_cfg.get("image_root", ""))
    return {
        "checked_templates": checked,
        "patch_crop_size": int(patch_cfg.get("crop_size", 32)),
        "patch_out_size": int(patch_cfg.get("out_size", 32)),
        "descriptor_type": "hardnet",
        **{
            field: descriptor_metadata[field]
            for field in (
                "descriptor_kind",
                "descriptor_metric",
                "descriptor_dim",
                "descriptor_storage",
                "descriptor_bitorder",
            )
        },
        "evaluation_dataset": image_root.name,
        "image_root": str(image_root),
        "overlap_image_required": require_overlap_image,
    }


def main() -> None:
    """串起完整实验流程。"""

    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    settings = resolve_run_settings(config)
    checkpoint = resolve_path(config, dict(config.get("model", {}))["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Descriptor checkpoint does not exist: {checkpoint}")
    descriptor_dim = resolve_expected_descriptor_dim(config, checkpoint)
    checkpoint_metadata = resolve_checkpoint_metadata(checkpoint)
    descriptor_metric = str(checkpoint_metadata["descriptor_metric"])
    config.setdefault("model", {})["descriptor_kind"] = str(
        checkpoint_metadata["descriptor_kind"]
    )
    config.setdefault("matching", {})["distance"] = descriptor_metric
    eval_dir_name = f"eval_hardnet_{descriptor_metric}"
    summary_stem = f"hardnet_{descriptor_metric}_summary"

    image_root = resolve_path(
        config,
        dict(config.get("data", {})).get("image_root", ""),
    )

    # 1. 直接扫描原始图像；调试时可按 identity 数和每个 identity 的图像数裁剪数据。
    output_dir = ensure_dir(settings["output_dir"])
    rows = load_raw_image_metadata(config)
    rows = limit_rows_for_debug(rows, int(settings["limit_identities"]), int(settings["limit_images_per_identity"]))
    if not rows:
        raise RuntimeError("No raw fingerprint images found.")
    enrollment = dict(config.get("enrollment", {}))
    enrollment_count = int(enrollment.get("enrollment_images_per_identity", 20))
    validate_identity_image_counts(rows, enrollment_count + 1, context="raw image input")
    write_csv_rows(output_dir / "metadata_all.csv", rows)

    # 2. 构建图像级模板；如果 skip_template_build，则按原始 metadata 复用已有模板。
    template_dir = output_dir / "image_templates"
    if settings["skip_template_build"]:
        success_rows = []
        for candidate in rows:
            key = (candidate["identity_id"], candidate["image_id"])
            template_path = template_dir / template_filename(*key)
            if not template_path.exists():
                continue
            row = dict(candidate)
            row["template_path"] = str(template_path)
            row["status"] = "success"
            success_rows.append(row)
        if not success_rows:
            raise RuntimeError("skip_template_build was set but no image templates were found.")
    else:
        report = build_hardnet_templates(rows, template_dir, config)
        write_json(output_dir / "build_report.json", {k: v for k, v in report.items() if k != "success_rows"})
        write_csv_rows(output_dir / "template_build_timings.csv", report.get("template_timings", []))
        success_rows = report["success_rows"]
        if not success_rows:
            raise RuntimeError("No templates were built successfully.")
    validate_identity_image_counts(success_rows, enrollment_count + 1, context="successfully built templates")
    validate_templates(
        template_dir,
        success_rows,
        config,
        descriptor_metadata=checkpoint_metadata,
    )

    # 3. 固定随机种子，为每个 identity 选择注册模板，其余作为 query。
    identity_templates_path = output_dir / f"identity_templates_{enrollment_count}.json"
    identity_payload, split_rows = build_identity_templates(
        success_rows,
        identity_templates_path,
        enrollment_count=enrollment_count,
        seed=int(enrollment.get("random_seed", 42)),
    )
    enrollment_timing = build_enrollment_timing_report(success_rows, identity_payload)
    write_csv_rows(output_dir / "enrollment_timing.csv", enrollment_timing["per_identity"])
    write_json(output_dir / "enrollment_timing.json", enrollment_timing)

    # 4. 执行身份验证评估，并在目标阈值下导出失败样本。
    result = run_hardnet_evaluation(
        metadata_rows=split_rows,
        identity_templates_path=identity_templates_path,
        template_dir=template_dir,
        config=config,
        output_dir=output_dir / eval_dir_name,
        max_impostor_identities_per_query=int(settings["max_impostor_identities_per_query"]),
        export_failures=bool(settings["export_failures"]),
    )
    # 5. 汇总一行 CSV/JSON，方便和其他实验横向比较。
    far_points = [float(point) for point in dict(config.get("evaluation", {})).get("far_points", [0.001, 0.0001])]
    summary_record = {
        "evaluation_dataset": image_root.name,
        "model_architecture": checkpoint_metadata.get("model_architecture", ""),
        "descriptor_dim": int(descriptor_dim),
        "checkpoint": str(checkpoint),
        **summary_row(result["metrics"], far_points),
    }
    summary = [summary_record]
    summary_csv = output_dir / f"{summary_stem}.csv"
    summary_json = output_dir / f"{summary_stem}.json"
    write_csv_rows(summary_csv, summary)
    write_json(
        summary_json,
        {
            "summary": summary,
            "identity_templates": str(identity_templates_path),
            "image_templates": str(template_dir),
            "eval_dir": str(output_dir / eval_dir_name),
            "checkpoint": str(checkpoint),
            "checkpoint_metadata": checkpoint_metadata,
            "descriptor_dim": int(descriptor_dim),
            "evaluation_dataset": resolve_path(
                config,
                dict(config.get("data", {})).get("image_root", ""),
            ).name,
            "effective_config": result["metrics"].get("effective_config"),
            "run_settings": {
                "output_dir": str(output_dir),
                "skip_template_build": settings["skip_template_build"],
                "max_impostor_identities_per_query": settings["max_impostor_identities_per_query"],
                "limit_identities": settings["limit_identities"],
                "limit_images_per_identity": settings["limit_images_per_identity"],
                "export_failures": settings["export_failures"],
            },
        },
    )
    print(
        json.dumps(
            {
                "summary_csv": str(summary_csv),
                "summary_json": str(summary_json),
                "far_frr_table_csv": result["far_frr_table_path"],
                "per_finger_far_frr_csv": result["per_finger_far_frr_path"],
                "outputs": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
