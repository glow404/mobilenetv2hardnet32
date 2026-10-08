"""HardNet 反色重试匹配效果评估入口。

除 genuine query 匹配失败后将原图黑白反转并重试一次外，配置、输出和
命令行参数均沿用 run_hardnet_matching.py。

运行：python match_new\\test_run_hardnet_matching.py
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# 与 run_hardnet_matching.py 保持一致：确保 myhardnet 目录可导入 match_new。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from match_new import evaluation, identity_matcher
from match_new.run_hardnet_matching import main as base_main
from match_new.runtime import HardNetDescriptor, build_sift
from match_new.template_builder import build_hardnet_template_from_image

_original_score_query_against_identity = evaluation.score_query_against_identity
_reverse_template_cache: dict[tuple[str, str], dict[str, Any]] = {}
_runtime_objects: dict[int, tuple[HardNetDescriptor, Any]] = {}


def _score_with_inverted_query_retry(
    query_template: dict[str, Any],
    identity: dict[str, Any],
    config: dict[str, Any],
    cache: dict[str, dict[str, Any]],
    **kwargs: Any,
) -> dict[str, Any]:
    """本人匹配低于阈值时，反转 query 灰度图并重新构建模板后重试。"""

    result = _original_score_query_against_identity(
        query_template, identity, config, cache, **kwargs
    )
    threshold = float(
        dict(config.get("identification", {})).get("match_score_threshold", 0.55)
    )
    query_identity = str(query_template.get("identity_id", ""))
    owner_identity = str(identity.get("identity_id", ""))
    if query_identity != owner_identity or float(result["score"]) >= threshold:
        return result

    image = np.asarray(query_template.get("overlap_image", np.empty((0, 0))))
    if image.ndim != 2 or image.size == 0:
        return result

    cache_key = (query_identity, str(query_template.get("image_id", "")))
    inverted_template = _reverse_template_cache.get(cache_key)
    if inverted_template is None:
        runtime_key = id(config)
        if runtime_key not in _runtime_objects:
            _runtime_objects[runtime_key] = (
                HardNetDescriptor(config),
                build_sift(config),
            )
        hardnet, sift = _runtime_objects[runtime_key]
        inverted = cv2.bitwise_not(image.astype(np.uint8, copy=False))
        row = {
            "identity_id": query_identity,
            "image_id": cache_key[1],
            "image_path": str(query_template.get("image_path", "")),
        }
        with tempfile.TemporaryDirectory(prefix="hardnet_inverted_query_") as temp_dir:
            inverted_path = Path(temp_dir) / "inverted_query.png"
            if not cv2.imwrite(str(inverted_path), inverted):
                return result
            inverted_row = {**row, "image_path": str(inverted_path)}
            inverted_template, _timings = build_hardnet_template_from_image(
                inverted_row, hardnet, sift, config
            )
        _reverse_template_cache[cache_key] = inverted_template
    else:
        inverted_template = dict(inverted_template)

    # 输出仍引用原 query 图像，避免临时反色文件路径泄漏到结果中。
    inverted_template["image_path"] = str(query_template.get("image_path", ""))

    # 使用临时反色模板评估第二次尝试；注册模板仍来自原始图像。
    retry_result = _original_score_query_against_identity(
        inverted_template, identity, config, cache, **kwargs
    )
    # 第二次机会只能改善或维持匹配结果，不能用更低分覆盖首次结果。
    if float(retry_result["score"]) > float(result["score"]):
        return retry_result
    return result


def main() -> None:
    """复用原评估入口及参数处理，仅替换查询匹配函数。"""

    evaluation.score_query_against_identity = _score_with_inverted_query_retry
    identity_matcher.score_query_against_identity = _score_with_inverted_query_retry
    base_main()


if __name__ == "__main__":
    main()
