"""Packed-Hamming candidate backend regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import match_new.hardnet_matcher as matcher


def _binary_template(descriptors: np.ndarray, image_id: str) -> dict[str, object]:
    rows = int(descriptors.shape[0])
    return {
        "image_id": image_id,
        "has_hardnet": True,
        "hardnet_descriptors": np.ascontiguousarray(descriptors, dtype=np.uint8),
        "hardnet_descriptor_dim": int(descriptors.shape[1] * 8),
        "hardnet_descriptor_kind": "binary",
        "hardnet_descriptor_metric": "hamming",
        "hardnet_descriptor_storage": "packed_uint8",
        "hardnet_descriptor_bitorder": "little",
        "keypoints_angle": np.zeros((rows,), dtype=np.float32),
    }


def _sample_descriptors() -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray([[0x00], [0xFF]], dtype=np.uint8)
    gallery = np.asarray([[0x00], [0x0F], [0xFF]], dtype=np.uint8)
    return query, gallery


def test_config_defaults_packed_hamming_to_cpu() -> None:
    config_path = Path(__file__).with_name("config_match_new.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["matching"]["hamming"]["backend"] == "cpu"


def test_hamming_knn_cpu_returns_distance_then_sorted_index() -> None:
    query, gallery = _sample_descriptors()

    distances, indices = matcher.hamming_knn(
        query,
        gallery,
        hash_bits=8,
        k=3,
    )

    assert distances.dtype == np.float32
    assert indices.dtype == np.int32
    np.testing.assert_array_equal(
        indices,
        np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.int32),
    )
    np.testing.assert_allclose(
        distances,
        np.asarray([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]], dtype=np.float32),
    )


def test_default_hamming_backend_never_dispatches_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, gallery = _sample_descriptors()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail_if_called(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        raise AssertionError("default packed-Hamming backend must stay on CPU")

    monkeypatch.setattr(matcher, "_hamming_knn_torch", fail_if_called)

    distances, indices = matcher.hamming_knn(query, gallery, 8, 2)

    assert distances.shape == indices.shape == (2, 2)


def test_explicit_cuda_falls_back_to_cpu_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query, gallery = _sample_descriptors()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    cpu_result = matcher.hamming_knn(query, gallery, 8, 3, backend="cpu")
    fallback_result = matcher.hamming_knn(query, gallery, 8, 3, backend="cuda")

    np.testing.assert_allclose(fallback_result[0], cpu_result[0])
    np.testing.assert_array_equal(fallback_result[1], cpu_result[1])


def test_candidate_builder_passes_metric_specific_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_descriptors, gallery_descriptors = _sample_descriptors()
    query = _binary_template(query_descriptors, "query")
    gallery = _binary_template(gallery_descriptors, "gallery")
    observed_backends: list[str] = []
    original = matcher.hamming_bidirectional_knn

    def record_backend(
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        hash_bits: int,
        forward_k: int,
        reverse_k: int,
        *,
        backend: str = "cpu",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        observed_backends.append(backend)
        return original(
            query_array,
            gallery_array,
            hash_bits,
            forward_k,
            reverse_k,
            backend="cpu",
        )

    monkeypatch.setattr(matcher, "hamming_bidirectional_knn", record_backend)
    config = {
        "distance": "auto",
        "candidate_policy": "ratio_only",
        "top_k": 1,
        "ratio_threshold": 0.85,
        "bidirectional_ratio_test": True,
        "abs_distance_threshold": 1.0,
        "distance_margin": 0.1,
        "allow_many_to_one_before_ransac": True,
        "hamming": {
            "backend": "cuda",
            "ratio_threshold": 0.90,
            "abs_distance_threshold": 1.0,
        },
    }

    candidates = matcher.build_descriptor_candidates(query, gallery, config)

    assert observed_backends == ["cuda"]
    assert [(item.query_idx, item.gallery_idx) for item in candidates] == [
        (0, 0),
        (1, 2),
    ]


def test_bidirectional_hamming_keeps_empty_shapes_consistent() -> None:
    query = np.zeros((0, 64), dtype=np.uint8)
    gallery = np.zeros((3, 64), dtype=np.uint8)

    forward_distances, forward_indices, reverse_distances, reverse_indices = (
        matcher.hamming_bidirectional_knn(
            query,
            gallery,
            512,
            5,
            2,
        )
    )

    assert forward_distances.shape == forward_indices.shape == (0, 0)
    assert reverse_distances is not None
    assert reverse_indices is not None
    assert reverse_distances.shape == reverse_indices.shape == (3, 0)


def test_hamming_rejects_unknown_backend() -> None:
    query, gallery = _sample_descriptors()

    with pytest.raises(ValueError, match="unsupported Hamming backend"):
        matcher.hamming_knn(query, gallery, 8, 1, backend="metal")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_hamming_matches_cpu_and_reuses_bidirectional_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260810)
    query = rng.integers(0, 256, size=(37, 64), dtype=np.uint8)
    gallery = rng.integers(0, 256, size=(41, 64), dtype=np.uint8)
    calls = 0
    original = matcher._hamming_distance_counts_torch

    def count_distance_builds(
        query_array: np.ndarray,
        gallery_array: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(query_array, gallery_array, device)

    monkeypatch.setattr(
        matcher,
        "_hamming_distance_counts_torch",
        count_distance_builds,
    )
    cpu_result = matcher.hamming_bidirectional_knn(
        query,
        gallery,
        512,
        5,
        2,
        backend="cpu",
    )
    cuda_result = matcher.hamming_bidirectional_knn(
        query,
        gallery,
        512,
        5,
        2,
        backend="cuda",
    )

    assert calls == 1
    for cpu_array, cuda_array in zip(cpu_result, cuda_result):
        assert cpu_array is not None
        assert cuda_array is not None
        if np.issubdtype(cpu_array.dtype, np.floating):
            np.testing.assert_allclose(cuda_array, cpu_array, atol=0.0, rtol=0.0)
        else:
            np.testing.assert_array_equal(cuda_array, cpu_array)