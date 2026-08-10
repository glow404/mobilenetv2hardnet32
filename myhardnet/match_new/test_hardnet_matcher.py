"""HardNet Hamming 候选生成的返回契约回归测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from match_new import hardnet_matcher


def _random_packed_codes(
    rows: int,
    packed_bytes: int,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(
        0,
        256,
        size=(rows, packed_bytes),
        dtype=np.uint8,
    )


def test_hamming_knn_torch_returns_distances_before_indices() -> None:
    query = _random_packed_codes(11, 8, seed=1)
    gallery = _random_packed_codes(13, 8, seed=2)

    distances, indices = hardnet_matcher._hamming_knn_torch(
        query,
        gallery,
        hash_bits=64,
        k=3,
        device=torch.device("cpu"),
    )

    assert distances.shape == indices.shape == (11, 3)
    assert distances.dtype == np.float32
    assert indices.dtype == np.int32
    assert np.all((0.0 <= distances) & (distances <= 1.0))
    assert np.all((0 <= indices) & (indices < len(gallery)))

    full_distances = hardnet_matcher._hamming_distance_torch(
        query,
        gallery,
        hash_bits=64,
        device=torch.device("cpu"),
    )
    selected_distances = np.take_along_axis(full_distances, indices, axis=1)
    np.testing.assert_array_equal(distances, selected_distances)
    np.testing.assert_array_equal(
        distances,
        np.sort(full_distances, axis=1)[:, :3],
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_hamming_knn_cuda_matches_cpu_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    query = _random_packed_codes(19, 64, seed=3)
    gallery = _random_packed_codes(23, 64, seed=4)

    monkeypatch.setattr(
        hardnet_matcher,
        "_candidate_device",
        torch.device("cuda"),
    )
    cuda_distances, cuda_indices = hardnet_matcher.hamming_knn(
        query,
        gallery,
        hash_bits=512,
        k=4,
    )

    monkeypatch.setattr(
        hardnet_matcher,
        "_candidate_device",
        torch.device("cpu"),
    )
    cpu_distances, _ = hardnet_matcher.hamming_knn(
        query,
        gallery,
        hash_bits=512,
        k=4,
    )
    full_cpu_distances = hardnet_matcher.pairwise_hamming(
        query,
        gallery,
        hash_bits=512,
    )

    assert cuda_distances.dtype == np.float32
    assert cuda_indices.dtype == np.int32
    np.testing.assert_array_equal(cuda_distances, cpu_distances)
    np.testing.assert_array_equal(
        cuda_distances,
        np.take_along_axis(full_cpu_distances, cuda_indices, axis=1),
    )