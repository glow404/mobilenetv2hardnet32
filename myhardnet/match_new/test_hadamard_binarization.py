"""不含 P/D 的 Hadamard 描述子二值化测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from match_new.hadamard_binarization import HadamardBinarizer, fwht


def test_binarizer_applies_direct_hadamard_without_pd() -> None:
    descriptors = np.asarray(
        [[1.0, 2.0, 3.0, 4.0], [-2.0, 1.0, 0.5, 3.0]],
        dtype=np.float32,
    )
    binarizer = HadamardBinarizer(order=4, bitorder="little")

    transformed = binarizer.transform(descriptors)
    expected = np.asarray(
        [
            [10.0, -2.0, -4.0, 0.0],
            [2.5, -5.5, -4.5, -0.5],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(transformed, expected)
    np.testing.assert_array_equal(transformed, fwht(descriptors))
    np.testing.assert_array_equal(
        binarizer.binarize(descriptors, packed=False),
        (expected >= 0.0).astype(np.uint8),
    )


def test_transform_identity_depends_only_on_public_parameters() -> None:
    first = HadamardBinarizer(order=8, bitorder="little", zero_is_one=True)
    second = HadamardBinarizer(order=8, bitorder="little", zero_is_one=True)
    changed = HadamardBinarizer(order=8, bitorder="big", zero_is_one=True)

    assert first.transform_id == second.transform_id
    assert first.transform_id != changed.transform_id
    assert "pd" not in first.transform_id
    assert "seed" not in first.metadata()
    assert "state_path" not in first.metadata()
