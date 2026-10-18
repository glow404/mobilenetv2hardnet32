"""Tests for compact Windows-safe match output directory names."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from match_new.utils import compact_path_token, short_case_directory_name


def test_short_case_directory_name_matches_requested_format() -> None:
    assert short_case_directory_name(5, "pair_35", "zys_R0") == "05_q_pair35_zysR0"


def test_short_case_directory_name_stays_compact_for_long_ids() -> None:
    name = short_case_directory_name(
        123,
        "query_" + "x" * 200,
        "owner_" + "y" * 200,
    )
    assert name.startswith("123_q_query")
    assert len(name) < 120
    assert "_" not in compact_path_token("pair_35")
