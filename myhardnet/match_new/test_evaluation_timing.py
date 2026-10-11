from __future__ import annotations

import json
from pathlib import Path

from match_new.evaluation import write_unlock_timing_report


def test_unlock_timing_match_only_includes_stage_summaries(tmp_path: Path) -> None:
    rows = [
        {
            "label": "1",
            "accepted_at_selected_threshold": "1",
            "unlock_match_ms": "100.0",
            "unlock_end_to_end_ms": "140.0",
            "unlock_end_to_end_core_ms": "130.0",
            "registered_template_load_ms": "2.0",
            "descriptor_prepare_ms": "3.0",
            "candidate_generation_ms": "20.0",
            "candidate_filter_ms": "1.0",
            "ransac_ms": "30.0",
            "inlier_refinement_ms": "4.0",
            "unique_inlier_dedup_ms": "5.0",
            "texture_similarity_ms": "10.0",
            "score_fusion_ms": "2.0",
            "postprocess_ms": "3.0",
            "identity_fusion_ms": "1.0",
            "identity_match_overhead_ms": "19.0",
        },
        {
            "label": "1",
            "accepted_at_selected_threshold": "0",
            "unlock_match_ms": "90.0",
            "registered_template_load_ms": "9.0",
        },
        {
            "label": "0",
            "accepted_at_selected_threshold": "1",
            "unlock_match_ms": "80.0",
            "candidate_generation_ms": "99.0",
        },
    ]

    summary = write_unlock_timing_report(rows, tmp_path)
    match_only = summary["match_only"]

    assert match_only["count"] == 1
    assert match_only["average_ms"] == 100.0
    assert match_only["by_stage"]["candidate_generation_ms"]["average_ms"] == 20.0
    assert match_only["by_stage"]["ransac_ms"]["average_ms"] == 30.0
    assert match_only["stage_sum_of_means_ms"] == 100.0
    assert match_only["unaccounted_mean_ms"] == 0.0

    persisted = json.loads(
        (tmp_path / "unlock_timing.json").read_text(encoding="utf-8")
    )
    assert persisted["match_only"]["by_stage"]["texture_similarity_ms"]["p50_ms"] == 10.0
