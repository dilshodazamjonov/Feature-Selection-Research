from __future__ import annotations

import pandas as pd

from credit_risk_fs.clip_final_comparison.diagnostics import jaccard, overlap_summary, semantic_pool_diagnostics
from credit_risk_fs.clip_final_comparison.temporal import construct_temporal_cutoffs, frozen_policy_manifest


def test_temporal_cutoff_construction_and_frozen_policy():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=120, freq="D"),
            "target": [0, 1] * 60,
        }
    )
    cutoffs = construct_temporal_cutoffs(
        frame,
        dataset="synthetic",
        date_column="date",
        target_column="target",
        min_dev_rows=20,
        min_oot_rows=20,
    )
    assert len(cutoffs) == 3
    assert cutoffs["eligibility_status"].eq("eligible").all()
    policy = frozen_policy_manifest()
    assert policy["refit_on_lendingclub_v2"] is False
    assert policy["oot_enters_selection"] is False


def test_candidate_pool_overlap_and_semantic_diagnostics():
    pool = pd.DataFrame(
        {
            "feature_name": ["income_mean", "income_max", "loan_ratio", "fico_flag"],
            "semantic_group": ["income", "income", "loan", "credit"],
        }
    )
    diagnostics = semantic_pool_diagnostics(pool)
    assert diagnostics["semantic_group_count"] == 3
    assert diagnostics["repeated_family_share"] > 0
    assert jaccard(["a", "b"], ["b", "c"]) == 1 / 3
    overlap = overlap_summary(["a", "b"], ["b", "c"], ["a"])
    assert overlap["jaccard_with_clip_v2_pool"] == 1 / 3

