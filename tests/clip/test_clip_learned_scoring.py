from __future__ import annotations

import pandas as pd

from credit_risk_fs.utils.io import read_json


def test_learned_scores_use_homecredit_anchor_and_keep_lendingclub_external(
    legacy_artifact_path,
):
    root = legacy_artifact_path(
        "results/corrected_homecredit_clip/training", required=False
    )
    anchor = read_json(f"{root}/learned_anchor_manifest.json")
    home = pd.read_csv(f"{root}/homecredit_learned_scores.csv")
    lc = pd.read_csv(f"{root}/lendingclub_v2_learned_scores.csv")

    assert anchor["anchor_dataset"] == "homecredit"
    assert anchor["anchor_count"] == 23
    assert "unchanged Home Credit" in anchor["lendingclub_v2_anchor_policy"]
    assert len(home) == 436
    assert len(lc) == 576
    assert set(lc["split"]) == {"external_validation"}
    assert home["learned_rank"].is_unique
    assert lc["learned_rank"].is_unique
    assert set(home["statistical_view_scope"]) == {"compact_target_free_v2"}
    assert set(lc["statistical_view_scope"]) == {"compact_target_free_v2"}
