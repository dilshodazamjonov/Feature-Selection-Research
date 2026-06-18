from __future__ import annotations

import pandas as pd

from credit_risk_fs.clip.group_split import build_group_split


def test_group_split_has_no_group_overlap_and_is_deterministic():
    frame = pd.DataFrame(
        {
            "dataset": ["homecredit"] * 6,
            "feature_name": [
                "BURO_AMT_SUM",
                "BURO_AMT_MEAN",
                "PREV_AMT_SUM",
                "PREV_AMT_MEAN",
                "AMT_CREDIT",
                "AMT_ANNUITY",
            ],
            "source_table": ["bureau", "bureau", "previous", "previous", "application", "application"],
            "semantic_group": ["a", "a", "b", "b", "c", "c"],
        }
    )

    first = build_group_split(frame, seed=42, validation_fraction=0.33)
    second = build_group_split(frame, seed=42, validation_fraction=0.33)

    assert first.split.equals(second.split)
    train_groups = set(first.split.loc[first.split["split"].eq("train"), "group_key"])
    val_groups = set(first.split.loc[first.split["split"].eq("validation"), "group_key"])
    assert not train_groups.intersection(val_groups)
    assert first.audit["group_overlap_count"] == 0

