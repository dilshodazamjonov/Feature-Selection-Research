from __future__ import annotations

import pytest

from credit_risk_fs.experiments.rank_voting import (
    aggregate_cross_dataset_rank_voting,
)


SCOPES = {
    "rf_corr_mrmr": "dev_fold_training_only",
    "boruta": "dev_fold_training_only",
}


def _aggregate(**overrides):
    values = {
        "eligible_features": ["a", "b", "c", "d"],
        "rankings": {
            "rf_corr_mrmr": ["a", "b", "c"],
            "boruta": ["b", "c", "d"],
        },
        "fit_scopes": SCOPES,
    }
    values.update(overrides)
    return aggregate_cross_dataset_rank_voting(**values)


def test_score_orientation_missing_rank_and_trace_columns():
    result = _aggregate()
    by_feature = result.set_index("feature")
    assert by_feature.loc["a", "rf_corr_mrmr_normalized_score"] == 1.0
    assert by_feature.loc["a", "boruta_normalized_score"] == 0.0
    assert by_feature.loc["d", "rf_corr_mrmr_present"] == False  # noqa: E712
    assert result.iloc[0]["feature"] == "b"


def test_ties_end_with_unicode_normalized_feature_name():
    result = _aggregate(
        eligible_features=["z", "A"],
        rankings={"rf_corr_mrmr": ["z"], "boruta": ["A"]},
    )
    assert result["feature"].tolist() == ["A", "z"]


@pytest.mark.parametrize("bad", ["TARGET", "loan_id", "recent_decision", "issue_d"])
def test_leakage_identity_and_split_fields_are_rejected(bad):
    with pytest.raises(ValueError, match="leakage/identity"):
        _aggregate(eligible_features=["a", "b", "c", bad])


def test_candidate_caps_and_voter_alias_deduplication():
    assert len(_aggregate(candidate_cap=2)) == 2
    with pytest.raises(ValueError, match="candidate_cap"):
        _aggregate(candidate_cap=5)
    with pytest.raises(ValueError, match="duplicates"):
        _aggregate(
            rankings={
                "rf_corr_mrmr": ["a"],
                "RandomForestRelevanceMRMRSelector": ["b"],
                "boruta": ["c"],
            }
        )


def test_all_supervised_voters_must_be_fold_local():
    with pytest.raises(ValueError, match="dev_fold_training_only"):
        _aggregate(
            fit_scopes={
                "rf_corr_mrmr": "full_dev",
                "boruta": "dev_fold_training_only",
            }
        )


def test_duplicate_unknown_and_canonical_collision_features_fail():
    with pytest.raises(ValueError, match="duplicate"):
        _aggregate(rankings={"rf_corr_mrmr": ["a", "a"], "boruta": ["b"]})
    with pytest.raises(ValueError, match="unknown"):
        _aggregate(rankings={"rf_corr_mrmr": ["a", "x"], "boruta": ["b"]})
    with pytest.raises(ValueError, match="canonical-name collisions"):
        _aggregate(eligible_features=["A", "a"])
