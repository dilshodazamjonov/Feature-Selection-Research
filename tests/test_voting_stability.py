"""Prompt 6 fold-level selection-stability tests."""

from __future__ import annotations

import pytest

from credit_risk_fs.analysis.voting_inference.stability import (
    fold_selection_inventory_rows,
    frozen_kuncheva_reference,
    pairwise_fold_stability,
    selection_frequency,
    summarise_pairwise_stability,
)


def test_identical_sets_give_maximum_stability() -> None:
    selections = {fold: ["a", "b", "c"] for fold in range(1, 6)}
    pairwise = pairwise_fold_stability(selections, universe_size=100)
    assert len(pairwise) == 10
    assert set(pairwise["jaccard"]) == {1.0}
    assert set(pairwise["kuncheva"]) == {1.0}


def test_disjoint_equal_size_sets_give_the_expected_negative_kuncheva() -> None:
    selections = {1: ["a", "b"], 2: ["c", "d"]}
    pairwise = pairwise_fold_stability(selections, universe_size=10)
    row = pairwise.iloc[0]
    assert row["jaccard"] == pytest.approx(0.0)
    # (0*10 - 2*2) / (10*2 - 2*2) = -4 / 16
    assert row["kuncheva"] == pytest.approx(-0.25)


def test_partial_overlap_matches_the_frozen_formula() -> None:
    selections = {1: ["a", "b", "c"], 2: ["b", "c", "d"]}
    pairwise = pairwise_fold_stability(selections, universe_size=20)
    row = pairwise.iloc[0]
    assert row["intersection_count"] == 2
    assert row["jaccard"] == pytest.approx(2 / 4)
    # (2*20 - 9) / (20*3 - 9) = 31 / 51
    assert row["kuncheva"] == pytest.approx(31 / 51)


def test_kuncheva_known_example_agrees_with_the_frozen_implementation() -> None:
    selections = {1: ["a", "b", "c"], 2: ["b", "c", "d"], 3: ["a", "c", "e"]}
    pairwise = pairwise_fold_stability(selections, universe_size=20)
    summary = summarise_pairwise_stability(pairwise)
    frozen = frozen_kuncheva_reference(selections, universe_size=20)
    assert summary["kuncheva_mean"] == pytest.approx(frozen, abs=1e-12)


def test_jaccard_treats_two_empty_sets_as_identical() -> None:
    pairwise = pairwise_fold_stability({1: [], 2: []}, universe_size=10)
    assert pairwise.iloc[0]["jaccard"] == pytest.approx(1.0)
    assert pairwise.iloc[0]["kuncheva"] is None
    assert pairwise.iloc[0]["kuncheva_unavailable_reason"] == "empty_selected_set"


def test_unequal_subset_sizes_leave_kuncheva_unavailable_with_a_reason() -> None:
    pairwise = pairwise_fold_stability({1: ["a", "b"], 2: ["a", "b", "c"]}, universe_size=10)
    row = pairwise.iloc[0]
    assert row["kuncheva"] is None
    assert row["kuncheva_unavailable_reason"] == "unequal_subset_sizes"
    assert row["jaccard"] == pytest.approx(2 / 3)


def test_universe_equal_to_subset_size_leaves_a_zero_denominator_reason() -> None:
    pairwise = pairwise_fold_stability({1: ["a", "b"], 2: ["a", "b"]}, universe_size=2)
    row = pairwise.iloc[0]
    assert row["kuncheva"] is None
    assert row["kuncheva_unavailable_reason"] == (
        "zero_denominator_universe_equals_subset_size"
    )


def test_invalid_universe_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="universe size"):
        pairwise_fold_stability({1: ["a"], 2: ["a"]}, universe_size=1)


def test_duplicate_selected_features_are_counted_not_silently_dropped() -> None:
    pairwise = pairwise_fold_stability(
        {1: ["a", "a", "b"], 2: ["a", "b"]}, universe_size=10
    )
    row = pairwise.iloc[0]
    assert row["left_duplicate_count"] == 1
    assert row["left_selected_count"] == 2
    assert row["kuncheva"] == pytest.approx(1.0)


def test_summary_retains_pair_counts_and_dispersion() -> None:
    selections = {1: ["a", "b"], 2: ["b", "c"], 3: ["a", "c"]}
    summary = summarise_pairwise_stability(
        pairwise_fold_stability(selections, universe_size=10)
    )
    assert summary["fold_pair_count"] == 3
    assert summary["jaccard_pair_count"] == 3
    assert summary["jaccard_mean"] == pytest.approx(1 / 3)
    assert summary["jaccard_min"] == pytest.approx(1 / 3)
    assert summary["jaccard_max"] == pytest.approx(1 / 3)
    assert summary["jaccard_std"] == pytest.approx(0.0, abs=1e-12)


def test_inventory_flags_missing_folds_and_pool_budget_conflicts() -> None:
    rows = fold_selection_inventory_rows(
        run_id="run",
        dataset="homecredit",
        model="lr",
        configuration="voting_k100",
        expected_budget=20,
        universe_size=529,
        fold_selections={1: [f"f{index}" for index in range(20)]},
        fold_candidate_pools={1: [f"f{index}" for index in range(10)]},
        declared_universe_counts={1: {529}},
        expected_fold_count=3,
    )
    assert len(rows) == 3
    assert rows[0]["present"] is True
    assert rows[0]["budget_matches"] is True
    assert rows[0]["candidate_pool_smaller_than_budget"] is True
    assert rows[0]["selected_outside_candidate_pool"] == 10
    assert rows[0]["run_declared_universe_matches_authenticated"] is True
    assert rows[1]["present"] is False
    assert rows[1]["selected_feature_count"] == 0


def test_selection_frequency_counts_distinct_folds() -> None:
    frame = selection_frequency({1: ["a", "b"], 2: ["a"], 3: ["a", "c"]})
    top = frame.iloc[0]
    assert top["feature"] == "a"
    assert top["selection_count"] == 3
    assert top["selection_frequency"] == pytest.approx(1.0)
