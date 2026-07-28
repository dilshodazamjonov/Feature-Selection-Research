"""Prompt 6 paired-inference tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from credit_risk_fs.analysis.voting_inference.paired import (
    Comparison,
    apply_holm_families,
    assert_bootstrap_equivalence,
    fast_paired_stratified_bootstrap,
    recover_predeclared_family,
    run_paired_delong,
)
from credit_risk_fs.evaluation.paired_inference import holm_adjust

ROOT = Path(__file__).resolve().parents[1]


def _aligned(count: int = 400, *, seed: int = 1, decimals: int = 2) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    target = generator.binomial(1, 0.25, count)
    if target.sum() < 2 or count - target.sum() < 2:
        raise AssertionError("fixture must contain both classes")
    return pd.DataFrame(
        {
            "stable_row_id": [f"{index:06d}" for index in range(count)],
            "target": target,
            "score_a": np.round(generator.random(count), decimals),
            "score_b": np.round(generator.random(count), decimals),
        }
    )


def test_accelerated_bootstrap_reproduces_the_frozen_implementation_exactly() -> None:
    for decimals in (0, 1, 3):
        aligned = _aligned(600, seed=decimals + 2, decimals=decimals)
        report = assert_bootstrap_equivalence(
            aligned, repetitions=40, minimum_valid=1, tolerance=0.0
        )
        assert report["equivalent"], report
        assert report["maximum_absolute_difference"] == 0.0
        assert report["valid_repetitions_frozen"] == report["valid_repetitions_fast"]


def test_bootstrap_is_deterministic_under_a_fixed_seed() -> None:
    aligned = _aligned(300, seed=9)
    first = fast_paired_stratified_bootstrap(aligned, repetitions=25, minimum_valid=1)
    second = fast_paired_stratified_bootstrap(aligned, repetitions=25, minimum_valid=1)
    for metric in ("auc", "ks", "lift_at_10"):
        assert first["metrics"][metric] == second["metrics"][metric]


def test_bootstrap_resamples_within_target_class() -> None:
    aligned = _aligned(500, seed=12)
    result = fast_paired_stratified_bootstrap(aligned, repetitions=5, minimum_valid=1)
    assert result["stratification"] == (
        "positive_and_negative_sampled_separately_with_paired_indices"
    )
    # Class-stratified resampling keeps the replicate class balance fixed, so no
    # replicate can fail for a missing class.
    assert result["failed_repetitions"] == 0
    assert result["valid_repetitions"] == 5


def test_identical_predictions_produce_a_zero_delta_everywhere() -> None:
    aligned = _aligned(400, seed=3)
    aligned["score_b"] = aligned["score_a"]
    result = fast_paired_stratified_bootstrap(aligned, repetitions=30, minimum_valid=1)
    for metric in ("auc", "ks", "lift_at_10"):
        block = result["metrics"][metric]
        assert block["observed_difference_a_minus_b"] == pytest.approx(0.0, abs=1e-12)
        assert block["ci95_percentile_lower"] == pytest.approx(0.0, abs=1e-12)
        assert block["ci95_percentile_upper"] == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_is_invariant_to_input_row_ordering() -> None:
    aligned = _aligned(400, seed=15)
    shuffled = aligned.sample(frac=1.0, random_state=7).reset_index(drop=True)
    ordered = aligned.sort_values("stable_row_id", kind="mergesort").reset_index(drop=True)
    reordered = shuffled.sort_values("stable_row_id", kind="mergesort").reset_index(drop=True)
    first = fast_paired_stratified_bootstrap(ordered, repetitions=20, minimum_valid=1)
    second = fast_paired_stratified_bootstrap(reordered, repetitions=20, minimum_valid=1)
    for metric in ("auc", "ks", "lift_at_10"):
        assert first["metrics"][metric] == second["metrics"][metric]


def _comparison() -> Comparison:
    return Comparison(
        family="homecredit_lr",
        dataset="homecredit",
        model="lr",
        reference_run_id="ref",
        comparator_run_id="cmp",
        candidate_pool_budget=200,
        designation="primary",
    )


def test_delong_uses_the_comparator_minus_reference_convention() -> None:
    generator = np.random.default_rng(21)
    target = generator.binomial(1, 0.3, 800)
    reference = generator.random(800)
    # A comparator that is strictly more informative must produce a positive delta.
    comparator = np.clip(0.7 * target + 0.3 * generator.random(800), 0.0, 1.0)
    aligned = pd.DataFrame(
        {
            "target": target,
            "score_reference": reference,
            "score_comparator": comparator,
        }
    )
    result = run_paired_delong(aligned, _comparison())
    assert result["direction_convention"] == "comparator_minus_reference"
    assert result["auc_comparator"] > result["auc_reference"]
    assert result["auc_delta_comparator_minus_reference"] > 0
    assert result["gini_delta_comparator_minus_reference"] == pytest.approx(
        2 * result["auc_delta_comparator_minus_reference"]
    )
    assert result["aligned_sample_size"] == 800
    assert result["positive_count"] + result["negative_count"] == 800
    assert 0.0 <= result["raw_two_sided_p_value"] <= 1.0

    swapped = aligned.rename(
        columns={
            "score_reference": "score_comparator",
            "score_comparator": "score_reference",
        }
    )
    mirrored = run_paired_delong(swapped, _comparison())
    assert mirrored["auc_delta_comparator_minus_reference"] == pytest.approx(
        -result["auc_delta_comparator_minus_reference"], abs=1e-12
    )


def test_delong_rejects_a_single_class_input() -> None:
    aligned = pd.DataFrame(
        {
            "target": [1, 1, 1, 1],
            "score_reference": [0.2, 0.4, 0.6, 0.8],
            "score_comparator": [0.3, 0.5, 0.7, 0.9],
        }
    )
    with pytest.raises(ValueError, match="binary target classes"):
        run_paired_delong(aligned, _comparison())


def test_holm_adjustment_matches_a_known_example() -> None:
    # Classic step-down example: 3 tests with p = 0.01, 0.02, 0.04.
    assert holm_adjust([0.01, 0.02, 0.04]) == pytest.approx([0.03, 0.04, 0.04])


def test_holm_families_are_separate_and_fully_audited() -> None:
    rows = [
        {
            "family": "homecredit_lr",
            "dataset": "homecredit",
            "model": "lr",
            "comparison_label": f"voting_pool_{budget}_vs_rf_corr_mrmr",
            "designation": "primary" if budget == 200 else "sensitivity",
            "candidate_pool_budget": budget,
            "reference_run_id": "ref-a",
            "comparator_run_id": f"cmp-a-{budget}",
            "raw_two_sided_p_value": raw,
        }
        for budget, raw in ((100, 0.01), (200, 0.02), (300, 0.04))
    ] + [
        {
            "family": "lendingclub_v2_catboost",
            "dataset": "lendingclub_v2",
            "model": "catboost",
            "comparison_label": f"voting_pool_{budget}_vs_rf_corr_mrmr",
            "designation": "primary" if budget == 200 else "sensitivity",
            "candidate_pool_budget": budget,
            "reference_run_id": "ref-b",
            "comparator_run_id": f"cmp-b-{budget}",
            "raw_two_sided_p_value": raw,
        }
        for budget, raw in ((100, 0.30), (200, 0.60), (300, 0.90))
    ]
    audit = apply_holm_families(rows, alpha=0.05)
    assert set(audit["family"]) == {"homecredit_lr", "lendingclub_v2_catboost"}
    assert set(audit["family_size"]) == {3}
    assert not audit["pooling_across_families"].any()
    homecredit = audit.loc[audit["family"] == "homecredit_lr"].sort_values("ordered_rank")
    assert list(homecredit["raw_two_sided_p_value"]) == [0.01, 0.02, 0.04]
    assert list(homecredit["holm_adjusted_p_value"]) == pytest.approx([0.03, 0.04, 0.04])
    assert list(homecredit["holm_threshold"]) == pytest.approx(
        [0.05 / 3, 0.05 / 2, 0.05 / 1]
    )
    assert homecredit["reject_null"].all()
    lending = audit.loc[audit["family"] == "lendingclub_v2_catboost"]
    assert not lending["reject_null"].any()


def test_predeclared_family_is_recovered_from_the_frozen_protocol() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs/protocols/cross_dataset_rank_voting_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    lookup = {
        (dataset, model, configuration): f"{dataset}-{model}-{configuration}"
        for dataset in ("homecredit", "lendingclub_v2")
        for model in ("lr", "catboost")
        for configuration in ("reference", "voting_k100", "voting_k200", "voting_k300")
    }
    comparisons = recover_predeclared_family(protocol, lookup)
    assert len(comparisons) == 12
    assert len({comparison.family for comparison in comparisons}) == 4
    assert sum(comparison.designation == "primary" for comparison in comparisons) == 4
    assert sum(comparison.designation == "sensitivity" for comparison in comparisons) == 8
    for comparison in comparisons:
        assert comparison.metric == "roc_auc"
        assert comparison.label.endswith("_vs_rf_corr_mrmr")
        assert comparison.reference_run_id.endswith("reference")


def test_bootstrap_rejects_an_invalid_repetition_contract() -> None:
    aligned = _aligned(200, seed=31)
    with pytest.raises(ValueError, match="repetition/minimum-valid contract"):
        fast_paired_stratified_bootstrap(aligned, repetitions=5, minimum_valid=9)


def test_bootstrap_rejects_a_single_class_input() -> None:
    aligned = _aligned(200, seed=33)
    aligned["target"] = 1
    with pytest.raises(ValueError, match="both target classes"):
        fast_paired_stratified_bootstrap(aligned, repetitions=3, minimum_valid=1)
