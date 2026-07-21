from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.evaluation.paired_inference import (
    align_paired_predictions,
    holm_adjust,
    ks_statistic,
    lift_at_fraction,
    paired_delong_test,
    paired_stratified_bootstrap,
)


def _predictions(scores, *, reverse=False):
    frame = pd.DataFrame(
        {
            "stable_row_id": [f"row-{index:02d}" for index in range(20)],
            "target": [0] * 10 + [1] * 10,
            "prediction_probability": scores,
        }
    )
    return frame.iloc[::-1].reset_index(drop=True) if reverse else frame


def test_pairing_is_identity_based_and_invariant_to_row_reordering():
    weak = np.linspace(0.2, 0.8, 20)
    aligned = align_paired_predictions(_predictions(weak), _predictions(weak, reverse=True))
    assert len(aligned) == 20
    assert np.array_equal(aligned["score_a"], aligned["score_b"])
    assert paired_delong_test(aligned["target"], aligned["score_a"], aligned["score_b"])[
        "two_sided_p_value"
    ] == 1.0


def test_pairing_rejects_mismatched_ids_targets_and_duplicates():
    scores = np.linspace(0.2, 0.8, 20)
    left = _predictions(scores)
    with pytest.raises(ValueError, match="identity sets differ"):
        align_paired_predictions(left, left.iloc[:-1])
    with pytest.raises(ValueError, match="targets disagree"):
        align_paired_predictions(left, left.assign(target=[1] + left["target"].tolist()[1:]))
    duplicated = pd.concat([left, left.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated"):
        align_paired_predictions(duplicated, left)


def test_score_direction_delong_ks_and_lift_favor_higher_default_scores():
    target = np.array([0] * 10 + [1] * 10)
    strong = np.array([0.1] * 10 + [0.9] * 10)
    reversed_score = 1 - strong
    result = paired_delong_test(target, strong, reversed_score)
    assert result["auc_difference_a_minus_b"] == 1.0
    assert result["two_sided_p_value"] <= 0.05
    assert ks_statistic(target, strong)[0] == 1.0
    ids = [f"id-{index}" for index in range(20)]
    assert lift_at_fraction(target, strong, ids) == 2.0


def test_ks_uses_smallest_score_when_separation_ties():
    statistic, threshold = ks_statistic([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4])
    assert statistic == 0.5
    assert threshold == 0.1


def test_stratified_bootstrap_is_paired_fixed_attempt_and_reproducible():
    target = np.array([0] * 10 + [1] * 10)
    strong = np.array([0.1] * 10 + [0.9] * 10)
    weak = np.linspace(0.2, 0.8, 20)
    aligned = align_paired_predictions(_predictions(strong), _predictions(weak, reverse=True))
    first = paired_stratified_bootstrap(aligned)
    second = paired_stratified_bootstrap(aligned)
    assert first == second
    assert first["attempted_repetitions"] == 2000
    assert first["valid_repetitions"] == 2000
    assert first["failed_repetitions"] == 0
    assert all(value["interval_valid"] for value in first["metrics"].values())


def test_holm_order_and_degenerate_class_failures():
    assert holm_adjust([0.01, 0.04, 0.02]) == pytest.approx([0.03, 0.04, 0.04])
    with pytest.raises(ValueError, match="both binary target classes"):
        paired_delong_test([0, 0, 0, 0], [0.1] * 4, [0.2] * 4)
    degenerate = pd.DataFrame(
        {
            "stable_row_id": ["a", "b"],
            "target": [1, 1],
            "score_a": [0.8, 0.9],
            "score_b": [0.7, 0.8],
        }
    )
    with pytest.raises(ValueError, match="both target classes"):
        paired_stratified_bootstrap(degenerate)
