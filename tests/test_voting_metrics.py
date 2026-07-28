"""Prompt 6 metric-recomputation tests."""

from __future__ import annotations

import numpy as np
import pytest

from credit_risk_fs.analysis.voting_inference.metrics import (
    lift_at_10_audit,
    recompute_discrimination,
)
from scripts.independently_verify_voting_metrics import (
    independent_auc,
    independent_gini,
    independent_ks,
    independent_lift_at_10,
)


def test_known_auc_and_gini_example() -> None:
    # Perfectly separated scores give AUC 1 and Gini 1.
    values = recompute_discrimination([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], ["a", "b", "c", "d"])
    assert values["auc"] == pytest.approx(1.0)
    assert values["gini"] == pytest.approx(1.0)
    # Fully tied scores give AUC 0.5 and Gini 0.
    tied = recompute_discrimination([0, 0, 1, 1], [0.5] * 4, ["a", "b", "c", "d"])
    assert tied["auc"] == pytest.approx(0.5)
    assert tied["gini"] == pytest.approx(0.0)


def test_auc_matches_the_independent_rank_formula() -> None:
    generator = np.random.default_rng(4)
    target = generator.binomial(1, 0.3, 500)
    score = np.round(generator.random(500), 2)
    values = recompute_discrimination(target, score, [str(index) for index in range(500)])
    assert values["auc"] == pytest.approx(independent_auc(target, score), abs=1e-12)
    assert values["gini"] == pytest.approx(independent_gini(values["auc"]), abs=1e-12)


def test_ks_matches_a_direct_calculation() -> None:
    target = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.8])
    values = recompute_discrimination(target, score, ["a", "b", "c", "d", "e"])
    # All negatives fall below all positives, so the CDFs separate completely.
    assert values["ks"] == pytest.approx(1.0)
    assert values["ks"] == pytest.approx(independent_ks(target, score), abs=1e-12)


def test_lift_at_10_definition_and_top_count_rule() -> None:
    target = [1] * 3 + [0] * 27
    score = list(np.linspace(1.0, 0.0, 30))
    identity = [f"{index:03d}" for index in range(30)]
    audit = lift_at_10_audit(target, score, identity)
    assert audit["top_count"] == 3  # ceil(0.10 * 30)
    assert audit["top_count_rule"] == "ceil(fraction * n)"
    assert audit["ratio_definition"] == (
        "top_decile_positive_rate_divided_by_overall_positive_rate"
    )
    assert audit["capture_rate_alternative_used"] is False
    assert audit["overall_positive_rate"] == pytest.approx(0.1)
    assert audit["top_decile_positive_rate"] == pytest.approx(1.0)
    assert audit["lift_at_10"] == pytest.approx(10.0)


def test_lift_at_10_boundary_ties_resolve_by_identity_ascending() -> None:
    # Ten rows, top count 1; the two highest scores tie, so identity decides.
    target = [0, 1] + [0] * 8
    score = [0.9, 0.9] + [0.1] * 8
    identity = ["aaa", "bbb"] + [f"z{index}" for index in range(8)]
    audit = lift_at_10_audit(target, score, identity)
    assert audit["top_count"] == 1
    assert audit["boundary_tie_group_size"] == 2
    assert audit["boundary_tie_group_is_target_homogeneous"] is False
    assert audit["boundary_tie_group_partially_included"] is True
    # "aaa" wins the tie and is a negative, so the top decile captures no positive.
    assert audit["lift_at_10"] == pytest.approx(0.0)
    assert audit["lift_at_10"] == pytest.approx(
        independent_lift_at_10(target, score, identity), abs=1e-12
    )


def test_constant_scores_are_accepted_and_degenerate() -> None:
    values = recompute_discrimination([0, 1, 0, 1], [0.3] * 4, ["a", "b", "c", "d"])
    assert values["auc"] == pytest.approx(0.5)
    assert values["ks"] == pytest.approx(0.0)


def test_single_class_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="both target classes"):
        recompute_discrimination([1, 1, 1], [0.2, 0.5, 0.9], ["a", "b", "c"])
    with pytest.raises(ValueError, match="both classes"):
        independent_auc([0, 0, 0], [0.2, 0.5, 0.9])


def test_non_finite_scores_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        recompute_discrimination([0, 1], [0.5, float("nan")], ["a", "b"])
