import numpy as np
import pandas as pd

from evaluation.feature_stability import (
    kuncheva_stability,
    mean_pairwise_jaccard,
    nogueira_stability,
    selection_frequency_frame,
)
from evaluation.stability_scores import calculate_psi, jaccard_similarity
from pipelines.common import credit_risk_utility


def test_nogueira_stability_identical_sets_is_one():
    sets = [{"a", "b"}, {"a", "b"}, {"a", "b"}]

    assert nogueira_stability(sets, total_features=5) == 1.0


def test_kuncheva_stability_fixed_size_sets():
    sets = [{"a", "b"}, {"a", "c"}]
    expected = (1 * 4 - 2**2) / (2 * (4 - 2))

    assert kuncheva_stability(sets, total_features=4) == expected


def test_pairwise_jaccard_cases():
    assert jaccard_similarity({"a"}, {"a"}) == 1.0
    assert jaccard_similarity({"a"}, {"b"}) == 0.0
    assert mean_pairwise_jaccard([{"a", "b"}, {"a", "c"}]) == 1 / 3


def test_selection_frequency_frame_counts_folds():
    tables = [
        pd.DataFrame({"fold_id": [1, 1], "feature_name": ["a", "b"], "rank": [1, 2]}),
        pd.DataFrame({"fold_id": [2, 2], "feature_name": ["a", "c"], "rank": [2, 1]}),
    ]

    freq = selection_frequency_frame(tables)
    row_a = freq[freq["feature_name"] == "a"].iloc[0]

    assert row_a["selection_count"] == 2
    assert row_a["selection_frequency"] == 1.0
    assert row_a["mean_rank_if_available"] == 1.5


def test_psi_near_zero_for_identical_and_higher_for_shifted():
    base = pd.Series(np.arange(100, dtype=float))
    identical = calculate_psi(base, base)
    shifted = calculate_psi(base, base + 100)

    assert identical < 1e-9
    assert shifted > identical


def test_credit_risk_lift_and_capture_at_10():
    y_true = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    scores = np.array([0.99, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    metrics = credit_risk_utility(y_true, scores, top_fracs=(0.1,))

    assert metrics["lift_at_10"] == 5.0
    assert metrics["bad_rate_capture_at_10"] == 0.5
