import itertools

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.evaluation.stability import (
    candidate_universe_from_frozen_pool,
    kuncheva_stability,
    mean_pairwise_jaccard,
    nogueira_stability,
    semantic_group_frequency_frame,
    selection_frequency_frame,
    write_feature_stability_artifacts,
)
from credit_risk_fs.evaluation.drift import calculate_psi, jaccard_similarity
from credit_risk_fs.pipelines.common import credit_risk_utility


def test_nogueira_stability_identical_sets_is_one():
    sets = [{"a", "b"}, {"a", "b"}, {"a", "b"}]

    assert nogueira_stability(sets, total_features=5) == 1.0


def test_nogueira_stability_uses_full_feature_universe():
    sets = [{"a", "b"}, {"a", "c"}]

    assert nogueira_stability(sets, total_features=4) == 0.0


def test_nogueira_stability_matches_kuncheva_for_fixed_size_sets():
    sets = [{"a", "b"}, {"a", "c"}, {"a", "d"}]

    assert nogueira_stability(sets, total_features=4) == kuncheva_stability(
        sets,
        total_features=4,
    )


def test_nogueira_stability_lower_bound_for_disjoint_sets():
    sets = [{"a", "b"}, {"c", "d"}, {"e", "f"}]

    assert np.isclose(nogueira_stability(sets, total_features=6), -0.5)


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


def test_semantic_group_stability_computes_correctly(tmp_path):
    features_dir = tmp_path / "exp" / "features"
    features_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"fold_id": 1, "feature_name": "EXT_SOURCE_1", "semantic_group": "external_score"},
            {"fold_id": 1, "feature_name": "BURO_AMT_CREDIT_SUM_DEBT_MEAN", "semantic_group": "bureau_debt"},
            {"fold_id": 2, "feature_name": "EXT_SOURCE_2", "semantic_group": "external_score"},
            {"fold_id": 2, "feature_name": "BURO_DAYS_CREDIT_MAX", "semantic_group": "bureau_credit_history"},
        ]
    ).to_csv(features_dir / "fold_selected_features.csv", index=False)

    metrics = write_feature_stability_artifacts(
        exp_dir=tmp_path / "exp",
        model="lr",
        selector="llm",
        total_candidate_features=10,
    )
    semantic_df = pd.read_csv(features_dir / "semantic_group_stability.csv")

    external_row = semantic_df[semantic_df["semantic_group"] == "external_score"].iloc[0]
    assert external_row["selection_frequency"] == 1.0
    assert metrics["semantic_group_jaccard"] == 1 / 3
    assert metrics["stable_semantic_group_count_80"] == 1
    assert metrics["semantic_group_stable_ratio_80"] == 1 / 3


def _write_candidate_pool(path, size):
    pd.DataFrame(
        {
            "feature_name": [f"f{index}" for index in range(size)],
            "candidate_pool_size": [size] * size,
            "candidate_pool_frozen_before_mrmr": [True] * size,
        }
    ).to_csv(path, index=False)


@pytest.mark.parametrize(("size", "budget"), [(60, 20), (100, 40)])
def test_stability_universe_comes_from_frozen_candidate_pool(
    tmp_path, size, budget
):
    pool = tmp_path / "candidate_pool.csv"
    _write_candidate_pool(pool, size)
    selected = [{f"f{index}" for index in range(budget)} for _ in range(5)]

    assert candidate_universe_from_frozen_pool(
        pool, selected_sets=selected
    ) == size


def test_full_model_universe_is_rejected_for_reverse_transfer_pool(tmp_path):
    pool = tmp_path / "candidate_pool.csv"
    _write_candidate_pool(pool, 529)
    frame = pd.read_csv(pool).iloc[:60].copy()
    frame["candidate_pool_size"] = 60
    frame.to_csv(pool, index=False)

    assert candidate_universe_from_frozen_pool(pool) == 60
    with pytest.raises(ValueError, match="declared size"):
        invalid = pd.read_csv(pool)
        invalid["candidate_pool_size"] = 529
        invalid.to_csv(pool, index=False)
        candidate_universe_from_frozen_pool(pool)


def test_nogueira_matches_independent_indicator_implementation():
    selected = [
        {"a", "b", "c"},
        {"a", "b", "d"},
        {"a", "b", "e"},
        {"a", "c", "d"},
        {"a", "b", "c"},
    ]
    universe = sorted(set().union(*selected) | {"f", "g"})
    matrix = np.array(
        [[feature in fold for feature in universe] for fold in selected],
        dtype=float,
    )
    sample_variances = matrix.var(axis=0, ddof=1)
    average_size = matrix.sum(axis=1).mean()
    expected = 1.0 - (
        sample_variances.mean()
        / ((average_size / len(universe)) * (1 - average_size / len(universe)))
    )

    assert np.isclose(nogueira_stability(selected, len(universe)), expected)


def test_kuncheva_matches_independent_pairwise_implementation():
    selected = [
        {"a", "b", "c"},
        {"a", "b", "d"},
        {"a", "c", "e"},
    ]
    universe_size = 8
    values = []
    for left, right in itertools.combinations(selected, 2):
        k = len(left)
        values.append(
            (len(left & right) * universe_size - k * k)
            / (k * (universe_size - k))
        )

    assert np.isclose(kuncheva_stability(selected, universe_size), np.mean(values))


def test_jaccard_does_not_depend_on_candidate_universe():
    selected = [{"a", "b"}, {"a", "c"}, {"a", "d"}]

    before = mean_pairwise_jaccard(selected)
    nogueira_stability(selected, 529)
    nogueira_stability(selected, 60)
    assert mean_pairwise_jaccard(selected) == before


def test_empty_semantic_groups_are_explicitly_empty(tmp_path):
    features_dir = tmp_path / "exp" / "features"
    features_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"fold_id": 1, "feature_name": "a", "semantic_group": None},
            {"fold_id": 2, "feature_name": "b", "semantic_group": None},
        ]
    ).to_csv(features_dir / "fold_selected_features.csv", index=False)

    metrics = write_feature_stability_artifacts(
        exp_dir=tmp_path / "exp",
        model="lr",
        selector="test",
        total_candidate_features=10,
    )

    assert pd.read_csv(features_dir / "semantic_group_stability.csv").empty
    assert np.isnan(metrics["semantic_group_stable_ratio_80"])
