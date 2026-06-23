from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk_fs.clip_final_comparison.screening import (
    cap_pool_size,
    correlation_filter_pool,
    cosine_similarity_ranking,
    full_mrmr_universe,
    random_screening_pool,
    type_aware_dispersion_scores,
)


def test_type_aware_dispersion_handles_numeric_categorical_and_missing():
    frame = pd.DataFrame(
        {
            "num": [0.0, 1.0, 2.0, 3.0],
            "cat": ["a", "a", "b", "c"],
            "missing": [None, None, None, None],
        }
    )
    scores = type_aware_dispersion_scores(frame)
    kinds = dict(zip(scores["feature_name"], scores["dispersion_kind"]))
    assert kinds["num"] == "numeric_variance"
    assert kinds["cat"] == "categorical_normalized_entropy"
    assert kinds["missing"] == "all_missing"
    assert set(scores["feature_name"]) == {"num", "cat", "missing"}


def test_random_pool_is_reproducible_and_capped():
    features = ["a", "b", "c", "d"]
    assert random_screening_pool(features, 3, seed=101) == random_screening_pool(features, 3, seed=101)
    assert len(random_screening_pool(features, 20, seed=101)) == 4
    assert cap_pool_size(20, 4) == 4


def test_correlation_filter_uses_dev_only_frame_and_keeps_until_pool_size():
    frame = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1, 2, 3, 4, 5],
            "c": [5, 1, 5, 1, 5],
            "cat": ["x", "x", "y", "z", "z"],
        }
    )
    selected, audit = correlation_filter_pool(frame, 3, threshold=0.99)
    assert "a" in selected or "b" in selected
    assert not {"a", "b"}.issubset(set(selected))
    assert len(selected) == 3
    assert set(audit.columns) == {"feature_name", "max_abs_association", "kept"}


def test_cosine_ranking_and_full_mrmr_universe_are_deterministic():
    vectors = pd.DataFrame({"feature_name": ["x", "y"], "v1": [1.0, 0.0], "v2": [0.0, 1.0]})
    ranked = cosine_similarity_ranking(vectors, np.array([1.0, 0.0]))
    assert ranked["feature_name"].tolist() == ["x", "y"]
    assert full_mrmr_universe(["a", "b", "a"]) == ["a", "b"]

