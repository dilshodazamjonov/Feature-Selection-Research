from __future__ import annotations

import numpy as np
import pandas as pd

import credit_risk_fs.selectors.mrmr as mrmr_module
from credit_risk_fs.selectors.mrmr import RandomForestRelevanceMRMRSelector


def test_rf_relevance_ties_and_iteration_seeds_are_deterministic(monkeypatch):
    seeds: list[int] = []

    class FakeRandomForest:
        def __init__(self, **kwargs) -> None:
            seeds.append(kwargs["random_state"])

        def fit(self, X, y):
            self.feature_importances_ = np.ones(X.shape[1], dtype=float)
            return self

    monkeypatch.setattr(mrmr_module, "RandomForestClassifier", FakeRandomForest)
    X = pd.DataFrame({"z": [0, 1], "a": [1, 0], "m": [2, 3]})

    selector = RandomForestRelevanceMRMRSelector(
        k=2,
        method="rf",
        n_iter=3,
        random_state=7,
    ).fit(X, pd.Series([0, 1]))

    assert seeds == [7, 8, 9]
    assert selector.selected_features_ == ["a", "m"]


def test_budget_larger_than_available_preserves_input_order():
    X = pd.DataFrame({"z": [0, 1], "a": [1, 0]})
    selector = RandomForestRelevanceMRMRSelector(k=50, method="mrmr").fit(
        X,
        pd.Series([0, 1]),
    )

    assert selector.selected_features_ == ["z", "a"]
