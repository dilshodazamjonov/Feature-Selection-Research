from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import credit_risk_fs.selectors.boruta as boruta_module
from credit_risk_fs.selectors.boruta import BorutaSelector
from credit_risk_fs.selectors.rfe import RFESelector


class _FakeBoruta:
    support = np.array([True, True, True])
    ranking = np.array([1, 1, 2])

    def __init__(self, **kwargs) -> None:
        self.support_ = self.support.copy()
        self.ranking_ = self.ranking.copy()

    def fit(self, X, y):
        return self


def test_boruta_ties_are_ordered_by_feature_name_and_budget_is_clamped(monkeypatch):
    monkeypatch.setattr(boruta_module, "BorutaPy", _FakeBoruta)
    X = pd.DataFrame({"b": [0, 1], "a": [1, 0], "c": [2, 3]})

    selector = BorutaSelector(n_features=10, random_state=7).fit(
        X,
        pd.Series([0, 1]),
    )

    assert selector.random_state == 7
    assert selector.selected_features_ == ["a", "b", "c"]


def test_boruta_zero_feature_outcome_is_not_backfilled(monkeypatch):
    class NoSupportBoruta(_FakeBoruta):
        support = np.array([False, False, False])

    monkeypatch.setattr(boruta_module, "BorutaPy", NoSupportBoruta)
    X = pd.DataFrame({"a": [0, 1], "b": [1, 0], "c": [2, 3]})
    selector = BorutaSelector(n_features=2).fit(X, pd.Series([0, 1]))

    assert selector.selected_features_ == []
    assert selector.transform(X).shape == (2, 0)


def test_rfe_budget_larger_than_width_selects_all_without_fitting_model():
    X = pd.DataFrame({"b": [0, 1], "a": [1, 0]})
    selector = RFESelector(n_features=20).fit(X, pd.Series([0, 1]))

    assert selector.selected_features_ == ["b", "a"]
    assert list(selector.transform(X).columns) == ["b", "a"]


def test_legacy_boruta_module_imports_remain_available():
    with pytest.deprecated_call():
        legacy_rfe = boruta_module.RFESelector
    with pytest.deprecated_call():
        legacy_combination = boruta_module.BorutaRFESelector

    assert legacy_rfe is RFESelector
    assert legacy_combination.__name__ == "BorutaRFESelector"
