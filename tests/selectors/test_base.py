from __future__ import annotations

import pickle

import pandas as pd
import pytest

from credit_risk_fs.selectors.base import (
    FeatureSelector,
    SelectedFeaturesMixin,
    get_selected_features,
    select_feature_frame,
    set_selected_features,
    validate_feature_frame,
)
from credit_risk_fs.selectors.boruta import BorutaSelector
from credit_risk_fs.selectors.boruta_then_rfe import BorutaThenRFESelector
from credit_risk_fs.selectors.domain_rule_baseline import DomainRuleBaselineSelector
from credit_risk_fs.selectors.fixed_rank_then_mrmr import FixedRankThenMRMRSelector
from credit_risk_fs.selectors.llm_screening import LLMSelector
from credit_risk_fs.selectors.llm_then_stat import LLMThenStatSelector
from credit_risk_fs.selectors.mrmr import RandomForestRelevanceMRMRSelector
from credit_risk_fs.selectors.pca import PCASelector
from credit_risk_fs.selectors.rfe import RFESelector
from credit_risk_fs.selectors.stable_core_llm_fill import StableCoreLLMFillSelector


class _LegacySelector:
    def __init__(self) -> None:
        self.selected_features = ["legacy"]


class _StatSelector(SelectedFeaturesMixin):
    def fit(self, X, y=None):
        self.selected_features_ = []
        return self

    def transform(self, X):
        return X.loc[:, []]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def test_public_selectors_implement_protocol(tmp_path):
    selectors = [
        BorutaSelector(),
        RFESelector(),
        BorutaThenRFESelector(),
        RandomForestRelevanceMRMRSelector(k=1, method="mrmr"),
        PCASelector(),
        LLMSelector(description_csv_path="unused.csv", cache_dir=str(tmp_path)),
        LLMThenStatSelector(
            description_csv_path="unused.csv",
            stat_selector_cls=_StatSelector,
        ),
        StableCoreLLMFillSelector(description_csv_path="unused.csv"),
        DomainRuleBaselineSelector(description_csv_path="unused.csv"),
        FixedRankThenMRMRSelector(
            ranking_path="unused.csv",
            feature_budget=1,
            screening_pool_size=1,
        ),
    ]

    assert all(isinstance(selector, FeatureSelector) for selector in selectors)
    assert all(hasattr(selector, "selected_features_") for selector in selectors)


def test_selected_features_legacy_property_is_bidirectional():
    selector = BorutaSelector()
    selector.selected_features_ = ["canonical"]
    assert selector.selected_features == ["canonical"]

    selector.selected_features = ["legacy_write"]
    assert selector.selected_features_ == ["legacy_write"]


def test_compatibility_adapter_handles_legacy_selector():
    selector = _LegacySelector()
    assert get_selected_features(selector) == ["legacy"]
    assert set_selected_features(selector, ["new"]) == ["new"]
    assert selector.selected_features_ == ["new"]
    assert selector.selected_features == ["new"]


def test_duplicate_and_missing_feature_names_fail_explicitly():
    duplicate = pd.DataFrame([[1, 2]], columns=["same", "same"])
    with pytest.raises(ValueError, match="must be unique"):
        validate_feature_frame(duplicate)

    frame = pd.DataFrame({"present": [1]})
    with pytest.raises(ValueError, match="missing 1 features"):
        select_feature_frame(frame, ["absent"], selector_name="TestSelector")


def test_zero_feature_result_and_serialization_are_supported():
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    selector = RandomForestRelevanceMRMRSelector(k=0, method="mrmr").fit(
        frame,
        pd.Series([0, 1]),
    )
    restored = pickle.loads(pickle.dumps(selector))

    assert restored.selected_features_ == []
    assert restored.selected_features == []
    assert restored.transform(frame).shape == (2, 0)
