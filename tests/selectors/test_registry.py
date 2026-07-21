from __future__ import annotations

from credit_risk_fs.experiments.config import apply_feature_budget_to_selector_kwargs
from credit_risk_fs.selectors.boruta import BorutaSelector
from credit_risk_fs.selectors.boruta_then_rfe import BorutaThenRFESelector
from credit_risk_fs.selectors.mrmr import RandomForestRelevanceMRMRSelector
from credit_risk_fs.selectors.registry import get_selector


def test_boruta_and_boruta_rfe_resolve_to_distinct_semantics():
    boruta_cls, boruta_kwargs = get_selector("boruta")
    combination_cls, combination_kwargs = get_selector("boruta_rfe")

    assert boruta_cls is BorutaSelector
    assert combination_cls is BorutaThenRFESelector
    assert "use_rfe" not in boruta_kwargs
    assert combination_kwargs["use_rfe"] is True


def test_feature_budget_updates_each_boruta_shape_without_enabling_rfe():
    _, boruta_defaults = get_selector("boruta")
    _, combination_defaults = get_selector("boruta_rfe")

    boruta = apply_feature_budget_to_selector_kwargs("boruta", boruta_defaults, 20)
    combination = apply_feature_budget_to_selector_kwargs(
        "boruta_rfe",
        combination_defaults,
        20,
    )

    assert boruta["n_features"] == 20
    assert "rfe_kwargs" not in boruta
    assert combination["n_features"] == 20
    assert combination["rfe_kwargs"]["n_features"] == 20
    assert combination["use_rfe"] is True


def test_mrmr_registry_name_points_to_accurately_named_custom_selector():
    selector_cls, kwargs = get_selector("mrmr")

    assert selector_cls is RandomForestRelevanceMRMRSelector
    assert kwargs["method"] == "mrmr"
    assert selector_cls.canonical_mrmr is False
    assert selector_cls.algorithm_name == "rf_relevance_correlation_redundancy"
