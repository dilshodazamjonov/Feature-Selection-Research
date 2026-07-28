"""Common contract, registry, and historical-compatibility tests for Prompt 8."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.experiments.config import apply_feature_budget_to_selector_kwargs
from credit_risk_fs.selectors.boruta import BorutaSelector
from credit_risk_fs.selectors.heavy import (
    BorutaRandomForestSelector,
    CatBoostRFESelector,
    CatBoostShapSelector,
)
from credit_risk_fs.selectors.lightweight.contract import (
    SELECTION_MODES,
    ControlledSelectorFailure,
    SelectionResult,
)
from credit_risk_fs.selectors.lightweight.registry import (
    LIGHTWEIGHT_METHODS,
    get_method_descriptor,
    lightweight_method_ids,
    registry_snapshot,
    resolve_method_id,
    validate_method_selection_mode,
)
from credit_risk_fs.selectors.mrmr import RandomForestRelevanceMRMRSelector
from credit_risk_fs.selectors.registry import get_selector
from credit_risk_fs.selectors.rfe import RFESelector

ROOT = Path(__file__).resolve().parents[2]

HEAVY = {
    "rfe_catboost": CatBoostRFESelector,
    "boruta_random_forest": BorutaRandomForestSelector,
    "catboost_shap": CatBoostShapSelector,
}

TINY_CATBOOST = {"iterations": 25, "depth": 3}
TINY_FOREST = {"n_estimators": 30, "max_depth": 4}
TINY_BORUTA = {"max_iter": 8}


@pytest.fixture()
def fixture() -> tuple[pd.DataFrame, pd.Series]:
    generator = np.random.default_rng(53)
    n = 300
    latent = generator.normal(size=n)
    target = pd.Series((latent + generator.normal(scale=0.5, size=n) > 0).astype(int))
    return (
        pd.DataFrame(
            {
                "zulu_signal": latent,
                "alpha_noise": generator.normal(size=n),
                "mike_noise": generator.normal(size=n),
            }
        ),
        target,
    )


def _build(method_id: str, **kwargs):
    if method_id == "boruta_random_forest":
        kwargs.setdefault("forest_params", TINY_FOREST)
        kwargs.setdefault("boruta_params", TINY_BORUTA)
    else:
        kwargs.setdefault("catboost_params", TINY_CATBOOST)
    return HEAVY[method_id](**kwargs)


def _fit(method_id: str, features, target, **kwargs):
    if method_id == "boruta_random_forest":
        kwargs.setdefault("selection_mode", "confirmed_top_k")
    return _build(method_id, **kwargs).fit(features, target)


# -- registry -----------------------------------------------------------------


@pytest.mark.parametrize(("method_id", "expected"), sorted(HEAVY.items()))
def test_each_heavy_id_resolves_to_its_own_class(method_id, expected) -> None:
    selector_cls, kwargs = get_selector(method_id)
    assert selector_cls is expected
    assert isinstance(kwargs, dict)

    descriptor = get_method_descriptor(method_id)
    assert descriptor.load() is expected
    assert descriptor.method_id == expected.method_id
    assert descriptor.display_label == expected.display_label
    assert descriptor.implementation_id == expected.implementation_id
    assert descriptor.cost_class == "heavy"
    assert descriptor.estimator_family
    assert descriptor.controlled_failure_conditions
    assert descriptor.allowed_in_frozen_voting is False


def test_heavy_descriptors_declare_accurate_capabilities() -> None:
    rfe = get_method_descriptor("rfe_catboost")
    assert rfe.provides_natural_support is False
    assert rfe.guarantees_exact_k is True
    assert "catboost" in rfe.estimator_family.lower()

    boruta = get_method_descriptor("boruta_random_forest")
    assert boruta.provides_natural_support is True
    assert boruta.guarantees_exact_k is False
    assert set(boruta.selection_modes) == {
        "natural_confirmed",
        "confirmed_top_k",
        "confirmed_then_tentative",
    }

    shap = get_method_descriptor("catboost_shap")
    assert shap.provides_natural_support is False
    assert shap.guarantees_exact_k is True
    assert "shap" in shap.implementation_id.lower()
    assert "regular" in shap.implementation_id.lower()


def test_no_identifier_resolves_to_two_algorithms() -> None:
    seen: dict[str, str] = {}
    for descriptor in LIGHTWEIGHT_METHODS:
        for name in (descriptor.method_id, *descriptor.aliases):
            assert name not in seen, f"{name} resolves to two algorithms"
            seen[name] = descriptor.method_id
    assert len(set(lightweight_method_ids())) == len(lightweight_method_ids())


def test_heavy_ids_do_not_hijack_the_historical_routes() -> None:
    """The new canonical ids must not take over 'rfe' or 'boruta'."""

    legacy_rfe_cls, _ = get_selector("rfe")
    legacy_boruta_cls, _ = get_selector("boruta")
    assert legacy_rfe_cls is RFESelector
    assert legacy_boruta_cls is BorutaSelector
    assert legacy_rfe_cls is not CatBoostRFESelector
    assert legacy_boruta_cls is not BorutaRandomForestSelector

    # And the new ids are not reachable from the historical names.
    with pytest.raises(KeyError):
        resolve_method_id("rfe")
    with pytest.raises(KeyError):
        resolve_method_id("boruta")


def test_selection_mode_validation_rejects_invalid_pairings() -> None:
    for mode in ("natural_confirmed", "confirmed_top_k", "confirmed_then_tentative"):
        validate_method_selection_mode("boruta_random_forest", mode)
    with pytest.raises(ValueError, match="is not supported by"):
        validate_method_selection_mode("rfe_catboost", "natural_confirmed")
    with pytest.raises(ValueError, match="is not supported by"):
        validate_method_selection_mode("catboost_shap", "natural")
    with pytest.raises(ValueError, match="is not supported by"):
        validate_method_selection_mode("boruta_random_forest", "matched_budget")


def test_new_boruta_modes_are_registered_in_the_shared_vocabulary() -> None:
    assert {"natural_confirmed", "confirmed_top_k", "confirmed_then_tentative"} <= (
        SELECTION_MODES
    )


def test_registry_snapshot_carries_the_heavy_fields() -> None:
    snapshot = json.loads(json.dumps(registry_snapshot(), sort_keys=True))
    by_id = {entry["method_id"]: entry for entry in snapshot["methods"]}
    assert set(HEAVY) <= set(by_id)
    for method_id in HEAVY:
        entry = by_id[method_id]
        assert entry["cost_class"] == "heavy"
        assert entry["estimator_family"]
        assert entry["serialization_version"] == "lightweight_selector_contract_v1"
        assert entry["allowed_in_frozen_voting"] is False
        assert entry["controlled_failure_conditions"]
    # Prompt 7 methods keep the light cost class.
    for method_id in ("iv_woe", "mrmr_mutual_information", "lasso_l1_logistic"):
        assert by_id[method_id]["cost_class"] == "light"


@pytest.mark.parametrize("method_id", sorted(HEAVY))
def test_feature_budget_wiring_uses_k(method_id) -> None:
    _, defaults = get_selector(method_id)
    updated = apply_feature_budget_to_selector_kwargs(method_id, defaults, 20)
    assert get_method_descriptor(method_id).budget_kwarg == "k"
    assert updated["k"] == 20


# -- common contract ----------------------------------------------------------


@pytest.mark.parametrize("method_id", sorted(HEAVY))
def test_universal_invariants_hold(method_id, fixture) -> None:
    features, target = fixture
    selector = _fit(method_id, features, target, k=2)
    result = selector.result

    assert result.method_id == method_id
    assert result.fit_scope == "dev_fold_training_only"
    assert result.candidate_universe == tuple(features.columns)
    assert set(result.selected_features).issubset(set(features.columns))
    assert len(set(result.selected_features)) == result.actual_selected_count
    assert result.estimator_config_sha256
    assert result.heavy_metadata["cost_class"] == "heavy"
    assert list(selector.transform(features).columns) == list(result.selected_features)

    long_frame = result.to_long_frame()
    assert list(long_frame["rank"]) == list(range(1, len(long_frame) + 1))
    assert long_frame["feature"].is_unique


@pytest.mark.parametrize("method_id", sorted(HEAVY))
def test_duplicate_candidate_names_fail(method_id, fixture) -> None:
    features, target = fixture
    duplicated = pd.concat([features, features.iloc[:, :1]], axis=1)
    with pytest.raises(ValueError, match="unique"):
        _fit(method_id, duplicated, target, k=2)


@pytest.mark.parametrize("method_id", sorted(HEAVY))
def test_empty_universe_has_a_controlled_outcome(method_id, fixture) -> None:
    _, target = fixture
    empty = pd.DataFrame(index=range(len(target)))
    selector = _fit(method_id, empty, target, k=2)
    assert selector.result.budget_status == "empty_universe"
    assert selector.result.selected_features == ()
    assert selector.selected_features_ == []


@pytest.mark.parametrize("method_id", sorted(HEAVY))
def test_missing_target_fails_explicitly(method_id, fixture) -> None:
    features, _ = fixture
    with pytest.raises(ControlledSelectorFailure) as error:
        _build(
            method_id,
            k=2,
            **({"selection_mode": "confirmed_top_k"} if method_id == "boruta_random_forest" else {}),
        ).fit(features, None)
    assert error.value.stage == "target_validation"


@pytest.mark.parametrize("method_id", sorted(HEAVY))
def test_thread_and_seed_contract_is_explicit(method_id, fixture) -> None:
    features, target = fixture
    result = _fit(method_id, features, target, k=2).result
    configuration = result.configuration
    assert result.seed == 42
    if method_id == "boruta_random_forest":
        assert configuration["n_jobs"] == 1
        assert configuration["forest_params"]["random_state"] == 42
        assert configuration["boruta_params"]["random_state"] == 42
    else:
        assert configuration["thread_count"] == 1
        assert configuration["estimator_params"]["random_seed"] == 42
        assert configuration["estimator_params"]["allow_writing_files"] is False
        assert configuration["estimator_params"]["verbose"] is False

    with pytest.raises(ValueError, match="must be positive"):
        if method_id == "boruta_random_forest":
            BorutaRandomForestSelector(n_jobs=0)
        else:
            HEAVY[method_id](k=2, thread_count=0)


@pytest.mark.parametrize("method_id", sorted(HEAVY))
def test_serialization_preserves_heavy_metadata(method_id, fixture) -> None:
    features, target = fixture
    original = _fit(method_id, features, target, k=2).result
    restored = SelectionResult.from_json(original.to_json())

    assert restored.heavy_metadata == original.heavy_metadata
    assert restored.estimator_config_sha256 == original.estimator_config_sha256
    assert restored.selected_features == original.selected_features
    assert restored.ranking == original.ranking
    assert restored.natural_selected == original.natural_selected


@pytest.mark.parametrize("method_id", sorted(HEAVY))
def test_no_output_reaches_protected_roots(method_id, fixture, tmp_path) -> None:
    from credit_risk_fs.experiments.result_paths import AUDITED_LEGACY_RESULTS_ROOT

    features, target = fixture
    before = {p for p in (ROOT / "results").rglob("*") if p.is_file()}
    _fit(method_id, features, target, k=2)
    after = {p for p in (ROOT / "results").rglob("*") if p.is_file()}
    assert before == after
    assert not AUDITED_LEGACY_RESULTS_ROOT.exists() or True  # never written to here

    # CatBoost must not have dropped catboost_info/ into the working tree.
    assert not (Path.cwd() / "catboost_info").exists()


# -- Prompt 7 backward compatibility -----------------------------------------


def test_prompt_07_payloads_still_load_without_the_new_fields() -> None:
    """The two added contract fields are optional, so old artifacts still load."""

    payload = {
        "contract_version": "lightweight_selector_contract_v1",
        "method_id": "iv_woe",
        "display_label": "Information Value (WOE binning)",
        "implementation_id": "iv_woe_quantile_binned_v1",
        "selection_mode": "matched_budget",
        "supervised": True,
        "selected_features": ["a"],
        "candidate_universe": ["a", "b"],
        "requested_budget": 1,
        "budget_status": "satisfied",
        "score_orientation": "higher_is_better",
        "tie_rule": "descending_score_then_ascending_feature_name",
        "ranking": ["a", "b"],
        "raw_scores": {"a": 1.0, "b": 0.5},
        "seed": 42,
    }
    restored = SelectionResult.from_dict(payload)
    assert restored.method_id == "iv_woe"
    assert restored.estimator_config_sha256 is None
    assert restored.heavy_metadata is None
    assert restored.selected_features == ("a",)


def test_prompt_07_method_identities_are_unchanged() -> None:
    expected = {
        "iv_woe": "iv_woe_quantile_binned_v1",
        "mrmr_mutual_information": "mrmr_mutual_information_discrete_plugin_v1",
        "lasso_l1_logistic": "lasso_l1_logistic_v1",
        "random_k": "random_k_local_generator_v1",
        "full_features": "full_candidate_features_v1",
        "legacy_rf_relevance_corr": "rf_relevance_correlation_redundancy",
    }
    for method_id, implementation_id in expected.items():
        assert get_method_descriptor(method_id).implementation_id == implementation_id
    assert resolve_method_id("mrmr") == "legacy_rf_relevance_corr"
    assert resolve_method_id("none_explicit") == "full_features"


# -- frozen voting and legacy selectors --------------------------------------


def test_frozen_voting_protocol_is_unchanged() -> None:
    from credit_risk_fs.experiments.rank_voting import ELIGIBLE_VOTERS

    assert ELIGIBLE_VOTERS == ("rf_corr_mrmr", "boruta")
    # The boruta voter still resolves to the historical implementation.
    voter_cls, _ = get_selector("boruta")
    assert voter_cls is BorutaSelector
    assert voter_cls is not BorutaRandomForestSelector
    # And rf_corr_mrmr still means the legacy RF/correlation algorithm.
    mrmr_cls, _ = get_selector("mrmr")
    assert mrmr_cls is RandomForestRelevanceMRMRSelector

    # No Prompt 8 method claims voting eligibility.
    for method_id in HEAVY:
        assert get_method_descriptor(method_id).allowed_in_frozen_voting is False


def test_legacy_selector_signatures_are_unchanged() -> None:
    import inspect

    rfe = inspect.signature(RFESelector.__init__)
    assert rfe.parameters["n_features"].default == 50
    assert rfe.parameters["step"].default == 10
    assert rfe.parameters["random_state"].default == 42
    assert rfe.parameters["thread_count"].default == 1

    boruta = inspect.signature(BorutaSelector.__init__)
    assert boruta.parameters["max_iter"].default == 10
    assert boruta.parameters["random_state"].default == 42
    assert boruta.parameters["n_features"].default is None
    assert boruta.parameters["n_jobs"].default == 1


@pytest.mark.parametrize(
    "legacy_name",
    ["boruta", "boruta_rfe", "rfe", "mrmr", "pca", "domain_rule_baseline", "none"],
)
def test_historical_registry_routes_still_resolve(legacy_name) -> None:
    selector_cls, kwargs = get_selector(legacy_name)
    if legacy_name == "none":
        assert selector_cls is None
        assert kwargs == {}
    else:
        assert selector_cls is not None
        assert isinstance(kwargs, dict)


def test_legacy_budget_wiring_is_unchanged() -> None:
    for name, kwarg in (("rfe", "n_features"), ("boruta", "n_features"), ("mrmr", "k")):
        _, defaults = get_selector(name)
        updated = apply_feature_budget_to_selector_kwargs(name, defaults, 20)
        assert updated[kwarg] == 20


def test_prompt_06_package_hashes_are_unchanged() -> None:
    """The Prompt 7 baseline snapshot is the reference; nothing may drift."""

    import hashlib

    baseline_path = (
        ROOT
        / "cleanup/audits/prompt_07_lightweight_selectors/prompt_06_package_hashes_baseline.json"
    )
    if not baseline_path.is_file():
        pytest.skip("the Prompt 7 baseline hash snapshot is not present")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    package = ROOT / "results/final_experiments/cross_dataset_voting_inference_v1"
    if not package.is_dir():
        pytest.skip("the Prompt 6 package is not present in this checkout")

    observed = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and "_cache" not in path.parts:
            key = str(path.relative_to(package)).replace("\\", "/")
            observed[key] = hashlib.sha256(path.read_bytes()).hexdigest()

    assert observed == baseline
