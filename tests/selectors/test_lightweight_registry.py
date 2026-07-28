"""Registry, configuration, and historical-compatibility tests for Prompt 7."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.experiments.config import apply_feature_budget_to_selector_kwargs
from credit_risk_fs.selectors.lightweight import (
    FullCandidateFeaturesSelector,
    InformationValueSelector,
    L1LogisticSelector,
    MutualInformationMRMRSelector,
    RandomKSelector,
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

CANONICAL = {
    "iv_woe": InformationValueSelector,
    "mrmr_mutual_information": MutualInformationMRMRSelector,
    "lasso_l1_logistic": L1LogisticSelector,
    "random_k": RandomKSelector,
    "full_features": FullCandidateFeaturesSelector,
}


@pytest.mark.parametrize(("method_id", "expected"), sorted(CANONICAL.items()))
def test_each_canonical_id_resolves_to_its_own_class(method_id, expected) -> None:
    selector_cls, kwargs = get_selector(method_id)
    assert selector_cls is expected
    assert isinstance(kwargs, dict)

    descriptor = get_method_descriptor(method_id)
    assert descriptor.load() is expected
    assert descriptor.method_id == expected.method_id
    assert descriptor.display_label == expected.display_label
    assert descriptor.implementation_id == expected.implementation_id
    assert descriptor.supervised == expected.supervised


def test_descriptor_capability_flags_match_the_implementations() -> None:
    for descriptor in LIGHTWEIGHT_METHODS:
        implementation = descriptor.load()
        if descriptor.method_id == "legacy_rf_relevance_corr":
            continue  # the legacy class predates the capability declarations
        assert descriptor.provides_ranking == implementation.supports_ranking
        assert descriptor.provides_natural_support == (
            implementation.supports_natural_support
        )
        assert descriptor.supports_fixed_budget == implementation.supports_fixed_budget
        assert implementation.default_selection_mode in descriptor.selection_modes


def test_no_identifier_resolves_to_two_algorithms() -> None:
    seen: dict[str, str] = {}
    for descriptor in LIGHTWEIGHT_METHODS:
        for name in (descriptor.method_id, *descriptor.aliases):
            assert name not in seen, f"{name} resolves to two algorithms"
            seen[name] = descriptor.method_id
    assert len(set(lightweight_method_ids())) == len(lightweight_method_ids())


def test_unknown_selector_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported selector"):
        get_selector("definitely_not_a_selector")
    with pytest.raises(KeyError, match="unknown selection method"):
        resolve_method_id("definitely_not_a_selector")


# -- the legacy mRMR question -------------------------------------------------


def test_legacy_mrmr_alias_still_loads_the_legacy_algorithm() -> None:
    """Historical configs naming 'mrmr' must keep their original algorithm."""

    alias_cls, alias_kwargs = get_selector("mrmr")
    canonical_cls, canonical_kwargs = get_selector("legacy_rf_relevance_corr")

    assert alias_cls is RandomForestRelevanceMRMRSelector
    assert canonical_cls is RandomForestRelevanceMRMRSelector
    assert alias_kwargs == canonical_kwargs
    assert alias_kwargs["method"] == "mrmr"
    assert alias_cls.canonical_mrmr is False
    assert alias_cls.algorithm_name == "rf_relevance_correlation_redundancy"

    assert resolve_method_id("mrmr") == "legacy_rf_relevance_corr"


def test_canonical_mi_mrmr_is_unreachable_from_the_legacy_alias() -> None:
    legacy_cls, _ = get_selector("mrmr")
    canonical_cls, _ = get_selector("mrmr_mutual_information")

    assert legacy_cls is not canonical_cls
    assert legacy_cls is not MutualInformationMRMRSelector
    assert canonical_cls is MutualInformationMRMRSelector
    # And the reverse: the canonical ID never yields the legacy class.
    assert resolve_method_id("mrmr_mutual_information") == "mrmr_mutual_information"
    assert resolve_method_id("mrmr") != "mrmr_mutual_information"


def test_legacy_descriptor_records_which_history_used_it() -> None:
    descriptor = get_method_descriptor("legacy_rf_relevance_corr")
    assert "rf_corr_mrmr" in descriptor.historical_use
    assert "cross_dataset_rank_voting_v1" in descriptor.historical_use
    assert "NOT canonical" in descriptor.notes
    assert descriptor.implementation_id == "rf_relevance_correlation_redundancy"


def test_new_artifacts_record_the_canonical_legacy_identity() -> None:
    """Resolving the old alias still reports the accurate algorithm name."""

    descriptor = get_method_descriptor("mrmr")
    assert descriptor.method_id == "legacy_rf_relevance_corr"
    assert descriptor.display_label == "Legacy RF relevance / correlation redundancy"
    assert "mrmr" not in descriptor.implementation_id


# -- budget wiring ------------------------------------------------------------


@pytest.mark.parametrize(
    "method_id", ["iv_woe", "mrmr_mutual_information", "lasso_l1_logistic", "random_k"]
)
def test_feature_budget_maps_onto_the_declared_budget_kwarg(method_id) -> None:
    _, defaults = get_selector(method_id)
    updated = apply_feature_budget_to_selector_kwargs(method_id, defaults, 20)
    descriptor = get_method_descriptor(method_id)
    assert descriptor.budget_kwarg == "k"
    assert updated["k"] == 20
    # The original defaults are not mutated in place.
    assert defaults.get("k") != 20 or defaults is not updated


def test_full_features_budget_request_is_not_written_into_kwargs() -> None:
    _, defaults = get_selector("full_features")
    updated = apply_feature_budget_to_selector_kwargs("full_features", defaults, 20)
    assert "k" not in updated
    assert get_method_descriptor("full_features").budget_kwarg is None


def test_legacy_mrmr_budget_wiring_is_unchanged() -> None:
    _, defaults = get_selector("mrmr")
    updated = apply_feature_budget_to_selector_kwargs("mrmr", defaults, 20)
    assert updated["k"] == 20
    assert updated["method"] == "mrmr"


# -- config round-trip and validation ----------------------------------------


@pytest.mark.parametrize("method_id", sorted(CANONICAL))
def test_registry_defaults_construct_and_fit_through_the_real_path(method_id) -> None:
    generator = np.random.default_rng(4)
    n = 300
    latent = generator.normal(size=n)
    target = pd.Series((latent > 0).astype(int))
    frame = pd.DataFrame(
        {"latent": latent, "noise": generator.normal(size=n), "extra": generator.normal(size=n)}
    )

    selector_cls, kwargs = get_selector(method_id)
    kwargs = apply_feature_budget_to_selector_kwargs(method_id, kwargs, 2)
    selector = selector_cls(**kwargs)
    selector.fit(frame, target if selector.supervised else None)

    result = selector.result
    assert result.method_id == method_id
    assert set(result.selected_features).issubset(set(frame.columns))
    validate_method_selection_mode(method_id, result.selection_mode)


def test_invalid_method_and_mode_pairing_fails_before_expensive_work() -> None:
    with pytest.raises(ValueError, match="is not supported by"):
        validate_method_selection_mode("iv_woe", "natural")
    with pytest.raises(ValueError, match="is not supported by"):
        validate_method_selection_mode("full_features", "matched_budget")
    with pytest.raises(ValueError, match="is not supported by"):
        validate_method_selection_mode("random_k", "coefficient_ranking")

    # LASSO is the only method allowed to reach the padded ranking mode.
    validate_method_selection_mode("lasso_l1_logistic", "coefficient_ranking")
    validate_method_selection_mode("lasso_l1_logistic", "natural")


def test_registry_snapshot_is_serializable_audit_evidence() -> None:
    import json

    snapshot = registry_snapshot()
    text = json.dumps(snapshot, sort_keys=True)
    restored = json.loads(text)

    assert restored["contract_version"] == "lightweight_selector_contract_v1"
    ids = {entry["method_id"] for entry in restored["methods"]}
    assert ids == set(lightweight_method_ids())
    for entry in restored["methods"]:
        assert entry["display_label"]
        assert entry["implementation_id"]
        assert entry["algorithm"]
        assert entry["selection_modes"]


# -- untouched historical routes ---------------------------------------------


@pytest.mark.parametrize(
    "legacy_name", ["boruta", "boruta_rfe", "rfe", "pca", "domain_rule_baseline", "none"]
)
def test_pre_existing_registry_routes_still_resolve(legacy_name) -> None:
    selector_cls, kwargs = get_selector(legacy_name)
    if legacy_name == "none":
        assert selector_cls is None
        assert kwargs == {}
    else:
        assert selector_cls is not None
        assert isinstance(kwargs, dict)


def test_voting_protocol_voters_are_unaffected() -> None:
    """The frozen voting protocol must still name exactly its two voters."""

    from credit_risk_fs.experiments.rank_voting import ELIGIBLE_VOTERS

    assert ELIGIBLE_VOTERS == ("rf_corr_mrmr", "boruta")
    # rf_corr_mrmr resolves to the legacy algorithm, not to canonical MI-mRMR.
    legacy_cls, _ = get_selector("mrmr")
    assert legacy_cls is RandomForestRelevanceMRMRSelector
