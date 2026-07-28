"""Boruta support-state and mode-policy tests.

Support-state policy is tested against a deterministic stub engine, because real
Boruta is stochastic and a real-engine assertion about *which* feature is
confirmed would be flaky. One tiny real-engine test covers wiring and status.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.selectors.boruta import BorutaSelector
from credit_risk_fs.selectors.heavy.boruta_rf import (
    CONFIRMED,
    REJECTED,
    TENTATIVE,
    BorutaRandomForestSelector,
)
from credit_risk_fs.selectors.lightweight.contract import (
    ControlledSelectorFailure,
    SelectionResult,
)

CANDIDATES = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")


class _StubEngine:
    """Deterministic BorutaPy stand-in with explicit support states.

    ``support_`` and ``support_weak_`` mirror the real engine's semantics,
    including the fact that a feature may appear in both.
    """

    def __init__(self, *, confirmed, tentative, ranking):
        self._confirmed = set(confirmed)
        self._tentative = set(tentative)
        self._ranking = ranking
        self.fit_calls = 0

    def fit(self, X, y):
        self.fit_calls += 1
        self.support_ = np.array([name in self._confirmed for name in CANDIDATES])
        self.support_weak_ = np.array([name in self._tentative for name in CANDIDATES])
        self.ranking_ = np.array([self._ranking[name] for name in CANDIDATES])
        return self


def _stub_factory(*, confirmed, tentative, ranking=None):
    ranking = ranking or {
        name: position + 1 for position, name in enumerate(CANDIDATES)
    }

    def factory(*, forest_params, boruta_params):
        return _StubEngine(confirmed=confirmed, tentative=tentative, ranking=ranking)

    return factory


@pytest.fixture()
def frame() -> tuple[pd.DataFrame, pd.Series]:
    generator = np.random.default_rng(5)
    n = 200
    return (
        pd.DataFrame({name: generator.normal(size=n) for name in CANDIDATES}),
        pd.Series(generator.integers(0, 2, n)),
    )


def _fit(frame, target, **kwargs):
    kwargs.setdefault(
        "engine_factory",
        _stub_factory(confirmed=("alpha", "bravo"), tentative=("charlie", "delta")),
    )
    return BorutaRandomForestSelector(**kwargs).fit(frame, target)


# -- support states -----------------------------------------------------------


def test_all_three_support_states_are_preserved_exactly(frame) -> None:
    features, target = frame
    selector = _fit(features, target, selection_mode="natural_confirmed")
    metadata = selector.result.heavy_metadata

    assert metadata["confirmed"] == ["alpha", "bravo"]
    assert metadata["tentative"] == ["charlie", "delta"]
    assert metadata["rejected"] == ["echo", "foxtrot"]
    assert metadata["confirmed_count"] == 2
    assert metadata["tentative_count"] == 2
    assert metadata["rejected_count"] == 2

    states = metadata["support_states"]
    assert states["alpha"] == CONFIRMED
    assert states["charlie"] == TENTATIVE
    assert states["echo"] == REJECTED
    # The three states partition the candidate universe exactly once.
    assert sorted(states) == sorted(CANDIDATES)


def test_a_feature_in_both_arrays_counts_as_confirmed(frame) -> None:
    features, target = frame
    selector = _fit(
        features,
        target,
        selection_mode="natural_confirmed",
        engine_factory=_stub_factory(
            confirmed=("alpha",), tentative=("alpha", "bravo")
        ),
    )
    metadata = selector.result.heavy_metadata
    assert metadata["confirmed"] == ["alpha"]
    assert metadata["tentative"] == ["bravo"]
    assert "alpha" not in metadata["tentative"]


def test_natural_support_contains_confirmed_only(frame) -> None:
    features, target = frame
    selector = _fit(features, target, selection_mode="natural_confirmed")
    result = selector.result

    assert result.natural_selected == ("alpha", "bravo")
    assert result.selected_features == ("alpha", "bravo")
    assert result.selection_mode == "natural_confirmed"
    assert result.budget_status == "not_applicable"
    # Tentative features are present in the ranking but not in the support.
    assert "charlie" in (result.ranking or ())
    assert "charlie" not in result.natural_selected


def test_natural_mode_ignores_a_budget_and_says_so(frame) -> None:
    features, target = frame
    selector = _fit(features, target, selection_mode="natural_confirmed", k=5)
    result = selector.result
    assert result.actual_selected_count == 2
    assert result.requested_budget is None
    assert any("ignores the requested budget 5" in item for item in result.warnings)


def test_empty_confirmed_support_is_a_valid_natural_outcome(frame) -> None:
    features, target = frame
    selector = _fit(
        features,
        target,
        selection_mode="natural_confirmed",
        engine_factory=_stub_factory(confirmed=(), tentative=("alpha", "bravo")),
    )
    result = selector.result
    assert result.natural_selected == ()
    assert result.selected_features == ()
    assert any("confirmed no feature" in item for item in result.warnings)
    # No tentative feature was promoted to cover the empty support.
    assert result.heavy_metadata["tentative_count"] == 2


# -- fixed-budget modes -------------------------------------------------------


def test_confirmed_top_k_succeeds_when_confirmed_is_sufficient(frame) -> None:
    features, target = frame
    selector = _fit(
        features,
        target,
        selection_mode="confirmed_top_k",
        k=2,
        engine_factory=_stub_factory(
            confirmed=("alpha", "bravo", "charlie"), tentative=("delta",)
        ),
    )
    result = selector.result
    assert result.selection_mode == "confirmed_top_k"
    assert result.budget_status == "satisfied"
    assert result.selected_features == ("alpha", "bravo")
    assert result.natural_selected == ("alpha", "bravo", "charlie")


def test_confirmed_top_k_reports_infeasible_without_padding(frame) -> None:
    features, target = frame
    selector = _fit(features, target, selection_mode="confirmed_top_k", k=4)
    result = selector.result

    assert result.budget_status == "infeasible_natural_support"
    assert result.selected_features == ("alpha", "bravo")
    # Neither tentative nor rejected features were used as filler.
    assert not set(result.selected_features) & {"charlie", "delta", "echo", "foxtrot"}
    assert any("tentative features were not used as filler" in i for i in result.warnings)


def test_confirmed_then_tentative_is_explicit_and_never_called_natural(frame) -> None:
    features, target = frame
    selector = _fit(features, target, selection_mode="confirmed_then_tentative", k=4)
    result = selector.result

    assert result.selection_mode == "confirmed_then_tentative"
    assert result.budget_status == "satisfied"
    assert result.selected_features == ("alpha", "bravo", "charlie", "delta")
    # The natural support stays confirmed-only even though the selection is larger.
    assert result.natural_selected == ("alpha", "bravo")
    assert set(result.selected_features) - set(result.natural_selected) == {
        "charlie",
        "delta",
    }
    assert any("not natural Boruta support" in item for item in result.warnings)


def test_rejected_features_never_pad_any_budget(frame) -> None:
    features, target = frame
    selector = _fit(features, target, selection_mode="confirmed_then_tentative", k=6)
    result = selector.result

    assert result.budget_status == "infeasible_natural_support"
    assert result.actual_selected_count == 4
    assert not set(result.selected_features) & {"echo", "foxtrot"}
    assert any("rejected features were not used as filler" in i for i in result.warnings)


def test_confirmed_ordering_uses_engine_rank_then_candidate_order(frame) -> None:
    features, target = frame
    # Deliberate rank tie between bravo and alpha; the candidate order decides.
    selector = _fit(
        features,
        target,
        selection_mode="natural_confirmed",
        engine_factory=_stub_factory(
            confirmed=("bravo", "alpha"),
            tentative=(),
            ranking={"alpha": 1, "bravo": 1, "charlie": 2, "delta": 3, "echo": 4, "foxtrot": 5},
        ),
    )
    assert selector.result.natural_selected == ("alpha", "bravo")


def test_fixed_budget_modes_require_k_before_the_engine_runs(frame) -> None:
    features, target = frame
    for mode in ("confirmed_top_k", "confirmed_then_tentative"):
        engine_calls: list[int] = []

        def factory(*, forest_params, boruta_params):
            engine_calls.append(1)
            raise AssertionError("the engine must not be constructed")

        with pytest.raises(ControlledSelectorFailure) as error:
            BorutaRandomForestSelector(
                k=None, selection_mode=mode, engine_factory=factory
            ).fit(features, target)
        assert error.value.stage == "budget_validation"
        assert not engine_calls

    with pytest.raises(ControlledSelectorFailure, match="must be positive"):
        _fit(features, target, selection_mode="confirmed_top_k", k=0)


def test_unknown_mode_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="selection_mode must be one of"):
        BorutaRandomForestSelector(selection_mode="all_relevant_plus_vibes")


# -- determinism, leakage, failure -------------------------------------------


def test_same_configuration_reproduces_the_result(frame) -> None:
    features, target = frame
    first = _fit(features, target, selection_mode="natural_confirmed")
    second = _fit(features, target, selection_mode="natural_confirmed")
    assert first.result.selected_features == second.result.selected_features
    assert first.result.ranking == second.result.ranking
    assert first.result.estimator_config_sha256 == second.result.estimator_config_sha256


def test_global_rng_state_is_untouched(frame) -> None:
    features, target = frame
    np.random.seed(4242)
    before = np.random.get_state()[1][:8].copy()
    _fit(features, target, selection_mode="natural_confirmed")
    assert np.array_equal(before, np.random.get_state()[1][:8])


def test_outer_validation_rows_cannot_influence_the_result(frame) -> None:
    features, target = frame
    half = len(features) // 2
    inside = _fit(
        features.iloc[:half], target.iloc[:half], selection_mode="natural_confirmed"
    )

    corrupted = target.copy()
    corrupted.iloc[half:] = 1 - corrupted.iloc[half:]
    recomputed = _fit(
        features.iloc[:half], corrupted.iloc[:half], selection_mode="natural_confirmed"
    )
    assert inside.result.selected_features == recomputed.result.selected_features
    assert (
        inside.result.training_identity_sha256
        == recomputed.result.training_identity_sha256
    )
    assert inside.result.training_row_count == half


def test_excluded_columns_never_reach_the_engine(frame) -> None:
    features, target = frame
    contaminated = features.assign(TARGET=target.to_numpy())
    with pytest.raises(ControlledSelectorFailure) as error:
        _fit(
            contaminated,
            target,
            selection_mode="natural_confirmed",
            excluded_columns=("TARGET",),
        )
    assert error.value.stage == "candidate_universe_validation"


def test_engine_failure_has_no_fallback(frame) -> None:
    features, target = frame

    class _Broken:
        def fit(self, X, y):
            raise RuntimeError("engine exploded")

    with pytest.raises(ControlledSelectorFailure) as error:
        _fit(
            features,
            target,
            selection_mode="natural_confirmed",
            engine_factory=lambda **_: _Broken(),
        )
    assert error.value.stage == "engine_fit"
    assert "engine exploded" in error.value.cause


def test_support_length_mismatch_fails_explicitly(frame) -> None:
    features, target = frame

    class _Wrong:
        def fit(self, X, y):
            self.support_ = np.array([True, False])
            self.support_weak_ = np.array([False, True])
            self.ranking_ = np.array([1, 2])
            return self

    with pytest.raises(ControlledSelectorFailure) as error:
        _fit(
            features,
            target,
            selection_mode="natural_confirmed",
            engine_factory=lambda **_: _Wrong(),
        )
    assert error.value.stage == "support_extraction"


def test_non_finite_candidates_are_refused_rather_than_imputed(frame) -> None:
    """The engine needs a finite matrix; the selector must not impute silently."""

    features, target = frame
    holed = features.copy()
    holed.loc[holed.index[:10], "alpha"] = np.nan

    with pytest.raises(ControlledSelectorFailure) as error:
        _fit(holed, target, selection_mode="natural_confirmed")
    assert error.value.stage == "design_matrix_validation"
    assert "alpha" in error.value.cause
    assert "impute upstream" in error.value.cause


def test_serialization_preserves_all_three_support_states(frame) -> None:
    features, target = frame
    original = _fit(features, target, selection_mode="confirmed_then_tentative", k=3).result
    restored = SelectionResult.from_json(original.to_json())

    assert restored.selection_mode == "confirmed_then_tentative"
    assert restored.natural_selected == original.natural_selected
    assert restored.selected_features == original.selected_features
    assert restored.heavy_metadata["confirmed"] == original.heavy_metadata["confirmed"]
    assert restored.heavy_metadata["tentative"] == original.heavy_metadata["tentative"]
    assert restored.heavy_metadata["rejected"] == original.heavy_metadata["rejected"]
    assert restored.heavy_metadata["support_states"] == original.heavy_metadata[
        "support_states"
    ]
    assert restored.estimator_config_sha256 == original.estimator_config_sha256


# -- real engine wiring -------------------------------------------------------


def test_tiny_real_engine_runs_and_reports_states(frame) -> None:
    """Wiring/status evidence only; support membership is not asserted here."""

    pytest.importorskip("boruta")
    generator = np.random.default_rng(17)
    n = 300
    latent = generator.normal(size=n)
    target = pd.Series((latent + generator.normal(scale=0.4, size=n) > 0).astype(int))
    features = pd.DataFrame(
        {
            "alpha": latent,
            "bravo": generator.normal(size=n),
            "charlie": generator.normal(size=n),
        }
    )
    selector = BorutaRandomForestSelector(
        selection_mode="natural_confirmed",
        forest_params={"n_estimators": 30, "max_depth": 4},
        boruta_params={"max_iter": 8},
    ).fit(features, target)

    metadata = selector.result.heavy_metadata
    total = (
        metadata["confirmed_count"] + metadata["tentative_count"] + metadata["rejected_count"]
    )
    assert total == features.shape[1]
    assert metadata["stop_reason"]
    assert set(metadata["support_states"]) == set(features.columns)
    assert selector.result.natural_selected is not None


# -- legacy compatibility -----------------------------------------------------


def test_legacy_boruta_selector_is_unchanged_and_lacks_tentative(frame) -> None:
    """The audit finding, asserted: the legacy path discards the tentative state."""

    legacy = BorutaSelector()
    assert not hasattr(legacy, "support_weak_")
    assert not hasattr(legacy, "tentative_")
    assert not hasattr(legacy, "support_states_")
    # And it exposes no contract result object at all.
    assert not hasattr(legacy, "result_")
    assert BorutaSelector.__module__ == "credit_risk_fs.selectors.boruta"
