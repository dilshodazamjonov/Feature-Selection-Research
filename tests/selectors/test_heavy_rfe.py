"""CatBoost-backed RFE tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.selectors.heavy.rfe_catboost import CatBoostRFESelector
from credit_risk_fs.selectors.lightweight.contract import (
    ControlledSelectorFailure,
    SelectionResult,
)
from credit_risk_fs.selectors.rfe import RFESelector

#: Deliberately tiny so the suite stays fast. This is a synthetic-test profile,
#: not the frozen research configuration.
TINY = {"iterations": 25, "depth": 3}


@pytest.fixture()
def fixture() -> tuple[pd.DataFrame, pd.Series]:
    generator = np.random.default_rng(29)
    n = 500
    linear = generator.normal(size=n)
    nonlinear = generator.normal(size=n)
    logit = 2.2 * linear + 1.6 * (nonlinear**2 - 1.0)
    target = pd.Series((1.0 / (1.0 + np.exp(-logit)) > generator.random(n)).astype(int))
    features = pd.DataFrame(
        {
            "linear_signal": linear,
            "nonlinear_signal": nonlinear,
            "noise_a": generator.normal(size=n),
            "noise_b": generator.normal(size=n),
            "noise_c": generator.normal(size=n),
            "noise_d": generator.normal(size=n),
        }
    )
    return features, target


def _fit(features, target, **kwargs):
    kwargs.setdefault("catboost_params", TINY)
    return CatBoostRFESelector(**kwargs).fit(features, target)


def test_signal_survives_longer_than_seeded_noise(fixture) -> None:
    features, target = fixture
    selector = _fit(features, target, k=2)
    selected = set(selector.result.selected_features)
    assert selected == {"linear_signal", "nonlinear_signal"}

    ranking = list(selector.result.ranking or ())
    for noise in ("noise_a", "noise_b", "noise_c", "noise_d"):
        assert ranking.index("linear_signal") < ranking.index(noise)
        assert ranking.index("nonlinear_signal") < ranking.index(noise)


def test_exact_k_is_returned(fixture) -> None:
    features, target = fixture
    for budget in (1, 2, 4):
        result = _fit(features, target, k=budget).result
        assert result.actual_selected_count == budget
        assert len(set(result.selected_features)) == budget
        assert result.budget_status == "satisfied"
        # Every candidate carries a rank, so the elimination order is complete.
        assert len(result.ranking or ()) == features.shape[1]


def test_k_none_fails_before_catboost_is_called(fixture, monkeypatch) -> None:
    features, target = fixture
    import catboost

    def _explode(*args, **kwargs):
        raise AssertionError("CatBoost must not be constructed")

    monkeypatch.setattr(catboost, "CatBoostClassifier", _explode)
    with pytest.raises(ControlledSelectorFailure) as error:
        CatBoostRFESelector(k=None, catboost_params=TINY).fit(features, target)
    assert error.value.stage == "budget_validation"
    assert "no natural stopping point" in error.value.cause


def test_non_positive_k_fails_before_fitting(fixture) -> None:
    features, target = fixture
    for budget in (0, -3):
        with pytest.raises(ControlledSelectorFailure) as error:
            _fit(features, target, k=budget)
        assert error.value.stage == "budget_validation"


def test_budget_equal_to_universe_performs_no_elimination(fixture) -> None:
    features, target = fixture
    selector = _fit(features, target, k=features.shape[1])
    result = selector.result

    assert selector.estimator_fit_count_ == 0
    assert result.heavy_metadata["estimator_fit_count"] == 0
    assert result.heavy_metadata["elimination_history"] == []
    assert result.heavy_metadata["elimination_skipped"] is True
    assert list(result.selected_features) == list(features.columns)
    assert any("no elimination was necessary" in item for item in result.warnings)


def test_budget_larger_than_universe_is_clipped_not_padded(fixture) -> None:
    features, target = fixture
    result = _fit(features, target, k=features.shape[1] + 10).result
    assert result.budget_status == "clipped_to_universe"
    assert result.actual_selected_count == features.shape[1]
    assert len(set(result.selected_features)) == features.shape[1]


def test_fractional_step_is_honored_and_history_is_complete(fixture) -> None:
    features, target = fixture
    selector = _fit(features, target, k=1, step_fraction=0.5)
    history = selector.elimination_history_

    assert not history.empty
    # 6 -> remove max(1, int(0.5*6))=3 -> 3 -> int(0.5*3)=1 -> 2 -> 1 -> 1
    assert list(history["surviving_before"]) == [6, 3, 2]
    assert list(history["requested_removals"]) == [3, 1, 1]
    assert list(history["realized_removals"]) == [3, 1, 1]
    assert history["realized_removals"].sum() == features.shape[1] - 1

    # One fit per elimination round plus one final ordering fit.
    assert selector.estimator_fit_count_ == len(history) + 1
    assert selector.result.heavy_metadata["step_fraction_configured"] == 0.5

    removed = [name for row in history["removed_features"] for name in row]
    assert len(removed) == len(set(removed))
    assert set(removed) | set(selector.result.selected_features) == set(features.columns)


def test_realized_removals_never_overshoot_the_budget(fixture) -> None:
    features, target = fixture
    selector = _fit(features, target, k=5, step_fraction=0.9)
    history = selector.elimination_history_
    # Requesting 90% of 6 would remove 5, but only 1 may go.
    assert list(history["requested_removals"]) == [5]
    assert list(history["realized_removals"]) == [1]
    assert selector.result.actual_selected_count == 5


def test_step_fraction_must_be_a_fraction() -> None:
    for value in (0.0, 1.0, 10, -0.5):
        with pytest.raises(ValueError, match="step_fraction must be a fraction"):
            CatBoostRFESelector(k=2, step_fraction=value)


def test_natural_support_is_absent_not_fabricated(fixture) -> None:
    features, target = fixture
    result = _fit(features, target, k=3).result
    assert result.natural_selected is None
    assert result.natural_selected_count is None
    assert CatBoostRFESelector.supports_natural_support is False
    assert result.configuration["natural_support"] == "unsupported_wrapper_method"


def test_same_seed_and_configuration_reproduce_selection_and_ranking(fixture) -> None:
    features, target = fixture
    first = _fit(features, target, k=3, random_state=7)
    second = _fit(features, target, k=3, random_state=7)

    assert first.result.selected_features == second.result.selected_features
    assert first.result.ranking == second.result.ranking
    assert first.estimator_fit_count_ == second.estimator_fit_count_
    assert first.result.estimator_config_sha256 == second.result.estimator_config_sha256
    pd.testing.assert_frame_equal(first.elimination_history_, second.elimination_history_)


def test_column_order_does_not_change_the_outcome(fixture) -> None:
    features, target = fixture
    forward = _fit(features, target, k=2)
    reversed_columns = features[list(reversed(features.columns))]
    backward = _fit(reversed_columns, target, k=2)
    assert set(forward.result.selected_features) == set(backward.result.selected_features)


def test_global_rng_state_is_untouched(fixture) -> None:
    features, target = fixture
    np.random.seed(1717)
    before = np.random.get_state()[1][:8].copy()
    _fit(features, target, k=2)
    assert np.array_equal(before, np.random.get_state()[1][:8])


def test_outer_validation_rows_cannot_influence_the_result(fixture) -> None:
    features, target = fixture
    half = len(features) // 2
    inside = _fit(features.iloc[:half], target.iloc[:half], k=2)

    corrupted = target.copy()
    corrupted.iloc[half:] = 1 - corrupted.iloc[half:]
    recomputed = _fit(features.iloc[:half], corrupted.iloc[:half], k=2)

    assert inside.result.selected_features == recomputed.result.selected_features
    assert inside.result.training_row_count == half
    assert (
        inside.result.training_identity_sha256
        == recomputed.result.training_identity_sha256
    )


def test_excluded_columns_cannot_enter_catboost(fixture) -> None:
    features, target = fixture
    contaminated = features.assign(
        TARGET=target.to_numpy(), SK_ID_CURR=np.arange(len(features))
    )
    with pytest.raises(ControlledSelectorFailure) as error:
        _fit(
            contaminated,
            target,
            k=2,
            excluded_columns=("TARGET", "SK_ID_CURR", "split", "time_index"),
        )
    assert error.value.stage == "candidate_universe_validation"
    assert "TARGET" in error.value.cause


def test_catboost_output_is_suppressed_and_writes_no_files(fixture, capfd) -> None:
    features, target = fixture
    selector = _fit(features, target, k=2)
    captured = capfd.readouterr()
    # No CatBoost iteration table on stdout/stderr.
    assert "learn:" not in captured.out
    assert "bestTest" not in captured.out

    params = selector.result.configuration["estimator_params"]
    assert params["allow_writing_files"] is False
    assert params["verbose"] is False
    assert params["thread_count"] == 1
    assert params["random_seed"] == 42
    assert params["task_type"] == "CPU"


def test_estimator_failure_produces_one_controlled_failure_without_fallback(
    fixture, monkeypatch
) -> None:
    features, target = fixture
    import catboost

    class _Broken:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            raise RuntimeError("boosting diverged")

    monkeypatch.setattr(catboost, "CatBoostClassifier", _Broken)
    with pytest.raises(ControlledSelectorFailure) as error:
        _fit(features, target, k=2)
    assert error.value.stage == "estimator_fit"
    assert "boosting diverged" in error.value.cause


def test_importance_length_mismatch_fails_explicitly(fixture, monkeypatch) -> None:
    features, target = fixture
    import catboost

    class _Wrong:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            return self

        def get_feature_importance(self):
            return np.array([1.0, 2.0])

    monkeypatch.setattr(catboost, "CatBoostClassifier", _Wrong)
    with pytest.raises(ControlledSelectorFailure) as error:
        _fit(features, target, k=2)
    assert error.value.stage == "importance_extraction"


def test_serialization_round_trip_is_exact(fixture) -> None:
    features, target = fixture
    original = _fit(features, target, k=2, step_fraction=0.5).result
    restored = SelectionResult.from_json(original.to_json())

    assert restored.method_id == "rfe_catboost"
    assert restored.implementation_id == "rfe_catboost_fractional_step_v1"
    assert restored.selected_features == original.selected_features
    assert restored.ranking == original.ranking
    assert restored.natural_selected is None
    assert restored.estimator_config_sha256 == original.estimator_config_sha256
    assert restored.heavy_metadata["estimator_fit_count"] == original.heavy_metadata[
        "estimator_fit_count"
    ]
    assert restored.heavy_metadata["elimination_history"] == original.heavy_metadata[
        "elimination_history"
    ]


def test_legacy_rfe_selector_keeps_its_integer_step(fixture) -> None:
    """The audited discrepancy, asserted: legacy is integer-step, new is fractional."""

    legacy = RFESelector()
    assert legacy.step == 10
    assert isinstance(legacy.step, int)
    assert legacy._estimator_config()["rfe_step"] == 10
    assert legacy._estimator_config()["implementation"] == "catboost.CatBoostClassifier"

    configuration = CatBoostRFESelector(k=2).describe_configuration()
    assert configuration["step_kind"] == "fraction_of_surviving_features"
    assert configuration["step_fraction"] == 0.20
    assert configuration["legacy_counterpart"]["step"] == 10
    assert configuration["legacy_counterpart"]["step_kind"] == (
        "integer_features_per_iteration"
    )
