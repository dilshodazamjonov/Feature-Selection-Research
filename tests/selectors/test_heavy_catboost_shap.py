"""CatBoost native SHAP ranking tests, including an independent oracle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.selectors.heavy.catboost_shap import (
    SHAP_AGGREGATION,
    SHAP_CALC_TYPE,
    SHAP_IMPORTANCE_TYPE,
    CatBoostShapSelector,
)
from credit_risk_fs.selectors.lightweight.contract import (
    ControlledSelectorFailure,
    SelectionResult,
)

#: Synthetic-test profile, not the frozen research configuration.
TINY = {"iterations": 30, "depth": 3}

#: CatBoost fitting is deterministic for a fixed seed, thread count, and CPU task
#: type, so the oracle comparison is exact up to float64 accumulation order in the
#: mean. 1e-12 is far tighter than any meaningful SHAP difference.
ORACLE_TOLERANCE = 1e-12


@pytest.fixture()
def fixture() -> tuple[pd.DataFrame, pd.Series]:
    generator = np.random.default_rng(41)
    n = 400
    strong = generator.normal(size=n)
    weak = generator.normal(size=n)
    logit = 2.4 * strong + 0.5 * weak
    target = pd.Series((1.0 / (1.0 + np.exp(-logit)) > generator.random(n)).astype(int))
    features = pd.DataFrame(
        {
            "strong": strong,
            "weak": weak,
            "noise_a": generator.normal(size=n),
            "noise_b": generator.normal(size=n),
        }
    )
    return features, target


def _fit(features, target, **kwargs):
    kwargs.setdefault("catboost_params", TINY)
    return CatBoostShapSelector(**kwargs).fit(features, target)


# -- independent oracle -------------------------------------------------------


def test_scores_equal_a_direct_native_shap_calculation(fixture) -> None:
    """Recompute mean|SHAP| straight from the CatBoost API and compare.

    ``explanation_sample_size=None`` makes the sample unambiguously "all training
    rows", so the oracle needs no copy of the sampling logic to reproduce it.
    """

    from catboost import CatBoostClassifier, EFstrType, Pool

    features, target = fixture
    selector = _fit(features, target, k=2, explanation_sample_size=None)
    produced = selector.shap_scores_

    params = dict(selector.result.configuration["estimator_params"])
    reference_model = CatBoostClassifier(**params).fit(features, target)
    raw = np.asarray(
        reference_model.get_feature_importance(
            Pool(features, target),
            type=EFstrType.ShapValues,
            shap_calc_type="Regular",
            thread_count=1,
        ),
        dtype="float64",
    )
    assert raw.shape == (len(features), features.shape[1] + 1)
    expected = np.abs(raw[:, :-1]).mean(axis=0)

    for name, value in zip(features.columns, expected, strict=True):
        assert produced[name] == pytest.approx(value, abs=ORACLE_TOLERANCE)


def test_expected_value_column_is_excluded(fixture) -> None:
    from catboost import CatBoostClassifier, EFstrType, Pool

    features, target = fixture
    selector = _fit(features, target, k=2, explanation_sample_size=None)

    params = dict(selector.result.configuration["estimator_params"])
    model = CatBoostClassifier(**params).fit(features, target)
    raw = np.asarray(
        model.get_feature_importance(
            Pool(features, target), type=EFstrType.ShapValues, shap_calc_type="Regular"
        ),
        dtype="float64",
    )
    base_column_mean = float(np.abs(raw[:, -1]).mean())

    assert len(selector.shap_scores_) == features.shape[1]
    assert set(selector.shap_scores_) == set(features.columns)
    # The base value is a large constant; if it had leaked in as a feature the
    # scores would contain it.
    assert base_column_mean > 0
    assert not any(
        value == pytest.approx(base_column_mean, abs=1e-9)
        for value in selector.shap_scores_.values()
    )


def test_score_to_feature_alignment_and_orientation(fixture) -> None:
    features, target = fixture
    selector = _fit(features, target, k=2, explanation_sample_size=None)
    scores = selector.shap_scores_

    assert scores["strong"] > scores["weak"]
    assert scores["weak"] > scores["noise_a"] or scores["weak"] > scores["noise_b"]
    assert all(value >= 0.0 for value in scores.values())

    ranking = list(selector.result.ranking or ())
    assert ranking[0] == "strong"
    assert ranking == sorted(ranking, key=lambda name: -scores[name])
    assert selector.result.selected_features == ("strong", ranking[1])
    assert selector.result.score_orientation == "higher_is_better"


def test_recorded_variant_identity_is_explicit(fixture) -> None:
    features, target = fixture
    result = _fit(features, target, k=2).result

    assert result.implementation_id == (
        "catboost_native_shap_regular_mean_abs_train_sample_v1"
    )
    assert result.configuration["feature_importance_type"] == SHAP_IMPORTANCE_TYPE
    assert result.configuration["shap_calc_type"] == SHAP_CALC_TYPE
    assert result.configuration["aggregation"] == SHAP_AGGREGATION
    assert result.configuration["fallback_importance"] == "none_permitted"
    assert result.heavy_metadata["shap_calc_type"] == "Regular"


# -- explanation sample -------------------------------------------------------


def test_explanation_sample_is_deterministic_and_recorded(fixture) -> None:
    features, target = fixture
    first = _fit(features, target, k=2, explanation_sample_size=120)
    second = _fit(features, target, k=2, explanation_sample_size=120)

    left = first.explanation_sample_
    right = second.explanation_sample_
    assert left["row_identity_sha256"] == right["row_identity_sha256"]
    assert left["realized_size"] == right["realized_size"] == 120
    assert left["requested_size"] == 120
    assert left["scope"] == "selector_training_partition_only"
    assert first.shap_scores_ == second.shap_scores_


def test_explanation_sample_preserves_class_prevalence(fixture) -> None:
    features, target = fixture
    selector = _fit(features, target, k=2, explanation_sample_size=100)
    sample = selector.explanation_sample_

    assert sample["positive_count"] + sample["negative_count"] == 100
    assert sample["positive_count"] > 0
    assert sample["negative_count"] > 0
    # Stratified, so the sample rate tracks the training rate closely.
    assert abs(sample["sample_positive_rate"] - sample["training_positive_rate"]) < 0.02


def test_a_different_sample_seed_is_supported_and_recorded(fixture) -> None:
    features, target = fixture
    first = _fit(features, target, k=2, explanation_sample_size=100, explanation_sample_seed=1)
    second = _fit(features, target, k=2, explanation_sample_size=100, explanation_sample_seed=2)

    assert first.explanation_sample_["seed"] == 1
    assert second.explanation_sample_["seed"] == 2
    assert (
        first.explanation_sample_["row_identity_sha256"]
        != second.explanation_sample_["row_identity_sha256"]
    )
    # Method identity does not change with the sample seed.
    assert first.result.implementation_id == second.result.implementation_id


def test_sample_larger_than_training_uses_every_training_row(fixture) -> None:
    features, target = fixture
    selector = _fit(features, target, k=2, explanation_sample_size=10_000)
    sample = selector.explanation_sample_

    assert sample["realized_size"] == len(features)
    assert sample["used_all_training_rows"] is True
    assert any("every training row was used" in item for item in selector.result.warnings)


def test_explanation_sample_never_reaches_outside_the_training_partition(fixture) -> None:
    features, target = fixture
    half = len(features) // 2
    inside = _fit(features.iloc[:half], target.iloc[:half], k=2, explanation_sample_size=50)

    corrupted = target.copy()
    corrupted.iloc[half:] = 1 - corrupted.iloc[half:]
    recomputed = _fit(
        features.iloc[:half], corrupted.iloc[:half], k=2, explanation_sample_size=50
    )

    assert (
        inside.explanation_sample_["row_identity_sha256"]
        == recomputed.explanation_sample_["row_identity_sha256"]
    )
    assert inside.shap_scores_ == recomputed.shap_scores_
    assert inside.result.selected_features == recomputed.result.selected_features
    assert inside.explanation_sample_["training_row_count"] == half


def test_no_sampling_happens_when_none_was_requested(fixture) -> None:
    features, target = fixture
    selector = _fit(features, target, k=2, explanation_sample_size=None)
    sample = selector.explanation_sample_
    assert sample["requested_size"] is None
    assert sample["realized_size"] == len(features)
    assert sample["used_all_training_rows"] is True


def test_global_rng_state_is_untouched(fixture) -> None:
    features, target = fixture
    np.random.seed(909)
    before = np.random.get_state()[1][:8].copy()
    _fit(features, target, k=2, explanation_sample_size=100)
    assert np.array_equal(before, np.random.get_state()[1][:8])


# -- budget and validation ----------------------------------------------------


def test_k_none_fails_before_catboost_is_fit(fixture, monkeypatch) -> None:
    features, target = fixture
    import catboost

    def _explode(*args, **kwargs):
        raise AssertionError("CatBoost must not be constructed")

    monkeypatch.setattr(catboost, "CatBoostClassifier", _explode)
    with pytest.raises(ControlledSelectorFailure) as error:
        CatBoostShapSelector(k=None, catboost_params=TINY).fit(features, target)
    assert error.value.stage == "budget_validation"
    assert "no defensible natural selection threshold" in error.value.cause


def test_non_positive_k_fails_before_fitting(fixture) -> None:
    features, target = fixture
    for budget in (0, -1):
        with pytest.raises(ControlledSelectorFailure) as error:
            _fit(features, target, k=budget)
        assert error.value.stage == "budget_validation"


def test_natural_support_is_absent(fixture) -> None:
    features, target = fixture
    result = _fit(features, target, k=2).result
    assert result.natural_selected is None
    assert CatBoostShapSelector.supports_natural_support is False
    assert result.configuration["natural_support"] == "unsupported_ranking_method"


def test_budget_larger_than_universe_is_clipped(fixture) -> None:
    features, target = fixture
    result = _fit(features, target, k=features.shape[1] + 5).result
    assert result.budget_status == "clipped_to_universe"
    assert result.actual_selected_count == features.shape[1]


# -- controlled failures ------------------------------------------------------


def _patch_shap(monkeypatch, matrix):
    import catboost

    class _Model:
        def __init__(self, **kwargs):
            self.tree_count_ = 1

        def fit(self, X, y):
            self._rows = len(X)
            return self

        def get_feature_importance(self, *args, **kwargs):
            return matrix(self._rows)

    monkeypatch.setattr(catboost, "CatBoostClassifier", _Model)


def test_all_zero_shap_follows_the_declared_warning_and_tie_policy(
    fixture, monkeypatch
) -> None:
    features, target = fixture
    _patch_shap(monkeypatch, lambda rows: np.zeros((rows, features.shape[1] + 1)))

    result = _fit(features, target, k=2, explanation_sample_size=None).result
    assert all(value == 0.0 for value in (result.raw_scores or {}).values())
    # Ties fall to the authenticated candidate order.
    assert list(result.ranking or ()) == list(features.columns)
    assert result.selected_features == tuple(features.columns[:2])
    assert result.budget_status == "satisfied"
    assert any("all_scores_zero" in item for item in result.warnings)


def test_invalid_shap_shape_fails_explicitly(fixture, monkeypatch) -> None:
    features, target = fixture
    # Missing the expected-value column entirely.
    _patch_shap(monkeypatch, lambda rows: np.zeros((rows, features.shape[1])))
    with pytest.raises(ControlledSelectorFailure) as error:
        _fit(features, target, k=2, explanation_sample_size=None)
    assert error.value.stage == "shap_shape_validation"
    assert "expected-value column" in error.value.cause


def test_non_finite_shap_values_fail_explicitly(fixture, monkeypatch) -> None:
    features, target = fixture

    def matrix(rows):
        values = np.zeros((rows, features.shape[1] + 1))
        values[0, 0] = np.nan
        values[1, 1] = np.inf
        return values

    _patch_shap(monkeypatch, matrix)
    with pytest.raises(ControlledSelectorFailure) as error:
        _fit(features, target, k=2, explanation_sample_size=None)
    assert error.value.stage == "shap_value_validation"
    assert "non-finite" in error.value.cause


def test_shap_failure_uses_no_fallback_importance(fixture, monkeypatch) -> None:
    features, target = fixture
    import catboost

    class _Model:
        def __init__(self, **kwargs):
            self.tree_count_ = 1

        def fit(self, X, y):
            return self

        def get_feature_importance(self, *args, **kwargs):
            if kwargs.get("type") is not None:
                raise RuntimeError("shap unavailable")
            return np.ones(features.shape[1])  # a fallback must NOT be used

    monkeypatch.setattr(catboost, "CatBoostClassifier", _Model)
    with pytest.raises(ControlledSelectorFailure) as error:
        _fit(features, target, k=2, explanation_sample_size=None)
    assert error.value.stage == "shap_calculation"
    assert "no substitute importance is permitted" in error.value.cause


def test_training_failure_is_controlled(fixture, monkeypatch) -> None:
    features, target = fixture
    import catboost

    class _Broken:
        def __init__(self, **kwargs):
            pass

        def fit(self, X, y):
            raise RuntimeError("training blew up")

    monkeypatch.setattr(catboost, "CatBoostClassifier", _Broken)
    with pytest.raises(ControlledSelectorFailure) as error:
        _fit(features, target, k=2)
    assert error.value.stage == "estimator_fit"
    assert "training blew up" in error.value.cause


def test_single_class_target_fails_explicitly(fixture) -> None:
    features, target = fixture
    with pytest.raises(ControlledSelectorFailure) as error:
        _fit(features, pd.Series(np.zeros(len(features), dtype=int)), k=2)
    assert "single class" in error.value.cause


def test_excluded_columns_never_reach_the_model(fixture) -> None:
    features, target = fixture
    contaminated = features.assign(TARGET=target.to_numpy())
    with pytest.raises(ControlledSelectorFailure) as error:
        _fit(contaminated, target, k=2, excluded_columns=("TARGET",))
    assert error.value.stage == "candidate_universe_validation"


# -- logging and serialization ------------------------------------------------


def test_catboost_progress_is_suppressed_and_stages_are_logged(fixture, caplog) -> None:
    import logging

    features, target = fixture
    with caplog.at_level(logging.INFO, logger="credit_risk_fs.selectors.heavy.catboost_shap"):
        _fit(features, target, k=2, explanation_sample_size=None)

    messages = [record.getMessage() for record in caplog.records]
    assert any(text.startswith("START") and "catboost_fit" in text for text in messages)
    assert any(text.startswith("DONE") and "catboost_fit" in text for text in messages)
    assert any(
        text.startswith("START") and "native_shap_values" in text for text in messages
    )
    assert any(
        text.startswith("DONE") and "native_shap_values" in text for text in messages
    )
    assert not any("learn:" in text for text in messages)


def test_serialization_round_trip_is_exact(fixture) -> None:
    features, target = fixture
    original = _fit(features, target, k=2, explanation_sample_size=100).result
    restored = SelectionResult.from_json(original.to_json())

    assert restored.method_id == "catboost_shap"
    assert restored.implementation_id == original.implementation_id
    assert restored.selected_features == original.selected_features
    assert restored.ranking == original.ranking
    assert restored.natural_selected is None
    assert restored.estimator_config_sha256 == original.estimator_config_sha256
    assert (
        restored.heavy_metadata["explanation_sample"]["row_identity_sha256"]
        == original.heavy_metadata["explanation_sample"]["row_identity_sha256"]
    )
    for name, value in original.raw_scores.items():
        assert restored.raw_scores[name] == pytest.approx(value, abs=0.0)


def test_repository_had_no_prior_shap_path(fixture) -> None:
    """CatBoostModel.get_feature_importance is PredictionValuesChange, not SHAP."""

    import inspect

    from credit_risk_fs.models.catboost_model import CatBoostModel

    source = inspect.getsource(CatBoostModel.get_feature_importance)
    assert "ShapValues" not in source
    assert "shap" not in source.lower()
    assert "get_feature_importance()" in source
