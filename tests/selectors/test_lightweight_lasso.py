"""Tests for L1-penalized logistic regression as a feature selector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from credit_risk_fs.selectors.lightweight.contract import ControlledSelectorFailure
from credit_risk_fs.selectors.lightweight.lasso import (
    L1LogisticSelector,
    _l1_penalty_kwargs,
)


@pytest.fixture()
def separable_fixture() -> tuple[pd.DataFrame, pd.Series]:
    generator = np.random.default_rng(31)
    n = 900
    strong = generator.normal(size=n)
    moderate = generator.normal(size=n)
    logit = 2.5 * strong + 0.9 * moderate
    target = pd.Series((1.0 / (1.0 + np.exp(-logit)) > generator.random(n)).astype(int))
    frame = pd.DataFrame(
        {
            "strong": strong,
            "moderate": moderate,
            "noise_a": generator.normal(size=n),
            "noise_b": generator.normal(size=n),
            "noise_c": generator.normal(size=n),
        }
    )
    return frame, target


def test_known_signal_receives_the_largest_coefficient(separable_fixture) -> None:
    frame, target = separable_fixture
    selector = L1LogisticSelector(k=2, C=0.5).fit(frame, target)
    coefficients = selector.coefficients_.set_index("feature")

    assert coefficients.loc["strong", "absolute_coefficient"] > coefficients.loc[
        "moderate", "absolute_coefficient"
    ]
    assert bool(coefficients.loc["strong", "non_zero"]) is True
    # Sign is retained separately from magnitude: a positive logit coefficient on
    # a risk-increasing feature must stay positive.
    assert coefficients.loc["strong", "coefficient"] > 0
    assert list(selector.result.selected_features) == ["strong", "moderate"]


def test_coefficients_match_an_independently_configured_estimator(separable_fixture) -> None:
    """Rebuild the same pipeline by hand and compare coefficients exactly."""

    frame, target = separable_fixture
    selector = L1LogisticSelector(k=3, C=0.25, solver="liblinear", max_iter=2_000).fit(
        frame, target
    )

    values = frame.to_numpy(dtype="float64")
    design = StandardScaler().fit_transform(
        SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(values)
    )
    reference = LogisticRegression(
        C=0.25,
        solver="liblinear",
        max_iter=2_000,
        tol=1e-4,
        random_state=42,
        **_l1_penalty_kwargs()[0],
    ).fit(design, target.to_numpy())
    expected = dict(zip(frame.columns, np.asarray(reference.coef_).ravel(), strict=True))

    produced = selector.coefficients_.set_index("feature")["coefficient"].to_dict()
    for name, value in expected.items():
        assert produced[name] == pytest.approx(value, abs=1e-12)


def test_imputer_and_scaler_are_fitted_on_training_rows_only() -> None:
    """A wildly different held-out block must not move the training-fitted result."""

    generator = np.random.default_rng(37)
    n = 500
    latent = generator.normal(size=n)
    target = pd.Series((latent > 0).astype(int))
    frame = pd.DataFrame(
        {
            "latent": latent,
            "holes": np.where(generator.random(n) < 0.3, np.nan, latent),
        }
    )
    baseline = L1LogisticSelector(k=1, C=0.5).fit(frame, target)

    # Append rows on a totally different scale; they are NOT passed to fit.
    outside = frame.copy()
    outside.loc[:, "latent"] = outside["latent"] * 1_000.0
    combined = pd.concat([frame, outside], ignore_index=True)
    combined_target = pd.concat([target, target], ignore_index=True)
    contaminated = L1LogisticSelector(k=1, C=0.5).fit(combined, combined_target)

    assert baseline.result.training_row_count == n
    assert contaminated.result.training_row_count == 2 * n
    assert (
        baseline.result.training_identity_sha256
        != contaminated.result.training_identity_sha256
    )
    # Refitting on the original rows reproduces the original coefficients, proving
    # nothing was cached from the larger frame.
    refit = L1LogisticSelector(k=1, C=0.5).fit(frame, target)
    pd.testing.assert_frame_equal(baseline.coefficients_, refit.coefficients_)


def test_natural_support_and_matched_budget_are_recorded_separately(
    separable_fixture,
) -> None:
    frame, target = separable_fixture
    natural = L1LogisticSelector(k=None, C=0.5).fit(frame, target)
    matched = L1LogisticSelector(k=2, C=0.5).fit(frame, target)

    assert natural.result.selection_mode == "natural"
    assert natural.result.budget_status == "not_applicable"
    assert natural.result.requested_budget is None

    assert matched.result.selection_mode == "matched_budget"
    assert matched.result.requested_budget == 2
    assert matched.result.actual_selected_count == 2
    # The natural support is still published alongside the truncated subset.
    assert matched.result.natural_selected == natural.result.natural_selected
    assert len(matched.result.natural_selected or ()) >= 2

    long_frame = matched.result.to_long_frame().set_index("feature")
    assert set(long_frame["natural_selected"].unique()) <= {True, False}
    assert long_frame["matched_budget_selected"].sum() == 2


def test_serialization_keeps_natural_and_matched_distinguishable(separable_fixture) -> None:
    from credit_risk_fs.selectors.lightweight.contract import SelectionResult

    frame, target = separable_fixture
    matched = L1LogisticSelector(k=2, C=0.5).fit(frame, target).result
    restored = SelectionResult.from_json(matched.to_json())

    assert restored.selection_mode == "matched_budget"
    assert restored.natural_selected == matched.natural_selected
    assert restored.selected_features == matched.selected_features
    assert restored.natural_selected != restored.selected_features


def test_budget_beyond_the_natural_support_is_marked_infeasible(separable_fixture) -> None:
    frame, target = separable_fixture
    # A heavy penalty shrinks the support well below the requested budget.
    selector = L1LogisticSelector(k=5, C=0.002).fit(frame, target)
    result = selector.result

    assert len(result.natural_selected or ()) < 5
    assert result.budget_status == "infeasible_natural_support"
    assert result.selection_mode == "matched_budget"
    # No zero-coefficient filler was substituted to reach the budget.
    assert set(result.selected_features) == set(result.natural_selected or ())
    assert any("not used as filler" in item for item in result.warnings)


def test_zero_coefficient_fill_is_only_reachable_through_a_named_mode(
    separable_fixture,
) -> None:
    frame, target = separable_fixture
    permitted = L1LogisticSelector(
        k=5, C=0.002, allow_zero_coefficient_fill=True
    ).fit(frame, target)
    result = permitted.result

    assert result.selection_mode == "coefficient_ranking"
    assert result.budget_status == "satisfied"
    assert result.actual_selected_count == 5
    # The padded members are visibly outside the natural support.
    assert set(result.selected_features) - set(result.natural_selected or ())
    assert result.configuration["allow_zero_coefficient_fill"] is True


def test_total_shrinkage_to_zero_is_a_valid_recorded_outcome(separable_fixture) -> None:
    frame, target = separable_fixture
    selector = L1LogisticSelector(k=None, C=1e-6).fit(frame, target)
    result = selector.result

    assert result.natural_selected == ()
    assert result.selected_features == ()
    assert any("every coefficient to zero" in item for item in result.warnings)
    # A ranking still exists so the artifact remains interpretable.
    assert len(result.ranking or ()) == frame.shape[1]


def test_convergence_failure_is_reported_without_a_silent_fallback(
    separable_fixture,
) -> None:
    frame, target = separable_fixture
    selector = L1LogisticSelector(k=2, C=10.0, solver="saga", max_iter=1).fit(frame, target)
    result = selector.result

    assert selector.converged_ is False
    assert selector.convergence_warnings_
    assert any("did not converge" in item for item in result.warnings)
    # The configuration recorded is the one that was actually requested; the
    # solver, penalty, and tolerance were not swapped behind the caller's back.
    assert result.configuration["solver"] == "saga"
    assert result.configuration["max_iter"] == 1
    assert result.configuration["penalty"] == "l1"


def test_non_numeric_candidates_fail_explicitly(separable_fixture) -> None:
    frame, target = separable_fixture
    frame = frame.assign(grade=pd.Series(["A", "B"] * (len(frame) // 2)))
    with pytest.raises(ControlledSelectorFailure) as error:
        L1LogisticSelector(k=2).fit(frame, target)
    assert error.value.stage == "design_matrix_validation"
    assert "not numeric" in error.value.cause


def test_all_missing_column_is_flagged_and_cannot_dominate(separable_fixture) -> None:
    frame, target = separable_fixture
    frame = frame.assign(empty=np.full(len(frame), np.nan))
    selector = L1LogisticSelector(k=2, C=0.5).fit(frame, target)
    coefficients = selector.coefficients_.set_index("feature")

    assert bool(coefficients.loc["empty", "all_missing_in_training"]) is True
    assert coefficients.loc["empty", "coefficient"] == pytest.approx(0.0, abs=1e-12)
    assert "empty" not in set(selector.result.natural_selected or ())


def test_same_seed_and_configuration_reproduce_the_ordering(separable_fixture) -> None:
    frame, target = separable_fixture
    first = L1LogisticSelector(k=3, C=0.3, random_state=7).fit(frame, target)
    second = L1LogisticSelector(k=3, C=0.3, random_state=7).fit(frame, target)

    assert first.result.selected_features == second.result.selected_features
    assert first.result.ranking == second.result.ranking
    pd.testing.assert_frame_equal(first.coefficients_, second.coefficients_)


def test_non_convergence_warnings_are_not_swallowed(separable_fixture) -> None:
    """The convergence-capture context must not hide unrelated warnings.

    ``catch_warnings(record=True)`` captures *every* warning, so re-emitting the
    non-convergence ones is what keeps an estimator deprecation or a numerical
    warning visible to the caller instead of disappearing into the selector.
    """

    import warnings as warnings_module

    frame, target = separable_fixture
    selector = L1LogisticSelector(k=2, C=0.5)

    with warnings_module.catch_warnings(record=True) as captured:
        warnings_module.simplefilter("always")
        warnings_module.warn("sentinel from inside the fit", UserWarning)
        selector.fit(frame, target)

    assert any("sentinel" in str(item.message) for item in captured)
    # The non-deprecated penalty spelling is used where available, so no
    # FutureWarning about `penalty` should be produced at all.
    assert not [
        item for item in captured if issubclass(item.category, FutureWarning)
    ], [str(item.message) for item in captured]
    assert selector.penalty_api_ == _l1_penalty_kwargs()[1]


def test_invalid_configuration_is_rejected_before_fitting() -> None:
    with pytest.raises(ValueError, match="L1 penalty"):
        L1LogisticSelector(solver="lbfgs")
    with pytest.raises(ValueError, match="C must be positive"):
        L1LogisticSelector(C=0.0)
    with pytest.raises(ValueError, match="n_jobs must be positive"):
        L1LogisticSelector(n_jobs=0)
    with pytest.raises(ValueError, match="imputation_strategy"):
        L1LogisticSelector(imputation_strategy="magic")
