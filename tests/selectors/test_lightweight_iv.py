"""Oracle tests for the standalone Information Value selector.

The primary test recomputes WOE and IV from explicit arithmetic rather than from
the production code path, so it fails if the implementation's definition drifts.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.selectors.lightweight.contract import ControlledSelectorFailure
from credit_risk_fs.selectors.lightweight.iv import (
    MISSING_BIN,
    InformationValueSelector,
    compute_feature_iv,
)


def _three_level_fixture() -> tuple[pd.Series, pd.Series]:
    """100 rows per grade with 10 / 30 / 60 bads respectively."""

    grades: list[str] = []
    targets: list[int] = []
    for grade, bad_count in (("A", 10), ("B", 30), ("C", 60)):
        grades.extend([grade] * 100)
        targets.extend([1] * bad_count + [0] * (100 - bad_count))
    return pd.Series(grades, name="grade"), pd.Series(targets, name="target")


def test_information_value_matches_a_hand_calculated_woe_table() -> None:
    grades, target = _three_level_fixture()

    # Independent arithmetic: distributions are shares *within* each class.
    total_bad, total_good = 100, 200
    hand_total = 0.0
    expected_woe = {}
    for bad, good, level in ((10, 90, "A"), (30, 70, "B"), (60, 40, "C")):
        dist_bad = bad / total_bad
        dist_good = good / total_good
        woe = math.log(dist_good / dist_bad)
        expected_woe[level] = woe
        hand_total += (dist_good - dist_bad) * woe

    # ln(4.5), ln(7/6), ln(1/3) -> 1.504077, 0.154151, -1.098612
    assert expected_woe["A"] == pytest.approx(1.5040774, abs=1e-6)
    assert expected_woe["B"] == pytest.approx(0.1541507, abs=1e-6)
    assert expected_woe["C"] == pytest.approx(-1.0986123, abs=1e-6)
    assert hand_total == pytest.approx(0.9735795, abs=1e-6)

    iv, table = compute_feature_iv(grades, target, smoothing=0.0)
    assert iv == pytest.approx(hand_total, abs=1e-12)

    by_bin = table.set_index("bin")
    for level, woe in expected_woe.items():
        assert by_bin.loc[level, "woe"] == pytest.approx(woe, abs=1e-12)
    assert by_bin["count"].sum() == 300
    assert by_bin["bad"].sum() == 100
    # The published bin table must be sufficient to recalculate the total.
    assert by_bin["iv_contribution"].sum() == pytest.approx(iv, abs=1e-12)


def test_information_value_agrees_with_the_installed_third_party_estimator() -> None:
    """Cross-check against ``iv_woe_filter`` where both definitions coincide.

    Both use ``WOE = ln(dist_good / dist_bad)`` and ``IV = sum((dist_good -
    dist_bad) * WOE)``. With smoothing disabled here, no empty class cell, and
    bin merging switched off there, the two must agree to floating-point noise.
    """

    iv_woe_filter = pytest.importorskip("iv_woe_filter")
    grades, target = _three_level_fixture()

    reference = iv_woe_filter.IVWOEFilter(
        min_bin_pct=None,
        drop_low_iv=False,
        encode=False,
        n_jobs=1,
        verbose=False,
    ).fit(pd.DataFrame({"grade": grades}), target)

    ours, _ = compute_feature_iv(grades, target, smoothing=0.0)
    assert ours == pytest.approx(float(reference.iv_table_data_["grade"]), abs=1e-9)


def test_smoothing_keeps_a_zero_count_bin_finite_and_configurable() -> None:
    # Bin "C" contains only bads, so the unsmoothed WOE is -inf.
    labels = pd.Series(["A"] * 100 + ["B"] * 100 + ["C"] * 20)
    target = pd.Series([1] * 20 + [0] * 80 + [1] * 50 + [0] * 50 + [1] * 20)

    unsmoothed, unsmoothed_table = compute_feature_iv(labels, target, smoothing=0.0)
    smoothed, smoothed_table = compute_feature_iv(labels, target, smoothing=0.5)

    assert np.isfinite(unsmoothed)
    # The degenerate bin is neutralised rather than allowed to emit inf.
    assert unsmoothed_table.loc[unsmoothed_table["bin"] == "C", "iv_contribution"].iloc[0] == 0.0
    assert np.isfinite(smoothed)
    assert smoothed > unsmoothed
    assert np.isfinite(smoothed_table["woe"]).all()


def test_missing_values_form_their_own_bin_and_can_carry_information() -> None:
    values = pd.Series([1.0] * 200 + [np.nan] * 200)
    target = pd.Series([0] * 200 + [1] * 200)
    iv, table = compute_feature_iv(
        pd.Series(["bin_01"] * 200 + [MISSING_BIN] * 200), target, smoothing=0.5
    )
    assert MISSING_BIN in set(table["bin"])
    assert iv > 1.0

    selector = InformationValueSelector(k=1).fit(
        pd.DataFrame({"missingness": values}), target
    )
    assert selector.result.selected_features == ("missingness",)
    diagnostics = selector.feature_diagnostics_.set_index("feature")
    assert int(diagnostics.loc["missingness", "missing_count"]) == 200


def test_stronger_feature_outranks_weaker_and_noise() -> None:
    generator = np.random.default_rng(11)
    n = 2_000
    latent = generator.normal(size=n)
    target = pd.Series((latent + generator.normal(scale=0.5, size=n) > 0).astype(int))
    frame = pd.DataFrame(
        {
            "strong": latent,
            "weak": latent + generator.normal(scale=3.0, size=n),
            "noise": generator.normal(size=n),
        }
    )
    selector = InformationValueSelector(k=2).fit(frame, target)
    ranking = list(selector.result.ranking or ())
    assert ranking[0] == "strong"
    assert ranking.index("weak") < ranking.index("noise")
    assert selector.result.selected_features == ("strong", "weak")


def test_degenerate_columns_score_zero_without_producing_nan() -> None:
    target = pd.Series([0, 1] * 100)
    frame = pd.DataFrame(
        {
            "constant": np.ones(200),
            "all_missing": np.full(200, np.nan),
            "real": np.arange(200, dtype="float64"),
        }
    )
    selector = InformationValueSelector(k=3).fit(frame, target)
    scores = dict(selector.result.raw_scores or {})
    assert scores["constant"] == pytest.approx(0.0, abs=1e-12)
    assert scores["all_missing"] == pytest.approx(0.0, abs=1e-12)
    assert all(np.isfinite(value) for value in scores.values())
    diagnostics = selector.feature_diagnostics_.set_index("feature")
    assert bool(diagnostics.loc["constant", "degenerate"]) is True
    assert bool(diagnostics.loc["all_missing", "degenerate"]) is True
    assert int(diagnostics.loc["constant", "realized_bins"]) == 1


def test_equal_information_value_breaks_ties_on_the_feature_name() -> None:
    target = pd.Series([0, 1] * 150)
    column = pd.Series(([0.0] * 2 + [1.0] * 2) * 75)
    frame = pd.DataFrame({"zebra": column, "alpha": column, "mango": column})
    selector = InformationValueSelector(k=3).fit(frame, target)
    scores = dict(selector.result.raw_scores or {})
    assert scores["alpha"] == pytest.approx(scores["zebra"], abs=1e-15)
    assert list(selector.result.ranking or ()) == ["alpha", "mango", "zebra"]

    # Reordering the input columns must not reorder the output.
    reordered = InformationValueSelector(k=3).fit(frame[["mango", "zebra", "alpha"]], target)
    assert list(reordered.result.ranking or ()) == ["alpha", "mango", "zebra"]


def test_high_cardinality_categorical_fails_explicitly() -> None:
    target = pd.Series([0, 1] * 100)
    frame = pd.DataFrame({"identifier": [f"id_{index}" for index in range(200)]})
    with pytest.raises(ControlledSelectorFailure) as error:
        InformationValueSelector(k=1, max_categorical_levels=10).fit(frame, target)
    assert error.value.stage == "binning"
    assert "max_categorical_levels" in error.value.cause


def test_binning_uses_only_the_rows_supplied_to_fit() -> None:
    """Edges learned on training rows must ignore an extreme unseen value."""

    target = pd.Series([0, 1] * 100)
    training = pd.DataFrame({"amount": np.linspace(0.0, 100.0, 200)})
    selector = InformationValueSelector(k=1, n_bins=4).fit(training, target)
    edges_seen = selector.bin_table_["upper_bound"].dropna()
    assert edges_seen.max() == np.inf
    finite_upper = edges_seen[np.isfinite(edges_seen)]
    # Every finite cut point sits inside the training range, so no validation or
    # OOT value could have contributed to it.
    assert finite_upper.max() <= 100.0
