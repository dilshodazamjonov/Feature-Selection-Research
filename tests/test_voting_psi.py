"""Prompt 6 PSI tests for score drift and type-aware feature drift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.analysis.voting_inference.psi import (
    MISSING_STATE,
    UNSEEN_STATE,
    classify_feature_type,
    feature_psi_record,
    score_psi_from_predictions,
    summarise_feature_psi,
    type_aware_feature_psi,
)
from scripts.independently_verify_voting_metrics import independent_score_psi


def _legacy_categorical_psi(
    dev: pd.Series, oot: pd.Series
) -> tuple[float, pd.DataFrame, dict[str, object]]:
    """Small-fixture reference for the exact pre-optimization semantics."""

    from credit_risk_fs.analysis.voting_inference.psi import (
        FEATURE_PSI_EPSILON,
        _share_table,
    )

    dev_states = dev.astype("object").map(
        lambda value: MISSING_STATE if pd.isna(value) else str(value)
    )
    oot_states = oot.astype("object").map(
        lambda value: MISSING_STATE if pd.isna(value) else str(value)
    )
    dev_levels = sorted(set(dev_states.unique()))
    unseen = sorted(set(oot_states.unique()) - set(dev_levels))
    oot_states = oot_states.map(
        lambda value: UNSEEN_STATE if value in set(unseen) else value
    )
    states = [*dev_levels, UNSEEN_STATE]
    frame, psi = _share_table(dev_states, oot_states, states)
    return psi, frame, {
        "dev_levels": dev_levels,
        "unseen_oot_level_count": len(unseen),
        "smoothing_epsilon": FEATURE_PSI_EPSILON,
    }


def test_unchanged_score_distribution_gives_approximately_zero_psi() -> None:
    generator = np.random.default_rng(2)
    scores = generator.random(5_000)
    result = score_psi_from_predictions(scores, scores)
    assert result.psi == pytest.approx(0.0, abs=1e-9)
    assert result.definition["reference_scope"] == "DEV_OOF"
    assert result.definition["comparison_scope"] == "oot"
    assert result.definition["binning_method"] == "DEV_OOF_quantile"


def test_shifted_score_distribution_gives_positive_psi() -> None:
    generator = np.random.default_rng(3)
    dev = generator.beta(2, 8, 8_000)
    oot = generator.beta(5, 5, 8_000)
    result = score_psi_from_predictions(dev, oot)
    assert result.psi > 0.1
    assert result.bins["psi_contribution"].sum() == pytest.approx(result.psi, abs=1e-12)


def test_score_psi_bins_are_defined_on_dev_and_applied_unchanged_to_oot() -> None:
    dev = np.linspace(0.0, 1.0, 1_000)
    oot = np.linspace(0.0, 0.5, 1_000)
    result = score_psi_from_predictions(dev, oot)
    assert result.bins["reference_count"].sum() == 1_000
    assert result.bins["comparison_count"].sum() == 1_000
    # DEV shares stay uniform because the bins came from DEV alone.
    assert result.bins["reference_share"].std() == pytest.approx(0.0, abs=1e-3)
    assert result.bins.loc[result.bins["lower_bound"] >= 0.5, "comparison_count"].sum() == 0


def test_score_psi_matches_the_independent_implementation() -> None:
    generator = np.random.default_rng(5)
    dev = generator.random(4_000)
    oot = np.clip(generator.random(4_000) * 0.8 + 0.1, 0.0, 1.0)
    primary = score_psi_from_predictions(dev, oot).psi
    assert primary == pytest.approx(independent_score_psi(dev, oot), abs=1e-12)


def test_score_psi_collapses_duplicate_quantile_edges() -> None:
    dev = np.array([0.2] * 900 + list(np.linspace(0.5, 0.9, 100)))
    oot = np.array([0.2] * 500 + list(np.linspace(0.5, 0.9, 500)))
    result = score_psi_from_predictions(dev, oot)
    edges = result.definition["bin_edges"]
    assert len(edges) == len(set(edges))
    assert result.definition["effective_bin_count"] == len(edges) - 1
    assert result.definition["duplicate_edge_policy"] == (
        "sort_unique_candidate_quantile_edges"
    )


def test_score_psi_rejects_probabilities_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        score_psi_from_predictions([0.1, 0.5], [0.2, 1.5])


def test_score_psi_rejects_missing_probabilities() -> None:
    with pytest.raises(ValueError):
        score_psi_from_predictions([0.1, 0.5], [0.2, float("nan")])


def test_numeric_feature_psi_retains_an_explicit_missing_state() -> None:
    dev = pd.Series(list(np.linspace(0, 100, 900)) + [np.nan] * 100)
    oot = pd.Series(list(np.linspace(0, 100, 500)) + [np.nan] * 500)
    psi, distribution, definition = type_aware_feature_psi(dev, oot)
    assert definition["feature_type"] == "numeric"
    assert definition["missing_handling"] == "explicit_missing_state"
    assert MISSING_STATE in set(distribution["state"])
    missing_row = distribution.loc[distribution["state"] == MISSING_STATE].iloc[0]
    assert missing_row["dev_share"] == pytest.approx(0.1)
    assert missing_row["oot_share"] == pytest.approx(0.5)
    assert psi > 0.2


def test_identical_numeric_distribution_gives_approximately_zero_feature_psi() -> None:
    generator = np.random.default_rng(6)
    values = pd.Series(generator.normal(size=3_000))
    psi, _, _ = type_aware_feature_psi(values, values)
    assert psi == pytest.approx(0.0, abs=1e-9)


def test_numeric_feature_psi_survives_edges_that_print_identically() -> None:
    # Regression: quantile edges on a large-magnitude, low-spread feature round
    # to the same text at ten significant digits, so bin labels must not be
    # derived from the formatted edge values.
    dev = pd.Series(np.linspace(1e9, 1e9 + 0.01, 1_000))
    oot = pd.Series(np.linspace(1e9, 1e9 + 0.005, 1_000))
    formatted = {f"{value:.10g}" for value in np.percentile(dev, np.linspace(0, 100, 11))}
    assert len(formatted) < 11, "fixture must produce colliding formatted edges"
    psi, distribution, definition = type_aware_feature_psi(dev, oot)
    assert definition["feature_type"] == "numeric"
    assert distribution["state"].is_unique
    assert distribution["dev_count"].sum() == 1_000
    assert distribution["oot_count"].sum() == 1_000
    assert {"lower_bound", "upper_bound"}.issubset(distribution.columns)
    assert psi > 0.0


def test_numeric_feature_psi_reports_bin_bounds_separately_from_labels() -> None:
    dev = pd.Series(np.linspace(0.0, 100.0, 1_000))
    _, distribution, _ = type_aware_feature_psi(dev, dev)
    bins = distribution.loc[distribution["state"] != MISSING_STATE]
    assert list(bins["state"]) == sorted(bins["state"])
    assert bins["lower_bound"].iloc[0] == -np.inf
    assert bins["upper_bound"].iloc[-1] == np.inf
    missing = distribution.loc[distribution["state"] == MISSING_STATE].iloc[0]
    assert pd.isna(missing["lower_bound"])


def test_share_table_rejects_lost_observations() -> None:
    from credit_risk_fs.analysis.voting_inference.psi import _share_table

    dev = pd.Series(["a", "b", "unlisted"])
    oot = pd.Series(["a", "a", "b"])
    with pytest.raises(ValueError, match="lost 1 DEV observation"):
        _share_table(dev, oot, ["a", "b"])
    with pytest.raises(ValueError, match="state labels must be unique"):
        _share_table(pd.Series(["a"]), pd.Series(["a"]), ["a", "a"])


def test_numeric_feature_psi_keeps_underflow_and_overflow_rows() -> None:
    dev = pd.Series(np.linspace(10.0, 20.0, 1_000))
    oot = pd.Series([-5.0] * 100 + list(np.linspace(10.0, 20.0, 800)) + [99.0] * 100)
    psi, distribution, definition = type_aware_feature_psi(dev, oot)
    assert definition["underflow_overflow_handling"] == (
        "retained_by_infinite_outer_edges"
    )
    assert distribution["oot_count"].sum() == 1_000
    assert psi > 0.0


def test_categorical_feature_psi_marks_unseen_oot_levels() -> None:
    dev = pd.Series(["a"] * 600 + ["b"] * 400)
    oot = pd.Series(["a"] * 300 + ["b"] * 300 + ["c"] * 400)
    psi, distribution, definition = type_aware_feature_psi(dev, oot)
    assert definition["feature_type"] == "categorical"
    assert definition["unseen_oot_level_count"] == 1
    assert UNSEEN_STATE in set(distribution["state"])
    unseen = distribution.loc[distribution["state"] == UNSEEN_STATE].iloc[0]
    assert unseen["dev_count"] == 0
    assert unseen["oot_count"] == 400
    assert psi > 0.0


def test_categorical_missing_values_become_an_explicit_state() -> None:
    dev = pd.Series(["a"] * 500 + [None] * 500)
    oot = pd.Series(["a"] * 900 + [None] * 100)
    _, distribution, _ = type_aware_feature_psi(dev, oot)
    row = distribution.loc[distribution["state"] == MISSING_STATE].iloc[0]
    assert row["dev_share"] == pytest.approx(0.5)
    assert row["oot_share"] == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("dev", "oot"),
    [
        (
            pd.Series(["a", None, "b", "a", pd.NA], dtype="object"),
            pd.Series(["b", "c", None, "a", "c"], dtype="object"),
        ),
        (
            pd.Series([1, 2, None, 2, 1], dtype="Int64"),
            pd.Series([1, 3, None, 3, 2], dtype="Int64"),
        ),
        (
            pd.Series([True, False, None, True], dtype="boolean"),
            pd.Series([False, None, True, False], dtype="boolean"),
        ),
        (
            pd.Series([1.25, np.nan, 1.25, 1.25]),
            pd.Series([2.5, np.nan, 1.25, 2.5]),
        ),
    ],
)
def test_optimized_categorical_psi_is_exactly_equivalent_to_legacy_semantics(
    dev: pd.Series, oot: pd.Series
) -> None:
    expected_psi, expected_distribution, expected = _legacy_categorical_psi(
        dev, oot
    )
    actual_psi, actual_distribution, actual = type_aware_feature_psi(dev, oot)
    assert actual_psi == expected_psi
    pd.testing.assert_frame_equal(
        actual_distribution,
        expected_distribution,
        check_exact=True,
    )
    assert actual["unseen_oot_level_count"] == expected["unseen_oot_level_count"]
    assert actual["smoothing_epsilon"] == expected["smoothing_epsilon"]


def test_all_missing_dev_high_cardinality_oot_collapses_without_losing_rows() -> None:
    dev = pd.Series([np.nan] * 20_000)
    oot = pd.Series(np.arange(10_000, dtype=float))
    psi, distribution, definition = type_aware_feature_psi(dev, oot)
    assert np.isfinite(psi)
    assert definition["feature_type"] == "all_missing_in_dev"
    assert definition["unseen_oot_level_count"] == 10_000
    assert list(distribution["state"]) == [MISSING_STATE, UNSEEN_STATE]
    assert distribution["dev_count"].tolist() == [20_000, 0]
    assert distribution["oot_count"].tolist() == [0, 10_000]


def test_feature_type_classification_covers_the_reported_families() -> None:
    assert classify_feature_type(pd.Series([0, 1, 0, 1])) == "binary"
    assert classify_feature_type(pd.Series([True, False])) == "binary"
    assert classify_feature_type(pd.Series(["x", "y", "z"])) == "categorical"
    assert classify_feature_type(pd.Series(np.linspace(0, 1, 200))) == "numeric"
    assert (
        classify_feature_type(pd.Series([1, 2, 3, 4, 5, 6, 7]))
        == "encoded_low_cardinality_integer"
    )
    # Degenerate columns are named accurately rather than mislabelled "binary",
    # because the label is published in feature_psi_definition_audit.csv.
    assert classify_feature_type(pd.Series([np.nan, np.nan])) == "all_missing_in_dev"
    assert classify_feature_type(pd.Series([7.0, 7.0, np.nan])) == "constant_in_dev"


def test_degenerate_columns_still_produce_a_valid_psi_record() -> None:
    for dev, oot, expected in (
        (pd.Series([np.nan] * 50), pd.Series([np.nan] * 30), "all_missing_in_dev"),
        (pd.Series([2.0] * 50), pd.Series([2.0] * 30), "constant_in_dev"),
        (pd.Series([2.0] * 50), pd.Series(np.linspace(0, 9, 30)), "constant_in_dev"),
    ):
        record, distribution = feature_psi_record(
            feature="degenerate", dev_values=dev, oot_values=oot
        )
        assert record["feature_type"] == expected
        assert record["psi_type_aware_available"] is True
        assert int(distribution["dev_count"].sum()) == len(dev)
        assert int(distribution["oot_count"].sum()) == len(oot)
        assert np.isfinite(record["psi_type_aware"])


def test_feature_psi_record_reports_both_definitions_and_availability() -> None:
    dev = pd.Series(["a"] * 600 + ["b"] * 400)
    oot = pd.Series(["a"] * 500 + ["b"] * 500)
    record, distribution = feature_psi_record(
        feature="grade", dev_values=dev, oot_values=oot
    )
    assert record["feature_type"] == "categorical"
    assert record["psi_frozen_numeric_available"] is False
    assert record["psi_type_aware_available"] is True
    assert record["psi_type_aware"] > 0.0
    assert not distribution.empty

    numeric_dev = pd.Series(np.linspace(0, 1, 500))
    numeric_record, _ = feature_psi_record(
        feature="ratio", dev_values=numeric_dev, oot_values=numeric_dev
    )
    assert numeric_record["psi_frozen_numeric_available"] is True
    assert numeric_record["psi_frozen_numeric"] == pytest.approx(0.0, abs=1e-9)


def test_feature_psi_summary_reports_availability_and_reference_shares() -> None:
    frame = pd.DataFrame(
        {
            "psi_frozen_numeric": [0.05, 0.3, None],
            "psi_type_aware": [0.05, 0.3, 0.4],
        }
    )
    summary = summarise_feature_psi(frame, references=[0.1, 0.25])
    assert summary["selected_features_evaluated"] == 3
    assert summary["frozen_numeric_available_count"] == 2
    assert summary["frozen_numeric_unavailable_count"] == 1
    assert summary["type_aware_available_count"] == 3
    assert summary["type_aware_share_above_0p25"] == pytest.approx(2 / 3)
