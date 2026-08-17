"""Behaviour and identity tests for canonical mutual-information mRMR."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mutual_info_score

from credit_risk_fs.selectors.lightweight.mi_mrmr import (
    MISSING_CODE,
    MutualInformationMRMRSelector,
    _discretize_column,
)
from credit_risk_fs.selectors.lightweight.mrmr_compact_cache import (
    CompactMRMRCheckpointError,
)
from credit_risk_fs.selectors.mrmr import RandomForestRelevanceMRMRSelector


@pytest.fixture()
def signal_fixture() -> tuple[pd.DataFrame, pd.Series]:
    generator = np.random.default_rng(19)
    n = 1_500
    latent = generator.normal(size=n)
    target = pd.Series((latent + generator.normal(scale=0.5, size=n) > 0).astype(int))
    frame = pd.DataFrame(
        {
            "strong": latent,
            "strong_duplicate": latent.copy(),
            "weak": latent + generator.normal(scale=4.0, size=n),
            "noise": generator.normal(size=n),
        }
    )
    return frame, target


def test_relevant_feature_is_selected_before_noise(signal_fixture) -> None:
    frame, target = signal_fixture
    selector = MutualInformationMRMRSelector(k=1).fit(frame, target)
    assert selector.result.selected_features[0] in {"strong", "strong_duplicate"}
    relevance = selector.relevance_
    assert relevance["strong"] > relevance["weak"] > relevance["noise"]


def test_exact_duplicate_is_penalised_once_its_source_is_selected(signal_fixture) -> None:
    frame, target = signal_fixture
    selector = MutualInformationMRMRSelector(k=4).fit(frame, target)
    ranking = list(selector.result.ranking or ())

    first = ranking[0]
    duplicate = "strong_duplicate" if first == "strong" else "strong"
    # The duplicate carries near-maximal relevance yet is pushed behind weaker,
    # independent features purely by the redundancy term.
    assert ranking.index(duplicate) > ranking.index("noise")

    trace = selector.selection_trace_.set_index("feature")
    assert float(trace.loc[duplicate, "mean_redundancy"]) > float(
        trace.loc["noise", "mean_redundancy"]
    )
    assert float(trace.loc[duplicate, "score"]) < float(trace.loc["noise", "score"])


def test_relevance_matches_an_independently_computed_plug_in_estimate(signal_fixture) -> None:
    frame, target = signal_fixture
    selector = MutualInformationMRMRSelector(k=2, n_bins=10).fit(frame, target)

    # Recompute one relevance value from the discretizer plus sklearn directly.
    codes = _discretize_column(frame["weak"], 10)
    expected = float(mutual_info_score(codes, target.to_numpy()))
    assert selector.relevance_["weak"] == pytest.approx(expected, abs=1e-12)


def test_selection_is_deterministic_and_order_independent(signal_fixture) -> None:
    frame, target = signal_fixture
    first = MutualInformationMRMRSelector(k=3, random_state=42).fit(frame, target)
    second = MutualInformationMRMRSelector(k=3, random_state=42).fit(frame, target)
    shuffled = MutualInformationMRMRSelector(k=3, random_state=42).fit(
        frame[["noise", "weak", "strong_duplicate", "strong"]], target
    )

    assert first.result.selected_features == second.result.selected_features
    # Identical relevance between the exact duplicates means the first pick is a
    # true tie; the name rule must resolve it the same way under any column order.
    assert first.result.selected_features == shuffled.result.selected_features


def test_exact_ties_resolve_on_the_ascending_feature_name() -> None:
    target = pd.Series([0, 1] * 200)
    column = np.tile([0.0, 1.0], 200)
    frame = pd.DataFrame({"zulu": column, "alpha": column, "kilo": column})
    selector = MutualInformationMRMRSelector(k=1).fit(frame, target)
    assert selector.result.selected_features == ("alpha",)


def test_zero_relevance_features_are_held_behind_informative_ones() -> None:
    generator = np.random.default_rng(5)
    n = 800
    latent = generator.normal(size=n)
    target = pd.Series((latent > 0).astype(int))
    frame = pd.DataFrame(
        {
            "constant": np.ones(n),
            "informative": latent,
            "all_missing": np.full(n, np.nan),
        }
    )
    selector = MutualInformationMRMRSelector(k=3).fit(frame, target)
    ranking = list(selector.result.ranking or ())
    assert ranking[0] == "informative"
    assert set(ranking[1:]) == {"all_missing", "constant"}
    assert selector.relevance_["constant"] == pytest.approx(0.0, abs=1e-15)

    # A budget of one must never spend itself on a zero-information column.
    single = MutualInformationMRMRSelector(k=1).fit(frame, target)
    assert single.result.selected_features == ("informative",)


def test_all_zero_relevance_universe_still_returns_a_deterministic_subset() -> None:
    target = pd.Series([0, 1] * 100)
    frame = pd.DataFrame(
        {"zulu": np.ones(200), "alpha": np.ones(200), "mike": np.zeros(200)}
    )
    selector = MutualInformationMRMRSelector(k=2).fit(frame, target)
    assert selector.result.selected_features == ("alpha", "mike")
    assert selector.result.budget_status == "satisfied"


def test_discretizer_keeps_missingness_as_its_own_code() -> None:
    values = pd.Series([1.0, 2.0, np.nan, 4.0, np.inf])
    codes = _discretize_column(values, 4)
    assert codes[2] == MISSING_CODE
    # A non-finite value is treated as missing rather than as a huge magnitude.
    assert codes[4] == MISSING_CODE
    assert set(codes[[0, 1, 3]]) != {MISSING_CODE}


def test_categorical_codes_do_not_depend_on_row_order() -> None:
    forward = _discretize_column(pd.Series(["b", "a", "c", "a"]), 10)
    backward = _discretize_column(pd.Series(["c", "a", "b", "a"]), 10)
    # Level "a" must receive the same code in both, because codes come from the
    # sorted level names rather than from first appearance.
    assert forward[1] == backward[1]
    assert forward[0] == backward[2]


def test_compact_disk_checkpoints_are_exact_restartable_and_authenticated(
    tmp_path: Path,
) -> None:
    generator = np.random.default_rng(20260817)
    rows = 257
    latent = generator.normal(size=rows).astype("float32")
    frame = pd.DataFrame(
        {
            "signal": latent,
            "duplicate": latent.copy(),
            "weak": (latent + generator.normal(scale=2.0, size=rows)).astype(
                "float32"
            ),
            "missing": np.where(
                np.arange(rows) % 7 == 0, np.nan, latent * 0.25
            ).astype("float32"),
            "nonfinite": np.where(
                np.arange(rows) % 11 == 0, np.inf, -latent
            ).astype("float32"),
            "noise": generator.normal(size=rows).astype("float32"),
        }
    )
    target = pd.Series(
        (latent + generator.normal(scale=0.65, size=rows) > 0).astype("int64")
    )
    legacy = MutualInformationMRMRSelector(k=5, n_bins=10).fit(frame, target)
    identity = {
        "fixture": "representative_numeric_float32_with_missing_nonfinite_v1",
        "rows": rows,
        "features": list(frame.columns),
    }
    root = tmp_path / "compact"
    cached = MutualInformationMRMRSelector(k=5, n_bins=10)
    cached.configure_execution_cache(
        root,
        execution_identity=identity,
        feature_batch_size=2,
    ).fit(frame, target)

    assert cached.result.selected_features == legacy.result.selected_features
    assert cached.result.ranking == legacy.result.ranking
    assert cached.result.raw_scores == legacy.result.raw_scores
    assert cached.result.configuration == legacy.result.configuration
    assert cached.relevance_ == legacy.relevance_
    pd.testing.assert_frame_equal(
        cached.selection_trace_, legacy.selection_trace_, check_exact=True
    )
    summary = cached.execution_checkpoint_summary_
    assert summary["storage_dtype"] == "int8"
    assert summary["row_count"] == rows
    assert summary["candidate_count"] == len(frame.columns)
    assert summary["scientific_semantics_changed"] is False

    first_batch = np.load(
        root / "code_batches" / "batch_001" / "codes.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    try:
        assert first_batch.dtype == np.dtype("int8")
        assert np.array_equal(
            first_batch[0].astype("int64"),
            _discretize_column(frame["signal"], 10),
        )
        assert np.array_equal(
            first_batch[1].astype("int64"),
            _discretize_column(frame["duplicate"], 10),
        )
    finally:
        first_batch._mmap.close()

    manifests_before = {
        path.relative_to(root).as_posix(): path.stat().st_mtime_ns
        for path in root.rglob("manifest.json")
    }
    resumed = MutualInformationMRMRSelector(k=5, n_bins=10)
    resumed.configure_execution_cache(
        root,
        execution_identity=identity,
        feature_batch_size=2,
    ).fit(frame, target)
    manifests_after = {
        path.relative_to(root).as_posix(): path.stat().st_mtime_ns
        for path in root.rglob("manifest.json")
    }
    assert resumed.result.selected_features == legacy.result.selected_features
    assert resumed.result.raw_scores == legacy.result.raw_scores
    assert manifests_after == manifests_before

    relevance_manifest = root / "mi" / "relevance" / "manifest.json"
    manifest = json.loads(relevance_manifest.read_text(encoding="utf-8"))
    manifest["identity"]["shape"] = [999]
    relevance_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    rejected = MutualInformationMRMRSelector(k=5, n_bins=10)
    rejected.configure_execution_cache(
        root,
        execution_identity=identity,
        feature_batch_size=2,
    )
    with pytest.raises(CompactMRMRCheckpointError, match="marker changed"):
        rejected.fit(frame, target)


def test_objective_choice_is_recorded_and_validated(signal_fixture) -> None:
    frame, target = signal_fixture
    selector = MutualInformationMRMRSelector(k=2, objective="miq").fit(frame, target)
    configuration = selector.result.configuration
    assert configuration["objective"] == "miq"
    assert configuration["mi_estimator"] == "sklearn.metrics.mutual_info_score"
    assert configuration["deterministic_without_rng"] is True

    with pytest.raises(ValueError, match="objective must be one of"):
        MutualInformationMRMRSelector(k=2, objective="something_else")


def test_labels_outside_the_supplied_partition_cannot_influence_the_result() -> None:
    generator = np.random.default_rng(23)
    n = 600
    latent = generator.normal(size=n)
    honest = pd.Series((latent > 0).astype(int))
    frame = pd.DataFrame({"latent": latent, "noise": generator.normal(size=n)})

    half = n // 2
    inside = MutualInformationMRMRSelector(k=1).fit(frame.iloc[:half], honest.iloc[:half])

    # Corrupting the labels of rows the selector never receives must not move it.
    corrupted = honest.copy()
    corrupted.iloc[half:] = 1 - corrupted.iloc[half:]
    recomputed = MutualInformationMRMRSelector(k=1).fit(
        frame.iloc[:half], corrupted.iloc[:half]
    )
    assert inside.result.selected_features == recomputed.result.selected_features
    assert inside.relevance_ == recomputed.relevance_


def test_canonical_and_legacy_methods_are_distinguishable() -> None:
    assert MutualInformationMRMRSelector.method_id == "mrmr_mutual_information"
    assert MutualInformationMRMRSelector.implementation_id == (
        "mrmr_mutual_information_discrete_plugin_v1"
    )
    assert RandomForestRelevanceMRMRSelector.canonical_mrmr is False
    assert RandomForestRelevanceMRMRSelector.algorithm_name == (
        "rf_relevance_correlation_redundancy"
    )
    assert not issubclass(MutualInformationMRMRSelector, RandomForestRelevanceMRMRSelector)
