from __future__ import annotations

import gc
import hashlib
import json
import weakref
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from credit_risk_fs.experiments.config import compute_config_hash
from credit_risk_fs.experiments.matrix import (
    expand_lendingclub_memory_capacity_scenarios,
)
from credit_risk_fs.experiments.prediction_contract import (
    CAPACITY_SINGLE_FOLD_COVERAGE,
    PROBABILITY_ORIENTATION,
    validate_prediction_frame,
)
from credit_risk_fs.experiments.rank_voting import (
    _canonical_first_fold,
    aggregate_cross_dataset_rank_voting,
    canonical_fold_projection,
    fit_rfe_memory_safe,
    fit_voters_sequentially_memory_safe,
)
from credit_risk_fs.preprocessing.encoding import OriginalFeatureNumericEncoder
from credit_risk_fs.selectors.lightweight.mi_mrmr import (
    MutualInformationMRMRSelector,
)


ROOT = Path(__file__).resolve().parents[1]
REFINEMENT = ROOT / "configs/execution/lendingclub_memory_safe_refinement_v1.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_canonical_projection_preserves_exact_row_order_without_feature_owner():
    X = pd.DataFrame({"a": np.arange(60), "b": np.arange(60) * 2})
    y = pd.Series(([0, 1] * 30), dtype="int8")
    ids = pd.Series([f"id-{value:03d}" for value in reversed(range(60))])
    times = pd.Series(np.repeat(np.arange(12), 5))
    old = _canonical_first_fold(X=X, y=y, stable_row_ids=ids, time_values=times)
    refined = canonical_fold_projection(
        y=y, stable_row_ids=ids, time_values=times, fold_id=1
    )
    pd.testing.assert_frame_equal(
        old["X"], X.iloc[refined["source_positions"]].reset_index(drop=True)
    )
    pd.testing.assert_series_equal(old["y"], refined["y"])
    pd.testing.assert_series_equal(old["ids"], refined["ids"])
    np.testing.assert_array_equal(
        old["training_indices"], refined["training_indices"]
    )
    np.testing.assert_array_equal(
        old["validation_indices"], refined["validation_indices"]
    )
    assert "X" not in refined
    assert set(refined["ids"].iloc[refined["training_indices"]]).isdisjoint(
        set(refined["ids"].iloc[refined["validation_indices"]])
    )


def test_contiguous_encoder_is_value_and_dtype_identical_to_prior_expression():
    source = pd.DataFrame(
        {
            "numeric": pd.Series([1.0, np.nan, np.inf, -2.0], dtype="float64"),
            "integer": pd.Series([1, 2, 3, 4], dtype="int64"),
            "category": pd.Series(["b", None, "A", "b"], dtype="string"),
        }
    )
    encoder = OriginalFeatureNumericEncoder().fit(source)
    expected = {}
    for name in encoder.feature_names_ or []:
        if name in encoder.numeric_columns_:
            expected[name] = (
                pd.to_numeric(source[name], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(encoder.fill_values_[name])
                .astype("float32")
            )
        else:
            values = source[name].astype("string").fillna("<MISSING>").astype(str)
            expected[name] = (
                values.map(encoder.category_maps_[name]).fillna(-1).astype("float32")
            )
    expected_frame = pd.DataFrame(expected, index=source.index, columns=source.columns)
    observed = encoder.transform(source)
    pd.testing.assert_frame_equal(observed, expected_frame, check_exact=True)
    assert set(observed.dtypes.astype(str)) == {"float32"}
    assert observed.to_numpy(copy=False).flags.c_contiguous


def test_destructive_encoder_is_exact_and_releases_every_source_column():
    original = pd.DataFrame(
        {
            "numeric": pd.array([1.0, np.nan, np.inf, -2.0], dtype="float64"),
            "nullable_integer": pd.array([1, None, 3, 4], dtype="Int64"),
            "boolean_as_loaded": pd.array([1.0, 0.0, None, 1.0], dtype="Float32"),
            "category": pd.array(["b", None, "A", "b"], dtype="string"),
        },
        index=pd.Index([10, 20, 30, 40], name="row"),
    )
    encoder = OriginalFeatureNumericEncoder().fit(original)
    expected = encoder.transform(original)
    source = original.copy(deep=True)

    observed = encoder.transform_releasing_source(source)

    pd.testing.assert_frame_equal(observed, expected, check_exact=True)
    assert source.empty
    assert list(source.columns) == []
    assert observed.to_numpy(copy=False).flags.c_contiguous
    assert set(observed.dtypes.astype(str)) == {"float32"}


def test_disk_backed_split_encoder_is_exact_and_preserves_mrmr_identity(
    tmp_path: Path,
):
    original = pd.DataFrame(
        {
            "numeric_a": [1.0, np.nan, np.inf, -2.0, 8.0, 3.0, 1.5, -4.0],
            "category": pd.array(
                ["b", None, "A", "b", "c", "A", "z", None], dtype="string"
            ),
            "nullable_integer": pd.array(
                [1, None, 3, 4, None, 2, None, -1], dtype="Int64"
            ),
            "numeric_b": [0.5, 0.1, np.nan, 4.0, 2.0, -1.0, 0.0, 3.5],
            "category_case": pd.array(
                ["x", "X", "y", None, "z", "x", "Y", "z"], dtype="string"
            ),
        },
        index=pd.Index([80, 10, 70, 20, 60, 30, 50, 40], name="frozen_row"),
    )
    dense_encoder = OriginalFeatureNumericEncoder().fit(original)
    dense = dense_encoder.transform(original)
    source = original.copy(deep=True)
    disk_encoder = OriginalFeatureNumericEncoder().fit(source)
    observed_splits: list[tuple[int, int, list[str]]] = []
    disk, mapping = disk_encoder.transform_releasing_source_to_memmap(
        source,
        tmp_path / "selector.npy",
        feature_split_size=2,
        before_split=lambda index, count, names: observed_splits.append(
            (index, count, names)
        ),
    )
    try:
        pd.testing.assert_frame_equal(disk, dense, check_exact=True)
        assert source.empty
        assert np.shares_memory(disk.to_numpy(copy=False), mapping)
        assert mapping.flags.f_contiguous
        assert observed_splits == [
            (1, 3, ["numeric_a", "category"]),
            (2, 3, ["nullable_integer", "numeric_b"]),
            (3, 3, ["category_case"]),
        ]

        target = pd.Series([0, 1, 0, 1, 1, 0, 1, 0], index=original.index)
        dense_selector = MutualInformationMRMRSelector(
            k=4,
            n_bins=3,
            objective="mid",
            random_state=42,
            fit_scope="equivalence_test",
        ).fit(dense, target)
        disk_selector = MutualInformationMRMRSelector(
            k=4,
            n_bins=3,
            objective="mid",
            random_state=42,
            fit_scope="equivalence_test",
        ).fit(disk, target)
        assert disk_selector.selected_features_ == dense_selector.selected_features_
        assert disk_selector.relevance_ == dense_selector.relevance_
        pd.testing.assert_frame_equal(
            disk_selector.selection_trace_,
            dense_selector.selection_trace_,
            check_exact=True,
        )
    finally:
        del disk
        mapping._mmap.close()
        gc.collect()


class _MRMR:
    def fit(self, X, y):
        self.selected_features_ = list(X.columns[:300])
        self.rf_importances_ = pd.Series(
            np.linspace(1, 0, X.shape[1]), index=X.columns
        )
        self.seen_targets = tuple(map(int, y))
        return self


class _Boruta:
    def fit(self, X, y):
        self.feature_ranking_ = list(reversed(X.columns))
        self.seen_targets = tuple(map(int, y))
        return self


def test_frozen_voters_are_never_simultaneously_resident():
    columns = [f"f{value:03d}" for value in range(675)]
    X = pd.DataFrame(np.zeros((8, 675), dtype=np.float32), columns=columns)
    y = pd.Series([0, 1] * 4, dtype="int8")
    refs: dict[str, weakref.ReferenceType] = {}
    events = []

    def observe(event, value):
        events.append(event)
        if value is not None:
            refs[event] = weakref.ref(value)
        if event == "boruta_constructed":
            assert refs["rf_corr_mrmr_constructed"]() is None

    result = fit_voters_sequentially_memory_safe(
        X_numeric=X,
        y=y,
        seed=42,
        estimator_threads=1,
        selector_factories={"rf_corr_mrmr": _MRMR, "boruta": _Boruta},
        lifetime_observer=observe,
    )
    assert events == [
        "rf_corr_mrmr_constructed",
        "rf_corr_mrmr_released",
        "boruta_constructed",
        "boruta_released",
    ]
    assert refs["boruta_constructed"]() is None
    assert len(result["rankings"]["rf_corr_mrmr"]) == 300
    assert len(result["rankings"]["boruta"]) == 675


class _RFE:
    effective_estimator_config_ = {
        "task_type": "CPU",
        "thread_count": 1,
        "rfe_step": 10,
    }

    def __init__(self, budget: int):
        self.budget = budget

    def fit(self, X, y):
        self.selected_features_ = list(X.columns[: self.budget])
        self.selection_trace_ = pd.DataFrame(
            {
                "feature": X.columns,
                "input_order": range(1, X.shape[1] + 1),
                "rfe_rank": [1] * self.budget + [2] * (X.shape[1] - self.budget),
                "selected": [True] * self.budget + [False] * (X.shape[1] - self.budget),
                "step": 10,
            }
        )
        return self


@pytest.mark.parametrize(("model", "budget"), [("lr", 20), ("catboost", 40)])
def test_branch_rfe_preserves_exact_budget_and_releases_selector(model, budget):
    columns = [f"f{value:03d}" for value in range(300)]
    X = pd.DataFrame(np.ones((6, 300), dtype=np.float32), columns=columns)
    refs = []

    def observe(event, value):
        if value is not None:
            refs.append(weakref.ref(value))

    result = fit_rfe_memory_safe(
        X_numeric=X,
        y=pd.Series([0, 1, 0, 1, 0, 1]),
        top_candidates=columns,
        model_name=model,
        seed=42,
        estimator_threads=1,
        selector_factory=lambda: _RFE(budget),
        lifetime_observer=observe,
    )
    assert result["selected_features"] == columns[:budget]
    assert len(result["rfe_trace"]) == 300
    assert refs[0]() is None


def test_encoder_retains_no_full_candidate_frame_after_last_consumer():
    source = pd.DataFrame(
        np.ones((10, 675), dtype=np.float32),
        columns=[f"f{value:03d}" for value in range(675)],
    )
    source_ref = weakref.ref(source)
    encoder = OriginalFeatureNumericEncoder()
    encoded = encoder.fit_transform(source)
    assert not np.shares_memory(source.to_numpy(copy=False), encoded.to_numpy(copy=False))
    del source
    gc.collect()
    assert source_ref() is None
    assert encoded.shape == (10, 675)


def test_rank_aggregation_tie_order_is_unchanged():
    features = ["B", "a", "C"]
    first = aggregate_cross_dataset_rank_voting(
        eligible_features=features,
        rankings={"rf_corr_mrmr": ["B", "a"], "boruta": ["a", "B", "C"]},
        fit_scopes={
            "rf_corr_mrmr": "dev_fold_training_only",
            "boruta": "dev_fold_training_only",
        },
    )
    second = aggregate_cross_dataset_rank_voting(
        eligible_features=features,
        rankings={"rf_corr_mrmr": ["B", "a"], "boruta": ["a", "B", "C"]},
        fit_scopes={
            "rf_corr_mrmr": "dev_fold_training_only",
            "boruta": "dev_fold_training_only",
        },
    )
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert first["feature"].tolist() == ["a", "B", "C"]


def test_full_dev_capacity_ranking_uses_registered_full_dev_scope():
    result = aggregate_cross_dataset_rank_voting(
        eligible_features=["a", "b", "c"],
        rankings={
            "rf_corr_mrmr": ["a", "b", "c"],
            "boruta": ["a", "c", "b"],
        },
        fit_scopes={
            "rf_corr_mrmr": "full_dev_only",
            "boruta": "full_dev_only",
        },
    )
    assert result["feature"].tolist() == ["a", "b", "c"]


def test_capacity_prediction_contract_is_single_fold_and_ineligible():
    frame = pd.DataFrame(
        {
            "stable_row_id": ["a", "b"],
            "target": [0, 1],
            "prediction_probability": [0.2, 0.8],
            "predicted_class": [0, 1],
            "fold_id": [5, 5],
            "split": ["DEV", "DEV"],
            "coverage_type": [CAPACITY_SINGLE_FOLD_COVERAGE] * 2,
            "research_eligible": [False, False],
            "comparison_eligible": [False, False],
            "probability_orientation": [PROBABILITY_ORIENTATION] * 2,
        }
    )
    result = validate_prediction_frame(
        frame,
        expected_identities=["a", "b"],
        expected_targets=[0, 1],
        coverage_type=CAPACITY_SINGLE_FOLD_COVERAGE,
        expected_split="DEV",
        research_eligible=False,
        comparison_eligible=False,
    )
    assert result["fold_ids"] == ["5"]
    with pytest.raises(ValueError):
        validate_prediction_frame(
            frame.assign(research_eligible=True),
            expected_identities=["a", "b"],
            expected_targets=[0, 1],
            coverage_type=CAPACITY_SINGLE_FOLD_COVERAGE,
            expected_split="DEV",
            research_eligible=True,
            comparison_eligible=False,
        )


def test_refinement_config_freezes_limits_precision_oot_and_audit_root():
    specs = expand_lendingclub_memory_capacity_scenarios(REFINEMENT)
    assert [item.candidate_pool for item in specs] == [200, 300, 300]
    assert [item.fold_id for item in specs] == [1, 5, None]
    assert all(not item.load_oot for item in specs)
    payload = yaml.safe_load(REFINEMENT.read_text(encoding="utf-8"))
    assert payload["numeric_semantics"]["selector_effective_dtype"] == "float32"
    assert payload["publication"]["capacity_results_root"].startswith("cleanup/audits/")
    assert payload["publication"]["canonical_results_registration_allowed"] is False
    assert payload["publication"]["implicit_all_column_requests_allowed"] is False


def test_capacity_checkpoint_hash_changes_for_any_scientific_scenario_drift():
    payload = yaml.safe_load(REFINEMENT.read_text(encoding="utf-8"))
    base = {"scenario": payload["scenarios"][1], "feature_budgets": {"lr": 20, "catboost": 40}}
    for field, value in (
        ("candidate_pool", 200),
        ("seed", 43),
        ("fold_id", 4),
        ("branches", ["lr"]),
    ):
        changed = json.loads(json.dumps(base))
        changed["scenario"][field] = value
        assert compute_config_hash(base) != compute_config_hash(changed)


def test_all_prompt4_pilot_artifacts_remain_byte_immutable():
    manifest_path = (
        ROOT / "cleanup/audits/cross_dataset_voting_integration_pilot/pilot_manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["registered_run_count"] == 4
    for run in payload["runs"]:
        for artifact in run["artifacts"].values():
            if artifact["present"]:
                path = ROOT / artifact["path"]
                assert path.stat().st_size == artifact["size_bytes"]
                assert _sha256(path) == artifact["sha256"]
