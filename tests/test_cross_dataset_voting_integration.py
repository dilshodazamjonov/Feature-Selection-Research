from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.evaluation.paired_inference import validate_paired_comparison_contract
from credit_risk_fs.experiments.compare import build_cross_dataset_voting_comparison_plan
from credit_risk_fs.experiments.matrix import (
    FROZEN_PILOT_IDS,
    cross_dataset_matrix_expansion_summary,
    expand_cross_dataset_voting_matrix,
    expand_cross_dataset_voting_pilot,
)
from credit_risk_fs.experiments.prediction_contract import (
    COMPLETE_OOF_COVERAGE,
    COMPLETE_OOT_COVERAGE,
    PILOT_COVERAGE,
    PROBABILITY_ORIENTATION,
    validate_prediction_frame,
)
from credit_risk_fs.experiments.rank_voting import (
    _canonical_first_fold,
    _resolve_effective_lr_penalty,
    build_long_voter_ranking_frame,
    fit_fold_local_voting_adapter,
)
from credit_risk_fs.preprocessing.encoding import OriginalFeatureNumericEncoder
from credit_risk_fs.pipelines.common import (
    _canonical_lendingclub_loan_ids,
    _homecredit_dev_identity_set,
    _homecredit_source_projection,
)
from credit_risk_fs.selectors.registry import get_selector
from credit_risk_fs.selectors.rfe import RFESelector


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "configs/experiments/cross_dataset_rank_voting_matrix_v1.yaml"
PILOT = ROOT / "configs/experiments/cross_dataset_rank_voting_pilot_v1.yaml"


def test_lendingclub_loan_ids_restore_frozen_decimal_string_semantics():
    observed = _canonical_lendingclub_loan_ids(pd.Series([123, "00456", 789.0]))
    assert observed.tolist() == ["123", "456", "789"]
    assert str(observed.dtype) == "string"
    with pytest.raises(ValueError):
        _canonical_lendingclub_loan_ids(pd.Series([1.5]))


class _FakeMRMR:
    def __init__(self, calls: list[tuple[str, list[int], list[int]]]):
        self.calls = calls

    def fit(self, X, y):
        self.calls.append(("mrmr", list(X.index), list(map(int, y))))
        ordered = list(X.columns)
        if int(pd.Series(y).sum()) % 2:
            ordered = list(reversed(ordered))
        self.selected_features_ = ordered[:300]
        self.rf_importances_ = pd.Series(
            np.linspace(1.0, 0.0, len(X.columns)), index=X.columns
        )
        return self


class _FakeBoruta:
    def __init__(self, calls: list[tuple[str, list[int], list[int]]]):
        self.calls = calls

    def fit(self, X, y):
        self.calls.append(("boruta", list(X.index), list(map(int, y))))
        ordered = list(X.columns)
        if int(pd.Series(y).sum()) % 2:
            ordered = list(reversed(ordered))
        self.feature_ranking_ = ordered
        return self


class _FakeRFE:
    effective_estimator_config_ = {"task_type": "CPU", "thread_count": 1}

    def __init__(self, calls: list[tuple[str, list[int], list[int]]], budget: int = 20):
        self.calls = calls
        self.budget = budget

    def fit(self, X, y):
        self.calls.append(("rfe", list(X.index), list(map(int, y))))
        ordered = list(X.columns)
        if int(pd.Series(y).sum()) % 2:
            ordered = list(reversed(ordered))
        self.selected_features_ = ordered[: self.budget]
        selected = set(self.selected_features_)
        self.selection_trace_ = pd.DataFrame(
            {
                "feature": list(X.columns),
                "input_order": range(1, X.shape[1] + 1),
                "rfe_rank": [1 if value in selected else 2 for value in X.columns],
                "selected": [value in selected for value in X.columns],
                "step": 10,
            }
        )
        return self


def _adapter_inputs():
    features = [f"feature_{index:03d}" for index in range(320)]
    X = pd.DataFrame(
        np.arange(60 * len(features), dtype=np.float32).reshape(60, len(features)),
        columns=features,
    )
    y = pd.Series(([0, 1] * 30), dtype="int8")
    ids = pd.Series([f"id-{index:03d}" for index in range(60)])
    times = pd.Series(np.arange(60))
    return X, y, ids, times


def _run_fake_adapter(y: pd.Series):
    X, _, ids, times = _adapter_inputs()
    calls: list[tuple[str, list[int], list[int]]] = []
    result = fit_fold_local_voting_adapter(
        X=X,
        y=y,
        stable_row_ids=ids,
        time_values=times,
        dataset="fixture",
        model_name="lr",
        protocol_sha256="a" * 64,
        input_artifact_hash="b" * 64,
        estimator_threads=1,
        selector_factories={
            "rf_corr_mrmr": lambda: _FakeMRMR(calls),
            "boruta": lambda: _FakeBoruta(calls),
            "rfe": lambda: _FakeRFE(calls),
        },
    )
    return result, calls


def test_fold_adapter_never_passes_held_out_rows_or_targets_to_supervised_stages():
    X, y, ids, times = _adapter_inputs()
    preview = _canonical_first_fold(X=X, y=y, stable_row_ids=ids, time_values=times)
    result, calls = _run_fake_adapter(y)
    expected_indices = set(map(int, preview["training_indices"]))
    assert calls
    assert all(set(indices) == expected_indices for _, indices, _ in calls)
    assert all(len(targets) == len(expected_indices) for _, _, targets in calls)
    assert set(result["training_ids"]).isdisjoint(set(result["validation_ids"]))


def test_held_out_target_change_cannot_change_rankings_or_selected_features():
    X, y, ids, times = _adapter_inputs()
    preview = _canonical_first_fold(X=X, y=y, stable_row_ids=ids, time_values=times)
    changed = y.copy()
    changed.iloc[preview["validation_indices"]] = 1 - changed.iloc[preview["validation_indices"]]
    first, _ = _run_fake_adapter(y)
    second, _ = _run_fake_adapter(changed)
    pd.testing.assert_frame_equal(first["voter_rankings"], second["voter_rankings"])
    assert first["candidate_features"] == second["candidate_features"]
    assert first["selected_features"] == second["selected_features"]


def test_training_target_change_can_change_supervised_rankings():
    X, y, ids, times = _adapter_inputs()
    preview = _canonical_first_fold(X=X, y=y, stable_row_ids=ids, time_values=times)
    changed = y.copy()
    first_training = int(preview["training_indices"][0])
    changed.iloc[first_training] = 1 - changed.iloc[first_training]
    first, _ = _run_fake_adapter(y)
    second, _ = _run_fake_adapter(changed)
    assert not first["voter_rankings"].equals(second["voter_rankings"])


def test_long_voter_schema_preserves_original_names_and_rejects_collisions():
    frame = build_long_voter_ranking_frame(
        dataset="fixture",
        fold_id=1,
        eligible_features=["Feature A", "Feature_B"],
        rankings={"rf_corr_mrmr": ["Feature A"], "boruta": ["Feature_B", "Feature A"]},
        raw_scores={"rf_corr_mrmr": {"Feature A": 0.5}, "boruta": {}},
        selector_configurations={"rf_corr_mrmr": {}, "boruta": {}},
        seed=42,
        training_row_identity_sha256="a" * 64,
        training_identity_target_sha256="b" * 64,
        input_artifact_hash="c" * 64,
        protocol_sha256="d" * 64,
    )
    assert len(frame) == 4
    assert set(frame["original_feature_name"]) == {"Feature A", "Feature_B"}
    missing = frame[(frame.voter_id == "rf_corr_mrmr") & (frame.original_feature_name == "Feature_B")]
    assert missing.iloc[0]["normalized_score"] == 0
    with pytest.raises(ValueError, match="collision"):
        build_long_voter_ranking_frame(
            dataset="fixture",
            fold_id=1,
            eligible_features=["A", "a"],
            rankings={"rf_corr_mrmr": ["A"], "boruta": ["A"]},
            raw_scores={"rf_corr_mrmr": {}, "boruta": {}},
            selector_configurations={"rf_corr_mrmr": {}, "boruta": {}},
            seed=42,
            training_row_identity_sha256="a",
            training_identity_target_sha256="b",
            input_artifact_hash="c",
            protocol_sha256="d",
        )


def test_original_feature_encoder_is_one_to_one_training_only_and_order_strict():
    training = pd.DataFrame({"numeric": [1.0, np.nan], "category": ["b", "a"]})
    encoded = OriginalFeatureNumericEncoder().fit_transform(training)
    assert list(encoded.columns) == ["numeric", "category"]
    assert encoded.shape == training.shape
    encoder = OriginalFeatureNumericEncoder().fit(training)
    with pytest.raises(ValueError, match="order mismatch"):
        encoder.transform(training[["category", "numeric"]])


def test_homecredit_frozen_source_projection_uses_exact_seven_table_stems():
    projection = _homecredit_source_projection(ROOT)
    assert list(projection) == [
        "application_train",
        "bureau",
        "bureau_balance",
        "credit_card_balance",
        "installments_payments",
        "POS_CASH_balance",
        "previous_application",
    ]
    assert [len(value) for value in projection.values()] == [122, 17, 3, 23, 8, 8, 37]


def test_homecredit_dev_identity_derivation_excludes_non_application_train_cohort():
    recent = pd.Series([-500, -500, -200, -700], index=[1, 2, 3, 4])
    assert _homecredit_dev_identity_set(recent, [1, 3, 4]) == frozenset({1})


def test_rfe_registry_exact_budget_trace_and_cpu_thread_contract():
    selector_cls, kwargs = get_selector("rfe")
    assert selector_cls is RFESelector
    assert kwargs["step"] == 10
    selector = RFESelector(n_features=2, thread_count=4).fit(
        pd.DataFrame({"a": [0, 1], "b": [1, 0]}), pd.Series([0, 1])
    )
    assert selector.selected_features_ == ["a", "b"]
    assert selector.selection_trace_["selected"].all()
    assert selector.effective_estimator_config_["task_type"] == "CPU"
    with pytest.raises(ValueError, match="exceeds"):
        RFESelector(n_features=3).fit(
            pd.DataFrame({"a": [0, 1], "b": [1, 0]}), pd.Series([0, 1])
        )
    with pytest.raises(ValueError, match="positive"):
        RFESelector(thread_count=0)


def test_effective_lr_penalty_accepts_only_l2_or_sklearn_l2_bridge():
    assert _resolve_effective_lr_penalty({"penalty": "l2"}) == "l2"
    assert "l2" in _resolve_effective_lr_penalty(
        {"penalty": "deprecated", "l1_ratio": 0.0}
    )
    with pytest.raises(ValueError):
        _resolve_effective_lr_penalty({"penalty": "deprecated", "l1_ratio": 1.0})


def test_matrix_dry_expansion_is_exact_and_has_no_filesystem_side_effect(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = list(tmp_path.iterdir())
    specs = expand_cross_dataset_voting_matrix(MATRIX)
    summary = cross_dataset_matrix_expansion_summary(specs)
    plan = build_cross_dataset_voting_comparison_plan(specs)
    assert summary["total_registered_runs"] == 16
    assert summary["total_dev_fold_executions"] == 80
    assert summary["final_full_dev_oot_fits"] == 16
    assert len(plan) == 12
    assert sum(row["comparison_type"] == "primary" for row in plan) == 4
    assert sum(row["comparison_type"] == "sensitivity" for row in plan) == 8
    assert list(tmp_path.iterdir()) == before


def test_pilot_expansion_is_exact_sequential_and_inference_ineligible():
    specs = expand_cross_dataset_voting_pilot(PILOT)
    assert tuple(item.run_id for item in specs) == FROZEN_PILOT_IDS
    assert [item.execution_order for item in specs] == [1, 2, 3, 4]
    payload = json.loads(json.dumps(__import__("yaml").safe_load(PILOT.read_text())))
    assert payload["research_eligible"] is False
    assert payload["shared"]["concurrent_experiment_runs"] == 1
    assert payload["shared"]["concurrent_folds"] == 1


def _prediction_frame(folds):
    count = len(folds)
    return pd.DataFrame(
        {
            "stable_row_id": [f"id-{index}" for index in range(count)],
            "target": [index % 2 for index in range(count)],
            "prediction_probability": np.linspace(0.1, 0.9, count),
            "predicted_class": [index % 2 for index in range(count)],
            "fold_id": folds,
            "split": "DEV",
            "coverage_type": COMPLETE_OOF_COVERAGE,
            "research_eligible": True,
            "comparison_eligible": True,
            "probability_orientation": PROBABILITY_ORIENTATION,
        }
    )


def test_synthetic_complete_oof_and_oot_prediction_hash_contracts():
    oof = _prediction_frame([1, 2, 3, 4, 5])
    oof_meta = validate_prediction_frame(
        oof,
        expected_identities=oof.stable_row_id,
        expected_targets=oof.target,
        coverage_type=COMPLETE_OOF_COVERAGE,
        expected_split="DEV",
        research_eligible=True,
        comparison_eligible=True,
    )
    assert oof_meta["fold_ids"] == ["1", "2", "3", "4", "5"]
    oot = oof.assign(
        split="OOT",
        fold_id="final",
        coverage_type=COMPLETE_OOT_COVERAGE,
    )
    oot_meta = validate_prediction_frame(
        oot,
        expected_identities=oot.stable_row_id,
        expected_targets=oot.target,
        coverage_type=COMPLETE_OOT_COVERAGE,
        expected_split="OOT",
        research_eligible=True,
        comparison_eligible=True,
    )
    assert oot_meta["row_count"] == 5


def test_single_fold_pilot_cannot_pass_as_complete_oof():
    pilot = _prediction_frame([1, 1, 1, 1, 1]).assign(
        coverage_type=PILOT_COVERAGE,
        research_eligible=False,
        comparison_eligible=False,
    )
    validate_prediction_frame(
        pilot,
        expected_identities=pilot.stable_row_id,
        expected_targets=pilot.target,
        coverage_type=PILOT_COVERAGE,
        expected_split="DEV",
        research_eligible=False,
        comparison_eligible=False,
    )
    with pytest.raises(ValueError):
        validate_prediction_frame(
            pilot,
            expected_identities=pilot.stable_row_id,
            expected_targets=pilot.target,
            coverage_type=COMPLETE_OOF_COVERAGE,
            expected_split="DEV",
            research_eligible=True,
            comparison_eligible=True,
        )


def test_paired_contract_rejects_pilot_fold_identity_target_and_orientation_mismatch():
    frame = _prediction_frame([1, 2, 3, 4, 5])
    base = {
        "dataset": "homecredit",
        "model": "lr",
        "split": "DEV_OOF",
        "fold_definition": "frozen-five-fold",
        "probability_orientation": PROBABILITY_ORIENTATION,
        "research_eligible": True,
        "comparison_eligible": True,
        "identity_target_sha256": "a" * 64,
    }
    assert len(validate_paired_comparison_contract(frame, base, frame, base)) == 5
    for key, value in (
        ("fold_definition", "other"),
        ("identity_target_sha256", "b" * 64),
        ("probability_orientation", "class_0"),
    ):
        changed = {**base, key: value}
        with pytest.raises(ValueError):
            validate_paired_comparison_contract(frame, base, frame, changed)
    with pytest.raises(ValueError, match="not eligible"):
        validate_paired_comparison_contract(
            frame, base, frame, {**base, "research_eligible": False}
        )
