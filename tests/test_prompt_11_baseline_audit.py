from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import json

from credit_risk_fs.analysis.baseline_audit import (
    BaselineAuditError,
    aggregate_dev_folds,
    aggregate_random_k_replicates,
    paired_stratified_auc_bootstrap_many,
    recompute_prediction_metrics,
    recompute_selection_stability,
    summarize_oot_availability,
    validate_prediction_frame,
)


def _prediction() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "stable_row_id": ["a", "b", "c", "d"],
            "target": [0, 1, 0, 1],
            "prediction_probability": [0.1, 0.9, 0.2, 0.8],
            "dataset": "synthetic",
            "model": "lr",
            "method": "iv_woe",
            "split": "oot",
            "run_id": "run",
        }
    )


def test_prediction_integrity_and_metric_recomputation() -> None:
    frame = validate_prediction_frame(_prediction(), expected_split="oot")
    metrics = recompute_prediction_metrics(frame)
    assert metrics["auc"] == 1.0
    assert metrics["gini"] == 1.0
    for mutation, message in (
        (lambda x: x.assign(stable_row_id=["a", "a", "c", "d"]), "duplicated"),
        (lambda x: x.assign(prediction_probability=[0.1, np.nan, 0.2, 0.8]), "finite"),
        (lambda x: x.assign(prediction_probability=[0.1, 1.1, 0.2, 0.8]), r"\[0, 1\]"),
        (lambda x: x.assign(target=[0, 2, 0, 1]), "binary"),
    ):
        with pytest.raises(BaselineAuditError, match=message):
            validate_prediction_frame(mutation(_prediction()))


def test_paired_stratified_bootstrap_is_deterministic_and_paired() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = {"a": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9], "b": [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]}
    left = paired_stratified_auc_bootstrap_many(y, scores, repetitions=25, seed=20260721, chunk_size=7)
    right = paired_stratified_auc_bootstrap_many(y, scores, repetitions=25, seed=20260721, chunk_size=4)
    assert np.array_equal(left["a"], right["a"])
    assert np.array_equal(left["b"], right["b"])
    with pytest.raises(BaselineAuditError, match="two-class"):
        paired_stratified_auc_bootstrap_many([0, 0], {"a": [0.1, 0.2]}, repetitions=2)


def test_dev_aggregation_preserves_folds_and_does_not_pool() -> None:
    frame = pd.DataFrame(
        [
            {
                "run_id": "r", "dataset": "d", "model": "lr", "method_id": "m", "fold": fold,
                "auc": fold / 10, "gini": fold / 5 - 1, "ks": fold / 20,
                "lift_at_10": 1 + fold / 10, "selected_features": 20,
            }
            for fold in range(1, 6)
        ]
    )
    result = aggregate_dev_folds(frame).iloc[0]
    assert result["fold_order"] == "1|2|3|4|5"
    assert result["valid_fold_count"] == 5
    assert result["pooling_performed"] is False or not bool(result["pooling_performed"])
    assert result["auc_mean"] == pytest.approx(0.3)


def test_stability_marks_varying_natural_support_not_applicable_for_kuncheva() -> None:
    rows = []
    for fold, features in enumerate((["a"], ["a", "b"], ["a"], ["a", "c"], ["a"]), start=1):
        rows.extend({"fold_id": fold, "feature_name": feature} for feature in features)
    result = recompute_selection_stability(pd.DataFrame(rows), candidate_count=5)
    assert result["natural_support_varies"] is True
    assert result["kuncheva_applicability"] == "not_applicable_varying_subset_size"
    assert np.isnan(result["kuncheva_stability"])


def test_oot_present_and_absent_paths_are_explicit() -> None:
    assert summarize_oot_availability(pd.DataFrame({"auc": [0.7]})) == {
        "oot_available": True,
        "oot_evaluation_units": 1,
        "interpretation": "authenticated_saved_oot_evidence_available",
    }
    absent = summarize_oot_availability(None)
    assert absent["oot_available"] is False
    assert "do_not_infer_from_dev" in absent["interpretation"]


def test_random_k_replicates_are_all_aggregated_without_seed_selection() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["d", "d", "d"],
            "model": ["lr", "lr", "lr"],
            "method_id": ["random_k"] * 3,
            "seed": [3, 1, 2],
            "auc": [0.60, 0.55, 0.65],
        }
    )
    result = aggregate_random_k_replicates(frame).iloc[0]
    assert result["replicate_count"] == 3
    assert result["seeds"] == "1|2|3"
    assert result["auc_mean"] == pytest.approx(0.60)
    assert result["favorable_seed_selected"] is False or not bool(result["favorable_seed_selected"])


def test_committed_audit_tables_have_exact_inference_and_reconciliation_counts() -> None:
    root = Path(__file__).resolve().parents[1]
    audit = root / "cleanup/audits/prompt_11_selector_combinations"
    reconciliation = pd.read_csv(audit / "baseline_metric_reconciliation.csv")
    comparisons = pd.read_csv(audit / "baseline_pairwise_comparisons.csv")
    families = pd.read_csv(audit / "baseline_holm_families.csv")
    assert len(reconciliation) == 396
    assert reconciliation["status"].eq("pass").all()
    assert len(comparisons) == 56
    assert comparisons["bootstrap_attempted"].eq(2000).all()
    assert comparisons["bootstrap_valid"].eq(2000).all()
    assert len(families) == 8
    assert families["family_size"].eq(7).all()


def test_prompt_10_preservation_verification_is_byte_hash_identical() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            root
            / "cleanup/audits/prompt_11_selector_combinations/baseline_preservation_verification.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["status"] == "byte_hash_identical"
    assert payload["expected_cells"] == payload["identical_cells"] == 36
    assert payload["mismatches"] == []
    assert payload["success_marker_identical"] is True
    assert payload["raw_dataset_paths_resolved"] is False
    assert payload["prompt_10_workload_executed"] is False
