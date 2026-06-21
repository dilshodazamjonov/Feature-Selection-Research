from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from credit_risk_fs.clip import evaluation_aggregation as agg


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_synthetic_run(root: Path, dataset: str, model: str, selector: str, *, stale_hash: bool = False) -> None:
    run_id = agg.expected_run_id(dataset, model, selector)
    run_dir = root / "runs" / run_id
    pred_path = root / "predictions" / f"{run_id}_oot_predictions.csv"
    budget = agg.EXPECTED_BUDGETS[model]
    features = [f"{run_id}_feature_{idx}" for idx in range(budget)]
    feature_hash = agg.feature_set_hash(features)
    pred = pd.DataFrame(
        {
            "dataset": dataset,
            "model": model,
            "selector": selector,
            "evaluation_index": [0, 1, 2, 3],
            "y_true": [0, 1, 0, 1],
            "y_pred_proba": [0.1, 0.9, 0.2, 0.8],
            "y_pred": [0, 1, 0, 1],
            "split": "OOT",
            "checkpoint_hash": "checkpoint",
            "anchor_hash": "anchor",
            "feature_set_hash": feature_hash,
            "run_id": run_id,
        }
    )
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(pred_path, index=False)
    metrics = pd.DataFrame(
        [
            {
                "auc": 1.0,
                "gini": 1.0,
                "ks": 1.0,
                "lift_at_10": 2.0,
                "model_score_psi": 0.01,
                "selected_feature_count": budget,
                "total_candidate_feature_count": budget + 5,
            }
        ]
    )
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "features").mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "results" / "oot_test_results.csv"
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame([{"model_score_psi": 0.01}]).to_csv(run_dir / "results" / "model_score_psi.csv", index=False)
    pd.DataFrame([{"total_runtime_seconds": 1.0, "final_training_time_sec": 0.2, "final_evaluation_time_sec": 0.1}]).to_csv(
        run_dir / "results" / "runtime_summary.csv",
        index=False,
    )
    pd.DataFrame({"feature": features}).to_csv(run_dir / "features" / "final_selected_features.csv", index=False)
    pd.DataFrame(
        {
            "feature_name": features,
            "final_selected": True,
            "final_rank": range(1, budget + 1),
            "clip_score": 1.0,
            "clip_rank": range(1, budget + 1),
            "screening_pool_member": True,
            "semantic_group": "synthetic",
            "source_table_or_formula": "synthetic",
            "blocked_feature_count": 0,
        }
    ).to_csv(run_dir / "selected_features_enriched.csv", index=False)
    _write_json(run_dir / "config_snapshot.json", {"config_hash": "config"})
    _write_json(run_dir / "data_split_manifest.json", {"dev": {"row_count": 10, "target_rate": 0.2}, "oot": {"row_count": 4, "target_rate": 0.5}})
    _write_json(
        run_dir / "leakage_audit.json",
        {
            "target_column_excluded": True,
            "temporal_split_disjoint": True,
            "oot_used_in_feature_selection": False,
        },
    )
    _write_json(run_dir / "source_hashes.json", {"source": "hash"})
    _write_json(
        run_dir / "RUN_COMPLETE.json",
        {
            "run_id": run_id,
            "dataset": dataset,
            "model": model,
            "selector": selector,
            "completed_at": "2026-01-01T00:00:00+00:00",
            "config_hash": "config",
            "feature_set_hash": feature_hash,
            "checkpoint_hash": "checkpoint",
            "anchor_hash": "anchor",
            "source_hashes": {"source": "hash"},
            "prediction_file_hash": "stale" if stale_hash else agg.sha256_file(pred_path),
            "metrics_file_hash": agg.sha256_file(metrics_path),
            "prediction_row_count": 4,
            "completion_status": "complete_valid",
        },
    )


def _write_all_synthetic_runs(root: Path) -> None:
    for dataset, model, selector in agg.EXPECTED_RUNS:
        _write_synthetic_run(root, dataset, model, selector)


def test_aggregate_discovers_all_completed_runs_and_ignores_in_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(agg, "expected_source_hashes", lambda: {"source": "hash"})
    _write_all_synthetic_runs(tmp_path)
    (tmp_path / "runs" / "homecredit_lr_clip.in_progress").mkdir()

    runs = agg.discover_completed_runs(tmp_path)
    evaluation = agg.build_evaluation_summary(runs)

    assert len(runs) == 8
    assert len(evaluation) == 8
    assert not evaluation.astype(str).apply(lambda col: col.str.contains(".in_progress", regex=False)).any().any()


def test_stale_prediction_hash_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(agg, "expected_source_hashes", lambda: {"source": "hash"})
    _write_all_synthetic_runs(tmp_path)
    marker = tmp_path / "runs" / "homecredit_lr_clip" / "RUN_COMPLETE.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["prediction_file_hash"] = "bad"
    marker.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="prediction hash mismatch"):
        agg.discover_completed_runs(tmp_path)


def test_missing_required_run_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(agg, "expected_source_hashes", lambda: {"source": "hash"})
    _write_all_synthetic_runs(tmp_path)
    marker = tmp_path / "runs" / "homecredit_lr_clip" / "RUN_COMPLETE.json"
    marker.unlink()

    with pytest.raises(ValueError, match="completed run coverage mismatch"):
        agg.discover_completed_runs(tmp_path)


def test_selected_feature_budget_and_psi_limitation_are_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(agg, "expected_source_hashes", lambda: {"source": "hash"})
    _write_all_synthetic_runs(tmp_path)
    runs = agg.discover_completed_runs(tmp_path)
    selected = agg.build_selected_features_long(runs)
    selected_summary = agg.build_selected_feature_summary(selected)
    psi = agg.build_score_psi_summary(runs)

    assert set(selected_summary["selected_feature_count"]) == {20, 40}
    assert selected["run_id"].nunique() == 8
    assert psi["run_id"].nunique() == 8
    assert psi["psi_recomputed"].eq(False).all()
    assert psi["psi_recomputation_limitation"].str.contains("not persisted").all()


def test_atomic_write_replaces_only_aggregate_files(tmp_path, monkeypatch):
    monkeypatch.setattr(agg, "expected_source_hashes", lambda: {"source": "hash"})
    _write_all_synthetic_runs(tmp_path)
    aggregates = {
        "evaluation_summary.csv": pd.DataFrame({"run_id": ["x"]}),
        "evaluation_summary.json": [{"run_id": "x"}],
        "run_manifest.json": {"runs": [{"run_id": "x"}]},
        "comparison_with_frozen_baselines.csv": pd.DataFrame({"run_id": ["x"]}),
        "selected_features_long.csv": pd.DataFrame({"run_id": ["x"]}),
        "selected_feature_summary.csv": pd.DataFrame({"run_id": ["x"]}),
        "semantic_coverage_summary.csv": pd.DataFrame({"run_id": ["x"]}),
        "redundancy_summary.csv": pd.DataFrame({"run_id": ["x"]}),
        "runtime_summary.csv": pd.DataFrame({"run_id": ["x"]}),
        "score_psi_summary.csv": pd.DataFrame({"run_id": ["x"]}),
        "statistical_significance_summary.csv": pd.DataFrame({"run_id": ["x"]}),
    }

    agg.atomic_write_aggregates(tmp_path, aggregates)

    assert pd.read_csv(tmp_path / "evaluation_summary.csv")["run_id"].tolist() == ["x"]
    assert not (tmp_path / ".aggregate_rebuild_tmp").exists()
