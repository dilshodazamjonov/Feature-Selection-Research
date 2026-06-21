from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import scripts.run_clip_final_evaluation as runner
from credit_risk_fs.evaluation.metrics import evaluate_model


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_complete_synthetic_run(root: Path, run_id: str = "homecredit_lr_clip") -> None:
    run_dir = root / "runs" / run_id
    (run_dir / "features").mkdir(parents=True)
    (run_dir / "results").mkdir()
    (run_dir / "models").mkdir()
    y_true = pd.Series([0, 1, 0, 1])
    y_score = pd.Series([0.1, 0.9, 0.3, 0.8])
    y_pred = pd.Series([0, 1, 0, 1])
    metrics = evaluate_model(y_true, y_score, y_pred=y_pred)
    metrics["lift_at_10"] = runner._lift_at_fraction(y_true, y_score)
    _write_json(run_dir / "config_snapshot.json", {"config_hash": "cfg"})
    _write_json(run_dir / "data_split_manifest.json", {"oot": {"row_count": 4}})
    _write_json(run_dir / "leakage_audit.json", {"status": "passed"})
    _write_json(run_dir / "checkpoint_anchor_binding.json", {"checkpoint_hash": "ckpt", "anchor_hash": "anchor"})
    _write_json(run_dir / "source_hashes.json", {"source": "hash"})
    _write_json(run_dir / "models" / "final_model_metadata.json", {"model": "lr"})
    pd.DataFrame({"feature_name": ["f1", "f2"]}).to_csv(run_dir / "features" / "final_selected_features.csv", index=False)
    pd.DataFrame({"feature_name": ["f1", "f2"], "feature_set_hash": ["features", "features"]}).to_csv(
        run_dir / "selected_features_enriched.csv",
        index=False,
    )
    pred = pd.DataFrame({"y_true": y_true, "y_pred_proba": y_score, "y_pred": y_pred})
    pred.to_csv(run_dir / "results" / "oot_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(run_dir / "results" / "oot_test_results.csv", index=False)
    pd.DataFrame([{"total_runtime_seconds": 1.0}]).to_csv(run_dir / "results" / "runtime_summary.csv", index=False)
    pd.DataFrame([{"status": "ok"}]).to_csv(run_dir / "results" / "experiment_summary.csv", index=False)
    top = pred.assign(
        dataset="homecredit",
        model="lr",
        selector="clip",
        evaluation_index=range(4),
        split="OOT",
        checkpoint_hash="ckpt",
        anchor_hash="anchor",
        feature_set_hash="features",
        run_id=run_id,
    )
    (root / "predictions").mkdir(parents=True)
    top[
        [
            "dataset",
            "model",
            "selector",
            "evaluation_index",
            "y_true",
            "y_pred_proba",
            "y_pred",
            "split",
            "checkpoint_hash",
            "anchor_hash",
            "feature_set_hash",
            "run_id",
        ]
    ].to_csv(root / "predictions" / f"{run_id}_oot_predictions.csv", index=False)


def test_complete_run_without_marker_is_recoverable_not_reusable(tmp_path, monkeypatch):
    _write_complete_synthetic_run(tmp_path)
    monkeypatch.setattr(runner, "_validate_hash_binding", lambda *args, **kwargs: (True, [], "features"))

    row = runner._classify_run(
        {"dataset": "homecredit", "model": "lr", "selector": "clip"},
        {"checkpoint_hash": "ckpt", "anchor_hash": "anchor"},
        tmp_path,
    )

    assert row["status"] == "complete_invalid"
    assert row["recovery_action"] == "write_recovered_completion_marker_after_validation"
    assert row["safe_to_reuse"] is False
    assert row["_recoverable_without_marker"] is True


def test_in_progress_directory_is_never_complete(tmp_path):
    progress = tmp_path / "runs" / "homecredit_lr_clip.in_progress"
    progress.mkdir(parents=True)
    (progress / "config_snapshot.json").write_text("{}", encoding="utf-8")

    row = runner._classify_run(
        {"dataset": "homecredit", "model": "lr", "selector": "clip"},
        {"checkpoint_hash": "ckpt", "anchor_hash": "anchor"},
        tmp_path,
    )

    assert row["status"] == "in_progress_interrupted"
    assert row["safe_to_reuse"] is False


def test_resume_without_execute_does_not_train(monkeypatch):
    args = argparse.Namespace(
        dataset=None,
        model=None,
        selector=None,
        all=False,
        dry_run=False,
        status=False,
        resume=True,
        execute=False,
    )
    monkeypatch.setattr(runner, "_parse_args", lambda: args)
    monkeypatch.setattr(runner, "_validate_boundaries", lambda datasets: {"binding": {"checkpoint_hash": "c", "anchor_hash": "a"}})
    monkeypatch.setattr(runner, "_audit_runs", lambda *args, **kwargs: ([], [], []))
    monkeypatch.setattr(runner, "_print_execution_plan", lambda specs, rows, execute: specs)

    def fail_execute(*args, **kwargs):
        raise AssertionError("resume without --execute must not train")

    monkeypatch.setattr(runner, "_execute_specs", fail_execute)

    assert runner.main() == 0
