from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.clip.score_cache import score_cache_key
from credit_risk_fs.selectors.registry import get_selector
from scripts import rebuild_clip_v2_evaluation_aggregates as aggregates
from scripts import run_clip_v2_final_evaluation as evaluation


def test_clip_v2_registry_entries_and_v1_defaults_are_separate():
    _, clip_defaults = get_selector("clip")
    _, clip_mrmr_defaults = get_selector("clip_then_mrmr")
    _, v2_defaults = get_selector("clip_v2")
    _, v2_mrmr_defaults = get_selector("clip_v2_then_mrmr")

    assert clip_defaults["config_path"] == "configs/clip/selector.yaml"
    assert clip_mrmr_defaults["config_path"] == "configs/clip/selector.yaml"
    assert v2_defaults["config_path"] == "configs/clip_v2/selector.yaml"
    assert v2_defaults["selector_label"] == "clip_v2"
    assert v2_mrmr_defaults["selector_label"] == "clip_v2_then_mrmr"
    assert v2_defaults["missing_feature_policy"] == "error"


def test_v1_v2_score_cache_keys_cannot_collide():
    common = {
        "dataset": "homecredit",
        "feature_name": "A",
        "checkpoint_hash": "checkpoint",
        "anchor_hash": "anchor",
        "text_embedding_hash": "text",
        "statistical_vector_hash": "stat",
        "preprocessor_hash": "prep",
        "fusion_rule": "fusion",
        "statistical_view_scope": "scope",
        "code_version": "code",
    }

    assert score_cache_key(experiment_version="clip_v1", **common) != score_cache_key(experiment_version="clip_v2", **common)


def test_v2_status_and_plan_are_read_only_without_checkpoint(capsys):
    args = SimpleNamespace(
        config="configs/clip_v2/evaluation.yaml",
        dataset="homecredit",
        model="lr",
        selector="clip_v2",
        plan=True,
        status=False,
        resume=False,
        execute=False,
    )

    assert evaluation.main.__module__ == "scripts.run_clip_v2_final_evaluation"
    specs = evaluation._specs(args)

    assert specs == [{"dataset": "homecredit", "model": "lr", "selector": "clip_v2"}]


def test_v2_execution_requires_explicit_execute():
    args = SimpleNamespace(dataset="homecredit", model="lr", selector="clip_v2")
    specs = evaluation._specs(args)
    assert len(specs) == 1
    assert specs[0]["selector"] == "clip_v2"


def test_one_run_execution_uses_in_progress_and_writes_completion_last(tmp_path, monkeypatch):
    output_root = tmp_path / "final_evaluation"
    spec = {"dataset": "homecredit", "model": "lr", "selector": "clip_v2"}
    binding = {"checkpoint_hash": "checkpoint", "anchor_hash": "anchor"}

    monkeypatch.setattr(evaluation, "_experiment_config", lambda spec, run_dir: SimpleNamespace(feature_budget=20))
    monkeypatch.setattr(
        evaluation,
        "prepare_modeling_data",
        lambda config: SimpleNamespace(
            X_train=pd.DataFrame(np.ones((4, 3)), columns=["a", "b", "c"]),
            y_train=pd.Series([0, 1, 0, 1]),
            X_oot=pd.DataFrame(np.ones((3, 3)), columns=["a", "b", "c"]),
            y_oot=pd.Series([0, 1, 0]),
            time_col="time",
        ),
    )

    def fake_run_experiment(config, prepared_data):
        run_dir = output_root / "runs" / "homecredit_lr_clip_v2.in_progress"
        (run_dir / "features").mkdir(parents=True, exist_ok=True)
        (run_dir / "results").mkdir(parents=True, exist_ok=True)
        (run_dir / "llm_responses" / "final_dev").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"feature_name": ["a", "b"]}).to_csv(run_dir / "features" / "final_selected_features.csv", index=False)
        pd.DataFrame({"feature_name": ["a", "b"], "final_selected": [True, True]}).to_csv(
            run_dir / "llm_responses" / "final_dev" / "clip_v2_selection_manifest.csv",
            index=False,
        )
        pd.DataFrame(
            {
                "auc": [1.0],
                "gini": [1.0],
                "ks": [1.0],
                "lift_at_10": [1.0],
                "model_score_psi": [0.01],
            }
        ).to_csv(run_dir / "results" / "oot_test_results.csv", index=False)
        pd.DataFrame({"total_runtime_seconds": [1.0]}).to_csv(run_dir / "results" / "runtime_summary.csv", index=False)
        pd.DataFrame({"model_score_psi": [0.01]}).to_csv(run_dir / "results" / "model_score_psi.csv", index=False)
        pd.DataFrame({"y_true": [0, 1, 0], "y_pred_proba": [0.1, 0.9, 0.2], "y_pred": [0, 1, 0]}).to_csv(
            run_dir / "results" / "oot_predictions.csv",
            index=False,
        )
        (run_dir / "leakage_report.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(exp_dir=run_dir)

    monkeypatch.setattr(evaluation, "run_experiment", fake_run_experiment)

    assert evaluation._execute_planned([spec], output_root=output_root, binding=binding) == 0
    final_dir = output_root / "runs" / "homecredit_lr_clip_v2"
    assert final_dir.exists()
    assert not (output_root / "runs" / "homecredit_lr_clip_v2.in_progress").exists()
    assert (final_dir / "RUN_COMPLETE.json").exists()
    assert (output_root / "predictions" / "homecredit_lr_clip_v2.parquet").exists()


def test_valid_completed_runs_are_skipped(tmp_path):
    root = tmp_path / "final_evaluation"
    spec = {"dataset": "homecredit", "model": "lr", "selector": "clip_v2"}
    run_id = "homecredit_lr_clip_v2"
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (root / "predictions").mkdir(parents=True)
    pd.DataFrame(
        {
            "dataset": ["homecredit"],
            "model": ["lr"],
            "selector": ["clip_v2"],
            "evaluation_index": [0],
            "y_true": [0],
            "y_pred_proba": [0.2],
            "y_pred": [0],
            "split": ["OOT"],
            "run_id": [run_id],
            "checkpoint_hash": ["checkpoint"],
            "anchor_hash": ["anchor"],
            "feature_set_hash": ["features"],
        }
    ).to_parquet(root / "predictions" / f"{run_id}.parquet", index=False)
    for name in ["config_snapshot.yaml", "execution.log"]:
        path = run_dir / name
        path.write_text("x\n", encoding="utf-8")
    pd.DataFrame({"feature_name": ["a"], "feature_set_hash": ["features"]}).to_csv(run_dir / "selected_features.csv", index=False)
    (run_dir / "feature_selection_manifest.json").write_text('{"feature_set_hash": "features"}', encoding="utf-8")
    (run_dir / "metrics.json").write_text('{"auc": 0.5}', encoding="utf-8")
    (run_dir / "runtime.json").write_text('{"total_runtime_seconds": 1.0}', encoding="utf-8")
    pd.DataFrame({"model_score_psi": [0.0]}).to_csv(run_dir / "model_score_psi.csv", index=False)
    (run_dir / "leakage_audit.json").write_text("{}", encoding="utf-8")
    (run_dir / "RUN_COMPLETE.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")

    status = evaluation._classify_run(root, spec, {"checkpoint_hash": "checkpoint", "anchor_hash": "anchor"})
    planned = evaluation._planned_runs([spec], [status], resume=False)

    assert status["status"] == "complete_valid"
    assert planned == []


def test_aggregate_builder_scans_completed_runs_without_fitting_models(tmp_path):
    root = tmp_path / "final_evaluation"
    scan = aggregates.scan_completed_runs(root)

    assert scan["status"] == "incomplete"
    assert scan["expected_run_count"] == 8
    assert scan["completed_run_count"] == 0
