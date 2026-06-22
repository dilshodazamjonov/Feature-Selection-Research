from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from scripts import build_clip_final_analysis as analysis


def test_prediction_metrics_enforce_gini_identity():
    frame = pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1],
            "y_pred_proba": [0.1, 0.4, 0.35, 0.9],
        }
    )

    metrics = analysis.prediction_metrics(frame)

    assert metrics["oot_gini"] == pytest.approx(2 * metrics["oot_auc"] - 1)


def test_final_analysis_outputs_have_required_shape_after_execute():
    output = Path("results/clip/final_analysis")
    assert output.exists()

    master = pd.read_csv(output / "master_results_table.csv")
    recomputed = pd.read_csv(output / "metric_recomputation.csv")
    plots = pd.read_csv(output / "plot_manifest.csv")
    summary = (output / "analysis_summary.json").read_text(encoding="utf-8")

    assert len(master[master["result_origin"].eq("clip_extension")]) == 8
    assert not master.duplicated(["dataset", "model", "selector"]).any()
    assert recomputed["status"].eq("pass").all()
    assert len(plots[plots["main_or_supplementary"].eq("main")]) == 5
    assert "limited DEV statistical view" in summary


def test_final_reports_avoid_forbidden_context_words():
    report = Path("reports/final_clip_credit_risk_report.md").read_text(encoding="utf-8").lower()
    verdict = Path("reports/final_clip_scientific_verdict.md").read_text(encoding="utf-8").lower()

    for forbidden in ["professor", "instructor", "assignment", "production-ready", "revolutionary"]:
        assert forbidden not in report
        assert forbidden not in verdict


def test_analysis_builder_does_not_import_pipeline_execution_modules():
    tree = ast.parse(Path("scripts/build_clip_final_analysis.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = {
        "credit_risk_fs.pipelines.common",
        "credit_risk_fs.clip.trainer",
        "scripts.run_clip_final_evaluation",
        "scripts.train_clip_encoder",
    }
    assert imported.isdisjoint(forbidden)


def test_train_dry_run_and_status_paths_are_read_only_in_source():
    train_source = Path("scripts/train_clip_encoder.py").read_text(encoding="utf-8")
    dry_run_source = train_source.split("def _dry_run", 1)[1].split("\ndef _write_full_outputs", 1)[0]
    assert "write_json" not in dry_run_source
    assert "mkdir" not in dry_run_source

    eval_source = Path("scripts/run_clip_final_evaluation.py").read_text(encoding="utf-8")
    assert "write_artifacts: bool = True" in eval_source
    assert "write_artifacts=False" in eval_source
