import json
from pathlib import Path

import pandas as pd

from scripts.aggregate_results import FINAL_COLUMNS, main


def test_aggregate_results_includes_new_metric_columns(tmp_path):
    run_dir = tmp_path / "results" / "lr" / "llm" / "run_a"
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "features").mkdir(parents=True)

    manifest = {
        "status": "completed",
        "run_id": "run_a",
        "model": "lr",
        "selector": "llm",
        "experiment_type": "llm",
        "feature_budget": 20,
        "llm_shared_ranking_enabled": True,
        "llm_ranking_budget": 40,
        "config_hash": "abc",
        "data_version": {"path": "data/inputs"},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    pd.DataFrame(
        [
            {
                "method": "llm",
                "model_name": "lr",
                "selector_name": "llm",
                "oot_auc": 0.7,
                "oot_gini": 0.4,
                "oot_ks": 0.3,
                "nogueira_stability": 0.8,
                "stable_feature_count_80": 12,
            }
        ]
    ).to_csv(run_dir / "results" / "experiment_summary.csv", index=False)
    pd.DataFrame({"fold": [1], "auc": [0.7], "gini": [0.4], "fold_time_sec": [1.0]}).to_csv(
        run_dir / "results" / "cv_results.csv",
        index=False,
    )

    exit_code = main([str(tmp_path / "results")])

    assert exit_code == 0
    output = pd.read_csv(tmp_path / "results" / "final_comparison_table.csv")
    assert list(output.columns) == FINAL_COLUMNS
    assert output.loc[0, "nogueira_stability"] == 0.8
    assert output.loc[0, "llm_shared_ranking_enabled"]
    assert (tmp_path / "results" / "llm_call_summary.csv").exists()
    assert (tmp_path / "results" / "failed_runs.csv").exists()


def test_aggregate_results_accepts_project_relative_matrix_paths(tmp_path):
    results_root = tmp_path / "results" / "homecredit"
    run_dir = results_root / "lr" / "statistical" / "run_a"
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "features").mkdir(parents=True)

    manifest = {
        "status": "completed",
        "run_id": "run_a",
        "model": "lr",
        "selector": "mrmr",
        "experiment_type": "statistical",
        "feature_budget": 20,
        "config_hash": "abc",
        "data_version": {"path": "data/inputs"},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    pd.DataFrame(
        [
            {
                "method": "mrmr",
                "model_name": "lr",
                "selector_name": "mrmr",
                "oot_auc": 0.71,
                "oot_gini": 0.42,
                "oot_ks": 0.31,
            }
        ]
    ).to_csv(run_dir / "results" / "experiment_summary.csv", index=False)

    relative_output = Path("results") / "homecredit" / "lr" / "statistical" / "run_a"
    pd.DataFrame(
        [
            {
                "run_id": "run_a",
                "model": "lr",
                "selector": "mrmr",
                "experiment_type": "statistical",
                "status": "completed",
                "config_hash": "abc",
                "output_folder": str(relative_output),
            }
        ]
    ).to_csv(results_root / "matrix_runs.csv", index=False)

    exit_code = main([str(results_root)])

    assert exit_code == 0
    output = pd.read_csv(results_root / "final_comparison_table.csv")
    assert len(output) == 1
    assert output.loc[0, "run_id"] == "run_a"
