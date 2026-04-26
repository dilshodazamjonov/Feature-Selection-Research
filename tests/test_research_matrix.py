import json

import pandas as pd

from experiments.config import compute_config_hash, load_project_config
from experiments.config import apply_feature_budget_to_selector_kwargs
from experiments.matrix import HYBRID_SELECTORS, MODELS, STAT_SELECTORS, iter_matrix
from experiments.tracking import is_completed_run, mark_completed
from pipelines.common import (
    ExperimentConfig,
    PreparedExperimentData,
    drop_excluded_feature_columns,
    run_experiment,
)
from training.cv_utils import GroupedTimeSeriesSplit


def test_matrix_explicitly_contains_required_runs():
    specs = list(iter_matrix())

    assert MODELS == ["lr", "catboost"]
    assert STAT_SELECTORS == ["mrmr", "boruta", "pca"]
    assert HYBRID_SELECTORS == ["mrmr", "boruta"]
    assert len(specs) == 12
    assert {spec.model for spec in specs} == {"lr", "catboost"}
    assert {spec.output_bucket for spec in specs if spec.experiment_type == "hybrid"} == {
        "hybrid_mrmr",
        "hybrid_boruta",
    }


def test_config_hash_changes_when_run_selector_changes():
    base = {"model_selector": "lr", "random_seed": 42}
    first = {**base, "matrix_run": {"selector": "llm"}}
    second = {**base, "matrix_run": {"selector": "llm_then_mrmr"}}

    assert compute_config_hash(first) != compute_config_hash(second)


def test_config_parser_reads_random_seed_and_results_dir(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "random_seed: 123\nresults_dir: custom_results\n",
        encoding="utf-8",
    )

    config = load_project_config(config_path)

    assert config["random_seed"] == 123
    assert config["results_dir"] == "custom_results"


def test_no_oot_leakage_columns_reach_model_features():
    X = pd.DataFrame(
        {
            "f1": [1.0, 2.0],
            "TARGET": [0, 1],
            "recent_decision": [-5, -4],
            "DAYS_DECISION": [-10, -9],
        }
    )

    cleaned = drop_excluded_feature_columns(
        X,
        time_col="recent_decision",
        excluded_columns=("TARGET", "DAYS_DECISION"),
    )

    assert list(cleaned.columns) == ["f1"]


def test_model_specific_selector_budgets_are_applied():
    assert apply_feature_budget_to_selector_kwargs("mrmr", {"k": 50}, 20)["k"] == 20
    assert (
        apply_feature_budget_to_selector_kwargs(
            "boruta",
            {"rfe_kwargs": {"n_features": 40}},
            20,
        )["rfe_kwargs"]["n_features"]
        == 20
    )
    assert apply_feature_budget_to_selector_kwargs("pca", {"n_components": 0.95}, 40)[
        "n_components"
    ] == 40
    assert apply_feature_budget_to_selector_kwargs("llm", {}, 20)["feature_budget"] == 20


def test_boruta_budget_sets_final_cap_without_enabling_rfe():
    kwargs = apply_feature_budget_to_selector_kwargs(
        "boruta",
        {"use_rfe": False, "n_features": 40, "rfe_kwargs": {"n_features": 40}},
        20,
    )

    assert kwargs["use_rfe"] is False
    assert kwargs["n_features"] == 20
    assert kwargs["rfe_kwargs"]["n_features"] == 20


def test_temporal_split_validation_stays_after_training():
    time_values = pd.Series([1, 1, 2, 2, 3, 3, 4, 4]).to_numpy()
    splitter = GroupedTimeSeriesSplit(n_splits=2, gap=0)

    for train_idx, val_idx in splitter.split(time_values):
        assert time_values[train_idx].max() < time_values[val_idx].min()


def test_pipeline_runs_on_tiny_dummy_dataset(tmp_path):
    X_train = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6, 7, 8],
            "f_num": [0.1, 1.1, 0.2, 1.2, 0.3, 1.3, 0.4, 1.4],
            "recent_decision": [10] * 8,
        }
    )
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    X_oot = pd.DataFrame(
        {
            "time": [9, 10],
            "f_num": [0.5, 1.5],
            "recent_decision": [10, 10],
        }
    )
    y_oot = pd.Series([0, 1])

    prepared = PreparedExperimentData(
        X_train=X_train,
        y_train=y_train,
        X_oot=X_oot,
        y_oot=y_oot,
        time_col="time",
    )
    config = ExperimentConfig(
        experiment_name="none",
        selector_name="none",
        model_name="lr",
        model_kwargs={"solver": "liblinear", "max_iter": 100, "class_weight": None, "random_state": 42},
        base_output_dir=str(tmp_path),
        experiment_output_dir=str(tmp_path / "run"),
        n_splits=2,
        cv_gap_groups=0,
        excluded_feature_columns=("recent_decision",),
        random_state=42,
    )

    run = run_experiment(config, prepared_data=prepared)

    assert (run.exp_dir / "results" / "oot_test_results.csv").exists()
    selected = pd.read_csv(run.exp_dir / "features" / "fold_selected_features.csv")
    assert {"fold_id", "selector", "feature_name", "rank", "score"}.issubset(selected.columns)
    assert (run.exp_dir / "features" / "feature_stability_metrics.csv").exists()
    assert (run.exp_dir / "results" / "runtime_summary.csv").exists()
    assert (run.exp_dir / "models" / "final_model.model").exists()
    assert (run.exp_dir / "models" / "final_preprocessor.pkl").exists()
    assert (run.exp_dir / "models" / "final_model_metadata.json").exists()
    assert (run.exp_dir / "data_split_manifest.json").exists()
    assert not (run.exp_dir / "features" / "fold_1" / "feature_statistics.csv").exists()
    assert not (run.exp_dir / "results" / "evaluation_metrics_summary.csv").exists()
    assert not (run.exp_dir / "results" / "model_score_psi.json").exists()
    report = json.loads((run.exp_dir / "leakage_report.json").read_text(encoding="utf-8"))
    assert report["oot_used_in_feature_selection"] is False
    split_manifest = json.loads((run.exp_dir / "data_split_manifest.json").read_text(encoding="utf-8"))
    assert split_manifest["time_column"] == "time"
    assert split_manifest["dev"]["row_count"] == 8
    assert split_manifest["oot"]["row_count"] == 2
    assert "dropped_older_row_count" in split_manifest
    metadata = json.loads((run.exp_dir / "models" / "final_model_metadata.json").read_text(encoding="utf-8"))
    assert metadata["training_scope"] == "full_DEV"
    assert metadata["target_column"] == "TARGET"
    assert "preprocessing_hash" in metadata


def test_completed_run_requires_lean_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    required_files = [
        "run_manifest.json",
        "leakage_report.json",
        "data_split_manifest.json",
        "features/final_selected_features.csv",
        "features/fold_selected_features.csv",
        "features/selection_frequency.csv",
        "features/feature_stability_metrics.csv",
        "models/final_model.model",
        "models/final_preprocessor.pkl",
        "models/final_model_metadata.json",
        "results/experiment_summary.csv",
        "results/cv_results.csv",
        "results/oot_test_results.csv",
        "results/oot_predictions.csv",
        "results/selected_feature_psi.csv",
        "results/model_score_psi.csv",
        "results/credit_risk_utility.csv",
        "results/runtime_summary.csv",
    ]
    for relative_path in required_files:
        path = run_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"status": "completed", "experiment_type": "statistical"}),
        encoding="utf-8",
    )
    mark_completed(run_dir)

    assert is_completed_run(run_dir)
    (run_dir / "models" / "final_model.model").unlink()
    assert not is_completed_run(run_dir)
