from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk_fs.clip_final_comparison import io as comparison_io
from credit_risk_fs.clip_final_comparison import execution as comparison_execution
from credit_risk_fs.clip_final_comparison.ablations import train_grouped_ablation_representations
from credit_risk_fs.clip_final_comparison.execution import PreparedFrame
from credit_risk_fs.clip_final_comparison.plans import build_core_experiment_plan


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_clip_final_comparison", "scripts/run_clip_final_comparison.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _synthetic_frame() -> PreparedFrame:
    rng = np.random.default_rng(2701)
    n_dev = 72
    n_oot = 36
    X_dev = pd.DataFrame(
        {
            "income_mean": rng.normal(size=n_dev),
            "loan_ratio": rng.normal(size=n_dev),
            "fico_score": rng.normal(size=n_dev),
            "delinq_flag": np.arange(n_dev) % 2,
            "noise": rng.normal(size=n_dev),
        }
    )
    y_dev = (X_dev["income_mean"] + X_dev["fico_score"] * 0.4 + rng.normal(scale=0.2, size=n_dev) > 0).astype(int)
    X_oot = pd.DataFrame(
        {
            "income_mean": rng.normal(size=n_oot),
            "loan_ratio": rng.normal(size=n_oot),
            "fico_score": rng.normal(size=n_oot),
            "delinq_flag": np.arange(n_oot) % 2,
            "noise": rng.normal(size=n_oot),
        }
    )
    y_oot = (X_oot["income_mean"] + X_oot["fico_score"] * 0.4 + rng.normal(scale=0.25, size=n_oot) > 0).astype(int)
    return PreparedFrame(
        X_dev=X_dev,
        y_dev=y_dev,
        X_oot=X_oot,
        y_oot=y_oot,
        time_dev=pd.Series(pd.date_range("2020-01-01", periods=n_dev, freq="D")),
        time_oot=pd.Series(pd.date_range("2020-04-01", periods=n_oot, freq="D")),
    )


def _synthetic_representation_views() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = [f"feature_{idx}" for idx in range(16)]
    rng = np.random.default_rng(33)
    text = pd.DataFrame({"feature_name": features})
    for idx in range(6):
        text[f"text_{idx}"] = rng.normal(size=len(features))
    stat = pd.DataFrame({"feature_name": features})
    descriptors = [
        "missing_rate",
        "unique_ratio",
        "concentration_share",
        "signed_log_mean",
        "log_standard_deviation",
        "clipped_skewness",
        "normalized_entropy",
        "is_numeric",
        "is_categorical",
        "is_binary",
        "numeric_stats_valid",
        "skewness_valid",
        "entropy_valid",
    ]
    for idx, name in enumerate(descriptors):
        stat[name] = rng.normal(loc=idx / 10, scale=0.5, size=len(features))
    return text, stat


def _isolate(monkeypatch, tmp_path: Path):
    runner = _load_runner()
    root = tmp_path / "clip_final_comparison"
    monkeypatch.setattr(runner, "OUTPUT_ROOT", root)
    monkeypatch.setattr(runner, "LOG_PATH", root / "pipeline_execution.log")
    monkeypatch.setattr(runner, "STATE_PATH", root / "pipeline_state.json")
    monkeypatch.setattr(runner, "LOCK_PATH", root / ".pipeline.lock")
    monkeypatch.setattr(comparison_io, "OUTPUT_ROOT", root)
    monkeypatch.setattr(comparison_execution, "OUTPUT_ROOT", root)
    monkeypatch.setattr(runner, "_prepared_frame", lambda dataset: _synthetic_frame())
    monkeypatch.setattr(runner, "_model_kwargs", lambda dataset, model: {"random_state": 7})
    return runner, root


def test_core_matrix_keys_are_unique_and_counted():
    plan = build_core_experiment_plan()
    assert len(plan) == 184
    assert plan["run_id"].is_unique
    assert plan["screening_method"].eq("random").sum() == 120
    assert len(plan) - int(plan["screening_method"].eq("random").sum()) == 64


def test_synthetic_random_seed_stage_executes_all_configured_keys(monkeypatch, tmp_path):
    runner, root = _isolate(monkeypatch, tmp_path)
    plan = pd.DataFrame(
        [
            {
                "run_id": f"synthetic_lr_random_2x_seed{seed}",
                "dataset": "synthetic",
                "model": "lr",
                "screening_method": "random",
                "pool_multiplier": 2,
                "candidate_pool_size": 4,
                "final_feature_budget": 2,
                "random_seed": seed,
            }
            for seed in [101, 202, 303]
        ]
    )
    result = runner._execute_plan_rows(plan, root / "candidate_pool/runs", expected_count=3, stage_name="random_repetitions")
    assert result["completed_runs"] == 3
    assert len(runner.completed_run_dirs(root / "candidate_pool/runs")) == 3
    distribution = runner._random_distribution(runner.completed_run_dirs(root / "candidate_pool/runs"))
    assert distribution.empty


def test_synthetic_seed_and_ablation_artifacts_are_distinct(monkeypatch, tmp_path):
    runner, root = _isolate(monkeypatch, tmp_path)
    seed_plan = pd.DataFrame(
        [
            {
                "run_id": f"synthetic_lr_variance_5x_seed{seed}",
                "dataset": "synthetic",
                "model": "lr",
                "screening_method": "variance",
                "pool_multiplier": 5,
                "candidate_pool_size": 5,
                "final_feature_budget": 2,
                "checkpoint_seed": seed,
            }
            for seed in [11, 22]
        ]
    )
    runner._execute_plan_rows(seed_plan, root / "seed_robustness/runs", expected_count=2, stage_name="seed_downstream")
    hashes = [
        pd.read_csv(run / "candidate_pool.csv")["checkpoint_seed"].iloc[0]
        for run in runner.completed_run_dirs(root / "seed_robustness/runs")
    ]
    assert sorted(hashes) == [11, 22]

    text, stat = _synthetic_representation_views()
    table = train_grouped_ablation_representations(output_root=root, text_view=text, statistical_view=stat)
    assert len(table) == 7
    selected = [root / "ablations/training" / name / "selected_checkpoint.json" for name in ["without_location_scale", "without_shape_diversity", "without_type_validity"]]
    assert all(path.exists() for path in selected)


def test_synthetic_temporal_uncertainty_and_audit_guards(monkeypatch, tmp_path):
    runner, root = _isolate(monkeypatch, tmp_path)
    cutoffs = runner.execute_temporal_cutoffs()
    assert Path(cutoffs["temporal_cutoff_manifest"]).exists()
    manifest = pd.read_csv(cutoffs["temporal_cutoff_manifest"])
    assert {"eligible", "rejection_reason", "label_maturity_rule"}.issubset(manifest.columns)

    plan = pd.DataFrame(
        [
            {
                "run_id": "synthetic_lr_clip_v2_5x",
                "dataset": "synthetic",
                "model": "lr",
                "screening_method": "variance",
                "pool_multiplier": 5,
                "candidate_pool_size": 5,
                "final_feature_budget": 2,
            },
            {
                "run_id": "synthetic_lr_random_5x_seed101",
                "dataset": "synthetic",
                "model": "lr",
                "screening_method": "random",
                "pool_multiplier": 5,
                "candidate_pool_size": 5,
                "final_feature_budget": 2,
                "random_seed": 101,
            },
        ]
    )
    runner._execute_plan_rows(plan, root / "candidate_pool/runs", expected_count=2, stage_name="core_candidate_pool_runs")
    uncertainty = runner.execute_paired_uncertainty()
    assert Path(uncertainty["paired_uncertainty"]).exists()
    text, stat = _synthetic_representation_views()
    train_grouped_ablation_representations(output_root=root, text_view=text, statistical_view=stat)
    monkeypatch.setattr(runner, "_validate_seed_score_cache_manifest", lambda: None)
    monkeypatch.setattr(runner, "_validate_ablation_score_cache_manifest", lambda: None)

    monkeypatch.setattr(
        runner,
        "status_payload",
        lambda: {
            "completed_scientific_runs": 183,
            "completed_seed_runs": 20,
            "completed_ablation_runs": 28,
            "stages": {
                **{stage: {"status": "complete_valid"} for stage in runner.STAGES},
                "core_candidate_pool_runs": {"status": "complete_valid", "completed_runs": 183, "expected_runs": 184},
                "random_repetitions": {"status": "complete_valid", "completed_runs": 120, "expected_runs": 120},
                "seed_downstream": {"status": "complete_valid", "completed_runs": 20, "expected_runs": 20},
                "ablation_downstream": {"status": "complete_valid", "completed_runs": 28, "expected_runs": 28},
            },
        },
    )
    try:
        runner.final_audit()
    except RuntimeError as exc:
        assert "core_candidate_pool_runs_incomplete" in str(exc)
    else:
        raise AssertionError("audit should fail when one core run is missing")

    monkeypatch.setattr(
        runner,
        "status_payload",
        lambda: {
            "completed_scientific_runs": 184,
            "completed_seed_runs": 20,
            "completed_ablation_runs": 28,
            "stages": {
                **{stage: {"status": "complete_valid"} for stage in runner.STAGES},
                "core_candidate_pool_runs": {"status": "complete_valid", "completed_runs": 184, "expected_runs": 184},
                "random_repetitions": {"status": "complete_valid", "completed_runs": 120, "expected_runs": 120},
                "seed_downstream": {"status": "complete_valid", "completed_runs": 20, "expected_runs": 20},
                "ablation_downstream": {"status": "complete_valid", "completed_runs": 28, "expected_runs": 28},
            },
        },
    )
    assert runner.final_audit()["status"] == "passed"
