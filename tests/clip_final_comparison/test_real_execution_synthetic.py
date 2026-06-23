from __future__ import annotations

import json

import numpy as np
import pandas as pd

from credit_risk_fs.clip_final_comparison.execution import (
    ComparisonRunSpec,
    PreparedFrame,
    aggregate_runs,
    execute_comparison_run,
    validate_run,
    write_minimal_plot,
)


def _synthetic_data() -> PreparedFrame:
    rng = np.random.default_rng(123)
    n_dev = 80
    n_oot = 40
    X_dev = pd.DataFrame(
        {
            "strong": rng.normal(size=n_dev),
            "weak": rng.normal(size=n_dev),
            "cat": np.where(np.arange(n_dev) % 3 == 0, "a", "b"),
            "noise": rng.normal(size=n_dev),
        }
    )
    y_dev = (X_dev["strong"] + rng.normal(scale=0.25, size=n_dev) > 0).astype(int)
    X_oot = pd.DataFrame(
        {
            "strong": rng.normal(size=n_oot),
            "weak": rng.normal(size=n_oot),
            "cat": np.where(np.arange(n_oot) % 2 == 0, "a", "c"),
            "noise": rng.normal(size=n_oot),
        }
    )
    y_oot = (X_oot["strong"] + rng.normal(scale=0.35, size=n_oot) > 0).astype(int)
    return PreparedFrame(X_dev=X_dev, y_dev=y_dev, X_oot=X_oot, y_oot=y_oot)


def test_synthetic_run_executes_from_screening_to_predictions_and_aggregation(tmp_path):
    spec = ComparisonRunSpec(
        run_id="synthetic_lr_variance_2x",
        dataset="synthetic",
        model="lr",
        screening_method="variance",
        pool_multiplier=2,
        candidate_pool_size=4,
        final_feature_budget=2,
    )
    run_dir = tmp_path / "runs" / spec.run_id
    validation = execute_comparison_run(spec, _synthetic_data(), run_dir, model_kwargs={"random_state": 7})
    assert validation["prediction_rows"] == 40
    assert (run_dir / "RUN_COMPLETE.json").exists()
    assert (run_dir / "oot_predictions.parquet").exists()
    assert validate_run(run_dir)["selected_count"] == 2

    paths = aggregate_runs([run_dir], tmp_path / "final_analysis")
    master = pd.read_csv(paths["master_results"])
    assert master.loc[0, "run_id"] == spec.run_id
    plot = tmp_path / "final_analysis" / "plots" / "auc.png"
    write_minimal_plot(master, plot)
    assert plot.exists()


def test_missing_predictions_invalidate_synthetic_run(tmp_path):
    spec = ComparisonRunSpec(
        run_id="synthetic_lr_random_2x_seed101",
        dataset="synthetic",
        model="lr",
        screening_method="random",
        pool_multiplier=2,
        candidate_pool_size=4,
        final_feature_budget=2,
        random_seed=101,
    )
    run_dir = tmp_path / "runs" / spec.run_id
    execute_comparison_run(spec, _synthetic_data(), run_dir, model_kwargs={"random_state": 7})
    (run_dir / "oot_predictions.parquet").unlink()
    try:
        validate_run(run_dir)
    except RuntimeError as exc:
        assert "missing artifacts" in str(exc)
    else:
        raise AssertionError("run without predictions should not validate")


def test_metric_mismatch_invalidate_run(tmp_path):
    spec = ComparisonRunSpec(
        run_id="synthetic_lr_variance_bad_metric",
        dataset="synthetic",
        model="lr",
        screening_method="variance",
        pool_multiplier=2,
        candidate_pool_size=4,
        final_feature_budget=2,
    )
    run_dir = tmp_path / "runs" / spec.run_id
    execute_comparison_run(spec, _synthetic_data(), run_dir, model_kwargs={"random_state": 7})
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics["auc"] = 0.0
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    try:
        validate_run(run_dir)
    except RuntimeError as exc:
        assert "metric mismatch" in str(exc)
    else:
        raise AssertionError("metric mismatch should not validate")
