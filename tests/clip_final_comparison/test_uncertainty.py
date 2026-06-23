from __future__ import annotations

import numpy as np

from credit_risk_fs.clip_final_comparison.uncertainty import (
    benjamini_hochberg,
    cluster_bootstrap_deltas,
    metric_bundle,
    paired_bootstrap_deltas,
    random_distribution_summary,
    summarize_uncertainty,
)


def test_metric_recompute_and_paired_bootstrap_identical_predictions_zero_delta():
    y = np.array([0, 0, 1, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.8, 0.7, 0.4, 0.9])
    metrics = metric_bundle(y, score)
    assert metrics["auc"] == 1.0
    samples = paired_bootstrap_deltas(y, score, score, n_replicates=25, seed=42)
    summary = summarize_uncertainty(samples)
    assert set(summary["metric"]) == {"auc", "ks", "lift_at_10"}
    assert summary["mean_delta"].abs().max() == 0.0


def test_cluster_bootstrap_preserves_paired_rows_by_cluster():
    y = np.array([0, 1, 0, 1, 0, 1])
    base = np.array([0.2, 0.6, 0.3, 0.7, 0.4, 0.8])
    candidate = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.95])
    clusters = np.array(["m1", "m1", "m2", "m2", "m3", "m3"])
    samples = cluster_bootstrap_deltas(y, base, candidate, clusters, n_replicates=20, seed=7)
    assert {"delta_auc", "delta_ks", "delta_lift_at_10"}.issubset(samples.columns)


def test_random_distribution_and_multiple_comparison_policy():
    summary = random_distribution_summary([0.60, 0.62, 0.64], clip_v2_value=0.63)
    assert summary["random_median"] == 0.62
    assert 0.0 <= summary["clip_v2_empirical_percentile"] <= 1.0
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
    assert len(adjusted) == 3
    assert all(0.0 <= value <= 1.0 for value in adjusted)

