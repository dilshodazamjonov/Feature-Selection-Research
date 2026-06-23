from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def ks_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


def lift_at_fraction(y_true: np.ndarray, scores: np.ndarray, fraction: float = 0.10) -> float:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = max(1, int(np.ceil(len(y_true) * fraction)))
    order = np.argsort(scores)[::-1][:n]
    base_rate = float(np.mean(y_true))
    if base_rate == 0.0:
        return float("nan")
    return float(np.mean(y_true[order]) / base_rate)


def metric_bundle(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "ks": ks_score(y_true, scores),
        "lift_at_10": lift_at_fraction(y_true, scores, 0.10),
    }


def paired_bootstrap_deltas(
    y_true: np.ndarray,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    n_replicates: int,
    seed: int,
) -> pd.DataFrame:
    y = np.asarray(y_true)
    base = np.asarray(base_scores)
    candidate = np.asarray(candidate_scores)
    if not (len(y) == len(base) == len(candidate)):
        raise ValueError("paired bootstrap inputs must have equal row counts")
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(n_replicates):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        b = metric_bundle(y[idx], base[idx])
        c = metric_bundle(y[idx], candidate[idx])
        rows.append({"replicate": replicate, **{f"delta_{key}": c[key] - b[key] for key in b}})
    return pd.DataFrame(rows)


def cluster_bootstrap_deltas(
    y_true: np.ndarray,
    base_scores: np.ndarray,
    candidate_scores: np.ndarray,
    clusters: np.ndarray,
    *,
    n_replicates: int,
    seed: int,
) -> pd.DataFrame:
    y = np.asarray(y_true)
    base = np.asarray(base_scores)
    candidate = np.asarray(candidate_scores)
    clusters = np.asarray(clusters)
    if not (len(y) == len(base) == len(candidate) == len(clusters)):
        raise ValueError("cluster bootstrap inputs must have equal row counts")
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    rows = []
    for replicate in range(n_replicates):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([np.flatnonzero(clusters == cluster) for cluster in sampled_clusters])
        if len(np.unique(y[idx])) < 2:
            continue
        b = metric_bundle(y[idx], base[idx])
        c = metric_bundle(y[idx], candidate[idx])
        rows.append({"replicate": replicate, **{f"delta_{key}": c[key] - b[key] for key in b}})
    return pd.DataFrame(rows)


def summarize_uncertainty(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in [col for col in samples.columns if col.startswith("delta_")]:
        values = pd.to_numeric(samples[column], errors="coerce").dropna().to_numpy()
        if len(values) == 0:
            rows.append({"metric": column.removeprefix("delta_"), "samples": 0})
            continue
        rows.append(
            {
                "metric": column.removeprefix("delta_"),
                "samples": int(len(values)),
                "mean_delta": float(values.mean()),
                "ci95_lower": float(np.quantile(values, 0.025)),
                "ci95_upper": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def random_distribution_summary(values: list[float], clip_v2_value: float) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("random distribution summary requires at least one value")
    summary = {
        "mean": float(arr.mean()),
        "standard_deviation": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "minimum": float(arr.min()),
        "maximum": float(arr.max()),
        "interquartile_range": float(np.quantile(arr, 0.75) - np.quantile(arr, 0.25)),
        "empirical_percentile_of_clip_v2": float((arr <= clip_v2_value).mean()),
        "number_of_valid_repetitions": int(arr.size),
    }
    summary.update(
        {
            "random_mean": summary["mean"],
            "random_std": summary["standard_deviation"],
            "random_median": summary["median"],
            "random_min": summary["minimum"],
            "random_max": summary["maximum"],
            "clip_v2_empirical_percentile": summary["empirical_percentile_of_clip_v2"],
        }
    )
    return summary


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 1.0
    m = len(p)
    for rank, idx in enumerate(order[::-1], start=1):
        original_rank = m - rank + 1
        running = min(running, p[idx] * m / original_rank)
        adjusted[idx] = running
    return adjusted.tolist()
