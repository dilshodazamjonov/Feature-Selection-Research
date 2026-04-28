from __future__ import annotations

import itertools
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from evaluation.stability_scores import calculate_psi, jaccard_similarity
from utils.feature_metadata import infer_semantic_group


def read_fold_feature_tables(features_dir: str | Path) -> list[pd.DataFrame]:
    """Read fold selected-feature CSV files in fold order."""
    root = Path(features_dir)
    tables: list[pd.DataFrame] = []

    combined_path = root / "fold_selected_features.csv"
    if combined_path.exists():
        combined = pd.read_csv(combined_path)
        if "feature_name" not in combined.columns and "feature" in combined.columns:
            combined["feature_name"] = combined["feature"]
        if "fold_id" in combined.columns:
            fold_ids = pd.to_numeric(combined["fold_id"], errors="coerce")
            for fold_id in sorted(fold_ids.dropna().astype(int).unique()):
                tables.append(
                    combined.loc[fold_ids == fold_id].copy().reset_index(drop=True)
                )
            if tables:
                return tables

    for fold_dir in sorted(root.glob("fold_*")):
        path = fold_dir / "selected_features.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "feature_name" not in df.columns and "feature" in df.columns:
            df["feature_name"] = df["feature"]
        if "fold_id" not in df.columns:
            try:
                df["fold_id"] = int(fold_dir.name.split("_")[-1])
            except ValueError:
                df["fold_id"] = len(tables) + 1
        tables.append(df)
    return tables


def selected_sets_from_tables(tables: Iterable[pd.DataFrame]) -> list[set[str]]:
    """Convert fold selected-feature tables into sets."""
    sets = []
    for table in tables:
        if "feature_name" not in table.columns:
            continue
        sets.append({str(value) for value in table["feature_name"].dropna().tolist()})
    return sets


def semantic_group_sets_from_tables(tables: Iterable[pd.DataFrame]) -> list[set[str]]:
    """Convert fold selected-feature tables into semantic-group sets."""
    sets = []
    for table in tables:
        working = table.copy()
        if "semantic_group" not in working.columns and "feature_name" in working.columns:
            working["semantic_group"] = working["feature_name"].map(infer_semantic_group)
        if "semantic_group" not in working.columns:
            continue
        sets.append({str(value) for value in working["semantic_group"].dropna().tolist()})
    return sets


def mean_pairwise_jaccard(selected_sets: list[set[str]]) -> float:
    """Mean pairwise Jaccard similarity across selected feature sets."""
    values = [
        jaccard_similarity(left, right)
        for left, right in itertools.combinations(selected_sets, 2)
    ]
    return float(np.mean(values)) if values else np.nan


def nogueira_stability(selected_sets: list[set[str]], total_features: int) -> float:
    """
    Approximate Nogueira stability over binary feature-selection indicators.

    Returns 1 for identical selected sets and approaches 0 for random-like
    selection under the observed average selected-set size.
    """
    if not selected_sets or total_features <= 1:
        return np.nan

    all_features = sorted(set().union(*selected_sets))
    if not all_features:
        return np.nan

    selected_counts = np.array([len(item) for item in selected_sets], dtype=float)
    k_bar = float(selected_counts.mean())
    if k_bar <= 0 or k_bar >= total_features:
        return 1.0

    frequencies = np.array(
        [sum(feature in selected for selected in selected_sets) / len(selected_sets) for feature in all_features],
        dtype=float,
    )
    observed_variance = float(np.mean(frequencies * (1.0 - frequencies)))
    expected_variance = (k_bar / total_features) * (1.0 - k_bar / total_features)
    if expected_variance <= 0:
        return np.nan
    return float(1.0 - observed_variance / expected_variance)


def kuncheva_stability(selected_sets: list[set[str]], total_features: int) -> float:
    """Mean pairwise Kuncheva stability for fixed-size selected sets."""
    if len(selected_sets) < 2 or total_features <= 1:
        return np.nan

    values = []
    for left, right in itertools.combinations(selected_sets, 2):
        k_left = len(left)
        k_right = len(right)
        if k_left == 0 or k_left != k_right or total_features == k_left:
            continue
        intersection = len(left & right)
        values.append((intersection * total_features - k_left**2) / (k_left * (total_features - k_left)))
    return float(np.mean(values)) if values else np.nan


def selection_frequency_frame(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Compute per-feature selection frequency and average rank/score."""
    if not tables:
        return pd.DataFrame(
            columns=[
                "feature_name",
                "selection_count",
                "total_folds",
                "selection_frequency",
                "mean_rank_if_available",
                "mean_score_if_available",
            ]
        )

    total_folds = len(tables)
    rows = []
    combined = pd.concat(tables, ignore_index=True)
    combined["feature_name"] = combined["feature_name"].astype(str)
    for feature, group in combined.groupby("feature_name"):
        ranks = (
            pd.to_numeric(group["rank"], errors="coerce")
            if "rank" in group.columns
            else pd.Series(dtype=float)
        )
        scores = (
            pd.to_numeric(group["score"], errors="coerce")
            if "score" in group.columns
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "feature_name": feature,
                "selection_count": int(group["fold_id"].nunique()),
                "total_folds": total_folds,
                "selection_frequency": float(group["fold_id"].nunique() / total_folds),
                "mean_rank_if_available": float(ranks.mean()) if ranks.notna().any() else np.nan,
                "mean_score_if_available": float(scores.mean()) if scores.notna().any() else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["selection_frequency", "selection_count", "feature_name"],
        ascending=[False, False, True],
    )


def semantic_group_frequency_frame(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Compute per-group selection frequency across folds."""
    if not tables:
        return pd.DataFrame(
            columns=["semantic_group", "selection_count", "total_folds", "selection_frequency"]
        )

    total_folds = len(tables)
    rows = []
    normalized = []
    for table in tables:
        working = table.copy()
        if "semantic_group" not in working.columns and "feature_name" in working.columns:
            working["semantic_group"] = working["feature_name"].map(infer_semantic_group)
        normalized.append(working)

    combined = pd.concat(normalized, ignore_index=True)
    if "semantic_group" not in combined.columns:
        return pd.DataFrame(
            columns=["semantic_group", "selection_count", "total_folds", "selection_frequency"]
        )

    combined["semantic_group"] = combined["semantic_group"].astype(str)
    for semantic_group, group in combined.groupby("semantic_group"):
        rows.append(
            {
                "semantic_group": semantic_group,
                "selection_count": int(group["fold_id"].nunique()),
                "total_folds": total_folds,
                "selection_frequency": float(group["fold_id"].nunique() / total_folds),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["selection_frequency", "selection_count", "semantic_group"],
        ascending=[False, False, True],
    )


def _rank_map(table: pd.DataFrame) -> dict[str, float]:
    if "feature_name" not in table.columns:
        return {}
    if "rank" in table.columns:
        ranks = pd.to_numeric(table["rank"], errors="coerce")
    elif "score" in table.columns:
        scores = pd.to_numeric(table["score"], errors="coerce")
        ranks = scores.rank(method="average", ascending=False)
    else:
        ranks = pd.Series(np.arange(1, len(table) + 1), index=table.index)

    return {
        str(feature): float(rank)
        for feature, rank in zip(table["feature_name"], ranks)
        if pd.notna(rank)
    }


def _spearman_from_ranks(left: dict[str, float], right: dict[str, float]) -> float:
    features = sorted(set(left) | set(right))
    if len(features) < 2:
        return np.nan
    left_missing = max(left.values(), default=0) + 1
    right_missing = max(right.values(), default=0) + 1
    x = np.array([left.get(feature, left_missing) for feature in features], dtype=float)
    y = np.array([right.get(feature, right_missing) for feature in features], dtype=float)
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _kendall_from_ranks(left: dict[str, float], right: dict[str, float]) -> float:
    features = sorted(set(left) | set(right))
    if len(features) < 2:
        return np.nan
    left_missing = max(left.values(), default=0) + 1
    right_missing = max(right.values(), default=0) + 1
    concordant = 0
    discordant = 0
    for feature_a, feature_b in itertools.combinations(features, 2):
        left_delta = left.get(feature_a, left_missing) - left.get(feature_b, left_missing)
        right_delta = right.get(feature_a, right_missing) - right.get(feature_b, right_missing)
        product = left_delta * right_delta
        if product > 0:
            concordant += 1
        elif product < 0:
            discordant += 1
    denom = concordant + discordant
    return float((concordant - discordant) / denom) if denom else np.nan


def rank_stability_frame(
    tables: list[pd.DataFrame],
    *,
    selector: str,
    model: str,
) -> pd.DataFrame:
    """Compute Spearman and Kendall rank stability across fold rankings."""
    rank_maps = [_rank_map(table) for table in tables]
    pairs = [(left, right) for left, right in itertools.combinations(rank_maps, 2) if left and right]
    rows = []
    for metric, fn in [
        ("spearman", _spearman_from_ranks),
        ("kendall_tau", _kendall_from_ranks),
    ]:
        values = [fn(left, right) for left, right in pairs]
        values = [value for value in values if pd.notna(value)]
        rows.append(
            {
                "selector": selector,
                "model": model,
                "metric": metric,
                "mean_value": float(np.mean(values)) if values else np.nan,
                "std_value": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                "n_pairs": len(values),
            }
        )
    return pd.DataFrame(rows)


def write_feature_stability_artifacts(
    *,
    exp_dir: str | Path,
    model: str,
    selector: str,
    total_candidate_features: int,
) -> dict[str, float]:
    """Write feature-selection stability artifacts for one experiment."""
    exp_path = Path(exp_dir)
    features_dir = exp_path / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    tables = read_fold_feature_tables(features_dir)
    selected_sets = selected_sets_from_tables(tables)
    semantic_group_sets = semantic_group_sets_from_tables(tables)
    frequency_df = selection_frequency_frame(tables)
    frequency_df.to_csv(features_dir / "selection_frequency.csv", index=False)
    semantic_group_frequency_df = semantic_group_frequency_frame(tables)
    semantic_group_frequency_df.to_csv(features_dir / "semantic_group_stability.csv", index=False)

    stable_count_80 = int((frequency_df["selection_frequency"] >= 0.8).sum()) if not frequency_df.empty else 0
    selected_feature_count = int(np.mean([len(item) for item in selected_sets])) if selected_sets else 0
    stable_ratio_80 = (
        float(stable_count_80 / selected_feature_count)
        if selected_feature_count
        else np.nan
    )
    stable_semantic_group_count_80 = (
        int((semantic_group_frequency_df["selection_frequency"] >= 0.8).sum())
        if not semantic_group_frequency_df.empty
        else 0
    )
    selected_semantic_group_count = (
        int(np.mean([len(item) for item in semantic_group_sets]))
        if semantic_group_sets
        else 0
    )
    semantic_group_stable_ratio_80 = (
        float(stable_semantic_group_count_80 / selected_semantic_group_count)
        if selected_semantic_group_count
        else np.nan
    )

    metrics = {
        "nogueira_stability": nogueira_stability(selected_sets, total_candidate_features),
        "kuncheva_stability": kuncheva_stability(selected_sets, total_candidate_features),
        "mean_pairwise_jaccard": mean_pairwise_jaccard(selected_sets),
        "semantic_group_jaccard": mean_pairwise_jaccard(semantic_group_sets),
        "stable_feature_count_80": stable_count_80,
        "stable_feature_ratio_80": stable_ratio_80,
        "stable_semantic_group_count_80": stable_semantic_group_count_80,
        "semantic_group_stable_ratio_80": semantic_group_stable_ratio_80,
        "total_candidate_feature_count": int(total_candidate_features),
    }

    ranking_tables = []
    llm_summary_path = features_dir / "llm_rankings_summary.csv"
    if llm_summary_path.exists():
        ranking_df = pd.read_csv(llm_summary_path)
        if "scope" in ranking_df.columns:
            ranking_df = ranking_df[ranking_df["scope"] == "fold"].copy()
        if "fold_id" in ranking_df.columns:
            fold_ids = pd.to_numeric(ranking_df["fold_id"], errors="coerce")
            for fold_id in sorted(fold_ids.dropna().astype(int).unique()):
                fold_rankings = ranking_df.loc[fold_ids == fold_id].copy()
                if "feature_name" in fold_rankings.columns:
                    ranking_tables.append(fold_rankings.reset_index(drop=True))

    rank_df = rank_stability_frame(ranking_tables or tables, selector=selector, model=model)
    for metric_name, output_prefix in [
        ("spearman", "spearman_rank_stability"),
        ("kendall_tau", "kendall_rank_stability"),
        ("rbo", "rbo_rank_stability"),
    ]:
        metric_rows = (
            rank_df[rank_df["metric"] == metric_name]
            if "metric" in rank_df.columns
            else pd.DataFrame()
        )
        if metric_rows.empty:
            metrics[f"{output_prefix}_mean"] = np.nan
            metrics[f"{output_prefix}_std"] = np.nan
        else:
            metrics[f"{output_prefix}_mean"] = float(metric_rows.iloc[0].get("mean_value", np.nan))
            metrics[f"{output_prefix}_std"] = float(metric_rows.iloc[0].get("std_value", np.nan))

    pd.DataFrame([metrics]).to_csv(features_dir / "feature_stability_metrics.csv", index=False)
    return metrics


def drift_level(psi_value: float) -> str:
    """Classify PSI drift using common credit-risk thresholds."""
    if pd.isna(psi_value):
        return "unknown"
    if psi_value < 0.1:
        return "low"
    if psi_value < 0.25:
        return "moderate"
    return "high"


def selected_feature_psi_frame(X_dev: pd.DataFrame, X_oot: pd.DataFrame) -> pd.DataFrame:
    """Compute DEV-vs-OOT PSI for selected final model features."""
    rows = []
    for feature in X_dev.columns:
        psi = calculate_psi(X_dev[feature], X_oot[feature])
        rows.append({"feature_name": feature, "psi": psi, "drift_level": drift_level(psi)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, na_position="last")


def selected_feature_psi_summary(psi_df: pd.DataFrame) -> dict[str, float]:
    """Summarize selected-feature PSI for aggregation."""
    if psi_df.empty or "psi" not in psi_df.columns:
        return {
            "selected_feature_psi_mean": np.nan,
            "selected_feature_psi_max": np.nan,
            "selected_feature_psi_median": np.nan,
            "selected_feature_psi_high_drift_ratio": np.nan,
            "selected_feature_psi_moderate_or_high_drift_ratio": np.nan,
        }
    psi = pd.to_numeric(psi_df["psi"], errors="coerce")
    return {
        "selected_feature_psi_mean": float(psi.mean()),
        "selected_feature_psi_max": float(psi.max()),
        "selected_feature_psi_median": float(psi.median()),
        "selected_feature_psi_high_drift_ratio": float((psi >= 0.25).mean()),
        "selected_feature_psi_moderate_or_high_drift_ratio": float((psi >= 0.1).mean()),
    }
