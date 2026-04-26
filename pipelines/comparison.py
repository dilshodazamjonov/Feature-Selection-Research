from __future__ import annotations

from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd


def _safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return float(numeric.mean())
    return float("nan")


def _safe_std(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= 2:
        return float(numeric.std())
    return float("nan")


def _safe_value(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return float("nan")
    return _safe_mean(df[column])


def _numeric_fold_rows(df: pd.DataFrame) -> pd.DataFrame:
    fold_numeric = pd.to_numeric(df["fold"], errors="coerce")
    mask = fold_numeric.notna()
    filtered = df.loc[mask].copy()
    filtered["fold"] = fold_numeric.loc[mask].astype(int)
    return filtered.sort_values("fold").reset_index(drop=True)


def _read_feature_file(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        return set()
    return {str(value) for value in df["feature"].dropna().tolist()}


def _safe_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def build_experiment_summary_row(
    exp_dir: str | Path,
    method_name: str,
    model_name: str,
    selector_name: str,
) -> dict[str, object]:
    exp_path = Path(exp_dir)
    cv_results_path = exp_path / "results" / "cv_results.csv"
    cv_df = pd.read_csv(cv_results_path)
    folds_df = _numeric_fold_rows(cv_df)

    summary: dict[str, object] = {
        "method": method_name,
        "model_name": model_name,
        "selector_name": selector_name,
        "exp_dir": str(exp_path.resolve()),
        "cv_fold_count": int(len(folds_df)),
    }

    for metric in [
        "auc",
        "gini",
        "ks",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "selected_features",
        "psi_feature_mean",
        "psi_feature_max",
        "psi_model",
        "jaccard_similarity",
        "fold_time_sec",
    ]:
        if metric in folds_df.columns:
            summary[f"cv_{metric}_mean"] = _safe_mean(folds_df[metric])
            summary[f"cv_{metric}_std"] = _safe_std(folds_df[metric])

    oot_path = exp_path / "results" / "oot_test_results.csv"
    if oot_path.exists():
        oot_df = pd.read_csv(oot_path)
        if not oot_df.empty:
            for metric in [
                "auc",
                "gini",
                "ks",
                "log_loss",
                "brier",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "decision_threshold",
                "selected_feature_count",
                "total_candidate_feature_count",
                "feature_budget",
                "feature_reduction_ratio",
                "oot_gini_per_feature",
                "oot_auc_per_feature",
                "oot_ks_per_feature",
                "selected_feature_psi_mean",
                "selected_feature_psi_max",
                "selected_feature_psi_median",
                "selected_feature_psi_high_drift_ratio",
                "selected_feature_psi_moderate_or_high_drift_ratio",
                "model_score_psi",
                "lift_at_10",
                "bad_rate_capture_at_10",
                "lift_at_20",
                "bad_rate_capture_at_20",
            ]:
                if metric in oot_df.columns:
                    summary[f"oot_{metric}"] = _safe_value(oot_df, metric)

    stability_path = exp_path / "features" / "feature_stability_metrics.csv"
    if stability_path.exists():
        stability_df = pd.read_csv(stability_path)
        if not stability_df.empty:
            summary.update(stability_df.iloc[0].to_dict())
    else:
        legacy_stability_path = exp_path / "results" / "feature_stability_metrics.json"
        if legacy_stability_path.exists():
            stability = json.loads(legacy_stability_path.read_text(encoding="utf-8"))
            summary.update(stability)

    rank_path = exp_path / "results" / "rank_stability.csv"
    if rank_path.exists():
        rank_df = pd.read_csv(rank_path)
        for metric_name, output_name in [
            ("spearman", "spearman_rank_stability_mean"),
            ("kendall_tau", "kendall_rank_stability_mean"),
            ("rbo", "rbo_rank_stability_mean"),
        ]:
            metric_rows = rank_df[rank_df.get("metric") == metric_name] if "metric" in rank_df.columns else pd.DataFrame()
            if not metric_rows.empty and "mean_value" in metric_rows.columns:
                summary[output_name] = _safe_value(metric_rows, "mean_value")

    final_features_path = exp_path / "features" / "final_selected_features.csv"
    final_features = _read_feature_file(final_features_path)
    summary["final_selected_feature_count"] = int(len(final_features)) if final_features else np.nan

    return summary


def build_experiment_summary_frame(
    experiments: Iterable[tuple[str, str | Path, str, str]],
) -> pd.DataFrame:
    rows = [
        build_experiment_summary_row(
            exp_dir=exp_dir,
            method_name=method_name,
            model_name=model_name,
            selector_name=selector_name,
        )
        for method_name, exp_dir, model_name, selector_name in experiments
    ]
    return pd.DataFrame(rows)


def load_fold_feature_sets(exp_dir: str | Path) -> dict[int, set[str]]:
    exp_path = Path(exp_dir)
    features_dir = exp_path / "features"
    fold_sets: dict[int, set[str]] = {}

    if not features_dir.exists():
        return fold_sets

    combined_path = features_dir / "fold_selected_features.csv"
    if combined_path.exists():
        combined = pd.read_csv(combined_path)
        feature_col = "feature_name" if "feature_name" in combined.columns else "feature"
        if "fold_id" in combined.columns and feature_col in combined.columns:
            fold_ids = pd.to_numeric(combined["fold_id"], errors="coerce")
            for fold in sorted(fold_ids.dropna().astype(int).unique()):
                fold_sets[int(fold)] = {
                    str(value)
                    for value in combined.loc[fold_ids == fold, feature_col].dropna().tolist()
                }
            if fold_sets:
                return fold_sets

    for fold_dir in sorted(features_dir.glob("fold_*")):
        try:
            fold = int(fold_dir.name.split("_")[-1])
        except ValueError:
            continue
        fold_sets[fold] = _read_feature_file(fold_dir / "selected_features.csv")

    return fold_sets


def build_feature_overlap_frame(
    left_label: str,
    left_exp_dir: str | Path,
    right_label: str,
    right_exp_dir: str | Path,
) -> pd.DataFrame:
    left_folds = load_fold_feature_sets(left_exp_dir)
    right_folds = load_fold_feature_sets(right_exp_dir)
    rows: list[dict[str, object]] = []

    common_folds = sorted(set(left_folds) & set(right_folds))
    for fold in common_folds:
        left_set = left_folds[fold]
        right_set = right_folds[fold]
        intersection = left_set & right_set
        union = left_set | right_set
        rows.append(
            {
                "stage": "fold",
                "fold": fold,
                "left_method": left_label,
                "right_method": right_label,
                f"{left_label}_feature_count": len(left_set),
                f"{right_label}_feature_count": len(right_set),
                "intersection_count": len(intersection),
                "union_count": len(union),
                "jaccard_similarity": _safe_jaccard(left_set, right_set),
            }
        )

    left_final = _read_feature_file(Path(left_exp_dir) / "features" / "final_selected_features.csv")
    right_final = _read_feature_file(Path(right_exp_dir) / "features" / "final_selected_features.csv")
    if left_final and right_final:
        intersection = left_final & right_final
        union = left_final | right_final
        rows.append(
            {
                "stage": "final",
                "fold": np.nan,
                "left_method": left_label,
                "right_method": right_label,
                f"{left_label}_feature_count": len(left_final),
                f"{right_label}_feature_count": len(right_final),
                "intersection_count": len(intersection),
                "union_count": len(union),
                "jaccard_similarity": _safe_jaccard(left_final, right_final),
            }
        )

    return pd.DataFrame(rows)


def compare_experiment_pair(
    left_label: str,
    left_exp_dir: str | Path,
    left_model_name: str,
    left_selector_name: str,
    right_label: str,
    right_exp_dir: str | Path,
    right_model_name: str,
    right_selector_name: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    left_summary = build_experiment_summary_row(
        exp_dir=left_exp_dir,
        method_name=left_label,
        model_name=left_model_name,
        selector_name=left_selector_name,
    )
    right_summary = build_experiment_summary_row(
        exp_dir=right_exp_dir,
        method_name=right_label,
        model_name=right_model_name,
        selector_name=right_selector_name,
    )
    overlap_df = build_feature_overlap_frame(
        left_label=left_label,
        left_exp_dir=left_exp_dir,
        right_label=right_label,
        right_exp_dir=right_exp_dir,
    )

    fold_overlap = overlap_df[overlap_df["stage"] == "fold"].copy() if not overlap_df.empty else pd.DataFrame()
    final_overlap = overlap_df[overlap_df["stage"] == "final"].copy() if not overlap_df.empty else pd.DataFrame()

    summary: dict[str, object] = {
        "left_method": left_label,
        "right_method": right_label,
        "left_exp_dir": left_summary["exp_dir"],
        "right_exp_dir": right_summary["exp_dir"],
        "feature_overlap_fold_count": int(len(fold_overlap)),
        "feature_overlap_fold_jaccard_mean": (
            _safe_mean(fold_overlap["jaccard_similarity"])
            if not fold_overlap.empty
            else np.nan
        ),
        "feature_overlap_fold_jaccard_std": (
            _safe_std(fold_overlap["jaccard_similarity"])
            if not fold_overlap.empty
            else np.nan
        ),
        "feature_overlap_final_jaccard": (
            float(final_overlap.iloc[0]["jaccard_similarity"])
            if not final_overlap.empty
            else np.nan
        ),
        "feature_overlap_final_intersection_count": (
            int(final_overlap.iloc[0]["intersection_count"])
            if not final_overlap.empty
            else np.nan
        ),
    }

    tracked_metrics = [
        "cv_auc_mean",
        "cv_gini_mean",
        "cv_ks_mean",
        "cv_precision_mean",
        "cv_recall_mean",
        "cv_f1_mean",
        "cv_accuracy_mean",
        "cv_selected_features_mean",
        "cv_psi_feature_mean_mean",
        "cv_psi_feature_max_mean",
        "cv_psi_model_mean",
        "cv_jaccard_similarity_mean",
        "oot_auc",
        "oot_gini",
        "oot_ks",
        "oot_precision",
        "oot_recall",
        "oot_f1",
        "oot_accuracy",
    ]

    for metric in tracked_metrics:
        left_value = left_summary.get(metric, np.nan)
        right_value = right_summary.get(metric, np.nan)
        summary[f"left_{metric}"] = left_value
        summary[f"right_{metric}"] = right_value

        if pd.notna(left_value) and pd.notna(right_value):
            summary[f"delta_{metric}_right_minus_left"] = round(
                float(right_value) - float(left_value),
                12,
            )
        else:
            summary[f"delta_{metric}_right_minus_left"] = np.nan

    return summary, overlap_df
