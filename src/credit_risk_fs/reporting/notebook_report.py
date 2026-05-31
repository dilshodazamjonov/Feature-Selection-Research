from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import json
import math

import pandas as pd
import matplotlib.pyplot as plt

from credit_risk_fs.experiments.config import load_named_project_config
from credit_risk_fs.feature_engineering.homecredit.assemble import build_application_time_proxy
from credit_risk_fs.feature_metadata.builder import infer_semantic_group
from credit_risk_fs.preprocessing.lendingclub import ensure_lendingclub_target_and_time
from credit_risk_fs.utils.paths import project_root, results_root


SUPPORTED_DATASETS = {"homecredit", "lendingclub"}
BASELINE_SELECTORS = {"mrmr", "boruta", "pca", "domain_rule_baseline"}
LLM_FAMILY_SELECTORS = {"llm", "llm_then_mrmr", "llm_then_boruta", "stable_core_llm_fill"}
FINAL_REPORT_PLOT_COLUMNS = [
    "plot_file",
    "source_table",
    "rows_used",
    "columns_used",
    "purpose",
    "status",
    "skip_reason",
]


@dataclass(frozen=True, slots=True)
class DatasetPaths:
    dataset_name: str
    root: Path
    results_dir: Path


def _normalize_dataset_name(dataset_name: str) -> str:
    normalized = str(dataset_name).strip().lower()
    if normalized not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return normalized


def _dataset_role(dataset_name: str) -> str:
    return "primary benchmark" if dataset_name == "homecredit" else "external validation"


def _display_dataset_name(dataset_name: str) -> str:
    return "Home Credit" if dataset_name == "homecredit" else "LendingClub"


def _round_numeric_frame(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    rounded = df.copy()
    numeric_cols = rounded.select_dtypes(include=["number"]).columns
    rounded[numeric_cols] = rounded[numeric_cols].round(digits)
    return rounded


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_relative_path(path_text: str | Path) -> str:
    path = Path(path_text)
    try:
        return path.resolve().relative_to(project_root().resolve()).as_posix()
    except (OSError, ValueError):
        return str(path_text).replace("\\", "/")


def _dataset_paths(dataset_name: str) -> DatasetPaths:
    normalized = _normalize_dataset_name(dataset_name)
    root = project_root()
    return DatasetPaths(
        dataset_name=normalized,
        root=root,
        results_dir=results_root(normalized),
    )


@lru_cache(maxsize=None)
def _dataset_config(dataset_name: str) -> dict[str, Any]:
    return load_named_project_config(_normalize_dataset_name(dataset_name), experiment_name="matrix")


def _first_existing_split_manifest(dataset_name: str) -> Path | None:
    paths = _dataset_paths(dataset_name)
    candidates = sorted(paths.results_dir.rglob("data_split_manifest.json"))
    return candidates[0] if candidates else None


def _safe_rate_delta(dev_rate: float, oot_rate: float) -> float:
    if pd.isna(dev_rate) or pd.isna(oot_rate):
        return math.nan
    return float(oot_rate - dev_rate)


@lru_cache(maxsize=None)
def _matrix_runs(dataset_name: str) -> pd.DataFrame:
    paths = _dataset_paths(dataset_name)
    matrix_df = _read_csv(paths.results_dir / "matrix_runs.csv")
    if not matrix_df.empty:
        return matrix_df

    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(paths.results_dir.rglob("run_manifest.json")):
        payload = _read_json(manifest_path)
        rows.append(
            {
                "run_id": payload.get("run_id"),
                "model": payload.get("model"),
                "selector": payload.get("selector"),
                "experiment_type": payload.get("experiment_type"),
                "status": payload.get("status"),
                "config_hash": payload.get("config_hash"),
                "output_folder": str(manifest_path.parent),
            }
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def _failed_runs(dataset_name: str) -> pd.DataFrame:
    paths = _dataset_paths(dataset_name)
    failed_df = _read_csv(paths.results_dir / "failed_runs.csv")
    return failed_df


def _rebuild_final_comparison(dataset_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(_dataset_paths(dataset_name).results_dir.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "completed":
            continue
        summary_path = run_dir / "results" / "experiment_summary.csv"
        if not summary_path.exists():
            continue
        summary_df = _read_csv(summary_path)
        if summary_df.empty:
            continue
        summary = summary_df.iloc[0]
        rows.append(
            {
                "dataset_name": dataset_name,
                "model": manifest.get("model"),
                "selector": manifest.get("selector"),
                "experiment_type": manifest.get("experiment_type"),
                "feature_budget": summary.get("oot_feature_budget"),
                "llm_shared_ranking_enabled": manifest.get("config", {}).get("llm", {}).get("shared_ranking_enabled"),
                "llm_ranking_budget": manifest.get("config", {}).get("llm", {}).get("ranking_budget", {}).get("max_shared_pool"),
                "oot_auc": summary.get("oot_auc"),
                "oot_gini": summary.get("oot_gini"),
                "oot_ks": summary.get("oot_ks"),
                "oot_log_loss": summary.get("oot_log_loss"),
                "oot_brier": summary.get("oot_brier"),
                "nogueira_stability": summary.get("nogueira_stability"),
                "kuncheva_stability": summary.get("kuncheva_stability"),
                "mean_pairwise_jaccard": summary.get("mean_pairwise_jaccard"),
                "semantic_group_jaccard": summary.get("semantic_group_jaccard"),
                "stable_feature_count_80": summary.get("stable_feature_count_80"),
                "stable_feature_ratio_80": summary.get("stable_feature_ratio_80"),
                "stable_semantic_group_count_80": summary.get("stable_semantic_group_count_80"),
                "semantic_group_stable_ratio_80": summary.get("semantic_group_stable_ratio_80"),
                "spearman_rank_stability_mean": summary.get("spearman_rank_stability_mean"),
                "kendall_rank_stability_mean": summary.get("kendall_rank_stability_mean"),
                "rbo_rank_stability_mean": summary.get("rbo_rank_stability_mean"),
                "selected_feature_psi_mean": summary.get("oot_selected_feature_psi_mean"),
                "selected_feature_psi_max": summary.get("oot_selected_feature_psi_max"),
                "selected_feature_psi_median": summary.get("oot_selected_feature_psi_median"),
                "selected_feature_psi_high_drift_ratio": summary.get("oot_selected_feature_psi_high_drift_ratio"),
                "selected_feature_psi_moderate_or_high_drift_ratio": summary.get(
                    "oot_selected_feature_psi_moderate_or_high_drift_ratio"
                ),
                "model_score_psi": summary.get("oot_model_score_psi"),
                "selected_feature_count": summary.get("oot_selected_feature_count"),
                "total_candidate_feature_count": summary.get("oot_total_candidate_feature_count"),
                "feature_reduction_ratio": summary.get("oot_feature_reduction_ratio"),
                "oot_gini_per_feature": summary.get("oot_oot_gini_per_feature"),
                "oot_auc_per_feature": summary.get("oot_oot_auc_per_feature"),
                "oot_ks_per_feature": summary.get("oot_oot_ks_per_feature"),
                "lift_at_10": summary.get("oot_lift_at_10"),
                "bad_rate_capture_at_10": summary.get("oot_bad_rate_capture_at_10"),
                "lift_at_20": summary.get("oot_lift_at_20"),
                "bad_rate_capture_at_20": summary.get("oot_bad_rate_capture_at_20"),
                "config_hash": manifest.get("config_hash"),
                "data_fingerprint": json.dumps(manifest.get("data_version", {}), sort_keys=True),
                "run_id": manifest.get("run_id"),
                "output_folder": str(run_dir),
                "runtime_seconds": summary.get("runtime_seconds"),
            }
        )
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def _final_comparison(dataset_name: str) -> pd.DataFrame:
    paths = _dataset_paths(dataset_name)
    final_df = _read_csv(paths.results_dir / "final_comparison_table.csv")
    if final_df.empty:
        final_df = _rebuild_final_comparison(dataset_name)
    if final_df.empty:
        return final_df
    return final_df.sort_values(["model", "oot_auc", "oot_gini"], ascending=[True, False, False]).reset_index(drop=True)


def _semantic_fallback(dataset_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    final_df = _final_comparison(dataset_name)
    for _, row in final_df.iterrows():
        run_dir = Path(str(row["output_folder"]))
        selected_path = run_dir / "features" / "final_selected_features.csv"
        selected_df = _read_csv(selected_path)
        if selected_df.empty or "semantic_group" not in selected_df.columns:
            continue
        summary = (
            selected_df.groupby("semantic_group", dropna=False)["feature_name"]
            .count()
            .reset_index(name="feature_count")
            .sort_values("feature_count", ascending=False)
        )
        total = float(summary["feature_count"].sum())
        summary["feature_ratio"] = summary["feature_count"] / total if total else math.nan
        for _, item in summary.iterrows():
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "run_id": row["run_id"],
                    "model": row["model"],
                    "selector": row["selector"],
                    "experiment_type": row["experiment_type"],
                    "semantic_group": item["semantic_group"],
                    "feature_count": item["feature_count"],
                    "feature_ratio": item["feature_ratio"],
                }
            )
    return pd.DataFrame(rows)


@lru_cache(maxsize=None)
def _semantic_coverage(dataset_name: str) -> pd.DataFrame:
    paths = _dataset_paths(dataset_name)
    semantic_df = pd.DataFrame()
    if _normalize_dataset_name(dataset_name) == "lendingclub":
        semantic_df = _read_csv(
            paths.results_dir
            / "analysis"
            / "semantic_redundancy"
            / "semantic_coverage_by_pipeline_relabelled.csv"
        )
    if semantic_df.empty:
        semantic_df = _read_csv(paths.results_dir / "semantic_coverage_table.csv")
    if semantic_df.empty:
        semantic_df = _semantic_fallback(dataset_name)
    return semantic_df


def _subset_or_merge_from_final(dataset_name: str, filename: str, columns: list[str]) -> pd.DataFrame:
    path = _dataset_paths(dataset_name).results_dir / filename
    df = _read_csv(path)
    if not df.empty:
        return df
    final_df = _final_comparison(dataset_name)
    available = [col for col in columns if col in final_df.columns]
    return final_df[available].copy() if available else pd.DataFrame()


@lru_cache(maxsize=None)
def _stability(dataset_name: str) -> pd.DataFrame:
    columns = [
        "dataset_name",
        "model",
        "selector",
        "experiment_type",
        "nogueira_stability",
        "kuncheva_stability",
        "mean_pairwise_jaccard",
        "semantic_group_jaccard",
        "stable_feature_count_80",
        "stable_feature_ratio_80",
        "stable_semantic_group_count_80",
        "semantic_group_stable_ratio_80",
        "spearman_rank_stability_mean",
        "kendall_rank_stability_mean",
        "rbo_rank_stability_mean",
    ]
    return _subset_or_merge_from_final(dataset_name, "feature_stability_table.csv", columns)


@lru_cache(maxsize=None)
def _drift(dataset_name: str) -> pd.DataFrame:
    columns = [
        "dataset_name",
        "model",
        "selector",
        "experiment_type",
        "selected_feature_psi_mean",
        "selected_feature_psi_max",
        "selected_feature_psi_median",
        "selected_feature_psi_high_drift_ratio",
        "selected_feature_psi_moderate_or_high_drift_ratio",
        "model_score_psi",
    ]
    return _subset_or_merge_from_final(dataset_name, "feature_drift_table.csv", columns)


@lru_cache(maxsize=None)
def _paired_fold(dataset_name: str) -> pd.DataFrame:
    return _read_csv(_dataset_paths(dataset_name).results_dir / "paired_fold_comparisons.csv")


@lru_cache(maxsize=None)
def _llm_calls(dataset_name: str) -> pd.DataFrame:
    return _read_csv(_dataset_paths(dataset_name).results_dir / "llm_call_summary.csv")


def _analysis_csv(dataset_name: str, relative_path: str) -> pd.DataFrame:
    return _read_csv(_dataset_paths(dataset_name).results_dir / "analysis" / Path(relative_path))


@lru_cache(maxsize=None)
def _feature_level_psi(dataset_name: str) -> pd.DataFrame:
    return _analysis_csv(dataset_name, "feature_level_drift/feature_level_psi_by_run.csv")


@lru_cache(maxsize=None)
def _psi_distribution(dataset_name: str) -> pd.DataFrame:
    return _analysis_csv(dataset_name, "feature_level_drift/psi_distribution_by_pipeline.csv")


@lru_cache(maxsize=None)
def _high_psi_features(dataset_name: str) -> pd.DataFrame:
    return _analysis_csv(dataset_name, "feature_level_drift/high_psi_features_by_pipeline.csv")


@lru_cache(maxsize=None)
def _llm_then_mrmr_drift_source(dataset_name: str) -> pd.DataFrame:
    return _analysis_csv(dataset_name, "feature_level_drift/llm_then_mrmr_drift_source_breakdown.csv")


@lru_cache(maxsize=None)
def _llm_top100_candidate_psi(dataset_name: str) -> pd.DataFrame:
    return _analysis_csv(dataset_name, "feature_level_drift/llm_top100_candidate_psi.csv")


@lru_cache(maxsize=None)
def _semantic_redundancy(dataset_name: str) -> pd.DataFrame:
    return _analysis_csv(dataset_name, "semantic_redundancy/semantic_coverage_redundancy_by_pipeline.csv")


@lru_cache(maxsize=None)
def _cross_dataset_stability_significance() -> pd.DataFrame:
    return _read_csv(
        project_root()
        / "results"
        / "cross_dataset"
        / "analysis"
        / "stability_significance"
        / "llm_stability_diagnosis.csv"
    )


@lru_cache(maxsize=None)
def _paired_fold_significance() -> pd.DataFrame:
    return _read_csv(
        project_root()
        / "results"
        / "cross_dataset"
        / "analysis"
        / "stability_significance"
        / "paired_fold_significance_tests.csv"
    )


def _report_text(relative_path: str) -> str:
    path = project_root() / relative_path
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _warnings(dataset_name: str) -> list[str]:
    paths = _dataset_paths(dataset_name)
    warnings: list[str] = []
    for filename in [
        "final_comparison_table.csv",
        "feature_stability_table.csv",
        "feature_drift_table.csv",
        "semantic_coverage_table.csv",
        "paired_fold_comparisons.csv",
        "llm_call_summary.csv",
        "matrix_runs.csv",
        "failed_runs.csv",
    ]:
        path = paths.results_dir / filename
        if not path.exists():
            warnings.append(f"Missing aggregate artifact: {path.relative_to(paths.root)}")
        elif _read_csv(path).empty and filename != "failed_runs.csv":
            warnings.append(f"Aggregate artifact is empty: {path.relative_to(paths.root)}")
    if dataset_name == "homecredit":
        warnings.append(
            "Home Credit split diagnostics use application_train plus previous_application recency only, because no saved processed modeling table exists under data/homecredit/processed."
        )
    if dataset_name == "lendingclub":
        warnings.append(
            "No separate encoded-feature-count artifact exists; the snapshot reports engineered candidate features from experiment summaries and source-table width from the processed application table."
        )
    analysis_files = [
        "feature_level_drift/feature_level_psi_by_run.csv",
        "feature_level_drift/psi_distribution_by_pipeline.csv",
        "feature_level_drift/high_psi_features_by_pipeline.csv",
        "feature_level_drift/llm_then_mrmr_drift_source_breakdown.csv",
        "feature_level_drift/llm_top100_candidate_psi.csv",
        "semantic_redundancy/semantic_coverage_redundancy_by_pipeline.csv",
    ]
    for relative_path in analysis_files:
        path = paths.results_dir / "analysis" / Path(relative_path)
        if not path.exists():
            warnings.append(f"Missing analysis artifact: {path.relative_to(paths.root)}")
    return warnings


def _selector_family(selector: str) -> str:
    if selector in BASELINE_SELECTORS:
        return "baseline"
    if selector in LLM_FAMILY_SELECTORS:
        return "llm_family"
    return "other"


def _model_order(value: str) -> int:
    return {"lr": 0, "catboost": 1}.get(str(value), 99)


@lru_cache(maxsize=None)
def _homecredit_minimal_time_frame() -> pd.DataFrame:
    root = project_root()
    app_path = root / "data" / "homecredit" / "raw" / "application_train.csv"
    prev_path = root / "data" / "homecredit" / "raw" / "previous_application.csv"
    if not app_path.exists() or not prev_path.exists():
        return pd.DataFrame()

    app_df = pd.read_csv(app_path, usecols=["SK_ID_CURR", "TARGET"])
    prev_df = pd.read_csv(prev_path, usecols=["SK_ID_CURR", "DAYS_DECISION"])
    prev_recent = (
        prev_df.dropna(subset=["DAYS_DECISION"])
        .groupby("SK_ID_CURR", as_index=False)["DAYS_DECISION"]
        .max()
        .rename(columns={"DAYS_DECISION": "recent_decision"})
    )
    merged = app_df.merge(prev_recent, on="SK_ID_CURR", how="left")
    merged = merged[merged["recent_decision"].notna()].copy()
    return merged


@lru_cache(maxsize=None)
def _lendingclub_minimal_time_frame() -> pd.DataFrame:
    root = project_root()
    app_path = root / "data" / "lendingclub" / "processed" / "application_train.csv"
    if not app_path.exists():
        return pd.DataFrame()

    header = list(pd.read_csv(app_path, nrows=0).columns)
    wanted = [column for column in ["TARGET", "recent_decision", "issue_d", "loan_status"] if column in header]
    frame = pd.read_csv(app_path, usecols=wanted)
    raw_issue = frame["issue_d"].copy() if "issue_d" in frame.columns else None
    frame = ensure_lendingclub_target_and_time(frame)
    if "issue_d" in frame.columns:
        frame["issue_d"] = pd.to_datetime(frame["issue_d"], errors="coerce")
        if raw_issue is not None and frame["issue_d"].isna().all():
            frame["issue_d"] = pd.to_datetime(raw_issue, errors="coerce")
    return frame


def _split_segment(series: pd.Series, *, dev_start_day: int, oot_start_day: int, oot_end_day: int) -> pd.Series:
    segment = pd.Series("outside", index=series.index, dtype="object")
    segment[(series >= dev_start_day) & (series < oot_start_day)] = "DEV"
    segment[(series >= oot_start_day) & (series <= oot_end_day)] = "OOT"
    return segment


def _format_day_window(start_day: Any, end_day: Any, *, end_inclusive: bool) -> str:
    if pd.isna(start_day) or pd.isna(end_day):
        return "unavailable"
    if end_inclusive:
        return f"[{int(start_day)}, {int(end_day)}]"
    return f"[{int(start_day)}, {int(end_day)})"


def _lendingclub_date_window(frame: pd.DataFrame, segment: str) -> tuple[str | None, str | None]:
    if frame.empty or "issue_d" not in frame.columns:
        return None, None
    subset = frame.loc[frame["split_segment"] == segment, "issue_d"].dropna()
    if subset.empty:
        return None, None
    return subset.min().date().isoformat(), subset.max().date().isoformat()


@lru_cache(maxsize=None)
def _time_frame(dataset_name: str) -> pd.DataFrame:
    dataset_name = _normalize_dataset_name(dataset_name)
    config = _dataset_config(dataset_name)
    if dataset_name == "homecredit":
        frame = _homecredit_minimal_time_frame().copy()
    else:
        frame = _lendingclub_minimal_time_frame().copy()
    if frame.empty:
        return frame
    frame["split_segment"] = _split_segment(
        pd.to_numeric(frame["recent_decision"], errors="coerce"),
        dev_start_day=int(config["dev_start_day"]),
        oot_start_day=int(config["oot_start_day"]),
        oot_end_day=int(config["oot_end_day"]),
    )
    frame = frame[frame["split_segment"].isin(["DEV", "OOT"])].copy()
    return frame


@lru_cache(maxsize=None)
def _time_bucket_summary(dataset_name: str) -> pd.DataFrame:
    dataset_name = _normalize_dataset_name(dataset_name)
    frame = _time_frame(dataset_name)
    if frame.empty:
        return frame

    if dataset_name == "homecredit":
        bucket_start = (frame["recent_decision"] // 30).astype(int) * 30
        frame["bucket_start"] = bucket_start
        frame["bucket_end"] = frame["bucket_start"] + 29
        frame["bucket_label"] = frame.apply(
            lambda row: f"{int(row['bucket_start'])} to {int(row['bucket_end'])}",
            axis=1,
        )
        grouped = (
            frame.groupby(["bucket_start", "bucket_label", "split_segment"], as_index=False)
            .agg(
                observation_count=("TARGET", "size"),
                bad_rate=("TARGET", "mean"),
            )
            .sort_values("bucket_start")
        )
        grouped["time_bucket"] = grouped["bucket_label"]
        return grouped[["time_bucket", "bucket_start", "split_segment", "observation_count", "bad_rate"]]

    frame["issue_month"] = pd.to_datetime(frame["issue_d"], errors="coerce").dt.to_period("M")
    grouped = (
        frame.groupby(["issue_month", "split_segment"], as_index=False)
        .agg(
            observation_count=("TARGET", "size"),
            bad_rate=("TARGET", "mean"),
            bucket_start=("recent_decision", "min"),
        )
        .sort_values("issue_month")
    )
    grouped["time_bucket"] = grouped["issue_month"].astype(str)
    return grouped[["time_bucket", "bucket_start", "split_segment", "observation_count", "bad_rate"]]


def _modeling_feature_count_if_available(dataset_name: str) -> float:
    if dataset_name != "lendingclub":
        return math.nan
    frame = _lendingclub_minimal_time_frame()
    if frame.empty:
        return math.nan
    app_path = project_root() / "data" / "lendingclub" / "processed" / "application_train.csv"
    header = list(pd.read_csv(app_path, nrows=0).columns)
    excluded = set(_dataset_config(dataset_name).get("excluded_feature_columns", []))
    return float(len([column for column in header if column not in excluded]))


def _matrix_overview(dataset_name: str) -> pd.DataFrame:
    final_df = _final_comparison(dataset_name)
    matrix_df = _matrix_runs(dataset_name)
    failed_df = _failed_runs(dataset_name)

    if final_df.empty and matrix_df.empty:
        return pd.DataFrame()

    models = ", ".join(sorted(final_df["model"].dropna().astype(str).unique())) if not final_df.empty else ""
    selectors = ", ".join(
        sorted(
            final_df["selector"].dropna().astype(str).unique(),
            key=lambda item: (
                ["mrmr", "boruta", "pca", "domain_rule_baseline", "llm", "llm_then_mrmr", "llm_then_boruta", "stable_core_llm_fill"].index(item)
                if item in ["mrmr", "boruta", "pca", "domain_rule_baseline", "llm", "llm_then_mrmr", "llm_then_boruta", "stable_core_llm_fill"]
                else 99
            ),
        )
    ) if not final_df.empty else ""
    budgets = ", ".join(
        sorted({str(int(value)) for value in final_df["feature_budget"].dropna().tolist()})
    ) if not final_df.empty else ""

    return pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "models": models,
                "selectors": selectors,
                "feature_budgets": budgets,
                "completed_run_count": int((matrix_df["status"] == "completed").sum()) if not matrix_df.empty else int(len(final_df)),
                "failed_run_count": int(len(failed_df)),
            }
        ]
    )


def _selector_semantic_summary(dataset_name: str) -> pd.DataFrame:
    semantic_df = _semantic_coverage(dataset_name)
    if semantic_df.empty:
        return semantic_df

    summary = (
        semantic_df.groupby(["model", "selector", "experiment_type"], as_index=False)
        .agg(
            semantic_group_count=("semantic_group", "nunique"),
            dominant_group_ratio=("feature_ratio", "max"),
        )
    )
    top_groups = (
        semantic_df.sort_values(["model", "selector", "feature_ratio"], ascending=[True, True, False])
        .groupby(["model", "selector"], as_index=False)
        .first()[["model", "selector", "semantic_group"]]
        .rename(columns={"semantic_group": "dominant_group"})
    )
    summary = summary.merge(top_groups, on=["model", "selector"], how="left")
    return summary.sort_values(["model", "selector"]).reset_index(drop=True)


def load_dataset_report_inputs(dataset_name: str) -> dict[str, Any]:
    dataset_name = _normalize_dataset_name(dataset_name)
    return {
        "dataset_name": dataset_name,
        "dataset_display_name": _display_dataset_name(dataset_name),
        "dataset_role": _dataset_role(dataset_name),
        "config": _dataset_config(dataset_name),
        "warnings": _warnings(dataset_name),
        "snapshot": load_dataset_snapshot(dataset_name),
        "split_summary": load_split_summary(dataset_name),
        "time_bucket_summary": load_time_bucket_summary(dataset_name),
        "final_comparison": load_final_comparison(dataset_name),
        "stability_table": load_stability_table(dataset_name),
        "drift_table": load_drift_table(dataset_name),
        "semantic_coverage_table": load_semantic_coverage_table(dataset_name),
        "feature_level_psi_by_run": load_feature_level_psi_by_run(dataset_name),
        "psi_distribution_by_pipeline": load_psi_distribution_by_pipeline(dataset_name),
        "high_psi_features_by_pipeline": load_high_psi_features_by_pipeline(dataset_name),
        "llm_then_mrmr_drift_source_breakdown": load_llm_then_mrmr_drift_source_breakdown(dataset_name),
        "llm_top100_candidate_psi": load_llm_top100_candidate_psi(dataset_name),
        "semantic_redundancy_table": load_semantic_redundancy_table(dataset_name),
        "llm_stability_diagnosis": load_llm_stability_diagnosis(dataset_name),
        "paired_fold_significance_tests": load_paired_fold_significance_tests(dataset_name),
        "governance_note": load_governance_note(dataset_name),
        "semantic_summary": _round_numeric_frame(_selector_semantic_summary(dataset_name)),
        "best_runs": load_best_runs(dataset_name),
        "matrix_overview": _matrix_overview(dataset_name),
        "paired_fold_comparisons": _round_numeric_frame(_paired_fold(dataset_name)),
        "llm_call_summary": load_compact_llm_cache_summary(dataset_name),
        "failed_runs": _failed_runs(dataset_name).copy(),
    }


def load_dataset_snapshot(dataset_name: str) -> pd.DataFrame:
    dataset_name = _normalize_dataset_name(dataset_name)
    config = _dataset_config(dataset_name)
    split_df = load_split_summary(dataset_name)
    final_df = _final_comparison(dataset_name)
    matrix_df = _matrix_runs(dataset_name)
    failed_df = _failed_runs(dataset_name)

    if split_df.empty:
        return pd.DataFrame()

    split_row = split_df.iloc[0]
    candidate_count = float(final_df["total_candidate_feature_count"].mode().iloc[0]) if not final_df.empty else math.nan
    snapshot = pd.DataFrame(
        [
            {
                "dataset_name": _display_dataset_name(dataset_name),
                "dataset_role": _dataset_role(dataset_name),
                "DEV_rows": split_row["DEV_rows"],
                "OOT_rows": split_row["OOT_rows"],
                "DEV_bad_rate": split_row["DEV_bad_rate"],
                "OOT_bad_rate": split_row["OOT_bad_rate"],
                "time_column": split_row["time_column"],
                "DEV_window": split_row["DEV_window"],
                "OOT_window": split_row["OOT_window"],
                "engineered_candidate_features": candidate_count,
                "encoded_or_modeling_features_if_available": _modeling_feature_count_if_available(dataset_name),
                "LR_feature_budget": float(config["feature_budgets"]["lr"]),
                "CatBoost_feature_budget": float(config["feature_budgets"]["catboost"]),
                "completed_runs": int((matrix_df["status"] == "completed").sum()) if not matrix_df.empty else int(len(final_df)),
                "failed_runs": int(len(failed_df)),
            }
        ]
    )
    return _round_numeric_frame(snapshot)


def load_split_summary(dataset_name: str) -> pd.DataFrame:
    dataset_name = _normalize_dataset_name(dataset_name)
    manifest_path = _first_existing_split_manifest(dataset_name)
    if manifest_path is None:
        return pd.DataFrame()

    payload = _read_json(manifest_path)
    config = _dataset_config(dataset_name)
    frame = _time_frame(dataset_name)

    dev_start_date = dev_end_date = oot_start_date = oot_end_date = None
    if dataset_name == "lendingclub" and not frame.empty:
        dev_start_date, dev_end_date = _lendingclub_date_window(frame, "DEV")
        oot_start_date, oot_end_date = _lendingclub_date_window(frame, "OOT")

    dev_rate = float(payload["dev"]["target_rate"])
    oot_rate = float(payload["oot"]["target_rate"])
    row = {
        "dataset": dataset_name,
        "dataset_display_name": _display_dataset_name(dataset_name),
        "time_column": payload.get("time_column", config.get("time_col", "recent_decision")),
        "DEV_start": payload["DEV_window"]["start_day_inclusive"],
        "DEV_end": payload["DEV_window"]["end_day_exclusive"],
        "OOT_start": payload["OOT_window"]["start_day_inclusive"],
        "OOT_end": payload["OOT_window"]["end_day_inclusive"],
        "DEV_window": _format_day_window(
            payload["DEV_window"]["start_day_inclusive"],
            payload["DEV_window"]["end_day_exclusive"],
            end_inclusive=False,
        ),
        "OOT_window": _format_day_window(
            payload["OOT_window"]["start_day_inclusive"],
            payload["OOT_window"]["end_day_inclusive"],
            end_inclusive=True,
        ),
        "DEV_rows": int(payload["dev"]["row_count"]),
        "OOT_rows": int(payload["oot"]["row_count"]),
        "DEV_bad_rate": dev_rate,
        "OOT_bad_rate": oot_rate,
        "bad_rate_difference": _safe_rate_delta(dev_rate, oot_rate),
        "OOT_DEV_row_ratio": float(payload["oot"]["row_count"] / payload["dev"]["row_count"]),
        "dropped_older_rows": int(payload.get("dropped_older_row_count", 0)),
        "dropped_missing_time_rows": int(payload.get("dropped_missing_time_row_count", 0)),
        "source_row_count": int(payload.get("source_row_count", 0)),
        "DEV_issue_date_start": dev_start_date,
        "DEV_issue_date_end": dev_end_date,
        "OOT_issue_date_start": oot_start_date,
        "OOT_issue_date_end": oot_end_date,
    }
    return _round_numeric_frame(pd.DataFrame([row]))


def load_time_bucket_summary(dataset_name: str) -> pd.DataFrame:
    return _round_numeric_frame(_time_bucket_summary(dataset_name).copy())


def load_final_comparison(dataset_name: str) -> pd.DataFrame:
    final_df = _final_comparison(dataset_name).copy()
    if final_df.empty:
        return final_df
    ordered = final_df.sort_values(
        ["model", "oot_auc", "oot_gini", "selector"],
        ascending=[True, False, False, True],
        key=lambda column: column.map(_model_order) if column.name == "model" else column,
    ).reset_index(drop=True)
    return _round_numeric_frame(ordered)


def load_stability_table(dataset_name: str) -> pd.DataFrame:
    stability_df = _stability(dataset_name).copy()
    if stability_df.empty:
        return stability_df
    stability_df = stability_df.sort_values(["model", "nogueira_stability"], ascending=[True, False])
    return _round_numeric_frame(stability_df.reset_index(drop=True))


def load_drift_table(dataset_name: str) -> pd.DataFrame:
    drift_df = _drift(dataset_name).copy()
    if drift_df.empty:
        return drift_df
    drift_df = drift_df.sort_values(["model", "selected_feature_psi_mean"], ascending=[True, True])
    return _round_numeric_frame(drift_df.reset_index(drop=True))


def load_semantic_coverage_table(dataset_name: str) -> pd.DataFrame:
    semantic_df = _semantic_coverage(dataset_name).copy()
    if semantic_df.empty:
        return semantic_df
    semantic_df = semantic_df.sort_values(["model", "selector", "feature_ratio"], ascending=[True, True, False])
    return _round_numeric_frame(semantic_df.reset_index(drop=True))


def load_feature_level_psi_by_run(dataset_name: str) -> pd.DataFrame:
    df = _feature_level_psi(dataset_name).copy()
    if df.empty:
        return df
    return _round_numeric_frame(df.sort_values(["model", "selector", "psi_dev_oot"], ascending=[True, True, False]))


def load_psi_distribution_by_pipeline(dataset_name: str) -> pd.DataFrame:
    df = _psi_distribution(dataset_name).copy()
    if df.empty:
        return df
    return _round_numeric_frame(df.sort_values(["model", "psi_mean", "selector"]))


def load_high_psi_features_by_pipeline(dataset_name: str) -> pd.DataFrame:
    df = _high_psi_features(dataset_name).copy()
    if df.empty:
        return df
    return _round_numeric_frame(df.sort_values(["model", "selector", "psi_dev_oot"], ascending=[True, True, False]))


def load_llm_then_mrmr_drift_source_breakdown(dataset_name: str) -> pd.DataFrame:
    df = _llm_then_mrmr_drift_source(dataset_name).copy()
    if df.empty:
        return df
    return _round_numeric_frame(df.sort_values(["model", "in_final_selected_set", "psi_dev_oot"], ascending=[True, False, False]))


def load_llm_top100_candidate_psi(dataset_name: str) -> pd.DataFrame:
    df = _llm_top100_candidate_psi(dataset_name).copy()
    if df.empty:
        return df
    sort_cols = [col for col in ["model", "selector", "llm_rank", "feature"] if col in df.columns]
    return _round_numeric_frame(df.sort_values(sort_cols).reset_index(drop=True))


def load_semantic_redundancy_table(dataset_name: str) -> pd.DataFrame:
    df = _semantic_redundancy(dataset_name).copy()
    if df.empty:
        return df
    return _round_numeric_frame(df.sort_values(["model", "selector"]))


def load_llm_stability_diagnosis(dataset_name: str | None = None) -> pd.DataFrame:
    df = _cross_dataset_stability_significance().copy()
    if dataset_name is not None and not df.empty:
        df = df[df["dataset"].eq(_normalize_dataset_name(dataset_name))].copy()
    return _round_numeric_frame(df)


def load_paired_fold_significance_tests(dataset_name: str | None = None) -> pd.DataFrame:
    df = _paired_fold_significance().copy()
    if dataset_name is not None and not df.empty:
        df = df[df["dataset"].eq(_normalize_dataset_name(dataset_name))].copy()
    return _round_numeric_frame(df)


def load_governance_note(dataset_name: str) -> str:
    dataset_name = _normalize_dataset_name(dataset_name)
    if dataset_name == "homecredit":
        return _report_text("reports/homecredit_temporal_semantics_note.md")
    return _report_text("reports/lendingclub_leakage_and_label_definition.md")


def load_compact_llm_cache_summary(dataset_name: str) -> pd.DataFrame:
    dataset_name = _normalize_dataset_name(dataset_name)
    df = _llm_calls(dataset_name).copy()
    columns = [
        "dataset",
        "LLM calls made",
        "cache hits",
        "total tokens",
        "prompt version",
        "shared ranking enabled",
        "number of runs sharing ranking",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    def _first_nonempty(series: pd.Series) -> str:
        values = [str(value) for value in series.dropna().tolist() if str(value).strip()]
        return values[0] if values else ""

    def _max_shared_runs(series: pd.Series) -> int:
        counts = []
        for value in series.dropna().astype(str):
            parts = [part for part in value.split(";") if part.strip()]
            if parts:
                counts.append(len(set(parts)))
        return max(counts) if counts else 0

    summary = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "LLM calls made": pd.to_numeric(df.get("llm_calls_actually_made", pd.Series(dtype=float)), errors="coerce").sum(),
                "cache hits": pd.to_numeric(df.get("llm_cache_hits", pd.Series(dtype=float)), errors="coerce").sum(),
                "total tokens": pd.to_numeric(df.get("llm_total_tokens", pd.Series(dtype=float)), errors="coerce").sum(),
                "prompt version": _first_nonempty(df.get("llm_prompt_versions", pd.Series(dtype=object))),
                "shared ranking enabled": bool(df.get("llm_shared_ranking_enabled", pd.Series(dtype=bool)).fillna(False).astype(bool).any()),
                "number of runs sharing ranking": _max_shared_runs(
                    df.get("runs_sharing_metadata_signatures", pd.Series(dtype=object))
                ),
            }
        ]
    )
    numeric_cols = ["LLM calls made", "cache hits", "total tokens", "number of runs sharing ranking"]
    summary[numeric_cols] = summary[numeric_cols].fillna(0).astype(int)
    return summary[columns]


def save_full_llm_cache_appendix(dataset_name: str) -> Path | None:
    dataset_name = _normalize_dataset_name(dataset_name)
    df = _llm_calls(dataset_name).copy()
    if df.empty:
        return None
    appendix_dir = _dataset_paths(dataset_name).results_dir / "final_report" / "appendix"
    appendix_dir.mkdir(parents=True, exist_ok=True)
    if "output_folder" in df.columns:
        df["output_folder"] = df["output_folder"].map(_repo_relative_path)
    path = appendix_dir / "full_llm_cache_summary.csv"
    df.to_csv(path, index=False)
    return path


def _top_items_as_text(series: pd.Series, *, limit: int = 5) -> str:
    values = [str(item) for item in series.dropna().tolist()[:limit]]
    return ", ".join(values) if values else "unavailable"


def _top_semantic_groups(selected_df: pd.DataFrame, *, limit: int = 3) -> str:
    if selected_df.empty or "semantic_group" not in selected_df.columns:
        return "unavailable"
    summary = (
        selected_df.groupby("semantic_group", dropna=False)["feature_name"]
        .count()
        .sort_values(ascending=False)
    )
    parts = [f"{group} ({count})" for group, count in summary.head(limit).items()]
    return ", ".join(parts) if parts else "unavailable"


def _frame_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    display_df = df.fillna("")
    columns = list(display_df.columns)
    rows = [[str(value) for value in row] for row in display_df.astype(object).values.tolist()]
    widths = []
    for idx, column in enumerate(columns):
        widths.append(max(len(str(column)), *(len(row[idx]) for row in rows)))

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    header = render_row([str(column) for column in columns])
    divider = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [render_row(row) for row in rows]
    return "\n".join([header, divider, *body])


def _best_row(df: pd.DataFrame, *, model: str | None = None, selector_set: set[str] | None = None) -> pd.Series | None:
    subset = df.copy()
    if model is not None:
        subset = subset[subset["model"] == model]
    if selector_set is not None:
        subset = subset[subset["selector"].isin(selector_set)]
    if subset.empty:
        return None
    subset = subset.sort_values(["oot_auc", "oot_gini", "nogueira_stability"], ascending=[False, False, False])
    return subset.iloc[0]


def load_best_runs(dataset_name: str) -> pd.DataFrame:
    dataset_name = _normalize_dataset_name(dataset_name)
    final_df = _final_comparison(dataset_name)
    if final_df.empty:
        return final_df

    picks: list[tuple[str, pd.Series | None]] = [
        ("best LR run", _best_row(final_df, model="lr")),
        ("best CatBoost run", _best_row(final_df, model="catboost")),
        ("strongest non-LLM baseline", _best_row(final_df, selector_set=BASELINE_SELECTORS)),
        ("best LLM/hybrid run", _best_row(final_df, selector_set=LLM_FAMILY_SELECTORS)),
    ]

    rows: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for label, row in picks:
        if row is None:
            continue
        run_id = str(row["run_id"])
        if run_id in seen_run_ids:
            continue
        seen_run_ids.add(run_id)
        artifacts = load_run_artifacts(dataset_name, run_id)
        summary_df = artifacts["summary_table"]
        cv_df = artifacts["cv_results"]
        selected_df = artifacts["selected_features"]
        summary_row = summary_df.iloc[0] if not summary_df.empty else {}
        rows.append(
            {
                "analysis_label": label,
                "run_id": run_id,
                "model": row["model"],
                "selector": row["selector"],
                "experiment_type": row["experiment_type"],
                "OOT_AUC": row["oot_auc"],
                "OOT_Gini": row["oot_gini"],
                "OOT_KS": row["oot_ks"],
                "CV_AUC_mean": summary_row.get("cv_auc_mean", math.nan) if isinstance(summary_row, pd.Series) else math.nan,
                "CV_AUC_std": summary_row.get("cv_auc_std", math.nan) if isinstance(summary_row, pd.Series) else math.nan,
                "selected_feature_count": row["selected_feature_count"],
                "top_features": _top_items_as_text(selected_df.get("feature_name", pd.Series(dtype="object"))),
                "top_semantic_groups": _top_semantic_groups(selected_df),
                "fold_behavior": (
                    f"CV AUC {summary_row.get('cv_auc_mean', math.nan):.4f} +/- {summary_row.get('cv_auc_std', math.nan):.4f}"
                    if isinstance(summary_row, pd.Series) and not pd.isna(summary_row.get("cv_auc_mean", math.nan))
                    else "CV summary unavailable"
                ),
                "why_it_matters": (
                    "highest OOT leaderboard entry for this slice"
                    if label.startswith("best")
                    else "reference baseline for non-LLM comparison"
                ),
            }
        )
    return _round_numeric_frame(pd.DataFrame(rows))


def load_run_artifacts(dataset_name: str, run_id: str) -> dict[str, Any]:
    dataset_name = _normalize_dataset_name(dataset_name)
    final_df = _final_comparison(dataset_name)
    match = final_df.loc[final_df["run_id"].astype(str) == str(run_id)]
    run_dir = None
    if not match.empty:
        run_dir = Path(str(match.iloc[0]["output_folder"]))
    else:
        manifest_matches = list(_dataset_paths(dataset_name).results_dir.rglob(f"{run_id}/run_manifest.json"))
        if manifest_matches:
            run_dir = manifest_matches[0].parent
    if run_dir is None or not run_dir.exists():
        return {
            "run_id": run_id,
            "run_dir": None,
            "manifest": {},
            "summary_table": pd.DataFrame(),
            "cv_results": pd.DataFrame(),
            "oot_results": pd.DataFrame(),
            "selected_features": pd.DataFrame(),
            "semantic_group_summary": pd.DataFrame(),
            "warnings": [f"Run artifacts not found for {run_id}"],
        }

    manifest = _read_json(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else {}
    summary_table = _read_csv(run_dir / "results" / "experiment_summary.csv")
    cv_results = _read_csv(run_dir / "results" / "cv_results.csv")
    oot_results = _read_csv(run_dir / "results" / "oot_test_results.csv")
    selected_features = _read_csv(run_dir / "features" / "final_selected_features.csv")
    if dataset_name == "lendingclub" and not selected_features.empty:
        feature_col = "feature_name" if "feature_name" in selected_features.columns else "feature"
        if feature_col in selected_features.columns:
            selected_features = selected_features.copy()
            selected_features["semantic_group"] = selected_features[feature_col].map(lambda feature: infer_semantic_group(str(feature)))
    semantic_group_summary = pd.DataFrame()
    if not selected_features.empty and "semantic_group" in selected_features.columns:
        semantic_group_summary = (
            selected_features.groupby("semantic_group", dropna=False)["feature_name"]
            .count()
            .reset_index(name="feature_count")
            .sort_values("feature_count", ascending=False)
        )
        total = float(semantic_group_summary["feature_count"].sum())
        semantic_group_summary["feature_ratio"] = semantic_group_summary["feature_count"] / total if total else math.nan

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "manifest": manifest,
        "summary_table": _round_numeric_frame(summary_table),
        "cv_results": _round_numeric_frame(cv_results),
        "oot_results": _round_numeric_frame(oot_results),
        "selected_features": _round_numeric_frame(selected_features),
        "semantic_group_summary": _round_numeric_frame(semantic_group_summary),
        "warnings": [],
    }


def _plot_split_background(ax: plt.Axes, plot_df: pd.DataFrame) -> None:
    dev_idx = plot_df.index[plot_df["split_segment"] == "DEV"].tolist()
    oot_idx = plot_df.index[plot_df["split_segment"] == "OOT"].tolist()
    if dev_idx:
        ax.axvspan(min(dev_idx) - 0.5, max(dev_idx) + 0.5, color="#dbeafe", alpha=0.35)
    if oot_idx:
        ax.axvspan(min(oot_idx) - 0.5, max(oot_idx) + 0.5, color="#fde68a", alpha=0.30)
        ax.axvline(min(oot_idx) - 0.5, color="#b45309", linestyle="--", linewidth=1.2)


def _base_theme() -> None:
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")


def plot_observation_count_by_time(dataset_name: str) -> plt.Figure:
    import matplotlib.pyplot as plt

    _base_theme()
    plot_df = _time_bucket_summary(dataset_name).copy()
    fig, ax = plt.subplots(figsize=(12, 4))
    if plot_df.empty:
        ax.text(0.5, 0.5, "Time-bucket summary unavailable.", ha="center", va="center")
        ax.axis("off")
        return fig

    ax.bar(range(len(plot_df)), plot_df["observation_count"], color="#2563eb")
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["time_bucket"], rotation=45, ha="right")
    ax.set_ylabel("Observation Count")
    ax.set_title(f"{_display_dataset_name(_normalize_dataset_name(dataset_name))}: observations by time bucket")
    _plot_split_background(ax, plot_df)
    fig.tight_layout()
    return fig


def plot_bad_rate_by_time(dataset_name: str) -> plt.Figure:
    import matplotlib.pyplot as plt

    _base_theme()
    plot_df = _time_bucket_summary(dataset_name).copy()
    fig, ax = plt.subplots(figsize=(12, 4))
    if plot_df.empty:
        ax.text(0.5, 0.5, "Time-bucket summary unavailable.", ha="center", va="center")
        ax.axis("off")
        return fig

    ax.plot(range(len(plot_df)), plot_df["bad_rate"], marker="o", color="#dc2626", linewidth=2)
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["time_bucket"], rotation=45, ha="right")
    ax.set_ylabel("Bad Rate")
    ax.set_title(f"{_display_dataset_name(_normalize_dataset_name(dataset_name))}: bad rate by time bucket")
    _plot_split_background(ax, plot_df)
    fig.tight_layout()
    return fig


def plot_dev_oot_split_diagnostics(dataset_name: str) -> plt.Figure:
    import matplotlib.pyplot as plt

    _base_theme()
    plot_df = _time_bucket_summary(dataset_name).copy()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    if plot_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "Time-bucket summary unavailable.", ha="center", va="center")
            ax.axis("off")
        return fig

    axes[0].bar(range(len(plot_df)), plot_df["observation_count"], color="#2563eb")
    axes[0].set_ylabel("Observation Count")
    axes[0].set_title(f"{_display_dataset_name(_normalize_dataset_name(dataset_name))}: DEV/OOT split diagnostics")
    _plot_split_background(axes[0], plot_df)

    axes[1].plot(range(len(plot_df)), plot_df["bad_rate"], marker="o", color="#dc2626", linewidth=2)
    axes[1].set_ylabel("Bad Rate")
    axes[1].set_xticks(range(len(plot_df)))
    axes[1].set_xticklabels(plot_df["time_bucket"], rotation=45, ha="right")
    _plot_split_background(axes[1], plot_df)
    fig.tight_layout()
    return fig


def plot_metric_leaderboard(
    final_df: pd.DataFrame,
    *,
    metric: str = "oot_auc",
    secondary_metric: str = "oot_gini",
) -> plt.Figure:
    import matplotlib.pyplot as plt

    _base_theme()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False)
    if final_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "Comparison table unavailable.", ha="center", va="center")
            ax.axis("off")
        return fig

    for ax, model_name in zip(axes, ["lr", "catboost"], strict=False):
        subset = final_df.loc[final_df["model"] == model_name].sort_values(metric, ascending=True)
        if subset.empty:
            ax.text(0.5, 0.5, f"No rows for {model_name}.", ha="center", va="center")
            ax.axis("off")
            continue
        labels = subset["selector"] + " (" + subset["experiment_type"] + ")"
        ax.barh(labels, subset[metric], color="#0f766e")
        ax.set_title(model_name.upper())
        ax.set_xlabel(metric.replace("_", " ").upper())
        for idx, (_, row) in enumerate(subset.iterrows()):
            ax.text(
                row[metric],
                idx,
                f" {row[secondary_metric]:.4f} {secondary_metric.replace('_', ' ')}",
                va="center",
                fontsize=9,
            )
    fig.suptitle("OOT metric leaderboard by model and selector")
    fig.tight_layout()
    return fig


def plot_stability_vs_performance(
    stability_df: pd.DataFrame,
    final_df: pd.DataFrame,
    *,
    stability_col: str = "nogueira_stability",
    performance_col: str = "oot_auc",
) -> plt.Figure:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _base_theme()
    merged = final_df.copy()
    required_cols = [stability_col, "semantic_group_jaccard"]
    if not set(required_cols).issubset(merged.columns):
        supplemental = stability_df[["model", "selector", "experiment_type", *required_cols]].copy()
        merged = merged.merge(supplemental, on=["model", "selector", "experiment_type"], how="left")
    fig, ax = plt.subplots(figsize=(8, 6))
    if merged.empty:
        ax.text(0.5, 0.5, "Stability comparison unavailable.", ha="center", va="center")
        ax.axis("off")
        return fig

    sns.scatterplot(
        data=merged,
        x=stability_col,
        y=performance_col,
        hue="model",
        style="experiment_type",
        s=120,
        ax=ax,
    )
    for _, row in merged.iterrows():
        ax.text(row[stability_col] + 0.002, row[performance_col] + 0.0005, row["selector"], fontsize=8)
    ax.set_title("Selection stability vs OOT performance")
    fig.tight_layout()
    return fig


def plot_drift_vs_performance(
    drift_df: pd.DataFrame,
    final_df: pd.DataFrame,
    *,
    drift_col: str = "selected_feature_psi_mean",
    performance_col: str = "oot_auc",
) -> plt.Figure:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _base_theme()
    merged = final_df.copy()
    required_cols = [drift_col, "model_score_psi"]
    if not set(required_cols).issubset(merged.columns):
        supplemental = drift_df[["model", "selector", "experiment_type", *required_cols]].copy()
        merged = merged.merge(supplemental, on=["model", "selector", "experiment_type"], how="left")
    fig, ax = plt.subplots(figsize=(8, 6))
    if merged.empty:
        ax.text(0.5, 0.5, "Drift comparison unavailable.", ha="center", va="center")
        ax.axis("off")
        return fig

    sns.scatterplot(
        data=merged,
        x=drift_col,
        y=performance_col,
        hue="model",
        style="experiment_type",
        size="model_score_psi",
        sizes=(60, 220),
        ax=ax,
    )
    for _, row in merged.iterrows():
        ax.text(row[drift_col] + 0.001, row[performance_col] + 0.0005, row["selector"], fontsize=8)
    ax.set_title("Drift vs OOT performance")
    fig.tight_layout()
    return fig


def plot_semantic_coverage(semantic_df: pd.DataFrame) -> plt.Figure:
    import matplotlib.pyplot as plt

    _base_theme()
    fig, ax = plt.subplots(figsize=(14, 6))
    if semantic_df.empty:
        ax.text(0.5, 0.5, "Semantic coverage table unavailable.", ha="center", va="center")
        ax.axis("off")
        return fig

    plot_df = (
        semantic_df.assign(model_selector=lambda df: df["model"] + ":" + df["selector"])
        .pivot_table(
            index="model_selector",
            columns="semantic_group",
            values="feature_ratio",
            aggfunc="mean",
            fill_value=0.0,
        )
    )
    if plot_df.shape[1] > 8:
        top_groups = plot_df.sum(axis=0).sort_values(ascending=False).head(8).index
        plot_df = plot_df.loc[:, top_groups]
    plot_df.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_ylabel("Mean Feature Ratio")
    ax.set_title("Semantic coverage by selector")
    ax.legend(title="Semantic Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_runtime_tradeoff(
    final_df: pd.DataFrame,
    *,
    performance_col: str = "oot_auc",
) -> plt.Figure:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _base_theme()
    fig, ax = plt.subplots(figsize=(8, 6))
    if final_df.empty:
        ax.text(0.5, 0.5, "Runtime tradeoff table unavailable.", ha="center", va="center")
        ax.axis("off")
        return fig

    sns.scatterplot(
        data=final_df,
        x="runtime_seconds",
        y=performance_col,
        hue="model",
        style="selector",
        size="selected_feature_count",
        sizes=(80, 240),
        ax=ax,
    )
    for _, row in final_df.iterrows():
        ax.text(row["runtime_seconds"] + 10, row[performance_col] + 0.0005, row["selector"], fontsize=8)
    ax.set_title("Runtime vs OOT performance")
    ax.set_xlabel("Runtime Seconds")
    fig.tight_layout()
    return fig


def lendingclub_monthly_bad_rate_observation_count_table() -> pd.DataFrame:
    frame = _time_frame("lendingclub").copy()
    if frame.empty or "issue_d" not in frame.columns:
        return pd.DataFrame(columns=["issue_month", "split_segment", "observation_count", "bad_rate"])
    frame["issue_month"] = pd.to_datetime(frame["issue_d"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    grouped = (
        frame.dropna(subset=["issue_month"])
        .groupby(["issue_month", "split_segment"], as_index=False)
        .agg(observation_count=("TARGET", "size"), bad_rate=("TARGET", "mean"))
        .sort_values("issue_month")
    )
    grouped["issue_month"] = grouped["issue_month"].dt.strftime("%Y-%m")
    return _round_numeric_frame(grouped[["issue_month", "split_segment", "observation_count", "bad_rate"]])


def plot_lendingclub_monthly_bad_rate_observation_count(monthly_df: pd.DataFrame | None = None) -> plt.Figure:
    import matplotlib.pyplot as plt

    _base_theme()
    monthly_df = lendingclub_monthly_bad_rate_observation_count_table() if monthly_df is None else monthly_df.copy()
    fig, ax_count = plt.subplots(figsize=(12, 5))
    if monthly_df.empty:
        ax_count.text(0.5, 0.5, "LendingClub monthly split diagnostics unavailable.", ha="center", va="center")
        ax_count.axis("off")
        return fig

    plot_df = monthly_df.sort_values("issue_month").reset_index(drop=True)
    x_values = list(range(len(plot_df)))
    colors = plot_df["split_segment"].map({"DEV": "#88b7d5", "OOT": "#f2c078"}).fillna("#d0d0d0")
    ax_count.bar(x_values, plot_df["observation_count"], color=colors, alpha=0.75, label="Observation count")
    ax_count.set_ylabel("Observation count")
    ax_count.set_xlabel("Issue month")

    ax_rate = ax_count.twinx()
    ax_rate.plot(x_values, plot_df["bad_rate"], color="#2f2f2f", marker="o", linewidth=1.8, label="Bad rate")
    ax_rate.set_ylabel("Bad rate")

    for segment, color in [("DEV", "#88b7d5"), ("OOT", "#f2c078")]:
        indexes = [idx for idx, value in enumerate(plot_df["split_segment"].tolist()) if value == segment]
        if indexes:
            ax_count.axvspan(min(indexes) - 0.5, max(indexes) + 0.5, color=color, alpha=0.12, label=f"{segment} region")

    tick_step = max(1, len(plot_df) // 12)
    tick_positions = list(range(0, len(plot_df), tick_step))
    ax_count.set_xticks(tick_positions)
    ax_count.set_xticklabels(plot_df.loc[tick_positions, "issue_month"], rotation=45, ha="right")
    ax_count.set_title("LendingClub Monthly Bad Rate and Observation Count by Split")
    handles_count, labels_count = ax_count.get_legend_handles_labels()
    handles_rate, labels_rate = ax_rate.get_legend_handles_labels()
    ax_count.legend(handles_count + handles_rate, labels_count + labels_rate, loc="upper left")
    fig.tight_layout()
    return fig


def _final_report_plots_dir(dataset_name: str) -> Path:
    path = _dataset_paths(dataset_name).results_dir / "final_report" / "plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plot_manifest_row(
    *,
    plot_file: str,
    source_table: str,
    rows_used: int,
    columns_used: list[str],
    purpose: str,
    status: str,
    skip_reason: str = "",
) -> dict[str, Any]:
    return {
        "plot_file": plot_file,
        "source_table": source_table,
        "rows_used": rows_used,
        "columns_used": ";".join(columns_used),
        "purpose": purpose,
        "status": status,
        "skip_reason": skip_reason,
    }


def _save_if_informative(
    *,
    dataset_name: str,
    plot_file: str,
    source_table: str,
    source_df: pd.DataFrame,
    columns_used: list[str],
    purpose: str,
    figure_factory: Any,
    category_columns: list[str] | None = None,
    value_columns: list[str] | None = None,
) -> dict[str, Any]:
    missing_columns = [column for column in columns_used if column not in source_df.columns]
    if source_df.empty:
        return _plot_manifest_row(
            plot_file=plot_file,
            source_table=source_table,
            rows_used=0,
            columns_used=columns_used,
            purpose=purpose,
            status="skipped",
            skip_reason="empty source data",
        )
    if missing_columns:
        return _plot_manifest_row(
            plot_file=plot_file,
            source_table=source_table,
            rows_used=len(source_df),
            columns_used=columns_used,
            purpose=purpose,
            status="skipped",
            skip_reason=f"missing columns: {', '.join(missing_columns)}",
        )
    for column in category_columns or []:
        if source_df[column].nunique(dropna=True) <= 1:
            return _plot_manifest_row(
                plot_file=plot_file,
                source_table=source_table,
                rows_used=len(source_df),
                columns_used=columns_used,
                purpose=purpose,
                status="skipped",
                skip_reason=f"only one category in {column}",
            )
    for column in value_columns or []:
        if source_df[column].nunique(dropna=True) <= 1:
            return _plot_manifest_row(
                plot_file=plot_file,
                source_table=source_table,
                rows_used=len(source_df),
                columns_used=columns_used,
                purpose=purpose,
                status="skipped",
                skip_reason=f"constant or unavailable values in {column}",
            )

    fig = figure_factory()
    fig.savefig(_final_report_plots_dir(dataset_name) / plot_file, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return _plot_manifest_row(
        plot_file=plot_file,
        source_table=source_table,
        rows_used=len(source_df),
        columns_used=columns_used,
        purpose=purpose,
        status="created",
    )


def save_final_report_plots(dataset_name: str) -> pd.DataFrame:
    """Save final-report plots and a plot manifest under results/<dataset>/final_report/plots."""
    dataset_name = _normalize_dataset_name(dataset_name)
    final_df = load_final_comparison(dataset_name)
    stability_df = load_stability_table(dataset_name)
    drift_df = load_drift_table(dataset_name)
    semantic_df = load_semantic_coverage_table(dataset_name)
    time_df = load_time_bucket_summary(dataset_name)

    rows = [
        _save_if_informative(
            dataset_name=dataset_name,
            plot_file="dev_oot_split_diagnostics.png",
            source_table="derived time-bucket summary",
            source_df=time_df,
            columns_used=["time_bucket", "split_segment", "observation_count", "bad_rate"],
            purpose="Show DEV/OOT observation counts and bad-rate behavior by time bucket.",
            figure_factory=lambda: plot_dev_oot_split_diagnostics(dataset_name),
            category_columns=["split_segment"],
            value_columns=["observation_count", "bad_rate"],
        ),
        _save_if_informative(
            dataset_name=dataset_name,
            plot_file="observation_count_by_time.png",
            source_table="derived time-bucket summary",
            source_df=time_df,
            columns_used=["time_bucket", "observation_count", "split_segment"],
            purpose="Show observations available by time bucket.",
            figure_factory=lambda: plot_observation_count_by_time(dataset_name),
            value_columns=["observation_count"],
        ),
        _save_if_informative(
            dataset_name=dataset_name,
            plot_file="bad_rate_by_time.png",
            source_table="derived time-bucket summary",
            source_df=time_df,
            columns_used=["time_bucket", "bad_rate", "split_segment"],
            purpose="Show target-rate behavior by time bucket.",
            figure_factory=lambda: plot_bad_rate_by_time(dataset_name),
            value_columns=["bad_rate"],
        ),
        _save_if_informative(
            dataset_name=dataset_name,
            plot_file="metric_leaderboard_oot_auc.png",
            source_table="final_comparison_table.csv",
            source_df=final_df,
            columns_used=["model", "selector", "experiment_type", "oot_auc", "oot_gini"],
            purpose="Compare OOT AUC leaderboard by model and selector.",
            figure_factory=lambda: plot_metric_leaderboard(final_df),
            category_columns=["selector"],
            value_columns=["oot_auc"],
        ),
        _save_if_informative(
            dataset_name=dataset_name,
            plot_file="stability_vs_performance.png",
            source_table="final_comparison_table.csv + feature_stability_table.csv",
            source_df=final_df,
            columns_used=["model", "selector", "experiment_type", "oot_auc", "nogueira_stability"],
            purpose="Compare exact feature stability against OOT AUC.",
            figure_factory=lambda: plot_stability_vs_performance(stability_df, final_df),
            category_columns=["selector"],
            value_columns=["oot_auc", "nogueira_stability"],
        ),
        _save_if_informative(
            dataset_name=dataset_name,
            plot_file="drift_vs_performance.png",
            source_table="final_comparison_table.csv + feature_drift_table.csv",
            source_df=final_df,
            columns_used=["model", "selector", "experiment_type", "oot_auc", "selected_feature_psi_mean"],
            purpose="Compare selected-feature PSI against OOT AUC.",
            figure_factory=lambda: plot_drift_vs_performance(drift_df, final_df),
            category_columns=["selector"],
            value_columns=["oot_auc", "selected_feature_psi_mean"],
        ),
        _save_if_informative(
            dataset_name=dataset_name,
            plot_file="semantic_coverage.png",
            source_table="analysis/semantic_redundancy/semantic_coverage_by_pipeline_relabelled.csv"
            if dataset_name == "lendingclub"
            else "semantic_coverage_table.csv",
            source_df=semantic_df,
            columns_used=["model", "selector", "semantic_group", "feature_ratio"],
            purpose="Show semantic group mix by selector.",
            figure_factory=lambda: plot_semantic_coverage(semantic_df),
            category_columns=["selector", "semantic_group"],
            value_columns=["feature_ratio"],
        ),
        _save_if_informative(
            dataset_name=dataset_name,
            plot_file="runtime_tradeoff.png",
            source_table="final_comparison_table.csv",
            source_df=final_df,
            columns_used=["runtime_seconds", "oot_auc", "model", "selector", "selected_feature_count"],
            purpose="Compare runtime, feature count, and OOT AUC.",
            figure_factory=lambda: plot_runtime_tradeoff(final_df),
            category_columns=["selector"],
            value_columns=["runtime_seconds", "oot_auc"],
        ),
    ]
    if dataset_name == "lendingclub":
        monthly_df = lendingclub_monthly_bad_rate_observation_count_table()
        split_dir = _dataset_paths(dataset_name).results_dir / "final_report" / "split_diagnostics"
        split_dir.mkdir(parents=True, exist_ok=True)
        monthly_df.to_csv(split_dir / "lendingclub_monthly_bad_rate_observation_count.csv", index=False)
        rows.append(
            _save_if_informative(
                dataset_name=dataset_name,
                plot_file="lendingclub_monthly_bad_rate_observation_count.png",
                source_table="final_report/split_diagnostics/lendingclub_monthly_bad_rate_observation_count.csv",
                source_df=monthly_df,
                columns_used=["issue_month", "split_segment", "observation_count", "bad_rate"],
                purpose="Connect LendingClub DEV/OOT split rationale to monthly bad-rate and observation-count behavior.",
                figure_factory=lambda: plot_lendingclub_monthly_bad_rate_observation_count(monthly_df),
                category_columns=["split_segment"],
                value_columns=["observation_count", "bad_rate"],
            )
        )
    save_full_llm_cache_appendix(dataset_name)
    manifest = pd.DataFrame(rows, columns=FINAL_REPORT_PLOT_COLUMNS)
    manifest.to_csv(_final_report_plots_dir(dataset_name) / "plot_manifest.csv", index=False)
    return manifest


def summarize_split_rationale(dataset_name: str) -> str:
    dataset_name = _normalize_dataset_name(dataset_name)
    split_df = load_split_summary(dataset_name)
    if split_df.empty:
        return "Split rationale is incomplete because the saved DEV/OOT split manifest is missing."

    row = split_df.iloc[0]
    shared = (
        "The split is time-based rather than random. DEV is the older window used for cross-validation, feature selection, and model fitting, while OOT is the newer holdout used only for final evaluation. "
        "The window choice is justified by observation counts and target-rate behavior across time, with the goal of keeping both periods large enough for comparison without leaking future information into selector or model tuning. "
        "OOT bad rate is reported only to justify the validation setup; it is not used to tune feature selection or hyperparameters."
    )

    if dataset_name == "homecredit":
        return (
            f"{shared} For Home Credit, DEV uses relative days from -600 inclusive to -240 exclusive, and OOT uses -240 inclusive to 0 inclusive. "
            f"This yields {int(row['DEV_rows']):,} DEV rows and {int(row['OOT_rows']):,} OOT rows, with bad rates of {row['DEV_bad_rate']:.4f} and {row['OOT_bad_rate']:.4f}; the OOT minus DEV difference is {row['bad_rate_difference']:.4f}. "
            "That framing preserves an older development period and a more recent out-of-time period for realistic future-period validation."
        )

    date_text = ""
    if row.get("DEV_issue_date_start") and row.get("OOT_issue_date_end"):
        date_text = (
            f" On the processed LendingClub application table, the DEV window corresponds approximately to issue dates {row['DEV_issue_date_start']} through {row['DEV_issue_date_end']}, "
            f"and OOT corresponds to {row['OOT_issue_date_start']} through {row['OOT_issue_date_end']}."
        )
    return (
        f"{shared} For LendingClub, the configured relative window uses DEV from {int(row['DEV_start'])} inclusive to {int(row['DEV_end'])} exclusive and OOT from {int(row['OOT_start'])} inclusive to {int(row['OOT_end'])} inclusive on `recent_decision`, which is derived from issue date to simulate future-loan validation.{date_text} "
        f"This produces {int(row['DEV_rows']):,} DEV rows and {int(row['OOT_rows']):,} OOT rows, with bad rates of {row['DEV_bad_rate']:.4f} and {row['OOT_bad_rate']:.4f}; the difference is {row['bad_rate_difference']:.4f}. "
        "OOT has a higher bad rate than DEV, which makes LendingClub a harder external validation period, while both DEV and OOT retain enough observations for a meaningful comparison."
    )


def _paired_statement(dataset_name: str, selector: str, model: str = "catboost") -> str:
    paired_df = _paired_fold(dataset_name)
    if paired_df.empty:
        return "Paired-fold comparison was unavailable."
    subset = paired_df[
        (paired_df["model"] == model)
        & (paired_df["candidate_selector"] == selector)
        & (paired_df["metric"] == "auc")
    ]
    if subset.empty:
        return "Paired-fold comparison was unavailable."
    row = subset.iloc[0]
    delta = float(row["mean_delta_candidate_minus_baseline"])
    lower = float(row["ci95_lower"])
    upper = float(row["ci95_upper"])
    if lower <= 0 <= upper:
        return f"Paired-fold CV deltas versus the mRMR baseline were inconclusive (mean AUC delta {delta:.4f}, 95% CI {lower:.4f} to {upper:.4f})."
    direction = "positive" if delta > 0 else "negative"
    return f"Paired-fold CV deltas versus the mRMR baseline were {direction} (mean AUC delta {delta:.4f}, 95% CI {lower:.4f} to {upper:.4f})."


def summarize_dataset_findings(dataset_name: str) -> dict[str, str]:
    dataset_name = _normalize_dataset_name(dataset_name)
    final_df = _final_comparison(dataset_name)
    stability_df = _stability(dataset_name)
    drift_df = _drift(dataset_name)
    semantic_summary = _selector_semantic_summary(dataset_name)
    llm_df = _llm_calls(dataset_name)

    if final_df.empty:
        fallback = "The comparison artifacts are missing or empty, so this section is incomplete."
        return {
            "experiment_matrix_markdown": fallback,
            "topline_markdown": fallback,
            "stability_markdown": fallback,
            "drift_markdown": fallback,
            "semantic_markdown": fallback,
            "efficiency_markdown": fallback,
            "failure_markdown": fallback,
            "conclusion_markdown": fallback,
            "next_actions_markdown": fallback,
        }

    best_lr = _best_row(final_df, model="lr")
    best_cat = _best_row(final_df, model="catboost")
    best_baseline = _best_row(final_df, selector_set=BASELINE_SELECTORS)
    best_llm_family = _best_row(final_df, selector_set=LLM_FAMILY_SELECTORS)

    assert best_lr is not None and best_cat is not None and best_baseline is not None and best_llm_family is not None

    lr_gap = float(best_lr["oot_auc"] - _best_row(final_df, model="lr", selector_set=BASELINE_SELECTORS)["oot_auc"])
    cat_gap = float(best_cat["oot_auc"] - _best_row(final_df, model="catboost", selector_set=BASELINE_SELECTORS)["oot_auc"])

    pca_row = _best_row(final_df, selector_set={"pca"})
    boruta_row = _best_row(final_df, selector_set={"boruta"})
    llm_cache_hits = int(llm_df["llm_cache_hits"].sum()) if not llm_df.empty else 0
    llm_tokens = int(llm_df["llm_total_tokens"].sum()) if not llm_df.empty else 0

    top_semantic = semantic_summary.sort_values("semantic_group_count", ascending=False).iloc[0] if not semantic_summary.empty else None
    top_drift = drift_df.sort_values("selected_feature_psi_mean").iloc[0] if not drift_df.empty else None
    top_stability = stability_df.sort_values("nogueira_stability", ascending=False).iloc[0] if not stability_df.empty else None

    if dataset_name == "homecredit":
        topline = (
            f"Home Credit remains the main benchmark, and the topline leaderboard is mixed rather than one-sided. "
            f"For LR, `{best_lr['selector']}` is best on OOT AUC ({best_lr['oot_auc']:.4f}); for CatBoost, `{best_cat['selector']}` leads at {best_cat['oot_auc']:.4f}. "
            f"The strongest non-LLM baseline is `{best_baseline['selector']}` at {best_baseline['oot_auc']:.4f}. "
            f"The OOT gains of the best LLM-family method over the best baseline are small: {lr_gap:.4f} AUC for LR and {cat_gap:.4f} for CatBoost. "
            f"{_paired_statement(dataset_name, str(best_cat['selector']))}"
        )
        stability = (
            f"Stability does not support a simple 'LLM dominates' claim. "
            f"The highest Nogueira stability belongs to `{top_stability['selector']}` on `{top_stability['model']}` at {top_stability['nogueira_stability']:.4f}. "
            "Deterministic selectors such as PCA and the domain baseline show perfect or near-perfect repeatability, but that exact repeatability is not sufficient when OOT discrimination is weak. "
            "The stable-core hybrid improves the balance between exact feature stability and semantic stability more than the pure LLM selector."
        )
        drift = (
            f"High OOT performance on Home Credit is not concentrated in the highest-drift methods. "
            f"The lowest-drift top run in the table is `{top_drift['selector']}` on `{top_drift['model']}` with feature PSI mean {top_drift['selected_feature_psi_mean']:.4f}. "
            f"PCA deserves specific caution: its OOT scores are weak and its drift indicators are materially worse than the better-performing selectors. "
            "OOT PSI is used only for evaluation; it is not a training or selection signal."
        )
        semantic = (
            "Semantic coverage is broader for mRMR and the better LLM hybrids than for PCA or the domain baseline. "
            f"The broadest selector/model combination in the saved coverage table is `{top_semantic['selector']}` on `{top_semantic['model']}` with {int(top_semantic['semantic_group_count'])} distinct semantic groups. "
            "That supports the narrower claim that LLM screening can help preserve business-relevant feature families, but it does not remove the need for statistical discipline."
        )
        efficiency = (
            f"Efficiency tradeoffs matter. Boruta is the slowest weak baseline, while the pure LLM LR run is cheap in wall-clock terms and reasonably competitive. "
            f"Shared cache usage is already visible in the current artifacts: {llm_cache_hits} cache hits are recorded and {llm_tokens} tokens were effectively spent in the saved summaries, which limits repeated LLM cost for reused metadata rankings."
        )
        failures = (
            "The main failure cases are consistent. Boruta underperforms despite long runtime, PCA looks mechanically stable but not robust, and `llm_then_boruta` is clearly weaker than mRMR-based comparators. "
            "Home Credit auxiliary-table timing is treated as historical based on relative-time field semantics, but strict row-level as-of validation remains a manual-review limitation."
        )
        conclusion = (
            "On Home Credit, the evidence supports a careful claim: LLM screening is useful as a first-stage helper, especially in the stable-core hybrid, but the improvement over mRMR is marginal rather than dominant. "
            "The strongest carry-forward method for cross-dataset discussion is `stable_core_llm_fill`, with mRMR as the non-LLM reference. "
            "The evidence is mixed across performance, stability, drift, and semantic coverage rather than coming from a single decisive metric."
        )
    else:
        topline = (
            f"LendingClub acts as external validation, and the OOT leaderboard is tighter than on Home Credit. "
            f"For LR, `{best_lr['selector']}` is best on OOT AUC ({best_lr['oot_auc']:.4f}); for CatBoost, `{best_cat['selector']}` is best at {best_cat['oot_auc']:.4f}. "
            f"The strongest non-LLM baseline is `{best_baseline['selector']}` at {best_baseline['oot_auc']:.4f}. "
            "The headline is not universal dominance: the best LLM-family methods sit near the top, but the margins over mRMR are modest. "
            f"{_paired_statement(dataset_name, 'stable_core_llm_fill')} {_paired_statement(dataset_name, 'llm')}"
        )
        stability = (
            f"The stability picture is better for the stronger hybrids than for the pure LLM selector. "
            f"`{top_stability['selector']}` on `{top_stability['model']}` has the highest saved Nogueira stability at {top_stability['nogueira_stability']:.4f}. "
            "Again, perfect-repeatability selectors such as PCA should not be overread: semantic concentration and weak robustness matter more than exact repeatability alone."
        )
        drift = (
            f"Drift on LendingClub is generally low for the best methods, which is encouraging for the external-validation claim. "
            f"The best low-drift run in the drift table is `{top_drift['selector']}` on `{top_drift['model']}` with mean feature PSI {top_drift['selected_feature_psi_mean']:.4f}. "
            "PCA is the obvious exception and should be flagged explicitly because its feature PSI is much higher than the rest of the table."
        )
        semantic = (
            "Semantic diversity on LendingClub is more interpretable after the report-layer mapping update, because common credit-score, capacity, revolving-utilization, bankcard, and account-activity features no longer collapse unnecessarily into `other`. "
            "This relabeling improves the semantic coverage evidence but does not change feature selection results. "
            "The safer reading is that some LLM-family methods remain performance-competitive under a leakage-audited external dataset, while semantic coverage remains dataset and rule dependent."
        )
        efficiency = (
            f"Efficiency is a more serious tradeoff on LendingClub. Boruta is expensive and weak, while the best LLM-family CatBoost runs are competitive but substantially slower than the best LR runs. "
            f"The cache behavior still helps: the saved artifacts record {llm_cache_hits} cache hits and {llm_tokens} total tokens, which indicates that shared ranking and reuse reduce repeated LLM cost."
        )
        failures = (
            "The main failure cases are again Boruta and PCA, with `llm_then_boruta` also clearly underperforming. "
            "LendingClub carries a separate data-governance caveat: the current processed dataset is the safe path, while raw direct use should remain blocked or tightly audited because the raw files contain post-origination leakage fields. "
            "OOT has a higher bad rate than DEV, making the OOT period a harder external validation period while still retaining enough observations in both windows."
        )
        conclusion = (
            "On LendingClub, the honest claim is still moderate: LLM screening is useful as a first-stage helper, but it does not universally dominate the statistical baselines. "
            "The carry-forward methods for cross-dataset discussion are the best OOT LLM-family variant together with mRMR as the stability-aware non-LLM reference. "
            "The evidence is mixed, with small performance gaps, useful drift behavior, and only limited semantic-coverage separation."
        )

    experiment_matrix = (
        "The matrix compares statistical baselines, pure LLM screening, and LLM-then-statistical hybrids under the same DEV/OOT protocol. "
        "The target comparison is therefore about first-stage screening utility, not about replacing the downstream LR or CatBoost evaluation vehicles."
    )
    next_actions = (
        "Concrete next actions after this reporting refactor are narrower: manually confirm Home Credit auxiliary-table as-of semantics, keep the LendingClub raw-data leakage blacklist audited as raw schemas change, and prepare the remaining future CLIP-style validation artifacts without training that method yet."
    )

    return {
        "experiment_matrix_markdown": experiment_matrix,
        "topline_markdown": topline,
        "stability_markdown": stability,
        "drift_markdown": drift,
        "semantic_markdown": semantic,
        "efficiency_markdown": efficiency,
        "failure_markdown": failures,
        "conclusion_markdown": conclusion,
        "next_actions_markdown": next_actions,
    }


def summarize_clip_validation_placeholder(dataset_name: str) -> dict[str, Any]:
    dataset_name = _normalize_dataset_name(dataset_name)
    feature_evidence_path = _dataset_paths(dataset_name).results_dir / "feature_level_evidence.csv"
    artifacts = pd.DataFrame(
        [
            {
                "planned_artifact": "feature_level_evidence.csv",
                "purpose": "one row per feature with semantic and empirical statistics",
                "status": "available" if feature_evidence_path.exists() else "planned",
                "notes": (
                    "generated from current aggregate and per-run artifacts"
                    if feature_evidence_path.exists()
                    else "input table for alignment and downstream audit"
                ),
            },
            {
                "planned_artifact": "contrastive_pairs.csv",
                "purpose": "positive, hard-negative, and easy-negative feature pairs for contrastive training",
                "status": "planned",
                "notes": "pairs should be constructed from DEV-only evidence",
            },
            {
                "planned_artifact": "clip_embedding_table.csv",
                "purpose": "learned feature embeddings and similarity to a stable-core anchor",
                "status": "planned",
                "notes": "dual-encoder or contrastive feature-space output",
            },
            {
                "planned_artifact": "clip_vs_llm_vs_mrmr_comparison.csv",
                "purpose": "compare the future CLIP-style screener against current selectors",
                "status": "planned",
                "notes": "must be evaluated under the same DEV/OOT protocol",
            },
        ]
    )
    summary = (
        "This section is a reserved placeholder for future CLIP-style semantic-statistical feature alignment. "
        "It is not image CLIP and is not implemented here. The future method would align feature text and metadata with empirical feature behavior in a shared representation space, using DEV-only evidence and comparing the resulting screener against the current selectors under the same DEV/OOT protocol. "
        "OOT metrics must not be used for CLIP-style training or feature selection; they remain final evaluation only."
    )
    return {
        "title": "Future Extension: CLIP-Style Semantic-Statistical Feature Alignment",
        "summary": summary,
        "artifacts": artifacts,
    }


def build_cross_dataset_summary_markdown() -> str:
    rows: list[dict[str, Any]] = []
    for dataset_name in ["homecredit", "lendingclub"]:
        final_df = load_final_comparison(dataset_name)
        stability_df = load_stability_table(dataset_name)
        psi_df = load_psi_distribution_by_pipeline(dataset_name)
        if final_df.empty:
            continue

        best_lr = final_df[final_df["model"].eq("lr")].sort_values("oot_auc", ascending=False).iloc[0]
        best_cat = final_df[final_df["model"].eq("catboost")].sort_values("oot_auc", ascending=False).iloc[0]
        baseline = final_df[final_df["selector"].isin(BASELINE_SELECTORS)].sort_values("oot_auc", ascending=False)
        best_baseline = baseline.iloc[0] if not baseline.empty else pd.Series(dtype=object)
        mrmr = final_df[final_df["selector"].eq("mrmr")]
        best_mrmr_auc = float(mrmr["oot_auc"].max()) if not mrmr.empty else math.nan
        llm_family = final_df[final_df["selector"].isin(LLM_FAMILY_SELECTORS)]
        best_llm_auc = float(llm_family["oot_auc"].max()) if not llm_family.empty else math.nan
        llm_psi = psi_df[psi_df["selector"].isin(LLM_FAMILY_SELECTORS)] if not psi_df.empty else pd.DataFrame()
        non_llm_psi = psi_df[~psi_df["selector"].isin(LLM_FAMILY_SELECTORS)] if not psi_df.empty else pd.DataFrame()
        stability_reference = stability_df[stability_df["selector"].eq("mrmr")]
        best_stability = (
            stability_reference.sort_values("nogueira_stability", ascending=False).iloc[0]
            if not stability_reference.empty
            else pd.Series(dtype=object)
        )
        caveat = (
            "Home Credit auxiliary-table timing is treated as historical based on relative-time field semantics, but strict row-level as-of validation remains a manual-review limitation."
            if dataset_name == "homecredit"
            else "LendingClub uses the processed leakage-audited path; OOT has a higher bad rate than DEV and is a harder validation period."
        )
        rows.append(
            {
                "dataset": dataset_name,
                "best LR selector": best_lr["selector"],
                "best LR OOT AUC": best_lr["oot_auc"],
                "best CatBoost selector": best_cat["selector"],
                "best CatBoost OOT AUC": best_cat["oot_auc"],
                "strongest non-LLM baseline": best_baseline.get("selector", ""),
                "mRMR OOT AUC": best_mrmr_auc,
                "best LLM-family delta vs mRMR": best_llm_auc - best_mrmr_auc
                if pd.notna(best_llm_auc) and pd.notna(best_mrmr_auc)
                else math.nan,
                "LLM-family mean feature PSI": llm_psi["psi_mean"].mean() if not llm_psi.empty else math.nan,
                "non-LLM mean feature PSI": non_llm_psi["psi_mean"].mean() if not non_llm_psi.empty else math.nan,
                "best exact-stability selector": best_stability.get("selector", ""),
                "key caveat": caveat,
            }
        )

    table = _round_numeric_frame(pd.DataFrame(rows), 4)
    lines = [
        "# Cross-Dataset Summary",
        "",
        "## Main Cross-Dataset Conclusion",
        "",
        "Across both datasets, LLM screening is competitive and consistently low-drift. mRMR remains the strongest exact-stability reference, especially when the question is repeatable feature identity rather than semantic coverage or drift. Home Credit favors `stable_core_llm_fill`, while LendingClub favors pure `llm`. The contribution is not universal dominance; it is LLM-assisted first-stage screening as a useful, drift-aware candidate generator.",
        "",
        "## Cross-Dataset Comparison Table",
        "",
        _frame_to_markdown(table) if not table.empty else "Cross-dataset comparison table unavailable.",
        "",
        "## Performance Pattern",
        "",
        "LLM-family selectors sit near the top of the OOT leaderboard on both datasets, but the margins over mRMR are small. The safest interpretation is that LLM screening is useful as a first-stage helper, not that it replaces mRMR or universally dominates statistical selectors.",
        "",
        "## Exact Stability Pattern",
        "",
        "mRMR and deterministic baselines remain important exact-stability references. Exact feature stability does not by itself settle the research question, because a perfectly repeatable selector can still be semantically narrow, higher drift, or weaker on OOT discrimination.",
        "",
        "## Drift Pattern",
        "",
        "The post-run PSI evidence supports the lower-drift part of the LLM claim more strongly than the performance-dominance claim. LLM-family selected pools generally avoid high average selected-feature PSI, while PCA is the recurring drift and performance caution case.",
        "",
        "## Semantic Coverage Pattern",
        "",
        "Semantic coverage is dataset and metadata-rule dependent. Home Credit has clearer source-table and business-concept separation. LendingClub previously overused `other`; the revised mapping makes the coverage evidence more interpretable but should still be treated as report-layer relabeling rather than changed selection results.",
        "",
        "## Dataset-Specific Behavior",
        "",
        "Home Credit supports the stable-core hybrid most clearly. LendingClub supports the pure LLM selector more clearly and also provides a leakage-audited external validation setting with a harder OOT period because OOT bad rate is higher than DEV.",
        "",
        "## Final Claim Wording",
        "",
        "Use this wording: LLM screening is useful as a first-stage helper. Do not say LLM replaces mRMR. Do not say LLM universally dominates statistical selectors.",
        "",
        "## Caveats",
        "",
        "- Home Credit auxiliary-table timing is treated as historical based on relative-time field semantics, but strict row-level as-of validation remains a manual-review limitation.",
        "- Paired fold tests do not strongly support many OOT gains; small AUC gaps should be described cautiously.",
        "- The LendingClub semantic grouping improvement is metadata/report relabeling only and does not change selected features or model results.",
    ]
    return "\n".join(lines) + "\n"


def build_dataset_report_markdown(dataset_name: str) -> str:
    dataset_name = _normalize_dataset_name(dataset_name)
    snapshot = load_dataset_snapshot(dataset_name)
    split_summary = load_split_summary(dataset_name)
    matrix_overview = _matrix_overview(dataset_name)
    final_df = load_final_comparison(dataset_name)
    stability_df = load_stability_table(dataset_name)
    drift_df = load_drift_table(dataset_name)
    psi_distribution = load_psi_distribution_by_pipeline(dataset_name)
    high_psi = load_high_psi_features_by_pipeline(dataset_name)
    llm_mrmr_drift = load_llm_then_mrmr_drift_source_breakdown(dataset_name)
    llm_top100 = load_llm_top100_candidate_psi(dataset_name)
    semantic_redundancy = load_semantic_redundancy_table(dataset_name)
    paired_significance = load_paired_fold_significance_tests(dataset_name)
    best_runs = load_best_runs(dataset_name)
    compact_llm = load_compact_llm_cache_summary(dataset_name)
    appendix_path = save_full_llm_cache_appendix(dataset_name)
    clip_placeholder = summarize_clip_validation_placeholder(dataset_name)
    findings = summarize_dataset_findings(dataset_name)
    warnings = _warnings(dataset_name)

    leaderboard_cols = [
        "model",
        "selector",
        "experiment_type",
        "oot_auc",
        "oot_gini",
        "oot_ks",
        "lift_at_10",
        "selected_feature_count",
        "runtime_seconds",
        "model_score_psi",
    ]
    stability_cols = [
        "model",
        "selector",
        "nogueira_stability",
        "kuncheva_stability",
        "mean_pairwise_jaccard",
        "semantic_group_jaccard",
        "stable_feature_count_80",
        "stable_feature_ratio_80",
    ]
    drift_cols = [
        "model",
        "selector",
        "selected_feature_count",
        "psi_mean",
        "psi_median",
        "psi_p90",
        "psi_max",
        "high_psi_feature_count",
        "high_psi_feature_ratio",
    ]
    semantic_cols = [
        "model",
        "selector",
        "selected feature count",
        "number of semantic groups",
        "semantic group entropy if easy",
        "largest group share",
        "average within-group absolute correlation",
        "max within-group absolute correlation",
        "redundancy risk flag",
    ]
    significance_cols = [
        "model",
        "candidate_selector",
        "baseline_selector",
        "metric",
        "mean_delta",
        "ttest_p_value",
        "wilcoxon_p_value",
        "significant_at_0_05",
        "interpretation",
    ]

    lines = [
        f"# {_display_dataset_name(dataset_name)} Final Report",
        "",
        f"Dataset role: {_dataset_role(dataset_name)}.",
        "",
        "## Research Question and Dataset Role",
        "",
        "This research checks whether LLM metadata screening is useful as a first-stage feature-selection helper. Home Credit is the primary benchmark, and LendingClub is the external validation dataset. Logistic Regression and CatBoost are evaluation vehicles rather than the main contribution. Calibration, stacking, production scoring, and deployment are out of scope.",
        "",
        "## Snapshot",
        "",
        _frame_to_markdown(snapshot) if not snapshot.empty else "Snapshot unavailable.",
        "",
        "## DEV/OOT Split Rationale",
        "",
        summarize_split_rationale(dataset_name),
        "",
        _frame_to_markdown(split_summary) if not split_summary.empty else "Split summary unavailable.",
        "",
        "## Experiment Matrix Overview",
        "",
        _frame_to_markdown(matrix_overview) if not matrix_overview.empty else "Matrix overview unavailable.",
        "",
        findings["experiment_matrix_markdown"],
        "",
        "## Topline Performance Comparison",
        "",
        _frame_to_markdown(final_df[[col for col in leaderboard_cols if col in final_df.columns]].head(16)) if not final_df.empty else "Final comparison table unavailable.",
        "",
        findings["topline_markdown"],
        "",
        "Paired fold significance tests against mRMR:",
        "",
        _frame_to_markdown(paired_significance[[col for col in significance_cols if col in paired_significance.columns]]) if not paired_significance.empty else "Paired fold significance tests unavailable.",
        "",
        "## Stability Review",
        "",
        _frame_to_markdown(stability_df[[col for col in stability_cols if col in stability_df.columns]].head(16)) if not stability_df.empty else "Stability table unavailable.",
        "",
        findings["stability_markdown"],
        "",
        "## Drift and Robustness Review",
        "",
        _frame_to_markdown(psi_distribution[[col for col in drift_cols if col in psi_distribution.columns]].head(16)) if not psi_distribution.empty else "Feature-level PSI distribution table unavailable.",
        "",
        "High-PSI selected features:",
        "",
        _frame_to_markdown(high_psi.head(20)) if not high_psi.empty else "No high-PSI selected features were flagged, or the artifact is unavailable.",
        "",
        "`llm_then_mrmr` drift-source breakdown:",
        "",
        _frame_to_markdown(llm_mrmr_drift.head(20)) if not llm_mrmr_drift.empty else "LLM/mRMR drift-source breakdown unavailable.",
        "",
        "LLM top-100 candidate PSI evidence:",
        "",
        _frame_to_markdown(llm_top100.head(30)) if not llm_top100.empty else "LLM top-100 candidate PSI table unavailable.",
        "",
        findings["drift_markdown"],
        "",
        "## Semantic Coverage and Redundancy Review",
        "",
        _frame_to_markdown(semantic_redundancy[[col for col in semantic_cols if col in semantic_redundancy.columns]]) if not semantic_redundancy.empty else "Semantic coverage/redundancy table unavailable.",
        "",
        findings["semantic_markdown"],
        "",
        "## Efficiency Tradeoff",
        "",
        findings["efficiency_markdown"],
        "",
        "LLM call/cache summary:",
        "",
        _frame_to_markdown(compact_llm),
        "",
        f"Full cache/hash appendix: `{_repo_relative_path(appendix_path)}`." if appendix_path is not None else "Full cache/hash appendix unavailable.",
        "",
        "## Best Runs Deep Dive",
        "",
        _frame_to_markdown(best_runs) if not best_runs.empty else "Best-run artifacts unavailable.",
        "",
        "## Failure Cases and Surprises",
        "",
        findings["failure_markdown"],
        "",
        "## Conclusions for This Dataset",
        "",
        findings["conclusion_markdown"],
        "",
        f"## {clip_placeholder['title']}",
        "",
        clip_placeholder["summary"],
        "",
        _frame_to_markdown(clip_placeholder["artifacts"]),
        "",
        "## Next Actions",
        "",
        findings["next_actions_markdown"],
    ]
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in warnings])
    return "\n".join(lines) + "\n"
