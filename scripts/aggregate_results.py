from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in [PROJECT_ROOT, SRC_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from credit_risk_fs.evaluation.semantic_coverage import semantic_coverage_frame  # noqa: E402


FINAL_COLUMNS = [
    "dataset_name",
    "model",
    "selector",
    "experiment_type",
    "feature_budget",
    "llm_shared_ranking_enabled",
    "llm_ranking_budget",
    "oot_auc",
    "oot_gini",
    "oot_ks",
    "oot_log_loss",
    "oot_brier",
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
    "selected_feature_psi_mean",
    "selected_feature_psi_max",
    "selected_feature_psi_median",
    "selected_feature_psi_high_drift_ratio",
    "selected_feature_psi_moderate_or_high_drift_ratio",
    "model_score_psi",
    "selected_feature_count",
    "total_candidate_feature_count",
    "feature_reduction_ratio",
    "oot_gini_per_feature",
    "oot_auc_per_feature",
    "oot_ks_per_feature",
    "lift_at_10",
    "bad_rate_capture_at_10",
    "lift_at_20",
    "bad_rate_capture_at_20",
    "config_hash",
    "data_fingerprint",
    "run_id",
    "output_folder",
    "runtime_seconds",
]

EXPECTED_SELECTORS = [
    "mrmr",
    "boruta",
    "pca",
    "domain_rule_baseline",
    "llm",
    "llm_then_mrmr",
    "llm_then_boruta",
    "stable_core_llm_fill",
]

BASELINE_SELECTORS = {"mrmr", "boruta", "pca", "domain_rule_baseline"}
LLM_FAMILY_SELECTORS = {"llm", "llm_then_mrmr", "llm_then_boruta", "stable_core_llm_fill"}

LLM_SUMMARY_COLUMNS = [
    "run_id",
    "model",
    "selector",
    "experiment_type",
    "status",
    "llm_shared_ranking_enabled",
    "llm_ranking_budget",
    "llm_calls_actually_made",
    "llm_cache_hits",
    "llm_cache_key",
    "llm_metadata_signatures",
    "llm_cache_key_hashes",
    "llm_prompt_versions",
    "llm_prompt_hashes",
    "llm_request_models",
    "llm_response_models",
    "llm_response_ids",
    "llm_cache_file_names",
    "llm_prompt_tokens",
    "llm_completion_tokens",
    "llm_total_tokens",
    "runs_sharing_metadata_signatures",
    "runs_sharing_cache_key_hashes",
    "output_folder",
]

FEATURE_LEVEL_EVIDENCE_COLUMNS = [
    "dataset_name",
    "feature_name",
    "source_table",
    "semantic_group",
    "description",
    "dtype",
    "missing_rate_mean",
    "missing_rate_max",
    "non_null_count_mean",
    "selected_in_final_run_count",
    "selected_in_mrmr_run_count",
    "selected_in_boruta_run_count",
    "selected_in_pca_run_count",
    "selected_in_domain_rule_baseline_run_count",
    "selected_in_llm_run_count",
    "selected_in_llm_then_mrmr_run_count",
    "selected_in_llm_then_boruta_run_count",
    "selected_in_stable_core_llm_fill_run_count",
    "selected_in_baseline_run_count",
    "selected_in_llm_family_run_count",
    "selected_in_lr_run_count",
    "selected_in_catboost_run_count",
    "max_within_run_selection_frequency",
    "mean_within_run_selection_frequency",
    "best_selected_rank",
    "mean_selected_rank",
    "best_llm_final_dev_rank",
    "mean_llm_final_dev_rank",
    "best_oot_auc_when_selected",
    "best_oot_gini_when_selected",
    "selectors_selected_by",
    "models_selected_by",
    "run_ids_selected_by",
    "llm_reason_example",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate completed matrix runs into final comparison tables.",
    )
    parser.add_argument("results_dir", nargs="?", default="results")
    parser.add_argument("--dataset", choices=["homecredit", "lendingclub", "lendingclub_v2"], default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to <results_dir>/final_comparison_table.csv.",
    )
    return parser


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _unique_sorted_strings(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        parts = values.split(";")
    elif isinstance(values, (list, tuple, set)):
        parts = list(values)
    else:
        parts = [values]
    normalized = {
        str(value).strip()
        for value in parts
        if value is not None and not pd.isna(value) and str(value).strip()
    }
    return sorted(normalized)


def _join_unique_strings(values: object) -> str:
    return ";".join(_unique_sorted_strings(values))


def _feature_name_column(df: pd.DataFrame) -> str | None:
    for candidate in ["feature_name", "feature", "name"]:
        if candidate in df.columns:
            return candidate
    return None


def _normalize_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feature_col = _feature_name_column(df)
    if feature_col is None or df.empty:
        return pd.DataFrame()
    normalized = df.copy()
    normalized["feature_name"] = normalized[feature_col].astype(str)
    return normalized


def _first_non_empty(series: pd.Series) -> object:
    for value in series:
        if value is None or pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return pd.NA


def _llm_ranking_stats(run_dir: Path, manifest: dict | None = None) -> dict[str, object]:
    summary_path = run_dir / "features" / "llm_rankings_summary.csv"
    defaults = {
        "llm_cache_key": None,
        "llm_metadata_signatures": [],
        "llm_cache_key_hashes": [],
        "llm_prompt_versions": [],
        "llm_prompt_hashes": [],
        "llm_request_models": [],
        "llm_response_models": [],
        "llm_response_ids": [],
        "llm_cache_file_names": [],
        "llm_calls_actually_made": 0,
        "llm_cache_hits": 0,
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "llm_total_tokens": 0,
    }
    if summary_path.exists():
        try:
            df = pd.read_csv(summary_path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        if not df.empty:
            stats = defaults.copy()
            list_mappings = {
                "llm_metadata_signatures": "metadata_signature",
                "llm_cache_key_hashes": "cache_key_hash",
                "llm_prompt_versions": "prompt_version",
                "llm_prompt_hashes": "prompt_hash",
                "llm_request_models": "request_model",
                "llm_response_models": "response_model",
                "llm_response_ids": "response_id",
                "llm_cache_file_names": "cache_file_name",
            }
            for target_key, column in list_mappings.items():
                if column in df.columns:
                    stats[target_key] = _unique_sorted_strings(df[column].dropna().tolist())

            metadata_signatures = stats["llm_metadata_signatures"]
            stats["llm_cache_key"] = metadata_signatures[0] if metadata_signatures else None

            if "cache_hit" in df.columns:
                scope_keys = [
                    "scope",
                    "fold_id",
                    "metadata_signature",
                    "cache_key_hash",
                    "cache_file_name",
                    "response_id",
                ]
                available_keys = [key for key in scope_keys if key in df.columns]
                call_df = df.drop_duplicates(subset=available_keys) if available_keys else df
                cache_flags = call_df["cache_hit"].astype(str).str.lower().isin(["true", "1"])
                stats["llm_cache_hits"] = int(cache_flags.sum())
                stats["llm_calls_actually_made"] = int((~cache_flags).sum())
                actual_call_df = call_df.loc[~cache_flags]
            else:
                actual_call_df = df

            for column, target_key in [
                ("prompt_tokens", "llm_prompt_tokens"),
                ("completion_tokens", "llm_completion_tokens"),
                ("total_tokens", "llm_total_tokens"),
            ]:
                if column in actual_call_df.columns:
                    value = pd.to_numeric(actual_call_df[column], errors="coerce").fillna(0).sum()
                    stats[target_key] = int(value)
            return stats

    manifest = manifest or {}
    return {
        **defaults,
        "llm_cache_key": manifest.get("llm_cache_key"),
        "llm_metadata_signatures": _unique_sorted_strings(manifest.get("llm_metadata_signatures")),
        "llm_cache_key_hashes": _unique_sorted_strings(manifest.get("llm_cache_key_hashes")),
        "llm_prompt_versions": _unique_sorted_strings(manifest.get("llm_prompt_versions")),
        "llm_prompt_hashes": _unique_sorted_strings(manifest.get("llm_prompt_hashes")),
        "llm_request_models": _unique_sorted_strings(manifest.get("llm_request_models")),
        "llm_response_models": _unique_sorted_strings(manifest.get("llm_response_models")),
        "llm_response_ids": _unique_sorted_strings(manifest.get("llm_response_ids")),
        "llm_cache_file_names": _unique_sorted_strings(manifest.get("llm_cache_file_names")),
        "llm_calls_actually_made": int(manifest.get("llm_calls_actually_made", 0) or 0),
        "llm_cache_hits": int(manifest.get("llm_cache_hits", 0) or 0),
        "llm_prompt_tokens": int(manifest.get("llm_prompt_tokens", 0) or 0),
        "llm_completion_tokens": int(manifest.get("llm_completion_tokens", 0) or 0),
        "llm_total_tokens": int(manifest.get("llm_total_tokens", 0) or 0),
    }


def _allowed_run_dirs(results_root: Path) -> set[Path] | None:
    matrix_path = results_root / "matrix_runs.csv"
    if not matrix_path.exists():
        return None
    matrix_df = pd.read_csv(matrix_path)
    if "output_folder" not in matrix_df.columns:
        return None
    allowed_dirs: set[Path] = set()
    for value in matrix_df["output_folder"].dropna().tolist():
        raw_path = Path(str(value))
        if raw_path.is_absolute():
            allowed_dirs.add(raw_path.resolve())
            continue

        # Matrix rows may be written relative to either the project root
        # (`results/homecredit/...`) or the active results root (`lr/...`).
        bases = [PROJECT_ROOT, Path.cwd(), results_root, *results_root.parents]
        for base in bases:
            allowed_dirs.add((base / raw_path).resolve())
    return allowed_dirs


def _numeric_folds(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "fold" not in df.columns:
        return pd.DataFrame()
    fold_numeric = pd.to_numeric(df["fold"], errors="coerce")
    folds = df.loc[fold_numeric.notna()].copy()
    folds["fold"] = fold_numeric.loc[fold_numeric.notna()].astype(int)
    return folds.sort_values("fold").reset_index(drop=True)


def _runtime_seconds(run_dir: Path, summary: dict) -> float:
    runtime_path = run_dir / "results" / "runtime_summary.csv"
    if runtime_path.exists():
        runtime_df = pd.read_csv(runtime_path)
        if not runtime_df.empty and "total_runtime_seconds" in runtime_df.columns:
            value = runtime_df.iloc[0]["total_runtime_seconds"]
            if pd.notna(value):
                return float(value)
    value = summary.get("runtime_seconds")
    if value is not None and pd.notna(value):
        return float(value)
    cv_path = run_dir / "results" / "cv_results.csv"
    folds = _numeric_folds(cv_path)
    if "fold_time_sec" in folds.columns and folds["fold_time_sec"].notna().any():
        return float(pd.to_numeric(folds["fold_time_sec"], errors="coerce").sum())
    value = summary.get("cv_fold_time_sec_mean")
    return float(value) if value is not None and pd.notna(value) else math.nan


def _summary_for_run(run_dir: Path, manifest: dict) -> dict | None:
    summary_path = run_dir / "results" / "experiment_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        if not df.empty:
            return df.iloc[0].to_dict()
    summary = manifest.get("summary")
    return summary if isinstance(summary, dict) else None


def _completed_run_rows(results_root: Path) -> list[dict]:
    rows = []
    allowed_dirs = _allowed_run_dirs(results_root)
    for manifest_path in sorted(results_root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        if run_dir == results_root:
            continue
        if allowed_dirs is not None and run_dir.resolve() not in allowed_dirs:
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "completed":
            continue

        summary = _summary_for_run(run_dir, manifest)
        if summary is None:
            continue

        selected_count = summary.get("oot_selected_feature_count", summary.get("final_selected_feature_count"))
        row = {
            "dataset_name": manifest.get("config", {}).get("dataset_name", results_root.name),
            "run_id": manifest.get("run_id"),
            "model": manifest.get("model") or summary.get("model_name"),
            "selector": manifest.get("selector") or summary.get("selector_name"),
            "experiment_type": manifest.get("experiment_type"),
            "feature_budget": manifest.get("feature_budget", summary.get("oot_feature_budget")),
            "llm_shared_ranking_enabled": manifest.get("llm_shared_ranking_enabled"),
            "llm_ranking_budget": manifest.get("llm_ranking_budget"),
            "config_hash": manifest.get("config_hash"),
            "data_fingerprint": json.dumps(manifest.get("data_version", {}), sort_keys=True),
            "output_folder": str(run_dir),
            "runtime_seconds": _runtime_seconds(run_dir, summary),
            "oot_auc": summary.get("oot_auc"),
            "oot_gini": summary.get("oot_gini"),
            "oot_ks": summary.get("oot_ks"),
            "oot_log_loss": summary.get("oot_log_loss"),
            "oot_brier": summary.get("oot_brier"),
            "nogueira_stability": summary.get("nogueira_stability"),
            "kuncheva_stability": summary.get("kuncheva_stability"),
            "mean_pairwise_jaccard": summary.get(
                "mean_pairwise_jaccard",
                summary.get("cv_jaccard_similarity_mean"),
            ),
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
            "selected_feature_psi_high_drift_ratio": summary.get(
                "oot_selected_feature_psi_high_drift_ratio"
            ),
            "selected_feature_psi_moderate_or_high_drift_ratio": summary.get(
                "oot_selected_feature_psi_moderate_or_high_drift_ratio"
            ),
            "model_score_psi": summary.get("oot_model_score_psi"),
            "selected_feature_count": selected_count,
            "total_candidate_feature_count": summary.get("oot_total_candidate_feature_count"),
            "feature_reduction_ratio": summary.get("oot_feature_reduction_ratio"),
            "oot_gini_per_feature": summary.get("oot_oot_gini_per_feature"),
            "oot_auc_per_feature": summary.get("oot_oot_auc_per_feature"),
            "oot_ks_per_feature": summary.get("oot_oot_ks_per_feature"),
            "lift_at_10": summary.get("oot_lift_at_10"),
            "bad_rate_capture_at_10": summary.get("oot_bad_rate_capture_at_10"),
            "lift_at_20": summary.get("oot_lift_at_20"),
            "bad_rate_capture_at_20": summary.get("oot_bad_rate_capture_at_20"),
        }
        rows.append(row)
    return rows


def _semantic_coverage_rows(rows: list[dict]) -> pd.DataFrame:
    coverage_rows: list[dict[str, object]] = []
    for row in rows:
        run_dir = Path(str(row["output_folder"]))
        selected_path = run_dir / "features" / "final_selected_features.csv"
        if not selected_path.exists():
            continue
        selected_df = pd.read_csv(selected_path)
        feature_col = "feature_name" if "feature_name" in selected_df.columns else "feature"
        if feature_col not in selected_df.columns:
            continue
        coverage = semantic_coverage_frame(selected_df[feature_col].dropna().astype(str).tolist())
        for _, coverage_row in coverage.iterrows():
            coverage_rows.append(
                {
                    "dataset_name": row.get("dataset_name"),
                    "run_id": row.get("run_id"),
                    "model": row.get("model"),
                    "selector": row.get("selector"),
                    "experiment_type": row.get("experiment_type"),
                    "semantic_group": coverage_row["semantic_group"],
                    "feature_count": coverage_row["feature_count"],
                    "feature_ratio": coverage_row["feature_ratio"],
                }
            )
    return pd.DataFrame(coverage_rows)


def _write_dataset_metric_tables(comparison_df: pd.DataFrame, rows: list[dict], results_root: Path) -> None:
    stability_columns = [
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
    drift_columns = [
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
    comparison_df.reindex(columns=stability_columns).to_csv(
        results_root / "feature_stability_table.csv",
        index=False,
    )
    comparison_df.reindex(columns=drift_columns).to_csv(
        results_root / "feature_drift_table.csv",
        index=False,
    )
    _semantic_coverage_rows(rows).to_csv(
        results_root / "semantic_coverage_table.csv",
        index=False,
    )


def _manifest_records(results_root: Path) -> list[tuple[Path, dict]]:
    records = []
    allowed_dirs = _allowed_run_dirs(results_root)
    for manifest_path in sorted(results_root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        if run_dir == results_root:
            continue
        if allowed_dirs is not None and run_dir.resolve() not in allowed_dirs:
            continue
        try:
            records.append((run_dir, _read_json(manifest_path)))
        except Exception:
            continue
    return records


def _write_llm_call_summary(results_root: Path) -> Path:
    rows = []
    for run_dir, manifest in _manifest_records(results_root):
        stats = _llm_ranking_stats(run_dir, manifest)
        rows.append(
            {
                "run_id": manifest.get("run_id"),
                "model": manifest.get("model"),
                "selector": manifest.get("selector"),
                "experiment_type": manifest.get("experiment_type"),
                "status": manifest.get("status"),
                "llm_shared_ranking_enabled": manifest.get("llm_shared_ranking_enabled"),
                "llm_ranking_budget": manifest.get("llm_ranking_budget"),
                "llm_calls_actually_made": stats["llm_calls_actually_made"],
                "llm_cache_hits": stats["llm_cache_hits"],
                "llm_cache_key": stats["llm_cache_key"],
                "llm_metadata_signatures": _join_unique_strings(stats["llm_metadata_signatures"]),
                "llm_cache_key_hashes": _join_unique_strings(stats["llm_cache_key_hashes"]),
                "llm_prompt_versions": _join_unique_strings(stats["llm_prompt_versions"]),
                "llm_prompt_hashes": _join_unique_strings(stats["llm_prompt_hashes"]),
                "llm_request_models": _join_unique_strings(stats["llm_request_models"]),
                "llm_response_models": _join_unique_strings(stats["llm_response_models"]),
                "llm_response_ids": _join_unique_strings(stats["llm_response_ids"]),
                "llm_cache_file_names": _join_unique_strings(stats["llm_cache_file_names"]),
                "llm_prompt_tokens": stats["llm_prompt_tokens"],
                "llm_completion_tokens": stats["llm_completion_tokens"],
                "llm_total_tokens": stats["llm_total_tokens"],
                "output_folder": str(run_dir),
            }
        )
    signature_to_runs: dict[str, list[str]] = {}
    cache_key_hash_to_runs: dict[str, list[str]] = {}
    for row in rows:
        for signature in str(row.get("llm_metadata_signatures") or "").split(";"):
            if signature:
                signature_to_runs.setdefault(signature, []).append(str(row["run_id"]))
        for cache_key_hash in str(row.get("llm_cache_key_hashes") or "").split(";"):
            if cache_key_hash:
                cache_key_hash_to_runs.setdefault(cache_key_hash, []).append(str(row["run_id"]))
    for row in rows:
        sharing = set()
        sharing_by_hash = set()
        for signature in str(row.get("llm_metadata_signatures") or "").split(";"):
            sharing.update(signature_to_runs.get(signature, []))
        for cache_key_hash in str(row.get("llm_cache_key_hashes") or "").split(";"):
            sharing_by_hash.update(cache_key_hash_to_runs.get(cache_key_hash, []))
        row["runs_sharing_metadata_signatures"] = ";".join(sorted(sharing))
        row["runs_sharing_cache_key_hashes"] = ";".join(sorted(sharing_by_hash))
    output = results_root / "llm_call_summary.csv"
    pd.DataFrame(rows, columns=LLM_SUMMARY_COLUMNS).to_csv(output, index=False)
    return output


def _write_failed_runs(results_root: Path) -> Path:
    columns = [
        "run_id",
        "model",
        "selector",
        "experiment_type",
        "status",
        "error",
        "failed_at",
        "output_folder",
    ]
    rows = []
    for run_dir, manifest in _manifest_records(results_root):
        if manifest.get("status") != "failed":
            continue
        rows.append(
            {
                "run_id": manifest.get("run_id"),
                "model": manifest.get("model"),
                "selector": manifest.get("selector"),
                "experiment_type": manifest.get("experiment_type"),
                "status": manifest.get("status"),
                "error": manifest.get("error"),
                "failed_at": manifest.get("failed_at"),
                "output_folder": str(run_dir),
            }
        )
    output = results_root / "failed_runs.csv"
    pd.DataFrame(rows, columns=columns).to_csv(output, index=False)
    return output


def _paired_fold_comparisons(rows: list[dict], results_root: Path) -> pd.DataFrame:
    run_frames = {}
    for row in rows:
        run_dir = Path(row["output_folder"])
        folds = _numeric_folds(run_dir / "results" / "cv_results.csv")
        if folds.empty:
            continue
        run_frames[row["run_id"]] = folds

    comparisons = []
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)

    for model, model_rows in by_model.items():
        baselines = [
            row
            for row in model_rows
            if row.get("experiment_type") == "statistical" and row.get("selector") == "mrmr"
        ]
        if not baselines:
            continue
        baseline = baselines[0]
        baseline_folds = run_frames.get(baseline["run_id"])
        if baseline_folds is None:
            continue

        for row in model_rows:
            if row["run_id"] == baseline["run_id"]:
                continue
            folds = run_frames.get(row["run_id"])
            if folds is None:
                continue
            merged = baseline_folds[["fold", "auc", "gini"]].merge(
                folds[["fold", "auc", "gini"]],
                on="fold",
                suffixes=("_baseline", "_candidate"),
            )
            if merged.empty:
                continue
            for metric in ["auc", "gini"]:
                deltas = merged[f"{metric}_candidate"] - merged[f"{metric}_baseline"]
                mean_delta = float(deltas.mean())
                std_delta = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
                stderr = std_delta / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
                margin = 1.96 * stderr
                comparisons.append(
                    {
                        "model": model,
                        "baseline_run_id": baseline["run_id"],
                        "candidate_run_id": row["run_id"],
                        "candidate_selector": row["selector"],
                        "candidate_experiment_type": row["experiment_type"],
                        "metric": metric,
                        "fold_count": int(len(deltas)),
                        "mean_delta_candidate_minus_baseline": mean_delta,
                        "ci95_lower": mean_delta - margin,
                        "ci95_upper": mean_delta + margin,
                    }
                )

    return pd.DataFrame(comparisons)


def _load_feature_metadata_for_run(run_dir: Path) -> pd.DataFrame:
    candidates = [
        run_dir / "llm_responses" / "final_dev" / "llm" / "feature_metadata.csv",
        run_dir / "feature_metadata.csv",
    ]
    candidates.extend(sorted(run_dir.glob("llm_responses/*/llm/feature_metadata.csv")))
    for candidate in candidates:
        df = _normalize_feature_frame(_read_csv(candidate))
        if df.empty:
            continue
        metadata = df.copy()
        if "table" in metadata.columns and "source_table" not in metadata.columns:
            metadata["source_table"] = metadata["table"]
        keep_columns = [
            "feature_name",
            "source_table",
            "semantic_group",
            "description",
            "dtype",
            "missing_rate",
            "non_null_count",
        ]
        available = [column for column in keep_columns if column in metadata.columns]
        return metadata[available].drop_duplicates(subset=["feature_name"])
    return pd.DataFrame(columns=["feature_name"])


def _write_feature_level_evidence(rows: list[dict], results_root: Path) -> Path:
    metadata_frames: list[pd.DataFrame] = []
    selection_frames: list[pd.DataFrame] = []
    frequency_frames: list[pd.DataFrame] = []
    llm_rank_frames: list[pd.DataFrame] = []

    for row in rows:
        run_dir = Path(str(row["output_folder"]))
        run_id = str(row.get("run_id"))
        selector = str(row.get("selector"))
        model = str(row.get("model"))

        metadata_df = _load_feature_metadata_for_run(run_dir)
        if not metadata_df.empty:
            metadata_df = metadata_df.copy()
            metadata_df["run_id"] = run_id
            metadata_frames.append(metadata_df)

        selected_df = _normalize_feature_frame(_read_csv(run_dir / "features" / "final_selected_features.csv"))
        if not selected_df.empty:
            selected_df = selected_df.copy()
            selected_df["run_id"] = run_id
            selected_df["selector"] = selector
            selected_df["model"] = model
            selected_df["oot_auc"] = row.get("oot_auc")
            selected_df["oot_gini"] = row.get("oot_gini")
            selection_frames.append(selected_df)

        frequency_df = _normalize_feature_frame(_read_csv(run_dir / "features" / "selection_frequency.csv"))
        if not frequency_df.empty:
            frequency_df = frequency_df.copy()
            frequency_df["run_id"] = run_id
            frequency_df["selector"] = selector
            frequency_df["model"] = model
            frequency_frames.append(frequency_df)

        llm_df = _normalize_feature_frame(_read_csv(run_dir / "features" / "llm_rankings_summary.csv"))
        if not llm_df.empty:
            llm_df = llm_df.copy()
            llm_df["run_id"] = run_id
            llm_df["selector"] = selector
            llm_df["model"] = model
            llm_rank_frames.append(llm_df)

    feature_names: set[str] = set()
    for frame in [*metadata_frames, *selection_frames, *frequency_frames, *llm_rank_frames]:
        if "feature_name" in frame.columns:
            feature_names.update(frame["feature_name"].dropna().astype(str).tolist())

    evidence_rows: list[dict[str, object]] = []
    metadata_all = pd.concat(metadata_frames, ignore_index=True) if metadata_frames else pd.DataFrame()
    selection_all = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    frequency_all = pd.concat(frequency_frames, ignore_index=True) if frequency_frames else pd.DataFrame()
    llm_rank_all = pd.concat(llm_rank_frames, ignore_index=True) if llm_rank_frames else pd.DataFrame()

    for feature_name in sorted(feature_names):
        feature_row: dict[str, object] = {
            "dataset_name": results_root.name,
            "feature_name": feature_name,
        }

        metadata_slice = (
            metadata_all.loc[metadata_all["feature_name"].astype(str) == feature_name].copy()
            if not metadata_all.empty
            else pd.DataFrame()
        )
        selection_slice = (
            selection_all.loc[selection_all["feature_name"].astype(str) == feature_name].copy()
            if not selection_all.empty
            else pd.DataFrame()
        )
        frequency_slice = (
            frequency_all.loc[frequency_all["feature_name"].astype(str) == feature_name].copy()
            if not frequency_all.empty
            else pd.DataFrame()
        )
        llm_slice = (
            llm_rank_all.loc[llm_rank_all["feature_name"].astype(str) == feature_name].copy()
            if not llm_rank_all.empty
            else pd.DataFrame()
        )

        if not metadata_slice.empty:
            for column in ["missing_rate", "non_null_count"]:
                if column in metadata_slice.columns:
                    metadata_slice[column] = pd.to_numeric(metadata_slice[column], errors="coerce")
            feature_row["source_table"] = _join_unique_strings(
                metadata_slice["source_table"].dropna().tolist() if "source_table" in metadata_slice.columns else []
            )
            feature_row["semantic_group"] = _first_non_empty(metadata_slice.get("semantic_group", pd.Series(dtype=object)))
            feature_row["description"] = _first_non_empty(metadata_slice.get("description", pd.Series(dtype=object)))
            feature_row["dtype"] = _first_non_empty(metadata_slice.get("dtype", pd.Series(dtype=object)))
            feature_row["missing_rate_mean"] = metadata_slice["missing_rate"].mean() if "missing_rate" in metadata_slice.columns else pd.NA
            feature_row["missing_rate_max"] = metadata_slice["missing_rate"].max() if "missing_rate" in metadata_slice.columns else pd.NA
            feature_row["non_null_count_mean"] = metadata_slice["non_null_count"].mean() if "non_null_count" in metadata_slice.columns else pd.NA
        else:
            fallback_semantic = (
                selection_slice["semantic_group"]
                if not selection_slice.empty and "semantic_group" in selection_slice.columns
                else pd.Series(dtype=object)
            )
            feature_row["source_table"] = pd.NA
            feature_row["semantic_group"] = _first_non_empty(fallback_semantic)
            feature_row["description"] = pd.NA
            feature_row["dtype"] = pd.NA
            feature_row["missing_rate_mean"] = pd.NA
            feature_row["missing_rate_max"] = pd.NA
            feature_row["non_null_count_mean"] = pd.NA

        if not selection_slice.empty:
            if "rank" in selection_slice.columns:
                selection_slice["rank"] = pd.to_numeric(selection_slice["rank"], errors="coerce")
            feature_row["selected_in_final_run_count"] = int(selection_slice["run_id"].nunique())
            for selector_name in EXPECTED_SELECTORS:
                feature_row[f"selected_in_{selector_name}_run_count"] = int(
                    selection_slice.loc[selection_slice["selector"] == selector_name, "run_id"].nunique()
                )
            feature_row["selected_in_baseline_run_count"] = int(
                selection_slice.loc[selection_slice["selector"].isin(BASELINE_SELECTORS), "run_id"].nunique()
            )
            feature_row["selected_in_llm_family_run_count"] = int(
                selection_slice.loc[selection_slice["selector"].isin(LLM_FAMILY_SELECTORS), "run_id"].nunique()
            )
            feature_row["selected_in_lr_run_count"] = int(
                selection_slice.loc[selection_slice["model"] == "lr", "run_id"].nunique()
            )
            feature_row["selected_in_catboost_run_count"] = int(
                selection_slice.loc[selection_slice["model"] == "catboost", "run_id"].nunique()
            )
            feature_row["best_selected_rank"] = (
                selection_slice["rank"].min() if "rank" in selection_slice.columns else pd.NA
            )
            feature_row["mean_selected_rank"] = (
                selection_slice["rank"].mean() if "rank" in selection_slice.columns else pd.NA
            )
            feature_row["best_oot_auc_when_selected"] = pd.to_numeric(
                selection_slice.get("oot_auc"),
                errors="coerce",
            ).max()
            feature_row["best_oot_gini_when_selected"] = pd.to_numeric(
                selection_slice.get("oot_gini"),
                errors="coerce",
            ).max()
            feature_row["selectors_selected_by"] = _join_unique_strings(selection_slice["selector"].tolist())
            feature_row["models_selected_by"] = _join_unique_strings(selection_slice["model"].tolist())
            feature_row["run_ids_selected_by"] = _join_unique_strings(selection_slice["run_id"].tolist())
        else:
            feature_row["selected_in_final_run_count"] = 0
            for selector_name in EXPECTED_SELECTORS:
                feature_row[f"selected_in_{selector_name}_run_count"] = 0
            feature_row["selected_in_baseline_run_count"] = 0
            feature_row["selected_in_llm_family_run_count"] = 0
            feature_row["selected_in_lr_run_count"] = 0
            feature_row["selected_in_catboost_run_count"] = 0
            feature_row["best_selected_rank"] = pd.NA
            feature_row["mean_selected_rank"] = pd.NA
            feature_row["best_oot_auc_when_selected"] = pd.NA
            feature_row["best_oot_gini_when_selected"] = pd.NA
            feature_row["selectors_selected_by"] = ""
            feature_row["models_selected_by"] = ""
            feature_row["run_ids_selected_by"] = ""

        if not frequency_slice.empty and "selection_frequency" in frequency_slice.columns:
            freq_values = pd.to_numeric(frequency_slice["selection_frequency"], errors="coerce")
            feature_row["max_within_run_selection_frequency"] = freq_values.max()
            feature_row["mean_within_run_selection_frequency"] = freq_values.mean()
        else:
            feature_row["max_within_run_selection_frequency"] = pd.NA
            feature_row["mean_within_run_selection_frequency"] = pd.NA

        if not llm_slice.empty:
            final_dev_slice = (
                llm_slice.loc[llm_slice["scope"].astype(str) == "final_dev"].copy()
                if "scope" in llm_slice.columns
                else llm_slice.copy()
            )
            if "rank" in final_dev_slice.columns:
                final_dev_slice["rank"] = pd.to_numeric(final_dev_slice["rank"], errors="coerce")
                feature_row["best_llm_final_dev_rank"] = final_dev_slice["rank"].min()
                feature_row["mean_llm_final_dev_rank"] = final_dev_slice["rank"].mean()
            else:
                feature_row["best_llm_final_dev_rank"] = pd.NA
                feature_row["mean_llm_final_dev_rank"] = pd.NA
            if "llm_reason" in final_dev_slice.columns and not final_dev_slice.empty:
                ranked = (
                    final_dev_slice.sort_values("rank", na_position="last")
                    if "rank" in final_dev_slice.columns
                    else final_dev_slice
                )
                feature_row["llm_reason_example"] = _first_non_empty(ranked["llm_reason"])
            else:
                feature_row["llm_reason_example"] = pd.NA
        else:
            feature_row["best_llm_final_dev_rank"] = pd.NA
            feature_row["mean_llm_final_dev_rank"] = pd.NA
            feature_row["llm_reason_example"] = pd.NA

        evidence_rows.append(feature_row)

    evidence_df = pd.DataFrame(evidence_rows).reindex(columns=FEATURE_LEVEL_EVIDENCE_COLUMNS)
    if not evidence_df.empty:
        evidence_df = evidence_df.sort_values(
            ["selected_in_final_run_count", "best_oot_auc_when_selected", "feature_name"],
            ascending=[False, False, True],
            na_position="last",
        )
    output = results_root / "feature_level_evidence.csv"
    evidence_df.to_csv(output, index=False)
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_root = (
        PROJECT_ROOT / "results" / args.dataset
        if args.dataset
        else Path(args.results_dir)
    )
    output_path = Path(args.output) if args.output else results_root / "final_comparison_table.csv"

    rows = _completed_run_rows(results_root)
    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.reindex(columns=FINAL_COLUMNS)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values(
            ["model", "oot_gini", "oot_ks"],
            ascending=[True, False, False],
            na_position="last",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)

    paired_df = _paired_fold_comparisons(rows, results_root)
    paired_path = output_path.parent / "paired_fold_comparisons.csv"
    paired_df.to_csv(paired_path, index=False)
    llm_summary_path = _write_llm_call_summary(results_root)
    failed_runs_path = _write_failed_runs(results_root)
    _write_dataset_metric_tables(comparison_df, rows, results_root)
    feature_evidence_path = _write_feature_level_evidence(rows, results_root)

    print(f"Final comparison table: {output_path.resolve()}")
    print(f"Paired fold comparisons: {paired_path.resolve()}")
    print(f"LLM call summary: {llm_summary_path.resolve()}")
    print(f"Failed runs: {failed_runs_path.resolve()}")
    print(f"Feature-level evidence: {feature_evidence_path.resolve()}")
    print(f"Completed runs aggregated: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
