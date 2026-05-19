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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate completed matrix runs into final comparison tables.",
    )
    parser.add_argument("results_dir", nargs="?", default="results")
    parser.add_argument("--dataset", choices=["homecredit", "lendingclub"], default=None)
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to <results_dir>/final_comparison_table.csv.",
    )
    return parser


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    columns = [
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
        "llm_prompt_tokens",
        "llm_completion_tokens",
        "llm_total_tokens",
        "runs_sharing_metadata_signatures",
        "output_folder",
    ]
    rows = []
    for run_dir, manifest in _manifest_records(results_root):
        rows.append(
            {
                "run_id": manifest.get("run_id"),
                "model": manifest.get("model"),
                "selector": manifest.get("selector"),
                "experiment_type": manifest.get("experiment_type"),
                "status": manifest.get("status"),
                "llm_shared_ranking_enabled": manifest.get("llm_shared_ranking_enabled"),
                "llm_ranking_budget": manifest.get("llm_ranking_budget"),
                "llm_calls_actually_made": manifest.get("llm_calls_actually_made", 0),
                "llm_cache_hits": manifest.get("llm_cache_hits", 0),
                "llm_cache_key": manifest.get("llm_cache_key"),
                "llm_metadata_signatures": ";".join(manifest.get("llm_metadata_signatures", []) or []),
                "llm_prompt_tokens": manifest.get("llm_prompt_tokens", 0),
                "llm_completion_tokens": manifest.get("llm_completion_tokens", 0),
                "llm_total_tokens": manifest.get("llm_total_tokens", 0),
                "output_folder": str(run_dir),
            }
        )
    signature_to_runs: dict[str, list[str]] = {}
    for row in rows:
        for signature in str(row.get("llm_metadata_signatures") or "").split(";"):
            if signature:
                signature_to_runs.setdefault(signature, []).append(str(row["run_id"]))
    for row in rows:
        sharing = set()
        for signature in str(row.get("llm_metadata_signatures") or "").split(";"):
            sharing.update(signature_to_runs.get(signature, []))
        row["runs_sharing_metadata_signatures"] = ";".join(sorted(sharing))
    output = results_root / "llm_call_summary.csv"
    pd.DataFrame(rows, columns=columns).to_csv(output, index=False)
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

    print(f"Final comparison table: {output_path.resolve()}")
    print(f"Paired fold comparisons: {paired_path.resolve()}")
    print(f"LLM call summary: {llm_summary_path.resolve()}")
    print(f"Failed runs: {failed_runs_path.resolve()}")
    print(f"Completed runs aggregated: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
