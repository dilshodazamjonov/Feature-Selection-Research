from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
OUT_DIR = RESULTS_ROOT / "cross_dataset_v2" / "analysis"
REPORT_PATH = PROJECT_ROOT / "reports" / "cross_dataset_v2_analysis.md"

DATASETS = ("homecredit", "lendingclub_v2")
BASELINE_SELECTORS = {"mrmr", "boruta", "pca", "domain_rule_baseline"}
LLM_FAMILY_SELECTORS = {"llm", "llm_then_mrmr", "llm_then_boruta", "stable_core_llm_fill"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    table = df.head(max_rows).copy() if max_rows is not None else df.copy()
    for column in table.columns:
        if pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda value: _fmt(value))
    text_df = table.fillna("").astype(str)
    header = "| " + " | ".join(text_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
    rows = [
        "| " + " | ".join(row[column] for column in text_df.columns) + " |"
        for _, row in text_df.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def _load_final() -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        path = RESULTS_ROOT / dataset / "final_comparison_table.csv"
        frame = _read_csv(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["dataset_name"] = dataset
        frame["selector_family"] = frame["selector"].map(
            lambda selector: "llm_family" if selector in LLM_FAMILY_SELECTORS else "statistical"
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_named_table(name: str) -> pd.DataFrame:
    frames = []
    for dataset in DATASETS:
        frame = _read_csv(RESULTS_ROOT / dataset / f"{name}.csv")
        if frame.empty:
            continue
        frame = frame.copy()
        if "dataset_name" not in frame.columns:
            frame.insert(0, "dataset_name", dataset)
        else:
            frame["dataset_name"] = dataset
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _best_rows(final: pd.DataFrame, family: str | None = None) -> pd.DataFrame:
    frame = final.copy()
    if family is not None:
        frame = frame[frame["selector_family"].eq(family)].copy()
    if frame.empty:
        return pd.DataFrame()
    idx = frame.groupby(["dataset_name", "model"])["oot_auc"].idxmax()
    return (
        frame.loc[idx]
        .sort_values(["dataset_name", "model"])
        [
            [
                "dataset_name",
                "model",
                "selector",
                "selector_family",
                "oot_auc",
                "oot_gini",
                "lift_at_10",
                "selected_feature_psi_mean",
                "model_score_psi",
                "nogueira_stability",
                "mean_pairwise_jaccard",
                "semantic_group_jaccard",
                "runtime_seconds",
            ]
        ]
        .reset_index(drop=True)
    )


def _deltas_vs_mrmr(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), group in final.groupby(["dataset_name", "model"]):
        mrmr = group[group["selector"].eq("mrmr")]
        if mrmr.empty:
            continue
        mrmr_row = mrmr.iloc[0]
        for row in group.to_dict("records"):
            rows.append(
                {
                    "dataset_name": dataset,
                    "model": model,
                    "selector": row["selector"],
                    "selector_family": row["selector_family"],
                    "oot_auc": row.get("oot_auc"),
                    "delta_auc_vs_mrmr": row.get("oot_auc") - mrmr_row.get("oot_auc"),
                    "delta_gini_vs_mrmr": row.get("oot_gini") - mrmr_row.get("oot_gini"),
                    "delta_feature_psi_mean_vs_mrmr": row.get("selected_feature_psi_mean")
                    - mrmr_row.get("selected_feature_psi_mean"),
                    "delta_model_score_psi_vs_mrmr": row.get("model_score_psi")
                    - mrmr_row.get("model_score_psi"),
                    "delta_nogueira_vs_mrmr": row.get("nogueira_stability")
                    - mrmr_row.get("nogueira_stability"),
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset_name", "model", "delta_auc_vs_mrmr"], ascending=[True, True, False])


def _family_summary(final: pd.DataFrame) -> pd.DataFrame:
    if final.empty:
        return pd.DataFrame()
    metrics = [
        "oot_auc",
        "oot_gini",
        "lift_at_10",
        "selected_feature_psi_mean",
        "selected_feature_psi_high_drift_ratio",
        "model_score_psi",
        "nogueira_stability",
        "mean_pairwise_jaccard",
        "semantic_group_jaccard",
        "runtime_seconds",
    ]
    return (
        final.groupby(["dataset_name", "model", "selector_family"])[metrics]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["dataset_name", "model", "selector_family"])
    )


def _paired_fold_evidence() -> pd.DataFrame:
    paired = _load_named_table("paired_fold_comparisons")
    if paired.empty:
        return paired
    paired = paired[paired["candidate_selector"].isin(LLM_FAMILY_SELECTORS)].copy()
    paired = paired[paired["metric"].isin(["auc", "gini"])].copy()
    paired["direction"] = paired["mean_delta_candidate_minus_baseline"].map(
        lambda value: "candidate_above_mrmr" if value > 0 else "candidate_below_mrmr" if value < 0 else "tie"
    )
    return paired.sort_values(
        ["dataset_name", "model", "metric", "mean_delta_candidate_minus_baseline"],
        ascending=[True, True, True, False],
    )


def _semantic_summary() -> pd.DataFrame:
    semantic = _load_named_table("semantic_coverage_table")
    if semantic.empty:
        return semantic
    rows = []
    for keys, group in semantic.groupby(["dataset_name", "model", "selector", "experiment_type", "run_id"]):
        dataset, model, selector, experiment_type, run_id = keys
        feature_count = pd.to_numeric(group["feature_count"], errors="coerce").sum()
        ratios = pd.to_numeric(group["feature_ratio"], errors="coerce")
        max_row = group.loc[ratios.idxmax()] if ratios.notna().any() else group.iloc[0]
        rows.append(
            {
                "dataset_name": dataset,
                "model": model,
                "selector": selector,
                "experiment_type": experiment_type,
                "run_id": run_id,
                "semantic_group_count": int(group["semantic_group"].nunique()),
                "selected_feature_count": int(feature_count) if not pd.isna(feature_count) else math.nan,
                "largest_semantic_group": max_row.get("semantic_group"),
                "largest_semantic_group_share": max_row.get("feature_ratio"),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_name", "model", "selector"])


def _run_health() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        matrix = _read_csv(RESULTS_ROOT / dataset / "matrix_runs.csv")
        failed = _read_csv(RESULTS_ROOT / dataset / "failed_runs.csv")
        status_counts = matrix["status"].value_counts().to_dict() if "status" in matrix.columns else {}
        rows.append(
            {
                "dataset_name": dataset,
                "matrix_rows": len(matrix),
                "completed_runs": int(status_counts.get("completed", 0)),
                "failed_runs": len(failed),
                "scheduled_or_pending_runs": int(status_counts.get("scheduled", 0) + status_counts.get("pending", 0)),
            }
        )
    return pd.DataFrame(rows)


def _top_feature_evidence() -> pd.DataFrame:
    evidence = _load_named_table("feature_level_evidence")
    if evidence.empty:
        return evidence
    keep = [
        "dataset_name",
        "feature_name",
        "semantic_group",
        "selected_in_final_run_count",
        "selected_in_llm_family_run_count",
        "selected_in_baseline_run_count",
        "selected_in_lr_run_count",
        "selected_in_catboost_run_count",
        "best_llm_final_dev_rank",
        "best_oot_auc_when_selected",
        "selectors_selected_by",
    ]
    keep = [column for column in keep if column in evidence.columns]
    out = evidence[keep].copy()
    out["selected_in_final_run_count"] = pd.to_numeric(out["selected_in_final_run_count"], errors="coerce")
    out["selected_in_llm_family_run_count"] = pd.to_numeric(out["selected_in_llm_family_run_count"], errors="coerce")
    return out.sort_values(
        ["dataset_name", "selected_in_final_run_count", "selected_in_llm_family_run_count"],
        ascending=[True, False, False],
    )


def _write_outputs(tables: dict[str, pd.DataFrame]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(OUT_DIR / f"{name}.csv", index=False)


def _report(tables: dict[str, pd.DataFrame]) -> str:
    final = tables["final"]
    best_overall = tables["best_overall"]
    best_llm = tables["best_llm_family"]
    deltas = tables["deltas_vs_mrmr"]
    family = tables["family_summary"]
    semantic = tables["semantic_summary"]
    paired = tables["paired_fold_evidence"]
    health = tables["run_health"]
    top_features = tables["top_feature_evidence"]

    llm_advantage = deltas[
        deltas["selector_family"].eq("llm_family") & deltas["delta_auc_vs_mrmr"].gt(0)
    ]
    low_drift = family.sort_values(["dataset_name", "model", "selected_feature_psi_mean"])
    top_feature_sample = top_features.groupby("dataset_name", group_keys=False).head(10)

    lines = [
        "# Cross-Dataset V2 Analysis",
        "",
        "This analysis compares `homecredit` and `lendingclub_v2` using existing completed aggregate artifacts only. It does not train models, rerun the experiment matrix, or call the LLM.",
        "",
        "## Run Completeness",
        "",
        _markdown_table(health),
        "",
        "## Main Interpretation",
        "",
        "- Do not compare raw AUC across datasets as if they were the same task; compare selector behavior within each dataset/model.",
        "- LLM-family selectors are strongest when judged as low-drift, semantically broad first-stage screeners.",
        "- mRMR remains the key exact-stability and non-LLM reference.",
        "- PCA is mostly useful as a caution baseline when it has weaker discrimination or higher drift.",
        "",
        "## Best Overall Runs",
        "",
        _markdown_table(best_overall),
        "",
        "## Best LLM-Family Runs",
        "",
        _markdown_table(best_llm),
        "",
        "## LLM-Family Wins Versus mRMR",
        "",
        _markdown_table(
            llm_advantage[
                [
                    "dataset_name",
                    "model",
                    "selector",
                    "oot_auc",
                    "delta_auc_vs_mrmr",
                    "delta_feature_psi_mean_vs_mrmr",
                    "delta_nogueira_vs_mrmr",
                ]
            ]
        ),
        "",
        "## Family-Level Pattern",
        "",
        _markdown_table(family),
        "",
        "## Lowest Mean Selected-Feature PSI By Dataset/Model",
        "",
        _markdown_table(
            low_drift.groupby(["dataset_name", "model"], group_keys=False).head(2)[
                [
                    "dataset_name",
                    "model",
                    "selector_family",
                    "selected_feature_psi_mean",
                    "model_score_psi",
                    "oot_auc",
                    "nogueira_stability",
                ]
            ]
        ),
        "",
        "## Paired Fold Evidence Versus mRMR",
        "",
        _markdown_table(
            paired[
                [
                    "dataset_name",
                    "model",
                    "candidate_selector",
                    "metric",
                    "mean_delta_candidate_minus_baseline",
                    "ci95_lower",
                    "ci95_upper",
                    "direction",
                ]
            ],
            max_rows=40,
        ),
        "",
        "## Semantic Coverage Summary",
        "",
        _markdown_table(
            semantic[
                [
                    "dataset_name",
                    "model",
                    "selector",
                    "semantic_group_count",
                    "selected_feature_count",
                    "largest_semantic_group",
                    "largest_semantic_group_share",
                ]
            ],
            max_rows=40,
        ),
        "",
        "## Most Repeated Selected Features",
        "",
        _markdown_table(top_feature_sample, max_rows=20),
        "",
        "## Output Tables",
        "",
        f"- `{(OUT_DIR / 'final.csv').relative_to(PROJECT_ROOT).as_posix()}`",
        f"- `{(OUT_DIR / 'best_overall.csv').relative_to(PROJECT_ROOT).as_posix()}`",
        f"- `{(OUT_DIR / 'best_llm_family.csv').relative_to(PROJECT_ROOT).as_posix()}`",
        f"- `{(OUT_DIR / 'deltas_vs_mrmr.csv').relative_to(PROJECT_ROOT).as_posix()}`",
        f"- `{(OUT_DIR / 'family_summary.csv').relative_to(PROJECT_ROOT).as_posix()}`",
        f"- `{(OUT_DIR / 'paired_fold_evidence.csv').relative_to(PROJECT_ROOT).as_posix()}`",
        f"- `{(OUT_DIR / 'semantic_summary.csv').relative_to(PROJECT_ROOT).as_posix()}`",
        f"- `{(OUT_DIR / 'top_feature_evidence.csv').relative_to(PROJECT_ROOT).as_posix()}`",
    ]
    return "\n".join(lines)


def main() -> int:
    final = _load_final()
    if final.empty:
        raise FileNotFoundError("No final_comparison_table.csv files found for configured datasets.")

    tables = {
        "final": final,
        "best_overall": _best_rows(final),
        "best_llm_family": _best_rows(final, family="llm_family"),
        "deltas_vs_mrmr": _deltas_vs_mrmr(final),
        "family_summary": _family_summary(final),
        "paired_fold_evidence": _paired_fold_evidence(),
        "semantic_summary": _semantic_summary(),
        "run_health": _run_health(),
        "top_feature_evidence": _top_feature_evidence(),
    }
    _write_outputs(tables)
    REPORT_PATH.write_text(_report(tables), encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote tables to {OUT_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
