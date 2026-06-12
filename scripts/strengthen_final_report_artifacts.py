from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from credit_risk_fs.evaluation.drift import calculate_psi  # noqa: E402
from credit_risk_fs.feature_metadata.builder import infer_semantic_group  # noqa: E402
from credit_risk_fs.reporting.markdown_report import (  # noqa: E402
    build_cross_dataset_summary_markdown,
    build_dataset_report_markdown,
)


DATASETS = ("homecredit", "lendingclub")
RESULTS_ROOT = Path("results")
REPORTS_DIR = Path("reports")
HIGH_PSI_THRESHOLD = 0.25
LENDINGCLUB_PROCESSED = Path("data/lendingclub/processed/application_train.csv")


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path(str(path_text).replace("\\", "/"))


def _load_matrix(dataset: str) -> pd.DataFrame:
    matrix = _read_csv(RESULTS_ROOT / dataset / "matrix_runs.csv")
    if matrix.empty:
        return matrix
    matrix = matrix[matrix["status"].eq("completed")].copy()
    matrix["output_folder"] = matrix["output_folder"].map(_normalize_path)
    return matrix


def _selected_features(run_folder: Path) -> pd.DataFrame:
    for rel in ["features/final_selected_features.csv", "selected_feature_sets/final_selected_features.csv"]:
        frame = _read_csv(run_folder / rel)
        if frame.empty:
            continue
        feature_col = "feature" if "feature" in frame.columns else "feature_name"
        if feature_col not in frame.columns:
            continue
        frame = frame.copy()
        frame["feature"] = frame[feature_col]
        return frame[["feature"]].drop_duplicates()
    return pd.DataFrame(columns=["feature"])


def _selected_psi(run_folder: Path) -> pd.DataFrame:
    frame = _read_csv(run_folder / "results" / "selected_feature_psi.csv")
    if frame.empty:
        return pd.DataFrame(columns=["feature", "psi_dev_oot"])
    return frame.rename(columns={"feature_name": "feature", "psi": "psi_dev_oot"})[["feature", "psi_dev_oot"]]


def _llm_ranking(run_folder: Path, model: str) -> pd.DataFrame:
    rankings = _read_csv(run_folder / "features" / "llm_rankings_summary.csv")
    if rankings.empty:
        rankings = _read_csv(run_folder / "feature_rankings" / "llm_rankings_summary.csv")
    if rankings.empty:
        return pd.DataFrame(columns=["feature", "llm_rank"])
    rankings = rankings[rankings["scope"].astype(str).eq("final_dev")].copy()
    if rankings.empty:
        return pd.DataFrame(columns=["feature", "llm_rank"])
    rankings["llm_rank"] = pd.to_numeric(rankings["rank"], errors="coerce")
    rankings = rankings[rankings["llm_rank"].le(100)].copy()
    if rankings.empty:
        return pd.DataFrame(columns=["feature", "llm_rank"])
    rankings = rankings.rename(columns={"feature_name": "feature"})
    return rankings[["feature", "llm_rank"]].drop_duplicates("feature")


def _feature_metadata(dataset: str) -> pd.DataFrame:
    evidence = _read_csv(RESULTS_ROOT / dataset / "feature_level_evidence.csv")
    if evidence.empty:
        return pd.DataFrame(columns=["feature", "semantic_group", "source_table"])
    frame = evidence.rename(columns={"feature_name": "feature"}).copy()
    return frame[["feature", "semantic_group", "source_table"]].drop_duplicates("feature")


def _psi_flag(value: Any) -> str:
    if pd.isna(value):
        return "unavailable"
    value = float(value)
    if value >= 0.25:
        return "high"
    if value >= 0.1:
        return "moderate"
    return "low"


def _frame_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows available."
    text_df = df.fillna("").astype(str)
    columns = list(text_df.columns)
    rows = text_df.values.tolist()
    widths = [
        max(len(str(column)), *(len(str(row[idx])) for row in rows))
        for idx, column in enumerate(columns)
    ]
    header = "| " + " | ".join(str(column).ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _lendingclub_candidate_psi(features: set[str]) -> tuple[dict[str, float], dict[str, str]]:
    if not LENDINGCLUB_PROCESSED.exists():
        return {}, {feature: "processed_safe_frame_missing" for feature in features}
    header = pd.read_csv(LENDINGCLUB_PROCESSED, nrows=0).columns.tolist()
    base_cols = [col for col in ["recent_decision"] if col in header]
    available = sorted(feature for feature in features if feature in header)
    missing = {feature: "feature_not_in_processed_safe_frame" for feature in features - set(available)}
    if "recent_decision" not in base_cols:
        missing.update({feature: "recent_decision_missing_from_processed_safe_frame" for feature in available})
        return {}, missing
    frame = pd.read_csv(LENDINGCLUB_PROCESSED, usecols=base_cols + available, low_memory=False)
    recent = pd.to_numeric(frame["recent_decision"], errors="coerce")
    dev_mask = recent.ge(-1795) & recent.lt(-1065)
    oot_mask = recent.ge(-1065) & recent.le(-730)
    psi: dict[str, float] = {}
    for feature in available:
        dev = pd.to_numeric(frame.loc[dev_mask, feature], errors="coerce")
        oot = pd.to_numeric(frame.loc[oot_mask, feature], errors="coerce")
        if dev.notna().sum() == 0 or oot.notna().sum() == 0:
            missing[feature] = "numeric_dev_oot_values_unavailable"
            continue
        value = calculate_psi(dev, oot)
        if pd.isna(value):
            missing[feature] = "psi_unavailable_constant_or_low_variance"
            continue
        psi[feature] = float(value)
    return psi, missing


def build_llm_top100_candidate_psi(dataset: str) -> pd.DataFrame:
    matrix = _load_matrix(dataset)
    metadata = _feature_metadata(dataset)
    llm_runs = matrix[matrix["selector"].isin(["llm", "llm_then_mrmr", "llm_then_boruta", "stable_core_llm_fill"])].copy()
    all_candidates: set[str] = set()
    candidate_frames: list[tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for _, run in llm_runs.iterrows():
        run_folder = Path(run["output_folder"])
        ranking = _llm_ranking(run_folder, str(run["model"]))
        selected = _selected_features(run_folder)
        selected_psi = _selected_psi(run_folder)
        all_candidates.update(ranking["feature"].dropna().astype(str).tolist())
        candidate_frames.append((run, ranking, selected, selected_psi))

    lc_psi: dict[str, float] = {}
    lc_missing: dict[str, str] = {}
    if dataset == "lendingclub":
        lc_psi, lc_missing = _lendingclub_candidate_psi(all_candidates)

    rows: list[dict[str, Any]] = []
    for run, ranking, selected, selected_psi in candidate_frames:
        selected_set = set(selected["feature"].astype(str).tolist())
        selected_psi_map = dict(zip(selected_psi["feature"].astype(str), pd.to_numeric(selected_psi["psi_dev_oot"], errors="coerce")))
        joined = ranking.merge(metadata, on="feature", how="left")
        for item in joined.to_dict("records"):
            feature = str(item["feature"])
            selected_flag = feature in selected_set
            psi_value = selected_psi_map.get(feature)
            missing_reason = ""
            semantic_group = item.get("semantic_group")
            if dataset == "lendingclub":
                semantic_group = infer_semantic_group(feature)
            if dataset == "lendingclub" and feature in lc_psi:
                psi_value = lc_psi[feature]
            elif pd.isna(psi_value):
                if dataset == "lendingclub":
                    missing_reason = lc_missing.get(feature, "psi_not_available_for_candidate")
                else:
                    missing_reason = "DEV/OOT design matrix unavailable for LLM rejected candidates; selected-feature PSI exists only for final selected features"
            rows.append(
                {
                    "dataset": dataset,
                    "model": run["model"],
                    "selector": run["selector"],
                    "run_id": run["run_id"],
                    "feature": feature,
                    "llm_rank": int(item["llm_rank"]) if pd.notna(item["llm_rank"]) else pd.NA,
                    "in_llm_top100": True,
                    "in_final_selected_set": selected_flag,
                    "selected_by_downstream_stat_selector": bool(selected_flag and str(run["selector"]) in {"llm_then_mrmr", "llm_then_boruta"}),
                    "semantic_group": semantic_group,
                    "source_table": item.get("source_table"),
                    "psi_dev_oot": psi_value,
                    "psi_flag": _psi_flag(psi_value),
                    "missing_from_dev_oot_reason": missing_reason,
                }
            )

    frame = pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "model",
            "selector",
            "run_id",
            "feature",
            "llm_rank",
            "in_llm_top100",
            "in_final_selected_set",
            "selected_by_downstream_stat_selector",
            "semantic_group",
            "source_table",
            "psi_dev_oot",
            "psi_flag",
            "missing_from_dev_oot_reason",
        ],
    )
    out_dir = RESULTS_ROOT / dataset / "analysis" / "feature_level_drift"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["model", "selector", "llm_rank", "feature"]).to_csv(out_dir / "llm_top100_candidate_psi.csv", index=False)
    return frame


def update_llm_then_mrmr_breakdown(dataset: str, top100: pd.DataFrame) -> pd.DataFrame:
    out_dir = RESULTS_ROOT / dataset / "analysis" / "feature_level_drift"
    subset = top100[top100["selector"].eq("llm_then_mrmr")].copy()
    if subset.empty:
        return pd.DataFrame()
    breakdown = subset[
        [
            "dataset",
            "model",
            "run_id",
            "feature",
            "in_llm_top100",
            "in_final_selected_set",
            "psi_dev_oot",
            "semantic_group",
            "source_table",
            "missing_from_dev_oot_reason",
        ]
    ].rename(columns={"in_llm_top100": "in_llm_top_pool"})
    breakdown.to_csv(out_dir / "llm_then_mrmr_drift_source_breakdown.csv", index=False)
    return breakdown


def relabel_lendingclub_semantic_redundancy() -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = _load_matrix("lendingclub")
    subset = matrix[matrix["selector"].isin(["llm", "llm_then_mrmr", "stable_core_llm_fill", "mrmr"])].copy()
    rows: list[dict[str, Any]] = []
    before_after_rows: list[dict[str, Any]] = []
    all_features: set[str] = set()
    selected_by_run: dict[str, pd.DataFrame] = {}

    for _, run in subset.iterrows():
        folder = Path(run["output_folder"])
        selected_raw = _read_csv(folder / "features" / "final_selected_features.csv")
        if selected_raw.empty:
            selected_raw = _read_csv(folder / "selected_feature_sets" / "final_selected_features.csv")
        feature_col = "feature" if "feature" in selected_raw.columns else "feature_name"
        selected = selected_raw.copy()
        selected["feature"] = selected[feature_col]
        selected["semantic_group_before"] = selected.get("semantic_group", pd.Series("unknown", index=selected.index)).fillna("unknown")
        selected["semantic_group"] = selected["feature"].map(lambda feature: infer_semantic_group(str(feature)))
        selected_by_run[str(run["run_id"])] = selected[["feature", "semantic_group", "semantic_group_before"]].drop_duplicates("feature")
        all_features.update(selected["feature"].dropna().astype(str).tolist())
        for item in selected_by_run[str(run["run_id"])].to_dict("records"):
            before_after_rows.append(
                {
                    "run_id": run["run_id"],
                    "model": run["model"],
                    "selector": run["selector"],
                    "experiment_type": run["experiment_type"],
                    "feature": item["feature"],
                    "semantic_group_before": item["semantic_group_before"],
                    "semantic_group_after": item["semantic_group"],
                }
            )

    corr_source = pd.DataFrame()
    if LENDINGCLUB_PROCESSED.exists() and all_features:
        header = pd.read_csv(LENDINGCLUB_PROCESSED, nrows=0).columns.tolist()
        usecols = sorted(all_features.intersection(header))
        if usecols:
            corr_source = pd.read_csv(LENDINGCLUB_PROCESSED, usecols=usecols, low_memory=False).apply(pd.to_numeric, errors="coerce")

    for _, run in subset.iterrows():
        selected = selected_by_run.get(str(run["run_id"]), pd.DataFrame())
        count = len(selected)
        group_counts = selected["semantic_group"].value_counts() if count else pd.Series(dtype=int)
        entropy = math.nan
        if count:
            probs = group_counts / group_counts.sum()
            entropy = float(-(probs * probs.map(math.log)).sum())
        correlations: list[float] = []
        if not corr_source.empty and count:
            for _, group in selected.groupby("semantic_group"):
                features = [feature for feature in group["feature"].tolist() if feature in corr_source.columns]
                if len(features) < 2:
                    continue
                corr = corr_source[features].corr().abs()
                for i, left in enumerate(features):
                    for right in features[i + 1 :]:
                        value = corr.loc[left, right]
                        if pd.notna(value):
                            correlations.append(float(value))
        avg_corr = float(pd.Series(correlations).mean()) if correlations else math.nan
        max_corr = float(pd.Series(correlations).max()) if correlations else math.nan
        largest_share = float(group_counts.max() / count) if count else math.nan
        if pd.notna(max_corr) and max_corr >= 0.9:
            risk_flag = "high_max_correlation"
        elif pd.notna(avg_corr) and avg_corr >= 0.75:
            risk_flag = "high_average_correlation"
        elif pd.notna(largest_share) and largest_share >= 0.75:
            risk_flag = "high_semantic_concentration"
        elif pd.notna(max_corr) and max_corr >= 0.75:
            risk_flag = "moderate"
        else:
            risk_flag = "low"
        rows.append(
            {
                "dataset": "lendingclub",
                "model": run["model"],
                "selector": run["selector"],
                "selected feature count": count,
                "number of semantic groups": int(group_counts.shape[0]) if count else 0,
                "semantic group entropy if easy": entropy,
                "largest group share": largest_share,
                "average within-group absolute correlation": avg_corr,
                "max within-group absolute correlation": max_corr,
                "redundancy risk flag": risk_flag,
            }
        )

    out_dir = RESULTS_ROOT / "lendingclub" / "analysis" / "semantic_redundancy"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).sort_values(["model", "selector"])
    mapping = pd.DataFrame(before_after_rows).sort_values(["model", "selector", "feature"])
    coverage = (
        mapping.groupby(["run_id", "model", "selector", "experiment_type", "semantic_group_after"], as_index=False)
        .size()
        .rename(columns={"semantic_group_after": "semantic_group", "size": "feature_count"})
    )
    totals = coverage.groupby("run_id")["feature_count"].transform("sum")
    coverage["dataset_name"] = "lendingclub"
    coverage["feature_ratio"] = coverage["feature_count"] / totals
    coverage = coverage[
        [
            "dataset_name",
            "run_id",
            "model",
            "selector",
            "experiment_type",
            "semantic_group",
            "feature_count",
            "feature_ratio",
        ]
    ].sort_values(["model", "selector", "feature_ratio"], ascending=[True, True, False])
    table.to_csv(out_dir / "semantic_coverage_redundancy_by_pipeline.csv", index=False)
    mapping.to_csv(out_dir / "semantic_group_mapping_before_after.csv", index=False)
    coverage.to_csv(out_dir / "semantic_coverage_by_pipeline_relabelled.csv", index=False)
    return table, mapping


def update_semantic_redundancy_plots(table: pd.DataFrame) -> None:
    plots_dir = RESULTS_ROOT / "lendingclub" / "analysis" / "semantic_redundancy" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    if table.empty or table["selector"].nunique() <= 1:
        manifest.append(
            {
                "plot_file": "largest_semantic_group_share_by_selector.png",
                "source_table": "semantic_coverage_redundancy_by_pipeline.csv",
                "rows_used": len(table),
                "columns_used": "selector;largest group share",
                "purpose": "Compare semantic concentration after LendingClub relabeling.",
                "status": "skipped",
                "skip_reason": "not enough selector categories",
            }
        )
    else:
        plot_df = table.copy()
        plot_df["pipeline"] = plot_df["model"] + "/" + plot_df["selector"]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(plot_df["pipeline"], plot_df["largest group share"], color="#4c78a8")
        ax.set_title("Largest Semantic Group Share by Pipeline")
        ax.set_ylabel("Largest group share")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(plots_dir / "largest_semantic_group_share_by_selector.png", dpi=160)
        plt.close(fig)
        manifest.append(
            {
                "plot_file": "largest_semantic_group_share_by_selector.png",
                "source_table": "semantic_coverage_redundancy_by_pipeline.csv",
                "rows_used": len(table),
                "columns_used": "selector;largest group share",
                "purpose": "Compare semantic concentration after LendingClub relabeling.",
                "status": "created",
                "skip_reason": "",
            }
        )
    pd.DataFrame(manifest).to_csv(plots_dir / "plot_manifest.csv", index=False)


def write_feature_level_drift_report() -> None:
    lines = [
        "# Feature-Level Drift Analysis",
        "",
        "## Updated LLM Top-100 Candidate Evidence",
        "",
        "This report now separates LLM nomination from downstream statistical selection. `llm_top100_candidate_psi.csv` lists saved final-dev LLM top-100 candidates and flags whether each candidate survived into the final selected set. No model retraining or selector rerun was performed.",
        "",
        "For LendingClub, top-100 candidate PSI was computed from the processed safe DEV/OOT frame where numeric candidate columns were available. For Home Credit, the full DEV/OOT design matrix for rejected LLM candidates is not present in saved artifacts, so rejected-candidate PSI is explicitly marked missing; selected-feature PSI remains available from per-run artifacts.",
        "",
        "## Interpretation",
        "",
        "- Did LLM nominate high-drift features? LendingClub candidate PSI shows the LLM top pool is mostly low-drift where PSI could be computed. Home Credit cannot fully answer this for rejected candidates without a saved DEV/OOT design matrix.",
        "- Did mRMR/Boruta keep or reject high-drift LLM candidates? The updated breakdown keeps `in_final_selected_set` and `selected_by_downstream_stat_selector`; LendingClub can evaluate this directly, while Home Credit can only evaluate final selected features.",
        "- Are high-PSI features caused by LLM nomination or downstream statistical selection? Current selected-feature evidence does not support a broad claim that LLM nomination itself causes high PSI. Where PSI is missing for rejected candidates, the report marks the limitation rather than inventing values.",
        "- Dataset difference: LendingClub has enough processed safe artifact coverage for a stronger top-pool PSI audit; Home Credit requires targeted artifact generation to audit rejected candidates.",
        "",
    ]
    for dataset in DATASETS:
        path = RESULTS_ROOT / dataset / "analysis" / "feature_level_drift" / "llm_top100_candidate_psi.csv"
        table = _read_csv(path)
        lines.extend([f"## {dataset}", ""])
        if table.empty:
            lines.extend(["Top-100 candidate table unavailable.", ""])
            continue
        summary = (
            table.groupby(["selector", "psi_flag"], dropna=False)
            .size()
            .reset_index(name="candidate_count")
            .sort_values(["selector", "psi_flag"])
        )
        lines.extend([_frame_to_markdown(summary), ""])
        missing = table["missing_from_dev_oot_reason"].fillna("").astype(str)
        missing_counts = missing[missing.ne("")].value_counts().reset_index()
        missing_counts.columns = ["missing_from_dev_oot_reason", "candidate_count"]
        lines.extend(["Missing PSI reasons:", ""])
        lines.extend([_frame_to_markdown(missing_counts) if not missing_counts.empty else "No missing PSI reasons.", ""])
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "feature_level_drift_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def write_semantic_report(table: pd.DataFrame, mapping: pd.DataFrame) -> None:
    before_other = int(mapping["semantic_group_before"].eq("other").sum()) if not mapping.empty else 0
    after_other = int(mapping["semantic_group_after"].eq("other").sum()) if not mapping.empty else 0
    lines = [
        "# Semantic Coverage and Redundancy",
        "",
        "## LendingClub Mapping Update",
        "",
        f"LendingClub semantic grouping was too coarse. The report-layer mapping reduced selected-feature `other` labels from {before_other} to {after_other} rows in the reviewed selected-feature artifacts. This does not change feature selection results or model metrics.",
        "",
        "## Updated LendingClub Redundancy Table",
        "",
        _frame_to_markdown(table) if not table.empty else "Updated table unavailable.",
        "",
        "## Interpretation",
        "",
        "LLM pipelines can support business-concept coverage, but the strength of that claim is dataset and metadata-rule dependent. The LendingClub update makes concepts such as FICO score, income capacity, revolving utilization, bankcard capacity, credit-history length, recent inquiries, account-opening activity, mortgage history, derogatory events, loan terms, and exposure amount visible in the report. If AUC differences are small, this semantic interpretability can be a defensible secondary advantage, but it should not be overstated as universal superiority.",
    ]
    (REPORTS_DIR / "semantic_coverage_and_redundancy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    top100_by_dataset = {}
    for dataset in DATASETS:
        top100 = build_llm_top100_candidate_psi(dataset)
        top100_by_dataset[dataset] = top100
        update_llm_then_mrmr_breakdown(dataset, top100)

    semantic_table, mapping = relabel_lendingclub_semantic_redundancy()
    update_semantic_redundancy_plots(semantic_table)
    write_feature_level_drift_report()
    write_semantic_report(semantic_table, mapping)

    (REPORTS_DIR / "cross_dataset_summary.md").write_text(build_cross_dataset_summary_markdown(), encoding="utf-8")
    for dataset in DATASETS:
        (REPORTS_DIR / f"{dataset}_report.md").write_text(build_dataset_report_markdown(dataset), encoding="utf-8")

    print("strengthened final report artifacts")
    for dataset, top100 in top100_by_dataset.items():
        missing = int(top100["missing_from_dev_oot_reason"].fillna("").astype(str).ne("").sum()) if not top100.empty else 0
        print(f"{dataset}: top100 rows={len(top100)}, missing_psi_rows={missing}")
    print(f"lendingclub semantic mapping rows={len(mapping)}")


if __name__ == "__main__":
    main()
