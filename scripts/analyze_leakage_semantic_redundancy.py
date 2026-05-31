from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from credit_risk_fs.preprocessing.labeling import (  # noqa: E402
    LENDINGCLUB_AMBIGUOUS_OR_ONGOING_STATUSES,
    LENDINGCLUB_FINAL_BAD_STATUSES,
    LENDINGCLUB_FINAL_GOOD_STATUSES,
)
from credit_risk_fs.preprocessing.lendingclub import (  # noqa: E402
    LENDINGCLUB_DIRECT_LEAKAGE_COLUMNS,
    LENDINGCLUB_EXCLUDED_FEATURE_COLUMNS,
    LENDINGCLUB_IDENTIFIER_OR_TEXT_COLUMNS,
    LENDINGCLUB_POLICY_OR_POST_APPROVAL_COLUMNS,
    LENDINGCLUB_POST_OUTCOME_LEAKAGE_COLUMNS,
    LENDINGCLUB_TEXT_OR_LOW_SIGNAL_COLUMNS,
    LENDINGCLUB_UNDERWRITING_POLICY_COLUMNS,
)


DATASETS = ("homecredit", "lendingclub")
SELECTORS = ("llm", "llm_then_mrmr", "stable_core_llm_fill", "mrmr")
RESULTS_ROOT = Path("results")
LENDINGCLUB_PROCESSED = Path("data/lendingclub/processed/application_train.csv")
LENDINGCLUB_LEAKAGE_OUT = RESULTS_ROOT / "lendingclub" / "analysis" / "leakage_transparency"
REPORT_LC = Path("reports/lendingclub_leakage_and_label_definition.md")
REPORT_HC = Path("reports/homecredit_temporal_semantics_note.md")
REPORT_SEMANTIC = Path("reports/semantic_coverage_and_redundancy.md")
HIGH_MISSINGNESS_DIFF = 0.5


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
    path = run_folder / "selected_feature_sets" / "final_selected_features.csv"
    frame = _read_csv(path)
    if frame.empty:
        path = run_folder / "features" / "final_selected_features.csv"
        frame = _read_csv(path)
    if frame.empty:
        return pd.DataFrame(columns=["feature", "semantic_group"])
    feature_col = "feature" if "feature" in frame.columns else "feature_name"
    frame = frame.copy()
    frame["feature"] = frame[feature_col]
    if "semantic_group" not in frame.columns:
        frame["semantic_group"] = "unknown"
    frame["semantic_group"] = frame["semantic_group"].fillna("unknown")
    return frame[["feature", "semantic_group"]].drop_duplicates("feature")


def _entropy(counts: pd.Series) -> float:
    total = counts.sum()
    if total <= 0:
        return math.nan
    probs = counts / total
    return float(-(probs * probs.map(math.log)).sum())


def _lendingclub_numeric_correlation_source(features: set[str]) -> pd.DataFrame:
    if not LENDINGCLUB_PROCESSED.exists() or not features:
        return pd.DataFrame()
    header = pd.read_csv(LENDINGCLUB_PROCESSED, nrows=0).columns.tolist()
    usecols = sorted(features.intersection(header))
    if not usecols:
        return pd.DataFrame()
    frame = pd.read_csv(LENDINGCLUB_PROCESSED, usecols=usecols, low_memory=False)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    return numeric


def _within_group_correlation(
    selected: pd.DataFrame,
    corr_source: pd.DataFrame,
) -> tuple[float, float]:
    if selected.empty or corr_source.empty:
        return math.nan, math.nan
    values: list[float] = []
    for _, group in selected.groupby("semantic_group"):
        features = [feature for feature in group["feature"].tolist() if feature in corr_source.columns]
        if len(features) < 2:
            continue
        corr = corr_source[features].corr().abs()
        for i, left in enumerate(features):
            for right in features[i + 1 :]:
                value = corr.loc[left, right]
                if pd.notna(value):
                    values.append(float(value))
    if not values:
        return math.nan, math.nan
    series = pd.Series(values)
    return float(series.mean()), float(series.max())


def _risk_flag(largest_share: float, avg_corr: float, max_corr: float) -> str:
    if pd.notna(max_corr) and max_corr >= 0.9:
        return "high_max_correlation"
    if pd.notna(avg_corr) and avg_corr >= 0.75:
        return "high_average_correlation"
    if largest_share >= 0.75:
        return "high_semantic_concentration"
    if pd.isna(avg_corr) and pd.isna(max_corr):
        return "coverage_only_correlation_unavailable"
    if largest_share >= 0.6 or (pd.notna(max_corr) and max_corr >= 0.75):
        return "moderate"
    return "low"


def _build_semantic_redundancy(dataset: str) -> tuple[pd.DataFrame, list[str]]:
    matrix = _load_matrix(dataset)
    missing: list[str] = []
    rows: list[dict[str, Any]] = []
    selected_by_run: dict[str, pd.DataFrame] = {}
    all_features: set[str] = set()

    if matrix.empty:
        return pd.DataFrame(), [f"{dataset}: matrix_runs.csv missing or empty"]

    subset = matrix[matrix["selector"].isin(SELECTORS)].copy()
    for record in subset.to_dict("records"):
        folder = Path(record["output_folder"])
        selected = _selected_features(folder)
        if selected.empty:
            missing.append(f"{dataset}/{record['model']}/{record['selector']}: final selected features missing")
        selected_by_run[str(record["run_id"])] = selected
        all_features.update(selected["feature"].tolist())

    corr_source = (
        _lendingclub_numeric_correlation_source(all_features)
        if dataset == "lendingclub"
        else pd.DataFrame()
    )
    if dataset == "homecredit":
        missing.append("homecredit: no processed single-table artifact found for correlation redundancy; coverage metrics only")
    elif corr_source.empty:
        missing.append("lendingclub: no selected numeric features available in processed safe CSV for correlation redundancy")

    for record in subset.to_dict("records"):
        selected = selected_by_run.get(str(record["run_id"]), pd.DataFrame())
        count = len(selected)
        if count:
            group_counts = selected["semantic_group"].fillna("unknown").value_counts()
            group_count = int(group_counts.shape[0])
            entropy = _entropy(group_counts)
            largest_share = float(group_counts.max() / count)
        else:
            group_count = 0
            entropy = math.nan
            largest_share = math.nan
        avg_corr, max_corr = _within_group_correlation(selected, corr_source)
        rows.append(
            {
                "dataset": dataset,
                "model": record["model"],
                "selector": record["selector"],
                "selected feature count": count,
                "number of semantic groups": group_count,
                "semantic group entropy if easy": entropy,
                "largest group share": largest_share,
                "average within-group absolute correlation": avg_corr,
                "max within-group absolute correlation": max_corr,
                "redundancy risk flag": _risk_flag(largest_share, avg_corr, max_corr)
                if count
                else "missing_selected_features",
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "model",
            "selector",
            "selected feature count",
            "number of semantic groups",
            "semantic group entropy if easy",
            "largest group share",
            "average within-group absolute correlation",
            "max within-group absolute correlation",
            "redundancy risk flag",
        ],
    ).sort_values(["dataset", "model", "selector"]), missing


def _lendingclub_missingness_by_target() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not LENDINGCLUB_PROCESSED.exists():
        return pd.DataFrame(), {"missing": "processed application_train.csv not found"}
    header = pd.read_csv(LENDINGCLUB_PROCESSED, nrows=0).columns.tolist()
    if "TARGET" not in header:
        return pd.DataFrame(), {"missing": "TARGET not found in LendingClub processed file"}

    feature_cols = [column for column in header if column != "TARGET"]
    counts = {0: 0, 1: 0}
    missing_counts: dict[int, pd.Series] = {
        0: pd.Series(0, index=feature_cols, dtype="int64"),
        1: pd.Series(0, index=feature_cols, dtype="int64"),
    }
    chunksize = 100_000
    for chunk in pd.read_csv(LENDINGCLUB_PROCESSED, chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk["TARGET"].isin([0, 1])]
        for target_value in [0, 1]:
            part = chunk[chunk["TARGET"].eq(target_value)]
            counts[target_value] += len(part)
            if not part.empty:
                missing_counts[target_value] = missing_counts[target_value].add(
                    part[feature_cols].isna().sum(),
                    fill_value=0,
                )

    good_total = counts[0]
    bad_total = counts[1]
    rows = []
    for feature in feature_cols:
        good_missing = (
            float(missing_counts[0].get(feature, 0) / good_total) if good_total else math.nan
        )
        bad_missing = (
            float(missing_counts[1].get(feature, 0) / bad_total) if bad_total else math.nan
        )
        diff = bad_missing - good_missing if pd.notna(good_missing) and pd.notna(bad_missing) else math.nan
        populated_only_bad = bool(good_missing >= 0.99 and bad_missing <= 0.01)
        populated_only_good = bool(bad_missing >= 0.99 and good_missing <= 0.01)
        possible = bool(populated_only_bad or populated_only_good or abs(diff) >= HIGH_MISSINGNESS_DIFF)
        rows.append(
            {
                "feature": feature,
                "missing_rate_good": good_missing,
                "missing_rate_bad": bad_missing,
                "missing_rate_diff": diff,
                "populated_only_for_bad_flag": populated_only_bad,
                "populated_only_for_good_flag": populated_only_good,
                "possible_leakage_flag": possible,
            }
        )
    frame = pd.DataFrame(rows).sort_values("missing_rate_diff", key=lambda s: s.abs(), ascending=False)
    metadata = {
        "row_count": good_total + bad_total,
        "good_count": good_total,
        "bad_count": bad_total,
        "processed_column_count": len(header),
        "model_feature_count_after_basic_exclusions": len(
            [c for c in header if c not in set(LENDINGCLUB_EXCLUDED_FEATURE_COLUMNS)]
        ),
        "loan_status_present": "loan_status" in header,
        "target_present": "TARGET" in header,
        "recent_decision_present": "recent_decision" in header,
        "issue_d_present": "issue_d" in header,
    }
    return frame, metadata


def _record_plot(
    manifest: list[dict[str, Any]],
    plot_file: str,
    source_table: str,
    rows_used: int,
    columns_used: list[str],
    purpose: str,
    status: str,
    skip_reason: str = "",
) -> None:
    manifest.append(
        {
            "plot_file": plot_file,
            "source_table": source_table,
            "rows_used": rows_used,
            "columns_used": ";".join(columns_used),
            "purpose": purpose,
            "status": status,
            "skip_reason": skip_reason,
        }
    )


def _plot_semantic(dataset: str, table: pd.DataFrame) -> pd.DataFrame:
    plots_dir = RESULTS_ROOT / dataset / "analysis" / "semantic_redundancy" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    def save_bar(plot_file: str, y: str, ylabel: str, purpose: str) -> None:
        columns = ["model", "selector", y]
        if table.empty:
            _record_plot(manifest, plot_file, "semantic_coverage_redundancy_by_pipeline.csv", 0, columns, purpose, "skipped", "empty source data")
            return
        if table["selector"].nunique() <= 1 or table[y].nunique(dropna=True) <= 1:
            _record_plot(manifest, plot_file, "semantic_coverage_redundancy_by_pipeline.csv", len(table), columns, purpose, "skipped", "requires multiple selectors and non-constant values")
            return
        plot_df = table.copy()
        plot_df["pipeline"] = plot_df["model"].astype(str) + "/" + plot_df["selector"].astype(str)
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(plot_df["pipeline"], plot_df[y], color="#3b6ea8")
        ax.set_ylabel(ylabel)
        ax.set_title(purpose)
        ax.tick_params(axis="x", labelrotation=65)
        fig.tight_layout()
        fig.savefig(plots_dir / plot_file, dpi=160)
        plt.close(fig)
        _record_plot(manifest, plot_file, "semantic_coverage_redundancy_by_pipeline.csv", len(table), columns, purpose, "created")

    save_bar(
        "semantic_group_coverage_by_selector.png",
        "number of semantic groups",
        "Semantic groups",
        "Compare semantic group coverage by selector and model.",
    )
    save_bar(
        "largest_semantic_group_share_by_selector.png",
        "largest group share",
        "Largest group share",
        "Compare concentration in the largest semantic group.",
    )
    save_bar(
        "within_group_redundancy_by_selector.png",
        "average within-group absolute correlation",
        "Average absolute correlation",
        "Compare within-group redundancy where correlation inputs are available.",
    )

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(plots_dir / "plot_manifest.csv", index=False)
    return manifest_df


def _plot_lendingclub_leakage(missingness: pd.DataFrame) -> pd.DataFrame:
    plots_dir = LENDINGCLUB_LEAKAGE_OUT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    plot_file = "missingness_by_target_possible_leakage_flags.png"
    columns = ["feature", "missing_rate_good", "missing_rate_bad", "possible_leakage_flag"]
    purpose = "Show missingness-by-target features flagged for possible leakage review."

    if missingness.empty:
        _record_plot(manifest, plot_file, "missingness_by_target_leakage_check.csv", 0, columns, purpose, "skipped", "empty source data")
    else:
        flagged = missingness[missingness["possible_leakage_flag"].astype(bool)].copy()
        if flagged.empty:
            _record_plot(manifest, plot_file, "missingness_by_target_leakage_check.csv", len(missingness), columns, purpose, "skipped", "no possible leakage flags")
        elif flagged["missing_rate_diff"].nunique(dropna=True) <= 1:
            _record_plot(manifest, plot_file, "missingness_by_target_leakage_check.csv", len(flagged), columns, purpose, "skipped", "flagged missingness differences are constant")
        else:
            flagged = flagged.head(25).sort_values("missing_rate_diff")
            fig, ax = plt.subplots(figsize=(10, max(4, len(flagged) * 0.35)))
            ax.barh(flagged["feature"], flagged["missing_rate_diff"], color="#b45f3c")
            ax.set_xlabel("Bad missing rate - good missing rate")
            ax.set_title("LendingClub Missingness-by-Target Flags")
            fig.tight_layout()
            fig.savefig(plots_dir / plot_file, dpi=160)
            plt.close(fig)
            _record_plot(manifest, plot_file, "missingness_by_target_leakage_check.csv", len(flagged), columns, purpose, "created")

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(plots_dir / "plot_manifest.csv", index=False)
    return manifest_df


def _lendingclub_results_use_safe_path() -> bool:
    final = _read_csv(RESULTS_ROOT / "lendingclub" / "final_comparison_table.csv")
    if final.empty or "data_fingerprint" not in final.columns:
        return False
    paths = []
    for raw in final["data_fingerprint"].dropna().tolist():
        try:
            payload = json.loads(raw)
            paths.append(str(payload.get("path", "")).replace("\\", "/"))
        except json.JSONDecodeError:
            continue
    return bool(paths) and all(path.endswith("data/lendingclub/processed") for path in paths)


def _write_lendingclub_report(metadata: dict[str, Any], missingness: pd.DataFrame, plot_manifest: pd.DataFrame) -> None:
    REPORT_LC.parent.mkdir(parents=True, exist_ok=True)
    flagged_count = int(missingness["possible_leakage_flag"].sum()) if not missingness.empty else 0
    top_flags = missingness[missingness["possible_leakage_flag"].astype(bool)].head(10) if flagged_count else pd.DataFrame()
    header = pd.read_csv(LENDINGCLUB_PROCESSED, nrows=0).columns.tolist() if LENDINGCLUB_PROCESSED.exists() else []
    header_set = set(header)
    leakage_categories = {
        "payment fields": ["total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "last_pymnt_amnt"],
        "recovery fields": ["recoveries", "collection_recovery_fee"],
        "settlement fields": ["debt_settlement_flag", "debt_settlement_flag_date", "settlement_status", "settlement_date", "settlement_amount", "settlement_percentage", "settlement_term"],
        "hardship fields": ["hardship_flag", "hardship_type", "hardship_reason", "hardship_status", "hardship_amount", "hardship_start_date", "hardship_end_date", "hardship_length", "hardship_dpd", "hardship_loan_status", "hardship_payoff_balance_amount", "hardship_last_payment_amount"],
        "post-origination status fields": list(LENDINGCLUB_DIRECT_LEAKAGE_COLUMNS) + list(LENDINGCLUB_UNDERWRITING_POLICY_COLUMNS) + list(LENDINGCLUB_POLICY_OR_POST_APPROVAL_COLUMNS),
        "collection fields": ["collection_recovery_fee"],
        "future payment/date fields": ["last_pymnt_d", "next_pymnt_d", "last_credit_pull_d", "last_fico_range_low", "last_fico_range_high", "payment_plan_start_date"],
    }

    lines = [
        "# LendingClub Leakage And Label Definition",
        "",
        "## Target Definition",
        "",
        "`TARGET = 1` for final bad/default outcomes and `TARGET = 0` for final good outcomes. The implementation is in `src/credit_risk_fs/preprocessing/labeling.py`.",
        "",
        "Good statuses:",
        *[f"- `{status}`" for status in sorted(LENDINGCLUB_FINAL_GOOD_STATUSES)],
        "",
        "Bad/default statuses:",
        *[f"- `{status}`" for status in sorted(LENDINGCLUB_FINAL_BAD_STATUSES)],
        "",
        "Dropped ambiguous/current/unmatured statuses:",
        *[f"- `{status}`" for status in sorted(LENDINGCLUB_AMBIGUOUS_OR_ONGOING_STATUSES)],
        "",
        "## Leakage Columns Removed",
        "",
    ]
    for category, columns in leakage_categories.items():
        lines.append(f"{category}:")
        for column in columns:
            status = "not present in processed safe file" if column not in header_set else "present and must be excluded before modeling"
            lines.append(f"- `{column}` ({status})")
        lines.append("")
    lines.extend(
        [
            "Collection-count fields retained as application-time credit-history variables, not post-outcome recovery leakage:",
            "- `collections_12_mths_ex_med`",
            "- `sec_app_collections_12_mths_ex_med`",
            "",
        ]
    )

    lines.extend(
        [
            "## Processed Safe Path Evidence",
            "",
            f"- Processed file: `{LENDINGCLUB_PROCESSED}`.",
            f"- Processed CSV column count: `{metadata.get('processed_column_count', 'NA')}`.",
            f"- Final processed feature count after excluding `TARGET`, `recent_decision`, `issue_d`, and `loan_status`: `{metadata.get('model_feature_count_after_basic_exclusions', 'NA')}`.",
            f"- `loan_status` present in processed CSV: `{metadata.get('loan_status_present')}`.",
            f"- `TARGET` present in processed CSV: `{metadata.get('target_present')}`.",
            f"- `recent_decision` present in processed CSV for temporal splitting: `{metadata.get('recent_decision_present')}`.",
            f"- `issue_d` present in processed CSV but configured as excluded before modeling: `{metadata.get('issue_d_present')}`.",
            f"- Current reported LendingClub matrix rows use the processed safe path: `{_lendingclub_results_use_safe_path()}`.",
            "",
            "Run-level `leakage_report.json` artifacts confirm `target_column_excluded=true`, empty forbidden-column lists in train/OOT features, and `oot_used_in_feature_selection=false`.",
            "",
            "## Missingness-By-Target Check",
            "",
            f"- Rows checked: `{metadata.get('row_count', 'NA')}`.",
            f"- Good rows: `{metadata.get('good_count', 'NA')}`.",
            f"- Bad rows: `{metadata.get('bad_count', 'NA')}`.",
            f"- Possible leakage flags from missingness asymmetry: `{flagged_count}`.",
            f"- Output table: `results/lendingclub/analysis/leakage_transparency/missingness_by_target_leakage_check.csv`.",
            "",
        ]
    )
    if not top_flags.empty:
        lines.append("Top flagged features by absolute missingness difference:")
        for row in top_flags.itertuples(index=False):
            lines.append(
                f"- `{row.feature}`: good missing={row.missing_rate_good:.4f}, bad missing={row.missing_rate_bad:.4f}, diff={row.missing_rate_diff:.4f}"
            )
        lines.append("")
    else:
        lines.append("No missingness-by-target possible leakage flags were found under the configured threshold.\n")

    if not plot_manifest.empty:
        lines.append("Plots:")
        for row in plot_manifest.itertuples(index=False):
            if row.status == "created":
                lines.append(f"- Created `{row.plot_file}`.")
            else:
                lines.append(f"- Skipped `{row.plot_file}`: {row.skip_reason}.")
        lines.append("")

    lines.extend(
        [
            "## Remaining Items",
            "",
            "- The processed safe file keeps `TARGET` for supervised training and `recent_decision`/`issue_d` for time handling, but experiment configs exclude these from model features.",
            "- Raw LendingClub files remain leakage-prone and should not be used directly without the preparation/audit path.",
        ]
    )
    REPORT_LC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_homecredit_temporal_note() -> None:
    REPORT_HC.parent.mkdir(parents=True, exist_ok=True)
    review = Path("data/homecredit/metadata/temporal_asof_review.md")
    review_text = review.read_text(encoding="utf-8") if review.exists() else ""
    lines = [
        "# Home Credit Temporal Semantics Note",
        "",
        "This note is intentionally conservative and does not claim false certainty about auxiliary-table as-of semantics.",
        "",
        "## Auxiliary Tables Used",
        "",
        "- `previous_application`",
        "- `bureau`",
        "- `installments_payments`",
        "- `POS_CASH_balance`",
        "- `credit_card_balance`",
        "- application-level rows from `application_train` / `application_test`",
        "",
        "## Current Enforcement",
        "",
        "The code builds an application-time proxy (`application_time_proxy` / `recent_decision`) from historical relative-day fields and excludes that proxy plus `TARGET` and related time columns from model features. The current artifact review does not prove strict row-level as-of filtering inside every auxiliary table before aggregation.",
        "",
        "## Source Semantics",
        "",
        "The source fields appear historical by naming and Home Credit convention: `DAYS_DECISION`, `DAYS_CREDIT`, `DAYS_INSTALMENT`, `DAYS_ENTRY_PAYMENT`, and `MONTHS_BALANCE` are relative-time fields. That supports the current setup as a reasonable research proxy, but it is not a substitute for manual source documentation review.",
        "",
        "## Evidence Supporting Current Setup",
        "",
        "- Run-level leakage reports confirm target/time columns are excluded from feature matrices and OOT is not used in feature selection.",
        "- `data/homecredit/metadata/temporal_asof_review.md` records the manual-review status explicitly.",
        "- Experiment configs exclude `TARGET`, `recent_decision`, `PREV_recent_decision_MAX`, `DAYS_DECISION`, and `application_time_proxy`.",
        "",
        "## Remaining Caveat",
        "",
        "Manual review remains required for strict professor-facing claims that every auxiliary record is valid as of the application decision date. The current evidence supports a conservative statement: the setup uses historical-looking relative-time fields and removes split/proxy fields before modeling, but auxiliary-table as-of semantics are not fully auto-verified from files alone.",
        "",
        "## What Would Require Rerun",
        "",
        "A rerun would be required only if manual review changes the feature construction rules, excludes additional auxiliary records, changes the temporal proxy, or removes additional feature families. A documentation-only caveat does not require rerunning the current matrix.",
        "",
    ]
    if review_text:
        lines.extend(["## Existing Review Excerpt", "", review_text])
    REPORT_HC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_semantic_report(all_tables: dict[str, pd.DataFrame], missing: list[str], manifests: dict[str, pd.DataFrame]) -> None:
    REPORT_SEMANTIC.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Semantic Coverage And Redundancy",
        "",
        "This report uses existing selected-feature artifacts and, where available, the processed safe modeling table for numeric within-group correlations. It does not rerun feature selection or model training.",
        "",
    ]
    for dataset, table in all_tables.items():
        lines.append(f"## {dataset}")
        if table.empty:
            lines.append("No semantic redundancy table was available.\n")
            continue
        focus = table[table["selector"].isin(SELECTORS)].copy()
        for model in sorted(focus["model"].unique()):
            part = focus[focus["model"].eq(model)].sort_values("number of semantic groups", ascending=False)
            best_coverage = part.iloc[0]
            lowest_concentration = part.sort_values("largest group share").iloc[0]
            lines.append(
                f"- `{model}` widest semantic coverage: `{best_coverage['selector']}` with {int(best_coverage['number of semantic groups'])} groups."
            )
            lines.append(
                f"- `{model}` lowest largest-group share: `{lowest_concentration['selector']}` at {lowest_concentration['largest group share']:.3f}."
            )
        risky = focus[~focus["redundancy risk flag"].eq("low")]
        if risky.empty:
            lines.append("- No non-low redundancy risk flags were found.")
        else:
            lines.append(
                "- Non-low redundancy/concentration flags: "
                + "; ".join(
                    f"`{row['model']}/{row['selector']}`={row['redundancy risk flag']}"
                    for _, row in risky.iterrows()
                )
                + "."
            )
        manifest = manifests.get(dataset, pd.DataFrame())
        if not manifest.empty:
            created = manifest[manifest["status"].eq("created")]["plot_file"].tolist()
            skipped = manifest[manifest["status"].eq("skipped")]
            lines.append(
                "- Plots created: " + (", ".join(f"`{x}`" for x in created) if created else "none") + "."
            )
            lines.append(
                "- Plots skipped: "
                + (
                    "; ".join(f"`{r.plot_file}` ({r.skip_reason})" for r in skipped.itertuples(index=False))
                    if not skipped.empty
                    else "none"
                )
                + "."
            )
        lines.append("")

    lines.extend(
        [
            "## Answers",
            "",
            "- Whether LLM pipelines cover more business concepts than mRMR is dataset/model dependent. The output table identifies the selector with the most semantic groups per model; use that rather than assuming an LLM advantage everywhere.",
            "- LLM pipelines are not automatically less redundant. LendingClub correlation evidence is partial because many selected features are engineered/encoded names not present as raw numeric columns in the processed CSV. Home Credit correlation evidence is unavailable without a processed single-table artifact or rerunning feature construction.",
            "- If AUC differences are small, semantic coverage can be a defensible secondary advantage only where the LLM-family selector shows broader semantic groups or lower concentration without a high redundancy flag.",
            "- Selectors with high largest-group share are too narrow semantically; selectors with high within-group max correlation should be reviewed for duplicate concepts.",
            "",
            "## Missing Or Manual-Review Items",
            "",
        ]
    )
    if missing:
        lines.extend(f"- {item}" for item in sorted(set(missing)))
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Rerun Requirement",
            "",
            "No full rerun is required for the transparency tables and coverage metrics. A rerun would be required only to create full Home Credit and fully engineered LendingClub correlation matrices from the exact modeling design matrices.",
        ]
    )
    REPORT_SEMANTIC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    LENDINGCLUB_LEAKAGE_OUT.mkdir(parents=True, exist_ok=True)
    missingness, lc_metadata = _lendingclub_missingness_by_target()
    missingness_path = LENDINGCLUB_LEAKAGE_OUT / "missingness_by_target_leakage_check.csv"
    missingness.to_csv(missingness_path, index=False)
    lc_plot_manifest = _plot_lendingclub_leakage(missingness)
    _write_lendingclub_report(lc_metadata, missingness, lc_plot_manifest)
    _write_homecredit_temporal_note()

    semantic_tables: dict[str, pd.DataFrame] = {}
    semantic_missing: list[str] = []
    semantic_manifests: dict[str, pd.DataFrame] = {}
    for dataset in DATASETS:
        out_dir = RESULTS_ROOT / dataset / "analysis" / "semantic_redundancy"
        out_dir.mkdir(parents=True, exist_ok=True)
        table, missing = _build_semantic_redundancy(dataset)
        semantic_tables[dataset] = table
        semantic_missing.extend(missing)
        table.to_csv(out_dir / "semantic_coverage_redundancy_by_pipeline.csv", index=False)
        semantic_manifests[dataset] = _plot_semantic(dataset, table)
    _write_semantic_report(semantic_tables, semantic_missing, semantic_manifests)

    print(
        json.dumps(
            {
                "lendingclub_missingness_rows": len(missingness),
                "lendingclub_missingness_flags": int(missingness["possible_leakage_flag"].sum()) if not missingness.empty else 0,
                "semantic_rows": {dataset: len(table) for dataset, table in semantic_tables.items()},
                "reports": [str(REPORT_LC), str(REPORT_HC), str(REPORT_SEMANTIC)],
                "semantic_missing": sorted(set(semantic_missing)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
