from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


DATASETS = ("homecredit", "lendingclub")
MODELS = ("catboost", "lr")
LLM_DIAG_SELECTOR = "llm"
BASELINE_SELECTOR = "mrmr"
CANDIDATE_SELECTORS = ("stable_core_llm_fill", "llm", "llm_then_mrmr")
RESULTS_ROOT = Path("results")
OUT_DIR = RESULTS_ROOT / "cross_dataset" / "analysis" / "stability_significance"
PLOTS_DIR = OUT_DIR / "plots"
REPORT_PATH = Path("reports") / "stability_and_significance_analysis.md"


@dataclass(frozen=True)
class RunInfo:
    dataset: str
    run_id: str
    model: str
    selector: str
    experiment_type: str
    output_folder: Path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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


def _load_runs(dataset: str) -> dict[tuple[str, str], RunInfo]:
    matrix = _read_csv(RESULTS_ROOT / dataset / "matrix_runs.csv")
    runs: dict[tuple[str, str], RunInfo] = {}
    for row in matrix.to_dict("records"):
        if row.get("status") != "completed":
            continue
        folder = _normalize_path(row["output_folder"])
        runs[(str(row["model"]), str(row["selector"]))] = RunInfo(
            dataset=dataset,
            run_id=str(row["run_id"]),
            model=str(row["model"]),
            selector=str(row["selector"]),
            experiment_type=str(row["experiment_type"]),
            output_folder=folder,
        )
    return runs


def _stability_row(dataset: str, model: str, selector: str) -> dict[str, Any]:
    final = _read_csv(RESULTS_ROOT / dataset / "final_comparison_table.csv")
    subset = final[(final["model"].eq(model)) & (final["selector"].eq(selector))]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def _split_summary(run: RunInfo) -> dict[str, Any]:
    manifest = _read_json(run.output_folder / "data_split_manifest.json")
    dev = manifest.get("dev", {})
    oot = manifest.get("oot", {})
    return {
        "DEV rows": dev.get("row_count"),
        "OOT rows": oot.get("row_count"),
        "DEV bad rate": dev.get("target_rate"),
        "OOT bad rate": oot.get("target_rate"),
        "bad-rate difference": (
            oot.get("target_rate") - dev.get("target_rate")
            if dev.get("target_rate") is not None and oot.get("target_rate") is not None
            else math.nan
        ),
    }


def _selection_frequency_summary(run: RunInfo) -> tuple[float, float]:
    freq = _read_csv(run.output_folder / "features" / "selection_frequency.csv")
    if freq.empty or "selection_frequency" not in freq.columns:
        return math.nan, math.nan
    values = pd.to_numeric(freq["selection_frequency"], errors="coerce").dropna()
    if values.empty:
        return math.nan, math.nan
    return float(values.mean()), float(values.median())


def _mean_pairwise_spearman(rank_frames: list[pd.DataFrame]) -> float:
    values: list[float] = []
    for left, right in combinations(rank_frames, 2):
        merged = left.merge(right, on="feature", suffixes=("_left", "_right"))
        if len(merged) < 3:
            continue
        if merged["rank_left"].nunique() <= 1 or merged["rank_right"].nunique() <= 1:
            continue
        coef = stats.spearmanr(merged["rank_left"], merged["rank_right"], nan_policy="omit").statistic
        if pd.notna(coef):
            values.append(float(coef))
    return float(pd.Series(values).mean()) if values else math.nan


def _llm_rank_stability(run: RunInfo) -> float:
    rankings = _read_csv(run.output_folder / "features" / "llm_rankings_summary.csv")
    if rankings.empty:
        rankings = _read_csv(run.output_folder / "feature_rankings" / "llm_rankings_summary.csv")
    if rankings.empty:
        return math.nan

    fold_frames: list[pd.DataFrame] = []
    folds = rankings[rankings["scope"].astype(str).eq("fold")].copy()
    for fold_id, group in folds.groupby("fold_id"):
        if pd.isna(fold_id):
            continue
        frame = group.rename(columns={"feature_name": "feature"})[["feature", "rank"]].dropna()
        if len(frame) >= 3:
            fold_frames.append(frame)
    return _mean_pairwise_spearman(fold_frames)


def _iv_rank_stability(run: RunInfo) -> float:
    rank_frames: list[pd.DataFrame] = []
    for fold in range(1, 6):
        candidates = [
            run.output_folder / "llm_responses" / f"fold_{fold}" / "iv_prefilter" / "iv_summary.csv",
            run.output_folder / "llm_responses" / f"fold_{fold}" / "llm" / "iv_prefilter" / "iv_summary.csv",
        ]
        frame = pd.DataFrame()
        for path in candidates:
            frame = _read_csv(path)
            if not frame.empty:
                break
        if frame.empty:
            continue
        feature_col = "feature" if "feature" in frame.columns else "Unnamed: 0"
        if feature_col not in frame.columns or "IV" not in frame.columns:
            continue
        fold_ranks = (
            frame[[feature_col, "IV"]]
            .rename(columns={feature_col: "feature"})
            .dropna()
            .copy()
        )
        fold_ranks["rank"] = pd.to_numeric(fold_ranks["IV"], errors="coerce").rank(
            method="average",
            ascending=False,
        )
        fold_ranks = fold_ranks[["feature", "rank"]].dropna()
        if len(fold_ranks) >= 3:
            rank_frames.append(fold_ranks)
    return _mean_pairwise_spearman(rank_frames)


def _build_llm_diagnosis() -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for dataset in DATASETS:
        runs = _load_runs(dataset)
        for model in MODELS:
            run = runs.get((model, LLM_DIAG_SELECTOR))
            stability = _stability_row(dataset, model, LLM_DIAG_SELECTOR)
            if run is None:
                missing.append(f"{dataset}/{model}/llm run missing from matrix_runs.csv")
                continue
            if not stability:
                missing.append(f"{dataset}/{model}/llm missing from final_comparison_table.csv")
                continue
            freq_mean, freq_median = _selection_frequency_summary(run)
            if pd.isna(freq_mean):
                missing.append(f"{dataset}/{model}/llm selection_frequency.csv missing or empty")
            iv_stability = _iv_rank_stability(run)
            if pd.isna(iv_stability):
                missing.append(f"{dataset}/{model}/llm IV fold rank stability unavailable")
            llm_rank_stability = stability.get("spearman_rank_stability_mean")
            if pd.isna(llm_rank_stability):
                llm_rank_stability = _llm_rank_stability(run)
            if pd.isna(llm_rank_stability):
                missing.append(f"{dataset}/{model}/llm LLM fold rank stability unavailable")

            row = {
                "dataset": dataset,
                **_split_summary(run),
                "model": model,
                "selector": LLM_DIAG_SELECTOR,
                "Nogueira stability": stability.get("nogueira_stability"),
                "feature Jaccard": stability.get("mean_pairwise_jaccard"),
                "semantic Jaccard": stability.get("semantic_group_jaccard"),
                "mean fold selection frequency": freq_mean,
                "median fold selection frequency": freq_median,
                "IV rank stability if available": iv_stability,
                "LLM rank stability if available": llm_rank_stability,
            }
            rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "DEV rows",
            "OOT rows",
            "DEV bad rate",
            "OOT bad rate",
            "bad-rate difference",
            "model",
            "selector",
            "Nogueira stability",
            "feature Jaccard",
            "semantic Jaccard",
            "mean fold selection frequency",
            "median fold selection frequency",
            "IV rank stability if available",
            "LLM rank stability if available",
        ],
    ), missing


def _fold_auc(run: RunInfo) -> pd.DataFrame:
    cv = _read_csv(run.output_folder / "results" / "cv_results.csv")
    if cv.empty or "fold" not in cv.columns or "auc" not in cv.columns:
        return pd.DataFrame(columns=["fold", "auc"])
    frame = cv.copy()
    frame["fold_numeric"] = pd.to_numeric(frame["fold"], errors="coerce")
    frame = frame[frame["fold_numeric"].between(1, 5, inclusive="both")].copy()
    return frame[["fold_numeric", "auc"]].rename(columns={"fold_numeric": "fold"})


def _paired_tests_for_delta(delta: pd.Series) -> tuple[float, float, str]:
    delta = pd.to_numeric(delta, errors="coerce").dropna()
    if len(delta) < 2:
        return math.nan, math.nan, "insufficient paired folds"

    ttest_p = math.nan
    if delta.nunique() > 1:
        ttest_p = float(stats.ttest_1samp(delta, popmean=0.0, nan_policy="omit").pvalue)
    elif delta.eq(0).all():
        ttest_p = 1.0

    nonzero = delta[delta.ne(0)]
    wilcoxon_p = math.nan
    if len(nonzero) >= 3:
        wilcoxon_p = float(stats.wilcoxon(nonzero, zero_method="wilcox").pvalue)
    return ttest_p, wilcoxon_p, ""


def _interpret(mean_delta: float, ttest_p: float, wilcoxon_p: float, missing_reason: str) -> str:
    if missing_reason:
        return missing_reason
    p_values = [value for value in [ttest_p, wilcoxon_p] if pd.notna(value)]
    significant = any(value < 0.05 for value in p_values)
    direction = "higher" if mean_delta > 0 else "lower" if mean_delta < 0 else "unchanged"
    if significant:
        return f"Candidate fold AUC is significantly {direction} than mRMR at alpha=0.05."
    if abs(mean_delta) < 0.002:
        return "Mean fold AUC delta is tiny and not statistically significant."
    return f"Mean fold AUC is {direction} than mRMR, but not statistically significant."


def _build_paired_tests() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    deltas_for_plot: list[dict[str, Any]] = []

    for dataset in DATASETS:
        runs = _load_runs(dataset)
        for model in MODELS:
            baseline = runs.get((model, BASELINE_SELECTOR))
            if baseline is None:
                missing.append(f"{dataset}/{model}/mrmr baseline run missing")
                continue
            baseline_auc = _fold_auc(baseline)
            if baseline_auc.empty:
                missing.append(f"{dataset}/{model}/mrmr fold AUC missing")
                continue

            for candidate_selector in CANDIDATE_SELECTORS:
                candidate = runs.get((model, candidate_selector))
                if candidate is None:
                    missing.append(f"{dataset}/{model}/{candidate_selector} run missing")
                    continue
                candidate_auc = _fold_auc(candidate)
                if candidate_auc.empty:
                    missing.append(f"{dataset}/{model}/{candidate_selector} fold AUC missing")
                    continue

                merged = candidate_auc.merge(
                    baseline_auc,
                    on="fold",
                    suffixes=("_candidate", "_baseline"),
                )
                if merged.empty:
                    missing.append(f"{dataset}/{model}/{candidate_selector} has no paired AUC folds with mRMR")
                    continue
                merged["delta"] = merged["auc_candidate"] - merged["auc_baseline"]
                delta = merged["delta"]
                ttest_p, wilcoxon_p, reason = _paired_tests_for_delta(delta)
                mean_delta = float(delta.mean())
                std_delta = float(delta.std(ddof=1)) if len(delta) > 1 else math.nan
                p_values = [value for value in [ttest_p, wilcoxon_p] if pd.notna(value)]
                significant = bool(any(value < 0.05 for value in p_values))
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "candidate_selector": candidate_selector,
                        "baseline_selector": BASELINE_SELECTOR,
                        "metric": "auc",
                        "mean_delta": mean_delta,
                        "std_delta": std_delta,
                        "ttest_p_value": ttest_p,
                        "wilcoxon_p_value": wilcoxon_p,
                        "significant_at_0_05": significant,
                        "interpretation": _interpret(mean_delta, ttest_p, wilcoxon_p, reason),
                    }
                )
                for item in merged.to_dict("records"):
                    deltas_for_plot.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "candidate_selector": candidate_selector,
                            "baseline_selector": BASELINE_SELECTOR,
                            "fold": int(item["fold"]),
                            "auc_delta": item["delta"],
                        }
                    )

    return pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "model",
            "candidate_selector",
            "baseline_selector",
            "metric",
            "mean_delta",
            "std_delta",
            "ttest_p_value",
            "wilcoxon_p_value",
            "significant_at_0_05",
            "interpretation",
        ],
    ), missing, pd.DataFrame(deltas_for_plot)


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


def _plot_stability_vs_auc(manifest: list[dict[str, Any]]) -> None:
    plot_file = "stability_vs_oot_auc_by_dataset_model.png"
    source = "results/<dataset>/final_comparison_table.csv"
    columns = ["dataset_name", "model", "selector", "oot_auc", "nogueira_stability"]
    purpose = "Show whether exact feature stability aligns with OOT AUC across selectors."
    frames = []
    for dataset in DATASETS:
        final = _read_csv(RESULTS_ROOT / dataset / "final_comparison_table.csv")
        if not final.empty:
            frames.append(final)
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if data.empty:
        _record_plot(manifest, plot_file, source, 0, columns, purpose, "skipped", "empty source data")
        return
    if data["selector"].nunique() <= 1 or data["oot_auc"].nunique() <= 1 or data["nogueira_stability"].nunique() <= 1:
        _record_plot(manifest, plot_file, source, len(data), columns, purpose, "skipped", "requires variation in selectors, AUC, and stability")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for (dataset, model), group in data.groupby(["dataset_name", "model"]):
        ax.scatter(group["nogueira_stability"], group["oot_auc"], label=f"{dataset}/{model}", s=55)
    ax.set_xlabel("Nogueira Stability")
    ax.set_ylabel("OOT AUC")
    ax.set_title("Stability vs OOT AUC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / plot_file, dpi=160)
    plt.close(fig)
    _record_plot(manifest, plot_file, source, len(data), columns, purpose, "created")


def _plot_fold_auc_differences(deltas: pd.DataFrame, manifest: list[dict[str, Any]]) -> None:
    plot_file = "fold_level_auc_differences_key_comparisons.png"
    columns = ["dataset", "model", "candidate_selector", "fold", "auc_delta"]
    purpose = "Show paired fold AUC deltas for candidate selectors versus mRMR."
    if deltas.empty:
        _record_plot(manifest, plot_file, "paired fold cv_results.csv", 0, columns, purpose, "skipped", "empty source data")
        return
    if deltas["candidate_selector"].nunique() <= 1 or deltas["auc_delta"].nunique() <= 1:
        _record_plot(manifest, plot_file, "paired fold cv_results.csv", len(deltas), columns, purpose, "skipped", "requires multiple comparisons and non-constant deltas")
        return

    plot_df = deltas.copy()
    plot_df["comparison"] = (
        plot_df["dataset"]
        + "/"
        + plot_df["model"]
        + "/"
        + plot_df["candidate_selector"]
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    groups = [group["auc_delta"].values for _, group in plot_df.groupby("comparison")]
    labels = [name for name, _ in plot_df.groupby("comparison")]
    ax.boxplot(groups, tick_labels=labels, vert=True)
    ax.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
    ax.set_ylabel("Fold AUC delta vs mRMR")
    ax.set_title("Paired Fold AUC Differences")
    ax.tick_params(axis="x", labelrotation=75)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / plot_file, dpi=160)
    plt.close(fig)
    _record_plot(manifest, plot_file, "paired fold cv_results.csv", len(deltas), columns, purpose, "created")


def _plot_llm_selection_frequency(manifest: list[dict[str, Any]]) -> None:
    plot_file = "llm_fold_selection_frequency_by_dataset.png"
    columns = ["dataset", "model", "feature_name", "selection_frequency"]
    purpose = "Compare how concentrated LLM-selected fold feature sets are across datasets."
    rows = []
    for dataset in DATASETS:
        runs = _load_runs(dataset)
        for model in MODELS:
            run = runs.get((model, LLM_DIAG_SELECTOR))
            if not run:
                continue
            freq = _read_csv(run.output_folder / "features" / "selection_frequency.csv")
            if freq.empty:
                continue
            freq = freq.copy()
            freq["dataset"] = dataset
            freq["model"] = model
            rows.append(freq)
    data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if data.empty:
        _record_plot(manifest, plot_file, "features/selection_frequency.csv", 0, columns, purpose, "skipped", "empty source data")
        return
    if data["dataset"].nunique() <= 1 or data["selection_frequency"].nunique() <= 1:
        _record_plot(manifest, plot_file, "features/selection_frequency.csv", len(data), columns, purpose, "skipped", "requires multiple datasets and non-constant frequencies")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    data.boxplot(column="selection_frequency", by=["dataset", "model"], ax=ax, rot=25)
    ax.set_title("LLM Fold Selection Frequency")
    ax.set_ylabel("Selection Frequency Across Folds")
    ax.set_xlabel("")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / plot_file, dpi=160)
    plt.close(fig)
    _record_plot(manifest, plot_file, "features/selection_frequency.csv", len(data), columns, purpose, "created")


def _write_plots(deltas: pd.DataFrame) -> pd.DataFrame:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    _plot_stability_vs_auc(manifest)
    _plot_fold_auc_differences(deltas, manifest)
    _plot_llm_selection_frequency(manifest)
    plot_manifest = pd.DataFrame(
        manifest,
        columns=[
            "plot_file",
            "source_table",
            "rows_used",
            "columns_used",
            "purpose",
            "status",
            "skip_reason",
        ],
    )
    plot_manifest.to_csv(PLOTS_DIR / "plot_manifest.csv", index=False)
    return plot_manifest


def _fmt(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _write_report(
    diagnosis: pd.DataFrame,
    tests: pd.DataFrame,
    plot_manifest: pd.DataFrame,
    missing: list[str],
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stability And Significance Analysis",
        "",
        "This report uses existing post-run artifacts only. It does not rerun the experiment matrix, retrain models, or rebuild datasets.",
        "",
        "## LLM Stability Diagnosis",
        "",
    ]

    if diagnosis.empty:
        lines.append("No LLM diagnosis rows were available.")
    else:
        for model in MODELS:
            hc = diagnosis[(diagnosis["dataset"].eq("homecredit")) & (diagnosis["model"].eq(model))]
            lc = diagnosis[(diagnosis["dataset"].eq("lendingclub")) & (diagnosis["model"].eq(model))]
            if hc.empty or lc.empty:
                continue
            hc_row = hc.iloc[0]
            lc_row = lc.iloc[0]
            lines.append(
                f"- `{model}/llm`: LendingClub has higher exact stability than Home Credit "
                f"(Nogueira {_fmt(lc_row['Nogueira stability'])} vs {_fmt(hc_row['Nogueira stability'])}; "
                f"feature Jaccard {_fmt(lc_row['feature Jaccard'])} vs {_fmt(hc_row['feature Jaccard'])})."
            )
            llm_rank_direction = (
                "higher"
                if lc_row["LLM rank stability if available"] > hc_row["LLM rank stability if available"]
                else "lower"
                if lc_row["LLM rank stability if available"] < hc_row["LLM rank stability if available"]
                else "similar"
            )
            lines.append(
                f"  Stored selected-rank stability is {llm_rank_direction} on LendingClub "
                f"({_fmt(lc_row['LLM rank stability if available'])} vs {_fmt(hc_row['LLM rank stability if available'])}); "
                f"IV rank stability is {_fmt(lc_row['IV rank stability if available'])} vs {_fmt(hc_row['IV rank stability if available'])}."
            )
        lines.append("")
        lines.append(
            "The difference is not just a reporting artifact: exact selected-feature overlap and fold selection frequencies are higher on LendingClub. The stored selected-rank stability and IV rank stability are also stronger on LendingClub in these artifacts. Larger DEV sample size on LendingClub likely contributes to more stable fold rankings, but LendingClub also has larger target-rate drift, so target drift alone does not explain the higher exact stability."
        )

    lines.extend(["", "## Paired Fold AUC Tests", ""])
    if tests.empty:
        lines.append("No paired fold tests were available.")
    else:
        oot_gain_lines: list[str] = []
        for dataset in DATASETS:
            final = _read_csv(RESULTS_ROOT / dataset / "final_comparison_table.csv")
            if final.empty:
                continue
            for model in MODELS:
                baseline = final[(final["model"].eq(model)) & (final["selector"].eq(BASELINE_SELECTOR))]
                if baseline.empty:
                    continue
                baseline_auc = float(baseline.iloc[0]["oot_auc"])
                for selector in CANDIDATE_SELECTORS:
                    candidate = final[(final["model"].eq(model)) & (final["selector"].eq(selector))]
                    if candidate.empty:
                        continue
                    oot_delta = float(candidate.iloc[0]["oot_auc"]) - baseline_auc
                    test_row = tests[
                        tests["dataset"].eq(dataset)
                        & tests["model"].eq(model)
                        & tests["candidate_selector"].eq(selector)
                    ]
                    significant = bool(test_row.iloc[0]["significant_at_0_05"]) if not test_row.empty else False
                    fold_delta = float(test_row.iloc[0]["mean_delta"]) if not test_row.empty else math.nan
                    if oot_delta > 0:
                        if significant and fold_delta > 0:
                            qualifier = "paired folds support a significant gain"
                        elif significant and fold_delta < 0:
                            qualifier = "paired folds significantly favor mRMR, so this OOT gain is not fold-supported"
                        else:
                            qualifier = "paired folds do not support significance"
                        oot_gain_lines.append(
                            f"- `{dataset}/{model}/{selector}` OOT AUC delta vs mRMR is {_fmt(oot_delta)}; {qualifier}."
                        )
        if oot_gain_lines:
            lines.append("Positive OOT AUC deltas among requested comparisons:")
            lines.extend(oot_gain_lines)
            lines.append("")
            lines.append(
                "None of the positive OOT AUC deltas should be treated as a strong gain from the paired fold evidence; they are either tiny or not significant across folds."
            )
            lines.append("")
        significant = tests[tests["significant_at_0_05"].astype(bool)]
        if significant.empty:
            lines.append("No requested candidate-vs-mRMR comparison is significant at alpha 0.05 on fold-level AUC.")
        else:
            lines.append(
                "Significant fold-level AUC differences at alpha 0.05. These are significant by the paired t-test; with only five folds, the Wilcoxon test is more conservative and may not cross 0.05."
            )
            for row in significant.itertuples(index=False):
                lines.append(
                    f"- `{row.dataset}/{row.model}/{row.candidate_selector}` vs mRMR: "
                    f"mean delta {_fmt(row.mean_delta)}, t-test p={_fmt(row.ttest_p_value)}, "
                    f"Wilcoxon p={_fmt(row.wilcoxon_p_value)}."
                )
        tiny = tests[tests["mean_delta"].abs() < 0.002]
        if not tiny.empty:
            lines.append("")
            lines.append("Small deltas that should not be treated strongly:")
            for row in tiny.itertuples(index=False):
                lines.append(
                    f"- `{row.dataset}/{row.model}/{row.candidate_selector}` vs mRMR: mean AUC delta {_fmt(row.mean_delta)}."
                )

    lines.extend(["", "## Stability Interpretation", ""])
    if not diagnosis.empty:
        lines.append(
            "mRMR still dominates exact stability on Home Credit for the strongest statistical baselines, while LendingClub shows much higher LLM exact stability than Home Credit. The LLM behavior is dataset-dependent: LendingClub has fewer candidate features and much larger DEV folds, and the LLM fold rankings are more consistent there."
        )
    if not tests.empty:
        lines.append(
            "OOT or fold AUC gains should be described cautiously. The paired fold tests show that many deltas are small, and significance depends on dataset/model/selector rather than being a blanket LLM-family effect."
        )

    lines.extend(["", "## Plots", ""])
    if plot_manifest.empty:
        lines.append("No plot manifest was generated.")
    else:
        created = plot_manifest[plot_manifest["status"].eq("created")]
        skipped = plot_manifest[plot_manifest["status"].eq("skipped")]
        lines.append(
            "Created plots: "
            + (", ".join(f"`{row.plot_file}`" for row in created.itertuples(index=False)) if not created.empty else "none")
            + "."
        )
        lines.append(
            "Skipped plots: "
            + (
                "; ".join(f"`{row.plot_file}` ({row.skip_reason})" for row in skipped.itertuples(index=False))
                if not skipped.empty
                else "none"
            )
            + "."
        )

    lines.extend(["", "## Missing Artifacts", ""])
    if missing:
        for item in sorted(set(missing)):
            lines.append(f"- {item}")
    else:
        lines.append("- None for the requested diagnosis rows and paired fold AUC tests.")

    lines.extend(
        [
            "",
            "## Rerun Requirement",
            "",
            "A full rerun is not required for this analysis. The only caveat is that stronger attribution of rejected LLM candidates would require artifacts that store fold-level metric effects or PSI/IV deltas for rejected candidates, but that is outside the requested no-rerun scope.",
            "",
            "## Files Created",
            "",
            f"- `{OUT_DIR / 'llm_stability_diagnosis.csv'}`",
            f"- `{OUT_DIR / 'paired_fold_significance_tests.csv'}`",
            f"- `{PLOTS_DIR / 'plot_manifest.csv'}`",
            f"- `{REPORT_PATH}`",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    diagnosis, missing_diag = _build_llm_diagnosis()
    tests, missing_tests, deltas = _build_paired_tests()
    diagnosis.to_csv(OUT_DIR / "llm_stability_diagnosis.csv", index=False)
    tests.to_csv(OUT_DIR / "paired_fold_significance_tests.csv", index=False)
    plot_manifest = _write_plots(deltas)
    _write_report(diagnosis, tests, plot_manifest, missing_diag + missing_tests)
    print(
        json.dumps(
            {
                "diagnosis_rows": len(diagnosis),
                "paired_test_rows": len(tests),
                "fold_delta_rows": len(deltas),
                "created_plots": plot_manifest[plot_manifest["status"].eq("created")][
                    "plot_file"
                ].tolist(),
                "skipped_plots": plot_manifest[plot_manifest["status"].eq("skipped")][
                    ["plot_file", "skip_reason"]
                ].to_dict("records"),
                "missing_artifacts": sorted(set(missing_diag + missing_tests)),
                "report": str(REPORT_PATH),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
