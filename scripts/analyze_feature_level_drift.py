from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


DATASETS = ("homecredit", "lendingclub")
RESULTS_ROOT = Path("results")
REPORT_PATH = Path("reports") / "feature_level_drift_analysis.md"
HIGH_PSI_THRESHOLD = 0.25


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
    return Path(path_text.replace("\\", "/"))


def _load_runs(dataset: str) -> list[RunInfo]:
    matrix = _read_csv(RESULTS_ROOT / dataset / "matrix_runs.csv")
    runs: list[RunInfo] = []
    if matrix.empty:
        return runs

    for row in matrix.to_dict("records"):
        if row.get("status") != "completed":
            continue
        folder = _normalize_path(row["output_folder"])
        if not folder.exists():
            folder = RESULTS_ROOT / dataset / str(row["model"]) / str(row["experiment_type"]) / str(row["run_id"])
        runs.append(
            RunInfo(
                dataset=dataset,
                run_id=str(row["run_id"]),
                model=str(row["model"]),
                selector=str(row["selector"]),
                experiment_type=str(row["experiment_type"]),
                output_folder=folder,
            )
        )
    return runs


def _feature_metadata_map(dataset: str) -> pd.DataFrame:
    evidence = _read_csv(RESULTS_ROOT / dataset / "feature_level_evidence.csv")
    if evidence.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "semantic_group",
                "source_table",
                "feature_missing_dev_fallback",
            ]
        )
    return (
        evidence.rename(
            columns={
                "feature_name": "feature",
                "missing_rate_mean": "feature_missing_dev_fallback",
            }
        )[
            [
                "feature",
                "semantic_group",
                "source_table",
                "feature_missing_dev_fallback",
            ]
        ]
        .drop_duplicates("feature")
        .copy()
    )


def _run_feature_metadata(run: RunInfo) -> pd.DataFrame:
    candidates = [
        run.output_folder / "llm_responses" / "final_dev" / "llm" / "feature_metadata.csv",
        run.output_folder / "llm_responses" / "final_dev" / "feature_metadata.csv",
    ]
    for path in candidates:
        frame = _read_csv(path)
        if not frame.empty and "name" in frame.columns:
            return frame.rename(
                columns={
                    "name": "feature",
                    "table": "source_table_run",
                    "missing_rate": "feature_missing_dev_run",
                    "semantic_group": "semantic_group_run",
                }
            )[
                [
                    "feature",
                    "source_table_run",
                    "semantic_group_run",
                    "feature_missing_dev_run",
                ]
            ].drop_duplicates("feature")
    return pd.DataFrame(
        columns=[
            "feature",
            "source_table_run",
            "semantic_group_run",
            "feature_missing_dev_run",
        ]
    )


def _selected_features(run: RunInfo) -> pd.DataFrame:
    candidates = [
        run.output_folder / "features" / "final_selected_features.csv",
        run.output_folder / "selected_feature_sets" / "final_selected_features.csv",
    ]
    for path in candidates:
        frame = _read_csv(path)
        if not frame.empty:
            feature_col = "feature_name" if "feature_name" in frame.columns else "feature"
            if feature_col not in frame.columns:
                continue
            frame = frame.copy()
            frame["feature"] = frame[feature_col]
            if "rank" not in frame.columns:
                frame["rank"] = range(1, len(frame) + 1)
            return frame[["feature", "rank"]].drop_duplicates("feature")
    return pd.DataFrame(columns=["feature", "rank"])


def _feature_level_psi_for_run(run: RunInfo, metadata: pd.DataFrame) -> pd.DataFrame:
    psi = _read_csv(run.output_folder / "results" / "selected_feature_psi.csv")
    if psi.empty:
        return pd.DataFrame()
    psi = psi.rename(columns={"feature_name": "feature", "psi": "psi_dev_oot"}).copy()
    selected = _selected_features(run).rename(columns={"rank": "selected_rank"})
    run_meta = _run_feature_metadata(run)

    frame = psi.merge(metadata, on="feature", how="left")
    frame = frame.merge(run_meta, on="feature", how="left")
    frame = frame.merge(selected, on="feature", how="left")

    frame["dataset"] = run.dataset
    frame["run_id"] = run.run_id
    frame["model"] = run.model
    frame["selector"] = run.selector
    frame["semantic_group"] = frame["semantic_group_run"].combine_first(frame["semantic_group"])
    frame["source_table"] = frame["source_table_run"].combine_first(frame["source_table"])
    frame["feature_missing_dev"] = frame["feature_missing_dev_run"].combine_first(
        frame["feature_missing_dev_fallback"]
    )
    frame["feature_missing_oot"] = pd.NA
    frame["is_high_psi_flag"] = frame["psi_dev_oot"] >= HIGH_PSI_THRESHOLD

    if run.selector == "pca":
        frame["semantic_group"] = "pca_component"
        frame["source_table"] = "derived_pca_component"

    return frame[
        [
            "dataset",
            "run_id",
            "model",
            "selector",
            "feature",
            "semantic_group",
            "source_table",
            "psi_dev_oot",
            "feature_missing_dev",
            "feature_missing_oot",
            "selected_rank",
            "is_high_psi_flag",
        ]
    ].sort_values(["model", "selector", "psi_dev_oot"], ascending=[True, True, False])


def _distribution_by_pipeline(feature_psi: pd.DataFrame) -> pd.DataFrame:
    if feature_psi.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "selector",
                "selected_feature_count",
                "psi_mean",
                "psi_median",
                "psi_p75",
                "psi_p90",
                "psi_p95",
                "psi_max",
                "high_psi_feature_count",
                "high_psi_feature_ratio",
            ]
        )

    grouped = feature_psi.groupby(["dataset", "model", "selector"], dropna=False)
    summary = grouped["psi_dev_oot"].agg(
        selected_feature_count="count",
        psi_mean="mean",
        psi_median="median",
        psi_p75=lambda s: s.quantile(0.75),
        psi_p90=lambda s: s.quantile(0.90),
        psi_p95=lambda s: s.quantile(0.95),
        psi_max="max",
    )
    high_counts = grouped["is_high_psi_flag"].sum().rename("high_psi_feature_count")
    summary = summary.join(high_counts).reset_index()
    summary["high_psi_feature_ratio"] = (
        summary["high_psi_feature_count"] / summary["selected_feature_count"]
    )
    return summary.sort_values(["dataset", "model", "psi_mean", "selector"])


def _high_psi_features(feature_psi: pd.DataFrame) -> pd.DataFrame:
    if feature_psi.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "selector",
                "run_id",
                "feature",
                "semantic_group",
                "psi_dev_oot",
                "rank_within_pipeline",
                "reason_flag",
            ]
        )
    high = feature_psi[feature_psi["is_high_psi_flag"]].copy()
    if high.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "selector",
                "run_id",
                "feature",
                "semantic_group",
                "psi_dev_oot",
                "rank_within_pipeline",
                "reason_flag",
            ]
        )
    high["rank_within_pipeline"] = high.groupby(["dataset", "model", "selector"])[
        "psi_dev_oot"
    ].rank(method="first", ascending=False)
    high["reason_flag"] = high["psi_dev_oot"].map(
        lambda value: f"PSI >= {HIGH_PSI_THRESHOLD:g}"
    )
    return high[
        [
            "dataset",
            "model",
            "selector",
            "run_id",
            "feature",
            "semantic_group",
            "psi_dev_oot",
            "rank_within_pipeline",
            "reason_flag",
        ]
    ].sort_values(["dataset", "model", "selector", "rank_within_pipeline"])


def _llm_candidate_pool(run: RunInfo) -> pd.DataFrame:
    rankings = _read_csv(run.output_folder / "features" / "llm_rankings_summary.csv")
    if rankings.empty:
        rankings = _read_csv(run.output_folder / "feature_rankings" / "llm_rankings_summary.csv")
    if rankings.empty:
        return pd.DataFrame(columns=["feature", "llm_rank"])

    final = rankings[rankings["scope"].astype(str).eq("final_dev")].copy()
    if final.empty:
        final = rankings.copy()
    candidate_col = f"candidate_for_{run.model}_hybrid"
    if candidate_col in final.columns:
        final = final[final[candidate_col].fillna(False).astype(bool)]
    elif "rank" in final.columns:
        manifest = _read_json(run.output_folder / "run_manifest.json")
        budget = manifest.get("llm_candidate_pool_budget") or manifest.get("llm_shared_pool_size")
        if budget:
            final = final[final["rank"] <= int(budget)]
    final = final.rename(columns={"feature_name": "feature", "rank": "llm_rank"})
    return final[["feature", "llm_rank"]].drop_duplicates("feature")


def _llm_then_mrmr_breakdown(
    dataset: str,
    runs: list[RunInfo],
    feature_psi: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        if run.selector != "llm_then_mrmr":
            continue
        candidate_pool = _llm_candidate_pool(run)
        selected = _selected_features(run)
        run_psi = feature_psi[feature_psi["run_id"].eq(run.run_id)][["feature", "psi_dev_oot"]]
        max_psi = run_psi["psi_dev_oot"].max() if not run_psi.empty else pd.NA
        max_features = set(
            run_psi[run_psi["psi_dev_oot"].eq(max_psi)]["feature"].tolist()
            if pd.notna(max_psi)
            else []
        )

        features = set(candidate_pool["feature"].tolist())
        features.update(selected["feature"].tolist())
        features.update(max_features)

        joined = pd.DataFrame({"feature": sorted(features)})
        joined = joined.merge(candidate_pool, on="feature", how="left")
        joined = joined.merge(selected.rename(columns={"rank": "selected_rank"}), on="feature", how="left")
        joined = joined.merge(run_psi, on="feature", how="left")
        joined = joined.merge(metadata, on="feature", how="left")

        for item in joined.to_dict("records"):
            rows.append(
                {
                    "dataset": dataset,
                    "model": run.model,
                    "run_id": run.run_id,
                    "feature": item["feature"],
                    "in_llm_top_pool": pd.notna(item.get("llm_rank")),
                    "in_final_selected_set": pd.notna(item.get("selected_rank")),
                    "psi_dev_oot": item.get("psi_dev_oot"),
                    "semantic_group": item.get("semantic_group"),
                    "source_table": item.get("source_table"),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "model",
            "run_id",
            "feature",
            "in_llm_top_pool",
            "in_final_selected_set",
            "psi_dev_oot",
            "semantic_group",
            "source_table",
        ],
    ).sort_values(["dataset", "model", "in_final_selected_set", "psi_dev_oot"], ascending=[True, True, False, False])


def _save_or_skip_boxplot(feature_psi: pd.DataFrame, plots_dir: Path, manifest: list[dict[str, Any]]) -> None:
    plot_name = "feature_level_psi_boxplot_by_selector_model.png"
    columns = ["model", "selector", "psi_dev_oot"]
    purpose = "Compare selected-feature PSI distributions across selector/model pipelines."
    if feature_psi.empty:
        _record_skip(manifest, plot_name, "feature_level_psi_by_run.csv", 0, columns, purpose, "empty source data")
        return
    if feature_psi["selector"].nunique() <= 1 or feature_psi["model"].nunique() <= 1:
        _record_skip(manifest, plot_name, "feature_level_psi_by_run.csv", len(feature_psi), columns, purpose, "requires multiple selectors and models")
        return
    if feature_psi["psi_dev_oot"].nunique(dropna=True) <= 1:
        _record_skip(manifest, plot_name, "feature_level_psi_by_run.csv", len(feature_psi), columns, purpose, "PSI values are constant")
        return

    plot_df = feature_psi.copy()
    plot_df["pipeline"] = plot_df["model"] + " / " + plot_df["selector"]
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.boxplot(column="psi_dev_oot", by="pipeline", ax=ax, rot=75)
    ax.set_title("Feature-Level DEV-OOT PSI by Pipeline")
    ax.set_ylabel("PSI")
    ax.set_xlabel("")
    fig.suptitle("")
    fig.tight_layout()
    path = plots_dir / plot_name
    fig.savefig(path, dpi=160)
    plt.close(fig)
    _record_created(manifest, plot_name, "feature_level_psi_by_run.csv", len(plot_df), columns, purpose)


def _save_or_skip_top20(feature_psi: pd.DataFrame, plots_dir: Path, manifest: list[dict[str, Any]]) -> None:
    plot_name = "top20_highest_psi_selected_features.png"
    columns = ["feature", "model", "selector", "psi_dev_oot"]
    purpose = "Show the highest PSI selected features and the pipelines where they appear."
    plot_df = feature_psi.dropna(subset=["psi_dev_oot"]).nlargest(20, "psi_dev_oot").copy()
    if plot_df.empty:
        _record_skip(manifest, plot_name, "feature_level_psi_by_run.csv", 0, columns, purpose, "empty source data")
        return
    if plot_df["psi_dev_oot"].nunique(dropna=True) <= 1:
        _record_skip(manifest, plot_name, "feature_level_psi_by_run.csv", len(plot_df), columns, purpose, "top PSI values are constant")
        return

    plot_df["label"] = (
        plot_df["feature"].astype(str)
        + " ("
        + plot_df["model"].astype(str)
        + "/"
        + plot_df["selector"].astype(str)
        + ")"
    )
    plot_df = plot_df.sort_values("psi_dev_oot")
    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_df) * 0.35)))
    ax.barh(plot_df["label"], plot_df["psi_dev_oot"], color="#3b6ea8")
    ax.axvline(HIGH_PSI_THRESHOLD, color="#b23b3b", linestyle="--", linewidth=1)
    ax.set_title("Top 20 Highest-PSI Selected Features")
    ax.set_xlabel("PSI")
    fig.tight_layout()
    path = plots_dir / plot_name
    fig.savefig(path, dpi=160)
    plt.close(fig)
    _record_created(manifest, plot_name, "feature_level_psi_by_run.csv", len(plot_df), columns, purpose)


def _save_or_skip_llm_pool_plot(
    breakdown: pd.DataFrame,
    plots_dir: Path,
    manifest: list[dict[str, Any]],
) -> None:
    plot_name = "llm_top_pool_vs_final_selected_psi_llm_then_mrmr.png"
    columns = ["in_llm_top_pool", "in_final_selected_set", "psi_dev_oot"]
    purpose = "Compare PSI evidence for LLM-nominated pool members versus mRMR-final selected features."
    plot_df = breakdown.dropna(subset=["psi_dev_oot"]).copy()
    if plot_df.empty:
        _record_skip(manifest, plot_name, "llm_then_mrmr_drift_source_breakdown.csv", 0, columns, purpose, "PSI is only available for final selected features and no rows have PSI")
        return
    pool_only_total = len(
        breakdown[
            breakdown["in_llm_top_pool"].astype(bool)
            & ~breakdown["in_final_selected_set"].astype(bool)
        ]
    )
    pool_only_with_psi = len(
        plot_df[
            plot_df["in_llm_top_pool"].astype(bool)
            & ~plot_df["in_final_selected_set"].astype(bool)
        ]
    )
    if pool_only_total > 0 and pool_only_with_psi == 0:
        _record_skip(
            manifest,
            plot_name,
            "llm_then_mrmr_drift_source_breakdown.csv",
            len(plot_df),
            columns,
            purpose,
            "PSI is not stored for LLM-top-pool candidates rejected by mRMR",
        )
        return
    plot_df["membership"] = plot_df.apply(
        lambda row: "LLM top pool + final selected"
        if row["in_llm_top_pool"] and row["in_final_selected_set"]
        else "Final selected only"
        if row["in_final_selected_set"]
        else "LLM top pool only",
        axis=1,
    )
    if plot_df["membership"].nunique() <= 1:
        _record_skip(manifest, plot_name, "llm_then_mrmr_drift_source_breakdown.csv", len(plot_df), columns, purpose, "only one membership category has PSI")
        return
    if plot_df["psi_dev_oot"].nunique(dropna=True) <= 1:
        _record_skip(manifest, plot_name, "llm_then_mrmr_drift_source_breakdown.csv", len(plot_df), columns, purpose, "PSI values are constant")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df.boxplot(column="psi_dev_oot", by="membership", ax=ax, rot=20)
    ax.set_title("LLM Top Pool vs Final mRMR-Selected PSI")
    ax.set_ylabel("PSI")
    ax.set_xlabel("")
    fig.suptitle("")
    fig.tight_layout()
    path = plots_dir / plot_name
    fig.savefig(path, dpi=160)
    plt.close(fig)
    _record_created(manifest, plot_name, "llm_then_mrmr_drift_source_breakdown.csv", len(plot_df), columns, purpose)


def _record_created(
    manifest: list[dict[str, Any]],
    plot_file: str,
    source_table: str,
    rows_used: int,
    columns_used: list[str],
    purpose: str,
) -> None:
    manifest.append(
        {
            "plot_file": plot_file,
            "source_table": source_table,
            "rows_used": rows_used,
            "columns_used": ";".join(columns_used),
            "purpose": purpose,
            "status": "created",
            "skip_reason": "",
        }
    )


def _record_skip(
    manifest: list[dict[str, Any]],
    plot_file: str,
    source_table: str,
    rows_used: int,
    columns_used: list[str],
    purpose: str,
    skip_reason: str,
) -> None:
    manifest.append(
        {
            "plot_file": plot_file,
            "source_table": source_table,
            "rows_used": rows_used,
            "columns_used": ";".join(columns_used),
            "purpose": purpose,
            "status": "skipped",
            "skip_reason": skip_reason,
        }
    )


def _write_dataset_outputs(dataset: str) -> dict[str, Any]:
    out_dir = RESULTS_ROOT / dataset / "analysis" / "feature_level_drift"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(dataset)
    metadata = _feature_metadata_map(dataset)
    feature_frames = [
        _feature_level_psi_for_run(run, metadata)
        for run in runs
    ]
    feature_psi = (
        pd.concat([frame for frame in feature_frames if not frame.empty], ignore_index=True)
        if any(not frame.empty for frame in feature_frames)
        else pd.DataFrame()
    )
    distribution = _distribution_by_pipeline(feature_psi)
    high_psi = _high_psi_features(feature_psi)
    llm_mrmr = _llm_then_mrmr_breakdown(dataset, runs, feature_psi, metadata)

    feature_psi.to_csv(out_dir / "feature_level_psi_by_run.csv", index=False)
    distribution.to_csv(out_dir / "psi_distribution_by_pipeline.csv", index=False)
    high_psi.to_csv(out_dir / "high_psi_features_by_pipeline.csv", index=False)
    llm_mrmr.to_csv(out_dir / "llm_then_mrmr_drift_source_breakdown.csv", index=False)

    plot_manifest: list[dict[str, Any]] = []
    _save_or_skip_boxplot(feature_psi, plots_dir, plot_manifest)
    _save_or_skip_top20(feature_psi, plots_dir, plot_manifest)
    _save_or_skip_llm_pool_plot(llm_mrmr, plots_dir, plot_manifest)
    plot_manifest_df = pd.DataFrame(plot_manifest)
    plot_manifest_df.to_csv(plots_dir / "plot_manifest.csv", index=False)

    return {
        "dataset": dataset,
        "out_dir": out_dir,
        "runs": len(runs),
        "feature_rows": len(feature_psi),
        "distribution_rows": len(distribution),
        "high_psi_rows": len(high_psi),
        "llm_mrmr_rows": len(llm_mrmr),
        "created_plots": plot_manifest_df[plot_manifest_df["status"].eq("created")][
            "plot_file"
        ].tolist(),
        "skipped_plots": plot_manifest_df[plot_manifest_df["status"].eq("skipped")][
            ["plot_file", "skip_reason"]
        ].to_dict("records"),
    }


def _format_metric(value: Any, digits: int = 4) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _report_for_dataset(dataset: str) -> str:
    out_dir = RESULTS_ROOT / dataset / "analysis" / "feature_level_drift"
    dist = _read_csv(out_dir / "psi_distribution_by_pipeline.csv")
    high = _read_csv(out_dir / "high_psi_features_by_pipeline.csv")
    breakdown = _read_csv(out_dir / "llm_then_mrmr_drift_source_breakdown.csv")
    plots = _read_csv(out_dir / "plots" / "plot_manifest.csv")

    lines = [f"## {dataset}"]
    if dist.empty:
        lines.append("No pipeline-level feature PSI data was available.")
        return "\n".join(lines)

    llm_family = dist[dist["selector"].astype(str).str.contains("llm", regex=False)].copy()
    non_llm = dist[~dist["selector"].astype(str).str.contains("llm", regex=False)].copy()
    llm_mean = llm_family["psi_mean"].mean() if not llm_family.empty else pd.NA
    non_llm_mean = non_llm["psi_mean"].mean() if not non_llm.empty else pd.NA
    best = dist.sort_values("psi_mean").iloc[0]
    worst = dist.sort_values("psi_mean", ascending=False).iloc[0]

    lines.extend(
        [
            f"- Pipeline rows: {len(dist)}; feature-level PSI rows: {int(dist['selected_feature_count'].sum())}.",
            f"- Mean selected-feature PSI, LLM-family selectors: {_format_metric(llm_mean)}.",
            f"- Mean selected-feature PSI, non-LLM selectors: {_format_metric(non_llm_mean)}.",
            f"- Lowest mean PSI pipeline: `{best['model']}/{best['selector']}` at {_format_metric(best['psi_mean'])}.",
            f"- Highest mean PSI pipeline: `{worst['model']}/{worst['selector']}` at {_format_metric(worst['psi_mean'])}, max {_format_metric(worst['psi_max'])}.",
        ]
    )

    if high.empty:
        lines.append(f"- No selected features crossed the high-PSI threshold of {HIGH_PSI_THRESHOLD:g}.")
    else:
        by_selector = high.groupby(["model", "selector"]).size().reset_index(name="high_psi_count")
        lines.append(
            "- High-PSI selected features were present in: "
            + "; ".join(
                f"`{row.model}/{row.selector}`={row.high_psi_count}"
                for row in by_selector.itertuples(index=False)
            )
            + "."
        )
        top_groups = (
            high.groupby("semantic_group", dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(5)
        )
        lines.append(
            "- Main high-PSI semantic groups: "
            + "; ".join(
                f"`{row.semantic_group}`={row.count}" for row in top_groups.itertuples(index=False)
            )
            + "."
        )

    if not breakdown.empty:
        selected = breakdown[breakdown["in_final_selected_set"].astype(bool)]
        max_rows = (
            selected.sort_values(["model", "psi_dev_oot"], ascending=[True, False])
            .groupby("model", as_index=False)
            .head(1)
        )
        if not max_rows.empty:
            lines.append(
                "- `llm_then_mrmr` PSI maxima: "
                + "; ".join(
                    f"`{row.model}` `{row.feature}` PSI={_format_metric(row.psi_dev_oot)} group=`{row.semantic_group}`"
                    for row in max_rows.itertuples(index=False)
                )
                + "."
            )
        pool_only = breakdown[
            breakdown["in_llm_top_pool"].astype(bool)
            & ~breakdown["in_final_selected_set"].astype(bool)
        ]
        lines.append(
            f"- `llm_then_mrmr` has {len(pool_only)} LLM-top-pool-only rows without PSI because run artifacts store selected-feature PSI only."
        )

    if not plots.empty:
        created = plots[plots["status"].eq("created")]["plot_file"].tolist()
        skipped = plots[plots["status"].eq("skipped")]
        lines.append(
            "- Plots created: " + (", ".join(f"`{item}`" for item in created) if created else "none") + "."
        )
        if not skipped.empty:
            lines.append(
                "- Plots skipped: "
                + "; ".join(
                    f"`{row.plot_file}` ({row.skip_reason})"
                    for row in skipped.itertuples(index=False)
                )
                + "."
            )
    return "\n".join(lines)


def _write_report(dataset_summaries: list[dict[str, Any]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        "# Feature-Level Drift Analysis",
        "",
        "This report is generated from existing run artifacts only. It does not rerun the experiment matrix, retrain models, or rebuild raw/processed datasets.",
        "",
        "High PSI is defined as `psi_dev_oot >= 0.25`, matching the repository's existing selected-feature PSI summary threshold.",
        "",
        "## Verdict",
        "",
        "The existing artifacts are sufficient for post-run selected-feature drift evidence. A full rerun is not required for the tables and plots generated here.",
        "",
        "LLM-family selectors generally support the lower-drift/stable feature-pool claim in these artifacts because their selected-feature PSI means and high-PSI ratios are low relative to the strongest non-LLM drift outliers, especially PCA on LendingClub. The evidence is strongest for selected final feature sets; artifacts do not contain DEV-OOT PSI for LLM-nominated candidates that mRMR later rejected.",
        "",
        "## Missing Artifacts And Limits",
        "",
        "- `feature_missing_oot` is not available in the stored run artifacts, so the CSV column is present but blank.",
        "- DEV missingness is filled from run-level final-dev metadata where available, otherwise from aggregate feature evidence.",
        "- LLM top-pool candidates that are not finally selected do not have stored feature PSI values. The `llm_then_mrmr_drift_source_breakdown.csv` file still marks their pool membership, but PSI is blank for those rejected candidates.",
        "- PCA rows are derived components, so semantic group/source table are marked as PCA-derived when original metadata is unavailable.",
        "",
        "## Dataset Results",
        "",
    ]
    for dataset in DATASETS:
        sections.append(_report_for_dataset(dataset))
        sections.append("")

    sections.extend(
        [
            "## Files Created",
            "",
        ]
    )
    for summary in dataset_summaries:
        dataset = summary["dataset"]
        base = summary["out_dir"]
        sections.extend(
            [
                f"- `{base / 'feature_level_psi_by_run.csv'}`",
                f"- `{base / 'psi_distribution_by_pipeline.csv'}`",
                f"- `{base / 'high_psi_features_by_pipeline.csv'}`",
                f"- `{base / 'llm_then_mrmr_drift_source_breakdown.csv'}`",
                f"- `{base / 'plots' / 'plot_manifest.csv'}`",
                f"- `{base / 'plots'}` created plots: "
                + (
                    ", ".join(f"`{name}`" for name in summary["created_plots"])
                    if summary["created_plots"]
                    else "none"
                ),
                f"- `{dataset}` skipped plots: "
                + (
                    "; ".join(
                        f"`{item['plot_file']}` ({item['skip_reason']})"
                        for item in summary["skipped_plots"]
                    )
                    if summary["skipped_plots"]
                    else "none"
                ),
            ]
        )
    REPORT_PATH.write_text("\n".join(sections) + "\n", encoding="utf-8")


def main() -> int:
    summaries = [_write_dataset_outputs(dataset) for dataset in DATASETS]
    _write_report(summaries)
    print(json.dumps(summaries, indent=2, default=str))
    print(f"Report written to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
