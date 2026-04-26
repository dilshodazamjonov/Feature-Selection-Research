from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(slots=True)
class ExperimentPlotData:
    label: str
    exp_dir: Path
    frame: pd.DataFrame
    oot_metrics: dict[str, float]


FOLD_METRICS = [
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
    "lift_at_10",
    "bad_rate_capture_at_10",
]


def _numeric_folds(df: pd.DataFrame) -> pd.DataFrame:
    fold_numeric = pd.to_numeric(df["fold"], errors="coerce")
    mask = fold_numeric.notna()
    filtered = df.loc[mask].copy()
    filtered["fold"] = fold_numeric.loc[mask].astype(int)
    return filtered.sort_values("fold").reset_index(drop=True)


def _format_month_bucket(day_value: float) -> str:
    months_back = int(round(abs(float(day_value)) / 30.0))
    return f"M-{months_back}"


def _format_window_label(start_day: float, end_day: float) -> str:
    return f"{_format_month_bucket(start_day)} to {_format_month_bucket(end_day)}"


def _infer_label(exp_dir: Path) -> str:
    manifest_path = exp_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            model = manifest.get("model")
            selector = manifest.get("selector")
            experiment_type = manifest.get("experiment_type")
            if model and selector and experiment_type:
                return f"{model}_{experiment_type}_{selector}"
        except Exception:
            pass
    return exp_dir.name


def load_plot_data(exp_dir: str | Path, label: str | None = None) -> ExperimentPlotData:
    exp_path = Path(exp_dir)
    results_dir = exp_path / "results"
    cv_path = results_dir / "cv_results.csv"
    if not cv_path.exists():
        raise ValueError(f"Missing cv_results.csv for plotting: {exp_path}")

    frame = _numeric_folds(pd.read_csv(cv_path))
    if not {"val_time_start", "val_time_end"}.issubset(frame.columns):
        legacy_time_path = results_dir / "fold_time_info.csv"
        if not legacy_time_path.exists():
            raise ValueError(f"Missing temporal fold columns for plotting: {exp_path}")
        time_df = _numeric_folds(pd.read_csv(legacy_time_path))
        frame = time_df.merge(frame, on="fold", how="left")

    frame["month_label"] = frame.apply(
        lambda row: _format_window_label(row["val_time_start"], row["val_time_end"]),
        axis=1,
    )

    oot_metrics: dict[str, float] = {}
    oot_path = results_dir / "oot_test_results.csv"
    if oot_path.exists():
        oot_df = pd.read_csv(oot_path)
        if not oot_df.empty:
            for metric in [
                "auc",
                "gini",
                "ks",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "lift_at_10",
                "bad_rate_capture_at_10",
            ]:
                if metric in oot_df.columns and pd.notna(oot_df.iloc[0][metric]):
                    oot_metrics[metric] = float(oot_df.iloc[0][metric])

    return ExperimentPlotData(
        label=label or _infer_label(exp_path),
        exp_dir=exp_path,
        frame=frame,
        oot_metrics=oot_metrics,
    )


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install dependencies with `uv sync`."
        ) from exc
    return plt


def _save_line_plot(
    *,
    plot_data: Iterable[ExperimentPlotData],
    metric: str,
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> bool:
    plt = _require_matplotlib()
    data = list(plot_data)
    series_to_plot = [
        item
        for item in data
        if metric in item.frame.columns and item.frame[metric].notna().any()
    ]
    if not series_to_plot:
        return False

    plt.figure(figsize=(14, 6))
    for item in series_to_plot:
        frame = item.frame
        plt.plot(
            frame["month_label"],
            frame[metric],
            marker="o",
            linewidth=2,
            markersize=7,
            label=item.label,
        )

    plt.title(title)
    plt.xlabel("Validation Month Window")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    return True


def _save_grouped_bar(
    *,
    df: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> bool:
    if df.empty or not {x, y, group}.issubset(df.columns) or df[y].dropna().empty:
        return False
    plt = _require_matplotlib()
    pivot = df.pivot_table(index=x, columns=group, values=y, aggfunc="mean")
    if pivot.empty:
        return False
    pivot.plot(kind="bar", figsize=(12, 6))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(x)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    return True


def _save_scatter(
    *,
    df: pd.DataFrame,
    x: str,
    y: str,
    label: str,
    group: str,
    title: str,
    output_path: str | Path,
) -> bool:
    if df.empty or not {x, y, label, group}.issubset(df.columns):
        return False
    plot_df = df[[x, y, label, group]].dropna()
    if plot_df.empty:
        return False
    plt = _require_matplotlib()
    plt.figure(figsize=(10, 7))
    for group_value, group_df in plot_df.groupby(group):
        plt.scatter(group_df[x], group_df[y], label=group_value, s=80)
        for _, row in group_df.iterrows():
            plt.annotate(str(row[label]), (row[x], row[y]), fontsize=8, alpha=0.8)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight")
    plt.close()
    return True


def generate_experiment_plots(
    *,
    experiments: Iterable[ExperimentPlotData],
    output_dir: str | Path,
) -> dict[str, bool]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_data = list(experiments)
    unique_periods = {
        str(label)
        for item in plot_data
        for label in item.frame.get("month_label", pd.Series(dtype=str)).dropna().tolist()
    }
    use_monthly_names = len(unique_periods) >= 2
    temporal_prefix = "monthly" if use_monthly_names else "temporal"

    results: dict[str, bool] = {
        f"{temporal_prefix}_gini_trend": _save_line_plot(
            plot_data=plot_data,
            metric="gini",
            title="Temporal Validation Gini Trend",
            ylabel="Gini",
            output_path=output_path / f"{temporal_prefix}_gini_trend.png",
        ),
        f"{temporal_prefix}_psi_trend": _save_line_plot(
            plot_data=plot_data,
            metric="psi_model",
            title="Temporal Validation Model PSI Trend",
            ylabel="Model PSI",
            output_path=output_path / f"{temporal_prefix}_psi_trend.png",
        ),
        f"{temporal_prefix}_lift_trend": _save_line_plot(
            plot_data=plot_data,
            metric="lift_at_10",
            title="Temporal Validation Lift@10 Trend",
            ylabel="Lift@10",
            output_path=output_path / f"{temporal_prefix}_lift_trend.png",
        ),
    }

    monthly_rows: list[dict[str, object]] = []
    for item in plot_data:
        for _, row in item.frame.iterrows():
            monthly_rows.append(
                {
                    "label": item.label,
                    "fold": int(row["fold"]),
                    "month_label": row["month_label"],
                    "val_time_start": row["val_time_start"],
                    "val_time_end": row["val_time_end"],
                    **{metric: row.get(metric, pd.NA) for metric in FOLD_METRICS},
                }
            )
    temporal_table_name = (
        "monthly_metric_table.csv" if use_monthly_names else "temporal_metric_table.csv"
    )
    pd.DataFrame(monthly_rows).to_csv(output_path / temporal_table_name, index=False)

    pd.DataFrame(
        [
            {
                "label": item.label,
                "exp_dir": str(item.exp_dir),
                **{f"oot_{metric}": value for metric, value in item.oot_metrics.items()},
            }
            for item in plot_data
        ]
    ).to_csv(output_path / "oot_metric_table.csv", index=False)

    return results


def generate_matrix_comparison_plots(
    *,
    comparison_df: pd.DataFrame,
    experiments: Iterable[ExperimentPlotData],
    output_dir: str | Path,
) -> dict[str, bool]:
    """Generate the clean research plot bundle from completed matrix runs."""
    _ = list(experiments)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data = comparison_df.copy()

    stability_columns = [
        "model",
        "selector",
        "experiment_type",
        "nogueira_stability",
        "kuncheva_stability",
        "mean_pairwise_jaccard",
        "stable_feature_count_80",
        "stable_feature_ratio_80",
    ]
    data.reindex(columns=stability_columns).to_csv(
        output_path / "stability_metric_table.csv",
        index=False,
    )

    return {
        "oot_performance_comparison": _save_grouped_bar(
            df=data,
            x="selector",
            y="oot_gini",
            group="model",
            title="OOT Gini by Selector and Model",
            ylabel="OOT Gini",
            output_path=output_path / "oot_performance_comparison.png",
        ),
        "stability_comparison": _save_grouped_bar(
            df=data,
            x="selector",
            y="nogueira_stability",
            group="model",
            title="Nogueira Stability by Selector and Model",
            ylabel="Nogueira Stability",
            output_path=output_path / "stability_comparison.png",
        ),
        "performance_vs_stability": _save_scatter(
            df=data,
            x="nogueira_stability",
            y="oot_gini",
            label="selector",
            group="model",
            title="OOT Gini vs Nogueira Stability",
            output_path=output_path / "performance_vs_stability.png",
        ),
        "feature_count_vs_gini": _save_scatter(
            df=data,
            x="selected_feature_count",
            y="oot_gini",
            label="selector",
            group="model",
            title="OOT Gini vs Selected Feature Count",
            output_path=output_path / "feature_count_vs_gini.png",
        ),
        "selected_feature_psi_comparison": _save_grouped_bar(
            df=data,
            x="selector",
            y="selected_feature_psi_mean",
            group="model",
            title="Selected Feature PSI by Selector and Model",
            ylabel="Mean Selected Feature PSI",
            output_path=output_path / "selected_feature_psi_comparison.png",
        ),
        "model_score_psi_comparison": _save_grouped_bar(
            df=data,
            x="selector",
            y="model_score_psi",
            group="model",
            title="Model Score PSI by Selector and Model",
            ylabel="Model Score PSI",
            output_path=output_path / "model_score_psi_comparison.png",
        ),
        "lift_at_10_comparison": _save_grouped_bar(
            df=data,
            x="selector",
            y="lift_at_10",
            group="model",
            title="Lift@10 by Selector and Model",
            ylabel="Lift@10",
            output_path=output_path / "lift_at_10_comparison.png",
        ),
    }
