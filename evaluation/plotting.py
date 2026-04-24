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


FOLD_METRIC_SOURCES: dict[str, tuple[str, str]] = {
    "auc": ("evaluation_metrics_summary.csv", "auc"),
    "gini": ("evaluation_metrics_summary.csv", "gini"),
    "ks": ("evaluation_metrics_summary.csv", "ks"),
    "precision": ("evaluation_metrics_summary.csv", "precision"),
    "recall": ("evaluation_metrics_summary.csv", "recall"),
    "f1": ("evaluation_metrics_summary.csv", "f1"),
    "accuracy": ("evaluation_metrics_summary.csv", "accuracy"),
    "selected_features": ("stability_metrics_summary.csv", "selected_features"),
    "psi_feature_mean": ("stability_metrics_summary.csv", "psi_feature_mean"),
    "psi_feature_max": ("stability_metrics_summary.csv", "psi_feature_max"),
    "psi_model": ("stability_metrics_summary.csv", "psi_model"),
    "jaccard_similarity": ("stability_metrics_summary.csv", "jaccard_similarity"),
}


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
    name = exp_dir.name
    tokens = name.split("_")
    if len(tokens) >= 2:
        return "_".join(tokens[:-2])
    return name


def load_plot_data(exp_dir: str | Path, label: str | None = None) -> ExperimentPlotData:
    exp_path = Path(exp_dir)
    results_dir = exp_path / "results"

    time_df = pd.read_csv(results_dir / "fold_time_info.csv")
    time_df = time_df.sort_values("fold").reset_index(drop=True)
    time_df["month_label"] = time_df.apply(
        lambda row: _format_window_label(row["val_time_start"], row["val_time_end"]),
        axis=1,
    )

    merged = time_df.copy()
    for metric_name, (filename, column) in FOLD_METRIC_SOURCES.items():
        metric_path = results_dir / filename
        if not metric_path.exists():
            continue
        metric_df = pd.read_csv(metric_path)
        if "fold" not in metric_df.columns or column not in metric_df.columns:
            continue
        metric_df = _numeric_folds(metric_df[["fold", column]])
        merged = merged.merge(metric_df, on="fold", how="left")
        if column != metric_name:
            merged = merged.rename(columns={column: metric_name})

    oot_metrics: dict[str, float] = {}
    oot_path = results_dir / "oot_test_results.csv"
    if oot_path.exists():
        oot_df = pd.read_csv(oot_path)
        if not oot_df.empty:
            for metric in ["auc", "gini", "ks", "precision", "recall", "f1", "accuracy"]:
                if metric in oot_df.columns:
                    oot_metrics[metric] = float(oot_df.iloc[0][metric])

    return ExperimentPlotData(
        label=label or _infer_label(exp_path),
        exp_dir=exp_path,
        frame=merged,
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
    series_to_plot = [item for item in data if metric in item.frame.columns and item.frame[metric].notna().any()]
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


def _save_bar_plot(
    *,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> bool:
    finite_pairs = [(label, value) for label, value in zip(labels, values) if pd.notna(value)]
    if not finite_pairs:
        return False

    plt = _require_matplotlib()
    plot_labels = [label for label, _ in finite_pairs]
    plot_values = [value for _, value in finite_pairs]

    plt.figure(figsize=(10, 6))
    plt.bar(plot_labels, plot_values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3, axis="y")
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
    plot_data = list(experiments)

    results: dict[str, bool] = {}
    line_metrics = {
        "gini": "Gini over Validation Windows",
        "auc": "ROC-AUC over Validation Windows",
        "ks": "KS over Validation Windows",
        "precision": "Precision over Validation Windows",
        "recall": "Recall over Validation Windows",
        "f1": "F1 over Validation Windows",
        "accuracy": "Accuracy over Validation Windows",
        "selected_features": "Selected Feature Count over Validation Windows",
        "psi_feature_mean": "Mean Feature PSI over Validation Windows",
        "psi_feature_max": "Max Feature PSI over Validation Windows",
        "psi_model": "Model PSI over Validation Windows",
        "jaccard_similarity": "Jaccard Stability over Validation Windows",
    }

    ylabels = {
        "gini": "Gini",
        "auc": "ROC-AUC",
        "ks": "KS",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "accuracy": "Accuracy",
        "selected_features": "Selected Features",
        "psi_feature_mean": "Mean Feature PSI",
        "psi_feature_max": "Max Feature PSI",
        "psi_model": "Model PSI",
        "jaccard_similarity": "Jaccard Similarity",
    }

    for metric, title in line_metrics.items():
        results[f"{metric}_over_time"] = _save_line_plot(
            plot_data=plot_data,
            metric=metric,
            title=title,
            ylabel=ylabels[metric],
            output_path=output_path / f"{metric}_over_time.png",
        )

    for metric, ylabel in [("gini", "Gini"), ("auc", "ROC-AUC"), ("ks", "KS")]:
        results[f"oot_{metric}_comparison"] = _save_bar_plot(
            labels=[item.label for item in plot_data],
            values=[item.oot_metrics.get(metric, float("nan")) for item in plot_data],
            title=f"OOT {ylabel} Comparison",
            ylabel=ylabel,
            output_path=output_path / f"oot_{metric}_comparison.png",
        )

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
                    **{
                        metric: row.get(metric, pd.NA)
                        for metric in FOLD_METRIC_SOURCES
                    },
                }
            )
    if monthly_rows:
        pd.DataFrame(monthly_rows).to_csv(output_path / "monthly_metric_table.csv", index=False)

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
