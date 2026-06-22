from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


FIXED_METHODS = ["clip", "clip_then_mrmr", "mrmr", "llm", "llm_then_mrmr"]
METHOD_LABELS = {
    "clip": "CLIP",
    "clip_then_mrmr": "CLIP->mRMR",
    "mrmr": "mRMR",
    "llm": "LLM",
    "llm_then_mrmr": "LLM->mRMR",
}
METHOD_COLORS = {
    "clip": "#2a6fbb",
    "clip_then_mrmr": "#009e73",
    "mrmr": "#777777",
    "llm": "#d55e00",
    "llm_then_mrmr": "#cc79a7",
}
PANEL_ORDER = [
    ("homecredit", "lr", "Home Credit LR"),
    ("homecredit", "catboost", "Home Credit CatBoost"),
    ("lendingclub_v2", "lr", "LendingClub v2 LR"),
    ("lendingclub_v2", "catboost", "LendingClub v2 CatBoost"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the fixed five-figure CLIP final-report plot set.")
    parser.add_argument("--input-root", type=Path, default=Path("results/clip/final_evaluation"))
    parser.add_argument("--training-root", type=Path, default=Path("results/clip/training"))
    parser.add_argument("--output-root", type=Path, default=Path("results/clip/final_report"))
    return parser.parse_args()


def _load_comparison(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "comparison_with_frozen_baselines.csv")
    frame = frame[frame["selector"].isin(FIXED_METHODS)].copy()
    if set(frame["selector"].unique()) != set(FIXED_METHODS):
        raise ValueError("comparison table does not contain the fixed five-method set")
    return frame


def _panel_axes() -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    return fig, axes.ravel()


def _method_positions() -> np.ndarray:
    return np.arange(len(FIXED_METHODS))


def _bar_panel(ax: plt.Axes, values: pd.Series, *, title: str, ylabel: str, ylim: tuple[float, float], decimals: int) -> None:
    x = _method_positions()
    y = [float(values.get(method, np.nan)) for method in FIXED_METHODS]
    ax.bar(x, y, color=[METHOD_COLORS[method] for method in FIXED_METHODS], width=0.72)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[method] for method in FIXED_METHODS], rotation=25, ha="right")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(y):
        if not np.isnan(value):
            ax.text(idx, value + (ylim[1] - ylim[0]) * 0.015, f"{value:.{decimals}f}", ha="center", va="bottom", fontsize=8)


def plot_oot_auc(comparison: pd.DataFrame, output: Path) -> None:
    fig, axes = _panel_axes()
    for ax, (dataset, model, title) in zip(axes, PANEL_ORDER, strict=True):
        subset = comparison[(comparison["dataset_name"].eq(dataset)) & (comparison["model"].eq(model))]
        values = subset.set_index("selector")["oot_auc"]
        _bar_panel(ax, values, title=title, ylabel="OOT AUC", ylim=(0.50, 0.82), decimals=3)
    fig.suptitle("Figure 1. Main OOT AUC comparison", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_auc_delta(significance: pd.DataFrame, output: Path) -> None:
    comparisons = [
        ("clip", "mrmr", "CLIP - mRMR"),
        ("clip", "llm", "CLIP - LLM"),
        ("clip_then_mrmr", "llm_then_mrmr", "CLIP->mRMR - LLM->mRMR"),
    ]
    fig, axes = _panel_axes()
    for ax, (dataset, model, title) in zip(axes, PANEL_ORDER, strict=True):
        rows = []
        for new_selector, baseline_selector, label in comparisons:
            match = significance[
                significance["dataset"].eq(dataset)
                & significance["model"].eq(model)
                & significance["new_selector"].eq(new_selector)
                & significance["baseline_selector"].eq(baseline_selector)
                & significance["status"].eq("ok")
            ]
            if not match.empty:
                row = match.iloc[0].to_dict()
                row["label"] = label
                rows.append(row)
        y = np.arange(len(rows))
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.8)
        if rows:
            diff = np.array([float(row["point_estimate_difference"]) for row in rows])
            low = np.array([float(row["ci95_lower"]) for row in rows])
            high = np.array([float(row["ci95_upper"]) for row in rows])
            ax.errorbar(diff, y, xerr=[diff - low, high - diff], fmt="o", color="#2a6fbb", ecolor="#666666", capsize=3)
            ax.set_yticks(y)
            ax.set_yticklabels([row["label"] for row in rows])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Paired OOT AUC difference")
        ax.set_xlim(-0.14, 0.02)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Figure 2. Paired OOT AUC differences with uncertainty", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_score_psi(comparison: pd.DataFrame, output: Path) -> None:
    fig, axes = _panel_axes()
    for ax, (dataset, model, title) in zip(axes, PANEL_ORDER, strict=True):
        subset = comparison[(comparison["dataset_name"].eq(dataset)) & (comparison["model"].eq(model))]
        values = subset.set_index("selector")["model_score_psi"]
        _bar_panel(ax, values, title=title, ylabel="DEV-to-OOT model score PSI", ylim=(0.0, 0.27), decimals=3)
        ax.axhline(0.10, color="#d55e00", linestyle="--", linewidth=1.0)
        ax.axhline(0.25, color="#b00020", linestyle="--", linewidth=1.0)
        ax.text(4.42, 0.102, "0.10", fontsize=8, color="#d55e00", va="bottom")
        ax.text(4.42, 0.252, "0.25", fontsize=8, color="#b00020", va="bottom")
    fig.suptitle("Figure 3. DEV-to-OOT model score PSI comparison", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _base_family(feature: str) -> str:
    parts = str(feature).split("_")
    if len(parts) >= 3 and parts[-1] in {"MEAN", "SUM", "MIN", "MAX", "VAR", "FLAG"}:
        return "_".join(parts[:-1])
    return str(feature)


def _selector_output_features(row: pd.Series) -> pd.DataFrame:
    if row["result_origin"] == "clip_extension":
        path = Path(str(row["run_dir"])) / "features" / "final_selected_features.csv"
    else:
        path = Path(str(row["output_folder"])) / "features" / "final_selected_features.csv"
    return pd.read_csv(path)


def build_semantic_plot_frame(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in comparison.to_dict("records"):
        source = pd.Series(row)
        selected = _selector_output_features(source)
        feature_col = "feature_name" if "feature_name" in selected.columns else "feature"
        features = selected[feature_col].astype(str)
        groups = (
            selected["semantic_group"].fillna("unknown").astype(str)
            if "semantic_group" in selected.columns
            else pd.Series(["unknown"] * len(selected))
        )
        families = features.map(_base_family)
        counts = groups.value_counts()
        rows.append(
            {
                "dataset_name": row["dataset_name"],
                "model": row["model"],
                "selector": row["selector"],
                "semantic_group_count": int(groups.nunique()),
                "largest_semantic_group_share": float(counts.iloc[0] / len(groups)) if len(groups) else np.nan,
                "repeated_base_family_share": float(families.duplicated(keep=False).mean()) if len(families) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_semantic_redundancy(semantic_frame: pd.DataFrame, output: Path) -> None:
    fig, axes_grid = plt.subplots(2, 2, figsize=(13, 8.8))
    axes = axes_grid.ravel()
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.15, wspace=0.25, hspace=0.34)
    handles: dict[str, Any] = {}
    for ax, (dataset, model, title) in zip(axes, PANEL_ORDER, strict=True):
        subset = semantic_frame[(semantic_frame["dataset_name"].eq(dataset)) & (semantic_frame["model"].eq(model))]
        for _, row in subset.iterrows():
            method = str(row["selector"])
            point = ax.scatter(
                row["semantic_group_count"],
                row["repeated_base_family_share"],
                s=80,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                alpha=0.9,
            )
            handles[method] = point
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Semantic-group count")
        ax.set_ylabel("Repeated base-family share")
        ax.set_xlim(0, max(12, int(semantic_frame["semantic_group_count"].max()) + 2))
        ax.set_ylim(-0.02, min(1.0, max(0.25, float(semantic_frame["repeated_base_family_share"].max()) + 0.08)))
        ax.grid(alpha=0.25)
    fig.suptitle("Figure 4. Semantic coverage and redundancy", fontsize=14)
    fig.legend(
        [handles[method] for method in FIXED_METHODS if method in handles],
        [METHOD_LABELS[method] for method in FIXED_METHODS if method in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=5,
        frameon=False,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_seed_robustness(seed_comparison: pd.DataFrame, selected_seed: int, output: Path) -> None:
    frame = seed_comparison.sort_values("seed").copy()
    colors = ["#009e73" if int(seed) == selected_seed else "#777777" for seed in frame["seed"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].bar(frame["seed"].astype(str), frame["best_validation_loss"], color=colors)
    axes[0].set_title("Validation contrastive loss")
    axes[0].set_xlabel("Seed")
    axes[0].set_ylabel("Loss (lower is better)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(frame["seed"].astype(str), frame["best_validation_mrr"], color=colors)
    axes[1].set_title("Validation retrieval MRR")
    axes[1].set_xlabel("Seed")
    axes[1].set_ylabel("MRR (higher is better)")
    axes[1].grid(axis="y", alpha=0.25)
    for ax in axes:
        for label in ax.get_xticklabels():
            if label.get_text() == str(selected_seed):
                label.set_fontweight("bold")
    fig.suptitle(f"Figure 5. CLIP seed robustness (selected seed {selected_seed})", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _manifest_row(
    *,
    file: str,
    question: str,
    methods: list[str],
    scope: str,
    source_artifacts: list[str],
    source_columns: list[str],
    main_finding: str,
    limitation: str,
    reason: str,
) -> dict[str, Any]:
    if len(methods) > 5:
        raise ValueError(f"{file}: main report figure includes more than five methods")
    return {
        "plot_file": file,
        "report_section": "main_report",
        "research_question": question,
        "included_methods": ", ".join(methods),
        "dataset_model_scope": scope,
        "source_artifacts": "; ".join(source_artifacts),
        "source_columns": ", ".join(source_columns),
        "main_finding": main_finding,
        "limitation": limitation,
        "reason_the_plot_is_necessary": reason,
    }


def write_manifest(rows: list[dict[str, Any]], output_root: Path) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "plot_manifest.csv", index=False)
    (output_root / "plot_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    figures = args.output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    comparison = _load_comparison(args.input_root)
    significance = pd.read_csv(args.input_root / "statistical_significance_summary.csv")
    seed_comparison = pd.read_csv(args.training_root / "seed_comparison.csv")
    model_selection = json.loads((args.training_root / "model_selection_manifest.json").read_text(encoding="utf-8"))
    selected_seed = int(model_selection["selected_seed"])
    semantic_frame = build_semantic_plot_frame(comparison)
    semantic_frame.to_csv(args.output_root / "semantic_redundancy_plot_data.csv", index=False)

    plot_oot_auc(comparison, figures / "01_oot_auc_main_comparison.png")
    plot_auc_delta(significance, figures / "02_oot_auc_delta_with_uncertainty.png")
    plot_score_psi(comparison, figures / "03_score_psi_comparison.png")
    plot_semantic_redundancy(semantic_frame, figures / "04_semantic_coverage_redundancy.png")
    plot_seed_robustness(seed_comparison, selected_seed, figures / "05_clip_seed_robustness.png")

    manifest = [
        _manifest_row(
            file="figures/01_oot_auc_main_comparison.png",
            question="Are the CLIP selectors predictively competitive with the main statistical and LLM baselines?",
            methods=FIXED_METHODS,
            scope="Home Credit LR/CatBoost and LendingClub v2 LR/CatBoost panels",
            source_artifacts=["results/clip/final_evaluation/comparison_with_frozen_baselines.csv"],
            source_columns=["dataset_name", "model", "selector", "oot_auc"],
            main_finding="Shows absolute OOT AUC for the prespecified five-method comparison set.",
            limitation="AUC alone does not show uncertainty or calibration.",
            reason="This is the primary predictive comparison requested for the final report.",
        ),
        _manifest_row(
            file="figures/02_oot_auc_delta_with_uncertainty.png",
            question="What is the paired OOT AUC difference for the most important prespecified comparisons?",
            methods=["clip", "clip_then_mrmr", "mrmr", "llm", "llm_then_mrmr"],
            scope="Home Credit LR/CatBoost and LendingClub v2 LR/CatBoost panels",
            source_artifacts=["results/clip/final_evaluation/statistical_significance_summary.csv"],
            source_columns=["point_estimate_difference", "ci95_lower", "ci95_upper", "status"],
            main_finding="Shows paired AUC deltas and confidence intervals against selected baselines.",
            limitation="Only comparisons with aligned saved prediction files are shown.",
            reason="Uncertainty around the key AUC differences is more informative than crowded ROC curves.",
        ),
        _manifest_row(
            file="figures/03_score_psi_comparison.png",
            question="Did any predictive gain come with materially worse score-distribution drift?",
            methods=FIXED_METHODS,
            scope="Home Credit LR/CatBoost and LendingClub v2 LR/CatBoost panels",
            source_artifacts=["results/clip/final_evaluation/comparison_with_frozen_baselines.csv"],
            source_columns=["dataset_name", "model", "selector", "model_score_psi"],
            main_finding="Shows DEV-to-OOT model score PSI against 0.10 and 0.25 reference lines.",
            limitation="Score PSI is saved run-level output, not independently recomputed from persisted DEV scores.",
            reason="The plot checks whether performance changes coincide with score-distribution drift.",
        ),
        _manifest_row(
            file="figures/04_semantic_coverage_redundancy.png",
            question="Do CLIP-based selectors provide broader semantic coverage without introducing excessive redundancy?",
            methods=FIXED_METHODS,
            scope="Home Credit LR/CatBoost and LendingClub v2 LR/CatBoost panels",
            source_artifacts=[
                "results/clip/final_evaluation/comparison_with_frozen_baselines.csv",
                "per-run features/final_selected_features.csv",
            ],
            source_columns=["feature_name", "semantic_group"],
            main_finding="Plots semantic-group count against repeated base-family share for the fixed methods.",
            limitation="Higher semantic-group count is descriptive and does not imply better predictive quality.",
            reason="This is the clearest compact view of semantic breadth versus redundancy.",
        ),
        _manifest_row(
            file="figures/05_clip_seed_robustness.png",
            question="Is the learned CLIP representation materially dependent on one favorable random seed?",
            methods=["clip"],
            scope="Prompt 5 seed-level representation training evidence",
            source_artifacts=[
                "results/clip/training/seed_comparison.csv",
                "results/clip/training/model_selection_manifest.json",
            ],
            source_columns=["seed", "best_validation_loss", "best_validation_mrr", "selected_seed"],
            main_finding="Shows selected seed 55 relative to other trained seeds on loss and retrieval MRR.",
            limitation="This is representation-level evidence, not downstream multi-seed evaluation.",
            reason="The figure documents seed robustness without implying downstream seed robustness.",
        ),
    ]
    write_manifest(manifest, args.output_root)
    print(json.dumps({"main_figure_count": len(manifest), "figures": [row["plot_file"] for row in manifest]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
