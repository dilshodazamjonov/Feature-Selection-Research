from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.utils.hashing import sha256_file
from credit_risk_fs.utils.io import read_json, write_json


AGG_ROOT = Path("results/clip_v2/final_evaluation")
ANALYSIS_ROOT = Path("results/clip_v2/final_analysis")
REPORT_PATHS = {
    "markdown": Path("reports/clip_v2_credit_risk_report.md"),
    "docx": Path("reports/clip_v2_credit_risk_report.docx"),
    "pdf": Path("reports/clip_v2_credit_risk_report.pdf"),
    "verdict": Path("reports/clip_v2_scientific_verdict.md"),
    "limitations": Path("reports/clip_v2_limitations.md"),
    "manifest": Path("reports/clip_v2_reproducibility_manifest.json"),
}

V2_FIXED_METHODS = ["clip_v2", "clip_v2_then_mrmr", "mrmr", "llm", "llm_then_mrmr"]
V2_METHOD_LABELS = {
    "clip_v2": "CLIP-v2",
    "clip_v2_then_mrmr": "CLIP-v2->mRMR",
    "mrmr": "mRMR",
    "llm": "LLM",
    "llm_then_mrmr": "LLM->mRMR",
}
V2_METHOD_COLORS = {
    "clip_v2": "#2a6fbb",
    "clip_v2_then_mrmr": "#009e73",
    "mrmr": "#777777",
    "llm": "#d55e00",
    "llm_then_mrmr": "#cc79a7",
}
V1_TO_V2_SELECTOR = {"clip": "clip_v2", "clip_then_mrmr": "clip_v2_then_mrmr"}
PANEL_ORDER = [
    ("homecredit", "lr", "Home Credit LR"),
    ("homecredit", "catboost", "Home Credit CatBoost"),
    ("lendingclub_v2", "lr", "LendingClub v2 LR"),
    ("lendingclub_v2", "catboost", "LendingClub v2 CatBoost"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CLIP-v2 final analysis tables and report files from saved artifacts.")
    parser.add_argument("--config", default="configs/clip_v2/analysis.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    readiness = _readiness()
    if args.dry_run or args.status or not args.execute:
        print(json.dumps({"status": readiness["status"], "execute": False, "model_trained": False, **readiness}, indent=2, default=str))
        return 0
    if readiness["status"] != "ready":
        print(json.dumps({"status": "failed", "reason": "complete aggregate artifacts are required", **readiness}, indent=2, default=str))
        return 1
    outputs = build_analysis()
    print(json.dumps({"status": "complete", "model_trained": False, "prediction_regenerated": False, "outputs": outputs}, indent=2))
    return 0


def _readiness() -> dict[str, Any]:
    required = [
        AGG_ROOT / "run_manifest.json",
        AGG_ROOT / "evaluation_summary.csv",
        AGG_ROOT / "selected_features_long.csv",
        AGG_ROOT / "selected_feature_summary.csv",
        AGG_ROOT / "semantic_coverage_summary.csv",
        AGG_ROOT / "redundancy_summary.csv",
        AGG_ROOT / "runtime_summary.csv",
        AGG_ROOT / "score_psi_summary.csv",
        AGG_ROOT / "aggregate_validation.json",
    ]
    missing = [str(path).replace("\\", "/") for path in required if not path.exists()]
    validation = read_json(AGG_ROOT / "aggregate_validation.json") if (AGG_ROOT / "aggregate_validation.json").exists() else {}
    ready = not missing and validation.get("complete") is True and int(validation.get("run_count", 0)) == 8
    return {
        "status": "ready" if ready else "incomplete",
        "missing_inputs": missing,
        "aggregate_validation": validation,
        "analysis_root": str(ANALYSIS_ROOT).replace("\\", "/"),
        "reports": {key: str(path).replace("\\", "/") for key, path in REPORT_PATHS.items()},
    }


def build_final_report_plots(
    *,
    master: pd.DataFrame,
    baselines: pd.DataFrame,
    semantic: pd.DataFrame,
    redundancy: pd.DataFrame,
    v1: pd.DataFrame,
    semantic_map_plot: Path,
) -> pd.DataFrame:
    plot_dir = ANALYSIS_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    comparison = _normalise_plot_comparison(baselines)
    plot_oot_auc_v2(comparison, plot_dir / "01_oot_auc_main_comparison.png")
    plot_auc_delta_v2(master, comparison, v1, plot_dir / "02_oot_auc_delta_with_uncertainty.png")
    plot_score_psi_v2(comparison, plot_dir / "03_score_psi_comparison.png")
    plot_semantic_redundancy_v2(semantic, redundancy, plot_dir / "04_semantic_coverage_redundancy.png")
    plot_seed_robustness_v2(plot_dir / "05_clip_seed_robustness.png")
    return pd.DataFrame(
        [
            _plot_manifest_row(
                "01_oot_auc_main_comparison",
                plot_dir / "01_oot_auc_main_comparison.png",
                "main",
                "Shows OOT AUC for CLIP-v2, CLIP-v2->mRMR, and frozen statistical/LLM baselines.",
            ),
            _plot_manifest_row(
                "02_oot_auc_delta_with_uncertainty",
                plot_dir / "02_oot_auc_delta_with_uncertainty.png",
                "main",
                "Shows saved OOT AUC point deltas; v2 paired-bootstrap intervals are explicitly not computed.",
            ),
            _plot_manifest_row(
                "03_score_psi_comparison",
                plot_dir / "03_score_psi_comparison.png",
                "main",
                "Shows DEV-to-OOT score PSI for CLIP-v2 and the fixed baseline comparison set.",
            ),
            _plot_manifest_row(
                "04_semantic_coverage_redundancy",
                plot_dir / "04_semantic_coverage_redundancy.png",
                "main",
                "Shows CLIP-v2 semantic-group breadth against selected-feature redundancy.",
            ),
            _plot_manifest_row(
                "05_clip_seed_robustness",
                plot_dir / "05_clip_seed_robustness.png",
                "main",
                "Shows CLIP-v2 seed-level validation loss and retrieval MRR with the selected seed highlighted.",
            ),
            _plot_manifest_row(
                "06_clip_v1_vs_v2_feature_semantic_map",
                semantic_map_plot,
                "supplementary",
                "Shows selected-feature coverage across frozen text-embedding semantic space.",
            ),
        ]
    )


def _plot_manifest_row(plot_id: str, path: Path, role: str, reason: str) -> dict[str, str]:
    return {
        "plot_id": plot_id,
        "path": str(path).replace("\\", "/"),
        "figure_role": role,
        "reason": reason,
    }


def _normalise_plot_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "dataset" not in out.columns:
        out["dataset"] = ""
    if "dataset_name" in out.columns:
        out["dataset"] = out["dataset"].fillna(out["dataset_name"])
    out["dataset"] = out["dataset"].astype(str)
    out["model"] = out["model"].astype(str)
    out["selector"] = out["selector"].astype(str)
    if "auc" not in out.columns:
        out["auc"] = np.nan
    if "oot_auc" in out.columns:
        out["auc"] = pd.to_numeric(out["auc"], errors="coerce").fillna(pd.to_numeric(out["oot_auc"], errors="coerce"))
    else:
        out["auc"] = pd.to_numeric(out["auc"], errors="coerce")
    out["model_score_psi"] = pd.to_numeric(out.get("model_score_psi", np.nan), errors="coerce")
    out = out[out["selector"].isin(V2_FIXED_METHODS)].copy()
    return out.sort_values(["dataset", "model", "selector"], kind="mergesort")


def _panel_axes() -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    return fig, axes.ravel()


def _method_positions() -> np.ndarray:
    return np.arange(len(V2_FIXED_METHODS))


def _bar_panel(ax: plt.Axes, values: pd.Series, *, title: str, ylabel: str, ylim: tuple[float, float], decimals: int) -> None:
    x = _method_positions()
    y = [float(values.get(method, np.nan)) for method in V2_FIXED_METHODS]
    ax.bar(x, y, color=[V2_METHOD_COLORS[method] for method in V2_FIXED_METHODS], width=0.72)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([V2_METHOD_LABELS[method] for method in V2_FIXED_METHODS], rotation=25, ha="right")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(y):
        if not np.isnan(value):
            ax.text(idx, value + (ylim[1] - ylim[0]) * 0.015, f"{value:.{decimals}f}", ha="center", va="bottom", fontsize=8)


def plot_oot_auc_v2(comparison: pd.DataFrame, output: Path) -> None:
    fig, axes = _panel_axes()
    for ax, (dataset, model, title) in zip(axes, PANEL_ORDER, strict=True):
        subset = comparison[(comparison["dataset"].eq(dataset)) & (comparison["model"].eq(model))]
        values = subset.set_index("selector")["auc"]
        _bar_panel(ax, values, title=title, ylabel="OOT AUC", ylim=(0.50, 0.82), decimals=3)
    fig.suptitle("Figure 1. CLIP-v2 OOT AUC comparison", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_auc_delta_v2(master: pd.DataFrame, comparison: pd.DataFrame, v1: pd.DataFrame, output: Path) -> None:
    fig, axes = _panel_axes()
    comparisons = [
        ("clip_v2", "clip", "CLIP-v2 - CLIP-v1"),
        ("clip_v2_then_mrmr", "clip_then_mrmr", "CLIP-v2->mRMR - CLIP-v1->mRMR"),
        ("clip_v2", "mrmr", "CLIP-v2 - mRMR"),
        ("clip_v2", "llm", "CLIP-v2 - LLM"),
        ("clip_v2_then_mrmr", "llm_then_mrmr", "CLIP-v2->mRMR - LLM->mRMR"),
    ]
    all_deltas: list[float] = []
    panel_rows: dict[tuple[str, str], list[dict[str, float | str]]] = {}
    for dataset, model, _title in PANEL_ORDER:
        rows = []
        for new_selector, baseline_selector, label in comparisons:
            new_auc = _auc_lookup(master, dataset=dataset, model=model, selector=new_selector)
            if baseline_selector in V1_TO_V2_SELECTOR:
                baseline_auc = _auc_lookup(v1, dataset=dataset, model=model, selector=baseline_selector)
            else:
                baseline_auc = _auc_lookup(comparison, dataset=dataset, model=model, selector=baseline_selector)
            if np.isfinite(new_auc) and np.isfinite(baseline_auc):
                delta = float(new_auc - baseline_auc)
                rows.append({"label": label, "delta": delta})
                all_deltas.append(delta)
        panel_rows[(dataset, model)] = rows
    max_abs = max([abs(value) for value in all_deltas] + [0.02])
    limit = min(0.16, max(0.04, max_abs * 1.25))
    for ax, (dataset, model, title) in zip(axes, PANEL_ORDER, strict=True):
        rows = panel_rows[(dataset, model)]
        y = np.arange(len(rows))
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.8)
        if rows:
            diff = np.array([float(row["delta"]) for row in rows])
            ax.scatter(diff, y, color="#2a6fbb", s=55)
            ax.set_yticks(y)
            ax.set_yticklabels([str(row["label"]) for row in rows])
            for idx, value in enumerate(diff):
                ax.text(value, idx + 0.12, f"{value:+.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Saved OOT AUC point difference")
        ax.set_xlim(-limit, limit)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Figure 2. CLIP-v2 OOT AUC deltas", fontsize=14)
    fig.text(0.5, 0.005, "No v2 paired-bootstrap confidence intervals are computed in current artifacts.", ha="center", fontsize=9)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _auc_lookup(frame: pd.DataFrame, *, dataset: str, model: str, selector: str) -> float:
    if frame.empty:
        return float("nan")
    dataset_col = "dataset" if "dataset" in frame.columns else "dataset_name"
    metric_col = "auc" if "auc" in frame.columns else "oot_auc"
    subset = frame[
        frame[dataset_col].astype(str).eq(dataset)
        & frame["model"].astype(str).eq(model)
        & frame["selector"].astype(str).eq(selector)
    ]
    if subset.empty:
        return float("nan")
    return float(pd.to_numeric(subset.iloc[0][metric_col], errors="coerce"))


def plot_score_psi_v2(comparison: pd.DataFrame, output: Path) -> None:
    fig, axes = _panel_axes()
    for ax, (dataset, model, title) in zip(axes, PANEL_ORDER, strict=True):
        subset = comparison[(comparison["dataset"].eq(dataset)) & (comparison["model"].eq(model))]
        values = subset.set_index("selector")["model_score_psi"]
        _bar_panel(ax, values, title=title, ylabel="DEV-to-OOT model score PSI", ylim=(0.0, 0.13), decimals=3)
        ax.axhline(0.10, color="#d55e00", linestyle="--", linewidth=1.0)
        ax.text(4.42, 0.102, "0.10", fontsize=8, color="#d55e00", va="bottom")
    fig.suptitle("Figure 3. CLIP-v2 DEV-to-OOT model score PSI comparison", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_semantic_redundancy_v2(semantic: pd.DataFrame, redundancy: pd.DataFrame, output: Path) -> None:
    merged = semantic.merge(
        redundancy[["dataset", "model", "selector", "repeated_base_family_share"]],
        on=["dataset", "model", "selector"],
        how="left",
    )
    fig, axes = _panel_axes()
    handles: dict[str, Any] = {}
    for ax, (dataset, model, title) in zip(axes, PANEL_ORDER, strict=True):
        subset = merged[(merged["dataset"].eq(dataset)) & (merged["model"].eq(model))]
        for _, row in subset.iterrows():
            method = str(row["selector"])
            point = ax.scatter(
                row["semantic_group_count"],
                row["repeated_base_family_share"],
                s=90,
                color=V2_METHOD_COLORS.get(method, "#777777"),
                label=V2_METHOD_LABELS.get(method, method),
                alpha=0.9,
            )
            handles[method] = point
            ax.text(row["semantic_group_count"] + 0.08, row["repeated_base_family_share"] + 0.003, V2_METHOD_LABELS.get(method, method), fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Semantic-group count")
        ax.set_ylabel("Repeated base-family share")
        ax.set_xlim(0, max(12, int(merged["semantic_group_count"].max()) + 2))
        ax.set_ylim(-0.02, max(0.12, float(merged["repeated_base_family_share"].max()) + 0.05))
        ax.grid(alpha=0.25)
    fig.suptitle("Figure 4. CLIP-v2 semantic coverage and redundancy", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_seed_robustness_v2(output: Path) -> None:
    seed_path = Path("results/clip_v2/training/seed_comparison.csv")
    manifest_path = Path("results/clip_v2/training/model_selection_manifest.json")
    if not seed_path.exists() or not manifest_path.exists():
        raise RuntimeError("CLIP-v2 seed robustness inputs are missing")
    frame = pd.read_csv(seed_path).sort_values("seed").copy()
    selected_seed = int(read_json(manifest_path)["selected_seed"])
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
    fig.suptitle(f"Figure 5. CLIP-v2 seed robustness (selected seed {selected_seed})", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_analysis() -> dict[str, str]:
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    evaluation = pd.read_csv(AGG_ROOT / "evaluation_summary.csv")
    selected = pd.read_csv(AGG_ROOT / "selected_features_long.csv")
    semantic = pd.read_csv(AGG_ROOT / "semantic_coverage_summary.csv")
    redundancy = pd.read_csv(AGG_ROOT / "redundancy_summary.csv")
    runtime = pd.read_csv(AGG_ROOT / "runtime_summary.csv")
    score_psi = pd.read_csv(AGG_ROOT / "score_psi_summary.csv")
    run_manifest = read_json(AGG_ROOT / "run_manifest.json")
    v1 = _load_v1_comparison()

    master = evaluation.copy()
    master["experiment_version"] = "clip_v2"
    master = master.sort_values(["dataset", "model", "selector"], kind="mergesort")
    v1_v2 = _v1_v2_comparison(master, v1)
    baselines = _baseline_comparison(master)
    external = master[master["dataset"].astype(str).eq("lendingclub_v2")].copy()
    metric_recomp = _metric_recomputation(run_manifest)
    claim = _claim_matrix(v1_v2, external)
    limitations = _limitations()
    source_audit = _source_audit()
    seed = _seed_robustness()
    semantic_map_data, semantic_map_plot = build_feature_semantic_map()
    plot_manifest = build_final_report_plots(
        master=master,
        baselines=baselines,
        semantic=semantic,
        redundancy=redundancy,
        v1=v1,
        semantic_map_plot=semantic_map_plot,
    )
    significance = pd.DataFrame(
        [
            {
                "comparison": "clip_v2_vs_clip_v1",
                "status": "not_computed_by_default",
                "reason": "paired bootstrap can be added after all v2 predictions exist and baseline prediction alignment is reviewed",
            }
        ]
    )
    outputs = {
        "source_artifact_audit": _write_csv(ANALYSIS_ROOT / "source_artifact_audit.csv", source_audit),
        "master_results_table": _write_csv(ANALYSIS_ROOT / "master_results_table.csv", master),
        "v1_vs_v2_comparison": _write_csv(ANALYSIS_ROOT / "v1_vs_v2_comparison.csv", v1_v2),
        "clip_v2_vs_baselines": _write_csv(ANALYSIS_ROOT / "clip_v2_vs_baselines.csv", baselines),
        "external_validation_comparison": _write_csv(ANALYSIS_ROOT / "external_validation_comparison.csv", external),
        "metric_recomputation": _write_csv(ANALYSIS_ROOT / "metric_recomputation.csv", metric_recomp),
        "score_drift_comparison": _write_csv(ANALYSIS_ROOT / "score_drift_comparison.csv", score_psi),
        "semantic_coverage_comparison": _write_csv(ANALYSIS_ROOT / "semantic_coverage_comparison.csv", semantic),
        "redundancy_comparison": _write_csv(ANALYSIS_ROOT / "redundancy_comparison.csv", redundancy),
        "seed_robustness": _write_csv(ANALYSIS_ROOT / "seed_robustness.csv", seed),
        "significance_comparison": _write_csv(ANALYSIS_ROOT / "significance_comparison.csv", significance),
        "claim_evidence_matrix": _write_csv(ANALYSIS_ROOT / "claim_evidence_matrix.csv", claim),
        "limitations_register": _write_csv(ANALYSIS_ROOT / "limitations_register.csv", limitations),
        "selected_feature_semantic_map_data": _write_csv(ANALYSIS_ROOT / "selected_feature_semantic_map_data.csv", semantic_map_data),
        "plot_manifest": _write_csv(ANALYSIS_ROOT / "plot_manifest.csv", plot_manifest),
        "analysis_summary": _write_json(
            ANALYSIS_ROOT / "analysis_summary.json",
            {
                "status": "complete",
                "scientific_question": "Did the compact target-free statistical vector improve feature screening relative to CLIP-v1?",
                "run_count": int(len(master)),
                "main_plot_count": int(plot_manifest["figure_role"].eq("main").sum()),
                "supplementary_plot_count": int(plot_manifest["figure_role"].eq("supplementary").sum()),
                "model_trained_by_analysis": False,
                "prediction_regenerated": False,
            },
        ),
    }
    outputs.update(_write_reports(master=master, v1_v2=v1_v2, limitations=limitations, source_audit=source_audit))
    return {key: str(value).replace("\\", "/") for key, value in outputs.items()}


def build_feature_semantic_map() -> tuple[pd.DataFrame, Path]:
    v1_selected = pd.read_csv("results/clip/final_evaluation/selected_features_long.csv")
    v2_selected_path = AGG_ROOT / "selected_features_long.csv"
    if not v2_selected_path.exists():
        raise RuntimeError("missing CLIP-v2 selected features for semantic map")
    v2_selected = pd.read_csv(v2_selected_path)
    rows = []
    for dataset in ["homecredit", "lendingclub_v2"]:
        embeddings = pd.read_parquet(f"results/clip/text_baseline/{dataset}_text_embeddings.parquet")
        feature_col = "feature_name"
        embedding_cols = sorted([col for col in embeddings.columns if re.fullmatch(r"embedding_\d+", str(col))])
        if not embedding_cols:
            raise RuntimeError(f"{dataset}: text embedding columns missing")
        coords = PCA(n_components=2, random_state=42).fit_transform(embeddings[embedding_cols].to_numpy(dtype=float))
        base = pd.DataFrame(
            {
                "dataset": dataset,
                "feature_name": embeddings[feature_col].astype(str),
                "pca_1": coords[:, 0],
                "pca_2": coords[:, 1],
                "semantic_group": embeddings.get("semantic_group", pd.Series(["unknown"] * len(embeddings))).fillna("unknown").astype(str),
            }
        )
        v1_clip = _selected_subset(v1_selected, dataset=dataset, selector="clip")
        v1_hybrid = _selected_subset(v1_selected, dataset=dataset, selector="clip_then_mrmr")
        v2_clip = _selected_subset(v2_selected, dataset=dataset, selector="clip_v2")
        v2_hybrid = _selected_subset(v2_selected, dataset=dataset, selector="clip_v2_then_mrmr")
        if v1_clip.empty or v2_clip.empty:
            raise RuntimeError(f"{dataset}: missing CLIP-v1 or CLIP-v2 CatBoost selections for semantic map")
        base["selected_by_clip_v1"] = base["feature_name"].isin(set(v1_clip["feature_name"].astype(str)))
        base["selected_by_clip_v2"] = base["feature_name"].isin(set(v2_clip["feature_name"].astype(str)))
        base["selected_by_clip_v1_then_mrmr"] = base["feature_name"].isin(set(v1_hybrid["feature_name"].astype(str)))
        base["selected_by_clip_v2_then_mrmr"] = base["feature_name"].isin(set(v2_hybrid["feature_name"].astype(str)))
        base["clip_v1_rank"] = base["feature_name"].map(_rank_map(v1_clip))
        base["clip_v2_rank"] = base["feature_name"].map(_rank_map(v2_clip))
        source_map = _source_map(v1_selected, v2_selected, dataset)
        base["source_table"] = base["feature_name"].map(source_map).fillna("")
        base["base_family"] = base["feature_name"].map(_base_family)
        rows.append(base)
    data = pd.concat(rows, ignore_index=True).sort_values(["dataset", "feature_name"], kind="mergesort")
    plot_path = ANALYSIS_ROOT / "plots" / "06_clip_v1_vs_v2_feature_semantic_map.png"
    _plot_semantic_map(data, plot_path)
    return data, plot_path


def _selected_subset(frame: pd.DataFrame, *, dataset: str, selector: str) -> pd.DataFrame:
    required = {"dataset", "model", "selector", "feature_name"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"selected feature table missing required columns: {sorted(missing)}")
    subset = frame[
        frame["dataset"].astype(str).eq(dataset)
        & frame["model"].astype(str).eq("catboost")
        & frame["selector"].astype(str).eq(selector)
    ].copy()
    if "final_selected" in subset.columns:
        subset = subset[subset["final_selected"].fillna(True).astype(bool)]
    if "final_rank" not in subset.columns:
        subset["final_rank"] = range(1, len(subset) + 1)
    return subset.sort_values(["final_rank", "feature_name"], kind="mergesort")


def _rank_map(frame: pd.DataFrame) -> dict[str, float]:
    rank_col = "clip_rank" if "clip_rank" in frame.columns else "final_rank"
    return dict(zip(frame["feature_name"].astype(str), pd.to_numeric(frame[rank_col], errors="coerce"), strict=False))


def _source_map(v1: pd.DataFrame, v2: pd.DataFrame, dataset: str) -> dict[str, str]:
    frames = []
    for frame in [v1, v2]:
        if "source_table_or_formula" in frame.columns:
            frames.append(frame[frame["dataset"].astype(str).eq(dataset)][["feature_name", "source_table_or_formula"]])
    if not frames:
        return {}
    merged = pd.concat(frames, ignore_index=True).dropna().drop_duplicates("feature_name")
    return dict(zip(merged["feature_name"].astype(str), merged["source_table_or_formula"].astype(str), strict=False))


def _base_family(feature: str) -> str:
    text = str(feature)
    suffixes = {"MEAN", "SUM", "MIN", "MAX", "VAR", "FLAG"}
    parts = text.split("_")
    if len(parts) > 2 and parts[-1] in suffixes:
        return "_".join(parts[:-1])
    return text


def _plot_semantic_map(data: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panels = [
        ("homecredit", "selected_by_clip_v1", "Home Credit: CLIP-v1"),
        ("homecredit", "selected_by_clip_v2", "Home Credit: CLIP-v2"),
        ("lendingclub_v2", "selected_by_clip_v1", "LendingClub v2: CLIP-v1"),
        ("lendingclub_v2", "selected_by_clip_v2", "LendingClub v2: CLIP-v2"),
    ]
    for ax, (dataset, flag, title) in zip(axes.ravel(), panels, strict=False):
        frame = data[data["dataset"].eq(dataset)].copy()
        selected = frame[frame[flag].astype(bool)]
        ax.scatter(frame["pca_1"], frame["pca_2"], s=10, c="#d0d5dd", alpha=0.45, linewidths=0)
        groups = selected["semantic_group"].astype(str).fillna("unknown")
        codes = pd.Categorical(groups).codes
        ax.scatter(selected["pca_1"], selected["pca_2"], s=45, c=codes, cmap="tab20", edgecolors="#111827", linewidths=0.4)
        v1_set = set(frame.loc[frame["selected_by_clip_v1"], "feature_name"])
        v2_set = set(frame.loc[frame["selected_by_clip_v2"], "feature_name"])
        jaccard = len(v1_set & v2_set) / len(v1_set | v2_set) if v1_set or v2_set else 0.0
        largest_share = groups.value_counts(normalize=True).iloc[0] if len(groups) else 0.0
        ax.set_title(title)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.text(
            0.02,
            0.98,
            f"selected={len(selected)}\nsemantic groups={groups.nunique()}\nJaccard v1/v2={jaccard:.2f}\nlargest group share={largest_share:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#d0d5dd"},
        )
    fig.suptitle("CLIP-v1 vs CLIP-v2 selected features in frozen text-embedding semantic space")
    fig.text(
        0.5,
        0.005,
        "PCA is visualization only; distances reflect frozen feature-text embeddings and do not prove predictive quality.",
        ha="center",
        fontsize=9,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_v1_comparison() -> pd.DataFrame:
    path = Path("results/clip/final_evaluation/evaluation_summary.csv")
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["experiment_version"] = "clip_v1"
    return frame


def _v1_v2_comparison(v2: pd.DataFrame, v1: pd.DataFrame) -> pd.DataFrame:
    if v1.empty:
        return pd.DataFrame({"status": ["missing_clip_v1_summary"]})
    v1_clip = v1[v1["selector"].astype(str).isin(["clip", "clip_then_mrmr"])].copy()
    v1_clip["selector_family"] = v1_clip["selector"].astype(str).str.replace("clip_then_mrmr", "clip_v2_then_mrmr").str.replace("clip", "clip_v2")
    v2_cmp = v2.copy()
    v2_cmp["selector_family"] = v2_cmp["selector"]
    metric_cols = [col for col in ["auc", "gini", "ks", "lift_at_10", "model_score_psi"] if col in v2.columns and col in v1.columns]
    merged = v2_cmp.merge(
        v1_clip[["dataset", "model", "selector_family", *metric_cols]],
        on=["dataset", "model", "selector_family"],
        how="left",
        suffixes=("_v2", "_v1"),
    )
    for metric in metric_cols:
        merged[f"{metric}_delta_v2_minus_v1"] = pd.to_numeric(merged[f"{metric}_v2"], errors="coerce") - pd.to_numeric(
            merged[f"{metric}_v1"], errors="coerce"
        )
    return merged


def _baseline_comparison(v2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in sorted(v2["dataset"].astype(str).unique()):
        baseline_path = Path("results") / dataset / "final_comparison_table.csv"
        if baseline_path.exists():
            base = pd.read_csv(baseline_path)
            base["source"] = "frozen_baseline"
            rows.append(base)
    out = v2.copy()
    out["source"] = "clip_v2_final_evaluation"
    rows.append(out)
    return pd.concat(rows, ignore_index=True, sort=False)


def _metric_recomputation(run_manifest: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in run_manifest:
        pred = pd.read_parquet(row["prediction_path"])
        rows.append(
            {
                "run_id": row["run_id"],
                "prediction_rows": int(len(pred)),
                "binary_target": set(pred["y_true"].astype(int).unique()).issubset({0, 1}),
                "probabilities_valid": bool(pd.to_numeric(pred["y_pred_proba"], errors="coerce").between(0, 1).all()),
                "prediction_hash": sha256_file(row["prediction_path"]),
            }
        )
    return pd.DataFrame(rows)


def _claim_matrix(v1_v2: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim": "CLIP-v2 improves over CLIP-v1",
                "evidence_table": "v1_vs_v2_comparison.csv",
                "status": "requires_metric_review",
            },
            {
                "claim": "LendingClub v2 remains external",
                "evidence_table": "external_validation_comparison.csv",
                "status": "pass" if not external.empty else "missing",
            },
            {
                "claim": "Analysis uses saved artifacts only",
                "evidence_table": "source_artifact_audit.csv",
                "status": "pass",
            },
        ]
    )


def _limitations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"limitation": "CLIP-v2 still depends on frozen feature text quality.", "severity": "medium"},
            {"limitation": "Descriptor set is compact and target-free, not exhaustive.", "severity": "medium"},
            {"limitation": "Success requires downstream OOT evidence, not representation loss alone.", "severity": "high"},
        ]
    )


def _source_audit() -> pd.DataFrame:
    paths = [
        AGG_ROOT / "evaluation_summary.csv",
        AGG_ROOT / "run_manifest.json",
        Path("results/clip_versions/v1/freeze_manifest.json"),
        Path("configs/clip_v2/analysis.yaml"),
    ]
    return pd.DataFrame(
        [{"path": str(path).replace("\\", "/"), "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else ""} for path in paths]
    )


def _seed_robustness() -> pd.DataFrame:
    path = Path("results/clip_v2/training/model_selection_manifest.json")
    if not path.exists():
        return pd.DataFrame([{"status": "missing_training_manifest"}])
    manifest = read_json(path)
    return pd.DataFrame(
        [
            {
                "selected_seed": manifest.get("selected_seed"),
                "seed_count": len(manifest.get("all_seed_results", [])),
                "selection_rule": manifest.get("selection_rule"),
                "lendingclub_v2_used_for_selection": manifest.get("lendingclub_v2_used_for_selection"),
            }
        ]
    )


def _write_reports(*, master: pd.DataFrame, v1_v2: pd.DataFrame, limitations: pd.DataFrame, source_audit: pd.DataFrame) -> dict[str, Path]:
    for path in REPORT_PATHS.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# CLIP-v2 Credit Risk Feature-Selection Report",
        "",
        "CLIP-v2 compares the frozen CLIP-v1 missingness-only statistical view with a 13-dimensional compact target-free statistical view.",
        "",
        f"Completed CLIP-v2 downstream runs: {len(master)}",
        "",
        "Representation evidence, downstream OOT evidence, and LendingClub v2 external validation must be interpreted separately.",
        "",
        "## Limitations",
        "",
        *[f"- {row.limitation}" for row in limitations.itertuples(index=False)],
    ]
    REPORT_PATHS["markdown"].write_text("\n".join(md) + "\n", encoding="utf-8")
    REPORT_PATHS["verdict"].write_text(
        "# CLIP-v2 Scientific Verdict\n\nVerdict requires review of `v1_vs_v2_comparison.csv` and external validation tables.\n",
        encoding="utf-8",
    )
    REPORT_PATHS["limitations"].write_text(_markdown_table(limitations) + "\n", encoding="utf-8")
    REPORT_PATHS["docx"].write_text("\n".join(md) + "\n", encoding="utf-8")
    REPORT_PATHS["pdf"].write_bytes(("%PDF-1.4\n% CLIP-v2 report placeholder generated from Markdown source.\n").encode("ascii"))
    write_json(
        REPORT_PATHS["manifest"],
        {
            "report_paths": {key: str(path).replace("\\", "/") for key, path in REPORT_PATHS.items()},
            "source_artifacts": source_audit.to_dict("records"),
            "model_trained_by_report_builder": False,
            "prediction_regenerated": False,
        },
    )
    return {f"report_{key}": path for key, path in REPORT_PATHS.items()}


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = frame.fillna("").astype(str).values.tolist()

    def clean(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    return "\n".join(lines)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_json(tmp, payload)
    tmp.replace(path)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
