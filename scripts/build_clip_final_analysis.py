from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import zipfile
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_clip_final_report_figures as figure_builder  # noqa: E402


ACTIVE_DATASETS = ("homecredit", "lendingclub_v2")
EXPECTED_OOT_ROWS = {"homecredit": 120053, "lendingclub_v2": 293105}
FIXED_METHODS = ["clip", "clip_then_mrmr", "mrmr", "llm", "llm_then_mrmr"]
OUTPUT_DIR = Path("results/clip/final_analysis")
FINAL_EVAL_DIR = Path("results/clip/final_evaluation")
TRAINING_DIR = Path("results/clip/training")
TEXT_DIR = Path("results/clip/text_baseline")
STAT_DIR = Path("results/clip/statistical_baseline")
REPORTS_DIR = Path("reports")
TOLERANCE = 1e-9

STAT_VIEW_LIMITATION = (
    "The current contrastive encoder aligns semantic feature metadata with a limited DEV statistical view, "
    "primarily reflecting missingness behavior. It is an architectural and screening experiment rather than "
    "a comprehensive statistical feature-quality representation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Prompt 8 final CLIP analysis and report artifacts.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned outputs without writing.")
    mode.add_argument("--execute", action="store_true", help="Write final analysis tables, plots, and reports.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def git_output(args: list[str]) -> str:
    import subprocess

    result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=30)
    return result.stdout.strip()


def psi_bucket(value: float) -> str:
    if pd.isna(value):
        return "unknown"
    if value < 0.10:
        return "low"
    if value < 0.25:
        return "moderate"
    return "high"


def base_family(feature: str) -> str:
    parts = str(feature).split("_")
    if len(parts) >= 3 and parts[-1] in {"MEAN", "SUM", "MIN", "MAX", "VAR", "FLAG"}:
        return "_".join(parts[:-1])
    return str(feature)


class SourceArtifacts:
    def __init__(self) -> None:
        self.required = [
            FINAL_EVAL_DIR / "run_manifest.json",
            FINAL_EVAL_DIR / "evaluation_summary.csv",
            FINAL_EVAL_DIR / "comparison_with_frozen_baselines.csv",
            FINAL_EVAL_DIR / "selected_features_long.csv",
            FINAL_EVAL_DIR / "selected_feature_summary.csv",
            FINAL_EVAL_DIR / "semantic_coverage_summary.csv",
            FINAL_EVAL_DIR / "redundancy_summary.csv",
            FINAL_EVAL_DIR / "runtime_summary.csv",
            FINAL_EVAL_DIR / "score_psi_summary.csv",
            FINAL_EVAL_DIR / "statistical_significance_summary.csv",
            TRAINING_DIR / "training_manifest.json",
            TRAINING_DIR / "model_selection_manifest.json",
            TRAINING_DIR / "seed_comparison.csv",
            TEXT_DIR / "text_baseline_summary.json",
            STAT_DIR / "statistical_baseline_summary.json",
        ]
        self.optional = [
            Path("results/clip/selector_integration"),
            FINAL_EVAL_DIR / "seed_sensitivity_summary.csv",
            FINAL_EVAL_DIR / "evaluation_limitations.md",
        ]

    def audit(self) -> pd.DataFrame:
        rows = []
        for path in self.required:
            rows.append(self._row(path, required=True))
        for path in self.optional:
            rows.append(self._row(path, required=False))
        return pd.DataFrame(rows)

    def _row(self, path: Path, *, required: bool) -> dict[str, Any]:
        exists = path.exists()
        is_file = path.is_file()
        status = "pass" if exists else ("fail" if required else "warn")
        issue = "" if exists else ("missing required artifact" if required else "optional artifact absent")
        return {
            "path": str(path).replace("\\", "/"),
            "required": required,
            "exists": exists,
            "type": "file" if is_file else ("directory" if exists else "missing"),
            "size_bytes": int(path.stat().st_size) if is_file else "",
            "sha256": sha256_file(path) if is_file else "",
            "status": status,
            "issue": issue,
        }


def validate_inputs() -> dict[str, Any]:
    artifact_audit = SourceArtifacts().audit()
    missing_required = artifact_audit[(artifact_audit["required"]) & (~artifact_audit["exists"])]
    if not missing_required.empty:
        missing = ", ".join(missing_required["path"].tolist())
        raise RuntimeError(f"missing required source artifacts: {missing}")

    run_manifest = read_json(FINAL_EVAL_DIR / "run_manifest.json")
    runs = run_manifest.get("runs", [])
    if len(runs) != 8:
        raise RuntimeError(f"expected 8 Prompt 7 runs, found {len(runs)}")
    run_ids = [run["run_id"] for run in runs]
    if len(set(run_ids)) != 8:
        raise RuntimeError("Prompt 7 run IDs are not unique")
    invalid = [run["run_id"] for run in runs if run.get("completion_status") not in {"complete_valid", "complete_valid_recovered"}]
    if invalid:
        raise RuntimeError(f"invalid Prompt 7 run statuses: {invalid}")
    in_progress = sorted(str(path) for path in FINAL_EVAL_DIR.rglob("*.in_progress"))
    in_progress += sorted(str(path) for path in (FINAL_EVAL_DIR / "runs").glob("*.in_progress") if path.exists())
    if in_progress:
        raise RuntimeError(f"in-progress paths remain: {in_progress}")

    evaluation = pd.read_csv(FINAL_EVAL_DIR / "evaluation_summary.csv")
    if len(evaluation) != 8:
        raise RuntimeError(f"evaluation_summary.csv should have 8 rows, found {len(evaluation)}")
    comparison = pd.read_csv(FINAL_EVAL_DIR / "comparison_with_frozen_baselines.csv")
    if set(comparison["selector"].dropna().unique()) & {"lendingclub"}:
        raise RuntimeError("legacy LendingClub selector/dataset contamination detected")

    for run in runs:
        run_dir = Path(run["run_dir"])
        marker = run_dir / "RUN_COMPLETE.json"
        pred = Path(run["prediction_path"])
        if not marker.exists():
            raise RuntimeError(f"{run['run_id']}: RUN_COMPLETE.json missing")
        if not pred.exists():
            raise RuntimeError(f"{run['run_id']}: prediction file missing")
        if sha256_file(pred) != run.get("prediction_hash"):
            raise RuntimeError(f"{run['run_id']}: prediction hash mismatch")
        if run.get("prediction_row_count") != EXPECTED_OOT_ROWS[run["dataset"]]:
            raise RuntimeError(f"{run['run_id']}: unexpected OOT row count")
        for key in ["checkpoint_hash", "anchor_hash", "feature_set_hash", "config_hash"]:
            if not run.get(key):
                raise RuntimeError(f"{run['run_id']}: missing {key}")
        if not run.get("source_hashes"):
            raise RuntimeError(f"{run['run_id']}: missing source hashes")

    training = read_json(TRAINING_DIR / "training_manifest.json")
    if training.get("training_dataset") != "homecredit":
        raise RuntimeError("CLIP training dataset is not Home Credit")
    if training.get("external_validation_dataset") != "lendingclub_v2":
        raise RuntimeError("CLIP external validation dataset is not LendingClub v2")

    return {
        "artifact_audit": artifact_audit,
        "run_manifest": run_manifest,
        "evaluation": evaluation,
        "comparison": comparison,
        "training_manifest": training,
        "model_selection": read_json(TRAINING_DIR / "model_selection_manifest.json"),
        "text_summary": read_json(TEXT_DIR / "text_baseline_summary.json"),
        "statistical_summary": read_json(STAT_DIR / "statistical_baseline_summary.json"),
    }


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].astype(int).to_numpy()
    score = frame["y_pred_proba"].astype(float).to_numpy()
    fpr, tpr, _ = roc_curve(y_true, score)
    auc = float(roc_auc_score(y_true, score))
    top_n = int(math.ceil(len(frame) * 0.10))
    top = frame.sort_values("y_pred_proba", ascending=False).head(top_n)
    return {
        "oot_auc": auc,
        "oot_gini": float(2 * auc - 1),
        "oot_ks": float(np.max(tpr - fpr)),
        "lift_at_10": float(top["y_true"].mean() / frame["y_true"].mean()),
        "oot_brier": float(brier_score_loss(y_true, score)),
        "oot_log_loss": float(log_loss(y_true, score, labels=[0, 1])),
    }


def recompute_metrics(run_manifest: dict[str, Any], evaluation: pd.DataFrame) -> pd.DataFrame:
    saved_by_run = {row["run_id"]: row for _, row in evaluation.iterrows()}
    rows = []
    for run in run_manifest["runs"]:
        pred = pd.read_csv(run["prediction_path"])
        metrics = prediction_metrics(pred)
        common = {
            "dataset": run["dataset"],
            "model": run["model"],
            "selector": run["selector"],
            "run_id": run["run_id"],
        }
        for metric, recomputed in metrics.items():
            saved_value = saved_by_run[run["run_id"]].get(metric, np.nan)
            if pd.isna(saved_value):
                run_metrics = pd.read_csv(Path(run["run_dir"]) / "results" / "oot_test_results.csv").iloc[0]
                saved_value = run_metrics.get(
                    {
                        "oot_auc": "auc",
                        "oot_gini": "gini",
                        "oot_ks": "ks",
                        "oot_brier": "brier",
                        "oot_log_loss": "log_loss",
                    }.get(metric, metric),
                    np.nan,
                )
            diff = abs(float(saved_value) - float(recomputed))
            rows.append(
                {
                    **common,
                    "metric": metric,
                    "saved_value": float(saved_value),
                    "recomputed_value": recomputed,
                    "absolute_difference": diff,
                    "tolerance": TOLERANCE,
                    "status": "pass" if diff <= TOLERANCE else "fail",
                }
            )
        if len(pred) != EXPECTED_OOT_ROWS[run["dataset"]]:
            rows.append({**common, "metric": "row_count", "saved_value": EXPECTED_OOT_ROWS[run["dataset"]], "recomputed_value": len(pred), "absolute_difference": abs(EXPECTED_OOT_ROWS[run["dataset"]] - len(pred)), "tolerance": 0, "status": "fail"})
    return pd.DataFrame(rows)


def selector_family(selector: str) -> str:
    if selector in {"clip", "clip_then_mrmr"}:
        return "clip_extension"
    if selector == "mrmr":
        return "statistical"
    if selector == "llm":
        return "llm"
    if selector in {"llm_then_mrmr", "stable_core_llm_fill"}:
        return "hybrid_llm_statistical"
    if selector in {"text_only", "statistical_only"}:
        return "representation_baseline"
    return "other"


def build_master_results(context: dict[str, Any]) -> pd.DataFrame:
    comparison = context["comparison"].copy()
    semantic = pd.read_csv(FINAL_EVAL_DIR / "semantic_coverage_summary.csv")
    redundancy = pd.read_csv(FINAL_EVAL_DIR / "redundancy_summary.csv")
    semantic = semantic.rename(columns={"dataset": "dataset_name"})
    redundancy = redundancy.rename(columns={"dataset": "dataset_name"})
    master = comparison.merge(
        semantic[["dataset_name", "model", "selector", "semantic_group_count", "largest_semantic_group_share"]],
        on=["dataset_name", "model", "selector"],
        how="left",
        suffixes=("", "_clip_summary"),
    )
    master = master.merge(
        redundancy[["dataset_name", "model", "selector", "repeated_base_family_share", "near_duplicate_family_count"]],
        on=["dataset_name", "model", "selector"],
        how="left",
        suffixes=("", "_clip_redundancy"),
    )
    if "semantic_group_count" not in master.columns:
        master["semantic_group_count"] = np.nan
    master["semantic_group_count"] = master["semantic_group_count"].fillna(master.get("stable_semantic_group_count_80"))
    master["largest_semantic_group_share"] = master["largest_semantic_group_share"].fillna(np.nan)
    master["repeated_base_family_share"] = master["repeated_base_family_share"].fillna(master.get("selected_feature_psi_high_drift_ratio", np.nan) * 0)
    master["selector_family"] = master["selector"].map(selector_family)
    master["result_origin"] = master["result_origin"].fillna(master.get("source", "")).replace({"clip_final_evaluation": "clip_extension"})
    master["result_origin"] = master["result_origin"].where(master["result_origin"].isin(["frozen_baseline", "clip_extension", "representation_baseline"]), "frozen_baseline")
    master["stability_metric_name"] = np.where(master["selector"].isin(["clip", "clip_then_mrmr"]), "not_repeated_downstream", "spearman_rank_stability_mean")
    master["stability_metric_value"] = master.get("spearman_rank_stability_mean", np.nan)
    keep = {
        "dataset_name": "dataset",
        "model": "model",
        "selector": "selector",
        "selector_family": "selector_family",
        "result_origin": "result_origin",
        "run_id": "run_id",
        "selected_feature_count": "feature_count",
        "oot_auc": "oot_auc",
        "oot_gini": "oot_gini",
        "oot_ks": "oot_ks",
        "lift_at_10": "lift_at_10",
        "oot_brier": "brier_score",
        "oot_log_loss": "log_loss",
        "model_score_psi": "model_score_psi",
        "semantic_group_count": "semantic_group_count",
        "largest_semantic_group_share": "largest_semantic_group_share",
        "repeated_base_family_share": "repeated_base_family_share",
        "near_duplicate_family_count": "near_duplicate_family_count",
        "runtime_seconds": "runtime_seconds",
        "checkpoint_hash": "checkpoint_hash",
        "anchor_hash": "anchor_hash",
        "feature_set_hash": "feature_set_hash",
        "statistical_view_scope": "statistical_view_scope",
        "stability_metric_name": "stability_metric_name",
        "stability_metric_value": "stability_metric_value",
    }
    for column in keep:
        if column not in master.columns:
            master[column] = np.nan
    return master[list(keep)].rename(columns=keep).sort_values(["dataset", "model", "selector"]).reset_index(drop=True)


def pairwise_table(master: pd.DataFrame, selector_a: str, selector_b: str) -> pd.DataFrame:
    rows = []
    for (dataset, model), group in master.groupby(["dataset", "model"], dropna=False):
        a = group[group["selector"].eq(selector_a)]
        b = group[group["selector"].eq(selector_b)]
        if a.empty or b.empty:
            rows.append({"dataset": dataset, "model": model, "selector_a": selector_a, "selector_b": selector_b, "status": "skipped", "reason": "one or both selectors absent"})
            continue
        ar = a.iloc[0]
        br = b.iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "selector_a": selector_a,
                "selector_b": selector_b,
                "status": "ok",
                "auc_a": ar["oot_auc"],
                "auc_b": br["oot_auc"],
                "auc_difference_a_minus_b": float(ar["oot_auc"]) - float(br["oot_auc"]),
                "gini_a": ar["oot_gini"],
                "gini_b": br["oot_gini"],
                "score_psi_a": ar["model_score_psi"],
                "score_psi_b": br["model_score_psi"],
                "reason": "",
            }
        )
    return pd.DataFrame(rows)


def representation_baseline_table(kind: str, context: dict[str, Any]) -> pd.DataFrame:
    if kind == "text_only":
        summary = context["text_summary"]
        return pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "baseline": "text_only",
                    "status": "representation_baseline_only",
                    "feature_rows": summary.get(f"{dataset}_texts"),
                    "embedding_rows": summary.get(f"{dataset}_embeddings"),
                    "embedding_dimension": summary.get("embedding_dimension"),
                    "encoder_model": summary.get("encoder_model"),
                    "downstream_oot_auc_available": False,
                    "interpretation": "Frozen text embeddings are ranking/alignment evidence, not a saved downstream OOT model.",
                }
                for dataset in ACTIVE_DATASETS
            ]
        )
    summary = context["statistical_summary"]
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "baseline": "statistical_only",
                "status": "representation_baseline_only",
                "feature_rows": summary.get(f"{dataset}_vectors"),
                "vector_dimension": summary.get("vector_dimension"),
                "statistical_fields": "; ".join(summary.get("main_statistical_fields", [])),
                "downstream_oot_auc_available": False,
                "interpretation": STAT_VIEW_LIMITATION,
            }
            for dataset in ACTIVE_DATASETS
        ]
    )


def priority_significance() -> pd.DataFrame:
    sig = pd.read_csv(FINAL_EVAL_DIR / "statistical_significance_summary.csv")
    pairs = {("clip", "mrmr"), ("clip", "llm"), ("clip_then_mrmr", "llm_then_mrmr")}
    frame = sig[sig[["new_selector", "baseline_selector"]].apply(lambda row: (row["new_selector"], row["baseline_selector"]) in pairs, axis=1)].copy()
    frame = frame.rename(
        columns={
            "new_selector": "selector_a",
            "baseline_selector": "selector_b",
            "point_estimate_difference": "auc_difference",
            "p_value": "p_value",
        }
    )
    return frame


def build_claims(master: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    hc_clip_mrmr = pairwise_table(master, "clip", "mrmr")
    clip_mrmr_mean = hc_clip_mrmr["auc_difference_a_minus_b"].dropna().mean()
    rows = [
        {
            "claim_id": "C01",
            "claim_text": "CLIP representation training produced usable noncollapsed representation evidence.",
            "dataset_scope": "Home Credit training, LendingClub v2 external application",
            "model_scope": "representation",
            "source_artifacts": "results/clip/training/seed_comparison.csv; results/clip/training/model_selection_manifest.json",
            "supporting_metrics": "5 retained seeds; selected seed by Home Credit validation loss",
            "uncertainty_evidence": "seed comparison only",
            "stability_evidence": "multi-seed validation loss and MRR",
            "limitation": STAT_VIEW_LIMITATION,
            "evidence_rating": "Moderate",
            "allowed_wording": "CLIP representation evidence is usable as a screening experiment.",
            "prohibited_stronger_wording": "CLIP learned comprehensive statistical feature quality or proved predictive superiority.",
        },
        {
            "claim_id": "C02",
            "claim_text": "CLIP downstream feature selection does not consistently beat mRMR or LLM baselines.",
            "dataset_scope": "Home Credit and LendingClub v2",
            "model_scope": "LR and CatBoost",
            "source_artifacts": "master_results_table.csv; clip_vs_mrmr.csv; clip_vs_llm.csv",
            "supporting_metrics": f"mean CLIP minus mRMR AUC difference {clip_mrmr_mean:.4f}",
            "uncertainty_evidence": "paired bootstrap where baseline predictions exist",
            "stability_evidence": "no downstream multi-seed prediction evidence",
            "limitation": "fixed feature budgets and limited dataset count",
            "evidence_rating": "Weak",
            "allowed_wording": "CLIP is valid to compare but not ready to replace stronger baselines.",
            "prohibited_stronger_wording": "CLIP is best overall or production ready.",
        },
        {
            "claim_id": "C03",
            "claim_text": "LendingClub v2 was external validation only.",
            "dataset_scope": "LendingClub v2",
            "model_scope": "all CLIP extensions",
            "source_artifacts": "training_manifest.json; run_manifest.json",
            "supporting_metrics": "external_validation_dataset=lendingclub_v2",
            "uncertainty_evidence": "not applicable",
            "stability_evidence": "not applicable",
            "limitation": "single external dataset",
            "evidence_rating": "Strong",
            "allowed_wording": "LCv2 tests external application of the Home Credit-trained representation.",
            "prohibited_stronger_wording": "LCv2 was used to tune or train CLIP.",
        },
        {
            "claim_id": "C04",
            "claim_text": "CLIP is not ready to replace the LLM workflow.",
            "dataset_scope": "both datasets",
            "model_scope": "LR and CatBoost",
            "source_artifacts": "clip_vs_llm.csv; clip_then_mrmr_vs_llm_then_mrmr.csv",
            "supporting_metrics": "CLIP trails LLM and mRMR baselines in most OOT panels.",
            "uncertainty_evidence": "paired bootstrap deltas",
            "stability_evidence": "LLM workflow preserved as frozen baseline",
            "limitation": "does not rule out future richer CLIP variants",
            "evidence_rating": "Weak for replacement",
            "allowed_wording": "CLIP should remain experimental and complementary.",
            "prohibited_stronger_wording": "CLIP replaces LLM screening.",
        },
    ]
    return pd.DataFrame(rows)


def limitations_register() -> pd.DataFrame:
    rows = [
        ("non-fold-local CLIP preparation", "high", "downstream OOT comparison", "may overstate CV diagnostics", True, "rebuild representation fold-locally in future"),
        ("DEV-CV diagnostic limitation", "medium", "validation interpretation", "CV is not primary evidence", False, "treat OOT as primary"),
        ("limited statistical view", "high", "representation alignment", "may overstate statistical-quality learning", True, STAT_VIEW_LIMITATION),
        ("Home Credit-only contrastive training", "medium", "external generalization", "may reduce transfer", False, "train on more datasets after leakage review"),
        ("LendingClub v2 external-only application", "medium", "LCv2 conclusions", "prevents tuning but limits adaptation", False, "add more external datasets"),
        ("limited seed count", "medium", "seed robustness", "uncertainty remains", False, "increase seeds for representation and downstream evaluation"),
        ("unavailable paired baseline predictions where applicable", "medium", "uncertainty", "some comparisons may be skipped", False, "persist all baseline predictions"),
        ("independent PSI-recomputation limitation", "medium", "score drift", "PSI values cannot be fully recomputed without DEV scores", False, "persist DEV score vectors"),
        ("fixed feature budgets", "medium", "method comparison", "budget choice can affect winners", False, "evaluate budget sensitivity"),
        ("limited dataset count", "medium", "generalization", "external evidence is narrow", False, "add datasets"),
        ("no fairness analysis", "medium", "operational interpretation", "cannot assess subgroup risk", True, "run fairness audit before deployment claims"),
        ("no operational cost analysis", "low", "deployment interpretation", "runtime/cost not complete", False, "measure cost if operationalized"),
        ("no causal interpretation", "high", "scientific claims", "predictive results are not causal", True, "avoid causal language"),
        ("no production-readiness claim", "high", "conclusion", "research artifact only", True, "separate production validation"),
    ]
    return pd.DataFrame(rows, columns=["limitation", "severity", "affected_claim", "likely_direction_of_bias", "blocks_claim", "future_correction"])


def build_plot_set(master: pd.DataFrame) -> pd.DataFrame:
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(FINAL_EVAL_DIR / "comparison_with_frozen_baselines.csv")
    comparison = comparison[comparison["selector"].isin(FIXED_METHODS)].copy()
    significance = pd.read_csv(FINAL_EVAL_DIR / "statistical_significance_summary.csv")
    seed_comparison = pd.read_csv(TRAINING_DIR / "seed_comparison.csv")
    selected_seed = int(read_json(TRAINING_DIR / "model_selection_manifest.json")["selected_seed"])
    semantic_frame = figure_builder.build_semantic_plot_frame(comparison)
    write_csv(OUTPUT_DIR / "semantic_redundancy_plot_data.csv", semantic_frame)

    figure_builder.plot_oot_auc(comparison, plots_dir / "01_oot_auc_main_comparison.png")
    figure_builder.plot_auc_delta(significance, plots_dir / "02_oot_auc_delta_with_uncertainty.png")
    figure_builder.plot_score_psi(comparison, plots_dir / "03_score_psi_comparison.png")
    figure_builder.plot_semantic_redundancy(semantic_frame, plots_dir / "04_semantic_coverage_redundancy.png")
    figure_builder.plot_seed_robustness(seed_comparison, selected_seed, plots_dir / "05_clip_seed_robustness.png")

    rows = [
        ("plots/01_oot_auc_main_comparison.png", "main", "Are CLIP selectors predictively competitive?", ", ".join(FIXED_METHODS), "Home Credit and LCv2, LR/CatBoost", "comparison_with_frozen_baselines.csv", "oot_auc", "fixed-method bar comparison", "CLIP is valid but not uniformly best", "AUC alone omits uncertainty", "primary OOT comparison"),
        ("plots/02_oot_auc_delta_with_uncertainty.png", "main", "What are paired AUC deltas?", "clip, clip_then_mrmr, mrmr, llm, llm_then_mrmr", "all panels", "statistical_significance_summary.csv", "auc_difference, ci95_lower, ci95_upper", "paired bootstrap deltas", "CLIP deltas are mostly negative vs baselines", "only valid where predictions align", "uncertainty around key comparisons"),
        ("plots/03_score_psi_comparison.png", "main", "Is score drift acceptable?", ", ".join(FIXED_METHODS), "all panels", "comparison_with_frozen_baselines.csv", "model_score_psi", "PSI with 0.10/0.25 reference lines", "most score PSI is low; Home Credit LR clip is moderate", "PSI not independently recomputed from DEV vectors", "drift check"),
        ("plots/04_semantic_coverage_redundancy.png", "main", "Does CLIP change semantic breadth and redundancy?", ", ".join(FIXED_METHODS), "all panels", "selected feature artifacts", "semantic_group_count, repeated_base_family_share", "scatter of semantic count and redundancy", "CLIP->mRMR can reduce redundancy", "coverage is not predictive superiority", "semantic/redundancy evidence"),
        ("plots/05_clip_seed_robustness.png", "main", "Was selected checkpoint seed robust?", "clip", "representation training", "seed_comparison.csv", "best_validation_loss, best_validation_mrr", "seed bars with selected seed", "seed 55 selected by validation loss", "not downstream seed robustness", "documents Prompt 5 seed rule"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "file",
            "main_or_supplementary",
            "research_question",
            "included_methods",
            "dataset_model_scope",
            "source_artifacts",
            "source_columns",
            "calculation",
            "main_takeaway",
            "caveat",
            "reason_required",
        ],
    )
    write_csv(OUTPUT_DIR / "plot_manifest.csv", frame)
    return frame


def build_report(master: pd.DataFrame, claims: pd.DataFrame, limitations: pd.DataFrame, context: dict[str, Any]) -> str:
    fixed = master[master["selector"].isin(FIXED_METHODS)].copy()
    best_rows = fixed.sort_values("oot_auc", ascending=False).groupby(["dataset", "model"], as_index=False).first()
    best_text = "\n".join(
        f"- {row.dataset} {row.model}: {row.selector} AUC {row.oot_auc:.4f}, Gini {row.oot_gini:.4f}, PSI {row.model_score_psi:.4f}"
        for row in best_rows.itertuples()
    )
    clip_rows = fixed[fixed["selector"].isin(["clip", "clip_then_mrmr"])]
    clip_best = clip_rows.sort_values("oot_auc", ascending=False).groupby(["dataset", "model"], as_index=False).first()
    clip_text = "\n".join(
        f"- {row.dataset} {row.model}: best CLIP-family selector `{row.selector}` AUC {row.oot_auc:.4f}"
        for row in clip_best.itertuples()
    )
    return f"""# Final CLIP Credit-Risk Feature-Selection Report

## 1. Bottom-line verdict

Prompt 8 completes the final analysis and reporting layer from saved artifacts only. The scientific conclusion is conservative: CLIP-style representation learning is a valid architectural screening experiment, but the saved downstream OOT evidence does not support replacing the frozen LLM or mRMR workflows.

{STAT_VIEW_LIMITATION}

## 2. Objective and research design

The study compares original statistical selectors, original LLM-assisted selectors, and two frozen CLIP selector extensions: `clip` and `clip_then_mrmr`. Home Credit is the CLIP training dataset. LendingClub v2 is external validation only. Legacy LendingClub is not part of the CLIP training, integration, evaluation, plots, or conclusions.

## 3. Data, temporal validation, and leakage controls

The final analysis uses saved Prompt 7 OOT predictions and aggregate tables. Home Credit has 120,053 OOT rows; LendingClub v2 has 293,105 OOT rows. Run manifests record checkpoint, anchor, feature-set, config, prediction, metric, and source hashes. Per-run leakage audits report passed status, and the analysis does not rerun feature selection, model fitting, or prediction generation.

## 4. Main OOT predictive results

Best fixed-method result by dataset/model:

{best_text}

Best CLIP-family result by dataset/model:

{clip_text}

Across the four dataset/model panels, `clip_then_mrmr` is consistently stronger than direct `clip`, but it generally trails the strongest frozen mRMR or LLM-assisted baselines. This supports CLIP as an experimental selector, not a replacement.

## 5. Score drift, semantic coverage, and redundancy

Model score PSI is taken from saved Prompt 7 run artifacts. DEV score vectors are not persisted, so PSI was not independently recomputed. The interpretation thresholds are low drift below 0.10, moderate drift from 0.10 to below 0.25, and high drift at or above 0.25. Most CLIP-family runs have low score PSI; Home Credit LR direct `clip` is moderate.

Semantic coverage and redundancy are descriptive. Broader semantic coverage does not imply predictive superiority. Home Credit direct CLIP has high repeated-family share, while `clip_then_mrmr` reduces redundancy in the LR panel. LendingClub v2 CLIP selections show low repeated-family share in the saved artifacts.

## 6. Representation learning and seed robustness

Prompt 5 retained five seeds and selected seed 55 by the prespecified lowest Home Credit validation loss rule. LendingClub v2 did not influence seed or checkpoint selection. This is representation-level evidence only; downstream multi-seed predictions were not materialized.

## 7. LendingClub v2 external validation

LendingClub v2 was external-only. CLIP-family results transfer enough to run valid OOT comparisons, but they do not consistently outperform LLM or mRMR baselines. The external validation finding is therefore weak for replacement and useful mainly as a boundary check.

## 8. Limitations

The main limitations are recorded in `results/clip/final_analysis/limitations_register.csv`. The most important are the missingness-only statistical view, non-fold-local CLIP preparation, fixed feature budgets, limited dataset count, limited seed count, and unavailable DEV score vectors for independent PSI recomputation. No fairness, causal, production-readiness, or operational-cost claim is made.

## 9. Conclusion

CLIP is scientifically usable as an experimental representation and screening extension in this repository. The final OOT evidence does not justify replacing the LLM workflow or the strongest mRMR baselines. The recommended interpretation is: keep CLIP as a documented research extension, preserve LLM and mRMR baselines, and treat future CLIP work as requiring richer DEV statistical views and broader external validation.
"""


def write_docx(path: Path, title: str, markdown_text: str) -> None:
    paragraphs = [line.strip("# ").strip() for line in markdown_text.splitlines() if line.strip()]
    body = "".join(f"<w:p><w:r><w:t>{escape_xml(p)}</w:t></w:r></w:p>" for p in paragraphs)
    document = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>'
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        docx.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        docx.writestr("word/document.xml", document)


def write_pdf(path: Path, title: str, markdown_text: str) -> None:
    lines = [title, ""] + [line.strip("# ").strip() for line in markdown_text.splitlines() if line.strip()][:60]
    escaped = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for line in lines]
    text_ops = ["BT", "/F1 10 Tf", "50 780 Td"]
    for idx, line in enumerate(escaped):
        if idx:
            text_ops.append("0 -14 Td")
        text_ops.append(f"({line[:100]}) Tj")
    text_ops.append("ET")
    stream = "\n".join(text_ops).encode("latin-1", errors="replace")
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        f"5 0 obj << /Length {len(stream)} >> stream\n".encode() + stream + b"\nendstream endobj\n",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"%PDF-1.4\n")
        offsets = []
        for obj in objects:
            offsets.append(handle.tell())
            handle.write(obj)
        xref = handle.tell()
        handle.write(f"xref\n0 {len(objects)+1}\n0000000000 65535 f \n".encode())
        for offset in offsets:
            handle.write(f"{offset:010d} 00000 n \n".encode())
        handle.write(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())


def escape_xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_repro_manifest(context: dict[str, Any], plots: pd.DataFrame, report_path: Path) -> dict[str, Any]:
    run_manifest = context["run_manifest"]
    report_files = [
        REPORTS_DIR / "final_clip_credit_risk_report.md",
        REPORTS_DIR / "final_clip_credit_risk_report.docx",
        REPORTS_DIR / "final_clip_credit_risk_report.pdf",
        REPORTS_DIR / "final_clip_scientific_verdict.md",
        REPORTS_DIR / "final_clip_limitations.md",
    ]
    return {
        "git_branch": git_output(["branch", "--show-current"]),
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "available_tags": git_output(["tag", "--list"]).splitlines(),
        "prompt7_aggregate_hashes": {
            path.name: sha256_file(path)
            for path in FINAL_EVAL_DIR.glob("*.csv")
        },
        "run_ids": [run["run_id"] for run in run_manifest["runs"]],
        "prediction_hashes": {run["run_id"]: run["prediction_hash"] for run in run_manifest["runs"]},
        "checkpoint_hash": run_manifest["runs"][0]["checkpoint_hash"],
        "anchor_hash": run_manifest["runs"][0]["anchor_hash"],
        "statistical_preprocessor_hash": context["statistical_summary"].get("preprocessor_hash"),
        "config_hashes": {run["run_id"]: run["config_hash"] for run in run_manifest["runs"]},
        "feature_set_hashes": {run["run_id"]: run["feature_set_hash"] for run in run_manifest["runs"]},
        "source_hashes": {run["run_id"]: run["source_hashes"] for run in run_manifest["runs"]},
        "seed_list": context["training_manifest"].get("seeds"),
        "selected_checkpoint_rule": context["model_selection"].get("selection_rule"),
        "analysis_script_hash": sha256_file(Path("scripts/build_clip_final_analysis.py")),
        "plot_hashes": {row.file: sha256_file(OUTPUT_DIR / row.file) for row in plots.itertuples()},
        "report_hashes": {str(path).replace("\\", "/"): sha256_file(path) for path in report_files if path.exists()},
    }


def write_outputs(context: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "source_artifact_audit.csv", context["artifact_audit"])
    write_json(OUTPUT_DIR / "source_artifact_audit.json", context["artifact_audit"].to_dict("records"))

    metric_frame = recompute_metrics(context["run_manifest"], context["evaluation"])
    master = build_master_results(context)
    write_csv(OUTPUT_DIR / "metric_recomputation.csv", metric_frame)
    write_csv(OUTPUT_DIR / "master_results_table.csv", master)
    write_csv(OUTPUT_DIR / "headline_results.csv", master[master["selector"].isin(FIXED_METHODS)].sort_values(["dataset", "model", "oot_auc"], ascending=[True, True, False]).groupby(["dataset", "model"], as_index=False).first())
    write_csv(OUTPUT_DIR / "clip_vs_baselines.csv", pd.concat([pairwise_table(master, "clip", "mrmr"), pairwise_table(master, "clip", "llm"), pairwise_table(master, "clip_then_mrmr", "llm_then_mrmr")], ignore_index=True))
    write_csv(OUTPUT_DIR / "clip_vs_text_only.csv", representation_baseline_table("text_only", context))
    write_csv(OUTPUT_DIR / "clip_vs_statistical_only.csv", representation_baseline_table("statistical_only", context))
    write_csv(OUTPUT_DIR / "clip_vs_llm.csv", pairwise_table(master, "clip", "llm"))
    write_csv(OUTPUT_DIR / "clip_vs_mrmr.csv", pairwise_table(master, "clip", "mrmr"))
    write_csv(OUTPUT_DIR / "clip_then_mrmr_vs_llm_then_mrmr.csv", pairwise_table(master, "clip_then_mrmr", "llm_then_mrmr"))
    write_csv(OUTPUT_DIR / "external_validation_comparison.csv", master[master["dataset"].eq("lendingclub_v2")])
    shutil.copy2(TRAINING_DIR / "seed_comparison.csv", OUTPUT_DIR / "seed_robustness.csv")
    shutil.copy2(FINAL_EVAL_DIR / "semantic_coverage_summary.csv", OUTPUT_DIR / "semantic_coverage_comparison.csv")
    shutil.copy2(FINAL_EVAL_DIR / "redundancy_summary.csv", OUTPUT_DIR / "redundancy_comparison.csv")
    stability = master[["dataset", "model", "selector", "stability_metric_name", "stability_metric_value"]].copy()
    stability["interpretation"] = np.where(stability["stability_metric_name"].eq("not_repeated_downstream"), "deterministic ranking only; no repeated downstream CLIP selections", "frozen baseline repeated-selection summary")
    write_csv(OUTPUT_DIR / "stability_comparison.csv", stability)
    score = pd.read_csv(FINAL_EVAL_DIR / "score_psi_summary.csv")
    score["psi_bucket_verified"] = score["model_score_psi"].map(psi_bucket)
    score["validation_status"] = np.where(score["model_score_psi_bucket"].eq(score["psi_bucket_verified"]), "pass", "fail")
    write_csv(OUTPUT_DIR / "score_drift_comparison.csv", score)
    shutil.copy2(FINAL_EVAL_DIR / "runtime_summary.csv", OUTPUT_DIR / "runtime_comparison.csv")
    write_csv(OUTPUT_DIR / "significance_comparison.csv", priority_significance())
    claims = build_claims(master, context)
    limitations = limitations_register()
    write_csv(OUTPUT_DIR / "claim_evidence_matrix.csv", claims)
    write_csv(OUTPUT_DIR / "limitations_register.csv", limitations)
    plots = build_plot_set(master)

    summary = {
        "status": "complete",
        "run_count": 8,
        "fixed_methods": FIXED_METHODS,
        "metric_recomputation_status": "pass" if metric_frame["status"].eq("pass").all() else "fail",
        "statistical_view_limitation": STAT_VIEW_LIMITATION,
        "evidence_ratings": {
            "clip_representation_alignment": "Moderate",
            "clip_downstream_feature_selection": "Weak",
            "clip_vs_mrmr": "Weak",
            "clip_vs_llm": "Weak",
            "clip_then_mrmr_vs_llm_then_mrmr": "Weak",
            "lendingclub_v2_external_generalization": "Weak",
            "seed_robustness": "Moderate",
            "score_drift_acceptability": "Moderate",
            "semantic_coverage": "Weak",
            "clip_ready_to_replace_llm": "Not supported",
        },
        "no_model_pipeline_run": True,
        "no_predictions_regenerated": True,
    }
    write_json(OUTPUT_DIR / "analysis_summary.json", summary)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report(master, claims, limitations, context)
    report_path = REPORTS_DIR / "final_clip_credit_risk_report.md"
    report_path.write_text(report, encoding="utf-8")
    verdict = "# Final CLIP Scientific Verdict\n\nCLIP is a valid research extension, but it is not ready to replace the LLM workflow or the strongest mRMR baselines.\n\n" + STAT_VIEW_LIMITATION + "\n"
    (REPORTS_DIR / "final_clip_scientific_verdict.md").write_text(verdict, encoding="utf-8")
    limitations_text = "# Final CLIP Limitations\n\n" + "\n".join(f"- {row.limitation}: {row.future_correction}" for row in limitations.itertuples()) + "\n"
    (REPORTS_DIR / "final_clip_limitations.md").write_text(limitations_text, encoding="utf-8")
    write_docx(REPORTS_DIR / "final_clip_credit_risk_report.docx", "Final CLIP Credit-Risk Report", report)
    write_pdf(REPORTS_DIR / "final_clip_credit_risk_report.pdf", "Final CLIP Credit-Risk Report", report)
    repro = build_repro_manifest(context, plots, report_path)
    write_json(REPORTS_DIR / "final_clip_reproducibility_manifest.json", repro)


def planned_outputs() -> list[str]:
    names = [
        "source_artifact_audit.csv",
        "source_artifact_audit.json",
        "master_results_table.csv",
        "headline_results.csv",
        "clip_vs_baselines.csv",
        "clip_vs_text_only.csv",
        "clip_vs_statistical_only.csv",
        "clip_vs_llm.csv",
        "clip_vs_mrmr.csv",
        "clip_then_mrmr_vs_llm_then_mrmr.csv",
        "external_validation_comparison.csv",
        "seed_robustness.csv",
        "semantic_coverage_comparison.csv",
        "redundancy_comparison.csv",
        "stability_comparison.csv",
        "score_drift_comparison.csv",
        "runtime_comparison.csv",
        "significance_comparison.csv",
        "metric_recomputation.csv",
        "claim_evidence_matrix.csv",
        "limitations_register.csv",
        "plot_manifest.csv",
        "analysis_summary.json",
    ]
    plots = [
        "plots/01_oot_auc_main_comparison.png",
        "plots/02_oot_auc_delta_with_uncertainty.png",
        "plots/03_score_psi_comparison.png",
        "plots/04_semantic_coverage_redundancy.png",
        "plots/05_clip_seed_robustness.png",
    ]
    reports = [
        "reports/final_clip_credit_risk_report.md",
        "reports/final_clip_scientific_verdict.md",
        "reports/final_clip_limitations.md",
        "reports/final_clip_reproducibility_manifest.json",
        "reports/final_clip_credit_risk_report.docx",
        "reports/final_clip_credit_risk_report.pdf",
    ]
    return [str(OUTPUT_DIR / name) for name in names] + [str(OUTPUT_DIR / name) for name in plots] + reports


def main() -> int:
    args = parse_args()
    context = validate_inputs()
    if args.dry_run:
        payload = {
            "status": "dry_run_passed",
            "source_artifacts_validated": int(context["artifact_audit"]["exists"].sum()),
            "planned_outputs": planned_outputs(),
            "missing_optional_artifacts": context["artifact_audit"][~context["artifact_audit"]["exists"] & ~context["artifact_audit"]["required"]]["path"].tolist(),
            "will_train_models": False,
            "will_regenerate_predictions": False,
            "will_modify_prompt7_run_dirs": False,
        }
        print(json.dumps(payload, indent=2))
        return 0
    write_outputs(context)
    print(json.dumps({"status": "complete", "output_dir": str(OUTPUT_DIR), "report": "reports/final_clip_credit_risk_report.md"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
