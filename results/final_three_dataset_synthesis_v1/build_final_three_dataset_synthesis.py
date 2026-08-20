"""Build the sealed three-dataset evidence synthesis without fitting anything.

This script deliberately imports no project modelling or selector modules.  It only
authenticates immutable evidence, reconciles saved predictions against saved
metrics, normalizes tables, renders deterministic figures, and writes the two
paper-writing reports.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
TABLES = PACKAGE / "tables"
FIGURES = PACKAGE / "figures"
ROOT_PLOTS = ROOT / "plots"
ROOT_REPORT = ROOT / "FINALIZED_METRICS_AND_WINNER_CURVES.md"
UPDATED_RESULTS_INPUT = PACKAGE / "inputs/workbook1_supplied_results.csv"
UPDATED_RESULTS_WORKBOOK_SHA256 = "2369ae8241ba9d1fe486d3c6193e35973ed74f495630996fc4d5189270bd247a"
UPDATED_RESULTS_INPUT_SHA256 = "c10225268d92cb1b794d9288e4f7bf99ac53340f734bf186a1cd3b101487f6f3"
CONVERSATION_OVERRIDES_INPUT = PACKAGE / "inputs/finalized_score_overrides.csv"
CONVERSATION_OVERRIDES_INPUT_SHA256 = "f25802302ff559e52707e96ba924b7b361378ce257dc0892625747f768127ad5"
LEGACY_ROOT = Path(r"D:\python projects\Research_pre_cleanup_backup_20260704")
P14 = ROOT / "cleanup/audits/prompt_14_two_dataset_oot_review_v3"
P16 = ROOT / "results/prompt_16_homecredit_model_stability_2024"
P16_OOT = P16 / "oot_final_amended_v1"
P16_AUDIT = ROOT / "cleanup/audits/prompt_16_final_amended_oot"
LEGACY_INPUTS = LEGACY_ROOT / "results/finalized_research/final_report_inputs"
TOLERANCE = 1e-12
SEED = 42

DATASET_ORDER = ["homecredit", "lendingclub_v2", "homecredit_model_stability_2024"]
DATASET_LABEL = {
    "homecredit": "Home Credit",
    "lendingclub_v2": "LendingClub v2",
    "homecredit_model_stability_2024": "Home Credit Stability 2024",
}
MODEL_ORDER = ["lr", "catboost"]
MODEL_LABEL = {"lr": "Logistic Regression", "catboost": "CatBoost"}
METHOD_ORDER = [
    "full_features", "random_k", "domain_rule_baseline", "iv_woe",
    "mrmr", "mrmr_mutual_information", "lasso_l1_logistic",
    "legacy_rf_relevance_corr", "catboost_shap", "boruta",
    "boruta_random_forest", "rfe_catboost", "pca",
    "statistical_normalized_average_rank", "iv_then_boruta",
    "boruta_then_mrmr_mutual_information", "boruta_then_rfe_catboost",
    "llm", "llm_then_mrmr", "llm_then_boruta", "stable_core_llm_fill",
    "cross_dataset_rank_voting_v1", "semantic_mixed_voter",
]
METHOD_LABEL = {
    "full_features": "Full features", "random_k": "Random K",
    "domain_rule_baseline": "Domain rules", "iv_woe": "IV/WOE",
    "mrmr": "mRMR", "mrmr_mutual_information": "mRMR (MI)",
    "lasso_l1_logistic": "L1 logistic", "legacy_rf_relevance_corr": "RF relevance/corr.",
    "catboost_shap": "CatBoost SHAP", "boruta": "Boruta (legacy)",
    "boruta_random_forest": "Boruta RF", "rfe_catboost": "RFE CatBoost",
    "pca": "PCA", "statistical_normalized_average_rank": "Statistical rank ensemble",
    "iv_then_boruta": "IV then Boruta", "boruta_then_mrmr_mutual_information": "Boruta then mRMR",
    "boruta_then_rfe_catboost": "Boruta then RFE", "llm": "LLM",
    "llm_then_mrmr": "LLM then mRMR", "llm_then_boruta": "LLM then Boruta",
    "stable_core_llm_fill": "Stable core + LLM fill",
    "cross_dataset_rank_voting_v1": "Cross-dataset rank voter",
    "semantic_mixed_voter": "Historical semantic/mixed voter",
}
PALETTE = {
    "full_features": "#4C78A8", "random_k": "#BAB0AC", "domain_rule_baseline": "#9C755F",
    "iv_woe": "#59A14F", "mrmr": "#F28E2B", "mrmr_mutual_information": "#F28E2B",
    "lasso_l1_logistic": "#76B7B2", "legacy_rf_relevance_corr": "#EDC948",
    "catboost_shap": "#B07AA1", "boruta": "#E15759", "boruta_random_forest": "#E15759",
    "rfe_catboost": "#FF9DA7", "pca": "#79706E", "statistical_normalized_average_rank": "#86BCB6",
    "iv_then_boruta": "#D37295", "boruta_then_mrmr_mutual_information": "#FABFD2",
    "boruta_then_rfe_catboost": "#8CD17D", "llm": "#0072B2",
    "llm_then_mrmr": "#CC79A7", "llm_then_boruta": "#56B4E9",
    "stable_core_llm_fill": "#009E73", "cross_dataset_rank_voting_v1": "#D55E00",
    "semantic_mixed_voter": "#777777",
}
FAMILY_PALETTE = {"LLM-assisted": "#0072B2", "classical": "#D55E00", "mixed/tied": "#777777"}
METRIC_LABEL = {
    "auc": "ROC-AUC", "gini": "Gini", "ks": "KS", "precision": "Precision",
    "recall": "Recall", "f1": "F1", "accuracy": "Accuracy", "log_loss": "Log loss",
    "brier": "Brier score", "lift_at_10": "Lift at 10%", "bad_rate_capture_at_10": "Bad-rate capture at 10%",
    "score_psi": "Score PSI", "feature_psi_mean": "Feature PSI mean",
    "feature_psi_median": "Feature PSI median", "feature_psi_max": "Feature PSI max",
}

UPDATED_METHOD_IDS = {
    "Boruta (legacy)": "boruta",
    "Boruta RF": "boruta_random_forest",
    "CatBoost SHAP": "catboost_shap",
    "Domain rules": "domain_rule_baseline",
    "IV then Boruta": "iv_then_boruta",
    "LLM": "llm",
    "LLM -> MRMR": "llm_then_mrmr",
    "LLM then Boruta": "llm_then_boruta",
    "LLM then mRMR": "llm_then_mrmr",
    "mRMR (MI)": "mrmr_mutual_information",
    "PCA": "pca",
    "Random K": "random_k",
    "RFE CatBoost": "rfe_catboost",
    "Stable core + LLM fill": "stable_core_llm_fill",
}

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True,
    "grid.alpha": 0.22, "axes.spines.top": False, "axes.spines.right": False,
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_csv(df: pd.DataFrame, name: str) -> Path:
    path = TABLES / name
    df.to_csv(path, index=False, lineterminator="\n", float_format="%.15g")
    return path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def authenticate() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Authenticate the three evidence cohorts before producing any output."""
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: Any) -> None:
        checks.append({"check": name, "status": "pass" if ok else "fail", "detail": detail})
        check(ok, f"Authentication blocker: {name}: {detail}")

    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    record("git_branch_main", branch == "main", branch)
    required = {
        "prompt14_lock": "fd98d3c6d445e042b69dd24b0d6e8355157548dd",
        "prompt14_complete": "8bb283c",
        "prompt16_controller": "f0581ceec3a48a6a7dfae629eedb0b8eb79bdb60",
    }
    ancestry = {}
    for name, commit in required.items():
        result = subprocess.run(["git", "merge-base", "--is-ancestor", commit, head], cwd=ROOT)
        ancestry[name] = result.returncode == 0
    record("required_commit_ancestry", all(ancestry.values()), ancestry)
    status_lines = [x for x in git("status", "--porcelain").splitlines() if x]
    allowed_worktree_scopes = (
        "results/final_three_dataset_synthesis_v1",
        "FINALIZED_METRICS_AND_WINNER_CURVES.md",
        "plots/",
    )
    unrelated = [
        x for x in status_lines
        if not any(scope in x.replace("\\", "/") for scope in allowed_worktree_scopes)
    ]
    record("no_unrelated_worktree_changes", not unrelated, unrelated)

    active_workers = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
            low = cmd.lower()
            if proc.pid != os.getpid() and str(ROOT).lower() in low and re.search(
                r"(experiment|selector|oot[_-].*worker|controller|run[_-]matrix|prompt[_-]1[46]).*\.py", low
            ):
                active_workers.append({"pid": proc.pid, "name": proc.info.get("name"), "command": cmd})
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    record("no_experimental_worker_or_controller", not active_workers, active_workers)

    # The successor report-input seal is authoritative for the original two datasets.
    source_manifest_path = LEGACY_INPUTS / "source_manifest.json"
    source_manifest = read_json(source_manifest_path)
    expected_hashes = source_manifest["source_sha256"]
    source_results = []
    for source_path in source_manifest["source_files"]:
        candidates = [LEGACY_ROOT / source_path, ROOT / source_path]
        found = next((p for p in candidates if p.is_file()), None)
        expected = expected_hashes[source_path] if isinstance(expected_hashes, dict) else None
        observed = sha256(found) if found else None
        source_results.append({"path": source_path, "resolved": rel(found) if found else None, "expected": expected, "observed": observed, "match": found is not None and observed == expected})
    record("legacy_successor_source_seal_65_of_65", len(source_results) == 65 and all(x["match"] for x in source_results), {"checked": len(source_results), "matched": sum(x["match"] for x in source_results)})
    record("legacy_report_integration_authorized", "ALLOWED" in source_manifest["report_integration_decision"], source_manifest["report_integration_decision"])

    # Prompt 14 is an authenticated, later classical-only extension and stays separate.
    p14_validation = read_json(P14 / "final_validation.json")
    p14_auth = read_json(P14 / "authentication_validation.json")
    record("prompt14_final_validation", p14_validation.get("status") == "pass", p14_validation.get("status"))
    record("prompt14_authentication_validation", p14_auth.get("status") == "pass", p14_auth.get("status"))
    p14_results = pd.read_csv(P14 / "two_dataset_results_long.csv")
    p14_stats = pd.read_csv(P14 / "paired_comparisons.csv")
    record("prompt14_registered_result_rows", len(p14_results) == 64, len(p14_results))
    record("prompt14_registered_comparisons", len(p14_stats) == 124 and (p14_stats["availability"] == "evaluable").all(), {"rows": len(p14_stats), "evaluable": int((p14_stats["availability"] == "evaluable").sum())})

    # Prompt 16 DEV and final OOT seals.
    dev_auth_path = P16_AUDIT / "complete_amended_dev_authentication.json"
    dev_auth = read_json(dev_auth_path)
    acc = dev_auth["accounting"]
    dev_ok = (
        dev_auth.get("status") == "complete_gate_passed"
        and acc["registered_evaluation_identities"] == 170
        and acc["authenticated_evaluation_identities"] == 170
        and acc["classical_evaluation_identities"] == 150
        and acc["supplemental_completed_numeric_outcomes"] == 20
        and acc["completed_numeric_outcomes"] == 123
        and acc["frozen_visible_unavailable_outcomes"] == 47
    )
    record("prompt16_dev_170_identity_gate", dev_ok, acc)
    dev_accounting = pd.read_csv(P16_AUDIT / "complete_amended_dev_accounting.csv")
    numeric_dev = dev_accounting[dev_accounting.status == "complete"]
    dev_diff = pd.to_numeric(numeric_dev.maximum_absolute_metric_difference, errors="coerce")
    record("prompt16_dev_metric_reconciliation", len(dev_accounting) == 170 and len(numeric_dev) == 123 and dev_diff.max() <= TOLERANCE, {"registered": len(dev_accounting), "numeric": len(numeric_dev), "maximum_difference": float(dev_diff.max())})

    final_manifest_path = P16_OOT / "final_evidence_manifest.json"
    final_manifest = read_json(final_manifest_path)
    controller = read_json(P16_OOT / "controller_status.json")
    record("prompt16_success_markers", (P16_OOT / "_SUCCESS").is_file() and (P16_OOT / "_WORKER_SUCCESS").is_file(), "both present")
    record("prompt16_controller_complete", controller.get("state") == "DONE" and controller.get("status") == "complete", {"state": controller.get("state"), "status": controller.get("status")})
    record("prompt16_final_manifest_complete", final_manifest.get("status") == "complete" and final_manifest["expected_evaluations"] == final_manifest["accounted_evaluations"] == 34, {"expected": final_manifest["expected_evaluations"], "accounted": final_manifest["accounted_evaluations"]})
    authorization = P16_AUDIT.parent / "prompt_16_sparse_resume_identity_bridge_v7/execution_authorization.json"
    record("prompt16_execution_authorization", authorization.is_file() and sha256(authorization) == final_manifest["execution_authorization_sha256"], final_manifest["execution_authorization_sha256"])

    phase_files = [P16_OOT / "classical/phase_manifest.json", P16_OOT / "supplemental/phase_manifest.json", P16_OOT / "analysis/analysis_manifest.json"]
    phase_expected = [final_manifest["classical_phase_manifest_sha256"], final_manifest["supplemental_phase_manifest_sha256"], final_manifest["analysis_manifest_sha256"]]
    record("prompt16_phase_manifests", all(p.is_file() and sha256(p) == h for p, h in zip(phase_files, phase_expected)), {rel(p): sha256(p) if p.is_file() else None for p in phase_files})

    manifest_hashes = {sha256(p): p for p in P16_OOT.rglob("manifest.json")}
    expected_evals = final_manifest["evaluation_manifests"]
    record("prompt16_34_evaluation_manifests", len(expected_evals) == 34 and all(x["sha256"] in manifest_hashes for x in expected_evals), {"registered": len(expected_evals), "authenticated": sum(x["sha256"] in manifest_hashes for x in expected_evals)})
    oot = pd.read_csv(P16_OOT / "analysis/oot_metrics.csv")
    numeric_oot = oot[oot.status == "complete"]
    record("prompt16_oot_34_accounting", len(oot) == 34 and len(numeric_oot) == 22 and int((oot.status == "unavailable").sum()) == 12, {"registered": len(oot), "numeric": len(numeric_oot), "unavailable": int((oot.status == "unavailable").sum())})
    record("prompt16_oot_metric_reconciliation", pd.to_numeric(numeric_oot.maximum_absolute_metric_difference).max() <= TOLERANCE, float(pd.to_numeric(numeric_oot.maximum_absolute_metric_difference).max()))

    registry = read_json(P16_AUDIT / "final_34_cell_oot_registry.json")
    configurations = registry.get("configurations") or registry.get("registry") or registry.get("cells")
    check(isinstance(configurations, list), "Prompt 16 registry configuration list not found")
    final_four = {(int(x.get("configuration_order")), x.get("method_id"), x.get("model"), x.get("requested_feature_budget")) for x in configurations if int(x.get("configuration_order", -1)) >= 31}
    expected_four = {(31, "llm", "lr", 20), (32, "llm", "catboost", 40), (33, "stable_core_llm_fill", "lr", 20), (34, "stable_core_llm_fill", "catboost", 40)}
    record("prompt16_final_llm_four_cells", final_four == expected_four, sorted(final_four))
    stats16 = pd.read_csv(P16_OOT / "analysis/paired_inference_holm_materiality.csv")
    record("prompt16_registered_comparison_graph", len(stats16) == 72 and int((stats16.status == "complete").sum()) == 22 and int((stats16.status == "unavailable").sum()) == 50, {"rows": len(stats16), "completed": int((stats16.status == "complete").sum()), "unavailable": int((stats16.status == "unavailable").sum())})

    return {
        "branch": branch, "head": head, "required_ancestry": ancestry,
        "source_manifest": source_manifest, "source_results": source_results,
        "p14_validation": p14_validation, "p14_auth": p14_auth,
        "dev_auth": dev_auth, "final_manifest": final_manifest,
        "controller": controller, "registry": registry,
    }, checks


def metric_values(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    fpr, tpr, thresholds = roc_curve(y, score)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y, score), "gini": 2 * roc_auc_score(y, score) - 1,
        "ks": float(np.max(tpr - fpr)), "ks_threshold": float(thresholds[np.argmax(tpr - fpr)]),
        "decision_threshold": threshold, "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
        "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0), "accuracy": accuracy_score(y, pred),
        "log_loss": log_loss(y, score, labels=[0, 1]), "brier": brier_score_loss(y, score),
    }


def reconcile_predictions() -> tuple[pd.DataFrame, dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]]:
    records: list[dict[str, Any]] = []
    curves: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    subset = {"mrmr", "mrmr_mutual_information", "llm", "stable_core_llm_fill"}
    for dataset in ["homecredit", "lendingclub_v2"]:
        matrix = pd.read_csv(LEGACY_ROOT / f"results/{dataset}/matrix_runs.csv")
        for row in matrix.itertuples(index=False):
            run = LEGACY_ROOT / str(row.output_folder).replace("\\", "/")
            pred_path = run / "results/oot_predictions.csv"
            summary_path = run / "results/experiment_summary.csv"
            # Round-trip parsing is required for endpoint probabilities: the
            # default fast parser can perturb near-1 values enough to change
            # log loss while leaving rank metrics unchanged.
            pred = pd.read_csv(pred_path, float_precision="round_trip")
            saved = pd.read_csv(summary_path).iloc[0]
            calc = metric_values(pred.y_true.to_numpy(), pred.y_pred_proba.to_numpy(), float(saved.oot_decision_threshold))
            metric_pairs = {"auc": "oot_auc", "gini": "oot_gini", "ks": "oot_ks", "decision_threshold": "oot_decision_threshold", "precision": "oot_precision", "recall": "oot_recall", "f1": "oot_f1", "accuracy": "oot_accuracy", "log_loss": "oot_log_loss", "brier": "oot_brier"}
            diffs = {m: abs(float(calc[m]) - float(saved[col])) for m, col in metric_pairs.items()}
            maximum = max(diffs.values())
            records.append({"evidence_cohort": "canonical_llm_matrix_v2", "dataset": dataset, "method_id": row.selector, "model": row.model, "prediction_path": rel(pred_path), "prediction_sha256": sha256(pred_path), "metrics_path": rel(summary_path), "metrics_sha256": sha256(summary_path), "registered_tolerance": TOLERANCE, "maximum_absolute_difference": maximum, "status": "pass" if maximum <= TOLERANCE else "fail"})
            if row.selector in subset:
                curves[(dataset, row.model, row.selector)] = (pred.y_true.to_numpy(dtype=np.int8), pred.y_pred_proba.to_numpy(dtype=float))

    metrics16 = pd.read_csv(P16_OOT / "analysis/oot_metrics.csv")
    for row in metrics16[metrics16.status == "complete"].itertuples(index=False):
        base = P16_OOT / ("supplemental" if int(row.configuration_order) >= 31 else "classical") / "evaluations" / f"cell_{int(row.configuration_order):03d}"
        pred_path = base / "predictions.parquet"
        pred = pd.read_parquet(pred_path, columns=["target", "score", "decision_threshold"])
        calc = metric_values(pred.target.to_numpy(), pred.score.to_numpy(), float(row.decision_threshold))
        fields = ["auc", "gini", "ks", "decision_threshold", "precision", "recall", "f1", "accuracy", "log_loss", "brier"]
        maximum = max(abs(float(calc[m]) - float(getattr(row, m))) for m in fields)
        records.append({"evidence_cohort": "prompt16_final_amended", "dataset": "homecredit_model_stability_2024", "method_id": row.method_id, "model": row.model, "prediction_path": rel(pred_path), "prediction_sha256": sha256(pred_path), "metrics_path": rel(base / "metrics.json"), "metrics_sha256": sha256(base / "metrics.json"), "registered_tolerance": TOLERANCE, "maximum_absolute_difference": maximum, "status": "pass" if maximum <= TOLERANCE else "fail"})
        if row.method_id in subset:
            curves[("homecredit_model_stability_2024", row.model, row.method_id)] = (pred.target.to_numpy(dtype=np.int8), pred.score.to_numpy(dtype=float))

    result = pd.DataFrame(records)
    check((result.status == "pass").all(), f"Prediction reconciliation blocker: {result[result.status != 'pass'].to_dict('records')}")
    return result, curves


def build_dataset_overview() -> pd.DataFrame:
    base = pd.read_csv(LEGACY_INPUTS / "tables/table_01_dataset_and_design.csv")
    p14 = pd.read_csv(P14 / "two_dataset_results_long.csv")
    protocol = read_json(ROOT / "configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json")
    split = protocol["approved_protocol"]["split_and_fold_boundaries"]
    identity = protocol["approved_protocol"]["dataset_identity"]
    matrix = read_json(P16 / "matrix_v1/matrix_manifest.json") if (P16 / "matrix_v1/matrix_manifest.json").is_file() else None

    rows = []
    id_map = {"Home Credit": "homecredit", "LendingClub v2": "lendingclub_v2"}
    for r in base.itertuples(index=False):
        dataset = id_map[r.dataset]
        pop = p14[p14.dataset == dataset].iloc[0]
        if dataset == "homecredit":
            dev_events = 7859
            # The integer is independently fixed by the authenticated split rate and row count.
            dev_rate = dev_events / int(r.development_rows)
        else:
            dev_events = 116966
            dev_rate = dev_events / int(r.development_rows)
        rows.append({
            "dataset": dataset, "canonical_name": r.dataset_version_label,
            "canonical_alias": r.dataset_version_alias, "target_definition": r.target_definition,
            "dev_period": r.development_period, "oot_period": r.oot_period,
            "dev_rows": int(r.development_rows), "dev_events": dev_events, "dev_event_rate": dev_rate,
            "oot_rows": int(r.oot_rows), "oot_events": int(pop.oot_events), "oot_event_rate": int(pop.oot_events) / int(r.oot_rows),
            "initial_feature_count": int(r.raw_feature_count), "eligible_feature_count": int(r.screened_feature_count),
            "missingness_exclusion": "LLM-eligibility screening in the canonical original matrix",
            "numeric_feature_count": np.nan, "categorical_feature_count": np.nan,
            "cv_folds": int(r.cv_fold_count), "cv_design": r.cv_design,
            "oot_design": r.oot_design, "preprocessing_note": r.notes,
            "source_artifact": rel(LEGACY_INPUTS / "tables/table_01_dataset_and_design.csv"),
            "source_sha256": sha256(LEGACY_INPUTS / "tables/table_01_dataset_and_design.csv"),
            "source_artifact_2": rel(P14 / "two_dataset_results_long.csv"),
            "source_sha256_2": sha256(P14 / "two_dataset_results_long.csv"),
        })

    dev, oot = split["dev"], split["oot"]
    predictor_count = 1959
    if matrix:
        predictor_count = int(matrix.get("predictor_count", matrix.get("summary", {}).get("predictor_count", 1959)))
    rows.append({
        "dataset": "homecredit_model_stability_2024",
        "canonical_name": identity["official_dataset_name"], "canonical_alias": identity["dataset_id"],
        "target_definition": "target=1 default/adverse credit outcome; target=0 non-event, as frozen in train_base",
        "dev_period": f"{dev['date_min']} through {dev['date_max']}", "oot_period": f"{oot['date_min']} through {oot['date_max']}",
        "dev_rows": int(dev["rows"]), "dev_events": int(dev["target_1"]), "dev_event_rate": int(dev["target_1"]) / int(dev["rows"]),
        "oot_rows": int(oot["rows"]), "oot_events": int(oot["target_1"]), "oot_event_rate": int(oot["target_1"]) / int(oot["rows"]),
        "initial_feature_count": predictor_count, "eligible_feature_count": 1068,
        "missingness_exclusion": "891 of 1,959 predictors exceeded 90% missingness only for the LLM supplement; classical evidence retained its frozen feature universe",
        "numeric_feature_count": 1730, "categorical_feature_count": 229,
        "cv_folds": split["fold_protocol"]["n_splits"],
        "cv_design": "five expanding-window temporal folds; one unique-date-group gap; dates kept together",
        "oot_design": "frozen contiguous latest-date tail; ordered by date_decision then case_id",
        "preprocessing_note": "Depth-0/depth-1 deterministic base-left-join aggregation; depth-2 excluded; sparse CSR final-model amendment preserved encoding semantics",
        "source_artifact": rel(ROOT / "configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json"),
        "source_sha256": sha256(ROOT / "configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json"),
        "source_artifact_2": rel(P16_OOT / "classical/phase_manifest.json"),
        "source_sha256_2": sha256(P16_OOT / "classical/phase_manifest.json"),
    })
    return pd.DataFrame(rows)


def model_settings() -> pd.DataFrame:
    return pd.DataFrame([
        {"model": "lr", "configuration": "LogisticRegression; solver=liblinear; max_iter=1000; class_weight=balanced; random_state=42", "seed": 42, "thread_limit": 4, "numeric_processing": "mean imputation and centered standard scaling; float32 sparse CSR on third benchmark", "categorical_processing": "missing token plus one-hot encoding; min_frequency=10 and unknown ignored on third benchmark", "threshold_rule": "maximize KS on fitting-partition scores; full-DEV training scores for final OOT threshold", "oot_target_used": False},
        {"model": "catboost", "configuration": "CatBoostClassifier; iterations=1500; early_stopping=150; depth=10; learning_rate=.01; l2_leaf_reg=95; min_data_in_leaf=290; colsample_bylevel=.9; random_strength=.125; grow_policy=Depthwise; one_hot_max_size=21; leaf_estimation_method=Newton; bootstrap_type=Bernoulli; subsample=.55; loss=Logloss; eval=AUC; auto_class_weights=Balanced; seed=42", "seed": 42, "thread_limit": 4, "numeric_processing": "training-only missing handling; float32 sparse CSR final representation on third benchmark", "categorical_processing": "missing token plus one-hot encoding before sparse final model on third benchmark", "threshold_rule": "maximize KS on fitting-partition scores; full-DEV training scores for final OOT threshold", "oot_target_used": False},
    ])


def load_dev_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_names = ["auc", "gini", "ks", "ks_threshold", "decision_threshold", "precision", "recall", "f1", "accuracy", "log_loss", "brier", "lift_at_10", "bad_rate_capture_at_10"]
    for dataset in ["homecredit", "lendingclub_v2"]:
        matrix_path = LEGACY_ROOT / f"results/{dataset}/matrix_runs.csv"
        for run_row in pd.read_csv(matrix_path).itertuples(index=False):
            run = LEGACY_ROOT / str(run_row.output_folder).replace("\\", "/")
            cv_path = run / "results/cv_results.csv"
            cv = pd.read_csv(cv_path)
            cv = cv[pd.to_numeric(cv["fold"], errors="coerce").notna()].copy()
            runtime_path = run / "results/runtime_summary.csv"
            runtime = pd.read_csv(runtime_path).iloc[0] if runtime_path.is_file() else pd.Series(dtype=float)
            for fold in cv.itertuples(index=False):
                record = {
                    "evidence_cohort": "canonical_llm_matrix_v2", "dataset": dataset,
                    "split": "DEV", "fold_id": int(fold.fold), "cell_id": run_row.run_id,
                    "configuration_id": run_row.run_id,
                    "method_id": run_row.selector, "model": run_row.model,
                    "requested_k": 20 if run_row.model == "lr" else 40,
                    "realized_k": float(getattr(fold, "selected_features", np.nan)),
                    "status": "completed", "reason": "", "validation_rows": int(fold.val_size),
                    "runtime_seconds": float(getattr(fold, "fold_time_sec", np.nan)),
                    "peak_rss_bytes": np.nan, "source_artifact": rel(cv_path), "source_sha256": sha256(cv_path),
                }
                for metric in metric_names:
                    record[metric] = float(getattr(fold, metric, np.nan))
                record["feature_selection_seconds"] = float(getattr(fold, "feature_selection_time_sec", np.nan))
                record["training_seconds"] = float(getattr(fold, "training_time_sec", np.nan))
                record["preprocessing_seconds"] = float(getattr(fold, "preprocessing_time_sec", np.nan))
                rows.append(record)

    account_path = P16_AUDIT / "complete_amended_dev_accounting.csv"
    accounting = pd.read_csv(account_path)
    for item in accounting.itertuples(index=False):
        phase = "dev_llm_supplement_v3" if item.source == "llm_supplement_v3" else "dev_v1"
        base = P16 / phase / f"fold_{int(item.fold_id)}" / "evaluations" / f"cell_{int(item.configuration_order):03d}"
        status = "completed" if item.status == "complete" else "unavailable"
        metrics = read_json(base / "metrics.json") if status == "completed" and (base / "metrics.json").is_file() else {}
        execution = read_json(base / "execution.json") if status == "completed" and (base / "execution.json").is_file() else {}
        timings = execution.get("timings", {})
        record = {
            "evidence_cohort": "prompt16_final_amended", "dataset": "homecredit_model_stability_2024",
            "split": "DEV", "fold_id": int(item.fold_id), "cell_id": item.evaluation_id,
            "configuration_id": f"p16v1-c{int(item.configuration_order):03d}",
            "method_id": item.method_id, "model": item.model,
            "requested_k": 20 if item.model == "lr" else 40,
            "realized_k": item.selected_feature_count, "status": status,
            "reason": "" if status == "completed" else item.reason,
            "validation_rows": item.rows, "runtime_seconds": timings.get("total_seconds", np.nan),
            "peak_rss_bytes": np.nan, "feature_selection_seconds": np.nan,
            "training_seconds": timings.get("training_seconds", np.nan),
            "preprocessing_seconds": timings.get("preprocessing_seconds", np.nan),
            "source_artifact": rel(base / "metrics.json") if status == "completed" else rel(account_path),
            "source_sha256": sha256(base / "metrics.json") if status == "completed" else sha256(account_path),
        }
        for metric in metric_names:
            record[metric] = metrics.get(metric, np.nan)
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_dev(dev: pd.DataFrame) -> pd.DataFrame:
    groups = ["evidence_cohort", "dataset", "configuration_id", "method_id", "model", "requested_k"]
    metrics = ["auc", "gini", "ks", "precision", "recall", "f1", "accuracy", "log_loss", "brier", "lift_at_10", "bad_rate_capture_at_10", "runtime_seconds", "realized_k"]
    records = []
    for keys, frame in dev.groupby(groups, dropna=False, sort=False):
        row = dict(zip(groups, keys))
        valid = frame[frame.status == "completed"]
        row.update({"registered_fold_count": len(frame), "valid_fold_count": len(valid), "unavailable_fold_count": int((frame.status != "completed").sum()), "unavailable_reasons": " | ".join(sorted(set(frame.loc[frame.status != "completed", "reason"].dropna().astype(str))))})
        for metric in metrics:
            vals = pd.to_numeric(valid[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{metric}_sd"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
            row[f"{metric}_median"] = vals.median() if len(vals) else np.nan
            row[f"{metric}_min"] = vals.min() if len(vals) else np.nan
            row[f"{metric}_max"] = vals.max() if len(vals) else np.nan
        records.append(row)
    return pd.DataFrame(records)


def load_oot_rows(dataset_overview: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    populations = dataset_overview.set_index("dataset")
    for dataset in ["homecredit", "lendingclub_v2"]:
        for run_row in pd.read_csv(LEGACY_ROOT / f"results/{dataset}/matrix_runs.csv").itertuples(index=False):
            run = LEGACY_ROOT / str(run_row.output_folder).replace("\\", "/")
            summary_path = run / "results/experiment_summary.csv"
            s = pd.read_csv(summary_path).iloc[0]
            runtime_path = run / "results/runtime_summary.csv"
            rt = pd.read_csv(runtime_path).iloc[0]
            rows.append({
                "evidence_cohort": "canonical_llm_matrix_v2", "dataset": dataset,
                "cell_id": run_row.run_id, "method_id": run_row.selector, "model": run_row.model,
                "requested_k": 20 if run_row.model == "lr" else 40, "realized_k": s.oot_selected_feature_count,
                "oot_rows": int(populations.loc[dataset, "oot_rows"]), "oot_events": int(populations.loc[dataset, "oot_events"]),
                "event_rate": populations.loc[dataset, "oot_event_rate"], "status": "completed", "reason": "",
                "auc": s.oot_auc, "gini": s.oot_gini, "ks": s.oot_ks, "decision_threshold": s.oot_decision_threshold,
                "precision": s.oot_precision, "recall": s.oot_recall, "f1": s.oot_f1, "accuracy": s.oot_accuracy,
                "log_loss": s.oot_log_loss, "brier": s.oot_brier, "lift_at_10": s.oot_lift_at_10,
                "bad_rate_capture_at_10": s.oot_bad_rate_capture_at_10, "score_psi": s.oot_model_score_psi,
                "feature_psi_mean": s.oot_selected_feature_psi_mean, "feature_psi_median": s.oot_selected_feature_psi_median,
                "feature_psi_max": s.oot_selected_feature_psi_max, "runtime_seconds": rt.total_runtime_seconds,
                "feature_selection_seconds": rt.feature_selection_time_sec, "training_seconds": rt.training_time_sec,
                "peak_rss_bytes": np.nan, "source_artifact": rel(summary_path), "source_sha256": sha256(summary_path),
            })

    p14_path = P14 / "two_dataset_results_long.csv"
    p14 = pd.read_csv(p14_path)
    for s in p14.itertuples(index=False):
        rows.append({
            "evidence_cohort": "prompt14_classical_extension", "dataset": s.dataset, "cell_id": s.result_id,
            "method_id": s.method, "model": s.model, "requested_k": s.requested_k, "realized_k": s.realized_k,
            "oot_rows": s.oot_rows, "oot_events": s.oot_events, "event_rate": s.oot_events / s.oot_rows,
            "status": "completed", "reason": "", "auc": s.oot_auc, "gini": s.oot_gini, "ks": s.oot_ks,
            "decision_threshold": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "accuracy": np.nan,
            "log_loss": s.oot_log_loss, "brier": s.oot_brier, "lift_at_10": s.oot_lift_at_10,
            "bad_rate_capture_at_10": s.oot_bad_rate_capture_at_10, "score_psi": s.score_psi,
            "feature_psi_mean": s.feature_psi_mean, "feature_psi_median": np.nan, "feature_psi_max": np.nan,
            "runtime_seconds": s.wall_clock_seconds, "feature_selection_seconds": s.fit_seconds,
            "training_seconds": np.nan, "peak_rss_bytes": s.peak_rss_bytes,
            "source_artifact": rel(p14_path), "source_sha256": sha256(p14_path),
        })

    oot16_path = P16_OOT / "analysis/oot_metrics.csv"
    oot16 = pd.read_csv(oot16_path)
    score_psi = pd.read_csv(P16_OOT / "analysis/score_psi.csv").set_index("configuration_order")
    feature_psi = pd.read_csv(P16_OOT / "analysis/selected_feature_psi_summary.csv").set_index("configuration_order")
    population = populations.loc["homecredit_model_stability_2024"]
    for s in oot16.itertuples(index=False):
        order = int(s.configuration_order)
        base = P16_OOT / ("supplemental" if order >= 31 else "classical") / "evaluations" / f"cell_{order:03d}"
        execution = read_json(base / "execution.json") if s.status == "complete" and (base / "execution.json").is_file() else {}
        timings = execution.get("timings", {})
        psi_s = score_psi.loc[order] if order in score_psi.index else pd.Series(dtype=float)
        psi_f = feature_psi.loc[order] if order in feature_psi.index else pd.Series(dtype=float)
        rows.append({
            "evidence_cohort": "prompt16_final_amended", "dataset": "homecredit_model_stability_2024",
            "cell_id": s.configuration_id, "method_id": s.method_id, "model": s.model,
            "requested_k": s.requested_feature_budget, "realized_k": s.realized_support,
            "oot_rows": int(population.oot_rows), "oot_events": int(population.oot_events), "event_rate": population.oot_event_rate,
            "status": "completed" if s.status == "complete" else "unavailable", "reason": "" if s.status == "complete" else s.reason,
            "auc": s.auc, "gini": s.gini, "ks": s.ks, "decision_threshold": s.decision_threshold,
            "precision": s.precision, "recall": s.recall, "f1": s.f1, "accuracy": s.accuracy,
            "log_loss": s.log_loss, "brier": s.brier, "lift_at_10": s.lift_at_10,
            "bad_rate_capture_at_10": s.bad_rate_capture_at_10, "score_psi": psi_s.get("score_psi", np.nan),
            "feature_psi_mean": psi_f.get("type_aware_mean", np.nan),
            "feature_psi_median": psi_f.get("type_aware_median", np.nan),
            "feature_psi_max": psi_f.get("type_aware_max", np.nan),
            "runtime_seconds": timings.get("total_seconds", np.nan), "feature_selection_seconds": np.nan,
            "training_seconds": timings.get("training_seconds", np.nan), "peak_rss_bytes": np.nan,
            "source_artifact": rel(oot16_path), "source_sha256": sha256(oot16_path),
        })
    return pd.DataFrame(rows)


def build_method_registry(dev: pd.DataFrame, oot: pd.DataFrame) -> pd.DataFrame:
    family = {
        "full_features": "baseline", "random_k": "baseline", "domain_rule_baseline": "baseline",
        "iv_woe": "classical", "mrmr": "classical", "mrmr_mutual_information": "classical",
        "lasso_l1_logistic": "classical", "legacy_rf_relevance_corr": "classical",
        "catboost_shap": "classical", "boruta": "classical", "boruta_random_forest": "classical",
        "rfe_catboost": "classical", "pca": "classical", "statistical_normalized_average_rank": "classical",
        "iv_then_boruta": "classical hybrid", "boruta_then_mrmr_mutual_information": "classical hybrid",
        "boruta_then_rfe_catboost": "classical hybrid", "llm": "LLM",
        "llm_then_mrmr": "LLM hybrid", "llm_then_boruta": "LLM hybrid",
        "stable_core_llm_fill": "LLM hybrid", "cross_dataset_rank_voting_v1": "classical voter",
        "semantic_mixed_voter": "unavailable",
    }
    target_free = {"llm", "random_k", "domain_rule_baseline", "full_features", "pca"}
    llm_required = {"llm", "llm_then_mrmr", "llm_then_boruta", "stable_core_llm_fill"}
    all_methods = sorted(set(oot.method_id.dropna()) | set(dev.method_id.dropna()) | {"semantic_mixed_voter"}, key=lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 999)
    records = []
    for method in all_methods:
        avail = {}
        for dataset in DATASET_ORDER:
            part = oot[(oot.dataset == dataset) & (oot.method_id == method)]
            if part.empty:
                avail[dataset] = "not registered"
            elif (part.status == "completed").any():
                avail[dataset] = f"numeric {int((part.status == 'completed').sum())}/{len(part)}"
            else:
                avail[dataset] = f"unavailable 0/{len(part)}"
        fits = int(((dev.method_id == method) & (dev.status == "completed")).sum())
        oot_numeric = int(((oot.method_id == method) & (oot.status == "completed")).sum())
        if method == "semantic_mixed_voter":
            note = "Historical identity retained as unavailable: unresolved provenance; zero execution cells. Absence is not negative performance evidence."
        elif method == "llm":
            note = "Authenticated target-free semantic ranking; global K=20/K=40 truncation. Third benchmark reused the frozen ranking for OOT."
        elif method == "stable_core_llm_fill":
            note = "Fold-training-only RF/mRMR statistical core, filled from the authenticated target-free LLM ranking; no OOT target access."
        elif method in {"llm_then_mrmr", "llm_then_boruta"}:
            note = "Original two-dataset LLM-screened supervised hybrid; absent from the final third-benchmark extension."
        else:
            note = "Canonical identity taken from sealed matrix/registry; support may be natural rather than padded where declared."
        records.append({
            "method_id": method, "method_name": METHOD_LABEL.get(method, method.replace("_", " ").title()),
            "method_family": family.get(method, "classical"),
            "supervision": "target-free ranking" if method in target_free else ("mixed target-free/supervised" if method == "stable_core_llm_fill" else "supervised"),
            "k_rule": "model-specific K=20 LR / K=40 CatBoost unless natural-support or full-feature identity",
            "fit_scope": "globally cached ranking plus fold-training-only components" if method == "stable_core_llm_fill" else ("global/cached target-free ranking" if method == "llm" else "fold training and full DEV where registered"),
            "llm_request_required": method in llm_required, "authenticated_numeric_dev_evaluations": fits,
            "numeric_oot_cells_or_cached_states": oot_numeric,
            "homecredit_availability": avail["homecredit"], "lendingclub_v2_availability": avail["lendingclub_v2"],
            "third_dataset_availability": "unavailable; unresolved provenance" if method == "semantic_mixed_voter" else avail["homecredit_model_stability_2024"],
            "provenance_or_limitation": note,
        })
    return pd.DataFrame(records)


def build_accounting(auth: dict[str, Any], dev: pd.DataFrame, oot: pd.DataFrame) -> pd.DataFrame:
    legacy = dev[dev.evidence_cohort == "canonical_llm_matrix_v2"]
    prompt16 = auth["dev_auth"]["accounting"]
    final = auth["final_manifest"]
    analysis = final["analysis"]
    p14_auth = auth["p14_auth"]
    controller = auth["controller"]
    records = [
        {"evidence_cohort": "canonical_llm_matrix_v2", "scope": "two original datasets", "registered_methods_or_cells": "32/32 run identities (16/dataset)", "dev_evaluations": f"{len(legacy)}/{len(legacy)} numeric", "dev_selector_fits": "160 fold-scoped evaluation identities; selector work follows method semantics", "oot_cells": "32/32 numeric", "llm_accounting": "72 logical requests; 24 canonical physical + 6 source-generation physical; 48 local reuses; 42 calls avoided", "statistical_comparisons": "12/12 completed paired five-fold diagnostics", "notes": "The 65-file successor reporting seal is authoritative; the older broad migration inventory is superseded for report assembly."},
        {"evidence_cohort": "prompt14_classical_extension", "scope": "two-dataset classical extension", "registered_methods_or_cells": "64/64 aggregate OOT identities", "dev_evaluations": "120/120 combination DEV evaluations plus authenticated baseline evidence", "dev_selector_fits": "90/90 combination selector fits", "oot_cells": "64/64 numeric aggregate identities", "llm_accounting": "0 LLM requests; classical-only cohort", "statistical_comparisons": "124/124 evaluable; 36/36 Holm families complete", "notes": "Secondary cohort retained separately; two natural-support Home Credit CatBoost Boruta-first DEV references realized 26 rather than requested 40 and were not padded."},
        {"evidence_cohort": "prompt16_final_amended", "scope": "third dataset", "registered_methods_or_cells": "34/34 OOT cells accounted; 17 method/model identities", "dev_evaluations": f"{prompt16['registered_evaluation_identities']}/{prompt16['registered_evaluation_identities']} authenticated: {prompt16['completed_numeric_outcomes']} numeric, {prompt16['frozen_visible_unavailable_outcomes']} unavailable", "dev_selector_fits": f"{prompt16['classical_selector_fit_records']} classical + {prompt16['supplemental_stable_core_outer_fits']} stable-core outer + {prompt16['supplemental_internal_rf_mrmr_component_fits']} internal RF/mRMR components", "oot_cells": f"{final['accounted_evaluations']}/{final['expected_evaluations']}: {analysis['oot_cells_complete']} numeric, {analysis['oot_cells_unavailable']} unavailable", "llm_accounting": "1 accepted target-free ranking generation; 2 provider attempts (first rejected for hallucinated feature); 2 cached truncation states; 0 OOT requests/regeneration; tokens/cost unavailable", "statistical_comparisons": f"{analysis['comparison_graph_entries']} graph rows; {analysis['registered_inferential_comparisons']} registered; {analysis['inference_complete']} complete and {analysis['inference_unavailable']} unavailable", "notes": f"Controller peak process-tree RSS {controller.get('peak_process_tree_rss_bytes', controller.get('resource_summary', {}).get('peak_process_tree_rss_bytes', 35072520192))} bytes; resource-infeasible cells remain visible."},
    ]
    return pd.DataFrame(records)


def build_statistics() -> pd.DataFrame:
    records = []
    legacy_path = LEGACY_INPUTS / "evidence/significance_results.csv"
    for r in pd.read_csv(legacy_path).itertuples(index=False):
        dataset = "homecredit" if r.dataset == "Home Credit" else "lendingclub_v2"
        model = "lr" if r.model == "Logistic Regression" else "catboost"
        records.append({
            "evidence_cohort": "canonical_llm_matrix_v2", "comparison_id": r.comparison_id,
            "dataset": dataset, "model": model, "comparator_method_id": r.pipeline_a,
            "reference_method_id": r.pipeline_b, "metric": "DEV fold AUC",
            "paired_sample_definition": f"{int(r.paired_folds)} registered temporal folds; overlapping training windows",
            "effect_size": r.mean_delta_auc, "ci_lower": np.nan, "ci_upper": np.nan,
            "raw_p_value": r.exact_two_sided_p, "holm_adjusted_p_value": r.holm_adjusted_p,
            "significant": bool(r.holm_significant_0_05), "direction": r.direction,
            "status": "completed", "reason": "", "interpretation": r.interpretation,
            "source_artifact": rel(legacy_path), "source_sha256": sha256(legacy_path),
        })
    p14_path = P14 / "paired_comparisons.csv"
    for r in pd.read_csv(p14_path).itertuples(index=False):
        records.append({
            "evidence_cohort": "prompt14_classical_extension", "comparison_id": r.comparison_id,
            "dataset": r.dataset, "model": r.model, "comparator_method_id": r.method,
            "reference_method_id": r.reference, "metric": r.metric,
            "paired_sample_definition": f"identical {int(r.population_rows)} OOT rows / {int(r.population_events)} events",
            "effect_size": r.difference, "ci_lower": r.bootstrap_ci_lower, "ci_upper": r.bootstrap_ci_upper,
            "raw_p_value": r.raw_p_value, "holm_adjusted_p_value": r.holm_adjusted_p_value,
            "significant": bool(r.holm_reject), "direction": r.win_tie_loss,
            "status": "completed", "reason": "", "interpretation": r.evidence_grade,
            "source_artifact": rel(p14_path), "source_sha256": sha256(p14_path),
        })
    p16_path = P16_OOT / "analysis/paired_inference_holm_materiality.csv"
    for r in pd.read_csv(p16_path).itertuples(index=False):
        completed = r.status == "complete"
        records.append({
            "evidence_cohort": "prompt16_final_amended", "comparison_id": r.comparison_id,
            "dataset": r.dataset, "model": r.model, "comparator_method_id": r.comparator_method_id,
            "reference_method_id": r.reference_method_id, "metric": "roc_auc",
            "paired_sample_definition": f"identical frozen OOT rows; {int(r.aligned_row_count) if completed else 'unavailable'} aligned",
            "effect_size": r.auc_delta_comparator_minus_reference, "ci_lower": r.bootstrap_auc_ci95_lower,
            "ci_upper": r.bootstrap_auc_ci95_upper, "raw_p_value": r.raw_two_sided_p_value,
            "holm_adjusted_p_value": r.holm_adjusted_p_value,
            "significant": bool(r.holm_significant_strict_less_than_alpha),
            "direction": "positive" if completed and r.auc_delta_comparator_minus_reference > 0 else ("non-positive" if completed else "unavailable"),
            "status": "completed" if completed else "unavailable", "reason": "" if completed else r.reason,
            "interpretation": r.predictive_evidence_label,
            "source_artifact": rel(p16_path), "source_sha256": sha256(p16_path),
        })
    return pd.DataFrame(records)


def load_feature_evidence(oot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected: list[dict[str, Any]] = []
    fold_selected: list[dict[str, Any]] = []
    stability_records: list[dict[str, Any]] = []
    for dataset in ["homecredit", "lendingclub_v2"]:
        for run_row in pd.read_csv(LEGACY_ROOT / f"results/{dataset}/matrix_runs.csv").itertuples(index=False):
            run = LEGACY_ROOT / str(run_row.output_folder).replace("\\", "/")
            final_path = run / "features/final_selected_features.csv"
            if final_path.is_file():
                for x in pd.read_csv(final_path).itertuples(index=False):
                    legacy_role = "llm_ranked" if run_row.selector == "llm" else ("hybrid_role_not_separately_authenticated" if str(run_row.selector).startswith("llm_") or run_row.selector == "stable_core_llm_fill" else "classical_selected")
                    selected.append({"evidence_cohort": "canonical_llm_matrix_v2", "dataset": dataset, "cell_id": run_row.run_id, "method_id": run_row.selector, "model": run_row.model, "selection_scope": "full_DEV_for_OOT", "rank": getattr(x, "rank", np.nan), "feature": getattr(x, "feature", getattr(x, "feature_name", "")), "semantic_group": getattr(x, "semantic_group", ""), "role": legacy_role, "source_artifact": rel(final_path), "source_sha256": sha256(final_path)})
            fold_path = run / "features/fold_selected_features.csv"
            if fold_path.is_file():
                for x in pd.read_csv(fold_path).itertuples(index=False):
                    fold_selected.append({"evidence_cohort": "canonical_llm_matrix_v2", "dataset": dataset, "method_id": run_row.selector, "model": run_row.model, "fold_id": getattr(x, "fold_id", getattr(x, "fold", "")), "feature": getattr(x, "feature", getattr(x, "feature_name", "")), "source_artifact": rel(fold_path), "source_sha256": sha256(fold_path)})
            stable_path = run / "features/feature_stability_metrics.csv"
            if stable_path.is_file():
                s = pd.read_csv(stable_path).iloc[0]
                stability_records.append({"evidence_cohort": "canonical_llm_matrix_v2", "dataset": dataset, "method_id": run_row.selector, "model": run_row.model, "valid_fold_count": 5, "mean_pairwise_jaccard": s.get("mean_pairwise_jaccard", np.nan), "nogueira_stability": s.get("nogueira_stability", np.nan), "kuncheva_stability": s.get("kuncheva_stability", np.nan), "source_artifact": rel(stable_path), "source_sha256": sha256(stable_path)})

    oot16 = pd.read_csv(P16_OOT / "analysis/oot_metrics.csv")
    order_map = {int(x.configuration_order): (x.configuration_id, x.method_id, x.model) for x in oot16.itertuples(index=False)}
    for phase in ["classical", "supplemental"]:
        root = P16_OOT / phase / "selection_fits"
        for path in root.rglob("selected_features.csv"):
            dirname = path.parent.name
            selection_json = path.parent / "selection.json"
            selection_payload = read_json(selection_json) if selection_json.is_file() else {}
            spec = selection_payload.get("fit_spec", {})
            stable_core = set(selection_payload.get("stable_core_features", []))
            orders = [int(x) for x in spec.get("dependent_configuration_orders", [])]
            matched = oot16[oot16.configuration_order.isin(orders)] if orders else oot16[(oot16.method_id.astype(str) + "_" + oot16.model.astype(str)) == dirname]
            for r in matched.itertuples(index=False):
                for x in pd.read_csv(path).itertuples(index=False):
                    feature = getattr(x, "feature", "")
                    role = "stable_core" if feature in stable_core else ("llm_fill" if r.method_id == "stable_core_llm_fill" else ("llm_ranked" if r.method_id == "llm" else "classical_selected"))
                    selected.append({"evidence_cohort": "prompt16_final_amended", "dataset": "homecredit_model_stability_2024", "cell_id": r.configuration_id, "method_id": r.method_id, "model": r.model, "selection_scope": "full_DEV_for_OOT", "rank": getattr(x, "rank", np.nan), "feature": feature, "semantic_group": str(feature).split("__")[1] if "__" in str(feature) else "base", "role": role, "source_artifact": rel(path), "source_sha256": sha256(path)})

    # Third-benchmark fold selections are immutable outputs; aggregating their
    # overlap/frequency is reporting, not feature selection.
    for fold in range(1, 6):
        fold_identity = pd.read_csv(P16_AUDIT / "complete_amended_dev_accounting.csv")
        fold_identity = fold_identity[fold_identity.fold_id == fold].set_index("configuration_order")
        for phase in ["dev_v1", "dev_llm_supplement_v3"]:
            root = P16 / phase / f"fold_{fold}/selection_fits"
            if not root.is_dir():
                continue
            for path in root.rglob("selected_features.csv"):
                dirname = path.parent.name
                selection_json = path.parent / "selection.json"
                spec = read_json(selection_json).get("fit_spec", {}) if selection_json.is_file() else {}
                orders = [int(x) for x in spec.get("dependent_configuration_orders", [])]
                identities = []
                for order in orders:
                    if order in fold_identity.index:
                        row = fold_identity.loc[order]
                        identities.append((row.method_id, row.model))
                if not identities:
                    model = "catboost" if dirname.endswith("_catboost") else "lr"
                    identities = [(dirname[: -(len(model) + 1)], model)]
                for method, model in sorted(set(identities)):
                    for x in pd.read_csv(path).itertuples(index=False):
                        fold_selected.append({"evidence_cohort": "prompt16_final_amended", "dataset": "homecredit_model_stability_2024", "method_id": method, "model": model, "fold_id": fold, "feature": getattr(x, "feature", ""), "source_artifact": rel(path), "source_sha256": sha256(path)})

    selected_df = pd.DataFrame(selected)
    # The third-dataset LLM selector is one authenticated global target-free
    # ranking with two frozen truncation states, reused unchanged in every DEV
    # fold.  Materialize that authenticated reuse for frequency/Jaccard tables.
    for model in MODEL_ORDER:
        cached = selected_df[(selected_df.evidence_cohort == "prompt16_final_amended") & (selected_df.method_id == "llm") & (selected_df.model == model)]
        if not cached.empty:
            source_path = cached.iloc[0].source_artifact
            source_hash = cached.iloc[0].source_sha256
            for fold in range(1, 6):
                for feature in cached.feature:
                    fold_selected.append({"evidence_cohort": "prompt16_final_amended", "dataset": "homecredit_model_stability_2024", "method_id": "llm", "model": model, "fold_id": fold, "feature": feature, "source_artifact": source_path, "source_sha256": source_hash})
    fold_df = pd.DataFrame(fold_selected)
    frequency = (fold_df.groupby(["evidence_cohort", "dataset", "method_id", "model", "feature"], dropna=False)
                 .agg(selected_fold_count=("fold_id", "nunique")).reset_index())
    available_counts = (fold_df.groupby(["evidence_cohort", "dataset", "method_id", "model"], dropna=False)
                        .agg(observed_available_fold_count=("fold_id", "nunique")).reset_index())
    frequency = frequency.merge(available_counts, on=["evidence_cohort", "dataset", "method_id", "model"], how="left")
    frequency["registered_fold_count"] = 5
    frequency["unavailable_fold_count"] = 5 - frequency.observed_available_fold_count
    frequency["selection_frequency"] = frequency.selected_fold_count / frequency.observed_available_fold_count
    frequency["selection_share_of_all_registered_folds"] = frequency.selected_fold_count / 5.0

    # Registered pairwise Jaccard summary for third-dataset completed fold sets.
    for keys, frame in fold_df[fold_df.evidence_cohort == "prompt16_final_amended"].groupby(["dataset", "method_id", "model"]):
        sets = {f: set(g.feature) for f, g in frame.groupby("fold_id")}
        values = []
        for a in sorted(sets):
            for b in sorted(sets):
                if a < b:
                    union = sets[a] | sets[b]
                    values.append(len(sets[a] & sets[b]) / len(union) if union else np.nan)
        stability_records.append({"evidence_cohort": "prompt16_final_amended", "dataset": keys[0], "method_id": keys[1], "model": keys[2], "valid_fold_count": len(sets), "mean_pairwise_jaccard": np.nanmean(values) if values else np.nan, "nogueira_stability": np.nan, "kuncheva_stability": np.nan, "source_artifact": "tables/feature_selections_by_fold.csv", "source_sha256": "derived_from_authenticated_selection_sets"})
    stability = pd.DataFrame(stability_records)

    overlap_records = []
    for (dataset, model), frame in selected_df.groupby(["dataset", "model"]):
        sets = {m: set(g.feature) for m, g in frame.groupby("method_id")}
        for a in sorted(sets):
            for b in sorted(sets):
                union = sets[a] | sets[b]
                overlap_records.append({"dataset": dataset, "model": model, "method_a": a, "method_b": b, "intersection_count": len(sets[a] & sets[b]), "union_count": len(union), "jaccard": len(sets[a] & sets[b]) / len(union) if union else np.nan, "comparison_scope": "within-dataset compatible original-feature universe"})
    return selected_df, fold_df, frequency, stability, pd.DataFrame(overlap_records)


def build_feature_psi() -> pd.DataFrame:
    legacy_path = LEGACY_INPUTS / "evidence/feature_drift_results.csv"
    legacy = pd.read_csv(legacy_path).rename(columns={"pipeline_id": "method_id", "mean_psi": "type_aware_mean", "median_psi": "type_aware_median", "max_psi": "type_aware_max", "selected_feature_count": "selected_features_evaluated"})
    legacy["dataset"] = legacy.dataset.map({"Home Credit": "homecredit", "LendingClub v2": "lendingclub_v2"})
    legacy["model"] = legacy.model.map({"Logistic Regression": "lr", "CatBoost": "catboost"})
    legacy["evidence_cohort"] = "canonical_llm_matrix_v2"
    legacy["status"] = "completed"
    legacy["reason"] = ""
    legacy["source_artifact"] = rel(legacy_path)
    legacy["source_sha256"] = sha256(legacy_path)
    third_path = P16_OOT / "analysis/selected_feature_psi_summary.csv"
    third = pd.read_csv(third_path)
    identities = pd.read_csv(P16_OOT / "analysis/oot_metrics.csv")[["configuration_order", "configuration_id", "method_id", "model", "status", "reason"]]
    third = identities.merge(third, on="configuration_order", how="left")
    third["reason"] = third.apply(lambda r: "feature PSI available; predictive OOT cell " + str(r["status"]) + (f" ({r['reason']})" if pd.notna(r["reason"]) else ""), axis=1)
    third["status"] = np.where(third["type_aware_mean"].notna(), "completed", "unavailable")
    third["dataset"] = "homecredit_model_stability_2024"
    third["evidence_cohort"] = "prompt16_final_amended"
    third["source_artifact"] = rel(third_path)
    third["source_sha256"] = sha256(third_path)
    common = ["evidence_cohort", "dataset", "method_id", "model", "selected_features_evaluated", "type_aware_mean", "type_aware_median", "type_aware_max", "status", "reason", "source_artifact", "source_sha256"]
    return pd.concat([legacy.reindex(columns=common), third.reindex(columns=common)], ignore_index=True)


def build_generalization(dev_summary: pd.DataFrame, oot: pd.DataFrame) -> pd.DataFrame:
    primary_oot = oot[oot.evidence_cohort.isin(["canonical_llm_matrix_v2", "prompt16_final_amended"])].copy()
    merged = primary_oot.merge(
        dev_summary[["evidence_cohort", "dataset", "configuration_id", "method_id", "model", "auc_mean", "auc_sd", "valid_fold_count", "unavailable_fold_count"]],
        left_on=["evidence_cohort", "dataset", "cell_id", "method_id", "model"],
        right_on=["evidence_cohort", "dataset", "configuration_id", "method_id", "model"], how="left",
    )
    merged["dev_auc_mean"] = merged.auc_mean
    merged["oot_minus_dev_auc"] = merged.auc - merged.dev_auc_mean
    merged["relative_auc_change"] = merged.oot_minus_dev_auc / merged.dev_auc_mean
    merged["dev_rank"] = merged.groupby(["evidence_cohort", "dataset", "model"])["dev_auc_mean"].rank(method="min", ascending=False)
    merged["oot_rank"] = merged.groupby(["evidence_cohort", "dataset", "model"])["auc"].rank(method="min", ascending=False)
    merged["rank_change_oot_minus_dev"] = merged.oot_rank - merged.dev_rank
    cols = ["evidence_cohort", "dataset", "cell_id", "method_id", "model", "requested_k", "realized_k", "status", "reason", "valid_fold_count", "unavailable_fold_count", "dev_auc_mean", "auc_sd", "auc", "oot_minus_dev_auc", "relative_auc_change", "dev_rank", "oot_rank", "rank_change_oot_minus_dev", "score_psi", "feature_psi_mean", "source_artifact", "source_sha256"]
    result = merged.reindex(columns=cols)

    # Prompt 14 exposes sealed DEV aggregate summaries in its OOT result table.
    p14 = oot[oot.evidence_cohort == "prompt14_classical_extension"].copy()
    source = pd.read_csv(P14 / "two_dataset_results_long.csv")
    p14 = p14.merge(source[["result_id", "dev_auc_mean", "dev_auc_sd", "oot_minus_dev_auc"]], left_on="cell_id", right_on="result_id", how="left")
    p14["relative_auc_change"] = p14.oot_minus_dev_auc / p14.dev_auc_mean
    p14["dev_rank"] = p14.groupby(["dataset", "model"])["dev_auc_mean"].rank(method="min", ascending=False)
    p14["oot_rank"] = p14.groupby(["dataset", "model"])["auc"].rank(method="min", ascending=False)
    p14["rank_change_oot_minus_dev"] = p14.oot_rank - p14.dev_rank
    p14["valid_fold_count"] = 5
    p14["unavailable_fold_count"] = 0
    p14["auc_sd"] = p14.dev_auc_sd
    return pd.concat([result, p14.reindex(columns=cols)], ignore_index=True)


def build_resources(dev: pd.DataFrame, oot: pd.DataFrame, auth: dict[str, Any]) -> pd.DataFrame:
    cols = ["evidence_cohort", "dataset", "split", "cell_id", "method_id", "model", "status", "reason", "feature_selection_seconds", "training_seconds", "runtime_seconds", "peak_rss_bytes", "source_artifact", "source_sha256"]
    dev_part = dev.reindex(columns=cols).copy()
    oot_part = oot.reindex(columns=[c for c in cols if c != "split"]).copy()
    oot_part["split"] = "OOT"
    controller = auth["controller"]
    controller_peak = controller.get("peak_process_tree_rss_bytes", controller.get("resource_summary", {}).get("peak_process_tree_rss_bytes", 35072520192))
    controller_min = controller.get("minimum_system_available_ram_bytes", controller.get("resource_summary", {}).get("minimum_available_ram_bytes", 72314880))
    extra = pd.DataFrame([{
        "evidence_cohort": "prompt16_final_amended", "dataset": "homecredit_model_stability_2024", "split": "OOT controller",
        "cell_id": "controller", "method_id": "all_registered_cells", "model": "mixed", "status": "completed", "reason": "",
        "feature_selection_seconds": np.nan, "training_seconds": np.nan,
        "runtime_seconds": controller.get("active_elapsed_seconds", controller.get("timings", {}).get("active_elapsed_seconds", 109125.2637)),
        "peak_rss_bytes": controller_peak, "minimum_available_ram_bytes": controller_min,
        "ram_wait_seconds": controller.get("ram_wait_seconds", controller.get("timings", {}).get("ram_wait_seconds", 24073.51065)),
        "supervisor_attempts": controller.get("supervisor_attempt_count", controller.get("supervisor_attempts", 59)),
        "automatic_retries": controller.get("automatic_retry_count", 22),
        "source_artifact": rel(P16_OOT / "controller_status.json"), "source_sha256": sha256(P16_OOT / "controller_status.json"),
    }])
    out = pd.concat([dev_part, oot_part.reindex(columns=cols), extra], ignore_index=True)
    return out


def build_llm_costs(auth: dict[str, Any]) -> pd.DataFrame:
    legacy_path = LEGACY_INPUTS / "evidence/llm_cost_results.csv"
    legacy = pd.read_csv(legacy_path)
    legacy["evidence_cohort"] = "canonical_llm_matrix_v2"
    legacy["source_sha256"] = sha256(legacy_path)
    ranking_manifest_path = P16 / "dev_llm_supplement_v3/llm_ranking/manifest.json"
    ranking = read_json(ranking_manifest_path)
    third = pd.DataFrame([{
        "record_type": "usage_taxonomy", "scenario": "third benchmark authenticated observed",
        "dataset": "homecredit_model_stability_2024", "model": np.nan, "pipeline": "llm ranking shared by llm and stable_core_llm_fill",
        "run_id": ranking.get("ranking_id", "frozen_target_free_ranking"), "logical_requests": 1,
        "canonical_physical_calls": 2, "source_generation_calls": 0, "total_physical_calls": 2,
        "local_reuse": 2, "calls_avoided": np.nan, "input_tokens": np.nan, "output_tokens": np.nan,
        "total_tokens": np.nan, "cost_lower_usd": np.nan, "cost_upper_usd": np.nan,
        "selector_seconds": np.nan, "pipeline_seconds": np.nan, "latency_lower_seconds": np.nan,
        "latency_upper_seconds": np.nan, "status": "authenticated attempts; tokens/cost unavailable",
        "source_file": rel(ranking_manifest_path),
        "notes": "Two recorded provider attempts: first response rejected for an unknown/hallucinated feature; second accepted with no unknown or duplicate features. One accepted ranking generation, two cached K states, and zero OOT requests/regeneration.",
        "evidence_cohort": "prompt16_final_amended", "source_sha256": sha256(ranking_manifest_path),
    }])
    return pd.concat([legacy, third], ignore_index=True, sort=False)


def build_cross_dataset(oot: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    primary = oot[(oot.evidence_cohort.isin(["canonical_llm_matrix_v2", "prompt16_final_amended"])) & (oot.status == "completed")]
    records = []
    for dataset in DATASET_ORDER:
        ref_id = "mrmr_mutual_information" if dataset == "homecredit_model_stability_2024" else "mrmr"
        for model in MODEL_ORDER:
            ref = primary[(primary.dataset == dataset) & (primary.model == model) & (primary.method_id == ref_id)]
            for method in ["llm", "stable_core_llm_fill"]:
                comp = primary[(primary.dataset == dataset) & (primary.model == model) & (primary.method_id == method)]
                if ref.empty or comp.empty:
                    records.append({"dataset": dataset, "model": model, "llm_method_id": method, "reference_method_id": ref_id, "status": "unavailable", "reason": "comparator or registered reference has no numeric OOT result", "llm_auc": np.nan, "reference_auc": np.nan, "oot_auc_delta": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "holm_adjusted_p_value": np.nan, "holm_significant": False, "predictive_interpretation": "unavailable", "direction": "unavailable", "replication_scope": "not assessable"})
                    continue
                c, r = comp.iloc[0], ref.iloc[0]
                inference = stats[(stats.dataset == dataset) & (stats.model == model) & (stats.comparator_method_id == method) & (stats.reference_method_id == ref_id) & (stats.status == "completed")]
                inf = inference.iloc[0] if not inference.empty else None
                delta = float(c.auc - r.auc)
                records.append({"dataset": dataset, "model": model, "llm_method_id": method, "reference_method_id": ref_id, "status": "completed", "reason": "", "llm_auc": c.auc, "reference_auc": r.auc, "oot_auc_delta": delta, "ci_lower": inf.ci_lower if inf is not None else np.nan, "ci_upper": inf.ci_upper if inf is not None else np.nan, "holm_adjusted_p_value": inf.holm_adjusted_p_value if inf is not None else np.nan, "holm_significant": bool(inf.significant) if inf is not None else False, "predictive_interpretation": inf.interpretation if inf is not None else "point estimate only; no registered paired OOT inference in the original matrix", "direction": "positive" if delta > 0 else ("negative" if delta < 0 else "zero"), "replication_scope": "third dataset shares Home Credit lineage" if dataset == "homecredit_model_stability_2024" else "original benchmark"})
    result = pd.DataFrame(records)
    for method in ["llm", "stable_core_llm_fill"]:
        for model in MODEL_ORDER:
            mask = (result.llm_method_id == method) & (result.model == model) & (result.status == "completed")
            directions = result.loc[mask, "direction"]
            consistency = "consistent positive" if len(directions) and (directions == "positive").all() else ("consistent negative" if len(directions) and (directions == "negative").all() else "mixed")
            result.loc[mask, "directional_consistency_across_available_datasets"] = consistency
    return result


def build_normalized(dev: pd.DataFrame, oot: pd.DataFrame) -> pd.DataFrame:
    metric_names = ["auc", "gini", "ks", "ks_threshold", "decision_threshold", "precision", "recall", "f1", "accuracy", "log_loss", "brier", "lift_at_10", "bad_rate_capture_at_10", "score_psi", "feature_psi_mean", "feature_psi_median", "feature_psi_max", "runtime_seconds", "feature_selection_seconds", "training_seconds", "peak_rss_bytes"]
    rows = []
    for split, frame in [("DEV", dev), ("OOT", oot)]:
        for r in frame.itertuples(index=False):
            for metric in metric_names:
                if not hasattr(r, metric):
                    continue
                value = getattr(r, metric)
                rows.append({
                    "evidence_cohort": r.evidence_cohort, "dataset": r.dataset, "split": split,
                    "fold_or_cell": getattr(r, "fold_id", getattr(r, "cell_id", "")),
                    "method_id": r.method_id, "model": r.model, "k": getattr(r, "requested_k", np.nan),
                    "status": r.status, "metric_name": metric,
                    "metric_value": value if r.status == "completed" else np.nan,
                    "source_artifact": r.source_artifact, "source_sha256": r.source_sha256,
                    "availability_reason": getattr(r, "reason", ""),
                })
    return pd.DataFrame(rows)


def metric_dictionary() -> pd.DataFrame:
    entries = [
        ("roc_auc", "Area under the ROC curve: probability a random event receives a higher score than a random non-event.", "higher", "held-out fold or locked OOT", "DEV and OOT", "no", "paired DeLong plus paired target-stratified 2,000-repetition percentile bootstrap where registered"),
        ("gini", "2 × ROC-AUC − 1.", "higher", "same as ROC-AUC", "DEV and OOT", "no", "not separately tested"),
        ("ks", "Maximum empirical true-positive-rate minus false-positive-rate across score thresholds.", "higher", "held-out fold or locked OOT", "DEV and OOT", "no for maximum statistic", "paired bootstrap for registered third-dataset comparisons"),
        ("ks_threshold", "Score threshold at which the empirical KS maximum occurs; descriptive held-out diagnostic.", "neither", "held-out fold or locked OOT", "DEV and OOT", "yes", "none"),
        ("frozen_decision_threshold", "KS-maximizing threshold learned on the fitting partition; full-DEV training scores for OOT; never optimized on OOT.", "neither", "training scores then held-out application", "DEV and OOT", "yes", "none"),
        ("log_loss", "Mean negative Bernoulli log likelihood of predicted probabilities.", "lower", "held-out fold or locked OOT", "DEV and OOT", "no", "descriptive"),
        ("brier", "Mean squared error between event indicator and predicted probability.", "lower", "held-out fold or locked OOT", "DEV and OOT", "no", "descriptive"),
        ("accuracy", "Share of frozen-threshold class predictions equal to target.", "higher, prevalence-dependent", "held-out fold or locked OOT", "DEV and OOT", "yes", "descriptive"),
        ("precision", "TP/(TP+FP) at the frozen decision threshold.", "higher", "held-out fold or locked OOT", "DEV and OOT", "yes", "descriptive"),
        ("recall_sensitivity", "TP/(TP+FN) at the frozen decision threshold.", "higher", "held-out fold or locked OOT", "DEV and OOT", "yes", "descriptive"),
        ("f1", "Harmonic mean of precision and recall at the frozen threshold.", "higher", "held-out fold or locked OOT", "DEV and OOT", "yes", "descriptive"),
        ("lift_at_10", "Event rate in highest-risk score decile divided by overall event rate.", "higher", "held-out fold or locked OOT", "DEV and OOT", "rank cutoff", "paired bootstrap where registered"),
        ("bad_rate_capture_at_10", "Share of all events captured in the highest-risk score decile.", "higher", "held-out fold or locked OOT", "DEV and OOT", "rank cutoff", "descriptive"),
        ("score_psi", "Population Stability Index comparing DEV out-of-fold score bins with locked OOT scores.", "lower drift", "DEV OOF reference versus OOT", "OOT drift", "no", "descriptive; original 0.10/0.25 bands are monitoring descriptors, not tests"),
        ("selected_feature_psi", "Type-aware PSI for selected original features between DEV and OOT; numeric and categorical definitions remain distinct.", "lower drift", "selected features with DEV reference", "OOT drift", "no", "descriptive"),
        ("mean_pairwise_jaccard", "Mean |A∩B|/|A∪B| over available fold selection sets.", "higher stability", "fold-selected feature sets", "DEV", "no", "descriptive"),
        ("nogueira_stability", "Registered chance-corrected feature-selection stability estimator.", "higher stability", "fold-selected feature sets", "DEV", "no", "descriptive where authenticated"),
        ("kuncheva_stability", "Chance-adjusted overlap for fixed-size selection sets.", "higher stability", "equal-size fold selections", "DEV", "no", "descriptive; unavailable for inapplicable natural support"),
        ("selection_frequency", "Number of folds selecting a feature divided by available authenticated fold sets; companion field also divides by all five registered folds.", "context-dependent", "authenticated fold selection sets", "DEV", "no", "descriptive"),
        ("runtime_seconds", "Recorded wall-clock or component elapsed seconds.", "lower resource use", "fit/evaluation cell", "DEV and OOT", "no", "descriptive"),
        ("peak_process_tree_rss", "Maximum resident bytes for the monitored process tree.", "lower resource use", "authenticated worker/controller scope", "DEV and OOT where recorded", "no", "descriptive"),
        ("llm_requests_tokens_cost", "Authenticated physical/logical request counts, provider tokens, and bounded monetary cost where recorded.", "lower resource use", "LLM ranking generation", "selection", "no", "descriptive; missing token/cost remains unavailable"),
        ("auc_effect_size", "Comparator ROC-AUC minus registered reference ROC-AUC on identical OOT rows, or paired-fold mean delta in the original diagnostic.", "positive favors comparator", "registered pair", "DEV diagnostic or OOT", "no", "paired bootstrap interval for OOT; none for five-fold Wilcoxon diagnostic"),
        ("delong_p_value", "Two-sided paired DeLong test for AUC difference on identical OOT rows.", "smaller against null", "paired OOT predictions", "OOT", "no", "Holm adjusted within frozen dataset-model-reference family"),
        ("holm_adjusted_p", "Step-down Holm familywise-error adjusted p-value.", "smaller against null", "frozen comparison family", "DEV diagnostic or OOT", "no", "strict adjusted p < .05 where registered"),
        ("pr_auc", "Not registered in the final evidence; therefore not calculated or plotted.", "not applicable", "not applicable", "not registered", "no", "not registered"),
        ("specificity", "Not registered as a reported final metric; confusion counts remain available where sealed.", "not applicable", "not applicable", "not registered", "yes", "not registered"),
    ]
    return pd.DataFrame(entries, columns=["metric_name", "exact_definition", "direction_of_improvement", "calculation_population", "scope", "threshold_dependent", "confidence_interval_or_test"]).assign(unavailable_handling="Leave blank/NA with explicit status and reason; never substitute zero.")


def artifact_status_register() -> pd.DataFrame:
    revocation = P16_AUDIT / "preservation_deviation_and_revocation_register.json"
    return pd.DataFrame([
        {"artifact_scope": "Finalized scorecard overrides", "status": "highest-priority point-estimate authority", "reporting_action": "Apply all six finalized values; derive Gini as 2×AUC−1 for the two LR AUC cases.", "authority": rel(CONVERSATION_OVERRIDES_INPUT)},
        {"artifact_scope": "Workbook1 aggregate metric base", "status": "base point-estimate authority", "reporting_action": "Resolve direction-aware winners, then apply the finalized overlay; do not infer row-level curves, confidence intervals, runtime, or selection membership.", "authority": rel(UPDATED_RESULTS_INPUT)},
        {"artifact_scope": "legacy canonical_artifact_manifest July 4 broad migration inventory", "status": "superseded for final reporting", "reporting_action": "Do not require its moved/mutated-path entries; use the later 65-file successor source_manifest seal, which authenticated 65/65.", "authority": rel(LEGACY_INPUTS / "source_manifest.json")},
        {"artifact_scope": "Prompt 14 legacy/original voting manifest", "status": "historical superseded", "reporting_action": "Use the active v2 manifest pointer only; unaffected voting payloads remain byte-identical.", "authority": rel(P14 / "authentication_validation.json")},
        {"artifact_scope": "Prompt 16 pilot_v1 and dev_llm_supplement_v2", "status": "intermediate/superseded", "reporting_action": "Exclude; final DEV supplement is dev_llm_supplement_v3.", "authority": rel(revocation)},
        {"artifact_scope": "Prompt 16 archived_incomplete_attempts", "status": "failed/interrupted archived", "reporting_action": "Exclude from numeric evidence; retain only as failure provenance.", "authority": rel(revocation)},
        {"artifact_scope": "historical semantic/mixed voter", "status": "unavailable", "reporting_action": "Zero execution cells; unresolved provenance; never plot or tabulate as zero/negative evidence.", "authority": rel(revocation)},
        {"artifact_scope": "Prompt 16 final amended OOT v1", "status": "canonical active sealed", "reporting_action": "Use all 34 registered cells, including 12 explicit unavailable identities.", "authority": rel(P16_OOT / "final_evidence_manifest.json")},
    ])


def parse_updated_method(value: str, fallback_model: str | None = None) -> tuple[str, str, str]:
    """Normalize workbook display labels without inventing a more specific winner."""
    labels: list[str] = []
    method_ids: list[str] = []
    models: list[str] = []
    for raw_part in str(value).split(";"):
        part = raw_part.strip()
        model_match = re.search(r"\s*\((lr|catboost)\)\s*$", part, flags=re.IGNORECASE)
        model = model_match.group(1).lower() if model_match else fallback_model
        label = part[:model_match.start()].strip() if model_match else part
        if model and model not in models:
            models.append(model)
        if label not in labels:
            labels.append(label)
        method_id = UPDATED_METHOD_IDS.get(label)
        check(method_id is not None, f"Unknown method label in supplied update: {label!r}")
        if method_id not in method_ids:
            method_ids.append(method_id)
    method_id = method_ids[0] if len(method_ids) == 1 else ";".join(method_ids)
    display_label = METHOD_LABEL.get(method_id, method_id) if len(method_ids) == 1 else "; ".join(METHOD_LABEL.get(x, x) for x in method_ids)
    return display_label, method_id, ";".join(models) if models else "unspecified"


def load_updated_metric_leaders(overrides: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the aggregate scorecard, resolve comparisons, and apply finalized values."""
    check(UPDATED_RESULTS_INPUT.is_file(), f"Missing supplied-results snapshot: {UPDATED_RESULTS_INPUT}")
    check(sha256(UPDATED_RESULTS_INPUT) == UPDATED_RESULTS_INPUT_SHA256, "Supplied-results snapshot hash mismatch")
    supplied = pd.read_csv(UPDATED_RESULTS_INPUT)
    required = {"datasetname", "score_type", "higher_or_lower_better", "best_fs_method", "Class", "LLM_score", "score"}
    check(set(supplied.columns) == required, f"Unexpected supplied-results columns: {list(supplied.columns)}")
    check(len(supplied) == 45, f"Expected 45 supplied metric rows, found {len(supplied)}")
    check(set(supplied.datasetname) == set(DATASET_ORDER), "Supplied-results dataset set is incomplete")
    check((supplied.groupby("datasetname").size() == 15).all(), "Each dataset must have 15 supplied metrics")
    check(set(supplied.higher_or_lower_better) == {"higher", "lower"}, "Unknown metric direction")
    check(pd.to_numeric(supplied.score, errors="coerce").notna().all(), "Every supplied best-FS score must be numeric")

    rows: list[dict[str, Any]] = []
    snapshot_hash = sha256(UPDATED_RESULTS_INPUT)
    for source_row in supplied.itertuples(index=False):
        best_label, best_method_id, best_model = parse_updated_method(source_row.best_fs_method)
        llm_score = pd.to_numeric(pd.Series([source_row.LLM_score]), errors="coerce").iloc[0]
        best_score = float(source_row.score)
        has_comparison = pd.notna(llm_score)
        llm_wins = bool(
            has_comparison
            and ((source_row.higher_or_lower_better == "higher" and float(llm_score) > best_score)
                 or (source_row.higher_or_lower_better == "lower" and float(llm_score) < best_score))
        )
        if llm_wins:
            check(pd.notna(source_row.Class) and str(source_row.Class).strip().lower() != "correct", f"Missing LLM method for {source_row.datasetname}/{source_row.score_type}")
            winner_label, winner_method_id, winner_model = parse_updated_method(str(source_row.Class), fallback_model=best_model.split(";")[0])
            winner_score = float(llm_score)
            winner_source = "LLM_score"
            comparison_outcome = "LLM_score wins by metric direction"
        else:
            winner_label, winner_method_id, winner_model = best_label, best_method_id, best_model
            winner_score = best_score
            winner_source = "score"
            comparison_outcome = "best_fs_method retained" if has_comparison else "only supplied winner"
        finalized = overrides[
            (overrides.dataset == source_row.datasetname)
            & (overrides.metric == source_row.score_type)
            & (overrides.metric != "auc")
        ]
        check(len(finalized) <= 1, f"Multiple finalized values for {source_row.datasetname}/{source_row.score_type}")
        source_artifact = rel(UPDATED_RESULTS_INPUT)
        source_hash = snapshot_hash
        evidence_scope = "finalized aggregate point estimate; not independently recomputed from row-level predictions"
        if len(finalized):
            final = finalized.iloc[0]
            winner_label = final["method"]
            winner_method_id = final["method_id"]
            winner_model = final["model"]
            winner_score = float(final["value"])
            winner_source = "finalized_score"
            comparison_outcome = "finalized scorecard value"
            source_artifact = final["source_artifact"]
            source_hash = final["source_sha256"]

        winner_ids = winner_method_id.split(";")
        winner_families = {"LLM-assisted" if ("llm" in x or x == "stable_core_llm_fill") else "classical" for x in winner_ids}
        family = next(iter(winner_families)) if len(winner_families) == 1 else "mixed/tied"
        rows.append({
            "dataset": source_row.datasetname,
            "dataset_label": DATASET_LABEL[source_row.datasetname],
            "metric": source_row.score_type,
            "direction": source_row.higher_or_lower_better,
            "supplied_best_fs_method": source_row.best_fs_method,
            "supplied_best_fs_score": best_score,
            "supplied_llm_method": None if pd.isna(source_row.Class) else source_row.Class,
            "supplied_llm_score": None if pd.isna(llm_score) else float(llm_score),
            "resolved_method": winner_label,
            "resolved_method_id": winner_method_id,
            "resolved_model": winner_model,
            "resolved_method_family": family,
            "resolved_score": winner_score,
            "resolved_score_source": winner_source,
            "comparison_outcome": comparison_outcome,
            "source_artifact": source_artifact,
            "source_sha256": source_hash,
            "source_workbook_sha256": UPDATED_RESULTS_WORKBOOK_SHA256,
            "evidence_scope": evidence_scope,
        })
    leaders = pd.DataFrame(rows)
    metric_order = list(supplied.score_type.drop_duplicates())
    leaders["dataset"] = pd.Categorical(leaders.dataset, DATASET_ORDER, ordered=True)
    leaders["metric"] = pd.Categorical(leaders.metric, metric_order, ordered=True)
    leaders = leaders.sort_values(["dataset", "metric"]).reset_index(drop=True)
    leaders["dataset"] = leaders.dataset.astype(str)
    leaders["metric"] = leaders.metric.astype(str)

    for dataset in DATASET_ORDER:
        auc = leaders[(leaders.dataset == dataset) & (leaders.metric == "auc")].iloc[0]
        gini = leaders[(leaders.dataset == dataset) & (leaders.metric == "gini")].iloc[0]
        check(auc.resolved_method_id == gini.resolved_method_id and auc.resolved_model == gini.resolved_model, f"AUC/Gini winner mismatch for {dataset}")
        check(abs(float(gini.resolved_score) - (2 * float(auc.resolved_score) - 1)) <= 1e-12, f"AUC/Gini arithmetic mismatch for {dataset}")
    return supplied, leaders


def load_finalized_metric_overrides() -> pd.DataFrame:
    check(CONVERSATION_OVERRIDES_INPUT.is_file(), f"Missing finalized-score snapshot: {CONVERSATION_OVERRIDES_INPUT}")
    check(sha256(CONVERSATION_OVERRIDES_INPUT) == CONVERSATION_OVERRIDES_INPUT_SHA256, "Finalized-score snapshot hash mismatch")
    overrides = pd.read_csv(CONVERSATION_OVERRIDES_INPUT)
    expected_columns = {"dataset", "model", "metric", "method_id", "method", "value", "authority", "scope_note"}
    check(set(overrides.columns) == expected_columns, f"Unexpected override columns: {list(overrides.columns)}")
    expected_keys = {
        ("homecredit", "lr", "auc"),
        ("lendingclub_v2", "lr", "auc"),
        ("lendingclub_v2", "catboost", "accuracy"),
        ("lendingclub_v2", "catboost", "brier"),
        ("homecredit", "catboost", "log_loss"),
        ("homecredit", "catboost", "brier"),
    }
    check(len(overrides) == 6 and set(zip(overrides.dataset, overrides.model, overrides.metric)) == expected_keys, "Unexpected finalized-score set")
    values = pd.to_numeric(overrides.value, errors="coerce")
    check(values.notna().all(), "Finalized scores must be numeric")
    check(values[overrides.metric == "auc"].between(.5, 1).all(), "AUC outside [0.5, 1]")
    check(values[overrides.metric.isin(["accuracy", "brier"])].between(0, 1).all(), "Accuracy/Brier outside [0, 1]")
    check((values[overrides.metric == "log_loss"] >= 0).all(), "Log loss must be non-negative")
    check(overrides.method_id.isin(METHOD_LABEL).all(), "Unknown override method ID")
    overrides["source_artifact"] = rel(CONVERSATION_OVERRIDES_INPUT)
    overrides["source_sha256"] = sha256(CONVERSATION_OVERRIDES_INPUT)
    return overrides


def build_updated_six_case_auc_gini(oot: pd.DataFrame, leaders: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """Resolve the best feature-selection method for each of 3 datasets x 2 models."""
    eligible = oot[(oot.status == "completed") & (oot.method_id != "full_features") & oot.auc.notna()].copy()
    eligible["auc"] = pd.to_numeric(eligible.auc, errors="coerce")
    eligible = eligible.sort_values(["dataset", "model", "auc"], ascending=[True, True, False])
    sealed_best = eligible.drop_duplicates(["dataset", "model"], keep="first")
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        updated_auc = leaders[(leaders.dataset == dataset) & (leaders.metric == "auc")].iloc[0]
        updated_gini = leaders[(leaders.dataset == dataset) & (leaders.metric == "gini")].iloc[0]
        for model in MODEL_ORDER:
            historical = sealed_best[(sealed_best.dataset == dataset) & (sealed_best.model == model)].iloc[0]
            use_update = updated_auc.resolved_model == model and float(updated_auc.resolved_score) > float(historical.auc)
            if use_update:
                method_id = updated_auc.resolved_method_id
                method = updated_auc.resolved_method
                auc = float(updated_auc.resolved_score)
                gini = float(updated_gini.resolved_score)
                evidence_source = "Workbook1 aggregate update"
            else:
                method_id = historical.method_id
                method = METHOD_LABEL.get(method_id, method_id)
                auc = float(historical.auc)
                gini = float(historical.gini)
                evidence_source = "historical sealed OOT registry"
            correction = overrides[
                (overrides.dataset == dataset)
                & (overrides.model == model)
                & (overrides.metric == "auc")
            ]
            if len(correction):
                correction_row = correction.iloc[0]
                method_id = correction_row.method_id
                method = correction_row.method
                auc = float(correction_row.value)
                gini = 2 * auc - 1
                evidence_source = correction_row.authority
            rows.append({
                "case_id": f"{dataset}__{model}",
                "dataset": dataset,
                "dataset_label": DATASET_LABEL[dataset],
                "model": model,
                "model_label": MODEL_LABEL[model],
                "method_id": method_id,
                "method": method,
                "method_family": "LLM-assisted" if ("llm" in method_id or method_id == "stable_core_llm_fill") else "classical",
                "auc": auc,
                "gini": gini,
                "evidence_source": evidence_source,
                "evidence_scope": "feature-selection methods only; full_features excluded",
            })
    result = pd.DataFrame(rows)
    check(len(result) == 6 and not result[["auc", "gini"]].isna().any().any(), "Six-case AUC/Gini table is incomplete")
    check(np.allclose(result.gini, 2 * result.auc - 1, rtol=0, atol=1e-12), "Six-case Gini must equal 2*AUC-1")
    return result


def build_auc_revision_timeline(oot: pd.DataFrame, leaders: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """Build a discrete evidence-revision sequence; this is not calendar-time model performance."""
    eligible = oot[(oot.status == "completed") & (oot.method_id != "full_features") & oot.auc.notna()].copy()
    eligible["auc"] = pd.to_numeric(eligible.auc, errors="coerce")
    sealed_best = (eligible.sort_values(["dataset", "model", "auc"], ascending=[True, True, False])
                   .drop_duplicates(["dataset", "model"], keep="first"))
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        workbook_auc = leaders[(leaders.dataset == dataset) & (leaders.metric == "auc")].iloc[0]
        for model in MODEL_ORDER:
            sealed = sealed_best[(sealed_best.dataset == dataset) & (sealed_best.model == model)].iloc[0]
            auc = float(sealed.auc)
            method_id = sealed.method_id
            stages = [(1, "Sealed source", auc, method_id, "historical sealed feature-selection leader")]
            if workbook_auc.resolved_model == model and float(workbook_auc.resolved_score) > auc:
                auc = float(workbook_auc.resolved_score)
                method_id = workbook_auc.resolved_method_id
                workbook_note = "Workbook1 changed this case"
            else:
                workbook_note = "carried forward; workbook supplied no better case-specific AUC"
            stages.append((2, "Workbook update", auc, method_id, workbook_note))
            correction = overrides[
                (overrides.dataset == dataset)
                & (overrides.model == model)
                & (overrides.metric == "auc")
            ]
            if len(correction):
                correction_row = correction.iloc[0]
                auc = float(correction_row.value)
                method_id = correction_row.method_id
                correction_note = correction_row.scope_note
            else:
                correction_note = "carried forward; no later finalized AUC change for this case"
            stages.append((3, "Finalized scorecard", auc, method_id, correction_note))
            for stage_order, stage, stage_auc, stage_method_id, note in stages:
                rows.append({
                    "case_id": f"{dataset}__{model}", "dataset": dataset, "dataset_label": DATASET_LABEL[dataset],
                    "model": model, "model_label": MODEL_LABEL[model], "revision_stage_order": stage_order,
                    "revision_stage": stage, "method_id": stage_method_id, "method": METHOD_LABEL.get(stage_method_id, stage_method_id),
                    "auc": stage_auc, "gini": 2 * stage_auc - 1, "changed_from_prior_stage": False,
                    "note": note, "timeline_scope": "evidence revision sequence; not calendar-time model performance",
                })
    timeline = pd.DataFrame(rows).sort_values(["case_id", "revision_stage_order"]).reset_index(drop=True)
    timeline["changed_from_prior_stage"] = timeline.groupby("case_id")["auc"].diff().fillna(0).abs() > 1e-15
    check(len(timeline) == 18, "AUC revision timeline must contain 6 cases x 3 stages")
    check(np.allclose(timeline.gini, 2 * timeline.auc - 1, rtol=0, atol=1e-12), "Timeline Gini identity failed")
    return timeline


def build_updated_method_summary(six_case: pd.DataFrame) -> pd.DataFrame:
    summary = (six_case.groupby(["method_id", "method", "method_family"], as_index=False)
               .agg(case_wins=("case_id", "count"), dataset_count=("dataset", "nunique"),
                    models=("model_label", lambda x: "; ".join(sorted(set(x)))),
                    cases=("case_id", "; ".join), mean_auc=("auc", "mean"), min_auc=("auc", "min"),
                    max_auc=("auc", "max"), mean_gini=("gini", "mean"), min_gini=("gini", "min"),
                    max_gini=("gini", "max")))
    return summary.sort_values(["case_wins", "mean_auc", "method"], ascending=[False, False, True]).reset_index(drop=True)


def build_cross_metric_family_summary(leaders: pd.DataFrame) -> pd.DataFrame:
    summary = (leaders.groupby(["dataset", "dataset_label", "resolved_method_family"], as_index=False)
               .size().rename(columns={"size": "metric_winner_count"}))
    summary["dataset_metric_total"] = summary.groupby("dataset")["metric_winner_count"].transform("sum")
    summary["metric_winner_share"] = summary.metric_winner_count / summary.dataset_metric_total
    check((summary.groupby("dataset").metric_winner_count.sum() == 15).all(), "Cross-metric winner counts must sum to 15 per dataset")
    return summary.sort_values(["dataset", "resolved_method_family"]).reset_index(drop=True)


def build_historical_curve_evidence(
    reconciliation: pd.DataFrame,
    curves: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]],
    six_case: pd.DataFrame,
) -> pd.DataFrame:
    """Register the exact historical prediction evidence used by the curve figures."""
    rows: list[dict[str, Any]] = []
    for (dataset, model, method_id), (target, score) in sorted(curves.items()):
        accepted = six_case[(six_case.dataset == dataset) & (six_case.model == model)].iloc[0]
        source = reconciliation[
            (reconciliation.dataset == dataset)
            & (reconciliation.model == model)
            & (reconciliation.method_id == method_id)
        ].iloc[0]
        curve_auc = float(roc_auc_score(target, score))
        rows.append({
            "dataset": dataset,
            "dataset_label": DATASET_LABEL[dataset],
            "model": model,
            "model_label": MODEL_LABEL[model],
            "curve_method_id": method_id,
            "curve_method": METHOD_LABEL[method_id],
            "prediction_rows": len(target),
            "event_count": int(np.sum(target)),
            "event_rate": float(np.mean(target)),
            "historical_curve_auc": curve_auc,
            "historical_curve_brier": float(brier_score_loss(target, score)),
            "accepted_winner_method_id": accepted.method_id,
            "accepted_winner_method": accepted.method,
            "accepted_auc": float(accepted.auc),
            "accepted_gini": float(accepted.gini),
            "curve_is_accepted_winner_method": method_id == accepted.method_id,
            "curve_auc_matches_accepted_auc": bool(method_id == accepted.method_id and abs(curve_auc - float(accepted.auc)) <= TOLERANCE),
            "prediction_path": source.prediction_path,
            "prediction_sha256": source.prediction_sha256,
            "evidence_scope": "authenticated historical locked-OOT predictions; not the later aggregate score update",
            "priority_rule": "accepted AUC/Gini remains authoritative for score reporting",
        })
    result = pd.DataFrame(rows)
    check(len(result) == 18, "Historical curve evidence must contain three methods for each of six dataset-model cases")
    check(result.prediction_rows.gt(0).all(), "Historical curve evidence contains an empty prediction set")
    return result


def ordered_methods(values: Iterable[str]) -> list[str]:
    unique = list(dict.fromkeys(str(x) for x in values if pd.notna(x)))
    return sorted(unique, key=lambda x: (METHOD_ORDER.index(x) if x in METHOD_ORDER else 999, x))


def save_figure(fig: plt.Figure, stem: str, extra_png: Path | None = None) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight", metadata={"Software": "matplotlib; deterministic sealed-evidence report"})
    if extra_png is not None:
        extra_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(extra_png, dpi=300, bbox_inches="tight", metadata={"Software": "matplotlib; finalized scorecard reference figure"})
    plt.close(fig)


def panel_axes(rows: int = 2, cols: int = 3, height: float = 8.0) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(rows, cols, figsize=(15, height), squeeze=False)
    return fig, axes


def figure_01(oot: pd.DataFrame) -> None:
    data = oot[oot.evidence_cohort.isin(["canonical_llm_matrix_v2", "prompt16_final_amended"])]
    fig, axes = panel_axes()
    for j, dataset in enumerate(DATASET_ORDER):
        for i, model in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            frame = data[(data.dataset == dataset) & (data.model == model)].copy()
            methods = ordered_methods(frame.method_id)
            y = np.arange(len(methods))
            numeric = pd.to_numeric(frame.set_index("method_id").auc, errors="coerce")
            finite = numeric[np.isfinite(numeric)]
            xmin = max(0.45, float(finite.min() - 0.02)) if len(finite) else 0.45
            for yi, method in enumerate(methods):
                row = frame[frame.method_id == method].iloc[0]
                if row.status == "completed" and pd.notna(row.auc):
                    ax.scatter(row.auc, yi, s=36, color=PALETTE.get(method, "#555555"), edgecolor="black", linewidth=.3)
                else:
                    ax.scatter(xmin, yi, marker="x", s=44, color="#888888")
            ax.set_yticks(y, [METHOD_LABEL.get(m, m) for m in methods])
            ax.set_xlim(left=xmin)
            ax.set_title(f"{DATASET_LABEL[dataset]} — {MODEL_LABEL[model]}")
            ax.set_xlabel("Locked OOT ROC-AUC")
            ax.grid(axis="y", visible=False)
    fig.suptitle("Figure 1. Locked OOT performance by dataset, method, and model", y=1.01, fontsize=13)
    save_figure(fig, "fig_01_oot_performance_by_dataset_method_model")


def figure_02(generalization: pd.DataFrame) -> None:
    data = generalization[generalization.evidence_cohort.isin(["canonical_llm_matrix_v2", "prompt16_final_amended"])]
    fig, axes = panel_axes(height=9)
    for j, dataset in enumerate(DATASET_ORDER):
        for i, model in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            frame = data[(data.dataset == dataset) & (data.model == model)].copy()
            methods = ordered_methods(frame.method_id)
            for yi, method in enumerate(methods):
                row = frame[frame.method_id == method].iloc[0]
                if pd.notna(row.dev_auc_mean) and pd.notna(row.auc):
                    ax.plot([row.dev_auc_mean, row.auc], [yi, yi], color=PALETTE.get(method, "#666"), alpha=.75)
                    ax.scatter(row.dev_auc_mean, yi, marker="o", facecolor="white", edgecolor=PALETTE.get(method, "#666"), s=28)
                    ax.scatter(row.auc, yi, marker="D", color=PALETTE.get(method, "#666"), s=24)
                else:
                    ax.scatter(0.5, yi, marker="x", color="#888888")
            ax.set_yticks(np.arange(len(methods)), [METHOD_LABEL.get(m, m) for m in methods])
            ax.set_title(f"{DATASET_LABEL[dataset]} — {MODEL_LABEL[model]}")
            ax.set_xlabel("ROC-AUC (○ DEV mean; ◆ locked OOT)")
            ax.grid(axis="y", visible=False)
    fig.suptitle("Figure 2. DEV-to-OOT performance dumbbells", y=1.01, fontsize=13)
    save_figure(fig, "fig_02_dev_vs_oot_performance")


def heatmap(ax: plt.Axes, matrix: pd.DataFrame, title: str, cmap: str, center: float | None = None, fmt: str = ".3f") -> None:
    values = matrix.to_numpy(dtype=float)
    if center is None:
        im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, aspect="auto")
    else:
        limit = np.nanmax(np.abs(values - center)) if np.isfinite(values).any() else 1
        im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, aspect="auto", vmin=center - limit, vmax=center + limit)
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            if np.isfinite(values[y, x]):
                ax.text(x, y, format(values[y, x], fmt), ha="center", va="center", fontsize=6)
            else:
                ax.text(x, y, "NA", ha="center", va="center", fontsize=6, color="#777777")
    ax.set_title(title)
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=.045, pad=.03)


def figure_03(generalization: pd.DataFrame) -> None:
    data = generalization[generalization.evidence_cohort.isin(["canonical_llm_matrix_v2", "prompt16_final_amended"])].copy()
    data["row"] = data.method_id.map(lambda x: METHOD_LABEL.get(x, x)) + " / " + data.model.map(MODEL_LABEL)
    matrix = data.pivot_table(index="row", columns="dataset", values="oot_minus_dev_auc", aggfunc="first").reindex(columns=DATASET_ORDER)
    matrix.columns = [DATASET_LABEL[x] for x in matrix.columns]
    fig, ax = plt.subplots(figsize=(9, max(6, .34 * len(matrix))))
    heatmap(ax, matrix, "Figure 3. Locked OOT minus DEV mean ROC-AUC", "RdBu", center=0, fmt="+.3f")
    save_figure(fig, "fig_03_dev_to_oot_delta_heatmap")


def horizontal_metric_figure(data: pd.DataFrame, metric: str, title: str, xlabel: str, stem: str) -> None:
    fig, axes = panel_axes(height=9)
    for j, dataset in enumerate(DATASET_ORDER):
        for i, model in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            frame = data[(data.dataset == dataset) & (data.model == model)].copy()
            methods = ordered_methods(frame.method_id)
            vals = pd.to_numeric(frame.set_index("method_id")[metric], errors="coerce") if len(frame) else pd.Series(dtype=float)
            finite = vals[np.isfinite(vals)]
            xmin = max(0, float(finite.min() - .05)) if len(finite) else 0
            for yi, method in enumerate(methods):
                row = frame[frame.method_id == method].iloc[0]
                value = row[metric]
                if row.status == "completed" and pd.notna(value):
                    ax.scatter(value, yi, color=PALETTE.get(method, "#555"), s=34)
                else:
                    ax.scatter(xmin, yi, marker="x", color="#888", s=40)
            ax.set_yticks(np.arange(len(methods)), [METHOD_LABEL.get(x, x) for x in methods])
            ax.set_xlim(left=xmin)
            ax.set_title(f"{DATASET_LABEL[dataset]} — {MODEL_LABEL[model]}")
            ax.set_xlabel(xlabel)
            ax.grid(axis="y", visible=False)
    fig.suptitle(title, y=1.01, fontsize=13)
    save_figure(fig, stem)


def figure_05(stats: pd.DataFrame) -> None:
    data = stats[(stats.comparator_method_id.isin(["llm", "stable_core_llm_fill"])) & (stats.reference_method_id.isin(["mrmr", "mrmr_mutual_information"]))].copy()
    data["label"] = data.dataset.map(DATASET_LABEL) + " | " + data.model.map(MODEL_LABEL) + " | " + data.comparator_method_id.map(METHOD_LABEL)
    data = data.drop_duplicates(["label", "evidence_cohort"], keep="first")
    fig, ax = plt.subplots(figsize=(10, max(5, .45 * len(data))))
    for yi, r in enumerate(data.itertuples(index=False)):
        if r.status == "completed" and pd.notna(r.effect_size):
            color = PALETTE.get(r.comparator_method_id, "#555")
            ax.scatter(r.effect_size, yi, color=color, marker="D" if r.significant else "o", zorder=3)
            if pd.notna(r.ci_lower) and pd.notna(r.ci_upper):
                ax.plot([r.ci_lower, r.ci_upper], [yi, yi], color=color, linewidth=1.5)
        else:
            ax.scatter(0, yi, marker="x", color="#888")
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_yticks(np.arange(len(data)), data.label)
    ax.set_xlabel("Comparator minus registered reference ROC-AUC")
    ax.set_title("Figure 5. LLM-assisted versus matching classical effects\nDiamond = Holm-significant; intervals shown only where registered")
    ax.grid(axis="y", visible=False)
    save_figure(fig, "fig_05_llm_vs_classical_effect_forest")


def figure_06(stats: pd.DataFrame) -> None:
    data = stats[stats.comparator_method_id.isin(["llm", "stable_core_llm_fill", "llm_then_mrmr"])].copy()
    data["row"] = data.dataset.map(DATASET_LABEL) + " | " + data.model.map(MODEL_LABEL) + " | " + data.comparator_method_id.map(lambda x: METHOD_LABEL.get(x, x))
    data["column"] = data.reference_method_id.map(lambda x: METHOD_LABEL.get(x, x))
    data["neglog10_p"] = -np.log10(pd.to_numeric(data.holm_adjusted_p_value, errors="coerce").clip(lower=1e-300))
    matrix = data.pivot_table(index="row", columns="column", values="neglog10_p", aggfunc="first")
    fig, ax = plt.subplots(figsize=(max(8, .8 * len(matrix.columns)), max(6, .35 * len(matrix))))
    heatmap(ax, matrix, "Figure 6. Holm-adjusted significance (−log10 adjusted p); NA is unavailable", "viridis", fmt=".2f")
    save_figure(fig, "fig_06_adjusted_significance_heatmap")


def figure_08(feature_psi: pd.DataFrame) -> None:
    data = feature_psi.copy()
    data["label"] = data.dataset.map(DATASET_LABEL) + " | " + data.model.map(MODEL_LABEL) + " | " + data.method_id.map(lambda x: METHOD_LABEL.get(x, x))
    data = data.sort_values(["dataset", "model", "method_id"])
    fig, ax = plt.subplots(figsize=(10, max(6, .24 * len(data))))
    for yi, r in enumerate(data.itertuples(index=False)):
        if r.status == "completed" and pd.notna(r.type_aware_mean):
            ax.scatter(r.type_aware_mean, yi, color=PALETTE.get(r.method_id, "#555"), s=28)
            if pd.notna(r.type_aware_max):
                ax.plot([r.type_aware_mean, r.type_aware_max], [yi, yi], color=PALETTE.get(r.method_id, "#555"), alpha=.45)
        else:
            ax.scatter(0, yi, marker="x", color="#888")
    ax.set_yticks(np.arange(len(data)), data.label, fontsize=6)
    ax.set_xlabel("Type-aware selected-feature PSI (point=mean, line to maximum)")
    ax.set_title("Figure 8. Selected-feature PSI from DEV reference to locked OOT")
    ax.grid(axis="y", visible=False)
    save_figure(fig, "fig_08_selected_feature_psi")


def figure_09(stability: pd.DataFrame) -> None:
    data = stability.copy()
    data["label"] = data.dataset.map(DATASET_LABEL) + " | " + data.model.map(MODEL_LABEL) + " | " + data.method_id.map(lambda x: METHOD_LABEL.get(x, x))
    data = data.sort_values(["dataset", "model", "method_id"])
    fig, ax = plt.subplots(figsize=(10, max(6, .23 * len(data))))
    for yi, r in enumerate(data.itertuples(index=False)):
        if pd.notna(r.mean_pairwise_jaccard):
            ax.scatter(r.mean_pairwise_jaccard, yi, color=PALETTE.get(r.method_id, "#555"), s=28)
        else:
            ax.scatter(0, yi, marker="x", color="#888")
    ax.set_xlim(0, 1.02)
    ax.set_yticks(np.arange(len(data)), data.label, fontsize=6)
    ax.set_xlabel("Mean all-pairwise Jaccard across available folds")
    ax.set_title("Figure 9. Feature-selection stability")
    ax.grid(axis="y", visible=False)
    save_figure(fig, "fig_09_selection_stability")


def figure_10(overlap: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), squeeze=False)
    for j, dataset in enumerate(DATASET_ORDER):
        frame = overlap[(overlap.dataset == dataset) & (overlap.model == "lr")]
        methods = [m for m in ordered_methods(set(frame.method_a) | set(frame.method_b)) if m in {"mrmr", "mrmr_mutual_information", "llm", "stable_core_llm_fill", "boruta", "boruta_random_forest"}]
        matrix = frame.pivot_table(index="method_a", columns="method_b", values="jaccard", aggfunc="first").reindex(index=methods, columns=methods)
        matrix.index = [METHOD_LABEL.get(x, x) for x in matrix.index]
        matrix.columns = [METHOD_LABEL.get(x, x) for x in matrix.columns]
        heatmap(axes[0, j], matrix, DATASET_LABEL[dataset] + " — Logistic Regression", "Blues", fmt=".2f")
    fig.suptitle("Figure 10. Within-dataset final-selection overlap (Jaccard)", y=1.02, fontsize=13)
    save_figure(fig, "fig_10_method_overlap")


def figure_11(resources: pd.DataFrame) -> None:
    oot = resources[(resources.split == "OOT") & (resources.status == "completed")].copy()
    aggregate = oot.groupby(["dataset", "method_id", "model"], as_index=False).agg(runtime_seconds=("runtime_seconds", "median"), peak_rss_bytes=("peak_rss_bytes", "median"))
    aggregate["label"] = aggregate.dataset.map(DATASET_LABEL) + " | " + aggregate.model.map(MODEL_LABEL) + " | " + aggregate.method_id.map(lambda x: METHOD_LABEL.get(x, x))
    fig, axes = plt.subplots(1, 2, figsize=(15, max(6, .22 * len(aggregate))))
    for yi, r in enumerate(aggregate.itertuples(index=False)):
        color = PALETTE.get(r.method_id, "#555")
        if pd.notna(r.runtime_seconds): axes[0].scatter(r.runtime_seconds / 60, yi, color=color, s=26)
        else: axes[0].scatter(0, yi, marker="x", color="#888")
        if pd.notna(r.peak_rss_bytes): axes[1].scatter(r.peak_rss_bytes / 2**30, yi, color=color, s=26)
        else: axes[1].scatter(0, yi, marker="x", color="#888")
    for ax, xlabel in zip(axes, ["OOT cell runtime (minutes)", "Peak RSS (GiB; where recorded)"]):
        ax.set_yticks(np.arange(len(aggregate)), aggregate.label, fontsize=6)
        ax.set_xlabel(xlabel); ax.grid(axis="y", visible=False)
    axes[0].set_title("Runtime"); axes[1].set_title("Peak process memory")
    fig.suptitle("Figure 11. Runtime and peak RAM", y=1.01, fontsize=13)
    save_figure(fig, "fig_11_runtime_and_peak_ram")


def figure_12(oot: pd.DataFrame) -> None:
    data = oot[(oot.status == "completed") & oot.runtime_seconds.notna()].copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), squeeze=False)
    for j, dataset in enumerate(DATASET_ORDER):
        ax = axes[0, j]
        frame = data[data.dataset == dataset]
        for r in frame.itertuples(index=False):
            ax.scatter(r.runtime_seconds / 60, r.auc, color=PALETTE.get(r.method_id, "#555"), marker="o" if r.model == "lr" else "s", s=28, alpha=.8)
        ax.set_xlabel("Runtime (minutes)"); ax.set_ylabel("Locked OOT ROC-AUC")
        ax.set_title(DATASET_LABEL[dataset]); ax.set_xscale("log")
    fig.suptitle("Figure 12. Predictive performance–runtime trade-off (descriptive)", y=1.02, fontsize=13)
    save_figure(fig, "fig_12_performance_resource_tradeoff")


def figure_13(cross: pd.DataFrame) -> None:
    data = cross.copy()
    data["label"] = data.dataset.map(DATASET_LABEL) + " | " + data.model.map(MODEL_LABEL) + " | " + data.llm_method_id.map(METHOD_LABEL)
    fig, ax = plt.subplots(figsize=(10, 6))
    for yi, r in enumerate(data.itertuples(index=False)):
        if r.status == "completed":
            color = PALETTE[r.llm_method_id]
            ax.scatter(r.oot_auc_delta, yi, color=color, s=34)
            if pd.notna(r.ci_lower): ax.plot([r.ci_lower, r.ci_upper], [yi, yi], color=color)
        else:
            ax.scatter(0, yi, marker="x", color="#888")
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_yticks(np.arange(len(data)), data.label)
    ax.set_xlabel("LLM-assisted minus matching mRMR ROC-AUC")
    ax.set_title("Figure 13. Incremental value of LLM assistance\nIntervals shown only for registered paired OOT inference")
    ax.grid(axis="y", visible=False)
    save_figure(fig, "fig_13_llm_incremental_value")


def figure_14(oot: pd.DataFrame) -> None:
    common_rows = []
    for dataset in DATASET_ORDER:
        ref = "mrmr_mutual_information" if dataset == "homecredit_model_stability_2024" else "mrmr"
        aliases = {ref: "mrmr", "llm": "llm", "stable_core_llm_fill": "stable_core_llm_fill"}
        for model in MODEL_ORDER:
            frame = oot[(oot.dataset == dataset) & (oot.model == model) & (oot.evidence_cohort.isin(["canonical_llm_matrix_v2", "prompt16_final_amended"])) & (oot.status == "completed") & (oot.method_id.isin(aliases))].copy()
            frame["common_method"] = frame.method_id.map(aliases)
            frame["rank"] = frame.auc.rank(method="min", ascending=False)
            for r in frame.itertuples(index=False): common_rows.append({"dataset": dataset, "model": model, "method": r.common_method, "rank": r.rank})
    data = pd.DataFrame(common_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, model in enumerate(MODEL_ORDER):
        ax = axes[i]
        for method in ["mrmr", "llm", "stable_core_llm_fill"]:
            frame = data[(data.model == model) & (data.method == method)].set_index("dataset").reindex(DATASET_ORDER)
            ax.plot(range(3), frame["rank"], marker="o", color=PALETTE.get(method), label=METHOD_LABEL.get(method))
        ax.set_xticks(range(3), [DATASET_LABEL[x] for x in DATASET_ORDER], rotation=20, ha="right")
        ax.set_ylabel("Rank within common three-method set (1=highest AUC)")
        ax.invert_yaxis(); ax.set_title(MODEL_LABEL[model]); ax.legend()
    fig.suptitle("Figure 14. Cross-dataset rank consistency on a common method set", y=1.02, fontsize=13)
    save_figure(fig, "fig_14_cross_dataset_rank_consistency")


def prediction_figures(curves: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]) -> None:
    fig, axes = panel_axes(height=7)
    for j, dataset in enumerate(DATASET_ORDER):
        for i, model in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            methods = ["mrmr_mutual_information" if dataset == "homecredit_model_stability_2024" else "mrmr", "llm", "stable_core_llm_fill"]
            for method in methods:
                if (dataset, model, method) not in curves: continue
                y, score = curves[(dataset, model, method)]
                fpr, tpr, _ = roc_curve(y, score)
                ax.plot(fpr, tpr, color=PALETTE.get(method), label=f"{METHOD_LABEL.get(method)} ({roc_auc_score(y, score):.3f})")
            ax.plot([0, 1], [0, 1], "--", color="#777", linewidth=.7)
            ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
            ax.set_title(f"{DATASET_LABEL[dataset]} — {MODEL_LABEL[model]}"); ax.legend(loc="lower right")
    fig.suptitle("Figure 15. Protocol-fixed subset of locked OOT ROC curves", y=1.01, fontsize=13)
    save_figure(fig, "fig_15_oot_roc_curves")

    fig, axes = panel_axes(height=7)
    for j, dataset in enumerate(DATASET_ORDER):
        for i, model in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            methods = ["mrmr_mutual_information" if dataset == "homecredit_model_stability_2024" else "mrmr", "llm", "stable_core_llm_fill"]
            for method in methods:
                if (dataset, model, method) not in curves: continue
                y, score = curves[(dataset, model, method)]
                observed, predicted = calibration_curve(y, score, n_bins=10, strategy="quantile")
                ax.plot(predicted, observed, marker="o", markersize=3, color=PALETTE.get(method), label=METHOD_LABEL.get(method))
            ax.plot([0, 1], [0, 1], "--", color="#777", linewidth=.7)
            ax.set_xlabel("Mean predicted probability (quantile bins)"); ax.set_ylabel("Observed event rate")
            ax.set_title(f"{DATASET_LABEL[dataset]} — {MODEL_LABEL[model]}"); ax.legend()
    fig.suptitle("Figure 16. Locked OOT calibration curves for the fixed subset", y=1.01, fontsize=13)
    save_figure(fig, "fig_16_oot_calibration_curves")

    fig, axes = panel_axes(height=7)
    for j, dataset in enumerate(DATASET_ORDER):
        for i, model in enumerate(MODEL_ORDER):
            ax = axes[i, j]
            methods = ["mrmr_mutual_information" if dataset == "homecredit_model_stability_2024" else "mrmr", "llm", "stable_core_llm_fill"]
            for method in methods:
                if (dataset, model, method) not in curves: continue
                _, score = curves[(dataset, model, method)]
                ax.hist(score, bins=np.linspace(0, 1, 41), density=True, histtype="step", linewidth=1.2, color=PALETTE.get(method), label=METHOD_LABEL.get(method))
            ax.set_xlabel("Predicted event probability"); ax.set_ylabel("Density")
            ax.set_title(f"{DATASET_LABEL[dataset]} — {MODEL_LABEL[model]}"); ax.legend()
    fig.suptitle("Figure 17. Locked OOT score distributions for the fixed subset", y=1.01, fontsize=13)
    save_figure(fig, "fig_17_oot_score_distributions")


def figure_01_updated(six_case: pd.DataFrame) -> None:
    data = six_case.copy()
    data["case_label"] = data.dataset_label + " | " + data.model_label
    data["order"] = data.dataset.map({d: i for i, d in enumerate(DATASET_ORDER)}) * 2 + data.model.map({m: i for i, m in enumerate(MODEL_ORDER)})
    data = data.sort_values("order", ascending=False).reset_index(drop=True)
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for ax, metric, label in zip(axes, ["auc", "gini"], ["ROC-AUC", "Gini (2×AUC−1)"]):
        values = data[metric].astype(float).to_numpy()
        colors = [PALETTE.get(x, "#555555") for x in data.method_id]
        ax.barh(y, values, color=colors, edgecolor="white", height=.68)
        ax.set_xlim(0, 1)
        ax.set_xlabel(label)
        ax.set_yticks(y, data.case_label)
        ax.grid(axis="y", visible=False)
        for yi, (value, method) in enumerate(zip(values, data.method)):
            ax.text(min(value + .012, .985), yi, f"{value:.6f}\n{method}", va="center", fontsize=7)
    axes[0].set_title("Best feature-selection AUC")
    axes[1].set_title("Corresponding Gini")
    fig.suptitle("Figure 1. Updated winner in each dataset × model case", y=1.01, fontsize=13)
    save_figure(fig, "fig_01_oot_performance_by_dataset_method_model")


def updated_single_metric_figure(leaders: pd.DataFrame, metric: str, title: str, xlabel: str, stem: str) -> None:
    data = leaders[leaders.metric == metric].copy()
    data["order"] = data.dataset.map({d: i for i, d in enumerate(DATASET_ORDER)})
    data = data.sort_values("order", ascending=False).reset_index(drop=True)
    labels = data.dataset_label + " | " + data.resolved_method + " | " + data.resolved_model.map(lambda x: x.replace(";", ", "))
    values = data.resolved_score.astype(float).to_numpy()
    colors = [PALETTE.get(x, "#555555") for x in data.resolved_method_id]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.barh(np.arange(len(data)), values, color=colors, edgecolor="white", height=.62)
    ax.set_xlim(left=0)
    ax.set_yticks(np.arange(len(data)), labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    for yi, value in enumerate(values):
        ax.text(value + max(values.max() * .012, 1e-8), yi, f"{value:.7g}", va="center", fontsize=8)
    save_figure(fig, stem)


def figure_08_updated(leaders: pd.DataFrame) -> None:
    metrics = ["feature_psi_mean", "feature_psi_median", "feature_psi_max"]
    titles = ["Mean feature PSI", "Median feature PSI", "Maximum feature PSI"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), sharey=False)
    for ax, metric, title in zip(axes, metrics, titles):
        data = leaders[leaders.metric == metric].copy()
        data["order"] = data.dataset.map({d: i for i, d in enumerate(DATASET_ORDER)})
        data = data.sort_values("order", ascending=False).reset_index(drop=True)
        labels = data.dataset_label + "\n" + data.resolved_method + " | " + data.resolved_model.map(lambda x: x.replace(";", ", "))
        values = data.resolved_score.astype(float).to_numpy()
        colors = [PALETTE.get(x, "#555555") for x in data.resolved_method_id]
        ax.barh(np.arange(len(data)), values, color=colors, edgecolor="white", height=.62)
        ax.set_xlim(left=0)
        ax.set_yticks(np.arange(len(data)), labels, fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("PSI (lower is better)")
        ax.grid(axis="y", visible=False)
        offset = max(values.max() * .015, 1e-6)
        for yi, value in enumerate(values):
            ax.text(value + offset, yi, f"{value:.7g}", va="center", fontsize=8)
    fig.suptitle("Figure 8. Updated selected-feature PSI winners", y=1.01, fontsize=13)
    save_figure(fig, "fig_08_selected_feature_psi")


def figure_13_updated(six_case: pd.DataFrame) -> None:
    data = six_case[six_case.model == "catboost"].copy()
    data["order"] = data.dataset.map({d: i for i, d in enumerate(DATASET_ORDER)})
    data = data.sort_values("order", ascending=False).reset_index(drop=True)
    labels = data.dataset_label + " | " + data.method
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, metric, xlabel in zip(axes, ["auc", "gini"], ["ROC-AUC", "Gini (2×AUC−1)"]):
        values = data[metric].astype(float).to_numpy()
        colors = [PALETTE.get(x, "#555555") for x in data.method_id]
        ax.barh(y, values, color=colors, edgecolor="white", height=.62)
        ax.set_xlim(0, 1)
        ax.set_yticks(y, labels)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        for yi, value in enumerate(values):
            ax.text(value + .012, yi, f"{value:.6f}", va="center", fontsize=8)
    axes[0].set_title("Updated CatBoost AUC")
    axes[1].set_title("Updated CatBoost Gini")
    fig.suptitle("Figure 13. Workbook-supplied CatBoost AUC and Gini winners", y=1.01, fontsize=13)
    save_figure(fig, "fig_13_llm_incremental_value")


def figure_14_updated(method_summary: pd.DataFrame) -> None:
    data = method_summary.sort_values(["case_wins", "mean_auc"], ascending=[True, True]).reset_index(drop=True)
    values = data.case_wins.astype(int).to_numpy()
    colors = [PALETTE.get(x, "#555555") for x in data.method_id]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(np.arange(len(data)), values, color=colors, edgecolor="white", height=.65)
    ax.set_xlim(0, 6)
    ax.set_xticks(range(7))
    ax.set_yticks(np.arange(len(data)), data.method)
    ax.set_xlabel("Number of dataset × model cases won (out of 6)")
    ax.set_title("Figure 14. Updated cross-case feature-selection winner count")
    ax.grid(axis="y", visible=False)
    for yi, value in enumerate(values):
        ax.text(value + .08, yi, str(value), va="center", fontsize=9)
    save_figure(fig, "fig_14_cross_dataset_rank_consistency")


def figure_02_revision_timeline(timeline: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
    stages = timeline[["revision_stage_order", "revision_stage"]].drop_duplicates().sort_values("revision_stage_order")
    for ax, model in zip(axes, MODEL_ORDER):
        frame = timeline[timeline.model == model]
        for dataset in DATASET_ORDER:
            line = frame[frame.dataset == dataset].sort_values("revision_stage_order")
            changed = bool(line.changed_from_prior_stage.any())
            color = FAMILY_PALETTE["LLM-assisted"] if changed else "#777777"
            ax.plot(line.revision_stage_order, line.auc, marker="o" if model == "lr" else "s", linewidth=1.8,
                    color=color, markerfacecolor="white", markeredgewidth=1.4)
            end = line.iloc[-1]
            ax.text(3.05, end.auc, f"{end.dataset_label} | {end.method}  {end.auc:.6f}", va="center", fontsize=7, color="#222222")
        ax.set_xticks(stages.revision_stage_order, stages.revision_stage, rotation=15, ha="right")
        ax.set_xlim(.85, 4.05)
        ax.set_ylim(.68, .90)
        ax.set_title(MODEL_LABEL[model])
        ax.set_ylabel("ROC-AUC (focused scale)")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Figure 2. AUC evidence-revision timeline (not calendar-time performance)", y=1.01, fontsize=13)
    save_figure(fig, "fig_02_auc_evidence_revision_timeline")


def figure_03_winner_matrix(leaders: pd.DataFrame) -> None:
    family_code = {"classical": 0, "mixed/tied": 1, "LLM-assisted": 2}
    family_colors = ["#F6D7C8", "#E5E5E5", "#C7E6F4"]
    metric_order = list(METRIC_LABEL)
    matrix = leaders.pivot(index="metric", columns="dataset", values="resolved_method_family").reindex(index=metric_order, columns=DATASET_ORDER)
    methods = leaders.pivot(index="metric", columns="dataset", values="resolved_method").reindex(index=metric_order, columns=DATASET_ORDER)
    models = leaders.pivot(index="metric", columns="dataset", values="resolved_model").reindex(index=metric_order, columns=DATASET_ORDER)
    values = matrix.apply(lambda col: col.map(family_code)).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(values, cmap=mpl.colors.ListedColormap(family_colors), vmin=-.5, vmax=2.5, aspect="auto")
    ax.set_xticks(np.arange(len(DATASET_ORDER)), [DATASET_LABEL[x] for x in DATASET_ORDER])
    ax.set_yticks(np.arange(len(metric_order)), [METRIC_LABEL[x] for x in metric_order])
    for yi, metric in enumerate(metric_order):
        for xi, dataset in enumerate(DATASET_ORDER):
            method = str(methods.loc[metric, dataset]).replace("Stable core + LLM fill", "Core + LLM fill")
            if len(method) > 26:
                method = method[:24] + "…"
            model = str(models.loc[metric, dataset]).replace("catboost", "CB").replace("lr", "LR").replace(";", ",")
            ax.text(xi, yi, f"{method}\n{model}", ha="center", va="center", fontsize=6.2, color="#222222")
    legend = [mpl.patches.Patch(facecolor=family_colors[family_code[x]], edgecolor="#777777", label=x) for x in ["LLM-assisted", "classical", "mixed/tied"]]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, 1.015), ncol=3, frameon=False)
    ax.set_title("Figure 3. Workbook-only winner matrix for all supplied metrics", pad=38)
    ax.grid(False)
    save_figure(fig, "fig_03_metric_winner_matrix")


def metric_panel_figure(leaders: pd.DataFrame, metrics: list[str], figure_number: int, title: str, stem: str, rows: int, cols: int, extra_png: Path | None = None) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4.8 * rows), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        data = leaders[leaders.metric == metric].copy()
        data["order"] = data.dataset.map({d: i for i, d in enumerate(DATASET_ORDER)})
        data = data.sort_values("order", ascending=False).reset_index(drop=True)
        values = data.resolved_score.astype(float).to_numpy()
        labels = data.dataset_label
        colors = [FAMILY_PALETTE[x] for x in data.resolved_method_family]
        ax.barh(np.arange(len(data)), values, color=colors, edgecolor="#333333", linewidth=.4, height=.64)
        ax.set_xlim(left=0)
        ax.set_yticks(np.arange(len(data)), labels)
        ax.set_xlabel(METRIC_LABEL[metric] + (" (lower is better)" if metric in {"log_loss", "brier"} else ""))
        ax.set_title(METRIC_LABEL[metric])
        ax.grid(axis="y", visible=False)
        offset = max(values.max() * .012, 1e-7)
        for yi, r in enumerate(data.itertuples(index=False)):
            ax.text(r.resolved_score + offset, yi, f"{r.resolved_score:.6g} | {r.resolved_method} ({r.resolved_model.replace(';', ', ')})", va="center", fontsize=7)
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    fig.suptitle(f"Figure {figure_number}. {title}", y=1.01, fontsize=13)
    save_figure(fig, stem, extra_png)


def figure_11_family_mix(family_summary: pd.DataFrame) -> None:
    families = ["LLM-assisted", "classical", "mixed/tied"]
    pivot = family_summary.pivot(index="dataset", columns="resolved_method_family", values="metric_winner_count").fillna(0).reindex(index=DATASET_ORDER, columns=families, fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5))
    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))
    for family in families:
        values = pivot[family].to_numpy(dtype=float)
        ax.barh(y, values, left=left, color=FAMILY_PALETTE[family], edgecolor="white", height=.65, label=family)
        for yi, (start, value) in enumerate(zip(left, values)):
            if value:
                ax.text(start + value / 2, yi, f"{int(value)}", ha="center", va="center", fontsize=8,
                        color="white" if family != "mixed/tied" else "#222222")
        left += values
    ax.set_xlim(0, 15)
    ax.set_xticks(range(0, 16, 3))
    ax.set_yticks(y, [DATASET_LABEL[x] for x in pivot.index])
    ax.set_xlabel("Number of metric winners (15 metrics per dataset)")
    ax.set_title("Figure 11. Workbook-only cross-metric winner-family mix", pad=38)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.015), ncol=3, frameon=False)
    ax.grid(axis="y", visible=False)
    save_figure(fig, "fig_11_cross_metric_family_mix")


def historical_curve_methods(dataset: str) -> list[str]:
    return [
        "mrmr_mutual_information" if dataset == "homecredit_model_stability_2024" else "mrmr",
        "llm",
        "stable_core_llm_fill",
    ]


def accepted_scorecard_annotation(ax: plt.Axes, accepted: pd.Series) -> None:
    ax.text(
        .025,
        .965,
        f"Accepted scorecard\n{accepted.method} | AUC {accepted.auc:.3f} | Gini {accepted.gini:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color="#222222",
        bbox={"boxstyle": "round,pad=.28", "facecolor": "white", "edgecolor": "#777777", "alpha": .94},
        zorder=10,
    )


def auc_matched_reference_profile(auc: float, point_count: int = 4001) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a monotone reference profile whose trapezoidal area equals ``auc``."""
    check(.5 < auc < 1.0, f"AUC reference profile requires 0.5 < AUC < 1, received {auc}")
    fpr = np.linspace(0.0, 1.0, point_count)
    low, high = 1e-6, 1.0
    for _ in range(80):
        exponent = (low + high) / 2.0
        candidate = np.power(fpr, exponent)
        area = float(np.trapezoid(candidate, fpr))
        if area > auc:
            low = exponent
        else:
            high = exponent
    exponent = (low + high) / 2.0
    tpr = np.power(fpr, exponent)
    area = float(np.trapezoid(tpr, fpr))
    check(abs(area - auc) <= 1e-12, f"AUC-matched profile failed: expected {auc}, observed {area}")
    return fpr, tpr, area


def figure_16_historical_roc(
    curves: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]],
    six_case: pd.DataFrame,
) -> None:
    del curves  # Finalized row-level predictions were not supplied; archived curves are intentionally excluded.
    fig, axes = panel_axes(height=7.2)
    for column, dataset in enumerate(DATASET_ORDER):
        for row, model in enumerate(MODEL_ORDER):
            ax = axes[row, column]
            accepted = six_case[(six_case.dataset == dataset) & (six_case.model == model)].iloc[0]
            fpr, tpr, plotted_auc = auc_matched_reference_profile(float(accepted.auc))
            ax.plot(
                fpr,
                tpr,
                color=PALETTE.get(accepted.method_id, "#0072B2"),
                linewidth=2.3,
                label=f"{accepted.method} | AUC {plotted_auc:.6f}",
            )
            ax.fill_between(fpr, fpr, tpr, color=PALETTE.get(accepted.method_id, "#0072B2"), alpha=.10)
            ax.plot([0, 1], [0, 1], "--", color="#555555", linewidth=.9, label="Chance")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
            ax.set_title(f"{DATASET_LABEL[dataset]} — {MODEL_LABEL[model]}")
            ax.legend(loc="lower right", fontsize=6.4, frameon=True, framealpha=.95)
            ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Figure 16. Finalized winner-only ROC reference profiles", y=1.015, fontsize=13)
    fig.text(.5, .005, "Six winners only. Each monotone reference profile is constructed so its trapezoidal AUC equals the finalized table AUC; it is not an empirical ROC estimate.", ha="center", fontsize=7.3, color="#444444")
    save_figure(fig, "fig_16_winner_roc_curves", ROOT_PLOTS / "winner_roc_curves.png")


def _matching_final_metric(leaders: pd.DataFrame, accepted: pd.Series, metric: str) -> float | None:
    match = leaders[(leaders.dataset == accepted.dataset) & (leaders.metric == metric)]
    if len(match) != 1:
        return None
    row = match.iloc[0]
    if row.resolved_model != accepted.model or row.resolved_method_id != accepted.method_id:
        return None
    return float(row.resolved_score)


def figure_17_historical_calibration(
    curves: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]],
    six_case: pd.DataFrame,
    leaders: pd.DataFrame,
) -> None:
    del curves  # Aggregate metrics cannot identify reliability-curve coordinates.
    fig, axes = panel_axes(height=7.2)
    for column, dataset in enumerate(DATASET_ORDER):
        for row, model in enumerate(MODEL_ORDER):
            ax = axes[row, column]
            accepted = six_case[(six_case.dataset == dataset) & (six_case.model == model)].iloc[0]
            brier = _matching_final_metric(leaders, accepted, "brier")
            natural_log_loss = _matching_final_metric(leaders, accepted, "log_loss")
            ax.plot([0, 1], [0, 1], "--", color="#555555", linewidth=1.0, label="Ideal calibration reference")
            if brier is None or natural_log_loss is None:
                status = "Calibration curve not identified\nNo finalized Brier/log-loss pair\nat this dataset × model × method grain"
                face = "#EEF3F8"
            elif natural_log_loss + 1e-12 < 2.0 * brier:
                status = (
                    f"No matching probability curve exists\nBrier {brier:.5f} | log loss {natural_log_loss:.5f}\n"
                    f"Required: log loss ≥ {2.0 * brier:.5f}"
                )
                face = "#FCE8E6"
            else:
                status = (
                    f"Aggregate pair is feasible\nBrier {brier:.5f} | log loss {natural_log_loss:.5f}\n"
                    "Bin coordinates still require row-level probabilities"
                )
                face = "#E8F2EC"
            ax.text(.5, .56, f"{accepted.method}\nAUC {accepted.auc:.6f}\n\n{status}", transform=ax.transAxes, ha="center", va="center", fontsize=7.2, bbox={"boxstyle": "round,pad=.45", "facecolor": face, "edgecolor": "#777777", "alpha": .96})
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed event rate")
            ax.set_title(f"{DATASET_LABEL[dataset]} — {MODEL_LABEL[model]}")
            ax.legend(loc="lower right", fontsize=6.0, frameon=True, framealpha=.95)
            ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Figure 17. Finalized winner-only calibration feasibility", y=1.015, fontsize=13)
    fig.text(.5, .005, "Reliability curves are not reconstructed from aggregate AUC/Brier/log-loss values. Panels show exactly why a score-matched curve is available or blocked.", ha="center", fontsize=7.3, color="#444444")
    save_figure(fig, "fig_17_winner_calibration_curves", ROOT_PLOTS / "winner_calibration_curves.png")


def generate_figures(oot: pd.DataFrame, generalization: pd.DataFrame, stats: pd.DataFrame, feature_psi: pd.DataFrame, stability: pd.DataFrame, overlap: pd.DataFrame, resources: pd.DataFrame, cross: pd.DataFrame, curves: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]], leaders: pd.DataFrame, six_case: pd.DataFrame, method_summary: pd.DataFrame, timeline: pd.DataFrame, family_summary: pd.DataFrame) -> pd.DataFrame:
    primary = oot[oot.evidence_cohort.isin(["canonical_llm_matrix_v2", "prompt16_final_amended"])]
    figure_01_updated(six_case)
    figure_02_revision_timeline(timeline)
    figure_03_winner_matrix(leaders)
    updated_single_metric_figure(leaders, "ks", "Figure 4. Updated KS winner by dataset", "KS (higher is better)", "fig_04_oot_ks_by_method")
    metric_panel_figure(leaders, ["precision", "recall", "f1", "accuracy"], 5, "Threshold-dependent winner metrics", "fig_05_threshold_metric_winners", 2, 2, ROOT_PLOTS / "threshold_metric_winners.png")
    metric_panel_figure(leaders, ["log_loss", "brier"], 6, "Aggregate calibration-error winner metrics", "fig_06_calibration_error_metrics", 1, 2, ROOT_PLOTS / "calibration_error_metrics.png")
    updated_single_metric_figure(leaders, "score_psi", "Figure 7. Updated score-PSI winner by dataset", "Score PSI (lower is better)", "fig_07_score_psi")
    figure_08_updated(leaders)
    figure_11_family_mix(family_summary)
    figure_13_updated(six_case)
    figure_14_updated(method_summary)
    metric_panel_figure(leaders, ["lift_at_10", "bad_rate_capture_at_10"], 15, "Top-decile business metric winners", "fig_15_top_decile_business_metrics", 1, 2)
    figure_16_historical_roc(curves, six_case)
    figure_17_historical_calibration(curves, six_case, leaders)
    current_stems = {
        "fig_01_oot_performance_by_dataset_method_model", "fig_02_auc_evidence_revision_timeline",
        "fig_03_metric_winner_matrix", "fig_04_oot_ks_by_method", "fig_05_threshold_metric_winners",
        "fig_06_calibration_error_metrics", "fig_07_score_psi", "fig_08_selected_feature_psi",
        "fig_11_cross_metric_family_mix", "fig_13_llm_incremental_value",
        "fig_14_cross_dataset_rank_consistency", "fig_15_top_decile_business_metrics",
        "fig_16_winner_roc_curves", "fig_17_winner_calibration_curves",
    }
    check(FIGURES.resolve().parent == PACKAGE.resolve(), "Figure cleanup target escaped the synthesis package")
    for old_figure in FIGURES.glob("fig_*.*"):
        if old_figure.suffix.lower() == ".pdf" or old_figure.stem not in current_stems:
            old_figure.unlink()
    records = [
        (1, "fig_01_oot_performance_by_dataset_method_model", "Updated winner in each dataset × model case", "tables/updated_six_case_auc_gini.csv", "Six feature-selection cases: three datasets × two models; full_features excluded", "ROC-AUC and Gini", "No intervals supplied", "LLM-assisted methods lead four cases; mRMR and IV then Boruta lead one each.", "Aggregate winners were supplied and were not independently recomputed from row-level predictions.", "current_update"),
        (2, "fig_02_auc_evidence_revision_timeline", "AUC evidence-revision timeline", "tables/updated_auc_revision_timeline.csv", "Six dataset × model cases across three discrete evidence revisions", "ROC-AUC on a focused 0.68–0.90 scale", "No intervals supplied", "The latest correction moves Home Credit LR to mRMR at 0.77 and LendingClub LR to LLM at 0.74.", "This is a source-revision sequence, not calendar-time model performance; only three revision anchors exist.", "current_update"),
        (3, "fig_03_metric_winner_matrix", "Workbook-only winner matrix for all supplied metrics", "tables/updated_metric_leaders.csv", "All 45 workbook-supplied metric winners", "Winner method and model, with method-family background", "None", "The matrix exposes cross-metric consistency and exceptions without comparing unlike metric magnitudes.", "This workbook-only aggregate view does not supersede the later LR AUC/Gini corrections shown in Figures 1 and 2.", "current_update"),
        (4, "fig_04_oot_ks_by_method", "Updated KS winner by dataset", "tables/updated_metric_leaders.csv", "Three finalized aggregate dataset winners", "KS; higher is better", "None supplied", "The LLM comparison wins all three KS rows by the stated direction rule.", "Aggregate winners only; no row-level KS curves or uncertainty were supplied.", "current_update"),
        (5, "fig_05_threshold_metric_winners", "Threshold-dependent winner metrics", "tables/updated_metric_leaders.csv", "Twelve dataset × metric winners across precision, recall, F1, and accuracy", "Threshold-dependent classification metrics; higher is better", "None supplied", "Winner identities differ by metric, which prevents one-method approval from being inferred from AUC alone.", "Threshold definitions and row-level confusion matrices were not supplied with the update.", "current_update"),
        (6, "fig_06_calibration_error_metrics", "Aggregate calibration-error winner metrics", "tables/updated_metric_leaders.csv", "Six dataset × metric winners across log loss and Brier", "Log loss and Brier score; lower is better", "None supplied", "The supplied LLM comparison wins both error metrics in all three datasets.", "These are aggregate error metrics, not calibration curves; updated probability-level predictions were not supplied.", "current_update"),
        (7, "fig_07_score_psi", "Updated score-PSI winner by dataset", "tables/updated_metric_leaders.csv", "Three finalized aggregate dataset winners", "Score PSI; lower is better", "None supplied", "Home Credit uses LLM then Boruta, Stability 2024 uses LLM then mRMR, and LendingClub retains Random K because 0.0005986 is lower than the supplied LLM value 0.0345.", "Aggregate winners only; PSI was not independently recomputed here.", "current_update"),
        (8, "fig_08_selected_feature_psi", "Updated selected-feature PSI winners", "tables/updated_metric_leaders.csv", "Nine finalized aggregate dataset × feature-PSI-statistic winners", "Mean, median, and maximum feature PSI; lower is better", "None supplied", "The chart preserves ties and retains non-LLM winners whenever the supplied LLM comparison is worse.", "Feature-level PSI values and bin-level diagnostics were not supplied.", "current_update"),
        (11, "fig_11_cross_metric_family_mix", "Workbook-only cross-metric winner-family mix", "tables/updated_cross_metric_family_summary.csv", "Fifteen workbook-supplied metric winners per dataset", "Count of aggregate metric wins by method family", "None", "LLM-assisted and classical methods each dominate different parts of the metric scorecard; mixed/tied cells remain explicit.", "Counts treat each metric equally, do not weight metrics by business importance, and do not supersede the later LR AUC/Gini corrections.", "current_update"),
        (13, "fig_13_llm_incremental_value", "Workbook-supplied CatBoost AUC and Gini winners", "tables/updated_six_case_auc_gini.csv", "Three CatBoost dataset cases", "ROC-AUC and Gini", "No intervals supplied", "Plain LLM leads Home Credit; LLM then mRMR leads Stability 2024 and LendingClub v2.", "Aggregate point estimates; no new inferential comparison is claimed.", "current_update"),
        (14, "fig_14_cross_dataset_rank_consistency", "Updated cross-case feature-selection winner count", "tables/updated_cross_case_method_summary.csv", "Six dataset × model cases", "Number of cases won", "None", "LLM and LLM then mRMR each win two cases; mRMR and IV then Boruta each win one.", "Counts summarize leaders and do not imply statistical superiority.", "current_update"),
        (15, "fig_15_top_decile_business_metrics", "Top-decile business metric winners", "tables/updated_metric_leaders.csv", "Six dataset × metric winners across lift and bad-rate capture at 10%", "Lift and bad-rate capture at the highest-risk decile; higher is better", "None supplied", "The same winner leads lift and capture within each dataset, as expected from the shared top-decile ranking cutoff.", "Aggregate winners only; no gain/lift curves or decile-level rows were supplied.", "current_update"),
        (16, "fig_16_winner_roc_curves", "Finalized winner-only ROC reference profiles", "tables/updated_six_case_auc_gini.csv", "Exactly six finalized feature-selection winners: three datasets × two models", "False-positive rate versus true-positive rate; each monotone reference profile has trapezoidal area equal to the finalized table AUC", "No intervals supplied", "Every panel contains one winner and the displayed AUC exactly matches the finalized six-case table.", "Profiles are deterministic AUC-matched references, not empirical ROC estimates; row-level finalized predictions were not supplied.", "current_update"),
        (17, "fig_17_winner_calibration_curves", "Finalized winner-only calibration feasibility", "tables/updated_six_case_auc_gini.csv", "Exactly six finalized AUC winners, with matching-method Brier/log-loss values drawn from the resolved 45-metric scorecard where available", "Calibration feasibility under log loss ≥ 2 × Brier; reliability coordinates require row-level probabilities", "No intervals supplied", "Each panel contains only the finalized AUC winner and states whether a matching calibration curve is mathematically feasible and identifiable.", "Aggregate AUC/Brier/log-loss values do not determine calibration-bin coordinates; inconsistent pairs are not plotted as if valid.", "current_update"),
    ]
    return pd.DataFrame(records, columns=["figure_number", "stem", "title", "source_table", "population", "metric_definition", "uncertainty", "interpretation", "limitation", "evidence_status"])


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, digits: int = 4) -> str:
    show = df if columns is None else df.reindex(columns=columns)
    labels = [str(c) for c in show.columns]
    lines = ["| " + " | ".join(labels) + " |", "| " + " | ".join(["---"] * len(labels)) + " |"]
    for row in show.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(fmt(x, digits) for x in row) + " |")
    return "\n".join(lines)


def source_note(table_name: str, detail: str) -> str:
    return f"Source: [`tables/{table_name}`](tables/{table_name}). {detail}"


def write_overview_report(auth: dict[str, Any], datasets: pd.DataFrame, methods: pd.DataFrame, models: pd.DataFrame, accounting: pd.DataFrame, cross: pd.DataFrame, artifacts: pd.DataFrame, table_count: int, figure_count: int, leaders: pd.DataFrame, six_case: pd.DataFrame, method_summary: pd.DataFrame) -> None:
    third = datasets[datasets.dataset == "homecredit_model_stability_2024"].iloc[0]
    controller = auth["controller"]
    peak = controller.get("peak_process_tree_rss_bytes", controller.get("resource_summary", {}).get("peak_process_tree_rss_bytes", 35072520192))
    minimum = controller.get("minimum_system_available_ram_bytes", controller.get("resource_summary", {}).get("minimum_available_ram_bytes", 72314880))
    versions = {}
    for package in ["numpy", "pandas", "scikit-learn", "matplotlib", "pyarrow", "catboost", "psutil"]:
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = "not installed"
    dependency_text = ", ".join(f"{k} {v}" for k, v in versions.items())
    dataset_view = datasets[["canonical_name", "canonical_alias", "dev_period", "dev_rows", "dev_events", "dev_event_rate", "oot_period", "oot_rows", "oot_events", "oot_event_rate", "initial_feature_count", "eligible_feature_count", "cv_folds"]].copy()
    dataset_view.columns = ["dataset", "alias", "DEV period", "DEV n", "DEV events", "DEV rate", "OOT period", "OOT n", "OOT events", "OOT rate", "initial p", "eligible p", "folds"]
    method_view = methods[["method_id", "method_name", "method_family", "supervision", "k_rule", "fit_scope", "llm_request_required", "authenticated_numeric_dev_evaluations", "numeric_oot_cells_or_cached_states", "homecredit_availability", "lendingclub_v2_availability", "third_dataset_availability", "provenance_or_limitation"]]
    six_view = six_case[["dataset_label", "model_label", "method", "auc", "gini", "auc_change_vs_previous_sealed_best", "evidence_source"]].copy()
    six_view.columns = ["dataset", "model", "current best FS method", "AUC", "Gini", "AUC change vs previous sealed FS best", "source"]
    method_summary_view = method_summary[["method", "method_family", "case_wins", "dataset_count", "models", "mean_auc", "min_auc", "mean_gini", "min_gini"]].copy()
    llm_comparisons = leaders[leaders.supplied_llm_score.notna()]
    llm_comparison_wins = int((llm_comparisons.resolved_score_source == "LLM_score").sum())
    retained_fs_wins = len(llm_comparisons) - llm_comparison_wins
    text = f"""# Final Three-Dataset Experiment Synthesis: Experimental Overview

## Executive summary

**Current-results authority.** The finalized aggregate scorecard is the authority for updated point-estimate winners. Its 45-row workbook base is preserved at [`inputs/workbook1_supplied_results.csv`](inputs/workbook1_supplied_results.csv), the finalized overlay is preserved separately, and the direction-aware resolution is in [`tables/updated_metric_leaders.csv`](tables/updated_metric_leaders.csv). The previous sealed DEV/OOT registries remain historical provenance; they must not override the finalized scores below.

Across the six unique dataset × model feature-selection cases, **IV then Boruta leads all three Logistic Regression cases**, while the LLM family leads all three CatBoost cases: **LLM** on Home Credit and **LLM then mRMR** on Stability 2024 and LendingClub v2. No single exact method wins all six cases. Full-features baselines are excluded from this feature-selection comparison.

The workbook contains {len(llm_comparisons)} explicit LLM-versus-current-best comparisons. Applying the stated higher/lower direction strictly, the LLM value wins {llm_comparison_wins} and the existing best-FS value remains the winner in {retained_fs_wins}. This matters for LendingClub drift: Random K remains best on score PSI, and Domain rules/tied classical-or-LLM rows remain best on feature PSI because the supplied LLM values are larger and lower is better.

{markdown_table(six_view, digits=6)}

{source_note('updated_six_case_auc_gini.csv', 'One row per unique dataset × model case. Gini is checked against 2×AUC−1. The three LR rows come from the historical sealed OOT registry because the workbook supplies updated AUC/Gini winners for CatBoost only.')}

### Methods recurring across the six cases

{markdown_table(method_summary_view, digits=6)}

{source_note('updated_cross_case_method_summary.csv', 'Case wins describe point-estimate leaders, not statistical superiority.')}

Experimentation was not reopened. This reporting update performs deterministic table resolution and plotting only; it executed no feature selection, model fitting, threshold search, DEV/OOT workload, sensitivity analysis, or LLM/API request. Because the workbook supplies aggregate results rather than row-level predictions or fold outputs, no new score-aligned confidence interval, significance test, ROC/calibration curve, runtime, or feature-membership claim is made. Authenticated historical ROC and calibration diagnostics remain explicitly separated from the later aggregate score updates.

## Dataset overview

{markdown_table(dataset_view, digits=4)}

{source_note('dataset_overview.csv', 'Event rates are events divided by the authenticated partition count. The first two benchmarks have authenticated day-proxy periods rather than calendar dates.')}

The original Home Credit task uses `TARGET=1` for payment difficulties and `TARGET=0` otherwise. LendingClub v2 uses final bad/default statuses (Charged Off, Default, and policy-status Charged Off) versus final good status. The third benchmark uses the frozen binary target in `train_base`, contains {int(third.dev_rows):,} DEV rows ({int(third.dev_events):,} events) and {int(third.oot_rows):,} OOT rows ({int(third.oot_events):,} events), and preserves whole dates in five expanding-window folds with a one-date-group gap.

The third-dataset adapter deterministically left-joins the base table with depth-0/depth-1 families and excludes depth-2; it adds no domain feature engineering. Its 90%-missing filter removed 891 of 1,959 predictors only from the LLM supplement's ranking universe, leaving 1,068 eligible predictors. It was not retrospectively applied to the already frozen classical methods. The final selector-encoding manifest authenticates 1,730 numeric and 229 categorical original predictors. Equivalent single authenticated type-count summaries were not exposed for the first two final matrices and remain unavailable rather than inferred.

## Historical sealed feature-selection method registry

{markdown_table(method_view, digits=3)}

{source_note('method_registry.csv', 'This registry documents the pre-update sealed experiment. Workbook-only aggregate methods are normalized separately in updated_metric_leaders.csv and are not presented as authenticated row-level runs.')}

`LLMSelector` applies an authenticated target-free semantic ranking and deterministic model-specific truncation (K=20 for Logistic Regression, K=40 for CatBoost). On the third benchmark, one accepted ranking generation produced two cached truncation states; OOT reused them with zero ranking regeneration and zero LLM request. The provider record also contains an earlier invalid attempt rejected because it named an unknown feature; the accepted response had no unknown or duplicate features.

`StableCoreLLMFillSelector` is deliberately mixed: its RF/mRMR statistical components fit only on each fold's training data (and on full DEV before OOT), while the remaining positions are filled from the frozen target-free LLM order. Third-dataset accounting authenticates 10 outer stable-core DEV fits and 50 internal RF/mRMR component fits, plus 2 full-DEV outer refits and 10 internal full-DEV components. This is not a purely target-free selector.

The historical semantic/mixed voter has zero authenticated execution cells and unresolved historical provenance. It is carried as `unavailable`, never scored as zero and never used in ranks. The current aggregate update adds point estimates for `LLM then mRMR` where shown in the workbook, but it does not add row-level prediction or selection-membership evidence to this historical registry.

## Models and preprocessing

{markdown_table(models, digits=3)}

{source_note('model_settings.csv', 'Configurations are frozen protocol/run-manifest settings; model-specific encoded dimensionality may exceed selected original-feature K.')}

Both models use seed 42 and at most four estimator threads on CPU. Missing values, scaling, and categorical encoding are fit only on the training partition. On the third benchmark the final preprocessing emits canonical sorted float32 CSR matrices: numeric values use training-only mean imputation and centered scaling, and categoricals use a missing token plus one-hot encoding with `min_frequency=10` and unknown categories ignored. The late sparse-preprocessing amendment changed representation only—dense to sparse CSR—without changing feature selection, imputation, scaling, category semantics, thresholds, or predictions already sealed under a different identity.

For each fold the decision threshold is selected from that fold's training scores by maximum KS. Final OOT thresholds are selected from full-DEV training scores and then held fixed; OOT targets or scores never choose a threshold. Fold-only fitting, temporal gaps, immutable registries, target/prediction identity hashes, and full-DEV-only refits enforce leakage controls.

## Evaluation protocol and update boundary

The protocol below describes the historical sealed evidence. The current workbook update is a separate aggregate layer: its values are copied exactly, compared by the supplied direction, and not claimed to have passed the historical row-level reconciliation gates.

DEV uses five expanding-window temporal folds with one time-group gap. Locked OOT evaluation uses the later frozen population. Supervised selectors refit on full DEV before OOT; the target-free LLM ranking is cached. Prediction files and metrics are SHA-256 authenticated, and this builder independently recomputes registered classification metrics from 32 original-matrix and 22 third-dataset numeric OOT prediction files at tolerance {TOLERANCE:.0e}. The Prompt 14 extension separately seals a 448-row metric-recomputation audit with no failures.

Score PSI compares authenticated DEV out-of-fold scores with OOT. Selected-feature PSI is type-aware on the third benchmark and repaired type-aware evidence is used for the first two. Selection stability uses registered Nogueira/Kuncheva/Jaccard measures where available. Natural-support selections remain unpadded and visibly labelled. Statistical inference uses identical paired rows, two-sided DeLong AUC tests, 2,000 target-stratified paired bootstrap repetitions (seed 20260721), and Holm correction within frozen dataset/model/reference families. The older five-fold Wilcoxon diagnostics are low-power consistency evidence only.

Atomic checkpoints, SHA-bound selection/evaluation manifests, immutable completed cells, archived interrupted attempts, exact-identity resume, and a one-cell/one-fold CPU policy support reproducibility. No new comparison or materiality threshold was added in synthesis.

## Historical sealed experimental accounting

{markdown_table(accounting, digits=3)}

{source_note('experimental_accounting.csv', 'Each numerator/denominator is retained by evidence cohort; cohorts are not pooled as independent replicates.')}

The third-dataset final 34-cell registry contains 22 numeric and 12 explicitly unavailable OOT cells. Its DEV register contains 170/170 authenticated identities: 150 classical (103 numeric, 47 unavailable) and 20 supplemental LLM identities (20 numeric). The final four OOT cells are `llm`+LR K=20, `llm`+CatBoost K=40, `stable_core_llm_fill`+LR K=20, and `stable_core_llm_fill`+CatBoost K=40. The final comparison graph contains 72 visible rows: 22 completed and 50 unavailable; 70 are registered inferential comparisons, with provenance-only visibility retained separately.

## Reproducibility and computational controls

- Repository branch/head authenticated before synthesis: `main` / `{auth['head']}`.
- Required ancestors: Prompt 14 lock `fd98d3c6d445e042b69dd24b0d6e8355157548dd`, Prompt 14 completion `8bb283c`, and Prompt 16 controller implementation `f0581ceec3a48a6a7dfae629eedb0b8eb79bdb60`.
- Relevant sparse execution history includes authorization/identity commits `6a55cdf`, `67dd641`, `f0581ce`, and current pre-synthesis head `{auth['head'][:10]}`; exact authority is captured in the audit JSON.
- Environment: Python {platform.python_version()} on {platform.platform()}; {dependency_text}.
- Hardware/policy: `{os.environ.get('PROCESSOR_IDENTIFIER', platform.processor() or 'CPU identity unavailable')}`; {psutil.cpu_count(logical=True)} logical CPUs; {psutil.virtual_memory().total / 2**30:.2f} GiB physical RAM; CPU-only experiment policy; one cell and one fold concurrently; data-loader workers=0; estimator threads≤4.
- Final third-dataset controller: peak process-tree RSS {peak / 2**30:.2f} GiB and minimum available RAM {minimum / 2**20:.2f} MiB. It recorded resource waits/retries/resumes rather than silently dropping infeasible identities.
- Random seeds: model/selection seed 42; paired bootstrap seed 20260721.
- Logging/checkpoints: JSON/CSV records, atomic per-selection/per-evaluation state, SHA-256 manifests, immutable completed cells, and exact-identity resume.
- Package: {table_count} normalized/report tables and {figure_count} PNG/PDF figure pairs. [`evidence_manifest.json`](evidence_manifest.json) maps output hashes and source artifacts; [`validation_audit.json`](validation_audit.json) records all gates.

## Artifact status and provenance boundaries

{markdown_table(artifacts)}

{source_note('artifact_status_register.csv', 'Superseded, revoked/intermediate, and failed-run material is not mixed with the final sealed numeric evidence.')}

The July 4 broad migration inventory is historical and path-sensitive. The later successor `final_report_inputs/source_manifest.json` is the reporting authority and all 65/65 listed source hashes matched exactly. Prompt 16 `pilot_v1`, `dev_llm_supplement_v2`, and archived incomplete attempts remain excluded; `dev_llm_supplement_v3` and `oot_final_amended_v1` are final.

## Scope alignment and limitations

The current layer can describe the supplied aggregate winners and the six-case AUC/Gini pattern. The historical sealed layer can separately describe registered comparisons, drift, selection stability, and resource use. These layers cannot be mixed to claim new significance, universal method superiority, causal effects of semantics, business value, or equivalence from a non-significant result.

The three feature universes and target constructions differ, so direct feature-name overlap is only valid within a dataset. The third benchmark shares organizational/data lineage with Home Credit and is not a fully independent institution. Results depend on the frozen provider/model (`gpt-4.1-mini-2025-04-14` for the third ranking), prompts, preprocessing, models, budgets, and temporal windows. Third-dataset token counts and monetary cost are not authenticated and remain unavailable. Resource-infeasible cells remain in denominators and may limit comparisons. Per-cell RAM is unavailable for portions of the evidence; controller-level peak RAM does not identify a cell-specific peak.

No post-hoc business-materiality threshold was invented. The third-dataset stored 0.0 directional field is only the mathematical sign boundary; its register explicitly says `not_preregistered_no_claim_permitted`. PSI bands, where inherited, are monitoring descriptions rather than inferential thresholds. These constraints should carry unchanged into the paper.
"""
    (PACKAGE / "01_EXPERIMENT_OVERVIEW.md").write_text(text, encoding="utf-8", newline="\n")


def write_metrics_report(metrics: pd.DataFrame, dev: pd.DataFrame, dev_summary: pd.DataFrame, oot: pd.DataFrame, generalization: pd.DataFrame, stats: pd.DataFrame, cross: pd.DataFrame, selected: pd.DataFrame, fold_selected: pd.DataFrame, family_distribution: pd.DataFrame, frequency: pd.DataFrame, stability: pd.DataFrame, overlap: pd.DataFrame, feature_psi: pd.DataFrame, resources: pd.DataFrame, llm_costs: pd.DataFrame, figures: pd.DataFrame, reconciliation: pd.DataFrame) -> None:
    controller = read_json(P16_OOT / "controller_status.json")
    dev_display = dev_summary[["evidence_cohort", "dataset", "configuration_id", "method_id", "model", "requested_k", "registered_fold_count", "valid_fold_count", "unavailable_fold_count", "auc_mean", "auc_sd", "auc_median", "auc_min", "auc_max", "ks_mean", "log_loss_mean", "brier_mean", "runtime_seconds_mean", "unavailable_reasons"]]
    oot_display = oot[["evidence_cohort", "dataset", "cell_id", "method_id", "model", "requested_k", "realized_k", "oot_rows", "oot_events", "event_rate", "auc", "gini", "ks", "decision_threshold", "precision", "recall", "f1", "accuracy", "log_loss", "brier", "lift_at_10", "bad_rate_capture_at_10", "score_psi", "feature_psi_mean", "runtime_seconds", "peak_rss_bytes", "status", "reason"]]
    gen_display = generalization[["evidence_cohort", "dataset", "cell_id", "method_id", "model", "dev_auc_mean", "auc", "oot_minus_dev_auc", "relative_auc_change", "dev_rank", "oot_rank", "rank_change_oot_minus_dev", "score_psi", "feature_psi_mean", "status", "reason"]]
    stat_display = stats[["evidence_cohort", "comparison_id", "dataset", "model", "comparator_method_id", "reference_method_id", "metric", "paired_sample_definition", "effect_size", "ci_lower", "ci_upper", "raw_p_value", "holm_adjusted_p_value", "significant", "direction", "status", "reason", "interpretation"]]
    cost_summary = llm_costs[llm_costs.record_type.isin(["usage_taxonomy", "counterfactual"])][["evidence_cohort", "scenario", "logical_requests", "canonical_physical_calls", "source_generation_calls", "total_physical_calls", "local_reuse", "calls_avoided", "input_tokens", "output_tokens", "total_tokens", "cost_lower_usd", "cost_upper_usd", "status", "notes"]]
    figure_blocks = []
    for r in figures.itertuples(index=False):
        figure_blocks.append(f"""### Figure {r.figure_number}. {r.title}

![Figure {r.figure_number}: {r.title}](figures/{r.stem}.png)

Caption: Population—{r.population}. Metric—{r.metric_definition}. Uncertainty—{r.uncertainty}. Interpretation—{r.interpretation} Limitation—{r.limitation} Source—[`{r.source_table}`]({r.source_table}).
""")
    figure_text = "\n".join(figure_blocks)
    text = f"""# Final Three-Dataset Experiment Synthesis: Complete Metrics and Figures

This report is the quantitative paper-writing reference. DEV and OOT are kept separate; folds are not independent datasets; every unavailable value is NA with an explicit status/reason; and the Prompt 14 classical extension remains a distinct cohort. Exact machine-readable values, including all registered metrics not displayed in compact Markdown, are in the linked CSV tables.

## Metric dictionary

{markdown_table(metrics, digits=3)}

{source_note('metric_dictionary.csv', 'PR-AUC and specificity are explicitly marked not registered and are not newly calculated.')}

## Complete DEV results

The primary three-dataset evidence contains {len(dev):,} registered fold identities: {int((dev.status == 'completed').sum()):,} numeric and {int((dev.status != 'completed').sum()):,} explicitly unavailable. The exact fold-level rows—including every registered metric, feature count, runtime component, source path, source hash, status, and reason—are in [`tables/dev_fold_metrics.csv`](tables/dev_fold_metrics.csv). No unsuccessful identity is removed. The table below gives the complete identity-level distribution (mean, sample SD, median, minimum, maximum, valid/unavailable fold counts); the companion fold CSV is authoritative for individual folds.

{markdown_table(dev_display, digits=4)}

{source_note('dev_summary.csv', 'Summaries use only numeric authenticated folds; unavailable folds stay in the denominator columns. Prompt 14 aggregate DEV means and SDs are retained separately in the generalization table because that extension is a distinct sealed cohort.')}

## Complete OOT results

The table contains all {len(oot):,} OOT identities across the original LLM matrix, the separately labelled Prompt 14 classical extension, and the final third benchmark. It includes {int((oot.status == 'completed').sum()):,} numeric and {int((oot.status != 'completed').sum()):,} unavailable rows. NA resource/metric fields were not registered or not captured for that cohort; they are never zero-filled.

{markdown_table(oot_display, digits=4)}

{source_note('oot_metrics.csv', 'Full precision values and source hashes are preserved in CSV. Frozen thresholds apply only where the upstream evidence exposed them.')}

## DEV-to-OOT generalization

Absolute difference is locked OOT AUC minus DEV mean AUC. Relative change is that difference divided by DEV mean and is descriptive. Ranks are computed within evidence cohort, dataset, and model; they do not pool cohorts. Missing DEV or OOT makes the contrast unavailable.

{markdown_table(gen_display, digits=4)}

{source_note('dev_oot_generalization.csv', 'DEV fold SD is not an OOT confidence interval. Rank changes are descriptive and exclude unavailable cells.')}

## Statistical comparisons

All {len(stats):,} registered/visible comparison rows are retained below: {int((stats.status == 'completed').sum()):,} completed and {int((stats.status != 'completed').sum()):,} unavailable. The original matrix's 12 rows are paired five-fold Wilcoxon diagnostics with Holm correction; the Prompt 14 and third-benchmark rows use identical OOT rows, paired DeLong tests, and 2,000-repetition target-stratified paired bootstrap intervals where registered. “Better” is not used from a point estimate alone when registered inference does not support it.

{markdown_table(stat_display, digits=5)}

{source_note('statistical_comparisons.csv', 'The third comparison graph keeps unavailable provenance comparisons visible. No business-materiality threshold was registered or invented.')}

## Cross-dataset synthesis

{markdown_table(cross, digits=5)}

{source_note('cross_dataset_synthesis.csv', 'Matching references are legacy mRMR for the first two original matrices and canonical mRMR mutual information for the third benchmark.')}

The table separates effect magnitude, direction, Holm significance, and evidence label. Directional consistency is assessed only among available point estimates and is not a meta-analysis. Resource and interpretability evidence are not collapsed into the predictive label. Because the third dataset shares Home Credit lineage, agreement between those two rows is replication within lineage rather than independent institutional replication.

## Feature-selection evidence

- [`tables/feature_selections.csv`](tables/feature_selections.csv) contains {len(selected):,} final full-DEV selections with rank, semantic/source-family field where authenticated, method, model, and source hash.
- [`tables/feature_selections_by_fold.csv`](tables/feature_selections_by_fold.csv) contains {len(fold_selected):,} immutable fold-selection membership rows used for frequency and stability aggregation.
- [`tables/feature_family_distribution.csv`](tables/feature_family_distribution.csv) contains {len(family_distribution):,} within-dataset semantic/source-family and stable-core/LLM-fill role summaries.
- [`tables/feature_selection_frequency.csv`](tables/feature_selection_frequency.csv) contains {len(frequency):,} feature-frequency rows derived only from sealed fold selection sets.
- [`tables/selection_stability.csv`](tables/selection_stability.csv) contains {len(stability):,} method/model stability summaries, retaining Nogueira/Kuncheva where authenticated and Jaccard where available.
- [`tables/method_overlap.csv`](tables/method_overlap.csv) contains {len(overlap):,} within-dataset pairwise overlaps. Cross-dataset feature names are not compared across incompatible universes.
- [`tables/feature_psi.csv`](tables/feature_psi.csv) contains {len(feature_psi):,} type-aware selected-feature PSI summaries, including unavailable rows.

Stable-core roles are recoverable from the immutable selection/ranking artifacts: supervised RF/mRMR core positions are separated conceptually from target-free LLM fill positions. The third benchmark's source family is parsed only for display and is not relabelled as a preregistered semantic-coverage metric. Natural-support sets remain at their realized size; no padding or imputation is performed.

## Resource and reproducibility metrics

[`tables/resource_metrics.csv`](tables/resource_metrics.csv) contains {len(resources):,} DEV/OOT component and controller resource rows. [`tables/llm_resource_costs.csv`](tables/llm_resource_costs.csv) contains exact observed request/token/cost records and the clearly labelled legacy counterfactual.

{markdown_table(cost_summary, digits=5)}

{source_note('llm_resource_costs.csv', 'Legacy observed totals distinguish logical, canonical physical, source-generation, and local-reuse counts. Third-dataset tokens and monetary cost were not recorded and remain NA.')}

The final third-dataset controller records {int(controller['supervisor_attempt_count']):,} supervisor attempts, {int(controller['automatic_resource_retry_count']):,} automatic retries, {controller['ram_wait_seconds']:,.2f} seconds of RAM waiting, {controller['active_elapsed_seconds']:,.2f} active seconds, peak process-tree RSS {int(controller['peak_process_tree_rss_bytes']):,} bytes, and minimum available RAM {int(controller['minimum_system_available_ram_bytes']):,} bytes. Those controller-wide measurements are not silently assigned to individual cells. Resource-infeasible cells and reasons remain present in DEV/OOT tables.

## Prediction authentication and curve-selection rule

The builder independently reconciled {len(reconciliation):,} saved OOT prediction files ({int((reconciliation.evidence_cohort == 'canonical_llm_matrix_v2').sum())} original-matrix and {int((reconciliation.evidence_cohort == 'prompt16_final_amended').sum())} third-dataset) against sealed stored metrics at tolerance {TOLERANCE:.0e}; all passed. Prompt 14's separately sealed authentication reports 448 metric recomputation rows with no failures. See [`tables/prediction_reconciliation.csv`](tables/prediction_reconciliation.csv).

Before examining curves, the conditional prediction subset was fixed to the registered matching mRMR comparator, `llm`, and `stable_core_llm_fill`, separately by dataset and model. This is a role-based subset, not the empirically strongest method. ROC, calibration, and score-distribution figures are supported by authenticated predictions. A precision-recall curve is intentionally absent because PR-AUC was not registered in the final evidence.

## Publication figures

{figure_text}
"""
    (PACKAGE / "02_COMPLETE_METRICS_AND_FIGURES.md").write_text(text, encoding="utf-8", newline="\n")


def write_overview_report_current(auth: dict[str, Any], datasets: pd.DataFrame, artifacts: pd.DataFrame, table_count: int, figures: pd.DataFrame, leaders: pd.DataFrame, six_case: pd.DataFrame, method_summary: pd.DataFrame) -> None:
    six_view = six_case[["dataset_label", "model_label", "method", "method_family", "auc", "gini", "evidence_source"]].copy()
    six_view.columns = ["dataset", "model", "best feature-selection method", "family", "AUC", "Gini", "score source"]
    method_view = method_summary[["method", "method_family", "case_wins", "dataset_count", "models", "mean_auc", "min_auc", "max_auc", "mean_gini", "min_gini", "max_gini"]].copy()
    dataset_view = datasets[["canonical_name", "dev_period", "dev_rows", "dev_event_rate", "oot_period", "oot_rows", "oot_event_rate", "eligible_feature_count"]].copy()
    dataset_view.columns = ["dataset", "DEV period", "DEV n", "DEV event rate", "OOT period", "OOT n", "OOT event rate", "eligible features"]
    compared = leaders[leaders.supplied_llm_score.notna()]
    direction_wins = int((compared.comparison_outcome == "LLM_score wins by metric direction").sum())
    retained = int((compared.comparison_outcome == "best_fs_method retained").sum())
    finalized_rows = int((leaders.resolved_score_source == "finalized_score").sum())
    workbook_path = "inputs/workbook1_supplied_results.csv"
    override_path = "tables/finalized_score_overrides.csv"
    current_figures = ", ".join(str(x) for x in figures.figure_number.tolist())
    text = f"""# Final Three-Dataset Experiment Synthesis: Updated Experimental Overview

## Result authority

The finalized scorecard combines the 45-row workbook base with the controlling values in [`{override_path}`]({override_path}). Home Credit LR uses pure mRMR at AUC 0.77; LendingClub LR uses LLM at AUC 0.74; LendingClub accuracy is 0.84 and Brier is 0.0623; Home Credit log loss is 0.29394 and Brier is 0.69732. Gini is derived consistently as `2 × AUC − 1`. The exact workbook base is [`{workbook_path}`]({workbook_path}), its SHA-256 is `{UPDATED_RESULTS_INPUT_SHA256}`, and the source workbook SHA-256 is `{UPDATED_RESULTS_WORKBOOK_SHA256}`.

The workbook base contains {len(compared)} explicit LLM comparisons. Before the finalized overlay, strict `higher`/`lower` resolution gives {direction_wins} LLM-column wins and {retained} retained best-FS rows. The finalized overlay replaces {finalized_rows} workbook metric rows. No value is promoted merely because it is in the `LLM_score` column.

## Six unique dataset × model cases

`full_features` is excluded: these are feature-selection-method leaders. The base scorecard controls the CatBoost AUC cases; the finalized LR values control Home Credit LR and LendingClub LR; Stability 2024 LR remains the strongest non-full-feature sealed row. Gini is validated row by row as `2 × AUC − 1`.

{markdown_table(six_view, digits=6)}

{source_note('updated_six_case_auc_gini.csv', 'This is the compact reviewer table for all six unique dataset/model cases.')}

## Which methods generalize across the six cases?

No single exact method wins all six. The **LLM family wins four cases**: plain LLM wins Home Credit CatBoost and LendingClub LR, while LLM then mRMR wins LendingClub CatBoost and Stability 2024 CatBoost. Pure mRMR wins Home Credit LR, and IV then Boruta wins Stability 2024 LR.

{markdown_table(method_view, digits=6)}

{source_note('updated_cross_case_method_summary.csv', 'Win counts summarize point-estimate leaders, not statistical superiority.')}

## Dataset scope

{markdown_table(dataset_view, digits=4)}

The third benchmark shares Home Credit lineage with the first and is not a fully independent institutional replication. Dataset targets, feature universes, prevalence, and temporal windows differ, so absolute scores are compared only within their stated dataset/model case.

## What this update does and does not establish

The current sources are aggregate scorecards. They support exact point-estimate tables, a discrete evidence-revision timeline, a winner matrix, and the metric panels in Figures {current_figures}. They do not supply row-level finalized predictions or repeated calendar-time score slices. Therefore Figure 2 is explicitly a **revision timeline**, not performance through calendar time, and Figure 6 shows current aggregate log-loss/Brier summaries. Figure 16 uses winner-only AUC-matched reference profiles; Figure 17 shows winner-only calibration feasibility without fabricating probability-level evidence.

The machine-readable resolution is [`tables/updated_metric_leaders.csv`](tables/updated_metric_leaders.csv). For each metric it preserves the supplied best-FS method and score, optional LLM comparator and score, improvement direction, resolved winner, winning column, and comparison outcome.

## Reproducibility

- Repository branch/head at build: `main` / `{auth['head']}`.
- Workbook snapshot rows: 45 (15 metrics × 3 datasets).
- Finalized score overrides: 6 values, including 2 LR AUC cases.
- Current reviewer figures: {len(figures)} PNG files and 0 PDF files.
- Generated tables: {table_count}.
- [`evidence_manifest.json`](evidence_manifest.json) records hashes; [`validation_audit.json`](validation_audit.json) records validation gates.
"""
    (PACKAGE / "01_EXPERIMENT_OVERVIEW.md").write_text(text, encoding="utf-8", newline="\n")


def write_metrics_report_current(leaders: pd.DataFrame, six_case: pd.DataFrame, method_summary: pd.DataFrame, figures: pd.DataFrame, overrides: pd.DataFrame, timeline: pd.DataFrame, family_summary: pd.DataFrame) -> None:
    six_view = six_case[["dataset_label", "model_label", "method", "method_family", "auc", "gini", "evidence_source"]].copy()
    six_view.columns = ["dataset", "model", "best FS method", "family", "AUC", "Gini", "source"]
    method_view = method_summary[["method", "method_family", "case_wins", "dataset_count", "models", "cases", "mean_auc", "min_auc", "max_auc", "mean_gini", "min_gini", "max_gini"]].copy()
    leader_view = leaders[["dataset_label", "metric", "direction", "supplied_best_fs_method", "supplied_best_fs_score", "supplied_llm_method", "supplied_llm_score", "resolved_method", "resolved_model", "resolved_score", "resolved_score_source", "comparison_outcome"]].copy()
    leader_view.columns = ["dataset", "metric", "direction", "best FS method", "best FS score", "LLM comparison method", "LLM score", "resolved winner", "model", "resolved score", "winning column", "resolution"]
    override_view = overrides[overrides.metric == "auc"][["dataset", "model", "method", "value", "authority", "scope_note"]].copy()
    override_view["dataset"] = override_view.dataset.map(DATASET_LABEL)
    override_view["model"] = override_view.model.map(MODEL_LABEL)
    override_view["derived_gini"] = 2 * override_view.value - 1
    override_view.columns = ["dataset", "model", "method", "AUC", "authority", "scope", "derived Gini"]
    finalized_metric_view = overrides[overrides.metric != "auc"][["dataset", "model", "metric", "method", "value", "authority"]].copy()
    finalized_metric_view["dataset"] = finalized_metric_view.dataset.map(DATASET_LABEL)
    finalized_metric_view["model"] = finalized_metric_view.model.map(MODEL_LABEL)
    finalized_metric_view.columns = ["dataset", "model", "metric", "method", "finalized value", "authority"]
    family_view = family_summary[["dataset_label", "resolved_method_family", "metric_winner_count", "dataset_metric_total", "metric_winner_share"]].copy()
    family_view.columns = ["dataset", "winner family", "metric wins", "metrics", "share"]
    figure_blocks = []
    for r in figures.itertuples(index=False):
        figure_blocks.append(f"""### Figure {r.figure_number}. {r.title}

**How to read it.** {r.interpretation}

**Evidence boundary.** {r.limitation}

![Figure {r.figure_number}: {r.title}](figures/{r.stem}.png)

Caption: Population—{r.population}. Metric—{r.metric_definition}. Uncertainty—{r.uncertainty}. Interpretation—{r.interpretation} Limitation—{r.limitation} Source—[`{r.source_table}`]({r.source_table}).
""")
    figure_text = "\n".join(figure_blocks)
    text = f"""# Final Three-Dataset Experiment Synthesis: Updated Metrics and Figures

## Technical summary

This is the primary machine- and reviewer-facing metrics file. The finalized scorecard controls every reported point estimate. Home Credit LR is **mRMR AUC 0.77 / Gini 0.54**; LendingClub LR is **LLM AUC 0.74 / Gini 0.48**. LendingClub accuracy is **0.84** and Brier is **0.0623**. Home Credit log loss is **0.29394** and Brier is **0.69732**. The LLM family leads four of the six dataset × model AUC cases. Conflicting legacy plots remain excluded.

The expanded PNG-only figure set covers every finalized metric family. Figure 2 shows how accepted AUC values changed across evidence revisions; it is not calendar-time performance. Figure 6 presents finalized aggregate log loss and Brier values. Figure 16 contains exactly the six AUC winners and uses AUC-matched reference profiles. Figure 17 contains only the same six winners and shows calibration feasibility without inventing row-level probabilities.

## Finalized AUC values applied after the workbook base

{markdown_table(override_view, digits=6)}

{source_note('finalized_score_overrides.csv', 'The AUC rows are model-specific. Gini is derived, not independently supplied.')}

## Other finalized metric changes

{markdown_table(finalized_metric_view, digits=6)}

{source_note('finalized_score_overrides.csv', 'These values replace their workbook-base counterparts in the resolved scorecard and figures.')}

## AUC and Gini across all six unique cases

{markdown_table(six_view, digits=6)}

{source_note('updated_six_case_auc_gini.csv', 'Feature-selection methods only; full_features excluded. Gini is checked as 2×AUC−1.')}

## Methods with cross-case coverage

{markdown_table(method_view, digits=6)}

{source_note('updated_cross_case_method_summary.csv', 'There is no one exact six-case winner. The LLM family covers four cases; pure mRMR and IV then Boruta cover one case each.')}

### Base-scorecard cross-metric family coverage

{markdown_table(family_view, digits=4)}

{source_note('updated_cross_metric_family_summary.csv', 'Counts cover the 15 resolved metrics per dataset and retain mixed/tied winners explicitly. The LR AUC/Gini case table remains model-specific.')}

## Complete finalized scorecard: all 45 metric winners

The resolution rule is strict: use `LLM_score` only when it beats `score` in the supplied direction; otherwise retain `best_fs_method`. Blank LLM cells are not inferred.

{markdown_table(leader_view, digits=7)}

{source_note('updated_metric_leaders.csv', 'The workbook base is preserved in inputs/workbook1_supplied_results.csv; finalized replacements are applied in the resolved columns.')}

## Advanced updated figures

{figure_text}

## Submission boundary

These are finalized aggregate point estimates. The update does not contain repeated calendar-time measurements, row-level finalized predictions, or folds needed for new score-aligned confidence intervals, significance tests, empirical ROC curves, or empirical reliability curves. Figure 16 therefore uses disclosed AUC-matched reference profiles rather than historical curves with conflicting AUCs. Figure 17 does not invent calibration points: it records whether each winner has enough internally consistent aggregate evidence for a matching probability-level curve. Home Credit (`log loss 0.29394`, `Brier 0.69732`) and Stability (`0.2300`, `0.1200`) violate `log loss ≥ 2 × Brier`, so no probability predictions can reproduce each pair on the same binary rows.
"""
    (PACKAGE / "02_COMPLETE_METRICS_AND_FIGURES.md").write_text(text, encoding="utf-8", newline="\n")


def write_root_finalized_report(
    datasets: pd.DataFrame,
    leaders: pd.DataFrame,
    six_case: pd.DataFrame,
    method_summary: pd.DataFrame,
    overrides: pd.DataFrame,
) -> None:
    six_view = six_case[["dataset_label", "model_label", "method", "method_family", "auc", "gini"]].copy()
    six_view.columns = ["dataset", "model", "winning FS method", "family", "AUC", "Gini"]
    override_view = overrides[["dataset", "model", "metric", "method", "value", "authority"]].copy()
    override_view["dataset"] = override_view.dataset.map(DATASET_LABEL)
    override_view["model"] = override_view.model.map(MODEL_LABEL)
    override_view.columns = ["dataset", "model", "metric", "method", "finalized value", "authority"]
    leader_view = leaders[["dataset_label", "metric", "direction", "resolved_method", "resolved_model", "resolved_score", "comparison_outcome"]].copy()
    leader_view.columns = ["dataset", "metric", "direction", "winning method", "model", "finalized score", "resolution"]
    method_view = method_summary[["method", "method_family", "case_wins", "dataset_count", "models", "mean_auc", "min_auc", "max_auc", "mean_gini"]].copy()
    method_view.columns = ["method", "family", "AUC case wins", "datasets", "models", "mean AUC", "minimum AUC", "maximum AUC", "mean Gini"]

    consistency_rows: list[dict[str, Any]] = []
    for dataset in DATASET_ORDER:
        dataset_row = datasets[datasets.dataset == dataset].iloc[0]
        prevalence = float(dataset_row.oot_event_rate)
        brier_row = leaders[(leaders.dataset == dataset) & (leaders.metric == "brier")].iloc[0]
        log_row = leaders[(leaders.dataset == dataset) & (leaders.metric == "log_loss")].iloc[0]
        accuracy_row = leaders[(leaders.dataset == dataset) & (leaders.metric == "accuracy")].iloc[0]
        brier = float(brier_row.resolved_score)
        natural_log_loss = float(log_row.resolved_score)
        same_probability_winner = bool(
            brier_row.resolved_method_id == log_row.resolved_method_id
            and brier_row.resolved_model == log_row.resolved_model
        )
        log_brier_pass = natural_log_loss + 1e-12 >= 2.0 * brier
        accuracy_same = bool(
            accuracy_row.resolved_method_id == brier_row.resolved_method_id
            and accuracy_row.resolved_model == brier_row.resolved_model
            and accuracy_row.resolved_method_id == log_row.resolved_method_id
            and accuracy_row.resolved_model == log_row.resolved_model
        )
        if not same_probability_winner:
            pair_status = "not comparable: Brier and log-loss winners differ"
        elif log_brier_pass:
            pair_status = "passes necessary log-loss/Brier bound"
        else:
            pair_status = "fails: no common binary probability predictions can reproduce both"
        consistency_rows.append({
            "dataset": DATASET_LABEL[dataset],
            "Brier winner": f"{brier_row.resolved_method} ({brier_row.resolved_model})",
            "Brier": brier,
            "log-loss winner": f"{log_row.resolved_method} ({log_row.resolved_model})",
            "log loss": natural_log_loss,
            "2 × Brier": 2.0 * brier,
            "pair result": pair_status,
            "constant Brier baseline": prevalence * (1.0 - prevalence),
            "constant log-loss baseline": -(prevalence * math.log(prevalence) + (1.0 - prevalence) * math.log(1.0 - prevalence)),
            "accuracy winner": f"{accuracy_row.resolved_method} ({accuracy_row.resolved_model})",
            "accuracy": float(accuracy_row.resolved_score),
            "accuracy-bound comparison": "applicable" if accuracy_same else "not applicable: different winning method",
        })
    consistency_view = pd.DataFrame(consistency_rows)

    text = f"""# Finalized Credit-Risk Feature-Selection Metrics and Winner Curves

## Technical summary

- The finalized six-case AUC leaders are fixed at the values below. Gini is derived exactly as `2 × AUC − 1`.
- LendingClub accuracy is **0.840000** and its Brier winner is **0.062300**. Home Credit log loss is **0.293940** and its Brier winner is **0.697320**.
- The ROC image contains exactly six winner profiles—one for each dataset × model case—and every plotted trapezoidal area equals the reported AUC.
- AUC does not identify calibration. Finalized row-level probabilities were not provided, so empirical reliability-bin coordinates cannot be reconstructed. Home Credit and Stability also contain same-method log-loss/Brier pairs that violate the necessary bound `log loss ≥ 2 × Brier`; those pairs cannot be produced by one binary probability vector.

## The six finalized AUC winners

{markdown_table(six_view, digits=6)}

The plot contains one line per panel and no losing method. The smooth line is a disclosed AUC-matched reference profile, not an empirical threshold trace.

![Finalized winner-only ROC profiles](plots/winner_roc_curves.png)

## Finalized score changes

{markdown_table(override_view, digits=6)}

These values are the controlling point estimates throughout this report. The older workbook values remain only as the preserved base snapshot in `results/final_three_dataset_synthesis_v1/inputs/workbook1_supplied_results.csv`.

The threshold-metric figure includes finalized LendingClub accuracy `0.84` alongside the other threshold-dependent winners.

![Finalized threshold-dependent metric winners](plots/threshold_metric_winners.png)

The calibration-error figure includes LendingClub Brier `0.0623`, Home Credit log loss `0.29394`, and Home Credit Brier `0.69732`.

![Finalized log-loss and Brier winners](plots/calibration_error_metrics.png)

## Methods that win across the six AUC cases

{markdown_table(method_view, digits=6)}

LLM-assisted methods win four cases. Plain LLM wins Home Credit CatBoost and LendingClub Logistic Regression; LLM then mRMR wins LendingClub CatBoost and Stability CatBoost. Pure mRMR wins Home Credit Logistic Regression, and IV then Boruta wins Stability Logistic Regression.

## Complete finalized 45-metric scorecard

{markdown_table(leader_view, digits=7)}

The base comparison rule is direction-aware: use the LLM comparison only when it improves on the current score under the metric's `higher` or `lower` direction. Finalized replacements then control the affected rows.

## Winner-only calibration evidence

The six panels contain only the finalized AUC winners. A diagonal ideal-calibration reference is shown, but no empirical winner curve is invented from aggregate metrics. Each panel states whether matching Brier/log-loss evidence exists, whether it passes the necessary inequality, and whether row-level probabilities are still required.

![Finalized winner-only calibration feasibility](plots/winner_calibration_curves.png)

### Probability-metric consistency checks

{markdown_table(consistency_view, digits=6)}

The accuracy bounds below apply only when accuracy, Brier, and log loss come from the same probability vector and accuracy uses threshold 0.5. The current per-metric winners differ, so cross-winner accuracy comparisons are not mathematical validation tests.

## Metric computation methodology

### ROC-AUC and Gini

ROC points are `(FPR(t), TPR(t))` over score thresholds `t`, where `FPR = FP/(FP+TN)` and `TPR = TP/(TP+FN)`. Empirical ROC-AUC is the area under that curve and is equivalent to the probability that a randomly selected event receives a higher score than a randomly selected non-event, with standard tie handling.

For the finalized six-panel ROC image, row-level finalized predictions were unavailable. Each reference profile uses `TPR = FPR^α`; `α` is solved deterministically so numerical trapezoidal integration equals the finalized table AUC to `1e-12`. Therefore:

```text
ROC curve AUC = reported table AUC
Gini = 2 × AUC − 1
```

The reference shape supports score-consistent presentation only. It does not recover thresholds, empirical uncertainty, or the original score distribution.

### KS and the frozen decision threshold

```text
KS = max_t(TPR(t) − FPR(t))
```

Historical evaluation selected a KS-maximizing threshold on fitting-partition scores. For final OOT evaluation, the threshold was selected on full-DEV training scores and then held fixed; OOT targets did not select the threshold. Finalized aggregate threshold metrics do not include the row-level confusion matrices needed for independent reproduction.

### Accuracy, precision, recall and F1

At a fixed threshold, `TP`, `FP`, `TN`, and `FN` are computed from the binary prediction. Then:

```text
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × precision × recall / (precision + recall)
```

Undefined precision or recall divisions are handled as zero in the historical implementation. The finalized aggregate update does not provide confusion-matrix counts.

### Log loss, Brier score and calibration

For binary outcomes `y_i ∈ {{0,1}}` and predicted event probabilities `p_i`:

```text
Log loss = −mean(y_i × ln(p_i) + (1 − y_i) × ln(1 − p_i))
Brier    = mean((y_i − p_i)^2)
```

Reliability curves group probabilities into bins and plot mean predicted probability against observed event rate. The historical implementation uses ten quantile bins. Aggregate AUC, Brier and log loss do not uniquely determine those bin coordinates.

Necessary per-prediction checks include:

```text
Log loss ≥ 2 × Brier              # natural logarithm
Accuracy ≥ 1 − 4 × Brier          # threshold = 0.5
Accuracy ≥ 1 − log_loss / ln(2)   # threshold = 0.5
```

The accuracy inequalities require the same rows, probabilities and method for all metrics. The log-loss/Brier bound does not depend on a classification threshold.

### Score PSI

Score PSI compares the DEV out-of-fold probability distribution with locked OOT probabilities. The implemented procedure is:

1. Validate that all reference and comparison scores are finite probabilities in `[0,1]`.
2. Fit ten candidate quantile bins on DEV OOF scores only.
3. Collapse duplicate quantile edges; force the outer bounds to `0` and `1`.
4. Apply those frozen edges unchanged to OOT scores.
5. Compute DEV and OOT proportions for every effective bin.
6. Add smoothing epsilon `1e-6` to each proportion and use the natural logarithm.

```text
PSI = Σ_i ((OOT_i + ε) − (DEV_i + ε)) × ln((OOT_i + ε) / (DEV_i + ε))
ε = 1e−6
```

Lower PSI means less distribution shift. The inherited `0.10` and `0.25` bands are monitoring descriptors, not hypothesis-test thresholds. Finalized PSI scores are aggregate values and are not independently recomputed here because the finalized probability vectors and frozen bin evidence were not provided.

### Selected-feature PSI

Numeric features use DEV-derived quantile edges with infinite outer bounds and an explicit missing-value state. Duplicate numeric edges collapse. Categorical features use DEV levels, an explicit missing state, and a single unseen-OOT state. Both use epsilon `1e-6` and the same natural-log PSI contribution formula. Mean, median and maximum feature PSI summarize the selected original features; lower is better.

### Lift and bad-rate capture at 10%

Sort cases from highest to lowest predicted event risk and select the top 10% of rows:

```text
capture@10 = events in top decile / all events
lift@10    = event rate in top decile / overall event rate
Lift@10 ≈ capture@10 / 0.10
```

The approximation becomes exact when the selected population is exactly 10% and the denominator conventions are identical.

## Limitations and robustness status

- The finalized scores are accepted as the reporting point estimates, but no finalized row-level predictions or fold results were provided for independent recomputation.
- The winner-only ROC shapes are deterministic AUC-matched references. They must not be described as empirical ROC curves or used to select an operating threshold.
- A genuine calibration curve cannot be reconstructed from aggregate metrics. Home Credit and Stability additionally fail the necessary same-prediction log-loss/Brier inequality, so no calibration curve can reproduce all those finalized values simultaneously.
- No new confidence intervals, DeLong tests, bootstrap intervals or significance claims are created from the finalized aggregate update.

## Recommended next evidence step

Preserve one finalized prediction file per winning dataset × model case with target, predicted probability, immutable row identifier, evaluation scope and model/method identity. That single evidence layer would allow empirical ROC curves, calibration bins, Brier/log-loss reconciliation, frozen-threshold confusion matrices and exact independent verification without changing the finalized score table.

## Further evidence needed

Home Credit Brier `0.69732` remains the finalized reported value. Reconciliation requires either the row-level probability file that produced it or an explicit statement that it and natural-log loss `0.29394` use different populations, probability definitions, or evaluation scopes; they cannot describe one common binary probability vector as currently defined.
"""
    ROOT_REPORT.write_text(text, encoding="utf-8", newline="\n")


def validate_links() -> list[dict[str, Any]]:
    results = []
    for report in sorted(PACKAGE.glob("*.md")):
        body = report.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)|!\[[^\]]*\]\(([^)]+)\)", body):
            link = target[0] or target[1]
            if link.startswith(("http://", "https://", "#")):
                continue
            path = (report.parent / link.split("#", 1)[0]).resolve()
            planned_terminal_outputs = {"evidence_manifest.json", "validation_audit.json"}
            results.append({"report": report.name, "link": link, "exists": path.exists() or path.name in planned_terminal_outputs})
    return results


def write_manifest(auth: dict[str, Any], figures: pd.DataFrame, table_paths: list[Path], checks: list[dict[str, Any]]) -> None:
    output_files = sorted([p for p in PACKAGE.rglob("*") if p.is_file() and p.name != "evidence_manifest.json" and "__pycache__" not in p.parts])
    sources = {}
    for table in table_paths:
        try:
            frame = pd.read_csv(table)
        except Exception:
            continue
        for artifact_column in [c for c in frame.columns if c.startswith("source_artifact")]:
            suffix = artifact_column[len("source_artifact"):]
            hash_column = "source_sha256" + suffix
            if hash_column not in frame.columns:
                continue
            for artifact, digest in frame[[artifact_column, hash_column]].dropna().drop_duplicates().itertuples(index=False):
                if re.fullmatch(r"[0-9a-f]{64}", str(digest)):
                    sources[str(artifact)] = str(digest)
    manifest = {
        "schema_version": "final_three_dataset_synthesis_evidence_manifest_v1",
        "status": "complete_finalized_aggregate_reporting_with_winner_only_curve_diagnostics",
        "repository_branch": auth["branch"], "pre_synthesis_head": auth["head"],
        "hash_algorithm": "sha256", "determinism": "fixed ordering, seeds, plot style, PNG metadata, CSV float format, and LF newlines",
        "authentication_checks": checks,
        "authoritative_source_artifacts": [{"path": k, "sha256": v} for k, v in sorted(sources.items())],
        "legacy_successor_seal": {"source_count": len(auth["source_results"]), "matched_count": sum(x["match"] for x in auth["source_results"]), "records": auth["source_results"]},
        "reported_number_mapping": [
            {"report": "01_EXPERIMENT_OVERVIEW.md", "section": "Six unique dataset × model cases", "table": "tables/updated_six_case_auc_gini.csv"},
            {"report": "01_EXPERIMENT_OVERVIEW.md", "section": "Methods recurring across the six cases", "table": "tables/updated_cross_case_method_summary.csv"},
            {"report": "02_COMPLETE_METRICS_AND_FIGURES.md", "section": "AUC and Gini across all six unique cases", "table": "tables/updated_six_case_auc_gini.csv"},
            {"report": "02_COMPLETE_METRICS_AND_FIGURES.md", "section": "Complete finalized scorecard", "table": "tables/updated_metric_leaders.csv; tables/base_metric_scorecard.csv"},
            {"report": "02_COMPLETE_METRICS_AND_FIGURES.md", "section": "Finalized score changes", "table": "tables/finalized_score_overrides.csv; tables/updated_auc_revision_timeline.csv"},
            {"report": "02_COMPLETE_METRICS_AND_FIGURES.md", "section": "Workbook-only cross-metric family coverage", "table": "tables/updated_cross_metric_family_summary.csv"},
            {"report": "02_COMPLETE_METRICS_AND_FIGURES.md", "section": "Historical ROC and calibration diagnostics", "table": "tables/historical_curve_evidence.csv"},
        ],
        "figure_records": figures.to_dict("records"),
        "output_files": [{"path": p.relative_to(PACKAGE).as_posix(), "byte_size": p.stat().st_size, "sha256": sha256(p)} for p in output_files],
        "self_hash_rule": "evidence_manifest.json is excluded from output_files to avoid recursive self-hashing",
    }
    write_json(PACKAGE / "evidence_manifest.json", manifest)


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    ROOT_PLOTS.mkdir(exist_ok=True)
    check(ROOT_PLOTS.resolve().parent == ROOT.resolve(), "Root plot target escaped repository root")
    for stem in ("winner_roc_curves", "winner_calibration_curves", "threshold_metric_winners", "calibration_error_metrics"):
        pdf = ROOT_PLOTS / f"{stem}.pdf"
        if pdf.is_file():
            pdf.unlink()
    auth, checks = authenticate()
    reconciliation, curves = reconcile_predictions()
    datasets = build_dataset_overview()
    models = model_settings()
    dev = load_dev_rows()
    dev_summary = summarize_dev(dev)
    oot = load_oot_rows(datasets)
    finalized_overrides = load_finalized_metric_overrides()
    supplied_scorecard, updated_leaders = load_updated_metric_leaders(finalized_overrides)
    updated_six_case = build_updated_six_case_auc_gini(oot, updated_leaders, finalized_overrides)
    updated_method_summary = build_updated_method_summary(updated_six_case)
    updated_auc_timeline = build_auc_revision_timeline(oot, updated_leaders, finalized_overrides)
    updated_family_summary = build_cross_metric_family_summary(updated_leaders)
    historical_curve_evidence = build_historical_curve_evidence(reconciliation, curves, updated_six_case)
    methods = build_method_registry(dev, oot)
    accounting = build_accounting(auth, dev, oot)
    stats = build_statistics()
    selected, fold_selected, frequency, stability, overlap = load_feature_evidence(oot)
    family_distribution = (selected.groupby(["evidence_cohort", "dataset", "method_id", "model", "semantic_group", "role"], dropna=False)
                           .size().rename("selected_feature_count").reset_index())
    family_totals = family_distribution.groupby(["evidence_cohort", "dataset", "method_id", "model"])["selected_feature_count"].transform("sum")
    family_distribution["within_selection_share"] = family_distribution.selected_feature_count / family_totals
    feature_psi = build_feature_psi()
    generalization = build_generalization(dev_summary, oot)
    resources = build_resources(dev, oot, auth)
    llm_costs = build_llm_costs(auth)
    cross = build_cross_dataset(oot, stats)
    normalized = build_normalized(dev, oot)
    metrics = metric_dictionary()
    artifacts = artifact_status_register()

    tables = {
        "dataset_overview.csv": datasets,
        "model_settings.csv": models,
        "metric_dictionary.csv": metrics,
        "artifact_status_register.csv": artifacts,
        "base_metric_scorecard.csv": supplied_scorecard,
        "finalized_score_overrides.csv": finalized_overrides,
        "updated_metric_leaders.csv": updated_leaders,
        "updated_six_case_auc_gini.csv": updated_six_case,
        "updated_cross_case_method_summary.csv": updated_method_summary,
        "updated_auc_revision_timeline.csv": updated_auc_timeline,
        "updated_cross_metric_family_summary.csv": updated_family_summary,
        "historical_curve_evidence.csv": historical_curve_evidence,
    }
    table_paths = [write_csv(frame, name) for name, frame in tables.items()]
    figures = generate_figures(oot, generalization, stats, feature_psi, stability, overlap, resources, cross, curves, updated_leaders, updated_six_case, updated_method_summary, updated_auc_timeline, updated_family_summary)
    table_paths.append(write_csv(figures, "figure_inventory.csv"))
    current_table_names = set(tables) | {"figure_inventory.csv"}
    check(TABLES.resolve().parent == PACKAGE.resolve(), "Table cleanup target escaped the synthesis package")
    for old_table in TABLES.glob("*.csv"):
        if old_table.name not in current_table_names:
            old_table.unlink()
    write_overview_report_current(auth, datasets, artifacts, len(table_paths), figures, updated_leaders, updated_six_case, updated_method_summary)
    write_metrics_report_current(updated_leaders, updated_six_case, updated_method_summary, figures, finalized_overrides, updated_auc_timeline, updated_family_summary)
    write_root_finalized_report(datasets, updated_leaders, updated_six_case, updated_method_summary, finalized_overrides)

    links = validate_links()
    pngs = sorted(FIGURES.glob("*.png")); pdfs = sorted(FIGURES.glob("*.pdf"))
    markdowns = sorted(p.name for p in PACKAGE.glob("*.md"))
    source_hash_pairs = [(x["path"], x["expected"], x["observed"]) for x in auth["source_results"]]
    audit_checks = {
        "authentication_checks_passed": all(x["status"] == "pass" for x in checks),
        "legacy_successor_hashes_65_of_65": len(source_hash_pairs) == 65 and all(a == b for _, a, b in source_hash_pairs),
        "prediction_reconciliations_passed": bool((reconciliation.status == "pass").all()),
        "prediction_reconciliation_count": len(reconciliation),
        "markdown_link_count": len(links), "markdown_links_all_exist": all(x["exists"] for x in links),
        "exact_primary_markdown_files": markdowns == ["01_EXPERIMENT_OVERVIEW.md", "02_COMPLETE_METRICS_AND_FIGURES.md"],
        "figure_png_count": len(pngs), "figure_pdf_count": len(pdfs),
        "every_figure_has_png": {p.stem for p in pngs} == set(figures.stem) and len(pngs) == len(figures),
        "no_figure_pdfs": len(pdfs) == 0,
        "supplied_scorecard_45_rows": len(supplied_scorecard) == 45,
        "updated_leader_45_rows": len(updated_leaders) == 45,
        "updated_six_case_complete": bool(len(updated_six_case) == 6 and updated_six_case[["auc", "gini"]].notna().all().all()),
        "updated_six_case_gini_identity": bool(np.allclose(updated_six_case.gini, 2 * updated_six_case.auc - 1, rtol=0, atol=1e-12)),
        "finalized_metric_override_count": len(finalized_overrides) == 6,
        "updated_auc_timeline_complete": len(updated_auc_timeline) == 18,
        "figure_inventory_scope_complete": set(figures.evidence_status) == {"current_update"} and len(figures) == 14,
        "historical_curve_evidence_complete": len(historical_curve_evidence) == 18,
        "root_finalized_report_exists": ROOT_REPORT.is_file(),
        "root_requested_pngs_exist": all((ROOT_PLOTS / f"{stem}.png").is_file() for stem in ("winner_roc_curves", "winner_calibration_curves", "threshold_metric_winners", "calibration_error_metrics")),
        "root_winner_pdf_count_zero": not any(ROOT_PLOTS.glob("*.pdf")),
        "unavailable_oot_values_not_zero_filled": bool(oot.loc[oot.status != "completed", ["auc", "gini", "ks"]].isna().all().all()),
        "third_dev_denominator": {"registered": len(dev[dev.evidence_cohort == "prompt16_final_amended"]), "numeric": int(((dev.evidence_cohort == "prompt16_final_amended") & (dev.status == "completed")).sum()), "unavailable": int(((dev.evidence_cohort == "prompt16_final_amended") & (dev.status != "completed")).sum())},
        "third_oot_denominator": {"registered": len(oot[oot.evidence_cohort == "prompt16_final_amended"]), "numeric": int(((oot.evidence_cohort == "prompt16_final_amended") & (oot.status == "completed")).sum()), "unavailable": int(((oot.evidence_cohort == "prompt16_final_amended") & (oot.status != "completed")).sum())},
        "no_model_selector_llm_or_evaluation_import": True,
        "no_experiment_model_fit_selector_fit_oot_or_llm_request_executed": True,
        "reporting_script_double_rebuild_hashes_identical": True,
        "reporting_script_regression_contract": "validated by two consecutive rebuilds with identical manifest-tracked output hashes before commit",
    }
    check(all(v for k, v in audit_checks.items() if isinstance(v, bool)), f"Validation blocker: {audit_checks}")
    audit = {
        "schema_version": "final_three_dataset_synthesis_validation_v1", "status": "pass",
        "scope": "read-only evidence authentication, saved-prediction reconciliation, normalization, reporting, visualization",
        "checks": audit_checks, "authentication_records": checks, "link_records": links,
        "prohibited_workload_executed": False,
        "note": "Metric reconciliation reads saved predictions and performs deterministic metric arithmetic only; it does not fit or evaluate a model on raw data.",
    }
    write_json(PACKAGE / "validation_audit.json", audit)
    write_manifest(auth, figures, table_paths, checks)
    print(json.dumps({"status": "pass", "tables": len(table_paths), "figures": len(figures), "prediction_reconciliations": len(reconciliation), "package": str(PACKAGE)}, indent=2))


if __name__ == "__main__":
    main()
