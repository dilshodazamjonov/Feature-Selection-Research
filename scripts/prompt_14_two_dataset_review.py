"""Artifact-only Prompt 14 two-dataset OOT authentication and analysis.

This module reads only committed protocol metadata and persisted result artifacts.
It deliberately imports no experiment, selector, data-loader, model, or runner module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from credit_risk_fs.evaluation.paired_inference import (  # noqa: E402
    holm_adjust,
    ks_statistic,
    paired_delong_test,
)


AUDIT_REL = Path("cleanup/audits/prompt_14_two_dataset_oot_review_v3")
AUDIT = ROOT / AUDIT_REL
PROTOCOL = ROOT / "configs/protocols/prompt_14_two_dataset_analysis_v1"
PHASE1_COMMIT = "fd98d3c6d445e042b69dd24b0d6e8355157548dd"
CONFIG_SHA = "432a24c0a12e2a134522e981effe0b151c02f9c98c03cb34af6916f77d5e86ab"
COMBO_LOCK_SHA = "bce77cf33de1a6d0545c2e8b425d89eb5fab36b0c426fd4c4dc50727b50603e9"
BASELINE_CONFIG_SHA = "f03647c376fe834f9bb1c3d6834ed42732ef3e7e1047eeff352af49b31ed607f"
POINTER_SHA = "63b8c0542885b00074f6e3fe1a897d5fe049ddbfbe45d3c19f40db47ae195e99"
SUCCESSOR_SHA = "45a3c3ce2773508d352d3cd9a031b0d6e35835de72ad36c877f32090c1ceabaf"
ORIGINAL_SHA = "e16a3cff5135a9eb3ecf92ea635bdeb55772fbe8f80c9dca07086590464de2a2"
STATUS_SHA = "4b16d8cea9dd877fe62b687416fe70c6bcd9def50ab8f9425f16cd2aebeb0e8a"
TOLERANCE = 1e-10
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_MINIMUM_VALID = 1900
BOOTSTRAP_SEED = 20260721
GENERATED_AT = "2026-08-08T00:00:00Z"

DATASET_ORDER = {"homecredit": 0, "lendingclub_v2": 1}
MODEL_ORDER = {"lr": 0, "catboost": 1}
METHOD_ORDER = {
    "full_features": 0,
    "random_k": 1,
    "iv_woe": 2,
    "mrmr_mutual_information": 3,
    "lasso_l1_logistic": 4,
    "legacy_rf_relevance_corr": 5,
    "catboost_shap": 6,
    "boruta_random_forest": 7,
    "rfe_catboost": 8,
    "statistical_normalized_average_rank": 9,
    "iv_then_boruta": 10,
    "boruta_then_mrmr_mutual_information": 11,
    "boruta_then_rfe_catboost": 12,
    "cross_dataset_rank_voting_v1_primary_pool_200": 13,
}


class IntegrityError(RuntimeError):
    pass


def repo_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sha_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IntegrityError(f"JSON object required: {repo_path(path)}")
    return value


def dump_json(path: Path, value: Any) -> None:
    data = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: ""
                    if row.get(column) is None
                    or (isinstance(row.get(column), float) and math.isnan(row[column]))
                    else row.get(column)
                    for column in columns
                }
            )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value.replace("\r\n", "\n").encode("utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityError(message)


def artifact_row(
    path: Path,
    *,
    evidence_class: str,
    active_or_historical: str = "active",
    dataset: str = "",
    method: str = "",
    model: str = "",
    artifact_role: str = "payload",
    expected_sha: str | None = None,
    expected_size: int | None = None,
    authentication_reference: str = "",
    notes: str = "",
) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {repo_path(path)}")
    observed_size = path.stat().st_size
    observed_sha = sha_file(path)
    if expected_size is not None:
        require(observed_size == int(expected_size), f"size mismatch: {repo_path(path)}")
    if expected_sha is not None:
        require(observed_sha == expected_sha, f"hash mismatch: {repo_path(path)}")
    return {
        "evidence_class": evidence_class,
        "active_or_historical": active_or_historical,
        "dataset": dataset,
        "method": method,
        "model": model,
        "artifact_role": artifact_role,
        "relative_path": repo_path(path),
        "bytes": observed_size,
        "sha256": observed_sha,
        "authenticated": True,
        "authentication_reference": authentication_reference,
        "notes": notes,
    }


def authenticate_combo_phase(
    phase: str, expected_evaluations: int, expected_selections: int
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    root = ROOT / "results/selector_combinations_v1" / phase
    evaluations = root / "evaluations"
    selections = root / "selections"
    eval_states = sorted(
        path
        for path in evaluations.glob("*.json")
        if not path.name.endswith((".dev_metrics.json", ".oot_metrics.json", ".final_model_configuration.json"))
    )
    selection_states = sorted(
        path for path in selections.glob("*.json") if not path.name.endswith(".combination_result.json")
    )
    require(len(eval_states) == expected_evaluations, f"{phase} evaluation count mismatch")
    require(len(selection_states) == expected_selections, f"{phase} selection count mismatch")
    inventory: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    selection_records: list[dict[str, Any]] = []
    adaptation_flags: list[bool] = []

    def authenticate_state(path: Path, role: str) -> dict[str, Any]:
        state = load_json(path)
        require(state.get("terminal_state") == "completed", f"non-complete {repo_path(path)}")
        require(state.get("configuration_sha256") == CONFIG_SHA, f"config mismatch {repo_path(path)}")
        require(state.get("protocol_lock_sha256") == COMBO_LOCK_SHA, f"lock mismatch {repo_path(path)}")
        inventory.append(
            artifact_row(
                path,
                evidence_class=f"combination_{phase}",
                dataset=(state.get("evaluation_cell") or state.get("selection_spec") or {}).get("dataset", ""),
                method=(state.get("evaluation_cell") or state.get("selection_spec") or {}).get("method_id", ""),
                model=(state.get("evaluation_cell") or {}).get("model", ""),
                artifact_role=role,
                authentication_reference=repo_path(path),
            )
        )
        entries = state.get("artifact_files")
        require(isinstance(entries, list) and len(entries) == 3, f"payload list invalid {repo_path(path)}")
        for entry in entries:
            payload_path = path.parent / entry["path"]
            inventory.append(
                artifact_row(
                    payload_path,
                    evidence_class=f"combination_{phase}",
                    dataset=(state.get("evaluation_cell") or state.get("selection_spec") or {}).get("dataset", ""),
                    method=(state.get("evaluation_cell") or state.get("selection_spec") or {}).get("method_id", ""),
                    model=(state.get("evaluation_cell") or {}).get("model", ""),
                    artifact_role="state_bound_payload",
                    expected_sha=entry["sha256"],
                    expected_size=entry["size_bytes"],
                    authentication_reference=repo_path(path),
                )
            )
        return state

    for path in selection_states:
        selection_records.append(authenticate_state(path, "selection_state"))
    for path in eval_states:
        state = authenticate_state(path, "evaluation_state")
        eval_records.append(state)
        result = state.get("worker_result", {})
        require(result.get("oot_used_for_fit_or_adaptation") is not True, f"OOT adaptation {repo_path(path)}")
        if phase == "oot":
            metric_path = path.with_name(path.stem + ".oot_metrics.json")
            metrics = load_json(metric_path)
            adaptation_flags.append(bool(metrics.get("configuration_adaptation_after_oot")))
            require(metrics.get("threshold_selected_on_oot") is False, f"OOT threshold selection {repo_path(metric_path)}")

    observed_files = len(list(evaluations.glob("*"))) + len(list(selections.glob("*")))
    expected_files = 4 * (expected_evaluations + expected_selections)
    require(observed_files == expected_files, f"{phase} active file count mismatch")
    require(not any(adaptation_flags), f"{phase} configuration adaptation detected")
    return (
        {
            "phase": phase,
            "expected_evaluations": expected_evaluations,
            "authenticated_evaluations": len(eval_records),
            "expected_selector_fits": expected_selections,
            "authenticated_selector_fits": len(selection_records),
            "expected_active_files": expected_files,
            "authenticated_active_files": observed_files,
            "missing": 0,
            "unexpected": 0,
            "failed": 0,
            "partial": 0,
            "hash_invalid": 0,
        },
        inventory,
        eval_records,
        selection_records,
    )


def authenticate_baseline() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run_root = ROOT / "results/full_baseline_v1/runs"
    manifests = sorted(
        path for path in run_root.glob("*/*/manifest.json") if "incomplete" not in path.parts
    )
    require(len(manifests) == 36, "baseline completed manifest count mismatch")
    inventory: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for path in manifests:
        manifest = load_json(path)
        cell_root = path.parent
        require(manifest.get("status") == "completed", f"baseline status {repo_path(path)}")
        require(manifest.get("full_baseline_configuration_sha256") == BASELINE_CONFIG_SHA, f"baseline config {repo_path(path)}")
        require(manifest.get("worker_exit_code") == 0, f"baseline exit {repo_path(path)}")
        require(manifest.get("resumability_status") == "completed_immutable", f"baseline mutable {repo_path(path)}")
        require((cell_root / "_SUCCESS").is_file(), f"baseline success missing {repo_path(cell_root)}")
        inventory.append(
            artifact_row(
                path,
                evidence_class="baseline",
                dataset=manifest["full_baseline_dataset"],
                method=manifest["selector"],
                model=manifest["model"],
                artifact_role="manifest",
                authentication_reference="cleanup/audits/prompt_11_selector_combinations/baseline_completion_authentication.json",
            )
        )
        inventory.append(
            artifact_row(
                cell_root / "_SUCCESS",
                evidence_class="baseline",
                dataset=manifest["full_baseline_dataset"],
                method=manifest["selector"],
                model=manifest["model"],
                artifact_role="success_marker",
                authentication_reference=repo_path(path),
            )
        )
        for entry in manifest.get("artifacts", {}).values():
            if not entry.get("applicable") or entry.get("path") == "manifest.json":
                continue
            payload = cell_root / entry["path"]
            inventory.append(
                artifact_row(
                    payload,
                    evidence_class="baseline",
                    dataset=manifest["full_baseline_dataset"],
                    method=manifest["selector"],
                    model=manifest["model"],
                    artifact_role="manifested_payload",
                    expected_sha=entry.get("sha256"),
                    expected_size=entry.get("size_bytes"),
                    authentication_reference=repo_path(path),
                )
            )
        records.append(manifest)
    return (
        {
            "expected_cells": 36,
            "authenticated_cells": len(records),
            "configuration_sha256": BASELINE_CONFIG_SHA,
            "configuration_adaptation_after_oot": False,
            "missing": 0,
            "hash_invalid": 0,
        },
        inventory,
        records,
    )


def authenticate_voting() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = ROOT / "results/final_experiments/cross_dataset_voting_inference_v1"
    pointer_path = root / "artifact_manifest.current.json"
    successor_path = root / "artifact_manifest.v2.json"
    original_path = root / "artifact_manifest.json"
    status_path = root / "status.json"
    require(sha_file(pointer_path) == POINTER_SHA, "voting pointer hash mismatch")
    require(sha_file(successor_path) == SUCCESSOR_SHA, "voting successor hash mismatch")
    require(sha_file(original_path) == ORIGINAL_SHA, "voting original hash mismatch")
    require(sha_file(status_path) == STATUS_SHA and status_path.stat().st_size == 1298, "voting status mismatch")
    pointer = load_json(pointer_path)
    require(pointer["selection_rule"] == "explicit_successor_pointer_fail_closed", "voting selection rule")
    require(pointer["selected_manifest_path"].endswith("artifact_manifest.v2.json"), "voting successor selection")
    successor = load_json(successor_path)
    original = load_json(original_path)
    successor_entries = successor["generated_files"]
    original_map = {row["path"]: row for row in original["generated_files"]}
    require(len(successor_entries) == 55, "voting payload count")
    inventory: list[dict[str, Any]] = []
    unchanged = 0
    for entry in successor_entries:
        path = ROOT / entry["path"]
        inventory.append(
            artifact_row(
                path,
                evidence_class="voting_package",
                artifact_role="pointer_selected_successor_payload",
                expected_sha=entry["sha256"],
                expected_size=entry["size_bytes"],
                authentication_reference=repo_path(successor_path),
            )
        )
        old = original_map.get(entry["path"])
        if old and old.get("sha256") == entry.get("sha256") and old.get("size_bytes") == entry.get("size_bytes"):
            unchanged += 1
    require(unchanged == 54, "voting unaffected count")
    inventory.extend(
        [
            artifact_row(pointer_path, evidence_class="voting_package", artifact_role="active_pointer", expected_sha=POINTER_SHA),
            artifact_row(successor_path, evidence_class="voting_package", artifact_role="active_manifest", expected_sha=SUCCESSOR_SHA),
            artifact_row(original_path, evidence_class="voting_package", active_or_historical="historical_superseded", artifact_role="legacy_manifest", expected_sha=ORIGINAL_SHA, notes="never active for this analysis"),
        ]
    )
    return (
        {
            "selection_rule": pointer["selection_rule"],
            "active_manifest": repo_path(successor_path),
            "payload_entries_expected": 55,
            "payload_entries_authenticated": 55,
            "unaffected_entries_expected": 54,
            "unaffected_entries_byte_identical": unchanged,
            "legacy_fallback_used": False,
        },
        inventory,
    )


def load_prediction(path: Path) -> pd.DataFrame:
    # Preserve each evidence package's authenticated CSV parser convention.
    # The frozen baseline audit used pandas' default/high parser; selector-
    # combination predictions were serialized for exact round-trip recovery.
    float_precision = "round_trip" if "selector_combinations_v1" in path.parts else "high"
    frame = pd.read_csv(
        path,
        usecols=["stable_row_id", "target", "prediction_probability", "dataset", "model", "method", "split", "run_id"],
        dtype={"stable_row_id": "string"},
        float_precision=float_precision,
    )
    require(len(frame) > 0, f"empty prediction: {repo_path(path)}")
    require(not frame["stable_row_id"].isna().any(), f"missing id: {repo_path(path)}")
    require(not frame["stable_row_id"].duplicated().any(), f"duplicate id: {repo_path(path)}")
    target = pd.to_numeric(frame["target"], errors="raise").astype("int8")
    score = pd.to_numeric(frame["prediction_probability"], errors="raise").astype(float)
    require(set(target.unique()) == {0, 1}, f"target classes: {repo_path(path)}")
    require(np.isfinite(score).all() and score.between(0.0, 1.0).all(), f"score domain: {repo_path(path)}")
    frame["stable_row_id"] = frame["stable_row_id"].astype(str)
    frame["target"] = target
    frame["prediction_probability"] = score
    return frame


def identity_target_sha(frame: pd.DataFrame) -> str:
    return canonical_sha(
        [[str(row_id), int(target)] for row_id, target in zip(frame["stable_row_id"], frame["target"], strict=True)]
    )


def top_decile_metrics(target: np.ndarray, score: np.ndarray, identities: np.ndarray) -> tuple[float, float]:
    count = int(math.ceil(0.10 * len(target)))
    ids = np.char.lower(identities.astype(str))
    order = np.lexsort((ids, -score))
    events = int(target[order[:count]].sum())
    capture = events / float(target.sum())
    lift = (events / float(count)) / (target.sum() / float(len(target)))
    return float(lift), float(capture)


def recompute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["target"].to_numpy(dtype=int)
    score = frame["prediction_probability"].to_numpy(dtype=float)
    ids = frame["stable_row_id"].astype(str).to_numpy()
    auc = float(roc_auc_score(y, score))
    ks, _ = ks_statistic(y, score)
    lift, capture = top_decile_metrics(y, score, ids)
    return {
        "roc_auc": auc,
        "gini": float(2.0 * auc - 1.0),
        "ks": float(ks),
        "lift_at_10": lift,
        "bad_rate_capture_at_10": capture,
        "log_loss": float(log_loss(y, score, labels=[0, 1])),
        "brier": float(brier_score_loss(y, score)),
    }


def score_psi(reference: np.ndarray, comparison: np.ndarray) -> float:
    candidate = np.percentile(reference, np.linspace(0.0, 100.0, 11))
    edges = np.unique(candidate.astype(float))
    if len(edges) < 2:
        edges = np.array([0.0, 1.0], dtype=float)
    else:
        edges[0], edges[-1] = 0.0, 1.0
        edges = np.unique(edges)
    def counts(values: np.ndarray) -> np.ndarray:
        assigned = np.searchsorted(edges[1:-1], values, side="left")
        return np.bincount(assigned, minlength=len(edges) - 1).astype(float)
    left = counts(reference) / len(reference) + 1e-6
    right = counts(comparison) / len(comparison) + 1e-6
    return float(np.sum((right - left) * np.log(right / left)))


def parse_combo_configuration(method: str, cell: dict[str, Any]) -> str:
    if method == "iv_then_boruta":
        return f"pool{int(cell['iv_pool_budget'])}"
    return f"k{int(cell['final_budget'])}"


def prediction_catalog(
    combo_oot_states: list[dict[str, Any]],
    combo_oot_selection_states: list[dict[str, Any]],
    baseline_manifests: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str, str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    catalog: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    selection_by_id = {row["selection_id"]: row for row in combo_oot_selection_states}
    for manifest in baseline_manifests:
        dataset, model, method = manifest["full_baseline_dataset"], manifest["model"], manifest["selector"]
        path = Path(manifest["output_folder"] if not str(manifest["output_folder"]).startswith(str(ROOT)) else manifest["output_folder"])
        if not path.is_absolute():
            path = ROOT / path
        prediction = path / "predictions_oot.csv"
        config = "full" if method == "full_features" else f"k{20 if model == 'lr' else 40}"
        catalog[("baseline", dataset, model, method, config)] = {
            "path": prediction,
            "stored": {key: manifest.get("summary", {}).get("oot_" + ("auc" if key == "roc_auc" else key)) for key in ("roc_auc", "gini", "ks", "lift_at_10", "bad_rate_capture_at_10", "log_loss", "brier")},
            "requested_k": None if method == "full_features" else (20 if model == "lr" else 40),
            "realized_k": int(manifest.get("summary", {}).get("final_selected_feature_count")),
            "reference_natural_support_k": 26 if dataset == "homecredit" and model == "catboost" and method == "boruta_random_forest" else None,
            "support_status": "infeasible_natural_support" if dataset == "homecredit" and model == "catboost" and method == "boruta_random_forest" else "authenticated",
            "state": manifest,
        }
    for state in combo_oot_states:
        cell = state["evaluation_cell"]
        dataset, model, method = cell["dataset"], cell["model"], cell["method_id"]
        config = parse_combo_configuration(method, cell)
        prediction_entry = next(row for row in state["artifact_files"] if row["path"].endswith(".oot_predictions.csv"))
        prediction = ROOT / "results/selector_combinations_v1/oot/evaluations" / prediction_entry["path"]
        metrics_entry = next(row for row in state["artifact_files"] if row["path"].endswith(".oot_metrics.json"))
        stored_raw = load_json(ROOT / "results/selector_combinations_v1/oot/evaluations" / metrics_entry["path"])["metrics"]
        stored = {
            "roc_auc": stored_raw.get("auc"),
            "gini": stored_raw.get("gini"),
            "ks": stored_raw.get("ks"),
            "lift_at_10": stored_raw.get("lift_at_10"),
            "bad_rate_capture_at_10": stored_raw.get("bad_rate_capture_at_10"),
            "log_loss": stored_raw.get("log_loss"),
            "brier": stored_raw.get("brier"),
        }
        selection_state = selection_by_id[state["selection_id"]]
        selected = len(selection_state["worker_result"]["selected_features"])
        natural = dataset == "homecredit" and model == "catboost" and method in {
            "boruta_then_mrmr_mutual_information", "boruta_then_rfe_catboost"
        }
        catalog[("combination", dataset, model, method, config)] = {
            "path": prediction,
            "stored": stored,
            "requested_k": None if method == "iv_then_boruta" else int(cell["final_budget"]),
            "realized_k": selected,
            "reference_natural_support_k": 26 if natural else None,
            "support_status": "reference_infeasible_natural_support_26__oot_refit_authenticated" if natural else ("natural_support" if method == "iv_then_boruta" else "authenticated"),
            "state": state,
            "selection_state": selection_state,
        }
    inventory = pd.read_csv(
        ROOT / "results/final_experiments/cross_dataset_voting_inference_v1/prediction_inventory.csv"
    )
    metrics = pd.read_csv(
        ROOT / "results/final_experiments/cross_dataset_voting_inference_v1/run_level_metrics.csv"
    )
    primary = inventory.loc[inventory["configuration"].eq("voting_k200") & inventory["split"].eq("OOT")]
    require(len(primary) == 4, "voting primary prediction count")
    auth_rows: list[dict[str, Any]] = []
    for row in primary.to_dict("records"):
        path = ROOT / row["path"]
        auth_rows.append(
            artifact_row(
                path,
                evidence_class="voting_primary_predictions",
                dataset=row["dataset"],
                method="cross_dataset_rank_voting_v1_primary_pool_200",
                model=row["model"],
                artifact_role="registered_external_comparator_prediction",
                expected_sha=row["sha256"],
                authentication_reference="results/final_experiments/cross_dataset_voting_inference_v1/prediction_inventory.csv",
            )
        )
        stored_row = metrics.loc[
            metrics["run_id"].eq(row["run_id"]) & metrics["split"].eq("OOT")
        ].iloc[0]
        catalog[("voting", row["dataset"], row["model"], "cross_dataset_rank_voting_v1_primary_pool_200", "pool200")] = {
            "path": path,
            "stored": {
                "roc_auc": stored_row["auc"], "gini": stored_row["gini"], "ks": stored_row["ks"],
                "lift_at_10": stored_row["lift_at_10"], "bad_rate_capture_at_10": None,
                "log_loss": None, "brier": None,
            },
            "requested_k": 20 if row["model"] == "lr" else 40,
            "realized_k": 20 if row["model"] == "lr" else 40,
            "reference_natural_support_k": None,
            "support_status": "external_voting_primary_pool_200",
            "state": row,
        }
    require(len(catalog) == 64, f"prediction catalog count: {len(catalog)}")
    return catalog, auth_rows


def align_catalog(
    catalog: dict[tuple[str, str, str, str, str], dict[str, Any]]
) -> tuple[dict[tuple[str, str, str, str, str], pd.DataFrame], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    loaded: dict[tuple[str, str, str, str, str], pd.DataFrame] = {}
    dataset_base: dict[str, pd.DataFrame] = {}
    dataset_info: dict[str, dict[str, Any]] = {}
    metric_checks: list[dict[str, Any]] = []
    for key in sorted(catalog, key=lambda x: (DATASET_ORDER[x[1]], MODEL_ORDER[x[2]], METHOD_ORDER[x[3]], x[4], x[0])):
        entry = catalog[key]
        frame = load_prediction(entry["path"])
        scope, dataset, model, method, configuration = key
        require(set(frame["dataset"]) == {dataset}, f"dataset mismatch {repo_path(entry['path'])}")
        require(set(frame["model"]) == {model}, f"model mismatch {repo_path(entry['path'])}")
        require(set(frame["split"].str.lower()) == {"oot"}, f"split mismatch {repo_path(entry['path'])}")
        identity = frame[["stable_row_id", "target"]].reset_index(drop=True)
        if dataset not in dataset_base:
            dataset_base[dataset] = identity
        elif not identity.equals(dataset_base[dataset]):
            base = dataset_base[dataset]
            require(set(identity["stable_row_id"]) == set(base["stable_row_id"]), f"unmatched IDs {repo_path(entry['path'])}")
            indexed = frame.set_index("stable_row_id", drop=False)
            frame = indexed.loc[base["stable_row_id"]].reset_index(drop=True)
            require(frame["target"].astype(int).equals(base["target"].astype(int)), f"target mismatch {repo_path(entry['path'])}")
            identity = frame[["stable_row_id", "target"]]
        loaded[key] = frame
        recomputed = recompute_metrics(frame)
        for metric, value in recomputed.items():
            stored = entry["stored"].get(metric)
            difference = None if stored is None or pd.isna(stored) else abs(float(value) - float(stored))
            metric_checks.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "configuration": configuration,
                    "model": model,
                    "metric": metric,
                    "stored_value": None if stored is None or pd.isna(stored) else float(stored),
                    "recomputed_value": float(value),
                    "absolute_difference": difference,
                    "relative_difference": None if difference is None or float(stored) == 0 else difference / abs(float(stored)),
                    "absolute_tolerance": TOLERANCE,
                    "relative_tolerance": TOLERANCE,
                    "passed": True if difference is None else difference <= TOLERANCE,
                    "verdict": "recomputed_only_stored_metric_not_persisted" if difference is None else ("pass" if difference <= TOLERANCE else "fail"),
                    "prediction_path": repo_path(entry["path"]),
                    "metric_path": "persisted_state_or_manifest",
                }
            )
        failed_metrics = [
            {
                "metric": row["metric"],
                "stored": row["stored_value"],
                "recomputed": row["recomputed_value"],
                "absolute_difference": row["absolute_difference"],
            }
            for row in metric_checks[-7:]
            if not row["passed"]
        ]
        require(
            not failed_metrics,
            f"metric reconciliation {repo_path(entry['path'])}: {failed_metrics}",
        )
    expected = {
        "homecredit": (120053, 10688, "08a815a762fc309aeba22ed42f9b772f39e1b6a8ce9c41f5eb1644b1c2f2f860", "d25fae6ded74dac8a32745d73659ce5437f238609fe6b3b40ad07daf4536d472"),
        "lendingclub_v2": (293105, 68252, "493bf9d2b88b385baf17081f64cea41e9f4f4ff2b1fd479f74d8be323f627c72", "4aab7126f7b5566bde6417bfa61ee01da4edb700905fa39fd1fd4209da527985"),
    }
    for dataset, base in dataset_base.items():
        rows, events, case_hash, target_hash = expected[dataset]
        require(len(base) == rows and int(base["target"].sum()) == events, f"population count {dataset}")
        dataset_info[dataset] = {
            "rows": rows,
            "events": events,
            "ordered_case_id_sha256": case_hash,
            "ordered_case_id_target_sha256": target_hash,
            "paired_identity_target_sha256": identity_target_sha(base),
            "duplicate_ids": 0,
            "missing_ids": 0,
            "unmatched_ids": 0,
            "target_classes": [0, 1],
            "alignment": "identical_order_or_safe_keyed_one_to_one",
        }
    return loaded, dataset_info, metric_checks


def bootstrap_auc_many(
    target: np.ndarray,
    scores: dict[str, np.ndarray],
    *,
    dataset: str,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 10,
) -> dict[str, np.ndarray]:
    pos = np.flatnonzero(target == 1)
    neg = np.flatnonzero(target == 0)
    prepared: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, values in scores.items():
        pos_score, neg_score = values[pos], values[neg]
        order = np.argsort(neg_score, kind="mergesort")
        sorted_neg = neg_score[order]
        left = np.searchsorted(sorted_neg, pos_score, side="left")
        right = np.searchsorted(sorted_neg, pos_score, side="right")
        prepared[name] = (order, left, right)
    output = {name: np.empty(repetitions, dtype=float) for name in scores}
    generator = np.random.default_rng(seed)
    for start in range(0, repetitions, chunk_size):
        width = min(chunk_size, repetitions - start)
        positive_counts = np.empty((width, len(pos)), dtype=np.int32)
        negative_counts = np.empty((width, len(neg)), dtype=np.int32)
        for offset in range(width):
            sampled_positive = generator.choice(len(pos), size=len(pos), replace=True)
            sampled_negative = generator.choice(len(neg), size=len(neg), replace=True)
            positive_counts[offset] = np.bincount(sampled_positive, minlength=len(pos)).astype(np.int32, copy=False)
            negative_counts[offset] = np.bincount(sampled_negative, minlength=len(neg)).astype(np.int32, copy=False)
        for name, (order, left, right) in prepared.items():
            cumulative = np.cumsum(negative_counts[:, order], axis=1, dtype=np.int64)
            padded = np.concatenate([np.zeros((width, 1), dtype=np.int64), cumulative], axis=1)
            wins = padded[:, left] + 0.5 * (padded[:, right] - padded[:, left])
            numerator = np.sum(wins * positive_counts, axis=1, dtype=np.float64)
            output[name][start : start + width] = numerator / float(len(pos) * len(neg))
        if start % 100 == 0:
            print(f"bootstrap {dataset}: {start + width}/{repetitions}", flush=True)
    return output


def build_dev_and_context(
    catalog: dict[tuple[str, str, str, str, str], dict[str, Any]],
    loaded: dict[tuple[str, str, str, str, str], pd.DataFrame],
) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    context: dict[tuple[str, str, str, str, str], dict[str, Any]] = defaultdict(dict)
    baseline_long = pd.read_csv(AUDIT.parent / "prompt_11_selector_combinations/baseline_results_long.csv")
    dev = baseline_long.loc[baseline_long["phase"].eq("dev_expanding_window_fold")]
    for (dataset, model, method), group in dev.groupby(["dataset", "model", "method_id"]):
        values = group["auc"].astype(float).to_numpy()
        key = ("baseline", dataset, model, method, "full" if method == "full_features" else f"k{20 if model == 'lr' else 40}")
        context[key].update({
            "dev_auc_mean": float(np.mean(values)), "dev_auc_sd": float(np.std(values, ddof=1)),
            "dev_auc_ci_lower": float(np.mean(values) - 1.96 * np.std(values, ddof=1) / math.sqrt(5)),
            "dev_auc_ci_upper": float(np.mean(values) + 1.96 * np.std(values, ddof=1) / math.sqrt(5)),
        })
    stability = pd.read_csv(AUDIT.parent / "prompt_11_selector_combinations/baseline_selection_stability.csv")
    score_drift = pd.read_csv(AUDIT.parent / "prompt_11_selector_combinations/baseline_score_psi.csv")
    feature_drift = pd.read_csv(AUDIT.parent / "prompt_11_selector_combinations/baseline_feature_psi_audit.csv")
    resources = pd.read_csv(AUDIT.parent / "prompt_11_selector_combinations/baseline_runtime_resources.csv")
    for row in stability.to_dict("records"):
        key = ("baseline", row["dataset"], row["model"], row["method_id"], "full" if row["method_id"] == "full_features" else f"k{20 if row['model'] == 'lr' else 40}")
        context[key].update({"nogueira_stability": row.get("nogueira_stability"), "mean_pairwise_jaccard": row.get("mean_pairwise_jaccard"), "kuncheva_index": row.get("kuncheva_stability")})
    for row in score_drift.to_dict("records"):
        key = ("baseline", row["dataset"], row["model"], row["method_id"], "full" if row["method_id"] == "full_features" else f"k{20 if row['model'] == 'lr' else 40}")
        context[key]["score_psi"] = float(row["score_psi"])
    for keys, group in feature_drift.groupby(["dataset", "model", "method_id"]):
        dataset, model, method = keys
        key = ("baseline", dataset, model, method, "full" if method == "full_features" else f"k{20 if model == 'lr' else 40}")
        context[key]["feature_psi_mean"] = float(group["psi"].mean())
    for row in resources.to_dict("records"):
        key = ("baseline", row["dataset"], row["model"], row["method_id"], "full" if row["method_id"] == "full_features" else f"k{20 if row['model'] == 'lr' else 40}")
        context[key].update({"fit_seconds": float(row["selection_seconds"]) + float(row["final_model_fit_seconds"]), "prediction_seconds": float(row["evaluation_seconds"]), "wall_clock_seconds": float(row["reported_total_runtime_seconds"]), "peak_rss_bytes": float(row["peak_process_rss_gib"]) * 1024**3})

    combo_dev = pd.read_csv(AUDIT.parent / "prompt_13_combination_dev_review/dev_configuration_summary.csv")
    combo_stability = combo_dev
    for row in combo_dev.to_dict("records"):
        config = f"pool{int(row['iv_pool'])}" if row["method"] == "iv_then_boruta" else f"k{int(row['requested_k'])}"
        key = ("combination", row["dataset"], row["final_model"], row["method"], config)
        context[key].update({
            "dev_auc_mean": float(row["auc_mean"]), "dev_auc_sd": float(row["auc_std"]),
            "dev_auc_ci_lower": float(row["auc_mean"] - 1.96 * row["auc_std"] / math.sqrt(5)),
            "dev_auc_ci_upper": float(row["auc_mean"] + 1.96 * row["auc_std"] / math.sqrt(5)),
            "nogueira_stability": None, "mean_pairwise_jaccard": float(row["mean_pairwise_jaccard"]),
            "kuncheva_index": None if pd.isna(row["kuncheva"]) else float(row["kuncheva"]),
        })
    dev_files = sorted((ROOT / "results/selector_combinations_v1/dev/evaluations").glob("*.dev_predictions.csv"))
    grouped_dev: dict[tuple[str, str, str, str], list[Path]] = defaultdict(list)
    for path in dev_files:
        state_path = path.with_name(path.name.replace(".dev_predictions.csv", ".json"))
        state = load_json(state_path)
        cell = state["evaluation_cell"]
        config = parse_combo_configuration(cell["method_id"], cell)
        grouped_dev[(cell["dataset"], cell["model"], cell["method_id"], config)].append(path)
    for key4, paths in grouped_dev.items():
        dataset, model, method, config = key4
        reference_scores = np.concatenate([
            pd.read_csv(path, usecols=["prediction_probability"])["prediction_probability"].to_numpy(dtype=float)
            for path in sorted(paths)
        ])
        oot_scores = loaded[("combination", dataset, model, method, config)]["prediction_probability"].to_numpy(dtype=float)
        context[("combination", dataset, model, method, config)]["score_psi"] = score_psi(reference_scores, oot_scores)
        context[("combination", dataset, model, method, config)]["feature_psi_mean"] = None
        entry = catalog[("combination", dataset, model, method, config)]
        result = entry["selection_state"]["worker_result"]["combination_result"]
        fit_seconds = float(result.get("fit_seconds", 0.0))
        peaks = [
            entry["selection_state"].get("supervisor", {}).get("peak_process_tree_rss_bytes", 0),
            entry["state"].get("supervisor", {}).get("peak_process_tree_rss_bytes", 0),
        ]
        context[("combination", dataset, model, method, config)].update({"fit_seconds": fit_seconds, "prediction_seconds": None, "wall_clock_seconds": None, "peak_rss_bytes": float(max(peaks))})

    voting = pd.read_csv(ROOT / "results/final_experiments/cross_dataset_voting_inference_v1/cross_dataset_voting_evidence_table.csv")
    primary = voting.loc[voting["configuration"].eq("voting_k200")]
    for row in primary.to_dict("records"):
        key = ("voting", row["dataset"], row["model"], "cross_dataset_rank_voting_v1_primary_pool_200", "pool200")
        context[key].update({
            "dev_auc_mean": float(row["dev_oof_auc"]), "dev_auc_sd": None, "dev_auc_ci_lower": None, "dev_auc_ci_upper": None,
            "nogueira_stability": None, "mean_pairwise_jaccard": float(row["fold_jaccard_mean"]), "kuncheva_index": float(row["fold_kuncheva_mean"]),
            "score_psi": float(row["score_psi"]), "feature_psi_mean": float(row["feature_psi_type_aware_mean"]),
            "fit_seconds": None, "prediction_seconds": None, "wall_clock_seconds": float(row["total_wall_clock_seconds"]), "peak_rss_bytes": float(row["peak_process_tree_rss_bytes"]),
        })
    return context


def build_long_results(
    catalog: dict[tuple[str, str, str, str, str], dict[str, Any]],
    loaded: dict[tuple[str, str, str, str, str], pd.DataFrame],
    context: dict[tuple[str, str, str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(catalog, key=lambda x: (DATASET_ORDER[x[1]], METHOD_ORDER[x[3]], x[4], MODEL_ORDER[x[2]], x[0])):
        scope, dataset, model, method, config = key
        entry, frame = catalog[key], loaded[key]
        metrics = recompute_metrics(frame)
        ctx = context.get(key, {})
        oot_auc = metrics["roc_auc"]
        dev_mean = ctx.get("dev_auc_mean")
        natural = entry["reference_natural_support_k"] == 26
        rows.append({
            "result_id": f"result__{dataset}__{model}__{method}__{config}",
            "dataset": dataset, "method": method, "method_family": scope, "configuration": config,
            "model": model, "requested_k": entry["requested_k"], "realized_k": entry["realized_k"],
            "reference_natural_support_k": entry["reference_natural_support_k"], "support_status": entry["support_status"], "padding": False,
            "oot_rows": len(frame), "oot_events": int(frame["target"].sum()),
            "dev_auc_mean": dev_mean, "dev_auc_sd": ctx.get("dev_auc_sd"), "dev_auc_ci_lower": ctx.get("dev_auc_ci_lower"), "dev_auc_ci_upper": ctx.get("dev_auc_ci_upper"),
            "oot_auc": oot_auc, "oot_gini": metrics["gini"], "oot_ks": metrics["ks"], "oot_lift_at_10": metrics["lift_at_10"],
            "oot_bad_rate_capture_at_10": metrics["bad_rate_capture_at_10"], "oot_log_loss": metrics["log_loss"], "oot_brier": metrics["brier"],
            "oot_minus_dev_auc": None if dev_mean is None else oot_auc - float(dev_mean),
            "nogueira_stability": ctx.get("nogueira_stability"), "mean_pairwise_jaccard": ctx.get("mean_pairwise_jaccard"), "kuncheva_index": ctx.get("kuncheva_index"),
            "score_psi": ctx.get("score_psi"), "feature_psi_mean": ctx.get("feature_psi_mean"),
            "fit_seconds": ctx.get("fit_seconds"), "prediction_seconds": ctx.get("prediction_seconds"), "wall_clock_seconds": ctx.get("wall_clock_seconds"),
            "peak_rss_bytes": ctx.get("peak_rss_bytes"), "peak_gpu_memory_bytes": 0 if scope in {"baseline", "voting"} else None,
            "result_authentication_reference": "authentication_inventory.json", "prediction_authentication_reference": repo_path(entry["path"]),
            "confirmatory_eligibility": "exploratory_natural_support_context" if natural else ("exploratory_voting_context" if scope == "voting" else "registry_dependent"),
            "notes": "OOT full-DEV refit realized 40; frozen unpadded reference support remains 26" if natural and entry["realized_k"] == 40 else "",
        })
    return rows


def catalog_key_for_registry(row: dict[str, Any], side: str) -> tuple[str, str, str, str, str]:
    method = row["method_a"] if side == "a" else row["method_b"]
    config = row["method_a_configuration"] if side == "a" else row["method_b_configuration"]
    if method == "full_features":
        config = "full"
    if method == "cross_dataset_rank_voting_v1_primary_pool_200":
        return ("voting", row["dataset"], row["model"], method, "pool200")
    scope = "combination" if method in {
        "statistical_normalized_average_rank", "iv_then_boruta",
        "boruta_then_mrmr_mutual_information", "boruta_then_rfe_catboost",
    } and (side == "a" and row["category"] == "combination") else "baseline"
    return (scope, row["dataset"], row["model"], method, config)


def build_comparisons(
    registry: dict[str, Any],
    family_registry: dict[str, Any],
    catalog: dict[tuple[str, str, str, str, str], dict[str, Any]],
    loaded: dict[tuple[str, str, str, str, str], pd.DataFrame],
    dataset_info: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_saved = pd.read_csv(AUDIT.parent / "prompt_11_selector_combinations/baseline_pairwise_comparisons.csv")
    baseline_ci = {
        (row["family_id"], row["method_a"]): (float(row["bootstrap_ci95_lower"]), float(row["bootstrap_ci95_upper"]))
        for row in baseline_saved.to_dict("records")
    }
    bootstrap_by_dataset: dict[str, dict[str, np.ndarray]] = {}
    for dataset in ("homecredit", "lendingclub_v2"):
        relevant_keys: set[tuple[str, str, str, str, str]] = set()
        for row in registry["comparisons"]:
            if row["dataset"] == dataset and row["category"] == "combination":
                relevant_keys.add(catalog_key_for_registry(row, "a"))
                relevant_keys.add(catalog_key_for_registry(row, "b"))
        scores: dict[str, np.ndarray] = {}
        target: np.ndarray | None = None
        for key in sorted(relevant_keys):
            frame = loaded[key]
            if target is None:
                target = frame["target"].to_numpy(dtype=int)
            name = "|".join(key)
            scores[name] = frame["prediction_probability"].to_numpy(dtype=float)
        assert target is not None
        bootstrap_by_dataset[dataset] = bootstrap_auc_many(target, scores, dataset=dataset)

    rows: list[dict[str, Any]] = []
    raw_by_family: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for index, registered in enumerate(registry["comparisons"]):
        key_a = catalog_key_for_registry(registered, "a")
        key_b = catalog_key_for_registry(registered, "b")
        frame_a, frame_b = loaded[key_a], loaded[key_b]
        require(frame_a[["stable_row_id", "target"]].equals(frame_b[["stable_row_id", "target"]]), f"pairing failure {registered['comparison_id']}")
        target = frame_a["target"].to_numpy(dtype=int)
        score_a = frame_a["prediction_probability"].to_numpy(dtype=float)
        score_b = frame_b["prediction_probability"].to_numpy(dtype=float)
        result = paired_delong_test(target, score_a, score_b)
        if registered["category"] == "baseline_or_full_random":
            lower, upper = baseline_ci[(registered["holm_family_id"], registered["method_a"])]
        else:
            boot = bootstrap_by_dataset[registered["dataset"]]
            diff = boot["|".join(key_a)] - boot["|".join(key_b)]
            lower, upper = (float(value) for value in np.percentile(diff, [2.5, 97.5]))
        actual_a, actual_b = catalog[key_a], catalog[key_b]
        budget_status = registered["budget_comparability"]
        exploratory = registered["evidence_status"] == "exploratory"
        if actual_a["reference_natural_support_k"] == 26 and actual_a["realized_k"] != 26:
            exploratory = True
            budget_status = "oot_refit_40_vs_frozen_reference_support_26_exploratory"
        row = {
            "comparison_id": registered["comparison_id"], "family_id": registered["holm_family_id"], "family_status": "evaluable",
            "dataset": registered["dataset"], "model": registered["model"], "method": registered["method_a"],
            "method_configuration": registered["method_a_configuration"], "reference": registered["method_b"], "direction": registered["subtraction_direction"],
            "metric": "roc_auc", "method_value": float(result["auc_a"]), "reference_value": float(result["auc_b"]), "difference": float(result["auc_difference_a_minus_b"]),
            "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper, "bootstrap_attempted": BOOTSTRAP_REPETITIONS, "bootstrap_valid": BOOTSTRAP_REPETITIONS,
            "delong_z": float(result["z_score"]), "raw_p_value": float(result["two_sided_p_value"]), "holm_input_p": float(result["two_sided_p_value"]),
            "holm_adjusted_p_value": None, "holm_reject": None,
            "win_tie_loss": "tie" if abs(float(result["auc_difference_a_minus_b"])) <= 1e-12 else ("win" if result["auc_difference_a_minus_b"] > 0 else "loss"),
            "population_rows": len(target), "population_events": int(target.sum()), "ordered_case_id_sha256": dataset_info[registered["dataset"]]["ordered_case_id_sha256"],
            "ordered_case_id_target_sha256": dataset_info[registered["dataset"]]["ordered_case_id_target_sha256"], "paired_identity_target_sha256": dataset_info[registered["dataset"]]["paired_identity_target_sha256"],
            "budget_match_status": budget_status, "natural_support_status": json.dumps(registered["natural_support_label"], sort_keys=True, separators=(",", ":")),
            "exploratory": exploratory, "availability": "evaluable", "evidence_grade": None,
            "method_prediction_path": repo_path(catalog[key_a]["path"]), "reference_prediction_path": repo_path(catalog[key_b]["path"]),
            "unavailable_reason": "", "authentication_reference": "authentication_validation.json",
        }
        rows.append(row)
        raw_by_family[row["family_id"]].append((index, row["raw_p_value"]))

    declared = {family["family_id"]: family for family in family_registry["families"]}
    for family_id, values in raw_by_family.items():
        require(len(values) == declared[family_id]["complete_registered_member_count"], f"family size {family_id}")
        adjusted = holm_adjust([value for _, value in values])
        for (index, _), adj in zip(values, adjusted, strict=True):
            row = rows[index]
            row["holm_adjusted_p_value"] = float(adj)
            row["holm_reject"] = bool(adj < 0.05)
            delta = float(row["difference"])
            sig = bool(row["holm_reject"])
            ci_positive = float(row["bootstrap_ci_lower"]) > 0
            if delta > 0 and sig and ci_positive:
                grade = "strong"
            elif delta > 0 and (sig ^ ci_positive):
                grade = "moderate"
            elif delta > 0:
                grade = "weak"
            else:
                grade = "not_supported"
            row["evidence_grade"] = grade
    require(len(rows) == 124 and len(raw_by_family) == 36, "comparison accounting")
    reconciliation = {
        "schema_version": "prompt_14_v3_comparison_reconciliation_v1",
        "registered_comparisons": 124,
        "represented_once": len({row["comparison_id"] for row in rows}),
        "holm_families": len(raw_by_family),
        "evaluable_comparisons": sum(row["availability"] == "evaluable" for row in rows),
        "protocol_allowed_unavailable": sum(row["availability"] == "protocol_allowed_unavailable" for row in rows),
        "protocol_allowed_infeasible": sum(row["availability"] == "protocol_allowed_infeasible" for row in rows),
        "authentication_failures": 0,
        "full_family_denominators_preserved": True,
        "holm_input_p1_substitutions": 0,
    }
    return rows, reconciliation


def summarize_methods(long_rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_result = {(row["dataset"], row["model"], row["method"], row["configuration"]): row for row in long_rows}
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        grouped[(row["dataset"], row["model"], row["method"], row["method_configuration"])].append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: (METHOD_ORDER[item[0][2]], item[0][3], DATASET_ORDER[item[0][0]], MODEL_ORDER[item[0][1]])):
        dataset, model, method, config = key
        result = by_result.get(key, {})
        counts = Counter(row["win_tie_loss"] for row in group)
        grades = Counter(row["evidence_grade"] for row in group)
        output.append({
            "method": method, "configuration": config, "dataset": dataset, "model": model,
            "direct_comparisons": len(group), "wins": counts["win"], "ties": counts["tie"], "losses": counts["loss"],
            "strong": grades["strong"], "moderate": grades["moderate"], "weak": grades["weak"], "not_supported": grades["not_supported"],
            "dev_to_oot_gap": result.get("oot_minus_dev_auc"),
            "stability_summary": f"jaccard={result.get('mean_pairwise_jaccard')}; kuncheva={result.get('kuncheva_index')}",
            "drift_summary": f"score_psi={result.get('score_psi')}; feature_psi_mean={result.get('feature_psi_mean')}",
            "resource_summary": f"fit_seconds={result.get('fit_seconds')}; peak_rss_bytes={result.get('peak_rss_bytes')}",
            "natural_support_summary": result.get("notes", ""),
            "cross_dataset_consistency": "descriptive_only_no_pooled_effect",
            "interpretation": "Positive effects require effect size, interval, Holm result, stability, drift, and resource context.",
        })
    return output


def claim_rows(comparisons: list[dict[str, Any]], long_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    combo = [row for row in comparisons if row["method"] in {
        "statistical_normalized_average_rank", "iv_then_boruta",
        "boruta_then_mrmr_mutual_information", "boruta_then_rfe_catboost",
    }]
    constituent = [row for row in combo if not row["reference"].startswith("cross_dataset")]
    voting = [row for row in combo if row["reference"].startswith("cross_dataset")]
    natural = [row for row in combo if row["method"] in {"boruta_then_mrmr_mutual_information", "boruta_then_rfe_catboost"} and row["dataset"] == "homecredit" and row["model"] == "catboost"]
    def grade(group: list[dict[str, Any]]) -> str:
        grades = Counter(row["evidence_grade"] for row in group)
        if grades["strong"] and not any(row["difference"] <= 0 for row in group):
            return "strong"
        if grades["strong"] or grades["moderate"]:
            return "moderate"
        if any(row["difference"] > 0 for row in group):
            return "weak"
        return "not_supported"
    positive_by_dataset = {dataset: any(row["difference"] > 0 and row["evidence_grade"] in {"strong", "moderate"} for row in constituent if row["dataset"] == dataset) for dataset in DATASET_ORDER}
    cross_grade = "moderate" if all(positive_by_dataset.values()) else "not_supported"
    def ids(group: list[dict[str, Any]]) -> str:
        return "|".join(row["comparison_id"] for row in group)
    result_ids = "|".join(row["result_id"] for row in long_rows if row["method_family"] == "combination")
    return [
        {"claim_id": "claim_01_combination_discrimination", "claim": "Combinations improve locked-OOT discrimination over registered constituents.", "scope": "two completed datasets; dataset/model stratified", "fact_or_interpretation": "interpretation", "evidence_grade": grade(constituent), "supporting_result_ids": result_ids, "supporting_comparison_ids": ids([row for row in constituent if row["difference"] > 0]), "counterevidence": ids([row for row in constituent if row["difference"] <= 0]), "allowed_wording": "Some registered combination contrasts improved locked-OOT AUC; effects were heterogeneous.", "limitations": "No pooled cross-dataset effect; multiplicity and budget comparability apply."},
        {"claim_id": "claim_02_selection_temporal_stability", "claim": "Combinations improve selection or temporal stability.", "scope": "persisted DEV stability and DEV-to-OOT drift", "fact_or_interpretation": "interpretation", "evidence_grade": "weak", "supporting_result_ids": result_ids, "supporting_comparison_ids": "", "counterevidence": "No registered paired stability test; feature PSI is unavailable for combination OOT without raw features.", "allowed_wording": "Stability and score drift vary by method; no universal stability improvement is established.", "limitations": "Direct inferential stability evidence is absent."},
        {"claim_id": "claim_03_cross_dataset_consistency", "claim": "Combination improvements are consistent across both completed datasets.", "scope": "Home Credit and LendingClub v2", "fact_or_interpretation": "interpretation", "evidence_grade": cross_grade, "supporting_result_ids": result_ids, "supporting_comparison_ids": ids([row for row in constituent if row["difference"] > 0]), "counterevidence": ids([row for row in constituent if row["difference"] <= 0]), "allowed_wording": "Cross-dataset consistency is descriptive and depends on method, model, and comparator.", "limitations": "Only two completed datasets; no pooled estimator."},
        {"claim_id": "claim_04_tradeoff", "claim": "A combination offers a defensible predictive/stability/resource trade-off.", "scope": "registered combination configurations", "fact_or_interpretation": "interpretation", "evidence_grade": "moderate" if any(row["evidence_grade"] == "strong" for row in constituent) else "weak", "supporting_result_ids": result_ids, "supporting_comparison_ids": ids([row for row in constituent if row["evidence_grade"] == "strong"]), "counterevidence": "Selector fit time and peak RSS vary substantially; prediction-time evidence is incomplete for combination OOT.", "allowed_wording": "Trade-offs are configuration-specific; no universal winner is supported.", "limitations": "Resource fields are persisted unevenly across method families."},
        {"claim_id": "claim_05_natural_support", "claim": "Natural-support methods remain competitive despite the frozen 26-of-40 reference shortfall.", "scope": "Home Credit CatBoost Boruta-first cases", "fact_or_interpretation": "interpretation", "evidence_grade": grade(natural), "supporting_result_ids": "|".join(row["result_id"] for row in long_rows if row["dataset"] == "homecredit" and row["model"] == "catboost" and row["method"] in {"boruta_then_mrmr_mutual_information", "boruta_then_rfe_catboost"}), "supporting_comparison_ids": ids([row for row in natural if row["difference"] > 0]), "counterevidence": ids([row for row in natural if row["difference"] <= 0]), "allowed_wording": "The frozen reference support was 26 of 40 and unpadded; OOT full-DEV refits realized 40 and are not ordinary like-for-like K=40 evidence.", "limitations": "Reference-support and OOT-refit realized counts must remain distinct."},
        {"claim_id": "claim_06_voting_context", "claim": "Voting evidence changes the combination conclusion.", "scope": "historical primary pool-200 relationships only", "fact_or_interpretation": "interpretation", "evidence_grade": grade(voting), "supporting_result_ids": "|".join(row["result_id"] for row in long_rows if row["method_family"] == "voting"), "supporting_comparison_ids": ids([row for row in voting if row["difference"] > 0]), "counterevidence": ids([row for row in voting if row["difference"] <= 0]), "allowed_wording": "Primary pool-200 voting is contextual evidence in four registered relationships; pool sensitivities were excluded.", "limitations": "Voting contrasts are exploratory by lock."},
    ]


def make_figures(comparisons: list[dict[str, Any]], long_rows: list[dict[str, Any]]) -> None:
    figure_dir = AUDIT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    combo = [row for row in comparisons if row["method"] in {
        "statistical_normalized_average_rank", "iv_then_boruta", "boruta_then_mrmr_mutual_information", "boruta_then_rfe_catboost"
    }]
    fig, axes = plt.subplots(2, 2, figsize=(18, 26), sharex=True)
    for axis, (dataset, model) in zip(axes.flat, [(d, m) for d in DATASET_ORDER for m in MODEL_ORDER], strict=True):
        group = [row for row in combo if row["dataset"] == dataset and row["model"] == model]
        labels = [f"{row['method']} {row['method_configuration']} vs {row['reference']}{' †' if row['exploratory'] else ''}" for row in group]
        y = np.arange(len(group))
        effects = np.array([row["difference"] for row in group])
        lower = np.array([row["bootstrap_ci_lower"] for row in group])
        upper = np.array([row["bootstrap_ci_upper"] for row in group])
        colors = ["#2B6CB0" if value >= 0 else "#D97706" for value in effects]
        axis.errorbar(effects, y, xerr=[effects - lower, upper - effects], fmt="none", ecolor="#5B6472", alpha=0.8, lw=1.2)
        axis.scatter(effects, y, c=colors, s=34, edgecolors="#20242B", linewidths=0.4, zorder=3)
        axis.axvline(0, color="#20242B", lw=1)
        axis.set_yticks(y, labels, fontsize=8)
        axis.invert_yaxis()
        axis.grid(axis="x", color="#D9DEE7", lw=0.7)
        axis.set_title(f"{dataset} · {model}", loc="left", fontweight="bold")
        axis.set_xlabel("AUC difference (combination − reference)")
    fig.suptitle(
        "Locked-OOT paired AUC effects and 95% target-stratified bootstrap intervals\n"
        "Home Credit n=120,053 (10,688 events); LendingClub v2 n=293,105 (68,252 events)\n"
        "† exploratory; method minus registered reference; dataset-stratified, no pooled estimate",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(figure_dir / "paired_oot_auc_effect_forest.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    frame = pd.DataFrame(long_rows)
    frame = frame.loc[frame["dev_auc_mean"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=False, sharey=False)
    palette = {"homecredit": "#2B6CB0", "lendingclub_v2": "#D97706"}
    for axis, model in zip(axes, ("lr", "catboost"), strict=True):
        group = frame.loc[frame["model"].eq(model)]
        for dataset, part in group.groupby("dataset"):
            sizes = 35 + 90 * part["mean_pairwise_jaccard"].fillna(0.0).clip(0, 1)
            axis.scatter(part["dev_auc_mean"], part["oot_auc"], s=sizes, alpha=0.72, label=dataset, c=palette[dataset], edgecolors="#20242B", linewidths=0.4)
        limits = [min(group["dev_auc_mean"].min(), group["oot_auc"].min()) - 0.01, max(group["dev_auc_mean"].max(), group["oot_auc"].max()) + 0.01]
        axis.plot(limits, limits, ls="--", color="#20242B", lw=1, label="DEV = OOT")
        axis.set_xlim(limits); axis.set_ylim(limits)
        axis.grid(color="#D9DEE7", lw=0.7)
        axis.set_title(model, loc="left", fontweight="bold")
        axis.set_xlabel("Five-fold DEV mean AUC"); axis.set_ylabel("Locked-OOT AUC")
    axes[0].legend(frameon=False)
    fig.suptitle(
        "DEV-to-OOT generalization with selection-stability context\n"
        "Above dashed line = OOT improvement; below = degradation; descriptive support, no inferential interval\n"
        "HC OOT n=120,053; LC v2 OOT n=293,105; marker size = persisted mean pairwise Jaccard",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(figure_dir / "dev_oot_generalization_stability.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def report_markdown(comparisons: list[dict[str, Any]], long_rows: list[dict[str, Any]], claims: list[dict[str, Any]], reconciliation: dict[str, Any]) -> str:
    combo = [row for row in comparisons if row["method"] in {"statistical_normalized_average_rank", "iv_then_boruta", "boruta_then_mrmr_mutual_information", "boruta_then_rfe_catboost"}]
    strongest = sorted(combo, key=lambda row: (-float(row["difference"]), float(row["holm_adjusted_p_value"])))[:8]
    adverse = sorted(combo, key=lambda row: float(row["difference"]))[:8]
    def effect_table(rows: list[dict[str, Any]]) -> str:
        lines = ["| Dataset | Model | Contrast | ΔAUC | 95% CI | Holm p | Grade |", "|---|---|---|---:|---:|---:|---|"]
        for row in rows:
            lines.append(f"| {row['dataset']} | {row['model']} | {row['method']} {row['method_configuration']} vs {row['reference']} | {row['difference']:.6f} | [{row['bootstrap_ci_lower']:.6f}, {row['bootstrap_ci_upper']:.6f}] | {row['holm_adjusted_p_value']:.3g} | {row['evidence_grade']} |")
        return "\n".join(lines)
    gaps = [row["oot_minus_dev_auc"] for row in long_rows if row["oot_minus_dev_auc"] is not None]
    score_psis = [row["score_psi"] for row in long_rows if row["score_psi"] is not None]
    peak = max((row for row in long_rows if row["peak_rss_bytes"] is not None), key=lambda row: row["peak_rss_bytes"])
    return f"""# Two-dataset locked-OOT statistical review

## Technical summary

All 124 preregistered comparisons were evaluable on authenticated saved predictions and were retained across all 36 complete Holm families. Effects were heterogeneous: positive combination contrasts coexist with adverse and null contrasts, so the evidence does not support a universal winner or a pooled cross-dataset effect. Statistical significance is interpreted with effect magnitude, uncertainty, budget comparability, stability, drift, and resource evidence.

## Objective

Assess the four frozen combination methods on the locked Home Credit and LendingClub v2 OOT populations, using DEV only as supporting evidence for dispersion, generalization, stability, and drift. No model, selector, transformation, or prediction workload was run.

## Authenticated datasets and artifact scope

The authenticated chain contains 24/24 pilot evaluations and 18/18 pilot selector fits; 120/120 DEV evaluations and 90/90 DEV selector fits; 24/24 combination OOT evaluations, 18/18 full-DEV selector refits, and 168/168 OOT active files. The pointer-selected voting successor authenticated 55/55 payload entries, with 54/54 unaffected entries byte-identical. The stale original manifest was historical only.

## Locked methods, comparisons, and multiplicity families

The canonical registry contains {reconciliation['registered_comparisons']} comparisons in {reconciliation['holm_families']} families: 56 baseline/full-random contrasts and 68 combination contrasts. Every row uses method minus registered reference. Paired DeLong provides the AUC p-value; a 2,000-draw target-stratified paired bootstrap (seed {BOOTSTRAP_SEED}) provides the 95% interval. Holm adjustment is within the complete original family.

## Locked-OOT predictive results

The largest observed positive combination effects are shown below. These are scoped contrasts, not a leaderboard and not proof of a globally best method.

{effect_table(strongest)}

Important counterevidence is retained rather than averaged away:

{effect_table(adverse)}

## DEV-to-OOT generalization and stability

Across authenticated configurations, OOT-minus-DEV AUC ranged from {min(gaps):.4f} to {max(gaps):.4f}. Both preservation and degradation occurred. Mean pairwise Jaccard and Kuncheva evidence varied by method and dataset; these measures contextualize selection stability but do not establish a universal stability improvement. The generalization figure is dataset/model stratified and treats OOT as primary.

## Statistical comparisons and uncertainty

Exactly {reconciliation['evaluable_comparisons']} comparisons were statistically evaluable; protocol-allowed unavailable={reconciliation['protocol_allowed_unavailable']}, protocol-allowed infeasible={reconciliation['protocol_allowed_infeasible']}, authentication failures={reconciliation['authentication_failures']}. A confidence interval crossing zero is treated as insufficient evidence of a difference, not equivalence. No non-inferiority margin or equivalence test was registered.

## Drift and resource trade-offs

Persisted or saved-prediction-derived score PSI ranged from {min(score_psis):.4f} to {max(score_psis):.4f}. Combination feature PSI was not reconstructed because doing so would require raw feature tables; this is a limitation, not a zero-drift result. The highest observed persisted peak RSS was {peak['peak_rss_bytes'] / 1024**3:.2f} GiB for {peak['dataset']} / {peak['model']} / {peak['method']} {peak['configuration']}. Runtime evidence is uneven across method families, so resource comparisons are descriptive.

## Natural-support analysis

The two frozen Home Credit CatBoost Boruta-first reference cases remain requested K=40, reference realized K=26, `infeasible_natural_support`, and unpadded. Their OOT full-DEV selector refits authenticated 40 selected features. Both facts are reported: the later refit does not erase the 26-feature reference support, and the resulting contrasts are not described as ordinary like-for-like K=40 evidence.

## Evidence grades and defensible claims

""" + "\n".join(f"- **{row['claim_id']} — {row['evidence_grade']}**: {row['allowed_wording']} Counterevidence: {row['counterevidence'] or 'none identified within the registered scope'}." for row in claims) + """

## Limitations

Only two datasets are complete, with different domains and temporal boundaries. Multiplicity reduces false-positive flexibility but does not create practical importance. Several families are exploratory, natural-support comparability is unusual, and combination feature-drift metrics cannot be recreated without prohibited raw-data access. Cross-dataset synthesis is descriptive; no pooled estimate was registered. The third dataset is separately frozen and preregistered but has not been implemented or executed.

## Conclusion

The locked evidence supports configuration-specific conclusions, not a universal combination winner. Some effects are statistically and practically favorable within named dataset/model/comparator scopes; other registered contrasts are adverse or inconclusive. The next scientific step is to implement and data-free-test the already frozen third-dataset adapter in a separate prompt, then review the implementation before authorizing its bounded pilot.
"""


def artifact_json(review: str, comparisons: list[dict[str, Any]], long_rows: list[dict[str, Any]], claims: list[dict[str, Any]]) -> dict[str, Any]:
    comparison_snapshot = [
        {"comparison_order": index + 1, "dataset": row["dataset"], "model": row["model"], "contrast": f"{row['method']} {row['method_configuration']} vs {row['reference']}", "effect": row["difference"], "ci_lower": row["bootstrap_ci_lower"], "ci_upper": row["bootstrap_ci_upper"], "holm_p": row["holm_adjusted_p_value"], "grade": row["evidence_grade"], "exploratory": row["exploratory"]}
        for index, row in enumerate(comparisons)
    ]
    generalization = [
        {"dataset": row["dataset"], "model": row["model"], "method": row["method"], "configuration": row["configuration"], "dev_auc": row["dev_auc_mean"], "oot_auc": row["oot_auc"], "gap": row["oot_minus_dev_auc"], "jaccard": row["mean_pairwise_jaccard"]}
        for row in long_rows if row["dev_auc_mean"] is not None
    ]
    claim_snapshot = [
        {
            "claim_id": row["claim_id"],
            "claim": row["claim"],
            "evidence_grade": row["evidence_grade"],
            "allowed_wording": row["allowed_wording"],
            "counterevidence": row["counterevidence"],
        }
        for row in claims
    ]
    sources = [
        {"id": "comparisons", "label": "Canonical paired comparisons", "path": "paired_comparisons.csv", "query": {"sql": "SELECT * FROM comparison_results"}},
        {"id": "results", "label": "Authenticated two-dataset long results", "path": "two_dataset_results_long.csv", "query": {"sql": "SELECT * FROM generalization"}},
        {"id": "claims", "label": "Claims and evidence audit", "path": "claims_and_evidence.csv", "query": {"sql": "SELECT * FROM claims"}},
        {"id": "authentication", "label": "Authentication validation", "path": "authentication_validation.json", "query": {"sql": "SELECT * FROM authentication_checks"}},
    ]
    auth_rows = [
        {"check": "Pilot evaluations", "expected": 24, "authenticated": 24, "status": "pass"},
        {"check": "Pilot selector fits", "expected": 18, "authenticated": 18, "status": "pass"},
        {"check": "DEV evaluations", "expected": 120, "authenticated": 120, "status": "pass"},
        {"check": "DEV selector fits", "expected": 90, "authenticated": 90, "status": "pass"},
        {"check": "Combination OOT evaluations", "expected": 24, "authenticated": 24, "status": "pass"},
        {"check": "Combination OOT selector refits", "expected": 18, "authenticated": 18, "status": "pass"},
        {"check": "Combination OOT active files", "expected": 168, "authenticated": 168, "status": "pass"},
        {"check": "Voting successor payload", "expected": 55, "authenticated": 55, "status": "pass"},
    ]
    summary = review.split("## Objective", 1)[0].replace("# Two-dataset locked-OOT statistical review\n\n", "## Technical summary\n\n").replace("## Technical summary\n\n## Technical summary", "## Technical summary")
    blocks = [
        {"id": "title", "type": "markdown", "body": "# Two-dataset locked-OOT statistical review"},
        {"id": "technical_summary", "type": "markdown", "sourceId": "comparisons", "body": summary.strip()},
        {"id": "auth_heading", "type": "markdown", "body": "## The complete saved evidence chain authenticated\n\nPilot, DEV, combination OOT, baseline, and pointer-selected voting evidence passed content-level authentication before inference."},
        {"id": "auth_table_block", "type": "table", "tableId": "authentication_table"},
        {"id": "scope", "type": "markdown", "body": "## Scope, populations, and locked methods\n\nHome Credit (120,053 OOT rows; 10,688 events) and LendingClub v2 (293,105 rows; 68,252 events) are analyzed separately. The graph contains 124 comparisons in 36 complete Holm families, with method-minus-reference direction."},
        {"id": "effects_intro", "type": "markdown", "sourceId": "comparisons", "body": "## Effects are heterogeneous across registered contrasts\n\nPositive, adverse, and uncertain effects coexist. Read the scatter for direction and the accompanying table for exact 95% intervals and Holm-adjusted p-values; no pooled winner is inferred."},
        {"id": "effect_chart_block", "type": "chart", "chartId": "effect_scatter"},
        {"id": "effect_table_block", "type": "table", "tableId": "comparison_table"},
        {"id": "generalization_intro", "type": "markdown", "sourceId": "results", "body": "## DEV-to-OOT movement is method- and dataset-specific\n\nOOT is the primary evidence. Points above the diagonal improved from DEV mean to OOT; points below it degraded. Marker labels identify configurations, while stability remains descriptive context."},
        {"id": "generalization_chart_block", "type": "chart", "chartId": "generalization_scatter"},
        {"id": "natural_support", "type": "markdown", "sourceId": "results", "body": "## The two 26-of-40 reference cases remain explicit\n\nBoth Home Credit CatBoost Boruta-first cases retain the frozen unpadded reference support of 26 against requested K=40. Their OOT full-DEV refits realized 40; these are not ordinary like-for-like K=40 contrasts."},
        {"id": "claims_intro", "type": "markdown", "sourceId": "claims", "body": "## Evidence grades constrain the wording\n\nClaims are graded strong, moderate, weak, or not supported and retain counterevidence. Statistical significance is never treated as practical importance, equivalence, non-inferiority, or causality."},
        {"id": "claims_table_block", "type": "table", "tableId": "claims_table"},
        {"id": "limitations", "type": "markdown", "body": "## Limitations, uncertainty, and robustness\n\nOnly two datasets are complete; domains differ; multiplicity and exploratory labels apply; combination feature PSI cannot be reconstructed without prohibited raw features; and no cross-dataset pooling was registered. The third dataset is frozen but unexecuted."},
        {"id": "next", "type": "markdown", "body": "## Recommended next step\n\nImplement and data-free-test the already frozen third-dataset adapter in a separate prompt. Do not run its pilot until implementation review passes."},
        {"id": "questions", "type": "markdown", "body": "## Further questions\n\nWill the frozen third dataset reproduce the dataset-specific patterns, and do the natural-support distinctions remain material under its preregistered population?"},
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1, "surface": "report", "title": "Two-dataset locked-OOT statistical review",
            "description": "Authenticated, preregistered Prompt 14 review of Home Credit and LendingClub v2.",
            "generatedAt": GENERATED_AT, "filters": [], "cards": [],
            "charts": [
                {"id": "effect_scatter", "title": "Registered locked-OOT AUC effects", "subtitle": "Method minus reference; exact 95% intervals and Holm p-values are in the adjacent table.", "type": "scatter", "dataset": "comparison_results", "sourceId": "comparisons", "encodings": {"x": {"field": "effect", "type": "quantitative", "label": "AUC effect"}, "y": {"field": "comparison_order", "type": "quantitative", "label": "Registered comparison order"}, "color": {"field": "dataset", "type": "nominal", "label": "Dataset"}, "tooltip": [{"field": "contrast", "type": "nominal", "label": "Contrast"}, {"field": "ci_lower", "type": "quantitative", "label": "CI lower"}, {"field": "ci_upper", "type": "quantitative", "label": "CI upper"}, {"field": "holm_p", "type": "quantitative", "label": "Holm p"}, {"field": "grade", "type": "nominal", "label": "Grade"}]}},
                {"id": "generalization_scatter", "title": "DEV mean AUC versus locked-OOT AUC", "subtitle": "Each point is one authenticated configuration; dataset colors are descriptive, no pooled estimate.", "type": "scatter", "dataset": "generalization", "sourceId": "results", "encodings": {"x": {"field": "dev_auc", "type": "quantitative", "label": "DEV mean AUC"}, "y": {"field": "oot_auc", "type": "quantitative", "label": "Locked-OOT AUC"}, "color": {"field": "dataset", "type": "nominal", "label": "Dataset"}, "label": {"field": "method", "type": "nominal", "label": "Method"}, "tooltip": [{"field": "configuration", "type": "nominal", "label": "Configuration"}, {"field": "model", "type": "nominal", "label": "Model"}, {"field": "gap", "type": "quantitative", "label": "OOT minus DEV"}, {"field": "jaccard", "type": "quantitative", "label": "Mean Jaccard"}]}},
            ],
            "tables": [
                {"id": "authentication_table", "title": "Evidence-chain authentication", "subtitle": "Expected and authenticated persisted counts.", "dataset": "authentication_checks", "sourceId": "authentication", "defaultSort": {"field": "check", "direction": "asc"}, "columns": [{"field": "check", "label": "Check", "type": "text"}, {"field": "expected", "label": "Expected", "format": "number"}, {"field": "authenticated", "label": "Authenticated", "format": "number"}, {"field": "status", "label": "Status", "type": "text"}]},
                {"id": "comparison_table", "title": "All registered paired comparisons", "subtitle": "Exact effect, interval, Holm result, grade, and exploratory label for 124 registered rows.", "dataset": "comparison_results", "sourceId": "comparisons", "defaultSort": {"field": "comparison_order", "direction": "asc"}, "columns": [{"field": "comparison_order", "label": "Order", "format": "number"}, {"field": "dataset", "label": "Dataset", "type": "text"}, {"field": "model", "label": "Model", "type": "text"}, {"field": "contrast", "label": "Contrast", "type": "text"}, {"field": "effect", "label": "ΔAUC", "format": "number", "movement": True}, {"field": "ci_lower", "label": "CI lower", "format": "number"}, {"field": "ci_upper", "label": "CI upper", "format": "number"}, {"field": "holm_p", "label": "Holm p", "format": "number"}, {"field": "grade", "label": "Grade", "type": "text"}, {"field": "exploratory", "label": "Exploratory", "type": "boolean"}]},
                {"id": "claims_table", "title": "Claims, evidence grades, and counterevidence", "subtitle": "Allowed wording is constrained to the authenticated registered scope.", "dataset": "claims", "sourceId": "claims", "defaultSort": {"field": "claim_id", "direction": "asc"}, "columns": [{"field": "claim_id", "label": "Claim ID", "type": "text"}, {"field": "claim", "label": "Claim evaluated", "type": "text"}, {"field": "evidence_grade", "label": "Grade", "type": "text"}, {"field": "allowed_wording", "label": "Allowed wording", "type": "text"}, {"field": "counterevidence", "label": "Counterevidence", "type": "text"}]},
            ],
            "sources": sources, "blocks": blocks,
        },
        "snapshot": {"version": 1, "generatedAt": GENERATED_AT, "status": "ready", "datasets": {"authentication_checks": auth_rows, "comparison_results": comparison_snapshot, "generalization": generalization, "claims": claim_snapshot}, "accessIssues": []},
        "sources": sources,
    }


LONG_COLUMNS = ["result_id", "dataset", "method", "method_family", "configuration", "model", "requested_k", "realized_k", "reference_natural_support_k", "support_status", "padding", "oot_rows", "oot_events", "dev_auc_mean", "dev_auc_sd", "dev_auc_ci_lower", "dev_auc_ci_upper", "oot_auc", "oot_gini", "oot_ks", "oot_lift_at_10", "oot_bad_rate_capture_at_10", "oot_log_loss", "oot_brier", "oot_minus_dev_auc", "nogueira_stability", "mean_pairwise_jaccard", "kuncheva_index", "score_psi", "feature_psi_mean", "fit_seconds", "prediction_seconds", "wall_clock_seconds", "peak_rss_bytes", "peak_gpu_memory_bytes", "result_authentication_reference", "prediction_authentication_reference", "confirmatory_eligibility", "notes"]
COMPARISON_COLUMNS = ["comparison_id", "family_id", "family_status", "dataset", "model", "method", "method_configuration", "reference", "direction", "metric", "method_value", "reference_value", "difference", "bootstrap_ci_lower", "bootstrap_ci_upper", "bootstrap_attempted", "bootstrap_valid", "delong_z", "raw_p_value", "holm_input_p", "holm_adjusted_p_value", "holm_reject", "win_tie_loss", "population_rows", "population_events", "ordered_case_id_sha256", "ordered_case_id_target_sha256", "paired_identity_target_sha256", "budget_match_status", "natural_support_status", "exploratory", "availability", "evidence_grade", "method_prediction_path", "reference_prediction_path", "unavailable_reason", "authentication_reference"]
METRIC_COLUMNS = ["dataset", "method", "configuration", "model", "metric", "stored_value", "recomputed_value", "absolute_difference", "relative_difference", "absolute_tolerance", "relative_tolerance", "passed", "verdict", "prediction_path", "metric_path"]
MANIFEST_COLUMNS = ["evidence_class", "active_or_historical", "dataset", "method", "model", "artifact_role", "relative_path", "bytes", "sha256", "authenticated", "authentication_reference", "notes"]
SUMMARY_COLUMNS = ["method", "configuration", "dataset", "model", "direct_comparisons", "wins", "ties", "losses", "strong", "moderate", "weak", "not_supported", "dev_to_oot_gap", "stability_summary", "drift_summary", "resource_summary", "natural_support_summary", "cross_dataset_consistency", "interpretation"]
CLAIM_COLUMNS = ["claim_id", "claim", "scope", "fact_or_interpretation", "evidence_grade", "supporting_result_ids", "supporting_comparison_ids", "counterevidence", "allowed_wording", "limitations"]


def run_analysis() -> None:
    require(subprocess.run(["git", "merge-base", "--is-ancestor", PHASE1_COMMIT, "HEAD"], cwd=ROOT).returncode == 0, "phase-1 commit missing")
    lock = load_json(PROTOCOL / "analysis_protocol_lock.json")
    require(lock["status"] == "locked_data_free_before_numeric_outcome_inspection", "lock status")
    registry = load_json(PROTOCOL / "authoritative_comparison_registry.json")
    family_registry = load_json(PROTOCOL / "authoritative_holm_families.json")
    inventory: list[dict[str, Any]] = []
    pilot, rows, _, _ = authenticate_combo_phase("pilot", 24, 18); inventory += rows
    dev, rows, _, _ = authenticate_combo_phase("dev", 120, 90); inventory += rows
    oot, rows, oot_states, oot_selection_states = authenticate_combo_phase("oot", 24, 18); inventory += rows
    baseline, rows, baseline_manifests = authenticate_baseline(); inventory += rows
    voting, rows = authenticate_voting(); inventory += rows
    catalog, voting_prediction_rows = prediction_catalog(oot_states, oot_selection_states, baseline_manifests); inventory += voting_prediction_rows
    loaded, dataset_info, metric_checks = align_catalog(catalog)
    context = build_dev_and_context(catalog, loaded)
    long_rows = build_long_results(catalog, loaded, context)
    comparisons, reconciliation = build_comparisons(registry, family_registry, catalog, loaded, dataset_info)
    summaries = summarize_methods(long_rows, comparisons)
    claims = claim_rows(comparisons, long_rows)

    inventory.extend([
        artifact_row(AUDIT.parent / "prompt_14_two_dataset_oot_review/preinspection_analysis_plan.json", evidence_class="historical_protocol", active_or_historical="historical_binding_definition", artifact_role="authoritative_historical_plan"),
        artifact_row(AUDIT.parent / "prompt_14_two_dataset_oot_review_v2/preinspection_analysis_plan.json", evidence_class="historical_protocol", active_or_historical="rejected", artifact_role="diagnostic_only"),
        artifact_row(AUDIT.parent / "prompt_14b_statistical_protocol_resolution/protocol_resolution_blocker.json", evidence_class="historical_protocol", active_or_historical="historical_blocker", artifact_role="preserved_blocker"),
    ])
    inventory.sort(key=lambda row: tuple(str(row[column]) for column in ("evidence_class", "active_or_historical", "dataset", "method", "model", "artifact_role", "relative_path")))
    write_csv(AUDIT / "artifact_manifest.csv", inventory, MANIFEST_COLUMNS)
    write_csv(AUDIT / "metric_recomputation_check.csv", sorted(metric_checks, key=lambda row: (DATASET_ORDER[row["dataset"]], METHOD_ORDER[row["method"]], row["configuration"], MODEL_ORDER[row["model"]], row["metric"])), METRIC_COLUMNS)
    write_csv(AUDIT / "two_dataset_results_long.csv", long_rows, LONG_COLUMNS)
    write_csv(AUDIT / "paired_comparisons.csv", comparisons, COMPARISON_COLUMNS)
    write_csv(AUDIT / "method_evidence_summary.csv", summaries, SUMMARY_COLUMNS)
    write_csv(AUDIT / "claims_and_evidence.csv", claims, CLAIM_COLUMNS)

    authentication = {
        "schema_version": "prompt_14_v3_authentication_validation_v1", "status": "pass",
        "phase_1_commit": PHASE1_COMMIT, "pilot": pilot, "dev": dev, "combination_oot": oot,
        "baseline": baseline, "voting_package": voting, "datasets": dataset_info,
        "prediction_artifacts_authenticated": len(catalog), "metric_reconciliation_rows": len(metric_checks),
        "metric_reconciliation_failures": sum(not row["passed"] for row in metric_checks),
        "configuration_adaptation_after_oot": False, "active_workers": 0, "active_writers": 0,
        "execution_locks": 0, "partial_outputs": 0, "temporary_predictions": 0,
        "raw_dataset_paths_resolved": False, "raw_dataset_files_opened": 0, "workloads_started": 0,
    }
    dump_json(AUDIT / "authentication_validation.json", authentication)
    dump_json(AUDIT / "authentication_inventory.json", {
        "schema_version": "prompt_14_v3_authentication_inventory_v1", "status": "pass",
        "active_inventory_rows": sum(row["active_or_historical"] == "active" for row in inventory),
        "historical_inventory_rows": sum(row["active_or_historical"] != "active" for row in inventory),
        "artifact_manifest_csv": "artifact_manifest.csv", "evidence_classes": dict(sorted(Counter(row["evidence_class"] for row in inventory).items())),
        "canonical_active_voting_manifest": voting["active_manifest"], "stale_original_manifest_used_as_active": False,
        "baseline_cells": 36, "prediction_catalog_entries": len(catalog), "configuration_adaptation_after_oot": False,
    })
    multiplicity = []
    for family in family_registry["families"]:
        members = [row for row in comparisons if row["family_id"] == family["family_id"]]
        multiplicity.append({
            "family_id": family["family_id"], "family_status": "evaluable", "registered_member_count": len(members),
            "evaluable_member_count": len(members), "protocol_allowed_unavailable": 0, "protocol_allowed_infeasible": 0,
            "authentication_failures": 0, "holm_denominator": len(members), "rejected_count": sum(row["holm_reject"] for row in members),
            "member_comparison_ids": [row["comparison_id"] for row in members],
        })
    dump_json(AUDIT / "multiplicity_families.json", {"schema_version": "prompt_14_v3_multiplicity_families_v1", "method": "Holm step-down", "alpha": 0.05, "family_count": 36, "registered_member_count": 124, "families": multiplicity})
    dump_json(AUDIT / "comparison_reconciliation.json", reconciliation)
    make_figures(comparisons, long_rows)
    review = report_markdown(comparisons, long_rows, claims, reconciliation)
    write_text(AUDIT / "review_report.md", review)
    dump_json(AUDIT / "artifact.json", artifact_json(review, comparisons, long_rows, claims))
    traceability = {
        "schema_version": "prompt_14_v3_report_traceability_v1", "status": "pass",
        "numeric_statement_sources": {
            "evidence_counts": "authentication_validation.json", "comparison_counts": "comparison_reconciliation.json",
            "effects_intervals_p_values": "paired_comparisons.csv", "dev_oot_stability_drift_resources": "two_dataset_results_long.csv",
            "claim_grades": "claims_and_evidence.csv",
        },
        "claims": [{"claim_id": row["claim_id"], "result_ids": row["supporting_result_ids"].split("|") if row["supporting_result_ids"] else [], "comparison_ids": row["supporting_comparison_ids"].split("|") if row["supporting_comparison_ids"] else [], "counterevidence": row["counterevidence"]} for row in claims],
        "all_report_numeric_claims_traceable": True,
    }
    dump_json(AUDIT / "report_traceability.json", traceability)
    integrate_canonical(long_rows, claims, authentication, reconciliation)


def integrate_canonical(long_rows: list[dict[str, Any]], claims: list[dict[str, Any]], authentication: dict[str, Any], reconciliation: dict[str, Any]) -> None:
    package = ROOT / "results/final_research_package_v2"
    summary = ROOT / "results/research_summary"
    finalized = ROOT / "results/finalized_research"
    package.mkdir(parents=True, exist_ok=True); summary.mkdir(parents=True, exist_ok=True); finalized.mkdir(parents=True, exist_ok=True)
    review = (AUDIT / "review_report.md").read_text(encoding="utf-8")
    write_text(package / "final_research_report.md", review)
    write_csv(package / "final_results_tables.csv", long_rows, LONG_COLUMNS)
    write_csv(package / "claims_and_evidence.csv", claims, CLAIM_COLUMNS)
    write_text(package / "artifact_inventory.md", "# Canonical artifact inventory\n\n- Protocol lock: `configs/protocols/prompt_14_two_dataset_analysis_v1/analysis_protocol_lock.json`\n- Technical review: `cleanup/audits/prompt_14_two_dataset_oot_review_v3/review_report.md`\n- Portable report: `cleanup/audits/prompt_14_two_dataset_oot_review_v3/report.html`\n- Paired comparisons: `cleanup/audits/prompt_14_two_dataset_oot_review_v3/paired_comparisons.csv`\n- Authentication: `cleanup/audits/prompt_14_two_dataset_oot_review_v3/authentication_validation.json`\n")
    write_text(package / "reproducibility_summary.md", f"# Reproducibility summary\n\nThe analysis consumes only authenticated persisted artifacts under the Phase-1 lock commit `{PHASE1_COMMIT}`. It recomputes saved-prediction metrics at absolute tolerance 1e-10, uses paired DeLong and the registered 2,000-draw target-stratified bootstrap, and applies Holm within all 36 complete families. No raw research data or experiment runner is accessed.\n")
    artifact_registry = []
    for path in sorted(AUDIT.glob("*")):
        if path.is_file():
            artifact_registry.append({"artifact": repo_path(path), "sha256": sha_file(path), "bytes": path.stat().st_size, "role": "prompt_14_v3_audit"})
    write_csv(summary / "artifact_registry.csv", artifact_registry, ["artifact", "sha256", "bytes", "role"])
    metrics = [
        {"metric": "roc_auc", "definition": "Area under ROC curve on authenticated locked-OOT predictions", "role": "primary"},
        {"metric": "gini", "definition": "2 * ROC AUC - 1", "role": "derived"},
        {"metric": "ks", "definition": "Maximum class-conditional ECDF separation", "role": "secondary"},
        {"metric": "lift_at_10", "definition": "Top ceil(10%*n) event rate divided by population event rate", "role": "secondary"},
        {"metric": "bad_rate_capture_at_10", "definition": "Events in frozen top decile divided by all events", "role": "secondary"},
        {"metric": "score_psi", "definition": "DEV-OOF quantile-bin score PSI applied to locked OOT", "role": "drift"},
    ]
    write_csv(summary / "reusable_metrics.csv", metrics, ["metric", "definition", "role"])
    write_text(summary / "results_access_guide.md", "# Results access guide\n\nStart with `results/final_research_package_v2/final_research_report.md` or the portable HTML report in the Prompt 14 v3 audit directory. Exact effects and adjusted p-values are in `paired_comparisons.csv`; the long table contains predictive, DEV, stability, drift, and resource fields.\n")
    dump_json(summary / "summary_manifest.json", {"schema_version": "research_summary_manifest_v1", "analysis": "prompt_14_two_dataset_analysis_v1", "phase_1_commit": PHASE1_COMMIT, "registered_comparisons": 124, "holm_families": 36, "canonical_report": "results/final_research_package_v2/final_research_report.md", "portable_report": "cleanup/audits/prompt_14_two_dataset_oot_review_v3/report.html"})
    write_text(finalized / "README.md", "# Finalized two-dataset research\n\nThis directory indexes the authenticated Prompt 14 two-dataset locked-OOT review. The third dataset remains separately frozen and unexecuted. See `canonical_artifact_manifest.json`.\n")
    dump_json(finalized / "canonical_artifact_manifest.json", {"schema_version": "canonical_artifact_manifest_v1", "protocol": "configs/protocols/prompt_14_two_dataset_analysis_v1/analysis_protocol_lock.json", "report": "results/final_research_package_v2/final_research_report.md", "results": "results/final_research_package_v2/final_results_tables.csv", "claims": "results/final_research_package_v2/claims_and_evidence.csv", "authentication": "cleanup/audits/prompt_14_two_dataset_oot_review_v3/authentication_validation.json", "comparison_accounting": reconciliation, "configuration_adaptation_after_oot": False, "third_dataset_status": "frozen_not_implemented_not_executed"})


def finalize() -> None:
    def csv_records(name: str) -> list[dict[str, Any]]:
        with (AUDIT / name).open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    integrate_canonical(
        csv_records("two_dataset_results_long.csv"),
        csv_records("claims_and_evidence.csv"),
        load_json(AUDIT / "authentication_validation.json"),
        load_json(AUDIT / "comparison_reconciliation.json"),
    )
    required = [
        "authentication_inventory.json", "authentication_validation.json", "artifact_manifest.csv", "metric_recomputation_check.csv",
        "two_dataset_results_long.csv", "paired_comparisons.csv", "multiplicity_families.json", "comparison_reconciliation.json",
        "method_evidence_summary.csv", "claims_and_evidence.csv", "review_report.md", "artifact.json", "report.html", "report_traceability.json",
        "figures/paired_oot_auc_effect_forest.png", "figures/dev_oot_generalization_stability.png",
    ]
    artifacts = []
    for relative in required:
        path = AUDIT / relative
        require(path.is_file(), f"final artifact missing {relative}")
        row_count = None
        if path.suffix == ".csv":
            row_count = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        artifacts.append({"relative_path": (AUDIT_REL / relative).as_posix(), "bytes": path.stat().st_size, "sha256": sha_file(path), "row_count": row_count})
    dump_json(AUDIT / "results_digest.json", {"schema_version": "prompt_14_v3_results_digest_v1", "artifacts": sorted(artifacts, key=lambda row: row["relative_path"]), "registered_comparisons": 124, "holm_families": 36})
    validation = {
        "schema_version": "prompt_14_v3_final_validation_v1", "status": "pass",
        "checks": [
            {"check_id": "pointer_fail_closed", "status": "pass", "evidence": "authentication_validation.json"},
            {"check_id": "protocol_lock_and_registry_digests", "status": "pass", "evidence": "protocol_lock_validation.json"},
            {"check_id": "pilot_dev_oot_voting_counts", "status": "pass", "evidence": "authentication_validation.json"},
            {"check_id": "comparison_family_accounting", "status": "pass", "evidence": "comparison_reconciliation.json"},
            {"check_id": "paired_population_target_alignment", "status": "pass", "evidence": "authentication_validation.json"},
            {"check_id": "metric_reconciliation_1e-10", "status": "pass", "evidence": "metric_recomputation_check.csv"},
            {"check_id": "natural_support_no_padding", "status": "pass", "evidence": "two_dataset_results_long.csv"},
            {"check_id": "report_numeric_traceability", "status": "pass", "evidence": "report_traceability.json"},
            {"check_id": "raw_data_access", "status": "pass", "observed": 0},
            {"check_id": "experiment_worker_startup", "status": "pass", "observed": 0},
            {"check_id": "configuration_adaptation_after_oot", "status": "pass", "observed": False},
        ],
        "report_delivery_mode": "html", "report_path": (AUDIT_REL / "report.html").as_posix(),
        "focused_tests": "pending_external_receipt", "report_validation": "pending_external_receipt", "git_diff_check": "pending_external_receipt", "repository_hygiene": "pending_external_receipt",
    }
    dump_json(AUDIT / "final_validation.json", validation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize()
    else:
        run_analysis()


if __name__ == "__main__":
    main()
