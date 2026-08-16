"""Final, immutable Prompt-16 amended OOT execution and analysis.

This module deliberately separates the pre-OOT evidence freeze from the only
code path that can resolve the OOT slice.  The public ``build_freeze`` function
never opens matrix rows.  ``run_final_oot_worker`` is callable only through a
self-authenticated final authorization and promotes artifacts only after their
identity and complete inventory authenticate.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd

from credit_risk_fs.analysis.voting_inference.paired import (
    fast_paired_stratified_bootstrap,
)
from credit_risk_fs.analysis.voting_inference.psi import (
    feature_psi_record,
    score_psi_from_predictions,
    summarise_feature_psi,
)
from credit_risk_fs.data.homecredit_model_stability_2024.contract import (
    canonical_sha256,
    file_sha256,
)
from credit_risk_fs.evaluation.metrics import evaluate_model
from credit_risk_fs.evaluation.paired_inference import (
    align_paired_predictions,
    holm_adjust,
    paired_delong_test,
)
from credit_risk_fs.evaluation.stability import (
    mean_pairwise_jaccard,
    nogueira_stability,
)
from credit_risk_fs.experiments.atomic_io import (
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic,
)
from credit_risk_fs.experiments.prompt_16_llm_supplement import (
    EXPECTED_CLASSICAL_BYTE_COUNT,
    EXPECTED_CLASSICAL_FILE_COUNT,
    EXPECTED_CLASSICAL_TREE_SHA256,
    EXPECTED_MATRIX_MANIFEST_SHA256,
    EXPECTED_UNIVERSE_SHA256,
    FROZEN_FEATURE_BUDGETS,
    FROZEN_LLM_MODEL,
    FROZEN_LLM_RANKING_BUDGET,
    FROZEN_LLM_TEMPERATURE,
    _evaluation_identity as _supplemental_dev_evaluation_identity,
    _load_recursive_sealed,
    _ranking_identity,
    _seal_recursive_directory,
    _selection_identity as _supplemental_selection_identity,
    classical_evaluation_manifest_identity,
    classical_tree_identity,
    load_supplemental_amendment,
    load_supplemental_authorization,
    supplemental_cells,
)
from credit_risk_fs.experiments.prompt_16_third_dataset import (
    EXPECTED_PROTOCOL_FILE_SHA256,
    EXPECTED_PROTOCOL_INTERNAL_SHA256,
    NON_PREDICTORS,
    Prompt16ExecutionError,
    _archive_incomplete,
    _check_stop,
    _evaluation_identity as _classical_evaluation_identity,
    _expected_scope,
    _fit_and_evaluate,
    _fit_identity,
    _fit_one_selection,
    _json,
    _load_sealed,
    _locked_alignment_summary,
    _matrix_identity,
    _protocol_payload,
    _publish_stage,
    _read_date_slice,
    _validate_scope_frame,
    canonical_registry,
    selection_fit_registry,
)
from credit_risk_fs.preprocessing.encoding import OriginalFeatureNumericEncoder
from credit_risk_fs.selectors.stable_core_llm_fill import StableCoreLLMFillSelector


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "prompt_16_final_amended_oot_v1"
DEV_AUTH_SCHEMA_VERSION = "prompt_16_complete_amended_dev_authentication_v1"
AUTHORIZATION_SCHEMA_VERSION = "prompt_16_final_amended_oot_authorization_v1"
FREEZE_RELATIVE_ROOT = Path("cleanup/audits/prompt_16_final_amended_oot")
MEMORY_AMENDMENT_RELATIVE_ROOT = Path(
    "cleanup/audits/prompt_16_final_oot_memory_amendment_v1"
)
RESOURCE_BLOCKER_RELATIVE_ROOT = Path(
    "cleanup/audits/prompt_16_final_amended_oot_blocker_20260816"
)
CLASSICAL_DEV_RELATIVE_ROOT = Path(
    "results/prompt_16_homecredit_model_stability_2024/dev_v1"
)
SUPPLEMENTAL_DEV_RELATIVE_ROOT = Path(
    "results/prompt_16_homecredit_model_stability_2024/dev_llm_supplement_v3"
)
MATRIX_RELATIVE_ROOT = Path(
    "outputs/prompt_16_homecredit_model_stability_2024/matrix_v1"
)
OOT_RELATIVE_ROOT = Path(
    "results/prompt_16_homecredit_model_stability_2024/oot_final_amended_v1"
)
OOT_LOG_RELATIVE_ROOT = Path(
    "outputs/prompt_16_homecredit_model_stability_2024/logs/final_amended_oot_v1"
)
TEMP_RELATIVE_ROOT = Path(
    "outputs/prompt_16_homecredit_model_stability_2024/temp/final_amended_oot_v1"
)
PROTOCOL_RELATIVE_PATH = Path(
    "configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json"
)
SUPPLEMENTAL_AUTHORIZATION_RELATIVE_PATH = Path(
    "cleanup/audits/prompt_16_llm_scope_correction/"
    "supplemental_dev_execution_authorization_v2.json"
)
SUPPLEMENTAL_AMENDMENT_RELATIVE_PATH = Path(
    "configs/protocols/homecredit_model_stability_2024_v2/"
    "prompt_16_llm_supplement_amendment.json"
)
EXECUTION_POLICY_RELATIVE_PATH = Path(
    "configs/execution/prompt_16_final_oot_v1.yaml"
)
RAM_POLICY_RELATIVE_PATH = Path(
    "configs/execution/prompt_16_final_oot_ram_wait_v1.yaml"
)

REQUIRED_ANCESTORS = (
    "50fb1505fbae6e0e6c6c235b6e2019b362949bd1",
    "3b43c47f9e08621906f9c1171e7c7e3844337d59",
)
EXPECTED_SUPPLEMENTAL_TERMINAL = {
    "status": "completed",
    "stop_code": None,
    "peak_process_tree_rss_bytes": 17_491_939_328,
    "minimum_system_available_ram_bytes": 5_309_255_680,
}
METRIC_TOLERANCE = 1e-12
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_MINIMUM_VALID = 1900
BOOTSTRAP_SEED = 20260721
HOLM_ALPHA = 0.05
MAX_RESOURCE_RECOVERY_RESTARTS_PER_SCOPE = 5
MAX_ESTIMATOR_THREADS = 4
PROCESS_TREE_RSS_HARD_CAP_GIB = 24
SYSTEM_AVAILABLE_RAM_HARD_FLOOR_GIB = 4
SOFT_AVAILABLE_RAM_GIB = 6
RESUME_AVAILABLE_RAM_GIB = 8
RESUME_STABILITY_POLLS = 3
PREDECESSOR_EXECUTION_AUTHORIZATION_SHA256 = (
    "e9e2b15aa2a0b0330a027ad0414fd225ec3a07460dac501b08fa0faac8205f2a"
)
RESOURCE_BLOCKER_AUTHENTICATION_SHA256 = (
    "c5d60e918af91da47d0ff0ac4544b4c34e4abff2d89a340dcf9caa4fcf3fe4b6"
)
INHERITED_RESOURCE_INFEASIBLE_FIT_IDS = (
    "fit_009",
    "fit_010",
    "fit_015",
    "fit_016",
    "fit_019",
    "fit_020",
    "fit_024",
    "fit_025",
    "fit_026",
    "fit_027",
)
INHERITED_RESOURCE_INFEASIBLE_CELL_ORDERS = (1, 2)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Prompt16ExecutionError(f"unreadable JSON artifact: {path}") from exc


def _git(root: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and completed.returncode != 0:
        raise Prompt16ExecutionError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _assert_required_ancestry(root: Path) -> dict[str, Any]:
    branch = _git(root, "branch", "--show-current")
    if branch != "main":
        raise Prompt16ExecutionError(f"Prompt-16 final OOT requires main, found {branch}")
    head = _git(root, "rev-parse", "HEAD")
    for commit in REQUIRED_ANCESTORS:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            raise Prompt16ExecutionError(f"required commit is not an ancestor: {commit}")
    return {"branch": branch, "head": head, "required_ancestors": list(REQUIRED_ANCESTORS)}


def _assert_clean_worktree(root: Path) -> None:
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise Prompt16ExecutionError(
            "final OOT requires a clean tracked/untracked worktree; first entry is "
            + status.splitlines()[0]
        )


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _relative(path, root),
        "byte_size": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _recursive_tree_identity(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    byte_count = 0
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest.update(f"{relative}\t{size}\t{file_sha256(path)}\n".encode("utf-8"))
        count += 1
        byte_count += size
    return {
        "tree_manifest_sha256": digest.hexdigest(),
        "file_count": count,
        "byte_count": byte_count,
        "serialization": (
            "relative POSIX path, TAB, byte length, TAB, lowercase SHA-256, LF; "
            "paths sorted by relative path"
        ),
    }


def final_oot_cells(protocol_lock: str | Path) -> list[dict[str, Any]]:
    """Return the exact immutable 30 classical + four supplemental cells."""

    classical = list(canonical_registry(protocol_lock)["matrix_cells"])
    cells = [dict(cell) for cell in classical] + supplemental_cells()
    orders = [int(cell["configuration_order"]) for cell in cells]
    if orders != list(range(1, 35)):
        raise Prompt16ExecutionError("final OOT registry is not exactly ordered 1..34")
    if len(cells) != 34:
        raise Prompt16ExecutionError("final OOT registry does not contain 34 cells")
    return cells


def final_full_dev_refits(protocol_lock: str | Path) -> list[dict[str, Any]]:
    """Return 27 classical refits plus two supervised hybrid full-DEV refits."""

    _, protocol = _protocol_payload(protocol_lock)
    matrix = protocol["approved_protocol"]["method_and_evaluation_matrix"]
    refits = [
        {
            **dict(fit),
            "refit_order": int(fit["fit_order"]),
            "refit_id": str(fit["fit_id"]),
            "scope": "full_dev_only",
            "supervised": True,
            "internal_component_fits": 1,
        }
        for fit in selection_fit_registry(matrix)
    ]
    for model in ("lr", "catboost"):
        order = len(refits) + 1
        refits.append(
            {
                "refit_order": order,
                "refit_id": f"fit_{order:03d}_stable_core_llm_fill_{model}",
                "family": "llm_supplement",
                "method_id": "stable_core_llm_fill",
                "model_budget_owner": model,
                "requested_feature_budget": FROZEN_FEATURE_BUDGETS[model],
                "dependent_configuration_orders": [33 if model == "lr" else 34],
                "scope": "full_dev_only",
                "supervised": True,
                "internal_component_fits": 5,
            }
        )
    if len(refits) != 29 or sum(int(row["internal_component_fits"]) for row in refits) != 37:
        raise Prompt16ExecutionError("full-DEV refit accounting changed")
    return refits


def _method_cell(cells: Sequence[Mapping[str, Any]], method: str, model: str) -> dict[str, Any]:
    matches = [
        dict(cell)
        for cell in cells
        if cell.get("method_id") == method and cell.get("model") == model
    ]
    if len(matches) != 1:
        raise Prompt16ExecutionError(
            f"comparison method is not unique for model: {method}/{model}"
        )
    return matches[0]


def paired_comparison_graph(protocol_lock: str | Path) -> list[dict[str, Any]]:
    """Expand only the preregistered v1 and supplemental comparison edges."""

    cells = final_oot_cells(protocol_lock)
    rows: list[dict[str, Any]] = []

    def add(
        *,
        model: str,
        comparator: Mapping[str, Any],
        reference_method: str,
        family: str,
        availability: str = "registered",
    ) -> None:
        reference = (
            None
            if reference_method == "cross_dataset_rank_voting_v1"
            else _method_cell(cells, reference_method, model)
        )
        rows.append(
            {
                "comparison_order": len(rows) + 1,
                "comparison_id": f"p16-oot-comparison-{len(rows) + 1:03d}",
                "dataset": "homecredit_model_stability_2024",
                "model": model,
                "comparator_configuration_order": int(comparator["configuration_order"]),
                "comparator_method_id": comparator["method_id"],
                "reference_configuration_order": (
                    None if reference is None else int(reference["configuration_order"])
                ),
                "reference_method_id": reference_method,
                "holm_family_id": family,
                "direction": "comparator_minus_reference",
                "availability": availability,
                "paired_rows": "identical_frozen_oot_case_ids_and_targets",
            }
        )

    baselines = (
        "iv_woe",
        "mrmr_mutual_information",
        "lasso_l1_logistic",
        "legacy_rf_relevance_corr",
        "catboost_shap",
        "boruta_random_forest",
        "rfe_catboost",
    )
    for model in ("lr", "catboost"):
        for reference in ("full_features", "random_k"):
            family = f"baseline_vs_{reference}__homecredit_model_stability_2024__{model}"
            for method in baselines:
                add(
                    model=model,
                    comparator=_method_cell(cells, method, model),
                    reference_method=reference,
                    family=family,
                )

    for model in ("lr", "catboost"):
        for cell in [
            item
            for item in cells
            if item.get("method_id") == "iv_then_boruta" and item.get("model") == model
        ]:
            for reference in ("iv_woe", "boruta_random_forest"):
                add(
                    model=model,
                    comparator=cell,
                    reference_method=reference,
                    family=(
                        f"iv_then_boruta_pool_{cell['iv_pool_budget']}_vs_{reference}__"
                        f"homecredit_model_stability_2024__{model}"
                    ),
                )
        for method, references in (
            ("boruta_then_rfe_catboost", ("boruta_random_forest", "rfe_catboost")),
            (
                "boruta_then_mrmr_mutual_information",
                ("boruta_random_forest", "mrmr_mutual_information"),
            ),
        ):
            comparator = _method_cell(cells, method, model)
            for reference in references:
                add(
                    model=model,
                    comparator=comparator,
                    reference_method=reference,
                    family=(
                        f"{method}_vs_{reference}__homecredit_model_stability_2024__{model}"
                    ),
                )
        comparator = _method_cell(cells, "statistical_normalized_average_rank", model)
        for reference in (
            "iv_woe",
            "lasso_l1_logistic",
            "rfe_catboost",
            "boruta_random_forest",
            "catboost_shap",
            "mrmr_mutual_information",
            "cross_dataset_rank_voting_v1",
        ):
            add(
                model=model,
                comparator=comparator,
                reference_method=reference,
                family=(
                    "statistical_normalized_average_rank_vs_"
                    f"{reference}__homecredit_model_stability_2024__{model}"
                ),
                availability=(
                    "unavailable_due_to_unresolved_historical_provenance"
                    if reference == "cross_dataset_rank_voting_v1"
                    else "registered"
                ),
            )

    for model in ("lr", "catboost"):
        for comparator, reference in (
            ("llm", "mrmr_mutual_information"),
            ("stable_core_llm_fill", "mrmr_mutual_information"),
            ("stable_core_llm_fill", "llm"),
            ("llm", "full_features"),
            ("stable_core_llm_fill", "full_features"),
        ):
            add(
                model=model,
                comparator=_method_cell(cells, comparator, model),
                reference_method=reference,
                family=f"third_dataset_llm_primary__{model}",
            )
    if len(rows) != 72 or sum(row["availability"] == "registered" for row in rows) != 70:
        raise Prompt16ExecutionError("paired comparison graph accounting changed")
    return rows


def _ranking_utility(target: Sequence[int], score: Sequence[float]) -> dict[str, float]:
    frame = pd.DataFrame({"target": target, "score": score}).sort_values(
        "score", ascending=False, kind="mergesort"
    )
    n_top = max(1, int(np.ceil(len(frame) * 0.1)))
    overall = float(frame["target"].mean())
    total = float(frame["target"].sum())
    top = frame.head(n_top)
    return {
        "lift_at_10": float(top["target"].mean() / overall) if overall else float("nan"),
        "bad_rate_capture_at_10": float(top["target"].sum() / total)
        if total
        else float("nan"),
    }


def _reconcile_prediction_metrics(
    prediction_path: Path,
    metrics_path: Path,
    *,
    expected_alignment: Mapping[str, Any],
) -> dict[str, Any]:
    predictions = pd.read_parquet(prediction_path)
    required = {
        "case_id",
        "target",
        "score",
        "decision_threshold",
        "predicted_default",
    }
    if required - set(predictions.columns):
        raise Prompt16ExecutionError(
            f"prediction schema is incomplete: {prediction_path}"
        )
    if predictions["case_id"].duplicated().any():
        raise Prompt16ExecutionError(f"prediction case IDs duplicate: {prediction_path}")
    if not predictions["target"].isin([0, 1]).all():
        raise Prompt16ExecutionError(f"prediction target orientation changed: {prediction_path}")
    score = pd.to_numeric(predictions["score"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
        raise Prompt16ExecutionError(f"prediction scores are invalid: {prediction_path}")
    observed_alignment = _locked_alignment_summary(
        predictions["case_id"].tolist(), predictions["target"].tolist()
    )
    for key, expected_key in (
        ("row_count", "rows"),
        ("ordered_case_id_sha256", "ordered_case_id_sha256"),
        ("ordered_case_id_target_sha256", "ordered_case_id_target_sha256"),
    ):
        if observed_alignment[key] != expected_alignment[expected_key]:
            raise Prompt16ExecutionError(
                f"prediction alignment mismatch ({key}): {prediction_path}"
            )
    thresholds = pd.to_numeric(
        predictions["decision_threshold"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(thresholds).all() or len(np.unique(thresholds)) != 1:
        raise Prompt16ExecutionError(f"prediction threshold is not frozen: {prediction_path}")
    expected_predicted = (score >= float(thresholds[0])).astype("int8")
    if not np.array_equal(
        expected_predicted,
        predictions["predicted_default"].to_numpy(dtype="int8"),
    ):
        raise Prompt16ExecutionError(f"predicted class threshold mismatch: {prediction_path}")
    stored = _read_json(metrics_path)
    recomputed = evaluate_model(
        predictions["target"].to_numpy(dtype=int),
        score,
        threshold=float(thresholds[0]),
    )
    recomputed.update(_ranking_utility(predictions["target"].tolist(), score.tolist()))
    if set(stored) != set(recomputed):
        raise Prompt16ExecutionError(f"metric key registry mismatch: {metrics_path}")
    differences: dict[str, float] = {}
    for key, expected in recomputed.items():
        observed = stored[key]
        left = float(expected)
        right = float(observed)
        difference = 0.0 if math.isnan(left) and math.isnan(right) else abs(left - right)
        differences[key] = difference
        if difference > METRIC_TOLERANCE:
            raise Prompt16ExecutionError(
                f"saved metric differs from prediction ({key}={difference}): {metrics_path}"
            )
    return {
        "prediction_sha256": file_sha256(prediction_path),
        "metrics_sha256": file_sha256(metrics_path),
        "rows": len(predictions),
        "threshold": float(thresholds[0]),
        "maximum_absolute_metric_difference": max(differences.values(), default=0.0),
        "metric_tolerance": METRIC_TOLERANCE,
        "alignment": observed_alignment,
    }


def _assert_scope_authentication(
    path: Path,
    *,
    train_expected: Mapping[str, Any],
    validation_expected: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _read_json(path)
    for label, expected in (("train", train_expected), ("validation", validation_expected)):
        declared = payload.get(label, {})
        if declared.get("authenticated") is not True:
            raise Prompt16ExecutionError(f"scope is not authenticated: {path}/{label}")
        checks = {
            "rows": int(expected["rows"]),
            "target_0": int(expected["target_0"]),
            "target_1": int(expected["target_1"]),
            "ordered_case_id_sha256": str(expected["ordered_case_id_sha256"]),
            "ordered_case_id_target_sha256": str(
                expected["ordered_case_id_target_sha256"]
            ),
        }
        if declared.get("expected") != checks or declared.get("observed") != checks:
            raise Prompt16ExecutionError(f"scope row identity changed: {path}/{label}")
    if int(payload.get("case_id_overlap", -1)) != 0:
        raise Prompt16ExecutionError(f"train/validation overlap is nonzero: {path}")
    return payload


def _selection_stability_row(
    *,
    order: int,
    cell: Mapping[str, Any],
    fold_sets: Sequence[set[str]],
    unavailable_folds: Sequence[int],
    candidate_count: int,
) -> dict[str, Any]:
    return {
        "configuration_order": order,
        "method_id": cell["method_id"],
        "model": cell["model"],
        "authenticated_selection_fold_count": len(fold_sets),
        "unavailable_selection_folds": list(unavailable_folds),
        "candidate_universe_count": candidate_count,
        "mean_pairwise_jaccard": (
            None if len(fold_sets) < 2 else float(mean_pairwise_jaccard(list(fold_sets)))
        ),
        "nogueira_stability": (
            None
            if len(fold_sets) < 2
            else float(nogueira_stability(list(fold_sets), candidate_count))
        ),
        "interpretation": "descriptive_only_no_significance_label",
    }


def authenticate_complete_dev(
    repository_root: str | Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Authenticate all 170 identities and recompute all 123 available outcomes."""

    root = Path(repository_root).resolve()
    protocol_path = root / PROTOCOL_RELATIVE_PATH
    classical_root = root / CLASSICAL_DEV_RELATIVE_ROOT
    supplement_root = root / SUPPLEMENTAL_DEV_RELATIVE_ROOT
    matrix_root = root / MATRIX_RELATIVE_ROOT
    _, protocol = _protocol_payload(protocol_path)
    matrix = protocol["approved_protocol"]["method_and_evaluation_matrix"]
    matrix_manifest, metadata = _matrix_identity(matrix_root)
    del matrix_manifest
    matrix_manifest_sha = file_sha256(matrix_root / "manifest.json")
    if matrix_manifest_sha != EXPECTED_MATRIX_MANIFEST_SHA256:
        raise Prompt16ExecutionError("matrix manifest digest changed")
    predictors = list(metadata.get("predictor_columns", []))
    if len(predictors) != 1959 or canonical_sha256(predictors) != EXPECTED_UNIVERSE_SHA256:
        raise Prompt16ExecutionError("ordered 1,959-feature universe changed")

    classical_tree = classical_tree_identity(classical_root)
    if classical_tree["tree_manifest_sha256"] != EXPECTED_CLASSICAL_TREE_SHA256:
        raise Prompt16ExecutionError("classical DEV tree digest changed")
    if classical_tree["file_count"] != EXPECTED_CLASSICAL_FILE_COUNT:
        raise Prompt16ExecutionError("classical DEV file count changed")
    if classical_tree["byte_count"] != EXPECTED_CLASSICAL_BYTE_COUNT:
        raise Prompt16ExecutionError("classical DEV byte count changed")
    classical_manifest_registry = classical_evaluation_manifest_identity(classical_root)

    fits = selection_fit_registry(matrix)
    cells = list(matrix["matrix_cells"])
    fit_by_order = {
        int(order): fit
        for fit in fits
        for order in fit["dependent_configuration_orders"]
    }
    classical_rows: list[dict[str, Any]] = []
    thresholds: dict[int, list[tuple[int, float]]] = {order: [] for order in range(1, 35)}
    selections: dict[int, list[set[str]]] = {order: [] for order in range(1, 35)}
    selection_unavailable: dict[int, list[int]] = {order: [] for order in range(1, 35)}
    classical_fit_records = 0
    classical_metric_reconciliations = 0
    classical_complete = 0
    classical_unavailable = 0
    split = protocol["approved_protocol"]["split_and_fold_boundaries"]

    for fold_id in range(1, 6):
        fold_root = classical_root / f"fold_{fold_id}"
        fold_spec = next(
            item for item in split["folds"] if int(item["fold_id"]) == fold_id
        )
        if str(fold_spec["train"]["date_max"]) >= str(fold_spec["validation"]["date_min"]):
            raise Prompt16ExecutionError(f"fold {fold_id} temporal direction changed")
        scope = _assert_scope_authentication(
            fold_root / "scope_authentication.json",
            train_expected=fold_spec["train"],
            validation_expected=fold_spec["validation"],
        )
        if scope.get("matrix_manifest_sha256") != matrix_manifest_sha:
            raise Prompt16ExecutionError(f"fold {fold_id} matrix binding changed")
        for fit in fits:
            fit_path = fold_root / "selection_fits" / str(fit["fit_id"])
            identity = _fit_identity(
                phase="dev",
                fold_id=fold_id,
                fit=fit,
                matrix_manifest_sha256=matrix_manifest_sha,
            )
            if _load_sealed(fit_path, identity) is None:
                raise Prompt16ExecutionError(f"classical fit is not sealed: {fit_path}")
            classical_fit_records += 1

        for cell in cells:
            order = int(cell["configuration_order"])
            fit = fit_by_order[order]
            fit_path = fold_root / "selection_fits" / str(fit["fit_id"])
            selection_manifest_sha = file_sha256(fit_path / "manifest.json")
            evaluation_path = fold_root / "evaluations" / f"cell_{order:03d}"
            identity = _classical_evaluation_identity(
                phase="dev",
                fold_id=fold_id,
                cell=cell,
                matrix_manifest_sha256=matrix_manifest_sha,
                selection_manifest_sha256=selection_manifest_sha,
            )
            if _load_sealed(evaluation_path, identity) is None:
                raise Prompt16ExecutionError(
                    f"classical evaluation is not sealed: {evaluation_path}"
                )
            selection = _read_json(fit_path / "selection.json")
            selected = [str(value) for value in selection.get("selected_features", [])]
            if len(selected) != len(set(selected)) or not set(selected).issubset(predictors):
                raise Prompt16ExecutionError(f"invalid classical selection: {fit_path}")
            if selected:
                selections[order].append(set(selected))
            else:
                selection_unavailable[order].append(fold_id)
            status = _read_json(evaluation_path / "status.json")
            base = {
                "evaluation_id": f"p16v1-dev-fold-{fold_id}-c{order:03d}",
                "source": "preserved_classical",
                "fold_id": fold_id,
                "configuration_order": order,
                "method_id": cell["method_id"],
                "model": cell["model"],
                "fit_id": fit["fit_id"],
                "status": status.get("status"),
                "reason": status.get("reason"),
                "selection_manifest_sha256": selection_manifest_sha,
                "evaluation_manifest_sha256": file_sha256(
                    evaluation_path / "manifest.json"
                ),
                "selected_feature_count": len(selected),
                "selected_features_sha256": canonical_sha256(selected),
            }
            if status.get("status") == "complete":
                reconciliation = _reconcile_prediction_metrics(
                    evaluation_path / "predictions.parquet",
                    evaluation_path / "metrics.json",
                    expected_alignment=fold_spec["validation"],
                )
                execution = _read_json(evaluation_path / "execution.json")
                configuration = execution.get("configuration", {})
                if configuration.get("probability_orientation") != "class_1_higher_default_risk":
                    raise Prompt16ExecutionError("classical probability orientation changed")
                if configuration.get("validation_target_used_for_fit") is not False:
                    raise Prompt16ExecutionError("classical validation target leaked into fit")
                if configuration.get("preprocessing", {}).get("fit_scope") != "dev_fold_training_only":
                    raise Prompt16ExecutionError("classical preprocessing fit boundary changed")
                if int(configuration.get("selected_original_feature_count", -1)) != len(selected):
                    raise Prompt16ExecutionError("classical selected-feature count mismatch")
                thresholds[order].append((fold_id, reconciliation["threshold"]))
                base.update(reconciliation)
                classical_complete += 1
                classical_metric_reconciliations += 1
            else:
                allowed = {
                    ("unavailable", "selector_resource_infeasible"),
                    ("unavailable", "resource_infeasible"),
                }
                if (status.get("status"), status.get("reason")) not in allowed:
                    raise Prompt16ExecutionError(
                        f"unregistered classical unavailable status: {evaluation_path}"
                    )
                if (evaluation_path / "predictions.parquet").exists() or (
                    evaluation_path / "metrics.json"
                ).exists():
                    raise Prompt16ExecutionError(
                        f"unavailable classical cell has outcome artifact: {evaluation_path}"
                    )
                classical_unavailable += 1
            classical_rows.append(base)

    if classical_fit_records != 135 or len(classical_rows) != 150:
        raise Prompt16ExecutionError("classical DEV 135/150 accounting changed")
    if (classical_complete, classical_unavailable) != (103, 47):
        raise Prompt16ExecutionError("classical DEV complete/unavailable accounting changed")

    supplemental_authorization, amendment = load_supplemental_authorization(
        root / SUPPLEMENTAL_AUTHORIZATION_RELATIVE_PATH
    )
    controller_success = _read_json(supplement_root / "_SUCCESS")
    controller_manifest = supplement_root / "controller_manifest.json"
    if controller_success.get("controller_manifest_sha256") != file_sha256(controller_manifest):
        raise Prompt16ExecutionError("supplemental controller completion marker mismatch")
    status_payload = _read_json(supplement_root / "controller_status.json")
    if status_payload.get("status") != "complete" or status_payload.get("oot_opened") is not False:
        raise Prompt16ExecutionError("supplemental controller is not a no-OOT completion")
    supervisor_path = (
        root
        / OOT_LOG_RELATIVE_ROOT.parent
        / "llm_supplement_v3"
        / "llm_supplement_v3_all_folds_dev_supervisor_summary.json"
    )
    supervisor = _read_json(supervisor_path)
    for key, value in EXPECTED_SUPPLEMENTAL_TERMINAL.items():
        if supervisor.get(key) != value:
            raise Prompt16ExecutionError(f"supplemental terminal evidence changed: {key}")
    if supervisor.get("return_value", {}).get("oot_started") is True:
        raise Prompt16ExecutionError("supplemental supervisor claims OOT started")

    availability = _read_json(supplement_root / "feature_availability_filter.json")
    retained = [str(value) for value in availability.get("retained_features", [])]
    dropped = [str(value) for value in availability.get("dropped_features", [])]
    freeze = availability.get("freeze", {})
    if (
        len(retained) != 1068
        or len(dropped) != 891
        or set(retained) & set(dropped)
        or set(retained) | set(dropped) != set(predictors)
    ):
        raise Prompt16ExecutionError("missingness filter does not partition all 1,959 features")
    if freeze.get("base_feature_universe_sha256") != EXPECTED_UNIVERSE_SHA256:
        raise Prompt16ExecutionError("missingness filter base universe changed")
    if freeze.get("target_or_validation_used") is not False:
        raise Prompt16ExecutionError("missingness filter used a target or validation")

    description_freeze = _read_json(supplement_root / "llm_ranking/provenance_freeze.json")
    if description_freeze != amendment.get("llm_provenance_freeze"):
        raise Prompt16ExecutionError("supplemental prompt provenance freeze changed")
    description_rows = _read_json(supplement_root / "llm_ranking/feature_descriptions.json")
    if len(description_rows) != 1068 or [row.get("name") for row in description_rows] != retained:
        raise Prompt16ExecutionError("LLM descriptions do not cover all eligible candidates")
    prompt_path = supplement_root / "llm_ranking/prompt.txt"
    rendered_prompt = prompt_path.read_text(encoding="utf-8")
    if hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest() != description_freeze.get(
        "rendered_prompt_sha256"
    ):
        raise Prompt16ExecutionError("rendered LLM prompt digest changed")

    entry = {
        "authorization": supplemental_authorization,
        "amendment": amendment,
        "matrix_manifest_sha256": matrix_manifest_sha,
        "description_freeze": {"freeze": description_freeze, "predictors": retained},
    }
    ranking_root = supplement_root / "llm_ranking"
    ranking_manifest = _load_recursive_sealed(ranking_root, _ranking_identity(entry))
    if ranking_manifest is None:
        raise Prompt16ExecutionError("authenticated LLM ranking is not recursively sealed")
    ranking_payload = _read_json(ranking_root / "ranking_payload.json")
    ranking = [str(value) for value in ranking_payload.get("selected_features", [])]
    if (
        len(ranking) != 100
        or len(ranking) != len(set(ranking))
        or not set(ranking).issubset(retained)
        or ranking_payload.get("fallback_used") is not False
        or ranking_payload.get("response_model") != FROZEN_LLM_MODEL
        or float(ranking_payload.get("temperature")) != FROZEN_LLM_TEMPERATURE
        or ranking_payload.get("candidate_coverage")
        != {
            "input_candidates": 1068,
            "ranked_features": 100,
            "unknown_features": 0,
            "duplicate_features": 0,
            "missing_required_rank_positions": 0,
        }
    ):
        raise Prompt16ExecutionError("LLM ranking parser/coverage contract changed")
    attempts = sorted((ranking_root / "attempts").glob("attempt_*"))
    if len(attempts) != 2:
        raise Prompt16ExecutionError("LLM application-attempt provenance changed")
    if _read_json(attempts[0] / "status.json").get("valid") is not False:
        raise Prompt16ExecutionError("LLM rejected hallucination attempt is missing")
    if _read_json(attempts[1] / "status.json").get("valid") is not True:
        raise Prompt16ExecutionError("LLM accepted parser attempt is missing")

    ranking_manifest_sha = file_sha256(ranking_root / "manifest.json")
    global_states: dict[str, tuple[Path, str]] = {}
    for model in ("lr", "catboost"):
        path = supplement_root / "shared_selection_states" / f"llm_{model}"
        identity = _supplemental_selection_identity(
            entry=entry,
            ranking_manifest_sha256=ranking_manifest_sha,
            method_id="llm",
            model=model,
            fold_id=None,
        )
        if _load_recursive_sealed(path, identity) is None:
            raise Prompt16ExecutionError(f"global LLM truncation state is not sealed: {model}")
        selected = _read_json(path / "selection.json").get("selected_features", [])
        if selected != ranking[: FROZEN_FEATURE_BUDGETS[model]]:
            raise Prompt16ExecutionError(f"global LLM truncation changed: {model}")
        global_states[model] = (path, file_sha256(path / "manifest.json"))

    supplemental_rows: list[dict[str, Any]] = []
    supplemental_metric_reconciliations = 0
    stable_outer_fits = 0
    stable_component_fits = 0
    for fold_id in range(1, 6):
        fold_root = supplement_root / f"fold_{fold_id}"
        fold_spec = next(
            item for item in split["folds"] if int(item["fold_id"]) == fold_id
        )
        fold_manifest_path = fold_root / "fold_manifest.json"
        fold_success = _read_json(fold_root / "_SUCCESS")
        if fold_success.get("fold_manifest_sha256") != file_sha256(fold_manifest_path):
            raise Prompt16ExecutionError(f"supplemental fold {fold_id} marker changed")
        stable_states: dict[str, tuple[Path, str]] = {}
        for model in ("lr", "catboost"):
            path = fold_root / "selection_fits" / f"stable_core_llm_fill_{model}"
            identity = _supplemental_selection_identity(
                entry=entry,
                ranking_manifest_sha256=ranking_manifest_sha,
                method_id="stable_core_llm_fill",
                model=model,
                fold_id=fold_id,
            )
            if _load_recursive_sealed(path, identity) is None:
                raise Prompt16ExecutionError(f"supplemental stable fit is unsealed: {path}")
            selection = _read_json(path / "selection.json")
            if selection.get("fit_scope") != "dev_fold_training_only":
                raise Prompt16ExecutionError("stable-core fit boundary changed")
            expected_train = {
                key: fold_spec["train"][key]
                for key in (
                    "rows",
                    "target_0",
                    "target_1",
                    "ordered_case_id_sha256",
                    "ordered_case_id_target_sha256",
                )
            }
            if selection.get("fold_train_alignment") != expected_train:
                raise Prompt16ExecutionError("stable-core fold-train alignment changed")
            bootstrap = selection.get("bootstrap", {})
            if (
                bootstrap.get("iterations") != 5
                or float(bootstrap.get("fraction")) != 0.8
                or float(bootstrap.get("stability_threshold")) != 0.8
                or bootstrap.get("base_seed") != 42
                or bootstrap.get("component_fit_count") != 5
            ):
                raise Prompt16ExecutionError("stable-core bootstrap contract changed")
            selected = [str(value) for value in selection.get("selected_features", [])]
            if len(selected) != FROZEN_FEATURE_BUDGETS[model] or not set(selected).issubset(retained):
                raise Prompt16ExecutionError("stable-core selected support changed")
            stable_states[model] = (path, file_sha256(path / "manifest.json"))
            stable_outer_fits += 1
            stable_component_fits += 5

        for cell in supplemental_cells():
            order = int(cell["configuration_order"])
            model = str(cell["model"])
            selection_path, selection_manifest_sha = (
                global_states[model]
                if cell["method_id"] == "llm"
                else stable_states[model]
            )
            selected = [
                str(value)
                for value in _read_json(selection_path / "selection.json").get(
                    "selected_features", []
                )
            ]
            selections[order].append(set(selected))
            evaluation_path = fold_root / "evaluations" / f"cell_{order:03d}"
            identity = _supplemental_dev_evaluation_identity(
                entry=entry,
                fold_id=fold_id,
                cell=cell,
                selection_manifest_sha256=selection_manifest_sha,
            )
            if _load_recursive_sealed(evaluation_path, identity) is None:
                raise Prompt16ExecutionError(f"supplemental evaluation unsealed: {evaluation_path}")
            status = _read_json(evaluation_path / "status.json")
            if status.get("status") != "complete" or status.get("oot_opened") is not False:
                raise Prompt16ExecutionError("supplemental DEV cell is not complete/no-OOT")
            reconciliation = _reconcile_prediction_metrics(
                evaluation_path / "predictions.parquet",
                evaluation_path / "metrics.json",
                expected_alignment=fold_spec["validation"],
            )
            execution = _read_json(evaluation_path / "execution.json")
            configuration = execution.get("configuration", {})
            if configuration.get("probability_orientation") != "class_1_higher_default_risk":
                raise Prompt16ExecutionError("supplemental probability orientation changed")
            if configuration.get("validation_target_used_for_fit") is not False:
                raise Prompt16ExecutionError("supplemental validation target leaked")
            thresholds[order].append((fold_id, reconciliation["threshold"]))
            supplemental_rows.append(
                {
                    "evaluation_id": f"p16v2-dev-fold-{fold_id}-c{order:03d}",
                    "source": "llm_supplement_v3",
                    "fold_id": fold_id,
                    "configuration_order": order,
                    "method_id": cell["method_id"],
                    "model": model,
                    "status": "complete",
                    "reason": None,
                    "selection_manifest_sha256": selection_manifest_sha,
                    "evaluation_manifest_sha256": file_sha256(
                        evaluation_path / "manifest.json"
                    ),
                    "selected_feature_count": len(selected),
                    "selected_features_sha256": canonical_sha256(selected),
                    **reconciliation,
                }
            )
            supplemental_metric_reconciliations += 1

    if len(supplemental_rows) != 20 or stable_outer_fits != 10 or stable_component_fits != 50:
        raise Prompt16ExecutionError("supplemental DEV accounting changed")

    all_rows = classical_rows + supplemental_rows
    all_ids = [row["evaluation_id"] for row in all_rows]
    if len(all_rows) != 170 or len(set(all_ids)) != 170:
        raise Prompt16ExecutionError("amended DEV identity accounting is not 170/170")
    completed = sum(row["status"] == "complete" for row in all_rows)
    unavailable = len(all_rows) - completed
    if (completed, unavailable) != (123, 47):
        raise Prompt16ExecutionError("amended DEV outcome accounting changed")
    threshold_freeze: dict[str, Any] = {}
    for order in range(1, 35):
        values = thresholds[order]
        threshold_freeze[str(order)] = {
            "rule": "maximize_ks_on_full_dev_training_scores_before_oot_scoring",
            "rule_frozen_before_oot": True,
            "oot_target_or_score_used_to_choose_threshold": False,
            "value_available_only_after_frozen_full_dev_model_fit": True,
            "fold_ids": [fold for fold, _ in values],
            "fold_thresholds": [value for _, value in values],
            "descriptive_median_available_dev_fold_threshold": (
                None
                if not values
                else float(np.median([value for _, value in values]))
            ),
        }
    cells34 = final_oot_cells(protocol_path)
    stability = [
        _selection_stability_row(
            order=order,
            cell=cells34[order - 1],
            fold_sets=selections[order],
            unavailable_folds=selection_unavailable[order],
            candidate_count=1959 if order <= 30 else 1068,
        )
        for order in range(1, 35)
    ]
    evaluation_registry_sha = canonical_sha256(all_rows)
    return {
        "schema_version": DEV_AUTH_SCHEMA_VERSION,
        "status": "complete_gate_passed",
        "authenticated_at_utc": _utc_now(),
        "dataset": "homecredit_model_stability_2024",
        "matrix_manifest_sha256": matrix_manifest_sha,
        "ordered_feature_count": 1959,
        "ordered_feature_universe_sha256": EXPECTED_UNIVERSE_SHA256,
        "classical_tree": classical_tree,
        "classical_evaluation_manifest_registry": classical_manifest_registry,
        "accounting": {
            "registered_evaluation_identities": 170,
            "authenticated_evaluation_identities": 170,
            "completed_numeric_outcomes": completed,
            "frozen_visible_unavailable_outcomes": unavailable,
            "classical_evaluation_identities": 150,
            "classical_completed_numeric_outcomes": classical_complete,
            "classical_visible_unavailable_outcomes": classical_unavailable,
            "supplemental_completed_numeric_outcomes": 20,
            "classical_selector_fit_records": classical_fit_records,
            "supplemental_stable_core_outer_fits": stable_outer_fits,
            "supplemental_internal_rf_mrmr_component_fits": stable_component_fits,
            "target_free_llm_ranking_generations": 1,
            "global_llm_truncation_states": 2,
            "metric_reconciliations": (
                classical_metric_reconciliations + supplemental_metric_reconciliations
            ),
        },
        "evaluation_registry_sha256": evaluation_registry_sha,
        "evaluation_rows": all_rows,
        "threshold_freeze": threshold_freeze,
        "stability_reference": stability,
        "llm_provenance": {
            "base_universe_count": 1959,
            "missingness_retained_count": 1068,
            "missingness_dropped_count": 891,
            "base_universe_fully_partitioned": True,
            "prompt_description_eligible_coverage_count": 1068,
            "ranking_count": 100,
            "request_model": FROZEN_LLM_MODEL,
            "temperature": FROZEN_LLM_TEMPERATURE,
            "ranking_manifest_sha256": ranking_manifest_sha,
            "ranking_payload_sha256": file_sha256(ranking_root / "ranking_payload.json"),
            "rendered_prompt_sha256": description_freeze["rendered_prompt_sha256"],
            "hallucinated_attempt_rejected": True,
            "missing_duplicate_hallucinated_features_accepted": 0,
            "new_llm_request_required_for_oot": False,
        },
        "semantic_mixed_voter": {
            "status": "unavailable_due_to_unresolved_historical_provenance",
            "execution_cells": 0,
        },
        "supplemental_controller": {
            "controller_manifest_sha256": file_sha256(controller_manifest),
            "controller_status_sha256": file_sha256(
                supplement_root / "controller_status.json"
            ),
            "supervisor_summary_sha256": file_sha256(supervisor_path),
            "terminal_log_sha256": file_sha256(
                supervisor_path.parent / "llm_supplement_v3_all_folds_dev.log"
            ),
            "peak_process_tree_rss_bytes": supervisor["peak_process_tree_rss_bytes"],
            "minimum_system_available_ram_bytes": supervisor[
                "minimum_system_available_ram_bytes"
            ],
            "oot_opened": False,
        },
    }


def _seal_final_directory(path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = [
        {
            "path": item.relative_to(path).as_posix(),
            "byte_size": item.stat().st_size,
            "sha256": file_sha256(item),
        }
        for item in sorted(
            (candidate for candidate in path.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(path).as_posix(),
        )
        if item.relative_to(path).as_posix() not in {"manifest.json", "_SUCCESS"}
        and not item.name.endswith(".partial")
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "identity": dict(identity),
        "artifacts": artifacts,
        "completed_at_utc": _utc_now(),
    }
    write_json_atomic(path / "manifest.json", manifest, overwrite=False)
    manifest_sha = file_sha256(path / "manifest.json")
    write_text_atomic(
        path / "_SUCCESS",
        json.dumps({"manifest_sha256": manifest_sha}, sort_keys=True) + "\n",
        overwrite=False,
    )
    return {**manifest, "manifest_sha256": manifest_sha}


def _load_final_sealed(
    path: Path, identity: Mapping[str, Any]
) -> dict[str, Any] | None:
    success_path = path / "_SUCCESS"
    if not success_path.is_file():
        return None
    success = _read_json(success_path)
    manifest_path = path / "manifest.json"
    if success.get("manifest_sha256") != file_sha256(manifest_path):
        raise Prompt16ExecutionError(f"final completion marker mismatch: {path}")
    manifest = _read_json(manifest_path)
    if manifest.get("identity") != dict(identity):
        raise Prompt16ExecutionError(f"final artifact identity mismatch: {path}")
    declared: set[str] = set()
    for item in manifest.get("artifacts", []):
        relative = str(item["path"])
        declared.add(relative)
        artifact = path / relative
        if not artifact.is_file() or artifact.stat().st_size != int(item["byte_size"]):
            raise Prompt16ExecutionError(f"final artifact size mismatch: {artifact}")
        if file_sha256(artifact) != item["sha256"]:
            raise Prompt16ExecutionError(f"final artifact digest mismatch: {artifact}")
    observed = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file()
        and item.relative_to(path).as_posix() not in {"manifest.json", "_SUCCESS"}
        and not item.name.endswith(".partial")
    }
    if observed != declared:
        raise Prompt16ExecutionError(f"final sealed inventory mismatch: {path}")
    return manifest


def _archive_final_incomplete(path: Path, archive_root: Path, scope: str) -> Path | None:
    if not path.exists() or (path / "_SUCCESS").is_file():
        return None
    archive_root.mkdir(parents=True, exist_ok=True)
    safe_scope = scope.replace(":", "_").replace("/", "_")
    destination = archive_root / (
        f"{safe_scope}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    )
    os.replace(path, destination)
    return destination


def _supplemental_oot_selection_identity(
    *,
    authorization_sha256: str,
    ranking_manifest_sha256: str,
    matrix_manifest_sha256: str,
    full_dev_alignment: Mapping[str, Any],
    method: str,
    model: str,
) -> dict[str, Any]:
    return {
        "operation": (
            "cached_llm_ranking_deterministic_truncation"
            if method == "llm"
            else "stable_core_llm_fill_full_dev_supervised_refit"
        ),
        "execution_authorization_sha256": authorization_sha256,
        "matrix_manifest_sha256": matrix_manifest_sha256,
        "ranking_manifest_sha256": ranking_manifest_sha256,
        "full_dev_alignment_sha256": canonical_sha256(full_dev_alignment),
        "method_id": method,
        "model": model,
        "feature_budget": FROZEN_FEATURE_BUDGETS[model],
        "fit_scope": (
            "outcome_independent_cached_state"
            if method == "llm"
            else "full_dev_only"
        ),
        "seed": None if method == "llm" else 42,
    }


def _supplemental_oot_evaluation_identity(
    *,
    authorization_sha256: str,
    matrix_manifest_sha256: str,
    cell: Mapping[str, Any],
    selection_manifest_sha256: str,
    oot_alignment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "operation": "final_amended_oot_evaluation",
        "execution_authorization_sha256": authorization_sha256,
        "matrix_manifest_sha256": matrix_manifest_sha256,
        "configuration_order": int(cell["configuration_order"]),
        "cell_sha256": canonical_sha256(cell),
        "selection_manifest_sha256": selection_manifest_sha256,
        "oot_alignment_sha256": canonical_sha256(oot_alignment),
    }


def _load_matrix_scope(
    *,
    matrix_root: Path,
    manifest: Mapping[str, Any],
    expected: Mapping[str, Any],
    predictors: Sequence[str],
    label: str,
    stage: str,
    stop_event: Any,
    stage_queue: Any,
    ram_ready_event: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _read_date_slice(
        matrix_root,
        manifest,
        date_min=str(expected["date_min"]),
        date_max=str(expected["date_max"]),
        predictors=predictors,
        stop_event=stop_event,
        stage_queue=stage_queue,
        stage=stage,
        fold_label=label,
        ram_ready_event=ram_ready_event,
    )
    authentication = _validate_scope_frame(frame, expected, label)
    return frame, authentication


def _ensure_cached_llm_oot_states(
    *,
    root: Path,
    archive_root: Path,
    authorization_sha256: str,
    ranking: Sequence[str],
    ranking_manifest_sha256: str,
    matrix_manifest_sha256: str,
    full_dev_alignment: Mapping[str, Any],
) -> dict[str, tuple[Path, str]]:
    outputs: dict[str, tuple[Path, str]] = {}
    for model in ("lr", "catboost"):
        path = root / "selection_fits" / f"llm_{model}"
        identity = _supplemental_oot_selection_identity(
            authorization_sha256=authorization_sha256,
            ranking_manifest_sha256=ranking_manifest_sha256,
            matrix_manifest_sha256=matrix_manifest_sha256,
            full_dev_alignment=full_dev_alignment,
            method="llm",
            model=model,
        )
        if _load_final_sealed(path, identity) is None:
            _archive_final_incomplete(path, archive_root, f"llm_{model}")
            path.mkdir(parents=True, exist_ok=False)
            selected = list(ranking[: FROZEN_FEATURE_BUDGETS[model]])
            write_json_atomic(
                path / "selection.json",
                {
                    "status": "complete",
                    "method_id": "llm",
                    "model": model,
                    "operation": "deterministic_truncation_of_authenticated_cached_ranking",
                    "llm_request_executed": False,
                    "ranking_regenerated": False,
                    "supervised_selector_fit": False,
                    "fit_scope": "outcome_independent_cached_state",
                    "requested_feature_budget": FROZEN_FEATURE_BUDGETS[model],
                    "realized_support": len(selected),
                    "selected_features": selected,
                    "natural_support_unpadded": True,
                    "ranking_manifest_sha256": ranking_manifest_sha256,
                },
                overwrite=False,
            )
            write_csv_atomic(
                path / "selected_features.csv",
                pd.DataFrame(
                    {"rank": range(1, len(selected) + 1), "feature": selected}
                ),
                required_columns=("rank", "feature"),
                ordered_row_identity_column="feature",
                overwrite=False,
            )
            _seal_final_directory(path, identity)
        outputs[model] = (path, file_sha256(path / "manifest.json"))
    return outputs


def _fit_full_dev_stable_states(
    *,
    root: Path,
    archive_root: Path,
    authorization_sha256: str,
    ranking: Sequence[str],
    ranking_manifest_sha256: str,
    matrix_manifest_sha256: str,
    full_dev: pd.DataFrame,
    retained: Sequence[str],
    full_dev_alignment: Mapping[str, Any],
    prompt_sha256: str,
    matrix_root: Path,
    stop_event: Any,
    stage_queue: Any,
) -> dict[str, tuple[Path, str]]:
    states: dict[str, tuple[Path, str]] = {}
    incomplete: list[str] = []
    identities: dict[str, dict[str, Any]] = {}
    for model in ("lr", "catboost"):
        path = root / "selection_fits" / f"stable_core_llm_fill_{model}"
        identity = _supplemental_oot_selection_identity(
            authorization_sha256=authorization_sha256,
            ranking_manifest_sha256=ranking_manifest_sha256,
            matrix_manifest_sha256=matrix_manifest_sha256,
            full_dev_alignment=full_dev_alignment,
            method="stable_core_llm_fill",
            model=model,
        )
        identities[model] = identity
        if _load_final_sealed(path, identity) is None:
            _archive_final_incomplete(
                path, archive_root, f"stable_core_llm_fill_{model}"
            )
            incomplete.append(model)
        else:
            states[model] = (path, file_sha256(path / "manifest.json"))
    if not incomplete:
        return states

    _publish_stage(
        stage_queue,
        "selection_encoding",
        "oot:oot:supplemental_selection_encoding",
        component="stable_core_llm_fill_full_dev_encoding",
        operation="selector_encoding",
        ram_recovery_barrier=True,
    )
    target = pd.Series(
        full_dev["target"].to_numpy(dtype=np.int64, copy=True),
        index=full_dev.index.copy(deep=True),
        name="target",
    )
    for name in NON_PREDICTORS:
        if name in full_dev:
            del full_dev[name]
    if list(full_dev.columns) != list(retained):
        raise Prompt16ExecutionError("supplemental full-DEV source order changed")
    encoder = OriginalFeatureNumericEncoder()
    encoder.fit(full_dev)
    numeric = encoder.transform_releasing_source(full_dev)
    if list(numeric.columns) != list(retained):
        raise Prompt16ExecutionError("supplemental full-DEV encoding changed candidate order")
    del encoder
    gc.collect()

    for model in incomplete:
        _check_stop(stop_event)
        path = root / "selection_fits" / f"stable_core_llm_fill_{model}"
        path.mkdir(parents=True, exist_ok=False)
        _publish_stage(
            stage_queue,
            "statistical_normalized_average_rank",
            f"oot:oot:fit_stable_core_llm_fill_{model}",
            component=f"stable_core_llm_fill_{model}_full_dev_selection",
            method_id="stable_core_llm_fill",
            model=model,
            operation="selector_fit",
        )
        started = time.perf_counter()
        selector = StableCoreLLMFillSelector(
            description_csv_path=str(matrix_root / "lineage.json"),
            cache_dir=str(root / "forbidden_llm_regeneration_cache"),
            llm_model=FROZEN_LLM_MODEL,
            llm_temperature=FROZEN_LLM_TEMPERATURE,
            llm_max_features=FROZEN_LLM_RANKING_BUDGET,
            llm_shared_ranking_enabled=True,
            llm_config_hash=prompt_sha256,
            llm_prompt_version="stability_expert_v5",
            llm_shared_pool_size=FROZEN_LLM_RANKING_BUDGET,
            final_feature_budget=FROZEN_FEATURE_BUDGETS[model],
            bootstrap_iterations=5,
            bootstrap_fraction=0.8,
            stability_threshold=0.8,
            random_state=42,
            component_n_jobs=MAX_ESTIMATOR_THREADS,
            iv_filter_kwargs={},
            allow_unranked_padding=False,
        )
        try:
            selector.fit_with_authenticated_ranking(
                numeric,
                target,
                ranked_features=list(ranking),
                ranking_manifest_sha256=ranking_manifest_sha256,
            )
            selected = list(selector.selected_features_ or [])
            if len(selected) != FROZEN_FEATURE_BUDGETS[model]:
                raise Prompt16ExecutionError("full-DEV stable-core support changed")
            if len(selected) != len(set(selected)) or not set(selected).issubset(retained):
                raise Prompt16ExecutionError("full-DEV stable-core selection escaped universe")
            write_json_atomic(
                path / "selection.json",
                {
                    "status": "complete",
                    "method_id": "stable_core_llm_fill",
                    "model": model,
                    "fit_scope": "full_dev_only",
                    "full_dev_alignment": dict(full_dev_alignment),
                    "validation_or_oot_used_for_fit": False,
                    "requested_feature_budget": FROZEN_FEATURE_BUDGETS[model],
                    "realized_support": len(selected),
                    "selected_features": selected,
                    "stable_core_features": list(selector.stable_core_features_ or []),
                    "natural_support_unpadded": True,
                    "ranking_manifest_sha256": ranking_manifest_sha256,
                    "bootstrap": {
                        "iterations": 5,
                        "fraction": 0.8,
                        "stability_threshold": 0.8,
                        "base_seed": 42,
                        "component_fit_count": 5,
                        "component_n_jobs": MAX_ESTIMATOR_THREADS,
                        "trace": selector.bootstrap_trace_,
                    },
                    "fit_seconds": time.perf_counter() - started,
                },
                overwrite=False,
            )
            write_csv_atomic(
                path / "selected_features.csv",
                pd.DataFrame(
                    {"rank": range(1, len(selected) + 1), "feature": selected}
                ),
                required_columns=("rank", "feature"),
                ordered_row_identity_column="feature",
                overwrite=False,
            )
            _seal_final_directory(path, identities[model])
        finally:
            del selector
            gc.collect()
        states[model] = (path, file_sha256(path / "manifest.json"))
    del numeric, target
    gc.collect()
    return states


def run_supplemental_oot_worker(
    *,
    repository_root: str,
    output_root: str,
    authorization_sha256: str,
    stop_event: Any = None,
    stage_queue: Any = None,
    ram_ready_event: Any = None,
) -> dict[str, Any]:
    """Run the four immutable supplemental OOT cells without an LLM request."""

    project = Path(repository_root).resolve()
    root = Path(output_root)
    success_path = root / "_SUCCESS"
    phase_manifest_path = root / "phase_manifest.json"
    if success_path.is_file():
        success = _read_json(success_path)
        if success.get("phase_manifest_sha256") != file_sha256(phase_manifest_path):
            raise Prompt16ExecutionError("supplemental OOT phase marker mismatch")
        completed = _read_json(phase_manifest_path)
        if completed.get("execution_authorization_sha256") != authorization_sha256:
            raise Prompt16ExecutionError("supplemental OOT authorization identity changed")
        return {**completed, "reused_completed_phase": True}
    root.mkdir(parents=True, exist_ok=True)
    archive_root = root / "archived_incomplete_attempts"
    protocol_path = project / PROTOCOL_RELATIVE_PATH
    _, protocol = _protocol_payload(protocol_path)
    matrix_settings = protocol["approved_protocol"]["method_and_evaluation_matrix"]
    split = protocol["approved_protocol"]["split_and_fold_boundaries"]
    matrix_root = project / MATRIX_RELATIVE_ROOT
    matrix_manifest, metadata = _matrix_identity(matrix_root)
    matrix_manifest_sha = file_sha256(matrix_root / "manifest.json")
    availability = _read_json(
        project / SUPPLEMENTAL_DEV_RELATIVE_ROOT / "feature_availability_filter.json"
    )
    retained = [str(value) for value in availability["retained_features"]]
    ranking_root = project / SUPPLEMENTAL_DEV_RELATIVE_ROOT / "llm_ranking"
    ranking_manifest_sha = file_sha256(ranking_root / "manifest.json")
    ranking_payload = _read_json(ranking_root / "ranking_payload.json")
    ranking = [str(value) for value in ranking_payload["selected_features"]]
    if len(ranking) != 100 or not set(ranking).issubset(retained):
        raise Prompt16ExecutionError("cached supplemental ranking is not reusable")
    prompt_sha = _read_json(ranking_root / "provenance_freeze.json")[
        "rendered_prompt_sha256"
    ]

    full_dev, full_dev_auth = _load_matrix_scope(
        matrix_root=matrix_root,
        manifest=matrix_manifest,
        expected=split["dev"],
        predictors=retained,
        label="supplemental_oot:full_dev",
        stage="full_dev_data_loading",
        stop_event=stop_event,
        stage_queue=stage_queue,
        ram_ready_event=ram_ready_event,
    )
    llm_states = _ensure_cached_llm_oot_states(
        root=root,
        archive_root=archive_root,
        authorization_sha256=authorization_sha256,
        ranking=ranking,
        ranking_manifest_sha256=ranking_manifest_sha,
        matrix_manifest_sha256=matrix_manifest_sha,
        full_dev_alignment=full_dev_auth["observed"],
    )
    stable_states = _fit_full_dev_stable_states(
        root=root,
        archive_root=archive_root,
        authorization_sha256=authorization_sha256,
        ranking=ranking,
        ranking_manifest_sha256=ranking_manifest_sha,
        matrix_manifest_sha256=matrix_manifest_sha,
        full_dev=full_dev,
        retained=retained,
        full_dev_alignment=full_dev_auth["observed"],
        prompt_sha256=prompt_sha,
        matrix_root=matrix_root,
        stop_event=stop_event,
        stage_queue=stage_queue,
    )
    if full_dev.shape[1] == 0:
        del full_dev
        gc.collect()
        full_dev, reloaded_auth = _load_matrix_scope(
            matrix_root=matrix_root,
            manifest=matrix_manifest,
            expected=split["dev"],
            predictors=retained,
            label="supplemental_oot:full_dev",
            stage="full_dev_selected_projection_reload",
            stop_event=stop_event,
            stage_queue=stage_queue,
            ram_ready_event=ram_ready_event,
        )
        if reloaded_auth != full_dev_auth:
            raise Prompt16ExecutionError("supplemental full-DEV reload identity changed")
    oot, oot_auth = _load_matrix_scope(
        matrix_root=matrix_root,
        manifest=matrix_manifest,
        expected=split["oot"],
        predictors=retained,
        label="supplemental_oot:oot",
        stage="locked_oot_data_loading",
        stop_event=stop_event,
        stage_queue=stage_queue,
        ram_ready_event=ram_ready_event,
    )
    if set(full_dev["case_id"].tolist()) & set(oot["case_id"].tolist()):
        raise Prompt16ExecutionError("supplemental full-DEV and OOT case IDs overlap")
    write_json_atomic(
        root / "scope_authentication.json",
        {
            "matrix_manifest_sha256": matrix_manifest_sha,
            "full_dev": full_dev_auth,
            "oot": oot_auth,
            "case_id_overlap": 0,
        },
    )

    completed = 0
    unavailable = 0
    for cell in supplemental_cells():
        _check_stop(stop_event)
        order = int(cell["configuration_order"])
        model = str(cell["model"])
        selection_path, selection_manifest_sha = (
            llm_states[model]
            if cell["method_id"] == "llm"
            else stable_states[model]
        )
        selection = _read_json(selection_path / "selection.json")
        selected = [str(value) for value in selection.get("selected_features", [])]
        if len(selected) != FROZEN_FEATURE_BUDGETS[model]:
            raise Prompt16ExecutionError("supplemental OOT support is not like-for-like")
        path = root / "evaluations" / f"cell_{order:03d}"
        identity = _supplemental_oot_evaluation_identity(
            authorization_sha256=authorization_sha256,
            matrix_manifest_sha256=matrix_manifest_sha,
            cell=cell,
            selection_manifest_sha256=selection_manifest_sha,
            oot_alignment=oot_auth["observed"],
        )
        sealed = _load_final_sealed(path, identity)
        if sealed is not None:
            status = _read_json(path / "status.json")
            completed += int(status.get("status") == "complete")
            unavailable += int(status.get("status") != "complete")
            continue
        _archive_final_incomplete(path, archive_root, f"cell_{order:03d}")
        path.mkdir(parents=True, exist_ok=False)
        _publish_stage(
            stage_queue,
            f"final_{model}",
            f"oot:oot:cell_{order:03d}",
            component=f"{cell['method_id']}_{model}_model_fit_and_evaluation",
            method_id=cell["method_id"],
            model=model,
            configuration_order=order,
            operation="oot_evaluation",
        )
        started = time.perf_counter()
        try:
            predictions, metrics, details = _fit_and_evaluate(
                cell=cell,
                selected=selected,
                train=full_dev,
                validation=oot,
                predictors=retained,
                matrix=matrix_settings,
                phase="oot",
                frozen_threshold=None,
                full_dev_training_ks_threshold=True,
            )
            prediction_auth = _locked_alignment_summary(
                predictions["case_id"].tolist(), predictions["target"].tolist()
            )
            if (
                prediction_auth["ordered_case_id_sha256"]
                != oot_auth["observed"]["ordered_case_id_sha256"]
                or prediction_auth["ordered_case_id_target_sha256"]
                != oot_auth["observed"]["ordered_case_id_target_sha256"]
            ):
                raise Prompt16ExecutionError("supplemental OOT prediction alignment changed")
            write_parquet_atomic(
                path / "predictions.parquet",
                predictions,
                required_columns=(
                    "case_id",
                    "target",
                    "score",
                    "decision_threshold",
                ),
                ordered_row_identity_column="case_id",
                overwrite=False,
            )
            write_json_atomic(path / "metrics.json", metrics, overwrite=False)
            write_json_atomic(path / "execution.json", details, overwrite=False)
            write_json_atomic(
                path / "status.json",
                {
                    "status": "complete",
                    "reason": None,
                    "configuration_order": order,
                    "cell": dict(cell),
                    "requested_feature_budget": cell["requested_feature_budget"],
                    "realized_support": len(selected),
                    "natural_support_like_for_like": True,
                    "selection_manifest_sha256": selection_manifest_sha,
                    "prediction_alignment": prediction_auth,
                    "validation_or_oot_target_used_for_fit": False,
                    "threshold_source": "full_dev_training_scores_maximize_ks",
                    "feature_psi_source": "classical/feature_psi/all_source_features.parquet",
                    "elapsed_seconds": time.perf_counter() - started,
                },
                overwrite=False,
            )
            completed += 1
        except Exception as exc:
            write_json_atomic(
                path / "failure.json",
                {
                    "status": "failed_unsealed_safe_to_resume",
                    "configuration_order": order,
                    "error": {"class": type(exc).__name__, "message": str(exc)},
                    "elapsed_seconds": time.perf_counter() - started,
                },
                overwrite=False,
            )
            raise
        _seal_final_directory(path, identity)

    if completed + unavailable != 4:
        raise Prompt16ExecutionError("supplemental OOT accounting does not cover four cells")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "phase": "supplemental_final_oot",
        "execution_authorization_sha256": authorization_sha256,
        "matrix_manifest_sha256": matrix_manifest_sha,
        "ranking_manifest_sha256": ranking_manifest_sha,
        "llm_request_count": 0,
        "ranking_regeneration_count": 0,
        "supervised_full_dev_refits": 2,
        "internal_rf_mrmr_component_fits": 10,
        "completed_evaluations": completed,
        "unavailable_evaluations": unavailable,
        "completed_at_utc": _utc_now(),
    }
    write_json_atomic(phase_manifest_path, manifest)
    write_text_atomic(
        success_path,
        json.dumps(
            {"phase_manifest_sha256": file_sha256(phase_manifest_path)},
            sort_keys=True,
        )
        + "\n",
    )
    del full_dev, oot
    gc.collect()
    return manifest


def _oot_cell_path(output_root: Path, order: int) -> Path:
    phase = "classical" if order <= 30 else "supplemental"
    return output_root / phase / "evaluations" / f"cell_{order:03d}"


def _oot_selection_path(output_root: Path, order: int, refits: Sequence[Mapping[str, Any]]) -> Path:
    if order <= 30:
        fit = next(
            row for row in refits[:27] if order in row["dependent_configuration_orders"]
        )
        return output_root / "classical" / "selection_fits" / str(fit["refit_id"])
    cell = {31: ("llm", "lr"), 32: ("llm", "catboost"), 33: ("stable_core_llm_fill", "lr"), 34: ("stable_core_llm_fill", "catboost")}[order]
    return output_root / "supplemental" / "selection_fits" / f"{cell[0]}_{cell[1]}"


def _prediction_for_pair(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=["case_id", "target", "score"])
    return frame.rename(
        columns={
            "case_id": "stable_row_id",
            "score": "prediction_probability",
        }
    )


def _dev_oof_prediction_paths(root: Path, order: int) -> list[Path]:
    if order <= 30:
        base = root / CLASSICAL_DEV_RELATIVE_ROOT
    else:
        base = root / SUPPLEMENTAL_DEV_RELATIVE_ROOT
    outputs: list[Path] = []
    for fold in range(1, 6):
        path = base / f"fold_{fold}" / "evaluations" / f"cell_{order:03d}"
        status = _read_json(path / "status.json")
        prediction = path / "predictions.parquet"
        if status.get("status") == "complete":
            if not prediction.is_file():
                raise Prompt16ExecutionError(f"complete DEV cell lacks prediction: {path}")
            outputs.append(prediction)
    return outputs


def run_final_oot_analysis(
    *,
    repository_root: str | Path,
    output_root: str | Path,
    authorization_sha256: str,
    analysis_plan_path: str | Path,
    stop_event: Any = None,
    stage_queue: Any = None,
) -> dict[str, Any]:
    project = Path(repository_root).resolve()
    output = Path(output_root)
    analysis_root = output / "analysis"
    success_path = analysis_root / "_SUCCESS"
    manifest_path = analysis_root / "analysis_manifest.json"
    if success_path.is_file():
        success = _read_json(success_path)
        if success.get("analysis_manifest_sha256") != file_sha256(manifest_path):
            raise Prompt16ExecutionError("final OOT analysis marker mismatch")
        manifest = _read_json(manifest_path)
        if manifest.get("execution_authorization_sha256") != authorization_sha256:
            raise Prompt16ExecutionError("final OOT analysis authorization changed")
        return {**manifest, "reused_completed_analysis": True}
    analysis_root.mkdir(parents=True, exist_ok=True)
    archive_root = analysis_root / "archived_incomplete_attempts"
    plan = _read_json(analysis_plan_path)
    if plan.get("schema_version") != "prompt_16_final_oot_analysis_plan_v1":
        raise Prompt16ExecutionError("final OOT analysis-plan schema changed")
    if plan.get("execution_authorization_sha256") not in {None, authorization_sha256}:
        raise Prompt16ExecutionError("analysis plan authorization binding changed")
    cells = final_oot_cells(project / PROTOCOL_RELATIVE_PATH)
    refits = final_full_dev_refits(project / PROTOCOL_RELATIVE_PATH)
    graph = paired_comparison_graph(project / PROTOCOL_RELATIVE_PATH)
    expected_oot = _read_json(output / "classical/scope_authentication.json")["validation"][
        "observed"
    ]
    cell_rows: list[dict[str, Any]] = []
    predictions: dict[int, Path] = {}
    selection_sets: dict[int, list[str]] = {}
    for cell in cells:
        order = int(cell["configuration_order"])
        path = _oot_cell_path(output, order)
        if not (path / "_SUCCESS").is_file():
            raise Prompt16ExecutionError(f"OOT cell is not sealed: {path}")
        status = _read_json(path / "status.json")
        selection_path = _oot_selection_path(output, order, refits)
        selected = [
            str(value)
            for value in _read_json(selection_path / "selection.json").get(
                "selected_features", []
            )
        ]
        selection_sets[order] = selected
        row = {
            "configuration_order": order,
            "configuration_id": cell.get("configuration_id", f"p16v1-c{order:03d}"),
            "method_id": cell["method_id"],
            "model": cell["model"],
            "status": status.get("status"),
            "reason": status.get("reason"),
            "requested_feature_budget": cell.get("requested_feature_budget"),
            "realized_support": len(selected),
            "natural_support_like_for_like": status.get("natural_support_like_for_like"),
            "selection_manifest_sha256": file_sha256(selection_path / "manifest.json"),
            "evaluation_manifest_sha256": file_sha256(path / "manifest.json"),
        }
        if status.get("status") == "complete":
            reconciliation = _reconcile_prediction_metrics(
                path / "predictions.parquet",
                path / "metrics.json",
                expected_alignment=expected_oot,
            )
            metrics = _read_json(path / "metrics.json")
            row.update(metrics)
            row.update(
                {
                    key: reconciliation[key]
                    for key in (
                        "prediction_sha256",
                        "metrics_sha256",
                        "maximum_absolute_metric_difference",
                    )
                }
            )
            predictions[order] = path / "predictions.parquet"
        cell_rows.append(row)
    if len(cell_rows) != 34:
        raise Prompt16ExecutionError("OOT metric accounting does not contain 34 cells")
    write_csv_atomic(analysis_root / "oot_metrics.csv", pd.DataFrame(cell_rows))

    _publish_stage(
        stage_queue,
        "oot_score_psi",
        "oot:analysis:score_psi",
        component="dev_oof_to_oot_score_psi",
        operation="descriptive_drift",
    )
    score_psi_rows: list[dict[str, Any]] = []
    score_psi_bins: list[pd.DataFrame] = []
    for order, prediction_path in sorted(predictions.items()):
        _check_stop(stop_event)
        dev_paths = _dev_oof_prediction_paths(project, order)
        dev_scores = pd.concat(
            [pd.read_parquet(path, columns=["score"]) for path in dev_paths],
            ignore_index=True,
        )["score"]
        oot_scores = pd.read_parquet(prediction_path, columns=["score"])["score"]
        result = score_psi_from_predictions(dev_scores, oot_scores)
        score_psi_rows.append(
            {
                "configuration_order": order,
                "dev_available_fold_count": len(dev_paths),
                "dev_oof_row_count": len(dev_scores),
                "oot_row_count": len(oot_scores),
                "score_psi": result.psi,
                "definition_sha256": canonical_sha256(result.definition),
            }
        )
        bins = result.bins.copy()
        bins.insert(0, "configuration_order", order)
        score_psi_bins.append(bins)
    write_csv_atomic(analysis_root / "score_psi.csv", pd.DataFrame(score_psi_rows))
    write_parquet_atomic(
        analysis_root / "score_psi_bins.parquet",
        pd.concat(score_psi_bins, ignore_index=True) if score_psi_bins else pd.DataFrame(),
    )

    feature_psi = pd.read_parquet(
        output / "classical/feature_psi/all_source_features.parquet"
    )
    feature_rows: list[pd.DataFrame] = []
    feature_summaries: list[dict[str, Any]] = []
    for order, selected in sorted(selection_sets.items()):
        subset = feature_psi.loc[feature_psi["feature"].isin(selected)].copy()
        order_map = {feature: index for index, feature in enumerate(selected)}
        subset["__order"] = subset["feature"].map(order_map)
        subset.sort_values("__order", inplace=True, kind="mergesort")
        subset.drop(columns="__order", inplace=True)
        if len(subset) != len(selected):
            raise Prompt16ExecutionError(f"feature PSI coverage changed for cell {order}")
        subset.insert(0, "configuration_order", order)
        feature_rows.append(subset)
        feature_summaries.append(
            {
                "configuration_order": order,
                **summarise_feature_psi(subset, references=(0.1, 0.25)),
            }
        )
    write_parquet_atomic(
        analysis_root / "selected_feature_psi.parquet",
        pd.concat(feature_rows, ignore_index=True),
    )
    write_csv_atomic(
        analysis_root / "selected_feature_psi_summary.csv",
        pd.DataFrame(feature_summaries),
    )

    comparison_root = analysis_root / "paired_comparisons"
    comparison_root.mkdir(exist_ok=True)
    inferential_rows: list[dict[str, Any]] = []
    for graph_row in graph:
        _check_stop(stop_event)
        comparison_order = int(graph_row["comparison_order"])
        path = comparison_root / f"comparison_{comparison_order:03d}"
        comparator_order = int(graph_row["comparator_configuration_order"])
        reference_value = graph_row["reference_configuration_order"]
        if graph_row["availability"] != "registered":
            inferential_rows.append(
                {
                    **graph_row,
                    "status": "unavailable",
                    "reason": graph_row["availability"],
                }
            )
            continue
        reference_order = int(reference_value)
        comparator_prediction = predictions.get(comparator_order)
        reference_prediction = predictions.get(reference_order)
        identity = {
            "operation": "preregistered_paired_oot_inference",
            "execution_authorization_sha256": authorization_sha256,
            "comparison_sha256": canonical_sha256(graph_row),
            "comparator_prediction_sha256": (
                None
                if comparator_prediction is None
                else file_sha256(comparator_prediction)
            ),
            "reference_prediction_sha256": (
                None if reference_prediction is None else file_sha256(reference_prediction)
            ),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_minimum_valid": BOOTSTRAP_MINIMUM_VALID,
        }
        sealed = _load_final_sealed(path, identity)
        if sealed is not None:
            inferential_rows.append(_read_json(path / "result.json"))
            continue
        _archive_final_incomplete(path, archive_root, f"comparison_{comparison_order:03d}")
        path.mkdir(parents=True, exist_ok=False)
        if comparator_prediction is None or reference_prediction is None:
            result_row = {
                **graph_row,
                "status": "unavailable",
                "reason": "required_oot_prediction_unavailable",
            }
            write_json_atomic(path / "result.json", result_row, overwrite=False)
            _seal_final_directory(path, identity)
            inferential_rows.append(result_row)
            continue
        _publish_stage(
            stage_queue,
            "oot_paired_inference",
            f"oot:analysis:comparison_{comparison_order:03d}",
            component=(
                f"paired_inference_{comparator_order:03d}_vs_{reference_order:03d}"
            ),
            operation="paired_delong_and_stratified_bootstrap",
            comparison_order=comparison_order,
        )
        left = _prediction_for_pair(comparator_prediction)
        right = _prediction_for_pair(reference_prediction)
        aligned = align_paired_predictions(left, right)
        delong = paired_delong_test(
            aligned["target"], aligned["score_a"], aligned["score_b"]
        )
        bootstrap = fast_paired_stratified_bootstrap(
            aligned,
            repetitions=BOOTSTRAP_REPETITIONS,
            seed=BOOTSTRAP_SEED,
            minimum_valid=BOOTSTRAP_MINIMUM_VALID,
        )
        auc_interval = bootstrap["metrics"]["auc"]
        result_row = {
            **graph_row,
            "status": "complete",
            "reason": None,
            "aligned_row_count": len(aligned),
            "aligned_target_1": int(aligned["target"].sum()),
            "auc_comparator": delong["auc_a"],
            "auc_reference": delong["auc_b"],
            "auc_delta_comparator_minus_reference": delong[
                "auc_difference_a_minus_b"
            ],
            "gini_delta_comparator_minus_reference": 2
            * delong["auc_difference_a_minus_b"],
            "delong_variance": delong["variance"],
            "delong_z": delong["z_score"],
            "raw_two_sided_p_value": delong["two_sided_p_value"],
            "bootstrap_auc_ci95_lower": auc_interval[
                "ci95_percentile_lower"
            ],
            "bootstrap_auc_ci95_upper": auc_interval[
                "ci95_percentile_upper"
            ],
            "bootstrap_auc_interval_valid": auc_interval["interval_valid"],
            "bootstrap_ks_delta": bootstrap["metrics"]["ks"][
                "observed_difference_a_minus_b"
            ],
            "bootstrap_ks_ci95_lower": bootstrap["metrics"]["ks"][
                "ci95_percentile_lower"
            ],
            "bootstrap_ks_ci95_upper": bootstrap["metrics"]["ks"][
                "ci95_percentile_upper"
            ],
            "bootstrap_lift_at_10_delta": bootstrap["metrics"]["lift_at_10"][
                "observed_difference_a_minus_b"
            ],
            "bootstrap_lift_at_10_ci95_lower": bootstrap["metrics"]["lift_at_10"][
                "ci95_percentile_lower"
            ],
            "bootstrap_lift_at_10_ci95_upper": bootstrap["metrics"]["lift_at_10"][
                "ci95_percentile_upper"
            ],
            "bootstrap_attempted_repetitions": bootstrap["attempted_repetitions"],
            "bootstrap_valid_repetitions": bootstrap["valid_repetitions"],
            "bootstrap_seed": bootstrap["seed"],
            "bootstrap_implementation": bootstrap["implementation"],
        }
        write_json_atomic(path / "result.json", result_row, overwrite=False)
        _seal_final_directory(path, identity)
        inferential_rows.append(result_row)
        del left, right, aligned
        gc.collect()

    complete_inference = [row for row in inferential_rows if row["status"] == "complete"]
    holm_rows: list[dict[str, Any]] = []
    for family, family_rows_iter in itertools.groupby(
        sorted(complete_inference, key=lambda row: row["holm_family_id"]),
        key=lambda row: row["holm_family_id"],
    ):
        family_rows = list(family_rows_iter)
        raw = [float(row["raw_two_sided_p_value"]) for row in family_rows]
        adjusted = holm_adjust(raw)
        for row, adjusted_p in zip(family_rows, adjusted, strict=True):
            delta = float(row["auc_delta_comparator_minus_reference"])
            interval_above_zero = bool(
                row["bootstrap_auc_interval_valid"]
                and float(row["bootstrap_auc_ci95_lower"]) > 0
            )
            holm_significant = adjusted_p < HOLM_ALPHA
            inferential_count = int(holm_significant) + int(interval_above_zero)
            if delta <= 0:
                label = "not_supported"
            elif inferential_count == 2:
                label = "strong"
            elif inferential_count == 1:
                label = "moderate"
            else:
                label = "weak"
            holm_rows.append(
                {
                    **row,
                    "holm_family_size": len(family_rows),
                    "holm_adjusted_p_value": float(adjusted_p),
                    "holm_alpha": HOLM_ALPHA,
                    "holm_significant_strict_less_than_alpha": holm_significant,
                    "bootstrap_auc_interval_wholly_above_zero": interval_above_zero,
                    "predictive_evidence_label": label,
                    "directional_materiality_threshold_auc_delta_exclusive": 0.0,
                    "business_materiality_threshold_status": "not_preregistered_no_claim_permitted",
                }
            )
    unavailable_inference = [
        {
            **row,
            "holm_family_size": None,
            "holm_adjusted_p_value": None,
            "holm_alpha": HOLM_ALPHA,
            "holm_significant_strict_less_than_alpha": False,
            "bootstrap_auc_interval_wholly_above_zero": False,
            "predictive_evidence_label": "not_supported",
            "directional_materiality_threshold_auc_delta_exclusive": 0.0,
            "business_materiality_threshold_status": "not_preregistered_no_claim_permitted",
        }
        for row in inferential_rows
        if row["status"] != "complete"
    ]
    final_inference = sorted(
        holm_rows + unavailable_inference,
        key=lambda row: int(row["comparison_order"]),
    )
    if len(final_inference) != 72:
        raise Prompt16ExecutionError("paired inference does not account for 72 graph entries")
    write_csv_atomic(
        analysis_root / "paired_inference_holm_materiality.csv",
        pd.DataFrame(final_inference),
    )
    write_json_atomic(
        analysis_root / "analysis_rules.json",
        {
            "paired_test": "two_sided_delong_on_identical_oot_rows",
            "bootstrap": {
                "type": "paired_stratified_by_target",
                "attempted_repetitions": BOOTSTRAP_REPETITIONS,
                "minimum_valid_repetitions": BOOTSTRAP_MINIMUM_VALID,
                "seed": BOOTSTRAP_SEED,
                "confidence_interval": "95_percentile",
                "metrics": ["auc", "ks", "lift_at_10"],
            },
            "holm_scope": "within_named_dataset_model_reference_family",
            "holm_alpha": HOLM_ALPHA,
            "evidence_language": plan["evidence_language"],
            "materiality": plan["materiality"],
            "psi": plan["psi_contract"],
        },
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "execution_authorization_sha256": authorization_sha256,
        "analysis_plan_sha256": file_sha256(analysis_plan_path),
        "oot_cells_accounted": 34,
        "oot_cells_complete": sum(row["status"] == "complete" for row in cell_rows),
        "oot_cells_unavailable": sum(row["status"] != "complete" for row in cell_rows),
        "comparison_graph_entries": 72,
        "registered_inferential_comparisons": 70,
        "inference_complete": len(complete_inference),
        "inference_unavailable": 72 - len(complete_inference),
        "metric_reconciliation_tolerance": METRIC_TOLERANCE,
        "metric_reconciliations": len(predictions),
        "score_psi_rows": len(score_psi_rows),
        "selected_feature_psi_cell_rows": len(feature_summaries),
        "completed_at_utc": _utc_now(),
    }
    write_json_atomic(manifest_path, manifest)
    write_text_atomic(
        success_path,
        json.dumps(
            {"analysis_manifest_sha256": file_sha256(manifest_path)}, sort_keys=True
        )
        + "\n",
    )
    return manifest


def _self_authenticated_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("artifact_authentication_sha256", None)
    return {
        **unsigned,
        "artifact_authentication_sha256": canonical_sha256(unsigned),
    }


def _write_frozen_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_atomic(path, _self_authenticated_payload(payload), overwrite=False)


def _assert_no_preexisting_prompt16_oot(root: Path) -> dict[str, Any]:
    results_root = root / "results/prompt_16_homecredit_model_stability_2024"
    output_root = root / "outputs/prompt_16_homecredit_model_stability_2024"
    matches: list[str] = []
    for search_root in (results_root, output_root):
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            relative = _relative(path, root).lower()
            path_parts = Path(relative).parts
            if any(
                part == "oot" or part.startswith("oot_") or part.startswith("oot-")
                for part in path_parts
            ):
                matches.append(relative)
    allowed_prefixes = {
        OOT_RELATIVE_ROOT.as_posix().lower(),
        OOT_LOG_RELATIVE_ROOT.as_posix().lower(),
        TEMP_RELATIVE_ROOT.as_posix().lower(),
    }
    unauthorized = [
        item
        for item in matches
        if not any(item == prefix or item.startswith(prefix + "/") for prefix in allowed_prefixes)
    ]
    if unauthorized:
        raise Prompt16ExecutionError(
            "unauthorized pre-existing third-dataset OOT path: " + unauthorized[0]
        )
    return {
        "checked_roots": [_relative(results_root, root), _relative(output_root, root)],
        "unauthorized_oot_paths": [],
        "final_output_root_exists": (root / OOT_RELATIVE_ROOT).exists(),
    }


def _freeze_holm_registry(graph: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families: dict[str, list[str]] = {}
    for row in graph:
        if row["availability"] != "registered":
            continue
        families.setdefault(str(row["holm_family_id"]), []).append(
            str(row["comparison_id"])
        )
    return {
        "schema_version": "prompt_16_final_oot_holm_registry_v1",
        "status": "frozen_before_oot",
        "alpha": HOLM_ALPHA,
        "procedure": "Holm_step_down",
        "scope": "within_named_dataset_model_reference_family_never_across_families",
        "family_count": len(families),
        "registered_comparison_count": sum(map(len, families.values())),
        "families": [
            {
                "holm_family_id": family,
                "family_size": len(comparisons),
                "comparison_ids": comparisons,
            }
            for family, comparisons in sorted(families.items())
        ],
    }


def build_freeze(
    *,
    repository_root: str | Path = PROJECT_ROOT,
    implementation_commit: str,
) -> dict[str, Any]:
    """Create the pre-OOT freeze using DEV artifacts only; never resolve OOT."""

    root = Path(repository_root).resolve()
    repository = _assert_required_ancestry(root)
    if _git(root, "rev-parse", implementation_commit) != implementation_commit:
        raise Prompt16ExecutionError("implementation commit must be a full commit identity")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, repository["head"]],
        cwd=root,
        check=False,
    ).returncode != 0:
        raise Prompt16ExecutionError("implementation commit is not an ancestor of HEAD")
    no_oot = _assert_no_preexisting_prompt16_oot(root)
    if no_oot["final_output_root_exists"]:
        raise Prompt16ExecutionError("final OOT output root exists before freeze")
    audit_root = root / FREEZE_RELATIVE_ROOT
    if audit_root.exists():
        raise Prompt16ExecutionError(f"final freeze directory already exists: {audit_root}")

    dev = authenticate_complete_dev(root)
    evaluation_rows = dev.pop("evaluation_rows")
    protocol_path = root / PROTOCOL_RELATIVE_PATH
    cells = final_oot_cells(protocol_path)
    refits = final_full_dev_refits(protocol_path)
    graph = paired_comparison_graph(protocol_path)
    holm = _freeze_holm_registry(graph)
    audit_root.mkdir(parents=True, exist_ok=False)

    write_csv_atomic(audit_root / "complete_amended_dev_accounting.csv", pd.DataFrame(evaluation_rows))
    _write_frozen_json(
        audit_root / "complete_amended_dev_authentication.json",
        dev,
    )
    _write_frozen_json(
        audit_root / "final_34_cell_oot_registry.json",
        {
            "schema_version": "prompt_16_final_34_cell_oot_registry_v1",
            "status": "immutable_frozen_before_oot",
            "dataset": "homecredit_model_stability_2024",
            "expected_evaluations": 34,
            "classical_evaluations": 30,
            "llm_evaluations": 2,
            "stable_core_llm_fill_evaluations": 2,
            "semantic_mixed_voter_execution_cells": 0,
            "cells": cells,
            "cells_sha256": canonical_sha256(cells),
            "immutability_rule": "no_add_remove_tune_reorder_reweight_or_reinterpret_after_oot_begins",
        },
    )
    _write_frozen_json(
        audit_root / "full_dev_selector_refit_registry.json",
        {
            "schema_version": "prompt_16_final_full_dev_refit_registry_v1",
            "status": "immutable_frozen_before_oot",
            "registered_outer_refits": 29,
            "classical_outer_refits": 27,
            "stable_core_outer_refits": 2,
            "stable_core_internal_rf_mrmr_component_fits": 10,
            "total_internal_fit_invocations_including_outer_classical": 37,
            "cached_llm_truncation_states": 2,
            "llm_api_calls": 0,
            "refits": refits,
            "refits_sha256": canonical_sha256(refits),
        },
    )
    _write_frozen_json(
        audit_root / "paired_comparison_graph.json",
        {
            "schema_version": "prompt_16_final_oot_comparison_graph_v1",
            "status": "immutable_frozen_before_oot",
            "graph_entries": 72,
            "registered_inferential_comparisons": 70,
            "unavailable_historical_comparisons": 2,
            "comparisons": graph,
            "graph_sha256": canonical_sha256(graph),
        },
    )
    _write_frozen_json(audit_root / "holm_family_registry.json", holm)
    psi_contract = {
        "schema_version": "prompt_16_final_oot_psi_contract_v1",
        "status": "frozen_before_oot",
        "score_psi": {
            "reference": "available_authenticated_DEV_OOF_scores_for_same_configuration",
            "bin_source": "DEV_OOF_quantiles_only",
            "requested_bins": 10,
            "outer_edges": "infinite",
            "smoothing_epsilon": 1e-6,
            "oot_used_to_fit_bins": False,
        },
        "selected_feature_psi": {
            "reference": "full_DEV_original_source_feature_values",
            "comparison": "same_original_source_feature_on_frozen_OOT_rows",
            "requested_bins": 10,
            "categorical_missing_and_unseen_states": "explicit",
            "descriptive_references": [0.1, 0.25],
            "significance_label": "none_descriptive_only",
        },
    }
    _write_frozen_json(audit_root / "psi_reference_and_binning_contract.json", psi_contract)
    materiality = {
        "schema_version": "prompt_16_final_oot_materiality_failure_rules_v1",
        "status": "frozen_before_oot",
        "directional_auc_threshold": {
            "operator": "strictly_greater_than",
            "value": 0.0,
            "source": "v1_exact_evidence_language_rule",
        },
        "business_materiality": {
            "status": "not_preregistered",
            "rule": "no_business_materiality_claim_may_be_added_after_oot",
        },
        "evidence_language": {
            "strong": "delta_auc_positive_and_holm_p_below_0p05_and_bootstrap_auc_ci_wholly_above_zero",
            "moderate": "delta_auc_positive_and_exactly_one_inferential_criterion",
            "weak": "delta_auc_positive_and_neither_inferential_criterion",
            "not_supported": "delta_auc_nonpositive_or_required_inference_unavailable_or_invalid",
        },
        "unavailable_handling": "visible_not_zero_effect_not_silently_deleted",
        "natural_support": "never_pad_and_label_requested_versus_realized_support",
        "metric_tolerance": METRIC_TOLERANCE,
    }
    _write_frozen_json(audit_root / "materiality_and_failure_rules.json", materiality)
    resource_plan = {
        "schema_version": "prompt_16_final_oot_resource_recovery_plan_v1",
        "status": "frozen_before_oot",
        "precedence": [
            "final_prompt_16_user_instruction_2026_08_16",
            "prompt_16_final_oot_execution_policy_v1",
            "earlier_prompt_16_operational_amendments",
            "base_prompt_16_resource_plan",
        ],
        "parallelism": {
            "max_concurrent_experiment_cells": 1,
            "max_concurrent_folds_or_refits": 1,
            "max_estimator_threads": MAX_ESTIMATOR_THREADS,
            "gpu_enabled": False,
        },
        "memory": {
            "process_tree_rss_hard_cap_gib": PROCESS_TREE_RSS_HARD_CAP_GIB,
            "system_available_ram_hard_floor_gib": SYSTEM_AVAILABLE_RAM_HARD_FLOOR_GIB,
            "soft_available_ram_threshold_gib": SOFT_AVAILABLE_RAM_GIB,
            "resume_available_ram_threshold_gib": RESUME_AVAILABLE_RAM_GIB,
            "resume_stability_polls": RESUME_STABILITY_POLLS,
            "poll_seconds": 5,
            "active_log_seconds": 30,
        },
        "automatic_recovery": {
            "maximum_restarts_per_incomplete_scope": MAX_RESOURCE_RECOVERY_RESTARTS_PER_SCOPE,
            "prestart_ram_wait_consumes_retry": False,
            "completed_cells_skipped": True,
            "partial_artifacts_promoted": False,
            "restart_identity": "same_authorization_data_code_seed_budget_and_checkpoint",
            "retry_counter_reset": "after_scope_completion_and_authentication",
            "identical_command_resume": True,
        },
        "execution_policy": _artifact(root / EXECUTION_POLICY_RELATIVE_PATH, root),
        "ram_wait_policy": _artifact(root / RAM_POLICY_RELATIVE_PATH, root),
    }
    _write_frozen_json(audit_root / "resource_and_automatic_recovery_plan.json", resource_plan)
    analysis_plan = {
        "schema_version": "prompt_16_final_oot_analysis_plan_v1",
        "status": "immutable_frozen_before_oot",
        "execution_authorization_sha256": None,
        "thresholds": dev["threshold_freeze"],
        "paired_inference": {
            "comparison_graph_file": "paired_comparison_graph.json",
            "two_sided_paired_delong": True,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_minimum_valid": BOOTSTRAP_MINIMUM_VALID,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_metrics": ["auc", "ks", "lift_at_10"],
            "confidence_interval": "95_percentile",
        },
        "holm_registry_file": "holm_family_registry.json",
        "evidence_language": materiality["evidence_language"],
        "materiality": materiality,
        "psi_contract": psi_contract,
        "stability_reference": dev["stability_reference"],
        "metric_reconciliation_tolerance": METRIC_TOLERANCE,
    }
    _write_frozen_json(audit_root / "oot_analysis_plan.json", analysis_plan)
    prompt14_roots = [
        root / "configs/protocols/prompt_14_two_dataset_analysis_v1",
        root / "cleanup/audits/prompt_14_two_dataset_oot_review_v3",
    ]
    preservation = {
        "schema_version": "prompt_16_final_oot_preservation_revocation_v1",
        "status": "frozen_before_oot",
        "no_oot_state": no_oot,
        "classical_dev_tree": classical_tree_identity(root / CLASSICAL_DEV_RELATIVE_ROOT),
        "supplemental_dev_tree": _recursive_tree_identity(
            root / SUPPLEMENTAL_DEV_RELATIVE_ROOT
        ),
        "prompt14_closed_tree_identities": [
            {
                "path": _relative(path, root),
                **_recursive_tree_identity(path),
                "numeric_contents_opened_for_prompt16_freeze": False,
            }
            for path in prompt14_roots
        ],
        "semantic_mixed_voter": {
            "status": "unavailable_due_to_unresolved_historical_provenance",
            "execution_cells": 0,
            "absence_is_not_zero_effect": True,
        },
        "revocations": {
            "former_classical_only_oot_template": "revoked",
            "all_earlier_prompt16_oot_commands": "revoked",
            "only_successor": "execution_authorization.json",
        },
        "one_time_rule": {
            "second_scientific_oot_attempt_permitted": False,
            "automatic_recovery_is_continuation_only_when_no_final_metric_promoted": True,
            "completed_authenticated_cells_never_retrained": True,
        },
    }
    _write_frozen_json(
        audit_root / "preservation_deviation_and_revocation_register.json",
        preservation,
    )
    readme = """# Prompt 16 final amended OOT freeze

This directory freezes the complete 170-identity DEV gate and the immutable
34-cell third-dataset OOT protocol before any third-dataset OOT outcome is
opened. The DEV gate contains 123 completed numeric outcomes and 47 visible,
previously frozen unavailable outcomes; all 170 registered identities are
authenticated and accounted for.

The LLM ranking is reused from its recursively sealed cache. No ranking
regeneration or new LLM request is authorized. The semantic/mixed voter remains
unavailable because historical provenance is unresolved and contributes zero
execution cells. Earlier Prompt-16 OOT commands remain revoked.

Resource recovery can continue only the same incomplete scope under identical
scientific configuration, and only when no final artifact was promoted.
Completed authenticated cells are immutable and skipped on resume. The final
overall `_SUCCESS` marker is written after every cell and registered analysis
artifact authenticates.
"""
    write_text_atomic(audit_root / "README.md", readme, overwrite=False)

    implementation_paths = [
        Path("src/credit_risk_fs/experiments/prompt_16_final_oot.py"),
        Path("src/credit_risk_fs/experiments/prompt_16_third_dataset.py"),
        Path("src/credit_risk_fs/experiments/research_logging.py"),
        Path("src/credit_risk_fs/experiments/resource_monitor.py"),
        Path("scripts/run_prompt_16_final_oot.py"),
        EXECUTION_POLICY_RELATIVE_PATH,
        RAM_POLICY_RELATIVE_PATH,
        Path("tests/test_prompt_16_final_oot.py"),
    ]
    for path in implementation_paths:
        if not (root / path).is_file():
            raise Prompt16ExecutionError(f"implementation freeze input is missing: {path}")
    frozen_inputs = [
        PROTOCOL_RELATIVE_PATH,
        SUPPLEMENTAL_AMENDMENT_RELATIVE_PATH,
        SUPPLEMENTAL_AUTHORIZATION_RELATIVE_PATH,
        MATRIX_RELATIVE_ROOT / "manifest.json",
        MATRIX_RELATIVE_ROOT / "metadata.json",
        SUPPLEMENTAL_DEV_RELATIVE_ROOT / "_SUCCESS",
        SUPPLEMENTAL_DEV_RELATIVE_ROOT / "controller_manifest.json",
        SUPPLEMENTAL_DEV_RELATIVE_ROOT / "controller_status.json",
        SUPPLEMENTAL_DEV_RELATIVE_ROOT / "llm_ranking/manifest.json",
        SUPPLEMENTAL_DEV_RELATIVE_ROOT / "llm_ranking/ranking_payload.json",
    ]
    freeze_files = sorted(
        (
            path
            for path in audit_root.iterdir()
            if path.is_file() and path.name != "execution_authorization.json"
        ),
        key=lambda path: path.name,
    )
    authorization = _self_authenticated_payload(
        {
            "schema_version": AUTHORIZATION_SCHEMA_VERSION,
            "status": "authorized_for_one_resumable_final_34_cell_oot_command",
            "operation": "prompt_16_final_amended_oot_once",
            "created_at_utc": _utc_now(),
            "created_before_third_dataset_oot_outcome_inspection": True,
            "implementation_commit": implementation_commit,
            "required_ancestor_commits": list(REQUIRED_ANCESTORS),
            "repository_branch": "main",
            "expected_cells": 34,
            "expected_full_dev_outer_refits": 29,
            "expected_stable_core_internal_component_fits": 10,
            "llm_api_requests_authorized": 0,
            "ranking_regeneration_authorized": False,
            "dev_reruns_authorized": False,
            "oot_registry_sha256": canonical_sha256(cells),
            "full_dev_refit_registry_sha256": canonical_sha256(refits),
            "paired_comparison_graph_sha256": canonical_sha256(graph),
            "dev_evaluation_registry_sha256": dev["evaluation_registry_sha256"],
            "classical_dev_tree_sha256": EXPECTED_CLASSICAL_TREE_SHA256,
            "matrix_manifest_sha256": EXPECTED_MATRIX_MANIFEST_SHA256,
            "feature_universe_sha256": EXPECTED_UNIVERSE_SHA256,
            "implementation_files": [
                _artifact(root / path, root) for path in implementation_paths
            ],
            "frozen_input_files": [_artifact(root / path, root) for path in frozen_inputs],
            "freeze_files": [_artifact(path, root) for path in freeze_files],
            "paths": {
                "repository_root": str(root),
                "output_root": str(root / OOT_RELATIVE_ROOT),
                "log_root": str(root / OOT_LOG_RELATIVE_ROOT),
                "temp_root": str(root / TEMP_RELATIVE_ROOT),
                "terminal_log": str(
                    root / OOT_LOG_RELATIVE_ROOT / "prompt_16_final_amended_oot.log"
                ),
                "controller_status": str(root / OOT_RELATIVE_ROOT / "controller_status.json"),
                "success_marker": str(root / OOT_RELATIVE_ROOT / "_SUCCESS"),
                "evidence_manifest": str(
                    root / OOT_RELATIVE_ROOT / "final_evidence_manifest.json"
                ),
                "analysis_plan": str(audit_root / "oot_analysis_plan.json"),
                "execution_policy": str(root / EXECUTION_POLICY_RELATIVE_PATH),
                "ram_wait_policy": str(root / RAM_POLICY_RELATIVE_PATH),
            },
            "recovery": resource_plan["automatic_recovery"],
            "old_oot_commands": "revoked",
            "second_scientific_oot_attempt": "forbidden",
            "completion_marker_rule": "overall_success_written_last_after_34_cells_and_analysis_authenticate",
        }
    )
    write_json_atomic(
        audit_root / "execution_authorization.json", authorization, overwrite=False
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "freeze_created_without_oot_access",
        "audit_root": str(audit_root),
        "implementation_commit": implementation_commit,
        "authorization_sha256": file_sha256(
            audit_root / "execution_authorization.json"
        ),
        "dev_accounting": dev["accounting"],
        "oot_cells": len(cells),
        "full_dev_refits": len(refits),
        "comparison_graph_entries": len(graph),
        "oot_opened": False,
        "llm_request_executed": False,
    }


def _validate_resource_blocker(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    blocker_path = root / RESOURCE_BLOCKER_RELATIVE_ROOT / "blocker.json"
    blocker = _read_json(blocker_path)
    unsigned = dict(blocker)
    claimed = unsigned.pop("artifact_authentication_sha256", None)
    if claimed != RESOURCE_BLOCKER_AUTHENTICATION_SHA256:
        raise Prompt16ExecutionError("resource blocker authentication identity changed")
    if claimed != canonical_sha256(unsigned):
        raise Prompt16ExecutionError("resource blocker internal digest mismatch")
    if blocker.get("promoted_evaluation_cells") != 0:
        raise Prompt16ExecutionError("resource blocker contains promoted OOT cells")
    if blocker.get("overall_success_marker_present") is not False:
        raise Prompt16ExecutionError("resource blocker unexpectedly records success")

    output = root / OOT_RELATIVE_ROOT
    logs = root / OOT_LOG_RELATIVE_ROOT
    expected_output_files = [
        "classical/scope_authentication.json",
        "controller_status.json",
    ]
    actual_output_files = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    )
    if actual_output_files != expected_output_files:
        raise Prompt16ExecutionError("blocked partial OOT output inventory changed")
    if (output / "_SUCCESS").exists() or (output / "_WORKER_SUCCESS").exists():
        raise Prompt16ExecutionError("blocked partial OOT unexpectedly has success marker")
    if list(output.glob("*/evaluations/cell_*/*")):
        raise Prompt16ExecutionError("blocked partial OOT contains cell artifacts")

    partial_artifacts = [
        _artifact(path, root)
        for path in sorted(
            [
                *(item for item in output.rglob("*") if item.is_file()),
                *(item for item in logs.rglob("*") if item.is_file()),
            ],
            key=lambda item: _relative(item, root),
        )
    ]
    bound = blocker["bound_artifacts"]
    bound_paths = {
        "controller_status_sha256": output / "controller_status.json",
        "scope_authentication_sha256": output / "classical/scope_authentication.json",
        "terminal_log_sha256": logs / "prompt_16_final_amended_oot.log",
        "events_log_sha256": logs / "events.jsonl",
        "debug_log_sha256": logs / "debug.log",
    }
    for key, path in bound_paths.items():
        if file_sha256(path) != bound[key]:
            raise Prompt16ExecutionError(f"blocked execution artifact changed: {key}")
    for row in blocker["attempts"]:
        attempt = int(row["attempt"])
        summary = logs / f"attempts/attempt_{attempt:03d}_supervisor_summary.json"
        trace = logs / f"attempts/attempt_{attempt:03d}_resource_samples.csv"
        if file_sha256(summary) != row["summary_sha256"]:
            raise Prompt16ExecutionError("blocked supervisor summary changed")
        if file_sha256(trace) != row["resource_trace_sha256"]:
            raise Prompt16ExecutionError("blocked resource trace changed")
    return blocker, partial_artifacts


def build_memory_amendment_authorization(
    *,
    repository_root: str | Path = PROJECT_ROOT,
    implementation_commit: str,
) -> dict[str, Any]:
    """Authorize only the audited memory-layout repair after zero promoted cells."""

    root = Path(repository_root).resolve()
    repository = _assert_required_ancestry(root)
    if _git(root, "rev-parse", implementation_commit) != implementation_commit:
        raise Prompt16ExecutionError("implementation commit must be a full commit identity")
    if implementation_commit != repository["head"]:
        raise Prompt16ExecutionError("memory amendment must bind the current committed HEAD")
    amendment_root = root / MEMORY_AMENDMENT_RELATIVE_ROOT
    if amendment_root.exists():
        raise Prompt16ExecutionError(f"memory amendment already exists: {amendment_root}")

    predecessor_path = root / FREEZE_RELATIVE_ROOT / "execution_authorization.json"
    if file_sha256(predecessor_path) != PREDECESSOR_EXECUTION_AUTHORIZATION_SHA256:
        raise Prompt16ExecutionError("predecessor execution authorization changed")
    predecessor_authorization = _read_json(predecessor_path)
    predecessor_unsigned = dict(predecessor_authorization)
    predecessor_claimed = predecessor_unsigned.pop(
        "artifact_authentication_sha256", None
    )
    if predecessor_claimed != canonical_sha256(predecessor_unsigned):
        raise Prompt16ExecutionError("predecessor authorization internal digest mismatch")
    blocker, partial_artifacts = _validate_resource_blocker(root)
    dev = authenticate_complete_dev(root)
    dev.pop("evaluation_rows")
    if dev["accounting"]["authenticated_evaluation_identities"] != 170:
        raise Prompt16ExecutionError("memory amendment DEV gate is not 170/170")

    inherited_fit_sources: list[dict[str, Any]] = []
    fold5_selection_root = root / CLASSICAL_DEV_RELATIVE_ROOT / "fold_5/selection_fits"
    for fit_id in INHERITED_RESOURCE_INFEASIBLE_FIT_IDS:
        selection_path = fold5_selection_root / fit_id / "selection.json"
        selection = _read_json(selection_path)
        if selection.get("status") != "resource_infeasible":
            raise Prompt16ExecutionError(
                f"largest DEV fold resource conclusion changed: {fit_id}"
            )
        inherited_fit_sources.append(_artifact(selection_path, root))

    inherited_cell_sources: list[dict[str, Any]] = []
    for order in INHERITED_RESOURCE_INFEASIBLE_CELL_ORDERS:
        for fold in range(1, 6):
            status_path = (
                root
                / CLASSICAL_DEV_RELATIVE_ROOT
                / f"fold_{fold}/evaluations/cell_{order:03d}/status.json"
            )
            status = _read_json(status_path)
            if status.get("status") != "unavailable" or status.get("reason") != (
                "resource_infeasible"
            ):
                raise Prompt16ExecutionError(
                    f"all-feature DEV resource conclusion changed: fold={fold}, cell={order}"
                )
            inherited_cell_sources.append(_artifact(status_path, root))

    amendment_root.mkdir(parents=True, exist_ok=False)
    amendment = {
        "schema_version": "prompt_16_final_oot_memory_amendment_v1",
        "status": "authorized_execution_only_repair_after_zero_promoted_cells",
        "authorized_by_user_at_utc_date": "2026-08-16",
        "predecessor_execution_authorization_sha256": (
            PREDECESSOR_EXECUTION_AUTHORIZATION_SHA256
        ),
        "resource_blocker_authentication_sha256": (
            RESOURCE_BLOCKER_AUTHENTICATION_SHA256
        ),
        "trigger": {
            "attempt_count": int(blocker["attempt_count"]),
            "failed_scope": blocker["failed_scope"],
            "failed_stage": blocker["failed_stage"],
            "promoted_evaluation_cells": 0,
            "promoted_predictions": 0,
            "promoted_metrics": 0,
        },
        "scientific_contract_unchanged": [
            "34_cell_identity_and_order",
            "matrix_rows_and_temporal_boundaries",
            "feature_universe_and_selected_feature_budgets",
            "selector_and_model_implementations",
            "seeds_hyperparameters_class_handling_and_threshold_rules",
            "paired_comparisons_holm_psi_materiality_and_failure_visibility",
            "24_gib_process_tree_cap_and_4_gib_system_available_floor",
        ],
        "execution_only_changes": [
            "authenticate_scope_with_ids_and_targets_before_feature_projection",
            "never_hold_wide_full_dev_and_wide_oot_frames_together",
            "fit_feasible_selectors_from_full_dev_without_resident_oot_features",
            "compute_feature_psi_in_authenticated_128_feature_batches",
            "load_only_each_cells_selected_features_for_model_fit_and_scoring",
            "release_pyarrow_mimalloc_pages_at_projection_boundaries",
            "migrate_the_exact_zero_cell_predecessor_checkpoint_once",
        ],
        "memory_strategy": (
            "identity_first_batched_psi_and_per_cell_selected_projection_v1"
        ),
        "inherited_resource_infeasible_fit_ids": list(
            INHERITED_RESOURCE_INFEASIBLE_FIT_IDS
        ),
        "inherited_resource_infeasible_cell_orders": list(
            INHERITED_RESOURCE_INFEASIBLE_CELL_ORDERS
        ),
        "inherited_fit_sources": inherited_fit_sources,
        "inherited_cell_sources": inherited_cell_sources,
        "oot_target_or_performance_used_for_resource_inheritance": False,
        "new_llm_request_authorized": False,
        "dev_rerun_authorized": False,
        "second_scientific_oot_attempt_authorized": False,
    }
    _write_frozen_json(amendment_root / "memory_amendment.json", amendment)
    write_text_atomic(
        amendment_root / "README.md",
        """# Prompt 16 final OOT memory amendment v1

This execution-only amendment responds to the authenticated six-attempt RAM
blocker after zero OOT cells, predictions, or metrics were promoted. It keeps
the full frozen scientific contract and hard resource limits unchanged while
preventing simultaneous residency of wide full-DEV and OOT feature frames.

Resource-unavailable states are inherited only from authenticated DEV evidence:
the two all-feature model cells failed for resources in all five folds, and ten
selector fits were already resource-infeasible in the largest temporal DEV fold.
They remain visible unavailable outcomes and are never interpreted as zero
effect. The same 34-cell order remains authoritative.
""",
        overwrite=False,
    )

    implementation_paths = [
        Path("src/credit_risk_fs/experiments/prompt_16_final_oot.py"),
        Path("src/credit_risk_fs/experiments/prompt_16_third_dataset.py"),
        Path("src/credit_risk_fs/experiments/research_logging.py"),
        Path("src/credit_risk_fs/experiments/resource_monitor.py"),
        Path("scripts/run_prompt_16_final_oot.py"),
        EXECUTION_POLICY_RELATIVE_PATH,
        RAM_POLICY_RELATIVE_PATH,
        Path("tests/test_prompt_16_final_oot.py"),
    ]
    immutable_inputs = [
        _artifact(predecessor_path, root),
        _artifact(root / RESOURCE_BLOCKER_RELATIVE_ROOT / "blocker.json", root),
        _artifact(root / RESOURCE_BLOCKER_RELATIVE_ROOT / "README.md", root),
        *inherited_fit_sources,
        *inherited_cell_sources,
    ]
    base_freeze_files = [
        _artifact(path, root)
        for path in sorted(
            (
                item
                for item in (root / FREEZE_RELATIVE_ROOT).iterdir()
                if item.is_file() and item.name != "execution_authorization.json"
            ),
            key=lambda item: item.name,
        )
    ]
    amendment_files = [
        _artifact(path, root)
        for path in sorted(amendment_root.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "execution_authorization.json"
    ]
    authorization_unsigned = dict(predecessor_authorization)
    authorization_unsigned.pop("artifact_authentication_sha256", None)
    authorization_unsigned.update(
        {
            "created_at_utc": _utc_now(),
            "created_before_third_dataset_oot_outcome_inspection": False,
            "created_after_oot_membership_authentication_before_any_oot_prediction_or_metric": True,
            "implementation_commit": implementation_commit,
            "dev_evaluation_registry_sha256": dev["evaluation_registry_sha256"],
            "implementation_files": [
                _artifact(root / path, root) for path in implementation_paths
            ],
            "frozen_input_files": [
                *predecessor_authorization["frozen_input_files"],
                *immutable_inputs,
            ],
            "freeze_files": [*base_freeze_files, *amendment_files],
            "memory_amendment": {
                "path": _relative(amendment_root / "memory_amendment.json", root),
                "strategy": amendment["memory_strategy"],
                "inherited_resource_infeasible_fit_ids": list(
                    INHERITED_RESOURCE_INFEASIBLE_FIT_IDS
                ),
                "inherited_resource_infeasible_cell_orders": list(
                    INHERITED_RESOURCE_INFEASIBLE_CELL_ORDERS
                ),
                "scientific_contract_changed": False,
            },
            "predecessor_partial_state": {
                "execution_authorization_sha256": (
                    PREDECESSOR_EXECUTION_AUTHORIZATION_SHA256
                ),
                "blocker_authentication_sha256": (
                    RESOURCE_BLOCKER_AUTHENTICATION_SHA256
                ),
                "output_file_inventory": [
                    "classical/scope_authentication.json",
                    "controller_status.json",
                ],
                "artifacts": partial_artifacts,
                "migration_permitted_once": True,
                "promoted_cell_count": 0,
            },
        }
    )
    authorization = _self_authenticated_payload(authorization_unsigned)
    write_json_atomic(
        amendment_root / "execution_authorization.json",
        authorization,
        overwrite=False,
    )
    return {
        "schema_version": "prompt_16_final_oot_memory_amendment_build_v1",
        "status": "authorized_memory_only_resume_after_zero_promoted_cells",
        "implementation_commit": implementation_commit,
        "authorization_path": str(amendment_root / "execution_authorization.json"),
        "authorization_sha256": file_sha256(
            amendment_root / "execution_authorization.json"
        ),
        "dev_identities_authenticated": 170,
        "registered_cells_unchanged": 34,
        "promoted_predecessor_cells": 0,
        "oot_work_executed": False,
    }


def load_final_authorization(
    path: str | Path,
    *,
    repository_root: str | Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], str]:
    root = Path(repository_root).resolve()
    candidate = Path(path).resolve()
    payload = _read_json(candidate)
    if not isinstance(payload, dict):
        raise Prompt16ExecutionError("final authorization must be a JSON object")
    if payload.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        raise Prompt16ExecutionError("final authorization schema changed")
    if payload.get("status") != "authorized_for_one_resumable_final_34_cell_oot_command":
        raise Prompt16ExecutionError("final authorization is not active")
    claimed = payload.get("artifact_authentication_sha256")
    unsigned = dict(payload)
    unsigned.pop("artifact_authentication_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise Prompt16ExecutionError("final authorization internal digest mismatch")
    repository = _assert_required_ancestry(root)
    implementation_commit = str(payload.get("implementation_commit", ""))
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, repository["head"]],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode != 0:
        raise Prompt16ExecutionError("authorized implementation commit is not an ancestor")
    for section in ("implementation_files", "frozen_input_files", "freeze_files"):
        rows = payload.get(section, [])
        if not rows:
            raise Prompt16ExecutionError(f"authorization file registry is empty: {section}")
        for row in rows:
            artifact = root / str(row.get("path", ""))
            if (
                not artifact.is_file()
                or artifact.stat().st_size != int(row.get("byte_size", -1))
                or file_sha256(artifact) != row.get("sha256")
            ):
                raise Prompt16ExecutionError(
                    f"authorized input changed: {row.get('path')}"
                )
    if payload.get("expected_cells") != 34:
        raise Prompt16ExecutionError("authorized OOT cell count changed")
    if payload.get("llm_api_requests_authorized") != 0:
        raise Prompt16ExecutionError("final authorization unexpectedly permits an LLM request")
    memory_amendment = payload.get("memory_amendment", {})
    if memory_amendment.get("strategy") != (
        "identity_first_batched_psi_and_per_cell_selected_projection_v1"
    ):
        raise Prompt16ExecutionError("memory-bounded OOT strategy is not authorized")
    if tuple(memory_amendment.get("inherited_resource_infeasible_fit_ids", [])) != (
        INHERITED_RESOURCE_INFEASIBLE_FIT_IDS
    ):
        raise Prompt16ExecutionError("resource-infeasible selector registry changed")
    if tuple(
        int(value)
        for value in memory_amendment.get(
            "inherited_resource_infeasible_cell_orders", []
        )
    ) != INHERITED_RESOURCE_INFEASIBLE_CELL_ORDERS:
        raise Prompt16ExecutionError("resource-infeasible model-cell registry changed")
    predecessor = payload.get("predecessor_partial_state", {})
    if predecessor.get("execution_authorization_sha256") != (
        PREDECESSOR_EXECUTION_AUTHORIZATION_SHA256
    ):
        raise Prompt16ExecutionError("predecessor OOT authorization identity changed")
    return payload, file_sha256(candidate)


def run_final_oot_worker(
    *,
    repository_root: str,
    authorization_path: str,
    stop_event: Any = None,
    stage_queue: Any = None,
    ram_ready_event: Any = None,
    **_controls: Any,
) -> dict[str, Any]:
    """Run the complete 34-cell lifecycle after parent-side preflight passes."""

    project = Path(repository_root).resolve()
    authorization, authorization_sha = load_final_authorization(
        authorization_path, repository_root=project
    )
    output = Path(authorization["paths"]["output_root"])
    final_success = output / "_SUCCESS"
    worker_success = output / "_WORKER_SUCCESS"
    final_manifest_path = output / "final_evidence_manifest.json"
    if final_success.is_file() or worker_success.is_file():
        marker = final_success if final_success.is_file() else worker_success
        success = _read_json(marker)
        if success.get("final_evidence_manifest_sha256") != file_sha256(final_manifest_path):
            raise Prompt16ExecutionError("final Prompt-16 OOT completion marker mismatch")
        manifest = _read_json(final_manifest_path)
        if manifest.get("execution_authorization_sha256") != authorization_sha:
            raise Prompt16ExecutionError("completed final OOT authorization changed")
        return {**manifest, "reused_completed_final_oot": True}
    output.mkdir(parents=True, exist_ok=True)

    from credit_risk_fs.experiments.prompt_16_third_dataset import run_phase_worker

    classical = run_phase_worker(
        matrix_root=str(project / MATRIX_RELATIVE_ROOT),
        output_root=str(output / "classical"),
        protocol_lock=str(project / PROTOCOL_RELATIVE_PATH),
        phase="oot",
        fold_id=None,
        oot_analysis_plan=None,
        stop_event=stop_event,
        stage_queue=stage_queue,
        ram_ready_event=ram_ready_event,
        execution_authorization_sha256=authorization_sha,
        allow_authenticated_oot_recovery=True,
        full_dev_training_ks_threshold=True,
        memory_bounded_oot=True,
        inherited_resource_infeasible_fit_ids=(
            INHERITED_RESOURCE_INFEASIBLE_FIT_IDS
        ),
        inherited_resource_infeasible_cell_orders=(
            INHERITED_RESOURCE_INFEASIBLE_CELL_ORDERS
        ),
    )
    supplemental = run_supplemental_oot_worker(
        repository_root=str(project),
        output_root=str(output / "supplemental"),
        authorization_sha256=authorization_sha,
        stop_event=stop_event,
        stage_queue=stage_queue,
        ram_ready_event=ram_ready_event,
    )
    analysis = run_final_oot_analysis(
        repository_root=project,
        output_root=output,
        authorization_sha256=authorization_sha,
        analysis_plan_path=authorization["paths"]["analysis_plan"],
        stop_event=stop_event,
        stage_queue=stage_queue,
    )
    evaluation_manifests: list[dict[str, Any]] = []
    for order in range(1, 35):
        path = _oot_cell_path(output, order) / "manifest.json"
        if not path.is_file():
            raise Prompt16ExecutionError(f"final OOT cell manifest missing: {order}")
        evaluation_manifests.append(
            {"configuration_order": order, "sha256": file_sha256(path)}
        )
    final_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "operation": "prompt_16_final_amended_oot_once",
        "execution_authorization_sha256": authorization_sha,
        "oot_started": True,
        "expected_evaluations": 34,
        "accounted_evaluations": 34,
        "classical_phase_manifest_sha256": file_sha256(
            output / "classical/phase_manifest.json"
        ),
        "supplemental_phase_manifest_sha256": file_sha256(
            output / "supplemental/phase_manifest.json"
        ),
        "analysis_manifest_sha256": file_sha256(
            output / "analysis/analysis_manifest.json"
        ),
        "evaluation_manifests": evaluation_manifests,
        "evaluation_manifest_registry_sha256": canonical_sha256(
            evaluation_manifests
        ),
        "llm_api_request_count": 0,
        "ranking_regeneration_count": 0,
        "classical_dev_rerun_count": 0,
        "supplemental_dev_rerun_count": 0,
        "classical_phase": classical,
        "supplemental_phase": supplemental,
        "analysis": analysis,
        "completed_at_utc": _utc_now(),
    }
    write_json_atomic(final_manifest_path, final_manifest)
    write_text_atomic(
        worker_success,
        json.dumps(
            {"final_evidence_manifest_sha256": file_sha256(final_manifest_path)},
            sort_keys=True,
        )
        + "\n",
    )
    return final_manifest


__all__ = [
    "AUTHORIZATION_SCHEMA_VERSION",
    "MAX_RESOURCE_RECOVERY_RESTARTS_PER_SCOPE",
    "authenticate_complete_dev",
    "build_freeze",
    "final_full_dev_refits",
    "final_oot_cells",
    "load_final_authorization",
    "paired_comparison_graph",
    "run_final_oot_analysis",
    "run_final_oot_worker",
    "run_supplemental_oot_worker",
]
