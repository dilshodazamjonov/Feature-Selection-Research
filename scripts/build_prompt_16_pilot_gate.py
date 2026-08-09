"""Authenticate the Prompt-16 pilot and build the pre-DEV gate artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


from credit_risk_fs.data.homecredit_model_stability_2024.adapter import (  # noqa: E402
    validate_output_manifest,
)
from credit_risk_fs.data.homecredit_model_stability_2024.contract import (  # noqa: E402
    file_sha256,
)
from credit_risk_fs.evaluation.metrics import evaluate_model  # noqa: E402
from credit_risk_fs.experiments.atomic_io import (  # noqa: E402
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)
from credit_risk_fs.experiments.prompt_16_third_dataset import (  # noqa: E402
    EXPECTED_PROTOCOL_FILE_SHA256,
    EXPECTED_PROTOCOL_INTERNAL_SHA256,
    Prompt16ExecutionError,
    _evaluation_identity,
    _fit_identity,
    _load_sealed,
    _protocol_payload,
    _ranking_utility,
    canonical_registry,
    load_execution_plan,
)


GIB = 1024**3
ALLOWANCE = 1.20
SCHEMA_VERSION = "prompt_16_pilot_gate_v1"


def _json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Prompt16ExecutionError(f"expected JSON object: {path}")
    return value


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_exact_seal_inventory(path: Path, manifest: Mapping[str, Any]) -> None:
    declared = {"manifest.json", "_SUCCESS"} | {
        str(item["path"]) for item in manifest.get("artifacts", [])
    }
    observed = {item.name for item in path.iterdir() if item.is_file()}
    if observed != declared:
        raise Prompt16ExecutionError(
            f"sealed artifact inventory mismatch: {path}; "
            f"missing={sorted(declared - observed)}, extra={sorted(observed - declared)}"
        )


def _metrics_reconcile(predictions: pd.DataFrame, stored: Mapping[str, Any]) -> None:
    thresholds = predictions["decision_threshold"].drop_duplicates().tolist()
    if len(thresholds) != 1:
        raise Prompt16ExecutionError("predictions do not contain one frozen threshold")
    y_true = predictions["target"].to_numpy(dtype=np.int64, copy=False)
    score = predictions["score"].to_numpy(dtype=float, copy=False)
    recalculated = evaluate_model(y_true, score, threshold=float(thresholds[0]))
    recalculated.update(_ranking_utility(y_true, score))
    if set(recalculated) != set(stored):
        raise Prompt16ExecutionError("saved metric key set does not reconcile")
    for key, expected in stored.items():
        observed = recalculated[key]
        if isinstance(expected, int) and not isinstance(expected, bool):
            matches = int(observed) == expected
        else:
            matches = bool(
                np.isclose(
                    float(observed),
                    float(expected),
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=True,
                )
            )
        if not matches:
            raise Prompt16ExecutionError(
                f"saved metric does not reconcile: {key}: {expected} != {observed}"
            )


def _resource_summary(
    pilot_root: Path, log_root: Path, prior_fix_record: Path
) -> dict[str, Any]:
    cell_1 = _json(pilot_root / "evaluations/cell_001/status.json")["supervisor"]
    cell_2 = _json(pilot_root / "evaluations/cell_002/status.json")["supervisor"]
    completed = _json(log_root / "pilot_fold_1_supervisor_summary.json")
    prior = _json(prior_fix_record)["triggers"]["demonstrated_ram_stall"]
    sessions = [
        {
            "name": "pre_fix_checkpoint_session",
            "status": prior["status_after_user_stop"],
            "active_seconds": float(prior["active_computation_seconds"]),
            "peak_rss_bytes": int(prior["peak_process_tree_rss_bytes"]),
            "minimum_available_bytes": int(prior["minimum_system_available_ram_bytes"]),
            "included_for_runtime": True,
        },
        {
            "name": "post_fix_selectors_and_cell_001",
            "status": cell_1["status"],
            "active_seconds": float(cell_1["active_computation_seconds"]),
            "peak_rss_bytes": int(cell_1["peak_process_tree_rss_bytes"]),
            "minimum_available_bytes": int(cell_1["minimum_system_available_ram_bytes"]),
            "included_for_runtime": True,
        },
        {
            "name": "cell_002_resource_infeasibility",
            "status": cell_2["status"],
            "active_seconds": float(cell_2["active_computation_seconds"]),
            "peak_rss_bytes": int(cell_2["peak_process_tree_rss_bytes"]),
            "minimum_available_bytes": int(cell_2["minimum_system_available_ram_bytes"]),
            "included_for_runtime": True,
        },
        {
            "name": "cells_003_through_030_completion",
            "status": completed["status"],
            "active_seconds": float(completed["active_computation_seconds"]),
            "peak_rss_bytes": int(completed["peak_process_tree_rss_bytes"]),
            "minimum_available_bytes": int(completed["minimum_system_available_ram_bytes"]),
            "included_for_runtime": True,
        },
    ]
    return {
        "sessions": sessions,
        "total_active_seconds": sum(item["active_seconds"] for item in sessions),
        "peak_process_tree_rss_bytes": max(item["peak_rss_bytes"] for item in sessions),
        "minimum_system_available_ram_bytes": min(
            item["minimum_available_bytes"] for item in sessions
        ),
        "completed_cells_peak_process_tree_rss_bytes": int(
            completed["peak_process_tree_rss_bytes"]
        ),
        "completed_cells_minimum_system_available_ram_bytes": int(
            completed["minimum_system_available_ram_bytes"]
        ),
    }


def _forecast_rows(
    *,
    split: Mapping[str, Any],
    selector_seconds: float,
    encoding_seconds: float,
    preprocessing_seconds: float,
    training_seconds: float,
    prediction_seconds: float,
    metric_seconds: float,
    packaging_seconds: float,
    total_active_seconds: float,
    selection_bytes: int,
    evaluation_bytes: int,
    log_bytes: int,
) -> list[dict[str, Any]]:
    folds = list(split["folds"])
    pilot_train = float(folds[0]["train"]["rows"])
    pilot_validation = float(folds[0]["validation"]["rows"])
    train_ratio_sum = sum(float(item["train"]["rows"]) / pilot_train for item in folds)
    validation_ratio_sum = sum(
        float(item["validation"]["rows"]) / pilot_validation for item in folds
    )
    combined_ratio_sum = sum(
        (float(item["train"]["rows"]) + float(item["validation"]["rows"]))
        / (pilot_train + pilot_validation)
        for item in folds
    )
    oot_train_ratio = float(split["dev"]["rows"]) / pilot_train
    oot_validation_ratio = float(split["oot"]["rows"]) / pilot_validation
    oot_combined_ratio = (
        float(split["dev"]["rows"]) + float(split["oot"]["rows"])
    ) / (pilot_train + pilot_validation)
    measured_components = (
        selector_seconds
        + encoding_seconds
        + preprocessing_seconds
        + training_seconds
        + prediction_seconds
        + metric_seconds
        + packaging_seconds
    )
    overhead_seconds = max(0.0, total_active_seconds - measured_components)

    def item(
        component: str,
        pilot_seconds: float,
        dev_scale: float,
        oot_scale: float,
        basis: str,
    ) -> dict[str, Any]:
        return {
            "measure": "runtime_seconds",
            "component": component,
            "pilot_measured": pilot_seconds,
            "dev_scale": dev_scale,
            "dev_forecast_with_20pct": pilot_seconds * dev_scale * ALLOWANCE,
            "oot_scale": oot_scale,
            "oot_forecast_with_20pct": pilot_seconds * oot_scale * ALLOWANCE,
            "scaling_basis": basis,
        }

    rows = [
        item(
            "adapter_matrix_build",
            0.0,
            0.0,
            0.0,
            "matrix is already authenticated and reused; actual build is reported separately",
        ),
        item(
            "selector_fit",
            selector_seconds,
            train_ratio_sum,
            oot_train_ratio,
            "sum of fold/full-DEV training-row ratios",
        ),
        item(
            "selection_encoding",
            encoding_seconds,
            train_ratio_sum,
            oot_train_ratio,
            "training-row ratio",
        ),
        item(
            "model_preprocessing",
            preprocessing_seconds,
            combined_ratio_sum,
            oot_combined_ratio,
            "train-plus-held-out row ratio",
        ),
        item(
            "model_training",
            training_seconds,
            train_ratio_sum,
            oot_train_ratio,
            "training-row ratio for the 28 executable cells",
        ),
        item(
            "prediction",
            prediction_seconds,
            validation_ratio_sum,
            oot_validation_ratio,
            "validation/OOT-row ratio",
        ),
        item(
            "metric_calculation",
            metric_seconds,
            validation_ratio_sum,
            oot_validation_ratio,
            "validation/OOT-row ratio",
        ),
        item(
            "checkpoint_packaging",
            packaging_seconds,
            validation_ratio_sum,
            oot_validation_ratio,
            "prediction-row ratio; selector checkpoint overhead is in operational overhead",
        ),
        item(
            "data_loading_resource_stops_and_operational_overhead",
            overhead_seconds,
            combined_ratio_sum,
            oot_combined_ratio,
            "train-plus-held-out row ratio; includes the two visible resource stops",
        ),
    ]

    def disk_item(
        component: str,
        pilot_bytes: int,
        dev_scale: float,
        oot_scale: float,
        basis: str,
    ) -> dict[str, Any]:
        return {
            "measure": "disk_bytes",
            "component": component,
            "pilot_measured": pilot_bytes,
            "dev_scale": dev_scale,
            "dev_forecast_with_20pct": math.ceil(pilot_bytes * dev_scale * ALLOWANCE),
            "oot_scale": oot_scale,
            "oot_forecast_with_20pct": math.ceil(pilot_bytes * oot_scale * ALLOWANCE),
            "scaling_basis": basis,
        }

    rows.extend(
        [
            disk_item(
                "selection_artifacts",
                selection_bytes,
                len(folds),
                1.0,
                "one 27-fit selection package per DEV fold or full-DEV refit",
            ),
            disk_item(
                "evaluation_artifacts",
                evaluation_bytes,
                validation_ratio_sum,
                oot_validation_ratio,
                "prediction-row ratio",
            ),
            disk_item(
                "logs_and_resource_samples",
                log_bytes,
                train_ratio_sum,
                oot_train_ratio,
                "conservative active-runtime proxy",
            ),
        ]
    )
    return rows


def build_gate(plan_path: Path, output_root: Path) -> dict[str, Any]:
    plan = load_execution_plan(plan_path)
    paths = plan["paths"]
    protocol_lock = Path(paths["protocol_lock"])
    matrix_root = Path(paths["matrix_root"])
    pilot_root = Path(paths["pilot_root"]) / "fold_1"
    log_root = Path(paths["log_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    if Path(paths["oot_root"]).exists():
        raise Prompt16ExecutionError("OOT root exists before the pilot gate")

    contract, protocol = _protocol_payload(protocol_lock)
    registry = canonical_registry(protocol_lock)
    matrix_spec = protocol["approved_protocol"]["method_and_evaluation_matrix"]
    split = protocol["approved_protocol"]["split_and_fold_boundaries"]
    matrix_manifest = validate_output_manifest(matrix_root)
    matrix_manifest_sha = file_sha256(matrix_root / "manifest.json")
    matrix_success_sha = file_sha256(matrix_root / "_SUCCESS")
    if matrix_manifest["summary"] != {
        "row_count": 1_526_659,
        "predictor_count": 1_959,
        "matrix_part_count": 31,
        "depth_2_files_opened": 0,
        "fits": 0,
        "evaluations": 0,
    }:
        raise Prompt16ExecutionError("matrix summary differs from the frozen gate")
    included_inventory = list(matrix_manifest["input_inventory"]["artifacts"])
    if len(included_inventory) != 19:
        raise Prompt16ExecutionError("authenticated included inventory is not 19 records")

    phase_success = _json(pilot_root / "_SUCCESS")
    phase_manifest_path = pilot_root / "phase_manifest.json"
    phase_manifest_sha = file_sha256(phase_manifest_path)
    if phase_success.get("phase_manifest_sha256") != phase_manifest_sha:
        raise Prompt16ExecutionError("pilot phase marker does not authenticate")
    phase_manifest = _json(phase_manifest_path)
    accounting = _json(pilot_root / "accounting.json")
    scope = _json(pilot_root / "scope_authentication.json")
    if not (
        phase_manifest.get("status") == "complete"
        and phase_manifest.get("phase") == "pilot"
        and phase_manifest.get("fold_id") == 1
        and phase_manifest.get("protocol_file_sha256")
        == EXPECTED_PROTOCOL_FILE_SHA256
        and phase_manifest.get("protocol_internal_sha256")
        == EXPECTED_PROTOCOL_INTERNAL_SHA256
        and phase_manifest.get("matrix_manifest_sha256") == matrix_manifest_sha
    ):
        raise Prompt16ExecutionError("pilot phase identity mismatch")
    if not (
        scope["train"]["authenticated"]
        and scope["validation"]["authenticated"]
        and scope["train"]["expected"] == scope["train"]["observed"]
        and scope["validation"]["expected"] == scope["validation"]["observed"]
        and scope["case_id_overlap"] == 0
    ):
        raise Prompt16ExecutionError("pilot scope authentication failed")

    fits = list(registry["fit_registry"])
    cells = list(registry["matrix_cells"])
    expected_fit_dirs = {str(item["fit_id"]) for item in fits}
    observed_fit_dirs = {
        item.name for item in (pilot_root / "selection_fits").iterdir() if item.is_dir()
    }
    if observed_fit_dirs != expected_fit_dirs:
        raise Prompt16ExecutionError("pilot selector directory accounting mismatch")
    expected_eval_dirs = {
        f"cell_{int(item['configuration_order']):03d}" for item in cells
    }
    observed_eval_dirs = {
        item.name for item in (pilot_root / "evaluations").iterdir() if item.is_dir()
    }
    if observed_eval_dirs != expected_eval_dirs:
        raise Prompt16ExecutionError("pilot evaluation directory accounting mismatch")

    accounting_rows: list[dict[str, Any]] = []
    fit_by_order: dict[int, Mapping[str, Any]] = {}
    selector_seconds = 0.0
    for fit in fits:
        fit_path = pilot_root / "selection_fits" / str(fit["fit_id"])
        identity = _fit_identity(
            phase="pilot",
            fold_id=1,
            fit=fit,
            matrix_manifest_sha256=matrix_manifest_sha,
        )
        manifest = _load_sealed(fit_path, identity)
        if manifest is None:
            raise Prompt16ExecutionError(f"selector is not sealed: {fit['fit_id']}")
        _assert_exact_seal_inventory(fit_path, manifest)
        selection = _json(fit_path / "selection.json")
        selected = list(selection.get("selected_features", []))
        if (
            selection.get("fit_spec") != fit
            or selection.get("realized_support") != len(selected)
            or len(selected) != len(set(selected))
            or selection.get("natural_support_unpadded") is not True
        ):
            raise Prompt16ExecutionError(f"selector evidence mismatch: {fit['fit_id']}")
        if selection.get("status") not in {
            "complete",
            "infeasible_natural_support",
            "failed",
            "resource_infeasible",
        }:
            raise Prompt16ExecutionError(f"unknown selector status: {fit['fit_id']}")
        selector_seconds += float(selection["fit_seconds"])
        for order in fit["dependent_configuration_orders"]:
            fit_by_order[int(order)] = fit
        accounting_rows.append(
            {
                "record_kind": "selection_fit",
                "record_id": fit["fit_id"],
                "fit_order": fit["fit_order"],
                "configuration_order": None,
                "fit_id": fit["fit_id"],
                "method_id": fit["method_id"],
                "model": None,
                "requested_feature_budget": fit.get("requested_feature_budget"),
                "realized_support": len(selected),
                "status": selection["status"],
                "reason": selection.get("error"),
                "natural_support_like_for_like": None,
                "prediction_rows": None,
                "prediction_alignment_authenticated": None,
                "metric_recalculation_authenticated": None,
                "fit_seconds": selection["fit_seconds"],
                "preprocessing_seconds": None,
                "training_seconds": None,
                "prediction_seconds": None,
                "metric_seconds": None,
                "total_seconds": selection["fit_seconds"],
                "manifest_sha256": file_sha256(fit_path / "manifest.json"),
                "artifact_path": fit_path.relative_to(PROJECT_ROOT).as_posix(),
            }
        )

    validation_expected = scope["validation"]["observed"]
    completed_evaluations = 0
    unavailable_evaluations = 0
    metric_recalculations = 0
    preprocessing_seconds = 0.0
    training_seconds = 0.0
    prediction_seconds = 0.0
    metric_seconds = 0.0
    packaging_seconds = 0.0
    unavailable_cells: list[dict[str, Any]] = []
    for cell in cells:
        order = int(cell["configuration_order"])
        fit = fit_by_order[order]
        fit_path = pilot_root / "selection_fits" / str(fit["fit_id"])
        selection_manifest_sha = file_sha256(fit_path / "manifest.json")
        cell_id = f"cell_{order:03d}"
        cell_path = pilot_root / "evaluations" / cell_id
        identity = _evaluation_identity(
            phase="pilot",
            fold_id=1,
            cell=cell,
            matrix_manifest_sha256=matrix_manifest_sha,
            selection_manifest_sha256=selection_manifest_sha,
        )
        manifest = _load_sealed(cell_path, identity)
        if manifest is None:
            raise Prompt16ExecutionError(f"evaluation is not sealed: {cell_id}")
        _assert_exact_seal_inventory(cell_path, manifest)
        status = _json(cell_path / "status.json")
        if status.get("configuration_order") != order or status.get("cell") != cell:
            raise Prompt16ExecutionError(f"evaluation identity mismatch: {cell_id}")
        row: dict[str, Any] = {
            "record_kind": "evaluation",
            "record_id": cell_id,
            "fit_order": fit["fit_order"],
            "configuration_order": order,
            "fit_id": fit["fit_id"],
            "method_id": cell["method_id"],
            "model": cell["model"],
            "requested_feature_budget": status.get("requested_feature_budget"),
            "realized_support": status.get("realized_support"),
            "status": status["status"],
            "reason": status.get("reason"),
            "natural_support_like_for_like": status.get("natural_support_like_for_like"),
            "prediction_rows": None,
            "prediction_alignment_authenticated": False,
            "metric_recalculation_authenticated": False,
            "fit_seconds": None,
            "preprocessing_seconds": None,
            "training_seconds": None,
            "prediction_seconds": None,
            "metric_seconds": None,
            "total_seconds": status.get("elapsed_seconds"),
            "manifest_sha256": file_sha256(cell_path / "manifest.json"),
            "artifact_path": cell_path.relative_to(PROJECT_ROOT).as_posix(),
        }
        if status["status"] == "complete":
            predictions = pd.read_parquet(
                cell_path / "predictions.parquet",
                columns=("case_id", "target", "score", "decision_threshold"),
            )
            observed_alignment = status["prediction_alignment"]
            if not (
                len(predictions) == int(validation_expected["rows"])
                and observed_alignment["row_count"] == int(validation_expected["rows"])
                and observed_alignment["ordered_case_id_sha256"]
                == validation_expected["ordered_case_id_sha256"]
                and observed_alignment["ordered_case_id_target_sha256"]
                == validation_expected["ordered_case_id_target_sha256"]
            ):
                raise Prompt16ExecutionError(f"prediction alignment mismatch: {cell_id}")
            stored_metrics = _json(cell_path / "metrics.json")
            _metrics_reconcile(predictions, stored_metrics)
            execution = _json(cell_path / "execution.json")
            timings = execution["timings"]
            if execution["configuration"]["validation_target_used_for_fit"] is not False:
                raise Prompt16ExecutionError(f"validation target entered fit: {cell_id}")
            preprocessing_seconds += float(timings["preprocessing_seconds"])
            training_seconds += float(timings["training_seconds"])
            prediction_seconds += float(timings["prediction_seconds"])
            metric_seconds += float(timings["evaluation_seconds"])
            packaging_seconds += max(
                0.0, float(status["elapsed_seconds"]) - float(timings["total_seconds"])
            )
            row.update(
                {
                    "prediction_rows": len(predictions),
                    "prediction_alignment_authenticated": True,
                    "metric_recalculation_authenticated": True,
                    "preprocessing_seconds": timings["preprocessing_seconds"],
                    "training_seconds": timings["training_seconds"],
                    "prediction_seconds": timings["prediction_seconds"],
                    "metric_seconds": timings["evaluation_seconds"],
                    "total_seconds": timings["total_seconds"],
                }
            )
            completed_evaluations += 1
            metric_recalculations += 1
            del predictions
        elif status["status"] == "unavailable" and status.get("reason") == "resource_infeasible":
            supervisor = status.get("supervisor", {})
            if supervisor.get("stop_code") not in {"ram_process_limit", "wall_clock_limit"}:
                raise Prompt16ExecutionError(f"unpermitted unavailable cell: {cell_id}")
            unavailable_evaluations += 1
            unavailable_cells.append(
                {
                    "configuration_order": order,
                    "method_id": cell["method_id"],
                    "model": cell["model"],
                    "stop_code": supervisor["stop_code"],
                    "peak_process_tree_rss_bytes": supervisor[
                        "peak_process_tree_rss_bytes"
                    ],
                }
            )
        else:
            raise Prompt16ExecutionError(f"unpermitted evaluation status: {cell_id}")
        accounting_rows.append(row)

    if (
        len(accounting_rows) != 57
        or len(fits) != 27
        or len(cells) != 30
        or completed_evaluations != 28
        or unavailable_evaluations != 2
        or metric_recalculations != 28
        or accounting.get("accounted_evaluations") != 30
        or accounting.get("all_registered_cells_visible") is not True
    ):
        raise Prompt16ExecutionError("pilot 27/30 accounting does not reconcile")

    resources = _resource_summary(
        pilot_root,
        log_root,
        output_root / "implementation_fix_003_one_gib_selector_memory_release.json",
    )
    encoding_seconds = float(phase_manifest["selector_encoding"]["elapsed_seconds"])
    selection_bytes = _directory_size(pilot_root / "selection_fits")
    evaluation_bytes = _directory_size(pilot_root / "evaluations")
    log_bytes = _directory_size(log_root)
    forecast_rows = _forecast_rows(
        split=split,
        selector_seconds=selector_seconds,
        encoding_seconds=encoding_seconds,
        preprocessing_seconds=preprocessing_seconds,
        training_seconds=training_seconds,
        prediction_seconds=prediction_seconds,
        metric_seconds=metric_seconds,
        packaging_seconds=packaging_seconds,
        total_active_seconds=float(resources["total_active_seconds"]),
        selection_bytes=selection_bytes,
        evaluation_bytes=evaluation_bytes,
        log_bytes=log_bytes,
    )
    runtime_rows = [item for item in forecast_rows if item["measure"] == "runtime_seconds"]
    disk_rows = [item for item in forecast_rows if item["measure"] == "disk_bytes"]
    dev_runtime = sum(float(item["dev_forecast_with_20pct"]) for item in runtime_rows)
    oot_runtime = sum(float(item["oot_forecast_with_20pct"]) for item in runtime_rows)
    dev_disk = sum(int(item["dev_forecast_with_20pct"]) for item in disk_rows)
    oot_disk = sum(int(item["oot_forecast_with_20pct"]) for item in disk_rows)
    maximum_fold_scale = max(
        float(item["train"]["rows"]) / float(split["folds"][0]["train"]["rows"])
        for item in split["folds"]
    )
    maximum_dev_fold_seconds = (
        float(resources["total_active_seconds"]) * maximum_fold_scale * ALLOWANCE
    )
    results_free_bytes = int(
        __import__("shutil").disk_usage(Path(paths["pilot_root"])).free
    )
    dev_fold_limit = float(plan["wall_time_limits_seconds"]["dev_fold_total"])
    oot_limit = float(plan["wall_time_limits_seconds"]["oot_total"])
    disk_floor = max(
        int(plan["resource_controls"]["disk_free_hard_floor_bytes"]),
        math.ceil(
            float(plan["resource_controls"]["disk_remaining_output_safety_factor"])
            * (dev_disk + oot_disk)
        ),
    )
    feasibility_checks = {
        "maximum_dev_fold_forecast_with_20pct_within_limit": maximum_dev_fold_seconds
        < dev_fold_limit,
        "oot_forecast_with_20pct_within_limit": oot_runtime < oot_limit,
        "disk_free_above_reforecast_floor": results_free_bytes >= disk_floor,
        "completed_non_full_feature_cells_peak_below_process_cap": int(
            resources["completed_cells_peak_process_tree_rss_bytes"]
        )
        < 24 * GIB,
        "resource_infeasible_cells_visible_and_not_dropped": len(unavailable_cells) == 2,
    }
    if not all(feasibility_checks.values()):
        raise Prompt16ExecutionError(f"pilot feasibility gate failed: {feasibility_checks}")

    accounting_path = output_root / "pilot_accounting.csv"
    forecast_path = output_root / "pilot_resource_forecast.csv"
    decision_path = output_root / "pilot_feasibility_decision.md"
    write_csv_atomic(accounting_path, pd.DataFrame(accounting_rows))
    write_csv_atomic(forecast_path, pd.DataFrame(forecast_rows))

    matrix_summary = _json(log_root / "matrix_supervisor_summary.json")
    decision_text = f"""# Prompt 16 authenticated pilot feasibility decision

## Decision

The resource pilot passes with the lock-declared resource-infeasibility rule applied. All 27 selector fits and all 30 evaluation cells are authenticated and visible. Twenty-eight evaluations completed; the two full-feature cells (LR and CatBoost) exceeded the frozen 24 GiB owned-process cap and remain sealed as unavailable rather than being dropped, approximated, or retried under altered settings.

DEV is authorized only after this gate is committed. OOT remains closed until complete authenticated five-fold DEV and the separate DEV-freeze commit.

## Authenticated scope and accounting

- Matrix: 1,526,659 rows, 1,959 predictors, 31 Parquet parts, and zero depth-2 files opened.
- Raw inventory: 19 authenticated included records; all declared depth-2 files remained excluded.
- Pilot fold: 200,661 training rows and 204,567 validation rows with zero case-ID overlap.
- Checkpoints: 27/27 selector seals and 30/30 evaluation seals; 28 complete and 2 resource-infeasible.
- Predictions and metrics: all 28 completed prediction files match the frozen validation identity and all saved metrics recalculate within 1e-12 absolute/relative tolerance.
- Leakage control: every completed evaluation records validation_target_used_for_fit=false; selection encoding is training-only and natural support is never padded.

## Resource feasibility

The authenticated matrix build took {float(matrix_summary['active_computation_seconds']) / 3600:.2f} active hours and peaked at {int(matrix_summary['peak_process_tree_rss_bytes']) / GIB:.2f} GiB. The pilot's reconciled active work took {float(resources['total_active_seconds']) / 3600:.2f} hours. The two full-feature attempts produced the overall {int(resources['peak_process_tree_rss_bytes']) / GIB:.2f} GiB sampled peak and are explicitly unavailable. Cells 003-030 completed in the final session with a {int(resources['completed_cells_peak_process_tree_rss_bytes']) / GIB:.2f} GiB peak.

With row-count scaling and the required 20% operational allowance, five-fold DEV is forecast at {dev_runtime / 3600:.1f} active hours and OOT at {oot_runtime / 3600:.1f} active hours. The largest DEV-fold bound is {maximum_dev_fold_seconds / 3600:.1f} hours versus the frozen 168-hour fold limit; OOT is below its 168-hour limit. Forecast incremental DEV plus OOT output is {(dev_disk + oot_disk) / GIB:.2f} GiB, while {results_free_bytes / GIB:.1f} GiB is currently free and the recalculated launch floor is {disk_floor / GIB:.1f} GiB.

These are resource forecasts, not performance conclusions. No pilot metric was used to change a method, model, budget, seed, split, fold, or comparison.

## Required next step

Commit this second Prompt-16 gate, validate the exact fold-scoped DEV command, and run folds 1 through 5 sequentially with checkpoint resume. Preserve full-feature resource-infeasibility in every fold where it recurs. Do not open OOT or the prior two-dataset numeric findings.
"""
    write_text_atomic(decision_path, decision_text)

    status = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "status": "passed_with_declared_resource_infeasibility",
        "source_branch": _git("branch", "--show-current"),
        "source_commit": _git("rev-parse", "HEAD"),
        "plan": {
            "path": plan_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": file_sha256(plan_path),
        },
        "protocol": {
            "path": protocol_lock.relative_to(PROJECT_ROOT).as_posix(),
            "file_sha256": contract.lock_file_sha256,
            "internal_sha256": contract.lock_internal_sha256,
        },
        "builder": {
            "path": Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix(),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "matrix": {
            "manifest_sha256": matrix_manifest_sha,
            "success_sha256": matrix_success_sha,
            "row_count": matrix_manifest["summary"]["row_count"],
            "predictor_count": matrix_manifest["summary"]["predictor_count"],
            "matrix_part_count": matrix_manifest["summary"]["matrix_part_count"],
            "included_inventory_records": len(included_inventory),
            "depth_2_files_opened": matrix_manifest["summary"]["depth_2_files_opened"],
            "authenticated": True,
        },
        "pilot": {
            "phase_manifest_sha256": phase_manifest_sha,
            "success_sha256": file_sha256(pilot_root / "_SUCCESS"),
            "selector_fits": 27,
            "evaluation_cells": 30,
            "completed_evaluations": completed_evaluations,
            "unavailable_evaluations": unavailable_evaluations,
            "unavailable_cells": unavailable_cells,
            "prediction_alignments_authenticated": completed_evaluations,
            "metric_recalculations_authenticated": metric_recalculations,
            "all_registered_cells_visible": True,
            "oot_opened": False,
        },
        "resource_evidence": {
            **resources,
            "matrix_active_seconds": matrix_summary["active_computation_seconds"],
            "matrix_peak_process_tree_rss_bytes": matrix_summary[
                "peak_process_tree_rss_bytes"
            ],
            "process_tree_rss_hard_cap_bytes": 24 * GIB,
            "system_available_wait_floor_bytes": 1 * GIB,
            "forecast_allowance": ALLOWANCE,
        },
        "forecast": {
            "dev_runtime_seconds_with_20pct": dev_runtime,
            "oot_runtime_seconds_with_20pct": oot_runtime,
            "maximum_dev_fold_seconds_with_20pct": maximum_dev_fold_seconds,
            "dev_incremental_disk_bytes_with_20pct": dev_disk,
            "oot_incremental_disk_bytes_with_20pct": oot_disk,
            "results_free_bytes": results_free_bytes,
            "reforecast_disk_floor_bytes": disk_floor,
            "checks": feasibility_checks,
        },
        "validation": {
            "matrix_manifest_and_artifacts": "authenticated",
            "pilot_phase_marker": "authenticated",
            "fold_scope_and_disjointness": "authenticated",
            "selector_seals": "27/27",
            "evaluation_seals": "30/30",
            "prediction_alignment": "28/28 completed",
            "metric_recalculation": "28/28 completed",
            "natural_support_padding": "none",
            "silent_fallback_or_duplicate_active_cell": "none",
            "resource_infeasibility_rule": "two full-feature cells preserved visibly",
            "oot_access": "closed_not_opened",
        },
        "source_artifacts": {
            "matrix_supervisor_summary_sha256": file_sha256(
                log_root / "matrix_supervisor_summary.json"
            ),
            "latest_pilot_supervisor_summary_sha256": file_sha256(
                log_root / "pilot_fold_1_supervisor_summary.json"
            ),
            "pilot_scope_authentication_sha256": file_sha256(
                pilot_root / "scope_authentication.json"
            ),
            "pilot_accounting_source_sha256": file_sha256(
                pilot_root / "accounting.json"
            ),
            "cell_001_resource_status_sha256": file_sha256(
                pilot_root / "evaluations/cell_001/status.json"
            ),
            "cell_002_resource_status_sha256": file_sha256(
                pilot_root / "evaluations/cell_002/status.json"
            ),
            "pre_fix_checkpoint_evidence_sha256": file_sha256(
                output_root
                / "implementation_fix_003_one_gib_selector_memory_release.json"
            ),
        },
        "decision": {
            "pilot_gate": "pass_after_commit",
            "dev_gate": "open_after_this_gate_commit",
            "oot_gate": "closed_pending_authenticated_complete_DEV_and_DEV_freeze_commit",
            "scientific_configuration_changed_from_pilot_performance": False,
        },
        "artifacts": {
            accounting_path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(
                accounting_path
            ),
            forecast_path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(forecast_path),
            decision_path.relative_to(PROJECT_ROOT).as_posix(): file_sha256(decision_path),
        },
    }
    write_json_atomic(output_root / "pilot_gate_status.json", status)
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticate Prompt-16 pilot artifacts and build the pre-DEV gate"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "cleanup/audits/prompt_16_final_third_dataset_execution",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan_path = args.plan.resolve()
    output_root = args.output_root.resolve()
    status = build_gate(plan_path, output_root)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
