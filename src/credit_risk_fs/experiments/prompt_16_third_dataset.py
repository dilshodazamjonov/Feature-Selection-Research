"""Fail-closed execution support for the frozen Prompt-16 third benchmark.

The module deliberately exposes only the canonical matrix, pilot, DEV, and OOT
operations.  Scientific configuration is recovered from the authenticated
third-dataset lock; the execution plan supplies only paths and resource limits.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timezone
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from credit_risk_fs.data.homecredit_model_stability_2024.adapter import (
    build_modeling_matrix,
    validate_output_manifest,
)
from credit_risk_fs.data.homecredit_model_stability_2024.contract import (
    AdapterContract,
    canonical_sha256,
    file_sha256,
    load_adapter_contract,
)
from credit_risk_fs.evaluation.metrics import determine_threshold, evaluate_model
from credit_risk_fs.experiments.atomic_io import (
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic,
)
from credit_risk_fs.experiments.row_alignment import split_alignment_summary
from credit_risk_fs.models.registry import get_model_bundle
from credit_risk_fs.preprocessing.encoding import (
    OriginalFeatureNumericEncoder,
    Preprocessor,
)
from credit_risk_fs.experiments.ram_control import wait_for_ram_ready
from credit_risk_fs.selectors.base import get_selected_features
from credit_risk_fs.selectors.combinations import (
    BorutaThenCatBoostRFESelector,
    BorutaThenMutualInformationMRMRSelector,
    IVThenBorutaSelector,
    StatisticalNormalizedAverageRankSelector,
)
from credit_risk_fs.selectors.lightweight.registry import get_method_descriptor
from credit_risk_fs.selectors.lightweight.mi_mrmr import (
    MutualInformationMRMRSelector,
)
from credit_risk_fs.selectors.lightweight.mrmr_compact_cache import (
    COMPACT_MRMR_CACHE_SCHEMA_VERSION,
    COMPACT_MRMR_MEMORY_STRATEGY,
    DEFAULT_FEATURE_BATCH_SIZE,
)


SCHEMA_VERSION = "prompt_16_third_dataset_execution_v1"
PLAN_SCHEMA_VERSION = "prompt_16_resource_pilot_plan_v1"
EXPECTED_PROTOCOL_FILE_SHA256 = (
    "e4b9f9f13286f15db0887c9dead09eb7e13f7912af786f2f2bc9c53d126b1860"
)
EXPECTED_PROTOCOL_INTERNAL_SHA256 = (
    "638e1fa2aa54bf98b771206b56ac13f6a6b77e2093deb291b794081d1a475df6"
)
EXPECTED_EXECUTION_PLAN_SHA256 = (
    "9fbad83000f6822e523b76add1b78f232a3d7e882bbf1e590216585b5745a31e"
)
NON_PREDICTORS = ("case_id", "date_decision", "MONTH", "WEEK_NUM", "target")
COMBINATION_PROTOCOL_SHA256 = (
    "bce77cf33de1a6d0545c2e8b425d89eb5fab36b0c426fd4c4dc50727b50603e9"
)
SELECTOR_MEMMAP_SCHEMA_VERSION = "prompt_16_selector_memmap_cache_v1"
SELECTOR_MEMMAP_FEATURE_SPLIT_SIZE = 64
SELECTOR_MEMMAP_MEMORY_STRATEGY = (
    "feature_split_disk_backed_fortran_float32_memmap_v1"
)
MRMR_COMPACT_CACHE_ROOT_NAME = "classical_mrmr_compact_v4"


class Prompt16ExecutionError(RuntimeError):
    """Raised when a frozen execution boundary cannot be authenticated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Prompt16ExecutionError(f"expected JSON object: {path}")
    return value


def _protocol_payload(path: str | Path) -> tuple[AdapterContract, dict[str, Any]]:
    contract = load_adapter_contract(path)
    if contract.lock_file_sha256 != EXPECTED_PROTOCOL_FILE_SHA256:
        raise Prompt16ExecutionError("third-dataset protocol file digest changed")
    if contract.lock_internal_sha256 != EXPECTED_PROTOCOL_INTERNAL_SHA256:
        raise Prompt16ExecutionError("third-dataset protocol internal digest changed")
    return contract, _json(path)


def load_execution_plan(path: str | Path) -> dict[str, Any]:
    if file_sha256(path) != EXPECTED_EXECUTION_PLAN_SHA256:
        raise Prompt16ExecutionError("Prompt-16 execution-plan digest mismatch")
    plan = _json(path)
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise Prompt16ExecutionError("unsupported Prompt-16 execution-plan schema")
    if plan.get("status") != "authorized_outcome_blind":
        raise Prompt16ExecutionError("Prompt-16 execution plan is not authorized")
    protocol = plan.get("protocol", {})
    if protocol.get("file_sha256") != EXPECTED_PROTOCOL_FILE_SHA256:
        raise Prompt16ExecutionError("execution plan protocol file digest mismatch")
    if protocol.get("internal_sha256") != EXPECTED_PROTOCOL_INTERNAL_SHA256:
        raise Prompt16ExecutionError("execution plan protocol internal digest mismatch")
    paths = plan.get("paths", {})
    expected_raw_root = Path(
        "D:/python projects/Research/data/homecredit_model_stability_2024"
    ).resolve()
    raw_root = Path(paths.get("raw_dataset_root", "")).resolve()
    if raw_root != expected_raw_root:
        raise Prompt16ExecutionError("execution plan raw root changed")
    for key in (
        "matrix_root",
        "pilot_root",
        "dev_root",
        "oot_root",
        "audit_root",
        "log_root",
        "temp_root",
    ):
        output = Path(paths.get(key, "")).resolve()
        if raw_root == output or raw_root in output.parents:
            raise Prompt16ExecutionError(f"planned {key} overlaps the raw dataset root")
    return plan


def canonical_registry(protocol_lock: str | Path) -> dict[str, Any]:
    """Return the exact data-free method/evaluation registry from the lock."""

    contract, payload = _protocol_payload(protocol_lock)
    matrix = payload["approved_protocol"]["method_and_evaluation_matrix"]
    phase = matrix["phase_design"]
    registry = {
        "schema_version": SCHEMA_VERSION,
        "protocol_file_sha256": contract.lock_file_sha256,
        "protocol_internal_sha256": contract.lock_internal_sha256,
        "method_order": matrix["method_order"],
        "variant_order": matrix["variant_order"],
        "models": matrix["models"],
        "matrix_cells": matrix["matrix_cells"],
        "phase_design": phase,
        "resource_controls": matrix["resource_controls"],
        "fit_registry": selection_fit_registry(matrix),
    }
    if len(registry["matrix_cells"]) != 30 or len(registry["fit_registry"]) != 27:
        raise Prompt16ExecutionError("canonical 27/30 registry accounting changed")
    return registry


def selection_fit_registry(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Derive the frozen 27 selector invocations and their dependent cells."""

    cells = list(matrix["matrix_cells"])
    if [int(cell["configuration_order"]) for cell in cells] != list(range(1, 31)):
        raise Prompt16ExecutionError("matrix configuration order is not exactly 1..30")
    fits: list[dict[str, Any]] = []
    for cell in cells[:18]:
        order = int(cell["configuration_order"])
        fits.append(
            {
                "fit_order": len(fits) + 1,
                "fit_id": f"fit_{len(fits) + 1:03d}",
                "family": cell["family"],
                "method_id": cell["method_id"],
                "iv_pool_budget": cell["iv_pool_budget"],
                "requested_feature_budget": cell["requested_feature_budget"],
                "model_budget_owner": cell["model"],
                "dependent_configuration_orders": [order],
            }
        )
    combination_groups: tuple[tuple[int, ...], ...] = (
        (19,),
        (20,),
        (21, 22),
        (23, 24),
        (25, 26),
        (27,),
        (28,),
        (29,),
        (30,),
    )
    by_order = {int(cell["configuration_order"]): cell for cell in cells}
    for group in combination_groups:
        first = by_order[group[0]]
        for order in group[1:]:
            other = by_order[order]
            if not (
                first["method_id"] == other["method_id"] == "iv_then_boruta"
                and first["iv_pool_budget"] == other["iv_pool_budget"]
            ):
                raise Prompt16ExecutionError("only exact IV-then-Boruta pairs may reuse a fit")
        fits.append(
            {
                "fit_order": len(fits) + 1,
                "fit_id": f"fit_{len(fits) + 1:03d}",
                "family": first["family"],
                "method_id": first["method_id"],
                "iv_pool_budget": first["iv_pool_budget"],
                "requested_feature_budget": first["requested_feature_budget"],
                "model_budget_owner": (
                    None if len(group) == 2 else first["model"]
                ),
                "dependent_configuration_orders": list(group),
            }
        )
    if len(fits) != 27:
        raise Prompt16ExecutionError(f"expected 27 selector fits, found {len(fits)}")
    dependent = [order for fit in fits for order in fit["dependent_configuration_orders"]]
    if sorted(dependent) != list(range(1, 31)):
        raise Prompt16ExecutionError("selector-fit dependency graph does not cover 30 cells once")
    return fits


def _check_stop(stop_event: Any) -> None:
    if stop_event is not None and stop_event.is_set():
        raise KeyboardInterrupt("supervisor requested graceful stop")


def _publish_stage(stage_queue: Any, stage: str, fold_id: Any, **values: Any) -> None:
    if stage_queue is not None:
        stage_queue.put({"stage": stage, "fold_id": fold_id, **values})


def selector_wall_clock_stage(method_id: str) -> str:
    """Map each frozen selector to its canonical resource-control stage."""

    lightweight = {
        "full_features",
        "random_k",
        "iv_woe",
        "mrmr_mutual_information",
        "lasso_l1_logistic",
        "legacy_rf_relevance_corr",
    }
    if method_id in lightweight:
        return "baseline_lightweight"
    mapping = {
        "catboost_shap": "baseline_catboost_shap",
        "boruta_random_forest": "baseline_boruta_random_forest",
        "rfe_catboost": "baseline_rfe_catboost",
        "statistical_normalized_average_rank": "statistical_normalized_average_rank",
        "iv_then_boruta": "iv_then_boruta",
        "boruta_then_mrmr_mutual_information": "boruta_then_mrmr_mutual_information",
        "boruta_then_rfe_catboost": "boruta_then_rfe_catboost",
    }
    try:
        return mapping[str(method_id)]
    except KeyError as exc:
        raise Prompt16ExecutionError(
            f"no canonical wall-clock stage for selector {method_id}"
        ) from exc


def run_matrix_worker(
    *,
    input_root: str,
    matrix_root: str,
    protocol_lock: str,
    shard_rows: int,
    resource_event_path: str,
    stop_event: Any = None,
    stage_queue: Any = None,
    **_controls: Any,
) -> dict[str, Any]:
    """Build and re-authenticate the real matrix inside a supervised worker."""

    contract, _ = _protocol_payload(protocol_lock)
    if int(shard_rows) != 50_000:
        raise Prompt16ExecutionError("Prompt-16 matrix shard size is frozen at 50000")
    event_final = Path(resource_event_path)
    event_final.parent.mkdir(parents=True, exist_ok=True)
    event_partial = event_final.with_name(event_final.name + ".incomplete")
    if event_final.exists():
        if not (Path(matrix_root) / "_SUCCESS").is_file():
            raise Prompt16ExecutionError(
                "matrix resource log exists without a completed matrix marker"
            )
        _publish_stage(stage_queue, "matrix_authenticated_reuse", "matrix")
        reuse_started = time.perf_counter()
        reused = build_modeling_matrix(
            input_root=input_root,
            output_root=matrix_root,
            contract=contract,
            mode="research",
            shard_rows=int(shard_rows),
        )
        if not reused.reused_completed_build:
            raise Prompt16ExecutionError("completed matrix did not authenticate for reuse")
        manifest = validate_output_manifest(matrix_root)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "manifest_path": str(reused.manifest_path),
            "manifest_sha256": file_sha256(reused.manifest_path),
            "row_count": int(reused.row_count),
            "predictor_count": int(reused.predictor_count),
            "matrix_part_count": len(reused.matrix_parts),
            "inventory_identity_sha256": manifest["identity"][
                "input_inventory_identity_sha256"
            ],
            "elapsed_seconds": time.perf_counter() - reuse_started,
            "authenticated_reuse_seconds": time.perf_counter() - reuse_started,
            "resource_event_path": str(event_final),
            "resource_event_sha256": file_sha256(event_final),
            "reused_completed_matrix": True,
        }
    if event_partial.exists():
        archive = event_partial.with_name(
            event_partial.name + f".archived-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        )
        os.replace(event_partial, archive)
    started = time.perf_counter()
    last_stage: tuple[str, Any] | None = None
    with event_partial.open("x", encoding="utf-8", buffering=1) as handle:
        def resource_hook(event: Mapping[str, Any]) -> None:
            nonlocal last_stage
            _check_stop(stop_event)
            record = {"captured_at_utc": _utc_now(), **dict(event)}
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            family = record.get("family", "matrix")
            shard = record.get("shard_id")
            event_name = str(record.get("event", "matrix"))
            if event_name == "input_batch":
                if family == "train_base":
                    stage_name = "matrix_base_scan"
                elif family in {"static", "static_cb"}:
                    stage_name = "matrix_depth_0_scan"
                else:
                    stage_name = "matrix_depth_1_scan_and_aggregation"
            elif event_name in {"checkpoint_reused", "family_shard_completed"}:
                stage_name = "matrix_checkpoint_unit"
            else:
                stage_name = "matrix_publication_unit"
            stage = (stage_name, f"{family}:{shard}")
            if stage != last_stage and record.get("event") in {
                "input_batch",
                "checkpoint_reused",
                "family_shard_completed",
                "matrix_shard_completed",
            }:
                _publish_stage(
                    stage_queue,
                    stage_name,
                    stage[1],
                    family=family,
                    shard_id=shard,
                    operation=event_name,
                )
                last_stage = stage

        _publish_stage(stage_queue, "matrix_inventory_and_build", "matrix")
        result = build_modeling_matrix(
            input_root=input_root,
            output_root=matrix_root,
            contract=contract,
            mode="research",
            shard_rows=int(shard_rows),
            resource_hook=resource_hook,
        )
        _check_stop(stop_event)
        _publish_stage(stage_queue, "matrix_authentication", "matrix")
        manifest = validate_output_manifest(matrix_root)
        reuse_started = time.perf_counter()
        reused = build_modeling_matrix(
            input_root=input_root,
            output_root=matrix_root,
            contract=contract,
            mode="research",
            shard_rows=int(shard_rows),
            resource_hook=resource_hook,
        )
        reuse_seconds = time.perf_counter() - reuse_started
        if not reused.reused_completed_build:
            raise Prompt16ExecutionError("completed matrix did not authenticate for reuse")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(event_partial, event_final)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "manifest_path": str(result.manifest_path),
        "manifest_sha256": file_sha256(result.manifest_path),
        "row_count": int(result.row_count),
        "predictor_count": int(result.predictor_count),
        "matrix_part_count": len(result.matrix_parts),
        "inventory_identity_sha256": manifest["identity"][
            "input_inventory_identity_sha256"
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "authenticated_reuse_seconds": reuse_seconds,
        "resource_event_path": str(event_final),
        "resource_event_sha256": file_sha256(event_final),
    }


def _matrix_identity(matrix_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(matrix_root)
    manifest = validate_output_manifest(root)
    metadata = _json(root / "metadata.json")
    predictors = list(metadata.get("predictor_columns", []))
    if len(predictors) != 1959 or len(predictors) != len(set(predictors)):
        raise Prompt16ExecutionError("matrix predictor universe is not exact 1959 unique columns")
    if manifest["summary"].get("predictor_count") != 1959:
        raise Prompt16ExecutionError("matrix manifest predictor count changed")
    if manifest["summary"].get("row_count") != 1_526_659:
        raise Prompt16ExecutionError("matrix manifest row count changed")
    return manifest, metadata


def _part_paths(matrix_root: Path, manifest: Mapping[str, Any]) -> list[Path]:
    parts = [
        matrix_root / str(item["path"])
        for item in manifest["artifacts"]
        if str(item["path"]).startswith("matrix/")
        and str(item["path"]).endswith(".parquet")
    ]
    parts.sort(key=lambda path: path.as_posix())
    if len(parts) != int(manifest["summary"]["matrix_part_count"]):
        raise Prompt16ExecutionError("matrix part inventory does not reconcile")
    return parts


def _read_date_slice(
    matrix_root: Path,
    manifest: Mapping[str, Any],
    *,
    date_min: str,
    date_max: str,
    predictors: Sequence[str],
    stop_event: Any,
    stage_queue: Any,
    stage: str,
    fold_label: str,
    ram_ready_event: Any = None,
) -> pd.DataFrame:
    columns = [*NON_PREDICTORS, *predictors]
    selected_tables: list[pa.Table] = []
    for index, path in enumerate(_part_paths(matrix_root, manifest), start=1):
        _check_stop(stop_event)
        wait_for_ram_ready(
            ram_ready_event,
            stop_event,
            boundary=f"{stage}:{fold_label}:matrix_part_{index}",
        )
        _publish_stage(
            stage_queue,
            stage,
            fold_label,
            matrix_part=index,
            matrix_part_count=int(manifest["summary"]["matrix_part_count"]),
        )
        boundary = pq.read_table(path, columns=["date_decision"])
        boundary_type = boundary["date_decision"].type
        if pa.types.is_date(boundary_type):
            lower = pa.scalar(date.fromisoformat(date_min), type=boundary_type)
            upper = pa.scalar(date.fromisoformat(date_max), type=boundary_type)
        elif pa.types.is_timestamp(boundary_type):
            lower = pa.scalar(pd.Timestamp(date_min), type=boundary_type)
            upper = pa.scalar(pd.Timestamp(date_max), type=boundary_type)
        else:
            lower = pa.scalar(date_min, type=boundary_type)
            upper = pa.scalar(date_max, type=boundary_type)
        mask = pc.and_(
            pc.greater_equal(boundary["date_decision"], lower),
            pc.less_equal(boundary["date_decision"], upper),
        )
        indices = pc.indices_nonzero(mask)
        if len(indices) == 0:
            continue
        table = pq.read_table(path, columns=columns).take(indices)
        selected_tables.append(table)
    if not selected_tables:
        raise Prompt16ExecutionError(f"no matrix rows found for {date_min}..{date_max}")
    table = pa.concat_tables(selected_tables)
    del selected_tables
    frame = table.to_pandas(split_blocks=True, self_destruct=True)
    del table
    # PyArrow uses mimalloc in the locked Windows environment.  Returning
    # unused pages at each projection boundary prevents prior Parquet buffers
    # from remaining resident while the next authenticated projection loads.
    pa.default_memory_pool().release_unused()
    # The lock requires nullable booleans to become 0/1 numeric with missing
    # preserved before fold-local imputation.
    for name in predictors:
        if pd.api.types.is_bool_dtype(frame[name].dtype):
            frame[name] = frame[name].astype("Float32")
    order_index = pd.MultiIndex.from_frame(frame[["date_decision", "case_id"]])
    if not order_index.is_monotonic_increasing:
        frame.sort_values(
            ["date_decision", "case_id"],
            kind="mergesort",
            inplace=True,
            ignore_index=True,
        )
    return frame


def _expected_scope(
    protocol: Mapping[str, Any], phase: str, fold_id: int | None
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    split = protocol["approved_protocol"]["split_and_fold_boundaries"]
    if phase in {"pilot", "dev"}:
        if fold_id not in {1, 2, 3, 4, 5}:
            raise Prompt16ExecutionError("DEV fold id must be in 1..5")
        fold = next(item for item in split["folds"] if int(item["fold_id"]) == fold_id)
        return fold["train"], fold["validation"]
    if phase == "oot":
        return split["dev"], split["oot"]
    raise Prompt16ExecutionError(f"unknown phase: {phase}")


def _validate_scope_frame(
    frame: pd.DataFrame, expected: Mapping[str, Any], label: str
) -> dict[str, Any]:
    summary = _locked_alignment_summary(
        frame["case_id"].tolist(), frame["target"].tolist()
    )
    checks = {
        "rows": int(expected["rows"]),
        "target_0": int(expected["target_0"]),
        "target_1": int(expected["target_1"]),
        "ordered_case_id_sha256": str(expected["ordered_case_id_sha256"]),
        "ordered_case_id_target_sha256": str(expected["ordered_case_id_target_sha256"]),
    }
    observed = {
        "rows": len(frame),
        "target_0": int((frame["target"] == 0).sum()),
        "target_1": int((frame["target"] == 1).sum()),
        "ordered_case_id_sha256": summary["ordered_case_id_sha256"],
        "ordered_case_id_target_sha256": summary[
            "ordered_case_id_target_sha256"
        ],
    }
    if observed != checks:
        raise Prompt16ExecutionError(
            f"{label} row/target/alignment authentication failed: expected={checks}, observed={observed}"
        )
    if not frame["case_id"].is_unique:
        raise Prompt16ExecutionError(f"{label} case_id is not unique")
    return {"expected": checks, "observed": observed, "authenticated": True}


def _locked_alignment_summary(
    row_ids: Sequence[Any], targets: Sequence[Any]
) -> dict[str, Any]:
    """Reproduce the exact line serialization frozen in the third-data lock."""

    ids = list(row_ids)
    target_values = list(targets)
    if len(ids) != len(target_values):
        raise Prompt16ExecutionError("row ID and target counts differ")
    id_digest = hashlib.sha256()
    target_digest = hashlib.sha256()
    for row_id, target in zip(ids, target_values, strict=True):
        case_id = int(row_id)
        outcome = int(target)
        id_digest.update(f"{case_id}\n".encode("utf-8"))
        target_digest.update(f"{case_id}\x1f{outcome}\n".encode("utf-8"))
    return {
        "hash_contract": "third_dataset_protocol_lock_decimal_line_v1",
        "row_count": len(ids),
        "ordered_case_id_sha256": id_digest.hexdigest(),
        "ordered_case_id_target_sha256": target_digest.hexdigest(),
        "canonical_alignment_v1": split_alignment_summary(ids, target_values),
    }


def _load_phase_frames(
    *,
    matrix_root: str | Path,
    protocol: Mapping[str, Any],
    phase: str,
    fold_id: int | None,
    stop_event: Any,
    stage_queue: Any,
    ram_ready_event: Any = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    root = Path(matrix_root)
    manifest, metadata = _matrix_identity(root)
    predictors = list(metadata["predictor_columns"])
    train_expected, validation_expected = _expected_scope(protocol, phase, fold_id)
    label = "oot" if phase == "oot" else f"fold_{fold_id}"
    train = _read_date_slice(
        root,
        manifest,
        date_min=str(train_expected["date_min"]),
        date_max=str(train_expected["date_max"]),
        predictors=predictors,
        stop_event=stop_event,
        stage_queue=stage_queue,
        stage="full_dev_data_loading" if phase == "oot" else "dev_data_loading",
        fold_label=label + ":train",
        ram_ready_event=ram_ready_event,
    )
    validation = _read_date_slice(
        root,
        manifest,
        date_min=str(validation_expected["date_min"]),
        date_max=str(validation_expected["date_max"]),
        predictors=predictors,
        stop_event=stop_event,
        stage_queue=stage_queue,
        stage="locked_oot_data_loading" if phase == "oot" else "dev_data_loading",
        fold_label=label + ":validation",
        ram_ready_event=ram_ready_event,
    )
    authentication = {
        "matrix_manifest_sha256": file_sha256(root / "manifest.json"),
        "train": _validate_scope_frame(train, train_expected, label + ":train"),
        "validation": _validate_scope_frame(
            validation, validation_expected, label + ":validation"
        ),
        "case_id_overlap": int(
            len(set(train["case_id"].tolist()) & set(validation["case_id"].tolist()))
        ),
    }
    if authentication["case_id_overlap"] != 0:
        raise Prompt16ExecutionError("training and held-out case IDs overlap")
    return train, validation, predictors, authentication


def _load_authenticated_scope_projection(
    *,
    matrix_root: Path,
    manifest: Mapping[str, Any],
    expected: Mapping[str, Any],
    expected_observed: Mapping[str, Any] | None,
    predictors: Sequence[str],
    label: str,
    stage: str,
    stop_event: Any,
    stage_queue: Any,
    ram_ready_event: Any = None,
) -> pd.DataFrame:
    """Load one bounded projection and re-authenticate its immutable rows."""

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
    observed = _validate_scope_frame(frame, expected, label)["observed"]
    if expected_observed is not None and observed != dict(expected_observed):
        raise Prompt16ExecutionError(
            f"{label} bounded projection changed authenticated row identity"
        )
    return frame


def _selector_memmap_identity(
    *,
    execution_authorization_sha256: str,
    matrix_manifest_sha256: str,
    expected_observed: Mapping[str, Any],
    predictors: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": SELECTOR_MEMMAP_SCHEMA_VERSION,
        "execution_authorization_sha256": execution_authorization_sha256,
        "matrix_manifest_sha256": matrix_manifest_sha256,
        "full_dev_scope_sha256": canonical_sha256(dict(expected_observed)),
        "ordered_predictor_sha256": canonical_sha256(list(predictors)),
        "row_count": int(expected_observed["rows"]),
        "column_count": len(predictors),
        "dtype": "float32",
        "array_order": "F",
        "feature_split_size": SELECTOR_MEMMAP_FEATURE_SPLIT_SIZE,
        "encoder": (
            "credit_risk_fs.preprocessing.encoding.OriginalFeatureNumericEncoder"
        ),
        "memory_strategy": SELECTOR_MEMMAP_MEMORY_STRATEGY,
    }


def _mrmr_compact_execution_identity(
    *,
    checkpoint_authorization_identity_sha256: str,
    matrix_manifest_sha256: str,
    expected_observed: Mapping[str, Any],
    predictors: Sequence[str],
    selector_cache_metadata: Mapping[str, Any],
    selector_settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind execution checkpoints to the frozen data, method, and source array."""

    artifact = dict(selector_cache_metadata.get("artifact", {}))
    if not artifact.get("sha256") or int(artifact.get("byte_size", -1)) <= 0:
        raise Prompt16ExecutionError(
            "mRMR compact checkpoints require an authenticated selector source"
        )
    return {
        "schema_version": COMPACT_MRMR_CACHE_SCHEMA_VERSION,
        "checkpoint_authorization_identity_sha256": (
            checkpoint_authorization_identity_sha256
        ),
        "matrix_manifest_sha256": matrix_manifest_sha256,
        "full_dev_scope_sha256": canonical_sha256(dict(expected_observed)),
        "ordered_predictor_sha256": canonical_sha256(list(predictors)),
        "row_count": int(expected_observed["rows"]),
        "column_count": len(predictors),
        "selector_source_artifact_sha256": str(artifact["sha256"]),
        "selector_source_artifact_byte_size": int(artifact["byte_size"]),
        "selector_source_identity_sha256": canonical_sha256(
            dict(selector_cache_metadata.get("identity", {}))
        ),
        "implementation_id": (
            MutualInformationMRMRSelector.implementation_id
        ),
        "selector_settings": dict(selector_settings),
        "missing_code": -1,
        "memory_strategy": COMPACT_MRMR_MEMORY_STRATEGY,
        "scientific_semantics_changed": False,
    }


def _close_selector_memmap(mapping: np.memmap | None) -> None:
    if mapping is None:
        return
    mmap_owner = getattr(mapping, "_mmap", None)
    if mmap_owner is not None and not mmap_owner.closed:
        mmap_owner.close()


def _open_selector_memmap_cache(
    *,
    cache_root: Path,
    identity: Mapping[str, Any],
    index: pd.Index,
    predictors: Sequence[str],
    authorized_predecessor_authorization_sha256: str | None = None,
) -> tuple[pd.DataFrame, np.memmap, dict[str, Any]] | None:
    success_path = cache_root / "_SUCCESS"
    metadata_path = cache_root / "metadata.json"
    matrix_path = cache_root / "encoded_train.npy"
    if not success_path.is_file():
        return None
    success = _json(success_path)
    if success.get("metadata_sha256") != file_sha256(metadata_path):
        raise Prompt16ExecutionError("selector memmap cache completion marker changed")
    metadata = _json(metadata_path)
    observed_identity = metadata.get("identity")
    expected_identity = dict(identity)
    if observed_identity != expected_identity:
        predecessor_identity = dict(expected_identity)
        predecessor_identity["execution_authorization_sha256"] = (
            authorized_predecessor_authorization_sha256
        )
        if (
            authorized_predecessor_authorization_sha256 is None
            or observed_identity != predecessor_identity
        ):
            raise Prompt16ExecutionError("selector memmap cache identity changed")
        metadata = {
            **metadata,
            "cache_reauthenticated_for_authorization_sha256": expected_identity.get(
                "execution_authorization_sha256"
            ),
            "cache_original_authorization_sha256": (
                authorized_predecessor_authorization_sha256
            ),
        }
    artifact = metadata.get("artifact", {})
    if (
        not matrix_path.is_file()
        or matrix_path.stat().st_size != int(artifact.get("byte_size", -1))
        or file_sha256(matrix_path) != artifact.get("sha256")
    ):
        raise Prompt16ExecutionError("selector memmap cache artifact changed")
    actual_inventory = sorted(
        path.name for path in cache_root.iterdir() if path.is_file()
    )
    if actual_inventory != ["_SUCCESS", "encoded_train.npy", "metadata.json"]:
        raise Prompt16ExecutionError("selector memmap cache inventory changed")
    mapping = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if not isinstance(mapping, np.memmap):
        raise Prompt16ExecutionError("selector cache is not memory mapped")
    expected_shape = (int(identity["row_count"]), int(identity["column_count"]))
    if mapping.shape != expected_shape or mapping.dtype != np.dtype("float32"):
        _close_selector_memmap(mapping)
        raise Prompt16ExecutionError("selector memmap cache shape or dtype changed")
    numeric = pd.DataFrame(
        mapping,
        index=index.copy(deep=True),
        columns=list(predictors),
        copy=False,
    )
    if not np.shares_memory(numeric.to_numpy(copy=False), mapping):
        del numeric
        _close_selector_memmap(mapping)
        raise Prompt16ExecutionError("selector cache was copied into process memory")
    return numeric, mapping, metadata


def _build_selector_memmap_cache(
    *,
    cache_root: Path,
    identity: Mapping[str, Any],
    encoder: OriginalFeatureNumericEncoder,
    source: pd.DataFrame,
    stop_event: Any,
    ram_ready_event: Any,
) -> tuple[pd.DataFrame, np.memmap, dict[str, Any]]:
    if cache_root.exists():
        # Only an unsealed cache reaches this builder. It is an execution-only
        # temporary artifact rooted under the authorization's dedicated temp
        # directory, so rebuilding it cannot alter a promoted checkpoint.
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=False)
    partial_path = cache_root / "encoded_train.partial.npy"
    final_path = cache_root / "encoded_train.npy"

    def before_split(
        split_index: int, split_count: int, split_names: list[str]
    ) -> None:
        _check_stop(stop_event)
        wait_for_ram_ready(
            ram_ready_event,
            stop_event,
            boundary=f"selector_encoding_feature_split:{split_index}/{split_count}",
        )
        if not split_names:
            raise Prompt16ExecutionError("selector encoding produced an empty split")

    numeric, mapping = encoder.transform_releasing_source_to_memmap(
        source,
        partial_path,
        feature_split_size=SELECTOR_MEMMAP_FEATURE_SPLIT_SIZE,
        before_split=before_split,
    )
    del numeric
    _close_selector_memmap(mapping)
    del mapping
    gc.collect()
    os.replace(partial_path, final_path)
    artifact = {
        "path": final_path.name,
        "byte_size": final_path.stat().st_size,
        "sha256": file_sha256(final_path),
    }
    metadata = {
        "schema_version": SELECTOR_MEMMAP_SCHEMA_VERSION,
        "status": "complete",
        "identity": dict(identity),
        "artifact": artifact,
        "numeric_column_count": len(encoder.numeric_columns_),
        "categorical_column_count": len(encoder.categorical_columns_),
        "encoding_semantics": {
            "numeric": "coerce_nonfinite_median_fill_float32",
            "categorical": "sorted_casefold_category_map_missing_token_unknown_minus_one_float32",
            "feature_order_preserved": True,
            "row_order_preserved": True,
            "scientific_preprocessing_changed": False,
        },
        "completed_at_utc": _utc_now(),
    }
    write_json_atomic(cache_root / "metadata.json", metadata, overwrite=False)
    write_text_atomic(
        cache_root / "_SUCCESS",
        json.dumps(
            {"metadata_sha256": file_sha256(cache_root / "metadata.json")},
            sort_keys=True,
        )
        + "\n",
        overwrite=False,
    )
    opened = _open_selector_memmap_cache(
        cache_root=cache_root,
        identity=identity,
        index=source.index,
        predictors=encoder.feature_names_ or [],
    )
    if opened is None:  # pragma: no cover - marker was written immediately above
        raise Prompt16ExecutionError("completed selector memmap cache did not reopen")
    return opened


def _archive_incomplete(path: Path, archive_root: Path) -> None:
    if not path.exists() or (path / "_SUCCESS").is_file():
        return
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / (
        path.name + f"-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    )
    os.replace(path, destination)


def _artifact_digest(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "byte_size": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _seal_directory(path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = [
        _artifact_digest(item)
        for item in sorted(path.iterdir(), key=lambda item: item.name)
        if item.is_file() and item.name not in {"manifest.json", "_SUCCESS"}
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


def _load_sealed(
    path: Path,
    identity: Mapping[str, Any],
    *,
    authorized_predecessor_authorization_sha256: str | None = None,
) -> dict[str, Any] | None:
    if not (path / "_SUCCESS").is_file():
        return None
    success = _json(path / "_SUCCESS")
    manifest_path = path / "manifest.json"
    if success.get("manifest_sha256") != file_sha256(manifest_path):
        raise Prompt16ExecutionError(f"completion marker mismatch: {path}")
    manifest = _json(manifest_path)
    observed_identity = manifest.get("identity")
    expected_identity = dict(identity)
    if observed_identity != expected_identity:
        predecessor_identity = dict(expected_identity)
        predecessor_identity["execution_authorization_sha256"] = (
            authorized_predecessor_authorization_sha256
        )
        if (
            authorized_predecessor_authorization_sha256 is None
            or observed_identity != predecessor_identity
        ):
            raise Prompt16ExecutionError(f"completed artifact identity mismatch: {path}")
        manifest = {
            **manifest,
            "checkpoint_reauthenticated_for_authorization_sha256": expected_identity.get(
                "execution_authorization_sha256"
            ),
            "checkpoint_original_authorization_sha256": (
                authorized_predecessor_authorization_sha256
            ),
        }
    for item in manifest.get("artifacts", []):
        artifact = path / str(item["path"])
        if not artifact.is_file() or artifact.stat().st_size != int(item["byte_size"]):
            raise Prompt16ExecutionError(f"completed artifact size mismatch: {artifact}")
        if file_sha256(artifact) != item["sha256"]:
            raise Prompt16ExecutionError(f"completed artifact digest mismatch: {artifact}")
    return manifest


def _fit_identity(
    *,
    phase: str,
    fold_id: int | None,
    fit: Mapping[str, Any],
    matrix_manifest_sha256: str,
    execution_authorization_sha256: str | None = None,
) -> dict[str, Any]:
    identity = {
        "protocol_file_sha256": EXPECTED_PROTOCOL_FILE_SHA256,
        "protocol_internal_sha256": EXPECTED_PROTOCOL_INTERNAL_SHA256,
        "matrix_manifest_sha256": matrix_manifest_sha256,
        "phase": phase,
        "fold_id": fold_id,
        "fit_id": fit["fit_id"],
        "fit_spec_sha256": canonical_sha256(fit),
    }
    if execution_authorization_sha256 is not None:
        identity["execution_authorization_sha256"] = str(
            execution_authorization_sha256
        )
    return identity


def _evaluation_identity(
    *,
    phase: str,
    fold_id: int | None,
    cell: Mapping[str, Any],
    matrix_manifest_sha256: str,
    selection_manifest_sha256: str,
    execution_authorization_sha256: str | None = None,
) -> dict[str, Any]:
    identity = {
        "protocol_file_sha256": EXPECTED_PROTOCOL_FILE_SHA256,
        "protocol_internal_sha256": EXPECTED_PROTOCOL_INTERNAL_SHA256,
        "matrix_manifest_sha256": matrix_manifest_sha256,
        "phase": phase,
        "fold_id": fold_id,
        "configuration_order": int(cell["configuration_order"]),
        "cell_sha256": canonical_sha256(cell),
        "selection_manifest_sha256": selection_manifest_sha256,
    }
    if execution_authorization_sha256 is not None:
        identity["execution_authorization_sha256"] = str(
            execution_authorization_sha256
        )
    return identity


def record_phase_resource_infeasibility(
    *,
    matrix_root: str | Path,
    output_root: str | Path,
    protocol_lock: str | Path,
    phase: str,
    fold_id: int,
    stopped_stage: str | None,
    stopped_scope: str | int | None,
    supervisor_evidence: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Seal one pilot/DEV fit or evaluation stopped by a locked hard limit."""

    if phase not in {"pilot", "dev"} or fold_id not in {1, 2, 3, 4, 5}:
        return None
    parts = str(stopped_scope or "").split(":")
    if len(parts) != 3 or parts[0] != phase or parts[1] != str(fold_id):
        return None
    contract, protocol = _protocol_payload(protocol_lock)
    del contract
    matrix = protocol["approved_protocol"]["method_and_evaluation_matrix"]
    matrix_manifest, _ = _matrix_identity(matrix_root)
    matrix_manifest_sha = file_sha256(Path(matrix_root) / "manifest.json")
    phase_root = Path(output_root)
    phase_root.mkdir(parents=True, exist_ok=True)
    archive_root = phase_root / "archived_incomplete_attempts"
    evidence = {
        "status": "resource_infeasible",
        "reason": "locked_resource_or_wall_clock_limit",
        "phase": phase,
        "fold_id": fold_id,
        "stopped_stage": stopped_stage,
        "stopped_scope": stopped_scope,
        "matrix_manifest_sha256": matrix_manifest_sha,
        "supervisor": dict(supervisor_evidence),
        "recorded_at_utc": _utc_now(),
    }
    del matrix_manifest
    item_id = parts[2]
    if item_id.startswith("fit_"):
        fit = next(
            (item for item in selection_fit_registry(matrix) if item["fit_id"] == item_id),
            None,
        )
        if fit is None or selector_wall_clock_stage(str(fit["method_id"])) != stopped_stage:
            return None
        path = phase_root / "selection_fits" / item_id
        identity = _fit_identity(
            phase=phase,
            fold_id=fold_id,
            fit=fit,
            matrix_manifest_sha256=matrix_manifest_sha,
        )
        if _load_sealed(path, identity) is not None:
            return None
        _archive_incomplete(path, archive_root)
        path.mkdir(parents=True, exist_ok=False)
        selection = {
            **evidence,
            "fit_spec": dict(fit),
            "requested_feature_budget": fit.get("requested_feature_budget"),
            "realized_support": None,
            "selected_features": [],
            "natural_support_unpadded": True,
            "selector_result": None,
            "fit_seconds": supervisor_evidence.get("active_computation_seconds"),
            "error": {
                "class": "SupervisedResourceLimit",
                "message": str(supervisor_evidence.get("stop_code")),
            },
        }
        write_json_atomic(path / "selection.json", selection, overwrite=False)
        write_csv_atomic(
            path / "selected_features.csv",
            pd.DataFrame({"rank": [], "feature": []}),
            overwrite=False,
        )
        sealed = _seal_directory(path, identity)
        return {"kind": "selection_fit", "id": item_id, **sealed}
    if item_id.startswith("cell_"):
        try:
            order = int(item_id.removeprefix("cell_"))
        except ValueError:
            return None
        cell = next(
            (
                item
                for item in matrix["matrix_cells"]
                if int(item["configuration_order"]) == order
            ),
            None,
        )
        if cell is None or f"final_{cell['model']}" != stopped_stage:
            return None
        fit = next(
            item
            for item in selection_fit_registry(matrix)
            if order in item["dependent_configuration_orders"]
        )
        fit_path = phase_root / "selection_fits" / str(fit["fit_id"])
        fit_identity = _fit_identity(
            phase=phase,
            fold_id=fold_id,
            fit=fit,
            matrix_manifest_sha256=matrix_manifest_sha,
        )
        if _load_sealed(fit_path, fit_identity) is None:
            return None
        selection_manifest_sha = file_sha256(fit_path / "manifest.json")
        path = phase_root / "evaluations" / item_id
        identity = _evaluation_identity(
            phase=phase,
            fold_id=fold_id,
            cell=cell,
            matrix_manifest_sha256=matrix_manifest_sha,
            selection_manifest_sha256=selection_manifest_sha,
        )
        if _load_sealed(path, identity) is not None:
            return None
        _archive_incomplete(path, archive_root)
        path.mkdir(parents=True, exist_ok=False)
        status = {
            **evidence,
            "status": "unavailable",
            "reason": "resource_infeasible",
            "configuration_order": order,
            "cell": dict(cell),
            "fit_id": fit["fit_id"],
        }
        write_json_atomic(path / "status.json", status, overwrite=False)
        write_json_atomic(path / "failure.json", evidence, overwrite=False)
        sealed = _seal_directory(path, identity)
        return {"kind": "evaluation", "id": item_id, **sealed}
    return None


def _filter_kwargs(target: type, values: Mapping[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(target.__init__)
    accepts_kwargs = any(
        value.kind is inspect.Parameter.VAR_KEYWORD
        for value in signature.parameters.values()
    )
    if accepts_kwargs:
        return dict(values)
    accepted = set(signature.parameters) - {"self"}
    unknown = set(values) - accepted
    if unknown:
        raise Prompt16ExecutionError(
            f"frozen arguments not accepted by {target.__name__}: {sorted(unknown)}"
        )
    return dict(values)


def _baseline_selector(
    fit: Mapping[str, Any], matrix: Mapping[str, Any], fit_scope: str
) -> Any:
    method = str(fit["method_id"])
    descriptor = get_method_descriptor(method)
    selector_cls = descriptor.load()
    kwargs = dict(matrix["selector_settings"][method])
    budget = fit.get("requested_feature_budget")
    if budget is not None:
        kwargs["k"] = int(budget)
    if "fit_scope" in inspect.signature(selector_cls.__init__).parameters:
        kwargs["fit_scope"] = fit_scope
    return selector_cls(**_filter_kwargs(selector_cls, kwargs))


def _combination_selector(
    fit: Mapping[str, Any], matrix: Mapping[str, Any], fit_scope: str
) -> Any:
    method = str(fit["method_id"])
    settings = matrix["combination_selector_settings"]
    common = {
        "protocol_lock_sha256": COMBINATION_PROTOCOL_SHA256,
        "random_state": int(matrix["seeds"]["experiment_selector_model"]),
        "fit_scope": fit_scope,
    }
    if method == "statistical_normalized_average_rank":
        components = {
            name: dict(settings[name])
            for name in (
                "iv_woe",
                "lasso_l1_logistic",
                "rfe_catboost",
                "boruta_random_forest",
                "catboost_shap",
            )
        }
        return StatisticalNormalizedAverageRankSelector(
            k=int(fit["requested_feature_budget"]),
            component_kwargs=components,
            **common,
        )
    if method == "iv_then_boruta":
        return IVThenBorutaSelector(
            iv_pool_budget=int(fit["iv_pool_budget"]),
            iv_kwargs=dict(settings["iv_woe"]),
            boruta_kwargs=dict(settings["boruta_random_forest"]),
            **common,
        )
    if method == "boruta_then_mrmr_mutual_information":
        return BorutaThenMutualInformationMRMRSelector(
            k=int(fit["requested_feature_budget"]),
            boruta_kwargs=dict(settings["boruta_random_forest"]),
            refiner_kwargs=dict(settings["mrmr_mutual_information"]),
            **common,
        )
    if method == "boruta_then_rfe_catboost":
        return BorutaThenCatBoostRFESelector(
            k=int(fit["requested_feature_budget"]),
            boruta_kwargs=dict(settings["boruta_random_forest"]),
            refiner_kwargs=dict(settings["rfe_catboost"]),
            **common,
        )
    raise Prompt16ExecutionError(f"unregistered combination method: {method}")


def _selection_result_payload(selector: Any) -> dict[str, Any]:
    result = getattr(selector, "result", None)
    if result is not None:
        if callable(result):
            result = result()
        if hasattr(result, "to_dict"):
            return dict(result.to_dict())
        if hasattr(result, "__dataclass_fields__"):
            return asdict(result)
    result = getattr(selector, "result_", None)
    if result is not None and hasattr(result, "to_dict"):
        return dict(result.to_dict())
    return {
        "implementation": selector.__class__.__module__
        + "."
        + selector.__class__.__name__,
        "selected_features": list(get_selected_features(selector) or []),
        "result_contract_unavailable": True,
    }


def _fit_one_selection(
    *,
    fit: Mapping[str, Any],
    matrix: Mapping[str, Any],
    numeric_train: pd.DataFrame,
    y_train: pd.Series,
    fit_scope: str,
    mrmr_checkpoint_root: str | Path | None = None,
    mrmr_execution_identity: Mapping[str, Any] | None = None,
    mrmr_progress_callback: Any = None,
) -> tuple[list[str], dict[str, Any]]:
    if fit["family"] == "canonical_baseline":
        selector = _baseline_selector(fit, matrix, fit_scope)
    else:
        selector = _combination_selector(fit, matrix, fit_scope)
    if isinstance(selector, MutualInformationMRMRSelector):
        if (mrmr_checkpoint_root is None) != (mrmr_execution_identity is None):
            raise Prompt16ExecutionError(
                "mRMR compact checkpoint root and identity must be supplied together"
            )
        if mrmr_checkpoint_root is not None:
            selector.configure_execution_cache(
                mrmr_checkpoint_root,
                execution_identity=dict(mrmr_execution_identity or {}),
                feature_batch_size=DEFAULT_FEATURE_BATCH_SIZE,
                progress_callback=mrmr_progress_callback,
            )
    selector.fit(numeric_train, y_train)
    selected = list(get_selected_features(selector) or [])
    if len(selected) != len(set(selected)):
        raise Prompt16ExecutionError(f"selector returned duplicate features: {fit['fit_id']}")
    if not set(selected).issubset(set(numeric_train.columns)):
        raise Prompt16ExecutionError(f"selector escaped candidate universe: {fit['fit_id']}")
    payload = _selection_result_payload(selector)
    execution_checkpoint = getattr(
        selector, "execution_checkpoint_summary_", None
    )
    if execution_checkpoint is not None:
        payload["execution_checkpoint"] = dict(execution_checkpoint)
    del selector
    gc.collect()
    return selected, payload


def _model_settings(matrix: Mapping[str, Any], model_name: str) -> dict[str, Any]:
    values = dict(matrix["final_model_settings"][model_name])
    if model_name == "catboost":
        values["thread_count"] = min(
            4, int(matrix["resource_controls"]["estimator_threads_maximum"])
        )
    return values


def _ranking_utility(y_true: Sequence[int], score: Sequence[float]) -> dict[str, float]:
    frame = pd.DataFrame({"target": y_true, "score": score}).sort_values(
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


def _fit_and_evaluate(
    *,
    cell: Mapping[str, Any],
    selected: Sequence[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    predictors: Sequence[str],
    matrix: Mapping[str, Any],
    phase: str,
    frozen_threshold: float | None,
    full_dev_training_ks_threshold: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if not selected:
        raise Prompt16ExecutionError("zero-feature natural support cannot be modeled")
    if not set(selected).issubset(set(predictors)):
        raise Prompt16ExecutionError("evaluation selection escaped candidate universe")
    X_train = train.loc[:, list(selected)]
    X_validation = validation.loc[:, list(selected)]
    y_train = train["target"].astype("int64")
    y_validation = validation["target"].astype("int64")
    timings: dict[str, float] = {}
    started = time.perf_counter()
    preprocessor = Preprocessor(
        num_strategy="mean",
        num_scaler="standard",
        cat_max_card=7,
        cat_missing="Missing",
        cat_min_frequency=10,
    )
    step = time.perf_counter()
    encoded_train = preprocessor.fit_transform(X_train)
    encoded_validation = preprocessor.transform(X_validation)
    timings["preprocessing_seconds"] = time.perf_counter() - step
    model_name = str(cell["model"])
    model_kwargs = _model_settings(matrix, model_name)
    get_model, _, predict_proba, _ = get_model_bundle(model_name, model_kwargs)
    model = get_model()
    step = time.perf_counter()
    # Held-out validation/OOT targets are never supplied to fitting or early stopping.
    model.fit(encoded_train, y_train, eval_set=None)
    timings["training_seconds"] = time.perf_counter() - step
    step = time.perf_counter()
    validation_score = np.asarray(predict_proba(model, encoded_validation), dtype=float)
    if phase == "oot":
        if full_dev_training_ks_threshold:
            if frozen_threshold is not None:
                raise Prompt16ExecutionError(
                    "full-DEV threshold rule cannot also receive a numeric threshold"
                )
            train_score = np.asarray(predict_proba(model, encoded_train), dtype=float)
            threshold = determine_threshold(y_train.to_numpy(), train_score)
            del train_score
        else:
            if frozen_threshold is None or not np.isfinite(float(frozen_threshold)):
                raise Prompt16ExecutionError("OOT evaluation lacks a DEV-frozen threshold")
            threshold = float(frozen_threshold)
    else:
        train_score = np.asarray(predict_proba(model, encoded_train), dtype=float)
        threshold = determine_threshold(y_train.to_numpy(), train_score)
        del train_score
    timings["prediction_seconds"] = time.perf_counter() - step
    if validation_score.ndim != 1 or len(validation_score) != len(validation):
        raise Prompt16ExecutionError("prediction shape mismatch")
    if not np.isfinite(validation_score).all():
        raise Prompt16ExecutionError("non-finite prediction produced")
    classes = [int(value) for value in model.model.classes_]
    if classes != [0, 1]:
        raise Prompt16ExecutionError(f"positive class orientation changed: {classes}")
    step = time.perf_counter()
    metrics = evaluate_model(
        y_validation.to_numpy(), validation_score, threshold=threshold
    )
    metrics.update(_ranking_utility(y_validation.to_numpy(), validation_score))
    timings["evaluation_seconds"] = time.perf_counter() - step
    timings["total_seconds"] = time.perf_counter() - started
    predictions = pd.DataFrame(
        {
            "case_id": validation["case_id"].astype("int64").to_numpy(),
            "target": y_validation.to_numpy(),
            "score": validation_score,
            "decision_threshold": np.full(len(validation), threshold, dtype=float),
            "predicted_default": (validation_score >= threshold).astype("int8"),
        }
    )
    configuration = {
        "model": model_name,
        "model_settings": model_kwargs,
        "actual_model_parameters": model.model.get_params(),
        "probability_classes": classes,
        "positive_probability_column": 1,
        "probability_orientation": "class_1_higher_default_risk",
        "validation_target_used_for_fit": False,
        "threshold_source": (
            "full_dev_training_scores_maximize_ks"
            if phase == "oot" and full_dev_training_ks_threshold
            else "pre_frozen_numeric_threshold"
            if phase == "oot"
            else "fold_training_scores_maximize_ks"
        ),
        "validation_or_oot_used_to_choose_threshold": False,
        "selected_original_feature_count": len(selected),
        "encoded_feature_count": int(encoded_train.shape[1]),
        "preprocessing": {
            "implementation": "credit_risk_fs.preprocessing.encoding.Preprocessor",
            "fit_scope": "full_dev_only" if phase == "oot" else "dev_fold_training_only",
            "numeric": "mean_imputation_standard_scaler_float32",
            "categorical": "missing_token_one_hot_min_frequency_10_dense_float32",
        },
    }
    del model, encoded_train, encoded_validation, X_train, X_validation, preprocessor
    gc.collect()
    return predictions, metrics, {"timings": timings, "configuration": configuration}


def _frozen_thresholds(path: str | Path | None) -> dict[int, float]:
    if path is None:
        return {}
    plan = _json(path)
    if plan.get("schema_version") != "prompt_16_oot_analysis_plan_v1":
        raise Prompt16ExecutionError("OOT analysis plan schema mismatch")
    if plan.get("protocol_file_sha256") != EXPECTED_PROTOCOL_FILE_SHA256:
        raise Prompt16ExecutionError("OOT analysis plan protocol mismatch")
    thresholds = plan.get("frozen_decision_thresholds", {})
    return {int(key): float(value) for key, value in thresholds.items()}


def run_phase_worker(
    *,
    matrix_root: str,
    output_root: str,
    protocol_lock: str,
    phase: str,
    fold_id: int | None = None,
    oot_analysis_plan: str | None = None,
    stop_event: Any = None,
    stage_queue: Any = None,
    ram_ready_event: Any = None,
    execution_authorization_sha256: str | None = None,
    allow_authenticated_oot_recovery: bool = False,
    full_dev_training_ks_threshold: bool = False,
    memory_bounded_oot: bool = False,
    selector_memmap_root: str | None = None,
    mrmr_checkpoint_root: str | None = None,
    mrmr_checkpoint_identity_sha256: str | None = None,
    authorized_predecessor_authorization_sha256: str | None = None,
    authorized_selector_memmap_predecessor_sha256: str | None = None,
    expected_resume_fit_id: str | None = None,
    inherited_resource_infeasible_fit_ids: Sequence[str] = (),
    inherited_resource_infeasible_cell_orders: Sequence[int] = (),
    **_controls: Any,
) -> dict[str, Any]:
    """Run one exact pilot/DEV fold or the single locked OOT phase."""

    contract, protocol = _protocol_payload(protocol_lock)
    matrix = protocol["approved_protocol"]["method_and_evaluation_matrix"]
    phase = str(phase)
    if phase == "pilot" and fold_id != 1:
        raise Prompt16ExecutionError("pilot is frozen to fold 1")
    if phase == "dev" and fold_id not in {1, 2, 3, 4, 5}:
        raise Prompt16ExecutionError("DEV fold id must be in 1..5")
    if phase == "oot" and fold_id is not None:
        raise Prompt16ExecutionError("OOT has no fold id")
    if phase not in {"pilot", "dev", "oot"}:
        raise Prompt16ExecutionError("phase must be pilot, dev, or oot")
    if phase == "oot":
        if not execution_authorization_sha256:
            raise Prompt16ExecutionError(
                "OOT worker requires the final execution authorization digest"
            )
        if not full_dev_training_ks_threshold:
            raise Prompt16ExecutionError(
                "final OOT requires the frozen full-DEV training KS threshold rule"
            )
        if not memory_bounded_oot:
            raise Prompt16ExecutionError(
                "final OOT requires the authorized memory-bounded projection path"
            )
        if not selector_memmap_root:
            raise Prompt16ExecutionError(
                "final OOT requires the authorized selector memmap root"
            )
        if not mrmr_checkpoint_root:
            raise Prompt16ExecutionError(
                "final OOT requires the authorized compact mRMR checkpoint root"
            )
        if not mrmr_checkpoint_identity_sha256:
            raise Prompt16ExecutionError(
                "final OOT requires the authorized compact mRMR identity"
            )
    elif any(
        (
            execution_authorization_sha256 is not None,
            allow_authenticated_oot_recovery,
            full_dev_training_ks_threshold,
            memory_bounded_oot,
            selector_memmap_root is not None,
            mrmr_checkpoint_root is not None,
            mrmr_checkpoint_identity_sha256 is not None,
            authorized_predecessor_authorization_sha256 is not None,
            authorized_selector_memmap_predecessor_sha256 is not None,
            expected_resume_fit_id is not None,
            bool(inherited_resource_infeasible_fit_ids),
            bool(inherited_resource_infeasible_cell_orders),
        )
    ):
        raise Prompt16ExecutionError("final OOT controls may not be used for pilot/DEV")
    phase_root = Path(output_root)
    if (phase_root / "_SUCCESS").is_file():
        success = _json(phase_root / "_SUCCESS")
        manifest_path = phase_root / "phase_manifest.json"
        if success.get("phase_manifest_sha256") != file_sha256(manifest_path):
            raise Prompt16ExecutionError("completed phase marker does not authenticate")
        completed = _json(manifest_path)
        if (
            completed.get("phase") != phase
            or completed.get("fold_id") != fold_id
            or completed.get("protocol_file_sha256") != contract.lock_file_sha256
            or completed.get("protocol_internal_sha256") != contract.lock_internal_sha256
            or completed.get("execution_authorization_sha256")
            != execution_authorization_sha256
        ):
            raise Prompt16ExecutionError("completed phase identity mismatch")
        return {**completed, "reused_completed_phase": True}
    phase_root.mkdir(parents=True, exist_ok=True)
    archive_root = phase_root / "archived_incomplete_attempts"
    matrix_path = Path(matrix_root)
    selection_target: pd.Series | None = None
    selector_row_index: pd.Index | None = None
    if phase == "oot":
        matrix_manifest, matrix_metadata = _matrix_identity(matrix_path)
        predictors = list(matrix_metadata["predictor_columns"])
        train_expected, validation_expected = _expected_scope(protocol, phase, fold_id)
        train_identity_frame = _load_authenticated_scope_projection(
            matrix_root=matrix_path,
            manifest=matrix_manifest,
            expected=train_expected,
            expected_observed=None,
            predictors=[],
            label="oot:train_identity",
            stage="full_dev_identity_authentication",
            stop_event=stop_event,
            stage_queue=stage_queue,
            ram_ready_event=ram_ready_event,
        )
        validation_identity_frame = _load_authenticated_scope_projection(
            matrix_root=matrix_path,
            manifest=matrix_manifest,
            expected=validation_expected,
            expected_observed=None,
            predictors=[],
            label="oot:validation_identity",
            stage="locked_oot_identity_authentication",
            stop_event=stop_event,
            stage_queue=stage_queue,
            ram_ready_event=ram_ready_event,
        )
        train_observed = _validate_scope_frame(
            train_identity_frame, train_expected, "oot:train_identity"
        )["observed"]
        validation_observed = _validate_scope_frame(
            validation_identity_frame,
            validation_expected,
            "oot:validation_identity",
        )["observed"]
        case_id_overlap = int(
            len(
                set(train_identity_frame["case_id"].tolist())
                & set(validation_identity_frame["case_id"].tolist())
            )
        )
        if case_id_overlap != 0:
            raise Prompt16ExecutionError("training and held-out case IDs overlap")
        scope_auth = {
            "matrix_manifest_sha256": file_sha256(matrix_path / "manifest.json"),
            "train": {
                "expected": dict(train_observed),
                "observed": dict(train_observed),
                "authenticated": True,
            },
            "validation": {
                "expected": dict(validation_observed),
                "observed": dict(validation_observed),
                "authenticated": True,
            },
            "case_id_overlap": case_id_overlap,
            "memory_strategy": (
                "identity_first_bounded_projection_and_selector_memmap_v2"
            ),
        }
        selector_row_index = train_identity_frame.index.copy(deep=True)
        selection_target = pd.Series(
            train_identity_frame["target"].to_numpy(dtype=np.int64, copy=True),
            index=selector_row_index.copy(deep=True),
            name="target",
        )
        del train_identity_frame, validation_identity_frame
        pa.default_memory_pool().release_unused()
        gc.collect()
        train: pd.DataFrame | None = None
        validation: pd.DataFrame | None = None
    else:
        train, validation, predictors, scope_auth = _load_phase_frames(
            matrix_root=matrix_root,
            protocol=protocol,
            phase=phase,
            fold_id=fold_id,
            stop_event=stop_event,
            stage_queue=stage_queue,
            ram_ready_event=ram_ready_event,
        )
    matrix_manifest_sha = scope_auth["matrix_manifest_sha256"]
    write_json_atomic(phase_root / "scope_authentication.json", scope_auth)
    fits = selection_fit_registry(matrix)
    cells = list(matrix["matrix_cells"])
    fit_by_cell = {
        int(order): fit
        for fit in fits
        for order in fit["dependent_configuration_orders"]
    }
    selection_root = phase_root / "selection_fits"
    evaluation_root = phase_root / "evaluations"
    selection_root.mkdir(exist_ok=True)
    evaluation_root.mkdir(exist_ok=True)
    completed_fit_count = 0
    completed_eval_count = 0
    unavailable_eval_count = 0
    fit_scope = "full_dev_only" if phase == "oot" else "dev_fold_training_only"
    inherited_fit_ids = {str(value) for value in inherited_resource_infeasible_fit_ids}
    registered_fit_ids = {str(fit["fit_id"]) for fit in fits}
    if not inherited_fit_ids.issubset(registered_fit_ids):
        raise Prompt16ExecutionError("inherited resource-infeasible fit registry changed")
    inherited_cell_orders = {
        int(value) for value in inherited_resource_infeasible_cell_orders
    }
    if not inherited_cell_orders.issubset(
        {int(cell["configuration_order"]) for cell in cells}
    ):
        raise Prompt16ExecutionError("inherited resource-infeasible cell registry changed")

    incomplete_fits: list[Mapping[str, Any]] = []
    for fit in fits:
        path = selection_root / str(fit["fit_id"])
        identity = _fit_identity(
            phase=phase,
            fold_id=fold_id,
            fit=fit,
            matrix_manifest_sha256=matrix_manifest_sha,
            execution_authorization_sha256=execution_authorization_sha256,
        )
        sealed = _load_sealed(
            path,
            identity,
            authorized_predecessor_authorization_sha256=(
                authorized_predecessor_authorization_sha256
            ),
        )
        if sealed is None:
            _archive_incomplete(path, archive_root)
            if phase == "oot" and str(fit["fit_id"]) in inherited_fit_ids:
                path.mkdir(parents=True, exist_ok=False)
                evidence = {
                    "status": "resource_infeasible",
                    "fit_spec": dict(fit),
                    "requested_feature_budget": fit.get("requested_feature_budget"),
                    "realized_support": None,
                    "selected_features": [],
                    "natural_support_unpadded": True,
                    "selector_result": None,
                    "fit_seconds": 0.0,
                    "error": None,
                    "reason": "inherited_from_authenticated_largest_dev_fold_resource_stop",
                    "oot_target_or_performance_used": False,
                    "full_dev_fit_started": False,
                }
                write_json_atomic(path / "selection.json", evidence, overwrite=False)
                write_csv_atomic(
                    path / "selected_features.csv",
                    pd.DataFrame(columns=["rank", "feature"]),
                    overwrite=False,
                )
                _seal_directory(path, identity)
                completed_fit_count += 1
            else:
                incomplete_fits.append(fit)
        else:
            completed_fit_count += 1

    if expected_resume_fit_id is not None:
        observed_resume_fit_id = (
            str(incomplete_fits[0]["fit_id"]) if incomplete_fits else None
        )
        if observed_resume_fit_id != expected_resume_fit_id:
            raise Prompt16ExecutionError(
                "authenticated selector resume point changed: "
                f"expected={expected_resume_fit_id}, observed={observed_resume_fit_id}"
            )

    numeric_train: pd.DataFrame | None = None
    numeric_train_mapping: np.memmap | None = None
    mrmr_execution_identity: dict[str, Any] | None = None
    encoding_path = phase_root / "selector_encoding.json"
    encoding_record: dict[str, Any] | None = (
        _json(encoding_path) if encoding_path.is_file() else None
    )
    if incomplete_fits:
        _check_stop(stop_event)
        cache_metadata: dict[str, Any] | None = None
        if phase == "oot":
            if selection_target is None or selector_row_index is None:
                raise Prompt16ExecutionError("full-DEV selector identity target is absent")
            cache_identity = _selector_memmap_identity(
                execution_authorization_sha256=str(execution_authorization_sha256),
                matrix_manifest_sha256=matrix_manifest_sha,
                expected_observed=scope_auth["train"]["observed"],
                predictors=predictors,
            )
            cache_root = Path(str(selector_memmap_root))
            cached = _open_selector_memmap_cache(
                cache_root=cache_root,
                identity=cache_identity,
                index=selector_row_index,
                predictors=predictors,
                authorized_predecessor_authorization_sha256=(
                    authorized_selector_memmap_predecessor_sha256
                ),
            )
            if cached is not None:
                numeric_train, numeric_train_mapping, cache_metadata = cached
                train = None
            else:
                train = _load_authenticated_scope_projection(
                    matrix_root=matrix_path,
                    manifest=matrix_manifest,
                    expected=train_expected,
                    expected_observed=scope_auth["train"]["observed"],
                    predictors=predictors,
                    label="oot:full_dev_selector_projection",
                    stage="full_dev_selector_data_loading",
                    stop_event=stop_event,
                    stage_queue=stage_queue,
                    ram_ready_event=ram_ready_event,
                )
        _publish_stage(stage_queue, "selection_encoding", fold_id or "oot")
        encoding_started = time.perf_counter()
        if selection_target is None:
            if train is None:
                raise Prompt16ExecutionError("selector target source is absent")
            selection_target = pd.Series(
                train["target"].to_numpy(dtype=np.int64, copy=True),
                index=train.index.copy(deep=True),
                name="target",
            )
        # Selection uses training rows only. The validation frame is reloaded
        # after all selector fits, so retaining it while materializing the
        # Fold-5 float32 selector matrix only adds an avoidable multi-GiB peak.
        if validation is not None:
            del validation
            validation = None
        encoder: OriginalFeatureNumericEncoder | None = None
        if numeric_train is None:
            if train is None:
                raise Prompt16ExecutionError("selector source projection is absent")
            if phase == "oot" and not np.array_equal(
                train["target"].to_numpy(dtype=np.int64, copy=False),
                selection_target.to_numpy(dtype=np.int64, copy=False),
            ):
                raise Prompt16ExecutionError(
                    "wide selector projection target changed authenticated row order"
                )
            for name in NON_PREDICTORS:
                if name in train:
                    del train[name]
            if list(train.columns) != predictors:
                raise Prompt16ExecutionError(
                    "selector source projection changed candidate order"
                )
            gc.collect()
            encoder = OriginalFeatureNumericEncoder()
            encoder.fit(train)
            if phase == "oot":
                numeric_train, numeric_train_mapping, cache_metadata = (
                    _build_selector_memmap_cache(
                        cache_root=cache_root,
                        identity=cache_identity,
                        encoder=encoder,
                        source=train,
                        stop_event=stop_event,
                        ram_ready_event=ram_ready_event,
                    )
                )
            else:
                numeric_train = encoder.transform_releasing_source(train)
            if train.shape[1] != 0:
                raise Prompt16ExecutionError(
                    "memory-bounded selector encoding did not release source columns"
                )
        if list(numeric_train.columns) != predictors:
            raise Prompt16ExecutionError("selector encoding changed candidate order")
        if phase == "oot":
            if cache_metadata is None:
                raise Prompt16ExecutionError(
                    "compact mRMR identity lacks selector cache authentication"
                )
            mrmr_execution_identity = _mrmr_compact_execution_identity(
                checkpoint_authorization_identity_sha256=str(
                    mrmr_checkpoint_identity_sha256
                ),
                matrix_manifest_sha256=matrix_manifest_sha,
                expected_observed=scope_auth["train"]["observed"],
                predictors=predictors,
                selector_cache_metadata=cache_metadata,
                selector_settings=matrix["selector_settings"][
                    "mrmr_mutual_information"
                ],
            )
        encoding_record = {
            "implementation": "credit_risk_fs.preprocessing.encoding.OriginalFeatureNumericEncoder",
            "memory_strategy": (
                SELECTOR_MEMMAP_MEMORY_STRATEGY
                if phase == "oot"
                else "destructive_source_column_release_v1"
            ),
            "fit_scope": fit_scope,
            "training_rows": len(selection_target),
            "candidate_count": len(predictors),
            "elapsed_seconds": time.perf_counter() - encoding_started,
            "numeric_column_count": (
                len(encoder.numeric_columns_)
                if encoder is not None
                else int(cache_metadata["numeric_column_count"])
            ),
            "categorical_column_count": (
                len(encoder.categorical_columns_)
                if encoder is not None
                else int(cache_metadata["categorical_column_count"])
            ),
            "training_only": True,
            "shared_across_registered_selector_invocations": True,
            "sharing_effect": "deterministic identical fold-local encoding; selector fits remain distinct",
            "source_phase_frames_released_before_selector_fits": True,
            "evaluation_phase_frames_reloaded_after_selector_fits": True,
            "disk_backed": phase == "oot",
            "feature_split_size": (
                SELECTOR_MEMMAP_FEATURE_SPLIT_SIZE if phase == "oot" else None
            ),
            "cache_reused": phase == "oot" and encoder is None,
            "encoding_semantics_changed": False,
        }
        write_json_atomic(encoding_path, encoding_record)
        # OOT now retains only a read-only file-backed float32 mapping and the
        # independent int64 target. DEV keeps the legacy C-contiguous owner.
        # Neither path retains the wider raw source or evaluation frame during
        # opaque selector fits.
        if encoder is not None:
            del encoder
        if train is not None:
            del train
        train = None
        pa.default_memory_pool().release_unused()
        gc.collect()

    for fit in fits:
        _check_stop(stop_event)
        path = selection_root / str(fit["fit_id"])
        identity = _fit_identity(
            phase=phase,
            fold_id=fold_id,
            fit=fit,
            matrix_manifest_sha256=matrix_manifest_sha,
            execution_authorization_sha256=execution_authorization_sha256,
        )
        if _load_sealed(
            path,
            identity,
            authorized_predecessor_authorization_sha256=(
                authorized_predecessor_authorization_sha256
            ),
        ) is not None:
            continue
        if numeric_train is None or selection_target is None:
            raise Prompt16ExecutionError("incomplete selector fit lacks training inputs")
        path.mkdir(parents=True, exist_ok=False)
        _publish_stage(
            stage_queue,
            selector_wall_clock_stage(str(fit["method_id"])),
            f"{phase}:{fold_id or 'oot'}:{fit['fit_id']}",
            component=f"{fit['method_id']}_selection",
            method_id=fit["method_id"],
            fit_order=fit["fit_order"],
            operation="selector_fit",
        )
        started = time.perf_counter()
        try:
            selected, result = _fit_one_selection(
                fit=fit,
                matrix=matrix,
                numeric_train=numeric_train,
                y_train=selection_target,
                fit_scope=fit_scope,
                mrmr_checkpoint_root=(
                    mrmr_checkpoint_root if phase == "oot" else None
                ),
                mrmr_execution_identity=mrmr_execution_identity,
            )
            status = "complete" if selected else "infeasible_natural_support"
            evidence = {
                "status": status,
                "fit_spec": dict(fit),
                "requested_feature_budget": fit.get("requested_feature_budget"),
                "realized_support": len(selected),
                "selected_features": selected,
                "natural_support_unpadded": True,
                "selector_result": result,
                "fit_seconds": time.perf_counter() - started,
                "error": None,
            }
        except Exception as exc:
            evidence = {
                "status": "failed",
                "fit_spec": dict(fit),
                "requested_feature_budget": fit.get("requested_feature_budget"),
                "realized_support": None,
                "selected_features": [],
                "natural_support_unpadded": True,
                "selector_result": None,
                "fit_seconds": time.perf_counter() - started,
                "error": {"class": type(exc).__name__, "message": str(exc)},
            }
        write_json_atomic(path / "selection.json", evidence, overwrite=False)
        write_csv_atomic(
            path / "selected_features.csv",
            pd.DataFrame(
                {
                    "rank": range(1, len(evidence["selected_features"]) + 1),
                    "feature": evidence["selected_features"],
                }
            ),
            overwrite=False,
        )
        _seal_directory(path, identity)
        completed_fit_count += 1

    if numeric_train is not None:
        del numeric_train
    _close_selector_memmap(numeric_train_mapping)
    numeric_train_mapping = None
    if selection_target is not None:
        del selection_target
    gc.collect()

    if incomplete_fits and phase != "oot":
        train, validation, reloaded_predictors, reloaded_scope_auth = _load_phase_frames(
            matrix_root=matrix_root,
            protocol=protocol,
            phase=phase,
            fold_id=fold_id,
            stop_event=stop_event,
            stage_queue=stage_queue,
            ram_ready_event=ram_ready_event,
        )
        if reloaded_predictors != predictors:
            raise Prompt16ExecutionError("evaluation reload changed candidate order")
        if reloaded_scope_auth != scope_auth:
            raise Prompt16ExecutionError("evaluation reload changed authenticated fold scope")
        del reloaded_predictors, reloaded_scope_auth

    feature_psi_manifest_sha: str | None = None
    if phase == "oot":
        from credit_risk_fs.analysis.voting_inference.psi import feature_psi_record

        feature_psi_path = phase_root / "feature_psi"
        feature_psi_identity = {
            "operation": "full_dev_to_oot_source_feature_psi",
            "execution_authorization_sha256": execution_authorization_sha256,
            "matrix_manifest_sha256": matrix_manifest_sha,
            "ordered_predictor_sha256": canonical_sha256(predictors),
            "predictor_count": len(predictors),
            "dev_scope_sha256": canonical_sha256(scope_auth["train"]["observed"]),
            "oot_scope_sha256": canonical_sha256(scope_auth["validation"]["observed"]),
        }
        if _load_sealed(feature_psi_path, feature_psi_identity) is None:
            identity_path = feature_psi_path / "checkpoint_identity.json"
            if feature_psi_path.exists():
                if not identity_path.is_file() or _json(identity_path) != feature_psi_identity:
                    _archive_incomplete(feature_psi_path, archive_root)
            if not feature_psi_path.exists():
                feature_psi_path.mkdir(parents=True, exist_ok=False)
                write_json_atomic(
                    identity_path, feature_psi_identity, overwrite=False
                )
            _publish_stage(
                stage_queue,
                "oot_feature_psi",
                "oot:oot:feature_psi",
                component="full_dev_to_oot_selected_source_feature_psi",
                operation="descriptive_drift",
                ram_recovery_barrier=True,
            )
            batches_root = feature_psi_path / "batches"
            batches_root.mkdir(exist_ok=True)
            batch_size = 128
            batch_frames: list[pd.DataFrame] = []
            for batch_index, start in enumerate(
                range(0, len(predictors), batch_size), start=1
            ):
                _check_stop(stop_event)
                batch_predictors = predictors[start : start + batch_size]
                batch_path = batches_root / f"batch_{batch_index:03d}"
                batch_identity = {
                    **feature_psi_identity,
                    "batch_index": batch_index,
                    "batch_size": len(batch_predictors),
                    "ordered_batch_predictor_sha256": canonical_sha256(
                        batch_predictors
                    ),
                }
                if _load_sealed(batch_path, batch_identity) is None:
                    _archive_incomplete(batch_path, archive_root)
                    batch_path.mkdir(parents=True, exist_ok=False)
                    wait_for_ram_ready(
                        ram_ready_event,
                        stop_event,
                        boundary=f"oot_feature_psi_batch:{batch_index}",
                    )
                    train_batch = _load_authenticated_scope_projection(
                        matrix_root=matrix_path,
                        manifest=matrix_manifest,
                        expected=train_expected,
                        expected_observed=scope_auth["train"]["observed"],
                        predictors=batch_predictors,
                        label=f"oot:feature_psi_train_batch_{batch_index:03d}",
                        stage="full_dev_feature_psi_batch_loading",
                        stop_event=stop_event,
                        stage_queue=stage_queue,
                        ram_ready_event=ram_ready_event,
                    )
                    validation_batch = _load_authenticated_scope_projection(
                        matrix_root=matrix_path,
                        manifest=matrix_manifest,
                        expected=validation_expected,
                        expected_observed=scope_auth["validation"]["observed"],
                        predictors=batch_predictors,
                        label=f"oot:feature_psi_oot_batch_{batch_index:03d}",
                        stage="locked_oot_feature_psi_batch_loading",
                        stop_event=stop_event,
                        stage_queue=stage_queue,
                        ram_ready_event=ram_ready_event,
                    )
                    batch_rows: list[dict[str, Any]] = []
                    for feature in batch_predictors:
                        record, _ = feature_psi_record(
                            feature=feature,
                            dev_values=train_batch[feature],
                            oot_values=validation_batch[feature],
                        )
                        batch_rows.append(record)
                    write_parquet_atomic(
                        batch_path / "feature_psi.parquet",
                        pd.DataFrame(batch_rows),
                        required_columns=("feature", "psi_type_aware"),
                        ordered_row_identity_column="feature",
                        overwrite=False,
                    )
                    _seal_directory(batch_path, batch_identity)
                    del train_batch, validation_batch, batch_rows
                    pa.default_memory_pool().release_unused()
                    gc.collect()
                batch_frames.append(pd.read_parquet(batch_path / "feature_psi.parquet"))
            feature_psi_frame = pd.concat(batch_frames, ignore_index=True)
            if feature_psi_frame["feature"].tolist() != predictors:
                raise Prompt16ExecutionError("batched feature PSI order changed")
            final_psi_path = feature_psi_path / "all_source_features.parquet"
            if final_psi_path.is_file():
                existing_feature_psi = pd.read_parquet(final_psi_path)
                try:
                    pd.testing.assert_frame_equal(
                        existing_feature_psi,
                        feature_psi_frame,
                        check_exact=True,
                    )
                except AssertionError as exc:
                    raise Prompt16ExecutionError(
                        "partial final feature PSI checkpoint changed"
                    ) from exc
                del existing_feature_psi
            else:
                write_parquet_atomic(
                    final_psi_path,
                    feature_psi_frame,
                    required_columns=("feature", "psi_type_aware"),
                    ordered_row_identity_column="feature",
                    overwrite=False,
                )
            _seal_directory(feature_psi_path, feature_psi_identity)
            del batch_frames, feature_psi_frame
            gc.collect()
        feature_psi_manifest_sha = file_sha256(feature_psi_path / "manifest.json")

    thresholds = _frozen_thresholds(oot_analysis_plan)
    for cell in cells:
        _check_stop(stop_event)
        order = int(cell["configuration_order"])
        cell_id = f"cell_{order:03d}"
        fit = fit_by_cell[order]
        fit_path = selection_root / str(fit["fit_id"])
        selection_manifest = _json(fit_path / "manifest.json")
        selection_manifest_sha = file_sha256(fit_path / "manifest.json")
        identity = _evaluation_identity(
            phase=phase,
            fold_id=fold_id,
            cell=cell,
            matrix_manifest_sha256=matrix_manifest_sha,
            selection_manifest_sha256=selection_manifest_sha,
            execution_authorization_sha256=execution_authorization_sha256,
        )
        path = evaluation_root / cell_id
        if _load_sealed(path, identity) is not None:
            record = _json(path / "status.json")
            completed_eval_count += int(record["status"] == "complete")
            unavailable_eval_count += int(record["status"] != "complete")
            continue
        if phase == "oot" and path.exists() and not allow_authenticated_oot_recovery:
            raise Prompt16ExecutionError(
                "incomplete OOT evaluation exists; fail closed because a target-linked "
                "prediction or metric may already have become inspectable"
            )
        _archive_incomplete(path, archive_root)
        path.mkdir(parents=True, exist_ok=False)
        selection = _json(fit_path / "selection.json")
        selected = list(selection.get("selected_features", []))
        requested = cell.get("requested_feature_budget")
        natural_support = (
            requested is None or len(selected) == int(requested)
        )
        _publish_stage(
            stage_queue,
            f"final_{cell['model']}",
            f"{phase}:{fold_id or 'oot'}:{cell_id}",
            component=f"{cell['method_id']}_{cell['model']}_model_fit_and_evaluation",
            method_id=cell["method_id"],
            model=cell["model"],
            configuration_order=order,
            operation="oot_evaluation" if phase == "oot" else "fold_evaluation",
        )
        started = time.perf_counter()
        if phase == "oot" and order in inherited_cell_orders:
            status = {
                "status": "unavailable",
                "reason": "inherited_from_all_five_authenticated_dev_resource_stops",
                "configuration_order": order,
                "cell": dict(cell),
                "fit_id": fit["fit_id"],
                "requested_feature_budget": requested,
                "realized_support": len(selected),
                "natural_support_like_for_like": natural_support,
                "elapsed_seconds": time.perf_counter() - started,
                "feature_psi_manifest_sha256": feature_psi_manifest_sha,
                "oot_target_or_performance_used": False,
                "full_dev_model_fit_started": False,
            }
            write_json_atomic(path / "status.json", status, overwrite=False)
            write_json_atomic(
                path / "failure.json",
                {
                    "reason": status["reason"],
                    "scientific_configuration_changed": False,
                    "visible_unavailable_not_zero_effect": True,
                },
                overwrite=False,
            )
            _seal_directory(path, identity)
            unavailable_eval_count += 1
            continue
        if selection.get("status") in {"failed", "resource_infeasible"} or not selected:
            status = {
                "status": "unavailable",
                "reason": (
                    "selector_resource_infeasible"
                    if selection.get("status") == "resource_infeasible"
                    else "selector_failed"
                    if selection.get("status") == "failed"
                    else "infeasible_zero_natural_support"
                ),
                "configuration_order": order,
                "cell": dict(cell),
                "fit_id": fit["fit_id"],
                "requested_feature_budget": requested,
                "realized_support": len(selected),
                "natural_support_like_for_like": natural_support,
                "elapsed_seconds": time.perf_counter() - started,
                "feature_psi_manifest_sha256": feature_psi_manifest_sha,
            }
            write_json_atomic(path / "status.json", status, overwrite=False)
            write_json_atomic(path / "failure.json", selection, overwrite=False)
            _seal_directory(path, identity)
            unavailable_eval_count += 1
            continue
        try:
            if phase == "oot":
                train = _load_authenticated_scope_projection(
                    matrix_root=matrix_path,
                    manifest=matrix_manifest,
                    expected=train_expected,
                    expected_observed=scope_auth["train"]["observed"],
                    predictors=selected,
                    label=f"oot:{cell_id}:full_dev_selected_projection",
                    stage="full_dev_selected_projection_loading",
                    stop_event=stop_event,
                    stage_queue=stage_queue,
                    ram_ready_event=ram_ready_event,
                )
                validation = _load_authenticated_scope_projection(
                    matrix_root=matrix_path,
                    manifest=matrix_manifest,
                    expected=validation_expected,
                    expected_observed=scope_auth["validation"]["observed"],
                    predictors=selected,
                    label=f"oot:{cell_id}:oot_selected_projection",
                    stage="locked_oot_selected_projection_loading",
                    stop_event=stop_event,
                    stage_queue=stage_queue,
                    ram_ready_event=ram_ready_event,
                )
            predictions, metrics, details = _fit_and_evaluate(
                cell=cell,
                selected=selected,
                train=train,
                validation=validation,
                predictors=predictors,
                matrix=matrix,
                phase=phase,
                frozen_threshold=thresholds.get(order),
                full_dev_training_ks_threshold=full_dev_training_ks_threshold,
            )
            prediction_auth = _locked_alignment_summary(
                predictions["case_id"].tolist(), predictions["target"].tolist()
            )
            expected_validation = scope_auth["validation"]["observed"]
            if (
                prediction_auth["ordered_case_id_sha256"]
                != expected_validation["ordered_case_id_sha256"]
                or prediction_auth["ordered_case_id_target_sha256"]
                != expected_validation["ordered_case_id_target_sha256"]
            ):
                raise Prompt16ExecutionError("prediction row alignment changed")
            write_parquet_atomic(
                path / "predictions.parquet",
                predictions,
                required_columns=("case_id", "target", "score", "decision_threshold"),
                ordered_row_identity_column="case_id",
                overwrite=False,
            )
            write_json_atomic(path / "metrics.json", metrics, overwrite=False)
            write_json_atomic(path / "execution.json", details, overwrite=False)
            status = {
                "status": "complete",
                "reason": None,
                "configuration_order": order,
                "cell": dict(cell),
                "fit_id": fit["fit_id"],
                "requested_feature_budget": requested,
                "realized_support": len(selected),
                "natural_support_like_for_like": natural_support,
                "prediction_alignment": prediction_auth,
                "elapsed_seconds": time.perf_counter() - started,
                "feature_psi_manifest_sha256": feature_psi_manifest_sha,
            }
            write_json_atomic(path / "status.json", status, overwrite=False)
            completed_eval_count += 1
        except Exception as exc:
            status = {
                "status": "failed",
                "reason": "evaluation_exception",
                "configuration_order": order,
                "cell": dict(cell),
                "fit_id": fit["fit_id"],
                "requested_feature_budget": requested,
                "realized_support": len(selected),
                "natural_support_like_for_like": natural_support,
                "elapsed_seconds": time.perf_counter() - started,
                "error": {"class": type(exc).__name__, "message": str(exc)},
            }
            write_json_atomic(path / "status.json", status, overwrite=False)
            write_json_atomic(path / "failure.json", status["error"], overwrite=False)
            unavailable_eval_count += 1
        _seal_directory(path, identity)
        if phase == "oot":
            del train, validation
            train = None
            validation = None
            pa.default_memory_pool().release_unused()
            gc.collect()

    accounting = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "fold_id": fold_id,
        "expected_selector_fits": 27,
        "completed_selector_fit_records": completed_fit_count,
        "expected_evaluations": 30,
        "completed_evaluations": completed_eval_count,
        "unavailable_or_failed_evaluations": unavailable_eval_count,
        "accounted_evaluations": completed_eval_count + unavailable_eval_count,
        "all_registered_cells_visible": completed_eval_count + unavailable_eval_count == 30,
        "depth_2_files_opened": 0,
        "oot_opened": phase == "oot",
    }
    write_json_atomic(phase_root / "accounting.json", accounting)
    if completed_fit_count != 27 or completed_eval_count + unavailable_eval_count != 30:
        raise Prompt16ExecutionError("phase accounting does not reconcile 27/30")
    phase_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "phase": phase,
        "fold_id": fold_id,
        "protocol_file_sha256": contract.lock_file_sha256,
        "protocol_internal_sha256": contract.lock_internal_sha256,
        "execution_authorization_sha256": execution_authorization_sha256,
        "matrix_manifest_sha256": matrix_manifest_sha,
        "selection_fit_count": completed_fit_count,
        "evaluation_accounting": accounting,
        "selector_encoding": encoding_record,
        "completed_at_utc": _utc_now(),
    }
    write_json_atomic(phase_root / "phase_manifest.json", phase_manifest)
    write_text_atomic(
        phase_root / "_SUCCESS",
        json.dumps(
            {"phase_manifest_sha256": file_sha256(phase_root / "phase_manifest.json")},
            sort_keys=True,
        )
        + "\n",
    )
    if train is not None:
        del train
    if validation is not None:
        del validation
    pa.default_memory_pool().release_unused()
    gc.collect()
    return phase_manifest


def execution_lock_path(output_root: str | Path) -> Path:
    root = Path(output_root).resolve()
    return root.parent / f".{root.name}.execution.lock"


def acquire_execution_lock(output_root: str | Path) -> Path:
    path = execution_lock_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"pid": os.getpid(), "created_at_utc": _utc_now()}, sort_keys=True
                )
                + "\n"
            )
    except FileExistsError as exc:
        raise Prompt16ExecutionError(f"execution lock already exists: {path}") from exc
    return path


def release_execution_lock(path: str | Path) -> None:
    Path(path).unlink(missing_ok=False)


def archive_partial_supervision_output(path: str | Path) -> Path | None:
    target = Path(path)
    if not target.exists():
        return None
    archive = target.with_name(
        target.name + f".archived-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    os.replace(target, archive)
    return archive


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def free_disk_bytes(path: str | Path) -> int:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return int(shutil.disk_usage(target).free)


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "Prompt16ExecutionError",
    "acquire_execution_lock",
    "archive_partial_supervision_output",
    "canonical_registry",
    "directory_size_bytes",
    "execution_lock_path",
    "free_disk_bytes",
    "load_execution_plan",
    "record_phase_resource_infeasibility",
    "release_execution_lock",
    "run_matrix_worker",
    "run_phase_worker",
    "selection_fit_registry",
]
