"""Resource-safe, resumable DEV-fold-1 pilots for the heavy selectors.

This module is intentionally separate from the frozen voting workflow. It reuses
the canonical authenticated DEV loader, fold projection, preprocessing, selector
registry, resource supervisor, logging, and atomic publication primitives, but it
does not contain an OOT loader or an evaluation path.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import logging
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from credit_risk_fs.experiments.atomic_io import sha256_file, write_json_atomic
from credit_risk_fs.experiments.resource_monitor import (
    MANUAL_INTERRUPT,
    PREFLIGHT_REJECTED,
    RAM_PROCESS_LIMIT,
    RAM_SYSTEM_HEADROOM,
    SupervisorResult,
    supervise_worker,
    wait_for_inter_run_readiness,
)
from credit_risk_fs.experiments.resource_policy import (
    DEFAULT_POLICY_PATH,
    ResolvedExecutionPolicy,
    detect_hardware,
    load_execution_policy,
    resolve_execution_policy,
    run_preflight,
)
from credit_risk_fs.experiments.research_logging import (
    DEFAULT_RESEARCH_LOG,
    ResearchLogSession,
    bind_research_context,
    emit_research_event,
    suppress_third_party_output,
)
from credit_risk_fs.selectors.lightweight.registry import get_method_descriptor


logger = logging.getLogger(__name__)

CONFIG_SCHEMA_VERSION = "heavy_selector_dev_pilot_config_v1"
ARTIFACT_SCHEMA_VERSION = "heavy_selector_dev_pilot_artifact_v1"
STATUS_SCHEMA_VERSION = "heavy_selector_dev_pilot_status_v1"
DEFAULT_CONFIG_PATH = Path("configs/experiments/heavy_selector_dev_pilots_v1.yaml")
DATASET_ORDER = ("homecredit", "lendingclub_v2")
METHOD_ORDER = ("catboost_shap", "boruta_random_forest", "rfe_catboost")
TERMINAL_STATES = {
    "completed",
    "failed",
    "manually_interrupted",
    "timed_out",
    "resource_aborted",
}
CONTROLLED_STOP_STATES = {
    "manually_interrupted",
    "timed_out",
    "resource_aborted",
}


class PilotConfigurationError(ValueError):
    """Raised before data access when the fixed pilot matrix is not exact."""


class PilotArtifactError(ValueError):
    """Raised when a cell artifact is not authenticated and complete."""


@dataclass(frozen=True, slots=True)
class PilotCell:
    cell_index: int
    cell_id: str
    dataset: str
    dataset_label: str
    fold_id: int
    method_id: str
    implementation_id: str
    expected_candidate_count: int
    stable_row_id_column: str
    experiment_config_path: str
    selector_kwargs: dict[str, Any]
    wall_clock_limit_seconds: float

    def to_worker_spec(
        self,
        *,
        plan: "PilotPlan",
        estimator_threads: int,
    ) -> dict[str, Any]:
        kwargs = copy.deepcopy(self.selector_kwargs)
        if self.method_id == "boruta_random_forest":
            kwargs["n_jobs"] = int(estimator_threads)
        else:
            kwargs["thread_count"] = int(estimator_threads)
        return {
            **asdict(self),
            "selector_kwargs": kwargs,
            "phase": "DEV",
            "oot_access": "forbidden",
            "performance_evaluation": False,
            "full_dev_refit": False,
            "configuration_sha256": plan.configuration_sha256,
            "csv_chunk_rows": plan.csv_chunk_rows,
            "lendingclub_csv_low_memory": plan.lendingclub_csv_low_memory,
            "estimator_threads": int(estimator_threads),
        }


@dataclass(frozen=True, slots=True)
class PilotPlan:
    repository_root: Path
    config_path: Path
    configuration_sha256: str
    configuration: dict[str, Any]
    configuration_status: str
    baseline_commit: str
    results_root: Path
    log_path: Path
    policy_path: Path
    heartbeat_interval_seconds: float
    csv_chunk_rows: int
    lendingclub_csv_low_memory: bool
    configured_estimator_threads: int | None
    cells: tuple[PilotCell, ...]


@dataclass(frozen=True, slots=True)
class ArtifactValidation:
    valid: bool
    reason: str
    payload: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    status: str
    completed_cells: int
    total_cells: int
    stop_code: str | None = None
    stop_cell_id: str | None = None


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    authenticated = copy.deepcopy(dict(payload))
    authenticated.pop("artifact_authentication_sha256", None)
    return hashlib.sha256(_canonical_json(authenticated).encode("utf-8")).hexdigest()


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotConfigurationError(f"{field} must be a mapping")
    return dict(value)


def _relative_config_path(root: Path, value: Any, field: str) -> Path:
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PilotConfigurationError(f"{field} must stay inside the repository") from exc
    if not resolved.is_file():
        raise PilotConfigurationError(f"{field} is missing: {resolved}")
    return resolved


def _validate_exact_protocol(configuration: Mapping[str, Any]) -> None:
    protocol = _mapping(configuration.get("protocol"), "protocol")
    exact = {
        "phase": "DEV",
        "fold_ids": [1],
        "ordering": "method_major_cheapest_first",
        "datasets": list(DATASET_ORDER),
        "methods": list(METHOD_ORDER),
        "concurrent_cells": 1,
        "concurrent_folds": 1,
        "oot_access": "forbidden",
        "performance_evaluation": False,
        "full_dev_refit": False,
        "frozen_voting_protocol_changes": "forbidden",
    }
    for field, expected in exact.items():
        if protocol.get(field) != expected:
            raise PilotConfigurationError(
                f"protocol.{field} must be exactly {expected!r}; "
                f"observed {protocol.get(field)!r}"
            )
    if configuration.get("configuration_status") != (
        "provisional_pending_six_cell_review"
    ):
        raise PilotConfigurationError(
            "pilot configuration must remain provisional pending six-cell review"
        )
    inventory = _mapping(
        configuration.get("configuration_inventory"), "configuration_inventory"
    )
    mi = _mapping(inventory.get("mi_mrmr_prompt_7_decision"), "MI-mRMR inventory")
    if (
        mi.get("method_id") != "mrmr_mutual_information"
        or mi.get("n_bins") != 10
        or mi.get("executed_by_this_pipeline") is not False
    ):
        raise PilotConfigurationError(
            "configuration inventory must carry Prompt 7 MI-mRMR n_bins=10 "
            "without scheduling it"
        )


def load_pilot_plan(
    repository_root: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> PilotPlan:
    """Load and fail-close the exact six-cell plan without reading a dataset."""

    root = Path(repository_root).resolve()
    path = _relative_config_path(root, config_path, "pilot config path")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PilotConfigurationError(f"pilot config is unreadable: {path}") from exc
    configuration = _mapping(raw, "pilot configuration")
    if configuration.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise PilotConfigurationError(
            f"unsupported pilot config schema: {configuration.get('schema_version')!r}"
        )
    _validate_exact_protocol(configuration)

    execution = _mapping(configuration.get("execution"), "execution")
    inventory = _mapping(configuration.get("dataset_inventory"), "dataset_inventory")
    method_settings = _mapping(configuration.get("method_settings"), "method_settings")
    limits = _mapping(
        execution.get("wall_clock_limits_seconds"), "wall_clock_limits_seconds"
    )
    if tuple(inventory) != DATASET_ORDER:
        raise PilotConfigurationError("dataset inventory order must match the pilot matrix")
    if tuple(method_settings) != METHOD_ORDER:
        raise PilotConfigurationError("method settings order must be cheapest-first")

    cells: list[PilotCell] = []
    index = 0
    for method_id in METHOD_ORDER:
        descriptor = get_method_descriptor(method_id)
        if descriptor.cost_class != "heavy" or descriptor.allowed_in_frozen_voting:
            raise PilotConfigurationError(
                f"{method_id} must be heavy and excluded from frozen voting"
            )
        limit = float(limits.get(method_id, 0))
        if limit <= 0:
            raise PilotConfigurationError(
                f"positive wall-clock limit required for {method_id}"
            )
        settings = _mapping(method_settings.get(method_id), f"settings for {method_id}")
        for dataset in DATASET_ORDER:
            index += 1
            data = _mapping(inventory.get(dataset), f"dataset inventory for {dataset}")
            expected_count = int(data.get("authenticated_candidate_count", 0))
            if expected_count <= 0:
                raise PilotConfigurationError(
                    f"authenticated candidate count must be positive for {dataset}"
                )
            experiment_path = _relative_config_path(
                root,
                data.get("experiment_config_path"),
                f"experiment config for {dataset}",
            )
            slug = method_id.replace("_", "-")
            cells.append(
                PilotCell(
                    cell_index=index,
                    cell_id=f"{index:02d}-{dataset}-fold-1-{slug}",
                    dataset=dataset,
                    dataset_label=str(data.get("display_label", dataset)),
                    fold_id=1,
                    method_id=method_id,
                    implementation_id=descriptor.implementation_id,
                    expected_candidate_count=expected_count,
                    stable_row_id_column=str(data.get("stable_row_id_column", "")),
                    experiment_config_path=experiment_path.relative_to(root).as_posix(),
                    selector_kwargs=copy.deepcopy(settings),
                    wall_clock_limit_seconds=limit,
                )
            )
    if len(cells) != 6:
        raise PilotConfigurationError(f"pilot plan must contain six cells, got {len(cells)}")

    results = Path(str(execution.get("results_root", "")))
    results_root = results.resolve() if results.is_absolute() else (root / results).resolve()
    try:
        results_root.relative_to(root)
    except ValueError as exc:
        raise PilotConfigurationError("results root must stay inside the repository") from exc
    configured_threads = execution.get("estimator_threads")
    if configured_threads is not None and int(configured_threads) <= 0:
        raise PilotConfigurationError("execution.estimator_threads must be positive or null")
    return PilotPlan(
        repository_root=root,
        config_path=path,
        configuration_sha256=sha256_file(path),
        configuration=configuration,
        configuration_status=str(configuration["configuration_status"]),
        baseline_commit=str(configuration.get("baseline_commit", "")),
        results_root=results_root,
        log_path=Path(str(execution.get("log_path", DEFAULT_RESEARCH_LOG))),
        policy_path=Path(str(execution.get("policy_path", DEFAULT_POLICY_PATH))),
        heartbeat_interval_seconds=float(execution.get("heartbeat_interval_seconds", 30)),
        csv_chunk_rows=int(execution.get("csv_chunk_rows", 25_000)),
        lendingclub_csv_low_memory=bool(
            execution.get("lendingclub_csv_low_memory", False)
        ),
        configured_estimator_threads=(
            None if configured_threads is None else int(configured_threads)
        ),
        cells=tuple(cells),
    )


def cell_artifact_path(plan: PilotPlan, cell: PilotCell) -> Path:
    return plan.results_root / "cells" / cell.cell_id / "artifact.json"


def _artifact_base(
    plan: PilotPlan,
    cell: PilotCell,
    *,
    terminal_state: str,
    stop_reason: str | None,
    estimator_threads: int,
) -> dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "pilot_cell": cell.cell_id,
        "cell_index": cell.cell_index,
        "dataset": cell.dataset,
        "dataset_label": cell.dataset_label,
        "phase": "DEV",
        "fold_id": cell.fold_id,
        "fold_definition": "grouped_time_series_cv_5_splits_gap_1_expanding_fold_1",
        "method_id": cell.method_id,
        "implementation_id": cell.implementation_id,
        "configuration_status": plan.configuration_status,
        "configuration_sha256": plan.configuration_sha256,
        "pilot_config_path": plan.config_path.relative_to(plan.repository_root).as_posix(),
        "selector_configuration_requested": copy.deepcopy(cell.selector_kwargs),
        "random_seed": int(cell.selector_kwargs.get("random_state", 42)),
        "thread_count": int(estimator_threads),
        "wall_clock_limit_seconds": cell.wall_clock_limit_seconds,
        "oot_access": "forbidden",
        "oot_accessed": False,
        "oot_evaluated": False,
        "performance_evaluation_performed": False,
        "frozen_voting_protocol_modified": False,
        "terminal_state": terminal_state,
        "stop_reason": stop_reason,
        "selector_result": None,
        "authenticated_dev_identity": None,
        "input_counts": {
            "dev_rows": None,
            "fold_training_rows": None,
            "fold_validation_rows": None,
            "candidate_features": cell.expected_candidate_count,
        },
        "selected_feature_count": None,
        "ordered_selected_features": None,
        "requested_budget": cell.selector_kwargs.get("k"),
        "natural_support_count": None,
        "feasibility_status": None,
        "runtime_seconds": None,
        "peak_process_tree_rss_bytes": None,
        "minimum_system_available_ram_bytes": None,
        "supervisor": None,
    }


def _with_authentication(payload: Mapping[str, Any]) -> dict[str, Any]:
    authenticated = copy.deepcopy(dict(payload))
    authenticated["artifact_authentication_sha256"] = _payload_sha256(authenticated)
    return authenticated


def _required_heavy_evidence(method_id: str, result: Mapping[str, Any]) -> None:
    heavy = result.get("heavy_metadata")
    if not isinstance(heavy, Mapping):
        raise PilotArtifactError("completed heavy result has no heavy_metadata")
    if method_id == "catboost_shap":
        sample = heavy.get("explanation_sample")
        required = {
            "feature_importance_type",
            "shap_calc_type",
            "aggregation",
            "estimator_fit_count",
            "shap_calculation_count",
        }
        if not isinstance(sample, Mapping) or not {
            "realized_size",
            "row_identity_sha256",
            "scope",
        } <= set(sample):
            raise PilotArtifactError("CatBoost-SHAP explanation sample is incomplete")
        if not required <= set(heavy):
            raise PilotArtifactError("CatBoost-SHAP method evidence is incomplete")
    elif method_id == "boruta_random_forest":
        required = {
            "confirmed_count",
            "tentative_count",
            "rejected_count",
            "selection_mode",
            "natural_support_definition",
            "forest_n_estimators_configured",
            "boruta_max_iter_configured",
            "engine_iteration_count",
            "estimator_fit_count",
        }
        if not required <= set(heavy):
            raise PilotArtifactError("Boruta method evidence is incomplete")
        support_total = sum(
            int(heavy[name])
            for name in ("confirmed_count", "tentative_count", "rejected_count")
        )
        if support_total != int(result.get("candidate_universe_count", -1)):
            raise PilotArtifactError("Boruta support states do not partition the universe")
    elif method_id == "rfe_catboost":
        required = {
            "initial_feature_count",
            "final_feature_count",
            "requested_elimination_steps",
            "realized_elimination_steps",
            "elimination_iteration_count",
            "elimination_history",
            "estimator_fit_count",
        }
        if not required <= set(heavy):
            raise PilotArtifactError("RFE method evidence is incomplete")
        if int(heavy["elimination_iteration_count"]) != len(
            heavy["elimination_history"]
        ):
            raise PilotArtifactError("RFE elimination history length is inconsistent")


def validate_cell_artifact(plan: PilotPlan, cell: PilotCell) -> ArtifactValidation:
    """Authenticate a completed artifact without reading any research dataset."""

    path = cell_artifact_path(plan, cell)
    if not path.is_file():
        return ArtifactValidation(False, "missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise PilotArtifactError("artifact root is not a mapping")
        observed_hash = payload.get("artifact_authentication_sha256")
        if not isinstance(observed_hash, str) or observed_hash != _payload_sha256(payload):
            raise PilotArtifactError("artifact authentication hash mismatch")
        exact = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "pilot_cell": cell.cell_id,
            "cell_index": cell.cell_index,
            "dataset": cell.dataset,
            "phase": "DEV",
            "fold_id": 1,
            "method_id": cell.method_id,
            "implementation_id": cell.implementation_id,
            "configuration_sha256": plan.configuration_sha256,
            "configuration_status": "provisional_pending_six_cell_review",
            "oot_access": "forbidden",
            "oot_accessed": False,
            "oot_evaluated": False,
            "performance_evaluation_performed": False,
            "frozen_voting_protocol_modified": False,
            "terminal_state": "completed",
        }
        for field, expected in exact.items():
            if payload.get(field) != expected:
                raise PilotArtifactError(
                    f"{field} mismatch: expected {expected!r}, got {payload.get(field)!r}"
                )
        result = payload.get("selector_result")
        if not isinstance(result, Mapping):
            raise PilotArtifactError("completed artifact has no selector result")
        if result.get("method_id") != cell.method_id:
            raise PilotArtifactError("selector method identity mismatch")
        if result.get("implementation_id") != cell.implementation_id:
            raise PilotArtifactError("selector implementation identity mismatch")
        if int(result.get("candidate_universe_count", -1)) != cell.expected_candidate_count:
            raise PilotArtifactError("candidate universe count mismatch")
        selected = result.get("selected_features")
        if not isinstance(selected, list) or len(selected) != len(set(selected)):
            raise PilotArtifactError("selected feature order is invalid")
        if payload.get("ordered_selected_features") != selected:
            raise PilotArtifactError("top-level selected feature order mismatch")
        if int(payload.get("selected_feature_count", -1)) != len(selected):
            raise PilotArtifactError("selected feature count mismatch")
        if payload.get("requested_budget") != result.get("requested_budget"):
            raise PilotArtifactError("requested budget mismatch")
        if payload.get("feasibility_status") != result.get("budget_status"):
            raise PilotArtifactError("feasibility status mismatch")
        identity = payload.get("authenticated_dev_identity")
        if not isinstance(identity, Mapping) or identity.get("fold_id") != 1:
            raise PilotArtifactError("authenticated DEV fold identity is missing")
        for field in (
            "dev_ordered_row_id_sha256",
            "training_ordered_row_id_sha256",
            "validation_ordered_row_id_sha256",
            "training_ordered_row_id_target_sha256",
        ):
            value = identity.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise PilotArtifactError(f"invalid DEV identity field: {field}")
        if payload.get("runtime_seconds") is None:
            raise PilotArtifactError("runtime is missing")
        if payload.get("peak_process_tree_rss_bytes") is None:
            raise PilotArtifactError("peak RSS is missing")
        if payload.get("minimum_system_available_ram_bytes") is None:
            raise PilotArtifactError("minimum available RAM is missing")
        _required_heavy_evidence(cell.method_id, result)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return ArtifactValidation(False, str(exc))
    return ArtifactValidation(True, "authenticated_complete", payload)


def _write_cell_state(
    plan: PilotPlan,
    cell: PilotCell,
    payload: Mapping[str, Any],
) -> None:
    existing = validate_cell_artifact(plan, cell)
    if existing.valid:
        raise FileExistsError(
            f"refusing to overwrite valid completed pilot cell {cell.cell_id}"
        )
    write_json_atomic(
        cell_artifact_path(plan, cell),
        _with_authentication(payload),
        overwrite=True,
    )


def _read_exclusions(root: Path, relative_path: str) -> tuple[str, ...]:
    payload = yaml.safe_load((root / relative_path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"experiment configuration is invalid: {relative_path}")
    exclusions = [str(value) for value in payload.get("excluded_feature_columns", [])]
    return tuple(dict.fromkeys(exclusions))


def _report_stage(
    stop_event: Any,
    stage_queue: Any,
    *,
    stage: str,
    fold_id: int,
    component: str,
    **fields: Any,
) -> None:
    if stop_event.is_set():
        raise RuntimeError(f"cooperative stop requested before {stage}")
    stage_queue.put(
        {"stage": stage, "fold_id": fold_id, "component": component, **fields}
    )


def heavy_selector_pilot_cell_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
    repository_root: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one real-data DEV cell in a supervised child process.

    Tests replace the canonical loader or the supervisor with synthetic fixtures;
    this function itself has no OOT import and performs no metric evaluation.
    """

    from credit_risk_fs.experiments.rank_voting import canonical_fold_projection
    from credit_risk_fs.experiments.row_alignment import (
        ordered_row_id_sha256,
        ordered_row_id_target_sha256,
    )
    from credit_risk_fs.pipelines.common import prepare_voting_pilot_dev_data
    from credit_risk_fs.preprocessing.encoding import OriginalFeatureNumericEncoder

    root = Path(repository_root).resolve()
    cell = dict(spec)
    if (
        cell.get("phase") != "DEV"
        or cell.get("fold_id") != 1
        or cell.get("oot_access") != "forbidden"
        or cell.get("performance_evaluation") is not False
        or cell.get("full_dev_refit") is not False
    ):
        raise PilotConfigurationError("worker received a non-DEV-fold-1 specification")
    if cell.get("dataset") not in DATASET_ORDER or cell.get("method_id") not in METHOD_ORDER:
        raise PilotConfigurationError("worker received an unauthorized pilot cell")

    dataset = str(cell["dataset"])
    fold_id = 1
    _report_stage(
        stop_event,
        stage_queue,
        stage="pilot_dev_data_loading",
        fold_id=fold_id,
        component="prepare_voting_pilot_dev_data",
        pilot_cell=cell["cell_id"],
        dataset=dataset,
        method_id=cell["method_id"],
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "csv_chunk_rows": int(cell["csv_chunk_rows"]),
    }
    if dataset == "lendingclub_v2":
        loader_kwargs["csv_low_memory"] = bool(cell["lendingclub_csv_low_memory"])
    prepared = prepare_voting_pilot_dev_data(root, **loader_kwargs)
    if any(int(item.get("oot_rows_retained", -1)) != 0 for item in prepared.data_access_log):
        raise RuntimeError("canonical DEV loader reported retained OOT rows")
    candidates = tuple(map(str, prepared.candidate_features))
    expected_count = int(cell["expected_candidate_count"])
    if len(candidates) != expected_count or candidates != tuple(
        map(str, prepared.candidate_universe or ())
    ):
        raise RuntimeError(
            f"authenticated candidate universe mismatch for {dataset}: "
            f"expected={expected_count}, observed={len(candidates)}"
        )

    _report_stage(
        stop_event,
        stage_queue,
        stage="pilot_fold_projection",
        fold_id=fold_id,
        component="canonical_fold_projection",
        dev_row_count=len(prepared.y),
        input_feature_count=len(candidates),
    )
    projection = canonical_fold_projection(
        y=prepared.y,
        stable_row_ids=prepared.stable_row_ids,
        time_values=prepared.time_values,
        fold_id=fold_id,
    )
    positions = projection["source_positions"]
    training_indices = projection["training_indices"]
    validation_indices = projection["validation_indices"]
    training_ids = projection["ids"].iloc[training_indices].astype(str).reset_index(drop=True)
    validation_ids = projection["ids"].iloc[validation_indices].astype(str).reset_index(drop=True)
    y_training = projection["y"].iloc[training_indices].reset_index(drop=True)
    X_training = prepared.X.iloc[positions[training_indices]].copy()
    X_training.index = training_ids.tolist()
    dev_row_count = int(len(projection["y"]))
    dev_identity = {
        "ordered_row_id_sha256": prepared.split_evidence["ordered_row_id_sha256"],
        "ordered_row_id_target_sha256": prepared.split_evidence[
            "ordered_row_id_target_sha256"
        ],
    }
    source_artifact_hashes = dict(prepared.source_artifact_hashes)
    data_access_roles = [
        {
            "role": item.get("role"),
            "requested_columns": item.get("requested_columns"),
            "oot_rows_retained": item.get("oot_rows_retained"),
        }
        for item in prepared.data_access_log
    ]
    del prepared, positions, projection
    gc.collect()

    _report_stage(
        stop_event,
        stage_queue,
        stage="pilot_selection_encoding",
        fold_id=fold_id,
        component="original_feature_numeric_encoder",
        input_row_count=len(X_training),
        input_feature_count=X_training.shape[1],
    )
    encoder = OriginalFeatureNumericEncoder()
    X_numeric = encoder.fit_transform(X_training)
    if tuple(map(str, X_numeric.columns)) != candidates:
        raise RuntimeError("selection encoding changed the authenticated feature order")
    del X_training, encoder
    gc.collect()

    descriptor = get_method_descriptor(str(cell["method_id"]))
    if descriptor.implementation_id != cell.get("implementation_id"):
        raise RuntimeError("selector registry implementation identity changed")
    selector_class = descriptor.load()
    kwargs = copy.deepcopy(dict(cell["selector_kwargs"]))
    kwargs["excluded_columns"] = tuple(
        dict.fromkeys(
            (
                *_read_exclusions(root, str(cell["experiment_config_path"])),
                str(cell["stable_row_id_column"]),
            )
        )
    )
    kwargs["fit_scope"] = "dev_fold_training_only"
    selector = selector_class(**kwargs)
    method_stage = f"pilot_{cell['method_id']}"
    _report_stage(
        stop_event,
        stage_queue,
        stage=method_stage,
        fold_id=fold_id,
        component=str(cell["method_id"]),
        input_row_count=len(X_numeric),
        input_feature_count=X_numeric.shape[1],
        requested_budget=kwargs.get("k"),
        thread_count=cell["estimator_threads"],
    )
    with suppress_third_party_output():
        selector.fit(X_numeric, y_training)
    result = selector.result.to_dict()

    return {
        "pilot_cell": cell["cell_id"],
        "selector_result": result,
        "authenticated_dev_identity": {
            "dataset": dataset,
            "phase": "DEV",
            "fold_id": 1,
            "fold_definition": (
                "grouped_time_series_cv_5_splits_gap_1_expanding_fold_1"
            ),
            "stable_row_id_column": cell["stable_row_id_column"],
            "dev_ordered_row_id_sha256": dev_identity["ordered_row_id_sha256"],
            "dev_ordered_row_id_target_sha256": dev_identity[
                "ordered_row_id_target_sha256"
            ],
            "training_ordered_row_id_sha256": ordered_row_id_sha256(training_ids),
            "training_ordered_row_id_target_sha256": ordered_row_id_target_sha256(
                training_ids, y_training
            ),
            "validation_ordered_row_id_sha256": ordered_row_id_sha256(validation_ids),
            "candidate_universe_sha256": result["candidate_universe_sha256"],
            "source_artifact_hashes": source_artifact_hashes,
            "data_access_roles": data_access_roles,
            "opened_oot_paths": [],
            "retained_oot_rows": 0,
        },
        "input_counts": {
            "dev_rows": dev_row_count,
            "fold_training_rows": int(len(training_indices)),
            "fold_validation_rows": int(len(validation_indices)),
            "candidate_features": int(X_numeric.shape[1]),
        },
        "preprocessing": {
            "implementation": (
                "credit_risk_fs.preprocessing.encoding.OriginalFeatureNumericEncoder"
            ),
            "fit_scope": "dev_fold_training_only",
            "output_dtype": "float32",
            "output_feature_count": int(X_numeric.shape[1]),
            "feature_order_preserved": True,
        },
        "oot_accessed": False,
        "performance_evaluation_performed": False,
    }


def _supervisor_runtime(result: Any, measured_seconds: float) -> float:
    samples = tuple(getattr(result, "samples", ()) or ())
    sampled = max((float(item.elapsed_seconds) for item in samples), default=0.0)
    return max(float(measured_seconds), sampled)


def _terminal_state(result: Any) -> str:
    status = str(getattr(result, "status", "failed"))
    return {
        "completed": "completed",
        "interrupted": "manually_interrupted",
        "timed_out": "timed_out",
        "aborted_resource_limit": "resource_aborted",
        "failed": "failed",
    }.get(status, "failed")


def _complete_terminal_payload(
    base: dict[str, Any],
    result: Any,
    *,
    runtime_seconds: float,
) -> dict[str, Any]:
    state = _terminal_state(result)
    stop_reason = getattr(result, "stop_code", None) or getattr(
        result, "worker_error", None
    )
    payload = copy.deepcopy(base)
    payload.update(
        {
            "terminal_state": state,
            "stop_reason": stop_reason,
            "runtime_seconds": float(runtime_seconds),
            "peak_process_tree_rss_bytes": int(
                getattr(result, "peak_process_tree_rss_bytes", 0) or 0
            ),
            "minimum_system_available_ram_bytes": getattr(
                result, "minimum_system_available_ram_bytes", None
            ),
            "supervisor": {
                "status": getattr(result, "status", "failed"),
                "stop_code": getattr(result, "stop_code", None),
                "worker_exit_code": getattr(result, "worker_exit_code", None),
                "worker_error": getattr(result, "worker_error", None),
                "peak_process_tree_rss_bytes": getattr(
                    result, "peak_process_tree_rss_bytes", 0
                ),
                "minimum_system_available_ram_bytes": getattr(
                    result, "minimum_system_available_ram_bytes", None
                ),
                "child_cleanup_confirmed": getattr(
                    result, "child_cleanup_confirmed", None
                ),
                "final_stage": getattr(result, "final_stage", None),
                "final_fold_id": getattr(result, "final_fold_id", None),
            },
        }
    )
    returned = getattr(result, "return_value", None)
    if state == "completed" and isinstance(returned, Mapping):
        selector_result = dict(returned["selector_result"])
        payload.update(
            {
                "selector_result": selector_result,
                "authenticated_dev_identity": dict(
                    returned["authenticated_dev_identity"]
                ),
                "input_counts": dict(returned["input_counts"]),
                "preprocessing": dict(returned["preprocessing"]),
                "selected_feature_count": int(
                    selector_result["actual_selected_count"]
                ),
                "ordered_selected_features": list(
                    selector_result["selected_features"]
                ),
                "requested_budget": selector_result["requested_budget"],
                "natural_support_count": selector_result[
                    "natural_selected_count"
                ],
                "feasibility_status": selector_result["budget_status"],
                "oot_accessed": bool(returned.get("oot_accessed", False)),
                "performance_evaluation_performed": bool(
                    returned.get("performance_evaluation_performed", False)
                ),
            }
        )
    return payload


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def authenticate_repository(plan: PilotPlan) -> dict[str, Any]:
    """Require a clean descendant of the Prompt 8 handoff before real execution."""

    head = _git(plan.repository_root, "rev-parse", "HEAD")
    dirty = _git(plan.repository_root, "status", "--porcelain", "--untracked-files=normal")
    if dirty:
        raise RuntimeError("heavy-selector pilot execution requires a clean worktree")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", plan.baseline_commit, head],
        cwd=plan.repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            f"Prompt 8 baseline {plan.baseline_commit} is not an ancestor of HEAD"
        )
    return {
        "git_commit": head,
        "baseline_commit": plan.baseline_commit,
        "baseline_is_ancestor": True,
        "git_dirty": False,
        "pilot_configuration_sha256": plan.configuration_sha256,
    }


def _resolve_policy_and_preflight(
    plan: PilotPlan,
) -> tuple[ResolvedExecutionPolicy, dict[str, Any]]:
    plan.results_root.mkdir(parents=True, exist_ok=True)
    configured = load_execution_policy(plan.repository_root, plan.policy_path)
    capacity = detect_hardware(plan.results_root, Path(tempfile.gettempdir()).resolve())
    resolved = resolve_execution_policy(configured, capacity)
    report = run_preflight(
        repository_root=plan.repository_root,
        config_path=plan.policy_path,
        results_root=plan.results_root,
        temp_root=Path(tempfile.gettempdir()).resolve(),
        requested_accelerator="cpu",
        capacity=capacity,
    )
    parallel = resolved.parallelism
    if (
        parallel.concurrent_experiment_runs != 1
        or parallel.concurrent_folds != 1
        or parallel.data_loader_workers != 0
    ):
        raise RuntimeError("pilot policy must allow only one cell and one fold at a time")
    if report.get("status") != "pass":
        raise RuntimeError(
            f"pilot preflight rejected: {report.get('blocking_reasons', [])}"
        )
    return resolved, report


def execute_pilot_pipeline(
    plan: PilotPlan,
    *,
    estimator_threads: int | None = None,
    require_clean_repository: bool = True,
    supervisor: Callable[..., Any] = supervise_worker,
    readiness_checker: Callable[..., Any] = wait_for_inter_run_readiness,
    policy_preflight: Callable[
        [PilotPlan], tuple[ResolvedExecutionPolicy, dict[str, Any]]
    ] = _resolve_policy_and_preflight,
) -> PipelineOutcome:
    """Run all six cells sequentially, stopping after the first non-completion."""

    provenance = (
        authenticate_repository(plan)
        if require_clean_repository
        else {
            "git_commit": "synthetic-test",
            "baseline_commit": plan.baseline_commit,
            "baseline_is_ancestor": True,
            "git_dirty": False,
            "pilot_configuration_sha256": plan.configuration_sha256,
        }
    )
    policy, preflight = policy_preflight(plan)
    configured_threads = (
        estimator_threads
        if estimator_threads is not None
        else plan.configured_estimator_threads
    )
    threads = int(
        policy.parallelism.estimator_threads
        if configured_threads is None
        else configured_threads
    )
    if threads <= 0 or threads > policy.parallelism.estimator_threads:
        raise PilotConfigurationError(
            f"requested threads={threads} exceed resolved limit "
            f"{policy.parallelism.estimator_threads}"
        )
    emit_research_event(
        "configuration_authenticated",
        message="Heavy-selector DEV pilot configuration authenticated",
        priority=True,
        configuration_sha256=plan.configuration_sha256,
        cell_count=len(plan.cells),
        fold_ids=[1],
        phase="DEV",
        oot_access="forbidden",
        estimator_threads=threads,
        provenance=provenance,
        preflight_status=preflight.get("status"),
    )

    completed = 0
    for cell in plan.cells:
        validation = validate_cell_artifact(plan, cell)
        if validation.valid:
            completed += 1
            emit_research_event(
                "run_resume_decision",
                message=f"Skipping authenticated completed pilot cell {cell.cell_id}",
                priority=True,
                pilot_cell=cell.cell_id,
                cell_index=cell.cell_index,
                dataset=cell.dataset,
                fold_id=1,
                selector=cell.method_id,
                decision="skip_authenticated_complete",
            )
            continue

        with bind_research_context(
            run_id=cell.cell_id,
            pilot_cell=cell.cell_id,
            cell_index=cell.cell_index,
            dataset=cell.dataset,
            fold_id=1,
            phase="DEV",
            selector=cell.method_id,
            seed=int(cell.selector_kwargs.get("random_state", 42)),
        ):
            emit_research_event(
                "run_resume_decision",
                message=(
                    "Executing earliest incomplete or invalid pilot cell; "
                    f"prior_state={validation.reason}"
                ),
                priority=True,
                decision="execute",
                prior_artifact_state=validation.reason,
            )
            readiness = readiness_checker(
                policy=policy,
                results_root=plan.results_root,
                temp_root=Path(tempfile.gettempdir()).resolve(),
            )
            running = _artifact_base(
                plan,
                cell,
                terminal_state="running",
                stop_reason=None,
                estimator_threads=threads,
            )
            running["repository_provenance"] = provenance
            running["preflight_status"] = preflight.get("status")
            if not readiness.ready:
                running.update(
                    {
                        "terminal_state": "resource_aborted",
                        "stop_reason": readiness.stop_code or PREFLIGHT_REJECTED,
                        "runtime_seconds": float(readiness.elapsed_seconds),
                        "peak_process_tree_rss_bytes": int(readiness.parent_rss_bytes),
                        "minimum_system_available_ram_bytes": int(
                            readiness.system_available_ram_bytes
                        ),
                    }
                )
                _write_cell_state(plan, cell, running)
                emit_research_event(
                    "session_controlled_stop",
                    level="ERROR",
                    message="Inter-cell resource readiness rejected the next pilot cell",
                    priority=True,
                    stop_code=running["stop_reason"],
                )
                return PipelineOutcome(
                    "resource_aborted",
                    completed,
                    len(plan.cells),
                    str(running["stop_reason"]),
                    cell.cell_id,
                )
            _write_cell_state(plan, cell, running)
            emit_research_event(
                "run_execution_started",
                message="Heavy-selector DEV pilot cell started",
                priority=True,
                method_id=cell.method_id,
                implementation_id=cell.implementation_id,
                wall_clock_limit_seconds=cell.wall_clock_limit_seconds,
                thread_count=threads,
            )
            spec = cell.to_worker_spec(plan=plan, estimator_threads=threads)
            started = time.monotonic()
            try:
                supervised = supervisor(
                    worker_target=(
                        "credit_risk_fs.experiments.heavy_selector_pilots:"
                        "heavy_selector_pilot_cell_worker"
                    ),
                    worker_kwargs={
                        "repository_root": str(plan.repository_root),
                        "spec": spec,
                    },
                    policy=policy,
                    results_root=plan.results_root,
                    temp_root=Path(tempfile.gettempdir()).resolve(),
                    run_association=cell.cell_id,
                    heartbeat_interval_seconds=plan.heartbeat_interval_seconds,
                    max_wall_clock_seconds=cell.wall_clock_limit_seconds,
                )
            except KeyboardInterrupt:
                supervised = _interrupt_result(cell)
            except BaseException as exc:
                logger.exception("Unexpected pilot supervisor error for %s", cell.cell_id)
                supervised = _failure_result(cell, exc)
            runtime = _supervisor_runtime(supervised, time.monotonic() - started)
            terminal = _complete_terminal_payload(running, supervised, runtime_seconds=runtime)
            if terminal["terminal_state"] == "completed":
                try:
                    _validate_completed_payload_before_write(plan, cell, terminal)
                except Exception as exc:
                    logger.exception("Completed worker payload is invalid for %s", cell.cell_id)
                    terminal = _complete_terminal_payload(
                        running,
                        _failure_result(cell, exc),
                        runtime_seconds=runtime,
                    )
            _write_cell_state(plan, cell, terminal)
            state = str(terminal["terminal_state"])
            emit_research_event(
                "run_finalized",
                level="INFO" if state == "completed" else "ERROR",
                message=f"Heavy-selector DEV pilot cell ended with {state}",
                priority=True,
                status=state,
                stop_code=terminal.get("stop_reason"),
                runtime_seconds=runtime,
                peak_process_tree_rss_bytes=terminal.get(
                    "peak_process_tree_rss_bytes"
                ),
                minimum_system_available_ram_bytes=terminal.get(
                    "minimum_system_available_ram_bytes"
                ),
            )
            if state != "completed":
                return PipelineOutcome(
                    state,
                    completed,
                    len(plan.cells),
                    None if terminal.get("stop_reason") is None else str(
                        terminal["stop_reason"]
                    ),
                    cell.cell_id,
                )
            post_write = validate_cell_artifact(plan, cell)
            if not post_write.valid:
                raise PilotArtifactError(
                    f"newly written artifact failed authentication: {post_write.reason}"
                )
            completed += 1
    return PipelineOutcome("completed", completed, len(plan.cells))


def _interrupt_result(cell: PilotCell) -> SupervisorResult:
    return SupervisorResult(
        status="interrupted",
        stop_code=MANUAL_INTERRUPT,
        worker_exit_code=None,
        return_value=None,
        worker_error=None,
        samples=(),
        warnings=(),
        peak_process_tree_rss_bytes=0,
        peak_process_gpu_bytes=None,
        minimum_system_available_ram_bytes=0,
        minimum_results_free_disk_bytes=None,
        minimum_temp_free_disk_bytes=None,
        child_cleanup_confirmed=True,
        final_stage=f"pilot_{cell.method_id}",
        final_fold_id=1,
    )


def _failure_result(cell: PilotCell, error: BaseException) -> SupervisorResult:
    return SupervisorResult(
        status="failed",
        stop_code="worker_crash",
        worker_exit_code=None,
        return_value=None,
        worker_error=f"{type(error).__name__}: {error}",
        samples=(),
        warnings=(),
        peak_process_tree_rss_bytes=0,
        peak_process_gpu_bytes=None,
        minimum_system_available_ram_bytes=0,
        minimum_results_free_disk_bytes=None,
        minimum_temp_free_disk_bytes=None,
        child_cleanup_confirmed=True,
        final_stage=f"pilot_{cell.method_id}",
        final_fold_id=1,
    )


def _validate_completed_payload_before_write(
    plan: PilotPlan,
    cell: PilotCell,
    payload: Mapping[str, Any],
) -> None:
    del plan
    result = payload.get("selector_result")
    if not isinstance(result, Mapping):
        raise PilotArtifactError("completed worker returned no selector result")
    if result.get("method_id") != cell.method_id:
        raise PilotArtifactError("completed worker returned the wrong method")
    if int(result.get("candidate_universe_count", -1)) != cell.expected_candidate_count:
        raise PilotArtifactError("completed worker returned the wrong candidate count")
    if payload.get("oot_accessed") is not False:
        raise PilotArtifactError("completed worker reported OOT access")
    if payload.get("performance_evaluation_performed") is not False:
        raise PilotArtifactError("completed worker reported performance evaluation")
    _required_heavy_evidence(cell.method_id, result)


def build_status_report(plan: PilotPlan) -> dict[str, Any]:
    """Read only config and pilot artifacts; no dataset loader is reachable here."""

    rows: list[dict[str, Any]] = []
    first_incomplete: PilotCell | None = None
    completed = 0
    for cell in plan.cells:
        validation = validate_cell_artifact(plan, cell)
        path = cell_artifact_path(plan, cell)
        payload: dict[str, Any] | None = validation.payload
        if payload is None and path.is_file():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                payload = raw if isinstance(raw, dict) else None
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                payload = None
        if validation.valid:
            completed += 1
            state = "completed"
        else:
            state = str((payload or {}).get("terminal_state", "missing_or_invalid"))
            if first_incomplete is None:
                first_incomplete = cell
        selector_result = (payload or {}).get("selector_result") or {}
        heavy = selector_result.get("heavy_metadata") or {}
        row = {
            "cell_index": cell.cell_index,
            "pilot_cell": cell.cell_id,
            "dataset": cell.dataset,
            "fold_id": 1,
            "method_id": cell.method_id,
            "state": state,
            "artifact_valid": validation.valid,
            "validation_reason": validation.reason,
            "runtime_seconds": (payload or {}).get("runtime_seconds"),
            "peak_process_tree_rss_bytes": (payload or {}).get(
                "peak_process_tree_rss_bytes"
            ),
            "minimum_system_available_ram_bytes": (payload or {}).get(
                "minimum_system_available_ram_bytes"
            ),
            "stop_reason": (payload or {}).get("stop_reason"),
            "boruta_support_counts": (
                {
                    name: heavy.get(f"{name}_count")
                    for name in ("confirmed", "tentative", "rejected")
                }
                if cell.method_id == "boruta_random_forest" and heavy
                else None
            ),
            "rfe_estimator_fit_count": (
                heavy.get("estimator_fit_count")
                if cell.method_id == "rfe_catboost" and heavy
                else None
            ),
            "catboost_shap_explanation_sample": (
                heavy.get("explanation_sample")
                if cell.method_id == "catboost_shap" and heavy
                else None
            ),
        }
        rows.append(row)
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "completed_cells": completed,
        "total_cells": len(plan.cells),
        "current_or_next_cell": (
            None
            if first_incomplete is None
            else {
                "cell_index": first_incomplete.cell_index,
                "pilot_cell": first_incomplete.cell_id,
                "dataset": first_incomplete.dataset,
                "fold_id": 1,
                "method_id": first_incomplete.method_id,
            }
        ),
        "controlled_stops": [
            {
                "pilot_cell": row["pilot_cell"],
                "state": row["state"],
                "stop_reason": row["stop_reason"],
            }
            for row in rows
            if row["state"] in CONTROLLED_STOP_STATES
        ],
        "cells": rows,
        "dataset_access_performed": False,
    }


def format_status_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"Heavy-selector DEV pilots: {report['completed_cells']}/{report['total_cells']} completed"
    ]
    next_cell = report.get("current_or_next_cell")
    if next_cell is None:
        lines.append("Current/next cell: none (all six authenticated complete)")
    else:
        lines.append(
            "Current/next cell: "
            f"{next_cell['cell_index']:02d} | {next_cell['dataset']} | "
            f"fold 1 | {next_cell['method_id']}"
        )
    for row in report["cells"]:
        detail = [
            f"{row['cell_index']:02d}",
            row["dataset"],
            "fold 1",
            row["method_id"],
            row["state"],
        ]
        if row.get("runtime_seconds") is not None:
            detail.append(f"runtime={float(row['runtime_seconds']):.1f}s")
        if row.get("peak_process_tree_rss_bytes") is not None:
            detail.append(
                f"peak_rss={float(row['peak_process_tree_rss_bytes']) / 1024**3:.2f}GiB"
            )
        if row.get("minimum_system_available_ram_bytes") is not None:
            detail.append(
                "min_available="
                f"{float(row['minimum_system_available_ram_bytes']) / 1024**3:.2f}GiB"
            )
        if row.get("stop_reason"):
            detail.append(f"stop={row['stop_reason']}")
        if row.get("boruta_support_counts"):
            counts = row["boruta_support_counts"]
            detail.append(
                "boruta="
                f"{counts['confirmed']}/{counts['tentative']}/{counts['rejected']}"
            )
        if row.get("rfe_estimator_fit_count") is not None:
            detail.append(f"rfe_fits={row['rfe_estimator_fit_count']}")
        if row.get("catboost_shap_explanation_sample"):
            sample = row["catboost_shap_explanation_sample"]
            detail.append(
                f"shap_sample={sample.get('realized_size')} "
                f"id={str(sample.get('row_identity_sha256', ''))[:12]}"
            )
        lines.append(" | ".join(map(str, detail)))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or inspect the complete six-cell heavy-selector DEV pilot."
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--status",
        action="store_true",
        help="Read config and existing pilot artifacts only; never load a dataset.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repository_root.resolve()
    plan = load_pilot_plan(root, args.config)
    if args.status:
        print(format_status_report(build_status_report(plan)))
        return 0

    command_arguments = list(sys.argv[1:] if argv is None else argv)
    with ResearchLogSession(
        plan.log_path,
        repository_root=root,
        command_arguments=command_arguments,
    ) as session:
        try:
            outcome = execute_pilot_pipeline(
                plan,
                estimator_threads=args.threads,
                require_clean_repository=True,
            )
        except KeyboardInterrupt:
            session.finish(
                "session_interrupted",
                level="ERROR",
                message="Heavy-selector DEV pilot interrupted manually",
                exception_class="KeyboardInterrupt",
                stop_code=MANUAL_INTERRUPT,
            )
            return 130
        except BaseException as exc:
            session.finish(
                "session_failed",
                level="ERROR",
                message=f"Heavy-selector DEV pilot failed: {type(exc).__name__}: {exc}",
                exception_class=type(exc).__name__,
                traceback=traceback.format_exc(),
            )
            return 1
        if outcome.status == "completed":
            session.finish(
                "session_completed",
                message="All six heavy-selector DEV pilot cells completed",
                completed_cells=outcome.completed_cells,
                total_cells=outcome.total_cells,
            )
            return 0
        if outcome.status == "manually_interrupted":
            session.finish(
                "session_interrupted",
                level="ERROR",
                message="Heavy-selector DEV pilot interrupted manually",
                exception_class="KeyboardInterrupt",
                stop_code=outcome.stop_code,
                pilot_cell=outcome.stop_cell_id,
            )
            return 130
        session.finish(
            (
                "session_controlled_stop"
                if outcome.status in CONTROLLED_STOP_STATES
                else "session_failed"
            ),
            level="ERROR",
            message=(
                f"Heavy-selector DEV pilot stopped at {outcome.stop_cell_id}: "
                f"{outcome.status} ({outcome.stop_code})"
            ),
            stop_code=outcome.stop_code,
            pilot_cell=outcome.stop_cell_id,
            status=outcome.status,
        )
        return 2 if outcome.status in CONTROLLED_STOP_STATES else 1


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "CONFIG_SCHEMA_VERSION",
    "DATASET_ORDER",
    "DEFAULT_CONFIG_PATH",
    "METHOD_ORDER",
    "ArtifactValidation",
    "PilotArtifactError",
    "PilotCell",
    "PilotConfigurationError",
    "PilotPlan",
    "PipelineOutcome",
    "build_status_report",
    "cell_artifact_path",
    "execute_pilot_pipeline",
    "format_status_report",
    "heavy_selector_pilot_cell_worker",
    "load_pilot_plan",
    "main",
    "validate_cell_artifact",
]
