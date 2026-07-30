"""Frozen, sequential, resumable full-baseline experiment orchestration.

The module intentionally keeps plan/status construction data-free. Real dataset
access is reachable only through ``execute_registered_run`` after the frozen
configuration, Prompt 9 evidence, repository provenance, and resource preflight
have all authenticated.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import yaml

from credit_risk_fs.experiments._common import build_experiment_config
from credit_risk_fs.experiments.atomic_io import sha256_file, write_json_atomic
from credit_risk_fs.experiments.checkpointing import CheckpointManager
from credit_risk_fs.experiments.execution import (
    ExecutionOutcome,
    RegisteredRunRequest,
    execute_registered_run,
)
from credit_risk_fs.experiments.full_baseline_ram_bridge import (
    authenticate_full_baseline_ram_bridge,
)
from credit_risk_fs.experiments.full_baseline_runtime import (
    DEFAULT_RUNTIME_POLICY_PATH,
    FullBaselineRuntimePolicy,
    WorkloadClassification,
    classify_full_baseline_workload,
    load_full_baseline_runtime_policy,
)
from credit_risk_fs.experiments.full_baseline_timeout_resume import (
    DEFAULT_AUTHORIZATION_PATH,
    NOT_RESUMABLE,
    RESUMABLE_FROM_CELL_BOUNDARY,
    TimeoutResumeValidation,
    validate_timeout_resume_authorization,
)
from credit_risk_fs.experiments.heavy_selector_pilots import (
    cell_artifact_path,
    load_pilot_plan,
    validate_cell_artifact,
)
from credit_risk_fs.experiments.resource_monitor import (
    MANUAL_INTERRUPT,
    wait_for_inter_run_readiness,
)
from credit_risk_fs.experiments.resource_policy import (
    GIB,
    ResolvedExecutionPolicy,
    detect_hardware,
    load_execution_policy,
    resolve_execution_policy,
    run_preflight,
)
from credit_risk_fs.experiments.ram_control import (
    ResolvedRamControlPolicy,
    load_ram_control_policy,
)
from credit_risk_fs.experiments.research_logging import ResearchLogSession
from credit_risk_fs.experiments.result_paths import (
    create_run_directory,
    initialize_results_layout,
    planned_run_directory,
)
from credit_risk_fs.experiments.tracking import is_completed_run
from credit_risk_fs.selectors.lightweight.registry import get_method_descriptor
from credit_risk_fs.selectors.original_feature_adapter import (
    OriginalFeatureSelectorAdapter,
)


CONFIG_SCHEMA_VERSION = "full_baseline_config_v1"
STATUS_SCHEMA_VERSION = "full_baseline_status_v1"
DEFAULT_CONFIG_PATH = Path("configs/experiments/full_baseline_v1.yaml")
DATASET_ORDER = ("homecredit", "lendingclub_v2")
MODEL_ORDER = ("lr", "catboost")
METHOD_ORDER = (
    "full_features",
    "random_k",
    "iv_woe",
    "mrmr_mutual_information",
    "lasso_l1_logistic",
    "legacy_rf_relevance_corr",
    "catboost_shap",
    "boruta_random_forest",
    "rfe_catboost",
)
HEAVY_METHODS = frozenset(
    {"catboost_shap", "boruta_random_forest", "rfe_catboost"}
)
EXPECTED_CELL_COUNT = 36


class FullBaselineConfigurationError(ValueError):
    """Raised when the frozen matrix/configuration contract does not authenticate."""


class FullBaselineArtifactError(ValueError):
    """Raised when an existing run cannot be safely skipped or resumed."""


@dataclass(frozen=True, slots=True)
class FullBaselineCell:
    cell_index: int
    cell_id: str
    dataset: str
    dataset_label: str
    model: str
    method_id: str
    implementation_id: str
    seed: int
    feature_budget: int | None
    experiment_config_path: str
    selector_kwargs: dict[str, Any]
    wall_clock_limit_seconds: float


@dataclass(frozen=True, slots=True)
class FullBaselinePlan:
    repository_root: Path
    config_path: Path
    configuration_sha256: str
    configuration: dict[str, Any]
    results_root: Path
    log_path: Path
    policy_path: Path
    runtime_policy: FullBaselineRuntimePolicy
    cells: tuple[FullBaselineCell, ...]


@dataclass(frozen=True, slots=True)
class CellInspection:
    state: str
    valid_completed: bool
    resumable: bool
    reason: str


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    status: str
    completed_cells: int
    total_cells: int
    stop_cell_id: str | None = None
    stop_code: str | None = None


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FullBaselineConfigurationError(f"{field} must be a mapping")
    return dict(value)


def _safe_repository_file(root: Path, value: Any, field: str) -> Path:
    supplied = Path(str(value))
    path = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    if not path.is_relative_to(root):
        raise FullBaselineConfigurationError(f"{field} must stay inside the repository")
    if not path.is_file():
        raise FullBaselineConfigurationError(f"{field} is missing: {path}")
    return path


def _validate_protocol(configuration: Mapping[str, Any]) -> None:
    if configuration.get("configuration_status") != (
        "frozen_after_authenticated_dev_pilot_review"
    ):
        raise FullBaselineConfigurationError("full-baseline configuration is not frozen")
    protocol = _mapping(configuration.get("protocol"), "protocol")
    exact = {
        "configuration_frozen_before_full_baseline_execution": True,
        "datasets": list(DATASET_ORDER),
        "models": list(MODEL_ORDER),
        "methods": list(METHOD_ORDER),
        "matrix_nesting": ["method", "dataset", "model"],
        "expected_cell_count": EXPECTED_CELL_COUNT,
        "random_seeds": [42],
        "n_splits": 5,
        "cv_gap_groups": 1,
        "concurrent_cells": 1,
        "concurrent_folds": 1,
        "selector_fit_boundary": "fold_training_original_feature_candidates",
        "selector_numeric_encoder": "OriginalFeatureNumericEncoder",
        "final_model_preprocessing_order": "after_original_feature_selection",
        "oot_policy": "locked_final_evaluation_after_configuration_freeze",
        "configuration_adaptation_after_oot": "forbidden",
        "frozen_voting_protocol_changes": "forbidden",
    }
    for field, expected in exact.items():
        if protocol.get(field) != expected:
            raise FullBaselineConfigurationError(
                f"protocol.{field} must be exactly {expected!r}; "
                f"observed {protocol.get(field)!r}"
            )


def load_full_baseline_plan(
    repository_root: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    runtime_policy_path: str | Path = DEFAULT_RUNTIME_POLICY_PATH,
) -> FullBaselinePlan:
    """Load the exact 36-cell plan without importing or opening research data."""

    root = Path(repository_root).resolve()
    runtime_policy = load_full_baseline_runtime_policy(root, runtime_policy_path)
    path = _safe_repository_file(root, config_path, "full-baseline config path")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FullBaselineConfigurationError(f"config is unreadable: {path}") from exc
    configuration = _mapping(raw, "full-baseline configuration")
    if configuration.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise FullBaselineConfigurationError(
            f"unsupported config schema: {configuration.get('schema_version')!r}"
        )
    _validate_protocol(configuration)

    provenance = _mapping(configuration.get("provenance"), "provenance")
    pilot_path = _safe_repository_file(
        root, provenance.get("pilot_config_path"), "Prompt 9 pilot config"
    )
    if sha256_file(pilot_path) != provenance.get("pilot_config_sha256"):
        raise FullBaselineConfigurationError("Prompt 9 pilot config hash mismatch")
    if provenance.get("authenticated_pilot_cells") != 6:
        raise FullBaselineConfigurationError("exactly six Prompt 9 pilots are required")
    if provenance.get("pilot_oot_accessed") is not False or provenance.get(
        "pilot_oot_evaluated"
    ) is not False:
        raise FullBaselineConfigurationError("Prompt 9 provenance must be DEV-only")

    execution = _mapping(configuration.get("execution"), "execution")
    inventory = _mapping(configuration.get("dataset_inventory"), "dataset_inventory")
    selector_settings = _mapping(
        configuration.get("selector_settings"), "selector_settings"
    )
    budgets = _mapping(configuration.get("feature_budgets"), "feature_budgets")
    model_settings = _mapping(
        configuration.get("final_model_settings"), "final_model_settings"
    )
    limits = _mapping(
        execution.get("wall_clock_limits_seconds"), "wall_clock_limits_seconds"
    )
    if tuple(inventory) != DATASET_ORDER:
        raise FullBaselineConfigurationError("dataset inventory order changed")
    if tuple(selector_settings) != METHOD_ORDER:
        raise FullBaselineConfigurationError("selector settings order changed")
    if tuple(budgets) != MODEL_ORDER or any(int(budgets[m]) <= 0 for m in MODEL_ORDER):
        raise FullBaselineConfigurationError("model feature budgets are invalid")
    if tuple(model_settings) != MODEL_ORDER:
        raise FullBaselineConfigurationError("final model settings order changed")
    catboost_model_settings = _mapping(
        model_settings["catboost"], "CatBoost final model settings"
    )
    _mapping(model_settings["lr"], "LR final model settings")
    if catboost_model_settings.get("iterations") != 1500:
        raise FullBaselineConfigurationError("frozen CatBoost must use 1,500 iterations")

    cells: list[FullBaselineCell] = []
    for method_id in METHOD_ORDER:
        descriptor = get_method_descriptor(method_id)
        if descriptor.method_id != method_id:
            raise FullBaselineConfigurationError(f"registry identity changed: {method_id}")
        settings = _mapping(selector_settings[method_id], f"settings for {method_id}")
        # Construction here catches unknown/frozen keyword drift without fitting.
        kwargs_template = copy.deepcopy(dict(descriptor.default_kwargs))
        kwargs_template.update(copy.deepcopy(settings))
        limit_key = method_id if method_id in HEAVY_METHODS else "lightweight"
        wall_limit = float(limits.get(limit_key, 0))
        if wall_limit <= 0:
            raise FullBaselineConfigurationError(
                f"positive wall-clock limit required for {method_id}"
            )
        for dataset in DATASET_ORDER:
            data = _mapping(inventory[dataset], f"dataset inventory for {dataset}")
            experiment_path = _safe_repository_file(
                root,
                data.get("experiment_config_path"),
                f"experiment config for {dataset}",
            )
            for model in MODEL_ORDER:
                index = len(cells) + 1
                slug = method_id.replace("_", "-")
                run_id = f"fbv1-{index:03d}-{dataset}-{model}-{slug}-s42"
                kwargs = copy.deepcopy(kwargs_template)
                budget = None if descriptor.budget_kwarg is None else int(budgets[model])
                if descriptor.budget_kwarg is not None:
                    kwargs[descriptor.budget_kwarg] = budget
                cells.append(
                    FullBaselineCell(
                        cell_index=index,
                        cell_id=run_id,
                        dataset=dataset,
                        dataset_label=str(data.get("display_label", dataset)),
                        model=model,
                        method_id=method_id,
                        implementation_id=descriptor.implementation_id,
                        seed=42,
                        feature_budget=budget,
                        experiment_config_path=experiment_path.relative_to(root).as_posix(),
                        selector_kwargs=kwargs,
                        wall_clock_limit_seconds=wall_limit,
                    )
                )
    if len(cells) != EXPECTED_CELL_COUNT or len({cell.cell_id for cell in cells}) != len(cells):
        raise FullBaselineConfigurationError("matrix is not exactly 36 unique cells")

    results_value = Path(str(execution.get("results_root", "")))
    results_root = (
        results_value.resolve()
        if results_value.is_absolute()
        else (root / results_value).resolve()
    )
    if not results_root.is_relative_to(root):
        raise FullBaselineConfigurationError("results root must stay inside repository")
    return FullBaselinePlan(
        repository_root=root,
        config_path=path,
        configuration_sha256=sha256_file(path),
        configuration=configuration,
        results_root=results_root,
        log_path=Path(str(execution.get("log_path", "logs/runs.log"))),
        policy_path=Path(str(execution.get("policy_path", ""))),
        runtime_policy=runtime_policy,
        cells=tuple(cells),
    )


def authenticate_prompt_9_evidence(plan: FullBaselinePlan) -> dict[str, Any]:
    """Authenticate all six DEV pilots while refusing every OOT-bearing result."""

    provenance = _mapping(plan.configuration["provenance"], "provenance")
    pilot_plan = load_pilot_plan(
        plan.repository_root, plan.repository_root / provenance["pilot_config_path"]
    )
    if pilot_plan.configuration_sha256 != provenance["pilot_config_sha256"]:
        raise FullBaselineArtifactError("Prompt 9 plan hash differs from frozen provenance")
    records: list[dict[str, Any]] = []
    for pilot_cell in pilot_plan.cells:
        validation = validate_cell_artifact(pilot_plan, pilot_cell)
        if not validation.valid or validation.payload is None:
            raise FullBaselineArtifactError(
                f"Prompt 9 pilot {pilot_cell.cell_id} is invalid: {validation.reason}"
            )
        payload = validation.payload
        if (
            payload.get("oot_accessed") is not False
            or payload.get("oot_evaluated") is not False
            or payload.get("performance_evaluation_performed") is not False
        ):
            raise FullBaselineArtifactError(
                f"Prompt 9 pilot {pilot_cell.cell_id} violates the DEV-only boundary"
            )
        result = _mapping(payload.get("selector_result"), "selector_result")
        heavy = _mapping(result.get("heavy_metadata"), "heavy_metadata")
        records.append(
            {
                "cell_id": pilot_cell.cell_id,
                "dataset": pilot_cell.dataset,
                "method_id": pilot_cell.method_id,
                "artifact_path": cell_artifact_path(pilot_plan, pilot_cell)
                .relative_to(plan.repository_root)
                .as_posix(),
                "artifact_file_sha256": sha256_file(
                    cell_artifact_path(pilot_plan, pilot_cell)
                ),
                "artifact_authentication_sha256": payload[
                    "artifact_authentication_sha256"
                ],
                "runtime_seconds": payload.get("runtime_seconds"),
                "peak_process_tree_rss_bytes": payload.get(
                    "peak_process_tree_rss_bytes"
                ),
                "minimum_system_available_ram_bytes": payload.get(
                    "minimum_system_available_ram_bytes"
                ),
                "requested_budget": result.get("requested_budget"),
                "actual_selected_count": result.get("actual_selected_count"),
                "natural_selected_count": result.get("natural_selected_count"),
                "budget_status": result.get("budget_status"),
                "confirmed_count": heavy.get("confirmed_count"),
                "tentative_count": heavy.get("tentative_count"),
                "rejected_count": heavy.get("rejected_count"),
                "estimator_fit_count": heavy.get("estimator_fit_count"),
                "oot_accessed": False,
                "oot_evaluated": False,
            }
        )
    return {
        "status": "authenticated",
        "pilot_configuration_sha256": pilot_plan.configuration_sha256,
        "pilot_cell_count": len(records),
        "oot_accessed": False,
        "oot_evaluated": False,
        "cells": records,
    }


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def authenticate_repository(plan: FullBaselinePlan) -> dict[str, Any]:
    """Require a clean descendant of every declared Prompt 8/9 checkpoint."""

    head = _git(plan.repository_root, "rev-parse", "HEAD")
    dirty = _git(
        plan.repository_root, "status", "--porcelain", "--untracked-files=normal"
    )
    if dirty:
        raise RuntimeError("full-baseline execution requires a clean worktree")
    provenance = _mapping(plan.configuration["provenance"], "provenance")
    ancestors: dict[str, bool] = {}
    for field in (
        "prompt_8_baseline_commit",
        "prompt_9_pipeline_commit",
        "prompt_9_logging_fix_commit",
    ):
        commit = str(provenance[field])
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=plan.repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"declared provenance commit is not an ancestor: {commit}")
        ancestors[commit] = True
    return {
        "git_commit": head,
        "git_dirty": False,
        "ancestor_checks": ancestors,
        "full_baseline_configuration_sha256": plan.configuration_sha256,
    }


def run_directory(plan: FullBaselinePlan, cell: FullBaselineCell) -> Path:
    return planned_run_directory(
        plan.results_root, dataset=cell.dataset, run_id=cell.cell_id
    )


def _validate_manifest_artifacts(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise FullBaselineArtifactError("completed manifest has no artifact contract")
    for name, raw in artifacts.items():
        if not isinstance(raw, Mapping):
            raise FullBaselineArtifactError(f"invalid artifact contract entry: {name}")
        if not raw.get("applicable") or name == "manifest":
            continue
        if not raw.get("present"):
            raise FullBaselineArtifactError(f"applicable artifact is absent: {name}")
        relative = Path(str(raw.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise FullBaselineArtifactError(f"unsafe artifact path: {relative}")
        path = (run_dir / relative).resolve()
        if not path.is_relative_to(run_dir) or not path.is_file():
            raise FullBaselineArtifactError(f"artifact is missing: {relative}")
        if raw.get("sha256") and sha256_file(path) != raw["sha256"]:
            raise FullBaselineArtifactError(f"artifact hash mismatch: {relative}")


def workload_classification(
    plan: FullBaselinePlan, cell: FullBaselineCell
) -> WorkloadClassification:
    return classify_full_baseline_workload(cell, plan.runtime_policy)


def _inspect_cell_control(plan: FullBaselinePlan, cell: FullBaselineCell) -> CellInspection:
    """Inspect only run control/artifact files; no dataset path is opened."""

    path = run_directory(plan, cell)
    if not path.exists():
        return CellInspection("missing", False, False, "run directory is missing")
    lock = path / ".execution.lock"
    if lock.exists():
        return CellInspection("locked", False, False, "execution lock is present")
    manifest_path = path / "manifest.json"
    config_path = path / "config.json"
    checkpoint_path = path / "checkpoint.json"
    if not manifest_path.is_file() or not config_path.is_file() or not checkpoint_path.is_file():
        return CellInspection(
            "invalid_partial", False, False, "registration control files are incomplete"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return CellInspection("invalid", False, False, f"control JSON unreadable: {exc}")

    expected = {
        "run_id": cell.cell_id,
        "dataset": cell.dataset,
        "model": cell.model,
        "selector": cell.method_id,
    }
    observed = {
        "run_id": manifest.get("run_id"),
        "dataset": manifest.get("full_baseline_dataset"),
        "model": manifest.get("model"),
        "selector": manifest.get("selector"),
    }
    if observed != expected:
        return CellInspection("invalid", False, False, "run identity mismatch")
    if (
        manifest.get("full_baseline_configuration_sha256")
        != plan.configuration_sha256
        or config.get("full_baseline_configuration_sha256")
        != plan.configuration_sha256
    ):
        return CellInspection("invalid", False, False, "frozen configuration hash mismatch")

    manifest_status = str(manifest.get("status", "unknown"))
    if manifest_status == "completed" or (path / "_SUCCESS").exists():
        try:
            if manifest_status != "completed" or checkpoint.get("status") != "completed":
                raise FullBaselineArtifactError("terminal status mismatch")
            if not is_completed_run(path):
                raise FullBaselineArtifactError("required completed-run artifacts are missing")
            _validate_manifest_artifacts(path, manifest)
            for relative, metadata in checkpoint.get("finalized_artifacts", {}).items():
                artifact = (path / relative).resolve()
                if not artifact.is_relative_to(path) or not artifact.is_file():
                    raise FullBaselineArtifactError(
                        f"checkpoint artifact missing: {relative}"
                    )
                if sha256_file(artifact) != metadata.get("sha256"):
                    raise FullBaselineArtifactError(
                        f"checkpoint artifact hash mismatch: {relative}"
                    )
        except (OSError, ValueError, FullBaselineArtifactError) as exc:
            return CellInspection("invalid_completed", False, False, str(exc))
        return CellInspection(
            "completed", True, False, "authenticated completed run is immutable"
        )
    resumable = manifest_status in {
        "failed",
        "interrupted",
        "aborted_resource_limit",
        "running",
    } and checkpoint.get("status") != "completed"
    return CellInspection(
        manifest_status,
        False,
        resumable,
        "explicit checkpoint validation required" if resumable else "run is not resumable",
    )


def validate_cell_timeout_resume(
    plan: FullBaselinePlan,
    cell: FullBaselineCell,
    *,
    authorization_path: str | Path = DEFAULT_AUTHORIZATION_PATH,
    process_records: Sequence[Mapping[str, Any]] | None = None,
    repository_state: Mapping[str, Any] | None = None,
) -> TimeoutResumeValidation:
    earlier = {
        prior.cell_id: _inspect_cell_control(plan, prior).valid_completed
        for prior in plan.cells
        if prior.cell_index < cell.cell_index
    }
    return validate_timeout_resume_authorization(
        repository_root=plan.repository_root,
        run_directory=run_directory(plan, cell),
        cell={
            "cell_id": cell.cell_id,
            "dataset": cell.dataset,
            "model": cell.model,
            "method_id": cell.method_id,
            "seed": cell.seed,
        },
        full_baseline_configuration_sha256=plan.configuration_sha256,
        workload_classification=workload_classification(plan, cell).to_dict(),
        earlier_cells_authenticated=earlier,
        authorization_path=authorization_path,
        process_records=process_records,
        repository_state=repository_state,
    )


def _inspect_cell_with_validation(
    plan: FullBaselinePlan,
    cell: FullBaselineCell,
    *,
    timeout_validator: Callable[
        [FullBaselinePlan, FullBaselineCell], TimeoutResumeValidation
    ] = validate_cell_timeout_resume,
) -> tuple[CellInspection, TimeoutResumeValidation | None]:
    historical = _inspect_cell_control(plan, cell)
    if historical.state != "timed_out":
        return historical, None
    try:
        validation = timeout_validator(plan, cell)
    except Exception as exc:
        return (
            CellInspection(
                "timed_out",
                False,
                False,
                f"timeout resume validation error: {type(exc).__name__}: {exc}",
            ),
            None,
        )
    return (
        CellInspection(
            "timed_out",
            False,
            validation.resumable,
            (
                "explicitly authorized restart from cell boundary"
                if validation.resumable
                else "; ".join(validation.reasons)
            ),
        ),
        validation,
    )


def inspect_cell(
    plan: FullBaselinePlan,
    cell: FullBaselineCell,
    *,
    timeout_validator: Callable[
        [FullBaselinePlan, FullBaselineCell], TimeoutResumeValidation
    ] = validate_cell_timeout_resume,
) -> CellInspection:
    """Inspect controls only and authorize a timeout only after explicit validation."""

    return _inspect_cell_with_validation(
        plan, cell, timeout_validator=timeout_validator
    )[0]


def build_status_report(plan: FullBaselinePlan) -> dict[str, Any]:
    """Build a data-free matrix status report."""

    rows: list[dict[str, Any]] = []
    first_pending: str | None = None
    for cell in plan.cells:
        inspection, timeout_validation = _inspect_cell_with_validation(plan, cell)
        classification = workload_classification(plan, cell)
        if first_pending is None and not inspection.valid_completed:
            first_pending = cell.cell_id
        rows.append(
            {
                "cell_index": cell.cell_index,
                "cell_id": cell.cell_id,
                "dataset": cell.dataset,
                "model": cell.model,
                "method_id": cell.method_id,
                "feature_budget": cell.feature_budget,
                "state": inspection.state,
                "valid_completed": inspection.valid_completed,
                "resumable": inspection.resumable,
                "reason": inspection.reason,
                "resume_validation_decision": (
                    timeout_validation.decision
                    if timeout_validation is not None
                    else None
                ),
                "restart_boundary": (
                    timeout_validation.intended_restart_boundary
                    if timeout_validation is not None
                    else None
                ),
                "partial_artifacts_exist": bool(
                    timeout_validation.partial_artifacts
                    if timeout_validation is not None
                    else ()
                ),
                "workload_classification": classification.to_dict(),
            }
        )
    completed = sum(bool(row["valid_completed"]) for row in rows)
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "configuration_status": plan.configuration["configuration_status"],
        "configuration_sha256": plan.configuration_sha256,
        "runtime_policy_sha256": plan.runtime_policy.source_sha256,
        "matrix_cell_count": len(rows),
        "completed_authenticated": completed,
        "remaining": len(rows) - completed,
        "next_cell": first_pending,
        "oot_accessed_by_status": False,
        "cells": rows,
    }


def format_status_report(report: Mapping[str, Any]) -> str:
    lines = [
        "Full baseline v1 | frozen 36-cell matrix",
        (
            f"Authenticated completed: {report['completed_authenticated']}/"
            f"{report['matrix_cell_count']} | Remaining: {report['remaining']}"
        ),
        f"Current/next cell: {report['next_cell'] or 'none (all authenticated complete)'}",
        "Status inspection accessed OOT: no",
    ]
    for row in report["cells"]:
        lines.append(
            f"{int(row['cell_index']):02d} | {row['dataset']} | {row['model']} | "
            f"{row['method_id']} | {row['state']} | "
            f"cost={row['workload_classification']['effective_cost_class']} | "
            f"timeout={int(row['workload_classification']['effective_wall_clock_limit_seconds'])}s"
        )
    return "\n".join(lines)


def build_resume_plan_report(
    plan: FullBaselinePlan,
    *,
    timeout_validator: Callable[
        [FullBaselinePlan, FullBaselineCell], TimeoutResumeValidation
    ] = validate_cell_timeout_resume,
) -> dict[str, Any]:
    """Build an execution plan from control metadata without opening a dataset."""

    head = _git(plan.repository_root, "rev-parse", "HEAD")
    branch = _git(plan.repository_root, "branch", "--show-current")
    dirty = bool(
        _git(
            plan.repository_root,
            "status",
            "--porcelain",
            "--untracked-files=normal",
        )
    )
    actions: list[dict[str, Any]] = []
    skipped: list[str] = []
    first_execute: str | None = None
    first_validation: TimeoutResumeValidation | None = None
    for cell in plan.cells:
        inspection, validation = _inspect_cell_with_validation(
            plan, cell, timeout_validator=timeout_validator
        )
        classification = workload_classification(plan, cell)
        if inspection.valid_completed:
            action = "SKIP"
            skipped.append(cell.cell_id)
        elif first_execute is None:
            if inspection.state == "timed_out" and inspection.resumable:
                action = "RESTART"
                first_execute = cell.cell_id
                first_validation = validation
            elif inspection.state == "missing":
                action = "EXECUTE"
                first_execute = cell.cell_id
            else:
                action = "BLOCKED"
                first_execute = cell.cell_id
                first_validation = validation
        else:
            action = "PENDING"
        actions.append(
            {
                "cell_index": cell.cell_index,
                "cell_id": cell.cell_id,
                "action": action,
                "historical_state": inspection.state,
                "reason": inspection.reason,
                "resume_validation_decision": (
                    validation.decision if validation is not None else None
                ),
                "restart_boundary": (
                    validation.intended_restart_boundary
                    if validation is not None
                    else None
                ),
                "partial_artifacts_exist": bool(
                    validation.partial_artifacts if validation is not None else ()
                ),
                "workload_classification": classification.to_dict(),
            }
        )
    return {
        "schema_version": "full_baseline_resume_plan_v1",
        "run_identity": "full_baseline_v1",
        "repository_commit": head,
        "repository_branch": branch,
        "repository_dirty": dirty,
        "configuration_sha256": plan.configuration_sha256,
        "runtime_policy_sha256": plan.runtime_policy.source_sha256,
        "completed_authenticated": len(skipped),
        "skipped_cell_ids": skipped,
        "earliest_incomplete_cell": first_execute,
        "first_cell_to_execute": first_execute,
        "historical_terminal_state": (
            first_validation.historical_terminal_state
            if first_validation is not None
            else None
        ),
        "historical_stop_reason": (
            first_validation.historical_stop_reason
            if first_validation is not None
            else None
        ),
        "resume_validation_decision": (
            first_validation.decision
            if first_validation is not None
            else NOT_RESUMABLE
            if first_execute is not None
            and next(
                row for row in actions if row["cell_id"] == first_execute
            )["action"]
            == "BLOCKED"
            else None
        ),
        "restart_boundary": (
            first_validation.intended_restart_boundary
            if first_validation is not None
            else None
        ),
        "partial_artifacts_exist": bool(
            first_validation.partial_artifacts
            if first_validation is not None
            else ()
        ),
        "active_processes": list(
            first_validation.active_processes
            if first_validation is not None
            else ()
        ),
        "lock_paths": list(
            first_validation.lock_paths if first_validation is not None else ()
        ),
        "actions": actions,
        "oot_accessed_by_plan": False,
        "worker_started_by_plan": False,
    }


def format_resume_plan_report(report: Mapping[str, Any]) -> str:
    first_id = report.get("first_cell_to_execute")
    first = next(
        (row for row in report["actions"] if row["cell_id"] == first_id), None
    )
    lines = [
        "Full baseline v1 | safe resume plan",
        (
            f"Repository: {report['repository_branch']}@{report['repository_commit']} | "
            f"clean={'yes' if not report['repository_dirty'] else 'no'}"
        ),
        f"Configuration SHA-256: {report['configuration_sha256']}",
        f"Authenticated completed: {report['completed_authenticated']}/36",
        f"Earliest incomplete: {report['earliest_incomplete_cell']}",
        f"Resume decision: {report['resume_validation_decision']}",
        f"Restart boundary: {report['restart_boundary']}",
        (
            "Active worker: no | Execution lock: no"
            if not report["active_processes"] and not report["lock_paths"]
            else "Active worker or execution lock blocks resume"
        ),
    ]
    if first is not None:
        profile = first["workload_classification"]
        lines.extend(
            [
                (
                    f"Workload: selector={profile['selector_cost_class']} | "
                    f"final_model={profile['final_model_cost_class']} | "
                    f"dataset={profile['dataset_cost_class']} | "
                    f"effective={profile['effective_cost_class']}"
                ),
                (
                    "Effective active-computation wall-clock limit: "
                    f"{int(profile['effective_wall_clock_limit_seconds'])}s "
                    f"({profile['effective_wall_clock_limit_seconds'] / 3600:g}h)"
                ),
                f"Partial attempt artifacts preserved: {'yes' if first['partial_artifacts_exist'] else 'no'}",
            ]
        )
    lines.append("Plan:")
    for row in report["actions"]:
        if row["action"] not in {"SKIP", "RESTART", "EXECUTE", "BLOCKED"}:
            continue
        detail = (
            "authenticated complete"
            if row["action"] == "SKIP"
            else "resumable from authenticated cell boundary"
            if row["action"] == "RESTART"
            else row["reason"]
        )
        lines.append(
            f"{int(row['cell_index']):03d} {row['action']} — {detail}"
        )
    lines.extend(
        [
            "Continuation inside a fold or CatBoost fit: no",
            "Plan inspection accessed OOT: no",
            "Plan inspection started a worker: no",
        ]
    )
    return "\n".join(lines)


def _resolve_policy_and_preflight(
    plan: FullBaselinePlan,
) -> tuple[ResolvedExecutionPolicy, dict[str, Any]]:
    plan.results_root.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.gettempdir()).resolve()
    configured = load_execution_policy(plan.repository_root, plan.policy_path)
    capacity = detect_hardware(plan.results_root, temp_root)
    resolved = resolve_execution_policy(configured, capacity)
    ram_control = load_ram_control_policy(
        plan.repository_root,
        total_physical_ram_bytes=int(capacity.total_ram_gb * GIB),
    )
    report = run_preflight(
        repository_root=plan.repository_root,
        config_path=plan.policy_path,
        results_root=plan.results_root,
        temp_root=temp_root,
        requested_accelerator="cpu",
        capacity=capacity,
        ram_control_policy=ram_control,
    )
    parallel = resolved.parallelism
    if (
        parallel.concurrent_experiment_runs != 1
        or parallel.concurrent_folds != 1
        or parallel.data_loader_workers != 0
        or parallel.allow_nested_parallelism
    ):
        raise RuntimeError("full baseline requires strictly sequential execution")
    if report.get("status") != "pass":
        raise RuntimeError(
            f"full-baseline preflight rejected: {report.get('blocking_reasons', [])}"
        )
    return resolved, report


def _load_dataset_configuration(plan: FullBaselinePlan, cell: FullBaselineCell) -> dict[str, Any]:
    path = plan.repository_root / cell.experiment_config_path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = _mapping(raw, f"dataset config for {cell.dataset}")
    if config.get("dataset_name") != cell.dataset:
        raise FullBaselineConfigurationError(
            f"dataset config identity mismatch for {cell.dataset}"
        )
    return config


def _effective_configuration(
    plan: FullBaselinePlan,
    cell: FullBaselineCell,
    policy: ResolvedExecutionPolicy,
) -> dict[str, Any]:
    config = _load_dataset_configuration(plan, cell)
    config["model_selector"] = cell.model
    config["random_seed"] = cell.seed
    config["feature_budgets"] = copy.deepcopy(plan.configuration["feature_budgets"])
    config["model_params"] = copy.deepcopy(plan.configuration["final_model_settings"])
    config["results_dir"] = str(plan.results_root)
    config["matrix_run"] = {
        "experiment_type": "full_baseline",
        "experiment_name": cell.method_id,
        "selector": cell.method_id,
        "model": cell.model,
        "cell_id": cell.cell_id,
    }
    config["selector_configuration"] = copy.deepcopy(cell.selector_kwargs)
    config["selector_fit_boundary"] = plan.configuration["protocol"][
        "selector_fit_boundary"
    ]
    config["selector_numeric_encoder"] = plan.configuration["protocol"][
        "selector_numeric_encoder"
    ]
    config["full_baseline_configuration_sha256"] = plan.configuration_sha256
    config["full_baseline_configuration_path"] = plan.config_path.relative_to(
        plan.repository_root
    ).as_posix()
    config["full_baseline_cell"] = asdict(cell)
    config["oot_policy"] = plan.configuration["protocol"]["oot_policy"]
    config["configuration_adaptation_after_oot"] = "forbidden"
    config["_resolved_execution_policy"] = policy.to_dict()
    config["csv_chunk_rows"] = int(plan.configuration["execution"]["csv_chunk_rows"])
    return config


def _experiment_configuration(
    plan: FullBaselinePlan,
    cell: FullBaselineCell,
    run_dir: Path,
    effective: dict[str, Any],
):
    descriptor = get_method_descriptor(cell.method_id)
    selector_cls = descriptor.load()
    args = SimpleNamespace(
        project_config=effective,
        model=cell.model,
        data_dir=effective["data_dir"],
        description_path=effective["description_path"],
        n_splits=int(effective["n_splits"]),
        dev_start_day=int(effective["dev_start_day"]),
        oot_start_day=int(effective["oot_start_day"]),
        oot_end_day=int(effective["oot_end_day"]),
        cv_gap_groups=int(effective["cv_gap_groups"]),
        random_seed=cell.seed,
    )
    return build_experiment_config(
        args=args,
        experiments_dir=run_dir,
        experiment_name=cell.method_id,
        selector_name=cell.method_id,
        selector_cls=OriginalFeatureSelectorAdapter,
        selector_kwargs={
            "selector_cls": selector_cls,
            "selector_kwargs": copy.deepcopy(cell.selector_kwargs),
            "random_state": cell.seed,
        },
        experiment_output_dir=run_dir,
    )


def _execute_real_cell(
    plan: FullBaselinePlan,
    cell: FullBaselineCell,
    policy: ResolvedExecutionPolicy,
    preflight: dict[str, Any],
    resume: bool,
) -> ExecutionOutcome:
    run_dir = run_directory(plan, cell)
    if not resume:
        created = create_run_directory(
            plan.results_root,
            dataset=cell.dataset,
            run_id=cell.cell_id,
            collision_policy="error",
        )
        if created != run_dir:
            raise RuntimeError("fixed full-baseline run path changed")
    effective = _effective_configuration(plan, cell, policy)
    experiment_config = _experiment_configuration(
        plan, cell, run_dir, effective
    )
    classification = workload_classification(plan, cell)
    checkpoint_identity_override = None
    resume_metadata = None
    resume_authorization = None
    if resume:
        checkpoint_payload = CheckpointManager(run_dir).load()
        checkpoint_commit = checkpoint_payload.get("identity", {}).get("git_commit")
        live_commit = preflight.get("git_commit")
        if checkpoint_payload.get("stop_code") == "wall_clock_limit":
            validation = validate_cell_timeout_resume(plan, cell)
            if not validation.resumable:
                raise FullBaselineArtifactError(
                    "controlled timeout is not authorized for resume: "
                    + "; ".join(validation.reasons)
                )
            resume_authorization = validation.to_dict()
            resume_metadata = {
                "timeout_recovery_authorization": validation.to_dict(),
                "workload_classification": classification.to_dict(),
                "historical_terminal_state_preserved": True,
            }
            checkpoint_identity_override = dict(checkpoint_payload["identity"])
        elif checkpoint_commit != live_commit:
            resume_metadata = authenticate_full_baseline_ram_bridge(
                plan.repository_root,
                run_id=cell.cell_id,
                checkpoint=checkpoint_payload,
                full_baseline_configuration_sha256=plan.configuration_sha256,
            )
            checkpoint_identity_override = dict(checkpoint_payload["identity"])
    return execute_registered_run(
        RegisteredRunRequest(
            repository_root=plan.repository_root,
            results_root=plan.results_root,
            run_directory=run_dir,
            dataset=cell.dataset,
            selector=cell.method_id,
            model=cell.model,
            experiment_type="full_baseline",
            split_protocol=str(plan.configuration["protocol"]["split_protocol"]),
            seed=cell.seed,
            effective_config=effective,
            experiment_config=experiment_config,
            preflight_report=preflight,
            resolved_policy=policy,
            resume=resume,
            manifest_metadata={
                "full_baseline_configuration_sha256": plan.configuration_sha256,
                "full_baseline_cell_index": cell.cell_index,
                "full_baseline_dataset": cell.dataset,
                "selector_implementation_id": cell.implementation_id,
                "feature_budget": cell.feature_budget,
                "oot_policy": plan.configuration["protocol"]["oot_policy"],
                "configuration_adaptation_after_oot": "forbidden",
                "workload_classification": classification.to_dict(),
                "effective_wall_clock_limit_seconds": (
                    classification.effective_wall_clock_limit_seconds
                ),
            },
            max_wall_clock_seconds=(
                classification.effective_wall_clock_limit_seconds
            ),
            checkpoint_identity_override=checkpoint_identity_override,
            resume_metadata=resume_metadata,
            resume_authorization=resume_authorization,
            ram_control_policy=ResolvedRamControlPolicy(
                **dict(preflight["ram_control_policy"])
            ),
        )
    )


def _write_matrix_progress(plan: FullBaselinePlan) -> None:
    report = build_status_report(plan)
    write_json_atomic(plan.results_root / "matrix_status.json", report)


def execute_full_baseline(
    plan: FullBaselinePlan,
    *,
    require_clean_repository: bool = True,
    authenticate_pilots: bool = True,
    policy_preflight: Callable[
        [FullBaselinePlan], tuple[ResolvedExecutionPolicy, dict[str, Any]]
    ] = _resolve_policy_and_preflight,
    inspector: Callable[[FullBaselinePlan, FullBaselineCell], CellInspection] = inspect_cell,
    executor: Callable[
        [FullBaselinePlan, FullBaselineCell, ResolvedExecutionPolicy, dict[str, Any], bool],
        Any,
    ] = _execute_real_cell,
    readiness_checker: Callable[..., Any] = wait_for_inter_run_readiness,
    progress_writer: Callable[[FullBaselinePlan], None] = _write_matrix_progress,
) -> PipelineOutcome:
    """Run earliest incomplete cells sequentially and stop on the first failure."""

    if require_clean_repository:
        authenticate_repository(plan)
    if authenticate_pilots:
        authenticate_prompt_9_evidence(plan)
    policy, preflight = policy_preflight(plan)
    ram_control = ResolvedRamControlPolicy(
        **dict(preflight["ram_control_policy"])
    ) if preflight.get("ram_control_policy") else None
    initialize_results_layout(plan.repository_root, results_root=plan.results_root)
    completed = 0
    for cell in plan.cells:
        state = inspector(plan, cell)
        if state.valid_completed:
            completed += 1
            continue
        if state.state in {"invalid", "invalid_partial", "invalid_completed", "locked"}:
            raise FullBaselineArtifactError(
                f"cannot safely continue at {cell.cell_id}: {state.reason}"
            )
        if run_directory(plan, cell).exists() and not state.resumable:
            raise FullBaselineArtifactError(
                f"existing run is neither authenticated nor resumable: {cell.cell_id}"
            )
        readiness = readiness_checker(
            policy=policy,
            results_root=plan.results_root,
            temp_root=preflight["temporary_root"],
            ram_control_policy=ram_control,
        )
        if not readiness.ready:
            progress_writer(plan)
            return PipelineOutcome(
                "blocked_resource_readiness",
                completed,
                len(plan.cells),
                cell.cell_id,
                readiness.stop_code,
            )
        outcome = executor(plan, cell, policy, preflight, state.resumable)
        status = str(getattr(outcome, "status", "failed"))
        if status != "completed":
            progress_writer(plan)
            return PipelineOutcome(
                status,
                completed,
                len(plan.cells),
                cell.cell_id,
                getattr(outcome, "stop_code", None),
            )
        post = inspector(plan, cell)
        if not post.valid_completed:
            raise FullBaselineArtifactError(
                f"completed cell failed artifact authentication: {cell.cell_id}: {post.reason}"
            )
        completed += 1
        progress_writer(plan)
    return PipelineOutcome("completed", completed, len(plan.cells))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or inspect the frozen 36-cell full selector baseline."
    )
    parser.add_argument(
        "--repository-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--runtime-policy", type=Path, default=DEFAULT_RUNTIME_POLICY_PATH
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Read only frozen config and run artifacts; never load a dataset.",
    )
    parser.add_argument(
        "--plan-resume",
        action="store_true",
        help="Validate and print the exact data-free resume plan; start no worker.",
    )
    parser.add_argument(
        "--audit-pilots",
        action="store_true",
        help="Authenticate the six Prompt 9 DEV artifacts; never access OOT.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repository_root.resolve()
    plan = load_full_baseline_plan(root, args.config, args.runtime_policy)
    if args.audit_pilots:
        print(json.dumps(authenticate_prompt_9_evidence(plan), indent=2, sort_keys=True))
        return 0
    if args.status:
        print(format_status_report(build_status_report(plan)))
        return 0
    if args.plan_resume:
        print(format_resume_plan_report(build_resume_plan_report(plan)))
        return 0

    command_arguments = list(sys.argv[1:] if argv is None else argv)
    with ResearchLogSession(
        plan.log_path,
        repository_root=root,
        command_arguments=command_arguments,
    ) as session:
        try:
            outcome = execute_full_baseline(plan)
        except KeyboardInterrupt:
            session.finish(
                "session_interrupted",
                level="ERROR",
                message="Full baseline interrupted manually; checkpoint retained",
                exception_class="KeyboardInterrupt",
                stop_code="manual_interrupt",
            )
            return 130
        except BaseException as exc:
            session.finish(
                "session_failed",
                level="ERROR",
                message=f"Full baseline failed: {type(exc).__name__}: {exc}",
                exception_class=type(exc).__name__,
                traceback=traceback.format_exc(),
            )
            return 1
        if outcome.status == "completed":
            session.finish(
                "session_completed",
                message="All 36 full-baseline cells completed and authenticated",
                completed_cells=outcome.completed_cells,
                total_cells=outcome.total_cells,
            )
            return 0
        if outcome.status == "interrupted" and outcome.stop_code == MANUAL_INTERRUPT:
            session.finish(
                "session_interrupted",
                level="ERROR",
                message="Full baseline interrupted manually; checkpoint retained",
                exception_class="KeyboardInterrupt",
                stop_code=MANUAL_INTERRUPT,
                run_id=outcome.stop_cell_id,
            )
            return 130
        session.finish(
            "session_controlled_stop",
            level="ERROR",
            message=(
                f"Full baseline stopped at {outcome.stop_cell_id}: "
                f"{outcome.status} ({outcome.stop_code})"
            ),
            stop_code=outcome.stop_code,
            status=outcome.status,
        )
        return 2


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "DATASET_ORDER",
    "DEFAULT_CONFIG_PATH",
    "EXPECTED_CELL_COUNT",
    "FullBaselineArtifactError",
    "FullBaselineCell",
    "FullBaselineConfigurationError",
    "FullBaselinePlan",
    "METHOD_ORDER",
    "MODEL_ORDER",
    "PipelineOutcome",
    "authenticate_prompt_9_evidence",
    "build_resume_plan_report",
    "build_status_report",
    "execute_full_baseline",
    "format_resume_plan_report",
    "format_status_report",
    "inspect_cell",
    "load_full_baseline_plan",
    "run_directory",
    "validate_cell_timeout_resume",
    "workload_classification",
]
