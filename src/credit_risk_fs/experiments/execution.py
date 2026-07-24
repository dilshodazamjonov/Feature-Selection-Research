"""Canonical registered-run lifecycle built on the existing runner contract."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from credit_risk_fs.experiments.atomic_io import inspect_artifact, sha256_file, write_json_atomic
from credit_risk_fs.experiments.checkpointing import CheckpointManager, ResumeValidationError
from credit_risk_fs.experiments.config import compute_config_hash
from credit_risk_fs.experiments.resource_monitor import SupervisorResult, supervise_worker
from credit_risk_fs.experiments.research_logging import emit_research_event, logged_stage
from credit_risk_fs.experiments.resource_policy import ResolvedExecutionPolicy
from credit_risk_fs.experiments.result_paths import (
    append_run_index_row,
    repository_relative_path,
    update_run_index_row,
)
from credit_risk_fs.experiments.tracking import (
    build_data_version,
    build_run_manifest,
    mark_completed,
    materialize_standard_artifacts,
    utc_timestamp,
    write_resource_usage,
    write_run_manifest,
)
from credit_risk_fs.pipelines.common import ExperimentConfig


@dataclass(frozen=True, slots=True)
class RegisteredRunRequest:
    repository_root: Path
    results_root: Path
    run_directory: Path
    dataset: str
    selector: str
    model: str
    experiment_type: str
    split_protocol: str
    seed: int
    effective_config: dict[str, Any]
    experiment_config: ExperimentConfig
    preflight_report: dict[str, Any]
    resolved_policy: ResolvedExecutionPolicy
    resume: bool = False
    worker_target: str = "credit_risk_fs.experiments.execution:experiment_worker"
    worker_kwargs: dict[str, Any] | None = None
    manifest_metadata: dict[str, Any] | None = None
    artifact_applicability: dict[str, bool] | None = None
    protocol_path: str | None = None
    row_contract_path: str | None = None
    merge_default_worker_kwargs: bool = False
    defer_terminal_success: bool = False
    deferred_success_status: str = "dev_complete"
    checkpoint_identity_override: dict[str, Any] | None = None
    resume_metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    run_id: str
    run_directory: Path
    status: str
    stop_code: str | None
    supervisor: SupervisorResult
    manifest: dict[str, Any]


def _checkpoint_identity(request: RegisteredRunRequest) -> dict[str, Any]:
    protocol_path = request.repository_root / (
        request.protocol_path or "configs/protocols/credit_scoring_extension_v1.yaml"
    )
    row_contract = request.repository_root / (
        request.row_contract_path or "configs/protocols/row_alignment_contract_v1.json"
    )
    protocol_version = "unknown"
    if protocol_path.is_file():
        for line in protocol_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("version:"):
                protocol_version = line.split(":", 1)[1].strip().strip("'\"")
                break
    data_fingerprint = build_data_version(request.experiment_config.data_dir)
    identity = {
        "run_id": request.run_directory.name,
        "dataset": request.dataset,
        "selector": request.selector,
        "model": request.model,
        "split_protocol": request.split_protocol,
        "seed": int(request.seed),
        "budgets": request.effective_config.get("feature_budgets", {}),
        "resolved_config_hash": compute_config_hash(request.effective_config),
        "protocol_version": protocol_version,
        "protocol_hash": sha256_file(protocol_path) if protocol_path.is_file() else "unavailable",
        "data_hash": compute_config_hash(data_fingerprint),
        "row_alignment_hash": sha256_file(row_contract) if row_contract.is_file() else "unavailable",
        "git_commit": request.preflight_report.get("git_commit", "unknown"),
        "git_dirty": request.preflight_report.get("git_dirty"),
    }
    if request.checkpoint_identity_override is None:
        return identity
    if not request.resume:
        raise ValueError("checkpoint identity overrides are resume-only")
    override = dict(request.checkpoint_identity_override)
    for field, observed in identity.items():
        if field == "git_commit":
            continue
        if override.get(field) != observed:
            raise ResumeValidationError(
                f"compatibility_override_mismatch_{field}",
                f"compatibility identity differs from live resume input: {field}",
            )
    return override


def _checkpoint_artifacts(run_directory: Path, stage: str) -> list[Any]:
    stage_paths = {
        "selection_completed": [run_directory / "features/final_selected_features.csv"],
        "model_fit_completed": [run_directory / "models/final_model_metadata.json"],
        "dev_prediction_completed": [run_directory / "results/dev_predictions.csv"],
        "oot_prediction_completed": [run_directory / "results/oot_predictions.csv"],
        "evaluation_completed": [
            run_directory / "results/prediction_metrics.csv",
            run_directory / "results/oot_test_results.csv",
        ],
    }
    artifacts = []
    for path in stage_paths.get(stage, []):
        if not path.is_file():
            continue
        artifacts.append(
            inspect_artifact(
                path,
                ordered_row_identity_column=(
                    "stable_row_id"
                    if path.name in {"dev_predictions.csv", "oot_predictions.csv"}
                    else None
                ),
            )
        )
    return artifacts


def experiment_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
    experiment_config: ExperimentConfig,
    checkpoint_identity: Mapping[str, Any],
    run_directory: str,
) -> dict[str, Any]:
    """Spawn-safe worker target; scientific imports occur after thread limits are inherited."""

    from credit_risk_fs.pipelines.common import prepare_modeling_data, run_experiment
    from credit_risk_fs.utils.logging import run_log_context

    run_dir = Path(run_directory)
    checkpoint = CheckpointManager(run_dir)
    def report_stage(stage: str, fold_id: str | int | None = None) -> None:
        stage_queue.put({"stage": stage, "fold_id": fold_id})

    experiment_config.stage_callback = report_stage
    experiment_config.cooperative_stop_event = stop_event
    if stop_event.is_set():
        raise RuntimeError("cooperative stop requested before data loading")
    prepared = prepare_modeling_data(experiment_config)
    checkpoint.transition("data_validated")
    stage_queue.put({"stage": "data_validated", "fold_id": None})
    if stop_event.is_set():
        raise RuntimeError("cooperative stop requested after data validation")
    with run_log_context(run_dir / "run.log"):
        completed = run_experiment(experiment_config, prepared_data=prepared)
    for stage in (
        "selection_completed",
        "model_fit_completed",
        "dev_prediction_completed",
        "oot_prediction_completed",
        "evaluation_completed",
    ):
        artifacts = _checkpoint_artifacts(run_dir, stage)
        if artifacts:
            checkpoint.transition(stage, artifacts=artifacts)
            stage_queue.put({"stage": stage, "fold_id": None})
    return {"summary": completed.summary, "exp_dir": str(completed.exp_dir)}


def _resource_payload(
    supervisor: SupervisorResult,
    request: RegisteredRunRequest,
) -> dict[str, Any]:
    samples = [
        {
            "elapsed_seconds": item.elapsed_seconds,
            "worker_pid": item.worker_pid,
            "child_pids": list(item.child_pids),
            "process_tree_rss_bytes": item.process_tree_rss_bytes,
            "system_available_ram_bytes": item.system_available_ram_bytes,
            "process_gpu_bytes": item.process_gpu_bytes,
            "results_free_disk_bytes": item.results_free_disk_bytes,
            "temp_free_disk_bytes": item.temp_free_disk_bytes,
            "process_tree_cpu_percent": item.process_tree_cpu_percent,
            "process_tree_cpu_seconds": item.process_tree_cpu_seconds,
            "stage": item.stage,
            "fold_id": item.fold_id,
        }
        for item in supervisor.samples
    ]
    return {
        "total_runtime_seconds": samples[-1]["elapsed_seconds"] if samples else None,
        "peak_process_tree_rss_bytes": supervisor.peak_process_tree_rss_bytes,
        "peak_process_gpu_bytes": supervisor.peak_process_gpu_bytes,
        "minimum_system_available_ram_bytes": supervisor.minimum_system_available_ram_bytes,
        "minimum_results_free_disk_bytes": supervisor.minimum_results_free_disk_bytes,
        "minimum_temp_free_disk_bytes": supervisor.minimum_temp_free_disk_bytes,
        "worker_exit_code": supervisor.worker_exit_code,
        "stop_code": supervisor.stop_code,
        "primary_stop_code": supervisor.primary_stop_code,
        "secondary_events": list(supervisor.secondary_events),
        "stop_lifecycle": list(supervisor.stop_lifecycle),
        "termination_condition": supervisor.termination_condition,
        "graceful_stop_completed": supervisor.graceful_stop_completed,
        "shutdown_elapsed_seconds": supervisor.shutdown_elapsed_seconds,
        "configured_shutdown_upper_bound_seconds": (
            request.resolved_policy.monitoring.graceful_stop_timeout_seconds
            + 2 * request.resolved_policy.monitoring.forced_stop_timeout_seconds
            + 1.0
        ),
        "status": supervisor.status,
        "samples": samples,
        "warnings": list(supervisor.warnings),
        "process_ownership": {
            "run_association": supervisor.run_association,
            "owned_processes": list(supervisor.owned_processes),
            "survivor_processes": list(supervisor.survivor_processes),
        },
        "cleanup_evidence": {
            "child_cleanup_confirmed": supervisor.child_cleanup_confirmed,
            "queue_cleanup_confirmed": supervisor.queue_cleanup_confirmed,
            "parent_rss_before_bytes": supervisor.parent_rss_before_bytes,
            "parent_rss_after_bytes": supervisor.parent_rss_after_bytes,
            "system_available_ram_after_bytes": (
                supervisor.system_available_ram_after_bytes
            ),
            "compact_result_payload_limit_bytes": 1024 * 1024,
        },
        "resolved_parallelism": request.resolved_policy.to_dict()["parallelism"],
        "policy_version": request.resolved_policy.profile_name,
        "preflight_status": request.preflight_report.get("status"),
        "checkpoint_path": "checkpoint.json",
        "resumability_status": (
            "phase_complete_explicit_resume_required"
            if supervisor.status == "completed" and request.defer_terminal_success
            else "completed_immutable"
            if supervisor.status == "completed"
            else "explicit_resume_validation_required"
        ),
    }


def execute_registered_run(request: RegisteredRunRequest) -> ExecutionOutcome:
    """Register, supervise, and finalize one run for both single and matrix entry points."""

    run_dir = request.run_directory.resolve()
    run_id = run_dir.name
    if request.preflight_report.get("status") != "pass":
        raise ValueError(
            "preflight_rejected: "
            + ", ".join(request.preflight_report.get("blocking_reasons", []))
        )
    identity = _checkpoint_identity(request)
    checkpoint = CheckpointManager(run_dir)
    config_path = run_dir / "config.json"
    preflight_path = run_dir / "preflight.json"
    manifest_path = run_dir / "manifest.json"
    log_context = {
        "run_id": run_id,
        "dataset": request.dataset,
        "model": request.model,
        "seed": request.seed,
        "selector": request.selector,
    }
    emit_research_event(
        "run_execution_started",
        message="Registered research run execution started",
        priority=True,
        resume=request.resume,
        run_directory=repository_relative_path(run_dir, request.repository_root),
        **log_context,
    )

    if request.resume:
        with logged_stage(
            "checkpoint_resume_validation",
            message="Validate checkpoint identity and finalized artifact hashes",
            component="checkpoint_manager",
            **log_context,
        ):
            checkpoint.validate_resume(identity)
        checkpoint.begin_resume_attempt()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "completed":
            raise ResumeValidationError("completed_run_immutable", "completed runs cannot be resumed")
        manifest["status"] = "running"
        manifest["resumed_at_utc"] = utc_timestamp()
        if request.resume_metadata:
            manifest["resume_mechanics_provenance"] = dict(request.resume_metadata)
        write_run_manifest(run_dir, manifest)
        update_run_index_row(
            request.results_root,
            run_id,
            {"status": "running", "notes": "explicit validated resume"},
        )
        emit_research_event(
            "run_resume_authenticated",
            message="Existing run checkpoint authenticated for resume",
            priority=True,
            phase=str((request.worker_kwargs or {}).get("phase", "")).upper(),
            checkpoint_path=repository_relative_path(
                checkpoint.path, request.repository_root
            ),
            completed_fold_ids=checkpoint.load().get("completed_fold_ids", []),
            last_successful_stage=checkpoint.load().get("last_successful_stage"),
            **log_context,
        )
    else:
        write_json_atomic(config_path, request.effective_config, overwrite=False)
        write_json_atomic(preflight_path, request.preflight_report, overwrite=False)
        checkpoint.initialize(identity, resolved_policy=request.resolved_policy.to_dict())
        checkpoint.transition(
            "initialized",
            artifacts=(inspect_artifact(config_path), inspect_artifact(preflight_path)),
        )
        manifest = build_run_manifest(
            run_id=run_id,
            model=request.model,
            selector=request.selector,
            experiment_type=request.experiment_type,
            config=request.effective_config,
            data_dir=request.experiment_config.data_dir,
            random_seed=request.seed,
            output_folder=run_dir,
            project_root=request.repository_root,
            status="running",
            artifact_applicability=request.artifact_applicability,
        )
        manifest.update(
            {
                "split_protocol": request.split_protocol,
                "preflight": "preflight.json",
                "execution_policy": request.resolved_policy.to_dict(),
                "resolved_parallelism": request.resolved_policy.to_dict()["parallelism"],
                "checkpoint_path": "checkpoint.json",
                "resumability_status": "explicit_only",
                "config_hash": identity["resolved_config_hash"],
                "protocol_hash": identity["protocol_hash"],
                "data_hash": identity["data_hash"],
                "row_alignment_hash": identity["row_alignment_hash"],
            }
        )
        if request.manifest_metadata:
            manifest.update(dict(request.manifest_metadata))
        write_run_manifest(run_dir, manifest)
        append_run_index_row(
            request.results_root,
            {
                "run_id": run_id,
                "dataset": request.dataset,
                "selector": request.selector,
                "model": request.model,
                "split_protocol": request.split_protocol,
                "seed": request.seed,
                "status": "running",
                "started_at_utc": manifest["started_at_utc"],
                "run_directory": repository_relative_path(run_dir, request.repository_root),
                "config_path": repository_relative_path(config_path, request.repository_root),
                "manifest_path": repository_relative_path(manifest_path, request.repository_root),
                "notes": (
                    f"policy={request.resolved_policy.profile_name}; "
                    f"parallelism={json.dumps(request.resolved_policy.to_dict()['parallelism'], sort_keys=True)}"
                ),
            },
        )

    lock_path = run_dir / ".execution.lock"
    try:
        with lock_path.open("x", encoding="utf-8") as lock:
            lock.write(f"pid={os.getpid()}\n")
            lock.flush()
            os.fsync(lock.fileno())
        emit_research_event(
            "execution_lock_created",
            message="Run execution lock created",
            priority=True,
            lock_path=repository_relative_path(lock_path, request.repository_root),
            **log_context,
        )
        default_worker_kwargs = {
            "experiment_config": request.experiment_config,
            "checkpoint_identity": identity,
            "run_directory": str(run_dir),
        }
        resolved_worker_kwargs = (
            dict(request.worker_kwargs)
            if request.worker_kwargs is not None
            else default_worker_kwargs
        )
        if request.worker_kwargs is not None and request.merge_default_worker_kwargs:
            resolved_worker_kwargs = {**default_worker_kwargs, **request.worker_kwargs}
        supervisor = supervise_worker(
            worker_target=request.worker_target,
            worker_kwargs=resolved_worker_kwargs,
            policy=request.resolved_policy,
            results_root=request.results_root,
            temp_root=request.preflight_report["temporary_root"],
            run_association=f"{run_id}:{utc_timestamp()}",
        )
    finally:
        if lock_path.exists():
            lock_path.unlink()
        emit_research_event(
            "execution_lock_released",
            message="Run execution lock released",
            priority=True,
            lock_path=repository_relative_path(lock_path, request.repository_root),
            **log_context,
        )

    resource_payload = _resource_payload(supervisor, request)
    resource_usage = write_resource_usage(run_dir, resource_payload)
    resource_artifact = inspect_artifact(run_dir / "resource_usage.json")
    emit_research_event(
        "resource_artifact_written",
        message="Resource usage artifact written and hash-validated",
        priority=True,
        artifact_path=repository_relative_path(
            run_dir / "resource_usage.json", request.repository_root
        ),
        artifact_size_bytes=resource_artifact.size_bytes,
        artifact_sha256=resource_artifact.sha256,
        **log_context,
    )
    now = utc_timestamp()
    manifest["status"] = supervisor.status
    manifest["stop_code"] = supervisor.stop_code
    manifest["primary_stop_code"] = supervisor.primary_stop_code
    manifest["secondary_events"] = list(supervisor.secondary_events)
    manifest["stop_lifecycle"] = list(supervisor.stop_lifecycle)
    manifest["termination_condition"] = supervisor.termination_condition
    manifest["worker_exit_code"] = supervisor.worker_exit_code
    manifest["resource_usage"] = "resource_usage.json"
    manifest["resource_peaks"] = {
        "peak_process_tree_rss_bytes": supervisor.peak_process_tree_rss_bytes,
        "peak_process_gpu_bytes": supervisor.peak_process_gpu_bytes,
        "minimum_system_available_ram_bytes": supervisor.minimum_system_available_ram_bytes,
        "minimum_results_free_disk_bytes": supervisor.minimum_results_free_disk_bytes,
        "minimum_temp_free_disk_bytes": supervisor.minimum_temp_free_disk_bytes,
    }
    if supervisor.worker_error:
        manifest["error"] = supervisor.worker_error

    published_status = supervisor.status
    if supervisor.status == "completed":
        manifest["summary"] = (supervisor.return_value or {}).get("summary", {})
        additional_artifacts = (supervisor.return_value or {}).get(
            "additional_artifacts", {}
        )
        if additional_artifacts:
            if not isinstance(additional_artifacts, Mapping):
                raise ValueError("worker additional_artifacts must be a mapping")
            manifest.setdefault("artifacts", {}).update(
                {
                    str(name): {
                        "applicable": True,
                        "path": str(relative),
                        "present": True,
                    }
                    for name, relative in additional_artifacts.items()
                }
            )
        if request.defer_terminal_success:
            if not request.deferred_success_status.strip():
                raise ValueError("deferred success status must not be empty")
            published_status = request.deferred_success_status
            manifest["status"] = published_status
            manifest["dev_phase_completed_at_utc"] = now
            manifest["resumability_status"] = "phase_complete_explicit_resume_required"
            checkpoint.transition(
                "dev_prediction_completed",
                artifacts=(resource_artifact,),
                resource_peaks=manifest["resource_peaks"],
            )
            write_run_manifest(run_dir, manifest)
        else:
            materialize_standard_artifacts(run_dir)
            manifest["completed_at_utc"] = now
            manifest["resumability_status"] = "completed_immutable"
            checkpoint.transition(
                "completed",
                artifacts=(resource_artifact,),
                resource_peaks=manifest["resource_peaks"],
            )
            write_run_manifest(run_dir, manifest)
            mark_completed(run_dir)
    else:
        terminal = (
            "aborted_resource_limit"
            if supervisor.status == "aborted_resource_limit"
            else "interrupted"
            if supervisor.status == "interrupted"
            else "failed"
        )
        manifest[f"{terminal}_at_utc"] = now
        manifest["resumability_status"] = "explicit_resume_validation_required"
        checkpoint.transition(
            terminal,
            artifacts=(resource_artifact,),
            resource_peaks=manifest["resource_peaks"],
            stop_code=supervisor.stop_code,
            error=supervisor.worker_error,
            termination_metadata={
                "primary_stop_code": supervisor.primary_stop_code,
                "secondary_events": supervisor.secondary_events,
                "stop_lifecycle": supervisor.stop_lifecycle,
                "termination_condition": supervisor.termination_condition,
                "cleanup_evidence": {
                    "child_cleanup_confirmed": supervisor.child_cleanup_confirmed,
                    "queue_cleanup_confirmed": supervisor.queue_cleanup_confirmed,
                    "parent_rss_before_bytes": supervisor.parent_rss_before_bytes,
                    "parent_rss_after_bytes": supervisor.parent_rss_after_bytes,
                    "system_available_ram_after_bytes": (
                        supervisor.system_available_ram_after_bytes
                    ),
                    "survivor_processes": supervisor.survivor_processes,
                },
            },
        )
        write_run_manifest(run_dir, manifest)

    update_run_index_row(
        request.results_root,
        run_id,
        {
            "status": published_status,
            "completed_at_utc": "" if request.defer_terminal_success else now,
            "runtime_seconds": resource_usage["timings_seconds"]["total"],
            "peak_ram_mb": resource_usage["peak_ram_mb"],
            "peak_gpu_mb": resource_usage["peak_gpu_mb"],
            "notes": f"stop_code={supervisor.stop_code or ''}; checkpoint=checkpoint.json",
        },
    )
    final_checkpoint = checkpoint.load()
    emit_research_event(
        "run_finalized",
        level="INFO" if published_status in {"completed", "dev_complete"} else "ERROR",
        message=f"Registered research run finalized as {published_status}",
        priority=True,
        status=published_status,
        stop_code=supervisor.stop_code,
        primary_stop_code=supervisor.primary_stop_code,
        secondary_events=supervisor.secondary_events,
        worker_exit_code=supervisor.worker_exit_code,
        child_cleanup_confirmed=supervisor.child_cleanup_confirmed,
        queue_cleanup_confirmed=supervisor.queue_cleanup_confirmed,
        survivor_pids=[item["pid"] for item in supervisor.survivor_processes],
        completed_fold_ids=final_checkpoint.get("completed_fold_ids", []),
        last_successful_stage=final_checkpoint.get("last_successful_stage"),
        checkpoint_status=final_checkpoint.get("status"),
        **log_context,
    )
    return ExecutionOutcome(
        run_id=run_id,
        run_directory=run_dir,
        status=published_status,
        stop_code=supervisor.stop_code,
        supervisor=supervisor,
        manifest=manifest,
    )
