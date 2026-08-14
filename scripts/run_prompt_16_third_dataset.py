"""Supervised CLI for the frozen Prompt-16 third-dataset execution."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MAX_AUTOMATIC_PHASE_RESOURCE_RETRIES = 57
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an authenticated, resource-supervised Prompt-16 operation"
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    registry = subparsers.add_parser(
        "registry", help="Print the canonical data-free 27/30 execution registry"
    )
    registry.add_argument("--protocol-lock", type=Path, required=True)

    matrix = subparsers.add_parser(
        "matrix", help="Build and authenticate the real modeling matrix"
    )
    matrix.add_argument("--plan", type=Path, required=True)

    phase = subparsers.add_parser(
        "phase", help="Run one locked pilot/DEV fold or the one-time OOT phase"
    )
    phase.add_argument("--plan", type=Path, required=True)
    phase.add_argument("--phase", choices=("pilot", "dev", "oot"), required=True)
    phase.add_argument("--fold-id", type=int, choices=(1, 2, 3, 4, 5))
    phase.add_argument("--oot-analysis-plan", type=Path)

    supplement = subparsers.add_parser(
        "supplemental-dev",
        help="Run/resume the authenticated two-method supplement across all five DEV folds",
    )
    supplement.add_argument("--authorization", type=Path, required=True)
    return parser


def _write_supervision(
    *,
    result: Any,
    log_root: Path,
    name: str,
    write_json_atomic: Any,
    write_csv_atomic: Any,
) -> None:
    import pandas as pd

    log_root.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    samples = payload.pop("samples", [])
    write_json_atomic(log_root / f"{name}_supervisor_summary.json", payload)
    write_csv_atomic(
        log_root / f"{name}_resource_samples.csv",
        pd.DataFrame(samples),
    )


def _run_with_automatic_resource_retries(
    *,
    supervise_attempt: Callable[[], Any],
    seal_resource_stop: Callable[[Any], Mapping[str, Any] | None],
    announce_retry: Callable[[int, Any, Mapping[str, Any]], None],
    maximum_retries: int = MAX_AUTOMATIC_PHASE_RESOURCE_RETRIES,
) -> tuple[Any, Mapping[str, Any] | None, tuple[dict[str, Any], ...]]:
    """Retry only newly sealed, fully cleaned-up phase resource stops."""

    if maximum_retries < 0:
        raise ValueError("maximum_retries must be non-negative")
    retry_count = 0
    seen_scopes: set[tuple[str, str]] = set()
    sealed_infeasibilities: list[dict[str, Any]] = []
    while True:
        result = supervise_attempt()
        infeasibility: Mapping[str, Any] | None = None
        controlled_resource_stop = (
            result.status in {"aborted_resource_limit", "timed_out"}
            and result.stop_code in {"ram_process_limit", "wall_clock_limit"}
        )
        cleanup_authenticated = (
            bool(getattr(result, "child_cleanup_confirmed", False))
            and bool(getattr(result, "queue_cleanup_confirmed", False))
            and not tuple(getattr(result, "survivor_processes", ()) or ())
        )
        if controlled_resource_stop and cleanup_authenticated:
            infeasibility = seal_resource_stop(result)

        seal_complete = (
            isinstance(infeasibility, Mapping)
            and infeasibility.get("status") == "complete"
            and bool(infeasibility.get("kind"))
            and bool(infeasibility.get("id"))
            and bool(infeasibility.get("manifest_sha256"))
        )
        if not seal_complete or retry_count >= maximum_retries:
            return result, infeasibility, tuple(sealed_infeasibilities)

        scope = (str(infeasibility["kind"]), str(infeasibility["id"]))
        if scope in seen_scopes:
            return result, infeasibility, tuple(sealed_infeasibilities)
        seen_scopes.add(scope)
        retry_count += 1
        sealed = dict(infeasibility)
        sealed_infeasibilities.append(sealed)
        announce_retry(retry_count, result, sealed)


def _run_supervised(args: argparse.Namespace) -> int:
    if args.operation == "supplemental-dev":
        return _run_supplemental_dev_supervised(args)

    from credit_risk_fs.experiments.atomic_io import write_csv_atomic, write_json_atomic
    from credit_risk_fs.experiments.prompt_16_third_dataset import (
        Prompt16ExecutionError,
        acquire_execution_lock,
        canonical_registry,
        directory_size_bytes,
        free_disk_bytes,
        load_execution_plan,
        record_phase_resource_infeasibility,
        release_execution_lock,
    )
    from credit_risk_fs.experiments.resource_monitor import supervise_worker
    from credit_risk_fs.experiments.research_logging import (
        ResearchLogSession,
        emit_research_event,
    )
    from credit_risk_fs.experiments.ram_control import load_ram_control_policy
    from credit_risk_fs.experiments.resource_policy import (
        GIB,
        detect_hardware,
        load_execution_policy,
        resolve_execution_policy,
    )

    if args.operation == "registry":
        print(
            json.dumps(
                canonical_registry(args.protocol_lock), indent=2, ensure_ascii=False
            )
        )
        return 0

    plan_path = args.plan.resolve()
    plan = load_execution_plan(plan_path)
    paths = plan["paths"]
    protocol_lock = Path(paths["protocol_lock"])
    if not protocol_lock.is_file():
        raise Prompt16ExecutionError("planned protocol lock is missing")
    policy_path = Path(plan["resource_controls"]["execution_policy_path"])
    temp_root = Path(paths["temp_root"])
    log_root = Path(paths["log_root"])
    temp_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    phase: str | None = None
    if args.operation == "matrix":
        output_root = Path(paths["matrix_root"])
        target = "credit_risk_fs.experiments.prompt_16_third_dataset:run_matrix_worker"
        worker_kwargs = {
            "input_root": paths["raw_dataset_root"],
            "matrix_root": paths["matrix_root"],
            "protocol_lock": paths["protocol_lock"],
            "shard_rows": int(plan["adapter"]["case_shard_rows"]),
            "resource_event_path": str(log_root / "matrix_resource_events.jsonl"),
        }
        max_seconds = float(plan["wall_time_limits_seconds"]["matrix_total"])
        stage_wall_clock_limits = {
            "matrix_inventory_and_build": float(
                plan["wall_time_limits_seconds"]["raw_inventory_scan"]
            ),
            "matrix_base_scan": float(
                plan["wall_time_limits_seconds"]["base_scan"]
            ),
            "matrix_depth_0_scan": float(
                plan["wall_time_limits_seconds"]["each_depth_0_scan"]
            ),
            "matrix_depth_1_scan_and_aggregation": float(
                plan["wall_time_limits_seconds"][
                    "each_depth_1_scan_and_aggregation"
                ]
            ),
            "matrix_checkpoint_unit": float(
                plan["wall_time_limits_seconds"]["each_checkpoint_write_or_read"]
            ),
            "matrix_publication_unit": float(
                plan["wall_time_limits_seconds"]["each_checkpoint_write_or_read"]
            ),
            "matrix_authentication": float(
                plan["wall_time_limits_seconds"]["raw_inventory_scan"]
            ),
        }
        name = "matrix"
    else:
        phase = str(args.phase)
        if phase == "pilot":
            if args.fold_id != 1 or args.oot_analysis_plan is not None:
                raise Prompt16ExecutionError("pilot command must use fold 1 and no OOT plan")
            output_root = Path(paths["pilot_root"]) / "fold_1"
            max_seconds = float(plan["wall_time_limits_seconds"]["pilot_total"])
            name = "pilot_fold_1"
        elif phase == "dev":
            if args.fold_id not in {1, 2, 3, 4, 5} or args.oot_analysis_plan is not None:
                raise Prompt16ExecutionError("DEV command requires one fold in 1..5 and no OOT plan")
            output_root = Path(paths["dev_root"]) / f"fold_{args.fold_id}"
            max_seconds = float(plan["wall_time_limits_seconds"]["dev_fold_total"])
            name = f"dev_fold_{args.fold_id}"
        else:
            raise Prompt16ExecutionError(
                "the former classical-only OOT command is revoked; no Prompt-16 OOT "
                "operation is authorized until the complete amended 170-cell DEV gate passes"
            )
        target = "credit_risk_fs.experiments.prompt_16_third_dataset:run_phase_worker"
        worker_kwargs = {
            "matrix_root": paths["matrix_root"],
            "output_root": str(output_root),
            "protocol_lock": paths["protocol_lock"],
            "phase": phase,
            "fold_id": args.fold_id,
            "oot_analysis_plan": (
                None if args.oot_analysis_plan is None else str(args.oot_analysis_plan.resolve())
            ),
        }
        stage_wall_clock_limits = dict(
            canonical_registry(protocol_lock)["resource_controls"][
                "wall_clock_limits_seconds"
            ]
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = free_disk_bytes(output_root.parent)
    forecast_total = int(plan["resource_forecast"]["forecast_total_output_bytes"])
    existing_bytes = sum(
        directory_size_bytes(path)
        for path in plan["resource_forecast"]["forecast_roots"]
    )
    remaining_forecast = max(0, forecast_total - existing_bytes)
    forecast_factor = float(
        plan["resource_controls"]["disk_remaining_output_safety_factor"]
    )
    floor_bytes = max(
        int(plan["resource_controls"]["disk_free_hard_floor_bytes"]),
        int(math.ceil(forecast_factor * remaining_forecast)),
    )
    if free_bytes < floor_bytes:
        raise Prompt16ExecutionError(
            f"results-volume free space is below the frozen floor: {free_bytes} < {floor_bytes}"
        )
    configured = load_execution_policy(PROJECT_ROOT, policy_path)
    capacity = detect_hardware(output_root.parent, temp_root)
    policy = resolve_execution_policy(configured, capacity)
    ram_control = load_ram_control_policy(
        PROJECT_ROOT,
        "configs/execution/prompt_16_ram_wait_resume_v1.yaml",
        total_physical_ram_bytes=int(capacity.total_ram_gb * GIB),
    )
    if policy.memory.abort_process_tree_rss_gb > 24.0 + 1e-12:
        raise Prompt16ExecutionError("resolved RSS cap exceeds the Prompt-16 hard ceiling")
    if abs(policy.memory.abort_if_system_available_below_gb - 1.0) > 1e-12:
        raise Prompt16ExecutionError("Prompt-16 system-available floor must remain 1 GiB")
    if policy.parallelism.concurrent_experiment_runs != 1:
        raise Prompt16ExecutionError("Prompt-16 requires one experiment cell at a time")
    if policy.parallelism.concurrent_folds != 1:
        raise Prompt16ExecutionError("Prompt-16 requires one fold at a time")
    if policy.parallelism.estimator_threads > 4:
        raise Prompt16ExecutionError("Prompt-16 permits at most four estimator threads")
    if ram_control.emergency_margin_bytes != 1 * GIB:
        raise Prompt16ExecutionError("Prompt-16 RAM wait threshold must remain 1 GiB")
    if ram_control.recovery_threshold_bytes != 2 * GIB:
        raise Prompt16ExecutionError("Prompt-16 RAM recovery threshold must remain 2 GiB")
    if ram_control.log_interval_seconds != 300:
        raise Prompt16ExecutionError("Prompt-16 RAM wait logging must remain five-minute")

    lock = acquire_execution_lock(output_root)
    try:
        terminal_log_path = log_root / f"{name}.log"
        with ResearchLogSession(
            terminal_log_path,
            repository_root=PROJECT_ROOT,
            command_arguments=[str(value) for value in sys.argv],
        ) as session:
            def supervise_attempt() -> Any:
                attempt_result = supervise_worker(
                    worker_target=target,
                    worker_kwargs=worker_kwargs,
                    policy=policy,
                    results_root=output_root.parent,
                    temp_root=temp_root,
                    run_association=f"prompt16:{name}",
                    heartbeat_interval_seconds=30.0,
                    max_wall_clock_seconds=max_seconds,
                    stage_wall_clock_limits_seconds=stage_wall_clock_limits,
                    enforce_process_tree_rss_limit=True,
                    ram_control_policy=ram_control,
                )
                _write_supervision(
                    result=attempt_result,
                    log_root=log_root,
                    name=name,
                    write_json_atomic=write_json_atomic,
                    write_csv_atomic=write_csv_atomic,
                )
                return attempt_result

            def seal_resource_stop(result: Any) -> Mapping[str, Any] | None:
                if args.operation != "phase" or phase not in {"pilot", "dev"}:
                    return None
                supervisor_evidence = result.to_dict()
                supervisor_evidence.pop("samples", None)
                return record_phase_resource_infeasibility(
                    matrix_root=paths["matrix_root"],
                    output_root=output_root,
                    protocol_lock=paths["protocol_lock"],
                    phase=str(phase),
                    fold_id=int(args.fold_id),
                    stopped_stage=result.final_stage,
                    stopped_scope=result.final_fold_id,
                    supervisor_evidence=supervisor_evidence,
                )

            def announce_retry(
                retry_count: int,
                result: Any,
                sealed: Mapping[str, Any],
            ) -> None:
                emit_research_event(
                    "plan_resume_decision",
                    message=(
                        f"{name} sealed resource-infeasible {sealed['kind']} "
                        f"{sealed['id']}; retrying automatically "
                        f"({retry_count}/{MAX_AUTOMATIC_PHASE_RESOURCE_RETRIES})"
                    ),
                    priority=True,
                    retry_count=retry_count,
                    retry_limit=MAX_AUTOMATIC_PHASE_RESOURCE_RETRIES,
                    stop_code=result.stop_code,
                    sealed_kind=sealed["kind"],
                    sealed_id=sealed["id"],
                    sealed_manifest_sha256=sealed["manifest_sha256"],
                )

            result, infeasibility, sealed_infeasibilities = (
                _run_with_automatic_resource_retries(
                    supervise_attempt=supervise_attempt,
                    seal_resource_stop=seal_resource_stop,
                    announce_retry=announce_retry,
                )
            )
            if result.status == "completed":
                session.finish(
                    "session_completed",
                    message=f"{name} completed and supervisor cleanup authenticated",
                )
            elif result.status in {
                "aborted_resource_limit",
                "interrupted",
                "timed_out",
            }:
                session.finish(
                    "session_controlled_stop",
                    message=f"{name} stopped under the frozen runtime controls",
                    stop_code=result.stop_code,
                )
            else:
                session.finish(
                    "session_failed",
                    level="ERROR",
                    message=f"{name} failed; inspect the authenticated debug log",
                    stop_code=result.stop_code,
                )
    finally:
        release_execution_lock(lock)
    print(
        json.dumps(
            {
                "operation": name,
                "status": result.status,
                "stop_code": result.stop_code,
                "peak_process_tree_rss_bytes": result.peak_process_tree_rss_bytes,
                "minimum_system_available_ram_bytes": result.minimum_system_available_ram_bytes,
                "output_root": str(output_root),
                "log_root": str(log_root),
                "terminal_log_path": str(terminal_log_path),
                "sealed_resource_infeasibility": infeasibility,
                "automatic_resource_retry_count": len(sealed_infeasibilities),
                "automatically_sealed_resource_infeasibilities": [
                    {
                        "kind": item["kind"],
                        "id": item["id"],
                        "manifest_sha256": item["manifest_sha256"],
                    }
                    for item in sealed_infeasibilities
                ],
            },
            indent=2,
        )
    )
    return 0 if result.status == "completed" else 2


def _run_supplemental_dev_supervised(args: argparse.Namespace) -> int:
    """Supervise the one all-fold DEV-only successor operation."""

    from credit_risk_fs.experiments.atomic_io import write_csv_atomic, write_json_atomic
    from credit_risk_fs.experiments.prompt_16_llm_supplement import (
        active_prompt16_workers,
        load_supplemental_authorization,
        prompt16_execution_locks,
        record_supplemental_resource_stop,
    )
    from credit_risk_fs.experiments.prompt_16_third_dataset import (
        Prompt16ExecutionError,
        acquire_execution_lock,
        canonical_registry,
        directory_size_bytes,
        free_disk_bytes,
        load_execution_plan,
        release_execution_lock,
    )
    from credit_risk_fs.experiments.ram_control import load_ram_control_policy
    from credit_risk_fs.experiments.research_logging import (
        ResearchLogSession as SupplementalResearchLogSession,
        emit_research_event,
    )
    from credit_risk_fs.experiments.resource_monitor import (
        supervise_worker as supervise_supplemental_worker,
    )
    from credit_risk_fs.experiments.resource_policy import (
        GIB,
        detect_hardware,
        load_execution_policy,
        resolve_execution_policy,
    )

    authorization_path = args.authorization.resolve()
    authorization, _ = load_supplemental_authorization(authorization_path)
    plan_path = PROJECT_ROOT / str(authorization["execution_plan_path"])
    plan = load_execution_plan(plan_path)
    paths = plan["paths"]
    workers = active_prompt16_workers()
    if workers:
        raise Prompt16ExecutionError(
            f"another Prompt-16 Python worker is active: {workers}"
        )
    execution_locks = prompt16_execution_locks(plan)
    if execution_locks:
        raise Prompt16ExecutionError(
            f"a Prompt-16 execution lock already exists: {execution_locks}"
        )
    output_root = Path(authorization["paths"]["output_root"])
    log_root = Path(authorization["paths"]["log_root"])
    temp_root = Path(paths["temp_root"])
    log_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    free_bytes = free_disk_bytes(output_root.parent)
    forecast_total = int(plan["resource_forecast"]["forecast_total_output_bytes"])
    existing_bytes = sum(
        directory_size_bytes(path)
        for path in plan["resource_forecast"]["forecast_roots"]
    )
    remaining_forecast = max(0, forecast_total - existing_bytes)
    forecast_factor = float(
        plan["resource_controls"]["disk_remaining_output_safety_factor"]
    )
    floor_bytes = max(
        int(plan["resource_controls"]["disk_free_hard_floor_bytes"]),
        int(math.ceil(forecast_factor * remaining_forecast)),
    )
    if free_bytes < floor_bytes:
        raise Prompt16ExecutionError(
            f"results-volume free space is below the frozen floor: {free_bytes} < {floor_bytes}"
        )

    policy_path = Path(plan["resource_controls"]["execution_policy_path"])
    configured = load_execution_policy(PROJECT_ROOT, policy_path)
    capacity = detect_hardware(output_root.parent, temp_root)
    policy = resolve_execution_policy(configured, capacity)
    ram_control = load_ram_control_policy(
        PROJECT_ROOT,
        "configs/execution/prompt_16_ram_wait_resume_v1.yaml",
        total_physical_ram_bytes=int(capacity.total_ram_gb * GIB),
    )
    if policy.memory.abort_process_tree_rss_gb > 24.0 + 1e-12:
        raise Prompt16ExecutionError("resolved RSS cap exceeds the Prompt-16 hard ceiling")
    if abs(policy.memory.abort_if_system_available_below_gb - 1.0) > 1e-12:
        raise Prompt16ExecutionError("Prompt-16 system-available floor must remain 1 GiB")
    if policy.parallelism.concurrent_experiment_runs != 1:
        raise Prompt16ExecutionError("Prompt-16 requires one experiment cell at a time")
    if policy.parallelism.concurrent_folds != 1:
        raise Prompt16ExecutionError("Prompt-16 requires one fold at a time")
    if policy.parallelism.data_loader_workers != 0:
        raise Prompt16ExecutionError("Prompt-16 requires zero data-loader workers")
    if policy.parallelism.estimator_threads > 4:
        raise Prompt16ExecutionError("Prompt-16 permits at most four estimator threads")
    if ram_control.emergency_margin_bytes != 1 * GIB:
        raise Prompt16ExecutionError("Prompt-16 RAM wait threshold must remain 1 GiB")
    if ram_control.recovery_threshold_bytes != 2 * GIB:
        raise Prompt16ExecutionError("Prompt-16 RAM recovery threshold must remain 2 GiB")
    if ram_control.log_interval_seconds != 300:
        raise Prompt16ExecutionError("Prompt-16 RAM wait logging must remain five-minute")

    name = "llm_supplement_v2_all_folds_dev"
    terminal_log_path = log_root / f"{name}.log"
    stage_wall_clock_limits = dict(
        canonical_registry(paths["protocol_lock"])["resource_controls"][
            "wall_clock_limits_seconds"
        ]
    )
    max_seconds = float(plan["wall_time_limits_seconds"]["dev_total"])
    target = (
        "credit_risk_fs.experiments.prompt_16_llm_supplement:"
        "run_supplemental_dev_worker"
    )
    worker_kwargs = {"authorization_path": str(authorization_path)}
    lock = acquire_execution_lock(output_root)
    try:
        with SupplementalResearchLogSession(
            terminal_log_path,
            repository_root=PROJECT_ROOT,
            command_arguments=[str(value) for value in sys.argv],
        ) as supplemental_session:

            supervisor_attempt = 0

            def supervise_attempt() -> Any:
                nonlocal supervisor_attempt
                supervisor_attempt += 1
                attempt_result = supervise_supplemental_worker(
                    worker_target=target,
                    worker_kwargs=worker_kwargs,
                    policy=policy,
                    results_root=output_root.parent,
                    temp_root=temp_root,
                    run_association="prompt16:llm_supplement_v2:all_folds_dev",
                    heartbeat_interval_seconds=30.0,
                    max_wall_clock_seconds=max_seconds,
                    stage_wall_clock_limits_seconds=stage_wall_clock_limits,
                    enforce_process_tree_rss_limit=True,
                    ram_control_policy=ram_control,
                )
                attempt_name = f"{name}_attempt_{supervisor_attempt:03d}"
                _write_supervision(
                    result=attempt_result,
                    log_root=log_root,
                    name=attempt_name,
                    write_json_atomic=write_json_atomic,
                    write_csv_atomic=write_csv_atomic,
                )
                _write_supervision(
                    result=attempt_result,
                    log_root=log_root,
                    name=name,
                    write_json_atomic=write_json_atomic,
                    write_csv_atomic=write_csv_atomic,
                )
                return attempt_result

            def seal_resource_stop(result: Any) -> Mapping[str, Any] | None:
                evidence = result.to_dict()
                evidence.pop("samples", None)
                return record_supplemental_resource_stop(
                    authorization_path=authorization_path,
                    supervisor_evidence=evidence,
                )

            def announce_retry(
                retry_count: int,
                result: Any,
                sealed: Mapping[str, Any],
            ) -> None:
                emit_research_event(
                    "plan_resume_decision",
                    message=(
                        f"{name} preserved {sealed['id']}; resuming the identical "
                        f"all-fold command ({retry_count}/{MAX_AUTOMATIC_PHASE_RESOURCE_RETRIES})"
                    ),
                    priority=True,
                    retry_count=retry_count,
                    retry_limit=MAX_AUTOMATIC_PHASE_RESOURCE_RETRIES,
                    stop_code=result.stop_code,
                    sealed_kind=sealed["kind"],
                    sealed_id=sealed["id"],
                    sealed_manifest_sha256=sealed["manifest_sha256"],
                )

            result, infeasibility, sealed_stops = _run_with_automatic_resource_retries(
                supervise_attempt=supervise_attempt,
                seal_resource_stop=seal_resource_stop,
                announce_retry=announce_retry,
            )
            if result.status == "completed":
                supplemental_session.finish(
                    "session_completed",
                    message=(
                        "all five supplemental DEV folds completed and supervisor "
                        "cleanup authenticated; OOT was not accessible"
                    ),
                )
            elif result.status in {
                "aborted_resource_limit",
                "interrupted",
                "timed_out",
            }:
                supplemental_session.finish(
                    "session_controlled_stop",
                    message="supplemental DEV stopped with resumable evidence preserved",
                    stop_code=result.stop_code,
                )
            else:
                supplemental_session.finish(
                    "session_failed",
                    level="ERROR",
                    message="supplemental DEV failed; rerun the identical command after inspection",
                    stop_code=result.stop_code,
                )
    finally:
        release_execution_lock(lock)

    print(
        json.dumps(
            {
                "operation": "supplemental_dev_only",
                "status": result.status,
                "stop_code": result.stop_code,
                "peak_process_tree_rss_bytes": result.peak_process_tree_rss_bytes,
                "minimum_system_available_ram_bytes": result.minimum_system_available_ram_bytes,
                "output_root": str(output_root),
                "status_path": str(output_root / "controller_status.json"),
                "success_marker": str(output_root / "_SUCCESS"),
                "ranking_provenance_root": str(output_root / "llm_ranking"),
                "log_root": str(log_root),
                "terminal_log_path": str(terminal_log_path),
                "oot_started": False,
                "sealed_resource_stop": infeasibility,
                "automatic_resource_retry_count": len(sealed_stops),
            },
            indent=2,
        )
    )
    return 0 if result.status == "completed" else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _run_supervised(args)


if __name__ == "__main__":
    raise SystemExit(main())
