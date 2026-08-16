"""One-command resilient controller for the final Prompt-16 34-cell OOT run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticate and run/resume the one-time final Prompt-16 34-cell OOT lifecycle"
    )
    parser.add_argument("--authorization", type=Path, required=True)
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scope_cell(scope: Any) -> int | None:
    match = re.search(r"cell_(\d{3})", str(scope or ""))
    if not match:
        return None
    value = int(match.group(1))
    return value if 1 <= value <= 34 else None


def _count_cells(output_root: Path) -> dict[str, Any]:
    completed: list[int] = []
    unavailable: list[int] = []
    failed: list[int] = []
    for order in range(1, 35):
        phase = "classical" if order <= 30 else "supplemental"
        cell = output_root / phase / "evaluations" / f"cell_{order:03d}"
        if not (cell / "_SUCCESS").is_file() or not (cell / "status.json").is_file():
            continue
        try:
            status = json.loads((cell / "status.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            failed.append(order)
            continue
        value = status.get("status")
        if value == "complete":
            completed.append(order)
        elif value in {"unavailable", "failed"}:
            unavailable.append(order)
        else:
            failed.append(order)
    accounted = sorted(completed + unavailable)
    next_cell = next((order for order in range(1, 35) if order not in accounted), None)
    return {
        "completed_cell_orders": completed,
        "unavailable_cell_orders": unavailable,
        "failed_or_corrupt_cell_orders": failed,
        "completed_count": len(completed),
        "unavailable_count": len(unavailable),
        "failed_count": len(failed),
        "accounted_count": len(accounted),
        "incomplete_count": 34 - len(accounted),
        "next_cell": next_cell,
    }


def _recover_stale_lock(lock_path: Path) -> None:
    if not lock_path.is_file():
        return
    import psutil

    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        pid = int(payload.get("pid"))
        active = psutil.pid_exists(pid)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        active = True
    if active:
        raise RuntimeError(f"Prompt-16 execution lock is active: {lock_path}")
    archive = lock_path.with_name(
        lock_path.name + f".stale-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    os.replace(lock_path, archive)


def _active_final_oot_workers() -> list[dict[str, Any]]:
    import psutil

    current_pid = os.getpid()
    current_lineage_pids = {current_pid}
    try:
        current_process = psutil.Process(current_pid)
        while True:
            parent = current_process.parent()
            if parent is None:
                break
            parent_pid = int(parent.pid)
            if parent_pid in current_lineage_pids:
                break
            current_lineage_pids.add(parent_pid)
            current_process = parent
    except (psutil.Error, TypeError, ValueError):
        pass
    workers: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = int(process.info["pid"])
            name = str(process.info.get("name") or "").lower()
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.Error, TypeError, ValueError):
            continue
        if pid in current_lineage_pids or not name.startswith("python"):
            continue
        normalized = command.lower().replace("\\", "/")
        if (
            "run_prompt_16_final_oot.py" in normalized
            or "prompt_16_final_oot:run_" in normalized
        ):
            workers.append({"pid": pid, "name": name, "command_line": command})
    return workers


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from credit_risk_fs.experiments.atomic_io import (
        write_csv_atomic,
        write_json_atomic,
        write_text_atomic,
    )
    from credit_risk_fs.experiments.prompt_16_final_oot import (
        MAX_RESOURCE_RECOVERY_RESTARTS_PER_SCOPE,
        _assert_clean_worktree,
        _assert_no_preexisting_prompt16_oot,
        load_final_authorization,
    )
    from credit_risk_fs.experiments.prompt_16_llm_supplement import (
        active_prompt16_workers,
    )
    from credit_risk_fs.experiments.prompt_16_third_dataset import (
        Prompt16ExecutionError,
        acquire_execution_lock,
        execution_lock_path,
        release_execution_lock,
    )
    from credit_risk_fs.experiments.ram_control import load_ram_control_policy
    from credit_risk_fs.experiments.research_logging import (
        ResearchLogSession,
        emit_research_event,
    )
    from credit_risk_fs.experiments.resource_monitor import (
        supervise_worker,
        wait_for_inter_run_readiness,
    )
    from credit_risk_fs.experiments.resource_policy import (
        GIB,
        detect_hardware,
        load_execution_policy,
        resolve_execution_policy,
    )

    authorization_path = args.authorization.resolve()
    authorization, authorization_sha = load_final_authorization(
        authorization_path, repository_root=PROJECT_ROOT
    )
    _assert_clean_worktree(PROJECT_ROOT)
    tracked = os.path.relpath(authorization_path, PROJECT_ROOT).replace("\\", "/")
    import subprocess

    tracked_check = subprocess.run(
        ["git", "ls-files", "--error-unmatch", tracked],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if tracked_check.returncode != 0:
        raise Prompt16ExecutionError("final OOT authorization is not committed")
    workers = active_prompt16_workers() + _active_final_oot_workers()
    if workers:
        raise Prompt16ExecutionError(f"another Prompt-16 worker is active: {workers}")
    _assert_no_preexisting_prompt16_oot(PROJECT_ROOT)

    output_root = Path(authorization["paths"]["output_root"])
    log_root = Path(authorization["paths"]["log_root"])
    temp_root = Path(authorization["paths"]["temp_root"])
    status_path = Path(authorization["paths"]["controller_status"])
    terminal_log = Path(authorization["paths"]["terminal_log"])
    output_root.parent.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    if output_root.exists() and not status_path.is_file() and not (
        output_root / "_SUCCESS"
    ).is_file():
        raise Prompt16ExecutionError(
            "final OOT root exists without an authenticated controller checkpoint"
        )
    prior_status: dict[str, Any] = {}
    if status_path.is_file():
        prior_status = json.loads(status_path.read_text(encoding="utf-8"))
        if prior_status.get("execution_authorization_sha256") != authorization_sha:
            raise Prompt16ExecutionError("existing controller checkpoint authorization changed")

    configured = load_execution_policy(
        PROJECT_ROOT, authorization["paths"]["execution_policy"]
    )
    capacity = detect_hardware(output_root.parent, temp_root)
    policy = resolve_execution_policy(configured, capacity)
    ram_policy = load_ram_control_policy(
        PROJECT_ROOT,
        authorization["paths"]["ram_wait_policy"],
        total_physical_ram_bytes=int(capacity.total_ram_gb * GIB),
    )
    if policy.parallelism.concurrent_experiment_runs != 1:
        raise Prompt16ExecutionError("final OOT requires one active experiment cell")
    if policy.parallelism.concurrent_folds != 1:
        raise Prompt16ExecutionError("final OOT requires one active refit/fold")
    if policy.parallelism.estimator_threads != 4:
        raise Prompt16ExecutionError("final OOT estimator thread count must be exactly four")
    if abs(policy.memory.abort_process_tree_rss_gb - 24.0) > 1e-12:
        raise Prompt16ExecutionError("final OOT process-tree RSS cap must be 24 GiB")
    if abs(policy.memory.abort_if_system_available_below_gb - 4.0) > 1e-12:
        raise Prompt16ExecutionError("final OOT hard RAM floor must be 4 GiB")
    if ram_policy.emergency_margin_bytes != 6 * GIB:
        raise Prompt16ExecutionError("final OOT soft RAM threshold must be 6 GiB")
    if ram_policy.recovery_threshold_bytes != 8 * GIB:
        raise Prompt16ExecutionError("final OOT RAM resume threshold must be 8 GiB")
    if ram_policy.recovery_consecutive_checks != 3:
        raise Prompt16ExecutionError("final OOT RAM recovery requires three stable polls")
    if ram_policy.check_interval_seconds > 5:
        raise Prompt16ExecutionError("final OOT resource polling must be at least every 5s")
    if ram_policy.log_interval_seconds > 30:
        raise Prompt16ExecutionError("final OOT waiting logs must occur at least every 30s")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    lock_path = execution_lock_path(output_root)
    _recover_stale_lock(lock_path)
    lock = acquire_execution_lock(output_root)
    result: Any = None
    total_active = float(prior_status.get("active_elapsed_seconds", 0.0))
    total_wait = float(prior_status.get("ram_wait_seconds", 0.0))
    peak_rss = int(prior_status.get("peak_process_tree_rss_bytes", 0))
    minimum_available = prior_status.get("minimum_system_available_ram_bytes")
    lifetime_retries = {
        str(key): int(value)
        for key, value in dict(
            prior_status.get("automatic_resource_retry_counts_by_scope", {})
        ).items()
    }
    invocation_retries: dict[str, int] = {}
    global_retries = int(prior_status.get("automatic_resource_retry_count", 0))
    attempt = int(prior_status.get("supervisor_attempt_count", 0))
    last_status_write = 0.0
    last_waiting: bool | None = None

    def publish_status(
        *,
        state: str,
        stop_code: str | None,
        sample: Mapping[str, Any] | None = None,
        force: bool = True,
    ) -> None:
        nonlocal last_status_write, last_waiting, peak_rss, minimum_available
        nonlocal total_active, total_wait
        now = time.monotonic()
        waiting = bool((sample or {}).get("ram_waiting", False))
        if not force and now - last_status_write < 30 and waiting == last_waiting:
            return
        if sample:
            peak_rss = max(peak_rss, int(sample.get("peak_process_tree_rss_bytes", 0)))
            available = sample.get("minimum_system_available_ram_bytes")
            if available is not None:
                minimum_available = (
                    int(available)
                    if minimum_available is None
                    else min(int(minimum_available), int(available))
                )
        counts = _count_cells(output_root)
        scope = (sample or {}).get("fold_id", prior_status.get("current_scope"))
        current_cell = _scope_cell(scope)
        payload = {
            "schema_version": "prompt_16_final_oot_controller_status_v1",
            "operation": "prompt_16_final_amended_oot_once",
            "state": state,
            "status": "complete" if state == "DONE" else "running" if stop_code is None else "resumable_non_success",
            "execution_authorization_sha256": authorization_sha,
            "implementation_commit": authorization["implementation_commit"],
            "expected_total": 34,
            **counts,
            "current_cell": current_cell,
            "next_cell": counts["next_cell"],
            "current_scope": scope,
            "current_stage": (sample or {}).get("stage"),
            "current_attempt": attempt,
            "supervisor_attempt_count": attempt,
            "waiting_count": int(waiting),
            "recovering_count": int(state == "RESOURCE_RECOVERY_REQUIRED"),
            "active_elapsed_seconds": total_active
            + float((sample or {}).get("active_elapsed_seconds", 0.0)),
            "ram_wait_seconds": total_wait
            + float((sample or {}).get("ram_wait_seconds", 0.0)),
            "process_tree_rss_bytes": (sample or {}).get("process_tree_rss_bytes"),
            "system_available_ram_bytes": (sample or {}).get("system_available_ram_bytes"),
            "peak_process_tree_rss_bytes": peak_rss,
            "minimum_system_available_ram_bytes": minimum_available,
            "automatic_resource_retry_counts_by_scope": lifetime_retries,
            "automatic_resource_retry_count": global_retries,
            "invocation_resource_retry_counts_by_scope": invocation_retries,
            "oot_started": (
                (output_root / "classical/scope_authentication.json").is_file()
                or (output_root / "supplemental/scope_authentication.json").is_file()
                or (output_root / "analysis/oot_metrics.csv").is_file()
            ),
            "final_success": state == "DONE",
            "stop_code": stop_code,
            "checkpoint_identity": (
                None
                if counts["accounted_count"] == 0
                else f"cell_{max(counts['completed_cell_orders'] + counts['unavailable_cell_orders']):03d}"
            ),
            "paths": {
                "terminal_log": str(terminal_log),
                "audit_log": str(log_root / "events.jsonl"),
                "debug_log": str(log_root / "debug.log"),
                "controller_status": str(status_path),
                "checkpoints": str(output_root),
                "predictions": str(output_root),
                "metrics": str(output_root / "analysis/oot_metrics.csv"),
                "selections": str(output_root),
                "resource_traces": str(log_root / "attempts"),
                "completion_marker": str(output_root / "_SUCCESS"),
                "evidence_manifest": str(output_root / "final_evidence_manifest.json"),
            },
            "updated_at_utc": _utc_now(),
        }
        output_root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(status_path, payload)
        last_status_write = now
        last_waiting = waiting

    try:
        with ResearchLogSession(
            terminal_log,
            repository_root=PROJECT_ROOT,
            command_arguments=[str(value) for value in sys.argv],
        ) as session:
            publish_status(state="START", stop_code=None)
            while True:
                readiness = wait_for_inter_run_readiness(
                    policy=policy,
                    results_root=output_root.parent,
                    temp_root=temp_root,
                    ram_control_policy=ram_policy,
                )
                if not readiness.ready:
                    publish_status(
                        state="ERROR", stop_code=readiness.stop_code, force=True
                    )
                    session.finish(
                        "session_failed",
                        level="ERROR",
                        message="Final OOT resource readiness failed",
                        stop_code=readiness.stop_code,
                    )
                    return 2
                total_wait += float(readiness.total_ram_wait_seconds)
                attempt += 1
                worker_kwargs = {
                    "repository_root": str(PROJECT_ROOT),
                    "authorization_path": str(authorization_path),
                    "phase": "oot",
                    "spec": {
                        "run_id": "prompt16-final-amended-oot",
                        "dataset": "homecredit_model_stability_2024",
                        "attempt": attempt,
                    },
                }

                def callback(sample: Mapping[str, Any]) -> None:
                    publish_status(
                        state=(
                            "WAITING_FOR_RAM"
                            if sample.get("ram_waiting")
                            else "ACTIVE"
                        ),
                        stop_code=None,
                        sample=sample,
                        force=False,
                    )

                result = supervise_worker(
                    worker_target=(
                        "credit_risk_fs.experiments.prompt_16_final_oot:"
                        "run_final_oot_worker"
                    ),
                    worker_kwargs=worker_kwargs,
                    policy=policy,
                    results_root=output_root.parent,
                    temp_root=temp_root,
                    run_association=f"prompt16:final_oot:attempt_{attempt:03d}",
                    heartbeat_interval_seconds=30.0,
                    max_wall_clock_seconds=None,
                    enforce_memory_limits=True,
                    enforce_process_tree_rss_limit=True,
                    ram_control_policy=ram_policy,
                    status_callback=callback,
                )
                total_active += float(result.active_computation_seconds)
                total_wait += float(result.total_ram_wait_seconds)
                peak_rss = max(peak_rss, int(result.peak_process_tree_rss_bytes))
                if result.minimum_system_available_ram_bytes is not None:
                    minimum_available = (
                        int(result.minimum_system_available_ram_bytes)
                        if minimum_available is None
                        else min(
                            int(minimum_available),
                            int(result.minimum_system_available_ram_bytes),
                        )
                    )
                attempts_root = log_root / "attempts"
                attempts_root.mkdir(parents=True, exist_ok=True)
                summary = result.to_dict()
                samples = summary.pop("samples", [])
                write_json_atomic(
                    attempts_root / f"attempt_{attempt:03d}_supervisor_summary.json",
                    summary,
                )
                import pandas as pd

                write_csv_atomic(
                    attempts_root / f"attempt_{attempt:03d}_resource_samples.csv",
                    pd.DataFrame(samples),
                )
                if result.status == "completed":
                    publish_status(state="FINALIZING", stop_code=None, force=True)
                    worker_marker = output_root / "_WORKER_SUCCESS"
                    evidence = output_root / "final_evidence_manifest.json"
                    if not worker_marker.is_file() or not evidence.is_file():
                        raise Prompt16ExecutionError(
                            "worker completed without final evidence authentication"
                        )
                    marker = json.loads(worker_marker.read_text(encoding="utf-8"))
                    from credit_risk_fs.data.homecredit_model_stability_2024.contract import file_sha256

                    if marker.get("final_evidence_manifest_sha256") != file_sha256(evidence):
                        raise Prompt16ExecutionError("worker final evidence marker mismatch")
                    publish_status(state="DONE", stop_code=None, force=True)
                    write_text_atomic(
                        output_root / "_SUCCESS",
                        json.dumps(marker, sort_keys=True) + "\n",
                        overwrite=False,
                    )
                    session.finish(
                        "session_completed",
                        message="Final Prompt-16 34-cell OOT lifecycle completed and authenticated",
                    )
                    break

                resource_failure = result.stop_code in {
                    "ram_process_limit",
                    "ram_system_headroom",
                    "ram_pause_unavailable",
                } or (
                    result.worker_error is not None
                    and any(
                        token in result.worker_error.lower()
                        for token in ("memoryerror", "out of memory", "bad_alloc", "oom")
                    )
                )
                scope = str(result.final_fold_id or result.final_stage or "unknown_scope")
                cleanup_ok = (
                    result.child_cleanup_confirmed
                    and result.queue_cleanup_confirmed
                    and not result.survivor_processes
                )
                if not resource_failure or not cleanup_ok:
                    code = result.stop_code or "worker_failure"
                    publish_status(state="ERROR", stop_code=code, force=True)
                    session.finish(
                        "session_failed",
                        level="ERROR",
                        message="Final OOT stopped safely; identical command remains resumable",
                        stop_code=code,
                    )
                    return 2
                invocation_retries[scope] = invocation_retries.get(scope, 0) + 1
                lifetime_retries[scope] = lifetime_retries.get(scope, 0) + 1
                global_retries += 1
                emit_research_event(
                    "resource_recovery_required",
                    level="WARNING",
                    message=(
                        f"same incomplete checkpoint will be recovered; retry "
                        f"{invocation_retries[scope]}/{MAX_RESOURCE_RECOVERY_RESTARTS_PER_SCOPE}"
                    ),
                    priority=True,
                    fold_id=scope,
                    stage=result.final_stage,
                    stop_code=result.stop_code,
                    retry_count=invocation_retries[scope],
                    retry_limit=MAX_RESOURCE_RECOVERY_RESTARTS_PER_SCOPE,
                )
                publish_status(
                    state="RESOURCE_RECOVERY_REQUIRED",
                    stop_code=result.stop_code,
                    sample={"fold_id": scope, "stage": result.final_stage},
                    force=True,
                )
                if invocation_retries[scope] > MAX_RESOURCE_RECOVERY_RESTARTS_PER_SCOPE:
                    publish_status(
                        state="ERROR",
                        stop_code="resource_recovery_retry_ceiling",
                        force=True,
                    )
                    session.finish(
                        "session_controlled_stop",
                        message=(
                            "Per-scope recovery ceiling reached; rerun the identical command "
                            "after machine conditions recover"
                        ),
                        stop_code="resource_recovery_retry_ceiling",
                    )
                    return 2
                backoff = min(30.0, float(2 ** (invocation_retries[scope] - 1)))
                time.sleep(backoff)
                emit_research_event(
                    "resuming_from_checkpoint",
                    message="RAM recovered; resuming the same incomplete authenticated scope",
                    priority=True,
                    fold_id=scope,
                    stage=result.final_stage,
                    retry_count=invocation_retries[scope],
                )
                publish_status(
                    state="RESUMING_FROM_CHECKPOINT",
                    stop_code=None,
                    sample={"fold_id": scope, "stage": result.final_stage},
                    force=True,
                )
    finally:
        release_execution_lock(lock)

    print(
        json.dumps(
            {
                "operation": "prompt_16_final_amended_oot_once",
                "status": "completed",
                "stop_code": None,
                "expected_evaluations": 34,
                "accounting": _count_cells(output_root),
                "automatic_resource_retry_count": global_retries,
                "peak_process_tree_rss_bytes": peak_rss,
                "minimum_system_available_ram_bytes": minimum_available,
                "output_root": str(output_root),
                "controller_status": str(status_path),
                "terminal_log": str(terminal_log),
                "success_marker": str(output_root / "_SUCCESS"),
                "evidence_manifest": str(
                    output_root / "final_evidence_manifest.json"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
