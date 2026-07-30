"""Fail-closed authorization for restarting a controlled timed-out cell."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from credit_risk_fs.experiments.atomic_io import sha256_file


AUTHORIZATION_SCHEMA_VERSION = "full_baseline_timeout_recovery_v2"
VALIDATOR_VERSION = "full_baseline_timeout_resume_validator_v2"
RESUMABLE_FROM_CELL_BOUNDARY = "RESUMABLE_FROM_CELL_BOUNDARY"
NOT_RESUMABLE = "NOT_RESUMABLE"
DEFAULT_AUTHORIZATION_PATH = Path(
    "configs/execution/full_baseline_timeout_recovery_cell_004_attempt_02_v1.json"
)


@dataclass(frozen=True, slots=True)
class TimeoutResumeValidation:
    decision: str
    run_id: str
    cell_id: str
    historical_terminal_state: str | None
    historical_stop_reason: str | None
    intended_restart_boundary: str
    checks: tuple[dict[str, Any], ...]
    reasons: tuple[str, ...]
    validation_timestamp_utc: str
    validator_version: str
    authorization_path: str | None
    authorization_sha256: str | None
    partial_artifacts: tuple[str, ...]
    active_processes: tuple[dict[str, Any], ...]
    lock_paths: tuple[str, ...]
    checkpoint_identity: dict[str, Any] | None

    @property
    def resumable(self) -> bool:
        return self.decision == RESUMABLE_FROM_CELL_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON control file is not a mapping: {path}")
    return dict(payload)


def _git_state(root: Path) -> dict[str, Any]:
    def command(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(command("status", "--porcelain", "--untracked-files=normal")),
    }


def inspect_research_processes(
    repository_root: str | Path,
    *,
    process_records: Iterable[Mapping[str, Any]] | None = None,
    current_invocation_process_ids: Iterable[int] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return active parent/orphan worker candidates without mutating processes."""

    root = Path(repository_root).resolve()
    records: list[dict[str, Any]] = []
    ignored_process_ids = (
        {int(pid) for pid in current_invocation_process_ids}
        if current_invocation_process_ids is not None
        else {os.getpid()}
    )
    if process_records is None:
        import psutil

        current = psutil.Process(os.getpid())
        while current is not None:
            ignored_process_ids.add(int(current.pid))
            try:
                current = current.parent()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                break
        supplied: list[dict[str, Any]] = []
        for process in psutil.process_iter(["pid", "ppid", "name", "cmdline"]):
            if int(process.pid) in ignored_process_ids:
                continue
            try:
                supplied.append(
                    {
                        "pid": int(process.pid),
                        "ppid": int(process.ppid()),
                        "name": str(process.name()),
                        "cmdline": " ".join(process.cmdline()),
                        "cwd": str(process.cwd()),
                    }
                )
            except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
                continue
        process_records = supplied
    for raw in process_records:
        item = dict(raw)
        try:
            pid = int(item.get("pid"))
        except (TypeError, ValueError):
            continue
        if pid in ignored_process_ids:
            continue
        command = str(item.get("cmdline", ""))
        name = str(item.get("name", ""))
        cwd_value = str(item.get("cwd", ""))
        try:
            same_root = bool(cwd_value) and Path(cwd_value).resolve() == root
        except OSError:
            same_root = False
        parent = "run_full_baseline.py" in command
        spawned = same_root and (
            "multiprocessing.spawn" in command or "spawn_main" in command
        )
        research_module = same_root and "credit_risk_fs" in command
        if not (parent or spawned or research_module):
            continue
        records.append(
            {
                "pid": pid,
                "ppid": item.get("ppid"),
                "name": name,
                "kind": "active_parent" if parent else "orphan_worker_candidate",
                "cmdline": command,
            }
        )
    return tuple(sorted(records, key=lambda value: int(value["pid"])))


def _unfinalized_artifacts(
    run_directory: Path,
    checkpoint: Mapping[str, Any],
) -> tuple[str, ...]:
    finalized = {
        (run_directory / str(relative)).resolve()
        for relative in checkpoint.get("finalized_artifacts", {})
    }
    controls = {
        "checkpoint.json",
        "manifest.json",
        "run_manifest.json",
        "run.log",
        ".execution.lock",
    }
    found = []
    incomplete = run_directory / "incomplete"
    for path in sorted(run_directory.rglob("*")):
        if not path.is_file() or path.is_relative_to(incomplete):
            continue
        if path.resolve() in finalized or path.name in controls:
            continue
        found.append(path.relative_to(run_directory).as_posix())
    return tuple(found)


def _largest_resource_sample_gap(resource: Mapping[str, Any]) -> dict[str, float]:
    samples = resource.get("samples", [])
    if not isinstance(samples, list) or len(samples) < 2:
        return {}
    largest: dict[str, float] = {}
    for before, after in zip(samples, samples[1:]):
        try:
            before_elapsed = float(before["elapsed_seconds"])
            after_elapsed = float(after["elapsed_seconds"])
            before_cpu = float(before["process_tree_cpu_seconds"])
            after_cpu = float(after["process_tree_cpu_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        gap = after_elapsed - before_elapsed
        if gap > largest.get("gap_seconds", -1.0):
            largest = {
                "gap_seconds": gap,
                "cpu_growth_seconds": max(0.0, after_cpu - before_cpu),
                "before_elapsed_seconds": before_elapsed,
                "after_elapsed_seconds": after_elapsed,
            }
    return largest


def validate_timeout_resume_authorization(
    *,
    repository_root: str | Path,
    run_directory: str | Path,
    cell: Mapping[str, Any],
    full_baseline_configuration_sha256: str,
    workload_classification: Mapping[str, Any],
    earlier_cells_authenticated: Mapping[str, bool],
    authorization_path: str | Path = DEFAULT_AUTHORIZATION_PATH,
    process_records: Iterable[Mapping[str, Any]] | None = None,
    repository_state: Mapping[str, Any] | None = None,
) -> TimeoutResumeValidation:
    """Evaluate every authorization condition without writing or loading data."""

    root = Path(repository_root).resolve()
    run_dir = Path(run_directory).resolve()
    supplied = Path(authorization_path)
    auth_path = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    checks: list[dict[str, Any]] = []
    reasons: list[str] = []

    def check(name: str, passed: bool, detail: str) -> None:
        value = bool(passed)
        checks.append({"name": name, "passed": value, "detail": str(detail)})
        if not value:
            reasons.append(f"{name}: {detail}")

    authorization: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    checkpoint: dict[str, Any] = {}
    resource: dict[str, Any] = {}
    authorization_hash: str | None = None
    try:
        check(
            "authorization_path_scoped",
            auth_path.is_relative_to(root),
            auth_path.as_posix(),
        )
        authorization = _read_json(auth_path)
        authorization_hash = sha256_file(auth_path)
        check(
            "authorization_schema",
            authorization.get("schema_version") == AUTHORIZATION_SCHEMA_VERSION,
            str(authorization.get("schema_version")),
        )
        check(
            "authorization_validator",
            authorization.get("validator_version") == VALIDATOR_VERSION,
            str(authorization.get("validator_version")),
        )
        check(
            "authorization_decision",
            authorization.get("decision") == RESUMABLE_FROM_CELL_BOUNDARY,
            str(authorization.get("decision")),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        check("authorization_readable", False, f"{type(exc).__name__}: {exc}")

    run_id = str(cell.get("cell_id", ""))
    check("run_directory_scoped", run_dir.is_relative_to(root), str(run_dir))
    check("run_directory_present", run_dir.is_dir(), str(run_dir))
    try:
        manifest = _read_json(run_dir / "manifest.json")
        checkpoint = _read_json(run_dir / "checkpoint.json")
        resource = _read_json(run_dir / "resource_usage.json")
        check("control_files_readable", True, "manifest/checkpoint/resource")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        check("control_files_readable", False, f"{type(exc).__name__}: {exc}")

    check("run_id_matches", run_id == str(authorization.get("run_id", "")), run_id)
    check(
        "cell_id_matches",
        run_id == str(authorization.get("cell_id", "")),
        str(authorization.get("cell_id")),
    )
    check(
        "control_run_ids_match",
        manifest.get("run_id") == run_id and checkpoint.get("run_id") == run_id,
        f"manifest={manifest.get('run_id')}; checkpoint={checkpoint.get('run_id')}",
    )
    expected_cell = dict(authorization.get("cell_identity", {}))
    observed_cell = {
        "dataset": cell.get("dataset"),
        "model": cell.get("model"),
        "selector": cell.get("method_id"),
        "seed": cell.get("seed"),
    }
    check("cell_identity_matches", observed_cell == expected_cell, str(observed_cell))
    check(
        "historical_terminal_state",
        manifest.get("status") == "timed_out"
        and authorization.get("historical_terminal_state") == "timed_out",
        str(manifest.get("status")),
    )
    check(
        "historical_stop_reason",
        manifest.get("stop_code") == "wall_clock_limit"
        and checkpoint.get("stop_code") == "wall_clock_limit"
        and resource.get("stop_code") == "wall_clock_limit",
        f"manifest={manifest.get('stop_code')}; checkpoint={checkpoint.get('stop_code')}",
    )
    lifecycle_states = [str(item.get("state")) for item in checkpoint.get("stop_lifecycle", [])]
    required_lifecycle = {
        "WALL_CLOCK_STOP_LATCHED",
        "COOPERATIVE_STOP_REQUESTED",
        "GRACE_PERIOD",
        "EXIT_CONFIRMED",
        "ARTIFACT_AND_STATE_FINALIZATION",
    }
    check(
        "controlled_supervisor_timeout",
        required_lifecycle.issubset(lifecycle_states)
        and checkpoint.get("primary_stop_code") == "wall_clock_limit",
        ",".join(lifecycle_states),
    )
    cleanup = dict(checkpoint.get("cleanup_evidence", {}))
    check(
        "worker_exit_and_cleanup_confirmed",
        cleanup.get("child_cleanup_confirmed") is True
        and cleanup.get("queue_cleanup_confirmed") is True
        and not cleanup.get("survivor_processes")
        and checkpoint.get("termination_condition") is None
        and manifest.get("worker_exit_code") in {0, 15},
        f"exit={manifest.get('worker_exit_code')}; survivors={len(cleanup.get('survivor_processes', []))}",
    )
    old_limit = float(authorization.get("historical_wall_clock_limit_seconds", 0) or 0)
    active_seconds = float(resource.get("active_computation_seconds", 0) or 0)
    suspension = authorization.get("historical_suspension_evidence")
    suspension_authenticated = False
    if isinstance(suspension, Mapping) and suspension:
        def evidence_number(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return float("nan")

        observed_gap = _largest_resource_sample_gap(resource)
        corrected_active = active_seconds - float(observed_gap.get("gap_seconds", 0.0))
        expected_values = {
            "gap_seconds": suspension.get("largest_sample_gap_seconds"),
            "cpu_growth_seconds": suspension.get("cpu_growth_during_gap_seconds"),
            "before_elapsed_seconds": suspension.get("before_elapsed_seconds"),
            "after_elapsed_seconds": suspension.get("after_elapsed_seconds"),
        }
        exact_evidence = bool(observed_gap) and all(
            abs(
                evidence_number(observed_gap.get(key))
                - evidence_number(expected)
            )
            <= 0.001
            for key, expected in expected_values.items()
        )
        suspension_authenticated = (
            suspension.get("accounting_defect")
            == "windows_sleep_counted_as_active_v1"
            and exact_evidence
            and float(observed_gap.get("gap_seconds", 0.0)) >= 300.0
            and float(observed_gap.get("cpu_growth_seconds", float("inf"))) <= 60.0
            and float(observed_gap.get("before_elapsed_seconds", float("inf")))
            < old_limit
            <= float(observed_gap.get("after_elapsed_seconds", 0.0))
            and corrected_active < old_limit
            and abs(
                corrected_active
                - evidence_number(
                    suspension.get("corrected_active_computation_seconds")
                )
            )
            <= 0.001
            and abs(
                active_seconds
                - evidence_number(
                    authorization.get(
                        "historical_reported_active_computation_seconds"
                    )
                )
            )
            <= 0.001
        )
        check(
            "historical_suspension_evidence",
            suspension_authenticated,
            (
                f"gap={observed_gap.get('gap_seconds')}; "
                f"cpu_growth={observed_gap.get('cpu_growth_seconds')}; "
                f"corrected_active={corrected_active}"
            ),
        )
    check(
        "timeout_reached_configured_limit",
        old_limit > 0
        and (
            old_limit <= active_seconds <= old_limit + 60
            or suspension_authenticated
        ),
        f"active={active_seconds}; limit={old_limit}",
    )
    completed_stages = set(map(str, checkpoint.get("completed_stages", [])))
    check(
        "no_completed_checkpoint",
        checkpoint.get("status") != "completed"
        and "completed" not in completed_stages
        and not (run_dir / "_SUCCESS").exists(),
        f"checkpoint_status={checkpoint.get('status')}",
    )
    check(
        "cell_boundary_restart_supported",
        authorization.get("intended_restart_boundary") == "cell_boundary"
        and checkpoint.get("last_successful_stage") == "data_validated"
        and not checkpoint.get("completed_fold_ids"),
        f"last_stage={checkpoint.get('last_successful_stage')}; folds={checkpoint.get('completed_fold_ids')}",
    )

    expected_identity = dict(authorization.get("checkpoint_identity", {}))
    check(
        "checkpoint_identity_matches",
        bool(expected_identity) and checkpoint.get("identity") == expected_identity,
        str(checkpoint.get("identity", {}).get("resolved_config_hash")),
    )
    check(
        "frozen_configuration_matches",
        authorization.get("full_baseline_configuration_sha256")
        == str(full_baseline_configuration_sha256),
        str(full_baseline_configuration_sha256),
    )
    expected_workload = dict(authorization.get("authorized_workload", {}))
    observed_workload = {
        key: workload_classification.get(key)
        for key in (
            "selector_cost_class",
            "final_model_cost_class",
            "dataset_cost_class",
            "effective_cost_class",
            "effective_wall_clock_limit_seconds",
            "composition_rule",
            "policy_sha256",
        )
    }
    check(
        "authorized_workload_matches",
        observed_workload == expected_workload,
        str(observed_workload),
    )

    for section, base in (
        ("historical_artifact_hashes", root),
        ("authorized_runtime_file_hashes", root),
        ("authorized_scientific_file_hashes", root),
        ("earlier_completed_artifact_hashes", root),
    ):
        expected_files = authorization.get(section, {})
        section_ok = isinstance(expected_files, Mapping) and bool(expected_files)
        mismatches = []
        if section_ok:
            for relative, expected_hash in expected_files.items():
                path = (base / str(relative)).resolve()
                if (
                    not path.is_relative_to(root)
                    or not path.is_file()
                    or sha256_file(path) != str(expected_hash)
                ):
                    mismatches.append(str(relative))
        check(section, section_ok and not mismatches, ",".join(mismatches) or "all hashes match")

    partials = _unfinalized_artifacts(run_dir, checkpoint)
    expected_partials = authorization.get("partial_artifact_hashes", {})
    partial_mismatches = []
    if isinstance(expected_partials, Mapping):
        if set(partials) != set(map(str, expected_partials)):
            partial_mismatches.append("inventory")
        for relative, expected_hash in expected_partials.items():
            path = (run_dir / str(relative)).resolve()
            if not path.is_file() or sha256_file(path) != str(expected_hash):
                partial_mismatches.append(str(relative))
    else:
        partial_mismatches.append("record_not_mapping")
    check(
        "partial_artifacts_are_unfinalized_evidence",
        bool(partials) and not partial_mismatches,
        ",".join(partial_mismatches) or f"count={len(partials)}",
    )

    expected_earlier = tuple(map(str, authorization.get("earlier_completed_run_ids", ())))
    earlier_ok = bool(expected_earlier) and all(
        earlier_cells_authenticated.get(run, False) for run in expected_earlier
    )
    check(
        "earlier_completed_cells_authenticated",
        earlier_ok,
        str({run: earlier_cells_authenticated.get(run, False) for run in expected_earlier}),
    )

    lock_paths = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(run_dir.glob(".execution.lock"))
        if path.is_file()
    )
    check("no_execution_lock", not lock_paths, ",".join(lock_paths) or "none")
    active_processes = inspect_research_processes(
        root, process_records=process_records
    )
    check(
        "no_active_or_orphan_worker",
        not active_processes,
        ",".join(str(item["pid"]) for item in active_processes) or "none",
    )
    try:
        live_repository = dict(repository_state or _git_state(root))
        check(
            "repository_clean",
            live_repository.get("dirty") is False,
            str(live_repository),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        check("repository_clean", False, f"{type(exc).__name__}: {exc}")

    static_checks = authorization.get("validation_checks", [])
    check(
        "authorization_record_checks_passed",
        isinstance(static_checks, list)
        and bool(static_checks)
        and all(item.get("result") == "pass" for item in static_checks),
        f"count={len(static_checks) if isinstance(static_checks, list) else 0}",
    )
    decision = RESUMABLE_FROM_CELL_BOUNDARY if not reasons else NOT_RESUMABLE
    relative_auth = (
        auth_path.relative_to(root).as_posix()
        if auth_path.is_relative_to(root)
        else None
    )
    return TimeoutResumeValidation(
        decision=decision,
        run_id=run_id,
        cell_id=run_id,
        historical_terminal_state=(
            str(manifest.get("status")) if manifest.get("status") is not None else None
        ),
        historical_stop_reason=(
            str(manifest.get("stop_code")) if manifest.get("stop_code") is not None else None
        ),
        intended_restart_boundary="cell_boundary",
        checks=tuple(checks),
        reasons=tuple(reasons),
        validation_timestamp_utc=_utc_now(),
        validator_version=VALIDATOR_VERSION,
        authorization_path=relative_auth,
        authorization_sha256=authorization_hash,
        partial_artifacts=partials,
        active_processes=active_processes,
        lock_paths=lock_paths,
        checkpoint_identity=(dict(checkpoint["identity"]) if checkpoint.get("identity") else None),
    )


__all__ = [
    "AUTHORIZATION_SCHEMA_VERSION",
    "DEFAULT_AUTHORIZATION_PATH",
    "NOT_RESUMABLE",
    "RESUMABLE_FROM_CELL_BOUNDARY",
    "TimeoutResumeValidation",
    "VALIDATOR_VERSION",
    "inspect_research_processes",
    "validate_timeout_resume_authorization",
]
