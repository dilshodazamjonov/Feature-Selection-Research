from __future__ import annotations

import io
import json
import logging
import multiprocessing
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import pytest

from credit_risk_fs.experiments.research_logging import (
    LOG_SCHEMA_VERSION,
    ResearchLogSession,
    bind_research_context,
    emit_contextual_research_event,
    emit_research_event,
    logged_stage,
    suppress_third_party_output,
)
from credit_risk_fs.experiments.resource_monitor import (
    DISK_RESULTS_LIMIT,
    RAM_PROCESS_LIMIT,
    ResourceSample,
    _OwnedProcessRegistry,
    supervise_worker,
)
from credit_risk_fs.experiments.resource_policy import (
    DiskPolicy,
    GpuPolicy,
    MemoryPolicy,
    MonitoringPolicy,
    ParallelismPolicy,
    ResolvedExecutionPolicy,
)


AUDIT_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
)
HUMAN_LINE_PATTERN = re.compile(
    r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC\] "
    r"(?:START|INFO|ACTIVE|DONE|WARN|STOP|ERROR) +\| .+$"
)


def _audit_path(log_path: Path) -> Path:
    return log_path.with_name("events.jsonl")


def _debug_path(log_path: Path) -> Path:
    return log_path.with_name("debug.log")


def _records(log_path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in _audit_path(log_path).read_text(encoding="utf-8").splitlines()
    ]


def _human_lines(log_path: Path) -> list[str]:
    return log_path.read_text(encoding="utf-8").splitlines()


def _policy(*, warn_ram: float = 1.0, abort_ram: float = 2.0) -> ResolvedExecutionPolicy:
    return ResolvedExecutionPolicy(
        schema_version="resource_safe_execution_policy_v1",
        profile_name="synthetic_logging_test",
        parallelism=ParallelismPolicy(1, 1, 0, 1, False),
        memory=MemoryPolicy(1.0, warn_ram, abort_ram, 0.001, 1.35),
        gpu=GpuPolicy(0.1, 1.0, 2.0, True),
        disk=DiskPolicy(0.001, 0.001, 2.5),
        monitoring=MonitoringPolicy(0.02, 0.08, 0.15),
        configured_policy_path="synthetic-test",
    )


class _DelayedHighMemorySampler:
    def __init__(self, **_kwargs):
        self.gpu = None
        self.sample_count = 0

    def sample(self, worker_pid, *, stage, fold_id):
        gib = 1024**3
        self.sample_count += 1
        return ResourceSample(
            elapsed_seconds=0.02,
            worker_pid=int(worker_pid),
            child_pids=(),
            process_tree_rss_bytes=int(
                (0.01 if self.sample_count < 5 else 0.2) * gib
            ),
            system_available_ram_bytes=10 * gib,
            process_gpu_bytes=None,
            results_free_disk_bytes=(100 * gib if self.sample_count < 5 else 0),
            temp_free_disk_bytes=100 * gib,
            process_tree_cpu_percent=1.0,
            process_tree_cpu_seconds=0.01,
            stage=stage,
            fold_id=fold_id,
        )


def test_human_format_append_audit_separation_and_immediate_flush(tmp_path):
    log_path = tmp_path / "logs" / "runs.log"
    terminal = io.StringIO()
    session_ids = []
    for invocation in range(2):
        with ResearchLogSession(
            Path("logs/runs.log"),
            repository_root=tmp_path,
            command_arguments=["--plan", str(invocation)],
            terminal_stream=terminal,
        ) as session:
            session_ids.append(session.session_id)
            with bind_research_context(
                run_id="cdv1-014-lendingclub-v2-voting-k100-catboost-s42",
                dataset="lendingclub_v2",
                model="catboost",
                seed=42,
                phase="DEV",
                fold_id=3,
            ):
                emit_research_event(
                    "run_execution_started",
                    message="run started",
                    selector="rank_voting_v1",
                )
                emit_research_event(
                    "stage_started",
                    message="Boruta started",
                    stage="voter_boruta",
                    component="boruta",
                )
                emit_research_event(
                    "stage_heartbeat",
                    message="Boruta fit active; internal iteration unavailable.",
                    stage="voter_boruta",
                    component="boruta",
                    elapsed_stage_seconds=30,
                    worker_rss_bytes=int(6.4 * 1024**3),
                    system_available_ram_bytes=int(18.2 * 1024**3),
                )
                emit_research_event(
                    "stage_completed",
                    message="Boruta completed",
                    stage="voter_boruta",
                    component="boruta",
                    elapsed_stage_seconds=805,
                )
                assert "Boruta started" in log_path.read_text(encoding="utf-8")
                assert "stage_started" in _audit_path(log_path).read_text(encoding="utf-8")
            session.finish("session_completed", message="Synthetic session completed")

    human_lines = _human_lines(log_path)
    audit_records = _records(log_path)
    human_text = "\n".join(human_lines)
    assert human_lines
    assert all(HUMAN_LINE_PATTERN.match(line) for line in human_lines)
    assert not any(line.lstrip().startswith("{") for line in human_lines)
    assert all(field not in human_text for field in ("session_id", "schema_version", '"pid"'))
    assert "START | Run 014 | LendingClub | Voting K=100 | CatBoost" in human_text
    assert "INFO  | Fold 3/5 | Boruta started" in human_text
    assert "ACTIVE | Fold 3/5 | Boruta running | Elapsed 30s | RAM 6.4 GiB | Available 18.2 GiB" in human_text
    assert "DONE  | Fold 3/5 | Boruta completed in 13m 25s" in human_text
    assert terminal.getvalue().splitlines() == human_lines
    assert len(set(session_ids)) == 2
    assert {str(item["session_id"]) for item in audit_records} == set(session_ids)
    assert all(item["schema_version"] == LOG_SCHEMA_VERSION for item in audit_records)
    assert all(
        AUDIT_TIMESTAMP_PATTERN.match(str(item["timestamp_utc"]))
        for item in audit_records
    )


def test_contextual_event_merge_has_safe_explicit_precedence(tmp_path):
    log_path = tmp_path / "logs" / "runs.log"
    with ResearchLogSession(
        Path("logs/runs.log"),
        repository_root=tmp_path,
        command_arguments=[],
        terminal_stream=io.StringIO(),
    ) as session:
        emitted = emit_contextual_research_event(
            "stage_started",
            {
                "pilot_cell": "worker-stage-value",
                "dataset": "worker-stage-value",
                "message": "must-not-shadow-event-message",
                "priority": False,
            },
            {
                "pilot_cell": "authenticated-supervisor-value",
                "dataset": "authenticated-supervisor-value",
            },
            message="Merged event emitted",
            priority=True,
            dataset="explicit-event-value",
        )
        assert emitted
        session.finish("session_completed", message="Synthetic session completed")

    event = next(item for item in _records(log_path) if item["event"] == "stage_started")
    assert event["message"] == "Merged event emitted"
    assert event["pilot_cell"] == "authenticated-supervisor-value"
    assert event["dataset"] == "explicit-event-value"


def test_existing_json_run_log_is_migrated_without_losing_audit_events(tmp_path):
    log_path = tmp_path / "logs" / "runs.log"
    log_path.parent.mkdir(parents=True)
    legacy = {
        "schema_version": LOG_SCHEMA_VERSION,
        "timestamp_utc": "2026-07-24T06:20:10.000Z",
        "level": "INFO",
        "pid": 123,
        "session_id": "legacy-session",
        "event": "run_execution_started",
        "message": "legacy start",
        "run_id": "cdv1-014-lendingclub-v2-voting-k100-catboost-s42",
        "dataset": "lendingclub_v2",
        "model": "catboost",
    }
    log_path.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    with ResearchLogSession(
        log_path,
        repository_root=tmp_path,
        command_arguments=[],
        terminal_stream=io.StringIO(),
    ) as session:
        session.finish("session_completed", message="New session completed")
    human = log_path.read_text(encoding="utf-8")
    records = _records(log_path)
    assert not any(line.lstrip().startswith("{") for line in human.splitlines())
    assert "Run 014 | LendingClub | Voting K=100 | CatBoost" in human
    assert any(item["session_id"] == "legacy-session" for item in records)
    assert any(item["event"] == "session_completed" for item in records)


def test_python_logging_is_audited_once_and_root_state_is_restored(tmp_path):
    log_path = tmp_path / "runs.log"
    root = logging.getLogger()
    original_level = root.level
    logger = logging.getLogger("research_logging.single_record")
    logger.setLevel(logging.INFO)
    logger.propagate = True
    with ResearchLogSession(
        log_path,
        repository_root=tmp_path,
        command_arguments=[],
        terminal_stream=io.StringIO(),
    ) as session:
        logger.info("one canonical message")
        session.finish("session_completed", message="done")
    matches = [
        item
        for item in _records(log_path)
        if item["event"] == "python_log" and item["message"] == "one canonical message"
    ]
    assert len(matches) == 1
    assert "one canonical message" not in log_path.read_text(encoding="utf-8")
    assert root.level == original_level


def test_traceback_is_separate_and_keyboard_interrupt_is_short(tmp_path):
    error_log_path = tmp_path / "error" / "runs.log"
    error_terminal = io.StringIO()
    with ResearchLogSession(
        error_log_path,
        repository_root=tmp_path,
        command_arguments=[],
        terminal_stream=error_terminal,
    ) as session:
        with pytest.raises(ValueError, match="synthetic failure"):
            with logged_stage("failure", message="failure stage"):
                raise ValueError("synthetic failure")
        session.finish(
            "session_failed",
            level="ERROR",
            message="Unexpected research error: synthetic failure",
            exception_class="ValueError",
        )

    interrupt_log_path = tmp_path / "interrupt" / "runs.log"
    interrupt_terminal = io.StringIO()
    with ResearchLogSession(
        interrupt_log_path,
        repository_root=tmp_path,
        command_arguments=[],
        terminal_stream=interrupt_terminal,
    ) as session:
        with pytest.raises(KeyboardInterrupt, match="synthetic interrupt"):
            with logged_stage("interrupt", message="interrupt stage"):
                raise KeyboardInterrupt("synthetic interrupt")
        session.finish(
            "session_interrupted",
            level="ERROR",
            message="Research run interrupted manually",
            exception_class="KeyboardInterrupt",
        )

    unexpected_log_path = tmp_path / "unexpected" / "runs.log"
    unexpected_terminal = io.StringIO()
    with ResearchLogSession(
        unexpected_log_path,
        repository_root=tmp_path,
        command_arguments=[],
        terminal_stream=unexpected_terminal,
    ) as session:
        try:
            raise RuntimeError("unexpected synthetic error")
        except RuntimeError as exc:
            session.finish(
                "session_failed",
                level="ERROR",
                message=f"Unexpected research error: {exc}",
                exception_class=type(exc).__name__,
                traceback=traceback.format_exc(),
            )
    all_audit_records = (
        _records(error_log_path)
        + _records(interrupt_log_path)
        + _records(unexpected_log_path)
    )
    assert all("traceback" not in item for item in all_audit_records)

    error_debug = _debug_path(error_log_path).read_text(encoding="utf-8")
    error_human = error_log_path.read_text(encoding="utf-8")
    assert "ValueError: synthetic failure" in error_debug
    assert error_human.count("ERROR | Unexpected research error: synthetic failure") == 1

    interrupt_human = interrupt_log_path.read_text(encoding="utf-8")
    interrupt_debug = _debug_path(interrupt_log_path).read_text(encoding="utf-8")
    assert interrupt_human.count("STOP  | Research run interrupted manually") == 1
    assert "KeyboardInterrupt" not in interrupt_debug
    assert "Traceback (most recent call last)" not in interrupt_terminal.getvalue()

    unexpected_human = unexpected_log_path.read_text(encoding="utf-8")
    unexpected_debug = _debug_path(unexpected_log_path).read_text(encoding="utf-8")
    assert "RuntimeError: unexpected synthetic error" in unexpected_debug
    assert unexpected_human.count(
        "ERROR | Unexpected research error: unexpected synthetic error"
    ) == 1
    assert "Details: unexpected/debug.log" in unexpected_human
    assert "Traceback (most recent call last)" not in unexpected_terminal.getvalue()


def test_synchronous_heartbeat_is_human_readable_and_thread_is_closed(tmp_path):
    log_path = tmp_path / "runs.log"
    with ResearchLogSession(
        log_path,
        repository_root=tmp_path,
        command_arguments=[],
        terminal_stream=io.StringIO(),
    ) as session:
        with logged_stage(
            "synchronous_slow_stage",
            message="slow parent stage",
            heartbeat_interval_seconds=0.02,
        ):
            time.sleep(0.065)
        session.finish("session_completed", message="done")
    records = _records(log_path)
    assert any(
        item["event"] == "stage_heartbeat"
        and item.get("stage") == "synchronous_slow_stage"
        for item in records
    )
    assert "ACTIVE | Synchronous Slow Stage running" in log_path.read_text(encoding="utf-8")
    assert not any(
        thread.name.startswith("research-stage-heartbeat-")
        for thread in threading.enumerate()
    )


def test_parent_heartbeat_and_truthful_boruta_component_records(tmp_path):
    log_path = tmp_path / "runs.log"
    with ResearchLogSession(
        log_path,
        repository_root=tmp_path,
        command_arguments=[],
        terminal_stream=io.StringIO(),
    ) as session:
        result = supervise_worker(
            worker_target="credit_risk_fs.experiments.synthetic_execution:mock_slow_boruta_worker",
            worker_kwargs={"duration_seconds": 0.18},
            policy=_policy(),
            results_root=tmp_path,
            temp_root=tmp_path,
            run_association="synthetic-boruta",
            heartbeat_interval_seconds=0.04,
        )
        session.finish("session_completed", message="done")
    assert result.status == "completed"
    records = _records(log_path)
    assert any(
        item["event"] == "component_started"
        and item.get("component") == "boruta"
        and item.get("internal_iteration_available") is False
        for item in records
    )
    assert any(
        item["event"] == "stage_heartbeat"
        and item.get("stage") == "voter_boruta"
        and item["message"] == "Boruta fit active; internal iteration unavailable."
        for item in records
    )
    completion = next(
        item
        for item in records
        if item["event"] == "component_completed" and item.get("component") == "boruta"
    )
    assert (completion["confirmed"], completion["tentative"], completion["rejected"]) == (4, 2, 6)
    assert "ACTIVE | Fold 5/5 | Boruta running" in log_path.read_text(encoding="utf-8")


def test_model_fit_worker_exception_and_interrupt_are_durable(tmp_path):
    log_path = tmp_path / "runs.log"
    targets = (
        "credit_risk_fs.experiments.synthetic_execution:mock_model_fit_worker",
        "credit_risk_fs.experiments.synthetic_execution:exception_worker",
        "credit_risk_fs.experiments.synthetic_execution:keyboard_interrupt_worker",
    )
    with ResearchLogSession(
        log_path,
        repository_root=tmp_path,
        command_arguments=[],
        terminal_stream=io.StringIO(),
    ) as session:
        results = [
            supervise_worker(
                worker_target=target,
                worker_kwargs={},
                policy=_policy(),
                results_root=tmp_path,
                temp_root=tmp_path,
                run_association=f"synthetic-{index}",
                heartbeat_interval_seconds=0.04,
            )
            for index, target in enumerate(targets)
        ]
        session.finish("session_completed", message="done")
    assert [item.status for item in results] == ["completed", "failed", "interrupted"]
    records = _records(log_path)
    worker_failure = next(item for item in records if item["event"] == "worker_failed")
    worker_interrupt = next(item for item in records if item["event"] == "worker_interrupted")
    debug = _debug_path(log_path).read_text(encoding="utf-8")
    assert worker_failure["debug_log_path"] == "debug.log"
    assert "traceback" not in worker_failure
    assert "traceback" not in worker_interrupt
    assert "RuntimeError: synthetic worker failure" in debug
    assert "KeyboardInterrupt" not in debug
    assert any(
        item["event"] == "stage_started" and item.get("stage") == "final_model_fit"
        for item in records
    )
    assert any(
        item["event"] == "stage_completed" and item.get("stage") == "final_model_fit"
        for item in records
    )


def test_non_ram_resource_abort_and_force_kill_lifecycle_are_human_and_durable(
    tmp_path, monkeypatch
):
    log_path = tmp_path / "runs.log"
    monkeypatch.setattr(
        _OwnedProcessRegistry,
        "terminate_phase",
        lambda self, *, timeout_seconds: self.alive(),
    )
    with ResearchLogSession(
        log_path,
        repository_root=tmp_path,
        command_arguments=[],
        terminal_stream=io.StringIO(),
    ) as session:
        result = supervise_worker(
            worker_target="credit_risk_fs.experiments.synthetic_execution:uncooperative_wait_worker",
            worker_kwargs={},
            policy=_policy(warn_ram=0.05, abort_ram=0.08),
            results_root=tmp_path,
            temp_root=tmp_path,
            sampler_factory=_DelayedHighMemorySampler,
            run_association="synthetic-resource-abort",
            heartbeat_interval_seconds=0.04,
        )
        session.finish("session_completed", message="done")
    assert result.status == "aborted_resource_limit"
    assert result.stop_code == DISK_RESULTS_LIMIT
    assert result.child_cleanup_confirmed
    records = _records(log_path)
    human = log_path.read_text(encoding="utf-8")
    lifecycle_states = {
        item.get("lifecycle_state")
        for item in records
        if item["event"] == "supervisor_lifecycle"
    }
    assert {
        "RESOURCE_STOP_LATCHED",
        "COOPERATIVE_STOP_REQUESTED",
        "FORCE_KILL_REMAINDERS",
        "EXIT_CONFIRMED",
    } <= lifecycle_states
    assert "WARN  | disk safety limit reached" in human
    assert "STOP  | disk safety limit reached" in human
    assert "WARN  | Force-stopping remaining worker processes" in human
    assert "DONE  | Research worker cleanup confirmed" in human


def test_concurrent_worker_audit_is_valid_json_and_backpressure_is_reported(tmp_path):
    log_path = tmp_path / "runs.log"
    context = multiprocessing.get_context("spawn")
    with ResearchLogSession(
        log_path,
        repository_root=tmp_path,
        command_arguments=[],
        terminal_stream=io.StringIO(),
        queue_capacity=8,
    ) as session:
        processes = [
            context.Process(
                target=__import__(
                    "credit_risk_fs.experiments.synthetic_execution",
                    fromlist=["logging_transport_process"],
                ).logging_transport_process,
                args=(*session.worker_transport(), f"p{index}", 2_000),
            )
            for index in range(3)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=20)
            assert process.exitcode == 0
            process.close()
        session.finish("session_completed", message="done")
    records = _records(log_path)
    assert records
    assert all(isinstance(item, dict) for item in records)
    assert any(item["event"] == "synthetic_transport_record" for item in records)
    backpressure = [item for item in records if item["event"] == "logging_backpressure"]
    assert backpressure
    assert int(backpressure[-1]["routine_records_dropped"]) > 0
    assert not session.listener_alive
    assert not any(line.lstrip().startswith("{") for line in _human_lines(log_path))


def test_third_party_stdout_and_stderr_are_suppressed(capsys):
    with suppress_third_party_output():
        print("noisy model progress")
        print("noisy warning", file=sys.stderr)
    captured = capsys.readouterr()
    assert "noisy" not in captured.out
    assert "noisy" not in captured.err


def test_runtime_logs_have_narrow_repository_ignore_rules():
    repository_root = Path(__file__).resolve().parents[1]
    ignore_lines = (repository_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert {"/logs/runs.log", "/logs/events.jsonl", "/logs/debug.log"} <= set(
        ignore_lines
    )
    assert "logs/" not in ignore_lines
    tracked_runtime_logs = subprocess.run(
        [
            "git",
            "ls-files",
            "--",
            "logs/runs.log",
            "logs/events.jsonl",
            "logs/debug.log",
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert tracked_runtime_logs == ""
    for relative in ("logs/runs.log", "logs/events.jsonl", "logs/debug.log"):
        assert not any(
            relative in path.as_posix()
            for path in (repository_root / "results").rglob("*")
        )
        for control_name in ("checkpoint.json", "manifest.json"):
            for path in (repository_root / "results" / "runs").rglob(control_name):
                assert relative not in path.read_text(encoding="utf-8")


def test_starting_default_logging_session_keeps_clean_git_worktree(tmp_path):
    (tmp_path / ".gitignore").write_text(
        "/logs/runs.log\n/logs/events.jsonl\n/logs/debug.log\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Logging Test",
            "-c",
            "user.email=logging-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture baseline",
        ],
        cwd=tmp_path,
        check=True,
    )
    with ResearchLogSession(
        Path("logs/runs.log"),
        repository_root=tmp_path,
        command_arguments=[],
        terminal_stream=io.StringIO(),
    ) as session:
        session.finish("session_completed", message="clean-worktree fixture complete")
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert status == ""
    assert (tmp_path / "logs" / "runs.log").is_file()
    assert (tmp_path / "logs" / "events.jsonl").is_file()
    assert (tmp_path / "logs" / "debug.log").is_file()
