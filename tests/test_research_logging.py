from __future__ import annotations

import io
import json
import logging
import multiprocessing
import re
import threading
import time
from pathlib import Path

import pytest

from credit_risk_fs.experiments.research_logging import (
    LOG_SCHEMA_VERSION,
    ResearchLogSession,
    bind_research_context,
    emit_research_event,
    logged_stage,
)
from credit_risk_fs.experiments.resource_monitor import (
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


TIMESTAMP_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


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
            results_free_disk_bytes=100 * gib,
            temp_free_disk_bytes=100 * gib,
            process_tree_cpu_percent=1.0,
            process_tree_cpu_seconds=0.01,
            stage=stage,
            fold_id=fold_id,
        )


def test_append_only_jsonl_schema_context_and_immediate_flush(tmp_path):
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
                run_id="synthetic-run",
                dataset="synthetic-data",
                model="synthetic-model",
                seed=42,
                phase="DEV",
                fold_id=5,
            ):
                emit_research_event(
                    "synthetic_observation",
                    message="record is immediately durable",
                    stage="unit_test",
                    component="logging",
                    scientific_payload=object(),
                )
                assert "synthetic_observation" in log_path.read_text(encoding="utf-8")
            session.finish("session_completed", message="Synthetic session completed")

    records = _records(log_path)
    assert len(records) >= 8
    assert len(set(session_ids)) == 2
    assert {str(item["session_id"]) for item in records} == set(session_ids)
    assert all(item["schema_version"] == LOG_SCHEMA_VERSION for item in records)
    assert all(TIMESTAMP_PATTERN.match(str(item["timestamp_utc"])) for item in records)
    assert all({"level", "pid", "event", "message"} <= item.keys() for item in records)
    observation = next(item for item in records if item["event"] == "synthetic_observation")
    assert observation["scientific_payload"] == "<object omitted>"
    assert observation["phase"] == "DEV"
    assert terminal.getvalue().count("synthetic_observation") == 2


def test_python_logging_is_bridged_once_and_root_state_is_restored(tmp_path):
    log_path = tmp_path / "runs.log"
    terminal = io.StringIO()
    root = logging.getLogger()
    original_level = root.level
    logger = logging.getLogger("research_logging.single_record")
    logger.setLevel(logging.INFO)
    logger.propagate = True
    with ResearchLogSession(
        log_path,
        repository_root=tmp_path,
        command_arguments=[],
        terminal_stream=terminal,
    ) as session:
        logger.info("one canonical message")
        session.finish("session_completed", message="done")
    matches = [
        item
        for item in _records(log_path)
        if item["event"] == "python_log" and item["message"] == "one canonical message"
    ]
    assert len(matches) == 1
    assert root.level == original_level


def test_logged_stage_records_success_failure_and_interrupt_tracebacks(tmp_path):
    log_path = tmp_path / "runs.log"
    with ResearchLogSession(
        log_path,
        repository_root=tmp_path,
        command_arguments=[],
        terminal_stream=io.StringIO(),
    ) as session:
        with logged_stage("success", message="success stage"):
            pass
        with logged_stage(
            "synchronous_slow_stage",
            message="slow parent stage",
            heartbeat_interval_seconds=0.02,
        ):
            time.sleep(0.065)
        with pytest.raises(ValueError, match="synthetic failure"):
            with logged_stage("failure", message="failure stage"):
                raise ValueError("synthetic failure")
        with pytest.raises(KeyboardInterrupt, match="synthetic interrupt"):
            with logged_stage("interrupt", message="interrupt stage"):
                raise KeyboardInterrupt("synthetic interrupt")
        session.finish("session_completed", message="done")
    records = _records(log_path)
    failed = next(item for item in records if item.get("stage") == "failure" and item["event"] == "stage_failed")
    interrupted = next(item for item in records if item.get("stage") == "interrupt" and item["event"] == "stage_interrupted")
    assert "ValueError: synthetic failure" in str(failed["traceback"])
    assert "KeyboardInterrupt: synthetic interrupt" in str(interrupted["traceback"])
    assert any(
        item["event"] == "stage_heartbeat"
        and item.get("stage") == "synchronous_slow_stage"
        for item in records
    )
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
    assert "RuntimeError: synthetic worker failure" in str(worker_failure["traceback"])
    assert "KeyboardInterrupt: synthetic worker interrupt" in str(worker_interrupt["traceback"])
    assert any(item["event"] == "stage_started" and item.get("stage") == "final_model_fit" for item in records)
    assert any(item["event"] == "stage_completed" and item.get("stage") == "final_model_fit" for item in records)


def test_resource_abort_and_force_kill_lifecycle_are_durable(tmp_path, monkeypatch):
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
    assert result.stop_code == RAM_PROCESS_LIMIT
    assert result.child_cleanup_confirmed
    records = _records(log_path)
    assert any(item["event"] == "resource_warning" for item in records)
    assert any(item["event"] == "stage_heartbeat" for item in records)
    lifecycle_states = {
        item.get("lifecycle_state")
        for item in records
        if item["event"] == "supervisor_lifecycle"
    }
    assert {"RESOURCE_STOP_LATCHED", "COOPERATIVE_STOP_REQUESTED", "FORCE_KILL_REMAINDERS", "EXIT_CONFIRMED"} <= lifecycle_states
    assert any(item["event"] == "stage_aborted" and item.get("stop_code") == RAM_PROCESS_LIMIT for item in records)


def test_concurrent_worker_records_are_valid_json_and_backpressure_is_reported(tmp_path):
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


def test_runtime_log_has_a_narrow_repository_ignore_rule():
    repository_root = Path(__file__).resolve().parents[1]
    ignore_lines = (repository_root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/logs/runs.log" in ignore_lines
    assert "logs/" not in ignore_lines
    assert not any("logs/runs.log" in path.as_posix() for path in (repository_root / "results").rglob("*"))
    for control_name in ("checkpoint.json", "manifest.json"):
        for path in (repository_root / "results" / "runs").rglob(control_name):
            assert "logs/runs.log" not in path.read_text(encoding="utf-8")
