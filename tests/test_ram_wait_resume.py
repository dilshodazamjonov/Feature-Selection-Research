from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from time import monotonic

import pandas as pd
import pytest

from credit_risk_fs.data.loaders import DataLoader
from credit_risk_fs.experiments.ram_control import (
    RamWaitState,
    resolve_ram_control_policy,
    wait_for_ram_ready,
)
from credit_risk_fs.experiments.full_baseline_ram_bridge import (
    authenticate_full_baseline_ram_bridge,
)
from credit_risk_fs.experiments.research_logging import (
    ResearchLogSession,
    emit_research_event,
)
from credit_risk_fs.experiments.resource_monitor import (
    MEMORY_ERROR,
    ResourceSample,
    supervise_worker,
    wait_for_inter_run_readiness,
)
from credit_risk_fs.experiments.resource_policy import (
    DiskPolicy,
    GpuPolicy,
    MemoryPolicy,
    MonitoringPolicy,
    ParallelismPolicy,
    ResolvedExecutionPolicy,
)


GIB = 1024**3


def _legacy_policy() -> ResolvedExecutionPolicy:
    return ResolvedExecutionPolicy(
        schema_version="resource_safe_execution_policy_v1",
        profile_name="synthetic",
        parallelism=ParallelismPolicy(1, 1, 0, 1, False),
        memory=MemoryPolicy(10, 24, 28, 8, 1.35),
        gpu=GpuPolicy(0.1, 1, 2, True),
        disk=DiskPolicy(0.001, 0.001, 2.5),
        monitoring=MonitoringPolicy(0.01, 0.2, 0.2),
        configured_policy_path="synthetic",
    )


def _ram_policy(*, check: float = 0.01, log: float = 300.0):
    return resolve_ram_control_policy(
        {
            "schema_version": "ram_wait_resume_policy_v1",
            "profile_name": "synthetic",
            "emergency_min_available_ram_gb": 1,
            "emergency_total_ram_fraction": 0.02,
            "recovery_available_ram_gb": 4,
            "recovery_consecutive_checks": 3,
            "check_interval_seconds": check,
            "log_interval_seconds": log,
            "opaque_stage_pause_mode": "process_tree_suspend",
        },
        total_physical_ram_bytes=32 * GIB,
    )


class _ClosedGpu:
    def close(self):
        return None


class _SequenceSampler:
    def __init__(self, available_gib):
        self.available = list(available_gib)
        self.index = 0
        self.started = monotonic()
        self.gpu = _ClosedGpu()

    def sample(self, worker_pid, *, stage=None, fold_id=None):
        index = min(self.index, len(self.available) - 1)
        available = self.available[index]
        self.index += 1
        return ResourceSample(
            elapsed_seconds=monotonic() - self.started,
            worker_pid=int(worker_pid),
            child_pids=(),
            process_tree_rss_bytes=128 * 1024**2,
            system_available_ram_bytes=int(available * GIB),
            process_gpu_bytes=0,
            results_free_disk_bytes=100 * GIB,
            temp_free_disk_bytes=100 * GIB,
            process_tree_cpu_percent=10.0,
            process_tree_cpu_seconds=0.1,
            stage=stage,
            fold_id=fold_id,
        )


def _sampler_factory(values):
    return lambda **_kwargs: _SequenceSampler(values)


def test_wait_state_immediate_periodic_and_exact_five_minute_logging():
    state = RamWaitState(_ram_policy())
    assert state.observe(int(0.5 * GIB), now=10).action == "wait_started"
    assert state.observe(int(0.6 * GIB), now=309.999) is None
    assert state.observe(int(0.6 * GIB), now=310).action == "wait_periodic"
    assert state.observe(int(0.6 * GIB), now=609.999) is None
    periodic = state.observe(int(0.6 * GIB), now=610)
    assert periodic.action == "wait_periodic"
    assert periodic.waiting_seconds == 600


def test_three_check_recovery_resets_after_dip_and_prevents_thrashing():
    state = RamWaitState(_ram_policy())
    state.observe(int(0.5 * GIB), now=0)
    assert state.observe(int(4.1 * GIB), now=5) is None
    assert state.consecutive_recovery_checks == 1
    assert state.observe(int(3.9 * GIB), now=10) is None
    assert state.consecutive_recovery_checks == 0
    assert state.observe(int(4.1 * GIB), now=15) is None
    assert state.observe(int(4.2 * GIB), now=20) is None
    resumed = state.observe(int(4.3 * GIB), now=25)
    assert resumed.action == "resumed"
    assert resumed.consecutive_recovery_checks == 3
    assert state.waiting is False


def test_indefinite_wait_has_no_terminal_transition_and_excludes_waiting_time():
    state = RamWaitState(_ram_policy())
    state.observe(int(0.5 * GIB), now=10)
    for now in range(15, 3600, 5):
        transition = state.observe(int(0.5 * GIB), now=now)
        assert transition is None or transition.action == "wait_periodic"
    assert state.waiting
    assert state.active_seconds(started=0, now=3600) == 10


def test_supervisor_waits_without_failing_parent_and_recovers_automatically(tmp_path):
    result = supervise_worker(
        worker_target=(
            "credit_risk_fs.experiments.synthetic_execution:"
            "cooperative_ram_gate_worker"
        ),
        worker_kwargs={"active_duration_seconds": 0.05},
        policy=_legacy_policy(),
        ram_control_policy=_ram_policy(),
        results_root=tmp_path,
        temp_root=tmp_path,
        sampler_factory=_sampler_factory([0.5, 4.5, 4.5, 4.5, 8]),
    )
    assert result.status == "completed"
    assert result.stop_code is None
    assert result.return_value == {"completed": True}
    assert result.ram_wait_count == 1
    assert result.total_ram_wait_seconds > 0
    assert [item["action"] for item in result.ram_wait_events] == [
        "wait_started",
        "resumed",
    ]


def test_opaque_stage_uses_safe_process_tree_boundary_suspend_and_resume(tmp_path):
    result = supervise_worker(
        worker_target=(
            "credit_risk_fs.experiments.synthetic_execution:"
            "opaque_ram_stage_worker"
        ),
        worker_kwargs={"duration_seconds": 0.08},
        policy=_legacy_policy(),
        ram_control_policy=_ram_policy(),
        results_root=tmp_path,
        temp_root=tmp_path,
        sampler_factory=_sampler_factory([8, 0.5, 4.5, 4.5, 4.5, 8]),
    )
    assert result.status == "completed"
    assert any(
        item["state"] == "OPAQUE_STAGE_PROCESS_TREE_SUSPENDED"
        for item in result.stop_lifecycle
    )
    assert any(
        item["state"] == "OPAQUE_STAGE_PROCESS_TREE_RESUMED"
        for item in result.stop_lifecycle
    )


def test_new_opaque_stage_waits_at_worker_boundary_until_stable_recovery(tmp_path):
    result = supervise_worker(
        worker_target=(
            "credit_risk_fs.experiments.synthetic_execution:"
            "cooperative_recovery_boundary_worker"
        ),
        worker_kwargs={},
        policy=_legacy_policy(),
        ram_control_policy=_ram_policy(),
        results_root=tmp_path,
        temp_root=tmp_path,
        sampler_factory=_sampler_factory([3.0] * 100 + [4.5, 4.5, 4.5, 8]),
    )
    assert result.status == "completed"
    assert result.ram_wait_count == 1
    assert [item["action"] for item in result.ram_wait_events] == [
        "wait_started",
        "resumed",
    ]
    assert result.ram_wait_events[0]["pause_mode"] == "worker_stage_boundary"
    assert not any(
        item["state"] == "OPAQUE_STAGE_PROCESS_TREE_SUSPENDED"
        for item in result.stop_lifecycle
    )


def test_waiting_time_does_not_consume_worker_wall_limit(tmp_path):
    result = supervise_worker(
        worker_target=(
            "credit_risk_fs.experiments.synthetic_execution:"
            "cooperative_ram_gate_worker"
        ),
        worker_kwargs={"active_duration_seconds": 0.5},
        policy=_legacy_policy(),
        ram_control_policy=_ram_policy(),
        results_root=tmp_path,
        temp_root=tmp_path,
        sampler_factory=_sampler_factory([8, 0.5, 4.5, 4.5, 4.5, 8]),
        max_wall_clock_seconds=0.2,
    )
    assert result.status == "timed_out"
    assert result.total_ram_wait_seconds > 0
    assert result.active_computation_seconds >= 0.2
    assert (
        result.supervisor_awake_elapsed_seconds
        > result.active_computation_seconds
    )


class _FakeEvent:
    def __init__(self):
        self.ready = True
        self.wait_calls = 0

    def is_set(self):
        return self.ready

    def clear(self):
        self.ready = False

    def wait(self, _timeout):
        self.wait_calls += 1
        self.ready = True


def test_chunked_loader_pauses_between_chunks_and_continues_in_source_order(tmp_path):
    source = pd.DataFrame({"a": range(7), "b": [f"v{i}" for i in range(7)]})
    source.to_csv(tmp_path / "application_train.csv", index=False)
    event = _FakeEvent()
    boundaries = []

    def gate(boundary):
        boundaries.append(boundary)
        if boundary == "application_train:before_csv_chunk:1":
            event.clear()
        wait_for_ram_ready(event, None, boundary=boundary, poll_seconds=0.001)

    loaded = DataLoader(tmp_path, memory_gate=gate).load_table(
        "application_train",
        columns=["a", "b"],
        csv_chunk_rows=2,
    )
    pd.testing.assert_frame_equal(loaded, source)
    assert event.wait_calls == 1
    assert "application_train:before_chunk_concat" in boundaries


class _MemorySequence:
    def __init__(self, values):
        self.values = list(values)
        self.index = 0

    def virtual_memory(self):
        index = min(self.index, len(self.values) - 1)
        self.index += 1
        return SimpleNamespace(total=32 * GIB, available=int(self.values[index] * GIB))

    @staticmethod
    def Process(_pid):
        return SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=64 * 1024**2))


def test_inter_run_parent_waits_indefinitely_then_recovers_without_block_status(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    clock = [0.0]
    memory = _MemorySequence([0.5, 0.5, 4.5, 4.5, 4.5])
    result = wait_for_inter_run_readiness(
        policy=_legacy_policy(),
        ram_control_policy=_ram_policy(check=5),
        results_root=tmp_path,
        temp_root=tmp_path,
        psutil_module=memory,
        monotonic_fn=lambda: clock[0],
        sleep_fn=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    assert result.ready
    assert result.stop_code is None
    assert result.ram_wait_count == 1
    assert result.sample_count == 4


def test_memory_error_is_failed_honestly_and_traceback_is_debug_only(tmp_path):
    logs = tmp_path / "logs/runs.log"
    logs.parent.mkdir(parents=True)
    logs.write_text("existing log line\n", encoding="utf-8")
    terminal = io.StringIO()
    with ResearchLogSession(
        logs,
        repository_root=tmp_path,
        command_arguments=["synthetic"],
        terminal_stream=terminal,
    ) as session:
        result = supervise_worker(
            worker_target=(
                "credit_risk_fs.experiments.synthetic_execution:memory_error_worker"
            ),
            worker_kwargs={},
            policy=_legacy_policy(),
            ram_control_policy=_ram_policy(),
            results_root=tmp_path,
            temp_root=tmp_path,
        )
        session.finish("session_failed", level="ERROR", message="Synthetic failure")
    assert result.status == "failed"
    assert result.stop_code == MEMORY_ERROR
    human = logs.read_text(encoding="utf-8")
    debug = (logs.parent / "debug.log").read_text(encoding="utf-8")
    assert human.startswith("existing log line\n")
    assert "Traceback (most recent call last)" not in human
    assert "MemoryError: synthetic allocation failure" in debug


def test_wait_and_resume_messages_are_immediate_human_append_only_records(tmp_path):
    log_path = tmp_path / "logs/runs.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("prior durable line\n", encoding="utf-8")
    terminal = io.StringIO()
    with ResearchLogSession(
        log_path,
        repository_root=tmp_path,
        command_arguments=["synthetic"],
        terminal_stream=terminal,
    ) as session:
        common = {
            "dataset": "lendingclub_v2",
            "stage": "data_loading",
            "process_tree_rss_bytes": int(7.7 * GIB),
            "system_available_ram_bytes": int(1.2 * GIB),
        }
        emit_research_event(
            "ram_wait_started", message="wait", priority=True, **common
        )
        emit_research_event(
            "ram_wait_periodic",
            message="wait",
            priority=True,
            waiting_seconds=300,
            **common,
        )
        emit_research_event(
            "ram_resumed",
            message="resume",
            priority=True,
            **{**common, "system_available_ram_bytes": int(5.1 * GIB)},
        )
        session.finish("session_completed", message="done")
    human = log_path.read_text(encoding="utf-8")
    assert human.startswith("prior durable line\n")
    assert "WAIT   | LendingClub loading paused | Process RAM 7.7 GiB | Available 1.2 GiB" in human
    assert "WAIT   | LendingClub loading paused | Waiting 5m | Process RAM 7.7 GiB" in human
    assert "RESUME | Available RAM stable at 5.1 GiB | Continuing LendingClub loading" in human
    assert terminal.getvalue().count("LendingClub loading paused") == 2


def test_historical_ram_resume_bridge_is_exact_scope_and_fails_on_tamper(tmp_path):
    frozen = tmp_path / "frozen.yaml"
    runtime = tmp_path / "runtime.py"
    frozen.write_text("science: unchanged\n", encoding="utf-8")
    runtime.write_text("mechanics = 'wait'\n", encoding="utf-8")
    from credit_risk_fs.experiments.atomic_io import sha256_file

    identity = {
        "git_commit": "predecessor",
        "resolved_config_hash": "scientific-identity",
    }
    bridge = {
        "schema_version": "full_baseline_ram_wait_compatibility_v1",
        "eligible_run_ids": ["cell-003"],
        "eligible_stop_codes": ["ram_system_headroom"],
        "predecessor_commit": "predecessor",
        "full_baseline_configuration_sha256": "frozen-config-hash",
        "interrupted_identity": identity,
        "frozen_scientific_files": {"frozen.yaml": sha256_file(frozen)},
        "runtime_mechanics_files": {"runtime.py": sha256_file(runtime)},
    }
    bridge_path = tmp_path / "bridge.json"
    bridge_path.write_text(json.dumps(bridge), encoding="utf-8")
    checkpoint = {
        "status": "aborted_resource_limit",
        "stop_code": "ram_system_headroom",
        "identity": identity,
    }
    result = authenticate_full_baseline_ram_bridge(
        tmp_path,
        run_id="cell-003",
        checkpoint=checkpoint,
        full_baseline_configuration_sha256="frozen-config-hash",
        bridge_path=bridge_path,
    )
    assert result["scope"] == "RAM supervision and cooperative loading mechanics only"

    resumed_then_interrupted = {
        **checkpoint,
        "status": "interrupted",
        "stop_code": "manual_interrupt",
        "attempt_history": [
            {
                "status": "aborted_resource_limit",
                "stop_code": "ram_system_headroom",
            }
        ],
    }
    authenticate_full_baseline_ram_bridge(
        tmp_path,
        run_id="cell-003",
        checkpoint=resumed_then_interrupted,
        full_baseline_configuration_sha256="frozen-config-hash",
        bridge_path=bridge_path,
    )

    runtime.write_text("mechanics = 'tampered'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime.py"):
        authenticate_full_baseline_ram_bridge(
            tmp_path,
            run_id="cell-003",
            checkpoint=checkpoint,
            full_baseline_configuration_sha256="frozen-config-hash",
            bridge_path=bridge_path,
        )
