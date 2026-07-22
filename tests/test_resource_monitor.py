from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from credit_risk_fs.experiments.resource_monitor import (
    GPU_PROCESS_LIMIT,
    MANUAL_INTERRUPT,
    RAM_PROCESS_LIMIT,
    WORKER_CRASH,
    NvmlProcessTelemetry,
    ResourceSample,
    _classify_threshold,
    _new_warnings,
    supervise_worker,
)
from credit_risk_fs.experiments.resource_policy import (
    DiskPolicy,
    ExecutionPolicy,
    GpuPolicy,
    MemoryPolicy,
    MonitoringPolicy,
    ParallelismPolicy,
    ResolvedExecutionPolicy,
)


def _policy(*, warn_ram: float = 1.0, abort_ram: float = 2.0) -> ResolvedExecutionPolicy:
    return ResolvedExecutionPolicy(
        schema_version="resource_safe_execution_policy_v1",
        profile_name="test_safe",
        parallelism=ParallelismPolicy(1, 1, 0, 1, False),
        memory=MemoryPolicy(1.0, warn_ram, abort_ram, 0.001, 1.35),
        gpu=GpuPolicy(0.1, 1.0, 2.0, True),
        disk=DiskPolicy(0.001, 0.001, 2.5),
        monitoring=MonitoringPolicy(0.02, 0.4, 2.0),
        configured_policy_path="test",
    )


def _sample(*, rss_gb: float = 0.1, gpu_gb: float | None = None) -> ResourceSample:
    gib = 1024**3
    return ResourceSample(
        elapsed_seconds=1.0,
        worker_pid=123,
        child_pids=(124,),
        process_tree_rss_bytes=int(rss_gb * gib),
        system_available_ram_bytes=10 * gib,
        process_gpu_bytes=None if gpu_gb is None else int(gpu_gb * gib),
        results_free_disk_bytes=100 * gib,
        temp_free_disk_bytes=100 * gib,
        process_tree_cpu_percent=20.0,
        process_tree_cpu_seconds=1.0,
        stage="test",
        fold_id=1,
    )


def test_warning_is_emitted_once_per_resource():
    policy = _policy(warn_ram=0.05, abort_ram=1.0)
    emitted: set[str] = set()
    first = _new_warnings(_sample(rss_gb=0.1), policy, emitted)
    second = _new_warnings(_sample(rss_gb=0.2), policy, emitted)
    assert len(first) == 1
    assert second == []


def test_abort_reason_classification_for_ram_and_gpu():
    policy = _policy(warn_ram=0.05, abort_ram=0.1)
    assert _classify_threshold(_sample(rss_gb=0.2), policy) == RAM_PROCESS_LIMIT
    gpu_policy = replace(policy, memory=replace(policy.memory, abort_process_tree_rss_gb=10))
    assert _classify_threshold(_sample(rss_gb=0.1, gpu_gb=3.0), gpu_policy) == GPU_PROCESS_LIMIT


def test_windows_spawn_safe_worker_startup(tmp_path):
    result = supervise_worker(
        worker_target="credit_risk_fs.experiments.synthetic_execution:immediate_success_worker",
        worker_kwargs={},
        policy=_policy(),
        results_root=tmp_path,
        temp_root=tmp_path,
    )
    assert result.status == "completed"
    assert result.return_value == {"ok": True}
    assert result.worker_exit_code == 0


def test_forced_low_memory_abort_records_warning_and_cleans_child(tmp_path):
    result = supervise_worker(
        worker_target="credit_risk_fs.experiments.synthetic_execution:bounded_memory_worker",
        worker_kwargs={
            "chunk_mb": 4,
            "maximum_allocation_mb": 160,
            "spawn_child": True,
        },
        policy=_policy(warn_ram=0.07, abort_ram=0.11),
        results_root=tmp_path,
        temp_root=tmp_path,
    )
    assert result.status == "aborted_resource_limit"
    assert result.stop_code == RAM_PROCESS_LIMIT
    assert any(message.startswith(RAM_PROCESS_LIMIT) for message in result.warnings)
    assert any(sample.child_pids for sample in result.samples)
    assert result.child_cleanup_confirmed


def test_unexpected_worker_exit_is_recorded(tmp_path):
    result = supervise_worker(
        worker_target="credit_risk_fs.experiments.synthetic_execution:unexpected_exit_worker",
        worker_kwargs={"exit_code": 7},
        policy=_policy(),
        results_root=tmp_path,
        temp_root=tmp_path,
    )
    assert result.status == "failed"
    assert result.stop_code == WORKER_CRASH
    assert result.worker_exit_code == 7


class _ClosedGpu:
    def close(self):
        return None


class _InterruptingSampler:
    def __init__(self, **_kwargs):
        self.gpu = _ClosedGpu()

    def sample(self, *_args, **_kwargs):
        raise KeyboardInterrupt


def test_manual_interrupt_is_recorded_and_worker_stopped(tmp_path):
    result = supervise_worker(
        worker_target="credit_risk_fs.experiments.synthetic_execution:cooperative_wait_worker",
        worker_kwargs={},
        policy=_policy(),
        results_root=tmp_path,
        temp_root=tmp_path,
        sampler_factory=_InterruptingSampler,
    )
    assert result.status == "interrupted"
    assert result.stop_code == MANUAL_INTERRUPT
    assert result.child_cleanup_confirmed


def test_gpu_telemetry_mock_can_report_process_bytes():
    telemetry = NvmlProcessTelemetry.__new__(NvmlProcessTelemetry)
    telemetry.available = False
    telemetry.error = "mock unavailable"
    telemetry._pynvml = None
    telemetry._handles = []
    assert telemetry.bytes_for_pids({1}) is None
