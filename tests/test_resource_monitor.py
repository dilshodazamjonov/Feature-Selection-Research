from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from credit_risk_fs.experiments.resource_monitor import (
    DISK_RESULTS_LIMIT,
    GPU_PROCESS_LIMIT,
    MANUAL_INTERRUPT,
    RAM_PROCESS_LIMIT,
    WORKER_CRASH,
    _StopCauseRecorder,
    _OwnedProcessRegistry,
    _shutdown_owned_worker,
    NvmlProcessTelemetry,
    ProcessTreeSampler,
    ResourceSample,
    _classify_threshold,
    _new_warnings,
    supervise_worker,
    wait_for_inter_run_readiness,
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


def test_process_rss_alone_emits_no_warning_or_stop():
    policy = _policy(warn_ram=0.05, abort_ram=1.0)
    emitted: set[str] = set()
    first = _new_warnings(_sample(rss_gb=0.1), policy, emitted)
    second = _new_warnings(_sample(rss_gb=0.2), policy, emitted)
    assert first == []
    assert second == []


def test_abort_reason_classification_ignores_ram_rss_but_keeps_gpu():
    policy = _policy(warn_ram=0.05, abort_ram=0.1)
    assert _classify_threshold(_sample(rss_gb=0.2), policy) is None
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


def test_duplicate_stage_and_supervisor_logging_context_is_safe(tmp_path):
    spec = {
        "cell_id": "01-homecredit-fold-1-catboost-shap",
        "cell_index": 1,
        "dataset": "homecredit",
        "method_id": "catboost_shap",
    }
    result = supervise_worker(
        worker_target=(
            "credit_risk_fs.experiments.synthetic_execution:"
            "duplicate_stage_context_worker"
        ),
        worker_kwargs={"spec": spec},
        policy=_policy(),
        results_root=tmp_path,
        temp_root=tmp_path,
        run_association=spec["cell_id"],
    )
    assert result.status == "completed"
    assert result.return_value == {"ok": True}
    assert result.final_stage == "pilot_dev_data_loading"
    assert result.final_fold_id == 1


def test_high_process_rss_alone_does_not_abort_worker(tmp_path):
    result = supervise_worker(
        worker_target="credit_risk_fs.experiments.synthetic_execution:immediate_success_worker",
        worker_kwargs={},
        policy=_policy(warn_ram=0.07, abort_ram=0.11),
        results_root=tmp_path,
        temp_root=tmp_path,
        sampler_factory=_HighRssSampler,
    )
    assert result.status == "completed"
    assert result.stop_code is None
    assert not any(message.startswith(RAM_PROCESS_LIMIT) for message in result.warnings)
    assert result.child_cleanup_confirmed


def test_opt_in_process_rss_ceiling_aborts_owned_worker(tmp_path):
    result = supervise_worker(
        worker_target="credit_risk_fs.experiments.synthetic_execution:uncooperative_wait_worker",
        worker_kwargs={},
        policy=_policy(warn_ram=0.07, abort_ram=0.11),
        results_root=tmp_path,
        temp_root=tmp_path,
        sampler_factory=_HighRssSampler,
        enforce_memory_limits=True,
    )
    assert result.status == "aborted_resource_limit"
    assert result.stop_code == RAM_PROCESS_LIMIT
    assert any(message.startswith(RAM_PROCESS_LIMIT) for message in result.warnings)
    assert result.child_cleanup_confirmed


class _HighRssSampler(ProcessTreeSampler):
    def sample(self, *args, **kwargs):
        return replace(
            super().sample(*args, **kwargs),
            process_tree_rss_bytes=100 * 1024**3,
        )


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


def test_stage_specific_wall_clock_limit_stops_owned_worker(tmp_path):
    result = supervise_worker(
        worker_target=(
            "credit_risk_fs.experiments.synthetic_execution:"
            "uncooperative_wait_worker"
        ),
        worker_kwargs={},
        policy=_policy(),
        results_root=tmp_path,
        temp_root=tmp_path,
        stage_wall_clock_limits_seconds={"uncooperative_wait": 0.06},
    )
    assert result.status == "timed_out"
    assert result.stop_code == "wall_clock_limit"
    assert result.final_stage == "uncooperative_wait"
    assert any(
        event["state"] == "STAGE_WALL_CLOCK_STOP_LATCHED"
        for event in result.stop_lifecycle
    )


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


def test_first_stop_cause_is_immutable_and_secondary_events_are_retained():
    recorder = _StopCauseRecorder()
    recorder.observe(DISK_RESULTS_LIMIT, elapsed_seconds=1.0, detail="threshold")
    recorder.observe(MANUAL_INTERRUPT, elapsed_seconds=2.0, detail="later interrupt")
    recorder.observe(WORKER_CRASH, elapsed_seconds=3.0, detail="later worker error")
    assert recorder.primary == DISK_RESULTS_LIMIT
    assert [item["code"] for item in recorder.secondary] == [
        MANUAL_INTERRUPT,
        WORKER_CRASH,
    ]
    manual_first = _StopCauseRecorder()
    manual_first.observe(MANUAL_INTERRUPT, elapsed_seconds=1.0, detail="interrupt")
    manual_first.observe(DISK_RESULTS_LIMIT, elapsed_seconds=2.0, detail="late sample")
    assert manual_first.primary == MANUAL_INTERRUPT


def test_uncooperative_worker_is_stopped_within_finite_bound(tmp_path):
    policy = replace(
        _policy(warn_ram=0.000001, abort_ram=0.000002),
        monitoring=MonitoringPolicy(0.01, 0.15, 0.2),
        disk=DiskPolicy(1_000_000_000, 0.001, 2.5),
    )
    result = supervise_worker(
        worker_target=(
            "credit_risk_fs.experiments.synthetic_execution:"
            "uncooperative_wait_worker"
        ),
        worker_kwargs={},
        policy=policy,
        results_root=tmp_path,
        temp_root=tmp_path,
        run_association="test:uncooperative",
    )
    assert result.status == "aborted_resource_limit"
    assert result.primary_stop_code == DISK_RESULTS_LIMIT
    assert result.graceful_stop_completed is False
    assert result.shutdown_elapsed_seconds is not None
    assert result.shutdown_elapsed_seconds < 1.5
    assert result.child_cleanup_confirmed
    assert result.queue_cleanup_confirmed
    assert not result.survivor_processes
    assert any(
        event["state"] == "TERMINATE_PROCESS_TREE"
        for event in result.stop_lifecycle
    )


def test_live_nested_owned_process_tree_is_fully_stopped(tmp_path):
    policy = replace(
        _policy(warn_ram=0.000001, abort_ram=0.000002),
        monitoring=MonitoringPolicy(0.01, 0.3, 0.3),
    )
    result = supervise_worker(
        worker_target=(
            "credit_risk_fs.experiments.synthetic_execution:"
            "uncooperative_wait_worker"
        ),
        worker_kwargs={"spawn_stubborn_child": True},
        policy=policy,
        results_root=tmp_path,
        temp_root=tmp_path,
        run_association="test:nested-tree",
        sampler_factory=_TriggerAfterChildSampler,
    )
    assert result.child_cleanup_confirmed
    assert not result.survivor_processes
    assert any(
        item["relationship"] == "observed_descendant"
        for item in result.owned_processes
    )


class _TriggerAfterChildSampler(ProcessTreeSampler):
    def sample(self, *args, **kwargs):
        sample = super().sample(*args, **kwargs)
        return replace(
            sample,
            results_free_disk_bytes=(0 if sample.child_pids else sample.results_free_disk_bytes),
        )


class _FakeStopEvent:
    def __init__(self):
        self.is_set = False

    def set(self):
        self.is_set = True


class _FakeProcess:
    pid = 991

    def __init__(self, *, interrupt_join=False):
        self.alive = True
        self.interrupt_join = interrupt_join

    def join(self, timeout=None):
        del timeout
        if self.interrupt_join:
            self.interrupt_join = False
            raise KeyboardInterrupt

    def is_alive(self):
        return self.alive


class _FakeOwnedRegistry:
    def __init__(self, process):
        self.process = process
        self.force_kill_called = False

    def alive(self):
        return [self.process] if self.process.alive else []

    def terminate_phase(self, *, timeout_seconds):
        assert timeout_seconds > 0
        return self.alive()

    def kill_phase(self, *, timeout_seconds):
        assert timeout_seconds > 0
        self.force_kill_called = True
        self.process.alive = False
        return []

    def survivor_records(self):
        return ()


def test_stubborn_process_reaches_force_kill_and_later_interrupt_is_secondary():
    process = _FakeProcess(interrupt_join=True)
    registry = _FakeOwnedRegistry(process)
    recorder = _StopCauseRecorder()
    recorder.observe(DISK_RESULTS_LIMIT, elapsed_seconds=0.0, detail="threshold")
    lifecycle = []
    ok, survivors, condition, elapsed, graceful = _shutdown_owned_worker(
        process=process,
        stop_event=_FakeStopEvent(),
        ownership=registry,
        policy=replace(_policy(), monitoring=MonitoringPolicy(0.01, 0.01, 0.01)),
        recorder=recorder,
        lifecycle=lifecycle,
        supervisor_started=0.0,
    )
    assert ok and not survivors and condition is None
    assert elapsed < 0.5
    assert graceful is False
    assert registry.force_kill_called
    assert recorder.primary == DISK_RESULTS_LIMIT
    assert recorder.secondary[0]["code"] == MANUAL_INTERRUPT
    assert any(item["state"] == "FORCE_KILL_REMAINDERS" for item in lifecycle)


def test_saturated_stage_queue_and_oversized_result_cannot_hang(tmp_path):
    saturated = supervise_worker(
        worker_target=(
            "credit_risk_fs.experiments.synthetic_execution:"
            "saturated_stage_queue_worker"
        ),
        worker_kwargs={},
        policy=_policy(),
        results_root=tmp_path,
        temp_root=tmp_path,
    )
    assert saturated.status == "completed"
    assert saturated.queue_cleanup_confirmed
    oversized = supervise_worker(
        worker_target=(
            "credit_risk_fs.experiments.synthetic_execution:oversized_result_worker"
        ),
        worker_kwargs={},
        policy=_policy(),
        results_root=tmp_path,
        temp_root=tmp_path,
    )
    assert oversized.status == "failed"
    assert oversized.stop_code == WORKER_CRASH
    assert "compact-metadata limit" in (oversized.worker_error or "")
    assert oversized.return_value is None

    near_limit = supervise_worker(
        worker_target=(
            "credit_risk_fs.experiments.synthetic_execution:near_limit_result_worker"
        ),
        worker_kwargs={},
        policy=_policy(),
        results_root=tmp_path,
        temp_root=tmp_path,
    )
    assert near_limit.status == "completed"
    assert near_limit.child_cleanup_confirmed
    assert near_limit.queue_cleanup_confirmed
    assert len(near_limit.return_value["bounded_metadata"]) == 512 * 1024


def test_sequential_synthetic_runs_leave_no_owned_worker_or_large_payload(tmp_path):
    results = [
        supervise_worker(
            worker_target=(
                "credit_risk_fs.experiments.synthetic_execution:"
                "immediate_success_worker"
            ),
            worker_kwargs={},
            policy=_policy(),
            results_root=tmp_path,
            temp_root=tmp_path,
            run_association=f"test:sequential:{index}",
        )
        for index in range(3)
    ]
    assert all(item.status == "completed" for item in results)
    assert all(item.child_cleanup_confirmed for item in results)
    assert all(not item.survivor_processes for item in results)
    assert all(item.return_value == {"ok": True} for item in results)


class _LowMemoryPsutil:
    class _Memory:
        available = 10 * 1024**3
        total = 32 * 1024**3

    class _Info:
        rss = 1234

    class _Process:
        def memory_info(self):
            return _LowMemoryPsutil._Info()

    @staticmethod
    def Process(_pid):
        return _LowMemoryPsutil._Process()

    @staticmethod
    def virtual_memory():
        return _LowMemoryPsutil._Memory()


def test_inter_run_readiness_keeps_disk_failure_terminal(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    result = wait_for_inter_run_readiness(
        policy=replace(
            _policy(), disk=DiskPolicy(1_000_000_000, 0.001, 2.5)
        ),
        results_root=tmp_path,
        temp_root=tmp_path,
        timeout_seconds=0.0,
        psutil_module=_LowMemoryPsutil,
    )
    assert not result.ready
    assert result.stop_code == DISK_RESULTS_LIMIT
    assert result.elapsed_seconds < 0.5


class _ReusedPidPsutil:
    class NoSuchProcess(Exception):
        pass

    class AccessDenied(Exception):
        pass

    terminate_called = False

    class _Process:
        pid = 77

        def create_time(self):
            return 200.0

        def is_running(self):
            return True

        def children(self, recursive=True):
            del recursive
            return []

        def terminate(self):
            _ReusedPidPsutil.terminate_called = True

    @staticmethod
    def Process(_pid):
        return _ReusedPidPsutil._Process()

    @staticmethod
    def wait_procs(processes, timeout):
        del timeout
        return [], processes


def test_process_ownership_never_targets_a_reused_unrelated_pid():
    registry = _OwnedProcessRegistry.__new__(_OwnedProcessRegistry)
    registry.psutil = _ReusedPidPsutil
    registry.root_pid = 77
    registry.association = "expected-run:attempt"
    registry._unverified_pids = set()
    registry._records = {
        77: {
            "pid": 77,
            "parent_pid_at_discovery": 10,
            "create_time": 100.0,
            "relationship": "spawned_worker",
            "run_association": "expected-run:attempt",
        }
    }
    _ReusedPidPsutil.terminate_called = False
    assert registry.terminate_phase(timeout_seconds=0.01) == []
    assert not _ReusedPidPsutil.terminate_called
