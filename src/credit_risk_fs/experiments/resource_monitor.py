"""Process-tree resource sampling and Windows-safe supervised execution."""

from __future__ import annotations

import importlib
import inspect
import gc
import logging
import multiprocessing
import os
import pickle
import queue
import shutil
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic as suspension_inclusive_monotonic
from time import sleep
from typing import Any, Mapping

from credit_risk_fs.experiments.active_clock import (
    ACTIVE_CLOCK_SOURCE,
    awake_monotonic as monotonic,
)

from credit_risk_fs.experiments.resource_policy import (
    GIB,
    ResolvedExecutionPolicy,
    apply_thread_environment,
)
from credit_risk_fs.experiments.ram_control import (
    RamWaitState,
    ResolvedRamControlPolicy,
    default_ram_control_policy,
)
from credit_risk_fs.experiments.research_logging import (
    STAGE_HEARTBEAT_INTERVAL_SECONDS,
    active_worker_transport,
    configure_worker_logging,
    emit_contextual_research_event,
    emit_research_event,
)


logger = logging.getLogger(__name__)

RAM_PROCESS_LIMIT = "ram_process_limit"
RAM_SYSTEM_HEADROOM = "ram_system_headroom"
GPU_PROCESS_LIMIT = "gpu_process_limit"
DISK_RESULTS_LIMIT = "disk_results_limit"
DISK_TEMP_LIMIT = "disk_temp_limit"
MANUAL_INTERRUPT = "manual_interrupt"
PREFLIGHT_REJECTED = "preflight_rejected"
WORKER_CRASH = "worker_crash"
MEMORY_ERROR = "memory_error"
RAM_PAUSE_UNAVAILABLE = "ram_pause_unavailable"
WORKER_TREE_TERMINATION_FAILED = "worker_tree_termination_failed"
WALL_CLOCK_LIMIT = "wall_clock_limit"
RESULT_PAYLOAD_LIMIT_BYTES = 1024 * 1024
MAX_QUEUE_DRAIN_MESSAGES = 1024

_STAGE_COMPONENTS = {
    "data_loading": "data_loading",
    "data_loading_allocation": "data_loading",
    "pilot_dev_data_loading": "prepare_voting_pilot_dev_data",
    "pilot_fold_projection": "canonical_fold_projection",
    "pilot_selection_encoding": "original_feature_numeric_encoder",
    "pilot_catboost_shap": "catboost_shap",
    "pilot_boruta_random_forest": "boruta_random_forest",
    "pilot_rfe_catboost": "rfe_catboost",
    "dev_data_loading": "data_loading",
    "target_extraction": "validated_target_projection",
    "feature_filtering_sanitization": "candidate_feature_contract",
    "row_boundary_selection": "fold_boundary",
    "selection_encoding": "preprocessing",
    "voter_rf_corr_mrmr": "rf_corr_mrmr",
    "reference_rf_corr_mrmr": "rf_corr_mrmr",
    "voter_boruta": "boruta",
    "rank_aggregation": "rank_voting",
    "rfe_encoding": "preprocessing",
    "rfe": "rfe",
    "selected_projection_reload": "data_loading",
    "final_preprocessing": "preprocessing",
    "final_model_fit": "model_fit",
    "final_prediction": "prediction",
    "fold_artifact_writing": "artifact_writer",
    "fold_checkpoint_finalization": "checkpoint",
    "dev_oof_aggregation": "prediction",
    "dev_evaluation": "evaluation",
    "dev_artifact_writing": "artifact_writer",
    "dev_checkpoint_finalization": "checkpoint",
    "full_dev_data_loading": "data_loading",
    "full_dev_target_extraction": "validated_target_projection",
    "full_dev_feature_filtering_sanitization": "candidate_feature_contract",
    "full_dev_selected_projection_reload": "data_loading",
    "full_dev_selected_feature_validation": "candidate_feature_contract",
    "locked_oot_data_loading": "data_loading",
    "oot_target_extraction": "validated_target_projection",
    "oot_feature_filtering_sanitization": "candidate_feature_contract",
    "full_dev_preprocessing": "preprocessing",
    "full_dev_model_fit": "model_fit",
    "full_dev_prediction": "prediction",
    "oot_prediction": "prediction",
    "oot_artifact_writing": "artifact_writer",
    "oot_evaluation": "evaluation",
    "oot_checkpoint_finalization": "checkpoint",
}

_COOPERATIVE_RAM_STAGES = {
    "data_loading",
    "pilot_dev_data_loading",
    "dev_data_loading",
    "selected_projection_reload",
    "full_dev_data_loading",
    "full_dev_selected_projection_reload",
    "locked_oot_data_loading",
}


@dataclass(frozen=True, slots=True)
class ResourceSample:
    elapsed_seconds: float
    worker_pid: int
    child_pids: tuple[int, ...]
    process_tree_rss_bytes: int
    system_available_ram_bytes: int
    process_gpu_bytes: int | None
    results_free_disk_bytes: int
    temp_free_disk_bytes: int
    process_tree_cpu_percent: float
    process_tree_cpu_seconds: float
    stage: str | None
    fold_id: str | int | None


@dataclass(frozen=True, slots=True)
class SupervisorResult:
    status: str
    stop_code: str | None
    worker_exit_code: int | None
    return_value: Any
    worker_error: str | None
    samples: tuple[ResourceSample, ...]
    warnings: tuple[str, ...]
    peak_process_tree_rss_bytes: int
    peak_process_gpu_bytes: int | None
    minimum_system_available_ram_bytes: int | None
    minimum_results_free_disk_bytes: int | None
    minimum_temp_free_disk_bytes: int | None
    child_cleanup_confirmed: bool
    final_stage: str | None
    final_fold_id: str | int | None
    primary_stop_code: str | None = None
    secondary_events: tuple[dict[str, Any], ...] = ()
    stop_lifecycle: tuple[dict[str, Any], ...] = ()
    owned_processes: tuple[dict[str, Any], ...] = ()
    survivor_processes: tuple[dict[str, Any], ...] = ()
    termination_condition: str | None = None
    graceful_stop_completed: bool | None = None
    shutdown_elapsed_seconds: float | None = None
    parent_rss_before_bytes: int | None = None
    parent_rss_after_bytes: int | None = None
    system_available_ram_after_bytes: int | None = None
    queue_cleanup_confirmed: bool = True
    run_association: str | None = None
    emergency_ram_margin_bytes: int | None = None
    ram_recovery_threshold_bytes: int | None = None
    ram_check_interval_seconds: float | None = None
    ram_log_interval_seconds: float | None = None
    ram_recovery_consecutive_checks: int | None = None
    total_ram_wait_seconds: float = 0.0
    active_computation_seconds: float = 0.0
    ram_wait_count: int = 0
    ram_wait_events: tuple[dict[str, Any], ...] = ()
    active_clock_source: str = ACTIVE_CLOCK_SOURCE
    system_suspend_seconds: float = 0.0
    system_suspend_excluded_from_active_time: bool = True
    supervisor_awake_elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["samples"] = [asdict(item) for item in self.samples]
        return payload


@dataclass(frozen=True, slots=True)
class InterRunReadiness:
    ready: bool
    stop_code: str | None
    elapsed_seconds: float
    sample_count: int
    parent_pid: int
    parent_rss_bytes: int
    system_available_ram_bytes: int
    results_free_disk_bytes: int
    temp_free_disk_bytes: int
    active_child_pids: tuple[int, ...]
    total_ram_wait_seconds: float = 0.0
    ram_wait_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _StopCauseRecorder:
    """Latch the first stop cause and retain every later termination event."""

    def __init__(self) -> None:
        self.primary: str | None = None
        self.primary_elapsed_seconds: float | None = None
        self.secondary: list[dict[str, Any]] = []

    def observe(self, code: str, *, elapsed_seconds: float, detail: str) -> None:
        event = {
            "code": str(code),
            "elapsed_seconds": float(elapsed_seconds),
            "timestamp_utc": _utc_now(),
            "detail": str(detail),
        }
        if self.primary is None:
            self.primary = str(code)
            self.primary_elapsed_seconds = float(elapsed_seconds)
        else:
            self.secondary.append(event)


class _NonBlockingPublisher:
    """Worker-side stage publisher that cannot block on a full parent queue."""

    def __init__(self, target_queue: Any) -> None:
        self._queue = target_queue

    def put(self, value: Any, *_args: Any, **_kwargs: Any) -> None:
        try:
            self._queue.put_nowait(value)
        except (queue.Full, BrokenPipeError, EOFError, OSError, ValueError):
            return


class NvmlProcessTelemetry:
    """One-time NVML setup used for per-process sampling without shelling out."""

    def __init__(self) -> None:
        self.available = False
        self.error: str | None = None
        self._pynvml: Any = None
        self._handles: list[Any] = []
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handles = [
                pynvml.nvmlDeviceGetHandleByIndex(index)
                for index in range(pynvml.nvmlDeviceGetCount())
            ]
            self.available = bool(self._handles)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def bytes_for_pids(self, pids: set[int]) -> int | None:
        if not self.available or self._pynvml is None:
            return None
        total = 0
        try:
            for handle in self._handles:
                seen: dict[int, int] = {}
                for family in ("Compute", "Graphics"):
                    functions = [
                        getattr(
                            self._pynvml,
                            f"nvmlDeviceGet{family}RunningProcesses_v3",
                            None,
                        ),
                        getattr(
                            self._pynvml,
                            f"nvmlDeviceGet{family}RunningProcesses",
                            None,
                        ),
                    ]
                    function = next((item for item in functions if item is not None), None)
                    if function is None:
                        continue
                    for process in function(handle):
                        pid = int(process.pid)
                        used = int(getattr(process, "usedGpuMemory", 0) or 0)
                        if pid in pids:
                            seen[pid] = max(seen.get(pid, 0), used)
                total += sum(seen.values())
            return total
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.available = False
            return None

    def close(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


class ProcessTreeSampler:
    def __init__(
        self,
        *,
        results_root: str | Path,
        temp_root: str | Path,
        gpu_telemetry: NvmlProcessTelemetry | None = None,
        psutil_module: Any | None = None,
    ) -> None:
        if psutil_module is None:
            import psutil as psutil_module
        self.psutil = psutil_module
        self.results_root = Path(results_root)
        self.temp_root = Path(temp_root)
        self.gpu = gpu_telemetry or NvmlProcessTelemetry()
        self.started = monotonic()

    def sample(
        self,
        worker_pid: int,
        *,
        stage: str | None = None,
        fold_id: str | int | None = None,
    ) -> ResourceSample:
        try:
            root = self.psutil.Process(worker_pid)
            processes = [root, *root.children(recursive=True)]
        except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
            processes = []
        pids: set[int] = set()
        rss = 0
        cpu_percent = 0.0
        cpu_seconds = 0.0
        for process in processes:
            try:
                pids.add(int(process.pid))
                rss += int(process.memory_info().rss)
                cpu_percent += float(process.cpu_percent(interval=None))
                cpu_times = process.cpu_times()
                cpu_seconds += float(cpu_times.user + cpu_times.system)
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                continue
        gpu_bytes = self.gpu.bytes_for_pids(pids) if pids else None
        memory = self.psutil.virtual_memory()
        results_disk = shutil.disk_usage(self.results_root)
        temp_disk = shutil.disk_usage(self.temp_root)
        return ResourceSample(
            elapsed_seconds=monotonic() - self.started,
            worker_pid=int(worker_pid),
            child_pids=tuple(sorted(pid for pid in pids if pid != int(worker_pid))),
            process_tree_rss_bytes=rss,
            system_available_ram_bytes=int(memory.available),
            process_gpu_bytes=gpu_bytes,
            results_free_disk_bytes=int(results_disk.free),
            temp_free_disk_bytes=int(temp_disk.free),
            process_tree_cpu_percent=cpu_percent,
            process_tree_cpu_seconds=cpu_seconds,
            stage=stage,
            fold_id=fold_id,
        )


def _load_worker_target(target: str):
    if ":" not in target:
        raise ValueError("worker target must use 'module:function' form")
    module_name, function_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not callable(function):
        raise TypeError(f"worker target is not callable: {target}")
    return function


def _worker_entry(
    target: str,
    kwargs: dict[str, Any],
    stop_event: Any,
    result_queue: Any,
    stage_queue: Any,
    ram_ready_event: Any,
    ram_recovery_threshold_bytes: int,
    run_association: str,
    logging_transport: tuple[Any, str, Any, Any] | None,
    logging_context: Mapping[str, Any],
) -> None:
    publisher = _NonBlockingPublisher(stage_queue)
    if logging_transport is not None:
        target_queue, session_id, routine_counter, priority_counter = logging_transport
        configure_worker_logging(
            target_queue,
            session_id=session_id,
            context=logging_context,
            routine_drop_counter=routine_counter,
            priority_drop_counter=priority_counter,
        )
    try:
        emit_research_event(
            "worker_started",
            message="Spawned research worker started",
            priority=True,
            worker_pid=os.getpid(),
            worker_target=target,
            run_association=run_association,
        )
        function = _load_worker_target(target)
        controls = {"stop_event": stop_event, "stage_queue": publisher}
        parameters = inspect.signature(function).parameters
        if "ram_ready_event" in parameters or any(
            item.kind is inspect.Parameter.VAR_KEYWORD
            for item in parameters.values()
        ):
            controls["ram_ready_event"] = ram_ready_event
        if "ram_recovery_threshold_bytes" in parameters or any(
            item.kind is inspect.Parameter.VAR_KEYWORD
            for item in parameters.values()
        ):
            controls["ram_recovery_threshold_bytes"] = int(
                ram_recovery_threshold_bytes
            )
        value = function(**controls, **kwargs)
        encoded = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if len(encoded) > RESULT_PAYLOAD_LIMIT_BYTES:
            raise ValueError(
                "worker result payload exceeded the compact-metadata limit "
                f"({len(encoded)} > {RESULT_PAYLOAD_LIMIT_BYTES} bytes)"
            )
        result_queue.put_nowait(
            {
                "kind": "result",
                "value": value,
                "serialized_size_bytes": len(encoded),
                "run_association": run_association,
            }
        )
        emit_research_event(
            "worker_completed",
            message="Spawned research worker completed its target",
            priority=True,
            worker_pid=os.getpid(),
            result_payload_bytes=len(encoded),
        )
    except KeyboardInterrupt:
        emit_research_event(
            "worker_interrupted",
            level="ERROR",
            message="Spawned research worker received KeyboardInterrupt",
            priority=True,
            worker_pid=os.getpid(),
            stop_code=MANUAL_INTERRUPT,
            exception_class="KeyboardInterrupt",
        )
        try:
            result_queue.put_nowait(
                {
                    "kind": "interrupt",
                    "error": MANUAL_INTERRUPT,
                    "run_association": run_association,
                }
            )
        except (queue.Full, BrokenPipeError, EOFError, OSError, ValueError):
            pass
        raise
    except Exception as exc:
        error_traceback = traceback.format_exc()
        emit_research_event(
            "worker_failed",
            level="ERROR",
            message=f"Spawned research worker failed: {type(exc).__name__}: {exc}",
            priority=True,
            worker_pid=os.getpid(),
            exception_class=type(exc).__name__,
            traceback=error_traceback,
        )
        try:
            result_queue.put_nowait(
                {
                "kind": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "exception_class": type(exc).__name__,
                "traceback": error_traceback,
                    "run_association": run_association,
                }
            )
        except (queue.Full, BrokenPipeError, EOFError, OSError, ValueError):
            pass
        raise


def _drain_stage_queue(
    stage_queue: Any,
    stage: str | None,
    fold_id: Any,
    *,
    on_update: Any | None = None,
) -> tuple[str | None, Any]:
    for _ in range(MAX_QUEUE_DRAIN_MESSAGES):
        try:
            update = stage_queue.get_nowait()
        except (queue.Empty, EOFError, OSError, ValueError):
            return stage, fold_id
        if isinstance(update, Mapping):
            next_stage = (
                str(update.get("stage"))
                if update.get("stage") is not None
                else stage
            )
            next_fold_id = update.get("fold_id", fold_id)
            if on_update is not None:
                on_update(stage, fold_id, next_stage, next_fold_id, update)
            stage = next_stage
            fold_id = next_fold_id
    return stage, fold_id


def _classify_threshold(sample: ResourceSample, policy: ResolvedExecutionPolicy) -> str | None:
    if sample.process_gpu_bytes is not None and sample.process_gpu_bytes >= int(
        policy.gpu.abort_process_vram_gb * GIB
    ):
        return GPU_PROCESS_LIMIT
    if sample.results_free_disk_bytes <= int(policy.disk.minimum_free_results_gb * GIB):
        return DISK_RESULTS_LIMIT
    if sample.temp_free_disk_bytes <= int(policy.disk.minimum_free_temp_gb * GIB):
        return DISK_TEMP_LIMIT
    return None


def _new_warnings(
    sample: ResourceSample,
    policy: ResolvedExecutionPolicy,
    emitted: set[str],
    *,
    enforce_process_tree_rss_limit: bool = False,
) -> list[str]:
    candidates = {
        RAM_PROCESS_LIMIT: (
            enforce_process_tree_rss_limit
            and sample.process_tree_rss_bytes
            >= int(policy.memory.warn_process_tree_rss_gb * GIB),
            "process-tree RAM crossed warning threshold",
        ),
        GPU_PROCESS_LIMIT: (
            sample.process_gpu_bytes is not None
            and sample.process_gpu_bytes >= int(policy.gpu.warn_process_vram_gb * GIB),
            "process GPU memory crossed warning threshold",
        ),
        DISK_RESULTS_LIMIT: (
            sample.results_free_disk_bytes
            <= int(policy.disk.minimum_free_results_gb * 1.25 * GIB),
            "results-volume free space is within 25% of the abort floor",
        ),
        DISK_TEMP_LIMIT: (
            sample.temp_free_disk_bytes
            <= int(policy.disk.minimum_free_temp_gb * 1.25 * GIB),
            "temporary-volume free space is within 25% of the abort floor",
        ),
    }
    messages = []
    for code, (crossed, message) in candidates.items():
        if crossed and code not in emitted:
            emitted.add(code)
            messages.append(f"{code}: {message}")
    return messages


class _OwnedProcessRegistry:
    """Track only the exact spawned worker and descendants observed by parentage."""

    def __init__(
        self,
        root_pid: int,
        *,
        association: str,
        psutil_module: Any | None = None,
    ) -> None:
        if psutil_module is None:
            import psutil as psutil_module
        self.psutil = psutil_module
        self.root_pid = int(root_pid)
        self.association = str(association)
        self._records: dict[int, dict[str, Any]] = {}
        self._unverified_pids: set[int] = set()
        try:
            root = self.psutil.Process(self.root_pid)
            self._remember(root, parent_pid=os.getpid(), relationship="spawned_worker")
        except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
            pass

    def _remember(self, process: Any, *, parent_pid: int, relationship: str) -> None:
        pid = int(process.pid)
        if pid in self._records:
            return
        try:
            created = float(process.create_time())
        except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
            return
        self._records[pid] = {
            "pid": pid,
            "parent_pid_at_discovery": int(parent_pid),
            "create_time": created,
            "relationship": relationship,
            "run_association": self.association,
        }

    def _matching_process(self, record: Mapping[str, Any]) -> Any | None:
        pid = int(record["pid"])
        try:
            process = self.psutil.Process(pid)
            if abs(float(process.create_time()) - float(record["create_time"])) > 1e-3:
                self._unverified_pids.discard(pid)
                return None
            self._unverified_pids.discard(pid)
            return process
        except self.psutil.NoSuchProcess:
            self._unverified_pids.discard(pid)
            return None
        except self.psutil.AccessDenied:
            self._unverified_pids.add(pid)
            return None

    def refresh(self) -> None:
        for record in list(self._records.values()):
            parent = self._matching_process(record)
            if parent is None:
                continue
            try:
                children = parent.children(recursive=True)
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                continue
            for child in children:
                try:
                    parent_pid = int(child.ppid())
                except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                    parent_pid = int(parent.pid)
                if parent_pid not in self._records and int(parent.pid) not in self._records:
                    continue
                self._remember(
                    child,
                    parent_pid=parent_pid,
                    relationship="observed_descendant",
                )

    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for _, item in sorted(self._records.items()))

    def alive(self) -> list[Any]:
        self.refresh()
        alive = []
        for record in self._records.values():
            process = self._matching_process(record)
            if process is not None:
                try:
                    if process.is_running():
                        alive.append(process)
                except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                    continue
        return alive

    def _ordered(self, processes: list[Any]) -> list[Any]:
        return sorted(processes, key=lambda item: int(item.pid) == self.root_pid)

    def suspend_phase(self) -> tuple[int, ...]:
        processes = self._ordered(self.alive())
        suspended: list[int] = []
        for process in processes:
            try:
                process.suspend()
                suspended.append(int(process.pid))
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                for prior in reversed(processes):
                    if int(prior.pid) not in suspended:
                        continue
                    try:
                        prior.resume()
                    except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                        pass
                return ()
        return tuple(suspended)

    def resume_phase(self) -> tuple[int, ...]:
        resumed: list[int] = []
        # Resume the spawned root before its descendants.
        processes = list(reversed(self._ordered(self.alive())))
        for process in processes:
            try:
                process.resume()
                resumed.append(int(process.pid))
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                continue
        return tuple(resumed)

    def terminate_phase(self, *, timeout_seconds: float) -> list[Any]:
        processes = self._ordered(self.alive())
        for process in processes:
            try:
                process.terminate()
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                continue
        if not processes:
            return []
        _, alive = self.psutil.wait_procs(
            processes, timeout=max(0.0, float(timeout_seconds))
        )
        return [process for process in alive if self._matching_process(
            self._records.get(int(process.pid), {})
        ) is not None]

    def kill_phase(self, *, timeout_seconds: float) -> list[Any]:
        processes = self._ordered(self.alive())
        for process in processes:
            try:
                process.kill()
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                continue
        if not processes:
            return []
        _, alive = self.psutil.wait_procs(
            processes, timeout=max(0.0, float(timeout_seconds))
        )
        return [process for process in alive if self._matching_process(
            self._records.get(int(process.pid), {})
        ) is not None]

    def survivor_records(self) -> tuple[dict[str, Any], ...]:
        alive_pids = {int(process.pid) for process in self.alive()} | set(
            self._unverified_pids
        )
        return tuple(
            dict(record)
            for pid, record in sorted(self._records.items())
            if pid in alive_pids
        )


def _lifecycle_event(
    lifecycle: list[dict[str, Any]],
    state: str,
    *,
    supervisor_started: float,
    detail: str,
    pids: tuple[int, ...] = (),
    log_context: Mapping[str, Any] | None = None,
) -> None:
    elapsed = monotonic() - supervisor_started
    payload = {
        "state": str(state),
        "timestamp_utc": _utc_now(),
        "elapsed_seconds": elapsed,
        "detail": str(detail),
        "pids": list(map(int, pids)),
    }
    lifecycle.append(payload)
    level = (
        "ERROR"
        if state == "WORKER_TREE_TERMINATION_FAILED"
        else "WARNING"
        if state in {
            "RESOURCE_STOP_LATCHED",
            "WALL_CLOCK_STOP_LATCHED",
            "TERMINATE_PROCESS_TREE",
            "FORCE_KILL_REMAINDERS",
            "SECONDARY_TERMINATION_EVENT",
        }
        else "INFO"
    )
    emit_contextual_research_event(
        "supervisor_lifecycle",
        dict(log_context or {}),
        level=level,
        message=f"Supervisor lifecycle state: {state} ({detail})",
        priority=True,
        lifecycle_state=state,
        elapsed_supervisor_seconds=elapsed,
        owned_pids=list(map(int, pids)),
    )


def _bounded_join(
    process: Any,
    *,
    timeout_seconds: float,
    recorder: _StopCauseRecorder,
    lifecycle: list[dict[str, Any]],
    supervisor_started: float,
    log_context: Mapping[str, Any] | None = None,
) -> None:
    try:
        process.join(timeout=max(0.0, float(timeout_seconds)))
    except KeyboardInterrupt:
        recorder.observe(
            MANUAL_INTERRUPT,
            elapsed_seconds=monotonic() - supervisor_started,
            detail="user KeyboardInterrupt during bounded worker join",
        )
        _lifecycle_event(
            lifecycle,
            "SECONDARY_TERMINATION_EVENT",
            supervisor_started=supervisor_started,
            detail="user_keyboard_interrupt",
            log_context=log_context,
        )


def _shutdown_owned_worker(
    *,
    process: Any,
    stop_event: Any,
    ownership: _OwnedProcessRegistry,
    policy: ResolvedExecutionPolicy,
    recorder: _StopCauseRecorder,
    lifecycle: list[dict[str, Any]],
    supervisor_started: float,
    log_context: Mapping[str, Any] | None = None,
) -> tuple[bool, tuple[dict[str, Any], ...], str | None, float, bool]:
    """Run finite cooperative, terminate, and force-kill phases."""

    shutdown_started = monotonic()
    stop_event.set()
    _lifecycle_event(
        lifecycle,
        "COOPERATIVE_STOP_REQUESTED",
        supervisor_started=supervisor_started,
        detail=f"primary_stop_code={recorder.primary}",
        log_context=log_context,
    )
    _lifecycle_event(
        lifecycle,
        "GRACE_PERIOD",
        supervisor_started=supervisor_started,
        detail=(
            f"timeout_seconds={policy.monitoring.graceful_stop_timeout_seconds}"
        ),
        log_context=log_context,
    )
    _bounded_join(
        process,
        timeout_seconds=policy.monitoring.graceful_stop_timeout_seconds,
        recorder=recorder,
        lifecycle=lifecycle,
        supervisor_started=supervisor_started,
        log_context=log_context,
    )
    alive = ownership.alive()
    graceful_completed = not alive and not process.is_alive()
    if alive or process.is_alive():
        pids = tuple(sorted({int(item.pid) for item in alive} | {int(process.pid)}))
        _lifecycle_event(
            lifecycle,
            "TERMINATE_PROCESS_TREE",
            supervisor_started=supervisor_started,
            detail=(
                f"timeout_seconds={policy.monitoring.forced_stop_timeout_seconds}"
            ),
            pids=pids,
            log_context=log_context,
        )
        try:
            alive = ownership.terminate_phase(
                timeout_seconds=policy.monitoring.forced_stop_timeout_seconds
            )
        except KeyboardInterrupt:
            recorder.observe(
                MANUAL_INTERRUPT,
                elapsed_seconds=monotonic() - supervisor_started,
                detail="user KeyboardInterrupt during terminate wait",
            )
            alive = ownership.alive()
        _bounded_join(
            process,
            timeout_seconds=0.0,
            recorder=recorder,
            lifecycle=lifecycle,
            supervisor_started=supervisor_started,
            log_context=log_context,
        )
    if alive or process.is_alive():
        pids = tuple(sorted({int(item.pid) for item in alive} | {int(process.pid)}))
        _lifecycle_event(
            lifecycle,
            "FORCE_KILL_REMAINDERS",
            supervisor_started=supervisor_started,
            detail=(
                f"timeout_seconds={policy.monitoring.forced_stop_timeout_seconds}"
            ),
            pids=pids,
            log_context=log_context,
        )
        try:
            alive = ownership.kill_phase(
                timeout_seconds=policy.monitoring.forced_stop_timeout_seconds
            )
        except KeyboardInterrupt:
            recorder.observe(
                MANUAL_INTERRUPT,
                elapsed_seconds=monotonic() - supervisor_started,
                detail="user KeyboardInterrupt during force-kill wait",
            )
            alive = ownership.alive()
        _bounded_join(
            process,
            timeout_seconds=0.0,
            recorder=recorder,
            lifecycle=lifecycle,
            supervisor_started=supervisor_started,
            log_context=log_context,
        )
    survivors = ownership.survivor_records()
    if survivors or process.is_alive():
        condition = WORKER_TREE_TERMINATION_FAILED
        _lifecycle_event(
            lifecycle,
            "WORKER_TREE_TERMINATION_FAILED",
            supervisor_started=supervisor_started,
            detail="owned process identities remain alive after force-kill wait",
            pids=tuple(int(item["pid"]) for item in survivors),
            log_context=log_context,
        )
    else:
        condition = None
        _lifecycle_event(
            lifecycle,
            "EXIT_CONFIRMED",
            supervisor_started=supervisor_started,
            detail="owned worker process tree is absent",
            log_context=log_context,
        )
    return (
        condition is None,
        survivors,
        condition,
        monotonic() - shutdown_started,
        graceful_completed,
    )


def _close_queue_nonblocking(target_queue: Any) -> bool:
    """Release one multiprocessing queue without joining its feeder thread."""

    ok = True
    try:
        target_queue.cancel_join_thread()
    except (AttributeError, OSError, ValueError):
        ok = False
    try:
        target_queue.close()
    except (AttributeError, OSError, ValueError):
        ok = False
    return ok


def terminate_process_tree(pid: int, *, timeout_seconds: float = 10.0) -> bool:
    """Terminate one exact PID/create-time tree with separate terminate/kill waits."""

    registry = _OwnedProcessRegistry(
        int(pid), association=f"direct-tree-termination:{int(pid)}"
    )
    registry.refresh()
    registry.terminate_phase(timeout_seconds=max(0.0, float(timeout_seconds)))
    registry.kill_phase(timeout_seconds=max(0.0, float(timeout_seconds)))
    return not registry.survivor_records()


def wait_for_inter_run_readiness(
    *,
    policy: ResolvedExecutionPolicy,
    results_root: str | Path,
    temp_root: str | Path,
    timeout_seconds: float | None = None,
    psutil_module: Any | None = None,
    ram_control_policy: ResolvedRamControlPolicy | None = None,
    monotonic_fn: Any = monotonic,
    sleep_fn: Any = sleep,
) -> InterRunReadiness:
    """Wait indefinitely for RAM recovery; retain terminal disk/process checks."""

    del timeout_seconds
    if psutil_module is None:
        import psutil as psutil_module
    memory = psutil_module.virtual_memory()
    ram_control = ram_control_policy or default_ram_control_policy(
        total_physical_ram_bytes=int(memory.total)
    )
    started = monotonic_fn()
    ram_state = RamWaitState(ram_control)
    samples = 0
    last_code: str | None = None
    while True:
        samples += 1
        active_children = tuple(
            sorted(
                int(child.pid)
                for child in multiprocessing.active_children()
                if child.is_alive()
            )
        )
        parent = psutil_module.Process(os.getpid())
        parent_rss = int(parent.memory_info().rss)
        available = int(psutil_module.virtual_memory().available)
        results_free = int(shutil.disk_usage(results_root).free)
        temp_free = int(shutil.disk_usage(temp_root).free)
        if active_children:
            last_code = WORKER_TREE_TERMINATION_FAILED
        elif results_free <= int(policy.disk.minimum_free_results_gb * GIB):
            last_code = DISK_RESULTS_LIMIT
        elif temp_free <= int(policy.disk.minimum_free_temp_gb * GIB):
            last_code = DISK_TEMP_LIMIT
        else:
            last_code = None
        now = monotonic_fn()
        transition = ram_state.observe(available, now=now)
        if transition is not None:
            event = {
                "wait_started": "ram_wait_started",
                "wait_periodic": "ram_wait_periodic",
                "resumed": "ram_resumed",
            }[transition.action]
            emit_research_event(
                event,
                message=(
                    "RAM recovered stably; pipeline readiness resumed"
                    if transition.action == "resumed"
                    else "Pipeline is waiting for RAM before the next worker"
                ),
                priority=True,
                stage="inter_run_readiness",
                pause_mode="parent_readiness_barrier",
                waiting_seconds=transition.waiting_seconds,
                process_tree_rss_bytes=parent_rss,
                system_available_ram_bytes=available,
                consecutive_recovery_checks=(
                    transition.consecutive_recovery_checks
                ),
                emergency_ram_margin_bytes=ram_control.emergency_margin_bytes,
                ram_recovery_threshold_bytes=ram_control.recovery_threshold_bytes,
            )
        elapsed = monotonic_fn() - started
        ram_ready = not ram_state.waiting
        if last_code is not None or (ram_ready and not active_children):
            result = InterRunReadiness(
                ready=last_code is None and ram_ready,
                stop_code=last_code,
                elapsed_seconds=elapsed,
                sample_count=samples,
                parent_pid=os.getpid(),
                parent_rss_bytes=parent_rss,
                system_available_ram_bytes=available,
                results_free_disk_bytes=results_free,
                temp_free_disk_bytes=temp_free,
                active_child_pids=active_children,
                total_ram_wait_seconds=ram_state.waiting_seconds(
                    now=monotonic_fn()
                ),
                ram_wait_count=ram_state.wait_count,
            )
            emit_research_event(
                "inter_run_readiness_result",
                level="INFO" if result.ready else "ERROR",
                message=(
                    "Inter-run resource readiness passed"
                    if result.ready
                    else f"Inter-run resource readiness blocked: {result.stop_code}"
                ),
                priority=True,
                ready=result.ready,
                stop_code=result.stop_code,
                elapsed_stage_seconds=result.elapsed_seconds,
                sample_count=result.sample_count,
                parent_pid=result.parent_pid,
                parent_rss_bytes=result.parent_rss_bytes,
                system_available_ram_bytes=result.system_available_ram_bytes,
                results_free_disk_bytes=result.results_free_disk_bytes,
                temp_free_disk_bytes=result.temp_free_disk_bytes,
                active_child_pids=result.active_child_pids,
            )
            return result
        sleep_fn(ram_control.check_interval_seconds)


def _supervisor_logging_context(
    worker_kwargs: Mapping[str, Any], association: str
) -> dict[str, Any]:
    spec = worker_kwargs.get("spec", {})
    if not isinstance(spec, Mapping):
        spec = {}
    run_directory = worker_kwargs.get("run_directory")
    run_id = spec.get("run_id") or spec.get("cell_id")
    if run_id is None and run_directory:
        run_id = Path(str(run_directory)).name
    phase = worker_kwargs.get("phase")
    return {
        "run_association": association,
        "run_id": run_id,
        "pilot_cell": spec.get("cell_id"),
        "cell_index": spec.get("cell_index"),
        "dataset": spec.get("dataset"),
        "model": spec.get("model"),
        "seed": spec.get("seed"),
        "phase": str(phase).upper() if phase is not None else None,
        "selector": spec.get("method_id"),
    }


def _stage_activity_message(stage: str | None, *, heartbeat: bool) -> str:
    label = str(stage or "unknown")
    if label in {"voter_boruta", "pilot_boruta_random_forest"} and heartbeat:
        return "Boruta fit active; internal iteration unavailable."
    if label == "pilot_rfe_catboost" and heartbeat:
        return "CatBoost RFE remains active; elimination is supervised externally."
    if label == "pilot_catboost_shap" and heartbeat:
        return "CatBoost fit or native SHAP calculation remains active."
    if heartbeat:
        return f"Stage {label} remains active"
    return f"Stage {label} started"


def supervise_worker(
    *,
    worker_target: str,
    worker_kwargs: Mapping[str, Any] | None,
    policy: ResolvedExecutionPolicy,
    results_root: str | Path,
    temp_root: str | Path,
    sampler_factory: Any = ProcessTreeSampler,
    run_association: str | None = None,
    heartbeat_interval_seconds: float = STAGE_HEARTBEAT_INTERVAL_SECONDS,
    max_wall_clock_seconds: float | None = None,
    stage_wall_clock_limits_seconds: Mapping[str, float] | None = None,
    enforce_memory_limits: bool = False,
    ram_control_policy: ResolvedRamControlPolicy | None = None,
) -> SupervisorResult:
    """Execute expensive work in one spawned child and supervise its process tree."""

    apply_thread_environment(policy.parallelism.estimator_threads)
    if max_wall_clock_seconds is not None and float(max_wall_clock_seconds) <= 0:
        raise ValueError("max_wall_clock_seconds must be positive or None")
    resolved_stage_wall_clock_limits = {
        str(stage_name): float(limit)
        for stage_name, limit in dict(stage_wall_clock_limits_seconds or {}).items()
    }
    if any(limit <= 0 for limit in resolved_stage_wall_clock_limits.values()):
        raise ValueError("stage wall-clock limits must be positive")
    association = run_association or f"supervised-worker:{uuid.uuid4()}"
    resolved_worker_kwargs = dict(worker_kwargs or {})
    supervisor_log_context = _supervisor_logging_context(
        resolved_worker_kwargs, association
    )
    worker_logging_transport = active_worker_transport()
    heartbeat_interval = min(
        STAGE_HEARTBEAT_INTERVAL_SECONDS,
        max(0.01, float(heartbeat_interval_seconds)),
    )
    import psutil

    ram_control = ram_control_policy or default_ram_control_policy(
        total_physical_ram_bytes=int(psutil.virtual_memory().total)
    )
    ram_state = RamWaitState(ram_control)
    parent_rss_before = int(psutil.Process(os.getpid()).memory_info().rss)
    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    ram_ready_event = context.Event()
    ram_ready_event.set()
    result_queue = context.Queue(maxsize=1)
    stage_queue = context.Queue(maxsize=256)
    process = context.Process(
        target=_worker_entry,
        args=(
            worker_target,
            resolved_worker_kwargs,
            stop_event,
            result_queue,
            stage_queue,
            ram_ready_event,
            ram_control.recovery_threshold_bytes,
            association,
            worker_logging_transport,
            supervisor_log_context,
        ),
        daemon=False,
    )
    emit_contextual_research_event(
        "worker_spawn_requested",
        supervisor_log_context,
        message="Starting supervised research worker",
        priority=True,
        worker_target=worker_target,
        parent_pid=os.getpid(),
    )
    process.start()
    suspension_inclusive_started = suspension_inclusive_monotonic()
    supervisor_started = monotonic()
    emit_contextual_research_event(
        "worker_spawned",
        supervisor_log_context,
        message="Supervised research worker spawned",
        priority=True,
        worker_pid=int(process.pid),
        parent_pid=os.getpid(),
    )
    ownership = _OwnedProcessRegistry(process.pid, association=association)
    sampler = sampler_factory(results_root=results_root, temp_root=temp_root)
    samples: list[ResourceSample] = []
    warnings: list[str] = []
    emitted_warnings: set[str] = set()
    stage: str | None = "initialized"
    fold_id: str | int | None = None
    recorder = _StopCauseRecorder()
    lifecycle: list[dict[str, Any]] = []
    ram_wait_events: list[dict[str, Any]] = []
    _lifecycle_event(
        lifecycle,
        "RUNNING",
        supervisor_started=supervisor_started,
        detail=f"run_association={association}",
        pids=(int(process.pid),),
        log_context=supervisor_log_context,
    )
    child_cleanup_confirmed = True
    queue_cleanup_confirmed = True
    survivors: tuple[dict[str, Any], ...] = ()
    termination_condition: str | None = None
    shutdown_elapsed: float | None = None
    graceful_stop_completed: bool | None = None
    message: Mapping[str, Any] | None = None
    stage_started_at = supervisor_started
    stage_waiting_baseline = 0.0
    last_heartbeat_at = supervisor_started
    opaque_tree_suspended = False
    ram_boundary_pending = False
    ram_pause_unavailable = False
    computation_ended_at: float | None = None
    suspension_inclusive_computation_ended_at: float | None = None

    def stage_active_elapsed(now: float) -> float:
        waiting_since_stage = max(
            0.0,
            ram_state.waiting_seconds(now=now) - stage_waiting_baseline,
        )
        return max(0.0, now - stage_started_at - waiting_since_stage)

    def handle_ram_transition(
        transition: Any,
        sample: ResourceSample,
        *,
        now: float,
    ) -> None:
        nonlocal opaque_tree_suspended, last_heartbeat_at
        nonlocal ram_boundary_pending, ram_pause_unavailable
        action = str(transition.action)
        pause_mode = (
            "worker_stage_boundary"
            if ram_boundary_pending
            else "cooperative_boundary"
            if stage in _COOPERATIVE_RAM_STAGES
            else "process_tree_suspend"
        )
        if action == "wait_started":
            ram_ready_event.clear()
            if pause_mode == "process_tree_suspend":
                suspended = ownership.suspend_phase()
                opaque_tree_suspended = bool(suspended)
                _lifecycle_event(
                    lifecycle,
                    "OPAQUE_STAGE_PROCESS_TREE_SUSPENDED",
                    supervisor_started=supervisor_started,
                    detail=f"stage={stage}; pids={list(suspended)}",
                    pids=suspended,
                    log_context=supervisor_log_context,
                )
                if not suspended and process.is_alive():
                    ram_pause_unavailable = True
                    recorder.observe(
                        RAM_PAUSE_UNAVAILABLE,
                        elapsed_seconds=now - supervisor_started,
                        detail=(
                            "owned process tree could not be suspended safely "
                            f"during opaque stage {stage}"
                        ),
                    )
        elif action == "resumed":
            if opaque_tree_suspended:
                resumed = ownership.resume_phase()
                opaque_tree_suspended = False
                _lifecycle_event(
                    lifecycle,
                    "OPAQUE_STAGE_PROCESS_TREE_RESUMED",
                    supervisor_started=supervisor_started,
                    detail=f"stage={stage}; pids={list(resumed)}",
                    pids=resumed,
                    log_context=supervisor_log_context,
                )
            ram_ready_event.set()
            ram_boundary_pending = False
            last_heartbeat_at = now

        event_name = {
            "wait_started": "ram_wait_started",
            "wait_periodic": "ram_wait_periodic",
            "resumed": "ram_resumed",
        }[action]
        record = {
            "action": action,
            "timestamp_utc": _utc_now(),
            "elapsed_supervisor_seconds": now - supervisor_started,
            "waiting_seconds": float(transition.waiting_seconds),
            "stage": stage,
            "fold_id": fold_id,
            "pause_mode": pause_mode,
            "worker_pid": sample.worker_pid,
            "process_tree_rss_bytes": sample.process_tree_rss_bytes,
            "system_available_ram_bytes": sample.system_available_ram_bytes,
            "consecutive_recovery_checks": (
                transition.consecutive_recovery_checks
            ),
        }
        ram_wait_events.append(record)
        emit_contextual_research_event(
            event_name,
            supervisor_log_context,
            message=(
                "RAM recovered stably; supervised work resumed"
                if action == "resumed"
                else "Supervised work is waiting for RAM"
            ),
            priority=True,
            **record,
            emergency_ram_margin_bytes=ram_control.emergency_margin_bytes,
            ram_recovery_threshold_bytes=ram_control.recovery_threshold_bytes,
            ram_recovery_required_checks=ram_control.recovery_consecutive_checks,
        )

    def on_stage_update(
        previous_stage: str | None,
        previous_fold: Any,
        next_stage: str | None,
        next_fold: Any,
        update: Mapping[str, Any],
    ) -> None:
        nonlocal stage_started_at, stage_waiting_baseline, last_heartbeat_at
        nonlocal ram_boundary_pending
        if bool(update.get("ram_recovery_barrier")):
            ram_boundary_pending = True
        if next_stage == previous_stage and next_fold == previous_fold:
            return
        now = monotonic()
        if previous_stage not in {None, "initialized"}:
            emit_contextual_research_event(
                "stage_completed",
                supervisor_log_context,
                message=f"Stage {previous_stage} completed",
                priority=True,
                fold_id=previous_fold,
                stage=previous_stage,
                component=_STAGE_COMPONENTS.get(str(previous_stage), previous_stage),
                elapsed_stage_seconds=stage_active_elapsed(now),
                worker_pid=int(process.pid),
            )
        stage_started_at = now
        stage_waiting_baseline = ram_state.waiting_seconds(now=now)
        last_heartbeat_at = now
        details = {
            str(key): value
            for key, value in update.items()
            if key not in {"stage", "fold_id"}
        }
        stage_component = details.pop(
            "component", _STAGE_COMPONENTS.get(str(next_stage), next_stage)
        )
        emit_contextual_research_event(
            "stage_started",
            details,
            supervisor_log_context,
            message=_stage_activity_message(next_stage, heartbeat=False),
            priority=True,
            fold_id=next_fold,
            stage=next_stage,
            component=stage_component,
            worker_pid=int(process.pid),
        )

    try:
        while process.is_alive():
            ownership.refresh()
            stage, fold_id = _drain_stage_queue(
                stage_queue, stage, fold_id, on_update=on_stage_update
            )
            if message is None:
                try:
                    candidate = result_queue.get_nowait()
                    if isinstance(candidate, Mapping):
                        message = candidate
                except (queue.Empty, EOFError, OSError, ValueError):
                    pass
            sample = sampler.sample(process.pid, stage=stage, fold_id=fold_id)
            samples.append(sample)
            now = monotonic()
            ram_transition = None
            if ram_boundary_pending and not ram_state.waiting:
                if (
                    sample.system_available_ram_bytes
                    < ram_control.recovery_threshold_bytes
                ):
                    ram_transition = ram_state.begin_wait(now=now)
                else:
                    ram_ready_event.set()
                    ram_boundary_pending = False
                    _lifecycle_event(
                        lifecycle,
                        "RAM_RECOVERY_BOUNDARY_PASSED",
                        supervisor_started=supervisor_started,
                        detail=f"stage={stage}; no wait required",
                        pids=(int(process.pid),),
                        log_context=supervisor_log_context,
                    )
            if ram_transition is None:
                ram_transition = ram_state.observe(
                    sample.system_available_ram_bytes,
                    now=now,
                )
            if ram_transition is not None:
                handle_ram_transition(ram_transition, sample, now=now)
            if ram_pause_unavailable:
                computation_ended_at = monotonic()
                suspension_inclusive_computation_ended_at = (
                    suspension_inclusive_monotonic()
                )
                ram_ready_event.set()
                (
                    child_cleanup_confirmed,
                    survivors,
                    termination_condition,
                    shutdown_elapsed,
                    graceful_stop_completed,
                ) = _shutdown_owned_worker(
                    process=process,
                    stop_event=stop_event,
                    ownership=ownership,
                    policy=policy,
                    recorder=recorder,
                    lifecycle=lifecycle,
                    supervisor_started=supervisor_started,
                    log_context={
                        **supervisor_log_context,
                        "stop_code": recorder.primary,
                        "fold_id": fold_id,
                        "stage": stage,
                        "worker_pid": int(process.pid),
                    },
                )
                break
            for warning_message in _new_warnings(
                sample,
                policy,
                emitted_warnings,
                enforce_process_tree_rss_limit=enforce_memory_limits,
            ):
                warnings.append(warning_message)
                logged = emit_contextual_research_event(
                    "resource_warning",
                    supervisor_log_context,
                    level="WARNING",
                    message=f"RESOURCE_WARNING {warning_message}",
                    priority=True,
                    warning_code=warning_message.split(":", 1)[0],
                    fold_id=fold_id,
                    stage=stage,
                    component=_STAGE_COMPONENTS.get(str(stage), stage),
                    worker_pid=sample.worker_pid,
                    worker_rss_bytes=sample.process_tree_rss_bytes,
                    system_available_ram_bytes=sample.system_available_ram_bytes,
                    parent_rss_bytes=int(
                        psutil.Process(os.getpid()).memory_info().rss
                    ),
                )
                if not logged:
                    logger.warning("RESOURCE_WARNING %s", warning_message)
            if (
                not ram_state.waiting
                and now - last_heartbeat_at >= heartbeat_interval
            ):
                emit_contextual_research_event(
                    "stage_heartbeat",
                    supervisor_log_context,
                    message=_stage_activity_message(stage, heartbeat=True),
                    fold_id=fold_id,
                    stage=stage,
                    component=_STAGE_COMPONENTS.get(str(stage), stage),
                    elapsed_stage_seconds=stage_active_elapsed(now),
                    worker_pid=sample.worker_pid,
                    worker_rss_bytes=sample.process_tree_rss_bytes,
                    parent_rss_bytes=int(
                        psutil.Process(os.getpid()).memory_info().rss
                    ),
                    system_available_ram_bytes=sample.system_available_ram_bytes,
                    process_tree_cpu_percent=sample.process_tree_cpu_percent,
                    process_tree_cpu_seconds=sample.process_tree_cpu_seconds,
                )
                last_heartbeat_at = now
            threshold = _classify_threshold(sample, policy)
            if threshold is None and enforce_memory_limits:
                if sample.process_tree_rss_bytes >= int(
                    policy.memory.abort_process_tree_rss_gb * GIB
                ):
                    threshold = RAM_PROCESS_LIMIT
                elif sample.system_available_ram_bytes <= int(
                    policy.memory.abort_if_system_available_below_gb * GIB
                ):
                    threshold = RAM_SYSTEM_HEADROOM
            if threshold is not None:
                computation_ended_at = monotonic()
                suspension_inclusive_computation_ended_at = (
                    suspension_inclusive_monotonic()
                )
                recorder.observe(
                    threshold,
                    elapsed_seconds=monotonic() - supervisor_started,
                    detail="resource abort threshold crossed",
                )
                _lifecycle_event(
                    lifecycle,
                    "RESOURCE_STOP_LATCHED",
                    supervisor_started=supervisor_started,
                    detail=f"primary_stop_code={threshold}",
                    pids=tuple(
                        sorted({sample.worker_pid, *sample.child_pids})
                    ),
                    log_context={
                        **supervisor_log_context,
                        "stop_code": threshold,
                        "fold_id": fold_id,
                        "stage": stage,
                        "worker_pid": sample.worker_pid,
                        "worker_rss_bytes": sample.process_tree_rss_bytes,
                        "system_available_ram_bytes": sample.system_available_ram_bytes,
                    },
                )
                if opaque_tree_suspended:
                    ownership.resume_phase()
                    opaque_tree_suspended = False
                ram_ready_event.set()
                (
                    child_cleanup_confirmed,
                    survivors,
                    termination_condition,
                    shutdown_elapsed,
                    graceful_stop_completed,
                ) = _shutdown_owned_worker(
                    process=process,
                    stop_event=stop_event,
                    ownership=ownership,
                    policy=policy,
                    recorder=recorder,
                    lifecycle=lifecycle,
                    supervisor_started=supervisor_started,
                    log_context={
                        **supervisor_log_context,
                        "stop_code": recorder.primary,
                        "fold_id": fold_id,
                        "stage": stage,
                        "worker_pid": int(process.pid),
                    },
                )
                break
            if (
                max_wall_clock_seconds is not None
                and ram_state.active_seconds(
                    started=supervisor_started, now=monotonic()
                )
                >= float(max_wall_clock_seconds)
            ):
                computation_ended_at = monotonic()
                suspension_inclusive_computation_ended_at = (
                    suspension_inclusive_monotonic()
                )
                recorder.observe(
                    WALL_CLOCK_LIMIT,
                    elapsed_seconds=ram_state.active_seconds(
                        started=supervisor_started, now=monotonic()
                    ),
                    detail=(
                        "per-worker wall-clock limit reached: "
                        f"{float(max_wall_clock_seconds):.3f} seconds"
                    ),
                )
                _lifecycle_event(
                    lifecycle,
                    "WALL_CLOCK_STOP_LATCHED",
                    supervisor_started=supervisor_started,
                    detail=f"primary_stop_code={WALL_CLOCK_LIMIT}",
                    pids=tuple(sorted({sample.worker_pid, *sample.child_pids})),
                    log_context={
                        **supervisor_log_context,
                        "stop_code": WALL_CLOCK_LIMIT,
                        "fold_id": fold_id,
                        "stage": stage,
                        "worker_pid": sample.worker_pid,
                        "worker_rss_bytes": sample.process_tree_rss_bytes,
                        "system_available_ram_bytes": sample.system_available_ram_bytes,
                        "max_wall_clock_seconds": float(max_wall_clock_seconds),
                    },
                )
                if opaque_tree_suspended:
                    ownership.resume_phase()
                    opaque_tree_suspended = False
                ram_ready_event.set()
                (
                    child_cleanup_confirmed,
                    survivors,
                    termination_condition,
                    shutdown_elapsed,
                    graceful_stop_completed,
                ) = _shutdown_owned_worker(
                    process=process,
                    stop_event=stop_event,
                    ownership=ownership,
                    policy=policy,
                    recorder=recorder,
                    lifecycle=lifecycle,
                    supervisor_started=supervisor_started,
                    log_context={
                        **supervisor_log_context,
                        "stop_code": recorder.primary,
                        "fold_id": fold_id,
                        "stage": stage,
                        "worker_pid": int(process.pid),
                    },
                )
                break
            stage_wall_clock_limit = resolved_stage_wall_clock_limits.get(
                str(stage)
            )
            if (
                stage_wall_clock_limit is not None
                and stage_active_elapsed(monotonic()) >= stage_wall_clock_limit
            ):
                computation_ended_at = monotonic()
                suspension_inclusive_computation_ended_at = (
                    suspension_inclusive_monotonic()
                )
                recorder.observe(
                    WALL_CLOCK_LIMIT,
                    elapsed_seconds=stage_active_elapsed(monotonic()),
                    detail=(
                        f"stage wall-clock limit reached for {stage}: "
                        f"{stage_wall_clock_limit:.3f} seconds"
                    ),
                )
                _lifecycle_event(
                    lifecycle,
                    "STAGE_WALL_CLOCK_STOP_LATCHED",
                    supervisor_started=supervisor_started,
                    detail=f"primary_stop_code={WALL_CLOCK_LIMIT}; stage={stage}",
                    pids=tuple(sorted({sample.worker_pid, *sample.child_pids})),
                    log_context={
                        **supervisor_log_context,
                        "stop_code": WALL_CLOCK_LIMIT,
                        "fold_id": fold_id,
                        "stage": stage,
                        "worker_pid": sample.worker_pid,
                        "worker_rss_bytes": sample.process_tree_rss_bytes,
                        "system_available_ram_bytes": sample.system_available_ram_bytes,
                        "stage_wall_clock_limit_seconds": stage_wall_clock_limit,
                    },
                )
                if opaque_tree_suspended:
                    ownership.resume_phase()
                    opaque_tree_suspended = False
                ram_ready_event.set()
                (
                    child_cleanup_confirmed,
                    survivors,
                    termination_condition,
                    shutdown_elapsed,
                    graceful_stop_completed,
                ) = _shutdown_owned_worker(
                    process=process,
                    stop_event=stop_event,
                    ownership=ownership,
                    policy=policy,
                    recorder=recorder,
                    lifecycle=lifecycle,
                    supervisor_started=supervisor_started,
                    log_context={
                        **supervisor_log_context,
                        "stop_code": recorder.primary,
                        "fold_id": fold_id,
                        "stage": stage,
                        "worker_pid": int(process.pid),
                    },
                )
                break
            sleep_seconds = float(ram_control.check_interval_seconds)
            if not ram_state.waiting:
                sleep_seconds = min(
                    sleep_seconds,
                    max(
                        0.001,
                        heartbeat_interval - (monotonic() - last_heartbeat_at),
                    ),
                )
                if max_wall_clock_seconds is not None:
                    remaining_active = (
                        float(max_wall_clock_seconds)
                        - ram_state.active_seconds(
                            started=supervisor_started,
                            now=monotonic(),
                        )
                    )
                    sleep_seconds = min(
                        sleep_seconds, max(0.001, remaining_active)
                    )
                stage_wall_clock_limit = resolved_stage_wall_clock_limits.get(
                    str(stage)
                )
                if stage_wall_clock_limit is not None:
                    sleep_seconds = min(
                        sleep_seconds,
                        max(
                            0.001,
                            stage_wall_clock_limit
                            - stage_active_elapsed(monotonic()),
                        ),
                    )
            process.join(timeout=sleep_seconds)
    except KeyboardInterrupt:
        computation_ended_at = monotonic()
        suspension_inclusive_computation_ended_at = (
            suspension_inclusive_monotonic()
        )
        recorder.observe(
            MANUAL_INTERRUPT,
            elapsed_seconds=monotonic() - supervisor_started,
            detail="user KeyboardInterrupt in supervisor",
        )
        if opaque_tree_suspended:
            ownership.resume_phase()
            opaque_tree_suspended = False
        ram_ready_event.set()
        (
            child_cleanup_confirmed,
            survivors,
            termination_condition,
            shutdown_elapsed,
            graceful_stop_completed,
        ) = _shutdown_owned_worker(
            process=process,
            stop_event=stop_event,
            ownership=ownership,
            policy=policy,
            recorder=recorder,
            lifecycle=lifecycle,
            supervisor_started=supervisor_started,
            log_context={
                **supervisor_log_context,
                "stop_code": recorder.primary,
                "fold_id": fold_id,
                "stage": stage,
                "worker_pid": int(process.pid),
            },
        )
    finally:
        if computation_ended_at is None:
            computation_ended_at = monotonic()
            suspension_inclusive_computation_ended_at = (
                suspension_inclusive_monotonic()
            )
        if opaque_tree_suspended:
            ownership.resume_phase()
            opaque_tree_suspended = False
        ram_ready_event.set()
        ownership.refresh()
        if shutdown_elapsed is None and (process.is_alive() or ownership.alive()):
            if recorder.primary is None:
                recorder.observe(
                    WORKER_CRASH,
                    elapsed_seconds=monotonic() - supervisor_started,
                    detail="supervisor finalizer found a live owned process tree",
                )
            (
                cleanup_ok,
                survivors,
                termination_condition,
                final_shutdown_elapsed,
                final_graceful,
            ) = _shutdown_owned_worker(
                process=process,
                stop_event=stop_event,
                ownership=ownership,
                policy=policy,
                recorder=recorder,
                lifecycle=lifecycle,
                supervisor_started=supervisor_started,
                log_context={
                    **supervisor_log_context,
                    "stop_code": recorder.primary,
                    "fold_id": fold_id,
                    "stage": stage,
                    "worker_pid": int(process.pid),
                },
            )
            child_cleanup_confirmed = child_cleanup_confirmed and cleanup_ok
            shutdown_elapsed = (
                final_shutdown_elapsed
                if shutdown_elapsed is None
                else shutdown_elapsed + final_shutdown_elapsed
            )
            if graceful_stop_completed is None:
                graceful_stop_completed = final_graceful
        gpu = getattr(sampler, "gpu", None)
        if gpu is not None:
            gpu.close()

    stage, fold_id = _drain_stage_queue(
        stage_queue, stage, fold_id, on_update=on_stage_update
    )
    try:
        if message is None:
            candidate = result_queue.get(timeout=0.5)
            if isinstance(candidate, Mapping):
                message = candidate
        for _ in range(15):
            try:
                candidate = result_queue.get_nowait()
                if isinstance(candidate, Mapping):
                    message = candidate
            except queue.Empty:
                break
    except (queue.Empty, EOFError, OSError, ValueError):
        pass
    queue_cleanup_confirmed = _close_queue_nonblocking(stage_queue)
    queue_cleanup_confirmed = (
        _close_queue_nonblocking(result_queue) and queue_cleanup_confirmed
    )
    return_value = message.get("value") if message and message.get("kind") == "result" else None
    worker_error = None
    if message and message.get("kind") in {"error", "interrupt"}:
        worker_error = str(message.get("error"))
    worker_traceback = str(message.get("traceback")) if message and message.get("traceback") else None
    if message and message.get("run_association") != association:
        worker_error = "worker result run association mismatch"
        return_value = None
        recorder.observe(
            WORKER_CRASH,
            elapsed_seconds=monotonic() - supervisor_started,
            detail=worker_error,
        )
    if message and message.get("kind") == "interrupt":
        recorder.observe(
            MANUAL_INTERRUPT,
            elapsed_seconds=monotonic() - supervisor_started,
            detail="worker reported KeyboardInterrupt",
        )
    elif message and message.get("kind") == "error":
        error_code = (
            MEMORY_ERROR
            if message.get("exception_class") == "MemoryError"
            else WORKER_CRASH
        )
        recorder.observe(
            error_code,
            elapsed_seconds=monotonic() - supervisor_started,
            detail=worker_error or "worker exception",
        )

    try:
        process.join(timeout=0.0)
    except (AssertionError, ValueError):
        pass
    worker_exit_code = process.exitcode

    stop_code = recorder.primary
    if termination_condition is not None:
        status = "failed"
        stop_code = stop_code or WORKER_TREE_TERMINATION_FAILED
        worker_error = worker_error or WORKER_TREE_TERMINATION_FAILED
    elif stop_code in {
        RAM_PROCESS_LIMIT,
        RAM_SYSTEM_HEADROOM,
        GPU_PROCESS_LIMIT,
        DISK_RESULTS_LIMIT,
        DISK_TEMP_LIMIT,
    }:
        status = "aborted_resource_limit"
    elif stop_code == MANUAL_INTERRUPT:
        status = "interrupted"
    elif stop_code == WALL_CLOCK_LIMIT:
        status = "timed_out"
    elif process.exitcode == 0 and message and message.get("kind") == "result":
        status = "completed"
    else:
        status = "failed"
        if stop_code is None:
            recorder.observe(
                WORKER_CRASH,
                elapsed_seconds=monotonic() - supervisor_started,
                detail="worker exited without a valid compact result",
            )
            stop_code = recorder.primary
        if worker_error is None:
            worker_error = f"worker exited with code {worker_exit_code} without a result"

    terminal_stage_event = {
        "completed": "stage_completed",
        "aborted_resource_limit": "stage_aborted",
        "interrupted": "stage_interrupted",
        "timed_out": "stage_aborted",
        "failed": "stage_failed",
    }[status]
    emit_contextual_research_event(
        terminal_stage_event,
        supervisor_log_context,
        level="INFO" if status == "completed" else "ERROR",
        message=f"Stage {stage} ended with worker status {status}",
        priority=True,
        fold_id=fold_id,
        stage=stage,
        component=_STAGE_COMPONENTS.get(str(stage), stage),
        elapsed_stage_seconds=stage_active_elapsed(monotonic()),
        worker_pid=int(process.pid),
        worker_exit_code=worker_exit_code,
        stop_code=stop_code,
        exception_class=(
            worker_error.split(":", 1)[0]
            if worker_error and ":" in worker_error
            else None
        ),
        traceback=worker_traceback,
    )

    rss_values = [item.process_tree_rss_bytes for item in samples]
    gpu_values = [item.process_gpu_bytes for item in samples if item.process_gpu_bytes is not None]
    available_values = [item.system_available_ram_bytes for item in samples]
    result_disk_values = [item.results_free_disk_bytes for item in samples]
    temp_disk_values = [item.temp_free_disk_bytes for item in samples]
    _lifecycle_event(
        lifecycle,
        "ARTIFACT_AND_STATE_FINALIZATION",
        supervisor_started=supervisor_started,
        detail=f"status={status}; stop_code={stop_code}",
        log_context={
            **supervisor_log_context,
            "stop_code": stop_code,
            "fold_id": fold_id,
            "stage": stage,
            "worker_pid": int(process.pid),
        },
    )
    del message
    gc.collect(0)
    parent = psutil.Process(os.getpid())
    parent_rss_after = int(parent.memory_info().rss)
    system_available_after = int(psutil.virtual_memory().available)
    owned_records = ownership.records()
    survivors = ownership.survivor_records()
    final_child_cleanup_confirmed = (
        child_cleanup_confirmed and not survivors and not process.is_alive()
    )
    emit_contextual_research_event(
        "worker_finalized",
        supervisor_log_context,
        level="INFO" if status == "completed" else "ERROR",
        message=f"Supervised worker finalized with status {status}",
        priority=True,
        worker_pid=int(process.pid),
        worker_exit_code=worker_exit_code,
        status=status,
        stop_code=stop_code,
        secondary_events=recorder.secondary,
        child_cleanup_confirmed=final_child_cleanup_confirmed,
        queue_cleanup_confirmed=queue_cleanup_confirmed,
        survivor_pids=[item["pid"] for item in survivors],
        parent_rss_before_bytes=parent_rss_before,
        parent_rss_after_bytes=parent_rss_after,
        system_available_ram_after_bytes=system_available_after,
    )
    if not process.is_alive():
        try:
            process.close()
        except (OSError, ValueError):
            pass
    return SupervisorResult(
        status=status,
        stop_code=stop_code,
        worker_exit_code=worker_exit_code,
        return_value=return_value,
        worker_error=worker_error,
        samples=tuple(samples),
        warnings=tuple(warnings),
        peak_process_tree_rss_bytes=max(rss_values, default=0),
        peak_process_gpu_bytes=max(gpu_values) if gpu_values else None,
        minimum_system_available_ram_bytes=min(available_values) if available_values else None,
        minimum_results_free_disk_bytes=min(result_disk_values) if result_disk_values else None,
        minimum_temp_free_disk_bytes=min(temp_disk_values) if temp_disk_values else None,
        child_cleanup_confirmed=final_child_cleanup_confirmed,
        final_stage=stage,
        final_fold_id=fold_id,
        primary_stop_code=recorder.primary,
        secondary_events=tuple(recorder.secondary),
        stop_lifecycle=tuple(lifecycle),
        owned_processes=owned_records,
        survivor_processes=survivors,
        termination_condition=termination_condition,
        graceful_stop_completed=graceful_stop_completed,
        shutdown_elapsed_seconds=shutdown_elapsed,
        parent_rss_before_bytes=parent_rss_before,
        parent_rss_after_bytes=parent_rss_after,
        system_available_ram_after_bytes=system_available_after,
        queue_cleanup_confirmed=queue_cleanup_confirmed,
        run_association=association,
        emergency_ram_margin_bytes=ram_control.emergency_margin_bytes,
        ram_recovery_threshold_bytes=ram_control.recovery_threshold_bytes,
        ram_check_interval_seconds=ram_control.check_interval_seconds,
        ram_log_interval_seconds=ram_control.log_interval_seconds,
        ram_recovery_consecutive_checks=ram_control.recovery_consecutive_checks,
        total_ram_wait_seconds=ram_state.waiting_seconds(
            now=computation_ended_at
        ),
        active_computation_seconds=ram_state.active_seconds(
            started=supervisor_started, now=computation_ended_at
        ),
        ram_wait_count=ram_state.wait_count,
        ram_wait_events=tuple(ram_wait_events),
        active_clock_source=ACTIVE_CLOCK_SOURCE,
        system_suspend_seconds=max(
            0.0,
            (
                suspension_inclusive_computation_ended_at
                - suspension_inclusive_started
                - max(0.0, computation_ended_at - supervisor_started)
            ),
        ),
        system_suspend_excluded_from_active_time=True,
        supervisor_awake_elapsed_seconds=max(
            0.0, computation_ended_at - supervisor_started
        ),
    )
