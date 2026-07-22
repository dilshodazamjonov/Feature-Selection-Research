"""Process-tree resource sampling and Windows-safe supervised execution."""

from __future__ import annotations

import importlib
import logging
import multiprocessing
import os
import queue
import shutil
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Mapping

from credit_risk_fs.experiments.resource_policy import (
    GIB,
    ResolvedExecutionPolicy,
    apply_thread_environment,
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

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["samples"] = [asdict(item) for item in self.samples]
        return payload


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
) -> None:
    try:
        function = _load_worker_target(target)
        value = function(stop_event=stop_event, stage_queue=stage_queue, **kwargs)
        result_queue.put({"kind": "result", "value": value})
    except KeyboardInterrupt:
        result_queue.put({"kind": "interrupt", "error": MANUAL_INTERRUPT})
        raise
    except Exception as exc:
        result_queue.put(
            {
                "kind": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        raise


def _drain_stage_queue(stage_queue: Any, stage: str | None, fold_id: Any) -> tuple[str | None, Any]:
    while True:
        try:
            update = stage_queue.get_nowait()
        except queue.Empty:
            return stage, fold_id
        if isinstance(update, Mapping):
            stage = str(update.get("stage")) if update.get("stage") is not None else stage
            fold_id = update.get("fold_id", fold_id)


def _classify_threshold(sample: ResourceSample, policy: ResolvedExecutionPolicy) -> str | None:
    if sample.process_tree_rss_bytes >= int(policy.memory.abort_process_tree_rss_gb * GIB):
        return RAM_PROCESS_LIMIT
    if sample.system_available_ram_bytes <= int(
        policy.memory.abort_if_system_available_below_gb * GIB
    ):
        return RAM_SYSTEM_HEADROOM
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
) -> list[str]:
    candidates = {
        RAM_PROCESS_LIMIT: (
            sample.process_tree_rss_bytes >= int(policy.memory.warn_process_tree_rss_gb * GIB),
            f"process tree RSS crossed warning threshold ({sample.process_tree_rss_bytes / GIB:.3f} GiB)",
        ),
        GPU_PROCESS_LIMIT: (
            sample.process_gpu_bytes is not None
            and sample.process_gpu_bytes >= int(policy.gpu.warn_process_vram_gb * GIB),
            "process GPU memory crossed warning threshold",
        ),
        RAM_SYSTEM_HEADROOM: (
            sample.system_available_ram_bytes
            <= int(policy.memory.reserve_system_ram_gb * GIB),
            "system-available RAM entered the configured reserve",
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


def terminate_process_tree(pid: int, *, timeout_seconds: float = 10.0) -> bool:
    """Terminate one exact worker tree and confirm no collected child remains."""

    import psutil

    try:
        root = psutil.Process(pid)
        descendants = root.children(recursive=True)
    except psutil.NoSuchProcess:
        return True
    processes = [*descendants, root]
    for process in processes:
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(processes, timeout=max(0.1, timeout_seconds))
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(alive, timeout=max(0.1, timeout_seconds))
    return not alive


def supervise_worker(
    *,
    worker_target: str,
    worker_kwargs: Mapping[str, Any] | None,
    policy: ResolvedExecutionPolicy,
    results_root: str | Path,
    temp_root: str | Path,
    sampler_factory: Any = ProcessTreeSampler,
) -> SupervisorResult:
    """Execute expensive work in one spawned child and supervise its process tree."""

    apply_thread_environment(policy.parallelism.estimator_threads)
    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    result_queue = context.Queue()
    stage_queue = context.Queue()
    process = context.Process(
        target=_worker_entry,
        args=(worker_target, dict(worker_kwargs or {}), stop_event, result_queue, stage_queue),
        daemon=False,
    )
    process.start()
    sampler = sampler_factory(results_root=results_root, temp_root=temp_root)
    samples: list[ResourceSample] = []
    warnings: list[str] = []
    emitted_warnings: set[str] = set()
    stage: str | None = "initialized"
    fold_id: str | int | None = None
    stop_code: str | None = None
    child_cleanup_confirmed = True

    try:
        while process.is_alive():
            stage, fold_id = _drain_stage_queue(stage_queue, stage, fold_id)
            sample = sampler.sample(process.pid, stage=stage, fold_id=fold_id)
            samples.append(sample)
            for message in _new_warnings(sample, policy, emitted_warnings):
                warnings.append(message)
                logger.warning(message)
            stop_code = _classify_threshold(sample, policy)
            if stop_code is not None:
                stop_event.set()
                process.join(timeout=policy.monitoring.graceful_stop_timeout_seconds)
                if process.is_alive():
                    child_cleanup_confirmed = terminate_process_tree(
                        process.pid,
                        timeout_seconds=policy.monitoring.forced_stop_timeout_seconds,
                    )
                    process.join(timeout=policy.monitoring.forced_stop_timeout_seconds)
                break
            stop_event.wait(policy.monitoring.sample_interval_seconds)
    except KeyboardInterrupt:
        stop_code = MANUAL_INTERRUPT
        stop_event.set()
        process.join(timeout=policy.monitoring.graceful_stop_timeout_seconds)
        if process.is_alive():
            child_cleanup_confirmed = terminate_process_tree(
                process.pid,
                timeout_seconds=policy.monitoring.forced_stop_timeout_seconds,
            )
            process.join(timeout=policy.monitoring.forced_stop_timeout_seconds)
    finally:
        if process.is_alive():
            child_cleanup_confirmed = terminate_process_tree(
                process.pid,
                timeout_seconds=policy.monitoring.forced_stop_timeout_seconds,
            ) and child_cleanup_confirmed
            process.join(timeout=policy.monitoring.forced_stop_timeout_seconds)
        sampler.gpu.close()

    stage, fold_id = _drain_stage_queue(stage_queue, stage, fold_id)
    message = None
    try:
        message = result_queue.get(timeout=0.5)
        while True:
            message = result_queue.get_nowait()
    except queue.Empty:
        pass
    return_value = message.get("value") if message and message.get("kind") == "result" else None
    worker_error = None
    if message and message.get("kind") in {"error", "interrupt"}:
        worker_error = str(message.get("error"))

    if stop_code in {
        RAM_PROCESS_LIMIT,
        RAM_SYSTEM_HEADROOM,
        GPU_PROCESS_LIMIT,
        DISK_RESULTS_LIMIT,
        DISK_TEMP_LIMIT,
    }:
        status = "aborted_resource_limit"
    elif stop_code == MANUAL_INTERRUPT:
        status = "interrupted"
    elif process.exitcode == 0 and message and message.get("kind") == "result":
        status = "completed"
    else:
        status = "failed"
        stop_code = stop_code or WORKER_CRASH
        if worker_error is None:
            worker_error = f"worker exited with code {process.exitcode} without a result"

    rss_values = [item.process_tree_rss_bytes for item in samples]
    gpu_values = [item.process_gpu_bytes for item in samples if item.process_gpu_bytes is not None]
    available_values = [item.system_available_ram_bytes for item in samples]
    result_disk_values = [item.results_free_disk_bytes for item in samples]
    temp_disk_values = [item.temp_free_disk_bytes for item in samples]
    return SupervisorResult(
        status=status,
        stop_code=stop_code,
        worker_exit_code=process.exitcode,
        return_value=return_value,
        worker_error=worker_error,
        samples=tuple(samples),
        warnings=tuple(warnings),
        peak_process_tree_rss_bytes=max(rss_values, default=0),
        peak_process_gpu_bytes=max(gpu_values) if gpu_values else None,
        minimum_system_available_ram_bytes=min(available_values) if available_values else None,
        minimum_results_free_disk_bytes=min(result_disk_values) if result_disk_values else None,
        minimum_temp_free_disk_bytes=min(temp_disk_values) if temp_disk_values else None,
        child_cleanup_confirmed=child_cleanup_confirmed and not process.is_alive(),
        final_stage=stage,
        final_fold_id=fold_id,
    )
