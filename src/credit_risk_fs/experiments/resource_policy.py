"""Versioned execution policy, hardware detection, and low-cost preflight checks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from credit_risk_fs.experiments.config import _parse_simple_yaml
from credit_risk_fs.experiments.result_paths import (
    AUDITED_LEGACY_RESULTS_ROOT,
    configured_legacy_results_root,
    reject_historical_write,
    resolve_results_root,
    validate_results_root_separation,
)


POLICY_SCHEMA_VERSION = "resource_safe_execution_policy_v1"
PREFLIGHT_SCHEMA_VERSION = "resource_safe_preflight_v1"
DEFAULT_POLICY_PATH = Path("configs/execution/local_laptop_safe_v1.yaml")
GIB = 1024**3


class ExecutionPolicyError(ValueError):
    """Raised when an execution policy is contradictory or unsafe."""


@dataclass(frozen=True, slots=True)
class ParallelismPolicy:
    concurrent_experiment_runs: int
    concurrent_folds: int
    data_loader_workers: int
    estimator_threads: int
    allow_nested_parallelism: bool


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    reserve_system_ram_gb: float
    warn_process_tree_rss_gb: float
    abort_process_tree_rss_gb: float
    abort_if_system_available_below_gb: float
    estimated_peak_safety_factor: float


@dataclass(frozen=True, slots=True)
class GpuPolicy:
    reserve_vram_gb: float
    warn_process_vram_gb: float
    abort_process_vram_gb: float
    require_gpu_telemetry_for_gpu_runs: bool


@dataclass(frozen=True, slots=True)
class DiskPolicy:
    minimum_free_results_gb: float
    minimum_free_temp_gb: float
    estimated_write_safety_factor: float


@dataclass(frozen=True, slots=True)
class MonitoringPolicy:
    sample_interval_seconds: float
    graceful_stop_timeout_seconds: float
    forced_stop_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class ExecutionPolicy:
    schema_version: str
    profile_name: str
    parallelism: ParallelismPolicy
    memory: MemoryPolicy
    gpu: GpuPolicy
    disk: DiskPolicy
    monitoring: MonitoringPolicy
    source_path: str
    configured: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("configured", None)
        return payload


@dataclass(frozen=True, slots=True)
class GpuCapacity:
    available: bool
    name: str | None
    total_vram_gb: float | None
    free_vram_gb: float | None
    driver_version: str | None
    cuda_visible: bool
    process_telemetry_available: bool
    telemetry_backend: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class HardwareCapacity:
    logical_cpu_count: int
    physical_cpu_count: int
    total_ram_gb: float
    available_ram_gb: float
    results_free_disk_gb: float
    temp_free_disk_gb: float
    results_volume: str
    temp_volume: str
    gpu: GpuCapacity


@dataclass(frozen=True, slots=True)
class ResolvedExecutionPolicy:
    schema_version: str
    profile_name: str
    parallelism: ParallelismPolicy
    memory: MemoryPolicy
    gpu: GpuPolicy
    disk: DiskPolicy
    monitoring: MonitoringPolicy
    configured_policy_path: str
    resolution_warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ExecutionPolicyError(f"execution policy section {key!r} must be a mapping")
    return value


def _positive(section: Mapping[str, Any], key: str) -> float:
    try:
        value = float(section[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExecutionPolicyError(f"execution policy {key!r} must be numeric") from exc
    if value <= 0:
        raise ExecutionPolicyError(f"execution policy {key!r} must be positive")
    return value


def _positive_int(section: Mapping[str, Any], key: str) -> int:
    value = _positive(section, key)
    if not value.is_integer():
        raise ExecutionPolicyError(f"execution policy {key!r} must be an integer")
    return int(value)


def load_execution_policy(
    repository_root: str | Path,
    config_path: str | Path = DEFAULT_POLICY_PATH,
) -> ExecutionPolicy:
    """Load one strict policy relative to an explicit repository root."""

    root = Path(repository_root).resolve()
    supplied = Path(config_path)
    path = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"execution policy is missing: {path}")
    payload = _parse_simple_yaml(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != POLICY_SCHEMA_VERSION:
        raise ExecutionPolicyError(
            f"unsupported execution-policy schema: {payload.get('schema_version')!r}"
        )
    profile_name = str(payload.get("profile_name", "")).strip()
    if not profile_name:
        raise ExecutionPolicyError("execution policy profile_name must not be empty")

    parallel = _mapping(payload, "parallelism")
    data_workers = int(parallel.get("data_loader_workers", -1))
    if data_workers < 0:
        raise ExecutionPolicyError("data_loader_workers must be zero or positive")
    allow_nested = parallel.get("allow_nested_parallelism")
    if not isinstance(allow_nested, bool):
        raise ExecutionPolicyError("allow_nested_parallelism must be boolean")
    parallelism = ParallelismPolicy(
        concurrent_experiment_runs=_positive_int(parallel, "concurrent_experiment_runs"),
        concurrent_folds=_positive_int(parallel, "concurrent_folds"),
        data_loader_workers=data_workers,
        estimator_threads=_positive_int(parallel, "estimator_threads"),
        allow_nested_parallelism=allow_nested,
    )
    _validate_parallelism(parallelism)

    memory_values = _mapping(payload, "memory")
    memory = MemoryPolicy(
        reserve_system_ram_gb=_positive(memory_values, "reserve_system_ram_gb"),
        warn_process_tree_rss_gb=_positive(memory_values, "warn_process_tree_rss_gb"),
        abort_process_tree_rss_gb=_positive(memory_values, "abort_process_tree_rss_gb"),
        abort_if_system_available_below_gb=_positive(
            memory_values, "abort_if_system_available_below_gb"
        ),
        estimated_peak_safety_factor=_positive(
            memory_values, "estimated_peak_safety_factor"
        ),
    )
    if memory.warn_process_tree_rss_gb >= memory.abort_process_tree_rss_gb:
        raise ExecutionPolicyError("RAM warning threshold must be below abort threshold")
    if memory.estimated_peak_safety_factor < 1.0:
        raise ExecutionPolicyError("memory safety factor must be at least 1.0")

    gpu_values = _mapping(payload, "gpu")
    required_telemetry = gpu_values.get("require_gpu_telemetry_for_gpu_runs")
    if not isinstance(required_telemetry, bool):
        raise ExecutionPolicyError("require_gpu_telemetry_for_gpu_runs must be boolean")
    gpu = GpuPolicy(
        reserve_vram_gb=_positive(gpu_values, "reserve_vram_gb"),
        warn_process_vram_gb=_positive(gpu_values, "warn_process_vram_gb"),
        abort_process_vram_gb=_positive(gpu_values, "abort_process_vram_gb"),
        require_gpu_telemetry_for_gpu_runs=required_telemetry,
    )
    if gpu.warn_process_vram_gb >= gpu.abort_process_vram_gb:
        raise ExecutionPolicyError("GPU warning threshold must be below abort threshold")

    disk_values = _mapping(payload, "disk")
    disk = DiskPolicy(
        minimum_free_results_gb=_positive(disk_values, "minimum_free_results_gb"),
        minimum_free_temp_gb=_positive(disk_values, "minimum_free_temp_gb"),
        estimated_write_safety_factor=_positive(
            disk_values, "estimated_write_safety_factor"
        ),
    )
    if disk.estimated_write_safety_factor < 1.0:
        raise ExecutionPolicyError("disk safety factor must be at least 1.0")

    monitor_values = _mapping(payload, "monitoring")
    monitoring = MonitoringPolicy(
        sample_interval_seconds=_positive(monitor_values, "sample_interval_seconds"),
        graceful_stop_timeout_seconds=_positive(
            monitor_values, "graceful_stop_timeout_seconds"
        ),
        forced_stop_timeout_seconds=_positive(
            monitor_values, "forced_stop_timeout_seconds"
        ),
    )
    return ExecutionPolicy(
        schema_version=POLICY_SCHEMA_VERSION,
        profile_name=profile_name,
        parallelism=parallelism,
        memory=memory,
        gpu=gpu,
        disk=disk,
        monitoring=monitoring,
        source_path=str(path),
        configured=dict(payload),
    )


def _validate_parallelism(policy: ParallelismPolicy, logical_cpus: int | None = None) -> None:
    if not policy.allow_nested_parallelism:
        nested_axes = sum(
            value > 1
            for value in (
                policy.concurrent_experiment_runs,
                policy.concurrent_folds,
                max(1, policy.data_loader_workers),
                policy.estimator_threads,
            )
        )
        if nested_axes > 1:
            raise ExecutionPolicyError(
                "nested parallelism is disabled; at most one concurrency axis may exceed one"
            )
    if policy.concurrent_experiment_runs > 1 and (
        policy.concurrent_folds != 1 or policy.estimator_threads != 1
    ):
        raise ExecutionPolicyError(
            "concurrent runs require one fold worker and one estimator thread"
        )
    product = (
        policy.concurrent_experiment_runs
        * policy.concurrent_folds
        * max(1, policy.data_loader_workers)
        * policy.estimator_threads
    )
    if logical_cpus is not None and product > logical_cpus:
        raise ExecutionPolicyError(
            f"parallelism product {product} exceeds {logical_cpus} logical CPUs"
        )


def _detect_gpu() -> GpuCapacity:
    try:
        import pynvml

        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() < 1:
            pynvml.nvmlShutdown()
            return GpuCapacity(False, None, None, None, None, False, False, "nvml")
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode("ascii", errors="replace")
        process_telemetry = True
        try:
            process_functions = [
                getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses_v3", None),
                getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None),
            ]
            process_function = next(item for item in process_functions if item is not None)
            process_function(handle)
        except Exception:
            process_telemetry = False
        pynvml.nvmlShutdown()
        return GpuCapacity(
            available=True,
            name=str(name),
            total_vram_gb=float(memory.total) / GIB,
            free_vram_gb=float(memory.free) / GIB,
            driver_version=str(driver),
            cuda_visible=True,
            process_telemetry_available=process_telemetry,
            telemetry_backend="nvml",
        )
    except Exception as exc:
        cuda_visible = False
        try:
            command = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            cuda_visible = command.returncode == 0 and bool(command.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
        return GpuCapacity(
            available=cuda_visible,
            name=None,
            total_vram_gb=None,
            free_vram_gb=None,
            driver_version=None,
            cuda_visible=cuda_visible,
            process_telemetry_available=False,
            telemetry_backend="unavailable",
            error=f"{type(exc).__name__}: {exc}",
        )


def detect_hardware(results_root: str | Path, temp_root: str | Path | None = None) -> HardwareCapacity:
    """Inspect capacity without allocating research-sized objects."""

    import psutil

    results = Path(results_root).resolve()
    temp = Path(temp_root or tempfile.gettempdir()).resolve()
    results_usage = shutil.disk_usage(results)
    temp_usage = shutil.disk_usage(temp)
    memory = psutil.virtual_memory()
    logical = int(psutil.cpu_count(logical=True) or 1)
    physical = int(psutil.cpu_count(logical=False) or logical)
    return HardwareCapacity(
        logical_cpu_count=logical,
        physical_cpu_count=physical,
        total_ram_gb=float(memory.total) / GIB,
        available_ram_gb=float(memory.available) / GIB,
        results_free_disk_gb=float(results_usage.free) / GIB,
        temp_free_disk_gb=float(temp_usage.free) / GIB,
        results_volume=str(results.anchor),
        temp_volume=str(temp.anchor),
        gpu=_detect_gpu(),
    )


def resolve_execution_policy(
    configured: ExecutionPolicy,
    capacity: HardwareCapacity,
) -> ResolvedExecutionPolicy:
    """Resolve limits downward for smaller machines while retaining headroom."""

    warnings: list[str] = []
    parallel = configured.parallelism
    resolved_threads = min(parallel.estimator_threads, capacity.logical_cpu_count)
    if resolved_threads < parallel.estimator_threads:
        warnings.append(
            f"estimator_threads scaled down from {parallel.estimator_threads} to {resolved_threads}"
        )
    parallel = ParallelismPolicy(
        concurrent_experiment_runs=parallel.concurrent_experiment_runs,
        concurrent_folds=parallel.concurrent_folds,
        data_loader_workers=parallel.data_loader_workers,
        estimator_threads=max(1, resolved_threads),
        allow_nested_parallelism=parallel.allow_nested_parallelism,
    )
    _validate_parallelism(parallel, capacity.logical_cpu_count)

    total = capacity.total_ram_gb
    if total <= 1.0:
        raise ExecutionPolicyError(f"detected system RAM is implausibly low: {total:.3f} GiB")
    minimum_reserve = max(total * 0.25, 6.0 if total >= 8.0 else total * 0.25)
    reserve = min(total * 0.75, max(configured.memory.reserve_system_ram_gb, minimum_reserve))
    maximum_process = total - reserve
    abort = min(configured.memory.abort_process_tree_rss_gb, maximum_process)
    warn = min(configured.memory.warn_process_tree_rss_gb, abort * 0.95)
    available_abort = min(
        configured.memory.abort_if_system_available_below_gb,
        max(0.25, reserve * 0.8),
    )
    if abort <= 0.25 or warn <= 0 or warn >= abort:
        raise ExecutionPolicyError(
            "detected RAM cannot support safe warning/abort thresholds with required headroom"
        )
    if abort < configured.memory.abort_process_tree_rss_gb:
        warnings.append(
            "RAM process abort threshold scaled down from "
            f"{configured.memory.abort_process_tree_rss_gb:.3f} to {abort:.3f} GiB"
        )
    if reserve != configured.memory.reserve_system_ram_gb:
        warnings.append(
            f"system RAM reserve resolved from {configured.memory.reserve_system_ram_gb:.3f} "
            f"to {reserve:.3f} GiB"
        )
    memory = MemoryPolicy(
        reserve_system_ram_gb=reserve,
        warn_process_tree_rss_gb=warn,
        abort_process_tree_rss_gb=abort,
        abort_if_system_available_below_gb=available_abort,
        estimated_peak_safety_factor=configured.memory.estimated_peak_safety_factor,
    )

    resolved_gpu = configured.gpu
    if capacity.gpu.total_vram_gb is not None:
        if configured.gpu.reserve_vram_gb >= capacity.gpu.total_vram_gb:
            raise ExecutionPolicyError("configured GPU reserve consumes detected total VRAM")
        maximum_gpu_process = capacity.gpu.total_vram_gb - configured.gpu.reserve_vram_gb
        gpu_abort = min(configured.gpu.abort_process_vram_gb, maximum_gpu_process)
        gpu_warn = min(configured.gpu.warn_process_vram_gb, gpu_abort * 0.95)
        if gpu_warn <= 0 or gpu_warn >= gpu_abort:
            raise ExecutionPolicyError("detected VRAM cannot support safe warning/abort thresholds")
        if gpu_abort < configured.gpu.abort_process_vram_gb:
            warnings.append(
                "GPU process abort threshold scaled down from "
                f"{configured.gpu.abort_process_vram_gb:.3f} to {gpu_abort:.3f} GiB"
            )
        resolved_gpu = GpuPolicy(
            reserve_vram_gb=configured.gpu.reserve_vram_gb,
            warn_process_vram_gb=gpu_warn,
            abort_process_vram_gb=gpu_abort,
            require_gpu_telemetry_for_gpu_runs=(
                configured.gpu.require_gpu_telemetry_for_gpu_runs
            ),
        )

    return ResolvedExecutionPolicy(
        schema_version=configured.schema_version,
        profile_name=configured.profile_name,
        parallelism=parallel,
        memory=memory,
        gpu=resolved_gpu,
        disk=configured.disk,
        monitoring=configured.monitoring,
        configured_policy_path=configured.source_path,
        resolution_warnings=tuple(warnings),
    )


def apply_thread_environment(estimator_threads: int) -> dict[str, str]:
    """Set inherited native thread-pool limits once before worker imports."""

    if int(estimator_threads) <= 0:
        raise ExecutionPolicyError("estimator_threads must be positive")
    value = str(int(estimator_threads))
    settings = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "LOKY_MAX_CPU_COUNT": value,
    }
    os.environ.update(settings)
    return settings


def apply_estimator_parallelism(
    model_name: str,
    model_kwargs: Mapping[str, Any] | None,
    selector_kwargs: Mapping[str, Any] | None,
    *,
    estimator_threads: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply one resolved thread cap and reject wider method-specific overrides."""

    limit = int(estimator_threads)
    if limit <= 0:
        raise ExecutionPolicyError("estimator thread limit must be positive")

    def constrain(value: Any) -> Any:
        if isinstance(value, Mapping):
            output = {key: constrain(item) for key, item in value.items()}
            for key in ("n_jobs", "thread_count"):
                if key not in output:
                    continue
                configured_value = int(output[key])
                if configured_value <= 0 or configured_value > limit:
                    raise ExecutionPolicyError(
                        f"method-specific {key}={configured_value} exceeds execution limit {limit}"
                    )
            return output
        if isinstance(value, list):
            return [constrain(item) for item in value]
        return value

    model = constrain(dict(model_kwargs or {}))
    selector = constrain(dict(selector_kwargs or {}))
    name = str(model_name).lower()
    if name == "catboost":
        model["thread_count"] = limit
    elif name in {"rf", "random_forest"}:
        model["n_jobs"] = limit
    return model, selector


def estimate_run_size(
    *,
    row_count: int,
    column_dtype_bytes: list[int],
    method_memory_multiplier: float | None,
    prediction_row_count: int = 0,
    prediction_bytes_per_row: int = 64,
    policy: ResolvedExecutionPolicy,
) -> dict[str, Any]:
    """Build a conservative declared-shape bound without reading a matrix."""

    if row_count <= 0 or not column_dtype_bytes or any(value <= 0 for value in column_dtype_bytes):
        raise ExecutionPolicyError("run-size estimate requires positive rows and dtype widths")
    projected = int(row_count * sum(column_dtype_bytes))
    dense_working = int(row_count * len(column_dtype_bytes) * max(column_dtype_bytes) * 2)
    prediction = int(max(0, prediction_row_count) * prediction_bytes_per_row)
    atomic_temp = int(prediction * policy.disk.estimated_write_safety_factor)
    if method_memory_multiplier is None or method_memory_multiplier < 1.0:
        return {
            "status": "estimate_unavailable",
            "reason": "explicit conservative method_memory_multiplier>=1 is required",
            "projected_input_bytes": projected,
            "dense_working_copy_bytes": dense_working,
            "prediction_output_bytes": prediction,
            "atomic_write_temporary_bytes": atomic_temp,
        }
    peak = int(
        (projected + dense_working * float(method_memory_multiplier))
        * policy.memory.estimated_peak_safety_factor
    )
    limit = int(policy.memory.abort_process_tree_rss_gb * GIB)
    return {
        "status": "available",
        "projected_input_bytes": projected,
        "dense_working_copy_bytes": dense_working,
        "method_memory_multiplier": float(method_memory_multiplier),
        "prediction_output_bytes": prediction,
        "atomic_write_temporary_bytes": atomic_temp,
        "estimated_peak_bytes": peak,
        "safety_factor": policy.memory.estimated_peak_safety_factor,
        "process_limit_bytes": limit,
        "remaining_headroom_bytes": limit - peak,
        "fits": peak < limit,
    }


def _check(name: str, passed: bool, detail: str, *, blocking: bool = True) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "blocking": blocking, "detail": detail}


def run_preflight(
    *,
    repository_root: str | Path,
    config_path: str | Path = DEFAULT_POLICY_PATH,
    results_root: str | Path = "results",
    temp_root: str | Path | None = None,
    requested_accelerator: str = "cpu",
    allow_gpu_without_telemetry: bool = False,
    estimate: dict[str, Any] | None = None,
    requested_run_directory: str | Path | None = None,
    capacity: HardwareCapacity | None = None,
) -> dict[str, Any]:
    """Run mechanical preflight checks and return a machine-readable report."""

    root = Path(repository_root).resolve()
    active = resolve_results_root(root, results_root)
    temp = Path(temp_root or tempfile.gettempdir()).resolve()
    configured = load_execution_policy(root, config_path)
    detected = capacity or detect_hardware(active, temp)
    resolved = resolve_execution_policy(configured, detected)
    checks: list[dict[str, Any]] = []
    warnings = list(resolved.resolution_warnings)

    legacy = configured_legacy_results_root() or AUDITED_LEGACY_RESULTS_ROOT.resolve()
    separation_ok = True
    if legacy is not None:
        try:
            validate_results_root_separation(active, legacy)
        except Exception:
            separation_ok = False
    checks.append(_check("active_legacy_root_separation", separation_ok, f"active={active}, legacy={legacy}"))

    write_probe = active / f".preflight-{uuid.uuid4().hex}.partial"
    writable = False
    try:
        reject_historical_write(write_probe).write_text("preflight\n", encoding="utf-8")
        with write_probe.open("r+b") as handle:
            os.fsync(handle.fileno())
        writable = True
    except OSError as exc:
        warnings.append(f"active results write probe failed: {exc}")
    finally:
        if write_probe.exists():
            write_probe.unlink()
    checks.append(_check("active_results_writable", writable, str(active)))
    legacy_blocked = True
    if legacy is not None:
        try:
            reject_historical_write(legacy / "preflight-must-not-write.tmp", legacy_root=legacy)
            legacy_blocked = False
        except PermissionError:
            legacy_blocked = True
    checks.append(_check("legacy_results_write_blocked", legacy_blocked, str(legacy)))

    checks.append(
        _check(
            "system_available_ram",
            detected.available_ram_gb > resolved.memory.abort_if_system_available_below_gb,
            f"available={detected.available_ram_gb:.3f} GiB, minimum={resolved.memory.abort_if_system_available_below_gb:.3f} GiB",
        )
    )
    checks.append(
        _check(
            "results_disk_free",
            detected.results_free_disk_gb >= resolved.disk.minimum_free_results_gb,
            f"free={detected.results_free_disk_gb:.3f} GiB, minimum={resolved.disk.minimum_free_results_gb:.3f} GiB",
        )
    )
    checks.append(
        _check(
            "temp_disk_free",
            detected.temp_free_disk_gb >= resolved.disk.minimum_free_temp_gb,
            f"free={detected.temp_free_disk_gb:.3f} GiB, minimum={resolved.disk.minimum_free_temp_gb:.3f} GiB",
        )
    )

    accelerator = str(requested_accelerator).lower()
    if accelerator not in {"cpu", "gpu"}:
        raise ExecutionPolicyError("requested_accelerator must be cpu or gpu")
    if accelerator == "gpu":
        telemetry_ok = detected.gpu.process_telemetry_available
        override = bool(allow_gpu_without_telemetry)
        checks.append(_check("gpu_available", detected.gpu.available, str(detected.gpu.name)))
        checks.append(
            _check(
                "gpu_process_telemetry",
                telemetry_ok or override or not resolved.gpu.require_gpu_telemetry_for_gpu_runs,
                f"available={telemetry_ok}, explicit_override={override}",
            )
        )
        if override and not telemetry_ok:
            warnings.append("GPU telemetry requirement explicitly overridden")
    else:
        checks.append(
            _check(
                "gpu_process_telemetry",
                True,
                "not required for requested CPU run",
                blocking=False,
            )
        )

    if requested_run_directory is not None:
        run_dir = Path(requested_run_directory).resolve()
        lock = run_dir / ".execution.lock"
        partials = list(run_dir.rglob("*.partial")) if run_dir.exists() else []
        checks.append(_check("stale_run_lock_absent", not lock.exists(), str(lock)))
        if partials:
            warnings.append(f"requested run contains {len(partials)} partial artifact(s)")

    resolved_estimate = estimate
    if estimate is not None:
        status = estimate.get("status")
        checks.append(
            _check(
                "run_size_estimate",
                status == "available" and bool(estimate.get("fits")),
                json.dumps(estimate, sort_keys=True),
            )
        )
        atomic_required = float(estimate.get("atomic_write_temporary_bytes", 0)) / GIB
        checks.append(
            _check(
                "atomic_write_space",
                detected.results_free_disk_gb
                >= resolved.disk.minimum_free_results_gb + atomic_required,
                f"required_additional={atomic_required:.6f} GiB",
            )
        )

    blocking_reasons = [item["name"] for item in checks if item["blocking"] and not item["passed"]]
    git_commit = "unknown"
    git_dirty: bool | None = None
    try:
        git_commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        git_status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        git_commit = git_commit_result.stdout.strip() or "unknown"
        git_dirty = bool(git_status_result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        warnings.append("Git provenance unavailable during preflight")
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(root),
        "active_results_root": str(active),
        "temporary_root": str(temp),
        "requested_accelerator": accelerator,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "gpu_telemetry_override": bool(allow_gpu_without_telemetry),
        "configured_policy": configured.configured,
        "resolved_policy": resolved.to_dict(),
        "detected_capacity": asdict(detected),
        "run_size_estimate": resolved_estimate,
        "checks": checks,
        "warnings": warnings,
        "blocking_reasons": blocking_reasons,
        "status": "pass" if not blocking_reasons else "fail",
    }
