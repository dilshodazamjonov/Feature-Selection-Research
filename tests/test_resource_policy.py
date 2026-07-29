from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from credit_risk_fs.experiments.matrix import iter_matrix
from credit_risk_fs.experiments.resource_policy import (
    ExecutionPolicyError,
    GpuCapacity,
    HardwareCapacity,
    apply_estimator_parallelism,
    apply_thread_environment,
    estimate_run_size,
    load_execution_policy,
    resolve_execution_policy,
    run_preflight,
)
from credit_risk_fs.experiments.result_paths import initialize_results_layout


POLICY_SOURCE = Path("configs/execution/local_laptop_safe_v1.yaml").resolve()


def _repository(tmp_path: Path, text: str | None = None) -> tuple[Path, Path]:
    config = tmp_path / "configs/execution/local_laptop_safe_v1.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(text or POLICY_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    results = initialize_results_layout(tmp_path)
    return config, results


def _capacity(
    *,
    total_ram: float = 40.0,
    available_ram: float = 20.0,
    result_disk: float = 100.0,
    temp_disk: float = 100.0,
    gpu_available: bool = True,
    gpu_telemetry: bool = True,
) -> HardwareCapacity:
    return HardwareCapacity(
        logical_cpu_count=16,
        physical_cpu_count=10,
        total_ram_gb=total_ram,
        available_ram_gb=available_ram,
        results_free_disk_gb=result_disk,
        temp_free_disk_gb=temp_disk,
        results_volume="D:\\",
        temp_volume="C:\\",
        gpu=GpuCapacity(
            available=gpu_available,
            name="test gpu" if gpu_available else None,
            total_vram_gb=8.0 if gpu_available else None,
            free_vram_gb=7.0 if gpu_available else None,
            driver_version="test",
            cuda_visible=gpu_available,
            process_telemetry_available=gpu_telemetry,
            telemetry_backend="mock",
        ),
    )


def test_valid_policy_loading_and_sequential_defaults(tmp_path):
    config, _ = _repository(tmp_path)
    policy = load_execution_policy(tmp_path, config)
    assert policy.profile_name == "local_laptop_safe_v1"
    assert policy.parallelism.concurrent_experiment_runs == 1
    assert policy.parallelism.concurrent_folds == 1
    assert policy.parallelism.data_loader_workers == 0
    assert policy.parallelism.estimator_threads == 4
    assert policy.parallelism.allow_nested_parallelism is False


@pytest.mark.parametrize(
    "old,new",
    [
        ("warn_process_tree_rss_gb: 24", "warn_process_tree_rss_gb: -1"),
        ("sample_interval_seconds: 1.0", "sample_interval_seconds: 0"),
        ("data_loader_workers: 0", "data_loader_workers: -1"),
    ],
)
def test_invalid_or_negative_policy_values_fail(tmp_path, old, new):
    text = POLICY_SOURCE.read_text(encoding="utf-8").replace(old, new)
    config, _ = _repository(tmp_path, text)
    with pytest.raises(ExecutionPolicyError):
        load_execution_policy(tmp_path, config)


def test_contradictory_thresholds_fail(tmp_path):
    text = POLICY_SOURCE.read_text(encoding="utf-8").replace(
        "warn_process_tree_rss_gb: 24", "warn_process_tree_rss_gb: 29"
    )
    config, _ = _repository(tmp_path, text)
    with pytest.raises(ExecutionPolicyError, match="warning threshold"):
        load_execution_policy(tmp_path, config)


def test_nested_parallelism_is_rejected(tmp_path):
    text = POLICY_SOURCE.read_text(encoding="utf-8").replace(
        "concurrent_folds: 1", "concurrent_folds: 2"
    )
    config, _ = _repository(tmp_path, text)
    with pytest.raises(ExecutionPolicyError, match="nested parallelism"):
        load_execution_policy(tmp_path, config)


def test_legacy_ram_capacity_estimates_remain_parseable_for_artifact_identity(tmp_path):
    config, _ = _repository(tmp_path)
    resolved = resolve_execution_policy(load_execution_policy(tmp_path, config), _capacity(total_ram=24))
    assert resolved.memory.abort_process_tree_rss_gb < 28
    assert resolved.memory.warn_process_tree_rss_gb <= 24
    assert resolved.memory.reserve_system_ram_gb >= 6
    assert resolved.memory.reserve_system_ram_gb >= 24 * 0.25


def test_capacity_exceeding_gpu_reserve_is_rejected(tmp_path):
    config, _ = _repository(tmp_path)
    capacity = _capacity(gpu_available=True)
    capacity = replace(
        capacity,
        gpu=GpuCapacity(True, "tiny", 1.0, 1.0, "x", True, True, "mock"),
    )
    with pytest.raises(ExecutionPolicyError, match="reserve"):
        resolve_execution_policy(load_execution_policy(tmp_path, config), capacity)


def test_preflight_ram_and_disk_pass_and_fail(tmp_path, monkeypatch):
    monkeypatch.delenv("CREDIT_RISK_LEGACY_RESULTS_ROOT", raising=False)
    config, results = _repository(tmp_path)
    passed = run_preflight(
        repository_root=tmp_path,
        config_path=config,
        results_root=results,
        temp_root=tmp_path,
        capacity=_capacity(),
    )
    assert passed["status"] == "pass"
    failed = run_preflight(
        repository_root=tmp_path,
        config_path=config,
        results_root=results,
        temp_root=tmp_path,
        capacity=_capacity(available_ram=0.5, result_disk=1, temp_disk=1),
    )
    assert failed["status"] == "fail"
    assert {"results_disk_free", "temp_disk_free"}.issubset(
        failed["blocking_reasons"]
    )
    assert "system_available_ram" not in failed["blocking_reasons"]


def test_gpu_run_requires_process_telemetry_unless_recorded_override(tmp_path, monkeypatch):
    monkeypatch.delenv("CREDIT_RISK_LEGACY_RESULTS_ROOT", raising=False)
    config, results = _repository(tmp_path)
    blocked = run_preflight(
        repository_root=tmp_path,
        config_path=config,
        results_root=results,
        temp_root=tmp_path,
        requested_accelerator="gpu",
        capacity=_capacity(gpu_telemetry=False),
    )
    assert blocked["status"] == "fail"
    assert "gpu_process_telemetry" in blocked["blocking_reasons"]
    overridden = run_preflight(
        repository_root=tmp_path,
        config_path=config,
        results_root=results,
        temp_root=tmp_path,
        requested_accelerator="gpu",
        allow_gpu_without_telemetry=True,
        capacity=_capacity(gpu_telemetry=False),
    )
    assert overridden["status"] == "pass"
    assert overridden["gpu_telemetry_override"] is True


def test_cpu_preflight_does_not_require_gpu(tmp_path, monkeypatch):
    monkeypatch.delenv("CREDIT_RISK_LEGACY_RESULTS_ROOT", raising=False)
    config, results = _repository(tmp_path)
    report = run_preflight(
        repository_root=tmp_path,
        config_path=config,
        results_root=results,
        temp_root=tmp_path,
        requested_accelerator="cpu",
        capacity=_capacity(gpu_available=False, gpu_telemetry=False),
    )
    assert report["status"] == "pass"


def test_run_size_estimate_requires_explicit_method_multiplier(tmp_path):
    config, _ = _repository(tmp_path)
    resolved = resolve_execution_policy(load_execution_policy(tmp_path, config), _capacity())
    unavailable = estimate_run_size(
        row_count=100,
        column_dtype_bytes=[8, 4],
        method_memory_multiplier=None,
        policy=resolved,
    )
    assert unavailable["status"] == "estimate_unavailable"
    oversized = estimate_run_size(
        row_count=10**10,
        column_dtype_bytes=[8] * 20,
        method_memory_multiplier=10,
        policy=resolved,
    )
    assert oversized["status"] == "available"
    assert oversized["fits"] is None
    assert oversized["termination_limit_bytes"] is None


def test_explicit_root_resolution_is_independent_of_cwd(tmp_path, monkeypatch):
    config, _ = _repository(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    policy = load_execution_policy(tmp_path, config.relative_to(tmp_path))
    assert Path(policy.source_path) == config.resolve()


def test_active_and_legacy_roots_are_separated_in_preflight(tmp_path, monkeypatch):
    config, results = _repository(tmp_path)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    monkeypatch.setenv("CREDIT_RISK_LEGACY_RESULTS_ROOT", str(legacy))
    report = run_preflight(
        repository_root=tmp_path,
        config_path=config,
        results_root=results,
        temp_root=tmp_path,
        capacity=_capacity(),
    )
    assert report["status"] == "pass"
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["active_legacy_root_separation"]["passed"]
    assert checks["legacy_results_write_blocked"]["passed"]


def test_estimator_thread_adapter_propagates_and_rejects_override():
    model, selector = apply_estimator_parallelism(
        "catboost", {}, {"nested": {"n_jobs": 1}}, estimator_threads=4
    )
    assert model["thread_count"] == 4
    assert selector["nested"]["n_jobs"] == 1
    with pytest.raises(ExecutionPolicyError, match="exceeds"):
        apply_estimator_parallelism("rf", {"n_jobs": -1}, {}, estimator_threads=4)


def test_native_thread_environment_prevents_nested_parallelism(monkeypatch):
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "LOKY_MAX_CPU_COUNT",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = apply_thread_environment(4)
    assert settings["OMP_NUM_THREADS"] == "1"
    assert settings["MKL_NUM_THREADS"] == "1"
    assert settings["OPENBLAS_NUM_THREADS"] == "1"
    assert settings["LOKY_MAX_CPU_COUNT"] == "4"


def test_matrix_order_is_deterministic():
    first = [item.run_label for item in iter_matrix()]
    second = [item.run_label for item in iter_matrix()]
    assert first == second
    assert first[0].startswith("lr_")
    assert first[-1].startswith("catboost_")
