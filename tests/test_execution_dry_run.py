from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from cleanup.tools.validate_repository_state import validate_active_results
from credit_risk_fs.experiments.atomic_io import atomic_publish, inspect_artifact, write_json_atomic
from credit_risk_fs.experiments.checkpointing import CheckpointManager, ResumeValidationError
from credit_risk_fs.experiments.config import compute_config_hash
from credit_risk_fs.experiments.execution import RegisteredRunRequest, execute_registered_run
from credit_risk_fs.experiments.resource_monitor import (
    MANUAL_INTERRUPT,
    RAM_PROCESS_LIMIT,
    SupervisorResult,
    WORKER_CRASH,
)
from credit_risk_fs.experiments.resource_policy import (
    DiskPolicy,
    GpuPolicy,
    MemoryPolicy,
    MonitoringPolicy,
    ParallelismPolicy,
    ResolvedExecutionPolicy,
)
from credit_risk_fs.experiments.result_paths import create_run_directory, initialize_results_layout
from credit_risk_fs.pipelines.common import ExperimentConfig


def _policy(*, warn_ram: float = 1.0, abort_ram: float = 2.0) -> ResolvedExecutionPolicy:
    return ResolvedExecutionPolicy(
        schema_version="resource_safe_execution_policy_v1",
        profile_name="synthetic_safe_v1",
        parallelism=ParallelismPolicy(1, 1, 0, 1, False),
        memory=MemoryPolicy(1, warn_ram, abort_ram, 0.001, 1.35),
        gpu=GpuPolicy(0.1, 1, 2, True),
        disk=DiskPolicy(0.001, 0.001, 2.5),
        monitoring=MonitoringPolicy(0.02, 0.4, 2.0),
        configured_policy_path="synthetic",
    )


def _fixture(tmp_path: Path, *, run_id: str, policy: ResolvedExecutionPolicy):
    data_dir = tmp_path / "data/synthetic"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "record_id": [f"s-{index:02d}" for index in range(14)],
            "TARGET": [index % 2 for index in range(14)],
            "recent_decision": list(range(-13, 1)),
            "f1": [float(index % 3) for index in range(14)],
            "unused": [f"u{index}" for index in range(14)],
        }
    )
    frame.to_csv(data_dir / "application_train.csv", index=False)
    results = initialize_results_layout(tmp_path)
    run_dir = create_run_directory(results, dataset="synthetic", run_id=run_id)
    effective = {
        "dataset_name": "synthetic",
        "data_dir": str(data_dir),
        "random_seed": 42,
        "feature_budgets": {"lr": 1},
        "input_column_projection": {
            "application_train": [
                "record_id",
                "TARGET",
                "recent_decision",
                "f1",
            ]
        },
        "_resolved_execution_policy": policy.to_dict(),
    }
    experiment = ExperimentConfig(
        experiment_name="synthetic_none",
        selector_name="none",
        dataset_name="synthetic",
        model_name="lr",
        model_kwargs={
            "solver": "liblinear",
            "max_iter": 100,
            "class_weight": None,
            "random_state": 42,
        },
        data_dir=str(data_dir),
        target="TARGET",
        time_col="recent_decision",
        drop_id_cols=("record_id",),
        base_output_dir=str(run_dir),
        experiment_output_dir=str(run_dir),
        dev_start_day=-13,
        oot_start_day=-3,
        oot_end_day=0,
        n_splits=2,
        cv_gap_groups=0,
        random_state=42,
        feature_budget=1,
        excluded_feature_columns=("TARGET", "recent_decision"),
        config_hash=compute_config_hash(effective),
        stable_row_id_column="record_id",
        input_column_projection={
            "application_train": ("record_id", "TARGET", "recent_decision", "f1")
        },
        require_full_candidate_projection=False,
        estimator_threads=1,
    )
    preflight = {
        "schema_version": "resource_safe_preflight_v1",
        "status": "pass",
        "blocking_reasons": [],
        "temporary_root": str(tmp_path),
        "git_commit": "synthetic-commit",
        "git_dirty": False,
    }
    request = RegisteredRunRequest(
        repository_root=tmp_path.resolve(),
        results_root=results,
        run_directory=run_dir,
        dataset="synthetic",
        selector="none",
        model="lr",
        experiment_type="synthetic_dry_run",
        split_protocol="synthetic_time",
        seed=42,
        effective_config=effective,
        experiment_config=experiment,
        preflight_report=preflight,
        resolved_policy=policy,
    )
    return request


def test_successful_synthetic_run_uses_real_registered_execution_contract(tmp_path, monkeypatch):
    monkeypatch.delenv("CREDIT_RISK_LEGACY_RESULTS_ROOT", raising=False)
    request = _fixture(tmp_path, run_id="synthetic-success", policy=_policy())
    outcome = execute_registered_run(request)
    assert outcome.status == "completed"
    assert outcome.supervisor.child_cleanup_confirmed
    assert (request.run_directory / "_SUCCESS").is_file()
    assert (request.run_directory / "predictions_dev.csv").is_file()
    assert (request.run_directory / "predictions_oot.csv").is_file()
    checkpoint = CheckpointManager(request.run_directory).load()
    assert checkpoint["status"] == "completed"
    assert "evaluation_completed" in checkpoint["completed_stages"]
    load_report = json.loads(
        (request.run_directory / "data_split_manifest.json").read_text(encoding="utf-8")
    )["column_projection"]["application_train"]
    assert load_report["requested_columns"] == [
        "record_id",
        "TARGET",
        "recent_decision",
        "f1",
    ]
    assert "unused" not in load_report["loaded_columns"]
    assert validate_active_results(tmp_path)["registered_runs"] == 1


def test_forced_low_memory_registered_run_aborts_and_remains_resumable(tmp_path, monkeypatch):
    monkeypatch.delenv("CREDIT_RISK_LEGACY_RESULTS_ROOT", raising=False)
    request = _fixture(
        tmp_path,
        run_id="synthetic-abort",
        policy=_policy(warn_ram=0.07, abort_ram=0.11),
    )
    request = replace(
        request,
        worker_target="credit_risk_fs.experiments.synthetic_execution:bounded_memory_worker",
        worker_kwargs={
            "chunk_mb": 4,
            "maximum_allocation_mb": 160,
            "spawn_child": True,
        },
    )
    outcome = execute_registered_run(request)
    assert outcome.status == "aborted_resource_limit"
    assert outcome.stop_code == RAM_PROCESS_LIMIT
    assert outcome.supervisor.child_cleanup_confirmed
    assert not (request.run_directory / "_SUCCESS").exists()
    assert outcome.manifest["status"] != "completed"
    checkpoint = CheckpointManager(request.run_directory)
    payload = checkpoint.load()
    assert payload["last_successful_stage"] == "initialized"
    assert payload["status"] == "aborted_resource_limit"
    assert checkpoint.validate_resume(payload["identity"]).resumable
    assert validate_active_results(tmp_path)["registered_runs"] == 1


def test_unexpected_worker_crash_is_finalized_in_manifest_and_index(tmp_path, monkeypatch):
    monkeypatch.delenv("CREDIT_RISK_LEGACY_RESULTS_ROOT", raising=False)
    request = _fixture(tmp_path, run_id="synthetic-crash", policy=_policy())
    request = replace(
        request,
        worker_target="credit_risk_fs.experiments.synthetic_execution:unexpected_exit_worker",
        worker_kwargs={"exit_code": 7},
    )
    outcome = execute_registered_run(request)
    assert outcome.status == "failed"
    assert outcome.stop_code == WORKER_CRASH
    manifest = json.loads((request.run_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["worker_exit_code"] == 7
    index = pd.read_csv(request.results_root / "run_index.csv")
    assert index.iloc[0]["status"] == "failed"
    assert validate_active_results(tmp_path)["registered_runs"] == 1


def test_manual_interrupt_status_is_finalized_in_manifest_and_index(tmp_path, monkeypatch):
    monkeypatch.delenv("CREDIT_RISK_LEGACY_RESULTS_ROOT", raising=False)
    request = _fixture(tmp_path, run_id="synthetic-interrupt", policy=_policy())
    interrupted = SupervisorResult(
        status="interrupted",
        stop_code=MANUAL_INTERRUPT,
        worker_exit_code=1,
        return_value=None,
        worker_error="manual interrupt",
        samples=(),
        warnings=(),
        peak_process_tree_rss_bytes=0,
        peak_process_gpu_bytes=None,
        minimum_system_available_ram_bytes=None,
        minimum_results_free_disk_bytes=None,
        minimum_temp_free_disk_bytes=None,
        child_cleanup_confirmed=True,
        final_stage="initialized",
        final_fold_id=None,
    )
    monkeypatch.setattr(
        "credit_risk_fs.experiments.execution.supervise_worker",
        lambda **_kwargs: interrupted,
    )
    outcome = execute_registered_run(request)
    assert outcome.status == "interrupted"
    manifest = json.loads((request.run_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stop_code"] == MANUAL_INTERRUPT
    index = pd.read_csv(request.results_root / "run_index.csv")
    assert index.iloc[0]["status"] == "interrupted"
    assert not (request.run_directory / "_SUCCESS").exists()


def test_interrupted_publication_restart_and_mismatch_proof(tmp_path):
    run = tmp_path / "results/runs/synthetic/interrupted"
    run.mkdir(parents=True)
    identity = {
        "run_id": "interrupted",
        "dataset": "synthetic",
        "selector": "none",
        "model": "lr",
        "split_protocol": "synthetic_time",
        "seed": 42,
        "budgets": {"lr": 1},
        "resolved_config_hash": "config-a",
        "protocol_version": "1",
        "protocol_hash": "protocol-a",
        "data_hash": "data-a",
        "row_alignment_hash": "row-a",
        "git_commit": "commit-a",
        "git_dirty": False,
    }
    manager = CheckpointManager(run)
    manager.initialize(identity)
    final = run / "stage.json"
    with pytest.raises(RuntimeError):
        atomic_publish(
            final,
            lambda partial: partial.write_text('{"valid": true}\n', encoding="utf-8"),
            artifact_format="json",
            before_replace=lambda *_: (_ for _ in ()).throw(RuntimeError("interrupt")),
        )
    assert not final.exists()
    validation = manager.validate_resume(identity)
    assert len(validation.quarantined_partials) == 1
    metadata = write_json_atomic(final, {"valid": True})
    manager.transition("selection_completed", artifacts=[metadata])
    assert inspect_artifact(final).sha256 == metadata.sha256
    with pytest.raises(ResumeValidationError) as exc:
        manager.validate_resume({**identity, "resolved_config_hash": "config-b"})
    assert exc.value.code == "resume_mismatch_resolved_config_hash"
