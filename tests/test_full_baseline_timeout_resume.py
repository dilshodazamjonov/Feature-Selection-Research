from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from credit_risk_fs.experiments.atomic_io import (
    inspect_artifact,
    sha256_file,
    write_json_atomic,
)
from credit_risk_fs.experiments.checkpointing import (
    CheckpointManager,
    ResumeValidationError,
)
from credit_risk_fs.experiments.full_baseline import (
    _execute_real_cell,
    build_resume_plan_report,
    build_status_report,
    inspect_cell,
    load_full_baseline_plan,
    workload_classification,
)
from credit_risk_fs.experiments.full_baseline_timeout_resume import (
    AUTHORIZATION_SCHEMA_VERSION,
    NOT_RESUMABLE,
    RESUMABLE_FROM_CELL_BOUNDARY,
    VALIDATOR_VERSION,
    TimeoutResumeValidation,
    inspect_research_processes,
    validate_timeout_resume_authorization,
)
from credit_risk_fs.experiments.ram_control import load_ram_control_policy
from credit_risk_fs.experiments.resource_policy import (
    DiskPolicy,
    GpuPolicy,
    MemoryPolicy,
    MonitoringPolicy,
    ParallelismPolicy,
    ResolvedExecutionPolicy,
)


ROOT = Path(__file__).resolve().parents[1]


def _identity() -> dict:
    return {
        "run_id": "fbv1-004-lendingclub_v2-catboost-full-features-s42",
        "dataset": "lendingclub_v2",
        "selector": "full_features",
        "model": "catboost",
        "split_protocol": "grouped_time_series_cv_5_splits_gap_1_then_locked_oot",
        "seed": 42,
        "budgets": {"lr": 20, "catboost": 40},
        "resolved_config_hash": "config-004",
        "protocol_version": "1.1.0",
        "protocol_hash": "protocol-v1",
        "data_hash": "data-v1",
        "row_alignment_hash": "rows-v1",
        "git_commit": "historical-commit",
        "git_dirty": False,
    }


def _workload() -> dict:
    return {
        "selector_cost_class": "light",
        "final_model_cost_class": "heavy",
        "dataset_cost_class": "light",
        "effective_cost_class": "heavy",
        "effective_wall_clock_limit_seconds": 43200.0,
        "composition_rule": (
            "maximum_cost_and_timeout_across_selector_final_model_dataset"
        ),
        "policy_sha256": "runtime-policy-v1",
    }


def _authorization_fixture(tmp_path: Path) -> dict:
    root = tmp_path
    run_id = "fbv1-004-lendingclub_v2-catboost-full-features-s42"
    run = root / "results/full_baseline_v1/runs/lendingclub_v2" / run_id
    run.mkdir(parents=True)
    earlier = root / "evidence/cell-001-success"
    runtime = root / "runtime/validator.py"
    scientific = root / "science/model.py"
    frozen = root / "configs/full_baseline.yaml"
    earlier.parent.mkdir(parents=True)
    runtime.parent.mkdir(parents=True)
    scientific.parent.mkdir(parents=True)
    frozen.parent.mkdir(parents=True)
    earlier.write_text("authenticated\n", encoding="utf-8")
    runtime.write_text("validator = 1\n", encoding="utf-8")
    scientific.write_text("frozen_model = True\n", encoding="utf-8")
    frozen.write_text("frozen = true\n", encoding="utf-8")

    identity = _identity()
    resource = {
        "status": "timed_out",
        "stop_code": "wall_clock_limit",
        "worker_exit_code": 15,
        "active_computation_seconds": 10800.025,
        "total_ram_wait_seconds": 0.0,
    }
    manifest = {
        "run_id": run_id,
        "status": "timed_out",
        "stop_code": "wall_clock_limit",
        "worker_exit_code": 15,
    }
    finalized = {
        "config.json": {},
        "preflight.json": {},
        "resource_usage.json": {},
    }
    checkpoint = {
        "schema_version": "experiment_stage_checkpoint_v1",
        "run_id": run_id,
        "run_directory": str(run.resolve()),
        "identity": identity,
        "status": "failed",
        "completed_stages": ["initialized", "data_validated"],
        "completed_fold_ids": [],
        "finalized_artifacts": finalized,
        "last_successful_stage": "data_validated",
        "stop_code": "wall_clock_limit",
        "primary_stop_code": "wall_clock_limit",
        "secondary_events": [],
        "stop_lifecycle": [
            {"state": "RUNNING"},
            {"state": "WALL_CLOCK_STOP_LATCHED"},
            {"state": "COOPERATIVE_STOP_REQUESTED"},
            {"state": "GRACE_PERIOD"},
            {"state": "TERMINATE_PROCESS_TREE"},
            {"state": "EXIT_CONFIRMED"},
            {"state": "ARTIFACT_AND_STATE_FINALIZATION"},
        ],
        "termination_condition": None,
        "cleanup_evidence": {
            "child_cleanup_confirmed": True,
            "queue_cleanup_confirmed": True,
            "survivor_processes": [],
        },
        "attempt_history": [],
    }
    write_json_atomic(run / "config.json", {"identity": "config-004"})
    write_json_atomic(run / "preflight.json", {"status": "pass"})
    write_json_atomic(run / "resource_usage.json", resource)
    write_json_atomic(run / "manifest.json", manifest)
    for name, stage in (
        ("config.json", "initialized"),
        ("preflight.json", "initialized"),
        ("resource_usage.json", "failed"),
    ):
        metadata = inspect_artifact(run / name).to_dict()
        metadata["stage"] = stage
        metadata["provenance"] = {
            field: identity[field]
            for field in (
                "resolved_config_hash",
                "protocol_hash",
                "data_hash",
                "row_alignment_hash",
                "git_commit",
            )
        }
        checkpoint["finalized_artifacts"][name] = metadata
    write_json_atomic(run / "checkpoint.json", checkpoint)
    partial = run / "selected_feature_sets/fold_5_selected_features.csv"
    partial.parent.mkdir(parents=True)
    partial.write_text("feature\na\n", encoding="utf-8")

    historical_paths = [
        run / "config.json",
        run / "preflight.json",
        run / "resource_usage.json",
        run / "manifest.json",
        run / "checkpoint.json",
    ]
    auth = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "validation_timestamp_utc": "2026-07-30T04:00:00+00:00",
        "decision": RESUMABLE_FROM_CELL_BOUNDARY,
        "run_id": run_id,
        "cell_id": run_id,
        "cell_identity": {
            "dataset": "lendingclub_v2",
            "model": "catboost",
            "selector": "full_features",
            "seed": 42,
        },
        "historical_terminal_state": "timed_out",
        "historical_stop_reason": "wall_clock_limit",
        "historical_wall_clock_limit_seconds": 10800,
        "intended_restart_boundary": "cell_boundary",
        "checkpoint_identity": identity,
        "full_baseline_configuration_sha256": sha256_file(frozen),
        "authorized_workload": _workload(),
        "earlier_completed_run_ids": ["cell-001"],
        "historical_artifact_hashes": {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in historical_paths
        },
        "authorized_runtime_file_hashes": {
            runtime.relative_to(root).as_posix(): sha256_file(runtime)
        },
        "authorized_scientific_file_hashes": {
            scientific.relative_to(root).as_posix(): sha256_file(scientific)
        },
        "earlier_completed_artifact_hashes": {
            earlier.relative_to(root).as_posix(): sha256_file(earlier)
        },
        "partial_artifact_hashes": {
            partial.relative_to(run).as_posix(): sha256_file(partial)
        },
        "validation_checks": [
            {"name": "controlled_timeout", "result": "pass"},
            {"name": "cell_boundary", "result": "pass"},
        ],
    }
    auth_path = root / "configs/execution/timeout-auth.json"
    auth_path.parent.mkdir(parents=True)
    write_json_atomic(auth_path, auth)
    return {
        "root": root,
        "run": run,
        "auth": auth,
        "auth_path": auth_path,
        "cell": {
            "cell_id": run_id,
            "dataset": "lendingclub_v2",
            "model": "catboost",
            "method_id": "full_features",
            "seed": 42,
        },
        "workload": _workload(),
        "frozen_hash": sha256_file(frozen),
        "partial": partial,
        "runtime": runtime,
        "earlier": earlier,
        "identity": identity,
    }


def _validate(fixture: dict, **overrides):
    values = {
        "repository_root": fixture["root"],
        "run_directory": fixture["run"],
        "cell": fixture["cell"],
        "full_baseline_configuration_sha256": fixture["frozen_hash"],
        "workload_classification": fixture["workload"],
        "earlier_cells_authenticated": {"cell-001": True},
        "authorization_path": fixture["auth_path"],
        "process_records": [],
        "repository_state": {
            "commit": "repair-commit",
            "branch": "main",
            "dirty": False,
        },
    }
    values.update(overrides)
    return validate_timeout_resume_authorization(**values)


def test_controlled_timeout_is_pending_until_explicit_authorization(tmp_path):
    fixture = _authorization_fixture(tmp_path)
    missing = _validate(
        fixture, authorization_path=tmp_path / "configs/execution/missing.json"
    )
    assert missing.decision == NOT_RESUMABLE
    assert any("authorization" in reason for reason in missing.reasons)

    validated = _validate(fixture)
    assert validated.decision == RESUMABLE_FROM_CELL_BOUNDARY
    assert validated.intended_restart_boundary == "cell_boundary"
    assert validated.partial_artifacts == (
        "selected_feature_sets/fold_5_selected_features.csv",
    )


def test_authenticated_windows_sleep_accounting_incident_can_be_recovered(tmp_path):
    fixture = _authorization_fixture(tmp_path)
    resource_path = fixture["run"] / "resource_usage.json"
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    resource["active_computation_seconds"] = 15000.0
    resource["samples"] = [
        {"elapsed_seconds": 5000.0, "process_tree_cpu_seconds": 100.0},
        {"elapsed_seconds": 15000.0, "process_tree_cpu_seconds": 101.0},
    ]
    write_json_atomic(resource_path, resource)

    auth = fixture["auth"]
    auth["historical_reported_active_computation_seconds"] = 15000.0
    auth["historical_suspension_evidence"] = {
        "accounting_defect": "windows_sleep_counted_as_active_v1",
        "largest_sample_gap_seconds": 10000.0,
        "cpu_growth_during_gap_seconds": 1.0,
        "before_elapsed_seconds": 5000.0,
        "after_elapsed_seconds": 15000.0,
        "corrected_active_computation_seconds": 5000.0,
    }
    resource_relative = resource_path.relative_to(fixture["root"]).as_posix()
    auth["historical_artifact_hashes"][resource_relative] = sha256_file(resource_path)
    write_json_atomic(fixture["auth_path"], auth)

    assert _validate(fixture).decision == RESUMABLE_FROM_CELL_BOUNDARY
    auth["historical_suspension_evidence"]["cpu_growth_during_gap_seconds"] = 2.0
    write_json_atomic(fixture["auth_path"], auth)
    rejected = _validate(fixture)
    assert rejected.decision == NOT_RESUMABLE
    assert any("historical_suspension_evidence" in item for item in rejected.reasons)


@pytest.mark.parametrize(
    "process",
    [
        {
            "pid": 101,
            "ppid": 1,
            "name": "python.exe",
            "cmdline": "python scripts/run_full_baseline.py",
            "cwd": "ignored",
        },
        {
            "pid": 102,
            "ppid": 999,
            "name": "python.exe",
            "cmdline": "python -c from multiprocessing.spawn import spawn_main",
            "cwd": "ROOT",
        },
    ],
)
def test_active_parent_or_orphan_worker_blocks_resume(tmp_path, process):
    fixture = _authorization_fixture(tmp_path)
    if process["cwd"] == "ROOT":
        process = {**process, "cwd": str(tmp_path)}
    result = _validate(fixture, process_records=[process])
    assert result.decision == NOT_RESUMABLE
    assert any("no_active_or_orphan_worker" in reason for reason in result.reasons)


def test_current_windows_launcher_chain_is_not_mistaken_for_an_orphan(tmp_path):
    records = [
        {
            "pid": 501,
            "ppid": 500,
            "name": "python.exe",
            "cmdline": "python.exe scripts/run_full_baseline.py --plan-resume",
            "cwd": str(tmp_path),
        },
        {
            "pid": 500,
            "ppid": 499,
            "name": "powershell.exe",
            "cmdline": "powershell.exe python scripts/run_full_baseline.py --plan-resume",
            "cwd": str(tmp_path),
        },
        {
            "pid": 900,
            "ppid": 1,
            "name": "python.exe",
            "cmdline": "python.exe scripts/run_full_baseline.py",
            "cwd": str(tmp_path),
        },
    ]

    observed = inspect_research_processes(
        tmp_path,
        process_records=records,
        current_invocation_process_ids=(500, 501),
    )

    assert [item["pid"] for item in observed] == [900]


def test_active_or_stale_lock_blocks_without_being_deleted(tmp_path):
    fixture = _authorization_fixture(tmp_path)
    lock = fixture["run"] / ".execution.lock"
    lock.write_text("pid=999999\n", encoding="utf-8")
    result = _validate(fixture)
    assert result.decision == NOT_RESUMABLE
    assert lock.is_file()
    assert any("no_execution_lock" in reason for reason in result.reasons)


@pytest.mark.parametrize(
    "mutation",
    ["config", "code", "input", "fold", "seed", "model", "selector"],
)
def test_any_scientific_code_or_boundary_identity_mismatch_blocks(
    tmp_path, mutation
):
    fixture = _authorization_fixture(tmp_path)
    cell = dict(fixture["cell"])
    if mutation == "config":
        fixture["frozen_hash"] = "changed-config"
    elif mutation == "code":
        fixture["runtime"].write_text("validator = 2\n", encoding="utf-8")
    elif mutation == "input":
        checkpoint_path = fixture["run"] / "checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint["identity"]["data_hash"] = "different-input"
        write_json_atomic(checkpoint_path, checkpoint)
    elif mutation == "fold":
        checkpoint_path = fixture["run"] / "checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint["completed_fold_ids"] = [5]
        write_json_atomic(checkpoint_path, checkpoint)
    else:
        field = {"seed": "seed", "model": "model", "selector": "method_id"}[mutation]
        cell[field] = {"seed": 43, "model": "lr", "selector": "random_k"}[mutation]
    result = _validate(fixture, cell=cell)
    assert result.decision == NOT_RESUMABLE


def test_corrupt_or_unauthenticated_earlier_cell_blocks_resume(tmp_path):
    fixture = _authorization_fixture(tmp_path)
    unauthenticated = _validate(
        fixture, earlier_cells_authenticated={"cell-001": False}
    )
    assert unauthenticated.decision == NOT_RESUMABLE
    fixture["earlier"].write_text("corrupt\n", encoding="utf-8")
    corrupted = _validate(fixture)
    assert corrupted.decision == NOT_RESUMABLE


def test_partial_attempt_is_preserved_and_never_promoted(tmp_path):
    fixture = _authorization_fixture(tmp_path)
    before = sha256_file(fixture["partial"])
    result = _validate(fixture)
    assert result.resumable
    assert fixture["partial"].is_file()
    assert sha256_file(fixture["partial"]) == before
    assert not (fixture["run"] / "_SUCCESS").exists()


def test_checkpoint_restart_archives_attempt_and_gets_fresh_active_timeout(tmp_path):
    fixture = _authorization_fixture(tmp_path)
    manager = CheckpointManager(fixture["run"])
    authorization = _validate(fixture).to_dict()
    with pytest.raises(ResumeValidationError) as exc:
        manager.validate_resume(fixture["identity"], quarantine_partials=False)
    assert exc.value.code == "timeout_resume_authorization_required"

    validation = manager.validate_resume(
        fixture["identity"], resume_authorization=authorization
    )
    assert validation.resumable
    archived_partial = (
        fixture["run"]
        / "incomplete/attempt_history/attempt_01/partial_artifacts"
        / fixture["partial"].relative_to(fixture["run"])
    )
    assert archived_partial.is_file()
    payload = manager.begin_resume_attempt(
        historical_manifest_path=fixture["run"] / "manifest.json",
        resume_authorization=authorization,
        new_active_timeout_seconds=21600,
    )
    history = payload["attempt_history"][0]
    assert history["historical_manifest_status"] == "timed_out"
    assert history["prior_active_computation_seconds"] == pytest.approx(10800.025)
    assert history["new_active_timeout_seconds"] == 21600
    assert history["restart_boundary"] == "cell_boundary"
    assert (fixture["run"] / history["archived_manifest_path"]).is_file()
    assert (fixture["run"] / history["archived_checkpoint_path"]).is_file()


def test_component_cost_composition_never_downgrades_heavy_work():
    plan = load_full_baseline_plan(ROOT)
    full_lr = workload_classification(plan, plan.cells[0])
    full_catboost = workload_classification(plan, plan.cells[1])
    shap_lr = workload_classification(plan, plan.cells[24])
    rfe_lr = workload_classification(plan, plan.cells[32])
    assert full_lr.selector_cost_class == "light"
    assert full_lr.effective_cost_class == "light"
    assert full_lr.effective_wall_clock_limit_seconds == 10800
    assert full_catboost.selector_cost_class == "light"
    assert full_catboost.final_model_cost_class == "heavy"
    assert full_catboost.effective_cost_class == "heavy"
    assert full_catboost.effective_wall_clock_limit_seconds == 43200
    assert shap_lr.selector_cost_class == "heavy"
    assert shap_lr.effective_cost_class == "heavy"
    assert rfe_lr.effective_cost_class == "heavy"
    assert rfe_lr.effective_wall_clock_limit_seconds == 28800


def _policy() -> ResolvedExecutionPolicy:
    return ResolvedExecutionPolicy(
        schema_version="resource_safe_execution_policy_v1",
        profile_name="synthetic",
        parallelism=ParallelismPolicy(1, 1, 0, 4, False),
        memory=MemoryPolicy(10, 24, 28, 8, 1.35),
        gpu=GpuPolicy(1.5, 5.5, 6.5, True),
        disk=DiskPolicy(0.001, 0.001, 2.5),
        monitoring=MonitoringPolicy(0.01, 0.1, 0.1),
        configured_policy_path="synthetic",
    )


def test_execution_and_manifest_use_same_effective_classification_and_timeout(
    tmp_path, monkeypatch
):
    plan = replace(
        load_full_baseline_plan(ROOT),
        results_root=tmp_path / "results/full_baseline_v1",
    )
    cell = plan.cells[1]
    captured = {}
    expected = workload_classification(plan, cell).to_dict()
    status = build_status_report(plan)
    resume_plan = build_resume_plan_report(plan)
    assert status["cells"][1]["workload_classification"] == expected
    assert resume_plan["actions"][1]["workload_classification"] == expected

    def capture(request):
        captured["request"] = request
        return SimpleNamespace(status="completed", stop_code=None)

    monkeypatch.setattr(
        "credit_risk_fs.experiments.full_baseline.execute_registered_run", capture
    )
    ram = load_ram_control_policy(ROOT).to_dict()
    _execute_real_cell(
        plan,
        cell,
        _policy(),
        {
            "status": "pass",
            "temporary_root": str(tmp_path),
            "git_commit": "synthetic",
            "ram_control_policy": ram,
        },
        False,
    )
    request = captured["request"]
    profile = request.manifest_metadata["workload_classification"]
    assert profile == expected
    assert request.max_wall_clock_seconds == 43200
    assert request.manifest_metadata["effective_wall_clock_limit_seconds"] == 43200


def test_plan_only_is_data_free_oot_free_and_starts_no_worker(tmp_path, monkeypatch):
    plan = replace(
        load_full_baseline_plan(ROOT),
        results_root=tmp_path / "results/full_baseline_v1",
    )
    monkeypatch.setattr(
        "credit_risk_fs.pipelines.common.prepare_modeling_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("data loaded")),
    )
    monkeypatch.setattr(
        "credit_risk_fs.experiments.resource_monitor.supervise_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("worker started")),
    )
    report = build_resume_plan_report(plan)
    assert report["first_cell_to_execute"] == plan.cells[0].cell_id
    assert report["oot_accessed_by_plan"] is False
    assert report["worker_started_by_plan"] is False


def test_timed_out_inspection_requires_validator_decision(tmp_path):
    plan = replace(
        load_full_baseline_plan(ROOT),
        results_root=tmp_path / "results/full_baseline_v1",
    )
    cell = plan.cells[3]
    run = plan.results_root / "runs" / cell.dataset / cell.cell_id
    run.mkdir(parents=True)
    effective_config = {
        "full_baseline_configuration_sha256": plan.configuration_sha256
    }
    write_json_atomic(run / "config.json", effective_config)
    write_json_atomic(
        run / "manifest.json",
        {
            "run_id": cell.cell_id,
            "full_baseline_dataset": cell.dataset,
            "model": cell.model,
            "selector": cell.method_id,
            "full_baseline_configuration_sha256": plan.configuration_sha256,
            "status": "timed_out",
        },
    )
    write_json_atomic(run / "checkpoint.json", {"status": "failed"})
    pending = inspect_cell(
        plan,
        cell,
        timeout_validator=lambda *_args: TimeoutResumeValidation(
            decision=NOT_RESUMABLE,
            run_id=cell.cell_id,
            cell_id=cell.cell_id,
            historical_terminal_state="timed_out",
            historical_stop_reason="wall_clock_limit",
            intended_restart_boundary="cell_boundary",
            checks=(),
            reasons=("pending explicit validation",),
            validation_timestamp_utc="now",
            validator_version=VALIDATOR_VERSION,
            authorization_path=None,
            authorization_sha256=None,
            partial_artifacts=(),
            active_processes=(),
            lock_paths=(),
            checkpoint_identity=None,
        ),
    )
    assert pending.state == "timed_out"
    assert pending.resumable is False
