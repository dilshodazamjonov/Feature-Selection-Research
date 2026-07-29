from __future__ import annotations

import inspect
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from credit_risk_fs.experiments import atomic_io
from credit_risk_fs.experiments.heavy_selector_pilots import (
    DATASET_ORDER,
    METHOD_ORDER,
    PilotConfigurationError,
    _artifact_base,
    _validate_exact_protocol,
    _write_cell_state,
    build_status_report,
    cell_artifact_path,
    execute_pilot_pipeline,
    format_status_report,
    heavy_selector_pilot_cell_worker,
    load_pilot_plan,
    validate_cell_artifact,
)
from credit_risk_fs.experiments.research_logging import ResearchLogSession
from credit_risk_fs.experiments.resource_monitor import (
    RAM_PROCESS_LIMIT,
    WALL_CLOCK_LIMIT,
    ResourceSample,
    SupervisorResult,
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


ROOT = Path(__file__).resolve().parents[1]


def _policy(*, warn_ram: float = 1.0, abort_ram: float = 2.0):
    return ResolvedExecutionPolicy(
        schema_version="resource_safe_execution_policy_v1",
        profile_name="synthetic_pilot_test",
        parallelism=ParallelismPolicy(1, 1, 0, 2, False),
        memory=MemoryPolicy(1.0, warn_ram, abort_ram, 0.001, 1.35),
        gpu=GpuPolicy(0.1, 1.0, 2.0, True),
        disk=DiskPolicy(0.001, 0.001, 2.5),
        monitoring=MonitoringPolicy(0.02, 0.1, 1.0),
        configured_policy_path="synthetic",
    )


def _plan(tmp_path):
    loaded = load_pilot_plan(ROOT)
    return replace(
        loaded,
        results_root=tmp_path / "results" / "heavy_selector_dev_pilots_v1",
        log_path=tmp_path / "logs" / "runs.log",
    )


def _preflight(plan):
    plan.results_root.mkdir(parents=True, exist_ok=True)
    return _policy(), {"status": "pass", "blocking_reasons": []}


def _readiness(**_kwargs):
    return SimpleNamespace(
        ready=True,
        stop_code=None,
        elapsed_seconds=0.01,
        parent_rss_bytes=10_000,
        system_available_ram_bytes=10 * 1024**3,
    )


def _worker_payload(spec):
    count = int(spec["expected_candidate_count"])
    selected = [f"feature_{index:04d}" for index in range(40)]
    method_id = spec["method_id"]
    if method_id == "catboost_shap":
        heavy = {
            "estimator_fit_count": 1,
            "shap_calculation_count": 1,
            "feature_importance_type": "ShapValues",
            "shap_calc_type": "Regular",
            "aggregation": "mean_absolute_shap_over_explanation_rows",
            "explanation_sample": {
                "realized_size": 100,
                "row_identity_sha256": "a" * 64,
                "scope": "selector_training_partition_only",
            },
        }
        natural_count = None
    elif method_id == "boruta_random_forest":
        confirmed = 50
        tentative = 20
        heavy = {
            "confirmed_count": confirmed,
            "tentative_count": tentative,
            "rejected_count": count - confirmed - tentative,
            "selection_mode": "confirmed_top_k",
            "natural_support_definition": "confirmed_only",
            "forest_n_estimators_configured": 500,
            "boruta_max_iter_configured": 10,
            "engine_iteration_count": 9,
            "estimator_fit_count": 9,
        }
        natural_count = confirmed
    else:
        heavy = {
            "initial_feature_count": count,
            "final_feature_count": 40,
            "requested_elimination_steps": count - 40,
            "realized_elimination_steps": count - 40,
            "elimination_iteration_count": 1,
            "elimination_history": [
                {
                    "iteration": 1,
                    "surviving_before": count,
                    "requested_removals": count - 40,
                    "realized_removals": count - 40,
                    "removed_features": [],
                }
            ],
            "estimator_fit_count": 2,
        }
        natural_count = None
    selector_result = {
        "method_id": method_id,
        "implementation_id": spec["implementation_id"],
        "configuration": dict(spec["selector_kwargs"]),
        "candidate_universe_count": count,
        "candidate_universe_sha256": "b" * 64,
        "selected_features": selected,
        "actual_selected_count": len(selected),
        "requested_budget": 40,
        "natural_selected_count": natural_count,
        "budget_status": "satisfied",
        "heavy_metadata": heavy,
    }
    return {
        "pilot_cell": spec["cell_id"],
        "selector_result": selector_result,
        "authenticated_dev_identity": {
            "dataset": spec["dataset"],
            "phase": "DEV",
            "fold_id": 1,
            "dev_ordered_row_id_sha256": "c" * 64,
            "dev_ordered_row_id_target_sha256": "d" * 64,
            "training_ordered_row_id_sha256": "e" * 64,
            "training_ordered_row_id_target_sha256": "f" * 64,
            "validation_ordered_row_id_sha256": "1" * 64,
        },
        "input_counts": {
            "dev_rows": 500,
            "fold_training_rows": 200,
            "fold_validation_rows": 50,
            "candidate_features": count,
        },
        "preprocessing": {
            "implementation": "OriginalFeatureNumericEncoder",
            "fit_scope": "dev_fold_training_only",
        },
        "oot_accessed": False,
        "performance_evaluation_performed": False,
    }


def _supervisor_result(spec, *, status="completed", stop_code=None):
    return SupervisorResult(
        status=status,
        stop_code=stop_code,
        worker_exit_code=0 if status == "completed" else 1,
        return_value=_worker_payload(spec) if status == "completed" else None,
        worker_error=None,
        samples=(),
        warnings=(),
        peak_process_tree_rss_bytes=256 * 1024**2,
        peak_process_gpu_bytes=None,
        minimum_system_available_ram_bytes=8 * 1024**3,
        minimum_results_free_disk_bytes=100 * 1024**3,
        minimum_temp_free_disk_bytes=100 * 1024**3,
        child_cleanup_confirmed=True,
        final_stage=f"pilot_{spec['method_id']}",
        final_fold_id=1,
    )


def _successful_supervisor(calls):
    def run(**kwargs):
        spec = kwargs["worker_kwargs"]["spec"]
        calls.append(spec["cell_id"])
        assert kwargs["max_wall_clock_seconds"] == spec["wall_clock_limit_seconds"]
        return _supervisor_result(spec)

    return run


def test_exact_six_cell_method_major_cheapest_first_order(tmp_path):
    plan = _plan(tmp_path)
    assert len(plan.cells) == 6
    assert [(cell.dataset, cell.method_id, cell.fold_id) for cell in plan.cells] == [
        (dataset, method, 1)
        for method in METHOD_ORDER
        for dataset in DATASET_ORDER
    ]
    calls = []
    outcome = execute_pilot_pipeline(
        plan,
        require_clean_repository=False,
        supervisor=_successful_supervisor(calls),
        readiness_checker=_readiness,
        policy_preflight=_preflight,
    )
    assert outcome.status == "completed"
    assert calls == [cell.cell_id for cell in plan.cells]
    assert all(validate_cell_artifact(plan, cell).valid for cell in plan.cells)


def test_dev_fold_one_and_oot_prohibition_fail_closed():
    plan = load_pilot_plan(ROOT)
    invalid_fold = json.loads(json.dumps(plan.configuration))
    invalid_fold["protocol"]["fold_ids"] = [2]
    with pytest.raises(PilotConfigurationError, match="fold_ids"):
        _validate_exact_protocol(invalid_fold)
    invalid_oot = json.loads(json.dumps(plan.configuration))
    invalid_oot["protocol"]["oot_access"] = "allowed"
    with pytest.raises(PilotConfigurationError, match="oot_access"):
        _validate_exact_protocol(invalid_oot)
    source = inspect.getsource(heavy_selector_pilot_cell_worker)
    assert "prepare_voting_pilot_dev_data" in source
    assert "prepare_voting_research_oot_data" not in source
    assert "evaluate_model" not in source


def test_valid_cells_skip_unchanged_and_corrupt_cell_resumes_first(tmp_path):
    plan = _plan(tmp_path)
    initial_calls = []
    execute_pilot_pipeline(
        plan,
        require_clean_repository=False,
        supervisor=_successful_supervisor(initial_calls),
        readiness_checker=_readiness,
        policy_preflight=_preflight,
    )
    before = {
        cell.cell_id: cell_artifact_path(plan, cell).read_bytes() for cell in plan.cells
    }
    no_calls = []
    execute_pilot_pipeline(
        plan,
        require_clean_repository=False,
        supervisor=_successful_supervisor(no_calls),
        readiness_checker=_readiness,
        policy_preflight=_preflight,
    )
    assert no_calls == []
    assert before == {
        cell.cell_id: cell_artifact_path(plan, cell).read_bytes() for cell in plan.cells
    }

    corrupted = plan.cells[2]
    cell_artifact_path(plan, corrupted).write_text("{broken", encoding="utf-8")
    assert not validate_cell_artifact(plan, corrupted).valid
    resume_calls = []
    outcome = execute_pilot_pipeline(
        plan,
        require_clean_repository=False,
        supervisor=_successful_supervisor(resume_calls),
        readiness_checker=_readiness,
        policy_preflight=_preflight,
    )
    assert outcome.status == "completed"
    assert resume_calls == [corrupted.cell_id]
    assert validate_cell_artifact(plan, corrupted).valid
    for cell in plan.cells:
        if cell != corrupted:
            assert cell_artifact_path(plan, cell).read_bytes() == before[cell.cell_id]


def test_atomic_state_write_does_not_publish_partial_on_replace_failure(
    tmp_path, monkeypatch
):
    plan = _plan(tmp_path)
    cell = plan.cells[0]
    payload = _artifact_base(
        plan,
        cell,
        terminal_state="running",
        stop_reason=None,
        estimator_threads=1,
    )

    def fail_replace(_source, _target):
        raise OSError("synthetic replace interruption")

    monkeypatch.setattr(atomic_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace interruption"):
        _write_cell_state(plan, cell, payload)
    assert not cell_artifact_path(plan, cell).exists()
    partials = list(cell_artifact_path(plan, cell).parent.glob("*.partial"))
    assert len(partials) == 1


@pytest.mark.parametrize(
    ("mode", "expected_state", "expected_code"),
    [
        ("interrupt", "manually_interrupted", "manual_interrupt"),
        ("timeout", "timed_out", WALL_CLOCK_LIMIT),
        ("ram", "resource_aborted", RAM_PROCESS_LIMIT),
    ],
)
def test_controlled_stop_states_preserve_prior_cells(
    tmp_path, mode, expected_state, expected_code
):
    plan = _plan(tmp_path)
    calls = []

    def supervisor(**kwargs):
        spec = kwargs["worker_kwargs"]["spec"]
        calls.append(spec["cell_id"])
        if len(calls) == 1:
            return _supervisor_result(spec)
        if mode == "interrupt":
            raise KeyboardInterrupt("synthetic manual stop")
        if mode == "timeout":
            return _supervisor_result(
                spec, status="timed_out", stop_code=WALL_CLOCK_LIMIT
            )
        return _supervisor_result(
            spec, status="aborted_resource_limit", stop_code=RAM_PROCESS_LIMIT
        )

    outcome = execute_pilot_pipeline(
        plan,
        require_clean_repository=False,
        supervisor=supervisor,
        readiness_checker=_readiness,
        policy_preflight=_preflight,
    )
    assert outcome.status == expected_state
    assert outcome.stop_code == expected_code
    assert len(calls) == 2
    assert validate_cell_artifact(plan, plan.cells[0]).valid
    stopped = json.loads(cell_artifact_path(plan, plan.cells[1]).read_text())
    assert stopped["terminal_state"] == expected_state
    assert stopped["stop_reason"] == expected_code
    assert not cell_artifact_path(plan, plan.cells[2]).exists()


def test_status_reads_artifacts_only_and_reports_method_evidence(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    execute_pilot_pipeline(
        plan,
        require_clean_repository=False,
        supervisor=_successful_supervisor([]),
        readiness_checker=_readiness,
        policy_preflight=_preflight,
    )

    def forbidden_loader(*_args, **_kwargs):
        raise AssertionError("status must not load a dataset")

    import credit_risk_fs.pipelines.common as common

    monkeypatch.setattr(common, "prepare_voting_pilot_dev_data", forbidden_loader)
    report = build_status_report(plan)
    rendered = format_status_report(report)
    assert report["completed_cells"] == report["total_cells"] == 6
    assert report["current_or_next_cell"] is None
    assert report["dataset_access_performed"] is False
    assert "boruta=50/20/" in rendered
    assert "rfe_fits=2" in rendered
    assert "shap_sample=100" in rendered


def test_logging_contains_cell_fold_method_resources_and_stop(tmp_path):
    plan = _plan(tmp_path)

    def stopped(**kwargs):
        spec = kwargs["worker_kwargs"]["spec"]
        return _supervisor_result(
            spec, status="aborted_resource_limit", stop_code=RAM_PROCESS_LIMIT
        )

    terminal = io.StringIO()
    with ResearchLogSession(
        plan.log_path,
        repository_root=ROOT,
        command_arguments=["synthetic"],
        terminal_stream=terminal,
    ) as session:
        outcome = execute_pilot_pipeline(
            plan,
            require_clean_repository=False,
            supervisor=stopped,
            readiness_checker=_readiness,
            policy_preflight=_preflight,
        )
        session.finish(
            "session_controlled_stop",
            level="ERROR",
            message="synthetic RAM stop",
            stop_code=outcome.stop_code,
        )
    events = [
        json.loads(line)
        for line in plan.log_path.with_name("events.jsonl").read_text().splitlines()
    ]
    finalized = next(item for item in events if item["event"] == "run_finalized")
    assert finalized["pilot_cell"] == plan.cells[0].cell_id
    assert finalized["dataset"] == "homecredit"
    assert finalized["fold_id"] == 1
    assert finalized["selector"] == "catboost_shap"
    assert finalized["peak_process_tree_rss_bytes"] == 256 * 1024**2
    assert finalized["minimum_system_available_ram_bytes"] == 8 * 1024**3
    assert finalized["stop_code"] == RAM_PROCESS_LIMIT
    assert "RAM safety limit reached" in terminal.getvalue()


def test_running_and_unexpected_failure_states_are_durable_with_traceback(
    tmp_path,
):
    plan = _plan(tmp_path)
    observed_running = []

    def unexpected(**kwargs):
        spec = kwargs["worker_kwargs"]["spec"]
        running = json.loads(
            cell_artifact_path(plan, plan.cells[0]).read_text(encoding="utf-8")
        )
        observed_running.append(running["terminal_state"])
        raise RuntimeError(f"synthetic unexpected failure for {spec['cell_id']}")

    with ResearchLogSession(
        plan.log_path,
        repository_root=ROOT,
        command_arguments=["synthetic-failure"],
        terminal_stream=io.StringIO(),
    ) as session:
        outcome = execute_pilot_pipeline(
            plan,
            require_clean_repository=False,
            supervisor=unexpected,
            readiness_checker=_readiness,
            policy_preflight=_preflight,
        )
        session.finish(
            "session_failed",
            level="ERROR",
            message="synthetic unexpected pilot failure",
            exception_class="RuntimeError",
        )
    assert observed_running == ["running"]
    assert outcome.status == "failed"
    artifact = json.loads(
        cell_artifact_path(plan, plan.cells[0]).read_text(encoding="utf-8")
    )
    assert artifact["terminal_state"] == "failed"
    assert artifact["stop_reason"] == "worker_crash"
    debug = plan.log_path.with_name("debug.log").read_text(encoding="utf-8")
    assert "RuntimeError: synthetic unexpected failure" in debug


class _HighRamSampler:
    def __init__(self, **_kwargs):
        self.gpu = SimpleNamespace(close=lambda: None)

    def sample(self, worker_pid, *, stage=None, fold_id=None):
        gib = 1024**3
        return ResourceSample(
            elapsed_seconds=0.02,
            worker_pid=int(worker_pid),
            child_pids=(),
            process_tree_rss_bytes=int(0.2 * gib),
            system_available_ram_bytes=10 * gib,
            process_gpu_bytes=None,
            results_free_disk_bytes=100 * gib,
            temp_free_disk_bytes=100 * gib,
            process_tree_cpu_percent=1.0,
            process_tree_cpu_seconds=0.01,
            stage=stage,
            fold_id=fold_id,
        )


def test_supervisor_wall_clock_stop_is_bounded_and_high_rss_is_nonterminal(tmp_path):
    timeout = supervise_worker(
        worker_target=(
            "credit_risk_fs.experiments.synthetic_execution:uncooperative_wait_worker"
        ),
        worker_kwargs={},
        policy=_policy(),
        results_root=tmp_path,
        temp_root=tmp_path,
        max_wall_clock_seconds=0.06,
        heartbeat_interval_seconds=0.02,
    )
    assert timeout.status == "timed_out"
    assert timeout.stop_code == WALL_CLOCK_LIMIT
    assert timeout.child_cleanup_confirmed

    ram = supervise_worker(
        worker_target=(
            "credit_risk_fs.experiments.synthetic_execution:immediate_success_worker"
        ),
        worker_kwargs={},
        policy=_policy(warn_ram=0.05, abort_ram=0.08),
        results_root=tmp_path,
        temp_root=tmp_path,
        sampler_factory=_HighRamSampler,
        heartbeat_interval_seconds=0.02,
    )
    assert ram.status == "completed"
    assert ram.stop_code is None
    assert ram.child_cleanup_confirmed
