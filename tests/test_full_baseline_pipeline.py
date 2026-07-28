from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from credit_risk_fs.experiments.full_baseline import (
    DATASET_ORDER,
    EXPECTED_CELL_COUNT,
    METHOD_ORDER,
    MODEL_ORDER,
    CellInspection,
    FullBaselineArtifactError,
    FullBaselineConfigurationError,
    _effective_configuration,
    _experiment_configuration,
    _validate_protocol,
    build_status_report,
    execute_full_baseline,
    load_full_baseline_plan,
)
from credit_risk_fs.experiments.resource_policy import (
    DiskPolicy,
    GpuPolicy,
    MemoryPolicy,
    MonitoringPolicy,
    ParallelismPolicy,
    ResolvedExecutionPolicy,
)
from credit_risk_fs.pipelines.common import _resolve_selector


ROOT = Path(__file__).resolve().parents[1]


def _policy() -> ResolvedExecutionPolicy:
    return ResolvedExecutionPolicy(
        schema_version="resource_safe_execution_policy_v1",
        profile_name="synthetic_full_baseline",
        parallelism=ParallelismPolicy(1, 1, 0, 4, False),
        memory=MemoryPolicy(10, 24, 28, 8, 1.35),
        gpu=GpuPolicy(1.5, 5.5, 6.5, True),
        disk=DiskPolicy(0.001, 0.001, 2.5),
        monitoring=MonitoringPolicy(0.02, 0.1, 1.0),
        configured_policy_path="synthetic",
    )


def _plan(tmp_path: Path):
    loaded = load_full_baseline_plan(ROOT)
    return replace(
        loaded,
        results_root=tmp_path / "results" / "full_baseline_v1",
        log_path=tmp_path / "logs" / "runs.log",
    )


def _preflight(_plan):
    return _policy(), {
        "status": "pass",
        "blocking_reasons": [],
        "temporary_root": str(_plan.results_root.parent),
    }


def _ready(**_kwargs):
    return SimpleNamespace(ready=True, stop_code=None)


def test_frozen_matrix_is_exactly_36_method_dataset_model_cells():
    plan = load_full_baseline_plan(ROOT)
    assert len(plan.cells) == EXPECTED_CELL_COUNT == 36
    assert [
        (cell.method_id, cell.dataset, cell.model, cell.seed) for cell in plan.cells
    ] == [
        (method, dataset, model, 42)
        for method in METHOD_ORDER
        for dataset in DATASET_ORDER
        for model in MODEL_ORDER
    ]
    assert len({cell.cell_id for cell in plan.cells}) == 36
    assert all(cell.feature_budget == 20 for cell in plan.cells if cell.model == "lr" and cell.method_id != "full_features")
    assert all(cell.feature_budget == 40 for cell in plan.cells if cell.model == "catboost" and cell.method_id != "full_features")
    assert all(cell.feature_budget is None for cell in plan.cells if cell.method_id == "full_features")


def test_boruta_shortfall_is_frozen_as_confirmed_only_without_padding():
    plan = load_full_baseline_plan(ROOT)
    settings = plan.configuration["selector_settings"]["boruta_random_forest"]
    decision = plan.configuration["pilot_review_decisions"]["homecredit_boruta"]
    assert settings["selection_mode"] == "confirmed_top_k"
    assert decision == {
        "requested_budget": 40,
        "confirmed_count": 26,
        "tentative_count": 25,
        "rejected_count": 478,
        "pilot_budget_status": "infeasible_natural_support",
        "decision": "retain_confirmed_only_without_padding",
        "full_baseline_interpretation": "actual_selected_count_may_be_below_model_budget",
    }
    assert "confirmed_then_tentative" not in str(settings)


def test_every_frozen_selector_and_model_configuration_constructs_without_data(tmp_path):
    plan = _plan(tmp_path)
    policy = _policy()
    for cell in plan.cells:
        run_dir = plan.results_root / "construction" / cell.cell_id
        effective = _effective_configuration(plan, cell, policy)
        experiment = _experiment_configuration(plan, cell, run_dir, effective)
        selector_cls, selector_kwargs = _resolve_selector(experiment)
        selector = selector_cls(**selector_kwargs)
        assert selector is not None
        assert experiment.n_splits == 5
        assert experiment.cv_gap_groups == 1
        assert experiment.estimator_threads == 4
        if cell.model == "catboost":
            assert experiment.model_kwargs["iterations"] == 1500
            assert experiment.model_kwargs["thread_count"] == 4
        else:
            assert experiment.model_kwargs["solver"] == "liblinear"


def test_protocol_mutations_fail_closed():
    plan = load_full_baseline_plan(ROOT)
    for field, value in (
        ("expected_cell_count", 35),
        ("configuration_adaptation_after_oot", "allowed"),
        ("concurrent_cells", 2),
        ("random_seeds", [7]),
    ):
        mutated = copy.deepcopy(plan.configuration)
        mutated["protocol"][field] = value
        with pytest.raises(FullBaselineConfigurationError, match=field):
            _validate_protocol(mutated)


def test_status_is_data_free_and_reports_earliest_missing(tmp_path, monkeypatch):
    plan = _plan(tmp_path)
    monkeypatch.setattr(
        "credit_risk_fs.pipelines.common.prepare_modeling_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("data accessed")),
    )
    report = build_status_report(plan)
    assert report["completed_authenticated"] == 0
    assert report["next_cell"] == plan.cells[0].cell_id
    assert report["oot_accessed_by_status"] is False


def test_runner_skips_completed_resumes_first_invalid_and_finishes_in_order(tmp_path):
    plan = _plan(tmp_path)
    states = {
        cell.cell_id: (
            CellInspection("completed", True, False, "valid")
            if cell.cell_index <= 2
            else CellInspection("failed", False, True, "resume")
            if cell.cell_index == 3
            else CellInspection("missing", False, False, "missing")
        )
        for cell in plan.cells
    }
    calls: list[tuple[str, bool]] = []

    def inspect(_plan, cell):
        return states[cell.cell_id]

    def execute(_plan, cell, _policy, _preflight, resume):
        calls.append((cell.cell_id, resume))
        states[cell.cell_id] = CellInspection("completed", True, False, "valid")
        return SimpleNamespace(status="completed", stop_code=None)

    outcome = execute_full_baseline(
        plan,
        require_clean_repository=False,
        authenticate_pilots=False,
        policy_preflight=_preflight,
        inspector=inspect,
        executor=execute,
        readiness_checker=_ready,
        progress_writer=lambda _plan: None,
    )
    assert outcome.status == "completed"
    assert outcome.completed_cells == 36
    assert calls == [
        (cell.cell_id, cell.cell_index == 3) for cell in plan.cells[2:]
    ]


def test_runner_stops_at_first_controlled_failure(tmp_path):
    plan = _plan(tmp_path)
    states = {
        cell.cell_id: CellInspection("missing", False, False, "missing")
        for cell in plan.cells
    }
    calls = []

    def execute(_plan, cell, _policy, _preflight, _resume):
        calls.append(cell.cell_id)
        return SimpleNamespace(status="aborted_resource_limit", stop_code="ram_process_limit")

    outcome = execute_full_baseline(
        plan,
        require_clean_repository=False,
        authenticate_pilots=False,
        policy_preflight=_preflight,
        inspector=lambda _plan, cell: states[cell.cell_id],
        executor=execute,
        readiness_checker=_ready,
        progress_writer=lambda _plan: None,
    )
    assert outcome.status == "aborted_resource_limit"
    assert outcome.stop_cell_id == plan.cells[0].cell_id
    assert calls == [plan.cells[0].cell_id]


def test_invalid_completed_artifact_is_never_overwritten(tmp_path):
    plan = _plan(tmp_path)
    called = False

    def execute(*_args):
        nonlocal called
        called = True

    with pytest.raises(FullBaselineArtifactError, match="cannot safely continue"):
        execute_full_baseline(
            plan,
            require_clean_repository=False,
            authenticate_pilots=False,
            policy_preflight=_preflight,
            inspector=lambda *_args: CellInspection(
                "invalid_completed", False, False, "hash mismatch"
            ),
            executor=execute,
            readiness_checker=_ready,
            progress_writer=lambda _plan: None,
        )
    assert called is False
