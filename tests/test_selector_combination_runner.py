from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import credit_risk_fs.experiments.selector_combinations as runner
from credit_risk_fs.experiments.selector_combinations import (
    CombinationGateClosed,
    build_status,
    build_phase_matrix,
    enforce_phase_gate,
    load_combination_plan,
    render_plan,
    _publish_selection_files,
    _validate_artifact,
    validate_prompt_10_baselines,
)
from credit_risk_fs.experiments.resource_monitor import MANUAL_INTERRUPT


ROOT = Path(__file__).resolve().parents[1]


def test_plan_is_exact_data_free_and_cheapest_first() -> None:
    plan = load_combination_plan(ROOT)
    rendered = render_plan(plan)
    assert rendered["pilot_selection_count"] == 18
    assert rendered["pilot_evaluation_count"] == 24
    assert rendered["raw_dataset_paths_resolved"] is False
    assert rendered["workers_started"] == 0
    assert rendered["first_cell"]["cell_id"] == (
        "scv1-pilot-001-homecredit-statistical-normalized-average-rank-lr-k20-s42"
    )
    assert [cell.method_id for cell in plan.evaluations] == sorted(
        [cell.method_id for cell in plan.evaluations],
        key=lambda item: rendered["execution_order"].index(item),
    )
    assert len({cell.selection_id for cell in plan.evaluations}) == 18
    assert all(cell.fold_id == 1 for cell in plan.evaluations)


def test_status_is_artifact_only_and_starts_no_worker(tmp_path: Path) -> None:
    plan = load_combination_plan(ROOT)
    object.__setattr__(plan, "results_root", tmp_path / "never-created")
    status = build_status(plan)
    assert status["raw_dataset_paths_resolved"] is False
    assert status["workers_started"] == 0
    assert status["pilot"]["authenticated_evaluations"] == 0
    assert not plan.results_root.exists()


def test_prompt_10_dependency_is_exactly_authenticated_without_refit() -> None:
    result = validate_prompt_10_baselines(load_combination_plan(ROOT))
    assert result["expected_cells"] == result["authenticated_cells"] == 36
    assert result["cell_036"] == "fbv1-036-lendingclub_v2-catboost-rfe-catboost-s42"
    assert result["raw_dataset_paths_resolved"] is False
    assert result["baseline_refit_performed"] is False


def test_dev_and_oot_are_technically_gated_before_pilot_approval() -> None:
    plan = load_combination_plan(ROOT)
    approval = ROOT / plan.configuration["gates"]["pilot_approval_lock_path"]
    assert not approval.exists()
    with pytest.raises(CombinationGateClosed, match="pilot review"):
        enforce_phase_gate(plan, "dev")
    with pytest.raises(CombinationGateClosed, match="pilot review"):
        enforce_phase_gate(plan, "oot")


def test_protocol_lock_keeps_exact_four_and_exact_five_voters() -> None:
    plan = load_combination_plan(ROOT)
    registry = json.loads(
        (plan.protocol_lock_path.parent / "combination_method_registry.json").read_text(encoding="utf-8")
    )
    methods = registry["methods"]
    assert [item["method_id"] for item in methods] == [
        "iv_then_boruta",
        "boruta_then_rfe_catboost",
        "boruta_then_mrmr_mutual_information",
        "statistical_normalized_average_rank",
    ]
    voter = methods[-1]
    assert voter["components"] == [
        "iv_woe", "lasso_l1_logistic", "rfe_catboost", "boruta_random_forest", "catboost_shap"
    ]
    assert voter["weights"] == [0.2] * 5


def test_preregistration_commit_precedes_outcome_audit_and_preserves_semantic_protocol() -> None:
    commit = "9bc69d0c25ef09d0df2d7dc71c45bea231c5a3e4"
    files = subprocess.run(
        ["git", "show", "--pretty=format:", "--name-only", commit],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert "configs/protocols/selector_combinations_v1/combination_protocol_lock.json" in files
    assert not any("baseline_pairwise_comparisons" in name for name in files)
    lock = json.loads(
        (ROOT / "configs/protocols/selector_combinations_v1/combination_protocol_lock.json").read_text(encoding="utf-8")
    )
    semantic = next(
        item for item in lock["decision_sources"]
        if item["path"] == "configs/protocols/cross_dataset_rank_voting_v1.yaml"
    )
    import hashlib

    assert hashlib.sha256((ROOT / semantic["path"]).read_bytes()).hexdigest() == semantic["sha256"]


def test_selection_publication_separates_and_authenticates_stage_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    worker = {
        "combination_result": {
            "selected_features": ["b"],
            "intermediate_features": ["a", "b"],
            "voting_evidence": None,
        }
    }
    files = _publish_selection_files(path, worker)
    assert {item["path"] for item in files} == {
        "selection.combination_result.json",
        "selection.final_selected_features.csv",
        "selection.intermediate_features.csv",
    }
    payload = {
        "terminal_state": "completed",
        "selection_id": "s",
        "configuration_sha256": "c" * 64,
        "artifact_files": files,
    }
    from credit_risk_fs.experiments.selector_combinations import _artifact_with_hash

    path.write_text(json.dumps(_artifact_with_hash(payload)), encoding="utf-8")
    valid, _, _ = _validate_artifact(
        path, {"selection_id": "s", "configuration_sha256": "c" * 64}
    )
    assert valid
    (tmp_path / "selection.final_selected_features.csv").write_text("corrupt", encoding="utf-8")
    valid, reason, _ = _validate_artifact(
        path, {"selection_id": "s", "configuration_sha256": "c" * 64}
    )
    assert not valid and reason.startswith("artifact_file_mismatch")


def test_heavy_chain_and_final_model_limits_cannot_be_downgraded() -> None:
    plan = load_combination_plan(ROOT)
    assert all(item.wall_clock_limit_seconds >= 21_600 for item in plan.selections)
    assert all(
        cell.wall_clock_limit_seconds == (10_800 if cell.model == "lr" else 43_200)
        for cell in plan.evaluations
    )


def test_post_approval_phase_matrices_are_exact_and_keep_oot_out_of_dev() -> None:
    plan = load_combination_plan(ROOT)
    dev_selections, dev_cells = build_phase_matrix(plan, phase="dev")
    oot_selections, oot_cells = build_phase_matrix(plan, phase="oot")
    assert len(dev_selections) == 90
    assert len(dev_cells) == 120
    assert {item.fold_id for item in dev_selections} == {1, 2, 3, 4, 5}
    assert all(item.selector_kwargs["fit_scope"] == "dev_fold_training_only" for item in dev_selections)
    assert len(oot_selections) == 18
    assert len(oot_cells) == 24
    assert {item.fold_id for item in oot_selections} == {0}
    assert all(item.selector_kwargs["fit_scope"] == "full_dev_only" for item in oot_selections)
    assert not set(cell.cell_id for cell in dev_cells) & set(cell.cell_id for cell in oot_cells)


class _FakeLogSession:
    instances: list["_FakeLogSession"] = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.finishes: list[tuple[str, dict]] = []
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def finish(self, event: str, **fields) -> None:
        self.finishes.append((event, fields))


def test_real_phase_uses_standard_log_session_and_preserves_manual_interrupt_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeLogSession.instances.clear()
    monkeypatch.setattr(runner, "ResearchLogSession", _FakeLogSession)
    monkeypatch.setattr(
        runner,
        "execute_pilot",
        lambda _plan: {
            "status": "interrupted",
            "stop_code": MANUAL_INTERRUPT,
            "stop_cell": "cell-001",
        },
    )
    assert runner.main(["--repository-root", str(ROOT), "--phase", "pilot"]) == 130
    assert len(_FakeLogSession.instances) == 1
    event, fields = _FakeLogSession.instances[0].finishes[-1]
    assert event == "session_controlled_stop"
    assert fields["stop_code"] == MANUAL_INTERRUPT


def test_unexpected_real_phase_error_is_bound_to_debug_logging_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeLogSession.instances.clear()
    monkeypatch.setattr(runner, "ResearchLogSession", _FakeLogSession)

    def fail(_plan):
        raise RuntimeError("synthetic worker failure")

    monkeypatch.setattr(runner, "execute_pilot", fail)
    assert runner.main(["--repository-root", str(ROOT), "--phase", "pilot"]) == 1
    event, fields = _FakeLogSession.instances[0].finishes[-1]
    assert event == "session_failed"
    assert fields["exception_class"] == "RuntimeError"
    assert "synthetic worker failure" in fields["traceback"]
