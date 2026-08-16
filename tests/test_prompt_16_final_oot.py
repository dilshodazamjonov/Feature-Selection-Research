from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk_fs.data.homecredit_model_stability_2024.contract import (
    canonical_sha256,
)
from credit_risk_fs.evaluation.metrics import evaluate_model
from credit_risk_fs.experiments.atomic_io import (
    write_json_atomic,
    write_text_atomic,
)
from credit_risk_fs.experiments.prompt_16_final_oot import (
    BOOTSTRAP_MINIMUM_VALID,
    BOOTSTRAP_REPETITIONS,
    FREEZE_RELATIVE_ROOT,
    HOLM_ALPHA,
    INHERITED_RESOURCE_INFEASIBLE_CELL_ORDERS,
    INHERITED_RESOURCE_INFEASIBLE_FIT_IDS,
    MAX_ESTIMATOR_THREADS,
    MAX_RESOURCE_RECOVERY_RESTARTS_PER_SCOPE,
    PROJECT_ROOT,
    PROTOCOL_RELATIVE_PATH,
    RESUME_AVAILABLE_RAM_GIB,
    RESUME_STABILITY_POLLS,
    SOFT_AVAILABLE_RAM_GIB,
    SYSTEM_AVAILABLE_RAM_HARD_FLOOR_GIB,
    _load_final_sealed,
    _reconcile_prediction_metrics,
    _seal_final_directory,
    _self_authenticated_payload,
    final_full_dev_refits,
    final_oot_cells,
    paired_comparison_graph,
)
from credit_risk_fs.experiments.prompt_16_third_dataset import (
    Prompt16ExecutionError,
    _fit_and_evaluate,
    _protocol_payload,
    run_phase_worker,
)
from credit_risk_fs.experiments.ram_control import load_ram_control_policy
from credit_risk_fs.experiments.resource_policy import GIB


PROTOCOL = PROJECT_ROOT / PROTOCOL_RELATIVE_PATH


def test_exact_34_cell_identity_and_ordering() -> None:
    cells = final_oot_cells(PROTOCOL)
    assert len(cells) == 34
    assert [cell["configuration_order"] for cell in cells] == list(range(1, 35))
    assert [(cell["method_id"], cell["model"]) for cell in cells[-4:]] == [
        ("llm", "lr"),
        ("llm", "catboost"),
        ("stable_core_llm_fill", "lr"),
        ("stable_core_llm_fill", "catboost"),
    ]


def test_full_dev_refit_accounting_is_exact() -> None:
    rows = final_full_dev_refits(PROTOCOL)
    assert len(rows) == 29
    assert sum(row["method_id"] == "stable_core_llm_fill" for row in rows) == 2
    assert sum(row["internal_component_fits"] for row in rows) == 37
    assert sum(
        row["internal_component_fits"]
        for row in rows
        if row["method_id"] == "stable_core_llm_fill"
    ) == 10


def test_comparison_and_holm_graph_integrity() -> None:
    graph = paired_comparison_graph(PROTOCOL)
    assert len(graph) == 72
    assert sum(row["availability"] == "registered" for row in graph) == 70
    assert sum(row["availability"] != "registered" for row in graph) == 2
    assert len({row["comparison_id"] for row in graph}) == 72
    supplemental = [
        row for row in graph if row["holm_family_id"].startswith("third_dataset_llm_primary")
    ]
    assert len(supplemental) == 10
    assert HOLM_ALPHA == 0.05
    assert BOOTSTRAP_REPETITIONS == 2000
    assert BOOTSTRAP_MINIMUM_VALID == 1900


def test_resource_policy_is_final_stricter_contract() -> None:
    policy = load_ram_control_policy(
        PROJECT_ROOT,
        "configs/execution/prompt_16_final_oot_ram_wait_v1.yaml",
        total_physical_ram_bytes=32 * GIB,
    )
    assert MAX_ESTIMATOR_THREADS == 4
    assert SYSTEM_AVAILABLE_RAM_HARD_FLOOR_GIB == 4
    assert SOFT_AVAILABLE_RAM_GIB == 6
    assert RESUME_AVAILABLE_RAM_GIB == 8
    assert RESUME_STABILITY_POLLS == 3
    assert policy.emergency_margin_bytes == 6 * GIB
    assert policy.recovery_threshold_bytes == 8 * GIB
    assert policy.recovery_consecutive_checks == 3
    assert policy.check_interval_seconds <= 5
    assert policy.log_interval_seconds <= 30
    assert MAX_RESOURCE_RECOVERY_RESTARTS_PER_SCOPE == 5


def test_old_classical_oot_entry_is_revoked_before_any_data_load(tmp_path: Path) -> None:
    with pytest.raises(Prompt16ExecutionError, match="authorization digest"):
        run_phase_worker(
            matrix_root=str(tmp_path / "missing_matrix"),
            output_root=str(tmp_path / "oot"),
            protocol_lock=str(PROTOCOL),
            phase="oot",
        )


def test_partial_artifact_is_rejected_and_complete_artifact_is_authenticated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cell"
    path.mkdir()
    write_json_atomic(path / "status.json", {"status": "complete"})
    identity = {"cell": 1, "authorization": "abc"}
    assert _load_final_sealed(path, identity) is None
    _seal_final_directory(path, identity)
    assert _load_final_sealed(path, identity) is not None
    write_text_atomic(path / "unexpected.txt", "not declared\n")
    with pytest.raises(Prompt16ExecutionError, match="inventory mismatch"):
        _load_final_sealed(path, identity)


def test_completed_artifact_identity_cannot_be_reused_for_changed_cell(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cell"
    path.mkdir()
    write_json_atomic(path / "status.json", {"status": "complete"})
    _seal_final_directory(path, {"cell": 1, "seed": 42})
    with pytest.raises(Prompt16ExecutionError, match="identity mismatch"):
        _load_final_sealed(path, {"cell": 1, "seed": 43})


def test_metric_and_prediction_alignment_reconciliation(tmp_path: Path) -> None:
    target = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    score = np.array([0.1, 0.8, 0.2, 0.7, 0.4, 0.6], dtype=float)
    threshold = 0.5
    predictions = pd.DataFrame(
        {
            "case_id": np.arange(1, 7),
            "target": target,
            "score": score,
            "decision_threshold": threshold,
            "predicted_default": (score >= threshold).astype("int8"),
        }
    )
    prediction_path = tmp_path / "predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    metrics = evaluate_model(target, score, threshold=threshold)
    ordered = predictions.sort_values("score", ascending=False, kind="mergesort")
    metrics["lift_at_10"] = float(ordered.head(1)["target"].mean() / target.mean())
    metrics["bad_rate_capture_at_10"] = float(ordered.head(1)["target"].sum() / target.sum())
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    from credit_risk_fs.experiments.prompt_16_third_dataset import (
        _locked_alignment_summary,
    )

    expected = _locked_alignment_summary(predictions["case_id"], target)
    expected["rows"] = expected.pop("row_count")
    result = _reconcile_prediction_metrics(
        prediction_path, metrics_path, expected_alignment=expected
    )
    assert result["maximum_absolute_metric_difference"] <= 1e-12
    metrics["auc"] += 1e-4
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(Prompt16ExecutionError, match="saved metric differs"):
        _reconcile_prediction_metrics(
            prediction_path, metrics_path, expected_alignment=expected
        )


def test_authorization_self_authentication_changes_on_any_field() -> None:
    payload = _self_authenticated_payload(
        {"schema_version": "x", "status": "frozen", "cells": 34}
    )
    claimed = payload.pop("artifact_authentication_sha256")
    assert claimed == canonical_sha256(payload)
    payload["cells"] = 35
    assert claimed != canonical_sha256(payload)


def test_cached_llm_reuse_has_no_ranking_generation_call() -> None:
    import credit_risk_fs.experiments.prompt_16_final_oot as final

    source = inspect.getsource(final.run_supplemental_oot_worker)
    stable_source = inspect.getsource(final._fit_full_dev_stable_states)
    assert "ensure_target_free_ranking" not in source + stable_source
    assert "rank_target_free" not in source + stable_source
    assert "fit_with_authenticated_ranking" in stable_source
    assert "llm_api_request_count" in inspect.getsource(final.run_final_oot_worker)


def test_full_dev_only_fit_and_threshold_boundaries_are_explicit() -> None:
    import credit_risk_fs.experiments.prompt_16_final_oot as final
    from credit_risk_fs.experiments.prompt_16_third_dataset import _fit_and_evaluate

    stable_source = inspect.getsource(final._fit_full_dev_stable_states)
    evaluation_source = inspect.getsource(_fit_and_evaluate)
    assert '"fit_scope": "full_dev_only"' in stable_source
    assert "validation_or_oot_used_for_fit" in stable_source
    assert "full_dev_training_ks_threshold" in evaluation_source
    assert "validation_or_oot_used_to_choose_threshold" in evaluation_source


def test_controller_has_process_tree_status_atomic_resume_and_marker_last() -> None:
    controller = (PROJECT_ROOT / "scripts/run_prompt_16_final_oot.py").read_text(
        encoding="utf-8"
    )
    monitor = inspect.getsource(
        __import__(
            "credit_risk_fs.experiments.resource_monitor", fromlist=["supervise_worker"]
        ).supervise_worker
    )
    assert "status_callback" in monitor
    assert "process_tree_rss_bytes" in monitor
    assert "write_json_atomic(status_path" in controller
    assert "RESUMING_FROM_CHECKPOINT" in controller
    assert "RESOURCE_RECOVERY_REQUIRED" in controller
    assert "_WORKER_SUCCESS" in controller
    assert controller.index('publish_status(state="DONE"') < controller.rindex(
        'output_root / "_SUCCESS"'
    )


def test_no_oot_plan_or_help_path_exists_in_final_cli() -> None:
    controller = (PROJECT_ROOT / "scripts/run_prompt_16_final_oot.py").read_text(
        encoding="utf-8"
    )
    assert "--authorization" in controller
    assert "--phase" not in controller
    assert "--cell" not in controller
    assert "--retry" not in controller
    assert "--threads" not in controller


def test_final_worker_detector_excludes_current_python_lineage() -> None:
    controller = (PROJECT_ROOT / "scripts/run_prompt_16_final_oot.py").read_text(
        encoding="utf-8"
    )
    assert "current_lineage_pids = {current_pid}" in controller
    assert "parent_pid in current_lineage_pids" in controller
    assert "pid in current_lineage_pids" in controller
    assert 'not name.startswith("python")' in controller


def test_memory_bounded_selected_projection_is_scientifically_equivalent() -> None:
    matrix = _protocol_payload(PROTOCOL)[1]["approved_protocol"][
        "method_and_evaluation_matrix"
    ]
    train = pd.DataFrame(
        {
            "case_id": np.arange(1, 41),
            "target": np.tile([0, 1], 20),
            "selected_numeric": np.linspace(-2.0, 2.0, 40),
            "selected_category": np.tile(["a", "b", None, "a"], 10),
            "unselected_wide_column": np.linspace(100.0, 200.0, 40),
        }
    )
    validation = pd.DataFrame(
        {
            "case_id": np.arange(101, 113),
            "target": np.tile([0, 1], 6),
            "selected_numeric": np.linspace(-1.5, 1.5, 12),
            "selected_category": np.tile(["b", "a", None], 4),
            "unselected_wide_column": np.linspace(300.0, 400.0, 12),
        }
    )
    selected = ["selected_numeric", "selected_category"]
    predictors = [*selected, "unselected_wide_column"]
    cell = {"model": "lr"}
    full_predictions, full_metrics, full_details = _fit_and_evaluate(
        cell=cell,
        selected=selected,
        train=train,
        validation=validation,
        predictors=predictors,
        matrix=matrix,
        phase="oot",
        frozen_threshold=None,
        full_dev_training_ks_threshold=True,
    )
    projected_predictions, projected_metrics, projected_details = _fit_and_evaluate(
        cell=cell,
        selected=selected,
        train=train[["case_id", "target", *selected]].copy(),
        validation=validation[["case_id", "target", *selected]].copy(),
        predictors=predictors,
        matrix=matrix,
        phase="oot",
        frozen_threshold=None,
        full_dev_training_ks_threshold=True,
    )
    pd.testing.assert_frame_equal(full_predictions, projected_predictions)
    assert full_metrics == projected_metrics
    assert full_details["configuration"] == projected_details["configuration"]


def test_memory_amendment_uses_only_authenticated_dev_resource_conclusions() -> None:
    fold5 = PROJECT_ROOT / (
        "results/prompt_16_homecredit_model_stability_2024/dev_v1/"
        "fold_5/selection_fits"
    )
    for fit_id in INHERITED_RESOURCE_INFEASIBLE_FIT_IDS:
        selection = json.loads((fold5 / fit_id / "selection.json").read_text())
        assert selection["status"] == "resource_infeasible"
    for order in INHERITED_RESOURCE_INFEASIBLE_CELL_ORDERS:
        for fold in range(1, 6):
            path = PROJECT_ROOT / (
                "results/prompt_16_homecredit_model_stability_2024/dev_v1/"
                f"fold_{fold}/evaluations/cell_{order:03d}/status.json"
            )
            status = json.loads(path.read_text())
            assert status["status"] == "unavailable"
            assert status["reason"] == "resource_infeasible"


def test_memory_amendment_stages_wide_data_and_migrates_only_exact_predecessor() -> None:
    import credit_risk_fs.experiments.prompt_16_final_oot as final
    import credit_risk_fs.experiments.prompt_16_third_dataset as third

    phase_source = inspect.getsource(third.run_phase_worker)
    final_source = inspect.getsource(final.run_final_oot_worker)
    controller = (PROJECT_ROOT / "scripts/run_prompt_16_final_oot.py").read_text()
    assert "full_dev_identity_authentication" in phase_source
    assert "locked_oot_identity_authentication" in phase_source
    assert "batch_size = 128" in phase_source
    assert "predictors=selected" in phase_source
    assert "memory_bounded_oot=True" in final_source
    assert "_authenticate_predecessor_partial_state" in controller
    assert "predecessor partial OOT inventory changed" in controller


def test_freeze_location_and_prompt14_preservation_are_explicit() -> None:
    import credit_risk_fs.experiments.prompt_16_final_oot as final

    source = inspect.getsource(final.build_freeze)
    assert FREEZE_RELATIVE_ROOT.as_posix() == "cleanup/audits/prompt_16_final_amended_oot"
    assert "prompt_14_two_dataset_oot_review_v3" in source
    assert "numeric_contents_opened_for_prompt16_freeze" in source
    assert "unavailable_due_to_unresolved_historical_provenance" in source


def test_real_170_identity_dev_authentication_and_metric_reconciliation() -> None:
    from credit_risk_fs.experiments.prompt_16_final_oot import (
        authenticate_complete_dev,
    )

    result = authenticate_complete_dev(PROJECT_ROOT)
    accounting = result["accounting"]
    assert accounting["registered_evaluation_identities"] == 170
    assert accounting["authenticated_evaluation_identities"] == 170
    assert accounting["completed_numeric_outcomes"] == 123
    assert accounting["frozen_visible_unavailable_outcomes"] == 47
    assert accounting["metric_reconciliations"] == 123
    assert result["classical_tree"]["tree_manifest_sha256"] == (
        "c956db2b2bb810805a1668916bf12c56f96ed994b03cd2a5c4acfde6fc6bd6ba"
    )
