from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import credit_risk_fs.experiments.prompt_16_third_dataset as prompt16

from credit_risk_fs.experiments.prompt_16_third_dataset import (
    EXPECTED_PROTOCOL_FILE_SHA256,
    EXPECTED_PROTOCOL_INTERNAL_SHA256,
    EXPECTED_EXECUTION_PLAN_SHA256,
    PLAN_SCHEMA_VERSION,
    Prompt16ExecutionError,
    _locked_alignment_summary,
    _validate_scope_frame,
    _baseline_selector,
    _combination_selector,
    _read_date_slice,
    canonical_registry,
    execution_lock_path,
    load_execution_plan,
    record_phase_resource_infeasibility,
    selection_fit_registry,
    selector_wall_clock_stage,
)
from credit_risk_fs.experiments.resource_policy import load_execution_policy
from credit_risk_fs.experiments.ram_control import load_ram_control_policy
from credit_risk_fs.experiments.research_logging import _human_line


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json"


def test_canonical_registry_is_exact_27_fits_and_30_cells():
    registry = canonical_registry(LOCK)
    assert registry["protocol_file_sha256"] == EXPECTED_PROTOCOL_FILE_SHA256
    assert registry["protocol_internal_sha256"] == EXPECTED_PROTOCOL_INTERNAL_SHA256
    assert len(registry["matrix_cells"]) == 30
    assert len(registry["fit_registry"]) == 27
    assert [item["fit_order"] for item in registry["fit_registry"]] == list(range(1, 28))
    dependencies = [
        order
        for fit in registry["fit_registry"]
        for order in fit["dependent_configuration_orders"]
    ]
    assert sorted(dependencies) == list(range(1, 31))
    reused = [
        fit for fit in registry["fit_registry"] if len(fit["dependent_configuration_orders"]) > 1
    ]
    assert [fit["method_id"] for fit in reused] == ["iv_then_boruta"] * 3
    assert [fit["iv_pool_budget"] for fit in reused] == [100, 200, 300]


def test_fit_registry_rejects_noncanonical_cell_order():
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    matrix = payload["approved_protocol"]["method_and_evaluation_matrix"]
    changed = dict(matrix)
    changed["matrix_cells"] = list(reversed(matrix["matrix_cells"]))
    with pytest.raises(Prompt16ExecutionError, match="configuration order"):
        selection_fit_registry(changed)


def test_execution_plan_authenticates_protocol_and_status(tmp_path: Path):
    authorized = (
        ROOT
        / "cleanup/audits/prompt_16_final_third_dataset_execution/resource_pilot_plan.json"
    )
    plan = load_execution_plan(authorized)
    assert plan["schema_version"] == PLAN_SCHEMA_VERSION
    assert EXPECTED_EXECUTION_PLAN_SHA256 == "9fbad83000f6822e523b76add1b78f232a3d7e882bbf1e590216585b5745a31e"
    plan = dict(plan)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(Prompt16ExecutionError, match="digest mismatch"):
        load_execution_plan(path)


def test_prompt_16_resource_policy_is_strict_and_sequential():
    policy = load_execution_policy(
        ROOT, "configs/execution/prompt_16_homecredit_2024_v1.yaml"
    )
    assert policy.parallelism.concurrent_experiment_runs == 1
    assert policy.parallelism.concurrent_folds == 1
    assert policy.parallelism.data_loader_workers == 0
    assert policy.parallelism.estimator_threads == 4
    assert policy.memory.abort_process_tree_rss_gb == 24
    assert policy.memory.abort_if_system_available_below_gb == 8
    assert policy.disk.minimum_free_results_gb == 80
    ram_control = load_ram_control_policy(
        ROOT, "configs/execution/prompt_16_ram_wait_resume_v1.yaml"
    )
    assert ram_control.emergency_margin_bytes == 8 * 1024**3
    assert ram_control.recovery_threshold_bytes == 10 * 1024**3
    assert ram_control.log_interval_seconds == 300


def test_execution_lock_is_outside_atomic_output_root(tmp_path: Path):
    output = tmp_path / "matrix_v1"
    assert execution_lock_path(output) == tmp_path / ".matrix_v1.execution.lock"


def test_all_frozen_selectors_map_to_canonical_wall_clock_stages():
    registry = canonical_registry(LOCK)
    locked_limits = registry["resource_controls"]["wall_clock_limits_seconds"]
    observed = {
        selector_wall_clock_stage(str(fit["method_id"]))
        for fit in registry["fit_registry"]
    }
    assert observed.issubset(set(locked_limits))
    assert observed == set(locked_limits) - {"final_lr", "final_catboost"}


def test_every_registered_selector_constructs_from_frozen_settings():
    payload = json.loads(LOCK.read_text(encoding="utf-8"))
    matrix = payload["approved_protocol"]["method_and_evaluation_matrix"]
    for fit in selection_fit_registry(matrix):
        if fit["family"] == "canonical_baseline":
            selector = _baseline_selector(fit, matrix, "synthetic_training_only")
        else:
            selector = _combination_selector(fit, matrix, "synthetic_training_only")
        assert selector is not None


def test_matrix_date_slice_uses_typed_date_bounds(tmp_path: Path):
    matrix_dir = tmp_path / "matrix"
    matrix_dir.mkdir()
    table = pa.table(
        {
            "case_id": pa.array([1, 2, 3], type=pa.int64()),
            "date_decision": pa.array(
                ["2020-01-03", "2020-01-02", "2020-01-01"], type=pa.string()
            ).cast(pa.date32()),
            "MONTH": pa.array([202001, 202001, 202001], type=pa.int64()),
            "WEEK_NUM": pa.array([1, 1, 1], type=pa.int64()),
            "target": pa.array([0, 1, 0], type=pa.int8()),
            "x": pa.array([1.0, 2.0, 3.0], type=pa.float32()),
        }
    )
    part = matrix_dir / "part-00000.parquet"
    pq.write_table(table, part)
    manifest = {
        "artifacts": [{"path": "matrix/part-00000.parquet"}],
        "summary": {"matrix_part_count": 1},
    }
    observed = _read_date_slice(
        tmp_path,
        manifest,
        date_min="2020-01-02",
        date_max="2020-01-03",
        predictors=["x"],
        stop_event=None,
        stage_queue=None,
        stage="test",
        fold_label="test",
    )
    assert observed["case_id"].tolist() == [2, 1]
    assert pd.api.types.is_float_dtype(observed["x"])


def test_scope_authentication_uses_the_lock_frozen_line_serialization():
    frame = pd.DataFrame({"case_id": [1, 2], "target": [0, 1]})
    expected = {
        "rows": 2,
        "target_0": 1,
        "target_1": 1,
        "ordered_case_id_sha256": hashlib.sha256(b"1\n2\n").hexdigest(),
        "ordered_case_id_target_sha256": hashlib.sha256(
            b"1\x1f0\n2\x1f1\n"
        ).hexdigest(),
    }
    result = _validate_scope_frame(frame, expected, "synthetic")
    assert result["authenticated"] is True
    locked = _locked_alignment_summary([1, 2], [0, 1])
    assert locked["ordered_case_id_sha256"] == expected["ordered_case_id_sha256"]


def test_resource_limited_selector_is_sealed_and_visible(tmp_path: Path, monkeypatch):
    matrix_root = tmp_path / "matrix"
    matrix_root.mkdir()
    (matrix_root / "manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        prompt16,
        "_matrix_identity",
        lambda _root: ({"summary": {}}, {"predictor_columns": []}),
    )
    output_root = tmp_path / "pilot" / "fold_1"
    result = record_phase_resource_infeasibility(
        matrix_root=matrix_root,
        output_root=output_root,
        protocol_lock=LOCK,
        phase="pilot",
        fold_id=1,
        stopped_stage="baseline_lightweight",
        stopped_scope="pilot:1:fit_001",
        supervisor_evidence={
            "stop_code": "wall_clock_limit",
            "active_computation_seconds": 10_800,
        },
    )
    assert result is not None and result["kind"] == "selection_fit"
    selection_root = output_root / "selection_fits" / "fit_001"
    assert (selection_root / "_SUCCESS").is_file()
    selection = json.loads((selection_root / "selection.json").read_text())
    assert selection["status"] == "resource_infeasible"
    assert selection["selected_features"] == []


def test_prompt_16_cli_import_is_inert(tmp_path: Path):
    before = list(tmp_path.iterdir())
    cli = ROOT / "scripts/run_prompt_16_third_dataset.py"
    spec = importlib.util.spec_from_file_location("prompt16_cli_import_test", cli)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert list(tmp_path.iterdir()) == before


def test_prompt_16_supervised_cli_mirrors_30_second_heartbeats_to_terminal():
    cli = ROOT / "scripts/run_prompt_16_third_dataset.py"
    tree = ast.parse(cli.read_text(encoding="utf-8"))
    session_contexts = [
        item.context_expr
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        for item in node.items
        if isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Name)
        and item.context_expr.func.id == "ResearchLogSession"
    ]
    assert len(session_contexts) == 1
    supervisor_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "supervise_worker"
    ]
    assert len(supervisor_calls) == 1
    keywords = {item.arg: item.value for item in supervisor_calls[0].keywords}
    assert ast.literal_eval(keywords["heartbeat_interval_seconds"]) == 30.0
    terminal_events = {
        ast.literal_eval(node.args[0])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "emit_research_event"
        and node.args
    }
    assert terminal_events == {
        "session_completed",
        "session_controlled_stop",
        "session_failed",
    }


def test_prompt_16_heartbeat_human_format_includes_stage_time_and_ram():
    rendered = _human_line(
        {
            "event": "stage_heartbeat",
            "level": "INFO",
            "timestamp_utc": "2026-08-08T20:00:00+00:00",
            "stage": "baseline_boruta_random_forest",
            "component": "boruta_random_forest_selection",
            "elapsed_stage_seconds": 1832,
            "worker_rss_bytes": 7 * 1024**3,
            "system_available_ram_bytes": 23 * 1024**3,
        },
        debug_log_path="logs/debug.log",
    )
    assert rendered == (
        "[2026-08-08 20:00:00 UTC] ACTIVE | "
        "Boruta Random Forest Selection running | Elapsed 30m 32s | "
        "RAM 7.0 GiB | Available 23.0 GiB"
    )
