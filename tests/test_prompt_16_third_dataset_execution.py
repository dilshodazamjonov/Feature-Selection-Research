from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def _load_prompt_16_cli():
    cli = ROOT / "scripts/run_prompt_16_third_dataset.py"
    spec = importlib.util.spec_from_file_location("prompt16_cli_test", cli)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert policy.memory.abort_if_system_available_below_gb == 1
    assert policy.disk.minimum_free_results_gb == 80
    ram_control = load_ram_control_policy(
        ROOT, "configs/execution/prompt_16_ram_wait_resume_v1.yaml"
    )
    assert ram_control.emergency_margin_bytes == 1 * 1024**3
    assert ram_control.recovery_threshold_bytes == 2 * 1024**3
    assert ram_control.log_interval_seconds == 300


def test_selector_fit_releases_wide_phase_frames_before_opaque_work(
    tmp_path: Path, monkeypatch
):
    frame_refs: dict[str, weakref.ReferenceType[pd.DataFrame]] = {}

    def fake_load_phase_frames(**_kwargs):
        train = pd.DataFrame(
            {
                "case_id": [1, 2, 3],
                "target": pd.Series([0, 1, 0], dtype="int8"),
                "x": pd.Series([1.0, 2.0, 3.0], dtype="float32"),
            }
        )
        validation = pd.DataFrame(
            {
                "case_id": [4],
                "target": pd.Series([0], dtype="int8"),
                "x": pd.Series([4.0], dtype="float32"),
            }
        )
        frame_refs["train"] = weakref.ref(train)
        frame_refs["validation"] = weakref.ref(validation)
        return train, validation, ["x"], {"matrix_manifest_sha256": "a" * 64}

    fit = {
        "fit_id": "fit_001",
        "fit_order": 1,
        "method_id": "lasso_l1_logistic",
        "family": "canonical_baseline",
        "dependent_configuration_orders": [],
    }
    protocol = {
        "approved_protocol": {
            "method_and_evaluation_matrix": {"matrix_cells": []}
        }
    }
    monkeypatch.setattr(
        prompt16,
        "_protocol_payload",
        lambda _path: (
            SimpleNamespace(
                lock_file_sha256=prompt16.EXPECTED_PROTOCOL_FILE_SHA256,
                lock_internal_sha256=prompt16.EXPECTED_PROTOCOL_INTERNAL_SHA256,
            ),
            protocol,
        ),
    )
    monkeypatch.setattr(prompt16, "_load_phase_frames", fake_load_phase_frames)
    monkeypatch.setattr(prompt16, "selection_fit_registry", lambda _matrix: [fit])

    def observe_released_frames(**kwargs):
        assert frame_refs["train"]() is None
        assert frame_refs["validation"]() is None
        assert kwargs["numeric_train"].dtypes.tolist() == [np.dtype("float32")]
        assert kwargs["y_train"].dtype == np.dtype("int64")
        raise KeyboardInterrupt("test stop after lifecycle assertion")

    monkeypatch.setattr(prompt16, "_fit_one_selection", observe_released_frames)
    with pytest.raises(KeyboardInterrupt, match="lifecycle assertion"):
        prompt16.run_phase_worker(
            matrix_root=str(tmp_path / "matrix"),
            output_root=str(tmp_path / "fold_1"),
            protocol_lock=str(tmp_path / "protocol.json"),
            phase="pilot",
            fold_id=1,
        )


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
    _load_prompt_16_cli()
    assert list(tmp_path.iterdir()) == before


def test_prompt_16_cli_automatically_retries_a_new_authenticated_resource_seal():
    cli = _load_prompt_16_cli()
    resource_stop = SimpleNamespace(
        status="aborted_resource_limit",
        stop_code="ram_process_limit",
        child_cleanup_confirmed=True,
        queue_cleanup_confirmed=True,
        survivor_processes=(),
    )
    completed = SimpleNamespace(
        status="completed",
        stop_code=None,
        child_cleanup_confirmed=True,
        queue_cleanup_confirmed=True,
        survivor_processes=(),
    )
    results = iter((resource_stop, completed))
    attempts: list[SimpleNamespace] = []
    announcements: list[tuple[int, str, str]] = []

    def supervise_attempt():
        result = next(results)
        attempts.append(result)
        return result

    def seal_resource_stop(_result):
        return {
            "kind": "evaluation",
            "id": "cell_001",
            "status": "complete",
            "manifest_sha256": "a" * 64,
        }

    def announce_retry(retry_count, _result, sealed):
        announcements.append((retry_count, sealed["kind"], sealed["id"]))

    result, current_seal, sealed = cli._run_with_automatic_resource_retries(
        supervise_attempt=supervise_attempt,
        seal_resource_stop=seal_resource_stop,
        announce_retry=announce_retry,
    )

    assert result is completed
    assert current_seal is None
    assert len(attempts) == 2
    assert announcements == [(1, "evaluation", "cell_001")]
    assert [item["id"] for item in sealed] == ["cell_001"]


@pytest.mark.parametrize(
    ("status", "stop_code", "child_cleanup", "queue_cleanup", "survivors"),
    [
        ("failed", "worker_crash", True, True, ()),
        ("interrupted", "manual_interrupt", True, True, ()),
        ("aborted_resource_limit", "ram_process_limit", False, True, ()),
        ("aborted_resource_limit", "ram_process_limit", True, False, ()),
        ("aborted_resource_limit", "ram_process_limit", True, True, (123,)),
    ],
)
def test_prompt_16_cli_does_not_retry_unsafe_or_nonresource_stops(
    status, stop_code, child_cleanup, queue_cleanup, survivors
):
    cli = _load_prompt_16_cli()
    stopped = SimpleNamespace(
        status=status,
        stop_code=stop_code,
        child_cleanup_confirmed=child_cleanup,
        queue_cleanup_confirmed=queue_cleanup,
        survivor_processes=survivors,
    )
    attempts = 0
    seal_calls = 0

    def supervise_attempt():
        nonlocal attempts
        attempts += 1
        return stopped

    def seal_resource_stop(_result):
        nonlocal seal_calls
        seal_calls += 1
        return {
            "kind": "evaluation",
            "id": "cell_001",
            "status": "complete",
            "manifest_sha256": "a" * 64,
        }

    result, current_seal, sealed = cli._run_with_automatic_resource_retries(
        supervise_attempt=supervise_attempt,
        seal_resource_stop=seal_resource_stop,
        announce_retry=lambda *_args: pytest.fail("unsafe retry announced"),
    )

    assert result is stopped
    assert attempts == 1
    assert seal_calls == (
        1
        if status == "aborted_resource_limit"
        and child_cleanup
        and queue_cleanup
        and not survivors
        else 0
    )
    assert current_seal is None
    assert sealed == ()


def test_prompt_16_cli_does_not_retry_without_a_new_complete_seal():
    cli = _load_prompt_16_cli()
    stopped = SimpleNamespace(
        status="timed_out",
        stop_code="wall_clock_limit",
        child_cleanup_confirmed=True,
        queue_cleanup_confirmed=True,
        survivor_processes=(),
    )
    attempts = 0

    def supervise_attempt():
        nonlocal attempts
        attempts += 1
        return stopped

    result, current_seal, sealed = cli._run_with_automatic_resource_retries(
        supervise_attempt=supervise_attempt,
        seal_resource_stop=lambda _result: None,
        announce_retry=lambda *_args: pytest.fail("unsealed retry announced"),
    )

    assert result is stopped
    assert attempts == 1
    assert current_seal is None
    assert sealed == ()


def test_prompt_16_cli_stops_if_a_retry_reports_the_same_sealed_scope():
    cli = _load_prompt_16_cli()
    stopped = SimpleNamespace(
        status="aborted_resource_limit",
        stop_code="ram_process_limit",
        child_cleanup_confirmed=True,
        queue_cleanup_confirmed=True,
        survivor_processes=(),
    )
    attempts = 0
    announcements: list[int] = []

    def supervise_attempt():
        nonlocal attempts
        attempts += 1
        return stopped

    result, current_seal, sealed = cli._run_with_automatic_resource_retries(
        supervise_attempt=supervise_attempt,
        seal_resource_stop=lambda _result: {
            "kind": "evaluation",
            "id": "cell_001",
            "status": "complete",
            "manifest_sha256": "a" * 64,
        },
        announce_retry=lambda retry_count, *_args: announcements.append(retry_count),
    )

    assert result is stopped
    assert attempts == 2
    assert current_seal is not None and current_seal["id"] == "cell_001"
    assert announcements == [1]
    assert [item["id"] for item in sealed] == ["cell_001"]


def test_prompt_16_cli_honors_the_automatic_retry_bound():
    cli = _load_prompt_16_cli()
    stopped = SimpleNamespace(
        status="timed_out",
        stop_code="wall_clock_limit",
        child_cleanup_confirmed=True,
        queue_cleanup_confirmed=True,
        survivor_processes=(),
    )
    attempts = 0
    seals = 0

    def supervise_attempt():
        nonlocal attempts
        attempts += 1
        return stopped

    def seal_resource_stop(_result):
        nonlocal seals
        seals += 1
        return {
            "kind": "selection_fit",
            "id": f"fit_{seals:03d}",
            "status": "complete",
            "manifest_sha256": f"{seals:064x}",
        }

    result, current_seal, sealed = cli._run_with_automatic_resource_retries(
        supervise_attempt=supervise_attempt,
        seal_resource_stop=seal_resource_stop,
        announce_retry=lambda *_args: None,
        maximum_retries=2,
    )

    assert result is stopped
    assert attempts == 3
    assert current_seal is not None and current_seal["id"] == "fit_003"
    assert [item["id"] for item in sealed] == ["fit_001", "fit_002"]


def test_prompt_16_supervised_cli_mirrors_30_second_heartbeats_to_terminal():
    cli = ROOT / "scripts/run_prompt_16_third_dataset.py"
    tree = ast.parse(cli.read_text(encoding="utf-8"))
    session_contexts = [
        item
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        for item in node.items
        if isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Name)
        and item.context_expr.func.id == "ResearchLogSession"
    ]
    assert len(session_contexts) == 1
    assert isinstance(session_contexts[0].optional_vars, ast.Name)
    assert session_contexts[0].optional_vars.id == "session"
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
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "session"
        and node.func.attr == "finish"
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
