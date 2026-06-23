from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_clip_v2_pipeline as pipeline


def test_pipeline_stage_order_and_eight_individual_evaluations():
    names = [stage["name"] for stage in pipeline.stages()]

    assert names == [
        "preflight",
        "statistical_view",
        "contrastive_data",
        "training_smoke",
        "training",
        "selector_integration",
        "downstream_evaluation",
        "aggregate_rebuild",
        "final_analysis",
        "tests",
        "final_audit",
    ]
    evaluation = next(stage for stage in pipeline.stages() if stage["name"] == "downstream_evaluation")
    assert len(evaluation["commands"]) == 8
    assert all("--execute" in command for command in evaluation["commands"])
    assert all("--all" not in command for command in evaluation["commands"])


def test_plan_mode_is_read_only_and_execution_requires_execute(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OUTPUT_ROOT", tmp_path / "clip_v2")
    monkeypatch.setattr(pipeline, "STATE_PATH", tmp_path / "clip_v2" / "pipeline_state.json")
    monkeypatch.setattr(pipeline, "LOG_PATH", tmp_path / "clip_v2" / "pipeline_execution.log")
    monkeypatch.setattr(pipeline, "LOCK_PATH", tmp_path / "clip_v2" / ".pipeline.lock")

    args = SimpleNamespace(plan=True, status=False, execute=False, fresh_start=False, resume=False, from_stage=None, to_stage=None)
    payload = pipeline.plan_payload(pipeline.select_stages(args), args)

    assert payload["execute"] is False
    assert payload["real_work_requires_execute"] is True
    assert not (tmp_path / "clip_v2" / "pipeline_state.json").exists()


def test_resume_skips_complete_valid_stages(tmp_path, monkeypatch):
    root = tmp_path / "clip_v2"
    root.mkdir()
    state = {
        "created_at": "now",
        "stages": {
            "preflight": {"status": "complete_valid"},
            "statistical_view": {"status": "failed"},
        },
    }
    (root / "pipeline_state.json").write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.setattr(pipeline, "OUTPUT_ROOT", root)
    monkeypatch.setattr(pipeline, "STATE_PATH", root / "pipeline_state.json")

    selected = pipeline.select_stages(SimpleNamespace(resume=True, from_stage=None, to_stage=None))

    assert selected[0]["name"] == "statistical_view"
    assert "preflight" not in [stage["name"] for stage in selected]


def test_fresh_start_archives_partial_artifacts_and_preserves_source(tmp_path, monkeypatch):
    output = tmp_path / "results" / "clip_v2"
    archive = tmp_path / "results" / "clip_v2_archives"
    output.mkdir(parents=True)
    (output / "partial.txt").write_text("partial", encoding="utf-8")
    source = tmp_path / "src" / "credit_risk_fs" / "clip"
    source.mkdir(parents=True)
    (source / "keep.py").write_text("# keep", encoding="utf-8")

    monkeypatch.setattr(pipeline, "OUTPUT_ROOT", output)
    monkeypatch.setattr(pipeline, "ARCHIVE_ROOT", archive)
    monkeypatch.setattr(pipeline, "active_clip_v2_processes", lambda: [])
    monkeypatch.setattr(pipeline, "has_complete_audited_study", lambda: False)

    archive_path = pipeline.fresh_start_reset(execute=True)

    assert archive_path is not None
    assert (archive / Path(archive_path).name / "reset_manifest.json").exists()
    assert not (output / "partial.txt").exists()
    assert (output / "statistical_view").exists()
    assert (source / "keep.py").exists()


def test_active_process_prevents_fresh_start(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "OUTPUT_ROOT", tmp_path / "clip_v2")
    monkeypatch.setattr(pipeline, "active_clip_v2_processes", lambda: [{"pid": 123, "command": "build_clip_v2"}])

    with pytest.raises(RuntimeError, match="active CLIP-v2 process"):
        pipeline.fresh_start_reset(execute=True)


def test_active_process_detector_ignores_own_query_and_current_process_tree(monkeypatch):
    rows = [
        {
            "ProcessId": 10,
            "ParentProcessId": 0,
            "Name": "powershell.exe",
            "CommandLine": "powershell uv run python scripts/run_clip_v2_pipeline.py --fresh-start --execute",
        },
        {
            "ProcessId": 20,
            "ParentProcessId": 10,
            "Name": "python.exe",
            "CommandLine": "python scripts/run_clip_v2_pipeline.py --fresh-start --execute",
        },
        {
            "ProcessId": 30,
            "ParentProcessId": 20,
            "Name": "powershell.exe",
            "CommandLine": "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'clip_v2' }",
        },
        {
            "ProcessId": 40,
            "ParentProcessId": 10,
            "Name": "python.exe",
            "CommandLine": "python scripts/run_clip_v2_final_evaluation.py --dataset homecredit --execute",
        },
        {
            "ProcessId": 50,
            "ParentProcessId": 10,
            "Name": "python.exe",
            "CommandLine": "python scripts/run_clip_v2_pipeline.py --plan",
        },
    ]
    monkeypatch.setattr(pipeline.os, "getpid", lambda: 20)

    ignored = pipeline.current_process_tree_pids(rows)
    active = [row["ProcessId"] for row in rows if pipeline.is_active_clip_v2_process(row, ignored)]

    assert ignored == {10, 20}
    assert active == [40]


def test_valid_lock_prevents_concurrent_execution(tmp_path, monkeypatch):
    root = tmp_path / "clip_v2"
    root.mkdir()
    lock = root / ".pipeline.lock"
    lock.write_text(json.dumps({"pid": 123, "started_at": "now"}), encoding="utf-8")
    monkeypatch.setattr(pipeline, "OUTPUT_ROOT", root)
    monkeypatch.setattr(pipeline, "LOCK_PATH", lock)
    monkeypatch.setattr(pipeline, "is_pid_active", lambda pid: True)

    with pytest.raises(RuntimeError, match="active pipeline lock"):
        pipeline.acquire_lock()


def test_stale_lock_is_replaced(tmp_path, monkeypatch):
    root = tmp_path / "clip_v2"
    root.mkdir()
    lock = root / ".pipeline.lock"
    lock.write_text(json.dumps({"pid": 999999, "started_at": "old"}), encoding="utf-8")
    monkeypatch.setattr(pipeline, "OUTPUT_ROOT", root)
    monkeypatch.setattr(pipeline, "LOCK_PATH", lock)
    monkeypatch.setattr(pipeline, "is_pid_active", lambda pid: False)

    acquired = pipeline.acquire_lock()

    assert acquired["pid"] != 999999
    pipeline.release_lock(acquired)
