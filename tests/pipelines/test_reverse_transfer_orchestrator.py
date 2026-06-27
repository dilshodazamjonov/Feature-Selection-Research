from __future__ import annotations

from pathlib import Path

import pytest

from credit_risk_fs.pipelines.reverse_transfer import (
    TransferStageError,
    build_parser,
    execute_plan,
    load_config_dir,
    run_cli,
)


def test_parser_exposes_only_reverse_transfer_stages() -> None:
    parser = build_parser()
    args = parser.parse_args(["--stage", "project", "--dry-run"])
    assert args.stage == "project"
    assert args.dry_run


def test_dry_run_does_not_create_output(tmp_path, capsys) -> None:
    output = tmp_path / "scientific-output"
    result = run_cli(
        [
            "--stage",
            "all",
            "--config-dir",
            "configs/corrected_lendingclub_to_homecredit",
            "--output-dir",
            str(output),
            "--dry-run",
        ]
    )
    assert result == 0
    assert not output.exists()
    captured = capsys.readouterr().out
    assert '"resolved_stages"' in captured
    assert '"baseline_execution_allowed": false' in captured


def test_orchestrator_source_never_references_deleted_matrix() -> None:
    source = Path(
        "src/credit_risk_fs/pipelines/reverse_transfer.py"
    ).read_text(encoding="utf-8")
    assert "run_clip_final_comparison.py" not in source
    assert "clip_final_comparison" not in source
    assert "run_corrected_homecredit_clip_pipelines" not in source
    assert "analyze_corrected_homecredit_clip" not in source


def test_dry_run_displays_resolved_anchor_contract(tmp_path, capsys) -> None:
    result = run_cli(
        [
            "--stage", "all",
            "--config-dir",
            "configs/corrected_lendingclub_to_homecredit",
            "--output-dir",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert result == 0
    output = capsys.readouterr().out
    assert '"source_dataset": "lendingclub_v2"' in output
    assert '"subwindow_boundaries"' in output
    assert "-1612.5" in output
    assert '"max_adjacent_window_psi": 0.1' in output
    assert '"member_count": 23' in output
    assert '"fail_closed_conditions"' in output


def test_stage_filter_skip_existing_and_overwrite_refusal(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_prepare(**kwargs):
        calls.append("prepare")
        return {"synthetic": True}

    monkeypatch.setattr(
        "credit_risk_fs.pipelines.reverse_transfer._prepare", fake_prepare
    )
    config = load_config_dir("configs/corrected_lendingclub_to_homecredit")
    kwargs = {
        "config": config,
        "stages": ("prepare",),
        "seeds": (11, 22, 33, 44, 55),
        "models": ("lr", "catboost"),
        "output_dir": tmp_path / "out",
        "resume": False,
    }
    execute_plan(**kwargs, skip_existing=False)
    assert calls == ["prepare"]
    execute_plan(**kwargs, skip_existing=True)
    assert calls == ["prepare"]
    with pytest.raises(TransferStageError, match="completed output exists"):
        execute_plan(**kwargs, skip_existing=False)
