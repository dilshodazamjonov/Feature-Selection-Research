from __future__ import annotations

from pathlib import Path

import pytest

from credit_risk_fs.pipelines.reverse_transfer import (
    TransferStageError,
    build_parser,
    execute_plan,
    load_config_dir,
    run_cli,
    resolve_plan,
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


def test_reverse_orchestrator_rejects_swapped_roles() -> None:
    config = load_config_dir("configs/corrected_lendingclub_to_homecredit")
    config["training_dataset"] = "homecredit"
    config["external_dataset"] = "lendingclub_v2"
    with pytest.raises(ValueError, match="requires training_dataset"):
        resolve_plan(
            config=config,
            stages=("prepare",),
            seeds=(11, 22, 33, 44, 55),
            models=("lr", "catboost"),
            output_dir=Path("unused"),
        )


def test_stage_filter_skip_existing_and_overwrite_refusal(tmp_path, monkeypatch) -> None:
    calls = []
    marker = tmp_path / "out" / "synthetic-artifact.txt"

    def fake_prepare(**kwargs):
        calls.append("prepare")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("complete", encoding="utf-8")
        return {"synthetic": True}

    monkeypatch.setattr(
        "credit_risk_fs.pipelines.reverse_transfer._prepare", fake_prepare
    )
    monkeypatch.setattr(
        "credit_risk_fs.pipelines.reverse_transfer._stage_artifact_paths",
        lambda *args, **kwargs: [marker],
    )
    monkeypatch.setattr(
        "credit_risk_fs.pipelines.reverse_transfer.DEFAULT_OUTPUT_ROOT",
        tmp_path / "out",
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
    execute_plan(**{**kwargs, "resume": True}, skip_existing=False)
    assert calls == ["prepare"]
    with pytest.raises(TransferStageError, match="completed output exists"):
        execute_plan(**kwargs, skip_existing=False)
    marker.write_text("corrupt", encoding="utf-8")
    with pytest.raises(TransferStageError, match="missing or corrupt"):
        execute_plan(**kwargs, skip_existing=True)


def test_skip_existing_revalidates_all_upstream_artifacts(tmp_path, monkeypatch) -> None:
    output = tmp_path / "out"
    artifacts = {
        "prepare": output / "prepare.txt",
        "train": output / "train.txt",
    }

    def fake_prepare(**kwargs):
        artifacts["prepare"].parent.mkdir(parents=True, exist_ok=True)
        artifacts["prepare"].write_text("prepare", encoding="utf-8")
        return {}

    def fake_train(**kwargs):
        artifacts["train"].write_text("train", encoding="utf-8")
        return {}

    monkeypatch.setattr(
        "credit_risk_fs.pipelines.reverse_transfer._prepare", fake_prepare
    )
    monkeypatch.setattr(
        "credit_risk_fs.pipelines.reverse_transfer._train", fake_train
    )
    monkeypatch.setattr(
        "credit_risk_fs.pipelines.reverse_transfer._stage_artifact_paths",
        lambda stage, *args, **kwargs: [artifacts[stage]],
    )
    monkeypatch.setattr(
        "credit_risk_fs.pipelines.reverse_transfer.DEFAULT_OUTPUT_ROOT", output
    )
    config = load_config_dir("configs/corrected_lendingclub_to_homecredit")
    common = {
        "config": config,
        "seeds": (11, 22, 33, 44, 55),
        "models": ("lr", "catboost"),
        "output_dir": output,
        "resume": False,
        "skip_existing": False,
    }
    execute_plan(**common, stages=("prepare",))
    execute_plan(**common, stages=("train",))
    artifacts["prepare"].write_text("tampered", encoding="utf-8")
    with pytest.raises(TransferStageError, match="prepare: artifact missing or corrupt"):
        execute_plan(
            **{**common, "stages": ("train",), "skip_existing": True}
        )


def test_skip_existing_revalidates_declared_current_inputs(tmp_path, monkeypatch) -> None:
    output = tmp_path / "out"
    artifact = output / "prepare.txt"
    source = tmp_path / "source.csv"
    source.write_text("a\n1\n", encoding="utf-8")

    def fake_prepare(**kwargs):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("prepare", encoding="utf-8")
        from credit_risk_fs.utils.hashing import sha256_file

        return {"input_hashes": {str(source): sha256_file(source)}}

    monkeypatch.setattr(
        "credit_risk_fs.pipelines.reverse_transfer._prepare", fake_prepare
    )
    monkeypatch.setattr(
        "credit_risk_fs.pipelines.reverse_transfer._stage_artifact_paths",
        lambda *args, **kwargs: [artifact],
    )
    monkeypatch.setattr(
        "credit_risk_fs.pipelines.reverse_transfer.DEFAULT_OUTPUT_ROOT", output
    )
    config = load_config_dir("configs/corrected_lendingclub_to_homecredit")
    kwargs = {
        "config": config,
        "stages": ("prepare",),
        "seeds": (11, 22, 33, 44, 55),
        "models": ("lr", "catboost"),
        "output_dir": output,
        "resume": False,
    }
    execute_plan(**kwargs, skip_existing=False)
    source.write_text("a\n2\n", encoding="utf-8")
    with pytest.raises(TransferStageError, match="input changed or missing"):
        execute_plan(**kwargs, skip_existing=True)
