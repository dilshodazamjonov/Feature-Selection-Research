from __future__ import annotations

from pathlib import Path

import pytest

from credit_risk_fs.pipelines import reverse_transfer
from credit_risk_fs.pipelines.reverse_transfer import (
    TransferStageError,
    _configuration_hash,
    _relative,
    _stage_artifact_paths,
    _validate_stage_manifest,
    _validate_unique_stage_artifact_ownership,
    execute_plan,
    load_config_dir,
)
from credit_risk_fs.utils.hashing import sha256_file
from credit_risk_fs.utils.io import write_json


SEEDS = (11, 22, 33, 44, 55)
MODELS = ("lr", "catboost")


def _complete_manifest(
    *,
    stage: str,
    artifacts: list[Path],
    config_hash: str,
    status: str = "complete",
) -> dict[str, object]:
    return {
        "stage": stage,
        "status": status,
        "configuration_hash": config_hash,
        "source_dataset": reverse_transfer.SOURCE_DATASET,
        "external_dataset": reverse_transfer.EXTERNAL_DATASET,
        "pairing_policy_version": reverse_transfer.PAIRING_POLICY_VERSION,
        "requested_seeds": list(SEEDS),
        "requested_models": list(MODELS),
        "artifact_hashes": {
            _relative(path): sha256_file(path) for path in artifacts
        },
    }


def _synthetic_handoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, object], Path, dict[str, Path]]:
    output = tmp_path / "out"
    artifacts = {
        "prepare": output / "prepare-owned.txt",
        "train": output / "train-owned.txt",
    }
    artifacts["prepare"].parent.mkdir(parents=True)
    artifacts["prepare"].write_text("authenticated prepare", encoding="utf-8")
    config = load_config_dir("configs/corrected_lendingclub_to_homecredit")
    manifest = _complete_manifest(
        stage="prepare",
        artifacts=[artifacts["prepare"]],
        config_hash=_configuration_hash(config),
    )
    write_json(
        output / "manifests/prepare_stage_manifest.json",
        manifest,
    )
    monkeypatch.setattr(reverse_transfer, "DEFAULT_OUTPUT_ROOT", output)
    monkeypatch.setattr(
        reverse_transfer,
        "_stage_artifact_paths",
        lambda stage, *args, **kwargs: [artifacts[stage]],
    )
    return config, output, artifacts


def test_prepare_anchor_intermediates_are_declared_and_train_outputs_are_distinct(
    tmp_path: Path,
) -> None:
    prepare = _stage_artifact_paths("prepare", tmp_path, SEEDS, MODELS)
    train = _stage_artifact_paths("train", tmp_path, SEEDS, MODELS)
    selection = tmp_path / "source_anchor/source_anchor_selection_manifest.json"
    template = tmp_path / "source_anchor/seed_anchor_manifest_template.csv"
    assert selection in prepare
    assert template in prepare
    assert selection not in train
    assert template not in train
    assert tmp_path / "source_anchor/source_anchor_manifest.json" in train
    assert tmp_path / "source_anchor/seed_anchor_manifest.csv" in train


def test_train_accepts_hash_authenticated_prepare_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "prepare-anchor.json"
    artifact.write_text('{"source_dataset":"lendingclub_v2"}', encoding="utf-8")
    manifest = _complete_manifest(
        stage="prepare", artifacts=[artifact], config_hash="config"
    )
    _validate_stage_manifest(
        stage="prepare",
        manifest=manifest,
        output_dir=tmp_path,
        config_hash="config",
        seeds=SEEDS,
        models=MODELS,
        artifact_paths=[artifact],
    )


def test_train_rejects_unmanifested_preexisting_train_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output, artifacts = _synthetic_handoff(tmp_path, monkeypatch)
    artifacts["train"].write_text("partial", encoding="utf-8")
    with pytest.raises(TransferStageError, match="unmanifested partial outputs"):
        execute_plan(
            config=config,
            stages=("train",),
            seeds=SEEDS,
            models=MODELS,
            output_dir=output,
            resume=False,
            skip_existing=False,
        )


def test_modified_prepare_anchor_is_rejected_by_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "prepare-anchor.json"
    artifact.write_text('{"valid":true}', encoding="utf-8")
    manifest = _complete_manifest(
        stage="prepare", artifacts=[artifact], config_hash="config"
    )
    artifact.write_text('{"valid":false}', encoding="utf-8")
    with pytest.raises(TransferStageError, match="missing or corrupt"):
        _validate_stage_manifest(
            stage="prepare",
            manifest=manifest,
            output_dir=tmp_path,
            config_hash="config",
            seeds=SEEDS,
            models=MODELS,
            artifact_paths=[artifact],
        )


def test_manifest_path_normalization_mismatch_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "prepare-anchor.json"
    artifact.write_text("valid", encoding="utf-8")
    manifest = _complete_manifest(
        stage="prepare", artifacts=[artifact], config_hash="config"
    )
    key = next(iter(manifest["artifact_hashes"]))
    manifest["artifact_hashes"] = {
        str(key).replace("/", "\\"): sha256_file(artifact)
    }
    with pytest.raises(TransferStageError, match="path normalization mismatch"):
        _validate_stage_manifest(
            stage="prepare",
            manifest=manifest,
            output_dir=tmp_path,
            config_hash="config",
            seeds=SEEDS,
            models=MODELS,
            artifact_paths=[artifact],
        )


def test_train_rejects_missing_prepare_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output, _ = _synthetic_handoff(tmp_path, monkeypatch)
    (output / "manifests/prepare_stage_manifest.json").unlink()
    with pytest.raises(TransferStageError, match="prepare has no manifest"):
        execute_plan(
            config=config,
            stages=("train",),
            seeds=SEEDS,
            models=MODELS,
            output_dir=output,
            resume=False,
            skip_existing=False,
        )


def test_train_rejects_incomplete_prepare_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output, artifacts = _synthetic_handoff(tmp_path, monkeypatch)
    manifest = _complete_manifest(
        stage="prepare",
        artifacts=[artifacts["prepare"]],
        config_hash=_configuration_hash(config),
        status="in_progress",
    )
    write_json(output / "manifests/prepare_stage_manifest.json", manifest)
    with pytest.raises(TransferStageError, match="prepare: stage is not complete"):
        execute_plan(
            config=config,
            stages=("train",),
            seeds=SEEDS,
            models=MODELS,
            output_dir=output,
            resume=False,
            skip_existing=False,
        )


def test_duplicate_cross_stage_artifact_declaration_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    duplicate = tmp_path / "same.txt"
    monkeypatch.setattr(
        reverse_transfer,
        "_stage_artifact_paths",
        lambda stage, *args, **kwargs: [duplicate],
    )
    with pytest.raises(TransferStageError, match="duplicate artifact declaration"):
        _validate_unique_stage_artifact_ownership(
            stages=("train",),
            output_dir=tmp_path,
            seeds=SEEDS,
            models=MODELS,
        )


def test_current_valid_prepare_to_train_handoff_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output, artifacts = _synthetic_handoff(tmp_path, monkeypatch)

    def fake_train(**kwargs: object) -> dict[str, object]:
        artifacts["train"].write_text("new train output", encoding="utf-8")
        return {"model_fitting_started": False}

    monkeypatch.setattr(reverse_transfer, "_train", fake_train)
    execute_plan(
        config=config,
        stages=("train",),
        seeds=SEEDS,
        models=MODELS,
        output_dir=output,
        resume=False,
        skip_existing=False,
    )
    assert artifacts["train"].exists()
    assert (
        reverse_transfer.read_json(
            output / "manifests/train_stage_manifest.json"
        )["status"]
        == "complete"
    )


def test_declared_prepare_input_tampering_remains_fail_closed(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "prepare-anchor.json"
    source = tmp_path / "source.csv"
    artifact.write_text("valid", encoding="utf-8")
    source.write_text("a\n1\n", encoding="utf-8")
    manifest = _complete_manifest(
        stage="prepare", artifacts=[artifact], config_hash="config"
    )
    manifest["input_hashes"] = {str(source): sha256_file(source)}
    source.write_text("a\n2\n", encoding="utf-8")
    with pytest.raises(TransferStageError, match="input changed or missing"):
        _validate_stage_manifest(
            stage="prepare",
            manifest=manifest,
            output_dir=tmp_path,
            config_hash="config",
            seeds=SEEDS,
            models=MODELS,
            artifact_paths=[artifact],
        )
