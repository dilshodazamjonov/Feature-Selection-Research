from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from credit_risk_fs.experiments.atomic_io import atomic_publish, inspect_artifact, write_csv_atomic, write_json_atomic
from credit_risk_fs.experiments.checkpointing import (
    CheckpointManager,
    ResumeValidationError,
    resolve_resume_target,
)


def _identity(**updates):
    identity = {
        "run_id": "run-1",
        "dataset": "synthetic",
        "selector": "none",
        "model": "lr",
        "split_protocol": "synthetic_time",
        "seed": 42,
        "budgets": {"lr": 2},
        "resolved_config_hash": "config-a",
        "protocol_version": "1.1.0",
        "protocol_hash": "protocol-a",
        "data_hash": "data-a",
        "row_alignment_hash": "rows-a",
        "git_commit": "commit-a",
        "git_dirty": False,
    }
    identity.update(updates)
    return identity


def _initialized(tmp_path: Path) -> tuple[Path, CheckpointManager, dict]:
    run = tmp_path / "results/runs/synthetic/run-1"
    run.mkdir(parents=True)
    manager = CheckpointManager(run)
    identity = _identity()
    manager.initialize(identity, resolved_policy={"profile_name": "safe"})
    return run, manager, identity


def test_checkpoint_is_atomic_and_validated_stage_can_be_reused(tmp_path):
    run, manager, identity = _initialized(tmp_path)
    artifact = write_csv_atomic(run / "selection.csv", pd.DataFrame({"feature": ["a", "b"]}))
    manager.transition("selection_completed", artifacts=[artifact], completed_fold_id=1)
    validation = manager.validate_resume(identity)
    assert "selection_completed" in validation.reusable_stages
    assert validation.completed_fold_ids == ("1",)
    assert not list(run.glob("*.partial"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("resolved_config_hash", "config-b"),
        ("protocol_hash", "protocol-b"),
        ("data_hash", "data-b"),
        ("row_alignment_hash", "rows-b"),
        ("seed", 43),
        ("git_commit", "commit-b"),
        ("git_dirty", True),
    ],
)
def test_resume_rejects_provenance_mismatches(tmp_path, field, value):
    _, manager, identity = _initialized(tmp_path)
    with pytest.raises(ResumeValidationError) as exc:
        manager.validate_resume({**identity, field: value})
    assert exc.value.code == f"resume_mismatch_{field}"


def test_resume_rejects_checksum_schema_and_row_count_mismatch(tmp_path):
    run, manager, identity = _initialized(tmp_path)
    artifact = write_csv_atomic(run / "selection.csv", pd.DataFrame({"feature": ["a"]}))
    manager.transition("selection_completed", artifacts=[artifact])
    (run / "selection.csv").write_text("feature\nb\n", encoding="utf-8")
    with pytest.raises(ResumeValidationError) as exc:
        manager.validate_resume(identity)
    assert exc.value.code in {"artifact_checksum_mismatch", "artifact_size_mismatch"}

    write_csv_atomic(run / "selection.csv", pd.DataFrame({"feature": ["a"]}))
    payload = manager.load()
    payload["finalized_artifacts"]["selection.csv"]["schema"] = {"different": "object"}
    write_json_atomic(manager.path, payload)
    with pytest.raises(ResumeValidationError) as exc:
        manager.validate_resume(identity)
    assert exc.value.code == "artifact_schema_mismatch"


def test_completed_run_is_immutable_and_not_resumable(tmp_path):
    _, manager, identity = _initialized(tmp_path)
    manager.transition("completed")
    with pytest.raises(ResumeValidationError) as exc:
        manager.validate_resume(identity)
    assert exc.value.code == "completed_run_immutable"
    with pytest.raises(ResumeValidationError):
        manager.transition("failed")


def test_explicit_resume_target_never_selects_newest_implicitly(tmp_path):
    results = tmp_path / "results"
    first = results / "runs/synthetic/run-1"
    second = results / "runs/other/run-2"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    assert resolve_resume_target(results, "run-1") == first.resolve()
    with pytest.raises(ResumeValidationError, match="explicit"):
        resolve_resume_target(results, "")


def test_partial_artifact_is_not_trusted_and_is_quarantined(tmp_path):
    run, manager, identity = _initialized(tmp_path)
    with pytest.raises(RuntimeError):
        atomic_publish(
            run / "selection.json",
            lambda partial: partial.write_text('{"ok": true}\n', encoding="utf-8"),
            artifact_format="json",
            before_replace=lambda *_: (_ for _ in ()).throw(RuntimeError("interrupt")),
        )
    validation = manager.validate_resume(identity)
    assert len(validation.quarantined_partials) == 1
    assert not (run / "selection.json").exists()
    valid = write_json_atomic(run / "selection.json", {"ok": True})
    manager.transition("selection_completed", artifacts=[valid])
    assert manager.validate_resume(identity).resumable


def test_artifact_provenance_mismatch_blocks_resume(tmp_path):
    run, manager, identity = _initialized(tmp_path)
    artifact = write_json_atomic(run / "artifact.json", {"ok": True})
    manager.transition("data_validated", artifacts=[artifact])
    payload = manager.load()
    payload["finalized_artifacts"]["artifact.json"]["provenance"]["data_hash"] = "wrong"
    write_json_atomic(manager.path, payload)
    with pytest.raises(ResumeValidationError) as exc:
        manager.validate_resume(identity)
    assert exc.value.code == "artifact_provenance_mismatch_data_hash"
