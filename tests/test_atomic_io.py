from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from credit_risk_fs.experiments.atomic_io import (
    ArtifactIntegrityError,
    atomic_publish,
    inspect_artifact,
    partial_artifacts,
    quarantine_partial_artifacts,
    write_csv_atomic,
    write_json_atomic,
    write_parquet_atomic,
)


def test_atomic_json_csv_and_parquet_publication(tmp_path):
    json_meta = write_json_atomic(tmp_path / "record.json", {"ok": True})
    assert json.loads((tmp_path / "record.json").read_text(encoding="utf-8"))["ok"]
    assert json_meta.sha256 == inspect_artifact(tmp_path / "record.json").sha256

    frame = pd.DataFrame({"stable_row_id": ["a", "b"], "value": [1.0, 2.0]})
    csv_meta = write_csv_atomic(
        tmp_path / "record.csv",
        frame,
        required_columns=["stable_row_id", "value"],
        ordered_row_identity_column="stable_row_id",
    )
    parquet_meta = write_parquet_atomic(
        tmp_path / "record.parquet",
        frame,
        required_columns=["stable_row_id", "value"],
        ordered_row_identity_column="stable_row_id",
    )
    assert csv_meta.row_count == parquet_meta.row_count == 2
    assert csv_meta.ordered_row_identity_sha256 == parquet_meta.ordered_row_identity_sha256


def test_validation_failure_never_publishes_invalid_final(tmp_path):
    target = tmp_path / "bad.csv"

    def writer(path: Path) -> None:
        path.write_text("wrong\n1\n", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="missing required"):
        atomic_publish(
            target,
            writer,
            artifact_format="csv",
            required_columns=["required"],
        )
    assert not target.exists()
    assert len(partial_artifacts(tmp_path)) == 1


def test_failed_replacement_preserves_existing_valid_final(tmp_path):
    target = tmp_path / "record.json"
    write_json_atomic(target, {"version": 1})
    original = target.read_bytes()

    def interrupt(_partial: Path, _target: Path) -> None:
        raise KeyboardInterrupt("simulated between validation and replace")

    with pytest.raises(KeyboardInterrupt):
        atomic_publish(
            target,
            lambda partial: partial.write_text('{"version": 2}\n', encoding="utf-8"),
            artifact_format="json",
            before_replace=interrupt,
        )
    assert target.read_bytes() == original
    assert len(partial_artifacts(tmp_path)) == 1


def test_partial_files_are_identified_and_quarantined_within_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(RuntimeError):
        atomic_publish(
            run_dir / "artifact.json",
            lambda partial: partial.write_text('{"ok": true}\n', encoding="utf-8"),
            artifact_format="json",
            before_replace=lambda *_: (_ for _ in ()).throw(RuntimeError("interrupt")),
        )
    moved = quarantine_partial_artifacts(run_dir)
    assert len(moved) == 1
    assert moved[0].is_relative_to(run_dir / "incomplete")
    assert not (run_dir / "artifact.json").exists()


def test_parquet_metadata_validation_rejects_row_count_and_schema(tmp_path):
    path = tmp_path / "data.parquet"
    write_parquet_atomic(path, pd.DataFrame({"a": [1, 2]}))
    with pytest.raises(ArtifactIntegrityError, match="row-count"):
        inspect_artifact(path, expected_row_count=3)
    with pytest.raises(ArtifactIntegrityError, match="missing required"):
        inspect_artifact(path, required_columns=["missing"])
