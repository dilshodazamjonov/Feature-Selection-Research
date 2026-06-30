from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest

from credit_risk_fs.clip.reverse_transfer import (
    REGISTRY_SCHEMAS,
    RegistryConflictError,
    append_registry_rows,
    atomic_registry_transaction,
    canonical_artifact_id,
    canonical_registry_value,
    registry_bundle_dry_run,
    validate_registry_bundle,
    validate_registry_frame,
    validate_sha256,
)
from credit_risk_fs.utils.hashing import sha256_file


def test_canonical_artifact_id_is_deterministic_and_path_aware() -> None:
    common = {
        "run_id": "run-1",
        "artifact_type": "table",
        "content_hash": "a" * 64,
    }
    first = canonical_artifact_id(
        **common, relative_path="./results/a.csv"
    )
    assert first == canonical_artifact_id(
        **common, relative_path="results/a.csv"
    )
    assert first != canonical_artifact_id(
        **common, relative_path="results/b.csv"
    )
    assert first != canonical_artifact_id(
        **{**common, "artifact_type": "manifest"},
        relative_path="results/a.csv",
    )


def test_identical_content_at_distinct_roles_has_distinct_artifact_ids() -> None:
    content_hash = "b" * 64
    alignment = canonical_artifact_id(
        run_id="",
        artifact_type="table",
        relative_path="results/training/alignment_metrics.csv",
        content_hash=content_hash,
    )
    retrieval = canonical_artifact_id(
        run_id="",
        artifact_type="table",
        relative_path="results/training/retrieval_metrics.csv",
        content_hash=content_hash,
    )
    assert alignment != retrieval


@pytest.mark.parametrize(
    "value",
    [
        "",
        " ",
        "unknown",
        "pending",
        "0" * 64,
        "a" * 63,
        "g" * 64,
    ],
)
def test_sha256_validation_rejects_malformed_values(value) -> None:
    with pytest.raises(ValueError):
        validate_sha256(value, field="prediction_hash")
    assert validate_sha256("A" * 64, field="prediction_hash") == "a" * 64


def test_schema_types_enums_json_and_primary_keys() -> None:
    schema = REGISTRY_SCHEMAS["run_index.csv"]
    row = {
        "run_id": "r1",
        "dataset": "HOMECREDIT",
        "method": "m",
        "model": "LR",
        "configuration_hash": "a" * 64,
        "data_manifest_hash": "b" * 64,
        "metric_artifact_path": "results/m.csv",
        "prediction_artifact_path": "results/p.csv",
        "selected_feature_path": "results/s.csv",
        "pairing_policy_version": "identity_equivalence_v2",
        "reuse_status": "NEWLY_EXECUTED",
        "seed": "42.0",
        "feature_budget": 20.0,
        "source_checkpoint_hashes": json.dumps({str(seed): "c" * 64 for seed in (11,22,33,44,55)}),
        "source_anchor_hashes": json.dumps({str(seed): "d" * 64 for seed in (11,22,33,44,55)}),
    }
    normalized = validate_registry_frame(pd.DataFrame([row]), schema=schema, strict=True)
    assert normalized.loc[0, "dataset"] == "homecredit"
    assert normalized.loc[0, "seed"] == "42"
    with pytest.raises(ValueError, match="primary key"):
        validate_registry_frame(
            pd.DataFrame([row, row]), schema=schema, strict=True
        )
    with pytest.raises(ValueError, match="missing required"):
        validate_registry_frame(
            pd.DataFrame([{k: v for k, v in row.items() if k != "run_id"}]),
            schema=schema,
            strict=True,
        )
    bad = dict(row, model="random_forest")
    with pytest.raises(ValueError, match="invalid enum"):
        validate_registry_frame(pd.DataFrame([bad]), schema=schema, strict=True)
    bad = dict(row, source_checkpoint_hashes="{bad")
    with pytest.raises(ValueError, match="malformed JSON"):
        validate_registry_frame(pd.DataFrame([bad]), schema=schema, strict=True)


def test_schema_aware_canonical_equivalence_and_path_safety() -> None:
    assert canonical_registry_value("feature_budget", 60, expected_type="integer") == canonical_registry_value(
        "feature_budget", "60.0", expected_type="integer"
    )
    assert canonical_registry_value("file_exists", True, expected_type="boolean") == canonical_registry_value(
        "file_exists", "true", expected_type="boolean"
    )
    assert canonical_registry_value("relative_path", r"results\x.csv", expected_type="path") == "results/x.csv"
    assert canonical_registry_value("payload", '{"b":2,"a":1}', expected_type="json") == '{"a":1,"b":2}'
    with pytest.raises(ValueError, match="integral"):
        canonical_registry_value("feature_budget", 60.5, expected_type="integer")
    with pytest.raises(ValueError, match="parent traversal"):
        canonical_registry_value("relative_path", "../outside.csv", expected_type="path")
    outside = Path(tempfile.gettempdir()).resolve() / "registry-outside.csv"
    assert Path.cwd().resolve() not in outside.parents
    with pytest.raises(ValueError, match="escapes"):
        canonical_registry_value("relative_path", str(outside), expected_type="path")


def _bundle(root: Path):
    artifact = root / "results" / "prediction.csv"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("x\n1\n", encoding="utf-8")
    h = sha256_file(artifact)
    run = pd.DataFrame([{
        "run_id": "r1", "dataset": "homecredit", "method": "m", "model": "lr",
        "configuration_hash": "a" * 64, "data_manifest_hash": "b" * 64,
        "metric_artifact_path": "results/prediction.csv",
        "prediction_artifact_path": "results/prediction.csv",
        "selected_feature_path": "results/prediction.csv",
        "pairing_policy_version": "identity_equivalence_v2",
        "reuse_status": "newly_executed",
    }])
    artifacts = pd.DataFrame([{
        "artifact_id": "c" * 64, "artifact_type": "prediction",
        "relative_path": "results/prediction.csv", "file_exists": True,
        "file_hash": h, "created_by_run_id": "r1", "depends_on_clip": True,
        "depends_on_old_pairing": False,
        "pairing_policy_version": "identity_equivalence_v2",
        "reuse_status": "newly_executed",
    }])
    metrics = pd.DataFrame([{
        "run_id": "r1", "dataset_name": "homecredit", "model": "lr",
        "selector": "m", "experiment_type": "corrected_reverse_transfer",
        "oot_auc": 0.7, "oot_ks": 0.3, "config_hash": "a" * 64,
        "data_manifest_hash": "b" * 64, "source_identity_manifest_hash": "e" * 64,
        "dev_prediction_hash": h, "oot_prediction_hash": h,
        "metric_artifact_path": "results/prediction.csv",
        "result_origin": "newly_executed", "reuse_status": "newly_executed",
        "pairing_policy_version": "identity_equivalence_v2",
    }])
    selected = pd.DataFrame([{
        "run_id": "r1", "dataset": "homecredit", "model": "lr",
        "selector": "m", "selected_feature_path": "results/prediction.csv",
        "selected_feature_hash": h,
        "configuration_hash": "a" * 64, "data_manifest_hash": "b" * 64,
        "source_identity_manifest_hash": "e" * 64,
        "pairing_policy_version": "identity_equivalence_v2",
        "reuse_status": "newly_executed",
    }])
    return {
        "run_index.csv": run,
        "artifact_registry.csv": artifacts,
        "reusable_metrics.csv": metrics,
        "selected_feature_registry.csv": selected,
    }


def _artifact_row(
    *,
    artifact_id: str,
    relative_path: str,
    file_hash: str,
    artifact_type: str = "prediction",
    owner: str = "r1",
) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "relative_path": relative_path,
        "file_exists": True,
        "file_hash": file_hash,
        "created_by_run_id": owner,
        "depends_on_clip": True,
        "depends_on_old_pairing": False,
        "configuration_hash": "a" * 64,
        "data_manifest_hash": "b" * 64,
        "pairing_policy_version": "identity_equivalence_v2",
        "reuse_status": "newly_executed",
        "dataset": "homecredit",
        "model": "lr",
        "method": "m",
        "scientific_stage": "evaluate",
    }


def _with_artifacts(
    root: Path, rows: list[dict[str, object]]
) -> dict[str, pd.DataFrame]:
    frames = _bundle(root)
    frames["artifact_registry.csv"] = pd.concat(
        [frames["artifact_registry.csv"], pd.DataFrame(rows)],
        ignore_index=True,
        sort=False,
    )
    return frames


def test_bundle_referential_integrity_and_artifact_hashes(tmp_path) -> None:
    frames = _bundle(tmp_path)
    validate_registry_bundle(
        frames, verify_artifacts=True, repository_root=tmp_path, enforced_run_ids={"r1"}
    )
    broken = {key: value.copy() for key, value in frames.items()}
    broken["reusable_metrics.csv"].loc[0, "run_id"] = "missing"
    with pytest.raises(ValueError, match="missing run"):
        validate_registry_bundle(broken, enforced_run_ids={"r1"})
    broken = {key: value.copy() for key, value in frames.items()}
    broken["selected_feature_registry.csv"].loc[0, "selected_feature_hash"] = "d" * 64
    with pytest.raises(ValueError, match="selected-feature"):
        validate_registry_bundle(broken, enforced_run_ids={"r1"})
    (tmp_path / "results" / "prediction.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        validate_registry_bundle(
            frames, verify_artifacts=True, repository_root=tmp_path, enforced_run_ids={"r1"}
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "new_id_two_paths",
        "equivalent_paths_conflicting_hashes",
        "new_id_two_hashes",
        "new_id_two_types",
        "new_id_two_owners",
        "one_path_two_ids",
        "existing_id_new_path",
        "existing_path_new_id",
        "existing_registry_id_two_paths",
        "windows_posix_collision",
        "dot_path_collision",
        "same_path_changed_content",
    ],
)
def test_artifact_identity_conflicts_are_rejected(
    tmp_path: Path, mutation: str
) -> None:
    first_id, second_id = "d" * 64, "f" * 64
    first_hash, second_hash = "1" * 64, "2" * 64
    path_a, path_b = "results/a.csv", "results/b.csv"
    rows: list[dict[str, object]]
    if mutation in {"new_id_two_paths", "existing_registry_id_two_paths"}:
        rows = [
            _artifact_row(
                artifact_id=first_id,
                relative_path=path_a,
                file_hash=first_hash,
            ),
            _artifact_row(
                artifact_id=first_id,
                relative_path=path_b,
                file_hash=second_hash,
            ),
        ]
    elif mutation == "equivalent_paths_conflicting_hashes":
        rows = [
            _artifact_row(
                artifact_id=first_id,
                relative_path=r"results\same.csv",
                file_hash=first_hash,
            ),
            _artifact_row(
                artifact_id=first_id,
                relative_path="results/same.csv",
                file_hash=second_hash,
            ),
        ]
    elif mutation in {"new_id_two_hashes", "same_path_changed_content"}:
        rows = [
            _artifact_row(
                artifact_id=first_id,
                relative_path=path_a,
                file_hash=first_hash,
            ),
            _artifact_row(
                artifact_id=first_id,
                relative_path=path_a,
                file_hash=second_hash,
            ),
        ]
    elif mutation == "new_id_two_types":
        rows = [
            _artifact_row(
                artifact_id=first_id,
                relative_path=path_a,
                file_hash=first_hash,
                artifact_type="prediction",
            ),
            _artifact_row(
                artifact_id=first_id,
                relative_path=path_a,
                file_hash=first_hash,
                artifact_type="metric",
            ),
        ]
    elif mutation == "new_id_two_owners":
        rows = [
            _artifact_row(
                artifact_id=first_id,
                relative_path=path_a,
                file_hash=first_hash,
                owner="r1",
            ),
            _artifact_row(
                artifact_id=first_id,
                relative_path=path_a,
                file_hash=first_hash,
                owner="r2",
            ),
        ]
    elif mutation in {
        "one_path_two_ids",
        "windows_posix_collision",
        "dot_path_collision",
    }:
        second_path = path_a
        if mutation == "windows_posix_collision":
            second_path = r"results\a.csv"
        elif mutation == "dot_path_collision":
            second_path = "./results/a.csv"
        rows = [
            _artifact_row(
                artifact_id=first_id,
                relative_path=path_a,
                file_hash=first_hash,
            ),
            _artifact_row(
                artifact_id=second_id,
                relative_path=second_path,
                file_hash=first_hash,
            ),
        ]
    elif mutation == "existing_id_new_path":
        rows = [
            _artifact_row(
                artifact_id="c" * 64,
                relative_path=path_b,
                file_hash=second_hash,
            )
        ]
    elif mutation == "existing_path_new_id":
        base_artifact = tmp_path / "results" / "prediction.csv"
        base_artifact.parent.mkdir(parents=True, exist_ok=True)
        base_artifact.write_text("x\n1\n", encoding="utf-8")
        base_hash = sha256_file(base_artifact)
        rows = [
            _artifact_row(
                artifact_id=first_id,
                relative_path="results/prediction.csv",
                file_hash=base_hash,
            )
        ]
    else:
        raise AssertionError(mutation)

    with pytest.raises(RegistryConflictError) as exc_info:
        validate_registry_bundle(
            _with_artifacts(tmp_path, rows),
            repository_root=tmp_path,
            enforced_run_ids={"r1"},
        )
    diagnostic = str(exc_info.value)
    for field in (
        "invariant=",
        "artifact_ids=",
        "canonical_paths=",
        "hashes=",
        "owning_run_ids=",
        "artifact_types=",
        "origins=",
    ):
        assert field in diagnostic


def test_current_versus_proposed_artifact_conflict_is_diagnostic(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "artifact_registry.csv"
    existing = _bundle(tmp_path)["artifact_registry.csv"]
    existing.to_csv(registry, index=False)
    proposed = pd.DataFrame(
        [
            _artifact_row(
                artifact_id="c" * 64,
                relative_path="results/different.csv",
                file_hash="1" * 64,
            )
        ]
    ).reindex(columns=existing.columns)
    with pytest.raises(RegistryConflictError) as exc_info:
        append_registry_rows(
            registry_path=registry,
            rows=proposed,
            equivalence_columns=["artifact_id"],
        )
    diagnostic = str(exc_info.value)
    assert "current" in diagnostic
    assert "proposed" in diagnostic
    assert "artifact_id->canonical_path" in diagnostic


@pytest.mark.parametrize(
    "field,other",
    [
        ("dataset", "lendingclub_v2"),
        ("model", "catboost"),
        ("method", "different_method"),
        ("configuration_hash", "9" * 64),
        ("data_manifest_hash", "8" * 64),
        ("pairing_policy_version", "different_policy"),
        ("scientific_stage", "project"),
    ],
)
def test_artifact_scientific_provenance_must_be_consistent(
    tmp_path: Path, field: str, other: str
) -> None:
    first = _artifact_row(
        artifact_id="d" * 64,
        relative_path="results/a.csv",
        file_hash="1" * 64,
    )
    second = dict(first)
    second[field] = other
    with pytest.raises(
        RegistryConflictError, match=f"artifact_id->{field}"
    ):
        validate_registry_bundle(
            _with_artifacts(tmp_path, [first, second]),
            repository_root=tmp_path,
            enforced_run_ids={"r1"},
        )


def test_artifact_identity_valid_idempotent_and_canonical_cases(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "results" / "new.csv"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("x\n2\n", encoding="utf-8")
    new_row = _artifact_row(
        artifact_id="d" * 64,
        relative_path="results/new.csv",
        file_hash=sha256_file(artifact),
    )
    frames = _with_artifacts(tmp_path, [new_row])
    validate_registry_bundle(
        frames,
        verify_artifacts=True,
        repository_root=tmp_path,
        enforced_run_ids={"r1"},
    )
    assert canonical_registry_value(
        "relative_path",
        r".\results\reverse\.\prediction.csv",
        expected_type="path",
    ) == "results/reverse/prediction.csv"

    registry = tmp_path / "artifact_registry.csv"
    frames["artifact_registry.csv"].to_csv(registry, index=False)
    equivalent = pd.DataFrame(
        [
            {
                **new_row,
                "relative_path": r".\results\new.csv",
            }
        ]
    ).reindex(columns=frames["artifact_registry.csv"].columns)
    combined = append_registry_rows(
        registry_path=registry,
        rows=equivalent,
        equivalence_columns=["artifact_id"],
    )
    assert len(combined) == len(frames["artifact_registry.csv"])
    assert combined.attrs["registry_changed"] is False
    assert set(combined["artifact_id"]) == {
        "c" * 64,
        "d" * 64,
    }


def test_registry_bundle_dry_run_reports_conflict_without_writes(
    tmp_path: Path,
) -> None:
    rows = [
        _artifact_row(
            artifact_id="d" * 64,
            relative_path="results/a.csv",
            file_hash="1" * 64,
        ),
        _artifact_row(
            artifact_id="d" * 64,
            relative_path="results/b.csv",
            file_hash="2" * 64,
        ),
    ]
    frames = _with_artifacts(tmp_path, rows)
    before = {
        path: path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    result = registry_bundle_dry_run(
        frames,
        repository_root=tmp_path,
        enforced_run_ids={"r1"},
    )
    assert result["transaction_outcome"] == "CONFLICT"
    assert "d" * 64 in result["conflict_diagnostic"]
    assert "results/a.csv" in result["conflict_diagnostic"]
    assert "results/b.csv" in result["conflict_diagnostic"]
    assert "1" * 64 in result["conflict_diagnostic"]
    assert "2" * 64 in result["conflict_diagnostic"]
    assert result["affected_active_files"] == []
    assert result["writes_performed"] is False
    assert result["success_transaction_manifest_written"] is False
    assert {
        path: path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == before

    valid = registry_bundle_dry_run(
        _bundle(tmp_path),
        verify_artifacts=True,
        repository_root=tmp_path,
        enforced_run_ids={"r1"},
    )
    assert valid["transaction_outcome"] == "NEW_TRANSACTION"
    assert valid["writes_performed"] is False


def test_transaction_new_noop_and_rollback(tmp_path, monkeypatch) -> None:
    one, two = tmp_path / "one.csv", tmp_path / "two.csv"
    manifest = tmp_path / "transaction.json"
    first = atomic_registry_transaction(
        {one: b"a\n1\n", two: b"b\n2\n"},
        transaction_manifest_path=manifest,
        metadata={"run_ids": ["r1"]},
    )
    assert first["transaction_outcome"] == "NEW_TRANSACTION"
    original = {one: one.read_bytes(), two: two.read_bytes(), manifest: manifest.read_bytes()}
    second = atomic_registry_transaction(
        {one: original[one], two: original[two]},
        transaction_manifest_path=manifest,
        metadata={"run_ids": ["r1"]},
    )
    assert second["transaction_outcome"] == "IDEMPOTENT_NO_OP"
    assert {path: path.read_bytes() for path in original} == original

    import credit_risk_fs.clip.reverse_transfer as module
    real_replace = module.os.replace
    calls = 0

    def fail_second(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("replacement failure")
        return real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_second)
    with pytest.raises(OSError):
        atomic_registry_transaction(
            {one: b"new-one", two: b"new-two"},
            transaction_manifest_path=manifest,
            metadata={"run_ids": ["r1"]},
        )
    assert one.read_bytes() == original[one]
    assert two.read_bytes() == original[two]
    assert manifest.read_bytes() == original[manifest]


def _transaction_fixture(
    root: Path,
) -> tuple[
    dict[Path, bytes],
    Path,
    dict[Path, bytes | None],
    Path,
]:
    frames = _bundle(root)
    active = root / "synthetic_registry"
    active.mkdir(parents=True, exist_ok=True)
    payloads: dict[Path, bytes] = {
        active / name: frame.to_csv(index=False).encode("utf-8")
        for name, frame in frames.items()
    }
    guide = active / "results_access_guide.md"
    payloads[guide] = b"# updated synthetic guide\n"
    summary = active / "summary_manifest.json"
    payloads[summary] = json.dumps(
        {
            "registry_version": "synthetic",
            "run_counts": {"homecredit": 1},
            "artifact_counts": {"prediction": 1},
            "registry_file_hashes": {
                "synthetic_registry/run_index.csv": "a" * 64
            },
        },
        indent=2,
    ).encode("utf-8")
    originals: dict[Path, bytes | None] = {}
    for index, path in enumerate(payloads):
        if path == guide:
            originals[path] = None
            continue
        original = f"ORIGINAL-{index}\r\n".encode("utf-8")
        path.write_bytes(original)
        originals[path] = original
    manifest = root / "transaction_manifest.json"
    originals[manifest] = None
    sentinel = root / "unrelated_task_1_task_3_baseline.bin"
    sentinel.write_bytes(b"UNRELATED-ORIGINAL-BYTES")
    return payloads, manifest, originals, sentinel


@pytest.mark.parametrize(
    "point,target_index,phase",
    [
        ("after_lock_acquired", None, None),
        ("after_originals_captured", None, None),
        ("during_temp_create", 0, None),
        ("during_temp_write", 0, None),
        ("during_temp_write", 2, None),
        ("during_temp_flush", 1, None),
        ("during_temp_fsync", 1, None),
        ("during_temp_schema_validation", 1, None),
        ("during_temp_hash_validation", 1, None),
        ("before_first_replace", None, None),
        ("after_first_replace", None, None),
        ("after_middle_replace", 2, None),
        ("after_final_replace", None, None),
        ("after_summary_replace", None, None),
        ("during_post_write_schema_validation", None, None),
        ("during_post_write_hash_validation", None, None),
        ("after_post_commit_validation", None, None),
        ("during_cleanup", None, None),
        ("before_transaction_manifest", None, None),
        ("during_transaction_manifest_write", None, None),
        ("during_transaction_manifest_validation", None, "persisted"),
        ("during_transaction_manifest_replace", None, None),
    ],
)
def test_every_precommit_failure_boundary_restores_exact_bytes_and_unlocks(
    tmp_path: Path,
    point: str,
    target_index: int | None,
    phase: str | None,
) -> None:
    payloads, manifest, originals, sentinel = _transaction_fixture(tmp_path)
    lock = manifest.with_suffix(".lock")

    def inject(actual_point: str, context: dict[str, object]) -> None:
        if actual_point != point:
            return
        if target_index is not None and context.get("target_index") != target_index:
            return
        if phase is not None and context.get("phase") != phase:
            return
        raise RuntimeError(f"injected failure at {point}")

    with pytest.raises(RuntimeError, match="injected failure") as exc_info:
        atomic_registry_transaction(
            payloads,
            transaction_manifest_path=manifest,
            metadata={"run_ids": ["r1"]},
            failure_injector=inject,
        )
    assert any(
        "rollback succeeded with byte verification" in note
        for note in getattr(exc_info.value, "__notes__", [])
    )
    for path, original in originals.items():
        assert (path.read_bytes() if path.exists() else None) == original
    assert sentinel.read_bytes() == b"UNRELATED-ORIGINAL-BYTES"
    assert not manifest.exists()
    assert not lock.exists()
    assert not list(tmp_path.rglob("*.tmp"))

    retry = atomic_registry_transaction(
        payloads,
        transaction_manifest_path=manifest,
        metadata={"run_ids": ["r1"]},
    )
    assert retry["transaction_outcome"] == "NEW_TRANSACTION"
    assert manifest.exists()
    assert not lock.exists()
    second = atomic_registry_transaction(
        payloads,
        transaction_manifest_path=manifest,
        metadata={"run_ids": ["r1"]},
    )
    assert second["transaction_outcome"] == "IDEMPOTENT_NO_OP"


def test_transaction_manifest_replacement_is_the_commit_boundary(
    tmp_path: Path,
) -> None:
    payloads, manifest, originals, sentinel = _transaction_fixture(tmp_path)

    def fail_after_commit(
        point: str, context: dict[str, object]
    ) -> None:
        if point == "after_transaction_manifest":
            assert context["commit_boundary_reached"] is True
            raise RuntimeError("post-commit injected failure")

    result = atomic_registry_transaction(
        payloads,
        transaction_manifest_path=manifest,
        metadata={"run_ids": ["r1"]},
        failure_injector=fail_after_commit,
    )
    assert result["transaction_outcome"] == "NEW_TRANSACTION"
    assert "post_commit_warnings" in result
    assert manifest.exists()
    persisted = json.loads(manifest.read_text(encoding="utf-8"))
    assert (
        persisted["commit_boundary"]
        == "validated_transaction_manifest_atomic_replacement"
    )
    for path in payloads:
        key = str(path).replace("\\", "/")
        assert persisted["original_file_existence"][key] is (
            originals[path] is not None
        )
        expected_hash = (
            hashlib.sha256(originals[path]).hexdigest()
            if originals[path] is not None
            else None
        )
        assert persisted["pre_transaction_hashes"][key] == expected_hash
    for path, content in payloads.items():
        assert path.read_bytes() == content
    assert sentinel.read_bytes() == b"UNRELATED-ORIGINAL-BYTES"
    assert not manifest.with_suffix(".lock").exists()
    second = atomic_registry_transaction(
        payloads,
        transaction_manifest_path=manifest,
        metadata={"run_ids": ["r1"]},
    )
    assert second["transaction_outcome"] == "IDEMPOTENT_NO_OP"
