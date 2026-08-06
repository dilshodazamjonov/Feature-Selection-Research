"""Focused, data-free regression tests for the Prompt 14A manifest repair."""

from __future__ import annotations

import ast
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from credit_risk_fs.analysis.voting_inference import manifest_authentication as auth


ROOT = Path(__file__).resolve().parents[1]


def _load_builder_module():
    path = ROOT / "scripts/build_voting_inference_evidence.py"
    spec = importlib.util.spec_from_file_location("prompt14a_builder_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def test_exact_legacy_blocker_reproduction_is_data_free() -> None:
    result = auth.reproduce_legacy_blocker(ROOT)
    assert result["declared_generated_file_count"] == 55
    assert result["matching_entries"] == 54
    assert result["mismatching_entries"] == 1
    mismatches = [entry for entry in result["ordered_inventory"] if not entry["authenticated"]]
    assert [entry["path"] for entry in mismatches] == [auth.STATUS_PATH.as_posix()]
    assert result["data_free_safety"] == {
        "raw_dataset_path_resolution_implemented": False,
        "worker_start_implemented": False,
        "raw_dataset_paths_resolved": False,
        "workers_started": 0,
    }


def test_successor_authenticates_exact_scope_and_selects_explicitly() -> None:
    result = auth.validate_voting_package(ROOT)
    assert result["selection_rule"] == "explicit_successor_pointer_fail_closed"
    assert result["ordered_payload_entry_count"] == 55
    assert result["authenticated_payload_entries"] == 55
    assert result["mismatching_payload_entries"] == 0
    assert result["unchanged_entries_checked"] == 54
    assert result["unchanged_entries_byte_identical"] == 54
    assert result["data_free_safety"]["raw_dataset_paths_resolved"] is False
    assert result["data_free_safety"]["workers_started"] == 0


def test_modified_payload_is_rejected_after_manifest_sealing(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"sealed")
    manifest = {
        "generated_files": [
            {
                "path": "payload.bin",
                "size_bytes": payload.stat().st_size,
                "sha256": auth.sha256_file(payload),
            }
        ],
        "generated_file_count": 1,
    }
    assert auth.authenticate_manifest_entries(tmp_path, manifest)[0]["authenticated"]
    payload.write_bytes(b"modified")
    assert not auth.authenticate_manifest_entries(tmp_path, manifest)[0]["authenticated"]


def test_modified_supersession_metadata_fails_closed(tmp_path: Path) -> None:
    selection = auth.select_manifest(ROOT)
    assert selection.pointer is not None
    for relative in (
        auth.LEGACY_MANIFEST,
        Path(selection.manifest_path),
        auth.SELECTION_POINTER,
        Path(selection.pointer["supersession_record_path"]),
        *(Path(item["path"]) for item in selection.supersession["provenance_evidence"]),
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    assert auth.select_manifest(tmp_path).selection_rule == "explicit_successor_pointer_fail_closed"

    supersession_path = tmp_path / selection.pointer["supersession_record_path"]
    supersession = json.loads(supersession_path.read_text(encoding="utf-8"))
    supersession["decision_code"] = "tampered"
    _write_json(supersession_path, supersession)
    with pytest.raises(auth.ManifestAuthenticationError, match="supersession record digest mismatch"):
        auth.select_manifest(tmp_path)


def test_unauthenticated_status_rejected_even_with_candidate_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / auth.PROMPT_07_CHECKPOINT
    checkpoint_manifest_path = tmp_path / auth.PROMPT_07_MANIFEST
    checkpoint = {
        "artifact_manifest.json": auth.LEGACY_MANIFEST_SHA256,
        "status.json": auth.CURRENT_STATUS_SHA256,
    }
    _write_json(checkpoint_path, checkpoint)
    checkpoint_sha = auth.sha256_file(checkpoint_path)
    _write_json(
        checkpoint_manifest_path,
        {
            "artifacts": [
                {
                    "path": auth.PROMPT_07_CHECKPOINT.as_posix(),
                    "sha256": checkpoint_sha,
                }
            ]
        },
    )
    monkeypatch.setattr(auth, "PROMPT_07_CHECKPOINT_SHA256", checkpoint_sha)
    monkeypatch.setattr(
        auth,
        "PROMPT_07_MANIFEST_SHA256",
        auth.sha256_file(checkpoint_manifest_path),
    )

    unauthenticated = sha256(b"candidate status whose hash was inserted").hexdigest()
    candidate_manifest = {
        "generated_files": [
            {"path": auth.STATUS_PATH.as_posix(), "size_bytes": 41, "sha256": unauthenticated}
        ],
        "generated_file_count": 1,
    }
    assert candidate_manifest["generated_files"][0]["sha256"] == unauthenticated
    with pytest.raises(auth.ManifestAuthenticationError, match="immutable Prompt 7 checkpoint"):
        auth.validate_status_checkpoint(tmp_path, unauthenticated)


def test_packager_writes_final_status_before_hashing_and_excludes_manifests(
    tmp_path: Path,
) -> None:
    builder = _load_builder_module()
    package = tmp_path / "package"
    audit = tmp_path / "audit"
    package.mkdir()
    audit.mkdir()
    (package / "payload.bin").write_bytes(b"payload")
    _write_json(package / "status.json", {"status": "STALE"})
    _write_json(package / "artifact_manifest.current.json", {"metadata": True})
    _write_json(package / "artifact_manifest.v2.json", {"metadata": True})
    _write_json(audit / "audit.json", {"status": "PASS"})
    config = SimpleNamespace(
        package_root=package,
        audit_root=audit,
        repository_root=tmp_path,
        payload={"analysis_id": "synthetic_data_free_test"},
        config_sha256="0" * 64,
    )
    status = {"phases": {}}
    manifest = builder._seal_final_status_and_manifest(
        config,
        state={"git_head": "test", "git_tags_at_head": []},
        frozen={"inputs": []},
        status=status,
        final_status={"status": "PASS", "success_marker": "TEST_PASS"},
    )
    status_entry = next(
        entry for entry in manifest["generated_files"] if entry["path"] == "package/status.json"
    )
    assert status_entry["sha256"] == auth.sha256_file(package / "status.json")
    assert status["phases"]["J_provenance"] == "PASS"
    assert not any(
        entry["path"].startswith("package/artifact_manifest")
        for entry in manifest["generated_files"]
    )


def test_validator_import_surface_cannot_start_workers_or_resolve_raw_paths() -> None:
    source = (ROOT / "src/credit_risk_fs/analysis/voting_inference/manifest_authentication.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert all(not name.startswith("credit_risk_fs") for name in imported)
    assert imported <= {
        "annotations",
        "dataclass",
        "sha256",
        "json",
        "Path",
        "Any",
        "Mapping",
        "Sequence",
    }
