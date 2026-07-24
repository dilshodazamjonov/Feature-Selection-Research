from __future__ import annotations

import json
from pathlib import Path

import pytest

from credit_risk_fs.experiments.atomic_io import sha256_file
from credit_risk_fs.experiments import provenance_bridge as bridge


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _fixture_bridge(root: Path) -> dict:
    runtime = root / "runtime.py"
    frozen = root / "frozen.yaml"
    runtime.write_text("runtime = 'mechanics-only'\n", encoding="utf-8")
    frozen.write_text("frozen: true\n", encoding="utf-8")
    run_entries = {}
    for index, run_id in enumerate(
        (*bridge.REUSABLE_RUN_IDS, bridge.INTERRUPTED_RUN_ID)
    ):
        checkpoint_commit = (
            bridge.ORIGINAL_COMMIT if index < 11 else bridge.SAFETY_COMMIT
        )
        run_dir = root / "results" / "runs" / "fixture" / run_id
        run_dir.mkdir(parents=True)
        artifact = run_dir / "immutable.txt"
        artifact.write_text(run_id + "\n", encoding="utf-8")
        _write_json(
            run_dir / "checkpoint.json",
            {
                "identity": {
                    "run_id": run_id,
                    "git_commit": checkpoint_commit,
                }
            },
        )
        _write_json(run_dir / "config.json", {"run_id": run_id})
        run_entries[run_id] = {
            "run_directory": run_dir.relative_to(root).as_posix(),
            "checkpoint_commit": checkpoint_commit,
            "immutable_artifacts": {
                "immutable.txt": {
                    "size_bytes": artifact.stat().st_size,
                    "sha256": sha256_file(artifact),
                }
            },
        }
    payload = {
        "schema_version": bridge.BRIDGE_SCHEMA_VERSION,
        "research_family": bridge.RESEARCH_FAMILY,
        "original_release": {
            "tag": bridge.ORIGINAL_TAG,
            "commit": bridge.ORIGINAL_COMMIT,
        },
        "safety_release": {
            "tag": bridge.SAFETY_TAG,
            "commit": bridge.SAFETY_COMMIT,
        },
        "observability_release": {
            "tag": bridge.OBSERVABILITY_TAG,
            "commit_binding": "annotated_tag_peels_to_current_head",
        },
        "reusable_run_ids": list(bridge.REUSABLE_RUN_IDS),
        "interrupted_run": {
            "run_id": bridge.INTERRUPTED_RUN_ID,
            "safe_resume_boundary": bridge.SAFE_RESUME_BOUNDARY,
        },
        "runtime_files": {
            "runtime.py": {
                "old_sha256": "fixture-old",
                "new_sha256": sha256_file(runtime),
                "new_size_bytes": runtime.stat().st_size,
            }
        },
        "frozen_files": {
            "frozen.yaml": {
                "sha256": sha256_file(frozen),
                "size_bytes": frozen.stat().st_size,
            }
        },
        "runs": run_entries,
    }
    _write_json(root / bridge.BRIDGE_PATH, payload)
    return payload


def _fake_git(_root, *args):
    if args == ("rev-list", "-n", "1", bridge.ORIGINAL_TAG):
        return bridge.ORIGINAL_COMMIT
    if args == ("rev-list", "-n", "1", bridge.SAFETY_TAG):
        return bridge.SAFETY_COMMIT
    if args == ("rev-list", "-n", "1", bridge.OBSERVABILITY_TAG):
        return "b" * 40
    if args[0:2] == ("cat-file", "-t"):
        return "tag"
    raise AssertionError(args)


def test_exact_bridge_authenticates_and_resolves_historical_identity(
    tmp_path, monkeypatch
):
    _fixture_bridge(tmp_path)
    monkeypatch.setattr(bridge, "_git", _fake_git)
    bridge._AUTHENTICATED_RELEASES.clear()
    payload = bridge.authenticate_compatibility_bridge(
        tmp_path,
        current_commit="b" * 40,
        current_tag=bridge.OBSERVABILITY_TAG,
    )
    assert payload["reusable_run_ids"] == list(bridge.REUSABLE_RUN_IDS)
    run_dir = (
        tmp_path / "results" / "runs" / "fixture" / bridge.INTERRUPTED_RUN_ID
    )
    identity, config, metadata = bridge.compatible_resume_identity(
        tmp_path,
        run_dir,
        current_commit="b" * 40,
        current_tag=bridge.OBSERVABILITY_TAG,
    )
    assert identity["git_commit"] == bridge.SAFETY_COMMIT
    assert config["run_id"] == bridge.INTERRUPTED_RUN_ID
    assert metadata["authorized_run_id"] == bridge.INTERRUPTED_RUN_ID


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        ("tag", "BRIDGE_CURRENT_TAG_MISMATCH"),
        ("runtime", "BRIDGE_RUNTIME_HASH_MISMATCH"),
        ("artifact", "BRIDGE_ARTIFACT_HASH_MISMATCH"),
        ("inventory", "BRIDGE_RUN_INVENTORY_MISMATCH"),
    ],
)
def test_any_tag_runtime_artifact_or_inventory_drift_blocks_reuse(
    tmp_path, monkeypatch, mutation, code
):
    payload = _fixture_bridge(tmp_path)
    monkeypatch.setattr(bridge, "_git", _fake_git)
    current_tag = bridge.OBSERVABILITY_TAG
    if mutation == "tag":
        current_tag = "wrong-tag"
    elif mutation == "runtime":
        (tmp_path / "runtime.py").write_text("changed\n", encoding="utf-8")
    elif mutation == "artifact":
        target = (
            tmp_path
            / payload["runs"][bridge.REUSABLE_RUN_IDS[0]]["run_directory"]
            / "immutable.txt"
        )
        target.write_bytes(b"x" * target.stat().st_size)
    else:
        payload["reusable_run_ids"] = payload["reusable_run_ids"][:-1]
        _write_json(tmp_path / bridge.BRIDGE_PATH, payload)
    bridge._AUTHENTICATED_RELEASES.clear()
    with pytest.raises(bridge.ProvenanceBridgeError) as error:
        bridge.authenticate_compatibility_bridge(
            tmp_path,
            current_commit="b" * 40,
            current_tag=current_tag,
        )
    assert error.value.code == code
