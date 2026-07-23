"""Narrow provenance bridge for the exact cdv1 mechanics-only resume release."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from credit_risk_fs.experiments.atomic_io import sha256_file


BRIDGE_SCHEMA_VERSION = "cdv1_resume_compatibility_bridge_v1"
BRIDGE_PATH = Path("configs/execution/cdv1_resume_compatibility_bridge_v1.json")
RESEARCH_FAMILY = "cdv1"
ORIGINAL_TAG = "cross-dataset-voting-pre-execution-v1"
MECHANICS_TAG = "cross-dataset-voting-resume-safety-v1"
ORIGINAL_COMMIT = "f00f474b6f263ee2619a178524c7c0fdf806024f"
REUSABLE_RUN_IDS = tuple(
    [
        "cdv1-001-homecredit-reference-rf-corr-mrmr-lr-s42",
        "cdv1-002-homecredit-voting-k100-lr-s42",
        "cdv1-003-homecredit-voting-k200-lr-s42",
        "cdv1-004-homecredit-voting-k300-lr-s42",
        "cdv1-005-homecredit-reference-rf-corr-mrmr-catboost-s42",
        "cdv1-006-homecredit-voting-k100-catboost-s42",
        "cdv1-007-homecredit-voting-k200-catboost-s42",
        "cdv1-008-homecredit-voting-k300-catboost-s42",
        "cdv1-009-lendingclub-v2-reference-rf-corr-mrmr-lr-s42",
        "cdv1-010-lendingclub-v2-voting-k100-lr-s42",
    ]
)
INTERRUPTED_RUN_ID = "cdv1-011-lendingclub-v2-voting-k200-lr-s42"
SAFE_RESUME_BOUNDARY = {
    "phase": "DEV",
    "fold_id": 3,
    "stage": "dev_data_loading",
    "first_selection_stage": "selection_encoding",
    "first_expensive_stage": "voter_rf_corr_mrmr",
    "discarded_in_memory_stage": "voter_boruta",
}
_AUTHENTICATED_RELEASES: set[tuple[str, str]] = set()


class ProvenanceBridgeError(RuntimeError):
    """Fail-closed bridge authentication error with a stable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise ProvenanceBridgeError(
            "BRIDGE_GIT_PROVENANCE_UNAVAILABLE", detail.strip()
        ) from exc
    return result.stdout.strip()


def load_compatibility_bridge(repository_root: str | Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    path = root / BRIDGE_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProvenanceBridgeError(
            "BRIDGE_MISSING", f"compatibility bridge is missing: {BRIDGE_PATH.as_posix()}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceBridgeError("BRIDGE_UNREADABLE", str(exc)) from exc
    if payload.get("schema_version") != BRIDGE_SCHEMA_VERSION:
        raise ProvenanceBridgeError("BRIDGE_SCHEMA_MISMATCH", "unsupported bridge schema")
    return payload


def _require_exact_bridge_identity(payload: Mapping[str, Any]) -> None:
    if payload.get("research_family") != RESEARCH_FAMILY:
        raise ProvenanceBridgeError("BRIDGE_FAMILY_MISMATCH", "research family differs")
    original = payload.get("original_release", {})
    mechanics = payload.get("mechanics_release", {})
    if original.get("tag") != ORIGINAL_TAG or original.get("commit") != ORIGINAL_COMMIT:
        raise ProvenanceBridgeError(
            "BRIDGE_ORIGINAL_RELEASE_MISMATCH", "original release identity differs"
        )
    if mechanics.get("tag") != MECHANICS_TAG:
        raise ProvenanceBridgeError(
            "BRIDGE_MECHANICS_TAG_MISMATCH", "mechanics-patch tag differs"
        )
    if mechanics.get("commit_binding") != "annotated_tag_peels_to_current_head":
        raise ProvenanceBridgeError(
            "BRIDGE_MECHANICS_BINDING_MISMATCH", "mechanics commit binding differs"
        )
    if tuple(payload.get("reusable_run_ids", [])) != REUSABLE_RUN_IDS:
        raise ProvenanceBridgeError(
            "BRIDGE_RUN_INVENTORY_MISMATCH", "reusable run inventory differs"
        )
    interrupted = payload.get("interrupted_run", {})
    if interrupted.get("run_id") != INTERRUPTED_RUN_ID:
        raise ProvenanceBridgeError(
            "BRIDGE_INTERRUPTED_RUN_MISMATCH", "interrupted run identity differs"
        )
    if interrupted.get("safe_resume_boundary") != SAFE_RESUME_BOUNDARY:
        raise ProvenanceBridgeError(
            "BRIDGE_RESUME_BOUNDARY_MISMATCH", "safe resume boundary differs"
        )


def _validate_release_pair(
    root: Path,
    payload: Mapping[str, Any],
    *,
    current_commit: str,
    current_tag: str,
) -> None:
    _require_exact_bridge_identity(payload)
    if current_tag != MECHANICS_TAG:
        raise ProvenanceBridgeError(
            "BRIDGE_CURRENT_TAG_MISMATCH", f"current tag is {current_tag!r}"
        )
    if _git(root, "rev-list", "-n", "1", ORIGINAL_TAG) != ORIGINAL_COMMIT:
        raise ProvenanceBridgeError(
            "BRIDGE_ORIGINAL_TAG_MOVED", "original annotated tag no longer peels to its commit"
        )
    if _git(root, "cat-file", "-t", f"refs/tags/{ORIGINAL_TAG}") != "tag":
        raise ProvenanceBridgeError("BRIDGE_ORIGINAL_TAG_TYPE", "original tag is not annotated")
    if _git(root, "cat-file", "-t", f"refs/tags/{MECHANICS_TAG}") != "tag":
        raise ProvenanceBridgeError("BRIDGE_MECHANICS_TAG_TYPE", "mechanics tag is not annotated")
    if _git(root, "rev-list", "-n", "1", MECHANICS_TAG) != current_commit:
        raise ProvenanceBridgeError(
            "BRIDGE_MECHANICS_COMMIT_MISMATCH",
            "mechanics tag does not peel to current HEAD",
        )


def _validate_hashed_files(
    root: Path,
    entries: Mapping[str, Any],
    *,
    category: str,
    hash_field: str = "sha256",
) -> None:
    if not entries:
        raise ProvenanceBridgeError(
            f"BRIDGE_{category.upper()}_EMPTY", f"bridge {category} inventory is empty"
        )
    for relative, metadata in sorted(entries.items()):
        path = (root / str(relative)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ProvenanceBridgeError(
                f"BRIDGE_{category.upper()}_MISSING", f"missing {category} file: {relative}"
            )
        expected = metadata.get(hash_field) if isinstance(metadata, Mapping) else metadata
        if sha256_file(path) != expected:
            raise ProvenanceBridgeError(
                f"BRIDGE_{category.upper()}_HASH_MISMATCH",
                f"{category} hash differs: {relative}",
            )
        if isinstance(metadata, Mapping) and "size_bytes" in metadata:
            if path.stat().st_size != int(metadata["size_bytes"]):
                raise ProvenanceBridgeError(
                    f"BRIDGE_{category.upper()}_SIZE_MISMATCH",
                    f"{category} size differs: {relative}",
                )


def _validate_runtime_and_frozen_files(root: Path, payload: Mapping[str, Any]) -> None:
    runtime = payload.get("runtime_files", {})
    current_entries = {
        str(path): {
            "sha256": metadata.get("new_sha256"),
            "size_bytes": metadata.get("new_size_bytes"),
        }
        for path, metadata in runtime.items()
        if isinstance(metadata, Mapping)
    }
    _validate_hashed_files(root, current_entries, category="runtime")
    _validate_hashed_files(
        root,
        payload.get("frozen_files", {}),
        category="frozen",
    )


def _validate_run_artifacts(root: Path, payload: Mapping[str, Any]) -> None:
    runs = payload.get("runs", {})
    expected = (*REUSABLE_RUN_IDS, INTERRUPTED_RUN_ID)
    if tuple(runs) != expected:
        raise ProvenanceBridgeError(
            "BRIDGE_RUN_TABLE_MISMATCH", "bridge run table is incomplete or reordered"
        )
    for run_id in expected:
        entry = runs[run_id]
        run_dir = (root / entry["run_directory"]).resolve()
        if not run_dir.is_relative_to(root / "results" / "runs") or not run_dir.is_dir():
            raise ProvenanceBridgeError(
                "BRIDGE_RUN_DIRECTORY_MISMATCH", f"invalid run directory for {run_id}"
            )
        checkpoint_path = run_dir / "checkpoint.json"
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProvenanceBridgeError(
                "BRIDGE_CHECKPOINT_UNREADABLE", f"{run_id}: {exc}"
            ) from exc
        identity = checkpoint.get("identity", {})
        if identity.get("run_id") != run_id or identity.get("git_commit") != ORIGINAL_COMMIT:
            raise ProvenanceBridgeError(
                "BRIDGE_CHECKPOINT_PROVENANCE_MISMATCH",
                f"checkpoint provenance differs for {run_id}",
            )
        artifacts = entry.get("immutable_artifacts", {})
        for relative, metadata in artifacts.items():
            path = (run_dir / relative).resolve()
            if not path.is_relative_to(run_dir) or not path.is_file():
                raise ProvenanceBridgeError(
                    "BRIDGE_ARTIFACT_MISSING", f"{run_id}: {relative}"
                )
            if path.stat().st_size != int(metadata.get("size_bytes", -1)):
                raise ProvenanceBridgeError(
                    "BRIDGE_ARTIFACT_SIZE_MISMATCH", f"{run_id}: {relative}"
                )
            if sha256_file(path) != metadata.get("sha256"):
                raise ProvenanceBridgeError(
                    "BRIDGE_ARTIFACT_HASH_MISMATCH", f"{run_id}: {relative}"
                )


def authenticate_compatibility_bridge(
    repository_root: str | Path,
    *,
    current_commit: str,
    current_tag: str,
    validate_inventory: bool = True,
) -> dict[str, Any]:
    """Authenticate only the exact old/new release pair and cdv1 inventory."""

    root = Path(repository_root).resolve()
    payload = load_compatibility_bridge(root)
    _validate_release_pair(
        root,
        payload,
        current_commit=current_commit,
        current_tag=current_tag,
    )
    _validate_runtime_and_frozen_files(root, payload)
    cache_key = (str(root), str(current_commit))
    if validate_inventory and cache_key not in _AUTHENTICATED_RELEASES:
        _validate_run_artifacts(root, payload)
    _AUTHENTICATED_RELEASES.add(cache_key)
    return payload


def compatible_resume_identity(
    repository_root: str | Path,
    run_directory: str | Path,
    *,
    current_commit: str,
    current_tag: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Return the historical identity/config only for an authenticated bridge run."""

    root = Path(repository_root).resolve()
    run_dir = Path(run_directory).resolve()
    checkpoint = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    identity = dict(checkpoint.get("identity", {}))
    if identity.get("git_commit") == current_commit:
        return None
    payload = authenticate_compatibility_bridge(
        root,
        current_commit=current_commit,
        current_tag=current_tag,
        validate_inventory=True,
    )
    run_id = run_dir.name
    if run_id not in {*REUSABLE_RUN_IDS, INTERRUPTED_RUN_ID}:
        raise ProvenanceBridgeError(
            "BRIDGE_RUN_NOT_AUTHORIZED", f"bridge does not authorize {run_id}"
        )
    if identity.get("git_commit") != ORIGINAL_COMMIT:
        raise ProvenanceBridgeError(
            "BRIDGE_RESUME_COMMIT_MISMATCH", f"historical commit differs for {run_id}"
        )
    effective_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metadata = {
        "bridge_path": BRIDGE_PATH.as_posix(),
        "bridge_schema_version": BRIDGE_SCHEMA_VERSION,
        "original_commit": ORIGINAL_COMMIT,
        "original_tag": ORIGINAL_TAG,
        "mechanics_commit": current_commit,
        "mechanics_tag": current_tag,
        "authorized_run_id": run_id,
    }
    return identity, effective_config, metadata


__all__ = [
    "BRIDGE_PATH",
    "BRIDGE_SCHEMA_VERSION",
    "INTERRUPTED_RUN_ID",
    "MECHANICS_TAG",
    "ORIGINAL_COMMIT",
    "ORIGINAL_TAG",
    "ProvenanceBridgeError",
    "REUSABLE_RUN_IDS",
    "SAFE_RESUME_BOUNDARY",
    "authenticate_compatibility_bridge",
    "compatible_resume_identity",
    "load_compatibility_bridge",
]
