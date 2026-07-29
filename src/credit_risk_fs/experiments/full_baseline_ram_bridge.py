"""Fail-closed provenance bridge for the one pre-patch full-baseline RAM stop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from credit_risk_fs.experiments.atomic_io import sha256_file


BRIDGE_SCHEMA_VERSION = "full_baseline_ram_wait_compatibility_v1"
DEFAULT_BRIDGE_PATH = Path(
    "configs/execution/full_baseline_ram_wait_compatibility_v1.json"
)


def authenticate_full_baseline_ram_bridge(
    repository_root: str | Path,
    *,
    run_id: str,
    checkpoint: Mapping[str, Any],
    full_baseline_configuration_sha256: str,
    bridge_path: str | Path = DEFAULT_BRIDGE_PATH,
) -> dict[str, Any]:
    """Authenticate exact old identity plus exact frozen/current file hashes."""

    root = Path(repository_root).resolve()
    supplied = Path(bridge_path)
    path = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != BRIDGE_SCHEMA_VERSION:
        raise ValueError("full-baseline RAM compatibility bridge schema mismatch")
    if str(run_id) not in set(map(str, payload.get("eligible_run_ids", []))):
        raise ValueError("run is outside the RAM compatibility bridge scope")
    if checkpoint.get("status") == "completed":
        raise ValueError("completed checkpoints are outside the RAM bridge scope")
    eligible_codes = set(payload.get("eligible_stop_codes", []))
    prior_attempts = [
        {
            "status": checkpoint.get("status"),
            "stop_code": checkpoint.get("stop_code"),
        },
        *list(checkpoint.get("attempt_history", [])),
    ]
    if not any(
        item.get("status") == "aborted_resource_limit"
        and item.get("stop_code") in eligible_codes
        for item in prior_attempts
    ):
        raise ValueError("checkpoint has no authenticated historical RAM stop")
    identity = checkpoint.get("identity", {})
    if identity.get("git_commit") != payload.get("predecessor_commit"):
        raise ValueError("RAM compatibility predecessor commit mismatch")
    expected_identity = payload.get("interrupted_identity", {})
    for field, expected in expected_identity.items():
        if identity.get(field) != expected:
            raise ValueError(f"RAM compatibility identity mismatch: {field}")
    if (
        str(full_baseline_configuration_sha256)
        != payload.get("full_baseline_configuration_sha256")
    ):
        raise ValueError("full-baseline frozen configuration changed")
    for section in ("frozen_scientific_files", "runtime_mechanics_files"):
        files = payload.get(section, {})
        if not isinstance(files, Mapping) or not files:
            raise ValueError(f"RAM compatibility bridge section is empty: {section}")
        for relative, expected_hash in files.items():
            candidate = (root / str(relative)).resolve()
            if not candidate.is_relative_to(root) or not candidate.is_file():
                raise ValueError(f"RAM compatibility file is missing: {relative}")
            if sha256_file(candidate) != str(expected_hash):
                raise ValueError(f"RAM compatibility file hash mismatch: {relative}")
    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "bridge_path": path.relative_to(root).as_posix(),
        "bridge_sha256": sha256_file(path),
        "predecessor_commit": payload["predecessor_commit"],
        "scope": "RAM supervision and cooperative loading mechanics only",
    }


__all__ = [
    "BRIDGE_SCHEMA_VERSION",
    "DEFAULT_BRIDGE_PATH",
    "authenticate_full_baseline_ram_bridge",
]
