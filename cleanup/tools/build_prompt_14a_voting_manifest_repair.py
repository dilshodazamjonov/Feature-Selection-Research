"""Build the authorized Prompt 14A manifest successor from existing bytes only."""

from __future__ import annotations

import csv
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = Path("cleanup/audits/prompt_14a_voting_manifest_repair")
PACKAGE_ROOT = Path("results/final_experiments/cross_dataset_voting_inference_v1")
LEGACY_MANIFEST = PACKAGE_ROOT / "artifact_manifest.json"
SUCCESSOR_MANIFEST = PACKAGE_ROOT / "artifact_manifest.v2.json"
SELECTION_POINTER = PACKAGE_ROOT / "artifact_manifest.current.json"
SUPERSESSION_RECORD = AUDIT_ROOT / "manifest_supersession.json"
BLOCKER_AUDIT = Path(
    "cleanup/audits/prompt_14_two_dataset_oot_review/authentication_blocker.json"
)
STATUS_PATH = PACKAGE_ROOT / "status.json"

VALIDATOR_PATH = (
    ROOT
    / "src"
    / "credit_risk_fs"
    / "analysis"
    / "voting_inference"
    / "manifest_authentication.py"
)
SPEC = importlib.util.spec_from_file_location("prompt14a_manifest_authentication", VALIDATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load data-free validator: {VALIDATOR_PATH}")
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR
SPEC.loader.exec_module(VALIDATOR)


def fixed_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".prompt14a.partial")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_bytes(path, fixed_json_bytes(payload))


def file_sha(relative: Path) -> str:
    return VALIDATOR.sha256_file(ROOT / relative)


def add_record_digest(payload: dict[str, Any]) -> dict[str, Any]:
    payload["canonical_record_digest"] = VALIDATOR.canonical_record_digest(payload)
    return payload


def evidence(path: str) -> dict[str, str]:
    relative = Path(path)
    return {"path": path, "sha256": file_sha(relative)}


def build_successor(legacy: Mapping[str, Any]) -> dict[str, Any]:
    generated_files: list[dict[str, Any]] = []
    for entry in legacy["generated_files"]:
        replacement = dict(entry)
        if entry["path"] == STATUS_PATH.as_posix():
            replacement["size_bytes"] = 1298
            replacement["sha256"] = VALIDATOR.CURRENT_STATUS_SHA256
        generated_files.append(replacement)

    successor = dict(legacy)
    successor["schema_version"] = "prompt_14a_artifact_manifest_successor_v1"
    successor["generated_files"] = generated_files
    successor["supersedes"] = {
        "manifest_path": LEGACY_MANIFEST.as_posix(),
        "manifest_sha256": VALIDATOR.LEGACY_MANIFEST_SHA256,
        "blocker_audit_path": BLOCKER_AUDIT.as_posix(),
        "blocker_audit_sha256": VALIDATOR.BLOCKER_SHA256,
        "decision_code": "legitimate_final_status_after_manifest_seal",
        "derivation": "existing_authenticated_payload_bytes_only",
    }
    successor["manifest_metadata_outside_payload_scope"] = [
        SUCCESSOR_MANIFEST.as_posix(),
        SELECTION_POINTER.as_posix(),
        SUPERSESSION_RECORD.as_posix(),
    ]
    return successor


def build_supersession(successor_sha256: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "prompt_14a_manifest_supersession_v1",
        "analysis_id": "cross_dataset_voting_inference_v1",
        "old_manifest_path": LEGACY_MANIFEST.as_posix(),
        "old_manifest_sha256": VALIDATOR.LEGACY_MANIFEST_SHA256,
        "old_manifest_preservation": "preserved_in_place_byte_identical",
        "preserved_old_manifest_path": LEGACY_MANIFEST.as_posix(),
        "preserved_old_manifest_sha256": VALIDATOR.LEGACY_MANIFEST_SHA256,
        "successor_manifest_path": SUCCESSOR_MANIFEST.as_posix(),
        "successor_manifest_sha256": successor_sha256,
        "status_path": STATUS_PATH.as_posix(),
        "old_status_size_bytes": 762,
        "old_status_sha256": VALIDATOR.OLD_STATUS_SHA256,
        "new_status_size_bytes": 1298,
        "new_status_sha256": VALIDATOR.CURRENT_STATUS_SHA256,
        "blocker_audit_path": BLOCKER_AUDIT.as_posix(),
        "blocker_audit_sha256": VALIDATOR.BLOCKER_SHA256,
        "decision_code": "legitimate_final_status_after_manifest_seal",
        "provenance_evidence": [
            evidence(f"{AUDIT_ROOT.as_posix()}/entry_validation.json"),
            evidence(f"{AUDIT_ROOT.as_posix()}/blocker_reproduction.json"),
            evidence(f"{AUDIT_ROOT.as_posix()}/status_provenance_review.json"),
            evidence(f"{AUDIT_ROOT.as_posix()}/status_structural_diff.json"),
            evidence(f"{AUDIT_ROOT.as_posix()}/repair_decision.json"),
        ],
        "ordered_payload_entry_count": 55,
        "unchanged_entry_count": 54,
        "unchanged_entries_byte_identical": True,
        "payload_files_rewritten": [],
        "scientific_locks_rewritten": [],
        "result_or_prediction_artifacts_rewritten": [],
        "reports_or_registries_rewritten": [],
        "successor_selection_rule": "explicit_successor_pointer_fail_closed",
        "selection_pointer_path": SELECTION_POINTER.as_posix(),
        "canonical_record_digest_algorithm": "sha256_of_sorted_compact_utf8_json_without_canonical_record_digest",
    }
    return add_record_digest(record)


def build_pointer(successor_sha256: str, supersession_sha256: str) -> dict[str, Any]:
    pointer: dict[str, Any] = {
        "schema_version": "prompt_14a_manifest_selection_v1",
        "analysis_id": "cross_dataset_voting_inference_v1",
        "selection_rule": "explicit_successor_pointer_fail_closed",
        "legacy_manifest_path": LEGACY_MANIFEST.as_posix(),
        "legacy_manifest_sha256": VALIDATOR.LEGACY_MANIFEST_SHA256,
        "selected_manifest_path": SUCCESSOR_MANIFEST.as_posix(),
        "selected_manifest_sha256": successor_sha256,
        "supersession_record_path": SUPERSESSION_RECORD.as_posix(),
        "supersession_record_sha256": supersession_sha256,
        "payload_entry_count": 55,
        "status_path": STATUS_PATH.as_posix(),
        "status_size_bytes": 1298,
        "status_sha256": VALIDATOR.CURRENT_STATUS_SHA256,
        "failure_policy": "do_not_fallback_to_legacy_manifest_when_pointer_exists",
    }
    return add_record_digest(pointer)


def reauthentication_csv(validation: Mapping[str, Any]) -> bytes:
    selected = VALIDATOR.select_manifest(ROOT)
    results = VALIDATOR.authenticate_manifest_entries(ROOT, selected.manifest)
    legacy = json.loads((ROOT / LEGACY_MANIFEST).read_text(encoding="utf-8"))
    old_by_path = {entry["path"]: entry for entry in legacy["generated_files"]}
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "index",
            "path",
            "size_bytes",
            "sha256",
            "authenticated",
            "legacy_entry_byte_identity",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    for entry in results:
        old = old_by_path[entry["path"]]
        legacy_same = (
            old["size_bytes"] == entry["expected_size_bytes"]
            and old["sha256"] == entry["expected_sha256"]
        )
        writer.writerow(
            {
                "index": entry["index"],
                "path": entry["path"],
                "size_bytes": entry["observed_size_bytes"],
                "sha256": entry["observed_sha256"],
                "authenticated": str(entry["authenticated"]).lower(),
                "legacy_entry_byte_identity": (
                    "status_superseded" if entry["path"] == STATUS_PATH.as_posix() else str(legacy_same).lower()
                ),
            }
        )
    if validation["authenticated_payload_entries"] != 55:
        raise RuntimeError("refusing to write reauthentication CSV before 55/55 validation")
    return buffer.getvalue().encode("utf-8")


def build_closure(supersession_sha256: str) -> dict[str, Any]:
    closure: dict[str, Any] = {
        "schema_version": "prompt_14a_blocked_attempt_closure_v1",
        "blocked_plan_commit": "bf2e53e3ea3fbe91e1336019f33b8425287d4564",
        "blocked_plan_sha256": "baa3616fec0e4a8498b018fe8d17c90c141d4fd1169bd2d66cb6252c83156901",
        "frozen_comparison_count": 124,
        "frozen_holm_family_count": 36,
        "comparisons_actually_analyzed": 0,
        "blocker_audit_path": BLOCKER_AUDIT.as_posix(),
        "blocker_audit_sha256": VALIDATOR.BLOCKER_SHA256,
        "repair_supersession_record_path": SUPERSESSION_RECORD.as_posix(),
        "repair_supersession_record_sha256": supersession_sha256,
        "state": "closed_blocked_attempt_not_resumable",
        "historical_plan_rule": "The prior plan and its commit remain historical evidence and must not be reused as the active plan.",
        "fresh_attempt_rule": "The next Prompt 14 attempt must begin again at Phase 0 and create, validate, and commit a new preinspection plan and digest before numeric outcome inspection.",
        "design_reproduction_rule": "The next plan may reproduce the 124-comparison and 36-Holm-family design from protocol evidence, but may not change it based on outcomes or this artifact repair.",
        "new_preinspection_plan_created_by_prompt_14a": False,
    }
    return add_record_digest(closure)


def main() -> int:
    legacy_path = ROOT / LEGACY_MANIFEST
    status_path = ROOT / STATUS_PATH
    blocker_path = ROOT / BLOCKER_AUDIT
    decision_path = ROOT / AUDIT_ROOT / "repair_decision.json"
    if file_sha(LEGACY_MANIFEST) != VALIDATOR.LEGACY_MANIFEST_SHA256:
        raise RuntimeError("legacy manifest precondition failed")
    if status_path.stat().st_size != 1298 or file_sha(STATUS_PATH) != VALIDATOR.CURRENT_STATUS_SHA256:
        raise RuntimeError("current status precondition failed")
    if file_sha(BLOCKER_AUDIT) != VALIDATOR.BLOCKER_SHA256:
        raise RuntimeError("blocker audit precondition failed")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if not decision.get("repair_authorized"):
        raise RuntimeError("repair decision does not authorize a successor")
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    reproduction = VALIDATOR.reproduce_legacy_blocker(ROOT)
    if not reproduction["exact_reported_one_file_mismatch_reproduced"]:
        raise RuntimeError("legacy blocker no longer reproduces")

    payload_before = {
        entry["path"]: VALIDATOR.sha256_file(ROOT / entry["path"])
        for entry in legacy["generated_files"]
    }
    successor = build_successor(legacy)
    atomic_write_json(ROOT / SUCCESSOR_MANIFEST, successor)
    successor_sha256 = file_sha(SUCCESSOR_MANIFEST)

    supersession = build_supersession(successor_sha256)
    atomic_write_json(ROOT / SUPERSESSION_RECORD, supersession)
    supersession_sha256 = file_sha(SUPERSESSION_RECORD)

    pointer = build_pointer(successor_sha256, supersession_sha256)
    atomic_write_json(ROOT / SELECTION_POINTER, pointer)

    validation = VALIDATOR.validate_voting_package(ROOT)
    payload_after = {
        entry["path"]: VALIDATOR.sha256_file(ROOT / entry["path"])
        for entry in legacy["generated_files"]
    }
    if payload_before != payload_after:
        raise RuntimeError("payload bytes changed during manifest repair")
    if file_sha(LEGACY_MANIFEST) != VALIDATOR.LEGACY_MANIFEST_SHA256:
        raise RuntimeError("legacy manifest was not preserved")

    validation = dict(validation)
    validation.update(
        {
            "legacy_manifest_preserved_in_place": True,
            "legacy_manifest_sha256": VALIDATOR.LEGACY_MANIFEST_SHA256,
            "payload_hash_snapshot_before_equals_after": True,
            "blocker_audit_sha256": VALIDATOR.BLOCKER_SHA256,
            "raw_dataset_paths_resolved": False,
            "workers_started": 0,
        }
    )
    atomic_write_json(ROOT / AUDIT_ROOT / "repaired_manifest_validation.json", validation)
    atomic_write_bytes(
        ROOT / AUDIT_ROOT / "artifact_reauthentication.csv",
        reauthentication_csv(validation),
    )
    atomic_write_json(
        ROOT / AUDIT_ROOT / "blocked_prompt_14_attempt_closure.json",
        build_closure(supersession_sha256),
    )
    print(
        json.dumps(
            {
                "successor_manifest_sha256": successor_sha256,
                "supersession_record_sha256": supersession_sha256,
                "authenticated_payload_entries": validation[
                    "authenticated_payload_entries"
                ],
                "unchanged_entries_byte_identical": validation[
                    "unchanged_entries_byte_identical"
                ],
                "raw_dataset_paths_resolved": False,
                "workers_started": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
