"""Data-free authentication for the cross-dataset voting evidence package.

This module deliberately has no experiment-runner or dataset-loader imports. It
parses only manifests, structural status metadata, provenance records, and a
predeclared comparison-family header. Payload files are streamed only to compute
byte sizes and SHA-256 digests.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


PACKAGE_ROOT = Path("results/final_experiments/cross_dataset_voting_inference_v1")
LEGACY_MANIFEST = PACKAGE_ROOT / "artifact_manifest.json"
SELECTION_POINTER = PACKAGE_ROOT / "artifact_manifest.current.json"
STATUS_PATH = PACKAGE_ROOT / "status.json"
PROMPT_07_CHECKPOINT = Path(
    "cleanup/audits/prompt_07_lightweight_selectors/"
    "prompt_06_package_hashes_baseline.json"
)
PROMPT_07_MANIFEST = Path(
    "cleanup/audits/prompt_07_lightweight_selectors/artifact_manifest.json"
)
PREDECLARED_FAMILY = PACKAGE_ROOT / "predeclared_comparison_family.json"

LEGACY_MANIFEST_SHA256 = (
    "e16a3cff5135a9eb3ecf92ea635bdeb55772fbe8f80c9dca07086590464de2a2"
)
OLD_STATUS_SHA256 = (
    "8da9422226d4204fccd2912ca2049f3c55b372a0878bf44219e12685fbfa2ddc"
)
CURRENT_STATUS_SHA256 = (
    "4b16d8cea9dd877fe62b687416fe70c6bcd9def50ab8f9425f16cd2aebeb0e8a"
)
BLOCKER_SHA256 = (
    "6e91e017af4791f67992397be5c2620392d6baf354de09cf75c7a84e23485a78"
)
PROMPT_07_CHECKPOINT_SHA256 = (
    "883a4e3dabb9c569a631898fa7c1906ce772e83f42da040c14dfd8ec9e2c2cb6"
)
PROMPT_07_MANIFEST_SHA256 = (
    "a7326af6a0eb426b676fa582f50cdec3e5d1a5d2b70302cfbf06a77a541e795a"
)

EXPECTED_STATUS_KEYS = (
    "schema_version",
    "analysis_id",
    "started_at_utc",
    "phases",
    "blockers",
    "needs_user_action",
    "completed_at_utc",
    "runtime_seconds",
    "peak_process_rss_bytes",
    "package_root",
    "audit_root",
    "comparison_family_count",
    "comparison_count",
    "delong_comparison_count",
    "bootstrap_comparison_count",
    "bootstrap_repetitions",
    "holm_family_count",
    "status",
    "success_marker",
)
EXPECTED_PHASES = (
    "A_frozen_inputs",
    "A_preservation",
    "A_prompt_05_completion",
    "H1_predeclared_family",
    "B_alignment",
    "C_metric_recomputation",
    "D_score_psi",
    "F_stability",
    "G_resources",
    "E_feature_psi",
    "C3_independent_recalculation",
    "H_paired_inference",
    "I_evidence_tables",
    "J_provenance",
)


class ManifestAuthenticationError(ValueError):
    """Raised when a package or its supersession metadata fails closed."""


@dataclass(frozen=True)
class ManifestSelection:
    manifest_path: str
    manifest_sha256: str
    manifest: Mapping[str, Any]
    selection_rule: str
    pointer: Mapping[str, Any] | None
    supersession: Mapping[str, Any] | None


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_record_digest(record: Mapping[str, Any]) -> str:
    projected = dict(record)
    projected.pop("canonical_record_digest", None)
    encoded = json.dumps(
        projected,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestAuthenticationError(message)


def _repo_path(root: Path, relative: str) -> Path:
    _fail(isinstance(relative, str) and relative != "", "invalid empty repository path")
    candidate = (root / Path(relative)).resolve()
    repository = root.resolve()
    try:
        candidate.relative_to(repository)
    except ValueError as error:
        raise ManifestAuthenticationError(
            f"path escapes repository root: {relative}"
        ) from error
    return candidate


def authenticate_manifest_entries(
    root: Path, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    entries = manifest.get("generated_files")
    _fail(isinstance(entries, list), "manifest generated_files must be a list")
    _fail(
        manifest.get("generated_file_count") == len(entries),
        "manifest generated_file_count mismatch",
    )
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        _fail(isinstance(entry, dict), f"manifest entry {index} is not an object")
        relative = entry.get("path")
        expected_size = entry.get("size_bytes")
        expected_hash = entry.get("sha256")
        _fail(isinstance(relative, str), f"manifest entry {index} has invalid path")
        _fail(relative not in seen, f"duplicate manifest path: {relative}")
        seen.add(relative)
        _fail(
            isinstance(expected_size, int) and expected_size >= 0,
            f"manifest entry {relative} has invalid size",
        )
        _fail(
            isinstance(expected_hash, str) and len(expected_hash) == 64,
            f"manifest entry {relative} has invalid SHA-256",
        )
        path = _repo_path(root, relative)
        present = path.is_file()
        actual_size = path.stat().st_size if present else None
        actual_hash = sha256_file(path) if present else None
        results.append(
            {
                "index": index,
                "path": relative,
                "expected_size_bytes": expected_size,
                "observed_size_bytes": actual_size,
                "expected_sha256": expected_hash,
                "observed_sha256": actual_hash,
                "size_match": present and actual_size == expected_size,
                "sha256_match": present and actual_hash == expected_hash,
                "authenticated": present
                and actual_size == expected_size
                and actual_hash == expected_hash,
            }
        )
    return results


def reproduce_legacy_blocker(root: Path) -> dict[str, Any]:
    manifest_path = root / LEGACY_MANIFEST
    _fail(manifest_path.is_file(), f"missing legacy manifest: {LEGACY_MANIFEST.as_posix()}")
    manifest_hash = sha256_file(manifest_path)
    _fail(manifest_hash == LEGACY_MANIFEST_SHA256, "legacy manifest digest mismatch")
    manifest = read_json(manifest_path)
    results = authenticate_manifest_entries(root, manifest)
    mismatches = [entry for entry in results if not entry["authenticated"]]
    _fail(len(results) == 55, "legacy manifest scope is not 55 entries")
    _fail(len(mismatches) == 1, "legacy mismatch count is not exactly one")
    mismatch = mismatches[0]
    _fail(mismatch["path"] == STATUS_PATH.as_posix(), "legacy mismatch is not status.json")
    _fail(mismatch["expected_size_bytes"] == 762, "old status size mismatch")
    _fail(mismatch["observed_size_bytes"] == 1298, "current status size mismatch")
    _fail(mismatch["expected_sha256"] == OLD_STATUS_SHA256, "old status hash mismatch")
    _fail(
        mismatch["observed_sha256"] == CURRENT_STATUS_SHA256,
        "current status hash mismatch",
    )
    return {
        "schema_version": "prompt_14a_blocker_reproduction_v1",
        "package": "cross_dataset_voting_inference_v1",
        "manifest_path": LEGACY_MANIFEST.as_posix(),
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_sha256": manifest_hash,
        "declared_generated_file_count": manifest.get("generated_file_count"),
        "ordered_inventory": results,
        "matching_entries": len(results) - len(mismatches),
        "mismatching_entries": len(mismatches),
        "exact_reported_one_file_mismatch_reproduced": True,
        "validator_identity": "credit_risk_fs.analysis.voting_inference.manifest_authentication:reproduce_legacy_blocker",
        "validator_schema_version": "prompt_14a_blocker_reproduction_v1",
        "data_free_safety": {
            "raw_dataset_path_resolution_implemented": False,
            "worker_start_implemented": False,
            "raw_dataset_paths_resolved": False,
            "workers_started": 0,
        },
    }


def _verify_digest_record(record: Mapping[str, Any], *, label: str) -> None:
    observed = record.get("canonical_record_digest")
    _fail(isinstance(observed, str), f"{label} lacks canonical_record_digest")
    _fail(observed == canonical_record_digest(record), f"{label} canonical digest mismatch")


def _verify_referenced_file(
    root: Path, record: Mapping[str, Any], *, path_key: str, hash_key: str, label: str
) -> Path:
    relative = record.get(path_key)
    expected_hash = record.get(hash_key)
    _fail(isinstance(relative, str), f"{label} path is invalid")
    _fail(isinstance(expected_hash, str), f"{label} hash is invalid")
    path = _repo_path(root, relative)
    _fail(path.is_file(), f"{label} is missing: {relative}")
    _fail(sha256_file(path) == expected_hash, f"{label} digest mismatch: {relative}")
    return path


def select_manifest(root: Path) -> ManifestSelection:
    pointer_path = root / SELECTION_POINTER
    if not pointer_path.is_file():
        legacy_path = root / LEGACY_MANIFEST
        _fail(legacy_path.is_file(), "legacy manifest is missing")
        return ManifestSelection(
            manifest_path=LEGACY_MANIFEST.as_posix(),
            manifest_sha256=sha256_file(legacy_path),
            manifest=read_json(legacy_path),
            selection_rule="legacy_fixed_path_no_successor_pointer",
            pointer=None,
            supersession=None,
        )

    pointer = read_json(pointer_path)
    _fail(
        pointer.get("schema_version") == "prompt_14a_manifest_selection_v1",
        "manifest selection pointer schema mismatch",
    )
    _verify_digest_record(pointer, label="manifest selection pointer")
    _fail(
        pointer.get("selection_rule") == "explicit_successor_pointer_fail_closed",
        "unsupported manifest selection rule",
    )
    _verify_referenced_file(
        root,
        pointer,
        path_key="legacy_manifest_path",
        hash_key="legacy_manifest_sha256",
        label="legacy manifest",
    )
    selected_path = _verify_referenced_file(
        root,
        pointer,
        path_key="selected_manifest_path",
        hash_key="selected_manifest_sha256",
        label="selected manifest",
    )
    supersession_path = _verify_referenced_file(
        root,
        pointer,
        path_key="supersession_record_path",
        hash_key="supersession_record_sha256",
        label="supersession record",
    )
    supersession = read_json(supersession_path)
    _fail(
        supersession.get("schema_version") == "prompt_14a_manifest_supersession_v1",
        "manifest supersession schema mismatch",
    )
    _verify_digest_record(supersession, label="manifest supersession")
    _fail(
        supersession.get("old_manifest_sha256") == pointer.get("legacy_manifest_sha256"),
        "pointer/supersession old manifest mismatch",
    )
    _fail(
        supersession.get("successor_manifest_sha256")
        == pointer.get("selected_manifest_sha256"),
        "pointer/supersession successor mismatch",
    )
    for evidence in supersession.get("provenance_evidence", []):
        _verify_referenced_file(
            root,
            evidence,
            path_key="path",
            hash_key="sha256",
            label="provenance evidence",
        )
    return ManifestSelection(
        manifest_path=str(pointer["selected_manifest_path"]),
        manifest_sha256=str(pointer["selected_manifest_sha256"]),
        manifest=read_json(selected_path),
        selection_rule=str(pointer["selection_rule"]),
        pointer=pointer,
        supersession=supersession,
    )


def validate_status_structure(root: Path, status_hash: str) -> dict[str, Any]:
    status_path = root / STATUS_PATH
    _fail(status_path.is_file(), "status.json is missing")
    _fail(sha256_file(status_path) == status_hash, "status digest is not selected")
    status = read_json(status_path)
    _fail(tuple(status.keys()) == EXPECTED_STATUS_KEYS, "status key set or order mismatch")
    _fail(
        status.get("schema_version") == "prompt_06_voting_inference_status_v1",
        "status schema mismatch",
    )
    _fail(status.get("analysis_id") == "cross_dataset_voting_inference_v1", "analysis id mismatch")
    _fail(tuple(status.get("phases", {}).keys()) == EXPECTED_PHASES, "status phase set mismatch")
    _fail(all(value == "PASS" for value in status["phases"].values()), "status phase failure")
    _fail(status.get("blockers") == [], "status contains blockers")
    _fail(status.get("needs_user_action") == [], "status needs user action")
    _fail(status.get("status") == "PASS", "status is not PASS")
    _fail(
        status.get("success_marker") == "PROMPT_06_VOTING_INFERENCE_EVIDENCE_PACKAGE_PASS",
        "status success marker mismatch",
    )
    _fail(status.get("package_root") == PACKAGE_ROOT.as_posix(), "status package root mismatch")
    _fail(
        status.get("audit_root")
        == "cleanup/audits/prompt_06_voting_inference_evidence_package",
        "status audit root mismatch",
    )

    specification = read_json(root / PREDECLARED_FAMILY)
    _fail(specification.get("constructed_after_viewing_oot_results") is False, "family was not predeclared")
    _fail(status.get("comparison_family_count") == specification.get("family_count"), "family count mismatch")
    _fail(status.get("holm_family_count") == specification.get("family_count"), "Holm family count mismatch")
    _fail(status.get("comparison_count") == specification.get("comparison_count"), "comparison count mismatch")
    _fail(status.get("delong_comparison_count") == specification.get("comparison_count"), "DeLong count mismatch")
    _fail(status.get("bootstrap_comparison_count") == specification.get("comparison_count"), "bootstrap count mismatch")
    _fail(status.get("bootstrap_repetitions") == 2000, "bootstrap repetition identity mismatch")
    return {
        "schema_valid": True,
        "required_keys_valid": True,
        "completion_predicates_valid": True,
        "configuration_identity_valid": True,
        "predeclared_family_consistency_valid": True,
        "predictive_metric_values_read": False,
    }


def validate_status_checkpoint(root: Path, status_hash: str) -> dict[str, Any]:
    checkpoint_path = root / PROMPT_07_CHECKPOINT
    checkpoint_manifest_path = root / PROMPT_07_MANIFEST
    _fail(sha256_file(checkpoint_path) == PROMPT_07_CHECKPOINT_SHA256, "Prompt 7 checkpoint digest mismatch")
    _fail(sha256_file(checkpoint_manifest_path) == PROMPT_07_MANIFEST_SHA256, "Prompt 7 manifest digest mismatch")
    checkpoint_manifest = read_json(checkpoint_manifest_path)
    entry = next(
        (
            item
            for item in checkpoint_manifest.get("artifacts", [])
            if item.get("path") == PROMPT_07_CHECKPOINT.as_posix()
        ),
        None,
    )
    _fail(entry is not None, "Prompt 7 manifest does not inventory the checkpoint")
    _fail(entry.get("sha256") == PROMPT_07_CHECKPOINT_SHA256, "Prompt 7 checkpoint is not authenticated")
    checkpoint = read_json(checkpoint_path)
    _fail(checkpoint.get("status.json") == status_hash, "status is absent from immutable Prompt 7 checkpoint")
    _fail(
        checkpoint.get("artifact_manifest.json") == LEGACY_MANIFEST_SHA256,
        "legacy manifest is absent from immutable Prompt 7 checkpoint",
    )
    return {
        "checkpoint_path": PROMPT_07_CHECKPOINT.as_posix(),
        "checkpoint_sha256": PROMPT_07_CHECKPOINT_SHA256,
        "checkpoint_manifest_path": PROMPT_07_MANIFEST.as_posix(),
        "checkpoint_manifest_sha256": PROMPT_07_MANIFEST_SHA256,
        "status_checkpoint_match": True,
        "legacy_manifest_checkpoint_match": True,
    }


def validate_voting_package(root: Path) -> dict[str, Any]:
    selection = select_manifest(root)
    selected_results = authenticate_manifest_entries(root, selection.manifest)
    _fail(len(selected_results) == 55, "selected manifest scope is not 55 entries")
    _fail(all(entry["authenticated"] for entry in selected_results), "selected manifest payload authentication failed")
    paths = [entry["path"] for entry in selected_results]
    _fail(paths.count(STATUS_PATH.as_posix()) == 1, "selected manifest status scope mismatch")
    selected_status = next(entry for entry in selected_results if entry["path"] == STATUS_PATH.as_posix())
    status_validation = validate_status_structure(root, str(selected_status["observed_sha256"]))
    checkpoint_validation = validate_status_checkpoint(root, str(selected_status["observed_sha256"]))

    legacy = reproduce_legacy_blocker(root)
    unchanged_count = 0
    if selection.supersession is not None:
        old_manifest = read_json(root / LEGACY_MANIFEST)
        old_entries = old_manifest["generated_files"]
        new_entries = selection.manifest["generated_files"]
        _fail(
            [entry["path"] for entry in old_entries]
            == [entry["path"] for entry in new_entries],
            "successor changed ordered payload scope",
        )
        for old, new in zip(old_entries, new_entries, strict=True):
            if old["path"] == STATUS_PATH.as_posix():
                _fail(old["size_bytes"] == 762 and old["sha256"] == OLD_STATUS_SHA256, "old status identity changed")
                _fail(new["size_bytes"] == 1298 and new["sha256"] == CURRENT_STATUS_SHA256, "new status identity mismatch")
            else:
                _fail(old == new, f"unchanged manifest entry drift: {old['path']}")
                unchanged_count += 1
        _fail(unchanged_count == 54, "unchanged entry count mismatch")
        supersession = selection.supersession
        _fail(supersession.get("blocker_audit_sha256") == BLOCKER_SHA256, "blocker audit linkage mismatch")
        _fail(supersession.get("decision_code") == "legitimate_final_status_after_manifest_seal", "repair decision mismatch")
        _fail(supersession.get("old_status_sha256") == OLD_STATUS_SHA256, "supersession old status mismatch")
        _fail(supersession.get("new_status_sha256") == CURRENT_STATUS_SHA256, "supersession new status mismatch")

    return {
        "schema_version": "prompt_14a_repaired_manifest_validation_v1",
        "selected_manifest_path": selection.manifest_path,
        "selected_manifest_sha256": selection.manifest_sha256,
        "selection_rule": selection.selection_rule,
        "ordered_payload_entry_count": len(selected_results),
        "authenticated_payload_entries": len(selected_results),
        "mismatching_payload_entries": 0,
        "unchanged_entries_checked": unchanged_count,
        "unchanged_entries_byte_identical": unchanged_count,
        "legacy_manifest_expected_failure_reproduced": legacy[
            "exact_reported_one_file_mismatch_reproduced"
        ],
        "status_validation": status_validation,
        "status_checkpoint_validation": checkpoint_validation,
        "data_free_safety": {
            "raw_dataset_path_resolution_implemented": False,
            "worker_start_implemented": False,
            "raw_dataset_paths_resolved": False,
            "workers_started": 0,
            "predictive_metric_values_read": False,
        },
    }


def changed_entry_paths(results: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(entry["path"]) for entry in results if not entry["authenticated"]]
