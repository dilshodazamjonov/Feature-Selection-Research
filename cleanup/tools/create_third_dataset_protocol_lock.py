from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


APPROVED_REVIEW_DIGEST = "3f537d1b5e79faad3a2f047ec13dbe4b1797e11d4d64c4d92a06e09762a53f1e"
STAGE_1_COMMIT = "ba4e46d4b545f395f25e1e3c094c16ace7d144aa"
PROTOCOL_ID = "homecredit_model_stability_2024_v1"
APPROVAL_TEXT = (
    "I explicitly approve third-dataset protocol review digest "
    f"{APPROVED_REVIEW_DIGEST} and the exact dataset identity, depth-0/depth-1 "
    "inclusion with depth-2 exclusion, temporal split and five folds, adapter, "
    "leakage/availability decisions, method matrix, and preregistered analysis scope "
    "listed in the Stage 1 review."
)
APPROVED_COMBINATION_ORDER = [
    "statistical_normalized_average_rank",
    "iv_then_boruta",
    "boruta_then_mrmr_mutual_information",
    "boruta_then_rfe_catboost",
]
PROTECTED_CHANGE_CLASSES = [
    "dataset_identity_or_included_raw_file_hash",
    "depth_or_raw_file_scope",
    "temporal_boundary_or_membership",
    "adapter_or_feature_scope",
    "method_or_variant_order",
    "feature_or_pool_budget",
    "model_or_bootstrap_seed",
    "metric_or_inference_rule",
]


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def repo_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def artifact_binding(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": repo_relative(root, path),
        "byte_size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def self_authenticate(payload: dict[str, Any]) -> dict[str, Any]:
    authenticated = copy.deepcopy(payload)
    authenticated.pop("artifact_authentication_sha256", None)
    authenticated["artifact_authentication_sha256"] = sha256_bytes(canonical_bytes(authenticated))
    return authenticated


def verify_self_authentication(payload: dict[str, Any]) -> bool:
    supplied = payload.get("artifact_authentication_sha256")
    unsigned = copy.deepcopy(payload)
    unsigned.pop("artifact_authentication_sha256", None)
    return isinstance(supplied, str) and sha256_bytes(canonical_bytes(unsigned)) == supplied


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_repository_identity(root: Path) -> dict[str, object]:
    branch = _run_git(root, "branch", "--show-current")
    commit = _run_git(root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", STAGE_1_COMMIT, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0
    if branch != "main":
        raise RuntimeError(f"expected branch main, observed {branch!r}")
    if not ancestor:
        raise RuntimeError(f"Stage 1 commit {STAGE_1_COMMIT} is not in current history")
    return {"branch": branch, "commit": commit, "stage_1_commit_is_ancestor": ancestor}


def stage_1_paths(root: Path) -> dict[str, Path]:
    audit = root / "cleanup/audits/third_dataset_protocol_freeze"
    return {
        "dataset_identity": audit / "dataset_identity.json",
        "raw_file_inventory": audit / "raw_file_inventory.csv",
        "schema_and_cardinality_profile": audit / "schema_and_cardinality_profile.csv",
        "feature_definition_coverage": audit / "feature_definition_coverage.json",
        "temporal_population_profile": audit / "temporal_population_profile.csv",
        "split_and_fold_boundaries": audit / "proposed_split_and_fold_boundaries.json",
        "leakage_and_availability_review": audit / "leakage_and_availability_review.csv",
        "adapter_protocol": audit / "proposed_adapter_protocol.json",
        "method_matrix": audit / "proposed_method_matrix.json",
        "preregistered_analysis": audit / "preregistered_hypotheses_and_analysis.md",
        "protocol_review": audit / "protocol_review.md",
        "review_digest": audit / "review_digest.json",
    }


def authenticate_stage_1(root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    paths = stage_1_paths(root)
    digest = read_json(paths["review_digest"])
    supplied = digest.get("review_digest_sha256")
    unsigned = copy.deepcopy(digest)
    unsigned.pop("review_digest_sha256", None)
    observed = sha256_bytes(canonical_bytes(unsigned))
    if supplied != APPROVED_REVIEW_DIGEST or observed != APPROVED_REVIEW_DIGEST:
        raise RuntimeError(
            f"Stage 1 review digest mismatch: approved={APPROVED_REVIEW_DIGEST}, "
            f"declared={supplied}, observed={observed}"
        )
    for binding in digest["artifact_hashes"]:
        path = root / binding["path"]
        if not path.is_file():
            raise RuntimeError(f"Stage 1 artifact missing: {binding['path']}")
        if path.stat().st_size != binding["byte_size"] or sha256_file(path) != binding["sha256"]:
            raise RuntimeError(f"Stage 1 artifact authentication failed: {binding['path']}")
    inventory = read_csv(paths["raw_file_inventory"])
    return digest, inventory


def verify_included_raw_files(root: Path, inventory: list[dict[str, str]]) -> dict[str, object]:
    included = [row for row in inventory if row["inclusion_status"] == "included"]
    mismatches: list[dict[str, object]] = []
    identity: list[dict[str, object]] = []
    for row in included:
        path = root / row["relative_path"]
        expected_size = int(row["byte_size"])
        if not path.is_file():
            mismatches.append({"path": row["relative_path"], "reason": "missing"})
            continue
        observed_size = path.stat().st_size
        observed_sha = sha256_file(path)
        if observed_size != expected_size or observed_sha != row["sha256"]:
            mismatches.append(
                {
                    "path": row["relative_path"],
                    "reason": "size_or_sha256_mismatch",
                    "expected_size": expected_size,
                    "observed_size": observed_size,
                    "expected_sha256": row["sha256"],
                    "observed_sha256": observed_sha,
                }
            )
        identity.append(
            {
                "relative_path": row["relative_path"],
                "byte_size": expected_size,
                "sha256": row["sha256"],
            }
        )
    observed_input_digest = sha256_bytes(canonical_bytes(identity))
    if mismatches:
        raise RuntimeError(f"included raw-file authentication failed: {mismatches}")
    if observed_input_digest != "8adb1db82c9dafb662657db08fd7d1dcf2eb4794d5ff7925e9ca4dd25f73fad2":
        raise RuntimeError(f"included raw-input digest changed: {observed_input_digest}")
    return {
        "verification_mode": "streaming_bytes_only_no_dataset_parse",
        "included_files_expected": len(included),
        "included_files_matched": len(included),
        "mismatches": 0,
        "included_raw_input_digest": observed_input_digest,
    }


def build_approved_protocol(root: Path) -> dict[str, Any]:
    paths = stage_1_paths(root)
    inventory = read_csv(paths["raw_file_inventory"])
    leakage_rows = read_csv(paths["leakage_and_availability_review"])
    matrix = read_json(paths["method_matrix"])
    return {
        "dataset_identity": read_json(paths["dataset_identity"]),
        "raw_file_scope": {
            "records": inventory,
            "included_count": sum(row["inclusion_status"] == "included" for row in inventory),
            "included_parquet_count": sum(
                row["inclusion_status"] == "included" and row["file_type"] == "parquet"
                for row in inventory
            ),
            "excluded_depth_2_parquet_count": sum(
                row["inclusion_status"] == "excluded"
                and row["file_type"] == "parquet"
                and row["relational_depth"] == "2"
                for row in inventory
            ),
        },
        "schema_and_cardinality_profile": read_csv(paths["schema_and_cardinality_profile"]),
        "feature_definition_coverage": read_json(paths["feature_definition_coverage"]),
        "temporal_population_profile": read_csv(paths["temporal_population_profile"]),
        "split_and_fold_boundaries": read_json(paths["split_and_fold_boundaries"]),
        "adapter_protocol": read_json(paths["adapter_protocol"]),
        "leakage_and_availability_scope": {
            "records": leakage_rows,
            "candidate_rows": len(leakage_rows),
            "included": sum(row["action"] == "include" for row in leakage_rows),
            "excluded": sum(row["action"] == "exclude" for row in leakage_rows),
            "unresolved": sum(row["action"] == "unresolved" for row in leakage_rows),
        },
        "method_and_evaluation_matrix": matrix,
        "preregistered_hypotheses_and_analysis": {
            "format": "markdown_utf8",
            "text": paths["preregistered_analysis"].read_text(encoding="utf-8"),
            "sha256": sha256_file(paths["preregistered_analysis"]),
        },
    }


def protected_fingerprints(protocol: dict[str, Any]) -> dict[str, str]:
    matrix = protocol["method_and_evaluation_matrix"]
    protected = {
        "dataset_identity_or_included_raw_file_hash": {
            "dataset_id": protocol["dataset_identity"]["dataset_id"],
            "official_dataset_name": protocol["dataset_identity"]["official_dataset_name"],
            "included_raw_input_digest": protocol["dataset_identity"]["included_raw_input_digest"],
            "included_records": [
                row for row in protocol["raw_file_scope"]["records"]
                if row["inclusion_status"] == "included"
            ],
        },
        "depth_or_raw_file_scope": protocol["raw_file_scope"],
        "temporal_boundary_or_membership": protocol["split_and_fold_boundaries"],
        "adapter_or_feature_scope": {
            "adapter": protocol["adapter_protocol"],
            "leakage": protocol["leakage_and_availability_scope"],
        },
        "method_or_variant_order": {
            "method_order": matrix["method_order"],
            "combination_order": matrix["combination_order"],
            "variant_order": matrix["variant_order"],
            "matrix_cells": matrix["matrix_cells"],
        },
        "feature_or_pool_budget": {
            "feature_budgets": matrix["feature_budgets"],
            "iv_pool_budgets": matrix["iv_pool_budgets"],
            "iv_pool_primary": matrix["iv_pool_primary"],
            "matrix_cell_budgets": [
                {
                    "configuration_order": cell["configuration_order"],
                    "requested_feature_budget": cell["requested_feature_budget"],
                    "iv_pool_budget": cell["iv_pool_budget"],
                    "feature_budget_semantics": cell["feature_budget_semantics"],
                }
                for cell in matrix["matrix_cells"]
            ],
        },
        "model_or_bootstrap_seed": matrix["seeds"],
        "metric_or_inference_rule": {
            "metrics": matrix["metrics"],
            "inference": matrix["inference"],
        },
    }
    return {name: sha256_bytes(canonical_bytes(value)) for name, value in protected.items()}


def build_lock(root: Path, *, created_at_utc: str) -> dict[str, Any]:
    digest, _ = authenticate_stage_1(root)
    protocol = build_approved_protocol(root)
    paths = stage_1_paths(root)
    source_path = root / "cleanup/tools/create_third_dataset_protocol_lock.py"
    test_path = root / "tests/test_third_dataset_protocol_lock.py"
    payload: dict[str, Any] = {
        "schema_version": "third_dataset_protocol_lock_v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_version": 1,
        "status": "canonical_approved_locked_no_execution",
        "created_at_utc": created_at_utc,
        "hash_algorithm": "sha256",
        "canonical_serialization": "UTF-8 JSON sort_keys=true separators=(',',':') ensure_ascii=false allow_nan=false",
        "self_authentication_rule": "artifact_authentication_sha256 hashes this JSON object with only artifact_authentication_sha256 omitted",
        "stage_1_authentication": {
            "approved_review_digest_sha256": APPROVED_REVIEW_DIGEST,
            "observed_review_digest_sha256": digest["review_digest_sha256"],
            "review_digest_path": repo_relative(root, paths["review_digest"]),
            "review_digest_file_sha256": sha256_file(paths["review_digest"]),
            "stage_1_commit": STAGE_1_COMMIT,
            "all_declared_artifacts_authenticated": True,
            "artifact_bindings": digest["artifact_hashes"],
        },
        "user_approval_record": {
            "approval_kind": "explicit_user_approval",
            "approval_stage": "Pre-Prompt 14 Stage 2",
            "recorded_at_utc": created_at_utc,
            "approved_review_digest_sha256": APPROVED_REVIEW_DIGEST,
            "approval_text": APPROVAL_TEXT,
            "received_transport_note": "A trailing Markdown escape backslash followed the sentence; it is not part of the approved scientific scope.",
            "explicitly_approved_scope": [
                "dataset_identity",
                "depth_0_and_depth_1_inclusion_with_depth_2_exclusion",
                "temporal_split_and_five_folds",
                "adapter_protocol",
                "leakage_and_availability_decisions",
                "method_matrix",
                "preregistered_analysis_scope",
            ],
        },
        "approved_protocol": protocol,
        "protected_contract": {
            "change_policy": "fail_closed; any protected change requires a versioned amendment written and authenticated before the affected result is inspected",
            "protected_change_classes": PROTECTED_CHANGE_CLASSES,
            "fingerprints_sha256": protected_fingerprints(protocol),
            "exact_approved_combination_order": APPROVED_COMBINATION_ORDER,
        },
        "validation_implementation": {
            "creator_validator": artifact_binding(root, source_path),
            "focused_tests": artifact_binding(root, test_path),
        },
        "gates": {
            "prompt_14": "next_required_manual_scientific_step_not_run_in_stage_2",
            "adapter": "specification_frozen_not_implemented",
            "pilot": "closed_until_Prompt_14_implements_and_validates_adapter_then_authorizes_bounded_manual_pilot",
            "dev": "closed_until_later_authenticated_pilot_review_and_explicit_approval",
            "oot": "closed_until_later_authenticated_complete_DEV_review_and_explicit_approval",
        },
        "execution_boundary": {
            "prompt_14_run": False,
            "adapter_implemented": False,
            "third_dataset_pilot_started_or_resumed": False,
            "third_dataset_dev_started": False,
            "third_dataset_oot_started": False,
            "existing_two_dataset_oot_metric_artifacts_opened": False,
            "model_or_selector_fit_run": False,
            "network_accessed": False,
            "raw_file_operation": "streaming SHA-256 verification only; no dataset load or modification",
        },
        "amendment": {
            "present": False,
            "required_before_any_protected_change": True,
            "minimum_binding": [
                "parent_lock_file_sha256",
                "reason",
                "changed_fields",
                "new_authenticated_values",
                "created_before_affected_result_inspection",
                "explicit_user_approval",
            ],
        },
    }
    return self_authenticate(payload)


def _specific_contract_errors(payload: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    observed_protocol = payload.get("approved_protocol")
    expected_protocol = expected["approved_protocol"]
    if not isinstance(observed_protocol, dict):
        return ["approved protocol is missing or invalid"]
    observed_fingerprints = protected_fingerprints(observed_protocol)
    expected_fingerprints = protected_fingerprints(expected_protocol)
    for change_class in PROTECTED_CHANGE_CLASSES:
        if observed_fingerprints[change_class] != expected_fingerprints[change_class]:
            errors.append(f"protected contract changed without versioned amendment: {change_class}")
    return errors


def validate_lock_payload(root: Path, payload: dict[str, Any], *, verify_raw: bool) -> list[str]:
    errors: list[str] = []
    try:
        authenticate_stage_1(root)
    except Exception as exc:  # validation must report rather than weaken a gate
        errors.append(str(exc))
    expected = build_lock(root, created_at_utc=str(payload.get("created_at_utc", "")))
    if payload.get("schema_version") != "third_dataset_protocol_lock_v1":
        errors.append("schema version changed")
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("protocol_version") != 1:
        errors.append("protocol identity or version changed")
    approval = payload.get("user_approval_record", {})
    if approval.get("approved_review_digest_sha256") != APPROVED_REVIEW_DIGEST:
        errors.append("approved review digest changed")
    if approval.get("approval_text") != APPROVAL_TEXT:
        errors.append("approval text changed")
    errors.extend(_specific_contract_errors(payload, expected))
    if payload.get("protected_contract") != expected["protected_contract"]:
        errors.append("protected contract fingerprints or amendment policy changed")
    if payload.get("gates") != expected["gates"]:
        errors.append("execution gates changed")
    if payload.get("execution_boundary") != expected["execution_boundary"]:
        errors.append("execution boundary changed")
    if payload.get("stage_1_authentication") != expected["stage_1_authentication"]:
        errors.append("Stage 1 artifact bindings changed")
    if payload.get("validation_implementation") != expected["validation_implementation"]:
        errors.append("validation implementation bindings changed")
    if payload.get("approved_protocol") != expected["approved_protocol"]:
        errors.append("approved protocol differs from the authenticated Stage 1 package")
    if not verify_self_authentication(payload):
        errors.append("lock self-authentication failed")
    if verify_raw:
        try:
            _, inventory = authenticate_stage_1(root)
            verify_included_raw_files(root, inventory)
        except Exception as exc:
            errors.append(str(exc))
    return errors


def build_validation_report(
    root: Path,
    lock: dict[str, Any],
    raw_verification: dict[str, object],
    repository_identity: dict[str, object],
    *,
    created_at_utc: str,
) -> dict[str, Any]:
    lock_path = root / f"configs/protocols/{PROTOCOL_ID}/third_dataset_protocol_lock.json"
    payload: dict[str, Any] = {
        "schema_version": "third_dataset_protocol_stage_2_validation_v1",
        "created_at_utc": created_at_utc,
        "source_branch": repository_identity["branch"],
        "source_commit": STAGE_1_COMMIT,
        "stage_1_commit": STAGE_1_COMMIT,
        "approved_review_digest_sha256": APPROVED_REVIEW_DIGEST,
        "precreation_checks": {
            "stage_1_commit_is_ancestor": repository_identity["stage_1_commit_is_ancestor"],
            "stage_1_package_authenticated": True,
            "review_digest_matches_explicit_approval": True,
            "approved_scope_matches_stage_1_review": True,
            "unrelated_worktree_changes": 0,
            "active_experiment_workers": 0,
            "execution_locks": 0,
        },
        "raw_reauthentication": raw_verification,
        "canonical_lock": {
            "path": repo_relative(root, lock_path),
            "file_sha256": sha256_file(lock_path),
            "artifact_authentication_sha256": lock["artifact_authentication_sha256"],
            "protected_change_classes": PROTECTED_CHANGE_CLASSES,
        },
        "gate_validation": lock["gates"],
        "execution_boundary": lock["execution_boundary"],
        "validation_results": [
            "Stage 1 digest and every declared Stage 1 artifact authenticated",
            "19/19 included raw files matched expected byte size and streaming SHA-256",
            "dataset/depth, temporal membership, adapter/feature scope, method order, budgets, seeds, and metrics are fail-closed",
            "Prompt 14, adapter implementation, pilot, DEV, and OOT were not run",
            "two-dataset OOT metric-bearing artifacts were not opened",
        ],
    }
    return self_authenticate(payload)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def create(root: Path) -> dict[str, object]:
    repository_identity = verify_repository_identity(root)
    digest, inventory = authenticate_stage_1(root)
    if digest["review_digest_sha256"] != APPROVED_REVIEW_DIGEST:
        raise RuntimeError("explicit approval does not match the current Stage 1 review")
    raw_verification = verify_included_raw_files(root, inventory)
    lock_path = root / f"configs/protocols/{PROTOCOL_ID}/third_dataset_protocol_lock.json"
    report_path = root / "cleanup/audits/third_dataset_protocol_approval/stage_2_validation.json"
    if lock_path.exists() or report_path.exists():
        raise RuntimeError("canonical Stage 2 artifact already exists; use --validate, never overwrite the lock")
    created_at_utc = utc_now()
    lock = build_lock(root, created_at_utc=created_at_utc)
    write_json(lock_path, lock)
    errors = validate_lock_payload(root, lock, verify_raw=False)
    if errors:
        raise RuntimeError(f"new lock validation failed: {errors}")
    report = build_validation_report(
        root,
        lock,
        raw_verification,
        repository_identity,
        created_at_utc=created_at_utc,
    )
    write_json(report_path, report)
    return {
        "valid": True,
        "lock_path": repo_relative(root, lock_path),
        "lock_file_sha256": sha256_file(lock_path),
        "lock_authentication_sha256": lock["artifact_authentication_sha256"],
        "validation_path": repo_relative(root, report_path),
        "validation_file_sha256": sha256_file(report_path),
        "included_raw_files_authenticated": raw_verification["included_files_matched"],
        "included_raw_input_digest": raw_verification["included_raw_input_digest"],
    }


def validate(root: Path, *, verify_raw: bool) -> dict[str, object]:
    verify_repository_identity(root)
    lock_path = root / f"configs/protocols/{PROTOCOL_ID}/third_dataset_protocol_lock.json"
    report_path = root / "cleanup/audits/third_dataset_protocol_approval/stage_2_validation.json"
    if not lock_path.is_file():
        return {"valid": False, "errors": [f"missing lock: {repo_relative(root, lock_path)}"]}
    lock = read_json(lock_path)
    errors = validate_lock_payload(root, lock, verify_raw=verify_raw)
    if not report_path.is_file():
        errors.append(f"missing validation report: {repo_relative(root, report_path)}")
    else:
        report = read_json(report_path)
        if not verify_self_authentication(report):
            errors.append("Stage 2 validation-report self-authentication failed")
        canonical = report.get("canonical_lock", {})
        if canonical.get("file_sha256") != sha256_file(lock_path):
            errors.append("Stage 2 validation report has the wrong lock file SHA-256")
        if canonical.get("artifact_authentication_sha256") != lock.get("artifact_authentication_sha256"):
            errors.append("Stage 2 validation report has the wrong lock authentication SHA-256")
    return {
        "valid": not errors,
        "errors": errors,
        "lock_path": repo_relative(root, lock_path),
        "lock_file_sha256": sha256_file(lock_path),
        "lock_authentication_sha256": lock.get("artifact_authentication_sha256"),
        "raw_files_verified": verify_raw,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or validate the approved third-dataset canonical protocol lock without loading a dataset or running research workloads."
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create", action="store_true")
    action.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--verify-raw",
        action="store_true",
        help="Stream and hash the 19 included files; never parses Parquet/CSV contents.",
    )
    args = parser.parse_args()
    root = args.repository_root.resolve()
    try:
        result = create(root) if args.create else validate(root, verify_raw=args.verify_raw)
    except Exception as exc:
        result = {"valid": False, "errors": [str(exc)]}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
