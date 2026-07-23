"""Validate central registries, manifests, canonical paths, and pending inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd


REGISTRY_FILES = (
    "run_index.csv",
    "artifact_registry.csv",
    "reusable_metrics.csv",
    "selected_feature_registry.csv",
)
SUMMARY_PAYLOADS = (*REGISTRY_FILES, "results_access_guide.md")
PENDING_PATTERNS = {
    "significance": (
        "**/cv_results.csv",
        "**/paired_fold_comparisons.csv",
        "**/paired_fold_evidence.csv",
        "**/*fold*manifest*.json",
        "**/selected_features*.csv",
    ),
    "feature_level_drift": (
        "results/homecredit/**/*psi*.csv",
        "results/homecredit/**/*drift*.csv",
        "results/lendingclub_v2/**/*psi*.csv",
        "results/lendingclub_v2/**/*drift*.csv",
    ),
    "llm_cost_scalability": (
        "artifacts/llm_cache/*.json",
        "**/llm_call_summary.csv",
        "**/full_llm_cache_summary.csv",
        "**/*runtime*.json",
        "**/*runtime*.csv",
        "**/*token*.json",
        "**/*token*.csv",
    ),
}
REMOVED_PATHS = (
    "results/final_research_package",
    "results/corrected_lendingclub_to_homecredit_transfer_failed_20260629_140834",
    "results/corrected_lendingclub_to_homecredit_transfer_implementation_backup",
    "results/corrected_lendingclub_to_homecredit_transfer/downstream/"
    "logistic_regression.incomplete_oof_overlap_20260629",
    "results/clip_pairing_repair/smoke",
    "results/clip_v2/contrastive_data",
    "results/clip_v2/dry_run",
    "results/clip_v2/final_analysis",
    "results/clip_v2/final_evaluation",
    "results/clip_v2/selector_integration",
    "results/clip_v2/text_baseline",
    "results/clip_v2/training",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_package(root: Path) -> dict[str, int]:
    package_root = root / "results" / "final_research_package_v2"
    manifest_path = package_root / "final_package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checked = 0
    for section in ("generated_files", "source_artifacts"):
        for entry in manifest[section]:
            path = root / entry["path"]
            if not path.is_file():
                raise ValueError(f"final package path missing: {entry['path']}")
            if path.stat().st_size != int(entry["size_bytes"]):
                raise ValueError(f"final package size mismatch: {entry['path']}")
            if sha256_file(path) != entry["sha256"]:
                raise ValueError(f"final package hash mismatch: {entry['path']}")
            checked += 1
    if manifest.get("source_audit_status") != "passed":
        raise ValueError("final package source audit is not passed")
    return {
        "package_generated_files": len(manifest["generated_files"]),
        "package_source_artifacts": len(manifest["source_artifacts"]),
        "package_paths_verified": checked,
    }


def validate_pending(root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task, patterns in PENDING_PATTERNS.items():
        matches = {
            path.resolve()
            for pattern in patterns
            for path in root.glob(pattern)
            if path.is_file()
        }
        counts[task] = len(matches)
        if not matches:
            raise ValueError(f"no preserved inputs found for pending task: {task}")
    return counts


def validate_canonical_manifest(root: Path) -> dict[str, int]:
    path = root / "results/finalized_research/canonical_artifact_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    if manifest["artifact_count"] != len(artifacts):
        raise ValueError("canonical artifact count mismatch")
    seen: set[str] = set()
    for entry in artifacts:
        relative = entry["path"]
        if relative in seen:
            raise ValueError(f"duplicate canonical path: {relative}")
        seen.add(relative)
        artifact = root / relative
        if not artifact.is_file():
            raise ValueError(f"canonical artifact missing: {relative}")
        if artifact.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"canonical artifact size mismatch: {relative}")
        if sha256_file(artifact) != entry["sha256"]:
            raise ValueError(f"canonical artifact hash mismatch: {relative}")
    return {"canonical_artifacts_verified": len(artifacts)}


def validate_stage_manifests(root: Path) -> dict[str, int]:
    manifest_root = (
        root / "results/corrected_lendingclub_to_homecredit_transfer/manifests"
    )
    checked_manifests = 0
    checked_artifacts = 0
    for path in sorted(manifest_root.glob("*_stage_manifest.json")):
        if ".pre_" in path.name or ".failed_" in path.name:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        artifact_hashes = payload.get("artifact_hashes", {})
        if not isinstance(artifact_hashes, dict):
            raise ValueError(f"stage manifest artifact hashes invalid: {path}")
        for relative, expected_hash in artifact_hashes.items():
            artifact = root / str(relative).replace("\\", "/")
            if not artifact.is_file():
                raise ValueError(f"successful stage artifact missing: {relative}")
            if sha256_file(artifact) != expected_hash:
                raise ValueError(f"successful stage artifact hash mismatch: {relative}")
            checked_artifacts += 1
        checked_manifests += 1
    if checked_manifests != 5:
        raise ValueError(
            f"expected five successful reverse-transfer stage manifests, found {checked_manifests}"
        )
    return {
        "successful_stage_manifests_verified": checked_manifests,
        "successful_stage_artifacts_verified": checked_artifacts,
    }


def validate_removed_paths(root: Path) -> dict[str, int]:
    remaining = [relative for relative in REMOVED_PATHS if (root / relative).exists()]
    if remaining:
        raise ValueError(f"removed cleanup paths still exist: {remaining}")
    return {"removed_paths_verified_absent": len(REMOVED_PATHS)}


def validate_active_paths(root: Path, frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    runs = frames["run_index.csv"]
    active_runs = runs[runs["reuse_status"].isin(["reusable_existing", "newly_executed"])]
    path_columns = [
        "metric_artifact_path",
        "prediction_artifact_path",
        "selected_feature_path",
        "checkpoint_path",
        "manifest_path",
    ]
    missing: list[str] = []
    checked: set[str] = set()
    for row in active_runs.itertuples(index=False):
        for column in path_columns:
            value = getattr(row, column, None)
            if pd.isna(value) or not str(value).strip():
                continue
            relative = str(value).replace("\\", "/")
            checked.add(relative)
            if not (root / relative).exists():
                missing.append(relative)
    if missing:
        raise ValueError(f"active run paths missing: {sorted(set(missing))}")
    return {"active_runs": len(active_runs), "active_run_paths_verified": len(checked)}


def _resolve_active_reference(
    repository_root: Path,
    results_root: Path,
    value: object,
    *,
    field: str,
) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError(f"active run {field} must not be empty")
    supplied = Path(text)
    if ".." in supplied.parts:
        raise ValueError(f"active run {field} contains path traversal: {text}")
    candidates = (
        [supplied.resolve()]
        if supplied.is_absolute()
        else [
            (repository_root / supplied).resolve(),
            (results_root / supplied).resolve(),
        ]
    )
    for candidate in candidates:
        try:
            candidate.relative_to(results_root)
        except ValueError:
            continue
        return candidate
    raise ValueError(f"active run {field} escapes results root: {text}")


def validate_active_results(root: Path) -> dict[str, object]:
    """Validate the new active-results layout and every registered run."""

    from credit_risk_fs.experiments.result_paths import (  # noqa: PLC0415
        RESULT_SUBDIRECTORIES,
        RUN_INDEX_COLUMNS,
        sanitize_component,
    )
    from credit_risk_fs.experiments.tracking import (  # noqa: PLC0415
        STANDARD_ARTIFACTS,
    )

    results_root = (root / "results").resolve()
    if not results_root.is_dir():
        raise ValueError(f"active results directory missing: {results_root}")
    readme_path = results_root / "README.md"
    if not readme_path.is_file():
        raise ValueError("active results README.md is missing")
    missing_directories = [
        name
        for name in RESULT_SUBDIRECTORIES
        if not (results_root / name).is_dir()
    ]
    if missing_directories:
        raise ValueError(
            f"active results directories missing: {missing_directories}"
        )

    index_path = results_root / "run_index.csv"
    if not index_path.is_file():
        raise ValueError("active results run_index.csv is missing")
    with index_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        header = tuple(reader.fieldnames or ())
        missing_columns = [
            column for column in RUN_INDEX_COLUMNS if column not in header
        ]
        if missing_columns:
            raise ValueError(
                f"active run index missing required columns: {missing_columns}"
            )
        rows = list(reader)

    run_ids = [str(row["run_id"]).strip() for row in rows]
    if any(not run_id for run_id in run_ids):
        raise ValueError("active run index contains an empty run_id")
    duplicates = sorted(
        run_id for run_id in set(run_ids) if run_ids.count(run_id) > 1
    )
    if duplicates:
        raise ValueError(f"active run index contains duplicate run IDs: {duplicates}")

    checked_artifacts = 0
    for row in rows:
        run_id = str(row["run_id"]).strip()
        empty_identity_fields = [
            field
            for field in ("dataset", "selector", "model", "status")
            if not str(row[field]).strip()
        ]
        if empty_identity_fields:
            raise ValueError(
                f"active run {run_id} has empty fields: {empty_identity_fields}"
            )
        dataset = sanitize_component(row["dataset"], field_name="dataset")
        run_directory = _resolve_active_reference(
            root,
            results_root,
            row["run_directory"],
            field="run_directory",
        )
        expected_parent = results_root / "runs" / dataset
        if run_directory.parent != expected_parent or run_directory.name != run_id:
            raise ValueError(
                f"active run directory does not match runs/<dataset>/<run_id>: "
                f"{row['run_directory']}"
            )
        if not run_directory.is_dir():
            raise ValueError(f"active run directory missing: {row['run_directory']}")

        config_path = _resolve_active_reference(
            root,
            results_root,
            row["config_path"],
            field="config_path",
        )
        manifest_path = _resolve_active_reference(
            root,
            results_root,
            row["manifest_path"],
            field="manifest_path",
        )
        for field, path in (
            ("config_path", config_path),
            ("manifest_path", manifest_path),
        ):
            try:
                path.relative_to(run_directory)
            except ValueError as exc:
                raise ValueError(
                    f"active run {field} is outside its run directory: {path}"
                ) from exc
            if not path.is_file():
                raise ValueError(f"active run references missing {field}: {path}")

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"active run manifest is unreadable: {manifest_path}") from exc
        if str(manifest.get("run_id", "")) != run_id:
            raise ValueError(f"active run manifest run_id mismatch: {manifest_path}")
        allowed_statuses = {
            "running",
            "completed",
            "failed",
            "interrupted",
            "aborted_resource_limit",
            "dev_complete",
        }
        row_status = str(row["status"]).strip().lower()
        manifest_status = str(manifest.get("status", "")).strip().lower()
        if row_status not in allowed_statuses or manifest_status not in allowed_statuses:
            raise ValueError(f"active run has an unsupported terminal status: {run_id}")
        if row_status != manifest_status:
            raise ValueError(f"active run index/manifest status mismatch: {run_id}")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(
                f"active run manifest lacks an artifact contract: {manifest_path}"
            )
        missing_artifact_entries = sorted(set(STANDARD_ARTIFACTS) - set(artifacts))
        if missing_artifact_entries:
            raise ValueError(
                f"active run manifest omits standard artifact entries: "
                f"{missing_artifact_entries}"
            )
        completed = row_status == "completed"
        for artifact_name, entry in artifacts.items():
            if not isinstance(entry, dict):
                raise ValueError(
                    f"active run artifact entry is invalid: {artifact_name}"
                )
            if not isinstance(entry.get("applicable"), bool) or not isinstance(
                entry.get("present"), bool
            ):
                raise ValueError(
                    f"active run artifact flags must be booleans: {artifact_name}"
                )
            applicable = entry["applicable"]
            present = entry["present"]
            relative = str(entry.get("path", "")).strip()
            if applicable and not relative:
                raise ValueError(
                    f"active run artifact path is empty: {artifact_name}"
                )
            if not relative:
                continue
            artifact_path = Path(relative)
            if artifact_path.is_absolute():
                raise ValueError(
                    f"active run artifact path must be relative: {artifact_name}"
                )
            if ".." in artifact_path.parts:
                raise ValueError(
                    f"active run artifact path contains traversal: {artifact_name}"
                )
            resolved_artifact = (run_directory / artifact_path).resolve()
            try:
                resolved_artifact.relative_to(run_directory)
            except ValueError as exc:
                raise ValueError(
                    f"active run artifact escapes run directory: {relative}"
                ) from exc
            if present and not resolved_artifact.is_file():
                raise ValueError(
                    f"active run references missing artifact "
                    f"{artifact_name}: {resolved_artifact}"
                )
            if not present and resolved_artifact.exists():
                raise ValueError(
                    f"active run manifest marks existing artifact absent: "
                    f"{artifact_name}"
                )
            if completed and applicable and not present:
                raise ValueError(
                    f"completed active run lacks applicable artifact: "
                    f"{artifact_name}"
                )
            checked_artifacts += int(present)
            if present and entry.get("size_bytes") is not None:
                if resolved_artifact.stat().st_size != int(entry["size_bytes"]):
                    raise ValueError(
                        f"active run artifact size mismatch: {artifact_name}"
                    )
            if present and entry.get("sha256"):
                if sha256_file(resolved_artifact) != str(entry["sha256"]):
                    raise ValueError(
                        f"active run artifact checksum mismatch: {artifact_name}"
                    )

        success_marker = run_directory / "_SUCCESS"
        hardened = isinstance(manifest.get("execution_policy"), dict)
        if not completed and success_marker.exists():
            raise ValueError(f"non-completed active run has a success marker: {run_id}")
        if hardened:
            if completed and not success_marker.is_file():
                raise ValueError(f"completed hardened run lacks success marker: {run_id}")
            checkpoint = run_directory / "checkpoint.json"
            if not checkpoint.is_file():
                raise ValueError(f"hardened run lacks checkpoint: {run_id}")
            checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            checkpoint_status = str(checkpoint_payload.get("status", "")).lower()
            expected_checkpoint_status = (
                "running" if row_status == "dev_complete" else row_status
            )
            if checkpoint_status != expected_checkpoint_status:
                raise ValueError(
                    f"active run checkpoint status mismatch: {run_id}"
                )
            if row_status == "dev_complete":
                if set(map(str, checkpoint_payload.get("completed_fold_ids", []))) != {
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                }:
                    raise ValueError(f"DEV-complete run lacks five folds: {run_id}")
                if not (run_directory / "results/dev_predictions.csv").is_file():
                    raise ValueError(f"DEV-complete run lacks OOF predictions: {run_id}")
                if (run_directory / "results/oot_predictions.csv").exists() or (
                    run_directory / "data_access_oot.json"
                ).exists():
                    raise ValueError(f"DEV-complete run opened OOT early: {run_id}")

    return {
        "status": "passed",
        "required_directories": len(RESULT_SUBDIRECTORIES),
        "run_index_columns": list(RUN_INDEX_COLUMNS),
        "registered_runs": len(rows),
        "artifacts_verified": checked_artifacts,
    }


def validate_legacy_repository(root: Path) -> dict[str, object]:
    """Run the preserved scientific checks against an explicit legacy checkout."""

    from credit_risk_fs.clip.reverse_transfer import (  # noqa: PLC0415
        validate_registry_bundle,
        validate_summary_manifest_payloads,
    )

    registry_root = root / "results" / "research_summary"
    frames = {
        name: pd.read_csv(registry_root / name) for name in REGISTRY_FILES
    }
    validate_registry_bundle(frames, verify_artifacts=True, repository_root=root)
    payloads = {
        registry_root / name: (registry_root / name).read_bytes()
        for name in SUMMARY_PAYLOADS
    }
    summary = json.loads(
        (registry_root / "summary_manifest.json").read_text(encoding="utf-8")
    )
    validate_summary_manifest_payloads(
        summary,
        registry_root=Path("results/research_summary"),
        payloads=payloads,
    )
    return {
        "status": "passed",
        "registry_rows": {name: len(frame) for name, frame in frames.items()},
        **validate_active_paths(root, frames),
        **validate_package(root),
        **validate_canonical_manifest(root),
        **validate_stage_manifests(root),
        "pending_inputs": validate_pending(root),
        "summary_manifest": "passed",
        "registry_bundle": "passed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output")
    parser.add_argument(
        "--legacy-repository-root",
        type=Path,
        default=None,
        help=(
            "Optional read-only legacy checkout containing results/research_summary "
            "and the other historical scientific artifacts."
        ),
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from credit_risk_fs.experiments.result_paths import (  # noqa: PLC0415
        LEGACY_RESULTS_ENV,
        configured_legacy_results_root,
        validate_results_root_separation,
    )

    configured_legacy = configured_legacy_results_root()
    legacy = (
        validate_legacy_repository(args.legacy_repository_root.resolve())
        if args.legacy_repository_root is not None
        else {
            "status": "configured_read_only",
            "location": str(configured_legacy),
            "file_count": sum(
                1 for path in configured_legacy.rglob("*") if path.is_file()
            ),
        }
        if configured_legacy is not None
        else {
            "status": "external_optional",
            "location": "not configured",
        }
    )
    if configured_legacy is not None:
        validate_results_root_separation(
            root / "results",
            configured_legacy,
            forbidden_legacy_roots=(
                root / "data",
                root / "tests_runtime",
                Path(os.environ.get("TEMP", root / "tests_runtime")),
            ),
        )
        legacy["configuration"] = LEGACY_RESULTS_ENV
    results = {
        "active_results": validate_active_results(root),
        "historical_results": legacy,
        **validate_removed_paths(root),
    }
    output = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output:
        (root / args.output).write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
