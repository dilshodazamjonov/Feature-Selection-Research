"""Validate central registries, manifests, canonical paths, and pending inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from credit_risk_fs.clip.reverse_transfer import (  # noqa: PLC0415
        validate_registry_bundle,
        validate_summary_manifest_payloads,
    )

    registry_root = root / "results" / "research_summary"
    frames = {
        name: pd.read_csv(registry_root / name) for name in REGISTRY_FILES
    }
    validate_registry_bundle(
        frames, verify_artifacts=True, repository_root=root
    )
    payloads = {
        registry_root / name: (registry_root / name).read_bytes()
        for name in SUMMARY_PAYLOADS
    }
    summary = json.loads(
        (registry_root / "summary_manifest.json").read_text(encoding="utf-8")
    )
    validate_summary_manifest_payloads(
        summary, registry_root=Path("results/research_summary"), payloads=payloads
    )

    results = {
        "registry_rows": {name: len(frame) for name, frame in frames.items()},
        **validate_active_paths(root, frames),
        **validate_package(root),
        **validate_canonical_manifest(root),
        **validate_stage_manifests(root),
        **validate_removed_paths(root),
        "pending_inputs": validate_pending(root),
        "summary_manifest": "passed",
        "registry_bundle": "passed",
    }
    output = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output:
        (root / args.output).write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
