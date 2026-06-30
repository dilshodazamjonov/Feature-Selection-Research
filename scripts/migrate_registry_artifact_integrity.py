from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.clip.reverse_transfer import (
    ARTIFACT_IDENTITY_VERSION,
    REGISTRY_SCHEMA_VERSION,
    atomic_registry_transaction,
    canonical_artifact_id,
    canonical_registry_value,
    validate_registry_bundle,
    validate_summary_manifest,
)
from credit_risk_fs.utils.hashing import sha256_file


REGISTRY_ROOT = Path("results/research_summary")
CENTRAL_NAMES = (
    "run_index.csv",
    "artifact_registry.csv",
    "reusable_metrics.csv",
    "selected_feature_registry.csv",
    "results_access_guide.md",
    "summary_manifest.json",
)
MIGRATION_VERSION = "registry_artifact_identity_migration_v1"


def _hash_bytes(content: bytes) -> str:
    import hashlib

    return hashlib.sha256(content).hexdigest()


def main() -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    migration_dir = (
        REGISTRY_ROOT / "migrations" / f"{MIGRATION_VERSION}_{timestamp}"
    )
    backup_dir = migration_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=False)

    central = {name: REGISTRY_ROOT / name for name in CENTRAL_NAMES}
    pre_hashes = {name: sha256_file(path) for name, path in central.items()}
    for name, path in central.items():
        shutil.copy2(path, backup_dir / name)
        if sha256_file(backup_dir / name) != pre_hashes[name]:
            raise RuntimeError(f"backup hash mismatch: {name}")

    run_index = pd.read_csv(central["run_index.csv"])
    artifacts = pd.read_csv(central["artifact_registry.csv"])
    metrics = pd.read_csv(central["reusable_metrics.csv"])
    selected = pd.read_csv(central["selected_feature_registry.csv"])
    canonical_paths = artifacts["relative_path"].map(
        lambda value: canonical_registry_value(
            "relative_path", value, expected_type="path"
        )
    )
    conflicting_ids = {
        str(artifact_id)
        for artifact_id, group in artifacts.assign(
            _canonical_path=canonical_paths
        ).groupby("artifact_id")
        if group["_canonical_path"].nunique() > 1
    }
    identity_changes: list[dict[str, str]] = []
    for index, row in artifacts.iterrows():
        if str(row["artifact_id"]) not in conflicting_ids:
            continue
        canonical_path = canonical_registry_value(
            "relative_path", row["relative_path"], expected_type="path"
        )
        new_id = canonical_artifact_id(
            run_id=str(row.get("created_by_run_id", "") or ""),
            artifact_type=str(row["artifact_type"]),
            relative_path=canonical_path,
            content_hash=str(row["file_hash"]),
        )
        identity_changes.append(
            {
                "old_artifact_id": str(row["artifact_id"]),
                "new_artifact_id": new_id,
                "canonical_path": canonical_path,
                "content_sha256": str(row["file_hash"]),
                "affected_row": str(index + 2),
            }
        )
        artifacts.at[index, "artifact_id"] = new_id
        artifacts.at[index, "relative_path"] = canonical_path

    missing_updates: list[dict[str, str]] = []
    derived_artifact_updates: list[dict[str, str]] = []
    for index, row in artifacts.iterrows():
        path = Path(str(row["relative_path"]))
        declared = str(row["file_exists"]).strip().lower() == "true"
        if declared and not path.exists():
            artifacts.at[index, "file_exists"] = False
            missing_updates.append(
                {
                    "artifact_id": str(artifacts.at[index, "artifact_id"]),
                    "canonical_path": str(row["relative_path"]),
                    "preserved_content_sha256": str(row["file_hash"]),
                    "repair": "file_exists:true->false",
                }
            )
        elif declared and path.exists():
            actual_hash = sha256_file(path)
            recorded_hash = str(row["file_hash"]).lower()
            if actual_hash != recorded_hash:
                canonical_path = canonical_registry_value(
                    "relative_path",
                    row["relative_path"],
                    expected_type="path",
                )
                if canonical_path != (
                    "results/research_summary/results_access_guide.md"
                ):
                    raise RuntimeError(
                        f"unexpected physical artifact hash mismatch: {canonical_path}"
                    )
                new_id = canonical_artifact_id(
                    run_id=str(row.get("created_by_run_id", "") or ""),
                    artifact_type=str(row["artifact_type"]),
                    relative_path=canonical_path,
                    content_hash=actual_hash,
                )
                artifacts.at[index, "artifact_id"] = new_id
                artifacts.at[index, "file_hash"] = actual_hash
                derived_artifact_updates.append(
                    {
                        "old_artifact_id": str(row["artifact_id"]),
                        "new_artifact_id": new_id,
                        "canonical_path": canonical_path,
                        "old_content_sha256": recorded_hash,
                        "new_content_sha256": actual_hash,
                        "reason": "derived access guide changed during registration",
                    }
                )

    normalized_output_folders = 0
    if "output_folder" in metrics:
        normalized = metrics["output_folder"].map(
            lambda value: canonical_registry_value(
                "output_folder", value, expected_type="path"
            )
            if pd.notna(value) and str(value).strip()
            else value
        )
        normalized_output_folders = int(
            (normalized.astype(str) != metrics["output_folder"].astype(str)).sum()
        )
        metrics["output_folder"] = normalized

    artifact_hash_by_path = {
        canonical_registry_value(
            "relative_path", row.relative_path, expected_type="path"
        ): str(row.file_hash).lower()
        for row in artifacts.itertuples()
    }
    foreign_key_updates: list[dict[str, str]] = []
    for index, row in selected.iterrows():
        canonical_path = canonical_registry_value(
            "selected_feature_path",
            row["selected_feature_path"],
            expected_type="path",
        )
        expected_hash = artifact_hash_by_path.get(canonical_path)
        observed_hash = str(row["selected_feature_hash"]).lower()
        if expected_hash is not None and observed_hash != expected_hash:
            selected.at[index, "selected_feature_hash"] = expected_hash
            foreign_key_updates.append(
                {
                    "registry": "selected_feature_registry.csv",
                    "row": str(index + 2),
                    "run_id": str(row["run_id"]),
                    "canonical_path": canonical_path,
                    "old_hash": observed_hash,
                    "new_hash": expected_hash,
                }
            )

    proposed_frames = {
        "run_index.csv": run_index,
        "artifact_registry.csv": artifacts,
        "reusable_metrics.csv": metrics,
        "selected_feature_registry.csv": selected,
    }
    validate_registry_bundle(
        proposed_frames,
        verify_artifacts=True,
        repository_root=Path.cwd(),
    )

    payloads = {
        central["run_index.csv"]: central["run_index.csv"].read_bytes(),
        central["artifact_registry.csv"]: artifacts.to_csv(
            index=False
        ).encode("utf-8"),
        central["reusable_metrics.csv"]: metrics.to_csv(
            index=False
        ).encode("utf-8"),
        central["selected_feature_registry.csv"]: selected.to_csv(
            index=False
        ).encode("utf-8"),
    }
    guide_path = central["results_access_guide.md"]
    # The guide is derived from run visibility, which this identity-only
    # migration does not change. Include its exact bytes in the transaction
    # so its summary hash is regenerated without invalidating its own
    # registered artifact hash.
    payloads[guide_path] = guide_path.read_bytes()

    summary_path = central["summary_manifest.json"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["registry_integrity_migration"] = {
        "migration_version": MIGRATION_VERSION,
        "artifact_identity_version": ARTIFACT_IDENTITY_VERSION,
        "reason": "replace content-only truncated artifact IDs that mapped to multiple canonical paths",
        "repaired_artifact_rows": len(identity_changes),
        "normalized_output_folders": normalized_output_folders,
        "missing_historical_artifacts_marked_absent": len(missing_updates),
        "foreign_key_hashes_repaired": len(foreign_key_updates),
        "derived_artifact_references_repaired": len(
            derived_artifact_updates
        ),
    }
    summary["registry_file_hashes"] = {
        str(path).replace("\\", "/"): _hash_bytes(payloads[path])
        for path in (
            central["run_index.csv"],
            central["artifact_registry.csv"],
            central["reusable_metrics.csv"],
            central["selected_feature_registry.csv"],
            guide_path,
        )
    }
    validate_summary_manifest(summary)
    payloads[summary_path] = json.dumps(
        summary, indent=2, ensure_ascii=False
    ).encode("utf-8")
    post_hashes = {
        path.name: _hash_bytes(content)
        for path, content in payloads.items()
    }
    migration_manifest_path = migration_dir / "migration_manifest.json"
    migration_manifest = {
        "migration_version": MIGRATION_VERSION,
        "artifact_identity_version": ARTIFACT_IDENTITY_VERSION,
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": "repair all pre-existing central registry identity and canonicalization conflicts",
        "identity_changes": identity_changes,
        "foreign_key_updates": foreign_key_updates,
        "missing_historical_artifact_updates": missing_updates,
        "derived_artifact_updates": derived_artifact_updates,
        "normalized_output_folder_rows": normalized_output_folders,
        "affected_files": [
            "artifact_registry.csv",
            "reusable_metrics.csv",
            "selected_feature_registry.csv",
            "summary_manifest.json",
        ],
        "backup_location": str(backup_dir).replace("\\", "/"),
        "pre_migration_registry_hashes": pre_hashes,
        "post_migration_registry_hashes": post_hashes,
        "validation_result": "passed_before_atomic_replacement",
    }
    payloads[migration_manifest_path] = json.dumps(
        migration_manifest, indent=2, ensure_ascii=False
    ).encode("utf-8")
    transaction_manifest_path = migration_dir / "transaction_manifest.json"
    atomic_registry_transaction(
        payloads,
        transaction_manifest_path=transaction_manifest_path,
        metadata={
            "migration_version": MIGRATION_VERSION,
            "artifact_identity_version": ARTIFACT_IDENTITY_VERSION,
            "identity_change_count": len(identity_changes),
        },
    )
    validate_registry_bundle(
        {
            name: pd.read_csv(REGISTRY_ROOT / name)
            for name in (
                "run_index.csv",
                "artifact_registry.csv",
                "reusable_metrics.csv",
            "selected_feature_registry.csv",
            )
        },
        verify_artifacts=True,
        repository_root=Path.cwd(),
    )
    print(
        json.dumps(
            {
                "migration_manifest": str(migration_manifest_path),
                "transaction_manifest": str(transaction_manifest_path),
                "identity_changes": len(identity_changes),
                "normalized_output_folders": normalized_output_folders,
                "missing_artifacts_marked_absent": len(missing_updates),
                "foreign_key_updates": len(foreign_key_updates),
                "derived_artifact_updates": len(derived_artifact_updates),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
