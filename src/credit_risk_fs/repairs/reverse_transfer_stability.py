from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from credit_risk_fs.clip.reverse_transfer import (
    atomic_registry_transaction,
    build_summary_manifest,
    validate_summary_manifest_payloads,
)
from credit_risk_fs.evaluation.stability import (
    candidate_universe_from_frozen_pool,
    read_fold_feature_tables,
    write_feature_stability_artifacts,
)
from credit_risk_fs.pipelines.reverse_transfer import _stage_artifact_paths
from credit_risk_fs.utils.hashing import sha256_file


REPAIR_VERSION = "final_pre_prompt4_stability_repair_v2"
MODELS = {
    "lr": ("logistic_regression", 60, 20),
    "catboost": ("catboost", 100, 40),
}


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _relative(path: Path) -> str:
    return path.as_posix()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def _metric_values(content: bytes) -> dict[str, Any]:
    frame = pd.read_csv(io.BytesIO(content))
    if len(frame) != 1:
        raise ValueError("stability artifact must contain exactly one row")
    return {
        key: (
            None
            if pd.isna(value)
            else value.item()
            if hasattr(value, "item")
            else value
        )
        for key, value in frame.iloc[0].items()
    }


def _build_stability_payload(
    *,
    output_root: Path,
    model: str,
    staging_root: Path,
) -> tuple[Path, bytes]:
    directory_name, expected_universe, expected_budget = MODELS[model]
    source_features = output_root / "downstream" / directory_name / "features"
    candidate_pool = output_root / "candidate_pools" / f"{model}_candidate_pool.csv"
    tables = read_fold_feature_tables(source_features)
    observed_universe = candidate_universe_from_frozen_pool(
        candidate_pool,
        selected_sets=[
            set(table["feature_name"].dropna().astype(str)) for table in tables
        ],
    )
    fold_sizes = [int(table["feature_name"].nunique()) for table in tables]
    if observed_universe != expected_universe:
        raise ValueError(
            f"{model} candidate universe is {observed_universe}, expected {expected_universe}"
        )
    if not fold_sizes or set(fold_sizes) != {expected_budget}:
        raise ValueError(
            f"{model} selected-feature counts are {fold_sizes}, expected {expected_budget}"
        )

    staged_exp = staging_root / directory_name
    staged_features = staged_exp / "features"
    staged_features.mkdir(parents=True, exist_ok=True)
    for name in ("fold_selected_features.csv", "llm_rankings_summary.csv"):
        source = source_features / name
        if source.exists():
            shutil.copyfile(source, staged_features / name)
    write_feature_stability_artifacts(
        exp_dir=staged_exp,
        model=model,
        selector="reverse_transfer_clip_then_mrmr",
        candidate_pool_path=candidate_pool,
    )
    for name in ("selection_frequency.csv", "semantic_group_stability.csv"):
        generated = staged_features / name
        existing = source_features / name
        if generated.read_bytes() != existing.read_bytes():
            raise ValueError(f"saved fold selections do not reproduce {existing}")
    target = source_features / "feature_stability_metrics.csv"
    return target, (staged_features / target.name).read_bytes()


def _protected_hashes(output_root: Path, allowed: set[Path]) -> dict[str, str]:
    return {
        _relative(path): sha256_file(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path not in allowed
    }


def _validate_stability_values(
    payloads: dict[Path, bytes],
    *,
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for model, (directory, universe, budget) in MODELS.items():
        path = (
            output_root
            / "downstream"
            / directory
            / "features"
            / "feature_stability_metrics.csv"
        )
        row = _metric_values(payloads[path])
        if int(row["total_candidate_feature_count"]) != universe:
            raise ValueError(f"{model} saved candidate universe is incorrect")
        for field in ("stable_feature_ratio_80", "semantic_group_stable_ratio_80"):
            value = row[field]
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{model} {field} is outside [0, 1]")
        row["selected_features_per_fold"] = budget
        values[model] = row
    return values


def repair(
    *,
    repository_root: str | Path = ".",
    output_root: str | Path = "results/corrected_lendingclub_to_homecredit_transfer",
    registry_root: str | Path = "results/research_summary",
    migration_root: str | Path | None = None,
) -> dict[str, Any]:
    """Apply the saved-selection stability repair without running scientific stages."""
    repository = Path(repository_root).resolve()
    output = (repository / output_root).resolve()
    registry = (repository / registry_root).resolve()
    migration = (
        (repository / migration_root).resolve()
        if migration_root is not None
        else registry / "migrations" / REPAIR_VERSION
    )
    repair_manifest_path = migration / "repair_manifest.json"
    transaction_manifest_path = migration / "transaction_manifest.json"

    if repair_manifest_path.exists() and transaction_manifest_path.exists():
        existing = json.loads(repair_manifest_path.read_text(encoding="utf-8"))
        for path_text, expected in existing["new_sha256"].items():
            path = repository / path_text
            if not path.exists() or sha256_file(path) != expected:
                raise ValueError("existing repair manifest does not match active files")
        return {
            "transaction_outcome": "IDEMPOTENT_NO_OP",
            "repair_manifest_path": _relative(repair_manifest_path.relative_to(repository)),
            "transaction_path": _relative(migration.relative_to(repository)),
            "manifest": existing,
        }

    stability_targets = {
        output
        / "downstream"
        / directory
        / "features"
        / "feature_stability_metrics.csv"
        for directory, _, _ in MODELS.values()
    }
    evaluate_manifest_path = output / "manifests" / "evaluate_stage_manifest.json"
    registration_transaction_path = (
        output / "manifests" / "registration_transaction_manifest.json"
    )
    register_manifest_path = output / "manifests" / "register_stage_manifest.json"
    summary_manifest_path = registry / "summary_manifest.json"
    existing_targets = {
        *stability_targets,
        evaluate_manifest_path,
        registration_transaction_path,
        register_manifest_path,
        summary_manifest_path,
    }

    # Capture all hashes before the first write, including creation of backup/staging dirs.
    old_sha256 = {
        _relative(path.relative_to(repository)): sha256_file(path)
        for path in sorted(existing_targets)
    }
    protected_before = _protected_hashes(output, existing_targets)

    migration.mkdir(parents=True, exist_ok=True)
    backup_dir = migration / "backups"
    staging_dir = migration / "staged"
    backup_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(existing_targets):
        backup = backup_dir / path.relative_to(repository)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, backup)
        if sha256_file(backup) != old_sha256[_relative(path.relative_to(repository))]:
            raise RuntimeError(f"backup verification failed for {path}")

    replacements: dict[Path, bytes] = {}
    for model in MODELS:
        target, content = _build_stability_payload(
            output_root=output,
            model=model,
            staging_root=staging_dir,
        )
        replacements[target] = content
    stability_values = _validate_stability_values(replacements, output_root=output)

    evaluate_manifest = json.loads(
        evaluate_manifest_path.read_text(encoding="utf-8")
    )
    artifact_paths = _stage_artifact_paths(
        "evaluate", output, (11, 22, 33, 44, 55), ("lr", "catboost")
    )
    evaluate_manifest["artifact_hashes"] = {
        _relative(path.relative_to(repository)): (
            _hash_bytes(replacements[path]) if path in replacements else sha256_file(path)
        )
        for path in artifact_paths
    }
    replacements[evaluate_manifest_path] = _json_bytes(evaluate_manifest)

    registry_payloads = {
        path: path.read_bytes()
        for path in (
            registry / "run_index.csv",
            registry / "artifact_registry.csv",
            registry / "reusable_metrics.csv",
            registry / "selected_feature_registry.csv",
            registry / "results_access_guide.md",
        )
    }
    old_summary = json.loads(summary_manifest_path.read_text(encoding="utf-8"))
    new_summary = build_summary_manifest(
        old_summary,
        registry_root=registry.relative_to(repository),
        payloads=registry_payloads,
    )
    validate_summary_manifest_payloads(
        new_summary,
        registry_root=registry.relative_to(repository),
        payloads=registry_payloads,
    )
    replacements[summary_manifest_path] = _json_bytes(new_summary)
    new_summary_hash = _hash_bytes(replacements[summary_manifest_path])

    registration_transaction = json.loads(
        registration_transaction_path.read_text(encoding="utf-8")
    )
    summary_key = _relative(summary_manifest_path.relative_to(repository))
    registration_transaction["updated_files"][summary_key] = new_summary_hash
    metadata_keys = (
        "source_dataset",
        "external_dataset",
        "pairing_policy_version",
        "configuration_hash",
        "data_manifest_hash",
        "raw_dev_statistical_evidence_hash",
        "run_ids",
        "old_invalid_rows_preserved",
    )
    transaction_metadata = {
        key: registration_transaction[key] for key in metadata_keys
    }
    transaction_targets = [
        Path(path) for path in registration_transaction["registry_paths"]
    ]
    registration_transaction["transaction_id"] = hashlib.sha256(
        json.dumps(
            {
                "targets": sorted(
                    str(path.resolve()) for path in transaction_targets
                ),
                "post_hashes": {
                    str(path): registration_transaction["updated_files"][
                        str(path).replace("\\", "/")
                    ]
                    for path in transaction_targets
                },
                "metadata": transaction_metadata,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    replacements[registration_transaction_path] = _json_bytes(
        registration_transaction
    )
    registration_transaction_hash = _hash_bytes(
        replacements[registration_transaction_path]
    )

    register_manifest = json.loads(
        register_manifest_path.read_text(encoding="utf-8")
    )
    register_manifest["registration_transaction_hash"] = (
        registration_transaction_hash
    )
    register_manifest["registration_transaction"] = registration_transaction
    register_manifest["registered_output_hashes"] = registration_transaction[
        "updated_files"
    ]
    register_manifest["artifact_hashes"] = {
        _relative(registration_transaction_path.relative_to(repository)): (
            registration_transaction_hash
        )
    }
    replacements[register_manifest_path] = _json_bytes(register_manifest)

    new_sha256 = {
        _relative(path.relative_to(repository)): _hash_bytes(content)
        for path, content in replacements.items()
    }
    old_values = {
        model: _metric_values(
            (
                output
                / "downstream"
                / directory
                / "features"
                / "feature_stability_metrics.csv"
            ).read_bytes()
        )
        for model, (directory, _, _) in MODELS.items()
    }
    repair_manifest = {
        "repair_version": REPAIR_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "replace full-model feature universe with each frozen downstream "
            "candidate pool, repair semantic-group ratio, and refresh stale summary"
        ),
        "files_changed": sorted(
            [
                *new_sha256,
                _relative(repair_manifest_path.relative_to(repository)),
                _relative(transaction_manifest_path.relative_to(repository)),
            ]
        ),
        "old_sha256": old_sha256,
        "new_sha256": new_sha256,
        "old_values": {
            "stability": old_values,
            "summary_counts": {
                "run_counts": old_summary["run_counts"],
                "artifact_counts": old_summary["artifact_counts"],
                "reusable_metric_rows": old_summary["reusable_metric_rows"],
                "selected_feature_artifact_rows": old_summary[
                    "selected_feature_artifact_rows"
                ],
            },
        },
        "new_values": {
            "stability": stability_values,
            "summary_counts": {
                "run_counts": new_summary["run_counts"],
                "artifact_counts": new_summary["artifact_counts"],
                "reusable_metric_rows": new_summary["reusable_metric_rows"],
                "selected_feature_artifact_rows": new_summary[
                    "selected_feature_artifact_rows"
                ],
            },
        },
        "candidate_universe_sizes": {"lr": 60, "catboost": 100},
        "selected_features_per_fold": {"lr": 20, "catboost": 40},
        "formula_versions": {
            "nogueira": "nogueira_binary_indicator_unbiased_variance_v1",
            "kuncheva": "mean_pairwise_fixed_size_kuncheva_v1",
            "jaccard": "mean_pairwise_set_jaccard_v1",
            "semantic_group_stable_ratio_80": (
                "stable_groups_at_80pct_over_groups_represented_in_any_fold_v2"
            ),
        },
        "backup_path": _relative(backup_dir.relative_to(repository)),
        "transaction_path": _relative(migration.relative_to(repository)),
        "rollback_supported": True,
        "protected_scientific_hashes": protected_before,
        "validation_result": "passed_before_atomic_replacement",
    }
    repair_manifest_content = _json_bytes(repair_manifest)

    summary_content = replacements.pop(summary_manifest_path)
    payloads = {
        **replacements,
        repair_manifest_path: repair_manifest_content,
        summary_manifest_path: summary_content,
    }
    transaction = atomic_registry_transaction(
        payloads,
        transaction_manifest_path=transaction_manifest_path,
        metadata={
            "repair_version": REPAIR_VERSION,
            "reason": "saved-selection stability and central-summary metadata repair",
        },
    )

    protected_after = _protected_hashes(output, existing_targets)
    if protected_after != protected_before:
        raise RuntimeError("protected scientific artifacts changed during repair")
    for path, content in replacements.items():
        if path.read_bytes() != content:
            raise RuntimeError(f"post-write byte validation failed for {path}")
    if summary_manifest_path.read_bytes() != summary_content:
        raise RuntimeError("post-write byte validation failed for summary manifest")
    repair_manifest["validation_result"] = "passed_after_atomic_replacement"
    # Preserve transaction atomicity: report the persisted pre-commit validation
    # and return the post-commit validation separately rather than rewriting it.
    return {
        "transaction_outcome": transaction["transaction_outcome"],
        "repair_manifest_path": _relative(repair_manifest_path.relative_to(repository)),
        "transaction_manifest_path": _relative(
            transaction_manifest_path.relative_to(repository)
        ),
        "transaction_path": _relative(migration.relative_to(repository)),
        "backup_path": _relative(backup_dir.relative_to(repository)),
        "rollback_validation": "byte-identical backups verified",
        "post_validation": "passed_after_atomic_replacement",
        "manifest": repair_manifest,
    }


def rollback(
    *,
    repository_root: str | Path = ".",
    migration_root: str | Path = (
        "results/research_summary/migrations/" + REPAIR_VERSION
    ),
) -> None:
    """Restore every replaced pre-repair file byte-for-byte from verified backups."""
    repository = Path(repository_root).resolve()
    migration = (repository / migration_root).resolve()
    repair_manifest = json.loads(
        (migration / "repair_manifest.json").read_text(encoding="utf-8")
    )
    backup_dir = migration / "backups"
    originals: dict[Path, bytes] = {}
    for path_text, expected_hash in repair_manifest["old_sha256"].items():
        backup = backup_dir / path_text
        if sha256_file(backup) != expected_hash:
            raise RuntimeError(f"rollback backup is corrupt: {backup}")
        originals[repository / path_text] = backup.read_bytes()
    rollback_manifest = migration / "rollback_transaction_manifest.json"
    atomic_registry_transaction(
        originals,
        transaction_manifest_path=rollback_manifest,
        metadata={"repair_version": REPAIR_VERSION, "operation": "rollback"},
    )
    for path_text, expected_hash in repair_manifest["old_sha256"].items():
        if sha256_file(repository / path_text) != expected_hash:
            raise RuntimeError(f"rollback verification failed: {path_text}")
