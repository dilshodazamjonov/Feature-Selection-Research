"""Build the read-only-derived cdv1 durable-logging bridge and audit evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from credit_risk_fs.experiments.atomic_io import sha256_file
from credit_risk_fs.experiments.manual_research import FROZEN_HASHES, build_manual_research_plan
from credit_risk_fs.experiments.provenance_bridge import (
    BRIDGE_PATH,
    BRIDGE_SCHEMA_VERSION,
    INTERRUPTED_RUN_ID,
    OBSERVABILITY_TAG,
    ORIGINAL_COMMIT,
    ORIGINAL_TAG,
    REUSABLE_RUN_IDS,
    SAFETY_COMMIT,
    SAFETY_TAG,
    SAFE_RESUME_BOUNDARY,
)


AUDIT_RELATIVE = Path("cleanup/audits/cross_dataset_voting_durable_logging")
RUNTIME_FILES = (
    "src/credit_risk_fs/experiments/checkpointing.py",
    "src/credit_risk_fs/experiments/cross_dataset_research.py",
    "src/credit_risk_fs/experiments/execution.py",
    "src/credit_risk_fs/experiments/manual_research.py",
    "src/credit_risk_fs/experiments/provenance_bridge.py",
    "src/credit_risk_fs/experiments/rank_voting.py",
    "src/credit_risk_fs/experiments/research_logging.py",
    "src/credit_risk_fs/experiments/resource_monitor.py",
    "src/credit_risk_fs/experiments/synthetic_execution.py",
    "src/credit_risk_fs/utils/logging.py",
)
TEST_FILES = (
    "tests/fixtures/cdv1_scientific_equivalence_golden.json",
    "tests/support/cdv1_scientific_equivalence_probe.py",
    "tests/test_cdv1_scientific_equivalence.py",
    "tests/test_checkpointing.py",
    "tests/test_manual_research_orchestration.py",
    "tests/test_provenance_bridge.py",
    "tests/test_research_logging.py",
    "tests/test_resource_monitor.py",
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty audit table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)
    os.replace(temporary, path)


def _git_bytes(root: Path, commit: str, relative: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return result.stdout if result.returncode == 0 else None


def _run_directory(root: Path, dataset: str, run_id: str) -> Path:
    return root / "results" / "runs" / dataset / run_id


def _artifact_inventory(run_dir: Path, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for relative, metadata in checkpoint.get("finalized_artifacts", {}).items():
        if relative == "resource_usage.json":
            continue
        path = run_dir / relative
        if not path.is_file():
            raise RuntimeError(f"finalized artifact missing: {run_dir.name}/{relative}")
        size = int(metadata["size_bytes"])
        digest = str(metadata["sha256"])
        if path.stat().st_size != size or sha256_file(path) != digest:
            raise RuntimeError(f"finalized artifact changed: {run_dir.name}/{relative}")
        inventory[str(relative)] = {"size_bytes": size, "sha256": digest}
    if not inventory:
        raise RuntimeError(f"empty finalized-artifact inventory: {run_dir.name}")
    return inventory


def _run_evidence(root: Path, plan: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_id = {spec.run_id: spec for spec in plan.run_specs}
    run_table: dict[str, Any] = {}
    integrity_rows: list[dict[str, Any]] = []
    for run_id in (*REUSABLE_RUN_IDS, INTERRUPTED_RUN_ID):
        spec = by_id[run_id]
        run_dir = _run_directory(root, spec.dataset, run_id)
        checkpoint = _json(run_dir / "checkpoint.json")
        manifest = _json(run_dir / "manifest.json")
        immutable = _artifact_inventory(run_dir, checkpoint)
        completed_folds = sorted(int(value) for value in checkpoint["completed_fold_ids"])
        expected_commit = (
            ORIGINAL_COMMIT if run_id in REUSABLE_RUN_IDS[:11] else SAFETY_COMMIT
        )
        if checkpoint["identity"]["git_commit"] != expected_commit:
            raise RuntimeError(f"unexpected checkpoint commit: {run_id}")
        run_table[run_id] = {
            "run_directory": run_dir.relative_to(root).as_posix(),
            "checkpoint_commit": expected_commit,
            "immutable_artifacts": immutable,
        }
        expected_folds = [1, 2, 3, 4, 5] if run_id in REUSABLE_RUN_IDS else [1, 2, 3, 4]
        status = str(manifest.get("status"))
        expected_status = "dev_complete" if run_id in REUSABLE_RUN_IDS else "aborted_resource_limit"
        valid = completed_folds == expected_folds and status == expected_status
        integrity_rows.append(
            {
                "run_id": run_id,
                "checkpoint_commit": expected_commit,
                "manifest_status": status,
                "completed_fold_ids": ";".join(map(str, completed_folds)),
                "expected_fold_ids": ";".join(map(str, expected_folds)),
                "immutable_artifact_count": len(immutable),
                "immutable_artifact_bytes": sum(
                    int(item["size_bytes"]) for item in immutable.values()
                ),
                "hash_and_size_validation": "pass",
                "authenticated_for_reuse_or_resume": valid,
            }
        )
        if not valid:
            raise RuntimeError(f"run boundary/status validation failed: {run_id}")
    return run_table, integrity_rows


def _bridge(root: Path, plan: Any, runs: Mapping[str, Any]) -> dict[str, Any]:
    runtime = {}
    for relative in RUNTIME_FILES:
        path = root / relative
        old = _git_bytes(root, SAFETY_COMMIT, relative)
        runtime[relative] = {
            "old_release": SAFETY_COMMIT,
            "old_sha256": hashlib.sha256(old).hexdigest() if old is not None else "absent",
            "new_sha256": sha256_file(path),
            "new_size_bytes": path.stat().st_size,
        }
    frozen = {
        relative: {"sha256": expected, "size_bytes": (root / relative).stat().st_size}
        for relative, expected in FROZEN_HASHES.values()
    }
    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "research_family": "cdv1",
        "original_release": {"tag": ORIGINAL_TAG, "commit": ORIGINAL_COMMIT},
        "safety_release": {"tag": SAFETY_TAG, "commit": SAFETY_COMMIT},
        "observability_release": {
            "tag": OBSERVABILITY_TAG,
            "commit_binding": "annotated_tag_peels_to_current_head",
        },
        "matrix_sha256": plan.matrix_sha256,
        "configuration_set_sha256": plan.configuration_set_sha256,
        "reusable_run_ids": list(REUSABLE_RUN_IDS),
        "interrupted_run": {
            "run_id": INTERRUPTED_RUN_ID,
            "safe_resume_boundary": SAFE_RESUME_BOUNDARY,
            "completed_fold_ids": [1, 2, 3, 4],
            "stop_code": "ram_system_headroom",
        },
        "runtime_files": runtime,
        "frozen_files": frozen,
        "runs": dict(runs),
        "allowed_change_categories": [
            "durable_structured_logging",
            "stage_and_component_observability",
            "parent_owned_bounded_worker_log_transport",
            "logging_tests",
            "documentation_and_audit",
        ],
        "forbidden_change_categories": [
            "datasets_or_row_membership",
            "dev_oot_boundary",
            "folds_or_split_assignments",
            "selectors_voters_or_hyperparameters",
            "preprocessing_or_feature_values",
            "models_or_hyperparameters",
            "seeds_or_thread_counts",
            "voting_ranking_or_tie_breaking",
            "predictions_metrics_or_inference",
            "checkpoint_eligibility",
            "resource_thresholds_precedence_or_shutdown_timing",
        ],
        "test_files": {relative: sha256_file(root / relative) for relative in TEST_FILES},
        "runtime_log_exclusion": {
            "path": "logs/runs.log",
            "gitignore_rule": "/logs/runs.log",
            "included_in_scientific_artifacts": False,
        },
        "scope_rule": (
            "Applies only to the exact original, Prompt 6.1 safety, and current "
            "observability annotated releases, cdv1, and the exact run inventory."
        ),
    }


def _run_014_reconciliation(root: Path, plan: Any) -> dict[str, Any]:
    spec = next(item for item in plan.run_specs if item.run_id == INTERRUPTED_RUN_ID)
    run_dir = _run_directory(root, spec.dataset, spec.run_id)
    checkpoint = _json(run_dir / "checkpoint.json")
    resource = _json(run_dir / "resource_usage.json")
    samples = resource["samples"]
    last = samples[-1]
    return {
        "schema_version": "cdv1_run_014_durable_logging_reconciliation_v1",
        "run_id": INTERRUPTED_RUN_ID,
        "terminal_status": resource["status"],
        "primary_stop_code": checkpoint["primary_stop_code"],
        "secondary_events": checkpoint.get("secondary_events", []),
        "completed_fold_ids": sorted(int(value) for value in checkpoint["completed_fold_ids"]),
        "finalized_artifact_count": len(checkpoint["finalized_artifacts"]),
        "last_sample": {
            "stage": last["stage"],
            "fold_id": last["fold_id"],
            "elapsed_seconds": last["elapsed_seconds"],
            "worker_rss_bytes": last["process_tree_rss_bytes"],
            "system_available_ram_bytes": last["system_available_ram_bytes"],
        },
        "stop_lifecycle": checkpoint["stop_lifecycle"],
        "worker_exit_code": resource["worker_exit_code"],
        "cleanup_evidence": checkpoint["cleanup_evidence"],
        "safe_resume_boundary": SAFE_RESUME_BOUNDARY,
        "fold_5_artifacts_present": (run_dir / "folds" / "fold_5").exists(),
        "oot_paths_present": bool(list(run_dir.rglob("*oot*"))),
        "research_executed_by_audit": False,
    }


def _canonical_results_snapshot(root: Path, audit: Path) -> dict[str, Any]:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted((root / "results").rglob("*"))
        if path.is_file()
    ]
    post_path = audit / "immutable_postchange_results_manifest.csv"
    _atomic_csv(post_path, rows)
    with (audit / "immutable_prechange_results_manifest.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        before = {
            str(row["path"]): (int(row["size_bytes"]), str(row["sha256"]))
            for row in csv.DictReader(handle)
        }
    after = {
        str(row["path"]): (int(row["size_bytes"]), str(row["sha256"]))
        for row in rows
    }
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(path for path in set(before) & set(after) if before[path] != after[path])
    return {
        "prechange_file_count": len(before),
        "postchange_file_count": len(after),
        "prechange_total_bytes": sum(value[0] for value in before.values()),
        "postchange_total_bytes": sum(value[0] for value in after.values()),
        "added_paths": added,
        "removed_paths": removed,
        "changed_paths": changed,
        "exact_path_size_sha256_match": not (added or removed or changed),
        "postchange_manifest_path": post_path.relative_to(root).as_posix(),
        "postchange_manifest_sha256": sha256_file(post_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    plan = build_manual_research_plan(root)
    runs, integrity_rows = _run_evidence(root, plan)
    bridge = _bridge(root, plan, runs)
    audit = root / AUDIT_RELATIVE
    _atomic_csv(audit / "authenticated_run_integrity.csv", integrity_rows)
    _atomic_json(audit / "run_014_reconciliation.json", _run_014_reconciliation(root, plan))
    _atomic_json(root / BRIDGE_PATH, bridge)
    results_snapshot = _canonical_results_snapshot(root, audit)
    summary = {
        "schema_version": "cdv1_durable_logging_evidence_build_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_runs_authenticated": len(REUSABLE_RUN_IDS),
        "incomplete_run_authenticated": INTERRUPTED_RUN_ID,
        "safe_resume_boundary": SAFE_RESUME_BOUNDARY,
        "bridge_path": BRIDGE_PATH.as_posix(),
        "canonical_results_snapshot": results_snapshot,
        "research_or_oot_executed": False,
    }
    _atomic_json(audit / "evidence_build_summary.json", summary)
    evidence_artifacts = {
        path.relative_to(root).as_posix(): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(audit.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _atomic_json(
        audit / "artifact_manifest.json",
        {
            "schema_version": "cdv1_durable_logging_audit_manifest_v1",
            "artifacts": evidence_artifacts,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
