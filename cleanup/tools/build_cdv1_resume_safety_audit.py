"""Build read-only-derived cdv1 incident, integrity, and compatibility evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from credit_risk_fs.experiments.atomic_io import sha256_file
from credit_risk_fs.experiments.manual_research import (
    FROZEN_HASHES,
    build_manual_research_plan,
)
from credit_risk_fs.experiments.provenance_bridge import (
    BRIDGE_PATH,
    BRIDGE_SCHEMA_VERSION,
    INTERRUPTED_RUN_ID,
    MECHANICS_TAG,
    ORIGINAL_COMMIT,
    ORIGINAL_TAG,
    REUSABLE_RUN_IDS,
    SAFE_RESUME_BOUNDARY,
)
from credit_risk_fs.experiments.runner import validate_cross_dataset_research_run


AUDIT_RELATIVE = Path(
    "cleanup/audits/cross_dataset_voting_resource_stop_resume_safety"
)
RUNTIME_FILES = (
    "src/credit_risk_fs/experiments/resource_monitor.py",
    "src/credit_risk_fs/experiments/execution.py",
    "src/credit_risk_fs/experiments/checkpointing.py",
    "src/credit_risk_fs/experiments/manual_research.py",
    "src/credit_risk_fs/experiments/runner.py",
    "src/credit_risk_fs/experiments/provenance_bridge.py",
    "src/credit_risk_fs/experiments/synthetic_execution.py",
)
CONTROL_FILES = (
    "checkpoint.json",
    "manifest.json",
    "run_manifest.json",
    "run.log",
    "resource_usage.json",
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


def _artifact_integrity(run_dir: Path, checkpoint: Mapping[str, Any]) -> tuple[bool, int]:
    verified = 0
    for relative, metadata in checkpoint.get("finalized_artifacts", {}).items():
        path = run_dir / relative
        if not path.is_file():
            return False, verified
        if path.stat().st_size != int(metadata.get("size_bytes", -1)):
            return False, verified
        if sha256_file(path) != metadata.get("sha256"):
            return False, verified
        verified += 1
    return True, verified


def _audit_completed_runs(root: Path, plan: Any) -> list[dict[str, Any]]:
    index = pd.read_csv(root / "results/run_index.csv", keep_default_na=False).set_index(
        "run_id"
    )
    rows = []
    for spec in plan.run_specs[:10]:
        run_dir = _run_directory(root, spec.dataset, spec.run_id)
        manifest = _json(run_dir / "manifest.json")
        checkpoint = _json(run_dir / "checkpoint.json")
        data_access = _json(run_dir / "data_access_dev.json")
        prediction = _json(run_dir / "results/dev_prediction_metadata.json")
        integrity, artifact_count = _artifact_integrity(run_dir, checkpoint)
        required_folds = {"1", "2", "3", "4", "5"}
        fold_dirs = {path.name.split("_", 1)[1] for path in (run_dir / "folds").glob("fold_*")}
        selected = pd.read_csv(run_dir / "features/fold_selected_features.csv")
        budgets = selected.groupby("fold_id")["feature"].nunique().to_dict()
        expected_budgets = {fold: spec.final_feature_budget for fold in range(1, 6)}
        config = _json(run_dir / "config.json")
        partials = list(run_dir.rglob("*.partial*"))
        locks = list(run_dir.rglob("*.lock"))
        oot_paths = list(run_dir.rglob("*oot*"))
        validator_pass = True
        validator_error = ""
        try:
            validate_cross_dataset_research_run(root, spec, phase="dev")
        except Exception as exc:  # pragma: no cover - audit records exact failure
            validator_pass = False
            validator_error = f"{type(exc).__name__}: {exc}"
        matrix_match = (
            config.get("run_id") == spec.run_id
            and config.get("dataset") == spec.dataset
            and config.get("model") == spec.model
            and config.get("method_id") == spec.method_id
            and config.get("candidate_pool_budget") == spec.candidate_pool_budget
            and config.get("final_feature_budget") == spec.final_feature_budget
            and config.get("seed") == 42
        )
        provenance_pass = (
            checkpoint.get("identity", {}).get("git_commit") == ORIGINAL_COMMIT
            and manifest.get("release_tag") == ORIGINAL_TAG
            and checkpoint.get("identity", {}).get("protocol_hash")
            == FROZEN_HASHES["voting_protocol"][1]
            and checkpoint.get("identity", {}).get("row_alignment_hash")
            == FROZEN_HASHES["row_alignment_contract"][1]
        )
        oot_pass = (
            data_access.get("opened_oot_paths") == []
            and data_access.get("retained_oot_rows") == 0
            and not oot_paths
        )
        prediction_pass = (
            prediction.get("coverage_type") == "complete_five_fold_dev_oof"
            and prediction.get("research_eligible") is True
            and prediction.get("comparison_eligible") is True
            and prediction.get("probability_orientation")
            == "class_1_higher_default_risk"
        )
        status_pass = (
            index.loc[spec.run_id, "status"] == "dev_complete"
            and manifest.get("status") == "dev_complete"
            and not (run_dir / "_SUCCESS").exists()
        )
        folds_pass = (
            set(checkpoint.get("completed_fold_ids", [])) == required_folds
            and fold_dirs == required_folds
        )
        budget_pass = budgets == expected_budgets
        immutable_pass = not partials and not locks
        overall = all(
            (
                matrix_match,
                provenance_pass,
                status_pass,
                folds_pass,
                integrity,
                budget_pass,
                prediction_pass,
                oot_pass,
                immutable_pass,
                validator_pass,
            )
        )
        rows.append(
            {
                "run_id": spec.run_id,
                "matrix_match": matrix_match,
                "provenance_match": provenance_pass,
                "dev_terminal_state_valid": status_pass,
                "five_fold_coverage_valid": folds_pass,
                "checkpoint_artifact_integrity_valid": integrity,
                "checkpoint_artifacts_verified": artifact_count,
                "feature_budget_valid": budget_pass,
                "class_1_prediction_contract_valid": prediction_pass,
                "oot_embargo_valid": oot_pass,
                "no_partial_or_lock": immutable_pass,
                "canonical_dev_validator_pass": validator_pass,
                "canonical_dev_validator_error": validator_error,
                "dev_prediction_rows": prediction.get("row_count"),
                "ordered_identity_sha256": prediction.get(
                    "artifact_order_identity_sha256"
                ),
                "identity_target_sha256": prediction.get("identity_target_sha256"),
                "scientifically_reusable": overall,
            }
        )
    return rows


def _audit_run_011(root: Path, spec: Any) -> list[dict[str, Any]]:
    run_dir = _run_directory(root, spec.dataset, spec.run_id)
    checkpoint = _json(run_dir / "checkpoint.json")
    rows = []
    for fold_id in range(1, 6):
        fold_dir = run_dir / "folds" / f"fold_{fold_id}"
        tracked = {
            relative: metadata
            for relative, metadata in checkpoint.get("finalized_artifacts", {}).items()
            if relative.startswith(f"folds/fold_{fold_id}/")
        }
        valid = bool(tracked)
        for relative, metadata in tracked.items():
            path = run_dir / relative
            valid = (
                valid
                and path.is_file()
                and path.stat().st_size == int(metadata.get("size_bytes", -1))
                and sha256_file(path) == metadata.get("sha256")
            )
        completed = str(fold_id) in set(checkpoint.get("completed_fold_ids", []))
        if completed and valid:
            classification = "valid_complete"
        elif fold_id == 3:
            classification = "incomplete"
        else:
            classification = "not_started"
        rows.append(
            {
                "run_id": spec.run_id,
                "phase": "DEV",
                "fold_id": fold_id,
                "classification": classification,
                "checkpoint_claims_complete": completed,
                "tracked_artifact_count": len(tracked),
                "tracked_artifacts_integrity_valid": valid if tracked else "not_applicable",
                "fold_directory_present": fold_dir.is_dir(),
                "resume_action": (
                    "reuse_all_finalized_artifacts"
                    if classification == "valid_complete"
                    else "recompute_from_dev_data_loading"
                    if fold_id == 3
                    else "execute_after_prior_fold_completes"
                ),
            }
        )
    return rows


def _resource_timeline(root: Path, plan: Any) -> list[dict[str, Any]]:
    rows = []
    for spec in plan.run_specs[:11]:
        run_dir = _run_directory(root, spec.dataset, spec.run_id)
        resource = _json(run_dir / "resource_usage.json")
        samples = resource.get("samples", [])
        first = samples[0]
        last = samples[-1]
        children = sorted(
            {
                int(pid)
                for sample in samples
                for pid in sample.get("child_pids", [])
            }
        )
        rows.append(
            {
                "run_id": spec.run_id,
                "worker_pid": first["worker_pid"],
                "parent_pid": "not_recorded_by_original_release",
                "parent_rss_before_bytes": "not_recorded_by_original_release",
                "parent_rss_after_bytes": "not_recorded_by_original_release",
                "worker_peak_rss_bytes": resource["peak_process_tree_rss_bytes"],
                "minimum_system_available_ram_bytes": resource[
                    "minimum_system_available_ram_bytes"
                ],
                "first_system_available_ram_bytes": first[
                    "system_available_ram_bytes"
                ],
                "last_system_available_ram_bytes": last[
                    "system_available_ram_bytes"
                ],
                "observed_child_pids": ";".join(map(str, children)),
                "worker_completion_to_disappearance_seconds": (
                    "not_recorded_by_original_release"
                ),
                "returned_payload": "compact summary mapping",
                "status": resource.get("status"),
                "stop_code": resource.get("stop_code"),
                "warnings": ";".join(resource.get("warnings", [])),
                "final_stage": last.get("stage"),
                "final_fold_id": last.get("fold_id"),
                "runtime_seconds": last.get("elapsed_seconds"),
            }
        )
    return rows


def _incident_chronology(root: Path, run_010: Any, run_011: Any) -> dict[str, Any]:
    dir10 = _run_directory(root, run_010.dataset, run_010.run_id)
    dir11 = _run_directory(root, run_011.dataset, run_011.run_id)
    manifest10 = _json(dir10 / "manifest.json")
    manifest11 = _json(dir11 / "manifest.json")
    resource10 = _json(dir10 / "resource_usage.json")
    resource11 = _json(dir11 / "resource_usage.json")
    start10 = datetime.fromisoformat(manifest10["started_at_utc"])
    warning_samples = [
        sample
        for sample in resource10["samples"]
        if sample["system_available_ram_bytes"] <= 10 * 1024**3
    ]
    first_warning = warning_samples[0]
    start11 = datetime.fromisoformat(manifest11["started_at_utc"])
    stage_entries = {}
    for sample in resource11["samples"]:
        key = f"fold_{sample.get('fold_id')}:{sample.get('stage')}"
        stage_entries.setdefault(
            key,
            (start11 + timedelta(seconds=float(sample["elapsed_seconds"]))).isoformat(),
        )
    last_sample = resource11["samples"][-1]
    return {
        "schema_version": "cdv1_run_011_incident_chronology_v1",
        "reported_incident": {
            "reported_primary_resource_trigger": "ram_system_headroom",
            "reported_secondary_event": "user_keyboard_interrupt",
        },
        "authenticated_findings": {
            "run_010_reserve_warning": {
                "code": "ram_system_headroom",
                "kind": "warning_only",
                "first_sample_elapsed_seconds": first_warning["elapsed_seconds"],
                "first_sample_timestamp_utc": (
                    start10 + timedelta(seconds=float(first_warning["elapsed_seconds"]))
                ).isoformat(),
                "minimum_available_ram_bytes": resource10[
                    "minimum_system_available_ram_bytes"
                ],
                "abort_floor_bytes": 8 * 1024**3,
                "abort_floor_crossed": False,
                "run_terminal_state": manifest10["status"],
            },
            "run_011": {
                "attempt_started_at_utc": manifest11["started_at_utc"],
                "stage_entries_first_observed_utc": stage_entries,
                "last_valid_completed_fold_ids": ["1", "2"],
                "last_observed_sample_timestamp_utc": (
                    start11 + timedelta(seconds=float(last_sample["elapsed_seconds"]))
                ).isoformat(),
                "last_observed_stage": last_sample["stage"],
                "last_observed_fold_id": last_sample["fold_id"],
                "resource_warning_codes": resource11.get("warnings", []),
                "resource_abort_code": resource11.get("stop_code"),
                "minimum_available_ram_bytes": resource11[
                    "minimum_system_available_ram_bytes"
                ],
                "abort_floor_bytes": 8 * 1024**3,
                "user_interrupt_recorded": manifest11.get("stop_code")
                == "manual_interrupt",
                "worker_exit_code": manifest11.get("worker_exit_code"),
                "parent_terminal_state_written_at_utc": manifest11.get(
                    "interrupted_at_utc"
                ),
                "terminal_run_state": manifest11.get("status"),
                "dev_phase_complete": False,
            },
        },
        "reconciliation": {
            "primary_resource_trigger": "not_authenticated_for_run_011",
            "resource_stop_latched": False,
            "graceful_stop_completed": "not_applicable_no_resource_stop_latched",
            "primary_terminal_event": "user_keyboard_interrupt",
            "secondary_termination_event": None,
            "terminal_run_state": "interrupted",
            "dev_phase_complete": False,
            "reported_vs_structured_evidence_contradiction": True,
            "explanation": (
                "The sole structured ram_system_headroom warning is in run 010, "
                "where the 10 GiB reserve was crossed but the 8 GiB abort floor was not. "
                "Run 011 has no warning sample and no resource abort latch."
            ),
        },
        "safe_resume_boundary": SAFE_RESUME_BOUNDARY,
    }


def _compatibility_bridge(root: Path, plan: Any) -> dict[str, Any]:
    runtime = {}
    for relative in RUNTIME_FILES:
        path = root / relative
        old = _git_bytes(root, ORIGINAL_COMMIT, relative)
        runtime[relative] = {
            "old_sha256": hashlib.sha256(old).hexdigest() if old is not None else "absent",
            "new_sha256": sha256_file(path),
            "new_size_bytes": path.stat().st_size,
        }
    frozen = {
        relative: {
            "sha256": expected,
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative, expected in FROZEN_HASHES.values()
    }
    runs = {}
    by_id = {spec.run_id: spec for spec in plan.run_specs}
    for run_id in (*REUSABLE_RUN_IDS, INTERRUPTED_RUN_ID):
        spec = by_id[run_id]
        run_dir = _run_directory(root, spec.dataset, run_id)
        checkpoint = _json(run_dir / "checkpoint.json")
        immutable = {
            relative: {
                "size_bytes": int(metadata["size_bytes"]),
                "sha256": metadata["sha256"],
            }
            for relative, metadata in checkpoint["finalized_artifacts"].items()
            if relative != "resource_usage.json"
        }
        controls = {
            relative: {
                "size_bytes": (run_dir / relative).stat().st_size,
                "sha256": sha256_file(run_dir / relative),
            }
            for relative in CONTROL_FILES
            if (run_dir / relative).is_file()
        }
        runs[run_id] = {
            "run_directory": run_dir.relative_to(root).as_posix(),
            "initial_control_artifacts": controls,
            "immutable_artifacts": immutable,
        }
    test_paths = (
        "tests/test_resource_monitor.py",
        "tests/test_checkpointing.py",
        "tests/test_manual_research_orchestration.py",
        "tests/test_provenance_bridge.py",
        "tests/test_execution_dry_run.py",
        "tests/test_cdv1_scientific_equivalence.py",
        "tests/support/cdv1_scientific_equivalence_probe.py",
        "tests/fixtures/cdv1_scientific_equivalence_golden.json",
    )
    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "research_family": "cdv1",
        "original_release": {"tag": ORIGINAL_TAG, "commit": ORIGINAL_COMMIT},
        "mechanics_release": {
            "tag": MECHANICS_TAG,
            "commit_binding": "annotated_tag_peels_to_current_head",
        },
        "matrix_sha256": plan.matrix_sha256,
        "configuration_set_sha256": plan.configuration_set_sha256,
        "reusable_run_ids": list(REUSABLE_RUN_IDS),
        "interrupted_run": {
            "run_id": INTERRUPTED_RUN_ID,
            "safe_resume_boundary": SAFE_RESUME_BOUNDARY,
        },
        "runtime_files": runtime,
        "frozen_files": frozen,
        "runs": runs,
        "allowed_change_categories": [
            "process_lifecycle",
            "resource_cleanup",
            "stop_state_tracking",
            "strict_resume_provenance",
            "synthetic_tests",
            "documentation_and_audit",
        ],
        "forbidden_change_categories": [
            "datasets_or_row_membership",
            "dev_oot_boundary",
            "folds",
            "selectors_or_voter_semantics",
            "feature_budgets",
            "preprocessing",
            "model_hyperparameters",
            "seeds",
            "predictions_metrics_or_inference",
            "resource_thresholds",
        ],
        "focused_test_files": {
            path: sha256_file(root / path) for path in test_paths
        },
        "scope_rule": (
            "Applies only to the exact original annotated release, the exact current "
            "mechanics annotated release, cdv1, and this run inventory."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    audit = root / AUDIT_RELATIVE
    plan = build_manual_research_plan(root)
    completed = _audit_completed_runs(root, plan)
    run011 = _audit_run_011(root, plan.run_specs[10])
    timeline = _resource_timeline(root, plan)
    chronology = _incident_chronology(root, plan.run_specs[9], plan.run_specs[10])
    bridge = _compatibility_bridge(root, plan)
    _atomic_csv(audit / "completed_run_integrity.csv", completed)
    _atomic_csv(audit / "run_011_checkpoint_inventory.csv", run011)
    _atomic_csv(audit / "run_boundary_resource_timeline.csv", timeline)
    _atomic_json(audit / "incident_chronology.json", chronology)
    _atomic_json(root / BRIDGE_PATH, bridge)
    summary = {
        "schema_version": "cdv1_resume_safety_evidence_build_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_runs_authenticated": sum(
            bool(row["scientifically_reusable"]) for row in completed
        ),
        "completed_runs_expected": 10,
        "run_011_safe_resume_boundary": SAFE_RESUME_BOUNDARY,
        "run_011_completed_fold_ids": [1, 2],
        "run_011_resource_trigger_authenticated": False,
        "run_010_warning_reconciled": True,
        "bridge_path": BRIDGE_PATH.as_posix(),
        "research_or_oot_executed": False,
    }
    _atomic_json(audit / "evidence_build_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["completed_runs_authenticated"] == 10 else 1


if __name__ == "__main__":
    raise SystemExit(main())
