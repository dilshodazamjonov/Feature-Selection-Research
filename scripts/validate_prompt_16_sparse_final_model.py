"""Run the one authorized full-scale DEV-only sparse final-model certification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

AUDIT_RELATIVE_ROOT = Path(
    "cleanup/audits/prompt_16_sparse_final_model_preprocessing_v6"
)
PRODUCTION_RELATIVE_ROOT = Path(
    "results/prompt_16_homecredit_model_stability_2024/oot_final_amended_v1"
)
GIB = 1024**3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tree_identity(root: Path) -> dict[str, Any]:
    from credit_risk_fs.data.homecredit_model_stability_2024.contract import (
        canonical_sha256,
        file_sha256,
    )

    artifacts = [
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "byte_size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return {
        "root": root.relative_to(PROJECT_ROOT).as_posix(),
        "file_count": len(artifacts),
        "identity_sha256": canonical_sha256(artifacts),
        "artifacts": artifacts,
    }


def _cell_accounting(root: Path) -> dict[str, Any]:
    complete: list[int] = []
    unavailable: list[int] = []
    for order in range(1, 35):
        phase = "classical" if order <= 30 else "supplemental"
        path = root / phase / "evaluations" / f"cell_{order:03d}"
        if not (path / "_SUCCESS").is_file() or not (path / "status.json").is_file():
            continue
        status = json.loads((path / "status.json").read_text(encoding="utf-8"))
        if status.get("status") == "complete":
            complete.append(order)
        elif status.get("status") in {"unavailable", "failed"}:
            unavailable.append(order)
    accounted = sorted(complete + unavailable)
    return {
        "complete": complete,
        "unavailable": unavailable,
        "next_cell": next(
            (order for order in range(1, 35) if order not in accounted), None
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-root", type=Path, default=PROJECT_ROOT / AUDIT_RELATIVE_ROOT
    )
    args = parser.parse_args(argv)

    from credit_risk_fs.data.homecredit_model_stability_2024.contract import (
        file_sha256,
    )
    from credit_risk_fs.experiments.atomic_io import (
        write_csv_atomic,
        write_json_atomic,
        write_text_atomic,
    )
    from credit_risk_fs.experiments.ram_control import load_ram_control_policy
    from credit_risk_fs.experiments.research_logging import ResearchLogSession
    from credit_risk_fs.experiments.resource_monitor import (
        supervise_worker,
        wait_for_inter_run_readiness,
    )
    from credit_risk_fs.experiments.resource_policy import (
        detect_hardware,
        load_execution_policy,
        resolve_execution_policy,
    )

    audit_root = args.audit_root.resolve()
    if audit_root.exists():
        raise RuntimeError(f"certification root already exists: {audit_root}")
    production_root = PROJECT_ROOT / PRODUCTION_RELATIVE_ROOT
    before_tree = _tree_identity(production_root)
    before_accounting = _cell_accounting(production_root)
    controller_path = production_root / "controller_status.json"
    controller_sha_before = file_sha256(controller_path)
    if before_accounting != {
        "complete": [],
        "unavailable": [1, 2],
        "next_cell": 3,
    }:
        raise RuntimeError(f"production OOT accounting changed: {before_accounting}")

    audit_root.mkdir(parents=True, exist_ok=False)
    validation_root = audit_root / "dev_only_certification"
    logs_root = audit_root / "logs"
    logs_root.mkdir()
    configured = load_execution_policy(
        PROJECT_ROOT, "configs/execution/prompt_16_final_oot_v1.yaml"
    )
    capacity = detect_hardware(production_root.parent, audit_root)
    policy = resolve_execution_policy(configured, capacity)
    ram_policy = load_ram_control_policy(
        PROJECT_ROOT,
        "configs/execution/prompt_16_final_oot_ram_wait_v1.yaml",
        total_physical_ram_bytes=int(capacity.total_ram_gb * GIB),
    )
    if policy.memory.abort_process_tree_rss_gb != 24.0:
        raise RuntimeError("certification process-tree cap is not 24 GiB")
    if policy.memory.abort_if_system_available_below_gb != 4.0:
        raise RuntimeError("certification hard available-RAM floor is not 4 GiB")
    if ram_policy.emergency_margin_bytes != 6 * GIB:
        raise RuntimeError("certification soft available-RAM threshold is not 6 GiB")
    if ram_policy.recovery_threshold_bytes != 8 * GIB:
        raise RuntimeError("certification recovery threshold is not 8 GiB")
    readiness = wait_for_inter_run_readiness(
        policy=policy,
        results_root=production_root.parent,
        temp_root=audit_root,
        ram_control_policy=ram_policy,
    )
    if not readiness.ready:
        raise RuntimeError(f"DEV-only certification readiness failed: {readiness.stop_code}")

    terminal_log = logs_root / "dev_only_certification.log"
    started_at = _utc_now()
    with ResearchLogSession(
        terminal_log,
        repository_root=PROJECT_ROOT,
        command_arguments=[str(value) for value in sys.argv],
    ) as session:
        result = supervise_worker(
            worker_target=(
                "credit_risk_fs.experiments.prompt_16_third_dataset:"
                "run_sparse_final_model_dev_certification_worker"
            ),
            worker_kwargs={
                "repository_root": str(PROJECT_ROOT),
                "validation_root": str(validation_root),
            },
            policy=policy,
            results_root=production_root.parent,
            temp_root=audit_root,
            run_association="prompt16:sparse_final_model:dev_only_certification",
            heartbeat_interval_seconds=30.0,
            max_wall_clock_seconds=None,
            enforce_memory_limits=True,
            enforce_process_tree_rss_limit=True,
            ram_control_policy=ram_policy,
        )
        if result.status != "completed":
            session.finish(
                "session_failed",
                level="ERROR",
                message="DEV-only sparse certification failed",
                stop_code=result.stop_code,
            )
            raise RuntimeError(
                f"DEV-only certification failed: {result.stop_code}: {result.worker_error}"
            )
        session.finish(
            "session_completed",
            message="DEV-only sparse certification worker completed",
        )

    summary = result.to_dict()
    samples = summary.pop("samples")
    write_json_atomic(
        audit_root / "supervisor_summary.json", summary, overwrite=False
    )
    write_csv_atomic(
        audit_root / "resource_samples.csv",
        pd.DataFrame(samples),
        overwrite=False,
    )
    worker_report_path = validation_root / "worker_report.json"
    worker_report = json.loads(worker_report_path.read_text(encoding="utf-8"))
    if worker_report["scope"]["locked_oot_rows_loaded"] != 0:
        raise RuntimeError("DEV-only worker loaded locked OOT rows")
    if worker_report["scope"]["locked_oot_outcomes_inspected"] is not False:
        raise RuntimeError("DEV-only worker inspected locked OOT outcomes")
    peak = int(result.peak_process_tree_rss_bytes)
    minimum_available = int(result.minimum_system_available_ram_bytes or 0)
    if peak > 16 * GIB:
        raise RuntimeError(
            f"in-memory CSR exceeded the 16 GiB fallback trigger: {peak / GIB:.3f} GiB"
        )
    if peak >= 24 * GIB:
        raise RuntimeError("in-memory CSR violated the 24 GiB hard acceptance bound")
    if minimum_available < 8 * GIB:
        raise RuntimeError(
            "in-memory CSR left less than the required 8 GiB system RAM available"
        )
    after_tree = _tree_identity(production_root)
    after_accounting = _cell_accounting(production_root)
    controller_sha_after = file_sha256(controller_path)
    if after_tree["identity_sha256"] != before_tree["identity_sha256"]:
        raise RuntimeError("DEV-only certification modified the production OOT tree")
    if after_accounting != before_accounting:
        raise RuntimeError("DEV-only certification changed OOT cell accounting")
    if controller_sha_after != controller_sha_before:
        raise RuntimeError("DEV-only certification changed controller status")

    report = {
        "schema_version": "prompt_16_sparse_final_model_resource_certification_v1",
        "status": "validated_in_memory_csr_no_disk_fallback_required",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "scientific_method_changed": False,
        "representation_only_change": True,
        "resource_contract_gib": {
            "process_tree_hard_cap": 24,
            "system_available_hard_floor": 4,
            "system_available_soft_pause": 6,
            "system_available_recovery": 8,
            "disk_fallback_trigger_peak_rss": 16,
            "certification_minimum_available": 8,
        },
        "measurements": {
            "peak_process_tree_rss_bytes": peak,
            "peak_process_tree_rss_gib": peak / GIB,
            "minimum_system_available_ram_bytes": minimum_available,
            "minimum_system_available_ram_gib": minimum_available / GIB,
            "ram_wait_count": int(result.ram_wait_count),
            "stop_code": result.stop_code,
            "worker_status": result.status,
        },
        "worker_report": {
            "path": worker_report_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": file_sha256(worker_report_path),
        },
        "supervisor_summary_sha256": file_sha256(
            audit_root / "supervisor_summary.json"
        ),
        "resource_samples_sha256": file_sha256(
            audit_root / "resource_samples.csv"
        ),
        "production_tree_before": before_tree,
        "production_tree_after_identity_sha256": after_tree["identity_sha256"],
        "production_cell_accounting_before": before_accounting,
        "production_cell_accounting_after": after_accounting,
        "controller_status_sha256_before": controller_sha_before,
        "controller_status_sha256_after": controller_sha_after,
        "production_oot_cell_executed": False,
        "selector_rerun": False,
        "llm_request_count": 0,
        "locked_oot_rows_loaded": 0,
        "locked_oot_outcomes_inspected": False,
        "disk_backed_csr_required": False,
    }
    report_path = audit_root / "dev_only_resource_certification.json"
    write_json_atomic(report_path, report, overwrite=False)
    write_text_atomic(
        audit_root / "_VALIDATION_SUCCESS",
        json.dumps({"report_sha256": file_sha256(report_path)}, sort_keys=True)
        + "\n",
        overwrite=False,
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
