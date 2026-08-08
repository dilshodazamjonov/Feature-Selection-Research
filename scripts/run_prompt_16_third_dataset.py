"""Supervised CLI for the frozen Prompt-16 third-dataset execution."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an authenticated, resource-supervised Prompt-16 operation"
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)

    registry = subparsers.add_parser(
        "registry", help="Print the canonical data-free 27/30 execution registry"
    )
    registry.add_argument("--protocol-lock", type=Path, required=True)

    matrix = subparsers.add_parser(
        "matrix", help="Build and authenticate the real modeling matrix"
    )
    matrix.add_argument("--plan", type=Path, required=True)

    phase = subparsers.add_parser(
        "phase", help="Run one locked pilot/DEV fold or the one-time OOT phase"
    )
    phase.add_argument("--plan", type=Path, required=True)
    phase.add_argument("--phase", choices=("pilot", "dev", "oot"), required=True)
    phase.add_argument("--fold-id", type=int, choices=(1, 2, 3, 4, 5))
    phase.add_argument("--oot-analysis-plan", type=Path)
    return parser


def _write_supervision(
    *,
    result: Any,
    log_root: Path,
    name: str,
    write_json_atomic: Any,
    write_csv_atomic: Any,
) -> None:
    import pandas as pd

    log_root.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    samples = payload.pop("samples", [])
    write_json_atomic(log_root / f"{name}_supervisor_summary.json", payload)
    write_csv_atomic(
        log_root / f"{name}_resource_samples.csv",
        pd.DataFrame(samples),
    )


def _run_supervised(args: argparse.Namespace) -> int:
    from credit_risk_fs.experiments.atomic_io import write_csv_atomic, write_json_atomic
    from credit_risk_fs.experiments.prompt_16_third_dataset import (
        Prompt16ExecutionError,
        acquire_execution_lock,
        canonical_registry,
        directory_size_bytes,
        free_disk_bytes,
        load_execution_plan,
        record_phase_resource_infeasibility,
        release_execution_lock,
    )
    from credit_risk_fs.experiments.resource_monitor import (
        RAM_PROCESS_LIMIT,
        WALL_CLOCK_LIMIT,
        supervise_worker,
    )
    from credit_risk_fs.experiments.resource_policy import (
        GIB,
        detect_hardware,
        load_execution_policy,
        resolve_execution_policy,
    )

    if args.operation == "registry":
        print(
            json.dumps(
                canonical_registry(args.protocol_lock), indent=2, ensure_ascii=False
            )
        )
        return 0

    plan_path = args.plan.resolve()
    plan = load_execution_plan(plan_path)
    paths = plan["paths"]
    protocol_lock = Path(paths["protocol_lock"])
    if not protocol_lock.is_file():
        raise Prompt16ExecutionError("planned protocol lock is missing")
    policy_path = Path(plan["resource_controls"]["execution_policy_path"])
    temp_root = Path(paths["temp_root"])
    log_root = Path(paths["log_root"])
    temp_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    if args.operation == "matrix":
        output_root = Path(paths["matrix_root"])
        target = "credit_risk_fs.experiments.prompt_16_third_dataset:run_matrix_worker"
        worker_kwargs = {
            "input_root": paths["raw_dataset_root"],
            "matrix_root": paths["matrix_root"],
            "protocol_lock": paths["protocol_lock"],
            "shard_rows": int(plan["adapter"]["case_shard_rows"]),
            "resource_event_path": str(log_root / "matrix_resource_events.jsonl"),
        }
        max_seconds = float(plan["wall_time_limits_seconds"]["matrix_total"])
        stage_wall_clock_limits = {
            "matrix_inventory_and_build": float(
                plan["wall_time_limits_seconds"]["raw_inventory_scan"]
            ),
            "matrix_base_scan": float(
                plan["wall_time_limits_seconds"]["base_scan"]
            ),
            "matrix_depth_0_scan": float(
                plan["wall_time_limits_seconds"]["each_depth_0_scan"]
            ),
            "matrix_depth_1_scan_and_aggregation": float(
                plan["wall_time_limits_seconds"][
                    "each_depth_1_scan_and_aggregation"
                ]
            ),
            "matrix_checkpoint_unit": float(
                plan["wall_time_limits_seconds"]["each_checkpoint_write_or_read"]
            ),
            "matrix_publication_unit": float(
                plan["wall_time_limits_seconds"]["each_checkpoint_write_or_read"]
            ),
            "matrix_authentication": float(
                plan["wall_time_limits_seconds"]["raw_inventory_scan"]
            ),
        }
        name = "matrix"
    else:
        phase = str(args.phase)
        if phase == "pilot":
            if args.fold_id != 1 or args.oot_analysis_plan is not None:
                raise Prompt16ExecutionError("pilot command must use fold 1 and no OOT plan")
            output_root = Path(paths["pilot_root"]) / "fold_1"
            max_seconds = float(plan["wall_time_limits_seconds"]["pilot_total"])
            name = "pilot_fold_1"
        elif phase == "dev":
            if args.fold_id not in {1, 2, 3, 4, 5} or args.oot_analysis_plan is not None:
                raise Prompt16ExecutionError("DEV command requires one fold in 1..5 and no OOT plan")
            output_root = Path(paths["dev_root"]) / f"fold_{args.fold_id}"
            max_seconds = float(plan["wall_time_limits_seconds"]["dev_fold_total"])
            name = f"dev_fold_{args.fold_id}"
        else:
            if args.fold_id is not None or args.oot_analysis_plan is None:
                raise Prompt16ExecutionError("OOT command requires the frozen analysis plan and no fold")
            output_root = Path(paths["oot_root"])
            max_seconds = float(plan["wall_time_limits_seconds"]["oot_total"])
            name = "oot"
        target = "credit_risk_fs.experiments.prompt_16_third_dataset:run_phase_worker"
        worker_kwargs = {
            "matrix_root": paths["matrix_root"],
            "output_root": str(output_root),
            "protocol_lock": paths["protocol_lock"],
            "phase": phase,
            "fold_id": args.fold_id,
            "oot_analysis_plan": (
                None if args.oot_analysis_plan is None else str(args.oot_analysis_plan.resolve())
            ),
        }
        stage_wall_clock_limits = dict(
            canonical_registry(protocol_lock)["resource_controls"][
                "wall_clock_limits_seconds"
            ]
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = free_disk_bytes(output_root.parent)
    forecast_total = int(plan["resource_forecast"]["forecast_total_output_bytes"])
    existing_bytes = sum(
        directory_size_bytes(path)
        for path in plan["resource_forecast"]["forecast_roots"]
    )
    remaining_forecast = max(0, forecast_total - existing_bytes)
    forecast_factor = float(
        plan["resource_controls"]["disk_remaining_output_safety_factor"]
    )
    floor_bytes = max(
        int(plan["resource_controls"]["disk_free_hard_floor_bytes"]),
        int(math.ceil(forecast_factor * remaining_forecast)),
    )
    if free_bytes < floor_bytes:
        raise Prompt16ExecutionError(
            f"results-volume free space is below the frozen floor: {free_bytes} < {floor_bytes}"
        )
    configured = load_execution_policy(PROJECT_ROOT, policy_path)
    capacity = detect_hardware(output_root.parent, temp_root)
    policy = resolve_execution_policy(configured, capacity)
    if policy.memory.abort_process_tree_rss_gb > 24.0 + 1e-12:
        raise Prompt16ExecutionError("resolved RSS cap exceeds the Prompt-16 hard ceiling")
    if policy.memory.abort_if_system_available_below_gb < 8.0 - 1e-12:
        raise Prompt16ExecutionError("resolved system-available floor weakened below 8 GiB")
    if policy.parallelism.concurrent_experiment_runs != 1:
        raise Prompt16ExecutionError("Prompt-16 requires one experiment cell at a time")
    if policy.parallelism.concurrent_folds != 1:
        raise Prompt16ExecutionError("Prompt-16 requires one fold at a time")
    if policy.parallelism.estimator_threads > 4:
        raise Prompt16ExecutionError("Prompt-16 permits at most four estimator threads")

    lock = acquire_execution_lock(output_root)
    try:
        result = supervise_worker(
            worker_target=target,
            worker_kwargs=worker_kwargs,
            policy=policy,
            results_root=output_root.parent,
            temp_root=temp_root,
            run_association=f"prompt16:{name}",
            max_wall_clock_seconds=max_seconds,
            stage_wall_clock_limits_seconds=stage_wall_clock_limits,
            enforce_memory_limits=True,
        )
        infeasibility = None
        if (
            args.operation == "phase"
            and phase in {"pilot", "dev"}
            and result.stop_code in {RAM_PROCESS_LIMIT, WALL_CLOCK_LIMIT}
        ):
            supervisor_evidence = result.to_dict()
            supervisor_evidence.pop("samples", None)
            infeasibility = record_phase_resource_infeasibility(
                matrix_root=paths["matrix_root"],
                output_root=output_root,
                protocol_lock=paths["protocol_lock"],
                phase=phase,
                fold_id=int(args.fold_id),
                stopped_stage=result.final_stage,
                stopped_scope=result.final_fold_id,
                supervisor_evidence=supervisor_evidence,
            )
        _write_supervision(
            result=result,
            log_root=log_root,
            name=name,
            write_json_atomic=write_json_atomic,
            write_csv_atomic=write_csv_atomic,
        )
    finally:
        release_execution_lock(lock)
    print(
        json.dumps(
            {
                "operation": name,
                "status": result.status,
                "stop_code": result.stop_code,
                "peak_process_tree_rss_bytes": result.peak_process_tree_rss_bytes,
                "minimum_system_available_ram_bytes": result.minimum_system_available_ram_bytes,
                "output_root": str(output_root),
                "log_root": str(log_root),
                "sealed_resource_infeasibility": infeasibility,
            },
            indent=2,
        )
    )
    return 0 if result.status == "completed" else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _run_supervised(args)


if __name__ == "__main__":
    raise SystemExit(main())
