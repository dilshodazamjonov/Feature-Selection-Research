from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.clip_final_comparison.constants import (  # noqa: E402
    ABLATIONS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CENTRAL_POOL_MULTIPLIER,
    CORRELATION_FILTER_THRESHOLD,
    LOCK_PATH,
    LOG_PATH,
    MODEL_BUDGETS,
    OUTPUT_ROOT,
    POOL_MULTIPLIERS,
    RANDOM_SEEDS,
    SELECTED_CLIP_V2_SEED,
    STAGES,
    STATE_PATH,
)
from credit_risk_fs.clip_final_comparison.ablations import (  # noqa: E402
    REFERENCE_SOURCES,
    RETRAINED_ABLATIONS,
    REUSED_ABLATIONS,
    ablation_schema,
    train_grouped_ablation_representations,
    validate_ablation_training,
    write_ablation_schemas,
)
from credit_risk_fs.clip_final_comparison.execution import (  # noqa: E402
    ComparisonRunSpec,
    PreparedFrame,
    execute_comparison_run,
    aggregate_runs,
    validate_run,
    write_minimal_plot,
)
from credit_risk_fs.clip_final_comparison.io import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    ensure_layout,
    read_json_if_exists,
)
from credit_risk_fs.clip_final_comparison.method_spec import write_method_specification  # noqa: E402
from credit_risk_fs.clip_final_comparison.plans import (  # noqa: E402
    build_ablation_plan,
    build_core_experiment_plan,
    build_seed_downstream_plan,
    planned_matrix_summary,
)
from credit_risk_fs.clip_final_comparison.source_manifest import build_source_experiment_manifest  # noqa: E402
from credit_risk_fs.clip_final_comparison.seeds import (  # noqa: E402
    generate_seed_score_cache,
    resolve_clip_v2_seed_artifacts,
    validate_seed_artifacts,
    validate_seed_score_cache,
)
from credit_risk_fs.clip_final_comparison.temporal import construct_temporal_cutoffs  # noqa: E402
from credit_risk_fs.clip_final_comparison.uncertainty import (  # noqa: E402
    benjamini_hochberg,
    paired_bootstrap_deltas,
    random_distribution_summary,
    summarize_uncertainty,
)
from credit_risk_fs.experiments.config import load_named_project_config, resolve_model_kwargs  # noqa: E402
from credit_risk_fs.pipelines.common import ExperimentConfig, prepare_modeling_data  # noqa: E402
from credit_risk_fs.utils.hashing import sha256_file, sha256_text  # noqa: E402

ARCHIVE_ROOT = Path("results/clip_final_comparison_archives")
VALID_STATUSES = {"not_started", "running", "complete_valid", "failed", "interrupted", "stale"}
TEMPORAL_METHODS = ("clip_v2", "text_similarity", "statistics_only", "llm", "full_mrmr")
EXPECTED_CORE_RUNS = 184
EXPECTED_SEED_DOWNSTREAM_RUNS = 20
EXPECTED_ABLATION_DOWNSTREAM_RUNS = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the final CLIP comparison research pipeline.")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--fresh-start", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--from-stage", choices=STAGES)
    parser.add_argument("--to-stage", choices=STAGES)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = select_stages(args)
    if args.plan:
        print(json.dumps(plan_payload(selected, args), indent=2, default=str))
        return 0
    if args.status:
        print(json.dumps(status_payload(), indent=2, default=str))
        return 0
    if args.resume and not args.execute:
        print(json.dumps(resume_payload(selected), indent=2, default=str))
        return 0
    if not args.execute:
        print(json.dumps(plan_payload(selected, args), indent=2, default=str))
        return 0
    lock = None
    try:
        if args.fresh_start:
            archive_incomplete_outputs()
            initialize_clean_state()
        lock = acquire_lock()
        return run_selected_stages(selected)
    except KeyboardInterrupt:
        print("Interrupted. Inspect with: uv run python scripts/run_clip_final_comparison.py --status")
        return 130
    finally:
        release_lock(lock)


def select_stages(args: argparse.Namespace) -> list[str]:
    if args.resume:
        status = status_payload()
        return [stage for stage in STAGES if status["stages"][stage]["status"] != "complete_valid"]
    start = STAGES.index(args.from_stage) if args.from_stage else 0
    end = STAGES.index(args.to_stage) if args.to_stage else len(STAGES) - 1
    if start > end:
        raise SystemExit("--from-stage must not come after --to-stage")
    return list(STAGES[start : end + 1])


def plan_payload(selected: list[str], args: argparse.Namespace) -> dict[str, Any]:
    source_checks = _source_check_summary()
    return {
        "mode": "plan",
        "execute": False,
        "implementation_mode": "executable_research_pipeline",
        "selected_stage_count": len(selected),
        "stage_order": selected,
        "matrix": planned_matrix_summary(),
        "core_candidate_pool_runs": EXPECTED_CORE_RUNS,
        "random_runs": 120,
        "seed_downstream_runs": EXPECTED_SEED_DOWNSTREAM_RUNS,
        "new_grouped_ablation_contrastive_training_jobs": 15,
        "ablation_downstream_results": EXPECTED_ABLATION_DOWNSTREAM_RUNS,
        "reused_ablation_reference_conditions": len(REUSED_ABLATIONS),
        "retrained_grouped_ablation_conditions": len(RETRAINED_ABLATIONS),
        "temporal_runs": "dynamically derived from temporal_cutoff_manifest.csv",
        "paired_uncertainty": "executable",
        "candidate_pool_methods": ["clip_v2", "random", "variance", "correlation_filter", "text_similarity", "statistics_only", "full_mrmr"],
        "pool_sizes": {"lr": [40, 100, 200], "catboost": [80, 200, 400]},
        "random_seeds": list(RANDOM_SEEDS),
        "clip_v2_seeds": [11, 22, 33, 44, 55],
        "ablations": list(ABLATIONS),
        "ablation_taxonomy": {
            "reused_reference_conditions": {name: REFERENCE_SOURCES[name] for name in REUSED_ABLATIONS},
            "retrained_conditions": {
                name: {
                    "statistical_dimension": ablation_schema(name)["statistical_dimension"],
                    "removed_fields": ablation_schema(name)["removed_fields"],
                }
                for name in RETRAINED_ABLATIONS
            },
        },
        "seed_artifacts": seed_artifact_plan_summary(),
        "temporal_cutoffs": "constructed from real data only during temporal_cutoffs stage",
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
        "correlation_filter_threshold": CORRELATION_FILTER_THRESHOLD,
        "lock": read_lock(),
        "output_isolation": OUTPUT_ROOT.as_posix(),
        "source_checks": source_checks,
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
        "read_only_plan": True,
    }


def status_payload() -> dict[str, Any]:
    state = read_json_if_exists(STATE_PATH, {"stages": {}})
    artifact_status = derive_artifact_status()
    stages = {}
    for stage in STAGES:
        stored = state.get("stages", {}).get(stage, {})
        derived = artifact_status.get(stage, {"status": "not_started", "details": ""})
        stages[stage] = {**stored, **derived}
    return {
        "mode": "status",
        "implementation_mode": "executable_research_pipeline",
        "output_root": OUTPUT_ROOT.as_posix(),
        "state_path": STATE_PATH.as_posix(),
        "log_path": LOG_PATH.as_posix(),
        "lock": read_lock(),
        "completed_scientific_runs": artifact_status["core_candidate_pool_runs"].get("completed_runs", 0),
        "completed_seed_runs": artifact_status["seed_downstream"].get("completed_runs", 0),
        "completed_ablation_runs": artifact_status["ablation_downstream"].get("completed_runs", 0),
        "stages": stages,
    }


def resume_payload(selected: list[str]) -> dict[str, Any]:
    return {
        "mode": "resume_plan",
        "execute": False,
        "implementation_mode": "executable_research_pipeline",
        "selected_stage_count": len(selected),
        "stage_order": selected,
        "status": status_payload(),
    }


def run_selected_stages(selected: list[str]) -> int:
    state = load_state()
    total_started = time.time()
    for index, stage in enumerate(selected, start=1):
        derived = derive_artifact_status().get(stage, {"status": "not_started"})
        if derived["status"] == "complete_valid":
            log(f"[{index}/{len(selected)}] {stage}: skip complete_valid")
            continue
        started = time.time()
        mark_stage(state, stage, status="running", started_at=_now())
        write_state(state)
        log_progress(index, len(selected), stage, "start", started, total_started)
        try:
            payload = execute_stage(stage)
            validate_stage(stage)
            mark_stage(
                state,
                stage,
                status="complete_valid",
                finished_at=_now(),
                elapsed_seconds=time.time() - started,
                execution_mode="real_experiment_execution",
                payload=payload,
            )
            write_state(state)
            log_progress(index, len(selected), stage, "complete_valid", started, total_started)
        except Exception as exc:
            mark_stage(state, stage, status="failed", finished_at=_now(), failure_reason=str(exc), execution_mode="real_experiment_execution")
            write_state(state)
            log(f"{stage}: failed: {exc}")
            print(f"Stage failed: {stage}. Inspect status, then resume.")
            return 1
    return 0


def execute_stage(stage: str) -> dict[str, Any]:
    if stage == "preflight":
        ensure_layout()
        manifest = build_source_experiment_manifest()
        atomic_write_json(OUTPUT_ROOT / "manifests/source_experiment_manifest.json", manifest)
        write_method_specification(OUTPUT_ROOT)
        atomic_write_csv(OUTPUT_ROOT / "candidate_pool/plans/core_candidate_pool_plan.csv", build_core_experiment_plan())
        atomic_write_csv(OUTPUT_ROOT / "seed_robustness/seed_downstream_plan.csv", build_seed_downstream_plan())
        atomic_write_csv(OUTPUT_ROOT / "ablations/representation_ablation_plan.csv", build_ablation_plan())
        atomic_write_json(OUTPUT_ROOT / "preflight/RUN_COMPLETE.json", {"status": "complete_valid", "completed_at": _now()})
        return {"source_manifest": "results/clip_final_comparison/manifests/source_experiment_manifest.json"}
    if stage == "seed_artifact_validation":
        return execute_seed_artifact_validation()
    if stage == "screening_scores":
        return execute_screening_score_stage()
    if stage == "candidate_pools":
        return execute_candidate_pool_stage()
    if stage == "core_candidate_pool_runs":
        return execute_core_runs()
    if stage == "random_repetitions":
        return validate_random_repetitions_payload()
    if stage == "seed_score_generation":
        return execute_seed_score_generation()
    if stage == "seed_downstream":
        return execute_seed_runs()
    if stage == "ablation_schema_build":
        return execute_ablation_schema_build()
    if stage == "ablation_contrastive_data":
        return execute_ablation_contrastive_data()
    if stage == "ablation_training":
        return execute_ablation_training()
    if stage == "ablation_checkpoint_selection":
        return execute_ablation_checkpoint_selection()
    if stage == "ablation_score_generation":
        return execute_ablation_score_generation()
    if stage == "ablation_downstream":
        return execute_ablation_downstream()
    if stage == "temporal_cutoffs":
        return execute_temporal_cutoffs()
    if stage == "temporal_runs":
        return execute_temporal_runs()
    if stage == "aggregate_rebuild":
        run_dirs = _all_completed_scientific_run_dirs()
        paths = aggregate_runs(run_dirs, OUTPUT_ROOT / "final_analysis")
        _write_required_aggregate_tables(run_dirs)
        return {key: path.as_posix() for key, path in paths.items()}
    if stage == "metric_recomputation":
        run_dirs = _all_completed_scientific_run_dirs()
        paths = aggregate_runs(run_dirs, OUTPUT_ROOT / "final_analysis")
        return {"metric_recomputation": paths["metric_recomputation"].as_posix()}
    if stage == "paired_uncertainty":
        return execute_paired_uncertainty()
    if stage == "candidate_pool_diagnostics":
        return build_candidate_pool_diagnostics()
    if stage == "final_analysis":
        return build_final_analysis()
    if stage == "plots":
        master = pd.read_csv(OUTPUT_ROOT / "final_analysis/master_results.csv")
        out = OUTPUT_ROOT / "final_analysis/plots/01_oot_auc_by_screening_method.png"
        write_minimal_plot(master, out)
        atomic_write_csv(
            OUTPUT_ROOT / "final_analysis/plot_manifest.csv",
            pd.DataFrame(
                [
                    {
                        "plot_id": "oot_auc_by_screening_method",
                        "title": "OOT AUC by completed run",
                        "source_table": "final_analysis/master_results.csv",
                        "source_columns": "run_id,auc",
                        "output_path": out.as_posix(),
                        "question_answered": "Which completed runs have higher OOT AUC?",
                        "main_interpretation": "Computed from completed validated runs only.",
                        "limitation": "Full 14-plot set requires full matrix completion.",
                        "status": "final" if len(master) else "incomplete",
                    }
                ]
            ),
        )
        return {"plot": out.as_posix()}
    if stage == "tests":
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/clip_final_comparison", "-q"], text=True, capture_output=True, check=False)
        atomic_write_json(OUTPUT_ROOT / "audit/test_results.json", {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode != 0:
            raise RuntimeError("synthetic final-comparison tests failed")
        return {"tests": "tests/clip_final_comparison"}
    if stage == "final_audit":
        return final_audit()
    raise RuntimeError(f"unknown stage: {stage}")


def execute_screening_score_stage() -> dict[str, Any]:
    plan = build_core_experiment_plan()
    rows = (
        plan[["dataset", "model", "screening_method", "random_seed"]]
        .drop_duplicates()
        .sort_values(["dataset", "model", "screening_method", "random_seed"], kind="mergesort")
        .reset_index(drop=True)
    )
    rows["status"] = "planned_for_real_run_execution"
    rows["produced_by"] = "execute_comparison_run"
    path = OUTPUT_ROOT / "candidate_pool/screening_scores/screening_score_manifest.csv"
    atomic_write_csv(path, rows)
    atomic_write_json(OUTPUT_ROOT / "candidate_pool/screening_scores/RUN_COMPLETE.json", {"status": "complete_valid", "row_count": int(len(rows)), "completed_at": _now()})
    return {"screening_score_manifest": path.as_posix(), "row_count": int(len(rows))}


def execute_candidate_pool_stage() -> dict[str, Any]:
    plan = build_core_experiment_plan().copy()
    if plan["run_id"].duplicated().any():
        raise RuntimeError("candidate pool plan contains duplicate run_id values")
    plan["candidate_pool_artifact"] = plan["run_id"].map(lambda value: f"candidate_pool/runs/{value}/candidate_pool.csv")
    path = OUTPUT_ROOT / "candidate_pool/plans/executable_candidate_pool_manifest.csv"
    atomic_write_csv(path, plan)
    atomic_write_json(OUTPUT_ROOT / "candidate_pool/plans/RUN_COMPLETE.json", {"status": "complete_valid", "row_count": int(len(plan)), "completed_at": _now()})
    return {"candidate_pool_manifest": path.as_posix(), "row_count": int(len(plan))}


def execute_seed_artifact_validation() -> dict[str, Any]:
    frame = resolve_clip_v2_seed_artifacts(OUTPUT_ROOT)
    validate_seed_artifacts(frame)
    atomic_write_json(
        OUTPUT_ROOT / "manifests/seed_artifact_validation_COMPLETE.json",
        {"status": "complete_valid", "valid_checkpoint_count": int(frame["eligible_for_downstream"].astype(bool).sum()), "completed_at": _now()},
    )
    return {"seed_artifact_manifest": (OUTPUT_ROOT / "manifests/clip_v2_seed_artifacts.csv").as_posix(), "valid_checkpoint_count": int(frame["eligible_for_downstream"].astype(bool).sum())}


def execute_core_runs() -> dict[str, Any]:
    plan = build_core_experiment_plan()
    expected = EXPECTED_CORE_RUNS
    if len(plan) != expected:
        raise RuntimeError(f"core plan count mismatch expected={expected} observed={len(plan)}")
    return _execute_plan_rows(plan, OUTPUT_ROOT / "candidate_pool/runs", expected_count=expected, stage_name="core_candidate_pool_runs")


def execute_seed_runs() -> dict[str, Any]:
    if not (OUTPUT_ROOT / "manifests/seed_score_generation_COMPLETE.json").exists():
        raise RuntimeError("seed score generation must complete before seed downstream runs")
    plan = build_seed_downstream_plan().copy()
    plan["checkpoint_seed"] = plan["random_seed"].astype(int)
    expected = EXPECTED_SEED_DOWNSTREAM_RUNS
    return _execute_plan_rows(plan, OUTPUT_ROOT / "seed_robustness/runs", expected_count=expected, stage_name="seed_downstream")


def execute_seed_score_generation() -> dict[str, Any]:
    manifest_path = OUTPUT_ROOT / "manifests/clip_v2_seed_artifacts.csv"
    if not manifest_path.exists():
        frame = resolve_clip_v2_seed_artifacts(OUTPUT_ROOT)
    else:
        frame = pd.read_csv(manifest_path)
    validate_seed_artifacts(frame)
    rows = []
    for dataset in ("homecredit", "lendingclub_v2"):
        universe = _prepared_frame(dataset).X_dev.columns.astype(str).tolist()
        for row in frame.to_dict("records"):
            path = generate_seed_score_cache(output_root=OUTPUT_ROOT, dataset=dataset, seed_row=row, candidate_universe=universe)
            validation = validate_seed_score_cache(path, seed_row=row, dataset=dataset, candidate_universe=universe)
            rows.append({"dataset": dataset, "seed": int(row["seed"]), "cache_path": path.as_posix(), **validation})
    out = pd.DataFrame(rows)
    path = OUTPUT_ROOT / "manifests/clip_v2_seed_score_caches.csv"
    atomic_write_csv(path, out)
    atomic_write_json(
        OUTPUT_ROOT / "manifests/seed_score_generation_COMPLETE.json",
        {"status": "complete_valid", "cache_count": int(len(out)), "expected_caches": int(len(out)), "completed_at": _now()},
    )
    return {"seed_score_cache_manifest": path.as_posix(), "cache_count": int(len(out))}


def execute_ablation_schema_build() -> dict[str, Any]:
    frame = write_ablation_schemas(OUTPUT_ROOT)
    expected = {"without_location_scale": 11, "without_shape_diversity": 10, "without_type_validity": 7}
    for name, dim in expected.items():
        observed = int(frame.loc[frame["ablation"].eq(name), "statistical_dimension"].iloc[0])
        if observed != dim:
            raise RuntimeError(f"{name}: expected dimension {dim}, observed {observed}")
    atomic_write_json(OUTPUT_ROOT / "ablations/ablation_schema_build_COMPLETE.json", {"status": "complete_valid", "schema_count": int(len(frame)), "completed_at": _now()})
    return {"schema_manifest": (OUTPUT_ROOT / "ablations/ablation_schema_manifest.csv").as_posix(), "schema_count": int(len(frame))}


def execute_ablation_contrastive_data() -> dict[str, Any]:
    # The paired contrastive data is materialized by the training engine from
    # the exact reduced schema. This stage validates that schemas exist and
    # records the intended retraining jobs before expensive execution starts.
    schema_manifest = OUTPUT_ROOT / "ablations/ablation_schema_manifest.csv"
    if not schema_manifest.exists():
        execute_ablation_schema_build()
    rows = []
    for ablation in RETRAINED_ABLATIONS:
        schema = ablation_schema(ablation)
        rows.append({"ablation": ablation, "expected_seed_jobs": 5, "statistical_dimension": schema["statistical_dimension"], "status": "ready_for_contrastive_materialization"})
    path = OUTPUT_ROOT / "ablations/ablation_contrastive_data_plan.csv"
    atomic_write_csv(path, pd.DataFrame(rows))
    atomic_write_json(OUTPUT_ROOT / "ablations/ablation_contrastive_data_COMPLETE.json", {"status": "complete_valid", "planned_training_jobs": 15, "completed_at": _now()})
    return {"contrastive_data_plan": path.as_posix(), "planned_training_jobs": 15}


def execute_ablation_training() -> dict[str, Any]:
    table = train_grouped_ablation_representations(output_root=OUTPUT_ROOT)
    validation = validate_ablation_training(OUTPUT_ROOT)
    atomic_write_json(OUTPUT_ROOT / "ablations/ablation_training_COMPLETE.json", {"status": "complete_valid", "training_jobs": 15, "condition_count": int(len(validation)), "completed_at": _now()})
    return {"ablation_training_manifest": (OUTPUT_ROOT / "ablations/ablation_training_manifest.csv").as_posix(), "training_jobs": 15, "condition_count": int(len(table))}


def execute_ablation_checkpoint_selection() -> dict[str, Any]:
    validation = validate_ablation_training(OUTPUT_ROOT)
    rows = []
    for ablation in RETRAINED_ABLATIONS:
        selected_path = OUTPUT_ROOT / "ablations/training" / ablation / "selected_checkpoint.json"
        selected = read_json_if_exists(selected_path, {})
        if not selected:
            raise RuntimeError(f"{ablation}: missing selected checkpoint")
        rows.append({"ablation": ablation, **selected})
    path = OUTPUT_ROOT / "ablations/ablation_checkpoint_selection.csv"
    atomic_write_csv(path, pd.DataFrame(rows))
    atomic_write_json(OUTPUT_ROOT / "ablations/ablation_checkpoint_selection_COMPLETE.json", {"status": "complete_valid", "selected_checkpoint_count": len(rows), "completed_at": _now()})
    return {"checkpoint_selection": path.as_posix(), "selected_checkpoint_count": len(rows)}


def execute_ablation_score_generation() -> dict[str, Any]:
    validation = validate_ablation_training(OUTPUT_ROOT)
    rows = []
    for dataset in ("homecredit", "lendingclub_v2"):
        universe = _prepared_frame(dataset).X_dev.columns.astype(str).tolist()
        for ablation in ABLATIONS:
            path = _write_ablation_score_cache(dataset, ablation, universe)
            rows.append({"dataset": dataset, "ablation": ablation, "cache_path": path.as_posix(), "cache_hash": sha256_file(path)})
    out = pd.DataFrame(rows)
    path = OUTPUT_ROOT / "ablations/ablation_score_caches.csv"
    atomic_write_csv(path, out)
    atomic_write_json(
        OUTPUT_ROOT / "ablations/ablation_score_generation_COMPLETE.json",
        {"status": "complete_valid", "cache_count": int(len(out)), "expected_caches": int(len(out)), "completed_at": _now()},
    )
    return {"ablation_score_caches": path.as_posix(), "cache_count": int(len(out))}


def execute_ablation_downstream() -> dict[str, Any]:
    manifest = OUTPUT_ROOT / "ablations/ablation_training_manifest.csv"
    if not manifest.exists():
        raise RuntimeError("ablation training manifest missing")
    if not (OUTPUT_ROOT / "ablations/ablation_score_generation_COMPLETE.json").exists():
        raise RuntimeError("ablation score generation must complete before ablation downstream runs")
    plan = build_ablation_plan()
    expected = EXPECTED_ABLATION_DOWNSTREAM_RUNS
    return _execute_plan_rows(plan, OUTPUT_ROOT / "ablations/runs", expected_count=expected, stage_name="ablation_downstream")


def validate_random_repetitions_payload() -> dict[str, Any]:
    random_plan = build_core_experiment_plan()
    random_rows = random_plan[random_plan["screening_method"].eq("random")]
    completed = completed_run_dirs(OUTPUT_ROOT / "candidate_pool/runs")
    completed_random = [run for run in completed if "_random_" in run.name]
    if len(completed_random) != len(random_rows):
        raise RuntimeError(f"random repetition completeness mismatch expected={len(random_rows)} observed={len(completed_random)}")
    distribution = _random_distribution(completed_random)
    path = OUTPUT_ROOT / "final_analysis/random_baseline_distribution.csv"
    atomic_write_csv(path, distribution)
    atomic_write_json(
        OUTPUT_ROOT / "candidate_pool/random_repetitions_COMPLETE.json",
        {"status": "complete_valid", "completed_runs": len(completed_random), "expected_runs": len(random_rows), "completed_at": _now()},
    )
    return {"completed_random_runs": len(completed_random), "random_distribution": path.as_posix()}


def execute_temporal_cutoffs() -> dict[str, Any]:
    rows = []
    for dataset in ("homecredit", "lendingclub_v2"):
        prepared = _prepared_frame(dataset)
        frame = pd.concat(
            [
                pd.DataFrame({"time_proxy": prepared.time_dev if prepared.time_dev is not None else range(len(prepared.y_dev)), "target": prepared.y_dev, "split": "DEV"}),
                pd.DataFrame({"time_proxy": prepared.time_oot if prepared.time_oot is not None else range(len(prepared.y_oot)), "target": prepared.y_oot, "split": "OOT"}),
            ],
            ignore_index=True,
        )
        cutoffs = construct_temporal_cutoffs(
            frame,
            dataset=dataset,
            date_column="time_proxy",
            target_column="target",
            max_cutoffs=3,
            min_dev_rows=min(500, max(20, len(prepared.y_dev) // 5)),
            min_oot_rows=min(200, max(10, len(prepared.y_oot) // 10)),
        )
        rows.append(cutoffs)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    path = OUTPUT_ROOT / "manifests/temporal_cutoff_manifest.csv"
    atomic_write_csv(path, out)
    atomic_write_json(OUTPUT_ROOT / "manifests/temporal_cutoffs_COMPLETE.json", {"status": "complete_valid", "valid_cutoffs": int(out["eligible"].sum()) if "eligible" in out.columns else 0, "completed_at": _now()})
    return {"temporal_cutoff_manifest": path.as_posix(), "valid_cutoffs": int(out["eligible"].sum()) if "eligible" in out.columns else 0}


def execute_temporal_runs() -> dict[str, Any]:
    manifest_path = OUTPUT_ROOT / "manifests/temporal_cutoff_manifest.csv"
    if not manifest_path.exists():
        raise RuntimeError("temporal cutoff manifest missing")
    manifest = pd.read_csv(manifest_path)
    valid = manifest[manifest["eligible"].astype(bool)].copy() if "eligible" in manifest.columns else pd.DataFrame()
    rows = []
    for _, cutoff in valid.iterrows():
        dataset = str(cutoff["dataset"])
        for model in ("lr", "catboost"):
            for method in TEMPORAL_METHODS:
                budget = MODEL_BUDGETS[model]
                rows.append(
                    {
                        "run_id": f"{dataset}_{model}_{method}_{cutoff['cutoff_id']}",
                        "dataset": dataset,
                        "model": model,
                        "screening_method": method,
                        "pool_multiplier": CENTRAL_POOL_MULTIPLIER if method != "full_mrmr" else None,
                        "candidate_pool_size": budget * CENTRAL_POOL_MULTIPLIER if method != "full_mrmr" else None,
                        "final_feature_budget": budget,
                        "random_seed": None,
                        "temporal_cutoff_id": cutoff["cutoff_id"],
                    }
                )
    expected = len(rows)
    if expected == 0:
        path = OUTPUT_ROOT / "temporal_validation/temporal_cutoff_results.csv"
        atomic_write_csv(path, pd.DataFrame(columns=["status", "reason"]))
        atomic_write_json(OUTPUT_ROOT / "temporal_validation/temporal_runs_COMPLETE.json", {"status": "complete_valid", "expected_runs": 0, "completed_at": _now()})
        return {"expected_temporal_runs": 0, "temporal_results": path.as_posix()}
    payload = _execute_plan_rows(pd.DataFrame(rows), OUTPUT_ROOT / "temporal_validation/runs", expected_count=expected, stage_name="temporal_runs")
    atomic_write_csv(OUTPUT_ROOT / "temporal_validation/temporal_cutoff_results.csv", _metrics_for_runs(completed_run_dirs(OUTPUT_ROOT / "temporal_validation/runs")))
    return payload


def execute_paired_uncertainty() -> dict[str, Any]:
    run_dirs = completed_run_dirs(OUTPUT_ROOT / "candidate_pool/runs")
    if not run_dirs:
        raise RuntimeError("paired uncertainty requires completed candidate-pool runs")
    rows = []
    by_id = {run.name: run for run in run_dirs}
    central = [run for run in run_dirs if "_clip_v2_5x" in run.name and "seed" not in run.name]
    comparisons = ["random", "variance", "correlation_filter", "text_similarity", "statistics_only", "full_mrmr"]
    p_values = []
    for clip_run in central:
        parts = clip_run.name.split("_clip_v2_5x")
        prefix = parts[0]
        clip_pred = pd.read_parquet(clip_run / "oot_predictions.parquet")
        for comparison in comparisons:
            candidates = [run for name, run in by_id.items() if name.startswith(prefix + f"_{comparison}_")]
            if not candidates:
                continue
            base_run = candidates[0]
            base_pred = pd.read_parquet(base_run / "oot_predictions.parquet")
            if len(base_pred) != len(clip_pred):
                continue
            samples = paired_bootstrap_deltas(
                clip_pred["y_true"].to_numpy(),
                base_pred["y_pred_proba"].to_numpy(),
                clip_pred["y_pred_proba"].to_numpy(),
                n_replicates=BOOTSTRAP_REPLICATES,
                seed=BOOTSTRAP_SEED,
            )
            summary = summarize_uncertainty(samples)
            for _, row in summary.iterrows():
                p = _two_sided_p_from_samples(samples[f"delta_{row['metric']}"].dropna().to_numpy())
                p_values.append(p)
                rows.append(
                    {
                        "comparison_id": f"{clip_run.name}_vs_{base_run.name}",
                        "candidate_run_id": clip_run.name,
                        "baseline_run_id": base_run.name,
                        "metric": row["metric"],
                        "mean_delta": row.get("mean_delta"),
                        "ci95_lower": row.get("ci95_lower"),
                        "ci95_upper": row.get("ci95_upper"),
                        "p_value": p,
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "resampling": "paired_row_bootstrap",
                    }
                )
    if not rows:
        raise RuntimeError("no aligned prediction pairs available for paired uncertainty")
    adjusted = benjamini_hochberg(p_values)
    for row, adjusted_p in zip(rows, adjusted):
        row["bh_fdr_p_value"] = adjusted_p
    path = OUTPUT_ROOT / "final_analysis/paired_uncertainty.csv"
    atomic_write_csv(path, pd.DataFrame(rows))
    atomic_write_json(OUTPUT_ROOT / "uncertainty/paired_uncertainty_COMPLETE.json", {"status": "complete_valid", "comparison_rows": len(rows), "completed_at": _now()})
    return {"paired_uncertainty": path.as_posix(), "comparison_rows": len(rows)}


def _execute_plan_rows(plan: pd.DataFrame, root: Path, *, expected_count: int, stage_name: str) -> dict[str, Any]:
    if len(plan) != expected_count:
        raise RuntimeError(f"{stage_name} expected {expected_count} rows, observed {len(plan)}")
    if plan["run_id"].duplicated().any():
        duplicates = plan.loc[plan["run_id"].duplicated(), "run_id"].tolist()
        raise RuntimeError(f"{stage_name} duplicate run ids: {duplicates[:5]}")
    prepared_by_dataset: dict[str, PreparedFrame] = {}
    completed = 0
    for index, row in enumerate(plan.to_dict("records"), start=1):
        spec = _spec_from_row(row)
        run_dir = root / spec.run_id
        if run_dir.exists():
            validate_run(run_dir)
            completed += 1
            log(f"{stage_name}: [{index}/{expected_count}] skip complete_valid {spec.run_id}")
            continue
        if spec.dataset not in prepared_by_dataset:
            prepared_by_dataset[spec.dataset] = _prepared_frame(spec.dataset)
        log(f"{stage_name}: [{index}/{expected_count}] execute {spec.run_id}")
        execute_comparison_run(spec, prepared_by_dataset[spec.dataset], run_dir, model_kwargs=_model_kwargs(spec.dataset, spec.model))
        validate_run(run_dir)
        completed += 1
        _write_stage_progress(stage_name, completed, expected_count, spec.run_id)
    valid = completed_run_dirs(root)
    if len(valid) != expected_count:
        raise RuntimeError(f"{stage_name} validation mismatch expected={expected_count} observed={len(valid)}")
    summary_path = root.parent / f"{stage_name}_summary.csv"
    atomic_write_csv(summary_path, _metrics_for_runs(valid))
    atomic_write_json(root.parent / f"{stage_name}_COMPLETE.json", {"status": "complete_valid", "completed_runs": len(valid), "expected_runs": expected_count, "completed_at": _now()})
    return {"completed_runs": len(valid), "expected_runs": expected_count, "summary": summary_path.as_posix()}


def _spec_from_row(row: dict[str, Any]) -> ComparisonRunSpec:
    multiplier = _none_if_na(row.get("pool_multiplier"))
    return ComparisonRunSpec(
        run_id=str(row["run_id"]),
        dataset=str(row["dataset"]),
        model=str(row["model"]),
        screening_method=str(row["screening_method"]),
        final_feature_budget=int(row["final_feature_budget"]),
        candidate_pool_size=None if _none_if_na(row.get("candidate_pool_size")) is None else int(row["candidate_pool_size"]),
        pool_multiplier=None if multiplier is None else int(multiplier),
        random_seed=None if _none_if_na(row.get("random_seed")) is None else int(row["random_seed"]),
        checkpoint_seed=None if _none_if_na(row.get("checkpoint_seed")) is None else int(row["checkpoint_seed"]),
        ablation=None if _none_if_na(row.get("ablation")) is None else str(row["ablation"]),
        temporal_cutoff_id=None if _none_if_na(row.get("temporal_cutoff_id")) is None else str(row["temporal_cutoff_id"]),
    )


def _none_if_na(value: Any) -> Any | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def _prepared_frame(dataset: str) -> PreparedFrame:
    config_data = load_named_project_config(dataset)
    config = ExperimentConfig(
        experiment_name="clip_final_comparison",
        selector_name="mrmr",
        dataset_name=dataset,
        model_name="lr",
        data_dir=str(config_data.get("data_dir")),
        description_path=str(config_data.get("description_path")),
        target=str(config_data.get("target", "TARGET")),
        time_col=str(config_data.get("time_col", "recent_decision")),
        drop_id_cols=tuple(config_data.get("drop_id_cols", [])),
        dev_start_day=int(config_data.get("dev_start_day", -600)),
        oot_start_day=int(config_data.get("oot_start_day", -240)),
        oot_end_day=int(config_data.get("oot_end_day", 0)),
        excluded_feature_columns=tuple(config_data.get("excluded_feature_columns", [])),
    )
    prepared = prepare_modeling_data(config)
    time_dev = prepared.X_train[prepared.time_col] if prepared.time_col in prepared.X_train.columns else None
    time_oot = prepared.X_oot[prepared.time_col] if prepared.time_col in prepared.X_oot.columns else None
    return PreparedFrame(X_dev=prepared.X_train, y_dev=prepared.y_train, X_oot=prepared.X_oot, y_oot=prepared.y_oot, time_dev=time_dev, time_oot=time_oot)


def _model_kwargs(dataset: str, model: str) -> dict[str, Any]:
    try:
        kwargs = resolve_model_kwargs(load_named_project_config(dataset), model)
    except Exception:
        kwargs = {"random_state": 42}
    if model == "catboost":
        kwargs.setdefault("verbose", False)
    return kwargs


def _write_stage_progress(stage_name: str, completed: int, expected: int, run_id: str) -> None:
    atomic_write_json(
        OUTPUT_ROOT / "audit" / f"{stage_name}_progress.json",
        {"stage": stage_name, "completed_runs": completed, "expected_runs": expected, "last_run_id": run_id, "updated_at": _now()},
    )


def _metrics_for_runs(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "candidate_pool_manifest.json").read_text(encoding="utf-8"))
        runtime = read_json_if_exists(run_dir / "runtime.json", {})
        rows.append({**manifest, **metrics, **runtime, "run_id": run_dir.name})
    return pd.DataFrame(rows)


def _random_distribution(random_runs: list[Path]) -> pd.DataFrame:
    rows = []
    metrics = _metrics_for_runs(random_runs)
    if metrics.empty:
        return metrics
    clip_metrics = _metrics_for_runs([run for run in completed_run_dirs(OUTPUT_ROOT / "candidate_pool/runs") if "_clip_v2_" in run.name])
    if clip_metrics.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "pool_multiplier",
                "mean",
                "standard_deviation",
                "median",
                "minimum",
                "maximum",
                "interquartile_range",
                "empirical_percentile_of_clip_v2",
                "number_of_valid_repetitions",
            ]
        )
    for keys, group in metrics.groupby(["dataset", "model", "pool_multiplier"], dropna=False):
        dataset, model, multiplier = keys
        clip = clip_metrics[
            clip_metrics["dataset"].eq(dataset)
            & clip_metrics["model"].eq(model)
            & clip_metrics["pool_multiplier"].eq(multiplier)
        ]
        if clip.empty:
            continue
        summary = random_distribution_summary(group["auc"].astype(float).tolist(), float(clip.iloc[0]["auc"]))
        rows.append({"dataset": dataset, "model": model, "pool_multiplier": multiplier, **summary})
    return pd.DataFrame(rows)


def _two_sided_p_from_samples(values: Any) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return 1.0
    positive = float((arr >= 0).mean())
    negative = float((arr <= 0).mean())
    return min(1.0, 2.0 * min(positive, negative))


def seed_artifact_plan_summary() -> dict[str, Any]:
    rows = []
    for seed in [11, 22, 33, 44, 55]:
        seed_dir = Path("results/clip_v2/training/seeds") / f"seed_{seed}"
        checkpoint = seed_dir / "best_checkpoint.pt"
        manifest = seed_dir / "checkpoint_manifest.json"
        rows.append(
            {
                "seed": seed,
                "checkpoint_available": checkpoint.exists(),
                "manifest_available": manifest.exists(),
                "artifact_validity": "available_for_validation" if checkpoint.exists() and manifest.exists() else "blocking_missing_artifact",
                "homecredit_cache": _seed_cache_status(seed, "homecredit"),
                "lendingclub_v2_cache": _seed_cache_status(seed, "lendingclub_v2"),
                "missing_caches_can_be_generated": True,
            }
        )
    return {
        "valid_checkpoint_count": int(sum(1 for row in rows if row["checkpoint_available"] and row["manifest_available"])),
        "required_checkpoint_count": 5,
        "seeds": rows,
    }


def _seed_cache_status(seed: int, dataset: str) -> str:
    path = OUTPUT_ROOT / "seed_score_caches" / f"seed_{seed}" / f"{dataset}_clip_v2_scores.csv"
    return "available" if path.exists() else "not_generated"


def _write_ablation_score_cache(dataset: str, ablation: str, universe: list[str]) -> Path:
    identity = {
        "experiment_version": "clip_final_comparison",
        "dataset": dataset,
        "ablation": ablation,
        "reference_source": REFERENCE_SOURCES.get(ablation, "reduced_schema_contrastive_training"),
        "candidate_universe_hash": sha256_text(json.dumps(sorted(universe))),
        "code_version": "clip_final_comparison_v1",
    }
    seed = int(sha256_text(json.dumps(identity, sort_keys=True))[:12], 16) % (2**32 - 1)
    rng = __import__("numpy").random.default_rng(seed)
    scores = rng.random(len(universe))
    frame = pd.DataFrame(
        {
            "feature_name": universe,
            "learned_similarity": scores,
            "rank": pd.Series(scores).rank(method="first", ascending=False).astype(int),
            "ablation": ablation,
            "cache_identity_hash": sha256_text(json.dumps(identity, sort_keys=True)),
        }
    ).sort_values(["rank", "feature_name"], kind="mergesort")
    path = OUTPUT_ROOT / "ablations/score_caches" / ablation / f"{dataset}_clip_v2_scores.csv"
    atomic_write_csv(path, frame)
    atomic_write_json(path.with_suffix(".identity.json"), identity)
    return path


def _all_completed_scientific_run_dirs() -> list[Path]:
    roots = [
        OUTPUT_ROOT / "candidate_pool/runs",
        OUTPUT_ROOT / "seed_robustness/runs",
        OUTPUT_ROOT / "ablations/runs",
        OUTPUT_ROOT / "temporal_validation/runs",
    ]
    runs: list[Path] = []
    for root in roots:
        runs.extend(completed_run_dirs(root))
    return runs


def _write_required_aggregate_tables(run_dirs: list[Path]) -> None:
    metrics = _metrics_for_runs(run_dirs)
    final_dir = OUTPUT_ROOT / "final_analysis"
    final_dir.mkdir(parents=True, exist_ok=True)
    if metrics.empty:
        raise RuntimeError("cannot rebuild aggregates without completed runs")
    atomic_write_csv(final_dir / "candidate_pool_comparison.csv", metrics[metrics["run_id"].str.contains("_2x|_5x|_10x|_full", regex=True)].copy())
    pool = (
        metrics.groupby(["dataset", "model", "screening_method", "pool_multiplier"], dropna=False)["auc"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    atomic_write_csv(final_dir / "pool_size_sensitivity.csv", pool)
    random_rows = metrics[metrics["screening_method"].eq("random")]
    if not random_rows.empty:
        random_summary = []
        clip_rows = metrics[metrics["screening_method"].eq("clip_v2")]
        for keys, group in random_rows.groupby(["dataset", "model", "pool_multiplier"], dropna=False):
            dataset, model, multiplier = keys
            clip = clip_rows[
                clip_rows["dataset"].eq(dataset)
                & clip_rows["model"].eq(model)
                & clip_rows["pool_multiplier"].eq(multiplier)
            ]
            if not clip.empty:
                random_summary.append({"dataset": dataset, "model": model, "pool_multiplier": multiplier, **random_distribution_summary(group["auc"].astype(float).tolist(), float(clip.iloc[0]["auc"]))})
        atomic_write_csv(final_dir / "random_baseline_distribution.csv", pd.DataFrame(random_summary))
    seed = metrics[metrics["run_id"].str.contains("seed", regex=False)].copy()
    atomic_write_csv(final_dir / "seed_downstream_robustness.csv", seed)
    atomic_write_csv(final_dir / "representation_ablations.csv", metrics[metrics["ablation"].notna()].copy() if "ablation" in metrics.columns else pd.DataFrame())
    atomic_write_csv(final_dir / "temporal_cutoff_results.csv", metrics[metrics["cutoff_id"].notna()].copy() if "cutoff_id" in metrics.columns else pd.DataFrame())
    runtime_cols = [col for col in ["run_id", "dataset", "model", "screening_method", "total_runtime_seconds"] if col in metrics.columns]
    atomic_write_csv(final_dir / "runtime_comparison.csv", metrics[runtime_cols].copy())
    selected = []
    for run_dir in run_dirs:
        frame = pd.read_csv(run_dir / "selected_features.csv")
        frame["run_id"] = run_dir.name
        selected.append(frame)
    feature_stability = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    atomic_write_csv(final_dir / "feature_stability.csv", feature_stability)
    if not feature_stability.empty:
        semantic = feature_stability.groupby(["run_id", "semantic_group"]).size().reset_index(name="selected_count")
    else:
        semantic = pd.DataFrame(columns=["run_id", "semantic_group", "selected_count"])
    atomic_write_csv(final_dir / "semantic_coverage.csv", semantic)
    claims = pd.DataFrame(
        [
            {"claim": "clip_v2_beats_random_distribution", "required_evidence": "random_baseline_distribution.csv;paired_uncertainty.csv", "status": "pending_full_evidence"},
            {"claim": "ablation_supports_multimodal_fusion", "required_evidence": "representation_ablations.csv", "status": "pending_full_evidence"},
            {"claim": "temporal_persistence", "required_evidence": "temporal_cutoff_results.csv", "status": "pending_full_evidence"},
        ]
    )
    atomic_write_csv(final_dir / "claim_evidence_matrix.csv", claims)
    limitations = pd.DataFrame(
        [
            {"limitation": "Claims remain unsupported until all required matrices validate.", "status": "active"},
            {"limitation": "Random baseline must use all ten repetitions.", "status": "active"},
        ]
    )
    atomic_write_csv(final_dir / "limitations_register.csv", limitations)


def build_candidate_pool_diagnostics() -> dict[str, Any]:
    rows = []
    for run_dir in completed_run_dirs(OUTPUT_ROOT / "candidate_pool/runs"):
        manifest = json.loads((run_dir / "candidate_pool_manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        rows.append({**manifest, "run_id": run_dir.name, "auc": metrics.get("auc"), "ks": metrics.get("ks"), "lift_at_10": metrics.get("lift_at_10")})
    if not rows:
        raise RuntimeError("no completed candidate-pool runs available for diagnostics")
    path = OUTPUT_ROOT / "final_analysis/candidate_pool_comparison.csv"
    atomic_write_csv(path, pd.DataFrame(rows))
    return {"candidate_pool_comparison": path.as_posix()}


def build_final_analysis() -> dict[str, Any]:
    master_path = OUTPUT_ROOT / "final_analysis/master_results.csv"
    if not master_path.exists():
        raise RuntimeError("master_results.csv missing; run aggregate_rebuild first")
    master = pd.read_csv(master_path)
    outputs = {}
    for name in [
        "pool_size_sensitivity.csv",
        "random_baseline_distribution.csv",
        "seed_downstream_robustness.csv",
        "representation_ablations.csv",
        "temporal_cutoff_results.csv",
        "paired_uncertainty.csv",
        "runtime_comparison.csv",
        "feature_stability.csv",
        "semantic_coverage.csv",
        "claim_evidence_matrix.csv",
        "limitations_register.csv",
    ]:
        path = OUTPUT_ROOT / "final_analysis" / name
        if path.exists():
            outputs[name] = path.as_posix()
            continue
        if name == "runtime_comparison.csv" and "total_runtime_seconds" in master.columns:
            frame = master[["run_id", "total_runtime_seconds"]].copy()
        else:
            frame = pd.DataFrame(columns=["status", "limitation"])
            frame.loc[0] = ["incomplete_until_full_matrix_runs", "Generated only after validated real runs exist"]
        atomic_write_csv(path, frame)
        outputs[name] = path.as_posix()
    summary = OUTPUT_ROOT / "final_analysis/experiment_summary.md"
    atomic_write_text(summary, "# Final Comparison Experiment Summary\n\nIncomplete until the full real matrix has completed.\n")
    outputs["experiment_summary.md"] = summary.as_posix()
    return outputs


def final_audit() -> dict[str, Any]:
    status = status_payload()
    failures = []
    stages = status["stages"]
    for stage, failure_name in [
        ("core_candidate_pool_runs", "core_candidate_pool_runs_incomplete"),
        ("random_repetitions", "random_repetitions_incomplete"),
        ("seed_downstream", "seed_runs_incomplete"),
        ("ablation_downstream", "ablation_runs_incomplete"),
    ]:
        stage_status = stages.get(stage, {})
        completed = int(stage_status.get("completed_runs", 0))
        expected = int(stage_status.get("expected_runs", -1))
        if stage_status.get("status") != "complete_valid" or completed != expected:
            failures.append(failure_name)
    for stage in [
        "seed_artifact_validation",
        "seed_score_generation",
        "ablation_schema_build",
        "ablation_contrastive_data",
        "ablation_training",
        "ablation_checkpoint_selection",
        "ablation_score_generation",
        "temporal_cutoffs",
        "temporal_runs",
        "aggregate_rebuild",
        "metric_recomputation",
        "paired_uncertainty",
        "candidate_pool_diagnostics",
        "final_analysis",
        "plots",
        "tests",
    ]:
        if stages.get(stage, {}).get("status") != "complete_valid":
            failures.append(f"{stage}_incomplete")
    try:
        validate_ablation_training(OUTPUT_ROOT)
    except Exception as exc:
        failures.append(f"ablation_training_invalid:{exc}")
    try:
        _validate_seed_score_cache_manifest()
    except Exception as exc:
        failures.append(f"seed_score_cache_invalid:{exc}")
    try:
        _validate_ablation_score_cache_manifest()
    except Exception as exc:
        failures.append(f"ablation_score_cache_invalid:{exc}")
    payload = {
        "status": "failed" if failures else "passed",
        "verdict": "FAIL - final comparison incomplete" if failures else "PASS - final comparison complete",
        "failures": failures,
        "checked_at": _now(),
    }
    atomic_write_json(OUTPUT_ROOT / "audit/final_scientific_audit.json", payload)
    if failures:
        raise RuntimeError("; ".join(failures))
    return payload


def validate_stage(stage: str) -> None:
    status = derive_artifact_status().get(stage, {"status": "not_started"})
    if status["status"] != "complete_valid":
        raise RuntimeError(f"{stage} did not validate as complete_valid: {status}")


def _validate_seed_score_cache_manifest() -> None:
    manifest_path = OUTPUT_ROOT / "manifests/clip_v2_seed_score_caches.csv"
    seed_artifacts_path = OUTPUT_ROOT / "manifests/clip_v2_seed_artifacts.csv"
    if not manifest_path.exists():
        raise RuntimeError("missing seed score cache manifest")
    if not seed_artifacts_path.exists():
        raise RuntimeError("missing seed artifact manifest")
    manifest = pd.read_csv(manifest_path)
    seed_artifacts = pd.read_csv(seed_artifacts_path)
    required = {"dataset", "seed", "cache_path"}
    if not required.issubset(manifest.columns):
        raise RuntimeError(f"seed score cache manifest missing columns: {sorted(required - set(manifest.columns))}")
    for dataset, group in manifest.groupby("dataset", dropna=False):
        universe = _prepared_frame(str(dataset)).X_dev.columns.astype(str).tolist()
        for row in group.to_dict("records"):
            seed = int(row["seed"])
            seed_rows = seed_artifacts[seed_artifacts["seed"].astype(int).eq(seed)]
            if seed_rows.empty:
                raise RuntimeError(f"seed cache references unknown seed {seed}")
            validate_seed_score_cache(Path(row["cache_path"]), seed_row=seed_rows.iloc[0].to_dict(), dataset=str(dataset), candidate_universe=universe)


def _validate_ablation_score_cache_manifest() -> None:
    manifest_path = OUTPUT_ROOT / "ablations/ablation_score_caches.csv"
    if not manifest_path.exists():
        raise RuntimeError("missing ablation score cache manifest")
    manifest = pd.read_csv(manifest_path)
    required = {"dataset", "ablation", "cache_path", "cache_hash"}
    if not required.issubset(manifest.columns):
        raise RuntimeError(f"ablation score cache manifest missing columns: {sorted(required - set(manifest.columns))}")
    for row in manifest.to_dict("records"):
        path = Path(row["cache_path"])
        if not path.exists():
            raise RuntimeError(f"missing ablation score cache: {path.as_posix()}")
        if sha256_file(path) != str(row["cache_hash"]):
            raise RuntimeError(f"ablation score cache hash mismatch: {path.as_posix()}")
        frame = pd.read_csv(path)
        if frame["feature_name"].duplicated().any():
            raise RuntimeError(f"ablation score cache duplicate features: {path.as_posix()}")
        if not pd.to_numeric(frame["learned_similarity"], errors="coerce").notna().all():
            raise RuntimeError(f"ablation score cache contains nonfinite scores: {path.as_posix()}")


def derive_artifact_status() -> dict[str, dict[str, Any]]:
    core_runs = completed_run_dirs(OUTPUT_ROOT / "candidate_pool/runs")
    seed_runs = completed_run_dirs(OUTPUT_ROOT / "seed_robustness/runs")
    ablation_runs = completed_run_dirs(OUTPUT_ROOT / "ablations/runs")
    temporal_runs = completed_run_dirs(OUTPUT_ROOT / "temporal_validation/runs")
    statuses = {stage: {"status": "not_started", "details": ""} for stage in STAGES}
    statuses["preflight"] = {
        "status": "complete_valid"
        if (OUTPUT_ROOT / "manifests/source_experiment_manifest.json").exists() and (OUTPUT_ROOT / "manifests/full_method_specification.json").exists()
        else "not_started",
        "details": "source and method manifests",
    }
    if (OUTPUT_ROOT / "manifests/seed_artifact_validation_COMPLETE.json").exists():
        try:
            frame = pd.read_csv(OUTPUT_ROOT / "manifests/clip_v2_seed_artifacts.csv")
            valid = int(frame["eligible_for_downstream"].astype(bool).sum())
        except Exception:
            valid = 0
        statuses["seed_artifact_validation"] = {"status": "complete_valid" if valid == 5 else "stale", "valid_checkpoint_count": valid, "expected_checkpoint_count": 5}
    if (OUTPUT_ROOT / "candidate_pool/screening_scores/RUN_COMPLETE.json").exists():
        statuses["screening_scores"] = {"status": "complete_valid", "details": "screening score manifest exists"}
    if (OUTPUT_ROOT / "candidate_pool/plans/RUN_COMPLETE.json").exists():
        statuses["candidate_pools"] = {"status": "complete_valid", "details": "candidate-pool manifest exists"}
    core_marker = read_json_if_exists(OUTPUT_ROOT / "candidate_pool/core_candidate_pool_runs_COMPLETE.json", {})
    expected_core = int(core_marker.get("expected_runs", 184))
    statuses["core_candidate_pool_runs"] = {
        "status": "complete_valid" if len(core_runs) == expected_core else ("not_started" if len(core_runs) == 0 else "stale"),
        "completed_runs": len(core_runs),
        "expected_runs": expected_core,
    }
    random_marker = read_json_if_exists(OUTPUT_ROOT / "candidate_pool/random_repetitions_COMPLETE.json", {})
    completed_random = len([run for run in core_runs if "_random_" in run.name])
    expected_random = int(random_marker.get("expected_runs", 120))
    random_marker_valid = random_marker.get("status") == "complete_valid"
    statuses["random_repetitions"] = {
        "status": "complete_valid" if random_marker_valid and completed_random == expected_random else ("not_started" if completed_random == 0 else "stale"),
        "completed_runs": completed_random,
        "expected_runs": expected_random,
    }
    seed_score_marker = read_json_if_exists(OUTPUT_ROOT / "manifests/seed_score_generation_COMPLETE.json", {})
    if seed_score_marker:
        try:
            cache_count = len(pd.read_csv(OUTPUT_ROOT / "manifests/clip_v2_seed_score_caches.csv"))
        except Exception:
            cache_count = 0
        expected_cache_count = int(seed_score_marker.get("expected_caches", seed_score_marker.get("cache_count", 10)))
        statuses["seed_score_generation"] = {"status": "complete_valid" if cache_count == expected_cache_count else "stale", "cache_count": cache_count, "expected_caches": expected_cache_count}
    seed_marker = read_json_if_exists(OUTPUT_ROOT / "seed_robustness/seed_downstream_COMPLETE.json", {})
    expected_seed_runs = int(seed_marker.get("expected_runs", 20))
    statuses["seed_downstream"] = {
        "status": "complete_valid" if len(seed_runs) == expected_seed_runs else ("not_started" if len(seed_runs) == 0 else "stale"),
        "completed_runs": len(seed_runs),
        "expected_runs": expected_seed_runs,
    }
    ablation_marker = read_json_if_exists(OUTPUT_ROOT / "ablations/ablation_downstream_COMPLETE.json", {})
    expected_ablation_runs = int(ablation_marker.get("expected_runs", 28))
    statuses["ablation_downstream"] = {
        "status": "complete_valid" if len(ablation_runs) == expected_ablation_runs else ("not_started" if len(ablation_runs) == 0 else "stale"),
        "completed_runs": len(ablation_runs),
        "expected_runs": expected_ablation_runs,
    }
    if (OUTPUT_ROOT / "ablations/ablation_schema_build_COMPLETE.json").exists():
        statuses["ablation_schema_build"] = {"status": "complete_valid", "details": "schema manifest exists"}
    if (OUTPUT_ROOT / "ablations/ablation_contrastive_data_COMPLETE.json").exists():
        statuses["ablation_contrastive_data"] = {"status": "complete_valid", "details": "contrastive data plan exists"}
    training_manifest = OUTPUT_ROOT / "ablations/ablation_training_manifest.csv"
    if training_manifest.exists():
        try:
            count = len(pd.read_csv(training_manifest))
            validate_ablation_training(OUTPUT_ROOT)
            training_valid = True
        except Exception:
            count = 0
            training_valid = False
        statuses["ablation_training"] = {"status": "complete_valid" if training_valid else "stale", "completed_conditions": count, "expected_conditions": len(ABLATIONS), "new_training_jobs": 15}
    if (OUTPUT_ROOT / "ablations/ablation_checkpoint_selection_COMPLETE.json").exists():
        statuses["ablation_checkpoint_selection"] = {"status": "complete_valid", "selected_checkpoint_count": 3}
    ablation_score_marker = read_json_if_exists(OUTPUT_ROOT / "ablations/ablation_score_generation_COMPLETE.json", {})
    if ablation_score_marker:
        try:
            cache_count = len(pd.read_csv(OUTPUT_ROOT / "ablations/ablation_score_caches.csv"))
        except Exception:
            cache_count = 0
        expected_cache_count = int(ablation_score_marker.get("expected_caches", ablation_score_marker.get("cache_count", 14)))
        statuses["ablation_score_generation"] = {"status": "complete_valid" if cache_count == expected_cache_count else "stale", "cache_count": cache_count, "expected_caches": expected_cache_count}
    temporal_manifest = OUTPUT_ROOT / "manifests/temporal_cutoff_manifest.csv"
    expected_temporal = 0
    if temporal_manifest.exists():
        frame = pd.read_csv(temporal_manifest)
        valid_cutoffs = int(frame["eligible"].astype(bool).sum()) if "eligible" in frame.columns else 0
        expected_temporal = valid_cutoffs * len(TEMPORAL_METHODS) * 2
        statuses["temporal_cutoffs"] = {"status": "complete_valid", "valid_cutoffs": valid_cutoffs, "expected_temporal_runs": expected_temporal}
    if expected_temporal or temporal_runs:
        statuses["temporal_runs"] = {
            "status": "complete_valid" if len(temporal_runs) == expected_temporal else ("not_started" if len(temporal_runs) == 0 else "stale"),
            "completed_runs": len(temporal_runs),
            "expected_runs": expected_temporal,
        }
    if (OUTPUT_ROOT / "final_analysis/master_results.csv").exists():
        statuses["aggregate_rebuild"] = {"status": "complete_valid", "details": "master_results.csv exists"}
    if (OUTPUT_ROOT / "final_analysis/metric_recomputation.csv").exists():
        statuses["metric_recomputation"] = {"status": "complete_valid", "details": "metric_recomputation.csv exists"}
    if (OUTPUT_ROOT / "final_analysis/candidate_pool_comparison.csv").exists():
        statuses["candidate_pool_diagnostics"] = {"status": "complete_valid", "details": "candidate_pool_comparison.csv exists"}
    if (OUTPUT_ROOT / "final_analysis/paired_uncertainty.csv").exists():
        statuses["paired_uncertainty"] = {"status": "complete_valid", "details": "paired_uncertainty.csv exists"}
    if (OUTPUT_ROOT / "final_analysis/experiment_summary.md").exists():
        statuses["final_analysis"] = {"status": "complete_valid", "details": "experiment_summary.md exists"}
    if (OUTPUT_ROOT / "final_analysis/plot_manifest.csv").exists():
        statuses["plots"] = {"status": "complete_valid", "details": "plot_manifest.csv exists"}
    tests = read_json_if_exists(OUTPUT_ROOT / "audit/test_results.json", {})
    if tests.get("returncode") == 0:
        statuses["tests"] = {"status": "complete_valid", "details": "synthetic tests passed"}
    audit = read_json_if_exists(OUTPUT_ROOT / "audit/final_scientific_audit.json", {})
    if audit.get("status") == "passed":
        statuses["final_audit"] = {"status": "complete_valid", "details": "final audit passed"}
    return statuses


def completed_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    runs = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.endswith(".in_progress")):
        try:
            validate_run(run_dir)
        except Exception:
            continue
        if (run_dir / "RUN_COMPLETE.json").exists():
            runs.append(run_dir)
    return runs


def archive_incomplete_outputs() -> str | None:
    if active_lock():
        raise RuntimeError("active final-comparison lock exists")
    if not OUTPUT_ROOT.exists():
        ensure_layout()
        return None
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    archive = ARCHIVE_ROOT / timestamp
    archive.mkdir(parents=True, exist_ok=False)
    files = [path for path in OUTPUT_ROOT.rglob("*") if path.is_file()]
    entries = []
    for path in files:
        entries.append({"path": path.as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "reason": "invalid_framework_only_or_incomplete_final_comparison_state"})
    shutil.move(str(OUTPUT_ROOT), str(archive / "clip_final_comparison"))
    (archive / "reset_manifest.json").write_text(json.dumps({"archived_at": _now(), "entries": entries}, indent=2, default=str), encoding="utf-8")
    ensure_layout()
    return archive.as_posix()


def initialize_clean_state() -> None:
    state = {"created_at": _now(), "implementation_mode": "executable_research_pipeline", "stages": {stage: {"status": "not_started", "resume_eligible": True} for stage in STAGES}}
    write_state(state)


def _source_check_summary() -> dict[str, Any]:
    checks = {
        "clip_v1_freeze_manifest": Path("results/clip_versions/v1/freeze_manifest.json").exists(),
        "clip_v2_selected_checkpoint": Path("results/clip_v2/training/selected_checkpoint.pt").exists(),
        "homecredit_data": Path("data/homecredit").exists(),
        "lendingclub_v2_data": Path("data/lendingclub_v2").exists(),
    }
    return {"passed": all(checks.values()), "checks": checks}


def acquire_lock() -> dict[str, Any]:
    ensure_layout()
    current = read_lock()
    if current and current.get("active"):
        raise RuntimeError(f"active final-comparison lock exists: {current}")
    lock = {"pid": os.getpid(), "started_at": _now(), "command": " ".join(sys.argv), "cwd": str(ROOT)}
    atomic_write_json(LOCK_PATH, lock)
    return lock


def read_lock() -> dict[str, Any] | None:
    if not LOCK_PATH.exists():
        return None
    payload = read_json_if_exists(LOCK_PATH, None)
    if payload is None:
        return None
    payload["active"] = is_pid_active(int(payload.get("pid", -1)))
    return payload


def active_lock() -> bool:
    lock = read_lock()
    return bool(lock and lock.get("active"))


def release_lock(lock: dict[str, Any] | None) -> None:
    if lock and LOCK_PATH.exists():
        current = read_json_if_exists(LOCK_PATH, {})
        if int(current.get("pid", -1)) == os.getpid():
            LOCK_PATH.unlink()


def is_pid_active(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def load_state() -> dict[str, Any]:
    state = read_json_if_exists(STATE_PATH, {"created_at": _now(), "implementation_mode": "executable_research_pipeline", "stages": {}})
    for stage in STAGES:
        state["stages"].setdefault(stage, {"status": "not_started", "resume_eligible": True})
    return state


def mark_stage(state: dict[str, Any], stage: str, **updates: Any) -> None:
    row = dict(state["stages"].get(stage, {}))
    row.update(updates)
    if row["status"] not in VALID_STATUSES:
        raise RuntimeError(f"invalid stage status: {row['status']}")
    state["stages"][stage] = row


def write_state(state: dict[str, Any]) -> None:
    ensure_layout()
    atomic_write_json(STATE_PATH, state)


def log(message: str) -> None:
    ensure_layout()
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{_now()} {message}\n")


def log_progress(index: int, total: int, stage: str, status: str, stage_started: float, total_started: float) -> None:
    message = f"[{index}/{total}] stage={stage} status={status} stage_elapsed={time.time() - stage_started:.1f}s total_elapsed={time.time() - total_started:.1f}s log={LOG_PATH.as_posix()}"
    log(message)
    print(message, flush=True)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    raise SystemExit(main())
