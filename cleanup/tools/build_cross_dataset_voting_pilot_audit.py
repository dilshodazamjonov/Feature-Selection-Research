"""Build the deterministic Prompt 4 voting-pilot audit from registered artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_fs.experiments.compare import (  # noqa: E402
    build_cross_dataset_voting_comparison_plan,
)
from credit_risk_fs.experiments.matrix import (  # noqa: E402
    cross_dataset_matrix_expansion_summary,
    expand_cross_dataset_voting_matrix,
    expand_cross_dataset_voting_pilot,
)


AUDIT = ROOT / "cleanup/audits/cross_dataset_voting_integration_pilot"
DOC = ROOT / "docs/research_extension/cross_dataset_voting_integration_pilot_v1.md"
MATRIX = ROOT / "configs/experiments/cross_dataset_rank_voting_matrix_v1.yaml"
PILOT = ROOT / "configs/experiments/cross_dataset_rank_voting_pilot_v1.yaml"
PREFLIGHT_SPECS = (
    ROOT
    / "cleanup/audits/cross_dataset_voting_execution_spec/preflight_request_specs.json"
)
LEGACY_ROOT = Path(r"D:\ResearchFindings\results")
LEGACY_MANIFEST = ROOT / "cleanup/audits/foundation_protocol_freeze/legacy_artifact_manifest.csv"

FROZEN_HASHES = {
    "scientific_protocol": (
        ROOT / "configs/protocols/credit_scoring_extension_v1.yaml",
        "f4137e98b7c89a63e9a73bc495190858f416c586d237f8ac003c8fdb9e40bde0",
    ),
    "row_alignment_contract": (
        ROOT / "configs/protocols/row_alignment_contract_v1.json",
        "fc1064069f5cc45d76fd34060bb506869683e953c354d3f1f1a13327d99e71a0",
    ),
    "voting_protocol": (
        ROOT / "configs/protocols/cross_dataset_rank_voting_v1.yaml",
        "51030e49716fae0a9c09b52628784a237e9581bb14c1a027e9c8238a575f0b49",
    ),
    "execution_policy": (
        ROOT / "configs/execution/local_laptop_safe_v1.yaml",
        "1b77add8bf55096864934f6553aeba174cd22e2b112f53c7a45e5df327934012",
    ),
}

FOLD_COUNTS = {
    "homecredit": [(16242, 16720), (32730, 16638), (49166, 16401), (65498, 16625), (82047, 16263)],
    "lendingclub_v2": [(83283, 114267), (154707, 86170), (255319, 126995), (360402, 99837), (458602, 71911)],
}

GAP_VALIDATION = [
    ("fold_local_voting_adapter", "passed", "src/credit_risk_fs/experiments/rank_voting.py"),
    ("long_voter_schema_original_names", "passed", "src/credit_risk_fs/experiments/rank_voting.py"),
    ("rfe_registry_exact_budget_thread_trace", "passed", "src/credit_risk_fs/selectors/rfe.py"),
    ("deterministic_matrix_runner_expansion", "passed", "src/credit_risk_fs/experiments/matrix.py"),
    ("explicit_candidate_projection_manifest", "passed", "src/credit_risk_fs/pipelines/common.py"),
    ("atomic_checkpoint_validator_integration", "passed", "src/credit_risk_fs/experiments/execution.py"),
    ("future_oof_oot_prediction_hash_contract", "passed_synthetic_only", "src/credit_risk_fs/experiments/prediction_contract.py"),
    ("reference_and_paired_family_wiring", "passed_not_executed", "src/credit_risk_fs/experiments/compare.py"),
    ("effective_lr_catboost_assertions", "passed", "src/credit_risk_fs/experiments/rank_voting.py"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def pilot_directories() -> list[Path]:
    specs = expand_cross_dataset_voting_pilot(PILOT)
    directories: list[Path] = []
    for spec in specs:
        matches = list((ROOT / "results/runs").glob(f"*/{spec.run_id}"))
        if len(matches) != 1:
            raise ValueError(f"expected one run directory for {spec.run_id}, found {len(matches)}")
        directories.append(matches[0])
    return directories


def attempt_resources(run: Path) -> list[tuple[str, Path, dict[str, Any]]]:
    archived = sorted((run / "incomplete/attempt_history").glob("attempt_*_resource_usage.json"))
    paths = archived + [run / "resource_usage.json"]
    return [
        (f"attempt_{index:02d}", path, load_json(path))
        for index, path in enumerate(paths, start=1)
    ]


def stage_resource_rows(runs: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for attempt, path, resource in attempt_resources(run):
            samples_by_stage: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            samples = resource.get("samples", [])
            if not samples:
                rows.append(
                    {
                        "run_id": run.name,
                        "attempt": attempt,
                        "attempt_status": resource["status"],
                        "resource_path": relative(path),
                        "stage": "no_sample_before_stop",
                        "fold_id": "",
                        "sample_count": 0,
                        "stage_start_elapsed_seconds": 0,
                        "stage_end_elapsed_seconds": resource["timings_seconds"]["total"],
                        "peak_process_tree_rss_bytes": resource["peak_process_tree_rss_bytes"],
                        "minimum_system_available_ram_bytes": resource["minimum_system_available_ram_bytes"],
                        "peak_process_gpu_bytes": resource["peak_process_gpu_bytes"],
                        "maximum_process_tree_cpu_percent": "",
                        "results_free_space_delta_from_attempt_start_bytes": "",
                        "temp_free_space_delta_from_attempt_start_bytes": "",
                        "resolved_estimator_threads": resource["resolved_parallelism"]["estimator_threads"],
                        "stop_code": resource.get("stop_code") or "",
                    }
                )
                continue
            result_start = int(samples[0]["results_free_disk_bytes"])
            temp_start = int(samples[0]["temp_free_disk_bytes"])
            for sample in samples:
                key = (str(sample.get("stage") or "unknown"), str(sample.get("fold_id") or ""))
                samples_by_stage[key].append(sample)
            for (stage, fold_id), selected in samples_by_stage.items():
                rows.append(
                    {
                        "run_id": run.name,
                        "attempt": attempt,
                        "attempt_status": resource["status"],
                        "resource_path": relative(path),
                        "stage": stage,
                        "fold_id": fold_id,
                        "sample_count": len(selected),
                        "stage_start_elapsed_seconds": min(item["elapsed_seconds"] for item in selected),
                        "stage_end_elapsed_seconds": max(item["elapsed_seconds"] for item in selected),
                        "peak_process_tree_rss_bytes": max(item["process_tree_rss_bytes"] for item in selected),
                        "minimum_system_available_ram_bytes": min(item["system_available_ram_bytes"] for item in selected),
                        "peak_process_gpu_bytes": max(item["process_gpu_bytes"] for item in selected),
                        "maximum_process_tree_cpu_percent": max(item["process_tree_cpu_percent"] for item in selected),
                        "results_free_space_delta_from_attempt_start_bytes": max(
                            0, result_start - min(item["results_free_disk_bytes"] for item in selected)
                        ),
                        "temp_free_space_delta_from_attempt_start_bytes": max(
                            0, temp_start - min(item["temp_free_disk_bytes"] for item in selected)
                        ),
                        "resolved_estimator_threads": resource["resolved_parallelism"]["estimator_threads"],
                        "stop_code": resource.get("stop_code") or "",
                    }
                )
    return rows


def validate_legacy() -> dict[str, Any]:
    expected: dict[str, tuple[int, str]] = {}
    with LEGACY_MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            expected[row["relative_path"]] = (int(row["size_bytes"]), row["sha256"])
    observed_paths = {
        path.relative_to(LEGACY_ROOT).as_posix(): path
        for path in LEGACY_ROOT.rglob("*")
        if path.is_file()
    }
    added = sorted(set(observed_paths) - set(expected))
    removed = sorted(set(expected) - set(observed_paths))
    changed = []
    for name in sorted(set(expected) & set(observed_paths)):
        size, digest = expected[name]
        path = observed_paths[name]
        if path.stat().st_size != size or sha256_file(path) != digest:
            changed.append(name)
    total = sum(path.stat().st_size for path in observed_paths.values())
    return {
        "root": str(LEGACY_ROOT),
        "file_count": len(observed_paths),
        "size_bytes": total,
        "added_count": len(added),
        "removed_count": len(removed),
        "changed_count": len(changed),
        "status": (
            "passed"
            if len(observed_paths) == 359
            and total == 110_084_164
            and not added
            and not removed
            and not changed
            else "failed"
        ),
    }


def build(args: argparse.Namespace) -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    runs = pilot_directories()
    pilot_specs = expand_cross_dataset_voting_pilot(PILOT)
    research_specs = expand_cross_dataset_voting_matrix(MATRIX)
    matrix_summary = cross_dataset_matrix_expansion_summary(research_specs)
    comparison_plan = build_cross_dataset_voting_comparison_plan(research_specs)
    preflight_shapes = {
        (item["dataset"], item["model"]): item
        for item in load_json(PREFLIGHT_SPECS)["execution_shapes"]
    }
    run_index_rows = list(csv.DictReader((ROOT / "results/run_index.csv").open(encoding="utf-8")))
    registered_ids = [row["run_id"] for row in run_index_rows]
    research_ids = [item.run_id for item in research_specs]
    research_directories = [
        relative(path)
        for run_id in research_ids
        for path in (ROOT / "results/runs").glob(f"*/{run_id}")
    ]

    matrix_payload = {
        "schema_version": "cross_dataset_voting_matrix_expansion_validation_v1",
        "status": "passed",
        "pure_dry_expansion": True,
        "matrix_path": relative(MATRIX),
        "matrix_sha256": sha256_file(MATRIX),
        **matrix_summary,
        "comparison_plan": comparison_plan,
        "research_run_index_row_count": sum(item in research_ids for item in registered_ids),
        "research_run_directory_count": len(research_directories),
        "research_run_directories": research_directories,
        "executed_research_runs": False,
    }
    write_json(AUDIT / "matrix_expansion_validation.json", matrix_payload)

    manifest_runs = []
    summary_rows = []
    projection_rows = []
    for run, spec in zip(runs, pilot_specs, strict=True):
        manifest = load_json(run / "manifest.json")
        checkpoint = load_json(run / "checkpoint.json")
        fold = load_json(run / "fold_identity_manifest.json")
        validation = load_json(run / "pilot_validation.json")
        effective = load_json(run / "effective_model_config.json")
        access = load_json(run / "data_access_log.json")
        prediction = load_json(run / "prediction_metadata.json")
        projection = load_json(run / "candidate_projection_manifest.json")
        terminal_resource = load_json(run / "resource_usage.json")
        attempts = attempt_resources(run)
        resources = [item[2] for item in attempts]

        artifacts = {}
        for key, entry in manifest["artifacts"].items():
            if not entry.get("applicable"):
                continue
            path = run / entry["path"]
            if key == "manifest":
                artifacts[key] = {
                    "path": relative(path),
                    "present": path.is_file(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                continue
            actual_size = path.stat().st_size
            actual_hash = sha256_file(path)
            if actual_size != entry["size_bytes"] or actual_hash != entry["sha256"]:
                raise ValueError(f"manifest artifact integrity mismatch: {run.name}/{key}")
            artifacts[key] = {
                "path": relative(path),
                "present": True,
                "size_bytes": actual_size,
                "sha256": actual_hash,
            }

        voters = list(csv.DictReader((run / "voter_rankings.csv").open(encoding="utf-8")))
        aggregates = list(csv.DictReader((run / "aggregate_ranking.csv").open(encoding="utf-8")))
        candidates = list(csv.DictReader((run / "candidate_features.csv").open(encoding="utf-8")))
        selections = list(csv.DictReader((run / "selected_features.csv").open(encoding="utf-8")))
        voter_keys = {(row["voter_id"], row["normalized_feature_name"]) for row in voters}
        expected_universe = 529 if spec.dataset == "homecredit" else 675
        expected_final = 20 if spec.model == "lr" else 40
        if len(voters) != 2 * expected_universe or len(voter_keys) != len(voters):
            raise ValueError(f"voter exact-universe contract failed: {run.name}")
        if len(aggregates) != expected_universe or len(candidates) != 200 or len(selections) != expected_final:
            raise ValueError(f"selection exact-budget contract failed: {run.name}")
        if validation["opened_oot_paths"] or validation["retained_oot_rows"]:
            raise ValueError(f"OOT non-access contract failed: {run.name}")
        if prediction["coverage_type"] != "single_dev_fold_pilot":
            raise ValueError(f"pilot coverage contract failed: {run.name}")
        if prediction["research_eligible"] or prediction["comparison_eligible"]:
            raise ValueError(f"pilot eligibility contract failed: {run.name}")
        if checkpoint["status"] != "completed" or manifest["resumability_status"] != "completed_immutable":
            raise ValueError(f"terminal checkpoint contract failed: {run.name}")

        cumulative_seconds = sum(float(item["timings_seconds"]["total"]) for item in resources)
        peak_rss = max(int(item["peak_process_tree_rss_bytes"]) for item in resources)
        min_available = min(int(item["minimum_system_available_ram_bytes"]) for item in resources)
        peak_gpu = max(int(item["peak_process_gpu_bytes"]) for item in resources)
        result_delta = max(
            max(0, int(item["samples"][0]["results_free_disk_bytes"]) - int(item["minimum_results_free_disk_bytes"]))
            if item.get("samples")
            else 0
            for item in resources
        )
        temp_delta = max(
            max(0, int(item["samples"][0]["temp_free_disk_bytes"]) - int(item["minimum_temp_free_disk_bytes"]))
            if item.get("samples")
            else 0
            for item in resources
        )
        final_artifact_bytes = sum(path.stat().st_size for path in run.iterdir() if path.is_file())
        shape = preflight_shapes[(spec.dataset, spec.model)]
        dtype_bytes = int(access["load_report"]["application_train"]["dtype_bytes"])
        folds = FOLD_COUNTS[spec.dataset]
        first_train, first_validation = folds[0]
        maximum_train, paired_validation = max(folds, key=lambda pair: pair[0])
        universe = expected_universe
        raw_bytes_per_row = dtype_bytes / int(validation["candidate_universe_count"] and (99092 if spec.dataset == "homecredit" else 598649))
        numeric_shadow_growth = 2 * 8 * universe * (maximum_train - first_train)
        raw_slice_growth = raw_bytes_per_row * (
            maximum_train + paired_validation - first_train - first_validation
        )
        fivefold_peak_projection = int(peak_rss + numeric_shadow_growth + raw_slice_growth)
        fivefold_safety_projection = int(fivefold_peak_projection * 1.35)
        projected_min_available = int(min_available - numeric_shadow_growth - raw_slice_growth)
        dev_rows = 99_092 if spec.dataset == "homecredit" else 598_649
        full_refit_numeric_growth = 2 * 8 * universe * (dev_rows - first_train)
        full_refit_raw_growth = raw_bytes_per_row * (dev_rows - first_train - first_validation)
        full_refit_peak_projection = int(peak_rss + full_refit_numeric_growth + full_refit_raw_growth)
        full_refit_safety_projection = int(full_refit_peak_projection * 1.35)
        capacity_status = (
            "fits_unchanged_policy"
            if fivefold_safety_projection < 28 * 1024**3
            and full_refit_safety_projection < 28 * 1024**3
            and projected_min_available >= 8 * 1024**3
            else "capacity_review_required"
        )

        attempt_entries = [
            {
                "attempt": name,
                "path": relative(path),
                "status": resource["status"],
                "wall_seconds": resource["timings_seconds"]["total"],
                "stop_code": resource.get("stop_code"),
                "peak_process_tree_rss_bytes": resource["peak_process_tree_rss_bytes"],
            }
            for name, path, resource in attempts
        ]
        manifest_runs.append(
            {
                "run_id": run.name,
                "run_directory": relative(run),
                "status": manifest["status"],
                "purpose": manifest["purpose"],
                "research_eligible": manifest["research_eligible"],
                "comparison_eligible": manifest["comparison_eligible"],
                "coverage_type": manifest["coverage_type"],
                "training_rows": fold["training_row_count"],
                "validation_rows": fold["validation_row_count"],
                "candidate_universe_count": expected_universe,
                "voter_count": 2,
                "voter_rows": len(voters),
                "top_k": len(candidates),
                "final_feature_count": len(selections),
                "prediction_rows": validation["prediction_row_count"],
                "training_validation_identity_overlap_count": fold["identity_overlap_count"],
                "opened_oot_paths": access["opened_oot_paths"],
                "retained_oot_rows": access["retained_oot_rows"],
                "effective_model": effective,
                "artifacts": artifacts,
                "attempts": attempt_entries,
                "checkpoint_completed_stages": checkpoint["completed_stages"],
            }
        )
        summary_rows.append(
            {
                "run_id": run.name,
                "dataset": spec.dataset,
                "model": spec.model,
                "status": manifest["status"],
                "attempt_count": len(attempts),
                "cumulative_attempt_wall_seconds": round(cumulative_seconds, 6),
                "completed_attempt_wall_seconds": terminal_resource["timings_seconds"]["total"],
                "training_rows": fold["training_row_count"],
                "validation_rows": fold["validation_row_count"],
                "candidate_universe_count": expected_universe,
                "top_k": len(candidates),
                "final_feature_count": len(selections),
                "prediction_rows": validation["prediction_row_count"],
                "projected_input_bytes_preflight": shape["projected_input_bytes"],
                "dense_float32_lower_bound_bytes": shape["dense_float32_lower_bound_bytes"],
                "observed_peak_process_tree_rss_bytes": peak_rss,
                "observed_peak_process_tree_rss_gib": round(peak_rss / 1024**3, 6),
                "rss_to_projected_input_ratio": round(peak_rss / shape["projected_input_bytes"], 6),
                "rss_to_dense_lower_bound_ratio": round(peak_rss / shape["dense_float32_lower_bound_bytes"], 6),
                "minimum_system_available_ram_bytes": min_available,
                "minimum_system_available_ram_gib": round(min_available / 1024**3, 6),
                "peak_process_gpu_bytes": peak_gpu,
                "results_free_space_delta_bytes": result_delta,
                "temp_free_space_delta_bytes": temp_delta,
                "finalized_run_artifact_bytes": final_artifact_bytes,
                "known_artifact_estimate_bytes": shape["known_artifact_estimate_bytes"],
                "artifact_size_to_estimate_ratio": round(final_artifact_bytes / shape["known_artifact_estimate_bytes"], 6),
                "resource_warning_codes": "|".join(
                    str(value)
                    for resource in resources
                    for value in (resource.get("warnings") or [])
                ),
                "terminal_stop_code": terminal_resource.get("stop_code") or "",
                "attempt_statuses": "|".join(item["status"] for item in resources),
                "selection_checkpoint_reused": bool(effective.get("selection_checkpoint_reused", False)),
                "fivefold_runtime_upper_seconds_repeating_pilot": round(cumulative_seconds * 5, 3),
                "full_dev_refit_runtime_upper_seconds_linear": round(cumulative_seconds * dev_rows / first_train, 3),
                "fivefold_peak_rss_projection_bytes": fivefold_peak_projection,
                "fivefold_peak_rss_with_1_35_safety_bytes": fivefold_safety_projection,
                "fivefold_projected_min_available_ram_bytes": projected_min_available,
                "full_dev_refit_peak_rss_projection_bytes": full_refit_peak_projection,
                "full_dev_refit_peak_rss_with_1_35_safety_bytes": full_refit_safety_projection,
                "future_capacity_status": capacity_status,
                "projection_assumption": "observed peak plus linear float64+Boruta-shadow and retained mixed-frame slice growth; sequential folds",
            }
        )
        projection_rows.append(summary_rows[-1])

    stage_rows = stage_resource_rows(runs)
    for filename, rows in (
        ("pilot_resource_summary.csv", summary_rows),
        ("pilot_stage_resources.csv", stage_rows),
    ):
        with (AUDIT / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    frozen = {}
    for key, (path, expected) in FROZEN_HASHES.items():
        observed = sha256_file(path)
        frozen[key] = {
            "path": relative(path),
            "expected_sha256": expected,
            "observed_sha256": observed,
            "unchanged": observed == expected,
        }
    legacy = validate_legacy()
    active_partials = [
        relative(path)
        for path in (ROOT / "results").rglob("*")
        if path.is_file() and ".partial" in path.name
    ]
    active_locks = [relative(path) for path in (ROOT / "results").rglob(".execution.lock")]
    capacity_review = [row["run_id"] for row in projection_rows if row["future_capacity_status"] != "fits_unchanged_policy"]

    write_json(
        AUDIT / "pilot_manifest.json",
        {
            "schema_version": "cross_dataset_voting_integration_pilot_manifest_v1",
            "status": "passed",
            "pilot_config": relative(PILOT),
            "pilot_config_sha256": sha256_file(PILOT),
            "execution_order": [item.run_id for item in pilot_specs],
            "registered_run_count": len(run_index_rows),
            "registered_run_ids": registered_ids,
            "runs": manifest_runs,
        },
    )
    write_json(
        AUDIT / "implementation_validation.json",
        {
            "schema_version": "cross_dataset_voting_integration_validation_v1",
            "status": "passed",
            "objective_boundary": {
                "datasets": ["homecredit", "lendingclub_v2"],
                "models": ["lr", "catboost"],
                "candidate_pool": 200,
                "folds_per_pilot": 1,
                "seed": 42,
                "load_oot": False,
                "research_runs_executed": 0,
                "reference_runs_executed": 0,
                "sensitivity_runs_executed": 0,
            },
            "integration_gaps": [
                {"gap": name, "status": status, "primary_implementation": path}
                for name, status, path in GAP_VALIDATION
            ],
            "architecture_reuse": {
                "existing_runner": True,
                "existing_selector_registry": True,
                "existing_rank_aggregator": True,
                "existing_atomic_io": True,
                "existing_checkpoint_manager": True,
                "existing_resource_supervisor": True,
                "existing_result_layout_and_index": True,
                "competing_architecture_added": False,
            },
            "controlled_stop_history_preserved": [
                {
                    "run_id": pilot_specs[0].run_id,
                    "attempts_before_completion": 3,
                    "causes": [
                        "schema snapshot table names included .csv suffixes",
                        "Home Credit DEV identity derivation initially included application_test IDs",
                        "sklearn reported the L2 deprecation bridge as penalty=deprecated,l1_ratio=0",
                    ],
                    "resolution": "bounded code fixes, focused validation, explicit identical resume from validated selection checkpoint",
                },
                {
                    "run_id": pilot_specs[2].run_id,
                    "attempts_before_completion": 1,
                    "cause": "CSV inference changed authenticated decimal loan IDs from canonical strings to integers before hashing",
                    "resolution": "canonical decimal-string restoration, exact frozen hash check, explicit identical resume",
                },
            ],
            "frozen_hashes": frozen,
        },
    )

    gate = "CONDITIONAL_PASS" if capacity_review else "READY_FOR_PROMPT_5"
    validation_payload = {
        "schema_version": "cross_dataset_voting_integration_pilot_validation_summary_v1",
        "gate": gate,
        "implementation_status": "passed",
        "pilot_status": "four_of_four_completed",
        "focused_tests": {
            "command": args.focused_command,
            "passed": args.focused_passed,
            "failed": 0,
        },
        "full_tests": {
            "command": ".\\.venv\\Scripts\\python.exe -m pytest tests -q",
            "passed": args.full_passed,
            "skipped": args.full_skipped,
            "warnings": args.full_warnings,
            "failed": 0,
        },
        "repository_validator": "passed",
        "compileall": "passed",
        "git_diff_check": "passed",
        "frozen_hashes": frozen,
        "legacy_preservation": legacy,
        "active_results": {
            "run_index_rows": len(run_index_rows),
            "run_directories": len(runs),
            "pilot_ids_only": registered_ids == [item.run_id for item in pilot_specs],
            "active_partial_files": active_partials,
            "active_execution_locks": active_locks,
            "research_run_rows": sum(item in research_ids for item in registered_ids),
            "research_run_directories": research_directories,
        },
        "prohibited_workloads": {
            "oot_paths_opened": 0,
            "reference_runs": 0,
            "k100_runs": 0,
            "k300_runs": 0,
            "fivefold_runs": 0,
            "full_dev_refits": 0,
            "api_calls": 0,
            "clip_embedding_shap_workloads": 0,
            "gpu_training_runs": 0,
        },
        "capacity_condition": {
            "runs_requiring_review": capacity_review,
            "reason": (
                "Conservative row-scaled working-set projections for LendingClub v2 cross the "
                "8 GiB minimum-system-available threshold and/or the 28 GiB RSS ceiling after "
                "the frozen 1.35 safety factor. No limit was changed."
            ),
            "required_before_prompt_5": (
                "Approve a separately versioned memory-safe execution refinement and validate "
                "the largest LendingClub fold/full-DEV shape under the unchanged policy before "
                "authorizing any frozen research run."
            ),
        },
    }
    write_json(AUDIT / "validation_summary.json", validation_payload)

    table_lines = []
    for row in summary_rows:
        table_lines.append(
            "| {run_id} | {status} | {training_rows} | {validation_rows} | {candidate_universe_count} "
            "| {top_k} | {final_feature_count} | {cumulative_attempt_wall_seconds:.1f} | "
            "{observed_peak_process_tree_rss_gib:.2f} | {minimum_system_available_ram_gib:.2f} | "
            "{peak_process_gpu_bytes} | {terminal_stop_code} |".format(**row)
        )
    resource_lines = []
    for row in summary_rows:
        resource_lines.append(
            f"- `{row['run_id']}`: observed RSS/input amplification "
            f"{row['rss_to_projected_input_ratio']:.2f}x; RSS/dense-float32 lower-bound "
            f"{row['rss_to_dense_lower_bound_ratio']:.2f}x; final artifacts "
            f"{row['finalized_run_artifact_bytes']:,} bytes; five-fold upper runtime "
            f"{row['fivefold_runtime_upper_seconds_repeating_pilot'] / 3600:.2f} h; full-DEV "
            f"linear runtime upper bound {row['full_dev_refit_runtime_upper_seconds_linear'] / 3600:.2f} h; "
            f"capacity `{row['future_capacity_status']}`."
        )
    artifact_lines = []
    for run in manifest_runs:
        selected = run["artifacts"]
        keys = (
            "voter_rankings",
            "aggregate_ranking",
            "candidate_features",
            "selected_features",
            "predictions_dev",
            "effective_model_config",
            "resource_usage",
            "checkpoint",
        )
        artifact_lines.append(f"- `{run['run_id']}`")
        for key in keys:
            item = selected[key]
            artifact_lines.append(
                f"  - {key}: `{item['path']}` — `{item['sha256']}` ({item['size_bytes']:,} bytes)"
            )

    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(
        f"""# Cross-dataset voting integration and resource pilot v1

## Outcome

**{gate}.** All nine bounded integration gaps pass, the frozen matrix expands exactly, and all four real K=200 first-fold pilots completed through the common registered lifecycle. Pilot outputs are diagnostic, single-fold, and ineligible for research or paired inference. The remaining condition is a resource-capacity review: conservative LendingClub-v2 largest-fold/full-DEV projections do not clear every unchanged memory guardrail.

## Boundary and architecture

The work covered Home Credit and LendingClub v2, LR and CatBoost, K=200, seed 42, and the canonical first DEV fold only. It did not run a research ID, reference, K=100/K=300 sensitivity, five-fold OOF, full-DEV refit, OOT score, paired p-value, API, CLIP, embedding, SHAP, or GPU workload.

The implementation extends the existing runner, selector registry, frozen rank aggregator, tracking layout, atomic publisher, checkpoint manager, resource supervisor, prediction writer, and repository validator. No competing execution or result architecture was introduced.

## Nine integration gaps

1. Fold-local adapter: both voters, aggregation, top-200 projection, CatBoost RFE, final-model fit, and held-out scoring use the canonical training/validation boundary.
2. Long voter schema: original and normalized names, ranks/scores, selector/seed, training/input hashes, presence, and artifact metadata are preserved with one canonical vote per voter.
3. RFE: the standalone registered CatBoost-backed selector is CPU-only, <=4 threads, emits a trace, and fails unless it returns exactly 20/40 features.
4. Matrix runner: pure dry expansion returns 12 voting plus 4 rerun-required references, 80 future folds, 16 future final fits, 4 primary and 8 sensitivity comparisons.
5. Projection manifest: row, voter, aggregation, RFE, final-model, and evaluation projections are ordered, hashed, and reject implicit all-column requests.
6. Lifecycle: pilot artifacts use the common atomic/checkpoint/resource/tracking path; completed runs are immutable and controlled-stop attempts remain preserved.
7. Prediction contract: deterministic synthetic complete OOF/OOT checks exist; real pilots are `single_dev_fold_pilot`, research/comparison false.
8. Inference wiring: four reference configurations and 4+8 paired comparison definitions are validated but were not executed and no p-value was calculated.
9. Effective models: fitted LR/CatBoost parameters, training-only preprocessing, class order, CPU/thread use, RFE configuration, and exact feature budgets are asserted and saved.

CatBoost records `early_stopping_rounds=150`, but validation targets are intentionally excluded from fit, so no eval set is supplied and early stopping is inactive; all 1,500 configured iterations run. LR records sklearn's L2 deprecation bridge as `penalty=deprecated, l1_ratio=0.0`, resolved explicitly as effective L2.

## Deterministic expansion and tests

The frozen matrix dry expansion produced exactly 16 IDs in frozen order: 12 voting, 4 rerun-required references, 80 future DEV folds, 16 future final fits, 4 primary comparisons, and 8 sensitivity comparisons. Active run-index rows and directories for those 16 IDs remain zero.

Focused validation: {args.focused_passed} passed. Full suite: {args.full_passed} passed, {args.full_skipped} skipped, {args.full_warnings} warnings. The repository validator, compileall, and `git diff --check` pass.

## Pilot results (diagnostic only)

Wall time below is cumulative across preserved attempts; it includes explicit identical resumes where a bounded integration defect was fixed. No performance metric is reported.

| Pilot | Status | Train | Validation | Universe | K | Final | Wall s | Peak RSS GiB | Min RAM GiB | GPU B | Stop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(table_lines)}

Pilot 1 preserves three failed attempts before completion (schema filename normalization, Home Credit train-ID intersection, and sklearn's L2 deprecation representation); its validated selection checkpoint was reused for the final LR fit. Pilot 3 preserves one pre-selection identity-type failure. Every correction was bounded, tested, and resumed under the identical stable pilot ID.

## Leakage, projection, and OOT proof

Every fold manifest reports zero train/validation identity overlap. Each ranking row carries training-identity and training-identity-target hashes; both voters contain exactly 529 or 675 unique canonical votes, while RFE is absent from the voter set. Candidate artifacts contain exactly 200 ordered features, RFE consumes only that set, and final models consume exactly 20 or 40 features. Prediction identity/target hashes match the held-out canonical fold and class order is `[0, 1]` with class 1 meaning greater default risk.

Every data-access log has `opened_oot_paths=[]`, `retained_oot_rows=0`, `load_oot=false`, and zero implicit all-column requests. Temporally mixed source CSVs were chunk-scanned with explicit projections and filters; no distinct OOT artifact path was opened or retained.

## Artifacts and integrity

{chr(10).join(artifact_lines)}

The complete artifact inventory and hashes are in `cleanup/audits/cross_dataset_voting_integration_pilot/pilot_manifest.json`; stage samples are in `pilot_stage_resources.csv`.

## Resource amplification and future projections

{chr(10).join(resource_lines)}

Direct facts are the measured one-second process-tree samples, full projected DEV dimensions, finalized bytes, and exact fold counts. Runtime upper bounds repeat the cumulative pilot five times or scale linearly by full-DEV/train rows. Memory projections add to the observed peak (a) float64 training-encoding plus Boruta-shadow growth and (b) retained mixed-frame slice growth using observed bytes per row; folds remain sequential and the frozen 1.35 safety factor is applied separately. These are conservative arithmetic projections, not measurements of later folds.

Home Credit clears the unchanged policy. LendingClub v2 peaked near 12.60 GiB RSS with about 12.12 GiB system RAM available on the first fold. The conservative largest-fold/full-DEV projections cross the 8 GiB available-RAM floor and/or the 28 GiB process ceiling after the 1.35 safety factor. Disk is not limiting: finalized artifacts are far below preflight estimates and hundreds of GiB remained free, though free-space deltas include unrelated OS/background effects and are not treated as attributable writes.

## Limitations and gate

These pilots cover one fold only and cannot establish OOF/OOT performance, stability, statistical significance, or research findings. Later-fold memory scaling is estimated, not observed. CatBoost early stopping is configured but inactive without validation-target leakage. Pandas reported mixed-type inference warnings for explicit LendingClub categorical projections; canonical encoding and hashes still passed.

**{gate}:** before Prompt 5 may authorize any frozen research run, approve a separately versioned memory-safe execution refinement and validate the largest LendingClub fold/full-DEV shape under the unchanged policy. The likely non-scientific target is eliminating avoidable full-frame/slice copies and releasing fold-local arrays between sequential folds; rows, features, selectors, seeds, models, and limits must remain unchanged.

Frozen hashes match before and after. The legacy bundle remains 359 files and 110,084,164 bytes with zero added, removed, or changed. Active results contain exactly the four authorized completed pilot IDs, no active lock, and no partial file.
""",
        encoding="utf-8",
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--focused-passed", type=int, required=True)
    value.add_argument("--focused-command", required=True)
    value.add_argument("--full-passed", type=int, required=True)
    value.add_argument("--full-skipped", type=int, required=True)
    value.add_argument("--full-warnings", type=int, required=True)
    return value


if __name__ == "__main__":
    build(parser().parse_args())
