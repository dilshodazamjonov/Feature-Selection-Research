"""Build the Prompt 6 cross-dataset voting inference and evidence package.

Usage
-----
    Set-Location "D:\\python projects\\Research"
    .\\.venv\\Scripts\\python.exe scripts\\build_voting_inference_evidence.py

The command authenticates every frozen input and completed run before computing
anything, recomputes all reported metrics from saved applicant-level
predictions, executes only the predeclared comparison family, and writes solely
into its own versioned package root.  It never fits a model, runs a selector,
regenerates a voting ranking, or writes into a completed run directory or the
frozen legacy bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from credit_risk_fs.analysis.voting_inference import figures as figure_builders  # noqa: E402
from credit_risk_fs.analysis.voting_inference.alignment import (  # noqa: E402
    align_predictions,
    dev_oot_disjoint_audit,
    frozen_leakage_exclusions,
    leakage_audit_row,
    load_prediction_frame,
)
from credit_risk_fs.analysis.voting_inference.config import (  # noqa: E402
    AnalysisConfig,
    authenticate_frozen_inputs,
    load_analysis_config,
    read_json,
)
from credit_risk_fs.analysis.voting_inference.inventory import (  # noqa: E402
    RunRecord,
    authenticate_prompt_05_completion,
    build_input_inventory,
    discover_runs,
    fold_candidate_universe_counts,
    read_final_selection,
    read_fold_candidate_pool,
    read_fold_selection,
)
from credit_risk_fs.analysis.voting_inference.metrics import (  # noqa: E402
    lift_at_10_audit,
    recompute_discrimination,
)
from credit_risk_fs.analysis.voting_inference.paired import (  # noqa: E402
    BOOTSTRAP_METRICS,
    apply_holm_families,
    assert_bootstrap_equivalence,
    fast_paired_stratified_bootstrap,
    recover_predeclared_family,
    run_paired_delong,
)
from credit_risk_fs.analysis.voting_inference.psi import (  # noqa: E402
    feature_psi_record,
    score_psi_from_predictions,
    summarise_feature_psi,
)
from credit_risk_fs.analysis.voting_inference.resources import (  # noqa: E402
    runtime_resource_row,
    stage_breakdown_rows,
)
from credit_risk_fs.analysis.voting_inference.stability import (  # noqa: E402
    fold_selection_inventory_rows,
    frozen_kuncheva_reference,
    pairwise_fold_stability,
    selection_frequency,
    summarise_pairwise_stability,
)
from credit_risk_fs.experiments.atomic_io import (  # noqa: E402
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)
from credit_risk_fs.experiments.result_paths import reject_historical_write  # noqa: E402
from credit_risk_fs.utils.hashing import sha256_file  # noqa: E402

SUCCESS_MARKER = "PROMPT_06_VOTING_INFERENCE_EVIDENCE_PACKAGE_PASS"
_START = time.perf_counter()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{stamp} | {time.perf_counter() - _START:8.1f}s | {message}", flush=True)


class Blocker(RuntimeError):
    """Raised when scientific validity is compromised."""


class NeedsUserAction(RuntimeError):
    """Raised when an external decision or missing input cannot be resolved."""


# ---------------------------------------------------------------------------
# Phase A helpers
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:  # pragma: no cover - git is expected to be present
        return ""


def repository_state() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name, module_name in (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("catboost", "catboost"),
        ("statsmodels", "statsmodels"),
        ("joblib", "joblib"),
        ("pyarrow", "pyarrow"),
        ("psutil", "psutil"),
        ("matplotlib", "matplotlib"),
    ):
        try:
            module = __import__(module_name)
        except ImportError:
            packages[name] = None
        else:
            packages[name] = str(getattr(module, "__version__", "unknown"))
    return {
        "schema_version": "prompt_06_repository_state_v1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_status_short_branch": _git("status", "--short", "--branch"),
        "git_head": _git("rev-parse", "HEAD"),
        "git_log_head": _git("log", "-1", "--oneline"),
        "git_tags_at_head": [
            tag for tag in _git("tag", "--points-at", "HEAD").splitlines() if tag
        ],
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "package_versions": packages,
        "statsmodels_available": packages["statsmodels"] is not None,
        "statsmodels_required_by_analysis": False,
    }


def preservation_audit(config: AnalysisConfig, runs: Sequence[RunRecord]) -> dict[str, Any]:
    failures: list[str] = []
    preservation = config.preservation
    legacy_root = Path(str(config.payload["paths"]["legacy_results_root"]))
    legacy_files: list[Path] = []
    legacy_bytes = 0
    if legacy_root.is_dir():
        legacy_files = [path for path in legacy_root.rglob("*") if path.is_file()]
        legacy_bytes = sum(path.stat().st_size for path in legacy_files)
        if len(legacy_files) != int(preservation["legacy_expected_file_count"]):
            failures.append(
                f"legacy file count {len(legacy_files)} != "
                f"{preservation['legacy_expected_file_count']}"
            )
        if legacy_bytes != int(preservation["legacy_expected_total_bytes"]):
            failures.append(
                f"legacy byte total {legacy_bytes} != "
                f"{preservation['legacy_expected_total_bytes']}"
            )
    manifest_path = config.repository_root / str(
        config.payload["paths"]["legacy_manifest"]
    )
    manifest_sha = sha256_file(manifest_path) if manifest_path.is_file() else None

    package_root = config.package_root
    audit_root = config.audit_root
    for label, path in (("package_root", package_root), ("audit_root", audit_root)):
        try:
            reject_historical_write(path)
        except Exception as error:  # pragma: no cover - defensive
            failures.append(f"{label} rejected by the write barrier: {error}")
        if legacy_root.is_dir() and path.resolve().is_relative_to(legacy_root.resolve()):
            failures.append(f"{label} resolves inside the frozen legacy bundle")
        for run in runs:
            if path.resolve().is_relative_to(run.directory.resolve()):
                failures.append(f"{label} resolves inside completed run {run.run_id}")

    # Listed at audit time, so entries this same run has already published (the
    # input inventory and the figures directory) appear here by design.
    package_entries = (
        sorted(str(path.name) for path in package_root.glob("*")) if package_root.is_dir() else []
    )
    run_ids = [run.run_id for run in runs]
    duplicates = sorted({value for value in run_ids if run_ids.count(value) > 1})
    if duplicates:
        failures.append(f"duplicate run ids: {duplicates}")

    return {
        "schema_version": "prompt_06_preservation_audit_v1",
        "legacy_results_root": str(legacy_root),
        "legacy_root_present": legacy_root.is_dir(),
        "legacy_file_count": len(legacy_files),
        "legacy_total_bytes": legacy_bytes,
        "legacy_expected_file_count": int(preservation["legacy_expected_file_count"]),
        "legacy_expected_total_bytes": int(preservation["legacy_expected_total_bytes"]),
        "legacy_manifest_path": str(manifest_path.relative_to(config.repository_root)).replace("\\", "/"),
        "legacy_manifest_sha256": manifest_sha,
        "legacy_unchanged": not failures,
        "completed_run_directories_treated_as_immutable": True,
        "duplicate_run_ids": duplicates,
        "package_root": _relative(package_root, config.repository_root),
        "audit_root": _relative(audit_root, config.repository_root),
        "package_root_entries_at_audit_time": package_entries,
        "analysis_writes_only_inside_package_and_audit_roots": True,
        "failures": failures,
        "status": "PASS" if not failures else "BLOCKED",
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# ---------------------------------------------------------------------------
# Phase B/C/D: per-run prediction evidence
# ---------------------------------------------------------------------------


def _cell_runs(runs: Sequence[RunRecord], dataset: str, model: str) -> list[RunRecord]:
    order = {"reference": 0, "voting_k100": 1, "voting_k200": 2, "voting_k300": 3}
    subset = [run for run in runs if run.dataset == dataset and run.model == model]
    return sorted(subset, key=lambda run: order.get(run.configuration, 99))


def process_prediction_evidence(
    config: AnalysisConfig, runs: Sequence[RunRecord]
) -> dict[str, Any]:
    """Phase B, C, and D over every dataset-model cell, one cell at a time."""

    exclusions = frozen_leakage_exclusions(config)
    alignment_rows: list[dict[str, Any]] = []
    disjoint_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    prediction_inventory: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    lift_rows: list[dict[str, Any]] = []
    psi_summary_rows: list[dict[str, Any]] = []
    psi_bin_rows: list[pd.DataFrame] = []
    aligned_oot: dict[tuple[str, str], pd.DataFrame] = {}

    for dataset in config.expected["datasets"]:
        for model in config.expected["models"]:
            cell = _cell_runs(runs, dataset, model)
            if not cell:
                raise Blocker(f"no runs registered for cell {dataset}/{model}")
            reference = next((run for run in cell if run.is_reference()), None)
            if reference is None:
                raise Blocker(f"cell {dataset}/{model} has no reference run")
            log(f"cell {dataset}/{model}: loading {len(cell)} runs")
            frames = {
                run.run_id: {
                    split: load_prediction_frame(run, split=split)
                    for split in ("DEV", "OOT")
                }
                for run in cell
            }
            for run in cell:
                dev = frames[run.run_id]["DEV"]
                oot = frames[run.run_id]["OOT"]
                disjoint_rows.append(dev_oot_disjoint_audit(dev, oot))
                leakage_rows.append(
                    leakage_audit_row(
                        config, run, leakage_exclusions=exclusions[run.dataset]
                    )
                )
                for split_frame in (dev, oot):
                    prediction_inventory.append(
                        {
                            "run_id": run.run_id,
                            "dataset": run.dataset,
                            "model": run.model,
                            "configuration": run.configuration,
                            "split": split_frame.split,
                            "path": _relative(split_frame.path, config.repository_root),
                            "sha256": sha256_file(split_frame.path),
                            "row_count": split_frame.row_count,
                            "unique_identity_count": int(
                                split_frame.frame["stable_row_id"].nunique()
                            ),
                            "positive_count": split_frame.positive_count,
                            "positive_rate": split_frame.positive_count
                            / max(split_frame.row_count, 1),
                            "score_minimum": float(split_frame.frame["score"].min()),
                            "score_maximum": float(split_frame.frame["score"].max()),
                            "score_null_count": int(
                                split_frame.frame["score"].isna().sum()
                            ),
                            "declared_row_count": split_frame.metadata.get("row_count"),
                            "declared_positive_count": split_frame.metadata.get(
                                "positive_target_count"
                            ),
                            "declared_coverage_type": split_frame.metadata.get(
                                "coverage_type"
                            ),
                            "declared_identity_target_sha256": split_frame.metadata.get(
                                "identity_target_sha256"
                            ),
                            "declared_artifact_sha256": (
                                split_frame.metadata.get("prediction_artifact") or {}
                            ).get("sha256"),
                        }
                    )

                # -- Phase C: recomputed discrimination -------------------
                for split_frame in (dev, oot):
                    values = recompute_discrimination(
                        split_frame.frame["target"],
                        split_frame.frame["score"],
                        split_frame.frame["stable_row_id"],
                    )
                    metric_rows.append(
                        {
                            "run_id": run.run_id,
                            "dataset": run.dataset,
                            "model": run.model,
                            "configuration": run.configuration,
                            "designation": run.designation,
                            "candidate_pool_budget": run.candidate_pool_budget,
                            "comparison_family": run.comparison_family,
                            "split": split_frame.split,
                            "row_count": split_frame.row_count,
                            "positive_count": split_frame.positive_count,
                            **values,
                            "gini_from_auc_consistent": bool(
                                abs(values["gini"] - (2 * values["auc"] - 1)) <= 1e-12
                            ),
                            "prediction_sha256": sha256_file(split_frame.path),
                        }
                    )
                    audit = lift_at_10_audit(
                        split_frame.frame["target"],
                        split_frame.frame["score"],
                        split_frame.frame["stable_row_id"],
                    )
                    lift_rows.append(
                        {
                            "run_id": run.run_id,
                            "dataset": run.dataset,
                            "model": run.model,
                            "configuration": run.configuration,
                            "split": split_frame.split,
                            **audit,
                        }
                    )

                # -- Phase D: score PSI -----------------------------------
                if dev.metadata.get("coverage_type") != "complete_five_fold_dev_oof":
                    raise Blocker(
                        f"{run.run_id}: DEV artifact is not complete out-of-fold "
                        f"coverage ({dev.metadata.get('coverage_type')!r}); score PSI "
                        "would use in-sample DEV scores"
                    )
                psi = score_psi_from_predictions(
                    dev.frame["score"], oot.frame["score"]
                )
                psi_summary_rows.append(
                    {
                        "run_id": run.run_id,
                        "dataset": run.dataset,
                        "model": run.model,
                        "configuration": run.configuration,
                        "candidate_pool_budget": run.candidate_pool_budget,
                        "score_psi": psi.psi,
                        "reference_scope": psi.definition["reference_scope"],
                        "comparison_scope": psi.definition["comparison_scope"],
                        "dev_oof_row_count": dev.row_count,
                        "oot_row_count": oot.row_count,
                        "requested_bin_count": psi.definition["requested_bin_count"],
                        "effective_bin_count": psi.definition["effective_bin_count"],
                        "binning_method": psi.definition["binning_method"],
                        "duplicate_edge_policy": psi.definition["duplicate_edge_policy"],
                        "out_of_range_policy": psi.definition["out_of_range_policy"],
                        "missing_value_policy": psi.definition["missing_value_policy"],
                        "smoothing_epsilon": psi.definition["smoothing_epsilon"],
                        "psi_implementation_version": psi.definition[
                            "psi_implementation_version"
                        ],
                        "bins_defined_on_dev_only": True,
                        "bins_applied_unchanged_to_oot": True,
                        "score_psi_available": True,
                    }
                )
                bins = psi.bins.copy()
                bins.insert(0, "run_id", run.run_id)
                bins.insert(1, "dataset", run.dataset)
                bins.insert(2, "model", run.model)
                bins.insert(3, "configuration", run.configuration)
                psi_bin_rows.append(bins)

            # -- Phase B: paired alignment against the cell reference -----
            for run in cell:
                if run.is_reference():
                    continue
                for split in ("DEV", "OOT"):
                    aligned, audit = align_predictions(
                        frames[reference.run_id][split], frames[run.run_id][split]
                    )
                    audit.update(
                        {
                            "model": model,
                            "configuration": run.configuration,
                            "candidate_pool_budget": run.candidate_pool_budget,
                            "comparison_family": run.comparison_family,
                        }
                    )
                    alignment_rows.append(audit)
                    if split == "OOT" and audit["decision"] == "aligned":
                        aligned_oot[(reference.run_id, run.run_id)] = aligned
            del frames

    return {
        "alignment": pd.DataFrame(alignment_rows),
        "dev_oot_disjoint": pd.DataFrame(disjoint_rows),
        "leakage": pd.DataFrame(leakage_rows),
        "prediction_inventory": pd.DataFrame(prediction_inventory),
        "run_level_metrics": pd.DataFrame(metric_rows),
        "lift10_audit": pd.DataFrame(lift_rows),
        "score_psi_summary": pd.DataFrame(psi_summary_rows),
        "score_psi_bins": pd.concat(psi_bin_rows, ignore_index=True)
        if psi_bin_rows
        else pd.DataFrame(),
        "aligned_oot": aligned_oot,
    }


# ---------------------------------------------------------------------------
# Phase E: feature PSI
# ---------------------------------------------------------------------------


def compute_feature_psi(
    config: AnalysisConfig, runs: Sequence[RunRecord]
) -> dict[str, pd.DataFrame]:
    from credit_risk_fs.pipelines.common import (
        prepare_voting_pilot_dev_data,
        prepare_voting_research_oot_data,
    )

    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    definition_rows: list[dict[str, Any]] = []
    references = [float(value) for value in config.payload["descriptive_psi_references"]]
    chunk_rows = int(config.payload["execution"]["csv_chunk_rows"])

    for dataset in config.expected["datasets"]:
        dataset_runs = [run for run in runs if run.dataset == dataset]
        selections = {run.run_id: read_final_selection(run) for run in dataset_runs}
        union = sorted({feature for values in selections.values() for feature in values})
        log(f"feature PSI {dataset}: loading DEV projection of {len(union)} features")
        dev = prepare_voting_pilot_dev_data(
            config.repository_root,
            dataset=dataset,
            csv_chunk_rows=chunk_rows,
            projected_candidate_features=union,
        )
        dev_frame = dev.X
        log(f"feature PSI {dataset}: loading OOT projection of {len(union)} features")
        oot = prepare_voting_research_oot_data(
            config.repository_root,
            dataset=dataset,
            projected_candidate_features=union,
            csv_chunk_rows=chunk_rows,
        )
        oot_frame = oot.X
        cache: dict[str, dict[str, Any]] = {}
        for feature in union:
            record, _ = feature_psi_record(
                feature=feature,
                dev_values=dev_frame[feature],
                oot_values=oot_frame[feature],
            )
            cache[feature] = record
            definition_rows.append(
                {
                    "dataset": dataset,
                    "feature": feature,
                    "feature_type": record["feature_type"],
                    "frozen_numeric_implementation": "credit_risk_fs.evaluation.drift.calculate_psi",
                    "frozen_numeric_available": record["psi_frozen_numeric_available"],
                    "frozen_numeric_missing_policy": "nonfinite_dropped",
                    "type_aware_implementation": (
                        "credit_risk_fs.analysis.voting_inference.psi.type_aware_feature_psi"
                    ),
                    "type_aware_available": record["psi_type_aware_available"],
                    "bin_definition_source": record["bin_definition_source"],
                    "effective_bin_count": record["effective_bin_count"],
                    "missing_handling": record["missing_handling"],
                    "unseen_level_handling": record["unseen_level_handling"],
                    "unseen_oot_level_count": record["unseen_oot_level_count"],
                    "smoothing_epsilon": record["smoothing_epsilon"],
                    "reference_distribution_scope": "DEV",
                    "comparison_distribution_scope": "OOT",
                    "bins_defined_on_dev_only": True,
                    "encoded_feature_relationship": record[
                        "encoded_feature_relationship"
                    ],
                }
            )
        del dev, oot, dev_frame, oot_frame

        for run in dataset_runs:
            run_rows = []
            for feature in selections[run.run_id]:
                record = dict(cache[feature])
                record.update(
                    {
                        "run_id": run.run_id,
                        "dataset": run.dataset,
                        "model": run.model,
                        "configuration": run.configuration,
                        "candidate_pool_budget": run.candidate_pool_budget,
                        "fold_or_final_run_context": "final_full_dev_selection",
                    }
                )
                run_rows.append(record)
            frame = pd.DataFrame(run_rows)
            long_rows.extend(run_rows)
            summary_rows.append(
                {
                    "run_id": run.run_id,
                    "dataset": run.dataset,
                    "model": run.model,
                    "configuration": run.configuration,
                    "candidate_pool_budget": run.candidate_pool_budget,
                    "fold_or_final_run_context": "final_full_dev_selection",
                    **summarise_feature_psi(frame, references=references),
                }
            )
    return {
        "feature_psi_long": pd.DataFrame(long_rows),
        "feature_psi_summary": pd.DataFrame(summary_rows),
        "feature_psi_definition_audit": pd.DataFrame(definition_rows),
    }


# ---------------------------------------------------------------------------
# Phase F: fold-level selection stability
# ---------------------------------------------------------------------------


def compute_stability(
    config: AnalysisConfig, runs: Sequence[RunRecord]
) -> dict[str, Any]:
    fold_count = int(config.expected["dev_folds_per_run"])
    inventory_rows: list[dict[str, Any]] = []
    pairwise_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    frequency_frames: list[pd.DataFrame] = []
    universe_conflicts: list[str] = []

    for run in runs:
        universe = config.dataset_universe_size(run.dataset)
        selections = {
            fold: read_fold_selection(run, fold)
            for fold in range(1, fold_count + 1)
            if run.fold_selection(fold).is_file()
        }
        pools = {
            fold: read_fold_candidate_pool(run, fold)
            for fold in range(1, fold_count + 1)
            if run.fold_candidates(fold).is_file()
        }
        declared = {
            fold: fold_candidate_universe_counts(run, fold)
            for fold in range(1, fold_count + 1)
        }
        for fold, values in declared.items():
            if values and values != {universe}:
                universe_conflicts.append(
                    f"{run.run_id} fold {fold} declares universe {sorted(values)}"
                )
        inventory_rows.extend(
            fold_selection_inventory_rows(
                run_id=run.run_id,
                dataset=run.dataset,
                model=run.model,
                configuration=run.configuration,
                expected_budget=config.final_budget(run.model),
                universe_size=universe,
                fold_selections=selections,
                fold_candidate_pools=pools,
                declared_universe_counts=declared,
                expected_fold_count=fold_count,
            )
        )
        if len(selections) != fold_count:
            raise Blocker(
                f"{run.run_id}: {len(selections)} of {fold_count} fold selections present"
            )
        pairwise = pairwise_fold_stability(selections, universe_size=universe)
        pairwise.insert(0, "run_id", run.run_id)
        pairwise.insert(1, "dataset", run.dataset)
        pairwise.insert(2, "model", run.model)
        pairwise.insert(3, "configuration", run.configuration)
        pairwise_frames.append(pairwise)
        summary = summarise_pairwise_stability(pairwise)
        frozen_reference = frozen_kuncheva_reference(selections, universe_size=universe)
        summary_rows.append(
            {
                "run_id": run.run_id,
                "dataset": run.dataset,
                "model": run.model,
                "configuration": run.configuration,
                "candidate_pool_budget": run.candidate_pool_budget,
                "fold_count": len(selections),
                "authenticated_universe_size": universe,
                "universe_definition": "frozen_per_dataset_leakage_safe_candidate_universe",
                **summary,
                "frozen_implementation_kuncheva_mean": frozen_reference,
                "frozen_implementation_agrees": bool(
                    frozen_reference is None
                    and summary["kuncheva_mean"] is None
                )
                or bool(
                    frozen_reference is not None
                    and summary["kuncheva_mean"] is not None
                    and abs(frozen_reference - summary["kuncheva_mean"]) <= 1e-12
                ),
            }
        )
        frequency = selection_frequency(selections)
        if not frequency.empty:
            frequency.insert(0, "run_id", run.run_id)
            frequency.insert(1, "dataset", run.dataset)
            frequency.insert(2, "model", run.model)
            frequency.insert(3, "configuration", run.configuration)
            frequency_frames.append(frequency)

    if universe_conflicts:
        raise NeedsUserAction(
            "the saved runs do not declare one consistent stability universe: "
            + "; ".join(universe_conflicts)
        )
    return {
        "fold_selection_inventory": pd.DataFrame(inventory_rows),
        "stability_pairwise": pd.concat(pairwise_frames, ignore_index=True),
        "stability_summary": pd.DataFrame(summary_rows),
        "fold_selection_frequency": pd.concat(frequency_frames, ignore_index=True)
        if frequency_frames
        else pd.DataFrame(),
    }


# ---------------------------------------------------------------------------
# Phase H: predeclared paired inference
# ---------------------------------------------------------------------------


def recover_family(
    config: AnalysisConfig, runs: Sequence[RunRecord]
) -> tuple[list[Any], dict[str, Any]]:
    protocol_path = config.repository_root / str(
        config.payload["frozen_inputs"]["voting_protocol"]["path"]
    )
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    inference = protocol.get("statistical_inference")
    if not inference or "primary_families" not in inference:
        raise NeedsUserAction(
            "the frozen voting protocol declares no predeclared comparison family"
        )
    lookup = {
        (run.dataset, run.model, run.configuration): run.run_id for run in runs
    }
    comparisons = recover_predeclared_family(protocol, lookup)
    settings = config.inference
    bootstrap = settings["bootstrap"]
    specification = {
        "schema_version": "prompt_06_predeclared_comparison_family_v1",
        "recovered_from": _relative(protocol_path, config.repository_root),
        "recovered_from_sha256": sha256_file(protocol_path),
        "protocol_name": protocol["protocol_name"],
        "protocol_version": protocol["version"],
        "recovered_before_any_p_value_was_computed": True,
        "constructed_after_viewing_oot_results": False,
        "direction_convention": settings["direction_convention"],
        "split": settings["split"],
        "primary_candidate_pool": int(protocol["primary_candidate_pool"]),
        "sensitivity_candidate_pools": [
            int(value) for value in protocol["sensitivity_candidate_pools"]
        ],
        "holm_scope": str(inference["holm_scope"]),
        "holm_alpha": float(settings["holm"]["alpha"]),
        "holm_pooling_across_families": False,
        "family_count": len({comparison.family for comparison in comparisons}),
        "comparison_count": len(comparisons),
        "families": sorted({comparison.family for comparison in comparisons}),
        "bootstrap": {
            "type": bootstrap["type"],
            "attempted_repetitions": int(bootstrap["attempted_repetitions"]),
            "minimum_valid_repetitions": int(bootstrap["minimum_valid_repetitions"]),
            "seed": int(bootstrap["seed"]),
            "confidence_interval": bootstrap["confidence_interval"],
            "differences": list(bootstrap["differences"]),
            "failure_policy": bootstrap["failure_policy"],
        },
        "comparisons": [
            {
                "family": comparison.family,
                "dataset": comparison.dataset,
                "model": comparison.model,
                "reference_run_id": comparison.reference_run_id,
                "comparator_run_id": comparison.comparator_run_id,
                "candidate_pool_budget": comparison.candidate_pool_budget,
                "designation": comparison.designation,
                "metric_tested": "roc_auc",
                "test": "two_sided_paired_delong_on_identical_oot_rows",
                "confidence_level": 0.95,
                "bootstrap_count": int(bootstrap["attempted_repetitions"]),
                "bootstrap_seed": int(bootstrap["seed"]),
                "multiplicity_family": comparison.family,
                "holm_scope": str(inference["holm_scope"]),
                "comparison_label": comparison.label,
            }
            for comparison in comparisons
        ],
        "excluded_comparisons": [],
        "descriptive_only": list(inference["descriptive_only"]),
    }
    return comparisons, specification


def run_paired_inference(
    config: AnalysisConfig,
    comparisons: Sequence[Any],
    aligned_oot: Mapping[tuple[str, str], pd.DataFrame],
    *,
    cache_directory: Path,
) -> dict[str, Any]:
    bootstrap_settings = config.inference["bootstrap"]
    repetitions = int(bootstrap_settings["attempted_repetitions"])
    minimum_valid = int(bootstrap_settings["minimum_valid_repetitions"])
    seed = int(bootstrap_settings["seed"])
    cache_directory.mkdir(parents=True, exist_ok=True)

    delong_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    replicate_manifest: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []

    for comparison in comparisons:
        key = (comparison.reference_run_id, comparison.comparator_run_id)
        if key not in aligned_oot:
            raise Blocker(
                f"{comparison.label} ({comparison.family}) has no aligned OOT rows"
            )
        aligned = aligned_oot[key].rename(
            columns={"score_reference": "score_a_reference"}
        )
        paired = pd.DataFrame(
            {
                "stable_row_id": aligned["stable_row_id"],
                "target": aligned["target"],
                "score_a": aligned["score_comparator"],
                "score_b": aligned["score_a_reference"],
            }
        )
        delong_input = pd.DataFrame(
            {
                "target": paired["target"],
                "score_comparator": paired["score_a"],
                "score_reference": paired["score_b"],
            }
        )
        delong_rows.append(run_paired_delong(delong_input, comparison))

        cache_path = cache_directory / f"bootstrap_{comparison.family}_k{comparison.candidate_pool_budget}.json"
        input_fingerprint = _fingerprint(paired)
        if cache_path.is_file():
            cached = read_json(cache_path)
            if (
                cached.get("input_fingerprint") == input_fingerprint
                and int(cached.get("attempted_repetitions", -1)) == repetitions
                and int(cached.get("seed", -1)) == seed
            ):
                log(f"bootstrap {comparison.family} K={comparison.candidate_pool_budget}: reusing cache")
                result = cached["result"]
                equivalence = cached["equivalence"]
            else:
                result, equivalence = _bootstrap_with_equivalence(
                    paired, comparison, repetitions, seed, minimum_valid
                )
                _write_bootstrap_cache(
                    cache_path, input_fingerprint, repetitions, seed, result, equivalence
                )
        else:
            result, equivalence = _bootstrap_with_equivalence(
                paired, comparison, repetitions, seed, minimum_valid
            )
            _write_bootstrap_cache(
                cache_path, input_fingerprint, repetitions, seed, result, equivalence
            )

        equivalence_rows.append(
            {
                "family": comparison.family,
                "comparison_label": comparison.label,
                **{
                    field: equivalence[field]
                    for field in (
                        "repetitions",
                        "seed",
                        "valid_repetitions_frozen",
                        "valid_repetitions_fast",
                        "maximum_absolute_difference",
                        "tolerance",
                        "equivalent",
                    )
                },
            }
        )
        row: dict[str, Any] = {
            "family": comparison.family,
            "dataset": comparison.dataset,
            "model": comparison.model,
            "comparison_label": comparison.label,
            "designation": comparison.designation,
            "candidate_pool_budget": comparison.candidate_pool_budget,
            "reference_run_id": comparison.reference_run_id,
            "comparator_run_id": comparison.comparator_run_id,
            "direction_convention": "comparator_minus_reference",
            "bootstrap_type": result["stratification"],
            "attempted_repetitions": result["attempted_repetitions"],
            "valid_repetitions": result["valid_repetitions"],
            "failed_repetitions": result["failed_repetitions"],
            "minimum_valid_repetitions": result["minimum_valid_repetitions"],
            "seed": result["seed"],
            "confidence_interval": bootstrap_settings["confidence_interval"],
            "aligned_row_count": int(len(paired)),
        }
        for metric in BOOTSTRAP_METRICS:
            block = result["metrics"][metric]
            # Named distinctly from the DeLong AUC delta so the joined inference
            # table never has to disambiguate two same-named columns.
            row[f"{metric}_delta_observed_comparator_minus_reference"] = block[
                "observed_difference_a_minus_b"
            ]
            row[f"{metric}_delta_ci95_lower"] = block["ci95_percentile_lower"]
            row[f"{metric}_delta_ci95_upper"] = block["ci95_percentile_upper"]
            row[f"{metric}_interval_valid"] = block["interval_valid"]
            row[f"{metric}_replicate_mean"] = block.get("replicate_mean")
            row[f"{metric}_replicate_std"] = block.get("replicate_std")
        bootstrap_rows.append(row)
        replicate_manifest.append(
            {
                "family": comparison.family,
                "comparison_label": comparison.label,
                "cache_artifact": _relative(cache_path, config.repository_root),
                "cache_sha256": sha256_file(cache_path),
                "input_fingerprint": input_fingerprint,
                "attempted_repetitions": result["attempted_repetitions"],
                "valid_repetitions": result["valid_repetitions"],
                "failed_repetitions": result["failed_repetitions"],
                "seed": result["seed"],
                "replicate_summary_retained": [
                    {
                        "metric": metric,
                        "replicate_mean": result["metrics"][metric].get("replicate_mean"),
                        "replicate_std": result["metrics"][metric].get("replicate_std"),
                        "ci95_percentile_lower": result["metrics"][metric][
                            "ci95_percentile_lower"
                        ],
                        "ci95_percentile_upper": result["metrics"][metric][
                            "ci95_percentile_upper"
                        ],
                    }
                    for metric in BOOTSTRAP_METRICS
                ],
                "unfavourable_replicates_discarded": False,
                "replication_count_changed_after_viewing_results": False,
            }
        )

    holm = apply_holm_families(delong_rows, alpha=float(config.inference["holm"]["alpha"]))
    return {
        "paired_delong_results": pd.DataFrame(delong_rows),
        "paired_bootstrap_results": pd.DataFrame(bootstrap_rows),
        "holm_adjustment_audit": holm,
        "bootstrap_equivalence": pd.DataFrame(equivalence_rows),
        "bootstrap_replicate_manifest": {
            "schema_version": "prompt_06_bootstrap_replicate_manifest_v1",
            "design": {
                "type": bootstrap_settings["type"],
                "attempted_repetitions": repetitions,
                "minimum_valid_repetitions": minimum_valid,
                "seed": seed,
                "confidence_interval": bootstrap_settings["confidence_interval"],
                "differences": list(bootstrap_settings["differences"]),
                "failure_policy": bootstrap_settings["failure_policy"],
                "paired_indices_shared_between_methods": True,
                "resampled_within_target_class": True,
            },
            "implementation": {
                "executed": "credit_risk_fs.analysis.voting_inference.paired.fast_paired_stratified_bootstrap",
                "frozen_reference": "credit_risk_fs.evaluation.paired_inference.paired_stratified_bootstrap",
                "equivalence_verified_per_comparison": True,
                "equivalence_tolerance": 0.0,
            },
            "comparisons": replicate_manifest,
        },
    }


def _fingerprint(paired: pd.DataFrame) -> str:
    from credit_risk_fs.utils.hashing import sha256_text

    parts = [
        str(len(paired)),
        str(int(paired["target"].sum())),
        f"{float(paired['score_a'].sum()):.12e}",
        f"{float(paired['score_b'].sum()):.12e}",
        str(paired["stable_row_id"].iloc[0]),
        str(paired["stable_row_id"].iloc[-1]),
    ]
    return sha256_text("|".join(parts))


def _bootstrap_with_equivalence(
    paired: pd.DataFrame,
    comparison: Any,
    repetitions: int,
    seed: int,
    minimum_valid: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    log(
        f"bootstrap {comparison.family} K={comparison.candidate_pool_budget}: "
        f"verifying accelerated equivalence on the real aligned rows"
    )
    equivalence = assert_bootstrap_equivalence(
        paired, repetitions=8, seed=seed, minimum_valid=1, tolerance=0.0
    )
    if not equivalence["equivalent"]:
        raise Blocker(
            f"{comparison.label}: accelerated bootstrap disagrees with the frozen "
            f"implementation (max abs diff {equivalence['maximum_absolute_difference']})"
        )
    log(
        f"bootstrap {comparison.family} K={comparison.candidate_pool_budget}: "
        f"running {repetitions} attempts"
    )
    result = fast_paired_stratified_bootstrap(
        paired, repetitions=repetitions, seed=seed, minimum_valid=minimum_valid
    )
    return result, equivalence


def _write_bootstrap_cache(
    path: Path,
    fingerprint: str,
    repetitions: int,
    seed: int,
    result: Mapping[str, Any],
    equivalence: Mapping[str, Any],
) -> None:
    write_json_atomic(
        path,
        {
            "schema_version": "prompt_06_bootstrap_cache_v1",
            "input_fingerprint": fingerprint,
            "attempted_repetitions": repetitions,
            "seed": seed,
            "result": result,
            "equivalence": equivalence,
        },
    )


# ---------------------------------------------------------------------------
# Phase I: budget curve and cross-dataset evidence tables
# ---------------------------------------------------------------------------


def build_evidence_tables(
    config: AnalysisConfig,
    *,
    run_metrics: pd.DataFrame,
    score_psi: pd.DataFrame,
    feature_psi_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    runtime: pd.DataFrame,
    inventory: pd.DataFrame,
    delong: pd.DataFrame,
    bootstrap: pd.DataFrame,
    holm: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    oot = run_metrics.loc[run_metrics["split"] == "OOT"].set_index("run_id")
    dev = run_metrics.loc[run_metrics["split"] == "DEV_OOF"].set_index("run_id")
    psi = score_psi.set_index("run_id")
    feature = feature_psi_summary.set_index("run_id")
    stability = stability_summary.set_index("run_id")
    resources = runtime.set_index("run_id")
    counts = inventory.set_index("run_id")

    rows: list[dict[str, Any]] = []
    for dataset in config.expected["datasets"]:
        for model in config.expected["models"]:
            cell = counts.loc[
                (counts["dataset"] == dataset) & (counts["model"] == model)
            ]
            reference_ids = cell.loc[cell["designation"] == "reference"].index.tolist()
            if len(reference_ids) != 1:
                raise Blocker(
                    f"cell {dataset}/{model} resolves {len(reference_ids)} references"
                )
            reference_id = reference_ids[0]
            for run_id in cell.index:
                row = {
                    "dataset": dataset,
                    "model": model,
                    "run_id": run_id,
                    "configuration": counts.at[run_id, "configuration"],
                    "designation": counts.at[run_id, "designation"],
                    "candidate_pool_budget": counts.at[run_id, "candidate_pool_budget"],
                    "reference_run_id": reference_id,
                    "is_reference": run_id == reference_id,
                    "selected_feature_count": counts.at[
                        run_id, "final_selected_feature_count"
                    ],
                    "dev_oof_row_count": dev.at[run_id, "row_count"],
                    "oot_row_count": oot.at[run_id, "row_count"],
                    "dev_oof_auc": dev.at[run_id, "auc"],
                    "dev_oof_gini": dev.at[run_id, "gini"],
                    "dev_oof_ks": dev.at[run_id, "ks"],
                    "dev_oof_lift_at_10": dev.at[run_id, "lift_at_10"],
                    "oot_auc": oot.at[run_id, "auc"],
                    "oot_gini": oot.at[run_id, "gini"],
                    "oot_ks": oot.at[run_id, "ks"],
                    "oot_lift_at_10": oot.at[run_id, "lift_at_10"],
                    "score_psi": psi.at[run_id, "score_psi"],
                    "feature_psi_frozen_numeric_mean": feature.at[
                        run_id, "frozen_numeric_mean"
                    ],
                    "feature_psi_frozen_numeric_max": feature.at[
                        run_id, "frozen_numeric_max"
                    ],
                    "feature_psi_frozen_numeric_unavailable_count": feature.at[
                        run_id, "frozen_numeric_unavailable_count"
                    ],
                    "feature_psi_type_aware_mean": feature.at[run_id, "type_aware_mean"],
                    "feature_psi_type_aware_median": feature.at[
                        run_id, "type_aware_median"
                    ],
                    "feature_psi_type_aware_max": feature.at[run_id, "type_aware_max"],
                    "fold_jaccard_mean": stability.at[run_id, "jaccard_mean"],
                    "fold_jaccard_min": stability.at[run_id, "jaccard_min"],
                    "fold_jaccard_max": stability.at[run_id, "jaccard_max"],
                    "fold_kuncheva_mean": stability.at[run_id, "kuncheva_mean"],
                    "fold_kuncheva_min": stability.at[run_id, "kuncheva_min"],
                    "fold_kuncheva_max": stability.at[run_id, "kuncheva_max"],
                    "stability_universe_size": stability.at[
                        run_id, "authenticated_universe_size"
                    ],
                    "total_wall_clock_seconds": resources.at[
                        run_id, "total_wall_clock_seconds"
                    ],
                    "peak_process_tree_rss_bytes": resources.at[
                        run_id, "peak_process_tree_rss_bytes"
                    ],
                    "peak_process_gpu_bytes": resources.at[
                        run_id, "peak_process_gpu_bytes"
                    ],
                }
                for metric in ("auc", "gini", "ks", "lift_at_10"):
                    row[f"oot_{metric}_delta_vs_reference"] = float(
                        oot.at[run_id, metric] - oot.at[reference_id, metric]
                    )
                row["score_psi_difference_vs_reference"] = float(
                    psi.at[run_id, "score_psi"] - psi.at[reference_id, "score_psi"]
                )
                rows.append(row)

    budget = pd.DataFrame(rows)
    inference_index = holm.set_index(["family", "comparison_label"]) if not holm.empty else None
    bootstrap_index = (
        bootstrap.set_index(["family", "comparison_label"]) if not bootstrap.empty else None
    )
    delong_index = (
        delong.set_index(["family", "comparison_label"]) if not delong.empty else None
    )
    evidence_rows: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        label = (
            f"voting_pool_{int(row['candidate_pool_budget'])}_vs_rf_corr_mrmr"
            if row["candidate_pool_budget"] is not None
            and not pd.isna(row["candidate_pool_budget"])
            else None
        )
        family = f"{row['dataset']}_{row['model']}"
        record["comparison_label"] = label
        if label and delong_index is not None and (family, label) in delong_index.index:
            record["delong_auc_delta"] = delong_index.at[
                (family, label), "auc_delta_comparator_minus_reference"
            ]
            record["delong_standard_error"] = delong_index.at[
                (family, label), "standard_error"
            ]
            record["delong_z_statistic"] = delong_index.at[(family, label), "z_statistic"]
            record["raw_two_sided_p_value"] = delong_index.at[
                (family, label), "raw_two_sided_p_value"
            ]
        if label and bootstrap_index is not None and (family, label) in bootstrap_index.index:
            for metric in BOOTSTRAP_METRICS:
                record[f"{metric}_delta_ci95_lower"] = bootstrap_index.at[
                    (family, label), f"{metric}_delta_ci95_lower"
                ]
                record[f"{metric}_delta_ci95_upper"] = bootstrap_index.at[
                    (family, label), f"{metric}_delta_ci95_upper"
                ]
        if label and inference_index is not None and (family, label) in inference_index.index:
            record["holm_adjusted_p_value"] = inference_index.at[
                (family, label), "holm_adjusted_p_value"
            ]
            record["holm_family_size"] = inference_index.at[(family, label), "family_size"]
            record["holm_reject_null"] = inference_index.at[(family, label), "reject_null"]
        evidence_rows.append(record)
    return {
        "voting_budget_results": budget,
        "cross_dataset_voting_evidence_table": pd.DataFrame(evidence_rows),
    }


def build_final_inference_table(
    config: AnalysisConfig, delong: pd.DataFrame, bootstrap: pd.DataFrame, holm: pd.DataFrame
) -> pd.DataFrame:
    merged = delong.merge(
        bootstrap.drop(
            columns=[
                column
                for column in ("dataset", "model", "designation", "candidate_pool_budget", "reference_run_id", "comparator_run_id", "direction_convention")
                if column in bootstrap.columns
            ]
        ),
        on=["family", "comparison_label"],
        how="inner",
        validate="one_to_one",
    ).merge(
        holm.drop(
            columns=[
                column
                for column in ("dataset", "model", "designation", "candidate_pool_budget", "reference_run_id", "comparator_run_id", "raw_two_sided_p_value")
                if column in holm.columns
            ]
        ),
        on=["family", "comparison_label"],
        how="inner",
        validate="one_to_one",
    )
    merged["effect_strength_label"] = (
        "not_labelled_no_frozen_effect_strength_rule"
        if not bool(config.inference["effect_strength_labels_frozen"])
        else "frozen_rule_required"
    )
    merged["descriptive_direction"] = [
        "higher in this locked comparison"
        if delta > 0
        else ("lower in this locked comparison" if delta < 0 else "identical")
        for delta in merged["auc_delta_comparator_minus_reference"]
    ]
    merged["significance_is_not_business_materiality"] = True
    merged["non_significant_is_not_equivalence"] = True
    return merged.sort_values(
        ["family", "candidate_pool_budget"], kind="mergesort"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Phase J: consistency, provenance, and narrative artifacts
# ---------------------------------------------------------------------------


def provenance_audit(
    config: AnalysisConfig,
    *,
    runs: Sequence[RunRecord],
    inventory: pd.DataFrame,
    prediction_inventory: pd.DataFrame,
    run_metrics: pd.DataFrame,
    alignment: pd.DataFrame,
    fold_inventory: pd.DataFrame,
    delong: pd.DataFrame,
    bootstrap: pd.DataFrame,
    holm: pd.DataFrame,
    specification: Mapping[str, Any],
    independent: pd.DataFrame | None,
) -> tuple[pd.DataFrame, list[str]]:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            failures.append(f"{name}: {detail}")

    record(
        "every_headline_metric_recomputed_from_saved_predictions",
        len(run_metrics) == 2 * len(runs),
        f"{len(run_metrics)} metric rows for {len(runs)} runs across two splits",
    )
    record(
        "every_run_id_maps_to_one_configuration",
        inventory["run_id"].is_unique
        and inventory.groupby("run_id")["configuration"].nunique().max() == 1,
        "run ids are unique and each maps to one configuration",
    )
    record(
        "every_prediction_file_maps_to_one_manifest_and_source_hash",
        bool(
            (
                prediction_inventory["sha256"]
                == prediction_inventory["declared_artifact_sha256"]
            ).all()
        ),
        "artifact hashes equal the manifest-declared prediction hashes",
    )
    expected_folds = int(config.expected["total_dev_fold_executions"])
    present_folds = int(fold_inventory["present"].sum())
    record(
        "all_expected_fold_selections_accounted_for",
        present_folds == expected_folds,
        f"{present_folds} of {expected_folds} fold selections present",
    )
    oot_artifacts = int(
        (prediction_inventory["split"] == "OOT").sum()
    )
    record(
        "all_expected_oot_predictions_accounted_for",
        oot_artifacts == int(config.expected["oot_prediction_artifacts"]),
        f"{oot_artifacts} OOT prediction artifacts inventoried",
    )
    record(
        "all_metric_signs_use_comparator_minus_reference",
        bool((delong["direction_convention"] == "comparator_minus_reference").all()),
        "every inference row declares the comparator-minus-reference convention",
    )
    record(
        "auc_and_gini_algebraically_consistent",
        bool(run_metrics["gini_from_auc_consistent"].all()),
        "gini equals 2*AUC-1 for every recomputed row",
    )
    record(
        "raw_and_holm_adjusted_p_values_both_retained",
        {"raw_two_sided_p_value", "holm_adjusted_p_value"}.issubset(set(holm.columns)),
        "the Holm audit retains both raw and adjusted p-values",
    )
    record(
        "alignment_gate_passed_for_every_comparison",
        bool((alignment["decision"] == "aligned").all()),
        f"{int((alignment['decision'] == 'aligned').sum())} of {len(alignment)} cells aligned",
    )
    record(
        "zero_target_mismatches",
        bool((alignment["target_mismatch_count"].fillna(-1) == 0).all()),
        "every aligned cell reports zero target mismatches",
    )
    predeclared_labels = [
        (entry["multiplicity_family"], entry["comparison_label"])
        for entry in specification["comparisons"]
    ]
    executed_labels = list(
        zip(delong["family"], delong["comparison_label"], strict=True)
    )
    record(
        "all_predeclared_comparisons_represented_exactly_once",
        sorted(predeclared_labels) == sorted(executed_labels),
        f"{len(executed_labels)} executed comparisons match {len(predeclared_labels)} predeclared",
    )
    record(
        "no_oot_result_redefined_the_analysis",
        not bool(specification["constructed_after_viewing_oot_results"]),
        "the family was recovered from the frozen protocol before any p-value",
    )
    record(
        "bootstrap_replication_count_unchanged",
        bool((bootstrap["attempted_repetitions"] == 2000).all())
        and bool((bootstrap["valid_repetitions"] >= bootstrap["minimum_valid_repetitions"]).all()),
        "every comparison attempted 2,000 replications and met the valid minimum",
    )
    if independent is not None:
        record(
            "independent_headline_recalculation_passes",
            bool((independent["pass"]).all()),
            f"{int(independent['pass'].sum())} of {len(independent)} independent checks pass",
        )
    else:
        record(
            "independent_headline_recalculation_passes",
            False,
            "independent recalculation audit is unavailable",
        )
    excluded = list(config.expected["approved_exclusions"])
    record(
        "excluded_runs_were_approved_before_prompt_6",
        excluded == [],
        "no run exclusions were declared or applied",
    )
    return pd.DataFrame(checks), failures


def claims_and_evidence_seed(
    config: AnalysisConfig,
    *,
    final_inference: pd.DataFrame,
    budget: pd.DataFrame,
    stability_summary: pd.DataFrame,
    score_psi: pd.DataFrame,
    feature_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    script = "scripts/build_voting_inference_evidence.py"

    for row in final_inference.itertuples(index=False):
        direction = "higher" if row.auc_delta_comparator_minus_reference > 0 else "lower"
        significant = bool(row.reject_null)
        interval_excludes_zero = bool(
            row.auc_delta_ci95_lower is not None
            and row.auc_delta_ci95_upper is not None
            and not pd.isna(row.auc_delta_ci95_lower)
            and not pd.isna(row.auc_delta_ci95_upper)
            and (row.auc_delta_ci95_lower > 0 or row.auc_delta_ci95_upper < 0)
        )
        support = (
            "strong"
            if significant and interval_excludes_zero
            else ("moderate" if significant or interval_excludes_zero else "weak")
        )
        rows.append(
            {
                "proposed_factual_statement": (
                    f"In the locked {row.family} comparison, voting with candidate pool "
                    f"K={int(row.candidate_pool_budget)} produced a {direction} OOT ROC AUC "
                    f"than the same-cell rf_corr_mrmr reference "
                    f"(delta {row.auc_delta_comparator_minus_reference:+.6f}, "
                    f"95% bootstrap interval "
                    f"[{row.auc_delta_ci95_lower:+.6f}, {row.auc_delta_ci95_upper:+.6f}], "
                    f"Holm-adjusted p={row.holm_adjusted_p_value:.3g})."
                ),
                "evidence_type": "predeclared paired statistical comparison",
                "source_artifact": "paired_inference_final.csv",
                "calculation_script": script,
                "run_ids": f"{row.comparator_run_id};{row.reference_run_id}",
                "alignment_status": "identity-aligned OOT rows, zero target mismatches",
                "statistical_support": (
                    f"paired two-sided DeLong raw p={row.raw_two_sided_p_value:.3g}; "
                    f"Holm adjusted within a family of {int(row.family_size)}; "
                    f"alpha={row.alpha}"
                ),
                "limitations": (
                    "One locked OOT evaluation, one seed, one dataset period, and one "
                    "model configuration per cell. Significance is not business "
                    "materiality, and a non-significant result would not establish "
                    "equivalence."
                ),
                "support_rating": support,
            }
        )

    primary = final_inference.loc[final_inference["designation"] == "primary"]
    positive = int((primary["auc_delta_comparator_minus_reference"] > 0).sum())
    rejected = int(primary["reject_null"].sum())
    rows.append(
        {
            "proposed_factual_statement": (
                f"Across the four primary K=200 comparisons, the voting configuration "
                f"had a higher locked OOT AUC than its reference in {positive} of "
                f"{len(primary)} cells, and {rejected} of {len(primary)} remained "
                f"significant after Holm adjustment within their own family."
            ),
            "evidence_type": "aggregate of predeclared primary comparisons",
            "source_artifact": "paired_inference_final.csv",
            "calculation_script": script,
            "run_ids": ";".join(sorted(primary["comparator_run_id"])),
            "alignment_status": "all four cells identity-aligned with zero target mismatches",
            "statistical_support": "four independent Holm families of three tests each",
            "limitations": (
                "Direction counts are not a test. No pooled cross-family inference was "
                "predeclared, so no combined significance statement is available."
            ),
            "support_rating": "moderate",
        }
    )

    voting = budget.loc[~budget["is_reference"]]
    reference = budget.loc[budget["is_reference"]]
    rows.append(
        {
            "proposed_factual_statement": (
                "Fold-level selection stability is a descriptive property of the "
                "selection process: across the 16 completed runs the mean pairwise fold "
                f"Jaccard ranged from {stability_summary['jaccard_mean'].min():.4f} to "
                f"{stability_summary['jaccard_mean'].max():.4f} and mean pairwise "
                f"Kuncheva from {stability_summary['kuncheva_mean'].min():.4f} to "
                f"{stability_summary['kuncheva_mean'].max():.4f} under the frozen "
                "per-dataset candidate universe."
            ),
            "evidence_type": "descriptive stability recomputation",
            "source_artifact": "stability_summary.csv; stability_pairwise.csv",
            "calculation_script": script,
            "run_ids": ";".join(sorted(stability_summary["run_id"])),
            "alignment_status": "not applicable; computed from saved fold selections",
            "statistical_support": "none; no inferential test was predeclared for stability",
            "limitations": (
                "Kuncheva depends on the universe definition; the frozen protocol lists "
                "it as descriptive only. Higher stability does not imply higher AUC."
            ),
            "support_rating": "moderate",
        }
    )
    rows.append(
        {
            "proposed_factual_statement": (
                "Score PSI between out-of-fold DEV and locked OOT scores is available and "
                f"valid for all {len(score_psi)} runs, ranging from "
                f"{score_psi['score_psi'].min():.6f} to {score_psi['score_psi'].max():.6f} "
                "with DEV-OOF quantile bins applied unchanged to OOT."
            ),
            "evidence_type": "descriptive drift recomputation",
            "source_artifact": "score_psi_summary.csv; score_psi_bins.csv",
            "calculation_script": script,
            "run_ids": ";".join(sorted(score_psi["run_id"])),
            "alignment_status": "DEV out-of-fold and locked OOT populations verified disjoint",
            "statistical_support": "none; PSI has no predeclared test",
            "limitations": (
                "PSI reference values 0.10 and 0.25 are descriptive only in this "
                "protocol. Lower score PSI is not evidence of better discrimination."
            ),
            "support_rating": "strong",
        }
    )
    rows.append(
        {
            "proposed_factual_statement": (
                "No universal best-selector claim is supported by this evidence package: "
                f"the comparison covers {len(reference)} dataset-model cells, "
                f"{len(voting)} voting configurations, one seed, and one locked OOT window."
            ),
            "evidence_type": "scope boundary",
            "source_artifact": "cross_dataset_voting_evidence_table.csv; limitations.md",
            "calculation_script": script,
            "run_ids": ";".join(sorted(budget["run_id"])),
            "alignment_status": "not applicable",
            "statistical_support": "none; this is a scope statement",
            "limitations": "Applies to this protocol, these datasets, and these budgets only.",
            "support_rating": "strong",
        }
    )
    rows.append(
        {
            "proposed_factual_statement": (
                "The two voters are deterministic supervised selectors -- "
                "RandomForestRelevanceMRMRSelector and BorutaSelector -- fitted on DEV "
                "training folds only. No component of this protocol is an LLM call."
            ),
            "evidence_type": "frozen protocol authentication",
            "source_artifact": "predeclared_comparison_family.json; frozen_input_authentication.json",
            "calculation_script": script,
            "run_ids": ";".join(sorted(budget["run_id"])),
            "alignment_status": "not applicable",
            "statistical_support": "none; this is a design statement",
            "limitations": (
                "The excluded api_backed_llm and Home-Credit domain-rule voters remain "
                "outside this protocol version."
            ),
            "support_rating": "strong",
        }
    )
    rows.append(
        {
            "proposed_factual_statement": (
                "Selected-feature drift is reported with two disclosed definitions: the "
                "frozen numeric quantile PSI, which is unavailable for "
                f"{int(feature_summary['frozen_numeric_unavailable_count'].sum())} "
                "selected-feature occurrences, and a type-aware extension that retains "
                "explicit missing and unseen-level states."
            ),
            "evidence_type": "descriptive drift recomputation with definition audit",
            "source_artifact": "feature_psi_summary.csv; feature_psi_definition_audit.csv",
            "calculation_script": script,
            "run_ids": ";".join(sorted(feature_summary["run_id"])),
            "alignment_status": "not applicable; DEV and OOT feature distributions",
            "statistical_support": "none; PSI has no predeclared test",
            "limitations": (
                "The frozen protocol names only a numeric quantile implementation, so the "
                "type-aware values are an additional documented measure rather than the "
                "frozen definition."
            ),
            "support_rating": "moderate",
        }
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _write_table(config: AnalysisConfig, name: str, frame: pd.DataFrame) -> Path:
    path = reject_historical_write(config.package_root / name)
    write_csv_atomic(path, frame)
    log(f"wrote {name} ({len(frame)} rows)")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/analysis/cross_dataset_voting_inference_v1.yaml")
    parser.add_argument(
        "--skip-feature-psi",
        action="store_true",
        help="Skip Phase E only for a fast structural rehearsal; the completion gate then fails.",
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=None,
        help="Rehearsal override; any value other than the frozen 2000 fails the gate.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace an already-passing package at the same version instead of stopping.",
    )
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    config = load_analysis_config(REPOSITORY_ROOT, config_path=arguments.config)
    package_root = config.package_root
    audit_root = config.audit_root
    existing_status = package_root / "status.json"
    if (
        existing_status.is_file()
        and not bool(config.payload["execution"]["overwrite_completed_package"])
        and not arguments.allow_overwrite
        and str(read_json(existing_status).get("status")) == "PASS"
    ):
        log(
            f"a completed package already exists at {_relative(package_root, config.repository_root)}; "
            "bump the analysis version or pass --allow-overwrite"
        )
        return 4
    package_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)
    figures_root = config.figures_root
    figures_root.mkdir(parents=True, exist_ok=True)
    cache_root = package_root / "_cache"

    status: dict[str, Any] = {
        "schema_version": "prompt_06_voting_inference_status_v1",
        "analysis_id": config.payload["analysis_id"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "phases": {},
        "blockers": [],
        "needs_user_action": [],
    }
    process = None
    try:
        import psutil

        process = psutil.Process(os.getpid())
    except Exception:  # pragma: no cover - psutil is a declared dependency
        process = None

    try:
        # ---- Phase A ----------------------------------------------------
        log("Phase A1: recording repository and environment state")
        state = repository_state()
        write_json_atomic(audit_root / "repository_state.json", state)

        log("Phase A: authenticating frozen inputs")
        frozen = authenticate_frozen_inputs(config)
        write_json_atomic(audit_root / "frozen_input_authentication.json", frozen)
        if frozen["status"] != "PASS":
            raise Blocker("; ".join(frozen["failures"]))
        status["phases"]["A_frozen_inputs"] = "PASS"

        log("Phase A3: discovering registered runs")
        runs = discover_runs(config)
        inventory = build_input_inventory(config, runs)
        _write_table(config, "input_inventory.csv", inventory)

        log("Phase A2: preservation audit")
        preservation = preservation_audit(config, runs)
        write_json_atomic(audit_root / "preservation_audit.json", preservation)
        if preservation["status"] != "PASS":
            raise Blocker("; ".join(preservation["failures"]))
        status["phases"]["A_preservation"] = "PASS"

        log("Phase A3: authenticating Prompt 5 completion")
        completion = authenticate_prompt_05_completion(config, runs, inventory)
        write_json_atomic(
            package_root / "prompt_05_completion_authentication.json", completion
        )
        if completion["status"] != "PASS":
            raise Blocker("; ".join(completion["failures"]))
        status["phases"]["A_prompt_05_completion"] = "PASS"

        # ---- Phase H1 before any p-value -------------------------------
        log("Phase H1: recovering the predeclared comparison family")
        comparisons, specification = recover_family(config, runs)
        write_json_atomic(
            package_root / "predeclared_comparison_family.json", specification
        )
        status["phases"]["H1_predeclared_family"] = "PASS"

        # ---- Phase B/C/D ------------------------------------------------
        log("Phase B/C/D: alignment, metric recomputation, and score PSI")
        evidence = process_prediction_evidence(config, runs)
        alignment = evidence["alignment"]
        _write_table(config, "alignment_audit.csv", alignment)
        write_json_atomic(
            package_root / "alignment_audit.json",
            {
                "schema_version": "prompt_06_alignment_audit_v1",
                "identity_column": config.expected["row_identifier"],
                "positive_class": int(config.expected["positive_class"]),
                "score_direction": str(config.expected["score_direction"]),
                "cells": json.loads(alignment.to_json(orient="records")),
                "dev_oot_disjoint": json.loads(
                    evidence["dev_oot_disjoint"].to_json(orient="records")
                ),
                "aligned_cell_count": int((alignment["decision"] == "aligned").sum()),
                "blocked_cell_count": int((alignment["decision"] != "aligned").sum()),
            },
        )
        _write_table(config, "leakage_audit.csv", evidence["leakage"])
        _write_table(config, "prediction_inventory.csv", evidence["prediction_inventory"])
        _write_table(config, "dev_oot_population_audit.csv", evidence["dev_oot_disjoint"])
        if not (alignment["decision"] == "aligned").all():
            raise Blocker(
                "alignment gate failed: "
                + "; ".join(
                    f"{row.reference_run_id}->{row.comparator_run_id} {row.split}: {row.decision_reason}"
                    for row in alignment.loc[alignment["decision"] != "aligned"].itertuples(index=False)
                )
            )
        if not (evidence["dev_oot_disjoint"]["decision"] == "disjoint").all():
            raise Blocker("DEV and OOT populations are not disjoint for every run")
        if not (evidence["leakage"]["decision"] == "clean").all():
            raise Blocker(
                "leakage audit failed for: "
                + "; ".join(
                    evidence["leakage"].loc[
                        evidence["leakage"]["decision"] != "clean", "run_id"
                    ]
                )
            )
        status["phases"]["B_alignment"] = "PASS"

        _write_table(config, "run_level_metrics.csv", evidence["run_level_metrics"])
        _write_table(config, "lift10_audit.csv", evidence["lift10_audit"])
        status["phases"]["C_metric_recomputation"] = "PASS"

        _write_table(config, "score_psi_summary.csv", evidence["score_psi_summary"])
        _write_table(config, "score_psi_bins.csv", evidence["score_psi_bins"])
        status["phases"]["D_score_psi"] = "PASS"

        # ---- Phase F ----------------------------------------------------
        log("Phase F: fold-level selection stability")
        stability = compute_stability(config, runs)
        _write_table(config, "fold_selection_inventory.csv", stability["fold_selection_inventory"])
        _write_table(config, "stability_pairwise.csv", stability["stability_pairwise"])
        _write_table(config, "stability_summary.csv", stability["stability_summary"])
        _write_table(
            config, "fold_selection_frequency.csv", stability["fold_selection_frequency"]
        )
        if not bool(stability["stability_summary"]["frozen_implementation_agrees"].all()):
            raise Blocker(
                "recomputed Kuncheva disagrees with the frozen repository implementation"
            )
        status["phases"]["F_stability"] = "PASS"

        # ---- Phase G ----------------------------------------------------
        log("Phase G: runtime and resource evidence")
        runtime = pd.DataFrame([runtime_resource_row(run) for run in runs])
        stage_rows: list[dict[str, Any]] = []
        for run in runs:
            stage_rows.extend(stage_breakdown_rows(run))
        _write_table(config, "runtime_resource_summary.csv", runtime)
        _write_table(config, "runtime_stage_breakdown.csv", pd.DataFrame(stage_rows))
        status["phases"]["G_resources"] = "PASS"

        # ---- Phase E ----------------------------------------------------
        if arguments.skip_feature_psi:
            log("Phase E: SKIPPED by request (rehearsal mode)")
            feature = {
                "feature_psi_long": pd.DataFrame(),
                "feature_psi_summary": pd.DataFrame(),
                "feature_psi_definition_audit": pd.DataFrame(),
            }
            status["phases"]["E_feature_psi"] = "SKIPPED"
        else:
            log("Phase E: type-aware feature PSI")
            feature = compute_feature_psi(config, runs)
            _write_table(config, "feature_psi_long.csv", feature["feature_psi_long"])
            _write_table(config, "feature_psi_summary.csv", feature["feature_psi_summary"])
            _write_table(
                config,
                "feature_psi_definition_audit.csv",
                feature["feature_psi_definition_audit"],
            )
            status["phases"]["E_feature_psi"] = "PASS"

        # ---- Phase C3: independent recalculation -----------------------
        # Runs before the long bootstrap so a recomputation disagreement stops
        # the pipeline early instead of after two hours of resampling.
        log("Phase C3: independent headline recalculation")
        independent_path = package_root / "independent_recalculation_audit.csv"
        completed = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / "scripts" / "independently_verify_voting_metrics.py"),
                "--primary-metrics",
                str(package_root / "run_level_metrics.csv"),
                "--primary-score-psi",
                str(package_root / "score_psi_summary.csv"),
                "--primary-alignment",
                str(package_root / "alignment_audit.csv"),
                "--primary-lift-audit",
                str(package_root / "lift10_audit.csv"),
                "--output",
                str(independent_path),
                "--tolerance",
                repr(config.tolerance),
            ],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        sys.stdout.write(completed.stdout)
        if completed.returncode != 0:
            sys.stderr.write(completed.stderr)
            raise Blocker(
                "independent recalculation audit failed with exit code "
                f"{completed.returncode}"
            )
        independent = pd.read_csv(independent_path)
        if not bool(independent["pass"].all()):
            raise Blocker(
                "independent recalculation exceeded tolerance for: "
                + "; ".join(
                    independent.loc[~independent["pass"], "metric"].astype(str).unique()
                )
            )
        status["phases"]["C3_independent_recalculation"] = "PASS"

        # ---- Phase H ----------------------------------------------------
        log("Phase H: predeclared paired inference")
        if arguments.bootstrap_repetitions is not None:
            config.inference["bootstrap"]["attempted_repetitions"] = int(
                arguments.bootstrap_repetitions
            )
            config.inference["bootstrap"]["minimum_valid_repetitions"] = 1
        inference = run_paired_inference(
            config, comparisons, evidence["aligned_oot"], cache_directory=cache_root
        )
        _write_table(config, "paired_delong_results.csv", inference["paired_delong_results"])
        _write_table(
            config, "paired_bootstrap_results.csv", inference["paired_bootstrap_results"]
        )
        _write_table(
            config, "holm_adjustment_audit.csv", inference["holm_adjustment_audit"]
        )
        _write_table(
            config, "bootstrap_equivalence_audit.csv", inference["bootstrap_equivalence"]
        )
        write_json_atomic(
            package_root / "bootstrap_replicate_manifest.json",
            inference["bootstrap_replicate_manifest"],
        )
        final_inference = build_final_inference_table(
            config,
            inference["paired_delong_results"],
            inference["paired_bootstrap_results"],
            inference["holm_adjustment_audit"],
        )
        _write_table(config, "paired_inference_final.csv", final_inference)
        status["phases"]["H_paired_inference"] = "PASS"

        # ---- Phase I ----------------------------------------------------
        log("Phase I: budget-curve and cross-dataset evidence tables")
        if feature["feature_psi_summary"].empty:
            raise Blocker(
                "feature PSI is unavailable, so the evidence table cannot be completed"
            )
        tables = build_evidence_tables(
            config,
            run_metrics=evidence["run_level_metrics"],
            score_psi=evidence["score_psi_summary"],
            feature_psi_summary=feature["feature_psi_summary"],
            stability_summary=stability["stability_summary"],
            runtime=runtime,
            inventory=inventory,
            delong=inference["paired_delong_results"],
            bootstrap=inference["paired_bootstrap_results"],
            holm=inference["holm_adjustment_audit"],
        )
        _write_table(config, "voting_budget_results.csv", tables["voting_budget_results"])
        _write_table(
            config,
            "cross_dataset_voting_evidence_table.csv",
            tables["cross_dataset_voting_evidence_table"],
        )
        captions = [
            figure_builders.voting_budget_auc(
                evidence["run_level_metrics"],
                figures_root / "voting_budget_auc.png",
                source_table="run_level_metrics.csv",
            ),
            figure_builders.paired_auc_delta_forest(
                final_inference,
                figures_root / "paired_auc_delta_forest.png",
                source_table="paired_inference_final.csv",
            ),
            figure_builders.drift_stability_comparison(
                tables["voting_budget_results"],
                figures_root / "drift_stability_comparison.png",
                source_table="voting_budget_results.csv",
                references=[float(v) for v in config.payload["descriptive_psi_references"]],
            ),
            figure_builders.runtime_comparison(
                runtime,
                figures_root / "runtime_comparison.png",
                source_table="runtime_resource_summary.csv",
            ),
        ]
        write_json_atomic(package_root / "figure_captions.json", {"figures": captions})
        status["phases"]["I_evidence_tables"] = "PASS"

        # ---- Phase J ----------------------------------------------------
        log("Phase J: consistency and provenance audit")
        provenance, provenance_failures = provenance_audit(
            config,
            runs=runs,
            inventory=inventory,
            prediction_inventory=evidence["prediction_inventory"],
            run_metrics=evidence["run_level_metrics"],
            alignment=alignment,
            fold_inventory=stability["fold_selection_inventory"],
            delong=inference["paired_delong_results"],
            bootstrap=inference["paired_bootstrap_results"],
            holm=inference["holm_adjustment_audit"],
            specification=specification,
            independent=independent,
        )
        _write_table(config, "provenance_audit.csv", provenance)
        if provenance_failures:
            raise Blocker("; ".join(provenance_failures))
        claims = claims_and_evidence_seed(
            config,
            final_inference=final_inference,
            budget=tables["voting_budget_results"],
            stability_summary=stability["stability_summary"],
            score_psi=evidence["score_psi_summary"],
            feature_summary=feature["feature_psi_summary"],
        )
        _write_table(config, "claims_and_evidence_seed.csv", claims)

        write_text_atomic(
            package_root / "limitations.md",
            _limitations_markdown(
                config,
                state=state,
                completion=completion,
                specification=specification,
                feature_summary=feature["feature_psi_summary"],
                stability_summary=stability["stability_summary"],
                runtime=runtime,
                final_inference=final_inference,
            ),
        )
        write_text_atomic(
            package_root / "validation_summary.md",
            _validation_markdown(
                config,
                state=state,
                frozen=frozen,
                preservation=preservation,
                completion=completion,
                alignment=alignment,
                run_metrics=evidence["run_level_metrics"],
                independent=independent,
                inference=inference,
                final_inference=final_inference,
                provenance=provenance,
            ),
        )
        elapsed = time.perf_counter() - _START
        _seal_final_status_and_manifest(
            config,
            state=state,
            frozen=frozen,
            status=status,
            final_status={
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": elapsed,
                "peak_process_rss_bytes": int(process.memory_info().rss) if process else None,
                "package_root": _relative(package_root, config.repository_root),
                "audit_root": _relative(audit_root, config.repository_root),
                "comparison_family_count": specification["family_count"],
                "comparison_count": specification["comparison_count"],
                "delong_comparison_count": int(len(inference["paired_delong_results"])),
                "bootstrap_comparison_count": int(len(inference["paired_bootstrap_results"])),
                "bootstrap_repetitions": int(
                    inference["paired_bootstrap_results"]["attempted_repetitions"].iloc[0]
                ),
                "holm_family_count": int(
                    inference["holm_adjustment_audit"]["family"].nunique()
                ),
                "status": "PASS",
                "success_marker": SUCCESS_MARKER,
            },
        )
        log(SUCCESS_MARKER)
        return 0

    except NeedsUserAction as error:
        status.update(
            {
                "status": "NEEDS_USER_ACTION",
                "needs_user_action": [str(error)],
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": time.perf_counter() - _START,
            }
        )
        write_json_atomic(package_root / "status.json", status)
        log(f"NEEDS USER ACTION | {error}")
        return 3
    except Exception as error:  # noqa: BLE001 - every failure must stay visible
        status.update(
            {
                "status": "BLOCKED",
                "blockers": [f"{type(error).__name__}: {error}"],
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": time.perf_counter() - _START,
            }
        )
        write_json_atomic(package_root / "status.json", status)
        log(f"BLOCKED | {type(error).__name__}: {error}")
        raise


def _seal_final_status_and_manifest(
    config: AnalysisConfig,
    *,
    state: Mapping[str, Any],
    frozen: Mapping[str, Any],
    status: dict[str, Any],
    final_status: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the immutable final status before hashing the package payload."""

    status["phases"]["J_provenance"] = "PASS"
    status.update(final_status)
    write_json_atomic(config.package_root / "status.json", status)
    manifest = _artifact_manifest(config, state=state, frozen=frozen)
    write_json_atomic(config.package_root / "artifact_manifest.json", manifest)
    return manifest


def _artifact_manifest(
    config: AnalysisConfig, *, state: Mapping[str, Any], frozen: Mapping[str, Any]
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for root in (config.package_root, config.audit_root):
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.parent == config.package_root and (
                path.name in {"artifact_manifest.json", "artifact_manifest.current.json"}
                or (
                    path.name.startswith("artifact_manifest.v")
                    and path.name.endswith(".json")
                )
            ):
                continue
            entries.append(
                {
                    "path": _relative(path, config.repository_root),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema_version": "prompt_06_artifact_manifest_v1",
        "analysis_id": config.payload["analysis_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": state["git_head"],
        "git_tags_at_head": state["git_tags_at_head"],
        "analysis_config_sha256": config.config_sha256,
        "frozen_inputs": frozen["inputs"],
        "generated_files": entries,
        "generated_file_count": len(entries),
        "writes_confined_to": [
            _relative(config.package_root, config.repository_root),
            _relative(config.audit_root, config.repository_root),
        ],
    }


def _limitations_markdown(
    config: AnalysisConfig,
    *,
    state: Mapping[str, Any],
    completion: Mapping[str, Any],
    specification: Mapping[str, Any],
    feature_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    runtime: pd.DataFrame,
    final_inference: pd.DataFrame,
) -> str:
    unavailable = (
        int(feature_summary["frozen_numeric_unavailable_count"].sum())
        if not feature_summary.empty
        else 0
    )
    return f"""# Prompt 6 limitations and unavailable evidence

Generated at {datetime.now(timezone.utc).isoformat()} from commit `{state["git_head"]}`.

## What this package is

An independent recomputation and statistical packaging of the 16 completed
`cross_dataset_rank_voting_v1` runs. Every number traces to a saved applicant
level prediction, a saved selection artifact, or a saved manifest. No model was
fitted, no selector was rerun, no ranking was regenerated, and no completed run
directory or legacy artifact was modified.

## Frozen assumptions carried into this analysis

- The comparison family, bootstrap design, seed, replication count, confidence
  level, and Holm scope were recovered from
  `configs/protocols/cross_dataset_rank_voting_v1.yaml`
  (SHA-256 `{specification["recovered_from_sha256"]}`) before any p-value was
  computed. They were not chosen or edited after viewing OOT results.
- Lift@10 uses the frozen `ceil(0.10*n)` top group by descending class-1
  probability with stable-identity ascending tie handling, and is reported as
  top-decile bad rate divided by overall bad rate. The capture-rate form was not
  used.
- Score PSI uses the frozen `dev_oof_quantile_psi_v1` implementation: bins are
  defined on out-of-fold DEV scores only and applied unchanged to OOT.
- Kuncheva uses the frozen per-dataset leakage-safe candidate universe
  ({config.metric_definitions["kuncheva"]["universe_size"]}). This is the only
  quantity the protocol, the run matrix, and the saved fold rankings all name as
  a universe. The alternative "method-specific candidate pool" reading is
  recorded and rejected in `configs/analysis/cross_dataset_voting_inference_v1.yaml`
  because the four reference runs have no voting pool, which would make every
  Kuncheva denominator zero for the reference arm.

## Real limitations

1. **One locked OOT window per dataset.** Home Credit OOT is `[-240,0]` and
   LendingClub v2 OOT is `[-1065,-730]`. Every inferential statement is about
   these specific windows, not about future periods.
2. **One seed, one model configuration per cell.** Seed 42 governs selection and
   model fitting. Between-seed variance was not estimated, so nothing here
   separates method effect from seed effect.
3. **DEV out-of-fold coverage is a subset of DEV.** Expanding-window folds never
   hold out the first training block, so the DEV-OOF artifacts cover fewer rows
   than the full DEV split. Score PSI therefore compares out-of-fold DEV scores
   for that covered subpopulation against the full locked OOT.
4. **Feature PSI has two disclosed definitions.** The frozen numeric quantile
   implementation is unavailable for {unavailable} selected-feature occurrences
   (categorical or constant source columns). The type-aware extension in this
   package retains explicit missing and unseen-level states and is reported
   beside the frozen value, not instead of it.
5. **Feature PSI is computed at the final full-DEV selection level.** The frozen
   definition compares a DEV reference distribution against an unchanged OOT
   distribution; per-fold selected-feature PSI against OOT is not part of that
   definition and was not invented here.
6. **Stage-level runtime timers are null in the saved artifacts.** Only total
   wall-clock time, peak RSS, minimum available RAM, GPU bytes, and disk minima
   were instrumented. The stage breakdown in
   `runtime_stage_breakdown.csv` is reconstructed from timestamped log markers
   and is labelled as log-derived. Runs were executed once, sequentially, on
   shared hardware, and several were interrupted and resumed, so runtime is
   observational cost evidence and not a controlled benchmark.
7. **No pooled cross-family inference exists.** The protocol predeclares four
   separate Holm families of three tests. Counting how many cells favour voting
   is descriptive, not a test.
8. **Bootstrap intervals are percentile intervals from 2,000 attempts.** They
   are not bias-corrected, and no interval is reported for a metric the protocol
   did not predeclare.
9. **PSI and stability are not discrimination.** Score PSI, feature PSI,
   Jaccard, and Kuncheva are descriptive in this protocol. A lower PSI or a
   higher Jaccard is not evidence of better ranking performance.
10. **The accelerated bootstrap is an implementation change, not a design
    change.** It reproduces the frozen implementation exactly at seed
    {specification["bootstrap"]["seed"]}; the per-comparison proof is in
    `bootstrap_equivalence_audit.csv`.
11. **`statsmodels` is not installed and is not used.** Holm adjustment uses the
    frozen repository implementation.
12. **Pilot runs are excluded by design.** The
    {len(completion["pilot_run_directories_excluded"])} `cdv1-pilot-*`
    single-fold integration runs are not part of the inference family and were
    not added to it.

## Statements this package does not support

- No claim that any selector is universally best.
- No causal claim. Predictive comparisons are associations under a fixed design.
- No equivalence claim from a non-significant p-value.
- No business-materiality claim from statistical significance. The largest
  primary-cell AUC delta observed here is
  {final_inference.loc[final_inference["designation"] == "primary", "auc_delta_comparator_minus_reference"].abs().max():.6f}.
- No description of the deterministic domain-rule ranking as an LLM call; that
  ranker is excluded from this protocol entirely.
"""


def _validation_markdown(
    config: AnalysisConfig,
    *,
    state: Mapping[str, Any],
    frozen: Mapping[str, Any],
    preservation: Mapping[str, Any],
    completion: Mapping[str, Any],
    alignment: pd.DataFrame,
    run_metrics: pd.DataFrame,
    independent: pd.DataFrame,
    inference: Mapping[str, Any],
    final_inference: pd.DataFrame,
    provenance: pd.DataFrame,
) -> str:
    oot = run_metrics.loc[run_metrics["split"] == "OOT"]
    dev = run_metrics.loc[run_metrics["split"] == "DEV_OOF"]
    rows = "\n".join(
        f"| {row.check} | {'PASS' if row.passed else 'FAIL'} | {row.detail} |"
        for row in provenance.itertuples(index=False)
    )
    return f"""# Prompt 6 validation summary

Commit `{state["git_head"]}`; tags {state["git_tags_at_head"]}.
Python {state["python_version"]}; numpy {state["package_versions"]["numpy"]},
pandas {state["package_versions"]["pandas"]}, scipy {state["package_versions"]["scipy"]},
scikit-learn {state["package_versions"]["scikit-learn"]},
catboost {state["package_versions"]["catboost"]}, joblib {state["package_versions"]["joblib"]},
pyarrow {state["package_versions"]["pyarrow"]}, psutil {state["package_versions"]["psutil"]},
statsmodels {state["package_versions"]["statsmodels"]}.

## Gates

| Gate | Result |
|---|---|
| Frozen input authentication | {frozen["status"]} ({len(frozen["inputs"])} inputs) |
| Preservation audit | {preservation["status"]} |
| Prompt 5 completion authentication | {completion["status"]} |
| Alignment cells aligned | {int((alignment["decision"] == "aligned").sum())} / {len(alignment)} |
| Target mismatches across all aligned cells | {int(alignment["target_mismatch_count"].fillna(0).sum())} |
| Independent recalculation checks passed | {int(independent["pass"].sum())} / {len(independent)} |
| Independent recalculation tolerance | {config.tolerance:g} |
| Maximum independent absolute difference | {float(independent["absolute_difference"].max()):.3e} |

## Prediction row counts

| Split | Distinct row counts observed |
|---|---|
| DEV out-of-fold | {sorted(set(int(v) for v in dev["row_count"]))} |
| Locked OOT | {sorted(set(int(v) for v in oot["row_count"]))} |

## Statistical package

- Comparison families: {inference["holm_adjustment_audit"]["family"].nunique()}
- Predeclared comparisons executed: {len(final_inference)}
- Paired DeLong tests: {len(inference["paired_delong_results"])}
- Bootstrap comparisons: {len(inference["paired_bootstrap_results"])}
- Bootstrap attempts per comparison: {int(inference["paired_bootstrap_results"]["attempted_repetitions"].iloc[0])}
- Valid bootstrap replications (minimum observed): {int(inference["paired_bootstrap_results"]["valid_repetitions"].min())}
- Failed bootstrap replications (maximum observed): {int(inference["paired_bootstrap_results"]["failed_repetitions"].max())}
- Accelerated/frozen bootstrap equivalence: {"all comparisons exact" if bool(inference["bootstrap_equivalence"]["equivalent"].all()) else "MISMATCH"}
- Excluded comparisons: none

## Consistency and provenance checks

| Check | Result | Detail |
|---|---|---|
{rows}
"""


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
