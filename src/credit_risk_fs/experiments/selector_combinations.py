"""Artifact-safe plan, gates, and bounded pilot for Prompt 11 combinations.

Plan/status/baseline validation never import a research dataset loader.  The
loader and estimators are imported only inside spawned real-workload workers.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from credit_risk_fs.experiments.atomic_io import (
    copy_atomic,
    sha256_file,
    write_csv_atomic,
    write_json_atomic,
)
from credit_risk_fs.experiments.full_baseline import inspect_cell, load_full_baseline_plan
from credit_risk_fs.experiments.resource_monitor import (
    MANUAL_INTERRUPT,
    supervise_worker,
    wait_for_inter_run_readiness,
)
from credit_risk_fs.experiments.resource_policy import (
    detect_hardware,
    load_execution_policy,
    resolve_execution_policy,
    run_preflight,
)
from credit_risk_fs.selectors.combinations import COMBINATION_CLASSES
from credit_risk_fs.experiments.research_logging import ResearchLogSession


CONFIG_SCHEMA_VERSION = "selector_combination_research_config_v1"
ARTIFACT_SCHEMA_VERSION = "selector_combination_pilot_artifact_v1"
DEFAULT_CONFIG_PATH = Path("configs/experiments/selector_combination_research_v1.yaml")
METHOD_ORDER = (
    "statistical_normalized_average_rank",
    "iv_then_boruta",
    "boruta_then_mrmr_mutual_information",
    "boruta_then_rfe_catboost",
)
DATASET_ORDER = ("homecredit", "lendingclub_v2")
MODEL_ORDER = ("lr", "catboost")
EXPECTED_CANDIDATES = {"homecredit": 529, "lendingclub_v2": 675}


class CombinationPipelineError(ValueError):
    pass


class CombinationGateClosed(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SelectionSpec:
    selection_index: int
    selection_id: str
    dataset: str
    fold_id: int
    method_id: str
    final_budget: int | None
    iv_pool_budget: int | None
    selector_kwargs: dict[str, Any]
    wall_clock_limit_seconds: float


@dataclass(frozen=True, slots=True)
class EvaluationCell:
    cell_index: int
    cell_id: str
    selection_id: str
    dataset: str
    fold_id: int
    method_id: str
    model: str
    final_budget: int | None
    iv_pool_budget: int | None
    wall_clock_limit_seconds: float


@dataclass(frozen=True, slots=True)
class CombinationPlan:
    repository_root: Path
    config_path: Path
    configuration_sha256: str
    protocol_lock_path: Path
    protocol_lock_sha256: str
    results_root: Path
    policy_path: Path
    log_path: Path
    configuration: dict[str, Any]
    selections: tuple[SelectionSpec, ...]
    evaluations: tuple[EvaluationCell, ...]


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _artifact_with_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload.pop("artifact_authentication_sha256", None)
    payload["artifact_authentication_sha256"] = _canonical_sha(payload)
    return payload


def _validate_artifact(path: Path, expected: Mapping[str, Any]) -> tuple[bool, str, dict[str, Any] | None]:
    if not path.is_file():
        return False, "missing", None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        observed = payload.pop("artifact_authentication_sha256")
        if observed != _canonical_sha(payload):
            return False, "authentication_sha256_mismatch", None
        for key, value in expected.items():
            if payload.get(key) != value:
                return False, f"identity_mismatch:{key}", None
        for entry in payload.get("artifact_files", []):
            relative = Path(str(entry.get("path", "")))
            candidate = (path.parent / relative).resolve()
            if not candidate.is_relative_to(path.parent.resolve()):
                return False, "artifact_path_escape", None
            if (
                not candidate.is_file()
                or candidate.stat().st_size != int(entry.get("size_bytes", -1))
                or sha256_file(candidate) != entry.get("sha256")
            ):
                return False, f"artifact_file_mismatch:{relative.as_posix()}", None
        if payload.get("terminal_state") != "completed":
            return False, f"terminal_state:{payload.get('terminal_state')}", None
        payload["artifact_authentication_sha256"] = observed
        return True, "authenticated_complete", payload
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return False, str(exc), None


def _selector_kwargs(config: Mapping[str, Any], method: str, budget: int | None, pool: int | None) -> dict[str, Any]:
    settings = dict(config["selector_settings"])
    common = {
        "random_state": 42,
        "fit_scope": "dev_fold_training_only",
    }
    if method == "iv_then_boruta":
        return {
            **common,
            "iv_pool_budget": int(pool),
            "iv_kwargs": copy.deepcopy(settings["iv_woe"]),
            "boruta_kwargs": copy.deepcopy(settings["boruta_random_forest"]),
        }
    if method == "boruta_then_mrmr_mutual_information":
        return {
            **common,
            "k": int(budget),
            "boruta_kwargs": copy.deepcopy(settings["boruta_random_forest"]),
            "refiner_kwargs": copy.deepcopy(settings["mrmr_mutual_information"]),
        }
    if method == "boruta_then_rfe_catboost":
        return {
            **common,
            "k": int(budget),
            "boruta_kwargs": copy.deepcopy(settings["boruta_random_forest"]),
            "refiner_kwargs": copy.deepcopy(settings["rfe_catboost"]),
        }
    if method == "statistical_normalized_average_rank":
        return {
            **common,
            "k": int(budget),
            "component_kwargs": {
                item: copy.deepcopy(settings[item])
                for item in (
                    "iv_woe",
                    "lasso_l1_logistic",
                    "rfe_catboost",
                    "boruta_random_forest",
                    "catboost_shap",
                )
            },
        }
    raise CombinationPipelineError(f"unregistered combination: {method}")


def _selection_key(dataset: str, fold: int, method: str, budget: int | None, pool: int | None) -> str:
    suffix = f"pool{pool}" if pool is not None else (f"k{budget}" if budget is not None else "natural")
    return f"scv1-{dataset}-fold-{fold}-{method.replace('_', '-')}-{suffix}-s42"


def _budget_slug(budget: int | None, pool: int | None) -> str:
    return f"pool{pool}" if pool is not None else f"k{budget or 'natural'}"


def _build_pilot_matrix(config: Mapping[str, Any]) -> tuple[tuple[SelectionSpec, ...], tuple[EvaluationCell, ...]]:
    limits = config["execution"]["wall_clock_limits_seconds"]
    selections: list[SelectionSpec] = []
    evaluations: list[EvaluationCell] = []
    seen: dict[str, SelectionSpec] = {}

    def add(dataset: str, method: str, model: str, *, budget: int | None = None, pool: int | None = None) -> None:
        selection_id = _selection_key(dataset, 1, method, budget, pool)
        if selection_id not in seen:
            spec = SelectionSpec(
                selection_index=len(selections) + 1,
                selection_id=selection_id,
                dataset=dataset,
                fold_id=1,
                method_id=method,
                final_budget=budget,
                iv_pool_budget=pool,
                selector_kwargs=_selector_kwargs(config, method, budget, pool),
                wall_clock_limit_seconds=float(limits[method]),
            )
            selections.append(spec)
            seen[selection_id] = spec
        evaluations.append(
            EvaluationCell(
                cell_index=len(evaluations) + 1,
                cell_id=f"scv1-pilot-{len(evaluations)+1:03d}-{dataset}-{method.replace('_', '-')}-{model}-{_budget_slug(budget, pool)}-s42",
                selection_id=selection_id,
                dataset=dataset,
                fold_id=1,
                method_id=method,
                model=model,
                final_budget=budget,
                iv_pool_budget=pool,
                wall_clock_limit_seconds=float(limits[f"final_{model}"]),
            )
        )

    # Exact cheapest-first preregistered order.  The nested order also fixes Cell 001.
    for dataset in DATASET_ORDER:
        for model in MODEL_ORDER:
            add(dataset, METHOD_ORDER[0], model, budget=int(config["protocol"]["final_feature_budgets"][model]))
    for pool in config["protocol"]["iv_pool_budgets"]:
        for dataset in DATASET_ORDER:
            for model in MODEL_ORDER:
                add(dataset, METHOD_ORDER[1], model, pool=int(pool))
    for method in METHOD_ORDER[2:]:
        for dataset in DATASET_ORDER:
            for model in MODEL_ORDER:
                add(dataset, method, model, budget=int(config["protocol"]["final_feature_budgets"][model]))
    return tuple(selections), tuple(evaluations)


def build_phase_matrix(
    plan: CombinationPlan,
    *,
    phase: str,
    retained_method_ids: tuple[str, ...] = METHOD_ORDER,
) -> tuple[tuple[SelectionSpec, ...], tuple[EvaluationCell, ...]]:
    """Expand the frozen pilot identities to five DEV folds or one full-DEV OOT fit."""

    if phase not in {"dev", "oot"}:
        raise ValueError("phase matrix must be dev or oot")
    if not retained_method_ids or any(item not in METHOD_ORDER for item in retained_method_ids):
        raise CombinationPipelineError("approval retained an unknown or empty method set")
    folds = (1, 2, 3, 4, 5) if phase == "dev" else (0,)
    selections: list[SelectionSpec] = []
    evaluations: list[EvaluationCell] = []
    seen: set[str] = set()
    base_selection = {item.selection_id: item for item in plan.selections}
    for fold_id in folds:
        for base in plan.evaluations:
            if base.method_id not in retained_method_ids:
                continue
            selection_id = _selection_key(
                base.dataset,
                fold_id,
                base.method_id,
                base.final_budget,
                base.iv_pool_budget,
            )
            if selection_id not in seen:
                source = base_selection[base.selection_id]
                kwargs = copy.deepcopy(source.selector_kwargs)
                kwargs["fit_scope"] = (
                    "dev_fold_training_only" if phase == "dev" else "full_dev_only"
                )
                selections.append(
                    SelectionSpec(
                        selection_index=len(selections) + 1,
                        selection_id=selection_id,
                        dataset=base.dataset,
                        fold_id=fold_id,
                        method_id=base.method_id,
                        final_budget=base.final_budget,
                        iv_pool_budget=base.iv_pool_budget,
                        selector_kwargs=kwargs,
                        wall_clock_limit_seconds=source.wall_clock_limit_seconds,
                    )
                )
                seen.add(selection_id)
            fold_label = f"fold-{fold_id}" if phase == "dev" else "full-dev"
            evaluations.append(
                EvaluationCell(
                    cell_index=len(evaluations) + 1,
                    cell_id=(
                        f"scv1-{phase}-{len(evaluations)+1:03d}-{base.dataset}-"
                        f"{base.method_id.replace('_', '-')}-{base.model}-{fold_label}-"
                        f"{_budget_slug(base.final_budget, base.iv_pool_budget)}-s42"
                    ),
                    selection_id=selection_id,
                    dataset=base.dataset,
                    fold_id=fold_id,
                    method_id=base.method_id,
                    model=base.model,
                    final_budget=base.final_budget,
                    iv_pool_budget=base.iv_pool_budget,
                    wall_clock_limit_seconds=base.wall_clock_limit_seconds,
                )
            )
    expected_evaluations = len(folds) * sum(
        item.method_id in retained_method_ids for item in plan.evaluations
    )
    if len(evaluations) != expected_evaluations:
        raise CombinationPipelineError("expanded phase matrix identity count changed")
    return tuple(selections), tuple(evaluations)


def load_combination_plan(repository_root: str | Path, config_path: str | Path = DEFAULT_CONFIG_PATH) -> CombinationPlan:
    root = Path(repository_root).resolve()
    path = Path(config_path)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise CombinationPipelineError("combination configuration schema mismatch")
    protocol = config["protocol"]
    if tuple(protocol["datasets"]) != DATASET_ORDER or tuple(protocol["models"]) != MODEL_ORDER:
        raise CombinationPipelineError("dataset/model order differs from the frozen protocol")
    if tuple(protocol["execution_order"]) != METHOD_ORDER:
        raise CombinationPipelineError("combination order differs from the frozen protocol")
    if protocol["pilot_fold_ids"] != [1] or protocol["iv_pool_budgets"] != [100, 200, 300]:
        raise CombinationPipelineError("pilot fold or IV sensitivity grid changed")
    protocol_path = (root / config["protocol_lock_path"]).resolve()
    if not protocol_path.is_file():
        raise CombinationPipelineError("committed combination protocol lock is missing")
    selections, evaluations = _build_pilot_matrix(config)
    if len(selections) != 18 or len(evaluations) != 24:
        raise CombinationPipelineError("pilot matrix must contain 18 selections and 24 evaluations")
    expected_first = "scv1-pilot-001-homecredit-statistical-normalized-average-rank-lr-k20-s42"
    if evaluations[0].cell_id != expected_first:
        raise CombinationPipelineError("pilot Cell 001 identity changed")
    return CombinationPlan(
        repository_root=root,
        config_path=path,
        configuration_sha256=sha256_file(path),
        protocol_lock_path=protocol_path,
        protocol_lock_sha256=sha256_file(protocol_path),
        results_root=(root / config["results_root"]).resolve(),
        policy_path=(root / config["execution_policy_path"]).resolve(),
        log_path=(root / config["log_path"]).resolve(),
        configuration=config,
        selections=selections,
        evaluations=evaluations,
    )


def render_plan(plan: CombinationPlan) -> dict[str, Any]:
    return {
        "schema_version": "selector_combination_plan_v1",
        "configuration_sha256": plan.configuration_sha256,
        "protocol_lock_sha256": plan.protocol_lock_sha256,
        "raw_dataset_paths_resolved": False,
        "workers_started": 0,
        "pilot_selection_count": len(plan.selections),
        "pilot_evaluation_count": len(plan.evaluations),
        "execution_order": list(METHOD_ORDER),
        "first_cell": asdict(plan.evaluations[0]),
        "selections": [asdict(item) for item in plan.selections],
        "evaluations": [asdict(item) for item in plan.evaluations],
        "dev_gate": "closed_pending_authenticated_pilot_approval_lock",
        "oot_gate": "closed_pending_pilot_approval_and_complete_authenticated_dev",
    }


def validate_prompt_10_baselines(plan: CombinationPlan) -> dict[str, Any]:
    baseline = load_full_baseline_plan(plan.repository_root)
    rows = [inspect_cell(baseline, cell) for cell in baseline.cells]
    failures = [item for item in rows if not item.valid_completed]
    final_cell = baseline.cells[-1]
    success = (
        baseline.results_root
        / "runs"
        / final_cell.dataset
        / final_cell.cell_id
        / "_SUCCESS"
    )
    if failures or len(baseline.cells) != 36 or not success.is_file():
        raise CombinationPipelineError(
            f"Prompt 10 dependency invalid: failures={len(failures)}, success={success.is_file()}"
        )
    return {
        "status": "authenticated_complete",
        "expected_cells": 36,
        "authenticated_cells": 36,
        "configuration_sha256": baseline.configuration_sha256,
        "runtime_policy_sha256": baseline.runtime_policy.source_sha256,
        "cell_036": final_cell.cell_id,
        "success_marker_sha256": sha256_file(success),
        "raw_dataset_paths_resolved": False,
        "baseline_refit_performed": False,
    }


def _selection_path(plan: CombinationPlan, selection_id: str, phase: str = "pilot") -> Path:
    return plan.results_root / phase / "selections" / f"{selection_id}.json"


def _evaluation_path(plan: CombinationPlan, cell_id: str, phase: str = "pilot") -> Path:
    return plan.results_root / phase / "evaluations" / f"{cell_id}.json"


def _publish_selection_files(path: Path, worker_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Publish separately authenticated combination, intermediate, final, and voter evidence."""

    result = dict(worker_result["combination_result"])
    stem = path.stem
    files: list[dict[str, Any]] = []

    def record(metadata: Any) -> None:
        value = metadata.to_dict()
        value["path"] = Path(value["path"]).name
        files.append(value)

    record(
        write_json_atomic(
            path.with_name(f"{stem}.combination_result.json"), result, overwrite=True
        )
    )
    final = list(map(str, result["selected_features"]))
    record(
        write_csv_atomic(
            path.with_name(f"{stem}.final_selected_features.csv"),
            pd.DataFrame({"rank": range(1, len(final) + 1), "feature": final}),
            overwrite=True,
        )
    )
    intermediate = result.get("intermediate_features")
    if intermediate is not None:
        values = list(map(str, intermediate))
        record(
            write_csv_atomic(
                path.with_name(f"{stem}.intermediate_features.csv"),
                pd.DataFrame({"rank": range(1, len(values) + 1), "feature": values}),
                overwrite=True,
            )
        )
    voting = result.get("voting_evidence")
    if voting is not None:
        record(
            write_csv_atomic(
                path.with_name(f"{stem}.voting_evidence.csv"),
                pd.DataFrame(voting),
                overwrite=True,
            )
        )
    return files


def _publish_state(plan: CombinationPlan, path: Path, payload: Mapping[str, Any]) -> None:
    """Archive a prior terminal attempt, then atomically publish the new state."""

    if path.is_file():
        digest = sha256_file(path)
        archive = (
            plan.results_root
            / "incomplete"
            / "attempt_history"
            / path.parent.name
            / f"{path.stem}.{digest}.json"
        )
        if not archive.is_file():
            copy_atomic(path, archive, overwrite=False)
    write_json_atomic(path, _artifact_with_hash(payload), overwrite=True)


def build_status(plan: CombinationPlan) -> dict[str, Any]:
    selections = []
    for item in plan.selections:
        valid, reason, _ = _validate_artifact(
            _selection_path(plan, item.selection_id),
            {"selection_id": item.selection_id, "configuration_sha256": plan.configuration_sha256},
        )
        selections.append({"selection_id": item.selection_id, "valid": valid, "state": reason})
    evaluations = []
    for item in plan.evaluations:
        valid, reason, _ = _validate_artifact(
            _evaluation_path(plan, item.cell_id),
            {"cell_id": item.cell_id, "configuration_sha256": plan.configuration_sha256},
        )
        evaluations.append({"cell_id": item.cell_id, "valid": valid, "state": reason})
    def progress(phase: str, cells: tuple[EvaluationCell, ...]) -> dict[str, Any]:
        rows = []
        for cell in cells:
            valid, reason, _ = _validate_artifact(
                _evaluation_path(plan, cell.cell_id, phase),
                {"cell_id": cell.cell_id, "configuration_sha256": plan.configuration_sha256},
            )
            rows.append({"cell_id": cell.cell_id, "valid": valid, "state": reason})
        return {
            "authenticated_evaluations": sum(item["valid"] for item in rows),
            "expected_evaluations": len(rows),
            "first_incomplete": next((item["cell_id"] for item in rows if not item["valid"]), None),
        }

    approval_payload = None
    try:
        approval_payload = _validate_approval_lock(plan)
        dev_gate = "open_authenticated_pilot_approval"
    except CombinationGateClosed as exc:
        dev_gate = f"closed:{exc}"
    dev_progress = None
    oot_progress = None
    oot_gate = "closed_pending_authenticated_pilot_approval_and_complete_dev"
    if approval_payload is not None:
        retained = tuple(approval_payload["retained_method_ids"])
        _, dev_cells = build_phase_matrix(plan, phase="dev", retained_method_ids=retained)
        _, oot_cells = build_phase_matrix(plan, phase="oot", retained_method_ids=retained)
        dev_progress = progress("dev", dev_cells)
        oot_progress = progress("oot", oot_cells)
        try:
            _validate_dev_completion_lock(plan, approval_payload)
            oot_gate = "open_authenticated_dev_complete"
        except CombinationGateClosed as exc:
            oot_gate = f"closed:{exc}"
    return {
        "schema_version": "selector_combination_status_v1",
        "configuration_sha256": plan.configuration_sha256,
        "raw_dataset_paths_resolved": False,
        "workers_started": 0,
        "pilot": {
            "authenticated_selections": sum(item["valid"] for item in selections),
            "expected_selections": 18,
            "authenticated_evaluations": sum(item["valid"] for item in evaluations),
            "expected_evaluations": 24,
            "first_incomplete": next((item["cell_id"] for item in evaluations if not item["valid"]), None),
            "selections": selections,
            "evaluations": evaluations,
        },
        "dev": dev_progress,
        "oot": oot_progress,
        "dev_gate": dev_gate,
        "oot_gate": oot_gate,
    }


def _git_clean(root: Path) -> None:
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise CombinationPipelineError("real execution requires a clean committed worktree")


def _load_fold(root: Path, dataset: str, fold_id: int, projected: list[str] | None = None):
    # Deliberately local imports: plan/status/validation cannot resolve data.
    from credit_risk_fs.experiments.rank_voting import canonical_fold_projection
    from credit_risk_fs.pipelines.common import prepare_voting_pilot_dev_data

    kwargs: dict[str, Any] = {"dataset": dataset, "csv_chunk_rows": 25_000, "csv_low_memory": False}
    if projected is not None:
        kwargs["projected_candidate_features"] = projected
    prepared = prepare_voting_pilot_dev_data(root, **kwargs)
    if any(int(item.get("oot_rows_retained", -1)) != 0 for item in prepared.data_access_log):
        raise CombinationPipelineError("DEV loader retained OOT rows")
    projection = canonical_fold_projection(
        y=prepared.y,
        stable_row_ids=prepared.stable_row_ids,
        time_values=prepared.time_values,
        fold_id=fold_id,
    )
    return prepared, projection


def combination_selection_worker(
    *, stop_event: Any, stage_queue: Any, repository_root: str, spec: Mapping[str, Any],
    protocol_lock_sha256: str, estimator_threads: int, phase: str = "pilot", **_: Any,
) -> dict[str, Any]:
    from credit_risk_fs.preprocessing.encoding import OriginalFeatureNumericEncoder
    from credit_risk_fs.experiments.row_alignment import ordered_row_id_sha256, ordered_row_id_target_sha256
    from credit_risk_fs.experiments.research_logging import suppress_third_party_output

    fold_id = int(spec["fold_id"])
    allowed = (
        (phase == "pilot" and fold_id == 1)
        or (phase == "dev" and fold_id in range(1, 6))
        or (phase == "oot" and fold_id == 0)
    )
    if not allowed or spec["dataset"] not in DATASET_ORDER:
        raise CombinationPipelineError("selection worker phase/fold boundary is invalid")
    if stop_event.is_set():
        raise RuntimeError("cooperative stop requested")
    stage_queue.put({"stage": f"{phase}_{spec['method_id']}_selection", "fold_id": fold_id})
    root = Path(repository_root)
    if fold_id == 0:
        from credit_risk_fs.pipelines.common import prepare_voting_pilot_dev_data

        prepared = prepare_voting_pilot_dev_data(
            root,
            dataset=str(spec["dataset"]),
            csv_chunk_rows=25_000,
            csv_low_memory=False,
        )
        if any(int(item.get("oot_rows_retained", -1)) != 0 for item in prepared.data_access_log):
            raise CombinationPipelineError("full-DEV selection loader retained OOT rows")
        projection = None
    else:
        prepared, projection = _load_fold(root, str(spec["dataset"]), fold_id)
    candidates = list(map(str, prepared.candidate_features))
    if len(candidates) != EXPECTED_CANDIDATES[str(spec["dataset"])]:
        raise CombinationPipelineError("candidate universe count mismatch")
    if projection is None:
        X = prepared.X.reset_index(drop=True)
        y = prepared.y.reset_index(drop=True)
        ids = prepared.stable_row_ids.astype(str).reset_index(drop=True)
    else:
        tr = projection["training_indices"]
        positions = projection["source_positions"]
        X = prepared.X.iloc[positions[tr]].reset_index(drop=True)
        y = projection["y"].iloc[tr].reset_index(drop=True)
        ids = projection["ids"].iloc[tr].astype(str).reset_index(drop=True)
    numeric = OriginalFeatureNumericEncoder().fit_transform(X)
    kwargs = copy.deepcopy(dict(spec["selector_kwargs"]))
    kwargs["protocol_lock_sha256"] = protocol_lock_sha256
    # Enforce the resolved worker limit throughout nested estimators.
    def cap(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: (min(int(item), estimator_threads) if key in {"n_jobs", "thread_count"} else cap(item)) for key, item in value.items()}
        return value
    kwargs = cap(kwargs)
    selector = COMBINATION_CLASSES[str(spec["method_id"])](**kwargs)
    with suppress_third_party_output():
        selector.fit(numeric, y)
    result = selector.result.to_dict()
    return {
        "selection_id": spec["selection_id"],
        "combination_result": result,
        "selected_features": result["selected_features"],
        "training_ordered_row_id_sha256": ordered_row_id_sha256(ids),
        "training_ordered_row_id_target_sha256": ordered_row_id_target_sha256(ids, y),
        "candidate_universe_sha256": result["candidate_universe_sha256"],
        "opened_oot_paths": [],
        "oot_rows_retained": 0,
    }


def combination_evaluation_worker(
    *, stop_event: Any, stage_queue: Any, repository_root: str, cell: Mapping[str, Any],
    selected_features: list[str], estimator_threads: int, phase: str = "pilot",
    output_state_path: str | None = None, **_: Any,
) -> dict[str, Any]:
    import numpy as np
    from credit_risk_fs.evaluation.metrics import evaluate_model
    from credit_risk_fs.experiments.rank_voting import _fit_final_model
    from credit_risk_fs.experiments.row_alignment import ordered_row_id_sha256, ordered_row_id_target_sha256

    fold_id = int(cell["fold_id"])
    if not (
        (phase == "pilot" and fold_id == 1)
        or (phase == "dev" and fold_id in range(1, 6))
    ):
        raise CombinationPipelineError("DEV evaluation worker phase/fold boundary is invalid")
    if not selected_features:
        raise CombinationPipelineError("natural-support selection is empty and cannot be evaluated")
    stage_queue.put({"stage": f"{phase}_final_{cell['model']}", "fold_id": fold_id})
    prepared, projection = _load_fold(Path(repository_root), str(cell["dataset"]), fold_id, selected_features)
    tr, va, positions = projection["training_indices"], projection["validation_indices"], projection["source_positions"]
    probabilities, effective = _fit_final_model(
        repository_root=Path(repository_root), dataset=str(cell["dataset"]), model_name=str(cell["model"]),
        selected_features=list(selected_features),
        X_train_raw=prepared.X.iloc[positions[tr]].reset_index(drop=True),
        y_train=projection["y"].iloc[tr].reset_index(drop=True),
        X_validation_raw=prepared.X.iloc[positions[va]].reset_index(drop=True),
        seed=42, estimator_threads=estimator_threads,
        stage_callback=lambda stage, fold, **fields: stage_queue.put({"stage": stage, "fold_id": fold, **fields}),
        fold_id=fold_id,
    )
    y_val = projection["y"].iloc[va].reset_index(drop=True)
    ids = projection["ids"].iloc[va].astype(str).reset_index(drop=True)
    metrics = evaluate_model(y_val, probabilities)
    artifact_files: list[dict[str, Any]] = []
    if output_state_path is not None:
        state_path = Path(output_state_path).resolve()
        stem = state_path.stem
        frame = pd.DataFrame(
            {
                "stable_row_id": ids,
                "target": y_val.astype("int8"),
                "prediction_probability": probabilities,
                "dataset": str(cell["dataset"]),
                "model": str(cell["model"]),
                "method": str(cell["method_id"]),
                "split": "dev",
                "fold_id": fold_id,
                "run_id": str(cell["cell_id"]),
            }
        )
        metadata = (
            write_csv_atomic(
                state_path.with_name(f"{stem}.dev_predictions.csv"),
                frame,
                ordered_row_identity_column="stable_row_id",
                overwrite=True,
            ),
            write_json_atomic(
                state_path.with_name(f"{stem}.dev_metrics.json"),
                {"metrics": {key: float(value) for key, value in metrics.items()}, "validation_targets_used_for_fit": False},
                overwrite=True,
            ),
            write_json_atomic(
                state_path.with_name(f"{stem}.final_model_configuration.json"),
                effective,
                overwrite=True,
            ),
        )
        for item in metadata:
            value = item.to_dict()
            value["path"] = Path(value["path"]).name
            artifact_files.append(value)
    return {
        "cell_id": cell["cell_id"],
        "metrics": {key: float(value) for key, value in metrics.items()},
        "effective_model_configuration": effective,
        "validation_row_count": len(ids),
        "validation_ordered_row_id_sha256": ordered_row_id_sha256(ids),
        "validation_ordered_row_id_target_sha256": ordered_row_id_target_sha256(ids, y_val),
        "prediction_sha256": _canonical_sha([float(value) for value in np.asarray(probabilities)]),
        "validation_targets_used_for_fit": False,
        "opened_oot_paths": [],
        "oot_rows_retained": 0,
        "artifact_files": artifact_files,
    }


def combination_oot_evaluation_worker(
    *,
    stop_event: Any,
    stage_queue: Any,
    repository_root: str,
    cell: Mapping[str, Any],
    selected_features: list[str],
    estimator_threads: int,
    output_state_path: str,
    **_: Any,
) -> dict[str, Any]:
    """Fit on full DEV, then open OOT once for transform/evaluation only."""

    import gc
    import numpy as np
    from credit_risk_fs.analysis.baseline_audit import recompute_prediction_metrics
    from credit_risk_fs.experiments.research_logging import suppress_third_party_output
    from credit_risk_fs.experiments.row_alignment import (
        ordered_row_id_sha256,
        ordered_row_id_target_sha256,
    )
    from credit_risk_fs.models.registry import get_model_bundle
    from credit_risk_fs.pipelines.common import (
        prepare_voting_pilot_dev_data,
        prepare_voting_research_oot_data,
    )
    from credit_risk_fs.preprocessing.encoding import Preprocessor

    if int(cell["fold_id"]) != 0 or not selected_features:
        raise CombinationPipelineError("OOT evaluation requires one non-empty full-DEV selection")
    if stop_event.is_set():
        raise RuntimeError("cooperative stop requested before full-DEV loading")
    root = Path(repository_root).resolve()
    state_path = Path(output_state_path).resolve()
    stage_queue.put({"stage": "oot_full_dev_data_loading", "fold_id": None})
    dev = prepare_voting_pilot_dev_data(
        root,
        dataset=str(cell["dataset"]),
        projected_candidate_features=selected_features,
        csv_chunk_rows=25_000,
        csv_low_memory=False,
    )
    if any(int(item.get("oot_rows_retained", -1)) != 0 for item in dev.data_access_log):
        raise CombinationPipelineError("full-DEV loader retained OOT rows")
    dataset_config = yaml.safe_load(
        (root / f"configs/experiments/{cell['dataset']}_matrix.yaml").read_text(encoding="utf-8")
    )
    frozen = yaml.safe_load(
        (root / "configs/experiments/full_baseline_v1.yaml").read_text(encoding="utf-8")
    )
    model_kwargs = copy.deepcopy(frozen["final_model_settings"][str(cell["model"])])
    model_kwargs["random_state"] = 42
    if cell["model"] == "catboost":
        model_kwargs["thread_count"] = min(int(estimator_threads), 4)
    preprocessor = Preprocessor(**dict(dataset_config.get("preprocessor_kwargs", {})))
    stage_queue.put({"stage": "oot_full_dev_preprocessing", "fold_id": None})
    X_dev = preprocessor.fit_transform(dev.X.loc[:, selected_features])
    get_model, _, predict_proba, _ = get_model_bundle(str(cell["model"]), model_kwargs)
    model = get_model()
    stage_queue.put({"stage": "oot_full_dev_model_fit", "fold_id": None})
    with suppress_third_party_output():
        model.fit(X_dev, dev.y, eval_set=None)
    classes = [int(value) for value in model.model.classes_]
    if classes != [0, 1]:
        raise CombinationPipelineError("full-DEV final model probability orientation is invalid")
    dev_source_hashes = dict(dev.source_artifact_hashes)
    del X_dev
    gc.collect()

    # OOT is unreachable until the final model is already fitted.
    stage_queue.put({"stage": "locked_oot_data_loading", "fold_id": None})
    oot = prepare_voting_research_oot_data(
        root,
        dataset=str(cell["dataset"]),
        projected_candidate_features=selected_features,
        csv_chunk_rows=25_000,
        csv_low_memory=False,
    )
    if _canonical_sha(dict(oot.source_artifact_hashes)) != _canonical_sha(dev_source_hashes):
        raise CombinationPipelineError("locked OOT source provenance differs from full DEV")
    stage_queue.put({"stage": "locked_oot_transform_prediction", "fold_id": None})
    X_oot = preprocessor.transform(oot.X.loc[:, selected_features])
    probabilities = np.asarray(predict_proba(model, X_oot), dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != len(oot.y):
        raise CombinationPipelineError("OOT prediction length is invalid")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise CombinationPipelineError("OOT predictions are not finite probabilities")
    prediction = pd.DataFrame(
        {
            "stable_row_id": oot.stable_row_ids.astype(str),
            "target": oot.y.astype("int8"),
            "prediction_probability": probabilities,
            "dataset": str(cell["dataset"]),
            "model": str(cell["model"]),
            "method": str(cell["method_id"]),
            "split": "oot",
            "run_id": str(cell["cell_id"]),
        }
    )
    metrics = recompute_prediction_metrics(prediction)
    stem = state_path.stem
    prediction_meta = write_csv_atomic(
        state_path.with_name(f"{stem}.oot_predictions.csv"),
        prediction,
        ordered_row_identity_column="stable_row_id",
        overwrite=True,
    )
    metric_meta = write_json_atomic(
        state_path.with_name(f"{stem}.oot_metrics.json"),
        {
            "metrics": metrics,
            "metric_scope": "locked_oot_saved_predictions_only",
            "configuration_adaptation_after_oot": False,
            "threshold_selected_on_oot": False,
        },
        overwrite=True,
    )
    config_meta = write_json_atomic(
        state_path.with_name(f"{stem}.final_model_configuration.json"),
        {
            "training_scope": "full_DEV",
            "selected_features": selected_features,
            "requested_model_configuration": model_kwargs,
            "actual_estimator_configuration": model.model.get_params(),
            "probability_classes": classes,
            "oot_used_for_fit": False,
            "oot_loaded_after_model_fit": True,
        },
        overwrite=True,
    )
    artifact_files = []
    for metadata in (prediction_meta, metric_meta, config_meta):
        item = metadata.to_dict()
        item["path"] = Path(item["path"]).name
        artifact_files.append(item)
    return {
        "cell_id": cell["cell_id"],
        "selected_feature_count": len(selected_features),
        "oot_row_count": len(prediction),
        "oot_ordered_row_id_sha256": ordered_row_id_sha256(prediction["stable_row_id"]),
        "oot_ordered_row_id_target_sha256": ordered_row_id_target_sha256(
            prediction["stable_row_id"], prediction["target"]
        ),
        "metrics": metrics,
        "artifact_files": artifact_files,
        "oot_loaded_after_model_fit": True,
        "oot_used_for_fit_or_adaptation": False,
    }


def _supervisor_payload(result: Any) -> dict[str, Any]:
    return {
        "status": result.status,
        "stop_code": result.stop_code,
        "worker_exit_code": result.worker_exit_code,
        "worker_error": result.worker_error,
        "peak_process_tree_rss_bytes": result.peak_process_tree_rss_bytes,
        "minimum_system_available_ram_bytes": result.minimum_system_available_ram_bytes,
        "warnings": list(result.warnings),
        "final_stage": result.final_stage,
        "final_fold_id": result.final_fold_id,
    }


def _run_supervised(plan: CombinationPlan, *, target: str, kwargs: dict[str, Any], association: str, wall: float, policy: Any, ram_control: Any) -> Any:
    readiness = wait_for_inter_run_readiness(
        policy=policy, results_root=plan.results_root, temp_root=Path(tempfile.gettempdir()).resolve(), ram_control_policy=ram_control
    )
    if not readiness.ready:
        raise CombinationPipelineError(f"resource readiness rejected {association}: {readiness.stop_code}")
    return supervise_worker(
        worker_target=target, worker_kwargs=kwargs, policy=policy,
        results_root=plan.results_root, temp_root=Path(tempfile.gettempdir()).resolve(),
        run_association=association, heartbeat_interval_seconds=float(plan.configuration["execution"]["heartbeat_interval_seconds"]),
        max_wall_clock_seconds=wall, ram_control_policy=ram_control,
    )


def _execute_dev_evaluation_matrix(
    plan: CombinationPlan,
    *,
    phase: str,
    selections: tuple[SelectionSpec, ...],
    evaluations: tuple[EvaluationCell, ...],
) -> dict[str, Any]:
    if phase not in {"pilot", "dev"}:
        raise ValueError("DEV evaluation matrix phase must be pilot or dev")
    _git_clean(plan.repository_root)
    validate_prompt_10_baselines(plan)
    plan.results_root.mkdir(parents=True, exist_ok=True)
    configured = load_execution_policy(plan.repository_root, plan.policy_path)
    temp_root = Path(tempfile.gettempdir()).resolve()
    capacity = detect_hardware(plan.results_root, temp_root)
    policy = resolve_execution_policy(configured, capacity)
    parallel = policy.parallelism
    if (parallel.concurrent_experiment_runs, parallel.concurrent_folds, parallel.data_loader_workers) != (1, 1, 0) or parallel.estimator_threads > 4:
        raise CombinationPipelineError("execution policy widened the frozen sequential resource limits")
    preflight = run_preflight(
        repository_root=plan.repository_root, config_path=plan.policy_path,
        results_root=plan.results_root, temp_root=temp_root, requested_accelerator="cpu", capacity=capacity,
    )
    if preflight.get("status") != "pass":
        raise CombinationPipelineError(f"{phase} preflight failed: {preflight.get('blocking_reasons')}")
    from credit_risk_fs.experiments.ram_control import ResolvedRamControlPolicy
    ram_control = ResolvedRamControlPolicy(**dict(preflight["ram_control_policy"]))
    completed = 0
    for cell in evaluations:
        cell_path = _evaluation_path(plan, cell.cell_id, phase)
        valid, _, _ = _validate_artifact(cell_path, {"cell_id": cell.cell_id, "configuration_sha256": plan.configuration_sha256})
        if valid:
            completed += 1
            continue
        selection = next(item for item in selections if item.selection_id == cell.selection_id)
        selection_path = _selection_path(plan, selection.selection_id, phase)
        selected_valid, _, selected_payload = _validate_artifact(
            selection_path, {"selection_id": selection.selection_id, "configuration_sha256": plan.configuration_sha256}
        )
        if not selected_valid:
            result = _run_supervised(
                plan, target="credit_risk_fs.experiments.selector_combinations:combination_selection_worker",
                kwargs={"repository_root": str(plan.repository_root), "spec": asdict(selection), "protocol_lock_sha256": plan.protocol_lock_sha256, "estimator_threads": parallel.estimator_threads, "phase": phase},
                association=selection.selection_id, wall=selection.wall_clock_limit_seconds, policy=policy, ram_control=ram_control,
            )
            payload = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "terminal_state": "completed" if result.status == "completed" else result.status,
                "selection_id": selection.selection_id,
                "configuration_sha256": plan.configuration_sha256,
                "protocol_lock_sha256": plan.protocol_lock_sha256,
                "selection_spec": asdict(selection),
                "worker_result": result.return_value,
                "supervisor": _supervisor_payload(result),
            }
            if result.status == "completed":
                payload["artifact_files"] = _publish_selection_files(
                    selection_path, result.return_value
                )
            _publish_state(plan, selection_path, payload)
            if result.status != "completed":
                return {
                    "status": result.status,
                    "stop_code": result.stop_code,
                    "completed_evaluations": completed,
                    "stop_cell": cell.cell_id,
                }
            selected_payload = _artifact_with_hash(payload)
        selected = list(selected_payload["worker_result"]["selected_features"])
        result = _run_supervised(
            plan, target="credit_risk_fs.experiments.selector_combinations:combination_evaluation_worker",
            kwargs={"repository_root": str(plan.repository_root), "cell": asdict(cell), "selected_features": selected, "estimator_threads": parallel.estimator_threads, "phase": phase, "output_state_path": str(cell_path)},
            association=cell.cell_id, wall=cell.wall_clock_limit_seconds, policy=policy, ram_control=ram_control,
        )
        payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "terminal_state": "completed" if result.status == "completed" else result.status,
            "cell_id": cell.cell_id,
            "selection_id": cell.selection_id,
            "configuration_sha256": plan.configuration_sha256,
            "protocol_lock_sha256": plan.protocol_lock_sha256,
            "evaluation_cell": asdict(cell),
            "selection_artifact_sha256": sha256_file(selection_path),
            "worker_result": result.return_value,
            "supervisor": _supervisor_payload(result),
            "artifact_files": (
                list(result.return_value["artifact_files"])
                if result.status == "completed"
                else []
            ),
        }
        _publish_state(plan, cell_path, payload)
        if result.status != "completed":
            return {
                "status": result.status,
                "stop_code": result.stop_code,
                "completed_evaluations": completed,
                "stop_cell": cell.cell_id,
            }
        completed += 1
    return {
        "status": "completed",
        "phase": phase,
        "completed_evaluations": completed,
        "expected_evaluations": len(evaluations),
        "authenticated_selection_identities": len(selections),
    }


def execute_pilot(plan: CombinationPlan) -> dict[str, Any]:
    return _execute_dev_evaluation_matrix(
        plan,
        phase="pilot",
        selections=plan.selections,
        evaluations=plan.evaluations,
    )


def _validate_approval_lock(plan: CombinationPlan) -> dict[str, Any]:
    path = plan.repository_root / plan.configuration["gates"]["pilot_approval_lock_path"]
    if not path.is_file():
        raise CombinationGateClosed("full combination DEV is gated pending authenticated pilot review and approval lock")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        authentication = payload.pop("artifact_authentication_sha256")
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CombinationGateClosed("pilot approval lock is unreadable or unauthenticated") from exc
    if authentication != _canonical_sha(payload):
        raise CombinationGateClosed("pilot approval lock authentication SHA-256 is invalid")
    expected_pilot = [item.cell_id for item in plan.evaluations]
    expected_selections = [item.selection_id for item in plan.selections]
    retained = tuple(payload.get("retained_method_ids", ()))
    valid = (
        payload.get("schema_version") == "selector_combination_pilot_approval_lock_v1"
        and payload.get("configuration_sha256") == plan.configuration_sha256
        and payload.get("protocol_lock_sha256") == plan.protocol_lock_sha256
        and payload.get("approved") is True
        and payload.get("pilot_artifacts_valid") is True
        and payload.get("completed_pilot_evaluation_ids") == expected_pilot
        and payload.get("completed_pilot_selection_ids") == expected_selections
        and payload.get("predictive_outcomes_used_for_configuration") is False
        and bool(payload.get("user_approval_record"))
        and bool(payload.get("stage_support_review"))
        and bool(payload.get("runtime_resource_review"))
        and bool(payload.get("retention_decisions"))
        and retained
        and all(item in METHOD_ORDER for item in retained)
        and len(set(retained)) == len(retained)
        and retained == tuple(item for item in METHOD_ORDER if item in retained)
    )
    if not valid:
        raise CombinationGateClosed("pilot approval lock is not authenticated for this exact completed pilot/configuration")
    for selection in plan.selections:
        complete, reason, _ = _validate_artifact(
            _selection_path(plan, selection.selection_id, "pilot"),
            {"selection_id": selection.selection_id, "configuration_sha256": plan.configuration_sha256},
        )
        if not complete:
            raise CombinationGateClosed(
                f"pilot approval dependency is invalid for selection {selection.selection_id}: {reason}"
            )
    for cell in plan.evaluations:
        complete, reason, _ = _validate_artifact(
            _evaluation_path(plan, cell.cell_id, "pilot"),
            {"cell_id": cell.cell_id, "configuration_sha256": plan.configuration_sha256},
        )
        if not complete:
            raise CombinationGateClosed(
                f"pilot approval dependency is invalid for evaluation {cell.cell_id}: {reason}"
            )
    payload["artifact_authentication_sha256"] = authentication
    return payload


def _dev_completion_lock_path(plan: CombinationPlan) -> Path:
    return plan.repository_root / plan.configuration["gates"]["dev_completion_lock_path"]


def _validate_dev_completion_lock(
    plan: CombinationPlan,
    approval: Mapping[str, Any],
) -> dict[str, Any]:
    path = _dev_completion_lock_path(plan)
    if not path.is_file():
        raise CombinationGateClosed("combination OOT is gated pending complete authenticated DEV")
    expected = {
        "configuration_sha256": plan.configuration_sha256,
        "protocol_lock_sha256": plan.protocol_lock_sha256,
        "pilot_approval_lock_sha256": sha256_file(
            plan.repository_root / plan.configuration["gates"]["pilot_approval_lock_path"]
        ),
    }
    valid, reason, payload = _validate_artifact(path, expected)
    if not valid or payload is None:
        raise CombinationGateClosed(f"combination DEV completion lock is invalid: {reason}")
    retained = tuple(approval["retained_method_ids"])
    _, expected_cells = build_phase_matrix(plan, phase="dev", retained_method_ids=retained)
    if payload.get("completed_dev_evaluation_ids") != [item.cell_id for item in expected_cells]:
        raise CombinationGateClosed("DEV completion lock does not bind the exact retained matrix")
    return payload


def execute_dev(plan: CombinationPlan) -> dict[str, Any]:
    approval = _validate_approval_lock(plan)
    retained = tuple(approval["retained_method_ids"])
    selections, evaluations = build_phase_matrix(
        plan, phase="dev", retained_method_ids=retained
    )
    result = _execute_dev_evaluation_matrix(
        plan,
        phase="dev",
        selections=selections,
        evaluations=evaluations,
    )
    if result.get("status") != "completed":
        return result
    approval_path = plan.repository_root / plan.configuration["gates"]["pilot_approval_lock_path"]
    payload = {
        "schema_version": "selector_combination_dev_completion_lock_v1",
        "terminal_state": "completed",
        "configuration_sha256": plan.configuration_sha256,
        "protocol_lock_sha256": plan.protocol_lock_sha256,
        "pilot_approval_lock_sha256": sha256_file(approval_path),
        "retained_method_ids": list(retained),
        "completed_dev_evaluation_ids": [item.cell_id for item in evaluations],
        "completed_dev_selection_ids": [item.selection_id for item in selections],
        "fold_ids": [1, 2, 3, 4, 5],
        "oot_accessed": False,
        "configuration_adaptation_after_dev": False,
    }
    write_json_atomic(_dev_completion_lock_path(plan), _artifact_with_hash(payload), overwrite=True)
    return {**result, "dev_completion_lock": str(_dev_completion_lock_path(plan))}


def _resolved_execution(plan: CombinationPlan) -> tuple[Any, Any]:
    _git_clean(plan.repository_root)
    validate_prompt_10_baselines(plan)
    plan.results_root.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.gettempdir()).resolve()
    configured = load_execution_policy(plan.repository_root, plan.policy_path)
    capacity = detect_hardware(plan.results_root, temp_root)
    policy = resolve_execution_policy(configured, capacity)
    parallel = policy.parallelism
    if (parallel.concurrent_experiment_runs, parallel.concurrent_folds, parallel.data_loader_workers) != (1, 1, 0) or parallel.estimator_threads > 4:
        raise CombinationPipelineError("execution policy widened the frozen sequential resource limits")
    preflight = run_preflight(
        repository_root=plan.repository_root,
        config_path=plan.policy_path,
        results_root=plan.results_root,
        temp_root=temp_root,
        requested_accelerator="cpu",
        capacity=capacity,
    )
    if preflight.get("status") != "pass":
        raise CombinationPipelineError(f"OOT preflight failed: {preflight.get('blocking_reasons')}")
    from credit_risk_fs.experiments.ram_control import ResolvedRamControlPolicy

    return policy, ResolvedRamControlPolicy(**dict(preflight["ram_control_policy"]))


def execute_oot(plan: CombinationPlan) -> dict[str, Any]:
    approval = _validate_approval_lock(plan)
    _validate_dev_completion_lock(plan, approval)
    validate_prompt_10_baselines(plan)
    retained = tuple(approval["retained_method_ids"])
    selections, evaluations = build_phase_matrix(
        plan, phase="oot", retained_method_ids=retained
    )
    policy, ram_control = _resolved_execution(plan)
    parallel = policy.parallelism
    completed = 0
    for cell in evaluations:
        state_path = _evaluation_path(plan, cell.cell_id, "oot")
        valid, _, _ = _validate_artifact(
            state_path,
            {"cell_id": cell.cell_id, "configuration_sha256": plan.configuration_sha256},
        )
        if valid:
            completed += 1
            continue
        selection = next(item for item in selections if item.selection_id == cell.selection_id)
        selection_path = _selection_path(plan, selection.selection_id, "oot")
        selected_valid, _, selected_payload = _validate_artifact(
            selection_path,
            {"selection_id": selection.selection_id, "configuration_sha256": plan.configuration_sha256},
        )
        if not selected_valid:
            selected_result = _run_supervised(
                plan,
                target="credit_risk_fs.experiments.selector_combinations:combination_selection_worker",
                kwargs={
                    "repository_root": str(plan.repository_root),
                    "spec": asdict(selection),
                    "protocol_lock_sha256": plan.protocol_lock_sha256,
                    "estimator_threads": parallel.estimator_threads,
                    "phase": "oot",
                },
                association=selection.selection_id,
                wall=selection.wall_clock_limit_seconds,
                policy=policy,
                ram_control=ram_control,
            )
            payload = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "terminal_state": "completed" if selected_result.status == "completed" else selected_result.status,
                "selection_id": selection.selection_id,
                "configuration_sha256": plan.configuration_sha256,
                "protocol_lock_sha256": plan.protocol_lock_sha256,
                "selection_spec": asdict(selection),
                "worker_result": selected_result.return_value,
                "supervisor": _supervisor_payload(selected_result),
            }
            if selected_result.status == "completed":
                payload["artifact_files"] = _publish_selection_files(
                    selection_path, selected_result.return_value
                )
            _publish_state(plan, selection_path, payload)
            if selected_result.status != "completed":
                return {
                    "status": selected_result.status,
                    "stop_code": selected_result.stop_code,
                    "phase": "oot",
                    "completed_evaluations": completed,
                    "stop_cell": cell.cell_id,
                }
            selected_payload = _artifact_with_hash(payload)
        selected = list(selected_payload["worker_result"]["selected_features"])
        result = _run_supervised(
            plan,
            target="credit_risk_fs.experiments.selector_combinations:combination_oot_evaluation_worker",
            kwargs={
                "repository_root": str(plan.repository_root),
                "cell": asdict(cell),
                "selected_features": selected,
                "estimator_threads": parallel.estimator_threads,
                "output_state_path": str(state_path),
            },
            association=cell.cell_id,
            wall=cell.wall_clock_limit_seconds,
            policy=policy,
            ram_control=ram_control,
        )
        payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "terminal_state": "completed" if result.status == "completed" else result.status,
            "cell_id": cell.cell_id,
            "selection_id": cell.selection_id,
            "configuration_sha256": plan.configuration_sha256,
            "protocol_lock_sha256": plan.protocol_lock_sha256,
            "evaluation_cell": asdict(cell),
            "selection_artifact_sha256": sha256_file(selection_path),
            "worker_result": result.return_value,
            "supervisor": _supervisor_payload(result),
            "artifact_files": (
                list(result.return_value["artifact_files"])
                if result.status == "completed"
                else []
            ),
        }
        _publish_state(plan, state_path, payload)
        if result.status != "completed":
            return {
                "status": result.status,
                "stop_code": result.stop_code,
                "phase": "oot",
                "completed_evaluations": completed,
                "stop_cell": cell.cell_id,
            }
        completed += 1
    return {
        "status": "completed",
        "phase": "oot",
        "completed_evaluations": completed,
        "expected_evaluations": len(evaluations),
        "configuration_adaptation_after_oot": False,
    }


def enforce_phase_gate(plan: CombinationPlan, phase: str) -> None:
    """Artifact-only gate probe retained for status/tests; execution uses execute_dev/execute_oot."""

    approval = _validate_approval_lock(plan)
    if phase == "dev":
        return
    if phase == "oot":
        _validate_dev_completion_lock(plan, approval)
        validate_prompt_10_baselines(plan)
        return
    raise ValueError("phase gate accepts dev or oot")


def validate_completed_artifacts(plan: CombinationPlan) -> dict[str, Any]:
    status = build_status(plan)
    phase_states = [status["pilot"], status.get("dev"), status.get("oot")]
    valid = True
    for item in phase_states:
        if item is None:
            continue
        observed = int(item["authenticated_evaluations"])
        expected = int(item["expected_evaluations"])
        if observed not in {0, expected}:
            valid = False
    return {
        "status": "valid" if valid else "partial_or_corrupt",
        "pilot": status["pilot"],
        "dev": status.get("dev"),
        "oot": status.get("oot"),
        "dev_gate": status["dev_gate"],
        "oot_gate": status["oot_gate"],
        "raw_dataset_paths_resolved": False,
        "workers_started": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--status", action="store_true")
    mode.add_argument("--validate-baselines", action="store_true")
    mode.add_argument("--phase", choices=("pilot", "dev", "oot"))
    mode.add_argument("--validate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = load_combination_plan(args.repository_root, args.config)
    if args.plan:
        result = render_plan(plan)
    elif args.status:
        result = build_status(plan)
    elif args.validate_baselines:
        result = validate_prompt_10_baselines(plan)
    elif args.validate:
        result = validate_completed_artifacts(plan)
    else:
        command_arguments = list(sys.argv[1:] if argv is None else argv)
        with ResearchLogSession(
            plan.log_path,
            repository_root=plan.repository_root,
            command_arguments=command_arguments,
        ) as session:
            try:
                if args.phase == "pilot":
                    result = execute_pilot(plan)
                elif args.phase == "dev":
                    result = execute_dev(plan)
                elif args.phase == "oot":
                    result = execute_oot(plan)
                else:
                    raise AssertionError("unreachable phase")
            except CombinationGateClosed as exc:
                result = {"status": "gated", "phase": args.phase, "reason": str(exc)}
                session.finish(
                    "session_gated",
                    level="ERROR",
                    message=str(exc),
                    stop_code="phase_gate_closed",
                )
                print(json.dumps(result, indent=2))
                return 2
            except KeyboardInterrupt:
                session.finish(
                    "session_interrupted",
                    level="ERROR",
                    message="Combination research interrupted manually; authenticated boundary retained",
                    exception_class="KeyboardInterrupt",
                    stop_code=MANUAL_INTERRUPT,
                )
                return 130
            except BaseException as exc:
                session.finish(
                    "session_failed",
                    level="ERROR",
                    message=f"Combination research failed: {type(exc).__name__}: {exc}",
                    exception_class=type(exc).__name__,
                    traceback=traceback.format_exc(),
                )
                return 1
            if result.get("status") == "completed":
                session.finish(
                    "session_completed",
                    message=f"Combination {args.phase} phase completed and authenticated",
                    phase=args.phase,
                )
            else:
                session.finish(
                    "session_controlled_stop",
                    level="ERROR",
                    message=(
                        f"Combination {args.phase} phase stopped at "
                        f"{result.get('stop_cell')}: {result.get('status')} "
                        f"({result.get('stop_code')})"
                    ),
                    phase=args.phase,
                    stop_code=result.get("stop_code"),
                )
        print(json.dumps(result, indent=2, default=str))
        if result.get("status") == "interrupted" and result.get("stop_code") == MANUAL_INTERRUPT:
            return 130
        return 0 if result.get("status") == "completed" else 1
    print(json.dumps(result, indent=2, default=str))
    return 0


__all__ = [
    "CombinationGateClosed", "CombinationPipelineError", "CombinationPlan", "EvaluationCell",
    "SelectionSpec", "build_phase_matrix", "build_status", "combination_evaluation_worker",
    "combination_oot_evaluation_worker", "combination_selection_worker", "enforce_phase_gate",
    "execute_dev", "execute_oot", "execute_pilot", "load_combination_plan", "main", "render_plan",
    "validate_completed_artifacts", "validate_prompt_10_baselines",
]
