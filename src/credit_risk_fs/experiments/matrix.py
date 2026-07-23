from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping
import hashlib

import yaml


MODELS = ["lr", "catboost"]
STAT_SELECTORS = ["mrmr", "boruta", "pca", "domain_rule_baseline"]
HYBRID_VARIANTS = [
    ("mrmr", "llm_then_mrmr", "llm_then_mrmr", "hybrid_mrmr"),
    ("boruta", "llm_then_boruta", "llm_then_boruta", "hybrid_boruta"),
    (
        "stable_core_llm_fill",
        "stable_core_llm_fill",
        "stable_core_llm_fill",
        "hybrid_stable_core_llm_fill",
    ),
]

LLM_SELECTOR = "llm"
EXPERIMENT_TYPES = {"statistical", "llm", "hybrid"}


@dataclass(frozen=True, slots=True)
class MatrixRunSpec:
    """One atomic experiment in the full research matrix."""

    model: str
    selector: str
    experiment_type: str
    experiment_name: str
    selector_name: str
    output_bucket: str

    @property
    def run_label(self) -> str:
        return f"{self.model}_{self.experiment_name}"


def iter_matrix() -> Iterator[MatrixRunSpec]:
    """Yield the full model/selector matrix in a stable, explicit order."""
    for model in MODELS:
        for selector in STAT_SELECTORS:
            yield MatrixRunSpec(
                model=model,
                selector=selector,
                experiment_type="statistical",
                experiment_name=selector,
                selector_name=selector,
                output_bucket="statistical",
            )

        yield MatrixRunSpec(
            model=model,
            selector=LLM_SELECTOR,
            experiment_type="llm",
            experiment_name=LLM_SELECTOR,
            selector_name=LLM_SELECTOR,
            output_bucket="llm",
        )

        for selector, experiment_name, selector_name, output_bucket in HYBRID_VARIANTS:
            yield MatrixRunSpec(
                model=model,
                selector=selector,
                experiment_type="hybrid",
                experiment_name=experiment_name,
                selector_name=selector_name,
                output_bucket=output_bucket,
            )


def validate_matrix() -> None:
    """Fail fast if the explicit matrix constants drift into invalid values."""
    if sorted(set(MODELS)) != sorted(MODELS):
        raise ValueError("MODELS contains duplicates.")
    if sorted(set(STAT_SELECTORS)) != sorted(STAT_SELECTORS):
        raise ValueError("STAT_SELECTORS contains duplicates.")

    hybrid_selector_names = [selector for selector, *_ in HYBRID_VARIANTS]
    if sorted(set(hybrid_selector_names)) != sorted(hybrid_selector_names):
        raise ValueError("HYBRID_VARIANTS contains duplicate selector ids.")

    if not {"mrmr", "boruta"}.issubset(set(STAT_SELECTORS)):
        raise ValueError("Statistical baselines must include mrmr and boruta.")


@dataclass(frozen=True, slots=True)
class CrossDatasetVotingRunSpec:
    """One frozen prospective cross-dataset voting-matrix entry."""

    run_id: str
    execution_order: int
    dataset: str
    model: str
    method_id: str
    candidate_pool_budget: int | None
    final_feature_budget: int
    designation: str
    comparison_family: str
    reference_method: str
    enabled: bool


FROZEN_CROSS_DATASET_COUNTS = {
    "voting_runs": 12,
    "reference_reruns": 4,
    "total_registered_runs": 16,
    "dev_folds_per_run": 5,
    "voting_dev_fold_executions": 60,
    "reference_dev_fold_executions": 20,
    "total_dev_fold_executions": 80,
    "final_full_dev_oot_fits": 16,
    "primary_voting_runs": 4,
    "sensitivity_voting_runs": 8,
    "primary_comparisons": 4,
    "sensitivity_comparisons": 8,
}

FROZEN_PILOT_IDS = (
    "cdv1-pilot-001-homecredit-voting-k200-lr-s42-f0",
    "cdv1-pilot-002-homecredit-voting-k200-catboost-s42-f0",
    "cdv1-pilot-003-lendingclub-v2-voting-k200-lr-s42-f0",
    "cdv1-pilot-004-lendingclub-v2-voting-k200-catboost-s42-f0",
)


@dataclass(frozen=True, slots=True)
class CrossDatasetVotingPilotSpec:
    run_id: str
    execution_order: int
    dataset: str
    model: str
    final_feature_budget: int
    candidate_pool_budget: int
    seed: int
    fold_id: int


@dataclass(frozen=True, slots=True)
class LendingClubMemoryCapacityScenarioSpec:
    scenario_id: str
    execution_order: int
    dataset: str
    mode: str
    fold_id: int | None
    candidate_pool: int
    seed: int
    branches: tuple[str, ...]
    load_oot: bool
    research_eligible: bool
    comparison_eligible: bool


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"matrix configuration must be a mapping: {candidate}")
    return dict(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expand_lendingclub_memory_capacity_scenarios(
    path: str | Path,
) -> tuple[LendingClubMemoryCapacityScenarioSpec, ...]:
    """Validate the separately versioned non-research mechanics scenarios."""

    candidate = Path(path).resolve()
    payload = _load_yaml_mapping(candidate)
    if payload.get("schema_version") != "lendingclub_memory_safe_refinement_v1":
        raise ValueError("unsupported LendingClub memory refinement schema")
    if payload.get("purpose") != "memory_capacity_validation":
        raise ValueError("memory refinement purpose drifted")
    for key in ("research_eligible", "comparison_eligible", "load_oot", "oot_scored"):
        if payload.get(key) is not False:
            raise ValueError(f"memory refinement must keep {key}=false")
    root = candidate.parents[2]
    parents = payload.get("parents")
    if not isinstance(parents, Mapping) or len(parents) != 4:
        raise ValueError("memory refinement parent references are incomplete")
    for name, values in parents.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"memory refinement parent is invalid: {name}")
        parent_path = root / str(values.get("path", ""))
        if not parent_path.is_file() or _sha256_file(parent_path) != values.get("sha256"):
            raise ValueError(f"memory refinement frozen parent hash mismatch: {name}")
    telemetry = payload.get("telemetry")
    if not isinstance(telemetry, Mapping) or {
        "process_tree_rss_warning_gib": telemetry.get("process_tree_rss_warning_gib"),
        "process_tree_rss_abort_gib": telemetry.get("process_tree_rss_abort_gib"),
        "system_available_ram_abort_below_gib": telemetry.get(
            "system_available_ram_abort_below_gib"
        ),
        "estimator_threads_maximum": telemetry.get("estimator_threads_maximum"),
        "concurrent_runs": telemetry.get("concurrent_runs"),
        "concurrent_folds": telemetry.get("concurrent_folds"),
        "data_loader_workers": telemetry.get("data_loader_workers"),
        "gpu": telemetry.get("gpu"),
    } != {
        "process_tree_rss_warning_gib": 24,
        "process_tree_rss_abort_gib": 28,
        "system_available_ram_abort_below_gib": 8,
        "estimator_threads_maximum": 4,
        "concurrent_runs": 1,
        "concurrent_folds": 1,
        "data_loader_workers": 0,
        "gpu": "disabled",
    }:
        raise ValueError("memory refinement widened the frozen resource policy")
    numeric = payload.get("numeric_semantics")
    if not isinstance(numeric, Mapping) or numeric.get("selector_effective_dtype") != "float32" or numeric.get(
        "rfe_effective_dtype"
    ) != "float32" or numeric.get("precision_change_allowed") is not False:
        raise ValueError("memory refinement changed or weakened numeric precision")
    publication = payload.get("publication")
    if (
        not isinstance(publication, Mapping)
        or not str(publication.get("capacity_results_root", "")).startswith(
            "cleanup/audits/lendingclub_memory_refinement_capacity_gate/"
        )
        or publication.get("canonical_results_registration_allowed") is not False
        or publication.get("implicit_all_column_requests_allowed") is not False
        or publication.get("unapproved_fallbacks_allowed") is not False
    ):
        raise ValueError("memory refinement publication boundary is invalid")
    scenarios = payload.get("scenarios")
    expected_ids = (
        "cdv1-equivalence-001-lendingclub-v2-first-fold-voting-k200-s42",
        "cdv1-capacity-001-lendingclub-v2-largest-fold-voting-k300-s42",
        "cdv1-capacity-002-lendingclub-v2-full-dev-voting-k300-s42",
    )
    if not isinstance(scenarios, list) or tuple(
        str(item.get("scenario_id")) for item in scenarios
    ) != expected_ids:
        raise ValueError("memory refinement scenario IDs/order drifted")
    specs = []
    for order, values in enumerate(scenarios, start=1):
        if not isinstance(values, Mapping) or values.get("execution_order") != order:
            raise ValueError("memory refinement execution order is invalid")
        mode = str(values.get("mode"))
        fold_value = values.get("fold_id")
        fold_id = None if fold_value is None else int(fold_value)
        pool = int(values.get("candidate_pool", -1))
        if mode not in {"fold", "full_dev"} or pool not in {200, 300}:
            raise ValueError("memory refinement scenario shape is invalid")
        if mode == "fold" and fold_id not in range(1, 6):
            raise ValueError("memory refinement fold is invalid")
        if mode == "full_dev" and fold_id is not None:
            raise ValueError("full-DEV memory scenario cannot name a fold")
        if (
            values.get("dataset") != "lendingclub_v2"
            or int(values.get("seed", -1)) != 42
            or list(values.get("branches", [])) != ["lr", "catboost"]
            or values.get("load_oot") is not False
            or values.get("research_eligible") is not False
            or values.get("comparison_eligible") is not False
        ):
            raise ValueError("memory refinement scientific/eligibility invariant drifted")
        specs.append(
            LendingClubMemoryCapacityScenarioSpec(
                scenario_id=str(values["scenario_id"]),
                execution_order=order,
                dataset="lendingclub_v2",
                mode=mode,
                fold_id=fold_id,
                candidate_pool=pool,
                seed=42,
                branches=("lr", "catboost"),
                load_oot=False,
                research_eligible=False,
                comparison_eligible=False,
            )
        )
    return tuple(specs)


def expand_cross_dataset_voting_matrix(
    path: str | Path,
) -> tuple[CrossDatasetVotingRunSpec, ...]:
    """Purely expand and validate the frozen 16-entry matrix.

    This function performs no result-layout initialization and no writes, so it
    is safe to use for an authorization-neutral dry expansion.
    """

    payload = _load_yaml_mapping(path)
    if payload.get("schema_version") != "cross_dataset_rank_voting_matrix_v1":
        raise ValueError("unsupported cross-dataset voting matrix schema")
    if payload.get("status") != "specification_only_not_authorized_for_execution":
        raise ValueError("research matrix authorization status changed unexpectedly")
    if payload.get("historical_replication") is not False:
        raise ValueError("prospective voting matrix must not claim historical replication")
    shared = payload.get("shared")
    if not isinstance(shared, Mapping):
        raise ValueError("matrix shared section is missing")
    if int(shared.get("master_seed", -1)) != 42:
        raise ValueError("cross-dataset voting matrix requires frozen seed 42")
    if list(shared.get("candidate_pool_budgets", [])) != [100, 200, 300]:
        raise ValueError("cross-dataset voting matrix candidate budgets drifted")
    if list(shared.get("voter_ids", [])) != ["rf_corr_mrmr", "boruta"]:
        raise ValueError("cross-dataset voting matrix voter set drifted")

    counts = payload.get("expected_counts")
    if not isinstance(counts, Mapping) or {
        key: int(counts.get(key, -1)) for key in FROZEN_CROSS_DATASET_COUNTS
    } != FROZEN_CROSS_DATASET_COUNTS:
        raise ValueError("cross-dataset voting matrix expected counts drifted")
    order = list(payload.get("run_order", []))
    runs = payload.get("runs")
    if not isinstance(runs, Mapping) or len(order) != 16 or len(set(order)) != 16:
        raise ValueError("cross-dataset voting matrix must contain 16 unique ordered IDs")
    if set(order) != set(runs):
        raise ValueError("matrix run_order and runs keys differ")

    specs: list[CrossDatasetVotingRunSpec] = []
    for expected_order, run_id in enumerate(order, start=1):
        values = runs[run_id]
        if not isinstance(values, Mapping):
            raise ValueError(f"matrix run {run_id} must be a mapping")
        if values.get("proposed_run_id") != run_id:
            raise ValueError(f"matrix run ID mismatch for {run_id}")
        dataset = str(values.get("dataset"))
        model = str(values.get("model"))
        method = str(values.get("method_id"))
        budget_value = values.get("candidate_pool_budget")
        budget = None if budget_value is None else int(budget_value)
        final_budget = int(values.get("final_feature_budget", -1))
        if dataset not in {"homecredit", "lendingclub_v2"}:
            raise ValueError(f"unknown matrix dataset: {dataset}")
        if model not in {"lr", "catboost"}:
            raise ValueError(f"unknown matrix model: {model}")
        if method not in {"rank_voting_v1", "rf_corr_mrmr"}:
            raise ValueError(f"unknown matrix method: {method}")
        if method == "rank_voting_v1" and budget not in {100, 200, 300}:
            raise ValueError(f"unsupported voting budget for {run_id}: {budget}")
        if method == "rf_corr_mrmr" and budget is not None:
            raise ValueError(f"reference run must not invent a voting budget: {run_id}")
        expected_final = 20 if model == "lr" else 40
        if final_budget != expected_final:
            raise ValueError(f"final feature budget drifted for {run_id}")
        if int(values.get("master_seed", -1)) != 42 or int(
            values.get("model_seed", -1)
        ) != 42 or int(values.get("selector_seed", -1)) != 42:
            raise ValueError(f"non-frozen seed in {run_id}")
        if int(values.get("execution_order", -1)) != expected_order:
            raise ValueError(f"execution order mismatch for {run_id}")
        specs.append(
            CrossDatasetVotingRunSpec(
                run_id=run_id,
                execution_order=expected_order,
                dataset=dataset,
                model=model,
                method_id=method,
                candidate_pool_budget=budget,
                final_feature_budget=final_budget,
                designation=str(values.get("designation")),
                comparison_family=str(values.get("comparison_family")),
                reference_method=str(values.get("reference_method")),
                enabled=bool(values.get("enabled")),
            )
        )
    if not all(item.enabled for item in specs):
        raise ValueError("all frozen prospective matrix entries must remain enabled")
    return tuple(specs)


def cross_dataset_matrix_expansion_summary(
    specs: tuple[CrossDatasetVotingRunSpec, ...],
) -> dict[str, Any]:
    voting = [item for item in specs if item.method_id == "rank_voting_v1"]
    references = [item for item in specs if item.method_id == "rf_corr_mrmr"]
    primary = [item for item in voting if item.candidate_pool_budget == 200]
    sensitivity = [item for item in voting if item.candidate_pool_budget in {100, 300}]
    observed = {
        "voting_runs": len(voting),
        "reference_reruns": len(references),
        "total_registered_runs": len(specs),
        "dev_folds_per_run": 5,
        "voting_dev_fold_executions": len(voting) * 5,
        "reference_dev_fold_executions": len(references) * 5,
        "total_dev_fold_executions": len(specs) * 5,
        "final_full_dev_oot_fits": len(specs),
        "primary_voting_runs": len(primary),
        "sensitivity_voting_runs": len(sensitivity),
        "primary_comparisons": len(primary),
        "sensitivity_comparisons": len(sensitivity),
    }
    if observed != FROZEN_CROSS_DATASET_COUNTS:
        raise ValueError(f"expanded matrix counts drifted: {observed}")
    return {"run_ids": [item.run_id for item in specs], **observed}


def expand_cross_dataset_voting_pilot(
    path: str | Path,
) -> tuple[CrossDatasetVotingPilotSpec, ...]:
    payload = _load_yaml_mapping(path)
    if payload.get("schema_version") != "cross_dataset_rank_voting_pilot_v1":
        raise ValueError("unsupported cross-dataset voting pilot schema")
    if payload.get("purpose") != "integration_resource_pilot":
        raise ValueError("pilot purpose drifted")
    if payload.get("research_eligible") is not False or payload.get(
        "comparison_eligible"
    ) is not False:
        raise ValueError("pilot must be ineligible for research and comparison")
    shared = payload.get("shared")
    jobs = payload.get("jobs")
    if not isinstance(shared, Mapping) or not isinstance(jobs, list):
        raise ValueError("pilot shared/jobs sections are invalid")
    expected_shared = {
        "method": "rank_voting_v1",
        "candidate_pool": 200,
        "seed": 42,
        "fold_count": 1,
        "canonical_fold": 1,
        "load_oot": False,
        "final_refit": False,
        "accelerator": "cpu",
        "concurrent_experiment_runs": 1,
        "concurrent_folds": 1,
        "data_loader_workers": 0,
        "maximum_estimator_threads": 4,
    }
    for key, expected in expected_shared.items():
        if shared.get(key) != expected:
            raise ValueError(f"pilot shared setting drifted: {key}")
    if len(jobs) != 4 or tuple(job.get("run_id") for job in jobs) != FROZEN_PILOT_IDS:
        raise ValueError("pilot must contain the four exact authorized IDs in order")
    specs: list[CrossDatasetVotingPilotSpec] = []
    for order, job in enumerate(jobs, start=1):
        if not isinstance(job, Mapping) or int(job.get("execution_order", -1)) != order:
            raise ValueError("pilot execution order is invalid")
        dataset = str(job.get("dataset"))
        model = str(job.get("model"))
        if dataset not in {"homecredit", "lendingclub_v2"} or model not in {
            "lr",
            "catboost",
        }:
            raise ValueError("pilot contains an unknown dataset/model")
        final_budget = int(job.get("final_feature_budget", -1))
        if final_budget != (20 if model == "lr" else 40):
            raise ValueError("pilot final feature budget drifted")
        specs.append(
            CrossDatasetVotingPilotSpec(
                run_id=str(job["run_id"]),
                execution_order=order,
                dataset=dataset,
                model=model,
                final_feature_budget=final_budget,
                candidate_pool_budget=200,
                seed=42,
                fold_id=1,
            )
        )
    return tuple(specs)
