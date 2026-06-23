from __future__ import annotations

import pandas as pd

from credit_risk_fs.clip_final_comparison.constants import (
    ABLATIONS,
    CENTRAL_POOL_MULTIPLIER,
    CLIP_V2_SEEDS,
    DATASETS,
    MODEL_BUDGETS,
    MODELS,
    POOL_MULTIPLIERS,
    RANDOM_SEEDS,
    SCREENING_METHODS,
)


def pool_size(model: str, multiplier: int, eligible_count: int | None = None) -> int:
    size = MODEL_BUDGETS[model] * int(multiplier)
    return min(size, int(eligible_count)) if eligible_count is not None else size


def build_core_experiment_plan() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for model in MODELS:
            final_budget = MODEL_BUDGETS[model]
            for method in SCREENING_METHODS:
                for multiplier in POOL_MULTIPLIERS:
                    rows.append(
                        _row(
                            dataset,
                            model,
                            method,
                            multiplier,
                            pool_size(model, multiplier),
                            final_budget,
                            seed=None,
                            experiment_family="core_candidate_pool",
                        )
                    )
            for multiplier in POOL_MULTIPLIERS:
                for seed in RANDOM_SEEDS:
                    rows.append(
                        _row(
                            dataset,
                            model,
                            "random",
                            multiplier,
                            pool_size(model, multiplier),
                            final_budget,
                            seed=seed,
                            experiment_family="random_candidate_pool",
                        )
                    )
            rows.append(
                _row(
                    dataset,
                    model,
                    "full_mrmr",
                    None,
                    None,
                    final_budget,
                    seed=None,
                    experiment_family="full_mrmr_reference",
                )
            )
    return pd.DataFrame(rows)


def build_seed_downstream_plan() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            for seed in CLIP_V2_SEEDS:
                rows.append(
                    _row(
                        dataset,
                        model,
                        "clip_v2",
                        CENTRAL_POOL_MULTIPLIER,
                        pool_size(model, CENTRAL_POOL_MULTIPLIER),
                        MODEL_BUDGETS[model],
                        seed=seed,
                        experiment_family="representation_seed_downstream",
                    )
                )
                rows[-1]["checkpoint_seed"] = seed
    return pd.DataFrame(rows)


def build_ablation_plan() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            for ablation in ABLATIONS:
                row = _row(
                    dataset,
                    model,
                    "clip_v2",
                    CENTRAL_POOL_MULTIPLIER,
                    pool_size(model, CENTRAL_POOL_MULTIPLIER),
                    MODEL_BUDGETS[model],
                    seed=None,
                    experiment_family="representation_ablation",
                )
                row["ablation"] = ablation
                row["run_id"] = f"{dataset}_{model}_{ablation}_5x"
                rows.append(row)
    return pd.DataFrame(rows)


def planned_matrix_summary() -> dict[str, int]:
    core = build_core_experiment_plan()
    random_runs = int(core["screening_method"].eq("random").sum())
    return {
        "core_candidate_pool_runs": int(len(core)),
        "random_runs_included_in_core_total": random_runs,
        "deterministic_screened_and_full_runs": int(len(core) - random_runs),
        "core_runs_per_dataset_model": 46,
        "representation_seed_downstream_runs": int(len(build_seed_downstream_plan())),
        "ablation_downstream_runs": int(len(build_ablation_plan())),
        "ablation_training_jobs": int(len(ABLATIONS)),
    }


def _row(
    dataset: str,
    model: str,
    method: str,
    multiplier: int | None,
    candidate_pool_size: int | None,
    final_budget: int,
    *,
    seed: int | None,
    experiment_family: str,
) -> dict[str, object]:
    size_part = f"{multiplier}x" if multiplier is not None else "full"
    seed_part = f"_seed{seed}" if seed is not None else ""
    run_id = f"{dataset}_{model}_{method}_{size_part}{seed_part}"
    return {
        "run_id": run_id,
        "dataset": dataset,
        "model": model,
        "screening_method": method,
        "pool_multiplier": multiplier,
        "candidate_pool_size": candidate_pool_size,
        "final_feature_budget": final_budget,
        "random_seed": seed,
        "experiment_family": experiment_family,
        "uses_mrmr": True,
        "dev_only_selection": True,
        "oot_allowed_for_selection": False,
        "execution_status": "planned",
    }
