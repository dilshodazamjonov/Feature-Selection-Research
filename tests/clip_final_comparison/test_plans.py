from __future__ import annotations

from credit_risk_fs.clip_final_comparison.constants import ABLATIONS, CLIP_V2_SEEDS, RANDOM_SEEDS
from credit_risk_fs.clip_final_comparison.plans import (
    build_ablation_plan,
    build_core_experiment_plan,
    build_seed_downstream_plan,
    planned_matrix_summary,
    pool_size,
)


def test_core_matrix_size_and_methods_are_prespecified():
    plan = build_core_experiment_plan()
    assert len(plan) == 184
    assert planned_matrix_summary()["core_runs_per_dataset_model"] == 46
    assert set(plan["screening_method"]) == {
        "clip_v2",
        "variance",
        "correlation_filter",
        "text_similarity",
        "statistics_only",
        "random",
        "full_mrmr",
    }
    per_panel = plan.groupby(["dataset", "model"]).size().to_dict()
    assert set(per_panel.values()) == {46}


def test_pool_size_rules_and_capping():
    assert [pool_size("lr", m) for m in [2, 5, 10]] == [40, 100, 200]
    assert [pool_size("catboost", m) for m in [2, 5, 10]] == [80, 200, 400]
    assert pool_size("catboost", 10, eligible_count=123) == 123


def test_random_seed_and_representation_seed_plans_are_complete():
    core = build_core_experiment_plan()
    random_rows = core[core["screening_method"].eq("random")]
    assert sorted(random_rows["random_seed"].dropna().astype(int).unique().tolist()) == list(RANDOM_SEEDS)
    assert len(random_rows) == 120

    seed_plan = build_seed_downstream_plan()
    assert len(seed_plan) == 20
    assert sorted(seed_plan["random_seed"].astype(int).unique().tolist()) == list(CLIP_V2_SEEDS)


def test_ablation_plan_uses_only_prespecified_groups():
    plan = build_ablation_plan()
    assert len(plan) == 28
    assert set(plan["ablation"]) == set(ABLATIONS)
    assert plan["pool_multiplier"].eq(5).all()

