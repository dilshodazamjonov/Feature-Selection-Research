from __future__ import annotations

import pandas as pd

from credit_risk_fs.clip.statistical_baseline import load_statistical_baseline_config
from credit_risk_fs.clip.statistical_fields import (
    assign_statistical_field_role,
    build_statistical_field_inventory,
    main_statistical_fields,
)


def test_only_approved_target_free_field_enters_main_statistical_view():
    config = load_statistical_baseline_config()
    home = pd.read_csv("results/homecredit/analysis/clip_readiness/dev_only_clip_training_evidence.csv")
    lc = pd.read_csv("results/lendingclub_v2/analysis/clip_readiness/dev_only_clip_training_evidence.csv")
    train = pd.read_csv("results/clip/dry_run/training_features.csv")
    external = pd.read_csv("results/clip/dry_run/external_validation_features.csv")

    inventory = build_statistical_field_inventory(
        config=config,
        homecredit_source=home,
        lendingclub_source=lc,
        training_features=train,
        external_validation_features=external,
    )

    assert main_statistical_fields(inventory) == ["missing_rate_dev"]
    included = inventory[inventory["included_in_main_statistical_view"].astype(bool)]
    assert set(included["field_name"]) == {"missing_rate_dev"}
    assert not included["target_aware"].any()
    assert not included["algorithm_derived"].any()


def test_forbidden_and_anchor_fields_are_not_main_inputs():
    config = load_statistical_baseline_config()

    for field in [
        "llm_best_rank",
        "llm_mean_rank_if_available",
        "selected_by_llm",
        "selected_by_stable_core_llm_fill",
        "oot_auc",
        "psi_median",
        "target",
        "loan_id",
        "prediction_score",
        "fold_id",
        "clip_training_split",
    ]:
        role, reason, risk, _ = assign_statistical_field_role(field, config)
        assert role == "forbidden", (field, role, reason, risk)

    role, reason, _, _ = assign_statistical_field_role("stable_core_membership", config)
    assert role == "anchor_only"
    assert "anchor" in reason


def test_algorithm_and_target_aware_fields_default_to_optional_ablation():
    config = load_statistical_baseline_config()

    for field in [
        "iv_score_if_available",
        "bootstrap_selection_frequency_if_available",
        "mrmr_selection_frequency",
        "boruta_selection_frequency",
    ]:
        role, reason, _, _ = assign_statistical_field_role(field, config)
        assert role == "optional_ablation_input", (field, role, reason)
