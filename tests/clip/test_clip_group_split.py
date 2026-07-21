from __future__ import annotations

import pandas as pd

from credit_risk_fs.clip.feature_family import derive_canonical_feature_family
from credit_risk_fs.clip.group_split import build_group_split


def test_group_split_has_no_group_overlap_and_is_deterministic():
    frame = pd.DataFrame(
        {
            "dataset": ["homecredit"] * 6,
            "feature_name": [
                "BURO_AMT_SUM",
                "BURO_AMT_MEAN",
                "PREV_AMT_SUM",
                "PREV_AMT_MEAN",
                "AMT_CREDIT",
                "AMT_ANNUITY",
            ],
            "source_table": ["bureau", "bureau", "previous", "previous", "application", "application"],
            "semantic_group": ["a", "a", "b", "b", "c", "c"],
        }
    )

    first = build_group_split(frame, seed=42, validation_fraction=0.33)
    second = build_group_split(frame, seed=42, validation_fraction=0.33)

    assert first.split.equals(second.split)
    train_groups = set(first.split.loc[first.split["split"].eq("train"), "group_key"])
    val_groups = set(first.split.loc[first.split["split"].eq("validation"), "group_key"])
    assert not train_groups.intersection(val_groups)
    assert first.audit["group_overlap_count"] == 0


def test_region_rating_alias_resolves_to_one_canonical_family():
    aliases = {"REGION_RATING_CLIENT_W_CITY": "REGION_RATING_CLIENT"}

    base = derive_canonical_feature_family("REGION_RATING_CLIENT", aliases=aliases)
    with_city = derive_canonical_feature_family("REGION_RATING_CLIENT_W_CITY", aliases=aliases)

    assert base.canonical_feature_family == "REGION_RATING_CLIENT"
    assert with_city.canonical_feature_family == "REGION_RATING_CLIENT"
    assert with_city.family_resolution_source == "configured_alias"


def test_canonical_family_split_keeps_all_members_together():
    frame = pd.DataFrame(
        {
            "dataset": ["homecredit"] * 4,
            "feature_name": [
                "REGION_RATING_CLIENT",
                "REGION_RATING_CLIENT_W_CITY",
                "REGION_RATING_CLIENT_OTHER",
                "REGION_POPULATION_RELATIVE",
            ],
            "source_table": ["application"] * 4,
            "semantic_group": ["application_amounts"] * 4,
        }
    )

    result = build_group_split(
        frame,
        seed=42,
        validation_fraction=0.5,
        derived_family_aliases={"REGION_RATING_CLIENT_W_CITY": "REGION_RATING_CLIENT"},
    )
    subset = result.split[result.split["canonical_feature_family"].eq("REGION_RATING_CLIENT")]

    assert subset["split"].nunique() == 1
    assert subset["feature_name"].nunique() == 2
    assert result.audit["train_validation_family_overlap_count"] == 0


def test_unrelated_prefix_sharing_features_are_not_automatically_grouped():
    aliases: dict[str, str] = {}

    first = derive_canonical_feature_family("REGION_RATING_CLIENT_OTHER", aliases=aliases)
    second = derive_canonical_feature_family("REGION_POPULATION_RELATIVE", aliases=aliases)

    assert first.canonical_feature_family != second.canonical_feature_family


def test_formula_derived_family_resolution_is_deterministic():
    first = derive_canonical_feature_family("BURO_AMT_CREDIT_SUM_MEAN", source_formula="mean(AMT_CREDIT_SUM)")
    second = derive_canonical_feature_family("BURO_AMT_CREDIT_SUM_MAX", source_formula="mean(AMT_CREDIT_SUM)")

    assert first == second


def test_real_homecredit_split_has_zero_canonical_family_overlap(legacy_artifact_path):
    split = pd.read_csv(
        legacy_artifact_path("results/clip/text_baseline/homecredit_group_split.csv")
    )

    train = set(split.loc[split["split"].eq("train"), "canonical_feature_family"].astype(str))
    validation = set(split.loc[split["split"].eq("validation"), "canonical_feature_family"].astype(str))
    region = split[split["feature_name"].isin(["REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY"])]

    assert not train.intersection(validation)
    assert region["split"].nunique() == 1
    assert region["canonical_feature_family"].nunique() == 1
