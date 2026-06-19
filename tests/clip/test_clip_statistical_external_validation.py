from __future__ import annotations

import pandas as pd

from credit_risk_fs.clip.statistical_baseline import build_statistical_baseline, load_statistical_baseline_config
from credit_risk_fs.clip.statistical_validation import validate_group_split_artifact


def test_legacy_lendingclub_is_rejected_for_statistical_baseline(tmp_path):
    config = load_statistical_baseline_config()
    bad = config.__class__(
        **{
            **config.__dict__,
            "external_validation_dataset": "lendingclub",
            "output_dir": tmp_path,
        }
    )

    try:
        build_statistical_baseline(config=bad, dry_run=True)
    except RuntimeError as exc:
        assert "lendingclub_v2" in str(exc) or "legacy" in str(exc)
    else:
        raise AssertionError("legacy LendingClub must be rejected")


def test_group_split_has_no_overlap_and_each_homecredit_feature_once():
    split = pd.read_csv("results/clip/text_baseline/homecredit_group_split.csv")
    audit = {
        "row_count": len(split),
        "train_rows": int(split["split"].eq("train").sum()),
        "validation_rows": int(split["split"].eq("validation").sum()),
        "group_overlap_count": 0,
    }
    training = pd.read_csv("results/clip/dry_run/training_features.csv")

    errors = validate_group_split_artifact(split=split, audit=audit, training_features=training)

    assert errors == []
    assert split["feature_name"].is_unique
    train_groups = set(split.loc[split["split"].eq("train"), "group_key"])
    validation_groups = set(split.loc[split["split"].eq("validation"), "group_key"])
    assert not train_groups.intersection(validation_groups)


def test_lendingclub_v2_is_transform_only_in_outputs(tmp_path):
    config = load_statistical_baseline_config()
    config = config.__class__(**{**config.__dict__, "output_dir": tmp_path})

    result = build_statistical_baseline(config=config, dry_run=False)

    lc_vectors = pd.read_parquet(result.output_paths["lendingclub_v2_statistical_vectors"])
    lc_rank = pd.read_csv(result.output_paths["lendingclub_v2_statistical_only_ranking"])

    assert set(lc_vectors["split"]) == {"external_validation"}
    assert set(lc_rank["split"]) == {"external_validation"}
    assert int(lc_rank["is_anchor_feature"].astype(bool).sum()) == 0
    assert lc_vectors["preprocessor_hash"].nunique() == 1
