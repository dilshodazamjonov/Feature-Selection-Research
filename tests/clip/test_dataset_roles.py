from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from credit_risk_fs.clip.pair_validation import (
    validate_contrastive_config,
    validate_group_split,
)
from credit_risk_fs.clip.reverse_transfer import DatasetRoles


def _roles(training: str, external: str) -> DatasetRoles:
    return DatasetRoles(
        training_dataset=training,
        external_dataset=external,
        training_feature_manifest="training.csv",
        external_feature_manifest="external.csv",
        training_raw_statistical_source="training.parquet",
        external_raw_statistical_source="external.parquet",
        training_statistical_fit_scope="dev_training_features_only",
        external_statistical_transform_scope="transform_only",
    )


def test_both_dataset_directions_are_supported() -> None:
    assert _roles("homecredit", "lendingclub_v2").manifest()["source_domain"] == "homecredit"
    assert _roles("lendingclub_v2", "homecredit").manifest()["external_domain"] == "homecredit"


def test_same_dataset_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be different"):
        _roles("homecredit", "homecredit").validate()


def test_contrastive_validator_uses_declared_roles_not_names() -> None:
    config = SimpleNamespace(
        training_dataset="lendingclub_v2",
        external_validation_dataset="homecredit",
        legacy_lendingclub_allowed=False,
        training_statistical_fit_scope="dev_training_features_only",
        external_statistical_transform_scope="transform_only",
        negative_policy={
            "explicit_hard_negatives_enabled": False,
            "cross_dataset_negatives_enabled": False,
            "validation_as_training_negative": False,
        },
        tensor_schema={
            "stable_core_as_input": False,
            "llm_rank_as_input": False,
            "oot_allowed": False,
            "psi_allowed": False,
            "target_allowed": False,
        },
    )
    assert validate_contrastive_config(config) == []


def test_group_split_validates_declared_dataset() -> None:
    split = pd.DataFrame(
        {
            "dataset": ["lendingclub_v2", "lendingclub_v2"],
            "feature_name": ["a", "b"],
            "split": ["train", "validation"],
            "group_key": ["g1", "g2"],
            "group_source": ["identity", "identity"],
        }
    )
    assert validate_group_split(split, dataset="lendingclub_v2") == []
    assert validate_group_split(split, dataset="homecredit")

