from __future__ import annotations

import pandas as pd

from credit_risk_fs.clip.schemas import ClipDatasetRole, ClipFieldRole, ClipFieldSpec
from credit_risk_fs.clip.validation import (
    forbidden_field_matches,
    validate_dataset_roles,
    validate_deterministic_order,
    validate_field_role_separation,
)


def test_homecredit_train_lendingclub_v2_external_validation_roles():
    assert validate_dataset_roles("homecredit", "lendingclub_v2") == []
    assert validate_dataset_roles("lendingclub_v2", "homecredit")


def test_legacy_lendingclub_role_is_forbidden():
    errors = validate_dataset_roles("homecredit", "lendingclub")

    assert any("legacy" in error for error in errors)


def test_forbidden_pattern_registry_is_conservative_on_field_names_only():
    assert forbidden_field_matches("target")
    assert forbidden_field_matches("y_score")
    assert forbidden_field_matches("issue_date")
    assert not forbidden_field_matches("description")
    assert not forbidden_field_matches("clip_training_text")


def test_input_role_separation_rejects_target_and_id_inputs():
    specs = [
        ClipFieldSpec(
            dataset="homecredit",
            field_name="TARGET",
            detected_dtype="int64",
            field_role=ClipFieldRole.STATISTICAL_INPUT,
            allowed_in_main_training_input=True,
            reason="bad config",
        ),
        ClipFieldSpec(
            dataset="homecredit",
            field_name="member_id",
            detected_dtype="object",
            field_role=ClipFieldRole.ANCHOR_ONLY,
            allowed_in_main_training_input=True,
            reason="bad config",
        ),
    ]

    errors = validate_field_role_separation(specs)

    assert len(errors) == 2


def test_deterministic_order_validation():
    unsorted = pd.DataFrame({"dataset": ["b", "a"], "feature": ["z", "a"]})
    sorted_frame = pd.DataFrame({"dataset": ["a", "b"], "feature": ["a", "z"]})

    assert validate_deterministic_order(unsorted, ["dataset", "feature"])
    assert validate_deterministic_order(sorted_frame, ["dataset", "feature"]) == []

