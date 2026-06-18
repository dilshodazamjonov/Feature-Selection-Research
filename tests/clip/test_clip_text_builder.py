from __future__ import annotations

import pandas as pd
import pytest

from credit_risk_fs.clip.text_builder import build_feature_text, build_feature_text_frame
from credit_risk_fs.clip.text_validation import validate_text_source_columns


def test_text_template_is_deterministic_and_uses_allowed_metadata_only():
    row = {
        "feature": "AMT_CREDIT",
        "description": "Credit amount  of\n the loan",
        "semantic_group": "loan_terms",
        "source_table": "application_train.csv",
        "llm_best_rank": 1,
        "stable_core_membership": True,
        "psi_dev_oot_if_available": 0.4,
    }

    first, missing_first = build_feature_text(row)
    second, missing_second = build_feature_text(row)

    assert first == second
    assert missing_first == missing_second == ()
    assert first == (
        "Feature: AMT_CREDIT. Description: Credit amount of the loan. "
        "Semantic group: loan_terms. Source or formula: application_train.csv."
    )
    assert "llm" not in first.lower()
    assert "psi" not in first.lower()
    assert "stable_core" not in first.lower()


def test_missing_metadata_is_reported_and_not_fabricated():
    with pytest.raises(ValueError, match="description"):
        build_feature_text({"feature": "x", "semantic_group": "group", "source_table": "source"})

    text, missing = build_feature_text(
        {"feature": "x", "semantic_group": "group", "source_table": "source"},
        allow_fallback=True,
    )

    assert "metadata unavailable" in text
    assert missing == ("description",)


def test_text_frame_records_presence_and_hashes():
    frame = pd.DataFrame(
        [
            {
                "dataset": "homecredit",
                "feature": "x",
                "description": "Description",
                "semantic_group": "group",
                "source_table": "source",
            }
        ]
    )

    out = build_feature_text_frame(frame, dataset="homecredit", source_manifest_hash="abc")

    assert out.loc[0, "description_present"] is True or bool(out.loc[0, "description_present"])
    assert out.loc[0, "text_length_chars"] > 0
    assert len(out.loc[0, "feature_text_hash"]) == 64


def test_forbidden_text_fields_are_rejected():
    errors = validate_text_source_columns(
        ["feature", "description", "semantic_group", "source_table", "llm_best_rank"],
        ["feature", "description", "llm_best_rank"],
    )

    assert any("forbidden fields requested" in error for error in errors)

