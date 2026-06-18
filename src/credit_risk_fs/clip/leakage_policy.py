from __future__ import annotations

ALLOWED_DATASETS = ("homecredit", "lendingclub_v2")
LEGACY_DATASETS = ("lendingclub",)

REQUIRED_TRAINING_COLUMNS = (
    "dataset",
    "feature",
    "clip_training_split",
    "clip_training_text",
    "description",
    "semantic_group",
    "allowed_for_clip_training",
    "clip_training_exclusion_reason",
    "leakage_review_status",
    "leakage_review_action",
    "leakage_rule",
    "prohibited_training_fields",
    "evaluation_only_fields",
)

FORBIDDEN_EXACT_COLUMNS = {
    "target",
    "label",
    "bad",
    "bad_flag",
    "is_bad",
    "default",
    "loan_status",
    "split",
    "fold_id",
    "sk_id_curr",
    "member_id",
    "id",
}

FORBIDDEN_COLUMN_FRAGMENTS = (
    "psi",
    "target",
    "loan_status",
    "last_pymnt",
    "next_pymnt",
    "recoveries",
    "settlement",
    "hardship",
)


def is_forbidden_training_column(column: str) -> bool:
    """Return true for fields that must not be used as CLIP training evidence."""
    lowered = column.strip().lower()
    if lowered in FORBIDDEN_EXACT_COLUMNS:
        return True
    if lowered.startswith("oot") or lowered.endswith("_oot") or "_oot_" in lowered:
        return True
    return any(fragment in lowered for fragment in FORBIDDEN_COLUMN_FRAGMENTS)

