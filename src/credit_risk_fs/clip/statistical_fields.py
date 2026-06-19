from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk_fs.clip.statistical_schema import STATISTICAL_FIELD_ROLES, StatisticalBaselineConfig
from credit_risk_fs.clip.validation import forbidden_field_matches
from credit_risk_fs.utils.io import write_json

MAIN_FIELD_PROVENANCE = {
    "missing_rate_dev": "target-free DEV missingness statistic",
}

TARGET_AWARE_FIELDS = {
    "iv_score_if_available",
}

ALGORITHM_DERIVED_FIELDS = {
    "bootstrap_selection_frequency_if_available",
    "mrmr_selection_frequency",
    "boruta_selection_frequency",
    "selected_by_any_pipeline",
    "selected_by_mrmr",
    "selected_by_llm",
    "selected_by_llm_then_mrmr",
    "selected_by_stable_core_llm_fill",
}

LLM_DERIVED_FIELDS = {
    "llm_best_rank",
    "llm_mean_rank_if_available",
    "selected_by_llm",
    "selected_by_llm_then_mrmr",
    "selected_by_stable_core_llm_fill",
}

ANCHOR_ONLY_FIELDS = {
    "feature",
    "stable_core_membership",
}

METADATA_FIELDS = {
    "dataset",
    "clip_training_text",
    "description",
    "semantic_group",
    "source_table",
    "dtype_if_available",
    "allowed_for_clip_training",
    "clip_training_exclusion_reason",
    "leakage_review_status",
    "leakage_review_action",
    "leakage_review_reason",
    "leakage_rule",
    "prohibited_training_fields",
    "evaluation_only_fields",
    "evidence_source_files",
}

SPLIT_OR_ID_FIELDS = {"clip_training_split"}


def build_statistical_field_inventory(
    *,
    config: StatisticalBaselineConfig,
    homecredit_source: pd.DataFrame,
    lendingclub_source: pd.DataFrame,
    training_features: pd.DataFrame,
    external_validation_features: pd.DataFrame,
) -> pd.DataFrame:
    source_frames = {
        config.train_dataset: homecredit_source,
        config.external_validation_dataset: lendingclub_source,
    }
    allowed_frames = {
        config.train_dataset: training_features,
        config.external_validation_dataset: external_validation_features,
    }
    all_fields = sorted(set(homecredit_source.columns).union(lendingclub_source.columns))
    records: list[dict[str, Any]] = []
    for field in all_fields:
        role, reason, risk, notes = assign_statistical_field_role(field, config)
        if role not in STATISTICAL_FIELD_ROLES:
            raise ValueError(f"invalid statistical field role for {field}: {role}")
        for dataset, frame in source_frames.items():
            other_dataset = (
                config.external_validation_dataset if dataset == config.train_dataset else config.train_dataset
            )
            other = source_frames[other_dataset]
            allowed = allowed_frames[dataset]
            other_allowed = allowed_frames[other_dataset]
            present = field in frame.columns
            other_present = field in other.columns
            allowed_present = field in allowed.columns
            other_allowed_present = field in other_allowed.columns
            values = frame[field] if present else pd.Series(dtype="float64")
            other_values = other[field] if other_present else pd.Series(dtype="float64")
            records.append(
                {
                    "field_name": field,
                    "dataset": dataset,
                    "dtype": str(values.dtype) if present else "",
                    "source_artifact": "dev_only_clip_training_evidence.csv",
                    "provenance": _provenance(field),
                    "calculated_from_DEV_only": _calculated_from_dev_only(field),
                    "target_aware": field in TARGET_AWARE_FIELDS,
                    "algorithm_derived": field in ALGORITHM_DERIVED_FIELDS,
                    "available_in_homecredit": field in homecredit_source.columns,
                    "available_in_lendingclub_v2": field in lendingclub_source.columns,
                    "missing_rate_homecredit": _missing_rate(homecredit_source, field),
                    "missing_rate_lendingclub_v2": _missing_rate(lendingclub_source, field),
                    "unique_count_homecredit": _unique_count(homecredit_source, field),
                    "unique_count_lendingclub_v2": _unique_count(lendingclub_source, field),
                    "proposed_role": role,
                    "included_in_main_statistical_view": bool(
                        role == "statistical_input"
                        and field in config.approved_main_statistical_fields
                        and allowed_present
                        and other_allowed_present
                    ),
                    "exclusion_reason": "" if role == "statistical_input" else reason,
                    "leakage_risk": risk,
                    "notes": notes,
                    "available_in_this_dataset": present,
                    "available_in_other_dataset": other_present,
                }
            )
    return pd.DataFrame(records).sort_values(["field_name", "dataset"], kind="mergesort").reset_index(drop=True)


def assign_statistical_field_role(field: str, config: StatisticalBaselineConfig) -> tuple[str, str, str, str]:
    if forbidden_field_matches(field) or field in SPLIT_OR_ID_FIELDS:
        return "forbidden", "field matches forbidden split, ID, target, OOT, PSI, prediction, or outcome policy", "high", ""
    if field in LLM_DERIVED_FIELDS:
        return "forbidden", "LLM-derived field is not allowed in the statistical view", "high", ""
    if field == "stable_core_membership":
        return "anchor_only", "stable-core membership may define anchors only", "medium", ""
    if field == "feature":
        return "anchor_only", "feature name is an identifier used for alignment and anchors only", "low", ""
    if field in config.approved_main_statistical_fields:
        if field in TARGET_AWARE_FIELDS:
            return "statistical_input", "approved DEV-only target-aware statistical input", "medium", "target-aware"
        if field in ALGORITHM_DERIVED_FIELDS and not config.algorithm_derived_fields_in_main_view:
            return "optional_ablation_input", "algorithm-derived field defaults to optional ablation", "medium", ""
        return "statistical_input", "approved shared DEV-only statistical input", "low", ""
    if field in config.optional_ablation_fields:
        return "optional_ablation_input", "optional ablation field, excluded from the main statistical vector", "medium", ""
    if field in TARGET_AWARE_FIELDS:
        return "optional_ablation_input", "DEV-only target-aware field excluded from conservative main vector", "medium", ""
    if field in ALGORITHM_DERIVED_FIELDS:
        return "optional_ablation_input", "algorithm-derived field excluded from conservative main vector", "medium", ""
    if field in METADATA_FIELDS:
        return "metadata_only", "metadata, audit, text, or leakage-review field", "low", ""
    return "metadata_only", "not explicitly approved for the main statistical vector", "low", ""


def main_statistical_fields(inventory: pd.DataFrame) -> list[str]:
    main = inventory[inventory["included_in_main_statistical_view"].astype(bool)]["field_name"].astype(str).unique()
    return sorted(main.tolist())


def write_statistical_field_inventory(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "statistical_field_inventory.csv"
    json_path = out / "statistical_field_inventory.json"
    frame.to_csv(csv_path, index=False)
    payload = {
        "row_count": int(len(frame)),
        "field_count": int(frame["field_name"].nunique()) if len(frame) else 0,
        "roles": frame.groupby("proposed_role").size().sort_index().to_dict() if len(frame) else {},
        "fields": frame.to_dict("records"),
    }
    write_json(json_path, payload)
    return {"statistical_field_inventory_csv": csv_path, "statistical_field_inventory_json": json_path}


def _missing_rate(frame: pd.DataFrame, field: str) -> float | None:
    if field not in frame.columns or len(frame) == 0:
        return None
    return float(frame[field].isna().mean())


def _unique_count(frame: pd.DataFrame, field: str) -> int | None:
    if field not in frame.columns:
        return None
    return int(frame[field].nunique(dropna=True))


def _calculated_from_dev_only(field: str) -> bool:
    return field in MAIN_FIELD_PROVENANCE or field in TARGET_AWARE_FIELDS or field in ALGORITHM_DERIVED_FIELDS


def _provenance(field: str) -> str:
    if field in MAIN_FIELD_PROVENANCE:
        return MAIN_FIELD_PROVENANCE[field]
    if field in TARGET_AWARE_FIELDS:
        return "DEV-only target-aware univariate statistic"
    if field in ALGORITHM_DERIVED_FIELDS:
        return "DEV-only selector or resampling evidence"
    return "metadata or audit field from DEV-only evidence artifact"
