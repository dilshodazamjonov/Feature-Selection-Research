from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


STATISTICAL_BASELINE_VERSION = "dev_only_statistical_baseline_v1"


@dataclass(frozen=True)
class StatisticalBaselineConfig:
    manifest_path: Path
    training_features_path: Path
    external_validation_features_path: Path
    field_role_manifest_path: Path
    source_hashes_path: Path
    group_split_path: Path
    group_split_audit_path: Path
    anchor_features_path: Path
    anchor_manifest_path: Path
    homecredit_feature_text_path: Path
    lendingclub_v2_feature_text_path: Path
    train_dataset: str
    external_validation_dataset: str
    legacy_lendingclub_allowed: bool
    approved_main_statistical_fields: list[str]
    optional_ablation_fields: list[str]
    forbidden_field_patterns: list[str]
    field_alignment_rules: dict[str, Any]
    missing_value_policy: str
    imputation_strategy: str
    scaling_strategy: str
    clipping_enabled: bool
    clipping_lower_quantile: float
    clipping_upper_quantile: float
    fit_preprocessing_on: str
    external_refit_allowed: bool
    algorithm_derived_fields_in_main_view: bool
    llm_fields_allowed: bool
    stable_core_as_input: bool
    stable_core_role: str
    oot_fields_allowed: bool
    psi_fields_allowed: bool
    target_fields_allowed: bool
    similarity_metric: str
    seed: int
    output_dir: Path
    anchor_field: str
    minimum_anchor_count: int


@dataclass(frozen=True)
class StatisticalBaselineResult:
    output_paths: dict[str, Path]
    summary: dict[str, Any]


STATISTICAL_FIELD_ROLES = {
    "statistical_input",
    "optional_ablation_input",
    "anchor_only",
    "supervision_only",
    "evaluation_only",
    "metadata_only",
    "forbidden",
}
