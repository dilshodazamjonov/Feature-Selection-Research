from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ClipDatasetRole(str, Enum):
    TRAIN = "train"
    EXTERNAL_VALIDATION = "external_validation"


class ClipFieldRole(str, Enum):
    TEXT_INPUT = "text_input"
    STATISTICAL_INPUT = "statistical_input"
    SUPERVISION_ONLY = "supervision_only"
    ANCHOR_ONLY = "anchor_only"
    EVALUATION_ONLY = "evaluation_only"
    FORBIDDEN = "forbidden"
    METADATA_ONLY = "metadata_only"


@dataclass(frozen=True)
class ClipDatasetSpec:
    name: str
    role: ClipDatasetRole
    source_file: Path


@dataclass(frozen=True)
class ClipFieldSpec:
    dataset: str
    field_name: str
    detected_dtype: str
    field_role: ClipFieldRole
    allowed_in_main_training_input: bool
    reason: str


@dataclass(frozen=True)
class ClipSourceArtifact:
    dataset: str
    role: ClipDatasetRole
    source_file: Path
    source_sha256: str
    row_count: int
    allowed_row_count: int
    blocked_row_count: int


@dataclass(frozen=True)
class ClipEvidenceAudit:
    dataset: str
    role: ClipDatasetRole
    source_file: Path
    source_sha256: str
    row_count: int
    allowed_row_count: int
    blocked_row_count: int
    dtypes: dict[str, str]
    missingness: dict[str, int]
    normalizations: list[str]
    duplicate_feature_names: list[str]
    duplicate_evidence_rows: int
    forbidden_fields_detected: dict[str, list[str]]
    validation_warnings: list[str]
    validation_errors: list[str]


@dataclass(frozen=True)
class ClipTrainingManifest:
    manifest_version: str
    created_at: str
    random_seed: int
    active_datasets: list[str]
    train_dataset: str
    external_validation_dataset: str
    source_files: dict[str, str]
    source_hashes: dict[str, str]
    source_row_counts: dict[str, int]
    allowed_row_counts: dict[str, int]
    blocked_row_counts: dict[str, int]
    allowed_feature_names: dict[str, list[str]]
    blocked_feature_names: dict[str, list[str]]
    block_reasons: dict[str, dict[str, str]]
    text_fields: list[str]
    candidate_statistical_fields: list[str]
    supervision_only_fields: list[str]
    anchor_only_fields: list[str]
    evaluation_only_fields: list[str]
    forbidden_fields_detected: dict[str, dict[str, list[str]]]
    missing_value_policy: str
    numeric_scaling_policy: str
    split_policy: str
    group_aware_split_fields: list[str]
    llm_rank_policy: str
    stable_core_policy: str
    oot_policy: str
    psi_policy: str
    validation_status: str
    validation_warnings: list[str]
    validation_errors: list[str]
    training_activity: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
