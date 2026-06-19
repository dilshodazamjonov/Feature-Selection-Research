from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONTRASTIVE_DATA_VERSION = "contrastive_data_boundary_v1"


@dataclass(frozen=True)
class ContrastiveDataConfig:
    manifest_path: Path
    source_hashes_path: Path
    group_split_path: Path
    homecredit_text_embeddings_path: Path
    lendingclub_v2_text_embeddings_path: Path
    embedding_cache_manifest_path: Path
    text_embedding_audit_path: Path
    homecredit_statistical_vectors_path: Path
    lendingclub_v2_statistical_vectors_path: Path
    statistical_feature_order_path: Path
    statistical_preprocessor_path: Path
    statistical_anchor_manifest_path: Path
    homecredit_feature_text_path: Path
    lendingclub_v2_feature_text_path: Path
    output_dir: Path
    seed: int
    training_dataset: str
    external_validation_dataset: str
    legacy_lendingclub_allowed: bool
    pair_key_fields: list[str]
    negative_policy: dict[str, Any]
    external_validation_policy: dict[str, Any]
    tensor_schema: dict[str, Any]


@dataclass(frozen=True)
class ContrastiveBuildResult:
    output_paths: dict[str, Path]
    summary: dict[str, Any]


PAIR_ROLES = {"train_positive", "validation_positive", "external_validation_positive"}
