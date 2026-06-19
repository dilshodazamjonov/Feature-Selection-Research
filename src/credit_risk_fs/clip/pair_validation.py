from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from credit_risk_fs.clip.contrastive_schema import ContrastiveDataConfig
from credit_risk_fs.utils.hashing import sha256_file


def embedding_columns(frame: pd.DataFrame) -> list[str]:
    return sorted([col for col in frame.columns if str(col).startswith("embedding_") and len(str(col)) == 14])


def statistical_columns(frame: pd.DataFrame) -> list[str]:
    return sorted([col for col in frame.columns if str(col).startswith("stat_") and len(str(col)) == 9])


def validate_required_contrastive_artifacts(paths: Iterable[str | Path]) -> list[str]:
    return [f"missing required contrastive input artifact: {path}" for path in paths if not Path(path).exists()]


def validate_contrastive_config(config: ContrastiveDataConfig) -> list[str]:
    errors = []
    if config.training_dataset != "homecredit":
        errors.append("contrastive data boundary requires homecredit as training_dataset")
    if config.external_validation_dataset != "lendingclub_v2":
        errors.append("contrastive data boundary requires lendingclub_v2 as external_validation_dataset")
    if config.legacy_lendingclub_allowed:
        errors.append("legacy LendingClub is forbidden")
    policy = config.negative_policy
    if policy.get("explicit_hard_negatives_enabled"):
        errors.append("explicit hard negatives must remain disabled")
    if policy.get("cross_dataset_negatives_enabled"):
        errors.append("cross-dataset negatives must remain disabled")
    if policy.get("validation_as_training_negative"):
        errors.append("validation features cannot be training negatives")
    tensor = config.tensor_schema
    for key in ["stable_core_as_input", "llm_rank_as_input", "oot_allowed", "psi_allowed", "target_allowed"]:
        if tensor.get(key):
            errors.append(f"tensor schema forbids {key}=true")
    return errors


def validate_manifest_boundary(*, manifest: dict, source_hashes: dict, config: ContrastiveDataConfig) -> list[str]:
    errors = []
    if manifest.get("train_dataset") != config.training_dataset:
        errors.append("Prompt 1 manifest train dataset mismatch")
    if manifest.get("external_validation_dataset") != config.external_validation_dataset:
        errors.append("Prompt 1 manifest external dataset mismatch")
    if set(manifest.get("active_datasets", [])) != {config.training_dataset, config.external_validation_dataset}:
        errors.append("Prompt 1 active datasets are not exactly homecredit and lendingclub_v2")
    activity = manifest.get("training_activity", {})
    if activity.get("model_trained") or activity.get("contrastive_pairs_created"):
        errors.append("Prompt 1 manifest indicates prohibited training or pair creation")
    for dataset in [config.training_dataset, config.external_validation_dataset]:
        source_file = str(manifest.get("source_files", {}).get(dataset, "")).replace("\\", "/")
        if "results/lendingclub/" in source_file or "feature_level_evidence_for_clip.csv" in source_file:
            errors.append(f"{dataset}: forbidden source path {source_file}")
        expected = source_hashes.get(dataset, {}).get("sha256")
        if not expected:
            errors.append(f"{dataset}: missing source hash")
        elif sha256_file(source_file) != expected:
            errors.append(f"{dataset}: source hash mismatch")
    return errors


def validate_view_frame(
    *,
    text: pd.DataFrame,
    stat: pd.DataFrame,
    dataset: str,
    expected_text_dim: int,
    expected_stat_dim: int,
) -> list[str]:
    errors = []
    if set(text["dataset"].astype(str)) != {dataset}:
        errors.append(f"{dataset}: text embedding dataset mismatch")
    if set(stat["dataset"].astype(str)) != {dataset}:
        errors.append(f"{dataset}: statistical vector dataset mismatch")
    if text["feature_name"].duplicated().any():
        errors.append(f"{dataset}: duplicate text embedding feature rows")
    if stat["feature_name"].duplicated().any():
        errors.append(f"{dataset}: duplicate statistical vector feature rows")
    text_features = set(text["feature_name"].astype(str))
    stat_features = set(stat["feature_name"].astype(str))
    if text_features != stat_features:
        errors.append(f"{dataset}: missing view alignment text_only={len(text_features-stat_features)} stat_only={len(stat_features-text_features)}")
    text_cols = embedding_columns(text)
    stat_cols = statistical_columns(stat)
    if len(text_cols) != expected_text_dim:
        errors.append(f"{dataset}: text dimension mismatch expected={expected_text_dim} observed={len(text_cols)}")
    if len(stat_cols) != expected_stat_dim:
        errors.append(f"{dataset}: statistical dimension mismatch expected={expected_stat_dim} observed={len(stat_cols)}")
    if text_cols and not np.isfinite(text[text_cols].to_numpy(dtype=float)).all():
        errors.append(f"{dataset}: non-finite text embeddings")
    if stat_cols and not np.isfinite(stat[stat_cols].to_numpy(dtype=float)).all():
        errors.append(f"{dataset}: non-finite statistical vectors")
    return errors


def validate_group_split(split: pd.DataFrame) -> list[str]:
    errors = []
    required = {"dataset", "feature_name", "split", "group_key", "group_source"}
    missing = sorted(required - set(split.columns))
    if missing:
        return [f"group split missing columns: {missing}"]
    if set(split["dataset"].astype(str)) != {"homecredit"}:
        errors.append("group split must be HomeCredit-only")
    if split["feature_name"].duplicated().any():
        errors.append("group split contains duplicate features")
    train_groups = set(split.loc[split["split"].eq("train"), "group_key"].astype(str))
    validation_groups = set(split.loc[split["split"].eq("validation"), "group_key"].astype(str))
    overlap = sorted(train_groups.intersection(validation_groups))
    if overlap:
        errors.append(f"group overlap exists: {overlap[:20]}")
    return errors


def validate_positive_pairs(pairs: pd.DataFrame, *, role: str, dataset: str, split: str) -> list[str]:
    errors = []
    if pairs.empty:
        errors.append(f"{role}: no pairs produced")
        return errors
    if set(pairs["dataset"].astype(str)) != {dataset}:
        errors.append(f"{role}: dataset mismatch")
    if set(pairs["pair_role"].astype(str)) != {role}:
        errors.append(f"{role}: pair role mismatch")
    if set(pairs["split"].astype(str)) != {split}:
        errors.append(f"{role}: split mismatch")
    if pairs["pair_id"].duplicated().any():
        errors.append(f"{role}: duplicate pair IDs")
    if pairs["feature_name"].duplicated().any():
        errors.append(f"{role}: duplicate feature pairs")
    if pairs["text_hash"].isna().any() or pairs["statistical_vector_hash"].isna().any():
        errors.append(f"{role}: missing hashes")
    if role == "external_validation_positive":
        if pairs["allowed_for_training"].any() or pairs["allowed_for_validation"].any() or not pairs["allowed_for_external_evaluation"].all():
            errors.append("LendingClub v2 pair permissions are invalid")
    return errors
