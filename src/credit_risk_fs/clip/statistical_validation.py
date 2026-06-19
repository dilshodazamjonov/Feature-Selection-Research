from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from credit_risk_fs.clip.statistical_schema import StatisticalBaselineConfig
from credit_risk_fs.clip.validation import forbidden_field_matches
from credit_risk_fs.utils.hashing import sha256_file


def validate_required_statistical_artifacts(paths: Iterable[str | Path]) -> list[str]:
    return [f"missing required statistical-baseline input artifact: {path}" for path in paths if not Path(path).exists()]


def validate_statistical_config_policy(config: StatisticalBaselineConfig) -> list[str]:
    errors = []
    if config.train_dataset != "homecredit":
        errors.append("statistical baseline requires homecredit as train_dataset")
    if config.external_validation_dataset != "lendingclub_v2":
        errors.append("statistical baseline requires lendingclub_v2 as external_validation_dataset")
    if config.legacy_lendingclub_allowed:
        errors.append("legacy LendingClub is forbidden for CLIP statistical baseline")
    if config.llm_fields_allowed:
        errors.append("LLM fields must not be allowed in the statistical baseline")
    if config.stable_core_as_input or config.stable_core_role != "anchor_only":
        errors.append("stable-core membership must remain anchor_only and outside statistical inputs")
    if config.oot_fields_allowed or config.psi_fields_allowed or config.target_fields_allowed:
        errors.append("OOT, PSI, and target fields must not be allowed")
    if config.external_refit_allowed:
        errors.append("external refit must not be allowed")
    if config.fit_preprocessing_on != "homecredit_train_split_only":
        errors.append("preprocessing must be fit on homecredit_train_split_only")
    if config.algorithm_derived_fields_in_main_view:
        errors.append("algorithm-derived fields must be excluded from the main statistical view by default")
    if config.similarity_metric != "cosine":
        errors.append("only cosine similarity is currently supported")
    return errors


def validate_manifest_sources_and_hashes(
    *,
    manifest: dict,
    source_hashes: dict,
    config: StatisticalBaselineConfig,
) -> list[str]:
    errors = []
    if manifest.get("train_dataset") != config.train_dataset:
        errors.append("Prompt 1 manifest train dataset mismatch")
    if manifest.get("external_validation_dataset") != config.external_validation_dataset:
        errors.append("Prompt 1 manifest external-validation dataset mismatch")
    active = set(manifest.get("active_datasets", []))
    if active != {config.train_dataset, config.external_validation_dataset}:
        errors.append(f"unexpected active datasets in Prompt 1 manifest: {sorted(active)}")
    activity = manifest.get("training_activity", {})
    if activity.get("model_trained") or activity.get("encoder_loaded") or activity.get("contrastive_pairs_created"):
        errors.append("Prompt 1 manifest indicates prohibited model activity")
    for dataset in [config.train_dataset, config.external_validation_dataset]:
        source_file = str(manifest.get("source_files", {}).get(dataset, "")).replace("\\", "/")
        if not source_file.endswith(f"results/{dataset}/analysis/clip_readiness/dev_only_clip_training_evidence.csv"):
            errors.append(f"{dataset}: source file is not approved DEV-only evidence: {source_file}")
        if "feature_level_evidence_for_clip.csv" in source_file or "results/lendingclub/" in source_file:
            errors.append(f"{dataset}: forbidden source file path: {source_file}")
        expected = source_hashes.get(dataset, {}).get("sha256")
        if not expected:
            errors.append(f"{dataset}: missing source hash")
            continue
        observed = sha256_file(source_file)
        if observed != expected:
            errors.append(f"{dataset}: source hash mismatch expected={expected} observed={observed}")
    return errors


def validate_main_statistical_fields(
    *,
    fields: list[str],
    training_features: pd.DataFrame,
    external_validation_features: pd.DataFrame,
) -> list[str]:
    errors = []
    if not fields:
        errors.append("no approved main statistical fields are available")
    for field in fields:
        matches = forbidden_field_matches(field)
        if matches:
            errors.append(f"forbidden main statistical field {field}: {matches}")
        for label, frame in [("homecredit", training_features), ("lendingclub_v2", external_validation_features)]:
            if field not in frame.columns:
                errors.append(f"{label}: missing main statistical field {field}")
                continue
            values = pd.to_numeric(frame[field], errors="coerce")
            non_missing_original = frame[field].notna()
            failed_parse = values.isna() & non_missing_original
            if bool(failed_parse.any()):
                errors.append(f"{label}: non-numeric values in main statistical field {field}")
            if values.nunique(dropna=True) == 0:
                errors.append(f"{label}: all-null main statistical field {field}")
    return errors


def validate_group_split_artifact(
    *,
    split: pd.DataFrame,
    audit: dict,
    training_features: pd.DataFrame,
    expected_dataset: str = "homecredit",
) -> list[str]:
    errors = []
    required = {"dataset", "feature_name", "split", "group_key", "group_source", "seed"}
    missing = sorted(required - set(split.columns))
    if missing:
        return [f"group split missing columns: {missing}"]
    if set(split["dataset"].astype(str)) != {expected_dataset}:
        errors.append("group split contains non-HomeCredit datasets")
    if split["feature_name"].duplicated().any():
        errors.append("group split contains duplicate HomeCredit features")
    expected_features = set(training_features["feature"].astype(str))
    observed_features = set(split["feature_name"].astype(str))
    if expected_features != observed_features:
        errors.append(
            f"group split feature set mismatch: missing={len(expected_features - observed_features)}, extra={len(observed_features - expected_features)}"
        )
    train_groups = set(split.loc[split["split"].eq("train"), "group_key"].astype(str))
    val_groups = set(split.loc[split["split"].eq("validation"), "group_key"].astype(str))
    overlap = sorted(train_groups.intersection(val_groups))
    if overlap:
        errors.append(f"group split overlap: {overlap[:20]}")
    if int(audit.get("row_count", -1)) != len(split):
        errors.append("group split row_count does not match audit")
    if int(audit.get("train_rows", -1)) != int(split["split"].eq("train").sum()):
        errors.append("group split train_rows does not match audit")
    if int(audit.get("validation_rows", -1)) != int(split["split"].eq("validation").sum()):
        errors.append("group split validation_rows does not match audit")
    if int(audit.get("group_overlap_count", -1)) != 0:
        errors.append("Prompt 2 group split audit reports group overlap")
    return errors


def validate_anchor_artifacts(
    *,
    anchors: pd.DataFrame,
    anchor_manifest: dict,
    split: pd.DataFrame,
    minimum_anchor_count: int,
) -> list[str]:
    errors = []
    if anchor_manifest.get("anchor_dataset") != "homecredit":
        errors.append("anchor manifest must use HomeCredit as anchor dataset")
    if anchor_manifest.get("external_validation_dataset") != "lendingclub_v2":
        errors.append("anchor manifest must name LendingClub v2 as external validation")
    if set(anchors["dataset"].astype(str)) != {"homecredit"}:
        errors.append("anchor features must be HomeCredit-only")
    train_features = set(split.loc[split["split"].eq("train"), "feature_name"].astype(str))
    train_anchor_count = int(anchors["feature_name"].astype(str).isin(train_features).sum())
    if train_anchor_count < minimum_anchor_count:
        errors.append(f"insufficient HomeCredit train-split anchors: {train_anchor_count}")
    return errors


def validate_vector_frame(frame: pd.DataFrame, *, dataset: str, dimension: int) -> list[str]:
    errors = []
    required = {
        "dataset",
        "feature_name",
        "split",
        "group_key",
        "stable_row_index",
        "input_field_hash",
        "preprocessor_hash",
        "statistical_vector_hash",
        "vector_dimension",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        errors.append(f"{dataset}: statistical vector frame missing columns: {missing}")
        return errors
    if set(frame["dataset"].astype(str)) != {dataset}:
        errors.append(f"{dataset}: vector dataset mismatch")
    if frame["feature_name"].duplicated().any():
        errors.append(f"{dataset}: duplicate vector rows")
    if not frame["vector_dimension"].eq(dimension).all():
        errors.append(f"{dataset}: inconsistent vector dimensions")
    stat_cols = [col for col in frame.columns if str(col).startswith("stat_")]
    if len(stat_cols) != dimension:
        errors.append(f"{dataset}: expected {dimension} stat columns, observed {len(stat_cols)}")
    else:
        values = frame[stat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            errors.append(f"{dataset}: non-finite or non-numeric statistical vectors")
    return errors
