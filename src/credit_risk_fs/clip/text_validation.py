from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from credit_risk_fs.clip.validation import forbidden_field_matches


FORBIDDEN_TEXT_COLUMNS = {
    "clip_training_text",
    "llm_best_rank",
    "llm_mean_rank_if_available",
    "stable_core_membership",
    "bootstrap_selection_frequency_if_available",
    "mrmr_selection_frequency",
    "boruta_selection_frequency",
    "missing_rate_dev",
    "iv_score_if_available",
}


def validate_prompt1_artifacts(paths: Iterable[str | Path]) -> list[str]:
    errors = []
    for path in paths:
        if not Path(path).exists():
            errors.append(f"missing required Prompt 1 artifact: {path}")
    return errors


def validate_text_source_columns(columns: Iterable[str], requested_text_fields: Iterable[str]) -> list[str]:
    errors = []
    requested = set(requested_text_fields)
    forbidden_requested = sorted(requested.intersection(FORBIDDEN_TEXT_COLUMNS))
    if forbidden_requested:
        errors.append(f"forbidden fields requested for text construction: {forbidden_requested}")
    for field in requested:
        matches = forbidden_field_matches(field)
        if matches:
            errors.append(f"forbidden field-name pattern requested for text construction: {field} -> {matches}")
    missing = sorted(requested - set(columns))
    if missing:
        errors.append(f"text source is missing requested fields: {missing}")
    return errors


def validate_feature_text_frame(frame: pd.DataFrame, dataset: str) -> list[str]:
    errors = []
    required = {
        "dataset",
        "feature_name",
        "feature_text",
        "description_present",
        "semantic_group_present",
        "source_formula_present",
        "text_length_chars",
        "source_manifest_hash",
        "text_template_version",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        errors.append(f"{dataset}: feature text missing columns: {missing}")
        return errors
    if set(frame["dataset"].dropna().astype(str)) != {dataset}:
        errors.append(f"{dataset}: feature text dataset mismatch")
    if frame["feature_name"].duplicated().any():
        errors.append(f"{dataset}: duplicate feature text rows")
    if frame["feature_text"].fillna("").astype(str).str.strip().eq("").any():
        errors.append(f"{dataset}: empty feature text rows")
    if not frame.sort_values(["dataset", "feature_name"], kind="mergesort")[["dataset", "feature_name"]].reset_index(drop=True).equals(
        frame[["dataset", "feature_name"]].reset_index(drop=True)
    ):
        errors.append(f"{dataset}: feature text rows are not deterministically sorted")
    return errors


def validate_embeddings(frame: pd.DataFrame, *, expected_dimension: int, normalize: bool) -> list[str]:
    errors = []
    embedding_cols = [col for col in frame.columns if re.fullmatch(r"embedding_\d{4}", str(col))]
    if len(embedding_cols) != expected_dimension:
        errors.append(f"embedding dimension mismatch: expected={expected_dimension}, observed={len(embedding_cols)}")
        return errors
    values = frame[embedding_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        errors.append("embedding table contains non-finite values")
    if normalize:
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            errors.append("normalized embeddings do not have unit norm")
    return errors


def validate_no_legacy_rows(*frames: pd.DataFrame) -> list[str]:
    errors = []
    for frame in frames:
        if "dataset" in frame.columns and frame["dataset"].astype(str).eq("lendingclub").any():
            errors.append("legacy lendingclub rows are present")
    return errors
