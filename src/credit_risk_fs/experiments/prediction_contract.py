"""Frozen prediction coverage, identity, and probability-orientation contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from credit_risk_fs.experiments.atomic_io import ArtifactMetadata, write_csv_atomic, write_json_atomic
from credit_risk_fs.experiments.row_alignment import (
    ordered_row_id_sha256,
    ordered_row_id_target_sha256,
)


PILOT_COVERAGE = "single_dev_fold_pilot"
CAPACITY_SINGLE_FOLD_COVERAGE = "capacity_validation_single_dev_fold"
COMPLETE_OOF_COVERAGE = "complete_five_fold_dev_oof"
COMPLETE_OOT_COVERAGE = "locked_complete_oot"
PROBABILITY_ORIENTATION = "class_1_higher_default_risk"


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    expected_identities: Iterable[Any],
    expected_targets: Iterable[Any],
    coverage_type: str,
    expected_split: str,
    research_eligible: bool,
    comparison_eligible: bool,
) -> dict[str, Any]:
    """Validate prediction rows and return stable hash/count evidence."""

    required = {
        "stable_row_id",
        "target",
        "prediction_probability",
        "predicted_class",
        "fold_id",
        "split",
        "coverage_type",
        "research_eligible",
        "comparison_eligible",
        "probability_orientation",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"prediction artifact columns missing: {sorted(missing)}")
    if frame.empty:
        raise ValueError("prediction artifact must not be empty")
    identities = frame["stable_row_id"].astype(str)
    if identities.isna().any() or identities.duplicated().any():
        raise ValueError("prediction identities must be non-missing and unique")
    targets = pd.to_numeric(frame["target"], errors="raise").astype("int8")
    if not targets.isin([0, 1]).all():
        raise ValueError("prediction targets must be binary 0/1")
    probabilities = pd.to_numeric(
        frame["prediction_probability"], errors="raise"
    ).astype(float)
    if not np.isfinite(probabilities.to_numpy()).all() or not probabilities.between(
        0.0, 1.0
    ).all():
        raise ValueError("prediction probabilities must be finite and within [0, 1]")
    classes = pd.to_numeric(frame["predicted_class"], errors="raise").astype("int8")
    if not classes.isin([0, 1]).all():
        raise ValueError("predicted classes must be binary 0/1")

    expected_ids = [str(value) for value in expected_identities]
    expected_y = [int(value) for value in expected_targets]
    if len(expected_ids) != len(expected_y) or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("expected prediction identity/target contract is invalid")
    observed_target_by_id = dict(zip(identities, targets, strict=True))
    if set(identities) != set(expected_ids):
        raise ValueError("prediction identities do not exactly match the expected coverage")
    if any(observed_target_by_id[row_id] != target for row_id, target in zip(expected_ids, expected_y, strict=True)):
        raise ValueError("prediction targets do not match the expected identity-target contract")
    if set(frame["split"].astype(str)) != {expected_split}:
        raise ValueError("prediction split label mismatch")
    if set(frame["coverage_type"].astype(str)) != {coverage_type}:
        raise ValueError("prediction coverage label mismatch")
    if set(frame["probability_orientation"].astype(str)) != {PROBABILITY_ORIENTATION}:
        raise ValueError("prediction probability orientation mismatch")
    if set(frame["research_eligible"].map(bool)) != {bool(research_eligible)}:
        raise ValueError("prediction research-eligibility flag mismatch")
    if set(frame["comparison_eligible"].map(bool)) != {bool(comparison_eligible)}:
        raise ValueError("prediction comparison-eligibility flag mismatch")

    fold_values = set(frame["fold_id"].astype(str))
    if coverage_type in {PILOT_COVERAGE, CAPACITY_SINGLE_FOLD_COVERAGE}:
        if research_eligible or comparison_eligible or len(fold_values) != 1:
            raise ValueError("a single-fold pilot can never be research/comparison eligible")
    elif coverage_type == COMPLETE_OOF_COVERAGE:
        if not research_eligible or not comparison_eligible or fold_values != {"1", "2", "3", "4", "5"}:
            raise ValueError("complete DEV OOF requires exactly five folds and full eligibility")
    elif coverage_type == COMPLETE_OOT_COVERAGE:
        if not research_eligible or not comparison_eligible:
            raise ValueError("locked complete OOT predictions require full eligibility")
    else:
        raise ValueError(f"unsupported prediction coverage type: {coverage_type}")

    return {
        "coverage_type": coverage_type,
        "research_eligible": bool(research_eligible),
        "comparison_eligible": bool(comparison_eligible),
        "probability_orientation": PROBABILITY_ORIENTATION,
        "row_count": len(frame),
        "target_count": len(targets),
        "positive_target_count": int(targets.sum()),
        "source_order_identity_sha256": ordered_row_id_sha256(expected_ids),
        "artifact_order_identity_sha256": ordered_row_id_sha256(identities.tolist()),
        "identity_target_sha256": ordered_row_id_target_sha256(expected_ids, expected_y),
        "fold_ids": sorted(fold_values),
    }


def publish_prediction_artifact(
    *,
    path: str | Path,
    metadata_path: str | Path,
    frame: pd.DataFrame,
    expected_identities: Iterable[Any],
    expected_targets: Iterable[Any],
    coverage_type: str,
    expected_split: str,
    research_eligible: bool,
    comparison_eligible: bool,
    context: Mapping[str, Any],
) -> tuple[ArtifactMetadata, ArtifactMetadata, dict[str, Any]]:
    """Atomically publish a validated prediction CSV and its integrity sidecar."""

    validation = validate_prediction_frame(
        frame,
        expected_identities=expected_identities,
        expected_targets=expected_targets,
        coverage_type=coverage_type,
        expected_split=expected_split,
        research_eligible=research_eligible,
        comparison_eligible=comparison_eligible,
    )
    prediction_metadata = write_csv_atomic(
        path,
        frame,
        required_columns=["stable_row_id", "target", "prediction_probability"],
        ordered_row_identity_column="stable_row_id",
        overwrite=False,
    )
    payload = {
        **dict(context),
        **validation,
        "prediction_artifact": prediction_metadata.to_dict(),
    }
    sidecar_metadata = write_json_atomic(metadata_path, payload, overwrite=False)
    return prediction_metadata, sidecar_metadata, payload


__all__ = [
    "CAPACITY_SINGLE_FOLD_COVERAGE",
    "COMPLETE_OOF_COVERAGE",
    "COMPLETE_OOT_COVERAGE",
    "PILOT_COVERAGE",
    "PROBABILITY_ORIENTATION",
    "publish_prediction_artifact",
    "validate_prediction_frame",
]
