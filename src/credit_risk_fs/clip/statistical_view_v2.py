from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from credit_risk_fs.clip.statistical_schema_v2 import DESCRIPTOR_COLUMNS_V2, FORBIDDEN_STATISTICAL_VIEW_INPUTS


TYPE_NUMERIC = "numeric"
TYPE_CATEGORICAL = "categorical"
TYPE_BINARY = "binary"


@dataclass(frozen=True)
class TypeResolution:
    feature_name: str
    original_dtype: str
    metadata_type: str
    resolved_type: str
    resolution_rule: str
    ambiguity_warning: str = ""


def validate_allowed_input_columns(columns: list[str]) -> None:
    lowered = {column.lower() for column in columns}
    violations = []
    for column in lowered:
        for pattern in FORBIDDEN_STATISTICAL_VIEW_INPUTS:
            if pattern in column:
                violations.append(column)
    if violations:
        raise ValueError(f"forbidden CLIP-v2 statistical-view inputs detected: {sorted(set(violations))}")


def resolve_feature_type(
    values: pd.Series,
    *,
    feature_name: str,
    metadata_type: str | None = None,
    binary_value_threshold: int = 2,
) -> TypeResolution:
    metadata = (metadata_type or "").strip().lower()
    dtype_name = str(values.dtype)
    nonmissing = values.dropna()
    unique_count = int(nonmissing.nunique(dropna=True))
    numeric_like = pd.api.types.is_numeric_dtype(values)

    if metadata in {"binary", "bool", "boolean", "flag", "indicator"}:
        return TypeResolution(feature_name, dtype_name, metadata, TYPE_BINARY, "metadata_binary")
    if metadata in {"categorical", "category", "object", "string", "str", "nominal"}:
        if unique_count <= binary_value_threshold and unique_count > 0:
            return TypeResolution(feature_name, dtype_name, metadata, TYPE_BINARY, "metadata_categorical_binary_cardinality")
        return TypeResolution(feature_name, dtype_name, metadata, TYPE_CATEGORICAL, "metadata_categorical")
    if metadata in {"numeric", "number", "float", "integer", "int", "continuous"}:
        if unique_count <= binary_value_threshold and unique_count > 0:
            return TypeResolution(feature_name, dtype_name, metadata, TYPE_BINARY, "metadata_numeric_binary_cardinality")
        return TypeResolution(feature_name, dtype_name, metadata, TYPE_NUMERIC, "metadata_numeric")

    if numeric_like and unique_count <= binary_value_threshold and unique_count > 0:
        return TypeResolution(feature_name, dtype_name, metadata, TYPE_BINARY, "dtype_numeric_binary_cardinality")
    if numeric_like:
        return TypeResolution(feature_name, dtype_name, metadata, TYPE_NUMERIC, "dtype_numeric", "metadata_type_missing")
    if unique_count <= binary_value_threshold and unique_count > 0:
        return TypeResolution(feature_name, dtype_name, metadata, TYPE_BINARY, "dtype_object_binary_cardinality", "metadata_type_missing")
    return TypeResolution(feature_name, dtype_name, metadata, TYPE_CATEGORICAL, "dtype_object_categorical", "metadata_type_missing")


def compute_feature_descriptors(
    values: pd.Series,
    *,
    feature_name: str,
    metadata_type: str | None = None,
    ddof: int = 0,
) -> dict[str, Any]:
    resolution = resolve_feature_type(values, feature_name=feature_name, metadata_type=metadata_type)
    n_total = int(len(values))
    nonmissing = values.dropna()
    n_nonmissing = int(len(nonmissing))
    missing_rate = 1.0 if n_total == 0 else float(values.isna().sum() / n_total)
    unique_ratio = 0.0 if n_nonmissing == 0 else float(nonmissing.nunique(dropna=True) / n_nonmissing)

    is_numeric = resolution.resolved_type == TYPE_NUMERIC
    is_categorical = resolution.resolved_type == TYPE_CATEGORICAL
    is_binary = resolution.resolved_type == TYPE_BINARY
    numeric_stats_valid = 0
    skewness_valid = 0
    entropy_valid = 0
    signed_log_mean = 0.0
    log_standard_deviation = 0.0
    clipped_skewness = 0.0
    normalized_entropy = 0.0

    if n_nonmissing == 0:
        concentration_share = 0.0
        concentration_definition = "all_missing"
    elif is_numeric:
        numeric = pd.to_numeric(nonmissing, errors="coerce").dropna()
        concentration_share = 0.0 if len(numeric) == 0 else float(np.isclose(numeric.to_numpy(dtype=float), 0.0).mean())
        concentration_definition = "numeric_zero_share"
        if len(numeric) > 0:
            mean = float(numeric.mean())
            signed_log_mean = math.copysign(math.log1p(abs(mean)), mean) if mean != 0 else 0.0
            std = float(numeric.std(ddof=ddof))
            log_standard_deviation = math.log1p(max(std, 0.0))
            numeric_stats_valid = 1
            if len(numeric) >= 3 and std > 0:
                skew = float(numeric.skew())
                if np.isfinite(skew):
                    clipped_skewness = float(np.clip(skew, -10.0, 10.0))
                    skewness_valid = 1
    else:
        counts = nonmissing.astype("object").value_counts(dropna=True)
        concentration_share = 0.0 if counts.empty else float(counts.iloc[0] / n_nonmissing)
        concentration_definition = "binary_majority_class_share" if is_binary else "categorical_mode_share"
        if len(counts) > 1:
            probabilities = counts.to_numpy(dtype=float) / float(n_nonmissing)
            entropy = float(-(probabilities * np.log(probabilities)).sum())
            normalized_entropy = float(entropy / math.log(len(counts)))
            entropy_valid = 1
        elif len(counts) == 1:
            normalized_entropy = 0.0
            entropy_valid = 1

    row = {
        "feature_name": feature_name,
        "original_dtype": resolution.original_dtype,
        "metadata_type": resolution.metadata_type,
        "resolved_type": resolution.resolved_type,
        "resolution_rule": resolution.resolution_rule,
        "ambiguity_warning": resolution.ambiguity_warning,
        "concentration_definition": concentration_definition,
        "missing_rate": missing_rate,
        "unique_ratio": min(max(unique_ratio, 0.0), 1.0),
        "concentration_share": min(max(concentration_share, 0.0), 1.0),
        "signed_log_mean": signed_log_mean,
        "log_standard_deviation": log_standard_deviation,
        "clipped_skewness": clipped_skewness,
        "normalized_entropy": min(max(normalized_entropy, 0.0), 1.0),
        "is_numeric": int(is_numeric),
        "is_categorical": int(is_categorical),
        "is_binary": int(is_binary),
        "numeric_stats_valid": int(numeric_stats_valid),
        "skewness_valid": int(skewness_valid),
        "entropy_valid": int(entropy_valid),
    }
    return row


def build_statistical_view_frame(
    data: pd.DataFrame,
    *,
    metadata: pd.DataFrame | None = None,
    feature_column: str = "feature_name",
    metadata_type_column: str = "metadata_type",
    forbidden_columns: list[str] | None = None,
) -> pd.DataFrame:
    forbidden_columns = forbidden_columns or []
    validate_allowed_input_columns(list(data.columns) + forbidden_columns)
    metadata_types = {}
    if metadata is not None and feature_column in metadata.columns:
        if metadata_type_column in metadata.columns:
            metadata_types = dict(zip(metadata[feature_column].astype(str), metadata[metadata_type_column].astype(str), strict=False))
    rows = [
        compute_feature_descriptors(
            data[column],
            feature_name=str(column),
            metadata_type=metadata_types.get(str(column)),
        )
        for column in data.columns
        if column not in forbidden_columns
    ]
    frame = pd.DataFrame(rows)
    return frame[[
        "feature_name",
        "original_dtype",
        "metadata_type",
        "resolved_type",
        "resolution_rule",
        "ambiguity_warning",
        "concentration_definition",
        *DESCRIPTOR_COLUMNS_V2,
    ]]
