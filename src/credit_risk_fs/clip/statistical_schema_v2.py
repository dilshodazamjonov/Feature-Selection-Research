from __future__ import annotations

from dataclasses import dataclass


STATISTICAL_VIEW_VERSION_V2 = "compact_target_free_v2"

DESCRIPTOR_COLUMNS_V2 = [
    "missing_rate",
    "unique_ratio",
    "concentration_share",
    "signed_log_mean",
    "log_standard_deviation",
    "clipped_skewness",
    "normalized_entropy",
    "is_numeric",
    "is_categorical",
    "is_binary",
    "numeric_stats_valid",
    "skewness_valid",
    "entropy_valid",
]

SCALED_DESCRIPTOR_COLUMNS_V2 = [
    "missing_rate",
    "unique_ratio",
    "concentration_share",
    "signed_log_mean",
    "log_standard_deviation",
    "clipped_skewness",
    "normalized_entropy",
]

UNSCALED_INDICATOR_COLUMNS_V2 = [
    "is_numeric",
    "is_categorical",
    "is_binary",
    "numeric_stats_valid",
    "skewness_valid",
    "entropy_valid",
]

FORBIDDEN_STATISTICAL_VIEW_INPUTS = [
    "target",
    "oot",
    "psi",
    "prediction",
    "llm_rank",
    "stable_core",
    "post_origination",
]


@dataclass(frozen=True)
class StatisticalViewV2Schema:
    version: str = STATISTICAL_VIEW_VERSION_V2
    descriptor_columns: tuple[str, ...] = tuple(DESCRIPTOR_COLUMNS_V2)
    scaled_columns: tuple[str, ...] = tuple(SCALED_DESCRIPTOR_COLUMNS_V2)
    unscaled_columns: tuple[str, ...] = tuple(UNSCALED_INDICATOR_COLUMNS_V2)
    dimension: int = len(DESCRIPTOR_COLUMNS_V2)


def descriptor_order_v2() -> list[str]:
    return list(DESCRIPTOR_COLUMNS_V2)


def validate_descriptor_frame_columns(columns: list[str]) -> None:
    missing = [column for column in DESCRIPTOR_COLUMNS_V2 if column not in columns]
    if missing:
        raise ValueError(f"CLIP-v2 descriptor frame missing columns: {missing}")
