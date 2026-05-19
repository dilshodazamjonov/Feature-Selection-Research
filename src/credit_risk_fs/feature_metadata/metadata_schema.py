from __future__ import annotations

from typing import TypedDict


class FeatureMetadataEntry(TypedDict, total=False):
    name: str
    description: str
    table: str
    semantic_group: str
    missing_rate: float
    non_null_count: int
    dtype: str
    mean: float | None
    min: float | None
    max: float | None
    std: float | None
    var: float | None
    p05: float | None
    p25: float | None
    p50: float | None
    p75: float | None
    p95: float | None
    unique_count: int
