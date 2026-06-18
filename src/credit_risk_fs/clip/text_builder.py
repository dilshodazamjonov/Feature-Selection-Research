from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

import pandas as pd

from credit_risk_fs.utils.hashing import sha256_text

TEXT_TEMPLATE_VERSION = "feature_text_v1"


@dataclass(frozen=True)
class FeatureTextRecord:
    dataset: str
    feature_name: str
    feature_text: str
    description_present: bool
    semantic_group_present: bool
    source_formula_present: bool
    text_length_chars: int
    source_manifest_hash: str
    text_template_version: str
    missing_components: tuple[str, ...]


def normalize_text_whitespace(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return re.sub(r"\s+", " ", text)


def build_feature_text(row: Mapping[str, object], *, allow_fallback: bool = False) -> tuple[str, tuple[str, ...]]:
    feature = normalize_text_whitespace(row.get("feature") or row.get("feature_name"))
    description = normalize_text_whitespace(row.get("description"))
    semantic_group = normalize_text_whitespace(row.get("semantic_group"))
    source = normalize_text_whitespace(row.get("source_table") or row.get("source_formula_or_table"))

    missing = []
    if not feature:
        missing.append("feature_name")
    if not description:
        missing.append("description")
    if not semantic_group:
        missing.append("semantic_group")
    if not source:
        missing.append("source_formula")

    if missing and not allow_fallback:
        raise ValueError(f"Cannot build feature text with missing components: {missing}")

    description = description or "metadata unavailable"
    semantic_group = semantic_group or "metadata unavailable"
    source = source or "metadata unavailable"

    text = (
        f"Feature: {feature}. "
        f"Description: {description}. "
        f"Semantic group: {semantic_group}. "
        f"Source or formula: {source}."
    )
    return normalize_text_whitespace(text), tuple(missing)


def build_feature_text_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
    source_manifest_hash: str,
    allow_fallback: bool = False,
    template_version: str = TEXT_TEMPLATE_VERSION,
) -> pd.DataFrame:
    records = []
    for row in frame.sort_values("feature", kind="mergesort").to_dict("records"):
        text, missing = build_feature_text(row, allow_fallback=allow_fallback)
        feature = normalize_text_whitespace(row.get("feature"))
        description = normalize_text_whitespace(row.get("description"))
        semantic_group = normalize_text_whitespace(row.get("semantic_group"))
        source = normalize_text_whitespace(row.get("source_table"))
        records.append(
            {
                "dataset": dataset,
                "feature_name": feature,
                "semantic_group": semantic_group,
                "source_formula_or_table": source,
                "feature_text": text,
                "description_present": bool(description),
                "semantic_group_present": bool(semantic_group),
                "source_formula_present": bool(source),
                "text_length_chars": len(text),
                "source_manifest_hash": source_manifest_hash,
                "text_template_version": template_version,
                "missing_components": ";".join(missing),
                "feature_text_hash": sha256_text(text),
            }
        )
    return pd.DataFrame(records).sort_values(["dataset", "feature_name"], kind="mergesort").reset_index(drop=True)
