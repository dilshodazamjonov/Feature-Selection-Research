from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd
import logging

from utils.logging_config import setup_logging

logger = setup_logging("llm_selector", level=logging.INFO)

def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find a column in a DataFrame using flexible candidate names."""
    lowered = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None

def _auto_description(feature_name: str) -> str:
    """Generate simple description from feature name."""
    return feature_name.replace("_", " ").lower()

def build_feature_metadata(
    X: pd.DataFrame,
    description_csv_path: str | Path,
) -> list[dict]:
    """Build feature metadata for LLM-based feature selection excluding statistical metrics."""
    description_path = Path(description_csv_path)
    if not description_path.exists():
        raise FileNotFoundError(f"Description file not found: {description_path}")

    # FIX: Attempt UTF-8 first, fallback to latin1 for Home Credit dataset quirks
    try:
        desc_df = pd.read_csv(description_path, encoding="latin1")
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decoding failed for {description_path}, falling back to latin1.")
        desc_df = pd.read_csv(description_path, encoding="utf-8")

    feature_col = _find_column(desc_df, ["row", "feature", "column", "columns", "name"])
    desc_col = _find_column(desc_df, ["description", "desc"])
    table_col = _find_column(desc_df, ["table", "source"])

    desc_map: dict[str, str] = {}
    table_map: dict[str, str] = {}

    if feature_col and desc_col:
        for _, row in desc_df.iterrows():
            feature_name = str(row[feature_col]).strip()
            if not feature_name or feature_name.lower() == "nan":
                continue
            desc_map[feature_name] = "" if pd.isna(row[desc_col]) else str(row[desc_col]).strip()
            if table_col:
                table_map[feature_name] = "" if pd.isna(row[table_col]) else str(row[table_col]).strip()

    metadata: list[dict] = []
    for feature in X.columns:
        series = X[feature]
        desc = desc_map.get(feature, "") or _auto_description(feature)

        entry = {
            "name": feature,
            "description": desc,
            "table": table_map.get(feature, ""),
            "missing_rate": round(float(series.isna().mean()), 4),
            "dtype": str(series.dtype),
        }

        if pd.api.types.is_numeric_dtype(series):
            entry.update({
                "mean": round(float(series.mean(skipna=True)), 2),
                "min": round(float(series.min(skipna=True)), 2),
                "max": round(float(series.max(skipna=True)), 2),
            })
        else:
            entry["unique_count"] = int(series.nunique())

        metadata.append(entry)
    print(pd.DataFrame(metadata))

    return metadata