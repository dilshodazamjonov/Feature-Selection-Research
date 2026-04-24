from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.logging_config import setup_logging

logger = setup_logging("feature_metadata", level=logging.INFO)

ENGINEERED_PREFIX_TABLES = {
    "BURO": "bureau",
    "PREV": "previous_application",
    "POS": "POS_CASH_balance",
    "INSTAL": "installments_payments",
    "CC": "credit_card_balance",
}

AGGREGATION_TOKENS = {
    "MIN": "minimum",
    "MAX": "maximum",
    "MEAN": "mean",
    "SUM": "sum",
    "VAR": "variance",
}


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find a column in a DataFrame using flexible candidate names."""
    lowered = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _load_description_frame(description_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(description_path, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning(
            "UTF-8 decoding failed for %s, falling back to latin1.",
            description_path,
        )
        return pd.read_csv(description_path, encoding="latin1")


def _normalize_token(token: str) -> str:
    return token.replace("_", " ").lower()


def _infer_engineered_metadata(feature_name: str) -> tuple[str, str]:
    parts = feature_name.split("_")
    if not parts:
        return "", feature_name.replace("_", " ").lower()

    prefix = parts[0]
    table = ENGINEERED_PREFIX_TABLES.get(prefix, "")

    if prefix in ENGINEERED_PREFIX_TABLES and len(parts) >= 3 and parts[-1] in AGGREGATION_TOKENS:
        aggregation = AGGREGATION_TOKENS[parts[-1]]
        base_feature = _normalize_token("_".join(parts[1:-1]))
        description = f"{aggregation} of {base_feature} from {table}"
        return table, description

    return table, feature_name.replace("_", " ").lower()


def _safe_round(value: float | int | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)


def build_feature_metadata(
    X: pd.DataFrame,
    description_csv_path: str | Path,
) -> list[dict]:
    """
    Build per-feature metadata for LLM-assisted feature selection.

    Metadata is generated from the current training slice so the LLM sees only
    fold-local information. When an engineered feature is absent from the source
    description CSV, a structured fallback description is inferred from the
    feature prefix and aggregation suffix.
    """
    description_path = Path(description_csv_path)
    if not description_path.exists():
        raise FileNotFoundError(f"Description file not found: {description_path}")

    desc_df = _load_description_frame(description_path)

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
        inferred_table, inferred_desc = _infer_engineered_metadata(feature)
        desc = desc_map.get(feature, "") or inferred_desc
        table = table_map.get(feature, "") or inferred_table

        entry = {
            "name": feature,
            "description": desc,
            "table": table,
            "missing_rate": round(float(series.isna().mean()), 4),
            "non_null_count": int(series.notna().sum()),
            "dtype": str(series.dtype),
        }

        if pd.api.types.is_numeric_dtype(series):
            entry.update(
                {
                    "mean": _safe_round(series.mean(skipna=True)),
                    "min": _safe_round(series.min(skipna=True)),
                    "max": _safe_round(series.max(skipna=True)),
                    "std": _safe_round(series.std(skipna=True)),
                    "var": _safe_round(series.var(skipna=True)),
                }
            )
        else:
            entry["unique_count"] = int(series.nunique(dropna=True))

        metadata.append(entry)

    return metadata
