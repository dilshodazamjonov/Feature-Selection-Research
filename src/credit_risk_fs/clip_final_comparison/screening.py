from __future__ import annotations

import numpy as np
import pandas as pd


def type_aware_dispersion_scores(X_dev: pd.DataFrame, feature_names: list[str] | None = None) -> pd.DataFrame:
    """Return deterministic unsupervised dispersion scores from DEV data only."""
    features = list(feature_names or X_dev.columns.astype(str))
    rows = []
    for feature in features:
        series = X_dev[feature]
        non_null = series.dropna()
        if non_null.empty:
            score = 0.0
            kind = "all_missing"
        elif pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce")
            score = float(values.var(ddof=0)) if values.notna().any() else 0.0
            kind = "numeric_variance"
        else:
            shares = non_null.astype(str).value_counts(normalize=True, dropna=True)
            entropy = -float((shares * np.log(shares)).sum()) if len(shares) else 0.0
            max_entropy = float(np.log(len(shares))) if len(shares) > 1 else 1.0
            score = entropy / max_entropy if max_entropy > 0 else 0.0
            kind = "categorical_normalized_entropy"
        rows.append({"feature_name": feature, "dispersion_score": score, "dispersion_kind": kind})
    return pd.DataFrame(rows).sort_values(["dispersion_score", "feature_name"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def random_screening_pool(features: list[str], pool_size: int, seed: int) -> list[str]:
    capped = min(int(pool_size), len(features))
    rng = np.random.default_rng(int(seed))
    if capped <= 0:
        return []
    return [features[index] for index in rng.choice(len(features), size=capped, replace=False)]


def correlation_filter_pool(
    X_dev: pd.DataFrame,
    pool_size: int,
    *,
    threshold: float,
    feature_names: list[str] | None = None,
) -> tuple[list[str], pd.DataFrame]:
    dispersion = type_aware_dispersion_scores(X_dev, feature_names)
    encoded = _encoded_association_frame(X_dev[dispersion["feature_name"].tolist()])
    selected: list[str] = []
    audit_rows = []
    for feature in dispersion["feature_name"]:
        if len(selected) >= min(pool_size, len(dispersion)):
            break
        max_assoc = 0.0
        if selected:
            associations = encoded[selected].corrwith(encoded[feature]).abs().fillna(0.0)
            max_assoc = float(associations.max()) if len(associations) else 0.0
        keep = max_assoc <= threshold
        audit_rows.append({"feature_name": feature, "max_abs_association": max_assoc, "kept": keep})
        if keep:
            selected.append(feature)
    return selected, pd.DataFrame(audit_rows)


def cosine_similarity_ranking(vectors: pd.DataFrame, anchor: np.ndarray, *, feature_column: str = "feature_name") -> pd.DataFrame:
    feature_names = vectors[feature_column].astype(str).tolist() if feature_column in vectors.columns else vectors.index.astype(str).tolist()
    numeric = vectors.drop(columns=[feature_column], errors="ignore").apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    anchor = np.asarray(anchor, dtype=float).reshape(-1)
    if numeric.shape[1] != anchor.shape[0]:
        raise ValueError(f"anchor dimension {anchor.shape[0]} does not match vectors {numeric.shape[1]}")
    scores = _cosine_rows(numeric, anchor)
    return (
        pd.DataFrame({"feature_name": feature_names, "similarity": scores})
        .sort_values(["similarity", "feature_name"], ascending=[False, True], kind="mergesort")
        .reset_index(drop=True)
    )


def full_mrmr_universe(features: list[str]) -> list[str]:
    return list(dict.fromkeys(features))


def cap_pool_size(requested_size: int, eligible_count: int) -> int:
    return min(int(requested_size), int(eligible_count))


def _encoded_association_frame(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            encoded[column] = pd.to_numeric(series, errors="coerce")
        else:
            encoded[column] = pd.factorize(series.astype("string").fillna("<NA>"), sort=True)[0]
    return encoded.astype(float).fillna(encoded.median(numeric_only=True)).fillna(0.0)


def _cosine_rows(matrix: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    row_norm = np.linalg.norm(matrix, axis=1)
    anchor_norm = float(np.linalg.norm(anchor))
    denom = np.where(row_norm == 0.0, np.nan, row_norm) * (anchor_norm if anchor_norm else np.nan)
    values = np.divide(matrix @ anchor, denom)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

