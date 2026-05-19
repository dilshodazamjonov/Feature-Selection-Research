from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def correlation_redundancy_frame(
    X: pd.DataFrame,
    *,
    threshold: float = 0.9,
) -> pd.DataFrame:
    numeric = X.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return pd.DataFrame(columns=["left_feature", "right_feature", "abs_correlation"])
    corr = numeric.corr().abs()
    rows: list[dict[str, object]] = []
    for left, right in itertools.combinations(corr.columns.tolist(), 2):
        value = corr.loc[left, right]
        if pd.notna(value) and float(value) >= threshold:
            rows.append(
                {
                    "left_feature": left,
                    "right_feature": right,
                    "abs_correlation": float(value),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["abs_correlation", "left_feature", "right_feature"],
        ascending=[False, True, True],
    )


def redundancy_summary(X: pd.DataFrame, *, threshold: float = 0.9) -> dict[str, float]:
    frame = correlation_redundancy_frame(X, threshold=threshold)
    return {
        "redundant_pair_count": int(len(frame)),
        "max_abs_correlation": float(frame["abs_correlation"].max()) if not frame.empty else 0.0,
    }
