from __future__ import annotations

from typing import Iterable

import pandas as pd

from credit_risk_fs.feature_metadata.semantic_groups import infer_semantic_group


def semantic_coverage_frame(features: Iterable[str]) -> pd.DataFrame:
    values = [str(feature) for feature in features]
    if not values:
        return pd.DataFrame(columns=["semantic_group", "feature_count", "feature_ratio"])
    frame = pd.DataFrame({"feature_name": values})
    frame["semantic_group"] = frame["feature_name"].map(infer_semantic_group)
    summary = (
        frame.groupby("semantic_group", as_index=False)
        .agg(feature_count=("feature_name", "count"))
        .sort_values(["feature_count", "semantic_group"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary["feature_ratio"] = summary["feature_count"] / float(len(frame))
    return summary


def semantic_coverage_summary(features: Iterable[str]) -> dict[str, float]:
    frame = semantic_coverage_frame(features)
    return {
        "semantic_group_count": int(len(frame)),
        "largest_semantic_group_ratio": float(frame["feature_ratio"].max()) if not frame.empty else 0.0,
    }
