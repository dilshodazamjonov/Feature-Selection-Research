"""Independent recomputation of headline metrics from saved predictions.

The primary path reuses the frozen repository implementations named by
``configs/protocols/credit_scoring_extension_v1.yaml``.  A deliberately separate
audit path lives in ``scripts/independently_verify_voting_metrics.py``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_risk_fs.evaluation.paired_inference import ks_statistic, lift_at_fraction

LIFT_FRACTION = 0.10


def recompute_discrimination(
    target: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    stable_row_ids: pd.Series | np.ndarray,
    *,
    fraction: float = LIFT_FRACTION,
) -> dict[str, float]:
    """Recompute AUC, Gini, KS, and Lift@10 with the frozen definitions."""

    target_array = np.asarray(target, dtype=int)
    score_array = np.asarray(score, dtype=float)
    if target_array.size == 0:
        raise ValueError("metric recomputation requires a non-empty prediction set")
    if not np.isfinite(score_array).all():
        raise ValueError("metric recomputation requires finite prediction scores")
    classes = set(np.unique(target_array).tolist())
    if classes != {0, 1}:
        raise ValueError(f"metric recomputation requires both target classes, saw {sorted(classes)}")
    auc = float(roc_auc_score(target_array, score_array))
    ks_value, ks_threshold = ks_statistic(target_array, score_array)
    lift = float(
        lift_at_fraction(target_array, score_array, stable_row_ids, fraction=fraction)
    )
    return {
        "auc": auc,
        "gini": 2.0 * auc - 1.0,
        "ks": float(ks_value),
        "ks_threshold": float(ks_threshold),
        "lift_at_10": lift,
    }


def lift_at_10_audit(
    target: pd.Series | np.ndarray,
    score: pd.Series | np.ndarray,
    stable_row_ids: pd.Series | np.ndarray,
    *,
    fraction: float = LIFT_FRACTION,
) -> dict[str, Any]:
    """Expose every operative detail of the authenticated Lift@10 definition."""

    target_array = np.asarray(target, dtype=int)
    score_array = np.asarray(score, dtype=float)
    row_count = int(target_array.size)
    top_count = int(math.ceil(fraction * row_count))
    frame = pd.DataFrame(
        {
            "target": target_array,
            "score": score_array,
            "identity": pd.Series(list(stable_row_ids), dtype="object").map(
                lambda value: str(value).casefold()
            ),
        }
    )
    ordered = frame.sort_values(
        ["score", "identity"], ascending=[False, True], kind="mergesort"
    )
    boundary_score = float(ordered["score"].iloc[top_count - 1])
    strictly_above = int((frame["score"] > boundary_score).sum())
    tie_group_size = int((frame["score"] == boundary_score).sum())
    tie_group_targets = frame.loc[frame["score"] == boundary_score, "target"]
    overall_rate = float(frame["target"].mean())
    top_rate = float(ordered.head(top_count)["target"].mean())
    return {
        "row_count": row_count,
        "fraction": float(fraction),
        "top_count_rule": "ceil(fraction * n)",
        "top_count": top_count,
        "top_group_meaning": "highest predicted class-1 default risk",
        "ratio_definition": "top_decile_positive_rate_divided_by_overall_positive_rate",
        "capture_rate_alternative_used": False,
        "boundary_tie_rule": "NFC-normalised casefolded stable row id ascending",
        "boundary_score": boundary_score,
        "rows_strictly_above_boundary_score": strictly_above,
        "boundary_tie_group_size": tie_group_size,
        "boundary_tie_group_is_target_homogeneous": bool(
            tie_group_targets.nunique(dropna=False) <= 1
        ),
        "boundary_tie_group_partially_included": bool(
            strictly_above < top_count < strictly_above + tie_group_size
        ),
        "top_decile_positive_rate": top_rate,
        "overall_positive_rate": overall_rate,
        "lift_at_10": float(top_rate / overall_rate),
    }


def stored_metric_row(path) -> dict[str, Any]:
    """Read the run-stored metric table so recomputation can be contrasted."""

    frame = pd.read_csv(path)
    output: dict[str, Any] = {}
    for _, row in frame.iterrows():
        split = str(row["split"]).upper()
        for column in ("auc", "gini", "ks"):
            if column in frame.columns:
                output[f"stored_{split.lower()}_{column}"] = float(row[column])
    return output


__all__ = [
    "LIFT_FRACTION",
    "lift_at_10_audit",
    "recompute_discrimination",
    "stored_metric_row",
]
