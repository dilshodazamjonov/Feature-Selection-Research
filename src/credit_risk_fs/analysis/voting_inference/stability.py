"""Fold-level feature-selection stability under an authenticated universe.

Jaccard and Kuncheva are stability measures.  Neither is a predictive-performance
measure and neither is reported as one.
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from credit_risk_fs.evaluation.drift import jaccard_similarity
from credit_risk_fs.evaluation.stability import kuncheva_stability


def pairwise_fold_stability(
    fold_selections: Mapping[int, Sequence[str]],
    *,
    universe_size: int,
) -> pd.DataFrame:
    """Compute every unordered fold-pair Jaccard and Kuncheva value.

    Kuncheva is defined for equal-size subsets; the frozen implementation skips a
    pair whose denominator is zero.  Each skipped pair stays visible with an
    explicit reason rather than being silently dropped.
    """

    if universe_size <= 1:
        raise ValueError("stability universe size must exceed one")
    rows: list[dict[str, Any]] = []
    folds = sorted(fold_selections)
    for left_fold, right_fold in itertools.combinations(folds, 2):
        left = list(fold_selections[left_fold])
        right = list(fold_selections[right_fold])
        left_set, right_set = set(left), set(right)
        k_left, k_right = len(left_set), len(right_set)
        intersection = len(left_set & right_set)
        jaccard = jaccard_similarity(left_set, right_set)
        denominator = universe_size * min(k_left, k_right) - k_left * k_right
        if k_left == 0 or k_right == 0:
            kuncheva: float | None = None
            reason = "empty_selected_set"
        elif k_left != k_right:
            kuncheva = None
            reason = "unequal_subset_sizes"
        elif denominator == 0:
            kuncheva = None
            reason = "zero_denominator_universe_equals_subset_size"
        else:
            kuncheva = float(
                (intersection * universe_size - k_left * k_right) / denominator
            )
            reason = ""
        rows.append(
            {
                "left_fold": left_fold,
                "right_fold": right_fold,
                "left_selected_count": k_left,
                "right_selected_count": k_right,
                "left_duplicate_count": len(left) - k_left,
                "right_duplicate_count": len(right) - k_right,
                "intersection_count": intersection,
                "union_count": len(left_set | right_set),
                "universe_size": int(universe_size),
                "jaccard": float(jaccard),
                "kuncheva": kuncheva,
                "kuncheva_unavailable_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def summarise_pairwise_stability(pairwise: pd.DataFrame) -> dict[str, Any]:
    """Aggregate pairwise values while retaining the pair count evidence."""

    summary: dict[str, Any] = {"fold_pair_count": int(len(pairwise))}
    for metric in ("jaccard", "kuncheva"):
        values = pd.to_numeric(pairwise[metric], errors="coerce").dropna()
        summary[f"{metric}_pair_count"] = int(len(values))
        summary[f"{metric}_mean"] = float(values.mean()) if len(values) else None
        summary[f"{metric}_median"] = float(values.median()) if len(values) else None
        summary[f"{metric}_min"] = float(values.min()) if len(values) else None
        summary[f"{metric}_max"] = float(values.max()) if len(values) else None
        summary[f"{metric}_std"] = (
            float(values.std(ddof=1)) if len(values) > 1 else None
        )
    unavailable = pairwise.loc[
        pairwise["kuncheva"].isna(), "kuncheva_unavailable_reason"
    ].unique()
    summary["kuncheva_unavailable_reasons"] = ";".join(
        sorted(str(value) for value in unavailable if str(value))
    )
    return summary


def frozen_kuncheva_reference(
    fold_selections: Mapping[int, Sequence[str]], *, universe_size: int
) -> float | None:
    """Cross-check the aggregate against the frozen repository implementation."""

    sets = [set(fold_selections[fold]) for fold in sorted(fold_selections)]
    value = kuncheva_stability(sets, universe_size)
    return None if value is None or pd.isna(value) else float(value)


def fold_selection_inventory_rows(
    *,
    run_id: str,
    dataset: str,
    model: str,
    configuration: str,
    expected_budget: int,
    universe_size: int,
    fold_selections: Mapping[int, Sequence[str]],
    fold_candidate_pools: Mapping[int, Sequence[str]],
    declared_universe_counts: Mapping[int, set[int]],
    expected_fold_count: int,
) -> list[dict[str, Any]]:
    """Describe each expected fold selection, present or missing."""

    rows: list[dict[str, Any]] = []
    for fold in range(1, expected_fold_count + 1):
        selected = list(fold_selections.get(fold, []))
        pool = list(fold_candidate_pools.get(fold, []))
        declared = sorted(declared_universe_counts.get(fold, set()))
        rows.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "model": model,
                "configuration": configuration,
                "fold_id": fold,
                "present": fold in fold_selections,
                "selected_feature_count": len(selected),
                "distinct_selected_feature_count": len(set(selected)),
                "duplicate_selected_feature_count": len(selected) - len(set(selected)),
                "expected_final_feature_budget": expected_budget,
                "budget_matches": len(selected) == expected_budget,
                "candidate_pool_size": len(pool),
                "candidate_pool_smaller_than_budget": bool(
                    pool and len(pool) < expected_budget
                ),
                "selected_outside_candidate_pool": (
                    len(set(selected) - set(pool)) if pool else None
                ),
                "authenticated_universe_size": int(universe_size),
                "run_declared_universe_sizes": ";".join(str(value) for value in declared),
                "run_declared_universe_matches_authenticated": bool(
                    declared == [int(universe_size)]
                ),
            }
        )
    return rows


def selection_frequency(
    fold_selections: Mapping[int, Sequence[str]]
) -> pd.DataFrame:
    """Per-feature fold selection frequency for descriptive reporting."""

    counts: dict[str, int] = {}
    for features in fold_selections.values():
        for feature in set(features):
            counts[feature] = counts.get(feature, 0) + 1
    total = max(len(fold_selections), 1)
    frame = pd.DataFrame(
        {
            "feature": list(counts),
            "selection_count": [counts[feature] for feature in counts],
        }
    )
    if frame.empty:
        return pd.DataFrame(columns=["feature", "selection_count", "total_folds", "selection_frequency"])
    frame["total_folds"] = total
    frame["selection_frequency"] = frame["selection_count"] / total
    return frame.sort_values(
        ["selection_frequency", "feature"], ascending=[False, True]
    ).reset_index(drop=True)


__all__ = [
    "fold_selection_inventory_rows",
    "frozen_kuncheva_reference",
    "pairwise_fold_stability",
    "selection_frequency",
    "summarise_pairwise_stability",
]
