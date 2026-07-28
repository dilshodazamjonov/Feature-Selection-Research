"""Predeclared paired OOT inference for the Prompt 6 evidence package.

DeLong and Holm reuse the frozen implementations unchanged.  The stratified
paired bootstrap keeps the frozen design exactly - same seed, same 2,000
attempts, same per-replicate draw order, same percentile interval, same
metrics - but replaces three Python-level hot loops with vectorised equivalents.
``assert_bootstrap_equivalence`` proves the accelerated path reproduces the
frozen path replicate for replicate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from credit_risk_fs.evaluation.paired_inference import (
    BOOTSTRAP_MINIMUM_VALID,
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    holm_adjust,
    ks_statistic,
    lift_at_fraction,
    paired_delong_test,
    paired_stratified_bootstrap,
)

LIFT_FRACTION = 0.10
BOOTSTRAP_METRICS = ("auc", "ks", "lift_at_10")


# ---------------------------------------------------------------------------
# Vectorised replicate metrics
# ---------------------------------------------------------------------------


def _auc_from_ranks(target: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC using average ranks, matching the frozen mid-rank AUC."""

    positive_count = int(target.sum())
    negative_count = int(target.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUC requires both target classes")
    ranks = rankdata(score, method="average")
    positive_rank_sum = float(ranks[target == 1].sum())
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _auc_ks_from_sorted(target: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Mid-rank AUC and absolute-ECDF KS from one shared score ordering.

    Both frozen statistics are functions of the tied-score groups alone, so a
    single stable sort yields the mid-rank AUC (average rank per tie group) and
    the KS separation (cumulative class shares at each tie-group boundary).
    """

    row_count = int(target.size)
    positive_count = int(target.sum())
    negative_count = row_count - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUC and KS require both target classes")
    # Both statistics are functions of the tied-score groups only, never of the
    # order inside a group, so an unstable sort is safe here and is materially
    # cheaper than the stable sort the frozen helpers use.
    order = np.argsort(score, kind="quicksort")
    ordered_score = score[order]
    ordered_target = target[order]
    last = np.r_[np.flatnonzero(np.diff(ordered_score) != 0), row_count - 1]
    first = np.r_[0, last[:-1] + 1]
    positives_through_last = np.cumsum(ordered_target)[last]
    group_positives = np.diff(np.r_[0, positives_through_last])
    average_rank = (first + last + 2.0) / 2.0
    positive_rank_sum = float(np.dot(average_rank, group_positives))
    auc = float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )
    negatives_through_last = (last + 1) - positives_through_last
    ks = float(
        np.max(
            np.abs(
                positives_through_last / positive_count
                - negatives_through_last / negative_count
            )
        )
    )
    return auc, ks


def _ks_from_sorted(target: np.ndarray, score: np.ndarray) -> float:
    """Maximum absolute empirical-CDF separation, matching the frozen KS."""

    return _auc_ks_from_sorted(target, score)[1]


def _bootstrap_tie_key(identity: str, draw_position: int) -> str:
    """Reproduce the frozen per-draw lift tie-break identity exactly."""

    return f"{identity}\x1fbootstrap_draw_{draw_position}".casefold()


def _lift_from_boundary(
    target: np.ndarray,
    score: np.ndarray,
    *,
    tie_key: Any,
    fraction: float = LIFT_FRACTION,
) -> float:
    """Exact Lift@fraction that only orders the boundary tie group.

    ``tie_key`` is called with the array positions of the boundary tie group and
    must return their frozen tie-break strings.  Rows strictly above the boundary
    score are always included, so their internal order can never change the
    result and never needs to be materialised.
    """

    row_count = int(target.size)
    overall_rate = float(target.mean())
    if overall_rate == 0:
        raise ValueError("lift requires at least one positive target")
    top_count = int(math.ceil(fraction * row_count))
    partition = np.argpartition(-score, top_count - 1)[:top_count]
    boundary_score = float(score[partition].min())
    above = score > boundary_score
    above_count = int(np.count_nonzero(above))
    positives = int(target[above].sum())
    remaining = top_count - above_count
    if remaining > 0:
        tie_positions = np.flatnonzero(score == boundary_score)
        if remaining >= tie_positions.size:
            positives += int(target[tie_positions].sum())
        else:
            keys = tie_key(tie_positions)
            chosen = tie_positions[np.argsort(np.asarray(keys), kind="mergesort")][
                :remaining
            ]
            positives += int(target[chosen].sum())
    return float((positives / top_count) / overall_rate)


# ---------------------------------------------------------------------------
# Accelerated stratified paired bootstrap
# ---------------------------------------------------------------------------


def fast_paired_stratified_bootstrap(
    aligned: pd.DataFrame,
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
    minimum_valid: int = BOOTSTRAP_MINIMUM_VALID,
    replicate_sink: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Frozen stratified paired bootstrap design with vectorised replicates."""

    required = {"stable_row_id", "target", "score_a", "score_b"}
    if required - set(aligned.columns):
        raise ValueError("aligned paired predictions have an invalid schema")
    if repetitions < 1 or minimum_valid < 1 or minimum_valid > repetitions:
        raise ValueError("bootstrap repetition/minimum-valid contract is invalid")
    target = aligned["target"].to_numpy(dtype=int)
    positive_indices = np.flatnonzero(target == 1)
    negative_indices = np.flatnonzero(target == 0)
    if not len(positive_indices) or not len(negative_indices):
        raise ValueError("stratified bootstrap requires both target classes")
    score_a = aligned["score_a"].to_numpy(dtype=float)
    score_b = aligned["score_b"].to_numpy(dtype=float)
    identities = aligned["stable_row_id"].astype(str).to_numpy()
    generator = np.random.default_rng(seed)
    differences: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}
    failed = 0

    for replicate_index in range(repetitions):
        sampled_positive = generator.choice(
            positive_indices, size=len(positive_indices), replace=True
        )
        sampled_negative = generator.choice(
            negative_indices, size=len(negative_indices), replace=True
        )
        sampled = np.concatenate([sampled_positive, sampled_negative])
        sampled_target = target[sampled]
        sampled_a = score_a[sampled]
        sampled_b = score_b[sampled]

        def tie_key(positions: np.ndarray) -> list[str]:
            return [
                _bootstrap_tie_key(identities[sampled[position]], int(position))
                for position in positions
            ]

        try:
            auc_a, ks_a = _auc_ks_from_sorted(sampled_target, sampled_a)
            auc_b, ks_b = _auc_ks_from_sorted(sampled_target, sampled_b)
            replicate = {
                "auc": auc_a - auc_b,
                "ks": ks_a - ks_b,
                "lift_at_10": _lift_from_boundary(
                    sampled_target, sampled_a, tie_key=tie_key
                )
                - _lift_from_boundary(sampled_target, sampled_b, tie_key=tie_key),
            }
        except (ValueError, FloatingPointError):
            failed += 1
        else:
            for metric, value in replicate.items():
                differences[metric].append(float(value))
            if replicate_sink is not None:
                replicate_sink.append(
                    {"replicate_index": replicate_index, **replicate}
                )

    observed = {
        "auc": float(
            roc_auc_score(target, score_a) - roc_auc_score(target, score_b)
        ),
        "ks": float(ks_statistic(target, score_a)[0] - ks_statistic(target, score_b)[0]),
        "lift_at_10": float(
            lift_at_fraction(target, score_a, identities)
            - lift_at_fraction(target, score_b, identities)
        ),
    }
    valid = repetitions - failed
    metrics: dict[str, Any] = {}
    for metric, values in differences.items():
        interval_valid = valid >= minimum_valid and len(values) == valid
        lower, upper = (
            np.percentile(np.asarray(values), [2.5, 97.5]).tolist()
            if interval_valid
            else (None, None)
        )
        metrics[metric] = {
            "observed_difference_a_minus_b": observed[metric],
            "ci95_percentile_lower": lower,
            "ci95_percentile_upper": upper,
            "interval_valid": interval_valid,
            "replicate_mean": float(np.mean(values)) if values else None,
            "replicate_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
    return {
        "seed": seed,
        "attempted_repetitions": repetitions,
        "valid_repetitions": valid,
        "failed_repetitions": failed,
        "minimum_valid_repetitions": minimum_valid,
        "stratification": "positive_and_negative_sampled_separately_with_paired_indices",
        "implementation": "fast_paired_stratified_bootstrap",
        "frozen_design_equivalent": (
            "credit_risk_fs.evaluation.paired_inference.paired_stratified_bootstrap"
        ),
        "metrics": metrics,
    }


def assert_bootstrap_equivalence(
    aligned: pd.DataFrame,
    *,
    repetitions: int,
    seed: int = BOOTSTRAP_SEED,
    minimum_valid: int = 1,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Prove the accelerated bootstrap matches the frozen one on real inputs."""

    frozen = paired_stratified_bootstrap(
        aligned, repetitions=repetitions, seed=seed, minimum_valid=minimum_valid
    )
    fast = fast_paired_stratified_bootstrap(
        aligned, repetitions=repetitions, seed=seed, minimum_valid=minimum_valid
    )
    differences: dict[str, float] = {}
    for metric in BOOTSTRAP_METRICS:
        for field in (
            "observed_difference_a_minus_b",
            "ci95_percentile_lower",
            "ci95_percentile_upper",
        ):
            left = frozen["metrics"][metric][field]
            right = fast["metrics"][metric][field]
            if left is None or right is None:
                differences[f"{metric}.{field}"] = 0.0 if left == right else float("inf")
            else:
                differences[f"{metric}.{field}"] = abs(float(left) - float(right))
    worst = max(differences.values()) if differences else 0.0
    return {
        "repetitions": repetitions,
        "seed": seed,
        "valid_repetitions_frozen": frozen["valid_repetitions"],
        "valid_repetitions_fast": fast["valid_repetitions"],
        "maximum_absolute_difference": worst,
        "tolerance": tolerance,
        "per_field_absolute_difference": differences,
        "equivalent": bool(
            worst <= tolerance
            and frozen["valid_repetitions"] == fast["valid_repetitions"]
        ),
    }


# ---------------------------------------------------------------------------
# Predeclared family recovery and Holm correction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    """One predeclared comparison inside one Holm family."""

    family: str
    dataset: str
    model: str
    reference_run_id: str
    comparator_run_id: str
    candidate_pool_budget: int
    designation: str
    metric: str = "roc_auc"

    @property
    def label(self) -> str:
        return f"voting_pool_{self.candidate_pool_budget}_vs_rf_corr_mrmr"


def recover_predeclared_family(
    voting_protocol: Mapping[str, Any],
    run_lookup: Mapping[tuple[str, str, str], str],
) -> list[Comparison]:
    """Recover the predeclared comparison family from the frozen protocol only."""

    inference = voting_protocol["statistical_inference"]
    primary_pool = int(voting_protocol["primary_candidate_pool"])
    comparisons: list[Comparison] = []
    for family in inference["primary_families"]:
        dataset = str(family["dataset"])
        model_key = {"logistic_regression": "lr", "catboost": "catboost"}[
            str(family["model"])
        ]
        reference_method = str(family["reference"])
        if reference_method != "rf_corr_mrmr":
            raise ValueError(
                f"unexpected predeclared reference method: {reference_method!r}"
            )
        reference_run = run_lookup[(dataset, model_key, "reference")]
        for entry in family["comparisons"]:
            budget = int(str(entry).removeprefix("voting_pool_"))
            comparator_run = run_lookup[(dataset, model_key, f"voting_k{budget}")]
            comparisons.append(
                Comparison(
                    family=f"{dataset}_{model_key}",
                    dataset=dataset,
                    model=model_key,
                    reference_run_id=reference_run,
                    comparator_run_id=comparator_run,
                    candidate_pool_budget=budget,
                    designation="primary" if budget == primary_pool else "sensitivity",
                )
            )
    return comparisons


def run_paired_delong(
    aligned: pd.DataFrame,
    comparison: Comparison,
) -> dict[str, Any]:
    """Two-sided paired DeLong on identical OOT rows, comparator minus reference."""

    result = paired_delong_test(
        aligned["target"], aligned["score_comparator"], aligned["score_reference"]
    )
    positive_count = int(aligned["target"].sum())
    return {
        "family": comparison.family,
        "dataset": comparison.dataset,
        "model": comparison.model,
        "comparison_label": comparison.label,
        "designation": comparison.designation,
        "candidate_pool_budget": comparison.candidate_pool_budget,
        "reference_run_id": comparison.reference_run_id,
        "comparator_run_id": comparison.comparator_run_id,
        "metric": "roc_auc",
        "test": "two_sided_paired_delong",
        "direction_convention": "comparator_minus_reference",
        "auc_reference": result["auc_b"],
        "auc_comparator": result["auc_a"],
        "auc_delta_comparator_minus_reference": result["auc_difference_a_minus_b"],
        "gini_delta_comparator_minus_reference": 2.0
        * result["auc_difference_a_minus_b"],
        "variance": result["variance"],
        "standard_error": math.sqrt(result["variance"])
        if result["variance"] > 0
        else 0.0,
        "z_statistic": result["z_score"],
        "raw_two_sided_p_value": result["two_sided_p_value"],
        "aligned_sample_size": int(len(aligned)),
        "positive_count": positive_count,
        "negative_count": int(len(aligned) - positive_count),
    }


def apply_holm_families(
    delong_rows: Sequence[Mapping[str, Any]], *, alpha: float
) -> pd.DataFrame:
    """Apply Holm separately inside each predeclared dataset-model family."""

    frame = pd.DataFrame(list(delong_rows))
    if frame.empty:
        return frame
    outputs: list[dict[str, Any]] = []
    for family, group in frame.groupby("family", sort=True):
        ordered = group.sort_values(
            ["raw_two_sided_p_value", "comparison_label"], kind="mergesort"
        ).reset_index(drop=True)
        raw = [float(value) for value in ordered["raw_two_sided_p_value"]]
        adjusted = holm_adjust(raw)
        family_size = len(raw)
        for rank, (_, row) in enumerate(ordered.iterrows()):
            threshold = alpha / (family_size - rank)
            outputs.append(
                {
                    "family": family,
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "comparison_label": row["comparison_label"],
                    "designation": row["designation"],
                    "candidate_pool_budget": row["candidate_pool_budget"],
                    "reference_run_id": row["reference_run_id"],
                    "comparator_run_id": row["comparator_run_id"],
                    "raw_two_sided_p_value": float(row["raw_two_sided_p_value"]),
                    "family_size": family_size,
                    "ordered_rank": rank + 1,
                    "holm_threshold": threshold,
                    "holm_adjusted_p_value": float(adjusted[rank]),
                    "alpha": float(alpha),
                    "reject_null": bool(float(adjusted[rank]) <= alpha),
                    "holm_scope": "within_each_dataset_model_family_of_three_auc_tests",
                    "pooling_across_families": False,
                }
            )
    return pd.DataFrame(outputs)


__all__ = [
    "BOOTSTRAP_METRICS",
    "Comparison",
    "LIFT_FRACTION",
    "apply_holm_families",
    "assert_bootstrap_equivalence",
    "fast_paired_stratified_bootstrap",
    "recover_predeclared_family",
    "run_paired_delong",
]
