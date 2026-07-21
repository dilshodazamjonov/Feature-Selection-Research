"""Predeclared paired OOT inference for the research extension protocol."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


BOOTSTRAP_SEED = 20260721
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_MINIMUM_VALID = 1900


def _canonical_id(value: Any) -> str:
    if pd.isna(value):
        raise ValueError("paired prediction identity contains a missing value")
    return unicodedata.normalize("NFC", str(value))


def _validated_prediction_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = {"stable_row_id", "target", "prediction_probability"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} prediction columns missing: {sorted(missing)}")
    output = frame.loc[:, ["stable_row_id", "target", "prediction_probability"]].copy()
    output["stable_row_id"] = output["stable_row_id"].map(_canonical_id)
    if output["stable_row_id"].duplicated().any():
        raise ValueError(f"{label} prediction identities are duplicated")
    output["target"] = pd.to_numeric(output["target"], errors="raise").astype("int8")
    if not output["target"].isin([0, 1]).all():
        raise ValueError(f"{label} targets must be binary 0/1")
    output["prediction_probability"] = pd.to_numeric(
        output["prediction_probability"], errors="raise"
    ).astype(float)
    if not np.isfinite(output["prediction_probability"].to_numpy()).all():
        raise ValueError(f"{label} prediction scores must be finite")
    return output


def align_paired_predictions(
    method_a: pd.DataFrame,
    method_b: pd.DataFrame,
) -> pd.DataFrame:
    """Align two methods by unique stable ID and verify identical targets."""

    left = _validated_prediction_frame(method_a, "method_a").rename(
        columns={"prediction_probability": "score_a", "target": "target_a"}
    )
    right = _validated_prediction_frame(method_b, "method_b").rename(
        columns={"prediction_probability": "score_b", "target": "target_b"}
    )
    left_ids = set(left["stable_row_id"])
    right_ids = set(right["stable_row_id"])
    if left_ids != right_ids:
        raise ValueError(
            "paired prediction identity sets differ: "
            f"missing_from_b={len(left_ids - right_ids)}, "
            f"missing_from_a={len(right_ids - left_ids)}"
        )
    aligned = left.merge(right, on="stable_row_id", how="inner", validate="one_to_one")
    if not aligned["target_a"].equals(aligned["target_b"]):
        mismatches = int(aligned["target_a"].ne(aligned["target_b"]).sum())
        raise ValueError(f"paired prediction targets disagree for {mismatches} identities")
    aligned = aligned.rename(columns={"target_a": "target"}).drop(columns="target_b")
    aligned["__canonical_order__"] = aligned["stable_row_id"].map(
        lambda value: unicodedata.normalize("NFC", value).casefold()
    )
    if aligned["__canonical_order__"].duplicated().any():
        raise ValueError("paired IDs have a Unicode-normalized ordering collision")
    return (
        aligned.sort_values("__canonical_order__", kind="mergesort")
        .drop(columns="__canonical_order__")
        .reset_index(drop=True)
    )


def ks_statistic(target: Iterable[int], score: Iterable[float]) -> tuple[float, float]:
    """Return absolute empirical-CDF KS and the smallest maximizing score."""

    frame = pd.DataFrame({"target": list(target), "score": list(score)})
    if frame.empty or not frame["target"].isin([0, 1]).all():
        raise ValueError("KS requires non-empty binary 0/1 targets")
    if not np.isfinite(pd.to_numeric(frame["score"], errors="raise")).all():
        raise ValueError("KS scores must be finite")
    positive_count = int(frame["target"].sum())
    negative_count = len(frame) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("KS requires both target classes")
    grouped = (
        frame.groupby("score", sort=True)["target"]
        .agg(positive="sum", total="size")
        .reset_index()
    )
    grouped["negative"] = grouped["total"] - grouped["positive"]
    separation = (
        grouped["positive"].cumsum() / positive_count
        - grouped["negative"].cumsum() / negative_count
    ).abs()
    maximum = float(separation.max())
    threshold = float(grouped.loc[separation.eq(maximum), "score"].iloc[0])
    return maximum, threshold


def lift_at_fraction(
    target: Iterable[int],
    score: Iterable[float],
    stable_row_ids: Iterable[Any],
    *,
    fraction: float = 0.10,
) -> float:
    """Return bad-rate lift in the deterministic highest-risk fraction."""

    frame = pd.DataFrame(
        {"target": list(target), "score": list(score), "stable_row_id": list(stable_row_ids)}
    )
    if not 0 < fraction <= 1 or frame.empty:
        raise ValueError("lift fraction must be in (0, 1] and inputs must be non-empty")
    if not frame["target"].isin([0, 1]).all():
        raise ValueError("lift targets must be binary 0/1")
    if not np.isfinite(pd.to_numeric(frame["score"], errors="raise")).all():
        raise ValueError("lift scores must be finite")
    frame["__identity_order__"] = frame["stable_row_id"].map(_canonical_id).map(
        lambda value: value.casefold()
    )
    if frame["__identity_order__"].duplicated().any():
        raise ValueError("lift stable row identities must be unique")
    overall_rate = float(frame["target"].mean())
    if overall_rate == 0:
        raise ValueError("lift requires at least one positive target")
    top_count = int(math.ceil(fraction * len(frame)))
    ordered = frame.sort_values(
        ["score", "__identity_order__"],
        ascending=[False, True],
        kind="mergesort",
    )
    return float(ordered.head(top_count)["target"].mean() / overall_rate)


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    positive = target == 1
    positive_count = int(positive.sum())
    negative_count = len(target) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUC requires both target classes")
    ranks = _midrank(score)
    return float(
        (ranks[positive].sum() - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def _ks_value(target: np.ndarray, score: np.ndarray) -> float:
    positive_count = int(target.sum())
    negative_count = len(target) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("KS requires both target classes")
    order = np.argsort(score, kind="mergesort")
    ordered_score = score[order]
    ordered_target = target[order]
    boundaries = np.r_[np.flatnonzero(np.diff(ordered_score) != 0), len(target) - 1]
    positive_cumulative = np.cumsum(ordered_target)[boundaries] / positive_count
    negative_cumulative = np.cumsum(1 - ordered_target)[boundaries] / negative_count
    return float(np.max(np.abs(positive_cumulative - negative_cumulative)))


def _lift_value(
    target: np.ndarray, score: np.ndarray, identity_order: list[str], fraction: float = 0.10
) -> float:
    overall_rate = float(target.mean())
    if overall_rate == 0:
        raise ValueError("lift requires at least one positive target")
    top_count = int(math.ceil(fraction * len(target)))
    order = sorted(
        range(len(target)),
        key=lambda index: (-float(score[index]), identity_order[index].casefold()),
    )
    return float(target[order[:top_count]].mean() / overall_rate)


def _fast_delong(predictions: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    classifiers, examples = predictions.shape
    negative_count = examples - positive_count
    positive = predictions[:, :positive_count]
    negative = predictions[:, positive_count:]
    tx = np.empty((classifiers, positive_count), dtype=float)
    ty = np.empty((classifiers, negative_count), dtype=float)
    tz = np.empty((classifiers, examples), dtype=float)
    for classifier in range(classifiers):
        tx[classifier] = _midrank(positive[classifier])
        ty[classifier] = _midrank(negative[classifier])
        tz[classifier] = _midrank(predictions[classifier])
    aucs = tz[:, :positive_count].sum(axis=1) / (positive_count * negative_count)
    aucs -= (positive_count + 1.0) / (2.0 * negative_count)
    v01 = (tz[:, :positive_count] - tx) / negative_count
    v10 = 1.0 - (tz[:, positive_count:] - ty) / positive_count
    sx = np.atleast_2d(np.cov(v01, bias=False))
    sy = np.atleast_2d(np.cov(v10, bias=False))
    return aucs, sx / positive_count + sy / negative_count


def paired_delong_test(
    target: Iterable[int],
    score_a: Iterable[float],
    score_b: Iterable[float],
) -> dict[str, float]:
    """Two-sided paired DeLong test, reporting AUC A minus AUC B."""

    target_array = np.asarray(list(target), dtype=int)
    scores_a = np.asarray(list(score_a), dtype=float)
    scores_b = np.asarray(list(score_b), dtype=float)
    if not (len(target_array) == len(scores_a) == len(scores_b)):
        raise ValueError("paired DeLong input lengths differ")
    if not np.isfinite(scores_a).all() or not np.isfinite(scores_b).all():
        raise ValueError("paired DeLong scores must be finite")
    if set(np.unique(target_array)) != {0, 1}:
        raise ValueError("paired DeLong requires both binary target classes")
    positive_count = int(target_array.sum())
    negative_count = len(target_array) - positive_count
    if positive_count < 2 or negative_count < 2:
        raise ValueError("paired DeLong requires at least two rows in each target class")
    order = np.argsort(-target_array, kind="mergesort")
    aucs, covariance = _fast_delong(
        np.vstack([scores_a[order], scores_b[order]]), positive_count
    )
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    difference = float(aucs[0] - aucs[1])
    if variance <= 0:
        p_value = 1.0 if abs(difference) <= 1e-15 else 0.0
        z_score = 0.0 if p_value == 1.0 else math.copysign(math.inf, difference)
    else:
        z_score = difference / math.sqrt(variance)
        p_value = math.erfc(abs(z_score) / math.sqrt(2.0))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_difference_a_minus_b": difference,
        "variance": variance,
        "z_score": float(z_score),
        "two_sided_p_value": float(p_value),
    }


def paired_stratified_bootstrap(
    aligned: pd.DataFrame,
    *,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
    minimum_valid: int = BOOTSTRAP_MINIMUM_VALID,
) -> dict[str, Any]:
    """Bootstrap paired AUC, KS, and Lift@10 differences using fixed attempts."""

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
    differences: dict[str, list[float]] = {"auc": [], "ks": [], "lift_at_10": []}
    failed = 0
    for _ in range(repetitions):
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
        sampled_ids = [
            f"{identities[source_index]}\x1fbootstrap_draw_{draw_position}"
            for draw_position, source_index in enumerate(sampled)
        ]
        try:
            replicate = {
                "auc": _binary_auc(sampled_target, sampled_a)
                - _binary_auc(sampled_target, sampled_b),
                "ks": _ks_value(sampled_target, sampled_a)
                - _ks_value(sampled_target, sampled_b),
                "lift_at_10": _lift_value(sampled_target, sampled_a, sampled_ids)
                - _lift_value(sampled_target, sampled_b, sampled_ids),
            }
        except (ValueError, FloatingPointError):
            failed += 1
        else:
            for metric, value in replicate.items():
                differences[metric].append(float(value))

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
        }
    return {
        "seed": seed,
        "attempted_repetitions": repetitions,
        "valid_repetitions": valid,
        "failed_repetitions": failed,
        "minimum_valid_repetitions": minimum_valid,
        "stratification": "positive_and_negative_sampled_separately_with_paired_indices",
        "metrics": metrics,
    }


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    """Return stable Holm step-down adjusted p-values in original order."""

    values = [float(value) for value in p_values]
    if not values or any(not math.isfinite(value) or value < 0 or value > 1 for value in values):
        raise ValueError("Holm p-values must be a non-empty finite list within [0, 1]")
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    adjusted = [0.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(values) - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def apply_holm_to_family(raw_p_values: Mapping[str, float]) -> dict[str, dict[str, float]]:
    """Adjust one predeclared dataset-model family without pooling families."""

    labels = list(raw_p_values)
    adjusted = holm_adjust(raw_p_values.values())
    return {
        label: {"raw_two_sided_p_value": float(raw_p_values[label]), "holm_p_value": value}
        for label, value in zip(labels, adjusted, strict=True)
    }


__all__ = [
    "BOOTSTRAP_MINIMUM_VALID",
    "BOOTSTRAP_REPETITIONS",
    "BOOTSTRAP_SEED",
    "align_paired_predictions",
    "apply_holm_to_family",
    "holm_adjust",
    "ks_statistic",
    "lift_at_fraction",
    "paired_delong_test",
    "paired_stratified_bootstrap",
]
