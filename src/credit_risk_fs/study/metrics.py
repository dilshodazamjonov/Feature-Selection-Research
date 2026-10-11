"""Evaluation metrics, drift, selection stability and paired inference."""

from __future__ import annotations

import itertools
import math
import unicodedata
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

MISSING_STATE = "__missing__"
UNSEEN_STATE = "__unseen__"
UNEXPECTED_STATE = "__unexpected__"


# --------------------------------------------------------------------- discrimination
def auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(np.asarray(y, dtype=int), np.asarray(score, dtype=float)))


def ks(y: np.ndarray, score: np.ndarray) -> float:
    """Maximum separation between the event and non-event score distributions."""

    target = np.asarray(y, dtype=int)
    values = np.asarray(score, dtype=float)
    positives = int(target.sum())
    negatives = len(target) - positives
    order = np.argsort(values, kind="mergesort")
    ordered_score = values[order]
    ordered_target = target[order]
    boundaries = np.r_[np.flatnonzero(np.diff(ordered_score) != 0), len(target) - 1]
    tpr = np.cumsum(ordered_target)[boundaries] / positives
    fpr = np.cumsum(1 - ordered_target)[boundaries] / negatives
    return float(np.max(np.abs(tpr - fpr)))


def brier(y: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(y, dtype=float)
    return float(np.mean((target - np.asarray(score, dtype=float)) ** 2))


def logloss(y: np.ndarray, score: np.ndarray) -> float:
    return float(log_loss(np.asarray(y, dtype=int), np.asarray(score, dtype=float), labels=[0, 1]))


def capture_and_lift(y: np.ndarray, score: np.ndarray, ids: Sequence[Any], fraction: float = 0.10) -> tuple[float, float]:
    """Capture@10 and Lift@10 over the top ``ceil(fraction * N)`` scores, ties by identifier."""

    target = np.asarray(y, dtype=float)
    values = np.asarray(score, dtype=float)
    keys = np.asarray([unicodedata.normalize("NFC", str(value)).casefold() for value in ids])
    order = np.lexsort((keys, -values))
    top = int(math.ceil(fraction * len(target)))
    selected = target[order[:top]]
    capture = float(selected.sum() / target.sum())
    lift = float(selected.mean() / target.mean())
    return capture, lift


def secondary_metrics(y: np.ndarray, score: np.ndarray, ids: Sequence[Any]) -> dict[str, float]:
    capture, lift = capture_and_lift(y, score, ids)
    return {
        "ks": ks(y, score),
        "brier": brier(y, score),
        "log_loss": logloss(y, score),
        "capture_at_10": capture,
        "lift_at_10": lift,
    }


# --------------------------------------------------------------------- feature drift
def _states(dev: pd.Series, ho: pd.Series, *, bins: int, discrete_max_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Type-aware states learned on DEV and applied unchanged to HO."""

    categorical = not pd.api.types.is_numeric_dtype(dev.dtype) or pd.api.types.is_bool_dtype(dev.dtype)
    if categorical:
        levels = sorted({str(value) for value in dev.dropna().astype("object").to_numpy()})

        def encode(values: pd.Series) -> np.ndarray:
            raw = values.astype("object")
            text = raw.astype(str)
            known = text.isin(levels).to_numpy()
            return np.where(
                raw.isna().to_numpy(),
                MISSING_STATE,
                np.where(known, text.to_numpy(dtype=object), UNSEEN_STATE),
            ).astype(object)

        return encode(dev), encode(ho)

    dev_values = pd.to_numeric(dev, errors="coerce").to_numpy(dtype="float64")
    ho_values = pd.to_numeric(ho, errors="coerce").to_numpy(dtype="float64")
    dev_present = np.isfinite(dev_values)
    ho_present = np.isfinite(ho_values)
    levels = np.unique(dev_values[dev_present])

    def labelled(values: np.ndarray, present: np.ndarray, labels: np.ndarray) -> np.ndarray:
        output = np.full(len(values), MISSING_STATE, dtype=object)
        output[present] = labels
        return output

    if levels.size and set(levels.tolist()) <= {0.0, 1.0}:
        def binary(values: np.ndarray, present: np.ndarray) -> np.ndarray:
            observed = values[present]
            labels = np.where(observed == 0.0, "0", np.where(observed == 1.0, "1", UNEXPECTED_STATE)).astype(object)
            return labelled(values, present, labels)

        return binary(dev_values, dev_present), binary(ho_values, ho_present)

    if levels.size and levels.size <= discrete_max_levels and np.all(np.equal(np.mod(levels, 1), 0)):

        def discrete(values: np.ndarray, present: np.ndarray) -> np.ndarray:
            observed = values[present]
            labels = np.where(np.isin(observed, levels), observed.astype(str), UNSEEN_STATE).astype(object)
            return labelled(values, present, labels)

        return discrete(dev_values, dev_present), discrete(ho_values, ho_present)

    observed = dev_values[dev_present]
    edges = np.unique(np.quantile(observed, np.linspace(0.0, 1.0, bins + 1))[1:-1]) if observed.size else np.array([])

    def binned(values: np.ndarray, present: np.ndarray) -> np.ndarray:
        labels = np.searchsorted(edges, values[present], side="right").astype(str).astype(object)
        return labelled(values, present, labels)

    return binned(dev_values, dev_present), binned(ho_values, ho_present)


def feature_psi(dev: pd.Series, ho: pd.Series, *, bins: int = 10, discrete_max_levels: int = 20, epsilon: float = 1e-6) -> float:
    """PSI over type-aware states: sum((o - r) * ln((o + eps) / (r + eps)))."""

    dev_states, ho_states = _states(dev.reset_index(drop=True), ho.reset_index(drop=True), bins=bins, discrete_max_levels=discrete_max_levels)
    reference = pd.Series(dev_states).value_counts(normalize=True)
    observed = pd.Series(ho_states).value_counts(normalize=True)
    states = reference.index.union(observed.index)
    r = reference.reindex(states, fill_value=0.0).to_numpy(dtype=float)
    o = observed.reindex(states, fill_value=0.0).to_numpy(dtype=float)
    return float(np.sum((o - r) * np.log((o + epsilon) / (r + epsilon))))


# --------------------------------------------------------------------- selection stability
def jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def pairwise_jaccard(subsets: Sequence[Sequence[str]]) -> list[float]:
    return [jaccard(a, b) for a, b in itertools.combinations(subsets, 2)]


def nogueira(subsets: Sequence[Sequence[str]], d: int) -> float:
    """Nogueira et al. stability over ``d`` candidate features."""

    from credit_risk_fs.evaluation.stability import nogueira_stability

    return float(nogueira_stability([set(subset) for subset in subsets], int(d)))


# --------------------------------------------------------------------- paired inference
class PairedBootstrap:
    """Target-stratified paired bootstrap of AUC differences on one population.

    Rows are put in canonical identifier order; each resample draws the positive and
    the negative rows with replacement (the same indices for both pipelines) from a
    generator seeded once, and the AUC of each pipeline is the Mann-Whitney statistic
    of the resample (ties count one half), computed from resample weights.
    """

    def __init__(self, y: np.ndarray, ids: Sequence[Any], *, resamples: int = 2000, seed: int = 20260721) -> None:
        keys = np.asarray([unicodedata.normalize("NFC", str(value)).casefold() for value in ids])
        self.order = np.argsort(keys, kind="mergesort")
        self.y = np.asarray(y, dtype=int)[self.order]
        self.positives = np.flatnonzero(self.y == 1)
        self.negatives = np.flatnonzero(self.y == 0)
        self.resamples = int(resamples)
        self.seed = int(seed)

    def _resample_weights(self):
        """The same resamples for every comparison: one generator seeded with ``seed``."""

        generator = np.random.default_rng(self.seed)
        for _ in range(self.resamples):
            sampled_positive = generator.choice(self.positives, size=len(self.positives), replace=True)
            sampled_negative = generator.choice(self.negatives, size=len(self.negatives), replace=True)
            yield (
                np.bincount(np.searchsorted(self.positives, sampled_positive), minlength=len(self.positives)),
                np.bincount(np.searchsorted(self.negatives, sampled_negative), minlength=len(self.negatives)),
            )

    def _prepare(self, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(score, dtype=float)[self.order]
        negative_scores = values[self.negatives]
        negative_order = np.argsort(negative_scores, kind="mergesort")
        sorted_negative = negative_scores[negative_order]
        positive_scores = values[self.positives]
        left = np.searchsorted(sorted_negative, positive_scores, side="left")
        right = np.searchsorted(sorted_negative, positive_scores, side="right")
        return negative_order, left, right

    @staticmethod
    def _weighted_auc(prepared: tuple[np.ndarray, np.ndarray, np.ndarray], positive_weights: np.ndarray, negative_weights: np.ndarray) -> float:
        negative_order, left, right = prepared
        cumulative = np.concatenate(([0], np.cumsum(negative_weights[negative_order], dtype=np.int64)))
        below = cumulative[left]
        equal = cumulative[right] - below
        numerator = float(np.dot(positive_weights, below + 0.5 * equal))
        return numerator / (float(positive_weights.sum()) * float(negative_weights.sum()))

    def auc_difference(self, score_a: np.ndarray, score_b: np.ndarray) -> dict[str, float]:
        prepared_a = self._prepare(score_a)
        prepared_b = self._prepare(score_b)
        differences = np.empty(self.resamples, dtype=float)
        for index, (positive_weights, negative_weights) in enumerate(self._resample_weights()):
            differences[index] = self._weighted_auc(prepared_a, positive_weights, negative_weights) - self._weighted_auc(
                prepared_b, positive_weights, negative_weights
            )
        y = self.y
        observed = auc(y, np.asarray(score_a, dtype=float)[self.order]) - auc(y, np.asarray(score_b, dtype=float)[self.order])
        low, high = np.percentile(differences, [2.5, 97.5]).tolist()
        return {"delta_auc": float(observed), "ci_low": float(low), "ci_high": float(high)}


def delong_p_value(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> float:
    from credit_risk_fs.evaluation.paired_inference import paired_delong_test

    return float(paired_delong_test(y, score_a, score_b)["two_sided_p_value"])


def holm(p_values: Sequence[float]) -> list[float]:
    from credit_risk_fs.evaluation.paired_inference import holm_adjust

    return list(holm_adjust(p_values))


__all__ = [
    "PairedBootstrap",
    "auc",
    "brier",
    "capture_and_lift",
    "delong_p_value",
    "feature_psi",
    "holm",
    "jaccard",
    "ks",
    "logloss",
    "nogueira",
    "pairwise_jaccard",
    "secondary_metrics",
]
