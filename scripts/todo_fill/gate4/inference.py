"""Paired HO inference: the routine behind ``priority_1/pairwise.csv``.

Observations are paired on the stable row identifier, the difference of AUCs is
bootstrapped with 2,000 target-stratified resamples drawn from
``numpy.random.default_rng(20260721)`` (positives then negatives per draw, exactly
as ``credit_risk_fs.evaluation.paired_inference.paired_stratified_bootstrap``), and the
95% interval is the 2.5/97.5 percentile of the differences.  The numba-accelerated
weighted-AUC evaluation of ``scripts/recompute_pairwise_csv.py`` is reused: it
verifies itself against ``roc_auc_score`` on the unresampled data and on the first
draw, so the interval equals the canonical routine's to floating-point precision.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from credit_risk_fs.evaluation.paired_inference import BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED, paired_delong_test


def _fast_bootstrap(target: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> tuple[float, float]:
    try:
        from scripts import recompute_pairwise_csv as fast
    except Exception:  # pragma: no cover - numba missing; fall back to the canonical routine
        return _canonical_bootstrap(target, score_a, score_b)
    if fast.SEED != BOOTSTRAP_SEED or fast.REPETITIONS != BOOTSTRAP_REPETITIONS:
        raise RuntimeError("fast bootstrap constants diverged from the canonical paired-inference contract")
    return fast._bootstrap_auc_difference(target, score_a, score_b)


def _canonical_bootstrap(target: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> tuple[float, float]:
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    positives = np.flatnonzero(target == 1)
    negatives = np.flatnonzero(target == 0)
    differences = np.empty(BOOTSTRAP_REPETITIONS)
    for index in range(BOOTSTRAP_REPETITIONS):
        sampled = np.concatenate(
            [generator.choice(positives, size=len(positives), replace=True), generator.choice(negatives, size=len(negatives), replace=True)]
        )
        differences[index] = roc_auc_score(target[sampled], score_a[sampled]) - roc_auc_score(target[sampled], score_b[sampled])
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(low), float(high)


def paired_auc_inference(target: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict[str, Any]:
    """AUC_A, AUC_B, delta, 95% paired bootstrap interval, DeLong p-value and n."""

    target = np.asarray(target, dtype=int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    if not (len(target) == len(score_a) == len(score_b)):
        raise ValueError("paired inference inputs have different lengths")
    if len(np.unique(target)) != 2:
        raise ValueError("paired inference needs both target classes")
    delong = paired_delong_test(target, score_a, score_b)
    low, high = _fast_bootstrap(target, score_a, score_b)
    return {
        "auc_A": float(delong["auc_a"]),
        "auc_B": float(delong["auc_b"]),
        "delta": float(delong["auc_difference_a_minus_b"]),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "p_value": float(delong["two_sided_p_value"]),
        "n_ho": int(len(target)),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "repetitions": BOOTSTRAP_REPETITIONS, "stratified": True, "paired_on": "stable_row_id"},
    }
