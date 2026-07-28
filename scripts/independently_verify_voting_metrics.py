"""Independent audit path for the Prompt 6 headline metrics.

This script deliberately shares no metric helper with the primary analysis.  It
re-reads the same saved prediction artifacts and recomputes AUC, Gini, KS,
Lift@10, score PSI, aligned row counts, and target mismatch counts from
transparent formulas written here, then compares them against the primary
package at a frozen numerical tolerance.

It imports nothing from ``credit_risk_fs.evaluation``,
``credit_risk_fs.pipelines``, or ``credit_risk_fs.analysis``; only the standard
library, numpy, and pandas are used.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\independently_verify_voting_metrics.py \\
        --primary-metrics results/final_experiments/cross_dataset_voting_inference_v1/run_level_metrics.csv \\
        --primary-score-psi results/final_experiments/cross_dataset_voting_inference_v1/score_psi_summary.csv \\
        --primary-alignment results/final_experiments/cross_dataset_voting_inference_v1/alignment_audit.csv \\
        --primary-lift-audit results/final_experiments/cross_dataset_voting_inference_v1/lift10_audit.csv \\
        --output results/final_experiments/cross_dataset_voting_inference_v1/independent_recalculation_audit.csv
"""

from __future__ import annotations

import argparse
import math
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = REPOSITORY_ROOT / "results" / "runs"
SPLIT_FILE = {"DEV_OOF": "dev_predictions.csv", "OOT": "oot_predictions.csv"}
PSI_BIN_COUNT = 10
PSI_EPSILON = 1e-6
LIFT_FRACTION = 0.10


# ---------------------------------------------------------------------------
# Self-contained metric formulas
# ---------------------------------------------------------------------------


def independent_auc(target: Sequence[int], score: Sequence[float]) -> float:
    """AUC as the normalised Mann-Whitney U statistic with averaged tied ranks.

    Ranks are built by hand: sort ascending, walk each block of equal scores,
    and assign every member of the block the mean of the 1-based positions it
    spans.  AUC = (sum of positive ranks - n_pos*(n_pos+1)/2) / (n_pos*n_neg).
    """

    values = np.asarray(score, dtype=float)
    labels = np.asarray(target, dtype=int)
    count = values.size
    positives = int(labels.sum())
    negatives = count - positives
    if positives == 0 or negatives == 0:
        raise ValueError("independent AUC requires both classes")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(count, dtype=float)
    index = 0
    while index < count:
        end = index + 1
        while end < count and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def independent_gini(auc: float) -> float:
    """Gini as the direct algebraic transform of AUC."""

    return 2.0 * float(auc) - 1.0


def independent_ks(target: Sequence[int], score: Sequence[float]) -> float:
    """KS as the largest absolute gap between the two empirical CDFs.

    Distinct score values are visited in ascending order; at each value the
    cumulative share of positives and of negatives at or below it is compared.
    """

    frame = pd.DataFrame({"target": np.asarray(target, dtype=int), "score": np.asarray(score, dtype=float)})
    positives = int(frame["target"].sum())
    negatives = int(len(frame) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("independent KS requires both classes")
    grouped = frame.groupby("score", sort=True)["target"].agg(["sum", "size"])
    positive_share = grouped["sum"].cumsum() / positives
    negative_share = (grouped["size"] - grouped["sum"]).cumsum() / negatives
    return float((positive_share - negative_share).abs().max())


def independent_lift_at_10(
    target: Sequence[int],
    score: Sequence[float],
    identity: Sequence[str],
    *,
    fraction: float = LIFT_FRACTION,
) -> float:
    """Lift@10 by explicit Python ordering of every row.

    The top group is the ``ceil(fraction * n)`` rows with the highest predicted
    class-1 probability; equal scores are broken by ascending casefolded
    identity.  Lift is the top-group bad rate divided by the overall bad rate.
    """

    labels = list(int(value) for value in target)
    scores = list(float(value) for value in score)
    identities = [unicodedata.normalize("NFC", str(value)).casefold() for value in identity]
    row_count = len(labels)
    if row_count == 0:
        raise ValueError("independent lift requires rows")
    overall = sum(labels) / row_count
    if overall == 0:
        raise ValueError("independent lift requires at least one positive")
    top_count = int(math.ceil(fraction * row_count))
    order = sorted(range(row_count), key=lambda index: (-scores[index], identities[index]))
    top_positives = sum(labels[index] for index in order[:top_count])
    return (top_positives / top_count) / overall


def independent_score_psi(
    reference: Sequence[float], comparison: Sequence[float]
) -> float:
    """Score PSI with quantile bins taken from the reference scores only.

    Eleven candidate quantile edges are computed on the reference distribution,
    duplicates are collapsed, the outer edges are pinned to 0 and 1, internal
    edges are right-inclusive, and each bin contributes
    ``(q_i - p_i) * ln(q_i / p_i)`` on epsilon-smoothed shares.
    """

    left = np.asarray(reference, dtype=float)
    right = np.asarray(comparison, dtype=float)
    if left.size == 0 or right.size == 0:
        raise ValueError("independent score PSI requires both distributions")
    edges = np.unique(
        np.percentile(left, np.linspace(0.0, 100.0, PSI_BIN_COUNT + 1)).astype(float)
    )
    if edges.size < 2:
        edges = np.array([0.0, 1.0], dtype=float)
    else:
        edges[0] = 0.0
        edges[-1] = 1.0
        edges = np.unique(edges)
    bin_count = edges.size - 1
    interior = edges[1:-1]

    def shares(values: np.ndarray) -> np.ndarray:
        assigned = np.searchsorted(interior, values, side="left")
        counts = np.bincount(assigned, minlength=bin_count).astype(float)
        return counts / values.size

    reference_share = shares(left) + PSI_EPSILON
    comparison_share = shares(right) + PSI_EPSILON
    return float(
        np.sum((comparison_share - reference_share) * np.log(comparison_share / reference_share))
    )


def independent_alignment(
    left: pd.DataFrame, right: pd.DataFrame
) -> tuple[int, int]:
    """Aligned row count and target mismatch count via explicit dictionaries."""

    left_map: dict[str, tuple[int, float]] = {}
    for identity, target, score in zip(
        left["identity"], left["target"], left["score"], strict=True
    ):
        if identity in left_map:
            raise ValueError("duplicate identity in the reference artifact")
        left_map[identity] = (int(target), float(score))
    aligned = 0
    mismatches = 0
    seen: set[str] = set()
    for identity, target in zip(right["identity"], right["target"], strict=True):
        if identity in seen:
            raise ValueError("duplicate identity in the comparator artifact")
        seen.add(identity)
        record = left_map.get(identity)
        if record is None:
            continue
        aligned += 1
        if record[0] != int(target):
            mismatches += 1
    return aligned, mismatches


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_split(dataset: str, run_id: str, split: str) -> pd.DataFrame:
    path = RUN_ROOT / dataset / run_id / "results" / SPLIT_FILE[split]
    frame = pd.read_csv(
        path,
        usecols=["stable_row_id", "target", "prediction_probability"],
        dtype={"stable_row_id": "string"},
    )
    return pd.DataFrame(
        {
            "identity": [
                unicodedata.normalize("NFC", str(value).strip())
                for value in frame["stable_row_id"]
            ],
            "target": frame["target"].astype(int),
            "score": frame["prediction_probability"].astype(float),
        }
    )


def _record(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    dataset: str,
    model: str,
    configuration: str,
    scope: str,
    primary: float | int,
    independent: float | int,
    tolerance: float,
    is_count: bool = False,
) -> None:
    difference = abs(float(primary) - float(independent))
    limit = 0.0 if is_count else float(tolerance)
    rows.append(
        {
            "metric": metric,
            "dataset": dataset,
            "model": model,
            "configuration": configuration,
            "scope": scope,
            "primary_value": float(primary),
            "independent_value": float(independent),
            "absolute_difference": difference,
            "tolerance": limit,
            "pass": bool(difference <= limit),
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-metrics", required=True)
    parser.add_argument("--primary-score-psi", required=True)
    parser.add_argument("--primary-alignment", required=True)
    parser.add_argument("--primary-lift-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    metrics = pd.read_csv(arguments.primary_metrics)
    score_psi = pd.read_csv(arguments.primary_score_psi)
    alignment = pd.read_csv(arguments.primary_alignment)
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, str], pd.DataFrame] = {}

    def cached(dataset: str, run_id: str, split: str) -> pd.DataFrame:
        key = (dataset, run_id, split)
        if key not in cache:
            cache.clear()
            cache[key] = load_split(dataset, run_id, split)
        return cache[key]

    for row in metrics.itertuples(index=False):
        frame = cached(row.dataset, row.run_id, row.split)
        auc = independent_auc(frame["target"], frame["score"])
        _record(
            rows,
            metric="auc",
            dataset=row.dataset,
            model=row.model,
            configuration=row.configuration,
            scope=f"{row.run_id}:{row.split}",
            primary=row.auc,
            independent=auc,
            tolerance=arguments.tolerance,
        )
        _record(
            rows,
            metric="gini",
            dataset=row.dataset,
            model=row.model,
            configuration=row.configuration,
            scope=f"{row.run_id}:{row.split}",
            primary=row.gini,
            independent=independent_gini(auc),
            tolerance=arguments.tolerance,
        )
        _record(
            rows,
            metric="ks",
            dataset=row.dataset,
            model=row.model,
            configuration=row.configuration,
            scope=f"{row.run_id}:{row.split}",
            primary=row.ks,
            independent=independent_ks(frame["target"], frame["score"]),
            tolerance=arguments.tolerance,
        )
        _record(
            rows,
            metric="lift_at_10",
            dataset=row.dataset,
            model=row.model,
            configuration=row.configuration,
            scope=f"{row.run_id}:{row.split}",
            primary=row.lift_at_10,
            independent=independent_lift_at_10(
                frame["target"], frame["score"], frame["identity"]
            ),
            tolerance=arguments.tolerance,
        )
        print(f"verified {row.run_id} {row.split}", flush=True)

    for row in score_psi.itertuples(index=False):
        dev = load_split(row.dataset, row.run_id, "DEV_OOF")
        oot = load_split(row.dataset, row.run_id, "OOT")
        _record(
            rows,
            metric="score_psi",
            dataset=row.dataset,
            model=row.model,
            configuration=row.configuration,
            scope=f"{row.run_id}:DEV_OOF_vs_OOT",
            primary=row.score_psi,
            independent=independent_score_psi(dev["score"], oot["score"]),
            tolerance=arguments.tolerance,
        )
        print(f"verified score PSI {row.run_id}", flush=True)

    for row in alignment.itertuples(index=False):
        split = "DEV_OOF" if str(row.split).upper().startswith("DEV") else "OOT"
        left = load_split(row.dataset, row.reference_run_id, split)
        right = load_split(row.dataset, row.comparator_run_id, split)
        aligned, mismatches = independent_alignment(left, right)
        scope = f"{row.reference_run_id}->{row.comparator_run_id}:{split}"
        _record(
            rows,
            metric="aligned_row_count",
            dataset=row.dataset,
            model=row.model,
            configuration=row.configuration,
            scope=scope,
            primary=row.aligned_row_count,
            independent=aligned,
            tolerance=arguments.tolerance,
            is_count=True,
        )
        _record(
            rows,
            metric="target_mismatch_count",
            dataset=row.dataset,
            model=row.model,
            configuration=row.configuration,
            scope=scope,
            primary=row.target_mismatch_count,
            independent=mismatches,
            tolerance=arguments.tolerance,
            is_count=True,
        )
        print(f"verified alignment {scope}", flush=True)

    audit = pd.DataFrame(rows)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)
    failed = audit.loc[~audit["pass"]]
    print(
        f"independent recalculation: {int(audit['pass'].sum())}/{len(audit)} within "
        f"tolerance; maximum absolute difference "
        f"{float(audit['absolute_difference'].max()):.3e}",
        flush=True,
    )
    if not failed.empty:
        for row in failed.itertuples(index=False):
            print(
                f"FAIL {row.metric} {row.scope}: primary={row.primary_value!r} "
                f"independent={row.independent_value!r} diff={row.absolute_difference!r}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
