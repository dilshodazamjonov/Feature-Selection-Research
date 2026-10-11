"""Standalone Information Value / Weight of Evidence selector.

Why this is implemented here rather than wrapped around the installed
``iv_woe_filter.IVWOEFilter``: that estimator is a filter *and* a transformer.
It selects by an IV threshold rather than a fixed budget, it drops sub-threshold
columns and replaces raw values with WOE inside ``transform``, its tie order is
whatever ``sort_values`` inherited from the input column order, and its
smoothing is a fixed ``1e-12`` applied to distribution shares. None of that can
express a budget-matched, value-preserving, deterministically-ordered selector,
so wrapping it would have distorted the contract more than reimplementing IV.

``tests/selectors/test_lightweight_iv.py`` cross-checks this implementation
against ``IVWOEFilter`` on a fixture where both definitions coincide, and
separately against a hand-calculated WOE/IV table.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from credit_risk_fs.selectors.lightweight.contract import (
    ControlledSelectorFailure,
    LightweightSelector,
    rank_by_score,
)

MISSING_BIN = "__MISSING__"

#: 1 = default event, matching the authenticated positive-class orientation
#: (``class_1_higher_default_risk``). Recorded in the result configuration so an
#: auditor never has to infer which class the WOE sign refers to.
TARGET_ORIENTATION = "class_1_is_default_event"

BINNING_STRATEGIES = frozenset({"quantile", "uniform"})


def _numeric_bin_edges(values: pd.Series, n_bins: int, strategy: str) -> np.ndarray:
    """Learn interior cut points from the training partition only."""

    observed = values.dropna().to_numpy(dtype="float64")
    observed = observed[np.isfinite(observed)]
    if observed.size == 0:
        return np.array([], dtype="float64")
    if strategy == "quantile":
        candidates = np.quantile(observed, np.linspace(0.0, 1.0, n_bins + 1))
    else:
        candidates = np.linspace(observed.min(), observed.max(), n_bins + 1)
    interior = np.unique(candidates[1:-1])
    return interior[np.isfinite(interior)]


def _assign_bins(values: pd.Series, edges: np.ndarray) -> pd.Series:
    """Map values to string bin labels, keeping missingness explicit.

    Labels are positional rather than formatted edge values: two distinct cut
    points on a large-magnitude, low-spread feature can format identically, and
    colliding labels would silently merge two bins. The numeric bounds travel in
    their own columns of the bin table instead.
    """

    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    labels = pd.Series(MISSING_BIN, index=values.index, dtype="object")
    present = numeric.notna()
    if not present.any():
        return labels
    positions = np.searchsorted(edges, numeric[present].to_numpy(dtype="float64"), side="right")
    width = max(len(str(len(edges) + 1)), 2)
    labels.loc[present] = [f"bin_{int(position) + 1:0{width}d}" for position in positions]
    return labels


def _categorical_bins(values: pd.Series) -> pd.Series:
    labels = values.astype("object").where(values.notna(), MISSING_BIN)
    return labels.map(lambda value: MISSING_BIN if value is MISSING_BIN else str(value))


def _numeric_bin_codes(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Integer form of :func:`_assign_bins`: 0 = missing, ``i`` = ``bin_{i:02d}``."""

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    codes = np.zeros(numeric.shape[0], dtype="int64")
    present = np.isfinite(numeric)
    if present.any():
        codes[present] = np.searchsorted(edges, numeric[present], side="right") + 1
    return codes


def compute_feature_iv_from_codes(
    codes: np.ndarray,
    target: np.ndarray,
    *,
    n_edges: int,
    smoothing: float,
) -> tuple[float, pd.DataFrame]:
    """Same quantity as :func:`compute_feature_iv` for numeric bin codes.

    Empty bins are dropped before smoothing, exactly as the label-based group-by
    never creates them, and the bins are summed in the same order (missing first,
    then ``bin_01``, ``bin_02``, ...), so the two functions agree exactly.
    """

    size = int(n_edges) + 2
    counts = np.bincount(codes, minlength=size)
    bads = np.bincount(codes, weights=target, minlength=size)
    keep = np.flatnonzero(counts > 0)
    width = max(len(str(int(n_edges) + 1)), 2)
    labels = [MISSING_BIN if code == 0 else f"bin_{int(code):0{width}d}" for code in keep]
    count = counts[keep].astype("int64")
    bad = np.rint(bads[keep]).astype("int64")
    table = pd.DataFrame({"bin": labels, "count": count, "bad": bad})
    table["good"] = table["count"] - table["bad"]
    smoothed_bad = table["bad"].to_numpy(dtype="float64") + smoothing
    smoothed_good = table["good"].to_numpy(dtype="float64") + smoothing
    total_bad = smoothed_bad.sum()
    total_good = smoothed_good.sum()
    if total_bad <= 0.0 or total_good <= 0.0:
        table["dist_bad"] = 0.0
        table["dist_good"] = 0.0
        table["woe"] = 0.0
        table["iv_contribution"] = 0.0
        return 0.0, table
    dist_bad = smoothed_bad / total_bad
    dist_good = smoothed_good / total_good
    with np.errstate(divide="ignore", invalid="ignore"):
        woe = np.log(dist_good / dist_bad)
    contribution = (dist_good - dist_bad) * woe
    contribution = np.where(np.isfinite(contribution), contribution, 0.0)
    woe = np.where(np.isfinite(woe), woe, 0.0)
    table["dist_bad"] = dist_bad
    table["dist_good"] = dist_good
    table["woe"] = woe
    table["iv_contribution"] = contribution
    return float(contribution.sum()), table


def compute_feature_iv(
    bin_labels: pd.Series,
    target: pd.Series,
    *,
    smoothing: float,
) -> tuple[float, pd.DataFrame]:
    """Compute total IV plus the bin table needed to recalculate it by hand.

    ``smoothing`` is added to the good and bad count of *every* bin before the
    class distributions are formed, so the totals stay consistent with the cell
    counts. With ``smoothing=0.0`` and no empty class cell this reduces exactly
    to the classical definition, which is what makes the hand-calculated oracle
    test meaningful.
    """

    frame = pd.DataFrame({"bin": bin_labels.to_numpy(), "target": target.to_numpy()})
    grouped = frame.groupby("bin", dropna=False, sort=True)["target"]
    counts = grouped.size().rename("count")
    bads = grouped.sum().rename("bad")
    table = pd.concat([counts, bads], axis=1)
    table["good"] = table["count"] - table["bad"]

    smoothed_bad = table["bad"].to_numpy(dtype="float64") + smoothing
    smoothed_good = table["good"].to_numpy(dtype="float64") + smoothing
    total_bad = smoothed_bad.sum()
    total_good = smoothed_good.sum()
    if total_bad <= 0.0 or total_good <= 0.0:
        # Reachable only when a whole partition lacks one class; the caller
        # rejects a single-class target before this point.
        table["dist_bad"] = 0.0
        table["dist_good"] = 0.0
        table["woe"] = 0.0
        table["iv_contribution"] = 0.0
        return 0.0, table.reset_index()

    dist_bad = smoothed_bad / total_bad
    dist_good = smoothed_good / total_good
    with np.errstate(divide="ignore", invalid="ignore"):
        woe = np.log(dist_good / dist_bad)
    # A zero class cell with smoothing disabled yields a non-finite WOE. Such a
    # bin contributes nothing rather than poisoning the feature total with inf.
    contribution = (dist_good - dist_bad) * woe
    finite = np.isfinite(contribution)
    contribution = np.where(finite, contribution, 0.0)
    woe = np.where(np.isfinite(woe), woe, 0.0)

    table["dist_bad"] = dist_bad
    table["dist_good"] = dist_good
    table["woe"] = woe
    table["iv_contribution"] = contribution
    return float(contribution.sum()), table.reset_index()


#: Eligibility interval for information value: features with 0.01 <= IV <= 0.50 on
#: the training partition are eligible. The lower bound removes uninformative
#: features; the upper bound is the leakage warning of scorecard practice.
DEFAULT_ELIGIBILITY_INTERVAL = (0.01, 0.50)


class InformationValueSelector(LightweightSelector):
    """Rank eligible candidate features by total Information Value and take the top k.

    A feature is eligible when its IV lies in ``eligibility_interval`` (inclusive
    bounds); only eligible features are ranked. Binning, WOE, and IV are all fitted
    on the rows handed to :meth:`fit`, which the fold runner restricts to the
    training portion of a DEV fold. Nothing in this selector reads a validation
    row, and it never transforms values -- it selects columns only, so downstream
    preprocessing stays unchanged.
    """

    method_id: ClassVar[str] = "iv_woe"
    display_label: ClassVar[str] = "Information Value (WOE binning)"
    implementation_id: ClassVar[str] = "iv_woe_quantile_binned_v1"
    supervised: ClassVar[bool] = True
    supports_ranking: ClassVar[bool] = True
    supports_natural_support: ClassVar[bool] = False
    supports_fixed_budget: ClassVar[bool] = True
    default_selection_mode: ClassVar[str] = "matched_budget"
    score_orientation: ClassVar[str] = "higher_is_better"

    def __init__(
        self,
        *,
        k: int | None = None,
        n_bins: int = 10,
        binning_strategy: str = "quantile",
        zero_count_smoothing: float = 0.5,
        max_categorical_levels: int = 100_000,
        eligibility_interval: tuple[float, float] | None = DEFAULT_ELIGIBILITY_INTERVAL,
        random_state: int = 42,
        excluded_columns: Sequence[str] | None = None,
        fit_scope: str = "dev_fold_training_only",
    ) -> None:
        super().__init__(
            k=k,
            random_state=random_state,
            excluded_columns=excluded_columns,
            fit_scope=fit_scope,
        )
        if int(n_bins) < 2:
            raise ValueError("n_bins must be at least 2")
        if str(binning_strategy) not in BINNING_STRATEGIES:
            raise ValueError(
                f"binning_strategy must be one of {sorted(BINNING_STRATEGIES)}"
            )
        if float(zero_count_smoothing) < 0.0:
            raise ValueError("zero_count_smoothing must be non-negative")
        if eligibility_interval is not None:
            low, high = (float(value) for value in eligibility_interval)
            if low > high:
                raise ValueError("eligibility_interval lower bound exceeds the upper bound")
            eligibility_interval = (low, high)
        self.n_bins = int(n_bins)
        self.binning_strategy = str(binning_strategy)
        self.zero_count_smoothing = float(zero_count_smoothing)
        self.max_categorical_levels = int(max_categorical_levels)
        self.eligibility_interval = eligibility_interval
        self.eligible_features_: list[str] | None = None
        self.iv_scores_: dict[str, float] | None = None
        self.bin_table_: pd.DataFrame | None = None
        self.feature_diagnostics_: pd.DataFrame | None = None

    def describe_configuration(self) -> dict[str, Any]:
        configuration = super().describe_configuration()
        configuration.update(
            {
                "n_bins": self.n_bins,
                "binning_strategy": self.binning_strategy,
                "zero_count_smoothing": self.zero_count_smoothing,
                "max_categorical_levels": self.max_categorical_levels,
                "eligibility_interval": (
                    None if self.eligibility_interval is None else list(self.eligibility_interval)
                ),
                "target_orientation": TARGET_ORIENTATION,
                "missing_handling": "explicit_missing_bin",
                "woe_definition": "log(dist_good / dist_bad)",
                "iv_definition": "sum((dist_good - dist_bad) * woe)",
            }
        )
        return configuration

    def _bin_one(self, values: pd.Series) -> tuple[pd.Series, str, np.ndarray]:
        if pd.api.types.is_numeric_dtype(values.dtype) and not pd.api.types.is_bool_dtype(
            values.dtype
        ):
            edges = _numeric_bin_edges(values, self.n_bins, self.binning_strategy)
            return _assign_bins(values, edges), "numeric", edges
        distinct = values.dropna().astype("object").nunique()
        if distinct > self.max_categorical_levels:
            raise ControlledSelectorFailure(
                method_id=self.method_id,
                stage="binning",
                cause=(
                    f"categorical feature has {distinct} levels, above the configured "
                    f"max_categorical_levels={self.max_categorical_levels}"
                ),
                configuration=self.describe_configuration(),
            )
        return _categorical_bins(values), "categorical", np.array([], dtype="float64")

    def _compute(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        *,
        candidate_order: Sequence[str],
    ) -> tuple[list[str], dict[str, float] | None, list[str] | None]:
        target = y
        if target is None:  # pragma: no cover - guarded by LightweightSelector.fit
            raise ControlledSelectorFailure(
                method_id=self.method_id,
                stage="target_validation",
                cause="Information Value requires the binary default target",
                configuration=self.describe_configuration(),
            )
        target = target.reset_index(drop=True)
        target_array = target.to_numpy(dtype="float64")

        scores: dict[str, float] = {}
        bin_frames: list[pd.DataFrame] = []
        diagnostics: list[dict[str, Any]] = []
        for name in candidate_order:
            values = X[name].reset_index(drop=True)
            if pd.api.types.is_numeric_dtype(values.dtype) and not pd.api.types.is_bool_dtype(
                values.dtype
            ):
                feature_type = "numeric"
                edges = _numeric_bin_edges(values, self.n_bins, self.binning_strategy)
                iv, table = compute_feature_iv_from_codes(
                    _numeric_bin_codes(values, edges),
                    target_array,
                    n_edges=len(edges),
                    smoothing=self.zero_count_smoothing,
                )
            else:
                labels, feature_type, edges = self._bin_one(values)
                iv, table = compute_feature_iv(
                    labels, target, smoothing=self.zero_count_smoothing
                )
            scores[name] = iv

            table.insert(0, "feature", name)
            table["feature_type"] = feature_type
            if feature_type == "numeric" and edges.size:
                lower = np.concatenate([[-np.inf], edges])
                upper = np.concatenate([edges, [np.inf]])
                bounds = {
                    f"bin_{index + 1:0{max(len(str(len(edges) + 1)), 2)}d}": (low, high)
                    for index, (low, high) in enumerate(zip(lower, upper, strict=True))
                }
                table["lower_bound"] = [
                    bounds.get(str(label), (np.nan, np.nan))[0] for label in table["bin"]
                ]
                table["upper_bound"] = [
                    bounds.get(str(label), (np.nan, np.nan))[1] for label in table["bin"]
                ]
            else:
                table["lower_bound"] = np.nan
                table["upper_bound"] = np.nan
            bin_frames.append(table)

            realized = int(table["bin"].nunique())
            observed = values.dropna()
            diagnostics.append(
                {
                    "feature": name,
                    "feature_type": feature_type,
                    "requested_bins": self.n_bins,
                    "realized_bins": realized,
                    "missing_count": int(values.isna().sum()),
                    "distinct_observed_values": int(observed.nunique()),
                    "degenerate": bool(observed.empty or observed.nunique() <= 1),
                    "iv": float(iv),
                }
            )

        self.iv_scores_ = dict(scores)
        self.bin_table_ = (
            pd.concat(bin_frames, ignore_index=True) if bin_frames else pd.DataFrame()
        )
        self.feature_diagnostics_ = pd.DataFrame(diagnostics)
        if self.eligibility_interval is None:
            eligible_scores = dict(scores)
        else:
            low, high = self.eligibility_interval
            eligible_scores = {
                name: value for name, value in scores.items() if low <= value <= high
            }
        ranking = rank_by_score(eligible_scores, candidate_order=candidate_order)
        self.eligible_features_ = list(ranking)
        return ranking, eligible_scores, None


__all__ = [
    "BINNING_STRATEGIES",
    "MISSING_BIN",
    "TARGET_ORIENTATION",
    "InformationValueSelector",
    "compute_feature_iv",
    "compute_feature_iv_from_codes",
]
