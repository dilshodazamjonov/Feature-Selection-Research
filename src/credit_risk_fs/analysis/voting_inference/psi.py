"""Score PSI and type-aware feature PSI for the Prompt 6 evidence package.

Score PSI reuses the frozen ``dev_oof_quantile_psi_v1`` implementation without
modification.  Feature PSI reports the frozen numeric implementation alongside a
clearly labelled type-aware extension: the frozen function drops non-finite
values and is undefined for source-level categories, so the extension is
reported as an additional descriptive measure rather than a replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from credit_risk_fs.evaluation.drift import calculate_psi
from credit_risk_fs.pipelines.common import (
    SCORE_PSI_REQUESTED_BIN_COUNT,
    SCORE_PSI_SMOOTHING_EPSILON,
    compute_score_psi,
)

FEATURE_PSI_BIN_COUNT = 10
FEATURE_PSI_EPSILON = 1e-6
MISSING_STATE = "__MISSING__"
UNSEEN_STATE = "__UNSEEN_IN_DEV__"
CATEGORICAL_MAX_DISTINCT = 50


@dataclass(frozen=True)
class ScorePsiResult:
    """One run's DEV-OOF versus OOT score PSI with full bin evidence."""

    psi: float
    bins: pd.DataFrame
    definition: dict[str, Any]


def score_psi_from_predictions(
    dev_oof_scores: Sequence[float] | np.ndarray | pd.Series,
    oot_scores: Sequence[float] | np.ndarray | pd.Series,
) -> ScorePsiResult:
    """Compute score PSI with DEV-OOF quantile bins applied unchanged to OOT."""

    psi, details, definition = compute_score_psi(
        pd.Series(list(dev_oof_scores), dtype=float),
        pd.Series(list(oot_scores), dtype=float),
        requested_bin_count=SCORE_PSI_REQUESTED_BIN_COUNT,
        smoothing_epsilon=SCORE_PSI_SMOOTHING_EPSILON,
    )
    return ScorePsiResult(psi=float(psi), bins=details, definition=dict(definition))


# ---------------------------------------------------------------------------
# Type-aware feature PSI
# ---------------------------------------------------------------------------


def classify_feature_type(dev_values: pd.Series) -> str:
    """Classify one source feature for PSI treatment from its DEV dtype."""

    observed = dev_values.dropna()
    if observed.empty:
        return "all_missing_in_dev"
    if isinstance(dev_values.dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_bool_dtype(dev_values.dtype):
        return "binary"
    if pd.api.types.is_numeric_dtype(dev_values.dtype):
        distinct = observed.unique()
        if len(distinct) == 1:
            return "constant_in_dev"
        if len(distinct) == 2:
            return "binary"
        if len(distinct) <= CATEGORICAL_MAX_DISTINCT and np.all(
            np.equal(np.mod(pd.to_numeric(pd.Series(distinct)).to_numpy(dtype=float), 1), 0)
        ):
            return "encoded_low_cardinality_integer"
        return "numeric"
    return "categorical"


def _numeric_bin_labels(edge_count: int) -> list[str]:
    """Positional bin labels.

    Labels are never derived from the formatted edge values: two distinct
    quantile edges on a large-magnitude, low-spread feature can round to the
    same text (``1000000000.001`` and ``1000000000.002`` both print as
    ``1000000000`` at ten significant digits), which would make the labels
    non-unique and reject the cut. The numeric bounds travel in their own
    columns instead.
    """

    width = max(len(str(edge_count)), 2)
    return [f"bin_{index + 1:0{width}d}" for index in range(edge_count)]


def _numeric_type_aware_psi(
    dev_values: pd.Series, oot_values: pd.Series
) -> tuple[float, pd.DataFrame, dict[str, Any]]:
    dev_numeric = pd.to_numeric(dev_values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    oot_numeric = pd.to_numeric(oot_values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    dev_finite = dev_numeric.dropna()
    if dev_finite.empty:
        return float("nan"), pd.DataFrame(), {"bin_definition_source": "unavailable_empty_dev"}
    candidate = np.percentile(
        dev_finite.to_numpy(dtype=float),
        np.linspace(0.0, 100.0, FEATURE_PSI_BIN_COUNT + 1),
    )
    edges = np.unique(candidate.astype(float))
    if len(edges) < 2:
        return (
            float("nan"),
            pd.DataFrame(),
            {"bin_definition_source": "unavailable_constant_dev"},
        )
    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = _numeric_bin_labels(len(edges) - 1)
    dev_states = pd.cut(dev_numeric, bins=edges, labels=labels, include_lowest=True)
    oot_states = pd.cut(oot_numeric, bins=edges, labels=labels, include_lowest=True)
    dev_series = dev_states.astype("object").where(dev_numeric.notna(), MISSING_STATE)
    oot_series = oot_states.astype("object").where(oot_numeric.notna(), MISSING_STATE)
    states = [*labels, MISSING_STATE]
    frame, psi = _share_table(dev_series, oot_series, states)
    frame["lower_bound"] = [*edges[:-1], np.nan]
    frame["upper_bound"] = [*edges[1:], np.nan]
    definition = {
        "bin_definition_source": "dev_quantile_edges_with_infinite_outer_edges",
        "requested_bin_count": FEATURE_PSI_BIN_COUNT,
        "effective_bin_count": len(labels),
        "duplicate_edge_policy": "sort_unique_candidate_quantile_edges",
        "missing_handling": "explicit_missing_state",
        "unseen_level_handling": "not_applicable_numeric",
        "underflow_overflow_handling": "retained_by_infinite_outer_edges",
        "smoothing_epsilon": FEATURE_PSI_EPSILON,
    }
    return psi, frame, definition


def _categorical_type_aware_psi(
    dev_values: pd.Series, oot_values: pd.Series
) -> tuple[float, pd.DataFrame, dict[str, Any]]:
    dev_series = _categorical_state_series(dev_values)
    oot_series = _categorical_state_series(oot_values)
    dev_level_set = set(dev_series.unique())
    dev_levels = sorted(dev_level_set)
    unseen = set(oot_series.unique()) - dev_level_set
    if unseen:
        # The former row lambda constructed ``set(unseen)`` afresh for every
        # OOT observation.  On a temporally introduced feature with 136k OOT
        # levels that turned a single membership pass into tens of billions of
        # repeated set insertions.  Membership in the DEV set is the exact
        # complement after canonicalization and pandas performs it once in a
        # vectorized hash-table pass.
        oot_series = oot_series.mask(~oot_series.isin(dev_level_set), UNSEEN_STATE)
    states = [*dev_levels, UNSEEN_STATE]
    frame, psi = _share_table(dev_series, oot_series, states)
    definition = {
        "bin_definition_source": "dev_source_level_categories",
        "requested_bin_count": len(dev_levels),
        "effective_bin_count": len(states),
        "duplicate_edge_policy": "not_applicable_categorical",
        "missing_handling": "explicit_missing_state",
        "unseen_level_handling": f"collapsed_into_{UNSEEN_STATE}",
        "unseen_oot_level_count": len(unseen),
        "smoothing_epsilon": FEATURE_PSI_EPSILON,
    }
    return psi, frame, definition


def _categorical_state_series(values: pd.Series) -> pd.Series:
    """Canonicalize source values with the frozen categorical PSI semantics.

    ``astype(str)`` applies the same Python string representation used by the
    former element-wise lambda for the matrix's supported scalar dtypes.  The
    missing mask is captured first so every missing scalar is still replaced
    by the explicit frozen state rather than its textual representation.
    """

    object_values = values.astype("object")
    missing = pd.isna(object_values)
    states = object_values.astype(str)
    if bool(missing.any()):
        states = states.mask(missing, MISSING_STATE)
    return states


def _share_table(
    dev_series: pd.Series, oot_series: pd.Series, states: list[str]
) -> tuple[pd.DataFrame, float]:
    if len(states) != len(set(states)):
        raise ValueError("PSI state labels must be unique")
    dev_counts = dev_series.value_counts().reindex(states, fill_value=0)
    oot_counts = oot_series.value_counts().reindex(states, fill_value=0)
    # Reindexing to a fixed state list would silently drop any observation whose
    # state is absent from that list, which would understate PSI instead of
    # failing. Every row must land in exactly one declared state.
    for scope, series, counts in (
        ("DEV", dev_series, dev_counts),
        ("OOT", oot_series, oot_counts),
    ):
        assigned = int(counts.sum())
        if assigned != len(series):
            raise ValueError(
                f"PSI binning lost {len(series) - assigned} {scope} observation(s): "
                f"{assigned} of {len(series)} assigned to a declared state"
            )
    dev_share = dev_counts / max(len(dev_series), 1)
    oot_share = oot_counts / max(len(oot_series), 1)
    smoothed_dev = dev_share + FEATURE_PSI_EPSILON
    smoothed_oot = oot_share + FEATURE_PSI_EPSILON
    contribution = (smoothed_oot - smoothed_dev) * np.log(smoothed_oot / smoothed_dev)
    frame = pd.DataFrame(
        {
            "state": states,
            "dev_count": dev_counts.to_numpy(dtype=int),
            "oot_count": oot_counts.to_numpy(dtype=int),
            "dev_share": dev_share.to_numpy(dtype=float),
            "oot_share": oot_share.to_numpy(dtype=float),
            "psi_contribution": contribution.to_numpy(dtype=float),
        }
    )
    return frame, float(np.sum(contribution.to_numpy(dtype=float)))


def type_aware_feature_psi(
    dev_values: pd.Series, oot_values: pd.Series
) -> tuple[float, pd.DataFrame, dict[str, Any]]:
    """Compute a type-aware DEV-to-OOT PSI with explicit missing/unseen states."""

    feature_type = classify_feature_type(dev_values)
    if feature_type in {"numeric"}:
        psi, frame, definition = _numeric_type_aware_psi(dev_values, oot_values)
    else:
        psi, frame, definition = _categorical_type_aware_psi(dev_values, oot_values)
    definition["feature_type"] = feature_type
    return psi, frame, definition


def feature_psi_record(
    *,
    feature: str,
    dev_values: pd.Series,
    oot_values: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return the frozen and type-aware PSI evidence for one source feature."""

    frozen_value = calculate_psi(dev_values, oot_values, bins=FEATURE_PSI_BIN_COUNT)
    type_aware_value, distribution, definition = type_aware_feature_psi(
        dev_values, oot_values
    )
    dev_missing = int(pd.isna(dev_values).sum())
    oot_missing = int(pd.isna(oot_values).sum())
    record = {
        "feature": feature,
        "source_feature": feature,
        "feature_type": definition["feature_type"],
        "encoded_feature_relationship": "selection_and_psi_use_the_original_source_column",
        "psi_frozen_numeric": (
            float(frozen_value) if frozen_value is not None and not pd.isna(frozen_value) else None
        ),
        "psi_frozen_numeric_available": bool(
            frozen_value is not None and not pd.isna(frozen_value)
        ),
        "psi_type_aware": None if pd.isna(type_aware_value) else float(type_aware_value),
        "psi_type_aware_available": not bool(pd.isna(type_aware_value)),
        "dev_row_count": int(len(dev_values)),
        "oot_row_count": int(len(oot_values)),
        "dev_missing_count": dev_missing,
        "oot_missing_count": oot_missing,
        "dev_missing_share": dev_missing / max(len(dev_values), 1),
        "oot_missing_share": oot_missing / max(len(oot_values), 1),
        "dev_distinct_count": int(pd.Series(dev_values).nunique(dropna=True)),
        "unseen_oot_level_count": int(definition.get("unseen_oot_level_count", 0)),
        "bin_definition_source": definition["bin_definition_source"],
        "effective_bin_count": definition.get("effective_bin_count"),
        "missing_handling": definition.get("missing_handling"),
        "unseen_level_handling": definition.get("unseen_level_handling"),
        "smoothing_epsilon": definition.get("smoothing_epsilon"),
    }
    return record, distribution


def summarise_feature_psi(
    frame: pd.DataFrame, *, references: Sequence[float]
) -> dict[str, Any]:
    """Summarise one run's selected-feature PSI distribution descriptively."""

    summary: dict[str, Any] = {
        "selected_features_evaluated": int(len(frame)),
    }
    for column, label in (
        ("psi_frozen_numeric", "frozen_numeric"),
        ("psi_type_aware", "type_aware"),
    ):
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        summary[f"{label}_available_count"] = int(len(values))
        summary[f"{label}_unavailable_count"] = int(len(frame) - len(values))
        summary[f"{label}_mean"] = float(values.mean()) if len(values) else None
        summary[f"{label}_median"] = float(values.median()) if len(values) else None
        summary[f"{label}_max"] = float(values.max()) if len(values) else None
        for reference in references:
            key = f"{label}_share_above_{str(reference).replace('.', 'p')}"
            summary[key] = (
                float((values >= float(reference)).mean()) if len(values) else None
            )
    return summary


__all__ = [
    "CATEGORICAL_MAX_DISTINCT",
    "FEATURE_PSI_BIN_COUNT",
    "FEATURE_PSI_EPSILON",
    "MISSING_STATE",
    "UNSEEN_STATE",
    "ScorePsiResult",
    "classify_feature_type",
    "feature_psi_record",
    "score_psi_from_predictions",
    "summarise_feature_psi",
    "type_aware_feature_psi",
]
