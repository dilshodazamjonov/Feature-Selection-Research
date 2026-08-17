"""Canonical mutual-information mRMR.

This is a genuinely different algorithm from
:class:`credit_risk_fs.selectors.mrmr.RandomForestRelevanceMRMRSelector`, which
uses random-forest impurity relevance and absolute-correlation redundancy. That
legacy method keeps its own identity and its historical artifacts; nothing here
rewrites or reinterprets them.

Implementation decision (recorded because it is a real choice, not a default):

*Estimator*  Both relevance and redundancy use the discrete plug-in estimator
``sklearn.metrics.mutual_info_score`` over variables discretized inside the
training partition. No new dependency is installed and no randomness enters, so
the result is bit-reproducible for a given input, configuration, and
discretization.

*Rejected alternative*  The k-nearest-neighbour (Kraskov) estimator behind
``sklearn.feature_selection.mutual_info_classif`` adds tie-breaking noise driven
by ``random_state`` and estimates continuous-continuous MI by a different
construction. It is defensible, but it changes both the algorithm and its
reproducibility characteristics, so adopting it must be a separate, explicitly
versioned ``implementation_id`` rather than a silent redefinition of this one.

*Consequence*  The discretization rule is part of the algorithm's identity and is
therefore recorded in the configuration of every result. Changing ``n_bins``
changes the estimator, not merely a tuning knob.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from credit_risk_fs.selectors.lightweight.contract import (
    ControlledSelectorFailure,
    LightweightSelector,
)
from credit_risk_fs.selectors.lightweight.mrmr_compact_cache import (
    DEFAULT_FEATURE_BATCH_SIZE,
    CompactMRMRCheckpointStore,
)

MISSING_CODE = -1

#: ``mid`` maximizes ``relevance - mean_redundancy`` (Peng et al. difference
#: form). ``miq`` maximizes ``relevance / max(mean_redundancy, epsilon)``. The
#: difference form is the default because the quotient form is unstable when the
#: mean redundancy approaches zero.
OBJECTIVES = frozenset({"mid", "miq"})


def _discretize_column(values: pd.Series, n_bins: int) -> np.ndarray:
    """Map one column to integer codes using training-partition statistics only.

    Missing values get their own code rather than being dropped or imputed, so a
    feature whose missingness is itself informative is measured honestly.
    """

    if pd.api.types.is_numeric_dtype(values.dtype) and not pd.api.types.is_bool_dtype(
        values.dtype
    ):
        numeric = pd.to_numeric(values, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        observed = numeric.dropna().to_numpy(dtype="float64")
        codes = np.full(len(numeric), MISSING_CODE, dtype="int64")
        present = numeric.notna().to_numpy()
        if observed.size == 0:
            return codes
        edges = np.unique(np.quantile(observed, np.linspace(0.0, 1.0, n_bins + 1))[1:-1])
        if edges.size == 0:
            codes[present] = 0
            return codes
        codes[present] = np.searchsorted(
            edges, numeric[present].to_numpy(dtype="float64"), side="right"
        )
        return codes
    # Categorical, boolean, and object columns: one code per observed level, with
    # a stable ordering derived from the sorted level names so the codes do not
    # depend on row order.
    as_object = values.astype("object")
    levels = sorted({str(value) for value in as_object.dropna().to_numpy()})
    lookup = {level: index for index, level in enumerate(levels)}
    return np.array(
        [
            MISSING_CODE if pd.isna(value) else lookup[str(value)]
            for value in as_object.to_numpy()
        ],
        dtype="int64",
    )


class MutualInformationMRMRSelector(LightweightSelector):
    """Sequential maximum-relevance / minimum-redundancy selection.

    At each step the feature maximizing the configured objective is appended.
    Relevance is ``I(feature; target)``; redundancy is the mean
    ``I(feature; already_selected)``. Both are discrete plug-in mutual
    information in nats.
    """

    method_id: ClassVar[str] = "mrmr_mutual_information"
    display_label: ClassVar[str] = "mRMR (mutual information)"
    implementation_id: ClassVar[str] = "mrmr_mutual_information_discrete_plugin_v1"
    supervised: ClassVar[bool] = True
    supports_ranking: ClassVar[bool] = True
    supports_natural_support: ClassVar[bool] = False
    supports_fixed_budget: ClassVar[bool] = True
    default_selection_mode: ClassVar[str] = "matched_budget"
    #: The greedy objective value is only comparable within its own step, so the
    #: published ordering -- not the score magnitude -- is the ranking evidence.
    score_orientation: ClassVar[str] = "rank_1_is_best"

    def __init__(
        self,
        *,
        k: int | None = None,
        n_bins: int = 10,
        objective: str = "mid",
        redundancy_epsilon: float = 1e-12,
        relevance_floor: float = 0.0,
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
        if str(objective) not in OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}")
        self.n_bins = int(n_bins)
        self.objective = str(objective)
        self.redundancy_epsilon = float(redundancy_epsilon)
        self.relevance_floor = float(relevance_floor)
        self.relevance_: dict[str, float] | None = None
        self.selection_trace_: pd.DataFrame | None = None
        self._mi_cache: dict[tuple[str, str], float] = {}
        self._codes: dict[str, np.ndarray] = {}
        self._execution_cache_root: Path | None = None
        self._execution_cache_identity: dict[str, Any] | None = None
        self._execution_cache_batch_size = DEFAULT_FEATURE_BATCH_SIZE
        self._execution_progress_callback: (
            Callable[[str, Mapping[str, Any]], None] | None
        ) = None
        self.execution_checkpoint_summary_: dict[str, Any] | None = None

    def configure_execution_cache(
        self,
        root: str | Path,
        *,
        execution_identity: Mapping[str, Any],
        feature_batch_size: int = DEFAULT_FEATURE_BATCH_SIZE,
        progress_callback: (
            Callable[[str, Mapping[str, Any]], None] | None
        ) = None,
    ) -> MutualInformationMRMRSelector:
        """Enable a restartable storage strategy without changing the method.

        This execution-only setting is intentionally absent from
        :meth:`describe_configuration`: it changes neither the scientific
        implementation identity nor any ranking semantics.
        """

        self._execution_cache_root = Path(root)
        self._execution_cache_identity = dict(execution_identity)
        self._execution_cache_batch_size = int(feature_batch_size)
        self._execution_progress_callback = progress_callback
        if self._execution_cache_batch_size <= 0:
            raise ValueError("mRMR execution-cache batch size must be positive")
        return self

    def describe_configuration(self) -> dict[str, Any]:
        configuration = super().describe_configuration()
        configuration.update(
            {
                "n_bins": self.n_bins,
                "objective": self.objective,
                "redundancy_epsilon": self.redundancy_epsilon,
                "mi_estimator": "sklearn.metrics.mutual_info_score",
                "mi_units": "nats",
                "discretization": "training_partition_quantile_with_explicit_missing_code",
                "relevance": "I(feature; target)",
                "redundancy": "mean I(feature; already_selected)",
                "relevance_floor": self.relevance_floor,
                "zero_relevance_policy": "ranked_last_by_ascending_feature_name",
                "deterministic_without_rng": True,
            }
        )
        return configuration

    def _pair_mi(self, left: str, right: str) -> float:
        key = (left, right) if left <= right else (right, left)
        cached = self._mi_cache.get(key)
        if cached is None:
            cached = float(mutual_info_score(self._codes[key[0]], self._codes[key[1]]))
            self._mi_cache[key] = cached
        return cached

    def _compute(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        *,
        candidate_order: Sequence[str],
    ) -> tuple[list[str], dict[str, float] | None, list[str] | None]:
        if y is None:  # pragma: no cover - guarded by LightweightSelector.fit
            raise ControlledSelectorFailure(
                method_id=self.method_id,
                stage="target_validation",
                cause="mutual-information mRMR requires the binary default target",
                configuration=self.describe_configuration(),
            )
        target_codes = np.asarray(y.reset_index(drop=True).to_numpy(), dtype="int64")

        self._mi_cache = {}
        self.execution_checkpoint_summary_ = None
        if self._execution_cache_root is None:
            self._codes = {
                name: _discretize_column(X[name].reset_index(drop=True), self.n_bins)
                for name in candidate_order
            }
            relevance = {
                name: float(mutual_info_score(self._codes[name], target_codes))
                for name in candidate_order
            }
            return self._rank_from_relevance(
                candidate_order=candidate_order,
                relevance=relevance,
                pair_mi=self._pair_mi,
            )

        if self._execution_cache_identity is None:
            raise ControlledSelectorFailure(
                method_id=self.method_id,
                stage="execution_cache_validation",
                cause="mRMR execution cache lacks an authenticated identity",
                configuration=self.describe_configuration(),
            )
        self._codes = {}
        store = CompactMRMRCheckpointStore.prepare(
            self._execution_cache_root,
            X=X,
            execution_identity=self._execution_cache_identity,
            candidate_order=candidate_order,
            n_bins=self.n_bins,
            discretize=_discretize_column,
            feature_batch_size=self._execution_cache_batch_size,
            progress_callback=self._execution_progress_callback,
        )
        pair_vectors: dict[str, np.memmap] = {}
        feature_positions = {
            name: index for index, name in enumerate(store.candidate_order)
        }

        def cached_pair_mi(left: str, right: str) -> float:
            selected = str(right)
            mapping = pair_vectors.get(selected)
            if mapping is None:
                mapping = store.pair_vector(selected)
                pair_vectors[selected] = mapping
            return float(mapping[feature_positions[str(left)]])

        try:
            relevance = store.relevance(target_codes)
            result = self._rank_from_relevance(
                candidate_order=candidate_order,
                relevance=relevance,
                pair_mi=cached_pair_mi,
            )
            self.execution_checkpoint_summary_ = store.summary()
            return result
        finally:
            for mapping in pair_vectors.values():
                owner = getattr(mapping, "_mmap", None)
                if owner is not None and not owner.closed:
                    owner.close()
            store.close()

    def _rank_from_relevance(
        self,
        *,
        candidate_order: Sequence[str],
        relevance: Mapping[str, float],
        pair_mi: Callable[[str, str], float],
    ) -> tuple[list[str], dict[str, float] | None, list[str] | None]:
        """Run the frozen greedy ranking over precomputed exact MI values."""

        relevance = {name: float(relevance[name]) for name in candidate_order}
        self.relevance_ = dict(relevance)

        steps = len(candidate_order) if self.k is None else min(int(self.k), len(candidate_order))
        if steps <= 0:
            self.selection_trace_ = pd.DataFrame(
                columns=["selection_rank", "feature", "relevance", "mean_redundancy", "score"]
            )
            return [], {}, None

        # A feature whose mutual information with the target is at or below the
        # floor carries no information about default, so it can never improve the
        # objective on its merits. Under the difference objective such a feature
        # scores 0 - 0 = 0 and would outrank a genuinely relevant feature that
        # merely overlaps an earlier pick, which would let a constant column
        # displace a predictor. Zero-relevance features are therefore held back
        # and appended only once the informative pool is exhausted. This is a
        # declared policy, not a silent filter: it is recorded in the result
        # configuration and the held-back features still appear in the ranking.
        informative = [
            name for name in candidate_order if relevance[name] > self.relevance_floor
        ]
        uninformative = [
            name for name in candidate_order if relevance[name] <= self.relevance_floor
        ]

        if not informative:
            ordered = sorted(uninformative)[:steps]
            self.selection_trace_ = pd.DataFrame(
                {
                    "selection_rank": range(1, len(ordered) + 1),
                    "feature": ordered,
                    "relevance": [relevance[name] for name in ordered],
                    "mean_redundancy": [0.0] * len(ordered),
                    "score": [relevance[name] for name in ordered],
                }
            )
            return ordered, {name: relevance[name] for name in ordered}, None

        remaining = list(informative)
        # First pick is pure maximum relevance; ties fall to the ascending
        # feature name, never to incidental column order.
        first = min(remaining, key=lambda name: (-relevance[name], name))
        selected = [first]
        remaining.remove(first)
        scores = {first: relevance[first]}
        trace: list[dict[str, Any]] = [
            {
                "selection_rank": 1,
                "feature": first,
                "relevance": relevance[first],
                "mean_redundancy": 0.0,
                "score": relevance[first],
            }
        ]

        while len(selected) < steps and remaining:
            best_name: str | None = None
            best_key: tuple[float, str] | None = None
            best_parts: tuple[float, float] = (0.0, 0.0)
            for name in remaining:
                redundancy = sum(pair_mi(name, chosen) for chosen in selected) / len(
                    selected
                )
                if self.objective == "mid":
                    score = relevance[name] - redundancy
                else:
                    score = relevance[name] / max(redundancy, self.redundancy_epsilon)
                key = (-score, name)
                if best_key is None or key < best_key:
                    best_key = key
                    best_name = name
                    best_parts = (redundancy, score)
            assert best_name is not None  # loop runs only while remaining is non-empty
            selected.append(best_name)
            remaining.remove(best_name)
            scores[best_name] = best_parts[1]
            trace.append(
                {
                    "selection_rank": len(selected),
                    "feature": best_name,
                    "relevance": relevance[best_name],
                    "mean_redundancy": best_parts[0],
                    "score": best_parts[1],
                }
            )

        for name in sorted(uninformative):
            if len(selected) >= steps:
                break
            selected.append(name)
            scores[name] = relevance[name]
            trace.append(
                {
                    "selection_rank": len(selected),
                    "feature": name,
                    "relevance": relevance[name],
                    "mean_redundancy": float("nan"),
                    "score": float("nan"),
                }
            )

        self.selection_trace_ = pd.DataFrame(trace)
        return selected, scores, None


__all__ = ["MISSING_CODE", "OBJECTIVES", "MutualInformationMRMRSelector"]
