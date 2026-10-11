"""Canonical mutual-information mRMR.

This is a genuinely different algorithm from
:class:`credit_risk_fs.selectors.mrmr.RandomForestRelevanceMRMRSelector`, which
uses random-forest impurity relevance and absolute-correlation redundancy.

Estimators (recorded in the configuration of every result):

*nearest_neighbour* (default)  Mutual information is estimated on the training
partition by the nearest-neighbour estimators: Ross (2014) for a continuous
feature against a discrete variable (``sklearn.feature_selection.mutual_info_classif``)
and Kraskov et al. (2004) for two continuous features
(``sklearn.feature_selection.mutual_info_regression``); two discrete variables use
the plug-in estimator ``sklearn.metrics.mutual_info_score``. A feature is discrete
when it is categorical, boolean, or takes at most two distinct observed values.
Continuous features are mean-imputed with training statistics. To keep the
estimator tractable on large partitions, it is computed on a target-stratified
sample of ``sample_rows`` training rows drawn with ``random_state``. The pairwise
redundancy terms of each newly selected feature are estimated against all remaining
candidates in one batched call (``n_jobs`` threads); the values do not depend on
``n_jobs``.

*discrete_plugin*  Both relevance and redundancy use ``mutual_info_score`` over
variables discretized inside the training partition into ``n_bins`` quantile bins
with an explicit missing code. Kept for comparison runs.

An optional source-family cap (``group_of`` and ``group_cap``) restricts the greedy
search so that no family contributes more than ``group_cap`` features; it is used
by the family-capped diversity control.
"""

from __future__ import annotations

import inspect
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

ESTIMATORS = frozenset({"nearest_neighbour", "discrete_plugin"})


def _prepare_nearest_neighbour_column(values: pd.Series) -> tuple[bool, np.ndarray]:
    """Return ``(is_discrete, prepared_values)`` for the nearest-neighbour estimator."""

    if pd.api.types.is_numeric_dtype(values.dtype) and not pd.api.types.is_bool_dtype(
        values.dtype
    ):
        numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        observed = numeric.dropna()
        levels = np.unique(observed.to_numpy(dtype="float64"))
        if levels.size <= 2:
            codes = np.full(len(numeric), MISSING_CODE, dtype="int64")
            present = numeric.notna().to_numpy()
            if levels.size:
                codes[present] = np.searchsorted(levels, numeric[present].to_numpy(dtype="float64"))
            return True, codes
        fill = float(observed.mean())
        return False, numeric.fillna(fill).to_numpy(dtype="float64")
    as_object = values.astype("object")
    levels = sorted({str(value) for value in as_object.dropna().to_numpy()})
    lookup = {level: index for index, level in enumerate(levels)}
    return True, np.array(
        [MISSING_CODE if pd.isna(value) else lookup[str(value)] for value in as_object.to_numpy()],
        dtype="int64",
    )


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
    ``I(feature; already_selected)``. Both are mutual information in nats, estimated
    as configured by ``estimator``.
    """

    method_id: ClassVar[str] = "mrmr_mutual_information"
    display_label: ClassVar[str] = "mRMR (mutual information)"
    implementation_id: ClassVar[str] = "mrmr_mutual_information_nearest_neighbour_v1"
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
        estimator: str = "nearest_neighbour",
        sample_rows: int | None = 20_000,
        n_neighbors: int = 3,
        n_bins: int = 10,
        objective: str = "mid",
        redundancy_epsilon: float = 1e-12,
        relevance_floor: float = 0.0,
        group_of: Mapping[str, str] | None = None,
        group_cap: int | None = None,
        random_state: int = 42,
        n_jobs: int = 1,
        excluded_columns: Sequence[str] | None = None,
        fit_scope: str = "dev_fold_training_only",
    ) -> None:
        super().__init__(
            k=k,
            random_state=random_state,
            excluded_columns=excluded_columns,
            fit_scope=fit_scope,
        )
        if int(n_jobs) <= 0:
            raise ValueError("n_jobs must be positive")
        self.n_jobs = int(n_jobs)
        if str(estimator) not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {sorted(ESTIMATORS)}")
        if sample_rows is not None and int(sample_rows) < 2:
            raise ValueError("sample_rows must be at least 2 or None")
        if int(n_neighbors) < 1:
            raise ValueError("n_neighbors must be positive")
        if int(n_bins) < 2:
            raise ValueError("n_bins must be at least 2")
        if str(objective) not in OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(OBJECTIVES)}")
        if (group_of is None) != (group_cap is None):
            raise ValueError("group_of and group_cap must be supplied together")
        if group_cap is not None and int(group_cap) < 1:
            raise ValueError("group_cap must be positive")
        self.estimator = str(estimator)
        self.implementation_id = f"mrmr_mutual_information_{self.estimator}_v1"
        self.sample_rows = None if sample_rows is None else int(sample_rows)
        self.n_neighbors = int(n_neighbors)
        self.n_bins = int(n_bins)
        self.objective = str(objective)
        self.redundancy_epsilon = float(redundancy_epsilon)
        self.relevance_floor = float(relevance_floor)
        self.group_of = None if group_of is None else {str(key): str(value) for key, value in group_of.items()}
        self.group_cap = None if group_cap is None else int(group_cap)
        self.sample_size_: int | None = None
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
        if self.estimator == "nearest_neighbour":
            estimator_details = {
                "mi_estimator": {
                    "continuous_vs_discrete": "Ross (2014), sklearn.feature_selection.mutual_info_classif",
                    "continuous_vs_continuous": "Kraskov et al. (2004), sklearn.feature_selection.mutual_info_regression",
                    "discrete_vs_discrete": "sklearn.metrics.mutual_info_score",
                },
                "n_neighbors": self.n_neighbors,
                "sample_rows": self.sample_rows,
                "sampling": "target-stratified sample of training rows without replacement",
                "discrete_definition": "categorical, boolean, or at most two distinct observed values",
                "continuous_missing_values": "training-sample mean imputation",
            }
        else:
            estimator_details = {
                "mi_estimator": "sklearn.metrics.mutual_info_score",
                "n_bins": self.n_bins,
                "discretization": "training_partition_quantile_with_explicit_missing_code",
            }
        configuration.update(
            {
                "estimator": self.estimator,
                **estimator_details,
                "objective": self.objective,
                "redundancy_epsilon": self.redundancy_epsilon,
                "mi_units": "nats",
                "relevance": "I(feature; target)",
                "redundancy": "mean I(feature; already_selected)",
                "relevance_floor": self.relevance_floor,
                "zero_relevance_policy": "ranked_last_by_ascending_feature_name",
                "group_cap": self.group_cap,
            }
        )
        return configuration

    def _sample_positions(self, target_codes: np.ndarray) -> np.ndarray:
        """Target-stratified training-row sample for the nearest-neighbour estimator."""

        n_rows = len(target_codes)
        if self.sample_rows is None or n_rows <= self.sample_rows:
            return np.arange(n_rows)
        generator = np.random.RandomState(self.random_state)
        positives = np.flatnonzero(target_codes == 1)
        negatives = np.flatnonzero(target_codes != 1)
        n_positive = int(round(self.sample_rows * len(positives) / n_rows))
        n_positive = min(max(n_positive, 1), len(positives))
        n_negative = min(self.sample_rows - n_positive, len(negatives))
        chosen = np.concatenate(
            [
                generator.choice(positives, size=n_positive, replace=False),
                generator.choice(negatives, size=n_negative, replace=False),
            ]
        )
        return np.sort(chosen)

    def _compute_nearest_neighbour(
        self,
        X: pd.DataFrame,
        target_codes: np.ndarray,
        candidate_order: Sequence[str],
    ) -> tuple[list[str], dict[str, float] | None, list[str] | None]:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

        positions = self._sample_positions(target_codes)
        self.sample_size_ = int(len(positions))
        sampled_target = target_codes[positions]
        names = [str(name) for name in candidate_order]
        # Rows first: the sample is small, the full training frame may not be.
        sample = X.iloc[positions]
        discrete: dict[str, bool] = {}
        values: dict[str, np.ndarray] = {}
        for name in names:
            discrete[name], values[name] = _prepare_nearest_neighbour_column(
                sample[name].reset_index(drop=True)
            )
        del sample
        continuous_names = [name for name in names if not discrete[name]]
        discrete_names = [name for name in names if discrete[name]]
        options: dict[str, Any] = {
            "n_neighbors": self.n_neighbors,
            "random_state": self.random_state,
        }
        if "n_jobs" in inspect.signature(mutual_info_classif).parameters:
            options["n_jobs"] = self.n_jobs

        def matrix(group: Sequence[str]) -> np.ndarray:
            return np.column_stack([values[name] for name in group]).astype("float64")

        relevance: dict[str, float] = {}
        if continuous_names:
            scores = mutual_info_classif(
                matrix(continuous_names), sampled_target, discrete_features=False, **options
            )
            relevance.update(zip(continuous_names, (float(value) for value in scores)))
        for name in discrete_names:
            relevance[name] = float(mutual_info_score(values[name], sampled_target))

        def store(anchor: str, others: Sequence[str], scores: Sequence[float]) -> None:
            for other, value in zip(others, scores):
                key = (other, anchor) if other <= anchor else (anchor, other)
                self._mi_cache[key] = max(0.0, float(value))

        def fill(anchor: str) -> None:
            """Estimate I(anchor; feature) for every candidate not yet paired with anchor."""

            def pending(group: Sequence[str]) -> list[str]:
                return [
                    name
                    for name in group
                    if name != anchor
                    and ((name, anchor) if name <= anchor else (anchor, name)) not in self._mi_cache
                ]

            others_continuous = pending(continuous_names)
            others_discrete = pending(discrete_names)
            anchor_values = values[anchor]
            if discrete[anchor]:
                if np.unique(anchor_values).size < 2:
                    store(anchor, others_continuous + others_discrete, [0.0] * (len(others_continuous) + len(others_discrete)))
                    return
                if others_continuous:
                    # Ross: continuous features against the discrete anchor.
                    store(
                        anchor,
                        others_continuous,
                        mutual_info_classif(
                            matrix(others_continuous), anchor_values, discrete_features=False, **options
                        ),
                    )
                store(
                    anchor,
                    others_discrete,
                    [mutual_info_score(values[name], anchor_values) for name in others_discrete],
                )
                return
            if others_continuous:
                # Kraskov et al.: continuous features against the continuous anchor.
                store(
                    anchor,
                    others_continuous,
                    mutual_info_regression(
                        matrix(others_continuous), anchor_values, discrete_features=False, **options
                    ),
                )
            if others_discrete:
                # Ross: discrete features against the continuous anchor.
                store(
                    anchor,
                    others_discrete,
                    mutual_info_regression(
                        matrix(others_discrete), anchor_values, discrete_features=True, **options
                    ),
                )

        def pair_mi(left: str, right: str) -> float:
            key = (left, right) if left <= right else (right, left)
            cached = self._mi_cache.get(key)
            if cached is None:
                fill(right)
                cached = self._mi_cache.get(key)
            if cached is None:
                fill(left)
                cached = self._mi_cache[key]
            return cached

        return self._rank_from_relevance(
            candidate_order=names,
            relevance=relevance,
            pair_mi=pair_mi,
        )

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
        if self.estimator == "nearest_neighbour":
            return self._compute_nearest_neighbour(X, target_codes, candidate_order)
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
        group_counts: dict[str, int] = {}

        def allowed(name: str) -> bool:
            if self.group_cap is None or self.group_of is None:
                return True
            return group_counts.get(self.group_of.get(name, name), 0) < self.group_cap

        def count(name: str) -> None:
            if self.group_of is not None:
                group = self.group_of.get(name, name)
                group_counts[group] = group_counts.get(group, 0) + 1

        if not informative:
            ordered = []
            for name in sorted(uninformative):
                if len(ordered) >= steps:
                    break
                if allowed(name):
                    ordered.append(name)
                    count(name)
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
        count(first)
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
            candidates = [name for name in remaining if allowed(name)]
            if not candidates:
                break
            best_name: str | None = None
            best_key: tuple[float, str] | None = None
            best_parts: tuple[float, float] = (0.0, 0.0)
            for name in candidates:
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
            assert best_name is not None  # loop runs only while candidates is non-empty
            selected.append(best_name)
            remaining.remove(best_name)
            count(best_name)
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
            if not allowed(name):
                continue
            selected.append(name)
            count(name)
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


__all__ = ["ESTIMATORS", "MISSING_CODE", "OBJECTIVES", "MutualInformationMRMRSelector"]
