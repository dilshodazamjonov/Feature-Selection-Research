"""Canonical registry metadata for the Prompt 7 selector controls.

One canonical ID per algorithm, one algorithm per canonical ID. Historical
aliases are preserved so old configurations keep loading, but an alias always
resolves to the algorithm it originally meant -- never to a newer method that
merely shares a name.

The central case is ``mrmr``. That alias historically meant the random-forest
relevance / absolute-correlation redundancy selector, so it keeps resolving
there. Canonical mutual-information mRMR is a separate ID and cannot be reached
through the legacy alias.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from credit_risk_fs.selectors.lightweight.contract import SELECTION_MODES


@dataclass(frozen=True)
class MethodDescriptor:
    """Declared capabilities of one registered selection method."""

    method_id: str
    display_label: str
    implementation_id: str
    algorithm: str
    supervised: bool
    selection_modes: tuple[str, ...]
    provides_ranking: bool
    provides_natural_support: bool
    supports_fixed_budget: bool
    budget_kwarg: str | None
    import_path: str
    default_kwargs: Mapping[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    historical_use: str = ""
    notes: str = ""
    #: ``light`` finishes in milliseconds on a fold; ``heavy`` fits a real
    #: estimator, possibly many times, and needs stage logging and resource
    #: supervision. Declared so a caller can budget before instantiating.
    cost_class: str = "light"
    #: Estimator family actually used, or ``None`` for the pure controls.
    estimator_family: str | None = None
    #: ``True`` when a successful fit always returns exactly ``k`` features.
    guarantees_exact_k: bool = False
    #: Score name and orientation for the published ranking.
    score_name: str = "score"
    #: Serialization contract this method's results conform to.
    serialization_version: str = "lightweight_selector_contract_v1"
    #: Whether the frozen voting protocol is permitted to use this method. Only
    #: the two historical voters may, and neither is a Prompt 8 addition.
    allowed_in_frozen_voting: bool = False
    #: Conditions under which the method raises a controlled failure.
    controlled_failure_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = set(self.selection_modes) - SELECTION_MODES
        if unknown:
            raise ValueError(
                f"{self.method_id} declares unknown selection mode(s): {sorted(unknown)}"
            )
        if self.cost_class not in {"light", "heavy"}:
            raise ValueError(f"{self.method_id} declares unknown cost class")

    def load(self) -> type:
        module_path, _, class_name = self.import_path.rpartition(".")
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)


LIGHTWEIGHT_METHODS: tuple[MethodDescriptor, ...] = (
    MethodDescriptor(
        method_id="iv_woe",
        display_label="Information Value (WOE binning)",
        implementation_id="iv_woe_quantile_binned_v1",
        algorithm="weight_of_evidence_information_value",
        supervised=True,
        selection_modes=("matched_budget",),
        provides_ranking=True,
        provides_natural_support=False,
        supports_fixed_budget=True,
        budget_kwarg="k",
        import_path="credit_risk_fs.selectors.lightweight.iv.InformationValueSelector",
        default_kwargs={
            "k": 40,
            "n_bins": 10,
            "binning_strategy": "quantile",
            "zero_count_smoothing": 0.5,
            "random_state": 42,
        },
        historical_use=(
            "New in Prompt 7. IV previously appeared only as a pre-filter inside "
            "the LLM screening selector, never as a standalone ranked selector."
        ),
    ),
    MethodDescriptor(
        method_id="mrmr_mutual_information",
        display_label="mRMR (mutual information)",
        implementation_id="mrmr_mutual_information_discrete_plugin_v1",
        algorithm="sequential_max_relevance_min_redundancy_mutual_information",
        supervised=True,
        selection_modes=("matched_budget",),
        provides_ranking=True,
        provides_natural_support=False,
        supports_fixed_budget=True,
        budget_kwarg="k",
        import_path=(
            "credit_risk_fs.selectors.lightweight.mi_mrmr.MutualInformationMRMRSelector"
        ),
        default_kwargs={
            "k": 40,
            "n_bins": 10,
            "objective": "mid",
            "random_state": 42,
        },
        historical_use="New in Prompt 7. No historical result used this algorithm.",
        notes=(
            "Not reachable through the legacy 'mrmr' alias, which means the "
            "random-forest relevance / correlation redundancy selector."
        ),
    ),
    MethodDescriptor(
        method_id="lasso_l1_logistic",
        display_label="LASSO (L1-penalized logistic regression)",
        implementation_id="lasso_l1_logistic_v1",
        algorithm="l1_penalized_logistic_regression_support",
        supervised=True,
        selection_modes=("natural", "matched_budget", "coefficient_ranking"),
        provides_ranking=True,
        provides_natural_support=True,
        supports_fixed_budget=True,
        budget_kwarg="k",
        import_path="credit_risk_fs.selectors.lightweight.lasso.L1LogisticSelector",
        default_kwargs={
            "k": 40,
            "C": 0.05,
            "solver": "liblinear",
            "max_iter": 2_000,
            "tol": 1e-4,
            "random_state": 42,
            "n_jobs": 1,
        },
        historical_use="New in Prompt 7. No historical result used this algorithm.",
    ),
    MethodDescriptor(
        method_id="random_k",
        display_label="Random-k control",
        implementation_id="random_k_local_generator_v1",
        algorithm="uniform_random_subset_without_replacement",
        supervised=False,
        selection_modes=("random_control",),
        provides_ranking=True,
        provides_natural_support=False,
        supports_fixed_budget=True,
        budget_kwarg="k",
        import_path="credit_risk_fs.selectors.lightweight.controls.RandomKSelector",
        default_kwargs={"k": 40, "random_state": 42},
        historical_use="New in Prompt 7.",
    ),
    MethodDescriptor(
        method_id="full_features",
        display_label="Full candidate features",
        implementation_id="full_candidate_features_v1",
        algorithm="no_selection_over_eligible_candidates",
        supervised=False,
        selection_modes=("full_control",),
        provides_ranking=True,
        provides_natural_support=False,
        supports_fixed_budget=False,
        budget_kwarg=None,
        import_path=(
            "credit_risk_fs.selectors.lightweight.controls.FullCandidateFeaturesSelector"
        ),
        default_kwargs={"random_state": 42},
        aliases=("none_explicit",),
        historical_use=(
            "Formalizes the pre-existing 'none' route, which stays available and "
            "continues to resolve to no selector object."
        ),
    ),
    MethodDescriptor(
        method_id="rfe_catboost",
        display_label="RFE (CatBoost)",
        implementation_id="rfe_catboost_fractional_step_v1",
        algorithm="recursive_feature_elimination_catboost_importance",
        supervised=True,
        selection_modes=("matched_budget",),
        provides_ranking=True,
        provides_natural_support=False,
        supports_fixed_budget=True,
        budget_kwarg="k",
        import_path="credit_risk_fs.selectors.heavy.rfe_catboost.CatBoostRFESelector",
        default_kwargs={
            "k": 40,
            "step_fraction": 0.20,
            "random_state": 42,
            "thread_count": 1,
        },
        cost_class="heavy",
        estimator_family="catboost.CatBoostClassifier",
        guarantees_exact_k=True,
        score_name="rank_derived_elimination_score",
        historical_use=(
            "New in Prompt 8. The historical 'rfe' route keeps sklearn RFE with an "
            "integer step of 10 and is unchanged."
        ),
        controlled_failure_conditions=(
            "k=None (no natural stopping point)",
            "k<=0",
            "CatBoost training failure",
            "importance length mismatch",
        ),
        notes=(
            "Fractional removal step (default 0.20) and explicit fit/elimination "
            "history, unlike the integer-step legacy path."
        ),
    ),
    MethodDescriptor(
        method_id="boruta_random_forest",
        display_label="Boruta (random forest)",
        implementation_id="boruta_random_forest_confirmed_tentative_v1",
        algorithm="all_relevant_shadow_feature_test_random_forest",
        supervised=True,
        selection_modes=(
            "natural_confirmed",
            "confirmed_top_k",
            "confirmed_then_tentative",
        ),
        provides_ranking=True,
        provides_natural_support=True,
        supports_fixed_budget=True,
        budget_kwarg="k",
        import_path=(
            "credit_risk_fs.selectors.heavy.boruta_rf.BorutaRandomForestSelector"
        ),
        default_kwargs={
            "k": None,
            "selection_mode": "natural_confirmed",
            "random_state": 42,
            "n_jobs": 1,
        },
        cost_class="heavy",
        estimator_family="sklearn.ensemble.RandomForestClassifier via boruta.BorutaPy",
        guarantees_exact_k=False,
        score_name="state_then_engine_rank_derived_score",
        historical_use=(
            "New in Prompt 8. The historical 'boruta' route -- including the frozen "
            "voting protocol's boruta voter -- keeps BorutaSelector unchanged."
        ),
        controlled_failure_conditions=(
            "k=None in a fixed-budget mode",
            "k<=0 in a fixed-budget mode",
            "Boruta engine failure",
            "support/ranking length mismatch",
        ),
        notes=(
            "Preserves confirmed/tentative/rejected. Natural support is confirmed "
            "only; the legacy selector discards the tentative state entirely."
        ),
    ),
    MethodDescriptor(
        method_id="catboost_shap",
        display_label="CatBoost-SHAP",
        implementation_id="catboost_native_shap_regular_mean_abs_train_sample_v1",
        algorithm="native_catboost_shapvalues_regular_mean_absolute",
        supervised=True,
        selection_modes=("matched_budget",),
        provides_ranking=True,
        provides_natural_support=False,
        supports_fixed_budget=True,
        budget_kwarg="k",
        import_path=(
            "credit_risk_fs.selectors.heavy.catboost_shap.CatBoostShapSelector"
        ),
        default_kwargs={
            "k": 40,
            "explanation_sample_size": 10_000,
            "random_state": 42,
            "thread_count": 1,
        },
        cost_class="heavy",
        estimator_family="catboost.CatBoostClassifier",
        guarantees_exact_k=True,
        score_name="mean_absolute_native_shap",
        historical_use=(
            "New in Prompt 8. The repository had no SHAP path; CatBoostModel."
            "get_feature_importance() returns PredictionValuesChange, not SHAP."
        ),
        controlled_failure_conditions=(
            "k=None (no defensible natural SHAP threshold)",
            "k<=0",
            "CatBoost training failure",
            "native SHAP failure",
            "unexpected SHAP array shape",
            "non-finite SHAP values",
        ),
        notes=(
            "Native EFstrType.ShapValues with shap_calc_type='Regular'; trailing "
            "expected-value column excluded; no fallback importance permitted."
        ),
    ),
    MethodDescriptor(
        method_id="legacy_rf_relevance_corr",
        display_label="Legacy RF relevance / correlation redundancy",
        implementation_id="rf_relevance_correlation_redundancy",
        algorithm="rf_impurity_relevance_over_mean_absolute_correlation",
        supervised=True,
        selection_modes=("matched_budget",),
        provides_ranking=True,
        provides_natural_support=False,
        supports_fixed_budget=True,
        budget_kwarg="k",
        import_path="credit_risk_fs.selectors.mrmr.RandomForestRelevanceMRMRSelector",
        default_kwargs={"k": 50, "method": "mrmr", "random_state": 42, "n_jobs": 1},
        aliases=("mrmr",),
        historical_use=(
            "Every historical artifact naming 'mrmr' or 'rf_corr_mrmr' used this "
            "algorithm, including the rf_corr_mrmr voter and the reference arm of "
            "all 16 cross_dataset_rank_voting_v1 runs."
        ),
        notes=(
            "Preserved unchanged for historical parity. It is NOT canonical "
            "mutual-information mRMR; the class already declares "
            "canonical_mrmr = False."
        ),
    ),
)

_BY_ID: dict[str, MethodDescriptor] = {
    descriptor.method_id: descriptor for descriptor in LIGHTWEIGHT_METHODS
}

_BY_ALIAS: dict[str, str] = {}
for _descriptor in LIGHTWEIGHT_METHODS:
    for _alias in _descriptor.aliases:
        if _alias in _BY_ALIAS:
            raise RuntimeError(f"alias {_alias!r} would resolve to two algorithms")
        _BY_ALIAS[_alias] = _descriptor.method_id


def lightweight_method_ids() -> tuple[str, ...]:
    """Canonical IDs in declaration order."""

    return tuple(_BY_ID)


def method_ids_by_cost_class(cost_class: str) -> tuple[str, ...]:
    """Canonical IDs whose declared cost class matches.

    The registry holds every contract-conformant method regardless of cost, so a
    caller that only wants the fast ones -- a quick fixture, a smoke test -- filters
    here rather than hard-coding a list that silently goes stale when a method is
    added.
    """

    if cost_class not in {"light", "heavy"}:
        raise ValueError(f"unknown cost class {cost_class!r}")
    return tuple(
        descriptor.method_id
        for descriptor in LIGHTWEIGHT_METHODS
        if descriptor.cost_class == cost_class
    )


def resolve_method_id(name: str) -> str:
    """Map a canonical ID or historical alias to its canonical ID."""

    key = str(name).strip().lower()
    if key in _BY_ID:
        return key
    if key in _BY_ALIAS:
        return _BY_ALIAS[key]
    raise KeyError(
        f"unknown selection method {name!r}; known: {sorted(_BY_ID)} "
        f"aliases: {sorted(_BY_ALIAS)}"
    )


def get_method_descriptor(name: str) -> MethodDescriptor:
    """Return the declared capabilities for a canonical ID or alias."""

    return _BY_ID[resolve_method_id(name)]


def registry_snapshot() -> dict[str, Any]:
    """Serializable snapshot used as audit evidence."""

    return {
        "contract_version": "lightweight_selector_contract_v1",
        "methods": [
            {
                "method_id": descriptor.method_id,
                "display_label": descriptor.display_label,
                "implementation_id": descriptor.implementation_id,
                "algorithm": descriptor.algorithm,
                "supervised": descriptor.supervised,
                "selection_modes": list(descriptor.selection_modes),
                "provides_ranking": descriptor.provides_ranking,
                "provides_natural_support": descriptor.provides_natural_support,
                "supports_fixed_budget": descriptor.supports_fixed_budget,
                "budget_kwarg": descriptor.budget_kwarg,
                "import_path": descriptor.import_path,
                "default_kwargs": dict(descriptor.default_kwargs),
                "aliases": list(descriptor.aliases),
                "historical_use": descriptor.historical_use,
                "notes": descriptor.notes,
                "cost_class": descriptor.cost_class,
                "estimator_family": descriptor.estimator_family,
                "guarantees_exact_k": descriptor.guarantees_exact_k,
                "score_name": descriptor.score_name,
                "serialization_version": descriptor.serialization_version,
                "allowed_in_frozen_voting": descriptor.allowed_in_frozen_voting,
                "controlled_failure_conditions": list(
                    descriptor.controlled_failure_conditions
                ),
            }
            for descriptor in LIGHTWEIGHT_METHODS
        ],
    }


def validate_method_selection_mode(name: str, selection_mode: str) -> None:
    """Reject an invalid method/mode pairing before any expensive work starts."""

    descriptor = get_method_descriptor(name)
    if selection_mode not in descriptor.selection_modes:
        raise ValueError(
            f"selection mode {selection_mode!r} is not supported by "
            f"{descriptor.method_id!r}; supported: {list(descriptor.selection_modes)}"
        )


__all__ = [
    "LIGHTWEIGHT_METHODS",
    "MethodDescriptor",
    "get_method_descriptor",
    "lightweight_method_ids",
    "method_ids_by_cost_class",
    "registry_snapshot",
    "resolve_method_id",
    "validate_method_selection_mode",
]
