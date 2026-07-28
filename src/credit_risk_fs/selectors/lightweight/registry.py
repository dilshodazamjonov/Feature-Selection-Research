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

    def __post_init__(self) -> None:
        unknown = set(self.selection_modes) - SELECTION_MODES
        if unknown:
            raise ValueError(
                f"{self.method_id} declares unknown selection mode(s): {sorted(unknown)}"
            )

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
    "registry_snapshot",
    "resolve_method_id",
    "validate_method_selection_mode",
]
