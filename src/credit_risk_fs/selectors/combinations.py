"""The four preregistered Prompt 11 selector combinations.

The public surface is intentionally closed: three explicit chains and one
explicit five-voter rank aggregator.  There is no generic chain language and no
outcome-aware tuning route.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from credit_risk_fs.selectors.base import (
    SelectedFeaturesMixin,
    select_feature_frame,
    validate_feature_frame,
)
from credit_risk_fs.selectors.heavy.boruta_rf import BorutaRandomForestSelector
from credit_risk_fs.selectors.heavy.catboost_shap import CatBoostShapSelector
from credit_risk_fs.selectors.heavy.rfe_catboost import CatBoostRFESelector
from credit_risk_fs.selectors.lightweight.contract import (
    SelectionResult,
    ordered_name_hash,
    training_identity_hash,
)
from credit_risk_fs.selectors.lightweight.iv import InformationValueSelector
from credit_risk_fs.selectors.lightweight.lasso import L1LogisticSelector
from credit_risk_fs.selectors.lightweight.mi_mrmr import MutualInformationMRMRSelector


COMBINATION_SCHEMA_VERSION = "selector_combination_result_v1"
PROTOCOL_ID = "selector_combinations_v1"
IV_POOL_BUDGETS = (100, 200, 300)
FINAL_BUDGETS = frozenset({20, 40})
STATISTICAL_VOTERS = (
    "iv_woe",
    "lasso_l1_logistic",
    "rfe_catboost",
    "boruta_random_forest",
    "catboost_shap",
)
STATISTICAL_WEIGHTS = {method: 0.2 for method in STATISTICAL_VOTERS}
COMPONENT_IMPLEMENTATIONS = {
    "iv_woe": "iv_woe_quantile_binned_v1",
    "lasso_l1_logistic": "lasso_l1_logistic_v1",
    "rfe_catboost": "rfe_catboost_fractional_step_v1",
    "boruta_random_forest": "boruta_random_forest_confirmed_tentative_v1",
    "catboost_shap": "catboost_native_shap_regular_mean_abs_train_sample_v1",
    "mrmr_mutual_information": "mrmr_mutual_information_discrete_plugin_v1",
}


class CombinationContractError(ValueError):
    """Raised before an invalid combination result can be published."""


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class CombinationResult:
    method_id: str
    implementation_id: str
    component_ids: tuple[str, ...]
    component_implementation_ids: tuple[str, ...]
    fit_scope: str
    seed: int
    protocol_id: str
    protocol_lock_sha256: str
    candidate_universe: tuple[str, ...]
    training_identity_sha256: str
    requested_budget: int | None
    selected_features: tuple[str, ...]
    feasibility_state: str
    intermediate_features: tuple[str, ...] | None
    stage_provenance: tuple[Mapping[str, Any], ...]
    configuration: Mapping[str, Any]
    fit_seconds: float
    warnings: tuple[str, ...] = ()
    terminal_state: str = "completed"
    voting_evidence: tuple[Mapping[str, Any], ...] | None = None
    schema_version: str = COMBINATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        universe = list(self.candidate_universe)
        if len(universe) != len(set(universe)):
            raise CombinationContractError("candidate universe contains duplicate names")
        selected = list(self.selected_features)
        if len(selected) != len(set(selected)) or not set(selected).issubset(universe):
            raise CombinationContractError("final subset is not a unique candidate subset")
        if self.intermediate_features is not None:
            intermediate = list(self.intermediate_features)
            if len(intermediate) != len(set(intermediate)) or not set(intermediate).issubset(
                universe
            ):
                raise CombinationContractError("intermediate pool is not a unique candidate subset")
            if not set(selected).issubset(intermediate):
                raise CombinationContractError("Stage 2 recovered features excluded by Stage 1")
        if self.feasibility_state not in {
            "completed",
            "infeasible_natural_support",
            "no_refinement_possible",
        }:
            raise CombinationContractError(
                f"unknown feasibility state: {self.feasibility_state}"
            )
        if self.feasibility_state == "completed" and self.requested_budget is not None:
            if len(selected) != self.requested_budget:
                raise CombinationContractError("completed exact-budget result has the wrong size")
        if self.feasibility_state == "infeasible_natural_support":
            if self.requested_budget is None or len(selected) >= self.requested_budget:
                raise CombinationContractError("invalid natural-support infeasibility state")
        if self.feasibility_state == "no_refinement_possible":
            if self.requested_budget is None or len(selected) != self.requested_budget:
                raise CombinationContractError("invalid no-refinement state")
        if tuple(STATISTICAL_WEIGHTS) == self.component_ids:
            if self.voting_evidence is None or len(self.voting_evidence) != len(universe):
                raise CombinationContractError("statistical voter requires one evidence row per candidate")

    @property
    def candidate_universe_sha256(self) -> str:
        return ordered_name_hash(self.candidate_universe)

    @property
    def intermediate_features_sha256(self) -> str | None:
        return (
            None
            if self.intermediate_features is None
            else ordered_name_hash(self.intermediate_features)
        )

    @property
    def selected_features_sha256(self) -> str:
        return ordered_name_hash(self.selected_features)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "implementation_id": self.implementation_id,
            "component_ids": list(self.component_ids),
            "component_implementation_ids": list(self.component_implementation_ids),
            "fit_scope": self.fit_scope,
            "seed": self.seed,
            "protocol_id": self.protocol_id,
            "protocol_lock_sha256": self.protocol_lock_sha256,
            "candidate_universe": list(self.candidate_universe),
            "candidate_universe_count": len(self.candidate_universe),
            "candidate_universe_sha256": self.candidate_universe_sha256,
            "training_identity_sha256": self.training_identity_sha256,
            "requested_budget": self.requested_budget,
            "realized_budget": len(self.selected_features),
            "feasibility_state": self.feasibility_state,
            "intermediate_features": (
                None if self.intermediate_features is None else list(self.intermediate_features)
            ),
            "intermediate_feature_count": (
                None if self.intermediate_features is None else len(self.intermediate_features)
            ),
            "intermediate_features_sha256": self.intermediate_features_sha256,
            "selected_features": list(self.selected_features),
            "selected_features_sha256": self.selected_features_sha256,
            "stage_provenance": [dict(item) for item in self.stage_provenance],
            "configuration": dict(self.configuration),
            "fit_seconds": self.fit_seconds,
            "warnings": list(self.warnings),
            "terminal_state": self.terminal_state,
            "voting_evidence": (
                None
                if self.voting_evidence is None
                else [dict(item) for item in self.voting_evidence]
            ),
        }


def _require_target(X: pd.DataFrame, y: pd.Series | Sequence[int] | None) -> pd.Series:
    validate_feature_frame(X)
    if y is None:
        raise CombinationContractError("combination selectors require a binary target")
    target = pd.Series(np.asarray(y)).reset_index(drop=True)
    if len(target) != len(X) or target.isna().any() or not target.isin([0, 1]).all():
        raise CombinationContractError("target must align with X and be binary 0/1")
    if target.nunique() != 2:
        raise CombinationContractError("target must contain both classes")
    return target.astype(int)


def _stage_provenance(result: SelectionResult, *, reuse: str = "new_fit") -> dict[str, Any]:
    return {
        "method_id": result.method_id,
        "implementation_id": result.implementation_id,
        "candidate_universe_sha256": result.candidate_universe_sha256,
        "training_identity_sha256": result.training_identity_sha256,
        "requested_budget": result.requested_budget,
        "realized_budget": result.actual_selected_count,
        "budget_status": result.budget_status,
        "fit_seconds": result.fit_seconds,
        "fit_provenance": reuse,
        "estimator_config_sha256": result.estimator_config_sha256,
        "heavy_metadata": None if result.heavy_metadata is None else dict(result.heavy_metadata),
    }


class _CombinationSelector(SelectedFeaturesMixin):
    method_id: ClassVar[str]
    implementation_id: ClassVar[str]
    component_ids: ClassVar[tuple[str, ...]]

    def __init__(
        self,
        *,
        random_state: int = 42,
        protocol_lock_sha256: str,
        fit_scope: str = "dev_fold_training_only",
    ) -> None:
        self.random_state = int(random_state)
        self.protocol_lock_sha256 = str(protocol_lock_sha256)
        if len(self.protocol_lock_sha256) != 64:
            raise ValueError("protocol_lock_sha256 must be a SHA-256 hex digest")
        self.fit_scope = str(fit_scope)
        self.selected_features_: list[str] | None = None
        self.result_: CombinationResult | None = None
        self.stage_results_: tuple[SelectionResult, ...] = ()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return select_feature_frame(
            X, self.selected_features_, selector_name=self.__class__.__name__
        )

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def result(self) -> CombinationResult:
        if self.result_ is None:
            raise ValueError(f"{self.__class__.__name__} must be fitted first")
        return self.result_


class IVThenBorutaSelector(_CombinationSelector):
    method_id = "iv_then_boruta"
    implementation_id = "iv_quantile_pool_then_boruta_rf_confirmed_only_v1"
    component_ids = ("iv_woe", "boruta_random_forest")

    def __init__(
        self,
        *,
        iv_pool_budget: int,
        protocol_lock_sha256: str,
        iv_kwargs: Mapping[str, Any] | None = None,
        boruta_kwargs: Mapping[str, Any] | None = None,
        random_state: int = 42,
        fit_scope: str = "dev_fold_training_only",
        iv_factory: Callable[..., Any] = InformationValueSelector,
        boruta_factory: Callable[..., Any] = BorutaRandomForestSelector,
    ) -> None:
        super().__init__(
            random_state=random_state,
            protocol_lock_sha256=protocol_lock_sha256,
            fit_scope=fit_scope,
        )
        if int(iv_pool_budget) not in IV_POOL_BUDGETS:
            raise ValueError(f"iv_pool_budget must be one of {IV_POOL_BUDGETS}")
        self.iv_pool_budget = int(iv_pool_budget)
        self.iv_kwargs = dict(iv_kwargs or {})
        self.boruta_kwargs = dict(boruta_kwargs or {})
        self.iv_factory = iv_factory
        self.boruta_factory = boruta_factory

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> IVThenBorutaSelector:
        started = time.perf_counter()
        target = _require_target(X, y)
        universe = tuple(validate_feature_frame(X))
        iv = self.iv_factory(
            k=self.iv_pool_budget,
            random_state=self.random_state,
            fit_scope=self.fit_scope,
            **self.iv_kwargs,
        ).fit(X, target)
        intermediate = tuple(iv.selected_features_)
        boruta = self.boruta_factory(
            k=None,
            selection_mode="natural_confirmed",
            random_state=self.random_state,
            fit_scope=self.fit_scope,
            **self.boruta_kwargs,
        ).fit(X.loc[:, list(intermediate)], target)
        final = tuple(boruta.selected_features_)
        self.stage_results_ = (iv.result, boruta.result)
        self.selected_features_ = list(final)
        self.result_ = CombinationResult(
            method_id=self.method_id,
            implementation_id=self.implementation_id,
            component_ids=self.component_ids,
            component_implementation_ids=tuple(
                COMPONENT_IMPLEMENTATIONS[item] for item in self.component_ids
            ),
            fit_scope=self.fit_scope,
            seed=self.random_state,
            protocol_id=PROTOCOL_ID,
            protocol_lock_sha256=self.protocol_lock_sha256,
            candidate_universe=universe,
            training_identity_sha256=training_identity_hash(X, target),
            requested_budget=None,
            selected_features=final,
            feasibility_state="completed",
            intermediate_features=intermediate,
            stage_provenance=tuple(_stage_provenance(item) for item in self.stage_results_),
            configuration={
                "iv_pool_budget": self.iv_pool_budget,
                "iv_kwargs": self.iv_kwargs,
                "boruta_kwargs": self.boruta_kwargs,
                "boruta_support": "confirmed_only_natural_support",
                "padding": "forbidden",
            },
            fit_seconds=time.perf_counter() - started,
        )
        return self


class _BorutaThenFixedRefiner(_CombinationSelector):
    refiner_id: ClassVar[str]
    refiner_implementation_id: ClassVar[str]
    refiner_factory_default: ClassVar[Callable[..., Any]]

    def __init__(
        self,
        *,
        k: int,
        protocol_lock_sha256: str,
        boruta_kwargs: Mapping[str, Any] | None = None,
        refiner_kwargs: Mapping[str, Any] | None = None,
        random_state: int = 42,
        fit_scope: str = "dev_fold_training_only",
        boruta_factory: Callable[..., Any] = BorutaRandomForestSelector,
        refiner_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            random_state=random_state,
            protocol_lock_sha256=protocol_lock_sha256,
            fit_scope=fit_scope,
        )
        if int(k) not in FINAL_BUDGETS:
            raise ValueError("final k must preserve the frozen LR/CatBoost budgets 20 or 40")
        self.k = int(k)
        self.boruta_kwargs = dict(boruta_kwargs or {})
        self.refiner_kwargs = dict(refiner_kwargs or {})
        self.boruta_factory = boruta_factory
        self.refiner_factory = refiner_factory or self.refiner_factory_default

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> _BorutaThenFixedRefiner:
        started = time.perf_counter()
        target = _require_target(X, y)
        universe = tuple(validate_feature_frame(X))
        boruta = self.boruta_factory(
            k=None,
            selection_mode="natural_confirmed",
            random_state=self.random_state,
            fit_scope=self.fit_scope,
            **self.boruta_kwargs,
        ).fit(X, target)
        intermediate = tuple(boruta.selected_features_)
        warnings: list[str] = []
        if len(intermediate) < self.k:
            final = intermediate
            state = "infeasible_natural_support"
            stage_results = (boruta.result,)
            warnings.append(
                f"Boruta confirmed {len(intermediate)} features, below requested {self.k}; no padding or refiner fit"
            )
        elif len(intermediate) == self.k:
            final = intermediate
            state = "no_refinement_possible"
            stage_results = (boruta.result,)
            warnings.append("Boruta support equals the final budget; Stage 2 was not fitted")
        else:
            refiner = self.refiner_factory(
                k=self.k,
                random_state=self.random_state,
                fit_scope=self.fit_scope,
                **self.refiner_kwargs,
            ).fit(X.loc[:, list(intermediate)], target)
            final = tuple(refiner.selected_features_)
            state = "completed"
            stage_results = (boruta.result, refiner.result)
        self.stage_results_ = stage_results
        self.selected_features_ = list(final)
        provenance = [_stage_provenance(boruta.result)]
        if len(stage_results) == 1:
            provenance.append(
                {
                    "method_id": self.refiner_id,
                    "implementation_id": self.refiner_implementation_id,
                    "fit_provenance": "not_fitted",
                    "reason": state,
                    "candidate_universe_sha256": ordered_name_hash(intermediate),
                    "training_identity_sha256": training_identity_hash(
                        X.loc[:, list(intermediate)], target
                    ),
                }
            )
        else:
            provenance.append(_stage_provenance(stage_results[1]))
        self.result_ = CombinationResult(
            method_id=self.method_id,
            implementation_id=self.implementation_id,
            component_ids=self.component_ids,
            component_implementation_ids=tuple(
                COMPONENT_IMPLEMENTATIONS[item] for item in self.component_ids
            ),
            fit_scope=self.fit_scope,
            seed=self.random_state,
            protocol_id=PROTOCOL_ID,
            protocol_lock_sha256=self.protocol_lock_sha256,
            candidate_universe=universe,
            training_identity_sha256=training_identity_hash(X, target),
            requested_budget=self.k,
            selected_features=final,
            feasibility_state=state,
            intermediate_features=intermediate,
            stage_provenance=tuple(provenance),
            configuration={
                "k": self.k,
                "boruta_kwargs": self.boruta_kwargs,
                "refiner_kwargs": self.refiner_kwargs,
                "boruta_support": "confirmed_only",
                "tentative_or_rejected_padding": False,
            },
            fit_seconds=time.perf_counter() - started,
            warnings=tuple(warnings),
        )
        return self


class BorutaThenCatBoostRFESelector(_BorutaThenFixedRefiner):
    method_id = "boruta_then_rfe_catboost"
    implementation_id = "boruta_rf_confirmed_then_catboost_rfe_v1"
    component_ids = ("boruta_random_forest", "rfe_catboost")
    refiner_id = "rfe_catboost"
    refiner_implementation_id = COMPONENT_IMPLEMENTATIONS[refiner_id]
    refiner_factory_default = CatBoostRFESelector


class BorutaThenMutualInformationMRMRSelector(_BorutaThenFixedRefiner):
    method_id = "boruta_then_mrmr_mutual_information"
    implementation_id = "boruta_rf_confirmed_then_mi_mrmr_v1"
    component_ids = ("boruta_random_forest", "mrmr_mutual_information")
    refiner_id = "mrmr_mutual_information"
    refiner_implementation_id = COMPONENT_IMPLEMENTATIONS[refiner_id]
    refiner_factory_default = MutualInformationMRMRSelector

    def __init__(self, **kwargs: Any) -> None:
        refiner_kwargs = dict(kwargs.pop("refiner_kwargs", {}) or {})
        frozen = {"n_bins": 10, "objective": "mid"}
        for key, expected in frozen.items():
            if key in refiner_kwargs and refiner_kwargs[key] != expected:
                raise ValueError(f"canonical MI-mRMR requires {key}={expected!r}")
            refiner_kwargs[key] = expected
        super().__init__(refiner_kwargs=refiner_kwargs, **kwargs)


def _result_quality(result: SelectionResult, candidate_order: Sequence[str]) -> tuple[np.ndarray, str, list[str]]:
    if tuple(result.candidate_universe) != tuple(candidate_order):
        raise CombinationContractError(
            f"{result.method_id} candidate-universe order/hash does not match the voting cell"
        )
    p = len(candidate_order)
    warnings: list[str] = []
    if p == 1:
        return np.ones(1, dtype=float), "single_candidate_quality_one", warnings
    if result.method_id == "boruta_random_forest":
        metadata = dict(result.heavy_metadata or {})
        states = metadata.get("support_states")
        engine = metadata.get("engine_ranking")
        if not isinstance(states, Mapping) or set(states) != set(candidate_order):
            raise CombinationContractError("Boruta voting evidence lacks authentic support states")
        block = {"confirmed": 0, "tentative": 1, "rejected": 2}
        if not set(states.values()).issubset(block):
            raise CombinationContractError("Boruta voting evidence has an unknown support state")
        if isinstance(engine, Mapping) and set(engine) == set(candidate_order):
            ordered_key = np.asarray(
                [block[str(states[name])] * (p + 1) + int(engine[name]) for name in candidate_order],
                dtype=float,
            )
            ranks = rankdata(ordered_key, method="average")
            adapter = "boruta_state_then_authentic_engine_rank_tied_midrank"
        else:
            ordered_key = np.asarray([block[str(states[name])] for name in candidate_order])
            ranks = rankdata(ordered_key, method="average")
            adapter = "boruta_state_block_tied_midrank"
            warnings.append("Boruta had no authentic within-state ranking; state-block midranks used")
    elif result.method_id == "lasso_l1_logistic":
        scores = dict(result.raw_scores or {})
        if set(scores) != set(candidate_order):
            raise CombinationContractError("LASSO voting evidence lacks full coefficients")
        magnitude = np.asarray([abs(float(scores[name])) for name in candidate_order])
        ranks = rankdata(-magnitude, method="average")
        adapter = "lasso_absolute_coefficient_tied_midrank_including_zero_block"
    elif result.method_id in {"iv_woe", "catboost_shap"}:
        scores = dict(result.raw_scores or {})
        if set(scores) != set(candidate_order):
            raise CombinationContractError(f"{result.method_id} voting evidence lacks full scores")
        values = np.asarray([float(scores[name]) for name in candidate_order])
        ranks = rankdata(-values, method="average")
        adapter = "authentic_higher_score_tied_midrank"
    elif result.method_id == "rfe_catboost":
        ranking = list(result.ranking or ())
        if len(ranking) != p or set(ranking) != set(candidate_order):
            raise CombinationContractError("RFE voting evidence lacks a complete authentic ranking")
        position = {name: index + 1 for index, name in enumerate(ranking)}
        ranks = np.asarray([position[name] for name in candidate_order], dtype=float)
        adapter = "authentic_complete_elimination_rank"
    else:
        raise CombinationContractError(f"unregistered statistical voter: {result.method_id}")
    return 1.0 - (ranks - 1.0) / (p - 1.0), adapter, warnings


class StatisticalNormalizedAverageRankSelector(_CombinationSelector):
    method_id = "statistical_normalized_average_rank"
    implementation_id = "equal_weight_five_selector_normalized_average_rank_v1"
    component_ids = STATISTICAL_VOTERS

    def __init__(
        self,
        *,
        k: int,
        protocol_lock_sha256: str,
        component_kwargs: Mapping[str, Mapping[str, Any]] | None = None,
        component_results: Mapping[str, SelectionResult] | None = None,
        component_factories: Mapping[str, Callable[..., Any]] | None = None,
        random_state: int = 42,
        fit_scope: str = "dev_fold_training_only",
    ) -> None:
        super().__init__(
            random_state=random_state,
            protocol_lock_sha256=protocol_lock_sha256,
            fit_scope=fit_scope,
        )
        if int(k) not in FINAL_BUDGETS:
            raise ValueError("statistical voter k must be the frozen LR/CatBoost budget 20 or 40")
        self.k = int(k)
        self.component_kwargs = {
            str(key): dict(value) for key, value in (component_kwargs or {}).items()
        }
        self.component_results = None if component_results is None else dict(component_results)
        defaults: dict[str, Callable[..., Any]] = {
            "iv_woe": InformationValueSelector,
            "lasso_l1_logistic": L1LogisticSelector,
            "rfe_catboost": CatBoostRFESelector,
            "boruta_random_forest": BorutaRandomForestSelector,
            "catboost_shap": CatBoostShapSelector,
        }
        defaults.update(component_factories or {})
        self.component_factories = defaults
        self.voting_frame_: pd.DataFrame | None = None

    def _fit_components(self, X: pd.DataFrame, target: pd.Series) -> tuple[dict[str, SelectionResult], str]:
        if self.component_results is not None:
            if tuple(self.component_results) != STATISTICAL_VOTERS:
                raise CombinationContractError("component reuse must provide the exact ordered five-voter set")
            return dict(self.component_results), "authenticated_reuse"
        results: dict[str, SelectionResult] = {}
        for method in STATISTICAL_VOTERS:
            kwargs = dict(self.component_kwargs.get(method, {}))
            if method == "boruta_random_forest":
                kwargs.update({"k": None, "selection_mode": "natural_confirmed"})
            else:
                kwargs["k"] = self.k
            selector = self.component_factories[method](
                random_state=self.random_state,
                fit_scope=self.fit_scope,
                **kwargs,
            ).fit(X, target)
            results[method] = selector.result
        return results, "new_fit"

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> StatisticalNormalizedAverageRankSelector:
        started = time.perf_counter()
        target = _require_target(X, y)
        universe = tuple(validate_feature_frame(X))
        if self.k > len(universe):
            raise CombinationContractError("final voter budget exceeds the candidate universe")
        expected_training = training_identity_hash(X, target)
        components, provenance = self._fit_components(X, target)
        if tuple(components) != STATISTICAL_VOTERS:
            raise CombinationContractError("statistical voter membership or order changed")
        for method, result in components.items():
            if result.method_id != method or result.implementation_id != COMPONENT_IMPLEMENTATIONS[method]:
                raise CombinationContractError(f"component identity mismatch for {method}")
            if result.training_identity_sha256 != expected_training:
                raise CombinationContractError(f"component training identity mismatch for {method}")
        qualities: dict[str, np.ndarray] = {}
        adapters: dict[str, str] = {}
        warnings: list[str] = []
        for method, result in components.items():
            quality, adapter, notices = _result_quality(result, universe)
            qualities[method] = quality
            adapters[method] = adapter
            warnings.extend(notices)
        aggregate = sum(STATISTICAL_WEIGHTS[name] * qualities[name] for name in STATISTICAL_VOTERS)
        # Stable mergesort preserves canonical candidate order inside an exact tie.
        order = np.argsort(-aggregate, kind="mergesort")
        selected = tuple(universe[index] for index in order[: self.k])
        rank_lookup = {int(index): position + 1 for position, index in enumerate(order)}
        rows: list[dict[str, Any]] = []
        for index, feature in enumerate(universe):
            row: dict[str, Any] = {
                "feature": feature,
                "canonical_candidate_position": index + 1,
                "aggregate_quality": float(aggregate[index]),
                "aggregate_rank": rank_lookup[index],
                "selected": feature in set(selected),
                "aggregate_tie_break": "canonical_candidate_universe_order",
                "missing_state": "complete",
            }
            for method in STATISTICAL_VOTERS:
                row[f"{method}__normalized_quality"] = float(qualities[method][index])
                row[f"{method}__weight"] = STATISTICAL_WEIGHTS[method]
                row[f"{method}__adapter"] = adapters[method]
                row[f"{method}__missing_state"] = "complete_or_legitimately_ranked"
            rows.append(row)
        self.voting_frame_ = pd.DataFrame(rows).sort_values(
            ["aggregate_rank"], kind="mergesort"
        )
        self.stage_results_ = tuple(components[name] for name in STATISTICAL_VOTERS)
        self.selected_features_ = list(selected)
        self.result_ = CombinationResult(
            method_id=self.method_id,
            implementation_id=self.implementation_id,
            component_ids=STATISTICAL_VOTERS,
            component_implementation_ids=tuple(
                COMPONENT_IMPLEMENTATIONS[item] for item in STATISTICAL_VOTERS
            ),
            fit_scope=self.fit_scope,
            seed=self.random_state,
            protocol_id=PROTOCOL_ID,
            protocol_lock_sha256=self.protocol_lock_sha256,
            candidate_universe=universe,
            training_identity_sha256=expected_training,
            requested_budget=self.k,
            selected_features=selected,
            feasibility_state="completed",
            intermediate_features=None,
            stage_provenance=tuple(
                _stage_provenance(components[item], reuse=provenance)
                for item in STATISTICAL_VOTERS
            ),
            configuration={
                "k": self.k,
                "voters": list(STATISTICAL_VOTERS),
                "weights": dict(STATISTICAL_WEIGHTS),
                "normalization": "q=1-(r-1)/(p-1); p=1 => q=1",
                "rank_ties": "midrank",
                "aggregate": "arithmetic_mean",
                "aggregate_tie_break": "canonical_candidate_universe_order",
                "missing_component_evidence": "invalidates_cell",
                "outcome_weighting": False,
                "component_kwargs": self.component_kwargs,
            },
            fit_seconds=time.perf_counter() - started,
            warnings=tuple(warnings),
            voting_evidence=tuple(rows),
        )
        return self


COMBINATION_CLASSES = {
    "iv_then_boruta": IVThenBorutaSelector,
    "boruta_then_rfe_catboost": BorutaThenCatBoostRFESelector,
    "boruta_then_mrmr_mutual_information": BorutaThenMutualInformationMRMRSelector,
    "statistical_normalized_average_rank": StatisticalNormalizedAverageRankSelector,
}


def save_combination_result(output_dir: str | Path, selector: _CombinationSelector) -> dict[str, Any]:
    """Publish distinct authenticated intermediate/final/component artifacts."""

    from credit_risk_fs.experiments.atomic_io import write_csv_atomic, write_json_atomic

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    result = selector.result
    artifacts: dict[str, Any] = {}
    if result.intermediate_features is not None:
        intermediate = pd.DataFrame(
            {
                "rank": range(1, len(result.intermediate_features) + 1),
                "feature": result.intermediate_features,
            }
        )
        artifacts["intermediate_features.csv"] = write_csv_atomic(
            root / "intermediate_features.csv", intermediate
        ).to_dict()
    final = pd.DataFrame(
        {"rank": range(1, len(result.selected_features) + 1), "feature": result.selected_features}
    )
    artifacts["final_selected_features.csv"] = write_csv_atomic(
        root / "final_selected_features.csv", final
    ).to_dict()
    if (
        isinstance(selector, StatisticalNormalizedAverageRankSelector)
        and selector.voting_frame_ is not None
    ):
        artifacts["voting_evidence.csv"] = write_csv_atomic(
            root / "voting_evidence.csv", selector.voting_frame_
        ).to_dict()
    artifacts["combination_result.json"] = write_json_atomic(
        root / "combination_result.json", result.to_dict()
    ).to_dict()
    manifest = {
        "schema_version": "selector_combination_artifact_manifest_v1",
        "method_id": result.method_id,
        "implementation_id": result.implementation_id,
        "protocol_lock_sha256": result.protocol_lock_sha256,
        "artifacts": artifacts,
    }
    write_json_atomic(root / "artifact_manifest.json", manifest)
    return manifest


__all__ = [
    "COMBINATION_CLASSES",
    "COMBINATION_SCHEMA_VERSION",
    "COMPONENT_IMPLEMENTATIONS",
    "CombinationContractError",
    "CombinationResult",
    "IVThenBorutaSelector",
    "BorutaThenCatBoostRFESelector",
    "BorutaThenMutualInformationMRMRSelector",
    "StatisticalNormalizedAverageRankSelector",
    "STATISTICAL_VOTERS",
    "STATISTICAL_WEIGHTS",
    "save_combination_result",
]
