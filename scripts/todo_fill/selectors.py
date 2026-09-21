"""Selector recipes that replay each paper method under its own frozen protocol.

Three protocols coexist in the repository and are reproduced here verbatim:

* ``baseline`` (full_baseline_v1 / third-dataset lock): contract selectors fitted
  on the ``OriginalFeatureNumericEncoder`` frame with the frozen settings.
* ``combination`` (selector_combinations_v1): ``IVThenBorutaSelector`` on the same
  encoded frame; the subset size is Boruta's natural confirmed support.
* ``legacy`` (original 8-arm matrix): selectors fitted *after* the dense
  ``Preprocessor`` unless the selector opts into raw selection, with the cached
  target-free LLM ranking replayed from ``artifacts/llm_cache`` instead of a new
  API call.  ``LLMThenStatSelector`` and ``StableCoreLLMFillSelector`` are driven
  through their own post-processing stages so padding and ordering match.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from scripts.todo_fill import llm_cache
from scripts.todo_fill.common import (
    BUDGETS,
    DESCRIPTION_PATH,
    FULL_DEV,
    HOMECREDIT,
    IV_THEN_BORUTA_POOL,
    LENDINGCLUB,
    LLM_POOL_BUDGET,
    LLM_SHARED_POOL_SIZE,
    REPO_ROOT,
    SEED,
    THIRD,
    FillError,
    ResourceSkip,
    Timer,
    available_ram_gb,
    frame_gib,
    log,
    require_ram,
)
from scripts.todo_fill.data import DataContext, dense_preprocess, encode_for_selection

FULL_BASELINE_CONFIG = REPO_ROOT / "configs/experiments/full_baseline_v1.yaml"
COMBINATION_CONFIG = REPO_ROOT / "configs/experiments/selector_combination_research_v1.yaml"
COMBINATION_LOCK = REPO_ROOT / "configs/protocols/selector_combinations_v1/combination_protocol_lock.json"

#: Approximate peak-memory multipliers over the encoded float32 training frame.
MEMORY_FACTOR: dict[str, float] = {
    "random_k": 1.2,
    "iv_woe": 2.5,
    "mrmr_mutual_information": 2.5,
    "lasso_l1_logistic": 3.0,
    "boruta_random_forest": 3.5,
    "rfe_catboost": 3.0,
    "catboost_shap": 3.0,
    "iv_then_boruta": 3.5,
    "mrmr_legacy": 3.5,
    "llm_then_mrmr": 2.0,
    "llm_then_boruta": 2.5,
    "stable_core_llm_fill": 3.5,
    "domain_rule_baseline": 1.5,
}


@dataclass
class SelectionOutput:
    features: list[str]
    ranked: list[str] | None
    fit_seconds: float
    protocol: str
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def n_selected(self) -> int:
        return len(self.features)


@dataclass
class FitContext:
    dataset: str
    backbone: str
    partition: str
    data: DataContext
    third_ranking: list[str] | None = None
    _raw: tuple[pd.DataFrame, pd.Series] | None = None
    _encoded: pd.DataFrame | None = None
    _dense: pd.DataFrame | None = None

    @property
    def k(self) -> int:
        return BUDGETS[self.backbone]

    def raw(self) -> tuple[pd.DataFrame, pd.Series]:
        if self._raw is None:
            self._raw = self.data.training_frame(self.dataset, self.partition)
        return self._raw

    def encoded(self) -> tuple[pd.DataFrame, pd.Series]:
        if self._encoded is None:
            X, y = self.raw()
            self._encoded = encode_for_selection(X)
        return self._encoded, self.raw()[1]

    def dense(self) -> tuple[pd.DataFrame, pd.Series]:
        if self._dense is None:
            X, y = self.raw()
            self._dense = dense_preprocess(X, self.dataset)
        return self._dense, self.raw()[1]

    def release(self) -> None:
        self._raw = None
        self._encoded = None
        self._dense = None


# ----------------------------------------------------------------- settings


def _yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def baseline_settings(dataset: str, method: str) -> dict[str, Any]:
    if dataset == THIRD:
        lock = json.loads((REPO_ROOT / "configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json").read_text(encoding="utf-8"))
        return dict(lock["approved_protocol"]["method_and_evaluation_matrix"]["selector_settings"].get(method, {}))
    return dict(_yaml(FULL_BASELINE_CONFIG)["selector_settings"].get(method, {}))


def combination_settings(dataset: str) -> dict[str, Any]:
    if dataset == THIRD:
        lock = json.loads((REPO_ROOT / "configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json").read_text(encoding="utf-8"))
        return dict(lock["approved_protocol"]["method_and_evaluation_matrix"]["combination_selector_settings"])
    return dict(_yaml(COMBINATION_CONFIG)["selector_settings"])


def combination_protocol_sha256() -> str:
    try:
        from credit_risk_fs.experiments.prompt_16_third_dataset import COMBINATION_PROTOCOL_SHA256

        return str(COMBINATION_PROTOCOL_SHA256)
    except Exception:
        from scripts.todo_fill.common import sha256_file

        return sha256_file(COMBINATION_LOCK)


def _filtered(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(kwargs)
    accepted = set(signature.parameters) - {"self"}
    return {key: value for key, value in kwargs.items() if key in accepted}


def _guard(ctx: FitContext, method: str, rows: int, columns: int) -> None:
    needed = frame_gib(rows, columns, 4) * MEMORY_FACTOR.get(method, 3.0) + 1.0
    require_ram(needed, f"{ctx.dataset}/{ctx.backbone}/{method}/{ctx.partition}")


def _result_output(selector: Any, seconds: float, protocol: str, **details: Any) -> SelectionOutput:
    from credit_risk_fs.selectors.base import get_selected_features

    features = [str(item) for item in (get_selected_features(selector) or [])]
    ranked = None
    result = getattr(selector, "result_", None) or getattr(selector, "result", None)
    if result is not None:
        try:
            if result.ranking:
                ranked = [str(item) for item in result.ranking]
            details.setdefault("budget_status", getattr(result, "budget_status", None))
            details.setdefault("selection_mode", getattr(result, "selection_mode", None))
            details.setdefault("warnings", list(getattr(result, "warnings", ()) or ()))
            details.setdefault("contract_fit_seconds", float(getattr(result, "fit_seconds", 0.0) or 0.0))
        except Exception:  # pragma: no cover - defensive against foreign result objects
            pass
    return SelectionOutput(features=features, ranked=ranked, fit_seconds=seconds, protocol=protocol, details=details)


# -------------------------------------------------------------- baseline line


def _fit_contract(ctx: FitContext, method: str, kwargs: dict[str, Any]) -> SelectionOutput:
    from credit_risk_fs.selectors.lightweight.registry import get_method_descriptor

    X, y = ctx.raw()
    _guard(ctx, method, len(X), X.shape[1])
    numeric, y = ctx.encoded()
    descriptor = get_method_descriptor(method)
    cls = descriptor.load()
    settings = dict(baseline_settings(ctx.dataset, method))
    settings.update(kwargs)
    settings.setdefault("random_state", SEED)
    if "fit_scope" in inspect.signature(cls.__init__).parameters:
        settings["fit_scope"] = "dev_fold_training_only"
    selector = cls(**_filtered(cls, settings))
    log(f"[fit] {ctx.dataset}/{ctx.backbone}/{method}/{ctx.partition}: {numeric.shape[0]} rows x {numeric.shape[1]} cols")
    timer = Timer()
    selector.fit(numeric, y)
    return _result_output(selector, timer.seconds(), "full_baseline_v1" if ctx.dataset != THIRD else "third_dataset_lock", settings={k: v for k, v in settings.items() if k != "fit_scope"})


def fit_random_k(ctx: FitContext) -> SelectionOutput:
    return _fit_contract(ctx, "random_k", {"k": ctx.k})


def fit_iv_woe(ctx: FitContext) -> SelectionOutput:
    return _fit_contract(ctx, "iv_woe", {"k": ctx.k})


def fit_mrmr_mutual_information(ctx: FitContext) -> SelectionOutput:
    return _fit_contract(ctx, "mrmr_mutual_information", {"k": ctx.k})


def fit_lasso(ctx: FitContext) -> SelectionOutput:
    return _fit_contract(ctx, "lasso_l1_logistic", {"k": ctx.k})


def fit_boruta_random_forest(ctx: FitContext) -> SelectionOutput:
    return _fit_contract(ctx, "boruta_random_forest", {"k": ctx.k})


def fit_rfe_catboost(ctx: FitContext) -> SelectionOutput:
    return _fit_contract(ctx, "rfe_catboost", {"k": ctx.k})


def fit_catboost_shap(ctx: FitContext) -> SelectionOutput:
    return _fit_contract(ctx, "catboost_shap", {"k": ctx.k})


def fit_iv_then_boruta(ctx: FitContext) -> SelectionOutput:
    from credit_risk_fs.selectors.combinations import IVThenBorutaSelector

    X, y = ctx.raw()
    _guard(ctx, "iv_then_boruta", len(X), X.shape[1])
    numeric, y = ctx.encoded()
    settings = combination_settings(ctx.dataset)
    pool = IV_THEN_BORUTA_POOL[ctx.dataset]
    selector = IVThenBorutaSelector(
        iv_pool_budget=pool,
        protocol_lock_sha256=combination_protocol_sha256(),
        iv_kwargs=dict(settings.get("iv_woe", {})),
        boruta_kwargs=dict(settings.get("boruta_random_forest", {})),
        random_state=SEED,
    )
    log(f"[fit] {ctx.dataset}/{ctx.backbone}/iv_then_boruta(pool={pool})/{ctx.partition}: {numeric.shape[0]} rows")
    timer = Timer()
    selector.fit(numeric, y)
    output = _result_output(selector, timer.seconds(), "selector_combinations_v1", iv_pool_budget=pool)
    result = getattr(selector, "result", None)
    if result is not None and getattr(result, "intermediate_features", None):
        output.details["intermediate_count"] = len(result.intermediate_features)
    return output


# ---------------------------------------------------------------- legacy line


def _legacy_ranking(ctx: FitContext, budget: int) -> llm_cache.CachedRanking | None:
    if ctx.dataset == THIRD:
        return None
    return llm_cache.ranking(ctx.dataset, ctx.partition, budget)


def _ranking_features(ctx: FitContext, budget: int) -> tuple[list[str], dict[str, Any]]:
    if ctx.dataset == THIRD:
        if not ctx.third_ranking:
            raise FillError("no target-free LLM ranking is available for the third dataset on this machine")
        return list(ctx.third_ranking), {"ranking_source": "regenerated_third_dataset_ranking"}
    cached = _legacy_ranking(ctx, budget)
    assert cached is not None
    return list(cached.features), {"ranking_source": cached.relative_path, "ranking_sha256": cached.sha256, "ranking_feature_budget": cached.feature_budget}


def fit_llm(ctx: FitContext) -> SelectionOutput:
    ranking, details = _ranking_features(ctx, ctx.k)
    if len(ranking) < ctx.k:
        details["warning"] = f"cached ranking holds only {len(ranking)} names"
    return SelectionOutput(features=ranking[: ctx.k], ranked=ranking[: ctx.k], fit_seconds=0.0, protocol="cached_llm_truncation", details=details)


def fit_pca(ctx: FitContext) -> SelectionOutput:
    features = [f"PC{i}" for i in range(1, ctx.k + 1)]
    return SelectionOutput(
        features=features,
        ranked=features,
        fit_seconds=0.0,
        protocol="legacy_pca_component_names",
        details={"note": "PCASelector publishes component names PC1..PCk; fold sets are identical by construction"},
    )


def fit_domain_rule_baseline(ctx: FitContext) -> SelectionOutput:
    from credit_risk_fs.selectors.domain_rule_baseline import DomainRuleBaselineSelector

    if ctx.dataset not in DESCRIPTION_PATH:
        raise FillError("the domain-rule baseline needs a description CSV; none is registered for this dataset")
    X, y = ctx.raw()
    _guard(ctx, "domain_rule_baseline", len(X), X.shape[1])
    selector = DomainRuleBaselineSelector(description_csv_path=str(DESCRIPTION_PATH[ctx.dataset]), feature_budget=ctx.k)
    log(f"[fit] {ctx.dataset}/{ctx.backbone}/domain_rule_baseline/{ctx.partition}")
    timer = Timer()
    selector.fit(X, y)
    return _result_output(selector, timer.seconds(), "legacy_matrix_raw_selection")


def fit_mrmr_legacy(ctx: FitContext) -> SelectionOutput:
    from credit_risk_fs.selectors.mrmr import RandomForestRelevanceMRMRSelector

    X, y = ctx.raw()
    _guard(ctx, "mrmr_legacy", len(X), int(X.shape[1] * 1.3))
    dense, y = ctx.dense()
    selector = RandomForestRelevanceMRMRSelector(k=ctx.k, method="mrmr", random_state=SEED, n_jobs=4)
    log(f"[fit] {ctx.dataset}/{ctx.backbone}/mrmr(legacy)/{ctx.partition}: {dense.shape[0]} rows x {dense.shape[1]} dense cols")
    timer = Timer()
    selector.fit(dense, y)
    return _result_output(selector, timer.seconds(), "legacy_matrix_after_dense_preprocessing", dense_columns=int(dense.shape[1]))


def _fit_llm_then_stat(ctx: FitContext, stat_name: str) -> SelectionOutput:
    from credit_risk_fs.selectors.boruta import BorutaSelector
    from credit_risk_fs.selectors.llm_then_stat import LLMThenStatSelector
    from credit_risk_fs.selectors.mrmr import RandomForestRelevanceMRMRSelector

    pool_budget = LLM_POOL_BUDGET[ctx.backbone]
    ranking, details = _ranking_features(ctx, pool_budget)
    X, y = ctx.raw()
    pool = [feature for feature in ranking[:pool_budget] if feature in X.columns]
    if not pool:
        raise FillError("the cached LLM pool has no overlap with the training candidates")
    _guard(ctx, f"llm_then_{stat_name}", len(X), max(len(pool) * 3, 1))
    X_pool = X.loc[:, pool]
    dense = dense_preprocess(X_pool, ctx.dataset)
    if stat_name == "mrmr":
        stat_cls, stat_kwargs = RandomForestRelevanceMRMRSelector, {"k": ctx.k, "method": "mrmr", "random_state": SEED, "n_jobs": 4}
    else:
        stat_cls, stat_kwargs = BorutaSelector, {"max_iter": 15, "random_state": SEED, "n_features": ctx.k, "n_jobs": 4}
    selector = LLMThenStatSelector(
        description_csv_path=str(DESCRIPTION_PATH.get(ctx.dataset, "")),
        stat_selector_cls=stat_cls,
        stat_selector_kwargs=stat_kwargs,
        llm_candidate_pool_budget=pool_budget,
        final_feature_budget=ctx.k,
    )
    selector.llm_selected_features_ = list(pool)
    log(f"[fit] {ctx.dataset}/{ctx.backbone}/llm_then_{stat_name}/{ctx.partition}: pool={len(pool)} dense cols={dense.shape[1]}")
    timer = Timer()
    selector.fit_postprocess(dense, y)
    from credit_risk_fs.selectors.base import get_selected_features

    stat_selected = [str(item) for item in (get_selected_features(selector.stat_selector) or [])]
    details.update({"llm_pool_size": len(pool), "llm_pool_budget": pool_budget, "statistical_stage_selected": len(stat_selected), "dense_columns": int(dense.shape[1])})
    if stat_name == "boruta":
        ranking_attr = getattr(selector.stat_selector, "selector", None)
        confirmed = getattr(ranking_attr, "support_", None)
        if confirmed is not None:
            details["boruta_confirmed_count"] = int(sum(bool(item) for item in confirmed))
    return SelectionOutput(
        features=[str(item) for item in selector.selected_features_ or []],
        ranked=None,
        fit_seconds=timer.seconds(),
        protocol="legacy_matrix_llm_pool_then_dense_statistical_stage",
        details=details,
    )


def fit_llm_then_mrmr(ctx: FitContext) -> SelectionOutput:
    return _fit_llm_then_stat(ctx, "mrmr")


def fit_llm_then_boruta(ctx: FitContext) -> SelectionOutput:
    return _fit_llm_then_stat(ctx, "boruta")


def fit_stable_core_llm_fill(ctx: FitContext) -> SelectionOutput:
    from credit_risk_fs.selectors.stable_core_llm_fill import StableCoreLLMFillSelector

    ranking, details = _ranking_features(ctx, LLM_SHARED_POOL_SIZE)
    X, y = ctx.raw()
    _guard(ctx, "stable_core_llm_fill", len(X), int(X.shape[1] * 1.3))
    dense, y = ctx.dense()
    selector = StableCoreLLMFillSelector(
        description_csv_path=str(DESCRIPTION_PATH.get(ctx.dataset, "")),
        llm_shared_pool_size=LLM_SHARED_POOL_SIZE,
        final_feature_budget=ctx.k,
        random_state=SEED,
        component_n_jobs=4,
    )
    selector.llm_selected_features_ = list(ranking)
    log(f"[fit] {ctx.dataset}/{ctx.backbone}/stable_core_llm_fill/{ctx.partition}: {dense.shape[0]} rows x {dense.shape[1]} dense cols")
    timer = Timer()
    core, frequency = selector._bootstrap_core(dense, y)
    selector.stable_core_features_ = core
    selector.stable_core_frequency_ = frequency
    selector.selected_features_ = selector._finalize_features(dense)
    details.update({"stable_core_count": len(core), "dense_columns": int(dense.shape[1])})
    return SelectionOutput(
        features=[str(item) for item in selector.selected_features_ or []],
        ranked=None,
        fit_seconds=timer.seconds(),
        protocol="legacy_matrix_bootstrap_core_then_llm_fill",
        details=details,
    )


RECIPES: dict[str, Callable[[FitContext], SelectionOutput]] = {
    "random_k": fit_random_k,
    "iv_woe": fit_iv_woe,
    "mrmr_mutual_information": fit_mrmr_mutual_information,
    "lasso_l1_logistic": fit_lasso,
    "boruta_random_forest": fit_boruta_random_forest,
    "rfe_catboost": fit_rfe_catboost,
    "catboost_shap": fit_catboost_shap,
    "iv_then_boruta": fit_iv_then_boruta,
    "llm": fit_llm,
    "pca": fit_pca,
    "domain_rule_baseline": fit_domain_rule_baseline,
    "mrmr_legacy": fit_mrmr_legacy,
    "llm_then_mrmr": fit_llm_then_mrmr,
    "llm_then_boruta": fit_llm_then_boruta,
    "stable_core_llm_fill": fit_stable_core_llm_fill,
}

#: Methods that never touch the data.
DATA_FREE_METHODS = {"llm", "pca"}

#: Rough cost tiers used to order work and to let ``--tier`` cap the run.
COST_TIER: dict[str, int] = {
    "llm": 0,
    "pca": 0,
    "random_k": 1,
    "domain_rule_baseline": 1,
    "iv_woe": 1,
    "llm_then_mrmr": 1,
    "llm_then_boruta": 2,
    "mrmr_mutual_information": 2,
    "lasso_l1_logistic": 2,
    "catboost_shap": 2,
    "mrmr_legacy": 2,
    "stable_core_llm_fill": 2,
    "rfe_catboost": 3,
    "boruta_random_forest": 3,
    "iv_then_boruta": 3,
}


def supported(method: str, dataset: str) -> tuple[bool, str]:
    if method not in RECIPES:
        return False, f"no executable recipe for {method}"
    if dataset == THIRD and method in {"domain_rule_baseline", "pca", "mrmr_legacy", "llm_then_mrmr", "llm_then_boruta"}:
        return False, "never part of the frozen third-dataset matrix and no executable protocol exists for it there"
    return True, ""
