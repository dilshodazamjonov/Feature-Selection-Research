from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from feature_selection.mrmr import MRMR
from utils.feature_metadata import build_feature_metadata


def _selector_features(selector: Any) -> list[str]:
    for attr in ["selected_features_", "selected_features"]:
        value = getattr(selector, attr, None)
        if value is not None:
            return list(value)
    return []


def _set_selector_features(selector: Any, features: list[str]) -> None:
    selector.selected_features_ = list(features)
    selector.selected_features = list(features)
    if hasattr(selector, "boruta") and getattr(selector, "boruta", None) is not None:
        selector.boruta.selected_features = list(features)


def _infer_final_budget(
    stat_selector_kwargs: dict[str, Any],
    fallback: int,
) -> int:
    if "k" in stat_selector_kwargs:
        return int(stat_selector_kwargs["k"])
    if "n_features" in stat_selector_kwargs:
        return int(stat_selector_kwargs["n_features"])
    rfe_kwargs = stat_selector_kwargs.get("rfe_kwargs", {})
    if isinstance(rfe_kwargs, dict) and "n_features" in rfe_kwargs:
        return int(rfe_kwargs["n_features"])
    if "keep" in stat_selector_kwargs:
        return int(stat_selector_kwargs["keep"])
    return int(fallback)


class LLMThenStatSelector:
    """
    Sequential selector that applies a broad LLM ranking before statistical refinement.
    """

    def __init__(
        self,
        description_csv_path: str,
        stat_selector_cls: type,
        stat_selector_kwargs: dict[str, Any] | None = None,
        cache_dir: str = "results/_llm_rankings_cache",
        llm_model: str = "gpt-4.1-mini",
        llm_temperature: float = 0.0,
        llm_max_features: int = 100,
        llm_candidate_pool_budget: int | None = None,
        llm_shared_ranking_enabled: bool = True,
        llm_config_hash: str | None = None,
        llm_prompt_version: str = "stability_expert_v3",
        llm_ranking_budget_config: dict[str, int] | None = None,
        llm_shared_pool_size: int | None = None,
        final_feature_budget: int | None = None,
        iv_filter_kwargs: dict[str, Any] | None = None,
        llm_selector_kwargs: dict[str, Any] | None = None,
        llm_selector_cls: type | None = None,
    ):
        if stat_selector_cls is None:
            raise ValueError("stat_selector_cls is required for LLMThenStatSelector.")

        self.description_csv_path = description_csv_path
        self.stat_selector_cls = stat_selector_cls
        self.stat_selector_kwargs = dict(stat_selector_kwargs or {})
        self.cache_dir = cache_dir
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_max_features = int(llm_max_features)
        self.llm_candidate_pool_budget = int(llm_candidate_pool_budget or llm_max_features)
        self.llm_shared_ranking_enabled = llm_shared_ranking_enabled
        self.llm_config_hash = llm_config_hash
        self.llm_prompt_version = llm_prompt_version
        self.llm_ranking_budget_config = dict(llm_ranking_budget_config or {})
        self.llm_shared_pool_size = int(llm_shared_pool_size or llm_max_features)
        self.final_feature_budget = int(
            final_feature_budget
            or _infer_final_budget(self.stat_selector_kwargs, self.llm_candidate_pool_budget)
        )
        self.iv_filter_kwargs = dict(iv_filter_kwargs or {})
        self.llm_selector_kwargs = dict(llm_selector_kwargs or {})
        self.llm_selector_cls = llm_selector_cls

        self.artifact_dir: Path | None = None
        self.ranking_context: dict[str, Any] = {}
        self.llm_selector: Any | None = None
        self.stat_selector: Any | None = None
        self.llm_selected_features_: list[str] | None = None
        self.selected_features: list[str] | None = None
        self.selected_features_: list[str] | None = None
        self.select_before_preprocessing = True
        self.apply_post_preprocessing = True

    def set_artifact_dir(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)

    def set_ranking_context(self, **kwargs: Any) -> None:
        self.ranking_context = dict(kwargs)

    def _build_llm_selector(self):
        llm_selector_cls = self.llm_selector_cls
        if llm_selector_cls is None:
            from feature_selection.llm_selector import LLMSelector

            llm_selector_cls = LLMSelector

        kwargs = {
            "description_csv_path": self.description_csv_path,
            "cache_dir": self.cache_dir,
            "model": self.llm_model,
            "temperature": self.llm_temperature,
            "max_features": self.llm_max_features,
            "ranking_budget": self.llm_shared_pool_size,
            "feature_budget": self.llm_candidate_pool_budget,
            "shared_ranking_enabled": self.llm_shared_ranking_enabled,
            "config_hash": self.llm_config_hash,
            "prompt_version": self.llm_prompt_version,
            "ranking_budget_config": self.llm_ranking_budget_config,
            "shared_pool_size": self.llm_shared_pool_size,
            "iv_filter_kwargs": dict(self.iv_filter_kwargs),
        }
        kwargs.update(self.llm_selector_kwargs)
        return llm_selector_cls(**kwargs)

    def _candidate_order(self, X: pd.DataFrame) -> list[str]:
        ordered = []
        for feature in self.llm_selected_features_ or []:
            if feature in X.columns and feature not in ordered:
                ordered.append(feature)
        for feature in X.columns.tolist():
            if feature not in ordered:
                ordered.append(feature)
        return ordered

    def _finalize_features(self, X: pd.DataFrame, selected: list[str]) -> list[str]:
        finalized = [feature for feature in selected if feature in X.columns]
        if len(finalized) > self.final_feature_budget:
            finalized = finalized[: self.final_feature_budget]

        if len(finalized) < self.final_feature_budget:
            for feature in self._candidate_order(X):
                if feature in finalized:
                    continue
                finalized.append(feature)
                if len(finalized) >= self.final_feature_budget:
                    break
        return finalized[: self.final_feature_budget]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        if y is None:
            raise ValueError("LLMThenStatSelector requires target labels during fit.")

        self.llm_selector = self._build_llm_selector()
        if self.artifact_dir is not None and hasattr(self.llm_selector, "set_artifact_dir"):
            self.llm_selector.set_artifact_dir(self.artifact_dir / "llm")
        if hasattr(self.llm_selector, "set_ranking_context"):
            self.llm_selector.set_ranking_context(**self.ranking_context)

        X_llm = self.llm_selector.fit_transform(X, y)
        self.llm_selected_features_ = X_llm.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.llm_selector is None:
            raise ValueError("LLMThenStatSelector must be fitted before transform.")
        return self.llm_selector.transform(X)

    def fit_postprocess(self, X: pd.DataFrame, y: pd.Series):
        self.stat_selector = self.stat_selector_cls(**self.stat_selector_kwargs)
        if self.artifact_dir is not None and hasattr(self.stat_selector, "set_artifact_dir"):
            self.stat_selector.set_artifact_dir(self.artifact_dir / "statistical")

        self.stat_selector.fit(X, y)
        selected = _selector_features(self.stat_selector) or X.columns.tolist()
        finalized = self._finalize_features(X, selected)
        _set_selector_features(self.stat_selector, finalized)
        self.selected_features = finalized
        self.selected_features_ = finalized
        return X[finalized]

    def transform_postprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise ValueError("Hybrid statistical selector must be fitted before transform_postprocess.")
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        X_llm = self.fit(X, y).transform(X)
        return self.fit_postprocess(X_llm, y)


class StableCoreLLMFillSelector:
    """
    Build a cheap statistically stable core, then fill remaining slots from the LLM ranking.
    """

    def __init__(
        self,
        description_csv_path: str,
        cache_dir: str = "results/_llm_rankings_cache",
        llm_model: str = "gpt-4.1-mini",
        llm_temperature: float = 0.0,
        llm_max_features: int = 100,
        llm_shared_ranking_enabled: bool = True,
        llm_config_hash: str | None = None,
        llm_prompt_version: str = "stability_expert_v3",
        llm_ranking_budget_config: dict[str, int] | None = None,
        llm_shared_pool_size: int | None = None,
        final_feature_budget: int = 40,
        bootstrap_iterations: int = 5,
        bootstrap_fraction: float = 0.8,
        stability_threshold: float = 0.8,
        random_state: int = 42,
        llm_selector_cls: type | None = None,
        llm_selector_kwargs: dict[str, Any] | None = None,
        iv_filter_kwargs: dict[str, Any] | None = None,
    ):
        self.description_csv_path = description_csv_path
        self.cache_dir = cache_dir
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.llm_max_features = int(llm_max_features)
        self.llm_shared_ranking_enabled = llm_shared_ranking_enabled
        self.llm_config_hash = llm_config_hash
        self.llm_prompt_version = llm_prompt_version
        self.llm_ranking_budget_config = dict(llm_ranking_budget_config or {})
        self.llm_shared_pool_size = int(llm_shared_pool_size or llm_max_features)
        self.final_feature_budget = int(final_feature_budget)
        self.bootstrap_iterations = int(bootstrap_iterations)
        self.bootstrap_fraction = float(bootstrap_fraction)
        self.stability_threshold = float(stability_threshold)
        self.random_state = int(random_state)
        self.llm_selector_cls = llm_selector_cls
        self.llm_selector_kwargs = dict(llm_selector_kwargs or {})
        self.iv_filter_kwargs = dict(iv_filter_kwargs or {})

        self.artifact_dir: Path | None = None
        self.ranking_context: dict[str, Any] = {}
        self.llm_selector: Any | None = None
        self.llm_selected_features_: list[str] | None = None
        self.stable_core_features_: list[str] | None = None
        self.stable_core_frequency_: pd.DataFrame | None = None
        self.selected_features: list[str] | None = None
        self.selected_features_: list[str] | None = None
        self.select_before_preprocessing = True
        self.apply_post_preprocessing = True

    def set_artifact_dir(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)

    def set_ranking_context(self, **kwargs: Any) -> None:
        self.ranking_context = dict(kwargs)

    def _build_llm_selector(self):
        llm_selector_cls = self.llm_selector_cls
        if llm_selector_cls is None:
            from feature_selection.llm_selector import LLMSelector

            llm_selector_cls = LLMSelector

        kwargs = {
            "description_csv_path": self.description_csv_path,
            "cache_dir": self.cache_dir,
            "model": self.llm_model,
            "temperature": self.llm_temperature,
            "max_features": self.llm_max_features,
            "ranking_budget": self.llm_shared_pool_size,
            "feature_budget": self.llm_shared_pool_size,
            "shared_ranking_enabled": self.llm_shared_ranking_enabled,
            "config_hash": self.llm_config_hash,
            "prompt_version": self.llm_prompt_version,
            "ranking_budget_config": self.llm_ranking_budget_config,
            "shared_pool_size": self.llm_shared_pool_size,
            "iv_filter_kwargs": dict(self.iv_filter_kwargs),
        }
        kwargs.update(self.llm_selector_kwargs)
        return llm_selector_cls(**kwargs)

    def _bootstrap_core(self, X: pd.DataFrame, y: pd.Series) -> tuple[list[str], pd.DataFrame]:
        if X.empty:
            return [], pd.DataFrame(columns=["feature_name", "selection_frequency", "mean_rank"])

        rows = []
        sample_size = max(2, int(np.ceil(len(X) * self.bootstrap_fraction)))
        k = min(self.final_feature_budget, X.shape[1])

        for iteration in range(self.bootstrap_iterations):
            sampled_idx = y.sample(
                n=sample_size,
                replace=True,
                random_state=self.random_state + iteration,
            ).index
            selector = MRMR(
                k=k,
                method="mrmr",
                random_state=self.random_state + iteration,
            )
            selector.fit(X.loc[sampled_idx], y.loc[sampled_idx])
            for rank, feature in enumerate(_selector_features(selector), start=1):
                rows.append(
                    {
                        "iteration": iteration + 1,
                        "feature_name": feature,
                        "rank": rank,
                    }
                )

        if not rows:
            return [], pd.DataFrame(columns=["feature_name", "selection_frequency", "mean_rank"])

        frequency_df = (
            pd.DataFrame(rows)
            .groupby("feature_name", as_index=False)
            .agg(
                selection_count=("iteration", "count"),
                mean_rank=("rank", "mean"),
            )
        )
        frequency_df["selection_frequency"] = (
            frequency_df["selection_count"] / float(self.bootstrap_iterations)
        )
        frequency_df = frequency_df.sort_values(
            ["selection_frequency", "mean_rank", "feature_name"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

        stable_core = frequency_df.loc[
            frequency_df["selection_frequency"] >= self.stability_threshold,
            "feature_name",
        ].tolist()
        stable_core = stable_core[: self.final_feature_budget]
        return stable_core, frequency_df

    def _finalize_features(self, X: pd.DataFrame) -> list[str]:
        finalized: list[str] = []
        for feature in self.stable_core_features_ or []:
            if feature in X.columns and feature not in finalized:
                finalized.append(feature)

        for feature in self.llm_selected_features_ or []:
            if len(finalized) >= self.final_feature_budget:
                break
            if feature in X.columns and feature not in finalized:
                finalized.append(feature)

        if len(finalized) < self.final_feature_budget and self.stable_core_frequency_ is not None:
            for feature in self.stable_core_frequency_["feature_name"].tolist():
                if len(finalized) >= self.final_feature_budget:
                    break
                if feature in X.columns and feature not in finalized:
                    finalized.append(feature)

        if len(finalized) < self.final_feature_budget:
            for feature in X.columns.tolist():
                if len(finalized) >= self.final_feature_budget:
                    break
                if feature not in finalized:
                    finalized.append(feature)

        return finalized[: self.final_feature_budget]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        if y is None:
            raise ValueError("StableCoreLLMFillSelector requires target labels during fit.")

        self.llm_selector = self._build_llm_selector()
        if self.artifact_dir is not None and hasattr(self.llm_selector, "set_artifact_dir"):
            self.llm_selector.set_artifact_dir(self.artifact_dir / "llm")
        if hasattr(self.llm_selector, "set_ranking_context"):
            self.llm_selector.set_ranking_context(**self.ranking_context)

        self.llm_selector.fit(X, y)
        self.llm_selected_features_ = list(getattr(self.llm_selector, "ranked_features_", []) or [])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.llm_selector is None:
            raise ValueError("StableCoreLLMFillSelector must be fitted before transform.")
        return X.copy()

    def fit_postprocess(self, X: pd.DataFrame, y: pd.Series):
        self.stable_core_features_, self.stable_core_frequency_ = self._bootstrap_core(X, y)
        self.selected_features_ = self._finalize_features(X)
        self.selected_features = list(self.selected_features_)

        if self.artifact_dir is not None and self.stable_core_frequency_ is not None:
            statistical_dir = self.artifact_dir / "statistical"
            statistical_dir.mkdir(parents=True, exist_ok=True)
            self.stable_core_frequency_.to_csv(
                statistical_dir / "stable_core_frequency.csv",
                index=False,
            )

        return X[self.selected_features_]

    def transform_postprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise ValueError("StableCoreLLMFillSelector must be fitted before transform_postprocess.")
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        X_raw = self.fit(X, y).transform(X)
        return self.fit_postprocess(X_raw, y)


class DomainRuleBaselineSelector:
    """
    Lightweight domain-rule baseline that mimics coarse expert-style pre-screening.
    """

    GROUP_PRIORITY = {
        "external_score": 100,
        "bureau_debt": 95,
        "bureau_credit_history": 90,
        "installment_repayment_behavior": 85,
        "delinquency_behavior": 80,
        "credit_card_utilization": 78,
        "income_capacity": 74,
        "previous_application_behavior": 70,
        "application_amounts": 64,
        "demographic_time_variables": 58,
        "missingness_or_unknown": 10,
        "other": 40,
    }

    def __init__(self, description_csv_path: str, feature_budget: int = 40):
        self.description_csv_path = description_csv_path
        self.feature_budget = int(feature_budget)
        self.selected_features: list[str] | None = None
        self.selected_features_: list[str] | None = None
        self.artifact_dir: Path | None = None
        self.select_before_preprocessing = True

    def set_artifact_dir(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        metadata = build_feature_metadata(X, self.description_csv_path)
        ranked = sorted(
            metadata,
            key=lambda item: (
                -self.GROUP_PRIORITY.get(str(item.get("semantic_group", "other")), 0),
                float(item.get("missing_rate", 1.0)),
                -float(item.get("non_null_count", 0)),
                str(item.get("name", "")),
            ),
        )
        self.selected_features_ = [item["name"] for item in ranked[: self.feature_budget]]
        self.selected_features = list(self.selected_features_)

        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(ranked).to_csv(self.artifact_dir / "domain_rule_ranking.csv", index=False)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise ValueError("DomainRuleBaselineSelector must be fitted before transform.")
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
