from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk_fs.selectors.base import (
    SelectedFeaturesMixin,
    get_selected_features,
    set_selected_features,
)


def _infer_final_budget(
    stat_selector_kwargs: dict[str, Any],
    fallback: int,
) -> int:
    """Infer a downstream selector budget from supported public kwargs."""

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


class LLMThenStatSelector(SelectedFeaturesMixin):
    """Apply a fold-local LLM screen before statistical refinement."""

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
    ) -> None:
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
        if self.final_feature_budget < 0:
            raise ValueError("final_feature_budget must be non-negative.")
        self.iv_filter_kwargs = dict(iv_filter_kwargs or {})
        self.llm_selector_kwargs = dict(llm_selector_kwargs or {})
        self.llm_selector_cls = llm_selector_cls

        self.artifact_dir: Path | None = None
        self.ranking_context: dict[str, Any] = {}
        self.llm_selector: Any | None = None
        self.stat_selector: Any | None = None
        self.llm_selected_features_: list[str] | None = None
        self.selected_features_ = None
        self.select_before_preprocessing = True
        self.apply_post_preprocessing = True

    def set_artifact_dir(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)

    def set_ranking_context(self, **kwargs: Any) -> None:
        self.ranking_context = dict(kwargs)

    def _build_llm_selector(self) -> Any:
        llm_selector_cls = self.llm_selector_cls
        if llm_selector_cls is None:
            from credit_risk_fs.selectors.llm_screening import LLMSelector

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
        ordered: list[str] = []
        for feature in self.llm_selected_features_ or []:
            if feature in X.columns and feature not in ordered:
                ordered.append(feature)
        for feature in X.columns.astype(str).tolist():
            if feature not in ordered:
                ordered.append(feature)
        return ordered

    def _finalize_features(self, X: pd.DataFrame, selected: list[str]) -> list[str]:
        finalized = list(dict.fromkeys(feature for feature in selected if feature in X.columns))
        finalized = finalized[: self.final_feature_budget]
        for feature in self._candidate_order(X):
            if len(finalized) >= self.final_feature_budget:
                break
            if feature not in finalized:
                finalized.append(feature)
        return finalized

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> LLMThenStatSelector:
        if y is None:
            raise ValueError("LLMThenStatSelector requires target labels during fit.")

        self.llm_selector = self._build_llm_selector()
        if self.artifact_dir is not None and hasattr(self.llm_selector, "set_artifact_dir"):
            self.llm_selector.set_artifact_dir(self.artifact_dir / "llm")
        if hasattr(self.llm_selector, "set_ranking_context"):
            self.llm_selector.set_ranking_context(**self.ranking_context)

        X_llm = self.llm_selector.fit_transform(X, y)
        self.llm_selected_features_ = X_llm.columns.astype(str).tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.llm_selector is None:
            raise ValueError("LLMThenStatSelector must be fitted before transform.")
        return self.llm_selector.transform(X)

    def fit_postprocess(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.stat_selector = self.stat_selector_cls(**self.stat_selector_kwargs)
        if self.artifact_dir is not None and hasattr(self.stat_selector, "set_artifact_dir"):
            self.stat_selector.set_artifact_dir(self.artifact_dir / "statistical")

        self.stat_selector.fit(X, y)
        selected = get_selected_features(self.stat_selector)
        if selected is None:
            raise ValueError("Downstream statistical selector did not expose fitted features.")
        finalized = self._finalize_features(X, selected)
        set_selected_features(self.stat_selector, finalized)
        self.selected_features_ = finalized
        return X.loc[:, finalized]

    def transform_postprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise ValueError("Hybrid statistical selector must be fitted first.")
        return X.loc[:, self.selected_features_]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pd.DataFrame:
        X_llm = self.fit(X, y).transform(X)
        return self.fit_postprocess(X_llm, y)


__all__ = ["LLMThenStatSelector"]
