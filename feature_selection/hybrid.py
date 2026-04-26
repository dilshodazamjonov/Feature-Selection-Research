from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class LLMThenStatSelector:
    """
    Sequential selector that applies LLM preselection before a statistical selector.

    The first stage keeps the raw feature names intact, which lets the second
    stage operate on a smaller, more curated candidate set.
    """

    def __init__(
        self,
        description_csv_path: str,
        stat_selector_cls: type,
        stat_selector_kwargs: dict[str, Any] | None = None,
        cache_dir: str = "outputs/llm_selector_cache",
        llm_model: str = "gpt-4.1-mini",
        llm_temperature: float = 0.0,
        llm_max_features: int = 50,
        llm_feature_budget: int | None = None,
        llm_shared_ranking_enabled: bool = True,
        llm_config_hash: str | None = None,
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
        self.llm_max_features = llm_max_features
        self.llm_feature_budget = llm_feature_budget or llm_max_features
        self.llm_shared_ranking_enabled = llm_shared_ranking_enabled
        self.llm_config_hash = llm_config_hash
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
            "ranking_budget": self.llm_max_features,
            "feature_budget": self.llm_feature_budget,
            "shared_ranking_enabled": self.llm_shared_ranking_enabled,
            "config_hash": self.llm_config_hash,
            "iv_filter_kwargs": dict(self.iv_filter_kwargs),
        }
        kwargs.update(self.llm_selector_kwargs)
        return llm_selector_cls(**kwargs)

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

        X_llm = self.llm_selector.transform(X)
        return X_llm

    def fit_postprocess(self, X: pd.DataFrame, y: pd.Series):
        self.stat_selector = self.stat_selector_cls(**self.stat_selector_kwargs)
        if self.artifact_dir is not None and hasattr(self.stat_selector, "set_artifact_dir"):
            self.stat_selector.set_artifact_dir(self.artifact_dir / "statistical")

        X_final = self.stat_selector.fit_transform(X, y)
        if not isinstance(X_final, pd.DataFrame):
            raise TypeError(
                "LLMThenStatSelector expects the statistical selector to return a pandas DataFrame."
            )

        self.selected_features = X_final.columns.tolist()
        self.selected_features_ = X_final.columns.tolist()
        return X_final

    def transform_postprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.stat_selector is None:
            raise ValueError("Hybrid statistical selector must be fitted before transform_postprocess.")

        X_final = self.stat_selector.transform(X)
        if not isinstance(X_final, pd.DataFrame):
            raise TypeError(
                "LLMThenStatSelector expects the statistical selector to return a pandas DataFrame."
            )
        return X_final

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        X_llm = self.fit(X, y).transform(X)
        return self.fit_postprocess(X_llm, y)
