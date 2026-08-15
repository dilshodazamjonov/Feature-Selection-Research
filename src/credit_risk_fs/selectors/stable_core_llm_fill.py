from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk_fs.selectors.base import (
    SelectedFeaturesMixin,
    get_selected_features,
    validate_feature_frame,
)
from credit_risk_fs.selectors.mrmr import RandomForestRelevanceMRMRSelector


class StableCoreLLMFillSelector(SelectedFeaturesMixin):
    """Build a bootstrap RF-mRMR-like core, then fill from an LLM ranking."""

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
        component_n_jobs: int = 1,
        llm_selector_cls: type | None = None,
        llm_selector_kwargs: dict[str, Any] | None = None,
        iv_filter_kwargs: dict[str, Any] | None = None,
        allow_unranked_padding: bool = True,
    ) -> None:
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
        self.component_n_jobs = int(component_n_jobs)
        self.llm_selector_cls = llm_selector_cls
        self.llm_selector_kwargs = dict(llm_selector_kwargs or {})
        self.iv_filter_kwargs = dict(iv_filter_kwargs or {})
        self.allow_unranked_padding = bool(allow_unranked_padding)
        if self.final_feature_budget < 0:
            raise ValueError("final_feature_budget must be non-negative.")
        if self.bootstrap_iterations <= 0:
            raise ValueError("bootstrap_iterations must be positive.")
        if not 0 < self.bootstrap_fraction <= 1:
            raise ValueError("bootstrap_fraction must be in (0, 1].")
        if not 0 <= self.stability_threshold <= 1:
            raise ValueError("stability_threshold must be in [0, 1].")
        if self.component_n_jobs <= 0:
            raise ValueError("component_n_jobs must be positive.")

        self.artifact_dir: Path | None = None
        self.ranking_context: dict[str, Any] = {}
        self.llm_selector: Any | None = None
        self.llm_selected_features_: list[str] | None = None
        self.stable_core_features_: list[str] | None = None
        self.stable_core_frequency_: pd.DataFrame | None = None
        self.selected_features_ = None
        self.authenticated_ranking_sha256_: str | None = None
        self.bootstrap_trace_: list[dict[str, Any]] = []
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

    def _bootstrap_core(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> tuple[list[str], pd.DataFrame]:
        empty = pd.DataFrame(
            columns=["feature_name", "selection_count", "mean_rank", "selection_frequency"]
        )
        if X.shape[1] == 0 or self.final_feature_budget == 0:
            return [], empty

        rows: list[dict[str, Any]] = []
        self.bootstrap_trace_ = []
        sample_size = max(2, int(np.ceil(len(X) * self.bootstrap_fraction)))
        k = min(self.final_feature_budget, X.shape[1])
        for iteration in range(self.bootstrap_iterations):
            sampled_indices = y.sample(
                n=sample_size,
                replace=True,
                random_state=self.random_state + iteration,
            ).index
            sample_digest = hashlib.sha256()
            for value in sampled_indices:
                sample_digest.update(f"{value}\n".encode("utf-8"))
            self.bootstrap_trace_.append(
                {
                    "iteration": iteration + 1,
                    "random_state": self.random_state + iteration,
                    "sample_size": len(sampled_indices),
                    "unique_training_index_count": len(set(sampled_indices)),
                    "ordered_sampled_training_index_sha256": sample_digest.hexdigest(),
                    "replace": True,
                    "training_index_only": True,
                }
            )
            selector = RandomForestRelevanceMRMRSelector(
                k=k,
                method="mrmr",
                random_state=self.random_state + iteration,
                n_jobs=self.component_n_jobs,
            )
            selector.fit(X.loc[sampled_indices], y.loc[sampled_indices])
            for rank, feature in enumerate(get_selected_features(selector) or [], start=1):
                rows.append(
                    {
                        "iteration": iteration + 1,
                        "feature_name": feature,
                        "rank": rank,
                    }
                )
        if not rows:
            return [], empty

        frequency = (
            pd.DataFrame(rows)
            .groupby("feature_name", as_index=False)
            .agg(selection_count=("iteration", "count"), mean_rank=("rank", "mean"))
        )
        frequency["selection_frequency"] = (
            frequency["selection_count"] / float(self.bootstrap_iterations)
        )
        frequency = frequency.sort_values(
            ["selection_frequency", "mean_rank", "feature_name"],
            ascending=[False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        stable_core = frequency.loc[
            frequency["selection_frequency"] >= self.stability_threshold,
            "feature_name",
        ].tolist()[: self.final_feature_budget]
        return stable_core, frequency

    def _finalize_features(self, X: pd.DataFrame) -> list[str]:
        available = set(X.columns.astype(str))
        finalized: list[str] = []
        candidate_groups = [
            self.stable_core_features_ or [],
            self.llm_selected_features_ or [],
        ]
        if self.allow_unranked_padding:
            candidate_groups.extend(
                [
                    (
                        self.stable_core_frequency_["feature_name"].astype(str).tolist()
                        if self.stable_core_frequency_ is not None
                        else []
                    ),
                    X.columns.astype(str).tolist(),
                ]
            )
        for candidates in candidate_groups:
            for feature in candidates:
                if len(finalized) >= self.final_feature_budget:
                    return finalized
                if feature in available and feature not in finalized:
                    finalized.append(feature)
        return finalized

    def fit_with_authenticated_ranking(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        ranked_features: list[str],
        ranking_manifest_sha256: str,
    ) -> StableCoreLLMFillSelector:
        """Fit only the hybrid's supervised stable core around a sealed LLM rank."""

        validate_feature_frame(X)
        if y is None:
            raise ValueError("StableCoreLLMFillSelector requires target labels during fit.")
        ranking = [str(value) for value in ranked_features]
        if len(ranking) != self.llm_shared_pool_size:
            raise ValueError("authenticated LLM ranking has the wrong shared-pool size")
        if len(ranking) != len(set(ranking)):
            raise ValueError("authenticated LLM ranking contains duplicates")
        if not set(ranking).issubset(set(X.columns.astype(str))):
            raise ValueError("authenticated LLM ranking escaped the training feature universe")
        if len(ranking_manifest_sha256) != 64:
            raise ValueError("authenticated LLM ranking manifest digest is invalid")
        self.llm_selected_features_ = ranking
        self.authenticated_ranking_sha256_ = str(ranking_manifest_sha256)
        self.stable_core_features_, self.stable_core_frequency_ = self._bootstrap_core(X, y)
        self.selected_features_ = self._finalize_features(X)
        return self

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> StableCoreLLMFillSelector:
        validate_feature_frame(X)
        if y is None:
            raise ValueError("StableCoreLLMFillSelector requires target labels during fit.")

        self.llm_selector = self._build_llm_selector()
        if self.artifact_dir is not None and hasattr(self.llm_selector, "set_artifact_dir"):
            self.llm_selector.set_artifact_dir(self.artifact_dir / "llm")
        if hasattr(self.llm_selector, "set_ranking_context"):
            self.llm_selector.set_ranking_context(**self.ranking_context)
        self.llm_selector.fit(X, y)
        self.llm_selected_features_ = [
            str(feature)
            for feature in (getattr(self.llm_selector, "ranked_features_", None) or [])
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.llm_selector is None:
            raise ValueError("StableCoreLLMFillSelector must be fitted before transform.")
        return X.copy()

    def fit_postprocess(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.stable_core_features_, self.stable_core_frequency_ = self._bootstrap_core(X, y)
        self.selected_features_ = self._finalize_features(X)

        if self.artifact_dir is not None:
            statistical_dir = self.artifact_dir / "statistical"
            statistical_dir.mkdir(parents=True, exist_ok=True)
            self.stable_core_frequency_.to_csv(
                statistical_dir / "stable_core_frequency.csv",
                index=False,
            )
        return X.loc[:, self.selected_features_]

    def transform_postprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise ValueError("StableCoreLLMFillSelector must be fitted first.")
        return X.loc[:, self.selected_features_]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pd.DataFrame:
        X_raw = self.fit(X, y).transform(X)
        return self.fit_postprocess(X_raw, y)


__all__ = ["StableCoreLLMFillSelector"]
