from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_risk_fs.clip.selector_adapter import ClipScoreAdapter
from credit_risk_fs.selectors.mrmr import MRMR


class ClipThenMRMRSelector:
    """Frozen CLIP screening followed by existing mRMR on DEV training data."""

    def __init__(
        self,
        *,
        config_path: str = "configs/clip/selector.yaml",
        dataset: str = "homecredit",
        feature_budget: int = 40,
        screening_pool_size: int = 100,
        model_name: str | None = None,
        missing_feature_policy: str = "error",
        random_state: int = 42,
        mrmr_method: str = "mrmr",
        selector_label: str = "clip_then_mrmr",
    ) -> None:
        self.config_path = config_path
        self.dataset = dataset
        self.feature_budget = int(feature_budget)
        self.screening_pool_size = int(screening_pool_size)
        self.model_name = model_name
        self.missing_feature_policy = missing_feature_policy
        self.random_state = int(random_state)
        self.mrmr_method = mrmr_method
        self.selector_label = selector_label
        self.selected_features_: list[str] | None = None
        self.selected_features: list[str] | None = None
        self.screened_features_: list[str] | None = None
        self.clip_ranking_: pd.DataFrame | None = None
        self.mrmr_selector_: MRMR | None = None
        self.selection_manifest_: pd.DataFrame | None = None
        self.mrmr_input_row_count_: int | None = None
        self.artifact_dir: Path | None = None
        self.select_before_preprocessing = True
        self.apply_post_preprocessing = True

    def set_artifact_dir(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        adapter = ClipScoreAdapter(self.config_path, dataset=self.dataset)
        ranking = adapter.rank_candidates(X.columns.astype(str).tolist(), missing_feature_policy=self.missing_feature_policy)
        scored = ranking[ranking["exclusion_reason"].astype(str).eq("")].copy()
        pool_size = min(self.screening_pool_size, len(scored))
        if pool_size == 0:
            raise RuntimeError("CLIP screening produced an empty mRMR pool")
        self.screened_features_ = scored.head(pool_size)["feature_name"].astype(str).tolist()
        self.clip_ranking_ = ranking
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.screened_features_ is None:
            raise ValueError("ClipThenMRMRSelector must be fitted before transform.")
        return X.loc[:, self.screened_features_]

    def fit_postprocess(self, X: pd.DataFrame, y: pd.Series):
        if y is None:
            raise ValueError("ClipThenMRMRSelector requires target labels for mRMR.")
        self.mrmr_input_row_count_ = int(len(X))
        final_budget = min(self.feature_budget, X.shape[1])
        self.mrmr_selector_ = MRMR(k=final_budget, method=self.mrmr_method, random_state=self.random_state)
        self.mrmr_selector_.fit(X, y)
        selected = list(self.mrmr_selector_.selected_features_ or [])
        self.selected_features_ = selected[:final_budget]
        self.selected_features = list(self.selected_features_)
        self.selection_manifest_ = self._build_manifest()
        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            name = (
                "clip_v2_then_mrmr_selection_manifest.csv"
                if self.selector_label == "clip_v2_then_mrmr"
                else "clip_then_mrmr_selection_manifest.csv"
            )
            self.selection_manifest_.to_csv(self.artifact_dir / name, index=False)
        return X.loc[:, self.selected_features_]

    def transform_postprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise ValueError("ClipThenMRMRSelector must be fitted before transform_postprocess.")
        return X.loc[:, self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        X_screened = self.fit(X, y).transform(X)
        return self.fit_postprocess(X_screened, y)

    def _build_manifest(self) -> pd.DataFrame:
        if self.clip_ranking_ is None or self.screened_features_ is None or self.selected_features_ is None:
            raise RuntimeError("ClipThenMRMRSelector manifest requested before fit")
        ranked = self.clip_ranking_[self.clip_ranking_["feature_name"].isin(self.screened_features_)].copy()
        ranked["screening_pool_member"] = True
        ranked["final_selected"] = ranked["feature_name"].isin(self.selected_features_)
        final_rank = {feature: rank for rank, feature in enumerate(self.selected_features_, start=1)}
        ranked["final_rank"] = ranked["feature_name"].map(final_rank)
        ranked["selector"] = self.selector_label
        ranked["model"] = self.model_name or ""
        ranked["clip_score"] = ranked["learned_similarity"]
        ranked["dataset"] = self.dataset
        return ranked[
            [
                "dataset",
                "model",
                "selector",
                "feature_name",
                "clip_score",
                "clip_rank",
                "screening_pool_member",
                "final_selected",
                "final_rank",
                "checkpoint_hash",
                "anchor_hash",
                "source_manifest_hash",
                "exclusion_reason",
                "statistical_view_scope",
            ]
        ].sort_values(["final_selected", "final_rank", "clip_rank"], ascending=[False, True, True], kind="mergesort")
