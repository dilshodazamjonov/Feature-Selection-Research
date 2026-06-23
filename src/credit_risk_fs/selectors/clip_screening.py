from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk_fs.clip.selector_adapter import ClipScoreAdapter


class ClipScreeningSelector:
    """Frozen CLIP-style semantic-statistical score selector."""

    def __init__(
        self,
        *,
        config_path: str = "configs/clip/selector.yaml",
        dataset: str = "homecredit",
        feature_budget: int = 40,
        model_name: str | None = None,
        missing_feature_policy: str = "error",
        selector_label: str = "clip",
    ) -> None:
        self.config_path = config_path
        self.dataset = dataset
        self.feature_budget = int(feature_budget)
        self.model_name = model_name
        self.missing_feature_policy = missing_feature_policy
        self.selector_label = selector_label
        self.selected_features_: list[str] | None = None
        self.selected_features: list[str] | None = None
        self.ranking_: pd.DataFrame | None = None
        self.selection_manifest_: pd.DataFrame | None = None
        self.artifact_dir: Path | None = None
        self.select_before_preprocessing = True

    def set_artifact_dir(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        adapter = ClipScoreAdapter(self.config_path, dataset=self.dataset)
        ranking = adapter.rank_candidates(X.columns.astype(str).tolist(), missing_feature_policy=self.missing_feature_policy)
        selected = ranking[ranking["exclusion_reason"].astype(str).eq("")].head(self.feature_budget).copy()
        if len(selected) < min(self.feature_budget, X.shape[1]):
            raise RuntimeError("CLIP selector could not satisfy requested feature budget")
        selected["screening_pool_member"] = True
        selected["final_selected"] = True
        selected["final_rank"] = range(1, len(selected) + 1)
        selected["selector"] = self.selector_label
        selected["model"] = self.model_name or ""
        selected["clip_score"] = selected["learned_similarity"]
        self.selected_features_ = selected["feature_name"].astype(str).tolist()
        self.selected_features = list(self.selected_features_)
        self.ranking_ = ranking
        self.selection_manifest_ = _selection_columns(selected, dataset=self.dataset)
        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            name = "clip_v2_selection_manifest.csv" if self.selector_label == "clip_v2" else "clip_selection_manifest.csv"
            self.selection_manifest_.to_csv(self.artifact_dir / name, index=False)
            ranking_name = "clip_v2_candidate_ranking.csv" if self.selector_label == "clip_v2" else "clip_candidate_ranking.csv"
            ranking.to_csv(self.artifact_dir / ranking_name, index=False)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise ValueError("ClipScreeningSelector must be fitted before transform.")
        return X.loc[:, self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def _selection_columns(frame: pd.DataFrame, *, dataset: str) -> pd.DataFrame:
    out = frame.copy()
    out["dataset"] = dataset
    return out[
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
    ].copy()
