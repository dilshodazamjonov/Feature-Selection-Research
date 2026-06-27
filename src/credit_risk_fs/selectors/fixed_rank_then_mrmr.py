from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_risk_fs.selectors.mrmr import MRMR


class FixedRankThenMRMRSelector:
    """Target-free frozen candidate ranking followed by DEV-only mRMR."""

    select_before_preprocessing = True
    apply_post_preprocessing = True

    def __init__(
        self,
        *,
        ranking_path: str,
        feature_budget: int,
        screening_pool_size: int,
        approved_features_path: str | None = None,
        approved_feature_column: str = "feature_name",
        random_state: int = 42,
        selector_label: str = "corrected_clip_then_mrmr",
        rank_column: str = "consensus_clip_rank",
    ) -> None:
        self.ranking_path = ranking_path
        self.feature_budget = int(feature_budget)
        self.screening_pool_size = int(screening_pool_size)
        self.approved_features_path = approved_features_path
        self.approved_feature_column = approved_feature_column
        self.random_state = int(random_state)
        self.selector_label = selector_label
        self.rank_column = rank_column
        self.artifact_dir: Path | None = None

    def set_artifact_dir(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        ranking = pd.read_csv(self.ranking_path).sort_values(
            [self.rank_column, "feature_name"], kind="mergesort"
        )
        eligible = set(X.columns.astype(str))
        if self.approved_features_path:
            approved = pd.read_csv(self.approved_features_path)
            eligible &= set(approved[self.approved_feature_column].astype(str))
        ranking = ranking[ranking["feature_name"].astype(str).isin(eligible)].copy()
        self.screened_features_ = ranking.head(self.screening_pool_size)[
            "feature_name"
        ].astype(str).tolist()
        if len(self.screened_features_) < self.feature_budget:
            raise RuntimeError("fixed candidate pool is smaller than the final feature budget")
        self.ranking_ = ranking
        self.raw_source_feature_count_ = len(self.screened_features_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.screened_features_]

    def fit_postprocess(self, X: pd.DataFrame, y: pd.Series):
        self.post_preprocessing_column_count_ = int(X.shape[1])
        selector = MRMR(
            k=min(self.feature_budget, X.shape[1]),
            method="mrmr",
            random_state=self.random_state,
        )
        selector.fit(X, y)
        self.selected_features_ = list(selector.selected_features_)[: self.feature_budget]
        self.selected_features = list(self.selected_features_)
        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            frame = self.ranking_.copy()
            frame["screening_pool_member"] = frame["feature_name"].isin(
                self.screened_features_
            )
            frame["final_selected"] = frame["feature_name"].isin(
                self.selected_features_
            )
            final_rank = {
                feature: rank
                for rank, feature in enumerate(self.selected_features_, start=1)
            }
            frame["final_rank"] = frame["feature_name"].map(final_rank)
            frame["selector"] = self.selector_label
            frame.to_csv(
                self.artifact_dir / f"{self.selector_label}_selection_manifest.csv",
                index=False,
            )
            source_features = sorted(self.screened_features_, key=len, reverse=True)
            lineage = []
            for column in X.columns.astype(str):
                source = next(
                    (
                        feature
                        for feature in source_features
                        if column == feature
                        or column.startswith(f"{feature}_")
                        or column.startswith(f"{feature}=")
                    ),
                    column,
                )
                lineage.append(
                    {
                        "source_feature": source,
                        "model_column": column,
                        "selected_by_mrmr": column in self.selected_features_,
                    }
                )
            pd.DataFrame(lineage).to_csv(
                self.artifact_dir / f"{self.selector_label}_source_to_model_lineage.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "raw_source_feature_count": self.raw_source_feature_count_,
                        "post_preprocessing_mrmr_column_count": self.post_preprocessing_column_count_,
                        "final_selected_model_feature_count": len(self.selected_features_),
                        "candidate_pool_frozen_before_mrmr": True,
                    }
                ]
            ).to_csv(
                self.artifact_dir / f"{self.selector_label}_mrmr_widths.csv",
                index=False,
            )
        return X.loc[:, self.selected_features_]

    def transform_postprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.loc[:, self.selected_features_]
