from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_risk_fs.selectors.base import (
    SelectedFeaturesMixin,
    resolve_feature_budget,
    select_feature_frame,
    validate_feature_frame,
)
from credit_risk_fs.selectors.domain_rules import rank_features


class DomainRuleBaselineSelector(SelectedFeaturesMixin):
    """Rank candidate names with the deterministic credit-domain scorer.

    The scorer reads feature names only (see :mod:`credit_risk_fs.selectors.domain_rules`):
    no data row, statistic or outcome is used, so the ranking is identical in every fold.
    """

    def __init__(self, description_csv_path: str | None = None, feature_budget: int = 40) -> None:
        self.description_csv_path = description_csv_path
        self.feature_budget = int(feature_budget)
        if self.feature_budget < 0:
            raise ValueError("feature_budget must be non-negative.")
        self.selected_features_ = None
        self.ranking_: list[str] | None = None
        self.ranking_table_: pd.DataFrame | None = None
        self.artifact_dir: Path | None = None
        self.select_before_preprocessing = True

    def set_artifact_dir(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> DomainRuleBaselineSelector:
        names = validate_feature_frame(X)
        ranked = rank_features(names)
        self.ranking_table_ = pd.DataFrame(ranked)
        self.ranking_ = [str(item["feature_name"]) for item in ranked]
        budget = resolve_feature_budget(self.feature_budget, len(self.ranking_))
        self.selected_features_ = self.ranking_[:budget]

        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            self.ranking_table_.to_csv(self.artifact_dir / "domain_rule_ranking.csv", index=False)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return select_feature_frame(
            X,
            self.selected_features_,
            selector_name=self.__class__.__name__,
        )

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


__all__ = ["DomainRuleBaselineSelector"]
