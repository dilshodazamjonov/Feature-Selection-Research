from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_risk_fs.feature_metadata.builder import build_feature_metadata
from credit_risk_fs.selectors.base import (
    SelectedFeaturesMixin,
    resolve_feature_budget,
    select_feature_frame,
    validate_feature_frame,
)


class DomainRuleBaselineSelector(SelectedFeaturesMixin):
    """Rank training-slice metadata with fixed, deterministic domain rules."""

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

    def __init__(self, description_csv_path: str, feature_budget: int = 40) -> None:
        self.description_csv_path = description_csv_path
        self.feature_budget = int(feature_budget)
        if self.feature_budget < 0:
            raise ValueError("feature_budget must be non-negative.")
        self.selected_features_ = None
        self.artifact_dir: Path | None = None
        self.select_before_preprocessing = True

    def set_artifact_dir(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> DomainRuleBaselineSelector:
        validate_feature_frame(X)
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
        budget = resolve_feature_budget(self.feature_budget, len(ranked))
        self.selected_features_ = [str(item["name"]) for item in ranked[:budget]]

        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(ranked).to_csv(
                self.artifact_dir / "domain_rule_ranking.csv",
                index=False,
            )
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
