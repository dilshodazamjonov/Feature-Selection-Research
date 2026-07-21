from __future__ import annotations

import logging

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_selection import RFE

from credit_risk_fs.selectors.base import (
    SelectedFeaturesMixin,
    resolve_feature_budget,
    select_feature_frame,
    validate_feature_frame,
)


logger = logging.getLogger(__name__)


class RFESelector(SelectedFeaturesMixin):
    """Apply CatBoost-backed recursive elimination within one training boundary."""

    def __init__(
        self,
        n_features: int = 50,
        step: int = 10,
        random_state: int = 42,
    ) -> None:
        self.n_features = int(n_features)
        self.step = int(step)
        self.random_state = int(random_state)
        self.selected_features_ = None
        self.selector: RFE | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RFESelector:
        """Fit RFE using training labels and deterministic CatBoost seeding."""

        validate_feature_frame(X)
        if y is None:
            raise ValueError("RFESelector requires target labels during fit.")
        budget = resolve_feature_budget(self.n_features, X.shape[1])
        if budget == 0:
            self.selected_features_ = []
            return self
        if budget == X.shape[1]:
            self.selected_features_ = [str(feature) for feature in X.columns]
            return self

        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            verbose=False,
            random_state=self.random_state,
            allow_writing_files=False,
        )
        self.selector = RFE(
            estimator=model,
            n_features_to_select=budget,
            step=self.step,
        )

        logger.info("Starting RFE feature selection")
        self.selector.fit(X, y)
        self.selected_features_ = [
            str(feature)
            for feature, supported in zip(X.columns, self.selector.support_, strict=True)
            if supported
        ]
        logger.info("RFE finished - selected features: %d", len(self.selected_features_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return select_feature_frame(
            X,
            self.selected_features_,
            selector_name=self.__class__.__name__,
        )

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


__all__ = ["RFESelector"]
