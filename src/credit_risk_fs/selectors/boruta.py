from __future__ import annotations

import logging
import warnings
from typing import Any

import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

from credit_risk_fs.selectors.base import (
    SelectedFeaturesMixin,
    resolve_feature_budget,
    select_feature_frame,
    validate_feature_frame,
)


logger = logging.getLogger(__name__)


class BorutaSelector(SelectedFeaturesMixin):
    """Select confirmed Boruta features using only the supplied training data."""

    def __init__(
        self,
        max_iter: int = 10,
        random_state: int = 42,
        n_features: int | None = None,
    ) -> None:
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.n_features = None if n_features is None else int(n_features)
        self.selected_features_ = None
        self.selector: BorutaPy | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BorutaSelector:
        """Fit Boruta on one training boundary; no OOT data may be supplied."""

        validate_feature_frame(X)
        if y is None:
            raise ValueError("BorutaSelector requires target labels during fit.")
        if X.shape[1] == 0:
            self.selected_features_ = []
            return self

        estimator = RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            n_jobs=-1,
            random_state=self.random_state,
        )
        self.selector = BorutaPy(
            estimator=estimator,
            n_estimators="auto",
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=0,
        )

        logger.info("Starting Boruta feature selection")
        self.selector.fit(X.to_numpy(), pd.Series(y).to_numpy())

        confirmed = [
            str(feature)
            for feature, supported in zip(X.columns, self.selector.support_, strict=True)
            if supported
        ]
        ranking = {
            str(feature): int(rank)
            for feature, rank in zip(X.columns, self.selector.ranking_, strict=True)
        }
        confirmed.sort(key=lambda feature: (ranking[feature], feature))
        if self.n_features is not None:
            confirmed = confirmed[
                : resolve_feature_budget(self.n_features, len(confirmed))
            ]
        self.selected_features_ = confirmed

        logger.info("Boruta finished - selected features: %d", len(confirmed))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Keep the confirmed fitted features, including a valid empty result."""

        return select_feature_frame(
            X,
            self.selected_features_,
            selector_name=self.__class__.__name__,
        )

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def __getattr__(name: str) -> Any:
    """Resolve old imports lazily without reintroducing mixed ownership."""

    if name == "RFESelector":
        warnings.warn(
            "Import RFESelector from credit_risk_fs.selectors.rfe.",
            DeprecationWarning,
            stacklevel=2,
        )
        from credit_risk_fs.selectors.rfe import RFESelector

        return RFESelector
    if name == "BorutaRFESelector":
        warnings.warn(
            "Import BorutaThenRFESelector from "
            "credit_risk_fs.selectors.boruta_then_rfe.",
            DeprecationWarning,
            stacklevel=2,
        )
        from credit_risk_fs.selectors.boruta_then_rfe import BorutaRFESelector

        return BorutaRFESelector
    raise AttributeError(name)


__all__ = ["BorutaSelector"]
