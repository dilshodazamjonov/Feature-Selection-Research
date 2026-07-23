from __future__ import annotations

import logging

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.feature_selection import RFE

from credit_risk_fs.selectors.base import (
    SelectedFeaturesMixin,
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
        thread_count: int = 1,
    ) -> None:
        self.n_features = int(n_features)
        self.step = int(step)
        self.random_state = int(random_state)
        self.thread_count = int(thread_count)
        if self.thread_count <= 0:
            raise ValueError("thread_count must be positive")
        self.selected_features_ = None
        self.selector: RFE | None = None
        self.selection_trace_: pd.DataFrame | None = None
        self.effective_estimator_config_: dict[str, object] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RFESelector:
        """Fit RFE using training labels and deterministic CatBoost seeding."""

        validate_feature_frame(X)
        if y is None:
            raise ValueError("RFESelector requires target labels during fit.")
        if self.n_features <= 0:
            raise ValueError("RFE exact feature budget must be positive")
        if self.n_features > X.shape[1]:
            raise ValueError(
                "RFE exact feature budget exceeds projected candidate count: "
                f"requested={self.n_features}, candidates={X.shape[1]}"
            )
        budget = self.n_features
        if budget == X.shape[1]:
            self.selected_features_ = [str(feature) for feature in X.columns]
            self.selection_trace_ = pd.DataFrame(
                {
                    "feature": self.selected_features_,
                    "input_order": range(1, len(self.selected_features_) + 1),
                    "rfe_rank": [1] * len(self.selected_features_),
                    "selected": [True] * len(self.selected_features_),
                    "step": [self.step] * len(self.selected_features_),
                }
            )
            self.effective_estimator_config_ = self._estimator_config()
            return self

        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            verbose=False,
            random_state=self.random_state,
            allow_writing_files=False,
            thread_count=self.thread_count,
            task_type="CPU",
        )
        self.effective_estimator_config_ = self._estimator_config()
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
        if len(self.selected_features_) != budget or len(set(self.selected_features_)) != budget:
            raise RuntimeError(
                "RFE failed its exact-budget contract: "
                f"requested={budget}, observed={len(self.selected_features_)}"
            )
        self.selection_trace_ = pd.DataFrame(
            {
                "feature": [str(feature) for feature in X.columns],
                "input_order": range(1, X.shape[1] + 1),
                "rfe_rank": [int(value) for value in self.selector.ranking_],
                "selected": [bool(value) for value in self.selector.support_],
                "step": [self.step] * X.shape[1],
            }
        )
        logger.info("RFE finished - selected features: %d", len(self.selected_features_))
        return self

    def _estimator_config(self) -> dict[str, object]:
        return {
            "implementation": "catboost.CatBoostClassifier",
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.05,
            "verbose": False,
            "random_state": self.random_state,
            "allow_writing_files": False,
            "thread_count": self.thread_count,
            "task_type": "CPU",
            "rfe_step": self.step,
            "n_features_to_select": self.n_features,
        }

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return select_feature_frame(
            X,
            self.selected_features_,
            selector_name=self.__class__.__name__,
        )

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


__all__ = ["RFESelector"]
