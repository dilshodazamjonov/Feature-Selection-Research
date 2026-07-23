from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

from credit_risk_fs.selectors.base import (
    SelectedFeaturesMixin,
    resolve_feature_budget,
    select_feature_frame,
    validate_feature_frame,
)


class RandomForestRelevanceMRMRSelector(
    SelectedFeaturesMixin,
    TransformerMixin,
    BaseEstimator,
    MetaEstimatorMixin,
):
    """Greedy mRMR-like selector with RF relevance and correlation redundancy.

    Relevance is mean random-forest impurity importance. Redundancy is the mean
    absolute configured correlation with already selected features, floored at
    0.05. The greedy score is relevance divided by redundancy. This is not the
    canonical mutual-information definition of mRMR. Fitting must receive only
    the current training boundary.
    """

    algorithm_name = "rf_relevance_correlation_redundancy"
    canonical_mrmr = False

    def __init__(
        self,
        *,
        k: int,
        method: str,
        n_iter: int = 1,
        correlation: str = "pearson",
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        self.k = int(k)
        self.method = str(method)
        self.n_iter = int(n_iter)
        self.correlation = str(correlation)
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        if self.n_jobs <= 0:
            raise ValueError("n_jobs must be positive")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.selected_features_ = None

    def get_rf_importances(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Calculate deterministic mean RF impurity relevance scores."""

        self.logger.info("[RF] Computing feature importances (%d iterations)", self.n_iter)
        importances: list[np.ndarray] = []
        for iteration in range(self.n_iter):
            estimator = RandomForestClassifier(
                n_estimators=128,
                min_samples_split=0.01,
                max_features=0.15,
                n_jobs=self.n_jobs,
                random_state=self.random_state + iteration,
            )
            estimator.fit(X, y)
            importances.append(estimator.feature_importances_)

        mean_importance = pd.DataFrame(importances, columns=X.columns).mean()
        ordered = (
            mean_importance.rename("importance")
            .rename_axis("feature")
            .reset_index()
            .assign(feature=lambda frame: frame["feature"].astype(str))
            .sort_values(
                ["importance", "feature"],
                ascending=[False, True],
                kind="mergesort",
            )
        )
        self.rf_importances_ = ordered.set_index("feature")["importance"]
        self.k_top_rf_ = self.rf_importances_.head(self.k).index.tolist()

    def get_mrmr_features(self, X: pd.DataFrame) -> None:
        """Run greedy RF-relevance/correlation-redundancy selection."""

        selected = [str(self.rf_importances_.index[0])]
        max_samples = 10_000
        if len(X) > max_samples:
            sample_indices = np.random.RandomState(self.random_state).choice(
                len(X),
                max_samples,
                replace=False,
            )
            X_sample = X.iloc[sample_indices]
        else:
            X_sample = X

        # Pearson correlations do not change as the selected set grows.  Build
        # the matrix once, then update mean redundancy from the relevant rows.
        # This is algebraically identical to repeatedly calling Series.corr but
        # avoids hundreds of thousands of redundant full-column scans for the
        # frozen 529/675-feature universes.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            correlation_matrix = X_sample.loc[:, self.rf_importances_.index].corr(
                method=self.correlation
            ).abs()
        correlation_matrix = correlation_matrix.replace([np.inf, -np.inf], np.nan)
        score_trace: list[dict[str, object]] = [
            {
                "selection_rank": 1,
                "feature": selected[0],
                "relevance": float(self.rf_importances_.loc[selected[0]]),
                "mean_absolute_correlation": 0.0,
                "selection_score": float(self.rf_importances_.loc[selected[0]]) / 0.05,
            }
        ]

        for _ in range(1, self.k):
            remaining = [
                str(feature)
                for feature in self.rf_importances_.index
                if feature not in selected
            ]
            if not remaining:
                break

            correlations = correlation_matrix.loc[selected, remaining]
            mean_correlation = correlations.mean(axis=0, skipna=True).fillna(0.0)
            redundancy = mean_correlation.clip(lower=0.05)

            scores = pd.DataFrame(
                {
                    "feature": remaining,
                    "score": [
                        float(self.rf_importances_.loc[feature]) / float(redundancy.loc[feature])
                        for feature in remaining
                    ],
                }
            ).sort_values(
                ["score", "feature"],
                ascending=[False, True],
                kind="mergesort",
            )
            chosen = str(scores.iloc[0]["feature"])
            selected.append(chosen)
            score_trace.append(
                {
                    "selection_rank": len(selected),
                    "feature": chosen,
                    "relevance": float(self.rf_importances_.loc[chosen]),
                    "mean_absolute_correlation": float(mean_correlation.loc[chosen]),
                    "selection_score": float(scores.iloc[0]["score"]),
                }
            )

        self.mrmr_features_ = selected
        self.selection_trace_ = pd.DataFrame(score_trace)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> RandomForestRelevanceMRMRSelector:
        """Fit the configured RF-top-k or custom mRMR-like mode."""

        names = validate_feature_frame(X)
        if y is None:
            raise ValueError(f"{self.__class__.__name__} requires target labels during fit.")
        budget = resolve_feature_budget(self.k, X.shape[1])
        if budget == 0:
            self.selected_features_ = []
            return self
        if budget == X.shape[1]:
            self.selected_features_ = names
            return self

        # The selection encoder already supplies canonical string column names.
        # Avoid copying the complete training matrix solely to assign an equal
        # Index; retain a shallow metadata view only when normalization is
        # actually required by a third-party caller.
        if list(X.columns) == names:
            X_named = X
        else:
            X_named = X.copy(deep=False)
            X_named.columns = names
        self.get_rf_importances(X_named, y)

        if self.method == "rf":
            self.selected_features_ = list(self.k_top_rf_)[:budget]
        elif self.method == "mrmr":
            self.get_mrmr_features(X_named)
            self.selected_features_ = list(self.mrmr_features_)[:budget]
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        self.logger.info(
            "[FIT] Feature selection completed (%d features)",
            len(self.selected_features_),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return select_feature_frame(
            X,
            self.selected_features_,
            selector_name=self.__class__.__name__,
        )


# Backward-compatible class name and registry vocabulary. A future canonical
# mutual-information implementation must use a separate class/registry entry.
MRMR = RandomForestRelevanceMRMRSelector


__all__ = ["RandomForestRelevanceMRMRSelector", "MRMR"]
