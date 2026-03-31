import logging
import warnings
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, MetaEstimatorMixin
from sklearn.ensemble import RandomForestClassifier

class MRMR(TransformerMixin, BaseEstimator, MetaEstimatorMixin):
    """
    RF / mRMR feature selector compatible with sklearn pipelines
    """
    def __init__(
        self,
        *,
        k: int,
        method: str,
        n_iter: int = 1,
        correlation: str = "pearson",
        random_state: int = 42,
    ):
        self.k = k
        self.method = method
        self.n_iter = n_iter
        self.correlation = correlation
        self.random_state = random_state
        self.logger = logging.getLogger(self.__class__.__name__)

    # ======== RF IMPORTANCE ========
    def get_rf_importances(self, X: pd.DataFrame, y: pd.Series):
        self.logger.info("[RF] Computing feature importances (%d iterations)", self.n_iter)

        importances = []

        for i in range(self.n_iter):
            self.logger.info("[RF] Training iteration %d/%d", i + 1, self.n_iter)

            rf = RandomForestClassifier(
                n_estimators=128,
                min_samples_split=0.01,
                max_features=0.15,
                n_jobs=-1,
                random_state=self.random_state + i,
            )
            rf.fit(X, y)
            importances.append(rf.feature_importances_)

        imp_df = pd.DataFrame(importances, columns=X.columns)
        self.rf_importances_ = imp_df.mean().sort_values(ascending=False)
        self.k_top_rf_ = self.rf_importances_.head(self.k).index.tolist()

        self.logger.info("[RF] Top 10 features: %s", self.k_top_rf_[:10])

    # ======== RFCQ SCORE (MRMR) ========
    @staticmethod
    def _rfcq_score(X, candidates, selected, relevance, corr_method):
        X_sel = X[selected]

        redundancy = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for c in candidates:
                corr = np.abs(X_sel.corrwith(X[c], method=corr_method))
                redundancy[c] = max(corr.mean(), 0.05)

        redundancy = pd.Series(redundancy)
        return relevance.loc[candidates] / redundancy

    # ======== MRMR FEATURE SELECTION ========
    def get_mrmr_features(self, X: pd.DataFrame):
        self.logger.info("[MRMR] Starting mRMR feature selection")
        self.logger.info("[MRMR] Target number of features: %d", self.k)

        selected = [self.rf_importances_.index[0]]
        selected_scores = [self.rf_importances_.iloc[0]]  # first feature score is its RF relevance
        self.logger.info("[MRMR] Initial feature selected: %s", selected[0])

        for step in range(1, self.k):
            remaining = [c for c in self.rf_importances_.index if c not in selected]

            if not remaining:
                self.logger.warning("[MRMR] No remaining features, stopping early")
                break

            self.logger.info(
                "[MRMR] Iteration %d/%d | Remaining features: %d",
                step,
                self.k,
                len(remaining),
            )

            n_jobs = min(cpu_count(), len(remaining))
            chunks = np.array_split(remaining, n_jobs)

            scores = Parallel(n_jobs=n_jobs)(
                delayed(self._rfcq_score)(
                    X,
                    chunk.tolist(),
                    selected,
                    self.rf_importances_,
                    self.correlation,
                )
                for chunk in chunks
            )

            scores = pd.concat(scores)
            next_feature = scores.idxmax()
            next_score = scores.loc[next_feature]

            selected.append(next_feature)
            selected_scores.append(next_score)  # store the score

            self.logger.info(
                "[MRMR] Selected feature %d: %s (score=%.6f)",
                step + 1,
                next_feature,
                next_score,
            )

        # ===== Store selected features and their scores =====
        self.mrmr_features_ = selected
        self.mrmr_scores_ = selected_scores
        self.logger.info("[MRMR] Completed mRMR selection")

    # ======== FIT ========
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.logger.info("[FIT] Fitting FeatureSelector (%s)", self.method)

        if self.k >= X.shape[1]:
            self.logger.warning("[FIT] k >= number of features, selecting all")
            self.selected_features_ = X.columns.tolist()
            return self

        self.get_rf_importances(X, y)

        if self.method == "rf":
            self.selected_features_ = self.k_top_rf_
        elif self.method == "mrmr":
            self.get_mrmr_features(X)
            self.selected_features_ = self.mrmr_features_
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        self.logger.info(
            "[FIT] Feature selection completed (%d features)",
            len(self.selected_features_),
        )

        return self

    # ======== TRANSFORM ========
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("[TRANSFORM] Transforming dataset")
        return X.loc[:, self.selected_features_]