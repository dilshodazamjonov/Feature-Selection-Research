import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from boruta import BorutaPy
from catboost import CatBoostClassifier
from utils.logging_config import setup_logging

# Setup module logger
logger = setup_logging("boruta_rfe", level=logging.INFO)


class BorutaSelector:
    """
    Performs feature selection using the Boruta algorithm.

    Boruta works by comparing real features with shadow (randomized) features
    and keeps only statistically significant ones.
    """

    def __init__(self, max_iter: int = 10, random_state: int = 42):
        """
        Args:
            max_iter: Maximum Boruta iterations.
            random_state: Seed for reproducibility.
        """
        self.max_iter = max_iter
        self.random_state = random_state
        self.selected_features = None
        self.selector = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits Boruta on the dataset.

        Args:
            X: Feature DataFrame
            y: Target Series
        """
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            n_jobs=-1,
            random_state=self.random_state
        )

        self.selector = BorutaPy(
            estimator=rf,
            n_estimators="auto",
            max_iter=self.max_iter,
            random_state=self.random_state,
            verbose=0
        )

        logger.info("Starting Boruta feature selection")

        self.selector.fit(X.values, y.values)

        mask = self.selector.support_
        self.selected_features = X.columns[mask].tolist()

        logger.info(f"Boruta finished - Selected features: {len(self.selected_features)}")

        return self

    def transform(self, X: pd.DataFrame):
        """
        Transforms dataset by keeping only selected features.

        Args:
            X: Input DataFrame

        Returns:
            Filtered DataFrame
        """
        if self.selected_features is None:
            raise ValueError("BorutaSelector not fitted")

        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits Boruta and transforms the dataset.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Filtered DataFrame
        """
        self.fit(X, y)
        return self.transform(X)


class RFESelector:
    """
    Performs Recursive Feature Elimination (RFE) using CatBoost.

    This class is intended for internal usage (e.g., after Boruta),
    not as a standalone selector in the pipeline.
    """

    def __init__(self, n_features: int = 50, step: int = 10, random_state: int = 42):
        """
        Args:
            n_features: Number of features to keep.
            step: Number of features to remove at each iteration.
            random_state: Seed for reproducibility.
        """
        self.n_features = n_features
        self.step = step
        self.random_state = random_state
        self.selected_features = None
        self.selector = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits RFE on the dataset.

        Args:
            X: Feature DataFrame
            y: Target Series
        """
        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            verbose=False,
            random_state=self.random_state
        )

        self.selector = RFE(
            estimator=model,
            n_features_to_select=self.n_features,
            step=self.step
        )

        logger.info("Starting RFE feature selection")

        self.selector.fit(X, y)

        mask = self.selector.support_
        self.selected_features = X.columns[mask].tolist()

        logger.info(f"RFE finished - Selected features: {len(self.selected_features)}")

        return self

    def transform(self, X: pd.DataFrame):
        """
        Transforms dataset using selected RFE features.

        Args:
            X: Input DataFrame

        Returns:
            Filtered DataFrame
        """
        if self.selected_features is None:
            raise ValueError("RFESelector not fitted")

        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits RFE and transforms dataset.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Filtered DataFrame
        """
        self.fit(X, y)
        return self.transform(X)


class BorutaRFESelector:
    """
    Boruta feature selector with optional RFE refinement.

    RFE is disabled by default for the research matrix because it repeatedly
    trains CatBoost and dominates runtime. When disabled, Boruta-selected
    features are capped to ``n_features`` using Boruta ranking/support.
    """

    def __init__(self, boruta_kwargs=None, rfe_kwargs=None, use_rfe: bool = False, n_features: int = 40):
        """
        Args:
            boruta_kwargs: Parameters for BorutaSelector
            rfe_kwargs: Parameters for RFESelector
            use_rfe: Whether to run the expensive RFE refinement stage.
            n_features: Final feature cap when RFE is disabled.
        """
        self.boruta = BorutaSelector(**(boruta_kwargs or {}))
        self.rfe_kwargs = dict(rfe_kwargs or {})
        self.use_rfe = use_rfe
        self.n_features = n_features
        self.rfe = RFESelector(**self.rfe_kwargs) if use_rfe else None
        self.selected_features = None

    def _boruta_capped_features(self, X: pd.DataFrame) -> list[str]:
        selector = self.boruta.selector
        if selector is None:
            return self.boruta.selected_features or X.columns.tolist()[: self.n_features]

        ranking = pd.Series(selector.ranking_, index=X.columns)
        support = pd.Series(selector.support_, index=X.columns)
        weak_support = pd.Series(selector.support_weak_, index=X.columns)

        ordered = (
            pd.DataFrame(
                {
                    "feature": X.columns,
                    "rank": ranking.values,
                    "support": support.values.astype(int),
                    "weak_support": weak_support.values.astype(int),
                }
            )
            .sort_values(["support", "weak_support", "rank", "feature"], ascending=[False, False, True, True])
            ["feature"]
            .tolist()
        )
        return ordered[: min(self.n_features, len(ordered))]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Runs Boruta, optionally followed by RFE.

        Args:
            X: Feature DataFrame
            y: Target Series
        """

        X_boruta = self.boruta.fit_transform(X, y)
        logger.info(f"After Boruta: {X_boruta.shape}")

        if self.use_rfe and self.rfe is not None:
            X_rfe = self.rfe.fit_transform(X_boruta, y)
            logger.info(f"After RFE: {X_rfe.shape}")
            self.selected_features = X_rfe.columns.tolist()
        else:
            self.selected_features = self._boruta_capped_features(X)
            logger.info(
                "RFE disabled - using top %s Boruta-ranked features",
                len(self.selected_features),
            )

        logger.info(f"Feature selection finished - Final selected features: {len(self.selected_features)}")

        return self

    def transform(self, X: pd.DataFrame):
        """
        Applies Boruta + RFE transformations.

        Args:
            X: Input DataFrame

        Returns:
            Filtered DataFrame
        """
        if self.selected_features is None:
            raise ValueError("BorutaRFESelector not fitted")
        if self.use_rfe and self.rfe is not None:
            X_boruta = self.boruta.transform(X)
            return self.rfe.transform(X_boruta)
        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the pipeline and transforms dataset.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Filtered DataFrame
        """
        self.fit(X, y)
        return self.transform(X)
