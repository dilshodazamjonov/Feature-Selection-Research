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
    Combined feature selection pipeline:
        1. Boruta (coarse filtering)
        2. RFE (fine selection)

    This is the main selector to be used when 'boruta' is specified.
    """

    def __init__(self, boruta_kwargs=None, rfe_kwargs=None):
        """
        Args:
            boruta_kwargs: Parameters for BorutaSelector
            rfe_kwargs: Parameters for RFESelector
        """
        self.boruta = BorutaSelector(**(boruta_kwargs or {}))
        self.rfe = RFESelector(**(rfe_kwargs or {}))
        self.selected_features = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Runs Boruta followed by RFE.

        Args:
            X: Feature DataFrame
            y: Target Series
        """

        X_boruta = self.boruta.fit_transform(X, y)
        logger.info(f"After Boruta: {X_boruta.shape}")

        X_rfe = self.rfe.fit_transform(X_boruta, y)
        logger.info(f"After RFE: {X_rfe.shape}")

        self.selected_features = X_rfe.columns.tolist()

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
        X_boruta = self.boruta.transform(X)
        X_rfe = self.rfe.transform(X_boruta)
        return X_rfe

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