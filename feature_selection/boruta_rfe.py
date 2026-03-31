import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.feature_selection import RFE
from catboost import CatBoostClassifier


class BorutaSelector:

    def __init__(self, max_iter: int = 10, random_state: int = 42):
        """
        Parameters:
            max_iter: max iterations for Boruta
            random_state: random seed
        """
        self.max_iter = max_iter
        self.random_state = random_state
        self.selected_features = None
        self.selector = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit Boruta feature selection on X, y and store selected features.
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
            verbose=2
        )

        print("\n========== BORUTA FEATURE SELECTION STARTED ==========\n")

        self.selector.fit(X.values, y.values)

        mask = self.selector.support_

        self.selected_features = X.columns[mask]

        print("\n========== BORUTA FINISHED ==========")
        print(f"Selected features: {len(self.selected_features)}")

        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform X by keeping only selected features.
        """
        if self.selected_features is None:
            raise ValueError("BorutaSelector not fitted")
        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit Boruta and transform X in one step.
        """
        self.fit(X, y)
        return self.transform(X)


class RFESelector:

    def __init__(self, n_features: int = 50, step: int = 10, random_state: int = 42):
        """
        Parameters:
            n_features: number of features to select
            step: step size for RFE elimination
            random_state: random seed
        """
        self.n_features = n_features
        self.step = step
        self.random_state = random_state
        self.selected_features = None
        self.selector = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit RFE feature selection using CatBoostClassifier as estimator.
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

        self.selector.fit(X, y)

        mask = self.selector.support_

        self.selected_features = X.columns[mask]

        print(f"RFE selected {len(self.selected_features)} features")

        return self

    def transform(self, X: pd.DataFrame):
        """
        Transform X by keeping only selected features.
        """
        if self.selected_features is None:
            raise ValueError("RFESelector not fitted")
        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit RFE and transform X in one step.
        """
        self.fit(X, y)
        return self.transform(X)