import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


class BorutaSelector:

    def __init__(
        self,
        max_iter: int = 5,
        random_state: int = 42
    ):
        self.max_iter = max_iter
        self.random_state = random_state
        self.selected_features = None
        self.selector = None

    def fit(self, X: pd.DataFrame, y: pd.Series):

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

        if self.selected_features is None:
            raise ValueError("BorutaSelector not fitted")

        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):

        self.fit(X, y)
        return self.transform(X)