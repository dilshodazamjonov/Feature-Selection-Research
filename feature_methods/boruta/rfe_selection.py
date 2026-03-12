import pandas as pd

from sklearn.feature_selection import RFE
from catboost import CatBoostClassifier


class RFESelector:

    def __init__(
        self,
        n_features: int = 50,
        step: int = 10,
        random_state: int = 42
    ):

        self.n_features = n_features
        self.step = step
        self.random_state = random_state

        self.selected_features = None
        self.selector = None

    def fit(self, X: pd.DataFrame, y: pd.Series):

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

        if self.selected_features is None:
            raise ValueError("RFESelector not fitted")

        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series):

        self.fit(X, y)
        return self.transform(X)