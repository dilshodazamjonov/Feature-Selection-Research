# models/logistic_regression_model.py

import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    """
    Logistic Regression binary classifier with pipeline-friendly interface.
    """

    def __init__(
        self,
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ):
        """
        Initialize Logistic Regression model with hyperparameters.
        """
        self.model = LogisticRegression(
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state
        )
        self.feature_names = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None):
        """
        Fit the Logistic Regression model on training data.
        """
        self.feature_names = X.columns
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame):
        """
        Predict probabilities for the positive class.
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold=0.5):
        """
        Predict binary labels based on threshold.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self):
        """
        Return a DataFrame of feature importance based on absolute coefficients.
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        coefs = self.model.coef_[0]
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": abs(coefs)
        }).sort_values("importance", ascending=False)

    def save(self, path: str):
        """
        Save Logistic Regression model to file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str):
        """
        Load Logistic Regression model from file.
        """
        self.model = joblib.load(path)
        self.fitted = True
        return self