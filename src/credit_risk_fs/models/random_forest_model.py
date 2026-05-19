# models/random_forest_model.py

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    """
    Random Forest binary classifier with pipeline-friendly interface.
    """

    def __init__(
        self,
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ):
        """
        Initialize Random Forest model with hyperparameters.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.feature_names = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None):
        """
        Fit the Random Forest model on training data.
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
        Return a DataFrame of feature importance.
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        importances = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

    def save(self, path: str):
        """
        Save Random Forest model to file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str):
        """
        Load Random Forest model from file.
        """
        self.model = joblib.load(path)
        self.fitted = True
        return self

    