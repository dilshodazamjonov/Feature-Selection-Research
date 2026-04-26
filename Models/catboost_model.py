# models/catboost_model.py

import os
import pandas as pd
from catboost import CatBoostClassifier

class CatBoostModel:
    """
    CatBoost binary classifier with pipeline-friendly interface.
    """

    def __init__(
        self,
        depth=10,
        learning_rate=0.01,
        l2_leaf_reg=95,
        min_data_in_leaf=290,
        colsample_bylevel=0.9,
        random_strength=0.125,
        grow_policy='Depthwise',
        one_hot_max_size=21,
        leaf_estimation_method='Newton',
        bootstrap_type='Bernoulli',
        subsample=0.55,
        loss_function='Logloss',
        eval_metric='AUC',
        auto_class_weights='Balanced',
        iterations=2200,
        early_stopping_rounds=150,
        verbose=100,
        random_state=42,
        allow_writing_files=False,
    ):
        """
        Initialize CatBoost model with hyperparameters.
        """
        self.model = CatBoostClassifier(
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            min_data_in_leaf=min_data_in_leaf,
            colsample_bylevel=colsample_bylevel,
            random_strength=random_strength,
            grow_policy=grow_policy,
            one_hot_max_size=one_hot_max_size,
            leaf_estimation_method=leaf_estimation_method,
            bootstrap_type=bootstrap_type,
            subsample=subsample,
            loss_function=loss_function,
            eval_metric=eval_metric,
            auto_class_weights=auto_class_weights,
            iterations=iterations,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            random_state=random_state,
            allow_writing_files=allow_writing_files,
        )
        self.feature_names = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series, eval_set=None):
        """
        Fit the CatBoost model on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training target
        eval_set : tuple, optional
            Validation set (X_val, y_val)
        """
        self.feature_names = X.columns
        self.model.fit(X, y, eval_set=eval_set)
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
        importances = self.model.get_feature_importance()
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

    def save(self, path: str):
        """
        Save CatBoost model to file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str):
        """
        Load CatBoost model from file.
        """
        self.model.load_model(path)
        self.fitted = True
        return self
