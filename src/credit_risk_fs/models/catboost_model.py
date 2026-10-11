# models/catboost_model.py

import os
import pandas as pd
from catboost import CatBoostClassifier


class CatBoostModel:
    """
    CatBoost binary classifier with pipeline-friendly interface.

    The defaults are the fixed backbone configuration: depth 10, learning rate 0.01,
    L2 leaf regularisation 95, at least 290 observations per leaf, feature-sampling
    rate (``rsm``) 0.90, random strength 0.125, depthwise growth, Bernoulli bootstrap
    with subsample 0.55, Logloss with AUC evaluation, balanced class weights, seed 42,
    four threads and a fixed 1,500 iterations without early stopping.
    """

    def __init__(
        self,
        depth=10,
        learning_rate=0.01,
        l2_leaf_reg=95,
        min_data_in_leaf=290,
        rsm=0.9,
        random_strength=0.125,
        grow_policy='Depthwise',
        one_hot_max_size=21,
        leaf_estimation_method='Newton',
        bootstrap_type='Bernoulli',
        subsample=0.55,
        loss_function='Logloss',
        eval_metric='AUC',
        auto_class_weights='Balanced',
        iterations=1500,
        verbose=0,
        random_state=42,
        allow_writing_files=False,
        thread_count=4,
        colsample_bylevel=None,
        early_stopping_rounds=None,
    ):
        """
        Initialize CatBoost model with hyperparameters.

        ``colsample_bylevel`` is CatBoost's alias of ``rsm`` and is accepted for older
        configuration files. ``early_stopping_rounds`` is accepted for the same reason but
        never forwarded: the model always runs the full number of iterations.
        """
        if colsample_bylevel is not None:
            rsm = colsample_bylevel
        self.model = CatBoostClassifier(
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            min_data_in_leaf=min_data_in_leaf,
            rsm=rsm,
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
            verbose=verbose,
            random_state=random_state,
            allow_writing_files=allow_writing_files,
            thread_count=thread_count,
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
            Ignored. No validation information enters the fitted model.
        """
        names = getattr(X, "columns", None)
        if names is None:
            names = getattr(X, "feature_names", None)
        self.feature_names = (
            list(names) if names is not None else list(range(int(X.shape[1])))
        )
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
