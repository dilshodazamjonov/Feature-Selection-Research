from __future__ import annotations

from typing import Any

import pandas as pd
from catboost import CatBoostClassifier


DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "depth": 10,
    "learning_rate": 0.01,
    "l2_leaf_reg": 95,
    "min_data_in_leaf": 290,
    "colsample_bylevel": 0.9,
    "random_strength": 0.125,
    "grow_policy": "Depthwise",
    "one_hot_max_size": 21,
    "leaf_estimation_method": "Newton",
    "bootstrap_type": "Bernoulli",
    "subsample": 0.55,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "auto_class_weights": "Balanced",
    "iterations": 2200,
    "early_stopping_rounds": 150,
    "verbose": 100,
    "random_seed": 42,
}


def build_catboost_model(
    custom_params: dict[str, Any] | None = None,
) -> CatBoostClassifier:
    """Build CatBoostClassifier with default or custom parameters."""
    params = DEFAULT_CATBOOST_PARAMS.copy()
    if custom_params:
        params.update(custom_params)
    return CatBoostClassifier(**params)


def train_catboost_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    custom_params: dict[str, Any] | None = None,
) -> tuple[CatBoostClassifier, Any, int]:
    """Train CatBoost on a single fold, return model, predictions and best iteration."""
    model = build_catboost_model(custom_params=custom_params)

    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
    )

    val_preds = model.predict_proba(X_val)[:, 1]
    best_iteration = model.get_best_iteration()

    return model, val_preds, best_iteration
