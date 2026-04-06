# fold.py
import os
import time
import logging

import numpy as np
import pandas as pd

from evaluation.feature_utils import (
    _extract_feature_importance,
    _save_correlation_matrix,
    _save_feature_statistics,
    _save_selected_features,
    _save_stagewise_selection,
    _to_df,
)
from evaluation.stability_scores import calculate_psi, feature_psi
from evaluation.metrics import evaluate_model_wrapper
from training.cv_utils import _to_1d_proba
from utils.logging_config import setup_logging

# Setup module logger
logger = setup_logging("fold", level=logging.INFO)


def process_fold(
    fold,
    tr_idx,
    va_idx,
    X_model,
    y_sorted,
    features_dir,
    models_dir,
    get_model,
    train_model,
    predict_proba,
    save_model,
    preprocessor,
    selector=None,
    prev_selected_features=None,
):  
    """
    Process a single fold of time-series cross-validation.

    This function handles all steps for a single fold:
    1. Creates fold-specific directories for features, PSI, and model outputs.
    2. Splits the data into training and validation sets based on indices.
    3. Applies preprocessing to the training and validation features.
    4. Optionally applies feature selection.
    5. Calculates feature-level PSI (Population Stability Index) and saves results.
    6. Saves selected features, feature statistics, and correlation matrix.
    7. Trains the specified model and saves it to disk.
    8. Extracts feature importance and saves it.
    9. Evaluates the model on the validation set.
    10. Computes model-level PSI and updates fold metrics.

    Parameters
    ----------
    fold : int
        Fold number (1-indexed) for logging and directory naming.
    tr_idx : array-like
        Indices of training samples for this fold.
    va_idx : array-like
        Indices of validation samples for this fold.
    X_model : pd.DataFrame
        Full feature matrix (without target) sorted by time.
    y_sorted : pd.Series
        Target values aligned with X_model.
    features_dir : str
        Path to the directory where fold-specific feature outputs will be saved.
    models_dir : str
        Path to the directory where trained models will be saved.
    get_model : callable
        Function that returns a new model instance.
    train_model : callable
        Function to train the model. Signature: train_model(model, X_train, y_train, X_val, y_val)
    predict_proba : callable
        Function to predict probabilities. Signature: predict_proba(model, X)
    save_model : callable
        Function to save a trained model. Signature: save_model(model, path)
    preprocessor : object
        Instantiated preprocessor object with `fit_transform` and `transform` methods.
    selector : object, optional
        Instantiated feature selector with `fit_transform` and `transform` methods (default: None).

    Returns
    -------
    fold_metrics : dict
        Dictionary containing evaluation metrics, feature selection statistics, PSI values, 
        fold size, and fold timing.
    val_proba : np.ndarray
        Predicted probabilities for the validation set of this fold.

    Notes
    -----
    - Uses _to_df to ensure that preprocessed/selected data remains a DataFrame.
    - PSI metrics (feature PSI and model PSI) are calculated and saved per fold.
    - Supports Boruta and RFE selectors for fold-level feature selection metrics.
    - Saves all fold outputs in a structured directory:
        features_dir/fold_<fold>/ and models_dir/model_fold_<fold>.model
    """

    logger.info(f"=== FOLD {fold} ===")

    fold_dir = os.path.join(features_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    psi_dir = os.path.join(fold_dir, "psi")
    os.makedirs(psi_dir, exist_ok=True)

    X_train = X_model.iloc[tr_idx].copy()
    X_val = X_model.iloc[va_idx].copy()
    y_train = y_sorted.iloc[tr_idx].copy()
    y_val = y_sorted.iloc[va_idx].copy()

    fold_start = time.time()

    # Preprocessing
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)

    if not isinstance(X_train_p, pd.DataFrame):
        X_train_p = _to_df(X_train_p, index=X_train.index)
    if not isinstance(X_val_p, pd.DataFrame):
        X_val_p = _to_df(X_val_p, index=X_val.index, columns=X_train_p.columns)

    # Feature selection
    if selector is not None:
        X_train_f = selector.fit_transform(X_train_p, y_train)
        X_val_f = selector.transform(X_val_p)

        if not isinstance(X_train_f, pd.DataFrame):
            X_train_f = _to_df(X_train_f, index=X_train_p.index)
        if not isinstance(X_val_f, pd.DataFrame):
            X_val_f = _to_df(X_val_f, index=X_val_p.index, columns=X_train_f.columns)
    else:
        X_train_f, X_val_f = X_train_p, X_val_p

    # PSI
    psi_df = feature_psi(X_train_f, X_val_f)
    psi_df.to_csv(os.path.join(psi_dir, "feature_psi.csv"), index=False)

    psi_mean = float(psi_df["psi"].mean()) if "psi" in psi_df.columns and len(psi_df) > 0 else np.nan
    psi_max = float(psi_df["psi"].max()) if "psi" in psi_df.columns and len(psi_df) > 0 else np.nan
    logger.info(f"Feature PSI | mean: {psi_mean:.4f}, max: {psi_max:.4f}")

    selected_features = X_train_f.columns.tolist()
    _save_selected_features(os.path.join(fold_dir, "selected_features.csv"), selected_features)
    if selector is not None:
        _save_stagewise_selection(selector, fold_dir)
    _save_feature_statistics(os.path.join(fold_dir, "feature_statistics.csv"), X_train_f)
    _save_correlation_matrix(os.path.join(fold_dir, "feature_correlation.csv"), X_train_f)

    # Train model
    model = get_model()
    model = train_model(model, X_train_f, y_train, X_val_f, y_val)
    save_model(model, os.path.join(models_dir, f"model_fold_{fold}.model"))

    importance_df = _extract_feature_importance(model, selected_features)
    if importance_df is not None:
        importance_df.to_csv(os.path.join(fold_dir, "feature_importance.csv"), index=False)

    val_proba = _to_1d_proba(predict_proba(model, X_val_f))
    train_proba = _to_1d_proba(predict_proba(model, X_train_f))

    try:
        psi_model = calculate_psi(train_proba, val_proba)
    except Exception:
        psi_model = np.nan

    pd.DataFrame({"fold": [fold], "model_psi": [psi_model]}).to_csv(
        os.path.join(psi_dir, "model_psi.csv"),
        index=False,
    )

    logger.info(f"Model PSI: {psi_model:.4f}" if np.isfinite(psi_model) else "Model PSI: nan")

    # Evaluate
    fold_metrics = evaluate_model_wrapper(
        y_true=y_val.values,
        y_pred_proba=val_proba,
        fold_number=fold,
        selected_features=selected_features,
        psi_feature_mean=psi_mean,
        psi_feature_max=psi_max,
        psi_model=psi_model,
        prev_selected_features=prev_selected_features,
    )

    fold_metrics.update({
        "fold": fold,
        "train_size": len(tr_idx),
        "val_size": len(va_idx),
        "selected_features": len(selected_features),
        "fold_time_sec": time.time() - fold_start,
        "psi_feature_mean": psi_mean,
        "psi_feature_max": psi_max,
        "psi_model": psi_model,
    })

    if selector is not None and hasattr(selector, "boruta") and hasattr(selector, "rfe"):
        boruta_feats = getattr(selector.boruta, "selected_features", None)
        rfe_feats = getattr(selector.rfe, "selected_features", None)
        fold_metrics["boruta_selected_features"] = len(boruta_feats) if boruta_feats is not None else np.nan
        fold_metrics["rfe_selected_features"] = len(rfe_feats) if rfe_feats is not None else np.nan

    return fold_metrics, val_proba, selected_features