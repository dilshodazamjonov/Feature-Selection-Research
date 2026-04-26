import time
import logging

import numpy as np
import pandas as pd

from evaluation.feature_utils import (
    _feature_score_lookup,
    _to_df,
)
from evaluation.stability_scores import calculate_psi, feature_psi
from evaluation.metrics import determine_threshold, evaluate_model_wrapper
from training.cv_utils import _to_1d_proba
from utils.logging_config import setup_logging

# Setup module logger
logger = setup_logging("fold", level=logging.INFO)


def _selected_feature_rows(fold, selector_name, selector, selected_features):
    score_lookup = _feature_score_lookup(selector, selected_features)
    return [
        {
            "fold_id": fold,
            "selector": selector_name,
            "feature_name": str(feature),
            "feature": str(feature),
            "rank": rank,
            "score": score_lookup.get(str(feature), pd.NA),
        }
        for rank, feature in enumerate(list(selected_features or []), start=1)
    ]


def _hybrid_trace_rows(fold, selector_name, selector):
    llm_features = getattr(selector, "llm_selected_features_", None)
    final_features = getattr(selector, "selected_features_", None)
    if not llm_features or final_features is None:
        return []

    final_rank = {str(feature): rank for rank, feature in enumerate(final_features, start=1)}
    return [
        {
            "scope": "fold",
            "fold_id": fold,
            "selector": selector_name,
            "llm_rank": rank,
            "feature_name": str(feature),
            "survived_hybrid": str(feature) in final_rank,
            "hybrid_rank": final_rank.get(str(feature), pd.NA),
        }
        for rank, feature in enumerate(llm_features, start=1)
    ]


def _ranking_utility(y_true, y_score, top_frac=0.1):
    frame = pd.DataFrame({"y_true": np.asarray(y_true), "score": np.asarray(y_score)})
    if frame.empty:
        return {"lift_at_10": np.nan, "bad_rate_capture_at_10": np.nan}
    frame = frame.sort_values("score", ascending=False).reset_index(drop=True)
    n_top = max(1, int(np.ceil(len(frame) * top_frac)))
    overall_bad_rate = float(frame["y_true"].mean())
    total_bads = float(frame["y_true"].sum())
    top = frame.head(n_top)
    top_bad_rate = float(top["y_true"].mean())
    return {
        "lift_at_10": top_bad_rate / overall_bad_rate if overall_bad_rate else np.nan,
        "bad_rate_capture_at_10": float(top["y_true"].sum() / total_bads) if total_bads else np.nan,
    }


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
    selector_name=None,
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

    X_train = X_model.iloc[tr_idx].copy()
    X_val = X_model.iloc[va_idx].copy()
    y_train = y_sorted.iloc[tr_idx].copy()
    y_val = y_sorted.iloc[va_idx].copy()

    fold_start = time.time()
    preprocessing_time_sec = 0.0
    feature_selection_time_sec = 0.0
    training_time_sec = 0.0
    evaluation_time_sec = 0.0

    selection_features = None

    if selector is not None and hasattr(selector, "set_ranking_context"):
        selector.set_ranking_context(
            scope="fold",
            fold_id=fold,
            ranking_artifact_dir=features_dir,
            selector_name=selector_name,
        )

    if selector is not None and getattr(selector, "select_before_preprocessing", False):
        selection_start = time.time()
        if getattr(selector, "apply_post_preprocessing", False):
            selector.fit(X_train, y_train)
            X_train_selected_raw = selector.transform(X_train)
        else:
            X_train_selected_raw = selector.fit_transform(X_train, y_train)
        X_val_selected_raw = selector.transform(X_val)
        feature_selection_time_sec += time.time() - selection_start

        if not isinstance(X_train_selected_raw, pd.DataFrame):
            X_train_selected_raw = _to_df(X_train_selected_raw, index=X_train.index)
        if not isinstance(X_val_selected_raw, pd.DataFrame):
            X_val_selected_raw = _to_df(
                X_val_selected_raw,
                index=X_val.index,
                columns=X_train_selected_raw.columns,
            )

        selection_features = getattr(selector, "llm_selected_features_", None) or getattr(
            selector,
            "selected_features",
            None,
        )

        preprocessing_start = time.time()
        X_train_p = preprocessor.fit_transform(X_train_selected_raw)
        X_val_p = preprocessor.transform(X_val_selected_raw)
        preprocessing_time_sec += time.time() - preprocessing_start

        if not isinstance(X_train_p, pd.DataFrame):
            X_train_p = _to_df(X_train_p, index=X_train_selected_raw.index)
        if not isinstance(X_val_p, pd.DataFrame):
            X_val_p = _to_df(X_val_p, index=X_val_selected_raw.index, columns=X_train_p.columns)

        if getattr(selector, "apply_post_preprocessing", False):
            selection_start = time.time()
            X_train_f = selector.fit_postprocess(X_train_p, y_train)
            X_val_f = selector.transform_postprocess(X_val_p)
            selection_features = getattr(selector, "selected_features_", None) or getattr(
                selector,
                "selected_features",
                None,
            )
            feature_selection_time_sec += time.time() - selection_start
        else:
            X_train_f, X_val_f = X_train_p, X_val_p
    else:
        # Preprocessing
        preprocessing_start = time.time()
        X_train_p = preprocessor.fit_transform(X_train)
        X_val_p = preprocessor.transform(X_val)
        preprocessing_time_sec += time.time() - preprocessing_start

        if not isinstance(X_train_p, pd.DataFrame):
            X_train_p = _to_df(X_train_p, index=X_train.index)
        if not isinstance(X_val_p, pd.DataFrame):
            X_val_p = _to_df(X_val_p, index=X_val.index, columns=X_train_p.columns)

        # Feature selection
        if selector is not None:
            selection_start = time.time()
            X_train_f = selector.fit_transform(X_train_p, y_train)
            X_val_f = selector.transform(X_val_p)
            feature_selection_time_sec += time.time() - selection_start

            if not isinstance(X_train_f, pd.DataFrame):
                X_train_f = _to_df(X_train_f, index=X_train_p.index)
            if not isinstance(X_val_f, pd.DataFrame):
                X_val_f = _to_df(X_val_f, index=X_val_p.index, columns=X_train_f.columns)
        else:
            X_train_f, X_val_f = X_train_p, X_val_p

    # PSI
    evaluation_start = time.time()
    psi_df = feature_psi(X_train_f, X_val_f)
    evaluation_time_sec += time.time() - evaluation_start

    psi_mean = float(psi_df["psi"].mean()) if "psi" in psi_df.columns and len(psi_df) > 0 else np.nan
    psi_max = float(psi_df["psi"].max()) if "psi" in psi_df.columns and len(psi_df) > 0 else np.nan
    logger.info(f"Feature PSI | mean: {psi_mean:.4f}, max: {psi_max:.4f}")

    model_feature_names = X_train_f.columns.tolist()
    selected_features = selection_features or model_feature_names
    if isinstance(selected_features, pd.Index):
        selected_features = selected_features.tolist()
    selected_rows = _selected_feature_rows(fold, selector_name, selector, selected_features)
    hybrid_rows = _hybrid_trace_rows(fold, selector_name, selector)

    # Train model
    model = get_model()
    training_start = time.time()
    model = train_model(model, X_train_f, y_train, X_val_f, y_val)
    training_time_sec += time.time() - training_start

    evaluation_start = time.time()
    val_proba = _to_1d_proba(predict_proba(model, X_val_f))
    train_proba = _to_1d_proba(predict_proba(model, X_train_f))
    decision_threshold = determine_threshold(y_train.values, train_proba)

    try:
        psi_model = calculate_psi(train_proba, val_proba)
    except Exception:
        psi_model = np.nan

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
        threshold=decision_threshold,
    )
    fold_metrics.update(_ranking_utility(y_val.values, val_proba))
    evaluation_time_sec += time.time() - evaluation_start

    fold_metrics.update({
        "fold": fold,
        "train_size": len(tr_idx),
        "val_size": len(va_idx),
        "selected_features": len(selected_features),
        "fold_time_sec": time.time() - fold_start,
        "psi_feature_mean": psi_mean,
        "psi_feature_max": psi_max,
        "psi_model": psi_model,
        "preprocessing_time_sec": preprocessing_time_sec,
        "feature_selection_time_sec": feature_selection_time_sec,
        "training_time_sec": training_time_sec,
        "evaluation_time_sec": evaluation_time_sec,
    })

    if selector is not None and hasattr(selector, "boruta") and hasattr(selector, "rfe"):
        boruta_feats = getattr(selector.boruta, "selected_features", None)
        rfe_feats = getattr(selector.rfe, "selected_features", None)
        fold_metrics["boruta_selected_features"] = len(boruta_feats) if boruta_feats is not None else np.nan
        fold_metrics["rfe_selected_features"] = len(rfe_feats) if rfe_feats is not None else np.nan

    return fold_metrics, val_proba, selected_features, decision_threshold, selected_rows, hybrid_rows
