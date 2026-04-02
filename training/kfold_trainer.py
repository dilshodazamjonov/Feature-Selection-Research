import os
import time
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from Preprocessing.preprocessing import Preprocessor
from evaluation.metrics import evaluate_model_wrapper
from evaluation.feature_utils import (
    _extract_feature_importance, 
    _save_correlation_matrix, 
    _save_feature_statistics, 
    _save_selected_features,
    _save_stagewise_selection,
    _to_df
)
from evaluation.stability_scores import calculate_psi, feature_psi


def _instantiate(cls_or_obj, kwargs):
    """
    Instantiates a class with provided keyword arguments or returns the object if already instantiated.
    """
    return cls_or_obj(**kwargs) if isinstance(cls_or_obj, type) else cls_or_obj


def _fit_selector(selector, X_train, y_train):
    """
    Fits the feature selector to the training data and returns the transformed features,
    handling various scikit-learn API signatures.
    """
    try:
        return selector.fit_transform(X_train, y_train)
    except TypeError:
        try:
            return selector.fit_transform(X_train)
        except TypeError:
            selector.fit(X_train, y_train)
            return selector.transform(X_train)


def _transform_selector(selector, X):
    """
    Applies the transformation of a previously fitted selector to the input data.
    """
    return selector.transform(X)


def run_kfold_training(
    X,
    y,
    time_col,
    get_model,
    train_model,
    predict_proba,
    save_model,
    preprocessor_cls=Preprocessor,
    preprocessor_kwargs=None,
    selector_cls=None,
    selector_kwargs=None,
    model_name="model",
    base_output_dir="outputs",
    n_splits=5,
    random_state=42,
):
    """
    Time-ordered K-fold training on the train portion only.

    Additionally:
    - Saves per-fold metrics
    - Saves selected features per fold
    - Saves per-fold feature statistics
    - Saves per-fold feature importance
    - Saves per-fold correlation matrices
    - Saves feature stability across folds
    - Appends MEAN and STD rows to final CSV
    """
    if time_col not in X.columns:
        raise ValueError(f"{time_col} not found in X")

    preprocessor_kwargs = preprocessor_kwargs or {}
    selector_kwargs = selector_kwargs or {}

    exp_dir = os.path.join(base_output_dir, f"{model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    results_dir = os.path.join(exp_dir, "results")
    features_dir = os.path.join(exp_dir, "features")
    models_dir = os.path.join(exp_dir, "models")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df = X.copy()
    df["_target_"] = y.values
    df = df.sort_values(time_col).reset_index(drop=True)

    y_sorted = df["_target_"].copy()
    X_model = df.drop(columns=["_target_", time_col])

    tss = TimeSeriesSplit(n_splits=n_splits)

    fold_results = []
    feature_sets = []
    feature_counter = Counter()
    oof_pred = np.zeros(len(df), dtype=float)
    oof_true = y_sorted.values.copy()

    start_total = time.time()

    for fold, (tr_idx, va_idx) in enumerate(tss.split(X_model), 1):
        print(f"\n========== FOLD {fold} ==========")

        fold_dir = os.path.join(features_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        psi_dir = os.path.join(fold_dir, "psi")
        os.makedirs(psi_dir, exist_ok=True)

        X_train, X_val = X_model.iloc[tr_idx].copy(), X_model.iloc[va_idx].copy()
        y_train, y_val = y_sorted.iloc[tr_idx].copy(), y_sorted.iloc[va_idx].copy()

        fold_start = time.time()

        # -------------------------
        # Preprocessing
        # -------------------------
        preprocessor = _instantiate(preprocessor_cls, preprocessor_kwargs)
        X_train_p = preprocessor.fit_transform(X_train)
        X_val_p = preprocessor.transform(X_val)

        if not isinstance(X_train_p, pd.DataFrame):
            X_train_p = _to_df(X_train_p, index=X_train.index)
        if not isinstance(X_val_p, pd.DataFrame):
            X_val_p = _to_df(X_val_p, index=X_val.index, columns=X_train_p.columns)

        # -------------------------
        # Feature Selection
        # -------------------------
        selector = None
        if selector_cls is not None:
            selector = _instantiate(selector_cls, selector_kwargs)

            X_train_f = _fit_selector(selector, X_train_p, y_train)
            X_val_f = _transform_selector(selector, X_val_p)

            X_train_f = _to_df(X_train_f, index=X_train_p.index)
            X_val_f = _to_df(X_val_f, index=X_val_p.index, columns=X_train_f.columns)
        else:
            X_train_f, X_val_f = X_train_p, X_val_p

        psi_df = feature_psi(X_train_f, X_val_f)
        psi_df.to_csv(os.path.join(psi_dir, "feature_psi.csv"), index=False)

        psi_mean = psi_df["psi"].mean()
        psi_max = psi_df["psi"].max()

        print(f"Feature PSI | mean: {psi_mean:.4f}, max: {psi_max:.4f}")

        selected_features = X_train_f.columns.tolist()
        feature_sets.append(selected_features)
        feature_counter.update(selected_features)

        # Save final selected features for this fold
        _save_selected_features(
            os.path.join(fold_dir, "selected_features.csv"),
            selected_features
        )

        # Save intermediate stage features if available
        _save_stagewise_selection(selector, fold_dir)

        # Save feature statistics
        _save_feature_statistics(
            os.path.join(fold_dir, "feature_statistics.csv"),
            X_train_f
        )

        # Save correlation matrix for selected numeric features
        _save_correlation_matrix(
            os.path.join(fold_dir, "feature_correlation.csv"),
            X_train_f
        )

        # -------------------------
        # Model Training
        # -------------------------
        model = get_model()
        model = train_model(model, X_train_f, y_train, X_val_f, y_val)

        save_model(model, os.path.join(models_dir, f"{model_name}_fold_{fold}.model"))

        # Save model-based feature importance
        importance_df = _extract_feature_importance(model, selected_features)
        if importance_df is not None:
            importance_df.to_csv(
                os.path.join(fold_dir, "feature_importance.csv"),
                index=False
            )

        # -------------------------
        # Validation
        # -------------------------
        val_proba = predict_proba(model, X_val_f)
        train_proba = predict_proba(model, X_train_f)

        psi_model = calculate_psi(train_proba, val_proba)

        pd.DataFrame({
            "fold": [fold],
            "model_psi": [psi_model]
        }).to_csv(os.path.join(psi_dir, "model_psi.csv"), index=False)

        print(f"Model PSI: {psi_model:.4f}")

        oof_pred[va_idx] = val_proba

        fold_metrics = evaluate_model_wrapper(
            y_true=y_val.values,
            y_pred_proba=val_proba,
            output_dir=results_dir,
            method_name=f"{model_name}_fold_{fold}",
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

        # Add selector-stage counts if available
        if selector is not None and hasattr(selector, "boruta") and hasattr(selector, "rfe"):
            boruta_feats = getattr(selector.boruta, "selected_features", None)
            rfe_feats = getattr(selector.rfe, "selected_features", None)
            fold_metrics["boruta_selected_features"] = len(boruta_feats) if boruta_feats is not None else np.nan
            fold_metrics["rfe_selected_features"] = len(rfe_feats) if rfe_feats is not None else np.nan

        fold_results.append(fold_metrics)
        print(fold_metrics)

    # FINAL RESULTS + MEAN / STD
    results_df = pd.DataFrame(fold_results)

    numeric_cols = results_df.select_dtypes(include=[np.number]).columns

    mean_row = results_df[numeric_cols].mean().to_frame().T
    std_row = results_df[numeric_cols].std().to_frame().T

    mean_row["fold"] = "mean"
    std_row["fold"] = "std"

    summary_df = pd.concat([mean_row, std_row], ignore_index=True)
    summary_df = summary_df.reindex(columns=results_df.columns)

    final_results_df = pd.concat([results_df, summary_df], ignore_index=True)

    final_results_df.to_csv(os.path.join(results_dir, "cv_results.csv"), index=False)

    # Save feature stability across folds
    feature_stability_df = (
        pd.DataFrame(
            {
                "feature": list(feature_counter.keys()),
                "selected_in_folds": list(feature_counter.values()),
            }
        )
        .sort_values(["selected_in_folds", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    feature_stability_df.to_csv(os.path.join(exp_dir, "feature_stability.csv"), index=False)

    # OOF Evaluation
    _ = evaluate_model_wrapper(
        y_true=oof_true,
        y_pred_proba=oof_pred,
        output_dir=results_dir,
        feature_sets=feature_sets,
        metrics_list=fold_results,
        method_name=model_name,
    )

    print("\n========== FINAL RESULTS ==========")
    print(final_results_df)
    print(f"\nSaved to {os.path.join(results_dir, 'cv_results.csv')}")
    print(f"Feature stability saved to {os.path.join(exp_dir, 'feature_stability.csv')}")
    print(f"Total Time: {(time.time() - start_total) / 60:.2f} minutes")

    return final_results_df