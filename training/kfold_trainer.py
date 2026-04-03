import os
import time
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from Preprocessing.preprocessing import Preprocessor
from evaluation.metrics import (
    evaluate_model,
    evaluate_model_wrapper
)

from training.fold import process_fold


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
    Run a time-series aware K-Fold training pipeline with preprocessing, feature selection, 
    model training, evaluation, and saving of results.

    This function performs the following steps:
    1. Sorts the data by a specified time column.
    2. Splits the data using TimeSeriesSplit for temporal cross-validation.
    3. For each fold:
        - Fits a preprocessor and optional feature selector on training data.
        - Transforms training and validation sets.
        - Calculates feature stability, correlation, and PSI (Population Stability Index).
        - Trains the specified model and saves it.
        - Extracts feature importance and evaluates fold metrics.
    4. Aggregates results across folds and computes out-of-fold (OOF) metrics.
    5. Saves detailed summary files:
        - CV results
        - Feature stability
        - Confusion matrices
        - Evaluation metrics
        - Stability metrics
        - Other fold metrics

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix, must include `time_col`.
    y : pd.Series or np.ndarray
        Target variable.
    time_col : str
        Name of the column in `X` used for temporal ordering.
    get_model : callable
        Function that returns a new instance of the model to train.
    train_model : callable
        Function to train the model. Signature: train_model(model, X_train, y_train, X_val, y_val)
    predict_proba : callable
        Function to generate predicted probabilities. Signature: predict_proba(model, X)
    save_model : callable
        Function to save a trained model. Signature: save_model(model, path)
    preprocessor_cls : class, optional
        Preprocessor class to instantiate for each fold (default: Preprocessor)
    preprocessor_kwargs : dict, optional
        Keyword arguments to pass to the preprocessor class.
    selector_cls : class, optional
        Feature selector class to instantiate for each fold (default: None)
    selector_kwargs : dict, optional
        Keyword arguments to pass to the feature selector class.
    model_name : str, optional
        Base name for the model and experiment folder (default: "model")
    base_output_dir : str, optional
        Root directory to save experiment outputs (default: "outputs")
    n_splits : int, optional
        Number of time-series CV folds (default: 5)
    random_state : int, optional
        Random seed for reproducibility (default: 42)

    Returns
    -------
    pd.DataFrame
        DataFrame containing detailed CV fold results, including evaluation metrics, 
        selected feature counts, PSI, and fold timing.

    Notes
    -----
    - The function uses TimeSeriesSplit to respect temporal order in the data.
    - Feature stability and PSI computations require that `process_fold` returns proper metrics.
    - All intermediate outputs (models, feature statistics, PSI, etc.) are saved under 
      `base_output_dir/<model_name>_<timestamp>/`.
    """
    if time_col not in X.columns:
        raise ValueError(f"{time_col} not found in X")

    preprocessor_kwargs = preprocessor_kwargs or {}
    selector_kwargs = selector_kwargs or {}

    exp_dir = os.path.join(
        base_output_dir,
        f"{model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
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
    feature_counter = Counter()
    oof_pred = np.zeros(len(df), dtype=float)
    oof_mask = np.zeros(len(df), dtype=bool)
    oof_true = y_sorted.values.copy()

    start_total = time.time()

    for fold, (tr_idx, va_idx) in enumerate(tss.split(X_model), 1):
        # Instantiate preprocessor and selector for this fold
        preprocessor = preprocessor_cls(**preprocessor_kwargs)
        selector = selector_cls(**selector_kwargs) if selector_cls is not None else None

        fold_metrics, val_proba = process_fold(
            fold=fold,
            tr_idx=tr_idx,
            va_idx=va_idx,
            X_model=X_model,
            y_sorted=y_sorted,
            features_dir=features_dir,
            models_dir=models_dir,
            get_model=get_model,
            train_model=train_model,
            predict_proba=predict_proba,
            save_model=save_model,
            preprocessor=preprocessor,
            selector=selector,
        )

        oof_pred[va_idx] = val_proba
        oof_mask[va_idx] = True
        fold_results.append(fold_metrics)
        print(fold_metrics)

    # OOF metrics
    oof_metrics_safe = evaluate_model(oof_true[oof_mask], oof_pred[oof_mask])

    oof_confusion = {
        "fold": "oof",
        "tn": oof_metrics_safe["tn"],
        "fp": oof_metrics_safe["fp"],
        "fn": oof_metrics_safe["fn"],
        "tp": oof_metrics_safe["tp"],
    }

    oof_eval = {
        "fold": "oof",
        "gini": oof_metrics_safe["gini"],
        "auc": oof_metrics_safe["auc"],
        "ks": oof_metrics_safe["ks"],
        "ks_threshold": oof_metrics_safe["ks_threshold"],
        "precision": oof_metrics_safe["precision"],
        "recall": oof_metrics_safe["recall"],
        "f1": oof_metrics_safe["f1"],
        "accuracy": oof_metrics_safe["accuracy"],
        "approval_rate": oof_metrics_safe["approval_rate"],
        "bad_rate_approved": oof_metrics_safe["bad_rate_approved"],
    }

    oof_stability = {
        "fold": "oof",
        "selected_features": np.nan,
        "psi_feature_mean": np.nan,
        "psi_feature_max": np.nan,
        "psi_model": np.nan,
        "jaccard_similarity": np.nan,
    }

    oof_other = {
        "fold": "oof",
        "train_size": int(oof_mask.sum()),
        "val_size": int(oof_mask.sum()),
        "fold_time_sec": time.time() - start_total,
        "boruta_selected_features": np.nan,
        "rfe_selected_features": np.nan,
    }

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

    # Feature stability
    feature_stability_df = pd.DataFrame({
        "feature": list(feature_counter.keys()),
        "selected_in_folds": list(feature_counter.values()),
    }).sort_values(["selected_in_folds", "feature"], ascending=[False, True]).reset_index(drop=True)
    feature_stability_df.to_csv(os.path.join(exp_dir, "feature_stability.csv"), index=False)

    # Summary files
    confusion_summary_df = pd.DataFrame(
        [{"fold": row["fold"], "tn": row["tn"], "fp": row["fp"], "fn": row["fn"], "tp": row["tp"]} for row in fold_results]
        + [oof_confusion]
    )
    eval_summary_df = pd.DataFrame(
        [dict(row, **{}) for row in fold_results] + [oof_eval]
    )
    stability_summary_df = pd.DataFrame(
        [dict(row, **{}) for row in fold_results] + [oof_stability]
    )
    other_summary_df = pd.DataFrame(
        [dict(row, **{}) for row in fold_results] + [oof_other]
    )

    confusion_summary_df.to_csv(os.path.join(results_dir, "confusion_matrix_summary.csv"), index=False)
    eval_summary_df.to_csv(os.path.join(results_dir, "evaluation_metrics_summary.csv"), index=False)
    stability_summary_df.to_csv(os.path.join(results_dir, "stability_metrics_summary.csv"), index=False)
    other_summary_df.to_csv(os.path.join(results_dir, "other_metrics_summary.csv"), index=False)

    _ = evaluate_model_wrapper(
        y_true=oof_true[oof_mask],
        y_pred_proba=oof_pred[oof_mask],
        fold_number="oof",
        selected_features=None,
        psi_feature_mean=None,
        psi_feature_max=None,
        psi_model=None,
    )

    print("\n========== FINAL RESULTS ==========")
    print(final_results_df)
    print(f"\nSaved to {os.path.join(results_dir, 'cv_results.csv')}")
    print(f"Feature stability saved to {os.path.join(exp_dir, 'feature_stability.csv')}")
    print(f"Saved summary files to {results_dir}")
    print(f"Total Time: {(time.time() - start_total) / 60:.2f} minutes")

    return final_results_df