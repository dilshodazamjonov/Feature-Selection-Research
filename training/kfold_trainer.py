import os
import time
from collections import Counter
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from Preprocessing.preprocessing import Preprocessor
from evaluation.metrics import (
    evaluate_model,
    evaluate_model_wrapper
)

from training.fold import process_fold
from utils.logging_config import setup_logging

# Setup module logger
logger = setup_logging("kfold_trainer", level=logging.INFO)


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

    # Use TimeSeriesSplit for expanding window CV
    # Each fold trains on all previous data and validates on next chunk
    tss = TimeSeriesSplit(n_splits=n_splits)

    fold_results = []
    feature_counter = Counter()
    oof_pred = np.zeros(len(df), dtype=float)
    oof_mask = np.zeros(len(df), dtype=bool)
    oof_true = y_sorted.values.copy()
    prev_selected_features = None  # For Jaccard similarity tracking

    # Track time periods for each fold (for later analysis)
    fold_time_info = []

    start_total = time.time()

    for fold, (tr_idx, va_idx) in enumerate(tss.split(X_model), 1):
        # Track time periods for this fold
        time_values = df[time_col].values
        fold_time_info.append({
            "fold": fold,
            "train_start_idx": tr_idx[0],
            "train_end_idx": tr_idx[-1],
            "val_start_idx": va_idx[0],
            "val_end_idx": va_idx[-1],
            "train_time_start": time_values[tr_idx[0]],
            "train_time_end": time_values[tr_idx[-1]],
            "val_time_start": time_values[va_idx[0]],
            "val_time_end": time_values[va_idx[-1]],
        })
        # Instantiate preprocessor and selector for this fold
        preprocessor = preprocessor_cls(**preprocessor_kwargs)
        selector = selector_cls(**selector_kwargs) if selector_cls is not None else None

        fold_metrics, val_proba, selected_features = process_fold(
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
            prev_selected_features=prev_selected_features,
        )

        oof_pred[va_idx] = val_proba
        oof_mask[va_idx] = True
        fold_results.append(fold_metrics)
        
        # Track feature selection frequency for stability analysis
        if selected_features:
            for feat in selected_features:
                feature_counter[feat] += 1
            # Update for next iteration's Jaccard calculation
            prev_selected_features = set(selected_features)
        
        # ========== Log fold results ==========
        time_info = fold_time_info[-1]
        train_size = fold_metrics.get("train_size", len(tr_idx))
        val_size = fold_metrics.get("val_size", len(va_idx))
        
        # Format values
        auc = fold_metrics.get("auc", np.nan)
        gini = fold_metrics.get("gini", np.nan)
        ks = fold_metrics.get("ks", np.nan)
        precision = fold_metrics.get("precision", np.nan)
        recall = fold_metrics.get("recall", np.nan)
        f1 = fold_metrics.get("f1", np.nan)
        psi_mean = fold_metrics.get("psi_feature_mean", np.nan)
        psi_max = fold_metrics.get("psi_feature_max", np.nan)
        jaccard = fold_metrics.get("jaccard_similarity", np.nan)
        n_features = fold_metrics.get("selected_features", np.nan)
        fold_time = fold_metrics.get("fold_time_sec", np.nan)
        
        # One line for metrics: fold | train_size | val_size | auc | gini | ks | precision | recall | f1 | time
        metrics_line = (
            f"Fold {fold} | "
            f"train={train_size:,} | "
            f"val={val_size:,} | "
            f"auc={auc:.4f} | "
            f"gini={gini:.4f} | "
            f"ks={ks:.4f} | "
            f"prec={precision:.4f} | "
            f"rec={recall:.4f} | "
            f"f1={f1:.4f} | "
            f"time={fold_time:.1f}s"
        )
        logger.info(metrics_line)
        
        # One line for stability: fold | features | psi_mean | psi_max | jaccard | time_range
        stability_line = (
            f"Fold {fold} | "
            f"features={int(n_features) if not np.isnan(n_features) else 'N/A'} | "
            f"psi_mean={psi_mean:.4f} | " if not np.isnan(psi_mean) else f"features={int(n_features) if not np.isnan(n_features) else 'N/A'} | psi_mean=N/A | "
            f"psi_max={psi_max:.4f} | " if not np.isnan(psi_max) else f"psi_max=N/A | "
            f"jaccard={jaccard:.4f} | " if not np.isnan(jaccard) else f"jaccard=N/A | "
            f"train_time=[{int(time_info['train_time_start'])},{int(time_info['train_time_end'])}] | "
            f"val_time=[{int(time_info['val_time_start'])},{int(time_info['val_time_end'])}]"
        )
        logger.info(stability_line)

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
        "train_size": int(oof_mask.sum()),
        "val_size": int(oof_mask.sum()),
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

    # Save fold time information (for ROC-AUC over time analysis)
    fold_time_df = pd.DataFrame(fold_time_info)
    fold_time_df.to_csv(os.path.join(results_dir, "fold_time_info.csv"), index=False)

    # Summary files - each file contains ONLY its unique columns (no duplicates)
    
    # 1. Confusion Matrix: only confusion metrics
    confusion_summary_df = pd.DataFrame([
        {
            "fold": row["fold"],
            "tn": row.get("tn", np.nan),
            "fp": row.get("fp", np.nan),
            "fn": row.get("fn", np.nan),
            "tp": row.get("tp", np.nan),
        }
        for row in fold_results
    ] + [oof_confusion])
    
    # 2. Evaluation Metrics: performance metrics + sample sizes
    eval_summary_df = pd.DataFrame([
        {
            "fold": row["fold"],
            "auc": row.get("auc", np.nan),
            "gini": row.get("gini", np.nan),
            "ks": row.get("ks", np.nan),
            "ks_threshold": row.get("ks_threshold", np.nan),
            "precision": row.get("precision", np.nan),
            "recall": row.get("recall", np.nan),
            "f1": row.get("f1", np.nan),
            "accuracy": row.get("accuracy", np.nan),
            "approval_rate": row.get("approval_rate", np.nan),
            "bad_rate_approved": row.get("bad_rate_approved", np.nan),
            "train_size": row.get("train_size", np.nan),
            "val_size": row.get("val_size", np.nan),
        }
        for row in fold_results
    ] + [oof_eval])
    
    # 3. Stability Metrics: only stability-related metrics
    stability_summary_df = pd.DataFrame([
        {
            "fold": row["fold"],
            "selected_features": row.get("selected_features", np.nan),
            "psi_feature_mean": row.get("psi_feature_mean", np.nan),
            "psi_feature_max": row.get("psi_feature_max", np.nan),
            "psi_model": row.get("psi_model", np.nan),
            "jaccard_similarity": row.get("jaccard_similarity", np.nan),
        }
        for row in fold_results
    ] + [oof_stability])
    
    # 4. Other Metrics: only size/timing/selection info
    other_summary_df = pd.DataFrame([
        {
            "fold": row["fold"],
            "train_size": row.get("train_size", np.nan),
            "val_size": row.get("val_size", np.nan),
            "fold_time_sec": row.get("fold_time_sec", np.nan),
            "boruta_selected_features": row.get("boruta_selected_features", np.nan),
            "rfe_selected_features": row.get("rfe_selected_features", np.nan),
        }
        for row in fold_results
    ] + [oof_other])

    # Add mean/std rows to each summary file
    def add_mean_std(df):
        """Add mean and std rows to a summary DataFrame."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return df
        
        mean_row = df[numeric_cols].mean().to_frame().T
        mean_row["fold"] = "mean"
        std_row = df[numeric_cols].std().to_frame().T
        std_row["fold"] = "std"
        
        # Ensure all columns are present
        for col in df.columns:
            if col not in mean_row.columns:
                mean_row[col] = np.nan
            if col not in std_row.columns:
                std_row[col] = np.nan
        
        return pd.concat([df, mean_row, std_row], ignore_index=True)
    
    confusion_summary_df = add_mean_std(confusion_summary_df)
    eval_summary_df = add_mean_std(eval_summary_df)
    stability_summary_df = add_mean_std(stability_summary_df)
    other_summary_df = add_mean_std(other_summary_df)

    confusion_summary_df.to_csv(os.path.join(results_dir, "confusion_matrix_summary.csv"), index=False)
    eval_summary_df.to_csv(os.path.join(results_dir, "evaluation_metrics_summary.csv"), index=False)
    stability_summary_df.to_csv(os.path.join(results_dir, "stability_metrics_summary.csv"), index=False)
    other_summary_df.to_csv(os.path.join(results_dir, "other_metrics_summary.csv"), index=False)

    # OOF evaluation - compute metrics but don't save (already computed above)
    evaluate_model_wrapper(
        y_true=oof_true[oof_mask],
        y_pred_proba=oof_pred[oof_mask],
        fold_number="oof",
        selected_features=None,
        psi_feature_mean=None,
        psi_feature_max=None,
        psi_model=None,
    )

    # ========== Log OOF (Out-of-Fold) results ==========
    logger.info("")
    logger.info(f"OOF | samples={int(oof_mask.sum()):,} | auc={oof_metrics_safe['auc']:.4f} | gini={oof_metrics_safe['gini']:.4f} | ks={oof_metrics_safe['ks']:.4f} | prec={oof_metrics_safe['precision']:.4f} | rec={oof_metrics_safe['recall']:.4f} | f1={oof_metrics_safe['f1']:.4f} | acc={oof_metrics_safe['accuracy']:.4f}")

    logger.info("")
    logger.info(f"CV Complete | results={os.path.join(results_dir, 'cv_results.csv')} | stability={os.path.join(exp_dir, 'feature_stability.csv')} | total_time={(time.time() - start_total) / 60:.2f}min")

    return final_results_df