import os
import random
import time
from datetime import datetime
import logging

import numpy as np
import pandas as pd

from Preprocessing.preprocessing import Preprocessor
from evaluation.metrics import (
    evaluate_model,
    evaluate_model_wrapper
)
from evaluation.feature_stability import (
    write_feature_stability_artifacts,
)

from training.cv_utils import GroupedTimeSeriesSplit
from training.fold import process_fold
from utils.logging_config import setup_logging

# Setup module logger
logger = setup_logging("kfold_trainer", level=logging.INFO)


def _build_stability_confidence_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    tracked_metrics = [
        "gini",
        "ks",
        "psi_feature_mean",
        "psi_feature_max",
        "psi_model",
        "jaccard_similarity",
    ]

    rows: list[dict[str, float | int | str]] = []
    z_value = 1.96

    for metric in tracked_metrics:
        if metric not in results_df.columns:
            continue

        numeric = pd.to_numeric(results_df[metric], errors="coerce").dropna()
        n_valid = int(numeric.shape[0])
        if n_valid == 0:
            continue

        mean_value = float(numeric.mean())
        std_value = float(numeric.std(ddof=1)) if n_valid >= 2 else 0.0
        stderr_value = std_value / np.sqrt(n_valid) if n_valid >= 2 else 0.0
        margin = z_value * stderr_value

        rows.append(
            {
                "metric": metric,
                "value": mean_value,
                "ci95_lower": float(mean_value - margin),
                "ci95_upper": float(mean_value + margin),
            }
        )

    return pd.DataFrame(rows)


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
    gap_groups=1,
    experiment_output_dir=None,
    selector_name=None,
    excluded_feature_columns=None,
    feature_budget=None,
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
    excluded_feature_columns = set(excluded_feature_columns or ())

    random.seed(random_state)
    np.random.seed(random_state)

    if experiment_output_dir is not None:
        exp_dir = str(experiment_output_dir)
    else:
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
    X_model = df.drop(
        columns=[
            col
            for col in {"_target_", time_col, *excluded_feature_columns}
            if col in df.columns
        ]
    )

    splitter = GroupedTimeSeriesSplit(n_splits=n_splits, gap=gap_groups)

    fold_results = []
    oof_pred = np.zeros(len(df), dtype=float)
    oof_pred_label = np.zeros(len(df), dtype=int)
    oof_mask = np.zeros(len(df), dtype=bool)
    oof_true = y_sorted.values.copy()
    prev_selected_features = None  # For Jaccard similarity tracking

    # Track time periods for each fold (for later analysis)
    fold_time_info = []
    fold_selected_rows = []
    hybrid_trace_rows = []

    start_total = time.time()

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(df[time_col].values), 1):
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

        (
            fold_metrics,
            val_proba,
            selected_features,
            decision_threshold,
            selected_rows,
            hybrid_rows,
        ) = process_fold(
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
            selector_name=selector_name,
            prev_selected_features=prev_selected_features,
        )
        fold_metrics.update(fold_time_info[-1])

        oof_pred[va_idx] = val_proba
        oof_pred_label[va_idx] = (val_proba >= decision_threshold).astype(int)
        oof_mask[va_idx] = True
        fold_results.append(fold_metrics)
        fold_selected_rows.extend(selected_rows)
        hybrid_trace_rows.extend(hybrid_rows)
        
        # Track feature selection frequency for stability analysis
        if selected_features:
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
        decision_threshold_metric = fold_metrics.get("decision_threshold", np.nan)
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
            f"thr={decision_threshold_metric:.4f} | "
            f"time={fold_time:.1f}s"
        )
        logger.info(metrics_line)
        
        stability_line = (
            f"Fold {fold} | "
            f"features={int(n_features) if not np.isnan(n_features) else 'N/A'} | "
            f"psi_mean={psi_mean:.4f} | "
            f"psi_max={psi_max:.4f} | "
            f"jaccard={jaccard:.4f} | "
            f"train_time=[{int(time_info['train_time_start'])},{int(time_info['train_time_end'])}] | "
            f"val_time=[{int(time_info['val_time_start'])},{int(time_info['val_time_end'])}]"
        )
        logger.info(stability_line)

    # OOF metrics
    oof_metrics_safe = evaluate_model(
        oof_true[oof_mask],
        oof_pred[oof_mask],
        y_pred=oof_pred_label[oof_mask],
    )

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
    if fold_selected_rows:
        pd.DataFrame(fold_selected_rows).to_csv(
            os.path.join(features_dir, "fold_selected_features.csv"),
            index=False,
        )
    if hybrid_trace_rows:
        pd.DataFrame(hybrid_trace_rows).to_csv(
            os.path.join(features_dir, "llm_hybrid_trace.csv"),
            index=False,
        )

    total_candidate_features = int(X_model.shape[1])
    write_feature_stability_artifacts(
        exp_dir=exp_dir,
        model=model_name.split("_", 1)[0],
        selector=selector_name or model_name,
        total_candidate_features=total_candidate_features,
    )

    # OOF evaluation - compute metrics but don't save (already computed above)
    evaluate_model_wrapper(
        y_true=oof_true[oof_mask],
        y_pred_proba=oof_pred[oof_mask],
        fold_number="oof",
        selected_features=None,
        psi_feature_mean=None,
        psi_feature_max=None,
        psi_model=None,
        y_pred=oof_pred_label[oof_mask],
    )

    # ========== Log OOF (Out-of-Fold) results ==========
    logger.info("")
    logger.info(f"OOF | samples={int(oof_mask.sum()):,} | auc={oof_metrics_safe['auc']:.4f} | gini={oof_metrics_safe['gini']:.4f} | ks={oof_metrics_safe['ks']:.4f} | prec={oof_metrics_safe['precision']:.4f} | rec={oof_metrics_safe['recall']:.4f} | f1={oof_metrics_safe['f1']:.4f} | acc={oof_metrics_safe['accuracy']:.4f}")

    logger.info("")
    logger.info(f"CV Complete | results={os.path.join(results_dir, 'cv_results.csv')} | stability={os.path.join(features_dir, 'feature_stability_metrics.csv')} | total_time={(time.time() - start_total) / 60:.2f}min")

    final_results_df.attrs["exp_dir"] = exp_dir
    final_results_df.attrs["cv_runtime_seconds"] = time.time() - start_total
    for metric in [
        "preprocessing_time_sec",
        "feature_selection_time_sec",
        "training_time_sec",
        "evaluation_time_sec",
    ]:
        if metric in results_df.columns:
            final_results_df.attrs[f"cv_{metric}"] = float(
                pd.to_numeric(results_df[metric], errors="coerce").sum()
            )
    return final_results_df
