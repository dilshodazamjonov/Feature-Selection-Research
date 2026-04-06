import logging

import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    confusion_matrix, 
    roc_curve,
    accuracy_score
)
import pandas as pd
from evaluation.stability_scores import jaccard_similarity
from utils.logging_config import setup_logging

# Setup module logger
logger = setup_logging("metrics", level=logging.INFO)


def ks_score(y_true, y_prob):
    """
    Calculates the Kolmogorov-Smirnov (KS) statistic for binary classification.
    
    KS measures the maximum separation between cumulative TPR and FPR.
    Returns:
        tuple: (KS statistic, threshold at KS)
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    diffs = tpr - fpr
    ks = np.max(diffs)
    ks_idx = np.argmax(diffs)
    return ks, thresholds[ks_idx]

def gini_score(y_true, y_pred_proba):
    """
    Calculates the Gini coefficient based on ROC AUC.
    Gini = 2*AUC - 1
    """
    return 2 * roc_auc_score(y_true, y_pred_proba) - 1

def precision_score(y_true, y_pred):
    """
    Precision = TP / (TP + FP)
    Returns 0 if TP + FP = 0
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp) if tp + fp > 0 else 0

def recall_score(y_true, y_pred):
    """
    Recall = TP / (TP + FN)
    Returns 0 if TP + FN = 0
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn) if tp + fn > 0 else 0

def f1_score(y_true, y_pred):
    """
    F1 = 2 * (precision * recall) / (precision + recall)
    Returns 0 if precision + recall = 0
    """
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r) if p + r > 0 else 0

def evaluate_model(y_true, y_pred_proba, threshold=None):
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    ks, ks_thresh = ks_score(y_true, y_pred_proba)
    
    if threshold is None:
        threshold = ks_thresh
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    approval_rate = float(np.mean(y_pred == 0))
    approved_mask = (y_pred == 0)
    bad_rate_approved = float(np.mean(y_true[approved_mask])) if np.sum(approved_mask) > 0 else 0
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()
    
    return {
        "gini": gini_score(y_true, y_pred_proba),  
        "auc": roc_auc_score(y_true, y_pred_proba),
        "ks": ks,
        "ks_threshold": ks_thresh,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "approval_rate": approval_rate,
        "bad_rate_approved": bad_rate_approved
    }

# ----- Wrapper -----
def evaluate_model_wrapper(
    y_true,
    y_pred_proba,
    fold_number,
    selected_features=None,
    psi_feature_mean=None,
    psi_feature_max=None,
    psi_model=None,
    threshold=None,
    prev_selected_features=None,
):
    """
    Evaluates one fold and returns organized results.
    
    Parameters
    ----------
    prev_selected_features : set, optional
        Set of selected features from previous fold for Jaccard similarity calculation.
        If provided, Jaccard similarity will be computed between current and previous features.
    """
    metrics_dict = evaluate_model(y_true, y_pred_proba, threshold)
    
    # Fold info - train_size not applicable in wrapper, it's set in kfold_trainer
    fold_info = {
        "fold": fold_number,
        "val_size": len(y_true),
    }
    
    # Feature stability
    stability_info = {
        "selected_features": len(selected_features) if selected_features else 0,
        "psi_feature_mean": psi_feature_mean,
        "psi_feature_max": psi_feature_max,
        "psi_model": psi_model,
    }
    
    # Jaccard similarity - use passed prev_selected_features to avoid global state
    if prev_selected_features is not None:
        curr_set = set(selected_features) if selected_features else set()
        stability_info["jaccard_similarity"] = jaccard_similarity(prev_selected_features, curr_set)
    else:
        stability_info["jaccard_similarity"] = np.nan
    
    # Combine everything
    result = {**fold_info, **metrics_dict, **stability_info}
    
    return result

def save_fold_results(all_fold_results, output_csv):
    """
    all_fold_results: list of dicts returned by evaluate_model_wrapper
    """
    df = pd.DataFrame(all_fold_results)
    
    # Compute mean and std rows
    metrics_cols = [c for c in df.columns if c not in ["fold", "selected_features"]]  # exclude fold-specific
    mean_row = df[metrics_cols].mean()
    mean_row["fold"] = "mean"
    
    std_row = df[metrics_cols].std()
    std_row["fold"] = "std"
    
    # Optional: Insert a blank row between folds and summary
    df = pd.concat([df, pd.DataFrame([{}]), pd.DataFrame([mean_row]), pd.DataFrame([std_row])], ignore_index=True)
    
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved metrics to {output_csv}")
