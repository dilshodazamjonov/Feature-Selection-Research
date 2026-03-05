# metrics.py
"""
Metrics utilities for PCA-based credit scoring experiments.
Calculates predictive performance, parsimony, and interpretability metrics.
"""

import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score
)
import numpy as np
import os

def calculate_metrics(y_true, y_pred, y_pred_proba, n_components, original_features, pca_explained_variance=None):
    """
    Calculate performance, parsimony, and interpretability metrics.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        y_pred_proba (array-like): Predicted probabilities for positive class
        n_components (int): Number of PCA components used
        original_features (int): Number of original features before PCA
        pca_explained_variance (array-like, optional): Explained variance ratio per component
        
    Returns:
        dict: Dictionary with all metrics
    """
    # --- Predictive metrics ---
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # --- Parsimony metrics ---
    reduction_ratio = (original_features - n_components) / original_features
    explained_variance_ratio = np.sum(pca_explained_variance) if pca_explained_variance is not None else np.nan
    
    # Combine all metrics
    metrics_dict = {
        "n_components": n_components,
        "explained_variance_ratio": explained_variance_ratio,
        "reduction_ratio": reduction_ratio,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Accuracy": accuracy,
        "F1": f1,
    }
    
    return metrics_dict

def save_metrics_to_csv(metrics_list, filepath="data/output/pca_metrics_results.csv"):
    """
    Save a list of metrics dictionaries to CSV safely.
    """
    df = pd.DataFrame(metrics_list)
    
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    
    # If file exists and is open, add a suffix instead
    try:
        df.to_csv(filepath, index=False)
    except PermissionError:
        base, ext = os.path.splitext(filepath)
        filepath_new = base + "_new" + ext
        df.to_csv(filepath_new, index=False)
        print(f"Original file locked. Metrics saved to {filepath_new}")
        return
    
    print(f"Metrics saved to {filepath}")