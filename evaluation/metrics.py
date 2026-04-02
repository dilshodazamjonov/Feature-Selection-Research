import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, 
    confusion_matrix, 
    roc_curve,
    accuracy_score
)
import os

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
    """
    Evaluates a binary classification model across ML and business metrics.

    If threshold is None, uses KS-based threshold.

    Returns a dictionary with:
        gini, auc, ks, ks_threshold, tn, fp, tp, fn,
        precision, recall, f1, accuracy,
        approval_rate (% of approved cases),
        bad_rate_approved (% of bads among approved)
    """
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

def evaluate_model_wrapper(
    y_true,
    y_pred_proba,
    threshold=None,
    output_dir="data/outputs",
    feature_sets: list = None,
    metrics_list: list = None,
    method_name="Method"
):
    """
    Wrapper to evaluate a model, save metrics and plots to disk.
    
    Parameters:
        y_true: true labels
        y_pred_proba: predicted probabilities
        threshold: optional threshold for classification; defaults to KS-based threshold
        output_dir: directory to save plots
        feature_sets: optional list of feature lists per fold (for stability plot)
        metrics_list: optional list of evaluate_model() dicts per fold (for metric distribution)
        method_name: name of the method/model for labeling plots
    
    Returns:
        metrics_dict: dictionary with evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Compute metrics
    metrics_dict = evaluate_model(y_true, y_pred_proba, threshold)

    # Save threshold analysis plot
    plot_path = os.path.join(output_dir, f"threshold_analysis_{method_name}.png")
    plt.figure(figsize=(8,5))
    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_pred_proba)
    thresholds = np.linspace(0, 1, 100)
    approval_rates, bad_rates = [], []
    for thresh in thresholds:
        y_pred = (y_prob_np >= thresh).astype(int)
        approval_rates.append(float(np.mean(y_pred == 0)))
        approved_mask = (y_pred == 0)
        bad_rate = float(np.mean(y_true_np[approved_mask])) if np.sum(approved_mask) > 0 else 0
        bad_rates.append(bad_rate)
    plt.plot(thresholds, approval_rates, label="Approval Rate")
    plt.plot(thresholds, bad_rates, label="Bad Rate among Approved")
    plt.axvline(x=metrics_dict['ks_threshold'], color='red', linestyle='--', label='KS Threshold')
    plt.xlabel("Probability Threshold")
    plt.ylabel("Rate")
    plt.title(f"Threshold Analysis ({method_name})")
    plt.legend()
    plt.savefig(plot_path)
    plt.close()
    
    # Metric distribution plot if metrics_list provided
    if metrics_list:
        plot_path_md = os.path.join(output_dir, f"metric_distribution_{method_name}.png")
        plt.figure(figsize=(6,4))
        values = [m['ks'] for m in metrics_list]  # example with KS; can be parameterized
        sns.boxplot(values)
        plt.title(f"KS Distribution Across Folds ({method_name})")
        plt.ylabel("KS")
        plt.savefig(plot_path_md)
        plt.close()

    return metrics_dict