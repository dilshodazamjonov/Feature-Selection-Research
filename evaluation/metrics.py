from sklearn.metrics import roc_curve
import numpy as np

def ks_statistic(y_true, y_prob):

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_scores = tpr - fpr
    ks_idx = np.argmax(ks_scores)
    ks = np.max(ks_scores)

    optimal_threshold = thresholds[ks_idx]

    return ks, optimal_threshold
