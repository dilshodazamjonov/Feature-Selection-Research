from sklearn.metrics import roc_curve
import numpy as np

def ks_statistic(y_true, y_prob):

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    ks = np.max(tpr - fpr)

    return ks