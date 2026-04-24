import pandas as pd
import numpy as np
import os

def _to_df(X, index=None, columns=None):
    """
    Converts input data to a pandas DataFrame if it isn't one already,
    generating default column names if none are provided.
    """
    if isinstance(X, pd.DataFrame):
        return X

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    cols = columns if columns is not None else [f"feature_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, index=index, columns=cols)


def _safe_get_selected_features(selector):
    """
    Tries to extract selected feature names from a selector if available.
    """
    if selector is None:
        return None

    if hasattr(selector, "selected_features") and selector.selected_features is not None:
        feats = selector.selected_features
        if isinstance(feats, pd.Index):
            return feats.tolist()
        if isinstance(feats, (list, tuple, np.ndarray)):
            return list(feats)

    if hasattr(selector, "boruta") and hasattr(selector.boruta, "selected_features"):
        feats = selector.boruta.selected_features
        if feats is not None:
            return list(feats)

    return None


def _save_selected_features(path, selected_features):
    """
    Saves selected feature names to CSV.
    """
    if selected_features is None:
        return

    pd.DataFrame({"feature": list(selected_features)}).to_csv(path, index=False)


def _save_feature_statistics(path, X_train_f):
    """
    Saves per-feature summary statistics for the selected training features.
    """
    stats_df = X_train_f.describe(include="all").T

    # Numeric summaries are most useful, but describe(include="all") may create mixed columns.
    # Add missingness explicitly.
    stats_df["missing_count"] = X_train_f.isna().sum()
    stats_df["missing_pct"] = X_train_f.isna().mean() * 100
    stats_df["n_unique"] = X_train_f.nunique(dropna=True)

    stats_df.to_csv(path, index=True)


def _extract_feature_importance(model, feature_names):
    """
    Returns a DataFrame with feature importances if supported.
    Handles CatBoost, RandomForest, or LogisticRegression (coef).
    """
    if hasattr(model, "get_feature_importance"):
        importance_df = model.get_feature_importance()
        if isinstance(importance_df, pd.DataFrame) and {"feature", "importance"}.issubset(importance_df.columns):
            return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    estimator = getattr(model, "model", model)

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_).flatten()
    else:
        # fallback: uniform importance if not available
        importances = np.zeros(len(feature_names))

    # Ensure lengths match
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Length mismatch: {len(importances)} importances vs {len(feature_names)} features"
        )

    return pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)


def _save_correlation_matrix(path, X_train_f):
    """
    Saves the correlation matrix for selected features.

    Only numeric columns are included. Non-numeric columns are ignored.
    """
    numeric_df = X_train_f.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        # Not enough numeric features to compute a correlation matrix
        pd.DataFrame().to_csv(path, index=True)
        return

    corr = numeric_df.corr()
    corr.to_csv(path, index=True)


def _save_stagewise_selection(selector, fold_dir):
    """
    Saves intermediate stage features if the selector exposes them,
    e.g. Boruta + RFE pipeline.
    """
    if selector is None:
        return

    boruta_feats = None
    rfe_feats = None

    if hasattr(selector, "boruta") and hasattr(selector.boruta, "selected_features"):
        boruta_feats = selector.boruta.selected_features

    if hasattr(selector, "rfe") and hasattr(selector.rfe, "selected_features"):
        rfe_feats = selector.rfe.selected_features

    if boruta_feats is not None:
        _save_selected_features(os.path.join(fold_dir, "boruta_features.csv"), boruta_feats)

    if rfe_feats is not None:
        _save_selected_features(os.path.join(fold_dir, "rfe_features.csv"), rfe_feats)
