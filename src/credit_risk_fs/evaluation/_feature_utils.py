import pandas as pd
import numpy as np

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


def _feature_score_lookup(selector, selected_features):
    """
    Best-effort score extraction for selectors that expose rankings/importances.
    Missing scores are acceptable; the CSV schema still records rank/order.
    """
    if selector is None or selected_features is None:
        return {}

    score_sources = []
    for candidate in [
        selector,
        getattr(selector, "stat_selector", None),
        getattr(selector, "rfe", None),
        getattr(selector, "boruta", None),
    ]:
        if candidate is None:
            continue
        if hasattr(candidate, "rf_importances_"):
            score_sources.append(getattr(candidate, "rf_importances_"))
        if hasattr(candidate, "explained_variance"):
            values = getattr(candidate, "explained_variance")
            if values is not None:
                score_sources.append(pd.Series(values, index=list(selected_features)))

    for source in score_sources:
        if isinstance(source, pd.Series):
            return {
                str(feature): float(source.loc[feature])
                for feature in selected_features
                if feature in source.index and pd.notna(source.loc[feature])
            }

    return {}


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
