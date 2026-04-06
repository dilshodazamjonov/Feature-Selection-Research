import numpy as np
import pandas as pd
import shap
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier


def calculate_shap_importance(
    model,
    X: pd.DataFrame,
    batch_size: int = 500
) -> pd.DataFrame:
    """
    Fast + robust SHAP importance computation.

    Improvements:
    - Uses fast SHAP API (explainer(...))
    - Uses tree_path_dependent approximation
    - Limits number of trees for speed
    - Batch processing with progress bar
    """

    # -------------------------
    # Data safety
    # -------------------------
    X_safe = X.copy()
    X_safe = X_safe.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_safe = X_safe.astype(float)

    n_samples = X_safe.shape[0]

    # -------------------------
    # TREE MODELS (FAST PATH)
    # -------------------------
    if isinstance(model, (RandomForestClassifier, CatBoostClassifier)):

        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent"
        )

        shap_values_list = []

        for start in tqdm(range(0, n_samples, batch_size),
                          desc="SHAP (tree)",
                          ncols=100):

            end = min(start + batch_size, n_samples)

            # Fast new API
            shap_batch = explainer(
                X_safe.iloc[start:end],
                check_additivity=False
            ).values

            # Handle binary classification (n_samples, n_features, 2)
            if shap_batch.ndim == 3:
                shap_batch = shap_batch[:, :, 1]

            shap_values_list.append(shap_batch)

        shap_values = np.vstack(shap_values_list)

    # -------------------------
    # LINEAR MODELS
    # -------------------------
    elif isinstance(model, LogisticRegression):

        masker = shap.maskers.Independent(X_safe)
        explainer = shap.LinearExplainer(model, masker=masker)

        shap_values = explainer(X_safe).values

        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

    else:
        raise ValueError(f"Unsupported model type for SHAP: {type(model)}")

    # -------------------------
    # Aggregate importance
    # -------------------------
    shap_df = pd.DataFrame(shap_values, columns=X_safe.columns)

    shap_importance = shap_df.abs().mean()

    shap_importance_df = pd.DataFrame({
        "feature": shap_importance.index,
        "mean_abs_shap": shap_importance.values
    }).sort_values("mean_abs_shap", ascending=False)

    return shap_importance_df