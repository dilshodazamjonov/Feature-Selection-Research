# data_preprocess.py
"""
Data preprocessing module for credit scoring ML experiments.
Handles missing values, encoding, scaling, and train/test split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def preprocess_data(
    file_path,
    target_col="target",
    test_size=0.3,
    random_state=42,
    verbose=True,
):
    """Preprocess dataset for modeling."""

    # 1. Load data
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    if verbose:
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Drop rows where target is missing
    df = df.dropna(subset=[target_col])

    # 3. Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 4. Replace infinities with NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 5. Fill missing values simply
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    if num_cols:
        X[num_cols] = X[num_cols].fillna(0)

    if cat_cols:
        X[cat_cols] = X[cat_cols].fillna("missing")

    # 6. Encode categorical features
    if cat_cols:
        ohe = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        X_ohe = ohe.fit_transform(X[cat_cols])
        ohe_feature_names = ohe.get_feature_names_out(cat_cols)
        X_ohe_df = pd.DataFrame(X_ohe, columns=ohe_feature_names, index=X.index)
        X = pd.concat([X.drop(columns=cat_cols), X_ohe_df], axis=1)

    # 7. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 8. Train/test split
    stratify_y = y if y.nunique() > 1 else None

    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_y,
    )

    feature_names = X.columns.tolist()

    if verbose:
        print(
            f"Preprocessing complete. Train shape: {X_train_scaled.shape}, "
            f"Test shape: {X_test_scaled.shape}"
        )

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


if __name__ == "__main__":

    X_train, X_test, y_train, y_test, feat_names = preprocess_data(
        "data/inputs/Master_Data_with_filtering_updated.csv",
        target_col="TARGET",
    )

    print("Features ready for modeling:", feat_names[:10], "...")