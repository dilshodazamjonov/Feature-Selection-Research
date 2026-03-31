# preprocessing/preprocessing.py

import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler


class NumericalScaler:
    """
    Preprocess numerical features: missing value filling and scaling.
    """

    def __init__(self, strategy: str = "mean", scaler: str = "standard"):
        self.strategy = strategy  # 'mean', 'median', 'zero', 'constant'
        self.scaler_type = scaler  # 'standard', 'minmax', None
        self.scaler = None
        self.num_cols: List[str] = []

    def fit(self, X: pd.DataFrame):
        self.num_cols = X.select_dtypes(include=["number"]).columns.tolist()
        X_num = X[self.num_cols].copy()

        # handle missing values
        if self.strategy == "mean":
            self.fill_values_ = X_num.replace([np.inf, -np.inf], np.nan).mean()
        elif self.strategy == "median":
            self.fill_values_ = X_num.replace([np.inf, -np.inf], np.nan).median()
        elif self.strategy == "zero":
            self.fill_values_ = pd.Series(0, index=self.num_cols)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        X_num = X_num.fillna(self.fill_values_)

        # fit scaler
        if self.scaler_type == "standard":
            self.scaler = StandardScaler().fit(X_num)
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler().fit(X_num)
        else:
            self.scaler = None

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_num = X[self.num_cols].copy()
        X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(self.fill_values_)

        if self.scaler:
            X_num = pd.DataFrame(
                self.scaler.transform(X_num),
                columns=self.num_cols,
                index=X.index
            )

        return X_num

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)


class CategoricalEncoder:
    """
    Preprocess categorical features: missing value filling and one-hot encoding.
    """

    def __init__(self, max_cardinality: int = 7, missing_value: str = "Missing"):
        self.max_cardinality = max_cardinality
        self.missing_value = missing_value
        self.cat_cols: List[str] = []
        self.low_card_cols: List[str] = []
        self.ohe: Optional[OneHotEncoder] = None

    def fit(self, X: pd.DataFrame):
        self.cat_cols = X.select_dtypes(include=["object", "category", str]).columns.tolist()
        self.low_card_cols = [c for c in self.cat_cols if X[c].nunique() <= self.max_cardinality]

        X_low = X[self.low_card_cols].fillna(self.missing_value).astype(str)
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.ohe.fit(X_low)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_low = X[self.low_card_cols].fillna(self.missing_value).astype(str)
        X_encoded = self.ohe.transform(X_low)

        X_encoded_df = pd.DataFrame(
            X_encoded,
            columns=self.ohe.get_feature_names_out(self.low_card_cols),
            index=X.index
        )

        return X_encoded_df

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)


class Preprocessor:
    """
    Full preprocessing pipeline combining numerical scaling and categorical encoding.
    """

    def __init__(self, num_strategy="mean", num_scaler="standard", cat_max_card=7, cat_missing="Missing"):
        self.num_scaler = NumericalScaler(strategy=num_strategy, scaler=num_scaler)
        self.cat_encoder = CategoricalEncoder(max_cardinality=cat_max_card, missing_value=cat_missing)

    def fit(self, X: pd.DataFrame):
        self.num_scaler.fit(X)
        self.cat_encoder.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_num = self.num_scaler.transform(X)
        X_cat = self.cat_encoder.transform(X)
        X_final = pd.concat([X_num, X_cat], axis=1)
        return X_final

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)