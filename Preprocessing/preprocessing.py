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
        X_num = X_num.replace([np.inf, -np.inf], np.nan)

        # handle missing values
        if self.strategy == "mean":
            self.fill_values_ = X_num.mean()
        elif self.strategy == "median":
            self.fill_values_ = X_num.median()
        elif self.strategy == "zero":
            self.fill_values_ = pd.Series(0, index=self.num_cols)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        self.fill_values_ = self.fill_values_.fillna(0)
        X_num = X_num.fillna(self.fill_values_)
        X_num = X_num.fillna(0)

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
        X_num = X_num.fillna(0)

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
    Preprocess categorical features with consistent one-hot encoding.
    """

    def __init__(
        self,
        max_cardinality: int = 7,
        missing_value: str = "Missing",
        min_frequency: int | float | None = 10,
    ):
        self.max_cardinality = max_cardinality
        self.missing_value = missing_value
        self.min_frequency = min_frequency
        self.cat_cols: List[str] = []
        self.ohe: Optional[OneHotEncoder] = None

    def fit(self, X: pd.DataFrame):
        self.cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

        if self.cat_cols:
            X_cat = X[self.cat_cols].fillna(self.missing_value).astype(str)
            self.ohe = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                min_frequency=self.min_frequency,
            )
            self.ohe.fit(X_cat)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.cat_cols and self.ohe is not None:
            X_cat = X[self.cat_cols].fillna(self.missing_value).astype(str)
            X_encoded = self.ohe.transform(X_cat)
            return pd.DataFrame(
                X_encoded,
                columns=self.ohe.get_feature_names_out(self.cat_cols),
                index=X.index
            )

        return pd.DataFrame(index=X.index)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)


class Preprocessor:
    """
    Full preprocessing pipeline combining numerical scaling and categorical encoding.
    """

    def __init__(
        self,
        num_strategy="mean",
        num_scaler="standard",
        cat_max_card=7,
        cat_missing="Missing",
        cat_min_frequency=10,
    ):
        self.num_scaler = NumericalScaler(strategy=num_strategy, scaler=num_scaler)
        self.cat_encoder = CategoricalEncoder(
            max_cardinality=cat_max_card,
            missing_value=cat_missing,
            min_frequency=cat_min_frequency,
        )

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
