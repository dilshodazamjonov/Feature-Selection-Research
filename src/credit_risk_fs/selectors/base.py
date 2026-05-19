from __future__ import annotations

from typing import Protocol

import pandas as pd


class FeatureSelector(Protocol):
    selected_features_: list[str] | None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None): ...
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame: ...
