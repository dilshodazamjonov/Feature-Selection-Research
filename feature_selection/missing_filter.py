from __future__ import annotations

from pathlib import Path

import pandas as pd


class MissingRateFilter:
    """
    Column filter that removes features with excessive missingness.
    """

    def __init__(self, max_missing_rate: float = 0.95):
        if not 0.0 <= max_missing_rate <= 1.0:
            raise ValueError("max_missing_rate must be in [0, 1].")
        self.max_missing_rate = max_missing_rate
        self.selected_features_: list[str] | None = None
        self.missing_summary_: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y=None):
        missing_rate = X.isna().mean().rename("missing_rate")
        self.missing_summary_ = (
            missing_rate.reset_index()
            .rename(columns={"index": "feature"})
            .sort_values(["missing_rate", "feature"], ascending=[False, True])
            .reset_index(drop=True)
        )
        self.selected_features_ = missing_rate[missing_rate <= self.max_missing_rate].index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selected_features_ is None:
            raise ValueError("MissingRateFilter must be fitted before transform.")
        return X.loc[:, self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save_summary(self, path: str | Path) -> None:
        if self.missing_summary_ is None:
            return
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.missing_summary_.to_csv(out_path, index=False)
