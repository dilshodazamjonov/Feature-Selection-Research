from __future__ import annotations

from typing import List

import pandas as pd

from credit_risk_fs.feature_engineering.lendingclub.application import (
    build_application_features,
)


def build_all_features(dataframes: dict[str, pd.DataFrame]) -> List[pd.DataFrame]:
    if "application_train" not in dataframes:
        return []
    return [build_application_features(dataframes["application_train"])]
