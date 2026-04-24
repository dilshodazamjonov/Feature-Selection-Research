import numpy as np
import pandas as pd

from Preprocessing.data_process import normalize_home_credit_sentinel_dates
from Preprocessing.feature_engineering import TIME_PROXY_COL, build_application_time_proxy
from Preprocessing.preprocessing import Preprocessor
from feature_selection.pca import PCASelector
from training.cv_utils import GroupedTimeSeriesSplit


def test_normalize_home_credit_sentinel_dates_replaces_only_day_columns():
    df = pd.DataFrame(
        {
            "DAYS_EMPLOYED": [-100.0, 365243.0],
            "OTHER_VALUE": [365243.0, 1.0],
        }
    )

    cleaned = normalize_home_credit_sentinel_dates(df, "application_train")

    assert cleaned.loc[0, "DAYS_EMPLOYED"] == -100.0
    assert np.isnan(cleaned.loc[1, "DAYS_EMPLOYED"])
    assert cleaned.loc[0, "OTHER_VALUE"] == 365243.0


def test_build_application_time_proxy_uses_more_than_previous_application():
    proxy = build_application_time_proxy(
        {
            "previous_application": pd.DataFrame(
                {"SK_ID_CURR": [1], "DAYS_DECISION": [-15.0]}
            ),
            "bureau": pd.DataFrame(
                {"SK_ID_CURR": [2], "DAYS_CREDIT": [-30.0]}
            ),
        }
    )

    assert set(proxy["SK_ID_CURR"]) == {1, 2}
    proxy_map = dict(zip(proxy["SK_ID_CURR"], proxy[TIME_PROXY_COL]))
    assert proxy_map[1] == -15.0
    assert proxy_map[2] == -30.0


def test_preprocessor_one_hot_encodes_high_cardinality_categories():
    X = pd.DataFrame(
        {
            "ORGANIZATION_TYPE": ["A", "B", "C", "A"],
            "AMT_CREDIT": [1.0, 2.0, 3.0, 4.0],
        }
    )

    transformed = Preprocessor(cat_min_frequency=1).fit_transform(X)

    assert "ORGANIZATION_TYPE" not in transformed.columns
    assert any(col.startswith("ORGANIZATION_TYPE_") for col in transformed.columns)


def test_preprocessor_replaces_infinite_values_before_scaling():
    X = pd.DataFrame(
        {
            "AMT_CREDIT": [1.0, np.inf, 3.0],
            "NAME_CONTRACT_TYPE": ["Cash", "Cash", "Revolving"],
        }
    )

    transformed = Preprocessor(cat_min_frequency=1).fit_transform(X)

    assert np.isfinite(transformed["AMT_CREDIT"]).all()


def test_pca_selector_accepts_optional_target_argument():
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [3.0, 2.0, 1.0]})
    y = pd.Series([0, 1, 0])

    transformed = PCASelector(n_components=1).fit_transform(X, y)

    assert list(transformed.columns) == ["PC1"]
    assert transformed.shape == (3, 1)


def test_grouped_time_series_split_keeps_same_time_values_together():
    time_values = np.array([-4, -4, -3, -3, -2, -2, -1, -1], dtype=float)
    splitter = GroupedTimeSeriesSplit(n_splits=2, gap=0)

    for train_idx, val_idx in splitter.split(time_values):
        train_times = set(time_values[train_idx])
        val_times = set(time_values[val_idx])
        assert train_times.isdisjoint(val_times)
