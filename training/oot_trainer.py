# preprocessing/oot_split.py

import pandas as pd
from typing import Tuple, Optional


def oot_split(
    df: pd.DataFrame,
    time_col: str,
    test_size: float = 0.2,
    min_train_size: Optional[int] = None,
    target_col: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Out-of-Time split based on a temporal column.

    Drops rows with missing time values, sorts by time, and splits
    into train/test while respecting temporal order.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    time_col : str
        Column used for temporal ordering
    test_size : float
        Fraction of most recent data used as test
    min_train_size : Optional[int]
        Minimum number of samples required in train
    target_col : Optional[str]
        Column to separate as target variable

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
        Split train/test sets, target optional
    """
    if time_col not in df.columns:
        raise ValueError(f"{time_col} not found in dataframe")

    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")

    df = df.dropna(subset=[time_col]).copy()
    df = df.sort_values(by=time_col)

    split_idx = int(len(df) * (1 - test_size))
    if min_train_size and split_idx < min_train_size:
        raise ValueError("Train size is smaller than min_train_size")

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    if target_col and target_col in df.columns:
        y_train = train_df[target_col]
        y_test = test_df[target_col]
        X_train = train_df.drop(columns=[target_col])
        X_test = test_df.drop(columns=[target_col])
    else:
        X_train, X_test = train_df, test_df
        y_train = y_test = pd.Series(dtype=float)

    return X_train, X_test, y_train, y_test