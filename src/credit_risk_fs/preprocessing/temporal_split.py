from __future__ import annotations

import pandas as pd


def split_dev_oot(
    df: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    dev_start_day: int,
    oot_start_day: int,
    oot_end_day: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered = df[(df[time_col] >= dev_start_day) & (df[time_col] <= oot_end_day)].copy()
    dev = filtered[(filtered[time_col] >= dev_start_day) & (filtered[time_col] < oot_start_day)].copy()
    oot = filtered[(filtered[time_col] >= oot_start_day) & (filtered[time_col] <= oot_end_day)].copy()
    if dev.empty or oot.empty:
        raise ValueError("Temporal split produced an empty DEV or OOT frame.")
    return dev, oot
