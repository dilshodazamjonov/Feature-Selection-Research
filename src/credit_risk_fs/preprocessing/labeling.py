from __future__ import annotations

import pandas as pd


LENDINGCLUB_FINAL_BAD_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}

LENDINGCLUB_FINAL_GOOD_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}

LENDINGCLUB_AMBIGUOUS_OR_ONGOING_STATUSES = {
    "Current",
    "In Grace Period",
    "Late (31-120 days)",
    "Late (16-30 days)",
    "Issued",
}

LENDINGCLUB_BAD_STATUSES = LENDINGCLUB_FINAL_BAD_STATUSES
LENDINGCLUB_GOOD_STATUSES = LENDINGCLUB_FINAL_GOOD_STATUSES


def build_lendingclub_target(df: pd.DataFrame, *, status_col: str = "loan_status") -> pd.Series:
    status = df[status_col].astype(str)
    return status.isin(LENDINGCLUB_FINAL_BAD_STATUSES).astype(int)
