from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from credit_risk_fs.preprocessing.labeling import (
    LENDINGCLUB_BAD_STATUSES,
    LENDINGCLUB_GOOD_STATUSES,
    build_lendingclub_target,
)
from credit_risk_fs.preprocessing.leakage import apply_leakage_blacklist

LENDINGCLUB_DIRECT_LEAKAGE_COLUMNS = (
    "loan_status",
)

LENDINGCLUB_POST_OUTCOME_LEAKAGE_COLUMNS = (
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "deferral_term",
    "payment_plan_start_date",
    "orig_projected_additional_accrued_interest",
)

LENDINGCLUB_POLICY_OR_POST_APPROVAL_COLUMNS = (
    "pymnt_plan",
    "disbursement_method",
)

LENDINGCLUB_UNDERWRITING_POLICY_COLUMNS = (
    "grade",
    "sub_grade",
    "int_rate",
    "installment",
    "funded_amnt",
    "funded_amnt_inv",
)

LENDINGCLUB_IDENTIFIER_OR_TEXT_COLUMNS = (
    "id",
    "member_id",
    "url",
    "desc",
    "emp_title",
    "zip_code",
)

LENDINGCLUB_TEXT_OR_LOW_SIGNAL_COLUMNS = (
    "title",
    "policy_code",
)

LENDINGCLUB_RAW_DATE_STRING_COLUMNS = (
    "earliest_cr_line",
    "sec_app_earliest_cr_line",
)

LENDINGCLUB_EXCLUDED_FEATURE_COLUMNS = (
    "TARGET",
    "recent_decision",
    "issue_d",
    *LENDINGCLUB_DIRECT_LEAKAGE_COLUMNS,
)


def lendingclub_model_blacklist() -> tuple[str, ...]:
    return (
        *LENDINGCLUB_DIRECT_LEAKAGE_COLUMNS,
        *LENDINGCLUB_POST_OUTCOME_LEAKAGE_COLUMNS,
        *LENDINGCLUB_POLICY_OR_POST_APPROVAL_COLUMNS,
        *LENDINGCLUB_UNDERWRITING_POLICY_COLUMNS,
        *LENDINGCLUB_IDENTIFIER_OR_TEXT_COLUMNS,
        *LENDINGCLUB_TEXT_OR_LOW_SIGNAL_COLUMNS,
    )


def _strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    object_cols = prepared.select_dtypes(include=["object", "string"]).columns.tolist()
    for column in object_cols:
        prepared[column] = prepared[column].astype("string").str.strip()
        prepared[column] = prepared[column].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return prepared


def _parse_issue_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%b-%Y", errors="coerce")


def ensure_lendingclub_target_and_time(
    df: pd.DataFrame,
    *,
    target_col: str = "TARGET",
    status_col: str = "loan_status",
    time_col: str = "recent_decision",
    issue_col: str = "issue_d",
) -> pd.DataFrame:
    prepared = df.copy()
    prepared = _strip_object_columns(prepared)
    prepared = prepared.replace([np.inf, -np.inf], np.nan)

    if issue_col in prepared.columns:
        prepared[issue_col] = _parse_issue_date(prepared[issue_col])

    if target_col not in prepared.columns:
        if status_col not in prepared.columns:
            raise ValueError("LendingClub frame must contain either TARGET or loan_status.")
        statuses_to_keep = LENDINGCLUB_BAD_STATUSES | LENDINGCLUB_GOOD_STATUSES
        prepared = prepared[prepared[status_col].isin(statuses_to_keep)].copy()
        prepared[target_col] = build_lendingclub_target(prepared, status_col=status_col)

    if time_col not in prepared.columns:
        if issue_col not in prepared.columns:
            raise ValueError("LendingClub frame must contain either recent_decision or issue_d.")
        max_issue = prepared[issue_col].max()
        prepared[time_col] = (prepared[issue_col] - max_issue).dt.days

    return prepared


def _drop_constant_columns(df: pd.DataFrame, protected: Iterable[str]) -> pd.DataFrame:
    protected_set = set(protected)
    drop_cols = [
        column
        for column in df.columns
        if column not in protected_set and df[column].nunique(dropna=True) <= 1
    ]
    return df.drop(columns=drop_cols, errors="ignore")


def prepare_lendingclub_application_frame(
    df: pd.DataFrame,
    *,
    target_col: str = "TARGET",
    time_col: str = "recent_decision",
    issue_col: str = "issue_d",
) -> pd.DataFrame:
    prepared = ensure_lendingclub_target_and_time(
        df,
        target_col=target_col,
        time_col=time_col,
        issue_col=issue_col,
    )
    prepared = apply_leakage_blacklist(prepared, lendingclub_model_blacklist())
    prepared = prepared[prepared[time_col].notna()].copy()
    prepared = _drop_constant_columns(
        prepared,
        protected=(target_col, time_col, issue_col),
    )
    return prepared


__all__ = [
    "LENDINGCLUB_BAD_STATUSES",
    "LENDINGCLUB_GOOD_STATUSES",
    "LENDINGCLUB_DIRECT_LEAKAGE_COLUMNS",
    "LENDINGCLUB_POST_OUTCOME_LEAKAGE_COLUMNS",
    "LENDINGCLUB_POLICY_OR_POST_APPROVAL_COLUMNS",
    "LENDINGCLUB_UNDERWRITING_POLICY_COLUMNS",
    "LENDINGCLUB_IDENTIFIER_OR_TEXT_COLUMNS",
    "LENDINGCLUB_TEXT_OR_LOW_SIGNAL_COLUMNS",
    "LENDINGCLUB_RAW_DATE_STRING_COLUMNS",
    "LENDINGCLUB_EXCLUDED_FEATURE_COLUMNS",
    "build_lendingclub_target",
    "ensure_lendingclub_target_and_time",
    "lendingclub_model_blacklist",
    "prepare_lendingclub_application_frame",
]
