# feature_engineering.py

import pandas as pd
from typing import List
from Preprocessing.data_process import build_aggregations


def build_bureau_features(bureau_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build aggregated bureau features per customer (SK_ID_CURR).

    Features include:
    - Credit exposure (total credit, total debt, overdue amounts)
    - Debt ratio (debt / credit)
    - Delinquency behavior (overdue flags, max overdue days)
    - Credit activity (active credit ratio)
    - Temporal signals (recency of credits)

    Returns:
        pd.DataFrame with one row per SK_ID_CURR
    """
    df = bureau_df.copy()

    df["debt_ratio"] = df["AMT_CREDIT_SUM_DEBT"] / (df["AMT_CREDIT_SUM"] + 1e-6)
    df["is_active"] = (df["CREDIT_ACTIVE"] == "Active").astype(int)
    df["is_overdue"] = (df["CREDIT_DAY_OVERDUE"] > 0).astype(int)

    bureau_config = {
        "bureau_count": ("SK_ID_BUREAU", "count"),
        "total_credit_sum": ("AMT_CREDIT_SUM", "sum"),
        "total_debt": ("AMT_CREDIT_SUM_DEBT", "sum"),
        "total_overdue": ("AMT_CREDIT_SUM_OVERDUE", "sum"),
        "avg_debt_ratio": ("debt_ratio", "mean"),
        "max_overdue_days": ("CREDIT_DAY_OVERDUE", "max"),
        "overdue_rate": ("is_overdue", "mean"),
        "active_rate": ("is_active", "mean"),
        "avg_days_credit": ("DAYS_CREDIT", "mean"),
        "recent_credit_days": ("DAYS_CREDIT", "max"),
    }

    return build_aggregations(df, "SK_ID_CURR", bureau_config)


def build_previous_app_features(prev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build aggregated previous application features per customer.

    Features include:
    - Application volume
    - Approval and rejection rates
    - Credit requested vs granted ratios
    - Financial characteristics (credit, annuity)
    - Recency of applications

    Returns:
        pd.DataFrame with one row per SK_ID_CURR
    """
    df = prev_df.copy()

    df["credit_diff"] = df["AMT_APPLICATION"] - df["AMT_CREDIT"]
    df["credit_ratio"] = df["AMT_CREDIT"] / (df["AMT_APPLICATION"] + 1e-6)
    df["is_approved"] = (df["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    df["is_rejected"] = (df["NAME_CONTRACT_STATUS"] == "Refused").astype(int)

    prev_config = {
        "prev_app_count": ("SK_ID_PREV", "count"),
        "approval_rate": ("is_approved", "mean"),
        "rejection_rate": ("is_rejected", "mean"),
        "avg_credit": ("AMT_CREDIT", "mean"),
        "max_credit": ("AMT_CREDIT", "max"),
        "avg_credit_ratio": ("credit_ratio", "mean"),
        "avg_annuity": ("AMT_ANNUITY", "mean"),
        "recent_decision": ("DAYS_DECISION", "max"),
        "avg_decision": ("DAYS_DECISION", "mean"),
    }

    return build_aggregations(df, "SK_ID_CURR", prev_config)


def build_credit_card_features(cc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build aggregated credit card balance features per customer.

    Features include:
    - Balance statistics
    - Credit utilization ratio
    - Payment behavior (payment ratios)
    - Delinquency metrics (DPD)

    Returns:
        pd.DataFrame with one row per SK_ID_CURR
    """
    df = cc_df.copy()

    df["utilization"] = df["AMT_BALANCE"] / (df["AMT_CREDIT_LIMIT_ACTUAL"] + 1e-6)
    df["payment_ratio"] = df["AMT_PAYMENT_TOTAL_CURRENT"] / (df["AMT_INST_MIN_REGULARITY"] + 1e-6)
    df["is_delinquent"] = (df["SK_DPD"] > 0).astype(int)

    cc_config = {
        "cc_count": ("SK_ID_PREV", "count"),
        "avg_balance": ("AMT_BALANCE", "mean"),
        "max_balance": ("AMT_BALANCE", "max"),
        "avg_utilization": ("utilization", "mean"),
        "total_payment": ("AMT_PAYMENT_TOTAL_CURRENT", "sum"),
        "avg_payment_ratio": ("payment_ratio", "mean"),
        "delinq_rate": ("is_delinquent", "mean"),
        "max_dpd": ("SK_DPD", "max"),
    }

    return build_aggregations(df, "SK_ID_CURR", cc_config)


def build_pos_cash_features(pos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build aggregated POS_CASH balance features per customer.

    Features include:
    - Installment structure (total and remaining)
    - Delinquency behavior (DPD, DPD_DEF)
    - Default frequency indicators

    Returns:
        pd.DataFrame with one row per SK_ID_CURR
    """
    df = pos_df.copy()

    df["is_delinquent"] = (df["SK_DPD"] > 0).astype(int)

    pos_config = {
        "pos_count": ("SK_ID_PREV", "count"),
        "avg_instalment": ("CNT_INSTALMENT", "mean"),
        "remaining_instalments": ("CNT_INSTALMENT_FUTURE", "mean"),
        "delinq_rate": ("is_delinquent", "mean"),
        "max_dpd": ("SK_DPD", "max"),
        "max_dpd_def": ("SK_DPD_DEF", "max"),
    }

    return build_aggregations(df, "SK_ID_CURR", pos_config)


def build_installments_features(inst_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build aggregated installment payment features per customer.

    Features include:
    - Payment amounts and totals
    - Late payment behavior (lateness days and frequency)
    - Payment ratio (paid vs expected)

    Returns:
        pd.DataFrame with one row per SK_ID_CURR
    """
    df = inst_df.copy()

    df["late_days"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    df["late_flag"] = (df["late_days"] > 0).astype(int)
    df["payment_ratio"] = df["AMT_PAYMENT"] / (df["AMT_INSTALMENT"] + 1e-6)

    inst_config = {
        "inst_count": ("SK_ID_PREV", "count"),
        "avg_instalment": ("AMT_INSTALMENT", "mean"),
        "total_paid": ("AMT_PAYMENT", "sum"),
        "late_rate": ("late_flag", "mean"),
        "max_late_days": ("late_days", "max"),
        "avg_payment_ratio": ("payment_ratio", "mean"),
    }

    return build_aggregations(df, "SK_ID_CURR", inst_config)


def build_all_features(dataframes: dict) -> List[pd.DataFrame]:
    """
    Build all feature tables from provided datasets.

    Expected keys in `dataframes` dict:
        - 'bureau'
        - 'previous_application'
        - 'credit_card_balance'
        - 'POS_CASH_balance'
        - 'installments_payments'

    Returns:
        List[pd.DataFrame]: list of aggregated feature tables
    """
    features = []

    if 'bureau' in dataframes:
        features.append(build_bureau_features(dataframes['bureau']))

    if 'previous_application' in dataframes:
        features.append(build_previous_app_features(dataframes['previous_application']))

    if 'credit_card_balance' in dataframes:
        features.append(build_credit_card_features(dataframes['credit_card_balance']))

    if 'POS_CASH_balance' in dataframes:
        features.append(build_pos_cash_features(dataframes['POS_CASH_balance']))

    if 'installments_payments' in dataframes:
        features.append(build_installments_features(dataframes['installments_payments']))

    return features