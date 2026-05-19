import gc
import pandas as pd
import numpy as np
from typing import List

TIME_PROXY_COL = "application_time_proxy"

def aggregate_dataframe(df: pd.DataFrame, group_col: str, prefix: str) -> pd.DataFrame:
    """
    Automated mass-aggregator.
    1. One-hot encodes categories.
    2. Calculates min, max, mean, sum, var for all numeric columns.
    """
    cat_cols = [
        c for c in df.columns 
        if df[c].dtype == "object" or df[c].dtype == "category"
    ]
    
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)

    numeric_df = df.select_dtypes(include=[np.number])
    
    agg_dict = {}
    for col in numeric_df.columns:
        if col == group_col or col.startswith("SK_ID"):
            continue
        
        if df[col].nunique() <= 2:
            agg_dict[col] = ["mean", "sum"]
        else:
            agg_dict[col] = ["min", "max", "mean", "sum", "var"]

    if not agg_dict:
        return pd.DataFrame({group_col: df[group_col].unique()})

    agg_df = df.groupby(group_col).agg(agg_dict)
    
    agg_df.columns = [
        f"{prefix}_{c[0]}_{c[1].upper()}" for c in agg_df.columns
    ]
    
    return agg_df.reset_index()


def _aggregate_recency(
    df: pd.DataFrame | None,
    group_col: str,
    source_col: str,
    output_col: str,
) -> pd.DataFrame | None:
    if df is None or group_col not in df.columns or source_col not in df.columns:
        return None

    return (
        df[[group_col, source_col]]
        .dropna(subset=[source_col])
        .groupby(group_col, as_index=False)[source_col]
        .max()
        .rename(columns={source_col: output_col})
    )


def build_application_time_proxy(all_data: dict) -> pd.DataFrame | None:
    """
    Build a broader application-time proxy from all historical tables.

    The original implementation relied only on ``previous_application`` and
    dropped applicants without prior applications, which narrowed the study
    population. This proxy keeps the full cohort by using the most recent
    available event across historical sources.
    """
    time_parts: list[pd.DataFrame] = []

    prev = all_data.get("previous_application")
    prev_proxy = _aggregate_recency(
        prev,
        group_col="SK_ID_CURR",
        source_col="DAYS_DECISION",
        output_col="prev_recent_days",
    )
    if prev_proxy is not None:
        time_parts.append(prev_proxy)

    bureau = all_data.get("bureau")
    bureau_proxy = _aggregate_recency(
        bureau,
        group_col="SK_ID_CURR",
        source_col="DAYS_CREDIT",
        output_col="bureau_recent_days",
    )
    if bureau_proxy is not None:
        time_parts.append(bureau_proxy)

    inst = all_data.get("installments_payments")
    if inst is not None and "SK_ID_CURR" in inst.columns:
        inst_day_cols = [col for col in ["DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"] if col in inst.columns]
        if inst_day_cols:
            inst_proxy = inst[["SK_ID_CURR"]].copy()
            inst_proxy["inst_recent_days"] = inst[inst_day_cols].max(axis=1, skipna=True)
            inst_proxy = _aggregate_recency(
                inst_proxy,
                group_col="SK_ID_CURR",
                source_col="inst_recent_days",
                output_col="inst_recent_days",
            )
            if inst_proxy is not None:
                time_parts.append(inst_proxy)

    pos = all_data.get("POS_CASH_balance")
    if pos is not None and "MONTHS_BALANCE" in pos.columns and "SK_ID_CURR" in pos.columns:
        pos_proxy = pos[["SK_ID_CURR", "MONTHS_BALANCE"]].copy()
        pos_proxy["pos_recent_days"] = pos_proxy["MONTHS_BALANCE"] * 30
        pos_proxy = _aggregate_recency(
            pos_proxy,
            group_col="SK_ID_CURR",
            source_col="pos_recent_days",
            output_col="pos_recent_days",
        )
        if pos_proxy is not None:
            time_parts.append(pos_proxy)

    cc = all_data.get("credit_card_balance")
    if cc is not None and "MONTHS_BALANCE" in cc.columns and "SK_ID_CURR" in cc.columns:
        cc_proxy = cc[["SK_ID_CURR", "MONTHS_BALANCE"]].copy()
        cc_proxy["cc_recent_days"] = cc_proxy["MONTHS_BALANCE"] * 30
        cc_proxy = _aggregate_recency(
            cc_proxy,
            group_col="SK_ID_CURR",
            source_col="cc_recent_days",
            output_col="cc_recent_days",
        )
        if cc_proxy is not None:
            time_parts.append(cc_proxy)

    if not time_parts:
        return None

    proxy_df = time_parts[0]
    for part in time_parts[1:]:
        proxy_df = proxy_df.merge(part, on="SK_ID_CURR", how="outer")

    recency_cols = [col for col in proxy_df.columns if col != "SK_ID_CURR"]
    proxy_df[TIME_PROXY_COL] = proxy_df[recency_cols].max(axis=1, skipna=True)

    return proxy_df[["SK_ID_CURR", TIME_PROXY_COL]]

def build_bureau_features(all_data: dict) -> pd.DataFrame:
    bureau = all_data.get("bureau")
    bb = all_data.get("bureau_balance")
    
    if bureau is None:
        return None

    if bb is not None:
        bb_agg = aggregate_dataframe(bb, "SK_ID_BUREAU", "BB")
        bureau = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
        del bb, bb_agg
        gc.collect()

    bureau["DEBT_CREDIT_DIFF"] = bureau["AMT_CREDIT_SUM"] - bureau["AMT_CREDIT_SUM_DEBT"]
    bureau["DEBT_RATIO"] = bureau["AMT_CREDIT_SUM_DEBT"] / (bureau["AMT_CREDIT_SUM"] + 1e-6)
    bureau["CREDIT_DURATION"] = bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_CREDIT"]
    
    return aggregate_dataframe(bureau, "SK_ID_CURR", "BURO")

def build_previous_app_features(prev_df: pd.DataFrame) -> pd.DataFrame:
    if prev_df is None:
        return None
    
    prev_df["ASK_GRANT_RATIO"] = prev_df["AMT_APPLICATION"] / (prev_df["AMT_CREDIT"] + 1e-6)
    prev_df["APPLICATION_DIFF"] = prev_df["AMT_APPLICATION"] - prev_df["AMT_CREDIT"]
    prev_df["PAYMENT_TERM"] = prev_df["AMT_CREDIT"] / (prev_df["AMT_ANNUITY"] + 1e-6)
    prev_df["recent_decision"] = prev_df["DAYS_DECISION"]
    prev_df["INTEREST_ESTIMATE"] = (
        (prev_df["CNT_PAYMENT"] * prev_df["AMT_ANNUITY"] - prev_df["AMT_CREDIT"]) 
        / (prev_df["AMT_CREDIT"] + 1e-6)
    )

    agg_prev = aggregate_dataframe(prev_df, "SK_ID_CURR", "PREV")
    
    if "PREV_recent_decision_MAX" in agg_prev.columns:
        agg_prev["recent_decision"] = agg_prev["PREV_recent_decision_MAX"]
        
    return agg_prev

def build_pos_cash_features(pos_df: pd.DataFrame) -> pd.DataFrame:
    if pos_df is None:
        return None
    
    pos_df["LATE_PAYMENT"] = (pos_df["SK_DPD"] > 0).astype(int)
    pos_df["TOTAL_INSTALMENT_PROG"] = pos_df["CNT_INSTALMENT_FUTURE"] / (pos_df["CNT_INSTALMENT"] + 1e-6)
    
    return aggregate_dataframe(pos_df, "SK_ID_CURR", "POS")

def build_installments_features(inst_df: pd.DataFrame) -> pd.DataFrame:
    if inst_df is None:
        return None
    
    inst_df["PAYMENT_DIFF"] = inst_df["AMT_INSTALMENT"] - inst_df["AMT_PAYMENT"]
    inst_df["PAYMENT_RATIO"] = inst_df["AMT_PAYMENT"] / (inst_df["AMT_INSTALMENT"] + 1e-6)
    inst_df["DAYS_DIFF"] = inst_df["DAYS_ENTRY_PAYMENT"] - inst_df["DAYS_INSTALMENT"]
    inst_df["IS_LATE"] = (inst_df["DAYS_DIFF"] > 0).astype(int)
    inst_df["IS_UNDERPAID"] = (inst_df["PAYMENT_DIFF"] > 0).astype(int)

    return aggregate_dataframe(inst_df, "SK_ID_CURR", "INSTAL")

def build_credit_card_features(cc_df: pd.DataFrame) -> pd.DataFrame:
    if cc_df is None:
        return None
    
    cc_df["LIMIT_USE"] = cc_df["AMT_BALANCE"] / (cc_df["AMT_CREDIT_LIMIT_ACTUAL"] + 1e-6)
    cc_df["PAYMENT_DIV_MIN"] = cc_df["AMT_PAYMENT_CURRENT"] / (cc_df["AMT_INST_MIN_REGULARITY"] + 1e-6)
    cc_df["DRAWING_RATIO"] = cc_df["AMT_DRAWINGS_CURRENT"] / (cc_df["AMT_CREDIT_LIMIT_ACTUAL"] + 1e-6)
    
    return aggregate_dataframe(cc_df, "SK_ID_CURR", "CC")

def build_all_features(dataframes: dict) -> List[pd.DataFrame]:
    """
    Orchestrator to build 500+ features.
    """
    features_list = []

    print("Processing Bureau & Bureau Balance...")
    buro_agg = build_bureau_features(dataframes)
    if buro_agg is not None:
        features_list.append(buro_agg)
    
    dataframes.pop("bureau", None)
    dataframes.pop("bureau_balance", None)
    gc.collect()

    print("Processing Previous Applications...")
    if "previous_application" in dataframes:
        prev_agg = build_previous_app_features(dataframes["previous_application"])
        features_list.append(prev_agg)
        dataframes.pop("previous_application", None)
        gc.collect()

    print("Processing POS CASH...")
    if "POS_CASH_balance" in dataframes:
        pos_agg = build_pos_cash_features(dataframes["POS_CASH_balance"])
        features_list.append(pos_agg)
        dataframes.pop("POS_CASH_balance", None)
        gc.collect()

    print("Processing Installments...")
    if "installments_payments" in dataframes:
        inst_agg = build_installments_features(dataframes["installments_payments"])
        features_list.append(inst_agg)
        dataframes.pop("installments_payments", None)
        gc.collect()

    print("Processing Credit Card...")
    if "credit_card_balance" in dataframes:
        cc_agg = build_credit_card_features(dataframes["credit_card_balance"])
        features_list.append(cc_agg)
        dataframes.pop("credit_card_balance", None)
        gc.collect()

    print(f"Total feature tables generated: {len(features_list)}")
    return features_list
