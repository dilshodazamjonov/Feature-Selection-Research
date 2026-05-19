from __future__ import annotations

import math

import numpy as np
import pandas as pd

from credit_risk_fs.preprocessing.lendingclub import LENDINGCLUB_RAW_DATE_STRING_COLUMNS

NUMERIC_CAST_COLUMNS = (
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "collections_12_mths_ex_med",
    "mths_since_last_major_derog",
    "annual_inc_joint",
    "dti_joint",
    "acc_now_delinq",
    "tot_coll_amt",
    "tot_cur_bal",
    "open_acc_6m",
    "open_act_il",
    "open_il_12m",
    "open_il_24m",
    "mths_since_rcnt_il",
    "total_bal_il",
    "il_util",
    "open_rv_12m",
    "open_rv_24m",
    "max_bal_bc",
    "all_util",
    "total_rev_hi_lim",
    "inq_fi",
    "total_cu_tl",
    "inq_last_12m",
    "acc_open_past_24mths",
    "avg_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "chargeoff_within_12_mths",
    "delinq_amnt",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mort_acc",
    "mths_since_recent_bc",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_inq",
    "mths_since_recent_revol_delinq",
    "num_accts_ever_120_pd",
    "num_actv_bc_tl",
    "num_actv_rev_tl",
    "num_bc_sats",
    "num_bc_tl",
    "num_il_tl",
    "num_op_rev_tl",
    "num_rev_accts",
    "num_rev_tl_bal_gt_0",
    "num_sats",
    "num_tl_120dpd_2m",
    "num_tl_30dpd",
    "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m",
    "pct_tl_nvr_dlq",
    "percent_bc_gt_75",
    "pub_rec_bankruptcies",
    "tax_liens",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
    "revol_bal_joint",
    "sec_app_fico_range_low",
    "sec_app_fico_range_high",
    "sec_app_inq_last_6mths",
    "sec_app_mort_acc",
    "sec_app_open_acc",
    "sec_app_revol_util",
    "sec_app_open_act_il",
    "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog",
    "TARGET",
    "recent_decision",
)


def _as_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float32")
    return pd.to_numeric(df[column], errors="coerce")


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    scale: float = 1.0,
) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    result = num * scale / den.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).astype("float32")


def _add_numeric_feature(df: pd.DataFrame, column: str, values: pd.Series) -> None:
    df[column] = pd.to_numeric(values, errors="coerce").astype("float32")


def _band_feature(
    df: pd.DataFrame,
    source_col: str,
    new_col: str,
    bins: list[float],
    labels: list[str],
) -> None:
    if source_col not in df.columns:
        return
    source = pd.to_numeric(df[source_col], errors="coerce")
    df[new_col] = pd.cut(
        source,
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    ).astype("category")


def _parse_emp_length(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    parsed = pd.Series(np.nan, index=series.index, dtype="float32")
    parsed = parsed.mask(values.eq("< 1 year"), 0.5)
    parsed = parsed.mask(values.eq("10+ years"), 10.0)
    for year in range(1, 10):
        parsed = parsed.mask(values.eq(f"{year} year"), float(year))
        parsed = parsed.mask(values.eq(f"{year} years"), float(year))
    return parsed


def _months_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    valid = later.notna() & earlier.notna()
    months = pd.Series(np.nan, index=later.index, dtype="float32")
    delta = (
        (later.dt.year - earlier.dt.year) * 12
        + (later.dt.month - earlier.dt.month)
    ).astype("float32")
    months.loc[valid] = delta.loc[valid]
    months = months.clip(lower=0)
    return months


def _add_missing_flags(
    df: pd.DataFrame,
    protected_columns: set[str],
    *,
    candidate_columns: list[str] | tuple[str, ...] | None = None,
) -> None:
    columns = list(candidate_columns) if candidate_columns is not None else list(df.columns)
    for column in columns:
        if column in protected_columns:
            continue
        missing_rate = float(df[column].isna().mean())
        if 0.0 < missing_rate < 1.0:
            df[f"{column}_missing_flag"] = df[column].isna().astype("int8")


def _feature_count_summary(df: pd.DataFrame) -> int:
    return int(len(df.columns))


def build_application_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    source_columns = list(features.columns)
    for column in NUMERIC_CAST_COLUMNS:
        if column in features.columns:
            features[column] = pd.to_numeric(features[column], errors="coerce")

    issue_date = pd.to_datetime(features["issue_d"], errors="coerce") if "issue_d" in features.columns else pd.Series(pd.NaT, index=features.index)
    earliest_cr_line = (
        pd.to_datetime(features["earliest_cr_line"], format="%b-%Y", errors="coerce")
        if "earliest_cr_line" in features.columns
        else pd.Series(pd.NaT, index=features.index)
    )
    sec_app_earliest = (
        pd.to_datetime(features["sec_app_earliest_cr_line"], format="%b-%Y", errors="coerce")
        if "sec_app_earliest_cr_line" in features.columns
        else pd.Series(pd.NaT, index=features.index)
    )

    loan_amnt = _as_numeric(features, "loan_amnt")
    annual_inc = _as_numeric(features, "annual_inc")
    dti = _as_numeric(features, "dti")
    total_acc = _as_numeric(features, "total_acc")
    open_acc = _as_numeric(features, "open_acc")
    revol_bal = _as_numeric(features, "revol_bal")
    revol_util = _as_numeric(features, "revol_util")
    bc_util = _as_numeric(features, "bc_util")
    all_util = _as_numeric(features, "all_util")
    il_util = _as_numeric(features, "il_util")
    tot_cur_bal = _as_numeric(features, "tot_cur_bal")
    total_bal_ex_mort = _as_numeric(features, "total_bal_ex_mort")
    total_bc_limit = _as_numeric(features, "total_bc_limit")
    total_rev_hi_lim = _as_numeric(features, "total_rev_hi_lim")
    total_il_high_credit_limit = _as_numeric(features, "total_il_high_credit_limit")
    tot_hi_cred_lim = _as_numeric(features, "tot_hi_cred_lim")
    bc_open_to_buy = _as_numeric(features, "bc_open_to_buy")
    total_bal_il = _as_numeric(features, "total_bal_il")
    mort_acc = _as_numeric(features, "mort_acc")
    inq_last_6mths = _as_numeric(features, "inq_last_6mths")
    inq_last_12m = _as_numeric(features, "inq_last_12m")
    inq_fi = _as_numeric(features, "inq_fi")
    delinq_2yrs = _as_numeric(features, "delinq_2yrs")
    pub_rec = _as_numeric(features, "pub_rec")
    pub_rec_bankruptcies = _as_numeric(features, "pub_rec_bankruptcies")
    acc_open_past_24mths = _as_numeric(features, "acc_open_past_24mths")
    num_tl_op_past_12m = _as_numeric(features, "num_tl_op_past_12m")
    num_tl_90g_dpd_24m = _as_numeric(features, "num_tl_90g_dpd_24m")
    num_tl_30dpd = _as_numeric(features, "num_tl_30dpd")
    num_accts_ever_120_pd = _as_numeric(features, "num_accts_ever_120_pd")
    num_bc_tl = _as_numeric(features, "num_bc_tl")
    num_bc_sats = _as_numeric(features, "num_bc_sats")
    num_il_tl = _as_numeric(features, "num_il_tl")
    num_op_rev_tl = _as_numeric(features, "num_op_rev_tl")
    num_rev_accts = _as_numeric(features, "num_rev_accts")
    num_rev_tl_bal_gt_0 = _as_numeric(features, "num_rev_tl_bal_gt_0")
    num_sats = _as_numeric(features, "num_sats")
    open_act_il = _as_numeric(features, "open_act_il")
    open_acc_6m = _as_numeric(features, "open_acc_6m")
    open_il_12m = _as_numeric(features, "open_il_12m")
    open_il_24m = _as_numeric(features, "open_il_24m")
    open_rv_12m = _as_numeric(features, "open_rv_12m")
    open_rv_24m = _as_numeric(features, "open_rv_24m")
    max_bal_bc = _as_numeric(features, "max_bal_bc")
    avg_cur_bal = _as_numeric(features, "avg_cur_bal")
    pct_tl_nvr_dlq = _as_numeric(features, "pct_tl_nvr_dlq")
    percent_bc_gt_75 = _as_numeric(features, "percent_bc_gt_75")
    fico_low = _as_numeric(features, "fico_range_low")
    fico_high = _as_numeric(features, "fico_range_high")
    annual_inc_joint = _as_numeric(features, "annual_inc_joint")
    dti_joint = _as_numeric(features, "dti_joint")
    sec_fico_low = _as_numeric(features, "sec_app_fico_range_low")
    sec_fico_high = _as_numeric(features, "sec_app_fico_range_high")
    sec_open_acc = _as_numeric(features, "sec_app_open_acc")
    sec_mort_acc = _as_numeric(features, "sec_app_mort_acc")
    sec_revol_util = _as_numeric(features, "sec_app_revol_util")
    sec_open_act_il = _as_numeric(features, "sec_app_open_act_il")
    sec_num_rev_accts = _as_numeric(features, "sec_app_num_rev_accts")
    sec_inq_last_6mths = _as_numeric(features, "sec_app_inq_last_6mths")

    if "term" in features.columns:
        _add_numeric_feature(
            features,
            "term_months",
            features["term"].astype("string").str.extract(r"(\d+)")[0],
        )
    if "emp_length" in features.columns:
        _add_numeric_feature(features, "emp_length_years", _parse_emp_length(features["emp_length"]))

    _add_numeric_feature(features, "issue_year", issue_date.dt.year)
    _add_numeric_feature(features, "issue_month", issue_date.dt.month)
    _add_numeric_feature(features, "issue_quarter", issue_date.dt.quarter)
    _add_numeric_feature(features, "issue_month_sin", np.sin(2 * math.pi * issue_date.dt.month.fillna(0) / 12))
    _add_numeric_feature(features, "issue_month_cos", np.cos(2 * math.pi * issue_date.dt.month.fillna(0) / 12))
    _add_numeric_feature(features, "credit_history_months", _months_between(issue_date, earliest_cr_line))
    _add_numeric_feature(features, "credit_history_years", _safe_ratio(features["credit_history_months"], pd.Series(12, index=features.index)))
    _add_numeric_feature(features, "sec_app_credit_history_months", _months_between(issue_date, sec_app_earliest))
    _add_numeric_feature(features, "sec_app_credit_history_years", _safe_ratio(features["sec_app_credit_history_months"], pd.Series(12, index=features.index)))

    _add_numeric_feature(features, "fico_mean", (fico_low + fico_high) / 2.0)
    _add_numeric_feature(features, "fico_span", fico_high - fico_low)
    _add_numeric_feature(features, "sec_app_fico_mean", (sec_fico_low + sec_fico_high) / 2.0)
    _add_numeric_feature(features, "sec_app_fico_span", sec_fico_high - sec_fico_low)

    _add_numeric_feature(features, "log_loan_amnt", np.log1p(loan_amnt.clip(lower=0)))
    _add_numeric_feature(features, "log_annual_inc", np.log1p(annual_inc.clip(lower=0)))
    _add_numeric_feature(features, "log_revol_bal", np.log1p(revol_bal.clip(lower=0)))
    _add_numeric_feature(features, "log_tot_cur_bal", np.log1p(tot_cur_bal.clip(lower=0)))
    _add_numeric_feature(features, "log_total_bal_ex_mort", np.log1p(total_bal_ex_mort.clip(lower=0)))

    _add_numeric_feature(features, "loan_to_income", _safe_ratio(loan_amnt, annual_inc, scale=12.0))
    _add_numeric_feature(features, "revol_bal_to_income", _safe_ratio(revol_bal, annual_inc, scale=12.0))
    _add_numeric_feature(features, "tot_cur_bal_to_income", _safe_ratio(tot_cur_bal, annual_inc, scale=12.0))
    _add_numeric_feature(features, "total_bal_ex_mort_to_income", _safe_ratio(total_bal_ex_mort, annual_inc, scale=12.0))
    _add_numeric_feature(features, "total_bc_limit_to_income", _safe_ratio(total_bc_limit, annual_inc, scale=12.0))
    _add_numeric_feature(features, "total_rev_hi_lim_to_income", _safe_ratio(total_rev_hi_lim, annual_inc, scale=12.0))
    _add_numeric_feature(features, "total_il_high_credit_limit_to_income", _safe_ratio(total_il_high_credit_limit, annual_inc, scale=12.0))
    _add_numeric_feature(features, "tot_hi_cred_lim_to_income", _safe_ratio(tot_hi_cred_lim, annual_inc, scale=12.0))
    _add_numeric_feature(features, "joint_income_to_single_income", _safe_ratio(annual_inc_joint, annual_inc))

    _add_numeric_feature(features, "revol_bal_per_open_acc", _safe_ratio(revol_bal, open_acc))
    _add_numeric_feature(features, "revol_bal_per_total_acc", _safe_ratio(revol_bal, total_acc))
    _add_numeric_feature(features, "tot_cur_bal_per_open_acc", _safe_ratio(tot_cur_bal, open_acc))
    _add_numeric_feature(features, "total_bal_ex_mort_per_open_acc", _safe_ratio(total_bal_ex_mort, open_acc))
    _add_numeric_feature(features, "total_bc_limit_per_bc_trade", _safe_ratio(total_bc_limit, num_bc_tl))
    _add_numeric_feature(features, "total_rev_hi_lim_per_rev_trade", _safe_ratio(total_rev_hi_lim, num_rev_accts))
    _add_numeric_feature(features, "loan_per_total_acc", _safe_ratio(loan_amnt, total_acc))
    _add_numeric_feature(features, "loan_per_open_acc", _safe_ratio(loan_amnt, open_acc))
    _add_numeric_feature(features, "avg_cur_bal_gap", avg_cur_bal - _as_numeric(features, "tot_cur_bal_per_open_acc"))
    _add_numeric_feature(features, "bc_open_to_buy_share", _safe_ratio(bc_open_to_buy, total_bc_limit))
    _add_numeric_feature(features, "revol_balance_to_total_balance", _safe_ratio(revol_bal, total_bal_ex_mort))
    _add_numeric_feature(features, "il_balance_to_total_balance", _safe_ratio(total_bal_il, total_bal_ex_mort))
    _add_numeric_feature(features, "mort_balance_pressure", _safe_ratio(total_bal_ex_mort, mort_acc + 1))

    _add_numeric_feature(features, "inq_6m_per_open_acc", _safe_ratio(inq_last_6mths, open_acc))
    _add_numeric_feature(features, "inq_12m_per_open_acc", _safe_ratio(inq_last_12m, open_acc))
    _add_numeric_feature(features, "delinq_2yrs_per_total_acc", _safe_ratio(delinq_2yrs, total_acc))
    _add_numeric_feature(features, "pub_rec_per_total_acc", _safe_ratio(pub_rec, total_acc))
    _add_numeric_feature(features, "acc_open_past_24mths_share", _safe_ratio(acc_open_past_24mths, total_acc))
    _add_numeric_feature(features, "num_tl_op_past_12m_share", _safe_ratio(num_tl_op_past_12m, total_acc))
    _add_numeric_feature(features, "open_installment_share", _safe_ratio(open_act_il, open_acc))
    _add_numeric_feature(features, "open_revolving_share", _safe_ratio(num_op_rev_tl, total_acc))
    _add_numeric_feature(features, "active_revolving_share", _safe_ratio(num_rev_tl_bal_gt_0, num_rev_accts))
    _add_numeric_feature(features, "num_bc_sats_share", _safe_ratio(num_bc_sats, num_bc_tl))
    _add_numeric_feature(features, "num_sats_share", _safe_ratio(num_sats, total_acc))
    _add_numeric_feature(features, "credit_lines_per_history_year", _safe_ratio(total_acc, features["credit_history_years"]))

    _add_numeric_feature(features, "bc_util_minus_revol_util", bc_util - revol_util)
    _add_numeric_feature(features, "all_util_minus_bc_util", all_util - bc_util)
    _add_numeric_feature(features, "all_util_minus_il_util", all_util - il_util)
    utilization_frame = pd.concat([revol_util, bc_util, all_util], axis=1)
    _add_numeric_feature(features, "utilization_pressure", utilization_frame.mean(axis=1, skipna=True))
    _add_numeric_feature(features, "loan_to_total_limit", _safe_ratio(loan_amnt, total_rev_hi_lim + total_bc_limit + total_il_high_credit_limit))
    _add_numeric_feature(features, "loan_to_total_balance", _safe_ratio(loan_amnt, total_bal_ex_mort + total_bal_il))
    _add_numeric_feature(features, "total_bc_limit_to_total_rev_limit", _safe_ratio(total_bc_limit, total_rev_hi_lim))
    _add_numeric_feature(features, "revolving_capacity_gap", total_rev_hi_lim - revol_bal)
    _add_numeric_feature(features, "bankcard_capacity_gap", total_bc_limit - max_bal_bc)
    _add_numeric_feature(features, "balance_to_high_credit_limit", _safe_ratio(total_bal_ex_mort, tot_hi_cred_lim))

    _add_numeric_feature(features, "recent_inquiry_density", _safe_ratio(inq_last_6mths + inq_fi + open_acc_6m, open_acc + 1))
    _add_numeric_feature(features, "recent_trade_density", _safe_ratio(open_il_12m + open_rv_12m + num_tl_op_past_12m, total_acc + 1))
    _add_numeric_feature(features, "delinquency_pressure", delinq_2yrs + num_tl_30dpd + num_tl_90g_dpd_24m + num_accts_ever_120_pd)
    _add_numeric_feature(features, "severe_delinquency_count", pub_rec_bankruptcies + num_tl_90g_dpd_24m + num_accts_ever_120_pd)
    _add_numeric_feature(features, "bankruptcy_or_pubrec_count", pub_rec + pub_rec_bankruptcies)
    _add_numeric_feature(features, "joint_dti_gap", dti_joint - dti)

    features["joint_application_flag"] = (
        features["application_type"].astype("string").str.upper().fillna("MISSING").str.contains("JOINT")
        if "application_type" in features.columns
        else pd.Series(False, index=features.index)
    ).astype("int8")
    features["second_app_present_flag"] = sec_fico_low.notna().astype("int8")
    features["has_mortgage_flag"] = mort_acc.fillna(0).gt(0).astype("int8")
    features["has_public_record_flag"] = pub_rec.fillna(0).gt(0).astype("int8")
    features["has_recent_inquiry_flag"] = inq_last_6mths.fillna(0).gt(0).astype("int8")
    features["delinquency_history_flag"] = delinq_2yrs.fillna(0).gt(0).astype("int8")
    features["bankruptcy_history_flag"] = pub_rec_bankruptcies.fillna(0).gt(0).astype("int8")
    features["high_dti_flag"] = dti.gt(20).fillna(False).astype("int8")
    features["very_high_dti_flag"] = dti.gt(30).fillna(False).astype("int8")
    features["low_fico_flag"] = features["fico_mean"].lt(660).fillna(False).astype("int8")
    features["prime_fico_flag"] = features["fico_mean"].ge(720).fillna(False).astype("int8")
    features["high_revol_util_flag"] = revol_util.ge(80).fillna(False).astype("int8")
    features["high_bc_util_flag"] = bc_util.ge(80).fillna(False).astype("int8")
    features["thin_file_flag"] = total_acc.lt(10).fillna(False).astype("int8")
    features["recent_opening_flag"] = acc_open_past_24mths.ge(4).fillna(False).astype("int8")
    features["recent_installment_activity_flag"] = open_il_12m.ge(2).fillna(False).astype("int8")
    features["recent_revolving_activity_flag"] = open_rv_12m.ge(2).fillna(False).astype("int8")
    features["joint_income_available_flag"] = annual_inc_joint.notna().astype("int8")
    features["credit_history_short_flag"] = features["credit_history_months"].lt(36).fillna(False).astype("int8")

    _band_feature(
        features,
        "loan_amnt",
        "loan_amnt_band",
        [0, 5000, 10000, 15000, 25000, 35000, np.inf],
        ["vsmall", "small", "mid", "upper_mid", "large", "xlarge"],
    )
    _band_feature(
        features,
        "annual_inc",
        "annual_inc_band",
        [0, 30000, 50000, 75000, 100000, 150000, np.inf],
        ["low", "lower_mid", "mid", "upper_mid", "high", "very_high"],
    )
    _band_feature(
        features,
        "fico_mean",
        "fico_band",
        [0, 640, 680, 700, 720, 760, 820, np.inf],
        ["subprime", "near_prime", "prime", "prime_plus", "super_prime", "elite", "top"],
    )
    _band_feature(
        features,
        "dti",
        "dti_band",
        [0, 10, 20, 30, 40, 50, np.inf],
        ["very_low", "low", "moderate", "high", "very_high", "extreme"],
    )
    _band_feature(
        features,
        "revol_util",
        "revol_util_band",
        [0, 20, 40, 60, 80, 100, np.inf],
        ["vlow", "low", "mid", "high", "vhigh", "extreme"],
    )
    _band_feature(
        features,
        "bc_util",
        "bc_util_band",
        [0, 20, 40, 60, 80, 100, np.inf],
        ["vlow", "low", "mid", "high", "vhigh", "extreme"],
    )
    _band_feature(
        features,
        "all_util",
        "all_util_band",
        [0, 20, 40, 60, 80, 100, np.inf],
        ["vlow", "low", "mid", "high", "vhigh", "extreme"],
    )
    _band_feature(
        features,
        "il_util",
        "il_util_band",
        [0, 20, 40, 60, 80, 100, np.inf],
        ["vlow", "low", "mid", "high", "vhigh", "extreme"],
    )
    _band_feature(
        features,
        "credit_history_months",
        "credit_history_band",
        [0, 24, 60, 120, 180, 240, np.inf],
        ["new", "young", "seasoned", "mature", "veteran", "deep"],
    )
    _band_feature(
        features,
        "open_acc",
        "open_acc_band",
        [0, 5, 10, 15, 20, 30, np.inf],
        ["very_low", "low", "mid", "upper_mid", "high", "very_high"],
    )
    _band_feature(
        features,
        "total_acc",
        "total_acc_band",
        [0, 10, 20, 30, 40, 60, np.inf],
        ["very_low", "low", "mid", "upper_mid", "high", "very_high"],
    )
    _band_feature(
        features,
        "inq_last_6mths",
        "inq_last_6mths_band",
        [0, 1, 2, 4, 6, np.inf],
        ["none", "light", "moderate", "high", "extreme"],
    )
    _band_feature(
        features,
        "inq_last_12m",
        "inq_last_12m_band",
        [0, 1, 3, 5, 8, 12, np.inf],
        ["none", "light", "moderate", "high", "very_high", "extreme"],
    )
    _band_feature(
        features,
        "pub_rec",
        "pub_rec_band",
        [0, 1, 2, 4, np.inf],
        ["none", "low", "moderate", "high"],
    )
    _band_feature(
        features,
        "delinq_2yrs",
        "delinq_2yrs_band",
        [0, 1, 2, 4, 8, np.inf],
        ["none", "low", "moderate", "high", "extreme"],
    )
    _band_feature(
        features,
        "mort_acc",
        "mort_acc_band",
        [0, 1, 2, 4, 8, np.inf],
        ["none", "light", "moderate", "high", "very_high"],
    )
    _band_feature(
        features,
        "acc_open_past_24mths",
        "acc_open_past_24mths_band",
        [0, 1, 2, 4, 6, 10, np.inf],
        ["none", "light", "moderate", "high", "very_high", "extreme"],
    )
    _band_feature(
        features,
        "num_tl_op_past_12m",
        "num_tl_op_past_12m_band",
        [0, 1, 2, 4, 6, 10, np.inf],
        ["none", "light", "moderate", "high", "very_high", "extreme"],
    )
    _band_feature(
        features,
        "loan_to_income",
        "loan_to_income_band",
        [0, 0.05, 0.10, 0.20, 0.35, 0.50, np.inf],
        ["tiny", "low", "moderate", "high", "very_high", "extreme"],
    )
    _band_feature(
        features,
        "revol_bal_to_income",
        "revol_bal_to_income_band",
        [0, 0.05, 0.10, 0.20, 0.35, 0.50, np.inf],
        ["tiny", "low", "moderate", "high", "very_high", "extreme"],
    )
    _band_feature(
        features,
        "total_bal_ex_mort_to_income",
        "total_bal_to_income_band",
        [0, 0.10, 0.25, 0.50, 0.75, 1.25, np.inf],
        ["tiny", "low", "moderate", "high", "very_high", "extreme"],
    )
    _band_feature(
        features,
        "total_bc_limit_to_income",
        "bc_limit_to_income_band",
        [0, 0.05, 0.10, 0.20, 0.35, 0.60, np.inf],
        ["tiny", "low", "moderate", "high", "very_high", "extreme"],
    )
    _band_feature(
        features,
        "utilization_pressure",
        "utilization_pressure_band",
        [0, 20, 40, 60, 80, 100, np.inf],
        ["vlow", "low", "mid", "high", "vhigh", "extreme"],
    )
    _band_feature(
        features,
        "credit_lines_per_history_year",
        "credit_lines_per_history_band",
        [0, 0.5, 1.0, 1.5, 2.5, 4.0, np.inf],
        ["very_low", "low", "mid", "upper_mid", "high", "very_high"],
    )
    _band_feature(
        features,
        "recent_inquiry_density",
        "recent_inquiry_density_band",
        [0, 0.05, 0.10, 0.20, 0.35, 0.60, np.inf],
        ["tiny", "low", "moderate", "high", "very_high", "extreme"],
    )
    _band_feature(
        features,
        "recent_trade_density",
        "recent_trade_density_band",
        [0, 0.05, 0.10, 0.20, 0.35, 0.60, np.inf],
        ["tiny", "low", "moderate", "high", "very_high", "extreme"],
    )
    _band_feature(
        features,
        "delinquency_pressure",
        "delinquency_pressure_band",
        [0, 1, 2, 4, 8, 12, np.inf],
        ["none", "low", "moderate", "high", "very_high", "extreme"],
    )

    if "term" in features.columns and "home_ownership" in features.columns:
        features["term_home_ownership"] = (
            features["term"].astype("string").fillna("Missing")
            + "__"
            + features["home_ownership"].astype("string").fillna("Missing")
        ).astype("category")
    if "term" in features.columns and "verification_status" in features.columns:
        features["term_verification_status"] = (
            features["term"].astype("string").fillna("Missing")
            + "__"
            + features["verification_status"].astype("string").fillna("Missing")
        ).astype("category")
    if "purpose" in features.columns and "verification_status" in features.columns:
        features["purpose_verification_status"] = (
            features["purpose"].astype("string").fillna("Missing")
            + "__"
            + features["verification_status"].astype("string").fillna("Missing")
        ).astype("category")
    if "application_type" in features.columns and "verification_status" in features.columns:
        features["application_type_verification_status"] = (
            features["application_type"].astype("string").fillna("Missing")
            + "__"
            + features["verification_status"].astype("string").fillna("Missing")
        ).astype("category")

    _add_missing_flags(
        features,
        protected_columns={"TARGET", "recent_decision", "issue_d"},
        candidate_columns=source_columns + [
            "credit_history_months",
            "sec_app_credit_history_months",
            "emp_length_years",
            "fico_mean",
            "sec_app_fico_mean",
        ],
    )
    features = features.drop(
        columns=[column for column in LENDINGCLUB_RAW_DATE_STRING_COLUMNS if column in features.columns],
        errors="ignore",
    )
    features.attrs["engineered_feature_count"] = _feature_count_summary(features)
    return features
