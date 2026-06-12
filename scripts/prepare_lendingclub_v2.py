from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in [PROJECT_ROOT, SRC_ROOT]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from credit_risk_fs.feature_engineering.lendingclub.application import (  # noqa: E402
    build_application_features,
)
from credit_risk_fs.feature_metadata.builder import infer_semantic_group  # noqa: E402
from credit_risk_fs.preprocessing.lendingclub import (  # noqa: E402
    LENDINGCLUB_DIRECT_LEAKAGE_COLUMNS,
    LENDINGCLUB_IDENTIFIER_OR_TEXT_COLUMNS,
    LENDINGCLUB_POLICY_OR_POST_APPROVAL_COLUMNS,
    LENDINGCLUB_POST_OUTCOME_LEAKAGE_COLUMNS,
    LENDINGCLUB_TEXT_OR_LOW_SIGNAL_COLUMNS,
    LENDINGCLUB_UNDERWRITING_POLICY_COLUMNS,
)


DATE_DERIVED_HELPER_COLUMNS = {
    "issue_year",
    "issue_month",
    "issue_quarter",
    "issue_month_sin",
    "issue_month_cos",
}
HELPER_COLUMNS = {
    "TARGET",
    "recent_decision",
    "issue_d",
    "loan_status",
    *DATE_DERIVED_HELPER_COLUMNS,
}
OUTPUT_HELPER_COLUMNS = ["TARGET", "recent_decision", "issue_d"]

RAW_SAFE_DESCRIPTIONS = {
    "loan_amnt": "Requested loan principal amount at application origination.",
    "term": "Loan repayment term requested at origination.",
    "emp_length": "Borrower reported employment length at application time.",
    "home_ownership": "Borrower home-ownership status at application time.",
    "annual_inc": "Borrower annual income reported at application time.",
    "verification_status": "Income verification status assigned during application review.",
    "purpose": "Borrower stated loan purpose at application time.",
    "addr_state": "Borrower state from the application record.",
    "dti": "Borrower debt-to-income ratio available at application time.",
    "delinq_2yrs": "Number of 30+ day delinquencies in the borrower's credit file over the past two years.",
    "earliest_cr_line": "Month when the borrower's earliest reported credit line was opened.",
    "fico_range_low": "Lower bound of the borrower's FICO score range at origination.",
    "fico_range_high": "Upper bound of the borrower's FICO score range at origination.",
    "inq_last_6mths": "Number of borrower credit inquiries in the last six months.",
    "mths_since_last_delinq": "Months since the borrower's most recent delinquency.",
    "mths_since_last_record": "Months since the borrower's most recent public record.",
    "open_acc": "Number of open credit lines in the borrower's credit file.",
    "pub_rec": "Number of derogatory public records in the borrower's credit file.",
    "revol_bal": "Total revolving credit balance at application time.",
    "revol_util": "Revolving credit utilization percentage at application time.",
    "total_acc": "Total number of credit lines in the borrower's credit file.",
    "initial_list_status": "Initial LendingClub listing status at origination.",
    "collections_12_mths_ex_med": "Collections excluding medical collections in the past 12 months.",
    "mths_since_last_major_derog": "Months since the borrower's last major derogatory event.",
    "application_type": "Application type, such as individual or joint.",
    "annual_inc_joint": "Joint applicant annual income when a joint application is present.",
    "dti_joint": "Joint applicant debt-to-income ratio when available.",
    "verification_status_joint": "Income verification status for a joint application.",
    "acc_now_delinq": "Number of accounts currently delinquent at application time.",
    "tot_coll_amt": "Total collection amounts in the credit file.",
    "tot_cur_bal": "Total current balance across credit accounts.",
    "open_acc_6m": "Number of open trades opened in the last six months.",
    "open_act_il": "Number of currently active installment trades.",
    "open_il_12m": "Number of installment accounts opened in the last 12 months.",
    "open_il_24m": "Number of installment accounts opened in the last 24 months.",
    "mths_since_rcnt_il": "Months since the most recent installment account was opened.",
    "total_bal_il": "Total current balance on installment accounts.",
    "il_util": "Installment loan utilization percentage.",
    "open_rv_12m": "Number of revolving trades opened in the last 12 months.",
    "open_rv_24m": "Number of revolving trades opened in the last 24 months.",
    "max_bal_bc": "Maximum current balance on bankcard accounts.",
    "all_util": "Aggregate utilization across open credit lines.",
    "total_rev_hi_lim": "Total revolving high credit or credit limit.",
    "inq_fi": "Number of finance inquiries.",
    "total_cu_tl": "Number of credit union trades.",
    "inq_last_12m": "Number of credit inquiries in the last 12 months.",
    "acc_open_past_24mths": "Number of trades opened in the past 24 months.",
    "avg_cur_bal": "Average current balance across credit accounts.",
    "bc_open_to_buy": "Available bankcard credit capacity.",
    "bc_util": "Bankcard utilization percentage.",
    "chargeoff_within_12_mths": "Charge-offs in the last 12 months reported at application time.",
    "delinq_amnt": "Amount currently delinquent at application time.",
    "mo_sin_old_il_acct": "Months since the oldest installment account was opened.",
    "mo_sin_old_rev_tl_op": "Months since the oldest revolving trade was opened.",
    "mo_sin_rcnt_rev_tl_op": "Months since the most recent revolving trade was opened.",
    "mo_sin_rcnt_tl": "Months since the most recent credit trade was opened.",
    "mort_acc": "Number of mortgage accounts in the credit file.",
    "mths_since_recent_bc": "Months since the most recent bankcard account was opened.",
    "mths_since_recent_bc_dlq": "Months since the most recent bankcard delinquency.",
    "mths_since_recent_inq": "Months since the most recent inquiry.",
    "mths_since_recent_revol_delinq": "Months since the most recent revolving delinquency.",
    "num_accts_ever_120_pd": "Number of accounts ever 120 or more days past due.",
    "num_actv_bc_tl": "Number of currently active bankcard trades.",
    "num_actv_rev_tl": "Number of currently active revolving trades.",
    "num_bc_sats": "Number of satisfactory bankcard accounts.",
    "num_bc_tl": "Number of bankcard trades.",
    "num_il_tl": "Number of installment trades.",
    "num_op_rev_tl": "Number of open revolving trades.",
    "num_rev_accts": "Number of revolving accounts.",
    "num_rev_tl_bal_gt_0": "Number of revolving trades with a balance above zero.",
    "num_sats": "Number of satisfactory accounts.",
    "num_tl_120dpd_2m": "Number of accounts 120+ days past due in the last two months.",
    "num_tl_30dpd": "Number of accounts 30+ days past due.",
    "num_tl_90g_dpd_24m": "Number of accounts 90+ days past due in the last 24 months.",
    "num_tl_op_past_12m": "Number of trades opened in the past 12 months.",
    "pct_tl_nvr_dlq": "Percentage of trades never delinquent.",
    "percent_bc_gt_75": "Percentage of bankcard accounts with utilization above 75 percent.",
    "pub_rec_bankruptcies": "Number of public-record bankruptcies.",
    "tax_liens": "Number of tax liens.",
    "tot_hi_cred_lim": "Total high credit or credit limit across accounts.",
    "total_bal_ex_mort": "Total credit balance excluding mortgage balances.",
    "total_bc_limit": "Total bankcard credit limit.",
    "total_il_high_credit_limit": "Total installment high credit limit.",
    "revol_bal_joint": "Joint revolving balance when a joint application is present.",
    "sec_app_fico_range_low": "Lower FICO range bound for the secondary applicant.",
    "sec_app_fico_range_high": "Upper FICO range bound for the secondary applicant.",
    "sec_app_earliest_cr_line": "Earliest credit line month for the secondary applicant.",
    "sec_app_inq_last_6mths": "Secondary applicant inquiries in the last six months.",
    "sec_app_mort_acc": "Secondary applicant mortgage-account count.",
    "sec_app_open_acc": "Secondary applicant open-account count.",
    "sec_app_revol_util": "Secondary applicant revolving utilization percentage.",
    "sec_app_open_act_il": "Secondary applicant active installment account count.",
    "sec_app_num_rev_accts": "Secondary applicant revolving account count.",
    "sec_app_chargeoff_within_12_mths": "Secondary applicant charge-offs within 12 months reported at application time.",
    "sec_app_collections_12_mths_ex_med": "Secondary applicant non-medical collections in the past 12 months.",
    "sec_app_mths_since_last_major_derog": "Months since secondary applicant last major derogatory event.",
}

AMOUNT_COLUMNS = [
    "loan_amnt",
    "annual_inc",
    "revol_bal",
    "tot_coll_amt",
    "tot_cur_bal",
    "total_bal_il",
    "max_bal_bc",
    "total_rev_hi_lim",
    "avg_cur_bal",
    "bc_open_to_buy",
    "delinq_amnt",
    "tot_hi_cred_lim",
    "total_bal_ex_mort",
    "total_bc_limit",
    "total_il_high_credit_limit",
    "revol_bal_joint",
    "annual_inc_joint",
]
COUNT_COLUMNS = [
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "total_acc",
    "collections_12_mths_ex_med",
    "acc_now_delinq",
    "open_acc_6m",
    "open_act_il",
    "open_il_12m",
    "open_il_24m",
    "open_rv_12m",
    "open_rv_24m",
    "inq_fi",
    "total_cu_tl",
    "inq_last_12m",
    "acc_open_past_24mths",
    "chargeoff_within_12_mths",
    "mort_acc",
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
    "pub_rec_bankruptcies",
    "tax_liens",
    "sec_app_inq_last_6mths",
    "sec_app_mort_acc",
    "sec_app_open_acc",
    "sec_app_open_act_il",
    "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
]
UTILIZATION_COLUMNS = ["revol_util", "bc_util", "all_util", "il_util", "sec_app_revol_util"]
MONTHS_SINCE_COLUMNS = [
    "mths_since_last_delinq",
    "mths_since_last_record",
    "mths_since_last_major_derog",
    "mths_since_rcnt_il",
    "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl",
    "mths_since_recent_bc",
    "mths_since_recent_bc_dlq",
    "mths_since_recent_inq",
    "mths_since_recent_revol_delinq",
    "sec_app_mths_since_last_major_derog",
]
BASE_CATEGORICAL_COLUMNS = [
    "term",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "initial_list_status",
    "application_type",
    "verification_status_joint",
]

REMOVED_HIGH_SEVERITY_OR_SPARSE_FEATURES = {
    "annual_inc_joint_is_zero",
    "loan_amnt_is_zero",
    "loan_amnt_to_loan_amnt",
    "mths_since_recent_inq_seasoned_60m_flag",
    "sec_app_fico_span",
    "total_acc_is_zero",
    "dti_adjusted_sec_app_revol_util",
    "fico_adjusted_sec_app_revol_util",
    "log_annual_inc_joint",
    "sqrt_annual_inc_joint",
    "log_revol_bal_joint",
    "sqrt_revol_bal_joint",
    "revol_bal_joint_to_income",
    "revol_bal_joint_to_loan_amnt",
    "log_sec_app_chargeoff_within_12_mths",
    "log_sec_app_collections_12_mths_ex_med",
    "log_sec_app_inq_last_6mths",
    "log_sec_app_mort_acc",
    "log_sec_app_num_rev_accts",
    "log_sec_app_open_acc",
    "log_sec_app_open_act_il",
    "sec_app_chargeoff_within_12_mths_per_credit_history_year",
    "sec_app_chargeoff_within_12_mths_per_open_acc",
    "sec_app_chargeoff_within_12_mths_per_total_acc",
    "sec_app_collections_12_mths_ex_med_per_credit_history_year",
    "sec_app_collections_12_mths_ex_med_per_open_acc",
    "sec_app_collections_12_mths_ex_med_per_total_acc",
    "sec_app_inq_last_6mths_per_credit_history_year",
    "sec_app_inq_last_6mths_per_open_acc",
    "sec_app_inq_last_6mths_per_total_acc",
    "sec_app_mort_acc_per_credit_history_year",
    "sec_app_mort_acc_per_open_acc",
    "sec_app_mort_acc_per_total_acc",
    "sec_app_mths_since_last_major_derog_inverse_recency",
    "sec_app_num_rev_accts_per_credit_history_year",
    "sec_app_num_rev_accts_per_open_acc",
    "sec_app_num_rev_accts_per_total_acc",
    "sec_app_open_acc_per_credit_history_year",
    "sec_app_open_acc_per_open_acc",
    "sec_app_open_acc_per_total_acc",
    "sec_app_open_act_il_per_credit_history_year",
    "sec_app_open_act_il_per_open_acc",
    "sec_app_open_act_il_per_total_acc",
}

EXTREME_RATIO_CLIP_CAPS = {
    "fico_to_income": (0.0, 10_000.0),
    "loan_to_income_x_fico": (0.0, 85_000.0),
    "term_x_loan_to_income": (0.0, 6_000.0),
    "tot_cur_bal_to_income": (0.0, 100.0),
    "tot_hi_cred_lim_to_income": (0.0, 100.0),
    "total_bal_ex_mort_to_income": (0.0, 100.0),
    "total_bal_il_to_income": (0.0, 100.0),
    "total_bc_limit_to_income": (0.0, 100.0),
    "total_il_high_credit_limit_to_income": (0.0, 100.0),
    "total_rev_hi_lim_per_rev_trade": (0.0, 250_000.0),
    "total_rev_hi_lim_to_income": (0.0, 100.0),
}


@dataclass
class FeatureInfo:
    feature: str
    description: str
    source_column_or_formula: str
    semantic_group: str
    feature_type: str
    leakage_review_status: str = "safe"
    availability_timing: str = "derived_from_safe_fields"
    notes: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare isolated LendingClub v2 engineered data and metadata only."
    )
    parser.add_argument(
        "--input-file",
        default="data/lendingclub/processed/application_train.csv",
        help="Existing safe LendingClub v1 processed source.",
    )
    parser.add_argument("--output-dir", default="data/lendingclub_v2/processed")
    parser.add_argument("--metadata-dir", default="data/lendingclub_v2/metadata")
    parser.add_argument("--reports-dir", default="reports")
    return parser


def _as_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float32")
    return pd.to_numeric(df[column], errors="coerce")


def _safe_ratio(numerator: pd.Series, denominator: pd.Series, scale: float = 1.0) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    out = pd.to_numeric(numerator, errors="coerce") * scale / den
    return out.replace([np.inf, -np.inf], np.nan).astype("float32")


def _log1p_nonnegative(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").clip(lower=0)
    return np.log1p(numeric).astype("float32")


def _sqrt_nonnegative(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").clip(lower=0)
    return np.sqrt(numeric).astype("float32")


def _bounded(series: pd.Series, lower: float | None = None, upper: float | None = None) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.clip(lower=lower, upper=upper).astype("float32")


def _zfill_category(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace({"": pd.NA}).fillna("Missing")


def _band(
    series: pd.Series,
    bins: list[float],
    labels: list[str],
) -> pd.Series:
    return pd.cut(
        pd.to_numeric(series, errors="coerce"),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    ).astype("string").fillna("Missing").astype("category")


def _state_region(series: pd.Series) -> pd.Series:
    regions = {
        "northeast": {"CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"},
        "midwest": {"IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"},
        "south": {"DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY", "MS", "TN", "AR", "LA", "OK", "TX"},
        "west": {"AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"},
    }
    mapping = {state: region for region, states in regions.items() for state in states}
    return _zfill_category(series).map(mapping).fillna("other_or_missing").astype("category")


def _purpose_group(series: pd.Series) -> pd.Series:
    mapping = {
        "debt_consolidation": "debt_refinance",
        "credit_card": "debt_refinance",
        "home_improvement": "asset_or_home",
        "house": "asset_or_home",
        "car": "asset_or_home",
        "major_purchase": "large_purchase",
        "medical": "life_event",
        "moving": "life_event",
        "vacation": "discretionary",
        "wedding": "life_event",
        "small_business": "business",
        "renewable_energy": "asset_or_home",
        "educational": "education",
        "other": "other",
    }
    return _zfill_category(series).str.lower().map(mapping).fillna("other_or_missing").astype("category")


def _home_ownership_group(series: pd.Series) -> pd.Series:
    normalized = _zfill_category(series).str.upper()
    mapping = {
        "MORTGAGE": "mortgage",
        "OWN": "own",
        "RENT": "rent",
        "OTHER": "other",
        "NONE": "other",
        "ANY": "other",
        "MISSING": "missing",
    }
    return normalized.map(mapping).fillna("other").astype("category")


def _verification_group(series: pd.Series) -> pd.Series:
    normalized = _zfill_category(series).str.upper()
    mapping = {
        "VERIFIED": "verified",
        "SOURCE VERIFIED": "source_verified",
        "NOT VERIFIED": "not_verified",
        "MISSING": "missing",
    }
    return normalized.map(mapping).fillna("other").astype("category")


def _emp_length_group(series: pd.Series) -> pd.Series:
    parsed = series.astype("string").str.lower().str.extract(r"(\d+)")[0]
    years = pd.to_numeric(parsed, errors="coerce")
    years = years.mask(series.astype("string").str.lower().str.contains("< 1", na=False), 0)
    years = years.mask(series.astype("string").str.lower().str.contains("10\\+", na=False), 10)
    return _band(
        years,
        [0, 1, 3, 5, 10, np.inf],
        ["under_1", "1_to_2", "3_to_4", "5_to_9", "10_plus"],
    )


def _feature_type(feature: str, dtype: str) -> str:
    if feature.endswith("_missing_flag"):
        return "missing_indicator"
    if feature.endswith("_flag") or feature.startswith("has_") or feature.endswith("_is_zero"):
        return "flag"
    if feature.endswith("_band") or feature.endswith("_group") or feature.endswith("_bucket"):
        return "binned"
    if "_to_" in feature or feature.endswith("_share") or feature.endswith("_ratio"):
        return "ratio"
    if "_x_" in feature or feature.startswith("interaction_"):
        return "interaction"
    if dtype in {"object", "string", "category"}:
        return "categorical"
    if feature.startswith("log_") or feature.startswith("sqrt_") or feature.startswith("capped_"):
        return "engineered"
    return "raw"


def _availability(feature: str) -> str:
    lower = feature.lower()
    if any(token in lower for token in ["fico", "inq", "delinq", "revol", "bc_", "il_", "mort", "pub_rec", "credit", "acc", "tl", "bal", "util"]):
        return "historical_credit_file"
    if any(token in lower for token in ["loan", "term", "purpose", "home_ownership", "verification", "annual_inc", "dti", "addr_state", "application_type"]):
        return "application_time"
    return "derived_from_safe_fields"


def _describe_feature(feature: str, dtype: str, source_columns: set[str]) -> tuple[str, str, str]:
    if feature in RAW_SAFE_DESCRIPTIONS:
        return RAW_SAFE_DESCRIPTIONS[feature], feature, _feature_type(feature, dtype)
    if feature in source_columns:
        readable = feature.replace("_", " ")
        return f"Application-time LendingClub field `{feature}` ({readable}).", feature, _feature_type(feature, dtype)
    if feature.startswith("log_"):
        base = feature.removeprefix("log_")
        return f"Log-transformed nonnegative value of `{base}`.", f"log1p(max({base}, 0))", "engineered"
    if feature.startswith("sqrt_"):
        base = feature.removeprefix("sqrt_")
        return f"Square-root transformed nonnegative value of `{base}`.", f"sqrt(max({base}, 0))", "engineered"
    if feature.startswith("capped_"):
        base = feature.removeprefix("capped_")
        return f"Domain-capped value of `{base}` to limit impossible/extreme scale without fitting on outcomes.", f"domain_cap({base})", "engineered"
    if feature.endswith("_missing_flag"):
        base = feature.removesuffix("_missing_flag")
        return f"Indicator that `{base}` is missing at application time.", f"is_missing({base})", "missing_indicator"
    if feature.endswith("_is_zero"):
        base = feature.removesuffix("_is_zero")
        return f"Indicator that `{base}` is zero.", f"{base} == 0", "flag"
    if feature.endswith("_positive_flag"):
        base = feature.removesuffix("_positive_flag")
        return f"Indicator that `{base}` is positive.", f"{base} > 0", "flag"
    if feature.endswith("_band"):
        base = feature.removesuffix("_band")
        return f"Fixed business-rule band for `{base}`.", f"fixed_bins({base})", "binned"
    if feature.endswith("_group"):
        base = feature.removesuffix("_group")
        return f"Grouped categorical representation of `{base}`.", f"group({base})", "categorical"
    if "_to_" in feature:
        return f"Ratio measuring `{feature.replace('_to_', '` relative to `')}`.", feature.replace("_to_", " / "), "ratio"
    if "_per_" in feature:
        return f"Per-unit ratio for `{feature.replace('_per_', '` per `')}`.", feature.replace("_per_", " / "), "ratio"
    if "_x_" in feature:
        return f"Interpretable interaction between `{feature.replace('_x_', '` and `')}`.", feature.replace("_x_", " * "), "interaction"
    readable = feature.replace("_", " ")
    return f"Engineered LendingClub application-time credit-risk feature `{feature}` ({readable}).", f"derived_from_safe_fields({feature})", _feature_type(feature, dtype)


def _add_feature(
    additions: dict[str, pd.Series],
    registry: dict[str, FeatureInfo],
    name: str,
    values: pd.Series,
    *,
    description: str,
    formula: str,
    semantic_group: str | None = None,
    feature_type: str = "engineered",
    availability_timing: str = "derived_from_safe_fields",
    notes: str = "",
) -> None:
    if name in registry:
        return
    additions[name] = values
    group = semantic_group or infer_semantic_group(name, description=description)
    if not group or group == "other":
        group = _fallback_semantic_group(name)
    registry[name] = FeatureInfo(
        feature=name,
        description=description,
        source_column_or_formula=formula,
        semantic_group=group,
        feature_type=feature_type,
        availability_timing=availability_timing,
        notes=notes,
    )


def _fallback_semantic_group(feature: str) -> str:
    lower = feature.lower()
    if "fico" in lower:
        return "fico_credit_score"
    if any(token in lower for token in ["income", "dti", "affordability"]):
        return "income_capacity"
    if any(token in lower for token in ["loan", "term", "purpose", "home_ownership", "verification"]):
        return "loan_terms"
    if any(token in lower for token in ["revol", "util", "bc_", "bankcard"]):
        return "revolving_utilization"
    if any(token in lower for token in ["inq", "recent"]):
        return "recent_inquiries"
    if any(token in lower for token in ["delinq", "derog", "chargeoff", "bankrupt", "tax_liens", "pub_rec"]):
        return "delinquency_derogatory"
    if any(token in lower for token in ["mort", "home"]):
        return "mortgage_history"
    if any(token in lower for token in ["acc", "tl", "trade", "credit_history"]):
        return "account_mix_credit_depth"
    if any(token in lower for token in ["bal", "limit", "capacity"]):
        return "balance_credit_limit_pressure"
    if any(token in lower for token in ["sec_app", "joint"]):
        return "joint_applicant"
    if "missing" in lower:
        return "missingness_or_unknown"
    return "application_profile"


def _register_existing_features(features: pd.DataFrame, source_columns: Iterable[str]) -> dict[str, FeatureInfo]:
    source_set = set(source_columns)
    registry: dict[str, FeatureInfo] = {}
    for feature in features.columns:
        if feature in HELPER_COLUMNS:
            continue
        dtype = str(features[feature].dtype)
        desc, formula, feature_type = _describe_feature(feature, dtype, source_set)
        group = infer_semantic_group(feature, description=desc)
        if not group or group == "other":
            group = _fallback_semantic_group(feature)
        registry[feature] = FeatureInfo(
            feature=feature,
            description=desc,
            source_column_or_formula=formula,
            semantic_group=group,
            feature_type=feature_type,
            availability_timing=_availability(feature),
            notes="Carried from safe LendingClub v1 source or v1 feature builder.",
        )
    return registry


def add_v2_features(features: pd.DataFrame, registry: dict[str, FeatureInfo]) -> pd.DataFrame:
    additions: dict[str, pd.Series] = {}
    annual_inc = _as_numeric(features, "annual_inc")
    loan_amnt = _as_numeric(features, "loan_amnt")
    dti = _as_numeric(features, "dti")
    fico_mean = _as_numeric(features, "fico_mean")
    open_acc = _as_numeric(features, "open_acc")
    total_acc = _as_numeric(features, "total_acc")
    credit_history_years = _as_numeric(features, "credit_history_years")
    total_credit_limit = (
        _as_numeric(features, "total_rev_hi_lim")
        + _as_numeric(features, "total_bc_limit")
        + _as_numeric(features, "total_il_high_credit_limit")
    )
    total_balance = _as_numeric(features, "total_bal_ex_mort") + _as_numeric(features, "total_bal_il")

    for column in AMOUNT_COLUMNS:
        if column not in features.columns:
            continue
        values = _as_numeric(features, column)
        if f"log_{column}" not in features.columns:
            _add_feature(
                additions,
                registry,
                f"log_{column}",
                _log1p_nonnegative(values),
                description=f"Log-transformed nonnegative value of `{column}` to represent diminishing marginal credit-risk scale.",
                formula=f"log1p(max({column}, 0))",
                semantic_group=_fallback_semantic_group(column),
            )
        _add_feature(
            additions,
            registry,
            f"sqrt_{column}",
            _sqrt_nonnegative(values),
            description=f"Square-root transformed nonnegative value of `{column}` for a moderated scale of the same application-time signal.",
            formula=f"sqrt(max({column}, 0))",
            semantic_group=_fallback_semantic_group(column),
        )
        _add_feature(
            additions,
            registry,
            f"{column}_is_zero",
            values.fillna(np.nan).eq(0).astype("int8"),
            description=f"Indicator that `{column}` is exactly zero, separating no-exposure cases from positive balances or limits.",
            formula=f"{column} == 0",
            semantic_group=_fallback_semantic_group(column),
            feature_type="flag",
            availability_timing=_availability(column),
        )
        if column not in {"annual_inc", "annual_inc_joint"}:
            _add_feature(
                additions,
                registry,
                f"{column}_to_income",
                _safe_ratio(values, annual_inc, scale=12.0),
                description=f"`{column}` scaled by borrower annual income to measure affordability or balance pressure.",
                formula=f"12 * {column} / annual_inc",
                semantic_group="income_capacity",
                feature_type="ratio",
            )
            _add_feature(
                additions,
                registry,
                f"{column}_to_loan_amnt",
                _safe_ratio(values, loan_amnt),
                description=f"`{column}` relative to requested loan amount.",
                formula=f"{column} / loan_amnt",
                semantic_group="balance_credit_limit_pressure",
                feature_type="ratio",
            )

    for column in COUNT_COLUMNS:
        if column not in features.columns:
            continue
        values = _as_numeric(features, column)
        _add_feature(
            additions,
            registry,
            f"log_{column}",
            _log1p_nonnegative(values),
            description=f"Log-transformed count for `{column}` to reduce domination by high-count tails.",
            formula=f"log1p(max({column}, 0))",
            semantic_group=_fallback_semantic_group(column),
        )
        _add_feature(
            additions,
            registry,
            f"{column}_is_zero",
            values.eq(0).fillna(False).astype("int8"),
            description=f"Indicator that `{column}` is zero, capturing absence of this credit-file event or account type.",
            formula=f"{column} == 0",
            semantic_group=_fallback_semantic_group(column),
            feature_type="flag",
            availability_timing=_availability(column),
        )
        _add_feature(
            additions,
            registry,
            f"{column}_positive_flag",
            values.gt(0).fillna(False).astype("int8"),
            description=f"Indicator that `{column}` is positive, capturing presence of this credit-file event or account type.",
            formula=f"{column} > 0",
            semantic_group=_fallback_semantic_group(column),
            feature_type="flag",
            availability_timing=_availability(column),
        )
        if column not in {"open_acc", "total_acc"}:
            _add_feature(
                additions,
                registry,
                f"{column}_per_open_acc",
                _safe_ratio(values, open_acc),
                description=f"`{column}` divided by open account count, measuring concentration among active accounts.",
                formula=f"{column} / open_acc",
                semantic_group=_fallback_semantic_group(column),
                feature_type="ratio",
            )
            _add_feature(
                additions,
                registry,
                f"{column}_per_total_acc",
                _safe_ratio(values, total_acc),
                description=f"`{column}` divided by total account count, measuring concentration in the full credit file.",
                formula=f"{column} / total_acc",
                semantic_group=_fallback_semantic_group(column),
                feature_type="ratio",
            )
            _add_feature(
                additions,
                registry,
                f"{column}_per_credit_history_year",
                _safe_ratio(values, credit_history_years),
                description=f"`{column}` divided by credit-history years, measuring event density over file age.",
                formula=f"{column} / credit_history_years",
                semantic_group=_fallback_semantic_group(column),
                feature_type="ratio",
            )

    for column in UTILIZATION_COLUMNS:
        if column not in features.columns:
            continue
        values = _as_numeric(features, column)
        for threshold in [30, 50, 75, 90, 100]:
            _add_feature(
                additions,
                registry,
                f"{column}_ge_{threshold}_flag",
                values.ge(threshold).fillna(False).astype("int8"),
                description=f"Indicator that `{column}` is at least {threshold} percent.",
                formula=f"{column} >= {threshold}",
                semantic_group="revolving_utilization",
                feature_type="flag",
            )
        _add_feature(
            additions,
            registry,
            f"fico_adjusted_{column}",
            _safe_ratio(values, fico_mean, scale=700.0),
            description=f"`{column}` adjusted by FICO score level to combine utilization pressure with borrower credit score strength.",
            formula=f"700 * {column} / fico_mean",
            semantic_group="revolving_utilization",
            feature_type="interaction",
        )
        _add_feature(
            additions,
            registry,
            f"dti_adjusted_{column}",
            (values * (1.0 + dti.fillna(0) / 100.0)).astype("float32"),
            description=f"`{column}` scaled by debt-to-income pressure.",
            formula=f"{column} * (1 + dti / 100)",
            semantic_group="income_capacity",
            feature_type="interaction",
        )

    for column in MONTHS_SINCE_COLUMNS:
        if column not in features.columns:
            continue
        values = _as_numeric(features, column)
        _add_feature(
            additions,
            registry,
            f"{column}_recent_12m_flag",
            values.le(12).fillna(False).astype("int8"),
            description=f"Indicator that `{column}` is within the past 12 months.",
            formula=f"{column} <= 12",
            semantic_group=_fallback_semantic_group(column),
            feature_type="flag",
        )
        _add_feature(
            additions,
            registry,
            f"{column}_recent_24m_flag",
            values.le(24).fillna(False).astype("int8"),
            description=f"Indicator that `{column}` is within the past 24 months.",
            formula=f"{column} <= 24",
            semantic_group=_fallback_semantic_group(column),
            feature_type="flag",
        )
        _add_feature(
            additions,
            registry,
            f"{column}_seasoned_60m_flag",
            values.ge(60).fillna(False).astype("int8"),
            description=f"Indicator that `{column}` is at least 60 months, representing older history.",
            formula=f"{column} >= 60",
            semantic_group=_fallback_semantic_group(column),
            feature_type="flag",
        )
        _add_feature(
            additions,
            registry,
            f"{column}_inverse_recency",
            _safe_ratio(pd.Series(1.0, index=features.index), values + 1.0),
            description=f"Inverse recency transform for `{column}`, with larger values for more recent events.",
            formula=f"1 / (1 + {column})",
            semantic_group=_fallback_semantic_group(column),
            feature_type="engineered",
        )

    categorical_specs = {
        "purpose_group": (
            _purpose_group(features["purpose"]) if "purpose" in features.columns else None,
            "Grouped loan purpose category designed for credit-risk interpretation.",
            "group(purpose)",
            "loan_terms",
        ),
        "home_ownership_group": (
            _home_ownership_group(features["home_ownership"]) if "home_ownership" in features.columns else None,
            "Grouped home-ownership category.",
            "group(home_ownership)",
            "loan_terms",
        ),
        "verification_group": (
            _verification_group(features["verification_status"]) if "verification_status" in features.columns else None,
            "Grouped income-verification category.",
            "group(verification_status)",
            "loan_terms",
        ),
        "state_region_group": (
            _state_region(features["addr_state"]) if "addr_state" in features.columns else None,
            "US census-style region derived from borrower state.",
            "region(addr_state)",
            "application_profile",
        ),
        "emp_length_group": (
            _emp_length_group(features["emp_length"]) if "emp_length" in features.columns else None,
            "Grouped borrower employment length.",
            "fixed_bins(emp_length)",
            "income_capacity",
        ),
    }
    for name, (series, desc, formula, group) in categorical_specs.items():
        if series is not None:
            _add_feature(
                additions,
                registry,
                name,
                series,
                description=desc,
                formula=formula,
                semantic_group=group,
                feature_type="categorical",
                availability_timing="application_time",
            )

    if "fico_mean" in features.columns:
        _add_feature(
            additions,
            registry,
            "fico_midpoint_scaled",
            (fico_mean / 850.0).astype("float32"),
            description="FICO midpoint scaled by the approximate maximum consumer FICO score.",
            formula="fico_mean / 850",
            semantic_group="fico_credit_score",
            feature_type="engineered",
            availability_timing="historical_credit_file",
        )
        _add_feature(
            additions,
            registry,
            "fico_to_income",
            _safe_ratio(fico_mean, annual_inc, scale=12000.0),
            description="FICO score level scaled by borrower annual income.",
            formula="12000 * fico_mean / annual_inc",
            semantic_group="fico_credit_score",
            feature_type="ratio",
        )
        _add_feature(
            additions,
            registry,
            "fico_to_loan_amount",
            _safe_ratio(fico_mean, loan_amnt, scale=10000.0),
            description="FICO score level scaled by requested loan amount.",
            formula="10000 * fico_mean / loan_amnt",
            semantic_group="fico_credit_score",
            feature_type="ratio",
        )

    pair_features = [
        ("loan_to_income", "fico_mean", "loan_to_income_x_fico", "Loan affordability pressure interacted with FICO strength.", "income_capacity"),
        ("dti", "revol_util", "dti_x_revol_util", "Debt-to-income pressure interacted with revolving utilization.", "income_capacity"),
        ("dti", "bc_util", "dti_x_bc_util", "Debt-to-income pressure interacted with bankcard utilization.", "income_capacity"),
        ("loan_to_income", "term_months", "term_x_loan_to_income", "Loan affordability pressure interacted with requested term length.", "loan_terms"),
        ("recent_inquiry_density", "fico_mean", "recent_inquiry_density_x_fico", "Recent credit-seeking density interacted with FICO strength.", "recent_inquiries"),
        ("delinquency_pressure", "fico_mean", "delinquency_pressure_x_fico", "Delinquency pressure interacted with FICO strength.", "delinquency_derogatory"),
        ("utilization_pressure", "fico_mean", "utilization_pressure_x_fico", "Utilization pressure interacted with FICO strength.", "revolving_utilization"),
        ("credit_history_years", "fico_mean", "credit_history_years_x_fico", "Credit file age interacted with FICO strength.", "credit_history_length"),
        ("open_acc_to_total_acc", "fico_mean", "open_acc_to_total_acc_x_fico", "Open-account share interacted with FICO strength.", "account_mix_credit_depth"),
    ]
    for left, right, name, desc, group in pair_features:
        if left in features.columns and right in features.columns:
            _add_feature(
                additions,
                registry,
                name,
                (_as_numeric(features, left) * _as_numeric(features, right)).astype("float32"),
                description=desc,
                formula=f"{left} * {right}",
                semantic_group=group,
                feature_type="interaction",
            )

    if "loan_amnt" in features.columns:
        _add_feature(
            additions,
            registry,
            "loan_to_total_credit_limit_v2",
            _safe_ratio(loan_amnt, total_credit_limit),
            description="Requested loan amount relative to total available revolving, bankcard, and installment credit limits.",
            formula="loan_amnt / (total_rev_hi_lim + total_bc_limit + total_il_high_credit_limit)",
            semantic_group="balance_credit_limit_pressure",
            feature_type="ratio",
        )
        _add_feature(
            additions,
            registry,
            "loan_to_total_balance_exposure_v2",
            _safe_ratio(loan_amnt, total_balance),
            description="Requested loan amount relative to non-mortgage and installment balances.",
            formula="loan_amnt / (total_bal_ex_mort + total_bal_il)",
            semantic_group="balance_credit_limit_pressure",
            feature_type="ratio",
        )

    if "purpose_group" in additions and "dti_band" in features.columns:
        _add_feature(
            additions,
            registry,
            "purpose_group_x_dti_band",
            (additions["purpose_group"].astype("string") + "__" + features["dti_band"].astype("string")).astype("category"),
            description="Categorical interaction between grouped loan purpose and debt-to-income band.",
            formula="purpose_group || dti_band",
            semantic_group="loan_terms",
            feature_type="interaction",
            availability_timing="application_time",
        )
    if "verification_group" in additions and "loan_to_income_band" in features.columns:
        _add_feature(
            additions,
            registry,
            "verification_group_x_loan_to_income_band",
            (additions["verification_group"].astype("string") + "__" + features["loan_to_income_band"].astype("string")).astype("category"),
            description="Categorical interaction between income-verification group and loan-to-income band.",
            formula="verification_group || loan_to_income_band",
            semantic_group="income_capacity",
            feature_type="interaction",
            availability_timing="application_time",
        )
    if "fico_band" in features.columns and "term" in features.columns:
        _add_feature(
            additions,
            registry,
            "fico_band_x_term",
            (features["fico_band"].astype("string") + "__" + _zfill_category(features["term"])).astype("category"),
            description="Categorical interaction between FICO band and loan term.",
            formula="fico_band || term",
            semantic_group="fico_credit_score",
            feature_type="interaction",
            availability_timing="derived_from_safe_fields",
        )

    if additions:
        features = pd.concat([features, pd.DataFrame(additions, index=features.index)], axis=1)
    return features.copy()


def _candidate_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column not in HELPER_COLUMNS]


def _apply_final_feature_policy(
    features: pd.DataFrame, registry: dict[str, FeatureInfo]
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    removed = sorted([feature for feature in REMOVED_HIGH_SEVERITY_OR_SPARSE_FEATURES if feature in features.columns])
    if removed:
        features = features.drop(columns=removed)
        for feature in removed:
            registry.pop(feature, None)

    clipped = []
    for feature, (lower, upper) in EXTREME_RATIO_CLIP_CAPS.items():
        if feature not in features.columns:
            continue
        features[feature] = _bounded(features[feature], lower=lower, upper=upper)
        clipped.append(feature)
        info = registry.get(feature)
        if info is not None:
            cap_note = f"Clipped to [{lower:g}, {upper:g}] by fixed pre-matrix v2 ratio policy."
            if "clipped" not in info.description.lower():
                info.description = f"{info.description} {cap_note}"
            if not str(info.source_column_or_formula).startswith("clip("):
                info.source_column_or_formula = f"clip({info.source_column_or_formula}, {lower:g}, {upper:g})"
            info.notes = (info.notes + " | " if info.notes else "") + "fixed_denominator_and_clip_policy"

    return features, {"removed_features": removed, "clipped_ratio_features": sorted(clipped)}


def _write_description_table(metadata_dir: Path, registry: dict[str, FeatureInfo], df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in _candidate_columns(df):
        info = registry.get(feature)
        if info is None:
            desc, formula, feature_type = _describe_feature(feature, str(df[feature].dtype), set())
            group = infer_semantic_group(feature, description=desc)
            if not group or group == "other":
                group = _fallback_semantic_group(feature)
            info = FeatureInfo(
                feature=feature,
                description=desc,
                source_column_or_formula=formula,
                semantic_group=group,
                feature_type=feature_type,
                availability_timing=_availability(feature),
                notes="Fallback metadata generated by v2 preparation script.",
            )
        rows.append(
            {
                "feature": feature,
                "description": info.description,
                "source_column_or_formula": info.source_column_or_formula,
                "semantic_group": info.semantic_group,
                "feature_type": info.feature_type,
                "leakage_review_status": info.leakage_review_status,
                "availability_timing": info.availability_timing,
                "notes": info.notes,
            }
        )
    desc_df = pd.DataFrame(rows)
    desc_df.to_csv(metadata_dir / "columns_description.csv", index=False)
    return desc_df


def _write_inventory(metadata_dir: Path, df: pd.DataFrame, desc_df: pd.DataFrame) -> None:
    candidates = _candidate_columns(df)
    missingness = df[candidates].isna().mean().reset_index()
    missingness.columns = ["feature", "missing_rate"]
    dtype_df = pd.DataFrame({"feature": candidates, "dtype": [str(df[c].dtype) for c in candidates]})
    inventory = desc_df.merge(dtype_df, on="feature", how="left").merge(missingness, on="feature", how="left")
    inventory["non_null_count"] = [int(df[c].notna().sum()) for c in candidates]
    inventory.to_csv(metadata_dir / "feature_inventory.csv", index=False)

    description_present = desc_df["description"].fillna("").astype(str).str.strip().ne("")
    coverage = pd.DataFrame(
        [
            {
                "total_candidate_features": len(candidates),
                "features_with_description": int(description_present.sum()),
                "features_missing_description": int((~description_present).sum()),
                "missing_description_feature_list": ";".join(desc_df.loc[~description_present, "feature"].tolist()),
                "coverage_ratio": round(float(description_present.mean()), 6) if len(desc_df) else 0.0,
            }
        ]
    )
    coverage.to_csv(metadata_dir / "feature_description_coverage.csv", index=False)

    semantic = (
        desc_df.groupby("semantic_group")["feature"]
        .agg(feature_count="size", examples=lambda s: "; ".join(s.head(5)))
        .reset_index()
        .sort_values("feature_count", ascending=False)
    )
    semantic["share_of_features"] = (semantic["feature_count"] / len(candidates)).round(6)
    semantic = semantic[["semantic_group", "feature_count", "share_of_features", "examples"]]
    semantic.to_csv(metadata_dir / "semantic_group_distribution.csv", index=False)

    missing_summary = (
        missingness.assign(non_null_count=[int(df[c].notna().sum()) for c in candidates])
        .sort_values("missing_rate", ascending=False)
    )
    missing_summary.to_csv(metadata_dir / "missingness_summary.csv", index=False)


def _write_leakage_review(metadata_dir: Path, desc_df: pd.DataFrame) -> pd.DataFrame:
    excluded_columns = []
    for column in [
        *LENDINGCLUB_DIRECT_LEAKAGE_COLUMNS,
        *LENDINGCLUB_POST_OUTCOME_LEAKAGE_COLUMNS,
        *LENDINGCLUB_POLICY_OR_POST_APPROVAL_COLUMNS,
        *LENDINGCLUB_UNDERWRITING_POLICY_COLUMNS,
        *LENDINGCLUB_IDENTIFIER_OR_TEXT_COLUMNS,
        *LENDINGCLUB_TEXT_OR_LOW_SIGNAL_COLUMNS,
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "last_credit_pull_d",
        "last_fico_range_low",
        "last_fico_range_high",
        "hardship_*",
        "settlement_*",
    ]:
        if column not in excluded_columns:
            excluded_columns.append(column)

    safe_rows = [
        {
            "feature": row.feature,
            "source_column_or_formula": row.source_column_or_formula,
            "leakage_review_status": row.leakage_review_status,
            "reason": "Generated from application-time or historical credit-file fields retained by the v1 safe preprocessing path.",
            "action": "include",
        }
        for row in desc_df.itertuples(index=False)
    ]
    excluded_rows = [
        {
            "feature": column,
            "source_column_or_formula": column,
            "leakage_review_status": "excluded",
            "reason": "Post-origination outcome, payment, settlement, hardship, underwriting-policy, identifier/text, or target-derived leakage category.",
            "action": "exclude",
        }
        for column in excluded_columns
    ]
    review = pd.DataFrame(safe_rows + excluded_rows)
    review.to_csv(metadata_dir / "leakage_review.csv", index=False)
    return review


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text_df = df.fillna("").astype(str)
    header = "| " + " | ".join(text_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
    rows = [
        "| " + " | ".join(row[col] for col in text_df.columns) + " |"
        for _, row in text_df.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _write_reports(
    reports_dir: Path,
    metadata_dir: Path,
    output_path: Path,
    df: pd.DataFrame,
    desc_df: pd.DataFrame,
    leakage_review: pd.DataFrame,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_columns(df)
    coverage = pd.read_csv(metadata_dir / "feature_description_coverage.csv")
    semantic = pd.read_csv(metadata_dir / "semantic_group_distribution.csv")
    missingness = pd.read_csv(metadata_dir / "missingness_summary.csv")
    manual_review = desc_df[desc_df["leakage_review_status"].eq("needs_manual_review")]
    excluded = leakage_review[leakage_review["action"].eq("exclude")]
    top_semantic = semantic.head(10)
    raw_input_cols = 96
    v1_engineered = 300
    v1_llm_desc = 76
    current_candidates = len(candidates)

    v1_audit = "\n".join(
        [
            "# LendingClub v1 Feature Engineering Gap Audit",
            "",
            "## Summary",
            "",
            f"- Raw safe processed input columns after excluding target/time/status helpers: `{raw_input_cols}`.",
            f"- Reported v1 engineered candidate features: approximately `{v1_engineered}`.",
            f"- v1 features with usable LLM descriptions: approximately `{v1_llm_desc}`.",
            "- `data/lendingclub/metadata/columns_description.csv` contains only the header in the current workspace, so v1 did not have a complete source-level description table.",
            "- The richer v1 LendingClub feature builder exists in `src/credit_risk_fs/feature_engineering/lendingclub/application.py`, but the v1 prepared `application_train.csv` contains the safe source table rather than all engineered features.",
            "",
            "## Answers",
            "",
            "1. Safe raw columns available after preprocessing: 96 candidate columns.",
            "2. Engineered candidate columns reported in v1 reports: about 300.",
            "3. Features with descriptions in v1 LLM/reporting artifacts: about 76.",
            "4. Underrepresented concepts: categorical grouping metadata, systematic missingness indicators, FICO-affordability interactions, account-depth ratios, recency flags, joint-applicant coverage, balance-to-limit pressure variants, and interpretable grouped categorical interactions.",
            "5. Selected or LLM-ranked features lacking descriptions cannot be fully rechecked from `results/lendingclub` because that result folder is absent in the current workspace; the available generated report shows 76 features with LLM rank/description and 429 broader-union features without descriptions.",
            "6. Vague semantic groups were most likely caused by the empty source description CSV and fallback name-only inference. v2 writes a semantic group for every candidate feature.",
            "7. Yes. LendingClub v1 is much simpler than Home Credit by metadata coverage and by number of described features.",
            "8. Yes, this simplicity could partly explain why pure LLM wins on LendingClub: the LLM was screening a smaller, more metadata-filtered candidate set, while the broader feature universe was under-described.",
        ]
    )
    (reports_dir / "lendingclub_v1_feature_engineering_gap_audit.md").write_text(v1_audit, encoding="utf-8")

    inventory_report = "\n".join(
        [
            "# LendingClub v2 Feature Inventory",
            "",
            f"- Processed file: `{output_path.as_posix()}`",
            f"- Rows: `{len(df):,}`",
            f"- Total columns including helpers: `{df.shape[1]:,}`",
            f"- Candidate feature columns: `{current_candidates:,}`",
            f"- Features with descriptions: `{int(coverage.loc[0, 'features_with_description']):,}`",
            f"- Description coverage: `{float(coverage.loc[0, 'coverage_ratio']):.2%}`",
            f"- Semantic groups: `{semantic['semantic_group'].nunique():,}`",
            "",
            "## Dominant Semantic Groups",
            "",
            _markdown_table(top_semantic),
            "",
            "## Highest Missingness Features",
            "",
            _markdown_table(missingness.head(15)),
        ]
    )
    (reports_dir / "lendingclub_v2_feature_inventory.md").write_text(inventory_report, encoding="utf-8")

    preapproval = "\n".join(
        [
            "# LendingClub v2 Preapproval Report",
            "",
            "This report covers metadata and feature-preparation readiness only. No CatBoost/LR/Boruta/mRMR experiment matrix was run.",
            "",
            "## Answers",
            "",
            f"1. Final candidate features in `data/lendingclub_v2/processed/application_train.csv`: `{current_candidates:,}`.",
            f"2. Features with descriptions: `{int(coverage.loc[0, 'features_with_description']):,}`.",
            f"3. Description coverage is 100%: `{'yes' if float(coverage.loc[0, 'coverage_ratio']) == 1.0 else 'no'}`.",
            f"4. Semantic groups: `{semantic['semantic_group'].nunique():,}`.",
            "5. Dominant semantic groups are listed in `data/lendingclub_v2/metadata/semantic_group_distribution.csv`; the largest groups are shown in the inventory report.",
            f"6. Features marked `needs_manual_review`: `{len(manual_review):,}`.",
            f"7. Leakage-risk columns excluded: `{len(excluded):,}` explicit columns/patterns are listed in `data/lendingclub_v2/metadata/leakage_review.csv`.",
            f"8. The feature space is richer than v1: v2 has `{current_candidates:,}` candidate features versus the v1 report's approximately `{v1_engineered}` engineered candidates and 76 described LLM features.",
            "9. The v2 design is more comparable to Home Credit in count and metadata coverage, while remaining LendingClub-specific and leakage-screened.",
            "10. It is approval-ready for human inspection if the reviewer accepts the generated feature families and leakage review.",
            "11. A full LendingClub v2 matrix rerun should not be run until human approval.",
            "12. After approval only, run:",
            "",
            "```bash",
            "python scripts/run_matrix.py --dataset lendingclub_v2",
            "python scripts/aggregate_results.py --dataset lendingclub_v2",
            "python scripts/make_plots.py --dataset lendingclub_v2",
            "```",
            "",
            "## Rerun Decision",
            "",
            "- Matrix run performed now: no.",
            "- Full rerun required before inspection: no.",
            "- Targeted artifact generation completed: v2 processed table and metadata inspection artifacts.",
        ]
    )
    (reports_dir / "lendingclub_v2_preapproval_report.md").write_text(preapproval, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = PROJECT_ROOT / args.input_file
    output_dir = PROJECT_ROOT / args.output_dir
    metadata_dir = PROJECT_ROOT / args.metadata_dir
    reports_dir = PROJECT_ROOT / args.reports_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    if "issue_d" not in df.columns and "recent_decision" in df.columns:
        raise ValueError("v2 preparation requires issue_d for diagnostics and split traceability.")
    source_columns = list(df.columns)

    features = build_application_features(df)
    registry = _register_existing_features(features, source_columns)
    features = add_v2_features(features, registry)
    features, feature_policy_actions = _apply_final_feature_policy(features, registry)

    ordered_cols = [col for col in OUTPUT_HELPER_COLUMNS if col in features.columns]
    candidate_cols = sorted([col for col in features.columns if col not in HELPER_COLUMNS])
    features = features[ordered_cols + candidate_cols].copy()

    output_path = output_dir / "application_train.csv"
    features.to_csv(output_path, index=False)

    desc_df = _write_description_table(metadata_dir, registry, features)
    _write_inventory(metadata_dir, features, desc_df)
    leakage_review = _write_leakage_review(metadata_dir, desc_df)
    _write_reports(reports_dir, metadata_dir, output_path.relative_to(PROJECT_ROOT), features, desc_df, leakage_review)

    summary = {
        "rows": int(len(features)),
        "total_columns": int(features.shape[1]),
        "candidate_features": int(len(_candidate_columns(features))),
        "description_coverage": float(
            desc_df["description"].fillna("").astype(str).str.strip().ne("").mean()
        ),
        "semantic_groups": int(desc_df["semantic_group"].nunique()),
        "manual_review_features": int(desc_df["leakage_review_status"].eq("needs_manual_review").sum()),
        "removed_features": feature_policy_actions["removed_features"],
        "clipped_ratio_features": feature_policy_actions["clipped_ratio_features"],
        "high_missingness_policy": (
            "Removed constant sanity-check features and redundant ultra-sparse joint/secondary-applicant "
            "derivatives; retained sparse raw joint/secondary fields and summary indicators where the "
            "missingness itself reflects non-joint applications."
        ),
        "extreme_ratio_policy": (
            "Ratios use zero-denominator-to-missing handling and fixed, outcome-independent clipping caps "
            "for the pre-matrix review feature set."
        ),
        "matrix_run": False,
    }
    (metadata_dir / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Prepared LendingClub v2 file: {output_path}")
    print(f"Rows: {summary['rows']:,} | Columns: {summary['total_columns']:,}")
    print(f"Candidate features: {summary['candidate_features']:,}")
    print(f"Description coverage: {summary['description_coverage']:.2%}")
    print(f"Semantic groups: {summary['semantic_groups']:,}")
    print(f"Manual-review features: {summary['manual_review_features']:,}")
    print("Matrix run: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
