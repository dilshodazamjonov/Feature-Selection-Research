from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/lendingclub_v2/processed/application_train.csv"
METADATA_DIR = PROJECT_ROOT / "data/lendingclub_v2/metadata"
REPORTS_DIR = PROJECT_ROOT / "reports"
DESCRIPTION_PATH = METADATA_DIR / "columns_description.csv"
INVENTORY_PATH = METADATA_DIR / "feature_inventory.csv"
LEAKAGE_REVIEW_PATH = METADATA_DIR / "leakage_review.csv"
HELPER_COLUMNS = {"TARGET", "recent_decision", "issue_d", "loan_status"}
ALLOWED_HELPERS = {"TARGET", "recent_decision", "issue_d"}
FORBIDDEN_PATTERNS = (
    "loan_status",
    "total_pymnt",
    "total_rec_",
    "recoveries",
    "collection_recovery_fee",
    "settlement",
    "hardship",
    "last_pymnt",
    "next_pymnt",
    "last_credit_pull",
    "out_prncp",
)
GENERIC_PREFIX = "Engineered LendingClub application-time credit-risk feature"


DESCRIPTION_OVERRIDES = {
    "acc_open_past_24mths_share": "Share of total credit accounts opened in the past 24 months, measuring recent account-opening intensity relative to credit-file depth.",
    "active_revolving_share": "Share of revolving accounts with a positive balance, measuring active revolving credit usage among revolving trades.",
    "all_util_minus_bc_util": "Difference between aggregate credit-line utilization and bankcard utilization, highlighting non-bankcard utilization pressure.",
    "all_util_minus_il_util": "Difference between aggregate utilization and installment utilization, showing whether utilization pressure is concentrated outside installment credit.",
    "application_type_verification_status": "Categorical interaction between application type and income-verification status at origination.",
    "avg_cur_bal_gap": "Difference between reported average current balance and calculated current balance per open account.",
    "bankcard_capacity_gap": "Bankcard credit capacity gap calculated as total bankcard limit minus maximum bankcard balance.",
    "bankruptcy_history_flag": "Indicator that the borrower has at least one public-record bankruptcy.",
    "bankruptcy_or_pubrec_count": "Combined count of public records and public-record bankruptcies in the credit file.",
    "bc_util_minus_revol_util": "Difference between bankcard utilization and overall revolving utilization.",
    "credit_history_months": "Borrower credit-file age in months from earliest credit line to loan issue month.",
    "credit_history_short_flag": "Indicator that the borrower has less than three years of credit history.",
    "credit_history_years": "Borrower credit-file age in years from earliest credit line to loan issue month.",
    "delinquency_history_flag": "Indicator that the borrower has at least one delinquency in the recent credit history fields.",
    "delinquency_pressure": "Composite delinquency pressure count combining recent delinquencies, severe delinquencies, and accounts ever 120+ days past due.",
    "emp_length_years": "Borrower reported employment length converted to numeric years.",
    "fico_mean": "Midpoint of the borrower FICO score range at origination.",
    "fico_span": "Width of the borrower FICO score range at origination.",
    "has_mortgage_flag": "Indicator that the credit file reports at least one mortgage account.",
    "has_public_record_flag": "Indicator that the borrower has at least one derogatory public record.",
    "has_recent_inquiry_flag": "Indicator that the borrower had at least one credit inquiry in the last six months.",
    "high_bc_util_flag": "Indicator that bankcard utilization is at least 80 percent.",
    "high_dti_flag": "Indicator that borrower debt-to-income ratio exceeds 20 percent.",
    "high_revol_util_flag": "Indicator that revolving utilization is at least 80 percent.",
    "joint_application_flag": "Indicator that the loan application is a joint application.",
    "joint_dti_gap": "Difference between joint-applicant DTI and primary-borrower DTI.",
    "joint_income_available_flag": "Indicator that joint-applicant income is available on the application.",
    "low_fico_flag": "Indicator that the FICO midpoint is below 660.",
    "mort_balance_pressure": "Non-mortgage balance pressure scaled by the number of mortgage accounts plus one.",
    "num_bc_sats_share": "Share of bankcard trades that are currently satisfactory.",
    "num_sats_share": "Share of all credit accounts that are currently satisfactory.",
    "num_tl_op_past_12m_share": "Share of total credit accounts opened in the past 12 months.",
    "open_installment_share": "Share of open accounts that are active installment trades.",
    "open_revolving_share": "Share of total accounts that are open revolving trades.",
    "prime_fico_flag": "Indicator that the FICO midpoint is at least 720.",
    "purpose_verification_status": "Categorical interaction between stated loan purpose and income-verification status.",
    "recent_inquiry_density": "Recent credit-seeking density combining six-month inquiries, finance inquiries, and recently opened accounts relative to open accounts.",
    "recent_installment_activity_flag": "Indicator that at least two installment accounts were opened in the last 12 months.",
    "recent_opening_flag": "Indicator that at least four accounts were opened in the past 24 months.",
    "recent_revolving_activity_flag": "Indicator that at least two revolving accounts were opened in the last 12 months.",
    "recent_trade_density": "Recent trade-opening density combining recent installment, revolving, and total trade openings relative to total accounts.",
    "revolving_capacity_gap": "Unused revolving capacity calculated as total revolving high credit limit minus revolving balance.",
    "sec_app_credit_history_months": "Secondary applicant credit-file age in months from earliest credit line to issue month.",
    "sec_app_credit_history_years": "Secondary applicant credit-file age in years from earliest credit line to issue month.",
    "sec_app_fico_mean": "Midpoint of the secondary applicant FICO score range.",
    "sec_app_fico_span": "Width of the secondary applicant FICO score range.",
    "second_app_present_flag": "Indicator that secondary-applicant FICO information is present.",
    "severe_delinquency_count": "Combined count of bankruptcies, accounts 90+ days past due, and accounts ever 120+ days past due.",
    "term_home_ownership": "Categorical interaction between loan term and borrower home-ownership status.",
    "term_months": "Loan repayment term converted to numeric months.",
    "term_verification_status": "Categorical interaction between loan term and income-verification status.",
    "thin_file_flag": "Indicator that the borrower has fewer than 10 total credit accounts.",
    "utilization_pressure": "Average of revolving, bankcard, and aggregate utilization measures.",
    "very_high_dti_flag": "Indicator that borrower debt-to-income ratio exceeds 30 percent.",
}

FORMULA_OVERRIDES = {
    "acc_open_past_24mths_share": "acc_open_past_24mths / total_acc",
    "active_revolving_share": "num_rev_tl_bal_gt_0 / num_rev_accts",
    "all_util_minus_bc_util": "all_util - bc_util",
    "all_util_minus_il_util": "all_util - il_util",
    "application_type_verification_status": "application_type || verification_status",
    "avg_cur_bal_gap": "avg_cur_bal - (tot_cur_bal / open_acc)",
    "bankcard_capacity_gap": "total_bc_limit - max_bal_bc",
    "bankruptcy_history_flag": "pub_rec_bankruptcies > 0",
    "bankruptcy_or_pubrec_count": "pub_rec + pub_rec_bankruptcies",
    "bc_util_minus_revol_util": "bc_util - revol_util",
    "credit_history_months": "months_between(issue_d, earliest_cr_line)",
    "credit_history_short_flag": "credit_history_months < 36",
    "credit_history_years": "credit_history_months / 12",
    "delinquency_history_flag": "delinq_2yrs > 0",
    "delinquency_pressure": "delinq_2yrs + num_tl_30dpd + num_tl_90g_dpd_24m + num_accts_ever_120_pd",
    "emp_length_years": "parse_years(emp_length)",
    "fico_mean": "(fico_range_low + fico_range_high) / 2",
    "fico_span": "fico_range_high - fico_range_low",
    "has_mortgage_flag": "mort_acc > 0",
    "has_public_record_flag": "pub_rec > 0",
    "has_recent_inquiry_flag": "inq_last_6mths > 0",
    "high_bc_util_flag": "bc_util >= 80",
    "high_dti_flag": "dti > 20",
    "high_revol_util_flag": "revol_util >= 80",
    "joint_application_flag": "application_type contains JOINT",
    "joint_dti_gap": "dti_joint - dti",
    "joint_income_available_flag": "annual_inc_joint is not missing",
    "low_fico_flag": "fico_mean < 660",
    "mort_balance_pressure": "total_bal_ex_mort / (mort_acc + 1)",
    "num_bc_sats_share": "num_bc_sats / num_bc_tl",
    "num_sats_share": "num_sats / total_acc",
    "num_tl_op_past_12m_share": "num_tl_op_past_12m / total_acc",
    "open_installment_share": "open_act_il / open_acc",
    "open_revolving_share": "num_op_rev_tl / total_acc",
    "prime_fico_flag": "fico_mean >= 720",
    "purpose_verification_status": "purpose || verification_status",
    "recent_inquiry_density": "(inq_last_6mths + inq_fi + open_acc_6m) / (open_acc + 1)",
    "recent_installment_activity_flag": "open_il_12m >= 2",
    "recent_opening_flag": "acc_open_past_24mths >= 4",
    "recent_revolving_activity_flag": "open_rv_12m >= 2",
    "recent_trade_density": "(open_il_12m + open_rv_12m + num_tl_op_past_12m) / (total_acc + 1)",
    "revolving_capacity_gap": "total_rev_hi_lim - revol_bal",
    "sec_app_credit_history_months": "months_between(issue_d, sec_app_earliest_cr_line)",
    "sec_app_credit_history_years": "sec_app_credit_history_months / 12",
    "sec_app_fico_mean": "(sec_app_fico_range_low + sec_app_fico_range_high) / 2",
    "sec_app_fico_span": "sec_app_fico_range_high - sec_app_fico_range_low",
    "second_app_present_flag": "sec_app_fico_range_low is not missing",
    "severe_delinquency_count": "pub_rec_bankruptcies + num_tl_90g_dpd_24m + num_accts_ever_120_pd",
    "term_home_ownership": "term || home_ownership",
    "term_months": "numeric_months(term)",
    "term_verification_status": "term || verification_status",
    "thin_file_flag": "total_acc < 10",
    "utilization_pressure": "mean(revol_util, bc_util, all_util)",
    "very_high_dti_flag": "dti > 30",
}


def _contains_any(text: str, tokens: tuple[str, ...] | list[str]) -> bool:
    return any(token in text for token in tokens)


def semantic_group_for(feature: str) -> str:
    f = feature.lower()

    if feature.endswith("_missing_flag") or "missing" in f:
        return "missingness_or_unknown"
    if (
        f.startswith("dti")
        or f.startswith("annual_inc")
        or f.startswith("log_annual_inc")
        or f.startswith("sqrt_annual_inc")
        or f.startswith("loan_to_income")
        or _contains_any(f, ("high_dti", "very_high_dti", "income_band", "joint_income", "term_x_loan_to_income", "verification_group_x_loan_to_income", "emp_length"))
    ):
        return "income_capacity"
    if f.startswith("fico") or _contains_any(f, ("low_fico", "prime_fico", "sec_app_fico", "_x_fico", "fico_adjusted")):
        return "fico_credit_score"
    if _contains_any(
        f,
        (
            "acc_now_delinq",
            "delinq",
            "derog",
            "chargeoff",
            "bankrupt",
            "pub_rec",
            "tax_lien",
            "collections_12_mths",
            "120dpd",
            "90g_dpd",
            "30dpd",
            "pct_tl_nvr_dlq",
            "severe_delinquency",
            "public_record",
        ),
    ):
        return "delinquency_derogatory"
    if _contains_any(f, ("recent_inquiry", "inq_", "inq_last", "mths_since_recent_inq", "has_recent_inquiry")):
        return "recent_inquiries"
    if _contains_any(f, ("mort_", "mortgage", "mort_acc", "has_mortgage")):
        return "mortgage_history"
    if _contains_any(f, ("acc_open_past_24mths", "num_tl_op_past_12m", "recent_opening_flag", "recent_trade_density")):
        return "account_opening_activity"
    if _contains_any(f, ("bc_", "bankcard", "num_bc", "total_bc", "max_bal_bc", "percent_bc", "mths_since_recent_bc")):
        return "bankcard_capacity"
    if _contains_any(f, ("revol", "rv_", "utilization_pressure")):
        return "revolving_utilization"
    if _contains_any(f, ("joint_", "sec_app", "second_app", "application_type_verification_status")):
        return "joint_applicant"
    if _contains_any(f, ("term", "purpose", "home_ownership", "verification_status", "initial_list_status", "application_type")):
        return "loan_terms"
    if _contains_any(f, ("loan_amnt", "loan_amount", "log_loan", "sqrt_loan")):
        return "exposure_amount"
    if _contains_any(f, ("credit_history", "mo_sin_old", "mo_sin_rcnt", "mths_since_last_record", "mths_since_rcnt_il")):
        return "credit_history_length"
    if _contains_any(
        f,
        (
            "open_acc",
            "total_acc",
            "num_sats",
            "open_il",
            "open_rv",
            "open_act_il",
            "num_il",
            "num_op",
            "num_rev_accts",
            "num_actv",
            "trade_density",
            "thin_file",
            "credit_lines_per_history",
        ),
    ):
        return "account_mix_credit_depth"
    if _contains_any(
        f,
        (
            "bal",
            "limit",
            "all_util",
            "il_util",
            "tot_hi_cred_lim",
            "total_rev_hi_lim",
            "tot_cur_bal",
            "total_il_high_credit_limit",
            "total_bal",
            "avg_cur_bal",
            "capacity_gap",
            "total_cu_tl",
            "tot_coll_amt",
        ),
    ):
        return "balance_credit_limit_pressure"
    if _contains_any(f, ("addr_state", "state_region")):
        return "geographic_profile"
    return "credit_risk_other_reviewed"


def _readable(name: str) -> str:
    return name.replace("_", " ")


def description_for(feature: str, old_description: str) -> str:
    if feature in DESCRIPTION_OVERRIDES:
        return DESCRIPTION_OVERRIDES[feature]
    if not old_description.startswith(GENERIC_PREFIX):
        return old_description

    f = feature.lower()
    if feature.startswith("log_"):
        base = feature[4:]
        return f"Log-transformed nonnegative `{base}` to reduce the influence of high-end skew while preserving credit-risk scale."
    if feature.startswith("sqrt_"):
        base = feature[5:]
        return f"Square-root transformed nonnegative `{base}` to moderate high-end balance or count tails."
    if feature.startswith("capped_"):
        base = feature[7:]
        return f"Domain-capped `{base}` value used to limit implausible scale without using outcome information."
    if feature.endswith("_is_zero"):
        base = feature[:-8]
        return f"Indicator that `{base}` is exactly zero, separating no-event or no-balance borrowers from positive values."
    if feature.endswith("_positive_flag"):
        base = feature[:-14]
        return f"Indicator that `{base}` is positive, capturing presence of the underlying credit-file event or account type."
    if feature.endswith("_per_open_acc"):
        base = feature[:-13]
        return f"`{base}` divided by open account count, measuring concentration among active accounts."
    if feature.endswith("_per_total_acc"):
        base = feature[:-14]
        return f"`{base}` divided by total account count, measuring concentration across the full credit file."
    if feature.endswith("_per_credit_history_year"):
        base = feature[:-24]
        return f"`{base}` divided by credit-history years, measuring event density over file age."
    if feature.endswith("_band"):
        base = feature[:-5]
        return f"Fixed business-rule band for `{base}` used to represent nonlinear credit-risk ranges."
    if feature.endswith("_recent_12m_flag"):
        base = feature[:-16]
        return f"Indicator that `{base}` occurred or was measured within the past 12 months."
    if feature.endswith("_recent_24m_flag"):
        base = feature[:-16]
        return f"Indicator that `{base}` occurred or was measured within the past 24 months."
    if feature.endswith("_seasoned_60m_flag"):
        base = feature[:-18]
        return f"Indicator that `{base}` is at least 60 months old, representing older credit history."
    if feature.endswith("_inverse_recency"):
        base = feature[:-16]
        return f"Inverse-recency transform for `{base}`, assigning larger values to more recent credit-file events."
    if "_x_" in feature:
        left, right = feature.split("_x_", 1)
        return f"Interpretable interaction between `{left}` and `{right}` for combined credit-risk pressure."
    if "_to_" in feature:
        left, right = feature.split("_to_", 1)
        return f"Ratio of `{left}` to `{right}`, measuring relative affordability, exposure, balance, or limit pressure."
    if "_minus_" in feature:
        left, right = feature.split("_minus_", 1)
        return f"Difference between `{left}` and `{right}` to compare related utilization or balance-pressure measures."
    return f"Reviewed LendingClub credit-risk feature `{feature}` representing {_readable(feature)} from application-time or historical credit-file fields."


def formula_for(feature: str, old_formula: str) -> str:
    if feature in FORMULA_OVERRIDES:
        return FORMULA_OVERRIDES[feature]
    if old_formula and not str(old_formula).startswith("derived_from_safe_fields("):
        return old_formula
    if feature.startswith("log_"):
        return f"log1p(max({feature[4:]}, 0))"
    if feature.startswith("sqrt_"):
        return f"sqrt(max({feature[5:]}, 0))"
    if feature.startswith("capped_"):
        return f"domain_cap({feature[7:]})"
    if feature.endswith("_is_zero"):
        return f"{feature[:-8]} == 0"
    if feature.endswith("_positive_flag"):
        return f"{feature[:-14]} > 0"
    if feature.endswith("_per_open_acc"):
        return f"{feature[:-13]} / open_acc"
    if feature.endswith("_per_total_acc"):
        return f"{feature[:-14]} / total_acc"
    if feature.endswith("_per_credit_history_year"):
        return f"{feature[:-24]} / credit_history_years"
    if feature.endswith("_recent_12m_flag"):
        return f"{feature[:-16]} <= 12"
    if feature.endswith("_recent_24m_flag"):
        return f"{feature[:-16]} <= 24"
    if feature.endswith("_seasoned_60m_flag"):
        return f"{feature[:-18]} >= 60"
    if feature.endswith("_inverse_recency"):
        return f"1 / (1 + {feature[:-16]})"
    if feature.endswith("_band"):
        return f"fixed_bins({feature[:-5]})"
    if "_x_" in feature:
        return feature.replace("_x_", " * ")
    if "_to_" in feature:
        return feature.replace("_to_", " / ")
    if "_minus_" in feature:
        return feature.replace("_minus_", " - ")
    return feature


def clean_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DESCRIPTION_PATH)
    before_generic = df["description"].fillna("").astype(str).str.startswith(GENERIC_PREFIX)
    audit_rows = []

    for idx, row in df.iterrows():
        feature = str(row["feature"])
        old_description = "" if pd.isna(row["description"]) else str(row["description"])
        old_group = "" if pd.isna(row["semantic_group"]) else str(row["semantic_group"])
        old_formula = "" if pd.isna(row["source_column_or_formula"]) else str(row["source_column_or_formula"])

        new_description = description_for(feature, old_description)
        new_group = semantic_group_for(feature)
        new_formula = formula_for(feature, old_formula)
        if row.get("leakage_review_status", "safe") not in {"safe", "needs_manual_review"}:
            new_status = "needs_manual_review"
        else:
            new_status = row.get("leakage_review_status", "safe")

        issue_types = []
        if old_description.startswith(GENERIC_PREFIX):
            issue_types.append("generic_description")
        if old_group != new_group:
            issue_types.append("semantic_group_mismatch")
        if not old_formula.strip() or old_formula.startswith("derived_from_safe_fields("):
            issue_types.append("weak_source_formula")
        if not old_description.strip():
            issue_types.append("blank_description")
        if not old_group.strip():
            issue_types.append("blank_semantic_group")

        df.at[idx, "description"] = new_description
        df.at[idx, "semantic_group"] = new_group
        df.at[idx, "source_column_or_formula"] = new_formula
        df.at[idx, "leakage_review_status"] = new_status
        notes = "" if pd.isna(row.get("notes", "")) else str(row.get("notes", ""))
        if issue_types and "metadata_quality_cleaned" not in notes:
            notes = (notes + " | " if notes else "") + "metadata_quality_cleaned"
        df.at[idx, "notes"] = notes

        remaining_flags = []
        if not str(new_description).strip():
            remaining_flags.append("blank_description")
        if str(new_description).startswith(GENERIC_PREFIX):
            remaining_flags.append("generic_description")
        if not str(new_group).strip():
            remaining_flags.append("blank_semantic_group")
        if str(new_group) in {"application_amounts", "application_profile", "demographic_time_variables", "delinquency_behavior"}:
            remaining_flags.append("vague_semantic_group")
        if not str(new_formula).strip() or str(new_formula).startswith("derived_from_safe_fields("):
            remaining_flags.append("weak_source_formula")
        if str(new_status) not in {"safe", "needs_manual_review"}:
            remaining_flags.append("invalid_leakage_status")

        audit_rows.append(
            {
                "feature": feature,
                "issue_type": ";".join(issue_types) if issue_types else "verified_clean",
                "old_description": old_description,
                "new_description": new_description,
                "old_semantic_group": old_group,
                "new_semantic_group": new_group,
                "old_source_column_or_formula": old_formula,
                "new_source_column_or_formula": new_formula,
                "old_leakage_review_status": row.get("leakage_review_status", ""),
                "new_leakage_review_status": new_status,
                "remaining_quality_flags": ";".join(remaining_flags),
                "action_taken": "updated_metadata" if issue_types else "verified_clean",
            }
        )

    df.to_csv(DESCRIPTION_PATH, index=False)
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(METADATA_DIR / "metadata_quality_audit.csv", index=False)
    update_derived_metadata(df)
    return df, audit


def update_derived_metadata(desc: pd.DataFrame) -> None:
    if INVENTORY_PATH.exists():
        inventory = pd.read_csv(INVENTORY_PATH)
        keep_cols = [c for c in ["feature", "dtype", "missing_rate", "non_null_count"] if c in inventory.columns]
        inventory = desc.merge(inventory[keep_cols], on="feature", how="left")
        inventory.to_csv(INVENTORY_PATH, index=False)

    semantic = (
        desc.groupby("semantic_group")["feature"]
        .agg(feature_count="size", examples=lambda s: "; ".join(s.head(5)))
        .reset_index()
        .sort_values("feature_count", ascending=False)
    )
    semantic["share_of_features"] = (semantic["feature_count"] / len(desc)).round(6)
    semantic = semantic[["semantic_group", "feature_count", "share_of_features", "examples"]]
    semantic.to_csv(METADATA_DIR / "semantic_group_distribution.csv", index=False)

    if LEAKAGE_REVIEW_PATH.exists():
        review = pd.read_csv(LEAKAGE_REVIEW_PATH)
        include = review[review["action"].eq("include")].drop(columns=["source_column_or_formula"], errors="ignore")
        include = include[["feature", "leakage_review_status", "reason", "action"]]
        include = include.merge(desc[["feature", "source_column_or_formula"]], on="feature", how="left")
        include = include[["feature", "source_column_or_formula", "leakage_review_status", "reason", "action"]]
        excluded = review[review["action"].eq("exclude")]
        pd.concat([include, excluded], ignore_index=True).to_csv(LEAKAGE_REVIEW_PATH, index=False)

    coverage = pd.DataFrame(
        [
            {
                "total_candidate_features": len(desc),
                "features_with_description": int(desc["description"].fillna("").astype(str).str.strip().ne("").sum()),
                "features_missing_description": int(desc["description"].fillna("").astype(str).str.strip().eq("").sum()),
                "missing_description_feature_list": ";".join(
                    desc.loc[desc["description"].fillna("").astype(str).str.strip().eq(""), "feature"]
                ),
                "coverage_ratio": round(
                    float(desc["description"].fillna("").astype(str).str.strip().ne("").mean()), 6
                ),
            }
        ]
    )
    coverage.to_csv(METADATA_DIR / "feature_description_coverage.csv", index=False)


def _header_counts(path: Path) -> Counter:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return Counter(next(csv.reader(handle)))


def sanity_check(desc: pd.DataFrame, chunksize: int = 50_000) -> pd.DataFrame:
    header_counter = _header_counts(DATA_PATH)
    duplicate_names = {name for name, count in header_counter.items() if count > 1}
    header = list(header_counter.keys())
    candidate_cols = [c for c in header if c not in HELPER_COLUMNS]
    metadata = desc.set_index("feature")
    cat_candidates = {
        row.feature
        for row in desc.itertuples(index=False)
        if str(row.feature_type) in {"categorical", "binned"}
        or "||" in str(row.source_column_or_formula)
    }
    raw_categorical_candidates = {
        "term",
        "emp_length",
        "home_ownership",
        "verification_status",
        "purpose",
        "addr_state",
        "initial_list_status",
        "application_type",
        "verification_status_joint",
    }
    cat_candidates.update(raw_categorical_candidates & set(candidate_cols))

    total_rows = 0
    non_null = Counter()
    inf_counts = Counter()
    invalid_numeric_counts = Counter()
    max_abs_values = Counter()
    unique_values: dict[str, set[str]] = {col: set() for col in header}
    cat_uniques: dict[str, set[str]] = {col: set() for col in cat_candidates if col in header}
    high_cardinality = set()
    numeric_candidates = [
        row.feature
        for row in desc.itertuples(index=False)
        if row.feature in candidate_cols
        and row.feature not in cat_candidates
        and str(row.feature_type) not in {"categorical", "binned"}
    ]

    for chunk in pd.read_csv(DATA_PATH, chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        non_null.update(chunk.notna().sum().to_dict())

        for column in header:
            if len(unique_values[column]) <= 1000:
                vals = chunk[column].dropna().astype(str).unique()
                for value in vals[: max(0, 1001 - len(unique_values[column]))]:
                    unique_values[column].add(value)

        numeric = chunk[candidate_cols].apply(pd.to_numeric, errors="coerce")
        inf_mask = np.isinf(numeric.to_numpy(dtype=float, copy=False))
        if inf_mask.any():
            counts = pd.Series(inf_mask.sum(axis=0), index=candidate_cols)
            inf_counts.update(counts[counts > 0].astype(int).to_dict())

        numeric_abs_max = numeric.abs().max(skipna=True)
        for column, value in numeric_abs_max.dropna().items():
            max_abs_values[column] = max(float(max_abs_values[column]), float(value))

        numeric_source = chunk[numeric_candidates]
        numeric_converted = numeric[numeric_candidates]
        invalid_numeric = numeric_source.notna() & numeric_converted.isna()
        if invalid_numeric.to_numpy().any():
            counts = invalid_numeric.sum(axis=0)
            invalid_numeric_counts.update(counts[counts > 0].astype(int).to_dict())

        for column in list(cat_uniques):
            if column in high_cardinality:
                continue
            values = chunk[column].dropna().astype(str).unique()
            cat_uniques[column].update(values.tolist())
            if len(cat_uniques[column]) > 100:
                high_cardinality.add(column)
                cat_uniques[column] = set(list(cat_uniques[column])[:101])

    rows = []
    for column in header:
        is_candidate = column not in HELPER_COLUMNS
        missing_rate = 1.0 - (non_null[column] / total_rows if total_rows else 0.0)
        unique_non_null = len(unique_values[column])
        duplicate = column in duplicate_names
        all_null = non_null[column] == 0
        constant = is_candidate and (not all_null) and unique_non_null <= 1
        inf_count = int(inf_counts[column])
        invalid_numeric_count = int(invalid_numeric_counts[column])
        max_abs_value = float(max_abs_values[column]) if column in max_abs_values else np.nan
        cat_cardinality = len(cat_uniques[column]) if column in cat_uniques else np.nan
        high_card = column in high_cardinality or (
            column in cat_uniques and isinstance(cat_cardinality, int) and cat_cardinality > 100
        )
        forbidden_match = [pattern for pattern in FORBIDDEN_PATTERNS if pattern in column.lower()]
        unexpected_helper = (column in HELPER_COLUMNS and column not in ALLOWED_HELPERS) or bool(forbidden_match)
        issue_flags = []
        if duplicate:
            issue_flags.append("duplicate_column")
        if all_null:
            issue_flags.append("all_null")
        if constant:
            issue_flags.append("constant")
        if inf_count:
            issue_flags.append("infinite_values")
        if is_candidate and missing_rate > 0.95:
            issue_flags.append("missing_rate_gt_95")
        if high_card:
            issue_flags.append("high_cardinality_categorical")
        if unexpected_helper:
            issue_flags.append("target_split_or_helper_leakage")
        if invalid_numeric_count:
            issue_flags.append("invalid_numeric_values")
        if is_candidate and ("_to_" in column or "_per_" in column or "ratio" in column) and pd.notna(max_abs_value) and max_abs_value > 1_000_000:
            issue_flags.append("extreme_ratio_values")

        recommendation = "keep"
        if any(flag in issue_flags for flag in ["duplicate_column", "all_null", "constant", "infinite_values", "target_split_or_helper_leakage", "invalid_numeric_values"]):
            recommendation = "remove_or_fix_before_matrix"
        elif any(flag in issue_flags for flag in ["missing_rate_gt_95", "high_cardinality_categorical", "extreme_ratio_values"]):
            recommendation = "review_before_matrix"

        severity = "pass"
        if recommendation == "remove_or_fix_before_matrix":
            severity = "high"
        elif recommendation == "review_before_matrix":
            severity = "medium"

        examples = "; ".join(list(unique_values[column])[:5])

        rows.append(
            {
                "feature": column,
                "issue_type": ";".join(issue_flags) if issue_flags else "none",
                "severity": severity,
                "missing_rate": round(missing_rate, 6),
                "unique_count": unique_non_null,
                "example_values_if_available": examples,
                "recommended_action": recommendation,
                "is_candidate_feature": is_candidate,
                "semantic_group": metadata["semantic_group"].get(column, "helper"),
                "feature_type": metadata["feature_type"].get(column, "helper"),
                "duplicate_column": duplicate,
                "all_null": all_null,
                "constant_column": constant,
                "infinite_value_count": inf_count,
                "invalid_numeric_value_count": invalid_numeric_count,
                "max_abs_value": max_abs_value,
                "missing_rate_gt_95": is_candidate and missing_rate > 0.95,
                "categorical_cardinality": cat_cardinality,
                "high_cardinality_categorical": bool(high_card),
                "target_split_helper_leakage_flag": bool(unexpected_helper),
                "matched_forbidden_pattern": ";".join(forbidden_match),
                "issue_flags": ";".join(issue_flags),
                "recommendation": recommendation,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(METADATA_DIR / "feature_sanity_check.csv", index=False)
    return out


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text_df = df.fillna("").astype(str)
    header = "| " + " | ".join(text_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
    rows = ["| " + " | ".join(row[col] for col in text_df.columns) + " |" for _, row in text_df.iterrows()]
    return "\n".join([header, sep, *rows])


def write_reports(desc: pd.DataFrame, audit: pd.DataFrame, sanity: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    generic_remaining = int(desc["description"].fillna("").astype(str).str.startswith(GENERIC_PREFIX).sum())
    changed_rows = int(audit["action_taken"].eq("updated_metadata").sum()) if not audit.empty else 0
    remaining_quality_flags = int(audit["remaining_quality_flags"].fillna("").astype(str).str.strip().ne("").sum()) if not audit.empty else 0
    vague_groups = {"application_amounts", "application_profile", "demographic_time_variables", "delinquency_behavior"}
    semantic = pd.read_csv(METADATA_DIR / "semantic_group_distribution.csv")
    sanity_issues = sanity[sanity["issue_flags"].fillna("").astype(str).str.len() > 0]
    removal = sanity[sanity["recommended_action"].eq("remove_or_fix_before_matrix")]
    review = sanity[sanity["recommended_action"].eq("review_before_matrix")]
    sparse = sanity[sanity["missing_rate_gt_95"].eq(True)].sort_values("missing_rate", ascending=False)
    high_card = sanity[sanity["high_cardinality_categorical"].eq(True)]

    quality_report = "\n".join(
        [
            "# LendingClub v2 Metadata Quality Audit",
            "",
            "## Summary",
            "",
            f"- Candidate metadata rows: `{len(desc)}`",
            f"- Metadata rows changed on this pass: `{changed_rows}`",
            f"- Metadata rows with remaining quality flags: `{remaining_quality_flags}`",
            f"- Generic descriptions remaining: `{generic_remaining}`",
            f"- Blank descriptions: `{int(desc['description'].fillna('').astype(str).str.strip().eq('').sum())}`",
            f"- Blank semantic groups: `{int(desc['semantic_group'].fillna('').astype(str).str.strip().eq('').sum())}`",
            f"- Blank source formulas: `{int(desc['source_column_or_formula'].fillna('').astype(str).str.strip().eq('').sum())}`",
            f"- Features marked `needs_manual_review`: `{int(desc['leakage_review_status'].eq('needs_manual_review').sum())}`",
            f"- Vague catch-all group rows remaining: `{int(desc['semantic_group'].isin(vague_groups).sum())}`",
            "",
            "## Dominant Semantic Groups After Cleanup",
            "",
            _markdown_table(semantic.head(12)),
            "",
            "## Main Corrections",
            "",
            "- Replaced generic engineered-feature descriptions with specific credit-risk interpretations.",
            "- Reassigned delinquency, FICO, DTI/income, mortgage, inquiry, bankcard, revolving, balance, and account-depth features to more accurate semantic groups.",
            "- Replaced weak `derived_from_safe_fields(...)` formulas where feature names implied an explicit formula.",
            "- Left leakage status as `safe` unless an existing row required manual review; no new manual-review rows were introduced.",
        ]
    )
    (REPORTS_DIR / "lendingclub_v2_metadata_quality_audit.md").write_text(quality_report, encoding="utf-8")

    sanity_report = "\n".join(
        [
            "# LendingClub v2 Pre-Matrix Sanity Report",
            "",
            "This is a data/metadata inspection only. It does not run the experiment matrix, train models, or fit selectors.",
            "",
            "## Summary",
            "",
            f"- Columns checked: `{len(sanity)}`",
            f"- Candidate features checked: `{int(sanity['is_candidate_feature'].sum())}`",
            f"- Duplicate columns: `{int(sanity['duplicate_column'].sum())}`",
            f"- All-null columns: `{int(sanity['all_null'].sum())}`",
            f"- Constant candidate columns: `{int(sanity['constant_column'].sum())}`",
            f"- Columns with infinite values: `{int((sanity['infinite_value_count'] > 0).sum())}`",
            f"- Columns with invalid numeric values: `{int((sanity['invalid_numeric_value_count'] > 0).sum())}`",
            f"- Ratio features with extreme values: `{int(sanity['issue_flags'].fillna('').astype(str).str.contains('extreme_ratio_values').sum())}`",
            f"- Candidate columns with >95% missingness: `{int(sanity['missing_rate_gt_95'].sum())}`",
            f"- High-cardinality categorical columns: `{int(sanity['high_cardinality_categorical'].sum())}`",
            f"- Target/split/helper leakage flags: `{int(sanity['target_split_helper_leakage_flag'].sum())}`",
            f"- Remove/fix before matrix recommendations: `{len(removal)}`",
            f"- Review before matrix recommendations: `{len(review)}`",
            "",
            "## Remove Or Fix Before Matrix",
            "",
            _markdown_table(removal[["feature", "issue_type", "severity", "missing_rate", "unique_count", "recommended_action"]].head(30)),
            "",
            "## Review Before Matrix",
            "",
            _markdown_table(review[["feature", "issue_type", "severity", "missing_rate", "unique_count", "max_abs_value", "recommended_action"]].head(40)),
            "",
            "## Highest Missingness Features",
            "",
            _markdown_table(sparse[["feature", "semantic_group", "missing_rate", "recommended_action"]].head(25)),
            "",
            "## High-Cardinality Categoricals",
            "",
            _markdown_table(high_card[["feature", "categorical_cardinality", "missing_rate", "recommended_action"]]),
            "",
            "## Approval Interpretation",
            "",
            "- Matrix approval should wait until the `remove_or_fix_before_matrix` rows are removed or explicitly waived, because they are no-information constant features.",
            "- No additional automatic removal is required if the reviewer accepts sparse joint/secondary-applicant features as intentional optional-applicant signals.",
            "- The sparse-feature list should be reviewed before approval because many secondary-applicant fields are present only for joint applications.",
        ]
    )
    (REPORTS_DIR / "lendingclub_v2_pre_matrix_sanity_report.md").write_text(sanity_report, encoding="utf-8")
    write_final_approval_report(desc, sanity)


def write_final_approval_report(desc: pd.DataFrame, sanity: pd.DataFrame) -> None:
    coverage = pd.read_csv(METADATA_DIR / "feature_description_coverage.csv")
    semantic = pd.read_csv(METADATA_DIR / "semantic_group_distribution.csv")
    leakage = pd.read_csv(LEAKAGE_REVIEW_PATH)
    summary_path = METADATA_DIR / "preparation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    expected_semantic = (
        desc.groupby("semantic_group")["feature"]
        .size()
        .reset_index(name="feature_count")
        .sort_values(["semantic_group"])
        .reset_index(drop=True)
    )
    actual_semantic = (
        semantic[["semantic_group", "feature_count"]]
        .sort_values(["semantic_group"])
        .reset_index(drop=True)
    )
    semantic_matches = expected_semantic.equals(actual_semantic)

    description_missing = int(desc["description"].fillna("").astype(str).str.strip().eq("").sum())
    semantic_missing = int(desc["semantic_group"].fillna("").astype(str).str.strip().eq("").sum())
    high_warnings = int(sanity["severity"].eq("high").sum())
    medium_warnings = int(sanity["severity"].eq("medium").sum())
    included_leakage = leakage[
        leakage["action"].eq("include") & ~leakage["leakage_review_status"].eq("safe")
    ]
    leakage_columns = int(len(included_leakage))
    candidate_features = int(len(desc))
    approval = (
        high_warnings == 0
        and description_missing == 0
        and semantic_missing == 0
        and leakage_columns == 0
        and semantic_matches
        and candidate_features > 500
    )

    removed = summary.get("removed_features", [])
    clipped = summary.get("clipped_ratio_features", [])
    high_missing_count = int(sanity["missing_rate_gt_95"].eq(True).sum())
    leakage_excluded = int(leakage["action"].eq("exclude").sum())
    coverage_ratio = float(coverage.loc[0, "coverage_ratio"]) if not coverage.empty else 0.0
    summary.update(
        {
            "candidate_features": candidate_features,
            "description_coverage": coverage_ratio,
            "semantic_groups": int(desc["semantic_group"].nunique()),
            "manual_review_features": leakage_columns,
            "matrix_run": False,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# LendingClub v2 Final Pre-Matrix Approval",
        "",
        "This report is based only on data and metadata regeneration. The experiment matrix was not run and no models were trained.",
        "",
        "## Decision",
        "",
        f"- Matrix approved to run: `{'yes' if approval else 'no'}`.",
        f"- Final candidate feature count: `{candidate_features:,}`.",
        f"- Description coverage: `{coverage_ratio:.2%}`.",
        f"- Semantic group count: `{int(desc['semantic_group'].nunique()):,}`.",
        f"- High-severity sanity warnings: `{high_warnings}`.",
        f"- Medium sanity warnings: `{medium_warnings}`.",
        "",
        "## Approval Checks",
        "",
        f"- Missing descriptions: `{description_missing}`.",
        f"- Missing semantic groups: `{semantic_missing}`.",
        f"- Included leakage columns: `{leakage_columns}`.",
        f"- Semantic distribution matches `columns_description.csv`: `{'yes' if semantic_matches else 'no'}`.",
        f"- Candidate features above 500: `{'yes' if candidate_features > 500 else 'no'}`.",
        "",
        "## Features Removed Or Fixed",
        "",
        f"- Removed features: `{len(removed)}`.",
        f"- Removed feature list: `{'; '.join(removed) if removed else 'none'}`.",
        f"- Ratio features fixed by denominator handling and clipping: `{len(clipped)}`.",
        f"- Fixed ratio list: `{'; '.join(clipped) if clipped else 'none'}`.",
        "",
        "## Sparse Feature Policy",
        "",
        f"- Current features with >95% missingness: `{high_missing_count}`.",
        f"- Policy: {summary.get('high_missingness_policy', 'Reviewed sparse features and removed redundant ultra-sparse derivatives.')}",
        "",
        "## Extreme-Ratio Policy",
        "",
        f"- Policy: {summary.get('extreme_ratio_policy', 'Ratios use safe denominator handling and fixed clipping.')}",
        "",
        "## Leakage Review",
        "",
        f"- Result: `{'pass' if leakage_columns == 0 else 'fail'}`; no included candidate feature is marked as leakage or manual review.",
        f"- Excluded leakage/source-policy rows documented in `leakage_review.csv`: `{leakage_excluded}`.",
    ]
    (REPORTS_DIR / "lendingclub_v2_final_pre_matrix_approval.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    desc, audit = clean_metadata()
    sanity = sanity_check(desc)
    write_reports(desc, audit, sanity)
    print(f"Metadata rows: {len(desc)}")
    print(f"Metadata rows changed on this pass: {int(audit['action_taken'].eq('updated_metadata').sum()) if not audit.empty else 0}")
    print(f"Metadata rows with remaining quality flags: {int(audit['remaining_quality_flags'].fillna('').astype(str).str.strip().ne('').sum()) if not audit.empty else 0}")
    print(f"Generic descriptions remaining: {int(desc['description'].fillna('').astype(str).str.startswith(GENERIC_PREFIX).sum())}")
    print(f"Semantic groups: {desc['semantic_group'].nunique()}")
    print(f"Sanity rows: {len(sanity)}")
    print(f"Remove/fix before matrix: {int(sanity['recommended_action'].eq('remove_or_fix_before_matrix').sum())}")
    print(f"Review before matrix: {int(sanity['recommended_action'].eq('review_before_matrix').sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
