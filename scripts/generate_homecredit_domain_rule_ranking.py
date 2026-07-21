"""Generate a deterministic Home Credit ranking from explicit domain rules.

This module makes no LLM or network call. It reads feature metadata only and
applies the checked-in exact, family, aggregation, and fallback rules below.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "results/corrected_homecredit_clip/feature_universe/"
    "feature_universe_reconciliation.csv"
)
OUTPUT_DIR = Path("results/domain_rule_rankings")
OUTPUT_PATH = OUTPUT_DIR / "homecredit_full_529_domain_rule_ranking.csv"
SUMMARY_PATH = OUTPUT_DIR / "homecredit_full_529_domain_rule_ranking_validation.txt"

REQUIRED_INPUT_COLUMNS = [
    "feature_id",
    "feature_name",
    "source_table",
    "semantic_group",
]
OUTPUT_COLUMNS = [
    "feature_id",
    "feature_name",
    "source_table",
    "semantic_group",
    "feature_family",
    "credit_relevance_score",
    "risk_concept",
    "reason_short",
    "rank",
    "normalized_rank",
    "leakage_status",
]

AGG_SUFFIXES = ("_MEAN", "_MAX", "_MIN", "_SUM", "_VAR", "_AVG", "_MEDI", "_MODE")
FORBIDDEN_COLUMN_MARKERS = (
    "target",
    "auc",
    "ks",
    "psi",
    "gini",
    "iv",
    "mrmr",
    "selected",
    "fold",
    "performance",
    "bad_rate",
    "label",
)

EXACT_RULES = {
    "AMT_INCOME_TOTAL": (
        91,
        "current_income_capacity",
        "repayment_capacity",
        "Total income is a direct semantic indicator of repayment capacity.",
    ),
    "AMT_ANNUITY": (
        89,
        "current_application_amount",
        "debt_burden",
        "Requested annuity summarizes the recurring payment burden of the current application.",
    ),
    "AMT_CREDIT": (
        88,
        "current_application_amount",
        "debt_burden",
        "Requested credit amount summarizes the size of the new exposure.",
    ),
    "AMT_GOODS_PRICE": (
        84,
        "current_application_amount",
        "application_quality",
        "Goods price summarizes the financed purchase size behind the application.",
    ),
    "DAYS_EMPLOYED": (
        85,
        "employment_stability",
        "repayment_capacity",
        "Employment tenure summarizes stability of repayment capacity.",
    ),
    "NAME_INCOME_TYPE": (
        82,
        "income_source_type",
        "repayment_capacity",
        "Income type summarizes the source and stability of borrower cash flow.",
    ),
    "OCCUPATION_TYPE": (
        78,
        "occupation_context",
        "repayment_capacity",
        "Occupation type provides contextual information about income stability.",
    ),
    "NAME_CONTRACT_TYPE": (
        75,
        "current_contract_type",
        "application_quality",
        "Contract type summarizes the form of the requested credit product.",
    ),
    "REGION_RATING_CLIENT_W_CITY": (
        78,
        "regional_risk_context",
        "demographic_or_household_context",
        "Regional rating with city context gives coarse borrower-environment risk context.",
    ),
    "REGION_RATING_CLIENT": (
        77,
        "regional_risk_context",
        "demographic_or_household_context",
        "Regional rating gives coarse borrower-environment risk context.",
    ),
    "ORGANIZATION_TYPE": (
        71,
        "employment_context",
        "repayment_capacity",
        "Employer organization type gives contextual information about income stability.",
    ),
    "NAME_EDUCATION_TYPE": (
        70,
        "education_context",
        "demographic_or_household_context",
        "Education type is an indirect socioeconomic context signal.",
    ),
    "DAYS_BIRTH": (
        68,
        "age_context",
        "demographic_or_household_context",
        "Borrower age is an indirect lifecycle and stability context signal.",
    ),
    "DAYS_LAST_PHONE_CHANGE": (
        67,
        "contact_stability",
        "application_quality",
        "Recent phone-change timing can indicate contact stability.",
    ),
    "DAYS_REGISTRATION": (
        66,
        "registration_stability",
        "application_quality",
        "Registration age gives an indirect stability signal.",
    ),
    "DAYS_ID_PUBLISH": (
        63,
        "identity_document_age",
        "application_quality",
        "Identity-document age is a weak application stability signal.",
    ),
    "OWN_CAR_AGE": (
        60,
        "asset_age_context",
        "demographic_or_household_context",
        "Car age provides weak asset and household context.",
    ),
    "REGION_POPULATION_RELATIVE": (
        59,
        "regional_population_context",
        "demographic_or_household_context",
        "Regional population density is an indirect borrower-environment signal.",
    ),
    "CODE_GENDER": (
        54,
        "demographic_context",
        "demographic_or_household_context",
        "Gender is a demographic context field with weak direct credit-risk meaning.",
    ),
    "NAME_FAMILY_STATUS": (
        57,
        "household_context",
        "demographic_or_household_context",
        "Family status provides weak household stability context.",
    ),
    "DEF_60_CNT_SOCIAL_CIRCLE": (
        80,
        "social_circle_default_context",
        "delinquency_history",
        "Social-circle default counts provide indirect delinquency environment context.",
    ),
    "DEF_30_CNT_SOCIAL_CIRCLE": (
        79,
        "social_circle_default_context",
        "delinquency_history",
        "Social-circle default counts provide indirect delinquency environment context.",
    ),
    "OBS_60_CNT_SOCIAL_CIRCLE": (
        60,
        "social_circle_observation_context",
        "demographic_or_household_context",
        "Observed social-circle counts provide weak neighborhood or peer context.",
    ),
    "OBS_30_CNT_SOCIAL_CIRCLE": (
        59,
        "social_circle_observation_context",
        "demographic_or_household_context",
        "Observed social-circle counts provide weak neighborhood or peer context.",
    ),
    "CNT_FAM_MEMBERS": (
        59,
        "household_context",
        "demographic_or_household_context",
        "Family-member count provides weak household burden context.",
    ),
    "CNT_CHILDREN": (
        57,
        "household_context",
        "demographic_or_household_context",
        "Children count provides weak household obligation context.",
    ),
    "FLAG_OWN_REALTY": (
        62,
        "asset_ownership_context",
        "demographic_or_household_context",
        "Realty ownership is an asset and housing stability context signal.",
    ),
    "FLAG_OWN_CAR": (
        58,
        "asset_ownership_context",
        "demographic_or_household_context",
        "Car ownership is a weak asset context signal.",
    ),
    "NAME_HOUSING_TYPE": (
        55,
        "housing_context",
        "demographic_or_household_context",
        "Housing type provides indirect household stability context.",
    ),
    "NAME_TYPE_SUITE": (
        50,
        "application_companion_context",
        "weak_or_administrative_signal",
        "Accompanying-party type is weakly related to borrower risk.",
    ),
    "WEEKDAY_APPR_PROCESS_START": (
        48,
        "application_timing_admin",
        "weak_or_administrative_signal",
        "Application weekday is mostly administrative timing context.",
    ),
    "HOUR_APPR_PROCESS_START": (
        47,
        "application_timing_admin",
        "weak_or_administrative_signal",
        "Application hour is mostly administrative timing context.",
    ),
    "FLAG_EMAIL": (
        49,
        "contact_flag",
        "application_quality",
        "Contact flags provide weak borrower reachability and employment-contact context.",
    ),
    "FLAG_PHONE": (
        50,
        "contact_flag",
        "application_quality",
        "Contact flags provide weak borrower reachability and employment-contact context.",
    ),
    "FLAG_WORK_PHONE": (
        52,
        "contact_flag",
        "application_quality",
        "Contact flags provide weak borrower reachability and employment-contact context.",
    ),
    "FLAG_EMP_PHONE": (
        51,
        "contact_flag",
        "application_quality",
        "Contact flags provide weak borrower reachability and employment-contact context.",
    ),
    "FLAG_MOBIL": (
        35,
        "contact_flag",
        "technical_or_low_signal",
        "Mobile-contact flags are near-administrative availability indicators.",
    ),
    "FLAG_CONT_MOBILE": (
        37,
        "contact_flag",
        "technical_or_low_signal",
        "Mobile-contact flags are near-administrative availability indicators.",
    ),
    "EMERGENCYSTATE_MODE": (
        48,
        "housing_quality_context",
        "demographic_or_household_context",
        "Housing-quality categories provide indirect household context.",
    ),
    "FONDKAPREMONT_MODE": (
        49,
        "housing_quality_context",
        "demographic_or_household_context",
        "Housing-quality categories provide indirect household context.",
    ),
    "HOUSETYPE_MODE": (
        51,
        "housing_quality_context",
        "demographic_or_household_context",
        "Housing-quality categories provide indirect household context.",
    ),
    "WALLSMATERIAL_MODE": (
        50,
        "housing_quality_context",
        "demographic_or_household_context",
        "Housing-quality categories provide indirect household context.",
    ),
}

PROPERTY_TERMS = {
    "TOTALAREA": 58,
    "LIVINGAREA": 57,
    "APARTMENTS": 55,
    "LIVINGAPARTMENTS": 54,
    "FLOORSMAX": 53,
    "ELEVATORS": 52,
    "FLOORSMIN": 51,
    "BASEMENTAREA": 50,
    "LANDAREA": 50,
    "ENTRANCES": 49,
    "COMMONAREA": 48,
    "NONLIVINGAREA": 47,
    "NONLIVINGAPARTMENTS": 46,
    "YEARS_BUILD": 50,
    "YEARS_BEGINEXPLUATATION": 49,
}

FALLBACKS = {
    "bureau_debt": (
        82,
        "bureau_debt_exposure",
        "debt_burden",
        "Bureau debt metadata summarizes external borrowing burden.",
    ),
    "bureau_credit_history": (
        72,
        "bureau_credit_history",
        "bureau_credit_history",
        "Bureau history metadata summarizes external credit experience.",
    ),
    "credit_card_utilization": (
        78,
        "credit_card_utilization",
        "credit_utilization",
        "Credit-card utilization metadata summarizes revolving credit behavior.",
    ),
    "installment_repayment_behavior": (
        76,
        "installment_behavior",
        "installment_behavior",
        "Installment metadata summarizes prior repayment behavior.",
    ),
    "previous_application_behavior": (
        68,
        "previous_application_behavior",
        "prior_application_behavior",
        "Previous-application metadata summarizes prior borrowing behavior.",
    ),
    "income_capacity": (
        74,
        "income_capacity",
        "repayment_capacity",
        "Income-capacity metadata summarizes repayment capacity context.",
    ),
    "delinquency_behavior": (
        88,
        "delinquency_behavior",
        "delinquency_history",
        "Delinquency metadata summarizes late or overdue repayment behavior.",
    ),
    "demographic_time_variables": (
        52,
        "demographic_time_context",
        "demographic_or_household_context",
        "Demographic timing metadata is an indirect context signal.",
    ),
    "application_amounts": (
        52,
        "application_context",
        "demographic_or_household_context",
        "Application context metadata is indirectly related to borrower risk.",
    ),
}


def strip_agg(name: str) -> tuple[str, str]:
    upper = name.upper()
    for suffix in AGG_SUFFIXES:
        if upper.endswith(suffix):
            return name[: -len(suffix)], suffix[1:].lower()
    return name, ""


def rule(
    score: int,
    family: str,
    concept: str,
    reason: str,
) -> tuple[int, str, str, str]:
    return score, family, concept, reason


def classify(name: str, semantic_group: str) -> tuple[int, str, str, str]:
    root, _ = strip_agg(name)
    full = name.upper()
    u = root.upper()

    if full in EXACT_RULES:
        return EXACT_RULES[full]
    if full.startswith("EXT_SOURCE_"):
        return rule(
            100,
            "external_score",
            "bureau_credit_history",
            "External score fields encode third-party credit-risk assessment at application time.",
        )
    if full.startswith("AMT_REQ_CREDIT_BUREAU_"):
        period = full.rsplit("_", 1)[-1]
        score = {
            "YEAR": 86,
            "MON": 84,
            "QRT": 82,
            "WEEK": 78,
            "DAY": 76,
            "HOUR": 74,
        }.get(period, 76)
        return rule(
            score,
            "recent_credit_inquiries",
            "bureau_credit_history",
            "Recent bureau inquiry counts summarize active credit-seeking behavior.",
        )
    if "SK_DPD_DEF" in u:
        return rule(
            98,
            "delinquency_days_past_due",
            "delinquency_history",
            "Days-past-due default summaries indicate prior delinquency severity.",
        )
    if "SK_DPD" in u:
        return rule(
            97,
            "delinquency_days_past_due",
            "delinquency_history",
            "Days-past-due summaries indicate prior delinquency behavior.",
        )
    if "LATE_PAYMENT" in u or "IS_LATE" in u:
        return rule(
            98,
            "late_payment_history",
            "payment_consistency",
            "Late-payment summaries directly capture missed or delayed repayment behavior.",
        )
    if "IS_UNDERPAID" in u:
        return rule(
            96,
            "underpayment_history",
            "payment_consistency",
            "Underpayment summaries capture incomplete repayment behavior.",
        )
    if "CREDIT_DAY_OVERDUE" in u:
        return rule(
            97,
            "bureau_days_overdue",
            "delinquency_history",
            "Bureau days-overdue summaries indicate delinquent external credit history.",
        )
    if "CREDIT_SUM_OVERDUE" in u:
        return rule(
            97,
            "bureau_overdue_amount",
            "delinquency_history",
            "Overdue bureau credit amounts indicate unpaid delinquent obligations.",
        )
    if "CREDIT_MAX_OVERDUE" in u:
        return rule(
            96,
            "bureau_overdue_amount",
            "delinquency_history",
            "Maximum overdue bureau amounts summarize severe prior arrears.",
        )
    if "PAYMENT_RATIO" in u:
        return rule(
            95,
            "installment_payment_ratio",
            "payment_consistency",
            "Payment-to-scheduled-amount ratios summarize repayment completeness.",
        )
    if "PAYMENT_DIFF" in u:
        return rule(
            94,
            "installment_lateness",
            "payment_consistency",
            "Payment shortfall or excess amounts summarize installment repayment consistency.",
        )
    if "DAYS_DIFF" in u:
        return rule(
            94,
            "installment_lateness",
            "payment_consistency",
            "Payment timing differences summarize whether installments were paid early or late.",
        )
    if "PAYMENT_DIV_MIN" in u:
        return rule(
            93,
            "credit_card_payment_to_minimum",
            "payment_consistency",
            "Payments relative to minimum due summarize credit-card repayment discipline.",
        )
    if "DEBT_RATIO" in u:
        return rule(
            95,
            "bureau_debt_ratio",
            "debt_burden",
            "Bureau debt ratios summarize outstanding external debt burden.",
        )
    if "CREDIT_SUM_DEBT" in u:
        return rule(
            94,
            "bureau_debt_exposure",
            "debt_burden",
            "Outstanding bureau debt amounts summarize external repayment burden.",
        )
    if "DEBT_CREDIT_DIFF" in u:
        return rule(
            92,
            "bureau_debt_exposure",
            "debt_burden",
            "Debt-minus-credit differences summarize remaining borrowing headroom.",
        )
    if "LIMIT_USE" in u:
        return rule(
            94,
            "credit_card_limit_utilization",
            "credit_utilization",
            "Credit-card limit utilization summarizes revolving credit stress.",
        )
    if "DRAWING_RATIO" in u:
        return rule(
            93,
            "credit_card_drawing_ratio",
            "credit_utilization",
            "Credit-card drawing ratios summarize reliance on available revolving credit.",
        )
    if "AMT_BALANCE" in u:
        return rule(
            91,
            "credit_card_balance_debt",
            "debt_burden",
            "Credit-card balances summarize current revolving debt exposure.",
        )
    if "AMT_TOTAL_RECEIVABLE" in u:
        return rule(
            90,
            "credit_card_receivable_debt",
            "debt_burden",
            "Total receivables summarize unpaid credit-card obligations.",
        )
    if "AMT_RECEIVABLE_PRINCIPAL" in u:
        return rule(
            89,
            "credit_card_receivable_debt",
            "debt_burden",
            "Principal receivables summarize unpaid revolving principal exposure.",
        )
    if "AMT_RECIVABLE" in u:
        return rule(
            88,
            "credit_card_receivable_debt",
            "debt_burden",
            "Receivable amounts summarize unpaid revolving obligations.",
        )
    if "AMT_CREDIT_LIMIT_ACTUAL" in u:
        return rule(
            86,
            "credit_card_credit_limit",
            "credit_utilization",
            "Credit-card limits summarize available revolving credit exposure.",
        )
    if "AMT_DRAWINGS" in u:
        return rule(
            85,
            "credit_card_drawings",
            "credit_utilization",
            "Drawn credit-card amounts summarize recent revolving-credit usage.",
        )
    if "CNT_DRAWINGS" in u:
        return rule(
            81,
            "credit_card_drawings",
            "credit_utilization",
            "Credit-card drawing counts summarize frequency of revolving-credit use.",
        )
    if "AMT_INST_MIN_REGULARITY" in u:
        return rule(
            82,
            "credit_card_minimum_payment",
            "payment_consistency",
            "Minimum regular installments summarize required card repayment burden.",
        )
    if "AMT_PAYMENT_TOTAL_CURRENT" in u or "AMT_PAYMENT_CURRENT" in u:
        return rule(
            84,
            "credit_card_payment_amount",
            "payment_consistency",
            "Credit-card payment amounts summarize repayment behavior on revolving accounts.",
        )
    if "CNT_INSTALMENT_MATURE_CUM" in u:
        return rule(
            75,
            "credit_card_account_maturity",
            "prior_credit_behavior",
            "Matured installment counts summarize account seasoning but are indirect risk signals.",
        )
    if full.startswith("CC_MONTHS_BALANCE"):
        return rule(
            62,
            "credit_card_month_history",
            "prior_credit_behavior",
            "Credit-card month-balance summaries describe history length more than repayment quality.",
        )
    if "AMT_CREDIT_SUM_LIMIT" in u:
        return rule(
            84,
            "bureau_credit_limit",
            "credit_utilization",
            "Bureau credit limits summarize external borrowing capacity and utilization context.",
        )
    if "AMT_CREDIT_SUM" in u:
        return rule(
            88,
            "bureau_credit_exposure",
            "debt_burden",
            "Bureau credit sums summarize external credit exposure.",
        )
    if full.startswith("BURO_") and "AMT_ANNUITY" in u:
        return rule(
            86,
            "bureau_annuity",
            "debt_burden",
            "Bureau annuity amounts summarize recurring external repayment burden.",
        )
    if "CNT_CREDIT_PROLONG" in u:
        return rule(
            84,
            "bureau_credit_prolongation",
            "prior_credit_behavior",
            "Credit prolongation counts summarize difficulty completing prior external credit as scheduled.",
        )
    if "CREDIT_DURATION" in u:
        return rule(
            80,
            "bureau_credit_duration",
            "bureau_credit_history",
            "Credit duration summaries capture the tenor of external borrowing history.",
        )
    if "DAYS_CREDIT_UPDATE" in u:
        return rule(
            77,
            "bureau_credit_update",
            "bureau_credit_history",
            "Recent bureau updates summarize freshness and activity of credit history.",
        )
    if "DAYS_ENDDATE_FACT" in u:
        return rule(
            79,
            "bureau_credit_enddate",
            "bureau_credit_history",
            "Actual bureau credit end dates summarize completed external loan history.",
        )
    if "DAYS_CREDIT_ENDDATE" in u:
        return rule(
            78,
            "bureau_credit_enddate",
            "bureau_credit_history",
            "Expected bureau credit end dates summarize remaining external credit tenor.",
        )
    if "DAYS_CREDIT" in u:
        return rule(
            76,
            "bureau_credit_recency",
            "bureau_credit_history",
            "Bureau credit timing summarizes recency of external borrowing activity.",
        )
    if "BURO_BB_MONTHS_BALANCE" in u:
        return rule(
            70,
            "bureau_balance_history_length",
            "bureau_credit_history",
            "Bureau-balance month summaries capture depth and recency of external account history.",
        )
    if "INTEREST_ESTIMATE" in u or "RATE_INTEREST_PRIMARY" in u:
        return rule(
            88,
            "previous_interest_or_price",
            "prior_application_behavior",
            "Prior interest or price terms summarize risk pricing on earlier applications.",
        )
    if "RATE_INTEREST_PRIVILEGED" in u:
        return rule(
            84,
            "previous_interest_or_price",
            "prior_application_behavior",
            "Privileged interest-rate terms summarize prior credit pricing context.",
        )
    if "ASK_GRANT_RATIO" in u:
        return rule(
            88,
            "previous_ask_grant_ratio",
            "prior_application_behavior",
            "Requested-to-granted ratios summarize how earlier applications were adjusted by lenders.",
        )
    if "APPLICATION_DIFF" in u:
        return rule(
            86,
            "previous_application_amount_gap",
            "prior_application_behavior",
            "Prior application-credit differences summarize mismatch between requested and granted credit.",
        )
    if "RATE_DOWN_PAYMENT" in u:
        return rule(
            83,
            "previous_down_payment_rate",
            "prior_application_behavior",
            "Prior down-payment rates summarize borrower contribution and application quality.",
        )
    if "AMT_DOWN_PAYMENT" in u:
        return rule(
            83,
            "previous_down_payment_amount",
            "prior_application_behavior",
            "Prior down-payment amounts summarize borrower contribution to previous credit.",
        )
    if full.startswith("PREV_AMT_CREDIT"):
        return rule(
            85,
            "previous_credit_amount",
            "prior_application_behavior",
            "Previous credit amounts summarize historical granted exposure.",
        )
    if full.startswith("PREV_AMT_APPLICATION"):
        return rule(
            84,
            "previous_application_amount",
            "prior_application_behavior",
            "Previous requested amounts summarize historical credit demand.",
        )
    if full.startswith("PREV_AMT_ANNUITY"):
        return rule(
            85,
            "previous_annuity",
            "debt_burden",
            "Previous annuity amounts summarize recurring burden on earlier credits.",
        )
    if full.startswith("PREV_AMT_GOODS_PRICE"):
        return rule(
            82,
            "previous_goods_price",
            "prior_application_behavior",
            "Previous goods prices summarize financed purchase sizes in prior applications.",
        )
    if "CNT_PAYMENT" in u or "PAYMENT_TERM" in u:
        return rule(
            82,
            "previous_payment_term",
            "prior_application_behavior",
            "Prior payment terms summarize tenor and repayment schedule length.",
        )
    if "RECENT_DECISION" in full:
        return rule(
            80,
            "previous_recent_decision",
            "prior_application_behavior",
            "Recent previous-application decisions summarize current credit-seeking history.",
        )
    if "DAYS_DECISION" in u:
        return rule(
            78,
            "previous_application_timing",
            "prior_application_behavior",
            "Timing of prior decisions summarizes recency of previous credit applications.",
        )
    if "DAYS_TERMINATION" in u:
        return rule(
            75,
            "previous_due_date_timing",
            "prior_application_behavior",
            "Prior termination timing summarizes lifecycle history of earlier credits.",
        )
    if "DAYS_LAST_DUE_1ST_VERSION" in u:
        return rule(
            74,
            "previous_due_date_timing",
            "prior_application_behavior",
            "Original last-due timing summarizes scheduled tenor of prior credit.",
        )
    if "DAYS_LAST_DUE" in u:
        return rule(
            73,
            "previous_due_date_timing",
            "prior_application_behavior",
            "Last-due timing summarizes maturity of prior credit schedules.",
        )
    if "DAYS_FIRST_DUE" in u:
        return rule(
            71,
            "previous_due_date_timing",
            "prior_application_behavior",
            "First-due timing summarizes initial repayment schedule of prior credit.",
        )
    if "DAYS_FIRST_DRAWING" in u:
        return rule(
            69,
            "previous_draw_timing",
            "prior_application_behavior",
            "First drawing timing is an indirect prior credit usage signal.",
        )
    if "NFLAG_INSURED_ON_APPROVAL" in u:
        return rule(
            65,
            "previous_insurance_flag",
            "prior_application_behavior",
            "Insurance-on-approval flags provide weak prior application quality context.",
        )
    if "NFLAG_LAST_APPL_IN_DAY" in u:
        return rule(
            50,
            "administrative_flag",
            "weak_or_administrative_signal",
            "Last-application-in-day flags are mostly administrative application metadata.",
        )
    if full.startswith("PREV_") and "HOUR_APPR_PROCESS_START" in u:
        return rule(
            54,
            "application_timing_admin",
            "weak_or_administrative_signal",
            "Prior application hour is mostly administrative timing context.",
        )
    if "SELLERPLACE_AREA" in u:
        return rule(
            45,
            "sellerplace_area",
            "weak_or_administrative_signal",
            "Seller-place area is only indirectly related to borrower repayment risk.",
        )
    if full.startswith("POS_CNT_INSTALMENT_FUTURE"):
        return rule(
            70,
            "pos_remaining_installments",
            "installment_behavior",
            "Remaining installment counts summarize outstanding POS repayment schedule.",
        )
    if full.startswith("POS_CNT_INSTALMENT"):
        return rule(
            68,
            "pos_installment_count",
            "installment_behavior",
            "POS installment counts summarize repayment schedule scale.",
        )
    if full.startswith("POS_TOTAL_INSTALMENT_PROG"):
        return rule(
            72,
            "pos_installment_progress",
            "installment_behavior",
            "POS installment progress summarizes how far prior repayment schedules had advanced.",
        )
    if full.startswith("POS_MONTHS_BALANCE"):
        return rule(
            57,
            "pos_month_history",
            "prior_credit_behavior",
            "POS month-balance summaries mostly describe account history length.",
        )
    if full.startswith("INSTAL_AMT_PAYMENT"):
        return rule(
            86,
            "installment_payment_amount",
            "installment_behavior",
            "Installment payment amounts summarize actual repayment behavior.",
        )
    if full.startswith("INSTAL_AMT_INSTALMENT"):
        return rule(
            85,
            "installment_scheduled_amount",
            "debt_burden",
            "Scheduled installment amounts summarize required repayment burden.",
        )
    if "DAYS_ENTRY_PAYMENT" in u:
        return rule(
            78,
            "installment_schedule_timing",
            "installment_behavior",
            "Payment-entry timing summarizes when installment payments were recorded.",
        )
    if "DAYS_INSTALMENT" in u:
        return rule(
            74,
            "installment_schedule_timing",
            "installment_behavior",
            "Scheduled installment timing summarizes prior repayment calendar structure.",
        )
    if "NUM_INSTALMENT_VERSION" in u:
        return rule(
            66,
            "installment_version_count",
            "technical_or_low_signal",
            "Installment version counts are indirect schedule metadata.",
        )
    if "NUM_INSTALMENT_NUMBER" in u:
        return rule(
            64,
            "installment_sequence_count",
            "technical_or_low_signal",
            "Installment sequence counts are mostly schedule metadata.",
        )
    if full.startswith("REG_CITY_NOT") or full.startswith("LIVE_CITY_NOT"):
        return rule(
            62,
            "residence_work_mismatch",
            "demographic_or_household_context",
            "City residence-work mismatch can indicate mobility and stability context.",
        )
    if full.startswith("REG_REGION_NOT") or full.startswith("LIVE_REGION_NOT"):
        return rule(
            58,
            "residence_work_mismatch",
            "demographic_or_household_context",
            "Region residence-work mismatch is a coarse stability context signal.",
        )
    if full.startswith("FLAG_DOCUMENT_"):
        doc_num = int(full.rsplit("_", 1)[-1])
        score = {
            3: 46,
            6: 43,
            8: 42,
            13: 40,
            14: 40,
            16: 40,
            18: 39,
            5: 38,
            9: 37,
            11: 37,
            2: 35,
            10: 34,
            12: 34,
            15: 34,
            17: 34,
            19: 33,
            20: 33,
            21: 33,
            4: 32,
            7: 32,
        }.get(doc_num, 34)
        return rule(
            score,
            "document_flag",
            "weak_or_administrative_signal",
            "Document flags are mostly administrative application indicators.",
        )
    for term, score in PROPERTY_TERMS.items():
        if term in u:
            return rule(
                score,
                "property_area_context",
                "demographic_or_household_context",
                "Property characteristics provide indirect housing and wealth context.",
            )
    return FALLBACKS.get(
        semantic_group,
        (
            40,
            "technical_or_low_signal",
            "technical_or_low_signal",
            "The feature name provides only weak or administrative credit-risk meaning.",
        ),
    )


def agg_adjustment(family: str, agg: str) -> float:
    if not agg:
        return 0.0
    high_bad = {
        "delinquency_days_past_due",
        "late_payment_history",
        "underpayment_history",
        "bureau_overdue_amount",
        "bureau_days_overdue",
        "installment_lateness",
    }
    ratios = {
        "credit_card_limit_utilization",
        "credit_card_drawing_ratio",
        "installment_payment_ratio",
        "bureau_debt_ratio",
        "previous_ask_grant_ratio",
        "previous_down_payment_rate",
        "credit_card_payment_to_minimum",
    }
    exposure = {
        "bureau_debt_exposure",
        "bureau_credit_exposure",
        "bureau_annuity",
        "credit_card_balance_debt",
        "credit_card_receivable_debt",
        "credit_card_drawings",
        "credit_card_credit_limit",
        "credit_card_payment_amount",
        "installment_payment_amount",
        "installment_scheduled_amount",
        "previous_application_amount",
        "previous_credit_amount",
        "previous_annuity",
        "previous_goods_price",
        "previous_down_payment_amount",
        "current_application_amount",
        "current_income_capacity",
        "previous_interest_or_price",
    }
    timing = {
        "bureau_credit_recency",
        "bureau_credit_enddate",
        "bureau_credit_update",
        "bureau_credit_duration",
        "previous_application_timing",
        "previous_due_date_timing",
        "installment_schedule_timing",
        "credit_card_month_history",
        "pos_month_history",
    }
    low_signal = {
        "property_area_context",
        "housing_quality_context",
        "document_flag",
        "contact_flag",
        "application_timing_admin",
        "sellerplace_area",
        "administrative_flag",
        "credit_card_account_maturity",
        "installment_sequence_count",
        "installment_version_count",
    }
    if family in high_bad:
        return {
            "max": 2.4,
            "sum": 2.1,
            "mean": 1.8,
            "var": 0.5,
            "min": -4.0,
        }.get(agg, 0.0)
    if family in ratios:
        return {
            "mean": 2.2,
            "max": 2.0,
            "sum": 0.5,
            "var": 0.3,
            "min": -2.6,
        }.get(agg, 0.0)
    if family in exposure:
        return {
            "sum": 2.1,
            "mean": 1.8,
            "max": 1.5,
            "var": 0.4,
            "min": -2.0,
        }.get(agg, 0.0)
    if family in timing:
        return {
            "mean": 1.4,
            "max": 1.2,
            "min": 0.6,
            "var": 0.1,
            "sum": -1.0,
        }.get(agg, 0.0)
    if family in low_signal:
        return {
            "avg": 0.2,
            "medi": 0.1,
            "mode": 0.0,
            "mean": 0.2,
            "max": 0.1,
            "sum": -0.2,
            "var": -0.5,
            "min": -0.8,
        }.get(agg, 0.0)
    return {
        "mean": 0.8,
        "max": 0.7,
        "sum": 0.5,
        "var": -0.2,
        "min": -0.7,
        "avg": 0.2,
        "medi": 0.1,
        "mode": 0.0,
    }.get(agg, 0.0)


def agg_priority(family: str, agg: str) -> int:
    if not agg:
        return 5
    if family in {
        "delinquency_days_past_due",
        "late_payment_history",
        "underpayment_history",
        "bureau_overdue_amount",
        "bureau_days_overdue",
    }:
        order = {"max": 9, "sum": 8, "mean": 7, "var": 4, "min": 1}
    elif family in {
        "credit_card_limit_utilization",
        "credit_card_drawing_ratio",
        "installment_payment_ratio",
        "bureau_debt_ratio",
        "previous_ask_grant_ratio",
        "previous_down_payment_rate",
    }:
        order = {"mean": 9, "max": 8, "sum": 5, "var": 4, "min": 2}
    elif family in {
        "bureau_debt_exposure",
        "bureau_credit_exposure",
        "credit_card_balance_debt",
        "credit_card_receivable_debt",
        "credit_card_payment_amount",
        "installment_payment_amount",
        "previous_application_amount",
        "previous_credit_amount",
        "previous_annuity",
        "previous_goods_price",
        "current_application_amount",
    }:
        order = {"sum": 9, "mean": 8, "max": 7, "var": 4, "min": 2}
    elif family in {
        "bureau_credit_recency",
        "bureau_credit_enddate",
        "bureau_credit_update",
        "previous_application_timing",
        "previous_due_date_timing",
        "installment_schedule_timing",
    }:
        order = {"mean": 8, "max": 7, "min": 6, "var": 4, "sum": 2}
    else:
        order = {
            "mean": 7,
            "max": 6,
            "sum": 5,
            "avg": 4,
            "medi": 3,
            "mode": 3,
            "var": 2,
            "min": 1,
        }
    return order.get(agg, 0)


def leakage_status(name: str) -> str:
    upper = name.upper()
    if any(marker in upper for marker in ("TARGET", "LABEL", "BAD_RATE", "OUTCOME")):
        return "POTENTIAL_LEAKAGE_NAME_REVIEW_NEEDED"
    if "FUTURE" in upper:
        return "POTENTIAL_LEAKAGE_NAME_REVIEW_NEEDED"
    return "LEAKAGE_SAFE_LABEL_FREE"


def sort_bonus(name: str, family: str, concept: str, agg: str) -> float:
    root, _ = strip_agg(name)
    upper = root.upper()
    concept_priority = {
        "delinquency_history": 0.95,
        "payment_consistency": 0.90,
        "debt_burden": 0.85,
        "credit_utilization": 0.80,
        "repayment_capacity": 0.70,
        "bureau_credit_history": 0.65,
        "prior_application_behavior": 0.60,
        "installment_behavior": 0.58,
        "application_quality": 0.50,
        "demographic_or_household_context": 0.30,
        "weak_or_administrative_signal": 0.10,
        "technical_or_low_signal": 0.00,
    }.get(concept, 0.20)
    bonus = agg_priority(family, agg) / 100.0 + concept_priority / 1000.0
    if any(marker in upper for marker in ("DEF", "OVERDUE", "LATE", "DPD")):
        bonus += 0.005
    if "RATIO" in upper or "LIMIT_USE" in upper:
        bonus += 0.004
    if "MEAN" in name.upper():
        bonus += 0.001
    return bonus


def normalized_source(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def validate_input(frame: pd.DataFrame) -> list[str]:
    errors = []
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in frame]
    if missing:
        errors.append(f"missing required input columns: {missing}")
    if len(frame) != 529:
        errors.append(f"input feature count is {len(frame)}, expected exactly 529")
    if "feature_id" in frame and frame["feature_id"].nunique(dropna=False) != 529:
        errors.append("feature_id values are not unique")
    if "feature_name" in frame and frame["feature_name"].nunique(dropna=False) != 529:
        errors.append("feature_name values are not unique")
    return errors


def validate_output(frame: pd.DataFrame) -> list[str]:
    errors = []
    if len(frame) != 529:
        errors.append(f"output rows={len(frame)}")
    if frame["feature_id"].nunique(dropna=False) != 529:
        errors.append("duplicate feature_id in output")
    if frame["feature_name"].nunique(dropna=False) != 529:
        errors.append("duplicate feature_name in output")
    if frame["rank"].tolist() != list(range(1, 530)):
        errors.append("rank values are not exactly 1..529")
    if frame["rank"].duplicated().any():
        errors.append("duplicate ranks")
    expected_leakage = {
        "LEAKAGE_SAFE_LABEL_FREE",
        "POTENTIAL_LEAKAGE_NAME_REVIEW_NEEDED",
    }
    if set(frame["leakage_status"]) - expected_leakage:
        errors.append("invalid leakage_status values")
    if (frame["leakage_status"].astype(str).str.len() == 0).any():
        errors.append("blank leakage_status values")
    for rank, normalized_rank in zip(frame["rank"], frame["normalized_rank"]):
        expected = f"{(int(rank) - 1) / 528:.9f}"
        if normalized_rank != expected:
            errors.append("normalized_rank mismatch")
            break
    return errors


def build_ranking(input_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in input_frame.itertuples(index=False):
        name = str(getattr(row, "feature_name"))
        source = normalized_source(getattr(row, "source_table"))
        group = str(getattr(row, "semantic_group"))
        _, agg = strip_agg(name)
        base_score, family, concept, reason = classify(name, group)
        adjusted_score = base_score + agg_adjustment(family, agg)
        score = max(1, min(100, int(round(adjusted_score))))
        rows.append(
            {
                "feature_id": str(getattr(row, "feature_id")),
                "feature_name": name,
                "source_table": source,
                "semantic_group": group,
                "feature_family": family,
                "credit_relevance_score": score,
                "risk_concept": concept,
                "reason_short": reason,
                "_fine_score": adjusted_score + sort_bonus(name, family, concept, agg),
                "leakage_status": leakage_status(name),
            }
        )
    ranked = pd.DataFrame(rows).sort_values(
        by=["_fine_score", "credit_relevance_score", "feature_name"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    ranked = ranked.reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["normalized_rank"] = ranked["rank"].map(
        lambda rank: f"{(rank - 1) / 528:.9f}"
    )
    return ranked[OUTPUT_COLUMNS]


def main() -> int:
    input_frame = pd.read_csv(INPUT_PATH, dtype=str).fillna("")
    input_errors = validate_input(input_frame)
    if input_errors:
        raise SystemExit("validation failed: " + "; ".join(input_errors))

    ignored_columns = [
        column
        for column in input_frame.columns
        if any(marker in column.lower() for marker in FORBIDDEN_COLUMN_MARKERS)
    ]
    output_frame = build_ranking(input_frame)
    output_errors = validate_output(output_frame)
    if output_errors:
        raise SystemExit("validation failed: " + "; ".join(output_errors))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(OUTPUT_PATH, index=False)
    leakage_count = int(
        (
            output_frame["leakage_status"]
            == "POTENTIAL_LEAKAGE_NAME_REVIEW_NEEDED"
        ).sum()
    )
    summary = [
        "ranking_method: deterministic_domain_rules",
        "uses_llm: false",
        f"input_feature_count: {len(input_frame)}",
        f"ranked_feature_count: {len(output_frame)}",
        f"duplicate_feature_ids: {int(input_frame['feature_id'].duplicated().sum())}",
        f"duplicate_ranks: {int(output_frame['rank'].duplicated().sum())}",
        f"leakage_flagged_features: {leakage_count}",
        "target_performance_columns_ignored: "
        + (
            "yes (" + ", ".join(ignored_columns) + ")"
            if ignored_columns
            else "no forbidden target/performance columns present"
        ),
        f"output_csv: {OUTPUT_PATH.as_posix()}",
    ]
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
