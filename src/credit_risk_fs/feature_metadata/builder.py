from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from credit_risk_fs.utils.logging import setup_logging

logger = setup_logging("feature_metadata", level=logging.INFO)

ENGINEERED_PREFIX_TABLES = {
    "BURO": "bureau",
    "PREV": "previous_application",
    "POS": "POS_CASH_balance",
    "INSTAL": "installments_payments",
    "CC": "credit_card_balance",
}

AGGREGATION_TOKENS = {
    "MIN": "minimum",
    "MAX": "maximum",
    "MEAN": "mean",
    "SUM": "sum",
    "VAR": "variance",
}
NUMERIC_PERCENTILES = (
    ("p05", 0.05),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p95", 0.95),
)


def infer_semantic_group(
    feature_name: str,
    description: str = "",
    table: str = "",
) -> str:
    """Infer a coarse business-semantic group from feature naming/metadata."""
    raw_name = str(feature_name)
    lower_name = raw_name.lower()
    name = str(feature_name).upper()
    desc = str(description).upper()
    table_name = str(table).upper()
    text = " ".join([name, desc, table_name])
    lower_text = " ".join([lower_name, str(description).lower(), str(table).lower()])

    if any(token in lower_name for token in ["fico_mean", "fico_range_low", "fico_range_high", "fico_band"]):
        return "fico_credit_score"

    if lower_name in {"annual_inc", "log_annual_inc", "loan_to_income", "dti"} or any(
        token in lower_name
        for token in [
            "_to_income",
            "income_band",
            "dti_band",
            "total_bal_ex_mort_to_income",
            "tot_hi_cred_lim_to_income",
            "total_il_high_credit_limit_to_income",
            "tot_cur_bal_to_income",
        ]
    ):
        return "income_capacity"

    if any(
        token in lower_name
        for token in [
            "revol_util",
            "revol_bal",
            "total_rev_hi_lim",
            "active_revolving_share",
            "balance_to_high_credit_limit",
            "revolving_capacity_gap",
            "open_revolving_share",
            "num_op_rev_tl",
            "num_actv_rev_tl",
            "num_rev_tl_bal_gt_0",
            "revol_bal_per_open_acc",
            "utilization_pressure",
        ]
    ):
        return "revolving_utilization"

    if any(
        token in lower_name
        for token in [
            "bc_open_to_buy",
            "total_bc_limit",
            "bankcard_capacity_gap",
            "bc_util",
            "percent_bc_gt_75",
            "mths_since_recent_bc",
        ]
    ):
        return "bankcard_capacity"

    if any(token in lower_name for token in ["inq_last_6mths", "inq_6m_per_open_acc", "mths_since_recent_inq"]):
        return "recent_inquiries"

    if lower_name == "earliest_cr_line" or lower_name == "credit_history_years" or lower_name.startswith("mths_since_") or any(
        token in lower_name for token in ["mo_sin_old", "mo_sin_rcnt", "credit_history"]
    ):
        return "credit_history_length"

    if any(
        token in lower_name
        for token in [
            "acc_open_past_24mths",
            "num_tl_op_past_12m",
            "num_sats_share",
            "loan_per_open_acc",
            "loan_per_total_acc",
        ]
    ):
        return "account_opening_activity"

    if "mort_acc" in lower_name or "mort_balance_pressure" in lower_name:
        return "mortgage_history"

    if any(
        token in lower_name
        for token in [
            "delinq_2yrs",
            "pub_rec",
            "collections_12_mths_ex_med",
            "chargeoff_within_12_mths",
            "tax_liens",
        ]
    ):
        return "delinquency_derogatory"

    if lower_name in {"term", "purpose", "home_ownership", "verification_status", "addr_state", "application_type"} or any(
        token in lower_name for token in ["term_", "purpose_", "verification_status", "home_ownership"]
    ):
        return "loan_terms"

    if lower_name in {"loan_amnt", "funded_amnt", "funded_amnt_inv", "log_loan_amnt"} or "loan_amnt_band" in lower_name:
        return "exposure_amount"

    if "lendingclub" in lower_text and "loan" in lower_text and "amount" in lower_text:
        return "exposure_amount"

    if "EXT_SOURCE" in name or "EXTERNAL SCORE" in text:
        return "external_score"

    if any(token in text for token in ["MISSING", "UNKNOWN", "UNAVAILABLE", "NOT KNOWN"]):
        return "missingness_or_unknown"

    if name.startswith("BURO_") or "BUREAU" in table_name:
        if any(
            token in text
            for token in [
                "AMT_CREDIT_SUM",
                "AMT_CREDIT_SUM_DEBT",
                "AMT_CREDIT_MAX_OVERDUE",
                "AMT_ANNUITY",
                "CREDIT_LIMIT",
                "OVERDUE",
                "DEBT",
                "UTILIZATION",
                "EXPOSURE",
                "LEVERAGE",
            ]
        ):
            return "bureau_debt"
        return "bureau_credit_history"

    if name.startswith("INSTAL_") or "INSTALLMENTS_PAYMENTS" in table_name:
        return "installment_repayment_behavior"

    if name.startswith("CC_") or "CREDIT_CARD_BALANCE" in table_name:
        return "credit_card_utilization"

    if name.startswith("PREV_") or "PREVIOUS_APPLICATION" in table_name:
        return "previous_application_behavior"

    if any(
        token in text
        for token in [
            "AMT_INCOME_TOTAL",
            "INCOME",
            "SALARY",
            "ANNUITY",
            "PAYMENT_RATE",
            "EMPLOY",
            "OCCUPATION",
            "CAPACITY",
            "FAMILY_MEMBERS",
        ]
    ):
        return "income_capacity"

    if any(
        token in text
        for token in [
            "AMT_CREDIT",
            "AMT_GOODS_PRICE",
            "AMT_ANNUITY",
            "AMT_APPLICATION",
            "DOWN_PAYMENT",
            "CREDIT_TO_ANNUITY",
            "APPLICATION",
        ]
    ):
        return "application_amounts"

    if any(
        token in text
        for token in [
            "OVERDUE",
            "DPD",
            "DBD",
            "DELINQ",
            "PAST DUE",
            "LATE",
            "DEFAULT",
            "BAD DEBT",
        ]
    ):
        return "delinquency_behavior"

    if any(
        token in text
        for token in [
            "DAYS_BIRTH",
            "AGE",
            "DAYS_REGISTRATION",
            "DAYS_ID_PUBLISH",
            "DAYS_LAST_PHONE_CHANGE",
            "REGION",
            "CITY",
            "ORGANIZATION_TYPE",
            "OWN_CAR_AGE",
            "HOUR_APPR_PROCESS_START",
            "WEEKDAY_APPR_PROCESS_START",
            "TIME",
        ]
    ):
        return "demographic_time_variables"

    return "other"


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Find a column in a DataFrame using flexible candidate names."""
    lowered = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _load_description_frame(description_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(description_path, encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning(
            "UTF-8 decoding failed for %s, falling back to latin1.",
            description_path,
        )
        return pd.read_csv(description_path, encoding="latin1")


def _normalize_token(token: str) -> str:
    return token.replace("_", " ").lower()


def _infer_engineered_metadata(feature_name: str) -> tuple[str, str]:
    parts = feature_name.split("_")
    if not parts:
        return "", feature_name.replace("_", " ").lower()

    prefix = parts[0]
    table = ENGINEERED_PREFIX_TABLES.get(prefix, "")

    if prefix in ENGINEERED_PREFIX_TABLES and len(parts) >= 3 and parts[-1] in AGGREGATION_TOKENS:
        aggregation = AGGREGATION_TOKENS[parts[-1]]
        base_feature = _normalize_token("_".join(parts[1:-1]))
        description = f"{aggregation} of {base_feature} from {table}"
        return table, description

    return table, feature_name.replace("_", " ").lower()


def _safe_round(value: float | int | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)


def _clean_numeric_summary_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace(
        [float("inf"), float("-inf")],
        float("nan"),
    )


def build_feature_metadata(
    X: pd.DataFrame,
    description_csv_path: str | Path,
) -> list[dict]:
    """
    Build per-feature metadata for LLM-assisted feature selection.

    Metadata is generated from the current training slice so the LLM sees only
    fold-local information. When an engineered feature is absent from the source
    description CSV, a structured fallback description is inferred from the
    feature prefix and aggregation suffix.
    """
    description_path = Path(description_csv_path)
    if not description_path.exists():
        raise FileNotFoundError(f"Description file not found: {description_path}")

    desc_df = _load_description_frame(description_path)

    feature_col = _find_column(desc_df, ["row", "feature", "column", "columns", "name"])
    desc_col = _find_column(desc_df, ["description", "desc"])
    table_col = _find_column(desc_df, ["table", "source"])

    desc_map: dict[str, str] = {}
    table_map: dict[str, str] = {}

    if feature_col and desc_col:
        for _, row in desc_df.iterrows():
            feature_name = str(row[feature_col]).strip()
            if not feature_name or feature_name.lower() == "nan":
                continue

            desc_map[feature_name] = "" if pd.isna(row[desc_col]) else str(row[desc_col]).strip()
            if table_col:
                table_map[feature_name] = "" if pd.isna(row[table_col]) else str(row[table_col]).strip()

    metadata: list[dict] = []
    for feature in X.columns:
        series = X[feature]
        inferred_table, inferred_desc = _infer_engineered_metadata(feature)
        desc = desc_map.get(feature, "") or inferred_desc
        table = table_map.get(feature, "") or inferred_table

        entry = {
            "name": feature,
            "description": desc,
            "table": table,
            "semantic_group": infer_semantic_group(feature, description=desc, table=table),
            "missing_rate": round(float(series.isna().mean()), 4),
            "non_null_count": int(series.notna().sum()),
            "dtype": str(series.dtype),
        }

        if pd.api.types.is_numeric_dtype(series):
            numeric_series = _clean_numeric_summary_series(series)
            entry.update(
                {
                    "mean": _safe_round(numeric_series.mean(skipna=True)),
                    "min": _safe_round(numeric_series.min(skipna=True)),
                    "max": _safe_round(numeric_series.max(skipna=True)),
                    "std": _safe_round(numeric_series.std(skipna=True)),
                    "var": _safe_round(numeric_series.var(skipna=True)),
                }
            )
            for percentile_name, percentile_value in NUMERIC_PERCENTILES:
                entry[percentile_name] = _safe_round(
                    numeric_series.quantile(percentile_value)
                )
        else:
            entry["unique_count"] = int(series.nunique(dropna=True))

        metadata.append(entry)

    return metadata
