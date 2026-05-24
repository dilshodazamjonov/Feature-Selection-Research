import numpy as np
import pandas as pd

from credit_risk_fs.data.loaders import normalize_home_credit_sentinel_dates
from credit_risk_fs.feature_engineering.homecredit.assemble import (
    TIME_PROXY_COL,
    build_application_time_proxy,
)
from credit_risk_fs.feature_engineering.lendingclub.application import (
    build_application_features as build_lendingclub_application_features,
)
from credit_risk_fs.feature_metadata.builder import build_feature_metadata
from credit_risk_fs.models._cv_utils import GroupedTimeSeriesSplit
from credit_risk_fs.preprocessing.encoding import Preprocessor
from credit_risk_fs.preprocessing.lendingclub import prepare_lendingclub_application_frame
from credit_risk_fs.selectors.pca import PCASelector


def test_normalize_home_credit_sentinel_dates_replaces_only_day_columns():
    df = pd.DataFrame(
        {
            "DAYS_EMPLOYED": [-100.0, 365243.0],
            "OTHER_VALUE": [365243.0, 1.0],
        }
    )

    cleaned = normalize_home_credit_sentinel_dates(df, "application_train")

    assert cleaned.loc[0, "DAYS_EMPLOYED"] == -100.0
    assert np.isnan(cleaned.loc[1, "DAYS_EMPLOYED"])
    assert cleaned.loc[0, "OTHER_VALUE"] == 365243.0


def test_build_application_time_proxy_uses_more_than_previous_application():
    proxy = build_application_time_proxy(
        {
            "previous_application": pd.DataFrame(
                {"SK_ID_CURR": [1], "DAYS_DECISION": [-15.0]}
            ),
            "bureau": pd.DataFrame(
                {"SK_ID_CURR": [2], "DAYS_CREDIT": [-30.0]}
            ),
        }
    )

    assert set(proxy["SK_ID_CURR"]) == {1, 2}
    proxy_map = dict(zip(proxy["SK_ID_CURR"], proxy[TIME_PROXY_COL]))
    assert proxy_map[1] == -15.0
    assert proxy_map[2] == -30.0


def test_preprocessor_one_hot_encodes_high_cardinality_categories():
    X = pd.DataFrame(
        {
            "ORGANIZATION_TYPE": ["A", "B", "C", "A"],
            "AMT_CREDIT": [1.0, 2.0, 3.0, 4.0],
        }
    )

    transformed = Preprocessor(cat_min_frequency=1).fit_transform(X)

    assert "ORGANIZATION_TYPE" not in transformed.columns
    assert any(col.startswith("ORGANIZATION_TYPE_") for col in transformed.columns)


def test_preprocessor_replaces_infinite_values_before_scaling():
    X = pd.DataFrame(
        {
            "AMT_CREDIT": [1.0, np.inf, 3.0],
            "NAME_CONTRACT_TYPE": ["Cash", "Cash", "Revolving"],
        }
    )

    transformed = Preprocessor(cat_min_frequency=1).fit_transform(X)

    assert np.isfinite(transformed["AMT_CREDIT"]).all()


def test_preprocessor_handles_missing_pandas_categorical_values():
    X = pd.DataFrame(
        {
            "purpose_band": pd.Series(
                pd.Categorical(["low", None, "high", "low"]),
            ),
            "term_band": pd.Series(
                pd.Categorical(["36m", "60m", None, "36m"]),
            ),
            "AMT_CREDIT": [1.0, 2.0, 3.0, 4.0],
        }
    )

    transformed = Preprocessor(cat_min_frequency=1).fit_transform(X)

    assert "purpose_band" not in transformed.columns
    assert "term_band" not in transformed.columns
    assert any(col.startswith("purpose_band_") for col in transformed.columns)
    assert any(col.startswith("term_band_") for col in transformed.columns)
    assert np.isfinite(transformed.to_numpy()).all()


def test_pca_selector_accepts_optional_target_argument():
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [3.0, 2.0, 1.0]})
    y = pd.Series([0, 1, 0])

    transformed = PCASelector(n_components=1).fit_transform(X, y)

    assert list(transformed.columns) == ["PC1"]
    assert transformed.shape == (3, 1)


def test_grouped_time_series_split_keeps_same_time_values_together():
    time_values = np.array([-4, -4, -3, -3, -2, -2, -1, -1], dtype=float)
    splitter = GroupedTimeSeriesSplit(n_splits=2, gap=0)

    for train_idx, val_idx in splitter.split(time_values):
        train_times = set(time_values[train_idx])
        val_times = set(time_values[val_idx])
        assert train_times.isdisjoint(val_times)


def test_build_feature_metadata_includes_numeric_percentiles(tmp_path):
    description_path = tmp_path / "descriptions.csv"
    description_path.write_text(
        "row,description,table\nAMT_CREDIT,Credit amount,application_train\n",
        encoding="utf-8",
    )
    X = pd.DataFrame({"AMT_CREDIT": [0.0, 10.0, 20.0, 30.0, 40.0]})

    metadata = build_feature_metadata(X, description_path)

    assert len(metadata) == 1
    entry = metadata[0]
    assert entry["name"] == "AMT_CREDIT"
    assert np.isclose(entry["p05"], 2.0)
    assert np.isclose(entry["p25"], 10.0)
    assert np.isclose(entry["p50"], 20.0)
    assert np.isclose(entry["p75"], 30.0)
    assert np.isclose(entry["p95"], 38.0)


def test_prepare_lendingclub_application_frame_drops_leakage_columns():
    raw = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "member_id": [11, 12, 13],
            "loan_amnt": [10000.0, 15000.0, 9000.0],
            "funded_amnt": [10000.0, 15000.0, 9000.0],
            "funded_amnt_inv": [10000.0, 15000.0, 9000.0],
            "annual_inc": [60000.0, 80000.0, 50000.0],
            "issue_d": ["Jan-2016", "Feb-2016", "Mar-2016"],
            "loan_status": ["Fully Paid", "Charged Off", "Late (31-120 days)"],
            "grade": ["A", "B", "C"],
            "sub_grade": ["A1", "B2", "C3"],
            "int_rate": [7.9, 12.5, 18.0],
            "installment": [312.0, 488.0, 355.0],
            "title": ["Debt consolidation", "Credit card refinance", "Other"],
            "desc": ["note one", "note two", "note three"],
            "emp_title": ["Analyst", "Teacher", "Driver"],
            "url": ["u1", "u2", "u3"],
            "zip_code": ["123xx", "456xx", "789xx"],
            "policy_code": [1.0, 1.0, 1.0],
            "pymnt_plan": ["n", "n", "n"],
            "debt_settlement_flag": ["N", "Y", "N"],
        }
    )

    prepared = prepare_lendingclub_application_frame(raw)

    assert "TARGET" in prepared.columns
    assert "recent_decision" in prepared.columns
    assert len(prepared) == 2
    assert "loan_status" not in prepared.columns
    assert "title" not in prepared.columns
    assert "policy_code" not in prepared.columns
    assert "pymnt_plan" not in prepared.columns
    assert "debt_settlement_flag" not in prepared.columns
    assert "grade" not in prepared.columns
    assert "sub_grade" not in prepared.columns
    assert "int_rate" not in prepared.columns
    assert "installment" not in prepared.columns
    assert "funded_amnt" not in prepared.columns
    assert "funded_amnt_inv" not in prepared.columns
    assert "id" not in prepared.columns
    assert "member_id" not in prepared.columns
    assert "url" not in prepared.columns
    assert "desc" not in prepared.columns
    assert "emp_title" not in prepared.columns
    assert "zip_code" not in prepared.columns


def test_build_lendingclub_application_features_adds_engineered_columns():
    prepared = pd.DataFrame(
        {
            "loan_amnt": [10000.0, 20000.0, 15000.0],
            "term": ["36 months", "60 months", "36 months"],
            "emp_length": ["2 years", "10+ years", "< 1 year"],
            "home_ownership": ["RENT", "MORTGAGE", "OWN"],
            "annual_inc": [60000.0, 120000.0, 85000.0],
            "verification_status": ["Verified", "Not Verified", "Source Verified"],
            "issue_d": pd.to_datetime(["2015-01-01", "2015-02-01", "2015-03-01"]),
            "purpose": ["debt_consolidation", "credit_card", "home_improvement"],
            "addr_state": ["CA", "TX", "NY"],
            "dti": [12.0, 18.0, 25.0],
            "delinq_2yrs": [0.0, 1.0, 0.0],
            "earliest_cr_line": ["Jan-2005", "Jan-2000", "Jan-2010"],
            "fico_range_low": [680.0, 640.0, 720.0],
            "fico_range_high": [684.0, 644.0, 724.0],
            "inq_last_6mths": [1.0, 2.0, 0.0],
            "mths_since_last_delinq": [np.nan, 12.0, np.nan],
            "mths_since_last_record": [np.nan, np.nan, 18.0],
            "open_acc": [8.0, 12.0, 15.0],
            "pub_rec": [0.0, 1.0, 0.0],
            "revol_bal": [5000.0, 12000.0, 9000.0],
            "revol_util": [32.0, 78.0, 45.0],
            "total_acc": [20.0, 35.0, 28.0],
            "initial_list_status": ["w", "f", "w"],
            "application_type": ["INDIVIDUAL", "JOINT APP", "INDIVIDUAL"],
            "annual_inc_joint": [np.nan, 180000.0, np.nan],
            "dti_joint": [np.nan, 15.0, np.nan],
            "verification_status_joint": [np.nan, "Verified", np.nan],
            "tot_cur_bal": [25000.0, 60000.0, 42000.0],
            "open_act_il": [2.0, 4.0, 3.0],
            "open_il_12m": [1.0, 2.0, 1.0],
            "open_il_24m": [2.0, 3.0, 2.0],
            "total_bal_il": [8000.0, 20000.0, 12000.0],
            "il_util": [40.0, 70.0, 50.0],
            "open_rv_12m": [1.0, 3.0, 2.0],
            "open_rv_24m": [2.0, 5.0, 3.0],
            "max_bal_bc": [3000.0, 9000.0, 4000.0],
            "all_util": [35.0, 74.0, 48.0],
            "total_rev_hi_lim": [15000.0, 25000.0, 20000.0],
            "inq_fi": [0.0, 2.0, 1.0],
            "inq_last_12m": [2.0, 4.0, 1.0],
            "acc_open_past_24mths": [3.0, 5.0, 2.0],
            "avg_cur_bal": [3000.0, 5000.0, 4000.0],
            "bc_open_to_buy": [7000.0, 4000.0, 9000.0],
            "bc_util": [30.0, 82.0, 40.0],
            "mo_sin_old_il_acct": [120.0, 160.0, 80.0],
            "mo_sin_old_rev_tl_op": [140.0, 220.0, 90.0],
            "mo_sin_rcnt_rev_tl_op": [6.0, 4.0, 8.0],
            "mo_sin_rcnt_tl": [5.0, 3.0, 7.0],
            "mort_acc": [0.0, 2.0, 1.0],
            "mths_since_recent_bc": [7.0, 5.0, 8.0],
            "mths_since_recent_inq": [2.0, 1.0, 4.0],
            "num_accts_ever_120_pd": [0.0, 1.0, 0.0],
            "num_actv_bc_tl": [4.0, 6.0, 5.0],
            "num_actv_rev_tl": [5.0, 8.0, 6.0],
            "num_bc_sats": [4.0, 6.0, 5.0],
            "num_bc_tl": [8.0, 11.0, 10.0],
            "num_il_tl": [6.0, 10.0, 8.0],
            "num_op_rev_tl": [7.0, 9.0, 8.0],
            "num_rev_accts": [11.0, 15.0, 13.0],
            "num_rev_tl_bal_gt_0": [5.0, 7.0, 6.0],
            "num_sats": [8.0, 12.0, 10.0],
            "num_tl_90g_dpd_24m": [0.0, 1.0, 0.0],
            "num_tl_op_past_12m": [2.0, 4.0, 1.0],
            "pct_tl_nvr_dlq": [98.0, 85.0, 96.0],
            "percent_bc_gt_75": [10.0, 55.0, 20.0],
            "pub_rec_bankruptcies": [0.0, 1.0, 0.0],
            "tax_liens": [0.0, 0.0, 0.0],
            "tot_hi_cred_lim": [40000.0, 80000.0, 55000.0],
            "total_bal_ex_mort": [15000.0, 35000.0, 22000.0],
            "total_bc_limit": [10000.0, 18000.0, 15000.0],
            "total_il_high_credit_limit": [12000.0, 25000.0, 16000.0],
            "sec_app_fico_range_low": [np.nan, 660.0, np.nan],
            "sec_app_fico_range_high": [np.nan, 664.0, np.nan],
            "sec_app_earliest_cr_line": [np.nan, "Jan-2008", np.nan],
            "sec_app_inq_last_6mths": [np.nan, 1.0, np.nan],
            "sec_app_mort_acc": [np.nan, 1.0, np.nan],
            "sec_app_open_acc": [np.nan, 7.0, np.nan],
            "sec_app_revol_util": [np.nan, 55.0, np.nan],
            "sec_app_open_act_il": [np.nan, 2.0, np.nan],
            "sec_app_num_rev_accts": [np.nan, 9.0, np.nan],
            "recent_decision": [-1430.0, -1399.0, -1371.0],
            "TARGET": [0, 1, 0],
        }
    )

    engineered = build_lendingclub_application_features(prepared)

    assert "term_months" in engineered.columns
    assert "fico_mean" in engineered.columns
    assert "credit_history_months" in engineered.columns
    assert "loan_to_income" in engineered.columns
    assert "loan_amnt_band" in engineered.columns
    assert "joint_application_flag" in engineered.columns
    assert "mths_since_last_delinq_missing_flag" in engineered.columns
    assert "earliest_cr_line" not in engineered.columns
    assert "sec_app_earliest_cr_line" not in engineered.columns
    assert engineered.attrs["engineered_feature_count"] > prepared.shape[1]
