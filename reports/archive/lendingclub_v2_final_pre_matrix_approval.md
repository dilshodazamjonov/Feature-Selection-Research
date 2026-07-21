# LendingClub v2 Final Pre-Matrix Approval

This report is based only on data and metadata regeneration. The experiment matrix was not run and no models were trained.

## Decision

- Matrix approved to run: `yes`.
- Final candidate feature count: `675`.
- Description coverage: `100.00%`.
- Semantic group count: `17`.
- High-severity sanity warnings: `0`.
- Medium sanity warnings: `20`.

## Approval Checks

- Missing descriptions: `0`.
- Missing semantic groups: `0`.
- Included leakage columns: `0`.
- Semantic distribution matches `columns_description.csv`: `yes`.
- Candidate features above 500: `yes`.

## Features Removed Or Fixed

- Removed features: `43`.
- Removed feature list: `annual_inc_joint_is_zero; dti_adjusted_sec_app_revol_util; fico_adjusted_sec_app_revol_util; loan_amnt_is_zero; loan_amnt_to_loan_amnt; log_annual_inc_joint; log_revol_bal_joint; log_sec_app_chargeoff_within_12_mths; log_sec_app_collections_12_mths_ex_med; log_sec_app_inq_last_6mths; log_sec_app_mort_acc; log_sec_app_num_rev_accts; log_sec_app_open_acc; log_sec_app_open_act_il; mths_since_recent_inq_seasoned_60m_flag; revol_bal_joint_to_income; revol_bal_joint_to_loan_amnt; sec_app_chargeoff_within_12_mths_per_credit_history_year; sec_app_chargeoff_within_12_mths_per_open_acc; sec_app_chargeoff_within_12_mths_per_total_acc; sec_app_collections_12_mths_ex_med_per_credit_history_year; sec_app_collections_12_mths_ex_med_per_open_acc; sec_app_collections_12_mths_ex_med_per_total_acc; sec_app_fico_span; sec_app_inq_last_6mths_per_credit_history_year; sec_app_inq_last_6mths_per_open_acc; sec_app_inq_last_6mths_per_total_acc; sec_app_mort_acc_per_credit_history_year; sec_app_mort_acc_per_open_acc; sec_app_mort_acc_per_total_acc; sec_app_mths_since_last_major_derog_inverse_recency; sec_app_num_rev_accts_per_credit_history_year; sec_app_num_rev_accts_per_open_acc; sec_app_num_rev_accts_per_total_acc; sec_app_open_acc_per_credit_history_year; sec_app_open_acc_per_open_acc; sec_app_open_acc_per_total_acc; sec_app_open_act_il_per_credit_history_year; sec_app_open_act_il_per_open_acc; sec_app_open_act_il_per_total_acc; sqrt_annual_inc_joint; sqrt_revol_bal_joint; total_acc_is_zero`.
- Ratio features fixed by denominator handling and clipping: `11`.
- Fixed ratio list: `fico_to_income; loan_to_income_x_fico; term_x_loan_to_income; tot_cur_bal_to_income; tot_hi_cred_lim_to_income; total_bal_ex_mort_to_income; total_bal_il_to_income; total_bc_limit_to_income; total_il_high_credit_limit_to_income; total_rev_hi_lim_per_rev_trade; total_rev_hi_lim_to_income`.

## Sparse Feature Policy

- Current features with >95% missingness: `20`.
- Policy: Removed constant sanity-check features and redundant ultra-sparse joint/secondary-applicant derivatives; retained sparse raw joint/secondary fields and summary indicators where the missingness itself reflects non-joint applications.

## Extreme-Ratio Policy

- Policy: Ratios use zero-denominator-to-missing handling and fixed, outcome-independent clipping caps for the pre-matrix review feature set.

## Leakage Review

- Result: `pass`; no included candidate feature is marked as leakage or manual review.
- Excluded leakage/source-policy rows documented in `leakage_review.csv`: `39`.