# LendingClub v2 Metadata Quality Audit

## Summary

- Candidate metadata rows: `675`
- Metadata rows changed on this pass: `387`
- Metadata rows with remaining quality flags: `0`
- Generic descriptions remaining: `0`
- Blank descriptions: `0`
- Blank semantic groups: `0`
- Blank source formulas: `0`
- Features marked `needs_manual_review`: `0`
- Vague catch-all group rows remaining: `0`

## Dominant Semantic Groups After Cleanup

| semantic_group | feature_count | share_of_features | examples |
| --- | --- | --- | --- |
| delinquency_derogatory | 111 | 0.164444 | acc_now_delinq; acc_now_delinq_is_zero; acc_now_delinq_per_credit_history_year; acc_now_delinq_per_open_acc; acc_now_delinq_per_total_acc |
| missingness_or_unknown | 88 | 0.13037 | acc_now_delinq_missing_flag; acc_open_past_24mths_missing_flag; all_util_missing_flag; annual_inc_joint_missing_flag; annual_inc_missing_flag |
| account_mix_credit_depth | 76 | 0.112593 | credit_lines_per_history_band; credit_lines_per_history_year; loan_per_open_acc; loan_per_total_acc; log_num_actv_rev_tl |
| balance_credit_limit_pressure | 69 | 0.102222 | all_util; all_util_band; all_util_ge_100_flag; all_util_ge_30_flag; all_util_ge_50_flag |
| bankcard_capacity | 65 | 0.096296 | all_util_minus_bc_util; bankcard_capacity_gap; bc_open_to_buy; bc_open_to_buy_is_zero; bc_open_to_buy_share |
| credit_history_length | 46 | 0.068148 | credit_history_band; credit_history_months; credit_history_short_flag; credit_history_years; mo_sin_old_il_acct |
| revolving_utilization | 45 | 0.066667 | active_revolving_share; high_revol_util_flag; log_open_rv_12m; log_open_rv_24m; log_revol_bal |
| recent_inquiries | 35 | 0.051852 | has_recent_inquiry_flag; inq_12m_per_open_acc; inq_6m_per_open_acc; inq_fi; inq_fi_is_zero |
| income_capacity | 30 | 0.044444 | annual_inc; annual_inc_band; annual_inc_is_zero; annual_inc_joint; bc_limit_to_income_band |
| fico_credit_score | 22 | 0.032593 | credit_history_years_x_fico; delinquency_pressure_x_fico; fico_adjusted_all_util; fico_adjusted_bc_util; fico_adjusted_il_util |
| account_opening_activity | 21 | 0.031111 | acc_open_past_24mths; acc_open_past_24mths_band; acc_open_past_24mths_is_zero; acc_open_past_24mths_per_credit_history_year; acc_open_past_24mths_per_open_acc |
| mortgage_history | 17 | 0.025185 | has_mortgage_flag; log_mort_acc; mort_acc; mort_acc_band; mort_acc_is_zero |

## Main Corrections

- Replaced generic engineered-feature descriptions with specific credit-risk interpretations.
- Reassigned delinquency, FICO, DTI/income, mortgage, inquiry, bankcard, revolving, balance, and account-depth features to more accurate semantic groups.
- Replaced weak `derived_from_safe_fields(...)` formulas where feature names implied an explicit formula.
- Left leakage status as `safe` unless an existing row required manual review; no new manual-review rows were introduced.