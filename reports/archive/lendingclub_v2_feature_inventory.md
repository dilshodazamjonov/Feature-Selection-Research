# LendingClub v2 Feature Inventory

- Processed file: `data/lendingclub_v2/processed/application_train.csv`
- Rows: `1,348,099`
- Total columns including helpers: `678`
- Candidate feature columns: `675`
- Features with descriptions: `675`
- Description coverage: `100.00%`
- Semantic groups: `19`

## Dominant Semantic Groups

| semantic_group | feature_count | share_of_features | examples |
| --- | --- | --- | --- |
| account_mix_credit_depth | 114 | 0.168889 | acc_open_past_24mths_is_zero; acc_open_past_24mths_per_credit_history_year; acc_open_past_24mths_per_open_acc; acc_open_past_24mths_per_total_acc; acc_open_past_24mths_positive_flag |
| revolving_utilization | 100 | 0.148148 | active_revolving_share; all_util; all_util_band; all_util_ge_100_flag; all_util_ge_30_flag |
| application_profile | 74 | 0.10963 | annual_inc_band; annual_inc_is_zero; collections_12_mths_ex_med_is_zero; collections_12_mths_ex_med_per_credit_history_year; collections_12_mths_ex_med_per_open_acc |
| recent_inquiries | 60 | 0.088889 | acc_now_delinq_is_zero; acc_now_delinq_per_credit_history_year; acc_now_delinq_per_open_acc; acc_now_delinq_per_total_acc; acc_now_delinq_positive_flag |
| delinquency_derogatory | 54 | 0.08 | chargeoff_within_12_mths; chargeoff_within_12_mths_is_zero; chargeoff_within_12_mths_missing_flag; chargeoff_within_12_mths_per_credit_history_year; chargeoff_within_12_mths_per_open_acc |
| missingness_or_unknown | 45 | 0.066667 | acc_now_delinq_missing_flag; all_util_missing_flag; annual_inc_joint_missing_flag; annual_inc_missing_flag; avg_cur_bal_missing_flag |
| income_capacity | 37 | 0.054815 | annual_inc; annual_inc_joint; avg_cur_bal_to_income; bc_limit_to_income_band; bc_open_to_buy_to_income |
| balance_credit_limit_pressure | 35 | 0.051852 | avg_cur_bal_is_zero; avg_cur_bal_to_loan_amnt; bc_open_to_buy_to_loan_amnt; delinq_amnt_to_loan_amnt; il_balance_to_total_balance |
| application_amounts | 29 | 0.042963 | acc_now_delinq; all_util_minus_il_util; avg_cur_bal_gap; bankruptcy_history_flag; bankruptcy_or_pubrec_count |
| credit_history_length | 27 | 0.04 | credit_history_band; credit_history_months; credit_history_months_missing_flag; credit_history_short_flag; credit_history_years |

## Highest Missingness Features

| feature | missing_rate | non_null_count |
| --- | --- | --- |
| sec_app_mths_since_last_major_derog | 0.9950678696445884 | 6649 |
| sec_app_revol_util | 0.9864193950147578 | 18308 |
| revol_bal_joint | 0.9861768312267868 | 18635 |
| sec_app_open_acc | 0.9861760894415024 | 18636 |
| sec_app_fico_range_low | 0.9861760894415024 | 18636 |
| sec_app_fico_range_high | 0.9861760894415024 | 18636 |
| sec_app_open_act_il | 0.9861760894415024 | 18636 |
| sec_app_num_rev_accts | 0.9861760894415024 | 18636 |
| sec_app_credit_history_years | 0.9861760894415024 | 18636 |
| sec_app_credit_history_months | 0.9861760894415024 | 18636 |
| sec_app_inq_last_6mths | 0.9861760894415024 | 18636 |
| sec_app_fico_mean | 0.9861760894415024 | 18636 |
| sec_app_collections_12_mths_ex_med | 0.9861760894415024 | 18636 |
| sec_app_mort_acc | 0.9861760894415024 | 18636 |
| sec_app_chargeoff_within_12_mths | 0.9861760894415024 | 18636 |