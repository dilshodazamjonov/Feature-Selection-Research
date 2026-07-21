# LendingClub v2 Pre-Matrix Sanity Report

This is a data/metadata inspection only. It does not run the experiment matrix, train models, or fit selectors.

## Summary

- Columns checked: `678`
- Candidate features checked: `675`
- Duplicate columns: `0`
- All-null columns: `0`
- Constant candidate columns: `0`
- Columns with infinite values: `0`
- Columns with invalid numeric values: `0`
- Ratio features with extreme values: `0`
- Candidate columns with >95% missingness: `20`
- High-cardinality categorical columns: `0`
- Target/split/helper leakage flags: `0`
- Remove/fix before matrix recommendations: `0`
- Review before matrix recommendations: `20`

## Remove Or Fix Before Matrix

_No rows._

## Review Before Matrix

| feature | issue_type | severity | missing_rate | unique_count | max_abs_value | recommended_action |
| --- | --- | --- | --- | --- | --- | --- |
| annual_inc_joint | missing_rate_gt_95 | medium | 0.980857 | 998 | 1837000.0 | review_before_matrix |
| dti_joint | missing_rate_gt_95 | medium | 0.98086 | 1001 | 69.49 | review_before_matrix |
| joint_dti_gap | missing_rate_gt_95 | medium | 0.981137 | 1001 | 994.9 | review_before_matrix |
| joint_income_to_single_income | missing_rate_gt_95 | medium | 0.981125 | 1001 | 242001.0 | review_before_matrix |
| revol_bal_joint | missing_rate_gt_95 | medium | 0.986177 | 1001 | 357135.0 | review_before_matrix |
| sec_app_chargeoff_within_12_mths | missing_rate_gt_95 | medium | 0.986176 | 15 | 20.0 | review_before_matrix |
| sec_app_collections_12_mths_ex_med | missing_rate_gt_95 | medium | 0.986176 | 12 | 16.0 | review_before_matrix |
| sec_app_credit_history_months | missing_rate_gt_95 | medium | 0.986176 | 563 | 999.0 | review_before_matrix |
| sec_app_credit_history_years | missing_rate_gt_95 | medium | 0.986176 | 563 | 83.25 | review_before_matrix |
| sec_app_fico_mean | missing_rate_gt_95 | medium | 0.986176 | 60 | 842.0 | review_before_matrix |
| sec_app_fico_range_high | missing_rate_gt_95 | medium | 0.986176 | 60 | 844.0 | review_before_matrix |
| sec_app_fico_range_low | missing_rate_gt_95 | medium | 0.986176 | 60 | 840.0 | review_before_matrix |
| sec_app_inq_last_6mths | missing_rate_gt_95 | medium | 0.986176 | 7 | 6.0 | review_before_matrix |
| sec_app_mort_acc | missing_rate_gt_95 | medium | 0.986176 | 18 | 18.0 | review_before_matrix |
| sec_app_mths_since_last_major_derog | missing_rate_gt_95 | medium | 0.995068 | 113 | 132.0 | review_before_matrix |
| sec_app_num_rev_accts | missing_rate_gt_95 | medium | 0.986176 | 68 | 92.0 | review_before_matrix |
| sec_app_open_acc | missing_rate_gt_95 | medium | 0.986176 | 54 | 82.0 | review_before_matrix |
| sec_app_open_act_il | missing_rate_gt_95 | medium | 0.986176 | 35 | 38.0 | review_before_matrix |
| sec_app_revol_util | missing_rate_gt_95 | medium | 0.986419 | 972 | 212.6 | review_before_matrix |
| verification_status_joint | missing_rate_gt_95 | medium | 0.98101 | 3 |  | review_before_matrix |

## Highest Missingness Features

| feature | semantic_group | missing_rate | recommended_action |
| --- | --- | --- | --- |
| sec_app_mths_since_last_major_derog | delinquency_derogatory | 0.995068 | review_before_matrix |
| sec_app_revol_util | revolving_utilization | 0.986419 | review_before_matrix |
| revol_bal_joint | revolving_utilization | 0.986177 | review_before_matrix |
| sec_app_chargeoff_within_12_mths | delinquency_derogatory | 0.986176 | review_before_matrix |
| sec_app_fico_range_low | fico_credit_score | 0.986176 | review_before_matrix |
| sec_app_fico_range_high | fico_credit_score | 0.986176 | review_before_matrix |
| sec_app_mort_acc | mortgage_history | 0.986176 | review_before_matrix |
| sec_app_inq_last_6mths | recent_inquiries | 0.986176 | review_before_matrix |
| sec_app_credit_history_years | joint_applicant | 0.986176 | review_before_matrix |
| sec_app_fico_mean | fico_credit_score | 0.986176 | review_before_matrix |
| sec_app_collections_12_mths_ex_med | delinquency_derogatory | 0.986176 | review_before_matrix |
| sec_app_credit_history_months | joint_applicant | 0.986176 | review_before_matrix |
| sec_app_num_rev_accts | joint_applicant | 0.986176 | review_before_matrix |
| sec_app_open_acc | joint_applicant | 0.986176 | review_before_matrix |
| sec_app_open_act_il | joint_applicant | 0.986176 | review_before_matrix |
| joint_dti_gap | joint_applicant | 0.981137 | review_before_matrix |
| joint_income_to_single_income | income_capacity | 0.981125 | review_before_matrix |
| verification_status_joint | loan_terms | 0.98101 | review_before_matrix |
| dti_joint | income_capacity | 0.98086 | review_before_matrix |
| annual_inc_joint | income_capacity | 0.980857 | review_before_matrix |

## High-Cardinality Categoricals

_No rows._

## Approval Interpretation

- Matrix approval should wait until the `remove_or_fix_before_matrix` rows are removed or explicitly waived, because they are no-information constant features.
- No additional automatic removal is required if the reviewer accepts sparse joint/secondary-applicant features as intentional optional-applicant signals.
- The sparse-feature list should be reviewed before approval because many secondary-applicant fields are present only for joint applications.