# LendingClub Leakage And Label Definition

## Target Definition

`TARGET = 1` for final bad/default outcomes and `TARGET = 0` for final good outcomes. The implementation is in `src/credit_risk_fs/preprocessing/labeling.py`.

Good statuses:
- `Does not meet the credit policy. Status:Fully Paid`
- `Fully Paid`

Bad/default statuses:
- `Charged Off`
- `Default`
- `Does not meet the credit policy. Status:Charged Off`

Dropped ambiguous/current/unmatured statuses:
- `Current`
- `In Grace Period`
- `Issued`
- `Late (16-30 days)`
- `Late (31-120 days)`

## Leakage Columns Removed

payment fields:
- `total_pymnt` (not present in processed safe file)
- `total_pymnt_inv` (not present in processed safe file)
- `total_rec_prncp` (not present in processed safe file)
- `total_rec_int` (not present in processed safe file)
- `total_rec_late_fee` (not present in processed safe file)
- `last_pymnt_amnt` (not present in processed safe file)

recovery fields:
- `recoveries` (not present in processed safe file)
- `collection_recovery_fee` (not present in processed safe file)

settlement fields:
- `debt_settlement_flag` (not present in processed safe file)
- `debt_settlement_flag_date` (not present in processed safe file)
- `settlement_status` (not present in processed safe file)
- `settlement_date` (not present in processed safe file)
- `settlement_amount` (not present in processed safe file)
- `settlement_percentage` (not present in processed safe file)
- `settlement_term` (not present in processed safe file)

hardship fields:
- `hardship_flag` (not present in processed safe file)
- `hardship_type` (not present in processed safe file)
- `hardship_reason` (not present in processed safe file)
- `hardship_status` (not present in processed safe file)
- `hardship_amount` (not present in processed safe file)
- `hardship_start_date` (not present in processed safe file)
- `hardship_end_date` (not present in processed safe file)
- `hardship_length` (not present in processed safe file)
- `hardship_dpd` (not present in processed safe file)
- `hardship_loan_status` (not present in processed safe file)
- `hardship_payoff_balance_amount` (not present in processed safe file)
- `hardship_last_payment_amount` (not present in processed safe file)

post-origination status fields:
- `loan_status` (not present in processed safe file)
- `grade` (not present in processed safe file)
- `sub_grade` (not present in processed safe file)
- `int_rate` (not present in processed safe file)
- `installment` (not present in processed safe file)
- `funded_amnt` (not present in processed safe file)
- `funded_amnt_inv` (not present in processed safe file)
- `pymnt_plan` (not present in processed safe file)
- `disbursement_method` (not present in processed safe file)

collection fields:
- `collection_recovery_fee` (not present in processed safe file)

future payment/date fields:
- `last_pymnt_d` (not present in processed safe file)
- `next_pymnt_d` (not present in processed safe file)
- `last_credit_pull_d` (not present in processed safe file)
- `last_fico_range_low` (not present in processed safe file)
- `last_fico_range_high` (not present in processed safe file)
- `payment_plan_start_date` (not present in processed safe file)

Collection-count fields retained as application-time credit-history variables, not post-outcome recovery leakage:
- `collections_12_mths_ex_med`
- `sec_app_collections_12_mths_ex_med`

## Processed Safe Path Evidence

- Processed file: `data\lendingclub\processed\application_train.csv`.
- Processed CSV column count: `99`.
- Final processed feature count after excluding `TARGET`, `recent_decision`, `issue_d`, and `loan_status`: `96`.
- `loan_status` present in processed CSV: `False`.
- `TARGET` present in processed CSV: `True`.
- `recent_decision` present in processed CSV for temporal splitting: `True`.
- `issue_d` present in processed CSV but configured as excluded before modeling: `True`.
- Current reported LendingClub matrix rows use the processed safe path: `True`.

Run-level `leakage_report.json` artifacts confirm `target_column_excluded=true`, empty forbidden-column lists in train/OOT features, and `oot_used_in_feature_selection=false`.

## Missingness-By-Target Check

- Rows checked: `1348099`.
- Good rows: `1078739`.
- Bad rows: `269360`.
- Possible leakage flags from missingness asymmetry: `0`.
- Output table: `results/lendingclub/analysis/leakage_transparency/missingness_by_target_leakage_check.csv`.

No missingness-by-target possible leakage flags were found under the configured threshold.

Plots:
- Skipped `missingness_by_target_possible_leakage_flags.png`: no possible leakage flags.

## Remaining Items

- The processed safe file keeps `TARGET` for supervised training and `recent_decision`/`issue_d` for time handling, but experiment configs exclude these from model features.
- Raw LendingClub files remain leakage-prone and should not be used directly without the preparation/audit path.
