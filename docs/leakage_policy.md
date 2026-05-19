# Leakage Policy

Leakage control is mandatory in this repository.

## Global Principles

- target and time columns must never reach model features
- OOT rows must never be used during selector fitting
- fold-local selectors must only see the fold training slice
- final model fitting must only use DEV, never OOT

## Home Credit

Protected columns include:
- `TARGET`
- `recent_decision`
- `PREV_recent_decision_MAX`
- `DAYS_DECISION`
- `application_time_proxy`

Home Credit also uses an application-time proxy derived from historical tables. That proxy is used for temporal alignment and then excluded from model features.

## LendingClub

The LendingClub external validation setup is application-time only.

Leakage columns are grouped into:
- post-outcome repayment columns
- policy/lender decision columns
- identifiers and free-text columns

Typical excluded LendingClub columns include:
- `out_prncp`
- `out_prncp_inv`
- `total_pymnt`
- `total_pymnt_inv`
- `total_rec_prncp`
- `total_rec_int`
- `recoveries`
- `collection_recovery_fee`
- `last_pymnt_d`
- `last_pymnt_amnt`
- `next_pymnt_d`
- `last_credit_pull_d`
- `last_fico_range_low`
- `last_fico_range_high`
- `grade`
- `sub_grade`
- `int_rate`
- `installment`
- `funded_amnt`
- `funded_amnt_inv`
- `id`
- `member_id`
- `url`
- `desc`
- `emp_title`
- `zip_code`

The full repository copy used by scripts lives in:
- `data/lendingclub/metadata/leakage_columns.yaml`
