# LendingClub v2 Feature Engineering Spec

## Purpose

LendingClub v2 is an isolated preparation track for improving feature richness and metadata coverage before any approved LendingClub rerun. It does not change the target definition, does not use post-origination fields, and does not run the experiment matrix.

## Count Interpretation

- Engineered candidates are columns produced for model candidate selection.
- Broader evidence-table union is the union of features mentioned across selected sets, rankings, and artifacts; it can include transformed, encoded, or derived references.
- Features with descriptions are features with usable metadata descriptions available to LLM/reporting.
- Missing descriptions are features that need metadata before LLM screening is fair.

## Safety Constraints

The v2 preparation uses the existing safe LendingClub processed source as input. It preserves `TARGET`, `recent_decision`, and `issue_d` as helper columns and excludes target/status/payment/outcome fields from candidate features.

Excluded leakage categories remain strict:

- payment and recovery fields
- settlement fields
- hardship fields
- post-origination status fields
- future payment/date fields
- last payment fields
- last credit-pull fields
- `loan_status`
- target-derived fields
- fields only known after loan performance

Examples explicitly excluded: `total_pymnt`, `total_rec_prncp`, `total_rec_int`, `recoveries`, `collection_recovery_fee`, `settlement_*`, `hardship_*`, `last_pymnt_*`, `next_pymnt_*`, `last_credit_pull_*`, and `loan_status`.

## Planned Feature Groups

### Credit Score / FICO

Examples: `fico_mean`, `fico_span`, `fico_midpoint_scaled`, `fico_to_income`, `fico_to_loan_amount`, `fico_band`, `fico_band_x_term`.

### Income and Affordability

Examples: `log_annual_inc`, `loan_to_income`, `dti_band`, `high_dti_flag`, `very_high_dti_flag`, amount-to-income ratios, verification-by-affordability interactions.

### Loan Exposure and Terms

Examples: `loan_amnt`, `log_loan_amnt`, `sqrt_loan_amnt`, `loan_amnt_band`, `term_months`, `term_home_ownership`, `term_verification_status`, `term_x_loan_to_income`.

### Revolving Utilization

Examples: `revol_util`, `revol_bal`, `log_revol_bal`, `revol_bal_to_income`, `revol_bal_to_loan_amnt`, `high_revol_util_flag`, fixed utilization threshold flags, `fico_adjusted_revol_util`.

### Bankcard Capacity

Examples: `bc_open_to_buy`, `bc_util`, `total_bc_limit`, `total_bc_limit_per_bc_trade`, `bc_open_to_buy_share`, `bankcard_capacity_gap`, `bc_util_ge_75_flag`.

### Credit History Length

Examples: `credit_history_months`, `credit_history_years`, `credit_history_band`, `credit_lines_per_history_year`, `mths_since_recent_bc_inverse_recency`, recency flags for recent bankcard/inquiry/delinquency fields.

### Recent Inquiries and Account Opening

Examples: `inq_last_6mths`, `inq_last_12m`, inquiry-per-account ratios, `recent_inquiry_density`, `recent_trade_density`, `acc_open_past_24mths_share`, `num_tl_op_past_12m_share`.

### Account Mix and Credit Depth

Examples: `total_acc`, `open_acc`, `open_acc_band`, `total_acc_band`, `num_actv_rev_tl`, `num_bc_tl`, `num_il_tl`, `open_installment_share`, `open_revolving_share`, account-count density features.

### Delinquency / Derogatory History

Examples: `delinq_2yrs`, `delinq_amnt`, `pub_rec`, `pub_rec_bankruptcies`, `tax_liens`, `collections_12_mths_ex_med`, `chargeoff_within_12_mths`, delinquency pressure, severe delinquency count, per-account derogatory ratios.

### Balance and Credit-Limit Pressure

Examples: `tot_cur_bal`, `tot_hi_cred_lim`, `total_bal_ex_mort`, `total_il_high_credit_limit`, balance-to-income ratios, balance-to-loan ratios, `loan_to_total_credit_limit_v2`, `balance_to_high_credit_limit`.

### Joint Applicant / Secondary Applicant

Examples: `annual_inc_joint`, `dti_joint`, `joint_income_to_single_income`, `sec_app_fico_mean`, `sec_app_mort_acc`, `sec_app_open_acc`, `sec_app_revol_util`, secondary-applicant missingness and presence flags.

### Categorical Application Features

Examples: `home_ownership`, `verification_status`, `purpose`, `addr_state`, `application_type`, `initial_list_status`, `purpose_group`, `home_ownership_group`, `verification_group`, `state_region_group`.

### Missingness Indicators

Examples: `mths_since_last_delinq_missing_flag`, `mths_since_last_record_missing_flag`, `mths_since_recent_inq_missing_flag`, `mths_since_recent_bc_missing_flag`, `revol_util_missing_flag`, `bc_util_missing_flag`, joint-applicant missingness indicators.

### Interaction Features

Interactions are limited to interpretable combinations: `loan_to_income_x_fico`, `dti_x_revol_util`, `dti_x_bc_util`, `recent_inquiry_density_x_fico`, `delinquency_pressure_x_fico`, `purpose_group_x_dti_band`, and `verification_group_x_loan_to_income_band`.

## Approval Standard

Before any rerun, every candidate feature in `data/lendingclub_v2/processed/application_train.csv` must have:

- nonblank description
- nonblank semantic group
- nonblank source column or formula
- nonblank leakage review status
- no `needs_manual_review` leakage status unless explicitly approved

The target is at least 500 defensible candidate features. If that cannot be achieved safely, the preparation report must state the maximum safe count and explain the constraint.
