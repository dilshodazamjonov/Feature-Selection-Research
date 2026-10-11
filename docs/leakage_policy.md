# Leakage policy

Leakage control is mandatory in this repository.

## Global principles

- Target, identifier and ordering-index columns never reach model features.
- HO rows are never used for screening, selection, ranking or tuning; they are scored once by the full-DEV pipeline.
- Fold-local selectors and preprocessing maps see only the training partition of their fold; the final pipeline is
  fitted on full DEV only.
- The language model receives definition-only feature records: names, descriptions, source families, lineage formulas
  and semantic groups. It never receives a data row, an outcome, a statistic, a validation result or HO information.

## Home Credit

Protected columns include `TARGET`, `recent_decision`, `PREV_recent_decision_MAX`, `DAYS_DECISION` and
`application_time_proxy`. Functions of the ordering index (previous-application decision days and their aggregates) are
outside every candidate set (`index_derived_patterns` in `configs/study/homecredit.yaml`). The application-time proxy
derived from historical tables is used for temporal alignment and excluded from model features.

## LendingClub

The LendingClub v2 universe is application-time only. Excluded source columns fall into post-outcome repayment columns,
the platform's pricing and grading fields, and identifiers and free text, for example `out_prncp`, `out_prncp_inv`,
`total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `recoveries`, `collection_recovery_fee`,
`last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`, `last_credit_pull_d`, `last_fico_range_low`, `last_fico_range_high`,
`hardship_flag`, `debt_settlement_flag`, `grade`, `sub_grade`, `int_rate`, `installment`, `funded_amnt`,
`funded_amnt_inv`, `id`, `member_id`, `url`, `desc`, `emp_title` and `zip_code`. The repository copy used by the scripts
is `data/lendingclub/metadata/leakage_columns.yaml`.

## Stability 2024

The protocol lock (`configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json`) records the
prediction-time availability review of every raw variable; only approved predictors enter the universe, depth-2 tables
and future information are excluded, and `case_id`, `date_decision`, `MONTH`, `WEEK_NUM` and `target` are never
predictors.
