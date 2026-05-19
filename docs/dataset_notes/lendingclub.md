# LendingClub Notes

LendingClub is treated as external validation, not the primary development dataset.

## Layout

Raw CSV:
- `data/lendingclub/raw`

Processed single-table modeling file:
- `data/lendingclub/processed/application_train.csv`

Metadata:
- `data/lendingclub/metadata`

## Target Construction

The preparation script builds:
- `TARGET`
- `recent_decision`

and filters to final resolved good/bad statuses for an application-time default task.

## Label And Leakage Audit

The LendingClub preparation step also writes a label and leakage audit under `data/lendingclub/metadata/`.

Saved artifacts:
- `target_definition.md`
- `leakage_columns.yaml`
- `label_distribution.csv`
- `issue_date_target_distribution.csv`

Audit policy:
- define `TARGET` from final resolved outcomes only
- remove ongoing or ambiguous statuses such as `Current`, `Issued`, `In Grace Period`, and `Late`
- audit 36-month and 60-month term distributions separately to monitor censoring risk
- drop post-origination, payment, settlement, recovery, policy, and text leakage fields before modeling

## Leakage

The LendingClub setup must drop:
- post-outcome payment and recovery fields
- underwriting policy outputs such as grade/sub-grade/int_rate/installment
- identifiers and free-text fields

This is mandatory if the dataset is used as external validation for the research claim.
