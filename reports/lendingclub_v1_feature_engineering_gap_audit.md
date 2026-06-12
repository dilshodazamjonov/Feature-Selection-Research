# LendingClub v1 Feature Engineering Gap Audit

## Summary

- Raw safe processed input columns after excluding target/time/status helpers: `96`.
- Reported v1 engineered candidate features: approximately `300`.
- v1 features with usable LLM descriptions: approximately `76`.
- `data/lendingclub/metadata/columns_description.csv` contains only the header in the current workspace, so v1 did not have a complete source-level description table.
- The richer v1 LendingClub feature builder exists in `src/credit_risk_fs/feature_engineering/lendingclub/application.py`, but the v1 prepared `application_train.csv` contains the safe source table rather than all engineered features.

## Answers

1. Safe raw columns available after preprocessing: 96 candidate columns.
2. Engineered candidate columns reported in v1 reports: about 300.
3. Features with descriptions in v1 LLM/reporting artifacts: about 76.
4. Underrepresented concepts: categorical grouping metadata, systematic missingness indicators, FICO-affordability interactions, account-depth ratios, recency flags, joint-applicant coverage, balance-to-limit pressure variants, and interpretable grouped categorical interactions.
5. Selected or LLM-ranked features lacking descriptions cannot be fully rechecked from `results/lendingclub` because that result folder is absent in the current workspace; the available generated report shows 76 features with LLM rank/description and 429 broader-union features without descriptions.
6. Vague semantic groups were most likely caused by the empty source description CSV and fallback name-only inference. v2 writes a semantic group for every candidate feature.
7. Yes. LendingClub v1 is much simpler than Home Credit by metadata coverage and by number of described features.
8. Yes, this simplicity could partly explain why pure LLM wins on LendingClub: the LLM was screening a smaller, more metadata-filtered candidate set, while the broader feature universe was under-described.