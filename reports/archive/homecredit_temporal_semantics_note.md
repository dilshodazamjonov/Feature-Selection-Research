# Home Credit Temporal Semantics Note

This note is intentionally conservative and does not claim false certainty about auxiliary-table as-of semantics.

Home Credit auxiliary-table timing is treated as historical based on relative-time field semantics, but strict row-level as-of validation remains a manual-review limitation.

## Auxiliary Tables Used

- `previous_application`
- `bureau`
- `installments_payments`
- `POS_CASH_balance`
- `credit_card_balance`
- application-level rows from `application_train` / `application_test`

## Current Enforcement

The code builds an application-time proxy (`application_time_proxy` / `recent_decision`) from historical relative-day fields and excludes that proxy plus `TARGET` and related time columns from model features. The current artifact review does not prove strict row-level as-of filtering inside every auxiliary table before aggregation.

## Source Semantics

The source fields appear historical by naming and Home Credit convention: `DAYS_DECISION`, `DAYS_CREDIT`, `DAYS_INSTALMENT`, `DAYS_ENTRY_PAYMENT`, and `MONTHS_BALANCE` are relative-time fields. That supports the current setup as a reasonable research proxy, but it is not a substitute for manual source documentation review.

## Evidence Supporting Current Setup

- Run-level leakage reports confirm target/time columns are excluded from feature matrices and OOT is not used in feature selection.
- `data/homecredit/metadata/temporal_asof_review.md` records the manual-review status explicitly.
- Experiment configs exclude `TARGET`, `recent_decision`, `PREV_recent_decision_MAX`, `DAYS_DECISION`, and `application_time_proxy`.

## Remaining Caveat

Manual review remains required for strict professor-facing claims that every auxiliary record is valid as of the application decision date. The current evidence supports a conservative statement: the setup uses historical-looking relative-time fields and removes split/proxy fields before modeling, but auxiliary-table as-of semantics are not fully auto-verified from files alone.

## What Would Require Rerun

A rerun would be required only if manual review changes the feature construction rules, excludes additional auxiliary records, changes the temporal proxy, or removes additional feature families. A documentation-only caveat does not require rerunning the current matrix.

## Existing Review Excerpt

# Home Credit Temporal As-Of Review

This artifact is intentionally conservative.

It does not claim that auxiliary Home Credit tables are fully verified as-of the application date.
Instead, it records which tables contribute temporal signals or proxy support and marks the current manual-review burden explicitly.

- Rows requiring explicit manual confirmation: `5`
- Automated verification is limited to repository structure and column usage.
- Source-table as-of semantics still require manual review for professor-facing reporting.

## Review Table

           table_name                           temporal_field                                                            used_for                                          current_repo_behavior       automated_verification_status                                                                                                 notes
    application_train                               row anchor                                      application-level modeling row    serves as the application-level anchor after feature merges anchored_by_primary_application_row                                  No auxiliary as-of assumption needed for the application row itself.
 previous_application                            DAYS_DECISION              direct recency signal and current split proxy fallback             feeds PREV_recent_decision_MAX and recent_decision              manual_review_required     Relative-day semantics look historical, but file-only checks cannot prove strict as-of alignment.
               bureau                              DAYS_CREDIT          application-time proxy support and bureau-history features           used in application_time_proxy and bureau aggregates              manual_review_required               Should represent historical bureau exposures, but semantic confirmation remains manual.
installments_payments     DAYS_INSTALMENT / DAYS_ENTRY_PAYMENT    application-time proxy support and installment behavior features        used in application_time_proxy and repayment aggregates              manual_review_required              Contains historical payment-event timing, but as-of treatment must be defended manually.
     POS_CASH_balance                           MONTHS_BALANCE            application-time proxy support and POS behavior features MONTHS_BALANCE is converted to relative days for proxy support              manual_review_required Relative-month balance history is plausible for historical use, not auto-verifiable from files alone.
  credit_card_balance                           MONTHS_BALANCE application-time proxy support and credit-card utilization features MONTHS_BALANCE is converted to relative days for proxy support              manual_review_required                    Historical balance interpretation should be confirmed manually for write-up rigor.
        derived_proxy application_time_proxy / recent_decision                                          DEV/OOT split construction          takes the most recent event across historical sources      depends_on_source_table_review                            The proxy is only as defensible as the source-table as-of semantics above.

## Practical Conclusion

Use the current Home Credit temporal split as a reasonable research proxy, but keep the auxiliary-table caveat in the final report until source semantics are manually confirmed.

