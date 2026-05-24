# LendingClub Final Report

Dataset role: external validation.

## Snapshot

| dataset_name | dataset_role        | DEV_rows | OOT_rows | DEV_bad_rate | OOT_bad_rate | time_column     | DEV_window     | OOT_window    | engineered_candidate_features | encoded_or_modeling_features_if_available | LR_feature_budget | CatBoost_feature_budget | completed_runs | failed_runs |
| ------------ | ------------------- | -------- | -------- | ------------ | ------------ | --------------- | -------------- | ------------- | ----------------------------- | ----------------------------------------- | ----------------- | ----------------------- | -------------- | ----------- |
| LendingClub  | external validation | 598649   | 293105   | 0.1954       | 0.2329       | recent_decision | [-1795, -1065) | [-1065, -730] | 300.0                         | 96.0                                      | 20.0              | 40.0                    | 16             | 0           |

## DEV/OOT Split Rationale

The split is time-based rather than random. DEV is the older window used for cross-validation, feature selection, and model fitting, while OOT is the newer holdout used only for final evaluation. The window choice is justified by observation counts and target-rate behavior across time, with the goal of keeping both periods large enough for comparison without leaking future information into selector or model tuning. OOT bad rate is reported only to justify the validation setup; it is not used to tune feature selection or hyperparameters. For LendingClub, the configured relative window uses DEV from -1795 inclusive to -1065 exclusive and OOT from -1065 inclusive to -730 inclusive on `recent_decision`, which is derived from issue date to simulate future-loan validation. On the processed LendingClub application table, the DEV window corresponds approximately to issue dates 2014-01-01 through 2015-12-01, and OOT corresponds to 2016-01-01 through 2016-12-01. This produces 598,649 DEV rows and 293,105 OOT rows, with bad rates of 0.1954 and 0.2329; the difference is 0.0375.

| dataset     | dataset_display_name | time_column     | DEV_start | DEV_end | OOT_start | OOT_end | DEV_window     | OOT_window    | DEV_rows | OOT_rows | DEV_bad_rate | OOT_bad_rate | bad_rate_difference | OOT_DEV_row_ratio | dropped_older_rows | dropped_missing_time_rows | source_row_count | DEV_issue_date_start | DEV_issue_date_end | OOT_issue_date_start | OOT_issue_date_end |
| ----------- | -------------------- | --------------- | --------- | ------- | --------- | ------- | -------------- | ------------- | -------- | -------- | ------------ | ------------ | ------------------- | ----------------- | ------------------ | ------------------------- | ---------------- | -------------------- | ------------------ | -------------------- | ------------------ |
| lendingclub | LendingClub          | recent_decision | -1795     | -1065   | -1065     | -730    | [-1795, -1065) | [-1065, -730] | 598649   | 293105   | 0.1954       | 0.2329       | 0.0375              | 0.4896            | 230706             | 0                         | 1348099          | 2014-01-01           | 2015-12-01         | 2016-01-01           | 2016-12-01         |

## Experiment Matrix Overview

| dataset     | models       | selectors                                                                                          | feature_budgets | completed_run_count | failed_run_count |
| ----------- | ------------ | -------------------------------------------------------------------------------------------------- | --------------- | ------------------- | ---------------- |
| lendingclub | catboost, lr | mrmr, boruta, pca, domain_rule_baseline, llm, llm_then_mrmr, llm_then_boruta, stable_core_llm_fill | 20, 40          | 16                  | 0                |

The matrix compares statistical baselines, pure LLM screening, and LLM-then-statistical hybrids under the same DEV/OOT protocol. The target comparison is therefore about first-stage screening utility, not about replacing the downstream LR or CatBoost evaluation vehicles.

## Topline Performance Comparison

LendingClub acts as external validation, and the OOT leaderboard is tighter than on Home Credit. For LR, `llm` is best on OOT AUC (0.6982); for CatBoost, `llm` is best at 0.7165. The strongest non-LLM baseline is `mrmr` at 0.7060. The headline is not universal dominance: the best LLM-family methods sit near the top, but the margins over mRMR are modest. Paired-fold CV deltas versus the mRMR baseline were positive (mean AUC delta 0.0007, 95% CI 0.0001 to 0.0012). Paired-fold CV deltas versus the mRMR baseline were inconclusive (mean AUC delta 0.0003, 95% CI -0.0115 to 0.0121).

## Stability Review

The stability picture is better for the stronger hybrids than for the pure LLM selector. `domain_rule_baseline` on `catboost` has the highest saved Nogueira stability at 1.0000. Again, perfect-repeatability selectors such as PCA should not be overread: semantic concentration and weak robustness matter more than exact repeatability alone.

## Drift and Robustness Review

Drift on LendingClub is generally low for the best methods, which is encouraging for the external-validation claim. The best low-drift run in the drift table is `domain_rule_baseline` on `lr` with mean feature PSI 0.0012. PCA is the obvious exception and should be flagged explicitly because its feature PSI is much higher than the rest of the table.

## Semantic Coverage Review

Semantic diversity is narrower on LendingClub than on Home Credit because many selected sets collapse into broad 'other' or amount-related groups. That means a good LendingClub result should not be interpreted as proof of richer semantic coverage. The safer reading is that some LLM-family methods remain performance-competitive under a leakage-audited external dataset, even when semantic separation is less expressive.

## Efficiency Tradeoff

Efficiency is a more serious tradeoff on LendingClub. Boruta is expensive and weak, while the best LLM-family CatBoost runs are competitive but substantially slower than the best LR runs. The cache behavior still helps: the saved artifacts record 42 cache hits and 66020 total tokens, which indicates that shared ranking and reuse reduce repeated LLM cost.

## Failure Cases and Surprises

The main failure cases are again Boruta and PCA, with `llm_then_boruta` also clearly underperforming. LendingClub carries a separate data-governance caveat: the current processed dataset is the safe path, while raw direct use should remain blocked or tightly audited because the raw files contain post-origination leakage fields.

## Conclusions

On LendingClub, the honest claim is still moderate: LLM screening is useful as a first-stage helper, but it does not universally dominate the statistical baselines. The carry-forward methods for cross-dataset discussion are the best OOT LLM-family variant together with mRMR as the stability-aware non-LLM reference. The evidence is mixed, with small performance gaps, useful drift behavior, and only limited semantic-coverage separation.

## Next Actions

Concrete next actions after this reporting refactor are narrower: manually confirm Home Credit auxiliary-table as-of semantics, keep the LendingClub raw-data leakage blacklist audited as raw schemas change, and prepare the remaining future CLIP-style validation artifacts without training that method yet.

## Warnings

- No separate encoded-feature-count artifact exists; the snapshot reports engineered candidate features from experiment summaries and source-table width from the processed application table.
