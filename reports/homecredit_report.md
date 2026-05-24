# Home Credit Final Report

Dataset role: primary benchmark.

## Snapshot

| dataset_name | dataset_role      | DEV_rows | OOT_rows | DEV_bad_rate | OOT_bad_rate | time_column     | DEV_window   | OOT_window | engineered_candidate_features | encoded_or_modeling_features_if_available | LR_feature_budget | CatBoost_feature_budget | completed_runs | failed_runs |
| ------------ | ----------------- | -------- | -------- | ------------ | ------------ | --------------- | ------------ | ---------- | ----------------------------- | ----------------------------------------- | ----------------- | ----------------------- | -------------- | ----------- |
| Home Credit  | primary benchmark | 99092    | 120053   | 0.0793       | 0.089        | recent_decision | [-600, -240) | [-240, 0]  | 529.0                         |                                           | 20.0              | 40.0                    | 16             | 0           |

## DEV/OOT Split Rationale

The split is time-based rather than random. DEV is the older window used for cross-validation, feature selection, and model fitting, while OOT is the newer holdout used only for final evaluation. The window choice is justified by observation counts and target-rate behavior across time, with the goal of keeping both periods large enough for comparison without leaking future information into selector or model tuning. OOT bad rate is reported only to justify the validation setup; it is not used to tune feature selection or hyperparameters. For Home Credit, DEV uses relative days from -600 inclusive to -240 exclusive, and OOT uses -240 inclusive to 0 inclusive. This yields 99,092 DEV rows and 120,053 OOT rows, with bad rates of 0.0793 and 0.0890; the OOT minus DEV difference is 0.0097. That framing preserves an older development period and a more recent out-of-time period for realistic future-period validation.

| dataset    | dataset_display_name | time_column     | DEV_start | DEV_end | OOT_start | OOT_end | DEV_window   | OOT_window | DEV_rows | OOT_rows | DEV_bad_rate | OOT_bad_rate | bad_rate_difference | OOT_DEV_row_ratio | dropped_older_rows | dropped_missing_time_rows | source_row_count | DEV_issue_date_start | DEV_issue_date_end | OOT_issue_date_start | OOT_issue_date_end |
| ---------- | -------------------- | --------------- | --------- | ------- | --------- | ------- | ------------ | ---------- | -------- | -------- | ------------ | ------------ | ------------------- | ----------------- | ------------------ | ------------------------- | ---------------- | -------------------- | ------------------ | -------------------- | ------------------ |
| homecredit | Home Credit          | recent_decision | -600      | -240    | -240      | 0       | [-600, -240) | [-240, 0]  | 99092    | 120053   | 0.0793       | 0.089        | 0.0097              | 1.2115            | 71912              | 16454                     | 307511           |                      |                    |                      |                    |

## Experiment Matrix Overview

| dataset    | models       | selectors                                                                                          | feature_budgets | completed_run_count | failed_run_count |
| ---------- | ------------ | -------------------------------------------------------------------------------------------------- | --------------- | ------------------- | ---------------- |
| homecredit | catboost, lr | mrmr, boruta, pca, domain_rule_baseline, llm, llm_then_mrmr, llm_then_boruta, stable_core_llm_fill | 20, 40          | 16                  | 0                |

The matrix compares statistical baselines, pure LLM screening, and LLM-then-statistical hybrids under the same DEV/OOT protocol. The target comparison is therefore about first-stage screening utility, not about replacing the downstream LR or CatBoost evaluation vehicles.

## Topline Performance Comparison

Home Credit remains the main benchmark, and the topline leaderboard is mixed rather than one-sided. For LR, `stable_core_llm_fill` is best on OOT AUC (0.7489); for CatBoost, `stable_core_llm_fill` leads at 0.7683. The strongest non-LLM baseline is `mrmr` at 0.7668. The OOT gains of the best LLM-family method over the best baseline are small: 0.0032 AUC for LR and 0.0015 for CatBoost. Paired-fold CV deltas versus the mRMR baseline were negative (mean AUC delta -0.0038, 95% CI -0.0063 to -0.0013).

## Stability Review

Stability does not support a simple 'LLM dominates' claim. The highest Nogueira stability belongs to `pca` on `catboost` at 1.0000. Deterministic selectors such as PCA and the domain baseline show perfect or near-perfect repeatability, but that exact repeatability is not sufficient when OOT discrimination is weak. The stable-core hybrid improves the balance between exact feature stability and semantic stability more than the pure LLM selector.

## Drift and Robustness Review

High OOT performance on Home Credit is not concentrated in the highest-drift methods. The lowest-drift top run in the table is `llm` on `lr` with feature PSI mean 0.0015. PCA deserves specific caution: its OOT scores are weak and its drift indicators are materially worse than the better-performing selectors. OOT PSI is used only for evaluation; it is not a training or selection signal.

## Semantic Coverage Review

Semantic coverage is broader for mRMR and the better LLM hybrids than for PCA or the domain baseline. The broadest selector/model combination in the saved coverage table is `mrmr` on `catboost` with 10 distinct semantic groups. That supports the narrower claim that LLM screening can help preserve business-relevant feature families, but it does not remove the need for statistical discipline.

## Efficiency Tradeoff

Efficiency tradeoffs matter. Boruta is the slowest weak baseline, while the pure LLM LR run is cheap in wall-clock terms and reasonably competitive. Shared cache usage is already visible in the current artifacts: 48 cache hits are recorded and 0 tokens were effectively spent in the saved summaries, which limits repeated LLM cost for reused metadata rankings.

## Failure Cases and Surprises

The main failure cases are consistent. Boruta underperforms despite long runtime, PCA looks mechanically stable but not robust, and `llm_then_boruta` is clearly weaker than mRMR-based comparators. Home Credit still carries a manual-review caveat: auxiliary tables must be semantically confirmed as historical or as-of the application date, otherwise temporal meaning can be overstated.

## Conclusions

On Home Credit, the evidence supports a careful claim: LLM screening is useful as a first-stage helper, especially in the stable-core hybrid, but the improvement over mRMR is marginal rather than dominant. The strongest carry-forward method for cross-dataset discussion is `stable_core_llm_fill`, with mRMR as the non-LLM reference. The evidence is mixed across performance, stability, drift, and semantic coverage rather than coming from a single decisive metric.

## Next Actions

Concrete next actions after this reporting refactor are narrower: manually confirm Home Credit auxiliary-table as-of semantics, keep the LendingClub raw-data leakage blacklist audited as raw schemas change, and prepare the remaining future CLIP-style validation artifacts without training that method yet.

## Warnings

- Home Credit split diagnostics use application_train plus previous_application recency only, because no saved processed modeling table exists under data/homecredit/processed.
