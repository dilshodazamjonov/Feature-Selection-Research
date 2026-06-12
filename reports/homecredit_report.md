# Home Credit Final Report

Dataset role: primary benchmark.

## Research Question and Dataset Role

This research checks whether LLM metadata screening is useful as a first-stage feature-selection helper. Home Credit is the primary benchmark, and LendingClub is the external validation dataset. Logistic Regression and CatBoost are evaluation vehicles rather than the main contribution. Calibration, stacking, production scoring, and deployment are out of scope.

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

| model    | selector             | experiment_type | oot_auc | oot_gini | oot_ks | lift_at_10 | selected_feature_count | runtime_seconds | model_score_psi |
| -------- | -------------------- | --------------- | ------- | -------- | ------ | ---------- | ---------------------- | --------------- | --------------- |
| lr       | stable_core_llm_fill | hybrid          | 0.7489  | 0.4977   | 0.3688 | 3.1033     | 20.0                   | 1475.4382       | 0.005           |
| lr       | mrmr                 | statistical     | 0.7457  | 0.4914   | 0.3618 | 3.0958     | 20.0                   | 728.3589        | 0.0065          |
| lr       | llm                  | llm             | 0.74    | 0.4799   | 0.3573 | 3.0322     | 20.0                   | 47.6981         | 0.0066          |
| lr       | llm_then_mrmr        | hybrid          | 0.7381  | 0.4762   | 0.3537 | 2.9508     | 20.0                   | 85.996          | 0.0038          |
| lr       | domain_rule_baseline | statistical     | 0.7249  | 0.4498   | 0.3346 | 2.8713     | 20.0                   | 17.8531         | 0.0112          |
| lr       | pca                  | statistical     | 0.6729  | 0.3458   | 0.2597 | 2.2697     | 20.0                   | 17.5848         | 0.0291          |
| lr       | llm_then_boruta      | hybrid          | 0.6488  | 0.2976   | 0.2151 | 2.1584     | 20.0                   | 248.9272        | 0.0009          |
| lr       | boruta               | statistical     | 0.6306  | 0.2612   | 0.1874 | 2.0358     | 20.0                   | 1011.1587       | 0.0024          |
| catboost | stable_core_llm_fill | hybrid          | 0.7683  | 0.5367   | 0.4022 | 3.427      | 40.0                   | 4111.4814       | 0.0076          |
| catboost | mrmr                 | statistical     | 0.7668  | 0.5337   | 0.4017 | 3.4036     | 40.0                   | 960.1895        | 0.0102          |
| catboost | llm_then_mrmr        | hybrid          | 0.763   | 0.5259   | 0.3929 | 3.3718     | 40.0                   | 725.9166        | 0.0083          |
| catboost | llm_then_boruta      | hybrid          | 0.7592  | 0.5185   | 0.3872 | 3.3325     | 40.0                   | 812.0996        | 0.0094          |
| catboost | llm                  | llm             | 0.7569  | 0.5137   | 0.3822 | 3.2904     | 40.0                   | 520.9378        | 0.0054          |
| catboost | domain_rule_baseline | statistical     | 0.7306  | 0.4612   | 0.3408 | 3.0359     | 40.0                   | 407.9447        | 0.0054          |
| catboost | pca                  | statistical     | 0.7042  | 0.4083   | 0.3022 | 2.6149     | 40.0                   | 588.3895        | 0.0382          |
| catboost | boruta               | statistical     | 0.6852  | 0.3704   | 0.2713 | 2.5045     | 40.0                   | 1239.3889       | 0.0032          |

Home Credit remains the main benchmark, and the topline leaderboard is mixed rather than one-sided. For LR, `stable_core_llm_fill` is best on OOT AUC (0.7489); for CatBoost, `stable_core_llm_fill` leads at 0.7683. The strongest non-LLM baseline is `mrmr` at 0.7668. The OOT gains of the best LLM-family method over the best baseline are small: 0.0032 AUC for LR and 0.0015 for CatBoost. Paired-fold CV deltas versus the mRMR baseline were negative (mean AUC delta -0.0038, 95% CI -0.0063 to -0.0013).

Paired fold significance tests against mRMR:

| model    | candidate_selector   | baseline_selector | metric | mean_delta | ttest_p_value | wilcoxon_p_value | significant_at_0_05 | interpretation                                                       |
| -------- | -------------------- | ----------------- | ------ | ---------- | ------------- | ---------------- | ------------------- | -------------------------------------------------------------------- |
| catboost | stable_core_llm_fill | mrmr              | auc    | -0.0038    | 0.0425        | 0.0625           | True                | Candidate fold AUC is significantly lower than mRMR at alpha=0.05.   |
| catboost | llm                  | mrmr              | auc    | -0.015     | 0.0062        | 0.0625           | True                | Candidate fold AUC is significantly lower than mRMR at alpha=0.05.   |
| catboost | llm_then_mrmr        | mrmr              | auc    | -0.0017    | 0.7086        | 0.625            | False               | Mean fold AUC delta is tiny and not statistically significant.       |
| lr       | stable_core_llm_fill | mrmr              | auc    | -0.0045    | 0.0634        | 0.0625           | False               | Mean fold AUC is lower than mRMR, but not statistically significant. |
| lr       | llm                  | mrmr              | auc    | -0.0129    | 0.0133        | 0.0625           | True                | Candidate fold AUC is significantly lower than mRMR at alpha=0.05.   |
| lr       | llm_then_mrmr        | mrmr              | auc    | -0.0033    | 0.1563        | 0.1875           | False               | Mean fold AUC is lower than mRMR, but not statistically significant. |

## Stability Review

| model    | selector             | nogueira_stability | kuncheva_stability | mean_pairwise_jaccard | semantic_group_jaccard | stable_feature_count_80 | stable_feature_ratio_80 |
| -------- | -------------------- | ------------------ | ------------------ | --------------------- | ---------------------- | ----------------------- | ----------------------- |
| catboost | domain_rule_baseline | 1.0                | 1.0                | 1.0                   | 1.0                    | 40.0                    | 1.0                     |
| catboost | pca                  | 1.0                | 1.0                | 1.0                   | 1.0                    | 40.0                    | 1.0                     |
| catboost | mrmr                 | 0.7296             | 0.7296             | 0.6037                | 0.94                   | 29.0                    | 0.725                   |
| catboost | stable_core_llm_fill | 0.7025             | 0.7025             | 0.5725                | 0.8291                 | 28.0                    | 0.7                     |
| catboost | boruta               | 0.5646             | 0.5646             | 0.4529                | 0.6315                 | 25.0                    | 0.625                   |
| catboost | llm_then_boruta      | 0.4402             | 0.4402             | 0.3247                | 0.7367                 | 15.0                    | 0.375                   |
| catboost | llm                  | 0.4294             | 0.4294             | 0.3149                | 0.7473                 | 18.0                    | 0.45                    |
| catboost | llm_then_mrmr        | 0.4077             | 0.4077             | 0.2966                | 0.8625                 | 13.0                    | 0.325                   |
| lr       | domain_rule_baseline | 1.0                | 1.0                | 1.0                   | 1.0                    | 20.0                    | 1.0                     |
| lr       | pca                  | 1.0                | 1.0                | 1.0                   | 1.0                    | 20.0                    | 1.0                     |
| lr       | mrmr                 | 0.7714             | 0.7714             | 0.6417                | 0.9333                 | 15.0                    | 0.75                    |
| lr       | stable_core_llm_fill | 0.7454             | 0.7454             | 0.611                 | 0.9333                 | 15.0                    | 0.75                    |
| lr       | llm_then_mrmr        | 0.5531             | 0.5531             | 0.4043                | 0.7473                 | 9.0                     | 0.45                    |
| lr       | llm                  | 0.5479             | 0.5479             | 0.4076                | 0.598                  | 11.0                    | 0.55                    |
| lr       | boruta               | 0.5219             | 0.5219             | 0.4098                | 0.4532                 | 10.0                    | 0.5                     |
| lr       | llm_then_boruta      | 0.4388             | 0.4388             | 0.3031                | 0.6655                 | 8.0                     | 0.4                     |

Stability does not support a simple 'LLM dominates' claim. The highest Nogueira stability belongs to `pca` on `catboost` at 1.0000. Deterministic selectors such as PCA and the domain baseline show perfect or near-perfect repeatability, but that exact repeatability is not sufficient when OOT discrimination is weak. The stable-core hybrid improves the balance between exact feature stability and semantic stability more than the pure LLM selector.

## Drift and Robustness Review

| model    | selector             | selected_feature_count | psi_mean | psi_median | psi_p90 | psi_max | high_psi_feature_count | high_psi_feature_ratio |
| -------- | -------------------- | ---------------------- | -------- | ---------- | ------- | ------- | ---------------------- | ---------------------- |
| catboost | llm                  | 72                     | 0.0022   | 0.0        | 0.0061  | 0.0207  | 0                      | 0.0                    |
| catboost | domain_rule_baseline | 40                     | 0.0033   | 0.0024     | 0.0077  | 0.0207  | 0                      | 0.0                    |
| catboost | boruta               | 40                     | 0.0062   | 0.0039     | 0.0129  | 0.0223  | 0                      | 0.0                    |
| catboost | llm_then_mrmr        | 40                     | 0.0108   | 0.0037     | 0.0331  | 0.0579  | 0                      | 0.0                    |
| catboost | stable_core_llm_fill | 40                     | 0.0121   | 0.0048     | 0.0226  | 0.0935  | 0                      | 0.0                    |
| catboost | llm_then_boruta      | 40                     | 0.0133   | 0.006      | 0.0328  | 0.0822  | 0                      | 0.0                    |
| catboost | mrmr                 | 40                     | 0.0187   | 0.006      | 0.0585  | 0.167   | 0                      | 0.0                    |
| catboost | pca                  | 40                     | 0.0292   | 0.0196     | 0.0619  | 0.1217  | 0                      | 0.0                    |
| lr       | llm                  | 52                     | 0.0015   | 0.0        | 0.0039  | 0.0207  | 0                      | 0.0                    |
| lr       | domain_rule_baseline | 20                     | 0.003    | 0.0018     | 0.008   | 0.0207  | 0                      | 0.0                    |
| lr       | boruta               | 20                     | 0.0041   | 0.0036     | 0.0074  | 0.0119  | 0                      | 0.0                    |
| lr       | llm_then_boruta      | 20                     | 0.0047   | 0.0036     | 0.0086  | 0.0123  | 0                      | 0.0                    |
| lr       | llm_then_mrmr        | 20                     | 0.0118   | 0.006      | 0.033   | 0.0548  | 0                      | 0.0                    |
| lr       | stable_core_llm_fill | 20                     | 0.0125   | 0.0053     | 0.025   | 0.0774  | 0                      | 0.0                    |
| lr       | mrmr                 | 20                     | 0.0133   | 0.0057     | 0.025   | 0.0774  | 0                      | 0.0                    |
| lr       | pca                  | 20                     | 0.036    | 0.0277     | 0.0795  | 0.1217  | 0                      | 0.0                    |

High-PSI selected features:

No high-PSI selected features were flagged, or the artifact is unavailable.

`llm_then_mrmr` drift-source breakdown:

| dataset    | model    | run_id                                     | feature                          | in_llm_top_pool | in_final_selected_set | psi_dev_oot | semantic_group                 | source_table                 | missing_from_dev_oot_reason |
| ---------- | -------- | ------------------------------------------ | -------------------------------- | --------------- | --------------------- | ----------- | ------------------------------ | ---------------------------- | --------------------------- |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | INSTAL_DAYS_INSTALMENT_VAR       | True            | True                  | 0.0579      | installment_repayment_behavior | installments_payments        |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | INSTAL_DAYS_ENTRY_PAYMENT_VAR    | True            | True                  | 0.0548      | installment_repayment_behavior | installments_payments        |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | INSTAL_DAYS_INSTALMENT_MEAN      | True            | True                  | 0.0371      | installment_repayment_behavior | installments_payments        |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | INSTAL_DAYS_ENTRY_PAYMENT_MEAN   | True            | True                  | 0.0367      | installment_repayment_behavior | installments_payments        |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | CC_CNT_DRAWINGS_CURRENT_MEAN     | True            | True                  | 0.0327      | credit_card_utilization        | credit_card_balance          |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | INSTAL_AMT_INSTALMENT_MIN        | True            | True                  | 0.0326      | installment_repayment_behavior | installments_payments        |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | CC_AMT_RECEIVABLE_PRINCIPAL_MEAN | True            | True                  | 0.0223      | credit_card_utilization        | credit_card_balance          |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | EXT_SOURCE_3                     | True            | True                  | 0.0207      | external_score                 | application_{train|test}.csv |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | INSTAL_AMT_PAYMENT_MIN           | True            | True                  | 0.0205      | installment_repayment_behavior | installments_payments        |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | BURO_DAYS_CREDIT_MAX             | True            | True                  | 0.0119      | bureau_credit_history          | bureau                       |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | CC_CNT_DRAWINGS_ATM_CURRENT_MAX  | True            | True                  | 0.0109      | credit_card_utilization        | credit_card_balance          |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | CC_CNT_DRAWINGS_ATM_CURRENT_MEAN | True            | True                  | 0.0108      | credit_card_utilization        | credit_card_balance          |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | INSTAL_PAYMENT_DIFF_MEAN         | True            | True                  | 0.0093      | installment_repayment_behavior | installments_payments        |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | BURO_AMT_CREDIT_SUM_DEBT_MEAN    | True            | True                  | 0.0083      | bureau_debt                    | bureau                       |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | AMT_ANNUITY                      | True            | True                  | 0.0062      | previous_application_behavior  | previous_application.csv     |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | BURO_DEBT_CREDIT_DIFF_MEAN       | True            | True                  | 0.0061      | bureau_debt                    | bureau                       |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | DAYS_BIRTH                       | True            | True                  | 0.0058      | application_amounts            | application_{train|test}.csv |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | DAYS_EMPLOYED                    | True            | True                  | 0.0039      | income_capacity                | application_{train|test}.csv |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | BURO_DAYS_CREDIT_ENDDATE_MAX     | True            | True                  | 0.0039      | bureau_credit_history          | bureau                       |                             |
| homecredit | catboost | catboost_hybrid_llm_then_mrmr_87fbcccf4952 | INSTAL_IS_LATE_MEAN              | True            | True                  | 0.0038      | installment_repayment_behavior | installments_payments        |                             |

LLM top-100 candidate PSI evidence:

| dataset    | model    | selector | run_id                        | feature                        | llm_rank | in_llm_top100 | in_final_selected_set | selected_by_downstream_stat_selector | semantic_group                | source_table                 | psi_dev_oot | psi_flag    | missing_from_dev_oot_reason                                                                                                 |
| ---------- | -------- | -------- | ----------------------------- | ------------------------------ | -------- | ------------- | --------------------- | ------------------------------------ | ----------------------------- | ---------------------------- | ----------- | ----------- | --------------------------------------------------------------------------------------------------------------------------- |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | EXT_SOURCE_3                   | 1        | True          | True                  | False                                | external_score                | application_{train|test}.csv | 0.0207      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | EXT_SOURCE_2                   | 2        | True          | True                  | False                                | external_score                | application_{train|test}.csv | 0.0018      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | EXT_SOURCE_1                   | 3        | True          | True                  | False                                | external_score                | application_{train|test}.csv | 0.0005      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | DAYS_BIRTH                     | 4        | True          | True                  | False                                | application_amounts           | application_{train|test}.csv | 0.0058      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | DAYS_EMPLOYED                  | 5        | True          | True                  | False                                | income_capacity               | application_{train|test}.csv | 0.0039      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | OCCUPATION_TYPE                | 6        | True          | True                  | False                                | income_capacity               | application_{train|test}.csv |             | unavailable | DEV/OOT design matrix unavailable for LLM rejected candidates; selected-feature PSI exists only for final selected features |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | NAME_INCOME_TYPE               | 7        | True          | True                  | False                                | income_capacity               | application_{train|test}.csv |             | unavailable | DEV/OOT design matrix unavailable for LLM rejected candidates; selected-feature PSI exists only for final selected features |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | NAME_EDUCATION_TYPE            | 8        | True          | True                  | False                                | application_amounts           | application_{train|test}.csv |             | unavailable | DEV/OOT design matrix unavailable for LLM rejected candidates; selected-feature PSI exists only for final selected features |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | NAME_FAMILY_STATUS             | 9        | True          | True                  | False                                | application_amounts           | application_{train|test}.csv |             | unavailable | DEV/OOT design matrix unavailable for LLM rejected candidates; selected-feature PSI exists only for final selected features |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | CODE_GENDER                    | 10       | True          | True                  | False                                | application_amounts           | application_{train|test}.csv |             | unavailable | DEV/OOT design matrix unavailable for LLM rejected candidates; selected-feature PSI exists only for final selected features |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | AMT_CREDIT                     | 11       | True          | True                  | False                                | previous_application_behavior | previous_application.csv     | 0.0035      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | AMT_GOODS_PRICE                | 12       | True          | True                  | False                                | previous_application_behavior | previous_application.csv     | 0.0048      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | AMT_ANNUITY                    | 13       | True          | True                  | False                                | previous_application_behavior | previous_application.csv     | 0.0062      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | BURO_DEBT_RATIO_MAX            | 14       | True          | True                  | False                                | bureau_debt                   | bureau                       | 0.0024      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | BURO_DEBT_RATIO_MEAN           | 15       | True          | True                  | False                                | bureau_debt                   | bureau                       | 0.0024      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | BURO_DAYS_CREDIT_MEAN          | 16       | True          | True                  | False                                | bureau_credit_history         | bureau                       | 0.0031      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | BURO_DAYS_CREDIT_UPDATE_MEAN   | 17       | True          | True                  | False                                | bureau_credit_history         | bureau                       | 0.0027      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | BURO_DAYS_CREDIT_ENDDATE_MEAN  | 18       | True          | True                  | False                                | bureau_credit_history         | bureau                       | 0.0028      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | BURO_DAYS_CREDIT_MAX           | 19       | True          | True                  | False                                | bureau_credit_history         | bureau                       | 0.0119      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | BURO_DAYS_CREDIT_MIN           | 20       | True          | True                  | False                                | bureau_credit_history         | bureau                       | 0.0031      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | BURO_DAYS_ENDDATE_FACT_MEAN    | 21       | True          | True                  | False                                | bureau_credit_history         | bureau                       | 0.0027      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | BURO_DEBT_CREDIT_DIFF_MEAN     | 22       | True          | True                  | False                                | bureau_debt                   | bureau                       | 0.0061      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | BURO_DEBT_RATIO_VAR            | 23       | True          | True                  | False                                | bureau_debt                   | bureau                       | 0.0002      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | BURO_AMT_CREDIT_SUM_DEBT_MEAN  | 24       | True          | True                  | False                                | bureau_debt                   | bureau                       | 0.0083      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | BURO_AMT_CREDIT_SUM_DEBT_SUM   | 25       | True          | True                  | False                                | bureau_debt                   | bureau                       | 0.0037      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | BURO_AMT_CREDIT_SUM_LIMIT_MEAN | 26       | True          | True                  | False                                | bureau_debt                   | bureau                       | 0.001       | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | BURO_AMT_CREDIT_SUM_LIMIT_SUM  | 27       | True          | True                  | False                                | bureau_debt                   | bureau                       | 0.002       | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | BURO_AMT_CREDIT_SUM_DEBT_VAR   | 28       | True          | True                  | False                                | bureau_debt                   | bureau                       | 0.0073      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | BURO_DAYS_CREDIT_UPDATE_MAX    | 29       | True          | True                  | False                                | bureau_credit_history         | bureau                       | 0.0123      | low         |                                                                                                                             |
| homecredit | catboost | llm      | catboost_llm_llm_d54c966a1d6e | BURO_DAYS_CREDIT_ENDDATE_MAX   | 30       | True          | True                  | False                                | bureau_credit_history         | bureau                       | 0.0039      | low         |                                                                                                                             |

High OOT performance on Home Credit is not concentrated in the highest-drift methods. The lowest-drift top run in the table is `llm` on `lr` with feature PSI mean 0.0015. PCA deserves specific caution: its OOT scores are weak and its drift indicators are materially worse than the better-performing selectors. OOT PSI is used only for evaluation; it is not a training or selection signal.

## Semantic Coverage and Redundancy Review

| model    | selector             | selected feature count | number of semantic groups | semantic group entropy if easy | largest group share | average within-group absolute correlation | max within-group absolute correlation | redundancy risk flag                  |
| -------- | -------------------- | ---------------------- | ------------------------- | ------------------------------ | ------------------- | ----------------------------------------- | ------------------------------------- | ------------------------------------- |
| catboost | llm                  | 40                     | 8                         | 1.6661                         | 0.4                 |                                           |                                       | coverage_only_correlation_unavailable |
| catboost | llm_then_mrmr        | 40                     | 10                        | 2.1164                         | 0.25                |                                           |                                       | coverage_only_correlation_unavailable |
| catboost | mrmr                 | 40                     | 10                        | 2.1637                         | 0.225               |                                           |                                       | coverage_only_correlation_unavailable |
| catboost | stable_core_llm_fill | 40                     | 10                        | 2.1615                         | 0.2                 |                                           |                                       | coverage_only_correlation_unavailable |
| lr       | llm                  | 20                     | 7                         | 1.8479                         | 0.25                |                                           |                                       | coverage_only_correlation_unavailable |
| lr       | llm_then_mrmr        | 20                     | 7                         | 1.751                          | 0.35                |                                           |                                       | coverage_only_correlation_unavailable |
| lr       | mrmr                 | 20                     | 9                         | 2.1116                         | 0.2                 |                                           |                                       | coverage_only_correlation_unavailable |
| lr       | stable_core_llm_fill | 20                     | 9                         | 2.1116                         | 0.2                 |                                           |                                       | coverage_only_correlation_unavailable |

Semantic coverage is broader for mRMR and the better LLM hybrids than for PCA or the domain baseline. The broadest selector/model combination in the saved coverage table is `mrmr` on `catboost` with 10 distinct semantic groups. That supports the narrower claim that LLM screening can help preserve business-relevant feature families, but it does not remove the need for statistical discipline.

## Efficiency Tradeoff

Efficiency tradeoffs matter. Boruta is the slowest weak baseline, while the pure LLM LR run is cheap in wall-clock terms and reasonably competitive. Shared cache usage is already visible in the current artifacts: 48 cache hits are recorded and 0 tokens were effectively spent in the saved summaries, which limits repeated LLM cost for reused metadata rankings.

LLM call/cache summary:

| dataset    | LLM calls made | cache hits | total tokens | prompt version      | shared ranking enabled | number of runs sharing ranking |
| ---------- | -------------- | ---------- | ------------ | ------------------- | ---------------------- | ------------------------------ |
| homecredit | 0              | 48         | 0            | stability_expert_v3 | True                   | 8                              |

Full cache/hash appendix: `results/homecredit/final_report/appendix/full_llm_cache_summary.csv`.

## Best Runs Deep Dive

| analysis_label             | run_id                                            | model    | selector             | experiment_type | OOT_AUC | OOT_Gini | OOT_KS | CV_AUC_mean | CV_AUC_std | selected_feature_count | top_features                                                                                    | top_semantic_groups                                                                    | fold_behavior            | why_it_matters                               |
| -------------------------- | ------------------------------------------------- | -------- | -------------------- | --------------- | ------- | -------- | ------ | ----------- | ---------- | ---------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------ | -------------------------------------------- |
| best LR run                | lr_hybrid_stable_core_llm_fill_1ddd0142e614       | lr       | stable_core_llm_fill | hybrid          | 0.7489  | 0.4977   | 0.3688 | 0.7317      | 0.0144     | 20.0                   | EXT_SOURCE_2, EXT_SOURCE_3, PREV_INTEREST_ESTIMATE_MAX, EXT_SOURCE_1, DAYS_EMPLOYED             | bureau_debt (4), previous_application_behavior (3), external_score (3)                 | CV AUC 0.7317 +/- 0.0144 | highest OOT leaderboard entry for this slice |
| best CatBoost run          | catboost_hybrid_stable_core_llm_fill_8993eae5a4f7 | catboost | stable_core_llm_fill | hybrid          | 0.7683  | 0.5367   | 0.4022 | 0.748       | 0.0165     | 40.0                   | EXT_SOURCE_2, EXT_SOURCE_3, PREV_INTEREST_ESTIMATE_MAX, EXT_SOURCE_1, DAYS_EMPLOYED             | bureau_credit_history (8), bureau_debt (6), installment_repayment_behavior (6)         | CV AUC 0.7480 +/- 0.0165 | highest OOT leaderboard entry for this slice |
| strongest non-LLM baseline | catboost_statistical_mrmr_3858b721e537            | catboost | mrmr                 | statistical     | 0.7668  | 0.5337   | 0.4017 | 0.7518      | 0.0143     | 40.0                   | EXT_SOURCE_2, EXT_SOURCE_3, PREV_INTEREST_ESTIMATE_MAX, EXT_SOURCE_1, BURO_DEBT_CREDIT_DIFF_MIN | installment_repayment_behavior (9), previous_application_behavior (6), bureau_debt (5) | CV AUC 0.7518 +/- 0.0143 | reference baseline for non-LLM comparison    |

## Failure Cases and Surprises

The main failure cases are consistent. Boruta underperforms despite long runtime, PCA looks mechanically stable but not robust, and `llm_then_boruta` is clearly weaker than mRMR-based comparators. Home Credit auxiliary-table timing is treated as historical based on relative-time field semantics, but strict row-level as-of validation remains a manual-review limitation.

## Conclusions for This Dataset

On Home Credit, the evidence supports a careful claim: LLM screening is useful as a first-stage helper, especially in the stable-core hybrid, but the improvement over mRMR is marginal rather than dominant. The strongest carry-forward method for cross-dataset discussion is `stable_core_llm_fill`, with mRMR as the non-LLM reference. The evidence is mixed across performance, stability, drift, and semantic coverage rather than coming from a single decisive metric.

## Future Extension: CLIP-Style Semantic-Statistical Feature Alignment

This section is a reserved placeholder for future CLIP-style semantic-statistical feature alignment. It is not image CLIP and is not implemented here. The future method would align feature text and metadata with empirical feature behavior in a shared representation space, using DEV-only evidence and comparing the resulting screener against the current selectors under the same DEV/OOT protocol. OOT metrics must not be used for CLIP-style training or feature selection; they remain final evaluation only.

| planned_artifact                   | purpose                                                                           | status    | notes                                                  |
| ---------------------------------- | --------------------------------------------------------------------------------- | --------- | ------------------------------------------------------ |
| feature_level_evidence.csv         | one row per feature with semantic and empirical statistics                        | available | generated from current aggregate and per-run artifacts |
| contrastive_pairs.csv              | positive, hard-negative, and easy-negative feature pairs for contrastive training | planned   | pairs should be constructed from DEV-only evidence     |
| clip_embedding_table.csv           | learned feature embeddings and similarity to a stable-core anchor                 | planned   | dual-encoder or contrastive feature-space output       |
| clip_vs_llm_vs_mrmr_comparison.csv | compare the future CLIP-style screener against current selectors                  | planned   | must be evaluated under the same DEV/OOT protocol      |

## Next Actions

Concrete next actions after this reporting refactor are narrower: manually confirm Home Credit auxiliary-table as-of semantics, keep the LendingClub raw-data leakage blacklist audited as raw schemas change, and prepare the remaining future CLIP-style validation artifacts without training that method yet.

## Warnings

- Home Credit split diagnostics use application_train plus previous_application recency only, because no saved processed modeling table exists under data/homecredit/processed.
