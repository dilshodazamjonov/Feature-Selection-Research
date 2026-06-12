# LendingClub Final Report

Dataset role: external validation.

## Research Question and Dataset Role

This research checks whether LLM metadata screening is useful as a first-stage feature-selection helper. Home Credit is the primary benchmark, and LendingClub is the external validation dataset. Logistic Regression and CatBoost are evaluation vehicles rather than the main contribution. Calibration, stacking, production scoring, and deployment are out of scope.

## Snapshot

| dataset_name | dataset_role        | DEV_rows | OOT_rows | DEV_bad_rate | OOT_bad_rate | time_column     | DEV_window     | OOT_window    | engineered_candidate_features | encoded_or_modeling_features_if_available | LR_feature_budget | CatBoost_feature_budget | completed_runs | failed_runs |
| ------------ | ------------------- | -------- | -------- | ------------ | ------------ | --------------- | -------------- | ------------- | ----------------------------- | ----------------------------------------- | ----------------- | ----------------------- | -------------- | ----------- |
| LendingClub  | external validation | 598649   | 293105   | 0.1954       | 0.2329       | recent_decision | [-1795, -1065) | [-1065, -730] | 300.0                         | 96.0                                      | 20.0              | 40.0                    | 16             | 0           |

## DEV/OOT Split Rationale

The split is time-based rather than random. DEV is the older window used for cross-validation, feature selection, and model fitting, while OOT is the newer holdout used only for final evaluation. The window choice is justified by observation counts and target-rate behavior across time, with the goal of keeping both periods large enough for comparison without leaking future information into selector or model tuning. OOT bad rate is reported only to justify the validation setup; it is not used to tune feature selection or hyperparameters. For LendingClub, the configured relative window uses DEV from -1795 inclusive to -1065 exclusive and OOT from -1065 inclusive to -730 inclusive on `recent_decision`, which is derived from issue date to simulate future-loan validation. On the processed LendingClub application table, the DEV window corresponds approximately to issue dates 2014-01-01 through 2015-12-01, and OOT corresponds to 2016-01-01 through 2016-12-01. This produces 598,649 DEV rows and 293,105 OOT rows, with bad rates of 0.1954 and 0.2329; the difference is 0.0375. OOT has a higher bad rate than DEV, which makes LendingClub a harder external validation period, while both DEV and OOT retain enough observations for a meaningful comparison.

| dataset     | dataset_display_name | time_column     | DEV_start | DEV_end | OOT_start | OOT_end | DEV_window     | OOT_window    | DEV_rows | OOT_rows | DEV_bad_rate | OOT_bad_rate | bad_rate_difference | OOT_DEV_row_ratio | dropped_older_rows | dropped_missing_time_rows | source_row_count | DEV_issue_date_start | DEV_issue_date_end | OOT_issue_date_start | OOT_issue_date_end |
| ----------- | -------------------- | --------------- | --------- | ------- | --------- | ------- | -------------- | ------------- | -------- | -------- | ------------ | ------------ | ------------------- | ----------------- | ------------------ | ------------------------- | ---------------- | -------------------- | ------------------ | -------------------- | ------------------ |
| lendingclub | LendingClub          | recent_decision | -1795     | -1065   | -1065     | -730    | [-1795, -1065) | [-1065, -730] | 598649   | 293105   | 0.1954       | 0.2329       | 0.0375              | 0.4896            | 230706             | 0                         | 1348099          | 2014-01-01           | 2015-12-01         | 2016-01-01           | 2016-12-01         |

## Experiment Matrix Overview

| dataset     | models       | selectors                                                                                          | feature_budgets | completed_run_count | failed_run_count |
| ----------- | ------------ | -------------------------------------------------------------------------------------------------- | --------------- | ------------------- | ---------------- |
| lendingclub | catboost, lr | mrmr, boruta, pca, domain_rule_baseline, llm, llm_then_mrmr, llm_then_boruta, stable_core_llm_fill | 20, 40          | 16                  | 0                |

The matrix compares statistical baselines, pure LLM screening, and LLM-then-statistical hybrids under the same DEV/OOT protocol. The target comparison is therefore about first-stage screening utility, not about replacing the downstream LR or CatBoost evaluation vehicles.

## Topline Performance Comparison

| model    | selector             | experiment_type | oot_auc | oot_gini | oot_ks | lift_at_10 | selected_feature_count | runtime_seconds | model_score_psi |
| -------- | -------------------- | --------------- | ------- | -------- | ------ | ---------- | ---------------------- | --------------- | --------------- |
| lr       | llm                  | llm             | 0.6982  | 0.3965   | 0.283  | 2.1306     | 20.0                   | 353.2234        | 0.0058          |
| lr       | llm_then_mrmr        | hybrid          | 0.694   | 0.3881   | 0.278  | 2.1082     | 20.0                   | 273.5664        | 0.0051          |
| lr       | pca                  | statistical     | 0.6918  | 0.3835   | 0.277  | 2.0954     | 20.0                   | 81.8562         | 0.0139          |
| lr       | mrmr                 | statistical     | 0.6908  | 0.3816   | 0.275  | 2.0998     | 20.0                   | 428.6104        | 0.0042          |
| lr       | stable_core_llm_fill | hybrid          | 0.6907  | 0.3815   | 0.2735 | 2.0976     | 20.0                   | 1705.7226       | 0.004           |
| lr       | llm_then_boruta      | hybrid          | 0.6534  | 0.3069   | 0.2219 | 1.7696     | 20.0                   | 2110.3166       | 0.0078          |
| lr       | boruta               | statistical     | 0.635   | 0.27     | 0.1977 | 1.6931     | 20.0                   | 4143.0061       | 0.0079          |
| lr       | domain_rule_baseline | statistical     | 0.6199  | 0.2399   | 0.1734 | 1.613      | 20.0                   | 58.3984         | 0.0242          |
| catboost | llm                  | llm             | 0.7165  | 0.4329   | 0.3103 | 2.2538     | 40.0                   | 3624.1757       | 0.0055          |
| catboost | llm_then_mrmr        | hybrid          | 0.7071  | 0.4143   | 0.2973 | 2.2009     | 40.0                   | 3581.0058       | 0.0071          |
| catboost | stable_core_llm_fill | hybrid          | 0.7069  | 0.4139   | 0.2969 | 2.1952     | 40.0                   | 7254.8595       | 0.0074          |
| catboost | mrmr                 | statistical     | 0.706   | 0.412    | 0.2953 | 2.1956     | 40.0                   | 2699.7742       | 0.0073          |
| catboost | pca                  | statistical     | 0.6912  | 0.3824   | 0.2759 | 2.0645     | 40.0                   | 5198.8435       | 0.0279          |
| catboost | domain_rule_baseline | statistical     | 0.6818  | 0.3636   | 0.2615 | 1.9924     | 40.0                   | 3443.9921       | 0.0012          |
| catboost | llm_then_boruta      | hybrid          | 0.662   | 0.324    | 0.2319 | 1.8418     | 40.0                   | 5162.3791       | 0.0029          |
| catboost | boruta               | statistical     | 0.6604  | 0.3207   | 0.2305 | 1.8349     | 40.0                   | 5948.2805       | 0.0038          |

LendingClub acts as external validation, and the OOT leaderboard is tighter than on Home Credit. For LR, `llm` is best on OOT AUC (0.6982); for CatBoost, `llm` is best at 0.7165. The strongest non-LLM baseline is `mrmr` at 0.7060. The headline is not universal dominance: the best LLM-family methods sit near the top, but the margins over mRMR are modest. Paired-fold CV deltas versus the mRMR baseline were positive (mean AUC delta 0.0007, 95% CI 0.0001 to 0.0012). Paired-fold CV deltas versus the mRMR baseline were inconclusive (mean AUC delta 0.0003, 95% CI -0.0115 to 0.0121).

Paired fold significance tests against mRMR:

| model    | candidate_selector   | baseline_selector | metric | mean_delta | ttest_p_value | wilcoxon_p_value | significant_at_0_05 | interpretation                                                       |
| -------- | -------------------- | ----------------- | ------ | ---------- | ------------- | ---------------- | ------------------- | -------------------------------------------------------------------- |
| catboost | stable_core_llm_fill | mrmr              | auc    | 0.0007     | 0.0729        | 0.125            | False               | Mean fold AUC delta is tiny and not statistically significant.       |
| catboost | llm                  | mrmr              | auc    | 0.0003     | 0.9658        | 0.625            | False               | Mean fold AUC delta is tiny and not statistically significant.       |
| catboost | llm_then_mrmr        | mrmr              | auc    | 0.0        | 0.9429        | 0.8125           | False               | Mean fold AUC delta is tiny and not statistically significant.       |
| lr       | stable_core_llm_fill | mrmr              | auc    | 0.0001     | 0.8913        | 0.8125           | False               | Mean fold AUC delta is tiny and not statistically significant.       |
| lr       | llm                  | mrmr              | auc    | -0.0027    | 0.7429        | 0.625            | False               | Mean fold AUC is lower than mRMR, but not statistically significant. |
| lr       | llm_then_mrmr        | mrmr              | auc    | 0.0014     | 0.1483        | 0.1875           | False               | Mean fold AUC delta is tiny and not statistically significant.       |

## Stability Review

| model    | selector             | nogueira_stability | kuncheva_stability | mean_pairwise_jaccard | semantic_group_jaccard | stable_feature_count_80 | stable_feature_ratio_80 |
| -------- | -------------------- | ------------------ | ------------------ | --------------------- | ---------------------- | ----------------------- | ----------------------- |
| catboost | pca                  | 1.0                | 1.0                | 1.0                   | 1.0                    | 40.0                    | 1.0                     |
| catboost | domain_rule_baseline | 1.0                | 1.0                | 1.0                   | 1.0                    | 40.0                    | 1.0                     |
| catboost | llm_then_mrmr        | 0.8096             | 0.8096             | 0.7197                | 0.85                   | 32.0                    | 0.8                     |
| catboost | mrmr                 | 0.7981             | 0.7981             | 0.7081                | 0.9                    | 34.0                    | 0.85                    |
| catboost | stable_core_llm_fill | 0.7779             | 0.7779             | 0.6834                | 1.0                    | 33.0                    | 0.825                   |
| catboost | llm_then_boruta      | 0.7288             | 0.7288             | 0.6421                | 0.85                   | 31.0                    | 0.775                   |
| catboost | llm                  | 0.7288             | 0.7288             | 0.623                 | 0.8                    | 28.0                    | 0.7                     |
| catboost | boruta               | 0.7029             | 0.7029             | 0.6179                | 0.715                  | 28.0                    | 0.7                     |
| lr       | pca                  | 1.0                | 1.0                | 1.0                   | 1.0                    | 20.0                    | 1.0                     |
| lr       | domain_rule_baseline | 1.0                | 1.0                | 1.0                   | 1.0                    | 20.0                    | 1.0                     |
| lr       | boruta               | 0.7589             | 0.7589             | 0.6579                | 1.0                    | 16.0                    | 0.8                     |
| lr       | mrmr                 | 0.7536             | 0.7536             | 0.6309                | 0.8667                 | 14.0                    | 0.7                     |
| lr       | stable_core_llm_fill | 0.7054             | 0.7054             | 0.5902                | 0.8667                 | 15.0                    | 0.75                    |
| lr       | llm_then_mrmr        | 0.6571             | 0.6571             | 0.521                 | 1.0                    | 14.0                    | 0.7                     |
| lr       | llm_then_boruta      | 0.6196             | 0.6196             | 0.4925                | 0.8667                 | 12.0                    | 0.6                     |
| lr       | llm                  | 0.5714             | 0.5714             | 0.4444                | 1.0                    | 12.0                    | 0.6                     |

The stability picture is better for the stronger hybrids than for the pure LLM selector. `domain_rule_baseline` on `catboost` has the highest saved Nogueira stability at 1.0000. Again, perfect-repeatability selectors such as PCA should not be overread: semantic concentration and weak robustness matter more than exact repeatability alone.

## Drift and Robustness Review

| model    | selector             | selected_feature_count | psi_mean | psi_median | psi_p90 | psi_max | high_psi_feature_count | high_psi_feature_ratio |
| -------- | -------------------- | ---------------------- | -------- | ---------- | ------- | ------- | ---------------------- | ---------------------- |
| catboost | llm                  | 117                    | 0.0027   | 0.0        | 0.0093  | 0.0357  | 0                      | 0.0                    |
| catboost | boruta               | 40                     | 0.0039   | 0.0        | 0.0161  | 0.0344  | 0                      | 0.0                    |
| catboost | llm_then_boruta      | 40                     | 0.0044   | 0.0        | 0.0161  | 0.0344  | 0                      | 0.0                    |
| catboost | mrmr                 | 40                     | 0.006    | 0.0011     | 0.0189  | 0.0331  | 0                      | 0.0                    |
| catboost | llm_then_mrmr        | 40                     | 0.0061   | 0.0013     | 0.0189  | 0.0331  | 0                      | 0.0                    |
| catboost | stable_core_llm_fill | 40                     | 0.0063   | 0.0012     | 0.0198  | 0.0331  | 0                      | 0.0                    |
| catboost | domain_rule_baseline | 137                    | 0.0098   | 0.0        | 0.0004  | 1.2304  | 1                      | 0.0073                 |
| catboost | pca                  | 40                     | 1.0802   | 0.5816     | 2.2712  | 9.7747  | 27                     | 0.675                  |
| lr       | domain_rule_baseline | 40                     | 0.0012   | 0.0        | 0.0005  | 0.0193  | 0                      | 0.0                    |
| lr       | llm                  | 85                     | 0.0021   | 0.0        | 0.0068  | 0.0357  | 0                      | 0.0                    |
| lr       | boruta               | 20                     | 0.0029   | 0.0        | 0.0058  | 0.0279  | 0                      | 0.0                    |
| lr       | mrmr                 | 20                     | 0.0046   | 0.0003     | 0.0153  | 0.0279  | 0                      | 0.0                    |
| lr       | stable_core_llm_fill | 20                     | 0.0054   | 0.0009     | 0.016   | 0.0279  | 0                      | 0.0                    |
| lr       | llm_then_boruta      | 20                     | 0.0076   | 0.0005     | 0.0284  | 0.0344  | 0                      | 0.0                    |
| lr       | llm_then_mrmr        | 20                     | 0.0086   | 0.0039     | 0.0256  | 0.0331  | 0                      | 0.0                    |
| lr       | pca                  | 20                     | 1.3468   | 0.5932     | 2.6672  | 9.7747  | 12                     | 0.6                    |

High-PSI selected features:

| dataset     | model    | selector             | run_id                                                 | feature               | semantic_group  | psi_dev_oot | rank_within_pipeline | reason_flag |
| ----------- | -------- | -------------------- | ------------------------------------------------------ | --------------------- | --------------- | ----------- | -------------------- | ----------- |
| lendingclub | catboost | domain_rule_baseline | catboost_statistical_domain_rule_baseline_60387d67d96a | bankcard_capacity_gap | income_capacity | 1.2304      | 1.0                  | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC1                   | pca_component   | 9.7747      | 1.0                  | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC9                   | pca_component   | 3.4746      | 2.0                  | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC19                  | pca_component   | 2.5775      | 3.0                  | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC38                  | pca_component   | 2.3609      | 4.0                  | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC14                  | pca_component   | 2.2612      | 5.0                  | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC8                   | pca_component   | 2.1937      | 6.0                  | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC29                  | pca_component   | 2.1819      | 7.0                  | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC26                  | pca_component   | 2.0005      | 8.0                  | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC30                  | pca_component   | 1.9568      | 9.0                  | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC27                  | pca_component   | 1.5791      | 10.0                 | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC2                   | pca_component   | 1.3643      | 11.0                 | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC25                  | pca_component   | 1.1499      | 12.0                 | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC6                   | pca_component   | 1.1032      | 13.0                 | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC12                  | pca_component   | 0.9583      | 14.0                 | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC15                  | pca_component   | 0.822       | 15.0                 | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC33                  | pca_component   | 0.7324      | 16.0                 | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC32                  | pca_component   | 0.6385      | 17.0                 | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC20                  | pca_component   | 0.6317      | 18.0                 | PSI >= 0.25 |
| lendingclub | catboost | pca                  | catboost_statistical_pca_0952e619b24c                  | PC23                  | pca_component   | 0.6089      | 19.0                 | PSI >= 0.25 |

`llm_then_mrmr` drift-source breakdown:

| dataset     | model    | run_id                                     | feature                        | in_llm_top_pool | in_final_selected_set | psi_dev_oot | semantic_group           | source_table | missing_from_dev_oot_reason |
| ----------- | -------- | ------------------------------------------ | ------------------------------ | --------------- | --------------------- | ----------- | ------------------------ | ------------ | --------------------------- |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | total_bc_limit_per_bc_trade    | True            | True                  | 0.0331      | bankcard_capacity        |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | acc_open_past_24mths_share     | True            | True                  | 0.0279      | account_opening_activity |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | bc_open_to_buy                 | True            | True                  | 0.0256      | bankcard_capacity        |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | total_rev_hi_lim_per_rev_trade | True            | True                  | 0.0247      | revolving_utilization    |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | loan_to_total_limit            | True            | True                  | 0.0182      | other                    |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | num_tl_op_past_12m_share       | True            | True                  | 0.0182      | account_opening_activity |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | num_tl_op_past_12m             | True            | True                  | 0.0157      | account_opening_activity |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | acc_open_past_24mths           | True            | True                  | 0.015       | account_opening_activity |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | loan_to_income                 | True            | True                  | 0.0126      | income_capacity          |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | mort_acc                       | True            | True                  | 0.009       | mortgage_history         |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | fico_mean                      | True            | True                  | 0.0049      | fico_credit_score        |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | fico_range_low                 | True            | True                  | 0.0049      | fico_credit_score        |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | fico_range_high                | True            | True                  | 0.0049      | fico_credit_score        |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | annual_inc                     | True            | True                  | 0.0048      | income_capacity          |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | log_annual_inc                 | True            | True                  | 0.0048      | income_capacity          |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | inq_6m_per_open_acc            | True            | True                  | 0.0039      | recent_inquiries         |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | tot_hi_cred_lim                | True            | True                  | 0.0036      | other                    |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | active_revolving_share         | True            | True                  | 0.0032      | revolving_utilization    |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | balance_to_high_credit_limit   | True            | True                  | 0.003       | revolving_utilization    |              |                             |
| lendingclub | catboost | catboost_hybrid_llm_then_mrmr_429b32e00b37 | mths_since_recent_inq          | True            | True                  | 0.0012      | recent_inquiries         |              |                             |

LLM top-100 candidate PSI evidence:

| dataset     | model    | selector | run_id                        | feature                              | llm_rank | in_llm_top100 | in_final_selected_set | selected_by_downstream_stat_selector | semantic_group           | source_table | psi_dev_oot | psi_flag    | missing_from_dev_oot_reason         |
| ----------- | -------- | -------- | ----------------------------- | ------------------------------------ | -------- | ------------- | --------------------- | ------------------------------------ | ------------------------ | ------------ | ----------- | ----------- | ----------------------------------- |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | fico_mean                            | 1        | True          | True                  | False                                | fico_credit_score        |              | 0.0049      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | loan_to_income                       | 2        | True          | True                  | False                                | income_capacity          |              | 0.0126      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | dti                                  | 3        | True          | True                  | False                                | income_capacity          |              | 0.0011      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | term                                 | 4        | True          | True                  | False                                | loan_terms               |              |             | unavailable | numeric_dev_oot_values_unavailable  |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | home_ownership                       | 5        | True          | True                  | False                                | loan_terms               |              |             | unavailable | numeric_dev_oot_values_unavailable  |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | addr_state                           | 6        | True          | True                  | False                                | loan_terms               |              |             | unavailable | numeric_dev_oot_values_unavailable  |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | num_tl_op_past_12m                   | 7        | True          | True                  | False                                | account_opening_activity |              | 0.0157      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | mths_since_recent_inq                | 8        | True          | True                  | False                                | recent_inquiries         |              | 0.0012      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | total_bal_ex_mort_to_income          | 9        | True          | True                  | False                                | income_capacity          |              | 0.0004      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | tot_hi_cred_lim_to_income            | 10       | True          | True                  | False                                | income_capacity          |              | 0.0008      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | revolving_capacity_gap               | 11       | True          | True                  | False                                | revolving_utilization    |              | 0.0193      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | mort_acc                             | 12       | True          | True                  | False                                | mortgage_history         |              | 0.009       | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | bc_util                              | 13       | True          | True                  | False                                | bankcard_capacity        |              | 0.0348      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | revol_util                           | 14       | True          | True                  | False                                | revolving_utilization    |              | 0.0357      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | loan_amnt                            | 15       | True          | True                  | False                                | exposure_amount          |              | 0.0098      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | annual_inc                           | 16       | True          | True                  | False                                | income_capacity          |              | 0.0048      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | num_rev_tl_bal_gt_0                  | 17       | True          | True                  | False                                | revolving_utilization    |              | 0.0081      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | inq_last_6mths                       | 18       | True          | True                  | False                                | recent_inquiries         |              | 0.003       | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | purpose                              | 19       | True          | True                  | False                                | loan_terms               |              |             | unavailable | numeric_dev_oot_values_unavailable  |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | acc_open_past_24mths                 | 20       | True          | True                  | False                                | account_opening_activity |              | 0.015       | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | tot_cur_bal_to_income                | 21       | True          | True                  | False                                | income_capacity          |              | 0.0003      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | total_il_high_credit_limit_to_income | 22       | True          | True                  | False                                | income_capacity          |              | 0.0041      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | loan_per_total_acc                   | 23       | True          | True                  | False                                | account_opening_activity |              | 0.0045      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | num_actv_rev_tl                      | 24       | True          | True                  | False                                | revolving_utilization    |              | 0.0087      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | mo_sin_rcnt_tl                       | 25       | True          | True                  | False                                | credit_history_length    |              | 0.0035      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | mths_since_recent_bc                 | 26       | True          | True                  | False                                | bankcard_capacity        |              | 0.0077      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | total_rev_hi_lim                     | 27       | True          | True                  | False                                | revolving_utilization    |              | 0.0034      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | bc_open_to_buy                       | 28       | True          | True                  | False                                | bankcard_capacity        |              | 0.0256      | low         |                                     |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | num_tl_op_past_12m_band              | 29       | True          | True                  | False                                | account_opening_activity |              |             | unavailable | feature_not_in_processed_safe_frame |
| lendingclub | catboost | llm      | catboost_llm_llm_fe14a4902388 | mort_balance_pressure                | 30       | True          | True                  | False                                | mortgage_history         |              | 0.005       | low         |                                     |

Drift on LendingClub is generally low for the best methods, which is encouraging for the external-validation claim. The best low-drift run in the drift table is `domain_rule_baseline` on `lr` with mean feature PSI 0.0012. PCA is the obvious exception and should be flagged explicitly because its feature PSI is much higher than the rest of the table.

## Semantic Coverage and Redundancy Review

| model    | selector             | selected feature count | number of semantic groups | semantic group entropy if easy | largest group share | average within-group absolute correlation | max within-group absolute correlation | redundancy risk flag |
| -------- | -------------------- | ---------------------- | ------------------------- | ------------------------------ | ------------------- | ----------------------------------------- | ------------------------------------- | -------------------- |
| catboost | llm                  | 40                     | 11                        | 2.1739                         | 0.2                 | 0.3513                                    | 0.982                                 | high_max_correlation |
| catboost | llm_then_mrmr        | 40                     | 9                         | 1.9339                         | 0.35                | 0.7501                                    | 1.0                                   | high_max_correlation |
| catboost | mrmr                 | 40                     | 10                        | 2.0234                         | 0.25                | 0.6732                                    | 1.0                                   | high_max_correlation |
| catboost | stable_core_llm_fill | 40                     | 8                         | 1.9434                         | 0.25                | 0.6732                                    | 1.0                                   | high_max_correlation |
| lr       | llm                  | 20                     | 9                         | 2.0127                         | 0.25                | 0.3967                                    | 0.7583                                | moderate             |
| lr       | llm_then_mrmr        | 20                     | 8                         | 1.8344                         | 0.3                 | 0.4493                                    | 0.7583                                | moderate             |
| lr       | mrmr                 | 20                     | 5                         | 1.4878                         | 0.4                 | 1.0                                       | 1.0                                   | high_max_correlation |
| lr       | stable_core_llm_fill | 20                     | 6                         | 1.6385                         | 0.35                | 0.8791                                    | 1.0                                   | high_max_correlation |

Semantic diversity on LendingClub is more interpretable after the report-layer mapping update, because common credit-score, capacity, revolving-utilization, bankcard, and account-activity features no longer collapse unnecessarily into `other`. This relabeling improves the semantic coverage evidence but does not change feature selection results. The safer reading is that some LLM-family methods remain performance-competitive under a leakage-audited external dataset, while semantic coverage remains dataset and rule dependent.

## Efficiency Tradeoff

Efficiency is a more serious tradeoff on LendingClub. Boruta is expensive and weak, while the best LLM-family CatBoost runs are competitive but substantially slower than the best LR runs. The cache behavior still helps: the saved artifacts record 42 cache hits and 66020 total tokens, which indicates that shared ranking and reuse reduce repeated LLM cost.

LLM call/cache summary:

| dataset     | LLM calls made | cache hits | total tokens | prompt version      | shared ranking enabled | number of runs sharing ranking |
| ----------- | -------------- | ---------- | ------------ | ------------------- | ---------------------- | ------------------------------ |
| lendingclub | 6              | 42         | 66020        | stability_expert_v3 | True                   | 8                              |

Full cache/hash appendix: `results/lendingclub/final_report/appendix/full_llm_cache_summary.csv`.

## Best Runs Deep Dive

| analysis_label             | run_id                                 | model    | selector | experiment_type | OOT_AUC | OOT_Gini | OOT_KS | CV_AUC_mean | CV_AUC_std | selected_feature_count | top_features                                                                                          | top_semantic_groups                                                          | fold_behavior            | why_it_matters                               |
| -------------------------- | -------------------------------------- | -------- | -------- | --------------- | ------- | -------- | ------ | ----------- | ---------- | ---------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------ | -------------------------------------------- |
| best LR run                | lr_llm_llm_94542581466b                | lr       | llm      | llm             | 0.6982  | 0.3965   | 0.283  | 0.7121      | 0.023      | 20.0                   | fico_mean, loan_to_income, dti, term, home_ownership                                                  | income_capacity (5), loan_terms (4), revolving_utilization (3)               | CV AUC 0.7121 +/- 0.0230 | highest OOT leaderboard entry for this slice |
| best CatBoost run          | catboost_llm_llm_fe14a4902388          | catboost | llm      | llm             | 0.7165  | 0.4329   | 0.3103 | 0.7247      | 0.0227     | 40.0                   | fico_mean, loan_to_income, dti, term, home_ownership                                                  | income_capacity (8), revolving_utilization (7), account_opening_activity (6) | CV AUC 0.7247 +/- 0.0227 | highest OOT leaderboard entry for this slice |
| strongest non-LLM baseline | catboost_statistical_mrmr_e9152da1cab6 | catboost | mrmr     | statistical     | 0.706   | 0.412    | 0.2953 | 0.7244      | 0.0102     | 40.0                   | term_36 months, acc_open_past_24mths_share, dti, fico_range_high, term_home_ownership_60 months__RENT | loan_terms (10), other (9), income_capacity (5)                              | CV AUC 0.7244 +/- 0.0102 | reference baseline for non-LLM comparison    |

## Failure Cases and Surprises

The main failure cases are again Boruta and PCA, with `llm_then_boruta` also clearly underperforming. LendingClub carries a separate data-governance caveat: the current processed dataset is the safe path, while raw direct use should remain blocked or tightly audited because the raw files contain post-origination leakage fields. OOT has a higher bad rate than DEV, making the OOT period a harder external validation period while still retaining enough observations in both windows.

## Conclusions for This Dataset

On LendingClub, the honest claim is still moderate: LLM screening is useful as a first-stage helper, but it does not universally dominate the statistical baselines. The carry-forward methods for cross-dataset discussion are the best OOT LLM-family variant together with mRMR as the stability-aware non-LLM reference. The evidence is mixed, with small performance gaps, useful drift behavior, and only limited semantic-coverage separation.

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

- No separate encoded-feature-count artifact exists; the snapshot reports engineered candidate features from experiment summaries and source-table width from the processed application table.
