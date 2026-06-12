# Cross-Dataset V2 Analysis

This analysis compares `homecredit` and `lendingclub_v2` using existing completed aggregate artifacts only. It does not train models, rerun the experiment matrix, or call the LLM.

## Run Completeness

| dataset_name | matrix_rows | completed_runs | failed_runs | scheduled_or_pending_runs |
| --- | --- | --- | --- | --- |
| homecredit | 16 | 16 | 0 | 0 |
| lendingclub_v2 | 16 | 16 | 0 | 0 |

## Main Interpretation

- Do not compare raw AUC across datasets as if they were the same task; compare selector behavior within each dataset/model.
- LLM-family selectors are strongest when judged as low-drift, semantically broad first-stage screeners.
- mRMR remains the key exact-stability and non-LLM reference.
- PCA is mostly useful as a caution baseline when it has weaker discrimination or higher drift.

## Best Overall Runs

| dataset_name | model | selector | selector_family | oot_auc | oot_gini | lift_at_10 | selected_feature_psi_mean | model_score_psi | nogueira_stability | mean_pairwise_jaccard | semantic_group_jaccard | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| homecredit | catboost | stable_core_llm_fill | llm_family | 0.7683 | 0.5367 | 3.4270 | 0.0121 | 0.0076 | 0.7025 | 0.5725 | 0.8291 | 4111.4814 |
| homecredit | lr | stable_core_llm_fill | llm_family | 0.7489 | 0.4977 | 3.1033 | 0.0125 | 0.0050 | 0.7454 | 0.6110 | 0.9333 | 1475.4382 |
| lendingclub_v2 | catboost | llm | llm_family | 0.7137 | 0.4275 | 2.2326 | 0.0058 | 0.0058 | 0.6280 | 0.4828 | 0.9636 | 4427.8900 |
| lendingclub_v2 | lr | llm | llm_family | 0.6927 | 0.3853 | 2.1095 | 0.0044 | 0.0065 | 0.5517 | 0.4011 | 0.8578 | 357.3504 |

## Best LLM-Family Runs

| dataset_name | model | selector | selector_family | oot_auc | oot_gini | lift_at_10 | selected_feature_psi_mean | model_score_psi | nogueira_stability | mean_pairwise_jaccard | semantic_group_jaccard | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| homecredit | catboost | stable_core_llm_fill | llm_family | 0.7683 | 0.5367 | 3.4270 | 0.0121 | 0.0076 | 0.7025 | 0.5725 | 0.8291 | 4111.4814 |
| homecredit | lr | stable_core_llm_fill | llm_family | 0.7489 | 0.4977 | 3.1033 | 0.0125 | 0.0050 | 0.7454 | 0.6110 | 0.9333 | 1475.4382 |
| lendingclub_v2 | catboost | llm | llm_family | 0.7137 | 0.4275 | 2.2326 | 0.0058 | 0.0058 | 0.6280 | 0.4828 | 0.9636 | 4427.8900 |
| lendingclub_v2 | lr | llm | llm_family | 0.6927 | 0.3853 | 2.1095 | 0.0044 | 0.0065 | 0.5517 | 0.4011 | 0.8578 | 357.3504 |

## LLM-Family Wins Versus mRMR

| dataset_name | model | selector | oot_auc | delta_auc_vs_mrmr | delta_feature_psi_mean_vs_mrmr | delta_nogueira_vs_mrmr |
| --- | --- | --- | --- | --- | --- | --- |
| homecredit | catboost | stable_core_llm_fill | 0.7683 | 0.0015 | -0.0067 | -0.0270 |
| homecredit | lr | stable_core_llm_fill | 0.7489 | 0.0032 | -0.0008 | -0.0260 |
| lendingclub_v2 | catboost | llm | 0.7137 | 0.0129 | -0.0019 | -0.1488 |
| lendingclub_v2 | catboost | llm_then_mrmr | 0.7042 | 0.0033 | 0.0003 | -0.2365 |
| lendingclub_v2 | catboost | stable_core_llm_fill | 0.7028 | 0.0020 | 0.0012 | -0.0638 |
| lendingclub_v2 | lr | llm | 0.6927 | 0.0040 | -0.0016 | -0.2473 |
| lendingclub_v2 | lr | llm_then_mrmr | 0.6907 | 0.0021 | 0.0030 | -0.2061 |

## Family-Level Pattern

| dataset_name | model | selector_family | oot_auc | oot_gini | lift_at_10 | selected_feature_psi_mean | selected_feature_psi_high_drift_ratio | model_score_psi | nogueira_stability | mean_pairwise_jaccard | semantic_group_jaccard | runtime_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| homecredit | catboost | llm_family | 0.7618 | 0.5237 | 3.3554 | 0.0096 | 0.0000 | 0.0077 | 0.4949 | 0.3771 | 0.7939 | 1542.6089 |
| homecredit | catboost | statistical | 0.7217 | 0.4434 | 2.8898 | 0.0144 | 0.0000 | 0.0142 | 0.8235 | 0.7641 | 0.8929 | 798.9782 |
| homecredit | lr | llm_family | 0.7189 | 0.4378 | 2.8112 | 0.0076 | 0.0000 | 0.0041 | 0.5713 | 0.4315 | 0.7360 | 464.5149 |
| homecredit | lr | statistical | 0.6935 | 0.3871 | 2.5682 | 0.0141 | 0.0000 | 0.0123 | 0.8233 | 0.7629 | 0.8466 | 443.7389 |
| lendingclub_v2 | catboost | llm_family | 0.7014 | 0.4029 | 2.1552 | 0.0078 | 0.0000 | 0.0047 | 0.6286 | 0.4916 | 0.9144 | 7695.6677 |
| lendingclub_v2 | catboost | statistical | 0.6635 | 0.3270 | 1.8983 | 0.3071 | 0.1938 | 0.0107 | 0.8831 | 0.8265 | 0.9451 | 5633.6248 |
| lendingclub_v2 | lr | llm_family | 0.6788 | 0.3576 | 2.0105 | 0.0077 | 0.0000 | 0.0053 | 0.5994 | 0.4577 | 0.8117 | 1993.3277 |
| lendingclub_v2 | lr | statistical | 0.6300 | 0.2601 | 1.7140 | 0.3362 | 0.1625 | 0.0157 | 0.8854 | 0.8269 | 0.9429 | 2634.1416 |

## Lowest Mean Selected-Feature PSI By Dataset/Model

| dataset_name | model | selector_family | selected_feature_psi_mean | model_score_psi | oot_auc | nogueira_stability |
| --- | --- | --- | --- | --- | --- | --- |
| homecredit | catboost | llm_family | 0.0096 | 0.0077 | 0.7618 | 0.4949 |
| homecredit | catboost | statistical | 0.0144 | 0.0142 | 0.7217 | 0.8235 |
| homecredit | lr | llm_family | 0.0076 | 0.0041 | 0.7189 | 0.5713 |
| homecredit | lr | statistical | 0.0141 | 0.0123 | 0.6935 | 0.8233 |
| lendingclub_v2 | catboost | llm_family | 0.0078 | 0.0047 | 0.7014 | 0.6286 |
| lendingclub_v2 | catboost | statistical | 0.3071 | 0.0107 | 0.6635 | 0.8831 |
| lendingclub_v2 | lr | llm_family | 0.0077 | 0.0053 | 0.6788 | 0.5994 |
| lendingclub_v2 | lr | statistical | 0.3362 | 0.0157 | 0.6300 | 0.8854 |

## Paired Fold Evidence Versus mRMR

| dataset_name | model | candidate_selector | metric | mean_delta_candidate_minus_baseline | ci95_lower | ci95_upper | direction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| homecredit | catboost | llm_then_mrmr | auc | -0.0017 | -0.0099 | 0.0065 | candidate_below_mrmr |
| homecredit | catboost | stable_core_llm_fill | auc | -0.0038 | -0.0063 | -0.0013 | candidate_below_mrmr |
| homecredit | catboost | llm_then_boruta | auc | -0.0087 | -0.0157 | -0.0017 | candidate_below_mrmr |
| homecredit | catboost | llm | auc | -0.0150 | -0.0205 | -0.0094 | candidate_below_mrmr |
| homecredit | catboost | llm_then_mrmr | gini | -0.0034 | -0.0198 | 0.0131 | candidate_below_mrmr |
| homecredit | catboost | stable_core_llm_fill | gini | -0.0076 | -0.0127 | -0.0025 | candidate_below_mrmr |
| homecredit | catboost | llm_then_boruta | gini | -0.0174 | -0.0313 | -0.0035 | candidate_below_mrmr |
| homecredit | catboost | llm | gini | -0.0299 | -0.0410 | -0.0188 | candidate_below_mrmr |
| homecredit | lr | llm_then_mrmr | auc | -0.0033 | -0.0070 | 0.0004 | candidate_below_mrmr |
| homecredit | lr | stable_core_llm_fill | auc | -0.0045 | -0.0080 | -0.0010 | candidate_below_mrmr |
| homecredit | lr | llm | auc | -0.0129 | -0.0189 | -0.0069 | candidate_below_mrmr |
| homecredit | lr | llm_then_boruta | auc | -0.0164 | -0.0327 | -0.0002 | candidate_below_mrmr |
| homecredit | lr | llm_then_mrmr | gini | -0.0066 | -0.0141 | 0.0008 | candidate_below_mrmr |
| homecredit | lr | stable_core_llm_fill | gini | -0.0090 | -0.0159 | -0.0021 | candidate_below_mrmr |
| homecredit | lr | llm | gini | -0.0259 | -0.0378 | -0.0139 | candidate_below_mrmr |
| homecredit | lr | llm_then_boruta | gini | -0.0329 | -0.0654 | -0.0004 | candidate_below_mrmr |
| lendingclub_v2 | catboost | llm | auc | 0.0090 | 0.0065 | 0.0115 | candidate_above_mrmr |
| lendingclub_v2 | catboost | stable_core_llm_fill | auc | 0.0018 | 0.0004 | 0.0032 | candidate_above_mrmr |
| lendingclub_v2 | catboost | llm_then_mrmr | auc | 0.0014 | -0.0006 | 0.0034 | candidate_above_mrmr |
| lendingclub_v2 | catboost | llm_then_boruta | auc | -0.0058 | -0.0080 | -0.0036 | candidate_below_mrmr |
| lendingclub_v2 | catboost | llm | gini | 0.0180 | 0.0130 | 0.0230 | candidate_above_mrmr |
| lendingclub_v2 | catboost | stable_core_llm_fill | gini | 0.0036 | 0.0009 | 0.0063 | candidate_above_mrmr |
| lendingclub_v2 | catboost | llm_then_mrmr | gini | 0.0028 | -0.0012 | 0.0069 | candidate_above_mrmr |
| lendingclub_v2 | catboost | llm_then_boruta | gini | -0.0117 | -0.0161 | -0.0072 | candidate_below_mrmr |
| lendingclub_v2 | lr | llm | auc | 0.0048 | 0.0021 | 0.0074 | candidate_above_mrmr |
| lendingclub_v2 | lr | llm_then_mrmr | auc | 0.0016 | 0.0009 | 0.0023 | candidate_above_mrmr |
| lendingclub_v2 | lr | stable_core_llm_fill | auc | 0.0002 | -0.0012 | 0.0016 | candidate_above_mrmr |
| lendingclub_v2 | lr | llm_then_boruta | auc | -0.0340 | -0.0516 | -0.0165 | candidate_below_mrmr |
| lendingclub_v2 | lr | llm | gini | 0.0096 | 0.0043 | 0.0148 | candidate_above_mrmr |
| lendingclub_v2 | lr | llm_then_mrmr | gini | 0.0032 | 0.0018 | 0.0046 | candidate_above_mrmr |
| lendingclub_v2 | lr | stable_core_llm_fill | gini | 0.0004 | -0.0023 | 0.0032 | candidate_above_mrmr |
| lendingclub_v2 | lr | llm_then_boruta | gini | -0.0681 | -0.1031 | -0.0330 | candidate_below_mrmr |

## Semantic Coverage Summary

| dataset_name | model | selector | semantic_group_count | selected_feature_count | largest_semantic_group | largest_semantic_group_share |
| --- | --- | --- | --- | --- | --- | --- |
| homecredit | catboost | boruta | 4 | 40 | bureau_credit_history | 0.4500 |
| homecredit | catboost | domain_rule_baseline | 2 | 40 | bureau_debt | 0.9250 |
| homecredit | catboost | llm | 8 | 40 | bureau_credit_history | 0.4000 |
| homecredit | catboost | llm_then_boruta | 9 | 40 | bureau_credit_history | 0.2750 |
| homecredit | catboost | llm_then_mrmr | 10 | 40 | installment_repayment_behavior | 0.2500 |
| homecredit | catboost | mrmr | 10 | 40 | installment_repayment_behavior | 0.2250 |
| homecredit | catboost | pca | 1 | 40 | other | 1.0000 |
| homecredit | catboost | stable_core_llm_fill | 10 | 40 | bureau_credit_history | 0.2000 |
| homecredit | lr | boruta | 3 | 20 | bureau_debt | 0.5000 |
| homecredit | lr | domain_rule_baseline | 2 | 20 | bureau_debt | 0.8500 |
| homecredit | lr | llm | 7 | 20 | bureau_credit_history | 0.2500 |
| homecredit | lr | llm_then_boruta | 5 | 20 | bureau_credit_history | 0.5500 |
| homecredit | lr | llm_then_mrmr | 7 | 20 | installment_repayment_behavior | 0.3500 |
| homecredit | lr | mrmr | 9 | 20 | installment_repayment_behavior | 0.2000 |
| homecredit | lr | pca | 1 | 20 | other | 1.0000 |
| homecredit | lr | stable_core_llm_fill | 9 | 20 | bureau_debt | 0.2000 |
| lendingclub_v2 | catboost | boruta | 7 | 40 | account_opening_activity | 0.2750 |
| lendingclub_v2 | catboost | domain_rule_baseline | 3 | 40 | delinquency_behavior | 0.7750 |
| lendingclub_v2 | catboost | llm | 11 | 40 | revolving_utilization | 0.2250 |
| lendingclub_v2 | catboost | llm_then_boruta | 8 | 40 | account_opening_activity | 0.2250 |
| lendingclub_v2 | catboost | llm_then_mrmr | 10 | 40 | loan_terms | 0.2000 |
| lendingclub_v2 | catboost | mrmr | 8 | 40 | other | 0.2500 |
| lendingclub_v2 | catboost | pca | 1 | 40 | other | 1.0000 |
| lendingclub_v2 | catboost | stable_core_llm_fill | 9 | 40 | loan_terms | 0.1750 |
| lendingclub_v2 | lr | boruta | 6 | 20 | account_opening_activity | 0.5500 |
| lendingclub_v2 | lr | domain_rule_baseline | 2 | 20 | delinquency_behavior | 0.9500 |
| lendingclub_v2 | lr | llm | 8 | 20 | loan_terms | 0.2000 |
| lendingclub_v2 | lr | llm_then_boruta | 7 | 20 | fico_credit_score | 0.3000 |
| lendingclub_v2 | lr | llm_then_mrmr | 8 | 20 | account_opening_activity | 0.1500 |
| lendingclub_v2 | lr | mrmr | 6 | 20 | loan_terms | 0.3500 |
| lendingclub_v2 | lr | pca | 1 | 20 | other | 1.0000 |
| lendingclub_v2 | lr | stable_core_llm_fill | 7 | 20 | loan_terms | 0.3000 |

## Most Repeated Selected Features

| dataset_name | feature_name | semantic_group | selected_in_final_run_count | selected_in_llm_family_run_count | selected_in_baseline_run_count | selected_in_lr_run_count | selected_in_catboost_run_count | best_llm_final_dev_rank | best_oot_auc_when_selected | selectors_selected_by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| homecredit | BURO_DEBT_CREDIT_DIFF_MEAN | bureau_debt | 11 | 7 | 4 | 4 | 7 | 22.0000 | 0.7683 | boruta;domain_rule_baseline;llm;llm_then_boruta;llm_then_mrmr;mrmr;stable_core_llm_fill |
| homecredit | EXT_SOURCE_1 | external_score | 11 | 7 | 4 | 5 | 6 | 3.0000 | 0.7683 | domain_rule_baseline;llm;llm_then_boruta;llm_then_mrmr;mrmr;stable_core_llm_fill |
| homecredit | EXT_SOURCE_2 | external_score | 11 | 7 | 4 | 5 | 6 | 2.0000 | 0.7683 | domain_rule_baseline;llm;llm_then_boruta;llm_then_mrmr;mrmr;stable_core_llm_fill |
| homecredit | EXT_SOURCE_3 | external_score | 11 | 7 | 4 | 5 | 6 | 1.0000 | 0.7683 | domain_rule_baseline;llm;llm_then_boruta;llm_then_mrmr;mrmr;stable_core_llm_fill |
| homecredit | BURO_DAYS_CREDIT_ENDDATE_MEAN | bureau_credit_history | 10 | 7 | 3 | 4 | 6 | 18.0000 | 0.7683 | boruta;llm;llm_then_boruta;llm_then_mrmr;mrmr;stable_core_llm_fill |
| homecredit | BURO_DAYS_CREDIT_MAX | bureau_credit_history | 10 | 7 | 3 | 4 | 6 | 19.0000 | 0.7683 | boruta;llm;llm_then_boruta;llm_then_mrmr;mrmr;stable_core_llm_fill |
| homecredit | BURO_DAYS_CREDIT_MEAN | bureau_credit_history | 10 | 7 | 3 | 4 | 6 | 16.0000 | 0.7683 | boruta;llm;llm_then_boruta;llm_then_mrmr;mrmr;stable_core_llm_fill |
| homecredit | AMT_ANNUITY | previous_application_behavior | 9 | 8 | 1 | 4 | 5 | 13.0000 | 0.7683 | llm;llm_then_boruta;llm_then_mrmr;mrmr;stable_core_llm_fill |
| homecredit | AMT_CREDIT | previous_application_behavior | 9 | 7 | 2 | 4 | 5 | 11.0000 | 0.7683 | boruta;llm;llm_then_boruta;llm_then_mrmr;stable_core_llm_fill |
| homecredit | DAYS_EMPLOYED | income_capacity | 9 | 7 | 2 | 4 | 5 | 5.0000 | 0.7683 | llm;llm_then_boruta;llm_then_mrmr;mrmr;stable_core_llm_fill |
| lendingclub_v2 | dti | income_capacity | 10 | 8 | 2 | 5 | 5 | 4.0000 | 0.7137 | llm;llm_then_boruta;llm_then_mrmr;mrmr;stable_core_llm_fill |
| lendingclub_v2 | acc_open_past_24mths | account_opening_activity | 10 | 7 | 3 | 4 | 6 | 12.0000 | 0.7137 | boruta;llm;llm_then_boruta;llm_then_mrmr;mrmr;stable_core_llm_fill |
| lendingclub_v2 | acc_open_past_24mths_per_credit_history_year | credit_history_length | 10 | 6 | 4 | 5 | 5 | 35.0000 | 0.7042 | boruta;llm_then_boruta;llm_then_mrmr;mrmr;stable_core_llm_fill |
| lendingclub_v2 | fico_mean | fico_credit_score | 9 | 7 | 2 | 5 | 4 | 1.0000 | 0.7137 | llm;llm_then_boruta;llm_then_mrmr;mrmr;stable_core_llm_fill |
| lendingclub_v2 | acc_open_past_24mths_per_total_acc | account_opening_activity | 9 | 5 | 4 | 3 | 6 | 10.0000 | 0.7137 | boruta;llm;llm_then_boruta;llm_then_mrmr;mrmr;stable_core_llm_fill |
| lendingclub_v2 | acc_open_past_24mths_share | account_opening_activity | 8 | 4 | 4 | 3 | 5 | 63.0000 | 0.7042 | boruta;llm_then_boruta;llm_then_mrmr;mrmr;stable_core_llm_fill |
| lendingclub_v2 | dti_x_revol_util | revolving_utilization | 7 | 6 | 1 | 2 | 5 | 20.0000 | 0.7137 | llm;llm_then_boruta;llm_then_mrmr;mrmr;stable_core_llm_fill |
| lendingclub_v2 | loan_amnt_to_income | income_capacity | 7 | 5 | 2 | 3 | 4 | 3.0000 | 0.7137 | llm;llm_then_mrmr;mrmr;stable_core_llm_fill |
| lendingclub_v2 | annual_inc | income_capacity | 7 | 4 | 3 | 3 | 4 | 5.0000 | 0.7137 | boruta;domain_rule_baseline;llm;llm_then_boruta |
| lendingclub_v2 | bc_open_to_buy_to_loan_amnt | bankcard_capacity | 6 | 5 | 1 | 1 | 5 | 9.0000 | 0.7137 | boruta;llm;llm_then_boruta;llm_then_mrmr;stable_core_llm_fill |

## Output Tables

- `results/cross_dataset_v2/analysis/final.csv`
- `results/cross_dataset_v2/analysis/best_overall.csv`
- `results/cross_dataset_v2/analysis/best_llm_family.csv`
- `results/cross_dataset_v2/analysis/deltas_vs_mrmr.csv`
- `results/cross_dataset_v2/analysis/family_summary.csv`
- `results/cross_dataset_v2/analysis/paired_fold_evidence.csv`
- `results/cross_dataset_v2/analysis/semantic_summary.csv`
- `results/cross_dataset_v2/analysis/top_feature_evidence.csv`