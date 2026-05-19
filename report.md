# Credit-Risk Feature Selection Research Report

Date: 2026-05-19

## Executive Summary

This report evaluates LLM-assisted feature selection for credit-risk scoring on two datasets:

- Home Credit: primary development dataset.
- LendingClub: external validation dataset.

The experiment matrix completed cleanly for both datasets. Each dataset has 16 completed runs: 2 models (`lr`, `catboost`) crossed with 8 selector variants. No failed runs were recorded in `failed_runs.csv`.

The evidence supports a conservative claim: LLM-based metadata screening is useful as a first-stage feature-selection helper, especially when combined with stability constraints or used as an external-validation screen. The evidence does not support a broad claim that LLM selection universally dominates statistical selectors. On Home Credit, `stable_core_llm_fill` was the best OOT performer for both LR and CatBoost, but mRMR was very close and sometimes more stable. On LendingClub, pure `llm` was the best OOT performer for both LR and CatBoost.

Top-line winners by OOT AUC:

| Dataset | Model | Best selector | CV AUC mean +/- SD | OOT AUC | OOT Gini | OOT KS | Nogueira stability | Model PSI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Home Credit | lr | stable_core_llm_fill | 0.7317 +/- 0.0144 | 0.7489 | 0.4977 | 0.3688 | 0.7454 | 0.0050 |
| Home Credit | catboost | stable_core_llm_fill | 0.7480 +/- 0.0165 | 0.7683 | 0.5367 | 0.4022 | 0.7025 | 0.0076 |
| LendingClub | lr | llm | 0.7121 +/- 0.0230 | 0.6982 | 0.3965 | 0.2830 | 0.5714 | 0.0058 |
| LendingClub | catboost | llm | 0.7247 +/- 0.0227 | 0.7165 | 0.4329 | 0.3103 | 0.7288 | 0.0055 |

Reproducibility note: the top-level aggregate CSVs under `results/homecredit/` and `results/lendingclub/` have been regenerated from the completed per-run outputs. The numeric values in this report match the populated aggregate tables and the underlying `results/experiment_summary.csv` files.

## Experiment Setup

| Item | Home Credit | LendingClub |
| --- | --- | --- |
| Role | Main development dataset | External validation dataset |
| Data path | `data/homecredit/raw` | `data/lendingclub/processed` |
| Metadata path | `data/homecredit/metadata/columns_description.csv` | `data/lendingclub/metadata/columns_description.csv` |
| DEV window | day -600 inclusive to -240 exclusive | day -1795 inclusive to -1065 exclusive |
| OOT window | day -240 inclusive to 0 inclusive | day -1065 inclusive to -730 inclusive |
| DEV rows | 99,092 | 598,649 |
| OOT rows | 120,053 | 293,105 |
| DEV target rate | 0.0793 | 0.1954 |
| OOT target rate | 0.0890 | 0.2329 |
| CV folds | 5 | 5 |
| LR feature budget | 20 | 20 |
| CatBoost feature budget | 40 | 40 |

Leakage reports were clean for both datasets:

| Dataset | Target excluded | Temporal split disjoint | Forbidden train features | Forbidden OOT features | OOT used in selection |
| --- | --- | --- | --- | --- | --- |
| Home Credit | True | True | none | none | False |
| LendingClub | True | True | none | none | False |

## Home Credit Results

Home Credit is the main development dataset. The strongest Home Credit result was CatBoost with `stable_core_llm_fill`, with OOT AUC 0.7683. The strongest LR result was also `stable_core_llm_fill`, with OOT AUC 0.7489.

### Home Credit Performance Table

| Model | Selector | CV AUC mean +/- SD | OOT AUC | OOT Gini | OOT KS | Lift@10 | Bad-rate capture@10 | Features | Runtime sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr | mrmr | 0.7362 +/- 0.0109 | 0.7457 | 0.4914 | 0.3618 | 3.0958 | 0.3096 | 20 | 728.4 |
| lr | boruta | 0.6416 +/- 0.0442 | 0.6306 | 0.2612 | 0.1874 | 2.0358 | 0.2036 | 20 | 1011.2 |
| lr | pca | 0.6568 +/- 0.0129 | 0.6729 | 0.3458 | 0.2597 | 2.2697 | 0.2270 | 20 | 17.6 |
| lr | domain_rule_baseline | 0.7136 +/- 0.0083 | 0.7249 | 0.4498 | 0.3346 | 2.8713 | 0.2871 | 20 | 17.9 |
| lr | llm | 0.7232 +/- 0.0135 | 0.7400 | 0.4799 | 0.3573 | 3.0322 | 0.3032 | 20 | 47.7 |
| lr | llm_then_mrmr | 0.7328 +/- 0.0117 | 0.7381 | 0.4762 | 0.3537 | 2.9508 | 0.2951 | 20 | 86.0 |
| lr | llm_then_boruta | 0.7197 +/- 0.0174 | 0.6488 | 0.2976 | 0.2151 | 2.1584 | 0.2158 | 20 | 248.9 |
| lr | stable_core_llm_fill | 0.7317 +/- 0.0144 | 0.7489 | 0.4977 | 0.3688 | 3.1033 | 0.3103 | 20 | 1475.4 |
| catboost | mrmr | 0.7518 +/- 0.0143 | 0.7668 | 0.5337 | 0.4017 | 3.4036 | 0.3404 | 40 | 960.2 |
| catboost | boruta | 0.7281 +/- 0.0235 | 0.6852 | 0.3704 | 0.2713 | 2.5045 | 0.2505 | 40 | 1239.4 |
| catboost | pca | 0.6940 +/- 0.0144 | 0.7042 | 0.4083 | 0.3022 | 2.6149 | 0.2615 | 40 | 588.4 |
| catboost | domain_rule_baseline | 0.7085 +/- 0.0101 | 0.7306 | 0.4612 | 0.3408 | 3.0359 | 0.3036 | 40 | 407.9 |
| catboost | llm | 0.7368 +/- 0.0194 | 0.7569 | 0.5137 | 0.3822 | 3.2904 | 0.3291 | 40 | 520.9 |
| catboost | llm_then_mrmr | 0.7501 +/- 0.0117 | 0.7630 | 0.5259 | 0.3929 | 3.3718 | 0.3372 | 40 | 725.9 |
| catboost | llm_then_boruta | 0.7431 +/- 0.0113 | 0.7592 | 0.5185 | 0.3872 | 3.3325 | 0.3333 | 40 | 812.1 |
| catboost | stable_core_llm_fill | 0.7480 +/- 0.0165 | 0.7683 | 0.5367 | 0.4022 | 3.4270 | 0.3427 | 40 | 4111.5 |

### Home Credit Stability And Drift Table

| Model | Selector | Nogueira | Kuncheva | Feature Jaccard | Semantic Jaccard | Feature PSI mean | Feature PSI max | Model PSI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr | mrmr | 0.7714 | 0.7714 | 0.6417 | 0.9333 | 0.0133 | 0.0774 | 0.0065 |
| lr | boruta | 0.5219 | 0.5219 | 0.4098 | 0.4532 | 0.0041 | 0.0119 | 0.0024 |
| lr | pca | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0360 | 0.1217 | 0.0291 |
| lr | domain_rule_baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0030 | 0.0207 | 0.0112 |
| lr | llm | 0.5479 | 0.5479 | 0.4076 | 0.5980 | 0.0015 | 0.0207 | 0.0066 |
| lr | llm_then_mrmr | 0.5531 | 0.5531 | 0.4043 | 0.7473 | 0.0118 | 0.0548 | 0.0038 |
| lr | llm_then_boruta | 0.4388 | 0.4388 | 0.3031 | 0.6655 | 0.0047 | 0.0123 | 0.0009 |
| lr | stable_core_llm_fill | 0.7454 | 0.7454 | 0.6110 | 0.9333 | 0.0125 | 0.0774 | 0.0050 |
| catboost | mrmr | 0.7296 | 0.7296 | 0.6037 | 0.9400 | 0.0187 | 0.1670 | 0.0102 |
| catboost | boruta | 0.5646 | 0.5646 | 0.4529 | 0.6315 | 0.0062 | 0.0223 | 0.0032 |
| catboost | pca | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0292 | 0.1217 | 0.0382 |
| catboost | domain_rule_baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0033 | 0.0207 | 0.0054 |
| catboost | llm | 0.4294 | 0.4294 | 0.3149 | 0.7473 | 0.0022 | 0.0207 | 0.0054 |
| catboost | llm_then_mrmr | 0.4077 | 0.4077 | 0.2966 | 0.8625 | 0.0108 | 0.0579 | 0.0083 |
| catboost | llm_then_boruta | 0.4402 | 0.4402 | 0.3247 | 0.7367 | 0.0133 | 0.0822 | 0.0094 |
| catboost | stable_core_llm_fill | 0.7025 | 0.7025 | 0.5725 | 0.8291 | 0.0121 | 0.0935 | 0.0076 |

### Home Credit Interpretation

`stable_core_llm_fill` is the best Home Credit selector by OOT AUC for both models. For LR, it improves over mRMR by 0.0032 AUC. For CatBoost, it improves over mRMR by 0.0015 AUC. These margins are directionally favorable but small, so the result should be described as a modest improvement rather than a decisive win.

mRMR remains a strong baseline. It has the highest CatBoost CV AUC mean at 0.7518 and high stability: Nogueira 0.7714 for LR and 0.7296 for CatBoost. The LLM methods are competitive, but pure `llm` is below mRMR for CatBoost OOT AUC: 0.7569 versus 0.7668.

Boruta is weak in this setup. It has the lowest OOT AUC for both LR and CatBoost and is also slow: 1011.2 seconds for LR and 1239.4 seconds for CatBoost. `llm_then_boruta` does not fix this weakness on LR and remains materially below mRMR and `stable_core_llm_fill`.

PCA and the domain-rule baseline have perfect measured feature-set stability because their selected structures are deterministic, but this does not translate into best predictive performance. PCA is particularly unattractive on Home Credit because its OOT AUC is low and model PSI is higher than most alternatives.

## LendingClub Results

LendingClub is the external validation dataset. The strongest LendingClub result was CatBoost with pure `llm`, with OOT AUC 0.7165. The strongest LR result was also pure `llm`, with OOT AUC 0.6982.

### LendingClub Performance Table

| Model | Selector | CV AUC mean +/- SD | OOT AUC | OOT Gini | OOT KS | Lift@10 | Bad-rate capture@10 | Features | Runtime sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr | mrmr | 0.7148 +/- 0.0081 | 0.6908 | 0.3816 | 0.2750 | 2.0998 | 0.2100 | 20 | 428.6 |
| lr | boruta | 0.6462 +/- 0.0047 | 0.6350 | 0.2700 | 0.1977 | 1.6931 | 0.1693 | 20 | 4143.0 |
| lr | pca | 0.7044 +/- 0.0084 | 0.6918 | 0.3835 | 0.2770 | 2.0954 | 0.2095 | 20 | 81.9 |
| lr | domain_rule_baseline | 0.6130 +/- 0.0031 | 0.6199 | 0.2399 | 0.1734 | 1.6130 | 0.1613 | 20 | 58.4 |
| lr | llm | 0.7121 +/- 0.0230 | 0.6982 | 0.3965 | 0.2830 | 2.1306 | 0.2131 | 20 | 353.2 |
| lr | llm_then_mrmr | 0.7161 +/- 0.0078 | 0.6940 | 0.3881 | 0.2780 | 2.1082 | 0.2108 | 20 | 273.6 |
| lr | llm_then_boruta | 0.6497 +/- 0.0035 | 0.6534 | 0.3069 | 0.2219 | 1.7696 | 0.1770 | 20 | 2110.3 |
| lr | stable_core_llm_fill | 0.7148 +/- 0.0079 | 0.6907 | 0.3815 | 0.2735 | 2.0976 | 0.2098 | 20 | 1705.7 |
| catboost | mrmr | 0.7244 +/- 0.0102 | 0.7060 | 0.4120 | 0.2953 | 2.1956 | 0.2196 | 40 | 2699.8 |
| catboost | boruta | 0.6691 +/- 0.0111 | 0.6604 | 0.3207 | 0.2305 | 1.8349 | 0.1835 | 40 | 5948.3 |
| catboost | pca | 0.7214 +/- 0.0129 | 0.6912 | 0.3824 | 0.2759 | 2.0645 | 0.2065 | 40 | 5198.8 |
| catboost | domain_rule_baseline | 0.6841 +/- 0.0105 | 0.6818 | 0.3636 | 0.2615 | 1.9924 | 0.1992 | 40 | 3444.0 |
| catboost | llm | 0.7247 +/- 0.0227 | 0.7165 | 0.4329 | 0.3103 | 2.2538 | 0.2254 | 40 | 3624.2 |
| catboost | llm_then_mrmr | 0.7244 +/- 0.0115 | 0.7071 | 0.4143 | 0.2973 | 2.2009 | 0.2201 | 40 | 3581.0 |
| catboost | llm_then_boruta | 0.6761 +/- 0.0132 | 0.6620 | 0.3240 | 0.2319 | 1.8418 | 0.1842 | 40 | 5162.4 |
| catboost | stable_core_llm_fill | 0.7251 +/- 0.0107 | 0.7069 | 0.4139 | 0.2969 | 2.1952 | 0.2195 | 40 | 7254.9 |

### LendingClub Stability And Drift Table

| Model | Selector | Nogueira | Kuncheva | Feature Jaccard | Semantic Jaccard | Feature PSI mean | Feature PSI max | Model PSI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr | mrmr | 0.7536 | 0.7536 | 0.6309 | 0.8667 | 0.0046 | 0.0279 | 0.0042 |
| lr | boruta | 0.7589 | 0.7589 | 0.6579 | 1.0000 | 0.0029 | 0.0279 | 0.0079 |
| lr | pca | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.3468 | 9.7747 | 0.0139 |
| lr | domain_rule_baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0012 | 0.0193 | 0.0242 |
| lr | llm | 0.5714 | 0.5714 | 0.4444 | 1.0000 | 0.0021 | 0.0357 | 0.0058 |
| lr | llm_then_mrmr | 0.6571 | 0.6571 | 0.5210 | 1.0000 | 0.0086 | 0.0331 | 0.0051 |
| lr | llm_then_boruta | 0.6196 | 0.6196 | 0.4925 | 0.8667 | 0.0076 | 0.0344 | 0.0078 |
| lr | stable_core_llm_fill | 0.7054 | 0.7054 | 0.5902 | 0.8667 | 0.0054 | 0.0279 | 0.0040 |
| catboost | mrmr | 0.7981 | 0.7981 | 0.7081 | 0.9000 | 0.0060 | 0.0331 | 0.0073 |
| catboost | boruta | 0.7029 | 0.7029 | 0.6179 | 0.7150 | 0.0039 | 0.0344 | 0.0038 |
| catboost | pca | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0802 | 9.7747 | 0.0279 |
| catboost | domain_rule_baseline | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0098 | 1.2304 | 0.0012 |
| catboost | llm | 0.7288 | 0.7288 | 0.6230 | 0.8000 | 0.0027 | 0.0357 | 0.0055 |
| catboost | llm_then_mrmr | 0.8096 | 0.8096 | 0.7197 | 0.8500 | 0.0061 | 0.0331 | 0.0071 |
| catboost | llm_then_boruta | 0.7288 | 0.7288 | 0.6421 | 0.8500 | 0.0044 | 0.0344 | 0.0029 |
| catboost | stable_core_llm_fill | 0.7779 | 0.7779 | 0.6834 | 1.0000 | 0.0063 | 0.0331 | 0.0074 |

### LendingClub Interpretation

Pure `llm` is the strongest selector on LendingClub for both models. For LR, it improves over mRMR by 0.0074 OOT AUC. For CatBoost, it improves over mRMR by 0.0105 OOT AUC. This is the strongest evidence that metadata screening can generalize beyond the development dataset.

The mRMR family remains competitive. `llm_then_mrmr`, `stable_core_llm_fill`, and mRMR are tightly grouped for CatBoost: 0.7071, 0.7069, and 0.7060 OOT AUC respectively. The LLM-only method wins, but the hybrid and mRMR variants are close enough that the conclusion should emphasize robustness rather than dominance.

PCA is problematic on LendingClub. Its OOT AUC is competitive for LR at 0.6918, but feature PSI is extremely high: mean 1.3468 for LR and 1.0802 for CatBoost, with max 9.7747 for both. This means PCA's perfect feature-set stability is not meaningful evidence of stable credit-risk behavior under temporal shift.

Boruta again underperforms and is expensive. LR Boruta OOT AUC is 0.6350 and CatBoost Boruta OOT AUC is 0.6604, with runtimes above 4,000 and 5,900 seconds respectively.

## Cross-Dataset Findings

Selector averages across the two model families show that the best selector family changes by dataset:

| Dataset | Selector | Mean OOT AUC across LR and CatBoost | Mean CV AUC | Mean Nogueira | Mean model PSI | Mean runtime sec |
| --- | --- | --- | --- | --- | --- | --- |
| Home Credit | stable_core_llm_fill | 0.7586 | 0.7398 | 0.7239 | 0.0063 | 2793.5 |
| Home Credit | mrmr | 0.7563 | 0.7440 | 0.7505 | 0.0083 | 844.3 |
| Home Credit | llm_then_mrmr | 0.7505 | 0.7415 | 0.4804 | 0.0060 | 406.0 |
| Home Credit | llm | 0.7484 | 0.7300 | 0.4886 | 0.0060 | 284.3 |
| Home Credit | domain_rule_baseline | 0.7278 | 0.7110 | 1.0000 | 0.0083 | 212.9 |
| Home Credit | llm_then_boruta | 0.7040 | 0.7314 | 0.4395 | 0.0051 | 530.5 |
| Home Credit | pca | 0.6885 | 0.6754 | 1.0000 | 0.0337 | 303.0 |
| Home Credit | boruta | 0.6579 | 0.6849 | 0.5433 | 0.0028 | 1125.3 |
| LendingClub | llm | 0.7073 | 0.7184 | 0.6501 | 0.0056 | 1988.7 |
| LendingClub | llm_then_mrmr | 0.7006 | 0.7203 | 0.7334 | 0.0061 | 1927.3 |
| LendingClub | stable_core_llm_fill | 0.6988 | 0.7200 | 0.7416 | 0.0057 | 4480.3 |
| LendingClub | mrmr | 0.6984 | 0.7196 | 0.7758 | 0.0058 | 1564.2 |
| LendingClub | pca | 0.6915 | 0.7129 | 1.0000 | 0.0209 | 2640.3 |
| LendingClub | llm_then_boruta | 0.6577 | 0.6629 | 0.6742 | 0.0054 | 3636.3 |
| LendingClub | domain_rule_baseline | 0.6509 | 0.6485 | 1.0000 | 0.0127 | 1751.2 |
| LendingClub | boruta | 0.6477 | 0.6576 | 0.7309 | 0.0058 | 5045.6 |

Main cross-dataset conclusions:

- CatBoost is the stronger evaluation vehicle in both datasets. The best CatBoost OOT AUC is 0.7683 on Home Credit and 0.7165 on LendingClub.
- LLM-assisted selectors occupy the top of both datasets, but the winning variant differs. Home Credit favors `stable_core_llm_fill`; LendingClub favors pure `llm`.
- mRMR remains the most credible non-LLM statistical baseline. It is near the top on both datasets and has stronger feature-set stability than pure `llm` on several runs.
- Boruta is consistently weak in this implementation. It underperforms on both datasets and has poor runtime efficiency.
- PCA's deterministic stability is misleading. It has perfect feature-set stability by construction, but materially weaker OOT performance and severe LendingClub feature PSI.
- The domain-rule baseline is stable and interpretable, but not competitive with LLM/mRMR methods on OOT AUC.

## LLM Call And Cache Behavior

The run used shared LLM ranking with a configured ranking budget of 100. The manifest-level LLM counters show substantial cache reuse:

| Dataset | LLM calls actually made | LLM cache hits | Total recorded LLM tokens |
| --- | --- | --- | --- |
| Home Credit | 0 | 48 | 0 |
| LendingClub | 6 | 42 | 66020 |

Home Credit LLM-related runs were served fully from cache. LendingClub made 6 actual LLM calls and then reused cached rankings across related selectors and model configurations. This supports the practical claim that shared metadata ranking can control LLM cost across a larger matrix.

## Publishability Verdict

The current evidence is publishable if the claim is framed conservatively:

> LLM-based metadata screening is a useful first-stage feature-selection helper for credit-risk modeling. It can produce competitive or leading OOT performance, especially on external validation, while retaining low drift and acceptable stability. However, it should be treated as a complement to statistical selection rather than a universal replacement.

The strongest result for the paper is the cross-dataset pattern: LLM-assisted methods are among the best selectors on both Home Credit and LendingClub. The strongest caveat is that mRMR remains very competitive, especially on Home Credit, and pure LLM is not always the best Home Credit method.

The aggregate reporting scripts now populate `final_comparison_table.csv`, `feature_stability_table.csv`, `feature_drift_table.csv`, `llm_call_summary.csv`, `paired_fold_comparisons.csv`, and `semantic_coverage_table.csv` for both datasets. This removes the earlier reproducibility weakness where the per-run files were complete but the top-level summaries were empty.

## Suggested Paper Wording

Across the Home Credit development dataset and the LendingClub external-validation dataset, LLM-assisted feature selection produced competitive OOT discrimination with compact feature sets. On Home Credit, the stability-constrained `stable_core_llm_fill` variant achieved the best OOT AUC for both Logistic Regression and CatBoost, reaching 0.7489 and 0.7683 respectively. On LendingClub, pure LLM metadata screening achieved the best OOT AUC for both Logistic Regression and CatBoost, reaching 0.6982 and 0.7165 respectively. These gains should be interpreted conservatively: mRMR remained a strong statistical baseline, especially on Home Credit, and LLM selection did not uniformly dominate every comparator. The results support the use of LLM metadata screening as a first-stage feature-selection aid, particularly when combined with stability-aware selection or external validation, rather than as a standalone replacement for statistical feature selection.
