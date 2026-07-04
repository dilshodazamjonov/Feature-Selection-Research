# Paired Fold Significance and Consistency Analysis

## Objective

This analysis tests whether three LLM-family feature-selection pipelines show consistent validation ROC AUC differences from the statistical mRMR baseline across the five saved cross-validation folds. It covers Home Credit and LendingClub v2 with Logistic Regression and CatBoost, yielding 12 planned primary comparisons. No training, feature selection, prediction generation, preprocessing, dataset construction, or CLIP work was rerun.

## Runs and sources

| Dataset | Model | Pipeline | Run ID | Fold source | Pairability |
|---|---|---|---|---|---|
| Home Credit | Logistic Regression | mRMR | `lr_statistical_mrmr_53a793cb32fe` | `results/homecredit/lr/statistical/lr_statistical_mrmr_53a793cb32fe/results/cv_results.csv` | authenticated |
| Home Credit | Logistic Regression | Stable-core + LLM-fill | `lr_hybrid_stable_core_llm_fill_1ddd0142e614` | `results/homecredit/lr/hybrid_stable_core_llm_fill/lr_hybrid_stable_core_llm_fill_1ddd0142e614/results/cv_results.csv` | authenticated |
| Home Credit | Logistic Regression | pure LLM | `lr_llm_llm_66fabfd650a1` | `results/homecredit/lr/llm/lr_llm_llm_66fabfd650a1/results/cv_results.csv` | authenticated |
| Home Credit | Logistic Regression | LLM-then-mRMR | `lr_hybrid_llm_then_mrmr_f69e1a0cffc2` | `results/homecredit/lr/hybrid_mrmr/lr_hybrid_llm_then_mrmr_f69e1a0cffc2/results/cv_results.csv` | authenticated |
| Home Credit | CatBoost | mRMR | `catboost_statistical_mrmr_3858b721e537` | `results/homecredit/catboost/statistical/catboost_statistical_mrmr_3858b721e537/results/cv_results.csv` | authenticated |
| Home Credit | CatBoost | Stable-core + LLM-fill | `catboost_hybrid_stable_core_llm_fill_8993eae5a4f7` | `results/homecredit/catboost/hybrid_stable_core_llm_fill/catboost_hybrid_stable_core_llm_fill_8993eae5a4f7/results/cv_results.csv` | authenticated |
| Home Credit | CatBoost | pure LLM | `catboost_llm_llm_d54c966a1d6e` | `results/homecredit/catboost/llm/catboost_llm_llm_d54c966a1d6e/results/cv_results.csv` | authenticated |
| Home Credit | CatBoost | LLM-then-mRMR | `catboost_hybrid_llm_then_mrmr_87fbcccf4952` | `results/homecredit/catboost/hybrid_mrmr/catboost_hybrid_llm_then_mrmr_87fbcccf4952/results/cv_results.csv` | authenticated |
| LendingClub v2 | Logistic Regression | mRMR | `lr_statistical_mrmr_c30cd6aff377` | `results/lendingclub_v2/lr/statistical/lr_statistical_mrmr_c30cd6aff377/results/cv_results.csv` | authenticated |
| LendingClub v2 | Logistic Regression | Stable-core + LLM-fill | `lr_hybrid_stable_core_llm_fill_497f694ad76d` | `results/lendingclub_v2/lr/hybrid_stable_core_llm_fill/lr_hybrid_stable_core_llm_fill_497f694ad76d/results/cv_results.csv` | authenticated |
| LendingClub v2 | Logistic Regression | pure LLM | `lr_llm_llm_bb103e2ac012` | `results/lendingclub_v2/lr/llm/lr_llm_llm_bb103e2ac012/results/cv_results.csv` | authenticated |
| LendingClub v2 | Logistic Regression | LLM-then-mRMR | `lr_hybrid_llm_then_mrmr_45d98f9ad95c` | `results/lendingclub_v2/lr/hybrid_mrmr/lr_hybrid_llm_then_mrmr_45d98f9ad95c/results/cv_results.csv` | authenticated |
| LendingClub v2 | CatBoost | mRMR | `catboost_statistical_mrmr_94a6a14a53a4` | `results/lendingclub_v2/catboost/statistical/catboost_statistical_mrmr_94a6a14a53a4/results/cv_results.csv` | authenticated |
| LendingClub v2 | CatBoost | Stable-core + LLM-fill | `catboost_hybrid_stable_core_llm_fill_8da7f3b51c4a` | `results/lendingclub_v2/catboost/hybrid_stable_core_llm_fill/catboost_hybrid_stable_core_llm_fill_8da7f3b51c4a/results/cv_results.csv` | authenticated |
| LendingClub v2 | CatBoost | pure LLM | `catboost_llm_llm_e6489647a93c` | `results/lendingclub_v2/catboost/llm/catboost_llm_llm_e6489647a93c/results/cv_results.csv` | authenticated |
| LendingClub v2 | CatBoost | LLM-then-mRMR | `catboost_hybrid_llm_then_mrmr_59865aa71763` | `results/lendingclub_v2/catboost/hybrid_mrmr/catboost_hybrid_llm_then_mrmr_59865aa71763/results/cv_results.csv` | authenticated |

The objective requires 16 canonical runs: four pipelines in each of four dataset/model strata. This produces 80 master fold rows. The prompt's separate estimate of eight runs and 40 rows is arithmetically inconsistent with its 12-comparison design, so the complete design takes precedence.

## Method

For every run, exactly five numeric fold rows were extracted from `cv_results.csv`; aggregate `mean` and `std` rows were excluded. The `auc` field was accepted only after its five-fold mean matched `run_manifest.json` and each stored Gini value satisfied `gini = 2 × auc − 1`. Fold identity is a SHA-256 digest of dataset and data-manifest identity, target contract, split contract, fold number, validation index bounds, validation time bounds, and validation row count. Every candidate fold identity had to equal its mRMR counterpart.

Differences are candidate AUC minus mRMR AUC. Zero differences are removed before ranking; tied absolute differences receive average ranks. The two-sided Wilcoxon p-value is calculated by exhaustively enumerating all sign assignments for the non-zero ranks. The rank-biserial correlation is `(W+ − W−) / (W+ + W−)`. Holm correction is applied across the 12 comparisons actually computed. With only five folds, inference is low-powered and descriptive fold direction is more informative than a binary threshold.

## Results

| Dataset | Model | Pipeline vs mRMR | Mean Δ | Median Δ | Wins-losses-ties | Raw p | Holm p | Rank-biserial | Consistency | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| Home Credit | Logistic Regression | Stable-core + LLM-fill | -0.004501 | -0.003955 | 0-5-0 | 0.0625 | 0.7500 | -1.000 | strong | COMPUTED |
| Home Credit | Logistic Regression | pure LLM | -0.012932 | -0.015090 | 0-5-0 | 0.0625 | 0.7500 | -1.000 | strong | COMPUTED |
| Home Credit | Logistic Regression | LLM-then-mRMR | -0.003316 | -0.003821 | 1-4-0 | 0.1875 | 0.7500 | -0.733 | moderate | COMPUTED |
| Home Credit | CatBoost | Stable-core + LLM-fill | -0.003798 | -0.002713 | 0-5-0 | 0.0625 | 0.7500 | -1.000 | strong | COMPUTED |
| Home Credit | CatBoost | pure LLM | -0.014957 | -0.014115 | 0-5-0 | 0.0625 | 0.7500 | -1.000 | strong | COMPUTED |
| Home Credit | CatBoost | LLM-then-mRMR | -0.001682 | -0.002981 | 2-3-0 | 0.6250 | 1.0000 | -0.333 | weak | COMPUTED |
| LendingClub v2 | Logistic Regression | Stable-core + LLM-fill | 0.000222 | -0.000568 | 2-3-0 | 1.0000 | 1.0000 | -0.067 | weak | COMPUTED |
| LendingClub v2 | Logistic Regression | pure LLM | 0.004775 | 0.005326 | 4-1-0 | 0.1250 | 0.7500 | 0.867 | moderate | COMPUTED |
| LendingClub v2 | Logistic Regression | LLM-then-mRMR | 0.001594 | 0.001686 | 5-0-0 | 0.0625 | 0.7500 | 1.000 | strong | COMPUTED |
| LendingClub v2 | CatBoost | Stable-core + LLM-fill | 0.001803 | 0.001598 | 5-0-0 | 0.0625 | 0.7500 | 1.000 | strong | COMPUTED |
| LendingClub v2 | CatBoost | pure LLM | 0.009000 | 0.009032 | 5-0-0 | 0.0625 | 0.7500 | 1.000 | strong | COMPUTED |
| LendingClub v2 | CatBoost | LLM-then-mRMR | 0.001414 | 0.001492 | 3-2-0 | 0.3125 | 0.9375 | 0.600 | weak | COMPUTED |

## Interpretation by dataset

### Home Credit

**Logistic Regression.**
Stable-core + LLM-fill had mean fold ΔAUC -0.004501, median -0.003955, and wins-losses-ties 0-5-0. Its direction was not aligned with the authenticated OOT ΔAUC (+0.003168).
pure LLM had mean fold ΔAUC -0.012932, median -0.015090, and wins-losses-ties 0-5-0. Its direction was aligned with the authenticated OOT ΔAUC (-0.005733).
LLM-then-mRMR had mean fold ΔAUC -0.003316, median -0.003821, and wins-losses-ties 1-4-0. Its direction was aligned with the authenticated OOT ΔAUC (-0.007590).

**CatBoost.**
Stable-core + LLM-fill had mean fold ΔAUC -0.003798, median -0.002713, and wins-losses-ties 0-5-0. Its direction was not aligned with the authenticated OOT ΔAUC (+0.001504).
pure LLM had mean fold ΔAUC -0.014957, median -0.014115, and wins-losses-ties 0-5-0. Its direction was aligned with the authenticated OOT ΔAUC (-0.009985).
LLM-then-mRMR had mean fold ΔAUC -0.001682, median -0.002981, and wins-losses-ties 2-3-0. Its direction was aligned with the authenticated OOT ΔAUC (-0.003885).

### LendingClub v2

**Logistic Regression.**
Stable-core + LLM-fill had mean fold ΔAUC +0.000222, median -0.000568, and wins-losses-ties 2-3-0. Its direction was not aligned with the authenticated OOT ΔAUC (-0.004587).
pure LLM had mean fold ΔAUC +0.004775, median +0.005326, and wins-losses-ties 4-1-0. Its direction was aligned with the authenticated OOT ΔAUC (+0.004026).
LLM-then-mRMR had mean fold ΔAUC +0.001594, median +0.001686, and wins-losses-ties 5-0-0. Its direction was aligned with the authenticated OOT ΔAUC (+0.002073).

**CatBoost.**
Stable-core + LLM-fill had mean fold ΔAUC +0.001803, median +0.001598, and wins-losses-ties 5-0-0. Its direction was aligned with the authenticated OOT ΔAUC (+0.001996).
pure LLM had mean fold ΔAUC +0.009000, median +0.009032, and wins-losses-ties 5-0-0. Its direction was aligned with the authenticated OOT ΔAUC (+0.012889).
LLM-then-mRMR had mean fold ΔAUC +0.001414, median +0.001492, and wins-losses-ties 3-2-0. Its direction was aligned with the authenticated OOT ΔAUC (+0.003305).

## Scientific interpretation

The LLM-family methods were not uniformly stronger than mRMR; direction depended on dataset, downstream model, and selector. Strong same-direction fold consistency occurred for Home Credit/Logistic Regression/Stable-core + LLM-fill, Home Credit/Logistic Regression/pure LLM, Home Credit/CatBoost/Stable-core + LLM-fill, Home Credit/CatBoost/pure LLM, LendingClub v2/Logistic Regression/LLM-then-mRMR, LendingClub v2/CatBoost/Stable-core + LLM-fill, LendingClub v2/CatBoost/pure LLM. 0 comparisons crossed raw p < 0.05 and 0 crossed Holm-adjusted p < 0.05. With five non-zero differences, even a perfect five-fold direction normally yields a minimum exact two-sided p-value of 0.0625. Non-significance therefore reflects both the observed data and the low-power design and cannot establish equivalence. Fold and OOT directions were assessed separately; discordance is retained rather than reconciled away. These tests describe consistency under the saved CV design and do not replace authenticated OOT rankings.

## Limitations

1. Five folds provide very low inferential power.
2. With five non-zero paired differences, the smallest attainable exact two-sided Wilcoxon p-value is normally 0.0625.
3. Consequently, no comparison with only five non-zero folds can normally cross a two-sided 0.05 threshold.
4. A non-significant result does not prove that two pipelines are equivalent.
5. Cross-validation folds are not fully independent scientific replications because their training samples may overlap.
6. The tests evaluate fold-level consistency under the saved CV design; they do not test the final OOT AUC difference.
7. OOT conclusions remain based on authenticated OOT metrics and are not replaced by these CV results.
8. Effect direction, fold wins, median difference, and observed variability are more informative here than a binary significance label.

## Conclusion

All 12 of 12 planned comparisons were authenticated and computed from direct fold evidence. The results support restrained claims about directional consistency, not definitive superiority or equivalence. The observed ranges and rank-biserial values should be read alongside OOT metrics and the limitations of five overlapping CV training samples.

## Reproducibility outputs

- `results/finalized_research/pending_analyses/significance_tests/fold_auc_master.csv`
- `results/finalized_research/pending_analyses/significance_tests/paired_difference_details.csv`
- `results/finalized_research/pending_analyses/significance_tests/paired_significance_results.csv`
- `results/finalized_research/pending_analyses/significance_tests/figures/paired_auc_difference_forest.png`
- `results/finalized_research/pending_analyses/significance_tests/significance_summary.md`
- `results/finalized_research/pending_analyses/significance_tests/significance_manifest.json`
- `results/finalized_research/pending_analyses/significance_tests/run_significance_analysis.py`
