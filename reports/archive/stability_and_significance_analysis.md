# Stability And Significance Analysis

This report uses existing post-run artifacts only. It does not rerun the experiment matrix, retrain models, or rebuild datasets.

## LLM Stability Diagnosis

- `catboost/llm`: LendingClub has higher exact stability than Home Credit (Nogueira 0.7288 vs 0.4294; feature Jaccard 0.6230 vs 0.3149).
  Stored selected-rank stability is higher on LendingClub (0.5590 vs 0.1631); IV rank stability is 0.9973 vs 0.9548.
- `lr/llm`: LendingClub has higher exact stability than Home Credit (Nogueira 0.5714 vs 0.5479; feature Jaccard 0.4444 vs 0.4076).
  Stored selected-rank stability is higher on LendingClub (0.5590 vs 0.1631); IV rank stability is 0.9973 vs 0.9548.

The difference is not just a reporting artifact: exact selected-feature overlap and fold selection frequencies are higher on LendingClub. The stored selected-rank stability and IV rank stability are also stronger on LendingClub in these artifacts. Larger DEV sample size on LendingClub likely contributes to more stable fold rankings, but LendingClub also has larger target-rate drift, so target drift alone does not explain the higher exact stability.

## Paired Fold AUC Tests

Positive OOT AUC deltas among requested comparisons:
- `homecredit/catboost/stable_core_llm_fill` OOT AUC delta vs mRMR is 0.0015; paired folds significantly favor mRMR, so this OOT gain is not fold-supported.
- `homecredit/lr/stable_core_llm_fill` OOT AUC delta vs mRMR is 0.0032; paired folds do not support significance.
- `lendingclub/catboost/stable_core_llm_fill` OOT AUC delta vs mRMR is 0.0010; paired folds do not support significance.
- `lendingclub/catboost/llm` OOT AUC delta vs mRMR is 0.0105; paired folds do not support significance.
- `lendingclub/catboost/llm_then_mrmr` OOT AUC delta vs mRMR is 0.0012; paired folds do not support significance.
- `lendingclub/lr/llm` OOT AUC delta vs mRMR is 0.0074; paired folds do not support significance.
- `lendingclub/lr/llm_then_mrmr` OOT AUC delta vs mRMR is 0.0032; paired folds do not support significance.

None of the positive OOT AUC deltas should be treated as a strong gain from the paired fold evidence; they are either tiny or not significant across folds.

Significant fold-level AUC differences at alpha 0.05. These are significant by the paired t-test; with only five folds, the Wilcoxon test is more conservative and may not cross 0.05.
- `homecredit/catboost/stable_core_llm_fill` vs mRMR: mean delta -0.0038, t-test p=0.0425, Wilcoxon p=0.0625.
- `homecredit/catboost/llm` vs mRMR: mean delta -0.0150, t-test p=0.0062, Wilcoxon p=0.0625.
- `homecredit/lr/llm` vs mRMR: mean delta -0.0129, t-test p=0.0133, Wilcoxon p=0.0625.

Small deltas that should not be treated strongly:
- `homecredit/catboost/llm_then_mrmr` vs mRMR: mean AUC delta -0.0017.
- `lendingclub/catboost/stable_core_llm_fill` vs mRMR: mean AUC delta 0.0007.
- `lendingclub/catboost/llm` vs mRMR: mean AUC delta 0.0003.
- `lendingclub/catboost/llm_then_mrmr` vs mRMR: mean AUC delta 0.0000.
- `lendingclub/lr/stable_core_llm_fill` vs mRMR: mean AUC delta 0.0001.
- `lendingclub/lr/llm_then_mrmr` vs mRMR: mean AUC delta 0.0014.

## Stability Interpretation

mRMR still dominates exact stability on Home Credit for the strongest statistical baselines, while LendingClub shows much higher LLM exact stability than Home Credit. The LLM behavior is dataset-dependent: LendingClub has fewer candidate features and much larger DEV folds, and the LLM fold rankings are more consistent there.
OOT or fold AUC gains should be described cautiously. The paired fold tests show that many deltas are small, and significance depends on dataset/model/selector rather than being a blanket LLM-family effect.

## Plots

Created plots: `stability_vs_oot_auc_by_dataset_model.png`, `fold_level_auc_differences_key_comparisons.png`, `llm_fold_selection_frequency_by_dataset.png`.
Skipped plots: none.

## Missing Artifacts

- None for the requested diagnosis rows and paired fold AUC tests.

## Rerun Requirement

A full rerun is not required for this analysis. The only caveat is that stronger attribution of rejected LLM candidates would require artifacts that store fold-level metric effects or PSI/IV deltas for rejected candidates, but that is outside the requested no-rerun scope.

## Files Created

- `results\cross_dataset\analysis\stability_significance\llm_stability_diagnosis.csv`
- `results\cross_dataset\analysis\stability_significance\paired_fold_significance_tests.csv`
- `results\cross_dataset\analysis\stability_significance\plots\plot_manifest.csv`
- `reports\stability_and_significance_analysis.md`
