# Prompt 11 baseline results audit

## Objective

Authenticate and audit the completed Prompt 10 individual-selector baseline evidence, then assess the preregistered combination families structurally without refitting a baseline or opening a raw research dataset.

## Completed evidence and authentication

Fact: exactly **36/36** frozen cells authenticated, including `fbv1-036-lendingclub_v2-catboost-rfe-catboost-s42`. The phase composition is 180 expanding-window DEV fold metric units, 36 full-DEV in-sample diagnostics, and 36 locked OOT evaluations.

Fact: all **396** supported metric reconciliation checks passed; discrepancies: **0**. No raw dataset path was resolved and no baseline was refit.

## Methodology

Saved prediction identity, target encoding, finiteness, probability range, ordering, file size, and SHA-256 bindings were checked before metric recomputation. OOT comparisons use aligned predictions, paired DeLong inference, 2,000 paired target-stratified bootstrap resamples (seed 20260721; percentile 95% interval), and Holm correction within eight named dataset/model/reference families of seven comparisons each.

## DEV robustness

Fact: 36 method/dataset/model summaries preserve all five expanding folds in temporal order. Fold predictions were not persisted, so no fold pooling or paired fold inference was attempted. Full fold-level means, standard deviations, ranges, counts, and selected-feature counts are in `baseline_dev_fold_summary.csv`.

## OOT evidence

Fact: 36 authenticated locked-OOT evaluation rows are available. OOT is treated as final predictive evidence, while the saved full-DEV prediction rows are explicitly labelled in-sample diagnostics.

## Paired comparisons

Fact: 56 preregistered comparisons were evaluated across 8 Holm families; 56 comparisons survived Holm and 56 received a strong or moderate scoped evidence label. No all-pairs search was run.

Scoped strong/moderate findings (method A minus reference):

| Dataset | Model | A | Reference | Delta AUC | 95% CI | Holm p | Strength |
|---|---|---|---|---:|---:|---:|---|
| homecredit | catboost | boruta_random_forest | full_features | -0.031830 | [-0.034504, -0.028978] | 0.000000 | strong |
| homecredit | catboost | catboost_shap | full_features | -0.006497 | [-0.008260, -0.004807] | 0.000000 | strong |
| homecredit | catboost | iv_woe | full_features | -0.017461 | [-0.019383, -0.015390] | 0.000000 | strong |
| homecredit | catboost | lasso_l1_logistic | full_features | -0.011806 | [-0.013585, -0.009976] | 0.000000 | strong |
| homecredit | catboost | legacy_rf_relevance_corr | full_features | -0.016227 | [-0.018139, -0.014354] | 0.000000 | strong |
| homecredit | catboost | mrmr_mutual_information | full_features | -0.026133 | [-0.028649, -0.023636] | 0.000000 | strong |
| homecredit | catboost | rfe_catboost | full_features | -0.004606 | [-0.006054, -0.003231] | 0.000000 | strong |
| homecredit | catboost | boruta_random_forest | random_k | 0.068809 | [0.063651, 0.073813] | 0.000000 | strong |
| homecredit | catboost | catboost_shap | random_k | 0.094142 | [0.089416, 0.099140] | 0.000000 | strong |
| homecredit | catboost | iv_woe | random_k | 0.083177 | [0.078613, 0.088041] | 0.000000 | strong |
| homecredit | catboost | lasso_l1_logistic | random_k | 0.088833 | [0.084236, 0.093648] | 0.000000 | strong |
| homecredit | catboost | legacy_rf_relevance_corr | random_k | 0.084411 | [0.079775, 0.089299] | 0.000000 | strong |
| homecredit | catboost | mrmr_mutual_information | random_k | 0.074506 | [0.069571, 0.079647] | 0.000000 | strong |
| homecredit | catboost | rfe_catboost | random_k | 0.096032 | [0.091543, 0.100797] | 0.000000 | strong |
| homecredit | lr | boruta_random_forest | full_features | -0.031758 | [-0.035148, -0.028345] | 0.000000 | strong |
| homecredit | lr | catboost_shap | full_features | -0.021366 | [-0.024410, -0.018328] | 0.000000 | strong |
| homecredit | lr | iv_woe | full_features | -0.033349 | [-0.036679, -0.029904] | 0.000000 | strong |
| homecredit | lr | lasso_l1_logistic | full_features | -0.026146 | [-0.029319, -0.022850] | 0.000000 | strong |
| homecredit | lr | legacy_rf_relevance_corr | full_features | -0.026744 | [-0.029935, -0.023493] | 0.000000 | strong |
| homecredit | lr | mrmr_mutual_information | full_features | -0.036380 | [-0.039832, -0.033078] | 0.000000 | strong |
| homecredit | lr | rfe_catboost | full_features | -0.024950 | [-0.028275, -0.021738] | 0.000000 | strong |
| homecredit | lr | boruta_random_forest | random_k | 0.111914 | [0.106488, 0.117399] | 0.000000 | strong |
| homecredit | lr | catboost_shap | random_k | 0.122306 | [0.116537, 0.128350] | 0.000000 | strong |
| homecredit | lr | iv_woe | random_k | 0.110323 | [0.104511, 0.116089] | 0.000000 | strong |
| homecredit | lr | lasso_l1_logistic | random_k | 0.117526 | [0.111782, 0.123409] | 0.000000 | strong |
| homecredit | lr | legacy_rf_relevance_corr | random_k | 0.116928 | [0.111324, 0.122632] | 0.000000 | strong |
| homecredit | lr | mrmr_mutual_information | random_k | 0.107292 | [0.101421, 0.113016] | 0.000000 | strong |
| homecredit | lr | rfe_catboost | random_k | 0.118722 | [0.112970, 0.124459] | 0.000000 | strong |
| lendingclub_v2 | catboost | boruta_random_forest | full_features | -0.037226 | [-0.038793, -0.035675] | 0.000000 | strong |
| lendingclub_v2 | catboost | catboost_shap | full_features | -0.007824 | [-0.008380, -0.007218] | 0.000000 | strong |
| lendingclub_v2 | catboost | iv_woe | full_features | -0.020301 | [-0.021162, -0.019431] | 0.000000 | strong |
| lendingclub_v2 | catboost | lasso_l1_logistic | full_features | -0.013346 | [-0.014139, -0.012566] | 0.000000 | strong |
| lendingclub_v2 | catboost | legacy_rf_relevance_corr | full_features | -0.015780 | [-0.016539, -0.015024] | 0.000000 | strong |
| lendingclub_v2 | catboost | mrmr_mutual_information | full_features | -0.020950 | [-0.021953, -0.019901] | 0.000000 | strong |
| lendingclub_v2 | catboost | rfe_catboost | full_features | -0.004012 | [-0.004499, -0.003520] | 0.000000 | strong |
| lendingclub_v2 | catboost | boruta_random_forest | random_k | 0.013901 | [0.012296, 0.015511] | 0.000000 | strong |
| lendingclub_v2 | catboost | catboost_shap | random_k | 0.043303 | [0.041434, 0.045124] | 0.000000 | strong |
| lendingclub_v2 | catboost | iv_woe | random_k | 0.030827 | [0.028941, 0.032760] | 0.000000 | strong |
| lendingclub_v2 | catboost | lasso_l1_logistic | random_k | 0.037781 | [0.035858, 0.039686] | 0.000000 | strong |
| lendingclub_v2 | catboost | legacy_rf_relevance_corr | random_k | 0.035348 | [0.033493, 0.037252] | 0.000000 | strong |
| lendingclub_v2 | catboost | mrmr_mutual_information | random_k | 0.030178 | [0.028306, 0.032011] | 0.000000 | strong |
| lendingclub_v2 | catboost | rfe_catboost | random_k | 0.047116 | [0.045330, 0.048956] | 0.000000 | strong |
| lendingclub_v2 | lr | boruta_random_forest | full_features | -0.055140 | [-0.057160, -0.052881] | 0.000000 | strong |
| lendingclub_v2 | lr | catboost_shap | full_features | -0.014345 | [-0.015499, -0.013124] | 0.000000 | strong |
| lendingclub_v2 | lr | iv_woe | full_features | -0.028777 | [-0.030145, -0.027336] | 0.000000 | strong |
| lendingclub_v2 | lr | lasso_l1_logistic | full_features | -0.029784 | [-0.031266, -0.028229] | 0.000000 | strong |
| lendingclub_v2 | lr | legacy_rf_relevance_corr | full_features | -0.019342 | [-0.020620, -0.018000] | 0.000000 | strong |
| lendingclub_v2 | lr | mrmr_mutual_information | full_features | -0.024964 | [-0.026353, -0.023558] | 0.000000 | strong |
| lendingclub_v2 | lr | rfe_catboost | full_features | -0.012007 | [-0.013146, -0.010842] | 0.000000 | strong |
| lendingclub_v2 | lr | boruta_random_forest | random_k | 0.051926 | [0.049664, 0.054243] | 0.000000 | strong |
| lendingclub_v2 | lr | catboost_shap | random_k | 0.092720 | [0.090049, 0.095410] | 0.000000 | strong |
| lendingclub_v2 | lr | iv_woe | random_k | 0.078288 | [0.075585, 0.080982] | 0.000000 | strong |
| lendingclub_v2 | lr | lasso_l1_logistic | random_k | 0.077282 | [0.074476, 0.080076] | 0.000000 | strong |
| lendingclub_v2 | lr | legacy_rf_relevance_corr | random_k | 0.087724 | [0.085128, 0.090284] | 0.000000 | strong |
| lendingclub_v2 | lr | mrmr_mutual_information | random_k | 0.082102 | [0.079400, 0.084617] | 0.000000 | strong |
| lendingclub_v2 | lr | rfe_catboost | random_k | 0.095059 | [0.092474, 0.097648] | 0.000000 | strong |

## Stability and drift

Fact: selection stability was recomputed for 36 configurations. 1 have varying fold subset sizes; Kuncheva is marked not applicable for those cases, while Jaccard and Nogueira-style results remain separately reported with their applicability labels.

Fact: 36 authenticated score-PSI summaries and 5595 saved feature-PSI rows were audited. All 36 score-PSI values are below the descriptive 0.10 threshold (range 0.0006 to 0.0695); 0.10 and 0.25 remain descriptive references, not hypothesis-test cutoffs. Score PSI was not recomputed because the saved artifacts do not preserve the frozen DEV bin edges; raw feature matrices were not reopened.

## Runtime and resources

Fact: 36 completed cell resource summaries separate active computation from RAM waiting. Peak process-tree RSS ranged from 7.27 to 28.35 GiB; recorded RAM waiting totaled 1.12 hours.

Interpretation: these measurements are scheduling and feasibility evidence, not predictive evidence. Heavy-chain workload classification must reflect every material selector stage and the final model.

## Combination feasibility implications

Fact: 80 fold/configuration feasibility rows cover only IV→Boruta, Boruta→CatBoost RFE, Boruta→canonical MI-mRMR, and the exact five-voter normalized-average-rank method. Prompt 10 baseline components are not silently reused because their saved identities do not authenticate the required chained training universe or complete five-voter rank bundle.

Recommendation: run the committed bounded 24-evaluation pilot (18 unique selector fits) unchanged. Review support, correctness, fit count, runtime, and RSS before creating the separate approval lock; do not tune from pilot predictive outcomes.

## Limitations

- Fold-level prediction vectors were not persisted, so fold metric reconciliation and paired fold inference are unsupported.
- Saved full-DEV predictions are in-sample diagnostics, not out-of-fold predictions.
- Score-PSI bin edges were not persisted, so score PSI is authenticated but not independently recomputed.
- Prompt 10 budget-capped Boruta artifacts prove shortfall when fewer than k features were retained, but reaching k does not reveal uncapped natural-support size.
- Random-k has one frozen seed in the completed matrix; no replicate distribution exists and no favorable seed was selected.
- Non-significance does not establish equivalence; natural-support/count mismatches remain explicit caveats.

## Conclusion

The completed baseline evidence is internally authenticated and all supported saved-prediction metrics reconcile. Conclusions remain scoped by dataset, final model, split, uncertainty, temporal robustness, selected-count semantics, and resource cost. The preregistered combinations remain hypotheses pending the real bounded pilot.
