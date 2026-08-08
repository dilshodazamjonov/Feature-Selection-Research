# Two-dataset locked-OOT statistical review

## Technical summary

All 124 preregistered comparisons were evaluable on authenticated saved predictions and were retained across all 36 complete Holm families. Effects were heterogeneous: positive combination contrasts coexist with adverse and null contrasts, so the evidence does not support a universal winner or a pooled cross-dataset effect. Statistical significance is interpreted with effect magnitude, uncertainty, budget comparability, stability, drift, and resource evidence.

## Objective

Assess the four frozen combination methods on the locked Home Credit and LendingClub v2 OOT populations, using DEV only as supporting evidence for dispersion, generalization, stability, and drift. No model, selector, transformation, or prediction workload was run.

## Authenticated datasets and artifact scope

The authenticated chain contains 24/24 pilot evaluations and 18/18 pilot selector fits; 120/120 DEV evaluations and 90/90 DEV selector fits; 24/24 combination OOT evaluations, 18/18 full-DEV selector refits, and 168/168 OOT active files. The pointer-selected voting successor authenticated 55/55 payload entries, with 54/54 unaffected entries byte-identical. The stale original manifest was historical only.

## Locked methods, comparisons, and multiplicity families

The canonical registry contains 124 comparisons in 36 families: 56 baseline/full-random contrasts and 68 combination contrasts. Every row uses method minus registered reference. Paired DeLong provides the AUC p-value; a 2,000-draw target-stratified paired bootstrap (seed 20260721) provides the 95% interval. Holm adjustment is within the complete original family.

## Locked-OOT predictive results

The largest observed positive combination effects are shown below. These are scoped contrasts, not a leaderboard and not proof of a globally best method.

| Dataset | Model | Contrast | ΔAUC | 95% CI | Holm p | Grade |
|---|---|---|---:|---:|---:|---|
| lendingclub_v2 | lr | iv_then_boruta pool300 vs boruta_random_forest | 0.054056 | [0.051817, 0.056085] | 0 | strong |
| lendingclub_v2 | lr | iv_then_boruta pool200 vs boruta_random_forest | 0.050676 | [0.048518, 0.052666] | 0 | strong |
| lendingclub_v2 | lr | iv_then_boruta pool100 vs boruta_random_forest | 0.047047 | [0.044956, 0.049044] | 0 | strong |
| lendingclub_v2 | lr | boruta_then_rfe_catboost k20 vs boruta_random_forest | 0.040422 | [0.038339, 0.042400] | 0 | strong |
| lendingclub_v2 | lr | statistical_normalized_average_rank k20 vs boruta_random_forest | 0.037385 | [0.035323, 0.039323] | 5.26e-318 | strong |
| lendingclub_v2 | lr | boruta_then_mrmr_mutual_information k20 vs boruta_random_forest | 0.037122 | [0.034831, 0.039263] | 6.87e-259 | strong |
| lendingclub_v2 | catboost | iv_then_boruta pool300 vs boruta_random_forest | 0.034400 | [0.032863, 0.035943] | 0 | strong |
| lendingclub_v2 | catboost | boruta_then_rfe_catboost k40 vs boruta_random_forest | 0.031697 | [0.030102, 0.033284] | 0 | strong |

Important counterevidence is retained rather than averaged away:

| Dataset | Model | Contrast | ΔAUC | 95% CI | Holm p | Grade |
|---|---|---|---:|---:|---:|---|
| lendingclub_v2 | catboost | statistical_normalized_average_rank k40 vs rfe_catboost | -0.005934 | [-0.006622, -0.005276] | 6.74e-68 | not_supported |
| lendingclub_v2 | lr | statistical_normalized_average_rank k20 vs rfe_catboost | -0.005748 | [-0.006469, -0.005056] | 2.51e-55 | not_supported |
| homecredit | catboost | boruta_then_rfe_catboost k40 vs rfe_catboost | -0.005516 | [-0.007050, -0.003996] | 3.42e-13 | not_supported |
| lendingclub_v2 | catboost | statistical_normalized_average_rank k40 vs cross_dataset_rank_voting_v1_primary_pool_200 | -0.004260 | [-0.004953, -0.003577] | 4.28e-36 | not_supported |
| lendingclub_v2 | lr | statistical_normalized_average_rank k20 vs catboost_shap | -0.003410 | [-0.004162, -0.002706] | 3.4e-20 | not_supported |
| homecredit | catboost | statistical_normalized_average_rank k40 vs rfe_catboost | -0.002833 | [-0.004467, -0.001131] | 0.000562 | not_supported |
| lendingclub_v2 | lr | boruta_then_rfe_catboost k20 vs rfe_catboost | -0.002711 | [-0.003216, -0.002213] | 2.71e-25 | not_supported |
| lendingclub_v2 | catboost | statistical_normalized_average_rank k40 vs catboost_shap | -0.002121 | [-0.002816, -0.001455] | 1.12e-09 | not_supported |

## DEV-to-OOT generalization and stability

Across authenticated configurations, OOT-minus-DEV AUC ranged from -0.0376 to 0.0265. Both preservation and degradation occurred. Mean pairwise Jaccard and Kuncheva evidence varied by method and dataset; these measures contextualize selection stability but do not establish a universal stability improvement. The generalization figure is dataset/model stratified and treats OOT as primary.

## Statistical comparisons and uncertainty

Exactly 124 comparisons were statistically evaluable; protocol-allowed unavailable=0, protocol-allowed infeasible=0, authentication failures=0. A confidence interval crossing zero is treated as insufficient evidence of a difference, not equivalence. No non-inferiority margin or equivalence test was registered.

## Drift and resource trade-offs

Persisted or saved-prediction-derived score PSI ranged from 0.0006 to 0.0695. Combination feature PSI was not reconstructed because doing so would require raw feature tables; this is a limitation, not a zero-drift result. The highest observed persisted peak RSS was 28.35 GiB for lendingclub_v2 / lr / lasso_l1_logistic k20. Runtime evidence is uneven across method families, so resource comparisons are descriptive.

## Natural-support analysis

The two frozen Home Credit CatBoost Boruta-first reference cases remain requested K=40, reference realized K=26, `infeasible_natural_support`, and unpadded. Their OOT full-DEV selector refits authenticated 40 selected features. Both facts are reported: the later refit does not erase the 26-feature reference support, and the resulting contrasts are not described as ordinary like-for-like K=40 evidence.

## Evidence grades and defensible claims

- **claim_01_combination_discrimination — moderate**: Some registered combination contrasts improved locked-OOT AUC; effects were heterogeneous. Counterevidence: p14-c-005__homecredit__lr__statistical_normalized_average_rank__k20__vs__catboost_shap|p14-c-020__homecredit__catboost__statistical_normalized_average_rank__k40__vs__rfe_catboost|p14-c-022__homecredit__catboost__statistical_normalized_average_rank__k40__vs__catboost_shap|p14-c-034__homecredit__catboost__boruta_then_rfe_catboost__k40__vs__rfe_catboost|p14-c-037__lendingclub_v2__lr__statistical_normalized_average_rank__k20__vs__rfe_catboost|p14-c-039__lendingclub_v2__lr__statistical_normalized_average_rank__k20__vs__catboost_shap|p14-c-051__lendingclub_v2__lr__boruta_then_rfe_catboost__k20__vs__rfe_catboost|p14-c-054__lendingclub_v2__catboost__statistical_normalized_average_rank__k40__vs__rfe_catboost|p14-c-056__lendingclub_v2__catboost__statistical_normalized_average_rank__k40__vs__catboost_shap|p14-c-068__lendingclub_v2__catboost__boruta_then_rfe_catboost__k40__vs__rfe_catboost.
- **claim_02_selection_temporal_stability — weak**: Stability and score drift vary by method; no universal stability improvement is established. Counterevidence: No registered paired stability test; feature PSI is unavailable for combination OOT without raw features..
- **claim_03_cross_dataset_consistency — moderate**: Cross-dataset consistency is descriptive and depends on method, model, and comparator. Counterevidence: p14-c-005__homecredit__lr__statistical_normalized_average_rank__k20__vs__catboost_shap|p14-c-020__homecredit__catboost__statistical_normalized_average_rank__k40__vs__rfe_catboost|p14-c-022__homecredit__catboost__statistical_normalized_average_rank__k40__vs__catboost_shap|p14-c-034__homecredit__catboost__boruta_then_rfe_catboost__k40__vs__rfe_catboost|p14-c-037__lendingclub_v2__lr__statistical_normalized_average_rank__k20__vs__rfe_catboost|p14-c-039__lendingclub_v2__lr__statistical_normalized_average_rank__k20__vs__catboost_shap|p14-c-051__lendingclub_v2__lr__boruta_then_rfe_catboost__k20__vs__rfe_catboost|p14-c-054__lendingclub_v2__catboost__statistical_normalized_average_rank__k40__vs__rfe_catboost|p14-c-056__lendingclub_v2__catboost__statistical_normalized_average_rank__k40__vs__catboost_shap|p14-c-068__lendingclub_v2__catboost__boruta_then_rfe_catboost__k40__vs__rfe_catboost.
- **claim_04_tradeoff — moderate**: Trade-offs are configuration-specific; no universal winner is supported. Counterevidence: Selector fit time and peak RSS vary substantially; prediction-time evidence is incomplete for combination OOT..
- **claim_05_natural_support — moderate**: The frozen reference support was 26 of 40 and unpadded; OOT full-DEV refits realized 40 and are not ordinary like-for-like K=40 evidence. Counterevidence: p14-c-034__homecredit__catboost__boruta_then_rfe_catboost__k40__vs__rfe_catboost.
- **claim_06_voting_context — weak**: Primary pool-200 voting is contextual evidence in four registered relationships; pool sensitivities were excluded. Counterevidence: p14-c-041__lendingclub_v2__lr__statistical_normalized_average_rank__k20__vs__cross_dataset_rank_voting_v1_primary_pool_200|p14-c-058__lendingclub_v2__catboost__statistical_normalized_average_rank__k40__vs__cross_dataset_rank_voting_v1_primary_pool_200.

## Limitations

Only two datasets are complete, with different domains and temporal boundaries. Multiplicity reduces false-positive flexibility but does not create practical importance. Several families are exploratory, natural-support comparability is unusual, and combination feature-drift metrics cannot be recreated without prohibited raw-data access. Cross-dataset synthesis is descriptive; no pooled estimate was registered. The third dataset is separately frozen and preregistered but has not been implemented or executed.

## Conclusion

The locked evidence supports configuration-specific conclusions, not a universal combination winner. Some effects are statistically and practically favorable within named dataset/model/comparator scopes; other registered contrasts are adverse or inconclusive. The next scientific step is to implement and data-free-test the already frozen third-dataset adapter in a separate prompt, then review the implementation before authorizing its bounded pilot.
