# todo_done fill log

Generated 2026-09-22T14:01:16Z after 102.5 minutes.

## CSV fill summary

| file | filled cells | total value cells | status |
|---|---|---|---|
| 39a_onehot_dimension.csv | 5 | 5 | complete |
| 44_mechanical.csv | 28 | 28 | complete |
| 45_repeated_calls.csv | 37 | 40 | partial |
| 47_obfuscation.csv | 42 | 56 | partial |

## Skipped or blank cells

- **44** {'dataset': 'lendingclub_v2', 'backbone': 'catboost', 'selector': 'LLM then mRMR'}: Appendix E subset carries 13 name(s) absent from this machine's candidate universe, e.g. ['term_60 months', 'term_36 months', 'term_home_ownership_60 months__RENT']
- **47** {'call': 'lendingclub_v2__fold1__b20__A_names_removed', 'dataset': 'lendingclub_v2', 'partition': 'fold1', 'condition': 'obfuscated'}: ranking call failed the frozen target-free contract; cells left blank: issue=4: ValueError: LLM response failed the strict target-free ranking contract after 3 attempts: ['attempt=1: ValueError: LLM response contains duplicate feature names', 'attempt=2: ValueError: LLM response contains duplicate feature names', 'attempt=3: ValueError: LLM response contains duplicate feature names']
- **47** {'call': 'lendingclub_v2__fold2__b20__A_names_removed', 'dataset': 'lendingclub_v2', 'partition': 'fold2', 'condition': 'obfuscated'}: ranking call failed the frozen target-free contract; cells left blank: issue=4: ValueError: LLM response failed the strict target-free ranking contract after 3 attempts: ["attempt=1: ValueError: LLM response contains unknown feature names: ['F553', 'F559', 'F551', 'F155', 'F200', 'F207', 'F594']", "attempt=2: ValueError: LLM response contains unknown feature names: ['F559', 'F553', 'F200', 'F207', 'F594']", 'attempt=3: ValueError: LLM response contains duplicate feature names']
- **47** {'call': 'lendingclub_v2__fold3__b20__A_names_removed', 'dataset': 'lendingclub_v2', 'partition': 'fold3', 'condition': 'obfuscated'}: ranking call failed the frozen target-free contract; cells left blank: issue=4: ValueError: LLM response failed the strict target-free ranking contract after 3 attempts: ['attempt=1: ValueError: LLM response contains duplicate feature names', 'attempt=2: ValueError: LLM response contains duplicate feature names', 'attempt=3: ValueError: LLM response contains duplicate feature names']
- **47** {'call': 'lendingclub_v2__fold4__b20__A_names_removed', 'dataset': 'lendingclub_v2', 'partition': 'fold4', 'condition': 'obfuscated'}: ranking call failed the frozen target-free contract; cells left blank: issue=4: ValueError: LLM response failed the strict target-free ranking contract after 3 attempts: ['attempt=1: ValueError: LLM response contains duplicate feature names', 'attempt=2: ValueError: LLM response contains duplicate feature names', 'attempt=3: ValueError: LLM response contains duplicate feature names']
- **47** {'call': 'lendingclub_v2__fold5__b20__A_names_removed', 'dataset': 'lendingclub_v2', 'partition': 'fold5', 'condition': 'obfuscated'}: ranking call failed the frozen target-free contract; cells left blank: issue=4: ValueError: LLM response failed the strict target-free ranking contract after 3 attempts: ['attempt=1: ValueError: LLM response contains duplicate feature names', 'attempt=2: ValueError: LLM response contains duplicate feature names', 'attempt=3: ValueError: LLM response contains duplicate feature names']
- **47** {'call': 'lendingclub_v2__full_dev__b20__A_names_removed', 'dataset': 'lendingclub_v2', 'partition': 'full_dev', 'condition': 'obfuscated'}: ranking call failed the frozen target-free contract; cells left blank: issue=4: ValueError: LLM response failed the strict target-free ranking contract after 3 attempts: ["attempt=1: ValueError: LLM response contains unknown feature names: ['F553', 'F559 / F392', 'F271 / F392', 'F667 / F392', 'F479 / F392', 'F135 / F392', 'F385 / F392', 'F553 / F392', 'F062 / F392', 'F559 / F200']", 'attempt=2: ValueError: LLM response contains duplicate feature names', 'attempt=3: ValueError: LLM response contains duplicate feature names']
- **47** {'call': 'lendingclub_v2__fold1__b40__A_names_removed', 'dataset': 'lendingclub_v2', 'partition': 'fold1', 'condition': 'obfuscated'}: ranking call failed the frozen target-free contract; cells left blank: issue=4: ValueError: LLM response failed the strict target-free ranking contract after 3 attempts: ["attempt=1: ValueError: LLM response contains unknown feature names: ['F559', 'F553', 'F290']", "attempt=2: ValueError: LLM response contains unknown feature names: ['F559', 'F553']", 'attempt=3: ValueError: LLM response contains duplicate feature names']
- **47** {'call': 'lendingclub_v2__fold2__b40__A_names_removed', 'dataset': 'lendingclub_v2', 'partition': 'fold2', 'condition': 'obfuscated'}: ranking call failed the frozen target-free contract; cells left blank: issue=4: ValueError: LLM response failed the strict target-free ranking contract after 3 attempts: ['attempt=1: ValueError: LLM response contains duplicate feature names', 'attempt=2: ValueError: LLM response contains duplicate feature names', 'attempt=3: ValueError: LLM response contains duplicate feature names']
- **47** {'call': 'lendingclub_v2__fold3__b40__A_names_removed', 'dataset': 'lendingclub_v2', 'partition': 'fold3', 'condition': 'obfuscated'}: ranking call failed the frozen target-free contract; cells left blank: issue=4: ValueError: LLM response failed the strict target-free ranking contract after 3 attempts: ['attempt=1: ValueError: LLM response contains duplicate feature names', "attempt=2: ValueError: LLM response contains unknown feature names: ['F559', 'F553']", 'attempt=3: ValueError: LLM response contains duplicate feature names']
- **47** {'call': 'lendingclub_v2__fold4__b40__A_names_removed', 'dataset': 'lendingclub_v2', 'partition': 'fold4', 'condition': 'obfuscated'}: ranking call failed the frozen target-free contract; cells left blank: issue=4: ValueError: LLM response failed the strict target-free ranking contract after 3 attempts: ['attempt=1: ValueError: LLM response contains duplicate feature names', 'attempt=2: ValueError: LLM response contains duplicate feature names', 'attempt=3: ValueError: LLM response contains duplicate feature names']
- **47** {'call': 'lendingclub_v2__fold5__b40__A_names_removed', 'dataset': 'lendingclub_v2', 'partition': 'fold5', 'condition': 'obfuscated'}: ranking call failed the frozen target-free contract; cells left blank: issue=4: ValueError: LLM response failed the strict target-free ranking contract after 3 attempts: ["attempt=1: ValueError: LLM response contains unknown feature names: ['F551', 'F553', 'F200', 'F207']", 'attempt=2: ValueError: LLM response contains duplicate feature names', 'attempt=3: ValueError: LLM response contains duplicate feature names']
- **47** {'call': 'lendingclub_v2__full_dev__b40__A_names_removed', 'dataset': 'lendingclub_v2', 'partition': 'full_dev', 'condition': 'obfuscated'}: ranking call failed the frozen target-free contract; cells left blank: issue=4: ValueError: LLM response failed the strict target-free ranking contract after 3 attempts: ['attempt=1: ValueError: LLM response contains duplicate feature names', 'attempt=2: ValueError: LLM response contains duplicate feature names', 'attempt=3: ValueError: LLM response contains duplicate feature names']
- **47** {'arm': 'A_names_removed', 'dataset': 'lendingclub_v2', 'backbone': 'lr', 'K': '20'}: arm full-DEV ranking unavailable (call not made)
- **47** {'arm': 'A_names_removed', 'dataset': 'lendingclub_v2', 'backbone': 'catboost', 'K': '40'}: arm full-DEV ranking unavailable (call not made)
- **45** {'call': 'lendingclub_v2__full_dev__b20__repeat01__named', 'dataset': 'lendingclub_v2', 'partition': 'full_dev', 'condition': 'named'}: ranking call failed the frozen target-free contract; cells left blank: issue=4: APIConnectionError: Connection error.
- **45** {'call': 'lendingclub_v2__full_dev__b20__repeat06__named', 'dataset': 'lendingclub_v2', 'partition': 'full_dev', 'condition': 'named'}: ranking call failed the frozen target-free contract; cells left blank: issue=4: ValueError: LLM response failed the strict target-free ranking contract after 3 attempts: ['attempt=1: ValueError: LLM response contains duplicate feature names', 'attempt=2: ValueError: LLM response contains duplicate feature names', 'attempt=3: ValueError: LLM response contains duplicate feature names']
- **45** {'call': 'lendingclub_v2__full_dev__b20__repeat07__named', 'dataset': 'lendingclub_v2', 'partition': 'full_dev', 'condition': 'named'}: ranking call failed the frozen target-free contract; cells left blank: issue=4: ValueError: LLM response failed the strict target-free ranking contract after 3 attempts: ["attempt=1: ValueError: LLM response contains unknown feature names: ['loan_amnt_to_loan_amnt']", 'attempt=2: ValueError: LLM response contains duplicate feature names', "attempt=3: ValueError: LLM response contains unknown feature names: ['loan_amnt_to_loan_amnt']"]
- **45** {'dataset': 'lendingclub_v2', 'repeat_id': '1', 'backbone': 'lr', 'K': '20'}: ranking call not made
- **45** {'dataset': 'lendingclub_v2', 'repeat_id': '6', 'backbone': 'lr', 'K': '20'}: ranking call not made
- **45** {'dataset': 'lendingclub_v2', 'repeat_id': '7', 'backbone': 'lr', 'K': '20'}: ranking call not made

## Notes

- **setup**: Appendix E headline subsets loaded from ~/projects/article/appendix_e_subsets.csv: 12 cells
- **39a**: homecredit/lr/mRMR: using the Appendix E subset (20 features) instead of a local refit
- **39a**: lendingclub_v2/lr/IV then Boruta: using the Appendix E subset (20 features) instead of a local refit
- **39a**: lendingclub_v2/catboost/IV then Boruta: using the Appendix E subset (40 features) instead of a local refit
- **39a**: homecredit_model_stability_2024/lr/RFE CatBoost: using the Appendix E subset (20 features) instead of a local refit
- **39a**: homecredit_model_stability_2024/catboost/CatBoost SHAP: using the Appendix E subset (40 features) instead of a local refit
- **44**: call plan narrowed to the skeleton: partitions=['full_dev'], LendingClub budgets=[20, 40, 100]
- **44**: {'dataset': 'homecredit', 'backbone': 'lr', 'selector': 'Pure LLM', 'K': '20'}: cached-description refit HO AUC 0.739956 vs prefilled ho_auc_cached 0.74363537 (artifacts/llm_cache full-DEV ranking truncation)
- **44**: {'dataset': 'homecredit', 'backbone': 'catboost', 'selector': 'Pure LLM', 'K': '40'}: cached-description refit HO AUC 0.758299 vs prefilled ho_auc_cached 0.79345 (artifacts/llm_cache full-DEV ranking truncation)
- **44**: {'dataset': 'lendingclub_v2', 'backbone': 'lr', 'selector': 'Pure LLM', 'K': '20'}: cached-description refit HO AUC 0.692657 vs prefilled ho_auc_cached 0.740234 (artifacts/llm_cache full-DEV ranking truncation)
- **44**: {'dataset': 'lendingclub_v2', 'backbone': 'catboost', 'selector': 'Pure LLM', 'K': '40'}: cached-description refit HO AUC 0.713673 vs prefilled ho_auc_cached 0.7335434 (artifacts/llm_cache full-DEV ranking truncation)
- **44**: {'dataset': 'lendingclub_v2', 'backbone': 'catboost', 'selector': 'LLM then mRMR', 'K': '40'}: cached-description refit HO AUC 0.710258 vs prefilled ho_auc_cached 0.770664 (recomputed:legacy_matrix_llm_pool_then_dense_statistical_stage)
- **44**: {'dataset': 'homecredit_model_stability_2024', 'backbone': 'lr', 'selector': 'Pure LLM', 'K': '20'}: cached-description refit HO AUC 0.694326 vs prefilled ho_auc_cached 0.8344 (regenerated named ranking (paper's cached Stability 2024 ranking is not on this machine))
- **44**: {'dataset': 'homecredit_model_stability_2024', 'backbone': 'catboost', 'selector': 'Pure LLM', 'K': '40'}: cached-description refit HO AUC 0.769226 vs prefilled ho_auc_cached 0.8784 (regenerated named ranking (paper's cached Stability 2024 ranking is not on this machine))
- **44**: row 6 n_ho: kept the pre-filled '369147'; this run computed '304916'
- **44**: row 7 n_ho: kept the pre-filled '369147'; this run computed '304916'
- **47**: lendingclub_v2/lr: named control refit HO AUC 0.692657 (prefilled ho_auc_named in the skeleton is the paper value)
- **47**: lendingclub_v2/catboost: named control refit HO AUC 0.713673 (prefilled ho_auc_named in the skeleton is the paper value)
- **47**: homecredit_model_stability_2024/lr: named control refit HO AUC 0.694326 (prefilled ho_auc_named in the skeleton is the paper value)
- **47**: homecredit_model_stability_2024/catboost: named control refit HO AUC 0.769226 (prefilled ho_auc_named in the skeleton is the paper value)
- **45**: homecredit/lr: 10 repeats, Nogueira 0.6557232704402516, mean Jaccard 0.5372002256532035, HO AUC median 0.7398305000000001 sd 0.011797489294572637
- **45**: homecredit/catboost: 10 repeats, Nogueira 0.6268233618233618, mean Jaccard 0.5315626114218339, HO AUC median 0.7609269999999999 sd 0.0075622358245869825
- **45**: lendingclub_v2/lr: 7 repeats, Nogueira 0.8477541371158392, mean Jaccard 0.7733161248689198, HO AUC median 0.673525 sd 0.0008982508399264332
- **45**: lendingclub_v2/catboost: 10 repeats, Nogueira 0.600087235996327, mean Jaccard 0.5467391141259476, HO AUC median 0.703822 sd 0.010127282843334081
- **setup**: ranking calls made in this run: 11; calls that failed the frozen contract: 15; 206 attempt rows in contract_issues.csv
- **setup**: 15 ranking call(s) exhausted the three frozen attempts; their cells are blank. Re-running fills them if the model complies, or use --retry-failed-calls N (a disclosed deviation from the frozen protocol).

## Cell sources

- recomputed: 236 cells
