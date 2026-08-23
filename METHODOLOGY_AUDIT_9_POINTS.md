# Nine-point methodology audit

This note answers the nine requested methodology questions from the repository as it exists on 2026-08-21. It covers the Home Credit, LendingClub v2, and Home Credit Model Stability 2024 experiments. It intentionally contains no predictive-performance scores, winner claims, or score comparisons.

## Executive status

| # | Resolved status | Repository-backed conclusion |
|---:|---|---|
| 1 | Confirmed, with scope notes | Executed data-fitted selectors are instantiated and fitted inside each DEV training fold. Pure LLM ranking is an intentional target-free shared-ranking exception. PCA is implemented but is not in the current frozen baseline or third-dataset matrix. |
| 2 | Confirmed for the recorded workflow | The frozen protocols prohibit changing rankings, feature sets, method definitions, hyperparameters, or threshold rules after OOT inspection. The third-dataset OOT authorization additionally records zero LLM calls and no ranking regeneration. |
| 3 | Procedure confirmed; rationale missing | Thresholds maximize training-sample KS/Youden and are applied unchanged to validation/OOT. The repository does not document why full-DEV in-sample probabilities were preferred to OOF probabilities for the final threshold. |
| 4 | Confirmed | LR uses balanced class weights and CatBoost uses automatic balanced class weights. No SMOTE, class-balancing over/undersampling, or `scale_pos_weight` is configured. |
| 5 | Confirmed | The final-model preprocessing is identifiable for every dataset-model pair, including imputation, scaling, categorical encoding, and unseen-category behavior. |
| 6 | Confirmed for repository-controlled settings | The frozen LR and CatBoost arguments and seeds are recorded. Parameters not passed by the wrappers remain library defaults under the locked dependency versions. |
| 7 | Partially confirmed | The third dataset has an exact attempt history: two application attempts, then success, with no fallback; OOT made zero LLM calls. The older two-dataset cache does not preserve per-request attempt counts, although no extant cache payload records fallback use. |
| 8 | Partially complete | Full selected-feature/ranking artifacts exist. The third dataset preserves the exact rendered prompt and each request/response. The older two-dataset cache preserves prompt hashes, inputs, raw responses, and rankings, but not verbatim rendered `prompt.txt` files. |
| 9 | Complete for the third dataset's frozen candidate scope; partial overall | The third dataset has a row-level decision-time availability review with no unresolved fields in its approved depth-0/depth-1 scope. Home Credit and LendingClub have enforced leakage exclusions, but not an equally complete per-feature point-in-time availability ledger. |

## 1. Fold-local selector fitting

**Answer:** yes for every data-fitted selector actually executed through the frozen pipelines.

The generic training loop creates a new preprocessor and selector for every fold and passes only that fold's training and validation partitions to the fold worker. The fold worker fits the selection encoder and selector on `X_train, y_train`, then only transforms `X_val`. This applies to supervised selectors such as IV, mRMR, Boruta, LASSO, CatBoost-SHAP, and RFE. The combination experiment and third-dataset implementation independently repeat the same fold-local pattern. A separate refit on all DEV data is performed only for the final model before OOT scoring.

There are two important qualifications:

- **Pure LLM ranking:** the accepted third-dataset LLM ranking is explicitly target-free and shared across folds. It is generated once from feature names/descriptions, then each fold reuses/truncates that sealed order. Statistical selectors and the statistical component of hybrid selectors remain fold-local.
- **PCA:** [`PCASelector`](src/credit_risk_fs/selectors/pca.py) is implemented and would be newly fitted by the generic per-fold selector factory. However, the current frozen [`full_baseline_v1`](configs/experiments/full_baseline_v1.yaml) does not register PCA, and the third-dataset lock explicitly says PCA is outside the current matrix. There are no PCA-named run artifacts in the current `results/` tree. Therefore, the repository supports the fold-local PCA mechanism, not a claim that PCA was executed in these frozen runs.

Primary evidence:

- [`src/credit_risk_fs/models/training.py`](src/credit_risk_fs/models/training.py), especially the fold loop and fresh preprocessor/selector construction.
- [`src/credit_risk_fs/models/_fold.py`](src/credit_risk_fs/models/_fold.py), where selection encoding and selector fitting use only the training slice.
- [`docs/research_extension/full_baseline_v1.md`](docs/research_extension/full_baseline_v1.md) and [`configs/experiments/full_baseline_v1.yaml`](configs/experiments/full_baseline_v1.yaml), which freeze `selector_fit_boundary: fold_training_original_feature_candidates`.
- [`src/credit_risk_fs/experiments/selector_combinations.py`](src/credit_risk_fs/experiments/selector_combinations.py) and [`src/credit_risk_fs/experiments/prompt_16_third_dataset.py`](src/credit_risk_fs/experiments/prompt_16_third_dataset.py), which fit selectors separately within each loaded fold.
- [`cleanup/audits/prompt_16_final_amended_oot/full_dev_selector_refit_registry.json`](cleanup/audits/prompt_16_final_amended_oot/full_dev_selector_refit_registry.json), which authenticates the final full-DEV refits.

## 2. No changes after inspecting OOT

**Answer:** confirmed by the frozen configuration and audit trail for the recorded workflow.

Report-ready statement:

> OOT was evaluation-only: no selector ranking, selected-feature set, model hyperparameter, threshold rule, method registry, or comparison definition was changed after OOT outcomes were inspected; resource-only amendments were versioned and frozen before the affected results were opened.

The two-dataset baseline sets `configuration_frozen_before_full_baseline_execution: true`, defines OOT as a locked final evaluation, and sets `configuration_adaptation_after_oot: forbidden`. The combination preregistration prohibits changing membership, order, budgets, weights, missing-rank rules, or tie rules after freeze. The two-dataset review also records that no method, budget, threshold, seed, fold, model, or ordering was changed.

For the third dataset, the final execution authorization was created before OOT inspection, authorizes no LLM API calls or ranking regeneration, forbids a second scientific OOT attempt, and preserves completed cells as immutable. The preservation/deviation register records the permitted operational amendments and states that they did not authorize scientific adaptation.

Primary evidence:

- [`configs/experiments/full_baseline_v1.yaml`](configs/experiments/full_baseline_v1.yaml)
- [`configs/protocols/selector_combinations_v1/combination_preregistration.md`](configs/protocols/selector_combinations_v1/combination_preregistration.md)
- [`cleanup/audits/prompt_13_combination_dev_review/review_report.md`](cleanup/audits/prompt_13_combination_dev_review/review_report.md)
- [`cleanup/audits/prompt_16_final_amended_oot/execution_authorization.json`](cleanup/audits/prompt_16_final_amended_oot/execution_authorization.json)
- [`cleanup/audits/prompt_16_final_amended_oot/preservation_deviation_and_revocation_register.json`](cleanup/audits/prompt_16_final_amended_oot/preservation_deviation_and_revocation_register.json)

This is strong evidence about the repository-recorded process. As with any source-code audit, it cannot prove that no unrecorded off-repository human inspection occurred.

## 3. Threshold derivation and the missing OOF rationale

**Implemented procedure:** `determine_threshold` uses the ROC curve and selects the threshold maximizing `TPR - FPR`. This is Youden's J and is also the maximum vertical KS separation under the implementation. In each DEV fold, the model is fitted on the fold-training slice, probabilities are generated for that same training slice, and the maximizing threshold is applied unchanged to the fold validation slice. For final evaluation, the model is fitted on all DEV, the threshold is selected from full-DEV in-sample probabilities, and that value is applied unchanged to OOT. OOT labels or probabilities are not used to select it.

Evidence:

- [`src/credit_risk_fs/evaluation/metrics.py`](src/credit_risk_fs/evaluation/metrics.py)
- [`src/credit_risk_fs/models/_fold.py`](src/credit_risk_fs/models/_fold.py)
- [`src/credit_risk_fs/pipelines/common.py`](src/credit_risk_fs/pipelines/common.py)
- [`src/credit_risk_fs/experiments/prompt_16_final_oot.py`](src/credit_risk_fs/experiments/prompt_16_final_oot.py)
- [`configs/protocols/credit_scoring_extension_v1.yaml`](configs/protocols/credit_scoring_extension_v1.yaml)
- [`cleanup/audits/prompt_16_final_amended_oot/oot_analysis_plan.json`](cleanup/audits/prompt_16_final_amended_oot/oot_analysis_plan.json)

**Unresolved documentation gap:** no repository document explains why the final operational threshold uses in-sample full-DEV probabilities instead of pooled DEV OOF probabilities. The repository does preserve OOF probabilities for other evaluation and drift uses, so the absence is specifically a rationale gap, not an inability to create OOF scores.

One plausible technical explanation is that the final threshold is paired with the single final model fitted on all DEV, whereas pooled OOF probabilities come from multiple fold-specific models. That explanation is an inference from the architecture, not a stated protocol rationale. It should not be presented as an author-confirmed reason unless added explicitly. The report should also acknowledge that optimizing a threshold on in-sample probabilities can be optimistic.

## 4. Class imbalance handling

**Final models:**

- Logistic regression passes `class_weight="balanced"`.
- CatBoost passes `auto_class_weights="Balanced"`.
- CatBoost does not pass `scale_pos_weight`.

**Sampling:** repository search and the frozen configurations show no SMOTE, class-balancing oversampling, or class-balancing undersampling in the model-training pipeline. The resampling inside `stable_core_llm_fill` is a feature-selection stability bootstrap; it is not a minority-class balancing procedure. The Boruta internal random forest is explicitly configured with `class_weight: null`, and the selector-internal LASSO/RFE/SHAP estimators do not add a separate imbalance weight unless stated in their frozen selector configuration.

Evidence:

- [`configs/models/lr.yaml`](configs/models/lr.yaml)
- [`configs/models/catboost.yaml`](configs/models/catboost.yaml)
- [`src/credit_risk_fs/models/logistic_regression.py`](src/credit_risk_fs/models/logistic_regression.py)
- [`src/credit_risk_fs/models/catboost_model.py`](src/credit_risk_fs/models/catboost_model.py)
- [`configs/experiments/full_baseline_v1.yaml`](configs/experiments/full_baseline_v1.yaml)

## 5. Dataset-by-model preprocessing

Both final models receive the same preprocessed matrix within a dataset. CatBoost is **not** given native categorical columns in these experiments.

| Dataset | Model | Numeric fields | Categorical fields | Unseen categories | Final representation |
|---|---|---|---|---|---|
| Home Credit | LR | infinities become missing; training mean imputation; standard scaling | missing token `Missing`; one-hot encoding; minimum frequency 10 | `handle_unknown="ignore"`, therefore no failure and an all-zero block for an unseen level | dense `float32` |
| Home Credit | CatBoost | same | same | same | dense `float32` |
| LendingClub v2 | LR | infinities become missing; training mean imputation; standard scaling | missing token `Missing`; one-hot encoding; minimum frequency 50 | `handle_unknown="ignore"` | dense `float32` |
| LendingClub v2 | CatBoost | same | same | same | dense `float32` |
| Home Credit Model Stability 2024 | LR | infinities become missing; training mean imputation; centered standard scaling | missing token `Missing`; one-hot encoding; minimum frequency 10 | `handle_unknown="ignore"` | canonical CSR sparse `float32` |
| Home Credit Model Stability 2024 | CatBoost | same | same | same | canonical CSR sparse `float32` |

All imputation values, scaler parameters, categories, and frequency grouping are fitted on the relevant training boundary: fold training for DEV validation and full DEV for OOT. The third-dataset sparse amendment changes storage/representation only; it explicitly preserves the dense preprocessor's numerical and categorical semantics.

Before final-model preprocessing, data-fitted selectors use [`OriginalFeatureNumericEncoder`](src/credit_risk_fs/preprocessing/encoding.py): numerical columns use fold-training medians, categorical missing values use `<MISSING>`, categorical levels receive a deterministic training-fitted integer map, and unseen levels map to `-1`. This keeps one encoded column per original candidate feature so selectors rank original features rather than one-hot expansions.

Evidence:

- [`src/credit_risk_fs/preprocessing/encoding.py`](src/credit_risk_fs/preprocessing/encoding.py)
- [`configs/experiments/homecredit_matrix.yaml`](configs/experiments/homecredit_matrix.yaml)
- [`configs/experiments/lendingclub_v2_matrix.yaml`](configs/experiments/lendingclub_v2_matrix.yaml)
- [`cleanup/audits/prompt_16_sparse_final_model_preprocessing_v6/sparse_final_model_amendment.json`](cleanup/audits/prompt_16_sparse_final_model_preprocessing_v6/sparse_final_model_amendment.json)

## 6. LR and CatBoost hyperparameters and seeds

The same frozen final-model settings are used across the three datasets.

### Logistic regression

| Parameter | Value |
|---|---|
| implementation | `credit_risk_fs.models.logistic_regression.LogisticRegressionModel` |
| solver | `liblinear` |
| maximum iterations | `1000` |
| class weight | `balanced` |
| random state | `42` |

The wrapper passes only those four estimator arguments. Other logistic-regression parameters are the defaults of the locked scikit-learn version in [`uv.lock`](uv.lock), rather than separately frozen values in the experiment YAML.

### CatBoost

| Parameter | Value |
|---|---|
| implementation | `credit_risk_fs.models.catboost_model.CatBoostModel` |
| depth | `10` |
| learning rate | `0.01` |
| L2 leaf regularization | `95` |
| minimum data in leaf | `290` |
| column sample by level | `0.9` |
| random strength | `0.125` |
| grow policy | `Depthwise` |
| one-hot maximum size | `21` |
| leaf estimation | `Newton` |
| bootstrap type | `Bernoulli` |
| subsample | `0.55` |
| loss | `Logloss` |
| evaluation metric | `AUC` |
| automatic class weights | `Balanced` |
| iterations | `1500` |
| early-stopping rounds | `150` configured |
| verbose interval | `100` |
| random state | `42` |
| write CatBoost files | `false` |
| estimator thread cap | `4` in the frozen executions |

Although `early_stopping_rounds=150` is configured, the frozen paths call model fitting without an external evaluation set; therefore early stopping is inactive and the configured iteration count governs the fit.

The model/experiment seed is `42`. Selector seeds are also frozen in the selector registries, while inferential bootstrap seeds are separate analysis seeds and should not be described as model-training seeds.

Evidence:

- [`configs/experiments/full_baseline_v1.yaml`](configs/experiments/full_baseline_v1.yaml)
- [`configs/protocols/credit_scoring_extension_v1.yaml`](configs/protocols/credit_scoring_extension_v1.yaml)
- [`configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json`](configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json)
- [`src/credit_risk_fs/experiments/resource_policy.py`](src/credit_risk_fs/experiments/resource_policy.py)
- [`src/credit_risk_fs/models/registry.py`](src/credit_risk_fs/models/registry.py)

## 7. LLM retry counts and fallback use

The implementation has two relevant contracts:

- The legacy Home Credit/LendingClub path permits up to three application-level attempts to obtain parseable JSON and may write a deterministic fallback ranking after all three fail.
- The third-dataset target-free path also allows exactly three validation attempts, but forbids fallback; it fails closed if no valid response is obtained.

**What can be established from artifacts:**

- The accepted third-dataset v3 ranking made two application calls. Attempt 1 was rejected because the response contained an unknown feature name. Attempt 2 passed validation. [`ranking_payload.json`](results/prompt_16_homecredit_model_stability_2024/dev_llm_supplement_v3/llm_ranking/ranking_payload.json) records `application_attempt: 2` and `fallback_used: false`.
- The final third-dataset OOT manifest records `llm_api_request_count: 0` and `ranking_regeneration_count: 0`; it reused the sealed DEV ranking.
- The current older two-dataset cache contains 51 terminal ranking payloads. None contains `fallback_reason` or `llm_response_parse_errors`, so there is no artifact evidence that a legacy deterministic fallback supplied one of those cached rankings.

**What cannot be established:** the older cache stores the terminal successful response, not an attempt-by-attempt request log. Consequently, 51 cache files must not be reported as 51 API calls, and exact retry counts for the Home Credit/LendingClub calls cannot be reconstructed from the repository.

Evidence:

- [`src/credit_risk_fs/selectors/llm_screening.py`](src/credit_risk_fs/selectors/llm_screening.py)
- [`results/prompt_16_homecredit_model_stability_2024/dev_llm_supplement_v3/llm_ranking/manifest.json`](results/prompt_16_homecredit_model_stability_2024/dev_llm_supplement_v3/llm_ranking/manifest.json)
- [`results/prompt_16_homecredit_model_stability_2024/dev_llm_supplement_v3/llm_ranking/attempts/attempt_001/status.json`](results/prompt_16_homecredit_model_stability_2024/dev_llm_supplement_v3/llm_ranking/attempts/attempt_001/status.json)
- [`results/prompt_16_homecredit_model_stability_2024/dev_llm_supplement_v3/llm_ranking/attempts/attempt_002/status.json`](results/prompt_16_homecredit_model_stability_2024/dev_llm_supplement_v3/llm_ranking/attempts/attempt_002/status.json)
- [`results/prompt_16_homecredit_model_stability_2024/oot_final_amended_v1/final_evidence_manifest.json`](results/prompt_16_homecredit_model_stability_2024/oot_final_amended_v1/final_evidence_manifest.json)
- [`artifacts/llm_cache/`](artifacts/llm_cache/)

## 8. Verbatim prompts and complete selected-feature lists

**Third dataset:** complete prompt and ranking provenance is present. The accepted rendered prompt is [`prompt.txt`](results/prompt_16_homecredit_model_stability_2024/dev_llm_supplement_v3/llm_ranking/prompt.txt); the individual requests/responses are under the adjacent `attempts/` directories; the full accepted order is in [`ranking.csv`](results/prompt_16_homecredit_model_stability_2024/dev_llm_supplement_v3/llm_ranking/ranking.csv) and [`ranking_payload.json`](results/prompt_16_homecredit_model_stability_2024/dev_llm_supplement_v3/llm_ranking/ranking_payload.json). Fold-specific and final full-DEV selections are stored below `selection_fits/`, including [`oot_final_amended_v1/supplemental/selection_fits/`](results/prompt_16_homecredit_model_stability_2024/oot_final_amended_v1/supplemental/selection_fits/).

**Home Credit and LendingClub:** the prompt structure/version is in [`docs/llm_prompt_protocol.md`](docs/llm_prompt_protocol.md) and [`llm_screening.py`](src/credit_risk_fs/selectors/llm_screening.py). Each file in [`artifacts/llm_cache/`](artifacts/llm_cache/) preserves the selected ranking, candidate names/metadata, prompt version/hash, model and response identifiers, and raw response. However, the current artifact tree contains no verbatim rendered `prompt.txt` for these two datasets. The prompts are reproducible from the builder plus frozen inputs, but that is weaker provenance than retaining the exact rendered request text. This also means the blanket statement in `docs/llm_prompt_protocol.md` that every LLM-assisted run stores prompt text is not fully satisfied by the current older artifact tree.

**Non-LLM and hybrid selected-feature lists:** complete full-DEV and fold-level lists are present in the run artifacts, notably:

- `results/full_baseline_v1/runs/<dataset>/<run>/selected_features.csv`
- `results/full_baseline_v1/runs/<dataset>/<run>/fold_selections.csv`
- `results/selector_combinations_v1/{dev,oot}/selections/*.final_selected_features.csv`
- `results/prompt_16_homecredit_model_stability_2024/**/selection_fits/**/selected_features.csv`

This audit references those lists but deliberately does not reproduce them.

## 9. Feature availability at decision time

### Home Credit

The pipeline excludes the target and temporal/control fields listed in [`data/homecredit/metadata/leakage_columns.yaml`](data/homecredit/metadata/leakage_columns.yaml), and the global policy requires application-time alignment. That is an enforced leakage boundary, but it is not a row-by-row ledger adjudicating the availability timestamp and operational source of every retained feature. Status: **partial**.

### LendingClub v2

The preparation layer uses a centralized blacklist covering the target, post-outcome repayment variables, underwriting/policy outputs, post-approval operational fields, identifiers, and free text. [`raw_leakage_blacklist_review.md`](data/lendingclub/metadata/raw_leakage_blacklist_review.md) verifies coverage of the known risky raw fields and itself says raw direct use should remain blocked or tightly audited. It does not establish that every retained field has a source timestamp proving availability at the exact underwriting decision. Status: **strong leakage controls, but not a complete point-in-time ledger**.

### Home Credit Model Stability 2024

This dataset does have a complete raw-feature decision-time review for its frozen candidate scope. The protocol admits depth-0/depth-1 sources and excludes depth-2 sources as a declared scope decision. Within the approved scope, [`leakage_and_availability_review.csv`](cleanup/audits/third_dataset_protocol_freeze/leakage_and_availability_review.csv) adjudicates all 461 reviewed raw rows: 434 application/prior-history predictors are included, 27 target/control/identifier rows are excluded, and zero rows are unresolved. [`feature_definition_coverage.json`](cleanup/audits/third_dataset_protocol_freeze/feature_definition_coverage.json) also confirms that every included raw predictor has a definition. The adapter contract enforces the approved included/excluded scope and prevents target, identifier, and split controls from entering predictors.

The later file named [`feature_availability_filter.json`](results/prompt_16_homecredit_model_stability_2024/dev_llm_supplement_v3/feature_availability_filter.json) is a frozen missingness filter based on the earliest DEV training fold; despite its filename, it is not the decision-time availability audit. The authoritative availability evidence is the protocol review CSV and its locked copy in the third-dataset protocol.

Primary evidence:

- [`docs/leakage_policy.md`](docs/leakage_policy.md)
- [`data/homecredit/metadata/leakage_columns.yaml`](data/homecredit/metadata/leakage_columns.yaml)
- [`data/lendingclub/metadata/leakage_columns.yaml`](data/lendingclub/metadata/leakage_columns.yaml)
- [`data/lendingclub/metadata/raw_leakage_blacklist_review.md`](data/lendingclub/metadata/raw_leakage_blacklist_review.md)
- [`src/credit_risk_fs/data/homecredit_model_stability_2024/contract.py`](src/credit_risk_fs/data/homecredit_model_stability_2024/contract.py)
- [`src/credit_risk_fs/data/homecredit_model_stability_2024/adapter.py`](src/credit_risk_fs/data/homecredit_model_stability_2024/adapter.py)

## Feature-selection stability tables

These tables cover every configuration in the repository's current authenticated final registries for all three datasets. Superseded archived experiments are excluded so that values from different study versions are not mixed. These are descriptive DEV-fold feature-selection agreement measures, not predictive-performance or OOT scores.

- **Nogueira** is the corrected stability of binary feature-selection indicators over the fixed candidate universe; higher values indicate more consistent selection.
- **Mean pairwise Jaccard** is the mean intersection-over-union across every unordered pair of authenticated fold-selection sets.
- Values are rounded to four decimal places. `—` means that the controlling artifact does not publish a valid value; it is not backfilled from Kuncheva or from an older experiment.
- `Fold sets` is the number of authenticated selection sets used out of the five planned DEV folds. Results based on fewer than five sets are not directly equivalent to complete five-fold estimates.

### Home Credit

Source: [`two_dataset_results_long.csv`](cleanup/audits/prompt_14_two_dataset_oot_review_v3/two_dataset_results_long.csv). All rows use five authenticated DEV-fold selection sets.

| Dataset | Model | FS method / configuration | Nogueira | Mean pairwise Jaccard | Fold sets |
|---|---|---|---:|---:|---:|
| Home Credit | LR | `full_features` | 1.0000 | 1.0000 | 5/5 |
| Home Credit | CatBoost | `full_features` | 1.0000 | 1.0000 | 5/5 |
| Home Credit | LR | `random_k [k20]` | 1.0000 | 1.0000 | 5/5 |
| Home Credit | CatBoost | `random_k [k40]` | 1.0000 | 1.0000 | 5/5 |
| Home Credit | LR | `iv_woe [k20]` | 0.8857 | 0.8053 | 5/5 |
| Home Credit | CatBoost | `iv_woe [k40]` | 0.8458 | 0.7538 | 5/5 |
| Home Credit | LR | `mrmr_mutual_information [k20]` | 0.6882 | 0.5432 | 5/5 |
| Home Credit | CatBoost | `mrmr_mutual_information [k40]` | 0.7566 | 0.6341 | 5/5 |
| Home Credit | LR | `lasso_l1_logistic [k20]` | 0.5583 | 0.4161 | 5/5 |
| Home Credit | CatBoost | `lasso_l1_logistic [k40]` | 0.5294 | 0.4045 | 5/5 |
| Home Credit | LR | `legacy_rf_relevance_corr [k20]` | 0.8129 | 0.6999 | 5/5 |
| Home Credit | CatBoost | `legacy_rf_relevance_corr [k40]` | 0.7431 | 0.6216 | 5/5 |
| Home Credit | LR | `catboost_shap [k20]` | 0.7454 | 0.6108 | 5/5 |
| Home Credit | CatBoost | `catboost_shap [k40]` | 0.6376 | 0.5023 | 5/5 |
| Home Credit | LR | `boruta_random_forest [k20]` | 0.5687 | 0.4335 | 5/5 |
| Home Credit | CatBoost | `boruta_random_forest [k40]` | 0.5113 | 0.3832 | 5/5 |
| Home Credit | LR | `rfe_catboost [k20]` | 0.6466 | 0.4958 | 5/5 |
| Home Credit | CatBoost | `rfe_catboost [k40]` | 0.5916 | 0.4543 | 5/5 |
| Home Credit | LR | `statistical_normalized_average_rank [k20]` | — | 0.4995 | 5/5 |
| Home Credit | CatBoost | `statistical_normalized_average_rank [k40]` | — | 0.4693 | 5/5 |
| Home Credit | LR | `iv_then_boruta [pool100]` | — | 0.5221 | 5/5 |
| Home Credit | CatBoost | `iv_then_boruta [pool100]` | — | 0.5221 | 5/5 |
| Home Credit | LR | `iv_then_boruta [pool200]` | — | 0.5145 | 5/5 |
| Home Credit | CatBoost | `iv_then_boruta [pool200]` | — | 0.5145 | 5/5 |
| Home Credit | LR | `iv_then_boruta [pool300]` | — | 0.5327 | 5/5 |
| Home Credit | CatBoost | `iv_then_boruta [pool300]` | — | 0.5327 | 5/5 |
| Home Credit | LR | `boruta_then_mrmr_mutual_information [k20]` | — | 0.3593 | 5/5 |
| Home Credit | CatBoost | `boruta_then_mrmr_mutual_information [k40]` | — | 0.4204 | 5/5 |
| Home Credit | LR | `boruta_then_rfe_catboost [k20]` | — | 0.4107 | 5/5 |
| Home Credit | CatBoost | `boruta_then_rfe_catboost [k40]` | — | 0.4411 | 5/5 |
| Home Credit | LR | `cross_dataset_rank_voting_v1_primary_pool_200 [pool200]` | — | 0.4897 | 5/5 |
| Home Credit | CatBoost | `cross_dataset_rank_voting_v1_primary_pool_200 [pool200]` | — | 0.4163 | 5/5 |

### LendingClub v2

Source: [`two_dataset_results_long.csv`](cleanup/audits/prompt_14_two_dataset_oot_review_v3/two_dataset_results_long.csv). All rows use five authenticated DEV-fold selection sets.

| Dataset | Model | FS method / configuration | Nogueira | Mean pairwise Jaccard | Fold sets |
|---|---|---|---:|---:|---:|
| LendingClub v2 | LR | `full_features` | 1.0000 | 1.0000 | 5/5 |
| LendingClub v2 | CatBoost | `full_features` | 1.0000 | 1.0000 | 5/5 |
| LendingClub v2 | LR | `random_k [k20]` | 1.0000 | 1.0000 | 5/5 |
| LendingClub v2 | CatBoost | `random_k [k40]` | 1.0000 | 1.0000 | 5/5 |
| LendingClub v2 | LR | `iv_woe [k20]` | 0.9279 | 0.8719 | 5/5 |
| LendingClub v2 | CatBoost | `iv_woe [k40]` | 0.9070 | 0.8424 | 5/5 |
| LendingClub v2 | LR | `mrmr_mutual_information [k20]` | 0.8351 | 0.7329 | 5/5 |
| LendingClub v2 | CatBoost | `mrmr_mutual_information [k40]` | 0.9070 | 0.8462 | 5/5 |
| LendingClub v2 | LR | `lasso_l1_logistic [k20]` | 0.6651 | 0.5201 | 5/5 |
| LendingClub v2 | CatBoost | `lasso_l1_logistic [k40]` | 0.7024 | 0.5712 | 5/5 |
| LendingClub v2 | LR | `legacy_rf_relevance_corr [k20]` | 0.8300 | 0.7235 | 5/5 |
| LendingClub v2 | CatBoost | `legacy_rf_relevance_corr [k40]` | 0.8113 | 0.7091 | 5/5 |
| LendingClub v2 | LR | `catboost_shap [k20]` | 0.7372 | 0.5975 | 5/5 |
| LendingClub v2 | CatBoost | `catboost_shap [k40]` | 0.7343 | 0.6055 | 5/5 |
| LendingClub v2 | LR | `boruta_random_forest [k20]` | 0.9588 | 0.9273 | 5/5 |
| LendingClub v2 | CatBoost | `boruta_random_forest [k40]` | 0.8273 | 0.7293 | 5/5 |
| LendingClub v2 | LR | `rfe_catboost [k20]` | 0.5981 | 0.4418 | 5/5 |
| LendingClub v2 | CatBoost | `rfe_catboost [k40]` | 0.6067 | 0.4616 | 5/5 |
| LendingClub v2 | LR | `statistical_normalized_average_rank [k20]` | — | 0.5826 | 5/5 |
| LendingClub v2 | CatBoost | `statistical_normalized_average_rank [k40]` | — | 0.5560 | 5/5 |
| LendingClub v2 | LR | `iv_then_boruta [pool100]` | — | 0.8437 | 5/5 |
| LendingClub v2 | CatBoost | `iv_then_boruta [pool100]` | — | 0.8437 | 5/5 |
| LendingClub v2 | LR | `iv_then_boruta [pool200]` | — | 0.8073 | 5/5 |
| LendingClub v2 | CatBoost | `iv_then_boruta [pool200]` | — | 0.8073 | 5/5 |
| LendingClub v2 | LR | `iv_then_boruta [pool300]` | — | 0.7901 | 5/5 |
| LendingClub v2 | CatBoost | `iv_then_boruta [pool300]` | — | 0.7901 | 5/5 |
| LendingClub v2 | LR | `boruta_then_mrmr_mutual_information [k20]` | — | 0.3966 | 5/5 |
| LendingClub v2 | CatBoost | `boruta_then_mrmr_mutual_information [k40]` | — | 0.4339 | 5/5 |
| LendingClub v2 | LR | `boruta_then_rfe_catboost [k20]` | — | 0.4333 | 5/5 |
| LendingClub v2 | CatBoost | `boruta_then_rfe_catboost [k40]` | — | 0.4753 | 5/5 |
| LendingClub v2 | LR | `cross_dataset_rank_voting_v1_primary_pool_200 [pool200]` | — | 0.4378 | 5/5 |
| LendingClub v2 | CatBoost | `cross_dataset_rank_voting_v1_primary_pool_200 [pool200]` | — | 0.4567 | 5/5 |

### Home Credit Model Stability 2024

Metric source: [`oot_analysis_plan.json`](cleanup/audits/prompt_16_final_amended_oot/oot_analysis_plan.json). Configuration-label source: [`final_34_cell_oot_registry.json`](cleanup/audits/prompt_16_final_amended_oot/final_34_cell_oot_registry.json).

| Dataset | Model | FS method / configuration | Nogueira | Mean pairwise Jaccard | Fold sets |
|---|---|---|---:|---:|---:|
| Home Credit Model Stability 2024 | LR | `full_features` | 1.0000 | 1.0000 | 5/5 |
| Home Credit Model Stability 2024 | CatBoost | `full_features` | 1.0000 | 1.0000 | 5/5 |
| Home Credit Model Stability 2024 | LR | `random_k [k20]` | 1.0000 | 1.0000 | 5/5 |
| Home Credit Model Stability 2024 | CatBoost | `random_k [k40]` | 1.0000 | 1.0000 | 5/5 |
| Home Credit Model Stability 2024 | LR | `iv_woe [k20]` | 0.7878 | 0.6666 | 5/5 |
| Home Credit Model Stability 2024 | CatBoost | `iv_woe [k40]` | 0.7831 | 0.6582 | 5/5 |
| Home Credit Model Stability 2024 | LR | `mrmr_mutual_information [k20]` | 0.4595 | 0.3083 | 5/5 |
| Home Credit Model Stability 2024 | CatBoost | `mrmr_mutual_information [k40]` | 0.5406 | 0.3912 | 5/5 |
| Home Credit Model Stability 2024 | LR | `lasso_l1_logistic [k20]` | — | — | 1/5 |
| Home Credit Model Stability 2024 | CatBoost | `lasso_l1_logistic [k40]` | — | — | 1/5 |
| Home Credit Model Stability 2024 | LR | `legacy_rf_relevance_corr [k20]` | 0.6716 | 0.5182 | 5/5 |
| Home Credit Model Stability 2024 | CatBoost | `legacy_rf_relevance_corr [k40]` | 0.6197 | 0.4687 | 5/5 |
| Home Credit Model Stability 2024 | LR | `catboost_shap [k20]` | 0.7474 | 0.6058 | 5/5 |
| Home Credit Model Stability 2024 | CatBoost | `catboost_shap [k40]` | 0.6733 | 0.5166 | 5/5 |
| Home Credit Model Stability 2024 | LR | `boruta_random_forest [k20]` | 0.5959 | 0.4454 | 3/5 |
| Home Credit Model Stability 2024 | CatBoost | `boruta_random_forest [k40]` | 0.6087 | 0.4688 | 3/5 |
| Home Credit Model Stability 2024 | LR | `rfe_catboost [k20]` | 0.5117 | 0.3572 | 4/5 |
| Home Credit Model Stability 2024 | CatBoost | `rfe_catboost [k40]` | 0.5746 | 0.4157 | 4/5 |
| Home Credit Model Stability 2024 | LR | `statistical_normalized_average_rank [k20]` | — | — | 1/5 |
| Home Credit Model Stability 2024 | CatBoost | `statistical_normalized_average_rank [k40]` | — | — | 1/5 |
| Home Credit Model Stability 2024 | LR | `iv_then_boruta [pool100]` | 0.7465 | 0.6165 | 5/5 |
| Home Credit Model Stability 2024 | CatBoost | `iv_then_boruta [pool100]` | 0.7465 | 0.6165 | 5/5 |
| Home Credit Model Stability 2024 | LR | `iv_then_boruta [pool200]` | 0.7358 | 0.6064 | 5/5 |
| Home Credit Model Stability 2024 | CatBoost | `iv_then_boruta [pool200]` | 0.7358 | 0.6064 | 5/5 |
| Home Credit Model Stability 2024 | LR | `iv_then_boruta [pool300]` | 0.7124 | 0.5877 | 5/5 |
| Home Credit Model Stability 2024 | CatBoost | `iv_then_boruta [pool300]` | 0.7124 | 0.5877 | 5/5 |
| Home Credit Model Stability 2024 | LR | `boruta_then_mrmr_mutual_information [k20]` | 0.3265 | 0.2037 | 3/5 |
| Home Credit Model Stability 2024 | CatBoost | `boruta_then_mrmr_mutual_information [k40]` | 0.4726 | 0.3260 | 3/5 |
| Home Credit Model Stability 2024 | LR | `boruta_then_rfe_catboost [k20]` | 0.4275 | 0.2805 | 3/5 |
| Home Credit Model Stability 2024 | CatBoost | `boruta_then_rfe_catboost [k40]` | 0.5406 | 0.3827 | 3/5 |
| Home Credit Model Stability 2024 | LR | `llm [k20]` | 1.0000 | 1.0000 | 5/5 |
| Home Credit Model Stability 2024 | CatBoost | `llm [k40]` | 1.0000 | 1.0000 | 5/5 |
| Home Credit Model Stability 2024 | LR | `stable_core_llm_fill [k20]` | 0.7554 | 0.6259 | 5/5 |
| Home Credit Model Stability 2024 | CatBoost | `stable_core_llm_fill [k40]` | 0.7091 | 0.5727 | 5/5 |

For the third dataset, incomplete fold coverage is an evidence limitation rather than a zero-stability result. In addition, the pure `llm` ranking is intentionally shared across folds, so its 1.0000 values measure mechanical repeatability of that fixed ranking, not stability across independently refitted fold-specific selectors. The 1.0000 values for `full_features` and `random_k` likewise reflect deterministic identical selections and should not be read as evidence of predictive quality.
