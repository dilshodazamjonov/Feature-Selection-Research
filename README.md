# Feature Selection Research

Repository for the manuscript **“Language-Model Feature Pre-Screening for Credit Scoring: A Frozen Final Scoring Evaluation with Temporal and Recency-Shift Holdouts.”** The study compares semantic, statistical, hybrid, and control feature-selection strategies under fixed logistic-regression and CatBoost backbones.

**Status:** Artefacts for the manuscript submitted to *Big Data Research*.

This README describes the final manuscript protocol. The checkout also contains code, configurations, and local result directories from earlier protocol stages. The [repository snapshot](#repository-snapshot-and-protocol-history) section identifies material differences; an existing path is not, by itself, evidence that it implements the final manuscript protocol.

## Datasets and DEV/HO design

`DEV` denotes development data and `HO` the untouched final held-out population.

| Dataset | Original features | Candidate features | DEV n | HO n | Ordering and split |
|---|---:|---:|---:|---:|---|
| Home Credit | 529 | 373 | 99,092 | 120,053 | Prior-application recency using `DAYS_DECISION`; HO is a recency-shift holdout |
| LendingClub v2 | 675 | 675 | 598,649 | 293,105 | Issue month; later issue months form HO |
| Stability 2024 | 1,959 | 1,068 | 1,157,512 | 369,147 | `date_decision`; HO begins at the 2020-02-01 cutoff |

Home Credit is not a conventional time-based out-of-time split: ordering by prior-application recency induces a recency-shift holdout. LendingClub v2 and Stability 2024 use later issue months and later `date_decision` observations, respectively. The remainder of this document therefore uses DEV and HO for all three datasets.

Development uses five expanding-window folds along the same dataset-specific ordering index. One index unit is omitted between each training and validation segment, and Stability 2024 keeps all decisions sharing a `date_decision` together. Selector choices and budgets are made on DEV only; HO does not influence them.

### Stability 2024 feature construction

The construction reviewed 461 source variables and approved 434 raw predictors. Depth 0 includes `static_0` and `static_cb_0`; depth 1 includes `applprev_1`, `tax_registry_a_1`, `tax_registry_b_1`, `tax_registry_c_1`, `credit_bureau_a_1`, `credit_bureau_b_1`, `other_1`, `person_1`, `deposit_1`, and `debitcard_1`. Depth-2 sources, ratios, and interactions are excluded. Dates are expressed relative to `date_decision`.

Numeric and date depth-1 fields are summarized with count, mean, minimum, maximum, standard deviation, and the chronologically latest value. Categorical fields use the mode, most-recent category, or distinct-count summary as appropriate. This produces 1,959 engineered features with retained lineage. A 90% missingness screen fixed from the earliest development fold leaves 1,068 candidates; the same screen is then reused rather than relearned from later observations.

## Methods

Full features is the reference condition, not a feature selector. “Local support” below records verified code in this checkout; it does not assert exact parity with the final manuscript where a qualification is shown.

| Paper name | Family / role | Uses target labels? | Verified local support |
|---|---|---:|---|
| Full features | Reference | No | `full_features` in `src/credit_risk_fs/selectors/lightweight/registry.py` |
| Domain rules | Semantic control | No | `domain_rule_baseline` in `src/credit_risk_fs/selectors/registry.py` |
| Random K | Random control | No | `random_k` in the lightweight registry |
| PCA | Unsupervised projection | No | `pca` in the main selector registry |
| IV/WOE | Filter | Yes | `iv_woe`; local standalone code ranks by IV but does not encode the manuscript’s accepted-IV interval as its selection rule |
| mRMR | MI relevance/redundancy selector | Yes | Exact KSG/Ross estimator absent; see the distinction below |
| Boruta RF | All-relevant selector | Yes | `boruta_random_forest` |
| IV then Boruta | Sequential hybrid | Yes | `iv_then_boruta` |
| RFE CatBoost | Recursive elimination | Yes | `rfe_catboost` |
| CatBoost SHAP | Model-importance selector | Yes | `catboost_shap` |
| Pure LLM | Definition-only semantic ranking | No | LLM ranking code exists; exact cache coverage differs by dataset |
| LLM then mRMR | Semantic shortlist, then MI mRMR | Statistical stage only | The local `llm_then_mrmr` route uses the legacy RF/correlation filter, not paper mRMR |
| LLM then Boruta | Semantic shortlist, then Boruta | Statistical stage only | `llm_then_boruta` |
| Stable core + LLM fill | Resampled statistical core with semantic fill | Stable-core stage only | `stable_core_llm_fill` |
| L1 logistic selector | Post-hoc / secondary selector | Yes | `lasso_l1_logistic` |

The manuscript’s **mRMR** is the mutual-information method defined by Eq. (2): KSG-style mutual-information estimation is used between continuous features, and Ross-style estimation is used for a continuous feature against the binary target. Standalone mRMR uses the full candidate pool; after an LLM ranking it uses a shortlist of 60 candidates for logistic regression and 100 for CatBoost.

Two local names must not be conflated with that method:

- The historical registry alias `mrmr` resolves to `RandomForestRelevanceMRMRSelector`. This **RF/correlation filter** divides random-forest impurity relevance by mean absolute Pearson redundancy, applies a redundancy floor of 0.05, and ranks greedily.
- The newer local key `mrmr_mutual_information` uses quantile discretization and `sklearn.metrics.mutual_info_score`, not the manuscript’s KSG/Ross estimators.

Other frozen selector definitions are:

- **IV/WOE:** ten quantile bins and accepted interval `0.01 <= IV <= 0.50`; it is also the first stage of IV then Boruta.
- **Boruta RF:** 500 trees, depth 6, alpha 0.05, two-step correction, confirmed features only, capped at K. Natural support below K is not padded.
- **RFE CatBoost:** remove 20% per step; the internal ranker uses 500 iterations, depth 6, and learning rate 0.05; stop at K.
- **CatBoost SHAP:** rank by aggregate absolute SHAP value on a 10,000-row training sample and retain the top K.
- **Random K:** seeded uniform sampling without replacement.
- **L1 logistic selector:** standardized median-imputed inputs, ordinal-coded categoricals, `liblinear`, `C = 0.05`, balanced class weights, and the K largest non-zero coefficients. It does not pad a below-budget support, and the chosen subset is refit with the frozen backbone.

## Frozen modelling protocol

Feature budgets count original variables: `K = 20` for logistic regression and `K = 40` for CatBoost. The one exception is LendingClub CatBoost with LLM then mRMR, where the budget counts one-hot columns. Budgets are frozen before HO scoring.

| Backbone | Frozen configuration |
|---|---|
| Logistic regression | `liblinear`; L2; `C = 1`; `max_iter = 1000`; balanced class weighting; seed 42 |
| CatBoost | depth 10; learning rate 0.01; `l2_leaf_reg = 95`; `min_data_in_leaf = 290`; `rsm = 0.90`; `random_strength = 0.125`; grow policy `Depthwise`; bootstrap type `Bernoulli`; subsample 0.55; loss `Logloss`; balanced class weighting; seed 42; four threads; 1,500 iterations; no early stopping |

The repository’s CatBoost configuration spells `rsm` as `colsample_bylevel`. It also retains an `early_stopping_rounds` field from earlier stages, but the relevant frozen runs supply no validation set to CatBoost, so that setting is inactive and all 1,500 iterations run. The final manuscript specification is the no-early-stopping configuration shown above.

## LLM ranking protocol

The experiment used `gpt-4.1-mini-2025-04-14` through Chat Completions at temperature 0 with a strict JSON response format. Inputs were definition-only: no target values, HO labels or statistics, IV, correlation, mutual information, SHAP, model importance, or validation performance entered the semantic ranking. Responses were checked for exact size, uniqueness, membership in the candidate universe, and model identity. A failed response could be retried up to three times; there was no fallback selector.

The call structure was:

- **Home Credit:** six calls returning 100 names each: five DEV-fold calls and one full-DEV call.
- **LendingClub v2:** 24 calls: six partitions (five folds plus full DEV) crossed with budgets `{20, 40, 60, 100}`.
- **Stability 2024:** one main full-DEV call returning 100 names, plus a repeated-call analysis of ten identical full-DEV requests.

The reproducible artefact is the accepted cached response, not a fresh API call. Temperature 0 does not guarantee deterministic API output. Fresh calls require `OPENAI_API_KEY`, may return different rankings, and must never log or commit the key.

In this checkout, `artifacts/llm_cache/` is empty, so the Home Credit and LendingClub cached rankings are unavailable through that advertised cache path. The ignored local Stability result tree does contain its accepted request/response, prompt, ranking, and manifest under `results/prompt_16_homecredit_model_stability_2024/dev_llm_supplement_v3/llm_ranking/`. Cache-only reproduction is therefore incomplete for the final three-dataset protocol in this snapshot.

## Leakage and reproducibility safeguards

- The candidate universe is fixed without HO outcomes; Stability’s missingness state comes from the earliest DEV fold.
- Selector fitting, tuning, fold comparisons, and budget decisions use DEV only.
- Final feature budgets, preprocessing maps, and subset identities are frozen before HO scoring where corresponding manifests exist.
- Semantic ranking inputs are separated from HO labels and statistics.
- LLM reproducibility is tied to cached accepted responses, not assumed API determinism.
- HO is reserved for final evaluation and predefined sub-population re-scoring, not selector tuning.

These controls reduce leakage risk but do not make claims beyond the recorded data definitions, lineage, and manifests.

## Supplementary analyses

**Diversity controls.** A family-capped classical control fills in the base selector’s order with cap `ceil(K/F)` and no artificial padding beyond that rule.

| Dataset | Families F | LR cap | CatBoost cap |
|---|---:|---:|---:|
| Home Credit | 6 | 4 | 7 |
| LendingClub v2 | 17 | 2 | 3 |
| Stability 2024 | 12 | 2 | 4 |

The Stability depth-0 control restricts candidates to `static_0` and `static_cb_0`.

**Semantic-input controls.** In obfuscation Arm A, names are replaced by `F001...` using seeded permutation 20260914 and literal names are scrubbed from descriptions. Arm B keeps names but blanks descriptions. Each arm uses six calls, after which decoded rankings are evaluated through the frozen pipeline. A lineage-only condition constructs descriptions from raw table, raw variable, depth, operation, window, and type rule without semantic credit meaning. Stability’s repeated-call analysis sends the identical full-DEV request ten times and propagates every accepted response through the frozen pipeline.

**Rank voting.** This is supplementary, not a main selector family. Ranks are normalized as `(r - 1) / (d - 1)` and averaged across the RF/correlation filter, Boruta, and RFE for pools `P in {100, 200, 300}`, followed by selection of K. The reference is the RF/correlation filter on the full candidate pool. Inference uses paired DeLong and a paired, target-stratified percentile bootstrap with 2,000 resamples and seed 20260721; Holm correction is applied within each case’s three voting tests.

The checked-in `configs/protocols/cross_dataset_rank_voting_v1.yaml` is related earlier work with a different voter construction. It is not the exact Appendix B protocol just described and is not cited as a preregistration for it.

**Predefined HO sub-populations.** LendingClub is re-scored for the matured 36-month cohort issued from 2016-01 through 2016-05 (`n = 123,898`), the 36-month (`n = 232,361`) and 60-month (`n = 60,744`) term subsets, and twelve issue-month slices. Stability 2024 is re-scored by month from February through October 2020. These analyses reuse frozen HO predictions; they do not refit selectors.

## Evaluation metrics

Primary and secondary evaluation includes ROC-AUC, KS, Brier score, log loss, Capture@10, and Lift@10. Diagnostics include type-aware feature PSI using frozen DEV states, effective dimension after encoding, and subset stability measured by Nogueira stability plus mean/minimum/maximum pairwise Jaccard. The DEV budget sweep evaluates `K in {10, 20, 30, 40, 50}`.

## Reproduction

The manuscript environment is Python 3.13.5, scikit-learn 1.8.0, CatBoost 1.2.10, and Boruta 0.4.3. `uv.lock` pins the three library versions; `.python-version` specifies Python 3.13 but not its patch release.

```bash
uv sync
uv run python scripts/run_matrix.py --dataset homecredit --dry-run
uv run python scripts/run_matrix.py --dataset lendingclub_v2 --dry-run
uv run python scripts/run_prompt_16_third_dataset.py --help
uv run python scripts/run_prompt_16_final_oot.py --help
```

These are verified repository entry points. The two dry runs expose the currently configured Home Credit and LendingClub matrices; the Stability commands require authenticated plan or authorization files. Because the checked-in protocol/configuration set does not encode every final-manuscript change, these commands document and exercise the checkout but are not a one-command exact reproduction of the final manuscript.

Reported experiments ran CPU-only on an Intel Core i7-13620H with 40 GiB RAM and four estimator threads. Reported peak process memory was 7.4–10.7 GB for Home Credit, 19.1–29.0 GB for LendingClub, and 27.0–29.0 GB for Stability 2024; approximately 32 GB RAM is advisable for the two larger datasets.

Seeds are 42 for classifiers/selectors, 42–46 for stable-core resamples, 20260721 for the inference bootstrap, and 20260914 for the obfuscation permutation.

Runtime values are descriptive because not all runs were measured under matched conditions: Home Credit logistic regression took 0.8 minutes for Pure LLM, 12.1 minutes for the RF/correlation filter, and 24.6 minutes for stable core; reported LendingClub CatBoost selector/run times ranged from 49.7 to 256.0 minutes. In the matched Stability fold-1 comparison, mRMR over 1,959 features took 522 seconds versus 21 seconds after a 100-feature shortlist.

## Paper-to-repository map

| Manuscript material | Verified repository support | Limitation |
|---|---|---|
| Table 1: methods | `src/credit_risk_fs/selectors/`, its two registries, and `configs/experiments/full_baseline_v1.yaml` | Local MI-mRMR is discretized rather than KSG/Ross; several local hybrid routes retain the RF/correlation method |
| Table 2: protocol versions | `configs/protocols/` and `docs/research_extension/` | These record multiple earlier stages, not a single consolidated final-manuscript protocol |
| Table 3: datasets | `configs/datasets/`, `src/credit_risk_fs/data/`, and the Stability adapter/contract package | The checked-in Stability lock uses an earlier split; see below |
| Core two-dataset selectors and controls | `src/credit_risk_fs/experiments/full_baseline.py` and `src/credit_risk_fs/experiments/selector_combinations.py` | Current configs retain earlier modelling details |
| Stability construction and sealed local evidence | `src/credit_risk_fs/data/homecredit_model_stability_2024/` and the local ignored `results/prompt_16_homecredit_model_stability_2024/` tree | Local sealed evidence uses the earlier boundary and row counts |
| Appendix A: LLM prompt mechanics | `src/credit_risk_fs/selectors/llm_screening.py`, `src/credit_risk_fs/experiments/prompt_16_llm_supplement.py`, and the local Stability ranking directory | Home Credit/LendingClub accepted caches are absent from `artifacts/llm_cache/` |
| Appendix B: voting machinery | `src/credit_risk_fs/experiments/rank_voting.py` and `src/credit_risk_fs/analysis/voting_inference/` | The checked-in voting protocol is an earlier, different design |
| Supplementary controls | `scripts/todo_fill/gate4/` | Generated `todo_done/` artefacts are local and ignored; several recorded cells remain partial |

The prompt-proposed consolidated paths such as `results/ho_metrics.csv`, `results/dev_fold_metrics.csv`, `results/headline_subsets.csv`, `results/stability.csv`, `results/budget_curves.csv`, and `results/subpopulations/*.csv` are absent. The existing `results/run_index.csv` and all current `results/` contents are ignored by Git, so they are local working-tree evidence rather than files guaranteed in a fresh clone.

## Repository layout

- `src/credit_risk_fs/` — data adapters, feature engineering, selectors, backbones, experiment runners, evaluation, and reporting.
- `scripts/` — preparation, execution, audit, and local evidence-assembly entry points.
- `configs/` — dataset, model, selector, experiment, execution, and protocol-stage configurations.
- `docs/` — pipeline notes and earlier research-extension records.
- `tests/` — unit and contract tests; no test suite is required merely to inspect this README.
- `data/`, `artifacts/`, `outputs/`, `results/`, and `todo_done/` — ignored local data, caches, working state, or generated evidence; their contents are not portable repository guarantees.

## Repository snapshot and protocol history

The final manuscript protocol is not represented by one exact executable configuration in this checkout. In particular:

- `configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json` and the local Stability results use a 2020-02-26 boundary with DEV/HO counts 1,221,743/304,916, rather than the manuscript’s 2020-02-01 boundary and 1,157,512/369,147 counts.
- No tracked manifest consolidates the final candidate counts 373/675/1,068 under the manuscript splits; an older local synthesis instead records screened counts 391/161/1,068.
- The local selector named `mrmr_mutual_information` uses a discretized plug-in estimator; the paper uses KSG/Ross-style MI estimation. The legacy `mrmr` key is the separate RF/correlation filter.
- The standalone local IV selector ranks top K rather than enforcing the final accepted-IV interval, and the local L1 default/config does not establish the manuscript’s balanced-weight setting.
- Home Credit and LendingClub accepted LLM cache files are absent from `artifacts/llm_cache/`.
- The local repeated-call log covers Home Credit and an incomplete LendingClub set, not the manuscript’s ten Stability repeats.
- No consolidated final-manuscript result package maps one-to-one to Tables 4–14 or Appendices C–D.

Local `results/final_research_package_v2/` and `results/final_three_dataset_synthesis_v1/` are earlier report packages. The latter uses the earlier Stability split and includes a CLIP-ranked row. CLIP and contrastive code under `src/credit_risk_fs/clip/`, `configs/clip*`, and `docs/clip/` belongs to earlier research lines and is not reported in the current manuscript. Legacy directory names may retain the term `oot`; they should be read as historical path names, not as the terminology of the final protocol.

## Data and licensing

The three datasets are third-party public/Kaggle data and are not tracked by this repository; `/data/` is ignored by Git. Obtain each dataset from its original provider and follow the provider’s terms. No repository-level `LICENSE` file is present in this snapshot, so the repository does not grant a software license by implication.

## Citation / manuscript reference

Please cite the manuscript by its current title:

> *Language-Model Feature Pre-Screening for Credit Scoring: A Frozen Final Scoring Evaluation with Temporal and Recency-Shift Holdouts.* Manuscript submitted to *Big Data Research*.

Formal citation metadata (`CITATION.cff`, DOI, volume, issue, and pages) is not included because the manuscript is submitted, not accepted or published.
