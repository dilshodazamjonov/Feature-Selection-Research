# Repository evidence for points 1–11

Prepared 10 September 2026, revision 2, from the repository at commit `90c3c4b8640de23c489129c0730cd1b64164335e`, its Git history and the recovered local research backups. No model, selector, LLM call, embedding, bootstrap test, DeLong test, hyperparameter search or robustness experiment was run. Additional calculations are limited to existing predictions, saved selections, source columns and recorded test values.

`00_source_provenance.csv` records source locations and hashes, including backup and archive members. Original run values, later manuscript/supplied aggregate values, and historical configurations retain their separate provenance. The Pure LLM method description in point 2 states the confirmed full-feature method; archived screened-run artifacts remain separately identified in the CSVs. No full-pool predictions were synthesized from screened-run records.

Blank cells mean unavailable or inapplicable, not zero. CSVs use UTF-8 with BOM. Source files and borrower-level records are not bundled. The package contains one Markdown file and CSV evidence for all eleven points.

## 1. Results, run matching and the five final Stability folds

[01_final_results.csv](01_final_results.csv) contains the **160 primary method/configuration records**: 32 original Home Credit/LendingClub v2 matrix records, 64 classical-extension/review records, 34 final Stability records including unavailable cells, four corrected Home Credit CLIP records, six Stability CLIP records, eight historical voting records and 12 additional prospective voting/reference records. The four prospective primary P=200 records are already among the 64 review records.

The table supplies DEV OOF AUC, OOT AUC, KS, Brier, Capture@10, score PSI, actual saved selected-column count, Nogueira and Jaccard wherever supported. KS, Brier and Capture@10 concern OOT. The historical selected-column count and unique source-feature count are separate where categorical encoding expanded a source variable. PCA counts components. Full-feature Stability has saved 1,959-feature selections but unavailable final models.

Original summaries for all 32 matrix and four corrected Home Credit CLIP runs have been recovered. Their exact values replace rounded or withheld historical-export values. [01_recovered_run_metric_reconciliation.csv](01_recovered_run_metric_reconciliation.csv) records 396 metric comparisons with the earlier export. For example, the original Home Credit LR mRMR run `lr_statistical_mrmr_53a793cb32fe` has OOT AUC **0.7456889163655954**. Later supplied/manuscript aggregate scores remain in `01_existing_supplied_aggregate_tables.csv`; they are not substituted into that run's paired predictions or other metrics.

[01_results_primary_and_additional_historical.csv](01_results_primary_and_additional_historical.csv) adds **60 distinct historical metric records** to the 160 primary records, for **220 rows**. `01_all_recovered_backup_experiment_summaries.csv` retains all **148 recovered summary copies**, including duplicate backups and earlier LendingClub v1/CLIP v2 runs. Their reported stability values are explicitly historical and have not been given the primary runs' recalculation status. The older configurations remain labelled separately.

Pooled DEV OOF AUC is calculated only from saved held-out predictions. `dev_fold_auc_mean` is distinct. Original matrix/corrected Home Credit CLIP run directories retain fold metrics but no pooled OOF probability file; the recovered fold metrics are in `01_recovered_historical_DEV_fold_metrics.csv`. The 36 full-baseline DEV probability files are in-sample and do not supply OOF AUC. Partial Stability OOF results identify the completed fold count. All 170 final registered Stability DEV cells and statuses are in `01_stability_dev_cell_accounting.csv`.

The five final Stability validation-fold sizes are **204,567; 203,798; 205,980; 201,466; 202,820**, totaling **1,018,631**. [01_stability_validation_fold_sizes.csv](01_stability_validation_fold_sizes.csv) includes training sizes, event counts, dates and authenticated row hashes. The locked dates and five v3 fold manifests agree.

Historical/full-baseline and CLIP score PSI use full-DEV final-model in-sample scores; combination/prospective voting uses DEV OOF; final Stability uses its available OOF folds and frozen bins. The source reference is recorded per run. Missing requested metrics are listed in `01_missing_metric_evidence.csv`.

## 2. Pure LLM eligibility and prompt data

**Pure LLM received the whole engineered predictor feature set, with no IV prescreening: 529 Home Credit features and 675 LendingClub v2 features.** The 529/675 counts are the Pure LLM candidate sizes. The 391/161 counts do not describe this Pure LLM method.

The repository's partition convention is **fold TRAIN for DEV validation and full DEV for final selection**. Validation/OOT rows do not supply screening or prompt statistics. Matching full-pool prompt-statistics logs are not present in the recovered artifacts. [02_Pure_LLM_method_definition.csv](02_Pure_LLM_method_definition.csv) records the full-pool definition and the partition evidence separately.

The 391/161 records are retained as an **archived screened implementation**, with their original cache and screening provenance in `02_llm_screening_and_cache_inventory.csv`, `02_llm_eligible_feature_membership.csv` and `02_archived_screened_variant_reduction_ledger.csv`. Their metrics are not authenticated as full-pool Pure LLM predictions. This distinction also appears in the primary results table's `method_definition_scope` column.

Final Stability v3 has a separate saved contract: no IV screening; 1,068 of 1,959 predictors pass missingness <=0.90 on the earliest DEV training fold, 200,661 rows from 1 January through 28 March 2019. Its prompt uses names/descriptions and omits row statistics, labels, IV and OOT. One sealed ranking supplies K=20/40 across folds and OOT. `02_stability_eligibility_settings.csv` preserves this contract.

## 3. Actual and recalculated Nogueira universes

[03_nogueira_universes.csv](03_nogueira_universes.csv) identifies each primary result's universe, fold coverage and calculation status. The original reported scalar is retained separately.

| Result scope | Universe |
| --- | ---: |
| Home Credit original matrix and classical/prospective analyses | 529 engineered predictors |
| LendingClub v2 original matrix and classical/prospective analyses | 675 engineered predictors |
| Final Stability classical methods | 1,959 engineered predictors |
| Final Stability LLM/stable-core supplement v3 | 1,068 frozen eligible predictors |
| Home Credit corrected CLIP fixed-pool downstream selection, LR / CatBoost | 60 / 100 source features |
| Stability CLIP fixed-pool downstream selection, LR / CatBoost | 60 / 100 source features |

**391/161 are not Nogueira denominators for the original matrix.** Those historical calculations used 529/675. The corrected Home Credit run summaries also used 529; the revised main table instead supplies conditional stability within their authenticated fixed 60/100 source-feature pools. Whole-pipeline P529 values remain in `03_historical_pool_membership_audit.csv` for comparison on that explicitly stated universe.

The recalculated fixed-pool Home Credit values are **0.797500** for LR CLIP→mRMR, **0.760000** for LR LLM→CLIP→mRMR, **0.816667** for CatBoost CLIP→mRMR, and **0.808333** for CatBoost LLM→CLIP→mRMR. All five saved pool manifests agree within each corrected run. `03_historical_fold_candidate_pool_members.csv` contains their exact membership.

Historical categorical model columns are mapped to their original source variables before original-feature stability is calculated. The exact mapping rule and names are in `07_historical_model_column_to_source_feature.csv`; saved scalar/model-column values remain in `reported_nogueira` and `reported_jaccard`. Source mapping uses exact names, otherwise the longest original-feature prefix with the repository's categorical separator rule. PCA has no such mapping.

Original LLM-hybrid fold candidate lists vary. Some saved selections are also outside the recovered fold candidate lists; `03_historical_pool_selection_mismatches.csv` identifies each case. A common conditional pool is not authenticated for those runs. Their reported P529/P675 and recalculated whole-pipeline source-feature values are retained with that scope.

Nogueira is `1 - sum_j Var_sample(Z_j) / [mean_k*(1-mean_k/P)]`. Jaccard averages all unordered pairs of available fold sets. Full-feature selection has a zero denominator; the repository's value 1 is explicitly a convention. Historical voting has no saved executed DEV-fold selection stability under its original protocol.

## 4. Stability full-feature and CLIP reference-run configuration

[04_reference_run_details.csv](04_reference_run_details.csv) identifies each saved full-feature reference and all six Stability CLIP results, with exact split, metrics and configuration source. The Stability split is **DEV 2019-01-01 through 2020-02-25; OOT 2020-02-26 through 2020-10-05**, with **1,221,743 DEV** and **304,916 OOT** rows.

The full-feature Stability LR/CatBoost configurations use 1,959 predictors and the frozen final-model settings. Both final OOT entries say **`unavailable`**, reason **`inherited_from_all_five_authenticated_dev_resource_stops`**. There is no completed numerical full-feature OOT baseline under this final registry. The saved selection of all variables remains distinct from completion of a model.

Stability CLIP uses that same **2020-02-26** cutoff, 1,959 ranked identities, fixed pools 60/100, RF-relevance/correlation-redundancy mRMR, K=20/40, and the saved sparse model preprocessing. Its six rows belong to `stability_clip_experiment_v1`.

**Matching 2020-02-01 baseline or CLIP results were not found** in the inspected repository, local Git history, uncompressed backup metadata and archive metadata. The protocol's occurrence of `2020-02-01` is a monthly population `date_min`, not the train/OOT cutoff. The exported references retain the explicit **historical 2020-02-26 cutoff** label. `04_requested_cutoff_search.csv` and `04_expanded_search_scope_and_results.csv` record this gap. The Git search traces the February 1 monthly-date entry to commit `e2a9a80`. Searches covered both research backup roots, two RAR archives and the scientific-cleanup ZIP; no matching February 1 result was recovered.

## 5. Voting tests and the claimed 24 comparisons

[05_historical_voting_24_inventory.csv](05_historical_voting_24_inventory.csv) identifies **24 historically executed comparisons**: two models × four voting variants × three comparators. The exact variants are **`voting_llm_clip_top100_then_mrmr`**, **`voting_llm_clip_top200_then_mrmr`**, **`voting_llm_clip_top300_then_mrmr`**, and **`voting_llm_clip_mrmr_three_voter_direct`**. The first three refine an LLM+CLIP pool with RF/correlation mRMR; the fourth is recorded as direct three-voter selection. The comparators are **`statistical_mrmr`**, **`hybrid_llm_then_mrmr`**, and **`llm_corrected_clip_then_mrmr`**, with exact saved run IDs in the CSV. The historical mRMR implementation uses random-forest impurity relevance and absolute Pearson-correlation redundancy; it is not the later mutual-information implementation.

The retained test inventory records 120,053 aligned Home Credit OOT borrowers per pair, zero unmatched rows and zero target mismatches. AUC p-values use two-sided paired DeLong. Confidence intervals use paired stratified bootstrap, **2,000 repetitions, seed 20260706**. Holm adjusts the **24 AUC p-values together**. KS confidence intervals were calculated, but no KS p-value test is recorded in this family.

[05_historical_voting_24_results.csv](05_historical_voting_24_results.csv) now contains **all 24 original numerical results**, including all four previously missing LR-versus-mRMR comparisons. It includes both AUCs, AUC/KS paired differences, 95% paired bootstrap AUC/KS confidence intervals, raw two-sided DeLong p-values and the original 24-test Holm-adjusted p-values. All 24 comparisons have 120,053 aligned borrowers, no unmatched IDs and no target mismatch.

The recovered manifest identifies `prompt5_powered_comparisons_v1`, created **2026-07-09T19:04:41+00:00**, commit `0fdd97a663275a2ccb3f035515dbc4189f666fbc`. All **80 checked source, prediction and generated-file hashes match**. AUCs recomputed from the existing paired scores match all 24 saved pairs within 1e-12. Holm arithmetic over the original 24 raw p-values also matches. No bootstrap or DeLong test was rerun. Checks are in `05_recovered_voting_manifest_hash_checks.csv` and `05_recovered_voting_prediction_checks.csv`.

For the previously missing `voting_01` LR P100 comparison, voting AUC is **0.7394324820732968**, original mRMR AUC **0.7456889163655955**, difference **−0.0062564342922987**, 95% CI **[−0.0080117829335841, −0.0045415534869432]**, raw p **6.17709227087134e−12**, Holm p **1.235418454174268e−10**. The remaining 23 pairs are fully enumerated in the CSV. These are the original comparisons against their saved comparator predictions.

The later prospective analysis is a **different 12-test study**, fully available in [05_current_cross_dataset_voting_12_tests.csv](05_current_cross_dataset_voting_12_tests.csv). Its voters are RF/correlation mRMR and Boruta, equal-weight normalized scores `1-(rank-1)/max(N-1,1)`, missing score zero, followed by CatBoost RFE. It excludes LLM/CLIP voters. It compares P=100/200/300 against a full-pool RF/correlation reference on both datasets and models. Holm uses **four separate families of three tests**; bootstrap seed is **20260721**, 2,000 repetitions. This CSV includes exact AUCs, deltas, raw/adjusted p-values, and AUC/KS/lift confidence intervals. The root six-row supplied inference table is preserved separately as aggregate evidence, not merged into either voting family.


`05_historical_voting_comparator_implementations.csv` supplies the seven exact comparator/variant implementations. Historical voting normalizes each complete rank as `(rank-1)/528`, averages voter ranks equally and sorts ascending; ties use LLM rank, CLIP rank and feature name. The third voter is RF relevance aggregated to source features. The historical RF has 128 trees, minimum split fraction 0.01, feature fraction 0.15 and seed 42; its correlation calculation samples at most 10,000 training rows. These are authenticated against the saved voting script and the test manifest's Git version of `MRMR`.

## 6. CLIP implementation and historical before/after evidence

[06_clip_13_descriptors.csv](06_clip_13_descriptors.csv) gives the definitions and exact order: **missing_rate, unique_ratio, concentration_share, signed_log_mean, log_standard_deviation, clipped_skewness, normalized_entropy, is_numeric, is_categorical, is_binary, numeric_stats_valid, skewness_valid, entropy_valid**. Statistics use DEV distributions. Home Credit/Stability scale the first seven by training-feature median/IQR and clip to [-8,8], leaving six indicators unchanged. Corrected LendingClub transfer instead uses its frozen median-imputation/standard-scaling preprocessor on all 13 fields.

The saved configuration uses frozen `all-MiniLM-L6-v2` semantic embeddings, 384 dimensions; projection heads **384→64→32** and **13→16→32**, GELU, dropout 0.05, L2 normalization; **27,488 parameters**. Training uses symmetric identity-pair cross-entropy, fixed temperature 0.07, AdamW learning rate 0.001, weight decay 0.01, batch 64, maximum 80 epochs, gradient clipping 1, no scheduler, early-stop patience 15 and minimum improvement 0.0001. Seeds are **11, 22, 33, 44, 55**; each checkpoint minimizes source validation loss. Representation train/validation splitting is group-aware 80/20 with seed 42. `identity_equivalence_v2` masks same identities, verified aliases, exact DEV duplicates and documented identity transforms; same-family/text-similarity proximity is diagnostic rather than an automatic identity exclusion. Settings and source-specific differences are in `06_clip_training_anchor_ranking_settings.csv`.

Stability and LendingClub anchor construction uses four equal-duration DEV subwindows, representation-TRAIN feature identities, maximum adjacent-window PSI <=0.10, maximum missingness difference <=0.05, at least 100 nonmissing observations per subwindow, and 23 members ordered by PSI, missingness difference and feature identity. Numeric bins=10, categorical minimum count=50, epsilon=1e-6; buckets are fitted on the first subwindow. The 23 Stability member names and statistics are in `06_stability_anchor_members.csv`.

Historical Home Credit uses **23 previously frozen training-split stable-core members**. The code traces membership through `bootstrap_selection_frequency_if_available >= 0.8`, taking the maximum available frequency by feature in the evidence builder, then restricting to approved training identities. Those frequencies originate from the supervised bootstrap stable-core selector. The member centroid is computed in corrected consensus space; membership is not recomputed from corrected neighbours. Thus the projection's exclusion of labels as inputs and the origin of its anchor membership are distinct recorded implementation facts. The historical file is recovered: `06_recovered_HomeCredit_23_anchor_members.csv` gives all 23 names, descriptions/source families, train identities and descriptor values; `06_recovered_HomeCredit_anchor_frequency_lineage.csv` gives the source supervised frequencies. The earlier text-anchor manifest has 28 names; the statistical builder retains the 23 representation-TRAIN identities. `06_HomeCredit_anchor_lineage_checks.csv` records the 28→23→23 chain. Every recovered member has maximum saved full-DEV bootstrap selection frequency >=0.8. The selector code defaults to five 80%-size bootstrap samples; the saved frequency counts support five selections, but a separate per-run bootstrap-fraction override record was not recovered.

For seed s, `z_s(f)=normalize((T_s(f)+S_s(f))/2)`. Seeds are aligned to seed 11 by orthogonal Procrustes; the mean is normalized. Stability-native and Home Credit transfer rank by cosine similarity between this consensus and their normalized source anchor. LendingClub transfer uses the mean of five **seed-specific anchor similarities**. Ties use descending score then ascending feature name. Downstream selection uses RF/correlation mRMR on the fixed 60/100 pool and K=20/40.

[06_historical_CLIP_before_after.csv](06_historical_CLIP_before_after.csv) records **LLM→RF/correlation mRMR** versus **LLM→corrected CLIP→RF/correlation mRMR**:

| Model | OOT AUC before → after | Jaccard before → after | Score PSI before → after |
| --- | --- | --- | --- |
| LR | 0.738098 → 0.736996 | 0.404344 → 0.729185 | 0.003798 → 0.004898 |
| CatBoost | 0.762954 → 0.763820 | 0.296557 → 0.795195 | 0.008286 → 0.004094 |

The before/after whole-pipeline source-feature Nogueira values at **P529** are **0.553104→0.833713** for LR and **0.407715→0.875593** for CatBoost. The after-run conditional fixed-pool values are **0.760000 at P60** and **0.808333 at P100**, respectively. These use different stated universes; the CSV retains both calculations and the original reported scalars.

The before pipeline uses fold-specific LLM selections. The after pipeline uses the frozen full-DEV LLM-approved names intersected with corrected CLIP identities, then ranks them by corrected consensus and applies RF/correlation mRMR. Thus both the representation screening and pool reuse differ. This is not an isolated before/after test of the pairing repair alone. `06_recovered_earlier_CLIP_comparison_tables.csv` additionally preserves earlier CLIP v1/v2 comparisons recovered from the cleanup ZIP, and the eight CLIP v2 summaries appear among the historical results. No isolated pairing-repair-only downstream AUC/Jaccard/Nogueira/PSI comparison was recovered.

## 7. Selected features, descriptions, source families and recency

[07_selected_features_with_descriptions.csv](07_selected_features_with_descriptions.csv) contains **72,426 saved feature-selection records** covering final full-DEV and available fold selections. It includes run identity, selection scope/order, description, source family, formula/operation, semantic group and provenance. [07_feature_dictionary.csv](07_feature_dictionary.csv) contains 3,299 unique dataset-feature entries. LendingClub v2 and Stability use their saved exact metadata. Home Credit engineered descriptions use the repository's prefix/aggregation description rules, explicitly labelled as such. `07_method_feature_overlaps.csv` gives 646 within-dataset/model/cohort method pairs, intersection counts, Jaccard and overlapping names; it does not select pairs based on performance.

[07_homecredit_recency_eligibility_and_selection.csv](07_homecredit_recency_eligibility_and_selection.csv) lists 113 recency/date fields and controls. Under the full-pool Pure LLM definition, all fields in the 529-variable predictor universe are eligible, including `PREV_DAYS_DECISION_MAX`. The archived 391-feature screened eligibility is preserved in a separate column. `PREV_recent_decision_MAX`, raw `DAYS_DECISION`, `recent_decision` and `application_time_proxy` are explicit excluded controls.

**`PREV_DAYS_DECISION_MAX` duplicates the split variable.** The assembler copies `DAYS_DECISION` into `recent_decision`, aggregates both by applicant, and splits on the maximum copied value. A calculation over the existing **1,670,214 previous-application rows and 338,857 applicant groups** confirms all five MIN/MAX/MEAN/SUM/VAR pairs are equal including missing values, with zero numerical differences. Joining the maximum to training applications reproduces **99,092 DEV** and **120,053 OOT** rows. The numeric checks are in `07_HomeCredit_recency_duplicate_check.csv`.

The duplicate is selected in four saved final runs: Home Credit full-feature LR; full-feature CatBoost; CatBoost mutual-information mRMR K40; and historical CatBoost `voting_llm_clip_top100_then_mrmr`. `07_HomeCredit_split_duplicate_affected_runs.csv` includes these and their available fold-selection occurrences, with exact run IDs. The duplicate belongs to the shared 529-feature starting universe; downstream screening may remove it.

`07_feature_engineering_specification.csv` records applicant-level joins, auxiliary-table aggregations, ratio/payment features, temporal proxy, exclusions and preprocessing. `DAYS_DECISION` is a relative previous-application decision time; the inspected data does not provide a row-level feature-availability timestamp ledger. The existing temporal-as-of audit records five auxiliary tables requiring manual confirmation and does not certify their availability at every simulated cutoff.

## 8. Outcome definitions, terms, cohorts and maturity

LendingClub retains **Fully Paid** and **Does not meet the credit policy. Status:Fully Paid** as good; **Charged Off**, **Default**, and **Does not meet the credit policy. Status:Charged Off** as bad. Current, grace, late, issued, missing or other unrecognized statuses are excluded. The raw table has **2,260,701 rows**; **1,348,099 retained**, **912,602 excluded**. This includes 33 missing-status rows among exclusions. The retained labels are **1,078,739 good** and **269,360 bad**.

The 36-month retained group has **1,023,206 rows, 163,926 bad (16.0208%)**; the 60-month group has **324,893 rows, 105,434 bad (32.4519%)**. `08_LendingClub_status_filter_counts.csv` gives status counts; `08_LendingClub_status_by_term_and_issue_cohort.csv` gives all 1,024 raw status×term×cohort combinations, tabulated from existing data. `08_LendingClub_existing_issue_cohort_by_term.csv` copies the existing 243-row resolved-status audit. `08_LendingClub_year_term_and_partition_counts.csv` supplies the annual/term/DEV/OOT aggregation, including 2014–2015 DEV and 2016 OOT.

The existing target note estimates observation end **2019-05-01** using maximum outcome-related dates. It documents resolved-status filtering and separate term/cohort audits. **No saved fixed-horizon, fully matured-loan sensitivity refit or survival/censoring experiment was found.** These audits do not establish per-loan label availability at every simulated training cutoff.

`08_target_horizons_and_label_availability.csv` records the known target definitions. Home Credit's supplied description leaves **X days / first Y installments** unspecified. LendingClub uses resolved status at the snapshot; 36/60 months are contractual terms, with no common fixed default horizon documented. Stability uses the supplied binary `target`; no numerical outcome horizon or per-case label-availability timestamp was recovered. No numerical horizons were inferred.

## 9. Tuning provenance and verification of Boruta/RFE settings

K=20 for LR and K=40 for CatBoost, and the final model settings, are frozen in the repository configuration. **An originating DEV hyperparameter trial history or 20-versus-40 budget optimization study was not found after the expanded repository, backup, archive and Git notebook search.** The feature set/DEV partition that originally selected those model settings cannot be established from the recovered evidence. `09_existing_tuning_provenance.csv` records this limitation; `09_frozen_model_and_dataset_settings.csv` preserves the exact configured settings and dataset inventories.

There are six completed heavy-selector **DEV fold-1 feasibility pilots**, with performance evaluation explicitly false and no OOT access. They use **16,242 training / 16,720 validation rows and 529 candidates** for Home Credit, and **83,283 / 114,267 rows and 675 candidates** for LendingClub v2. Each tests CatBoost SHAP, Boruta, or RFE. Home Credit Boruta confirms 26 features; LendingClub confirms 161 and uses top 40. These pilots support feasibility/configuration freeze, not an AUC-based optimization claim. Exact counts, timings and config hashes are in `09_DEV_pilot_summary.csv` and the detailed pilot CSV.

Verification confirms canonical Boruta: RF 500/depth 6, Boruta automatic tree count, 10 iterations, percentile 100, alpha 0.05, two-step correction, seed 42, confirmed-only support without rejected-feature padding. Canonical CatBoost RFE uses 500 iterations, depth 6, learning rate 0.05 and floor(20% of surviving features) removal, bounded to stop at K. Historical registry settings differ: Boruta 15 iterations and integer-step RFE 10. `09_verified_Boruta_RFE_and_budgets.csv` preserves the canonical configuration fields. The DEV combination review records no configuration selection/removal/tuning/reordering; P=100/200/300 are preregistered pools, with P=200 primary for prospective voting.

`09_recovered_original_run_configurations.csv` adds the 32 original run manifests, configuration hashes, run dates, code commits and saved budget settings. `09_Git_notebook_search_inventory.csv` covers 20 reachable notebook versions; the matched cells provide no hyperparameter trial or 20/40 optimization evidence. The archived original runs and later frozen settings are retained as separate versions. Earlier method AUC differences alone do not document how a setting or budget was chosen.

## 10. Existing runtime and memory measurements

`10_saved_run_runtime_and_memory.csv` exports 60 saved resource records. Their four dedicated selection/training/prediction/evaluation timing slots are null; total time is available. `10_saved_OOT_worker_stage_measurements.csv` separately preserves the 64 review rows' available fit, prediction, worker-wall-time and memory fields. These have their original worker/reference scopes and are not relabelled total pipeline costs.

`10_voting_runtime_summary.csv` preserves 16 prospective runs with interruptions/resume information and cached-path limitations. `10_voting_log_stage_spans.csv` contains 80 log-marker spans and explicitly says no instrumented stage timer is available; spans may include pauses. `10_CLIP_saved_fold_metrics_and_timings.csv` contains 30 saved fold records with elapsed seconds and RSS observations; point RSS is not labelled peak RSS. Stability supervisor/attempt measurements are in `10_Stability_supervisor_measurements.csv`. The six feasibility pilot runtimes are separately scoped in point 9.

The recorded Stability machine profile is **Windows 11 10.0.26200, Intel Core i7-13620H, 10 physical cores / 16 logical processors, 42,555,686,912 bytes RAM**. Run preflights record **NVIDIA GeForce RTX 4060 Laptop GPU, approximately 8 GiB VRAM**; the documented downstream experiments and CLIP projection training use CPU. Estimator thread caps are four in the frozen executions. `10_recorded_run_hardware.csv` preserves per-run dates, accelerator, hardware capacity and Git commit; `10_Stability_recorded_hardware_plan.csv` preserves the recorded machine profile. CPU model is absent from individual preflight fields and is not silently backfilled there.

LLM rankings, embeddings, completed seeds and downstream artifacts have reuse/cache paths. The available records do **not** constitute a controlled cold-cache versus warm-cache benchmark. Zero measured GPU allocation for a CPU run is preserved as a recorded measurement; missing measurements stay blank. Packaging time is not added to experimental timing.

`10_recovered_historical_run_timings.csv` adds 36 original matrix/CLIP runtime summaries from the backups. All 36 contain saved preprocessing, feature-selection, training and evaluation durations, separately for CV and the final stage, plus total runtime. These are stage timers; this CSV does not contain memory observations. Run IDs link them to the recovered metrics and configurations.

## 11. Existing robustness/control experiments and absences

[11_existing_controls_and_absences.csv](11_existing_controls_and_absences.csv) records the following:

- **Controlled independent LLM replication with fully identical authenticated requests: not established.** The 51 existing cache payloads contain 12 groups with multiple distinct response IDs under matching saved base-prompt hashes, requested model and temperature. Six LendingClub v2 groups contain four different responses each, with cache feature budgets 20/40/60/100; six other cache groups contain two responses each. Exact terminal retry request text and historical retry attempts are not saved. `11_existing_LLM_response_identity_records.csv` and `11_same_base_prompt_response_groups.csv` provide these facts without claiming a planned replication study.
- **Controlled prompt ablations: absent** in inspected evidence.
- **Voting comparisons holding exact candidate-feature membership and downstream reduction constant across LLM/CLIP/component controls: absent** in inspected evidence.
- **Prospective voting pool-budget sensitivity: present**, P=100/200/300 with the distinct statistical-voter protocol described in point 5.
- **Five CLIP projection-seed runs: present**, with source-validation checkpoint selection and frozen consensus.
- **Random-k and full-feature controls: present**, including unavailable Stability full-feature final models. Fixed-seed repeated random-k sets are not independent random resampling experiments.
- **Stability LLM v3 retry history: present**, one rejected attempt followed by one accepted attempt; these are not two accepted replicate models.
- **Maturity-restricted robustness experiment: absent**; existing term/cohort audits are exported.

`00_validation_checks.csv` preserves the earlier export checks with their original scope. `00_revision_validation_checks.csv` and `00_final_package_checks.csv` record the revision checks. `00_package_manifest.csv` lists the delivered files and hashes. Missing results remain missing; no new experiments were launched to fill them.
