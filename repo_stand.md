# Repository-status audit

## 1. Repository snapshot

- Audit date: 2026-06-27 (Asia/Tashkent).
- Repository root: `D:\python projects\Research`.
- Current branch: `main`.
- Current commit: `f3a5b3ed194e99b63657151dac8eed83ff92fd6d`.
- Git status before this audit file was written: ` D RUN.md`. This pre-existing deletion was not touched by the audit. After this file is created, `repo_stand.md` is the only additional status entry expected.
- Repaired pairing policy: `identity_equivalence_v2`; policy file: `results/clip_pairing_repair/mask_policy.json`.
- Deleted matrix code remains absent: both `scripts/run_clip_final_comparison.py` and `src/credit_risk_fs/clip_final_comparison` were checked directly and do not exist.
- Legacy invalid CLIP artifacts remain physically present for audit, but are excluded from scientific reuse by `results/clip_pairing_repair/checkpoint_invalidation.json` and `results/research_summary/artifact_registry.csv`. The invalidation file classifies 398 artifacts as `invalid_pairing_policy`; it does not delete or overwrite them.
- Primary registries: `results/research_summary/run_index.csv`, `results/research_summary/artifact_registry.csv`, `results/research_summary/reusable_metrics.csv`, and `results/research_summary/selected_feature_registry.csv`.
- Primary manifests: `results/clip_pairing_repair/repair_manifest.json`, `results/research_summary/summary_manifest.json`, `results/corrected_homecredit_clip/task_manifest.json`, and `results/corrected_homecredit_clip/provenance_fix/provenance_fix_manifest.json`.
- Audit method: file reads, focused text searches, directory listings, CSV/JSON inspection, Git status/branch/hash reads, and SHA-256 reads only. No Python or project entry point was executed.

## 2. Executive task-status table

| Task | Implementation status | Execution status | Verification status | Final current status | Main evidence | Main blocker |
| ---- | --------------------- | ---------------- | ------------------- | -------------------- | ------------- | ------------ |
| 1. LLM → corrected CLIP → mRMR | `implemented` | `executed` | `scientifically_verified` with provenance limitations | `completed_with_limitation` | Two run-local model/selection/prediction/metric sets, combined manifests, matching registry hashes, and frozen stage artifacts | No stable row IDs or DEV row predictions; exact post-one-hot mRMR width was not persisted |
| 2. LendingClub-trained corrected CLIP → frozen Home Credit | `not_implemented` | `not_executed` | `not_implemented` | `not_implemented` | Current validators/configs hard-code Home Credit as training and LendingClub v2 as external-only; no matching registry row or checkpoint | Dataset-role reversal, LendingClub-specific pairing/split evidence, reverse scoring, downstream orchestration, and row-ID exports are absent |
| 3. Home Credit embedding validation | `implemented` | `executed` | `scientifically_verified` with limited independent stability support | `completed_with_limitation` | Hash-matched 436-row embeddings/UMAP, diagnostics manifest, 20-row anchor table, and stability evidence table | Independent feature-level drift evidence is available for only 3/20 neighbours |

## 3. Task 1 — LLM → corrected CLIP → mRMR

### 3.1 LLM stage

The recoverable final-DEV flow is:

`529 modeling features → 520 pass missing-rate filter → 391 pass IV prefilter and enter the LLM prompt → 100 frozen ranked LLM selections → model-specific pool`

- Input feature universe: 529, recorded by the run summary and `results/corrected_homecredit_clip/feature_universe/feature_universe_summary.json`.
- Exact frozen LLM source: `results/homecredit/catboost/hybrid_mrmr/catboost_hybrid_llm_then_mrmr_87fbcccf4952/features/llm_rankings_summary.csv`, final-DEV scope.
- Full final-DEV LLM evidence: `results/homecredit/catboost/hybrid_mrmr/catboost_hybrid_llm_then_mrmr_87fbcccf4952/llm_responses/final_dev/llm/selection_payload.json`.
- Missingness evidence: `.../llm/missing_filter_summary.csv` has 529 rows; 9 exceed the configured maximum missing rate of 0.95.
- IV evidence: `.../llm/iv_prefilter/feature_audit.csv` has 520 rows. The selector configuration in `src/credit_risk_fs/experiments/runner.py` uses `min_iv=0.01`, `max_iv_for_leakage=0.5`, and encoding. The saved payload records 391 IV-selected prompt candidates.
- LLM-approved count: 100 persisted selections. The raw response contained 133 names, but `LLMSelector._normalize_llm_response` truncates to the configured ranking budget of 100; the frozen scientific artifact is therefore 100, not 133.
- Rejected/not retained:
  - 9 removed by missingness before IV/LLM;
  - 129 of the 520 missingness-qualified features not passed by the IV prefilter;
  - 291 of the 391 prompt candidates not retained in the normalized LLM top 100;
  - total outside the frozen top 100: 429 of 529.
- Decision criterion: this is a ranking, not a binary label or probability threshold. Prompt version `stability_expert_v3` asks for up to 100 stability-aware features. Downstream flags select ranks `<=60` for LR hybrid use and `<=100` for CatBoost hybrid use.
- Frozen/independent status: yes. The LLM source run is non-CLIP (`pairing_policy_version=not_applicable_non_clip`) and is registered `reusable_existing`. Its manifest reports completion from 2026-05-18T09:38:15Z to 2026-05-18T09:50:21Z; the final selection payload was written at 2026-05-18T09:47:29Z. The later combined pipeline did not call the LLM.
- Compatibility: all 100 frozen CatBoost-pool names and all 60 LR-pool names occur in `corrected_consensus_clip_scores.csv`. The feature-universe reconciliation maps the 100 names one-to-one to 100 non-empty, unique `feature_id` values. The frozen LLM files contain names rather than IDs, so compatibility is established by the unique reconciliation table, not by IDs embedded in the LLM artifact.
- Model-specific frozen copies:
  - LR: `results/corrected_homecredit_clip/combined_pipeline/frozen_llm_approved_lr.csv` — 60 rows.
  - CatBoost: `results/corrected_homecredit_clip/combined_pipeline/frozen_llm_approved_catboost.csv` — 100 rows.

### 3.2 CLIP stage

- Features received from the LLM:
  - Logistic Regression: 60.
  - CatBoost: 100.
- Corrected score artifact: `results/corrected_homecredit_clip/combined_pipeline/corrected_consensus_clip_scores.csv`, 436 rows.
- Pairing policy: `identity_equivalence_v2`.
- Seed aggregation: per-seed L2-normalized averages of projected semantic/statistical views; seed spaces are orthogonal-Procrustes aligned to seed 11, then normalized and averaged across seeds 11, 22, 33, 44, and 55. Evidence: `results/corrected_homecredit_clip/embeddings/embedding_manifest.json`.
- Candidate rule: exact-name intersection of the frozen LLM pool, modeling columns, and corrected CLIP ranking, sorted by `consensus_clip_rank`, followed by fixed head selection.
- CLIP-to-mRMR source-feature count:
  - Logistic Regression: 60.
  - CatBoost: 100.
- The number differs by model because the fixed pool is model-specific: 60 for LR and 100 for CatBoost.
- Rule type: fixed count, not a threshold, percentile, or sweep. It is configured directly in `scripts/run_corrected_homecredit_clip_pipelines.py` and recorded in `results/corrected_homecredit_clip/combined_pipeline/combined_pipeline_manifest.json`.
- `candidate_pool_manifest.csv` has exactly 60 LR combined rows and 100 CatBoost combined rows.

Explicit answers:

- How many features does the LLM pass to CLIP? **60 for Logistic Regression and 100 for CatBoost.**
- How many features does CLIP pass to mRMR? **60 raw source features for Logistic Regression and 100 raw source features for CatBoost.**

The latter counts are before preprocessing. `FixedRankThenMRMRSelector.fit_postprocess` runs mRMR after preprocessing. One-hot expansion is visible in selected names such as `NAME_EDUCATION_TYPE_Secondary / secondary special` and `CODE_GENDER_F`, but the exact expanded mRMR input width was not saved. Thus the raw semantic-feature input is exact; the encoded-column width is `UNKNOWN — not recoverable from current artifacts`.

### 3.3 mRMR and downstream stage

#### Logistic Regression

- Raw mRMR source input: 60 CLIP-ranked source features.
- Post-preprocessing mRMR column count: `UNKNOWN — not recoverable from current artifacts`.
- Final selected count: 20.
- Selected-feature artifact: `results/corrected_homecredit_clip/combined_pipeline/runs/homecredit_lr_llm_then_corrected_clip_then_mrmr/features/final_selected_features.csv`.
- Selection trace: `.../llm_responses/final_dev/llm_then_corrected_clip_then_mrmr_selection_manifest.csv`. It has 60 raw ranking rows and marks 19 direct raw-name final matches; the twentieth selected model column is a one-hot expansion of an approved raw feature.
- Model configuration: solver `liblinear`, `max_iter=1000`, balanced classes, random state 42. Full configuration is in `.../models/final_model_metadata.json`.
- DEV prediction path: none. `results/cv_results.csv` contains fold-level metrics, not row-level DEV predictions.
- OOT prediction path: `.../results/oot_predictions.csv`, 120,053 rows.
- Prediction columns: `y_true,y_pred_proba,y_pred`; stable row IDs and provenance columns are absent.
- Metric paths: `.../results/experiment_summary.csv` and `.../results/oot_test_results.csv`.
- Recorded metrics: DEV CV AUC 0.730158; OOT AUC 0.736996; OOT KS 0.354224; OOT log loss 0.626619; model-score PSI 0.004898.
- Run ID: `homecredit_lr_llm_then_corrected_clip_then_mrmr`.
- Configuration hash: `49c9e8af43347dc4ef11f2159045a2215b44fbcc1c8dfac0c05e4bd68fe403ea`.
- Data-manifest hash: `a6acde75b4b382b08030a83ce21d212f0667d9a63a68aaa68486841f8c229f9e`.
- Artifact timestamps: split/selection artifacts around 2026-06-25T17:07:32Z; final model around 17:07:43Z; metrics/predictions around 17:07:44Z.
- Result origin: `newly_executed`.
- Metric-to-prediction trace: run-local code and matching artifacts link the OOT metrics to the 120,053 OOT predictions, and their current hashes match the registry. Exact row-level lineage cannot be independently audited because stable IDs are absent.

#### CatBoost

- Raw mRMR source input: 100 CLIP-ranked source features.
- Post-preprocessing mRMR column count: `UNKNOWN — not recoverable from current artifacts`.
- Final selected count: 40.
- Selected-feature artifact: `results/corrected_homecredit_clip/combined_pipeline/runs/homecredit_catboost_llm_then_corrected_clip_then_mrmr/features/final_selected_features.csv`.
- Selection trace: `.../llm_responses/final_dev/llm_then_corrected_clip_then_mrmr_selection_manifest.csv`. It has 100 raw ranking rows and marks 38 direct raw-name final matches; two selected model columns are one-hot expansions of approved raw features.
- Model configuration: fixed CatBoost configuration in `.../models/final_model_metadata.json`, including depth 10, learning rate 0.01, 1,500 iterations, balanced classes, early stopping 150, and random state 42.
- DEV prediction path: none. `results/cv_results.csv` contains fold-level metrics only.
- OOT prediction path: `.../results/oot_predictions.csv`, 120,053 rows.
- Prediction columns: `y_true,y_pred_proba,y_pred`; stable row IDs and provenance columns are absent.
- Metric paths: `.../results/experiment_summary.csv` and `.../results/oot_test_results.csv`.
- Recorded metrics: DEV CV AUC 0.750443; OOT AUC 0.763820; OOT KS 0.392046; OOT log loss 0.477026; model-score PSI 0.004094.
- Run ID: `homecredit_catboost_llm_then_corrected_clip_then_mrmr`.
- Configuration hash: `df289220513a43f5768e85e63260655402132b577cedb9b1a9a690213cae9769`.
- Data-manifest hash: `a6acde75b4b382b08030a83ce21d212f0667d9a63a68aaa68486841f8c229f9e`.
- Artifact timestamps: split/selection artifacts around 2026-06-25T17:20:37Z; final model around 17:22:41Z; metrics/predictions around 17:22:43Z.
- Result origin: `newly_executed`.
- Metric-to-prediction trace: run-local code and matching artifacts link the metrics to the OOT predictions, and current hashes match the registry. Stable row-level lineage is absent.

For both models, the saved DEV/OOT windows are DEV `[-600,-240)` with 99,092 rows and OOT `[-240,0]` with 120,053 rows. The prediction limitation is explicitly recorded in `results/corrected_homecredit_clip/provenance_fix/provenance_fix_summary.md`.

### 3.4 Did it truly run?

Classification: `executed_but_incompletely_proven`.

This is not a registry-only claim. Both model variants have:

- raw LLM and corrected-CLIP stage artifacts;
- model-specific fixed candidate-pool manifests;
- final/fold selection artifacts;
- fitted model and preprocessor files;
- split and leakage manifests;
- 120,053-row OOT prediction files;
- run-local metrics and runtime values;
- sequential execution timestamps;
- matching current/registered SHA-256 values for the selected-feature, prediction, and metric files.

The proof is incomplete because no stable row IDs or row-level DEV predictions were exported, and the exact post-one-hot mRMR input width was not persisted. Those limitations prevent `executed_and_verified`, but the artifact chain is too substantial for `placeholder_or_registry_only`.

### 3.5 Can it run independently?

`NO — current entry point forces unrelated reruns`

`scripts/run_corrected_homecredit_clip_pipelines.py:27` defines a no-argument `main`. Lines 48–55 hard-code both models and both methods (`corrected_clip_then_mrmr` and `llm_then_corrected_clip_then_mrmr`). There is no CLI stage/method/model/output flag. It also writes into the existing fixed output directory and would overwrite aggregate outputs. `src/credit_risk_fs/pipelines/common.py:475` performs each experiment; line 757 exports only three prediction columns.

### 3.6 Exact manual command

`NOT READY FOR MANUAL EXECUTION`

No command should be run now: this task already has executed artifacts, and the only current command would rerun four methods and overwrite aggregate paths.

Missing separation:

- `--method llm_then_corrected_clip_then_mrmr`;
- model selection such as `--models lr,catboost`;
- a non-overwriting `--output-dir`;
- resume/skip guards for existing run IDs;
- stable row IDs and required prediction provenance columns;
- optional persisted preprocessed mRMR input width.

Smallest coherent code change:

1. Add argument parsing and filtered `specs` construction in `scripts/run_corrected_homecredit_clip_pipelines.py::main`.
2. Preserve stable IDs in `src/credit_risk_fs/pipelines/common.py::PreparedExperimentData` and `prepare_modeling_data`, then export DEV/OOT predictions with `stable_row_id,dataset,split,target,prediction,run_id,data_manifest_hash` in `run_experiment`.
3. Save the preprocessed input width from `FixedRankThenMRMRSelector.fit_postprocess`.

No such change was made during this audit.

## 4. Task 2 — LendingClub-trained CLIP → Home Credit reverse transfer

- LendingClub v2 corrected CLIP training code: not available as a valid configured path. The generic model/trainer classes exist, but the active boundary is hard-coded to Home Credit training and LendingClub external validation.
- Pairing policy: `identity_equivalence_v2` exists globally, but no LendingClub-training manifest applies it to a LendingClub train split.
- Text view readiness: 576 LendingClub v2 text embeddings exist at `results/clip/text_baseline/lendingclub_v2_text_embeddings.parquet`, dimension 384, and are independently reusable.
- Statistical view readiness: 576 LendingClub v2 statistical vectors/external pairs exist, dimension 13, but they were transformed with the unchanged Home Credit-fitted statistical preprocessor. They are scientifically valid for the completed Home Credit→LendingClub forward application, not as a fitted LendingClub-training view.
- Eligible LendingClub count: 576 of 796 evidence rows. Exclusions in `results/lendingclub_v2/analysis/clip_readiness/dev_only_clip_training_evidence.csv`: 79 missing description, 99 missing saved DEV training signal, and 42 missing both.
- LendingClub CLIP train/validation feature counts: **UNKNOWN — no LendingClub-training feature split is configured or persisted.** All 576 current pairs are labeled external-only in the corrected Home Credit boundary.
- Duplicate policy: intended policy would mask only same feature, verified alias, exact DEV duplicate, or documented identity-preserving transform; source table/text/statistical similarity remain diagnostic only. However, the saved exact-duplicate evidence is Home Credit DEV-specific. No LendingClub DEV exact-duplicate artifact or exclusion manifest exists.
- Approved LendingClub modeling boundaries: DEV `[-1795,-1065)` with 598,649 rows and OOT `[-1065,-730]` with 293,105 rows. Evidence: reusable LendingClub run split manifests and `run_index.csv`.
- Corrected LendingClub-trained checkpoint: none.
- Any repaired-policy checkpoint: yes, five Home Credit-trained corrected checkpoints exist; none was trained on LendingClub.
- Existing old checkpoint evidence: old `results/clip_v2/training` checkpoints and dependent LendingClub downstream outputs are classified `invalid_pairing_policy`. They cannot be reused.
- Frozen application to Home Credit: not implemented for a LendingClub-trained encoder.
- Current forward mapping: Home Credit-trained projection heads accept LendingClub 384-dimensional text and 13-dimensional statistical views transformed by the Home Credit-fitted preprocessor, then score them against the unchanged Home Credit anchor.
- Required reverse mapping: a future LendingClub-trained representation must fit its statistical preprocessor and anchor using LendingClub DEV/training features only, freeze them, then transform compatible Home Credit views without refitting. No code/config currently implements this direction.
- Incompatible/missing metadata handling: not implemented for reverse transfer. The current Home Credit projection excludes 93 of 529 features lacking complete views (84 missing description; 9 missing description plus saved DEV signal). A reverse pipeline would need an explicit exclusion/reconciliation artifact rather than fabrication.
- Downstream selector/model path: generic `FixedRankThenMRMRSelector` and LR/CatBoost experiment machinery are reusable components, but no reverse-transfer candidate-ranking or orchestration entry point connects them.
- Forward-transfer evidence: corrected Home Credit→LendingClub representation-level scoring is valid and reusable at `results/corrected_homecredit_clip/training/lendingclub_v2_joint_embeddings.parquet` and `.../lendingclub_v2_learned_scores.csv` (576 rows, corrected checkpoint/policy hashes). It is representation/scoring evidence only. Old downstream LendingClub CLIP metrics under `results/clip_v2/final_evaluation` are invalid.
- Reverse-transfer results: no run-index row, registry artifact, checkpoint, score file, prediction, metric, or result directory matching corrected LendingClub training/reverse transfer exists.

Classification: `not_implemented`.

### 4.1 Standalone manual execution readiness

`NO`

Blocking source boundaries include:

- `src/credit_risk_fs/clip/pair_validation.py:27-30`, which requires Home Credit training and LendingClub external validation;
- `src/credit_risk_fs/clip/training_validation.py:190`, which enforces the same roles;
- `src/credit_risk_fs/clip/pair_builder.py`, which writes Home Credit train/validation and LendingClub external artifacts;
- `scripts/train_clip_encoder.py:174-205`, whose full-output manifests hard-code those roles;
- `src/credit_risk_fs/clip/learned_scoring.py:48-65`, which writes direction-specific Home Credit/LendingClub outputs;
- absence of a reverse-transfer orchestration entry point.

### 4.2 Exact manual commands

No scientifically valid ordered command sequence exists. Supplying one would invent an unsupported configuration.

Minimum coherent implementation required:

1. Generalize pair building/validation/training manifests to explicit training and external dataset roles.
2. Add fixed `configs/corrected_lendingclub_clip/contrastive_data.yaml` and `training.yaml`.
3. Build a group-safe LendingClub feature train/validation split and LendingClub DEV-only exact-duplicate evidence under `identity_equivalence_v2`.
4. Fit the statistical preprocessor/anchor on LendingClub training evidence only.
5. Add frozen Home Credit projection with explicit compatibility/exclusion reconciliation.
6. Add one standalone reverse-transfer entry point that fixes the candidate pool, runs mRMR once per fixed model budget, evaluates LR/CatBoost only, writes stable-ID DEV/OOT predictions, metrics, and registry-ready manifests, and refuses the old matrix.

Likely implementation locations are the source files above plus a new narrowly scoped script such as `scripts/run_corrected_lendingclub_to_homecredit_transfer.py`. No files were added or changed for this implementation.

## 5. Task 3 — Embedding validation and stable-core evidence

- Expected Home Credit feature count: 529.
- Corrected embedding count: 436.
- UMAP point count: 436.
- Contrastive training-pair count: 349.
- Group-safe validation-pair count: 87.
- Excluded count: 93 — 84 `missing_description`; 9 `missing_description;no_dev_training_signal_saved`.
- Embedding dimensionality: 32.
- Semantic kNN purity, cosine k=10: 0.687156.
- Shuffled-label reference: 200 permutations, seed 20260625; mean purity 0.139755 and 97.5th percentile 0.151393.
- Original-space silhouette: 0.012817; shuffled 97.5th percentile -0.161176.
- UMAP trustworthiness, k=10: 0.991335.
- Semantic groups: persisted in `feature_universe_reconciliation.csv`, `feature_embeddings.*`, and `umap_coordinates.csv`.
- Diagnostics paths:
  - `results/corrected_homecredit_clip/diagnostics/diagnostics_manifest.json`;
  - `results/corrected_homecredit_clip/diagnostics/cluster_metrics.csv`;
  - `results/corrected_homecredit_clip/diagnostics/shuffled_label_control.csv`;
  - `results/corrected_homecredit_clip/diagnostics/umap_coordinates.csv`;
  - `results/corrected_homecredit_clip/diagnostics/homecredit_feature_umap.png`.
- Stable-core anchor: normalized centroid of 23 frozen Home Credit training-split stable-core features; no target or OOT data. Manifest: `results/corrected_homecredit_clip/stable_core/anchor_manifest.json`.
- Top-20 path: `results/corrected_homecredit_clip/stable_core/top20_anchor_neighbours.csv`.
- Reproducibility: internally partial but measured across five seeds. Six neighbours occur in every seed top 20, six in 4/5, five in 3/5, and three in 2/5. The consensus artifact and registered hash match. This supports a reproducible consensus ranking, not identical top-20 membership in every seed.
- Independent stability evidence: `results/corrected_homecredit_clip/stable_core/anchor_neighbour_stability_evidence.csv`. Its sources are independent non-CLIP drift/baseline evidence. Feature-level PSI/evidence status is available for 3/20 and unavailable for 17/20; valid-baseline selection frequency is populated for 9/20; two have `known_stable_flag=True`.
- Integrity: current SHA-256 values for embeddings, UMAP coordinates, top-20, and stability evidence match their registry rows.
- Limitations: 93 features legitimately lack complete frozen views; UMAP is qualitative; semantic structure is not predictive-value proof; the anchor analysis is partly post hoc; seed membership is not identical; independent drift coverage is sparse.

Classification: `completed_with_limitation`.

No further manual command is required. Do not rerun UMAP merely to force 529 points: 436 is the scientifically valid complete-view set.

## 6. Artifact validity map

### Reusable scientific evidence

- `results/corrected_homecredit_clip/training/seeds/seed_*/best_checkpoint.pt` — corrected Home Credit CLIP; depends on corrected pairs and `identity_equivalence_v2`; `newly_executed`; five traceable checkpoints and manifests exist.
- `results/corrected_homecredit_clip/embeddings/feature_embeddings.parquet` — corrected five-seed consensus embeddings; depends on corrected checkpoints; `newly_executed`; 436 rows and hash verified.
- `results/corrected_homecredit_clip/diagnostics/*` — UMAP and quantitative structure diagnostics; depends on corrected embeddings; `newly_executed`; manifest, controls, coordinates, and hashes exist.
- `results/corrected_homecredit_clip/stable_core/*` — corrected anchor/top-20 and independent evidence join; depends on corrected embeddings plus frozen non-CLIP stable-core/drift sources; `newly_executed`; valid with sparse-support limitation.
- `results/corrected_homecredit_clip/combined_pipeline/runs/homecredit_*_llm_then_corrected_clip_then_mrmr/*` — Task 1 selections/models/OOT predictions/metrics; depends on frozen non-CLIP LLM ranking and corrected CLIP scores; `newly_executed`; usable with documented row-ID/DEV-prediction limitation.
- `results/corrected_homecredit_clip/training/lendingclub_v2_joint_embeddings.parquet` and `.../lendingclub_v2_learned_scores.csv` — corrected forward representation application; depends on Home Credit-trained corrected checkpoint and Home Credit-fitted transform; reusable representation evidence, not reverse transfer or downstream performance.
- `results/clip/text_baseline/*embeddings*` and `results/clip_v2/statistical_view/*vectors*` — frozen input views; independent of old CLIP score/ranking; reusable within their documented fit/transform direction.
- `results/research_summary/reusable_metrics.csv` rows with `reuse_status=reusable_existing` — 32 completed non-CLIP LR/CatBoost baselines across Home Credit and LendingClub v2; no old CLIP dependency.

### Invalidated evidence

- `results/clip_v2/training/*` — old CLIP-v2 checkpoints/scores; depends on the faulty pre-`identity_equivalence_v2` pairing policy; `invalid_pairing_policy`.
- `results/clip_v2/final_evaluation/*` — old CLIP rankings, selections, predictions, and metrics, including LendingClub downstream “transfer”; depends on invalid old scores/checkpoints; `invalid_pairing_policy`.
- `results/clip_v2/final_analysis/*` — analyses derived from invalid CLIP outputs; `invalid_pairing_policy`.
- `results/clip_versions/v1/*` CLIP-dependent artifacts and final CLIP reports under `reports/final_clip_*`; old-policy dependency; `invalid_pairing_policy`.

### Unknown or unresolved evidence

- `results/clip_pairing_repair/smoke/smoke_checkpoint.pt`, its manifest, and `NOT_FOR_SCIENTIFIC_USE.txt` — repaired-policy smoke only; registry status `unknown_requires_review`; explicitly prohibited from scientific use.
- Task 1 stable row-level provenance — dependency is the existing prediction export; unresolved because IDs were dropped before saving; metrics remain usable only with the documented limitation.
- LendingClub-trained corrected checkpoint/reverse transfer — no artifact exists; status `not_implemented`, not evidence.
- Exact post-preprocessing mRMR input width for Task 1 — raw pool is known, encoded width is not persisted.

## 7. Do-not-rerun list

The following completed work is verified and must be reused:

- Corrected Home Credit CLIP seeds 11, 22, 33, 44, and 55 under `results/corrected_homecredit_clip/training/seeds`.
- Corrected Home Credit embedding generation: 436 features.
- Existing 436-point UMAP and shuffled-label diagnostics.
- Stable-core anchor and top-20 consensus-neighbour calculation.
- Task 1’s two combined LLM → corrected CLIP → mRMR downstream runs and its two corrected-CLIP-only comparison runs.
- Frozen Home Credit LLM semantic screening from `catboost_hybrid_llm_then_mrmr_87fbcccf4952`; do not call the LLM again.
- Verified full mRMR baselines in `run_index.csv` for LR/CatBoost on Home Credit and LendingClub v2.
- Verified LLM → mRMR baselines in `run_index.csv` for LR/CatBoost on Home Credit and LendingClub v2.
- The 32 verified non-CLIP baselines represented by `reusable_metrics.csv`.
- Corrected Home Credit→LendingClub forward representation/scoring artifacts (576 external features).

An all-feature comparator is not present in the verified registry for the corrected Home Credit comparison. It is therefore not falsely listed here as completed, and it is not required by the three audited tasks.

## 8. Required remaining work

| Task | Current status | Prerequisite | Exact manual command availability | Expected output directory | Source-code modification required | Later scientific audit required |
| ---- | -------------- | ------------ | --------------------------------- | ------------------------- | --------------------------------- | ------------------------------- |
| Corrected LendingClub CLIP training and frozen Home Credit reverse transfer | `not_implemented` | Dataset-role generalization; LendingClub train/validation feature split; DEV-only duplicate policy; LC-fitted preprocessor/anchor; reverse compatibility manifest; standalone downstream orchestration | None | Proposed `results/corrected_lendingclub_to_homecredit_transfer/` after implementation | Yes | Yes |
| Unified final scientific analysis after reverse transfer | `blocked` by Task 2 | Audited reverse selections, stable-ID predictions, metrics, hashes, and registry entries | None until Task 2 exists | A new final-analysis directory chosen by the future implementation | Possibly, depending on existing analysis entry-point coverage | Yes |

No further execution is required for Tasks 1 or 3. Their limitations must be carried into the final report rather than “fixed” by unrequested reruns.

## 9. Recommended execution order

1. Treat Task 1 as executed; retain its row-ID and encoded-width limitations and do not rerun it.
2. Implement the minimal standalone corrected LendingClub→Home Credit reverse-transfer path described in section 4.2.
3. Manually run only that fixed, predeclared reverse-transfer sequence after code review.
4. Audit its checkpoint, feature flow, stable-ID DEV/OOT predictions, metrics, hashes, timestamps, and registry entries.
5. Run one unified final scientific analysis using reused Tasks 1/3 evidence, valid baselines, valid forward representation evidence, and the newly audited reverse transfer.

## 10. Manual command checklist

### READY NOW

None. The next required experiment has no verified standalone command.

### NOT READY — CODE CHANGE REQUIRED

- Task 1 standalone rerun command: not available because `scripts/run_corrected_homecredit_clip_pipelines.py` hard-codes four runs, overwrites fixed outputs, and does not export required IDs. A rerun is not currently required.
- Task 2 corrected LendingClub training/reverse transfer: not available because the data roles, pairing evidence, training config, reverse projection, downstream orchestration, prediction provenance, and registry update are not implemented.
- Working directory for future commands: `D:\python projects\Research`.
- Expected future Task 2 inputs: frozen LendingClub text metadata/embeddings, LendingClub DEV-only statistical evidence, approved LendingClub DEV/OOT boundaries, Home Credit frozen compatible views, and fixed model budgets.
- Expected future Task 2 outputs: corrected LendingClub seed checkpoints, frozen Home Credit embeddings/scores, fixed candidate pools, mRMR selections, LR/CatBoost DEV/OOT stable-ID predictions, metrics, and manifests.
- Start/resume policy: start corrected LendingClub training fresh; never resume an old CLIP checkpoint; reuse valid non-CLIP baselines and forward evidence.

### ALREADY COMPLETED — DO NOT RUN

- Corrected Home Credit training command recorded by the manifest:

  ```powershell
  Set-Location 'D:\python projects\Research'
  .\.venv\Scripts\python.exe scripts\train_clip_encoder.py --config configs/corrected_homecredit_clip/training.yaml --all-seeds
  ```

  Existing output: `results/corrected_homecredit_clip/training`. Post-run evidence already exists; inspect `training_summary.json`. Do not run.

- Existing combined-pipeline entry point:

  ```powershell
  Set-Location 'D:\python projects\Research'
  .\.venv\Scripts\python.exe scripts\run_corrected_homecredit_clip_pipelines.py
  ```

  It starts four fresh runs rather than resuming, has no baseline-safe method flag, and writes to `results/corrected_homecredit_clip/combined_pipeline`. Inspect the existing `combined_pipeline_manifest.json`; do not run.

- Embedding generation, UMAP, stable-core top-20, frozen LLM screening, full mRMR, LLM→mRMR, and the 32 valid independent baselines: no command is listed because each is already complete and must be reused.

## 11. Final audit conclusion

### Current truth

- Completed with limitation: Task 1, LLM → corrected CLIP → mRMR. It truly ran for LR and CatBoost; stable prediction IDs, row-level DEV predictions, and exact encoded mRMR width are absent.
- Not completed: Task 2, LendingClub-trained corrected CLIP → frozen Home Credit. It is not implemented and has not run.
- Completed with limitation: Task 3, Home Credit embedding validation. Embeddings, UMAP, semantic diagnostics, stable-core top-20, and independent evidence table are verified; independent feature-level drift evidence covers only 3/20 neighbours.
- Uncertain: no task’s basic execution state remains uncertain. The uncertainties are limited to the documented provenance/coverage fields above.

### Immediate next action

Create a minimal standalone corrected LendingClub→Home Credit reverse-transfer entry point before any experiment is run.

### Readiness verdict

`PARTIALLY READY — repository status is known, but a minimal code change is required`
