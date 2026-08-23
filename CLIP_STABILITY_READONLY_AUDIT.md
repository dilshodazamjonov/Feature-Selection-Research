# CLIP + Stability 2024 Read-Only Audit

Evidence-status vocabulary used throughout: **CONFIRMED** means directly supported by audited code, configuration, or authenticated result artifacts; **INFERRED** means strongly supported but not stated explicitly; **UNRESOLVED** means the evidence is incomplete or conflicting; **NOT FOUND** means a repository-wide targeted search did not locate supporting evidence. Paths are repository-relative unless explicitly identified as external historical evidence.

## 1. Executive Summary

- **CONFIRMED — reported AUCs.** The four reported Home-Credit-target classifier results are reproduced exactly from the preserved corrected-result artifacts:

  | CLIP direction | Classifier | DEV/CV AUC | Final OOT AUC | Metric meaning |
  |---|---:|---:|---:|---|
  | Home Credit → Home Credit | LR | 0.7172285750 | 0.7336398378 | five-fold mean CV; classifier AUC |
  | Home Credit → Home Credit | CatBoost | 0.7480551123 | 0.7626757740 | five-fold mean CV; classifier AUC |
  | LendingClub v2 → Home Credit | LR | 0.5890154000 | 0.5732026847 | pooled DEV OOF; classifier AUC |
  | LendingClub v2 → Home Credit | CatBoost | 0.6563253234 | 0.6766033515 | pooled DEV OOF; classifier AUC |

- **CONFIRMED — final CLIP generation.** The research conclusions use the corrected `identity_equivalence_v2` generation: corrected Home-Credit-trained CLIP, followed by the later LendingClub-trained reverse-transfer study. The older `results/clip_v2/{contrastive_data,training,selector_integration,final_evaluation,final_analysis}` generation is explicitly classified as scientifically invalid because it predates the identity-equivalence repair. Evidence: `cleanup/audit/cleanup_summary.md:32-38`, Git tag `corrected-homecredit-clip-v1` at commit `f3a5b3e` (2026-06-26), and the corrected configs under `configs/corrected_*`.
- **CONFIRMED — executed directions.** HC→HC completed a downstream CLIP-ranking → mRMR → LR/CatBoost study. HC→LC completed representation-only external projection/scoring, with downstream LC classifiers explicitly forbidden. LC→LC trained and validated a source representation but has no located downstream LC classifier study. LC→HC completed frozen representation transfer, target-side mRMR, target-side LR/CatBoost fitting, and Home Credit OOT evaluation.
- **CONFIRMED — what CLIP does and does not score.** The CLIP encoder is evaluated with contrastive loss and retrieval metrics (Recall@1/5/10 and MRR), not ROC-AUC. ROC-AUC is computed only after a dataset-specific CLIP ranking is screened, reduced with target-supervised mRMR, and passed to LR or CatBoost. No per-CLIP-seed downstream AUC artifact was found.
- **CONFIRMED — historical Home Credit comparator.** The corrected CLIP package reports ordinary historical mRMR OOT AUCs of 0.7456889164 (LR) and 0.7668389382 (CatBoost). This historical `mrmr` implementation is RF-impurity relevance divided by correlation redundancy, not the later canonical mutual-information mRMR.
- **CONFIRMED — artifact-location caveat.** The active repository retains CLIP source/configuration/history but not the complete historical CLIP result roots. The exact results above survive in the adjacent read-only backup `D:\python projects\Research_pre_cleanup_backup_20260704\results\...`; the repository's own cleanup documentation says the current CLIP profile is incomplete and 31 historical tests require a complete external profile. Consequently, the metrics are verified historical evidence, not reproducible solely from the current active `results` tree.
- **CONFIRMED — Stability readiness.** Stability 2024 has an authenticated 1,959-feature matrix, raw/engineered lineage, type metadata, a frozen temporal DEV/OOT protocol, mRMR paths, and LR K=20/CatBoost K=40 downstream paths. It does **not** have Stability-specific 13-field CLIP descriptors, cached CLIP text embeddings, a CLIP-ready pair table, CLIP configuration, checkpoint, source anchor, training route, or final CLIP ranking.
- **UNRESOLVED — top decisions.** The repository does not decide whether Stability should train its own encoder or be a transfer target, which corrected generation/direction set should be frozen, whether CLIP should rank all 1,959 engineered candidates or a smaller raw-variable universe, how Stability text should be rendered, or which source-fitted statistical transform and anchor should govern each transfer direction.
- **UNRESOLVED — Stability target semantics.** The audited code fixes `target` as binary and orients class 1 as higher default risk, but no repository artifact states the dataset-native event definition and prediction horizon precisely enough to quote them without external knowledge.

## 2. Audit Scope and Repository State

| Item | Status | Audited value / evidence |
|---|---|---|
| Repository root | CONFIRMED | `D:\python projects\Research` |
| Branch | CONFIRMED | `main` |
| Commit | CONFIRMED | `2073a5d0e5255b73a415ed9a898d64579a621ff6` |
| Commit subject/date | CONFIRMED | `added finalized reports`, author date `2026-08-21T21:09:30+05:00` |
| Local audit time | CONFIRMED | `2026-08-22T22:04:55+05:00` (Asia/Tashkent) |
| Initial Git status | CONFIRMED | `## main...origin/main`; pre-existing untracked `METHODOLOGY_AUDIT_9_POINTS.md` |
| Requested audit file initially present | CONFIRMED | No; `CLIP_STABILITY_READONLY_AUDIT.md` was absent |
| Repository instruction file | NOT FOUND | No `AGENTS.md` found |
| Existing-file changes by this audit | CONFIRMED | None |
| Experiments/tests/scripts executed | CONFIRMED | None; inspection used read-only shell and Git commands only |

**CONFIRMED.** The audit read active source, configs, documentation, manifests, CSV/JSON results, Git history, cleanup records, and the adjacent historical backup. It did not execute Python modules, notebooks, tests, training, evaluation, or artifact builders.

**CONFIRMED.** The active results root is documented as `D:\python projects\Research\results`, while `docs/research_extension/foundation_protocol_freeze.md:11` names `D:\ResearchFindings\results` as a historical root. That latter root did not contain the required CLIP groups at audit time. The complete artifacts used here were found instead in `D:\python projects\Research_pre_cleanup_backup_20260704`. This location difference is reported rather than silently remapped.

**CONFIRMED.** The adjacent archive `D:\python projects\Research_pre_cleanup_backup_20260704\scientific_cleanup_candidates.zip` is 90,490,889 bytes with SHA-256 `4036DA53C4E91074CD84A4C561981EEB35923B125925C7DD9FEDCEABBB762E4D`. It was inspected only as inventory evidence; no extraction or mutation was performed during this audit.

## 3. CLIP Artifact Inventory

| Important artifact or group | Type | Classification | Evidence / purpose |
|---|---|---|---|
| `src/credit_risk_fs/clip/` | Source | Active implementation | Text building/encoding/cache, compact statistical view, pair construction, negative policy, model/loss/trainer, checkpointing, learned scoring, source anchors, reverse transfer |
| `src/credit_risk_fs/pipelines/reverse_transfer.py` | Source | Active/final directional implementation | LC-source → HC-target downstream orchestration; fixed pools 60/100 and budgets 20/40 at lines 220-221 |
| `src/credit_risk_fs/selectors/fixed_rank_then_mrmr.py` | Source | Active historical CLIP integration | Frozen target-free ranking followed by target-supervised `MRMR`; class and fit order at lines 11-77 |
| `src/credit_risk_fs/selectors/mrmr.py` | Source | Active legacy-compatible implementation | RF-relevance/correlation selector; `MRMR` alias at line 219; explicitly not canonical MI at lines 25-34 |
| `configs/clip/`, `configs/clip_v2/` | Config | Boundary/base generation | Readiness, text baseline, training manifest, and compact statistical-view configuration |
| `configs/corrected_homecredit_clip/` | Config | Corrected/final HC-source generation | Pairing, identity policy, deterministic five-seed training, source/target restrictions |
| `configs/corrected_lendingclub_to_homecredit/` | Config | Corrected/final LC-source generation | Pairing, training, frozen reverse projection, downstream target-label and OOT boundaries |
| `scripts/build_clip_*`, `scripts/train_*clip*`, `scripts/analyze_corrected_homecredit_clip.py` | Script | Mixed active/history | Construction/training/analysis entry points; not executed by this audit |
| `scripts/run_corrected_homecredit_clip_pipelines.py` | Script | Final HC downstream protocol | LR/CB budgets 20/40, pools 60/100, no OOT tuning at lines 59-140 |
| `scripts/run_corrected_lendingclub_to_homecredit_transfer.py` | Script | Final reverse-transfer entry point | LC-trained → HC transfer orchestration; not executed |
| `docs/clip/*.md` | Documentation | Boundary evidence | Training input, text baseline, and contrastive-data contracts |
| `README.md:45-46` | Documentation | Current high-level claim | States corrected HC CLIP→mRMR and reverse-transfer implementation exist |
| `cleanup/audit/cleanup_summary.md` | Documentation/registry | Authoritative cleanup classification | Marks old-policy CLIP-v2 products invalid, incomplete runs failed, corrected roots as replacements |
| `cleanup/repository_cleanup_report.md` | Documentation | Current reproducibility caveat | Lines 13-15 and 126: 31 tests depend on missing saved CLIP roots; lines 22-23 and 59-62 define legacy mRMR |
| `docs/research_extension/foundation_protocol_freeze.md` | Documentation | Current protocol boundary | Lines 41 and 55-63 exclude CLIP from the later protocol unless a complete authenticated profile is restored |
| `results/final_research_package_v2/final_results_tables.csv` | Result | Later non-CLIP synthesis | Contains later canonical selector baselines; no corrected CLIP rows |
| `D:\python projects\Research_pre_cleanup_backup_20260704\results\corrected_homecredit_clip` | External historical result/checkpoint root | Corrected/final evidence | HC pairs, five checkpoints, representation metrics/rankings, downstream runs |
| `D:\python projects\Research_pre_cleanup_backup_20260704\results\corrected_lendingclub_to_homecredit_transfer` | External historical result/checkpoint root | Corrected/final evidence | LC pairing/training/anchors, HC frozen projection, LR/CB predictions and metrics |
| `D:\python projects\Research_pre_cleanup_backup_20260704\results\final_research_package_v2\final_results_tables.csv` | External historical result | Final CLIP-specific synthesis | Authenticated table containing corrected CLIP rows and historical comparators |
| External backup `results/clip`, `results/clip_v2`, `results/clip_pairing_repair` | External historical artifacts | Mixed base/invalid/superseded | Useful for chronology; old `clip_v2` trained/evaluation products are not final scientific evidence |
| Stability paths under `configs/protocols/homecredit_model_stability_2024_*`, `outputs/prompt_16_*`, `results/prompt_16_*` | Config/result | Active third-dataset evidence | No Stability-specific CLIP artifact found |

**CONFIRMED.** The current source still points several corrected HC config fields to `results/clip/...`, `results/clip_v2/...`, and `results/corrected_homecredit_clip/...`, but those active paths are incomplete. The current compatibility tests therefore skip/fail their required historical profile rather than proving the archived execution can be rerun in-place.

**CONFIRMED.** `scripts/analyze_corrected_homecredit_clip.py:40-41` expects `TrainingDataBundle.homecredit_pairs/homecredit_text/homecredit_stat`, while the current bundle API uses source-oriented fields such as `source_pairs/training_text/training_stat`. The saved ranking is valid historical evidence, but the script is stale relative to the current API and is not assumed runnable without code change.

## 4. CLIP Experiment Chronology

| Date | Commit/tag | Generation | Status and evidence |
|---|---|---|---|
| 2026-06-17 | `7dec622` | Preparation | Prototype/preparation: “prepared the repo for Application of CLip” |
| 2026-06-18 | `737486a`, tag `clip-prompts-1-2-verified` | Boundary/text baseline | Frozen input boundary and text embeddings introduced |
| 2026-06-19 | `16922a0` | Remediation | Data-boundary/readiness fixes |
| 2026-06-20 | `a88476b`, tag `clip-prompts-5-6-verified` | First trained integration | CLIP-style selectors trained/integrated |
| 2026-06-21 | `5159ead` | First downstream integration | CLIP with mRMR/Boruta |
| 2026-06-22 | `d4d8860`, tag `clip-v1-frozen` | V1 | Missingness-only study frozen; superseded methodologically by compact V2 |
| 2026-06-23 | `a510484`, tag `clip-v2-pipeline-ready` | V2 | Compact target-free statistical pipeline |
| 2026-06-25 | `0c340e5` | Paired CLIP | Paired semantic/statistical contrastive selection |
| 2026-06-26 | `f3a5b3e`, tag `corrected-homecredit-clip-v1` | Corrected HC source | Pairing repaired with `identity_equivalence_v2`; final HC evidence generation |
| 2026-06-27 | `cadb523` | LC source introduced | LendingClub-trained encoder with frozen Home Credit projection begins |
| 2026-06-30 | `a74a958` | Reverse transfer implementation | LC→HC pipeline implementation |
| 2026-06-30 | `53a407a` | Reverse transfer finalized | Validated reverse-transfer pipeline/registry evidence |
| 2026-07-04 | `af823ae` | Scientific cleanup | Old/invalid/duplicate results removed or migrated; corrected results retained externally |
| 2026-08 | current Stability work | Third dataset | Stability protocol/results exist; CLIP deliberately excluded from the newer protocol pending complete provenance/new protocol version |

**CONFIRMED.** At least three generations must remain distinct: V1 missingness-only, pre-repair compact/paired V2, and the corrected identity-equivalence studies. The final HC and LC-source studies share architecture and high-level training rules but are separate directional executions with different source splits, preprocessors, anchors, checkpoints, and output roots.

**CONFIRMED.** The old `clip_v2` downstream artifacts are not merely “older but usable”: `cleanup/audit/cleanup_summary.md:32-38` labels their contrastive/training/selector/evaluation/analysis lineage scientifically invalid. Failed reverse-transfer attempts are separately labeled incomplete at lines 39-40.

## 5. Final CLIP Methodology

### 5.1 Feature text representation

- **CONFIRMED.** Each original feature is rendered with template version `feature_text_v1` in `src/credit_risk_fs/clip/text_builder.py:11,42-62`:

  `Feature: {feature}. Description: {description}. Semantic group: {semantic_group}. Source or formula: {source}.`

- **CONFIRMED.** This is richer than a feature name alone. Missing required feature name/description fields fail unless an explicitly supplied fallback is available; whitespace is normalized.
- **CONFIRMED.** Text is encoded by frozen `sentence-transformers/all-MiniLM-L6-v2`, revision `main`, producing 384-dimensional float32 vectors. The sentence-transformer is put in evaluation mode, its parameters are frozen, and embeddings are L2-normalized. Evidence: `src/credit_risk_fs/clip/text_encoder.py:18-53`, corrected tensor manifests, and both training YAML files.
- **CONFIRMED.** Embeddings are cached as Parquet columns `embedding_0000...`; the cache identity includes dataset, feature name, rendered-text hash, model, revision, normalization, and template version. Evidence: `src/credit_risk_fs/clip/embedding_cache.py:23-41`.
- **CONFIRMED.** Historical caches were stored under `results/clip/text_baseline/`, including `homecredit_text_embeddings.parquet`, `lendingclub_v2_text_embeddings.parquet`, and `embedding_cache_manifest.json`. These are absent from the complete active profile but present in the external backup.

### 5.2 Statistical descriptors

**CONFIRMED.** Final schema `compact_target_free_v2` has a fixed ordered 13-dimensional vector (`src/credit_risk_fs/clip/statistical_schema_v2.py:9-50`):

| Order | Descriptor | Exact definition / role | Target use |
|---:|---|---|---|
| 1 | `missing_rate` | missing values / total rows | None |
| 2 | `unique_ratio` | unique non-missing values / non-missing rows | None |
| 3 | `concentration_share` | numeric zero share; categorical modal share; binary majority share | None |
| 4 | `signed_log_mean` | `sign(mean) * log1p(abs(mean))` for valid numeric statistics | None |
| 5 | `log_standard_deviation` | `log1p(population_std)` with `ddof=0` | None |
| 6 | `clipped_skewness` | numeric skewness when at least three values and nonzero std, clipped to [-10,10] | None |
| 7 | `normalized_entropy` | Shannon entropy divided by `log(k)` | None |
| 8-10 | `is_numeric`, `is_categorical`, `is_binary` | mutually informative type flags | None |
| 11-13 | `numeric_stats_valid`, `skewness_valid`, `entropy_valid` | validity flags | None |

- **CONFIRMED.** The first seven are continuous summary fields and the last six are indicators. The descriptor code forbids target, OOT, PSI, prediction, LLM rank, stable-core, and post-origination inputs. No target-dependent CLIP descriptor was found. Evidence: `src/credit_risk_fs/clip/statistical_schema_v2.py:43-50` and `statistical_view_v2.py`.
- **CONFIRMED — HC source transform.** The corrected HC path fits only on the Home Credit feature-training split: median/IQR robust scaling of the seven continuous fields, division by IQR, clipping to [-8,8], indicators left unscaled, then float32 conversion. External LC vectors are transformed without refitting. Evidence: `src/credit_risk_fs/clip/statistical_preprocessor_v2.py` and external corrected pair/tensor manifests.
- **CONFIRMED — LC source transform.** The LC-source study uses median imputation and standard scaling for all 13 fields, clipping disabled, fitted only on LC train. The saved preprocessor identifies `fit_dataset=lendingclub_v2`, `fit_split=train`, and hash `693133fd3cd2f8bae7144f328664b90d0124a5cb691ff2c4d517961ef3dbe350`. HC descriptors are recomputed from HC data, then transformed without external refit.
- **CONFIRMED.** The preprocessing difference is directional and source-specific; the two corrected studies must not be treated as one interchangeable fitted encoder package.

### 5.3 Encoder architecture

| Component | CONFIRMED final value |
|---|---|
| Text input | 384 |
| Statistical input | 13 |
| Text hidden width | 64 |
| Statistical hidden width | 16 |
| Shared output space | 32 |
| Head form | Linear → GELU → Dropout → Linear → L2 normalization |
| Dropout | 0.05 |
| Parameter sharing | Separate text/statistical heads; no shared head weights |
| Temperature | initial 0.07; non-trainable; clamp [0.02, 0.5] |
| Parameter count | 27,488 |

**CONFIRMED.** The implementation is `CreditRiskCLIP` in `src/credit_risk_fs/clip/model.py:14-86`; corrected YAML files repeat the dimensions. “Shared embedding” means a common 32-dimensional comparison space, not shared projection parameters.

### 5.4 Contrastive objective

- **CONFIRMED.** A positive pair is the text embedding and statistical vector for the same `(dataset, feature_name)` identity. Pair IDs also bind dataset, split, group, and source hashes. Source training and validation contain source positives; external data supplies positive pairs only and cannot enter source optimization/model selection.
- **CONFIRMED.** For a batch of normalized text projections `T` and statistical projections `S`, logits are `T @ S.T / temperature`; the target is the diagonal identity. Loss is symmetric: `(CE(logits, identity) + CE(logits.T, identity)) / 2`. Evidence: `src/credit_risk_fs/clip/loss.py:31-53`.
- **CONFIRMED.** Negative policy `identity_equivalence_v2` treats every off-diagonal in-batch source feature as negative except verified alias, exact DEV duplicate, or documented identity transform exclusions. Same-feature identity is the diagonal positive. Same-family, high text similarity, statistical similarity, and same-source-table relations are diagnostics only. Explicit hard negatives, cross-dataset negatives, and validation-as-training negatives are disabled.
- **CONFIRMED.** HC had 349 training positives, 87 validation positives, and 576 LC external positives; its saved exclusion mask produced zero exclusions/warnings. LC had 576 eligible features split into 395 training and 181 validation features; exact-duplicate identity grouping produced 272 directional exclusions.
- **CONFIRMED.** Validation records text→stat and stat→text Recall@1/5/10, reciprocal ranks, directional MRR, and average MRR. These are representation diagnostics, not classifier AUC.

### 5.5 Training and seeds

| Setting | CONFIRMED value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Batch size | 64; batches smaller than 2 skipped |
| Maximum epochs | 80 |
| Scheduler | NOT FOUND / not configured |
| Early stopping | patience 15, minimum loss improvement 0.0001 |
| Gradient clipping | enabled, norm 1.0 |
| Seeds | 11, 22, 33, 44, 55 |
| Determinism/device | deterministic algorithms, CPU policy |
| Epoch shuffle | deterministic seed derived from training seed and epoch |

**CONFIRMED.** Each seed trains a separate pair of projection heads. The frozen sentence encoder is not fine-tuned. Evidence: corrected training YAML files and `src/credit_risk_fs/clip/trainer.py:54-185,278-283`.

### 5.6 Checkpoint/model selection

- **CONFIRMED.** Within each seed, the best checkpoint is the minimum **source validation loss**, with the configured minimum improvement. MRR is logged but is not the selection criterion.
- **CONFIRMED — HC single-checkpoint diagnostic.** HC `training_summary.json` orders seed 55 first because its loss 3.4469790459 is the best of the five. `model_selection_manifest.json` selects seed 55 for the saved single-checkpoint learned-scoring artifacts.
- **CONFIRMED — HC downstream ranking.** The corrected HC downstream ranking does not use only seed 55. `scripts/analyze_corrected_homecredit_clip.py:34-88` builds a joint feature vector per seed, uses seed 11 as the reference, aligns seeds 22/33/44/55 by orthogonal Procrustes, averages, and L2-normalizes. Thus the final ranking is a five-seed consensus.
- **CONFIRMED — LC→HC ranking.** `src/credit_risk_fs/clip/reverse_transfer.py:763-804` requires exactly the same five seeds, aligns to seed 11, and averages all fixed-seed scores before ranking. It does not select a single “best seed” for downstream transfer.
- **NOT FOUND.** No saved downstream LR/CB AUC broken out per CLIP seed was located.

### 5.7 Feature scoring/ranking

- **CONFIRMED.** At scoring time, each feature's frozen text vector and target-free statistical vector are projected. Their L2-normalized mean is the joint representation. A source anchor centroid is calculated in the joint space and normalized; feature score is dot product/cosine similarity to that centroid. Evidence: `src/credit_risk_fs/clip/learned_scoring.py:46-103`.
- **CONFIRMED — HC anchor.** The downstream HC consensus uses the frozen 23-feature Home Credit stable-core anchor. The scoring implementation uses only approved source-training anchors; target/OOT labels are not inputs.
- **CONFIRMED — LC anchor.** The LC-source anchor has 23 LC features chosen from four equal-duration LC DEV-training subwindows using adjacent-window PSI ≤ 0.10 and maximum missing-rate difference ≤ 0.05, with source target/OOT/external data excluded. Stable-anchor selection ranks by PSI, missingness difference, then feature ID.
- **CONFIRMED — transfer mechanics.** HC target descriptors in LC→HC are recomputed target-side but transformed with frozen LC objects. Encoder/projector weights are frozen; `external_refit=false`; HC target and OOT are forbidden in projection. The saved reverse manifest projects 436 HC features.
- **CONFIRMED.** Rankings are dataset-specific because names, descriptions, raw distributions, type flags, and source-anchor similarities differ by dataset. CLIP scoring itself is label-free. Scores sort descending, with deterministic feature-name/feature-ID ordering used to break ties.
- **CONFIRMED.** CLIP ranks original model variables, not one-hot-expanded classifier columns. In HC, 436 CLIP identities are represented; the downstream matrix reports 529 total eligible raw model columns before the ranking is intersected with eligibility and screened.

### 5.8 CLIP → mRMR → classifier pipeline

1. **CONFIRMED.** Freeze a target-free CLIP ranking.
2. **CONFIRMED.** Intersect it with eligible original variables and take the top 60 for LR or top 100 for CatBoost.
3. **CONFIRMED.** Apply `FixedRankThenMRMRSelector` after the model's preliminary original-feature selection interface. mRMR uses target labels and is fitted only on the current target DEV training partition/fold.
4. **CONFIRMED.** Select K=20 original variables for LR and K=40 for CatBoost; no candidate-pool sweep is permitted.
5. **CONFIRMED.** Fit target-dataset preprocessing and the target classifier on DEV training data; use OOT only for final evaluation.

**CONFIRMED.** Historical CLIP “mRMR” is `RandomForestRelevanceMRMRSelector`: relevance is the mean impurity importance from 128-tree random forests, redundancy is mean absolute configured correlation to selected variables, redundancy is floored at 0.05, and greedy score is relevance/redundancy. Evidence: `src/credit_risk_fs/selectors/mrmr.py:19-29,59-142` and `cleanup/repository_cleanup_report.md:22-31,59-62`. It uses labels for relevance.

**CONFIRMED.** This must not be silently equated to current `mrmr_mutual_information`, which is a different selector generation and protocol. All audited corrected HC and LC→HC downstream runs use pools 60/100 and final K 20/40.

## 6. Source → Target Direction Semantics

**CONFIRMED.** `X-trained → Y` means that X supplies CLIP representation training pairs, source-fitted statistical normalization, trained projection heads, and the source anchor; Y supplies feature texts and freshly computed target-free descriptors for frozen projection/ranking. It does **not** mean an X-trained row-level credit classifier predicts Y.

| Direction | Representation training | Target descriptor handling / refit | Labels used by CLIP | Labels used after CLIP | Downstream fit/evaluation |
|---|---|---|---|---|---|
| HC→HC | HC feature train/validation split | HC descriptors; HC train-fitted transform; no external domain | None | HC DEV labels in mRMR and classifier | HC DEV folds/full DEV; HC OOT final |
| HC→LC | HC only | LC descriptors transformed as external data; no representation refit | None | None in located study | No LC LR/CB downstream run; config explicitly forbids it |
| LC→LC | LC feature train/validation split | LC descriptors with LC train-fitted transform | None | None in located study | Contrastive validation/checkpoints only |
| LC→HC | LC only | HC descriptors recomputed, then frozen LC preprocessor and five checkpoints; no HC representation refit | None | HC DEV labels in mRMR and classifier | HC DEV pooled OOF/full DEV; HC OOT final |

**CONFIRMED.** In LC→HC, the Home Credit target is absent from CLIP projection but present downstream. Therefore the final AUC measures transferred target-free representation ranking plus target-supervised feature reduction and target-trained classifiers; it is not zero-shot target prediction.

**CONFIRMED.** OOT observations/labels are excluded from CLIP, mRMR fitting, feature selection, and hyperparameter selection. They are used for final target-classifier evaluation.

## 7. Confirmed Direction Matrix

| CLIP source | Evaluation target | Evidence it was run | Final result artifact | Status |
|---|---|---|---|---|
| Home Credit | Home Credit | Corrected pairs/checkpoints, five-seed consensus ranking, LR/CB downstream runs | External backup `corrected_homecredit_clip/combined_pipeline/.../experiment_summary.csv` | **CONFIRMED** — full downstream |
| Home Credit | LendingClub v2 | 576 external positive pairs; `lendingclub_v2_joint_embeddings.parquet` and `lendingclub_v2_learned_scores.csv`; external policy forbids LC downstream | External backup `corrected_homecredit_clip/training/lendingclub_v2_learned_scores.csv` | **CONFIRMED** — representation-only; downstream AUC **NOT FOUND** |
| LendingClub v2 | LendingClub v2 | 395/181 source train/validation split and five checkpoint/representation-metric sets | External backup `corrected_lendingclub_to_homecredit_transfer/training/seeds/seed_*/...` | **CONFIRMED** — representation training/validation only; downstream AUC **NOT FOUND** |
| LendingClub v2 | Home Credit | Frozen five-seed reverse projection plus authenticated HC DEV OOF/OOT predictions | External backup `corrected_lendingclub_to_homecredit_transfer/downstream/{logistic_regression,catboost}/results/prediction_metrics.csv` | **CONFIRMED** — full downstream |

**NOT FOUND.** No independent full downstream HC→LC or LC→LC LR/CatBoost result artifact was located. Representation evidence for those cells must not be relabeled as classifier evaluation.

## 8. Verified CLIP Results

### Corrected CLIP downstream results

| Direction / model | DEV value and scope | OOT value and scope | Exact primary source | Authentication notes |
|---|---:|---:|---|---|
| HC→HC LR | `cv_auc_mean=0.7172285750378105`; five-fold arithmetic mean, SD 0.0122905676 | `oot_auc=0.7336398378322221`; 120,053 rows | `D:\python projects\Research_pre_cleanup_backup_20260704\results\corrected_homecredit_clip\combined_pipeline\runs\homecredit_lr_corrected_clip_then_mrmr\results\experiment_summary.csv`, row `corrected_clip_then_mrmr`, columns `cv_auc_mean`, `oot_auc` | Saved OOT predictions; no stable borrower IDs or pooled DEV OOF in final synthesis |
| HC→HC CatBoost | `cv_auc_mean=0.7480551123480346`; five-fold arithmetic mean, SD 0.0128500325 | `oot_auc=0.7626757739835102`; 120,053 rows | `D:\python projects\Research_pre_cleanup_backup_20260704\results\corrected_homecredit_clip\combined_pipeline\runs\homecredit_catboost_corrected_clip_then_mrmr\results\experiment_summary.csv`, row `corrected_clip_then_mrmr`, columns `cv_auc_mean`, `oot_auc` | Same limited-row-identity status |
| LC→HC LR | `auc=0.5890154000387499`; pooled `DEV_OOF`, 82,647 rows | `auc=0.573202684690282`; `oot_holdout_final_model`, 120,053 rows | `D:\python projects\Research_pre_cleanup_backup_20260704\results\corrected_lendingclub_to_homecredit_transfer\downstream\logistic_regression\results\prediction_metrics.csv`, rows `DEV_OOF`, `oot`, column `auc` | Saved predictions/manifests with stable `SK_ID_CURR`; AUC drop DEV−OOT 0.0158127153 |
| LC→HC CatBoost | `auc=0.6563253233976742`; pooled `DEV_OOF`, 82,647 rows | `auc=0.6766033514680966`; `oot_holdout_final_model`, 120,053 rows | `D:\python projects\Research_pre_cleanup_backup_20260704\results\corrected_lendingclub_to_homecredit_transfer\downstream\catboost\results\prediction_metrics.csv`, rows `DEV_OOF`, `oot`, column `auc` | Stable identities; AUC drop DEV−OOT -0.0202780281 |

**CONFIRMED.** The external CLIP-specific synthesis `D:\python projects\Research_pre_cleanup_backup_20260704\results\final_research_package_v2\final_results_tables.csv` repeats these OOT values and labels HC rows `authenticated_oot_limited_row_identity` and reverse-transfer rows `authenticated_direct_predictions_and_manifests`. The direct per-run files above are the primary metric sources.

### Historical ordinary mRMR comparator

| Model | OOT AUC | Source / interpretation | Status |
|---|---:|---|---|
| LR | 0.7456889163655954 | External CLIP-specific `final_results_tables.csv`, row `Home Credit mRMR baseline`, source metric path `results/homecredit/lr/statistical/lr_statistical_mrmr_53a793cb32fe/results/experiment_summary.csv` | CONFIRMED historical comparator |
| CatBoost | 0.7668389381913719 | Same table, CatBoost row, source path `results/homecredit/catboost/statistical/catboost_statistical_mrmr_3858b721e537/results/experiment_summary.csv` | CONFIRMED historical comparator |

**CONFIRMED.** These two baselines use the historical RF-relevance/correlation `mrmr` implementation and are the like-era comparator for the corrected CLIP package.

### Conflicting/later metric layers

- **CONFIRMED — later canonical baseline conflict.** Current `results/final_research_package_v2/final_results_tables.csv:8-9` reports Home Credit `mrmr_mutual_information` OOT AUC 0.7351157546 (LR) and 0.7598987801 (CatBoost). These do not supersede the historical comparator within the CLIP study; they arise from a later selector/protocol generation.
- **CONFIRMED — Stability reporting conflict.** Current locked prediction evidence at `results/prompt_16_homecredit_model_stability_2024/oot_final_amended_v1/analysis/oot_metrics.csv:32-33` gives pure-LLM Stability OOT AUC 0.6984222198 (LR) and 0.7663857377 (CatBoost). Root `FINALIZED_METRICS_AND_WINNER_CURVES.md:18-41` instead reports externally supplied estimates 0.8344/0.8784 and explicitly says empirical curve AUCs do not reproduce the later supplied values. For executable evidence in this audit, the locked-vector `oot_metrics.csv` values are confirmed; supplied scorecard values remain a separate, conflicting reporting layer.
- **CONFIRMED.** The reported CLIP AUCs are downstream classifier AUCs. CLIP's own metrics are loss/retrieval metrics, and no conversion from MRR to AUC exists.

## 9. Seed and Checkpoint Evidence

### Home-Credit-trained encoder

| Seed | Best epoch | Final/stop epoch | Best validation loss | MRR at best-loss epoch | Checkpoint SHA-256 |
|---:|---:|---:|---:|---:|---|
| 11 | 13 | 28 | 3.5351297855 | 0.2124967799 | `163794df541fd0ffa2c5f7d54e5a995c1a8c7cc04aa3234858f9c0fd6db9eb8a` |
| 22 | 30 | 45 | 3.6069912910 | 0.2486129031 | `37aa0fb7cf49f79607b4e0ebd7dd67ecaca4af61f74f23b796e5f12a79c83feb` |
| 33 | 22 | 37 | 3.5961155891 | 0.2153129131 | `ecc516f9a5542779859078ae29e1502e9497ee1624f8dd53dbde81f59d3f4e19` |
| 44 | 26 | 41 | 3.6454682350 | 0.2451922596 | `a4dcad12cd1360c4e4c51a8cb87aa34d1b99163b725ab6d2b51402adeb740401` |
| 55 | 21 | 36 | 3.4469790459 | 0.2402670383 | `920a52d22ab32cfed0986db19f8b5e91207c841a7d95f5241be2ccb871d0448b` |

**CONFIRMED.** Primary evidence is external `corrected_homecredit_clip/training/training_summary.json` plus each `seeds/seed_*/checkpoint_manifest.json`. All checkpoints have the declared 27,488-parameter architecture. Seed 55 wins the single-checkpoint minimum-loss selection even though seed 22 has the largest MRR in this table; that directly proves MRR was not the model-selection criterion.

### LendingClub-trained encoder

| Seed | Best epoch | Best validation loss | Checkpoint bytes | Checkpoint SHA-256 |
|---:|---:|---:|---:|---|
| 11 | 45 | 4.0372982025 | 114,817 | `1b17c0272b72757675a15152639772b4a77d47348923f66f56b50321f6984d1b` |
| 22 | 44 | 3.9427235126 | 114,817 | `a7cef93c01fa22694855aee1e5501b80404324624d107774532d15b123077a3e` |
| 33 | 31 | 3.9929618835 | 114,817 | `1adf078f57c28e3d65ed9f61b9c914e9172099eeec99f5ce4cfb8afd4f941243` |
| 44 | 29 | 4.0084428787 | 114,817 | `d4a4a5e57db1f0e6c42efdbaf2642be3e1a00351e95cdd09b725ea35f0c9877c` |
| 55 | 27 | 3.9772253036 | 114,817 | `0cdbe2ee27a4a7d27f18c1c67054213b0f456360aa6c2c8af61fa0336323ae8f` |

**CONFIRMED.** Primary evidence is external `corrected_lendingclub_to_homecredit_transfer/training/seeds/seed_*/checkpoint_manifest.json` and the checkpoint files. Manifest hashes match the reverse-projection manifest. The five-seed reverse consensus consumes all of them; seed 22's lowest loss is not used to discard the other four.

**CONFIRMED.** A seed controls separate projection-head initialization/training, deterministic shuffling, and resulting representation. It is not a downstream classifier seed sweep. Downstream reported AUCs belong to the final consensus ranking pipeline, not to an individual checkpoint.

## 10. Home Credit / LendingClub Methodological Comparison

| Dimension | Corrected Home Credit source | Corrected LendingClub v2 source | Status / consequence |
|---|---|---|---|
| Feature identities | 436 total HC CLIP identities; feature-level 80/20 group split, 349/87, seed 42 | 576 LC identities grouped by exact equivalence; 395/181 after group-safe split | CONFIRMED; source validation composition differs |
| External role | 576 LC features are representation-only external positives | 436 HC features are frozen projection targets, then downstream target data | CONFIRMED; HC→LC and LC→HC are not symmetric full studies |
| Text encoder/template | Frozen MiniLM 384-d, `feature_text_v1` | Same | CONFIRMED parity |
| Statistical schema | `compact_target_free_v2`, 13 fixed fields | Same ordered 13 fields | CONFIRMED parity |
| Statistical preprocessing | HC-train median/IQR robust transform; continuous clip [-8,8], flags unscaled | LC-train median + standard scaling of all 13; clipping off | CONFIRMED directional difference |
| Contrastive model/loss | 384/13 → 64/16 → 32, symmetric CE, temperature 0.07 | Same | CONFIRMED parity |
| Seeds/training controls | 11/22/33/44/55; AdamW, CPU, deterministic | Same | CONFIRMED parity |
| False-negative evidence | No mask-producing HC pairs in final manifest | Exact-duplicate identity groups generate 272 directional exclusions | CONFIRMED dataset-specific identity structure |
| Source anchor | 23 frozen HC stable-core variables | 23 LC variables stable across four DEV subwindows | CONFIRMED different anchor construction |
| Final score | five-seed Procrustes consensus cosine to HC anchor | five-seed Procrustes consensus score relative to LC anchors | CONFIRMED same aggregation family, different fitted objects |
| Target labels | HC labels after CLIP for mRMR/classifier | HC labels only after frozen transfer for mRMR/classifier | CONFIRMED CLIP itself target-free |
| OOT | HC OOT final | HC OOT final | CONFIRMED same evaluation target, not same representation source |

**CONFIRMED.** The corrected studies intentionally reuse an architectural protocol, not a single fitted representation. Their preprocessing and anchors bind each model to its source dataset.

**INFERRED.** Higher LC→HC CatBoost OOT than DEV OOF does not show superior representation transfer in isolation; it is a property of the complete transferred ranking + HC-supervised mRMR + HC-trained CatBoost pipeline under the particular temporal split.

## 11. Stability 2024 Current Pipeline

### 11.1 Dataset and target

| Item | Status | Verified value / evidence |
|---|---|---|
| Dataset | CONFIRMED | `Home Credit - Credit Risk Model Stability 2024`; ID `homecredit_model_stability_2024`; local root `data/homecredit_model_stability_2024` |
| Raw scope | CONFIRMED | `feature_definitions.csv`, `train_base`, all train depth-0/depth-1 Parquet; depth-2 excluded; 19 included raw files |
| Base rows/IDs | CONFIRMED | 1,526,659 rows; `case_id` complete and unique |
| Target column/domain | CONFIRMED | `target`, int64, binary {0,1}, no missing |
| Target counts | CONFIRMED | 0: 1,478,665; 1: 47,994; prevalence 3.143728% |
| Class-1 orientation | CONFIRMED | Higher class-1 probability is treated as greater default risk by the current selector/model contracts |
| Exact native event definition/horizon | UNRESOLVED | No audited repository artifact precisely defines the Stability target event window; external dataset knowledge was intentionally not substituted |
| Feature-definition review | CONFIRMED | 461 candidate rows reviewed: 434 included, 27 excluded, 0 unresolved |
| Engineered predictor universe | CONFIRMED | 1,959 original modeling variables after deterministic depth-0 identity/depth-1 aggregation; matrix has 31 parts |
| Selector encoding width | CONFIRMED | One-to-one 1,959 float32 columns: 1,730 numeric, 229 categorical |
| Final model encoded width | CONFIRMED | Method/model dependent after feature selection and OHE; no single global count. Example MI-mRMR cells encode 20 selected originals to 92 LR columns and 40 originals to 125 CatBoost columns |

Primary evidence: `configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json:104-150,1413-1429`; `outputs/prompt_16_homecredit_model_stability_2024/matrix_v1/manifest.json:148`; `outputs/.../matrix_v1/metadata.json`; selector-cache `metadata.json` under `outputs/.../temp/final_amended_oot_v1/classical_selector_encoding_v2/`.

### 11.2 Temporal split

**CONFIRMED.** Chronological authority is `date_decision`; dates span 2019-01-01 through 2020-10-05 (644 unique dates).

| Partition | Rule / date span | Rows | Positives | Prevalence |
|---|---|---:|---:|---:|
| DEV | `date_decision < 2020-02-26`; through 2020-02-25 | 1,221,743 | 39,645 | 3.244954% |
| OOT | `date_decision >= 2020-02-26`; through 2020-10-05 | 304,916 | 8,349 | 2.738131% |

**CONFIRMED.** OOT is 19.972764% of all rows. `outputs/.../matrix_v1/metadata.json:9836-9839` binds the rule to `GroupedTimeSeriesSplit`, five expanding splits, and a one-unique-date gap. Identical calendar dates remain in one time group.

| Fold | Expanding train rows / end | Validation rows / dates |
|---:|---|---|
| 1 | 200,661; through 2019-03-28 | 204,567; 2019-03-30–2019-06-19 |
| 2 | 402,103; through 2019-06-17 | 203,798; 2019-06-20–2019-08-24 |
| 3 | 604,598; through 2019-08-22 | 205,980; 2019-08-25–2019-10-27 |
| 4 | 810,904; through 2019-10-25 | 201,466; 2019-10-28–2019-12-20 |
| 5 | 1,012,061; through 2019-12-18 | 202,820; 2019-12-21–2020-02-25 |

**CONFIRMED.** Each preprocessing/selector fit is scoped to the current fold's training rows; full-DEV fit is required before OOT transform. OOT is not used for fitting or early stopping.

### 11.3 Preprocessing

- **CONFIRMED — selector stage.** `OriginalFeatureNumericEncoder` preserves exactly one column per original engineered feature. Numeric nonfinite values become missing, are filled with training-only median (fallback 0), and cast float32. Categorical values use a training-only deterministic category map with an explicit missing token; unseen categories map to -1. Evidence: `src/credit_risk_fs/preprocessing/encoding.py:13-...` and selector-cache metadata.
- **CONFIRMED — model stage for both LR and CatBoost.** `SparsePreprocessor` is fitted only on the relevant training partition/full DEV. Numeric variables receive training mean imputation (fallback 0), centered `StandardScaler`, and float32 sparse output. Categorical variables receive `Missing` fill and `OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse float32)`. Evidence: `src/credit_risk_fs/preprocessing/encoding.py:294-319,441-589` and `src/credit_risk_fs/experiments/prompt_16_third_dataset.py:1590-...`.
- **CONFIRMED.** CatBoost receives this OHE numeric CSR representation; the Stability implementation does not use CatBoost's native categorical feature path.
- **CONFIRMED.** The original-feature identity mapping is retained through selection, while a selected raw variable can expand into multiple OHE columns for final modeling.
- **NOT FOUND.** No final preprocessing step that globally removes duplicate or constant predictors was located. All-missing/constant training features receive deterministic fallback mappings rather than outcome-based deletion.
- **CONFIRMED.** The 1,068-feature LLM supplement is a separate availability-filtered universe: retain a predictor when fold-1 training missing rate ≤0.90. It drops 891 of 1,959 without using target/validation data. This filter does not redefine the full classical universe.

### 11.4 LR protocol

| Setting | CONFIRMED value |
|---|---|
| Estimator | Logistic regression |
| Solver | `liblinear` |
| `max_iter` | 1,000 |
| Class weights | `balanced` |
| Random seed | 42 |
| Other material defaults | C=1, tolerance 0.0001, intercept enabled |
| Feature budget | K=20 original variables |
| Fit/evaluation | fold training → fold validation; full DEV → locked OOT |

Evidence: `src/credit_risk_fs/experiments/prompt_16_third_dataset.py:1990-2009`, protocol lock, and completed cell manifests.

### 11.5 CatBoost protocol

| Setting | CONFIRMED configured value |
|---|---|
| Iterations | 1,500 |
| Depth / grow policy | depth 10 / `Depthwise` |
| Learning rate | 0.01 |
| L2 leaf reg / min data in leaf | 95 / 290 |
| Column sampling | 0.9 |
| Random strength | 0.125 |
| Bootstrap / subsample | `Bernoulli` / 0.55 |
| Leaf estimation | `Newton` |
| Loss / eval metric | `Logloss` / `AUC` |
| Class balancing | `auto_class_weights=Balanced` |
| Seed / threads | 42 / 4 |
| Files | `allow_writing_files=False` |
| Feature budget | K=40 original variables |

**CONFIRMED — early-stopping nuance.** The model parameter set carries `early_stopping_rounds=150`, but both fold and full-DEV calls use `model.fit(..., eval_set=None)` (`src/credit_risk_fs/experiments/prompt_16_third_dataset.py:1679,2009`). Therefore no held-out target partition drives early stopping; operationally the protocol uses the fixed 1,500-iteration cap. The previously stated “fixed 1500, no early stopping” is accurate operationally but incomplete unless this dormant parameter is disclosed.

### 11.6 Existing feature-selection protocol

**CONFIRMED.** Primary metric is OOT ROC-AUC; the framework also records Gini, KS, lift/capture, precision/recall/F1/accuracy, log loss, Brier, feature/score drift, stability, and resources. Decision threshold maximizes KS on training data and is then frozen. Selector/model seed is 42; paired bootstrap seed is 20260721.

Current authenticated OOT accounting is 34/34 cells: 22 complete and 12 unavailable, primarily because of inherited/resource-infeasible selector runs (`final_evidence_manifest.json:7-8,169-174,221-232`). Relevant paths and OOT AUCs are:

| Method | LR OOT AUC / status | CatBoost OOT AUC / status | Integration relevance |
|---|---:|---:|---|
| Full features | unavailable | unavailable | 1,959-column resource stops; no final OOT AUC |
| Random-K | 0.6615114461 | 0.7726061207 | K=20/K=40 control |
| IV/WOE | 0.7008900760 | 0.7821626199 | complete classical baseline |
| `mrmr_mutual_information` | 0.7398852280 | 0.7870183350 | current canonical target-supervised mRMR |
| Lasso L1 | unavailable | unavailable | resource infeasible |
| `legacy_rf_relevance_corr` | 0.7221567110 | 0.8205667229 | closest named analogue to historical CLIP-era `mrmr` |
| CatBoost SHAP | 0.7335875761 | 0.8381833716 | complete |
| RFE CatBoost | 0.7386497743 | 0.8499687274 | complete |
| Boruta RF | unavailable | unavailable | resource infeasible |
| Statistical normalized average rank | unavailable | unavailable | resource infeasible |
| IV→Boruta, realized support 95 | 0.7607880348 | 0.7974527658 | natural-support result |
| IV→Boruta, realized support 178 | 0.7920505148 | 0.8289723980 | natural-support result |
| IV→Boruta, realized support 231 | 0.8029562236 | 0.8386415593 | natural-support result |
| Boruta→MI-mRMR | unavailable | unavailable | resource infeasible |
| Boruta→RFE | unavailable | unavailable | resource infeasible |
| Pure LLM (`llm`) | 0.6984222198 | 0.7663857377 | 1,068-feature target-free ranking universe, K=20/40 |
| Stable core + LLM fill | 0.6814770948 | 0.8145720972 | complete supplement |
| LLM→mRMR | **NOT FOUND** | **NOT FOUND** | No cell with this method in the final 34-cell Stability execution |

Primary evidence: `results/prompt_16_homecredit_model_stability_2024/oot_final_amended_v1/analysis/oot_metrics.csv:2-35`.

**UNRESOLVED/conflicting reporting.** `FINALIZED_METRICS_AND_WINNER_CURVES.md:69` mentions “LLM then mRMR” for a supplied Home Credit scorecard winner, but no such method exists in the current Stability execution registry. That claim cannot be used as evidence that the Stability downstream path already exists.

## 12. Stability CLIP-Prerequisite Inventory

| Requirement | Exists? | Path/evidence | Compatible with prior CLIP? | Notes |
|---|---|---|---|---|
| Raw feature list | YES | `outputs/.../matrix_v1/metadata.json`, `lineage.json`; protocol lock | PARTIAL | 1,959 engineered candidates, not the 436/576 CLIP raw identities |
| Human-readable feature names | PARTIAL | Engineered deterministic names in metadata/lineage | PARTIAL | Technically unique, often not naturally readable |
| Feature descriptions/text metadata | PARTIAL | Official `feature_definitions.csv`; `dev_llm_supplement_v3/llm_ranking/feature_descriptions.json` | NO | LLM artifact describes only the missingness-filtered 1,068 features and uses another rendering contract |
| Feature-type metadata | YES | Matrix Arrow types; lineage logical/source types | PARTIAL | Sufficient raw inputs exist, but CLIP type mapping has not been validated |
| Target-free statistical descriptors | NO | No Stability `compact_target_free_v2` artifact found | NO | 13-field vectors and source-fit manifest absent |
| Cached text embeddings | NO | No Stability MiniLM cache/manifest found | NO | LLM prompt descriptions are not embeddings |
| CLIP-ready feature table | NO | No Stability positive-pair/tensor-schema bundle found | NO | No joined text-vector/stat-vector identity table |
| CLIP config | NO | No Stability CLIP YAML/JSON found | NO | Generic and HC/LC configs only |
| CLIP checkpoint | NO | No Stability-trained checkpoint found | NO | HC/LC checkpoints are only in incomplete/external historical roots |
| Dataset-specific CLIP training script | NO | No Stability route found | NO | Current validator/config loader is HC/LC-specific |
| Transfer-scoring script | PARTIAL | Generic learned scoring and LC→HC reverse code exist | NO | Anchors/dataset maps and direction contracts are HC/LC-specific |
| mRMR integration | PARTIAL | Stability MI-mRMR works; historical `FixedRankThenMRMRSelector` exists | PARTIAL | No Stability CLIP-ranking handoff; selector-definition choice unresolved |
| LR K=20 downstream path | YES | Stability cell registry and LR execution | YES | Target pipeline exists independently of CLIP |
| CatBoost K=40 downstream path | YES | Stability cell registry and CB execution | YES | Target pipeline exists independently of CLIP |
| DEV/CV protocol | YES | Protocol lock and `GroupedTimeSeriesSplit` | PARTIAL | Row-temporal CV differs from feature-level CLIP train/validation splitting |
| OOT evaluation path | YES | Final OOT runner/manifests/predictions | YES | Must remain evaluation-only |

**CONFIRMED.** The LLM description artifact contains exactly 1,068 records because its amendment first freezes a fold-1-training missingness ≤0.90 universe (`configs/protocols/homecredit_model_stability_2024_v2/prompt_16_llm_supplement_amendment.json:189-200`). Its provenance identifies contract `prompt_16_target_free_adapter_lineage_v2`, prompt version `stability_expert_v5`, and a 100-feature ranking. This is useful lineage evidence but is not a drop-in `feature_text_v1` cache and does not cover all 1,959 candidates.

**CONFIRMED.** `src/credit_risk_fs/clip/training_validation.py:31-175` and its path defaults name only Home Credit and LendingClub artifacts. Generic mathematical components can accept 384/13 tensors, but the validated data boundary, dataset mapping, source anchor, and historical artifact profile do not include Stability.

## 13. Three-Dataset CLIP Compatibility Matrix

| Component | Home Credit | LendingClub v2 | Stability 2024 | Compatibility issue |
|---|---|---|---|---|
| Unit ranked by CLIP | 436 source feature identities | 576 source identities; 436 HC target identities in reverse projection | 1,959 engineered modeling variables derived from 434 included raw definition rows | Stability granularity must be frozen; raw source field versus engineered aggregation is methodologically different |
| Text contract | `feature_text_v1` with name, description, group, source/formula | Same contract | 1,068 LLM-rendered records under another contract; raw definitions/lineage for broader universe | No authenticated all-universe CLIP text table/cache |
| Text embedding | Frozen MiniLM, 384, normalized | Same | None | Missing Stability cache/model identity manifest |
| Descriptor schema | 13-field `compact_target_free_v2` | Same field order | Not generated | Fixed 13-dimensional input is mandatory |
| Descriptor fit scope | HC feature-training split | LC feature-training split | No CLIP feature split/source preprocessor | Must not substitute row-model preprocessing for feature-descriptor preprocessing |
| Source statistical transform | robust median/IQR, clip continuous | standard scaling all fields, no clipping | None | Direction-specific fitted transform must be declared |
| CLIP train/val split | feature identity groups, 349/87 | identity-equivalence groups, 395/181 | Row-time five-fold CV exists, but no feature-identity split | Row CV and representation feature split solve different problems |
| Identity-equivalence policy | Corrected, no final exclusions | Corrected, exact duplicate groups/exclusions | Not audited/constructed | False negatives among engineered aggregations are unknown |
| Source anchor | 23 HC stable-core features | 23 LC temporally stable features | None | Similarity score has no Stability source reference |
| Target transform | LC descriptors transformed without refit | HC descriptors transformed without refit | No route | Generic rule exists; Stability artifact/contract does not |
| Pre-model selection | CLIP top 60/100 → legacy mRMR | Same on HC target | Current MI-mRMR and other selectors | Historical versus canonical mRMR choice changes the experiment |
| Classifier encoding | Original variables selected before encoding | HC original variables selected before encoding | Original engineered variables selected, then sparse OHE/scaling | Compatible in principle only if CLIP ranks the same original identities |
| Downstream budgets | LR20/CB40 | LR20/CB40 on HC target | LR20/CB40 | Budget parity exists |
| Current code/data readiness | Historical corrected evidence, active profile incomplete | Historical corrected evidence, active profile incomplete | No CLIP artifacts | Cannot feed Stability without methodological and code-boundary changes |

**CONFIRMED.** Stability cannot be passed into the current validated CLIP pipeline “without changes.” The obstacle is not just a missing CLI option: the scientific unit of representation, full text universe, 13-vector evidence, feature-equivalence policy, source fit split, source anchor, and mRMR definition are all unfrozen.

**INFERRED.** Each engineered Stability variable could technically be treated as a CLIP feature identity because lineage exposes source family, original field, aggregation, type, and definition. Doing so would create a new 1,959-identity methodology rather than reproduce the 436/576 raw-feature studies; this audit does not endorse or implement it.

## 14. Cross-Dataset Transfer Interpretation

**CONFIRMED.** The most precise description of LC-trained CLIP → HC is: **portability of a source-learned, target-free feature semantic/statistical representation and source-anchor ranking rule to a different dataset, followed by target-supervised feature reduction and target-trained prediction.** “Cross-dataset feature-semantic/statistical representation transfer” is also accurate.

The experiment legitimately tests:

- **CONFIRMED:** whether frozen LC-trained projection heads can embed HC feature text and HC target-free distributions in the same 32-dimensional comparison space without representation refit;
- **CONFIRMED:** whether similarity to an LC-derived stable source anchor induces a useful candidate ranking on HC;
- **CONFIRMED:** whether that transferred ranking remains useful after HC-label mRMR and HC-trained LR/CatBoost under HC DEV/OOT evaluation;
- **INFERRED:** a limited form of domain portability/generalization at the feature-representation and ranking-prior level.

It does **not** establish:

- **CONFIRMED NOT SUPPORTED:** row-level model transfer, because the credit classifier is trained on HC rows/labels;
- **CONFIRMED NOT SUPPORTED:** zero-shot default prediction, because HC labels enter mRMR and the downstream classifier;
- **CONFIRMED NOT SUPPORTED:** label-function transfer, because CLIP never observes source or target default labels;
- **CONFIRMED NOT SUPPORTED:** common raw-feature alignment, because LC and HC features are separately described and embedded rather than joined by shared column identity;
- **CONFIRMED NOT SUPPORTED:** causal influence of one dataset on another, causal feature importance, or production generalization;
- **CONFIRMED NOT SUPPORTED:** symmetric HC↔LC downstream evidence, because HC→LC stops at representation scoring;
- **UNRESOLVED:** whether the same portability claim will hold for Stability, whose feature granularity and description coverage differ materially.

**INFERRED.** Calling the study simply “domain transfer” is defensible only with the qualifier that transfer occurs in the feature-ranking representation. Calling it a transferred credit-risk model would be incorrect.

## 15. Missing Artifacts for Stability

Only missing Stability CLIP items are listed here:

- **NOT FOUND:** a frozen decision on the Stability CLIP feature universe (434 source fields, 1,959 engineered variables, the 1,068 availability subset, or another predeclared set);
- **NOT FOUND:** all-universe `feature_text_v1` records with semantic group and source/formula fields;
- **NOT FOUND:** Stability MiniLM text embeddings and authenticated cache manifest;
- **NOT FOUND:** raw `compact_target_free_v2` 13-field descriptor table;
- **NOT FOUND:** source-fitted Stability CLIP statistical-preprocessor artifact and fit-scope manifest;
- **NOT FOUND:** Stability CLIP feature split and identity-equivalence/false-negative policy evidence;
- **NOT FOUND:** positive-pair Parquet files, tensor schema, pair hashes, and CLIP-ready feature table;
- **NOT FOUND:** a Stability source-anchor definition/manifest;
- **NOT FOUND:** a Stability CLIP training configuration and dataset-specific validated route;
- **NOT FOUND:** Stability CLIP checkpoints and per-seed representation metrics;
- **NOT FOUND:** a frozen HC/LC→Stability transfer projection manifest/ranking;
- **NOT FOUND:** a Stability CLIP→mRMR method registry entry specifying historical RF-correlation versus canonical MI mRMR;
- **NOT FOUND:** authenticated Stability CLIP downstream predictions, DEV/OOF metrics, OOT metrics, or final synthesis row;
- **NOT FOUND:** a complete current active historical CLIP evidence profile that would let the existing 31 scientific assertions run without external restoration/remapping.

## 16. Methodological Questions That Must Be Decided Before Implementation

1. **UNRESOLVED:** Should Stability train its own CLIP encoder, be only a frozen transfer target, or participate in both roles?
2. **UNRESOLVED:** Is the reproducibility target the corrected HC-source generation, the corrected LC-source generation, or a newly versioned common protocol that preserves their documented source-specific differences?
3. **UNRESOLVED:** Which source→target cells are scientifically necessary: selected hypotheses or a full 3×3 matrix? Representation-only and full downstream cells must be predeclared separately.
4. **UNRESOLVED:** What is one Stability “feature” for CLIP: each of 1,959 engineered variables, each of 434 included source fields, or a predeclared filtered set?
5. **UNRESOLVED:** If engineered variables are used, how are aggregation semantics rendered into the exact `feature_text_v1` fields without reusing the incompatible 1,068-feature LLM prompt artifact as though it were a CLIP cache?
6. **UNRESOLVED:** Which rows define Stability's target-free descriptors when it is a source, and which source-fitted transform is frozen when it is a target?
7. **UNRESOLVED:** How are identity-equivalent Stability aggregations grouped so they do not become false in-batch negatives?
8. **UNRESOLVED:** What source anchor defines similarity for Stability-trained CLIP, and should it follow the HC stable-core rule, the LC temporal-subwindow rule, or a new predeclared rule?
9. **UNRESOLVED:** Must CLIP rank original engineered variables before `SparsePreprocessor` expansion? Existing evidence strongly favors that order, but the Stability granularity decision must make the identity mapping explicit.
10. **UNRESOLVED:** Will screening remain fixed at 60 (LR) and 100 (CatBoost), with K=20/K=40, or will a different pool be preregistered? No OOT-driven sweep should be implied by historical values.
11. **UNRESOLVED:** Which post-CLIP selector is scientifically intended: historical RF-relevance/correlation `mrmr` for replication, current `mrmr_mutual_information` for protocol alignment, or both as separately named studies?
12. **UNRESOLVED:** Should downstream AUC use mean fold AUC or pooled OOF AUC? Historical HC→HC and LC→HC report different DEV summaries and are not directly interchangeable.
13. **UNRESOLVED:** Should all five checkpoints enter a Procrustes consensus, or should one minimum-loss checkpoint drive scoring? Historical final downstream rankings use consensus despite single-checkpoint diagnostics.
14. **UNRESOLVED:** Which dataset is the Procrustes reference and which seed is the alignment reference? Existing corrected studies fix seed 11, but a new multi-source design would require a versioned rule.
15. **UNRESOLVED:** Must the complete historical CLIP profile be restored and authenticated before any Stability extension, as required by the later foundation protocol, or will a new protocol explicitly separate historical verification from prospective work?
16. **UNRESOLVED:** What exact dataset-native Stability target event and prediction horizon should appear in the preregistration? The repository currently does not document them precisely.

## 17. Evidence Index

| Claim / item | Status | Primary evidence path | Secondary evidence | Notes |
|---|---|---|---|---|
| Corrected generation is final | CONFIRMED | `cleanup/audit/cleanup_summary.md:32-40` | Git `f3a5b3e`, corrected configs | Old-policy outputs explicitly invalid |
| Active historical profile incomplete | CONFIRMED | `docs/research_extension/foundation_protocol_freeze.md:55-63` | `cleanup/repository_cleanup_report.md:13-15,126` | 31 CLIP assertions require external complete profile |
| Feature text template | CONFIRMED | `src/credit_risk_fs/clip/text_builder.py:11,42-62` | `docs/clip/TEXT_BASELINE.md` | Name + description + group + source/formula |
| Frozen 384-d MiniLM | CONFIRMED | `src/credit_risk_fs/clip/text_encoder.py:18-53` | Corrected tensor manifests | Normalized, frozen |
| 13-field target-free schema | CONFIRMED | `src/credit_risk_fs/clip/statistical_schema_v2.py:9-50` | `statistical_view_v2.py` | Fixed order `compact_target_free_v2` |
| Architecture/temperature | CONFIRMED | `src/credit_risk_fs/clip/model.py:14-86` | Corrected training YAMLs | 384/13→64/16→32; 27,488 params |
| Symmetric contrastive loss | CONFIRMED | `src/credit_risk_fs/clip/loss.py:31-53` | Pair/negative manifests | Diagonal positives |
| Five deterministic seeds | CONFIRMED | `configs/corrected_homecredit_clip/training.yaml:31-49` | LC training YAML; checkpoint manifests | 11,22,33,44,55 |
| Checkpoint criterion | CONFIRMED | `src/credit_risk_fs/clip/trainer.py:151-185` | `training_summary.json` | Minimum source validation loss |
| Five-seed HC consensus | CONFIRMED | `scripts/analyze_corrected_homecredit_clip.py:34-88` | saved ranking/analysis manifests | Script currently API-stale |
| Five-seed reverse consensus | CONFIRMED | `src/credit_risk_fs/clip/reverse_transfer.py:763-804` | `reverse_projection_manifest.json` | Seed 11 reference |
| CLIP ranking anchor/cosine | CONFIRMED | `src/credit_risk_fs/clip/learned_scoring.py:46-103` | source-anchor manifests | Label-free scoring |
| CLIP→mRMR ordering | CONFIRMED | `src/credit_risk_fs/selectors/fixed_rank_then_mrmr.py:11-77` | downstream configs/scripts | Pools 60/100, K20/K40 |
| Historical mRMR definition | CONFIRMED | `src/credit_risk_fs/selectors/mrmr.py:19-29,59-142,219` | `cleanup/repository_cleanup_report.md:22-31,59-62` | RF relevance, correlation redundancy |
| HC→HC AUCs | CONFIRMED | External corrected HC per-run `experiment_summary.csv` files | External CLIP-specific final table | Mean fold AUC + OOT |
| LC→HC AUCs | CONFIRMED | External reverse `prediction_metrics.csv` files | External CLIP-specific final table | Pooled DEV OOF + OOT |
| Historical HC mRMR baselines | CONFIRMED | External CLIP-specific `final_results_tables.csv` | historical source metric paths in table | Not current MI-mRMR |
| No per-seed downstream AUC | NOT FOUND | Checkpoint/seed/result inventory | final synthesis | Representation metrics per seed only |
| Stability dataset/split | CONFIRMED | `configs/protocols/homecredit_model_stability_2024_v1/third_dataset_protocol_lock.json:104-150,1413-1749` | matrix metadata | 1,526,659 rows; fixed DEV/OOT |
| Stability matrix 1,959 | CONFIRMED | `outputs/.../matrix_v1/manifest.json:148` | matrix metadata/lineage | 31 parts |
| Stability selector encoding | CONFIRMED | `outputs/.../classical_selector_encoding_v2/metadata.json` | `OriginalFeatureNumericEncoder` | 1,730 numeric + 229 categorical |
| Stability model preprocessing | CONFIRMED | `src/credit_risk_fs/preprocessing/encoding.py:294-319,441-589` | cell preprocessing manifests | Sparse scaled numeric + OHE categorical |
| CatBoost fixed-cap behavior | CONFIRMED | `src/credit_risk_fs/experiments/prompt_16_third_dataset.py:1679,2009` | protocol parameters | `eval_set=None`; no active early stop |
| Stability final cell accounting | CONFIRMED | `results/.../oot_final_amended_v1/final_evidence_manifest.json:7-8,169-174,221-232` | `analysis/oot_metrics.csv` | 22 complete, 12 unavailable |
| Stability LLM descriptions | CONFIRMED | `results/.../dev_llm_supplement_v3/llm_ranking/provenance_freeze.json` | amendment lines 189-200 | Partial coverage: 1,068 descriptions, not 1,959 |
| Stability CLIP artifacts | NOT FOUND | Repository-wide CLIP/Stability inventory | training validator hard-coded HC/LC | Prerequisite categories in Section 12 |
| Stability exact target horizon | UNRESOLVED | Protocol/code/documentation search | binary/default-risk orientation contracts | No precise native horizon found |
| Stability pure-LLM AUC conflict | CONFIRMED conflict | `results/.../analysis/oot_metrics.csv:32-33` | `FINALIZED_METRICS_AND_WINNER_CURVES.md:7-8,18-41` | Locked empirical 0.6984/0.7664 vs supplied 0.8344/0.8784 |

## 18. Final Read-Only Integrity Check

- **CONFIRMED:** No pre-existing repository file was edited, reformatted, moved, renamed, or deleted.
- **CONFIRMED:** No experiment, test, notebook, model fit, CLIP training, downstream training, embedding job, or result-regeneration script was executed.
- **CONFIRMED:** No package was installed, removed, or upgraded; no environment/configuration file was changed.
- **CONFIRMED:** No branch/commit checkout, reset, revert, or commit was performed.
- **CONFIRMED:** The pre-existing untracked `METHODOLOGY_AUDIT_9_POINTS.md` was not touched.
- **CONFIRMED:** The only file created by this audit is repository-root `CLIP_STABILITY_READONLY_AUDIT.md`.
