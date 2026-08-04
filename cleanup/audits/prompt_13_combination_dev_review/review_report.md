# Combination DEV Authentication and OOT-Readiness Review

## Technical summary

The saved selector-combination DEV phase is scientifically and cryptographically complete: **90/90 selector fits and 120/120 held-out evaluation cells authenticate**, with exactly five folds for every frozen configuration. The preceding pilot remains authenticated at 18/18 fits and 24/24 evaluations. Combination OOT remains absent at 0/24, and the exact 24-cell OOT scope was frozen in a dedicated commit before any combination-DEV performance value was opened.

The decision is **ready_for_manual_oot** for the immutable 24-cell scope. This is authorization to run that exact scope later, not a claim that OOT has run. The current runner's completion lock opens its technical gate; the Prompt 13 review lock is a separate authenticated provenance authorization and is not represented as an execution-enforced hook.

DEV results remain diagnostic. No winner was chosen, no IV pool was pruned, and no method, budget, threshold, seed, fold, model, or ordering was changed. Each of the two Home Credit CatBoost Boruta-first configurations has a fold-1 **natural-support result of 26 against requested K=40**; folds 2–5 reached 40. The 26-feature cases are not described as matched K=40 and were not padded. The frozen OOT label retains 26 as its authenticated reference while requiring the future full-DEV refit to report its own realized support.

## The locked scope remained immutable before results were reviewed

The scope contains the approved methods in their immutable order: statistical normalized-average-rank voting; IV then Boruta with all frozen pools 100, 200, and 300; Boruta then mRMR mutual information; and Boruta then CatBoost RFE. Crossing those identities with two datasets, two final models, the applicable budget or pool variants, and seed 42 yields 18 unique full-DEV selector refits and 24 OOT evaluation cells.

The scope artifact authenticates its ordered identities, configuration hashes, Prompt 12 approval lineage, no-padding policy, and the declarations that combination DEV performance had not been opened and combination OOT had not been accessed. A safe plan-only call agreed field-for-field with all 18 selection and 24 evaluation identities while reporting zero raw paths resolved and zero workers started.

## Every saved DEV cell passed content-level authentication

Authentication went beyond completion counts. Each selector state and evaluation state passed its canonical internal digest and every contract file passed its recorded size and SHA-256. The audit verified unique and ordered cell identities, phase, fold, dataset, method, pool, model family, requested budget, seed, feature universe, configuration identity, selection lineage, terminal state, worker exit, and supervisor status.

For all 120 prediction artifacts, row counts matched state metadata, validation IDs were unique and in the authenticated order, targets were binary, probabilities were finite and in [0,1], and the ordered ID, ordered ID/target, and prediction-value hashes matched. Within each dataset-fold, all 24 configuration evaluations used the identical ordered validation IDs and targets. Every fit records `validation_targets_used_for_fit=false`, `opened_oot_paths=[]`, and `oot_rows_retained=0`.

There were no missing, extra, duplicate, partial, interrupted, failed, timed-out, stale, or hash-invalid active DEV artifacts. The active DEV contract is exactly 840/840 files. The DEV-completion lock binds the Prompt 12 approval lock and exact ordered 90-fit/120-cell inventories.

## DEV evidence shows variation without authorizing selection

Each number below is an expanding-window held-out fold result, never a full-DEV in-sample diagnostic. The detailed table retains all 120 fold rows and every consistently available primary metric. The 24-row configuration table reports mean, sample standard deviation, median, minimum, maximum, and selected-count range.

- **homecredit / catboost:** across the immutable configurations, mean held-out AUC spans 0.7477–0.7610 and mean KS spans 0.3739–0.3988. These ranges are diagnostic; no configuration was removed or promoted.
- **homecredit / lr:** across the immutable configurations, mean held-out AUC spans 0.7328–0.7447 and mean KS spans 0.3543–0.3721. These ranges are diagnostic; no configuration was removed or promoted.
- **lendingclub_v2 / catboost:** across the immutable configurations, mean held-out AUC spans 0.7283–0.7336 and mean KS spans 0.3306–0.3401. These ranges are diagnostic; no configuration was removed or promoted.
- **lendingclub_v2 / lr:** across the immutable configurations, mean held-out AUC spans 0.7147–0.7297 and mean KS spans 0.3117–0.3336. These ranges are diagnostic; no configuration was removed or promoted.

Fold-specific weakness is not hidden in the aggregate: `dev_fold_results.csv` contains all five AUC, KS, confusion-matrix, calibration, approval, and bad-rate values for every configuration. Differences among IV pools 100, 200, and 300 are retained as descriptive sensitivity only; all three remain frozen for OOT. With only five dependent temporal folds, the audit assigns **moderate descriptive evidence** to within-configuration patterns and **not_supported** to definitive superiority or significance claims.

## Strict baseline alignment is not supported by persisted Prompt 10 evidence

Prompt 10 remains authenticated at 36/36 cells and its DEV-only `cv_results.csv` files preserve five fold summaries per baseline. However, it did not persist held-out fold prediction vectors or ordered fold row/target identity hashes. Its saved `predictions_dev.csv` is a full-DEV in-sample final-model diagnostic with blank fold IDs, so it is not a substitute for held-out fold predictions.

Consequently, identical evaluation rows and target ordering cannot be authenticated between Prompt 10 and the combination DEV cells. All requested standalone-component, full-feature, and random-k pairings are therefore recorded as **not_supported**, with the exact reason, rather than reporting coerced fold deltas. No baseline OOT metric, prediction, or conclusion contributes to this review. Row-level paired inference, DeLong, bootstrap, pooled inference, ordinary fold t-tests, and Wilcoxon tests are not performed.

## Stability and stage support are authentic and feasible

Across the 24 configurations, mean pairwise fold Jaccard spans **0.359–0.844**. The stability file preserves fold selected counts, Jaccard mean/range, each feature's selection frequency, and Kuncheva only when a common candidate universe and equal selected-set size satisfy its assumptions. Variable natural-support sizes are explicitly marked not applicable for Kuncheva.

All 90 selector fits reached an authenticated completed or valid natural-support terminal state. `stage_support_audit.csv` records the support after every saved stage, candidate-universe and selected-feature hashes, fit provenance, shortfall warnings, and no-padding verification. The fold-1 row for each of the two Home Credit CatBoost Boruta-first configurations shows requested 40, realized 26, and `natural_support_26_of_requested_40`; folds 2–5 reached matched 40. No tentative or rejected feature was appended to either shortfall.

## Sequential resources and resume controls are safe for manual OOT

The DEV session authenticated 90 selector workers plus 120 evaluation workers under the sequential resource contract. Selector fits were reused where the LR and CatBoost evaluation cells shared a selection identity, yielding 90 fits rather than 120. Persisted worker intervals total **27.09 active-worker hours**; they are not presented as parallel-summed wall time. The matching event span is **27.10 hours**, and readiness checks recorded **0.04 seconds** in readiness-wait stages.

Peak process-tree RSS was **19.87 GiB** and the lowest persisted available system RAM was **9.79 GiB**. No selector or evaluation timeout, stop code, worker error, survivor process, partial file, or active/stale execution lock remains. Artifact size and stage-level timing/RAM detail are retained in `resource_summary.csv`. These controls are adequate for a later manual, sequential 24-cell OOT run; the OOT command must still be issued by the user.

## Focused, relevant, and complete validation passed

The focused Prompt 13 set passed 16 tests, the broader runner/authentication/resource/metrics/stability set passed 134 tests, and the complete repository suite passed 969 tests with 31 expected skips. The first full-suite shell wrapper expired at 120 seconds without a test failure and left no test process; the clean rerun completed in 238.15 seconds. Portable report schema validation and packaging passed. Its enhanced browser check is accurately limited to structural verification because no installed Chromium headless shell was available and none was downloaded.

## Limitations define what this review does not establish

- DEV is diagnostic and does not establish final superiority; OOT is the final evidence.
- The five folds are dependent expanding-window partitions, not independent experimental replicates.
- Prompt 10 baseline fold summaries cannot prove row/target pairing, so aligned baseline effects and inference are unavailable.
- Process timing comes from persisted worker lifecycle events and selector-reported fit time. It distinguishes summed active worker time from observed session span; non-worker Python overhead is not reconstructed as compute.
- The portable report is a reviewed snapshot. The CSV and JSON artifacts, not rounded narrative text, are the exact audit record.

## The exact frozen scope is ready for the user's manual OOT run

The audit found no authentication, contamination, feasibility, preservation, or resource-safety defect requiring a stop. The scientific decision is **ready_for_manual_oot**. Weak DEV performance is not a reason to modify the frozen scope, and every pool and natural-support configuration remains present. OOT must stay untouched until the user manually runs the exact command recorded in the decision artifact.

Generated from persisted authenticated artifacts only at 2026-08-04T15:30:01.424Z. No raw dataset path was resolved, no research loader or estimator was invoked, and no pilot, DEV, full-DEV refit, baseline, or OOT workload was executed.
