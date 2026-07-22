# Cross-Dataset Voting Execution Specification v1

Date: 2026-07-22  
Scope: implementation specification only; no research execution  
Protocol: `cross_dataset_rank_voting_v1` 1.0.0  
Gate: `READY_FOR_PROMPT_4`

## Decision

The frozen protocol is internally consistent and implementable for Home Credit and LendingClub v2. The two eligible, non-duplicate supervised voters are executable for both datasets, the rank aggregation rule already has one canonical implementation, the downstream selector has an existing implementation, and every remaining integration gap is bounded to an existing owner. The matrix contains 12 prospective voting runs and four required reference reruns, for 16 registered runs when execution is later authorized.

This is forward parity, not historical replication. No legacy output is authenticated or relabelled as `cross_dataset_rank_voting_v1`. Corrected CLIP remains excluded, and no LLM, API, semantic-ranking, selector, model, or prediction workload was invoked for this specification.

## Frozen prerequisites and evidence

The baseline repository commit was `cbb049109ba8dc5c74cf9aaf6b39f2f070658da5`. Prompt 1.1's scientific gate is `PASS`, and Prompt 2's execution gate is approved following the user's confirmation that its live hardware readings are plausible. The confirmed operating envelope remains 10 physical/16 logical CPUs, 39.63/24.50 GiB total/available RAM, an NVIDIA RTX 4060 Laptop GPU with 8.00/6.72 GiB total/free VRAM, 254.10 GiB free on the results volume, 81.82 GiB free on the temp volume, one run, one fold, zero loader workers, and four estimator threads. No resource limit was raised.

`AGENTS.md` or an equivalent repository instruction file is not present. The prompt-listed `configs/datasets/lendingclub.yaml` is the older dataset definition; the frozen v2 equivalents used here are `configs/datasets/lendingclub_v2.yaml` and `configs/experiments/lendingclub_v2_matrix.yaml`.

| Frozen input | Version/profile | SHA-256 |
|---|---|---|
| `configs/protocols/credit_scoring_extension_v1.yaml` | 1.1.0 | `f4137e98b7c89a63e9a73bc495190858f416c586d237f8ac003c8fdb9e40bde0` |
| `configs/protocols/row_alignment_contract_v1.json` | 1.1.0 / `credit_risk_ordered_row_alignment_v1` | `fc1064069f5cc45d76fd34060bb506869683e953c354d3f1f1a13327d99e71a0` |
| `configs/protocols/cross_dataset_rank_voting_v1.yaml` | 1.0.0 | `51030e49716fae0a9c09b52628784a237e9581bb14c1a027e9c8238a575f0b49` |
| `configs/execution/local_laptop_safe_v1.yaml` | `local_laptop_safe_v1` | `1b77add8bf55096864934f6553aeba174cd22e2b112f53c7a45e5df327934012` |

The frozen protocol and policy files required no correction. The canonical common lifecycle remains `credit_risk_fs.experiments.execution.execute_registered_run`, used by the existing matrix and single-run entry points.

## Row contracts and targets

| Dataset | Identity | Positive target | DEV | OOT | Candidate universe |
|---|---|---|---:|---:|---:|
| Home Credit | `SK_ID_CURR` | `TARGET=1`, payment difficulty/default risk | 99,092 | 120,053 | 529 |
| LendingClub v2 | authenticated `loan_id` sidecar | `TARGET=1`, finalized bad outcome/default risk | 598,649 | 293,105 | 675 |

Home Credit ordered hashes are DEV ID `722f897d531415852d00904b3c9b34f664831126b8d7afe066f2536e0a25c9b7`, DEV ID+target `63e9afa9e572745520c96bdefb011ab7547854b7b018d4a5f9a80d5608e6851c`, OOT ID `3e90101f56774b7e44086b3bccfa91bd5be35f3afd60a8bce0e5a53313acda7a`, and OOT ID+target `f384956b737e46a4b7238064ff9ad2942abfc03eea86048e73f34751cde11895`.

LendingClub v2 ordered hashes are DEV ID `4d4cd7973f00eb946fef0a6bb09e61fe6d2b9be92892786f352446660c68818e`, DEV ID+target `1373baf30fc16b022d1d9059e400df65b5d3bb3f9ae76e3397b3172578f91590`, OOT ID `86840e88a94f78f328d62e36754f14377c1765a31fb3bc73cbb3f7b2d45f8092`, and OOT ID+target `9787d44d278d7965b0a966f19717dec1d9718ea8dfb6b96aa531d0f52d0a53e2`.

Both targets have the same probability orientation: higher class-1 probability means greater default risk. Dataset-specific identity, dates, rows, features, and business wording are predeclared inputs, not parity violations.

## Voter registry

| Voter | Status and provenance | Scope and output | Prompt 4 work |
|---|---|---|---|
| `rf_corr_mrmr` | Included. `credit_risk_fs.selectors.mrmr.RandomForestRelevanceMRMRSelector`; config key `mrmr`; source SHA-256 `30fabf22c3dad4e67b42d2723a8189200aa9b9035e7bacb3a81277682db099c7`. | Supervised RF impurity relevance plus correlation redundancy over the complete frozen universe. Rank 1 is best; top 300 are finite, and later features are missing. This is not canonical mutual-information mRMR. | Capture its original-feature order in the common long schema. `rf_importances_` may be recorded as relevance, but absent greedy-step scores remain null. |
| `boruta` | Included. `credit_risk_fs.selectors.boruta.BorutaSelector.feature_ranking_`; config key `boruta`; source SHA-256 `e96271f1b23c80a41fa64f90b2420ac6da96427d91bf66c86354adbe8ad84e25`. | Supervised Boruta shadow-feature ranking over the complete frozen universe. Rank 1 is best; all eligible features are ordered by Boruta rank and then feature name. | Capture the full original-feature ranking. The implementation does not expose a canonical raw score, so the shared field is null. |
| `domain_rule` | Excluded. Existing deterministic Home Credit rule has no verified LendingClub conceptual equivalent. It is not LLM-generated. | No vote. | None. |
| `api_backed_llm` | Excluded. No approved immutable ranking with complete LendingClub provenance and matching universe. | No vote; no API call or generation is allowed. | None. |
| `corrected_clip` | Excluded by the frozen protocol because complete corrected-CLIP provenance is absent. | No vote. | A future restoration would require a new protocol version. |
| `rfe` | Excluded as a voter and retained as the downstream selector. | Cannot cast a conceptually duplicate vote. | Registry/config integration only. |

Both included voters are available from executable source for both datasets. Every voter fit is restricted to the current DEV training fold; the final selection refit is restricted to full DEV. Neither voter may consume held-out DEV labels, OOT labels, OOT predictions, model performance, or future-period statistics. No data-independent ranking is eligible for cross-fold reuse.

`credit_risk_fs.experiments.rank_voting.VOTER_ALIASES` canonicalizes aliases before aggregation. A second alias for the same conceptual voter is rejected rather than counted as another vote.

## Aggregation and deterministic ties

The canonical implementation is `credit_risk_fs.experiments.rank_voting.aggregate_cross_dataset_rank_voting` (source SHA-256 `94b393cefe7266c6b7508613ca9d4e9d0f22608bf67bddf25ade8c75e4f5d0f6`). Prompt 4 must call it, not duplicate it.

For an eligible universe of size `N`, voter rank `r` becomes:

```text
normalized_score = 1 - (r - 1) / max(N - 1, 1)
```

Each eligible voter has weight 0.5. A missing rank contributes zero. Aggregation is the arithmetic mean. Input validation rejects unknown or forbidden features, duplicate features, NFC/casefold name collisions, duplicate voter aliases, non-finite scores, and rankings with the wrong fit scope. Sorting is aggregate score descending, voter presence count descending, best individual rank ascending, then normalized feature name ascending. Top `K` means aggregate ranks 1 through `K`, inclusive.

For `N=4`, suppose mRMR ranks `A,B,C` and Boruta ranks `B,C,D`. `B` scores `(2/3,1)` and ranks first at `5/6`; `C` scores `(1/3,2/3)` and ranks second at `1/2`; `A` scores `(1,0)` and also totals `1/2` but ranks third because it has only one voter present; `D` scores `(0,1/3)` and ranks fourth at `1/6`. If all numeric ties remain, canonical name `a` sorts before `z`.

## Downstream selection and models

The downstream selector is `credit_risk_fs.selectors.rfe.RFESelector` (source SHA-256 `149b1ecc3084f52b139cd2b2f1de210392f3d0ca109e19ade8c12c51ff1d322c`). It uses a CatBoost classifier with 500 iterations, depth 6, learning rate 0.05, step 10, seed 42, `allow_writing_files=false`, and the policy-resolved four threads. It receives only the current top-K original feature columns and the current training-fold target.

Prompt 4 must fit a selection-only numeric encoding inside the training boundary, preserve a one-to-one mapping to original candidate names, run RFE, and then fit the final model preprocessor only on the selected original columns. Candidate columns enter RFE in aggregate-rank order; the pinned scikit-learn implementation and that order are the deterministic fallback for equal importances. Selected features are emitted in aggregate-rank order. A candidate pool smaller than the exact final budget is a pre-fit error; the requested budget must never be silently reduced.

The frozen candidate budgets are 100, 200, and 300; 200 is primary, while 100 and 300 are sensitivities. LR must receive exactly 20 final features. CatBoost must receive exactly 40. The final LR configuration comes from `configs/base.yaml`; CatBoost must likewise resolve the frozen config, including 1,500 iterations rather than the class constructor's default.

## Leakage-safe fit sequence

For each of the five expanding grouped-time DEV folds with one unique-time-group gap:

```text
DEV training fold
  -> fit voter-local numeric encoding on training rows only
  -> fit rf_corr_mrmr and Boruta on training rows and labels only
  -> map ranks back to the frozen original candidate universe
  -> aggregate ranks with the canonical implementation
  -> take top K
  -> fit downstream RFE and its encoding on training rows and labels only
  -> fit final-model preprocessing on selected training columns only
  -> fit the final model on training rows and labels only
  -> predict and identity-reconcile the held-out DEV fold
```

The comparison DEV artifact is the complete union of held-out predictions: one out-of-fold prediction per eligible DEV identity. It is never the in-sample output of a full-DEV fit.

After configuration lock, the final path repeats both supervised voters, aggregation, RFE, preprocessing, and model fitting on full DEV only. Locked OOT is then transformed and predicted once. OOT labels and metrics may be used only for the predeclared final evaluation, never to choose a voter, pool, feature, model configuration, or implementation design.

## Forward parity review

The complete review is `cleanup/audits/cross_dataset_voting_execution_spec/forward_parity_table.csv`. The compact decision table is:

| Component | Home Credit | LendingClub v2 | Required parity | Status |
|---|---|---|---|---|
| Target and identity | `SK_ID_CURR`; class 1 payment difficulty | authenticated `loan_id`; class 1 finalized bad outcome | Same class-1 risk orientation and frozen identity contract | Verified |
| Candidate universe | 529 frozen post-engineering eligible features | 675 frozen inventory features | Same construction discipline; dataset inputs may differ | Verified; Prompt 4 freezes per-run manifest |
| Voters | fold-local `rf_corr_mrmr` and Boruta | same | Identical conceptual rules, weights, and fit boundary | Available; schema adapter required |
| Aggregation | common rank normalizer and tie chain | same | Identical code and rules | Implemented |
| Pools | 100 / 200 primary / 300 | same | Identical | Verified |
| Downstream selection | CatBoost-backed RFE | same | Identical selector; final budget depends on model | Available; registry/trace adapter required |
| LR | fold-local preprocessor; 20 features | same | Identical frozen model config and budget | Verified; effective config assertion required |
| CatBoost | fold-local preprocessor; 40 features | same, with predeclared LC categorical-frequency input | Identical frozen model config and budget | Verified; effective config assertion required |
| DEV/OOT | five grouped temporal OOF folds; full-DEV refit; one locked OOT score | same | Identical boundaries and output schema | Verified; artifact adapter required |
| Evaluation | AUC/Gini/KS/Lift@10/PSI/stability; paired family inference | same | Identical definitions; dataset/model families remain separate | Verified; family integration required |
| Execution | 1 run / 1 fold / 0 loader workers / 4 threads | same | Identical conservative policy | Verified |

There is no unresolved critical mismatch.

## Frozen run matrix

The matrix is `configs/experiments/cross_dataset_rank_voting_matrix_v1.yaml`, status `specification_only_not_authorized_for_execution`. Its deterministic expansion is:

| Count | Value |
|---|---:|
| Voting runs | 12 |
| Required reference reruns | 4 |
| Total future registered runs | 16 |
| DEV folds per run | 5 |
| Voting fold executions | 60 |
| Reference fold executions | 20 |
| Total DEV fold executions | 80 |
| Final full-DEV/OOT fits | 16 |
| Primary voting runs/comparisons (`K=200`) | 4 |
| Sensitivity voting runs/comparisons (`K=100,300`) | 8 |

The exact proposed order is:

1. `cdv1-001-homecredit-reference-rf-corr-mrmr-lr-s42`
2. `cdv1-002-homecredit-voting-k100-lr-s42`
3. `cdv1-003-homecredit-voting-k200-lr-s42`
4. `cdv1-004-homecredit-voting-k300-lr-s42`
5. `cdv1-005-homecredit-reference-rf-corr-mrmr-catboost-s42`
6. `cdv1-006-homecredit-voting-k100-catboost-s42`
7. `cdv1-007-homecredit-voting-k200-catboost-s42`
8. `cdv1-008-homecredit-voting-k300-catboost-s42`
9. `cdv1-009-lendingclub-v2-reference-rf-corr-mrmr-lr-s42`
10. `cdv1-010-lendingclub-v2-voting-k100-lr-s42`
11. `cdv1-011-lendingclub-v2-voting-k200-lr-s42`
12. `cdv1-012-lendingclub-v2-voting-k300-lr-s42`
13. `cdv1-013-lendingclub-v2-reference-rf-corr-mrmr-catboost-s42`
14. `cdv1-014-lendingclub-v2-voting-k100-catboost-s42`
15. `cdv1-015-lendingclub-v2-voting-k200-catboost-s42`
16. `cdv1-016-lendingclub-v2-voting-k300-catboost-s42`

Every run records dataset, protocol and row contract, method and voters, candidate designation, downstream selector, model and final budget, fold protocol, seeds, policy, projection reference, expected row hashes, comparison family, artifact set, dependencies, order, and enabled status. All entries are enabled as a future design but remain unauthorized for execution by this prompt.

## Reference audit

All four references must be rerun under the forward contract. The detailed evidence is in `cleanup/audits/cross_dataset_voting_execution_spec/reference_reuse_audit.csv`.

The two historical Home Credit paths labelled `mrmr` are only conceptually compatible. Their prediction schema is `y_true,y_pred_proba,y_pred`; stable identities, ordered row hashes, complete OOF provenance, current code provenance, and the forward artifact schema are missing, so filenames alone cannot authenticate reuse. No compatible saved LendingClub v2 reference exists. The matrix therefore schedules one `rf_corr_mrmr` reference for each dataset/model pair with the same row contract, final budget, model configuration, and seed policy as its voting family.

## Explicit projections and preflight requests

No stage may request `columns=None`. Identity, target, time, split, and leakage helpers are metadata and can never become selectable features. `credit_risk_fs.pipelines.common.calculate_required_columns` remains the projection constructor, and `credit_risk_fs.data.loaders.DataLoader` remains the consumer. Prompt 4 must extend that path with a validated candidate-universe manifest; it must not add a loader or pipeline.

| Stage | Smallest projection |
|---|---|
| Row validation | stable identity, `TARGET`, split/time |
| Voter generation | all 529 or 675 frozen candidates plus identity, target, and fold/time metadata; this is the only stage that genuinely needs the complete universe |
| Aggregation | validated ranking artifacts only; no dataset matrix |
| Downstream selection | current top-K candidate columns plus identity, target, and fold |
| Model fit/predict | exactly 20 LR or 40 CatBoost selected columns plus identity, target, and fold/order metadata |
| Evaluation | identity, target, prediction probability, split/fold, and order key |

Home Credit source projection is the explicit schema for the seven modeling tables: 122 application, 17 bureau, 3 bureau-balance, 23 credit-card, 8 installments, 8 POS cash, and 37 previous-application source columns. `application_test.csv` and `sample_submission.csv` are excluded. The frozen pipeline derives and validates 529 eligible features. The source files total 2,657,120,381 bytes; the retained 219,145 by 529 float32 dense lower bound is 463,710,820 bytes.

LendingClub projects the 675 frozen inventory features, target/time helpers, and authenticated identity sidecar. The candidate dtype inventory is 85 float64, 268 int8, 274 float32, 39 category, and 9 string columns. Feature and sidecar source files total 4,214,264,495 bytes; the retained 891,754 by 675 float32 dense lower bound is 2,407,735,800 bytes, while the known physical source lower bound is 3,639,867,300 bytes.

The four preflight shapes are Home Credit/LR, Home Credit/CatBoost, LendingClub/LR, and LendingClub/CatBoost. All are `pilot_required`: exact peak-RAM multipliers are not claimed because RF/Boruta shadow features, selection encoding, RFE, categorical transformations, and estimator-internal copies have not been measured together. The known artifact estimates are 32,113,280 bytes for Home Credit and 119,328,512 bytes for LendingClub. Accounting for an atomic temporary plus final artifact and applying the 2.5 disk safety factor requires 160,566,400 and 596,642,560 free bytes respectively. All runs are expected to use CPU, not GPU.

Prompt 4 may admit only one monitored pilot at a time after a fresh live preflight approves the unchanged limits. It must avoid simultaneous full DEV/OOT/fold copies, release intermediates between folds, measure parent and child RSS plus disk and wall time, and feed observations into later preflights. Exact fields and formulas are frozen in `cleanup/audits/cross_dataset_voting_execution_spec/preflight_request_specs.json`.

## Per-run artifacts and prediction contracts

The existing `result_paths`, `tracking`, `atomic_io`, `checkpointing`, and `execution.execute_registered_run` layers own all publication. A voting run manifest declares only applicable artifacts, normally:

- resolved `config.json` or `config.yaml`, `manifest.json`, `checkpoint.json`, and `run.log`;
- `voter_rankings.*`, `aggregate_ranking.*`, `candidate_features.csv`, `fold_selections.csv`, and `selected_features.csv`;
- `predictions_dev.*` and `predictions_oot.*`;
- `metrics.*`, `stability.*`, and applicable paired-inference artifacts;
- `resource_usage.json` and terminal `_SUCCESS`.

No placeholder artifact is allowed. Every publication uses an adjacent temporary, validation, atomic replace, manifest size/SHA-256 registration, and parent-owned terminal status.

The ranking schema is:

```text
dataset, run_id, fold, voter_id, normalized_feature_name,
raw_rank, raw_score_if_available, normalized_score, present,
aggregate_score, presence_count, best_individual_rank,
aggregate_rank, candidate_pool_membership
```

The fold-selection schema is:

```text
dataset, run_id, fold, model, candidate_budget,
feature, selected, selection_rank, selection_score_if_available
```

DEV predictions are complete OOF records with the roles `stable_row_id`, `target`, `prediction_probability`, `fold_id`, and `row_position_or_order_key`. OOT uses the same roles except `fold_id` is null. Metadata records artifact-order identity and identity+target hashes, source-contract identity and identity+target hashes, class-1 default-risk orientation, row count, positive-target count, artifact SHA-256, and artifact size. Source-order hashes validate frozen inputs; artifact-order hashes validate the independently fixed prediction order. Identity sets and identity-target mappings must reconcile exactly across both domains.

## Checkpoint and resume contract

| Stage | Atomic outputs and integrity | Resume behavior |
|---|---|---|
| `initialized` | resolved config, preflight, checkpoint, manifest; hashes and run identity agree | reuse after validation; otherwise restart registration/preflight |
| `data_validated` | split, candidate-universe, and load manifests; schema, candidate hash/count, row hashes | reuse validated inputs; otherwise repeat validation |
| `selection_completed` | voter/aggregate/candidate/fold/final selection artifacts; schema, K, budgets, names, fold scope, hashes | reuse completed validated folds/full-DEV selection; restart interrupted selection boundary |
| `model_fit_completed` | model, preprocessor, metadata; config/feature/preprocessing hashes and exact training scope | reuse validated fit; otherwise restart current fit |
| `dev_prediction_completed` | complete OOF artifact; one finite fold-provenanced prediction per DEV row and ordered hashes | reuse validated folds/artifact; otherwise restart incomplete fold/publication |
| `oot_prediction_completed` | OOT artifact; one finite prediction per row, no OOT fit, ordered hashes | reuse validated artifact; otherwise restart OOT publication |
| `evaluation_completed` | metrics, stability, and applicable paired inference; aligned identities and fixed statistical configuration | reuse validated evaluation; otherwise restart evaluation |
| `completed` | terminal manifest, resource usage, `_SUCCESS`, and index row agree | immutable; no resume |

A hard resource breach or interruption leaves no terminal success marker. Only the parent process may finalize status. Explicit resume first validates every claimed input/output and restarts the smallest invalid or incomplete stage.

## Prompt 4 acceptance checks

The pilot is acceptable only if all checks pass before inspecting performance:

- exact source DEV/OOT identity and identity+target hashes match;
- DEV contains one finite out-of-fold class-1 probability per eligible identity;
- OOT contains one finite class-1 probability per eligible identity;
- target, identity, time, and leakage columns never appear as selected features;
- LR has exactly 20 and CatBoost exactly 40 final features;
- candidate membership never exceeds the configured K;
- spies prove every supervised voter, selection encoding, selector, preprocessor, and model respects its training boundary;
- same-seed synthetic aggregation, selection, and expansion checks are deterministic;
- status, manifest, checkpoint, index, applicable-artifact declaration, sizes, and hashes agree;
- atomic-integrity validation and the unchanged resource policy pass;
- no legacy file changes.

Scientific performance is not an implementation acceptance criterion and cannot be used to alter the frozen design.

## Bounded Prompt 4 implementation gaps

The machine-readable source of truth is `cleanup/audits/cross_dataset_voting_execution_spec/implementation_map.json`.

| Gap | Existing owner to extend | Required outcome | Risk / pilot block |
|---|---|---|---|
| `gap_p4_01_rank_voting_adapter` | `pipelines.common`, `models._fold`, existing `rank_voting` | Execute both voters, canonical aggregation, K, and RFE inside fold/full-DEV boundaries | High leakage risk; blocks LR/CB |
| `gap_p4_02_voter_schema_adapter` | existing mRMR, Boruta, and rank-voting modules | Long schema, original-name provenance, correct missing/null scores, alias safety | High scientific risk; blocks LR/CB |
| `gap_p4_03_downstream_rfe_contract` | selector registry and experiment config | Wire existing RFE, exact budgets/thread cap, complete trace | Medium scientific/low implementation risk; blocks LR/CB |
| `gap_p4_04_matrix_wiring` | existing matrix and runner | Expand exact order and dispatch sequentially through common lifecycle | Low scientific/medium implementation risk; blocks LR/CB |
| `gap_p4_05_candidate_projection_manifest` | existing projection function and loader | Explicit projections, exact 529/675 names and hash, no fallback | High scientific risk; blocks LR/CB |
| `gap_p4_06_voting_artifact_checkpoint_validation` | existing atomic/checkpoint/tracking/execution/validator layers | Atomic schemas, stage resume, complete/interrupted validation | Medium risk; blocks LR/CB |
| `gap_p4_07_prediction_contract` | existing export and row-alignment modules | Complete OOF/OOT identity, hashes, order, probability orientation | High inference risk; blocks LR/CB |
| `gap_p4_08_reference_and_family_inference` | existing compare and paired-inference modules | Four new references; aligned four-run families; three-test Holm scope | High scientific/low implementation risk; does not block an individual pilot |
| `gap_p4_09_model_config_pin` | existing experiment config/runner validation | Assert effective LR/CB parameters and inject threads without drift | Medium scientific/low implementation risk; blocks LR/CB |

Forbidden implementation directions are a second runner, registry, aggregator, result tree, tracker, monitor, or checkpoint system; dataset-specific aggregation copies; parallel folds/runs; `columns=None`; semantic/LLM/CLIP generation; OOT-informed choices; or silent budget changes.

## Validation and limitations

The pre-edit repository gate passed. The baseline full suite reported 459 passed and 31 explicitly skipped legacy-CLIP integrations; those skips remain missing-evidence classifications rather than regressions. The repository validator reported zero active registered runs and zero active artifacts. Final focused/full test totals and preservation evidence are recorded in `cleanup/audits/cross_dataset_voting_execution_spec/validation_summary.json`.

This specification does not provide a measured selector/model peak-RAM multiplier and therefore does not authorize the full matrix. Prompt 4 must implement the bounded gaps and run only the requested monitored two-model pilot after explicit user approval. It must not start the remaining matrix based on this gate.
