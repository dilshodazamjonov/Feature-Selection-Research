# 1. Audit snapshot

- Audit time: 2026-06-28 11:14:55 +05:00 (Asia/Tashkent).
- Branch: `main`.
- Commit: `cadb523797fba54447a3841df074bb8fc859477c`.
- Initial Git status: clean.
- Initial `git diff --stat`: empty.
- Initial `git diff --check`: clean.
- Implementation changed-file count: 26 files in commit `cadb523` (17 source/script files, 4 configuration files, 4 tests, and 1 runbook).
- Pairing-policy version: `identity_equivalence_v2`.
- Source dataset: `lendingclub_v2`.
- External dataset: `homecredit`.
- Scientific execution: none.
- Audit actions: Git snapshot/diff inspection; all Prompt 1 file review; status/methodology/registry/manifest reads; safe CSV/Parquet schema reads; CLI help; standalone `all --dry-run`; AST parsing of all 17 changed Python files; command parsing/plan resolution for all six stages; 32 new targeted tests; a 42-test shared compatibility suite; and an 18-test artifact-independent shared regression suite.
- Test cache generation was disabled. Python bytecode generation was disabled. Pytest-created temporary material under `tests_runtime/pytest` was removed after testing; no test artifact remains in the repository.
- Deleted matrix code remains absent: `scripts/run_clip_final_comparison.py` and `src/credit_risk_fs/clip_final_comparison/` do not exist.

# 2. Executive verdict table

| Area | Status | Evidence | Blocking issue |
| ---- | ------ | -------- | -------------- |
| implementation completeness | FAIL | Real handlers exist in `src/credit_risk_fs/pipelines/reverse_transfer.py`, but several contracts are not enforced end to end. | Projection row alignment, resume protection, and required provenance are incomplete. |
| dataset roles | FAIL | `DatasetRoles` accepts either direction and rejects identical roles; the exact YAML resolves correctly. | Stage handlers and manifests use module constants rather than the declared roles, so a changed role config can be accepted while execution remains hard-coded to LendingClub→Home Credit. |
| pairing | FAIL | Same-feature positives and identity-only masking are implemented; source-table/text/statistical similarity remain diagnostic-only. | The real reverse caller never supplies verified-alias or documented-identity-transform evidence to splitting, anchor selection, or negative masking. |
| feature universe | PASS WITH LIMITATION | 796 evidence rows, 576 predeclared eligible rows, 576 unique LendingClub text rows, 678 raw columns; missing views fail rather than fabricate. | Text dataset/source-manifest provenance is not validated by `_prepare`; runtime count is manifest-driven. |
| DEV/OOT isolation | PASS | LendingClub descriptors/duplicates/anchor use `[-1795,-1065)`; Home Credit descriptors use prepared DEV `[-600,-240)`; no OOT input is passed to mRMR/model fitting. | None found in the direct data path. |
| statistical preprocessing | PASS WITH LIMITATION | Thirteen target-free descriptors; preprocessor fits only LendingClub training-split feature descriptors; validation and Home Credit use transform-only. | The configured training schema/manifests are bypassed by a manually constructed `TrainingDataBundle`. |
| feature split | FAIL | Deterministic, target-free, group-safe union-find split; exact duplicates cannot cross. | Known alias/identity-transform relations cannot enter the real caller, so its group-safety claim is incomplete. |
| source anchor | FAIL | Correct half-open windows, first-window frozen bins, fixed thresholds/rank, exact-23 failure, and per-seed normalized centroids are implemented. | Real caller omits alias/identity-transform evidence; source manifest omits preprocessor and per-seed checkpoint hashes required by the stated provenance contract. |
| CLIP training | PASS WITH LIMITATION | Architecture/hyperparameters match corrected Home Credit; seeds fixed to 11/22/33/44/55; selection is LendingClub validation loss. | Per-seed restart/resume is not implemented; a failed `train` rerun retrains and overwrites completed seeds. |
| frozen projection | FAIL | Source anchors and checkpoint metadata are validated before Home Credit loading; model parameters and source preprocessor are frozen. | `_project` sorts metadata/text by feature ID but passes the unsorted transformed statistical matrix, so semantic/statistical rows are not guaranteed to describe the same feature. |
| seed consensus | PASS WITH LIMITATION | All five seeds required; seed 11 fixed; Procrustes alignment and arithmetic score mean are deterministic. | Final score/projection artifacts do not propagate checkpoint, preprocessor, anchor, configuration, and data hashes. |
| candidate pools | PASS | Fixed 60/100 pools, deterministic ranks/ties, no sweep. Same budgets were also predeclared for corrected in-domain CLIP-only evaluation. | None. |
| mRMR | PASS | DEV target enters only fold/final mRMR; OOT is transform-only; raw and encoded widths plus lineage are persisted. | None found. |
| downstream evaluation | PASS WITH LIMITATION | Only LR/CatBoost are accepted; standard Home Credit windows, model parameters, class weighting, and seed are reused; no baseline/LLM route is referenced. | DEV prediction exports are full-DEV in-sample predictions, while reported DEV metrics are CV means. |
| prediction provenance | PASS | Home Credit IDs use `SK_ID_CURR`, are hashed with dataset, checked unique and disjoint, and all 14 required fields are exported for DEV/OOT. | None for row identity itself. |
| metrics | FAIL | OOT AUC/KS/PSI and CV metrics use established functions. | Metrics are computed before, not from, saved predictions; saved DEV predictions cannot reproduce reported CV DEV AUC/KS, so AUC-drop traceability is incomplete. |
| forward comparator | PASS WITH LIMITATION | Corrected Home Credit→LendingClub embeddings/scores exist. | No valid corrected forward downstream comparator exists; reverse execution alone cannot support bidirectional downstream competitiveness. |
| orchestration | FAIL | `prepare/train/project/evaluate/register/all` exist and dry-run is non-scientific. | Stage hashes/order/completeness are not persisted or validated robustly; `--resume` reruns completed stages. |
| runbook | FAIL | CLI syntax, paths, seeds, models, and stages parse. | Exact commands are not safe because projection and overwrite defects remain; advertised resume behavior is false. |
| overwrite/resume | FAIL | A completed stage without flags is refused. | `--resume` explicitly permits overwrite; `--skip-existing` trusts only `status=complete` and does not validate artifacts/configuration/hashes/model set. |
| registry | FAIL | Existing rows are read and incoming IDs are de-duplicated. | Registration is non-atomic, omits required source/checkpoint/preprocessor/anchor provenance, and does not validate the complete upstream stage chain. |
| tests | PASS WITH LIMITATION | New suite: 32/32 passed; artifact-independent shared suite: 18/18 passed. | Tests miss real projection ordering and assert incomplete overwrite behavior; 21 legacy tests fail because invalid artifacts are intentionally absent. |
| operational readiness | FAIL | CPU-only, sequential fixed seeds and seed-specific directories avoid uncontrolled concurrency. | Failed-stage resume overwrites valid seeds; raw prepare can hold a wide 598k-row DEV matrix without memory guidance. |

# 3. End-to-end call graph

| Stage | Real implementation | Input and prerequisite checks | Output/failure/resume behavior |
| ----- | ------------------- | ----------------------------- | ------------------------------ |
| CLI/config | `scripts/run_corrected_lendingclub_to_homecredit_transfer.py` → `run_cli`, `load_config_dir`, `resolve_plan` in `src/credit_risk_fs/pipelines/reverse_transfer.py` | Four YAML files; fixed seeds; LR/CatBoost subset; fixed budgets; policy string | Dry-run prints a plan. Role values are not required to equal the direction hard-coded by handlers. |
| prepare: universe | `_prepare` → `reconcile_feature_universe` | LendingClub evidence, raw CSV header, text path | Writes reconciliation. Missing descriptions/statistical eligibility/raw columns are explicit exclusions. |
| prepare: descriptors/duplicates/split | `_prepare` → `find_exact_dev_duplicate_pairs`, `deterministic_feature_split`, `build_statistical_view_frame` | LendingClub DEV rows only | Writes split, duplicate evidence, raw-to-transformed statistical vectors. Alias/identity-transform evidence is not passed. |
| prepare: source preprocessor | `_prepare` → `StatisticalPreprocessor.fit/transform`, `build_vector_frame` | Fits on feature rows assigned `train` | Writes source-fitted preprocessor and train/validation vectors; mismatched fit dataset/split raises. |
| prepare: anchor selection | `_prepare` → `build_feature_stability_evidence`, `select_anchor_members`, `write_anchor_selection_artifacts` | Four source DEV windows, training split, exact duplicates | Fails below 23 eligible members. No thresholds relax. Provenance manifest is incomplete. |
| prepare: pairs/policy | `_prepare` → `build_feature_positive_pairs`, `build_negative_policy` | Same feature ID/name must exist in both views | Writes train/validation positives and exclusions. No synthetic fallback. Alias/identity-transform caller inputs are absent. |
| train | `_train` → `train_seed` → `symmetric_masked_contrastive_loss`; `load_checkpoint`; `build_seed_anchor` | Pair/anchor/preprocessor files and selection hashes | Trains sequential seeds and writes checkpoints/anchors. No seed-safe resume; rerun overwrites. |
| project | `_project` → `prepare_modeling_data`, frozen `StatisticalPreprocessor.transform`, `frozen_project`, `aggregate_seed_embeddings` | All seed anchors validated before Home Credit loading; checkpoints/policy/hashes checked | Writes per-seed embeddings and consensus. Statistical rows are passed in pre-sort order against feature-ID-sorted text/metadata. |
| evaluate: pools/mRMR | `_evaluate` → `fixed_candidate_pool`, `FixedRankThenMRMRSelector` | Projection score file | Writes fixed pools, encoded lineage, widths, and model-specific selections. |
| evaluate: models/predictions/metrics | `_evaluate` → `prepare_modeling_data`, `run_experiment` in `src/credit_risk_fs/pipelines/common.py` | Home Credit DEV/OOT contract and standard fixed model config | Fits LR/CatBoost; writes DEV/OOT predictions and metrics. No baseline or LLM route. DEV metric/prediction scopes differ. |
| register | `_register` → `append_registry_rows` | Registry payload and selected run files | Sequentially rewrites four CSVs, guide, and summary manifest. Not transactional and incomplete in provenance validation. |
| stage control | `execute_plan` | Checks only stage-manifest existence/status | Completed stage is refused by default, skipped without integrity checks under `--skip-existing`, and overwritten under `--resume`. |

Searches across all 17 changed Python files found no `TODO`, `FIXME`, `NotImplementedError`, mock, placeholder, synthetic fallback, pending approval, or temporary shortcut in the scientific path. The two `pass` occurrences are an empty exception class and a path-relativization exception handler. The defects above are real logic/contract defects, not textual placeholders.

# 4. Leakage audit

| Object | Fit dataset | Fit split/scope | Target used | OOT used | External data used | Verdict |
| ------ | ----------- | --------------- | ----------: | -------: | -----------------: | ------- |
| LendingClub raw descriptor | lendingclub_v2 | DEV `[-1795,-1065)` | 0 | 0 | 0 | PASS |
| exact-duplicate detector | lendingclub_v2 | aligned DEV rows | 0 | 0 | 0 | PASS |
| feature train/validation split | lendingclub_v2 | eligible feature identities | 0 | 0 | 0 | PASS WITH LIMITATION: incomplete identity inputs |
| statistical preprocessor | lendingclub_v2 | contrastive training feature descriptors | 0 | 0 | 0 | PASS |
| source-anchor PSI bucketizers | lendingclub_v2 | first DEV subwindow only | 0 | 0 | 0 | PASS |
| source-anchor member selector | lendingclub_v2 | training-split features across four DEV subwindows | 0 | 0 | 0 | PASS WITH LIMITATION: incomplete identity inputs |
| CLIP projection heads | lendingclub_v2 | train features; validation-loss checkpoint selection | 0 | 0 | 0 | PASS |
| per-seed source anchor | lendingclub_v2 | fixed 23 train members in each seed space | 0 | 0 | 0 | PASS |
| Home Credit raw descriptor | homecredit | DEV `[-600,-240)` | 0 | 0 | 1 | PASS |
| Home Credit statistical transform | no fit | frozen LendingClub preprocessor | 0 | 0 | 1 | PASS |
| Home Credit frozen CLIP projection | no fit | frozen source heads/anchors | 0 | 0 | 1 | FAIL: row alignment is not guaranteed |
| mRMR | homecredit | DEV/fold-train target only | 1 | 0 | 1 | PASS |
| LR/CatBoost | homecredit | DEV/fold-train or full DEV | 1 | 0 | 1 | PASS |
| metrics | homecredit | CV DEV and held-out OOT | 1 | 1 | 1 | FAIL for saved-prediction traceability, not for representation leakage |

# 5. Pairing and split audit

- Positive rule: semantic view of feature `i` ↔ statistical view of the same feature `i`.
- Mask-producing relations: verified alias, exact DEV duplicate, documented identity transform; the diagonal is the positive.
- Diagnostic-only relations: source table, semantic/family relation, text similarity, statistical-descriptor equality, high correlation/business similarity.
- Both historical mask construction points are safe in isolation:
  - `build_negative_policy` produces exclusions only for the three allowed identity reasons.
  - `false_negative_mask` rejects unsupported reasons, stale policy, stale order hash, asymmetry, and zero-negative rows.
- Actual reverse caller: passes only exact duplicates. It passes neither configured verified aliases nor documented identity transforms to `deterministic_feature_split`, `build_feature_stability_evidence`, or `build_negative_policy`.
- Duplicate policy: exact equality of aligned LendingClub DEV values and missingness.
- Runtime feature universe: manifest-driven. Current lightweight evidence is 796 rows, 576 predeclared eligible rows, and 576 unique LendingClub text rows; it is not hard-coded to 576.
- Split: SHA-256 ordering with seed 42, semantic-group stratification, union-find identity groups, and a 20% validation target.
- Verdict: **FAIL** because the real identity relation set is incomplete even though the generic utility APIs can accept the missing relations.

# 6. Source-anchor audit

- Actual subwindows are four half-open intervals:
  - `[-1795,-1612.5)`
  - `[-1612.5,-1430)`
  - `[-1430,-1247.5)`
  - `[-1247.5,-1065)`
- Integer-day allocation:
  - `-1795…-1613` (183 days)
  - `-1612…-1431` (182 days)
  - `-1430…-1248` (183 days)
  - `-1247…-1066` (182 days)
  These are disjoint and cover every integer day in DEV exactly once. OOT starts at `-1065`.
- Numerical PSI: quantile edges are fitted on window 1, duplicate edges are de-duplicated, outer edges become ±infinity, constant features use `VALUE/OTHER/MISSING`, and later values use the frozen mapping.
- Categorical PSI: window-1 levels meeting count 50 are frozen; rare/unseen values map to `OTHER`; missing maps to `MISSING`.
- PSI: adjacent-window PSI with fixed epsilon `1e-6`. Bins are not refit.
- Missingness definition: `max(window missing rate) - min(window missing rate)`, equivalent to maximum pairwise absolute difference, not maximum adjacent difference.
- Support: at least 100 non-missing observations in every subwindow.
- Thresholds: max adjacent PSI ≤ 0.10; max missing-rate difference ≤ 0.05.
- Ranking: PSI ascending, missingness difference ascending, feature ID ascending.
- Selection: exactly 23; no relaxation/fallback; fewer than 23 raises.
- Identity filtering: exact duplicate groups work; alias and identity-transform APIs exist but their evidence is not supplied by `_prepare`.
- Per seed: approved members are selected in that seed’s normalized joint space, members are L2-normalized, centroid is averaged and L2-normalized, then frozen before Home Credit loading.
- Provenance present: dataset, DEV/subwindows, leakage flags, PSI scope, thresholds, ranking, counts, IDs, training-only flag, policy, configuration/data/evidence/member hashes, and per-seed anchor hashes.
- Provenance missing from the source anchor manifest: statistical-preprocessor hash and per-seed checkpoint hashes.
- Verdict: **FAIL** due to incomplete identity filtering and incomplete fail-closed provenance.

# 7. Training and checkpoint audit

- Architecture: text 384→64, statistical 13→16, shared 32, GELU, dropout 0.05; identical mechanical architecture to corrected Home Credit.
- Optimizer/training: AdamW, LR 0.001, weight decay 0.01, batch 64, max 80 epochs, patience 15, gradient clipping 1.0, CPU deterministic mode.
- Seeds: exactly 11, 22, 33, 44, 55.
- Selection: minimum LendingClub feature-validation loss only.
- Prohibited evidence: no Home Credit, downstream AUC/KS/PSI, or OOT metric enters checkpoint selection.
- Old-checkpoint rejection: source dataset, pairing policy, configuration, data, preprocessor, source-anchor-selection and checkpoint hashes are checked. Home Credit/old-policy checkpoints fail.
- Start behavior: `train_seed` initializes a fresh model.
- Resume safeguard: **FAIL**. `--resume` does not resume a seed/checkpoint; it allows the completed `train` stage to execute again and overwrite seed directories. Partial seed completion is not safely resumed.

# 8. Frozen projection audit

The intended source objects are:

1. LendingClub-fitted `statistical_preprocessor.joblib`.
2. Five LendingClub seed checkpoints.
3. Five per-seed LendingClub anchors.
4. Home Credit frozen text embeddings.
5. Home Credit DEV-only raw target-free descriptors.

The model and source objects are frozen, and every source anchor is validated before Home Credit is loaded. No external fit/partial-fit call exists.

Blocking alignment defect:

- `_project` builds `transformed` in descriptor construction order.
- It then sorts `text`, `descriptors`, and `features` by `feature_id`.
- It passes `transformed[DESCRIPTOR_COLUMNS_V2].to_numpy()` without applying the same sort/reindex.
- `frozen_project` checks only the external dataset and frozen parameters; it does not verify row IDs/names for the two arrays.

Therefore the statistical vector at row `j` can be paired with the text/name of a different feature. This violates the core corrected positive-pair contract and makes projected ranks scientifically invalid.

The final score CSV and projection manifest also omit source checkpoint, preprocessor, anchor, data-manifest, and configuration hashes required for downstream provenance.

# 9. Downstream feature-flow table

| Model | CLIP input universe | CLIP candidate count | mRMR input | Final count | DEV target use | OOT target use |
| ----- | ------------------: | -------------------: | ---------: | ----------: | -------------- | -------------- |
| Logistic Regression | compatible Home Credit target-free projected features | 60 | encoded columns derived from fixed 60 raw features | 20 | mRMR and model fitting only | evaluation only |
| CatBoost | compatible Home Credit target-free projected features | 100 | encoded columns derived from fixed 100 raw features | 40 | mRMR and model fitting only | evaluation only |

The 60/20 and 100/40 budgets are fixed in YAML and enforced by `resolve_plan`. They are also the fixed budgets used by the existing corrected Home Credit CLIP-only pipeline, not only the LLM-combined path, so budget comparability is defensible.

# 10. Prediction and metric audit

- Actual prediction fields: `stable_row_id,dataset,split,target,prediction_probability,predicted_class,run_id,method,model,source_training_dataset,external_dataset,data_manifest_hash,configuration_hash,pairing_policy_version`.
- Stable Home Credit identifier: `SK_ID_CURR`, hashed with dataset. Fallback to DataFrame index exists generically but is not used when Home Credit schema is valid.
- IDs are checked unique within DEV/OOT and disjoint across splits.
- Target/probability alignment is preserved from prepared frame to export.
- Both DEV and OOT files are written.
- OOT AUC/KS/log loss/Brier and score PSI use the same in-memory OOT arrays later saved.
- DEV AUC/KS in `experiment_summary.csv` are time-CV aggregates.
- `dev_predictions.csv` contains full-DEV final-model in-sample predictions, not out-of-fold predictions.
- Metrics are calculated before prediction files are written; no metric loader recomputes them from saved predictions.
- Consequently the saved DEV predictions cannot reproduce reported DEV AUC/KS or AUC drop. Metric provenance is not sufficient for the intended comparison.
- No unsupported uncertainty claim is made.

# 11. Bidirectional-comparison readiness

Classification: **ONLY CORRECTED REPRESENTATION/SCORE EVIDENCE EXISTS**.

- Corrected forward evidence: Home Credit-trained corrected embeddings and scores projected to LendingClub.
- Corrected forward downstream LR/CatBoost comparator: absent. Old Home Credit→LendingClub downstream CLIP metrics remain invalid under the old pairing policy.
- Corrected reverse evidence expected: LendingClub-trained representation, frozen Home Credit scores, and downstream Home Credit LR/CatBoost metrics.
- Representation-level comparison in both directions: possible after a valid reverse run.
- Downstream-performance comparison in both directions: not possible from the planned reverse run alone.
- The implementation/runbook does not include a narrow corrected forward downstream evaluation using existing corrected Home Credit checkpoints/scores.
- This absence is non-blocking for a correctly repaired reverse experiment, but blocking for any later statement that downstream performance is “competitive in both directions.”

# 12. Test results

| Command | Passed | Failed | Skipped | Runtime | Failure classification |
| ------- | -----: | -----: | ------: | ------- | ---------------------- |
| `pytest tests/clip/test_dataset_roles.py tests/clip/test_reverse_transfer.py tests/clip/test_source_anchor.py tests/pipelines/test_reverse_transfer_orchestrator.py -q -p no:cacheprovider` | 32 | 0 | 0 | 26.41 s | None |
| Shared 14-file compatibility suite covering checkpointing, pair building/validation, learned scoring, loss, negative policy, statistical preprocessing, training, and selectors | 21 | 21 | 0 | 4.07 s | All 21: `EXPECTED LEGACY-ARTIFACT ABSENCE`; every failed module is marked `legacy_clip_v1` and depends on intentionally absent invalid files under `results/clip/statistical_baseline`, `results/clip/contrastive_data`, or `results/clip/training`. No artifact was restored. |
| Artifact-independent shared suite: group split, loss, negative policy, pairing repair, statistical preprocessor | 18 | 0 | 0 | 1.53 s | None |
| AST parse of all changed Python files and parser/plan resolution for six stage commands | 23 checks | 0 | 0 | 25.3 s | None |
| Isolated exact CLI `all --dry-run` | 1 | 0 | 0 | 6.24 s | None |

An earlier combined `--help` plus dry-run shell invocation reached its 30.19-second shell timeout after printing the complete help and plan; the isolated rerun exited 0. This was not a scientific execution.

Test-quality limitations:

- No test checks that feature IDs/names align with the statistical matrix after `_project` sorting.
- The overwrite test checks refusal without flags and skipping, but does not test `--resume`; code inspection shows `--resume` overwrites.
- No test corrupts a completed artifact then verifies `--skip-existing` fails.
- No test verifies registry transactionality or full upstream hashes.
- The shared legacy tests should skip when invalid artifacts are intentionally absent rather than failing, but these failures do not demonstrate a shared-code regression.
- One shared test invocation transiently changed whitespace in `RUNS.md`; it was restored exactly and no diff remains.

# 13. Dry-run resolution

The real CLI resolved:

- Status: `dry_run_no_scientific_execution`.
- Stages: `prepare, train, project, evaluate, register`.
- Roles: source/training `lendingclub_v2`; external `homecredit`.
- Pairing: `identity_equivalence_v2`.
- Seeds: `11,22,33,44,55`.
- Models: `lr,catboost`.
- LendingClub DEV: `[-1795,-1065)`.
- Anchor boundaries: `[-1795,-1612.5,-1430,-1247.5,-1065]`.
- Anchor thresholds: PSI 0.10; missingness 0.05; minimum support 100; member count 23.
- Budgets: LR 60→20; CatBoost 100→40.
- Inputs: all six listed input paths exist.
- Output root: `results/corrected_lendingclub_to_homecredit_transfer`.
- Safeguards advertised: no external refit, source OOT, pre-mRMR external target, baseline, LLM, Home Credit retraining, UMAP, or completed-output overwrite.
- Safeguard discrepancy: actual `--resume` behavior contradicts `overwrite_completed_outputs=false`.
- Output-contract discrepancy: dry-run reports `pairing/lendingclub_v2_positive_pairs.parquet`, but `_prepare` writes separate `lendingclub_v2_train_positive_pairs.parquet` and `lendingclub_v2_validation_positive_pairs.parquet`.

# 14. Approved manual commands

No scientific command is approved. The runbook commands were syntactically parsed, but the scientific stages are unsafe until blocking repairs are implemented and re-audited.

| Order | Stage | Exact command | Inputs verified | Output path | Safe to run |
| ----: | ----- | ------------- | --------------- | ----------- | ----------- |
| 1 | dry-run | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage all --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost --dry-run` | YES; executed exit 0 | none | YES, dry-run only |
| 2 | prepare | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage prepare --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | Paths/schema YES; identity provenance NO | `results/corrected_lendingclub_to_homecredit_transfer` | NO |
| 3 | train | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage train --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | Runtime prerequisites not generated; resume unsafe | same root | NO |
| 4 | project | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage project --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | Static prerequisites understood; row alignment FAIL | same root | NO |
| 5 | evaluate | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage evaluate --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | Depends on invalid projection; metric traceability FAIL | same root | NO |
| 6 | register | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage register --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | Registry files exist; upstream/atomic safety FAIL | central registries | NO |

The runbook’s intermediate PowerShell prediction-inspection command is read-only, but it is not sufficient to validate metric scope or upstream hashes and therefore cannot cure the blockers.

# 15. Stop conditions

Stop immediately if:

- any scientific stage is about to run before the blockers in section 17 are repaired and re-audited;
- a Home Credit statistical row is not joined/reindexed by feature identity to its text row;
- pairing policy is not exactly `identity_equivalence_v2`;
- verified aliases or identity transforms are known but absent from split/mask/anchor evidence;
- source/external role metadata differs anywhere;
- LendingClub OOT, Home Credit OOT, or a target enters representation construction;
- anchor bins are refit after window 1, thresholds relax, or member count differs from 23;
- fewer than five seed artifacts are valid;
- an old/Home Credit-trained checkpoint is selected;
- `--resume` would rerun a completed stage or seed;
- `--skip-existing` has not verified artifact completeness and hashes;
- a prediction ID is missing, duplicated, or overlaps DEV/OOT;
- saved predictions cannot reproduce the claimed metric scope;
- registration would modify any registry before complete validation of all payloads;
- a baseline, LLM, Task 1, Task 3, deleted matrix, or old CLIP path is invoked.

# 16. Non-blocking limitations

- CPU-only five-seed training will be slow.
- The 60/100 pools are fixed and comparable, but no uncertainty analysis is planned.
- Corrected forward evidence is representation-only; this limits later claims but does not inherently invalidate a repaired reverse run.
- The wide LendingClub DEV read can be memory-intensive; resource guidance should be added, but this is secondary to the correctness blockers.
- Legacy artifact-dependent tests fail rather than skip when invalid artifacts are absent.

# 17. Blocking findings

1. **Frozen projection misalignment:** Home Credit descriptors are transformed before a feature-ID sort, then the unsorted statistical matrix is paired with sorted text/metadata. There is no identity assertion in `frozen_project`.
2. **Unsafe overwrite/resume:** `--resume` reruns completed stages and overwrites outputs; completed seeds are not preserved. `--skip-existing` validates only a status string, not configuration, requested models, hashes, or artifact completeness.
3. **Incomplete identity policy in the real caller:** verified aliases and documented identity transforms are never supplied to the split, source anchor, or negative policy.
4. **Role config is not authoritative:** the plan accepts generic declared roles, but handlers/manifests use hard-coded `SOURCE_DATASET`/`EXTERNAL_DATASET`; contradictory role config does not fail closed.
5. **Insufficient projection/anchor provenance:** final scores/projection manifest omit configuration, data, preprocessor, per-seed checkpoint, and per-seed anchor hashes; the source-anchor manifest omits preprocessor and per-seed checkpoint hashes.
6. **Metric provenance mismatch:** saved DEV predictions are in-sample full-DEV predictions while reported DEV AUC/KS are CV aggregates; claimed DEV metrics and AUC drop cannot be reconstructed from saved predictions.
7. **Registry safety is incomplete:** updates are non-atomic, required source/checkpoint/preprocessor/anchor provenance is not registered, and full upstream stage completion/hashes are not validated before writes.
8. **Dry-run artifact contract is inaccurate:** it advertises a source-pair path that the real prepare stage never creates.

# 18. Final verdict

The exact manual sequence is not scientifically safe. The frozen Home Credit projection can pair a feature’s text with another feature’s statistical vector, and completed/partial stages can be overwritten under the documented resume workflow. Identity evidence, metric traceability, projection provenance, and registry validation also fail the stated fail-closed contract. These are material implementation and scientific-provenance blockers; no real stage should be executed until they are repaired and independently re-audited.

NOT SAFE TO RUN
