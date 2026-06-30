# 1. Audit snapshot

- Re-audit time: 2026-06-28 12:56:03 +05:00 (Asia/Tashkent).
- Branch: `main`.
- Commit: `cadb523797fba54447a3841df074bb8fc859477c`.
- Initial Git status:
  - modified: `REVERSE_TRANSFER_RUNBOOK.md`
  - modified: `configs/corrected_lendingclub_to_homecredit/contrastive_data.yaml`
  - modified: `src/credit_risk_fs/clip/reverse_transfer.py`
  - modified: `src/credit_risk_fs/clip/source_anchor.py`
  - modified: `src/credit_risk_fs/clip/trainer.py`
  - modified: `src/credit_risk_fs/pipelines/common.py`
  - modified: `src/credit_risk_fs/pipelines/reverse_transfer.py`
  - modified: `tests/clip/test_reverse_transfer.py`
  - modified: `tests/clip/test_source_anchor.py`
  - modified: `tests/pipelines/test_reverse_transfer_orchestrator.py`
  - untracked: `REVERSE_TRANSFER_PRE_RUN_AUDIT.md`
  - untracked: `configs/corrected_lendingclub_to_homecredit/identity_evidence.json`
- `git diff --check`: clean.
- Pairing policy: `identity_equivalence_v2`.
- Declared source/external roles: `lendingclub_v2` → `homecredit`.
- Changed implementation files inspected: all files listed above, the standalone script, four reverse-transfer YAML files, the implementation handoff files, prior audit, runbook, status report, pairing repair policy/invalidation evidence, and four central CSV registries.
- Audit execution:
  - CLI help.
  - Exact `all --dry-run`.
  - Stage-specific `prepare --dry-run`.
  - Claimed 44-test targeted suite.
  - Claimed 18-test artifact-independent corrected-CLIP suite.
  - Temporary-directory adversarial probes for stable IDs, skipped-stage upstream validation, and registry idempotency.
  - Registry/schema/hash and source-path inspections.
- Python bytecode and pytest cache generation were disabled; pytest temporary roots were outside the repository.
- No real stage ran. The scientific output root still contains only `implementation/`.

# 2. Previous blockers

| Previous blocker | Repair claimed | Direct evidence | Re-audit status |
| ---------------- | -------------- | --------------- | --------------- |
| Pairing/identity evidence not propagated | Explicit stable-ID JSON evidence reaches split, masks, and anchor | `load_identity_evidence`; `_prepare` passes relations to `deterministic_feature_split`, `build_feature_stability_evidence`, and `build_negative_policy`; mask reasons remain restricted | RESOLVED |
| Dataset roles not enforced throughout runtime | Exact reverse roles fail closed | `resolve_plan` requires LendingClub v2→Home Credit; source preprocessor/checkpoint/anchor validators reject role mismatches | RESOLVED |
| Home Credit views aligned by position | Feature-ID one-to-one alignment | `align_external_feature_views` validates IDs/datasets/names, sorts deterministically, and `_project` uses aligned `stat_*` rows | RESOLVED |
| Anchor/projection provenance incomplete | Complete source-chain hashes | Most hashes were added, but `raw_statistical_evidence_hash` hashes the eligibility CSV rather than the raw DEV values used for descriptors/PSI; not all stage manifests carry/validate current inputs/upstream state | NOT RESOLVED |
| Resume/skip/overwrite unsafe | Hash-validating skip and safe seed resume | Current-stage hashes are checked and completed outputs are not overwritten; adversarial probe proved completed `train` skips before validating invalid `prepare` upstream | NOT RESOLVED |
| Saved predictions did not reproduce metric scopes | Saved-file metric recomputation with explicit scopes | Official in-sample DEV/OOT metrics are reproducible, but persistent IDs do not fail closed and CV metrics are still produced without saved OOF predictions | NOT RESOLVED |
| Registry validation non-atomic/incomplete | Rollback transaction and idempotent append | Rollback mechanism exists, but realistic identical registry rows (`20.0` existing vs `20` incoming) are rejected as conflicts, violating idempotent second registration | NOT RESOLVED |

# 3. Executive readiness table

| Area | Status | Evidence | Blocking issue |
| ---- | ------ | -------- | -------------- |
| pairing and identity | PASS | Explicit source-role JSON, stable IDs, source-file hash, split/mask/anchor propagation, restricted mask reasons | None found |
| dataset roles | PASS | Exact reverse roles enforced before execution; source preprocessor/checkpoint/anchor role checks | None found |
| feature alignment | PASS | Both views sorted and compared by `feature_id`; shuffled/duplicate/missing/name/dataset tests pass | None found |
| DEV/OOT isolation | PASS | LendingClub representation uses `[-1795,-1065)`; Home Credit representation uses DEV only; OOT remains downstream evaluation | None found |
| anchor methodology | PASS | Fixed windows, first-window frozen bins, fixed thresholds/rank, exact 23, training split only | None found |
| anchor provenance | FAIL | Universe/split/identity/preprocessor/stability/member/config/data/seed hashes exist | “Raw statistical evidence” hash is bound to the eligibility manifest, not actual raw DEV statistical evidence |
| projection provenance | PASS WITH LIMITATION | Per-seed outputs and projection manifest carry roles/source/anchor/checkpoint/preprocessor/external/alignment/config/data hashes | Completed-stage skip does not revalidate current external inputs/upstream chain |
| five-seed consensus | PASS | Fixed `[11,22,33,44,55]`; complete set required; seed 11 reference | None found |
| candidate pools | PASS | Fixed 60 and 100, deterministic ranking, no search | None |
| mRMR | PASS | Home Credit DEV target only; encoded width/lineage persisted | None |
| downstream models | PASS WITH LIMITATION | Only LR/CatBoost, fixed Home Credit model configuration | Official DEV scope is explicitly in-sample, not a generalization estimate |
| prediction provenance | FAIL | Required columns and split overlap checks exist | `_stable_row_ids` silently hashes transient DataFrame index when no persistent identifier exists |
| metric reproducibility | FAIL | Saved-file AUC/KS/PSI recomputation passes synthetic test | CV fold metrics are still emitted without saved OOF predictions; no OOF artifact exists |
| resume | FAIL | Completed seeds can be reused; incompatible stage metadata is rejected | Some incomplete seed states fail rather than resume; upstream/current-input completeness is not universally revalidated |
| skip-existing | FAIL | Current stage status/artifact hashes are checked | Completed stage returns before validating its upstream stage; reproduced directly |
| overwrite protection | PASS | Completed current-stage artifacts are refused by default and skipped under validated flags | None for direct overwrite |
| registry atomicity | FAIL | Temporary files, rollback, transaction manifest, and conflict checks exist | Actual idempotency is type-fragile and fails for semantically identical numeric values |
| command isolation | PASS | Reverse script references no Task 1, UMAP, LLM, baseline, or deleted matrix entry point | None |
| forward comparator | PASS WITH LIMITATION | Corrected representation/score artifacts exist | No valid corrected downstream forward comparator |
| tests | PASS WITH LIMITATION | Claimed suites pass | They omit persistent-ID fallback, skipped-stage invalid-upstream, realistic registry idempotency, and OOF persistence |
| operational readiness | FAIL | CPU/sequential seed operation and isolated root are declared | Fail-closed resume/skip/provenance requirements remain unmet |

# 4. End-to-end execution graph

| Stage | Runtime path | Principal artifacts and checks |
| ----- | ------------ | ------------------------------ |
| CLI | `scripts/run_corrected_lendingclub_to_homecredit_transfer.py` → `run_cli` → `resolve_plan` | Four YAML files plus identity JSON; exact roles, seeds, models, budgets |
| prepare | `_prepare` → `reconcile_feature_universe` → duplicate discovery → identity-safe split → source descriptors/preprocessor → anchor selection → pair/mask construction | Reconciliation, identity relations/manifest, split, pairs, preprocessor, source-anchor selection, data manifest |
| train | `_train` → `train_seed` → checkpoint reload/validation → `build_seed_anchor` | Five seed checkpoints, validation scores, seed anchors, consensus validation scores |
| project | `_project` → validate source chain → load Home Credit DEV → source transform → `align_external_feature_views` → `frozen_project` → `aggregate_seed_embeddings` | Raw descriptors, statistical vectors, alignment manifest, five seed projections, consensus scores/embeddings |
| evaluate | `_evaluate` → fixed candidate pool → `prepare_modeling_data` → `run_experiment` | 60/20 LR and 100/40 CatBoost flows, selections, predictions, saved-file metrics |
| register | `_register` → validate stage manifests/artifacts → construct rows → `atomic_registry_transaction` | Four registries, guide, summary manifest, registration transaction manifest |

Isolation search found no Task 1, UMAP, LLM, deleted matrix, or baseline execution call in the standalone runtime path. No textual placeholder is present in the scientific path.

# 5. Leakage and fitting-scope table

| Object | Fit dataset | Fit scope | Target used | OOT used | External data used | Verdict |
| ------ | ----------- | --------- | ----------: | -------: | -----------------: | ------- |
| LendingClub descriptors | lendingclub_v2 | DEV `[-1795,-1065)` | 0 | 0 | 0 | PASS |
| duplicate detector | lendingclub_v2 | aligned DEV rows | 0 | 0 | 0 | PASS |
| feature split | lendingclub_v2 | eligible feature identities | 0 | 0 | 0 | PASS |
| statistical preprocessor | lendingclub_v2 | train-split feature descriptors | 0 | 0 | 0 | PASS |
| PSI bucketizers | lendingclub_v2 | first DEV subwindow | 0 | 0 | 0 | PASS |
| anchor selector | lendingclub_v2 | four DEV subwindows, train features only | 0 | 0 | 0 | PASS |
| CLIP heads | lendingclub_v2 | train features; source validation loss | 0 | 0 | 0 | PASS |
| seed anchor | lendingclub_v2 | fixed 23 training members | 0 | 0 | 0 | PASS |
| Home Credit descriptor | homecredit | DEV `[-600,-240)` | 0 | 0 | 1 | PASS |
| Home Credit transform/projection | no fit | frozen source objects | 0 | 0 | 1 | PASS |
| mRMR | homecredit | DEV/fold-train | 1 | 0 | 1 | PASS |
| LR/CatBoost | homecredit | DEV | 1 | 0 | 1 | PASS |
| OOT metrics | homecredit | held-out OOT evaluation | 1 | 1 | 1 | PASS |

# 6. Pairing and identity audit

- Policy: `identity_equivalence_v2`.
- Positive: semantic and statistical view of the same stable feature identity.
- Explicit evidence: `configs/corrected_lendingclub_to_homecredit/identity_evidence.json`.
- Current explicit alias and identity-transform lists are empty; this is explicit rather than inferred.
- Loader validates policy, source role, external role, stable names and stable feature IDs, and hashes the JSON plus normalized relation table.
- `_prepare` combines explicit relations with exact DEV duplicates before union-find splitting.
- The same explicit relations enter negative-policy construction.
- The same relations enter source-anchor unified identity conflict groups.
- `false_negative_mask` accepts only verified alias, exact DEV duplicate, and documented identity transform; stale policy/order and asymmetric masks fail.
- Source table, semantic family, text/statistical similarity, correlation-like diagnostics, and business grouping do not produce masks.
- Home Credit-labeled identity evidence fails source-role validation.

Verdict: PASS.

# 7. Alignment audit

- Key: `feature_id`, never row number.
- Required metadata: `feature_id`, `feature_name`, `dataset`.
- Cardinality: duplicate or missing IDs fail.
- Role: both views must be Home Credit.
- Set equality: semantic and statistical ID sets must match.
- Name equality: names must match after deterministic feature-ID ordering.
- Ordering: stable mergesort by `feature_id`.
- `_project` passes the aligned semantic embedding columns and aligned `stat_*` columns.
- Persisted evidence: source text hash, serialized statistical-vector hash, joined identity hash, alignment-manifest file hash, and explicit reconciliation reasons for model-universe features lacking text or deterministic identity.
- Tests cover shuffled order, missing semantic/statistical row, duplicate ID, conflicting name, and dataset mismatch.

Verdict: PASS.

# 8. Anchor and projection provenance audit

Anchor methodology is unchanged and correct:

- DEV `[-1795,-1065)`.
- Four half-open subwindows ending at `-1612.5`, `-1430`, `-1247.5`, and `-1065`.
- PSI ≤ 0.10.
- Max-minus-min missingness ≤ 0.05.
- Minimum non-missing support 100.
- Exactly 23 train-split members.
- Rank by PSI, missingness, feature ID.
- First-window numeric/categorical buckets are frozen; `MISSING` and `OTHER` are explicit.
- No target, OOT, Home Credit, threshold relaxation, validation fallback, or downstream criterion.

Most required provenance is present and validated:

- source feature-universe hash;
- split hash;
- identity-evidence hash;
- preprocessor hash;
- stability-evidence hash;
- member-table hash;
- configuration/data-manifest hashes;
- five checkpoint hashes;
- five anchor hashes;
- external universe/raw-descriptor/text/alignment hashes.

Blocking provenance defect:

- `_prepare` sets `raw_statistical_evidence_hash = sha256_file(evidence_path)`.
- `evidence_path` is `results/lendingclub_v2/analysis/clip_readiness/dev_only_clip_training_evidence.csv`, an eligibility/metadata table.
- It is not the raw LendingClub DEV values used by descriptor construction, duplicate discovery, or PSI.
- Therefore the manifest field does not bind the actual raw statistical evidence its name and contract claim to bind.

# 9. Resume and overwrite audit

Resolved behavior:

- Output must be under the dedicated reverse-transfer root.
- Default refuses a complete stage.
- Completion manifest is written after handler output/hash collection.
- Current-stage required artifacts and hashes are validated.
- Complete train seeds validate checkpoint/anchor hashes and metadata before reuse.
- Non-train partial-stage resume is refused.
- Unmanifested partial outputs are refused.

Blocking skip behavior:

- `execute_plan` validates a completed current stage and immediately `continue`s for `--skip-existing` or `--resume`.
- The upstream-stage validation code appears after that `continue`.
- Temporary adversarial probe created a valid completed `train` manifest and an invalid `prepare` manifest.
- `execute_plan(... stages=('train',), skip_existing=True)` returned successfully.
- Thus skipped stages do not prove current upstream completeness/hashes.

Additional gaps:

- `input_hashes` are added by `prepare`, not universally by project/evaluate.
- A completed project/evaluation can therefore be skipped without comparing all current external/modeling inputs.
- A “complete” resumed seed with checkpoint/anchor files but missing validation-score output enters the reuse branch and fails instead of resuming that incomplete seed.

Verdict: FAIL.

# 10. Prediction and metric audit

Persisted prediction columns include:

`stable_row_id,dataset,split,target,prediction_probability,predicted_class,run_id,method,model,source_training_dataset,external_dataset,data_manifest_hash,configuration_hash,pairing_policy_version,fit_scope`.

Positive findings:

- Home Credit normally supplies `SK_ID_CURR`.
- DEV/OOT IDs are checked unique and disjoint.
- DEV and OOT files are both saved.
- Official reverse DEV scope is named `dev_in_sample_final_model`.
- OOT scope is named `oot_holdout_final_model`.
- `prediction_metrics_from_saved_files` reads saved files and recomputes AUC, KS, PSI, row counts, and file hashes.
- Synthetic recomputation test passes.

Blocking stable-ID defect:

- `_stable_row_ids` falls back to `frame.index.astype(str)` when no persistent ID column exists.
- The re-audit called it with a frame lacking a persistent identifier; it returned IDs successfully.
- This is explicitly transient DataFrame-position provenance and must fail closed.

Blocking CV reproducibility defect:

- The pipeline still runs time-series cross-validation and emits fold/CV results.
- No OOF prediction artifact is produced; repository search found no OOF-prediction path in this runtime/tests.
- Renaming the official DEV summary to in-sample does not make the separately emitted CV prediction-based metrics reproducible from saved rows.

Verdict: FAIL.

# 11. Registry transaction audit

Positive findings:

- Registration requires completed prepare/train/project/evaluate manifests.
- Required run predictions, metrics, and selections are loaded and validated.
- Prediction hashes in metric rows are checked.
- Existing registries are loaded and extended; invalid old rows are not reclassified.
- Updates are staged in same-directory temporary files.
- On replacement failure, already replaced targets are restored.
- A transaction manifest is written after successful replacement.
- Synthetic rollback and simple-string idempotency tests pass.

Blocking idempotency defect:

- Duplicate-key equality compares `str()` values.
- A temporary probe used an existing `feature_budget=20.0` and identical incoming `feature_budget=20`.
- `append_registry_rows` raised `conflicting existing registry key`.
- Central CSV type inference commonly represents nullable numeric fields as floats, while new payloads use integers.
- Therefore a semantically identical second registration is not reliably idempotent.

Atomic rollback is materially improved, but the registration contract remains incomplete until semantic normalization makes identical rows reliably idempotent.

# 12. Reverse feature-flow table

| Model | CLIP candidate count | mRMR target scope | Encoded width saved | Final count | OOT used in selection |
| ----- | -------------------: | ----------------- | ------------------- | ----------: | --------------------- |
| Logistic Regression | 60 | Home Credit DEV only | YES | 20 | NO |
| CatBoost | 100 | Home Credit DEV only | YES | 40 | NO |

No pool-size sweep, threshold search, seed search, hyperparameter tuning, LLM stage, or OOT-based selection exists.

# 13. Dry-run resolution

- Roles: LendingClub v2 source; Home Credit external.
- LendingClub DEV/OOT configuration: DEV `[-1795,-1065)`; OOT `[-1065,-730]`.
- Anchor windows: `[-1795,-1612.5)`, `[-1612.5,-1430)`, `[-1430,-1247.5)`, `[-1247.5,-1065)`.
- Thresholds: PSI 0.10; missingness 0.05; minimum support 100; members 23.
- Seeds: 11, 22, 33, 44, 55.
- Budgets: LR 60→20; CatBoost 100→40.
- Stages: prepare, train, project, evaluate, register.
- Inputs: source/external feature manifests, raw sources, text embeddings, and identity evidence all resolve.
- Outputs: dedicated reverse-transfer feature/pair/training/anchor/projection/candidate/downstream paths.
- Advertised safeguards: no external refit, source OOT, pre-mRMR target, baseline, LLM, Home Credit retraining, UMAP, or completed-output overwrite.
- Exact `all --dry-run`: exit 0 in 5.90 seconds and created no stage manifest.
- `prepare --dry-run`: exit 0 and created no output.

Limitation: the printed dry-run does not display the source OOT bounds or the actual upstream-validation ordering defect.

# 14. Test results

| Command/check | Passed | Failed | Skipped | Runtime | Failure classification |
| ------------- | -----: | -----: | ------: | ------- | ---------------------- |
| `pytest tests/clip/test_dataset_roles.py tests/clip/test_reverse_transfer.py tests/clip/test_source_anchor.py tests/pipelines/test_reverse_transfer_orchestrator.py -q -p no:cacheprovider` | 44 | 0 | 0 | 28.17 s | None |
| `pytest tests/clip/test_clip_group_split.py tests/clip/test_clip_loss.py tests/clip/test_clip_negative_policy.py tests/clip/test_clip_pairing_repair.py tests/clip/test_clip_statistical_preprocessor.py -q -p no:cacheprovider` | 18 | 0 | 0 | 1.57 s | None |
| CLI help | 1 | 0 | 0 | not material | None |
| Exact `all --dry-run` | 1 | 0 | 0 | 5.90 s | None |
| Stage-specific `prepare --dry-run` | 1 | 0 | 0 | included in 14.5 s combined CLI check | None |
| Persistent-ID negative probe | 0 | 1 | 0 | <1 s | REAL REGRESSION: transient index IDs accepted |
| Invalid-upstream skip probe | 0 | 1 | 0 | 6.2 s combined probe | REAL REGRESSION: completed train skipped with invalid prepare |
| Realistic registry idempotency probe | 0 | 1 | 0 | 1.7 s | REAL REGRESSION: semantically identical numeric row rejected |

The pytest warning about unregistered `legacy_clip_v1` marker is a test-design warning, not a failure.

# 15. Approved manual command sequence

Only read-only dry-runs are approved. No scientific command is approved.

| Order | Stage | Exact command | Inputs valid | Output isolated | Safe |
| ----: | ----- | ------------- | ------------ | --------------- | ---- |
| 1 | all dry-run | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage all --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost --dry-run` | YES | YES | YES |
| 2 | prepare | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage prepare --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | NO: provenance/skip contract unresolved | YES | NO |
| 3 | train | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage train --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | NO | YES | NO |
| 4 | project | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage project --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | NO | YES | NO |
| 5 | evaluate | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage evaluate --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | NO | YES | NO |
| 6 | register | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage register --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | NO | central registries | NO |

# 16. Forward-comparator status

Classification: **ONLY CORRECTED REPRESENTATION/SCORE EVIDENCE EXISTS**.

- Valid corrected Home Credit-trained LendingClub embeddings/scores exist.
- Old forward downstream CLIP metrics remain invalid.
- No corrected forward downstream LR/CatBoost comparator exists.
- A valid reverse run would permit representation-level comparison in both directions.
- It would not permit a symmetric downstream-performance claim.
- A later narrow corrected forward downstream evaluation remains required for that claim.

This limitation is not itself a blocker for the reverse experiment; the execution blockers in sections 9–11 are.

# 17. Non-blocking limitations

- CPU-only five-seed training will be slow.
- Official reverse DEV performance is in-sample and must be interpreted only under that explicit scope.
- Corrected forward evidence is representation/score-only.
- `repo_stand.md` still describes Task 2 as not implemented; this is stale documentation but does not itself execute code.
- The unregistered pytest marker produces warnings.

# 18. Blocking findings

1. `_stable_row_ids` accepts DataFrame index fallback instead of requiring a persistent source identifier.
2. `--skip-existing`/completed-stage `--resume` returns before validating the upstream stage; reproduced with an invalid prepare manifest.
3. Current external/project/evaluation inputs are not universally recorded and revalidated for completed-stage skipping.
4. Source-anchor `raw_statistical_evidence_hash` hashes the eligibility metadata CSV, not the raw DEV statistical evidence used by the anchor/descriptors.
5. CV prediction-based results are emitted without saved out-of-fold predictions.
6. Registry idempotency is type-fragile; semantically identical numeric rows can be rejected as conflicts.

# 19. Final verdict

The repairs materially improve pairing, roles, alignment, direct overwrite protection, saved-file metrics, and transactional rollback, but fail-closed provenance, skip/upstream validation, persistent row identity, CV reproducibility, and registry idempotency remain demonstrably incomplete. The exact scientific command sequence is not safe.

NOT SAFE TO RUN
