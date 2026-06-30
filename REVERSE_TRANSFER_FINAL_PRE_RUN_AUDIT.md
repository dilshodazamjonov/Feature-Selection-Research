# 1. Audit snapshot

- Audit time: 2026-06-28 14:41:49 +05:00 (Asia/Tashkent).
- Branch: `main`.
- Commit: `cadb523797fba54447a3841df074bb8fc859477c`.
- Initial Git status:
  - modified: `REVERSE_TRANSFER_RUNBOOK.md`;
  - modified: `configs/corrected_lendingclub_to_homecredit/contrastive_data.yaml`;
  - modified: `configs/corrected_lendingclub_to_homecredit/downstream.yaml`;
  - modified: `src/credit_risk_fs/clip/reverse_transfer.py`;
  - modified: `src/credit_risk_fs/clip/source_anchor.py`;
  - modified: `src/credit_risk_fs/clip/statistical_preprocessor.py`;
  - modified: `src/credit_risk_fs/clip/trainer.py`;
  - modified: `src/credit_risk_fs/models/training.py`;
  - modified: `src/credit_risk_fs/pipelines/common.py`;
  - modified: `src/credit_risk_fs/pipelines/reverse_transfer.py`;
  - modified: `tests/clip/test_reverse_transfer.py`;
  - modified: `tests/clip/test_source_anchor.py`;
  - modified: `tests/pipelines/test_reverse_transfer_orchestrator.py`;
  - untracked pre-existing audit/config material: `REVERSE_TRANSFER_PRE_RUN_AUDIT.md`, `REVERSE_TRANSFER_PRE_RUN_REAUDIT.md`, and `configs/corrected_lendingclub_to_homecredit/identity_evidence.json`.
- Initial `git diff --stat`: 13 tracked files, 2,118 insertions, 59 deletions.
- Initial `git diff --check`: passed.
- Audited changed files: every source, configuration, test, runbook, and implementation-handoff file involved in the five repairs, including all files listed above and the five JSON/Markdown contracts under `results/corrected_lendingclub_to_homecredit_transfer/implementation/`.
- Required historical documents read: `REVERSE_TRANSFER_PRE_RUN_AUDIT.md`, `REVERSE_TRANSFER_PRE_RUN_REAUDIT.md`, `REVERSE_TRANSFER_RUNBOOK.md`, and `repo_stand.md`.
- Commands executed: repository snapshot commands; source/diff searches; AST parsing; targeted pytest suite; artifact-independent corrected-CLIP suite; the three former regression probes; direct stable-ID, OOF-export, canonical-registry, malformed-hash, and duplicate-registry probes; CLI help; all five stage dry-runs; and `all --dry-run`.
- Test temporary directories and captured dry-run output were placed under the operating-system temporary directory. Pytest cache creation was disabled and `PYTHONDONTWRITEBYTECODE=1` was used for test and dry-run commands.
- No real scientific stage was executed.

# 2. Five-blocker table

| Blocker | Claimed repair | Direct evidence | Adversarial result | Status |
| ------- | -------------- | --------------- | ------------------ | ------ |
| Stable row IDs | `SK_ID_CURR` is required and index-derived IDs fail | Configuration and runtime use `SK_ID_CURR`; it is validated before filtering and dropped from model features. `_stable_row_ids` rejects only a small set of generated column names. | The former `column="index"` probe passes, but an actual `reset_index(names="SK_ID_CURR")` positional ID is accepted as unique persistent identity. | NOT RESOLVED |
| Skip/resume | One central validator checks current inputs and all upstream stages | `_validate_reuse_chain` is used by completed skip/resume and partial train resume. `_validate_stage_manifest` checks identity, seeds/models, artifacts, current data provenance, file inputs, dataset fingerprints, and registered outputs. | The previous invalid-upstream skip probe passes; current-input and corruption tests pass. | RESOLVED |
| Raw DEV evidence | Canonical SHA-256 binds actual raw LendingClub DEV values | `canonical_raw_dev_evidence` hashes schema plus canonical CSV bytes for the selected DEV rows/columns, with target exclusion, range/role checks, missing encoding, float formatting, and deterministic column order. The hash propagates through preprocessor, anchor, projection, evaluation, and registration metadata. | Value, row, column, OOT, target, and Home Credit synthetic mutations are rejected or change the hash. | RESOLVED |
| OOF prediction/metrics | `dev_oof_predictions.csv` is persisted and drives DEV metrics | Fold collection uses validation indices and records stable ID, fold, target, probability, and class. The downstream artifact/metric plumbing expects `DEV_OOF`. | The actual export statement supplies `dataset` both explicitly and through `prediction_common`; Python raises `TypeError: DataFrame.assign() got multiple values for keyword argument 'dataset'` before the OOF file is written. No current test exercises this branch. | NOT RESOLVED |
| Registry normalization/atomicity | Canonical comparison and atomic no-op/rollback | Numeric, boolean, path, whitespace, JSON, and enum equivalence work; atomic replacement and rollback tests pass; identical byte payloads return `idempotent_noop`. | `prediction_file_hash="not-a-sha256"` is accepted; two pre-existing rows with the same registry key are accepted and retained. There is no complete schema, uniqueness, or referential-integrity validation before commit. | NOT RESOLVED |

# 3. Stable row-ID audit

The downstream configuration explicitly sets `stable_row_id_column: SK_ID_CURR`, and `resolve_plan` refuses any other configured value. Home Credit feature assembly uses `SK_ID_CURR` as its established application identifier.

`prepare_modeling_data` validates the identifier on the assembled modelling input before missing-time filtering, temporal filtering, splitting, sorting, preprocessing, or fitting. It validates DEV and OOT again after splitting. Nulls and duplicates fail. DEV and OOT prediction hashes are checked for overlap. `SK_ID_CURR` is removed through the default ID-drop columns and never enters `X_model`. The stable-ID vector is attached before temporal sorting, so ordinary row shuffling preserves the ID-to-target mapping.

However, provenance is inferred only from the configured column name. `_stable_row_ids` rejects names such as `index`, `level_0`, and `row_number`, but cannot reject a positional/reset-index series renamed to `SK_ID_CURR`. The direct probe:

```python
transient = pd.DataFrame({"target": [0, 1, 0]}).reset_index(names="SK_ID_CURR")
_stable_row_ids(
    transient,
    dataset="homecredit",
    stable_row_id_column="SK_ID_CURR",
)
```

returned three unique hashes instead of failing. The existing test only calls `reset_index()` and then explicitly selects the column named `index`; it does not test a reset-index identifier presented under the configured name. This fails the required transient/reset-index adversarial contract.

# 4. Skip/resume audit

`execute_plan` calls `_validate_reuse_chain` for:

- completed stages under `--skip-existing` or `--resume`;
- partial train-stage resume, excluding the incomplete current stage;
- all upstream dependencies before any new downstream handler runs.

The chain is fixed as `prepare -> train -> project -> evaluate -> register`. Every reused manifest must be present and complete. `_validate_stage_manifest` verifies:

- configuration, source/external roles, and pairing-policy identity;
- exact requested seeds and models;
- required artifact existence and SHA-256;
- current `data_manifest.json` hash;
- identity-evidence, source-universe, split, raw-DEV-evidence, and preprocessor hashes;
- declared current file hashes;
- declared dataset fingerprints;
- registered-output hashes.

The configuration hash includes all four YAML files and the identity-evidence JSON. Prepare records current hashes for the feature evidence, raw LendingClub source, text embeddings, and identity evidence. Project records external manifest/text hashes and a Home Credit data fingerprint. Evaluation records the Home Credit data fingerprint.

Completed stages are refused without an explicit reuse flag. Directory existence without a manifest and unmanifested partial artifacts fail. Train resume reuses a seed only when checkpoint, checkpoint manifest, anchor vector, and anchor manifest all exist and validate; a corrupt completed checkpoint fails. Required artifacts are hashed and the completion manifest is written only after handler completion and artifact checks.

The former invalid-upstream skip probe passed. The current suite also covers changed inputs, missing/corrupt artifacts, completed-stage refusal, partial-seed reuse, and invalid checkpoint rejection. This blocker is resolved.

# 5. Raw DEV provenance audit

`canonical_raw_dev_evidence` requires dataset `lendingclub_v2`, rejects target inclusion in the selected feature list, requires all declared columns, rejects empty evidence, checks every time value is inside `[-1795,-1065)`, and optionally validates stable source-row IDs when available.

Its SHA-256 input consists of:

1. canonical JSON schema metadata containing dataset, split, DEV bounds, row-order policy, column-order policy, null encoding, float format, encoding, target-exclusion flag, time column, optional stable-ID declaration, sorted feature list, source dtypes, and row/column counts;
2. UTF-8 CSV bytes for the exact selected DEV rows, with deterministic column order, `<NA>` missing representation, `%.17g` float formatting, header, and source-file row order.

The target column is not part of the hashed frame. OOT rows fail the range check. A Home Credit role fails. Changing a value, row scope, feature set, dtype metadata, or row/column ordering changes the digest.

The hash is written to `pairing/raw_dev_statistical_evidence_manifest.json` and `pairing/data_manifest.json`, included in `StatisticalPreprocessor` state/hash, added to stability evidence, candidate audit, and member rows, included in the source-anchor manifest and each per-seed anchor manifest, checked during projection, propagated into the reverse-projection manifest, copied into the evaluation stage provenance, and included in registration transaction metadata. The anchor-member CSV contains the raw hash before its own hash is calculated, binding membership to raw evidence.

Synthetic mutations for one value, row removal, feature removal, OOT insertion, target inclusion, and Home Credit role all passed their expected rejection/change checks. This blocker is resolved.

# 6. OOF metric audit

`run_kfold_training` attaches stable IDs before sorting and excludes them from `X_model`. For each fold it writes probabilities/classes only at `va_idx`, records the fold number, and exports rows selected by `oof_mask`. These mechanics make training and validation indices disjoint under the splitter. The intended OOF schema is completed in `run_experiment` with dataset, split, run, method, model, source/external roles, configuration/data hashes, policy, and fit scope. `prediction_metrics_from_saved_files` validates unique IDs, fold presence, probability range, DEV/OOT ID disjointness, expected row/target coverage, and recomputes AUC, KS, count, file hash, and DEV-to-OOT score PSI.

The implementation is not runnable:

```python
prediction_common = {
    "dataset": config.dataset_name,
    ...
}

oof_predictions = oof_predictions.assign(
    dataset=config.dataset_name,
    split="DEV_OOF",
    **prediction_common,
)
```

Python rejects the duplicate `dataset` keyword before `dev_oof_predictions.csv` can be saved. A direct runtime-equivalent probe produced:

```text
TypeError pandas.core.frame.DataFrame.assign() got multiple values for keyword argument 'dataset'
```

The 49-test suite passes because it tests `prediction_metrics_from_saved_files` with manually created OOF CSVs but never invokes the corrected reverse-transfer branch of `run_experiment`. Consequently no real future LR or CatBoost reverse-transfer evaluation can currently persist the required OOF file or reach saved-file metric recomputation.

This is a `REAL REGRESSION` and a blocking failure.

# 7. Registry transaction audit

`canonical_registry_value` successfully normalizes:

- `None`, `NaN`, and empty strings to `<NULL>`;
- supported boolean schema columns;
- numeric values for numeric types/tokens;
- Windows/POSIX separators and repository-relative absolute paths;
- JSON objects with sorted keys and compact formatting;
- JSON arrays while retaining order;
- selected enum/status fields to lowercase;
- whitespace;
- timestamps;
- hash casing.

`append_registry_rows` uses this function for keys and shared-value conflict comparison. The tested numeric, boolean, path, and JSON formatting variants were equivalent. Same-key differing shared scientific values raise.

`atomic_registry_transaction` writes proposed bytes to same-directory temporary files, replaces targets, writes the transaction manifest last, and rolls replaced targets back from saved original bytes on an exception. The rollback test passes. If every proposed byte string already equals its target, it returns the prior manifest with `idempotent_noop=True` without writing target or manifest bytes.

Required validation is incomplete:

- Hash validation occurs only when the supplied string already has length 64. A non-empty malformed hash of another length, such as `not-a-sha256`, is accepted.
- Existing registry uniqueness is not checked. A CSV containing two rows with the same `run_id` was loaded, retained, and extended without error.
- Existing registry schemas are not validated beyond incoming equivalence-column presence.
- No explicit cross-registry referential-integrity validation is performed before replacement.
- Proposed rows are compared canonically, but complete proposed registry values are not normalized into a canonical stored representation.
- Unrelated-row byte/content preservation is an incidental result of pandas round-tripping rather than an explicit validated invariant when a registry changes.

The required malformed-provenance, uniqueness, schema, and referential-integrity precommit guarantees are therefore absent. This blocker is not resolved even though equivalence and rollback mechanics pass.

# 8. Regression-boundary audit

The already-passing scientific boundaries remain intact:

- active policy remains `identity_equivalence_v2`;
- explicit alias and documented identity evidence are loaded from the hashed identity JSON and propagated to split, stability, and negative masking;
- source-table, similarity, correlation, and broad-family relations remain diagnostic-only;
- source is fixed to LendingClub v2 and external is fixed to Home Credit;
- Home Credit semantic/statistical views align one-to-one by `feature_id`, with duplicate/missing/name/dataset conflicts rejected;
- LendingClub representation fitting uses DEV only;
- Home Credit statistical processing remains transform-only with the frozen LendingClub preprocessor;
- the source anchor retains four fixed DEV subwindows, fixed thresholds, and exactly 23 members;
- seeds remain exactly `11,22,33,44,55`;
- LR remains `60 -> mRMR -> 20`;
- CatBoost remains `100 -> mRMR -> 40`;
- incompatible/old checkpoint metadata remains rejected;
- no baseline execution, LLM invocation, UMAP generation, deleted matrix, or old final-comparison route is reachable from the reverse-transfer entry point.

The artifact-independent corrected-CLIP regression suite passed 18/18.

# 9. Test results

| Command/check | Passed | Failed | Skipped | Runtime | Failure classification |
| ------------- | -----: | -----: | ------: | ------- | ---------------------- |
| `pytest tests/clip/test_reverse_transfer.py tests/clip/test_source_anchor.py tests/pipelines/test_reverse_transfer_orchestrator.py -q -p no:cacheprovider` | 49 | 0 | 0 | 43.85 s pytest / 45.00 s wall | None; one non-failing unregistered-marker warning |
| `pytest tests/clip/test_clip_group_split.py tests/clip/test_clip_loss.py tests/clip/test_clip_negative_policy.py tests/clip/test_clip_pairing_repair.py tests/clip/test_clip_statistical_preprocessor.py -q -p no:cacheprovider` | 18 | 0 | 0 | 1.61 s pytest / 2.36 s wall | None; one non-failing unregistered-marker warning |
| Three previous adversarial pytest probes | 3 | 0 | 0 | 12.07 s pytest / 13.00 s wall | None |
| AST parse of five central changed Python modules | 5 | 0 | 0 | <1 s | None |
| OOF export duplicate-keyword probe | 0 | 1 | 0 | <1 s | REAL REGRESSION |
| Renamed reset-index-as-`SK_ID_CURR` probe | 0 | 1 | 0 | <1 s | REAL REGRESSION |
| Canonical numeric/boolean/path/JSON equivalence probe | 4 | 0 | 0 | <1 s | None |
| Malformed hash validation probe | 0 | 1 | 0 | <1 s | UNRESOLVED |
| Existing duplicate registry-key probe | 0 | 1 | 0 | <1 s | UNRESOLVED |
| First path-normalization inline probe | 0 | 1 | 0 | <1 s | TEST DESIGN PROBLEM: malformed raw-string literal; immediately corrected and rerun |
| CLI help plus `prepare/train/project/evaluate/register/all --dry-run` | 7 | 0 | 0 | 40.37 s | None for exit/artifact isolation |

The passing tests do not override the directly reproduced runtime and contract failures. Under the required classification rule, the two `REAL REGRESSION` findings and unresolved registry guarantees require `NOT SAFE TO RUN`.

# 10. Dry-run and command table

Dry-run scientific file count was zero before and after all commands. Every dry-run exited 0. The printed plan reports source/external roles, pairing policy, five seeds, budgets, output paths, `SK_ID_CURR`, raw-DEV provenance, OOF artifact/scope, reuse-chain policy, registry equivalence/atomicity text, source-anchor DEV bounds/subwindows, and anchor thresholds.

The dry-run does not explicitly report the Home Credit DEV/OOT windows or the LendingClub OOT interval, despite the audit requirement to report DEV/OOT windows. This is a dry-run observability defect.

| Order | Stage | Exact command | Inputs valid | Output isolated | Safe |
| ----: | ----- | ------------- | ------------ | --------------- | ---- |
| 1 | help | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --help` | YES | YES | YES |
| 2 | all dry-run | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage all --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost --dry-run` | YES | YES | YES, as non-scientific resolution only |
| 3 | prepare | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage prepare --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | YES | dedicated root | NO |
| 4 | train | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage train --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | YES | dedicated root | NO |
| 5 | project | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage project --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | YES | dedicated root | NO |
| 6 | evaluate | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage evaluate --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | YES | dedicated root | NO |
| 7 | register | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage register --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | YES | NO: central registries | NO |

All stage-specific dry-run equivalents of rows 3-7 passed and created no scientific output. Real commands are not approved because the OOF export path cannot complete and the stable-ID/registry contracts remain incomplete.

# 11. Forward-comparator limitation

Corrected forward representation/score evidence exists.
Corrected forward downstream evidence does not exist.
Bidirectional downstream-performance claims remain unsupported.

This is non-blocking for a future corrected reverse-transfer execution after the implementation blockers are repaired. It remains blocking for claims of bidirectional downstream predictive competitiveness.

# 12. Blocking findings

1. `_stable_row_ids` accepts a positional reset-index identifier when it is renamed to the configured `SK_ID_CURR` column, so the required transient-ID adversarial contract is not enforced.
2. The corrected reverse-transfer OOF export supplies `dataset` twice to `DataFrame.assign`, causing a deterministic `TypeError` before either model can persist `dev_oof_predictions.csv` or compute reported DEV metrics from it.
3. Registry precommit validation accepts malformed non-64-character hash values and pre-existing duplicate registry keys; complete schema, uniqueness, and referential-integrity validation is absent.
4. Dry-run output omits explicit Home Credit DEV/OOT and LendingClub OOT windows required by the final preflight observability contract.

# 13. Final verdict

NOT SAFE TO RUN
