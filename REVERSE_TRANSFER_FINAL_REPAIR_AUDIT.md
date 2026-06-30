# 1. Audit snapshot

- Audit time: 2026-06-28 20:57:29 +05:00 (Asia/Tashkent).
- Branch: `main`.
- Commit: `cadb523797fba54447a3841df074bb8fc859477c`.
- Initial Git status contained the pre-existing Prompt 2 implementation changes and untracked audit/config/test files. This audit did not revert or modify them.
- Initial `git diff --stat`: 13 tracked files, 3,686 insertions, 83 deletions, plus the untracked focused registry test.
- Initial `git diff --check`: passed.
- Inspected:
  - `configs/corrected_lendingclub_to_homecredit/downstream.yaml`;
  - `src/credit_risk_fs/pipelines/common.py`;
  - `src/credit_risk_fs/models/training.py`;
  - `src/credit_risk_fs/pipelines/reverse_transfer.py`;
  - `src/credit_risk_fs/clip/reverse_transfer.py`;
  - `tests/clip/test_reverse_transfer.py`;
  - `tests/pipelines/test_reverse_transfer_orchestrator.py`;
  - `tests/pipelines/test_reverse_transfer_registry.py`;
  - all required audit, runbook, repository-status, and implementation-contract files.
- Executed: the required 64-test suite, direct stable-ID probes, direct persisted-prediction/metric-provenance probes, direct registry bundle probes, AST and JSON validation, CLI help, every stage dry-run, `register --dry-run`, and `all --dry-run`.
- All synthetic fixtures and pytest base directories were outside the repository.
- No real scientific stage or registry update ran.

# 2. Three-repair verdict table

| Repair | Status | Direct evidence | Adversarial result | Blocking issue |
| ------ | ------ | --------------- | ------------------ | -------------- |
| Authentic stable row IDs | PASS | Raw identity is built from `data/homecredit/raw/application_train.csv` before feature engineering/filtering and binds file, schema, complete ID set, target mapping, counts, and manifest hashes. Current modelling subsets are authenticated before fitting. | Renamed reset-index IDs were rejected as unauthenticated; genuine shuffled IDs passed with target mapping intact. Existing null, duplicate, replacement, missing-manifest, hash, and overlap tests passed. | None |
| OOF prediction and metric provenance | FAIL | Typed metadata and the canonical exporter fix the duplicate-`dataset` crash. Fold train/validation sets, hashes, coverage, probabilities, and saved-file AUC/KS/count are validated. | A synthetic valid persisted DEV OOF/OOT pair was accepted after setting `auc_drop=999.0`; adding `psi_value=999.0` was also accepted. `validate_metric_provenance` validates AUC/KS/count entries and PSI split names, but not the reported AUC-drop or PSI value against saved predictions. | Reported AUC-drop and score PSI remain forgeable/inconsistent with persisted predictions. |
| Registry integrity | FAIL | Explicit schemas, strict SHA-256 validation, canonical types/paths/JSON/enums, run/metric/selection references, new artifact file hash checks, lock, no-op, and rollback machinery exist. | A new-run bundle containing the same `artifact_id` at two different paths with different valid hashes passed `validate_registry_bundle`. The composite key `(artifact_id, relative_path)` permits the same new scientific artifact ID to identify multiple paths. | New-run artifact-ID identity/path uniqueness is not enforced. Required multi-point rollback coverage is also incomplete: the focused suite simulates replacement failure but not every required failure boundary. |

# 3. Stable row-ID audit

The configured identifier is `SK_ID_CURR`. `prepare_modeling_data` creates `SourceIdentityProvenance` immediately after loading raw `application_train.csv`, before feature construction, filtering, temporal splitting, sorting, reset-index operations, or preprocessing.

The source identity manifest includes:

- source artifact path and SHA-256;
- original ordered column list and its hash;
- order-independent complete source `SK_ID_CURR` hash;
- ID-to-target alignment hash;
- source row, unique-ID, and null counts;
- dataset, ID column, and `creation_stage=raw_input`;
- source identity manifest hash.

`validate_source_identity_subset` verifies the manifest, current source artifact, schema hash, authenticated ID set, counts, and target mapping. DEV and OOT are subsets of the authenticated source set and are checked for overlap. IDs are retained separately, attached before fold sorting, and excluded from `X_model`.

Direct probe:

```text
ROW_ID_FAKE_REJECTED current stable IDs are not authenticated source IDs
ROW_ID_GENUINE ['100017', '100001', '100099']
```

The genuine result followed a row shuffle. The fake frame was created by renaming a reset index to `SK_ID_CURR`. This repair passes.

# 4. OOF and metric audit

`PredictionMetadata` provides one typed source for dataset, split, run, model, method, configuration, data, policy, source-identity, and stable-ID provenance. `prediction_metadata_from_sources` rejects any duplicate key, including identical duplicates, before file writing. The former explicit-plus-`**prediction_common` collision no longer exists.

`run_kfold_training` records fold-specific:

- `training_ids` and `validation_ids`;
- `training_id_hash` and `validation_id_hash`;
- validation count;
- `model_fit_scope=fold_training_ids_only`.

The exporter verifies:

- training/validation disjointness;
- validation-fold disjointness;
- exact OOF-to-validation membership per fold;
- validation union equals eligible DEV IDs;
- absence of OOT IDs;
- unique/non-null authenticated IDs;
- target alignment;
- finite `[0,1]` probabilities;
- threshold-consistent classes;
- persisted file row/ID equivalence after reload.

Future output paths are:

- `results/dev_oof_predictions.csv`;
- `results/oot_predictions.csv`;
- `manifests/fold_manifest.json`;
- `manifests/prediction_manifest.json`;
- `manifests/metric_manifest.json`.

`prediction_metrics_from_saved_files` recomputes pooled DEV OOF and OOT AUC, KS, row count, hashes, and DEV-reference/OOT-comparison PSI from reloaded files. The output metric table records `auc_drop = DEV_OOF_AUC - OOT_AUC` and the documented PSI bin scopes.

Blocking direct probe:

```text
WRONG_AUC_DROP_ACCEPTED
WRONG_PSI_VALUE_ACCEPTED
```

`validate_metric_provenance` iterates only the manifest's AUC, KS, and row-count entries. It checks PSI split labels but does not recompute/compare a manifest PSI value. It also ignores the manifest's `auc_drop`. Therefore the strongest claimed prediction-to-metric linkage is incomplete, and this repair fails.

# 5. Registry audit

Declared versions:

- schema: `reverse_transfer_registry_v2`;
- canonicalization: `schema_aware_registry_v2`.

Explicit schemas exist for run index, artifact registry, reusable metrics, and selected-feature registry. Dedicated validators exist for summary and transaction manifests.

The implementation validates:

- non-empty, non-placeholder, nonzero, exact 64-hex SHA-256 values;
- required columns for new rows;
- booleans, integral values, finite required metrics, enums, deterministic JSON, timestamps, and repository-relative paths;
- no `..` traversal or absolute paths outside the repository;
- run-index, metrics, and selection primary keys;
- artifact composite key `(artifact_id, relative_path)`;
- run references from metrics, selections, and owned artifacts;
- metric/run dataset, model, method, configuration, data-manifest, and policy consistency;
- prediction hashes resolving to artifact hashes;
- selection hash resolving to an artifact;
- new artifact existence and recomputed hash;
- five checkpoint and anchor hashes in seed JSON.

Transaction outcomes are `NEW_TRANSACTION`, `IDEMPOTENT_NO_OP`, and `CONFLICT`. The writer uses an exclusive lock, retains original bytes, writes same-directory temporary files, flushes/fsyncs, verifies temporary bytes, replaces targets, writes the transaction manifest last, and restores/verifies replaced files after failure. Identical payloads return `IDEMPOTENT_NO_OP` without writing.

Blocking direct probe:

```text
DUPLICATE_ARTIFACT_ID_DIFFERENT_PATH_ACCEPTED
```

This used a bundle whose two new artifacts had the same `artifact_id`, different paths, different actual files, and matching valid per-file hashes. Because the schema primary key is the composite `(artifact_id, relative_path)`, both rows were unique under the implementation and passed. Historical content-derived IDs may legitimately appear at multiple paths, but the required current rule states that one new artifact ID cannot point to two paths. The validator does not distinguish that stricter new-run constraint.

The dedicated tests also simulate only a replacement failure. They do not directly simulate all required boundaries: temporary-write failure, temporary-validation failure, third replacement, summary update, and post-replacement/pre-manifest failure. The rollback mechanism is general, but the required independent adversarial evidence is incomplete.

# 6. Regression-boundary confirmation

Narrow inspection confirms:

- `identity_equivalence_v2` remains active;
- source remains LendingClub v2;
- external remains Home Credit;
- external preprocessing remains frozen/transform-only;
- Home Credit views remain aligned by `feature_id`;
- the four source DEV subwindows and 23-member anchor remain configured;
- seeds remain exactly 11, 22, 33, 44, and 55;
- LR remains 60 -> mRMR -> 20;
- CatBoost remains 100 -> mRMR -> 40;
- baseline, LLM, and UMAP execution remain disabled in the plan;
- deleted matrix entry points remain absent;
- no Task 1, Task 3, baseline, or old comparison route was executed.

# 7. Tests and dry-runs

| Command/check | Passed | Failed | Skipped | Runtime | Classification |
| ------------- | -----: | -----: | ------: | ------- | -------------- |
| `pytest tests/clip/test_reverse_transfer.py tests/pipelines/test_reverse_transfer_orchestrator.py tests/pipelines/test_reverse_transfer_registry.py -q -p no:cacheprovider` | 64 | 0 | 0 | 51.07 s pytest / 52.53 s wall | None; one non-failing marker warning |
| Stable-ID direct probes | 2 | 0 | 0 | <1 s | None |
| Positive saved OOF/OOT export and recomputation setup | 1 | 0 | 0 | <1 s | None |
| Wrong AUC-drop direct probe | 0 | 1 | 0 | <1 s | UNRESOLVED |
| Wrong PSI-value direct probe | 0 | 1 | 0 | <1 s | UNRESOLVED |
| Duplicate new artifact ID/different path direct probe | 0 | 1 | 0 | <1 s | UNRESOLVED |
| AST parse | 4 | 0 | 0 | <1 s | None |
| Implementation JSON validation | 3 | 0 | 0 | <1 s | None |
| CLI help and six stage/all dry-runs | 7 | 0 | 0 | 35.68 s combined | None |
| Initial `rg` with a PowerShell-incompatible wildcard | 0 | 1 | 0 | <1 s | TEST DESIGN PROBLEM; immediately rerun against the directory successfully |

Dry-run isolation:

- real registry files changed: 0;
- scientific files before: 0;
- scientific files after: 0.

Passing unit tests do not override the unresolved direct probes.

# 8. Approved manual commands

Only dry-run/help commands are approved.

| Order | Stage | Exact command | Verified | Safe |
| ----: | ----- | ------------- | -------- | ---- |
| 1 | Help | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --help` | YES | YES |
| 2 | Preflight | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage all --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost --dry-run` | YES | YES |
| 3 | Prepare | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage prepare --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | syntax/dry-run verified | NO |
| 4 | Train | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage train --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | syntax/dry-run verified | NO |
| 5 | Project | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage project --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | syntax/dry-run verified | NO |
| 6 | Evaluate | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage evaluate --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | syntax/dry-run verified | NO |
| 7 | Register dry-run | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage register --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost --dry-run` | YES | YES |
| 8 | Register commit | `.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage register --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost` | syntax/dry-run verified | NO |

Real execution is not approved because two repaired areas fail direct verification.

# 9. Non-blocking limitation

Corrected forward representation/score evidence exists.
Corrected forward downstream evidence does not exist.
Bidirectional downstream-performance claims remain unsupported.

# 10. Blocking findings

1. `validate_metric_provenance` accepts a materially incorrect `auc_drop` instead of recomputing `DEV_OOF_AUC - OOT_AUC`.
2. `validate_metric_provenance` accepts an incorrect PSI value; it validates only the reference/comparison split labels, not the saved-prediction-derived PSI result.
3. `validate_registry_bundle` accepts the same new-run `artifact_id` at multiple different paths because artifact uniqueness is only the composite `(artifact_id, relative_path)`.
4. Independent rollback tests do not cover every required transaction failure boundary.

# 11. Final verdict

NOT SAFE TO RUN
