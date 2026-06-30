# Corrected LendingClub v2 to Home Credit reverse-transfer runbook

Status: final repaired implementation only; ready for an independent final pre-run audit. No scientific command below has been executed.

## Preconditions

1. Use branch `main`. The implementation was based on commit `f3a5b3ed194e99b63657151dac8eed83ff92fd6d`; record the actual pre-run commit with `git rev-parse HEAD`.
2. Run `git status --short` and account for every entry. Do not overwrite an existing completed output root.
3. Activate the repository environment: `.\.venv\Scripts\Activate.ps1`.
4. Check free disk space with `Get-PSDrive -Name (Split-Path -Qualifier (Get-Location))`.
5. Verify these inputs:
   - `data/lendingclub_v2/processed/application_train.csv`
   - `results/lendingclub_v2/analysis/clip_readiness/dev_only_clip_training_evidence.csv`
   - `results/clip/text_baseline/lendingclub_v2_text_embeddings.parquet`
   - `results/clip/text_baseline/homecredit_text_embeddings.parquet`
   - `results/corrected_homecredit_clip/feature_universe/feature_universe_reconciliation.csv`
   - the four YAML files under `configs/corrected_lendingclub_to_homecredit/`
6. Confirm `pairing_policy_version: identity_equivalence_v2`, `training_dataset: lendingclub_v2`, and `external_dataset: homecredit`.
7. Confirm the fixed source-anchor rule in `contrastive_data.yaml`: LendingClub DEV `[-1795,-1065)`, four equal-duration windows with boundaries `[-1795,-1612.5,-1430,-1247.5,-1065]`, PSI threshold `0.10`, missing-rate-difference threshold `0.05`, minimum non-missing support `100` per window, and exactly `23` members.
8. Inspect `configs/corrected_lendingclub_to_homecredit/identity_evidence.json`. Only explicit verified aliases and documented identity-preserving transforms belong there. Each non-empty relation must carry matching stable feature IDs; source table, similar name, semantic group, text similarity, correlation, and equal descriptors are never identity evidence.
9. Confirm `stable_row_id_column: SK_ID_CURR` in `downstream.yaml`. Identity is authenticated at the raw `data/homecredit/raw/application_train.csv` boundary before feature engineering or filtering. `data/source_identity_manifest.json` binds the source file hash, original column-list hash, complete order-independent source-ID values hash, ID-to-target alignment hash, counts, and `creation_stage=raw_input`. DEV/OOT IDs must be authenticated subsets. A column name, uniqueness, or sequential/reset-index values alone are insufficient, and `SK_ID_CURR` never enters the model feature matrix.

## LendingClub source-anchor rule

The anchor uses 23 members to match the existing corrected Home Credit stable-core anchor size, giving the two transfer directions a symmetric representation-level comparison. The member count is fixed before execution and is not chosen using downstream performance.

Only the LendingClub DEV interval `[-1795,-1065)` is used. It is divided into four contiguous equal-duration windows:

1. `[-1795,-1612.5)`
2. `[-1612.5,-1430)`
3. `[-1430,-1247.5)`
4. `[-1247.5,-1065)`

For each training-split feature, the implementation records non-missing counts, missing rates, maximum missing-rate difference, and PSI for each adjacent window pair. Numerical quantile bins and categorical buckets are fitted once on window 1 and frozen for windows 2–4. Missingness is an explicit bucket; unseen or rare categorical values use `OTHER`.

The rule is target-free, OOT-free, LendingClub-only, and independent of Home Credit and downstream metrics. Features must have at least 100 observed values in every window and satisfy both `max_adjacent_window_psi <= 0.10` and `max_missing_rate_difference <= 0.05`. Qualifying features are ranked by PSI, then missing-rate difference, then feature ID. Identity-equivalent members cannot coexist.

If fewer than 23 qualifying training-split features remain, `prepare` fails closed. Thresholds, membership count, feature split, or identity safeguards must not be relaxed to continue.

## Exact manual commands

Run from the repository root in PowerShell. Complete the separate read-only pre-run audit before step 2.

### 1. Preflight/dry-run

```powershell
.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage all --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost --dry-run
```

### 2. Prepare

```powershell
.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage prepare --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost
```

### 3. Five-seed LendingClub CLIP training

```powershell
.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage train --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost
```

### 4. Frozen Home Credit projection

```powershell
.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage project --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost
```

### 5. Downstream LR and CatBoost evaluation

```powershell
.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage evaluate --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost
```

Verify metrics and prediction provenance before registration:

```powershell
$root = 'results\corrected_lendingclub_to_homecredit_transfer'; Get-ChildItem -LiteralPath $root -Recurse -Filter '*predictions.csv' | ForEach-Object { $rows = Import-Csv -LiteralPath $_.FullName; [pscustomobject]@{ Path=$_.FullName; Rows=$rows.Count; MissingStableIds=($rows | Where-Object { -not $_.stable_row_id }).Count } }; Get-ChildItem -LiteralPath $root -Recurse -Filter 'oot_test_results.csv' | ForEach-Object { Import-Csv -LiteralPath $_.FullName }
```

### 6. Post-evaluation registry dry-run

This dry-run must succeed before registry commit. It validates registry schemas, hashes, cross-registry references, metric provenance, artifact identity, and the proposed transaction outcome without writing registry files or a transaction manifest. `CONFLICT` or any validation failure stops execution. Do not run the real commit command after a failed dry-run.

```powershell
.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage register --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost --dry-run
```

### 7. Real registry commit

```powershell
.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage register --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost
```

## Expected outputs

- `prepare`: feature reconciliation, deterministic feature split, exact-duplicate evidence, train/validation positive pairs, negative exclusions, `pairing/raw_dev_statistical_evidence_manifest.json`, LendingClub-fitted statistical preprocessor, and all source-anchor stability/member artifacts.
- `train`: five seed directories containing checkpoints, checkpoint manifests, training logs and representation metrics; per-seed approved source-anchor files.
- `project`: `homecredit_reverse_embeddings.parquet`, `homecredit_reverse_scores.csv`, `homecredit_reverse_feature_reconciliation.csv`, and `reverse_projection_manifest.json`.
- `evaluate`: fixed LR/CatBoost candidate pools; selected-feature lineage; per-model `data/source_identity_manifest.json`; authenticated DEV/OOT prediction identity; `results/dev_oof_predictions.csv`; `results/oot_predictions.csv`; full-precision `results/prediction_metrics.csv`; reproducible `results/psi_details.csv`; and `manifests/fold_manifest.json`, `prediction_manifest.json`, and `metric_manifest.json`.
- `register`: appended reverse-transfer rows in the four CSV registries plus updated access-guide and summary-manifest provenance.

## Repaired safety contract

- Every stage manifest records exact source/external roles, configuration hash, policy, requested seeds/models, required artifact paths, and file hashes. The completion manifest is written last.
- Home Credit semantic and transformed statistical views are joined one-to-one by `feature_id`. Duplicate, missing, cross-dataset, or name-conflicting identities fail before projection. Inspect `reverse_projection/alignment_manifest.json` and require `semantic_feature_ids_equal_statistical_feature_ids=true`.
- The source anchor is bound to the canonical target-free raw LendingClub DEV values actually consumed. `raw_dev_statistical_evidence_hash` changes with row scope, feature columns, values, dtypes, or ordering and is validated through the preprocessor, stability/candidate/member artifacts, per-seed anchors, projection, evaluation, and registration.
- One central reuse validator serves both `--skip-existing` and `--resume`. It validates the current stage, every transitive upstream completion manifest/artifact hash, current declared file and dataset inputs, roles, configuration, policy, seeds, identity, universe, split, raw DEV evidence, and preprocessor provenance.
- Stop before fitting if the raw Home Credit source lacks `SK_ID_CURR`, the source-identity manifest/file/schema/ID/target hash is absent or changed, a current ID is outside the authenticated source set, IDs or targets are replaced, duplicates/nulls occur, or DEV and OOT overlap. Filtering and reordering are allowed only when authenticated ID-to-target pairs remain unchanged.
- Reverse DEV performance is pooled `dev_oof_cross_validated`, distinct from any `mean_fold_auc`. Each fold manifest hashes disjoint training/validation ID sets, and OOF rows must exactly equal that fold's validation IDs with complete non-overlapping DEV coverage. The authoritative inputs are the saved full-precision `dev_oof_predictions.csv` and `oot_predictions.csv`. Generation and validation share the same saved-file computation path. AUC drop is strictly `DEV_OOF_AUC - OOT_AUC`, with no alternate sign or fold-mean substitution; the absolute validation tolerance is `1e-12` with zero relative tolerance.
- Score PSI uses saved DEV OOF probabilities as reference and saved OOT probabilities as comparison. It requests 10 quantile bins fitted on DEV OOF only, removes duplicate candidate edges deterministically, fixes the probability boundaries at `[0,1]`, and applies those frozen edges to OOT without refitting. Missing, non-finite, or out-of-range probabilities are rejected. Each share receives epsilon `1e-6`; the natural-log formula is `SUM((smoothed_comparison_share - smoothed_reference_share) * LN(smoothed_comparison_share / smoothed_reference_share))`. The manifest persists the requested/effective bin counts, exact edges, policies, scopes, hashes, counts, implementation versions, and tolerance. `psi_details.csv` persists every bin's bounds, counts, raw/smoothed shares, and contribution; its hash is bound to the manifest and its contributions must sum to the full-precision PSI within `1e-12`.
- Before registration, `validate_metric_provenance` reloads and authenticates both prediction artifacts, recomputes pooled AUCs, AUC drop, DEV-fitted bins, per-bin PSI, and total PSI, and verifies the metric CSV, PSI-detail artifact, and manifest claims. Any value, sign, scope, path, hash, count, bin edge, smoothing rule, contribution, run, model, configuration, data-manifest, source-identity, implementation-version, or required-artifact mismatch is a hard stop.
- Registration uses schema `reverse_transfer_registry_v2` and canonicalization `schema_aware_registry_v2`. New rows require valid non-placeholder lowercase SHA-256 values, typed booleans/integers/finite metrics, repository-relative paths, valid enums, and deterministic JSON. Artifact identity is logical `artifact_id`, not an unrestricted `(artifact_id,relative_path)` pair: `artifact_id -> exactly one canonical relative_path` and `canonical relative_path -> exactly one artifact_id`. Windows separators, POSIX separators, leading `./`, and redundant `.` components normalize before comparison. Paths must remain non-empty and inside the repository. Artifact ID and path groups must also have one hash, type, owning run/reusable identity, dataset, model, method, configuration hash, data-manifest hash, pairing-policy version, and scientific stage. Existing or proposed conflicts fail before writing with diagnostics containing IDs, paths, hashes, owners, types, origins, and the violated invariant.
- Registry outcomes are `NEW_TRANSACTION`, `IDEMPOTENT_NO_OP`, or `CONFLICT`. Dry-run performs the same artifact-identity validation but reports no affected active files and writes no transaction manifest. A single-writer lock guards staged writes, flush/fsync, staged schema/hash validation, ordered replacement, post-write schema/hash validation, and summary replacement. Original existence, raw bytes, and SHA-256 are captured for every active target. Any pre-commit failure restores every original byte exactly, removes originally absent targets and false success manifests, verifies restoration hashes, removes temporary files, releases the lock, and permits a clean retry. The exact commit boundary is atomic replacement of the fully written and validated transaction manifest, which occurs last. Failures after that replacement are post-commit warnings and do not roll back a committed transaction.

## Stop conditions

Stop if the pairing policy differs, an old checkpoint is selected, Home Credit enters source fitting, LendingClub OOT enters descriptors or fitting, identity groups cross splits, fewer or more than 23 anchor members are selected, anchor thresholds/boundaries/hashes differ from configuration, an output would overwrite completed artifacts, a prediction lacks stable row IDs, a derived metric or PSI detail cannot be reproduced exactly within its declared `1e-12` tolerance from the saved prediction artifacts, an artifact ID/path/hash/type/owner/provenance invariant conflicts, byte-identical registry rollback cannot be verified, or any baseline/LLM/Task 1/Task 3 command is about to run.

## Resume behaviour

Use `--skip-existing` only after inspecting the output root; the CLI independently validates completeness and hashes. Use `--resume` only for an understood stage whose manifest is `in_progress`. A validated complete stage is skipped, never rerun. Completed valid seed checkpoints/anchors are preserved. Any configuration, role, policy, split, identity, preprocessor, upstream, or artifact-hash mismatch fails closed. Use a new output directory for incompatible state.

## Forward-comparator limitation

Corrected forward representation/score evidence exists.
Corrected forward downstream evidence does not exist.
Bidirectional downstream-performance claims remain unsupported.

## Post-run handoff

Before training continues, inspect `stability_subwindow_config.json`, `feature_stability_evidence.csv`, `anchor_candidate_audit.csv`, `anchor_members.csv`, and `source_anchor_manifest.json`. Confirm 23 unique training-split members, fixed thresholds/boundaries, no repeated identity groups, and false target/OOT/external-use flags.

The read-only post-run audit must also inspect `seed_anchor_manifest.csv`, all five per-seed anchor manifests/hashes, every stage manifest, feature reconciliation, split manifest, duplicate evidence, negative-policy manifest, preprocessor state/hash, five checkpoint manifests, projection manifest, candidate pools, mRMR widths/lineage, DEV/OOT predictions, run-local metrics, registry payload, and all six central registry files.
