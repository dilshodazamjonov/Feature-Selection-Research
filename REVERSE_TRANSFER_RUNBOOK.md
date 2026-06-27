# Corrected LendingClub v2 to Home Credit reverse-transfer runbook

Status: implementation only. No command below has been executed against the scientific data by this implementation task.

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

```powershell
.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage all --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost --dry-run
```

```powershell
.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage prepare --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost
```

```powershell
.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage train --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost
```

```powershell
.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage project --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost
```

```powershell
.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage evaluate --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost
```

Verify metrics and prediction provenance before registration:

```powershell
$root = 'results\corrected_lendingclub_to_homecredit_transfer'; Get-ChildItem -LiteralPath $root -Recurse -Filter '*predictions.csv' | ForEach-Object { $rows = Import-Csv -LiteralPath $_.FullName; [pscustomobject]@{ Path=$_.FullName; Rows=$rows.Count; MissingStableIds=($rows | Where-Object { -not $_.stable_row_id }).Count } }; Get-ChildItem -LiteralPath $root -Recurse -Filter 'oot_test_results.csv' | ForEach-Object { Import-Csv -LiteralPath $_.FullName }
```

```powershell
.\.venv\Scripts\python.exe scripts\run_corrected_lendingclub_to_homecredit_transfer.py --stage register --config-dir configs\corrected_lendingclub_to_homecredit --output-dir results\corrected_lendingclub_to_homecredit_transfer --seeds 11,22,33,44,55 --models lr,catboost
```

## Expected outputs

- `prepare`: feature reconciliation, deterministic feature split, exact-duplicate evidence, train/validation positive pairs, negative exclusions, LendingClub-fitted statistical preprocessor, and all source-anchor stability/member artifacts.
- `train`: five seed directories containing checkpoints, checkpoint manifests, training logs and representation metrics; per-seed approved source-anchor files.
- `project`: `homecredit_reverse_embeddings.parquet`, `homecredit_reverse_scores.csv`, `homecredit_reverse_feature_reconciliation.csv`, and `reverse_projection_manifest.json`.
- `evaluate`: fixed LR/CatBoost candidate pools; selected-feature and source-to-model-column lineage; DEV/OOT predictions with stable row IDs; run-local metrics and manifests.
- `register`: appended reverse-transfer rows in the four CSV registries plus updated access-guide and summary-manifest provenance.

## Stop conditions

Stop if the pairing policy differs, an old checkpoint is selected, Home Credit enters source fitting, LendingClub OOT enters descriptors or fitting, identity groups cross splits, fewer or more than 23 anchor members are selected, anchor thresholds/boundaries/hashes differ from configuration, an output would overwrite completed artifacts, a prediction lacks stable row IDs, or any baseline/LLM/Task 1/Task 3 command is about to run.

## Resume behaviour

Use `--skip-existing` to skip stages whose stage manifest says `complete`. Use `--resume` only for an understood incomplete output root. A completed stage is never overwritten by default. If artifact hashes or role metadata disagree, use a new output directory; do not reuse the incompatible files.

## Post-run handoff

Before training continues, inspect `stability_subwindow_config.json`, `feature_stability_evidence.csv`, `anchor_candidate_audit.csv`, `anchor_members.csv`, and `source_anchor_manifest.json`. Confirm 23 unique training-split members, fixed thresholds/boundaries, no repeated identity groups, and false target/OOT/external-use flags.

The read-only post-run audit must also inspect `seed_anchor_manifest.csv`, all five per-seed anchor manifests/hashes, every stage manifest, feature reconciliation, split manifest, duplicate evidence, negative-policy manifest, preprocessor state/hash, five checkpoint manifests, projection manifest, candidate pools, mRMR widths/lineage, DEV/OOT predictions, run-local metrics, registry payload, and all six central registry files.
