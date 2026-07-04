# Research Results Access Guide

## Read First
Later agents must read:
1. `results/research_summary/run_index.csv`
2. `results/research_summary/results_access_guide.md`
3. `results/research_summary/summary_manifest.json`
before scanning broader result directories.

## Status Meaning
- `reusable_existing`: independence from the faulty CLIP pairing policy was verified.
- `invalid_pairing_policy`: historical status retained in compact cleanup tombstones; these rows are not present in the active registry.
- `unknown_requires_review`: dependency or scientific eligibility needs manual review.

Historical names alone do not indicate corrected status. Canonical corrected Home Credit and reverse-transfer evidence uses `identity_equivalence_v2` and is registered as `newly_executed`.

## Valid Existing Baselines
Verified completed baselines are stored under `results/homecredit/<model>/<experiment_type>/<run_id>/` and `results/lendingclub_v2/<model>/<experiment_type>/<run_id>/`. Use `reusable_metrics.csv` for comparison rows and `selected_feature_registry.csv` for selector outputs. Original files remain in place. Reusable aggregate evidence is under `results/cross_dataset_v2/`, `results/analysis_plots/`, and `reports/cross_dataset_v2_analysis.md`.

## Invalid CLIP Evidence
Old-policy CLIP checkpoints and derived outputs were removed from active evidence. Their former registry rows, hashes, and reasons are preserved at `results/finalized_research/audit/history/invalid_old_policy_registry_tombstones.json`. Frozen text embeddings, approved splits, source manifests, and target-free DEV descriptors remain reusable where their independence was verified.

## Manual Review
Rows marked `unknown_requires_review` must not enter comparisons until reviewed. The repaired two-epoch `NOT_FOR_SCIENTIFIC_USE` smoke artifacts were removed from the active registry and retained as compact cleanup tombstones.

## Loading and Merging
Load `reusable_metrics.csv` and preserve `result_origin` as `reused_existing` or `newly_executed`. Join `run_index.csv` by `run_id` for configuration, data, split, prediction, and selected-feature paths. Historical invalid rows exist only in cleanup tombstones and must never enter scientific comparisons.

Before execution, check equivalence using dataset, method, model, seed, split definition, feature budget, configuration hash, data-manifest hash, and pairing-policy version for CLIP-dependent work.

## Adding Corrected CLIP Results
Corrected Home Credit CLIP rows are registered under `identity_equivalence_v2` with source manifests, hashes, and `result_origin = newly_executed`. Future rows must pass the same schema and equivalence validation and must not reuse historical old-policy identities.

## Execution Contract
Execute only genuinely new professor-requested experiments or experiments invalidated by the old CLIP pairing policy. Reuse all verified valid comparison experiments through the registry. Do not rerun a completed valid experiment merely because it is required in a final comparison.

Do not rerun the 32 completed valid baseline experiments. Do not restore or recreate the deleted approximately 182-run experiment matrix. Do not restore intentionally deleted legacy CLIP-v1 files or treat their 24 artifact-dependent tests as a restoration request. Do not start corrected CLIP retraining during registry maintenance.


## Corrected Home Credit CLIP (2026-06-25)

Corrected `identity_equivalence_v2` training, 436-feature projection evidence, and four new Home Credit downstream runs are registered under `results/corrected_homecredit_clip/`. The 93 remaining modeling features lack complete frozen semantic views and were not embedded. The all-feature comparator is not present in the verified registry and was not rerun.

## Prediction Row-ID Limitation (Corrected Home Credit CLIP)

Corrected Home Credit OOT prediction artifacts created before this provenance repair do not contain stable row IDs. Exact historical one-to-one row-ID mapping was not recoverable from saved prediction indices, split manifests, source-row-order manifests, or prediction-generation manifests without rerunning model execution, so no sidecar mapping was created. The saved split manifests show disjoint temporal DEV/OOT windows, but historical row-level overlap cannot be directly audited from the prediction files alone.

All future prediction exports, including corrected LendingClub training and reverse-transfer predictions, must include: stable row ID, dataset, split, target, prediction, run ID, and data-manifest hash.


## Corrected LendingClub v2 to Home Credit Reverse Transfer

New reverse-transfer rows are source-trained on LendingClub v2 and applied frozen to Home Credit under `identity_equivalence_v2`. DEV and OOT predictions include stable row IDs, and DEV metrics are computed from persisted fold-exclusive OOF predictions. Existing baselines remain reused.
