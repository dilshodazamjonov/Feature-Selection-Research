# Cross-dataset voting inference and evidence package v1

Roadmap stage: Prompt 6 — voting inference and evidence package
Analysis id: `cross_dataset_voting_inference_v1`
Status: **PASS** — executed 2026-07-28T06:17:03Z to 06:39:26Z (1,342.9 s,
peak RSS 975 MB), all 14 phases PASS, marker
`PROMPT_06_VOTING_INFERENCE_EVIDENCE_PACKAGE_PASS`

This stage turns the 16 completed `cross_dataset_rank_voting_v1` runs into a
reproducible statistical evidence package. It fits no model, runs no selector,
regenerates no ranking, and writes into no completed run directory or the frozen
legacy bundle.

## Entry point

```powershell
Set-Location "D:\python projects\Research"
.\.venv\Scripts\python.exe scripts\build_voting_inference_evidence.py
```

Success marker: `PROMPT_06_VOTING_INFERENCE_EVIDENCE_PACKAGE_PASS`.
Machine-readable status: `<package root>/status.json`.
Exit codes: `0` PASS, `1` BLOCKED, `3` NEEDS USER ACTION, `4` a completed
package already exists at this version.

Outputs are confined to two roots:

- package: `results/final_experiments/cross_dataset_voting_inference_v1`
- audit: `cleanup/audits/prompt_06_voting_inference_evidence_package`

`results/` is git-ignored by repository convention, so the package tables are
local evidence; the audit root's JSON files are the version-controlled
provenance record.

## Authenticated frozen definitions

Every definition below was recovered from a hashed repository artifact before
any metric was computed. The machine-readable copy is
`configs/analysis/cross_dataset_voting_inference_v1.yaml`.

| Element | Authenticated value | Source |
|---|---|---|
| Runs | 16 (2 datasets x 2 models x 4 configurations) | `cross_dataset_voting_configuration_lock.json` |
| DEV folds | 5 per run, 80 total | `cross_dataset_rank_voting_matrix_v1.yaml` |
| Final budgets | LR 20, CatBoost 40 | `cross_dataset_rank_voting_v1.yaml` |
| Primary pool | K=200; K=100/300 are sensitivities | `cross_dataset_rank_voting_v1.yaml` |
| Positive class / direction | 1; higher score means higher default risk | `credit_scoring_extension_v1.yaml` |
| AUC | `sklearn.metrics.roc_auc_score` | `credit_scoring_extension_v1.yaml` |
| Gini | `2*AUC-1`, not separately tested | `credit_scoring_extension_v1.yaml` |
| KS | `paired_inference.ks_statistic`, absolute ECDF gap, smallest score on ties | `credit_scoring_extension_v1.yaml` |
| Lift@10 | `ceil(0.10*n)` highest scores; top bad rate / overall bad rate; stable-id ascending on ties | `credit_scoring_extension_v1.yaml` |
| Score PSI | `pipelines.common.compute_score_psi` (`dev_oof_quantile_psi_v1`), 10 bins on DEV OOF applied unchanged to OOT, epsilon 1e-6 | `credit_scoring_extension_v1.yaml` |
| Feature PSI | frozen numeric `evaluation.drift.calculate_psi` plus a disclosed type-aware extension | `credit_scoring_extension_v1.yaml` |
| Jaccard | all unordered fold pairs; both-empty is 1.0 | `credit_scoring_extension_v1.yaml` |
| Kuncheva | `mean((r*n-k_i*k_j)/(n*min(k_i,k_j)-k_i*k_j))`, zero-denominator pairs skipped | `credit_scoring_extension_v1.yaml` |
| Comparison family | 4 dataset-model families x 3 voting-pool comparisons against the same-cell `rf_corr_mrmr` reference | `cross_dataset_rank_voting_v1.yaml` |
| Paired AUC test | two-sided DeLong on identical OOT rows | `cross_dataset_rank_voting_v1.yaml` |
| Bootstrap | stratified paired, 2,000 attempts, minimum 1,900 valid, seed 20260721, 95% percentile, metrics AUC/KS/Lift@10 | `cross_dataset_rank_voting_v1.yaml` |
| Holm | within each family of three only; no pooling | `credit_scoring_extension_v1.yaml` |
| Effect-strength labels | none frozen, so none are invented | `cross_dataset_rank_voting_v1.yaml` |

## Kuncheva universe decision

The frozen formula names a universe size `n` without naming the object. Exactly
one quantity in this protocol is called a universe, and three independent
sources agree on it:

- `cross_dataset_rank_voting_v1.yaml` -> `candidate_universe.homecredit_size: 529`,
  `lendingclub_v2_size: 675`;
- `cross_dataset_rank_voting_matrix_v1.yaml` -> `candidate_universe_size` per dataset;
- every saved fold ranking -> `candidate_universe_count` equal to the same value
  (verified for all 80 folds by `test_voting_stability`/`test_voting_evidence_provenance`).

Two alternatives were considered and rejected on the record:

- *method-specific candidate pool*: the protocol names `candidate_pool_budgets`
  as a separate concept, and the four reference runs have no voting pool -- their
  pool equals the final budget, so `n*min(k_i,k_j) - k_i*k_j = 0` and Kuncheva
  would be structurally unavailable for the entire reference arm;
- *fold-specific eligible universe*: empirically identical to the frozen
  per-dataset universe in all 80 folds.

The decision therefore comes from artifacts, not from viewing OOT results. It is
recorded in `configs/analysis/cross_dataset_voting_inference_v1.yaml` under
`metric_definitions.kuncheva` and disclosed again in `limitations.md`.

## Accelerated bootstrap equivalence

The frozen `paired_stratified_bootstrap` costs about 9 s per replicate at the
LendingClub OOT size, which is roughly 30 hours for the 12 required
comparisons. `analysis.voting_inference.paired.fast_paired_stratified_bootstrap`
keeps the frozen design unchanged -- same seed, same 2,000 attempts, same
per-replicate draw order, same percentile interval, same three metrics -- and
replaces three Python-level hot loops with vectorised equivalents:

- mid-rank AUC and absolute-ECDF KS are derived from one shared score ordering,
  because both statistics depend only on the tied-score groups;
- Lift@10 orders only the boundary tie group and reproduces the frozen per-draw
  tie-break identity exactly for that group.

Equivalence is not assumed. `assert_bootstrap_equivalence` runs both
implementations on the real aligned rows of every comparison before the full
resampling and fails the gate on any difference; the per-comparison proof lands
in `bootstrap_equivalence_audit.csv` with tolerance `0.0`.
`test_voting_paired_inference` repeats the proof on tie-heavy synthetic inputs.

## Independent recalculation gate

`scripts/independently_verify_voting_metrics.py` re-reads the same saved
prediction artifacts and recomputes AUC, Gini, KS, Lift@10, score PSI, aligned
row count, and target mismatch count from formulas written inside that file. It
imports nothing from `credit_risk_fs.evaluation`, `credit_risk_fs.pipelines`, or
`credit_risk_fs.analysis`. The primary pipeline invokes it as a subprocess and
treats any absolute difference above `1e-9` (or any count difference) as
BLOCKED.

## Phase order

`A` repository/preservation/completion -> `H1` predeclared family recovery ->
`B/C/D` alignment, metric recomputation, score PSI -> `F` stability -> `G`
runtime -> `E` feature PSI -> `C3` independent recalculation -> `H` DeLong,
bootstrap, Holm -> `I` evidence tables and figures -> `J` provenance, claims,
limitations, manifest.

`H1` runs before any p-value so the family cannot be shaped by results. `C3`
runs before the long bootstrap so a recomputation disagreement stops the
pipeline early. Phase `H` caches each comparison, so the command is safe to
rerun and resumes the bootstrap where it stopped.

## Reported artifacts

Package root: `input_inventory.csv`, `prompt_05_completion_authentication.json`,
`predeclared_comparison_family.json`, `alignment_audit.csv/.json`,
`leakage_audit.csv`, `prediction_inventory.csv`,
`dev_oot_population_audit.csv`, `run_level_metrics.csv`, `lift10_audit.csv`,
`score_psi_summary.csv`, `score_psi_bins.csv`, `feature_psi_long.csv`,
`feature_psi_summary.csv`, `feature_psi_definition_audit.csv`,
`fold_selection_inventory.csv`, `stability_pairwise.csv`,
`stability_summary.csv`, `fold_selection_frequency.csv`,
`runtime_resource_summary.csv`, `runtime_stage_breakdown.csv`,
`independent_recalculation_audit.csv`, `paired_delong_results.csv`,
`paired_bootstrap_results.csv`, `bootstrap_replicate_manifest.json`,
`bootstrap_equivalence_audit.csv`, `holm_adjustment_audit.csv`,
`paired_inference_final.csv`, `voting_budget_results.csv`,
`cross_dataset_voting_evidence_table.csv`, `figure_captions.json`,
`claims_and_evidence_seed.csv`, `provenance_audit.csv`,
`artifact_manifest.json`, `limitations.md`, `validation_summary.md`,
`status.json`, and `figures/`.

Audit root: `repository_state.json`, `frozen_input_authentication.json`,
`preservation_audit.json`.

## Executed result

Gates: frozen input authentication PASS (8 inputs); preservation PASS; Prompt 5
completion PASS; 24/24 alignment cells aligned with 0 target mismatches; 192/192
independent recalculation checks within `1e-9` at a maximum absolute difference
of `4.441e-16`; 12/12 DeLong tests; 12/12 bootstrap comparisons at 2,000 valid
replications and 0 failures; accelerated/frozen bootstrap equivalence exact for
every comparison; no comparisons excluded.

Descriptive outcome of the 12 predeclared tests: every voting comparator has a
higher OOT AUC than its same-cell `rf_corr_mrmr` reference, and all 12 survive
Holm within their family of three at alpha 0.05. Primary-cell (K=200) AUC deltas
are `+0.0045` (HC LR), `+0.0088` (HC CatBoost), `+0.0036` (LC LR), `+0.0101`
(LC CatBoost); every primary-cell 95% interval excludes zero. The largest
primary-cell delta in the package is `0.010094`. Statistical significance is not
business materiality, and the protocol predeclares no pooled cross-family test,
so "4 of 4 cells favour voting" is descriptive only.

Countervailing evidence retained in the package: fold-level selection stability
is *lower* for every voting run than for its reference (Jaccard `0.396`-`0.462`
against `0.622`-`0.724`; Kuncheva `0.543`-`0.607` against `0.743`-`0.830`), and
voting runs cost roughly 2-9x the reference wall clock. Score PSI is below `0.06`
for all 16 runs and feature PSI medians are below `0.014`, so neither drift
measure separates the arms.

## Handoff

The next roadmap stage after a PASS is Prompt 7 — lightweight selector
implementation. Prompt 7 must not reopen the frozen protocol, the comparison
family, the bootstrap design, or any completed run directory.
