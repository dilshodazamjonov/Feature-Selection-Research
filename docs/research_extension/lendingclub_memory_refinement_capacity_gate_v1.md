# LendingClub Memory Refinement Capacity Gate v1

## Decision

**PASS.** This is the Prompt 5.1 continuation, not a research execution. The previously completed largest canonical DEV fold was authenticated and reconciled without rerun, and the one missing exact full-DEV LendingClub-v2 `K=300` capacity workload completed through both frozen branches. No research ID, research fold, OOT path, comparison, inference, GPU, API, CLIP, embedding, or SHAP workload was executed.

## Scope and frozen inputs

The gate used the existing memory-safe mechanics in `configs/execution/lendingclub_memory_safe_refinement_v1.yaml` (SHA-256 `4e2a17b93a751bbcb7443d8e82b15781f8a0467a07aa0037a3c298abff4132d7`) without changing it. Frozen hashes were unchanged before and after:

| Input | SHA-256 |
|---|---|
| Scientific protocol | `f4137e98b7c89a63e9a73bc495190858f416c586d237f8ac003c8fdb9e40bde0` |
| Row-alignment contract | `fc1064069f5cc45d76fd34060bb506869683e953c354d3f1f1a13327d99e71a0` |
| Voting protocol | `51030e49716fae0a9c09b52628784a237e9581bb14c1a027e9c8238a575f0b49` |
| Execution policy | `1b77add8bf55096864934f6553aeba174cd22e2b112f53c7a45e5df327934012` |

The neutral refinements are sequential voter fit/release, contiguous `float32` selector encoding, explicit ordered top-`K` reload, branch-local RFE release, and capacity-worker lifecycle/cleanup. Scientific semantics, rows, order, identities, features, ranking, tie rules, estimators, budgets, seeds, precision, and limits remain frozen.

## Equivalence and stale-manifest authentication

The existing equivalence package remains valid: exact voter rankings, aggregate ranking, candidate/selected identities, effective configurations, and probability orientation matched; maximum LR and CatBoost probability differences were both `0.0`. The completed real-data replay remains non-research, comparison-ineligible, and OOT-free.

The largest-fold manifest originally recorded `running`. It was reconciled, not rerun, from its terminal `manifest.json`, `checkpoint.json`, `resource_usage.json`, `capacity_validation.json`, stage timings, and capacity run index. The terminal status is `completed`, worker exit code `0`, stop code `null`, and no warning.

## Measured capacity scenarios

| Scenario | Rows (train/validation) | Universe / K | Branch budgets | Wall seconds | Peak process-tree RSS | Minimum available RAM | Results/temp free minimum | GPU | Stop/status |
|---|---:|---:|---|---:|---:|---:|---:|---|---|
| Largest canonical DEV fold (fold 5) | 458,602 / 71,911 | 675 / 300 | LR-20; CatBoost-40 | 7,749.863 | 6.49 GiB | 11.57 GiB | 57.67 / 79.58 GiB | 0 | none / completed |
| Full DEV capacity fit | 598,649 / 0 | 675 / 300 | LR-20; CatBoost-40 | 9,882.599 | 7.96 GiB | 16.28 GiB | 169.77 / 77.68 GiB | 0 | none / completed |

The full-DEV stage peak was the Boruta stage. The full-DEV checkpoint completed `initialized`, `data_validated`, `selection_completed`, and `model_fit_completed`; terminal `_SUCCESS` was published. The LR selected-feature artifact has 20 rows and the CatBoost selected-feature artifact has 40 rows. No predictions or metrics were produced by the capacity fit.

## Future-shape coverage

`post_refinement_capacity_projection.csv` covers all 16 frozen future research IDs, 80 DEV-fold executions, 16 full-DEV/OOT fits, both datasets, reference and voting methods, LR and CatBoost, `K=100/200/300`, and the four primary/eight sensitivity comparisons. LendingClub full-DEV `K=300` is measured directly. Home Credit is conservatively bounded because its frozen DEV/OOT rows (99,092/120,053) and candidate universe (529) are strict subsets of the measured LendingClub 598,649-row/675-feature path. Reference paths omit Boruta and top-`K` voting work. K100/K200 paths are ordered strict subsets of the measured K300 top-K matrix with identical dtype/order semantics. OOT rows were not opened; frozen row counts, schemas, prediction/publication accounting, and available disk are used only for conservative capacity accounting.

## Preservation and leakage boundary

The canonical `results/run_index.csv` still has exactly four Prompt 4 pilot rows and four run roots; no research or capacity row entered canonical results. The Prompt 4 pilot manifest hash remains `3efc9ddf42829813f2d3444b00ae13a5f7321cbfe4ab40088500b2650d6911df`. The legacy bundle remains 359 files and 110,084,164 bytes with zero added, removed, or changed files. Capacity outputs remain isolated under `cleanup/audits/lendingclub_memory_refinement_capacity_gate/capacity_execution` and are non-research/comparison-ineligible. No OOT path was opened or retained, and no worker, lock, partial, or memory map remains after completion.

## Validation and limitations

Focused Prompt 5 tests report 12 passed. The full suite reports 492 passed, 31 skipped, and 107 warnings; repository validation passed with 84 artifacts and four registered runs; compileall and `git diff --check` passed. The only limitation is that future OOT publication is conservatively projected from frozen schemas and measured capacity; it was intentionally not opened in this gate. Nothing was committed, staged, pushed, amended, reset, or discarded.

The executable research runbook is intentionally separate and is created only after the final validation commands pass: `docs/research_extension/cross_dataset_voting_research_runbook_v1.md`.
